import gc
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor

from rexvqa_models.peft_patches import patch_peft_gemma4_clippable_linear
from rexvqa_models.types import PathLike

LOGGER = logging.getLogger(__name__)


def _merge_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _adapter_cache_key(adapter_path: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter_path / filename
        digest.update(filename.encode())
        if path.is_file():
            stat = path.stat()
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
            if filename == "adapter_config.json":
                digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _default_merged_model_path(adapter_path: Path, output_root: Path) -> Path:
    run_name = (
        adapter_path.parent.name if adapter_path.name == "adapter" else adapter_path.name
    )
    return output_root / f"{run_name}-{_adapter_cache_key(adapter_path)}"


def _has_model_config(model_path: Path) -> bool:
    return model_path.is_dir() and (model_path / "config.json").is_file()


def _ensure_vllm_shared_kv_norms(model_path: Path) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return

    with open(model_path / "config.json") as handle:
        config = json.load(handle)
    text_config = config.get("text_config") or {}
    num_hidden_layers = int(text_config.get("num_hidden_layers") or 0)
    num_kv_shared_layers = int(text_config.get("num_kv_shared_layers") or 0)
    first_shared_layer = num_hidden_layers - num_kv_shared_layers
    if num_hidden_layers <= 0 or num_kv_shared_layers <= 0 or first_shared_layer <= 0:
        return

    with open(index_path) as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map") or {}
    q_norm_pattern = re.compile(
        r"^(?P<prefix>.*language_model\.layers\.)(?P<layer>\d+)"
        r"(?P<suffix>\.self_attn\.)q_norm\.weight$"
    )

    missing_keys: dict[str, str] = {}
    for q_norm_key in sorted(weight_map):
        match = q_norm_pattern.match(q_norm_key)
        if not match:
            continue
        layer_idx = int(match.group("layer"))
        if layer_idx < first_shared_layer:
            continue
        k_norm_key = (
            f"{match.group('prefix')}{layer_idx}"
            f"{match.group('suffix')}k_norm.weight"
        )
        if k_norm_key not in weight_map:
            missing_keys[k_norm_key] = q_norm_key

    if not missing_keys:
        return

    tensors: dict[str, torch.Tensor] = {}
    for k_norm_key, q_norm_key in missing_keys.items():
        shard_name = weight_map[q_norm_key]
        with safe_open(model_path / shard_name, framework="pt") as shard:
            tensors[k_norm_key] = torch.ones_like(shard.get_tensor(q_norm_key))

    extra_shard_name = "model-vllm-shared-kv-norms.safetensors"
    save_file(
        tensors,
        model_path / extra_shard_name,
        metadata={"format": "pt"},
    )
    metadata = index.setdefault("metadata", {})
    metadata["total_size"] = int(metadata.get("total_size") or 0)
    for key, tensor in tensors.items():
        weight_map[key] = extra_shard_name
        metadata["total_size"] += tensor.numel() * tensor.element_size()

    with open(index_path, "w") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
    LOGGER.info(
        "Added %s shared-KV k_norm tensors for vLLM compatibility.",
        len(tensors),
    )


def merge_peft_adapter_for_vllm(
    *,
    model_name: str,
    adapter_path: PathLike,
    output_root: PathLike,
    output_path: PathLike | None = None,
) -> str:
    adapter_root = Path(adapter_path).expanduser().resolve()
    merged_model_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else _default_merged_model_path(
            adapter_root,
            Path(output_root).expanduser().resolve(),
        )
    )

    if _has_model_config(merged_model_path):
        _ensure_vllm_shared_kv_norms(merged_model_path)
        LOGGER.info("Using cached merged model at %s.", merged_model_path)
        return str(merged_model_path)

    if merged_model_path.exists():
        raise FileExistsError(
            f"Merged model path exists but does not look complete: {merged_model_path}"
        )

    merged_model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = merged_model_path.with_name(f".{merged_model_path.name}.tmp")
    if tmp_path.exists():
        raise FileExistsError(
            f"Temporary merge path already exists, likely from an interrupted merge: {tmp_path}"
        )

    LOGGER.info(
        "Merging LoRA adapter %s into base model %s for vLLM.",
        adapter_root,
        model_name,
    )
    model_kwargs: dict[str, Any] = {
        "dtype": _merge_dtype(),
        "device_map": "auto",
        "trust_remote_code": True,
    }
    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("peft is required to merge LoRA adapters.") from exc

    with patch_peft_gemma4_clippable_linear():
        model = PeftModel.from_pretrained(model, str(adapter_root), is_trainable=False)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        tmp_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    processor = AutoProcessor.from_pretrained(str(adapter_root), trust_remote_code=True)
    processor.save_pretrained(tmp_path)
    with open(tmp_path / "merge_manifest.json", "w") as handle:
        json.dump(
            {
                "base_model_name_or_path": model_name,
                "adapter_path": str(adapter_root),
            },
            handle,
            indent=2,
        )

    _ensure_vllm_shared_kv_norms(tmp_path)
    tmp_path.rename(merged_model_path)
    del model, merged_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    LOGGER.info("Saved merged vLLM model to %s.", merged_model_path)
    return str(merged_model_path)
