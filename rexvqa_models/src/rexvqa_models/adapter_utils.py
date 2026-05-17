import json
from pathlib import Path
from typing import Any

from rexvqa_models.types import ConfigDict

_VLLM_SUPPORTED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def normalize_adapter_path(adapter_path: str | Path | None) -> str | None:
    if adapter_path in (None, ""):
        return None
    return str(Path(adapter_path).expanduser().resolve())


def is_peft_adapter_path(path: str | Path | None) -> bool:
    if path in (None, ""):
        return False
    adapter_root = Path(path).expanduser()
    return adapter_root.is_dir() and (adapter_root / "adapter_config.json").is_file()


def load_adapter_config(adapter_path: str | Path | None) -> ConfigDict | None:
    normalized_path = normalize_adapter_path(adapter_path)
    if not normalized_path:
        return None

    config_path = Path(normalized_path) / "adapter_config.json"
    if not config_path.is_file():
        return None

    with open(config_path) as handle:
        return json.load(handle)


def load_adapter_base_model_name(adapter_path: str | Path | None) -> str | None:
    adapter_config = load_adapter_config(adapter_path) or {}
    base_model_name = str(adapter_config.get("base_model_name_or_path") or "").strip()
    return base_model_name or None


def resolve_inference_model_refs(
    model_name: Any,
    adapter_path: str | Path | None = None,
) -> tuple[str, str | None]:
    model_name = str(model_name or "").strip()
    model_name_is_adapter = is_peft_adapter_path(model_name)
    resolved_adapter_path = normalize_adapter_path(adapter_path)

    if resolved_adapter_path is None and model_name_is_adapter:
        resolved_adapter_path = normalize_adapter_path(model_name)

    resolved_model_name = model_name
    if resolved_adapter_path is not None:
        adapter_base_model_name = load_adapter_base_model_name(resolved_adapter_path)
        if adapter_base_model_name:
            resolved_model_name = adapter_base_model_name
        elif model_name_is_adapter:
            raise ValueError(
                f"Adapter path '{resolved_adapter_path}' is missing "
                "'base_model_name_or_path' in adapter_config.json. "
                "Pass the base model explicitly with --model_name."
            )

    return resolved_model_name, resolved_adapter_path


def resolve_vllm_max_lora_rank(
    adapter_path: str | Path | None,
    default: int = 16,
) -> int:
    adapter_config = load_adapter_config(adapter_path) or {}
    try:
        adapter_rank = int(adapter_config.get("r") or default)
    except (TypeError, ValueError):
        return default

    for supported_rank in _VLLM_SUPPORTED_LORA_RANKS:
        if adapter_rank <= supported_rank:
            return supported_rank
    return _VLLM_SUPPORTED_LORA_RANKS[-1]
