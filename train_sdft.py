import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from rexvqa_models.io_utils import load_json, write_json
from rexvqa_models.peft_patches import patch_peft_gemma4_clippable_linear
from rexvqa_models.quota_dataset import summarize_cases
from rexvqa_models.types import CaseData, ConfigDict, PathLike

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("TRACKIO_PROJECT", "rexvqa-train")

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def _to_dict(node: Any) -> ConfigDict:
    if node is None:
        return {}
    return cast(ConfigDict, OmegaConf.to_container(node, resolve=True))


def _project_path(path: PathLike) -> Path:
    resolved = Path(str(path))
    if resolved.is_absolute():
        return resolved
    return Path(get_original_cwd()) / resolved


def _find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    for candidate in checkpoints_dir.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        try:
            step = int(candidate.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, candidate))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def _resolve_resume_from_checkpoint(
    cfg: DictConfig, run_dir: Path
) -> str | bool | None:
    resume_from_checkpoint = OmegaConf.select(
        cfg,
        "training.resume_from_checkpoint",
        default=None,
    )
    if resume_from_checkpoint in (None, False, ""):
        return None
    if (
        resume_from_checkpoint is True
        or str(resume_from_checkpoint).lower() == "latest"
    ):
        return True

    checkpoint_path = Path(str(resume_from_checkpoint))
    if not checkpoint_path.is_absolute():
        cwd_relative_path = checkpoint_path.expanduser().resolve(strict=False)
        run_relative_path = (
            (run_dir / checkpoint_path).expanduser().resolve(strict=False)
        )
        checkpoint_path = (
            cwd_relative_path if cwd_relative_path.exists() else run_relative_path
        )
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=False)

    if checkpoint_path.name.startswith("checkpoint-"):
        if checkpoint_path.is_dir():
            return str(checkpoint_path)
        raise FileNotFoundError(f"SDFT checkpoint does not exist: {checkpoint_path}")

    checkpoints_dir = checkpoint_path / "checkpoints"
    latest_checkpoint = (
        _find_latest_checkpoint(checkpoints_dir)
        if checkpoints_dir.is_dir()
        else _find_latest_checkpoint(checkpoint_path)
        if checkpoint_path.is_dir()
        else None
    )
    if latest_checkpoint is not None:
        return str(latest_checkpoint)

    raise FileNotFoundError(
        "resume_from_checkpoint must be 'latest', a checkpoint-* directory, "
        f"or a run/checkpoints directory. Got: {checkpoint_path}"
    )


def _resolve_trainer_kwargs(cfg: DictConfig, run_dir: Path) -> ConfigDict:
    trainer_kwargs = _to_dict(cfg.training.trainer)
    eval_cfg = _to_dict(OmegaConf.select(cfg, "eval", default={}))
    warmup_ratio = trainer_kwargs.pop("warmup_ratio", None)
    if trainer_kwargs.get("warmup_steps") is None and warmup_ratio is not None:
        max_steps = int(trainer_kwargs.get("max_steps", 0) or 0)
        if max_steps > 0:
            trainer_kwargs["warmup_steps"] = max(
                1, int(round(max_steps * float(warmup_ratio)))
            )

    dataloader_num_workers = int(trainer_kwargs.get("dataloader_num_workers", 0) or 0)
    if dataloader_num_workers <= 0:
        trainer_kwargs["dataloader_num_workers"] = 0
        trainer_kwargs["dataloader_persistent_workers"] = False
        trainer_kwargs["dataloader_prefetch_factor"] = None
    else:
        trainer_kwargs.setdefault("dataloader_persistent_workers", True)
        trainer_kwargs.setdefault("dataloader_prefetch_factor", 2)

    trainer_kwargs["use_vllm"] = False
    trainer_kwargs["output_dir"] = str(run_dir / "checkpoints")
    trainer_kwargs["report_to"] = list(cfg.logging.report_to)
    if eval_cfg.get("enabled", False):
        trainer_kwargs["do_eval"] = True
        trainer_kwargs["eval_strategy"] = eval_cfg.get("strategy", "steps")
        trainer_kwargs["eval_steps"] = eval_cfg.get("steps")
        trainer_kwargs["eval_on_start"] = bool(eval_cfg.get("on_start", False))
        trainer_kwargs["per_device_eval_batch_size"] = int(
            eval_cfg.get("batch_size", 1) or 1
        )
        trainer_kwargs["num_generations_eval"] = 1
        trainer_kwargs["remove_unused_columns"] = False
    trainer_kwargs.setdefault(
        "chat_template_kwargs",
        {"enable_thinking": bool(cfg.model.get("enable_thinking", False))},
    )
    return trainer_kwargs


def _load_subset(data_cfg: ConfigDict, run_dir: Path) -> dict[str, CaseData]:
    subset_file = _project_path(data_cfg["subset_file"])
    LOGGER.info("Loading SDFT subset from %s.", subset_file)
    subset = cast(dict[str, CaseData], load_json(subset_file))
    write_json(summarize_cases(subset), run_dir / "subset_summary.json")
    return subset


def _load_eval_cases(eval_cfg: ConfigDict, run_dir: Path) -> dict[str, CaseData] | None:
    if not eval_cfg.get("enabled", False):
        return None

    input_json_file = _project_path(eval_cfg["input_json_file"])
    LOGGER.info("Loading VQA eval cases from %s.", input_json_file)
    cases = cast(dict[str, CaseData], load_json(input_json_file))
    write_json(summarize_cases(cases), run_dir / "eval_subset_summary.json")
    return cases


def _validate_lora_targets(model: Any) -> None:
    risky_targets = []
    for name, module in model.named_modules():
        if not getattr(module, "disable_adapters", None):
            continue
        if any(
            marker in name.lower()
            for marker in ("vision", "visual", "image", "mm_projector")
        ):
            risky_targets.append(name)
    if risky_targets:
        LOGGER.warning(
            "Found PEFT adapter modules under vision-looking paths: %s. "
            "SDFT teacher adapter toggling is safest when LoRA targets only LLM layers.",
            risky_targets[:20],
        )


@hydra.main(version_base=None, config_path="conf", config_name="train_sdft")
def main(cfg: DictConfig) -> None:
    _configure_logging()
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(_to_dict(cfg), run_dir / "resolved_config.json")

    from rexvqa_models.sdft_dataset import build_runtime_sdft_dataset
    from rexvqa_models.sdft_vision import VisionSDFTTrainer
    from unsloth import FastVisionModel

    SDFTConfig = getattr(import_module("trl.experimental.sdft"), "SDFTConfig")

    data_cfg = _to_dict(cfg.data)
    dataset_cfg = _to_dict(cfg.dataset)
    model_cfg = _to_dict(cfg.model)
    eval_cfg = _to_dict(OmegaConf.select(cfg, "eval", default={}))
    subset_cases = _load_subset(data_cfg, run_dir)
    eval_cases = _load_eval_cases(eval_cfg, run_dir)

    train_dataset = build_runtime_sdft_dataset(
        cases=subset_cases,
        image_root=_project_path(data_cfg["image_root"]),
        **dataset_cfg,
    )
    eval_dataset = (
        build_runtime_sdft_dataset(
            cases=eval_cases,
            image_root=_project_path(data_cfg["image_root"]),
            **dataset_cfg,
        )
        if eval_cases is not None
        else None
    )

    from_pretrained_kwargs = _to_dict(cfg.model.from_pretrained)
    from_pretrained_kwargs["model_name"] = model_cfg["model_name"]
    model, processor = FastVisionModel.from_pretrained(**from_pretrained_kwargs)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    peft_kwargs = _to_dict(cfg.model.peft)
    peft_kwargs["random_state"] = int(cfg.experiment.seed)
    with patch_peft_gemma4_clippable_linear():
        model = FastVisionModel.get_peft_model(model, **peft_kwargs)
    _validate_lora_targets(model)

    training_args = SDFTConfig(**_resolve_trainer_kwargs(cfg, run_dir))
    if eval_cases is not None:
        training_args.eval_max_new_tokens = int(
            eval_cfg.get("max_new_tokens", training_args.max_completion_length)
        )
    trainer = VisionSDFTTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.configure_completion_logging(
        **_to_dict(OmegaConf.select(cfg, "training.completion_logging", default={}))
    )

    resume_from_checkpoint = _resolve_resume_from_checkpoint(cfg, run_dir)
    if resume_from_checkpoint:
        LOGGER.info("Resuming SDFT training from %s.", resume_from_checkpoint)
    LOGGER.info("Starting SDFT training on %s examples.", len(train_dataset))
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    write_json(
        {
            "run_dir": str(run_dir),
            "adapter_dir": str(adapter_dir),
            "subset_file": str(_project_path(data_cfg["subset_file"])),
            "num_train_examples": int(len(train_dataset)),
            "num_eval_examples": int(len(eval_cases)) if eval_cases is not None else 0,
            "model_name": str(cfg.model.model_name),
            "prompt_style": str(dataset_cfg.get("prompt_style", "")),
            "trainer": "VisionSDFTTrainer",
        },
        run_dir / "summary.json",
    )
    LOGGER.info(
        "SDFT training complete. Summary written to %s.", run_dir / "summary.json"
    )


if __name__ == "__main__":
    main()
