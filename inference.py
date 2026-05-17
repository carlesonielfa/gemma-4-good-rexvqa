import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from rexvqa_models import build_inferencer
from rexvqa_models.adapter_utils import resolve_inference_model_refs
from rexvqa_models.io_utils import load_json, write_json
from rexvqa_models.merge_utils import merge_peft_adapter_for_vllm
from rexvqa_models.types import CaseData, ConfigDict, PathLike

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _to_dict(node: Any) -> ConfigDict:
    if node is None:
        return {}
    return cast(ConfigDict, OmegaConf.to_container(node, resolve=True))


def _project_path(path: PathLike) -> Path:
    path = Path(str(path))
    if path.is_absolute():
        return path
    return Path(get_original_cwd()) / path


def _derive_results_subfolder(input_file: PathLike) -> str:
    input_name = Path(input_file).name
    suffix = "_vqa_data.json"
    if input_name.endswith(suffix):
        return input_name[: -len(suffix)]
    return Path(input_file).stem


def _derive_output_file(
    input_file: PathLike,
    model_name: str,
    output_root: PathLike,
    output_file: PathLike | None = None,
) -> Path:
    if output_file:
        return _project_path(output_file)

    model_slug = Path(model_name).name.removesuffix("-it")
    results_subfolder = _derive_results_subfolder(input_file)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        _project_path(output_root)
        / "inference_results"
        / results_subfolder
        / f"{model_slug}-{timestamp}.json"
    )


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


@hydra.main(version_base=None, config_path="conf", config_name="inference")
def main(cfg: DictConfig) -> None:
    _configure_logging()

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(_to_dict(cfg), run_dir / "resolved_config.json")

    data_cfg = _to_dict(cfg.data)
    model_cfg = _to_dict(cfg.model)
    runtime_cfg = _to_dict(cfg.runtime)
    run_cfg = _to_dict(cfg.run)

    input_json_file = _project_path(data_cfg.pop("input_json_file"))
    image_root = _project_path(data_cfg.pop("image_root"))
    output_root = run_cfg.pop("output_root")
    output_file = run_cfg.pop("output_file", None)

    model_name, adapter_path = resolve_inference_model_refs(
        model_name=model_cfg.pop("model_name"),
        adapter_path=model_cfg.pop("adapter_path", None),
    )
    backend = runtime_cfg.pop("backend")
    merge_adapter_for_vllm = _as_bool(model_cfg.pop("merge_adapter_for_vllm", True))
    merged_model_path = model_cfg.pop("merged_model_path", None)
    merged_model_root = model_cfg.pop("merged_model_root", "results/merged_models")

    output_file = _derive_output_file(
        input_file=input_json_file,
        model_name=adapter_path or model_name,
        output_root=output_root,
        output_file=output_file,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if adapter_path:
        LOGGER.info(
            "Loading base model %s with LoRA adapter %s.",
            model_name,
            adapter_path,
        )

    if backend == "vllm" and adapter_path and merge_adapter_for_vllm:
        model_name = merge_peft_adapter_for_vllm(
            model_name=model_name,
            adapter_path=adapter_path,
            output_root=_project_path(merged_model_root),
            output_path=_project_path(merged_model_path) if merged_model_path else None,
        )
        LOGGER.info("Loading merged model %s with vLLM.", model_name)
        adapter_path = None

    inferencer_kwargs = {
        **model_cfg,
        **runtime_cfg,
        **run_cfg,
        "adapter_path": adapter_path,
    }
    inferencer = None
    try:
        inferencer = build_inferencer(
            model_name=model_name,
            backend=backend,
            **inferencer_kwargs,
        )
        LOGGER.info("Initialized %s.", inferencer.model_label)

        LOGGER.info("Loading data from %s.", input_json_file)
        cases = cast(dict[str, CaseData], load_json(input_json_file))
        LOGGER.info("Loaded %s cases.", len(cases))

        inferencer.process_batch(
            cases,
            base_path=str(image_root),
            output_file=str(output_file),
        )
        LOGGER.info("Results saved to %s.", output_file)

        eval_results = inferencer.evaluate_results(str(output_file))
        LOGGER.info("Evaluation: %s", eval_results)

        write_json(
            {
                "run_dir": str(run_dir),
                "results_file": str(output_file),
                "model_name": model_name,
                "adapter_path": adapter_path,
                "backend": backend,
                "evaluation": eval_results,
            },
            run_dir / "summary.json",
        )
    finally:
        if inferencer is not None:
            shutdown = getattr(inferencer, "shutdown", None)
            if callable(shutdown):
                shutdown()


if __name__ == "__main__":
    main()
