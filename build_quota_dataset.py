import logging
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from rexvqa_models.io_utils import write_json
from rexvqa_models.quota_dataset import build_quota_subset_file
from rexvqa_models.types import ConfigDict, PathLike

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


def _project_path(path: PathLike | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(str(path))
    if resolved.is_absolute():
        return resolved
    return Path(get_original_cwd()) / resolved


@hydra.main(version_base=None, config_path="conf", config_name="build_quota_dataset")
def main(cfg: DictConfig) -> None:
    _configure_logging()

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(_to_dict(cfg), run_dir / "resolved_config.json")

    dataset_cfg = _to_dict(cfg.quota_dataset)
    input_file = _project_path(dataset_cfg.pop("input_file"))
    output_file = _project_path(dataset_cfg.pop("output_file"))
    summary_file = _project_path(dataset_cfg.pop("summary_file", None))
    target_total = dataset_cfg.pop("target_total")
    category_distribution = dataset_cfg.pop("category_distribution")
    dataset_cfg.pop("name", None)

    if input_file is None or output_file is None:
        raise ValueError("Both input_file and output_file are required.")

    subset = build_quota_subset_file(
        input_file=input_file,
        output_file=output_file,
        summary_file=summary_file,
        target_total=int(target_total),
        category_distribution=category_distribution,
        **dataset_cfg,
    )

    LOGGER.info("Wrote %s cases to %s.", len(subset), output_file)
    if summary_file:
        LOGGER.info("Wrote summary to %s.", summary_file)


if __name__ == "__main__":
    main()
