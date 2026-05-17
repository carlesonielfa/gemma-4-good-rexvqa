import json
from pathlib import Path
from typing import Any

from rexvqa_models.types import PathLike


def load_json(path: PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(payload: Any, path: PathLike) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
