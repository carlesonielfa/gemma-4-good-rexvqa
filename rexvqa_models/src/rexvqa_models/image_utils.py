from ast import literal_eval
from pathlib import Path
from typing import Any, cast

from PIL import Image

from rexvqa_models.types import CaseData, ImageInput, PathLike

FRONTAL_VIEW_TAGS = {
    "AP",
    "PA",
    "AP PORTABLE",
    "AP_PORTABLE",
    "PORTABLE AP",
    "ANTERO_POSTERIOR",
    "POSTERO_ANTERIOR",
}
LATERAL_VIEW_MARKERS = ("LAT", "LATERAL", "LL")


def normalize_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = literal_eval(stripped)
            except (SyntaxError, ValueError):
                return [value]
            if isinstance(parsed, list | tuple):
                return [str(item) for item in parsed]
        return [value]
    return [str(value)]


def score_view_position(view_position: Any) -> int:
    if not view_position:
        return 0

    normalized = str(view_position).upper().replace("-", " ").replace("_", " ")
    compact = normalized.replace(" ", "")
    if normalized in FRONTAL_VIEW_TAGS or compact in {
        "AP",
        "PA",
        "APPORTABLE",
        "PORTABLEAP",
        "ANTEROPOSTERIOR",
        "POSTEROANTERIOR",
    }:
        return 3
    if any(marker in normalized for marker in LATERAL_VIEW_MARKERS):
        return 1
    return 2


def select_best_image_paths(case_data: CaseData) -> list[str]:
    image_paths = normalize_list_field(case_data.get("ImagePath"))
    if not image_paths:
        return []

    view_positions = normalize_list_field(case_data.get("ImageViewPosition"))
    if len(view_positions) != len(image_paths):
        return image_paths[:1]

    ranked_paths = sorted(
        enumerate(image_paths),
        key=lambda item: (-score_view_position(view_positions[item[0]]), item[0]),
    )
    return [path for _, path in ranked_paths]


def resolve_image_path(image_root: PathLike, image_path: PathLike) -> Path:
    path = Path(str(image_path))
    if path.is_absolute():
        return path
    return (Path(image_root) / path).resolve(strict=False)


def normalize_image(image: Image.Image) -> ImageInput:
    image.load()
    image = image.copy()

    if image.mode in {"I", "I;16", "I;16B", "I;16L"}:
        min_value, max_value = cast(tuple[int | float, int | float], image.getextrema())
        if max_value > min_value:
            scale = 255.0 / (max_value - min_value)
            image = image.point(lambda x: (x - min_value) * scale, mode="I").convert(
                "L"
            )
        else:
            image = image.convert("L")
        return image.convert("RGB")

    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def load_case_images(
    case_data: CaseData,
    image_root: PathLike,
    max_images: int = 1,
    image_size: tuple[int, int] | None = (512, 512),
) -> list[ImageInput]:
    images: list[ImageInput] = []
    for image_path in select_best_image_paths(case_data)[
        : max(1, int(max_images or 1))
    ]:
        with Image.open(str(resolve_image_path(image_root, image_path))) as raw_image:
            image = normalize_image(raw_image)
            if image_size:
                image = image.resize(image_size)
            images.append(image)
    return images
