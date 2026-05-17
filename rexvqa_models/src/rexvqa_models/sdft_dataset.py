from typing import Any

from datasets import Dataset

from rexvqa_models.image_utils import load_case_images, normalize_list_field
from rexvqa_models.prompting import (
    _case_fields,
    build_mcq_prompt,
    build_training_messages,
    resolve_mcq_prompt_style,
)
from rexvqa_models.types import CaseData, PathLike

CaseMap = dict[str, CaseData]


def _build_dataset_rows(cases: CaseMap) -> list[dict[str, Any]]:
    return [
        {
            **{key: str(value or "").strip() for key, value in case_data.items()},
            "question_id": str(question_id),
            "options": normalize_list_field(case_data.get("options")),
            "ImagePath": normalize_list_field(case_data.get("ImagePath")),
            "ImageViewPosition": normalize_list_field(
                case_data.get("ImageViewPosition")
            ),
        }
        for question_id, case_data in cases.items()
    ]


def _build_privileged_context(case_data: CaseData, template: str) -> str:
    return template.format_map(_case_fields(case_data))


def _build_runtime_row(
    question_id: str,
    case_data: CaseData,
    image_root: PathLike,
    max_images: int,
    image_size: tuple[int, int],
    prompt_style: str,
    privileged_context_template: str,
) -> dict[str, Any]:
    images = load_case_images(
        case_data=case_data,
        image_root=image_root,
        max_images=max_images,
        image_size=image_size,
    )
    if not images:
        raise ValueError(
            f"No images could be loaded for question_id={question_id} "
            f"study_id={case_data.get('study_id', '')}."
        )

    prompt = build_mcq_prompt(case_data, prompt_style=prompt_style)
    return {
        "prompt": build_training_messages(
            prompt, prompt_style=prompt_style, num_images=len(images)
        ),
        "privileged_context": _build_privileged_context(
            case_data, privileged_context_template
        ),
        "image": images[0] if len(images) == 1 else images,
        "correct_answer": str(case_data.get("correct_answer", "")),
        "question_id": str(question_id),
        "study_id": str(case_data.get("study_id", "")),
        "category": str(case_data.get("category", "")),
        "subcategory": str(case_data.get("subcategory", "")),
    }


def _build_transform(
    image_root: PathLike,
    max_images: int,
    image_size: tuple[int, int],
    prompt_style: str,
    privileged_context_template: str,
):
    def transform(example: dict[str, Any]) -> dict[str, Any]:
        is_batched = bool(example) and isinstance(next(iter(example.values())), list)
        batch = (
            example if is_batched else {key: [value] for key, value in example.items()}
        )

        rows = [
            _build_runtime_row(
                question_id=str(batch["question_id"][index]),
                case_data={key: value[index] for key, value in batch.items()},
                image_root=image_root,
                max_images=max_images,
                image_size=image_size,
                prompt_style=prompt_style,
                privileged_context_template=privileged_context_template,
            )
            for index in range(len(batch["question_id"]))
        ]
        output = {key: [row[key] for row in rows] for key in rows[0]}
        if is_batched:
            return output
        return {key: value[0] for key, value in output.items()}

    return transform


DEFAULT_PRIVILEGED_CONTEXT_TEMPLATE = "{correct_answer_explanation}"


def build_runtime_sdft_dataset(
    cases: CaseMap,
    image_root: PathLike,
    max_images: int = 1,
    image_size: tuple[int, int] = (512, 512),
    prompt_style: str = "sdft_explanation",
    privileged_context_template: str = DEFAULT_PRIVILEGED_CONTEXT_TEMPLATE,
) -> Dataset:
    prompt_style = resolve_mcq_prompt_style(prompt_style).name
    dataset = Dataset.from_list(_build_dataset_rows(cases))
    dataset.set_transform(
        _build_transform(
            image_root=image_root,
            max_images=int(max_images),
            image_size=tuple(image_size),
            prompt_style=prompt_style,
            privileged_context_template=privileged_context_template,
        )
    )
    return dataset
