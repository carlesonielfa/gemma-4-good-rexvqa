from collections.abc import Sequence
from dataclasses import dataclass

from rexvqa_models.types import CaseData, Conversation, ImageInput

DEFAULT_SYSTEM_PROMPT = "You are an expert medical professional specializing in medical image analysis and diagnosis."


@dataclass(frozen=True)
class MCQPromptStyle:
    name: str
    system_prompt: str | None
    prompt_template: str


_STANDARD_TEMPLATE = (
    "Question: {question}\n"
    "Options:\n"
    "{options}\n"
    "Answer with only the single best option letter (A, B, C, or D).\n"
    "Answer:"
)

_FREE_TEXT_TEMPLATE = (
    "Question: {question}\n"
    "Provide a concise free-text answer based on the chest X-ray images. "
    "Do not mention option letters.\n"
    "Answer:"
)

MCQ_PROMPT_STYLES = {
    "standard": MCQPromptStyle(
        name="standard",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        prompt_template=_STANDARD_TEMPLATE,
    ),
    "chexone_boxed": MCQPromptStyle(
        name="chexone_boxed",
        system_prompt="You are a chest X-ray reasoning assistant. Follow the user's instruction carefully and provide a boxed final answer.",
        prompt_template=(
            "Question: {question}\n"
            "Options:\n"
            "{options}\n"
            "Please reason step by step from the chest X-ray images before answering. "
            r"End with only one boxed final answer in the form \boxed{{A}}, \boxed{{B}}, "
            r"\boxed{{C}}, or \boxed{{D}}."
        ),
    ),
    "answer_only": MCQPromptStyle(
        name="answer_only",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        prompt_template=(
            "Question: {question}\n"
            "Options:\n"
            "{options}\n"
            "Return exactly one uppercase letter: A, B, C, or D. "
            "Do not output any reasoning, tags, punctuation, or any other text."
        ),
    ),
    "sdft_explanation": MCQPromptStyle(
        name="sdft_explanation",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        prompt_template=(
            "Question: {question}\n"
            "Context: Clinical indication: {Indication}\n"
            "Patient Sex: {PatientSex}\n"
            "Patient Age: {PatientAge}\n"
            "Options:\n"
            "{options}\n"
            "Answer in exactly two lines. "
            "Line 1: one concise sentence explanation with the key image evidence. "
            "Line 2: exactly one final answer block: <answer>A|B|C|D</answer>. "
            "Do not use markdown, bullets, or extra text."
        ),
    ),
}


def resolve_mcq_prompt_style(prompt_style: str | None = None) -> MCQPromptStyle:
    normalized = str(prompt_style or "standard").strip().lower()
    try:
        return MCQ_PROMPT_STYLES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(MCQ_PROMPT_STYLES))
        raise ValueError(
            f"Unsupported prompt_style '{prompt_style}'. Supported values: {supported}."
        ) from exc


def _case_fields(case_data: CaseData) -> dict[str, str]:
    fields = {key: str(value or "").strip() for key, value in case_data.items()}
    fields["options"] = "\n".join(str(o) for o in case_data.get("options", []))
    return fields


def build_mcq_prompt(
    case_data: CaseData,
    prompt_style: str = "standard",
) -> str:
    prompt_style_spec = resolve_mcq_prompt_style(prompt_style)
    rendered = prompt_style_spec.prompt_template.format_map(_case_fields(case_data))
    return "\n".join(line for line in rendered.splitlines() if line.strip())


def build_free_text_prompt(case_data: CaseData) -> str:
    rendered = _FREE_TEXT_TEMPLATE.format_map(_case_fields(case_data))
    return "\n".join(line for line in rendered.splitlines() if line.strip())


def _text_item(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _image_item(image: ImageInput | None = None) -> dict[str, object]:
    item: dict[str, object] = {"type": "image"}
    if image is not None:
        item["image"] = image
    return item


def build_vision_messages(
    prompt: str,
    images: Sequence[ImageInput | None],
    system_prompt: str | None,
) -> Conversation:
    return [
        {
            "role": "system",
            "content": [_text_item(system_prompt or DEFAULT_SYSTEM_PROMPT)],
        },
        {
            "role": "user",
            "content": [_image_item(image) for image in images] + [_text_item(prompt)],
        },
    ]


def build_training_messages(
    prompt: str,
    prompt_style: str,
    num_images: int = 1,
) -> Conversation:
    prompt_style_spec = resolve_mcq_prompt_style(prompt_style)
    return build_vision_messages(
        prompt=prompt,
        images=[None] * max(1, int(num_images or 1)),
        system_prompt=prompt_style_spec.system_prompt,
    )
