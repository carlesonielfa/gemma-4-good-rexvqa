import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from queue import Empty
from threading import Thread
from typing import Any, cast

import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)

try:
    import spaces
except ImportError:
    spaces = None

LOGGER = logging.getLogger(__name__)

BASE_MODEL_ID = "unsloth/gemma-4-E2B-it"
ADAPTER_ID = "carlesonielfa/ReXVQA-SDFT-gemma-4-E2B"
EXAMPLES_DIR = Path("demo_assets/examples")
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert medical professional specializing in medical image analysis "
    "and diagnosis."
)

PROMPT_PRESETS = {
    "Concise findings": "Describe the most important chest X-ray findings in one concise paragraph.",
    "Radiology impression": "Provide a radiology-style impression for this chest X-ray. Be concise.",
    "Clinical diagnosis": "What is the most likely clinical diagnosis suggested by this chest X-ray?",
    "Critical findings": "Are there any urgent or critical findings on this chest X-ray?",
}

BUNDLED_EXAMPLES = [
    {
        "image": "pneumonia_ap.png",
        "diagnosis": "Pneumonia",
        "explanation": "New coarse lung markings in the right infrahilar region are worrisome for an infiltrate; the clinical indication of sepsis and pneumonia supports pneumonia.",
    },
    {
        "image": "pulmonary_vascularity_ap.png",
        "diagnosis": "Engorged and indistinct pulmonary vascularity",
        "explanation": "The pulmonary vascularity appears engorged and indistinct, suggesting increased pulmonary blood flow or congestion.",
    },
    {
        "image": "hemidiaphragm_view1.png",
        "diagnosis": "Elevation of the right hemidiaphragm",
        "explanation": "The right hemidiaphragm is elevated relative to the expected normal position.",
    },
    {
        "image": "inspiration_view1.png",
        "diagnosis": "Suboptimal inspiration",
        "explanation": "Low lung volumes limit the exam quality, making suboptimal inspiration the most notable image-quality finding.",
    },
]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@contextmanager
def _patch_peft_gemma4_clippable_linear() -> Iterator[None]:
    try:
        from peft.tuners.lora.model import LoraModel
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        yield
        return

    lora_model = cast(Any, LoraModel)
    original_create_and_replace = lora_model._create_and_replace

    def patched_create_and_replace(
        self: Any,
        peft_config: Any,
        adapter_name: str,
        target: Any,
        target_name: str,
        parent: Any,
        current_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(target, Gemma4ClippableLinear):
            return original_create_and_replace(
                self,
                peft_config,
                adapter_name,
                target.linear,
                "linear",
                target,
                current_key=current_key,
                **kwargs,
            )
        return original_create_and_replace(
            self,
            peft_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key=current_key,
            **kwargs,
        )

    lora_model._create_and_replace = patched_create_and_replace
    try:
        yield
    finally:
        lora_model._create_and_replace = original_create_and_replace


def _model_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _load_model() -> tuple[Any, Any]:
    from peft import PeftModel

    dtype = _model_dtype()
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["attn_implementation"] = "sdpa"

    LOGGER.info("Loading processor from %s.", ADAPTER_ID)
    processor = AutoProcessor.from_pretrained(ADAPTER_ID, trust_remote_code=True)

    LOGGER.info("Loading base model %s.", BASE_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        **model_kwargs,
    )

    LOGGER.info("Loading adapter %s.", ADAPTER_ID)
    with _patch_peft_gemma4_clippable_linear():
        model = PeftModel.from_pretrained(model, ADAPTER_ID, is_trainable=False)

    model.eval()
    return processor, model


PROCESSOR, MODEL = _load_model()


def _model_device() -> torch.device:
    for tensor in list(MODEL.parameters()) + list(MODEL.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    raise RuntimeError(
        "Model only has meta tensors; it did not load onto a real device."
    )


def _normalize_image(image: Image.Image) -> Image.Image:
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


def _load_demo_image(
    image_value: str | Path | Image.Image | None,
    selected_example_path: str | None = None,
) -> Image.Image:
    selected_example_path = str(selected_example_path or "").strip()
    if selected_example_path:
        selected_path = Path(selected_example_path)
        if selected_path.is_file():
            LOGGER.info("Loading selected example image from %s.", selected_path)
            with Image.open(selected_path) as image:
                return _normalize_image(image)
        LOGGER.warning("Selected example image path is not a file: %s", selected_path)

    if isinstance(image_value, Image.Image):
        LOGGER.info("Loading image from PIL upload.")
        return _normalize_image(image_value)

    image_path = str(image_value or "").strip()
    path = Path(image_path)
    if image_path and path.is_file():
        LOGGER.info("Loading preview image from %s.", path)
        with Image.open(path) as image:
            return _normalize_image(image)

    raise ValueError("Upload or select at least one chest X-ray image.")


def _build_free_text_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Enter a question or prompt before examining the image.")
    return (
        f"Question: {question}\n"
        "Provide a concise free-text answer based on the chest X-ray images. "
        "Do not mention option letters.\n"
        "Answer:"
    )


def _text_item(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _image_item(image: Image.Image) -> dict[str, object]:
    return {"type": "image", "image": image}


def _build_vision_messages(
    prompt: str, images: Sequence[Image.Image]
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [_text_item(DEFAULT_SYSTEM_PROMPT)],
        },
        {
            "role": "user",
            "content": [_image_item(image) for image in images] + [_text_item(prompt)],
        },
    ]


def _generation_kwargs(processor: Any) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    kwargs: dict[str, Any] = {
        "max_new_tokens": 384,
        "do_sample": False,
        "use_cache": True,
    }
    if eos_token_id is not None:
        kwargs["pad_token_id"] = eos_token_id
    return kwargs


def _generate_response(messages: list[dict[str, Any]]) -> Iterator[str]:
    inputs = PROCESSOR.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": "longest"},
    )
    device = _model_device()
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    tokenizer = getattr(PROCESSOR, "tokenizer", PROCESSOR)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        timeout=1.0,
        skip_special_tokens=True,
    )
    generation_kwargs = {
        **inputs,
        **_generation_kwargs(PROCESSOR),
        "streamer": streamer,
    }

    generation_error: list[BaseException] = []

    def generate() -> None:
        try:
            with torch.inference_mode():
                MODEL.generate(**generation_kwargs)
        except BaseException as exc:
            generation_error.append(exc)
            streamer.end()

    thread = Thread(target=generate, daemon=True)
    thread.start()

    response = ""
    while True:
        try:
            token_text = next(streamer)
        except StopIteration:
            break
        except Empty:
            if generation_error:
                break
            if not thread.is_alive():
                break
            continue
        response += token_text
        display_text = response.strip()
        if display_text:
            yield display_text

    thread.join()
    if generation_error:
        raise generation_error[0]


def _bundled_examples() -> list[list[str]]:
    examples: list[list[str]] = []
    for example in BUNDLED_EXAMPLES:
        image = EXAMPLES_DIR / str(example["image"])
        if image.exists():
            examples.append(
                [
                    str(image),
                    str(example["diagnosis"]),
                    str(example["explanation"]),
                ]
            )
    return examples


def _gallery_examples(examples: list[list[str]]) -> list[tuple[str, str]]:
    return [(image_path, diagnosis) for image_path, diagnosis, _ in examples]


EXAMPLES = _bundled_examples()


def select_example(evt: gr.SelectData) -> tuple[str, str, str, str]:
    index = evt.index[0] if isinstance(evt.index, list | tuple) else evt.index
    image_path, correct_answer, explanation = EXAMPLES[int(index)]
    return image_path, image_path, correct_answer, explanation


def apply_preset(preset_name: str) -> str:
    return PROMPT_PRESETS.get(preset_name, "")


def clear_selected_example() -> str:
    return ""


def _gpu_decorator(func: Any) -> Any:
    if spaces is None:
        return func
    return spaces.GPU()(func)


@_gpu_decorator
def examine(
    image_value: str | None,
    selected_example_path: str | None,
    prompt_text: str,
) -> Iterator[str]:
    try:
        image = _load_demo_image(image_value, selected_example_path)
        rendered_prompt = _build_free_text_prompt(prompt_text)
        messages = _build_vision_messages(rendered_prompt, [image])
        response_stream = _generate_response(messages)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Generation failed.")
        raise gr.Error(f"Generation failed: {exc}") from exc

    try:
        yield from response_stream
    except Exception as exc:
        LOGGER.exception("Generation failed.")
        raise gr.Error(f"Generation failed: {exc}") from exc


CSS = """
.gradio-container {
    box-sizing: border-box;
    max-width: min(1180px, calc(100vw - 24px)) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
#demo-title h1 {font-size: 2.1rem; margin-bottom: 0.15rem;}
#demo-title p {opacity: 0.72; margin-top: 0;}
#response-card {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 14px 16px;}
#response-heading {font-weight: 650; margin-bottom: 10px;}
#send-button {min-width: 72px;}
#send-button button {min-height: 132px;}
#example-details textarea {min-height: 112px !important;}
#response {min-height: 180px;}
#prompt-row, #prompt-row > * {
    min-width: 0 !important;
    max-width: 100%;
}
#media-row {
    align-items: stretch;
}
#image-column {
    min-width: 300px !important;
}
#examples-column {
    min-width: 520px !important;
}

@media (max-width: 640px) {
    .gradio-container {
        max-width: 100% !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
    }
    #demo-title h1 {font-size: 1.55rem;}
    #media-row {
        flex-direction: column !important;
    }
    #media-row > * {
        flex: 1 1 100% !important;
        min-width: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    #primary-image {height: 320px !important;}
    #example-gallery {height: 260px !important;}
    #send-button {
        min-width: 56px !important;
        flex-grow: 0 !important;
    }
    #send-button button {min-height: 44px;}
}
"""


with gr.Blocks(title="ReXVQA-SDFT Gemma 4 E2B") as demo:
    selected_example_path = gr.State(value="")

    gr.Markdown(
        """
        # ReXVQA Chest X-ray Demo
        Try a Gemma 4 E2B adapter trained with SDFT on bundled chest X-ray examples or your own image.
        """,
        elem_id="demo-title",
    )

    with gr.Row(elem_id="media-row"):
        with gr.Column(scale=3, min_width=300, elem_id="image-column"):
            primary_image = gr.Image(
                label="Image preview",
                type="filepath",
                image_mode="RGB",
                height=430,
                sources=["upload", "clipboard"],
                elem_id="primary-image",
            )
        with gr.Column(scale=7, min_width=520, elem_id="examples-column"):
            example_gallery = gr.Gallery(
                value=_gallery_examples(EXAMPLES),
                label="Example images",
                columns=4,
                rows=1,
                height=230,
                object_fit="contain",
                allow_preview=False,
                elem_id="example-gallery",
            )
            selected_answer = gr.Textbox(
                label="Correct answer",
                interactive=False,
                lines=1,
            )
            selected_explanation = gr.Textbox(
                label="Correct answer explanation",
                interactive=False,
                lines=4,
                elem_id="example-details",
            )

    preset = gr.Dropdown(
        label="Prompt preset",
        choices=list(PROMPT_PRESETS),
        value="Concise findings",
    )

    with gr.Row(equal_height=True, elem_id="prompt-row"):
        prompt_input = gr.Textbox(
            label="Prompt",
            value=PROMPT_PRESETS["Concise findings"],
            lines=3,
            scale=12,
            min_width=0,
            elem_id="prompt-input",
        )
        send_button = gr.Button(
            "▶",
            variant="primary",
            scale=1,
            min_width=56,
            elem_id="send-button",
        )

    with gr.Group(elem_id="response-card"):
        gr.Markdown("Model response", elem_id="response-heading")
        output = gr.Markdown(elem_id="response")

    example_gallery.select(
        select_example,
        inputs=None,
        outputs=[
            primary_image,
            selected_example_path,
            selected_answer,
            selected_explanation,
        ],
    )
    primary_image.upload(clear_selected_example, outputs=selected_example_path)
    preset.change(apply_preset, preset, prompt_input)
    send_button.click(
        examine,
        [primary_image, selected_example_path, prompt_input],
        output,
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    _configure_logging()
    if not os.getenv("GRADIO_HOT_RELOAD"):
        demo.launch(show_error=True, css=CSS)
