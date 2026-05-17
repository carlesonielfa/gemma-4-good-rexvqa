from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from rexvqa_models.adapter_utils import normalize_adapter_path
from rexvqa_models.base import BaseVisionMCQInference
from rexvqa_models.peft_patches import patch_peft_gemma4_clippable_linear
from rexvqa_models.types import ConfigDict, Conversation, ModelResponse


class BaseTransformersVisionMCQInference(BaseVisionMCQInference):
    default_batch_size = 2

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        device: str = "cuda",
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        checkpoint_interval: int | None = None,
        attn_implementation: str | None = "auto",
        enable_thinking: bool = False,
        prompt_style: str = "standard",
        task: str = "mcq",
    ) -> None:
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            checkpoint_interval=checkpoint_interval,
            enable_thinking=enable_thinking,
            prompt_style=prompt_style,
            task=task,
        )
        self.device = device
        self.adapter_path = normalize_adapter_path(adapter_path)
        self.model_dtype = self._model_dtype()
        self.attn_implementation = self._attention_implementation(attn_implementation)
        self._before_model_load()
        self.processor = AutoProcessor.from_pretrained(
            self._processor_source(self.model_name, adapter_path=self.adapter_path),
            **self._processor_kwargs(),
        )
        self.model = self._load_model(self.model_name)
        self.model.eval()

    def _before_model_load(self) -> None:
        return None

    def _model_dtype(self) -> torch.dtype:
        if str(self.device).startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _attention_implementation(self, attn_implementation: str | None) -> str | None:
        if attn_implementation != "auto":
            return attn_implementation
        return "sdpa" if str(self.device).startswith("cuda") else None

    def _processor_source(
        self,
        model_name: str,
        adapter_path: str | None = None,
    ) -> str:
        return adapter_path or model_name

    def _processor_kwargs(self) -> ConfigDict:
        return {}

    def _model_kwargs(self) -> ConfigDict:
        kwargs: ConfigDict = {"dtype": self.model_dtype, "device_map": "auto"}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        return kwargs

    def _load_model(self, model_name: str) -> Any:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **self._model_kwargs(),
        )
        if self.adapter_path:
            from peft import PeftModel

            with patch_peft_gemma4_clippable_linear():
                model = PeftModel.from_pretrained(
                    model,
                    self.adapter_path,
                    is_trainable=False,
                )
        return model

    def _generation_kwargs(self) -> ConfigDict:
        tokenizer = getattr(self.processor, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        if eos_token_id is not None:
            kwargs["pad_token_id"] = eos_token_id
        return kwargs

    def _chat_template_inputs(self, conversations: list[Conversation]) -> ConfigDict:
        return self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
            **self._chat_template_kwargs(),
        )

    def _move_inputs_to_device(self, inputs: ConfigDict) -> ConfigDict:
        model_inputs: ConfigDict = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(self.model.device)
            else:
                model_inputs[key] = value
        return model_inputs

    def _run_generate(self, conversations: list[Conversation]) -> list[ModelResponse]:
        inputs = self._chat_template_inputs(conversations)
        prompt_length = inputs["input_ids"].shape[1]
        inputs = self._move_inputs_to_device(inputs)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **self._generation_kwargs())

        generated = generated[:, prompt_length:]
        return [
            response.strip()
            for response in self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        ]
