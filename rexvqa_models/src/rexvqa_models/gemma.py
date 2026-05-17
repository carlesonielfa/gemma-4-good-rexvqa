from typing import ClassVar

from rexvqa_models.base_transformers import BaseTransformersVisionMCQInference
from rexvqa_models.base_vllm import BaseVLLMVisionMCQInference
from rexvqa_models.types import ConfigDict


class Gemma4Mixin:
    def _chat_template_kwargs(self) -> ConfigDict:
        if getattr(self, "enable_thinking", False):
            return {"enable_thinking": True}
        return {}

    def _processor_kwargs(self) -> ConfigDict:
        return {"trust_remote_code": True}


class Gemma4MCQInference(Gemma4Mixin, BaseTransformersVisionMCQInference):
    model_label: ClassVar[str] = "Gemma4-transformers"
    default_model_name: ClassVar[str] = "google/gemma-4-E2B-it"

    def _model_kwargs(self) -> ConfigDict:
        kwargs = super()._model_kwargs()
        kwargs["trust_remote_code"] = True
        return kwargs


class Gemma4VLLMMCQInference(Gemma4Mixin, BaseVLLMVisionMCQInference):
    model_label: ClassVar[str] = "Gemma4-vLLM"
    default_model_name: ClassVar[str] = "google/gemma-4-E2B-it"

    def _engine_kwargs(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool,
    ) -> ConfigDict:
        engine_kwargs = super()._engine_kwargs(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )
        engine_kwargs["trust_remote_code"] = True
        return engine_kwargs
