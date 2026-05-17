import os
from typing import ClassVar

from rexvqa_models.base_transformers import BaseTransformersVisionMCQInference
from rexvqa_models.base_vllm import BaseVLLMVisionMCQInference
from rexvqa_models.types import ConfigDict


class CheXOneMixin:
    default_model_name: ClassVar[str] = "StanfordAIMI/CheXOne"
    default_max_new_tokens: ClassVar[int] = 256
    recommended_max_pixels: ClassVar[int] = 512 * 512

    def _before_engine_init(self) -> None:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    def _processor_kwargs(self) -> ConfigDict:
        return {
            "max_pixels": self.recommended_max_pixels,
            "use_fast": False,
        }


class CheXOneMCQInference(CheXOneMixin, BaseTransformersVisionMCQInference):
    model_label: ClassVar[str] = "CheXOne-transformers"


class CheXOneVLLMMCQInference(CheXOneMixin, BaseVLLMVisionMCQInference):
    model_label: ClassVar[str] = "CheXOne-vLLM"
