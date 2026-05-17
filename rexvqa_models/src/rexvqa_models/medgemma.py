from typing import ClassVar

from rexvqa_models.gemma import Gemma4MCQInference, Gemma4VLLMMCQInference


class MedGemmaMCQInference(Gemma4MCQInference):
    model_label: ClassVar[str] = "MedGemma-transformers"
    default_model_name: ClassVar[str] = "google/medgemma-4b-it"


class MedGemmaVLLMMCQInference(Gemma4VLLMMCQInference):
    model_label: ClassVar[str] = "MedGemma-vLLM"
    default_model_name: ClassVar[str] = "google/medgemma-4b-it"
