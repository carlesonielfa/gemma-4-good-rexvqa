from __future__ import annotations

import re
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rexvqa_models.types import BackendName, ConfigDict, ModelFamily

if TYPE_CHECKING:
    from rexvqa_models.base import BaseVisionMCQInference
    from rexvqa_models.evaluator import (
        AnswerExtractor,
        FreeTextResultEvaluator,
        MCQResultEvaluator,
    )

_LAZY_EXPORTS = {
    "BaseVisionMCQInference": ("rexvqa_models.base", "BaseVisionMCQInference"),
    "CheXOneMCQInference": ("rexvqa_models.chexone", "CheXOneMCQInference"),
    "CheXOneVLLMMCQInference": ("rexvqa_models.chexone", "CheXOneVLLMMCQInference"),
    "AnswerExtractor": ("rexvqa_models.evaluator", "AnswerExtractor"),
    "FreeTextResultEvaluator": ("rexvqa_models.evaluator", "FreeTextResultEvaluator"),
    "Gemma4MCQInference": ("rexvqa_models.gemma", "Gemma4MCQInference"),
    "Gemma4VLLMMCQInference": ("rexvqa_models.gemma", "Gemma4VLLMMCQInference"),
    "MCQResultEvaluator": ("rexvqa_models.evaluator", "MCQResultEvaluator"),
    "MedGemmaMCQInference": ("rexvqa_models.medgemma", "MedGemmaMCQInference"),
    "MedGemmaVLLMMCQInference": ("rexvqa_models.medgemma", "MedGemmaVLLMMCQInference"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, object_name = _LAZY_EXPORTS[name]
    module = __import__(module_name, fromlist=[object_name])
    value = getattr(module, object_name)
    globals()[name] = value
    return value


def _collapse_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(model_name).strip().lower())


def _candidate_model_names(model_name: str) -> set[str]:
    raw_name = str(model_name).strip().lower()
    path_name = Path(raw_name).name
    candidates = {raw_name, path_name}
    candidates.update(part for part in raw_name.split("/") if part)
    return candidates


def infer_model_family(model_name: str) -> ModelFamily:
    collapsed_candidates = {
        _collapse_model_name(candidate)
        for candidate in _candidate_model_names(model_name)
    }

    if any("medgemma" in candidate for candidate in collapsed_candidates):
        return "medgemma"
    if any("chexone" in candidate for candidate in collapsed_candidates):
        return "chexone"
    if any("gemma4" in candidate for candidate in collapsed_candidates):
        return "gemma4"
    if any(candidate.startswith("gemma") for candidate in collapsed_candidates):
        return "gemma4"

    supported = "google/medgemma-4b-it, google/gemma-4-E2B-it, StanfordAIMI/CheXOne"
    raise ValueError(
        f"Unsupported model name '{model_name}'. Supported model families: {supported}."
    )


def _normalize_backend_name(backend: str) -> BackendName:
    aliases = {
        "auto": "vllm",
        "vllm": "vllm",
        "hf": "transformers",
        "huggingface": "transformers",
        "transformer": "transformers",
        "transformers": "transformers",
    }
    normalized = str(backend).strip().lower().replace("_", "-")
    if normalized not in aliases:
        raise ValueError("Unsupported backend. Use 'vllm' or 'transformers'.")
    return cast(BackendName, aliases[normalized])


def _filter_constructor_kwargs(
    type_: type[Any],
    kwargs: ConfigDict,
) -> ConfigDict:
    accepted_params = signature(type_.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in accepted_params}


def build_inferencer(
    model_name: str,
    backend: str = "vllm",
    **kwargs: Any,
) -> BaseVisionMCQInference:
    from rexvqa_models.chexone import CheXOneMCQInference, CheXOneVLLMMCQInference
    from rexvqa_models.gemma import Gemma4MCQInference, Gemma4VLLMMCQInference
    from rexvqa_models.medgemma import MedGemmaMCQInference, MedGemmaVLLMMCQInference

    inferencer_cls = {
        ("medgemma", "vllm"): MedGemmaVLLMMCQInference,
        ("medgemma", "transformers"): MedGemmaMCQInference,
        ("gemma4", "vllm"): Gemma4VLLMMCQInference,
        ("gemma4", "transformers"): Gemma4MCQInference,
        ("chexone", "vllm"): CheXOneVLLMMCQInference,
        ("chexone", "transformers"): CheXOneMCQInference,
    }.get((infer_model_family(model_name), _normalize_backend_name(backend)))
    if inferencer_cls is None:
        raise ValueError("Selected model family does not support this backend.")
    return inferencer_cls(
        model_name=model_name,
        **_filter_constructor_kwargs(inferencer_cls, kwargs),
    )


def build_answer_extractor(model_name: str) -> AnswerExtractor:
    from rexvqa_models.evaluator import AnswerExtractor

    infer_model_family(model_name)
    return AnswerExtractor()


def build_evaluator(model_name: str) -> MCQResultEvaluator:
    from rexvqa_models.evaluator import MCQResultEvaluator

    return MCQResultEvaluator(answer_extractor=build_answer_extractor(model_name))


__all__ = [
    "CheXOneMCQInference",
    "CheXOneVLLMMCQInference",
    "AnswerExtractor",
    "FreeTextResultEvaluator",
    "Gemma4MCQInference",
    "Gemma4VLLMMCQInference",
    "MCQResultEvaluator",
    "MedGemmaMCQInference",
    "MedGemmaVLLMMCQInference",
    "build_answer_extractor",
    "build_evaluator",
    "build_inferencer",
    "infer_model_family",
]
