import json
import re
from abc import ABC
from collections import Counter
from typing import Any, cast

from rexvqa_models.types import EvaluationStats, PathLike, ResultPayload, ResultsById


def _load_results(results_file: PathLike) -> ResultsById:
    with open(results_file) as f:
        results = json.load(f)

    if isinstance(results, list):
        normalized: ResultsById = {}
        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            result = cast(dict[str, Any], result)
            question_id = result.get("question_id") if "question_id" in result else idx
            normalized[str(question_id)] = result
        return normalized
    return cast(ResultsById, results)


def extract_correct_option_text(result: ResultPayload) -> str:
    correct_answer = str(result.get("correct_answer", "")).strip().upper()
    if not correct_answer:
        return ""

    for option in result.get("options", []) or []:
        if isinstance(option, dict):
            label = str(
                option.get("label")
                or option.get("letter")
                or option.get("key")
                or option.get("option")
                or ""
            ).strip().upper()
            text = str(
                option.get("text")
                or option.get("answer")
                or option.get("value")
                or option.get("content")
                or ""
            ).strip()
            if label == correct_answer:
                return text
            continue

        option_text = str(option).strip()
        match = re.match(r"^\s*([ABCD])(?:[.)\]:-]|\s+-|\s+)(.*)$", option_text)
        if match and match.group(1).upper() == correct_answer:
            return match.group(2).strip()

    return ""


def _normalize_text(text: Any) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return " ".join(normalized.split())


def _tokens(text: Any) -> list[str]:
    normalized = _normalize_text(text)
    return normalized.split() if normalized else []


def _token_f1(prediction: Any, reference: Any) -> float:
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l_f1(prediction: Any, reference: Any) -> float:
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    lcs = _lcs_length(prediction_tokens, reference_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


class AnswerExtractor(ABC):
    def _extract_answer_search_space(
        self,
        model_response: Any,
        result: ResultPayload | None = None,
    ) -> str:
        return str(model_response or "")

    def _extract_from_response(self, response_text: str) -> str:
        prioritized_patterns = [
            r"<answer>\s*([ABCD])\s*</answer>",
            r"\\BOXED\s*\{\s*([ABCD])\s*\}",
            r"\bFINAL\s+ANSWER\b[^ABCD]{0,16}([ABCD])\b",
            r"\b(?:BEST\s+)?ANSWER\b[^ABCD]{0,12}([ABCD])\b",
            r"\bOPTION\b[^ABCD]{0,8}([ABCD])\b",
            r"\*\*([ABCD])(?:[.):\s]|$)",
        ]

        for pattern in prioritized_patterns:
            matches = re.findall(pattern, response_text, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()

        bare_matches = re.findall(
            r"\b([ABCD])(?:[.):\s]|$)",
            response_text,
            flags=re.IGNORECASE,
        )
        if bare_matches:
            return bare_matches[-1].upper()

        return ""

    def extract_answer_letter(
        self,
        model_response: Any,
        result: ResultPayload | None = None,
    ) -> str:
        search_space = self._extract_answer_search_space(
            model_response,
            result=result,
        ).strip()
        if not search_space:
            return ""
        return self._extract_from_response(search_space)


class MCQResultEvaluator:
    def __init__(self, answer_extractor: AnswerExtractor | None = None) -> None:
        self.answer_extractor = answer_extractor or AnswerExtractor()

    def _load_results(self, results_file: PathLike) -> ResultsById:
        return _load_results(results_file)

    def evaluate_results(self, results_file: PathLike) -> EvaluationStats:
        results = self._load_results(results_file)
        correct = 0
        total = 0
        category_stats: dict[str, dict[str, float | int]] = {}

        for result in results.values():
            if "error" in result:
                continue

            total += 1
            correct_answer = str(result.get("correct_answer", "")).strip().upper()
            model_answer = self._extract_model_answer(result)

            if model_answer == correct_answer:
                correct += 1

            category = result.get("category", "Unknown")
            category_stats.setdefault(category, {"correct": 0, "total": 0})
            category_stats[category]["total"] += 1
            if model_answer == correct_answer:
                category_stats[category]["correct"] += 1

        for stats in category_stats.values():
            stats["accuracy"] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            )

        return {
            "overall_accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "category_stats": category_stats,
        }

    def _extract_model_answer(self, result: ResultPayload) -> str:
        explicit_answer = str(result.get("final_answer", "")).strip().upper()
        if explicit_answer in {"A", "B", "C", "D"}:
            return explicit_answer

        return self.answer_extractor.extract_answer_letter(
            result.get("model_response", ""),
            result=result,
        )


class FreeTextResultEvaluator:
    def _load_results(self, results_file: PathLike) -> ResultsById:
        return _load_results(results_file)

    def evaluate_results(self, results_file: PathLike) -> EvaluationStats:
        results = self._load_results(results_file)
        total = 0
        exact_sum = 0.0
        token_f1_sum = 0.0
        rouge_l_sum = 0.0
        category_stats: dict[str, dict[str, float | int]] = {}

        for result in results.values():
            if "error" in result:
                continue

            reference = str(
                result.get("correct_answer_text") or extract_correct_option_text(result)
            ).strip()
            prediction = str(result.get("model_response", "")).strip()
            if not reference:
                continue

            total += 1
            exact_match = float(_normalize_text(prediction) == _normalize_text(reference))
            token_f1 = _token_f1(prediction, reference)
            rouge_l = _rouge_l_f1(prediction, reference)

            exact_sum += exact_match
            token_f1_sum += token_f1
            rouge_l_sum += rouge_l

            category = str(result.get("category") or "Unknown")
            category_stats.setdefault(
                category,
                {
                    "exact_match": 0.0,
                    "token_f1": 0.0,
                    "rouge_l": 0.0,
                    "total": 0,
                },
            )
            category_stats[category]["exact_match"] += exact_match
            category_stats[category]["token_f1"] += token_f1
            category_stats[category]["rouge_l"] += rouge_l
            category_stats[category]["total"] += 1

        for stats in category_stats.values():
            stats_total = int(stats["total"])
            if stats_total <= 0:
                continue
            stats["exact_match"] = float(stats["exact_match"]) / stats_total
            stats["token_f1"] = float(stats["token_f1"]) / stats_total
            stats["rouge_l"] = float(stats["rouge_l"]) / stats_total

        token_f1_score = token_f1_sum / total if total > 0 else 0.0
        return {
            "task": "free_text",
            "primary_score": token_f1_score,
            "exact_match": exact_sum / total if total > 0 else 0.0,
            "token_f1": token_f1_score,
            "rouge_l": rouge_l_sum / total if total > 0 else 0.0,
            "total": total,
            "category_stats": category_stats,
        }
