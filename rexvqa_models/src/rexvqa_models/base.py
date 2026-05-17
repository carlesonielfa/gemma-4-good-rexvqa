import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from tqdm import tqdm

from rexvqa_models.evaluator import (
    AnswerExtractor,
    FreeTextResultEvaluator,
    MCQResultEvaluator,
    extract_correct_option_text,
)
from rexvqa_models.image_utils import (
    normalize_image,
    score_view_position,
    select_best_image_paths,
)
from rexvqa_models.prompting import (
    build_free_text_prompt,
    build_mcq_prompt,
    build_vision_messages,
    resolve_mcq_prompt_style,
)
from rexvqa_models.types import (
    CaseBatch,
    CaseData,
    Conversation,
    EvaluationStats,
    ImageInput,
    ModelResponse,
    PathLike,
    PreparedCase,
    ResultPayload,
    ResultsById,
)


class BaseVisionMCQInference(ABC):
    model_label: ClassVar[str] = "Model"
    default_model_name: ClassVar[str | None] = None
    default_batch_size: ClassVar[int] = 32
    default_max_new_tokens: ClassVar[int] = 128
    default_checkpoint_interval: ClassVar[int] = 64
    max_images: ClassVar[int] = 2

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        checkpoint_interval: int | None = None,
        enable_thinking: bool = False,
        prompt_style: str = "standard",
        task: str = "mcq",
    ) -> None:
        resolved_model_name = model_name or self.default_model_name
        if resolved_model_name is None:
            raise ValueError("model_name is required when default_model_name is unset.")
        self.model_name: str = resolved_model_name
        self.batch_size = max(1, int(batch_size or self.default_batch_size))
        self.max_new_tokens = int(max_new_tokens or self.default_max_new_tokens)
        self.checkpoint_interval = max(
            1,
            int(checkpoint_interval or self.default_checkpoint_interval),
        )
        self.enable_thinking = bool(enable_thinking)
        self.task = self._normalize_task(task)
        self.prompt_style = resolve_mcq_prompt_style(prompt_style).name
        self.answer_extractor = self.build_answer_extractor()

    def _normalize_task(self, task: str | None = None) -> str:
        normalized = str(task or "mcq").strip().lower().replace("-", "_")
        if normalized not in {"mcq", "free_text"}:
            raise ValueError("Unsupported inference task. Use 'mcq' or 'free_text'.")
        return normalized

    def _build_question_id(self, case_data: CaseData) -> str:
        return case_data.get("study_id", "") + "_" + case_data.get("task_name", "")

    def _build_result_payload(self, case_data: CaseData) -> ResultPayload:
        payload = {
            "question_id": self._build_question_id(case_data),
            "question": case_data.get("question", ""),
            "options": case_data.get("options", []),
            "correct_answer": case_data.get("correct_answer", ""),
            "category": case_data.get("category", ""),
            "subcategory": case_data.get("subcategory", ""),
        }
        correct_answer_text = extract_correct_option_text(payload)
        if correct_answer_text:
            payload["correct_answer_text"] = correct_answer_text
        correct_answer_explanation = case_data.get("correct_answer_explanation")
        if correct_answer_explanation:
            payload["correct_answer_explanation"] = correct_answer_explanation
        return payload

    def _save_results(self, results: ResultsById, output_file: PathLike) -> None:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def _normalize_image(self, image: Image.Image) -> Image.Image:
        return normalize_image(image)

    def _score_view_position(self, view_position: Any) -> int:
        return score_view_position(view_position)

    def _select_best_image_paths(self, case_data: CaseData) -> list[str]:
        return select_best_image_paths(case_data)

    def load_images(
        self, image_paths: list[str], base_path: PathLike = ""
    ) -> list[ImageInput]:
        images: list[ImageInput] = []
        for img_path in image_paths:
            full_path = Path(base_path) / str(img_path).lstrip("/")
            with Image.open(str(full_path)) as raw_image:
                images.append(normalize_image(raw_image))
        return images[: self.max_images]

    def _build_case_prompt(self, case_data: CaseData) -> str:
        if self.task == "free_text":
            return build_free_text_prompt(case_data)
        return build_mcq_prompt(case_data, prompt_style=self.prompt_style)

    def _system_prompt_text(self) -> str | None:
        return resolve_mcq_prompt_style(self.prompt_style).system_prompt

    def _prepare_case(
        self, case_data: CaseData, base_path: PathLike = ""
    ) -> PreparedCase:
        selected_image_paths = self._select_best_image_paths(case_data)
        images = self.load_images(selected_image_paths, base_path)
        return {
            "messages": self.build_messages(self._build_case_prompt(case_data), images),
            "num_images_used": len(images),
        }

    def build_messages(self, prompt: str, images: list[ImageInput]) -> Conversation:
        return build_vision_messages(
            prompt=prompt,
            images=images,
            system_prompt=self._system_prompt_text(),
        )

    def build_answer_extractor(self) -> AnswerExtractor:
        return AnswerExtractor()

    def _chat_template_kwargs(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def _run_generate(self, conversations: list[Conversation]) -> list[ModelResponse]:
        raise NotImplementedError

    def _process_case_batch(
        self,
        batch_cases: CaseBatch,
        base_path: PathLike = "",
    ) -> ResultsById:
        results: ResultsById = {}
        prepared_cases: list[tuple[str, CaseData, PreparedCase]] = []

        for case_id, case_data in batch_cases:
            try:
                prepared_case = self._prepare_case(case_data, base_path)
            except Exception as exc:
                results[case_id] = {
                    **self._build_result_payload(case_data),
                    "error": f"Preparation error: {exc}",
                }
            else:
                prepared_cases.append((case_id, case_data, prepared_case))

        if not prepared_cases:
            return results

        try:
            responses = self._run_generate(
                [prepared["messages"] for _, _, prepared in prepared_cases]
            )
        except Exception as exc:
            if len(prepared_cases) > 1:
                for case_id, case_data, _ in prepared_cases:
                    results.update(
                        self._process_case_batch([(case_id, case_data)], base_path)
                    )
                return results

            case_id, case_data, _ = prepared_cases[0]
            return {
                case_id: {
                    **self._build_result_payload(case_data),
                    "error": f"Processing error: {exc}",
                }
            }

        for response, (case_id, case_data, prepared_case) in zip(
            responses,
            prepared_cases,
            strict=True,
        ):
            response_payload = (
                response
                if isinstance(response, dict)
                else {"model_response": str(response)}
            )
            results[case_id] = {
                **self._build_result_payload(case_data),
                **response_payload,
                "num_images_used": prepared_case["num_images_used"],
            }

        return results

    def process_single_case(
        self,
        case_data: CaseData,
        base_path: PathLike = "",
    ) -> ResultPayload:
        return self._process_case_batch([("__single_case__", case_data)], base_path)[
            "__single_case__"
        ]

    def process_batch(
        self,
        json_data: dict[str, CaseData],
        base_path: PathLike = "",
        output_file: PathLike = "results/inference.json",
    ) -> ResultsById:
        results: ResultsById = {}
        if os.path.exists(output_file):
            with open(output_file) as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {output_file}")

        pending_cases = [
            (case_id, case_data)
            for case_id, case_data in json_data.items()
            if case_id not in results
        ]

        pbar = tqdm(
            total=len(json_data),
            desc=f"Processing cases with {self.model_label} (batch_size={self.batch_size})",
        )
        pbar.update(len(results))

        pending_since_save = 0
        for batch_start in range(0, len(pending_cases), self.batch_size):
            batch_cases = pending_cases[batch_start : batch_start + self.batch_size]
            batch_results = self._process_case_batch(batch_cases, base_path)
            results.update(batch_results)
            pending_since_save += len(batch_results)

            if pending_since_save >= self.checkpoint_interval:
                self._save_results(results, output_file)
                pending_since_save = 0

            pbar.update(len(batch_cases))

        if pending_since_save > 0:
            self._save_results(results, output_file)

        pbar.close()
        print(f"Processing complete. Processed {len(results)} cases.")
        return results

    def evaluate_results(self, results_file: PathLike) -> EvaluationStats:
        if self.task == "free_text":
            return FreeTextResultEvaluator().evaluate_results(results_file)
        return MCQResultEvaluator(self.answer_extractor).evaluate_results(results_file)
