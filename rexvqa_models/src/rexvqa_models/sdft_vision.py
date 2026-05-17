from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, cast

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
import torch.nn.functional as F
from accelerate.utils import gather_object, is_peft_model
from peft.peft_model import PeftModel
from transformers import EvalPrediction, ProcessorMixin
from trl.data_utils import apply_chat_template, prepare_multimodal_messages
from trl.experimental.sdft import SDFTTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.base_trainer import _BaseTrainer
from trl.trainer.utils import (
    entropy_from_logits,
    pad,
    selective_log_softmax,
    use_adapter,
)

from rexvqa_models.evaluator import AnswerExtractor


LOGGER = logging.getLogger(__name__)

VISION_FORWARD_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "image_sizes",
    "token_type_ids",
    "mm_token_type_ids",
    "image_position_ids",
}


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content or "").strip()


def _message_images(content: Any) -> list[Any]:
    if not isinstance(content, list):
        return []
    return [
        part["image"]
        for part in content
        if isinstance(part, dict) and part.get("type") == "image" and "image" in part
    ]


def _extract_prompt_images(prompt: Any) -> list[Any] | None:
    if not isinstance(prompt, list):
        return None
    images = []
    for message in prompt:
        images.extend(_message_images(message.get("content")))
    return images or None


def _zero_completion_extension(
    reference: torch.Tensor, completion_ids: torch.Tensor
) -> torch.Tensor:
    return reference.new_zeros(
        (reference.size(0), completion_ids.size(1)), dtype=reference.dtype
    )


def _truncate_text(text: Any, limit: int = 4000) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _prefixed_forward_kwargs(
    prefix: str, forward_kwargs: dict[str, Any]
) -> dict[str, Any]:
    # TRL splits buffered batches by slicing every top-level value on dim 0. This is correct
    # for Gemma with max_images=1; multi-image Qwen-style tensors may need custom splitting.
    return {f"{prefix}_{key}": value for key, value in forward_kwargs.items()}


def _unprefix_forward_kwargs(prefix: str, inputs: dict[str, Any]) -> dict[str, Any]:
    marker = f"{prefix}_"
    return {
        key.removeprefix(marker): value
        for key, value in inputs.items()
        if key.startswith(marker) and value is not None
    }


def _answer_to_id(answer: Any) -> int:
    normalized = str(answer or "").strip().upper()
    if normalized in {"A", "B", "C", "D"}:
        return ord(normalized) - ord("A")
    return -100


class VisionDemonstrationTeacherContextBuilder:
    """SDFT teacher prompt builder that preserves image blocks for VLM training."""

    def __init__(self, trainer: "VisionSDFTTrainer"):
        self.trainer = trainer

    def _stringify_privileged_context(self, privileged_context: Any) -> str:
        if privileged_context is None:
            raise ValueError("`privileged_context` must not be None for SDFT.")
        if isinstance(privileged_context, str):
            return privileged_context
        if isinstance(privileged_context, list):
            chunks = []
            for message in privileged_context:
                if isinstance(message, dict):
                    text = _message_text(message.get("content", ""))
                else:
                    text = str(message)
                if text:
                    chunks.append(text)
            return "\n".join(chunks)
        return str(privileged_context)

    def compose_teacher_prompt(self, prompt: Any, privileged_context: Any) -> Any:
        privileged_text = self._stringify_privileged_context(privileged_context)
        args = cast(Any, self.trainer.args)
        if not isinstance(prompt, list):
            return args.teacher_prompt_template.format(
                prompt=str(prompt),
                privileged_context=privileged_text,
            )

        teacher_prompt = copy.deepcopy(prompt)
        last_message = teacher_prompt[-1]
        if last_message.get("role") != "user":
            raise ValueError(
                "SDFT vision prompts must end with a user message so the teacher-only "
                "context can be injected into that turn."
            )

        content = last_message.get("content", "")
        prompt_text = _message_text(content)
        teacher_text = args.teacher_prompt_template.format(
            prompt=prompt_text,
            privileged_context=privileged_text,
        )
        if isinstance(content, list):
            image_parts = [
                part
                for part in content
                if isinstance(part, dict) and part.get("type") == "image"
            ]
            last_message["content"] = [
                *image_parts,
                {"type": "text", "text": teacher_text},
            ]
        else:
            last_message["content"] = teacher_text
        return teacher_prompt

    def select_generation_prompts(
        self, prompts: list[Any], privileged_contexts: list[Any]
    ) -> list[Any]:
        if not self.trainer.generate_from_teacher:
            return prompts
        return [
            self.compose_teacher_prompt(prompt, privileged_context)
            for prompt, privileged_context in zip(
                prompts, privileged_contexts, strict=True
            )
        ]


class VisionSDFTTrainer(SDFTTrainer):
    """Small TRL SDFT patch that carries VLM image tensors through generation and loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.processing_class, ProcessorMixin):
            raise TypeError("VisionSDFTTrainer requires a multimodal ProcessorMixin.")
        self.teacher_context_builder = VisionDemonstrationTeacherContextBuilder(self)
        self.log_completions = False
        self.num_completions_to_print = 4
        self.num_teacher_completions_to_log = 4
        self._completion_logs = []
        eval_max_new_tokens = getattr(self.args, "eval_max_new_tokens", None)
        self.eval_max_new_tokens = (
            int(eval_max_new_tokens)
            if eval_max_new_tokens is not None
            else int(self.max_completion_length or 128)
        )
        if self._is_vqa_eval_dataset(getattr(self, "eval_dataset", None)):
            self.answer_extractor = AnswerExtractor()
            self.compute_metrics = self._compute_vqa_metrics

    def configure_completion_logging(
        self,
        enabled: bool = False,
        num_completions_to_print: int = 4,
        num_teacher_completions_to_log: int = 4,
    ) -> None:
        self.log_completions = bool(enabled)
        self.num_completions_to_print = max(0, int(num_completions_to_print or 0))
        self.num_teacher_completions_to_log = max(
            0, int(num_teacher_completions_to_log or 0)
        )
        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(
                os.path.join(str(self.args.output_dir), "completions"), exist_ok=True
            )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "privileged_context",
                "image",
                "images",
            ]

    @staticmethod
    def _is_vqa_eval_batch(inputs: Any) -> bool:
        return (
            isinstance(inputs, list)
            and bool(inputs)
            and isinstance(inputs[0], dict)
            and "correct_answer" in inputs[0]
            and "prompt" in inputs[0]
        )

    @staticmethod
    def _is_vqa_eval_dataset(dataset: Any) -> bool:
        try:
            return len(dataset) > 0 and VisionSDFTTrainer._is_vqa_eval_batch(
                [dataset[0]]
            )
        except Exception:
            return False

    def prediction_step(
        self,
        model,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        if not self._is_vqa_eval_batch(inputs):
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        if prediction_loss_only:
            return None, None, None

        examples = cast(list[dict[str, Any]], inputs)
        prompts, _, images = self._split_prompt_privileged_and_images(examples)
        prompts = self._attach_images(prompts, images)
        tokenized = self._tokenize_vision_prompts(prompts)
        generate_inputs = {
            "input_ids": tokenized["prompt_ids"],
            "attention_mask": tokenized["prompt_mask"],
            **tokenized["forward_kwargs"],
        }
        tokenizer = cast(Any, self.processing_class).tokenizer
        generation_kwargs = {
            "max_new_tokens": self.eval_max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=cast(Any, self.args).ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.inference_mode(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs,
                **generation_kwargs,
            )

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        processor = cast(Any, self.processing_class)
        responses = processor.batch_decode(completion_ids, skip_special_tokens=True)
        predictions = torch.tensor(
            [
                _answer_to_id(self.answer_extractor.extract_answer_letter(response))
                for response in responses
            ],
            device=self.accelerator.device,
            dtype=torch.long,
        )
        labels = torch.tensor(
            [_answer_to_id(example.get("correct_answer")) for example in examples],
            device=self.accelerator.device,
            dtype=torch.long,
        )
        return None, predictions, labels

    @staticmethod
    def _compute_vqa_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
        predictions = torch.as_tensor(eval_prediction.predictions)
        labels = torch.as_tensor(eval_prediction.label_ids)
        valid = labels >= 0
        total = int(valid.sum().item())
        correct = int(((predictions == labels) & valid).sum().item())
        return {
            "accuracy": correct / total if total else 0.0,
            "correct": float(correct),
            "total": float(total),
        }

    @staticmethod
    def _split_prompt_privileged_and_images(inputs: list[dict[str, Any]]):
        prompts = [example["prompt"] for example in inputs]
        privileged_contexts = [example.get("privileged_context") for example in inputs]
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [
                [example.get("image")] if example.get("image") is not None else None
                for example in inputs
            ]
        else:
            images = None
        return prompts, privileged_contexts, images

    @staticmethod
    def _attach_images(
        prompts: list[Any], images: list[list[Any] | None] | None
    ) -> list[Any]:
        if images is None:
            return prompts
        return [
            prepare_multimodal_messages(prompt, images=image_list or [])
            for prompt, image_list in zip(prompts, images, strict=True)
        ]

    def _processor_call(self, **kwargs):
        processor = cast(Any, self.processing_class)
        try:
            return processor(**kwargs)
        except TypeError:
            kwargs.pop("padding_side", None)
            return processor(**kwargs)

    def _tokenize_vision_prompts(self, prompts: list[Any]) -> dict[str, Any]:
        processor = cast(Any, self.processing_class)
        prompt_text = [
            apply_chat_template(
                {"prompt": prompt},
                processor,
                **self.chat_template_kwargs,
            )["prompt"]
            for prompt in prompts
        ]
        images = [_extract_prompt_images(prompt) for prompt in prompts]
        has_images = any(image_list for image_list in images)

        processor_kwargs: dict[str, Any] = {
            "text": prompt_text,
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            "max_length": self.max_prompt_length,
            "truncation": True,
            "add_special_tokens": False,
        }
        if has_images:
            processor_kwargs["images"] = images
        prompt_inputs = self._processor_call(**processor_kwargs)
        prompt_inputs = _BaseTrainer._prepare_inputs(self, prompt_inputs)
        forward_kwargs = {
            key: value
            for key, value in prompt_inputs.items()
            if key in VISION_FORWARD_KEYS
        }
        return {
            "prompt_ids": prompt_inputs["input_ids"],
            "prompt_mask": prompt_inputs["attention_mask"],
            "forward_kwargs": forward_kwargs,
            "num_images": [
                len(image_list) if image_list else 0 for image_list in images
            ]
            if has_images
            else None,
        }

    @staticmethod
    def _extend_sequence_forward_kwargs(
        forward_kwargs: dict[str, Any], completion_ids: torch.Tensor
    ) -> dict[str, Any]:
        extended = dict(forward_kwargs)
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in extended:
                extended[key] = torch.cat(
                    [
                        extended[key],
                        _zero_completion_extension(extended[key], completion_ids),
                    ],
                    dim=1,
                )
        return extended

    def _generate_completion_ids(
        self, prompts: list[Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self._tokenize_vision_prompts(prompts)
        generate_inputs = {
            "input_ids": tokenized["prompt_ids"],
            "attention_mask": tokenized["prompt_mask"],
            **tokenized["forward_kwargs"],
        }

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=cast(Any, self.args).ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs,
                generation_config=self.generation_config,
            )

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=completion_ids.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).long()
        completion_ids = [
            c[m].tolist()
            for c, m in zip(completion_ids, completion_mask.bool(), strict=True)
        ]
        completion_ids = [
            torch.tensor(ids, device=self.accelerator.device) for ids in completion_ids
        ]
        completion_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in completion_ids
        ]
        return (
            pad(completion_ids, padding_value=self.pad_token_id, padding_side="right"),
            pad(completion_mask, padding_value=0, padding_side="right"),
        )

    def _build_buffered_batch(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        prompts, privileged_contexts, images = self._split_prompt_privileged_and_images(
            inputs
        )
        prompts = self._attach_images(prompts, images)
        generation_prompts = self.teacher_context_builder.select_generation_prompts(
            prompts, privileged_contexts
        )
        completion_ids, completion_mask = self._generate_completion_ids(
            generation_prompts
        )
        teacher_context_builder = cast(
            VisionDemonstrationTeacherContextBuilder, self.teacher_context_builder
        )
        teacher_prompts = [
            teacher_context_builder.compose_teacher_prompt(prompt, privileged_context)
            for prompt, privileged_context in zip(
                prompts, privileged_contexts, strict=True
            )
        ]
        self._record_completion_logs(inputs, prompts, teacher_prompts, completion_ids)

        student_batch = self._tokenize_vision_prompts(prompts)
        teacher_batch = self._tokenize_vision_prompts(teacher_prompts)
        teacher_input_ids = torch.cat(
            [teacher_batch["prompt_ids"], completion_ids], dim=1
        )
        teacher_attention_mask = torch.cat(
            [teacher_batch["prompt_mask"], completion_mask], dim=1
        )

        prompt_completion_ids = torch.cat(
            [student_batch["prompt_ids"], completion_ids], dim=1
        )
        attention_mask = torch.cat(
            [student_batch["prompt_mask"], completion_mask], dim=1
        )
        logits_to_keep = completion_ids.size(1)

        old_per_token_logps = None
        args = cast(Any, self.args)
        with torch.no_grad():
            generate_every = args.steps_per_generation * self.num_iterations
            if (
                not self.generate_from_teacher
                and args.gradient_accumulation_steps % generate_every != 0
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    num_images=student_batch["num_images"],
                    **self._extend_sequence_forward_kwargs(
                        student_batch["forward_kwargs"], completion_ids
                    ),
                    compute_entropy=False,
                )

        output = {
            "prompt_ids": student_batch["prompt_ids"],
            "prompt_mask": student_batch["prompt_mask"],
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "student_num_images": student_batch["num_images"],
            "teacher_num_images": teacher_batch["num_images"],
            **_prefixed_forward_kwargs(
                "student",
                self._extend_sequence_forward_kwargs(
                    student_batch["forward_kwargs"],
                    completion_ids,
                ),
            ),
            **_prefixed_forward_kwargs(
                "teacher",
                self._extend_sequence_forward_kwargs(
                    teacher_batch["forward_kwargs"],
                    completion_ids,
                ),
            ),
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        return output

    def _record_completion_logs(
        self,
        inputs: list[dict[str, Any]],
        prompts: list[Any],
        teacher_prompts: list[Any],
        completion_ids: torch.Tensor,
    ) -> None:
        if not self.log_completions:
            return

        processor = cast(Any, self.processing_class)
        generated_completions = processor.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if self.generate_from_teacher:
            teacher_completions = generated_completions
            student_completions = [""] * len(generated_completions)
        else:
            student_completions = generated_completions
            teacher_completions = [""] * len(generated_completions)
            num_teacher = min(self.num_teacher_completions_to_log, len(teacher_prompts))
            if num_teacher > 0:
                teacher_completion_ids, _ = self._generate_completion_ids(
                    teacher_prompts[:num_teacher]
                )
                teacher_completions[:num_teacher] = processor.batch_decode(
                    teacher_completion_ids,
                    skip_special_tokens=True,
                )

        rows = []
        for (
            example,
            prompt,
            teacher_prompt,
            student_completion,
            teacher_completion,
        ) in zip(
            inputs,
            prompts,
            teacher_prompts,
            student_completions,
            teacher_completions,
            strict=True,
        ):
            rows.append(
                {
                    "question_id": str(example.get("question_id", "")),
                    "category": str(example.get("category", "")),
                    "subcategory": str(example.get("subcategory", "")),
                    "correct_answer": str(example.get("correct_answer", "")),
                    "prompt": _truncate_text(
                        _message_text(prompt[-1].get("content"))
                        if isinstance(prompt, list)
                        else prompt
                    ),
                    "teacher_prompt": _truncate_text(
                        _message_text(teacher_prompt[-1].get("content"))
                        if isinstance(teacher_prompt, list)
                        else teacher_prompt
                    ),
                    "teacher_completion": _truncate_text(teacher_completion),
                    "student_completion": _truncate_text(student_completion),
                }
            )

        self._completion_logs.extend(gather_object(rows))

    def _flush_completion_logs(self) -> None:
        if (
            not self.log_completions
            or not self.accelerator.is_main_process
            or not self._completion_logs
        ):
            self._completion_logs.clear()
            return

        step = int(getattr(self.state, "global_step", 0))
        output_dir = os.path.join(str(self.args.output_dir), "completions")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"completions_{step:05d}.jsonl")
        with open(output_path, "w") as handle:
            for row in self._completion_logs:
                row["step"] = step
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        if self.num_completions_to_print:
            LOGGER.info("SDFT completion samples at step %s:", step)
            for row in self._completion_logs[: self.num_completions_to_print]:
                LOGGER.info(
                    "question_id=%s correct=%s student=%s teacher=%s",
                    row.get("question_id", ""),
                    row.get("correct_answer", ""),
                    row.get("student_completion", "").replace("\n", "\\n")[:500],
                    row.get("teacher_completion", "").replace("\n", "\\n")[:500],
                )
        LOGGER.info(
            "Wrote %s SDFT completions to %s", len(self._completion_logs), output_path
        )
        self._completion_logs.clear()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        super().log(logs, start_time)
        self._flush_completion_logs()

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        compute_entropy=False,
        batch_size=None,
        num_images=None,
        **forward_kwargs,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        model_inputs.update(
            {key: value for key, value in forward_kwargs.items() if value is not None}
        )
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits = logits / self.temperature
        completion_ids = input_ids[:, -logits_to_keep:]
        selected_logps = selective_log_softmax(logits, completion_ids)
        entropies = entropy_from_logits(logits) if compute_entropy else None
        return selected_logps, entropies

    def _compute_self_distillation_loss(
        self, model, inputs: dict[str, Any]
    ) -> torch.Tensor:
        args = cast(Any, self.args)
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        logits_to_keep = completion_ids.size(1)

        response_mask = completion_mask
        if response_mask.sum() == 0:
            return torch.tensor(0.0, device=completion_ids.device, requires_grad=True)

        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        student_model_inputs = {
            "input_ids": student_input_ids,
            "attention_mask": student_attention_mask,
            "use_cache": False,
            **_unprefix_forward_kwargs("student", inputs),
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            student_model_inputs["logits_to_keep"] = logits_to_keep + 1

        student_logits = model(**student_model_inputs).logits
        student_logits = student_logits[:, :-1, :]
        student_logits = student_logits[:, -logits_to_keep:, :] / self.temperature

        teacher_model_inputs = {
            "input_ids": inputs["teacher_input_ids"],
            "attention_mask": inputs["teacher_attention_mask"],
            "use_cache": False,
            **_unprefix_forward_kwargs("teacher", inputs),
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            teacher_model_inputs["logits_to_keep"] = logits_to_keep + 1

        teacher_model = self._get_teacher_model_for_self_distillation(model)
        with torch.no_grad(), self._get_teacher_context_for_self_distillation(model):
            teacher_logits = teacher_model(**teacher_model_inputs).logits
            teacher_logits = teacher_logits[:, :-1, :]
            teacher_logits = teacher_logits[:, -logits_to_keep:, :] / self.temperature

        use_topk_distillation = args.distillation_topk is not None and (
            args.full_logit_distillation
            or self._allow_topk_without_full_logit_distillation()
        )
        if use_topk_distillation:
            student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            topk_student_logits, topk_indices = torch.topk(
                student_logits, k=args.distillation_topk, dim=-1
            )
            topk_student_log_probs = topk_student_logits - student_logsumexp

            teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
            topk_teacher_logits = torch.gather(
                teacher_logits, dim=-1, index=topk_indices
            )
            topk_teacher_log_probs = topk_teacher_logits - teacher_logsumexp

            if args.distillation_add_tail:
                topk_student_log_probs = self._add_tail(topk_student_log_probs)
                topk_teacher_log_probs = self._add_tail(topk_teacher_log_probs)
            else:
                topk_student_log_probs = self._renorm_topk_log_probs(
                    topk_student_log_probs
                )
                topk_teacher_log_probs = self._renorm_topk_log_probs(
                    topk_teacher_log_probs
                )
            per_token_loss = self._compute_divergence(
                topk_student_log_probs,
                topk_teacher_log_probs,
                args.distillation_alpha,
            )
        elif args.full_logit_distillation:
            per_token_loss = self._compute_divergence(
                F.log_softmax(student_logits, dim=-1),
                F.log_softmax(teacher_logits, dim=-1),
                args.distillation_alpha,
            )
        else:
            if args.distillation_alpha != 1.0:
                raise ValueError(
                    "Token-level SDFT without full logits requires distillation_alpha=1.0."
                )
            idx = completion_ids.unsqueeze(-1)
            student_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            teacher_lse = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
            student_per_token_logps = (
                torch.gather(student_logits, dim=-1, index=idx) - student_lse
            ).squeeze(-1)
            teacher_per_token_logps = (
                torch.gather(teacher_logits, dim=-1, index=idx) - teacher_lse
            ).squeeze(-1)
            per_token_loss = self._compute_token_level_distillation_loss(
                student_per_token_logps,
                teacher_per_token_logps,
            )

        if (
            args.distillation_is_clip is not None
            and inputs.get("old_per_token_logps") is not None
        ):
            with torch.no_grad():
                student_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
                idx = completion_ids.unsqueeze(-1)
                student_per_token_logps = (
                    torch.gather(student_logits, dim=-1, index=idx) - student_lse
                ).squeeze(-1)
            per_token_loss = self._apply_importance_sampling_clipping(
                per_token_loss,
                student_per_token_logps,
                inputs["old_per_token_logps"],
                args.distillation_is_clip,
            )

        loss = self._aggregate_self_distillation_loss(per_token_loss, response_mask)
        mode = "train" if model.training else "eval"
        mean_distill_loss = (
            per_token_loss * response_mask
        ).sum() / response_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(
            mode,
            "distillation_loss",
            self.accelerator.gather(mean_distill_loss).mean().item(),
        )
        return loss

    def _get_teacher_context_for_self_distillation(self, model):
        args = cast(Any, self.args)
        if isinstance(self.model, PeftModel):
            unwrapped = self.accelerator.unwrap_model(self.model)
            if args.sync_ref_model and "teacher" in unwrapped.peft_config:
                return use_adapter(unwrapped, adapter_name="teacher")
            return use_adapter(unwrapped, adapter_name=None)
        if is_peft_model(self.model):
            return use_adapter(
                self.accelerator.unwrap_model(self.model), adapter_name=None
            )
        return super()._get_teacher_context_for_self_distillation(model)
