import asyncio
import uuid
from abc import ABC
from pathlib import Path
from typing import Any, cast

from transformers import AutoProcessor, GenerationConfig

from rexvqa_models.adapter_utils import (
    normalize_adapter_path,
    resolve_vllm_max_lora_rank,
)
from rexvqa_models.base import BaseVisionMCQInference
from rexvqa_models.types import ConfigDict, Conversation, ModelResponse


class BaseVLLMVisionMCQInference(BaseVisionMCQInference, ABC):
    default_batch_size = 32

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        checkpoint_interval: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        enforce_eager: bool = False,
        seed: int = 0,
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
        self.adapter_path = normalize_adapter_path(adapter_path)
        self.seed = int(seed)
        self._loop = asyncio.new_event_loop()
        self._before_engine_init()
        try:
            from vllm import SamplingParams

            try:
                from vllm import AsyncEngineArgs
            except ImportError:
                from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Install the vLLM dependency group before running inference."
            ) from exc

        self._SamplingParams = SamplingParams
        self._AsyncEngineArgs = AsyncEngineArgs
        self._AsyncLLMEngine = AsyncLLMEngine
        self._LoRARequest = LoRARequest

        self.processor = AutoProcessor.from_pretrained(
            self._processor_source(
                model_name=self.model_name, adapter_path=self.adapter_path
            ),
            **self._processor_kwargs(),
        )
        generation_config = self._load_generation_config(
            model_name=self.model_name,
            adapter_path=self.adapter_path,
        )
        self.sampling_params = SamplingParams(
            **self._sampling_kwargs(generation_config)
        )
        self.engine_args = AsyncEngineArgs(
            **self._engine_kwargs(
                model_name=self.model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
            )
        )
        asyncio.set_event_loop(self._loop)
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.model = self.engine
        self.lora_request = None
        if self.adapter_path and self._supports_lora():
            self.lora_request = LoRARequest(
                lora_name=Path(self.adapter_path).name,
                lora_int_id=1,
                lora_path=self.adapter_path,
                base_model_name=self.model_name,
            )

    def _before_engine_init(self) -> None:
        return None

    def _load_generation_config(
        self,
        model_name: str,
        adapter_path: str | None = None,
    ) -> GenerationConfig | None:
        for config_source in (adapter_path, model_name):
            if not config_source:
                continue
            try:
                return GenerationConfig.from_pretrained(config_source)
            except Exception:
                continue
        return None

    def _supports_lora(self) -> bool:
        return True

    def _processor_source(
        self,
        model_name: str,
        adapter_path: str | None = None,
    ) -> str:
        del adapter_path
        return model_name

    def _processor_kwargs(self) -> ConfigDict:
        return {}

    def _sampling_kwargs(
        self, generation_config: GenerationConfig | None
    ) -> ConfigDict:
        sampling_kwargs: ConfigDict = {
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
        }
        if generation_config is not None and generation_config.temperature is not None:
            sampling_kwargs["temperature"] = generation_config.temperature
        if generation_config is not None and generation_config.top_p is not None:
            sampling_kwargs["top_p"] = generation_config.top_p
        if generation_config is not None and generation_config.top_k is not None:
            sampling_kwargs["top_k"] = generation_config.top_k
        return sampling_kwargs

    def _engine_kwargs(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool,
    ) -> ConfigDict:
        engine_kwargs: ConfigDict = dict(
            model=model_name,
            tensor_parallel_size=max(1, tensor_parallel_size),
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": self.max_images},
            max_model_len=max_model_len,
            dtype="auto",
            enforce_eager=enforce_eager,
            kv_cache_dtype="auto",
        )
        if self.adapter_path and self._supports_lora():
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_loras"] = 1
            engine_kwargs["max_lora_rank"] = resolve_vllm_max_lora_rank(
                self.adapter_path
            )
        return engine_kwargs

    def _conversation_to_engine_inputs(self, conversation: Conversation) -> ConfigDict:
        images: list[Any] = []
        rendered_conversation: Conversation = []
        for message in conversation:
            rendered_message = {"role": message["role"]}
            images.extend(message.get("_images", []))
            content = message.get("content", [])
            if isinstance(content, list):
                rendered_content: list[dict[str, Any]] = []
                for item in content:
                    if item.get("type") == "image":
                        if "image" in item:
                            images.append(item["image"])
                        rendered_content.append({"type": "image"})
                    else:
                        rendered_content.append(item)
                rendered_message["content"] = rendered_content
            else:
                rendered_message["content"] = content
            rendered_conversation.append(rendered_message)

        prompt = self.processor.apply_chat_template(
            rendered_conversation,
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs(),
        )

        inputs: ConfigDict = {"prompt": prompt}
        if images:
            inputs["multi_modal_data"] = {
                "image": images[0] if len(images) == 1 else images
            }
        return inputs

    async def _generate_one(self, request_id: str, inputs: ConfigDict) -> ModelResponse:
        final_output = None
        async for request_output in self.engine.generate(
            cast(Any, inputs),
            self.sampling_params,
            request_id=request_id,
            lora_request=self.lora_request,
        ):
            final_output = request_output

        if final_output is None or not final_output.outputs:
            return ""
        return self._format_completion_output(final_output.outputs[0])

    def _format_completion_output(self, completion_output: Any) -> ModelResponse:
        return completion_output.text.strip()

    async def _run_generate_async(
        self,
        conversations: list[Conversation],
    ) -> list[ModelResponse]:
        tasks: list[Any] = []
        for conversation in conversations:
            request_id = uuid.uuid4().hex
            inputs = self._conversation_to_engine_inputs(conversation)
            tasks.append(self._generate_one(request_id, inputs))
        return await asyncio.gather(*tasks)

    def _run_generate(self, conversations: list[Conversation]) -> list[ModelResponse]:
        asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(self._run_generate_async(conversations))

    def shutdown(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        pending_tasks = [
            task for task in asyncio.all_tasks(self._loop) if not task.done()
        ]
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            self._loop.run_until_complete(
                asyncio.gather(*pending_tasks, return_exceptions=True)
            )
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.close()
