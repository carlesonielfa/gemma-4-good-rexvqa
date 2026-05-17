from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast


@contextmanager
def patch_peft_gemma4_clippable_linear() -> Iterator[None]:
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
