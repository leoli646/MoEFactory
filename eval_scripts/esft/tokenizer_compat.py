from __future__ import annotations


def patch_qwen2_tokenizer_for_vllm() -> None:
    # In the current llamafactory env, the local Qwen3 snapshot is loaded as a
    # slow Qwen2Tokenizer. vLLM 0.11 expects all_special_tokens_extended to
    # exist when it wraps the tokenizer cache.
    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    except Exception:
        return

    if hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        return

    @property
    def all_special_tokens_extended(self) -> list[str]:
        return list(self.all_special_tokens)

    Qwen2Tokenizer.all_special_tokens_extended = all_special_tokens_extended
