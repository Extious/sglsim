from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelInfo:
    name: str = "mock-model"
    model_path: str = "mock-model"
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    num_key_value_heads: int = 8
    head_dim: int = 128
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    context_len: int | None = None

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        *,
        context_len: int | None = None,
    ) -> ModelInfo:
        return cls(name=model_path, model_path=model_path, context_len=context_len)

    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0
