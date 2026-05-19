from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

_INSTALLED = False


class NativeSimulatedModelConfig:
    """Small ModelConfig-compatible object for the CPU simulator runtime."""

    def __init__(self, server_args: Any):
        context_len = int(getattr(server_args, "context_length", None) or 32768)
        self.model_path = str(getattr(server_args, "model_path", "simulator-model"))
        self.context_len = context_len
        self.dtype = torch.bfloat16
        self.hidden_size = 4096
        self.head_dim = 128
        self.v_head_dim = 128
        self.swa_head_dim = 128
        self.swa_v_head_dim = 128
        self.num_attention_heads = 32
        self.num_key_value_heads = 8
        self.num_hidden_layers = 36
        self.num_attention_layers = 36
        self.num_nextn_predict_layers = None
        self.vocab_size = 151936
        self.attention_chunk_size = None
        self.sliding_window_size = None
        self.attention_arch = None
        self.is_generation = True
        self.is_multimodal = False
        self.is_audio_model = False
        self.is_encoder_decoder = False
        self.is_local_attention_model = False
        self.is_multimodal_chunked_prefill_supported = False
        self.is_piecewise_cuda_graph_disabled_model = False
        self.is_hybrid_swa = False
        self.is_hybrid_swa_compress = False
        self.disable_hybrid_swa_memory = True
        self.use_ngram_embedding = False
        self.hf_eos_token_id = [0]
        self.image_token_id = None
        self.matryoshka_dimensions = None
        self.is_matryoshka = False
        self.swa_attention_layer_ids: list[int] = []
        self.full_attention_layer_ids = list(range(self.num_hidden_layers))

        self.hf_text_config = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
            model_type="qwen3",
            context_len=context_len,
            max_position_embeddings=context_len,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            torch_dtype="bfloat16",
        )
        self.hf_config = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
            model_type="qwen3",
            context_len=context_len,
            max_position_embeddings=context_len,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            torch_dtype="bfloat16",
            encoder_only=False,
            language_only=True,
        )

        from sglsim.srt.configs.model_config import AttentionArch

        self.attention_arch = AttentionArch.MHA

    def get_total_num_attention_heads(self) -> int:
        return self.num_attention_heads

    def get_num_attention_heads(self, tensor_parallel_size: int) -> int:
        return max(1, self.num_attention_heads // max(1, int(tensor_parallel_size)))

    def get_total_num_kv_heads(self) -> int:
        return self.num_key_value_heads

    def get_num_kv_heads(self, tensor_parallel_size: int) -> int:
        return max(1, self.num_key_value_heads // max(1, int(tensor_parallel_size)))

    def get_swa_num_kv_heads(self, tensor_parallel_size: int) -> int:
        return self.get_num_kv_heads(tensor_parallel_size)

    def get_quantization_config_log_str(self) -> None:
        return None


def install(config: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from sglsim.srt.configs import model_config as model_config_mod

    original_from_server_args = model_config_mod.ModelConfig.from_server_args

    def from_server_args(server_args, *args, **kwargs):
        model_path = str(getattr(server_args, "model_path", "") or "")
        if model_path in {"", "dummy", "none", "simulator-model"}:
            return NativeSimulatedModelConfig(server_args)
        try:
            return original_from_server_args(server_args, *args, **kwargs)
        except Exception:
            if getattr(config, "enable_native_runtime", False):
                return NativeSimulatedModelConfig(server_args)
            raise

    model_config_mod.ModelConfig.from_server_args = staticmethod(from_server_args)


def is_installed() -> bool:
    return _INSTALLED
