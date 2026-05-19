from __future__ import annotations

from typing import Any

import torch

_INSTALLED = False


def _to_torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value in {"float16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "float"}:
        return torch.float32
    return torch.float16


def _batch_size(batch: Any) -> int:
    value = getattr(batch, "batch_size", None)
    if value is not None:
        return int(value)
    reqs = getattr(batch, "reqs", None)
    if reqs is not None:
        return len(reqs)
    return 1


def _resolve_max_total_tokens(self) -> int:
    server_args = self.server_args
    if getattr(server_args, "max_total_tokens", None) is not None:
        return int(server_args.max_total_tokens)

    sim_config_path = getattr(server_args, "sim_config_path", None)
    if sim_config_path:
        try:
            from sglsim.srt.simulator.config import SimulatorSettings
            from sglsim.srt.simulator.kv_capacity import estimate_kv_cache_pool_capacity
            from sglsim.srt.simulator.predictor import (
                SimulatorConfigManager,
                resolve_model_info,
                resolve_scheduler_config,
            )

            settings = SimulatorSettings.from_server_args(server_args)
            manager = SimulatorConfigManager(settings)
            manager.set_model_info(resolve_model_info(self.model_config))
            manager.set_scheduler_config(resolve_scheduler_config(server_args, self.model_config))
            model = manager.get_model_info()
            sched_config = manager.get_scheduler_config()
            if model is not None and sched_config is not None:
                return estimate_kv_cache_pool_capacity(
                    model,
                    manager.get_accelerator_info(),
                    sched_config,
                )
        except Exception:
            pass
    return 262144


def _model_num_layers(model_config: Any, is_draft_worker: bool) -> tuple[int, int, int]:
    model_has_mtp_layers = getattr(model_config, "num_nextn_predict_layers", None) is not None
    if is_draft_worker and model_has_mtp_layers:
        num_layers = int(model_config.num_nextn_predict_layers)
    else:
        num_layers = max(
            int(getattr(model_config, "num_hidden_layers", 1) or 1),
            int(getattr(model_config, "num_attention_layers", 1) or 1),
        )
    start_layer = 0
    end_layer = num_layers
    return start_layer, end_layer, end_layer - start_layer


def install(config) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from sglsim.srt.model_executor import model_runner as model_runner_mod

    ModelRunner = model_runner_mod.ModelRunner

    def initialize(self, *args, **kwargs):
        from sglsim.srt.simulator.native.debug import trace

        trace("ModelRunner.initialize:start")
        from sglsim.srt.mem_cache.allocator import (
            PagedTokenToKVPoolAllocator,
            TokenToKVPoolAllocator,
        )
        from sglsim.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

        class MockModel:
            start_layer = 0
            supports_torch_tp = False

            def forward(self, *args, **kwargs):
                return None

        self.model = MockModel()
        self.loader = None
        self.sampler = None
        self.dtype = _to_torch_dtype(getattr(self.model_config, "dtype", torch.float16))
        self.kv_cache_dtype = self.dtype
        self.device = "cpu"
        self.page_size = int(getattr(self, "page_size", 1) or 1)
        self.page_size = max(self.page_size, 1)
        self.max_total_num_tokens = _resolve_max_total_tokens(self)
        if self.page_size > 1:
            self.max_total_num_tokens = (
                self.max_total_num_tokens // self.page_size * self.page_size
            )
        self.max_total_num_tokens = max(self.max_total_num_tokens, self.page_size)

        context_len = int(getattr(self.model_config, "context_len", 32768) or 32768)
        max_num_reqs = min(
            max(
                int(self.max_total_num_tokens / max(context_len, 1) * 512),
                2048,
            ),
            4096,
        )
        if getattr(self.server_args, "max_running_requests", None) is not None:
            max_num_reqs = min(
                int(self.server_args.max_running_requests),
                max_num_reqs,
            )
        max_num_reqs = max(max_num_reqs, 1)

        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=context_len,
            device="cpu",
            enable_memory_saver=False,
        )
        is_draft_worker = bool(getattr(self, "is_draft_worker", False))
        start_layer, end_layer, num_effective_layers = _model_num_layers(
            self.model_config, is_draft_worker
        )
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_effective_layers = num_effective_layers
        self.model.start_layer = start_layer
        self.token_to_kv_pool = MHATokenToKVPool(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=1,
            head_dim=1,
            layer_num=self.num_effective_layers,
            device="cpu",
            enable_memory_saver=False,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            enable_alt_stream=False,
        )
        if self.page_size == 1:
            self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                size=self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                device="cpu",
                kvcache=self.token_to_kv_pool,
                need_sort=False,
            )
        else:
            self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                size=self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                device="cpu",
                kvcache=self.token_to_kv_pool,
                need_sort=False,
            )
        self.attn_backend = None
        self.graph_runner = None
        self.piecewise_cuda_graph_runner = None
        self.graph_mem_usage = 0
        self.weight_load_mem_usage = 10
        dp_divisor = (
            int(getattr(self.server_args, "dp_size", 1) or 1)
            if getattr(self.server_args, "enable_dp_attention", False)
            else 1
        )
        max_running = getattr(self.server_args, "max_running_requests", None)
        if max_running is None:
            max_running = self.max_total_num_tokens // 2
        else:
            max_running = int(max_running) // max(dp_divisor, 1)
        self.max_running_requests = min(max_running, self.req_to_token_pool.size)
        self.use_ngram_embedding = False
        self.hisparse_coordinator = None
        self.eplb_manager = None
        self.expert_backup_client = None
        self.support_pp = False
        trace("ModelRunner.initialize:end")

    def forward(self, batch, *args, **kwargs):
        from sglsim.srt.layers.logits_processor import LogitsProcessorOutput
        from sglsim.srt.model_executor.model_runner import ModelRunnerOutput

        vocab_size = int(getattr(self.model_config, "vocab_size", 32000) or 32000)
        logits = torch.zeros(
            (_batch_size(batch), vocab_size),
            dtype=torch.float32,
            device="cpu",
        )
        if vocab_size > 1:
            logits[:, 1] = 1.0
        return ModelRunnerOutput(
            logits_output=LogitsProcessorOutput(next_token_logits=logits),
            can_run_graph=False,
            expert_distribution_metrics=None,
        )

    def sample(self, logits_output, *args, **kwargs):
        batch_size = int(logits_output.next_token_logits.shape[0])
        return torch.ones((batch_size,), dtype=torch.int64, device="cpu")

    def compute_logprobs_only(self, *args, **kwargs):
        return None

    ModelRunner.initialize = initialize
    ModelRunner.forward = forward
    ModelRunner.sample = sample
    ModelRunner.compute_logprobs_only = compute_logprobs_only


def is_installed() -> bool:
    return _INSTALLED
