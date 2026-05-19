from __future__ import annotations

from sglsim.srt.simulator.predictor_aiconfigurator import get_perf_model
from sglsim.srt.simulator.types import AcceleratorInfo, ModelInfo, SchedulerConfig


def calc_kv_cache_cell_elems(model_info: ModelInfo, tp_size: int, pp_size: int) -> int:
    num_layers = int(model_info.num_hidden_layers or 1) // max(pp_size, 1)
    if model_info.is_mla():
        return (
            int(model_info.kv_lora_rank or 0) + int(model_info.qk_rope_head_dim or 0)
        ) * num_layers
    num_kv_heads = max(int(model_info.num_key_value_heads or 1) // max(tp_size, 1), 1)
    head_dim = int(model_info.head_dim or 1)
    return num_kv_heads * head_dim * num_layers * 2


def calc_kv_cache_per_layer_elems(model_info: ModelInfo, tp_size: int, pp_size: int) -> int:
    del pp_size
    if model_info.is_mla():
        return int(model_info.kv_lora_rank or 0) + int(model_info.qk_rope_head_dim or 0)
    num_kv_heads = max(int(model_info.num_key_value_heads or 1) // max(tp_size, 1), 1)
    head_dim = int(model_info.head_dim or 1)
    return num_kv_heads * head_dim * 2


def get_kv_cache_bytes(model_info: ModelInfo, sched_config: SchedulerConfig) -> int:
    dtype = sched_config.kv_cache_data_type or sched_config.data_type
    bytes_per_elem = dtype.bytes if dtype is not None else 2
    return (
        calc_kv_cache_cell_elems(model_info, sched_config.tp_size, sched_config.pp_size)
        * bytes_per_elem
    )


def get_kv_cache_bytes_per_layer(model_info: ModelInfo, sched_config: SchedulerConfig) -> int:
    dtype = sched_config.kv_cache_data_type or sched_config.data_type
    bytes_per_elem = dtype.bytes if dtype is not None else 2
    return (
        calc_kv_cache_per_layer_elems(model_info, sched_config.tp_size, sched_config.pp_size)
        * bytes_per_elem
    )


def estimate_kv_cache_pool_capacity(
    model: ModelInfo,
    device: AcceleratorInfo,
    scheduler_config: SchedulerConfig,
) -> int:
    perf_model = get_perf_model(scheduler_config, model)
    weights = 0.0
    for op in perf_model.context_ops:
        weights += float(op.get_weights())
    weights /= max(int(getattr(perf_model.config, "pp_size", 1) or 1), 1)

    mem_fraction = scheduler_config.mem_fraction_static
    if mem_fraction is None:
        mem_fraction = 0.9
    hbm_capacity_gb = device.hbm_capacity_gb or 80
    framework_reserved_mem_gb = 1.4
    rest_memory = (mem_fraction * float(hbm_capacity_gb) - framework_reserved_mem_gb) * (
        1 << 30
    ) - weights
    kv_cache_space_per_token = get_kv_cache_bytes(model, scheduler_config)
    if kv_cache_space_per_token <= 0:
        return 262144
    return max(int(rest_memory / kv_cache_space_per_token), 1)
