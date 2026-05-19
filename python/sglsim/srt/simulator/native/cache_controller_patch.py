from __future__ import annotations

import json
import logging
from queue import Empty
from typing import Any

import torch

from sglsim.srt.simulator.state import get_active_state

logger = logging.getLogger(__name__)

_INSTALLED = False
_CONFIG: Any | None = None
_HOST_READ_BANDWIDTH_BYTES = 200e9
_HOST_WRITE_BANDWIDTH_BYTES = 100e9
_STORAGE_READ_BANDWIDTH_BYTES = 8e9
_STORAGE_WRITE_BANDWIDTH_BYTES = 4e9
_KV_CACHE_BYTES: int | None = None
_KV_CACHE_BYTES_PER_LAYER: int | None = None


def estimate_host_transfer_seconds(
    num_tokens: int, bytes_per_token: int, bandwidth_bytes: float
) -> float:
    return _estimate_transfer_seconds(num_tokens, bytes_per_token, bandwidth_bytes)


def estimate_storage_transfer_seconds(
    num_tokens: int, bytes_per_token: int, bandwidth_bytes: float
) -> float:
    return _estimate_transfer_seconds(num_tokens, bytes_per_token, bandwidth_bytes)


def _estimate_transfer_seconds(
    num_tokens: int, bytes_per_token: int, bandwidth_bytes: float
) -> float:
    if num_tokens <= 0 or bytes_per_token <= 0 or bandwidth_bytes <= 0:
        return 0.0
    return float(num_tokens) * float(bytes_per_token) / float(bandwidth_bytes)


def _load_platform_bandwidths(config: Any) -> None:
    global _HOST_READ_BANDWIDTH_BYTES
    global _HOST_WRITE_BANDWIDTH_BYTES
    global _STORAGE_READ_BANDWIDTH_BYTES
    global _STORAGE_WRITE_BANDWIDTH_BYTES
    global _KV_CACHE_BYTES
    global _KV_CACHE_BYTES_PER_LAYER

    sim_config_path = str(getattr(config, "sim_config_path", "") or "")
    if not sim_config_path:
        return
    try:
        raw = json.loads(open(sim_config_path, encoding="utf-8").read())
    except OSError:
        return
    except json.JSONDecodeError:
        logger.warning("Failed to parse simulator config for transfer bandwidths.")
        return

    platform = raw.get("platform", {})
    _HOST_READ_BANDWIDTH_BYTES = float(
        platform.get("memory_read_bandwidth_gb") or _HOST_READ_BANDWIDTH_BYTES / 1e9
    ) * 1e9
    _HOST_WRITE_BANDWIDTH_BYTES = float(
        platform.get("memory_write_bandwidth_gb") or _HOST_WRITE_BANDWIDTH_BYTES / 1e9
    ) * 1e9
    _STORAGE_READ_BANDWIDTH_BYTES = float(
        platform.get("disk_read_bandwidth_gb") or _STORAGE_READ_BANDWIDTH_BYTES / 1e9
    ) * 1e9
    _STORAGE_WRITE_BANDWIDTH_BYTES = float(
        platform.get("disk_write_bandwidth_gb") or _STORAGE_WRITE_BANDWIDTH_BYTES / 1e9
    ) * 1e9

    try:
        from sglsim.srt.simulator.kv_capacity import (
            get_kv_cache_bytes,
            get_kv_cache_bytes_per_layer,
        )
        from sglsim.srt.simulator.types import DataType, ModelInfo, SchedulerConfig

        model_raw = raw.get("model", {})
        sched_raw = raw.get("scheduler", {})
        model = ModelInfo(
            model_path=model_raw.get("model_path"),
            attention_arch=model_raw.get("attention_arch"),
            context_len=model_raw.get("context_len"),
            hidden_size=model_raw.get("hidden_size"),
            head_dim=model_raw.get("head_dim"),
            num_attention_heads=model_raw.get("num_attention_heads"),
            num_hidden_layers=model_raw.get("num_hidden_layers"),
            num_key_value_heads=model_raw.get("num_key_value_heads"),
            v_head_dim=model_raw.get("v_head_dim"),
            vocab_size=model_raw.get("vocab_size"),
            kv_lora_rank=model_raw.get("kv_lora_rank"),
            qk_rope_head_dim=model_raw.get("qk_rope_head_dim"),
            qk_nope_head_dim=model_raw.get("qk_nope_head_dim"),
            torch_dtype=model_raw.get("torch_dtype"),
        )
        sched_config = SchedulerConfig(
            tp_size=int(sched_raw.get("tp_size", 1)),
            ep_size=int(sched_raw.get("ep_size", 1)),
            dp_size=int(sched_raw.get("dp_size", 1)),
            pp_size=int(sched_raw.get("pp_size", 1)),
            backend_name=str(sched_raw.get("backend_name", "sglang")),
            backend_version=sched_raw.get("backend_version"),
        )
        for field_name in ("data_type", "kv_cache_data_type"):
            value = sched_raw.get(field_name)
            if value is not None:
                setattr(sched_config, field_name, DataType(value))
        _KV_CACHE_BYTES = get_kv_cache_bytes(model, sched_config)
        _KV_CACHE_BYTES_PER_LAYER = get_kv_cache_bytes_per_layer(model, sched_config)
    except Exception:
        logger.debug("Failed to derive KV cache byte sizes from simulator config.")


def _len_tensor(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except TypeError:
        return int(getattr(value, "numel", lambda: 0)())


def _record_hicache_load(dur: float) -> None:
    state = get_active_state()
    if state is not None:
        state.inc_hicache_l2_load_dur(dur)


def _record_hicache_backup(dur: float) -> None:
    state = get_active_state()
    if state is not None:
        state.inc_hicache_l2_backup_dur(dur)


def _host_bytes_per_token(host_pool: Any, *, per_layer: bool = False) -> int:
    if per_layer and _KV_CACHE_BYTES_PER_LAYER is not None:
        return _KV_CACHE_BYTES_PER_LAYER
    if not per_layer and _KV_CACHE_BYTES is not None:
        return _KV_CACHE_BYTES
    size_per_token = int(getattr(host_pool, "size_per_token", 0) or 0)
    if not per_layer:
        return max(size_per_token, 1)
    layer_num = int(getattr(host_pool, "layer_num", 1) or 1)
    return max(1, size_per_token // max(1, layer_num))


def _segment_lengths(src_indices: Any, dst_indices: Any) -> list[int]:
    src = _tensor_to_list(src_indices)
    dst = _tensor_to_list(dst_indices)
    if not src or len(src) != len(dst):
        return [max(1, len(src))]
    lengths: list[int] = []
    start = 0
    for i in range(1, len(src)):
        if src[i] != src[i - 1] + 1 or dst[i] != dst[i - 1] + 1:
            lengths.append(i - start)
            start = i
    lengths.append(len(src) - start)
    return lengths


def _est_bandwidth_batch(size_bytes_arr: list[float], cat: str) -> list[float]:
    if cat == "H2D":
        eff = 0.85
        t0 = 6.67e-6
        bw = _HOST_READ_BANDWIDTH_BYTES * eff
    else:
        eff = 0.85
        t0 = 4e-6
        bw = _HOST_WRITE_BANDWIDTH_BYTES * eff
    out: list[float] = []
    for size_bytes in size_bytes_arr:
        x = float(size_bytes)
        out.append(x * bw / (t0 * bw + x))
    return out


def _segmented_host_transfer_seconds(
    host_indices: Any,
    device_indices: Any,
    bytes_per_elem: int,
    *,
    cat: str,
) -> float:
    seg_lens = _segment_lengths(host_indices, device_indices)
    size_bytes_arr = [length * float(bytes_per_elem) for length in seg_lens]
    bandwidth_arr = _est_bandwidth_batch(size_bytes_arr, cat)
    return float(sum(size / bw for size, bw in zip(size_bytes_arr, bandwidth_arr)))


def _tensor_to_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        listed = value.tolist()
        if isinstance(listed, list):
            return [int(x) for x in listed]
        return [int(listed)]
    return [int(x) for x in value]


def simulated_load_to_device_per_layer(
    self: Any,
    device_pool: Any,
    host_indices: Any,
    device_indices: Any,
    layer_id: int,
    io_backend: str,
) -> None:
    del device_pool, layer_id, io_backend
    bytes_per_elem = _host_bytes_per_token(self, per_layer=True)
    dur = _segmented_host_transfer_seconds(
        host_indices, device_indices, bytes_per_elem, cat="H2D"
    )
    _record_hicache_load(dur)


def simulated_backup_from_device_all_layer(
    self: Any, device_pool: Any, host_indices: Any, device_indices: Any, io_backend: str
) -> None:
    del device_pool, io_backend
    bytes_per_elem = _host_bytes_per_token(self, per_layer=False)
    dur = _segmented_host_transfer_seconds(
        host_indices, device_indices, bytes_per_elem, cat="D2H"
    )
    _record_hicache_backup(dur)


def simulated_get_data_page(self: Any, index: int, flat: bool = True) -> torch.Tensor:
    return torch.tensor([[int(index)]], dtype=torch.int64)


def simulated_set_from_flat_data_page(
    self: Any, index: int, data_page: torch.Tensor
) -> None:
    return None


def _prefetch_capacity_tokens(controller: Any, remain_dur: float, bytes_per_token: int) -> int:
    if remain_dur <= 0 or bytes_per_token <= 0:
        return 0
    return int(remain_dur * _STORAGE_READ_BANDWIDTH_BYTES // bytes_per_token)


def handle_backup_operation(self: Any) -> None:
    if not getattr(self, "enable_storage", False):
        return
    while True:
        try:
            operation = self.backup_queue.get(block=False)
        except Empty:
            return
        if operation is None:
            return

        if not getattr(self, "backup_skip", False):
            self._page_backup(operation)
        completed_tokens = int(getattr(operation, "completed_tokens", 0) or 0)
        if completed_tokens <= 0:
            completed_tokens = _len_tensor(getattr(operation, "host_indices", []))
        bytes_per_token = _host_bytes_per_token(getattr(self, "mem_pool_host", None), per_layer=False)
        _record_hicache_backup(
            estimate_storage_transfer_seconds(
                completed_tokens, bytes_per_token, _STORAGE_WRITE_BANDWIDTH_BYTES
            )
        )
        self.ack_backup_queue.put(operation)


def _record_request_prefetch_tokens(operation: Any) -> None:
    request_id = getattr(operation, "request_id", None)
    if not request_id:
        return
    # Scheduler patch owns request_stats; use active runtime only when present.
    try:
        from sglsim.srt.simulator.state import get_active_runtime

        runtime = get_active_runtime()
    except Exception:
        runtime = None
    if runtime is not None and hasattr(runtime, "request_stats"):
        runtime.request_stats[request_id].prefetch_complete_tokens = int(
            getattr(operation, "completed_tokens", 0) or 0
        )


def handle_prefetch_operation(self: Any) -> None:
    if not getattr(self, "enable_storage", False):
        return
    state = get_active_state()
    remain_dur = float(getattr(state, "current_inference_dur", 0.0) if state else 0.0)
    bytes_per_token = _host_bytes_per_token(getattr(self, "mem_pool_host", None), per_layer=False)

    chunked = getattr(self, "chunked_prefetch_operation", None)
    if chunked is not None:
        operation = chunked["operation"]
        storage_hit_count = int(chunked["storage_hit_count"])
        pending = max(0, storage_hit_count - int(operation.completed_tokens))
        completed = min(
            pending, _prefetch_capacity_tokens(self, remain_dur, bytes_per_token)
        )
        operation.completed_tokens += completed
        remain_dur -= estimate_storage_transfer_seconds(
            completed, bytes_per_token, _STORAGE_READ_BANDWIDTH_BYTES
        )
        if operation.completed_tokens >= storage_hit_count:
            operation.completed_tokens = storage_hit_count
            operation.mark_terminate()
            setattr(self, "chunked_prefetch_operation", None)
            self.append_host_mem_release(operation.host_indices[storage_hit_count:])
        _record_request_prefetch_tokens(operation)

    while remain_dur > 0:
        try:
            operation = self.prefetch_queue.get(block=False)
        except Empty:
            return
        if operation is None:
            return

        hash_value, storage_hit_count = self._storage_hit_query(operation)
        threshold = getattr(self, "prefetch_threshold", None)
        if threshold is not None and storage_hit_count < threshold:
            operation.mark_terminate()
            self.append_host_mem_release(operation.host_indices)
            continue

        storage_hit_count = int(storage_hit_count // self.page_size * self.page_size)
        operation.storage_query_count = storage_hit_count
        operation.hash_value = hash_value[: storage_hit_count // self.page_size]
        capacity = _prefetch_capacity_tokens(self, remain_dur, bytes_per_token)
        completed = min(storage_hit_count, capacity)
        operation.completed_tokens = completed
        dur = estimate_storage_transfer_seconds(
            completed, bytes_per_token, _STORAGE_READ_BANDWIDTH_BYTES
        )
        _record_hicache_load(dur)
        remain_dur -= dur
        if completed < storage_hit_count:
            setattr(
                self,
                "chunked_prefetch_operation",
                {"operation": operation, "storage_hit_count": storage_hit_count},
            )
            _record_request_prefetch_tokens(operation)
            return

        operation.mark_terminate()
        self.append_host_mem_release(operation.host_indices[storage_hit_count:])
        _record_request_prefetch_tokens(operation)


def _generic_page_set(self: Any, hash_values: Any, host_indices: Any, extra_info=None) -> bool:
    data = [
        self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
        for i in range(len(hash_values))
    ]
    try:
        return bool(self.storage_backend.batch_set(hash_values, data, extra_info))
    except TypeError:
        return bool(self.storage_backend.batch_set(hash_values, data))


def _native_cpu_host_allocate(
    dims: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: Any,
) -> torch.Tensor:
    return allocator.allocate(dims, dtype=dtype, device=device)


def _patch_host_memory_pool(memory_pool_host_mod: Any) -> None:
    memory_pool_host_mod.ALLOC_MEMORY_FUNCS["cpu"] = _native_cpu_host_allocate
    HostKVCache = memory_pool_host_mod.HostKVCache
    if getattr(HostKVCache, "_sglsim_native_init_patched", False):
        return

    original_init = HostKVCache.__init__

    def native_host_kv_cache_init(
        self: Any,
        device_pool: Any,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ) -> None:
        native_ratio = max(float(host_to_device_ratio or 1.0), 1.0)
        if host_size:
            logger.info(
                "Native simulator ignores hicache_size=%s GB and sizes host KV "
                "metadata from device pool ratio %.3f.",
                host_size,
                native_ratio,
            )
        original_init(
            self,
            device_pool,
            native_ratio,
            0,
            page_size,
            layout,
            False,
            device,
            allocator_type,
        )

    HostKVCache.__init__ = native_host_kv_cache_init
    HostKVCache._sglsim_native_init_patched = True
    HostKVCache._sglsim_native_original_init = original_init


def install(config: Any) -> None:
    global _INSTALLED
    global _CONFIG
    if _INSTALLED:
        return
    _INSTALLED = True
    _CONFIG = config
    _load_platform_bandwidths(config)

    from sglsim.srt.managers.cache_controller import HiCacheController
    from sglsim.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglsim.srt.mem_cache import memory_pool_host as memory_pool_host_mod
    from sglsim.srt.mem_cache.memory_pool_host import (
        MHATokenToKVPoolHost,
        MLATokenToKVPoolHost,
        NSATokenToKVPoolHost,
    )

    def noop_thread(self: Any, *args, **kwargs) -> None:
        return None

    original_check_hicache_events = HiRadixCache.check_hicache_events

    def check_hicache_events(self: Any, *args, **kwargs):
        self.cache_controller.handle_backup_operation()
        self.cache_controller.handle_prefetch_operation()
        return original_check_hicache_events(self, *args, **kwargs)

    HiCacheController.backup_thread_func = noop_thread
    HiCacheController.prefetch_thread_func = noop_thread
    HiCacheController.handle_backup_operation = handle_backup_operation
    HiCacheController.handle_prefetch_operation = handle_prefetch_operation
    HiCacheController._generic_page_set = _generic_page_set
    HiRadixCache.check_hicache_events = check_hicache_events
    _patch_host_memory_pool(memory_pool_host_mod)

    for cls in (MHATokenToKVPoolHost, MLATokenToKVPoolHost, NSATokenToKVPoolHost):
        cls.load_to_device_per_layer = simulated_load_to_device_per_layer
        cls.backup_from_device_all_layer = simulated_backup_from_device_all_layer
        cls.get_data_page = simulated_get_data_page
        cls.set_from_flat_data_page = simulated_set_from_flat_data_page


def is_installed() -> bool:
    return _INSTALLED
