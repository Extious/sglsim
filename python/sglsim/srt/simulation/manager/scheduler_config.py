from __future__ import annotations

from dataclasses import dataclass


def infer_chunked_prefill_size(hbm_capacity_gb: float) -> int:
    """Match SGLang server_args memory-tier defaults (gpu_mem in MB)."""
    gpu_mem_mb = float(hbm_capacity_gb) * 1024.0
    if gpu_mem_mb < 20 * 1024:
        return 2048
    if gpu_mem_mb < 35 * 1024:
        return 2048
    if gpu_mem_mb < 60 * 1024:
        return 4096
    if gpu_mem_mb < 160 * 1024:
        return 8192
    return 16384


def resolve_chunked_prefill_size(
    raw_value: object | None,
    *,
    hbm_capacity_gb: float,
    infer_if_missing: bool,
) -> int | None:
    if not infer_if_missing:
        if raw_value is None:
            return None
        size = int(raw_value)
        return None if size <= 0 else size
    if raw_value is None:
        return infer_chunked_prefill_size(hbm_capacity_gb)
    size = int(raw_value)
    return None if size <= 0 else size


def hbm_capacity_gb_from_platform(raw_platform: dict | None) -> float:
    platform = raw_platform if isinstance(raw_platform, dict) else {}
    accel = platform.get("accelerator", {})
    if isinstance(accel, dict) and accel.get("hbm_capacity_gb") is not None:
        return float(accel["hbm_capacity_gb"])
    return 80.0


def _parse_max_running(raw: dict) -> int | None:
    if raw.get("max_running_requests") is not None:
        value = int(raw["max_running_requests"])
        return None if value <= 0 else value
    if raw.get("max_running_reqs") is not None:
        value = int(raw["max_running_reqs"])
        return None if value <= 0 else value
    return None


@dataclass
class SchedulerSimConfig:
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    backend_name: str = "sglang"
    backend_version: str | None = None
    mem_fraction_static: float = 0.88
    max_running_reqs: int = 16
    max_running_requests: int | None = None
    max_prefill_tokens: int = 16384
    chunked_prefill_size: int | None = None
    disable_overlap_schedule: bool = False
    load_balance_method: str = "round_robin"
    schedule_policy: str = "fcfs"
    decode_log_interval: int = 40
    prefill_max_requests: int | None = None
    init_new_token_ratio: float = 0.7
    pp_max_micro_batch_size: int | None = None
    enable_routing_key_affinity: bool = False

    @property
    def effective_max_running_requests(self) -> int:
        if self.max_running_requests is not None:
            return self.max_running_requests
        return self.max_running_reqs
