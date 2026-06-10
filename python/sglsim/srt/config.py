from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopologyConfig:
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1


@dataclass(frozen=True)
class SchedulerConfig:
    max_running_reqs: int = 16
    max_prefill_tokens: int = 16384
    chunked_prefill_size: int | None = None


@dataclass(frozen=True)
class CacheConfig:
    page_size: int = 16
    l1_capacity_pages: int = 1024
    l2_capacity_pages: int = 0
    l3_capacity_pages: int = 0
    hicache_write_policy: str = "write_through_selective"
    write_through_threshold: int = 2
    load_back_threshold: int = 10
    prefetch_threshold: int = 256
    prefetch_stop_policy: str = "best_effort"
    prefetch_capacity_limit_ratio: float = 0.8
    hicache_storage_backend: str | None = None


@dataclass(frozen=True)
class SGLSimConfig:
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
