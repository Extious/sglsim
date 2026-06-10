from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SimulationMode(str, Enum):
    STREAMING = "STREAMING"


@dataclass
class RequestStats:
    rid: str = ""
    status: str = "running"
    last_event_time: float = 0.0
    input_length: int = 0
    output_length: int = 0
    generated_tokens: int = 0
    cache_hit_tokens: int = 0
    load_back_tokens: int = 0
    prefetch_complete_tokens: int = 0
    queue_start: float = -1.0
    queue_end: float = -1.0
    created_time: float = -1.0
    service_time_s: float = 0.0
    gen_token_latencies: list[float] = field(default_factory=list)
    is_failover_retried: bool = False
    failed_dp_rank: int = -1
    pre_failover_output_tokens: int = 0
    pre_failover_backed_up_tokens: int = 0
    recompute_tokens: int = 0
    failover_overhead_s: float = 0.0
    failover_injected_time: float = 0.0
    failover_retry_time: float = 0.0
    agent_id: str = ""
    critical_agent: bool = False

    def is_complete(self) -> bool:
        return self.status == "completed"

    def to_sim_stats_dict(self) -> dict:
        return {
            "rid": self.rid,
            "status": self.status,
            "created_time": self.created_time,
            "queue_start": self.queue_start,
            "queue_end": self.queue_end,
            "last_event_time": self.last_event_time,
            "service_time_s": self.service_time_s,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "generated_tokens": self.generated_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "load_back_tokens": self.load_back_tokens,
            "prefetch_complete_tokens": self.prefetch_complete_tokens,
            "gen_token_latencies": list(self.gen_token_latencies),
            "is_failover_retried": self.is_failover_retried,
            "failed_dp_rank": self.failed_dp_rank,
            "pre_failover_output_tokens": self.pre_failover_output_tokens,
            "pre_failover_backed_up_tokens": self.pre_failover_backed_up_tokens,
            "recompute_tokens": self.recompute_tokens,
            "failover_overhead_s": self.failover_overhead_s,
            "failover_injected_time": self.failover_injected_time,
            "failover_retry_time": self.failover_retry_time,
            "agent_id": self.agent_id,
            "critical_agent": self.critical_agent,
        }
