from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sglsim.srt.observability.scheduler_metrics_mixin import PrefillStats


class RequestStatus(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"
    ABORTED = "aborted"


class ForwardMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class SamplingParams:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Req:
    rid: str
    input_ids: list[int]
    sampling_params: SamplingParams
    input_token_count: int | None = None
    created_time: float = 0.0
    routing_key: str | None = None
    queue_start_time: float = -1.0
    queue_end_time: float = -1.0
    cached_length: int = 0
    cache_hit_tokens: int = 0
    cache_hit_recorded: bool = False
    cache_hit_log_pending: int = 0
    output_ids: list[int] = field(default_factory=list)
    assigned_rank: int | None = None
    priority: int = 0
    status: RequestStatus = RequestStatus.WAITING
    prefill_chunks_remaining: list[int] = field(default_factory=list)
    prefill_plan_synced: bool = False
    kv_committed_length: int = 0
    cache_protected_len: int = 0
    last_node_id: int | None = None
    host_hit_length: int = 0
    admit_load_back_tokens: int = 0
    last_host_node_id: int | None = None
    prefetch_status: str = "none"
    is_retracted: bool = False
    decode_pages: list = field(default_factory=list)
    prefill_alloc_page_ids: list[int] = field(default_factory=list)
    prefill_blocked_retries: int = 0
    kv_released: bool = False
    original_prompt_len: int | None = None
    original_max_new_tokens: int | None = None
    failover_prefix_ids: list[int] | None = None
    pre_failover_output_tokens: int = 0
    pre_failover_backed_up_tokens: int = 0
    failover_source_dp_rank: int | None = None
    is_failover_retried: bool = False
    failover_recompute_tokens: int = 0
    peer_backed_up_tokens: int = 0

    @property
    def fill_ids(self) -> list[int]:
        """Current sequence tokens mirrored from SGLang ``Req.fill_ids``."""
        return self.input_ids + self.output_ids

    @property
    def prefill_length(self) -> int:
        return len(self.fill_ids)

    @property
    def decode_length(self) -> int:
        return len(self.output_ids)

    @property
    def total_length(self) -> int:
        return self.prefill_length

    @property
    def remaining_decode_tokens(self) -> int:
        if self.original_max_new_tokens is not None:
            return max(self.original_max_new_tokens - self.total_generated_tokens, 0)
        return max(self.sampling_params.max_new_tokens - self.decode_length, 0)

    def finished(self) -> bool:
        # Mirrors SGLang Req.finished(): this is true only after the result
        # processor has observed a finish condition and marked terminal state.
        return self.status in (
            RequestStatus.COMPLETED,
            RequestStatus.ABORTED,
        )

    def reached_decode_limit(self) -> bool:
        return self.remaining_decode_tokens == 0

    def mark_finished_length(self) -> bool:
        if not self.reached_decode_limit():
            return False
        self.status = RequestStatus.COMPLETED
        return True

    @property
    def total_generated_tokens(self) -> int:
        return self.pre_failover_output_tokens + self.decode_length

    @property
    def extend_length(self) -> int:
        uncached_prefill = max(self.prefill_length - self.kv_committed_length, 0)
        if self.prefill_chunks_remaining:
            if uncached_prefill <= 0:
                return 0
            return min(self.prefill_chunks_remaining[0], uncached_prefill)
        return uncached_prefill

    def append_output_token(self, token_id: int) -> None:
        self.output_ids.append(token_id)

    @property
    def is_prefill_only(self) -> bool:
        return self.sampling_params.max_new_tokens == 0


@dataclass
class ScheduleBatch:
    batch_id: int
    rank_id: int
    mode: ForwardMode
    reqs: list[Req]
    started_time: float = 0.0
    finished_time: float = 0.0
    inference_dur_s: float = 0.0
    pipeline_overlap: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    batch_is_full: bool = False
    prefill_stats: PrefillStats | None = None

    def _planned_prefill_tokens(self, req: Req) -> int:
        planned = self.metadata.get("prefill_extend_tokens")
        if isinstance(planned, dict):
            value = planned.get(req.rid)
            if value is not None:
                return int(value)
        return req.extend_length

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def copy(self) -> ScheduleBatch:
        """Snapshot fields needed by overlap result processing.

        Mirrors SGLang ``ScheduleBatch.copy``: the request objects are shared,
        but the req list and metadata dict are copied so later in-place
        ``last_batch.filter_batch`` / merge mutations do not corrupt the
        pending result.
        """
        return ScheduleBatch(
            batch_id=self.batch_id,
            rank_id=self.rank_id,
            mode=self.mode,
            reqs=self.reqs[:],
            started_time=self.started_time,
            finished_time=self.finished_time,
            inference_dur_s=self.inference_dur_s,
            pipeline_overlap=self.pipeline_overlap,
            metadata=dict(self.metadata),
            batch_is_full=self.batch_is_full,
            prefill_stats=self.prefill_stats,
        )

    @property
    def is_prefill_only(self) -> bool:
        return bool(self.reqs) and all(req.is_prefill_only for req in self.reqs)

    def is_extend(self) -> bool:
        return self.mode == ForwardMode.PREFILL

    def merge_batch(self, other: ScheduleBatch) -> None:
        self.reqs.extend(other.reqs)

    def filter_batch(self, exclude: list[Req] | None = None) -> None:
        excluded_ids = {id(req) for req in (exclude or [])}
        kept: list[Req] = []
        for req in self.reqs:
            if id(req) in excluded_ids:
                continue
            if req.finished():
                continue
            kept.append(req)
        self.reqs = kept

    def prepare_for_extend(self) -> None:
        self.mode = ForwardMode.PREFILL

    def prepare_for_decode(self) -> None:
        self.mode = ForwardMode.DECODE
        for req in self.reqs:
            if req.finished() or req.reached_decode_limit():
                continue
            req.status = RequestStatus.DECODE

    def new_tokens_required_next_decode(self, page_size: int) -> int:
        if page_size <= 1:
            return len(self.reqs)
        new_pages = sum(
            1 for req in self.reqs if req.kv_committed_length % page_size == 0
        )
        return new_pages * page_size

    def check_decode_mem(self, cache) -> bool:
        page_size = cache.config.page_size
        needed = self.new_tokens_required_next_decode(page_size)
        cache.evict_from_tree_cache(needed, owner="check_decode_mem")
        return cache.available_device_tokens() >= needed

    @property
    def prediction_items(self) -> list[dict[str, int]]:
        if self.mode == ForwardMode.PREFILL:
            return [
                {
                    "extend_length": self._planned_prefill_tokens(req),
                    "past_kv_length": req.kv_committed_length,
                }
                for req in self.reqs
            ]
        return [
            {"extend_length": 1, "past_kv_length": req.kv_committed_length}
            for req in self.reqs
        ]

    def to_predict_batch(self):
        from sglsim.srt.time_predictor.base import ScheduleBatch as PredBatch
        from sglsim.srt.time_predictor.base import ScheduleRequest

        if self.mode == ForwardMode.PREFILL:
            past_lengths = self.metadata.get("prefill_past_kv_lengths", {})
            items = [
                ScheduleRequest(
                    extend_length=self._planned_prefill_tokens(req),
                    past_kv_length=int(
                        past_lengths.get(req.rid, req.kv_committed_length)
                    ),
                )
                for req in self.reqs
            ]
        else:
            items = [
                ScheduleRequest(
                    extend_length=1,
                    past_kv_length=req.kv_committed_length,
                )
                for req in self.reqs
            ]
        return PredBatch(reqs=items)


Batch = ScheduleBatch
