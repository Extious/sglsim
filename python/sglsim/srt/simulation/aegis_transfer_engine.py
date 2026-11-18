"""Policy-free transfer queue for current-design AegisKV.

Storage control has already selected a pool and any victims before this engine
is called.  The engine receives an immutable source-owned copy submission and
does only queueing plus calibrated per-lane rate limiting.  It deliberately
has no placement, Landlord, quota, or victim-selection API.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

from sglsim.srt.simulation.aegis_independent_tiers import ChunkSpec, ProtectionTier


@dataclass(frozen=True)
class TransferSubmission:
    tier: ProtectionTier
    spec: ChunkSpec
    destination_id: str
    copy_kind: str
    ready_at_ms: float
    service_ms: float
    transfer_succeeded: bool
    acked_shards: tuple[int, ...]
    source_owned: bool = True
    target_owned: bool = False
    hbm_to_dram: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", ProtectionTier(self.tier))
        if not str(self.destination_id).strip():
            raise ValueError("transfer destination_id must be nonempty")
        if not math.isfinite(float(self.ready_at_ms)) or self.ready_at_ms < 0.0:
            raise ValueError("transfer ready_at_ms must be finite and nonnegative")
        if not math.isfinite(float(self.service_ms)) or self.service_ms < 0.0:
            raise ValueError("transfer service_ms must be finite and nonnegative")
        expected_kind = (
            "source_d2d" if self.tier is ProtectionTier.HBM else "source_d2h"
        )
        if self.copy_kind != expected_kind:
            raise ValueError("transfer copy_kind does not match its independent tier")
        if (
            self.tier is ProtectionTier.HBM
            and self.destination_id != self.spec.hbm_target
        ):
            raise ValueError("Peer-HBM transfer must target the route-bound HBM worker")
        if (
            self.tier is ProtectionTier.DRAM
            and self.destination_id == self.spec.hbm_target
        ):
            raise ValueError("DRAM transfer must target final Store DRAM directly")
        if not self.source_owned or self.target_owned or self.hbm_to_dram:
            raise ValueError("current design permits only direct source-owned transfer")
        expected_shards = tuple(range(self.spec.shard_count))
        if self.transfer_succeeded:
            if tuple(sorted(set(self.acked_shards))) != expected_shards:
                raise ValueError("successful transfer requires all shard ACKs")
        elif self.acked_shards:
            raise ValueError("failed transfer cannot claim shard ACKs")


@dataclass
class TransferJob:
    job_id: str
    submission: TransferSubmission
    started_at_ms: float
    completed_at_ms: float
    cancelled: bool = False

    @property
    def tier(self) -> ProtectionTier:
        return self.submission.tier

    @property
    def spec(self) -> ChunkSpec:
        return self.submission.spec

    @property
    def transfer_succeeded(self) -> bool:
        return self.submission.transfer_succeeded

    @property
    def acked_shards(self) -> tuple[int, ...]:
        return self.submission.acked_shards


class AegisTransferEngine:
    """Two independent serial lanes with no storage-policy capabilities."""

    def __init__(self) -> None:
        self._lane_available_ms = {
            ProtectionTier.HBM: 0.0,
            ProtectionTier.DRAM: 0.0,
        }
        self._heap: list[tuple[float, int, str]] = []
        self._jobs: dict[str, TransferJob] = {}
        self._sequence = 0

    def enqueue(self, submission: TransferSubmission) -> TransferJob:
        tier = submission.tier
        start = max(
            float(submission.ready_at_ms), self._lane_available_ms[tier]
        )
        finish = start + float(submission.service_ms)
        self._lane_available_ms[tier] = finish
        self._sequence += 1
        job_id = f"{tier.value}:{submission.spec.logical_id}:{self._sequence}"
        job = TransferJob(
            job_id=job_id,
            submission=submission,
            started_at_ms=start,
            completed_at_ms=finish,
        )
        self._jobs[job_id] = job
        heapq.heappush(self._heap, (finish, self._sequence, job_id))
        return job

    def pop_next_ready(self, through_ms: float) -> TransferJob | None:
        target = float(through_ms)
        if not self._heap or self._heap[0][0] > target + 1e-9:
            return None
        _finish, _sequence, job_id = heapq.heappop(self._heap)
        return self._jobs.pop(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(str(job_id))
        if job is None:
            return False
        job.cancelled = True
        return True

    @property
    def pending_count(self) -> int:
        return len(self._jobs)

    @property
    def maximum_completion_ms(self) -> float | None:
        return max((row[0] for row in self._heap), default=None)

    @property
    def lane_available_ms(self) -> dict[ProtectionTier, float]:
        return dict(self._lane_available_ms)

    def snapshot(self) -> dict[str, Any]:
        return {
            "control_boundary": "queue_and_rate_limit_only",
            "storage_control_selects_victims": True,
            "transfer_engine_selects_victims": False,
            "rate_limit_mode": "independent_calibrated_serial_lane",
            "pending_jobs": self.pending_count,
            "lane_available_ms": {
                tier.value: value
                for tier, value in self._lane_available_ms.items()
            },
        }


__all__ = ["AegisTransferEngine", "TransferJob", "TransferSubmission"]
