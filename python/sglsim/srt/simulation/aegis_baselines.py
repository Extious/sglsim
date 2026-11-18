"""Capacity-matched metadata baselines for AegisKV evaluation.

The classes in this module implement *declared component adaptations* of KV
protection systems whose complete runtimes do not share SGLSim's dense DP/PP
topology.  They deliberately reuse the same transfer/recompute coefficients and
byte limits as :mod:`aegis_full_design`.

The implementation is event-level: a sealed chunk reserves capacity, enters a
serialized copy lane, and has no recovery value until its transfer commits.
Completion, eviction, and request teardown invalidate pending work.  Recovery
can consume only the largest contiguous committed prefix.

These controls must be reported with their adaptation labels.  In particular,
``GHOSTSERVE_PARITY`` is meaningful only when the experiment explicitly sets a
TP-like ``parity_data_shards > 1`` scenario; it is not a TP=1 reproduction of
GhostServe.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Sequence

from sglsim.srt.simulation.aegis_full_design import CalibratedTransferPath, SealedChunk
from sglsim.srt.simulation.aegis_independent_tiers import RecoveryDecision, RecoveryTiming


class BaselinePolicy(str, Enum):
    """Protection component represented by :class:`ProtectionBaselineRuntime`."""

    DEJAVU_NEIGHBOR = "dejavu_neighbor"
    RAIDSERVE_HOST = "raidserve_host"
    KEVLARFLOW_RING = "kevlarflow_ring"
    GHOSTSERVE_PARITY = "ghostserve_parity"


class BaselineTier(str, Enum):
    HBM = "HBM"
    DRAM = "DRAM"


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


@dataclass(frozen=True)
class ProtectionBaselineConfig:
    """Shared resource contract for capacity-matched protection baselines."""

    policy: BaselinePolicy
    hbm_capacity_bytes: int
    dram_capacity_bytes: int
    kv_bytes_per_token: int
    hbm_copy: CalibratedTransferPath
    dram_copy: CalibratedTransferPath
    recompute: RecoveryTiming
    hbm_load: RecoveryTiming
    dram_load: RecoveryTiming
    pp_size: int = 1
    tp_size: int = 1
    parity_data_shards: int = 8
    parity_shards: int = 2
    parity_encode_ms_per_token: float = 0.0
    parity_decode_ms_per_token: float = 0.0
    event_log_limit: int = 128

    def __post_init__(self) -> None:
        for name in (
            "hbm_capacity_bytes",
            "dram_capacity_bytes",
            "kv_bytes_per_token",
            "pp_size",
            "tp_size",
            "parity_data_shards",
            "parity_shards",
        ):
            _positive_int(name, getattr(self, name))
        if self.parity_shards > self.parity_data_shards:
            raise ValueError("parity_shards cannot exceed parity_data_shards")
        _finite_nonnegative(
            "parity_encode_ms_per_token", self.parity_encode_ms_per_token
        )
        _finite_nonnegative(
            "parity_decode_ms_per_token", self.parity_decode_ms_per_token
        )
        if isinstance(self.event_log_limit, bool) or int(self.event_log_limit) < 0:
            raise ValueError("event_log_limit must be nonnegative")
        if (
            self.policy is BaselinePolicy.GHOSTSERVE_PARITY
            and self.parity_data_shards <= 1
        ):
            raise ValueError("GhostServe parity requires more than one data shard")

    @property
    def tier(self) -> BaselineTier:
        if self.policy in (
            BaselinePolicy.DEJAVU_NEIGHBOR,
            BaselinePolicy.KEVLARFLOW_RING,
        ):
            return BaselineTier.HBM
        return BaselineTier.DRAM

    @property
    def capacity_bytes(self) -> int:
        return (
            int(self.hbm_capacity_bytes)
            if self.tier is BaselineTier.HBM
            else int(self.dram_capacity_bytes)
        )


@dataclass(frozen=True)
class BaselineRequestProfile:
    request_id: str
    source_worker: str
    source_failure_domain: str
    target_worker: str
    target_failure_domain: str
    route_version: int
    total_tokens: int
    slack_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "source_worker",
            "source_failure_domain",
            "target_worker",
            "target_failure_domain",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must be nonempty")
        if isinstance(self.route_version, bool) or int(self.route_version) < 0:
            raise ValueError("route_version must be nonnegative")
        if isinstance(self.total_tokens, bool) or int(self.total_tokens) < 0:
            raise ValueError("total_tokens must be nonnegative")
        _finite_nonnegative("slack_ms", self.slack_ms)


@dataclass
class _RequestState:
    profile: BaselineRequestProfile
    registered_at_ms: float
    last_admitted_at_ms: float
    generation: int = 0
    next_ordinal: int = 0
    reserved_frontier: int = 0
    committed_frontier: int = 0
    used_bytes: int = 0
    blocked: bool = False
    completion_requested: bool = False
    eviction_count: int = 0


@dataclass(frozen=True)
class BaselineAdmission:
    request_id: str
    ordinal: int
    admitted: bool
    tier: BaselineTier
    reason: str
    logical_bytes: int
    stored_bytes: int
    evicted_requests: tuple[str, ...] = ()


@dataclass
class _CopyJob:
    request_id: str
    ordinal: int
    token_start: int
    token_end: int
    logical_bytes: int
    stored_bytes: int
    generation: int
    finish_ms: float


class ProtectionBaselineRuntime:
    """One fixed protection target with a serialized asynchronous copy lane."""

    def __init__(
        self,
        config: ProtectionBaselineConfig,
        *,
        target_id: str,
    ) -> None:
        self.config = config
        self.target_id = str(target_id)
        if not self.target_id:
            raise ValueError("target_id must be nonempty")
        self._now_ms = 0.0
        self._lane_available_ms = 0.0
        self._used_bytes = 0
        self._sequence = 0
        self._requests: dict[str, _RequestState] = {}
        self._jobs: list[tuple[float, int, _CopyJob]] = []
        self._metrics: dict[str, float] = {
            "registered_requests": 0,
            "sealed_chunks": 0,
            "admitted_chunks": 0,
            "capacity_misses": 0,
            "blocked_suffix_chunks": 0,
            "copy_jobs": 0,
            "copy_logical_bytes": 0,
            "copy_stored_bytes": 0,
            "copy_service_ms": 0,
            "committed_chunks": 0,
            "cancelled_jobs": 0,
            "evicted_requests": 0,
            "evicted_bytes": 0,
            "reclaimed_requests": 0,
        }
        self._events: list[dict[str, Any]] = []

    @property
    def now_ms(self) -> float:
        return self._now_ms

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def lane_available_ms(self) -> float:
        return self._lane_available_ms

    def free_bytes(self) -> int:
        return int(self.config.capacity_bytes) - self._used_bytes

    def _emit(self, kind: str, **fields: Any) -> None:
        limit = int(self.config.event_log_limit)
        if limit and len(self._events) >= limit:
            return
        self._events.append(
            {
                "time_ms": self._now_ms,
                "kind": kind,
                "policy": self.config.policy.value,
                **fields,
            }
        )

    def _copy_path(self) -> CalibratedTransferPath:
        return (
            self.config.hbm_copy
            if self.config.tier is BaselineTier.HBM
            else self.config.dram_copy
        )

    def _stored_bytes(self, logical_bytes: int) -> int:
        if self.config.policy is BaselinePolicy.GHOSTSERVE_PARITY:
            numerator = int(logical_bytes) * int(self.config.parity_shards)
            denominator = int(self.config.parity_data_shards)
            return max((numerator + denominator - 1) // denominator, 1)
        return int(logical_bytes)

    def _copy_duration_ms(
        self, *, logical_tokens: int, logical_bytes: int, stored_bytes: int
    ) -> float:
        duration = self._copy_path().duration_ms(stored_bytes)
        if self.config.policy is BaselinePolicy.GHOSTSERVE_PARITY:
            # Encoding consumes the full logical stripe while only parity bytes
            # cross the D2H path.
            duration += (
                int(logical_tokens) * self.config.parity_encode_ms_per_token
            )
        return duration

    def register_request(
        self, profile: BaselineRequestProfile, *, time_ms: float
    ) -> None:
        self.advance_to(time_ms)
        if profile.request_id in self._requests:
            raise ValueError(f"request {profile.request_id!r} already registered")
        self._requests[profile.request_id] = _RequestState(
            profile=profile,
            registered_at_ms=self._now_ms,
            last_admitted_at_ms=self._now_ms,
        )
        self._metrics["registered_requests"] += 1
        self._emit("request_registered", request_id=profile.request_id)

    def has_request(self, request_id: str) -> bool:
        return str(request_id) in self._requests

    def request_replica_state(self, request_id: str) -> dict[str, Any]:
        """Return the exact per-request tier residency without exposing internals.

        A request can remain registered after capacity pressure releases its
        physical bytes.  Callers that model content-addressed aliases must not
        confuse that registered-but-blocked identity with a resident replica.
        The method is intentionally read-only; ``advance_to`` remains the
        caller's responsibility so event time is explicit.
        """

        state = self._requests.get(str(request_id))
        if state is None:
            return {
                "registered": False,
                "presence": "ABSENT",
                "used_bytes": 0,
                "reserved_frontier": 0,
                "committed_frontier": 0,
                "blocked": False,
            }
        if state.used_bytes > 0 and state.committed_frontier > 0:
            presence = "COMMITTED"
        elif (
            state.used_bytes > 0
            and state.reserved_frontier > state.committed_frontier
            and not state.blocked
        ):
            presence = "PENDING"
        else:
            presence = "ABSENT"
        return {
            "registered": True,
            "presence": presence,
            "used_bytes": int(state.used_bytes),
            "reserved_frontier": int(state.reserved_frontier),
            "committed_frontier": int(state.committed_frontier),
            "blocked": bool(state.blocked),
        }

    def _release_request(self, request_id: str, *, reason: str) -> int:
        state = self._requests.get(str(request_id))
        if state is None:
            return 0
        released = state.used_bytes
        self._used_bytes -= released
        if self._used_bytes < 0:
            raise AssertionError("baseline capacity accounting underflow")
        state.used_bytes = 0
        state.generation += 1
        state.reserved_frontier = 0
        state.committed_frontier = 0
        state.blocked = True
        self._emit(
            "request_state_released",
            request_id=request_id,
            released_bytes=released,
            reason=reason,
        )
        return released

    def _evict_for(self, request_id: str, bytes_needed: int) -> tuple[str, ...]:
        if self.config.policy is not BaselinePolicy.KEVLARFLOW_RING:
            return ()
        victims = sorted(
            (
                state
                for rid, state in self._requests.items()
                if rid != request_id and state.used_bytes > 0
            ),
            key=lambda state: (
                state.last_admitted_at_ms,
                state.registered_at_ms,
                state.profile.request_id,
            ),
        )
        evicted: list[str] = []
        for victim in victims:
            if self.free_bytes() >= bytes_needed:
                break
            released = self._release_request(
                victim.profile.request_id, reason="kevlarflow_pressure_drop"
            )
            victim.eviction_count += 1
            self._metrics["evicted_requests"] += 1
            self._metrics["evicted_bytes"] += released
            evicted.append(victim.profile.request_id)
        return tuple(evicted)

    def seal_chunk(
        self,
        request_id: str,
        chunk: SealedChunk,
        *,
        time_ms: float,
    ) -> BaselineAdmission:
        self.advance_to(time_ms)
        state = self._requests[str(request_id)]
        if state.completion_requested:
            raise ValueError("cannot seal after request completion")
        if int(chunk.ordinal) != state.next_ordinal:
            raise ValueError("chunk ordinal is not progressive")
        state.next_ordinal += 1
        self._metrics["sealed_chunks"] += 1
        logical_bytes = int(chunk.token_count) * int(self.config.kv_bytes_per_token)
        stored_bytes = self._stored_bytes(logical_bytes)
        if state.blocked:
            self._metrics["blocked_suffix_chunks"] += 1
            return BaselineAdmission(
                request_id=str(request_id),
                ordinal=int(chunk.ordinal),
                admitted=False,
                tier=self.config.tier,
                reason="prefix_already_unprotected",
                logical_bytes=logical_bytes,
                stored_bytes=stored_bytes,
            )
        if int(chunk.token_start) != state.reserved_frontier:
            raise AssertionError("baseline reserved prefix became discontinuous")
        evicted = self._evict_for(str(request_id), stored_bytes)
        if self.free_bytes() < stored_bytes:
            state.blocked = True
            self._metrics["capacity_misses"] += 1
            self._emit(
                "capacity_miss",
                request_id=request_id,
                ordinal=chunk.ordinal,
                bytes_needed=stored_bytes,
                free_bytes=self.free_bytes(),
            )
            return BaselineAdmission(
                request_id=str(request_id),
                ordinal=int(chunk.ordinal),
                admitted=False,
                tier=self.config.tier,
                reason="capacity_miss",
                logical_bytes=logical_bytes,
                stored_bytes=stored_bytes,
                evicted_requests=evicted,
            )
        state.used_bytes += stored_bytes
        self._used_bytes += stored_bytes
        state.reserved_frontier = int(chunk.token_end)
        state.last_admitted_at_ms = self._now_ms
        duration = self._copy_duration_ms(
            logical_tokens=chunk.token_count,
            logical_bytes=logical_bytes,
            stored_bytes=stored_bytes,
        )
        finish = max(self._now_ms, self._lane_available_ms) + duration
        self._lane_available_ms = finish
        job = _CopyJob(
            request_id=str(request_id),
            ordinal=int(chunk.ordinal),
            token_start=int(chunk.token_start),
            token_end=int(chunk.token_end),
            logical_bytes=logical_bytes,
            stored_bytes=stored_bytes,
            generation=state.generation,
            finish_ms=finish,
        )
        self._sequence += 1
        heapq.heappush(self._jobs, (finish, self._sequence, job))
        self._metrics["admitted_chunks"] += 1
        self._metrics["copy_jobs"] += 1
        self._metrics["copy_logical_bytes"] += logical_bytes
        self._metrics["copy_stored_bytes"] += stored_bytes
        self._metrics["copy_service_ms"] += duration
        self._emit(
            "copy_scheduled",
            request_id=request_id,
            ordinal=chunk.ordinal,
            finish_ms=finish,
            stored_bytes=stored_bytes,
        )
        return BaselineAdmission(
            request_id=str(request_id),
            ordinal=int(chunk.ordinal),
            admitted=True,
            tier=self.config.tier,
            reason="admitted",
            logical_bytes=logical_bytes,
            stored_bytes=stored_bytes,
            evicted_requests=evicted,
        )

    def advance_to(self, time_ms: float) -> None:
        target = float(time_ms)
        if not math.isfinite(target) or target < self._now_ms - 1e-9:
            raise ValueError("baseline time cannot regress")
        while self._jobs and self._jobs[0][0] <= target + 1e-9:
            finish, _sequence, job = heapq.heappop(self._jobs)
            self._now_ms = finish
            state = self._requests.get(job.request_id)
            if (
                state is None
                or state.generation != job.generation
                or state.completion_requested
            ):
                self._metrics["cancelled_jobs"] += 1
                continue
            if job.token_start != state.committed_frontier:
                # A capacity pressure drop invalidates a whole replica.  A
                # later job must never punch a recoverable hole.
                self._metrics["cancelled_jobs"] += 1
                continue
            state.committed_frontier = job.token_end
            self._metrics["committed_chunks"] += 1
            self._emit(
                "copy_committed",
                request_id=job.request_id,
                ordinal=job.ordinal,
                token_end=job.token_end,
            )
        self._now_ms = target

    def finish_request(self, request_id: str, *, time_ms: float) -> None:
        self.advance_to(time_ms)
        state = self._requests.pop(str(request_id))
        state.completion_requested = True
        self._used_bytes -= state.used_bytes
        if self._used_bytes < 0:
            raise AssertionError("baseline capacity accounting underflow")
        state.generation += 1
        self._metrics["reclaimed_requests"] += 1
        self._emit(
            "request_finished",
            request_id=request_id,
            released_bytes=state.used_bytes,
        )

    def rotate_target(
        self,
        request_id: str,
        *,
        replacement_target: str,
        replacement_failure_domain: str,
        time_ms: float,
    ) -> None:
        """Invalidate fixed-neighbor/ring replicas after their target fails.

        Host/parity checkpoints are in a different failure domain and survive.
        The capacity-matched adaptations do not silently rebuild a complete
        fixed-neighbor prefix from a failed target.
        """

        self.advance_to(time_ms)
        state = self._requests.get(str(request_id))
        if state is None:
            return
        state.profile = replace(
            state.profile,
            target_worker=str(replacement_target),
            target_failure_domain=str(replacement_failure_domain),
            route_version=state.profile.route_version + 1,
        )
        if self.config.tier is BaselineTier.HBM:
            self._release_request(str(request_id), reason="protection_target_failed")

    def invalidate_target_failure(self, request_id: str, *, time_ms: float) -> int:
        """Invalidate HBM protection on a failed worker without re-protection.

        Host/parity state is outside the failed GPU worker and therefore
        survives.  The large-scale source-failure simulator uses this method
        for every policy before evaluating the request currently executing on
        that worker.
        """

        self.advance_to(time_ms)
        if self.config.tier is not BaselineTier.HBM:
            return 0
        state = self._requests.get(str(request_id))
        if state is None:
            return 0
        # A logical target can receive more than one physical-shard failure
        # before repair, and the adapter also records every sealed chunk that
        # arrives while that target is unavailable.  Once the replica has no
        # bytes and its suffix is blocked, another invalidation is an exact
        # retry: do not advance the generation again.  A blocked request may
        # still own an earlier admitted prefix after a capacity miss, so
        # ``blocked`` alone is not sufficient for this idempotence check.
        if state.blocked and state.used_bytes == 0:
            return 0
        return self._release_request(
            str(request_id), reason="physical_peer_worker_fail_stop"
        )

    def _prefix_recovery_ms(self, prefix_tokens: int) -> tuple[str, float]:
        prefix = max(int(prefix_tokens), 0)
        if prefix == 0:
            return "RECOMPUTE", 0.0
        if self.config.policy is BaselinePolicy.GHOSTSERVE_PARITY:
            parity_fraction = (
                float(self.config.parity_shards)
                / float(self.config.parity_data_shards)
            )
            surviving_fraction = (
                float(self.config.parity_data_shards - 1)
                / float(self.config.parity_data_shards)
            )
            value = self.config.dram_load.setup_ms
            value += prefix * self.config.dram_load.ms_per_token * parity_fraction
            value += prefix * self.config.hbm_load.ms_per_token * surviving_fraction
            value += prefix * self.config.parity_decode_ms_per_token
            return "GHOST_PARITY_RECONSTRUCT", value
        timing = (
            self.config.hbm_load
            if self.config.tier is BaselineTier.HBM
            else self.config.dram_load
        )
        action = "PEER_HBM" if self.config.tier is BaselineTier.HBM else "HOST_DRAM"
        return action, timing.setup_ms + prefix * timing.ms_per_token

    def estimate_source_failure(
        self,
        request_id: str,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        activated_target: str | None = None,
        time_ms: float,
    ) -> RecoveryDecision:
        self.advance_to(time_ms)
        state = self._requests.get(str(request_id))
        total = max(int(produced_tokens), 0)
        prefix = 0 if state is None else min(state.committed_frontier, total)
        action, recovery = self._prefix_recovery_ms(prefix)
        suffix = max(total - prefix, 0)
        if suffix:
            recovery += (
                self.config.recompute.setup_ms
                + suffix * self.config.recompute.ms_per_token
            )
        if prefix == 0:
            action = "RECOMPUTE"
        elif suffix:
            action += "+RECOMPUTE"
        target = (
            str(activated_target)
            if activated_target is not None
            else (state.profile.target_worker if state is not None else self.target_id)
        )
        route_version = 1 if state is None else state.profile.route_version
        slack = max(float(remaining_slack_ms), 0.0)
        return RecoveryDecision(
            request_id=str(request_id),
            action=action,
            restored_tokens=prefix,
            recompute_tokens=suffix,
            recovery_ms=recovery,
            slack_ms=slack,
            visible_harm_ms=max(recovery - slack, 0.0),
            route_version=route_version,
            activated_target=target,
        )

    def flush(self) -> None:
        if self._jobs:
            self.advance_to(max(self._lane_available_ms, self._now_ms))

    def validate_invariants(self) -> None:
        used = sum(state.used_bytes for state in self._requests.values())
        if used != self._used_bytes:
            raise AssertionError("baseline used-byte accounting mismatch")
        if not 0 <= self._used_bytes <= self.config.capacity_bytes:
            raise AssertionError("baseline capacity exceeded")
        for state in self._requests.values():
            if not 0 <= state.committed_frontier <= state.reserved_frontier:
                raise AssertionError("invalid baseline commit frontier")

    def snapshot(self) -> dict[str, Any]:
        self.validate_invariants()
        return {
            "policy": self.config.policy.value,
            "fidelity": "capacity_matched_component_adaptation",
            "target_id": self.target_id,
            "tier": self.config.tier.value,
            "now_ms": self._now_ms,
            "lane_available_ms": self._lane_available_ms,
            "capacity_bytes": self.config.capacity_bytes,
            "used_bytes": self._used_bytes,
            "active_requests": len(self._requests),
            "pending_jobs": len(self._jobs),
            "metrics": dict(self._metrics),
            "events": list(self._events),
        }


@dataclass(frozen=True)
class LumenConfig:
    """Load-aware holder adaptation of LUMEN's checkpoint component."""

    holder_count: int
    bytes_per_holder: int
    kv_bytes_per_token: int
    copy_path: CalibratedTransferPath
    recompute: RecoveryTiming
    host_load: RecoveryTiming
    placement_lambda: float = 1.0
    dispatch_overload_factor: float = 1.25

    def __post_init__(self) -> None:
        for name in ("holder_count", "bytes_per_holder", "kv_bytes_per_token"):
            _positive_int(name, getattr(self, name))
        _finite_nonnegative("placement_lambda", self.placement_lambda)
        factor = float(self.dispatch_overload_factor)
        if not math.isfinite(factor) or factor < 1.0:
            raise ValueError("dispatch_overload_factor must be finite and >= 1")


@dataclass
class _LumenRequest:
    profile: BaselineRequestProfile
    holder: int | None
    reservation_bytes: int
    generation: int = 0
    next_ordinal: int = 0
    committed_frontier: int = 0
    copied_bytes: int = 0


@dataclass
class _LumenHolder:
    reserved_bytes: int = 0
    lane_available_ms: float = 0.0


@dataclass
class _LumenJob:
    request_id: str
    token_start: int
    token_end: int
    generation: int
    finish_ms: float
    bytes: int


class LumenBaselineRuntime:
    """Cluster-level load-aware checkpoint holder and dispatch adaptation."""

    def __init__(self, config: LumenConfig) -> None:
        self.config = config
        self._now_ms = 0.0
        self._holders = [_LumenHolder() for _ in range(config.holder_count)]
        self._requests: dict[str, _LumenRequest] = {}
        self._jobs: list[tuple[float, int, _LumenJob]] = []
        self._sequence = 0
        self._metrics: dict[str, float] = {
            "registered_requests": 0,
            "placed_requests": 0,
            "reservation_failures": 0,
            "reserved_bytes_peak": 0,
            "copy_bytes": 0,
            "copy_service_ms": 0,
            "committed_chunks": 0,
            "cancelled_jobs": 0,
            "dispatch_moves": 0,
        }

    def _candidate_score(self, holder_id: int, reservation_bytes: int) -> float:
        holder = self._holders[holder_id]
        queue_ms = max(holder.lane_available_ms - self._now_ms, 0.0)
        projected_copy_ms = (
            holder.reserved_bytes + reservation_bytes
        ) / self.config.copy_path.bandwidth_bytes_per_ms
        return queue_ms + self.config.placement_lambda * projected_copy_ms

    def register_request(
        self,
        profile: BaselineRequestProfile,
        *,
        source_rank: int,
        time_ms: float,
    ) -> int | None:
        self.advance_to(time_ms)
        if profile.request_id in self._requests:
            raise ValueError(f"request {profile.request_id!r} already registered")
        reservation = int(profile.total_tokens) * int(self.config.kv_bytes_per_token)
        candidates = [
            holder_id
            for holder_id, holder in enumerate(self._holders)
            if holder_id != int(source_rank)
            and holder.reserved_bytes + reservation <= self.config.bytes_per_holder
        ]
        holder_id = (
            min(candidates, key=lambda value: (self._candidate_score(value, reservation), value))
            if candidates
            else None
        )
        state = _LumenRequest(
            profile=profile,
            holder=holder_id,
            reservation_bytes=reservation,
        )
        self._requests[profile.request_id] = state
        self._metrics["registered_requests"] += 1
        if holder_id is None:
            self._metrics["reservation_failures"] += 1
        else:
            self._holders[holder_id].reserved_bytes += reservation
            self._metrics["placed_requests"] += 1
            self._metrics["reserved_bytes_peak"] = max(
                self._metrics["reserved_bytes_peak"],
                sum(holder.reserved_bytes for holder in self._holders),
            )
        return holder_id

    def has_request(self, request_id: str) -> bool:
        return str(request_id) in self._requests

    def seal_chunk(
        self, request_id: str, chunk: SealedChunk, *, time_ms: float
    ) -> bool:
        self.advance_to(time_ms)
        state = self._requests[str(request_id)]
        if chunk.ordinal != state.next_ordinal:
            raise ValueError("LUMEN chunk ordinal is not progressive")
        state.next_ordinal += 1
        if state.holder is None:
            return False
        byte_size = int(chunk.token_count) * int(self.config.kv_bytes_per_token)
        if state.copied_bytes + byte_size > state.reservation_bytes:
            raise AssertionError("LUMEN copy exceeded request reservation")
        holder = self._holders[state.holder]
        duration = self.config.copy_path.duration_ms(byte_size)
        finish = max(holder.lane_available_ms, self._now_ms) + duration
        holder.lane_available_ms = finish
        self._sequence += 1
        heapq.heappush(
            self._jobs,
            (
                finish,
                self._sequence,
                _LumenJob(
                    request_id=str(request_id),
                    token_start=int(chunk.token_start),
                    token_end=int(chunk.token_end),
                    generation=state.generation,
                    finish_ms=finish,
                    bytes=byte_size,
                ),
            ),
        )
        state.copied_bytes += byte_size
        self._metrics["copy_bytes"] += byte_size
        self._metrics["copy_service_ms"] += duration
        return True

    def advance_to(self, time_ms: float) -> None:
        target = float(time_ms)
        if not math.isfinite(target) or target < self._now_ms - 1e-9:
            raise ValueError("LUMEN time cannot regress")
        while self._jobs and self._jobs[0][0] <= target + 1e-9:
            finish, _sequence, job = heapq.heappop(self._jobs)
            self._now_ms = finish
            state = self._requests.get(job.request_id)
            if state is None or state.generation != job.generation:
                self._metrics["cancelled_jobs"] += 1
                continue
            if job.token_start != state.committed_frontier:
                self._metrics["cancelled_jobs"] += 1
                continue
            state.committed_frontier = job.token_end
            self._metrics["committed_chunks"] += 1
        self._now_ms = target

    def finish_request(self, request_id: str, *, time_ms: float) -> None:
        self.advance_to(time_ms)
        state = self._requests.pop(str(request_id))
        state.generation += 1
        if state.holder is not None:
            holder = self._holders[state.holder]
            holder.reserved_bytes -= state.reservation_bytes
            if holder.reserved_bytes < 0:
                raise AssertionError("LUMEN reservation accounting underflow")

    def estimate_source_failure(
        self,
        request_id: str,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        time_ms: float,
    ) -> RecoveryDecision:
        self.advance_to(time_ms)
        state = self._requests.get(str(request_id))
        total = max(int(produced_tokens), 0)
        prefix = 0 if state is None else min(state.committed_frontier, total)
        suffix = total - prefix
        recovery = 0.0
        if prefix:
            recovery += (
                self.config.host_load.setup_ms
                + prefix * self.config.host_load.ms_per_token
            )
        if suffix:
            recovery += (
                self.config.recompute.setup_ms
                + suffix * self.config.recompute.ms_per_token
            )
        action = "RECOMPUTE"
        if prefix:
            action = "LUMEN_HOST" + ("+RECOMPUTE" if suffix else "")
        holder = None if state is None else state.holder
        # A checkpoint holder is also the first recovery serving target in the
        # LUMEN adaptation.  Keep the common target-* namespace so queueing is
        # directly comparable with other policies.
        target = f"target-{holder}" if holder is not None else "unplaced"
        slack = max(float(remaining_slack_ms), 0.0)
        return RecoveryDecision(
            request_id=str(request_id),
            action=action,
            restored_tokens=prefix,
            recompute_tokens=suffix,
            recovery_ms=recovery,
            slack_ms=slack,
            visible_harm_ms=max(recovery - slack, 0.0),
            route_version=1 if state is None else state.profile.route_version,
            activated_target=target,
        )

    def dispatch(
        self,
        decisions: Sequence[RecoveryDecision],
        *,
        target_count: int,
    ) -> tuple[RecoveryDecision, ...]:
        """Apply LUMEN-style locality-first overload shedding.

        Initial work stays with the checkpoint holder.  A holder whose queued
        service exceeds ``overload_factor * mean`` sheds the smallest
        checkpoint first to the least-loaded serving target.  Moving a request
        charges an additional host-state transfer over the calibrated path.
        """

        _positive_int("target_count", target_count)
        mutable = list(decisions)
        loads = {f"target-{rank}": 0.0 for rank in range(int(target_count))}
        by_holder: dict[str, list[int]] = {}
        for index, decision in enumerate(mutable):
            holder = decision.activated_target
            by_holder.setdefault(holder, []).append(index)
        holder_loads = {
            holder: sum(mutable[index].recovery_ms for index in indices)
            for holder, indices in by_holder.items()
        }
        total = sum(holder_loads.values())
        mean = total / max(int(target_count), 1)
        threshold = self.config.dispatch_overload_factor * mean
        for holder, indices in sorted(by_holder.items()):
            current = holder_loads[holder]
            if holder == "unplaced":
                ordered = sorted(indices, key=lambda i: (-mutable[i].recovery_ms, i))
            else:
                ordered = sorted(
                    indices,
                    key=lambda i: (mutable[i].restored_tokens, mutable[i].request_id),
                )
            for index in ordered:
                if holder != "unplaced" and current <= threshold + 1e-9:
                    break
                target = min(loads, key=lambda name: (loads[name], name))
                decision = mutable[index]
                movement = 0.0
                if holder != "unplaced" and decision.restored_tokens:
                    movement = self.config.copy_path.duration_ms(
                        decision.restored_tokens * self.config.kv_bytes_per_token
                    )
                moved = replace(
                    decision,
                    recovery_ms=decision.recovery_ms + movement,
                    visible_harm_ms=max(
                        decision.recovery_ms + movement - decision.slack_ms, 0.0
                    ),
                    activated_target=target,
                )
                mutable[index] = moved
                loads[target] += moved.recovery_ms
                current -= decision.recovery_ms
                self._metrics["dispatch_moves"] += 1
        return tuple(mutable)

    def flush(self) -> None:
        if self._jobs:
            self.advance_to(max(holder.lane_available_ms for holder in self._holders))

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": "lumen_load_aware",
            "fidelity": "capacity_matched_component_adaptation",
            "now_ms": self._now_ms,
            "holders": [asdict(holder) for holder in self._holders],
            "active_requests": len(self._requests),
            "pending_jobs": len(self._jobs),
            "metrics": dict(self._metrics),
        }


@dataclass(frozen=True)
class RestorePlan:
    planner: str
    total_tokens: int
    externally_available_tokens: int
    restored_tokens: int
    recomputed_tokens: int
    completion_ms: float
    io_service_ms: float
    compute_service_ms: float
    peak_staging_bytes: int
    request_commit: bool
    fallback_reason: str = ""


def restore_or_recompute_split(
    *,
    total_tokens: int,
    mandatory_recompute_tokens: int,
    io_ms_per_token: float,
    recompute_ms_per_token: float,
    kv_bytes_per_token: int,
) -> RestorePlan:
    """Closed-form parallel restore/recompute split from the anonymous draft.

    ``mandatory_recompute_tokens`` is the suffix unavailable in external
    storage.  The remaining prefix is split so parallel I/O and recomputation
    finish as close together as possible.
    """

    total = max(int(total_tokens), 0)
    mandatory = min(max(int(mandatory_recompute_tokens), 0), total)
    x = _finite_nonnegative("io_ms_per_token", io_ms_per_token)
    y = _finite_nonnegative("recompute_ms_per_token", recompute_ms_per_token)
    if x + y <= 0.0:
        restored = total - mandatory
    else:
        recompute_total = total * x / (x + y)
        optional_recompute = max(
            0,
            min(total - mandatory, int(round(recompute_total - mandatory))),
        )
        restored = total - mandatory - optional_recompute
    recomputed = total - restored
    io_ms = restored * x
    compute_ms = recomputed * y
    return RestorePlan(
        planner="restore_or_recompute_split",
        total_tokens=total,
        externally_available_tokens=total - mandatory,
        restored_tokens=restored,
        recomputed_tokens=recomputed,
        completion_ms=max(io_ms, compute_ms),
        io_service_ms=io_ms,
        compute_service_ms=compute_ms,
        peak_staging_bytes=restored * int(kv_bytes_per_token),
        request_commit=True,
    )


def cacheflow_3d_restore(
    *,
    total_tokens: int,
    externally_available_tokens: int,
    io_ms_per_token: float,
    recompute_ms_per_token: float,
    kv_bytes_per_token: int,
    layers: int,
    gpu_lanes: int,
    token_tiles: int,
) -> RestorePlan:
    """Metadata model of CacheFlow's token/layer/GPU restoration overlap.

    Work is tiled in all three dimensions.  External tiles are loaded when
    doing so has positive marginal recomputation reduction; the declared GPU
    lanes overlap I/O with recomputation.  This function is a recovery-executor
    control, not a CacheFlow end-to-end reproduction.
    """

    total = max(int(total_tokens), 0)
    external = min(max(int(externally_available_tokens), 0), total)
    layer_count = _positive_int("layers", layers)
    lanes = _positive_int("gpu_lanes", gpu_lanes)
    tiles = _positive_int("token_tiles", token_tiles)
    io_rate = _finite_nonnegative("io_ms_per_token", io_ms_per_token)
    compute_rate = _finite_nonnegative("recompute_ms_per_token", recompute_ms_per_token)
    # Positive marginal reduction chooses I/O for all externally available
    # tokens only when their per-layer transfer is cheaper than recomputation.
    use_external = external if io_rate < compute_rate else 0
    recomputed = total - use_external
    tile_parallelism = max(min(lanes, layer_count, tiles), 1)
    io_ms = use_external * io_rate
    compute_ms = recomputed * compute_rate
    completion = max(io_ms, compute_ms / tile_parallelism)
    return RestorePlan(
        planner="cacheflow_3d_restore",
        total_tokens=total,
        externally_available_tokens=external,
        restored_tokens=use_external,
        recomputed_tokens=recomputed,
        completion_ms=completion,
        io_service_ms=io_ms,
        compute_service_ms=compute_ms,
        peak_staging_bytes=(
            math.ceil(max(use_external, 0) / tiles)
            * int(kv_bytes_per_token)
            * tile_parallelism
        ),
        request_commit=True,
    )


def bounded_state_restore(
    *,
    total_tokens: int,
    externally_available_tokens: int,
    chunk_tokens: int,
    window_chunks: int,
    kv_bytes_per_token: int,
    io_timing: RecoveryTiming,
    recompute_timing: RecoveryTiming,
    failed_install_chunk: int | None = None,
) -> RestorePlan:
    """Bounded-State Restoration working-set and all-or-fallback semantics."""

    total = max(int(total_tokens), 0)
    external = min(max(int(externally_available_tokens), 0), total)
    chunk = _positive_int("chunk_tokens", chunk_tokens)
    window = _positive_int("window_chunks", window_chunks)
    chunks = math.ceil(external / chunk) if external else 0
    failure = failed_install_chunk is not None and 0 <= int(failed_install_chunk) < chunks
    if failure:
        restored = 0
        recomputed = total
        io_ms = 0.0
        compute_ms = (
            recompute_timing.setup_ms + total * recompute_timing.ms_per_token
            if total
            else 0.0
        )
        return RestorePlan(
            planner="bounded_state_restore",
            total_tokens=total,
            externally_available_tokens=external,
            restored_tokens=0,
            recomputed_tokens=recomputed,
            completion_ms=compute_ms,
            io_service_ms=io_ms,
            compute_service_ms=compute_ms,
            peak_staging_bytes=min(external, window * chunk)
            * int(kv_bytes_per_token),
            request_commit=False,
            fallback_reason="partial_install_invalidated",
        )
    restored = external
    recomputed = total - restored
    io_ms = (
        io_timing.setup_ms + restored * io_timing.ms_per_token if restored else 0.0
    )
    compute_ms = (
        recompute_timing.setup_ms
        + recomputed * recompute_timing.ms_per_token
        if recomputed
        else 0.0
    )
    return RestorePlan(
        planner="bounded_state_restore",
        total_tokens=total,
        externally_available_tokens=external,
        restored_tokens=restored,
        recomputed_tokens=recomputed,
        completion_ms=io_ms + compute_ms,
        io_service_ms=io_ms,
        compute_service_ms=compute_ms,
        peak_staging_bytes=min(external, window * chunk)
        * int(kv_bytes_per_token),
        request_commit=True,
    )


def aggregate_snapshots(
    runtimes: Iterable[ProtectionBaselineRuntime],
) -> dict[str, Any]:
    snapshots = [runtime.snapshot() for runtime in runtimes]
    metrics: dict[str, float] = {}
    for snapshot in snapshots:
        for name, value in snapshot["metrics"].items():
            metrics[name] = metrics.get(name, 0.0) + float(value)
    return {
        "targets": len(snapshots),
        "policy": snapshots[0]["policy"] if snapshots else "",
        "metrics": metrics,
        "peak_or_final_used_bytes": sum(
            int(snapshot["used_bytes"]) for snapshot in snapshots
        ),
        "snapshots": snapshots,
    }
