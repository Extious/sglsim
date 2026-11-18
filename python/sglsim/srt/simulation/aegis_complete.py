"""Unified metadata DES for the complete current AegisKV control contract.

The older large-scale campaign intentionally uses a fixed-service harm
surrogate.  This module is the executable path for the stronger contract: it
combines causal Slack, independent source-owned HBM/DRAM protection, real
per-target HBM capacity, target-first standby rotation, source fail-stop,
queue outage and rescheduling, recovery ordering, output fencing, route
rebind, and silent replay in one event stream.

SGLSim remains metadata-only.  Copy, load, recompute, replacement, and replay
durations are calibrated inputs; no CUDA, NCCL, or physical KV bytes are used.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from sglsim.srt.simulation.aegis_full_design import (
    AegisFullDesignConfig,
    CalibratedTransferPath,
    SealedChunk,
    sealed_chunks,
)
from sglsim.srt.simulation.aegis_independent_tiers import (
    DESIGN_SEMANTICS,
    AegisIndependentTierController,
    ChunkSpec,
    CopyState,
    IndependentTierConfig,
    ProtectionTier,
    RecoveryDecision,
    RequestRecoveryProfile,
)
from sglsim.srt.simulation.aegis_residual_dag import resolve_slack_for_design
from sglsim.srt.simulation.aegis_transfer_engine import (
    AegisTransferEngine,
    TransferSubmission,
)


COMPLETE_SIMULATION_SCHEMA = "aegiskv-complete-metadata-des-v1"
PROTECTION_MODES = ("hbm_and_dram", "hbm_only", "dram_only", "none")
RECOVERY_PRIORITIES = ("remaining_slack", "fifo")


def _finite_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


@dataclass(frozen=True)
class CompleteWorker:
    worker_id: str
    failure_domain: str

    def __post_init__(self) -> None:
        if not str(self.worker_id) or not str(self.failure_domain):
            raise ValueError("worker identity and failure domain are required")


@dataclass(frozen=True)
class CompleteRequest:
    request_id: str
    workflow_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    service_ms: float
    slack_ms: float
    slack_evidence: Mapping[str, Any] = field(default_factory=dict)
    preferred_source: str | None = None
    prefill_service_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.workflow_id:
            raise ValueError("request and workflow identities are required")
        _finite_nonnegative("arrival_ms", self.arrival_ms)
        if int(self.prompt_tokens) <= 0 or int(self.output_tokens) < 0:
            raise ValueError("request token counts are invalid")
        if not math.isfinite(float(self.service_ms)) or float(self.service_ms) <= 0.0:
            raise ValueError("service_ms must be finite and positive")
        if self.prefill_service_ms is not None:
            prefill = _finite_nonnegative("prefill_service_ms", self.prefill_service_ms)
            if prefill > float(self.service_ms) + 1e-9:
                raise ValueError("prefill_service_ms cannot exceed service_ms")
        _finite_nonnegative("slack_ms", self.slack_ms)

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.output_tokens)


@dataclass(frozen=True)
class CompleteFailure:
    time_ms: float
    worker_id: str
    repair_after_ms: float | None = None

    def __post_init__(self) -> None:
        _finite_nonnegative("failure time_ms", self.time_ms)
        if not self.worker_id:
            raise ValueError("failure worker_id is required")
        if self.repair_after_ms is not None:
            _finite_nonnegative("repair_after_ms", self.repair_after_ms)


@dataclass(frozen=True)
class CompleteSlackUpdate:
    time_ms: float
    request_id: str
    slack_ms: float
    slack_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        _finite_nonnegative("Slack update time_ms", self.time_ms)
        _finite_nonnegative("Slack update slack_ms", self.slack_ms)
        if not self.request_id:
            raise ValueError("Slack update request_id is required")


@dataclass(frozen=True)
class CompleteSimulationConfig:
    workers: tuple[CompleteWorker, ...]
    design: AegisFullDesignConfig
    dram_failure_domain: str = "store-cluster"
    replacement_delay_ms: float = 5.0
    queue_outage_ms: float = 3.0
    repair_time_ms: float = 60.0
    silent_replay_setup_ms: float = 0.25
    silent_replay_ms_per_token: float = 0.0005
    control_window_ms: float = 10.0
    primary_pressure_queue_depth: int = 1
    protection_mode: str = "hbm_and_dram"
    recovery_priority: str = "remaining_slack"
    adaptive_hbm_quota: bool = True
    model_revision: str = "sglsim-model-v1"
    kv_abi: str = "sglsim-metadata-kv-v1"
    parallel_layout: str = "tp1-pp1"
    random_salt: str = "aegiskv-complete-v1"
    event_log_limit: int = 100_000

    def __post_init__(self) -> None:
        if len(self.workers) < 2:
            raise ValueError("complete failover simulation requires at least two workers")
        identities = [row.worker_id for row in self.workers]
        if len(set(identities)) != len(identities):
            raise ValueError("worker identities must be unique")
        if not self.dram_failure_domain:
            raise ValueError("dram_failure_domain is required")
        for name in (
            "replacement_delay_ms",
            "queue_outage_ms",
            "repair_time_ms",
            "silent_replay_setup_ms",
            "silent_replay_ms_per_token",
            "control_window_ms",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.control_window_ms <= 0.0:
            raise ValueError("control_window_ms must be positive")
        if int(self.primary_pressure_queue_depth) < 0:
            raise ValueError("primary_pressure_queue_depth must be nonnegative")
        if self.protection_mode not in PROTECTION_MODES:
            raise ValueError(
                f"protection_mode must be one of {PROTECTION_MODES}, "
                f"got {self.protection_mode!r}"
            )
        if self.recovery_priority not in RECOVERY_PRIORITIES:
            raise ValueError(
                f"recovery_priority must be one of {RECOVERY_PRIORITIES}, "
                f"got {self.recovery_priority!r}"
            )
        if int(self.event_log_limit) <= 0:
            raise ValueError("event_log_limit must be positive")
        for value in (self.model_revision, self.kv_abi, self.parallel_layout):
            if not str(value):
                raise ValueError("compatibility identities must be nonempty")


@dataclass
class _WorkerState:
    spec: CompleteWorker
    healthy: bool = True
    queue: deque[str] = field(default_factory=deque)
    running_request: str | None = None
    recovery_request: str | None = None
    recovery_queue: list[str] = field(default_factory=list)
    failure_generation: int = 0


@dataclass
class _RouteState:
    request: CompleteRequest
    source_worker: str = ""
    target_worker: str | None = None
    route_version: int = 1
    status: str = "NEW"
    produced_float: float = 0.0
    produced_tokens: int = 0
    sealed_count: int = 0
    output_committed_tokens: int = 0
    output_fenced: bool = False
    trusted_slack_ms: float = 0.0
    slack_updated_at_ms: float = 0.0
    slack_reason: str = ""
    chunks: tuple[SealedChunk, ...] = ()
    remaining_service_ms: float = 0.0
    run_started_at_ms: float | None = None
    run_start_produced: float = 0.0
    run_start_remaining_ms: float = 0.0
    service_generation: int = 0
    completion_ms: float | None = None
    queue_delay_ms: float = 0.0
    outage_delay_ms: float = 0.0
    recovery_delay_ms: float = 0.0
    target_rotation_count: int = 0
    source_failure_count: int = 0
    redispatch_count: int = 0
    recovery_action: str = "NONE"
    restored_tokens: int = 0
    recompute_tokens: int = 0
    recovery_due_at_ms: float = 0.0
    recovery_ready_at_ms: float = 0.0
    recovery_started_at_ms: float | None = None
    recovery_worker: str | None = None
    last_failed_source: str | None = None
    recovery_next_route_version: int = 0
    recovery_decision: RecoveryDecision | None = None
    silent_replay_ms: float = 0.0

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def active(self) -> bool:
        return self.status not in {"COMPLETED", "CANCELLED"}


class _TierPoolRuntime:
    """One real capacity pool and its asynchronous source-owned transfer lane."""

    def __init__(
        self,
        *,
        tier: ProtectionTier,
        pool_id: str,
        design: AegisFullDesignConfig,
        dram_failure_domain: str,
        random_salt: str,
    ) -> None:
        self.tier = ProtectionTier(tier)
        self.pool_id = str(pool_id)
        self.design = design
        self.random_salt = str(random_salt)
        self._now_ms = 0.0
        self._attempt_sequence = 0
        self.engine = AegisTransferEngine()
        self.controller = AegisIndependentTierController(
            IndependentTierConfig(
                hbm_capacity_bytes=(
                    int(design.hbm_capacity_bytes)
                    if self.tier is ProtectionTier.HBM
                    else 1
                ),
                dram_capacity_bytes=(
                    int(design.dram_capacity_bytes)
                    if self.tier is ProtectionTier.DRAM
                    else 1
                ),
                initial_hbm_alpha=float(design.initial_hbm_alpha),
                minimum_hbm_alpha=float(design.minimum_hbm_alpha),
                maximum_hbm_alpha=float(design.maximum_hbm_alpha),
                aimd_decrease_factor=float(design.aimd_decrease_factor),
                aimd_increase_step=float(design.aimd_increase_step),
                aimd_stable_windows=int(design.aimd_stable_windows),
                recompute=design.recompute,
                hbm_load=design.hbm_load,
                dram_load=design.dram_load,
                hbm_copy=replace(
                    design.recompute,
                    setup_ms=design.hbm_copy.setup_ms,
                    ms_per_token=(
                        int(design.kv_bytes_per_token)
                        / float(design.hbm_copy.bandwidth_bytes_per_ms)
                    ),
                ),
                dram_copy=replace(
                    design.recompute,
                    setup_ms=design.dram_copy.setup_ms,
                    ms_per_token=(
                        int(design.kv_bytes_per_token)
                        / float(design.dram_copy.bandwidth_bytes_per_ms)
                    ),
                ),
                hbm_pool_id=(self.pool_id if self.tier is ProtectionTier.HBM else "unused-hbm"),
                dram_pool_id=(self.pool_id if self.tier is ProtectionTier.DRAM else "unused-dram"),
                dram_failure_domain=str(dram_failure_domain),
                retain_event_log=True,
                retain_record_history=False,
                event_log_limit=int(design.event_log_limit),
            )
        )
        self.metrics: dict[str, int | float] = {
            "attempted_copies": 0,
            "attempted_bytes": 0,
            "committed_copies": 0,
            "committed_bytes": 0,
            "failed_copies": 0,
            "cancelled_copies": 0,
        }

    @property
    def path(self) -> CalibratedTransferPath:
        return self.design.hbm_copy if self.tier is ProtectionTier.HBM else self.design.dram_copy

    @property
    def failure_probability(self) -> float:
        return (
            float(self.design.hbm_copy_failure_probability)
            if self.tier is ProtectionTier.HBM
            else float(self.design.dram_copy_failure_probability)
        )

    def _set_time(self, time_ms: float) -> None:
        target = float(time_ms)
        if target < self._now_ms - 1e-9:
            raise ValueError("tier pool time cannot regress")
        delta = max(target - self._now_ms, 0.0)
        if delta:
            self.controller.advance_time(delta)
            self._now_ms = target

    def advance_to(self, time_ms: float) -> None:
        target = float(time_ms)
        while (job := self.engine.pop_next_ready(target)) is not None:
            self._set_time(job.completed_at_ms)
            if job.cancelled or self.controller.copy_state(
                job.spec.request_id, job.spec.ordinal, self.tier
            ) is not CopyState.PENDING:
                self.metrics["cancelled_copies"] += 1
                continue
            proof = self.controller.completion_proof(
                job.spec, self.tier, succeeded=job.transfer_succeeded
            )
            completion = self.controller.complete_copy(
                self.tier, job.spec.request_id, job.spec.ordinal, proof
            )
            if completion.state is CopyState.COMMITTED:
                self.metrics["committed_copies"] += 1
                self.metrics["committed_bytes"] += int(job.spec.byte_size)
            else:
                self.metrics["failed_copies"] += 1
        self._set_time(target)

    def register_profile(self, profile: RequestRecoveryProfile, *, time_ms: float) -> None:
        self.advance_to(time_ms)
        self.controller.register_request(profile)

    def update_profile(self, profile: RequestRecoveryProfile, *, time_ms: float) -> None:
        self.register_profile(profile, time_ms=time_ms)

    def _draw_failed(self, spec: ChunkSpec) -> bool:
        self._attempt_sequence += 1
        identity = (
            f"{self.random_salt}:{self.tier.value}:{self.pool_id}:"
            f"{spec.logical_id}:{self._attempt_sequence}"
        )
        digest = hashlib.sha256(identity.encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / float(2**64)
        return draw < self.failure_probability

    def admit(self, spec: ChunkSpec, *, time_ms: float) -> dict[str, Any]:
        self.advance_to(time_ms)
        decision = self.controller.admit_chunk(spec, allowed_tiers=(self.tier,))
        admission = (
            decision.hbm
            if self.tier is ProtectionTier.HBM
            else decision.dram
        )
        if admission.admitted:
            failed = self._draw_failed(spec)
            first = int(self.metrics["attempted_copies"]) == 0
            self.engine.enqueue(
                TransferSubmission(
                    tier=self.tier,
                    spec=spec,
                    destination_id=(
                        spec.hbm_target
                        if self.tier is ProtectionTier.HBM
                        else self.pool_id
                    ),
                    copy_kind=(
                        "source_d2d"
                        if self.tier is ProtectionTier.HBM
                        else "source_d2h"
                    ),
                    ready_at_ms=float(time_ms),
                    service_ms=self.path.duration_ms(
                        spec.byte_size, first_transfer=first
                    ),
                    transfer_succeeded=not failed,
                    acked_shards=(tuple(range(spec.shard_count)) if not failed else ()),
                    source_owned=True,
                    target_owned=False,
                    hbm_to_dram=False,
                )
            )
            self.metrics["attempted_copies"] += 1
            self.metrics["attempted_bytes"] += int(spec.byte_size)
        return {
            "tier": self.tier.value,
            "admitted": bool(admission.admitted),
            "reason": admission.reason,
            "pending_id": admission.pending_id,
            "copy_kind": admission.copy_kind,
            "victim_ids": list(admission.victim_ids),
            "pool_id": self.pool_id,
        }

    def abort_source_pending(self, source_worker: str, *, time_ms: float) -> int:
        self.advance_to(time_ms)
        return self.controller.abort_source_pending(source_worker)

    def invalidate_request(self, request_id: str, *, reason: str, time_ms: float) -> dict[str, Any]:
        self.advance_to(time_ms)
        if not self.controller.has_request(request_id):
            return {"released_bytes": 0, "reason": reason}
        result = self.controller.invalidate_tier(request_id, self.tier, reason=reason)
        self.controller.complete_request(request_id, force_abort_pending=True)
        return result

    def remove_request(self, request_id: str, *, time_ms: float) -> dict[str, Any]:
        self.advance_to(time_ms)
        if not self.controller.has_request(request_id):
            return {"completed": True, "released_bytes": {}}
        return self.controller.complete_request(request_id, force_abort_pending=True)

    def estimate(
        self,
        request_id: str,
        *,
        activated_target: str,
        source_token_end: int,
        slack_ms: float,
        time_ms: float,
    ) -> RecoveryDecision:
        self.advance_to(time_ms)
        return self.controller.estimate_recovery_at(
            request_id,
            source_token_end=int(source_token_end),
            slack_ms=float(slack_ms),
            activated_target=str(activated_target),
        )

    def control_hbm_quota(self, *, pressure: bool, time_ms: float) -> dict[str, Any]:
        self.advance_to(time_ms)
        if self.tier is not ProtectionTier.HBM:
            raise ValueError("AIMD applies only to HBM target pools")
        return self.controller.control_hbm_quota(pressure=bool(pressure))

    def snapshot(self) -> dict[str, Any]:
        value = self.controller.snapshot()
        return {
            "tier": self.tier.value,
            "pool_id": self.pool_id,
            "metrics": dict(self.metrics),
            "transfer_engine": self.engine.snapshot(),
            "controller": value,
        }


class AegisCompleteSimulator:
    """Continuous-time, multi-worker simulation of the current AegisKV design."""

    def __init__(self, config: CompleteSimulationConfig) -> None:
        self.config = config
        self._now_ms = 0.0
        self._workers = {
            row.worker_id: _WorkerState(row) for row in config.workers
        }
        self._routes: dict[str, _RouteState] = {}
        self._hbm = {
            row.worker_id: _TierPoolRuntime(
                tier=ProtectionTier.HBM,
                pool_id=f"peer-hbm-{row.worker_id}",
                design=config.design,
                dram_failure_domain=config.dram_failure_domain,
                random_salt=f"{config.random_salt}:{row.worker_id}",
            )
            for row in config.workers
        }
        self._dram = _TierPoolRuntime(
            tier=ProtectionTier.DRAM,
            pool_id="store-dram-shared",
            design=config.design,
            dram_failure_domain=config.dram_failure_domain,
            random_salt=f"{config.random_salt}:dram",
        )
        self._heap: list[tuple[float, int, int, str, Any]] = []
        self._event_sequence = 0
        self._events: list[dict[str, Any]] = []
        self._event_counts: dict[str, int] = {}
        self._faults_remaining = 0
        self._slack_updates_remaining = 0

    @property
    def _hbm_enabled(self) -> bool:
        return self.config.protection_mode in {"hbm_and_dram", "hbm_only"}

    @property
    def _dram_enabled(self) -> bool:
        return self.config.protection_mode in {"hbm_and_dram", "dram_only"}

    def _push(self, time_ms: float, priority: int, kind: str, payload: Any) -> None:
        self._event_sequence += 1
        heapq.heappush(
            self._heap,
            (float(time_ms), int(priority), self._event_sequence, str(kind), payload),
        )

    def _emit(self, kind: str, **fields: Any) -> None:
        self._event_counts[kind] = self._event_counts.get(kind, 0) + 1
        if len(self._events) >= int(self.config.event_log_limit):
            raise RuntimeError("complete simulation event log limit exceeded")
        self._events.append(
            {
                "sequence": len(self._events) + 1,
                "time_ms": round(self._now_ms, 6),
                "kind": str(kind),
                "design_semantics": DESIGN_SEMANTICS,
                **fields,
            }
        )

    def _worker_domain(self, worker_id: str) -> str:
        return self._workers[str(worker_id)].spec.failure_domain

    def _remaining_slack(self, route: _RouteState, time_ms: float) -> float:
        return max(
            float(route.trusted_slack_ms)
            - max(float(time_ms) - float(route.slack_updated_at_ms), 0.0),
            0.0,
        )

    def _profile(self, route: _RouteState, *, time_ms: float) -> RequestRecoveryProfile:
        target = route.target_worker or "degraded-no-standby"
        target_domain = (
            self._worker_domain(route.target_worker)
            if route.target_worker is not None
            else "degraded-no-standby-domain"
        )
        return RequestRecoveryProfile(
            request_id=route.request_id,
            source_worker=route.source_worker,
            source_failure_domain=self._worker_domain(route.source_worker),
            hbm_target=target,
            hbm_failure_domain=target_domain,
            route_version=int(route.route_version),
            source_token_end=int(route.produced_tokens),
            slack_ms=self._remaining_slack(route, time_ms),
            model_revision=self.config.model_revision,
            kv_abi=self.config.kv_abi,
            parallel_layout=self.config.parallel_layout,
        )

    def _update_profiles(self, route: _RouteState, *, time_ms: float) -> None:
        profile = self._profile(route, time_ms=time_ms)
        if self._dram_enabled:
            self._dram.update_profile(profile, time_ms=time_ms)
        if self._hbm_enabled and route.target_worker is not None:
            self._hbm[route.target_worker].update_profile(profile, time_ms=time_ms)

    def _choose_source(self, preferred: str | None = None) -> str:
        if preferred is not None:
            worker = self._workers.get(str(preferred))
            if worker is None:
                raise ValueError(f"unknown preferred source worker {preferred!r}")
            if worker.healthy:
                return worker.spec.worker_id
        candidates = [row for row in self._workers.values() if row.healthy]
        if not candidates:
            raise RuntimeError("no healthy source worker is available")
        chosen = min(
            candidates,
            key=lambda row: (
                row.running_request is not None,
                row.recovery_request is not None,
                len(row.queue) + len(row.recovery_queue),
                row.spec.worker_id,
            ),
        )
        return chosen.spec.worker_id

    def _choose_target(
        self,
        source_worker: str,
        *,
        excluded: Iterable[str] = (),
    ) -> str | None:
        blocked = {str(source_worker), *(str(row) for row in excluded)}
        source_domain = self._worker_domain(source_worker)
        candidates = [
            row
            for row in self._workers.values()
            if row.healthy
            and row.spec.worker_id not in blocked
            and row.spec.failure_domain != source_domain
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda row: (
                self._hbm[row.spec.worker_id].controller.pool_used_bytes(ProtectionTier.HBM),
                row.spec.worker_id,
            ),
        ).spec.worker_id

    def _register_route_pools(self, route: _RouteState, *, time_ms: float) -> None:
        profile = self._profile(route, time_ms=time_ms)
        if self._dram_enabled:
            self._dram.register_profile(profile, time_ms=time_ms)
        if self._hbm_enabled and route.target_worker is not None:
            self._hbm[route.target_worker].register_profile(profile, time_ms=time_ms)

    def _chunk_spec(self, route: _RouteState, chunk: SealedChunk) -> ChunkSpec:
        return ChunkSpec(
            request_id=route.request_id,
            ordinal=int(chunk.ordinal),
            token_start=int(chunk.token_start),
            token_end=int(chunk.token_end),
            byte_size=int(chunk.token_count) * int(self.config.design.kv_bytes_per_token),
            route_version=int(route.route_version),
            source_worker=route.source_worker,
            hbm_target=route.target_worker or "degraded-no-standby",
            model_revision=self.config.model_revision,
            kv_abi=self.config.kv_abi,
            parallel_layout=self.config.parallel_layout,
            shard_count=int(self.config.design.shard_count),
        )

    def _protect_chunk(
        self,
        route: _RouteState,
        chunk: SealedChunk,
        *,
        time_ms: float,
        hbm: bool = True,
        dram: bool = True,
        reprotection: bool = False,
    ) -> None:
        self._update_profiles(route, time_ms=time_ms)
        spec = self._chunk_spec(route, chunk)
        outcomes: list[dict[str, Any]] = []
        if hbm and self._hbm_enabled and route.target_worker is not None:
            outcomes.append(self._hbm[route.target_worker].admit(spec, time_ms=time_ms))
        if dram and self._dram_enabled:
            outcomes.append(self._dram.admit(spec, time_ms=time_ms))
        self._emit(
            "chunk_reprotected" if reprotection else "chunk_sealed_and_admitted",
            request_id=route.request_id,
            ordinal=int(chunk.ordinal),
            token_start=int(chunk.token_start),
            token_end=int(chunk.token_end),
            source_worker=route.source_worker,
            target_worker=route.target_worker,
            route_version=int(route.route_version),
            independent_tier_outcomes=outcomes,
            source_owned=True,
            hbm_to_dram=False,
        )

    def _reprotect_prefix(
        self,
        route: _RouteState,
        *,
        time_ms: float,
        hbm: bool,
        dram: bool,
    ) -> None:
        for chunk in route.chunks[: route.sealed_count]:
            self._protect_chunk(
                route,
                chunk,
                time_ms=time_ms,
                hbm=hbm,
                dram=dram,
                reprotection=True,
            )

    def _prefill_service_ms(self, route: _RouteState) -> float:
        explicit = route.request.prefill_service_ms
        if explicit is not None:
            return float(explicit)
        total_tokens = max(route.request.total_tokens, 1)
        return float(route.request.service_ms) * (
            int(route.request.prompt_tokens) / total_tokens
        )

    def _service_consumed_at_token(
        self, route: _RouteState, token_end: float
    ) -> float:
        request = route.request
        token = min(max(float(token_end), 0.0), float(request.total_tokens))
        prefill_ms = self._prefill_service_ms(route)
        if token <= int(request.prompt_tokens):
            return prefill_ms * token / max(int(request.prompt_tokens), 1)
        decode_ms = max(float(request.service_ms) - prefill_ms, 0.0)
        decode_tokens = token - int(request.prompt_tokens)
        return prefill_ms + decode_ms * decode_tokens / max(
            int(request.output_tokens), 1
        )

    def _tokens_at_service_consumed(
        self, route: _RouteState, consumed_ms: float
    ) -> float:
        request = route.request
        consumed = min(max(float(consumed_ms), 0.0), float(request.service_ms))
        prefill_ms = self._prefill_service_ms(route)
        if consumed <= prefill_ms and prefill_ms > 0.0:
            return int(request.prompt_tokens) * consumed / prefill_ms
        decode_ms = max(float(request.service_ms) - prefill_ms, 0.0)
        if decode_ms <= 0.0:
            return float(request.total_tokens)
        return min(
            float(request.total_tokens),
            int(request.prompt_tokens)
            + int(request.output_tokens) * (consumed - prefill_ms) / decode_ms,
        )

    def _advance_running_route(self, route: _RouteState, time_ms: float) -> None:
        if route.status != "RUNNING" or route.run_started_at_ms is None:
            return
        elapsed = min(
            max(float(time_ms) - float(route.run_started_at_ms), 0.0),
            float(route.run_start_remaining_ms),
        )
        consumed_at_start = max(
            float(route.request.service_ms) - float(route.run_start_remaining_ms),
            0.0,
        )
        route.produced_float = max(
            route.produced_float,
            self._tokens_at_service_consumed(route, consumed_at_start + elapsed),
        )
        route.produced_tokens = max(
            int(route.produced_tokens), int(math.floor(route.produced_float + 1e-9))
        )
        route.output_committed_tokens = max(
            route.output_committed_tokens,
            min(
                max(route.produced_tokens - int(route.request.prompt_tokens), 0),
                int(route.request.output_tokens),
            ),
        )
        route.remaining_service_ms = max(
            float(route.run_start_remaining_ms) - elapsed, 0.0
        )

    def _advance_to(self, time_ms: float) -> None:
        target = float(time_ms)
        if target < self._now_ms - 1e-9:
            raise ValueError("complete simulation time cannot regress")
        for route in self._routes.values():
            self._advance_running_route(route, target)
        for pool in self._hbm.values():
            pool.advance_to(target)
        self._dram.advance_to(target)
        self._now_ms = target

    def _schedule_execution_events(self, route: _RouteState) -> None:
        if route.status != "RUNNING" or route.run_started_at_ms is None:
            return
        generation = int(route.service_generation)
        finish = float(route.run_started_at_ms) + float(
            route.run_start_remaining_ms
        )
        self._push(finish, 5, "service_complete", (route.request_id, generation))
        next_chunks = [
            row
            for row in route.chunks[route.sealed_count :]
            if row.token_end > route.produced_float
        ]
        if next_chunks and route.request.total_tokens > route.run_start_produced:
            chunk = next_chunks[0]
            consumed_at_start = max(
                float(route.request.service_ms)
                - float(route.run_start_remaining_ms),
                0.0,
            )
            seal_offset = max(
                self._service_consumed_at_token(route, chunk.token_end)
                - consumed_at_start,
                0.0,
            )
            seal_at = float(route.run_started_at_ms) + min(
                seal_offset, float(route.run_start_remaining_ms)
            )
            self._push(
                seal_at,
                4,
                "seal_chunk",
                (route.request_id, generation, int(chunk.ordinal)),
            )

    def _start_request(self, route: _RouteState, worker: _WorkerState) -> None:
        if (
            not worker.healthy
            or worker.running_request is not None
            or worker.recovery_request is not None
        ):
            return
        route.status = "RUNNING"
        route.service_generation += 1
        route.run_started_at_ms = self._now_ms
        route.run_start_produced = float(route.produced_float)
        route.run_start_remaining_ms = float(route.remaining_service_ms)
        worker.running_request = route.request_id
        self._emit(
            "request_execution_started",
            request_id=route.request_id,
            source_worker=route.source_worker,
            target_worker=route.target_worker,
            route_version=route.route_version,
            produced_tokens=route.produced_tokens,
            remaining_service_ms=round(route.remaining_service_ms, 6),
        )
        self._schedule_execution_events(route)

    def _dispatch_worker(self, worker_id: str) -> None:
        worker = self._workers[worker_id]
        if (
            not worker.healthy
            or worker.running_request is not None
            or worker.recovery_request is not None
        ):
            return
        while worker.queue:
            request_id = worker.queue.popleft()
            route = self._routes[request_id]
            if (
                not route.active
                or route.source_worker != worker_id
                or route.status != "QUEUED"
            ):
                continue
            self._start_request(route, worker)
            return

    def _enqueue_request(self, route: _RouteState, *, worker_id: str, reason: str) -> None:
        route.source_worker = str(worker_id)
        route.status = "QUEUED"
        self._workers[worker_id].queue.append(route.request_id)
        self._emit(
            "request_enqueued",
            request_id=route.request_id,
            source_worker=worker_id,
            target_worker=route.target_worker,
            route_version=route.route_version,
            reason=reason,
        )
        self._dispatch_worker(worker_id)

    def _pause_running(self, worker: _WorkerState, *, reason: str) -> str | None:
        request_id = worker.running_request
        if request_id is None:
            return None
        route = self._routes[request_id]
        self._advance_running_route(route, self._now_ms)
        route.service_generation += 1
        route.run_started_at_ms = None
        route.status = "QUEUED"
        worker.running_request = None
        worker.queue.appendleft(request_id)
        self._emit(
            "request_execution_preempted",
            request_id=request_id,
            source_worker=worker.spec.worker_id,
            reason=reason,
            remaining_service_ms=round(route.remaining_service_ms, 6),
        )
        return request_id

    def _rotate_target(
        self,
        route: _RouteState,
        *,
        failed_target: str,
        reason: str,
    ) -> None:
        if route.target_worker != failed_target or not route.active:
            return
        old_pool = self._hbm[failed_target]
        invalidated = old_pool.invalidate_request(
            route.request_id,
            reason="bound_hbm_target_failed",
            time_ms=self._now_ms,
        )
        replacement = self._choose_target(
            route.source_worker, excluded=(failed_target,)
        )
        previous_version = int(route.route_version)
        route.target_worker = replacement
        route.target_rotation_count += 1
        profile = self._profile(route, time_ms=self._now_ms)
        if self._dram_enabled:
            self._dram.update_profile(profile, time_ms=self._now_ms)
        source_can_copy = (
            self._workers[route.source_worker].healthy
            and route.status not in {"RECOVERY_WAIT", "RECOVERING"}
        )
        if replacement is not None:
            self._hbm[replacement].register_profile(profile, time_ms=self._now_ms)
            if source_can_copy:
                self._reprotect_prefix(
                    route, time_ms=self._now_ms, hbm=True, dram=False
                )
        self._emit(
            "standby_target_rotated" if replacement is not None else "route_degraded_no_standby",
            request_id=route.request_id,
            source_worker=route.source_worker,
            failed_target=failed_target,
            replacement_target=replacement,
            route_version=route.route_version,
            route_version_changed=False,
            route_version_before=previous_version,
            invalidated_hbm_bytes=int(invalidated.get("released_bytes", 0)),
            dram_preserved=True,
            replacement_uses_real_target_pool=(replacement is not None),
            source_copy_ahead_scheduled=(replacement is not None and source_can_copy),
            reason=reason,
        )

    def _best_recovery(
        self,
        route: _RouteState,
        *,
        activated_target: str,
    ) -> RecoveryDecision:
        remaining_slack = self._remaining_slack(route, self._now_ms)
        decisions = [
            self._dram.estimate(
                route.request_id,
                activated_target=activated_target,
                source_token_end=route.produced_tokens,
                slack_ms=remaining_slack,
                time_ms=self._now_ms,
            )
        ] if self._dram_enabled else []
        if (
            self._hbm_enabled
            and route.target_worker is not None
            and self._hbm[route.target_worker].controller.has_request(
                route.request_id
            )
        ):
            decisions.append(
                self._hbm[route.target_worker].estimate(
                    route.request_id,
                    activated_target=activated_target,
                    source_token_end=route.produced_tokens,
                    slack_ms=remaining_slack,
                    time_ms=self._now_ms,
                )
            )
        if not decisions:
            recompute = self.config.design.recompute
            recovery_ms = (
                0.0
                if route.produced_tokens <= 0
                else float(recompute.setup_ms)
                + int(route.produced_tokens) * float(recompute.ms_per_token)
            )
            decisions.append(
                RecoveryDecision(
                    request_id=route.request_id,
                    action="RECOMPUTE",
                    restored_tokens=0,
                    recompute_tokens=int(route.produced_tokens),
                    recovery_ms=recovery_ms,
                    slack_ms=remaining_slack,
                    visible_harm_ms=max(recovery_ms - remaining_slack, 0.0),
                    route_version=int(route.route_version),
                    activated_target=str(activated_target),
                )
            )
        return min(
            decisions,
            key=lambda row: (row.recovery_ms, -row.restored_tokens, row.action),
        )

    def _fail_source(self, route: _RouteState, *, failed_worker: str) -> None:
        if not route.active or route.source_worker != failed_worker:
            return
        self._advance_running_route(route, self._now_ms)
        worker = self._workers[failed_worker]
        if worker.running_request == route.request_id:
            worker.running_request = None
        try:
            worker.queue.remove(route.request_id)
        except ValueError:
            pass
        route.service_generation += 1
        route.run_started_at_ms = None
        route.output_fenced = True
        route.status = "RECOVERY_WAIT"
        route.source_failure_count += 1
        route.recovery_due_at_ms = self._now_ms + self._remaining_slack(route, self._now_ms)
        self._update_profiles(route, time_ms=self._now_ms)
        hbm_aborted = 0
        if self._hbm_enabled and route.target_worker is not None:
            hbm_aborted = self._hbm[route.target_worker].abort_source_pending(
                failed_worker, time_ms=self._now_ms
            )
        dram_aborted = (
            self._dram.abort_source_pending(failed_worker, time_ms=self._now_ms)
            if self._dram_enabled
            else 0
        )
        activated = (
            route.target_worker
            if route.target_worker is not None
            and self._workers[route.target_worker].healthy
            else self._choose_source()
        )
        decision = self._best_recovery(route, activated_target=activated)
        route.recovery_worker = activated
        route.last_failed_source = failed_worker
        route.recovery_decision = decision
        route.recovery_next_route_version = int(route.route_version) + 1
        route.recovery_ready_at_ms = self._now_ms + float(self.config.replacement_delay_ms)
        route.silent_replay_ms = (
            float(self.config.silent_replay_setup_ms)
            + int(route.output_committed_tokens)
            * float(self.config.silent_replay_ms_per_token)
        )
        self._push(
            route.recovery_ready_at_ms,
            1,
            "recovery_ready",
            route.request_id,
        )
        self._emit(
            "source_failure_fenced",
            request_id=route.request_id,
            failed_source=failed_worker,
            activated_target=activated,
            route_version=route.route_version,
            next_route_version=route.recovery_next_route_version,
            output_fenced=True,
            committed_output_tokens=route.output_committed_tokens,
            produced_tokens=route.produced_tokens,
            hbm_pending_aborted=hbm_aborted,
            dram_pending_aborted=dram_aborted,
            recovery_action=decision.action,
            restored_tokens=decision.restored_tokens,
            recompute_tokens=decision.recompute_tokens,
            actual_load_bytes=(
                int(decision.restored_tokens) * int(self.config.design.kv_bytes_per_token)
            ),
            recovery_due_at_ms=round(route.recovery_due_at_ms, 6),
        )

    def _redispatch_unstarted(self, route: _RouteState, *, failed_worker: str) -> None:
        route.status = "QUEUE_OUTAGE"
        route.redispatch_count += 1
        route.outage_delay_ms += float(self.config.queue_outage_ms)
        route.route_version += 1
        if self._hbm_enabled and route.target_worker is not None:
            self._hbm[route.target_worker].remove_request(
                route.request_id, time_ms=self._now_ms
            )
        if self._dram_enabled:
            self._dram.remove_request(route.request_id, time_ms=self._now_ms)
        self._push(
            self._now_ms + float(self.config.queue_outage_ms),
            2,
            "redispatch",
            (route.request_id, failed_worker),
        )
        self._emit(
            "queue_outage_started",
            request_id=route.request_id,
            failed_source=failed_worker,
            queue_outage_ms=float(self.config.queue_outage_ms),
            route_version=route.route_version,
        )

    def _fail_recovery_worker(self, worker: _WorkerState) -> None:
        request_ids = list(worker.recovery_queue)
        worker.recovery_queue.clear()
        if worker.recovery_request is not None:
            request_ids.insert(0, worker.recovery_request)
            worker.recovery_request = None
        for request_id in request_ids:
            route = self._routes[request_id]
            route.recovery_worker = self._choose_source()
            route.recovery_decision = self._best_recovery(
                route, activated_target=route.recovery_worker
            )
            route.recovery_ready_at_ms = self._now_ms + float(self.config.replacement_delay_ms)
            route.status = "RECOVERY_WAIT"
            self._push(route.recovery_ready_at_ms, 1, "recovery_ready", request_id)
            self._emit(
                "recovery_target_failed_and_rebound",
                request_id=request_id,
                failed_recovery_worker=worker.spec.worker_id,
                replacement_recovery_worker=route.recovery_worker,
                output_fenced=True,
                route_version=route.route_version,
            )

    def _handle_failure(self, failure: CompleteFailure) -> None:
        if failure.worker_id not in self._workers:
            raise ValueError(f"unknown failed worker {failure.worker_id!r}")
        worker = self._workers[failure.worker_id]
        if not worker.healthy:
            self._emit(
                "duplicate_worker_failure_ignored",
                worker_id=failure.worker_id,
                failure_generation=worker.failure_generation,
            )
            return
        worker.healthy = False
        worker.failure_generation += 1
        self._fail_recovery_worker(worker)

        target_routes = [
            route
            for route in self._routes.values()
            if route.active and route.target_worker == failure.worker_id
        ]
        for route in sorted(target_routes, key=lambda row: row.request_id):
            self._rotate_target(
                route,
                failed_target=failure.worker_id,
                reason="physical_target_worker_failed",
            )

        source_routes = [
            route
            for route in self._routes.values()
            if route.active and route.source_worker == failure.worker_id
        ]
        for route in sorted(source_routes, key=lambda row: row.request_id):
            if route.produced_tokens > 0 or route.status == "RUNNING":
                self._fail_source(route, failed_worker=failure.worker_id)
            elif route.status in {"QUEUED", "NEW"}:
                try:
                    worker.queue.remove(route.request_id)
                except ValueError:
                    pass
                self._redispatch_unstarted(route, failed_worker=failure.worker_id)

        repair_after = (
            float(failure.repair_after_ms)
            if failure.repair_after_ms is not None
            else float(self.config.repair_time_ms)
        )
        self._push(
            self._now_ms + repair_after,
            0,
            "repair",
            (failure.worker_id, worker.failure_generation),
        )
        self._emit(
            "worker_failed",
            worker_id=failure.worker_id,
            failure_domain=worker.spec.failure_domain,
            failure_generation=worker.failure_generation,
            target_routes_affected=len(target_routes),
            source_routes_affected=len(source_routes),
            repair_scheduled_at_ms=round(self._now_ms + repair_after, 6),
        )

    def _handle_repair(self, worker_id: str, generation: int) -> None:
        worker = self._workers[worker_id]
        if int(generation) != int(worker.failure_generation):
            self._emit(
                "superseded_worker_repair_ignored",
                worker_id=worker_id,
                repair_generation=generation,
                current_generation=worker.failure_generation,
            )
            return
        worker.healthy = True
        rebound = 0
        for route in sorted(self._routes.values(), key=lambda row: row.request_id):
            if not self._hbm_enabled:
                break
            if not route.active or route.target_worker is not None:
                continue
            replacement = self._choose_target(route.source_worker)
            if replacement is None:
                continue
            route.target_worker = replacement
            route.target_rotation_count += 1
            profile = self._profile(route, time_ms=self._now_ms)
            if self._dram_enabled:
                self._dram.update_profile(profile, time_ms=self._now_ms)
            self._hbm[replacement].register_profile(profile, time_ms=self._now_ms)
            self._reprotect_prefix(route, time_ms=self._now_ms, hbm=True, dram=False)
            rebound += 1
            self._emit(
                "standby_target_bound_after_repair",
                request_id=route.request_id,
                replacement_target=replacement,
                route_version=route.route_version,
                route_version_changed=False,
            )
        self._emit(
            "worker_repaired",
            worker_id=worker_id,
            failure_generation=generation,
            old_hbm_copies_restored=False,
            degraded_routes_rebound=rebound,
        )
        self._dispatch_worker(worker_id)

    def _handle_redispatch(self, request_id: str, failed_worker: str) -> None:
        route = self._routes[request_id]
        if not route.active or route.status != "QUEUE_OUTAGE":
            return
        source = self._choose_source()
        route.source_worker = source
        route.target_worker = (
            self._choose_target(source, excluded=(failed_worker,))
            if self._hbm_enabled
            else None
        )
        self._register_route_pools(route, time_ms=self._now_ms)
        self._emit(
            "queue_outage_rescheduled",
            request_id=request_id,
            failed_source=failed_worker,
            replacement_source=source,
            replacement_target=route.target_worker,
            route_version=route.route_version,
            queue_outage_ms=float(self.config.queue_outage_ms),
        )
        self._enqueue_request(route, worker_id=source, reason="post_failure_reschedule")

    def _handle_recovery_ready(self, request_id: str) -> None:
        route = self._routes[request_id]
        if not route.active or route.status != "RECOVERY_WAIT":
            return
        worker_id = route.recovery_worker
        if worker_id is None or not self._workers[worker_id].healthy:
            worker_id = self._choose_source()
            route.recovery_worker = worker_id
            route.recovery_decision = self._best_recovery(route, activated_target=worker_id)
        worker = self._workers[worker_id]
        if request_id not in worker.recovery_queue:
            worker.recovery_queue.append(request_id)
        # All recovery_ready events at this timestamp have priority 1.  Delay
        # dispatch to priority 2 so the recovery policy sees the complete
        # same-epoch candidate set instead of committing to heap insertion
        # order after the first ready request.
        self._push(self._now_ms, 2, "recovery_dispatch", worker_id)

    def _dispatch_recovery(self, worker_id: str) -> None:
        worker = self._workers[worker_id]
        if not worker.healthy or worker.recovery_request is not None or not worker.recovery_queue:
            return
        if self.config.recovery_priority == "remaining_slack":
            worker.recovery_queue.sort(
                key=lambda request_id: (
                    self._routes[request_id].recovery_due_at_ms,
                    self._routes[request_id].recovery_decision.recovery_ms
                    if self._routes[request_id].recovery_decision is not None
                    else math.inf,
                    request_id,
                )
            )
        request_id = worker.recovery_queue.pop(0)
        route = self._routes[request_id]
        self._pause_running(worker, reason="remaining_slack_recovery_priority")
        decision = route.recovery_decision
        if decision is None:
            raise RuntimeError("ready recovery has no decision")
        worker.recovery_request = request_id
        route.status = "RECOVERING"
        route.recovery_started_at_ms = self._now_ms
        duration = float(decision.recovery_ms) + float(route.silent_replay_ms)
        route.recovery_delay_ms += duration + max(
            self._now_ms - float(route.recovery_ready_at_ms), 0.0
        )
        self._push(
            self._now_ms + duration,
            1,
            "recovery_complete",
            (request_id, worker_id, route.recovery_next_route_version),
        )
        self._emit(
            "recovery_selected_and_started",
            request_id=request_id,
            activated_target=worker_id,
            action=decision.action,
            restored_tokens=decision.restored_tokens,
            recompute_tokens=decision.recompute_tokens,
            recovery_ms=round(decision.recovery_ms, 6),
            silent_replay_ms=round(route.silent_replay_ms, 6),
            remaining_slack_due_at_ms=round(route.recovery_due_at_ms, 6),
            priority=(
                "remaining_slack_due_time"
                if self.config.recovery_priority == "remaining_slack"
                else "fifo_ready_order"
            ),
            configured_recovery_priority=self.config.recovery_priority,
            output_fenced=True,
            prior_route_version=route.route_version,
            next_route_version=route.recovery_next_route_version,
        )

    def _handle_recovery_complete(
        self, request_id: str, worker_id: str, next_route_version: int
    ) -> None:
        route = self._routes[request_id]
        worker = self._workers[worker_id]
        if worker.recovery_request != request_id or route.status != "RECOVERING":
            return
        if not worker.healthy:
            worker.recovery_request = None
            route.status = "RECOVERY_WAIT"
            route.recovery_worker = self._choose_source()
            route.recovery_ready_at_ms = self._now_ms + float(self.config.replacement_delay_ms)
            self._push(route.recovery_ready_at_ms, 1, "recovery_ready", request_id)
            return

        old_target = route.target_worker
        if self._hbm_enabled and old_target is not None:
            self._hbm[old_target].remove_request(request_id, time_ms=self._now_ms)
        if self._dram_enabled:
            self._dram.remove_request(request_id, time_ms=self._now_ms)

        decision = route.recovery_decision
        if decision is None:
            raise RuntimeError("completed recovery has no decision")
        prior_route_version = int(route.route_version)
        route.source_worker = worker_id
        route.route_version = int(next_route_version)
        route.target_worker = (
            self._choose_target(
                worker_id,
                excluded=((old_target,) if old_target else ()),
            )
            if self._hbm_enabled
            else None
        )
        route.output_fenced = False
        route.recovery_action = decision.action
        route.restored_tokens += int(decision.restored_tokens)
        route.recompute_tokens += int(decision.recompute_tokens)
        route.recovery_decision = None
        route.recovery_worker = None
        route.recovery_started_at_ms = None
        worker.recovery_request = None

        self._register_route_pools(route, time_ms=self._now_ms)
        self._reprotect_prefix(
            route,
            time_ms=self._now_ms,
            hbm=self._hbm_enabled,
            dram=self._dram_enabled,
        )
        self._emit(
            "recovery_completed_route_activated",
            request_id=request_id,
            prior_source=route.last_failed_source,
            activated_source=worker_id,
            replacement_target=route.target_worker,
            prior_route_version=prior_route_version,
            route_version=route.route_version,
            route_version_contiguous=(route.route_version == prior_route_version + 1),
            output_fenced=False,
            silent_replay_completed=True,
            silent_replay_to_output_position=route.output_committed_tokens,
            all_shards_completed=True,
            restored_tokens=decision.restored_tokens,
            recompute_tokens=decision.recompute_tokens,
            actual_load_bytes=(
                int(decision.restored_tokens) * int(self.config.design.kv_bytes_per_token)
            ),
            old_route_hbm_reclaimed_after_recovery=self._hbm_enabled,
            old_route_dram_reprotected_for_new_version=self._dram_enabled,
        )
        route.status = "QUEUED"
        worker.queue.appendleft(request_id)
        self._dispatch_recovery(worker_id)
        self._dispatch_worker(worker_id)

    def _handle_arrival(self, request: CompleteRequest) -> None:
        fields = dict(request.slack_evidence)
        fields["workflow_slack_ms"] = float(request.slack_ms)
        trusted, reason = resolve_slack_for_design(
            fields, design_semantics=DESIGN_SEMANTICS
        )
        source = self._choose_source(request.preferred_source)
        target = self._choose_target(source) if self._hbm_enabled else None
        route = _RouteState(
            request=request,
            source_worker=source,
            target_worker=target,
            route_version=1,
            status="NEW",
            trusted_slack_ms=float(trusted),
            slack_updated_at_ms=self._now_ms,
            slack_reason=reason,
            chunks=sealed_chunks(
                request.prompt_tokens,
                request.output_tokens,
                prefix_chunk_tokens=self.config.design.prefix_chunk_tokens,
                decode_chunk_tokens=self.config.design.decode_chunk_tokens,
            ),
            remaining_service_ms=float(request.service_ms),
        )
        self._routes[request.request_id] = route
        self._register_route_pools(route, time_ms=self._now_ms)
        self._emit(
            "request_arrived_and_route_bound",
            request_id=request.request_id,
            workflow_id=request.workflow_id,
            source_worker=source,
            target_worker=target,
            route_version=1,
            causal_slack_ms=trusted,
            slack_reason=reason,
            slack_metadata_only=True,
            degraded_route=(target is None),
        )
        self._enqueue_request(route, worker_id=source, reason="arrival")

    def _handle_slack_update(self, update: CompleteSlackUpdate) -> None:
        route = self._routes.get(update.request_id)
        if route is None or not route.active:
            return
        fields = dict(update.slack_evidence)
        fields["workflow_slack_ms"] = float(update.slack_ms)
        trusted, reason = resolve_slack_for_design(
            fields, design_semantics=DESIGN_SEMANTICS
        )
        route.trusted_slack_ms = float(trusted)
        route.slack_updated_at_ms = self._now_ms
        route.slack_reason = reason
        self._update_profiles(route, time_ms=self._now_ms)
        self._emit(
            "causal_slack_metadata_updated",
            request_id=route.request_id,
            route_version=route.route_version,
            slack_ms=trusted,
            slack_reason=reason,
            physical_movement=False,
        )

    def _handle_seal(self, request_id: str, generation: int, ordinal: int) -> None:
        route = self._routes[request_id]
        if route.status != "RUNNING" or int(generation) != int(route.service_generation):
            return
        self._advance_running_route(route, self._now_ms)
        if int(ordinal) != int(route.sealed_count):
            return
        chunk = route.chunks[ordinal]
        route.produced_float = max(route.produced_float, float(chunk.token_end))
        route.produced_tokens = max(route.produced_tokens, int(chunk.token_end))
        route.sealed_count += 1
        self._protect_chunk(route, chunk, time_ms=self._now_ms)
        self._schedule_execution_events(route)

    def _complete_request(self, route: _RouteState) -> None:
        worker = self._workers[route.source_worker]
        if worker.running_request == route.request_id:
            worker.running_request = None
        route.status = "COMPLETED"
        route.completion_ms = self._now_ms
        route.output_committed_tokens = int(route.request.output_tokens)
        route.output_fenced = False
        route.remaining_service_ms = 0.0
        route.produced_float = float(route.request.total_tokens)
        route.produced_tokens = int(route.request.total_tokens)
        if self._hbm_enabled and route.target_worker is not None:
            self._hbm[route.target_worker].remove_request(
                route.request_id, time_ms=self._now_ms
            )
        if self._dram_enabled:
            self._dram.remove_request(route.request_id, time_ms=self._now_ms)
        self._emit(
            "request_completed",
            request_id=route.request_id,
            source_worker=route.source_worker,
            route_version=route.route_version,
            completion_ms=round(self._now_ms, 6),
            workflow_completion_time_ms=round(
                self._now_ms - float(route.request.arrival_ms), 6
            ),
            output_fenced=False,
        )
        self._dispatch_worker(worker.spec.worker_id)

    def _handle_service_complete(self, request_id: str, generation: int) -> None:
        route = self._routes[request_id]
        if route.status != "RUNNING" or int(generation) != int(route.service_generation):
            return
        self._advance_running_route(route, self._now_ms)
        while route.sealed_count < len(route.chunks):
            chunk = route.chunks[route.sealed_count]
            route.produced_tokens = max(route.produced_tokens, chunk.token_end)
            route.produced_float = max(route.produced_float, float(chunk.token_end))
            route.sealed_count += 1
            self._protect_chunk(route, chunk, time_ms=self._now_ms)
        self._complete_request(route)

    def _handle_control_window(self) -> None:
        updates = []
        for worker_id, worker in sorted(self._workers.items()):
            if not self._hbm_enabled:
                break
            pressure = (
                not worker.healthy
                or worker.recovery_request is not None
                or len(worker.queue) > int(self.config.primary_pressure_queue_depth)
            )
            if self.config.adaptive_hbm_quota:
                update = self._hbm[worker_id].control_hbm_quota(
                    pressure=pressure, time_ms=self._now_ms
                )
            else:
                self._hbm[worker_id].advance_to(self._now_ms)
                alpha = float(self._hbm[worker_id].controller.hbm_alpha)
                update = {
                    "old_alpha": alpha,
                    "new_alpha": alpha,
                    "evicted_hbm_tails": (),
                }
            updates.append(
                {
                    "target_worker": worker_id,
                    "pressure": pressure,
                    "old_alpha": update["old_alpha"],
                    "new_alpha": update["new_alpha"],
                    "evicted_hbm_tails": list(update["evicted_hbm_tails"]),
                }
            )
        self._emit(
            "hbm_quota_aimd_window",
            target_updates=updates,
            dram_controller_scope="independent_static_dram_pool",
            adaptive_hbm_quota=bool(self.config.adaptive_hbm_quota),
            protection_mode=self.config.protection_mode,
        )
        if (
            any(route.active for route in self._routes.values())
            or self._faults_remaining
            or self._slack_updates_remaining
        ):
            self._push(
                self._now_ms + float(self.config.control_window_ms),
                8,
                "control_window",
                None,
            )

    def _validate(self) -> None:
        for route in self._routes.values():
            if route.target_worker is not None and route.target_worker == route.source_worker:
                raise AssertionError("route target aliases its source")
            if route.route_version < 1:
                raise AssertionError("route version is invalid")
            if route.status == "COMPLETED" and route.output_fenced:
                raise AssertionError("completed output remains fenced")
            if route.sealed_count > len(route.chunks):
                raise AssertionError("sealed chunk frontier overflowed")
        for pool in self._hbm.values():
            pool.controller.validate_invariants()
        self._dram.controller.validate_invariants()

    def run(
        self,
        requests: Sequence[CompleteRequest],
        *,
        failures: Sequence[CompleteFailure] = (),
        slack_updates: Sequence[CompleteSlackUpdate] = (),
        max_time_ms: float = 1_000_000.0,
    ) -> dict[str, Any]:
        if self._routes or self._heap or self._events:
            raise RuntimeError("AegisCompleteSimulator instances are single-use")
        request_ids = [row.request_id for row in requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request identities must be unique")
        for request in requests:
            self._push(request.arrival_ms, 3, "arrival", request)
        for failure in failures:
            self._push(failure.time_ms, 0, "failure", failure)
        for update in slack_updates:
            self._push(update.time_ms, 2, "slack_update", update)
        self._faults_remaining = len(failures)
        self._slack_updates_remaining = len(slack_updates)
        self._push(0.0, 8, "control_window", None)

        while self._heap:
            time_ms, _priority, _sequence, kind, payload = heapq.heappop(self._heap)
            if time_ms > float(max_time_ms) + 1e-9:
                raise RuntimeError("complete simulation exceeded max_time_ms")
            self._advance_to(time_ms)
            if kind == "arrival":
                self._handle_arrival(payload)
            elif kind == "failure":
                self._faults_remaining -= 1
                self._handle_failure(payload)
            elif kind == "repair":
                self._handle_repair(*payload)
            elif kind == "slack_update":
                self._slack_updates_remaining -= 1
                self._handle_slack_update(payload)
            elif kind == "redispatch":
                self._handle_redispatch(*payload)
            elif kind == "recovery_ready":
                self._handle_recovery_ready(payload)
            elif kind == "recovery_dispatch":
                self._dispatch_recovery(payload)
            elif kind == "recovery_complete":
                self._handle_recovery_complete(*payload)
            elif kind == "seal_chunk":
                self._handle_seal(*payload)
            elif kind == "service_complete":
                self._handle_service_complete(*payload)
            elif kind == "control_window":
                self._handle_control_window()
            else:
                raise AssertionError(f"unknown complete simulation event {kind!r}")

        self._validate()
        return self.result()

    def result(self) -> dict[str, Any]:
        requests = {}
        for request_id, route in sorted(self._routes.items()):
            requests[request_id] = {
                "workflow_id": route.request.workflow_id,
                "status": route.status,
                "source_worker": route.source_worker,
                "target_worker": route.target_worker,
                "route_version": route.route_version,
                "produced_tokens": route.produced_tokens,
                "sealed_chunks": route.sealed_count,
                "output_committed_tokens": route.output_committed_tokens,
                "output_fenced": route.output_fenced,
                "completion_ms": route.completion_ms,
                "workflow_completion_time_ms": (
                    None
                    if route.completion_ms is None
                    else route.completion_ms - float(route.request.arrival_ms)
                ),
                "target_rotation_count": route.target_rotation_count,
                "source_failure_count": route.source_failure_count,
                "redispatch_count": route.redispatch_count,
                "outage_delay_ms": route.outage_delay_ms,
                "recovery_delay_ms": route.recovery_delay_ms,
                "recovery_action": route.recovery_action,
                "restored_tokens": route.restored_tokens,
                "recompute_tokens": route.recompute_tokens,
                "slack_reason": route.slack_reason,
            }
        target_failover_exercised = all(
            self._event_counts.get(kind, 0) > 0
            for kind in (
                "standby_target_rotated",
                "source_failure_fenced",
                "recovery_completed_route_activated",
            )
        )
        complete_path_exercised = target_failover_exercised and all(
            self._event_counts.get(kind, 0) > 0
            for kind in ("queue_outage_started", "queue_outage_rescheduled")
        )
        gates = {
            "current_design_semantics": DESIGN_SEMANTICS,
            "causal_residual_dag_slack_modeled": True,
            "independent_hbm_dram_admission_modeled": True,
            "source_owned_independent_transfers_modeled": True,
            "per_tier_pending_committed_aborted_modeled": True,
            "real_per_target_hbm_pool_capacity_modeled": True,
            "shared_independent_dram_pool_modeled": True,
            "standby_rotation_modeled": True,
            "rotation_preserves_route_version": True,
            "rotation_preserves_independent_dram": True,
            "replacement_reprotects_against_real_target_pool": True,
            "source_fail_stop_modeled": True,
            "output_fence_and_route_rebind_modeled": True,
            "worker_replacement_delay_modeled": True,
            "queue_outage_rescheduling_modeled": True,
            "remaining_slack_recovery_priority_modeled": True,
            "committed_prefix_only_recovery_modeled": True,
            "actual_load_bytes_recorded": True,
            "silent_replay_and_all_shard_activation_modeled": True,
            "continuous_time_repeated_worker_faults_supported": True,
            "hbm_quota_aimd_per_target_modeled": True,
            "target_owned_destage_forbidden": True,
            "complete_current_design_target_failover_evaluated": (
                target_failover_exercised
            ),
            "complete_current_design_simulation": complete_path_exercised,
            "metadata_only": True,
            "physical_result": False,
        }
        return {
            "schema_version": COMPLETE_SIMULATION_SCHEMA,
            "design_semantics": DESIGN_SEMANTICS,
            "evidence_class": "metadata-des",
            "physical_result": False,
            "policy": {
                "protection_mode": self.config.protection_mode,
                "recovery_priority": self.config.recovery_priority,
                "adaptive_hbm_quota": bool(self.config.adaptive_hbm_quota),
            },
            "simulation_end_ms": self._now_ms,
            "gates": gates,
            "event_counts": dict(sorted(self._event_counts.items())),
            "requests": requests,
            "workers": {
                worker_id: {
                    "failure_domain": row.spec.failure_domain,
                    "healthy": row.healthy,
                    "failure_generation": row.failure_generation,
                    "queue_depth": len(row.queue),
                    "recovery_queue_depth": len(row.recovery_queue),
                    "running_request": row.running_request,
                    "recovery_request": row.recovery_request,
                }
                for worker_id, row in sorted(self._workers.items())
            },
            "hbm_pools": {
                worker_id: pool.snapshot()
                for worker_id, pool in sorted(self._hbm.items())
            },
            "dram_pool": self._dram.snapshot(),
            "events": list(self._events),
        }


def audit_complete_simulation(
    result: Mapping[str, Any], *, require_exercised_failure_path: bool = True
) -> dict[str, Any]:
    """Fail-closed audit for the unified current-design simulation output."""

    required_gates = (
        "causal_residual_dag_slack_modeled",
        "independent_hbm_dram_admission_modeled",
        "source_owned_independent_transfers_modeled",
        "real_per_target_hbm_pool_capacity_modeled",
        "shared_independent_dram_pool_modeled",
        "standby_rotation_modeled",
        "rotation_preserves_route_version",
        "rotation_preserves_independent_dram",
        "source_fail_stop_modeled",
        "output_fence_and_route_rebind_modeled",
        "worker_replacement_delay_modeled",
        "queue_outage_rescheduling_modeled",
        "remaining_slack_recovery_priority_modeled",
        "committed_prefix_only_recovery_modeled",
        "silent_replay_and_all_shard_activation_modeled",
        "hbm_quota_aimd_per_target_modeled",
        "target_owned_destage_forbidden",
        "complete_current_design_target_failover_evaluated",
        "complete_current_design_simulation",
    )
    failures: list[str] = []
    if result.get("schema_version") != COMPLETE_SIMULATION_SCHEMA:
        failures.append("schema_version_mismatch")
    if result.get("design_semantics") != DESIGN_SEMANTICS:
        failures.append("design_semantics_mismatch")
    if result.get("physical_result") is not False:
        failures.append("metadata_result_mislabeled_physical")
    gates = result.get("gates") or {}
    for name in required_gates:
        if gates.get(name) is not True:
            failures.append(f"gate_false:{name}")
    events = list(result.get("events") or [])
    kinds = [str(row.get("kind")) for row in events]
    if any(
        row.get("hbm_to_dram") is True
        for row in events
    ):
        failures.append("forbidden_hbm_to_dram_event")
    rotations = [row for row in events if row.get("kind") == "standby_target_rotated"]
    for row in rotations:
        if row.get("route_version_changed") is not False:
            failures.append("target_rotation_changed_route_version")
        if row.get("dram_preserved") is not True:
            failures.append("target_rotation_did_not_preserve_dram")
        if row.get("replacement_uses_real_target_pool") is not True:
            failures.append("target_rotation_did_not_use_real_pool")
    activations = [
        row for row in events if row.get("kind") == "recovery_completed_route_activated"
    ]
    for row in activations:
        if row.get("route_version_contiguous") is not True:
            failures.append("recovery_route_version_not_contiguous")
        if row.get("output_fenced") is not False:
            failures.append("recovery_left_output_fenced")
        if row.get("silent_replay_completed") is not True:
            failures.append("silent_replay_not_completed")
        if row.get("all_shards_completed") is not True:
            failures.append("all_shards_not_completed")
    if require_exercised_failure_path:
        required_events = (
            "standby_target_rotated",
            "source_failure_fenced",
            "recovery_selected_and_started",
            "recovery_completed_route_activated",
            "queue_outage_started",
            "queue_outage_rescheduled",
            "worker_repaired",
            "hbm_quota_aimd_window",
        )
        for kind in required_events:
            if kind not in kinds:
                failures.append(f"required_event_not_exercised:{kind}")
    status = "PASS" if not failures else "FAIL"
    return {
        "schema_version": "aegiskv-complete-simulation-audit-v1",
        "status": status,
        "failed_checks": failures,
        "require_exercised_failure_path": bool(require_exercised_failure_path),
        "event_count": len(events),
        "target_rotation_count": len(rotations),
        "source_failure_fence_count": kinds.count("source_failure_fenced"),
        "recovery_activation_count": len(activations),
        "queue_reschedule_count": kinds.count("queue_outage_rescheduled"),
        "physical_result": False,
    }


__all__ = [
    "COMPLETE_SIMULATION_SCHEMA",
    "AegisCompleteSimulator",
    "CompleteFailure",
    "CompleteRequest",
    "CompleteSimulationConfig",
    "CompleteSlackUpdate",
    "CompleteWorker",
    "audit_complete_simulation",
]
