"""Event-level simulator for the complete current AegisKV design.

This module composes the metadata controller in ``aegis_independent_tiers``
with the pieces that deliberately do not belong in that policy object:

* causal progressive chunk sealing;
* independent source-owned D2D and D2H transfer queues;
* all-shard completion barriers;
* request-completion reclamation after both paths become terminal; and
* source/standby failure hooks used by large trace replay.

The simulator remains metadata-only.  Bytes are accounted exactly, while
transfer and recovery time are driven by explicit calibration parameters.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from sglsim.srt.simulation.aegis_independent_tiers import (
    DESIGN_SEMANTICS,
    AegisIndependentTierController,
    ChunkSpec,
    CopyState,
    IndependentTierConfig,
    PlacementDecision,
    ProtectionTier,
    RecoveryDecision,
    RecoveryTiming,
    RequestRecoveryProfile,
)
from sglsim.srt.simulation.aegis_transfer_engine import (
    AegisTransferEngine,
    TransferSubmission,
)
from sglsim.srt.simulation.aegis_residual_dag import (
    ResidualDagSlackTicket,
    resolve_slack_for_design,
)


_PROTECTION_PATHS = {
    ProtectionTier.HBM: ("source_to_peer_hbm", "source_d2d"),
    ProtectionTier.DRAM: ("source_to_store_dram", "source_d2h"),
}


@dataclass(frozen=True)
class CalibratedTransferPath:
    setup_ms: float
    bandwidth_bytes_per_ms: float
    provenance: str
    first_transfer_warmup_ms: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.setup_ms)) or float(self.setup_ms) < 0.0:
            raise ValueError("transfer setup_ms must be finite and nonnegative")
        if not math.isfinite(float(self.bandwidth_bytes_per_ms)) or float(
            self.bandwidth_bytes_per_ms
        ) <= 0.0:
            raise ValueError("transfer bandwidth must be finite and positive")
        if not self.provenance:
            raise ValueError("transfer calibration provenance is required")
        if not math.isfinite(float(self.first_transfer_warmup_ms)) or float(
            self.first_transfer_warmup_ms
        ) < 0.0:
            raise ValueError(
                "transfer first_transfer_warmup_ms must be finite and nonnegative"
            )

    def duration_ms(self, byte_size: int, *, first_transfer: bool = False) -> float:
        if int(byte_size) <= 0:
            raise ValueError("transfer byte_size must be positive")
        return (
            float(self.setup_ms)
            + (
                float(self.first_transfer_warmup_ms)
                if bool(first_transfer)
                else 0.0
            )
            + int(byte_size) / float(self.bandwidth_bytes_per_ms)
        )


@dataclass(frozen=True)
class AegisFullDesignConfig:
    hbm_capacity_bytes: int
    dram_capacity_bytes: int
    kv_bytes_per_token: int
    hbm_copy: CalibratedTransferPath
    dram_copy: CalibratedTransferPath
    recompute: RecoveryTiming
    hbm_load: RecoveryTiming
    dram_load: RecoveryTiming
    prefix_chunk_tokens: int = 4096
    decode_chunk_tokens: int = 256
    pp_size: int = 1
    tp_size: int = 1
    initial_hbm_alpha: float = 1.0
    minimum_hbm_alpha: float = 0.10
    maximum_hbm_alpha: float = 1.0
    aimd_decrease_factor: float = 0.50
    aimd_increase_step: float = 0.10
    aimd_stable_windows: int = 3
    hbm_copy_failure_probability: float = 0.0
    dram_copy_failure_probability: float = 0.0
    random_salt: str = "aegiskv-full-design-v1"
    event_log_limit: int = 256

    def __post_init__(self) -> None:
        for name in (
            "hbm_capacity_bytes",
            "dram_capacity_bytes",
            "kv_bytes_per_token",
            "prefix_chunk_tokens",
            "decode_chunk_tokens",
            "pp_size",
            "tp_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "hbm_copy_failure_probability",
            "dram_copy_failure_probability",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if isinstance(self.event_log_limit, bool) or int(self.event_log_limit) < 0:
            raise ValueError("event_log_limit must be nonnegative")

    @property
    def shard_count(self) -> int:
        return int(self.pp_size) * int(self.tp_size)


@dataclass(frozen=True)
class SealedChunk:
    ordinal: int
    token_start: int
    token_end: int
    kind: str

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


def sealed_chunks(
    prompt_tokens: int,
    output_tokens: int,
    *,
    prefix_chunk_tokens: int = 4096,
    decode_chunk_tokens: int = 256,
) -> tuple[SealedChunk, ...]:
    """Return chunks that become immutable before request completion.

    Every prefix chunk is sealed, including a short final prefix chunk.  Decode
    chunks are sealed only at full boundaries; the terminal open decode suffix
    remains unprotected and is recomputed after a failure.
    """

    prompt = max(int(prompt_tokens), 0)
    output = max(int(output_tokens), 0)
    prefix_q = int(prefix_chunk_tokens)
    decode_q = int(decode_chunk_tokens)
    if prefix_q <= 0 or decode_q <= 0:
        raise ValueError("chunk sizes must be positive")
    chunks: list[SealedChunk] = []
    cursor = 0
    ordinal = 0
    while cursor < prompt:
        end = min(cursor + prefix_q, prompt)
        chunks.append(SealedChunk(ordinal, cursor, end, "prefix"))
        cursor = end
        ordinal += 1
    decode_end = prompt + (output // decode_q) * decode_q
    while cursor < decode_end:
        end = cursor + decode_q
        chunks.append(SealedChunk(ordinal, cursor, end, "decode"))
        cursor = end
        ordinal += 1
    return tuple(chunks)


@dataclass
class _RequestRuntime:
    prompt_tokens: int
    output_tokens: int
    slack_budget_ms: float
    slack_updated_at_ms: float
    slack_evidence_reason: str
    slack_graph_version: str = ""
    slack_node_id: str = ""
    slack_metadata_only: bool = True
    next_ordinal: int = 0
    completion_requested: bool = False
    reclaimed: bool = False
    chunks: list[SealedChunk] = field(default_factory=list)
    decisions: list[PlacementDecision] = field(default_factory=list)


class AegisFullDesignRuntime:
    """One logical route target with independent HBM and DRAM pools."""

    def __init__(
        self,
        config: AegisFullDesignConfig,
        *,
        target_id: str = "target-0",
        dram_pool_id: str = "store-dram-0",
        dram_failure_domain: str = "store-domain-0",
    ) -> None:
        self.config = config
        self.target_id = str(target_id)
        controller_config = IndependentTierConfig(
            hbm_capacity_bytes=int(config.hbm_capacity_bytes),
            dram_capacity_bytes=int(config.dram_capacity_bytes),
            initial_hbm_alpha=float(config.initial_hbm_alpha),
            minimum_hbm_alpha=float(config.minimum_hbm_alpha),
            maximum_hbm_alpha=float(config.maximum_hbm_alpha),
            aimd_decrease_factor=float(config.aimd_decrease_factor),
            aimd_increase_step=float(config.aimd_increase_step),
            aimd_stable_windows=int(config.aimd_stable_windows),
            recompute=config.recompute,
            hbm_load=config.hbm_load,
            dram_load=config.dram_load,
            hbm_copy=RecoveryTiming(
                config.hbm_copy.setup_ms,
                config.kv_bytes_per_token
                / config.hbm_copy.bandwidth_bytes_per_ms,
            ),
            dram_copy=RecoveryTiming(
                config.dram_copy.setup_ms,
                config.kv_bytes_per_token
                / config.dram_copy.bandwidth_bytes_per_ms,
            ),
            hbm_pool_id=f"peer-hbm-{self.target_id}",
            dram_pool_id=str(dram_pool_id),
            dram_failure_domain=str(dram_failure_domain),
            retain_event_log=bool(config.event_log_limit),
            retain_record_history=False,
            event_log_limit=int(config.event_log_limit),
        )
        self.controller = AegisIndependentTierController(controller_config)
        self._now_ms = 0.0
        self.transfer_engine = AegisTransferEngine()
        self._hbm_local_resource_operations_since_sample: list[dict[str, Any]] = []
        self._transfer_attempt_sequence = 0
        self._requests: dict[str, _RequestRuntime] = {}
        self._protection_paths: dict[ProtectionTier, dict[str, Any]] = {
            tier: {
                "path_id": path_id,
                "tier": tier.value,
                "copy_kind": copy_kind,
                "attempted_copies": 0,
                "attempted_bytes": 0,
                "successful_copies": 0,
                "successful_bytes": 0,
                "failed_copies": 0,
                "cancelled_copies": 0,
                "first_transfer_warmup_count": 0,
                "healthy_period_attempted_copies": 0,
                "healthy_period_attempted_bytes": 0,
                "healthy_period_first_transfer_warmup_count": 0,
                "healthy_period_transfer_service_ms": 0.0,
            }
            for tier, (path_id, copy_kind) in _PROTECTION_PATHS.items()
        }
        self._recovery_estimate_action_counts: dict[str, int] = {}
        self._recovery_execution_action_counts: dict[str, int] = {}
        self._metrics: dict[str, float] = {
            "sealed_chunks": 0,
            "hbm_jobs": 0,
            "dram_jobs": 0,
            "hbm_copy_bytes": 0,
            "dram_copy_bytes": 0,
            "hbm_copy_service_ms": 0,
            "dram_copy_service_ms": 0,
            "copy_shard_acks": 0,
            "copy_failures": 0,
            "copy_cancellations": 0,
            "request_reclaims": 0,
            "request_reclaim_waits": 0,
            "source_failures": 0,
            "target_rotations": 0,
        }

    @property
    def now_ms(self) -> float:
        return self._now_ms

    @staticmethod
    def _probability_draw(identity: str) -> float:
        digest = hashlib.sha256(identity.encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def _set_time(self, time_ms: float) -> None:
        target = float(time_ms)
        if not math.isfinite(target) or target < self._now_ms - 1e-9:
            raise ValueError("Aegis full-design time cannot regress")
        delta = max(target - self._now_ms, 0.0)
        if delta:
            self.controller.advance_time(delta)
            self._now_ms = target

    def _copy_path(self, tier: ProtectionTier) -> CalibratedTransferPath:
        return self.config.hbm_copy if tier is ProtectionTier.HBM else self.config.dram_copy

    def _copy_failure_probability(self, tier: ProtectionTier) -> float:
        return (
            self.config.hbm_copy_failure_probability
            if tier is ProtectionTier.HBM
            else self.config.dram_copy_failure_probability
        )

    def _hbm_used_bytes(self) -> int:
        # This is hot-path accounting (several calls per sealed chunk).  A
        # full controller snapshot serializes every live record and victim
        # frontier, turning a long trace into quadratic work.  The controller
        # already maintains exact used-byte accounting, so read that scalar
        # directly without changing admission or reclaim semantics.
        return self.controller.pool_used_bytes(ProtectionTier.HBM)

    def _record_hbm_reclaim(self, used_before: int) -> None:
        used_after = self._hbm_used_bytes()
        if used_after < int(used_before):
            self._hbm_local_resource_operations_since_sample.append(
                {"operation": "reclaim", "bytes": int(used_before) - used_after}
            )

    def _admit_chunk(
        self,
        spec: ChunkSpec,
        *,
        allowed_tiers: Iterable[ProtectionTier] | None = None,
    ) -> PlacementDecision:
        """Admit independently and expose target-local HBM eviction work."""

        used_before = self._hbm_used_bytes()
        decision = self.controller.admit_chunk(spec, allowed_tiers=allowed_tiers)
        if decision.hbm.admitted:
            # Admission reserves spec.byte_size after deleting any selected
            # HBM victims. The allocation itself is recorded when its source
            # D2D job is scheduled; account only the victim bytes here.
            used_after = self._hbm_used_bytes()
            reclaimed = max(used_before + int(spec.byte_size) - used_after, 0)
            if reclaimed:
                self._hbm_local_resource_operations_since_sample.append(
                    {"operation": "reclaim", "bytes": reclaimed}
                )
        return decision

    def _schedule_transfer(self, tier: ProtectionTier, spec: ChunkSpec) -> None:
        path = self._copy_path(tier)
        path_metrics = self._protection_paths[tier]
        first_transfer = int(path_metrics["attempted_copies"]) == 0
        duration = path.duration_ms(
            spec.byte_size, first_transfer=first_transfer
        )
        self._transfer_attempt_sequence += 1
        failed = self._probability_draw(
            f"{self.config.random_salt}:{tier.value}:{spec.logical_id}:"
            f"{self._transfer_attempt_sequence}"
        ) < self._copy_failure_probability(tier)
        acked = tuple(range(spec.shard_count)) if not failed else ()
        self.transfer_engine.enqueue(
            TransferSubmission(
                tier=tier,
                spec=spec,
                destination_id=(
                    spec.hbm_target
                    if tier is ProtectionTier.HBM
                    else self.controller.config.dram_pool_id
                ),
                copy_kind=(
                    "source_d2d"
                    if tier is ProtectionTier.HBM
                    else "source_d2h"
                ),
                ready_at_ms=self._now_ms,
                service_ms=duration,
                transfer_succeeded=not failed,
                acked_shards=acked,
                source_owned=True,
                target_owned=False,
                hbm_to_dram=False,
            )
        )
        prefix = tier.value.lower()
        self._metrics[f"{prefix}_jobs"] += 1
        self._metrics[f"{prefix}_copy_bytes"] += spec.byte_size
        self._metrics[f"{prefix}_copy_service_ms"] += duration
        path_metrics["attempted_copies"] += 1
        path_metrics["attempted_bytes"] += int(spec.byte_size)
        if first_transfer:
            path_metrics["first_transfer_warmup_count"] += 1
        if int(self._metrics["source_failures"]) == 0:
            path_metrics["healthy_period_attempted_copies"] += 1
            path_metrics["healthy_period_attempted_bytes"] += int(spec.byte_size)
            if first_transfer:
                path_metrics["healthy_period_first_transfer_warmup_count"] += 1
            path_metrics["healthy_period_transfer_service_ms"] += float(duration)
        if tier is ProtectionTier.HBM:
            # HBM capacity is reserved before the asynchronous source D2D copy.
            # This models the target-local allocator operation, not transfer
            # service time.
            self._hbm_local_resource_operations_since_sample.append(
                {"operation": "allocation", "bytes": int(spec.byte_size)}
            )

    def _try_reclaim(self, request_id: str) -> None:
        runtime = self._requests.get(request_id)
        if runtime is None or runtime.reclaimed or not runtime.completion_requested:
            return
        used_before = self._hbm_used_bytes()
        result = self.controller.complete_request(request_id)
        self._record_hbm_reclaim(used_before)
        if result["completed"]:
            runtime.reclaimed = True
            self._metrics["request_reclaims"] += 1
            self._requests.pop(request_id, None)

    def _decay_request_slack(self, request_id: str) -> float:
        """Apply Algorithm 1's controller-side elapsed-time decay."""

        runtime = self._requests[str(request_id)]
        remaining = max(
            runtime.slack_budget_ms
            - max(self._now_ms - runtime.slack_updated_at_ms, 0.0),
            0.0,
        )
        current = self.controller.request_profile(request_id).slack_ms
        if not math.isclose(current, remaining, rel_tol=0.0, abs_tol=1e-9):
            self.controller.update_slack(request_id, remaining)
        return remaining

    @staticmethod
    def _resolve_residual_slack(
        slack_ms: float,
        slack_evidence: Mapping[str, Any] | None,
    ) -> tuple[float, str, dict[str, Any]]:
        fields = dict(slack_evidence or {})
        fields["workflow_slack_ms"] = max(float(slack_ms), 0.0)
        value, reason = resolve_slack_for_design(
            fields, design_semantics=DESIGN_SEMANTICS
        )
        return value, reason, fields

    def advance_to(self, time_ms: float) -> None:
        target = float(time_ms)
        if target < self._now_ms - 1e-9:
            raise ValueError("Aegis full-design time cannot regress")
        while (job := self.transfer_engine.pop_next_ready(target)) is not None:
            self._set_time(job.completed_at_ms)
            path = self._protection_paths[job.tier]
            if job.cancelled or self.controller.copy_state(
                job.spec.request_id, job.spec.ordinal, job.tier
            ) is not CopyState.PENDING:
                self._metrics["copy_cancellations"] += 1
                path["cancelled_copies"] += 1
                self._try_reclaim(job.spec.request_id)
                continue
            proof = self.controller.completion_proof(
                job.spec, job.tier, succeeded=job.transfer_succeeded
            )
            if job.transfer_succeeded:
                self._metrics["copy_shard_acks"] += len(job.acked_shards)
                path["successful_copies"] += 1
                path["successful_bytes"] += int(job.spec.byte_size)
            else:
                self._metrics["copy_failures"] += 1
                path["failed_copies"] += 1
            used_before = self._hbm_used_bytes()
            self.controller.complete_copy(
                job.tier, job.spec.request_id, job.spec.ordinal, proof
            )
            if job.tier is ProtectionTier.HBM:
                self._record_hbm_reclaim(used_before)
            self._try_reclaim(job.spec.request_id)
        self._set_time(target)

    def register_request(
        self,
        *,
        request_id: str,
        source_worker: str,
        source_failure_domain: str,
        hbm_target: str,
        hbm_failure_domain: str,
        route_version: int,
        prompt_tokens: int,
        output_tokens: int,
        slack_ms: float,
        model_revision: str,
        kv_abi: str,
        parallel_layout: str,
        time_ms: float,
        slack_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.advance_to(time_ms)
        if request_id in self._requests:
            raise ValueError(f"request {request_id!r} is already registered")
        trusted_slack, slack_reason, slack_fields = self._resolve_residual_slack(
            slack_ms, slack_evidence
        )
        self.controller.register_request(
            RequestRecoveryProfile(
                request_id=str(request_id),
                source_worker=str(source_worker),
                source_failure_domain=str(source_failure_domain),
                hbm_target=str(hbm_target),
                hbm_failure_domain=str(hbm_failure_domain),
                route_version=int(route_version),
                source_token_end=0,
                slack_ms=trusted_slack,
                model_revision=str(model_revision),
                kv_abi=str(kv_abi),
                parallel_layout=str(parallel_layout),
            )
        )
        self._requests[str(request_id)] = _RequestRuntime(
            prompt_tokens=max(int(prompt_tokens), 0),
            output_tokens=max(int(output_tokens), 0),
            slack_budget_ms=trusted_slack,
            slack_updated_at_ms=float(time_ms),
            slack_evidence_reason=slack_reason,
            slack_graph_version=str(
                slack_fields.get("aegis_slack_graph_version") or ""
            ),
            slack_node_id=str(slack_fields.get("aegis_slack_node_id") or ""),
            slack_metadata_only=True,
        )

    def update_slack(
        self,
        request_id: str,
        *,
        slack_ms: float,
        time_ms: float,
        slack_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self.advance_to(time_ms)
        value, reason, fields = self._resolve_residual_slack(
            slack_ms, slack_evidence
        )
        runtime = self._requests[str(request_id)]
        runtime.slack_budget_ms = value
        runtime.slack_updated_at_ms = float(time_ms)
        runtime.slack_evidence_reason = reason
        runtime.slack_graph_version = str(
            fields.get("aegis_slack_graph_version") or ""
        )
        runtime.slack_node_id = str(fields.get("aegis_slack_node_id") or "")
        runtime.slack_metadata_only = True
        self.controller.update_slack(request_id, value)

    def update_slack_from_ticket(
        self,
        request_id: str,
        ticket: ResidualDagSlackTicket,
        *,
        time_ms: float,
    ) -> None:
        self.update_slack(
            request_id,
            slack_ms=ticket.slack_ms,
            time_ms=time_ms,
            slack_evidence=ticket.to_simulation_fields(),
        )

    def prioritize_recovery(
        self,
        request_ids: Iterable[str],
        *,
        activated_targets: Mapping[str, str],
        time_ms: float,
    ) -> tuple[dict[str, Any], ...]:
        """Order ready recovery by remaining-Slack due time."""

        self.advance_to(time_ms)
        rows: list[dict[str, Any]] = []
        for request_id in request_ids:
            request_id = str(request_id)
            remaining = self._decay_request_slack(request_id)
            decision = self.controller.estimate_recovery(
                request_id,
                activated_target=str(activated_targets[request_id]),
            )
            runtime = self._requests[request_id]
            rows.append(
                {
                    "request_id": request_id,
                    "remaining_slack_ms": remaining,
                    "due_at_ms": self._now_ms + remaining,
                    "recovery_ms": decision.recovery_ms,
                    "action": decision.action,
                    "slack_evidence_reason": runtime.slack_evidence_reason,
                    "slack_graph_version": runtime.slack_graph_version,
                    "metadata_only": runtime.slack_metadata_only,
                }
            )
        rows.sort(
            key=lambda row: (
                row["due_at_ms"],
                row["recovery_ms"],
                row["request_id"],
            )
        )
        return tuple(rows)

    def seal_chunk(
        self,
        request_id: str,
        chunk: SealedChunk,
        *,
        time_ms: float,
        allowed_tiers: Iterable[ProtectionTier] = (
            ProtectionTier.HBM,
            ProtectionTier.DRAM,
        ),
    ) -> PlacementDecision:
        self.advance_to(time_ms)
        runtime = self._requests[str(request_id)]
        if runtime.completion_requested:
            raise ValueError("cannot seal a chunk after request completion")
        if chunk.ordinal != runtime.next_ordinal:
            raise ValueError("sealed chunk ordinal is not progressive")
        self._decay_request_slack(request_id)
        profile = self.controller.request_profile(request_id)
        self.controller.update_source_frontier(request_id, chunk.token_end)
        spec = ChunkSpec(
            request_id=str(request_id),
            ordinal=int(chunk.ordinal),
            token_start=int(chunk.token_start),
            token_end=int(chunk.token_end),
            byte_size=int(chunk.token_count) * int(self.config.kv_bytes_per_token),
            route_version=profile.route_version,
            source_worker=profile.source_worker,
            hbm_target=profile.hbm_target,
            model_revision=profile.model_revision,
            kv_abi=profile.kv_abi,
            parallel_layout=profile.parallel_layout,
            shard_count=self.config.shard_count,
        )
        decision = self._admit_chunk(spec, allowed_tiers=allowed_tiers)
        runtime.decisions.append(decision)
        runtime.chunks.append(chunk)
        runtime.next_ordinal += 1
        self._metrics["sealed_chunks"] += 1
        if decision.hbm.admitted:
            self._schedule_transfer(ProtectionTier.HBM, spec)
        if decision.dram.admitted:
            self._schedule_transfer(ProtectionTier.DRAM, spec)
        return decision

    def finish_request(self, request_id: str, *, time_ms: float) -> dict[str, Any]:
        self.advance_to(time_ms)
        runtime = self._requests[str(request_id)]
        runtime.completion_requested = True
        used_before = self._hbm_used_bytes()
        result = self.controller.complete_request(request_id)
        self._record_hbm_reclaim(used_before)
        if result["completed"]:
            runtime.reclaimed = True
            self._metrics["request_reclaims"] += 1
            self._requests.pop(str(request_id), None)
        else:
            self._metrics["request_reclaim_waits"] += 1
        return result

    def rotate_target(
        self,
        request_id: str,
        *,
        failed_target: str,
        replacement_target: str,
        replacement_failure_domain: str,
        time_ms: float,
    ) -> dict[str, Any]:
        self.advance_to(time_ms)
        if str(request_id) not in self._requests:
            return {
                "request_id": str(request_id),
                "skipped": "request_reclaimed_before_target_failure",
                "route_version": None,
                "replacement_target": str(replacement_target),
                "invalidated_hbm_records": 0,
                "dram_records_preserved": 0,
                "replacement_hbm_reprotections": 0,
            }
        self._decay_request_slack(request_id)
        used_before = self._hbm_used_bytes()
        result = self.controller.rotate_hbm_target(
            request_id,
            failed_target=failed_target,
            replacement_target=replacement_target,
            replacement_failure_domain=replacement_failure_domain,
        )
        self._record_hbm_reclaim(used_before)
        # The source still owns its production buffers.  Rebuild the replacement
        # target's continuous HBM prefix from ordinal zero; old target records
        # were invalidated, never relabelled.  DRAM is neither consulted nor
        # modified by this replacement copy-ahead.
        runtime = self._requests[str(request_id)]
        profile = self.controller.request_profile(request_id)
        reprotected = 0
        for chunk in runtime.chunks:
            spec = ChunkSpec(
                request_id=str(request_id),
                ordinal=chunk.ordinal,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
                byte_size=chunk.token_count * int(self.config.kv_bytes_per_token),
                route_version=profile.route_version,
                source_worker=profile.source_worker,
                hbm_target=profile.hbm_target,
                model_revision=profile.model_revision,
                kv_abi=profile.kv_abi,
                parallel_layout=profile.parallel_layout,
                shard_count=self.config.shard_count,
            )
            decision = self._admit_chunk(
                spec, allowed_tiers=(ProtectionTier.HBM,)
            )
            runtime.decisions.append(decision)
            if decision.hbm.admitted:
                self._schedule_transfer(ProtectionTier.HBM, spec)
                reprotected += 1
        result = {**result, "replacement_hbm_reprotections": reprotected}
        self._metrics["target_rotations"] += 1
        return result

    def observe_source_frontier(
        self,
        request_id: str,
        *,
        source_token_end: int,
        time_ms: float,
    ) -> None:
        """Reveal produced but not necessarily sealed tokens at a failure cut."""

        self.advance_to(time_ms)
        current = self.controller.request_profile(request_id).source_token_end
        if int(source_token_end) < current:
            raise ValueError("observed source frontier regressed")
        self.controller.update_source_frontier(request_id, int(source_token_end))

    def has_request(self, request_id: str) -> bool:
        return str(request_id) in self._requests

    def estimate_source_failure(
        self,
        request_id: str,
        *,
        activated_target: str,
        source_token_end: int,
        remaining_slack_ms: float,
        time_ms: float,
        slack_evidence: Mapping[str, Any] | None = None,
    ) -> RecoveryDecision:
        """Take a non-destructive fail-now snapshot for a paired campaign."""

        self.observe_source_frontier(
            request_id,
            source_token_end=int(source_token_end),
            time_ms=time_ms,
        )
        self.update_slack(
            request_id,
            slack_ms=max(float(remaining_slack_ms), 0.0),
            time_ms=time_ms,
            slack_evidence=slack_evidence,
        )
        result = self.controller.estimate_recovery(
            request_id, activated_target=str(activated_target)
        )
        self._recovery_estimate_action_counts[result.action] = (
            self._recovery_estimate_action_counts.get(result.action, 0) + 1
        )
        return result

    def estimate_recovery_at(
        self,
        request_id: str,
        *,
        activated_target: str,
        source_token_end: int,
        slack_ms: float,
        time_ms: float,
    ) -> RecoveryDecision:
        """Advance physical copies, then issue a pure counterfactual query.

        Advancing time is part of normal event execution.  The query itself
        neither overwrites the source frontier/slack metadata nor increments
        the runtime's recovery action counters, so inserting shadow probes
        cannot affect later admission, eviction, AIMD, or failure behavior.
        """

        self.advance_to(time_ms)
        return self.controller.estimate_recovery_at(
            request_id,
            source_token_end=int(source_token_end),
            slack_ms=max(float(slack_ms), 0.0),
            activated_target=str(activated_target),
        )

    def recover_source_failure(
        self,
        request_id: str,
        *,
        activated_target: str,
        next_route_version: int,
        remaining_slack_ms: float,
        time_ms: float,
        slack_evidence: Mapping[str, Any] | None = None,
    ) -> RecoveryDecision:
        self.advance_to(time_ms)
        self.update_slack(
            request_id,
            slack_ms=max(float(remaining_slack_ms), 0.0),
            time_ms=time_ms,
            slack_evidence=slack_evidence,
        )
        used_before = self._hbm_used_bytes()
        result = self.controller.recover_after_source_failure(
            request_id,
            activated_target=activated_target,
            next_route_version=int(next_route_version),
        )
        self._record_hbm_reclaim(used_before)
        self._metrics["source_failures"] += 1
        self._recovery_execution_action_counts[result.action] = (
            self._recovery_execution_action_counts.get(result.action, 0) + 1
        )
        return result

    def control_hbm_quota(self, *, pressure: bool, time_ms: float) -> dict[str, Any]:
        self.advance_to(time_ms)
        used_before = self._hbm_used_bytes()
        result = self.controller.control_hbm_quota(pressure=bool(pressure))
        self._record_hbm_reclaim(used_before)
        return result

    def sample_hbm_local_resource_operations(
        self, *, time_ms: float
    ) -> dict[str, Any]:
        """Drain simulated target-local HBM allocation/reclaim operations."""

        self.advance_to(time_ms)
        operations = list(self._hbm_local_resource_operations_since_sample)
        self._hbm_local_resource_operations_since_sample.clear()
        allocation = [
            row for row in operations if row["operation"] == "allocation"
        ]
        reclaim = [row for row in operations if row["operation"] == "reclaim"]
        return {
            "operation_count": len(operations),
            "allocation_count": len(allocation),
            "reclaim_count": len(reclaim),
            "allocation_bytes": sum(int(row["bytes"]) for row in allocation),
            "reclaim_bytes": sum(int(row["bytes"]) for row in reclaim),
        }

    def flush(self) -> None:
        completion = self.transfer_engine.maximum_completion_ms
        if completion is not None:
            self.advance_to(completion)

    def validate_invariants(self) -> None:
        self.controller.validate_invariants()
        for request_id, runtime in self._requests.items():
            if runtime.reclaimed:
                try:
                    self.controller.request_profile(request_id)
                except KeyError:
                    continue
                raise AssertionError("reclaimed request remains in the controller")

    def snapshot(self, *, include_controller: bool = False) -> dict[str, Any]:
        controller = self.controller.snapshot()
        protection_paths: dict[str, Any] = {}
        for tier, path in self._protection_paths.items():
            attempted = int(path["attempted_copies"])
            terminal = sum(
                int(path[name])
                for name in (
                    "successful_copies",
                    "failed_copies",
                    "cancelled_copies",
                )
            )
            protection_paths[str(path["path_id"])] = {
                key: (
                    int(value)
                    if isinstance(value, float) and float(value).is_integer()
                    else value
                )
                for key, value in path.items()
            }
            protection_paths[str(path["path_id"])]["pending_copies"] = (
                attempted - terminal
            )
        payload = {
            "design_semantics": DESIGN_SEMANTICS,
            "target_id": self.target_id,
            "time_ms": self._now_ms,
            "metrics": {
                key: int(value) if float(value).is_integer() else value
                for key, value in sorted(self._metrics.items())
            },
            "pending_transfer_jobs": self.transfer_engine.pending_count,
            "lane_available_ms": self.transfer_engine.snapshot()[
                "lane_available_ms"
            ],
            "transfer_engine": self.transfer_engine.snapshot(),
            "protection_paths": protection_paths,
            "pool_churn": {
                tier.value: dict(controller["pools"][tier.value]["churn"])
                for tier in (ProtectionTier.HBM, ProtectionTier.DRAM)
            },
            "recovery_estimate_action_counts": dict(
                sorted(self._recovery_estimate_action_counts.items())
            ),
            "recovery_execution_action_counts": dict(
                sorted(self._recovery_execution_action_counts.items())
            ),
            "healthy_period_definition": (
                "protection submissions before the first mutating source failure"
            ),
            "active_requests": sum(
                not runtime.reclaimed for runtime in self._requests.values()
            ),
            "controller_event_counts": controller["event_counts"],
            "controller_event_sequence": controller["event_sequence"],
            "controller_event_log_dropped": controller["event_log_dropped"],
            "config": {
                **asdict(self.config),
                "hbm_copy": asdict(self.config.hbm_copy),
                "dram_copy": asdict(self.config.dram_copy),
                "recompute": asdict(self.config.recompute),
                "hbm_load": asdict(self.config.hbm_load),
                "dram_load": asdict(self.config.dram_load),
            },
        }
        if include_controller:
            payload["controller"] = controller
        return payload


__all__ = [
    "AegisFullDesignConfig",
    "AegisFullDesignRuntime",
    "CalibratedTransferPath",
    "SealedChunk",
    "sealed_chunks",
]
