"""v3.1.3 S4/S5 extensions with isolated ablations and control telemetry.

The validated v3.1.3 scheduler remains unchanged.  This module adds only the
three paper ablations and richer, event-derived control-window snapshots.  In
particular, ``no Algorithm 2`` is not mapped to a different baseline runtime:
it keeps the Aegis controller, causal source copies, physical target capacity,
and Algorithm 3, but replaces independent per-tier SlackLandlord admission
with coupled FCFS all-or-nothing admission when both pools have direct space.
"""

from __future__ import annotations

import heapq
from typing import Any, Iterable

import active_hbm_v313 as v313
import run_preliminary as core
from sglsim.srt.simulation.aegis_independent_tiers import (
    AegisIndependentTierController,
    ChunkSpec,
    PlacementDecision,
    ProtectionTier,
)


GIB = 1024**3


class CoupledFCFSTierAblationController(AegisIndependentTierController):
    """Algorithm-2 ablation that preserves every other Aegis mechanism.

    A chunk is admitted to both tiers only when both next-frontier positions
    have positive recovery gain and direct capacity.  It never invokes the
    independent per-pool victim selector.  This is a mechanism ablation, not a
    deliberately undersized pool or a remapping to Static FCFS.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.coupled_attempt_count = 0
        self.coupled_success_count = 0
        self.coupled_rejection_count = 0

    def admit_chunk(
        self,
        spec: ChunkSpec,
        *,
        allowed_tiers: Iterable[ProtectionTier] | None = None,
    ) -> PlacementDecision:
        self._validate_spec(spec)
        allowed = (
            {ProtectionTier(tier) for tier in allowed_tiers}
            if allowed_tiers is not None
            else {ProtectionTier.HBM, ProtectionTier.DRAM}
        )
        if allowed != {ProtectionTier.HBM, ProtectionTier.DRAM}:
            raise ValueError("coupled-tier ablation requires both frozen tiers")
        self.coupled_attempt_count += 1
        reasons: dict[ProtectionTier, str] = {}
        for tier in (ProtectionTier.HBM, ProtectionTier.DRAM):
            pool = self._pools[tier]
            if tier is ProtectionTier.HBM and self._hbm_admission_paused:
                reasons[tier] = "hbm_admission_paused_for_quota_reclaim"
            elif spec.ordinal != len(self._live(pool, spec.request_id)):
                reasons[tier] = "noncontiguous_frontier"
            elif self._candidate_gain_ms(pool, spec) <= 0.0:
                reasons[tier] = "nonpositive_recovery_gain"
            elif pool.used_bytes + spec.byte_size > pool.limit_bytes:
                reasons[tier] = "coupled_fcfs_insufficient_direct_capacity"
        if not reasons:
            decision = AegisIndependentTierController.admit_chunk(self, spec)
            if not (decision.hbm.admitted and decision.dram.admitted):
                raise AssertionError("coupled preflight disagreed with direct admission")
            self.coupled_success_count += 1
            self._emit(
                "coupled_tier_ablation_decision",
                request_id=spec.request_id,
                ordinal=spec.ordinal,
                admitted=True,
                reason="both_direct_capacity",
            )
            return decision

        self.coupled_rejection_count += 1
        reason = ";".join(
            f"{tier.value}:{reasons.get(tier, 'peer_tier_blocked')}"
            for tier in (ProtectionTier.HBM, ProtectionTier.DRAM)
        )
        admissions = {}
        for tier in (ProtectionTier.HBM, ProtectionTier.DRAM):
            pool = self._pools[tier]
            pool.admission_attempt_count += 1
            pool.admission_rejection_count += 1
            pool.rejection_reason_counts[reason] = (
                pool.rejection_reason_counts.get(reason, 0) + 1
            )
            admissions[tier] = self._disabled_admission(
                tier, reason="coupled_tier_all_or_nothing_rejection"
            )
        decision = PlacementDecision(
            request_id=spec.request_id,
            ordinal=spec.ordinal,
            hbm=admissions[ProtectionTier.HBM],
            dram=admissions[ProtectionTier.DRAM],
        )
        self._emit(
            "coupled_tier_ablation_decision",
            request_id=spec.request_id,
            ordinal=spec.ordinal,
            admitted=False,
            reason=reason,
        )
        return decision


class S4S5Replay(v313.CausalPhysicalTargetReplay):
    """Frozen v3.1.3 replay plus isolated S4 policies and S5 snapshots."""

    SUPPORTED_POLICIES = frozenset({
        "Native Mooncake",
        "AegisKV",
        "AegisKV no Algorithm 1",
        "AegisKV no Algorithm 2",
        "AegisKV no Algorithm 3",
    })

    @property
    def is_aegis(self) -> bool:
        return self.policy in {
            "AegisKV",
            "AegisKV no Algorithm 1",
            "AegisKV no Algorithm 2",
            "AegisKV no Algorithm 3",
        }

    def __init__(
        self, protocol: dict[str, Any], policy: str,
        capacity_protocol: dict[str, Any],
    ) -> None:
        super().__init__(protocol, policy, capacity_protocol)
        if policy == "AegisKV no Algorithm 2":
            self.aegis = [
                CoupledFCFSTierAblationController(controller.config)
                for controller in self.aegis
            ]
        self.rich_control_trace: list[dict[str, Any]] = []

    def start(self, task: core.Task, at: float) -> None:
        if (
            self.is_aegis
            and self.policy != "AegisKV no Algorithm 1"
            and task.predicted_slack_ms > 0.0
        ):
            self.slack_nonzero_request_count += 1
            self.slack_used_total_ms += float(task.predicted_slack_ms)
        super().start(task, at)

    def _rich_control_snapshot(
        self, at: float, *, external: dict[int, bool], before_backup: int,
        released: int, waiting_before: int, active_before: int,
        dispatched: int,
    ) -> None:
        snapshots = [controller.telemetry_snapshot() for controller in self.aegis]
        hbm = [row["pools"]["HBM"] for row in snapshots]
        dram = [row["pools"]["DRAM"] for row in snapshots]
        event_counts = [row.get("event_counts", {}) for row in snapshots]
        control_row = self.control_trace[-1] if self.control_trace else {}
        self.rich_control_trace.append({
            "time_ms": float(at),
            "alpha_after_mean": control_row.get("alpha_after_mean"),
            "pressure_fraction": control_row.get("pressure_fraction"),
            "external_memory_pressure_fraction": control_row.get(
                "external_memory_pressure_fraction"
            ),
            "realized_active_requests_per_gpu": active_before / self.dp,
            "active_after": sum(len(rows) for rows in self.active_by_rank.values()),
            "waiting_before": int(waiting_before),
            "waiting_after": sum(len(rows) for rows in self.waiting),
            "serving_hbm_bytes_total": sum(self.active_serving_bytes),
            "peer_hbm_backup_before_bytes_total": int(before_backup),
            "peer_hbm_backup_after_bytes_total": sum(
                self.hbm_backup_used_bytes(rank) for rank in range(self.dp)
            ),
            "dram_used_bytes_total": sum(int(row["used_bytes"]) for row in dram),
            "peer_hbm_quota_bytes_total": sum(int(row["limit_bytes"]) for row in hbm),
            "shared_serving_plus_backup_budget_bytes_total": self.shared_budget_bytes * self.dp,
            "hbm_admission_success_count": sum(
                int(row["churn"]["admission_success_count"]) for row in hbm
            ),
            "hbm_admission_rejection_count": sum(
                int(row["churn"]["admission_rejection_count"]) for row in hbm
            ),
            "dram_admission_success_count": sum(
                int(row["churn"]["admission_success_count"]) for row in dram
            ),
            "dram_admission_rejection_count": sum(
                int(row["churn"]["admission_rejection_count"]) for row in dram
            ),
            "hbm_eviction_count": sum(int(row["churn"]["eviction_count"]) for row in hbm),
            "hbm_evicted_bytes": sum(int(row["churn"]["evicted_bytes"]) for row in hbm),
            "primary_priority_reclaimed_bytes": int(self.primary_priority_reclaimed_bytes),
            "aimd_backup_released_bytes_this_window": int(released),
            "hbm_admission_resumed_event_count": sum(
                int(row.get("hbm_admission_resumed", 0)) for row in event_counts
            ),
            "dispatched_after_control": int(dispatched),
            "external_memory_pressure_rank_count": sum(external.values()),
        })

    def summarize_mechanism(self, control_horizon_ms: float) -> dict[str, Any]:
        mechanism = super().summarize_mechanism(control_horizon_ms)
        for rank in range(self.dp):
            self.integrate_occupancy(rank, float(control_horizon_ms))
        coupled = [
            controller for controller in self.aegis
            if isinstance(controller, CoupledFCFSTierAblationController)
        ]
        mechanism["active_hbm_v313_s4s5"] = {
            "algorithm2_ablation_semantics": (
                "same Aegis controller/capacity/AIMD and causal source copies; "
                "independent per-tier SlackLandlord admission replaced by coupled "
                "FCFS all-or-nothing admission with direct capacity only"
            ),
            "algorithm2_ablation_is_static_baseline": False,
            "coupled_tier_attempt_count": sum(
                row.coupled_attempt_count for row in coupled
            ),
            "coupled_tier_success_count": sum(
                row.coupled_success_count for row in coupled
            ),
            "coupled_tier_rejection_count": sum(
                row.coupled_rejection_count for row in coupled
            ),
            "rich_control_trace": list(self.rich_control_trace),
            "capacity_integral_byte_ms": dict(self.occupancy_area_byte_ms),
            "capacity_integral_gib_seconds_total": {
                tier: value / GIB / 1000.0
                for tier, value in self.occupancy_area_byte_ms.items()
            },
            "occupancy_time_mean_gib_per_gpu": {
                tier: value / max(float(control_horizon_ms), 1e-12) / GIB / self.dp
                for tier, value in self.occupancy_area_byte_ms.items()
            },
        }
        return mechanism

    def run_coupled(
        self, nominal_tasks: list[core.Task], probes: list[core.FaultProbe], *,
        control_horizon_ms: float,
    ) -> tuple[list[core.Task], list[dict[str, float]], dict[str, Any], list[dict[str, Any]]]:
        self.task_by_id = {row.request_id: row for row in nominal_tasks}
        self.occupancy_observation_horizon_ms = float(control_horizon_ms)
        for rank in range(self.dp):
            self.record_capacity(rank, 0.0, "initial")
        for row in sorted(nominal_tasks, key=lambda item: (item.arrival_ms, item.request_id)):
            self.push(row.arrival_ms, 1, "coupled_arrival", row)
        for probe in probes:
            self.push(probe.time_ms, 2, "probe", probe)
        window = float(self.protocol["aegiskv"]["window_ms"])
        at = window
        while at <= float(control_horizon_ms) + 1e-9:
            self.scheduled_control_opportunity_times_ms.append(float(at))
            self.push(at, 3, "coupled_control", None)
            at += window
        recoveries: list[dict[str, float]] = []
        while self.events:
            at, _priority, _sequence, kind, payload = heapq.heappop(self.events)
            if kind == "aegis_copy":
                self.complete_copy(payload, at)
            elif kind == "protection_seal":
                self.handle_protection_seal(payload, at)
            elif kind == "coupled_arrival":
                task = payload
                self.waiting[task.source_rank].append(task)
                self.try_dispatch(task.source_rank, at, event="arrival")
            elif kind == "coupled_finish":
                task = payload
                rank = task.source_rank
                if task.request_id not in self.active_by_rank.get(rank, set()):
                    continue
                self.active_by_rank[rank].discard(task.request_id)
                self.active_serving_bytes[rank] -= self.serving_bytes_by_request.pop(task.request_id)
                heapq.heappush(self.free_lanes[rank], task.lane)
                self.finish(task, at)
                self.record_capacity(rank, at, "finish_release")
                self._retry_all(at, "finish")
            elif kind == "coupled_control":
                external = {
                    rank: self.dispatch_block_reason(rank, at) == "SHARED_HBM"
                    for rank in range(self.dp)
                }
                before = sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
                waiting_before = sum(len(rows) for rows in self.waiting)
                active_before = sum(len(rows) for rows in self.active_by_rank.values())
                self.control(
                    at,
                    {rank: len(rows) for rank, rows in self.active_by_rank.items()},
                    external_pressure_by_rank=external,
                )
                after = sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
                released = max(before - after, 0)
                self.backup_bytes_released_by_control += released
                dispatched = self._retry_all(at, "control")
                self.dispatch_at_control_count += dispatched
                self.control_capacity_trace.append({
                    "time_ms": float(at),
                    "memory_pressure_rank_count": sum(external.values()),
                    "backup_before_bytes": before,
                    "backup_after_bytes": after,
                    "backup_released_bytes": released,
                    "waiting_before": waiting_before,
                    "waiting_after": sum(len(rows) for rows in self.waiting),
                    "active_before": active_before,
                    "active_after": sum(len(rows) for rows in self.active_by_rank.values()),
                    "dispatched_after_control": dispatched,
                })
                if self.is_aegis and self.policy != "AegisKV no Algorithm 3":
                    self._rich_control_snapshot(
                        at, external=external, before_backup=before,
                        released=released, waiting_before=waiting_before,
                        active_before=active_before, dispatched=dispatched,
                    )
            elif kind == "probe":
                probe = payload
                delays = {}
                for rank in v313.v312.v31.failure_domain_ranks(
                    probe,
                    gpus_per_node=int(self.protocol["hardware"]["gpus_per_node"]),
                    dp=self.dp,
                ):
                    for request_id in sorted(self.active_by_rank.get(rank, ())):
                        task = self.task_by_id[request_id]
                        delay, _action = self.recovery(task, probe)
                        delays[request_id] = delay
                recoveries.append(delays)
            else:
                raise AssertionError(f"unknown v3.1.3 S4/S5 event {kind!r}")
        if any(self.waiting) or any(self.active_by_rank.values()):
            raise AssertionError("v3.1.3 S4/S5 scheduler failed to drain")
        if len(self.scheduled_tasks) != len(nominal_tasks):
            raise AssertionError("v3.1.3 S4/S5 scheduler lost requests")
        mechanism = self.summarize_mechanism(control_horizon_ms)
        return (
            sorted(self.scheduled_tasks, key=lambda row: (row.start_ms, row.request_id)),
            recoveries,
            mechanism,
            list(self.capacity_trace),
        )
