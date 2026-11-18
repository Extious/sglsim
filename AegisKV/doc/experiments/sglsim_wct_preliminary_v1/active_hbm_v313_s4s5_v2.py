"""Mechanism-targeted v3.1.3 extensions for the second S4/S5 pilot.

This module does not change serving, copy, recovery, capacity, or AIMD
semantics.  It adds per-target control telemetry and a preregistered tier
benefit mask used only by the Algorithm-2 mechanism workload.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable

import active_hbm_v313_s4s5 as v1
import run_preliminary as core
from sglsim.srt.simulation.aegis_independent_tiers import (
    AegisIndependentTierController,
    ChunkSpec,
    PlacementDecision,
    ProtectionTier,
)


class CoupledFCFSTierAblationControllerV2(
    v1.CoupledFCFSTierAblationController
):
    """Coupled ablation that fail-closes a one-tier benefit opportunity."""

    def admit_chunk(
        self,
        spec: ChunkSpec,
        *,
        allowed_tiers: Iterable[ProtectionTier] | None = None,
    ) -> PlacementDecision:
        allowed = (
            {ProtectionTier(tier) for tier in allowed_tiers}
            if allowed_tiers is not None
            else {ProtectionTier.HBM, ProtectionTier.DRAM}
        )
        if allowed == {ProtectionTier.HBM, ProtectionTier.DRAM}:
            return super().admit_chunk(spec, allowed_tiers=allowed)

        self._validate_spec(spec)
        self.coupled_attempt_count += 1
        self.coupled_rejection_count += 1
        reason = "coupled_ablation_cannot_take_independent_tier_benefit"
        admissions = {}
        for tier in (ProtectionTier.HBM, ProtectionTier.DRAM):
            pool = self._pools[tier]
            pool.admission_attempt_count += 1
            pool.admission_rejection_count += 1
            pool.rejection_reason_counts[reason] = (
                pool.rejection_reason_counts.get(reason, 0) + 1
            )
            admissions[tier] = self._disabled_admission(tier, reason=reason)
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
            allowed_tiers=sorted(tier.value for tier in allowed),
        )
        return decision


class TargetedReplay(v1.S4S5Replay):
    """S4/S5 replay with per-physical-target AIMD telemetry."""

    def __init__(
        self, protocol: dict[str, Any], policy: str,
        capacity_protocol: dict[str, Any],
    ) -> None:
        super().__init__(protocol, policy, capacity_protocol)
        if policy == "AegisKV no Algorithm 2":
            self.aegis = [
                CoupledFCFSTierAblationControllerV2(controller.config)
                for controller in self.aegis
            ]
        self.target_control_trace: list[dict[str, Any]] = []

    def control(
        self, at: float, active_by_rank: dict[int, int],
        external_pressure_by_rank: dict[int, bool] | None = None,
    ) -> None:
        if not self.is_aegis or self.policy == "AegisKV no Algorithm 3":
            return
        self.control_window_count += 1
        self.control_opportunity_times_ms.append(float(at))
        target_rows = []
        for target, source in sorted(self.source_controller_for_target.items()):
            self.advance_aegis(source, at)
            controller = self.aegis[source]
            pre = controller.telemetry_snapshot()
            pool = pre["pools"]["HBM"]
            occupancy = int(pool["used_bytes"]) / max(int(pool["limit_bytes"]), 1)
            external = bool(
                (external_pressure_by_rank or {}).get(target, False)
            )
            active = int(active_by_rank.get(target, 0))
            pressure = external or active > 2 or occupancy >= 0.95
            self.control_target_evaluations += 1
            self.pressure_target_evaluations += int(pressure)
            result = controller.control_hbm_quota(pressure=pressure)
            self.refresh_occupancy(source, at)
            post = controller.telemetry_snapshot()
            target_rows.append({
                "time_ms": float(at),
                "physical_target_rank": int(target),
                "logical_source_rank": int(source),
                "active_requests": active,
                "external_memory_pressure": external,
                "occupancy_fraction_before": occupancy,
                "pressure": bool(pressure),
                "alpha_before": float(result["old_alpha"]),
                "alpha_after": float(result["new_alpha"]),
                "stable_windows_after": int(result["stable_windows"]),
                "quota_bytes_after": int(result["quota_bytes"]),
                "hbm_used_bytes_after": int(post["pools"]["HBM"]["used_bytes"]),
                "dram_used_bytes_after": int(post["pools"]["DRAM"]["used_bytes"]),
                "evicted_tail_count": len(result["evicted_hbm_tails"]),
                "aborted_pending_count": len(result["aborted_pending"]),
                "admission_resumed": bool(result["admission_resumed"]),
            })
        self.target_control_trace.extend(target_rows)
        self.control_trace.append({
            "time_ms": float(at),
            "alpha_after_mean": statistics.fmean(row["alpha_after"] for row in target_rows),
            "pressure_fraction": statistics.fmean(float(row["pressure"]) for row in target_rows),
            "external_memory_pressure_fraction": statistics.fmean(
                float(row["external_memory_pressure"]) for row in target_rows
            ),
            "active_requests_per_gpu": statistics.fmean(
                row["active_requests"] for row in target_rows
            ),
            "hbm_used_bytes_total": float(sum(row["hbm_used_bytes_after"] for row in target_rows)),
            "dram_used_bytes_total": float(sum(row["dram_used_bytes_after"] for row in target_rows)),
        })
        for target in range(self.dp):
            self._enforce_target_headroom(target, at)

    def summarize_mechanism(self, control_horizon_ms: float) -> dict[str, Any]:
        mechanism = super().summarize_mechanism(control_horizon_ms)
        mechanism["active_hbm_v313_s4s5_v2"] = {
            "target_control_trace": list(self.target_control_trace),
            "pressure_definition": (
                "per physical target: external shared-HBM block OR active_requests > 2 "
                "OR pre-control backup occupancy >= 0.95 of current quota"
            ),
            "target_trace_is_aggregate_inference": False,
        }
        return mechanism


class BenefitMaskedReplay(TargetedReplay):
    """Apply a preregistered recovery-benefit eligibility mask per request."""

    def __init__(
        self, protocol: dict[str, Any], policy: str,
        capacity_protocol: dict[str, Any], *,
        allowed_tiers_by_request: dict[str, tuple[ProtectionTier, ...]],
    ) -> None:
        self.allowed_tiers_by_request = dict(allowed_tiers_by_request)
        super().__init__(protocol, policy, capacity_protocol)

    def handle_protection_seal(
        self, payload: tuple[str, int, int, int], at: float,
    ) -> None:
        request_id = payload[0]
        if not self.is_aegis:
            return super().handle_protection_seal(payload, at)
        controller = self.aegis[self.task_by_id[request_id].source_rank]
        original = controller.admit_chunk
        allowed = self.allowed_tiers_by_request[request_id]
        controller.admit_chunk = lambda spec: original(spec, allowed_tiers=allowed)
        try:
            super().handle_protection_seal(payload, at)
        finally:
            controller.admit_chunk = original
