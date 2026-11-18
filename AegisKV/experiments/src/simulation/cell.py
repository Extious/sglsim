"""Run one matrix cell through the unified AegisKV complete metadata DES."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from sglsim.srt.simulation.aegis_complete import (
    AegisCompleteSimulator,
    CompleteSlackUpdate,
)

from ..config import EXPERIMENT_ROOT, GroupConfig, canonical_json_bytes, sha256_bytes
from ..deployment import DeploymentRuntime, build_deployment
from ..failure import build_failure_schedule, operational_failure_probability
from ..request import Workload, build_workload


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _workflow_rows(
    workload: Workload,
    result: Mapping[str, Any],
    shadow: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in workload.rows:
        grouped[row.workflow_id].append(row)
    observations = []
    for workflow_id, rows in sorted(grouped.items()):
        request_ids = [row.request_id for row in rows]
        arrival = min(row.arrival_ms for row in rows)
        completion = max(float(result["requests"][rid]["completion_ms"]) for rid in request_ids)
        shadow_completion = max(float(shadow["requests"][rid]["completion_ms"]) for rid in request_ids)
        wct_ms = completion - arrival
        shadow_wct_ms = shadow_completion - arrival
        signed_delta = wct_ms - shadow_wct_ms
        observations.append(
            {
                "workflow_id": workflow_id,
                "request_count": len(rows),
                "wct_ms": wct_ms,
                "shadow_wct_ms": shadow_wct_ms,
                "signed_wct_delta_ms": signed_delta,
                "failure_added_wct_ms": max(signed_delta, 0.0),
                "mean_reuse_fraction": statistics.fmean(row.reuse_fraction for row in rows),
                "provider": rows[0].provider,
            }
        )
    return observations


def _analytical_recovery_rows(
    result: Mapping[str, Any], deployment: DeploymentRuntime
) -> list[dict[str, Any]]:
    selected = {
        str(event["request_id"]): event
        for event in result.get("events") or []
        if event.get("kind") == "recovery_selected_and_started"
    }
    service = deployment.model["service_time"]
    kv_bytes = int(deployment.model["kv_bytes_per_token"])
    bandwidth_bytes_per_ms = (
        deployment.effective_bandwidth_gbytes_per_s * 1_000_000.0
    )
    load_setup = float(deployment.simulation_config.design.hbm_load.setup_ms)
    rows = []
    for request_id, request in result["requests"].items():
        event = selected.get(request_id)
        if event is None:
            continue
        restored = int(event["restored_tokens"])
        recomputed = int(event["recompute_tokens"])
        analytical = 0.0
        if restored:
            analytical += load_setup + restored * kv_bytes / bandwidth_bytes_per_ms
        if recomputed:
            analytical += float(service["setup_ms"]) + recomputed * float(
                service["prefill_ms_per_uncached_token"]
            )
        simulated = float(event["recovery_ms"])
        rows.append(
            {
                "request_id": request_id,
                "action": event["action"],
                "restored_tokens": restored,
                "recompute_tokens": recomputed,
                "analytical_recovery_ms": analytical,
                "simulated_recovery_ms": simulated,
                "absolute_error_ms": abs(analytical - simulated),
            }
        )
    return rows


@dataclass(frozen=True)
class CellResult:
    cell_id: str
    summary: dict[str, Any]
    observations: tuple[dict[str, Any], ...]
    timeline: tuple[dict[str, Any], ...]
    verification: tuple[dict[str, Any], ...]
    raw_result: dict[str, Any]
    shadow_result: dict[str, Any]
    workload_metadata: dict[str, Any]
    failure_metadata: dict[str, Any]


def build_inputs(
    group: GroupConfig,
    axes: Mapping[str, Any],
    *,
    suite_profile: Mapping[str, Any],
    experiment_root: Path = EXPERIMENT_ROOT,
) -> tuple[DeploymentRuntime, Workload, tuple[Any, ...]]:
    seed = int(axes["seed"])
    deployment = build_deployment(group.deployment, group.failure, axes, seed=seed)
    workload = build_workload(
        group.request,
        deployment.model,
        axes,
        profile=suite_profile,
        experiment_root=experiment_root,
    )
    requests = workload.complete_requests(deployment.worker_ids, seed=seed)
    return deployment, workload, requests


def _max_time_ms(workload: Workload, deployment: DeploymentRuntime) -> float:
    latest_arrival = max((row.arrival_ms for row in workload.rows), default=0.0)
    total_service = sum(row.service_ms for row in workload.rows)
    return (
        latest_arrival
        + total_service / max(len(deployment.workers), 1)
        + float(deployment.simulation_config.repair_time_ms)
        + 1_000_000.0
    )


def build_shadow_result(
    deployment: DeploymentRuntime,
    workload: Workload,
    requests: tuple[Any, ...],
) -> dict[str, Any]:
    config = replace(
        deployment.simulation_config,
        protection_mode="none",
        recovery_priority="fifo",
        adaptive_hbm_quota=False,
        random_salt="policy-independent-shadow-v2",
    )
    return AegisCompleteSimulator(config).run(
        requests,
        failures=(),
        max_time_ms=_max_time_ms(workload, deployment),
    )


def run_cell(
    group: GroupConfig,
    axes: Mapping[str, Any],
    *,
    suite_profile: Mapping[str, Any],
    shadow_result: dict[str, Any] | None = None,
    experiment_root: Path = EXPERIMENT_ROOT,
) -> CellResult:
    deployment, workload, requests = build_inputs(
        group,
        axes,
        suite_profile=suite_profile,
        experiment_root=experiment_root,
    )
    if shadow_result is None:
        shadow_result = build_shadow_result(deployment, workload, requests)
    schedule = build_failure_schedule(
        shadow_result,
        group.failure,
        axes,
        deployment.workers,
    )
    request_by_id = {request.request_id: request for request in requests}
    slack_updates = tuple(
        CompleteSlackUpdate(
            max(float(schedule.metadata["failure_time_ms"]) - 0.001, 0.0),
            request_id,
            float(request_by_id[request_id].slack_ms),
            request_by_id[request_id].slack_evidence,
        )
        for request_id in schedule.slack_refresh_request_ids
    )
    result = AegisCompleteSimulator(deployment.simulation_config).run(
        requests,
        failures=schedule.failures,
        slack_updates=slack_updates,
        max_time_ms=_max_time_ms(workload, deployment),
    )
    observations = _workflow_rows(workload, result, shadow_result)
    mttf = float(
        axes.get("mttf_gpu_hours", group.failure["hazard_weighting"]["central_mttf_gpu_hours"])
    )
    observation_duration_ms = max(
        max((row.arrival_ms for row in workload.rows), default=0.0),
        max((float(row["completion_ms"]) for row in shadow_result["requests"].values()), default=0.0),
    )
    gpu_count = int(group.deployment["cluster"]["node_count"]) * int(
        group.deployment["cluster"]["gpus_per_node"]
    )
    hazard_probability = operational_failure_probability(
        duration_ms=observation_duration_ms,
        gpu_count=gpu_count,
        mttf_gpu_hours=mttf,
    )
    for observation in observations:
        observation["hazard_weighted_added_wct_ms"] = (
            observation["failure_added_wct_ms"] * hazard_probability
        )

    arrivals = {row.request_id: row.arrival_ms for row in workload.rows}
    request_latencies = [
        float(row["completion_ms"]) - arrivals[request_id]
        for request_id, row in result["requests"].items()
    ]
    wcts = [float(row["wct_ms"]) for row in observations]
    signed_deltas = [float(row["signed_wct_delta_ms"]) for row in observations]
    added = [float(row["failure_added_wct_ms"]) for row in observations]
    affected_added = [value for value in added if value > 1e-9]
    actions = Counter(
        str(event["action"])
        for event in result.get("events") or []
        if event.get("kind") == "recovery_selected_and_started"
    )
    recovery_due = {
        str(event["request_id"]): float(event["remaining_slack_due_at_ms"])
        for event in result.get("events") or []
        if event.get("kind") == "recovery_selected_and_started"
    }
    recovery_completed = {
        str(event["request_id"]): float(event["time_ms"])
        for event in result.get("events") or []
        if event.get("kind") == "recovery_completed_route_activated"
    }
    recovery_deadline_misses = [
        max(recovery_completed[request_id] - due_at, 0.0)
        for request_id, due_at in recovery_due.items()
        if request_id in recovery_completed
    ]
    timeline = []
    for event in result.get("events") or []:
        if event.get("kind") != "hbm_quota_aimd_window":
            continue
        updates = event.get("target_updates") or []
        timeline.append(
            {
                "time_ms": float(event["time_ms"]),
                "mean_alpha": statistics.fmean(float(row["new_alpha"]) for row in updates)
                if updates
                else math.nan,
                "pressured_targets": sum(bool(row["pressure"]) for row in updates),
                "target_count": len(updates),
            }
        )
    verification = _analytical_recovery_rows(result, deployment)
    identity = {
        "group": group.group,
        "axes": dict(sorted(axes.items())),
        "workload": workload.metadata,
        "failure": schedule.metadata,
    }
    cell_id = sha256_bytes(canonical_json_bytes(identity))[:16]
    summary = {
        "cell_id": cell_id,
        "group": group.group,
        **dict(axes),
        "request_count": len(workload.rows),
        "workflow_count": len(observations),
        "workload_sha256": sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "request_id": row.request_id,
                        "workflow_id": row.workflow_id,
                        "arrival_ms": row.arrival_ms,
                        "prompt_tokens": row.prompt_tokens,
                        "output_tokens": row.output_tokens,
                        "cached_prefix_tokens": row.cached_prefix_tokens,
                        "service_ms": row.service_ms,
                        "prefill_service_ms": row.prefill_service_ms,
                        "slack_ms": row.slack_ms,
                    }
                    for row in workload.rows
                ]
            )
        ),
        "request_latency_p50_ms": _quantile(request_latencies, 0.50),
        "request_latency_p95_ms": _quantile(request_latencies, 0.95),
        "request_latency_p99_ms": _quantile(request_latencies, 0.99),
        "workflow_wct_mean_ms": statistics.fmean(wcts) if wcts else math.nan,
        "workflow_wct_p50_ms": _quantile(wcts, 0.50),
        "workflow_wct_p95_ms": _quantile(wcts, 0.95),
        "signed_wct_delta_mean_ms": (
            statistics.fmean(signed_deltas) if signed_deltas else math.nan
        ),
        "failure_added_wct_mean_ms": statistics.fmean(added) if added else math.nan,
        "failure_added_wct_p95_ms": _quantile(added, 0.95),
        "affected_workflow_count": len(affected_added),
        "affected_workflow_fraction": (
            len(affected_added) / len(added) if added else math.nan
        ),
        "affected_failure_added_wct_mean_ms": (
            statistics.fmean(affected_added) if affected_added else 0.0
        ),
        "affected_failure_added_wct_p50_ms": _quantile(affected_added, 0.50)
        if affected_added
        else 0.0,
        "affected_failure_added_wct_p95_ms": _quantile(affected_added, 0.95)
        if affected_added
        else 0.0,
        "hazard_probability": hazard_probability,
        "hazard_weighted_added_wct_mean_ms": (
            statistics.fmean(added) * hazard_probability if added else math.nan
        ),
        "recovery_count": sum(actions.values()),
        "recovery_actions": dict(sorted(actions.items())),
        "scheduled_failure_count": len(schedule.failures),
        "failed_worker_count": len(schedule.metadata.get("failed_workers") or []),
        "failed_workers": list(schedule.metadata.get("failed_workers") or []),
        "fault_domain": schedule.metadata.get("fault_domain", "none"),
        "failure_time_ms": schedule.metadata.get("failure_time_ms", math.nan),
        "recovery_deadline_miss_max_ms": max(recovery_deadline_misses, default=0.0),
        "recovery_deadline_miss_mean_ms": (
            statistics.fmean(recovery_deadline_misses)
            if recovery_deadline_misses
            else 0.0
        ),
        "target_rotation_count": int(result["event_counts"].get("standby_target_rotated", 0)),
        "queue_reschedule_count": int(result["event_counts"].get("queue_outage_rescheduled", 0)),
        "simulation_end_ms": float(result["simulation_end_ms"]),
        "metadata_only": True,
        "physical_result": False,
        "physical_calibration": False,
        "service_time_status": deployment.model["service_time"]["status"],
        "complete_failure_path_exercised": bool(
            result["event_counts"].get("source_failure_fenced", 0)
            and result["event_counts"].get("recovery_completed_route_activated", 0)
        )
        if schedule.failures
        else False,
        "analytical_recovery_max_abs_error_ms": max(
            (float(row["absolute_error_ms"]) for row in verification), default=0.0
        ),
    }
    tagged_observations = tuple({"cell_id": cell_id, "group": group.group, **dict(axes), **row} for row in observations)
    tagged_timeline = tuple({"cell_id": cell_id, "group": group.group, **dict(axes), **row} for row in timeline)
    tagged_verification = tuple({"cell_id": cell_id, "group": group.group, **dict(axes), **row} for row in verification)
    return CellResult(
        cell_id=cell_id,
        summary=summary,
        observations=tagged_observations,
        timeline=tagged_timeline,
        verification=tagged_verification,
        raw_result=result,
        shadow_result=shadow_result,
        workload_metadata=workload.metadata,
        failure_metadata=schedule.metadata,
    )


__all__ = ["CellResult", "build_inputs", "build_shadow_result", "run_cell"]
