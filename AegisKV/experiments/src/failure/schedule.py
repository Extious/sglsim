"""Derive policy-independent conditional fault schedules from a shadow run."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from sglsim.srt.simulation.aegis_complete import CompleteFailure, CompleteWorker


@dataclass(frozen=True)
class FailureSchedule:
    failures: tuple[CompleteFailure, ...]
    metadata: dict[str, Any]
    slack_refresh_request_ids: tuple[str, ...] = ()


def _active_workers_at(
    shadow_result: Mapping[str, Any], time_ms: float
) -> dict[str, str]:
    starts: dict[str, tuple[float, str]] = {}
    active: dict[str, str] = {}
    for event in shadow_result.get("events") or []:
        request_id = str(event.get("request_id", ""))
        if event.get("kind") == "request_execution_started":
            starts[request_id] = (float(event["time_ms"]), str(event["source_worker"]))
        elif event.get("kind") == "request_completed" and request_id in starts:
            start, worker_id = starts.pop(request_id)
            if start <= float(time_ms) <= float(event["time_ms"]):
                active[worker_id] = request_id
    return active


def _active_interval(
    shadow_result: Mapping[str, Any], *, mode: str, failure_time_fraction: float
) -> tuple[float, str, str]:
    starts: dict[str, tuple[float, str]] = {}
    intervals: list[tuple[float, float, str, str]] = []
    for event in shadow_result.get("events") or []:
        kind = event.get("kind")
        request_id = str(event.get("request_id", ""))
        if kind == "request_execution_started":
            starts[request_id] = (float(event["time_ms"]), str(event["source_worker"]))
        elif kind == "request_completed" and request_id in starts:
            start, worker = starts.pop(request_id)
            end = float(event["time_ms"])
            if end > start:
                intervals.append((start, end, worker, request_id))
    if not intervals:
        raise RuntimeError("shadow schedule contains no active interval")
    workflow_tail: dict[str, str] = {}
    for request_id, request in shadow_result["requests"].items():
        workflow_id = str(request["workflow_id"])
        previous = workflow_tail.get(workflow_id)
        if previous is None or float(request["completion_ms"]) > float(
            shadow_result["requests"][previous]["completion_ms"]
        ):
            workflow_tail[workflow_id] = str(request_id)
    if mode == "critical_tail":
        tail_ids = set(workflow_tail.values())
        tail_intervals = [row for row in intervals if row[3] in tail_ids]
        if tail_intervals:
            intervals = tail_intervals
        target = sorted(
            (start + end) / 2.0 for start, end, _worker, _rid in intervals
        )[len(intervals) // 2]
    elif mode == "early_active":
        maximum_arrival = max(
            float(event["time_ms"])
            for event in shadow_result.get("events") or []
            if event.get("kind") == "request_arrived_and_route_bound"
        )
        target = min(max(float(failure_time_fraction), 0.0), 1.0) * maximum_arrival
        active = [row for row in intervals if row[0] <= target <= row[1]]
        if active:
            intervals = active
    else:
        raise ValueError(f"unknown failure trigger mode {mode!r}")
    start, end, worker, request_id = min(
        intervals, key=lambda row: abs(((row[0] + row[1]) / 2.0) - target)
    )
    time_ms = min(max(target, start + 1e-6), end - 1e-6)
    return time_ms, worker, request_id


def build_failure_schedule(
    shadow_result: Mapping[str, Any],
    failure_config: Mapping[str, Any],
    axes: Mapping[str, Any],
    workers: Sequence[CompleteWorker],
) -> FailureSchedule:
    scenario = str(axes.get("failure_scenario", failure_config["fault_model"]["central_scenario"]))
    repair_ms = float(axes.get("repair_time_ms", failure_config["fault_model"]["repair_time_ms"]))
    if scenario == "no_failure":
        return FailureSchedule((), {"scenario": scenario, "conditional": False}, ())
    fault_model = failure_config["fault_model"]
    failure_time_ms, active_worker, active_request = _active_interval(
        shadow_result,
        mode=str(fault_model.get("trigger_mode", "critical_tail")),
        failure_time_fraction=float(fault_model.get("failure_time_fraction", 0.50)),
    )
    by_id = {worker.worker_id: worker for worker in workers}
    if active_worker not in by_id:
        raise RuntimeError("shadow run selected an unknown worker")
    ordered = [worker.worker_id for worker in workers]
    if scenario.startswith("worker_"):
        count = int(scenario.split("_", 1)[1])
        active_by_worker = _active_workers_at(shadow_result, failure_time_ms)
        active_candidates = sorted(
            worker for worker in active_by_worker if worker != active_worker
        )
        inactive_candidates = [
            worker
            for worker in ordered
            if worker != active_worker and worker not in active_candidates
        ]
        selected = [active_worker] + (active_candidates + inactive_candidates)[
            : max(count - 1, 0)
        ]
        domain = "replica"
    elif scenario == "node_1":
        failed_domain = by_id[active_worker].failure_domain
        selected = [
            worker.worker_id for worker in workers if worker.failure_domain == failed_domain
        ]
        domain = "node"
    else:
        raise ValueError(f"unknown failure scenario {scenario!r}")
    ordering_stress = str(failure_config["fault_model"].get("ordering_stress", "none"))
    selected_slack_ms: dict[str, float] = {}
    selected_active_request: dict[str, str] = {}
    ordering_adversary_adjustment_ms = 0.0
    if ordering_stress == "largest_slack_first_fifo_adversary":
        if not scenario.startswith("worker_"):
            raise ValueError(
                "largest-slack FIFO adversary requires a worker_N scenario"
            )
        slack_by_request = {
            str(event["request_id"]): float(event["causal_slack_ms"])
            for event in shadow_result.get("events") or []
            if event.get("kind") == "request_arrived_and_route_bound"
        }
        active_starts: dict[str, tuple[float, str]] = {}
        active_intervals: list[tuple[float, float, str, str]] = []
        for event in shadow_result.get("events") or []:
            request_id = str(event.get("request_id", ""))
            if event.get("kind") == "request_execution_started":
                active_starts[request_id] = (
                    float(event["time_ms"]), str(event["source_worker"])
                )
            elif event.get("kind") == "request_completed" and request_id in active_starts:
                start, source = active_starts.pop(request_id)
                active_intervals.append(
                    (start, float(event["time_ms"]), source, request_id)
                )
        boundaries = sorted(
            {value for start, end, _source, _request in active_intervals for value in (start, end)}
        )
        adversarial_points: list[
            tuple[float, float, tuple[tuple[str, str, float], ...]]
        ] = []
        for left, right in zip(boundaries, boundaries[1:]):
            if right - left <= 2e-6:
                continue
            candidate_time_ms = (left + right) / 2.0
            active_by_worker: dict[str, tuple[str, float]] = {}
            for start, end, source, request_id in active_intervals:
                if not (start <= candidate_time_ms <= end):
                    continue
                slack_ms = slack_by_request.get(request_id, 0.0)
                previous = active_by_worker.get(source)
                if previous is None or slack_ms > previous[1]:
                    active_by_worker[source] = (request_id, slack_ms)
            ranked = sorted(
                (
                    (worker_id, request_id, slack_ms)
                    for worker_id, (request_id, slack_ms) in active_by_worker.items()
                ),
                key=lambda row: (-row[2], row[0], row[1]),
            )[:count]
            if len(ranked) != count or ranked[0][2] <= ranked[-1][2] + 1e-12:
                continue
            adversarial_points.append(
                (abs(candidate_time_ms - failure_time_ms), candidate_time_ms, tuple(ranked))
            )
        if not adversarial_points:
            raise RuntimeError(
                "controlled FIFO adversary has no overlap with distinct causal slack"
            )
        _distance, adversarial_time_ms, ranked = min(adversarial_points)
        ordering_adversary_adjustment_ms = adversarial_time_ms - failure_time_ms
        failure_time_ms = adversarial_time_ms
        selected = [worker_id for worker_id, _request_id, _slack in ranked]
        selected_active_request = {
            worker_id: request_id for worker_id, request_id, _slack in ranked
        }
        selected_slack_ms = {
            worker_id: slack_ms for worker_id, _request_id, slack_ms in ranked
        }
        active_worker = selected[0]
        active_request = selected_active_request[active_worker]
    elif ordering_stress != "none":
        raise ValueError(f"unknown ordering_stress {ordering_stress!r}")
    failures = tuple(
        CompleteFailure(failure_time_ms, worker_id, repair_after_ms=repair_ms)
        for worker_id in selected
    )
    return FailureSchedule(
        failures,
        {
            "scenario": scenario,
            "conditional": True,
            "trigger": str(fault_model["trigger"]),
            "estimand": (
                "conditional_active_workflow_critical_tail"
                if fault_model.get("trigger_mode") == "critical_tail"
                else "conditional_early_active_worker"
            ),
            "failure_time_ms": failure_time_ms,
            "active_request": active_request,
            "anchor_worker": active_worker,
            "failed_workers": selected,
            "failed_worker_slack_ms": selected_slack_ms,
            "failed_worker_active_request": selected_active_request,
            "ordering_stress": ordering_stress,
            "ordering_adversary_adjustment_ms": ordering_adversary_adjustment_ms,
            "fault_domain": domain,
            "repair_time_ms": repair_ms,
            "paired_policy_independent": True,
        },
        tuple(selected_active_request.values()),
    )


def operational_failure_probability(
    *, duration_ms: float, gpu_count: int, mttf_gpu_hours: float
) -> float:
    exposure_hours = max(float(duration_ms), 0.0) / 3_600_000.0 * max(int(gpu_count), 0)
    return 1.0 - math.exp(-exposure_hours / max(float(mttf_gpu_hours), 1e-12))


__all__ = ["FailureSchedule", "build_failure_schedule", "operational_failure_probability"]
