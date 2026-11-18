"""Common machine-readable telemetry and online/offline gate helpers."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples) if samples else 0.0,
        "p50": percentile(samples, 0.50),
        "p90": percentile(samples, 0.90),
        "p99": percentile(samples, 0.99),
        "max": max(samples, default=0.0),
    }


def build_common_calibration_metrics(
    *,
    request_stats: Mapping[str, Mapping[str, Any]],
    scheduler_telemetry: Iterable[Mapping[str, Any]],
    failure_events: Iterable[Mapping[str, Any]] = (),
    aegis_events: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the schema consumed by the online-vs-offline gate."""

    normalized_requests = {
        str(rid): dict(stats) for rid, stats in request_stats.items()
    }
    status_counts = Counter(
        str(stats.get("status", "unknown"))
        for stats in normalized_requests.values()
    )
    response_times: list[float] = []
    queue_times: list[float] = []
    service_times: list[float] = []
    ttft: list[float] = []
    itl: list[float] = []
    completion_by_rank: dict[int, list[tuple[float, str]]] = defaultdict(list)
    routes: dict[str, int] = {}

    created_times: list[float] = []
    completed_times: list[float] = []
    for rid, stats in normalized_requests.items():
        created = float(stats.get("created_time", -1.0))
        completed = float(stats.get("last_event_time", -1.0))
        if created >= 0.0:
            created_times.append(created)
        if completed >= 0.0:
            completed_times.append(completed)
        if created >= 0.0 and completed >= created:
            response_times.append(completed - created)
        queue_start = float(stats.get("queue_start", -1.0))
        queue_end = float(stats.get("queue_end", -1.0))
        if queue_start >= 0.0 and queue_end >= queue_start:
            queue_times.append(queue_end - queue_start)
        service_times.append(max(float(stats.get("service_time_s", 0.0)), 0.0))
        token_latencies = [
            max(float(value), 0.0)
            for value in stats.get("gen_token_latencies", ())
        ]
        if token_latencies:
            ttft.append(token_latencies[0])
            itl.extend(token_latencies[1:])
        rank = int(stats.get("aegis_assigned_dp_rank", -1))
        routes[rid] = rank
        if rank >= 0 and completed >= 0.0:
            completion_by_rank[rank].append((completed, rid))

    telemetry = [dict(row) for row in scheduler_telemetry]
    accounting: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {
            "batch_count": 0,
            "forward_s": 0.0,
            "h2d_s": 0.0,
            "backup_pending_s": 0.0,
            "pipeline_base_s": 0.0,
            "cpu_overflow_s": 0.0,
            "comm_h2d_visible_s": 0.0,
            "comm_d2h_remainder_s": 0.0,
        }
    )
    accounting_by_stage: dict[
        tuple[int, int], dict[str, float | int]
    ] = defaultdict(
        lambda: {
            "batch_count": 0,
            "forward_s": 0.0,
            "h2d_s": 0.0,
            "backup_pending_s": 0.0,
            "pipeline_base_s": 0.0,
            "cpu_overflow_s": 0.0,
            "comm_h2d_visible_s": 0.0,
            "comm_d2h_remainder_s": 0.0,
        }
    )
    queue_depth_by_rank: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    queue_depth_by_stage: dict[
        tuple[int, int], list[dict[str, float | int]]
    ] = defaultdict(list)
    for row in telemetry:
        rank = int(row.get("dp_rank", -1))
        pp_rank = int(row.get("pp_rank", 0))
        if rank < 0:
            continue
        if str(row.get("batch_mode", "idle")) != "idle":
            accounting[rank]["batch_count"] += 1
            accounting_by_stage[(rank, pp_rank)]["batch_count"] += 1
        for field in (
            "forward_s",
            "h2d_s",
            "backup_pending_s",
            "pipeline_base_s",
            "cpu_overflow_s",
            "comm_h2d_visible_s",
            "comm_d2h_remainder_s",
        ):
            value = float(row.get(field, 0.0))
            accounting[rank][field] += value
            accounting_by_stage[(rank, pp_rank)][field] += value
        depth = {
            "timestamp_s": float(row.get("completed_at", 0.0)),
            "waiting": int(
                row.get("waiting_depth_after", row.get("waiting_depth", 0))
            ),
            "running": int(
                row.get("running_depth_after", row.get("running_depth", 0))
            ),
        }
        queue_depth_by_stage[(rank, pp_rank)].append(depth)
        # PP0 owns external admission and is the unambiguous DP queue view.
        if pp_rank == 0:
            queue_depth_by_rank[rank].append(dict(depth))

    stage_forward_by_dp: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (rank, pp_rank), values in accounting_by_stage.items():
        stage_forward_by_dp[rank].append(
            (pp_rank, float(values["pipeline_base_s"]))
        )

    event_counts = Counter(
        str(event.get("kind", event.get("type", "unknown")))
        for event in aegis_events
    )
    normalized_failures = [dict(event) for event in failure_events]
    failure_order = [str(event.get("event_id", "")) for event in normalized_failures]
    makespan = (
        max(completed_times) - min(created_times)
        if created_times and completed_times
        else 0.0
    )
    return {
        "schema_version": 1,
        "request_count": len(normalized_requests),
        "status_counts": dict(sorted(status_counts.items())),
        "routes": dict(sorted(routes.items())),
        "response_time_s": metric_summary(response_times),
        "queue_time_s": metric_summary(queue_times),
        "service_time_s": metric_summary(service_times),
        "ttft_s": metric_summary(ttft),
        "itl_s": metric_summary(itl),
        "makespan_s": makespan,
        "completion_order_by_dp": {
            str(rank): [rid for _timestamp, rid in sorted(rows)]
            for rank, rows in sorted(completion_by_rank.items())
        },
        "queue_depth_by_dp": {
            str(rank): rows for rank, rows in sorted(queue_depth_by_rank.items())
        },
        "queue_depth_by_stage": {
            f"dp{rank}.pp{pp_rank}": rows
            for (rank, pp_rank), rows in sorted(queue_depth_by_stage.items())
        },
        "pipeline_accounting_by_dp": {
            str(rank): dict(values) for rank, values in sorted(accounting.items())
        },
        "pipeline_accounting_by_stage": {
            f"dp{rank}.pp{pp_rank}": dict(values)
            for (rank, pp_rank), values in sorted(accounting_by_stage.items())
        },
        "pipeline_critical_path_by_dp": {
            str(rank): {
                "stage_count": len(rows),
                "max_stage_pipeline_base_s": max(
                    (value for _pp_rank, value in rows),
                    default=0.0,
                ),
                "bottleneck_pp_rank": max(
                    rows,
                    key=lambda item: (item[1], -item[0]),
                    default=(-1, 0.0),
                )[0],
            }
            for rank, rows in sorted(stage_forward_by_dp.items())
        },
        "aegis_event_counts": dict(sorted(event_counts.items())),
        "failure_event_order": failure_order,
        "failure_events": normalized_failures,
    }


def build_offline_cluster_artifact(
    result,
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request_stats: dict[str, dict[str, Any]] = {}
    for snapshot in result.rank_snapshots.values():
        request_stats.update(
            {
                str(rid): dict(stats)
                for rid, stats in snapshot.get("request_stats", {}).items()
            }
        )
    metrics = build_common_calibration_metrics(
        request_stats=request_stats,
        scheduler_telemetry=result.telemetry,
        failure_events=result.failure_events,
        aegis_events=result.aegis_status.get("events", ()),
    )
    physical_stage_snapshots: dict[str, dict[str, Any]] = {}
    logical_clocks_by_stage: dict[str, float] = {}
    pp_sizes: set[int] = set()
    process_count = 0
    for dp_rank, snapshot in sorted(result.rank_snapshots.items()):
        stages = snapshot.get("stages")
        if not isinstance(stages, Mapping):
            stages = {"0": snapshot}
        pp_sizes.add(len(stages))
        process_count += int(snapshot.get("process_count", len(stages)))
        for pp_rank, stage_snapshot in sorted(
            stages.items(), key=lambda item: int(item[0])
        ):
            if not isinstance(stage_snapshot, Mapping):
                continue
            key = f"dp{int(dp_rank)}.pp{int(pp_rank)}"
            physical_stage_snapshots[key] = dict(stage_snapshot)
            logical_clocks_by_stage[key] = float(
                stage_snapshot.get("logical_now_s", 0.0)
            )

    return {
        "schema_version": 1,
        "simulation_time_mode": "logical",
        "evidence_tier": "sglsim-offline-unvalidated",
        "offline_accuracy_gate_passed": False,
        "identity": dict(identity or {}),
        "calibration_metrics": metrics,
        "request_stats": request_stats,
        "scheduler_telemetry": list(result.telemetry),
        "physical_topology": {
            "topology_contract": "one_scheduler_process_per_dp_pp_tp_stage",
            "dp_size": len(result.rank_snapshots),
            "pp_size": next(iter(pp_sizes)) if len(pp_sizes) == 1 else None,
            "tp_size": 1,
            "expected_process_count": (
                len(result.rank_snapshots) * next(iter(pp_sizes))
                if len(pp_sizes) == 1
                else None
            ),
            "process_count": process_count,
            "stage_count": len(physical_stage_snapshots),
            "stage_ids": [
                f"{stage_id}.tp0" for stage_id in sorted(physical_stage_snapshots)
            ],
            "logical_clocks_by_stage_s": logical_clocks_by_stage,
        },
        "physical_stage_snapshots": physical_stage_snapshots,
        "aegis_status": dict(result.aegis_status),
    }
