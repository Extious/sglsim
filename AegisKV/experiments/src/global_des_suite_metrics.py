"""Derived metrics for real-Scheduler global-DES experiment cells.

The raw event/resource tables remain authoritative.  This module only derives
small summaries from an :class:`OfflineClusterResult`; it never fabricates a
service time, transfer time, or failure outcome from a static formula.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable, Mapping

from sglsim.srt.simulation.calibration_metrics import (
    build_offline_cluster_artifact,
)


METRICS_SCHEMA = "aegiskv.global-des-scheduler-cell-metrics.v1"
MODELED_EVIDENCE_CLASS = (
    "modeled_offline_global_virtual_clock_real_scheduler_simulation"
)
PRELIMINARY_GROUPS = frozenset({"P11", "P12"})
RECOVERY_ACTIVATION_PATH = "peer_hbm_ownership_activation"


def _peak_by_key(
    rows: Iterable[Mapping[str, Any]],
    *,
    key_field: str,
    used_fields: tuple[str, ...],
) -> dict[str, int]:
    peaks: dict[str, int] = defaultdict(int)
    for row in rows:
        key = str(row.get(key_field) or "")
        if not key:
            continue
        used = sum(max(int(row.get(field, 0)), 0) for field in used_fields)
        peaks[key] = max(peaks[key], used)
    return dict(sorted(peaks.items()))


def _gpu_peak(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    peaks: dict[str, int] = defaultdict(int)
    for row in rows:
        if not all(
            field in row
            for field in ("node_id", "local_gpu_id", "dp_rank", "pp_rank", "tp_rank")
        ):
            continue
        key = (
            f"{row['node_id']}/gpu{int(row['local_gpu_id'])}"
            f"/dp{int(row['dp_rank'])}/pp{int(row['pp_rank'])}/tp{int(row['tp_rank'])}"
        )
        used = sum(
            max(int(row.get(field, 0)), 0)
            for field in (
                "weights",
                "runtime",
                "primary_kv",
                "prefix_kv",
                "aegis_pending",
                "aegis_committed",
            )
        )
        peaks[key] = max(peaks[key], used)
    return dict(sorted(peaks.items()))


def _physical_migration_summary(
    jobs: Iterable[Mapping[str, Any]],
    *,
    id_field: str,
    outcomes: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in jobs]
    path_counts = Counter(str(row.get("path_kind", "unknown")) for row in rows)
    outcome_counts = Counter(
        str(outcomes.get(str(row.get(id_field, "")), "UNKNOWN"))
        for row in rows
    )
    byte_movement = [
        row
        for row in rows
        if str(row.get("path_kind", "")) != RECOVERY_ACTIVATION_PATH
    ]
    aborted_outcomes = {"ABORTED", "FAILED", "CANCELLED", "CANCELED"}
    return {
        "count": len(rows),
        "byte_movement_count": len(byte_movement),
        "ownership_activation_count": len(rows) - len(byte_movement),
        "bytes": sum(max(int(row.get("size_bytes", 0)), 0) for row in byte_movement),
        "path_kind": dict(sorted(path_counts.items())),
        "queue_delay_ns": {
            "sum": sum(max(int(row.get("queue_delay_ns", 0)), 0) for row in rows),
            "max": max(
                (max(int(row.get("queue_delay_ns", 0)), 0) for row in rows),
                default=0,
            ),
        },
        "duration_ns": {
            "sum": sum(max(int(row.get("duration_ns", 0)), 0) for row in rows),
            "max": max(
                (max(int(row.get("duration_ns", 0)), 0) for row in rows),
                default=0,
            ),
        },
        "aborted": sum(
            str(outcomes.get(str(row.get(id_field, "")), "UNKNOWN")).upper()
            in aborted_outcomes
            for row in rows
        ),
        "outcomes": dict(sorted(outcome_counts.items())),
    }


def evidence_claims(
    group: str,
    *,
    smoke: bool,
    calibration_lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return an explicit evidence boundary for one cell.

    P11/P12 are allowed to execute without hardware calibration so that the
    DES/kernel path can be exercised, but such output is unconditionally
    labelled preliminary and is not paper-ready evidence.
    """

    calibration = dict(calibration_lineage or {})
    has_calibration = bool(
        calibration.get("sha256") and calibration.get("path")
    )
    preliminary = bool(smoke or group in PRELIMINARY_GROUPS or not has_calibration)
    return {
        "evidence_class": MODELED_EVIDENCE_CLASS,
        "modeled_simulation": True,
        "real_hardware_measurement": False,
        "real_hardware_calibration_present": has_calibration,
        "calibration_lineage": calibration,
        "smoke_request_subset": bool(smoke),
        "preliminary": preliminary,
        "paper_ready": False,
        "paper_ready_reason": (
            "modeled_smoke_subset"
            if smoke
            else "P11_P12_require_real_hardware_calibration"
            if group in PRELIMINARY_GROUPS and not has_calibration
            else "modeled_result_requires_external_physical_validation"
        ),
    }


def build_scheduler_cell_metrics(
    result: Any,
    *,
    group: str,
    request_rows: Iterable[Mapping[str, Any]],
    smoke: bool,
    calibration_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive scheduler/resource/failure summaries from completed raw state."""

    requests = [dict(row) for row in request_rows]
    common = build_offline_cluster_artifact(
        result,
        identity={"group": str(group)},
    )["calibration_metrics"]
    timeseries = result.global_des_timeseries
    gpu_rows = list(timeseries.get("gpu_hbm", ()))
    dram_rows = list(timeseries.get("node_dram", ()))
    transfers = list(timeseries.get("transfers", ()))
    physical_jobs = list(result.physical_aegis_transfers.get("jobs", ()))
    recovery_jobs = list(
        result.physical_aegis_transfers.get("recovery_jobs", ())
    )
    recovery_outcomes = {
        str(key): str(value)
        for key, value in dict(
            result.physical_aegis_transfers.get("recovery_outcomes", {})
        ).items()
    }
    ordinary_hicache_rows = list(
        result.physical_hicache_transfers.get("transfers", ())
    )
    ordinary_hicache_groups = list(
        result.physical_hicache_transfers.get("groups", ())
    )
    ordinary_hicache_states = Counter(
        str(row.get("state", "UNKNOWN")) for row in ordinary_hicache_rows
    )
    ordinary_hicache_paths = Counter(
        str(row.get("path_kind", row.get("path", "unknown")))
        for row in ordinary_hicache_rows
    )
    ordinary_hicache_directions = Counter(
        str(row.get("direction", "unknown")) for row in ordinary_hicache_rows
    )
    ordinary_hicache_bytes = sum(
        max(int(row.get("size_bytes", 0)), 0)
        for row in ordinary_hicache_rows
    )
    final_transfer_states: dict[str, str] = {}
    for row in transfers:
        transfer_id = str(row.get("transfer_id", ""))
        if transfer_id:
            final_transfer_states[transfer_id] = str(row.get("state", "UNKNOWN"))
    transfer_states = Counter(str(row.get("state", "unknown")) for row in transfers)
    transfer_paths = Counter(str(row.get("path", "unknown")) for row in transfers)
    transfer_bytes = sum(max(int(row.get("size_bytes", 0)), 0) for row in transfers)

    completed_rows = [
        row
        for row in requests
        if str(row.get("status", "")) == "completed"
    ]
    replay_rows = [row for row in requests if bool(row.get("failover_replay"))]
    affected = sum(
        len(tuple(failure.get("affected_request_ids", ())))
        for failure in result.failure_events
    )
    analytical_checks: list[dict[str, Any]] = []
    for phase, job in (
        *(("protection", row) for row in physical_jobs),
        *(("recovery", row) for row in recovery_jobs),
    ):
        if (
            phase == "recovery"
            and str(job.get("path_kind", "")) == RECOVERY_ACTIVATION_PATH
        ):
            continue
        size = int(job.get("size_bytes", 0))
        bandwidth = int(job.get("bottleneck_bytes_per_s", 0))
        setup = int(job.get("setup_ns", 0))
        if size <= 0 or bandwidth <= 0:
            continue
        analytical = setup + math.ceil(size * 1_000_000_000 / bandwidth)
        observed = int(job.get("duration_ns", 0))
        analytical_checks.append(
            {
                "phase": phase,
                "transfer_id": str(
                    job.get("transfer_id") or job.get("recovery_id") or ""
                ),
                "analytical_duration_ns": analytical,
                "des_duration_ns": observed,
                "absolute_error_ns": abs(observed - analytical),
            }
        )
    result_metrics = {
        "schema": METRICS_SCHEMA,
        "group": str(group),
        "claims": evidence_claims(
            str(group),
            smoke=smoke,
            calibration_lineage=calibration_lineage,
        ),
        "scheduler": common,
        "requests": {
            "artifact_rows": len(requests),
            "completed_rows": len(completed_rows),
            "failover_replay_rows": len(replay_rows),
            "failure_affected_original_rows": affected,
        },
        "resources": {
            "gpu_hbm_peak_used_bytes": _gpu_peak(gpu_rows),
            "node_dram_peak_used_bytes": _peak_by_key(
                dram_rows,
                key_field="node_id",
                used_fields=(
                    "runtime",
                    "hicache_primary",
                    "aegis_pending",
                    "aegis_committed",
                ),
            ),
        },
        "transfers": {
            "count": len(transfers),
            "bytes": transfer_bytes,
            "states": dict(sorted(transfer_states.items())),
            "paths": dict(sorted(transfer_paths.items())),
        },
        "physical_migrations": {
            "protection": _physical_migration_summary(
                physical_jobs,
                id_field="transfer_id",
                outcomes=final_transfer_states,
            ),
            "recovery": _physical_migration_summary(
                recovery_jobs,
                id_field="recovery_id",
                outcomes=recovery_outcomes,
            ),
            "ordinary_hicache": {
                "count": len(ordinary_hicache_rows),
                "group_count": len(ordinary_hicache_groups),
                "bytes": ordinary_hicache_bytes,
                "states": dict(sorted(ordinary_hicache_states.items())),
                "paths": dict(sorted(ordinary_hicache_paths.items())),
                "directions": dict(
                    sorted(ordinary_hicache_directions.items())
                ),
            },
            "total_job_count": (
                len(physical_jobs)
                + len(recovery_jobs)
                + len(ordinary_hicache_rows)
            ),
            "total_moved_bytes": (
                sum(max(int(row.get("size_bytes", 0)), 0) for row in physical_jobs)
                + sum(
                    max(int(row.get("size_bytes", 0)), 0)
                    for row in recovery_jobs
                    if str(row.get("path_kind", ""))
                    != RECOVERY_ACTIVATION_PATH
                )
                + ordinary_hicache_bytes
            ),
        },
        "failures": {
            "count": len(result.failure_events),
            "hard_fail_stop_count": sum(
                str(row.get("failure_mode", "")) == "hard_fail_stop"
                for row in result.failure_events
            ),
            "affected_original_requests": affected,
            "event_ids": [
                str(row.get("event_id", "")) for row in result.failure_events
            ],
        },
        "resource_audit_errors": list(result.resource_audit_errors),
    }
    if str(group) == "P12":
        result_metrics["analytical_vs_des"] = {
            "evidence": "modeled_preliminary_not_hardware_calibration",
            "checks": analytical_checks,
            "all_exact": bool(analytical_checks)
            and all(row["absolute_error_ns"] == 0 for row in analytical_checks),
            "max_absolute_error_ns": max(
                (row["absolute_error_ns"] for row in analytical_checks),
                default=0,
            ),
        }
    return result_metrics


__all__ = [
    "METRICS_SCHEMA",
    "MODELED_EVIDENCE_CLASS",
    "PRELIMINARY_GROUPS",
    "build_scheduler_cell_metrics",
    "evidence_claims",
]
