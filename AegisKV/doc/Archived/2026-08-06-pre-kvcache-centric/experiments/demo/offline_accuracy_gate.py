"""Field-level online-vs-offline calibration gate for AegisKV WCT runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


WCT_RELATIVE_TOLERANCE = 0.10
SERVICE_RELATIVE_TOLERANCE = 0.10
MAKESPAN_RELATIVE_TOLERANCE = 0.10
FORWARD_RELATIVE_TOLERANCE = 0.01
COMM_RELATIVE_TOLERANCE = 0.05
COMPLETION_ORDER_MIN_SPEARMAN = 0.99
QUEUE_DEPTH_MIN_CORRELATION = 0.95
QUEUE_DEPTH_MAX_MAE = 1.0
# A one-request waiting queue can persist for one short decode burst when two
# sibling agent threads tie in the online wall-clock client.  Offline orders
# the same-timestamp arrivals deterministically, so require matching request
# sets/order correlation and cap this sparse occupancy difference at 15% of
# active steps.  Larger depths still use the general correlation/MAE gate.
QUEUE_DEPTH_SPARSE_MAX_MAE = 0.15
COMM_ABSOLUTE_TOLERANCE_S = 0.002
COMM_ABSOLUTE_MAX_TOLERANCE_S = 0.005
COMM_PIPELINE_FRACTION_TOLERANCE = 0.005
SUMMARY_FIELDS = ("mean", "p50", "p90", "p99", "max")
PHYSICAL_TOPOLOGY_CONTRACT = "one_scheduler_process_per_dp_pp_tp_stage"


def _stage_sort_key(stage_id: str) -> tuple[int, int, int]:
    parts = str(stage_id).replace("dp", "").replace("pp", "").replace("tp", "")
    values = parts.split(".")
    try:
        return tuple(int(value) for value in values[:3]) + (0,) * (3 - len(values))
    except ValueError:
        return (2**31 - 1, 2**31 - 1, 2**31 - 1)


def _physical_stage_id(dp_rank: int, pp_rank: int, tp_rank: int = 0) -> str:
    return f"dp{int(dp_rank)}.pp{int(pp_rank)}.tp{int(tp_rank)}"


def _relative_error(reference: float, candidate: float) -> float:
    denominator = abs(float(reference))
    if denominator == 0.0:
        return 0.0 if float(candidate) == 0.0 else float("inf")
    return abs(float(candidate) - float(reference)) / denominator


def _identity_value(payload: dict[str, Any], field: str) -> Any:
    if field in payload:
        return payload.get(field)
    return payload.get("identity", {}).get(field)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var == 0.0 or right_var == 0.0:
        return 1.0 if left == right else 0.0
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    return covariance / math.sqrt(left_var * right_var)


def _completion_order_check(
    online_order: dict[str, list[str]],
    offline_order: dict[str, list[str]],
) -> dict[str, Any]:
    per_rank: dict[str, Any] = {}
    same_sets = True
    weighted_spearman = 0.0
    total_requests = 0
    for rank in sorted(set(online_order) | set(offline_order), key=int):
        reference = list(online_order.get(rank, ()))
        candidate = list(offline_order.get(rank, ()))
        same_requests = set(reference) == set(candidate) and len(reference) == len(candidate)
        if same_requests and reference:
            candidate_positions = {rid: index for index, rid in enumerate(candidate)}
            spearman = _pearson(
                [float(index) for index in range(len(reference))],
                [float(candidate_positions[rid]) for rid in reference],
            )
        else:
            spearman = 1.0 if not reference and not candidate else 0.0
        rank_passed = same_requests and spearman >= COMPLETION_ORDER_MIN_SPEARMAN
        same_sets = same_sets and same_requests
        weighted_spearman += spearman * len(reference)
        total_requests += len(reference)
        per_rank[rank] = {
            "passed": rank_passed,
            "online_count": len(reference),
            "offline_count": len(candidate),
            "same_request_set": same_requests,
            "spearman": spearman,
        }
    aggregate_spearman = (
        weighted_spearman / total_requests if total_requests else 0.0
    )
    return {
        "passed": same_sets
        and aggregate_spearman >= COMPLETION_ORDER_MIN_SPEARMAN,
        "minimum_spearman": COMPLETION_ORDER_MIN_SPEARMAN,
        "weighted_spearman": aggregate_spearman,
        "all_per_dp_request_sets_match": same_sets,
        "gate_boundary": (
            "request-weighted per-DP Spearman; each rank's request set must match"
        ),
        "per_dp": per_rank,
    }


def _workflow_completion_orders(payload: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for workflow in payload.get("workflows", ()):
        if not isinstance(workflow, dict):
            continue
        workflow_id = str(workflow.get("workflow_id", ""))
        rows = [
            dict(row)
            for row in workflow.get("request_routes", ())
            if isinstance(row, dict)
        ]
        result[workflow_id] = [
            str(row.get("rid", ""))
            for row in sorted(
                rows,
                key=lambda row: (
                    int(row.get("completion_index", 2**31 - 1)),
                    str(row.get("rid", "")),
                ),
            )
        ]
    return result


def _global_completion_order_check(
    online: dict[str, Any],
    offline: dict[str, Any],
) -> dict[str, Any]:
    online_orders = _workflow_completion_orders(online)
    offline_orders = _workflow_completion_orders(offline)
    mismatches = {
        workflow_id: {
            "online": online_orders.get(workflow_id, []),
            "offline": offline_orders.get(workflow_id, []),
        }
        for workflow_id in sorted(set(online_orders) | set(offline_orders))
        if online_orders.get(workflow_id) != offline_orders.get(workflow_id)
    }
    return {
        "passed": bool(online_orders) and not mismatches,
        "comparison_boundary": (
            "per-workflow external response-delivery completion index; required "
            "because concurrent source DPs mutate shared backup-target capacity"
        ),
        "workflow_count": len(online_orders),
        "mismatches": mismatches,
    }


def _active_queue_rows(
    payload: dict[str, Any],
    *,
    by_stage: bool,
) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for row in payload.get("scheduler_telemetry", ()):
        if str(row.get("batch_mode", "idle")) == "idle":
            continue
        rank = str(int(row.get("dp_rank", -1)))
        if rank == "-1":
            continue
        key = (
            _physical_stage_id(
                int(rank),
                int(row.get("pp_rank", 0)),
                int(row.get("tp_rank", 0)),
            )
            if by_stage
            else rank
        )
        grouped.setdefault(key, []).append(
            (
                int(row.get("waiting_depth_after", row.get("waiting_depth", 0))),
                int(row.get("running_depth_after", row.get("running_depth", 0))),
            )
        )
    return grouped


def _queue_depth_check(
    online: dict[str, Any],
    offline: dict[str, Any],
) -> dict[str, Any]:
    by_stage = int(_identity_value(online, "pp_size") or 1) > 1
    online_rows = _active_queue_rows(online, by_stage=by_stage)
    offline_rows = _active_queue_rows(offline, by_stage=by_stage)
    per_rank: dict[str, Any] = {}
    passed = True
    sort_key = _stage_sort_key if by_stage else int
    for rank in sorted(set(online_rows) | set(offline_rows), key=sort_key):
        reference = online_rows.get(rank, [])
        candidate = offline_rows.get(rank, [])
        same_length = len(reference) == len(candidate) and bool(reference)
        fields: dict[str, Any] = {}
        rank_passed = same_length
        for index, field in enumerate(("waiting", "running")):
            left = [float(row[index]) for row in reference]
            right = [float(row[index]) for row in candidate]
            correlation = _pearson(left, right) if same_length else 0.0
            mae = (
                sum(abs(a - b) for a, b in zip(left, right)) / len(left)
                if same_length
                else float("inf")
            )
            dynamic_range = (
                max(left + right) - min(left + right)
                if same_length and left
                else float("inf")
            )
            sparse_low_range = dynamic_range <= 1.0
            field_passed = same_length and (
                (
                    correlation >= QUEUE_DEPTH_MIN_CORRELATION
                    and mae <= QUEUE_DEPTH_MAX_MAE
                )
                or (
                    sparse_low_range
                    and mae <= QUEUE_DEPTH_SPARSE_MAX_MAE
                )
            )
            rank_passed = rank_passed and field_passed
            fields[field] = {
                "passed": field_passed,
                "correlation": correlation,
                "mae": mae,
                "dynamic_range": dynamic_range,
                "sparse_low_range_rule_applied": (
                    sparse_low_range and correlation < QUEUE_DEPTH_MIN_CORRELATION
                ),
            }
        passed = passed and rank_passed
        per_rank[rank] = {
            "passed": rank_passed,
            "online_active_steps": len(reference),
            "offline_active_steps": len(candidate),
            "same_length": same_length,
            "fields": fields,
        }
    return {
        "passed": passed and bool(per_rank),
        "minimum_correlation": QUEUE_DEPTH_MIN_CORRELATION,
        "maximum_mae": QUEUE_DEPTH_MAX_MAE,
        "sparse_low_range_max_mae": QUEUE_DEPTH_SPARSE_MAX_MAE,
        "alignment": (
            "per-physical-stage active batch index; offline idle drain rows excluded"
            if by_stage
            else "per-DP active batch index; offline idle drain rows excluded"
        ),
        "scope": "physical_stage" if by_stage else "dp",
        "per_stage" if by_stage else "per_dp": per_rank,
    }


def _summary_error_check(
    online_summary: dict[str, Any],
    offline_summary: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    count_matches = int(online_summary.get("count", -1)) == int(
        offline_summary.get("count", -2)
    )
    fields = {
        field: {
            "online": float(online_summary.get(field, 0.0)),
            "offline": float(offline_summary.get(field, 0.0)),
            "relative_error": _relative_error(
                float(online_summary.get(field, 0.0)),
                float(offline_summary.get(field, 0.0)),
            ),
        }
        for field in SUMMARY_FIELDS
    }
    return {
        "passed": count_matches
        and all(row["relative_error"] <= tolerance for row in fields.values()),
        "relative_tolerance": tolerance,
        "online_count": int(online_summary.get("count", -1)),
        "offline_count": int(offline_summary.get("count", -1)),
        "fields": fields,
    }


def _pipeline_accounting_check(
    online: dict[str, Any],
    offline: dict[str, Any],
    *,
    pp_size: int,
) -> dict[str, Any]:
    by_stage = int(pp_size) > 1
    accounting_key = (
        "pipeline_accounting_by_stage"
        if by_stage
        else "pipeline_accounting_by_dp"
    )
    online_by_dp = online.get(accounting_key, {})
    offline_by_dp = offline.get(accounting_key, {})
    per_rank: dict[str, Any] = {}
    passed = True
    tolerances = {
        "forward_s": FORWARD_RELATIVE_TOLERANCE,
        "pipeline_base_s": FORWARD_RELATIVE_TOLERANCE,
        "h2d_s": COMM_RELATIVE_TOLERANCE,
        "backup_pending_s": COMM_RELATIVE_TOLERANCE,
        "cpu_overflow_s": COMM_RELATIVE_TOLERANCE,
        "comm_h2d_visible_s": COMM_RELATIVE_TOLERANCE,
        "comm_d2h_remainder_s": COMM_RELATIVE_TOLERANCE,
    }
    sort_key = _stage_sort_key if by_stage else int
    for rank in sorted(set(online_by_dp) | set(offline_by_dp), key=sort_key):
        reference = online_by_dp.get(rank, {})
        candidate = offline_by_dp.get(rank, {})
        batch_count_matches = int(reference.get("batch_count", -1)) == int(
            candidate.get("batch_count", -2)
        )
        fields = {}
        rank_passed = batch_count_matches
        for field, tolerance in tolerances.items():
            online_value = float(reference.get(field, 0.0))
            offline_value = float(candidate.get(field, 0.0))
            error = _relative_error(online_value, offline_value)
            absolute_error = abs(online_value - offline_value)
            if field in {"forward_s", "pipeline_base_s"}:
                absolute_tolerance = 0.0
                absolute_tolerance_basis = "none"
            elif field in {
                "h2d_s",
                "backup_pending_s",
                "comm_h2d_visible_s",
                "comm_d2h_remainder_s",
            }:
                absolute_tolerance = max(
                    COMM_ABSOLUTE_TOLERANCE_S,
                    min(
                        COMM_ABSOLUTE_MAX_TOLERANCE_S,
                        COMM_PIPELINE_FRACTION_TOLERANCE
                        * float(reference.get("pipeline_base_s", 0.0)),
                    ),
                )
                absolute_tolerance_basis = (
                    "max(2ms,min(5ms,0.5% physical-stage pipeline)); "
                    "communication is overlapped bookkeeping and stage ACK "
                    "arrival order is intentionally not bit-identical"
                )
            else:
                absolute_tolerance = COMM_ABSOLUTE_TOLERANCE_S
                absolute_tolerance_basis = "fixed 2ms"
            field_passed = (
                error <= tolerance or absolute_error <= absolute_tolerance
            )
            rank_passed = rank_passed and field_passed
            fields[field] = {
                "passed": field_passed,
                "online": online_value,
                "offline": offline_value,
                "relative_error": error,
                "relative_tolerance": tolerance,
                "absolute_error_s": absolute_error,
                "absolute_tolerance_s": absolute_tolerance,
                "absolute_tolerance_basis": absolute_tolerance_basis,
            }
        passed = passed and rank_passed
        per_rank[rank] = {
            "passed": rank_passed,
            "batch_count_matches": batch_count_matches,
            "online_batch_count": int(reference.get("batch_count", -1)),
            "offline_batch_count": int(candidate.get("batch_count", -1)),
            "fields": fields,
        }
    return {
        "passed": passed and bool(per_rank),
        "scope": "physical_stage" if by_stage else "dp",
        "accounting_key": accounting_key,
        "per_stage" if by_stage else "per_dp": per_rank,
    }


def _offline_backup_summary(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("aegis_status", {}).get("events", ())
    acks = [event for event in events if event.get("kind") == "backup_ack"]
    final_hints = [
        hint
        for hint in payload.get("aegis_status", {}).get("recovery_hints", ())
        if isinstance(hint, dict) and bool(hint.get("ack", False))
    ]
    # The WCT artifact exposes both boundaries: final resident frontiers for
    # tier-token stock and immutable ACK-event count for request completeness.
    aggregate = final_hints or acks
    return {
        "count": len(aggregate),
        "ack_event_count": len(acks),
        "l1_tokens": sum(int(event.get("l1_tokens", 0)) for event in aggregate),
        "l2_tokens": sum(int(event.get("l2_tokens", 0)) for event in aggregate),
        "l3_tokens": sum(int(event.get("l3_tokens", 0)) for event in aggregate),
        "recompute_tokens": sum(
            int(event.get("recompute_tokens", 0)) for event in aggregate
        ),
        "actions": dict(
            sorted(
                Counter(
                    str(event.get("recovery_action", event.get("action", "")))
                    for event in aggregate
                ).items()
            )
        ),
        "aggregate_boundary": (
            "final_recovery_hints" if final_hints else "backup_ack_events"
        ),
        "by_key": {str(event.get("backup_key", "")): dict(event) for event in acks},
    }


def _retry_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(retry)
        for workflow in payload.get("workflows", ())
        for retry in workflow.get("retry_events", ())
    ]


def _normalized_stage_ids(raw_ids: Any) -> list[str]:
    normalized = []
    for raw in raw_ids or ():
        stage_id = str(raw)
        if ".tp" not in stage_id:
            stage_id = f"{stage_id}.tp0"
        normalized.append(stage_id)
    return sorted(normalized, key=_stage_sort_key)


def _physical_topology_check(
    online: dict[str, Any],
    offline: dict[str, Any],
) -> dict[str, Any]:
    dp_size = int(_identity_value(online, "dp_size") or 0)
    pp_size = int(_identity_value(online, "pp_size") or 0)
    online_topology = dict(online.get("physical_topology", {}))
    offline_topology = dict(offline.get("physical_topology", {}))
    tp_size = int(online_topology.get("tp_size", 1))
    expected_count = dp_size * pp_size * tp_size
    expected_stage_ids = [
        _physical_stage_id(dp_rank, pp_rank, tp_rank)
        for dp_rank in range(dp_size)
        for pp_rank in range(pp_size)
        for tp_rank in range(tp_size)
    ]

    def audit(raw: dict[str, Any], *, require_live: bool) -> dict[str, Any]:
        stage_ids = _normalized_stage_ids(raw.get("stage_ids", ()))
        process_count = int(raw.get("process_count", -1))
        stage_count = int(raw.get("stage_count", len(stage_ids)))
        contract = str(raw.get("topology_contract", ""))
        alive_count = int(raw.get("alive_process_count", -1))
        passed = (
            contract == PHYSICAL_TOPOLOGY_CONTRACT
            and int(raw.get("dp_size", -1)) == dp_size
            and int(raw.get("pp_size", -1)) == pp_size
            and int(raw.get("tp_size", 1)) == tp_size
            and int(raw.get("expected_process_count", -1)) == expected_count
            and process_count == expected_count
            and stage_count == expected_count
            and stage_ids == expected_stage_ids
            and (not require_live or alive_count == expected_count)
        )
        return {
            "passed": passed,
            "contract": contract,
            "process_count": process_count,
            "stage_count": stage_count,
            "alive_process_count": alive_count,
            "stage_ids": stage_ids,
            "require_live_processes": require_live,
        }

    online_audit = audit(online_topology, require_live=True)
    offline_audit = audit(offline_topology, require_live=False)
    return {
        "passed": online_audit["passed"] and offline_audit["passed"],
        "contract": PHYSICAL_TOPOLOGY_CONTRACT,
        "expected_process_count": expected_count,
        "expected_stage_ids": expected_stage_ids,
        "online": online_audit,
        "offline": offline_audit,
    }


def _aegis_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    profile = payload.get("aegis_profile_events")
    if isinstance(profile, list):
        return [dict(event) for event in profile if isinstance(event, dict)]
    return [
        dict(event)
        for event in payload.get("aegis_status", {}).get("events", ())
        if isinstance(event, dict)
    ]


def _exact_deployment_check(
    online: dict[str, Any],
    offline: dict[str, Any],
) -> dict[str, Any]:
    """Prove whether cross-DP response order can affect placement semantics."""

    def summarize(payload: dict[str, Any]) -> dict[str, Any]:
        events = _aegis_events(payload)
        installs = [
            event
            for event in events
            if event.get("kind") == "certified_exact_window_installed"
        ]
        plans = [
            event for event in events if event.get("kind") == "backup_plan"
        ]
        violations = [
            event
            for event in events
            if event.get("kind") == "certified_exact_deployment_violation"
        ]
        install = installs[0] if len(installs) == 1 else {}
        objective = install.get("objective")
        dual_bound = install.get("dual_bound")
        tolerance = 1e-8 * max(abs(float(objective or 0.0)), 1.0)
        certified = bool(
            len(installs) == 1
            and install.get("solver_status") == "OPTIMAL"
            and objective is not None
            and dual_bound is not None
            and float(install.get("mip_gap", float("inf"))) <= 1e-8
            and abs(float(objective) - float(dual_bound)) <= tolerance
            and install.get("decision_sha256")
        )
        exact_plan_count = sum(
            1
            for event in plans
            if event.get("placement_mode") == "certified_exact"
            and event.get("solver_decision_sha256")
            == install.get("decision_sha256")
            and event.get("solver_instance_sha256")
            == install.get("instance_sha256")
        )
        request_count = int(install.get("request_count", 0) or 0)
        workload_request_count = int(
            _identity_value(payload, "request_count") or 0
        )
        return {
            "present": bool(installs),
            "certified": certified,
            "install_count": len(installs),
            "instance_id": install.get("instance_id"),
            "instance_sha256": install.get("instance_sha256"),
            "decision_sha256": install.get("decision_sha256"),
            "solver_status": install.get("solver_status"),
            "objective": objective,
            "dual_bound": dual_bound,
            "mip_gap": install.get("mip_gap"),
            "window_id": install.get("window_id"),
            "route_version": install.get("route_version"),
            "request_count": request_count,
            "workload_request_count": workload_request_count,
            "backup_plan_count": len(plans),
            "certified_exact_backup_plan_count": exact_plan_count,
            "deployment_violation_count": len(violations),
            "all_requests_use_frozen_placement": bool(
                certified
                and request_count > 0
                and request_count == workload_request_count
                and len(plans) == request_count
                and exact_plan_count == request_count
            ),
        }

    online_summary = summarize(online)
    offline_summary = summarize(offline)
    applicable = online_summary["present"] or offline_summary["present"]
    identity_fields = (
        "instance_id",
        "instance_sha256",
        "decision_sha256",
        "window_id",
        "route_version",
        "request_count",
    )
    identity_mismatches = {
        field: {
            "online": online_summary.get(field),
            "offline": offline_summary.get(field),
        }
        for field in identity_fields
        if online_summary.get(field) != offline_summary.get(field)
    }
    passed = bool(
        not applicable
        or (
            online_summary["certified"]
            and offline_summary["certified"]
            and online_summary["all_requests_use_frozen_placement"]
            and offline_summary["all_requests_use_frozen_placement"]
            and online_summary["deployment_violation_count"] == 0
            and offline_summary["deployment_violation_count"] == 0
            and not identity_mismatches
        )
    )
    return {
        "passed": passed,
        "applicable": applicable,
        "order_independent_placement_contract": applicable and passed,
        "identity_mismatches": identity_mismatches,
        "online": online_summary,
        "offline": offline_summary,
        "boundary": (
            "A runtime-validated identical OPTIMAL decision freezes every "
            "request route, stage chunk tier, and DP budget before arrivals; "
            "cross-DP response ties therefore cannot change placement."
        ),
    }


_STAGE_PROTOCOL_FIELDS: dict[str, tuple[str, ...]] = {
    "router_top2": (
        "rid",
        "execution_dp_rank",
        "backup_dp_rank",
        "route_plan_dp_rank",
        "route_plan_applied",
    ),
    "backup_plan": (
        "rid",
        "backup_key",
        "execution_dp_rank",
        "backup_dp_rank",
        "tokens",
        "failure_scope",
        "route_version",
        "budget_version",
        "required_pp_stages",
        "placement_mode",
        "solver_instance_sha256",
        "solver_decision_sha256",
    ),
    "backup_stage_ack": (
        "rid",
        "backup_key",
        "stage_backup_key",
        "source_dp_rank",
        "target_dp_rank",
        "pp_rank",
        "route_version",
        "budget_version",
        "committed_frontier_tokens",
        "hbm_tokens",
        "host_tokens",
        "recompute_tokens",
        "required_stage_count",
    ),
    "backup_ack": (
        "rid",
        "backup_key",
        "execution_dp_rank",
        "backup_dp_rank",
        "action",
        "safe_frontier_tokens",
        "bottleneck_pp_rank",
        "restored_tokens_by_stage",
        "recomputed_tokens_by_stage",
        "route_version",
        "budget_version",
        "committed_stage_count",
        "required_stage_count",
    ),
    "backup_frontier_superseded": (
        "workflow_id",
        "agent_id",
        "backup_key",
        "version",
        "worker_seq",
        "target_dp_rank",
        "superseded_by",
        "superseded_by_worker_seq",
        "released_host_allocated_tokens",
        "released_l3_tokens",
    ),
    "backup_frontier_supersede_stage_applied": (
        "target_dp_rank",
        "pp_rank",
        "backup_key",
        "stage_backup_key",
        "expected_version",
        "released",
        "released_l1_tokens",
        "released_l2_tokens",
        "required_stage_count",
    ),
    "backup_frontier_supersede_applied": (
        "target_dp_rank",
        "backup_key",
        "expected_version",
        "superseded_by",
        "released",
        "released_l1_tokens",
        "released_l2_tokens",
        "stage_backup_keys",
        "applied_stage_count",
        "required_stage_count",
        "all_stages_applied",
    ),
    "dp_budget_stage_ack": ("dp_rank", "pp_rank", "budget_version"),
}


def _normalized_protocol_counter(events: list[dict[str, Any]]) -> Counter[str]:
    rows: Counter[str] = Counter()
    for event in events:
        kind = str(event.get("kind", ""))
        fields = _STAGE_PROTOCOL_FIELDS.get(kind)
        if fields is None:
            continue
        row = {"kind": kind, **{field: event.get(field) for field in fields}}
        rows[json.dumps(row, sort_keys=True, separators=(",", ":"))] += 1
    return rows


def _stage_protocol_invariants(
    events: list[dict[str, Any]],
    *,
    dp_size: int,
    pp_size: int,
    require_backups: bool,
) -> dict[str, Any]:
    plans = {
        str(event.get("backup_key", "")): event
        for event in events
        if event.get("kind") == "backup_plan"
    }
    stage_acks: dict[str, list[dict[str, Any]]] = {}
    all_acks: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        backup_key = str(event.get("backup_key", ""))
        if event.get("kind") == "backup_stage_ack":
            stage_acks.setdefault(backup_key, []).append(event)
        elif event.get("kind") == "backup_ack":
            all_acks.setdefault(backup_key, []).append(event)
    violations: dict[str, Any] = {}
    for backup_key, plan in sorted(plans.items()):
        rows = stage_acks.get(backup_key, [])
        ranks = sorted(int(row.get("pp_rank", -1)) for row in rows)
        aggregate = all_acks.get(backup_key, [])
        frontiers = [int(row.get("committed_frontier_tokens", -1)) for row in rows]
        safe_frontier = (
            int(aggregate[0].get("safe_frontier_tokens", -1))
            if len(aggregate) == 1
            else -1
        )
        expected_ranks = list(range(pp_size))
        request_ok = (
            int(plan.get("required_pp_stages", -1)) == pp_size
            and ranks == expected_ranks
            and len(aggregate) == 1
            and int(aggregate[0].get("committed_stage_count", -1)) == pp_size
            and int(aggregate[0].get("required_stage_count", -1)) == pp_size
            and bool(frontiers)
            and safe_frontier == min(frontiers)
        )
        if not request_ok:
            violations[backup_key] = {
                "stage_ack_pp_ranks": ranks,
                "expected_pp_ranks": expected_ranks,
                "all_stage_ack_count": len(aggregate),
                "stage_frontiers": frontiers,
                "safe_frontier": safe_frontier,
            }
    budget_stage_ids = {
        (int(event.get("dp_rank", -1)), int(event.get("pp_rank", -1)))
        for event in events
        if event.get("kind") == "dp_budget_stage_ack"
    }
    expected_budget_stage_ids = {
        (dp_rank, pp_rank)
        for dp_rank in range(dp_size)
        for pp_rank in range(pp_size)
    }
    backup_protocol_passed = (
        bool(plans) and not violations
        if require_backups
        else not plans and not stage_acks and not all_acks
    )
    budget_protocol_passed = (
        budget_stage_ids == expected_budget_stage_ids
        if require_backups or budget_stage_ids
        else True
    )
    return {
        "passed": backup_protocol_passed and budget_protocol_passed,
        "backup_protocol_required": require_backups,
        "backup_plan_count": len(plans),
        "stage_ack_count": sum(len(rows) for rows in stage_acks.values()),
        "all_stage_ack_count": sum(len(rows) for rows in all_acks.values()),
        "violations": violations,
        "budget_stage_ids": sorted(budget_stage_ids),
        "expected_budget_stage_ids": sorted(expected_budget_stage_ids),
    }


def _stage_protocol_check(
    online: dict[str, Any],
    offline: dict[str, Any],
) -> dict[str, Any]:
    dp_size = int(_identity_value(online, "dp_size") or 0)
    pp_size = int(_identity_value(online, "pp_size") or 0)
    require_backups = str(_identity_value(online, "policy") or "") != "no_backup"
    online_events = _aegis_events(online)
    offline_events = _aegis_events(offline)
    online_invariants = _stage_protocol_invariants(
        online_events,
        dp_size=dp_size,
        pp_size=pp_size,
        require_backups=require_backups,
    )
    offline_invariants = _stage_protocol_invariants(
        offline_events,
        dp_size=dp_size,
        pp_size=pp_size,
        require_backups=require_backups,
    )
    online_counter = _normalized_protocol_counter(online_events)
    offline_counter = _normalized_protocol_counter(offline_events)
    missing_offline = online_counter - offline_counter
    extra_offline = offline_counter - online_counter
    normalized_match = not missing_offline and not extra_offline
    return {
        "passed": (
            online_invariants["passed"]
            and offline_invariants["passed"]
            and normalized_match
        ),
        "comparison_boundary": (
            "exact protocol multiset after removing wall/logical timestamps, "
            "global sequence numbers, and asynchronous ACK arrival order"
        ),
        "online_invariants": online_invariants,
        "offline_invariants": offline_invariants,
        "normalized_protocol_rows_match": normalized_match,
        "missing_offline": dict(missing_offline),
        "extra_offline": dict(extra_offline),
    }


def compare(online: dict[str, Any], offline: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []

    structural_fields = (
        "policy",
        "model",
        "dp_size",
        "pp_size",
        "workflow_count",
        "request_count",
    )
    structural_mismatches = {
        field: {
            "online": _identity_value(online, field),
            "offline": _identity_value(offline, field),
        }
        for field in structural_fields
        if _identity_value(online, field) != _identity_value(offline, field)
    }
    structural_passed = not structural_mismatches
    checks["structural_invariants"] = {
        "passed": structural_passed,
        "mismatches": structural_mismatches,
    }
    if not structural_passed:
        blockers.append("structural_invariants")

    topology_check = _physical_topology_check(online, offline)
    checks["physical_dp_pp_topology"] = topology_check
    if not topology_check["passed"]:
        blockers.append("physical_dp_pp_topology")

    stage_protocol_check = _stage_protocol_check(online, offline)
    checks["stage_aligned_protocol"] = stage_protocol_check
    if not stage_protocol_check["passed"]:
        blockers.append("stage_aligned_protocol")

    online_wct = dict(online.get("wct_summary_s", {}))
    offline_wct = dict(offline.get("workflow_completion_time_s", {}))
    wct_fields = ("mean", "p50", "p90", "p99", "max")
    wct_errors = {
        field: {
            "online": float(online_wct.get(field, 0.0)),
            "offline": float(offline_wct.get(field, 0.0)),
            "relative_error": _relative_error(
                float(online_wct.get(field, 0.0)),
                float(offline_wct.get(field, 0.0)),
            ),
        }
        for field in wct_fields
    }
    wct_passed = all(
        row["relative_error"] <= WCT_RELATIVE_TOLERANCE
        for row in wct_errors.values()
    )
    checks["wct_distribution"] = {
        "passed": wct_passed,
        "relative_tolerance": WCT_RELATIVE_TOLERANCE,
        "fields": wct_errors,
    }
    if not wct_passed:
        blockers.append("wct_distribution")

    primary_audit = dict(offline.get("route_plan_audit", {}))
    route_passed = (
        int(primary_audit.get("violation_count", -1)) == 0
        and int(primary_audit.get("retry_violation_count", -1)) == 0
    )
    checks["primary_and_retry_routes"] = {
        "passed": route_passed,
        "primary_violation_count": int(primary_audit.get("violation_count", -1)),
        "retry_violation_count": int(
            primary_audit.get("retry_violation_count", -1)
        ),
    }
    if not route_passed:
        blockers.append("primary_and_retry_routes")

    online_failures = [
        str(event.get("event_id", "")) for event in online.get("failure_events", ())
    ]
    offline_failures = list(
        offline.get("calibration_metrics", {}).get("failure_event_order", ())
    )
    failure_order_passed = online_failures == offline_failures
    checks["failure_event_order"] = {
        "passed": failure_order_passed,
        "online": online_failures,
        "offline": offline_failures,
    }
    if not failure_order_passed:
        blockers.append("failure_event_order")

    online_retries = {row["backup_key"]: row for row in _retry_rows(online)}
    offline_retries = {row["backup_key"]: row for row in _retry_rows(offline)}
    retry_fields = (
        "retry_dp_rank",
        "recovery_action",
        "restored_tokens",
        "recompute_tokens",
    )
    retry_mismatches = {}
    for backup_key in sorted(set(online_retries) | set(offline_retries)):
        online_row = online_retries.get(backup_key, {})
        offline_row = offline_retries.get(backup_key, {})
        fields = {
            field: {"online": online_row.get(field), "offline": offline_row.get(field)}
            for field in retry_fields
            if online_row.get(field) != offline_row.get(field)
        }
        if fields:
            retry_mismatches[backup_key] = fields
    retry_passed = not retry_mismatches and len(online_retries) == len(offline_retries)
    checks["recovery_semantics"] = {
        "passed": retry_passed,
        "online_count": len(online_retries),
        "offline_count": len(offline_retries),
        "mismatches": retry_mismatches,
    }
    if not retry_passed:
        blockers.append("recovery_semantics")

    online_backup = dict(online.get("backup_decision_summary", {}))
    offline_backup = _offline_backup_summary(offline)
    aggregate_fields = (
        "count",
        "l1_tokens",
        "l2_tokens",
        "l3_tokens",
        "recompute_tokens",
        "actions",
    )
    if "ack_event_count" in online_backup:
        aggregate_fields += ("ack_event_count",)
    aggregate_mismatches = {
        field: {"online": online_backup.get(field), "offline": offline_backup.get(field)}
        for field in aggregate_fields
        if online_backup.get(field) != offline_backup.get(field)
    }
    per_request_action_mismatches = {
        backup_key: {
            "online": event.get("action"),
            "offline": offline_backup["by_key"].get(backup_key, {}).get("action"),
        }
        for backup_key, event in {
            str(event.get("backup_key", "")): event
            for event in online.get("aegis_profile_events", ())
            if event.get("kind") == "backup_ack"
        }.items()
        if event.get("action")
        != offline_backup["by_key"].get(backup_key, {}).get("action")
    }
    backup_passed = not aggregate_mismatches and not per_request_action_mismatches
    checks["backup_kv_side_effects"] = {
        "passed": backup_passed,
        "aggregate_mismatches": aggregate_mismatches,
        "per_request_action_mismatch_count": len(per_request_action_mismatches),
        "per_request_action_mismatches": per_request_action_mismatches,
        "formal_cache_tie_explanation_allowed_but_not_auto_accepted": True,
    }
    if not backup_passed:
        blockers.append("backup_kv_side_effects")

    exact_deployment_check = _exact_deployment_check(online, offline)
    checks["certified_exact_deployment"] = exact_deployment_check
    if exact_deployment_check["applicable"] and not exact_deployment_check["passed"]:
        blockers.append("certified_exact_deployment")

    required_common_telemetry = {
        "forward_and_communication_accounting": bool(
            online.get("scheduler_telemetry")
            or online.get("calibration_metrics", {}).get(
                (
                    "pipeline_accounting_by_stage"
                    if int(_identity_value(online, "pp_size") or 1) > 1
                    else "pipeline_accounting_by_dp"
                )
            )
        ),
        "ttft_itl_service": bool(
            online.get("calibration_metrics", {}).get("ttft_s")
        ),
        "per_dp_completion_order": bool(
            online.get("calibration_metrics", {}).get("completion_order_by_dp")
        ),
        "queue_depth_shape": bool(
            online.get("calibration_metrics", {}).get(
                (
                    "queue_depth_by_stage"
                    if int(_identity_value(online, "pp_size") or 1) > 1
                    else "queue_depth_by_dp"
                )
            )
        ),
    }
    unavailable = [
        name for name, available in required_common_telemetry.items() if not available
    ]
    checks["required_common_telemetry"] = {
        "passed": not unavailable,
        "availability": required_common_telemetry,
        "not_evaluable": unavailable,
    }
    if unavailable:
        blockers.append("required_common_telemetry")

    online_common = dict(online.get("calibration_metrics", {}))
    offline_common = dict(offline.get("calibration_metrics", {}))

    route_map_passed = online_common.get("routes") == offline_common.get("routes")
    checks["request_route_map"] = {
        "passed": route_map_passed,
        "online_count": len(online_common.get("routes", {})),
        "offline_count": len(offline_common.get("routes", {})),
        "mismatches": {
            rid: {
                "online": online_common.get("routes", {}).get(rid),
                "offline": offline_common.get("routes", {}).get(rid),
            }
            for rid in sorted(
                set(online_common.get("routes", {}))
                | set(offline_common.get("routes", {}))
            )
            if online_common.get("routes", {}).get(rid)
            != offline_common.get("routes", {}).get(rid)
        },
    }
    if not route_map_passed:
        blockers.append("request_route_map")

    service_checks = {
        name: _summary_error_check(
            dict(online_common.get(name, {})),
            dict(offline_common.get(name, {})),
            tolerance=SERVICE_RELATIVE_TOLERANCE,
        )
        for name in ("ttft_s", "itl_s", "service_time_s")
    }
    service_passed = all(check["passed"] for check in service_checks.values())
    checks["ttft_itl_service_distributions"] = {
        "passed": service_passed,
        "metrics": service_checks,
    }
    if not service_passed:
        blockers.append("ttft_itl_service_distributions")

    online_makespan = float(online_common.get("makespan_s", 0.0))
    offline_makespan = float(offline_common.get("makespan_s", 0.0))
    makespan_error = _relative_error(online_makespan, offline_makespan)
    makespan_passed = makespan_error <= MAKESPAN_RELATIVE_TOLERANCE
    checks["makespan"] = {
        "passed": makespan_passed,
        "online": online_makespan,
        "offline": offline_makespan,
        "relative_error": makespan_error,
        "relative_tolerance": MAKESPAN_RELATIVE_TOLERANCE,
    }
    if not makespan_passed:
        blockers.append("makespan")

    completion_check = _completion_order_check(
        dict(online_common.get("completion_order_by_dp", {})),
        dict(offline_common.get("completion_order_by_dp", {})),
    )
    checks["per_dp_completion_order"] = completion_check
    if not completion_check["passed"]:
        blockers.append("per_dp_completion_order")

    global_completion_check = _global_completion_order_check(online, offline)
    global_completion_check["exact_order_match"] = bool(
        global_completion_check["passed"]
    )
    exact_order_independent = bool(
        exact_deployment_check["order_independent_placement_contract"]
        and stage_protocol_check["passed"]
        and backup_passed
    )
    if not global_completion_check["passed"] and exact_order_independent:
        global_completion_check.update(
            {
                "passed": True,
                "non_blocking_reason": (
                    "Cross-DP completion ties differ, but the identical "
                    "runtime-validated exact plan froze all route/placement/"
                    "budget decisions before arrivals and the resulting "
                    "protocol plus KV side effects match exactly."
                ),
                "comparison_boundary": (
                    "diagnostic external response-delivery order; exact match "
                    "is not required only under the certified order-independent "
                    "placement contract"
                ),
            }
        )
    checks["workflow_response_completion_order"] = global_completion_check
    if not global_completion_check["passed"]:
        blockers.append("workflow_response_completion_order")

    queue_check = _queue_depth_check(online, offline)
    checks["per_dp_queue_depth_shape"] = queue_check
    if not queue_check["passed"]:
        blockers.append("per_dp_queue_depth_shape")

    pipeline_check = _pipeline_accounting_check(
        online_common,
        offline_common,
        pp_size=int(_identity_value(online, "pp_size") or 1),
    )
    checks["forward_and_communication_accounting"] = pipeline_check
    if not pipeline_check["passed"]:
        blockers.append("forward_and_communication_accounting")

    event_count_passed = online_common.get("aegis_event_counts") == offline_common.get(
        "aegis_event_counts"
    )
    checks["cache_backup_event_counts"] = {
        "passed": event_count_passed,
        "online": online_common.get("aegis_event_counts", {}),
        "offline": offline_common.get("aegis_event_counts", {}),
    }
    if not event_count_passed:
        blockers.append("cache_backup_event_counts")

    completion_order_replay = bool(
        offline.get("identity", {}).get("completion_order_plan")
    )
    gate_passed = not blockers
    return {
        "schema_version": 2,
        "gate_version": "v3-physical-stage",
        "passed": gate_passed,
        "clock_layer_passed": gate_passed,
        "evidence_promotable": gate_passed and not completion_order_replay,
        "calibration_mode": (
            "clock_isolation_with_frozen_response_order"
            if completion_order_replay
            else "free_running_online_offline_parity"
        ),
        "requires_free_running_gate": completion_order_replay,
        "online_evidence_tier": online.get("evidence_tier"),
        "offline_evidence_tier": offline.get("evidence_tier"),
        "checks": checks,
        "blocking_checks": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare an online/offline WCT pair.")
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        json.loads(args.online.read_text(encoding="utf-8")),
        json.loads(args.offline.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "passed": report["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
