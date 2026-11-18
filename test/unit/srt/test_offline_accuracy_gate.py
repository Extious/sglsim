from __future__ import annotations

import copy

from AegisKV.demo.offline_accuracy_gate import _stage_protocol_check, compare


def _summary(value: float = 1.0) -> dict:
    return {
        "count": 1,
        "mean": value,
        "p50": value,
        "p90": value,
        "p99": value,
        "max": value,
    }


def _common() -> dict:
    by_stage = {
        f"dp0.pp{pp_rank}": {
            "batch_count": 1,
            "forward_s": 0.25,
            "h2d_s": 0.0,
            "backup_pending_s": 0.0,
            "pipeline_base_s": 0.25,
            "cpu_overflow_s": 0.0,
            "comm_h2d_visible_s": 0.0,
            "comm_d2h_remainder_s": 0.0,
        }
        for pp_rank in range(4)
    }
    return {
        "request_count": 1,
        "routes": {"r0": 0},
        "ttft_s": _summary(),
        "itl_s": _summary(),
        "service_time_s": _summary(),
        "makespan_s": 1.0,
        "completion_order_by_dp": {"0": ["r0"]},
        "queue_depth_by_dp": {"0": []},
        "queue_depth_by_stage": {
            f"dp0.pp{pp_rank}": [] for pp_rank in range(4)
        },
        "pipeline_accounting_by_dp": {
            "0": {
                "batch_count": 1,
                "forward_s": 1.0,
                "h2d_s": 0.0,
                "backup_pending_s": 0.0,
                "pipeline_base_s": 1.0,
                "cpu_overflow_s": 0.0,
                "comm_h2d_visible_s": 0.0,
                "comm_d2h_remainder_s": 0.0,
            }
        },
        "pipeline_accounting_by_stage": by_stage,
        "aegis_event_counts": {"router_top2": 1},
        "failure_event_order": [],
    }


def _pair() -> tuple[dict, dict]:
    identity = {
        "policy": "no_backup",
        "model": "Qwen/Qwen3-32B",
        "dp_size": 4,
        "pp_size": 4,
        "workflow_count": 1,
        "request_count": 1,
    }
    online = {
        **identity,
        "evidence_tier": "sglsim-online-instrumented",
        "wct_summary_s": {field: 1.0 for field in ("mean", "p50", "p90", "p99", "max")},
        "failure_events": [],
        "workflows": [
            {
                "workflow_id": "w0",
                "request_routes": [{"rid": "r0", "completion_index": 1}],
                "retry_events": [],
            }
        ],
        "backup_decision_summary": {
            "count": 0,
            "l1_tokens": 0,
            "l2_tokens": 0,
            "l3_tokens": 0,
            "recompute_tokens": 0,
            "actions": {},
        },
        "aegis_profile_events": [],
        "physical_topology": {
            "topology_contract": "one_scheduler_process_per_dp_pp_tp_stage",
            "dp_size": 4,
            "pp_size": 4,
            "tp_size": 1,
            "expected_process_count": 16,
            "process_count": 16,
            "alive_process_count": 16,
            "stage_count": 16,
            "stage_ids": [
                f"dp{dp_rank}.pp{pp_rank}.tp0"
                for dp_rank in range(4)
                for pp_rank in range(4)
            ],
        },
        "scheduler_telemetry": [
            {
                "dp_rank": 0,
                "pp_rank": pp_rank,
                "batch_mode": "decode",
                "waiting_depth_after": 0,
                "running_depth_after": 1,
            }
            for pp_rank in range(4)
        ],
        "calibration_metrics": _common(),
    }
    offline = {
        **identity,
        "evidence_tier": "sglsim-offline-unvalidated",
        "workflow_completion_time_s": {
            "count": 1,
            **{field: 1.0 for field in ("mean", "p50", "p90", "p99", "max")},
        },
        "workflows": [
            {
                "workflow_id": "w0",
                "request_routes": [{"rid": "r0", "completion_index": 1}],
                "retry_events": [],
            }
        ],
        "route_plan_audit": {"violation_count": 0, "retry_violation_count": 0},
        "aegis_status": {"events": []},
        "physical_topology": {
            "topology_contract": "one_scheduler_process_per_dp_pp_tp_stage",
            "dp_size": 4,
            "pp_size": 4,
            "tp_size": 1,
            "expected_process_count": 16,
            "process_count": 16,
            "stage_count": 16,
            "stage_ids": [
                f"dp{dp_rank}.pp{pp_rank}.tp0"
                for dp_rank in range(4)
                for pp_rank in range(4)
            ],
        },
        "scheduler_telemetry": copy.deepcopy(online["scheduler_telemetry"]),
        "calibration_metrics": _common(),
    }
    return online, offline


def test_v1_gate_quantitatively_checks_common_online_offline_telemetry():
    online, offline = _pair()

    report = compare(online, offline)

    assert report["gate_version"] == "v3-physical-stage"
    assert report["passed"] is True
    assert report["checks"]["forward_and_communication_accounting"]["passed"]
    assert report["checks"]["per_dp_queue_depth_shape"]["passed"]


def test_v1_gate_blocks_queue_depth_shape_drift():
    online, offline = _pair()
    offline["scheduler_telemetry"][0]["waiting_depth_after"] = 4

    report = compare(online, offline)

    assert report["passed"] is False
    assert "per_dp_queue_depth_shape" in report["blocking_checks"]


def test_v2_gate_allows_one_request_same_timestamp_tie_boundary():
    online, offline = _pair()
    online_rows = [
        {
            "dp_rank": 0,
            "pp_rank": 0,
            "batch_mode": "decode",
            "waiting_depth_after": 0,
            "running_depth_after": 1,
        }
        for _ in range(20)
    ]
    offline_rows = copy.deepcopy(online_rows)
    for row in offline_rows[:3]:
        row["waiting_depth_after"] = 1
    online["scheduler_telemetry"] = online_rows
    offline["scheduler_telemetry"] = offline_rows

    report = compare(online, offline)

    queue = report["checks"]["per_dp_queue_depth_shape"]
    assert queue["passed"] is True
    assert queue["sparse_low_range_max_mae"] == 0.15

    offline_rows[3]["waiting_depth_after"] = 1
    report = compare(online, offline)
    assert report["checks"]["per_dp_queue_depth_shape"]["passed"] is False


def test_v3_gate_blocks_missing_physical_stage_process():
    online, offline = _pair()
    offline["physical_topology"]["process_count"] = 15

    report = compare(online, offline)

    assert report["passed"] is False
    assert "physical_dp_pp_topology" in report["blocking_checks"]


def _stage_protocol_payload() -> dict:
    events = [
        {
            "kind": "dp_budget_stage_ack",
            "dp_rank": dp_rank,
            "pp_rank": pp_rank,
            "budget_version": 1,
        }
        for dp_rank in range(4)
        for pp_rank in range(4)
    ]
    events.extend(
        [
            {
                "kind": "router_top2",
                "rid": "r0",
                "execution_dp_rank": 0,
                "backup_dp_rank": 1,
            },
            {
                "kind": "backup_plan",
                "rid": "r0",
                "backup_key": "r0",
                "execution_dp_rank": 0,
                "backup_dp_rank": 1,
                "tokens": 8,
                "route_version": 1,
                "budget_version": 1,
                "required_pp_stages": 4,
            },
        ]
    )
    events.extend(
        {
            "kind": "backup_stage_ack",
            "rid": "r0",
            "backup_key": "r0",
            "stage_backup_key": f"r0::pp{pp_rank}",
            "source_dp_rank": 0,
            "target_dp_rank": 1,
            "pp_rank": pp_rank,
            "route_version": 1,
            "budget_version": 1,
            "committed_frontier_tokens": 8,
            "hbm_tokens": 8,
            "host_tokens": 0,
            "recompute_tokens": 0,
            "required_stage_count": 4,
        }
        for pp_rank in range(4)
    )
    events.append(
        {
            "kind": "backup_ack",
            "rid": "r0",
            "backup_key": "r0",
            "execution_dp_rank": 0,
            "backup_dp_rank": 1,
            "action": "stage_aligned_backup",
            "safe_frontier_tokens": 8,
            "bottleneck_pp_rank": 0,
            "restored_tokens_by_stage": [8, 8, 8, 8],
            "recomputed_tokens_by_stage": [0, 0, 0, 0],
            "route_version": 1,
            "budget_version": 1,
            "committed_stage_count": 4,
            "required_stage_count": 4,
        }
    )
    return {
        "policy": "aegiskv",
        "dp_size": 4,
        "pp_size": 4,
        "aegis_profile_events": events,
    }


def test_v3_stage_protocol_requires_every_stage_ack():
    online = _stage_protocol_payload()
    offline = copy.deepcopy(online)
    assert _stage_protocol_check(online, offline)["passed"] is True

    offline["aegis_profile_events"] = [
        event
        for event in offline["aegis_profile_events"]
        if not (
            event.get("kind") == "backup_stage_ack"
            and event.get("pp_rank") == 3
        )
    ]
    report = _stage_protocol_check(online, offline)
    assert report["passed"] is False
    assert "r0" in report["offline_invariants"]["violations"]


def _certified_exact_events(request_ids: tuple[str, ...]) -> list[dict]:
    events = [
        {
            "kind": "dp_budget_stage_ack",
            "dp_rank": dp_rank,
            "pp_rank": pp_rank,
            "budget_version": 2,
        }
        for dp_rank in range(4)
        for pp_rank in range(4)
    ]
    events.append(
        {
            "kind": "certified_exact_window_installed",
            "instance_id": "exact-order-unit",
            "instance_sha256": "instance-sha",
            "decision_sha256": "decision-sha",
            "solver_status": "OPTIMAL",
            "objective": 1.0,
            "dual_bound": 1.0,
            "mip_gap": 0.0,
            "window_id": 1,
            "route_version": 2,
            "request_count": len(request_ids),
        }
    )
    for index, rid in enumerate(request_ids):
        source = index
        target = (index + 1) % 4
        events.extend(
            [
                {
                    "kind": "router_top2",
                    "rid": rid,
                    "execution_dp_rank": source,
                    "backup_dp_rank": target,
                },
                {
                    "kind": "backup_plan",
                    "rid": rid,
                    "backup_key": rid,
                    "execution_dp_rank": source,
                    "backup_dp_rank": target,
                    "tokens": 8,
                    "route_version": 2,
                    "budget_version": 2,
                    "required_pp_stages": 4,
                    "placement_mode": "certified_exact",
                    "solver_instance_sha256": "instance-sha",
                    "solver_decision_sha256": "decision-sha",
                },
            ]
        )
        events.extend(
            {
                "kind": "backup_stage_ack",
                "rid": rid,
                "backup_key": rid,
                "stage_backup_key": f"{rid}::pp{pp_rank}",
                "source_dp_rank": source,
                "target_dp_rank": target,
                "pp_rank": pp_rank,
                "route_version": 2,
                "budget_version": 2,
                "committed_frontier_tokens": 8,
                "hbm_tokens": 8,
                "host_tokens": 0,
                "recompute_tokens": 0,
                "required_stage_count": 4,
            }
            for pp_rank in range(4)
        )
        events.append(
            {
                "kind": "backup_ack",
                "rid": rid,
                "backup_key": rid,
                "execution_dp_rank": source,
                "backup_dp_rank": target,
                "action": "stage_aligned_backup",
                "recovery_action": "stage_aligned_backup",
                "safe_frontier_tokens": 8,
                "l1_tokens": 8,
                "l2_tokens": 0,
                "l3_tokens": 0,
                "recompute_tokens": 0,
                "bottleneck_pp_rank": 0,
                "restored_tokens_by_stage": [8, 8, 8, 8],
                "recomputed_tokens_by_stage": [0, 0, 0, 0],
                "route_version": 2,
                "budget_version": 2,
                "committed_stage_count": 4,
                "required_stage_count": 4,
            }
        )
    return events


def test_v3_exact_plan_makes_cross_dp_completion_tie_nonblocking():
    online, offline = _pair()
    request_ids = ("r0", "r1")
    events = _certified_exact_events(request_ids)
    for payload in (online, offline):
        payload["policy"] = "aegiskv"
        payload["request_count"] = 2
        payload["workflows"][0]["request_routes"] = [
            {"rid": "r0", "completion_index": 1},
            {"rid": "r1", "completion_index": 2},
        ]
        payload["calibration_metrics"]["request_count"] = 2
        payload["calibration_metrics"]["routes"] = {"r0": 0, "r1": 1}
        payload["calibration_metrics"]["completion_order_by_dp"] = {
            "0": ["r0"],
            "1": ["r1"],
        }
        payload["calibration_metrics"]["aegis_event_counts"] = {
            "certified_exact_window_installed": 1,
            "backup_plan": 2,
        }
    offline["workflows"][0]["request_routes"] = [
        {"rid": "r0", "completion_index": 2},
        {"rid": "r1", "completion_index": 1},
    ]
    online["aegis_profile_events"] = copy.deepcopy(events)
    offline["aegis_status"] = {"events": copy.deepcopy(events)}
    online["backup_decision_summary"] = {
        "count": 2,
        "l1_tokens": 16,
        "l2_tokens": 0,
        "l3_tokens": 0,
        "recompute_tokens": 0,
        "actions": {"stage_aligned_backup": 2},
    }

    report = compare(online, offline)

    assert report["passed"] is True
    exact = report["checks"]["certified_exact_deployment"]
    assert exact["order_independent_placement_contract"] is True
    completion = report["checks"]["workflow_response_completion_order"]
    assert completion["passed"] is True
    assert completion["exact_order_match"] is False
    assert "non_blocking_reason" in completion

    online["aegis_profile_events"] = [
        event
        for event in online["aegis_profile_events"]
        if event["kind"] != "certified_exact_window_installed"
    ]
    offline["aegis_status"]["events"] = [
        event
        for event in offline["aegis_status"]["events"]
        if event["kind"] != "certified_exact_window_installed"
    ]
    report = compare(online, offline)
    assert report["passed"] is False
    assert "workflow_response_completion_order" in report["blocking_checks"]
