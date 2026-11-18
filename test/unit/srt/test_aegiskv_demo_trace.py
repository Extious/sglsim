from __future__ import annotations

import json
import os
import threading
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import pytest

from AegisKV.demo.workflow_client import (
    DEFAULT_TRACE,
    DEFAULT_TRACE_ORACLE,
    DemoWorkflow,
    RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION,
    RETRY_ROUTE_MODE_ROUTER_COMMITTED,
    _algorithm1_trace_position,
    _load_workflow,
    _scaled_tokens,
    _trace_token_ids,
)
from benchmark.agentic_workflows.weka_trace import annotate_trace_oracle_slack
from AegisKV.demo.build_matched_failure_manifest import (
    build_matched_manifest,
)
from AegisKV.demo.build_execution_route_plan import build_execution_route_plan
from AegisKV.demo.build_qwen3_final_summary import paired_profile
from AegisKV.demo.plot_wct_cdf import ecdf
from AegisKV.demo.plot_detailed_analysis import (
    _node_beta_rows,
    _rank_alpha_rows,
)
from AegisKV.demo.run_wct_cdf_suite import (
    DEFAULT_FORMAL_FAILURE_MANIFEST,
    DEFAULT_QUICK_FAILURE_MANIFEST,
    acquire_run_lock,
    acquire_run_lock,
    archive_interrupted_artifacts,
    build_parser,
    policy_result_is_complete,
    resolve_profile,
    snapshot_suite_configuration,
    _terminate_cleanly,
    ROOT,
)
from AegisKV.demo.wct_cdf_client import (
    DEFAULT_FAILURE_MANIFEST,
    FailureManifestCoordinator,
    OrderedWorkflowLaunchGate,
    has_fan_out_and_main_fan_in,
    load_failure_manifest,
    load_execution_route_plan,
    load_workflows,
    percentile,
    progress_output_path,
    write_progress_checkpoint,
)


def test_real_cc_demo_workflow_has_fan_out_and_fan_in() -> None:
    rows = _load_workflow(DEFAULT_TRACE, "00ca01c4")
    agents_by_stage: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        agents_by_stage[int(row["stage_id"])].add(str(row["agent_id"]))

    assert len(rows) == 35
    assert [len(agents_by_stage[stage]) for stage in sorted(agents_by_stage)] == [
        1,
        3,
        1,
    ]


def test_qwen_yarn_128k_replay_scales_prompt_and_output_together() -> None:
    rows = _load_workflow(DEFAULT_TRACE, "00ca01c4")
    replayed = [
        _scaled_tokens(
            row,
            input_scale=1.0,
            max_input_tokens=None,
            max_output_tokens=None,
            context_length=131072,
        )
        for row in rows
    ]

    assert sum(scale < 1.0 for *_, scale in replayed) == 16
    assert sum(input_scaled for _, _, input_scaled, _, _ in replayed) == 16
    assert sum(output_scaled for _, _, _, output_scaled, _ in replayed) == 15
    assert (
        max(input_len + output_len for input_len, output_len, *_ in replayed)
        == 131071
    )


def test_trace_token_carrier_is_prefix_stable_per_agent() -> None:
    rows = _load_workflow(DEFAULT_TRACE, "00ca01c4")
    main_rows = [row for row in rows if str(row["agent_id"]).endswith("-main")]
    sub_row = next(row for row in rows if str(row["agent_id"]).endswith("-sub1"))

    short = _trace_token_ids(main_rows[0], 128)
    long = _trace_token_ids(main_rows[1], 1024)

    assert long[: len(short)] == short
    assert _trace_token_ids(sub_row, 128) != short


def test_algorithm1_trace_position_and_payload_use_calibrated_slack() -> None:
    main = {"rid": "wf-p1-m2", "source_model_role": "main"}
    sub = {"rid": "wf-p2-s0-t4", "source_model_role": "subagent"}
    assert _algorithm1_trace_position(main) == ("main_phase", 3)
    assert _algorithm1_trace_position(sub) == ("subagent", 5)

    row = {
        **sub,
        "workflow_id": "workflow-a",
        "agent_id": "workflow-a-sub1",
        "stage_id": 2,
        "stage_width": 3,
        "stage_local_agent_index": 0,
        "critical_agent": True,
    }
    workflow = DemoWorkflow(Namespace(policy="aegiskv", model="Qwen/Qwen3-32B"), [row])
    sim = workflow._request_payload(
        row,
        request_id=row["rid"],
        input_tokens=1024,
        output_tokens=16,
    )["simulation"]

    # A single-node residual stage is on the CPM critical path.  Duration is
    # not Slack, even if the trace metadata says the original stage width was 3.
    assert sim["workflow_slack_ms"] == 0.0
    assert sim["aegis_slack_target_coverage"] == 0.9
    assert sim["aegis_slack_calibration_id"].startswith(
        "alg1-fixed-dag-runtime"
    )
    assert sim["aegis_slack_source"] == "fixed_dag_singleton_stage_exact_zero"
    assert sim["aegis_slack_semantics"] == "fixed_dag_cpm_slack_v2"
    assert sim["worker_seq"] == 1


def test_trace_oracle_uses_measured_api_completion_and_bypasses_algorithm1() -> None:
    rows = [
        {
            "rid": "wf-p1-s0-t0",
            "workflow_id": "wf",
            "agent_id": "agent-a",
            "stage_id": 1,
            "stage_width": 2,
            "stage_local_agent_index": 0,
            "group_id": 0,
            "arrival_t": 10.0,
            "trace_api_time_s": 4.0,
            "prompt_len": 128,
            "output_len": 8,
        },
        {
            "rid": "wf-p1-s1-t0",
            "workflow_id": "wf",
            "agent_id": "agent-b",
            "stage_id": 1,
            "stage_width": 2,
            "stage_local_agent_index": 1,
            "group_id": 1,
            "arrival_t": 11.0,
            "trace_api_time_s": 1.0,
            "prompt_len": 128,
            "output_len": 8,
        },
    ]
    annotate_trace_oracle_slack(rows)
    assert rows[0]["trace_oracle_slack_ms"] == 0.0
    assert rows[1]["trace_oracle_slack_ms"] == 2000.0

    workflow = DemoWorkflow(
        Namespace(
            policy="aegiskv",
            model="Qwen/Qwen3-32B",
            slack_source="trace_oracle",
        ),
        rows,
    )
    sim = workflow._request_payload(
        rows[1],
        request_id=rows[1]["rid"],
        input_tokens=128,
        output_tokens=8,
    )["simulation"]

    assert sim["workflow_slack_ms"] == 2000.0
    assert sim["critical_agent"] is False
    assert sim["aegis_critical_probability"] == 0.0
    assert sim["aegis_slack_source"] == "trace_oracle_future_api_completion"
    assert sim["aegis_slack_uses_future_information"] is True
    assert workflow.algorithm1_progress_trackers == {}


def test_formal_trace_oracle_has_five_exactly_matched_workflows() -> None:
    workflows = load_workflows(DEFAULT_TRACE_ORACLE)

    assert len(workflows) == 5
    assert sum(len(rows) for rows in workflows.values()) == 465
    assert all(
        bool(row.get("trace_oracle_uses_future_information"))
        for rows in workflows.values()
        for row in rows
    )


def test_lineage_worker_sequence_is_monotonic_across_main_agent_phases() -> None:
    rows = _load_workflow(DEFAULT_TRACE, "07dd4053")
    workflow = DemoWorkflow(
        Namespace(policy="aegiskv", model="Qwen/Qwen3-32B"),
        rows,
    )

    assert workflow.lineage_worker_seq["07dd4053-p1-m7"] == 8
    assert workflow.lineage_worker_seq["07dd4053-p3-m0"] == 9


def test_workflow_payload_carries_optional_execution_route_plan() -> None:
    row = {
        "rid": "wf-p1-m0",
        "workflow_id": "workflow-a",
        "agent_id": "workflow-a-main",
        "stage_id": 1,
        "stage_width": 1,
        "stage_local_agent_index": 0,
    }
    workflow = DemoWorkflow(
        Namespace(
            policy="aegiskv",
            model="Qwen/Qwen3-32B",
            execution_route_plan={"wf-p1-m0": 2},
        ),
        [row],
    )

    sim = workflow._request_payload(
        row,
        request_id=row["rid"],
        input_tokens=128,
        output_tokens=8,
    )["simulation"]

    assert sim["aegis_route_plan_dp_rank"] == 2


def test_measured_execution_route_plan_round_trip(tmp_path) -> None:
    source = tmp_path / "no_backup.json"
    source.write_text(
        json.dumps(
            {
                "policy": "no_backup",
                "model": "Qwen/Qwen3-32B",
                "dp_size": 4,
                "pp_size": 4,
                "request_count": 2,
                "workflows": [
                    {
                        "request_routes": [
                            {"rid": "r0", "execution_dp_rank": 1},
                            {"rid": "r1", "execution_dp_rank": 3},
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = build_execution_route_plan(source)
    output = tmp_path / "routes.json"
    output.write_text(json.dumps(plan), encoding="utf-8")

    loaded = load_execution_route_plan(output, dp_size=4)

    assert loaded["source_policy"] == "no_backup"
    assert loaded["route_count"] == 2
    assert loaded["routes"] == {
        "r0": {"execution_dp_rank": 1},
        "r1": {"execution_dp_rank": 3},
    }


def test_real_cc_trace_contains_20_fan_out_fan_in_workflows() -> None:
    workflows = load_workflows(DEFAULT_TRACE)
    selected = {
        workflow_id: rows
        for workflow_id, rows in workflows.items()
        if has_fan_out_and_main_fan_in(rows)
    }

    assert len(workflows) == 24
    assert sum(len(rows) for rows in workflows.values()) == 2068
    assert len(selected) == 20
    assert sum(len(rows) for rows in selected.values()) == 1755


def test_all_real_cc_workflows_respect_yarn_128k_boundary() -> None:
    workflows = load_workflows(DEFAULT_TRACE)
    workflows = {
        workflow_id: rows
        for workflow_id, rows in workflows.items()
        if has_fan_out_and_main_fan_in(rows)
    }
    replayed = [
        _scaled_tokens(
            row,
            input_scale=1.0,
            max_input_tokens=None,
            max_output_tokens=None,
            context_length=131072,
        )
        for rows in workflows.values()
        for row in rows
    ]

    assert sum(scale < 1.0 for *_, scale in replayed) == 446
    assert sum(input_scaled for _, _, input_scaled, _, _ in replayed) == 446
    assert sum(output_scaled for _, _, _, output_scaled, _ in replayed) == 444
    assert (
        max(input_len + output_len for input_len, output_len, *_ in replayed)
        == 131071
    )
    assert abs(min(scale for *_, scale in replayed) - 0.4131877347) < 1e-9


def test_ecdf_and_percentile_are_deterministic() -> None:
    xs, ys = ecdf([3.0, 1.0, 2.0, 4.0])

    assert xs == [1.0, 2.0, 3.0, 4.0]
    assert ys == [0.25, 0.5, 0.75, 1.0]
    assert percentile(xs, 0.50) == 2.5


def test_failure_manifest_fixes_time_rank_and_duration() -> None:
    manifest = load_failure_manifest(
        DEFAULT_FAILURE_MANIFEST,
        dp_size=8,
        failure_count=2,
    )

    assert manifest["time_origin"] == "policy_workload_start"
    assert [event["offset_s"] for event in manifest["events"]] == [12.0, 24.0]
    assert [event["dp_rank"] for event in manifest["events"]] == [6, 1]
    assert {event["duration_s"] for event in manifest["events"]} == {10.0}

    formal = load_failure_manifest(
        DEFAULT_FORMAL_FAILURE_MANIFEST,
        dp_size=8,
        failure_count=4,
    )
    assert [event["offset_s"] for event in formal["events"]] == [
        30.0,
        90.0,
        150.0,
        210.0,
    ]
    assert [event["dp_rank"] for event in formal["events"]] == [6, 1, 4, 7]
    assert {event["duration_s"] for event in formal["events"]} == {10.0}

    dp4_pp4_path = (
        Path(__file__).resolve().parents[3]
        / "AegisKV"
        / "demo"
        / "config"
        / "failure_manifest_qwen3_32b_dp4_pp4_32k.json"
    )
    dp4_pp4 = load_failure_manifest(
        dp4_pp4_path,
        dp_size=4,
        pp_size=4,
        failure_count=4,
    )
    assert [event["pp_rank"] for event in dp4_pp4["events"]] == [1, 3, 0, 2]
    assert {event["failure_scope"] for event in dp4_pp4["events"]} == {"gpu"}
    assert "simulator_boundary" in dp4_pp4["deployment"]

    node_base = load_failure_manifest(
        dp4_pp4_path.with_name(
            "failure_manifest_qwen3_32b_dp4_pp4_32k_node_base.json"
        ),
        dp_size=4,
        pp_size=4,
        failure_count=4,
    )
    assert [event["pp_rank"] for event in node_base["events"]] == [-1] * 4
    assert {event["failure_scope"] for event in node_base["events"]} == {
        "node"
    }


def test_failure_manifest_validates_and_preserves_matched_victims(tmp_path) -> None:
    path = tmp_path / "matched.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "time_origin": "policy_workload_start",
                "events": [
                    {
                        "event_id": "failure-01",
                        "offset_s": 12.0,
                        "dp_rank": 6,
                        "duration_s": 10.0,
                        "victims": [
                            {
                                "backup_key": "request-a",
                                "execution_dp_rank": 6,
                                "backup_dp_rank": 2,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_failure_manifest(path, dp_size=8)

    assert manifest["events"][0]["victims"] == [
        {
            "backup_key": "request-a",
            "execution_dp_rank": 6,
            "pp_rank": -1,
            "failure_scope": "node",
            "retry_route_mode": "manifest_fixed_calibration",
            "backup_dp_rank": 2,
        }
    ]


def test_demo_workflow_freezes_only_matched_victim_top2() -> None:
    row = {
        "rid": "request-a",
        "workflow_id": "workflow-a",
        "agent_id": "workflow-a-main",
        "stage_id": 1,
        "stage_width": 1,
    }
    args = Namespace(
        policy="aegiskv",
        model="Qwen/Qwen3-8B",
        failure_manifest={
            "events": [
                {
                    "event_id": "failure-01",
                    "offset_s": 12.0,
                    "dp_rank": 6,
                    "victims": [
                        {
                            "backup_key": "request-a",
                            "execution_dp_rank": 6,
                            "backup_dp_rank": 2,
                        }
                    ],
                }
            ]
        },
    )
    workflow = DemoWorkflow(args, [row])

    normal = workflow._request_payload(
        row,
        request_id="request-a",
        input_tokens=100,
        output_tokens=10,
    )["simulation"]
    retry = workflow._request_payload(
        row,
        request_id="request-a-retry",
        input_tokens=100,
        output_tokens=10,
        retry=True,
        failed_dp_rank=6,
        backup_key="request-a",
    )["simulation"]

    assert normal["force_dp_rank"] == 6
    assert normal["aegis_backup_dp_rank"] == 2
    assert normal["aegis_matched_failure_victim"] is True
    assert "force_dp_rank" not in retry
    assert retry["aegis_route_plan_retry_dp_rank"] == 2


def test_formal_failure_victim_does_not_override_router_or_retry_rank() -> None:
    row = {
        "rid": "request-a",
        "workflow_id": "workflow-a",
        "agent_id": "workflow-a-main",
        "stage_id": 1,
        "stage_width": 1,
    }
    args = Namespace(
        policy="aegiskv",
        model="Qwen/Qwen3-8B",
        execution_route_plan={"request-a": 1},
        failure_manifest={
            "schema_version": 2,
            "retry_route_mode": RETRY_ROUTE_MODE_ROUTER_COMMITTED,
            "events": [
                {
                    "event_id": "failure-01",
                    "offset_s": 12.0,
                    "dp_rank": 1,
                    "victims": [
                        {
                            "backup_key": "request-a",
                            "execution_dp_rank": 1,
                        }
                    ],
                }
            ],
        },
    )
    workflow = DemoWorkflow(args, [row])

    normal = workflow._request_payload(
        row,
        request_id="request-a",
        input_tokens=100,
        output_tokens=10,
    )["simulation"]
    retry = workflow._request_payload(
        row,
        request_id="request-a-retry",
        input_tokens=100,
        output_tokens=10,
        retry=True,
        failed_dp_rank=1,
        backup_key="request-a",
    )["simulation"]

    assert normal["aegis_route_plan_dp_rank"] == 1
    assert normal["aegis_matched_failure_victim"] is True
    assert normal["aegis_failure_retry_route_mode"] == "router_committed"
    assert "force_dp_rank" not in normal
    assert "aegis_backup_dp_rank" not in normal
    assert "aegis_route_plan_retry_dp_rank" not in retry
    assert retry["aegis_retry_route_source"] == (
        "router_committed_recovery_hint"
    )


def test_formal_failure_manifest_rejects_policy_backup_rank(tmp_path) -> None:
    path = tmp_path / "formal.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "time_origin": "policy_workload_start",
                "retry_route_mode": RETRY_ROUTE_MODE_ROUTER_COMMITTED,
                "events": [
                    {
                        "event_id": "failure-01",
                        "offset_s": 1.0,
                        "duration_s": 1.0,
                        "dp_rank": 1,
                        "victims": [
                            {
                                "backup_key": "request-a",
                                "execution_dp_rank": 1,
                                "backup_dp_rank": 0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not prescribe backup_dp_rank"):
        load_failure_manifest(path, dp_size=2)


def test_build_matched_manifest_uses_calibration_routes_and_retry_victims(
    tmp_path,
) -> None:
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "time_origin": "policy_workload_start",
                "events": [
                    {
                        "event_id": "failure-01",
                        "offset_s": 12.0,
                        "dp_rank": 6,
                        "duration_s": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "aegiskv.json"
    calibration.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "request_routes": [
                            {
                                "rid": "request-a",
                                "execution_dp_rank": 6,
                                "backup_dp_rank": 2,
                            }
                        ],
                        "retry_events": [
                            {
                                "backup_key": "request-a",
                                "failed_dp_rank": 6,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build_matched_manifest(
        calibration_result=calibration,
        base_manifest=base,
    )

    assert manifest["victim_count"] == 1
    assert manifest["schema_version"] == 2
    assert manifest["retry_route_mode"] == RETRY_ROUTE_MODE_ROUTER_COMMITTED
    assert manifest["events"][0]["victims"] == [
        {
            "backup_key": "request-a",
            "execution_dp_rank": 6,
            "retry_route_mode": RETRY_ROUTE_MODE_ROUTER_COMMITTED,
        }
    ]

    calibration_manifest = build_matched_manifest(
        calibration_result=calibration,
        base_manifest=base,
        retry_route_mode=RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION,
    )
    assert calibration_manifest["schema_version"] == 1
    assert calibration_manifest["events"][0]["victims"][0][
        "backup_dp_rank"
    ] == 2


def test_build_matched_manifest_infers_dp4_pp4_deployment(tmp_path) -> None:
    base = tmp_path / "base-dp4-pp4.json"
    base.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "time_origin": "policy_workload_start",
                "deployment": {"dp_size": 4, "pp_size": 4},
                "events": [
                    {
                        "event_id": "failure-01",
                        "offset_s": 45.0,
                        "dp_rank": 3,
                        "pp_rank": 3,
                        "duration_s": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "aegiskv-dp4-pp4.json"
    calibration.write_text(
        json.dumps(
            {
                "dp_size": 4,
                "pp_size": 4,
                "workflows": [
                    {
                        "request_routes": [
                            {
                                "rid": "request-pp3",
                                "execution_dp_rank": 3,
                                "backup_dp_rank": 0,
                                "completion_elapsed_s": 20.0,
                            }
                        ],
                        "retry_events": [
                            {
                                "backup_key": "request-pp3",
                                "failed_dp_rank": 3,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_matched_manifest(
        calibration_result=calibration,
        base_manifest=base,
    )

    assert manifest["resolved_dp_size"] == 4
    assert manifest["resolved_pp_size"] == 4
    assert manifest["events"][0]["pp_rank"] == 3
    assert manifest["victim_count"] == 1


def test_wct_suite_resolves_formal_and_quick_profiles() -> None:
    parser = build_parser()

    formal = resolve_profile(parser.parse_args([]))
    assert formal.failure_manifest == DEFAULT_FORMAL_FAILURE_MANIFEST
    assert formal.failure_count == 4
    assert formal.workflow_concurrency == 12
    assert formal.max_workflows == 12
    assert formal.max_requests_per_workflow == 128
    assert formal.input_scale == 1.0
    assert formal.context_length == 20480
    assert formal.arrival_speedup == 0.0

    quick = resolve_profile(parser.parse_args(["--quick"]))
    assert quick.failure_manifest == DEFAULT_QUICK_FAILURE_MANIFEST
    assert quick.failure_count == 2
    assert quick.workflow_concurrency == 20
    assert quick.max_workflows == 20
    assert quick.max_requests_per_workflow == 0
    assert quick.input_scale == 0.008
    assert quick.context_length == 1024
    assert quick.max_input_tokens == 1008
    assert quick.max_output_tokens == 16
    assert ROOT == Path(__file__).resolve().parents[3]


def test_wct_suite_preserves_explicit_failure_overrides(tmp_path) -> None:
    manifest = tmp_path / "custom.json"
    args = resolve_profile(
        build_parser().parse_args(
            [
                "--quick",
                "--failure-manifest",
                str(manifest),
                "--failure-count",
                "1",
            ]
        )
    )

    assert args.failure_manifest == manifest
    assert args.failure_count == 1


def test_wct_suite_accepts_bounded_policy_subset() -> None:
    args = resolve_profile(
        build_parser().parse_args(["--policies", "l1_only", "l2_only"])
    )

    assert args.policies == ["l1_only", "l2_only"]
    assert args.context_length == 20480


def test_wct_suite_derives_32k_context_and_dp4_pp4_from_server_config() -> None:
    config = (
        Path(__file__).resolve().parents[3]
        / "AegisKV"
        / "demo"
        / "config"
        / "server_qwen3_32b_dp4_pp4_32k.json"
    )
    args = resolve_profile(
        build_parser().parse_args(["--server-config", str(config)])
    )

    raw = json.loads(config.read_text(encoding="utf-8"))
    assert args.context_length == 32768
    assert raw["scheduler"]["dp_size"] == 4
    assert raw["scheduler"]["pp_size"] == 4
    assert raw["aegis_l1_l2"]["failure_hazard_probability"] == 0.01
    assert raw["aegis_l1_l2"]["cost_weight"] == 0.35
    assert raw["aegis_l1_l2"]["max_alpha"] == 0.60
    assert raw["aegis_l1_l2"]["rehome_after_recovery"] is True


def test_wct_suite_snapshots_recompute_penalty(tmp_path) -> None:
    config = tmp_path / "server.json"
    config.write_text(
        json.dumps(
            {
                "aegis_l1_l2": {
                    "recompute_setup_ms": 0.0,
                    "recompute_ms_per_token": 2.2,
                }
            }
        ),
        encoding="utf-8",
    )
    args = resolve_profile(
        build_parser().parse_args(["--quick", "--server-config", str(config)])
    )

    snapshot = snapshot_suite_configuration(args, tmp_path)
    suite = json.loads((tmp_path / "suite_config.json").read_text(encoding="utf-8"))

    assert snapshot == tmp_path / "server_config.json"
    assert suite["profile"] == "quick-1k"
    assert suite["recompute_ms_per_token"] == 2.2
    assert suite["recompute_penalty_multiplier"] == 10.0


def test_formal_20k_profile_scales_every_request_below_boundary() -> None:
    workflows = load_workflows(DEFAULT_TRACE)
    selected = {
        workflow_id: rows
        for workflow_id, rows in workflows.items()
        if has_fan_out_and_main_fan_in(rows) and len(rows) <= 128
    }
    selected = dict(list(sorted(selected.items()))[:12])
    replayed = [
        _scaled_tokens(
            row,
            input_scale=1.0,
            max_input_tokens=None,
            max_output_tokens=None,
            context_length=20480,
        )
        for rows in selected.values()
        for row in rows
    ]

    assert len(replayed) == 1043
    assert sum(input_len + output_len for input_len, output_len, *_ in replayed) == 20866471
    assert sum(scale < 1.0 for *_, scale in replayed) == 1014
    assert max(input_len + output_len for input_len, output_len, *_ in replayed) == 20479


def test_wct_suite_resume_reuses_only_complete_policy_results(tmp_path) -> None:
    result = tmp_path / "aegiskv.json"
    result.write_text(
        json.dumps(
            {
                "policy": "aegiskv",
                "workflow_count": 2,
                "workflows": [{"workflow_id": "a"}, {"workflow_id": "b"}],
            }
        ),
        encoding="utf-8",
    )

    assert policy_result_is_complete(
        result,
        policy="aegiskv",
        expected_workflows=2,
    )
    assert not policy_result_is_complete(
        result,
        policy="l1_only",
        expected_workflows=2,
    )
    assert not policy_result_is_complete(
        result,
        policy="aegiskv",
        expected_workflows=3,
    )

    result.write_text(
        json.dumps(
            {
                "policy": "aegiskv",
                "eligible_workflow_count": 2,
                "workflow_count": 2,
                "workflows": [
                    {"workflow_id": "a"},
                    {"workflow_id": "b"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert policy_result_is_complete(
        result,
        policy="aegiskv",
        expected_workflows=23,
    )


def test_detailed_plot_separates_replica_alpha_from_node_beta() -> None:
    payload = {
        "rank_alpha_updates": [
            {
                "sequence": 10,
                "dp_rank": 2,
                "window_id": 3,
                "trigger": "target_time",
                "old_alpha": 0.08,
                "new_alpha": 0.12,
                "old_beta": 0.25,
                "new_beta": 0.25,
            }
        ],
        "node_beta_updates": [
            {
                "sequence": 11,
                "node_id": 2,
                "window_id": 4,
                "trigger": "overflow_shock",
                "D": 1024,
                "U": 768,
                "O": 256,
                "old_beta": 0.25,
                "new_beta": 0.30,
            }
        ],
    }

    assert _rank_alpha_rows(payload) == [
        {
            "sequence": 10,
            "dp_rank": 2,
            "window_id": 3,
            "trigger": "target_time",
            "old_alpha": 0.08,
            "new_alpha": 0.12,
        }
    ]
    assert _node_beta_rows(payload) == [
        {
            "sequence": 11,
            "node_id": 2,
            "window_id": 4,
            "trigger": "overflow_shock",
            "D": 1024,
            "U": 768,
            "O": 256,
            "old_beta": 0.25,
            "new_beta": 0.30,
        }
    ]


def test_qwen3_final_gate_reads_matched_paired_profile(tmp_path) -> None:
    common = {
        "evidence_tier": "sglsim-online-measured",
        "model": "Qwen/Qwen3-32B",
        "dp_size": 4,
        "pp_size": 4,
        "workflow_filter": "fan_out_then_main_fan_in",
        "workflow_concurrency": 4,
        "failure_manifest": {
            "deployment": {"failure_scope": "gpu"},
            "events": [{"event_id": "failure-01"}],
        },
        "execution_route_plan": {
            "schema_version": 1,
            "dp_size": 4,
            "routes": {"r0": {"execution_dp_rank": 0}},
        },
        "execution_route_plan_audit": {
            "enabled": True,
            "exact_match": True,
            "violations": [],
        },
        "replay": {"context_length": 32768},
        "failure_count_injected": 1,
        "matched_victim_audit": {
            "enabled": True,
            "exact_match": True,
            "route_violations": [],
            "timing_violations": [],
        },
    }
    (tmp_path / "aegiskv.json").write_text(
        json.dumps(
            {
                **common,
                "workflows": [
                    {"workflow_id": "a", "wct_s": 9.0},
                    {"workflow_id": "b", "wct_s": 18.0},
                ],
                "wct_summary_s": {"p90": 18.0},
                "retry_summary": {
                    "count": 1,
                    "restored_tokens": 80,
                    "recompute_tokens": 20,
                    "restored_fraction": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "no_backup.json").write_text(
        json.dumps(
            {
                **common,
                "workflows": [
                    {"workflow_id": "a", "wct_s": 10.0},
                    {"workflow_id": "b", "wct_s": 20.0},
                ],
                "wct_summary_s": {"p90": 20.0},
                "retry_summary": {
                    "count": 1,
                    "restored_tokens": 0,
                    "recompute_tokens": 100,
                    "restored_fraction": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    profile, errors = paired_profile(tmp_path)

    assert errors == []
    assert profile["p90_wct_gain_percent"] == 10.0
    assert profile["mean_wct_gain_percent"] == 10.0
    assert profile["workflow_win_fraction"] == 1.0
    assert profile["restored_fraction"] == 0.8

    aegis = json.loads((tmp_path / "aegiskv.json").read_text(encoding="utf-8"))
    aegis["matched_victim_audit"]["timing_violations"] = [
        {"backup_key": "r0", "retry_after_injection": False}
    ]
    (tmp_path / "aegiskv.json").write_text(
        json.dumps(aegis), encoding="utf-8"
    )
    _profile, errors = paired_profile(tmp_path)
    assert any("retry timing violations" in error for error in errors)


def test_wct_suite_resume_archives_interrupted_policy_artifacts(tmp_path) -> None:
    partial = tmp_path / "l1_only.partial.json"
    log = tmp_path / "server-l1_only.log"
    partial.write_text("{}", encoding="utf-8")
    log.write_text("interrupted", encoding="utf-8")

    archived = archive_interrupted_artifacts(tmp_path, "l1_only")

    assert {path.suffix for path in archived} == {".json", ".log"}
    assert all(".interrupted-" in path.name for path in archived)
    assert all(path.exists() for path in archived)
    assert not partial.exists()
    assert not log.exists()


def test_wct_suite_sigterm_unwinds_cleanup() -> None:
    try:
        _terminate_cleanly(15, None)
    except SystemExit as exc:
        assert exc.code == 143
    else:
        raise AssertionError("SIGTERM handler must unwind the runner")


def test_wct_suite_lock_ignores_stale_or_reused_pid_text(tmp_path) -> None:
    (tmp_path / ".suite.lock").write_text(str(os.getpid()), encoding="utf-8")

    lock = acquire_run_lock(tmp_path)
    try:
        assert (tmp_path / ".suite.lock").read_text(encoding="utf-8") == str(
            os.getpid()
        )
    finally:
        lock.close()


def test_wct_suite_supports_versioned_global_lock_name(tmp_path) -> None:
    lock = acquire_run_lock(tmp_path, lock_name=".suite-v2.lock")
    try:
        assert (tmp_path / ".suite-v2.lock").exists()
        assert not (tmp_path / ".suite.lock").exists()
    finally:
        lock.close()


def test_wct_suite_run_directory_lock_rejects_concurrent_runner(tmp_path) -> None:
    first = acquire_run_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already used by suite PID"):
            acquire_run_lock(tmp_path)
    finally:
        first.close()

    second = acquire_run_lock(tmp_path)
    second.close()


def test_policy_progress_checkpoint_is_atomic_and_sorted(tmp_path) -> None:
    output = tmp_path / "aegiskv.json"
    summaries = [
        {"workflow_id": "b", "wct_s": 2.0},
        {"workflow_id": "a", "wct_s": 1.0},
    ]

    path = write_progress_checkpoint(
        output=output,
        policy="aegiskv",
        summaries=summaries,
        planned_workflows=20,
        total_requests=1755,
        failure_events=[{"event_id": "failure-01"}],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == progress_output_path(output)
    assert payload["complete"] is False
    assert payload["completed_workflows"] == 2
    assert payload["planned_workflows"] == 20
    assert payload["request_count"] == 1755
    assert [item["workflow_id"] for item in payload["workflows"]] == ["a", "b"]
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_ordered_workflow_launch_gate_has_stable_starts_and_backfill() -> None:
    gate = OrderedWorkflowLaunchGate(
        workflow_count=5,
        concurrency=2,
        start_monotonic=time.monotonic(),
        stagger_s=0.01,
    )
    durations = [0.06, 0.01, 0.01, 0.01, 0.0]
    started: dict[int, float] = {}
    finished: dict[int, float] = {}
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def run(index: int) -> None:
        nonlocal active, peak_active
        gate.enter(index)
        try:
            with lock:
                started[index] = time.monotonic()
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(durations[index])
            with lock:
                active -= 1
                finished[index] = time.monotonic()
        finally:
            gate.leave(index)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert started[0] < started[1]
    assert started[2] >= finished[0]
    assert started[3] >= finished[1]
    assert started[4] >= finished[2]
    assert peak_active == 2


def test_failure_status_timeout_is_diagnostic_not_workflow_fatal(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise TimeoutError("transient control-plane timeout")

    monkeypatch.setattr(
        "AegisKV.demo.workflow_client.request_json",
        timeout,
    )
    workflow = DemoWorkflow(
        Namespace(base_url="http://127.0.0.1:1", failure_status_timeout=0.01),
        [{"workflow_id": "wf", "rid": "r0"}],
    )

    workflow._maybe_retry_failed_rank()

    assert workflow.retry_lock.acquire(blocking=False)
    workflow.retry_lock.release()
    assert [event["kind"] for event in workflow.events] == [
        "failure_status_unavailable"
    ]
    assert workflow.events[0]["error_type"] == "TimeoutError"


def test_matched_retries_fire_after_rank_recovers_and_include_all_due_victims(
    monkeypatch,
) -> None:
    class InjectedEvents:
        def __init__(self) -> None:
            self.ids: set[str] = set()

        def was_injected(self, event_id: str) -> bool:
            return event_id in self.ids

    injected_events = InjectedEvents()
    victims = [
        {
            "backup_key": "r0",
            "execution_dp_rank": 2,
            "backup_dp_rank": 0,
        },
        {
            "backup_key": "r1",
            "execution_dp_rank": 2,
            "backup_dp_rank": 1,
        },
    ]
    workflow = DemoWorkflow(
        Namespace(
            base_url="http://127.0.0.1:1",
            failure_status_timeout=0.01,
            failure_manifest={
                "events": [
                    {
                        "event_id": "failure-01",
                        "offset_s": 0.0,
                        "dp_rank": 2,
                        "victims": victims,
                    }
                ]
            },
            failure_coordinator=injected_events,
        ),
        [
            {"workflow_id": "wf", "agent_id": "shared", "rid": "r0"},
            {"workflow_id": "wf", "agent_id": "shared", "rid": "r1"},
        ],
    )
    workflow.completed = [
        {
            "rid": victim["backup_key"],
            "execution_dp_rank": 2,
            "completion_index": index,
            "row": {"agent_id": "shared"},
        }
        for index, victim in enumerate(victims, start=1)
    ]
    retried: list[tuple[str, int]] = []

    # The manifest offset has elapsed, but the server has already recovered the
    # rank. Exact matched replay must therefore not depend on the instantaneous
    # failure-status snapshot, nor collapse multiple victims from one agent.
    monkeypatch.setattr(
        "AegisKV.demo.workflow_client.request_json",
        lambda *args, **kwargs: {
            "soft_failed_ranks": [],
            "hard_failed_ranks": [],
        },
    )
    monkeypatch.setattr(
        workflow,
        "_retry",
        lambda record, failed_dp_rank: retried.append(
            (str(record["rid"]), int(failed_dp_rank))
        ),
    )

    workflow._maybe_retry_failed_rank()
    assert retried == []
    assert workflow.handled_failure_keys == set()

    injected_events.ids.add("failure-01")
    workflow._maybe_retry_failed_rank()

    assert retried == [("r0", 2), ("r1", 2)]
    assert workflow.handled_failure_keys == {"r0", "r1"}


def test_failure_manifest_offsets_do_not_wait_for_previous_recovery(
    monkeypatch,
) -> None:
    second_injected = threading.Event()
    injected_ranks: list[int] = []

    def fake_request_json(base_url, path, *, payload=None, **kwargs):
        rank = int(payload["dp_rank"])
        if path == "/simulate/fail_dp":
            injected_ranks.append(rank)
            if rank == 1:
                second_injected.set()
            return {"accepted": True}
        if path == "/simulate/recover_dp":
            if rank == 0:
                assert second_injected.wait(1.0)
            return {"released": True, "l1_clear": {}}
        raise AssertionError(path)

    monkeypatch.setattr(
        "AegisKV.demo.wct_cdf_client.request_json",
        fake_request_json,
    )
    coordinator = FailureManifestCoordinator(
        base_url="http://127.0.0.1:1",
        manifest={
            "events": [
                {
                    "event_id": "failure-00",
                    "offset_s": 0.0,
                    "duration_s": 0.0,
                    "dp_rank": 0,
                },
                {
                    "event_id": "failure-01",
                    "offset_s": 0.01,
                    "duration_s": 0.0,
                    "dp_rank": 1,
                },
            ]
        },
        start_monotonic=time.monotonic(),
    )

    coordinator.start()
    coordinator.join()

    assert injected_ranks == [0, 1]
    assert coordinator.injected_failures == 2
    assert coordinator.was_injected("failure-00")
    assert coordinator.was_injected("failure-01")
    assert coordinator.injected_offset_s("failure-00") is not None
    assert coordinator.injected_offset_s("failure-01") is not None
    assert [event["event_id"] for event in coordinator.snapshot_events()] == [
        "failure-00",
        "failure-01",
    ]
