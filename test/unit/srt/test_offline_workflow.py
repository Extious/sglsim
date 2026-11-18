from __future__ import annotations

import argparse
import json

from AegisKV.demo.offline_workflow import OfflineDemoWorkflow
from AegisKV.demo.workflow_client import DEFAULT_SLACK_CALIBRATION
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster


def _row(
    rid: str,
    *,
    stage: int,
    agent: str,
    order: int,
    arrival: float,
    width: int,
) -> dict:
    return {
        "workflow_id": "offline-workflow-smoke",
        "source_dataset": "unit-real-cc-shape",
        "rid": rid,
        "stage_id": stage,
        "agent_id": agent,
        "group_id": order,
        "arrival_t": arrival,
        "prompt_len": 32 + order * 8,
        "output_len": 2,
        "stage_width": width,
        "stage_local_agent_index": 0 if agent.endswith("a") else 1,
        "critical_agent": agent.endswith("a") or agent.endswith("main"),
        "source_model_role": "main" if agent.endswith("main") else "subagent",
    }


def test_offline_workflow_preserves_agent_sequence_and_stage_barrier(tmp_path) -> None:
    config_path = tmp_path / "offline-workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "pp_size": 1,
                    "load_balance_method": "cache_aware",
                    "max_prefill_tokens": 1024,
                    "chunked_prefill_size": 1024,
                    "max_running_requests": 16,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 4096,
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "initial_alpha": 0.5,
                    "initial_beta": 0.5,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        _row("p0-a-t0", stage=0, agent="agent-a", order=0, arrival=0.0, width=2),
        _row("p0-a-t1", stage=0, agent="agent-a", order=1, arrival=1.0, width=2),
        _row("p0-b-t0", stage=0, agent="agent-b", order=0, arrival=0.0, width=2),
        _row("p0-b-t1", stage=0, agent="agent-b", order=1, arrival=1.0, width=2),
        _row("p1-main-t0", stage=1, agent="agent-main", order=0, arrival=2.0, width=1),
    ]
    route_plan = {
        "p0-a-t0": 0,
        "p0-a-t1": 0,
        "p0-b-t0": 1,
        "p0-b-t1": 1,
        "p1-main-t0": 0,
    }
    args = argparse.Namespace(
        model="mock-model",
        policy="aegiskv",
        slack_calibration=DEFAULT_SLACK_CALIBRATION,
        input_scale=1.0,
        max_input_tokens=None,
        max_output_tokens=None,
        context_length=4096,
        arrival_speedup=1.0,
        execution_route_plan=route_plan,
        failure_manifest={},
        policy_start_monotonic=0.0,
        stage_pause=0.0,
        quiet=True,
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        result = OfflineDemoWorkflow(
            cluster=cluster,
            args=args,
            rows=rows,
        ).run(max_events=1000)

    workflow = result.workflow
    assert workflow["request_count"] == 5
    assert workflow["retry_count"] == 0
    assert workflow["wct_s"] > 2.0
    routes = {
        row["rid"]: row["execution_dp_rank"]
        for row in workflow["request_routes"]
    }
    assert routes == route_plan
    completion = {
        row["rid"]: row["completion_elapsed_s"]
        for row in workflow["request_routes"]
    }
    assert completion["p0-a-t0"] <= completion["p0-a-t1"]
    assert completion["p0-b-t0"] <= completion["p0-b-t1"]
    assert completion["p1-main-t0"] >= max(
        completion["p0-a-t1"], completion["p0-b-t1"]
    )
    assert len(workflow["algorithm1_slack_evaluation"]) == 3


def test_offline_workflow_replays_matched_failure_victim(tmp_path) -> None:
    config_path = tmp_path / "offline-workflow-failure.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "pp_size": 1,
                    "load_balance_method": "cache_aware",
                    "max_prefill_tokens": 1024,
                    "chunked_prefill_size": 1024,
                    "max_running_requests": 16,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 4096,
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "initial_alpha": 0.5,
                    "initial_beta": 0.5,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        _row("p0-a-t0", stage=0, agent="agent-a", order=0, arrival=0.0, width=2),
        _row("p0-a-t1", stage=0, agent="agent-a", order=1, arrival=1.0, width=2),
        _row("p0-b-t0", stage=0, agent="agent-b", order=0, arrival=0.0, width=2),
        _row("p1-main-t0", stage=1, agent="agent-main", order=0, arrival=2.0, width=1),
    ]
    manifest = {
        "schema_version": 2,
        "retry_route_mode": "router_committed",
        "deployment": {"failure_scope": "gpu"},
        "events": [
            {
                "event_id": "matched-failure",
                "offset_s": 1.5,
                "dp_rank": 0,
                "duration_s": 0.1,
                "victims": [
                    {
                        "backup_key": "p0-a-t0",
                        "execution_dp_rank": 0,
                        "failure_scope": "gpu",
                    }
                ],
            }
        ],
    }
    args = argparse.Namespace(
        model="mock-model",
        policy="aegiskv",
        slack_calibration=DEFAULT_SLACK_CALIBRATION,
        input_scale=1.0,
        max_input_tokens=None,
        max_output_tokens=None,
        context_length=4096,
        arrival_speedup=1.0,
        execution_route_plan={
            "p0-a-t0": 0,
            "p0-a-t1": 0,
            "p0-b-t0": 1,
            "p1-main-t0": 0,
        },
        failure_manifest=manifest,
        policy_start_monotonic=0.0,
        stage_pause=0.0,
        quiet=True,
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        cluster.schedule_soft_failure(
            event_id="matched-failure",
            dp_rank=0,
            offset_s=1.5,
            duration_s=0.1,
            payload={"victims": manifest["events"][0]["victims"]},
        )
        result = OfflineDemoWorkflow(
            cluster=cluster,
            args=args,
            rows=rows,
        ).run(max_events=2000)

    workflow = result.workflow
    assert workflow["request_count"] == 4
    assert workflow["retry_count"] == 1
    retry = workflow["retry_events"][0]
    assert retry["backup_key"] == "p0-a-t0"
    assert retry["retry_dp_rank"] == 1
    assert retry["expected_retry_dp_rank"] == 1
    assert retry["expected_retry_rank_source"] == (
        "original_router_immutable_backup"
    )
    assert retry["retry_route_mode"] == "router_committed"
    assert retry["event_id"] == "matched-failure"
    assert result.cluster.failure_events[0]["event_id"] == "matched-failure"
