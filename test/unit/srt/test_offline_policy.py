from __future__ import annotations

import argparse
import json

from AegisKV.demo.offline_policy import OfflinePolicyReplay
from AegisKV.demo.workflow_client import DEFAULT_SLACK_CALIBRATION
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster


def _workflow(workflow_id: str) -> list[dict]:
    return [
        {
            "workflow_id": workflow_id,
            "rid": f"{workflow_id}-a0",
            "stage_id": 0,
            "agent_id": f"{workflow_id}-a",
            "group_id": 0,
            "arrival_t": 0.0,
            "prompt_len": 32,
            "output_len": 2,
            "stage_width": 2,
            "critical_agent": True,
            "source_model_role": "subagent",
        },
        {
            "workflow_id": workflow_id,
            "rid": f"{workflow_id}-b0",
            "stage_id": 0,
            "agent_id": f"{workflow_id}-b",
            "group_id": 0,
            "arrival_t": 0.0,
            "prompt_len": 48,
            "output_len": 2,
            "stage_width": 2,
            "critical_agent": False,
            "source_model_role": "subagent",
        },
        {
            "workflow_id": workflow_id,
            "rid": f"{workflow_id}-main0",
            "stage_id": 1,
            "agent_id": f"{workflow_id}-main",
            "group_id": 0,
            "arrival_t": 1.0,
            "prompt_len": 64,
            "output_len": 2,
            "stage_width": 1,
            "critical_agent": True,
            "source_model_role": "main",
        },
    ]


def test_offline_policy_preserves_ordered_predecessor_launch(tmp_path) -> None:
    config_path = tmp_path / "offline-policy.json"
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
    selected = [("wf0", _workflow("wf0")), ("wf1", _workflow("wf1"))]
    routes = {
        "wf0-a0": 0,
        "wf0-b0": 1,
        "wf0-main0": 0,
        "wf1-a0": 1,
        "wf1-b0": 0,
        "wf1-main0": 1,
    }
    args = argparse.Namespace(
        model="mock-model",
        policy="aegiskv",
        slack_calibration=DEFAULT_SLACK_CALIBRATION,
        input_scale=1.0,
        max_input_tokens=None,
        max_output_tokens=None,
        context_length=4096,
        arrival_speedup=0.0,
        execution_route_plan=routes,
        backup_route_plan={
            "wf0-a0": 1,
            "wf0-b0": 0,
            "wf0-main0": 1,
            "wf1-a0": 0,
            "wf1-b0": 1,
            "wf1-main0": 0,
        },
        failure_manifest={"schema_version": 1, "events": []},
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
        result = OfflinePolicyReplay(
            cluster=cluster,
            args=args,
            selected_workflows=selected,
            failure_manifest={"schema_version": 1, "events": []},
            workflow_concurrency=1,
            launch_stagger_s=0.25,
        ).run(max_events=5000)

    assert len(result.workflows) == 2
    assert [event["workflow_id"] for event in result.launch_events] == ["wf0", "wf1"]
    assert result.launch_events[0]["launched_at_s"] == 0.25
    assert result.launch_events[1]["launched_at_s"] >= result.workflows[0][
        "completed_at_s"
    ]
    assert all(workflow["request_count"] == 3 for workflow in result.workflows)
    assert sum(workflow["retry_count"] for workflow in result.workflows) == 0
    observed_backups = {
        route["rid"]: route["backup_dp_rank"]
        for workflow in result.workflows
        for route in workflow["request_routes"]
    }
    assert observed_backups == args.backup_route_plan
