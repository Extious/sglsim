"""Run one Real-CC workflow through the event-driven SGLSim offline cluster.

This is the first reusable calibration entrypoint.  It intentionally runs a
single workflow without failures; matched failure/retry orchestration belongs
to the multi-workflow policy driver and must not be inferred from this smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.calibration_metrics import (
    build_offline_cluster_artifact,
)
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster

from .offline_workflow import OfflineDemoWorkflow
from .evidence_journal import (
    bind_fresh_event_journal,
    event_journal_path_for_artifact,
)
from .wct_cdf_client import (
    has_fan_out_and_main_fan_in,
    load_execution_route_plan,
    load_failure_manifest,
    load_workflows,
)
from .workflow_client import DEFAULT_SLACK_CALIBRATION, DEFAULT_TRACE, SLACK_SOURCES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_CONFIG = (
    Path(__file__).resolve().parent
    / "config"
    / "server_qwen3_32b_dp4_pp4_32k.json"
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _select_workflow(
    trace: Path,
    *,
    workflow_prefix: str | None,
    max_requests: int,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = [
        (workflow_id, rows)
        for workflow_id, rows in load_workflows(trace).items()
        if has_fan_out_and_main_fan_in(rows)
        and (max_requests <= 0 or len(rows) <= max_requests)
        and (
            workflow_prefix is None
            or workflow_id.startswith(str(workflow_prefix))
        )
    ]
    if not candidates:
        suffix = (
            f" matching prefix {workflow_prefix!r}"
            if workflow_prefix is not None
            else ""
        )
        raise ValueError(f"no eligible fan-out/fan-in workflow{suffix} in {trace}")
    return candidates[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    slack_source = str(getattr(args, "slack_source", "algorithm1"))
    server_config_path = args.server_config.resolve()
    trace_path = args.trace.resolve()
    output_path = args.output.resolve()
    event_journal_path = event_journal_path_for_artifact(output_path)
    config = json.loads(server_config_path.read_text(encoding="utf-8"))
    scheduler = dict(config.get("scheduler", {}))
    model = str(args.model or config.get("model_path") or "mock-model")
    dp_size = int(args.dp_size or scheduler.get("dp_size", 1))
    pp_size = int(args.pp_size or scheduler.get("pp_size", 1))
    context_length = int(args.context_length or config.get("context_length", 131072))
    workflow_id, rows = _select_workflow(
        trace_path,
        workflow_prefix=args.workflow_prefix,
        max_requests=int(args.max_requests_per_workflow),
    )
    selected_rids = {str(row["rid"]) for row in rows}

    failure_manifest_path: Path | None = None
    failure_manifest: dict[str, Any] = {
        "schema_version": 1,
        "time_origin": "policy_workload_start",
        "deployment": {},
        "events": [],
    }
    if args.failure_manifest is not None and int(args.failure_count) > 0:
        failure_manifest_path = args.failure_manifest.resolve()
        loaded_manifest = load_failure_manifest(
            failure_manifest_path,
            dp_size=dp_size,
            pp_size=pp_size,
            failure_count=int(args.failure_count),
        )
        relevant_events = []
        for event in loaded_manifest["events"]:
            victims = [
                dict(victim)
                for victim in event.get("victims", ())
                if str(victim.get("backup_key", "")) in selected_rids
            ]
            if victims:
                relevant_events.append({**dict(event), "victims": victims})
        failure_manifest = {
            **loaded_manifest,
            "events": relevant_events,
        }

    route_plan: dict[str, int] = {}
    route_plan_path: Path | None = None
    if args.route_plan is not None:
        route_plan_path = args.route_plan.resolve()
        route_payload = load_execution_route_plan(
            route_plan_path,
            dp_size=dp_size,
        )
        route_plan = {
            str(rid): int(spec["execution_dp_rank"])
            for rid, spec in route_payload["routes"].items()
        }
        missing = sorted(str(row["rid"]) for row in rows if str(row["rid"]) not in route_plan)
        if missing:
            raise ValueError(
                f"execution route plan misses {len(missing)} selected requests; "
                f"first={missing[0]}"
            )

    workflow_args = argparse.Namespace(
        model=model,
        policy=args.policy,
        slack_calibration=args.slack_calibration.resolve(),
        slack_source=slack_source,
        input_scale=float(args.input_scale),
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        context_length=context_length,
        arrival_speedup=float(args.arrival_speedup),
        execution_route_plan=route_plan,
        failure_manifest=failure_manifest,
        policy_start_monotonic=0.0,
        stage_pause=float(args.stage_pause),
        quiet=True,
    )
    server_args = ServerArgs(
        model_path=model,
        dp_size=dp_size,
        pp_size=pp_size,
        load_balance_method=str(
            scheduler.get("load_balance_method", "cache_aware")
        ),
        streaming_arrival=True,
        sim_config_path=str(server_config_path),
        log_level="warning",
    )

    started = time.monotonic()
    with bind_fresh_event_journal(event_journal_path):
        with OfflineSchedulerCluster(server_args) as cluster:
            for event in failure_manifest["events"]:
                cluster.schedule_soft_failure(
                    event_id=str(event["event_id"]),
                    dp_rank=int(event["dp_rank"]),
                    offset_s=float(event["offset_s"]),
                    duration_s=float(event["duration_s"]),
                    payload={
                        key: value
                        for key, value in event.items()
                        if key
                        not in {
                            "event_id",
                            "dp_rank",
                            "offset_s",
                            "duration_s",
                        }
                    },
                )
            result = OfflineDemoWorkflow(
                cluster=cluster,
                args=workflow_args,
                rows=rows,
            ).run(max_events=int(args.max_events))
            host_runtime_s = time.monotonic() - started

    artifact = build_offline_cluster_artifact(
        result.cluster,
        identity={
            "model": model,
            "dp_size": dp_size,
            "pp_size": pp_size,
            "context_length": context_length,
            "server_config": str(server_config_path),
            "trace": str(trace_path),
            "route_plan": str(route_plan_path) if route_plan_path else None,
            "failure_manifest": (
                str(failure_manifest_path) if failure_manifest_path else None
            ),
            "failure_count": len(failure_manifest["events"]),
            "workflow_id": workflow_id,
            "policy": args.policy,
            "slack_source": slack_source,
            "trace_oracle_upper_bound": slack_source == "trace_oracle",
            "artifact_path": str(output_path),
            "event_journal": str(event_journal_path),
        },
    )
    observed_routes = artifact["calibration_metrics"]["routes"]
    selected_expected_routes = {
        str(row["rid"]): route_plan[str(row["rid"])]
        for row in rows
        if str(row["rid"]) in route_plan
    }
    route_violations = {
        rid: {"expected": expected, "observed": observed_routes.get(rid)}
        for rid, expected in selected_expected_routes.items()
        if observed_routes.get(rid) != expected
    }
    retry_route_violations = []
    for retry in result.workflow["retry_events"]:
        victim = next(
            (
                victim
                for event in failure_manifest["events"]
                for victim in event.get("victims", ())
                if str(victim.get("backup_key", "")) == retry["backup_key"]
            ),
            {},
        )
        expected_retry_rank = int(victim.get("backup_dp_rank", -1))
        if int(retry["retry_dp_rank"]) != expected_retry_rank:
            retry_route_violations.append(
                {
                    "rid": retry["rid"],
                    "expected": expected_retry_rank,
                    "observed": int(retry["retry_dp_rank"]),
                }
            )
    artifact.update(
        {
            "policy": args.policy,
            "workflow_count": 1,
            "request_count": len(rows),
            "host_runtime_s": host_runtime_s,
            "workflows": [result.workflow],
            "route_plan_audit": {
                "expected_route_count": len(selected_expected_routes),
                "observed_route_count": len(observed_routes),
                "violation_count": len(route_violations),
                "violations": route_violations,
                "retry_violation_count": len(retry_route_violations),
                "retry_violations": retry_route_violations,
            },
            "rank_logical_clocks_s": {
                str(rank): float(snapshot.get("logical_now_s", 0.0))
                for rank, snapshot in sorted(result.cluster.rank_snapshots.items())
            },
            "limitations": ["single workflow only", "offline accuracy gate has not passed"],
        }
    )
    _atomic_write_json(output_path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real agentic workflow on the isolated per-DP logical-clock "
            "SGLSim cluster."
        )
    )
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--route-plan", type=Path, default=None)
    parser.add_argument("--failure-manifest", type=Path, default=None)
    parser.add_argument("--failure-count", type=int, default=0)
    parser.add_argument("--workflow-prefix", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("aegiskv", "no_backup"), default="aegiskv")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dp-size", type=int, default=None)
    parser.add_argument("--pp-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-requests-per-workflow", type=int, default=128)
    parser.add_argument("--arrival-speedup", type=float, default=0.0)
    parser.add_argument("--stage-pause", type=float, default=0.0)
    parser.add_argument("--max-events", type=int, default=1_000_000)
    parser.add_argument(
        "--slack-calibration",
        type=Path,
        default=DEFAULT_SLACK_CALIBRATION,
    )
    parser.add_argument(
        "--slack-source",
        choices=SLACK_SOURCES,
        default="algorithm1",
    )
    return parser


def main() -> int:
    artifact = run(build_parser().parse_args())
    workflow = artifact["workflows"][0]
    print(
        json.dumps(
            {
                "output": artifact["identity"]["artifact_path"],
                "workflow_id": workflow["workflow_id"],
                "requests": workflow["request_count"],
                "logical_wct_s": workflow["wct_s"],
                "host_runtime_s": artifact["host_runtime_s"],
                "route_violations": artifact["route_plan_audit"]["violation_count"],
                "rank_logical_clocks_s": artifact["rank_logical_clocks_s"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
