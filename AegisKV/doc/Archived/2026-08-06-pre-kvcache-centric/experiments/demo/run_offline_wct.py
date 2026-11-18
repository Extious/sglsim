"""Run the matched Real-CC WCT workload on the logical-time SGLSim cluster."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.calibration_metrics import (
    build_offline_cluster_artifact,
    metric_summary,
)
from sglsim.srt.simulation.aegis_exact_contract import (
    load_certified_exact_window_plan,
)
from sglsim.srt.simulation.aegis_exact_optimizer import (
    ExactObjectiveWeights,
    ExactTopology,
)
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster

from .causal_exact_rolling import CausalExactRollingPlanner
from .offline_policy import OfflinePolicyReplay
from .backup_route_plan import load_backup_route_plan
from .completion_order_plan import load_completion_order_plan
from .evidence_journal import (
    bind_fresh_event_journal,
    event_journal_path_for_artifact,
)
from .wct_cdf_client import (
    WORKFLOW_LAUNCH_STAGGER_S,
    has_fan_out_and_main_fan_in,
    load_execution_route_plan,
    load_failure_manifest,
    load_workflows,
)
from .workflow_client import DEFAULT_SLACK_CALIBRATION, DEFAULT_TRACE, SLACK_SOURCES


DEFAULT_SERVER_CONFIG = (
    Path(__file__).resolve().parent
    / "config"
    / "server_qwen3_32b_dp4_pp4_32k.json"
)
DEFAULT_FAILURE_MANIFEST = (
    Path(__file__).resolve().parent
    / "config"
    / "failure_manifest_qwen3_32b_dp4_pp4_32k_matched_v13.json"
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _select_workflows(args: argparse.Namespace) -> list[tuple[str, list[dict[str, Any]]]]:
    selected = [
        (workflow_id, rows)
        for workflow_id, rows in load_workflows(args.trace.resolve()).items()
        if has_fan_out_and_main_fan_in(rows)
        and (
            int(args.max_requests_per_workflow) <= 0
            or len(rows) <= int(args.max_requests_per_workflow)
        )
    ]
    if not selected:
        raise ValueError("no eligible fan-out/fan-in workflows")
    return selected[: int(args.max_workflows) or None]


def _rank_clock_activity(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in telemetry:
        by_rank[int(row.get("dp_rank", -1))].append(row)
    return {
        str(rank): {
            "step_count": len(rows),
            "first_started_at_s": min(
                (float(row.get("started_at", 0.0)) for row in rows),
                default=0.0,
            ),
            "last_completed_at_s": max(
                (float(row.get("completed_at", 0.0)) for row in rows),
                default=0.0,
            ),
            "active_pipeline_s": sum(
                float(row.get("pipeline_base_s", 0.0)) for row in rows
            ),
            "batch_modes": dict(
                sorted(Counter(str(row.get("batch_mode", "idle")) for row in rows).items())
            ),
        }
        for rank, rows in sorted(by_rank.items())
        if rank >= 0
    }


def _compact_artifact(artifact: dict[str, Any]) -> None:
    """Retain sweep metrics while dropping per-step calibration evidence."""

    metrics = dict(artifact.get("calibration_metrics") or {})
    queue_depth = dict(metrics.pop("queue_depth_by_dp", {}) or {})
    metrics["queue_depth_summary_by_dp"] = {
        str(rank): {
            "samples": len(rows),
            "max_waiting": max(
                (int(row.get("waiting", 0)) for row in rows), default=0
            ),
            "max_running": max(
                (int(row.get("running", 0)) for row in rows), default=0
            ),
            "mean_waiting": (
                sum(int(row.get("waiting", 0)) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_running": (
                sum(int(row.get("running", 0)) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        }
        for rank, rows in sorted(queue_depth.items())
    }
    status = dict(artifact.get("aegis_status") or {})
    artifact["calibration_metrics"] = metrics
    artifact["aegis_status_summary"] = {
        key: status[key]
        for key in (
            "rank_states",
            "node_host_states",
            "persistent_pool_state",
        )
        if key in status
    }
    artifact.pop("request_stats", None)
    artifact.pop("scheduler_telemetry", None)
    artifact.pop("aegis_status", None)
    artifact["artifact_detail"] = "compact"


def _tuple_rows(values: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(row) for row in values)


def _causal_exact_inputs_from_artifact(
    raw: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    finite = raw.get("finite_instance")
    if not isinstance(finite, dict):
        raise ValueError(
            "causal exact rolling requires a full solver artifact with "
            "finite_instance, not a bare wire plan"
        )
    topology_raw = finite.get("topology")
    weights_raw = finite.get("weights")
    if not isinstance(topology_raw, dict) or not isinstance(weights_raw, dict):
        raise ValueError("exact solver artifact lacks topology/objective weights")
    topology = ExactTopology(
        pp_size=int(topology_raw["pp_size"]),
        dp_to_node=tuple(int(value) for value in topology_raw["dp_to_node"]),
        hbm_capacity_tokens=_tuple_rows(
            topology_raw["hbm_capacity_tokens"]
        ),
        host_policy_max_tokens=tuple(
            int(value) for value in topology_raw["host_policy_max_tokens"]
        ),
        node_host_capacity_tokens=_tuple_rows(
            topology_raw["node_host_capacity_tokens"]
        ),
        link_bandwidth_bytes=_tuple_rows(
            topology_raw["link_bandwidth_bytes"]
        ),
        hbm_failure_domains=_tuple_rows(
            topology_raw["hbm_failure_domains"]
        ),
        host_failure_domains=_tuple_rows(
            topology_raw["host_failure_domains"]
        ),
    )
    weights = ExactObjectiveWeights(**dict(weights_raw))
    inputs = dict(raw.get("inputs") or {})
    provenance = dict(raw.get("topology_provenance") or {})
    aegis = dict(config.get("aegis_l1_l2") or {})
    return {
        "topology": topology,
        "weights": weights,
        "failure_hazard_probability": float(
            inputs.get("failure_hazard_probability_prior", 0.01)
        ),
        "failure_scope": str(inputs.get("failure_scope") or "node"),
        "chunk_tokens": int(inputs.get("chunk_tokens") or aegis.get("chunk_tokens", 256)),
        "recompute_setup_s": float(aegis.get("recompute_setup_ms", 0.0)) / 1000.0,
        "recompute_s_per_token": float(
            aegis.get("recompute_ms_per_token", 0.0)
        )
        / 1000.0,
        "protected_restore_setup_s": float(
            aegis.get("gpu_direct_reuse_setup_ms", 0.0)
        )
        / 1000.0,
        "protected_restore_s_per_token": float(
            aegis.get("l1_restore_ms_per_token", 0.0)
        )
        / 1000.0,
        "link_window_duration_s": float(
            provenance.get(
                "window_duration_s",
                float(aegis.get("window_target_ms", 5000.0)) / 1000.0,
            )
        ),
        "kv_bytes_per_token_per_stage": int(
            provenance.get(
                "kv_bytes_per_token_per_stage",
                weights.hbm_copy_bytes_per_token,
            )
        ),
        "time_limit_s": float(finite.get("time_limit_s", 300.0)),
        "deterministic_seed": int(finite.get("deterministic_seed", 0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.resolve()
    event_journal_path = event_journal_path_for_artifact(output_path)
    slack_source = str(getattr(args, "slack_source", "algorithm1"))
    artifact_detail = str(getattr(args, "artifact_detail", "full"))
    base_server_config_path = args.server_config.resolve()
    server_config_path = base_server_config_path
    trace_path = args.trace.resolve()
    config = json.loads(server_config_path.read_text(encoding="utf-8"))
    aegis_overrides: dict[str, Any] = {}
    if args.aegis_overrides is not None:
        override_path = args.aegis_overrides.resolve()
        raw_overrides = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(raw_overrides, dict):
            raise ValueError("AegisKV override artifact must contain a JSON object")
        aegis_overrides = dict(raw_overrides)
        merged = dict(config.get("aegis_l1_l2", {}))
        merged.update(aegis_overrides)
        config["aegis_l1_l2"] = merged
        config["offline_search_provenance"] = {
            "base_server_config": str(base_server_config_path),
            "aegis_overrides": str(override_path),
        }
        server_config_path = output_path.with_suffix(".server-config.json")
        _atomic_write_json(server_config_path, config)
    scheduler = dict(config.get("scheduler", {}))
    model = str(args.model or config.get("model_path") or "mock-model")
    dp_size = int(args.dp_size or scheduler.get("dp_size", 1))
    pp_size = int(args.pp_size or scheduler.get("pp_size", 1))
    context_length = int(args.context_length or config.get("context_length", 131072))
    selected = _select_workflows(args)
    selected_rids = {
        str(row["rid"])
        for _workflow_id, rows in selected
        for row in rows
    }

    failure_manifest_path = args.failure_manifest.resolve()
    failure_manifest = load_failure_manifest(
        failure_manifest_path,
        dp_size=dp_size,
        pp_size=pp_size,
        failure_count=int(args.failure_count),
    )
    route_plan_path = (
        args.route_plan.resolve() if args.route_plan is not None else None
    )
    route_payload: dict[str, Any] | None = None
    route_plan: dict[str, int] = {}
    if route_plan_path is not None:
        route_payload = load_execution_route_plan(
            route_plan_path, dp_size=dp_size
        )
        route_plan = {
            str(rid): int(spec["execution_dp_rank"])
            for rid, spec in route_payload["routes"].items()
        }
        missing_routes = sorted(selected_rids - set(route_plan))
        if missing_routes:
            raise ValueError(
                "execution route plan misses "
                f"{len(missing_routes)} selected requests; "
                f"first={missing_routes[0]}"
            )
    exact_window_plan = None
    exact_window_plan_path: Path | None = None
    exact_solver_artifact: dict[str, Any] | None = None
    causal_exact_rolling = bool(getattr(args, "causal_exact_rolling", False))
    causal_exact_journal_path = output_path.with_suffix(".exact-solves.jsonl")
    if args.exact_window_plan is not None:
        if args.policy != "aegiskv":
            raise ValueError(
                "certified exact deployment is only valid for policy=aegiskv"
            )
        if route_plan_path is None:
            raise ValueError(
                "certified exact deployment requires a paired execution route plan"
            )
        exact_window_plan_path = args.exact_window_plan.resolve()
        exact_window_plan = load_certified_exact_window_plan(
            exact_window_plan_path
        )
        raw_exact = json.loads(
            exact_window_plan_path.read_text(encoding="utf-8")
        )
        if not isinstance(raw_exact, dict):
            raise ValueError("exact window artifact must contain a JSON object")
        exact_solver_artifact = raw_exact
        exact_routes = {
            route.request_id: route for route in exact_window_plan.routes
        }
        if not causal_exact_rolling:
            missing_exact_routes = sorted(selected_rids - set(exact_routes))
            if missing_exact_routes:
                raise ValueError(
                    "certified exact window misses "
                    f"{len(missing_exact_routes)} selected requests; "
                    f"first={missing_exact_routes[0]}"
                )
            route_conflicts = [
                rid
                for rid in sorted(selected_rids)
                if exact_routes[rid].execution_dp_rank != route_plan[rid]
            ]
            if route_conflicts:
                raise ValueError(
                    "certified exact execution route conflicts with the matched "
                    f"route plan; first={route_conflicts[0]}"
                )
    if causal_exact_rolling and exact_window_plan is None:
        raise ValueError(
            "--causal-exact-rolling requires --exact-window-plan to publish "
            "the initial certified Algorithm-3 budget"
        )
    backup_route_plan: dict[str, int] = {}
    backup_route_plan_path: Path | None = None
    if args.backup_route_plan is not None:
        if args.policy != "aegiskv":
            raise ValueError("backup route plan is calibration-only for aegiskv")
        backup_route_plan_path = args.backup_route_plan.resolve()
        backup_route_plan = load_backup_route_plan(
            backup_route_plan_path,
            dp_size=dp_size,
        )
        missing_backup_routes = sorted(selected_rids - set(backup_route_plan))
        if missing_backup_routes:
            raise ValueError(
                "backup route plan misses "
                f"{len(missing_backup_routes)} selected requests; "
                f"first={missing_backup_routes[0]}"
            )

    completion_order_plan: dict[str, int] = {}
    completion_order_plan_path: Path | None = None
    if args.completion_order_plan is not None:
        completion_order_plan_path = args.completion_order_plan.resolve()
        completion_order_plan = load_completion_order_plan(
            completion_order_plan_path,
            selected_request_ids=selected_rids,
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
        backup_route_plan=backup_route_plan,
        completion_order_plan=completion_order_plan,
        failure_manifest=failure_manifest,
        policy_start_monotonic=0.0,
        stage_pause=float(args.stage_pause),
        driver_startup_overhead_s=float(args.driver_startup_overhead_s),
        driver_request_overhead_s=float(args.driver_request_overhead_s),
        quiet=True,
    )
    logical_pp = bool(getattr(args, "logical_pp", False)) or bool(
        config.get("logical_pp", False)
    )
    server_args = ServerArgs(
        model_path=model,
        dp_size=dp_size,
        pp_size=pp_size,
        load_balance_method=str(scheduler.get("load_balance_method", "cache_aware")),
        streaming_arrival=True,
        sim_config_path=str(server_config_path),
        log_level="warning",
        logical_pp=logical_pp,
    )

    started = time.monotonic()
    with bind_fresh_event_journal(event_journal_path):
        with OfflineSchedulerCluster(server_args) as cluster:
            exact_installation = None
            exact_budget_barrier = None
            causal_planner = None
            if exact_window_plan is not None:
                cluster.synchronize_dp_budget_publication()
                exact_installation = (
                    cluster.dp_controller.install_certified_exact_window(
                        exact_window_plan
                    )
                )
                exact_budget_barrier = (
                    cluster.synchronize_dp_budget_publication()
                )
                if causal_exact_rolling:
                    assert exact_solver_artifact is not None
                    causal_inputs = _causal_exact_inputs_from_artifact(
                        exact_solver_artifact,
                        config=config,
                    )
                    causal_planner = CausalExactRollingPlanner(
                        **causal_inputs,
                        first_window_id=exact_window_plan.window_id + 1,
                        first_route_version=exact_window_plan.route_version + 1,
                        journal_path=causal_exact_journal_path,
                    )
                    cluster.set_causal_exact_plan_provider(causal_planner)
            result = OfflinePolicyReplay(
                cluster=cluster,
                args=workflow_args,
                selected_workflows=selected,
                failure_manifest=failure_manifest,
                workflow_concurrency=int(args.workflow_concurrency),
                launch_stagger_s=float(args.launch_stagger_s),
            ).run(max_events=int(args.max_events))
            host_runtime_s = time.monotonic() - started

    artifact = build_offline_cluster_artifact(
        result.cluster,
        identity={
            "model": model,
            "dp_size": dp_size,
            "pp_size": pp_size,
            "logical_pp": logical_pp,
            "process_topology": (
                "logical_pp_smoke" if logical_pp else "physical_dp_pp"
            ),
            "context_length": context_length,
            "input_scale": float(args.input_scale),
            "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "arrival_speedup": float(args.arrival_speedup),
            "stage_pause_s": float(args.stage_pause),
            "server_config": str(server_config_path),
            "base_server_config": str(base_server_config_path),
            "aegis_overrides": aegis_overrides,
            "trace": str(trace_path),
            "route_plan": (
                str(route_plan_path) if route_plan_path is not None else None
            ),
            "backup_route_plan": (
                str(backup_route_plan_path) if backup_route_plan_path else None
            ),
            "completion_order_plan": (
                str(completion_order_plan_path)
                if completion_order_plan_path
                else None
            ),
            "exact_window_plan": (
                str(exact_window_plan_path)
                if exact_window_plan_path is not None
                else None
            ),
            "causal_exact_rolling": causal_exact_rolling,
            "causal_exact_solver_journal": (
                str(causal_exact_journal_path)
                if causal_exact_rolling
                else None
            ),
            "failure_manifest": str(failure_manifest_path),
            "policy": args.policy,
            "slack_source": slack_source,
            "trace_oracle_upper_bound": slack_source == "trace_oracle",
            "artifact_path": str(output_path),
            "event_journal": str(event_journal_path),
        },
    )
    observed_original_route_rows = {
        str(route["rid"]): dict(route)
        for workflow in result.workflows
        for route in workflow["request_routes"]
    }
    observed_original_routes = {
        rid: int(route["execution_dp_rank"])
        for rid, route in observed_original_route_rows.items()
    }
    normal_route_violations = {
        rid: {"expected": route_plan[rid], "observed": observed_original_routes.get(rid)}
        for rid in sorted(selected_rids & set(route_plan))
        if observed_original_routes.get(rid) != route_plan[rid]
        and observed_original_route_rows.get(rid, {}).get(
            "route_plan_applied"
        ) is not False
    }
    availability_fallbacks = {
        rid: {
            "planned": route_plan[rid],
            "observed": observed_original_routes.get(rid),
            "route_plan_applied": False,
        }
        for rid in sorted(selected_rids & set(route_plan))
        if observed_original_routes.get(rid) != route_plan[rid]
        and observed_original_route_rows.get(rid, {}).get(
            "route_plan_applied"
        ) is False
    }
    backup_route_violations = {
        str(route["rid"]): {
            "expected": backup_route_plan[str(route["rid"])],
            "observed": int(route["backup_dp_rank"]),
        }
        for workflow in result.workflows
        for route in workflow["request_routes"]
        if str(route["rid"]) in backup_route_plan
        and int(route["backup_dp_rank"])
        != backup_route_plan[str(route["rid"])]
    }
    victims = {
        str(victim["backup_key"]): dict(victim)
        for event in failure_manifest["events"]
        for victim in event.get("victims", ())
    }
    retry_events = [
        dict(retry)
        for workflow in result.workflows
        for retry in workflow["retry_events"]
    ]
    retry_route_violations = []
    for retry in retry_events:
        expected_raw = retry.get("expected_retry_dp_rank")
        if expected_raw is None:
            continue
        expected = int(expected_raw)
        if int(retry["retry_dp_rank"]) != expected:
            retry_route_violations.append(
                {
                    "rid": retry["rid"],
                    "expected": expected,
                    "observed": int(retry["retry_dp_rank"]),
                    "expected_source": retry.get(
                        "expected_retry_rank_source"
                    ),
                    "retry_route_mode": retry.get("retry_route_mode"),
                }
            )
    wct_values = [float(workflow["wct_s"]) for workflow in result.workflows]
    artifact.update(
        {
            "policy": args.policy,
            "slack_source": slack_source,
            "trace_oracle_upper_bound": slack_source == "trace_oracle",
            "workflow_count": len(result.workflows),
            "workflow_concurrency": int(args.workflow_concurrency),
            "input_scale": float(args.input_scale),
            "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "workflow_launch_policy": "ordered_predecessor_chain_v1",
            "workflow_launch_stagger_s": float(args.launch_stagger_s),
            "request_count": sum(
                int(workflow["request_count"]) for workflow in result.workflows
            ),
            "retry_count": len(retry_events),
            "failure_count_requested": int(args.failure_count),
            "failure_count_injected": len(result.cluster.failure_events),
            "workflow_completion_time_s": metric_summary(wct_values),
            "workflows": result.workflows,
            "launch_events": result.launch_events,
            "route_plan_audit": {
                "enabled": route_plan_path is not None,
                "expected_route_count": (
                    len(selected_rids) if route_plan_path is not None else 0
                ),
                "observed_route_count": len(observed_original_routes),
                "violation_count": len(normal_route_violations),
                "violations": normal_route_violations,
                "availability_fallback_count": len(availability_fallbacks),
                "availability_fallbacks": availability_fallbacks,
                "retry_expected_count": len(victims),
                "retry_observed_count": len(retry_events),
                "retry_violation_count": len(retry_route_violations),
                "retry_violations": retry_route_violations,
                "backup_expected_count": len(backup_route_plan),
                "backup_violation_count": len(backup_route_violations),
                "backup_violations": backup_route_violations,
            },
            "rank_clock_activity": _rank_clock_activity(result.cluster.telemetry),
            "rank_logical_clocks_s": {
                str(rank): float(snapshot.get("logical_now_s", 0.0))
                for rank, snapshot in sorted(result.cluster.rank_snapshots.items())
            },
            "host_runtime_s": host_runtime_s,
            "driver_calibration": {
                "startup_overhead_s": float(args.driver_startup_overhead_s),
                "request_overhead_s": float(args.driver_request_overhead_s),
                "scope": "external workflow driver only",
            },
            "exact_window_installation": exact_installation,
            "exact_window_budget_barrier": exact_budget_barrier,
            "causal_exact_rolling": {
                "enabled": causal_exact_rolling,
                "future_failure_manifest_used": False,
                "solver_journal": (
                    str(causal_exact_journal_path)
                    if causal_exact_rolling
                    else None
                ),
                "solve_count": (
                    len(causal_planner.history)
                    if causal_planner is not None
                    else 0
                ),
                "solves": (
                    list(causal_planner.history)
                    if causal_planner is not None
                    else []
                ),
                "installations": list(
                    result.cluster.exact_rolling_installations
                ),
            },
            "limitations": ["offline accuracy gate has not passed"],
        }
    )
    if artifact_detail == "compact":
        _compact_artifact(artifact)
    else:
        artifact["artifact_detail"] = "full"
    _atomic_write_json(output_path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the matched AegisKV WCT workload with per-DP logical clocks."
    )
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG)
    parser.add_argument(
        "--aegis-overrides",
        type=Path,
        default=None,
        help="JSON object merged into aegis_l1_l2 for an auditable offline sweep.",
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--route-plan",
        type=Path,
        default=None,
        help=(
            "paired execution route plan; omit only for a no-backup route "
            "calibration run used to construct that plan"
        ),
    )
    parser.add_argument("--backup-route-plan", type=Path, default=None)
    parser.add_argument(
        "--exact-window-plan",
        type=Path,
        default=None,
        help=(
            "Offline-certified finite-window artifact installed before any "
            "request is dispatched. Only solver status OPTIMAL with a zero "
            "validated gap is accepted."
        ),
    )
    parser.add_argument(
        "--causal-exact-rolling",
        action="store_true",
        help=(
            "Before each original request admission, solve a new certified "
            "finite window from causal controller stock/availability while "
            "reusing the currently ACKed exact alpha/beta budget. Requires a "
            "full --exact-window-plan solver artifact."
        ),
    )
    parser.add_argument(
        "--completion-order-plan",
        type=Path,
        default=None,
        help=(
            "Calibration-only online response-completion order. It freezes only "
            "the external driver causal order, not Router or placement decisions."
        ),
    )
    parser.add_argument(
        "--failure-manifest", type=Path, default=DEFAULT_FAILURE_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("aegiskv", "l1_only", "l2_only", "l3_only", "no_backup"),
        required=True,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--dp-size", type=int, default=None)
    parser.add_argument("--pp-size", type=int, default=None)
    parser.add_argument(
        "--logical-pp",
        action="store_true",
        help="Spawn one offline Scheduler per DP (no physical PP worker mesh).",
    )
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--workflow-concurrency", type=int, default=4)
    parser.add_argument("--launch-stagger-s", type=float, default=WORKFLOW_LAUNCH_STAGGER_S)
    parser.add_argument("--max-workflows", type=int, default=5)
    parser.add_argument("--max-requests-per-workflow", type=int, default=128)
    parser.add_argument("--failure-count", type=int, default=3)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--arrival-speedup", type=float, default=0.0)
    parser.add_argument("--stage-pause", type=float, default=0.0)
    parser.add_argument(
        "--driver-startup-overhead-s",
        type=float,
        default=0.0,
        help="Calibration-only one-time external client/tokenizer setup delay.",
    )
    parser.add_argument(
        "--driver-request-overhead-s",
        type=float,
        default=0.0,
        help="Calibration-only external HTTP/client delay per completed request.",
    )
    parser.add_argument("--max-events", type=int, default=10_000_000)
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
    parser.add_argument(
        "--artifact-detail",
        choices=("full", "compact"),
        default="full",
        help="compact drops per-step telemetry and keeps sweep-level summaries",
    )
    return parser


def main() -> int:
    artifact = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "output": artifact["identity"]["artifact_path"],
                "policy": artifact["policy"],
                "workflows": artifact["workflow_count"],
                "requests": artifact["request_count"],
                "retries": artifact["retry_count"],
                "failures": artifact["failure_count_injected"],
                "wct": artifact["workflow_completion_time_s"],
                "host_runtime_s": artifact["host_runtime_s"],
                "route_violations": artifact["route_plan_audit"]["violation_count"],
                "retry_route_violations": artifact["route_plan_audit"][
                    "retry_violation_count"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
