"""Build and certify a finite AegisKV plan from a frozen Real-CC window."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sglsim.srt.simulation.aegis_exact_optimizer import (
    ExactObjectiveWeights,
    ExactTopology,
    build_certified_exact_window_plan,
    solve_exact_placement,
)
from sglsim.srt.simulation.aegis_exact_rolling import (
    ExactWindowRequest,
    build_bootstrap_exact_instance,
)
from sglsim.srt.simulation.manager.config import ConfigManager

from .wct_cdf_client import (
    has_fan_out_and_main_fan_in,
    load_execution_route_plan,
    load_workflows,
)
from .workflow_client import DEFAULT_TRACE_ORACLE, _scaled_tokens


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_CONFIG = (
    Path(__file__).resolve().parent
    / "config"
    / "server_qwen3_32b_dp4_pp4_32k.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _selected_workflows(
    trace: Path,
    *,
    max_workflows: int,
    max_requests_per_workflow: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    selected = [
        (workflow_id, rows)
        for workflow_id, rows in load_workflows(trace).items()
        if has_fan_out_and_main_fan_in(rows)
        and (
            max_requests_per_workflow <= 0
            or len(rows) <= max_requests_per_workflow
        )
    ]
    if not selected:
        raise ValueError("no eligible fan-out/fan-in workflows")
    return selected[: max_workflows or None]


def _runtime_topology(
    config_path: Path,
    *,
    model: str,
    dp_size: int,
    pp_size: int,
    failure_scope: str,
    window_duration_s: float,
    kv_bytes_per_token_per_stage: int,
) -> tuple[ExactTopology, dict[str, Any]]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    aegis = dict(raw.get("aegis_l1_l2", {}))
    topology_raw = dict(aegis.get("topology", {}))
    dp_to_node = tuple(
        int(value)
        for value in topology_raw.get("dp_rank_to_node", range(dp_size))
    )
    if len(dp_to_node) != dp_size:
        raise ValueError("server topology dp_rank_to_node does not match dp_size")

    ConfigManager.load(str(config_path))
    cache = ConfigManager.get_cache_config(model)
    hbm_tokens = int(
        cache.aegis_total_l1_capacity_pages
        if cache.aegis_total_l1_capacity_pages is not None
        else cache.l1_capacity_pages
    ) * int(cache.page_size)
    host_tokens = int(cache.l2_capacity_pages) * int(cache.page_size)
    if hbm_tokens <= 0 or host_tokens <= 0:
        raise ValueError("exact deployment requires positive HBM and Host capacity")

    nodes = tuple(sorted(set(dp_to_node)))
    platform = dict(raw.get("platform", {}))
    inter_node_gb_s = float(platform.get("inter_node_bandwidth_gb", 25.0))
    link_capacity_bytes = int(
        inter_node_gb_s * 1_000_000_000 * window_duration_s
    )
    if link_capacity_bytes <= 0:
        raise ValueError("exact window link capacity must be positive")
    hbm_domains = tuple(
        tuple(
            (
                f"node:{dp_to_node[dp_rank]}"
                if failure_scope == "node"
                else f"dp:{dp_rank}"
            )
            for _ in range(pp_size)
        )
        for dp_rank in range(dp_size)
    )
    topology = ExactTopology(
        pp_size=pp_size,
        dp_to_node=dp_to_node,
        hbm_capacity_tokens=tuple(
            tuple(hbm_tokens for _ in range(pp_size))
            for _ in range(dp_size)
        ),
        host_policy_max_tokens=tuple(host_tokens for _ in range(dp_size)),
        node_host_capacity_tokens=tuple(
            (node_id, host_tokens) for node_id in nodes
        ),
        link_bandwidth_bytes=tuple(
            (source, target, link_capacity_bytes)
            for source in range(dp_size)
            for target in range(dp_size)
            if source != target
        ),
        hbm_failure_domains=hbm_domains,
        host_failure_domains=tuple(
            (node_id, f"node:{node_id}") for node_id in nodes
        ),
    )
    return topology, {
        "hbm_reference_tokens_per_stage": hbm_tokens,
        "host_reference_tokens_per_node": host_tokens,
        "window_duration_s": window_duration_s,
        "inter_node_bandwidth_gb_s": inter_node_gb_s,
        "link_capacity_bytes_per_window": link_capacity_bytes,
        "kv_bytes_per_token_per_stage": kv_bytes_per_token_per_stage,
        "capacity_semantics": {
            "hbm": "one identical DP-owned budget applied to every PP stage",
            "host_policy": "one DP-owned beta budget across its PP stages",
            "host_physical": "node-shared capacity independently arbitrated",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.server_config.resolve()
    trace_path = args.trace.resolve()
    route_path = args.route_plan.resolve()
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    scheduler = dict(raw_config.get("scheduler", {}))
    aegis = dict(raw_config.get("aegis_l1_l2", {}))
    model = str(args.model or raw_config.get("model_path") or "mock-model")
    dp_size = int(args.dp_size or scheduler.get("dp_size", 1))
    pp_size = int(args.pp_size or scheduler.get("pp_size", 1))
    if dp_size <= 1 or pp_size <= 1:
        raise ValueError("exact DP x PP deployment requires dp_size, pp_size > 1")
    context_length = int(
        args.context_length or raw_config.get("context_length", 131072)
    )
    selected = _selected_workflows(
        trace_path,
        max_workflows=int(args.max_workflows),
        max_requests_per_workflow=int(args.max_requests_per_workflow),
    )
    route_payload = load_execution_route_plan(route_path, dp_size=dp_size)
    routes = {
        str(rid): int(value["execution_dp_rank"])
        for rid, value in route_payload["routes"].items()
    }

    requests: list[ExactWindowRequest] = []
    transformation_rows: list[dict[str, Any]] = []
    for workflow_id, rows in selected:
        for row in rows:
            rid = str(row["rid"])
            if rid not in routes:
                raise ValueError(f"execution route plan misses request {rid}")
            input_tokens, output_tokens, input_clipped, output_clipped, scale = (
                _scaled_tokens(
                    row,
                    input_scale=float(args.input_scale),
                    max_input_tokens=args.max_input_tokens,
                    max_output_tokens=args.max_output_tokens,
                    context_length=context_length,
                )
            )
            if args.slack_source == "trace_oracle":
                if not bool(row.get("trace_oracle_uses_future_information", False)):
                    raise ValueError(
                        f"trace-oracle marker missing for request {rid}"
                    )
                slack_s = max(
                    float(row.get("trace_oracle_slack_ms", 0.0)) / 1000.0,
                    0.0,
                )
                critical_weight = (
                    1.0 if bool(row.get("trace_oracle_critical_agent", False)) else 0.0
                )
            else:
                slack_s = 0.0
                critical_weight = 1.0 if bool(row.get("critical_agent", False)) else 0.0
            token_count = input_tokens + output_tokens
            requests.append(
                ExactWindowRequest(
                    request_id=rid,
                    workflow_id=workflow_id,
                    source_dp_rank=routes[rid],
                    token_count=token_count,
                    slack_s=slack_s,
                    critical_weight=critical_weight,
                    admission_time_s=0.0,
                    admission_state_id="all_available",
                )
            )
            transformation_rows.append(
                {
                    "request_id": rid,
                    "workflow_id": workflow_id,
                    "execution_dp_rank": routes[rid],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "token_count": token_count,
                    "input_clipped": input_clipped,
                    "output_clipped": output_clipped,
                    "context_scale": scale,
                    "slack_s": slack_s,
                    "critical_weight": critical_weight,
                    "admission_time_s": 0.0,
                    "admission_state_id": "all_available",
                }
            )

    chunk_tokens = int(args.chunk_tokens or aegis.get("chunk_tokens", 256))
    window_duration_s = float(
        args.window_duration_s
        or float(aegis.get("window_target_ms", 5000.0)) / 1000.0
    )
    kv_bytes = int(args.kv_bytes_per_token_per_stage)
    topology, topology_provenance = _runtime_topology(
        config_path,
        model=model,
        dp_size=dp_size,
        pp_size=pp_size,
        failure_scope=args.failure_scope,
        window_duration_s=window_duration_s,
        kv_bytes_per_token_per_stage=kv_bytes,
    )
    hbm_reference = min(topology.hbm_capacity_tokens[0])
    host_reference = topology.host_policy_max_tokens[0]
    initial_alpha = max(min(float(aegis.get("initial_alpha", 0.0)), 1.0), 0.0)
    initial_beta = max(min(float(aegis.get("initial_beta", 0.0)), 1.0), 0.0)
    weights = ExactObjectiveWeights(
        expected_workflow_loss=float(args.expected_loss_weight),
        cvar=float(args.cvar_weight),
        cvar_tau=float(args.cvar_tau),
        hbm_normal_cost_per_token=(
            float(aegis.get("l1_write_ms_per_token", 0.0)) / 1000.0
        ),
        host_normal_cost_per_token=(
            float(aegis.get("d2h_ms_per_token", 0.0))
            + float(aegis.get("h2h_ms_per_token", 0.0))
        )
        / 1000.0,
        copy_bandwidth_cost_per_byte=0.0,
        hbm_budget_cost_per_token=float(args.hbm_budget_cost_per_token),
        host_budget_cost_per_token=float(args.host_budget_cost_per_token),
        hbm_switch_cost_per_token=float(args.hbm_switch_cost_per_token),
        host_switch_cost_per_token=float(args.host_switch_cost_per_token),
        hbm_copy_bytes_per_token=kv_bytes,
        host_copy_bytes_per_token=kv_bytes,
    )
    instance = build_bootstrap_exact_instance(
        instance_id=str(args.instance_id),
        requests=tuple(requests),
        topology=topology,
        weights=weights,
        failure_hazard_probability=float(args.failure_hazard_probability),
        failure_scope=args.failure_scope,
        chunk_tokens=chunk_tokens,
        recompute_setup_s=float(aegis.get("recompute_setup_ms", 0.0)) / 1000.0,
        recompute_s_per_token=(
            float(aegis.get("recompute_ms_per_token", 0.0)) / 1000.0
        ),
        protected_restore_setup_s=(
            float(aegis.get("gpu_direct_reuse_setup_ms", 0.0)) / 1000.0
        ),
        protected_restore_s_per_token=(
            float(aegis.get("l1_restore_ms_per_token", 0.0)) / 1000.0
        ),
        previous_hbm_budget_tokens=tuple(
            int(initial_alpha * hbm_reference) for _ in range(dp_size)
        ),
        previous_host_budget_tokens=tuple(
            int(initial_beta * host_reference) for _ in range(dp_size)
        ),
        time_limit_s=float(args.time_limit_s),
        deterministic_seed=int(args.seed),
    )
    result = solve_exact_placement(instance).require_global_optimum()
    replay = solve_exact_placement(instance).require_global_optimum()
    if (
        result.certificate.decision_sha256
        != replay.certificate.decision_sha256
        or result.certificate.objective != replay.certificate.objective
    ):
        raise RuntimeError(
            "deterministic exact replay changed the decision digest/objective"
        )
    plan = build_certified_exact_window_plan(
        instance,
        result,
        window_id=int(args.window_id),
        route_version=int(args.route_version),
        valid_from_logical_time=0.0,
    )
    artifact = {
        "schema_version": 1,
        "artifact_type": "aegiskv_certified_exact_bootstrap_window",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": _git_revision(),
        "claim_boundary": {
            "finite_problem": "zero-residency bootstrap rolling window",
            "global_optimum_scope": "this serialized finite MILP only",
            "trace_oracle_uses_future_information": (
                args.slack_source == "trace_oracle"
            ),
            "admission_availability_information": "causal_observed",
            "realized_failure_manifest_used_for_routing": False,
            "future_manifest_aware_plan_is_diagnostic_only": True,
            "formal_evidence_ready": False,
            "reason": (
                "free-running online/offline accuracy gate and richer "
                "tier-specific DAG WCT calibration remain required"
            ),
        },
        "inputs": {
            "server_config": str(config_path),
            "server_config_sha256": _sha256_file(config_path),
            "trace": str(trace_path),
            "trace_sha256": _sha256_file(trace_path),
            "execution_route_plan": str(route_path),
            "execution_route_plan_sha256": _sha256_file(route_path),
            "model": model,
            "dp_size": dp_size,
            "pp_size": pp_size,
            "context_length": context_length,
            "workflow_count": len(selected),
            "request_count": len(requests),
            "workflow_ids": [workflow_id for workflow_id, _rows in selected],
            "input_scale": float(args.input_scale),
            "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "slack_source": args.slack_source,
            "failure_hazard_probability_prior": float(
                args.failure_hazard_probability
            ),
            "actual_injected_failure_rate": None,
            "failure_scope": args.failure_scope,
            "chunk_tokens": chunk_tokens,
            "deterministic_seed": int(args.seed),
        },
        "topology_provenance": topology_provenance,
        "request_transformations": transformation_rows,
        "finite_instance": asdict(instance),
        "optimization_result": asdict(result),
        "deterministic_replay": {
            "performed": True,
            "decision_sha256": replay.certificate.decision_sha256,
            "objective": replay.certificate.objective,
            "solver_status": replay.certificate.status,
            "mip_gap": replay.certificate.mip_gap,
            "validation_errors": list(replay.certificate.validation_errors),
            "solve_time_s": replay.certificate.solve_time_s,
        },
        "certified_exact_window_plan": plan.to_dict(),
    }
    _atomic_write_json(args.output.resolve(), artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve and certify one frozen Real-CC AegisKV window."
    )
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_ORACLE)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dp-size", type=int, default=None)
    parser.add_argument("--pp-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--max-workflows", type=int, default=1)
    parser.add_argument("--max-requests-per-workflow", type=int, default=128)
    parser.add_argument("--input-scale", type=float, default=0.008)
    parser.add_argument("--max-input-tokens", type=int, default=1008)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument(
        "--slack-source",
        choices=("trace_oracle", "critical_label_no_slack"),
        default="trace_oracle",
    )
    parser.add_argument("--failure-hazard-probability", type=float, default=0.01)
    parser.add_argument("--failure-scope", choices=("node", "dp"), default="node")
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--window-duration-s", type=float, default=None)
    parser.add_argument("--kv-bytes-per-token-per-stage", type=int, default=65536)
    parser.add_argument("--expected-loss-weight", type=float, default=1.0)
    parser.add_argument("--cvar-weight", type=float, default=1.0)
    parser.add_argument("--cvar-tau", type=float, default=0.95)
    parser.add_argument("--hbm-budget-cost-per-token", type=float, default=1e-12)
    parser.add_argument("--host-budget-cost-per-token", type=float, default=1e-12)
    parser.add_argument("--hbm-switch-cost-per-token", type=float, default=0.0)
    parser.add_argument("--host-switch-cost-per-token", type=float, default=0.0)
    parser.add_argument("--time-limit-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-id", type=int, default=1)
    parser.add_argument("--route-version", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = run(args)
    plan = artifact["certified_exact_window_plan"]
    result = artifact["optimization_result"]["certificate"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "instance_id": plan["instance_id"],
                "requests": len(plan["routes"]),
                "placements": len(plan["placements"]),
                "status": result["status"],
                "objective": result["objective"],
                "dual_bound": result["dual_bound"],
                "mip_gap": result["mip_gap"],
                "decision_sha256": result["decision_sha256"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
