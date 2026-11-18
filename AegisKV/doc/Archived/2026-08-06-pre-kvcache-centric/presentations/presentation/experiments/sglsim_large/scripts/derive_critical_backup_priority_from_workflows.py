#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT.parent.parent
    / "figures"
    / "data"
    / "measured"
    / "fig_critical_backup_priority.csv"
)

for path in (REPO_ROOT, REPO_ROOT / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from sglsim.srt.simulation.recovery_model import aegis_restore_overhead_s  # noqa: E402


POLICIES = (
    "critical-first",
    "size-first",
    "random",
    "non-critical-first",
)
BUDGETS = tuple(round(x / 10.0, 1) for x in range(11))


@dataclass(frozen=True)
class Candidate:
    workflow_id: str
    row_id: str
    agent_id: str
    critical: bool
    decode_dominant: bool
    stage_width: int
    token_count: int
    no_backup_penalty_s: float
    gpu_replica_penalty_s: float

    @property
    def critical_path_weight(self) -> float:
        if self.critical:
            weight = 1.0
            if self.decode_dominant:
                weight += 0.25
            if self.stage_width <= 1:
                weight += 0.10
            return weight
        return 0.18 / max(self.stage_width, 1)

    @property
    def unprotected_cost_s(self) -> float:
        return self.no_backup_penalty_s * self.critical_path_weight

    @property
    def protected_cost_s(self) -> float:
        return self.gpu_replica_penalty_s * self.critical_path_weight

    @property
    def saved_cost_s(self) -> float:
        return max(self.unprotected_cost_s - self.protected_cost_s, 0.0)

    @property
    def value_density(self) -> float:
        return self.saved_cost_s / max(self.token_count, 1)


@dataclass(frozen=True)
class WorkflowTrace:
    workflow_id: str
    baseline_wct_s: float
    candidates: tuple[Candidate, ...]


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    return int(round(_float(value, float(default))))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "simulator_config.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _original_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(run_dir / "request_details.csv"):
        if str(row.get("status") or "").strip() != "completed":
            continue
        if _bool(row.get("aegis_failover_replay")):
            continue
        if _bool(row.get("aegis_cross_rank_retry")):
            continue
        if not str(row.get("workflow_id") or "").strip():
            continue
        rows.append(row)
    return rows


def _row_latency_s(row: dict[str, str]) -> float:
    return max(
        _float(row.get("client_wall_e2e_latency_s")),
        _float(row.get("e2e_latency_s")),
        _float(row.get("total_latency_s")),
    )


def _token_count(row: dict[str, str]) -> int:
    return max(
        1,
        _int(row.get("aegis_candidate_pre_failure_total_tokens")),
        _int(row.get("aegis_safe_frontier_tokens")),
        _int(row.get("input_length")) + _int(row.get("generated_tokens")),
        _int(row.get("input_length")) + _int(row.get("target_output_tokens")),
    )


def _candidate_from_row(row: dict[str, str]) -> Candidate:
    tokens = _token_count(row)
    observed_no_backup = _float(row.get("aegis_candidate_failover_overhead_s"))
    no_backup = max(
        observed_no_backup,
        aegis_restore_overhead_s(
            action="recompute",
            token_count=tokens,
            restored_tokens=0,
        ),
    )
    gpu_replica = aegis_restore_overhead_s(
        action="gpu_replica",
        token_count=tokens,
        restored_tokens=tokens,
    )
    row_id = str(row.get("rid") or row.get("agent_id") or "")
    return Candidate(
        workflow_id=str(row.get("workflow_id") or ""),
        row_id=row_id,
        agent_id=str(row.get("agent_id") or row_id),
        critical=_bool(row.get("critical_agent")),
        decode_dominant=_bool(row.get("decode_dominant")),
        stage_width=max(_int(row.get("stage_width"), 1), 1),
        token_count=tokens,
        no_backup_penalty_s=no_backup,
        gpu_replica_penalty_s=gpu_replica,
    )


def workflow_traces(run_dir: Path) -> list[WorkflowTrace]:
    rows = _original_rows(run_dir)
    by_workflow: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_workflow.setdefault(str(row["workflow_id"]), []).append(row)

    traces: list[WorkflowTrace] = []
    for workflow_id, workflow_rows in sorted(by_workflow.items()):
        start = min(_float(row.get("created_time_s")) for row in workflow_rows)
        finish = max(_float(row.get("finish_time_s")) for row in workflow_rows)
        baseline_wct = finish - start
        if baseline_wct <= 0:
            baseline_wct = sum(_row_latency_s(row) for row in workflow_rows)
        candidates = tuple(_candidate_from_row(row) for row in workflow_rows)
        if baseline_wct > 0 and candidates:
            traces.append(
                WorkflowTrace(
                    workflow_id=workflow_id,
                    baseline_wct_s=baseline_wct,
                    candidates=candidates,
                )
            )
    return traces


def _random_key(candidate: Candidate) -> str:
    key = f"{candidate.workflow_id}:{candidate.row_id}:{candidate.agent_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def _ordered_candidates(candidates: tuple[Candidate, ...], policy: str) -> list[Candidate]:
    if policy == "critical-first":
        return sorted(
            candidates,
            key=lambda item: (
                not item.critical,
                -item.value_density,
                -item.saved_cost_s,
                -item.token_count,
                item.row_id,
            ),
        )
    if policy == "size-first":
        return sorted(
            candidates,
            key=lambda item: (-item.token_count, not item.critical, item.row_id),
        )
    if policy == "random":
        return sorted(candidates, key=_random_key)
    if policy == "non-critical-first":
        return sorted(
            candidates,
            key=lambda item: (
                item.critical,
                -item.token_count,
                -item.value_density,
                item.row_id,
            ),
        )
    raise ValueError(f"unknown placement policy: {policy}")


def _selected_for_budget(
    candidates: tuple[Candidate, ...], *, policy: str, budget_fraction: float
) -> set[str]:
    capacity = int(round(sum(item.token_count for item in candidates) * budget_fraction))
    selected: set[str] = set()
    used = 0
    for item in _ordered_candidates(candidates, policy):
        if used + item.token_count > capacity:
            continue
        selected.add(item.row_id)
        used += item.token_count
    return selected


def _normalized_wct(
    trace: WorkflowTrace, *, policy: str, budget_fraction: float
) -> float:
    selected = _selected_for_budget(
        trace.candidates,
        policy=policy,
        budget_fraction=budget_fraction,
    )
    no_backup_expected = mean(item.unprotected_cost_s for item in trace.candidates)
    policy_expected = mean(
        item.protected_cost_s if item.row_id in selected else item.unprotected_cost_s
        for item in trace.candidates
    )
    # The online client enforces per-workflow stage barriers. For this
    # motivation figure we isolate the failure-induced WCT penalty so placement
    # policies are compared at the same observed request timing and token mix.
    denominator = no_backup_expected
    if denominator <= 0:
        return 1.0
    return policy_expected / denominator


def derive_rows(
    *,
    run_dirs: list[Path],
    run_id: str,
    evidence_tier: str,
    min_workflows: int,
) -> list[dict[str, object]]:
    traces: list[WorkflowTrace] = []
    config_payloads = [_load_config(run_dir) for run_dir in run_dirs]
    for run_dir in run_dirs:
        traces.extend(workflow_traces(run_dir))
    if len(traces) < min_workflows:
        raise ValueError(f"only {len(traces)} workflow traces found; need {min_workflows}")

    first_workload = (
        config_payloads[0].get("workload")
        if config_payloads and isinstance(config_payloads[0].get("workload"), dict)
        else {}
    )
    workflow_shape = first_workload.get("gsp_workflow_shape", "")
    if isinstance(workflow_shape, list):
        workflow_shape = "-".join(str(item) for item in workflow_shape)

    base_meta = {
        "evidence_tier": evidence_tier,
        "source": "sglsim_agentic_online_trace_placement_counterfactual",
        "run_id": run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": f"staged-{workflow_shape or 'unknown'}",
        "dp_size": _int(first_workload.get("dp_size"), 4),
        "tp_size": 1,
        "failure_rank": "",
        "repeat": f"mean-of-{len(traces)}-workflow-traces",
        "note": (
            "Derived from observed online SGLSim 1-3-1-3-1 workflow traces. "
            "The script keeps the same request timings and token counts, then "
            "counterfactually changes which request KV segments enter the "
            "GPUReplica budget under different placement orders. Values are "
            "normalized failure-induced WCT penalty relative to no backup; "
            "the source traces use per-workflow stage barriers and concurrent "
            "fan-out within each stage."
        ),
        "runtime_workflow_traces": len(traces),
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(str(path) for path in run_dirs),
    }

    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        for budget in BUDGETS:
            values = [
                _normalized_wct(trace, policy=policy, budget_fraction=budget)
                for trace in traces
            ]
            rows.append(
                {
                    "backup_budget_fraction": budget,
                    "placement_policy": policy,
                    "normalized_wct_after_failure": mean(values),
                    **base_meta,
                    "normalized_wct_after_failure_std": metric_std(values),
                    "normalized_wct_after_failure_ci95": metric_ci95(values),
                    "runtime_min_normalized_wct_after_failure": min(values),
                    "runtime_max_normalized_wct_after_failure": max(values),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 1b Critical-first GPUReplica placement rows from "
            "observed online SGLSim workflow request_details.csv files."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-workflows", type=int, default=4)
    parser.add_argument("--evidence-tier", default="sglsim-trace-derived")
    args = parser.parse_args()

    rows = derive_rows(
        run_dirs=args.run_dir,
        run_id=args.run_id,
        evidence_tier=args.evidence_tier,
        min_workflows=args.min_workflows,
    )
    write_csv(args.output_csv, rows)
    best_at_half = {
        row["placement_policy"]: row["normalized_wct_after_failure"]
        for row in rows
        if float(row["backup_budget_fraction"]) == 0.5
    }
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "workflow_traces": rows[0]["runtime_workflow_traces"],
                "normalized_wct_at_budget_0.5": best_at_half,
                "evidence_tier": args.evidence_tier,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
