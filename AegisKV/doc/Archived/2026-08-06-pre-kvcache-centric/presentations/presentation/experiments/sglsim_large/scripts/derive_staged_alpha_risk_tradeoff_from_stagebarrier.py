#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    / "fig_staged_alpha_risk_tradeoff.csv"
)

for path in (REPO_ROOT, REPO_ROOT / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from sglsim.srt.simulation.recovery_model import (  # noqa: E402
    DEFAULT_GPU_MS_PER_TOKEN,
    DEFAULT_GPU_SETUP_MS,
    aegis_restore_overhead_s,
)


ALPHAS = tuple(round(x / 10.0, 1) for x in range(11))
FAILURE_RATES = (0.001, 0.003, 0.01, 0.03, 0.10)
VISIBLE_BACKUP_OVERLAP_FACTOR = 0.10


@dataclass(frozen=True)
class Candidate:
    workflow_id: str
    row_id: str
    critical: bool
    decode_dominant: bool
    stage_width: int
    token_count: int

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
    def recompute_cost_s(self) -> float:
        return (
            aegis_restore_overhead_s(
                action="recompute",
                token_count=self.token_count,
                restored_tokens=0,
            )
            * self.critical_path_weight
        )

    @property
    def full_gpu_cost_s(self) -> float:
        return (
            aegis_restore_overhead_s(
                action="gpu_replica",
                token_count=self.token_count,
                restored_tokens=self.token_count,
            )
            * self.critical_path_weight
        )

    @property
    def value_density(self) -> float:
        saved = max(self.recompute_cost_s - self.full_gpu_cost_s, 0.0)
        return saved / max(self.token_count, 1)


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
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
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


def _token_count(row: dict[str, str]) -> int:
    return max(
        1,
        _int(row.get("aegis_candidate_pre_failure_total_tokens")),
        _int(row.get("aegis_safe_frontier_tokens")),
        _int(row.get("input_length")) + _int(row.get("generated_tokens")),
        _int(row.get("input_length")) + _int(row.get("target_output_tokens")),
    )


def _candidate_from_row(row: dict[str, str]) -> Candidate:
    row_id = str(row.get("rid") or row.get("agent_id") or "")
    return Candidate(
        workflow_id=str(row.get("workflow_id") or ""),
        row_id=row_id,
        critical=_bool(row.get("critical_agent")),
        decode_dominant=_bool(row.get("decode_dominant")),
        stage_width=max(_int(row.get("stage_width"), 1), 1),
        token_count=_token_count(row),
    )


def workflow_traces(run_dir: Path) -> list[WorkflowTrace]:
    rows = _original_rows(run_dir)
    by_workflow: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_workflow.setdefault(str(row["workflow_id"]), []).append(row)

    traces: list[WorkflowTrace] = []
    for workflow_id, workflow_rows in sorted(by_workflow.items()):
        stage_max_latency: dict[int, float] = {}
        for row in workflow_rows:
            stage_id = _int(row.get("stage_id"), -1)
            latency = max(
                _float(row.get("client_wall_e2e_latency_s")),
                _float(row.get("e2e_latency_s")),
                _float(row.get("total_latency_s")),
            )
            stage_max_latency[stage_id] = max(stage_max_latency.get(stage_id, 0.0), latency)
        baseline_wct = sum(stage_max_latency.values())
        if baseline_wct <= 0:
            baseline_wct = sum(_float(row.get("total_latency_s")) for row in workflow_rows)
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


def _protected_tokens_by_candidate(
    candidates: tuple[Candidate, ...], alpha: float
) -> dict[str, int]:
    critical = [item for item in candidates if item.critical]
    capacity = int(round(sum(item.token_count for item in critical) * alpha))
    protected: dict[str, int] = {}
    remaining = max(capacity, 0)
    for item in sorted(
        critical,
        key=lambda candidate: (
            -candidate.value_density,
            not candidate.decode_dominant,
            -candidate.token_count,
            candidate.row_id,
        ),
    ):
        if remaining <= 0:
            break
        take = min(item.token_count, remaining)
        if take > 0:
            protected[item.row_id] = take
            remaining -= take
    return protected


def _visible_backup_cost_s(protected_tokens: dict[str, int]) -> float:
    raw_ms = 0.0
    for tokens in protected_tokens.values():
        raw_ms += DEFAULT_GPU_SETUP_MS + tokens * DEFAULT_GPU_MS_PER_TOKEN
    return (raw_ms / 1000.0) * VISIBLE_BACKUP_OVERLAP_FACTOR


def _recovery_cost_s(trace: WorkflowTrace, protected_tokens: dict[str, int]) -> float:
    total = 0.0
    for candidate in trace.candidates:
        restored = protected_tokens.get(candidate.row_id, 0)
        if restored <= 0:
            cost = aegis_restore_overhead_s(
                action="recompute",
                token_count=candidate.token_count,
                restored_tokens=0,
            )
        else:
            cost = aegis_restore_overhead_s(
                action="gpu_replica",
                token_count=candidate.token_count,
                restored_tokens=restored,
            )
        total += cost * candidate.critical_path_weight
    return total


def _expected_wct_s(trace: WorkflowTrace, *, alpha: float, failure_rate: float) -> tuple[float, float, float, float]:
    protected = _protected_tokens_by_candidate(trace.candidates, alpha)
    backup_cost = _visible_backup_cost_s(protected)
    recovery_cost = _recovery_cost_s(trace, protected)
    expected_failure_cost = failure_rate * recovery_cost
    expected_wct = trace.baseline_wct_s + backup_cost + expected_failure_cost
    critical_tokens = sum(item.token_count for item in trace.candidates if item.critical)
    protected_fraction = sum(protected.values()) / max(critical_tokens, 1)
    return expected_wct, backup_cost, expected_failure_cost, protected_fraction


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

    critical_token_share_values = []
    for trace in traces:
        all_tokens = sum(item.token_count for item in trace.candidates)
        critical_tokens = sum(item.token_count for item in trace.candidates if item.critical)
        critical_token_share_values.append(critical_tokens / max(all_tokens, 1))

    baseline_by_rate: dict[float, float] = {}
    raw: dict[tuple[float, float], list[tuple[float, float, float, float]]] = {}
    for failure_rate in FAILURE_RATES:
        alpha0_values = [
            _expected_wct_s(trace, alpha=0.0, failure_rate=failure_rate)[0]
            for trace in traces
        ]
        baseline_by_rate[failure_rate] = mean(alpha0_values)
        for alpha in ALPHAS:
            raw[(failure_rate, alpha)] = [
                _expected_wct_s(trace, alpha=alpha, failure_rate=failure_rate)
                for trace in traces
            ]

    best_alpha_by_rate: dict[float, float] = {}
    for failure_rate in FAILURE_RATES:
        best_alpha_by_rate[failure_rate] = min(
            ALPHAS,
            key=lambda alpha: mean(item[0] for item in raw[(failure_rate, alpha)]),
        )

    base_meta = {
        "evidence_tier": evidence_tier,
        "source": "sglsim_agentic_stagebarrier_alpha_risk_counterfactual",
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
            "Trace-derived expected WCT objective from observed online SGLSim "
            "1-3-1-3-1 stage-barrier requests. Alpha controls only Critical "
            "agent GPUReplica tokens. Expected WCT combines observed baseline "
            "WCT, overlapped visible backup cost, and failure-rate-weighted "
            "recovery cost from the SGLSim recovery model."
        ),
        "runtime_workflow_traces": len(traces),
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(str(path) for path in run_dirs),
        "runtime_mean_critical_token_share": mean(critical_token_share_values),
        "visible_backup_overlap_factor": VISIBLE_BACKUP_OVERLAP_FACTOR,
    }

    rows: list[dict[str, object]] = []
    for failure_rate in FAILURE_RATES:
        baseline = baseline_by_rate[failure_rate]
        for alpha in ALPHAS:
            values = raw[(failure_rate, alpha)]
            expected_wcts = [item[0] for item in values]
            backup_costs = [item[1] for item in values]
            failure_costs = [item[2] for item in values]
            protected_fractions = [item[3] for item in values]
            normalized = [value / max(baseline, 1e-9) for value in expected_wcts]
            rows.append(
                {
                    "failure_rate_per_window": failure_rate,
                    "alpha_backup_fraction": alpha,
                    "normalized_expected_wct": mean(normalized),
                    "normalized_expected_wct_std": metric_std(normalized),
                    "normalized_expected_wct_ci95": metric_ci95(normalized),
                    "expected_wct_s": mean(expected_wcts),
                    "normal_backup_overhead_s": mean(backup_costs),
                    "expected_failure_recovery_s": mean(failure_costs),
                    "protected_critical_token_fraction": mean(protected_fractions),
                    "optimal_alpha_for_rate": best_alpha_by_rate[failure_rate],
                    **base_meta,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a staged 1-3-1-3-1 alpha/failure-risk tradeoff from "
            "observed SGLSim stage-barrier request_details.csv files."
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
    best = {
        row["failure_rate_per_window"]: row["optimal_alpha_for_rate"]
        for row in rows
        if float(row["alpha_backup_fraction"]) == 0.0
    }
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "workflow_traces": rows[0]["runtime_workflow_traces"],
                "best_alpha_by_failure_rate": best,
                "evidence_tier": args.evidence_tier,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
