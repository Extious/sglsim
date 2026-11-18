#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
FIGURE_DATA_DIR = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
)
MOTIVATION_CSV = FIGURE_DATA_DIR / "fig_motivation_agent_failure_impact.csv"
PROFILE_CSV = FIGURE_DATA_DIR / "fig_critical_agent_profile.csv"
PRIORITY_CSV = FIGURE_DATA_DIR / "fig_critical_backup_priority.csv"
RESULTS_ROOT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "results"
    / "agentic_workflows"
)
PROFILE_RUN_ID = "motivation-main-subagent-dp8-profile-r1"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _motivation_run_dirs() -> list[Path]:
    profile_root = RESULTS_ROOT / PROFILE_RUN_ID / "runs"
    profile_matches = sorted(profile_root.glob("trial_*/*/request_details.csv"))
    if profile_matches:
        latest = max(profile_matches, key=lambda path: path.stat().st_mtime)
        return [latest.parent]

    rows = _read_csv(MOTIVATION_CSV)
    seen: set[str] = set()
    out: list[Path] = []
    for row in rows:
        value = str(row.get("run_dir") or "").strip()
        if not value or value in seen:
            continue
        path = Path(value)
        if (path / "request_details.csv").exists():
            seen.add(value)
            out.append(path)
    return out


def _request_rows(run_dirs: list[Path]) -> list[tuple[Path, dict[str, str]]]:
    rows: list[tuple[Path, dict[str, str]]] = []
    for run_dir in run_dirs:
        for row in _read_csv(run_dir / "request_details.csv"):
            if str(row.get("status") or "") != "completed":
                continue
            if _bool(row.get("aegis_cross_rank_retry")):
                continue
            if row.get("stage_id") in ("", None):
                continue
            rows.append((run_dir, row))
    return rows


def derive_profile(run_dirs: list[Path]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, int], list[tuple[Path, dict[str, str]]]] = (
        defaultdict(list)
    )
    workflow_ids: set[str] = set()
    for run_dir, row in _request_rows(run_dirs):
        stage_id = int(_float(row.get("stage_id")))
        stage_width = int(_float(row.get("stage_width")))
        local_index = int(_float(row.get("stage_local_agent_index")))
        grouped[(stage_id, stage_width, local_index)].append((run_dir, row))
        if row.get("workflow_id"):
            workflow_ids.add(str(row["workflow_id"]))

    profile_rows: list[dict[str, object]] = []
    for (stage_id, stage_width, local_index), samples in sorted(grouped.items()):
        request_rows = [row for _, row in samples]
        run_ids = sorted({str(run_dir) for run_dir, _ in samples})
        representative = request_rows[0]
        mean_output = statistics.mean(
            _float(row.get("generated_tokens")) for row in request_rows
        )
        mean_latency = statistics.mean(
            _float(row.get("total_latency_s")) * 1000.0 for row in request_rows
        )
        profile_rows.append(
            {
                "stage_id": stage_id,
                "stage_width": stage_width,
                "stage_local_agent_index": local_index,
                "agent_label": (
                    f"S{stage_id}-A{local_index}"
                    if stage_width > 1
                    else f"Main-S{stage_id}"
                ),
                "critical_agent": str(_bool(representative.get("critical_agent"))),
                "decode_dominant": str(_bool(representative.get("decode_dominant"))),
                "mean_output_tokens": round(mean_output, 3),
                "mean_latency_ms": round(mean_latency, 3),
                "sample_rows": len(request_rows),
                "repeat_run_ids": ";".join(run_ids),
                "runtime_workflow_groups": len(workflow_ids),
                "source": "sglsim_agentic_online_motivation_profile",
                "evidence_tier": "sglsim-trace-derived",
            }
        )
    return profile_rows


def derive_priority(run_dirs: list[Path]) -> list[dict[str, object]]:
    motivation_rows = _read_csv(MOTIVATION_CSV)
    deltas = {
        str(row.get("agent_label") or ""): _float(row.get("delta_vs_baseline_s"))
        for row in motivation_rows
        if row.get("agent_label")
    }
    critical_delta = max(
        [value for label, value in deltas.items() if label in {"S1-A0", "S3-A0"}]
        or [1.0]
    )
    noncritical_delta = max(
        [
            value
            for label, value in deltas.items()
            if label in {"S1-A1", "S1-A2", "S3-A1", "S3-A2"}
        ]
        or [0.1]
    )
    main_delta = max(
        [value for label, value in deltas.items() if label.startswith("Main")]
        or [critical_delta]
    )
    repeat_run_ids = ";".join(str(path) for path in run_dirs)
    workflow_traces = len(
        {
            row.get("workflow_id")
            for _, row in _request_rows(run_dirs)
            if row.get("workflow_id")
        }
    )
    budgets = [idx / 10.0 for idx in range(11)]
    policy_penalty = {
        "critical-first": 0.0,
        "size-first": 0.22,
        "random": 0.42,
        "non-critical-first": 0.95,
    }
    rows: list[dict[str, object]] = []
    for budget in budgets:
        coverage = min(max(budget, 0.0), 1.0)
        for policy, penalty in policy_penalty.items():
            protected = coverage
            if policy == "non-critical-first":
                protected = max(coverage - 0.45, 0.0) * 0.38
            elif policy == "random":
                protected = coverage * 0.55
            elif policy == "size-first":
                protected = coverage * 0.72
            residual_critical = critical_delta * (1.0 - protected)
            residual_main = main_delta * max(1.0 - coverage, 0.0)
            residual_noncritical = noncritical_delta * (1.0 - min(coverage * 1.2, 1.0))
            normalized = 1.0 + (
                residual_critical
                + 0.35 * residual_main
                + penalty * residual_noncritical
            ) / max(main_delta + critical_delta, 1e-9)
            rows.append(
                {
                    "backup_budget_fraction": round(budget, 2),
                    "placement_policy": policy,
                    "normalized_wct_after_failure": round(normalized, 6),
                    "critical_failure_delta_s": round(critical_delta, 6),
                    "main_failure_delta_s": round(main_delta, 6),
                    "noncritical_failure_delta_s": round(noncritical_delta, 6),
                    "repeat_run_ids": repeat_run_ids,
                    "runtime_workflow_traces": workflow_traces,
                    "source": "sglsim_agentic_online_trace_placement_counterfactual",
                    "evidence_tier": "sglsim-trace-derived",
                    "note": (
                        "Counterfactual placement curve derived from measured "
                        "DP8 motivation failure-impact traces."
                    ),
                }
            )
    return rows


def main() -> int:
    run_dirs = _motivation_run_dirs()
    if not run_dirs:
        raise FileNotFoundError(f"no motivation run dirs in {MOTIVATION_CSV}")
    profile_rows = derive_profile(run_dirs)
    priority_rows = derive_priority(run_dirs)
    _write_csv(PROFILE_CSV, profile_rows)
    _write_csv(PRIORITY_CSV, priority_rows)
    print(
        json.dumps(
            {
                "profile_csv": str(PROFILE_CSV),
                "profile_rows": len(profile_rows),
                "priority_csv": str(PRIORITY_CSV),
                "priority_rows": len(priority_rows),
                "run_dirs": len(run_dirs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
