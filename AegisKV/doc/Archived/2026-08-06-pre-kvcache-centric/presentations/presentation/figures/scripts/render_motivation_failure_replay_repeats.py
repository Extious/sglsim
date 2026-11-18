#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
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
TRACE_RESULTS_ROOT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "results"
    / "claude_code_trace_dp8"
)
MEASURED_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "figures" / "data" / "measured"
BEAMER_FIG_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "Figures"
DRAFT_PNG_DIR = BEAMER_FIG_DIR

OUT_CSV = MEASURED_DIR / "fig_motivation_failure_replay_repeats.csv"
OUT_PNG = DRAFT_PNG_DIR / "fig_motivation_failure_replay_repeats.png"
OUT_BEAMER = BEAMER_FIG_DIR / "fig_motivation_failure_replay_repeats.png"


BASELINE_RUNS = {
    9250: "motivation-main-subagent-dp8-no-failure-r1",
    9350: "motivation-repeat-seed9350-no_failure-r1",
    9450: "motivation-repeat-seed9450-no_failure-r1",
}

CASE_RUNS = [
    {
        "seed": 9250,
        "agent_label": "S1-A0",
        "agent_class": "Critical target",
        "run_id": "motivation-main-subagent-dp8-subagent-failure-r1",
    },
    {
        "seed": 9250,
        "agent_label": "S1-A1",
        "agent_class": "Non-Critical target",
        "run_id": "motivation-main-subagent-dp8-noncritical-subagent-failure-r1",
    },
    {
        "seed": 9250,
        "agent_label": "S3-A0",
        "agent_class": "Critical target",
        "run_id": "motivation-main-subagent-dp8-s3-a0-failure-r1",
    },
    {
        "seed": 9250,
        "agent_label": "S3-A1",
        "agent_class": "Non-Critical target",
        "run_id": "motivation-main-subagent-dp8-s3-a1-failure-r1",
    },
    {
        "seed": 9350,
        "agent_label": "S1-A0",
        "agent_class": "Critical target",
        "run_id": "motivation-repeat-seed9350-subagent_failure-r1",
    },
    {
        "seed": 9350,
        "agent_label": "S1-A1",
        "agent_class": "Non-Critical target",
        "run_id": "motivation-repeat-seed9350-noncritical_subagent_failure-r1",
    },
    {
        "seed": 9350,
        "agent_label": "S3-A0",
        "agent_class": "Critical target",
        "run_id": "motivation-repeat-seed9350-s3_a0_failure-r1",
    },
    {
        "seed": 9350,
        "agent_label": "S3-A1",
        "agent_class": "Non-Critical target",
        "run_id": "motivation-repeat-seed9350-s3_a1_failure-r1",
    },
    {
        "seed": 9450,
        "agent_label": "S1-A0",
        "agent_class": "Critical target",
        "run_id": "motivation-repeat-seed9450-subagent_failure-r1",
    },
    {
        "seed": 9450,
        "agent_label": "S1-A1",
        "agent_class": "Non-Critical target",
        "run_id": "motivation-repeat-seed9450-noncritical_subagent_failure-r1",
    },
    {
        "seed": 9450,
        "agent_label": "S3-A0",
        "agent_class": "Critical target",
        "run_id": "motivation-repeat-seed9450-s3_a0_failure-r1",
    },
    {
        "seed": 9450,
        "agent_label": "S3-A1",
        "agent_class": "Non-Critical target",
        "run_id": "motivation-repeat-seed9450-s3_a1_failure-r1",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_run_dir(run_id: str) -> Path:
    run_root = RESULTS_ROOT / run_id / "runs"
    candidates = sorted(
        path
        for path in run_root.glob("trial_*/*")
        if path.is_dir() and (path / "request_details.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"missing request_details.csv for {run_id}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_trace_dir() -> Path:
    candidates = sorted(
        path
        for path in TRACE_RESULTS_ROOT.glob("*")
        if path.is_dir() and (path / "request_details.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"missing real trace request_details.csv under {TRACE_RESULTS_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _workflow_wct_s(rows: list[dict[str, str]]) -> float:
    starts = [_float(row.get("created_time_s"), -1.0) for row in rows]
    finishes = [_float(row.get("finish_time_s"), -1.0) for row in rows]
    starts = [value for value in starts if value >= 0.0]
    finishes = [value for value in finishes if value >= 0.0]
    if not starts or not finishes:
        return 0.0
    return max(finishes) - min(starts)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _manifest_seed(run_id: str, fallback: int) -> int:
    manifest = RESULTS_ROOT / run_id / "agentic_workflow_manifest.json"
    if not manifest.exists():
        return fallback
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    plan = payload.get("plan") if isinstance(payload.get("plan"), list) else []
    if plan and isinstance(plan[0], dict):
        return int(plan[0].get("seed") or fallback)
    return fallback


def _staged_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline: dict[int, tuple[float, Path]] = {}
    for seed, run_id in BASELINE_RUNS.items():
        run_dir = _latest_run_dir(run_id)
        request_rows = _read_csv(run_dir / "request_details.csv")
        baseline[seed] = (_workflow_wct_s(request_rows), run_dir)

    case_rows: list[dict[str, object]] = []
    for case in CASE_RUNS:
        seed = int(case["seed"])
        baseline_wct, _ = baseline[seed]
        run_id = str(case["run_id"])
        run_dir = _latest_run_dir(run_id)
        request_rows = _read_csv(run_dir / "request_details.csv")
        wct_s = _workflow_wct_s(request_rows)
        retry_rows = sum(_bool(row.get("aegis_cross_rank_retry")) for row in request_rows)
        soft_failure_rows = sum(_bool(row.get("aegis_soft_failure_injected")) for row in request_rows)
        hard_failure_rows = sum(_bool(row.get("aegis_hard_failure_injected")) for row in request_rows)
        candidate_retry_rows = sum(_bool(row.get("aegis_candidate_retry")) for row in request_rows)
        restored_tokens = sum(_float(row.get("aegis_replay_restored_tokens")) for row in request_rows)
        recompute_tokens = sum(_float(row.get("aegis_replay_recompute_tokens")) for row in request_rows)
        case_rows.append(
            {
                "row_type": "staged_failure_case",
                "seed": _manifest_seed(run_id, seed),
                "agent_label": case["agent_label"],
                "agent_class": case["agent_class"],
                "workflow_wct_s": f"{wct_s:.6f}",
                "baseline_wct_s": f"{baseline_wct:.6f}",
                "delta_vs_seed_baseline_s": f"{wct_s - baseline_wct:.6f}",
                "request_rows": len(request_rows),
                "retry_rows": retry_rows,
                "soft_failure_rows": soft_failure_rows,
                "hard_failure_rows": hard_failure_rows,
                "candidate_retry_rows": candidate_retry_rows,
                "restored_tokens": int(restored_tokens),
                "recompute_tokens": int(recompute_tokens),
                "run_id": run_id,
                "run_dir": str(run_dir),
                "source": "sglsim_agentic_online_motivation_failure_repeat",
                "evidence_tier": "sglsim-agentic-online-repeat-dp8",
                "note": (
                    "Critical targets are eligible replay candidates and receive "
                    "soft-failure injection; Non-Critical targets are recorded as "
                    "ineligible target controls when candidate_retry_rows is zero."
                ),
            }
        )

    summary_rows: list[dict[str, object]] = []
    for agent_class in ("Critical target", "Non-Critical target"):
        samples = [row for row in case_rows if row["agent_class"] == agent_class]
        deltas = [_float(row["delta_vs_seed_baseline_s"]) for row in samples]
        summary_rows.append(
            {
                "row_type": "staged_class_summary",
                "agent_class": agent_class,
                "samples": len(samples),
                "seed_count": len({int(row["seed"]) for row in samples}),
                "target_positions": len({str(row["agent_label"]) for row in samples}),
                "mean_delta_vs_seed_baseline_s": f"{statistics.mean(deltas):.6f}",
                "ci95_delta_vs_seed_baseline_s": f"{_ci95(deltas):.6f}",
                "min_delta_vs_seed_baseline_s": f"{min(deltas):.6f}",
                "max_delta_vs_seed_baseline_s": f"{max(deltas):.6f}",
                "retry_rows": sum(int(row["retry_rows"]) for row in samples),
                "soft_failure_rows": sum(int(row["soft_failure_rows"]) for row in samples),
                "candidate_retry_rows": sum(int(row["candidate_retry_rows"]) for row in samples),
                "source": "sglsim_agentic_online_motivation_failure_repeat",
                "evidence_tier": "sglsim-agentic-online-repeat-dp8",
            }
        )
    return case_rows, summary_rows


def _real_trace_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_dir = _latest_trace_dir()
    request_rows = [
        row
        for row in _read_csv(run_dir / "request_details.csv")
        if str(row.get("status") or "") == "completed"
    ]
    position_rows: list[dict[str, object]] = []
    for row in request_rows:
        overhead_ms = _float(row.get("aegis_candidate_failover_overhead_s")) * 1000.0
        position_rows.append(
            {
                "row_type": "real_trace_failure_position",
                "run_id": run_dir.name,
                "workflow_id": row.get("workflow_id", ""),
                "rid": row.get("rid", ""),
                "critical_agent": str(_bool(row.get("critical_agent"))),
                "decode_dominant": str(_bool(row.get("decode_dominant"))),
                "input_tokens": int(_float(row.get("input_length"))),
                "output_tokens": int(_float(row.get("output_length"))),
                "candidate_retry": str(_bool(row.get("aegis_candidate_retry"))),
                "candidate_failed_dp_rank": int(_float(row.get("aegis_candidate_failed_dp_rank"), -1.0)),
                "candidate_pre_failure_output_tokens": int(
                    _float(row.get("aegis_candidate_pre_failure_output_tokens"))
                ),
                "candidate_backed_up_tokens": int(_float(row.get("aegis_candidate_backed_up_tokens"))),
                "candidate_recompute_tokens": int(_float(row.get("aegis_candidate_recompute_tokens"))),
                "candidate_failover_overhead_ms": f"{overhead_ms:.6f}",
                "candidate_recovery_action": row.get("aegis_candidate_recovery_action", ""),
                "source": "sglsim_claude_code_trace_failure_position_replay",
                "evidence_tier": "sglsim-real-trace-failure-position-replay",
                "run_dir": str(run_dir),
            }
        )

    overheads = [_float(row["candidate_failover_overhead_ms"]) for row in position_rows]
    recompute = [_float(row["candidate_recompute_tokens"]) for row in position_rows]
    backed = [_float(row["candidate_backed_up_tokens"]) for row in position_rows]
    workflows = {str(row.get("workflow_id") or "") for row in request_rows}
    summary_rows = [
        {
            "row_type": "real_trace_summary",
            "metric": "failure_positions",
            "value": len(position_rows),
            "unit": "requests",
            "source": "sglsim_claude_code_trace_failure_position_replay",
            "evidence_tier": "sglsim-real-trace-failure-position-replay",
        },
        {
            "row_type": "real_trace_summary",
            "metric": "workflows",
            "value": len(workflows),
            "unit": "workflows",
            "source": "sglsim_claude_code_trace_failure_position_replay",
            "evidence_tier": "sglsim-real-trace-failure-position-replay",
        },
        {
            "row_type": "real_trace_summary",
            "metric": "p50_candidate_failover_overhead_ms",
            "value": f"{_percentile(overheads, 0.50):.6f}",
            "unit": "ms",
            "source": "sglsim_claude_code_trace_failure_position_replay",
            "evidence_tier": "sglsim-real-trace-failure-position-replay",
        },
        {
            "row_type": "real_trace_summary",
            "metric": "p90_candidate_failover_overhead_ms",
            "value": f"{_percentile(overheads, 0.90):.6f}",
            "unit": "ms",
            "source": "sglsim_claude_code_trace_failure_position_replay",
            "evidence_tier": "sglsim-real-trace-failure-position-replay",
        },
        {
            "row_type": "real_trace_summary",
            "metric": "p99_candidate_failover_overhead_ms",
            "value": f"{_percentile(overheads, 0.99):.6f}",
            "unit": "ms",
            "source": "sglsim_claude_code_trace_failure_position_replay",
            "evidence_tier": "sglsim-real-trace-failure-position-replay",
        },
        {
            "row_type": "real_trace_summary",
            "metric": "mean_candidate_recompute_tokens",
            "value": f"{statistics.mean(recompute):.6f}" if recompute else "0.000000",
            "unit": "tokens",
            "source": "sglsim_claude_code_trace_failure_position_replay",
            "evidence_tier": "sglsim-real-trace-failure-position-replay",
        },
        {
            "row_type": "real_trace_summary",
            "metric": "max_candidate_recompute_tokens",
            "value": int(max(recompute) if recompute else 0),
            "unit": "tokens",
            "source": "sglsim_claude_code_trace_failure_position_replay",
            "evidence_tier": "sglsim-real-trace-failure-position-replay",
        },
        {
            "row_type": "real_trace_summary",
            "metric": "mean_candidate_backed_up_tokens",
            "value": f"{statistics.mean(backed):.6f}" if backed else "0.000000",
            "unit": "tokens",
            "source": "sglsim_claude_code_trace_failure_position_replay",
            "evidence_tier": "sglsim-real-trace-failure-position-replay",
        },
    ]
    return position_rows, summary_rows


def _render(case_rows: list[dict[str, object]], summary_rows: list[dict[str, object]], position_rows: list[dict[str, object]]) -> None:
    DRAFT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    BEAMER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.4,
            "figure.dpi": 180,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"wspace": 0.30})

    order = ["Non-Critical target", "Critical target"]
    colors = {
        "Critical target": "#DC2626",
        "Non-Critical target": "#64748B",
    }
    x = np.arange(len(order))
    means = [
        _float(next(row for row in summary_rows if row["agent_class"] == label)["mean_delta_vs_seed_baseline_s"])
        for label in order
    ]
    cis = [
        _float(next(row for row in summary_rows if row["agent_class"] == label)["ci95_delta_vs_seed_baseline_s"])
        for label in order
    ]
    by_seed: dict[int, dict[str, list[float]]] = {}
    for row in case_rows:
        seed = int(_float(row.get("seed")))
        by_seed.setdefault(seed, {}).setdefault(str(row["agent_class"]), []).append(
            _float(row["delta_vs_seed_baseline_s"])
        )
    for seed, seed_values in sorted(by_seed.items()):
        paired: list[float] = []
        for label in order:
            values = seed_values.get(label, [])
            paired.append(statistics.mean(values) if values else 0.0)
        ax0.plot(
            x,
            paired,
            color="#94A3B8",
            linewidth=1.0,
            alpha=0.85,
            zorder=1,
        )
        ax0.scatter(
            x,
            paired,
            s=30,
            color="#334155",
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        ax0.text(
            x[0] - 0.05,
            paired[0],
            str(seed),
            ha="right",
            va="center",
            fontsize=7.0,
            color="#64748B",
        )

    for idx, label in enumerate(order):
        samples = [
            row
            for row in case_rows
            if str(row["agent_class"]) == label
        ]
        for point_idx, row in enumerate(samples):
            jitter = ((point_idx % 3) - 1) * 0.045
            ax0.scatter(
                idx + jitter,
                _float(row["delta_vs_seed_baseline_s"]),
                s=38,
                color="#111827",
                alpha=0.85,
                zorder=4,
            )
        ax0.errorbar(
            idx,
            means[idx],
            yerr=cis[idx],
            fmt="D",
            markersize=6.2,
            capsize=5,
            color=colors[label],
            markeredgecolor="#111827",
            markeredgewidth=0.7,
            linewidth=1.2,
            zorder=5,
        )
    ax0.axhline(0.0, color="#9CA3AF", linewidth=1.0, linestyle="--")
    ax0.set_xticks(x, ["Non-Critical\ntarget", "Critical\neligible"])
    ax0.set_ylabel("WCT delta vs same-seed baseline (s)")
    ax0.set_title("(a) DP8 paired failure-position repeats")
    ax0.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax0.text(
        0.02,
        0.96,
        "seed-paired lines; diamonds show mean +/- 95% CI",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#374151",
    )

    overheads = sorted(_float(row["candidate_failover_overhead_ms"]) for row in position_rows)
    cdf = [(idx + 1) / max(len(overheads), 1) for idx in range(len(overheads))]
    ax1.plot(
        overheads,
        cdf,
        color="#2563EB",
        linewidth=2.4,
        label="real-trace positions",
    )
    for pct, color in [(0.50, "#111827"), (0.90, "#F59E0B"), (0.99, "#DC2626")]:
        value = _percentile(overheads, pct)
        ax1.axvline(value, color=color, linewidth=1.4, linestyle="--")
        ax1.text(
            value,
            0.88 - 0.11 * [0.50, 0.90, 0.99].index(pct),
            f"P{int(pct * 100)}={value:.1f}ms",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=color,
        )
    ax1.set_xlabel("Candidate failover overhead on real trace (ms)")
    ax1.set_ylabel("empirical CDF")
    ax1.set_title(f"(b) {len(position_rows)} real-trace-derived failure positions")
    ax1.grid(True, color="#E5E7EB", linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, loc="lower right")

    fig.suptitle(
        "Motivation strict replay: Critical failures move WCT; real trace supplies failure positions",
        fontsize=12.2,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Critical repeats use measured online SGLSim soft-failure replay. Non-Critical target controls expose the current Critical-only replay eligibility boundary.",
        ha="center",
        fontsize=8.4,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_BEAMER, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    case_rows, class_summary_rows = _staged_rows()
    trace_position_rows, trace_summary_rows = _real_trace_rows()
    rows = case_rows + class_summary_rows + trace_position_rows + trace_summary_rows
    _write_csv(OUT_CSV, rows)
    _render(case_rows, class_summary_rows, trace_position_rows)
    print(
        json.dumps(
            {
                "csv": str(OUT_CSV),
                "png": str(OUT_PNG),
                "beamer_png": str(OUT_BEAMER),
                "staged_failure_cases": len(case_rows),
                "real_trace_positions": len(trace_position_rows),
                "rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
