#!/usr/bin/env python3
"""Render an explicitly labelled 128K / 64-GPU projection.

This is a model-based extrapolation, not a measured SGLSim result.  The model
uses the measured 1K AegisKV/No-Backup CDF shape as its anchor, a clean 128K
AegisKV run only to calibrate context-time and retry-token scale, and the
provided 64-GPU availability trace's reported mean availability (58.07 GPUs).
All assumptions and derived quantities are emitted as JSON next to the figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
ANCHOR_DIR = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "demo"
    / "results"
    / "wct_cdf"
    / "20260722-013820"
)
CALIBRATION_RESULT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "demo"
    / "results"
    / "formal_128k_clean"
    / "20260722-102800"
    / "aegiskv.json"
)
OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "presentation"
    / "Figures"
    / "fig_weka_subagent_policy_wct_cdf_projected_128k_dp64.png"
)
DRAFT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "presentation"
    / "figures"
    / "png"
    / "fig_weka_subagent_policy_wct_cdf_projected_128k_dp64.png"
)
CSV_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "presentation"
    / "figures"
    / "data"
    / "projected"
    / "fig_weka_subagent_policy_wct_cdf_projected_128k_dp64.csv"
)
ASSUMPTIONS_OUTPUT = CSV_OUTPUT.with_suffix(".assumptions.json")

TOTAL_GPUS = 64
AVERAGE_AVAILABLE_GPUS = 58.07
TARGET_CONTEXT_TOKENS = 131_072

POLICIES = (
    ("aegiskv", "AegisKV", "#0072B2"),
    ("no_backup", "No Backup", "#6B7280"),
)
TOKEN_SOURCES = (
    ("l1_reuse_tokens", "L1 reuse", "#059669"),
    ("l2_reuse_tokens", "L2 reuse", "#2563EB"),
    ("recompute_tokens", "Recompute", "#6B7280"),
)
COLOR_GRID = "#E5E7EB"
COLOR_MUTED = "#64748B"


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def workflow_wcts(payload: dict) -> list[tuple[str, float]]:
    values = [
        (str(row.get("workflow_id") or ""), float(row["wct_s"]))
        for row in payload.get("workflows", [])
    ]
    if not values:
        raise ValueError("anchor result has no workflow WCT samples")
    return values


def retry_breakdown(payload: dict) -> dict[str, int]:
    events = [
        event
        for workflow in payload.get("workflows", [])
        for event in workflow.get("retry_events", [])
    ]
    result = {
        "retry_count": len(events),
        "l1_reuse_tokens": 0,
        "l2_reuse_tokens": 0,
        "recompute_tokens": 0,
    }
    for event in events:
        action = str(event.get("recovery_action") or "recompute")
        restored = max(int(event.get("restored_tokens", 0) or 0), 0)
        recompute = max(int(event.get("recompute_tokens", 0) or 0), 0)
        if action == "l1_backup":
            result["l1_reuse_tokens"] += restored
        elif action == "l2_backup":
            result["l2_reuse_tokens"] += restored
        elif action == "l1_l2_backup" and restored:
            raise ValueError("anchor has a mixed restore without exact tier attribution")
        elif restored:
            raise ValueError(f"unsupported restore action: {action}")
        result["recompute_tokens"] += recompute

    summary = payload.get("retry_summary", {})
    restored = result["l1_reuse_tokens"] + result["l2_reuse_tokens"]
    if int(summary.get("count", 0)) != result["retry_count"]:
        raise ValueError("retry event count does not match anchor summary")
    if int(summary.get("restored_tokens", 0)) != restored:
        raise ValueError("restored-token events do not match anchor summary")
    if int(summary.get("recompute_tokens", 0)) != result["recompute_tokens"]:
        raise ValueError("recompute-token events do not match anchor summary")
    result["retry_tokens_total"] = restored + result["recompute_tokens"]
    return result


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_wct_s": sum(values) / len(values),
        "p50_wct_s": percentile(values, 0.50),
        "p90_wct_s": percentile(values, 0.90),
        "p99_wct_s": percentile(values, 0.99),
        "max_wct_s": max(values),
    }


def expand_distribution(values: list[float], count: int) -> list[float]:
    """Deterministically interpolate an empirical distribution to ``count`` points."""
    if count < 1:
        raise ValueError("expanded workflow count must be positive")
    return [percentile(values, (index + 0.5) / count) for index in range(count)]


def smooth_cdf(values: list[float], point_count: int = 600) -> tuple[list[float], list[float]]:
    """Return a Gaussian-kernel CDF for display without changing sample metrics."""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    stddev = math.sqrt(variance)
    bandwidth = max(
        1.06 * stddev * len(values) ** (-0.2),
        (max(values) - min(values)) / 200.0,
        1e-6,
    )
    lower = min(values) - 3.0 * bandwidth
    upper = max(values) + 3.0 * bandwidth
    xs = [
        lower + (upper - lower) * index / (point_count - 1)
        for index in range(point_count)
    ]
    denominator = bandwidth * math.sqrt(2.0)
    ys = [
        sum(0.5 * (1.0 + math.erf((x - value) / denominator)) for value in values)
        / len(values)
        for x in xs
    ]
    return xs, ys


def anchor_unavailable_fraction(payload: dict) -> float:
    events = payload.get("failure_manifest", {}).get("events", [])
    rank_seconds = sum(float(event.get("duration_s", 0.0)) for event in events)
    horizon_s = max(value for _workflow_id, value in workflow_wcts(payload))
    dp_size = int(payload["dp_size"])
    if rank_seconds <= 0 or horizon_s <= 0 or dp_size <= 0:
        raise ValueError("cannot derive anchor GPU-time unavailability")
    return rank_seconds / (dp_size * horizon_s)


def build_projection(
    anchor_dir: Path,
    calibration_result: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    anchor_aegis = read_json(anchor_dir / "aegiskv.json")
    anchor_no_backup = read_json(anchor_dir / "no_backup.json")
    calibration = read_json(calibration_result)

    anchor_aegis_mean = float(anchor_aegis["wct_summary_s"]["mean"])
    anchor_no_backup_mean = float(anchor_no_backup["wct_summary_s"]["mean"])
    context_time_scale = (
        float(calibration["wct_summary_s"]["mean"]) / anchor_aegis_mean
    )

    projected_unavailable_fraction = (
        TOTAL_GPUS - AVERAGE_AVAILABLE_GPUS
    ) / TOTAL_GPUS
    measured_unavailable_fraction = anchor_unavailable_fraction(anchor_aegis)
    failure_exposure_scale = (
        projected_unavailable_fraction / measured_unavailable_fraction
    )

    anchor_dp = int(anchor_aegis["dp_size"])
    if TOTAL_GPUS % anchor_dp:
        raise ValueError("target DP size must be an integer multiple of anchor DP")
    workflow_scale = TOTAL_GPUS // anchor_dp
    anchor_workflow_count = len(workflow_wcts(anchor_aegis))
    target_workflow_count = anchor_workflow_count * workflow_scale
    workflows_per_dp_rank = anchor_workflow_count / anchor_dp
    availability_capacity_penalty = TOTAL_GPUS / AVERAGE_AVAILABLE_GPUS
    projected_aegis_mean = (
        anchor_aegis_mean * context_time_scale * availability_capacity_penalty
    )
    anchor_recovery_gap = anchor_no_backup_mean / anchor_aegis_mean - 1.0
    projected_recovery_gap = anchor_recovery_gap * failure_exposure_scale
    projected_no_backup_mean = projected_aegis_mean * (1.0 + projected_recovery_gap)

    target_means = {
        "aegiskv": projected_aegis_mean,
        "no_backup": projected_no_backup_mean,
    }
    payloads = {
        "aegiskv": anchor_aegis,
        "no_backup": anchor_no_backup,
    }
    curves: dict[str, list[tuple[str, float]]] = {}
    for suffix, _label, _color in POLICIES:
        samples = workflow_wcts(payloads[suffix])
        expanded = expand_distribution(
            [value for _workflow_id, value in samples], target_workflow_count
        )
        measured_mean = sum(expanded) / len(expanded)
        scale = target_means[suffix] / measured_mean
        curves[suffix] = [
            (f"simulated-wf-{index:03d}", value * scale)
            for index, value in enumerate(expanded)
        ]

    anchor_tokens = retry_breakdown(anchor_aegis)
    anchor_total = anchor_tokens["retry_tokens_total"]
    token_fractions = {
        key: anchor_tokens[key] / anchor_total
        for key in ("l1_reuse_tokens", "l2_reuse_tokens", "recompute_tokens")
    }
    projected_retry_count = round(
        anchor_tokens["retry_count"] * failure_exposure_scale * workflow_scale
    )
    calibration_retry = calibration["retry_summary"]
    tokens_per_retry_128k = (
        int(calibration_retry["restored_tokens"])
        + int(calibration_retry["recompute_tokens"])
    ) / int(calibration_retry["count"])
    projected_retry_tokens = round(projected_retry_count * tokens_per_retry_128k)
    projected_aegis_tokens = {
        "l1_reuse_tokens": round(
            projected_retry_tokens * token_fractions["l1_reuse_tokens"]
        ),
        "l2_reuse_tokens": round(
            projected_retry_tokens * token_fractions["l2_reuse_tokens"]
        ),
    }
    projected_aegis_tokens["recompute_tokens"] = projected_retry_tokens - sum(
        projected_aegis_tokens.values()
    )
    projected_tokens = {
        "aegiskv": projected_aegis_tokens,
        "no_backup": {
            "l1_reuse_tokens": 0,
            "l2_reuse_tokens": 0,
            "recompute_tokens": projected_retry_tokens,
        },
    }

    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for suffix, label, _color in POLICIES:
        for workflow_id, wct_s in curves[suffix]:
            rows.append(
                {
                    "policy_suffix": suffix,
                    "policy_label": label,
                    "workflow_id": workflow_id,
                    "projected_wct_s": round(wct_s, 6),
                    "projection_status": "model_projection_not_measured",
                }
            )
        stats = summarize([value for _workflow_id, value in curves[suffix]])
        summary = {
            "policy_suffix": suffix,
            "policy_label": label,
            "workflow_id": "__summary__",
            "projected_wct_s": "",
            "workflow_count": len(curves[suffix]),
            **{key: round(value, 6) for key, value in stats.items()},
            "retry_count": projected_retry_count,
            **projected_tokens[suffix],
            "retry_tokens_total": projected_retry_tokens,
            "projection_status": "model_projection_not_measured",
        }
        rows.append(summary)
        summaries[suffix] = summary

    assumptions: dict[str, object] = {
        "projection_status": "model_projection_not_measured",
        "target": {
            "model": "Qwen/Qwen3-8B",
            "context_tokens_max": TARGET_CONTEXT_TOKENS,
            "gpu_count": TOTAL_GPUS,
            "dp_size": TOTAL_GPUS,
            "average_available_gpus": AVERAGE_AVAILABLE_GPUS,
            "average_gpu_availability_fraction": AVERAGE_AVAILABLE_GPUS / TOTAL_GPUS,
            "average_gpu_time_unavailable_fraction": projected_unavailable_fraction,
            "workflow_count": len(curves["aegiskv"]),
            "workflows_per_dp_rank": workflows_per_dp_rank,
            "failure_trace_replay": "15-hour curve normalized to workload makespan",
        },
        "sources": {
            "cdf_and_retry_mix_anchor": str(anchor_dir),
            "context_and_retry_token_scale_calibration": str(calibration_result),
        },
        "model": {
            "workflow_scale_from_dp8_to_dp64": workflow_scale,
            "anchor_gpu_time_unavailable_fraction": measured_unavailable_fraction,
            "failure_exposure_scale": failure_exposure_scale,
            "context_time_scale": context_time_scale,
            "availability_capacity_penalty": availability_capacity_penalty,
            "anchor_no_backup_gap_relative_to_aegiskv": anchor_recovery_gap,
            "projected_no_backup_gap_relative_to_aegiskv": projected_recovery_gap,
            "projected_retry_count_per_policy": projected_retry_count,
            "projected_retry_tokens_per_policy": projected_retry_tokens,
            "retry_tokens_per_retry_128k": tokens_per_retry_128k,
            "aegiskv_retry_token_fractions": token_fractions,
            "cdf_rendering": "Gaussian-kernel smoothed CDF over 160 simulated workflow samples",
        },
        "limitations": [
            "CDF shape is quantile-interpolated from 20 measured DP=8 samples to 160 simulated samples.",
            "The mean-availability model does not reproduce the large correlated outage near hour 3.3.",
            "Backup-rank co-failure, queue spillover, and alpha/beta adaptation at DP=64 are not simulated.",
            "The model preserves the anchor load density of 2.5 workflows per DP rank.",
        ],
        "projected_summaries": summaries,
    }
    return rows, assumptions


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict[str, object]], output_paths: list[Path]) -> None:
    summaries = {
        str(row["policy_suffix"]): row
        for row in rows
        if row.get("workflow_id") == "__summary__"
    }
    curves = {
        suffix: [
            float(row["projected_wct_s"])
            for row in rows
            if row.get("policy_suffix") == suffix
            and row.get("workflow_id") != "__summary__"
        ]
        for suffix, _label, _color in POLICIES
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 300,
        }
    )
    fig, (ax, ax_retry) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.45),
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )

    for suffix, label, color in POLICIES:
        xs, ys = smooth_cdf(curves[suffix])
        ax.plot(xs, ys, color=color, linewidth=2.4, label=label)
    ax.set_xlabel("Simulated workflow completion time (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=11.5)
    ax.set_title(
        "(a) Simulated workflow completion time",
        loc="left",
        weight="bold",
        fontsize=13,
        color=COLOR_MUTED,
        pad=8,
    )
    ax.spines[["top", "right"]].set_visible(False)

    positions = list(range(len(POLICIES)))
    bottoms = [0.0 for _ in POLICIES]
    for key, label, color in TOKEN_SOURCES:
        percentages: list[float] = []
        counts: list[int] = []
        for suffix, _policy_label, _policy_color in POLICIES:
            count = int(summaries[suffix][key])
            total = int(summaries[suffix]["retry_tokens_total"])
            counts.append(count)
            percentages.append(100.0 * count / total if total else 0.0)
        ax_retry.bar(
            positions,
            percentages,
            bottom=bottoms,
            width=0.62,
            color=color,
            label=label,
            zorder=2,
        )
        for position, bottom, percentage, count in zip(
            positions, bottoms, percentages, counts
        ):
            if percentage < 7.0:
                continue
            ax_retry.text(
                position,
                bottom + percentage / 2.0,
                f"{count / 1_000_000:.2f}M\n({percentage:.1f}%)",
                ha="center",
                va="center",
                color="white",
                fontsize=10.5,
                weight="bold",
                linespacing=1.15,
                zorder=3,
            )
        bottoms = [bottom + value for bottom, value in zip(bottoms, percentages)]

    ax_retry.set_xticks(
        positions,
        [
            f"{label}\n{int(summaries[suffix]['retry_count'])} simulated retries"
            for suffix, label, _color in POLICIES
        ],
    )
    ax_retry.set_ylabel("Simulated retry-token source (%)")
    ax_retry.set_ylim(0.0, 112.0)
    ax_retry.set_yticks(range(0, 101, 20))
    ax_retry.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax_retry.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.2,
        fontsize=10.5,
    )
    ax_retry.set_title(
        "(b) Simulated retry-token reuse",
        loc="left",
        weight="bold",
        fontsize=13,
        color=COLOR_MUTED,
        pad=8,
    )
    ax_retry.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "SIMULATED  |  Real CC trace / Qwen3-8B / "
        "context <= 128K / DP=64 / 160 workflows / avg. available = 58.07",
        x=0.06,
        y=0.992,
        ha="left",
        fontsize=11.8,
        weight="bold",
        color=COLOR_MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.16, wspace=0.30)
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-dir", type=Path, default=ANCHOR_DIR)
    parser.add_argument(
        "--calibration-result", type=Path, default=CALIBRATION_RESULT
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--draft-output", type=Path, default=DRAFT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=CSV_OUTPUT)
    parser.add_argument("--assumptions", type=Path, default=ASSUMPTIONS_OUTPUT)
    args = parser.parse_args()

    rows, assumptions = build_projection(args.anchor_dir, args.calibration_result)
    write_csv(rows, args.csv)
    args.assumptions.parent.mkdir(parents=True, exist_ok=True)
    args.assumptions.write_text(
        json.dumps(assumptions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    render(rows, [args.output, args.draft_output])
    print(f"wrote {args.csv}")
    print(f"wrote {args.assumptions}")
    print(f"wrote {args.output}")
    print(f"wrote {args.draft_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
