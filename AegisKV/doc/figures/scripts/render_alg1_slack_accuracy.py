#!/usr/bin/env python3
"""Render an honest Algorithm-1 Slack prediction accuracy baseline.

The default input is the archived, workflow-disjoint DP4 x PP4 SGLSim runtime
evaluation.  Only fan-out stages are plotted by default: singleton stages have
CPM Slack fixed to zero and would otherwise make the point-error metrics look
artificially strong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "Archived"
    / "2026-08-06-pre-kvcache-centric"
    / "presentations"
    / "presentation"
    / "figures"
    / "data"
    / "measured"
    / "fig_alg1_runtime_slack_calibration.csv"
)
DEFAULT_OUTPUT_PREFIX = ROOT / "figures" / "alg1-slack-prediction-accuracy"
DEFAULT_SUMMARY = (
    ROOT / "figures" / "data" / "alg1-slack-prediction-accuracy-summary.json"
)

REQUIRED_COLUMNS = {
    "workflow_id",
    "stage_width",
    "split",
    "actual_slack_ms",
    "predicted_slack_lcb_ms",
    "predicted_slack_q50_ms",
    "predicted_slack_q90_ms",
    "actual_critical",
    "predicted_critical",
    "evidence_tier",
    "model",
    "dp_size",
    "pp_size",
    "context_length",
}


def _quantile(values: Iterable[float], q: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=float), q))


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_rows(
    path: Path, *, split: str, include_singletons: bool
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        rows = [
            row
            for row in reader
            if row["split"] == split
            and (include_singletons or int(row["stage_width"]) > 1)
        ]
    if not rows:
        raise ValueError(
            f"no rows matched split={split!r}, "
            f"include_singletons={include_singletons}"
        )
    return rows


def _classification_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[float, float]:
    true_positive = int(np.sum(actual & predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return precision, recall


def _summary(rows: list[dict[str, str]], input_path: Path) -> dict[str, object]:
    actual_ms = np.asarray([float(row["actual_slack_ms"]) for row in rows])
    lower_ms = np.asarray(
        [float(row["predicted_slack_lcb_ms"]) for row in rows]
    )
    median_ms = np.asarray(
        [float(row["predicted_slack_q50_ms"]) for row in rows]
    )
    upper_ms = np.asarray(
        [float(row["predicted_slack_q90_ms"]) for row in rows]
    )
    errors_ms = np.abs(median_ms - actual_ms)
    actual_critical = np.asarray(
        [_as_bool(row["actual_critical"]) for row in rows], dtype=bool
    )
    predicted_critical = np.asarray(
        [_as_bool(row["predicted_critical"]) for row in rows], dtype=bool
    )
    critical_precision, critical_recall = _classification_metrics(
        actual_critical, predicted_critical
    )

    scopes = {
        "evidence_tier": sorted({row["evidence_tier"] for row in rows}),
        "model": sorted({row["model"] for row in rows}),
        "dp_size": sorted({int(row["dp_size"]) for row in rows}),
        "pp_size": sorted({int(row["pp_size"]) for row in rows}),
        "context_length": sorted({int(row["context_length"]) for row in rows}),
    }
    return {
        "schema_version": 1,
        "figure": "alg1-slack-prediction-accuracy",
        "source_csv": str(input_path.resolve()),
        "source_status": "archived_pre_kvcache_centric_runtime_baseline",
        "selection": {
            "split": rows[0]["split"],
            "fanout_only": all(int(row["stage_width"]) > 1 for row in rows),
            "examples": len(rows),
            "workflows": len({row["workflow_id"] for row in rows}),
        },
        "scope": scopes,
        "point_accuracy_ms": {
            "mae": float(np.mean(errors_ms)),
            "rmse": float(math.sqrt(float(np.mean(errors_ms**2)))),
            "median_absolute_error": _quantile(errors_ms, 0.50),
            "p90_absolute_error": _quantile(errors_ms, 0.90),
        },
        "uncertainty": {
            "lower_bound_coverage": float(np.mean(actual_ms >= lower_ms)),
            "lcb_to_q90_interval_coverage": float(
                np.mean((actual_ms >= lower_ms) & (actual_ms <= upper_ms))
            ),
            "nonzero_lower_bound_fraction": float(np.mean(lower_ms > 0.0)),
        },
        "critical_classification": {
            "precision": critical_precision,
            "recall": critical_recall,
        },
        "interpretation": (
            "The archived baseline identifies critical agents on this split, "
            "but every conservative Slack lower bound collapses to zero. "
            "Coverage is therefore safe but not yet useful for Algorithm 2."
        ),
    }


def render(
    rows: list[dict[str, str]],
    *,
    output_prefix: Path,
    summary: dict[str, object],
) -> None:
    actual_s = np.asarray([float(row["actual_slack_ms"]) for row in rows]) / 1000.0
    predicted_s = (
        np.asarray([float(row["predicted_slack_q50_ms"]) for row in rows])
        / 1000.0
    )
    errors_s = np.abs(predicted_s - actual_s)
    actual_critical = np.asarray(
        [_as_bool(row["actual_critical"]) for row in rows], dtype=bool
    )

    point = summary["point_accuracy_ms"]
    uncertainty = summary["uncertainty"]
    selection = summary["selection"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.35))

    noncritical = ~actual_critical
    left.scatter(
        actual_s[noncritical],
        predicted_s[noncritical],
        s=48,
        color="#2878B5",
        edgecolor="white",
        linewidth=0.6,
        label="Positive observed Slack",
        zorder=3,
    )
    left.scatter(
        actual_s[actual_critical],
        predicted_s[actual_critical],
        s=48,
        marker="s",
        color="#D9534F",
        edgecolor="white",
        linewidth=0.6,
        label="Observed critical",
        zorder=3,
    )
    upper = max(float(np.max(actual_s)), float(np.max(predicted_s)), 1.0) * 1.06
    left.plot([0.0, upper], [0.0, upper], color="#4A4A4A", linestyle="--", linewidth=1.2)
    left.text(
        upper * 0.60,
        upper * 0.56,
        "ideal: y = x",
        color="#4A4A4A",
        rotation=45,
        ha="center",
        va="center",
        fontsize=9,
    )
    left.set_xlim(0.0, upper)
    left.set_ylim(0.0, upper)
    left.set_aspect("equal", adjustable="box")
    left.set_xlabel("Observed Slack (s)")
    left.set_ylabel("Predicted median Slack (s)")
    left.set_title("(a) Point accuracy on fan-out agents", loc="left")
    left.grid(True, linewidth=0.5, alpha=0.25)
    left.legend(loc="upper left", frameon=False)
    left.text(
        0.98,
        0.04,
        "MAE  {mae:.1f} s\nMedian AE  {med:.1f} s\nP90 AE  {p90:.1f} s".format(
            mae=float(point["mae"]) / 1000.0,
            med=float(point["median_absolute_error"]) / 1000.0,
            p90=float(point["p90_absolute_error"]) / 1000.0,
        ),
        transform=left.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#B8B8B8", "alpha": 0.92},
    )

    sorted_errors = np.sort(errors_s)
    ecdf = np.arange(1, len(sorted_errors) + 1, dtype=float) / len(sorted_errors)
    right.step(sorted_errors, ecdf, where="post", color="#2878B5", linewidth=2.0)
    median_error_s = float(point["median_absolute_error"]) / 1000.0
    p90_error_s = float(point["p90_absolute_error"]) / 1000.0
    right.axvline(median_error_s, color="#2A9D8F", linestyle="--", linewidth=1.4)
    right.axvline(p90_error_s, color="#D9534F", linestyle="--", linewidth=1.4)
    right.scatter(
        [median_error_s, p90_error_s],
        [0.50, 0.90],
        s=38,
        color=["#2A9D8F", "#D9534F"],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    right.annotate(
        f"median {median_error_s:.1f} s",
        (median_error_s, 0.50),
        xytext=(6, -17),
        textcoords="offset points",
        color="#247E73",
    )
    right.annotate(
        f"p90 {p90_error_s:.1f} s",
        (p90_error_s, 0.90),
        xytext=(6, -17),
        textcoords="offset points",
        color="#B23E3A",
    )
    right.set_xlim(left=0.0)
    right.set_ylim(0.0, 1.02)
    right.set_xlabel("Absolute q50 prediction error (s)")
    right.set_ylabel("Fraction of fan-out agents")
    right.set_title("(b) Error distribution and safety utility", loc="left")
    right.grid(True, linewidth=0.5, alpha=0.25)
    right.text(
        0.98,
        0.05,
        "90% LCB coverage  {coverage:.0%}\nNonzero LCB rate  {nonzero:.0%}".format(
            coverage=float(uncertainty["lower_bound_coverage"]),
            nonzero=float(uncertainty["nonzero_lower_bound_fraction"]),
        ),
        transform=right.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#B8B8B8", "alpha": 0.92},
    )

    fig.suptitle(
        "Algorithm 1 Slack prediction accuracy — archived runtime baseline",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.925,
        (
            f"Evaluation-only, workflow-disjoint: {selection['workflows']} workflows, "
            f"{selection['examples']} fan-out agents · SGLSim online measured · "
            "Qwen3-32B · DP4 x PP4 · 32K context"
        ),
        ha="center",
        va="top",
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Singleton stages are excluded. The 100% lower-bound coverage is "
            "vacuous here because every conservative Slack LCB is 0 s."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.90), w_pad=2.0)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--include-singletons", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = _load_rows(
        args.input,
        split=args.split,
        include_singletons=args.include_singletons,
    )
    summary = _summary(rows, args.input)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    render(rows, output_prefix=args.output_prefix, summary=summary)


if __name__ == "__main__":
    main()
