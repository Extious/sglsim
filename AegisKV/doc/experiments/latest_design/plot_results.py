#!/usr/bin/env python3
"""Render the checked-in modeled AegisKV optimization result."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "recompute_only": "Recompute",
    "static_independent_fcfs": "Static independent",
    "aegiskv_hbm_only": "HBM only",
    "aegiskv_dram_only": "DRAM only",
    "aegiskv_latest": "AegisKV latest",
}
COLORS = {
    "recompute_only": "#9CA3AF",
    "static_independent_fcfs": "#64748B",
    "aegiskv_hbm_only": "#60A5FA",
    "aegiskv_dram_only": "#34D399",
    "aegiskv_latest": "#7C3AED",
}


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument(
        "--output", type=Path, default=root / "results" / "optimization_effect.png"
    )
    args = parser.parse_args()
    summary = json.loads(
        (args.results_dir / "summary.json").read_text(encoding="utf-8")
    )
    with (args.results_dir / "raw_failures.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        raw = list(csv.DictReader(handle))
    policies = [row["policy"] for row in summary["policies"]]
    summary_by_policy = {row["policy"]: row for row in summary["policies"]}
    harm_by_policy = defaultdict(list)
    for row in raw:
        harm_by_policy[row["policy"]].append(float(row["visible_harm_ms"]))

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    values = [summary_by_policy[policy]["mean_visible_harm_ms"] for policy in policies]
    bars = axes[0].bar(
        range(len(policies)),
        values,
        color=[COLORS[policy] for policy in policies],
        width=0.72,
    )
    axes[0].set_xticks(
        range(len(policies)),
        [LABELS[policy] for policy in policies],
        rotation=24,
        ha="right",
    )
    axes[0].set_ylabel("Mean Slack-visible recovery harm (ms)")
    axes[0].set_title("Matched failure outcomes")
    axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    latest = summary_by_policy["aegiskv_latest"]
    axes[0].text(
        0.98,
        0.97,
        (
            f"Latest vs recompute: "
            f"-{latest['visible_harm_reduction_vs_recompute_pct']:.1f}%\n"
            f"Latest vs static: "
            f"-{latest['visible_harm_reduction_vs_static_pct']:.1f}%"
        ),
        transform=axes[0].transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )

    for policy in (
        "recompute_only",
        "static_independent_fcfs",
        "aegiskv_latest",
    ):
        values = sorted(harm_by_policy[policy])
        y = [(index + 1) / len(values) for index in range(len(values))]
        axes[1].plot(
            values,
            y,
            label=LABELS[policy],
            color=COLORS[policy],
            linewidth=2.0 if policy == "aegiskv_latest" else 1.6,
        )
    axes[1].set_xlabel("Slack-visible recovery harm (ms)")
    axes[1].set_ylabel("CDF")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_title("Failure-level harm distribution")
    axes[1].legend(frameon=True, loc="lower right")
    figure.suptitle(
        "AegisKV source-owned independent HBM/DRAM (modeled SGLSim campaign)",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
