#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_ablation.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_ablation.png"
)
DEFAULT_BEAMER_OUTPUT = DEFAULT_OUTPUT

ORDER = [
    "AegisKV",
    "no lifecycle",
    "no segment granularity",
    "no restore/recompute",
    "no pressure",
    "no hysteresis",
    "no critical restore",
    "static alpha",
    "size-only backup",
    "no peer locality",
    "no marginal accounting",
]
LABELS = {
    "AegisKV": "AegisKV",
    "no lifecycle": "No life",
    "no segment granularity": "No segment",
    "no restore/recompute": "No R/R",
    "no pressure": "No pressure",
    "no hysteresis": "No hyst.",
    "no critical restore": "No crit.",
    "static alpha": "Static alpha",
    "size-only backup": "Size-only",
    "no peer locality": "No peer loc.",
    "no marginal accounting": "No marginal",
}


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ordered_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_variant = {row.get("variant", ""): row for row in rows}
    missing = [variant for variant in ORDER if variant not in by_variant]
    if missing:
        raise ValueError(f"missing ablation variants: {missing}")
    return [by_variant[variant] for variant in ORDER]


def render(rows: list[dict[str, str]], *, output: Path, beamer_output: Path) -> None:
    ordered = _ordered_rows(rows)
    xs = list(range(len(ordered)))
    labels = [LABELS[row["variant"]] for row in ordered]
    normalized = [_float(row.get("normalized_wct"), 1.0) for row in ordered]
    ci95 = [_float(row.get("normalized_wct_ci95")) for row in ordered]
    host = [_float(row.get("runtime_host_restore_actions")) for row in ordered]
    peer = [_float(row.get("runtime_peer_gpu_actions")) for row in ordered]
    recompute = [_float(row.get("runtime_recompute_actions")) for row in ordered]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 170,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.2),
        gridspec_kw={"width_ratios": [1.3, 1.0]},
    )

    ax = axes[0]
    colors = ["#2f6f9f"] + ["#8fa0ad"] * (len(ordered) - 1)
    markers = ["D"] + ["o"] * (len(ordered) - 1)
    ax.plot(xs, normalized, color="#6b7280", lw=0.9, alpha=0.55, zorder=1)
    for idx, (value, error, color, marker) in enumerate(
        zip(normalized, ci95, colors, markers, strict=True)
    ):
        ax.errorbar(
            idx,
            value,
            yerr=error,
            fmt=marker,
            ms=6.2,
            color="#243447",
            markerfacecolor=color,
            markeredgecolor="#243447",
            ecolor="#243447",
            elinewidth=1.1,
            capsize=3.5,
            zorder=3,
        )
    ax.axhline(1.0, color="#28313f", lw=1.1, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.975, 1.030)
    ax.set_ylabel("normalized workflow WCT")
    ax.set_title("WCT impact, current online sweep")
    ax.grid(True, axis="y", color="#dfe5ec", lw=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(
        xs,
        host,
        label="Host restore",
        color="#f59e0b",
        marker="o",
        lw=1.8,
    )
    ax.plot(
        xs,
        peer,
        label="Peer GPU",
        color="#2563eb",
        marker="s",
        lw=1.8,
    )
    ax.plot(
        xs,
        recompute,
        label="Recompute",
        color="#ef4444",
        marker="D",
        lw=1.8,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("actions")
    ax.set_title("Recovery path changes")
    ax.grid(True, axis="y", color="#dfe5ec", lw=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3)

    fig.suptitle("AegisKV component ablation: early online evidence", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "33 online SGLSim trials; WCT deltas are small at this workload, while recovery-path semantics change visibly.",
        ha="center",
        fontsize=10,
        color="#405062",
    )
    fig.tight_layout(pad=1.2, rect=(0.0, 0.05, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    beamer_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(beamer_output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--beamer-output", type=Path, default=DEFAULT_BEAMER_OUTPUT)
    args = parser.parse_args()
    rows = _read_csv(args.csv)
    render(rows, output=args.output, beamer_output=args.beamer_output)
    print(
        json.dumps(
            {
                "csv": str(args.csv),
                "output": str(args.output),
                "beamer_output": str(args.beamer_output),
                "rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
