#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
CSV_PATH = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_online_alpha_adaptation.csv"
)
OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_online_alpha_adaptation.png"
)
DRAFT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "png"
    / "fig_online_alpha_adaptation.png"
)


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _case_label(row: dict[str, str]) -> str:
    load = "base" if row.get("load_level") == "base load" else "high"
    risk = "low" if _float(row.get("failure_rate_per_window")) < 0.01 else "high"
    return f"{load}\n{risk} risk"


def render() -> None:
    rows = _read_rows()
    critical = [row for row in rows if row.get("request_class") == "Critical agents"]
    noncritical = [
        row for row in rows if row.get("request_class") == "Non-Critical fan-out"
    ]
    order = {
        ("base load", 0.003): 0,
        ("base load", 0.03): 1,
        ("high load", 0.003): 2,
        ("high load", 0.03): 3,
    }
    critical.sort(
        key=lambda row: order.get(
            (row.get("load_level"), _float(row.get("failure_rate_per_window"))),
            99,
        )
    )
    noncritical.sort(
        key=lambda row: order.get(
            (row.get("load_level"), _float(row.get("failure_rate_per_window"))),
            99,
        )
    )
    labels = [_case_label(row) for row in critical]
    x = list(range(len(labels)))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.8,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.65))

    critical_effective = [
        _float(row.get("effective_gpu_replica_alpha")) for row in critical
    ]
    noncritical_effective = [
        _float(row.get("effective_gpu_replica_alpha")) for row in noncritical
    ]
    tuned = [_float(row.get("online_tuned_alpha")) for row in critical]
    axes[0].plot(
        x,
        tuned,
        color="#D97706",
        marker="o",
        lw=1.9,
        label="online tuned alpha",
    )
    axes[0].plot(
        x,
        critical_effective,
        color="#2878B5",
        marker="D",
        lw=1.7,
        label="Critical effective alpha",
    )
    axes[0].plot(
        x,
        noncritical_effective,
        color="#6B7280",
        marker="x",
        lw=1.5,
        label="Non-Critical effective alpha",
    )
    for idx, (crit, noncrit) in enumerate(
        zip(critical_effective, noncritical_effective, strict=True)
    ):
        axes[0].vlines(
            idx,
            ymin=min(crit, noncrit),
            ymax=max(crit, noncrit),
            color="#CBD5E1",
            linewidth=1.0,
            zorder=0,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 0.66)
    axes[0].set_ylabel("alpha")
    axes[0].set_title("(a) MDP alpha applied only to Critical requests")
    axes[0].legend(frameon=False, loc="lower right")

    critical_counts = [_float(row.get("observed_request_rows")) for row in critical]
    noncritical_counts = [
        _float(row.get("observed_request_rows")) for row in noncritical
    ]
    total_counts = [
        critical_counts[idx] + noncritical_counts[idx] for idx in range(len(x))
    ]
    axes[1].plot(
        x,
        critical_counts,
        color="#2D8B57",
        marker="o",
        lw=1.8,
        label="Critical request rows",
    )
    axes[1].plot(
        x,
        noncritical_counts,
        color="#6B7280",
        marker="s",
        lw=1.6,
        label="Non-Critical request rows",
    )
    axes[1].plot(
        x,
        total_counts,
        color="#111827",
        marker="D",
        lw=1.5,
        linestyle="--",
        label="total rows",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("completed request rows")
    axes[1].set_title("(b) 4 online SGLSim runs, DP8/100K context")
    axes[1].legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.grid(True, axis="y", color="#E5E7EB", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Online alpha adaptation evidence: MDP policy table + bounded online update",
        y=1.03,
        fontsize=10.5,
    )
    fig.tight_layout(pad=1.0, w_pad=1.6)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DRAFT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(DRAFT_OUTPUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
