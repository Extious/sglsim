#!/usr/bin/env python3
"""Plot audited S3/S6/S7 preliminary extension artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent

AEGIS = "#0072B2"
NATIVE = "#4D4D4D"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_s3(input_dir: Path, aggregate: list[dict[str, str]]) -> Path:
    rate = 5.952
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    for ax, axis, xlabel in (
        (axes[0], "scale", "System DP (GPU shards)"),
        (axes[1], "context", "Context tokens"),
    ):
        selected = [
            row for row in aggregate
            if row["axis"] == axis and row["policy"] == "AegisKV"
            and abs(float(row["failure_rate"]) - rate) < 1e-9
        ]
        selected.sort(key=lambda row: float(row["value"]))
        x = np.asarray([float(row["value"]) for row in selected])
        ratio = np.asarray([float(row["normalized_to_native"]) for row in selected])
        ratio_low = np.asarray([float(row["normalized_ci95_low"]) for row in selected])
        ratio_high = np.asarray([float(row["normalized_ci95_high"]) for row in selected])
        y = (1.0 - ratio) * 100.0
        low = (1.0 - ratio_high) * 100.0
        high = (1.0 - ratio_low) * 100.0
        ax.errorbar(
            x, y, yerr=[y - low, high - y], marker="o", color=AEGIS,
            capsize=4, lw=2.0, markersize=6.5,
        )
        ax.axhline(0.0, color="black", lw=1, ls="--")
        ax.set_xticks(x, [f"{int(value / 1024)}K" if axis == "context" else f"{int(value)}" for value in x])
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.18, linewidth=0.8)
        ax.set_title(
            "GPU scale (16K context)" if axis == "scale" else "Context length (system-DP=64)",
            fontsize=11, pad=7,
        )
    shared_limit = max(
        max(abs(value) for axis in axes for value in axis.get_ylim()),
        0.02,
    )
    for axis in axes:
        axis.set_ylim(-shared_limit, shared_limit)
    axes[0].set_ylabel("Mean WCT reduction vs modeled Native (%)")
    fig.suptitle("S3. Scale and context do not create a material center-load effect", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.82, bottom=0.20, wspace=0.20)
    fig.text(
        0.5, 0.03,
        "Positive is better; all absolute changes < 0.02%; Qwen3-32B target series was not run; paired 3-seed 95% CI",
        ha="center", fontsize=8, color="#555555",
    )
    base = input_dir / "figures" / "s3_scale_context_preliminary"
    save(fig, base)
    return base


def mean_ci(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = np.mean(rng.choice(array, size=(10000, len(array)), replace=True), axis=1)
    return float(np.mean(array)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def plot_s6(input_dir: Path, summary: dict) -> Path:
    cells = summary["cells"]
    reuse_values = sorted({float(cell["extension_spec"]["reuse"]) for cell in cells})
    base_wct = {
        int(cell["seed"]): next(
            float(row["hazard_integrated_mean_wct_ms"])
            for row in cell["rate_rows"]
            if abs(float(row["failure_rate_per_1000_gpu_shard_hour"]) - 5.952) < 1e-9
        )
        for cell in cells if float(cell["extension_spec"]["reuse"]) == 0.0
    }
    metrics: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name in (
            "wct", "hbm", "dram", "copy_amp", "hbm_evictions",
            "dram_evictions", "recovery_hit",
        )
    }
    for index, reuse in enumerate(reuse_values):
        selected = [cell for cell in cells if float(cell["extension_spec"]["reuse"]) == reuse]
        wct_ratio = []
        for cell in selected:
            value = next(
                float(row["hazard_integrated_mean_wct_ms"])
                for row in cell["rate_rows"]
                if abs(float(row["failure_rate_per_1000_gpu_shard_hour"]) - 5.952) < 1e-9
            )
            wct_ratio.append(value / base_wct[int(cell["seed"])])
        values = {
            "wct": wct_ratio,
            "hbm": [float(cell["mechanism"]["occupancy_time_mean_gib_per_gpu"]["HBM"]) for cell in selected],
            "dram": [float(cell["mechanism"]["occupancy_time_mean_gib_per_gpu"]["DRAM"]) for cell in selected],
            "copy_amp": [float(cell["mechanism"]["copy_amplification"]) for cell in selected],
            "hbm_evictions": [
                float(cell["mechanism"]["tier_churn"]["HBM"]["eviction_count"])
                / (float(cell["workload"]["request_count"]) / 1000.0)
                for cell in selected
            ],
            "dram_evictions": [
                float(cell["mechanism"]["tier_churn"]["DRAM"]["eviction_count"])
                / (float(cell["workload"]["request_count"]) / 1000.0)
                for cell in selected
            ],
            "recovery_hit": [float(cell["mechanism"]["recovery_hit_fraction"]) for cell in selected],
        }
        for offset, (name, rows_) in enumerate(values.items()):
            metrics[name].append(mean_ci(rows_, 20260826 + index * 17 + offset))

    x = np.asarray(reuse_values) * 100.0
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharex=True)
    # Top row deliberately mirrors the old target figure: reuse benefit/copy
    # cost on the left and per-tier evictions on the right.
    data = metrics["recovery_hit"]
    mean = np.asarray([row[0] for row in data]); low = np.asarray([row[1] for row in data]); high = np.asarray([row[2] for row in data])
    axes[0, 0].errorbar(
        x, mean * 100.0,
        yerr=[(mean - low) * 100.0, (high - mean) * 100.0],
        marker="o", capsize=4, color="#009E73", linewidth=2.0,
        label="Recovery hit rate",
    )
    axes[0, 0].set_ylim(0.0, 100.0)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[0, 0].set_ylabel("Recovery hit rate", color="#009E73")
    axes[0, 0].tick_params(axis="y", colors="#009E73")
    copy_ax = axes[0, 0].twinx()
    data = metrics["copy_amp"]
    cmean = np.asarray([row[0] for row in data]); clow = np.asarray([row[1] for row in data]); chigh = np.asarray([row[2] for row in data])
    copy_ax.errorbar(
        x, cmean, yerr=[cmean - clow, chigh - cmean], marker="s",
        capsize=4, color="#D55E00", linewidth=1.8, linestyle="--",
        label="Copy amplification",
    )
    copy_ax.set_ylabel("Copied / fresh KV bytes", color="#D55E00")
    copy_ax.tick_params(axis="y", colors="#D55E00")
    handles_left, labels_left = axes[0, 0].get_legend_handles_labels()
    handles_right, labels_right = copy_ax.get_legend_handles_labels()
    axes[0, 0].legend(handles_left + handles_right, labels_left + labels_right, fontsize=7.5, loc="upper center")
    axes[0, 0].set_title("Target-aligned reuse benefit and copy cost")

    width = 8.5
    for offset, (name, color, label) in enumerate((
        ("hbm_evictions", "#0072B2", "HBM evictions"),
        ("dram_evictions", "#E69F00", "DRAM evictions"),
    )):
        data = metrics[name]
        emean = np.asarray([row[0] for row in data]); elow = np.asarray([row[1] for row in data]); ehigh = np.asarray([row[2] for row in data])
        xpos = x + (-width / 2 if offset == 0 else width / 2)
        axes[0, 1].bar(xpos, emean, width=width, color=color, alpha=0.82, label=label)
        axes[0, 1].errorbar(xpos, emean, yerr=[emean - elow, ehigh - emean], fmt="none", ecolor="#333333", capsize=3, linewidth=0.9)
    axes[0, 1].set_ylabel("Evictions / 1K requests")
    axes[0, 1].set_title("Target-aligned protection churn")
    axes[0, 1].legend(fontsize=7.5)

    # Bottom row retains the extra evidence requested after the target sketch:
    # audited byte-ms occupancy and the within-Aegis reuse WCT effect.
    for name, color, label in (
        ("hbm", "#E69F00", "Peer-HBM"), ("dram", "#CC79A7", "DRAM"),
    ):
        data = metrics[name]
        mean = np.asarray([row[0] for row in data]); low = np.asarray([row[1] for row in data]); high = np.asarray([row[2] for row in data])
        axes[1, 0].errorbar(x, mean, yerr=[mean - low, high - mean], marker="o", capsize=4, color=color, linewidth=2.0, label=label)
        for xpos, ypos in zip(x, mean):
            axes[1, 0].annotate(f"{ypos:.1f}", (xpos, ypos), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.2, color=color)
    axes[1, 0].set_ylabel("Time-mean occupancy (GiB/GPU)")
    axes[1, 0].set_title("Audited capacity integral")
    axes[1, 0].legend(fontsize=8)

    data = metrics["wct"]
    ratio = np.asarray([row[0] for row in data]); ratio_low = np.asarray([row[1] for row in data]); ratio_high = np.asarray([row[2] for row in data])
    mean = (1.0 - ratio) * 100.0
    low = (1.0 - ratio_high) * 100.0
    high = (1.0 - ratio_low) * 100.0
    axes[1, 1].errorbar(x, mean, yerr=[mean - low, high - mean], marker="o", capsize=4, color=AEGIS, linewidth=2.0)
    axes[1, 1].axhline(0.0, color="black", ls="--", lw=1)
    axes[1, 1].set_ylabel("WCT reduction vs reuse=0 (%)")
    axes[1, 1].set_title("Within-AegisKV reuse benefit")
    for xpos, ypos in zip(x, mean):
        axes[1, 1].annotate(f"{ypos:.1f}%", (xpos, ypos), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.2, color=AEGIS)
    for ax in axes.flat:
        ax.set_xticks(x, [f"{value:g}%" for value in x])
        ax.set_xlabel("Modeled source-prefix reuse")
        ax.grid(alpha=0.16, linewidth=0.8)
    fig.suptitle("S6. Reuse benefit and protection cost - observed", fontsize=14)
    fig.subplots_adjust(top=0.88, bottom=0.13, hspace=0.34, wspace=0.42)
    fig.text(
        0.5, 0.01,
        "Top row matches the old target question; bottom row adds audited used_bytes occupancy and WCT; 3-seed bootstrap 95% CI",
        ha="center", fontsize=8, color="#555555",
    )
    base = input_dir / "figures" / "s6_reuse_occupancy_preliminary"
    save(fig, base)
    return base


def plot_s7(input_dir: Path, aggregate: list[dict[str, str]], summary: dict) -> Path:
    topology_order = ["same+local", "same+remote", "cross+local", "cross+remote"]
    topology_rows = {
        row["value"]: row for row in aggregate
        if row["axis"] == "topology" and row["policy"] == "AegisKV"
        and abs(float(row["failure_rate"]) - 5.952) < 1e-9
    }
    ratio = np.asarray([float(topology_rows[name]["normalized_to_native"]) for name in topology_order])
    ratio_low = np.asarray([float(topology_rows[name]["normalized_ci95_low"]) for name in topology_order])
    ratio_high = np.asarray([float(topology_rows[name]["normalized_ci95_high"]) for name in topology_order])
    values = (1.0 - ratio) * 100.0
    low = (1.0 - ratio_high) * 100.0
    high = (1.0 - ratio_low) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    route_labels = ["Same HBM\nLocal DRAM", "Same HBM\nRemote DRAM", "Cross HBM\nLocal DRAM", "Cross HBM\nRemote DRAM"]
    y = np.arange(len(topology_order))
    axes[0].barh(
        y, values, xerr=[values - low, high - values], capsize=4,
        color=["#56B4E9", "#0072B2", "#E69F00", "#CC79A7"],
        edgecolor="black", linewidth=0.5, height=0.62,
    )
    axes[0].axvline(0.0, color="black", lw=1, ls="--")
    axes[0].set_yticks(y, route_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean WCT reduction vs modeled Native (%)")
    axes[0].set_title("Topology at 60 s repair")
    axes[0].text(
        0.02, 0.02, "Old target's 1-50 GB/s sweep was not executed",
        transform=axes[0].transAxes, fontsize=7.5, color="#991B1B",
        ha="left", va="bottom",
    )
    axes[0].grid(axis="x", alpha=0.18, linewidth=0.8)
    for xpos, ypos in zip(values, y):
        axes[0].annotate(
            f"{xpos:+.3f}%", (xpos, ypos), xytext=(6, 0),
            textcoords="offset points", ha="left",
            va="center", fontsize=7.5,
        )

    repair_cells = [
        cell for cell in summary["cells"]
        if any(axis["axis"] == "repair" for axis in cell["extension_spec"]["axes"])
    ]
    delays = [30000.0, 60000.0]
    policies = ["Native Mooncake", "AegisKV"]
    for offset, (policy, color) in enumerate(zip(policies, (NATIVE, AEGIS))):
        ci = []
        for index, delay in enumerate(delays):
            vals = [
                float(cell["fault_conditioned_wct_mean_ms"])
                for cell in repair_cells
                if cell["policy"] == policy and float(cell["extension_spec"]["repair_ms"]) == delay
            ]
            ci.append(mean_ci(vals, 20260826 + index + offset * 11))
        mean = np.asarray([row[0] for row in ci]); low_ = np.asarray([row[1] for row in ci]); high_ = np.asarray([row[2] for row in ci])
        positions = np.asarray(delays) / 1000.0
        axes[1].errorbar(
            positions, mean, yerr=[mean - low_, high_ - mean],
            marker="o" if offset else "s", capsize=4, color=color,
            linewidth=2.0, markersize=6.5,
            label="Native (modeled)" if policy == "Native Mooncake" else policy,
        )
        for xpos, ypos in zip(positions, mean):
            axes[1].annotate(
                f"{ypos:.0f}", (xpos, ypos), xytext=(0, 8 if offset else -13),
                textcoords="offset points", ha="center", fontsize=7.5, color=color,
            )
    axes[1].set_xticks([30, 60], ["30 s", "60 s"])
    axes[1].set_xlabel("Replacement/repair delay")
    axes[1].set_ylabel("Fault-conditioned whole-workload mean WCT (ms)")
    axes[1].set_title("Executed repair-delay sensitivity")
    axes[1].grid(alpha=0.18, linewidth=0.8)
    axes[1].legend(fontsize=8)
    fig.suptitle("S7. Network/repair target - topology proxy and observed repair sweep", fontsize=14)
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.30)
    fig.text(
        0.5, 0.035,
        "Positive is better; Peer-HBM same/cross = 50/10 GB/s; DRAM local/remote = 16/10 GB/s; paired 3-seed 95% CI",
        ha="center", fontsize=8, color="#555555",
    )
    base = input_dir / "figures" / "s7_topology_repair_preliminary"
    save(fig, base)
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", choices=("s3", "s6", "s7"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.input_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("selection") != args.selection.upper():
        raise ValueError("extension plot requires the matching passed audit")
    aggregate = rows(args.input_dir / "aggregate_ci.csv")
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    if args.selection == "s3":
        base = plot_s3(args.input_dir, aggregate)
    elif args.selection == "s6":
        base = plot_s6(args.input_dir, summary)
    elif args.selection == "s7":
        base = plot_s7(args.input_dir, aggregate, summary)
    else:
        raise NotImplementedError(args.selection)
    manifest = {
        "status": "PRELIMINARY_FIGURE_COMPLETE",
        "selection": args.selection.upper(),
        "source_audit": str(audit_path),
        "source_audit_sha256": sha256(audit_path),
        "source_summary_sha256": audit["summary_sha256"],
        "pdf": str(base.with_suffix(".pdf")),
        "pdf_sha256": sha256(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
        "png_sha256": sha256(base.with_suffix(".png")),
        "evidence_boundary": audit["evidence_boundary"],
    }
    (base.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
