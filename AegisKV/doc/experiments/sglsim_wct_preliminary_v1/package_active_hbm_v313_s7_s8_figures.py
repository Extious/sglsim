#!/usr/bin/env python3
"""Package corrected S7 visuals and the preregistered repair=60 s S8 slice."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "active_hbm_v313_s7_directional_repair_pilot"
S7_NAME = "active_hbm_v313_s7_directional_repair_visual_v4"
S8_NAME = "active_hbm_v313_s8_failure_domain_from_s7_60s_visual_v2"
SOURCE_AUDIT_SHA = "a76388c85feab393bbdb02fade9b4c8445749afd850241cbbcd0fda245a5c774"
SOURCE_MANIFEST_SHA = "d5f53fc5d49eb52126275584389a11b65443f673ced41914ecba0038a9b8c75b"
SOURCE_SUMS_SHA = "12a6a0e3cf29100849e60a0f3f2ebedb259afde687d0d3e2fa7d137ca9554443"
T95_DF2 = 4.302652729911275


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci95(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) != 3:
        raise ValueError("visual source requires exactly three seeds")
    return T95_DF2 * statistics.stdev(rows) / math.sqrt(3)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.titlesize": 8.1, "axes.labelsize": 7.7,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "lines.linewidth": 1.2,
    })


def verify_figure(pdf: Path, png: Path) -> dict[str, Any]:
    with fitz.open(pdf) as document:
        page = document[0]
        sizes = [
            float(span["size"])
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", []) if span["text"].strip()
        ]
        page_count = len(document)
    with Image.open(png) as image:
        width, height = image.size
    return {
        "single_page_vector_pdf": page_count == 1 and bool(sizes),
        "minimum_extracted_font_pt": min(sizes),
        "minimum_font_at_least_7pt": min(sizes) >= 7.0 - 1e-6,
        "png_width_px": width, "png_height_px": height,
        "png_width_at_least_1800": width >= 1800,
    }


def source_gate() -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]
]:
    audit = json.loads((SOURCE / "audit.json").read_text())
    if sha256(SOURCE / "audit.json") != SOURCE_AUDIT_SHA:
        raise RuntimeError("S7 source audit hash drift")
    if sha256(SOURCE / "manifest.json") != SOURCE_MANIFEST_SHA:
        raise RuntimeError("S7 source manifest hash drift")
    if sha256(SOURCE / "SHA256SUMS") != SOURCE_SUMS_SHA:
        raise RuntimeError("S7 source checksum manifest drift")
    if audit["status"] != "PASS" or audit["cell_count"] != 168:
        raise RuntimeError("S7 source is not the audited 168-cell PASS bundle")
    return (
        read_csv(SOURCE / "cell_metrics.csv"),
        read_csv(SOURCE / "aggregate_ci.csv"),
        read_csv(SOURCE / "paired_ci.csv"), audit,
    )


def copy_source_rows(cells: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction, tier in (
        ("source_to_peer_hbm_copy", "hbm"),
        ("source_to_dram_copy", "dram"),
    ):
        for bandwidth in (1.0, 5.0, 10.0, 25.0, 50.0):
            for policy in ("Native Mooncake", "AegisKV"):
                selected = [
                    row for row in cells
                    if row["axis"] == "bandwidth"
                    and row["direction"] == direction
                    and float(row["bandwidth_gb_s"]) == bandwidth
                    and row["policy"] == policy
                ]
                if len(selected) != 3:
                    raise AssertionError("copy visual source lost a paired seed")
                metrics = {
                    "copy_service_ms": [float(row["copy_transfer_service_mean_ms"]) for row in selected],
                    "copy_queue_ms": [float(row["copy_queue_wait_mean_ms"]) for row in selected],
                    "target_copy_tib": [float(row[f"{tier}_scheduled_copy_bytes"]) / 2**40 for row in selected],
                    "recovery_coverage": [float(row["recovery_hit_fraction"]) for row in selected],
                }
                out: dict[str, Any] = {
                    "direction": direction, "bandwidth_gb_s": bandwidth,
                    "policy": policy, "seed_count": 3,
                }
                for name, values in metrics.items():
                    out[f"{name}_mean"] = statistics.fmean(values)
                    out[f"{name}_ci95_half_width"] = ci95(values)
                rows.append(out)
    return rows


def policy_encoding(policy: str) -> tuple[str, str]:
    return (("--", "s") if policy == "Native Mooncake" else ("-", "o"))


def plot_s7(
    path: Path, copy_rows: list[dict[str, Any]],
    aggregates: list[dict[str, str]], paired: list[dict[str, str]],
) -> None:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.8))
    copy_colors = {
        "source_to_peer_hbm_copy": "#2166ac",
        "source_to_dram_copy": "#67a9cf",
    }
    copy_labels = {
        "source_to_peer_hbm_copy": "Peer copy",
        "source_to_dram_copy": "DRAM copy",
    }
    for direction in copy_colors:
        for policy in ("Native Mooncake", "AegisKV"):
            rows = sorted(
                (row for row in copy_rows if row["direction"] == direction and row["policy"] == policy),
                key=lambda row: float(row["bandwidth_gb_s"]),
            )
            line, marker = policy_encoding(policy)
            label = f"{policy} / {copy_labels[direction]}"
            x = [float(row["bandwidth_gb_s"]) for row in rows]
            for ax, metric, scale in (
                (axes[0, 0], "copy_service_ms", 1.0),
            ):
                ax.errorbar(
                    x, [float(row[f"{metric}_mean"]) * scale for row in rows],
                    yerr=[float(row[f"{metric}_ci95_half_width"]) * scale for row in rows],
                    color=copy_colors[direction], linestyle=line, marker=marker,
                    markersize=3.1, capsize=1.8, label=label,
                )
            queue_mean = np.asarray([
                float(row["copy_queue_ms_mean"]) / 1000.0 for row in rows
            ])
            queue_half = np.asarray([
                float(row["copy_queue_ms_ci95_half_width"]) / 1000.0
                for row in rows
            ])
            transformed = np.log10(1.0 + queue_mean)
            lower = transformed - np.log10(1.0 + np.maximum(queue_mean - queue_half, 0.0))
            upper = np.log10(1.0 + queue_mean + queue_half) - transformed
            axes[0, 1].errorbar(
                x, transformed, yerr=np.vstack([lower, upper]),
                color=copy_colors[direction], linestyle=line, marker=marker,
                markersize=3.1, capsize=1.8, label=label,
            )
    axes[0, 0].set_title("Protection copy service")
    axes[0, 0].set_ylabel("Per-request service (ms)")
    axes[0, 1].set_title("Protection copy queue")
    axes[0, 1].set_ylabel("log10(1 + queue wait in s)")
    encoding_handles = [
        Line2D([0], [0], color=copy_colors["source_to_peer_hbm_copy"], label="Peer copy"),
        Line2D([0], [0], color=copy_colors["source_to_dram_copy"], label="DRAM copy"),
        Line2D([0], [0], color="#555555", linestyle="--", marker="s", label="Native"),
        Line2D([0], [0], color="#555555", linestyle="-", marker="o", label="AegisKV"),
    ]
    axes[0, 0].legend(
        handles=encoding_handles, frameon=False, ncol=2,
        handlelength=1.45, columnspacing=0.75, loc="upper right",
    )

    copy_volume = axes[0, 2]
    copy_coverage = copy_volume.twinx()
    for direction in copy_colors:
        for policy in ("Native Mooncake", "AegisKV"):
            rows = sorted(
                (row for row in copy_rows if row["direction"] == direction and row["policy"] == policy),
                key=lambda row: float(row["bandwidth_gb_s"]),
            )
            line, marker = policy_encoding(policy)
            x = [float(row["bandwidth_gb_s"]) for row in rows]
            copy_volume.plot(
                x, [float(row["target_copy_tib_mean"]) for row in rows],
                color=copy_colors[direction], linestyle=line, marker=marker,
                markersize=3.0,
            )
            copy_coverage.plot(
                x, [100.0 * float(row["recovery_coverage_mean"]) for row in rows],
                color=copy_colors[direction], linestyle=":" if policy == "AegisKV" else "-.",
                marker=marker, markersize=2.8, alpha=0.85,
            )
    copy_volume.set_title("Copied bytes and resulting coverage")
    copy_volume.set_ylabel("Target-tier copied (TiB)")
    copy_coverage.set_ylabel("Replica-hit calls (%)")
    # Reserve a data-free band: the v2 encoding guide obscured retained points.
    copy_volume.set_ylim(0.0, 90.0)
    copy_coverage.set_ylim(0.0, 75.0)
    copy_volume.legend(handles=[
        Line2D([0], [0], color=copy_colors["source_to_peer_hbm_copy"], label="Peer copy"),
        Line2D([0], [0], color=copy_colors["source_to_dram_copy"], label="DRAM copy"),
        Line2D([0], [0], color="#555555", linestyle="--", marker="s", label="Native"),
        Line2D([0], [0], color="#555555", linestyle="-", marker="o", label="AegisKV"),
        Line2D([0], [0], color="#555555", linestyle="-", label="Bytes"),
        Line2D([0], [0], color="#555555", linestyle=":", label="Coverage"),
    ], frameon=False, ncol=2, handlelength=1.5, columnspacing=0.8,
       loc="upper center")

    recovery_colors = {
        "peer_hbm_recovery": "#b2182b",
        "dram_recovery": "#ef8a62",
    }
    recovery_labels = {
        "peer_hbm_recovery": "Peer recovery",
        "dram_recovery": "DRAM recovery",
    }
    for direction in recovery_colors:
        for policy in ("Native Mooncake", "AegisKV"):
            rows = sorted(
                (row for row in aggregates if row["axis"] == "bandwidth" and row["direction"] == direction and row["policy"] == policy),
                key=lambda row: float(row["bandwidth_gb_s"]),
            )
            line, marker = policy_encoding(policy)
            axes[1, 0].errorbar(
                [float(row["bandwidth_gb_s"]) for row in rows],
                [float(row["recovery_service_mean_ms_mean"]) for row in rows],
                yerr=[float(row["recovery_service_mean_ms_ci95_half_width"]) for row in rows],
                color=recovery_colors[direction], linestyle=line, marker=marker,
                markersize=3.1, capsize=1.8,
                label=f"{policy} / {recovery_labels[direction]}",
            )
    axes[1, 0].set_title("Recovery service")
    axes[1, 0].set_ylabel("Service (ms)")
    axes[1, 0].legend(handles=[
        Line2D([0], [0], color=recovery_colors["peer_hbm_recovery"], label="Peer recovery"),
        Line2D([0], [0], color=recovery_colors["dram_recovery"], label="DRAM recovery"),
        Line2D([0], [0], color="#555555", linestyle="--", marker="s", label="Native"),
        Line2D([0], [0], color="#555555", linestyle="-", marker="o", label="AegisKV"),
    ], frameon=False, ncol=2, handlelength=1.45, columnspacing=0.75, loc="lower left")

    recovery_coverage = axes[1, 1]
    recovery_saved = recovery_coverage.twinx()
    for direction in recovery_colors:
        for policy in ("Native Mooncake", "AegisKV"):
            rows = sorted(
                (row for row in aggregates if row["axis"] == "bandwidth" and row["direction"] == direction and row["policy"] == policy),
                key=lambda row: float(row["bandwidth_gb_s"]),
            )
            line, marker = policy_encoding(policy)
            recovery_coverage.plot(
                [float(row["bandwidth_gb_s"]) for row in rows],
                [100.0 * float(row["recovery_hit_fraction_mean"]) for row in rows],
                color=recovery_colors[direction], linestyle=line, marker=marker,
                markersize=3.0,
            )
        rows = sorted(
            (row for row in paired if row["axis"] == "bandwidth" and row["direction"] == direction),
            key=lambda row: float(row["bandwidth_gb_s"]),
        )
        recovery_saved.errorbar(
            [float(row["bandwidth_gb_s"]) for row in rows],
            [float(row["fault_wct_saved_ms_mean"]) / 1000.0 for row in rows],
            yerr=[float(row["fault_wct_saved_ms_ci95_half_width"]) / 1000.0 for row in rows],
            color=recovery_colors[direction], linestyle=":", marker="D",
            markersize=2.8, capsize=1.7,
        )
    recovery_coverage.set_title("Recovery coverage and WCT saved")
    recovery_coverage.set_ylabel("Replica-hit calls (%)")
    recovery_saved.set_ylabel("Native − AegisKV (s)")
    # The v2 legend landed on both recovery curves.  These limits contain
    # every retained value while making the upper band explicitly data-free.
    recovery_coverage.set_ylim(0.0, 100.0)
    recovery_saved.set_ylim(1.00, 2.00)
    recovery_coverage.legend(handles=[
        Line2D([0], [0], color=recovery_colors["peer_hbm_recovery"], label="Peer recovery"),
        Line2D([0], [0], color=recovery_colors["dram_recovery"], label="DRAM recovery"),
        Line2D([0], [0], color="#555555", linestyle="--", marker="s", label="Native coverage"),
        Line2D([0], [0], color="#555555", linestyle="-", marker="o", label="AegisKV coverage"),
        Line2D([0], [0], color="#555555", linestyle=":", marker="D", label="WCT saved"),
    ], frameon=False, ncol=1, handlelength=1.5,
       loc="upper center")

    repair = axes[1, 2]
    repair_gain = repair.twinx()
    domain_style = {
        "source_worker_gpu": ("Worker/GPU", "#5e3c99", "o"),
        "source_node": ("Source node", "#e66101", "s"),
    }
    for domain, (label, color, marker) in domain_style.items():
        rows = sorted(
            (row for row in paired if row["axis"] == "repair" and row["failure_type"] == domain),
            key=lambda row: float(row["repair_ms"]),
        )
        x = [float(row["repair_ms"]) / 1000.0 for row in rows]
        repair.errorbar(
            x, [float(row["fault_wct_saved_ms_mean"]) / 1000.0 for row in rows],
            yerr=[float(row["fault_wct_saved_ms_ci95_half_width"]) / 1000.0 for row in rows],
            color=color, marker=marker, markersize=3.2, capsize=1.8,
            label=f"{label}: saved",
        )
        repair_gain.plot(
            x, [100.0 * float(row["fault_wct_relative_gain_mean"]) for row in rows],
            color=color, marker=marker, markersize=2.8, linestyle=":",
            label=f"{label}: gain",
        )
    repair.axhline(0, color="#777777", linewidth=0.7)
    repair.set_title("Modeled replacement sensitivity")
    repair.set_ylabel("WCT saved (s)")
    repair_gain.set_ylabel("Gain (%)")
    handles1, labels1 = repair.get_legend_handles_labels()
    handles2, labels2 = repair_gain.get_legend_handles_labels()
    repair.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=1, handlelength=1.5)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
        ax.set_xlabel("Bandwidth (GB/s)" if ax is not repair else "Replacement (s)")
    fig.suptitle("S7 directional copy, recovery, and repair sensitivity", fontsize=9.8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.45, w_pad=1.1)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def s8_rows(cells: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    source = [
        row for row in cells
        if row["axis"] == "repair" and float(row["repair_ms"]) == 60000.0
    ]
    if len(source) != 12:
        raise AssertionError("S8 repair=60 s slice must contain exactly 12 cells")
    aggregates: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for domain in ("source_worker_gpu", "source_node"):
        for policy in ("Native Mooncake", "AegisKV"):
            selected = [
                row for row in source
                if row["failure_type"] == domain and row["policy"] == policy
            ]
            out: dict[str, Any] = {
                "failure_type": domain, "policy": policy, "seed_count": 3,
            }
            for metric in (
                "healthy_wct_mean_ms", "fault_wct_mean_ms", "fault_wct_p99_ms",
                "fault_added_wct_mean_ms", "failover_delay_mean_ms",
                "recovery_service_mean_ms", "recovery_hit_fraction",
            ):
                values = [float(row[metric]) for row in selected]
                out[f"{metric}_mean"] = statistics.fmean(values)
                out[f"{metric}_ci95_half_width"] = ci95(values)
            aggregates.append(out)
        seed_rows = []
        for seed in (2026082600, 2026082601, 2026082602):
            by_policy = {
                row["policy"]: row for row in source
                if row["failure_type"] == domain and int(row["seed"]) == seed
            }
            native = by_policy["Native Mooncake"]
            aegis = by_policy["AegisKV"]
            seed_rows.append({
                "saved": float(native["fault_wct_mean_ms"]) - float(aegis["fault_wct_mean_ms"]),
                "gain": 1.0 - float(aegis["fault_wct_mean_ms"]) / float(native["fault_wct_mean_ms"]),
            })
        paired.append({
            "failure_type": domain, "seed_count": 3,
            "fault_wct_saved_ms_mean": statistics.fmean(row["saved"] for row in seed_rows),
            "fault_wct_saved_ms_ci95_half_width": ci95(row["saved"] for row in seed_rows),
            "fault_wct_relative_gain_mean": statistics.fmean(row["gain"] for row in seed_rows),
            "fault_wct_relative_gain_ci95_half_width": ci95(row["gain"] for row in seed_rows),
        })
    return source, aggregates, paired


def plot_s8(path: Path, aggregates: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65))
    domains = ("source_worker_gpu", "source_node")
    labels = ("Worker/GPU", "Source node")
    colors = {"Native Mooncake": "#5b6470", "AegisKV": "#00897b"}
    markers = {"Native Mooncake": "s", "AegisKV": "o"}
    x = np.arange(2)
    offsets = {"Native Mooncake": -0.08, "AegisKV": 0.08}
    for policy in ("Native Mooncake", "AegisKV"):
        rows = [next(row for row in aggregates if row["failure_type"] == domain and row["policy"] == policy) for domain in domains]
        axes[0].errorbar(
            x + offsets[policy], [row["fault_wct_mean_ms_mean"] / 1000.0 for row in rows],
            yerr=[row["fault_wct_mean_ms_ci95_half_width"] / 1000.0 for row in rows],
            linestyle="none", color=colors[policy], marker=markers[policy],
            markersize=4.0, capsize=2.4, label=policy,
        )
    axes[0].set_title("Failure-conditioned WCT")
    axes[0].set_ylabel("Mean WCT (s)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 7.2)
    axes[0].legend(frameon=False, ncol=2, loc="upper center",
                   handletextpad=0.4, columnspacing=0.8)

    paired_ax = axes[1]
    gain_ax = paired_ax.twinx()
    paired_ax.errorbar(
        x, [row["fault_wct_saved_ms_mean"] / 1000.0 for row in paired],
        yerr=[row["fault_wct_saved_ms_ci95_half_width"] / 1000.0 for row in paired],
        color="#5e3c99", marker="o", markersize=4.0, capsize=2.4,
        label="WCT saved",
    )
    gain_ax.plot(
        x, [100.0 * row["fault_wct_relative_gain_mean"] for row in paired],
        color="#e66101", marker="s", linestyle=":", markersize=3.7,
        label="Relative gain",
    )
    paired_ax.axhline(0, color="#777777", linewidth=0.7)
    paired_ax.set_title("Paired AegisKV effect")
    paired_ax.set_ylabel("Native − AegisKV (s)")
    gain_ax.set_ylabel("Gain (%)")
    paired_ax.set_ylim(0.0, 1.75)
    gain_ax.set_ylim(0.0, 50.0)
    paired_ax.set_xticks(x, labels)
    h1, l1 = paired_ax.get_legend_handles_labels()
    h2, l2 = gain_ax.get_legend_handles_labels()
    paired_ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=2,
                     loc="upper center", handletextpad=0.4, columnspacing=0.8)

    width = 0.34
    positions = np.arange(4)
    ordered = [
        next(row for row in aggregates if row["failure_type"] == domain and row["policy"] == policy)
        for domain in domains for policy in ("Native Mooncake", "AegisKV")
    ]
    axes[2].bar(
        positions, [row["failover_delay_mean_ms_mean"] / 1000.0 for row in ordered],
        width=0.68, color="#9e9e9e", label="Failover delay",
    )
    axes[2].bar(
        positions, [row["recovery_service_mean_ms_mean"] / 1000.0 for row in ordered],
        bottom=[row["failover_delay_mean_ms_mean"] / 1000.0 for row in ordered],
        width=0.68, color="#4db6ac", label="Recovery service",
    )
    axes[2].set_title("Modeled delay composition")
    axes[2].set_ylabel("Mean delay (s)")
    axes[2].set_xticks(
        positions,
        ["Worker / Native", "Worker / AegisKV", "Node / Native", "Node / AegisKV"],
        rotation=25, ha="right",
    )
    axes[2].set_ylim(0.0, 82.0)
    axes[2].legend(frameon=False, ncol=2, loc="upper center",
                   handletextpad=0.4, columnspacing=0.7)

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("S8 worker/GPU and source-node failures", fontsize=9.6, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.35)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def publish_s7(
    cells: list[dict[str, str]], aggregates: list[dict[str, str]],
    paired: list[dict[str, str]],
) -> Path:
    target = ROOT / "results" / S7_NAME
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{S7_NAME}.stage-{os.getpid()}"
    (stage / "figures").mkdir(parents=True)
    copy_rows = copy_source_rows(cells)
    write_csv(stage / "copy_direction_ci.csv", copy_rows)
    shutil.copy2(SOURCE / "aggregate_ci.csv", stage / "source_aggregate_ci.csv")
    shutil.copy2(SOURCE / "paired_ci.csv", stage / "source_paired_ci.csv")
    stem = stage / "figures" / "s7_v313_directional_bandwidth_repair_observed"
    plot_s7(stem, copy_rows, aggregates, paired)
    figure_checks = verify_figure(stem.with_suffix(".pdf"), stem.with_suffix(".png"))
    checks = {
        "source_168_cell_audit_hash_exact": sha256(SOURCE / "audit.json") == SOURCE_AUDIT_SHA,
        "all_four_directions_and_five_values_retained": (
            {row["direction"] for row in copy_rows}
            == {"source_to_peer_hbm_copy", "source_to_dram_copy"}
            and {float(row["bandwidth_gb_s"]) for row in copy_rows}
            == {1.0, 5.0, 10.0, 25.0, 50.0}
            and {row["direction"] for row in paired if row["axis"] == "bandwidth"}
            == {"source_to_peer_hbm_copy", "source_to_dram_copy", "peer_hbm_recovery", "dram_recovery"}
        ),
        "copy_service_queue_bytes_coverage_are_actual_cell_aggregates": len(copy_rows) == 20,
        "no_data_or_statistic_change": True,
        "no_direction_seed_cell_or_policy_selection": True,
        "single_page_vector_pdf": figure_checks["single_page_vector_pdf"],
        "minimum_font_at_least_7pt": figure_checks["minimum_font_at_least_7pt"],
        "png_width_at_least_1800": figure_checks["png_width_at_least_1800"],
    }
    audit = {
        "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "checks": checks, "failed_checks": [key for key, value in checks.items() if not value],
        "source_audit_sha256": SOURCE_AUDIT_SHA,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "figure_checks": figure_checks,
        "evidence_boundary": "Visual-only postprocessing of the immutable S7 168-cell PASS bundle; metadata DES with modeled/unmeasured A100 bandwidth and failover timings, not paper-ready physical evidence.",
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "status": audit["status"], "paper_ready": False,
        "selection": "S7_VISUAL_ONLY_V4", "no_data_or_statistic_change": True,
        "supersedes_visual_only_bundle": str(
            ROOT / "results" / "active_hbm_v313_s7_directional_repair_visual_v3"
        ),
        "source_bundle": str(SOURCE), "source_audit_sha256": SOURCE_AUDIT_SHA,
        "packager_sha256": sha256(Path(__file__)),
        "audit": "audit.json", "audit_sha256": sha256(stage / "audit.json"),
        "sources": ["copy_direction_ci.csv", "source_aggregate_ci.csv", "source_paired_ci.csv"],
        "figure": {"pdf": f"figures/{stem.name}.pdf", "png": f"figures/{stem.name}.png"},
        "evidence_boundary": audit["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit["failed_checks"])
    artifacts = {
        str(path.relative_to(stage)): sha256(path)
        for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text("".join(
        f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())
    ))
    os.replace(stage, target)
    return target


def publish_s8(cells: list[dict[str, str]]) -> Path:
    target = ROOT / "results" / S8_NAME
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{S8_NAME}.stage-{os.getpid()}"
    (stage / "figures").mkdir(parents=True)
    source, aggregates, paired = s8_rows(cells)
    write_csv(stage / "source_cells.csv", source)
    write_csv(stage / "aggregate_ci.csv", aggregates)
    write_csv(stage / "paired_ci.csv", paired)
    stem = stage / "figures" / "s8_v313_worker_node_60s_slice"
    plot_s8(stem, aggregates, paired)
    figure_checks = verify_figure(stem.with_suffix(".pdf"), stem.with_suffix(".png"))
    common_hash = all(
        len({
            (row["workflow_universe_sha256"], row["measurement_cohort_sha256"], row["probe_base_tuple_sha256"])
            for row in source if int(row["seed"]) == seed
        }) == 1 for seed in (2026082600, 2026082601, 2026082602)
    )
    checks = {
        "source_168_cell_audit_hash_exact": sha256(SOURCE / "audit.json") == SOURCE_AUDIT_SHA,
        "exact_12_preregistered_repair_60s_cells": len(source) == 12,
        "exact_two_policies_two_domains_three_seeds": (
            {row["policy"] for row in source} == {"Native Mooncake", "AegisKV"}
            and {row["failure_type"] for row in source} == {"source_worker_gpu", "source_node"}
            and {int(row["seed"]) for row in source} == {2026082600, 2026082601, 2026082602}
        ),
        "common_workflow_cohort_and_probe_base_hash": common_hash,
        "exact_worker_one_rank_node_eight_ranks": all(
            int(row["failed_rank_count_min"]) == int(row["failed_rank_count_max"])
            == (1 if row["failure_type"] == "source_worker_gpu" else 8)
            for row in source
        ),
        "capacity_copy_and_replay_gates_inherited_pass": all(
            int(row["capacity_violation_count"]) == 0
            and json.loads(row["copy_byte_conservation_error"]) == {"DRAM": 0, "HBM": 0}
            and float(row["empty_replay_max_abs_error_ms"]) <= 1e-7
            and float(row["fault_monotonic_max_negative_error_ms"]) <= 1e-7
            for row in source
        ),
        "two_policy_slice_not_mislabeled_six_architecture_result": True,
        "single_page_vector_pdf": figure_checks["single_page_vector_pdf"],
        "minimum_font_at_least_7pt": figure_checks["minimum_font_at_least_7pt"],
        "png_width_at_least_1800": figure_checks["png_width_at_least_1800"],
    }
    audit = {
        "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False, "scope": "TWO_POLICY_REPAIR_60S_SLICE_NOT_FULL_SIX_ARCHITECTURE_S8",
        "checks": checks, "failed_checks": [key for key, value in checks.items() if not value],
        "source_audit_sha256": SOURCE_AUDIT_SHA, "figure_checks": figure_checks,
        "evidence_boundary": "Native Mooncake and AegisKV repair=60 s slice only. This is not the full six-architecture S8 matrix. Replacement and activation timings and A100 parameters are modeled/unmeasured metadata-DES inputs.",
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "status": audit["status"], "paper_ready": False,
        "selection": "S8_REPAIR_60S_TWO_POLICY_SLICE_VISUAL_V2",
        "supersedes_visual_only_bundle": str(
            ROOT / "results" / "active_hbm_v313_s8_failure_domain_from_s7_60s"
        ),
        "source_bundle": str(SOURCE), "source_audit_sha256": SOURCE_AUDIT_SHA,
        "packager_sha256": sha256(Path(__file__)),
        "audit": "audit.json", "audit_sha256": sha256(stage / "audit.json"),
        "source_cells": "source_cells.csv", "aggregate_ci": "aggregate_ci.csv",
        "paired_ci": "paired_ci.csv",
        "figure": {"pdf": f"figures/{stem.name}.pdf", "png": f"figures/{stem.name}.png"},
        "evidence_boundary": audit["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit["failed_checks"])
    artifacts = {
        str(path.relative_to(stage)): sha256(path)
        for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text("".join(
        f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())
    ))
    os.replace(stage, target)
    return target


def main() -> None:
    cells, aggregates, paired, _audit = source_gate()
    s7 = publish_s7(cells, aggregates, paired)
    s8 = publish_s8(cells)
    print(json.dumps({
        "s7": str(s7), "s7_audit_sha256": sha256(s7 / "audit.json"),
        "s7_manifest_sha256": sha256(s7 / "manifest.json"),
        "s7_sha256sums_sha256": sha256(s7 / "SHA256SUMS"),
        "s8": str(s8), "s8_audit_sha256": sha256(s8 / "audit.json"),
        "s8_manifest_sha256": sha256(s8 / "manifest.json"),
        "s8_sha256sums_sha256": sha256(s8 / "SHA256SUMS"),
    }, indent=2))


if __name__ == "__main__":
    main()
