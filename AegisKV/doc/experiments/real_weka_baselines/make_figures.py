#!/usr/bin/env python3
"""Render the audited academic figure suite for the real-Weka baseline study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AegisKV.experiments.real_weka_baselines.run import (
    preflight_baseline_suite,
)
from benchmark.agentic_workflows.weka_full_trace import sha256_file


ROOT = Path(__file__).resolve().parent
POLICY_LABELS = {
    "recompute_fcfs": "Recompute\n(FCFS)",
    "recompute_load_aware": "Recompute\n(load-aware)",
    "dejavu_neighbor": "DejaVu\nneighbor*",
    "raidserve_host": "RaidServe\nhost*",
    "kevlarflow_ring": "KevlarFlow\nring*",
    "lumen_load_aware": "LUMEN\nload-aware*",
    "ghostserve_parity_tp8": "GhostServe\nTP=8 control*",
    "aegiskv_latest": "AegisKV\nlatest",
}
COLORS = {
    "recompute_fcfs": "#7A7A7A",
    "recompute_load_aware": "#56B4E9",
    "dejavu_neighbor": "#E69F00",
    "raidserve_host": "#D55E00",
    "kevlarflow_ring": "#CC79A7",
    "lumen_load_aware": "#009E73",
    "ghostserve_parity_tp8": "#B89B00",
    "aegiskv_latest": "#0072B2",
}
PAPER_ADAPTATIONS = {
    "dejavu_neighbor",
    "raidserve_host",
    "kevlarflow_ring",
    "lumen_load_aware",
    "ghostserve_parity_tp8",
}
CONTROLS = {"recompute_load_aware", "ghostserve_parity_tp8"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_ready_evidence(
    suite: dict[str, Any],
    resolved: dict[str, Any] | None = None,
    *,
    input_binding: dict[str, Any] | None = None,
) -> None:
    if suite.get("status") != "READY":
        raise ValueError(
            "baseline figure suite is not READY: "
            f"{suite.get('status', 'missing')}"
        )
    if resolved is None:
        return
    embedded_suite = resolved.get("suite")
    base_config = resolved.get("base_config")
    calibration = (
        base_config.get("calibration") if isinstance(base_config, dict) else None
    )
    if not isinstance(embedded_suite, dict) or embedded_suite.get("status") != "READY":
        raise ValueError("baseline result does not pin a READY suite")
    if not isinstance(calibration, dict) or calibration.get("status") != "READY":
        raise ValueError("baseline result does not pin a READY physical calibration")
    if input_binding is None or resolved.get("input_binding") != input_binding:
        raise ValueError("baseline result input SHA binding mismatch")


def _verify_baseline_result(
    results: Path,
    *,
    suite: dict[str, Any],
    input_binding: dict[str, Any],
) -> dict[str, Any]:
    expected_name = (
        "full_393_workflows_baselines_v100_"
        f"{input_binding['physical_profile_sha256'][:12]}"
    )
    if not results.is_dir() or results.name != expected_name:
        raise ValueError("baseline figure result directory is not a V100 anchor")
    hashes_path = results / "artifact_hashes.json"
    hashes = _load_json(hashes_path)
    required = {
        "audit.json",
        "calibration_provenance.json",
        "config_resolved.json",
        "dataset_manifest.json",
        "failure_requests.csv",
        "paired_bootstrap.json",
        "restore_control_summary.json",
        "runtime_aggregate.json",
        "summary.json",
    }
    if not isinstance(hashes, dict) or not required.issubset(set(hashes)):
        raise ValueError("baseline figure result artifact set is incomplete")
    for name, expected_sha in hashes.items():
        if Path(str(name)).name != str(name) or name == "artifact_hashes.json":
            raise ValueError("baseline figure result artifact name is unsafe")
        path = results / str(name)
        if not path.is_file() or sha256_file(path) != str(expected_sha):
            raise ValueError(
                f"baseline figure result artifact sha256 mismatch: {name}"
            )
    resolved = _load_json(results / "config_resolved.json")
    _require_ready_evidence(
        suite,
        resolved,
        input_binding=input_binding,
    )
    audit = _load_json(results / "audit.json")
    calibration = _load_json(results / "calibration_provenance.json")
    if not (
        audit.get("status") == "PASS"
        and audit.get("paper_ready") is False
        and calibration.get("physical_profile", {}).get("sha256")
        == input_binding["physical_profile_sha256"]
        and calibration.get("source_manifest", {}).get("sha256")
        == input_binding["source_manifest_sha256"]
        and calibration.get("config_manifest", {}).get("sha256")
        == input_binding["config_manifest_sha256"]
        and calibration.get("config_verification", {}).get("sha256")
        == input_binding["config_verification_sha256"]
        and Path(str(resolved.get("anchor_result_dir", ""))).resolve()
        == Path(input_binding["anchor_result"]["path"]).resolve()
    ):
        raise ValueError("baseline figure result evidence contract did not pass")
    return resolved


def _derived_figure_output_dir(
    requested: Path | None, *, profile_sha256: str
) -> Path:
    expected_name = f"figures_v100_{profile_sha256[:12]}"
    output = (
        requested.resolve()
        if requested is not None
        else (ROOT / expected_name).resolve()
    )
    if output.name != expected_name:
        raise ValueError(
            "figure output directory must be derived from the physical profile "
            f"hash and named {expected_name!r}"
        )
    if output.exists():
        raise FileExistsError(f"academic figure output is immutable: {output}")
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.28,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.09,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
    )


def _save(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    suite: PdfPages,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )
    suite.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _policy_hatches(policy: str) -> str:
    if policy == "ghostserve_parity_tp8":
        return "///"
    if policy == "recompute_load_aware":
        return "..."
    return ""


def figure_end_to_end(
    summary: dict[str, Any],
    bootstrap: dict[str, Any],
) -> plt.Figure:
    order = [
        "recompute_load_aware",
        "lumen_load_aware",
        "aegiskv_latest",
        "dejavu_neighbor",
        "kevlarflow_ring",
        "recompute_fcfs",
        "raidserve_host",
        "ghostserve_parity_tp8",
    ]
    recompute = float(summary["recompute_fcfs"]["total_join_barrier_harm_ms"])
    relative = [
        100.0
        * (float(summary[policy]["total_join_barrier_harm_ms"]) / recompute - 1.0)
        for policy in order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(order))
    bars = ax.barh(
        y,
        relative,
        color=[COLORS[policy] for policy in order],
        edgecolor="#2C2C2C",
        linewidth=0.45,
    )
    for bar, policy in zip(bars, order, strict=True):
        bar.set_hatch(_policy_hatches(policy))
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y, [POLICY_LABELS[policy] for policy in order])
    ax.invert_yaxis()
    ax.set_xlabel("Join-barrier harm change vs. Recompute-FCFS (%)\n← lower is better")
    ax.set_title("Complete Weka replay (955 paired failures)")
    ax.grid(axis="x")
    for yi, value in enumerate(relative):
        ax.text(
            value + (0.16 if value >= 0 else -0.16),
            yi,
            f"{value:+.2f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7.1,
        )
    ax.set_xlim(min(relative) - 1.7, max(relative) + 1.7)
    _panel_label(ax, "a")

    ax = axes[1]
    comparisons = bootstrap["comparisons"]
    compare_order = [
        "recompute_load_aware",
        "lumen_load_aware",
        "dejavu_neighbor",
        "kevlarflow_ring",
        "recompute_fcfs",
        "raidserve_host",
        "ghostserve_parity_tp8",
    ]
    estimates = np.asarray(
        [comparisons[policy]["reduction_percent"] for policy in compare_order]
    )
    lows = np.asarray(
        [comparisons[policy]["bootstrap_95ci_percent"][0] for policy in compare_order]
    )
    highs = np.asarray(
        [comparisons[policy]["bootstrap_95ci_percent"][1] for policy in compare_order]
    )
    y = np.arange(len(compare_order))
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        color=COLORS["aegiskv_latest"],
        ecolor="#40566D",
        capsize=2.4,
        markersize=4.2,
        linewidth=1.0,
    )
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y, [POLICY_LABELS[policy] for policy in compare_order])
    ax.invert_yaxis()
    ax.set_xlabel("AegisKV harm reduction vs. baseline (%)\n→ AegisKV is better")
    ax.set_title("Paired workflow bootstrap (4,000 resamples)")
    ax.grid(axis="x")
    for yi, estimate in enumerate(estimates):
        ax.text(
            estimate + (0.25 if estimate >= 0 else -0.25),
            yi,
            f"{estimate:+.2f}%",
            va="center",
            ha="left" if estimate >= 0 else "right",
            fontsize=7.0,
        )
    ax.set_xlim(min(lows) - 2.0, max(highs) + 2.0)
    _panel_label(ax, "b")
    fig.suptitle(
        "AegisKV improves over FCFS; load-aware dispatch remains stronger",
        fontsize=10.7,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "* capacity-matched component adaptation; hatched entries are dedicated controls, not dense TP=1 paper-system reproductions.",
        ha="center",
        fontsize=7.1,
    )
    return fig


def _runtime_metrics(
    baseline_runtime: dict[str, Any], aegis_runtime: dict[str, Any]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {
        "recompute_fcfs": {"copy_bytes": 0.0, "jobs": 0.0, "commits": 0.0},
        "recompute_load_aware": {"copy_bytes": 0.0, "jobs": 0.0, "commits": 0.0},
    }
    for policy in (
        "dejavu_neighbor",
        "raidserve_host",
        "kevlarflow_ring",
        "ghostserve_parity_tp8",
    ):
        metrics = baseline_runtime[policy]["metrics"]
        result[policy] = {
            "copy_bytes": float(metrics["copy_stored_bytes"]),
            "jobs": float(metrics["copy_jobs"]),
            "commits": float(metrics["committed_chunks"]),
        }
    lumen = baseline_runtime["lumen_load_aware"]["metrics"]
    result["lumen_load_aware"] = {
        "copy_bytes": float(lumen["copy_bytes"]),
        "jobs": float(lumen["committed_chunks"] + lumen["cancelled_jobs"]),
        "commits": float(lumen["committed_chunks"]),
    }
    aegis = aegis_runtime["aegiskv"]
    result["aegiskv_latest"] = {
        "copy_bytes": float(aegis["metrics"]["hbm_copy_bytes"])
        + float(aegis["metrics"]["dram_copy_bytes"]),
        "jobs": float(aegis["metrics"]["hbm_jobs"])
        + float(aegis["metrics"]["dram_jobs"]),
        "commits": float(aegis["controller_event_counts"]["tier_copy_committed"]),
    }
    return result


def figure_cost_pareto(
    summary: dict[str, Any],
    runtime: dict[str, dict[str, float]],
) -> plt.Figure:
    policies = [
        "recompute_fcfs",
        "recompute_load_aware",
        "lumen_load_aware",
        "aegiskv_latest",
        "dejavu_neighbor",
        "raidserve_host",
        "kevlarflow_ring",
        "ghostserve_parity_tp8",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), constrained_layout=True)
    ax = axes[0]
    for policy in policies:
        x = runtime[policy]["copy_bytes"] / 1e15
        y = float(summary[policy]["total_join_barrier_harm_ms"]) / 3.6e6
        marker = "s" if policy in CONTROLS else "o"
        ax.scatter(
            x,
            y,
            s=45 if policy == "aegiskv_latest" else 31,
            marker=marker,
            color=COLORS[policy],
            edgecolor="#222222",
            linewidth=0.55,
            zorder=3,
        )
        label_positions = {
            "recompute_fcfs": (0.02, 5.972),
            "recompute_load_aware": (0.02, 5.365),
            "lumen_load_aware": (0.13, 5.555),
            "aegiskv_latest": (0.54, 5.835),
            "dejavu_neighbor": (0.28, 5.845),
            "raidserve_host": (0.40, 6.025),
            "kevlarflow_ring": (0.57, 5.985),
            "ghostserve_parity_tp8": (0.16, 6.045),
        }
        ax.annotate(
            POLICY_LABELS[policy].replace("\n", " "),
            (x, y),
            xytext=label_positions[policy],
            textcoords="data",
            fontsize=6.7,
            arrowprops={
                "arrowstyle": "-",
                "color": "#555555",
                "lw": 0.45,
                "shrinkA": 1.5,
                "shrinkB": 2.5,
            },
        )
    ax.set_xlabel("Protection bytes sent over the full trace (PB, decimal)")
    ax.set_ylabel("Total join-barrier harm (hours)")
    ax.set_title("End-to-end harm versus normal-path copy work")
    ax.grid(True)
    ax.set_xlim(-0.035, 0.755)
    ax.set_ylim(5.35, 6.06)
    _panel_label(ax, "a")

    ax = axes[1]
    commit_policies = [
        "aegiskv_latest",
        "dejavu_neighbor",
        "kevlarflow_ring",
        "lumen_load_aware",
        "ghostserve_parity_tp8",
        "raidserve_host",
    ]
    efficiency = [
        100.0 * runtime[p]["commits"] / max(runtime[p]["jobs"], 1.0)
        for p in commit_policies
    ]
    y = np.arange(len(commit_policies))
    bars = ax.barh(
        y,
        efficiency,
        color=[COLORS[p] for p in commit_policies],
        edgecolor="#222222",
        linewidth=0.45,
    )
    for bar, policy in zip(bars, commit_policies, strict=True):
        bar.set_hatch(_policy_hatches(policy))
    ax.set_yticks(y, [POLICY_LABELS[p] for p in commit_policies])
    ax.invert_yaxis()
    ax.set_xlim(0, 106)
    ax.set_xlabel("Committed copy jobs (%)")
    ax.set_title("Useful-copy completion efficiency")
    ax.grid(axis="x")
    for yi, value in enumerate(efficiency):
        ax.text(value + 1.0, yi, f"{value:.1f}%", va="center", fontsize=7.1)
    _panel_label(ax, "b")
    fig.suptitle(
        "AegisKV's clear advantage is selective copy completion—not default WCT",
        fontsize=10.5,
        fontweight="bold",
    )
    return fig


def figure_protection_coverage(
    failure_rows: list[dict[str, str]],
) -> plt.Figure:
    policies = [
        "aegiskv_latest",
        "dejavu_neighbor",
        "kevlarflow_ring",
        "lumen_load_aware",
        "ghostserve_parity_tp8",
        "raidserve_host",
    ]
    hit_rate: list[float] = []
    token_coverage: list[float] = []
    for policy in policies:
        selected = [row for row in failure_rows if row["policy"] == policy]
        hit_rate.append(
            100.0
            * sum(int(row["restored_tokens"]) > 0 for row in selected)
            / len(selected)
        )
        token_coverage.append(
            100.0
            * sum(int(row["restored_tokens"]) for row in selected)
            / sum(int(row["produced_tokens"]) for row in selected)
        )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95), constrained_layout=True)
    labels = [POLICY_LABELS[p] for p in policies]
    x = np.arange(len(policies))
    for ax, values, title, ylabel, panel in (
        (
            axes[0],
            hit_rate,
            "Failures with any committed prefix",
            "Failure requests covered (%)",
            "a",
        ),
        (
            axes[1],
            token_coverage,
            "Produced token-work restored",
            "Restored / produced tokens (%)",
            "b",
        ),
    ):
        bars = ax.bar(
            x,
            values,
            color=[COLORS[p] for p in policies],
            edgecolor="#222222",
            linewidth=0.45,
            width=0.72,
        )
        for bar, policy in zip(bars, policies, strict=True):
            bar.set_hatch(_policy_hatches(policy))
        ax.set_xticks(x, labels, rotation=31, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_ylim(0, max(values) * 1.20 + 1e-9)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )
        _panel_label(ax, panel)
    aegis_token_coverage = token_coverage[policies.index("aegiskv_latest")]
    fig.suptitle(
        "Request coverage overstates protected work: AegisKV restores only "
        f"{aegis_token_coverage:.1f}% of failed token-work",
        fontsize=10.3,
        fontweight="bold",
    )
    return fig


def figure_restore_controls(
    restore_summary: dict[str, Any],
    summary: dict[str, Any],
) -> plt.Figure:
    planners = [
        "raidserve_serial",
        "restore_or_recompute_split",
        "bounded_state_restore",
        "cacheflow_3d_restore",
    ]
    labels = [
        "Serial host\nrestore*",
        "Restore/recompute\nsplit*",
        "BSR windowed\nrestore*",
        "CacheFlow 3D\nexecutor*",
    ]
    means = [
        float(summary["raidserve_host"]["mean_recovery_service_ms"]) / 1000.0,
        float(restore_summary["restore_or_recompute_split"]["mean_completion_ms"]) / 1000.0,
        float(restore_summary["bounded_state_restore"]["mean_completion_ms"]) / 1000.0,
        float(restore_summary["cacheflow_3d_restore"]["mean_completion_ms"]) / 1000.0,
    ]
    p95 = [
        float(summary["raidserve_host"]["p95_recovery_service_ms"]) / 1000.0,
        float(restore_summary["restore_or_recompute_split"]["p95_completion_ms"]) / 1000.0,
        float(restore_summary["bounded_state_restore"]["p95_completion_ms"]) / 1000.0,
        float(restore_summary["cacheflow_3d_restore"]["p95_completion_ms"]) / 1000.0,
    ]
    staging = [
        float("nan"),
        float(restore_summary["restore_or_recompute_split"]["max_peak_staging_gib"]),
        float(restore_summary["bounded_state_restore"]["max_peak_staging_gib"]),
        float(restore_summary["cacheflow_3d_restore"]["max_peak_staging_gib"]),
    ]
    colors = ["#D55E00", "#8C6BB1", "#009E73", "#56B4E9"]
    y = np.arange(len(planners))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), constrained_layout=True)
    ax = axes[0]
    width = 0.34
    ax.barh(
        y - width / 2,
        means,
        height=width,
        label="Mean",
        color=colors,
        edgecolor="#222",
        linewidth=0.45,
    )
    ax.barh(
        y + width / 2,
        p95,
        height=width,
        label="P95",
        color=colors,
        alpha=0.45,
        edgecolor="#222",
        linewidth=0.45,
        hatch="///",
    )
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Executor completion time (s, log scale)")
    ax.set_title("Same 955 external-state opportunities")
    ax.grid(axis="x", which="both")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    _panel_label(ax, "a")

    ax = axes[1]
    shown_y = y[1:]
    shown = np.asarray(staging[1:])
    bars = ax.barh(
        shown_y,
        shown,
        color=colors[1:],
        edgecolor="#222",
        linewidth=0.45,
    )
    ax.set_yticks(shown_y, labels[1:])
    ax.invert_yaxis()
    ax.set_xlabel("Maximum local staging working set (GiB)")
    ax.set_title("BSR bounds local materialization")
    ax.grid(axis="x")
    ax.set_xlim(0, max(shown) * 1.23 + 1e-9)
    for bar, value in zip(bars, shown, strict=True):
        ax.text(
            value + max(shown) * 0.035,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g}",
            ha="left",
            va="center",
            fontsize=7.0,
        )
    _panel_label(ax, "b")
    fig.suptitle(
        "Recovery executors are component controls, not protection-policy baselines",
        fontsize=10.3,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "* modeled adaptation. CacheFlow uses 8 declared GPU lanes; most host prefixes are unavailable or slower than recomputation.",
        ha="center",
        fontsize=7.1,
    )
    return fig


def figure_scale_and_predictor(
    dataset: dict[str, Any],
    workload: dict[str, Any],
    algorithm1: dict[str, Any],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1), constrained_layout=True)
    ax = axes[0]
    names = ["Workflows", "Requests", "Input tokens", "Output tokens", "Sealed chunks"]
    values = [
        dataset["workflow_count"],
        dataset["request_count"],
        dataset["input_tokens"],
        dataset["output_tokens"],
        workload["intended_sealed_chunks"],
    ]
    colors = ["#7A7A7A", "#56B4E9", "#0072B2", "#009E73", "#E69F00"]
    bars = ax.bar(np.arange(len(values)), values, color=colors, edgecolor="#222", linewidth=0.45)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(values)), names, rotation=28, ha="right")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Complete real-Weka scale")
    ax.grid(axis="y", which="both")
    for bar, value in zip(bars, values, strict=True):
        text = f"{value / 1e9:.2f}B" if value >= 1e9 else (f"{value / 1e6:.2f}M" if value >= 1e6 else f"{value:,}")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.18,
            text,
            ha="center",
            va="bottom",
            fontsize=7.0,
            rotation=18,
        )
    _panel_label(ax, "a")

    ax = axes[1]
    error_names = ["Rem.\nmedian", "Rem.\nP90", "Rem.\nMAE", "Slack\nmedian", "Slack\nP90"]
    errors = np.asarray(
        [
            algorithm1["remaining_p50_absolute_error_ms"],
            algorithm1["remaining_p90_absolute_error_ms"],
            algorithm1["remaining_mae_ms"],
            algorithm1["slack_p50_absolute_error_ms"],
            algorithm1["slack_p90_absolute_error_ms"],
        ]
    ) / 1000.0
    bars = ax.bar(
        np.arange(len(errors)),
        errors,
        color=["#0072B2", "#56B4E9", "#0072B2", "#E69F00", "#F0B45B"],
        edgecolor="#222",
        linewidth=0.45,
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(errors)), error_names, rotation=18, ha="right")
    ax.set_ylabel("Absolute prediction error (s, log scale)")
    ax.set_title("Workflow-disjoint Algorithm 1 accuracy")
    ax.grid(axis="y", which="both")
    for bar, value in zip(bars, errors, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.0f}s",
            ha="center",
            fontsize=7.0,
        )
    coverage = 100.0 * float(algorithm1["remaining_interval_coverage"])
    ax.text(
        0.98,
        0.96,
        f"90% interval target\nobserved coverage: {coverage:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#777", "alpha": 0.95},
    )
    _panel_label(ax, "b")
    fig.suptitle(
        "Scale is real; predictor quality and calibration range are not paper-ready",
        fontsize=10.3,
        fontweight="bold",
    )
    return fig


def figure_advantage_ceiling(
    failure_rows: list[dict[str, str]],
    runtime_config: dict[str, Any],
    algorithm1: dict[str, Any],
    workload: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[plt.Figure, list[dict[str, float]]]:
    aegis_rows = [row for row in failure_rows if row["policy"] == "aegiskv_latest"]
    token_coverage = (
        sum(int(row["restored_tokens"]) for row in aegis_rows)
        / sum(int(row["produced_tokens"]) for row in aegis_rows)
    )
    recompute_rate = float(runtime_config["recompute"]["ms_per_token"])
    hbm_rate = float(runtime_config["hbm_load"]["ms_per_token"])
    cost_ratio = hbm_rate / recompute_rate
    coverages = np.linspace(0.0, 1.0, 201)
    ratios = np.linspace(0.0, 1.2, 181)
    surface = 100.0 * np.maximum(coverages[None, :] * (1.0 - ratios[:, None]), 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.2), constrained_layout=True)
    ax = axes[0]
    mesh = ax.pcolormesh(
        100.0 * coverages,
        ratios,
        surface,
        cmap="viridis",
        shading="auto",
        vmin=0,
        vmax=50,
    )
    contours = ax.contour(
        100.0 * coverages,
        ratios,
        surface,
        levels=[1, 5, 10, 20, 40],
        colors="white",
        linewidths=0.65,
    )
    ax.clabel(contours, inline=True, fontsize=6.6, fmt="%g%%")
    ax.scatter(
        [100.0 * token_coverage],
        [cost_ratio],
        s=55,
        marker="X",
        color="#D55E00",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Current calibrated point",
    )
    current_ceiling = 100.0 * token_coverage * (1.0 - cost_ratio)
    empirical = 100.0 * (
        float(summary["recompute_fcfs"]["total_join_barrier_harm_ms"])
        - float(summary["aegiskv_latest"]["total_join_barrier_harm_ms"])
    ) / float(summary["recompute_fcfs"]["total_join_barrier_harm_ms"])
    ax.annotate(
        f"{100.0 * token_coverage:.2f}% token coverage\n"
        f"restore/recompute={cost_ratio:.3f}\n"
        f"work ceiling≈{current_ceiling:.2f}%\n"
        f"empirical harm={empirical:.2f}%",
        xy=(100.0 * token_coverage, cost_ratio),
        xytext=(25, 0.78),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333"},
        fontsize=7.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92, "edgecolor": "#777"},
    )
    ax.set_xlabel("Protected failed token-work (%)")
    ax.set_ylabel("HBM restore cost / recompute cost")
    ax.set_title("Analytic recovery-work reduction ceiling")
    ax.legend(frameon=False, loc="lower right")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Maximum recovery-work reduction (%)")
    _panel_label(ax, "a")

    diagnostics = {
        "Restore cost / recompute": cost_ratio,
        "Unprotected failed token-work": 1.0 - token_coverage,
        "Remaining-time interval misses": 1.0 - float(algorithm1["remaining_interval_coverage"]),
        "Requests beyond calibration": float(workload["fraction_requests_beyond_calibrated_context"]),
    }
    ax = axes[1]
    names = list(diagnostics)
    values = 100.0 * np.asarray(list(diagnostics.values()))
    colors = ["#D55E00", "#E69F00", "#56B4E9", "#CC79A7"]
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=colors, edgecolor="#222", linewidth=0.45)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Ratio / fraction (%)")
    ax.set_title("Current bottlenecks and evidence gaps")
    ax.grid(axis="x")
    for yi, value in enumerate(values):
        ax.text(value + 1.3, yi, f"{value:.1f}%", va="center", fontsize=7.1)
    _panel_label(ax, "b")
    fig.suptitle(
        "Why a large AegisKV advantage cannot be guaranteed in the current regime",
        fontsize=10.3,
        fontweight="bold",
    )
    phase_rows = [
        {
            "protected_token_fraction": float(coverage),
            "restore_to_recompute_cost_ratio": float(ratio),
            "recovery_work_reduction_ceiling_percent": float(
                max(100.0 * coverage * (1.0 - ratio), 0.0)
            ),
        }
        for ratio in ratios
        for coverage in coverages
    ]
    return fig, phase_rows


def write_metric_table(
    path: Path,
    summary: dict[str, Any],
    runtime: dict[str, dict[str, float]],
    failure_rows: list[dict[str, str]],
) -> None:
    fields = [
        "policy",
        "fidelity",
        "total_join_barrier_harm_hours",
        "change_vs_recompute_percent",
        "copy_bytes_decimal_pb",
        "copy_commit_efficiency_percent",
        "failure_request_coverage_percent",
        "failed_token_coverage_percent",
    ]
    recompute = float(summary["recompute_fcfs"]["total_join_barrier_harm_ms"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for policy in POLICY_LABELS:
            selected = [row for row in failure_rows if row["policy"] == policy]
            protected = sum(int(row["restored_tokens"]) > 0 for row in selected)
            restored = sum(int(row["restored_tokens"]) for row in selected)
            produced = sum(int(row["produced_tokens"]) for row in selected)
            jobs = runtime[policy]["jobs"]
            commits = runtime[policy]["commits"]
            writer.writerow(
                {
                    "policy": policy,
                    "fidelity": (
                        "native"
                        if policy in ("aegiskv_latest", "recompute_fcfs")
                        else (
                            "dedicated_control"
                            if policy in CONTROLS
                            else "capacity_matched_adaptation"
                        )
                    ),
                    "total_join_barrier_harm_hours": float(summary[policy]["total_join_barrier_harm_ms"]) / 3.6e6,
                    "change_vs_recompute_percent": 100.0 * (float(summary[policy]["total_join_barrier_harm_ms"]) / recompute - 1.0),
                    "copy_bytes_decimal_pb": runtime[policy]["copy_bytes"] / 1e15,
                    "copy_commit_efficiency_percent": 100.0 * commits / jobs if jobs else 0.0,
                    "failure_request_coverage_percent": 100.0 * protected / len(selected) if selected else 0.0,
                    "failed_token_coverage_percent": 100.0 * restored / produced if produced else 0.0,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suite-config", type=Path, default=ROOT / "config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = args.suite_config.resolve()
    (
        suite,
        _base_config,
        _runtime_config,
        _calibration,
        _projection,
        full_root,
        input_binding,
    ) = preflight_baseline_suite(suite_path)
    results = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else (
            ROOT
            / "results"
            / (
                "full_393_workflows_baselines_v100_"
                f"{input_binding['physical_profile_sha256'][:12]}"
            )
        ).resolve()
    )
    resolved = _verify_baseline_result(
        results,
        suite=suite,
        input_binding=input_binding,
    )
    output = _derived_figure_output_dir(
        args.output_dir,
        profile_sha256=input_binding["physical_profile_sha256"],
    )
    summary = _load_json(results / "summary.json")
    bootstrap = _load_json(results / "paired_bootstrap.json")
    baseline_runtime = _load_json(results / "runtime_aggregate.json")
    restore_summary = _load_json(results / "restore_control_summary.json")
    dataset = _load_json(results / "dataset_manifest.json")
    failure_rows = _read_csv(results / "failure_requests.csv")
    aegis_runtime = _load_json(full_root / "runtime_aggregate.json")
    algorithm1 = _load_json(full_root / "algorithm1_metrics.json")
    workload = _load_json(full_root / "workload_projection.json")
    runtime = _runtime_metrics(baseline_runtime, aegis_runtime)
    output.mkdir(parents=True, exist_ok=False)
    data_dir = output / "data"
    data_dir.mkdir(parents=False, exist_ok=False)
    _style()
    with PdfPages(output / "academic_figure_suite.pdf") as suite:
        _save(figure_end_to_end(summary, bootstrap), output, "fig2_end_to_end_harm", suite)
        _save(figure_cost_pareto(summary, runtime), output, "fig3_cost_pareto", suite)
        _save(figure_protection_coverage(failure_rows), output, "fig4_protection_coverage", suite)
        _save(figure_restore_controls(restore_summary, summary), output, "fig5_restore_executors", suite)
        _save(figure_scale_and_predictor(dataset, workload, algorithm1), output, "fig6_scale_and_predictor", suite)
        phase, phase_rows = figure_advantage_ceiling(
            failure_rows,
            resolved["runtime_config"],
            algorithm1,
            workload,
            summary,
        )
        _save(phase, output, "fig7_advantage_ceiling", suite)
    write_metric_table(data_dir / "headline_metrics.csv", summary, runtime, failure_rows)
    with (data_dir / "advantage_ceiling_surface.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)
    manifest = {
        "schema_version": 1,
        "source_results": str(results),
        "source_anchor": str(full_root),
        "input_binding": input_binding,
        "figures": sorted(
            path.name
            for path in output.iterdir()
            if path.is_file() and path.name != "manifest.json"
        ),
        "data": sorted(path.name for path in data_dir.iterdir()),
        "evidence_boundary": (
            "Full real-Weka metadata simulation with calibrated projection; "
            "not physical end-to-end Mooncake evidence."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
