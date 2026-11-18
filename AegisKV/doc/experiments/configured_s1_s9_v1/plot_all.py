"""Render S1--S9 from the audited CSV declared by each deployment config."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from .config import GroupConfig
except ImportError:
    from config import GroupConfig  # type: ignore[no-redef]


PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#6A4C93"]
MODELED_FOOTER = "SGLSim metadata DES (modeled); preliminary, not physical A100 evidence"
PHYSICAL_FOOTER = "Independent RTX4090 physical holdout vs matched fitted simulator; not an A100 WCT result"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _measured(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") != "NOT_MEASURED"]


def _finish(fig: Any, output_stem: Path, footer: str) -> list[Path]:
    fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _errorbar(ax: Any, x: list[float], y: list[float], low: list[float], high: list[float], **kwargs: Any) -> None:
    lower = np.maximum(0.0, np.asarray(y) - np.asarray(low))
    upper = np.maximum(0.0, np.asarray(high) - np.asarray(y))
    ax.errorbar(x, y, yerr=[lower, upper], capsize=3, **kwargs)


def plot_s1(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = _rows(config.source_csv)
    domains = ["source_worker_gpu", "source_node"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    for ax, domain in zip(axes, domains):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["failure_domain"] == domain:
                grouped[row["display_policy"]].append(row)
        for color, (policy, values) in zip(PALETTE, grouped.items()):
            values.sort(key=lambda row: _f(row, "wct_ms"))
            ax.plot(
                [_f(row, "wct_ms") / 1000.0 for row in values],
                [_f(row, "ecdf") for row in values],
                label=policy,
                color=color,
                linewidth=1.5,
            )
        ax.set_title("Worker/GPU fault" if domain == "source_worker_gpu" else "Source-node fault")
        ax.set_xlabel("Workflow completion time (s)")
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("CDF")
    fig.suptitle("S1 — Six-architecture WCT distributions (10 paired seeds)")
    return _finish(fig, output_stem, MODELED_FOOTER)


def plot_s2(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = _rows(config.source_csv)
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_f(row, "rate_per_1000_gpu_shard_hour")].append(row)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for color, rate in zip(PALETTE, sorted(grouped)):
        values = sorted(grouped[rate], key=lambda row: _f(row, "concurrent_workflows_per_worker"))
        _errorbar(
            ax,
            [_f(row, "concurrent_workflows_per_worker") for row in values],
            [_f(row, "wct_mean_gain_pct_mean") for row in values],
            [_f(row, "wct_mean_gain_pct_ci95_low") for row in values],
            [_f(row, "wct_mean_gain_pct_ci95_high") for row in values],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"failure rate {rate:g}/1k GPU-shard-h",
        )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title("S2 — Concurrency and failure-rate sensitivity")
    ax.set_xlabel("Active requests per GPU")
    ax.set_ylabel("AegisKV WCT improvement vs modeled Native (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return _finish(fig, output_stem, MODELED_FOOTER)


def plot_s3(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = _rows(config.source_csv)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharey=True)
    models = ["Qwen3-8B", "Qwen3-32B"]
    for row_index, model in enumerate(models):
        for col_index, panel_axis in enumerate(("scale", "context")):
            ax = axes[row_index, col_index]
            selected = [
                row
                for row in rows
                if row["model"] == model and row["panel_axis"] == panel_axis
            ]
            grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for row in selected:
                context_label = row["context_tokens"] if panel_axis == "scale" else "all"
                grouped[(row["topology"], context_label)].append(row)
            for line_index, ((topology, context_label), values) in enumerate(sorted(grouped.items())):
                x_key = "system_dp" if panel_axis == "scale" else "context_tokens"
                values.sort(key=lambda row: _f(row, x_key))
                x = [_f(row, x_key) for row in values]
                y = [_f(row, "aegis_wct_reduction_pct_mean") for row in values]
                half = [_f(row, "aegis_wct_reduction_pct_ci95_half_width") for row in values]
                label = topology.replace("_", "-")
                if panel_axis == "scale" and len(grouped) > 2:
                    label += f", {int(float(context_label)) // 1024}K"
                ax.errorbar(
                    x,
                    y,
                    yerr=half,
                    marker="o",
                    capsize=2,
                    linewidth=1.4,
                    color=PALETTE[line_index % len(PALETTE)],
                    label=label,
                )
            ax.axhline(0, color="#444444", linewidth=0.8, linestyle=":")
            ax.set_title(f"{model} {'DP scale' if panel_axis == 'scale' else 'context'}")
            ax.set_xlabel("System DP" if panel_axis == "scale" else "Context tokens")
            ax.set_ylabel("Mean WCT reduction vs Native (%)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle("S3 — Modeled scale/context across Qwen3-8B and Qwen3-32B layouts")
    return _finish(fig, output_stem, MODELED_FOOTER)


def plot_s4(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = _rows(config.source_csv)
    metrics = [
        ("healthy_wct_mean_ms", "Healthy mean WCT"),
        ("healthy_wct_p99_ms", "Healthy p99 WCT"),
        ("fault_wct_mean_ms", "Fault mean WCT"),
        ("fault_wct_p99_ms", "Fault p99 WCT"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [row for row in rows if row["metric"] == metric]
        labels = [row["algorithm"].replace("Algorithm ", "Alg. ") for row in values]
        means = [(_f(row, "ablation_over_full_mean") - 1.0) * 100.0 for row in values]
        errors = [_f(row, "paired_ratio_ci95_half_width") * 100.0 for row in values]
        x = np.arange(len(values))
        ax.bar(x, means, color=PALETTE[: len(values)])
        ax.errorbar(x, means, yerr=errors, fmt="none", color="#222222", capsize=3)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Ablation vs full Δ (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("S4 — Algorithm 1/2/3 paired ablation")
    return _finish(fig, output_stem, MODELED_FOOTER)


def plot_s5(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = sorted(_rows(config.source_csv), key=lambda row: _f(row, "time_s"))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    x = [_f(row, "time_s") for row in rows]
    panels = [
        (
            axes[0, 0],
            "Request pressure",
            [
                ("offered_active_requests_per_gpu", "Offered", PALETTE[0]),
                ("realized_active_requests_per_gpu", "Realized", PALETTE[1]),
            ],
            "requests/GPU",
        ),
        (
            axes[0, 1],
            "AIMD control",
            [
                ("pressure_fraction", "Pressure", PALETTE[4]),
                ("alpha", "α", PALETTE[2]),
            ],
            "fraction",
        ),
        (
            axes[1, 0],
            "Serving and quota",
            [
                ("quota_gib_per_target", "Quota/target", PALETTE[2]),
                ("serving_hbm_gib_per_gpu", "Serving HBM", PALETTE[0]),
            ],
            "GiB/GPU",
        ),
        (
            axes[1, 1],
            "Independent backup tiers",
            [
                ("peer_hbm_backup_gib_per_gpu", "Peer HBM", PALETTE[1]),
                ("dram_gib_per_gpu", "DRAM", PALETTE[4]),
            ],
            "GiB/GPU",
        ),
    ]
    for ax, title, series, ylabel in panels:
        for prefix, label, color in series:
            y = [_f(row, f"{prefix}_mean") for row in rows]
            half = [_f(row, f"{prefix}_ci95_half_width") for row in rows]
            ax.plot(x, y, color=color, linewidth=1.6, label=label)
            ax.fill_between(x, np.asarray(y) - half, np.asarray(y) + half, color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("S5 — Online control and independent-tier occupancy")
    return _finish(fig, output_stem, MODELED_FOOTER)


def plot_s6(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = _rows(config.source_csv)
    metrics = [
        ("recovery_request_hit_fraction", "Recovery request hit"),
        ("recovery_prefix_token_hit_fraction", "Recovered prefix-token fraction"),
        ("copy_amplification_physical_over_logical", "Copy amplification"),
        ("eviction_plus_reclaim_gib", "Eviction + reclaim (GiB)"),
        ("hbm_occupancy_time_mean_gib_per_gpu", "HBM occupancy (GiB/GPU)"),
        ("dram_occupancy_time_mean_gib_per_gpu", "DRAM occupancy (GiB/GPU)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3))
    for ax, (metric, title) in zip(axes.flat, metrics):
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["policy"], row["topology"])].append(row)
        for index, ((policy, topology), values) in enumerate(sorted(grouped.items())):
            values.sort(key=lambda row: _f(row, "reuse_fraction"))
            x = [_f(row, "reuse_fraction") for row in values]
            y = [_f(row, f"{metric}_mean") for row in values]
            half = [_f(row, f"{metric}_ci95_half_width") for row in values]
            ax.errorbar(
                x,
                y,
                yerr=half,
                marker="o",
                capsize=2,
                linewidth=1.3,
                color=PALETTE[index],
                label=f"{policy}, {topology.replace('_', '-')}",
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Shared-prefix reuse fraction")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=5.8)
    fig.suptitle("S6 — Shared-prefix reuse, cost, and occupancy")
    return _finish(fig, output_stem, MODELED_FOOTER)


def plot_s7(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = _rows(config.source_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    bandwidth_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    repair_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["axis"] == "bandwidth":
            bandwidth_groups[row["direction"]].append(row)
        else:
            repair_groups[row["failure_type"]].append(row)
    for index, (direction, values) in enumerate(sorted(bandwidth_groups.items())):
        values.sort(key=lambda row: _f(row, "bandwidth_gb_s"))
        axes[0].errorbar(
            [_f(row, "bandwidth_gb_s") for row in values],
            [_f(row, "failure_target_wct_relative_gain_mean") * 100.0 for row in values],
            yerr=[_f(row, "failure_target_wct_relative_gain_ci95_half_width") * 100.0 for row in values],
            marker="o",
            capsize=2,
            linewidth=1.4,
            color=PALETTE[index],
            label=direction.replace("_", " "),
        )
    for index, (failure_type, values) in enumerate(sorted(repair_groups.items())):
        values.sort(key=lambda row: _f(row, "repair_ms"))
        axes[1].errorbar(
            [_f(row, "repair_ms") / 1000.0 for row in values],
            [_f(row, "failure_target_wct_relative_gain_mean") * 100.0 for row in values],
            yerr=[_f(row, "failure_target_wct_relative_gain_ci95_half_width") * 100.0 for row in values],
            marker="o",
            capsize=2,
            linewidth=1.4,
            color=PALETTE[index + 4],
            label=failure_type.replace("_", " "),
        )
    for ax, title, xlabel in (
        (axes[0], "Four directional transfer paths", "Bandwidth (GB/s)"),
        (axes[1], "Worker/GPU and node repair", "Repair duration (s)"),
    ):
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Failure-target WCT reduction (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("S7 — Transfer-path and repair sensitivity")
    return _finish(fig, output_stem, MODELED_FOOTER)


def plot_s8(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = _rows(config.source_csv)
    domains = ["source_worker_gpu", "source_node"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True)
    for ax, domain in zip(axes, domains):
        values = [row for row in rows if row["failure_domain"] == domain]
        x = np.arange(len(values))
        means = [_f(row, "wct_mean_ms_mean") / 1000.0 for row in values]
        lows = [_f(row, "wct_mean_ms_ci95_low") / 1000.0 for row in values]
        highs = [_f(row, "wct_mean_ms_ci95_high") / 1000.0 for row in values]
        ax.bar(x, means, color=PALETTE[: len(values)])
        ax.errorbar(
            x,
            means,
            yerr=[np.asarray(means) - lows, np.asarray(highs) - means],
            fmt="none",
            color="#222222",
            capsize=3,
        )
        ax.set_xticks(x, [row["display_policy"] for row in values], rotation=20, ha="right")
        ax.set_title("Worker/GPU fault" if domain == "source_worker_gpu" else "Source-node fault")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean workflow completion time (s)")
    fig.suptitle("S8 — Six architectures across two failure domains (derived from S1)")
    return _finish(fig, output_stem, MODELED_FOOTER)


def plot_s9(config: GroupConfig, output_stem: Path) -> list[Path]:
    rows = _rows(config.source_csv)
    path_order = ["source_to_peer_hbm", "source_to_dram", "peer_hbm_recovery", "dram_recovery"]
    titles = ["Source → peer HBM", "Source → DRAM", "Peer HBM recovery", "DRAM recovery"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.7))
    for ax, path, title, color in zip(axes.flat, path_order, titles, PALETTE):
        values = sorted([row for row in rows if row["path"] == path], key=lambda row: _f(row, "payload_mib"))
        x = [_f(row, "payload_mib") for row in values]
        observed = [_f(row, "observed_mean_ms") for row in values]
        low = [_f(row, "observed_mean_ci95_low_ms") for row in values]
        high = [_f(row, "observed_mean_ci95_high_ms") for row in values]
        predicted = [_f(row, "predicted_transfer_ms") for row in values]
        _errorbar(ax, x, observed, low, high, marker="o", color=color, linewidth=1.8, label="physical holdout")
        ax.plot(x, predicted, marker="s", linestyle="--", color="#333333", linewidth=1.4, label="fitted simulator")
        for row in values:
            if row["cell_acceptance_pass"].strip().lower() not in {"true", "1", "pass"}:
                ax.scatter([_f(row, "payload_mib")], [_f(row, "observed_mean_ms")], s=95, facecolors="none", edgecolors="#D00000", linewidths=2, zorder=5)
                ax.annotate(f"FAIL: {_f(row, 'mape_percent'):.1f}%", (_f(row, "payload_mib"), _f(row, "observed_mean_ms")), xytext=(5, 8), textcoords="offset points", fontsize=8, color="#D00000")
        ax.set_title(title)
        ax.set_xlabel("Payload (MiB)")
        ax.set_ylabel("Transfer time (ms)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("S9 — Four-direction independent calibration holdout (gate: FAIL, 15/16 pass)")
    return _finish(fig, output_stem, PHYSICAL_FOOTER)


PLOTTERS: dict[str, Callable[[GroupConfig, Path], list[Path]]] = {
    "S1": plot_s1,
    "S2": plot_s2,
    "S3": plot_s3,
    "S4": plot_s4,
    "S5": plot_s5,
    "S6": plot_s6,
    "S7": plot_s7,
    "S8": plot_s8,
    "S9": plot_s9,
}


def render_group(config: GroupConfig, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = figures_dir / f"{config.group.lower()}_{config.deployment['execution']['figure_slug']}"
    return PLOTTERS[config.group](config, stem)


def render_montage(png_paths: list[Path], figures_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, path in zip(axes.flat, png_paths):
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(path.name.split("_", 1)[0].upper(), fontsize=11)
        ax.axis("off")
    fig.suptitle("AegisKV S1–S9 configured experiment figures", fontsize=16)
    return _finish(
        fig,
        figures_dir / "s1_s9_overview_montage",
        "S1–S8: modeled preliminary DES; S9: independent RTX4090 holdout; paper_ready=false",
    )
