#!/usr/bin/env python3
"""Academic-style comparison plots for offline fault-tolerance sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


STRATEGY_ORDER = ["baseline", "host_backup", "gpu_backup"]
def _load_context_length(results_dir: Path) -> int:
    for strategy in STRATEGY_ORDER:
        jobs_path = results_dir / strategy / "jobs.json"
        if not jobs_path.is_file():
            continue
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
        if jobs:
            return int(jobs[0].get("input_len", 0))
    return 0


def _format_context_length(input_len: int) -> str:
    if input_len >= 1000 and input_len % 1000 == 0:
        return f"{input_len // 1000}k"
    return str(input_len)


def _load_scenario_concurrency(repo_root: Path | None = None) -> int:
    root = repo_root or Path(__file__).resolve().parents[2]
    client_path = root / "benchmark/fault_tolerance/config/scenario/client.json"
    if not client_path.is_file():
        return 16
    data = json.loads(client_path.read_text(encoding="utf-8"))
    return int(data.get("fixed", {}).get("app_workers", 16))


STRATEGY_LABELS = {
    "baseline": "Baseline (L1+L2 HiCache)",
    "host_backup": "Host Backup",
    "gpu_backup": "GPU Backup",
}
CACHE_STRATEGY_LABELS = {
    "baseline": "Baseline\n(Without Backup)",
    "host_backup": "Host Backup",
    "gpu_backup": "GPU Backup",
}
STRATEGY_COLORS = {
    "baseline": "#c0392b",
    "host_backup": "#2471a3",
    "gpu_backup": "#27ae60",
}
REUSE_COLOR = "#9ecae9"
RECOMPUTE_COLOR = "#d9d9d9"
RESTORE_COLOR = "#f39c12"

LATENCY_SEGMENT_COLORS = {
    "queue": "#E0E0E0",
    "prefill": "#4C72B0",
    "decode": "#55A868",
    "load_backup": "#DD8452",
    "recompute": "#CBB682",
    "decode_after": "#2CA02C",
}
LATENCY_SEGMENT_LABELS = {
    "prefill": "Prefill",
    "decode": "Decode",
    "load_backup": "Load Backup",
    "recompute": "Recompute",
    "decode_after": "Decode (after failover)",
}
LEGEND_SEGMENT_ORDER = ["prefill", "decode", "load_backup", "recompute", "decode_after"]
FAULT_STRATEGY_TITLES = {
    "baseline": "Baseline (Without Backup)",
    "host_backup": "Host Backup",
    "gpu_backup": "GPU Backup",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 240,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.axisbelow": True,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_strategy_metrics(results_dir: Path, strategy: str) -> dict:
    metrics_path = results_dir / strategy / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def affected_request_stats(results_dir: Path, strategy: str) -> dict[str, float]:
    cache_rows = read_csv(results_dir / strategy / "cache_hits.csv")
    task_rows = {row["job_id"]: row for row in read_csv(results_dir / strategy / "task_summary.csv")}
    affected = [row for row in cache_rows if str(row.get("is_failover_retried", "0")) == "1"]
    if not affected:
        return {"n": 0.0, "avg_reuse": 0.0, "avg_recompute": 0.0}

    reuse_vals = []
    recompute_vals = []
    for row in affected:
        job_id = row["job_id"]
        prompt = int(task_rows[job_id]["prompt_tokens"])
        reuse = (
            int(row.get("l2_match") or 0)
            + int(row.get("remote_match") or 0)
            + int(row.get("reused_device") or 0)
        )
        reuse_vals.append(reuse)
        recompute_vals.append(max(prompt - reuse, 0))

    return {
        "n": float(len(affected)),
        "avg_reuse": float(np.mean(reuse_vals)),
        "avg_recompute": float(np.mean(recompute_vals)),
    }


def plot_concurrency_sweep(sweep_dir: Path, output_path: Path, *, title: str = "") -> None:
    apply_style()
    sweep_json = sweep_dir / "concurrency_sweep.json"
    if not sweep_json.is_file():
        raise FileNotFoundError(f"missing concurrency sweep output: {sweep_json}")
    summary = json.loads(sweep_json.read_text(encoding="utf-8"))
    levels = summary["concurrency_levels"]
    ref_tput = summary.get("reference", {}).get("output_throughput_tok_s", 164.67)
    ref_c = summary.get("reference", {}).get("concurrency", 16)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for strategy in STRATEGY_ORDER:
        rows = summary["strategies"].get(strategy, [])
        if not rows:
            continue
        y = [row["output_throughput"] for row in rows]
        ax.plot(
            levels[: len(y)],
            y,
            marker="o",
            linewidth=2.0,
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
        )
    ax.axhline(
        ref_tput,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"with_hicache_sglang @ c={ref_c}",
    )
    ax.set_xlabel("Client concurrency")
    ax.set_ylabel("Output throughput (tok/s)")
    ax.set_title("Steady-state throughput vs concurrency")
    ax.legend(framealpha=0.95, fontsize=7.5)

    default_title = (
        "SGLSim Fault-Tolerance — steady-state sweep (no fault, dp=2)\n"
        "Calibrated to with_hicache_sglang (~165 tok/s @ c=16); GPU backup halves effective KV budget under load"
    )
    fig.suptitle(title or default_title, fontsize=11, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(suffix))
    plt.close(fig)
    print(f"saved {output_path.with_suffix('.png')}")


def plot_cache_hit_comparison(results_dir: Path, output_path: Path, *, title: str = "") -> None:
    apply_style()
    metrics = {name: load_strategy_metrics(results_dir, name) for name in STRATEGY_ORDER}
    labels = [CACHE_STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
    colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
    concurrency = _load_scenario_concurrency()
    x = np.arange(len(STRATEGY_ORDER))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    prefill = [100.0 * metrics[s].get("mean_prefill_cache_hit_rate", 0.0) for s in STRATEGY_ORDER]
    failover = [100.0 * metrics[s].get("mean_failover_cache_hit_rate", 0.0) for s in STRATEGY_ORDER]
    ax.bar(x - width / 2, prefill, width, color=colors, edgecolor="black", linewidth=0.5, label="Common KV Reuse")
    ax.bar(x + width / 2, failover, width, color=colors, edgecolor="black", linewidth=0.5, alpha=0.55, label="Failover KV Reuse")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("(a) Common vs failover KV reuse")
    ax.legend(framealpha=0.95)
    for idx, val in enumerate(prefill):
        ax.text(x[idx] - width / 2, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
    for idx, val in enumerate(failover):
        if val > 0:
            ax.text(x[idx] + width / 2, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax = axes[1]
    tier_labels = ["L1 prefix", "Host backup", "Peer GPU"]
    tier_colors = ["#8c564b", REUSE_COLOR, "#27ae60"]
    tier_data = []
    for strategy in STRATEGY_ORDER:
        cache_rows = [r for r in read_csv(results_dir / strategy / "cache_hits.csv") if str(r.get("is_failover_retried", "0")) == "1"]
        if not cache_rows:
            tier_data.append([0, 0, 0])
            continue
        tier_data.append(
            [
                np.mean([int(r.get("l1_match") or 0) for r in cache_rows]),
                np.mean([int(r.get("l2_match") or 0) + int(r.get("reused_host") or 0) for r in cache_rows]),
                np.mean([int(r.get("reused_device") or 0) for r in cache_rows]),
            ]
        )
    bottom = np.zeros(len(STRATEGY_ORDER))
    for tier_idx, tier_name in enumerate(tier_labels):
        vals = [tier_data[sidx][tier_idx] for sidx in range(len(STRATEGY_ORDER))]
        ax.bar(x, vals, width=0.62, bottom=bottom, color=tier_colors[tier_idx], edgecolor="black", linewidth=0.4, label=tier_name)
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg. reusable tokens / affected request")
    ax.set_title("(b) Failover recovery tier breakdown (GPU KV lost on failed rank)")
    ax.legend(framealpha=0.95, ncol=2)

    default_title = (
        f"SGLSim Fault-Tolerance — cache under rank failure + reroute (concurrency={concurrency})\n"
        "Failed rank GPU KV cleared; healthy rank retries from L1/L2, host backup, or peer GPU"
    )
    fig.suptitle(title or default_title, fontsize=11, y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(suffix))
    plt.close(fig)
    print(f"saved {output_path.with_suffix('.png')}")


def load_strategy_jobs(results_dir: Path, strategy: str) -> list[dict]:
    jobs_path = results_dir / strategy / "jobs.json"
    if jobs_path.is_file():
        return json.loads(jobs_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"missing {jobs_path}; re-run offline benchmark to generate per-request latency segments"
    )


def _stack_segments(job: dict) -> list[tuple[str, float]]:
    seg = job.get("latency_segments") or {}
    if not job.get("affected"):
        order = [
            ("prefill", seg.get("prefill_s", 0.0)),
            ("decode", seg.get("decode_s", 0.0)),
        ]
    else:
        order = [
            ("prefill", seg.get("prefill_before_s", 0.0)),
            ("decode", seg.get("decode_before_s", 0.0)),
            ("load_backup", seg.get("load_backup_s", 0.0)),
            ("recompute", seg.get("recompute_after_s", 0.0)),
            ("decode_after", seg.get("decode_after_s", 0.0)),
        ]
    return [(name, float(value)) for name, value in order if float(value) > 1e-6]


def _service_latency(job: dict) -> float:
    segments = _stack_segments(job)
    return sum(value for _, value in segments)


def _draw_request_latency_panel(ax, jobs: list[dict], *, strategy: str) -> None:
    jobs = sorted(jobs, key=lambda item: int(item["job_id"]))
    x_positions = np.array([int(job["job_id"]) for job in jobs])
    bar_width = 0.72
    max_latency = max(_service_latency(job) for job in jobs)

    for x, job in zip(x_positions, jobs):
        bottom = 0.0
        segments = _stack_segments(job)
        service_total = sum(value for _, value in segments)
        for seg_name, height in segments:
            ax.bar(
                x,
                height,
                bottom=bottom,
                width=bar_width,
                color=LATENCY_SEGMENT_COLORS[seg_name],
                edgecolor="black",
                linewidth=0.35,
            )
            bottom += height

        if job.get("affected"):
            ax.bar(
                x,
                service_total,
                width=bar_width,
                fill=False,
                edgecolor="#c0392b",
                linewidth=1.2,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(x) for x in x_positions], fontsize=7)
    ax.set_xlabel("Request ID")
    ax.set_ylabel("Latency (s)")
    ax.set_title(FAULT_STRATEGY_TITLES.get(strategy, strategy))
    ax.set_ylim(0.0, max_latency * 1.08)


def plot_comparison(results_dir: Path, output_path: Path, *, title: str = "") -> None:
    apply_style()
    concurrency = _load_scenario_concurrency()
    context_len = _load_context_length(results_dir)
    context_label = _format_context_length(context_len) if context_len else "?"
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 11.5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.10, top=0.86, bottom=0.06, left=0.08, right=0.98)

    legend_handles = [
        mpatches.Patch(
            facecolor=LATENCY_SEGMENT_COLORS[key],
            edgecolor="black",
            linewidth=0.4,
            label=LATENCY_SEGMENT_LABELS[key],
        )
        for key in LEGEND_SEGMENT_ORDER
    ]
    legend_handles.append(
        mpatches.Patch(facecolor="none", edgecolor="#c0392b", linewidth=1.2, label="Failover-affected request")
    )

    max_latency = 0.0
    for strategy in STRATEGY_ORDER:
        jobs = load_strategy_jobs(results_dir, strategy)
        max_latency = max(max_latency, max(_service_latency(job) for job in jobs))

    for idx, (ax, strategy) in enumerate(zip(axes, STRATEGY_ORDER)):
        jobs = load_strategy_jobs(results_dir, strategy)
        _draw_request_latency_panel(ax, jobs, strategy=strategy)
        ax.set_ylim(0.0, max_latency * 1.08)
        if idx < len(STRATEGY_ORDER) - 1:
            ax.set_xlabel("")

    default_title = (
        f"SGLSim Fault-Tolerance — per-request service latency "
        f"(concurrency={concurrency}, context length={context_label})\n"
        "Single-request compute phases only (excludes queue); failover-affected requests marked with red outline"
    )
    fig.suptitle(title or default_title, fontsize=11, y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        framealpha=0.95,
        fontsize=7.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_path.with_suffix(suffix))
    plt.close(fig)
    print(f"saved {output_path.with_suffix('.png')}")
    print(f"saved {output_path.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot offline fault-tolerance comparison.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument(
        "--mode",
        choices=["fault", "concurrency", "cache"],
        default="fault",
        help="fault: failover panels; concurrency: throughput/cache vs concurrency; cache: cache hit breakdown",
    )
    args = parser.parse_args()
    if args.mode == "concurrency":
        plot_concurrency_sweep(args.results_dir, args.output, title=args.title)
    elif args.mode == "cache":
        plot_cache_hit_comparison(args.results_dir, args.output, title=args.title)
    else:
        plot_comparison(args.results_dir, args.output, title=args.title)


if __name__ == "__main__":
    main()
