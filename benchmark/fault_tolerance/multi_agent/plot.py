#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


STRATEGY_LABELS = {
    "baseline": "Baseline",
    "gpu_backup_agent": "Agent GPU backup",
    "aegiskv_segment": "AegisKV segment",
}
COLORS = {
    "baseline": "#c0392b",
    "gpu_backup_agent": "#2471a3",
    "aegiskv_segment": "#27ae60",
    "reuse": "#4C78A8",
    "suffix": "#F58518",
    "recompute": "#BAB0AC",
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
            "figure.dpi": 220,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.axisbelow": True,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def group_mean(rows: list[dict[str, str]], keys: tuple[str, ...], metric: str) -> dict[tuple[str, ...], float]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(f(row, metric))
    return {key: mean(values) for key, values in groups.items()}


def group_stderr(rows: list[dict[str, str]], keys: tuple[str, ...], metric: str) -> dict[tuple[str, ...], float]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(f(row, metric))
    out: dict[tuple[str, ...], float] = {}
    for key, values in groups.items():
        if len(values) <= 1:
            out[key] = 0.0
            continue
        avg = mean(values)
        var = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
        out[key] = (var ** 0.5) / (len(values) ** 0.5)
    return out


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_strategy_recovery(rows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in rows if row["experiment"] == "strategy"]
    strategies = ["baseline", "gpu_backup_agent", "aegiskv_segment"]
    reuse = group_mean(selected, ("strategy",), "gpu_replica_reuse_tokens")
    suffix = group_mean(selected, ("strategy",), "suffix_replay_tokens")
    recompute = group_mean(selected, ("strategy",), "total_recomputed_tokens")
    x = np.arange(len(strategies))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    r = np.array([reuse.get((s,), 0.0) for s in strategies])
    su = np.array([suffix.get((s,), 0.0) for s in strategies])
    re = np.array([recompute.get((s,), 0.0) for s in strategies])
    ax.bar(x, r, color=COLORS["reuse"], label="GPUReplica reuse")
    ax.bar(x, su, bottom=r, color=COLORS["suffix"], label="suffix replay")
    ax.bar(x, re, bottom=r + su, color=COLORS["recompute"], label="recompute")
    ax.set_xticks(x, [STRATEGY_LABELS[s] for s in strategies], rotation=12)
    ax.set_ylabel("Tokens per failure")
    ax.set_title("Segment recovery under failures")
    ax.legend(ncols=1)
    save(fig, output / "fig_strategy_recovery")


def plot_wct_failure_site(rows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in rows if row["experiment"] == "failure_site"]
    sites = [
        "E1_RepoMapper",
        "E2_ToolBlocked",
        "O0_J1_Diagnosis",
        "F1_Patch",
        "F2_ToolBlocked",
        "O0_J2_Final",
    ]
    wct = group_mean(selected, ("failure_site",), "workflow_completion_time_s")
    err = group_stderr(selected, ("failure_site",), "workflow_completion_time_s")
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.errorbar(
        np.arange(len(sites)),
        [wct.get((site,), 0.0) for site in sites],
        yerr=[err.get((site,), 0.0) for site in sites],
        marker="o",
        capsize=2.5,
        linewidth=1.5,
        color=COLORS["aegiskv_segment"],
    )
    ax.set_xticks(np.arange(len(sites)), sites, rotation=25, ha="right")
    ax.set_ylabel("Workflow completion time (s)")
    ax.set_title("Failure-site sensitivity")
    save(fig, output / "fig_wct_failure_site")


def plot_budget_efficiency(rows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in rows if row["experiment"] == "budget"]
    groups = group_mean(selected, ("strategy", "backup_budget_ratio"), "critical_segment_survival_rate")
    errs = group_stderr(selected, ("strategy", "backup_budget_ratio"), "critical_segment_survival_rate")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for strategy in ("gpu_backup_agent", "aegiskv_segment"):
        budgets = sorted(float(key[1]) for key in groups if key[0] == strategy)
        vals = [groups[(strategy, f"{budget:g}")] if (strategy, f"{budget:g}") in groups else groups[(strategy, str(budget))] for budget in budgets]
        yerr = [
            errs.get((strategy, f"{budget:g}"), errs.get((strategy, str(budget)), 0.0))
            for budget in budgets
        ]
        ax.errorbar(
            budgets,
            vals,
            yerr=yerr,
            marker="o",
            capsize=2.5,
            linewidth=1.5,
            label=STRATEGY_LABELS[strategy],
            color=COLORS[strategy],
        )
    ax.set_xlabel("Backup budget ratio")
    ax.set_ylabel("Critical segment survival")
    ax.set_ylim(0, 1.05)
    ax.set_title("Budget efficiency")
    ax.legend()
    save(fig, output / "fig_budget_efficiency")


def plot_backup_lag(rows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in rows if row["experiment"] == "lag"]
    reuse = group_mean(selected, ("peer_backup_lag_tokens",), "kv_reuse_rate")
    reuse_err = group_stderr(selected, ("peer_backup_lag_tokens",), "kv_reuse_rate")
    suffix = group_mean(selected, ("peer_backup_lag_tokens",), "suffix_replay_tokens")
    suffix_err = group_stderr(selected, ("peer_backup_lag_tokens",), "suffix_replay_tokens")
    lags = sorted(int(key[0]) for key in reuse)
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    coverage_handle = ax1.errorbar(
        lags,
        [reuse[(str(lag),)] for lag in lags],
        yerr=[reuse_err.get((str(lag),), 0.0) for lag in lags],
        marker="o",
        capsize=2.5,
        linewidth=1.5,
        color=COLORS["reuse"],
        label="Checkpoint coverage",
    )
    ax1.set_xlabel("Peer backup lag (tokens)")
    ax1.set_ylabel("Checkpoint coverage")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    suffix_handle = ax2.errorbar(
        lags,
        [suffix[(str(lag),)] for lag in lags],
        yerr=[suffix_err.get((str(lag),), 0.0) for lag in lags],
        marker="s",
        capsize=2.5,
        linewidth=1.5,
        color=COLORS["suffix"],
        label="suffix replay",
    )
    ax2.set_ylabel("Suffix replay tokens")
    ax1.set_title("GPUReplica lag sensitivity")
    ax1.legend(
        [coverage_handle, suffix_handle],
        ["Checkpoint coverage", "suffix replay"],
        loc="center right",
    )
    save(fig, output / "fig_backup_lag")


def plot_recovery_paths(paths: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in paths if row["experiment"] == "strategy"]
    strategies = ["baseline", "gpu_backup_agent", "aegiskv_segment"]
    path_order = ["full_segment_restore", "prefix_restore_suffix_replay", "no_valid_checkpoint"]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in selected:
        counts[(row["strategy"], row["recovery_path"])] += 1
        totals[row["strategy"]] += 1
    x = np.arange(len(strategies))
    bottom = np.zeros(len(strategies))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    colors = ["#59A14F", "#F28E2B", "#B0B0B0"]
    labels = ["full restore", "prefix + suffix replay", "no checkpoint"]
    for path, color, label in zip(path_order, colors, labels):
        vals = np.array([
            counts.get((strategy, path), 0) / max(totals.get(strategy, 0), 1)
            for strategy in strategies
        ])
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_xticks(x, [STRATEGY_LABELS[s] for s in strategies], rotation=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of recoveries")
    ax.set_title("Recovery path distribution")
    ax.legend(loc="upper left")
    save(fig, output / "fig_recovery_path_distribution")


def plot_scale(rows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in rows if row["experiment"] == "scale"]
    groups = group_mean(selected, ("strategy", "num_workflows"), "total_recomputed_tokens")
    errs = group_stderr(selected, ("strategy", "num_workflows"), "total_recomputed_tokens")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for strategy in ("baseline", "aegiskv_segment"):
        scales = sorted(int(key[1]) for key in groups if key[0] == strategy)
        ax.errorbar(
            scales,
            [groups[(strategy, str(scale))] for scale in scales],
            yerr=[errs.get((strategy, str(scale)), 0.0) for scale in scales],
            marker="o",
            capsize=2.5,
            linewidth=1.5,
            label=STRATEGY_LABELS[strategy],
            color=COLORS[strategy],
        )
    ax.set_xlabel("Concurrent workflows")
    ax.set_ylabel("Recomputed tokens")
    ax.set_title("Scale sensitivity")
    ax.legend()
    save(fig, output / "fig_scale")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot multi-agent AegisKV artifacts")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.results_dir / "runs.csv")
    paths = read_csv(args.results_dir / "recovery_paths.csv")
    plot_strategy_recovery(rows, args.results_dir)
    plot_wct_failure_site(rows, args.results_dir)
    plot_budget_efficiency(rows, args.results_dir)
    plot_backup_lag(rows, args.results_dir)
    plot_recovery_paths(paths, args.results_dir)
    plot_scale(rows, args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
