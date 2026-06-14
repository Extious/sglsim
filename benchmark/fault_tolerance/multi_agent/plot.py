#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


STRATEGY_LABELS = {
    "baseline": "Baseline",
    "naive_backup": "NaiveBackup",
    "aegiskv": "AegisKV",
}
STRATEGIES = ("baseline", "naive_backup", "aegiskv")
COLORS = {
    "baseline": "#c0392b",
    "naive_backup": "#2471a3",
    "aegiskv": "#27ae60",
    "reuse": "#4C78A8",
    "suffix": "#F58518",
    "recompute": "#BAB0AC",
    "recovery": "#e74c3c",
    "normal": "#4C78A8",
}
TIMELINE_ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("O0", "workflow_creation"),
    ("S1", "round1_fanout"),
    ("S2", "round1_fanout"),
    ("S3", "round1_fanout"),
    ("O0", "round1_fanin"),
    ("S4", "round2_fanout"),
    ("S5", "round2_fanout"),
    ("S6", "round2_fanout"),
    ("O0", "round2_fanin"),
)
TIMELINE_ROW_LABELS: tuple[str, ...] = (
    "O0",
    "S1",
    "S2",
    "S3",
    "O0 (fan-in)",
    "S4",
    "S5",
    "S6",
    "O0 (final)",
)
CRITICAL_AGENTS = {"O0", "S2", "S5"}
PRIMARY_FAILURE_SITE = "S5_ToolBlocked"


def _timeline_row_key(row: dict[str, str]) -> tuple[str, str]:
    return row["agent_id"], row["phase_id"]


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


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.array([]), np.array([])
    sorted_vals = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    return sorted_vals, y


def plot_wct_cdf(workflows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [
        row
        for row in workflows
        if row["experiment"] == "strategy"
        and row["failure_site"] == PRIMARY_FAILURE_SITE
        and row.get("impacted_by_failure") == "1"
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for strategy in STRATEGIES:
        values = [f(row, "wct_s") for row in selected if row["strategy"] == strategy]
        if not values:
            continue
        x, y = _ecdf(values)
        med = median(values)
        ax.plot(
            x,
            y,
            linewidth=1.8,
            label=f"{STRATEGY_LABELS[strategy]} (p50={med:.1f}s)",
            color=COLORS[strategy],
        )
        ax.axvline(med, color=COLORS[strategy], linestyle="--", alpha=0.35, linewidth=1.0)
    ax.set_xlabel("Workflow completion time (s)")
    ax.set_ylabel("CDF")
    ax.set_title("WCT under failure (S5 critical site)")
    ax.legend(loc="lower right")
    save(fig, output / "fig_wct_cdf")


def plot_wct_cdf_by_site(workflows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    sites = [
        "S1_RepoMapper",
        "S2_ToolBlocked",
        "O0_Diagnosis",
        "S4_Patch",
        "S5_ToolBlocked",
        "O0_Final",
    ]
    selected = [row for row in workflows if row["experiment"] == "failure_site"]
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), sharey=True)
    for ax, site in zip(axes.flat, sites):
        for strategy in STRATEGIES:
            values = [
                f(row, "wct_s")
                for row in selected
                if row["strategy"] == strategy and row["failure_site"] == site
            ]
            if not values:
                continue
            x, y = _ecdf(values)
            ax.plot(x, y, linewidth=1.4, color=COLORS[strategy], label=STRATEGY_LABELS[strategy])
        ax.set_title(site, fontsize=8)
        ax.set_xlabel("WCT (s)")
    axes[0, 0].set_ylabel("CDF")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("WCT CDF by failure site", y=1.02)
    save(fig, output / "fig_wct_cdf_by_site")


def _pick_representative_run(timelines: list[dict[str, str]]) -> str:
    candidates = [
        row["run_id"]
        for row in timelines
        if row["experiment"] == "strategy"
        and row["strategy"] == "aegiskv"
        and row["failure_site"] == PRIMARY_FAILURE_SITE
        and row["repeat_idx"] == "0"
    ]
    return candidates[0] if candidates else timelines[0]["run_id"]


def plot_agent_timeline(timelines: list[dict[str, str]], output: Path) -> None:
    apply_style()
    run_id = _pick_representative_run(timelines)
    run_rows = [row for row in timelines if row["run_id"] == run_id]
    workflow_ids = sorted({row["workflow_id"] for row in run_rows})
    impacted = [wf for wf in workflow_ids if any(
        row["workflow_id"] == wf and row.get("impacted_by_failure") == "1"
        for row in run_rows
    )]
    unaffected = [wf for wf in workflow_ids if wf not in impacted]
    pick = []
    if impacted:
        pick.append(impacted[0])
    pick.extend(unaffected[:2])
    if not pick:
        pick = workflow_ids[:3]

    fig, axes = plt.subplots(len(pick), 1, figsize=(7.0, 1.6 * len(pick)), sharex=True)
    if len(pick) == 1:
        axes = [axes]
    row_to_y = {
        key: idx for idx, key in enumerate(reversed(TIMELINE_ROW_ORDER))
    }

    for ax, wf_id in zip(axes, pick):
        wf_rows = [row for row in run_rows if row["workflow_id"] == wf_id]
        for row in wf_rows:
            y = row_to_y.get(_timeline_row_key(row), 0)
            start_s = f(row, "start_s")
            end_s = f(row, "end_s")
            recovery = f(row, "recovery_extra_s")
            is_target = row.get("is_failure_target") == "1"
            is_critical = row.get("is_critical") == "1"
            bar_color = COLORS["normal"]
            if is_critical:
                bar_color = COLORS["aegiskv"] if not is_target else COLORS["recovery"]
            ax.barh(y, end_s - start_s, left=start_s, height=0.55, color=bar_color, alpha=0.85)
            if recovery > 0:
                ax.barh(
                    y,
                    recovery,
                    left=end_s,
                    height=0.55,
                    color=COLORS["recovery"],
                    alpha=0.9,
                    label="recovery" if y == 0 else None,
                )
            if is_target:
                ax.axvline(end_s, color=COLORS["recovery"], linestyle=":", alpha=0.7, linewidth=1.0)
        ax.set_yticks(list(row_to_y.values()))
        ax.set_yticklabels(list(reversed(TIMELINE_ROW_LABELS)))
        impacted_flag = "impacted" if any(
            row.get("impacted_by_failure") == "1" for row in wf_rows
        ) else "unaffected"
        ax.set_title(f"{wf_id} ({impacted_flag})", fontsize=8, loc="left")
    axes[-1].set_xlabel("Time within workflow (s)")
    fig.suptitle(f"Agent timeline (AegisKV, {PRIMARY_FAILURE_SITE})", y=1.01)
    save(fig, output / "fig_agent_timeline")


def plot_agent_timeline_compare(timelines: list[dict[str, str]], output: Path) -> None:
    apply_style()
    impacted_wf = ""
    for row in timelines:
        if (
            row["experiment"] == "strategy"
            and row["failure_site"] == PRIMARY_FAILURE_SITE
            and row.get("impacted_by_failure") == "1"
            and row.get("is_failure_target") == "1"
        ):
            impacted_wf = row["workflow_id"]
            break
    if not impacted_wf:
        return

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8), sharey=True)
    row_to_y = {
        key: idx for idx, key in enumerate(reversed(TIMELINE_ROW_ORDER))
    }
    for ax, strategy in zip(axes, STRATEGIES):
        run_rows = [
            row
            for row in timelines
            if row["experiment"] == "strategy"
            and row["strategy"] == strategy
            and row["failure_site"] == PRIMARY_FAILURE_SITE
            and row["repeat_idx"] == "0"
            and row["workflow_id"] == impacted_wf
        ]
        for row in run_rows:
            y = row_to_y.get(_timeline_row_key(row), 0)
            start_s = f(row, "start_s")
            end_s = f(row, "end_s")
            recovery = f(row, "recovery_extra_s")
            is_target = row.get("is_failure_target") == "1"
            color = COLORS["recovery"] if is_target else COLORS["normal"]
            ax.barh(y, end_s - start_s, left=start_s, height=0.55, color=color, alpha=0.75)
            if recovery > 0:
                ax.barh(y, recovery, left=end_s, height=0.55, color="#922b21", alpha=0.95)
        ax.set_title(STRATEGY_LABELS[strategy])
        ax.set_xlabel("Time (s)")
    axes[0].set_yticks(list(row_to_y.values()))
    axes[0].set_yticklabels(list(reversed(TIMELINE_ROW_LABELS)))
    fig.suptitle(f"Failure impact on {impacted_wf} (S5 target)", y=1.02)
    save(fig, output / "fig_agent_timeline_compare")


def plot_agent_duration_violin(timelines: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [
        row
        for row in timelines
        if row["experiment"] == "strategy" and row["failure_site"] == PRIMARY_FAILURE_SITE
    ]
    agents = ["S1", "S2", "S4", "S5"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    positions: list[float] = []
    data: list[list[float]] = []
    colors: list[str] = []
    pos = 0
    width = 0.22
    for agent in agents:
        for idx, strategy in enumerate(STRATEGIES):
            vals = [
                f(row, "total_duration_s")
                for row in selected
                if row["agent_id"] == agent and row["strategy"] == strategy
            ]
            positions.append(pos + idx * width)
            data.append(vals)
            colors.append(COLORS[strategy])
        pos += 1.0
    parts = ax.violinplot(data, positions=positions, widths=width * 0.9, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)
    tick_pos = [i for i in range(len(agents))]
    labels = []
    for agent in agents:
        star = "*" if agent in CRITICAL_AGENTS else ""
        labels.append(f"{agent}{star}")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Agent duration (s)")
    ax.set_title("Agent duration distribution (S5 failure site)")
    legend_handles = [
        mpl.patches.Patch(color=COLORS[s], label=STRATEGY_LABELS[s]) for s in STRATEGIES
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    save(fig, output / "fig_agent_duration_violin")


def plot_strategy_recovery(rows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in rows if row["experiment"] == "strategy"]
    reuse = group_mean(selected, ("strategy",), "gpu_replica_reuse_tokens")
    suffix = group_mean(selected, ("strategy",), "suffix_replay_tokens")
    recompute = group_mean(selected, ("strategy",), "total_recomputed_tokens")
    x = np.arange(len(STRATEGIES))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    r = np.array([reuse.get((s,), 0.0) for s in STRATEGIES])
    su = np.array([suffix.get((s,), 0.0) for s in STRATEGIES])
    re = np.array([recompute.get((s,), 0.0) for s in STRATEGIES])
    ax.bar(x, r, color=COLORS["reuse"], label="GPUReplica reuse")
    ax.bar(x, su, bottom=r, color=COLORS["suffix"], label="suffix replay")
    ax.bar(x, re, bottom=r + su, color=COLORS["recompute"], label="recompute")
    ax.set_xticks(x, [STRATEGY_LABELS[s] for s in STRATEGIES], rotation=12)
    ax.set_ylabel("Tokens per failure")
    ax.set_title("Recovery token breakdown")
    ax.legend(ncols=1)
    save(fig, output / "fig_strategy_recovery")


def plot_backup_cost(rows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in rows if row["experiment"] == "strategy"]
    backed = group_mean(selected, ("strategy",), "backed_up_tokens")
    overhead = group_mean(selected, ("strategy",), "backup_overhead_s")
    x = np.arange(len(STRATEGIES))
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    vals = [backed.get((s,), 0.0) for s in STRATEGIES]
    ax1.bar(x, vals, color=[COLORS[s] for s in STRATEGIES], alpha=0.85)
    ax1.set_xticks(x, [STRATEGY_LABELS[s] for s in STRATEGIES], rotation=12)
    ax1.set_ylabel("Backed-up tokens")
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        [overhead.get((s,), 0.0) for s in STRATEGIES],
        color="#555555",
        marker="o",
        linewidth=1.5,
        label="backup overhead (s)",
    )
    ax2.set_ylabel("Backup overhead (s)")
    ax1.set_title("Backup cost")
    save(fig, output / "fig_backup_cost")


def plot_recovery_vs_cost(rows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in rows if row["experiment"] == "failure_site"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for strategy in STRATEGIES:
        pts = [row for row in selected if row["strategy"] == strategy]
        if not pts:
            continue
        xs = [f(row, "backed_up_tokens") for row in pts]
        ys = [f(row, "total_recomputed_tokens") for row in pts]
        ax.scatter(xs, ys, color=COLORS[strategy], alpha=0.55, s=28, label=STRATEGY_LABELS[strategy])
        ax.scatter(
            mean(xs),
            mean(ys),
            color=COLORS[strategy],
            s=90,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
        )
    ax.set_xlabel("Backed-up tokens")
    ax.set_ylabel("Recomputed tokens")
    ax.set_title("Recovery vs backup cost")
    ax.legend()
    save(fig, output / "fig_recovery_vs_cost")


def plot_recovery_paths(paths: list[dict[str, str]], output: Path) -> None:
    apply_style()
    selected = [row for row in paths if row["experiment"] == "strategy"]
    path_order = ["full_segment_restore", "prefix_restore_suffix_replay", "no_valid_checkpoint"]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in selected:
        counts[(row["strategy"], row["recovery_path"])] += 1
        totals[row["strategy"]] += 1
    x = np.arange(len(STRATEGIES))
    bottom = np.zeros(len(STRATEGIES))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bar_colors = ["#59A14F", "#F28E2B", "#B0B0B0"]
    labels = ["full restore", "prefix + suffix replay", "no checkpoint"]
    for path, color, label in zip(path_order, bar_colors, labels):
        vals = np.array([
            counts.get((strategy, path), 0) / max(totals.get(strategy, 0), 1)
            for strategy in STRATEGIES
        ])
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_xticks(x, [STRATEGY_LABELS[s] for s in STRATEGIES], rotation=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of recoveries")
    ax.set_title("Recovery path distribution")
    ax.legend(loc="upper left")
    save(fig, output / "fig_recovery_path_distribution")


def plot_scale(rows: list[dict[str, str]], workflows: list[dict[str, str]], output: Path) -> None:
    apply_style()
    run_rows = [row for row in rows if row["experiment"] == "scale"]
    wf_rows = [row for row in workflows if row["experiment"] == "scale"]
    fig, ax1 = plt.subplots(figsize=(5.4, 3.2))
    for strategy in STRATEGIES:
        scales = sorted({
            int(row["num_workflows"])
            for row in run_rows
            if row["strategy"] == strategy
        })
        backed = [
            mean(
                f(row, "backed_up_tokens")
                for row in run_rows
                if row["strategy"] == strategy and int(row["num_workflows"]) == scale
            )
            for scale in scales
        ]
        ax1.plot(scales, backed, marker="o", linewidth=1.5, color=COLORS[strategy], label=f"{STRATEGY_LABELS[strategy]} backup")
    ax1.set_xlabel("Concurrent workflows")
    ax1.set_ylabel("Backed-up tokens")
    ax2 = ax1.twinx()
    for strategy in ("baseline", "aegiskv"):
        scales = sorted({
            int(row["num_workflows"])
            for row in wf_rows
            if row["strategy"] == strategy
        })
        wct_med = [
            median(
                f(row, "wct_s")
                for row in wf_rows
                if row["strategy"] == strategy and int(row["num_workflows"]) == scale
            )
            for scale in scales
        ]
        ax2.plot(
            scales,
            wct_med,
            marker="s",
            linestyle="--",
            linewidth=1.2,
            color=COLORS[strategy],
            alpha=0.7,
            label=f"{STRATEGY_LABELS[strategy]} WCT p50",
        )
    ax1.set_title("Scale: backup cost vs WCT")
    ax1.legend(loc="upper left", fontsize=7)
    save(fig, output / "fig_scale")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot multi-agent AegisKV artifacts")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.results_dir / "runs.csv")
    paths = read_csv(args.results_dir / "recovery_paths.csv")
    workflows = read_csv(args.results_dir / "workflows.csv")
    timelines = read_csv(args.results_dir / "agent_timelines.csv")

    plot_wct_cdf(workflows, args.results_dir)
    plot_wct_cdf_by_site(workflows, args.results_dir)
    plot_agent_timeline(timelines, args.results_dir)
    plot_agent_timeline_compare(timelines, args.results_dir)
    plot_agent_duration_violin(timelines, args.results_dir)
    plot_strategy_recovery(rows, args.results_dir)
    plot_backup_cost(rows, args.results_dir)
    plot_recovery_vs_cost(rows, args.results_dir)
    plot_recovery_paths(paths, args.results_dir)
    plot_scale(rows, workflows, args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
