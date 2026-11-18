"""Render S1--S9 without consulting archived or frozen preliminary CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "recompute": "#6B7280",
    "host_checkpoint": "#D97706",
    "neighbor_replica": "#2563EB",
    "aegiskv": "#059669",
    "aegiskv_no_slack": "#7C3AED",
    "aegiskv_no_aimd": "#DB2777",
    "aegiskv_dram_only": "#0891B2",
}
LABELS = {
    "recompute": "Recompute",
    "host_checkpoint": "Host checkpoint adapter",
    "neighbor_replica": "Neighbor-copy adapter",
    "aegiskv": "AegisKV",
    "aegiskv_no_slack": "No causal priority",
    "aegiskv_no_aimd": "No AIMD",
    "aegiskv_dram_only": "No peer HBM",
}

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    24: 2.064,
    29: 2.045,
}


def _t_critical_95(count: int) -> float:
    degrees = max(int(count) - 1, 1)
    if degrees in T_CRITICAL_95:
        return T_CRITICAL_95[degrees]
    smaller = [value for value in T_CRITICAL_95 if value <= degrees]
    return T_CRITICAL_95[max(smaller)] if degrees < 30 and smaller else 1.96


def _aggregate_ci95(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric: str,
    *,
    nonnegative: bool = True,
) -> pd.DataFrame:
    aggregate = (
        frame.groupby(group_columns, observed=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate["ci95_half_width"] = [
        0.0
        if int(count) <= 1 or not np.isfinite(std)
        else _t_critical_95(int(count)) * float(std) / np.sqrt(int(count))
        for std, count in zip(aggregate["std"], aggregate["count"])
    ]
    aggregate["ci95_low"] = aggregate["mean"] - aggregate["ci95_half_width"]
    aggregate["ci95_high"] = aggregate["mean"] + aggregate["ci95_half_width"]
    if nonnegative:
        aggregate["ci95_low"] = aggregate["ci95_low"].clip(lower=0.0)
    return aggregate


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
        }
    )


def _save(fig: plt.Figure, figure_dir: Path, slug: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [figure_dir / f"{slug}.pdf", figure_dir / f"{slug}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight", dpi=220)
    plt.close(fig)
    return paths


def _source(df: pd.DataFrame, source_dir: Path, slug: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(source_dir / f"{slug}.csv", index=False)


def _s1(summary: pd.DataFrame, obs: pd.DataFrame, *_: Any) -> tuple[plt.Figure, pd.DataFrame]:
    frame = obs[(obs.group == "S1") & (obs.failure_added_wct_ms > 1e-9)].copy()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    curves: list[tuple[np.ndarray, list[str]]] = []
    for policy in frame.policy.drop_duplicates():
        values = np.sort(
            frame.loc[frame.policy == policy, "failure_added_wct_ms"].to_numpy()
        )
        matching = next(
            (
                curve
                for curve in curves
                if len(curve[0]) == len(values)
                and np.allclose(curve[0], values, rtol=0.0, atol=1e-9)
            ),
            None,
        )
        if matching is not None:
            matching[1].append(str(policy))
        else:
            curves.append((values, [str(policy)]))
    for values, policies in curves:
        y = np.arange(1, len(values) + 1) / max(len(values), 1)
        representative = "aegiskv" if "aegiskv" in policies else policies[0]
        short_labels = {
            "host_checkpoint": "Host",
            "neighbor_replica": "Neighbor",
            "aegiskv": "AegisKV",
        }
        label = " / ".join(
            short_labels.get(policy, LABELS.get(policy, policy)) for policy in policies
        )
        if len(policies) > 1:
            label += " (tied)"
        ax.step(
            values,
            y,
            where="post",
            label=label,
            color=COLORS.get(representative),
            linewidth=1.6,
        )
    ax.set_xlabel("Failure-added WCT among harmed sessions (ms)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(axis="both", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Conditional one-replica failure")
    return fig, frame


def _s2(summary: pd.DataFrame, *_: Any) -> tuple[plt.Figure, pd.DataFrame]:
    frame = summary[summary.group == "S2"].copy()
    agg = _aggregate_ci95(
        frame,
        ["policy", "failure_scenario", "load_multiplier"],
        "affected_failure_added_wct_p95_ms",
    )
    scenarios = list(agg.failure_scenario.drop_duplicates())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.9, 2.25), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, scenarios):
        part = agg[agg.failure_scenario == scenario]
        for policy in part.policy.drop_duplicates():
            row = part[part.policy == policy].sort_values("load_multiplier")
            ax.plot(row.load_multiplier, row["mean"], marker="o", label=LABELS.get(policy, policy), color=COLORS.get(policy))
            ax.fill_between(
                row.load_multiplier,
                row["ci95_low"],
                row["ci95_high"],
                color=COLORS.get(policy),
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("Load multiplier")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("p95 harmed-session added WCT (ms)")
    axes[-1].legend(frameon=False, title="Mean ± 95% CI")
    return fig, agg


def _s3(summary: pd.DataFrame, *_: Any) -> tuple[plt.Figure, pd.DataFrame]:
    frame = summary[summary.group == "S3"].copy()
    agg = _aggregate_ci95(
        frame,
        ["model", "context_tokens"],
        "affected_failure_added_wct_p95_ms",
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for model in agg.model.drop_duplicates():
        row = agg[agg.model == model].sort_values("context_tokens")
        x = row.context_tokens / 1024.0
        ax.errorbar(
            x,
            row["mean"],
            yerr=np.vstack(
                (
                    row["mean"] - row["ci95_low"],
                    row["ci95_high"] - row["mean"],
                )
            ),
            marker="o",
            capsize=2,
            label=model,
        )
    ax.set_xlabel("Projected context cap (K tokens)")
    ax.set_ylabel("p95 harmed-session added WCT (ms)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return fig, agg


def _bar_with_seed_points(ax: plt.Axes, frame: pd.DataFrame, order: list[str], metric: str) -> None:
    means = [frame.loc[frame.policy == policy, metric].mean() for policy in order]
    positions = np.arange(len(order))
    ax.bar(positions, means, color=[COLORS.get(policy, "#64748B") for policy in order], width=0.68)
    for index, policy in enumerate(order):
        values = frame.loc[frame.policy == policy, metric].to_numpy()
        ax.scatter(np.full(len(values), index), values, s=9, facecolors="none", edgecolors="#111827", linewidths=0.5)
    ax.set_xticks(positions, [LABELS.get(policy, policy) for policy in order], rotation=18, ha="right")


def _s4(summary: pd.DataFrame, obs: pd.DataFrame, timeline: pd.DataFrame, verification: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    frame = summary[summary.group == "S4"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))
    priority_order = ["aegiskv", "aegiskv_no_slack"]
    _bar_with_seed_points(
        axes[0], frame, priority_order, "recovery_deadline_miss_max_ms"
    )
    axes[0].set_ylabel("Maximum recovery lateness (ms)")
    axes[0].set_title("(a) Causal recovery priority")
    axes[0].grid(axis="y", alpha=0.2)

    alpha = timeline[
        (timeline.group == "S4")
        & timeline.policy.isin(["aegiskv", "aegiskv_no_aimd"])
    ].copy()
    alpha_by_seed = (
        alpha.groupby(["policy", "seed"], as_index=False)["mean_alpha"].mean()
    )
    _bar_with_seed_points(
        axes[1], alpha_by_seed, ["aegiskv", "aegiskv_no_aimd"], "mean_alpha"
    )
    axes[1].set_ylabel("Mean peer-HBM quota α")
    axes[1].set_ylim(0, 0.65)
    axes[1].set_title("(b) HBMQuotaAIMD under pressure")
    axes[1].grid(axis="y", alpha=0.2)
    source = frame.merge(
        alpha_by_seed.rename(columns={"mean_alpha": "timeline_mean_alpha"}),
        on=["policy", "seed"],
        how="left",
    )
    return fig, source


def _s5(summary: pd.DataFrame, obs: pd.DataFrame, timeline: pd.DataFrame, *_: Any) -> tuple[plt.Figure, pd.DataFrame]:
    frame = timeline[timeline.group == "S5"].copy()
    grid = np.linspace(0.0, 1.0, 21)
    normalized_rows: list[dict[str, float | int]] = []
    for (cell_id, seed, load), cell in frame.groupby(
        ["cell_id", "seed", "load_multiplier"]
    ):
        cell = cell.sort_values("time_ms")
        final_time = max(float(cell.time_ms.max()), 1e-9)
        progress = cell.time_ms.to_numpy(dtype=float) / final_time
        alpha = cell.mean_alpha.to_numpy(dtype=float)
        for point, value in zip(grid, np.interp(grid, progress, alpha)):
            normalized_rows.append(
                {
                    "cell_id": str(cell_id),
                    "seed": int(seed),
                    "load_multiplier": float(load),
                    "normalized_progress": float(point),
                    "mean_alpha": float(value),
                }
            )
    normalized = pd.DataFrame(normalized_rows)
    aggregate = _aggregate_ci95(
        normalized,
        ["load_multiplier", "normalized_progress"],
        "mean_alpha",
        nonnegative=True,
    )
    fig, ax = plt.subplots(figsize=(5.0, 2.35))
    for load in aggregate.load_multiplier.drop_duplicates():
        row = aggregate[aggregate.load_multiplier == load].sort_values(
            "normalized_progress"
        )
        line = ax.plot(
            row.normalized_progress,
            row["mean"],
            label=f"load {load:g}",
        )[0]
        ax.fill_between(
            row.normalized_progress,
            row.ci95_low,
            row.ci95_high,
            color=line.get_color(),
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlabel("Normalized simulation progress")
    ax.set_ylabel("Mean peer-HBM quota α")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, title="Mean ± 95% CI")
    return fig, aggregate


def _s6(summary: pd.DataFrame, obs: pd.DataFrame, *_: Any) -> tuple[plt.Figure, pd.DataFrame]:
    frame = obs[obs.group == "S6"].copy()
    bins = [-0.001, 0.90, 0.99, 1.001]
    labels = ["<90%", "90-99%", "≥99%"]
    frame["reuse_bin"] = pd.cut(frame.mean_reuse_fraction, bins=bins, labels=labels)
    paired = frame.pivot_table(index=["seed", "workflow_id", "reuse_bin"], columns="policy", values="wct_ms", observed=True).reset_index()
    paired["aegiskv_minus_recompute_ms"] = paired["aegiskv"] - paired["recompute"]
    per_seed = (
        paired.groupby(["seed", "reuse_bin"], observed=True, as_index=False)[
            "aegiskv_minus_recompute_ms"
        ].mean()
    )
    agg = _aggregate_ci95(
        per_seed,
        ["reuse_bin"],
        "aegiskv_minus_recompute_ms",
        nonnegative=False,
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    bars = ax.bar(
        agg.reuse_bin.astype(str),
        agg["mean"],
        yerr=np.vstack(
            (
                agg["mean"] - agg.ci95_low,
                agg.ci95_high - agg["mean"],
            )
        ),
        capsize=2,
        color=COLORS["aegiskv"],
    )
    for bar, value in zip(bars, agg["mean"]):
        annotation = {
            "xy": (
                bar.get_x() + bar.get_width() / 2.0,
                value / 2.0 if value < 0 else 0.0,
            ),
            "ha": "center",
            "va": "center" if value < 0 else "top",
            "color": "white" if value < 0 else "#111827",
            "fontsize": 7,
        }
        if value >= 0:
            annotation.update({"xytext": (0, -3), "textcoords": "offset points"})
        ax.annotate(f"{value:.0f}", **annotation)
    ax.axhline(0, color="#111827", linewidth=0.7)
    ax.set_xlabel("Projected cached-prefix fraction")
    ax.set_ylabel("AegisKV − recompute WCT (ms)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Paired seed means ± 95% CI")
    return fig, agg


def _s7(summary: pd.DataFrame, *_: Any) -> tuple[plt.Figure, pd.DataFrame]:
    frame = summary[summary.group == "S7"].copy()
    agg = _aggregate_ci95(
        frame,
        ["effective_bandwidth_GB_per_s", "repair_time_ms"],
        "failure_added_wct_p95_ms",
    )
    pivot = agg.pivot(
        index="repair_time_ms", columns="effective_bandwidth_GB_per_s", values="mean"
    )
    deviation = agg.pivot(
        index="repair_time_ms",
        columns="effective_bandwidth_GB_per_s",
        values="ci95_half_width",
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{x/1000:g}" for x in pivot.index])
    ax.set_xlabel("Effective transfer bandwidth (GB/s)")
    ax.set_ylabel("Repair duration (s)")
    ax.set_title("Cells: mean ± 95% CI across seeds")
    for y in range(len(pivot.index)):
        for x in range(len(pivot.columns)):
            ax.text(x, y, f"{pivot.iloc[y, x]:.0f}\n±{deviation.iloc[y, x]:.0f}", ha="center", va="center", color="white" if pivot.iloc[y, x] > np.nanmedian(pivot.to_numpy()) else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean p95 added WCT (ms)")
    return fig, agg


def _s8(summary: pd.DataFrame, *_: Any) -> tuple[plt.Figure, pd.DataFrame]:
    frame = summary[summary.group == "S8"].copy()
    agg = _aggregate_ci95(
        frame,
        ["failure_scenario", "policy"],
        "failure_added_wct_p95_ms",
    )
    scenarios = list(agg.failure_scenario.drop_duplicates())
    policies = list(agg.policy.drop_duplicates())
    x = np.arange(len(scenarios))
    width = 0.75 / len(policies)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for index, policy in enumerate(policies):
        rows = [
            agg[(agg.failure_scenario == scenario) & (agg.policy == policy)].iloc[0]
            for scenario in scenarios
        ]
        values = [float(row["mean"]) for row in rows]
        errors = np.vstack(
            (
                [float(row["mean"] - row["ci95_low"]) for row in rows],
                [float(row["ci95_high"] - row["mean"]) for row in rows],
            )
        )
        positions = x + (index - (len(policies)-1)/2)*width
        bars = ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=2,
            label=LABELS.get(policy, policy),
            color=COLORS.get(policy),
        )
        for position, scenario, bar, value in zip(positions, scenarios, bars, values):
            seed_values = frame[
                (frame.failure_scenario == scenario) & (frame.policy == policy)
            ].failure_added_wct_p95_ms.to_numpy()
            ax.scatter(
                np.full(len(seed_values), position),
                seed_values,
                s=9,
                facecolors="none",
                edgecolors="#111827",
                linewidths=0.5,
            )
            ax.annotate(
                f"{value:.0f}",
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, [s.replace("_", " ") for s in scenarios])
    ax.set_ylabel("p95 failure-added WCT (ms)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, title="Mean ± 95% CI")
    return fig, agg


def _s9(summary: pd.DataFrame, obs: pd.DataFrame, timeline: pd.DataFrame, verification: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    frame = verification[verification.group == "S9"].copy()
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for model in frame.model.drop_duplicates():
        row = frame[frame.model == model]
        ax.scatter(row.analytical_recovery_ms, row.simulated_recovery_ms, s=16, alpha=0.65, label=model)
    low = min(frame.analytical_recovery_ms.min(), frame.simulated_recovery_ms.min())
    high = max(frame.analytical_recovery_ms.max(), frame.simulated_recovery_ms.max())
    ax.plot([low, high], [low, high], linestyle="--", color="#111827", linewidth=0.8, label="y=x")
    ax.set_xlabel("Analytical recovery time (ms)")
    ax.set_ylabel("DES recovery time (ms)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title("Model verification, not A100 calibration")
    return fig, frame


def _e1(
    summary: pd.DataFrame,
    obs: pd.DataFrame,
    *_: Any,
) -> tuple[plt.Figure, pd.DataFrame]:
    frame = summary[summary.group == "E1"].copy()
    observation = obs[obs.group == "E1"].copy()
    policy_order = ["recompute", "aegiskv"]
    aggregate = _aggregate_ci95(
        frame,
        ["policy"],
        "affected_failure_added_wct_mean_ms",
    )
    paired = observation.pivot_table(
        index=["seed", "workflow_id"],
        columns="policy",
        values="failure_added_wct_ms",
    ).reset_index()
    paired["aegiskv_minus_recompute_ms"] = (
        paired["aegiskv"] - paired["recompute"]
    )
    per_seed = paired.groupby("seed", as_index=False)[
        "aegiskv_minus_recompute_ms"
    ].mean()
    paired_aggregate = _aggregate_ci95(
        per_seed,
        ["seed"],
        "aegiskv_minus_recompute_ms",
        nonnegative=False,
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))
    _bar_with_seed_points(
        axes[0], frame, policy_order, "affected_failure_added_wct_mean_ms"
    )
    axes[0].set_ylabel("Harmed-workflow added WCT (ms)")
    axes[0].set_title("(a) Observed AgentX workflows")
    axes[0].grid(axis="y", alpha=0.2)

    values = per_seed.aegiskv_minus_recompute_ms.to_numpy(dtype=float)
    axes[1].scatter(
        per_seed.seed.astype(str),
        values,
        s=18,
        facecolors="none",
        edgecolors=COLORS["aegiskv"],
    )
    axes[1].axhline(0.0, color="#111827", linewidth=0.7)
    axes[1].set_xticklabels([])
    axes[1].set_xlabel("Independent workflow-sampling seeds")
    axes[1].set_ylabel("AegisKV − recompute added WCT (ms)")
    axes[1].set_title("(b) Paired all-workflow seed means")
    axes[1].grid(axis="y", alpha=0.2)
    source = aggregate.assign(source_table="policy_seed_summary")
    source = pd.concat(
        [
            source,
            per_seed.assign(source_table="paired_seed_effect"),
        ],
        ignore_index=True,
        sort=False,
    )
    return fig, source


PLOTS: dict[str, Callable[..., tuple[plt.Figure, pd.DataFrame]]] = {
    "S1": _s1,
    "S2": _s2,
    "S3": _s3,
    "S4": _s4,
    "S5": _s5,
    "S6": _s6,
    "S7": _s7,
    "S8": _s8,
    "S9": _s9,
    "E1": _e1,
}


def render_all(lineage_dir: Path) -> list[Path]:
    _style()
    lineage_dir = Path(lineage_dir)
    manifest = json.loads((lineage_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(lineage_dir / "summary.csv")
    observations = pd.read_csv(lineage_dir / "observations.csv")
    timeline = pd.read_csv(lineage_dir / "timeline.csv")
    verification = pd.read_csv(lineage_dir / "verification.csv")
    figures = []
    for group in manifest["groups"]:
        renderer = PLOTS[group]
        figure, source = renderer(summary, observations, timeline, verification)
        slug = str(
            json.loads((lineage_dir / "resolved_configs" / group / "deployment.json").read_text())["figure"]["slug"]
        )
        _source(source, lineage_dir / "figure_sources", slug)
        figures.extend(_save(figure, lineage_dir / "figures", slug))
    return figures


__all__ = ["render_all"]
