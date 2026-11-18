#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
FIGURES_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "Figures"
DRAFT_PNG_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "figures" / "png"
MEASURED_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "figures" / "data" / "measured"
IMPACT_CSV = MEASURED_DIR / "fig_motivation_agent_failure_impact.csv"

COLOR_DELTA = "#DC2626"
COLOR_GRID = "#E5E7EB"
COLOR_MUTED = "#64748B"
COLOR_ZERO = "#6B7280"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    ordered = sorted(values)
    ys = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    return ordered, ys


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _collect_delta_rows() -> tuple[float, list[dict[str, object]]]:
    impact_rows = _read_csv(IMPACT_CSV)
    cdf_rows = [
        {
            "curve": row.get("curve", "Paired delta WCT"),
            "agent_label": row["agent_label"],
            "seed": row.get("seed", ""),
            "baseline_wct_s": row["baseline_wct_s"],
            "failure_wct_s": row.get("failure_wct_s", ""),
            "delta_wct_s": row["delta_wct_s"],
            "baseline_run_id": row.get("baseline_run_id", ""),
            "failure_run_id": row.get("failure_run_id", ""),
            "source": row.get("source", ""),
        }
        for row in impact_rows
        if row.get("panel") == "cdf_failure_case"
    ]
    if not cdf_rows:
        raise ValueError("missing cdf_failure_case rows in impact CSV")
    deltas = [float(row["delta_wct_s"]) for row in cdf_rows]
    return mean(deltas), cdf_rows


def _draw_delta_cdf_panel(
    ax: plt.Axes,
    mean_delta_s: float,
    cdf_rows: list[dict[str, object]],
) -> None:
    delta_values = [float(row["delta_wct_s"]) for row in cdf_rows]
    xs, ys = _ecdf(delta_values)
    ax.step(
        xs,
        ys,
        where="post",
        color=COLOR_DELTA,
        linewidth=2.0,
        label="Paired delta WCT",
        zorder=2,
    )
    for x, y in zip(xs, ys, strict=True):
        ax.scatter(
            [x],
            [y],
            s=42,
            color=COLOR_DELTA,
            edgecolor="#111827",
            linewidth=0.6,
            zorder=4,
        )

    ax.axvline(
        0.0,
        color=COLOR_ZERO,
        linewidth=1.2,
        linestyle="--",
        zorder=1,
        label="No change",
    )
    ax.axvline(
        mean_delta_s,
        color=COLOR_MUTED,
        linewidth=1.2,
        linestyle=":",
        zorder=1,
        label=f"Mean delta ({mean_delta_s:+.1f}s)",
    )
    p50 = _percentile(delta_values, 0.50)
    p90 = _percentile(delta_values, 0.90)
    ax.axhline(0.5, color=COLOR_GRID, linewidth=0.8, linestyle=":", zorder=0)
    ax.axhline(0.9, color=COLOR_GRID, linewidth=0.8, linestyle=":", zorder=0)
    ax.text(
        p50,
        0.54,
        f"P50={p50:+.1f}s",
        ha="center",
        va="bottom",
        fontsize=11,
        color=COLOR_MUTED,
    )
    ax.text(
        p90,
        0.94,
        f"P90={p90:+.1f}s",
        ha="center",
        va="bottom",
        fontsize=11,
        color=COLOR_MUTED,
    )

    xmin = min(min(delta_values, default=0.0) - 0.8, -1.2)
    xmax = max(max(delta_values, default=0.0) + 0.8, 8.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Paired delta WCT: controlled failure - baseline (s)")
    ax.set_ylabel("CDF")
    ax.set_title(
        "Controlled single-target failure replay (3-seed where available)",
        loc="left",
        weight="bold",
        pad=6,
    )
    ax.grid(axis="both", color=COLOR_GRID, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)


def render() -> None:
    if not IMPACT_CSV.is_file():
        raise FileNotFoundError(f"missing measured CSV: {IMPACT_CSV}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    MEASURED_DIR.mkdir(parents=True, exist_ok=True)

    mean_delta_s, cdf_rows = _collect_delta_rows()
    _write_csv(cdf_rows, MEASURED_DIR / "fig_motivation_agent_failure_wct_cdf.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.dpi": 300,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.4))
    _draw_delta_cdf_panel(ax, mean_delta_s, cdf_rows)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.20)

    for out in (
        FIGURES_DIR / "fig_motivation_agent_failure_wct_cdf.png",
        DRAFT_PNG_DIR / "fig_motivation_agent_failure_wct_cdf.png",
    ):
        fig.savefig(out, dpi=300, pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    render()
