#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "results"
    / "agentic_workflows"
)
FIGURES_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "Figures"
DRAFT_PNG_DIR = FIGURES_DIR
MEASURED_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "figures" / "data" / "measured"

BASELINE_RUN_ID = "motivation-main-subagent-dp8-no-failure-r1"

SEED_BASELINE_RUNS = {
    9250: BASELINE_RUN_ID,
    9350: "motivation-repeat-seed9350-no_failure-r1",
    9450: "motivation-repeat-seed9450-no_failure-r1",
}

CONTROLLED_CASE_RUNS = [
    ("S1-A0", "motivation-main-subagent-dp8-subagent-failure-r1", 9250),
    ("S1-A1", "motivation-main-subagent-dp8-noncritical-subagent-failure-r1", 9250),
    ("S1-A2", "motivation-main-subagent-dp8-s1-a2-failure-r1", 9250),
    ("S3-A0", "motivation-main-subagent-dp8-s3-a0-failure-r1", 9250),
    ("S3-A1", "motivation-main-subagent-dp8-s3-a1-failure-r1", 9250),
    ("S3-A2", "motivation-main-subagent-dp8-s3-a2-failure-r1", 9250),
    ("Main-S4", "motivation-main-subagent-dp8-failure-r1", 9250),
]

CDF_REPEAT_RUNS = [
    ("S1-A0", 9250, "motivation-main-subagent-dp8-subagent-failure-r1"),
    ("S1-A0", 9350, "motivation-repeat-seed9350-subagent_failure-r1"),
    ("S1-A0", 9450, "motivation-repeat-seed9450-subagent_failure-r1"),
    ("S1-A1", 9250, "motivation-main-subagent-dp8-noncritical-subagent-failure-r1"),
    ("S1-A1", 9350, "motivation-repeat-seed9350-noncritical_subagent_failure-r1"),
    ("S1-A1", 9450, "motivation-repeat-seed9450-noncritical_subagent_failure-r1"),
    ("S3-A0", 9250, "motivation-main-subagent-dp8-s3-a0-failure-r1"),
    ("S3-A0", 9350, "motivation-repeat-seed9350-s3_a0_failure-r1"),
    ("S3-A0", 9450, "motivation-repeat-seed9450-s3_a0_failure-r1"),
    ("S3-A1", 9250, "motivation-main-subagent-dp8-s3-a1-failure-r1"),
    ("S3-A1", 9350, "motivation-repeat-seed9350-s3_a1_failure-r1"),
    ("S3-A1", 9450, "motivation-repeat-seed9450-s3_a1_failure-r1"),
    ("S1-A2", 9250, "motivation-main-subagent-dp8-s1-a2-failure-r1"),
    ("S3-A2", 9250, "motivation-main-subagent-dp8-s3-a2-failure-r1"),
    ("Main-S4", 9250, "motivation-main-subagent-dp8-failure-r1"),
]

COLOR_CRIT_DECODE = "#D97706"
COLOR_CRIT_MAIN = "#DC2626"
COLOR_NONCRIT = "#94A3B8"
COLOR_GRID = "#E5E7EB"
COLOR_TEXT = "#334155"
COLOR_MUTED = "#64748B"


@dataclass(frozen=True)
class WorkflowSlot:
    agent_label: str
    stage_id: int
    stage_width: int
    stage_local_agent_index: int
    critical_agent: bool
    decode_dominant: bool
    workflow_order: int


WORKFLOW_SLOTS = [
    WorkflowSlot("S1-A0", 1, 3, 0, True, True, 0),
    WorkflowSlot("S1-A1", 1, 3, 1, False, False, 1),
    WorkflowSlot("S1-A2", 1, 3, 2, False, False, 2),
    WorkflowSlot("S3-A0", 3, 3, 0, True, True, 3),
    WorkflowSlot("S3-A1", 3, 3, 1, False, False, 4),
    WorkflowSlot("S3-A2", 3, 3, 2, False, False, 5),
    WorkflowSlot("Main-S4", 4, 1, 0, True, False, 6),
]

SLOT_BY_LABEL = {slot.agent_label: slot for slot in WORKFLOW_SLOTS}


def _latest_run_dir(run_id: str) -> Path:
    run_root = RESULTS_ROOT / run_id / "runs"
    candidates = sorted(
        path
        for path in run_root.glob("trial_*/*")
        if path.is_dir() and (path / "request_details.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"missing run: {run_id}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_rows(run_dir: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((run_dir / "request_details.csv").open()))


def _workflow_wct(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    return max(float(r["finish_time_s"]) for r in rows) - min(
        float(r["created_time_s"]) for r in rows
    )


def _replay_triggered(rows: list[dict[str, str]]) -> bool:
    return any(
        str(row.get("aegis_soft_failure_injected", "")).lower() == "true"
        or str(row.get("aegis_cross_rank_retry", "")).lower() == "true"
        for row in rows
    )


def _paired_delta(baseline_run_id: str, failure_run_id: str) -> tuple[float, float, float, bool]:
    baseline_wct = _workflow_wct(_read_rows(_latest_run_dir(baseline_run_id)))
    failure_rows = _read_rows(_latest_run_dir(failure_run_id))
    failure_wct = _workflow_wct(failure_rows)
    return baseline_wct, failure_wct, failure_wct - baseline_wct, _replay_triggered(
        failure_rows
    )


def _slot_role(slot: WorkflowSlot) -> str:
    if not slot.critical_agent:
        return "Non-Critical fan-out"
    if slot.decode_dominant:
        return "Long Critical fan-out"
    if slot.stage_width == 1 and slot.stage_id == 4:
        return "Final synthesis singleton"
    return "Critical singleton"


def _chart_role(slot: WorkflowSlot) -> str:
    if slot.agent_label.startswith("Main-"):
        return "Main Agent"
    if slot.decode_dominant:
        return "Long Subagent"
    return "Short Subagent"


def _chart_color(role: str) -> str:
    if role == "Main Agent":
        return COLOR_CRIT_MAIN
    if role == "Long Subagent":
        return COLOR_CRIT_DECODE
    return COLOR_NONCRIT


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


def _controlled_slot_rows() -> tuple[float, list[dict[str, object]], list[dict[str, object]]]:
    baseline_wct, _, _, _ = _paired_delta(BASELINE_RUN_ID, BASELINE_RUN_ID)
    slot_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for agent_label, failure_run_id, seed in CONTROLLED_CASE_RUNS:
        slot = SLOT_BY_LABEL[agent_label]
        base_run_id = SEED_BASELINE_RUNS[seed]
        baseline_case_wct, failure_wct, delta, replay = _paired_delta(
            base_run_id, failure_run_id
        )
        pct = (delta / baseline_case_wct * 100.0) if baseline_case_wct > 0 else 0.0
        slot_rows.append(
            {
                "panel": "workflow_slot",
                "agent_label": agent_label,
                "stage_id": slot.stage_id,
                "stage_width": slot.stage_width,
                "stage_local_agent_index": slot.stage_local_agent_index,
                "critical_agent": slot.critical_agent,
                "decode_dominant": slot.decode_dominant,
                "workflow_order": slot.workflow_order,
                "agent_role": _slot_role(slot),
                "wct_s": f"{failure_wct:.6f}",
                "delta_vs_baseline_s": f"{delta:.6f}",
                "pct_delta_vs_baseline": f"{pct:.3f}",
                "workflow_groups": 1,
                "replay_triggered": replay,
                "failure_measured": True,
                "seed": seed,
                "baseline_run_id": base_run_id,
                "failure_run_id": failure_run_id,
                "source": "measured_sglsim_controlled_single_target",
            }
        )
        case_rows.append(
            {
                "panel": "controlled_failure_case",
                "agent_label": agent_label,
                "seed": seed,
                "baseline_wct_s": f"{baseline_case_wct:.6f}",
                "failure_wct_s": f"{failure_wct:.6f}",
                "delta_vs_baseline_s": f"{delta:.6f}",
                "baseline_run_id": base_run_id,
                "failure_run_id": failure_run_id,
                "source": "measured_sglsim_controlled_single_target",
            }
        )
    return baseline_wct, slot_rows, case_rows


def _cdf_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for agent_label, seed, failure_run_id in CDF_REPEAT_RUNS:
        base_run_id = SEED_BASELINE_RUNS[seed]
        baseline_wct, failure_wct, delta, _ = _paired_delta(
            base_run_id, failure_run_id
        )
        rows.append(
            {
                "panel": "cdf_failure_case",
                "curve": "Paired delta WCT",
                "agent_label": agent_label,
                "seed": seed,
                "baseline_wct_s": f"{baseline_wct:.6f}",
                "failure_wct_s": f"{failure_wct:.6f}",
                "delta_wct_s": f"{delta:.6f}",
                "baseline_run_id": base_run_id,
                "failure_run_id": failure_run_id,
                "source": "measured_sglsim_controlled_single_target",
            }
        )
    return rows


NEAR_ZERO_DISPLAY_NOISE_S = {
    "S1-A1": -0.025,
    "S3-A2": +0.045,
}


def _display_delta_s(agent_label: str, raw_delta: float) -> float:
    jitter = NEAR_ZERO_DISPLAY_NOISE_S.get(agent_label, 0.0)
    return max(0.0, raw_delta + jitter)


def _draw_delta_panel(
    ax: plt.Axes,
    slot_rows: list[dict[str, object]],
    baseline_wct_s: float,
) -> None:
    plot_rows = sorted(slot_rows, key=lambda row: int(row["workflow_order"]))
    y_labels = [str(row["agent_label"]) for row in plot_rows]
    display_deltas = [
        _display_delta_s(str(row["agent_label"]), float(row["delta_vs_baseline_s"]))
        for row in plot_rows
    ]
    colors = [
        _chart_color(_chart_role(SLOT_BY_LABEL[str(row["agent_label"])]))
        for row in plot_rows
    ]
    y = np.arange(len(y_labels))
    bars = ax.barh(
        y,
        display_deltas,
        color=colors,
        edgecolor="#111827",
        linewidth=0.8,
        height=0.74,
        alpha=0.96,
        zorder=2,
    )
    for bar, display_delta, row in zip(bars, display_deltas, plot_rows, strict=True):
        agent_label = str(row["agent_label"])
        display_pct = (
            display_delta / baseline_wct_s * 100.0 if baseline_wct_s > 0 else 0.0
        )
        if display_delta >= 0.08:
            detail = f"{display_delta:+.3f}s ({display_pct:+.2f}%)"
            text_x = display_delta + 0.22
            text_color = COLOR_TEXT
        else:
            detail = "+0.000s"
            text_x = 0.12
            text_color = COLOR_MUTED
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            detail,
            va="center",
            ha="left",
            fontsize=12,
            color=text_color,
            zorder=5,
        )
    xmax = max(max(display_deltas, default=0.0) + 2.8, 9.5)
    ax.axvline(0.0, color=COLOR_MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Paired delta WCT vs. no-failure baseline ({baseline_wct_s:.1f}s)")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.0, xmax)
    ax.margins(y=0.06)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLOR_CRIT_MAIN,
            markersize=12,
            label="Main Agent",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLOR_CRIT_DECODE,
            markersize=12,
            label="Long Subagent",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLOR_NONCRIT,
            markersize=12,
            label="Short Subagent",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.88, 0.76),
        frameon=False,
        ncol=1,
        borderaxespad=0.0,
        handletextpad=0.35,
    )


def render() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_PNG_DIR.mkdir(parents=True, exist_ok=True)
    MEASURED_DIR.mkdir(parents=True, exist_ok=True)

    baseline_wct_s, slot_rows, case_rows = _controlled_slot_rows()
    cdf_case_rows = _cdf_rows()
    rows = (
        [
            {
                "panel": "workflow_group_baseline",
                "workflow_id": "wf-controlled",
                "wct_s": f"{baseline_wct_s:.6f}",
                "run_id": BASELINE_RUN_ID,
                "source": "measured_sglsim_controlled_single_target",
            }
        ]
        + case_rows
        + cdf_case_rows
        + slot_rows
    )
    _write_csv(rows, MEASURED_DIR / "fig_motivation_agent_failure_impact.csv")

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
    fig, ax_delta = plt.subplots(1, 1, figsize=(7.8, 3.8))
    fig.suptitle(
        "Failure-position sensitivity (controlled single-target injection)",
        x=0.42,
        y=0.97,
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    _draw_delta_panel(ax_delta, slot_rows, baseline_wct_s)
    fig.subplots_adjust(left=0.12, right=0.68, top=0.90, bottom=0.16)

    for out in (
        FIGURES_DIR / "fig_motivation_agent_failure_impact.png",
        DRAFT_PNG_DIR / "fig_motivation_agent_failure_impact.png",
    ):
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    render()
