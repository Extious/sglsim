#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[5]
BEAMER_FIGURES_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "Figures"
DRAFT_PNG_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "figures" / "png"
DEFAULT_OUTPUT = BEAMER_FIGURES_DIR / "fig_system_overview_generated_v2_tight.png"
DRAFT_OUTPUT = DRAFT_PNG_DIR / "fig_system_overview_generated_v2_tight.png"


COLORS = {
    "ink": "#142033",
    "muted": "#66758A",
    "line": "#334155",
    "grid": "#D8E0EA",
    "lane": "#F7F9FC",
    "critical": "#FFE599",
    "noncritical": "#E8EDF3",
    "control": "#ECE7FF",
    "control_edge": "#5B4E86",
    "serve": "#E4F3FF",
    "serve_edge": "#2A668A",
    "kv": "#CFE4FF",
    "replica": "#FFD7A8",
    "host": "#E8E3DD",
    "recompute": "#FFFFFF",
    "restore": "#DDF8E7",
    "danger": "#B42318",
    "blue": "#246BFE",
    "brown": "#9A4F12",
    "green": "#16803C",
    "violet": "#6D3FD3",
}


def rounded(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str,
    ec: str = COLORS["line"],
    lw: float = 0.8,
    radius: float = 0.010,
    z: int = 5,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=z,
        )
    )


def label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fs: float = 6.5,
    ha: str = "center",
    va: str = "center",
    color: str = COLORS["ink"],
    weight: str = "normal",
    z: int = 30,
    linespacing: float = 0.95,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color,
        weight=weight,
        linespacing=linespacing,
        zorder=z,
    )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str = COLORS["line"],
    lw: float = 0.75,
    fs: float = 6.1,
    weight: str = "normal",
    radius: float = 0.010,
    z: int = 10,
) -> None:
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, radius=radius, z=z)
    label(ax, x + w / 2, y + h / 2, text, fs=fs, weight=weight, z=z + 1)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label_text: str = "",
    color: str = COLORS["line"],
    lw: float = 0.85,
    rad: float = 0.0,
    ls: str = "-",
    fs: float = 5.2,
    z: int = 25,
    text_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=lw,
            linestyle=ls,
            color=color,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
        )
    )
    if label_text:
        x = (start[0] + end[0]) / 2 + text_offset[0]
        y = (start[1] + end[1]) / 2 + text_offset[1]
        label(
            ax,
            x,
            y,
            label_text,
            fs=fs,
            color=color,
            z=z + 1,
            linespacing=0.90,
        )


def lane(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
) -> None:
    rounded(ax, x, y, w, h, fc=COLORS["lane"], ec=COLORS["grid"], lw=0.85, radius=0.018, z=0)
    label(ax, x + 0.014, y + h - 0.020, title, fs=7.2, ha="left", weight="bold")


def draw_workflow(ax: plt.Axes) -> None:
    lane(ax, 0.045, 0.565, 0.310, 0.335, "Agentic workflow and failure scope")
    y = 0.744
    stage_w = 0.034
    singleton_h = 0.040
    box(ax, 0.060, y, stage_w, singleton_h, "S0", fc="#DCEAFF", fs=5.2, weight="bold")
    box(ax, 0.181, y, stage_w, singleton_h, "S2", fc="#DCEAFF", fs=5.2, weight="bold")
    box(ax, 0.306, y, stage_w, singleton_h, "S4", fc="#DCEAFF", fs=5.2, weight="bold")

    def fanout(x: float, prefix: str) -> None:
        rounded(ax, x, y - 0.070, 0.062, 0.160, fc="#FFFFFF", ec=COLORS["line"], lw=0.80, radius=0.010, z=6)
        for idx in range(3):
            yy = y + 0.046 - idx * 0.043
            fc = COLORS["critical"] if idx == 0 else COLORS["noncritical"]
            box(
                ax,
                x + 0.012,
                yy,
                0.038,
                0.026,
                f"{prefix}{idx}",
                fc=fc,
                fs=4.7,
                lw=0.50,
                weight="bold" if idx == 0 else "normal",
                radius=0.005,
            )
        label(ax, x + 0.031, y - 0.050, "join", fs=4.6, color=COLORS["muted"])

    fanout(0.111, "A")
    fanout(0.236, "B")
    arrow(ax, (0.094, y + 0.020), (0.111, y + 0.020), lw=0.70)
    arrow(ax, (0.173, y + 0.020), (0.181, y + 0.020), lw=0.70)
    arrow(ax, (0.215, y + 0.020), (0.236, y + 0.020), lw=0.70)
    arrow(ax, (0.298, y + 0.020), (0.306, y + 0.020), lw=0.70)
    ax.scatter([0.157, 0.282], [y + 0.083, y + 0.083], marker="x", s=34, color=COLORS["danger"], linewidths=1.2, zorder=35)
    label(ax, 0.060, 0.654, "1-3-1-3-1 shape", fs=5.0, ha="left", color=COLORS["muted"])
    label(ax, 0.060, 0.620, "Only Critical agents are eligible for GPUReplica.", fs=5.4, ha="left", color=COLORS["muted"])


def draw_control_plane(ax: plt.Axes) -> None:
    lane(ax, 0.385, 0.565, 0.570, 0.335, "AegisKV control plane")
    y = 0.756
    steps = [
        (0.410, 0.094, "Request\nmetadata", "#FFFFFF"),
        (0.532, 0.102, "Criticality\nclassifier", COLORS["control"]),
        (0.666, 0.116, "MDP policy\n+ online alpha", COLORS["control"]),
        (0.818, 0.110, "Placement /\nretry plan", COLORS["restore"]),
    ]
    for x, w, text, fc in steps:
        box(
            ax,
            x,
            y,
            w,
            0.062,
            text,
            fc=fc,
            ec=COLORS["control_edge"] if fc == COLORS["control"] else COLORS["line"],
            fs=5.6,
            weight="bold" if fc != "#FFFFFF" else "normal",
        )
    for (x, w, *_), (nx, *_rest) in zip(steps, steps[1:]):
        arrow(ax, (x + w, y + 0.031), (nx, y + 0.031), lw=0.80)

    box(ax, 0.485, 0.635, 0.118, 0.055, "Workflow\nWCT reward", fc="#FFFFFF", fs=5.3)
    box(ax, 0.657, 0.635, 0.160, 0.055, "HBM pressure\n+ failure rate", fc="#FFFFFF", fs=5.3)
    arrow(ax, (0.544, 0.690), (0.583, y), color=COLORS["blue"], lw=0.75, rad=-0.10)
    arrow(ax, (0.736, 0.690), (0.724, y), color=COLORS["blue"], lw=0.75)
    label(
        ax,
        0.405,
        0.602,
        "The reward is workflow-level WCT; node-local rewards are not separated.",
        fs=5.3,
        ha="left",
        color=COLORS["muted"],
    )


def hbm_bar(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y), w * 0.66, h, facecolor=COLORS["kv"], edgecolor=COLORS["line"], linewidth=0.45, zorder=16))
    ax.add_patch(Rectangle((x + w * 0.66, y), w * 0.34, h, facecolor=COLORS["replica"], edgecolor=COLORS["line"], linewidth=0.45, zorder=16))
    label(ax, x + w * 0.33, y + h / 2, "serving KV", fs=4.4, z=17)
    label(ax, x + w * 0.83, y + h / 2, "replica", fs=4.4, z=17)


def rank(ax: plt.Axes, x: float, y: float, text: str, *, failed: bool = False) -> None:
    rounded(ax, x, y, 0.118, 0.080, fc="#FFFFFF", ec=COLORS["line"], lw=0.8, radius=0.010, z=10)
    label(ax, x + 0.059, y + 0.060, text, fs=5.9, weight="bold")
    hbm_bar(ax, x + 0.014, y + 0.015, 0.090, 0.023)
    if failed:
        ax.scatter([x + 0.114], [y + 0.058], marker="x", s=56, color=COLORS["danger"], linewidths=1.6, zorder=40)


def draw_data_plane(ax: plt.Axes) -> None:
    lane(ax, 0.045, 0.145, 0.910, 0.355, "Serving data plane and replay hierarchy")
    box(ax, 0.073, 0.322, 0.120, 0.064, "SGLSim / SGLang\nonline serving", fc=COLORS["serve"], ec=COLORS["serve_edge"], fs=5.5, weight="bold")
    rank(ax, 0.255, 0.334, "DP0 failed", failed=True)
    rank(ax, 0.415, 0.334, "DP1 peer")
    rank(ax, 0.575, 0.334, "DP2")
    rank(ax, 0.735, 0.334, "DP3...DP7")
    box(ax, 0.295, 0.202, 0.338, 0.054, "Host backup directory / CPU KV store", fc=COLORS["host"], fs=5.7)
    box(ax, 0.753, 0.210, 0.108, 0.050, "recompute\nfallback", fc=COLORS["recompute"], fs=5.2)

    arrow(ax, (0.193, 0.354), (0.255, 0.371), label_text="serve", fs=5.0, text_offset=(0.000, 0.015))
    arrow(ax, (0.373, 0.374), (0.415, 0.374), label_text="peer GPU", color=COLORS["green"], lw=0.85, text_offset=(0.000, 0.016))
    arrow(ax, (0.300, 0.334), (0.340, 0.256), label_text="D2H", color=COLORS["brown"], rad=0.10, text_offset=(-0.010, 0.005))
    arrow(ax, (0.485, 0.334), (0.510, 0.256), label_text="H2D", color=COLORS["blue"], rad=-0.08, text_offset=(0.012, 0.006))
    arrow(ax, (0.789, 0.334), (0.807, 0.260), label_text="miss", color=COLORS["violet"], rad=-0.08, text_offset=(0.012, 0.006))

    label(ax, 0.075, 0.178, "Replay priority: peer GPUReplica -> host restore -> recompute.", fs=5.4, ha="left", color=COLORS["muted"])
    label(ax, 0.550, 0.178, "Shared-prefix overlap is charged before alpha allocation.", fs=5.4, ha="center", color=COLORS["muted"])


def draw_legend(ax: plt.Axes) -> None:
    items = [
        (0.070, COLORS["critical"], "Critical"),
        (0.155, COLORS["noncritical"], "Non-Critical"),
        (0.275, COLORS["kv"], "Inference KV"),
        (0.395, COLORS["replica"], "GPUReplica KV"),
        (0.535, COLORS["host"], "Host backup"),
        (0.665, COLORS["restore"], "Recovery plan"),
    ]
    for x, color, text in items:
        ax.add_patch(Rectangle((x, 0.070), 0.016, 0.012, facecolor=color, edgecolor=COLORS["line"], linewidth=0.40, zorder=20))
        label(ax, x + 0.021, 0.076, text, fs=5.3, ha="left", color=COLORS["muted"])


def draw(output: Path = DEFAULT_OUTPUT) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=270)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    label(ax, 0.045, 0.940, "AegisKV system overview", fs=9.0, ha="left", weight="bold")
    label(
        ax,
        0.955,
        0.940,
        "workflow WCT objective | Critical-only GPUReplica | adaptive alpha",
        fs=5.8,
        ha="right",
        color=COLORS["muted"],
    )

    draw_workflow(ax)
    draw_control_plane(ax)
    arrow(
        ax,
        (0.355, 0.742),
        (0.410, 0.788),
        label_text="role + token\nmetadata",
        color=COLORS["blue"],
        lw=0.75,
        rad=0.10,
        fs=4.9,
        text_offset=(0.000, 0.028),
    )
    draw_data_plane(ax)
    arrow(
        ax,
        (0.873, 0.756),
        (0.823, 0.500),
        label_text="alpha, rank,\nretry route",
        color=COLORS["blue"],
        lw=0.75,
        rad=-0.12,
        fs=4.9,
        text_offset=(0.026, -0.005),
    )
    draw_legend(ax)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    if output != DRAFT_OUTPUT:
        DRAFT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        copyfile(output, DRAFT_OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    draw()
