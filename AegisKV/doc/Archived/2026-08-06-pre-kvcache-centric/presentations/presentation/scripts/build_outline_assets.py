#!/usr/bin/env python3
"""Build paper-outline diagram PNGs and the compact motivation figure.

The editable diagram sources live under ``diagrams/*.drawio``.  This script
renders matching high-resolution PNG companions without requiring a draw.io
desktop installation on the build host.
"""

from __future__ import annotations

import csv
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "Figures"
COLORS = {
    "blue": (0.18, 0.39, 0.56),
    "blue_light": (0.92, 0.96, 0.99),
    "purple": (0.40, 0.32, 0.61),
    "purple_light": (0.95, 0.93, 0.98),
    "orange": (0.71, 0.44, 0.16),
    "orange_light": (1.00, 0.94, 0.87),
    "green": (0.20, 0.49, 0.38),
    "green_light": (0.88, 0.95, 0.91),
    "red": (0.71, 0.29, 0.26),
    "red_light": (0.99, 0.90, 0.89),
    "gray": (0.35, 0.39, 0.44),
    "gray_light": (0.97, 0.97, 0.98),
    "white": (1.0, 1.0, 1.0),
}


def _box(
    page: fitz.Page,
    rect: tuple[float, float, float, float],
    text: str,
    *,
    fill: tuple[float, float, float],
    stroke: tuple[float, float, float],
    size: float = 15,
    bold: bool = False,
    align: int = fitz.TEXT_ALIGN_CENTER,
) -> fitz.Rect:
    r = fitz.Rect(rect)
    page.draw_rect(r, color=stroke, fill=fill, width=1.4)
    page.insert_textbox(
        r + (10, 10, -10, -8),
        text,
        fontsize=size,
        fontname="hebo" if bold else "helv",
        color=(0.10, 0.19, 0.26),
        align=align,
        lineheight=1.12,
    )
    return r


def _label(page: fitz.Page, rect: tuple[float, float, float, float], text: str, size: float = 18) -> None:
    page.insert_textbox(
        fitz.Rect(rect),
        text,
        fontsize=size,
        fontname="hebo",
        color=(0.10, 0.23, 0.34),
        align=fitz.TEXT_ALIGN_CENTER,
    )


def _arrow(
    page: fitz.Page,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: tuple[float, float, float] | None = None,
    width: float = 1.8,
) -> None:
    color = color or COLORS["gray"]
    a = fitz.Point(*start)
    b = fitz.Point(*end)
    page.draw_line(a, b, color=color, width=width)
    dx, dy = b.x - a.x, b.y - a.y
    norm = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    tip = b
    left = fitz.Point(b.x - 11 * ux + 5 * px, b.y - 11 * uy + 5 * py)
    right = fitz.Point(b.x - 11 * ux - 5 * px, b.y - 11 * uy - 5 * py)
    shape = page.new_shape()
    shape.draw_polyline([tip, left, right, tip])
    shape.finish(color=color, fill=color)
    shape.commit()


def _save(doc: fitz.Document, stem: str) -> None:
    png_path = FIGURES / f"{stem}.png"
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(png_path)


def build_system_overview() -> None:
    doc = fitz.open()
    page = doc.new_page(width=1280, height=720)
    page.draw_rect(page.rect, color=COLORS["white"], fill=COLORS["white"], width=0)
    _label(page, (170, 18, 1110, 55), "AegisKV: workflow-aware KV protection and recovery", 22)

    _box(page, (30, 75, 295, 665), "", fill=COLORS["blue_light"], stroke=COLORS["blue"])
    _label(page, (45, 88, 280, 118), "Agentic workflow", 18)
    dag = _box(page, (60, 145, 265, 220), "Instantiated execution graph\n(revealed dependencies)", fill=COLORS["white"], stroke=COLORS["blue"])
    meta = _box(page, (60, 280, 265, 360), "workflow / stage / turn\nowner / token span", fill=COLORS["white"], stroke=COLORS["blue"])
    wct = _box(page, (60, 500, 265, 585), "Objective\nworkflow completion time", fill=COLORS["green_light"], stroke=COLORS["green"], bold=True)

    _box(page, (350, 75, 770, 665), "", fill=COLORS["purple_light"], stroke=COLORS["purple"])
    _label(page, (370, 88, 750, 118), "AegisKV control plane", 18)
    loss = _box(page, (390, 140, 730, 215), "1. Workflow loss estimator\nslack + failure scenarios", fill=COLORS["white"], stroke=COLORS["purple"])
    placement = _box(page, (390, 260, 730, 335), "2. L1/L2 Backup placement\nvalue-density greedy under budget", fill=COLORS["white"], stroke=COLORS["purple"])
    directory = _box(page, (390, 380, 730, 455), "3. Safe-frontier directory\nACK + version + valid locations", fill=COLORS["white"], stroke=COLORS["purple"])
    dispatch = _box(page, (390, 500, 730, 575), "4. Recovery dispatcher\nmin(GPU, Host, recompute)", fill=COLORS["green_light"], stroke=COLORS["green"], bold=True)

    _box(page, (825, 75, 1250, 665), "", fill=COLORS["orange_light"], stroke=COLORS["orange"])
    _label(page, (845, 88, 1230, 118), "SGLang / SGLSim data plane", 18)
    owner = _box(page, (860, 145, 1025, 220), "Owner DP rank\nactive KV", fill=COLORS["white"], stroke=COLORS["orange"])
    peer = _box(page, (1085, 145, 1215, 220), "Peer rank\nL1 Backup HBM", fill=COLORS["green_light"], stroke=COLORS["green"])
    host = _box(page, (860, 310, 1025, 390), "L2 Backup\ncommitted pages", fill=COLORS["gray_light"], stroke=COLORS["gray"])
    replica = _box(page, (1085, 310, 1215, 390), "L1 Backup\ncommitted atoms", fill=COLORS["green_light"], stroke=COLORS["green"])
    failure = _box(page, (860, 500, 1025, 575), "Owner fail-stop", fill=COLORS["red_light"], stroke=COLORS["red"], bold=True)
    resume = _box(page, (1085, 490, 1215, 585), "Retry\nrestore + replay", fill=COLORS["green_light"], stroke=COLORS["green"], bold=True)

    _arrow(page, (dag.x1, dag.y1 - 20), (loss.x0, loss.y0 + 20), color=COLORS["blue"])
    _arrow(page, (meta.x1, meta.y0 + 20), (loss.x0, loss.y1 - 20), color=COLORS["blue"])
    _arrow(page, (560, loss.y1), (560, placement.y0), color=COLORS["purple"])
    _arrow(page, (placement.x1, 297), (peer.x0, 182), color=COLORS["purple"])
    _arrow(page, (owner.x0 + 80, owner.y1), (host.x0 + 80, host.y0), color=COLORS["orange"])
    _arrow(page, (owner.x1, 182), (replica.x0, 350), color=COLORS["green"])
    _arrow(page, (host.x0, 350), (directory.x1, 407), color=COLORS["gray"])
    _arrow(page, (replica.x0, 350), (directory.x1, 430), color=COLORS["green"])
    _arrow(page, (failure.x0, 538), (dispatch.x1, 538), color=COLORS["red"])
    _arrow(page, (dispatch.x1, 552), (resume.x0, 538), color=COLORS["green"])
    _arrow(page, (resume.x0, 570), (wct.x1, 555), color=COLORS["green"])
    _save(doc, "fig_aegiskv_system_overview")


def build_lifecycle() -> None:
    doc = fitz.open()
    page = doc.new_page(width=1280, height=680)
    page.draw_rect(page.rect, color=COLORS["white"], fill=COLORS["white"], width=0)
    _label(page, (170, 18, 1110, 55), "KV atom lifecycle: only committed copies carry recovery value", 22)
    local = _box(page, (55, 245, 240, 330), "Local GPU atom\nradix node extended", fill=COLORS["blue_light"], stroke=COLORS["blue"])
    hp = _box(page, (330, 110, 525, 195), "L2 copy pending\nD2H issued", fill=COLORS["orange_light"], stroke=COLORS["orange"])
    hc = _box(page, (625, 110, 835, 195), "L2 committed\nACK + version", fill=COLORS["green_light"], stroke=COLORS["green"], bold=True)
    pp = _box(page, (330, 360, 525, 445), "L1 copy pending\nP2P issued", fill=COLORS["orange_light"], stroke=COLORS["orange"])
    pc = _box(page, (625, 360, 835, 445), "L1 committed\nACK + anti-affinity", fill=COLORS["green_light"], stroke=COLORS["green"], bold=True)
    directory = _box(page, (940, 220, 1225, 340), "Safe-frontier directory\nlocations = {L1, L2}\ndeepest committed span", fill=COLORS["purple_light"], stroke=COLORS["purple"], bold=True)
    invalid = _box(page, (940, 455, 1225, 535), "Invalidated\neviction / stale epoch / end", fill=COLORS["red_light"], stroke=COLORS["red"])
    _arrow(page, (local.x1, 270), (hp.x0, 155), color=COLORS["orange"])
    _arrow(page, (hp.x1, 152), (hc.x0, 152), color=COLORS["green"])
    _arrow(page, (local.x1, 310), (pp.x0, 402), color=COLORS["orange"])
    _arrow(page, (pp.x1, 402), (pc.x0, 402), color=COLORS["green"])
    _arrow(page, (hc.x1, 152), (directory.x0, 252), color=COLORS["purple"])
    _arrow(page, (pc.x1, 402), (directory.x0, 305), color=COLORS["purple"])
    _arrow(page, (1082, directory.y1), (1082, invalid.y0), color=COLORS["red"])
    _box(page, (220, 580, 1060, 640), "Invariant: pending bytes are not a checkpoint; recovery uses only a committed and surviving location.", fill=COLORS["gray_light"], stroke=COLORS["gray"], size=15, bold=True)
    _save(doc, "fig_kv_lifecycle")


def build_recovery() -> None:
    doc = fitz.open()
    page = doc.new_page(width=1280, height=680)
    page.draw_rect(page.rect, color=COLORS["white"], fill=COLORS["white"], width=0)
    _label(page, (170, 18, 1110, 55), "Failure-time recovery: choose one valid path, then replay the suffix", 22)
    failure = _box(page, (40, 260, 195, 340), "Owner rank fails", fill=COLORS["red_light"], stroke=COLORS["red"], bold=True)
    lookup = _box(page, (260, 250, 450, 350), "Lookup deepest valid\ncommitted frontier", fill=COLORS["purple_light"], stroke=COLORS["purple"])
    gpu = _box(page, (535, 90, 735, 175), "L1 Backup path\nroute + queue + resume", fill=COLORS["green_light"], stroke=COLORS["green"])
    host = _box(page, (535, 255, 735, 340), "L2 Backup path\nroute + H2D + resume", fill=COLORS["orange_light"], stroke=COLORS["orange"])
    recompute = _box(page, (535, 420, 735, 505), "Recompute path\nroute + prefill suffix", fill=COLORS["blue_light"], stroke=COLORS["blue"])
    choose = _box(page, (830, 245, 1040, 350), "Choose minimum predicted\nworkflow loss path", fill=COLORS["gray_light"], stroke=COLORS["gray"], bold=True)
    finish = _box(page, (1105, 245, 1245, 350), "Restore prefix\nreplay suffix\nresume", fill=COLORS["green_light"], stroke=COLORS["green"], bold=True)
    _arrow(page, (failure.x1, 300), (lookup.x0, 300), color=COLORS["red"])
    _arrow(page, (lookup.x1, 270), (gpu.x0, 132), color=COLORS["green"])
    _arrow(page, (lookup.x1, 300), (host.x0, 297), color=COLORS["orange"])
    _arrow(page, (lookup.x1, 330), (recompute.x0, 462), color=COLORS["blue"])
    _arrow(page, (gpu.x1, 132), (choose.x0, 270), color=COLORS["green"])
    _arrow(page, (host.x1, 297), (choose.x0, 297), color=COLORS["orange"])
    _arrow(page, (recompute.x1, 462), (choose.x0, 330), color=COLORS["blue"])
    _arrow(page, (choose.x1, 297), (finish.x0, 297), color=COLORS["green"])
    _box(page, (200, 575, 1080, 635), "L1 Backup, L2 Backup and recompute are mutually exclusive alternatives; queue, restore, replay and resume are charged once.", fill=COLORS["gray_light"], stroke=COLORS["gray"], size=15)
    _save(doc, "fig_recovery_path")


def build_motivation_delta() -> None:
    import matplotlib.pyplot as plt

    source = ROOT / "data" / "causal_failure_delta_wct.csv"
    rows: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    rows = [row for row in rows if row["target_class"] != "baseline"]
    classes = ["critical_replay", "no_replay_control"]
    colors = {"critical_replay": "#2D638F", "no_replay_control": "#B56F28"}
    labels = {"critical_replay": "Critical replay", "no_replay_control": "Sibling control"}

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, cls in enumerate(classes):
        vals = [float(r["delta_workflow_wct_s"]) for r in rows if r["target_class"] == cls]
        mean = sum(vals) / len(vals)
        ax.bar(i, mean, width=0.55, color=colors[cls], alpha=0.82, zorder=2)
        offsets = [-0.15, -0.05, 0.05, 0.15][: len(vals)]
        ax.scatter([i + o for o in offsets], vals, color="#202A33", s=34, zorder=3)
        ax.text(i, mean + (0.13 if mean >= 0 else -0.22), f"mean {mean:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#59636F", linewidth=1)
    ax.set_xticks(range(len(classes)), [labels[c] for c in classes])
    ax.set_ylabel(r"Paired $\Delta$ workflow completion time (s)")
    ax.set_title("Failure impact is workflow-position dependent")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.02, "SGLSim online, 2 seeds, 4 paired cases/class\npreliminary motivation evidence", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#59636F")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_motivation_causal_delta.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_system_overview()
    build_lifecycle()
    build_recovery()
    build_motivation_delta()


if __name__ == "__main__":
    main()
