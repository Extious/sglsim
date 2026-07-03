#!/usr/bin/env python3
"""Generate the AegisKV figure package.

This script intentionally separates two artifact classes:

* mechanism diagrams: editable draw.io XML plus PNG previews.
* result figures: template plots that encode the expected trends from
  sections 07 and 08. These are not experimental evidence yet.

The generated manifest marks the evidence status for each artifact so paper
text cannot accidentally cite a template as a measured result.
"""

from __future__ import annotations

import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
PNG_DIR = ROOT / "png"
DATA_DIR = ROOT / "data" / "template"
MEASURED_DIR = ROOT / "data" / "measured"
TABLE_DIR = ROOT / "tables"


COLORS = {
    "none": "#6B7280",
    "host": "#2563EB",
    "gpu": "#DC2626",
    "aegis": "#059669",
    "static": "#7C3AED",
    "checkpoint": "#D97706",
    "ordinary": "#94A3B8",
    "recompute": "#F59E0B",
    "ink": "#111827",
    "grid": "#E5E7EB",
    "soft": "#F8FAFC",
}


@dataclass(frozen=True)
class Box:
    key: str
    label: str
    x: float
    y: float
    w: float = 1.9
    h: float = 0.72
    fill: str = "#EFF6FF"
    stroke: str = "#1D4ED8"


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""
    color: str = "#374151"
    dashed: bool = False


@dataclass(frozen=True)
class Diagram:
    figure: str
    stem: str
    title: str
    boxes: tuple[Box, ...]
    edges: tuple[Edge, ...]
    note: str = ""


def ensure_dirs() -> None:
    for path in (SRC_DIR, PNG_DIR, DATA_DIR, MEASURED_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_plot(fig: plt.Figure, stem: str) -> None:
    fig.savefig(PNG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(stem: str, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with (DATA_DIR / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_measured_csv(stem: str, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = MEASURED_DIR / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="both", labelsize=8, colors="#334155")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def add_data_badge(
    ax: plt.Axes,
    source: str,
    *,
    x: float = 0.99,
    y: float = 0.98,
    ha: str = "right",
    va: str = "top",
) -> None:
    if source == "measured":
        text = "measured data"
        color = "#047857"
        edge = "#6EE7B7"
        fill = "#ECFDF5"
    elif source == "mixed-preliminary":
        text = "mixed preliminary"
        color = "#6D28D9"
        edge = "#C4B5FD"
        fill = "#F5F3FF"
    elif source == "sglsim-preliminary":
        text = "SGLSim preliminary"
        color = "#0369A1"
        edge = "#7DD3FC"
        fill = "#F0F9FF"
    elif source == "trace-derived":
        text = "trace-derived data"
        color = "#0F766E"
        edge = "#5EEAD4"
        fill = "#F0FDFA"
    elif source == "model-based":
        text = "model-based policy"
        color = "#7C2D12"
        edge = "#FDBA74"
        fill = "#FFF7ED"
    elif source == "preliminary":
        text = "preliminary data"
        color = "#A16207"
        edge = "#FDE68A"
        fill = "#FFFBEB"
    else:
        text = "expected-trend template"
        color = "#64748B"
        edge = "#CBD5E1"
        fill = "#F8FAFC"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7,
        color=color,
        bbox=dict(boxstyle="round,pad=0.22", fc=fill, ec=edge, lw=0.8),
    )


def read_measured_rows(stem: str) -> list[dict[str, str]] | None:
    path = MEASURED_DIR / f"{stem}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows or None


def read_physical_copy_aggregate() -> dict[str, list[dict[str, str]]]:
    path = (
        ROOT.parent
        / "experiments"
        / "sglang_small"
        / "results"
        / "physical_kv_copy_aggregate"
        / "physical_kv_copy_aggregate.csv"
    )
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_path: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_path.setdefault(str(row.get("path") or ""), []).append(row)
    for path_rows in by_path.values():
        path_rows.sort(key=lambda row: float(row["tokens"]))
    return by_path


def read_physical_recovery_aggregate() -> dict[str, list[dict[str, str]]]:
    path = (
        ROOT.parent
        / "experiments"
        / "sglang_small"
        / "results"
        / "physical_recovery_path_aggregate"
        / "physical_recovery_path_aggregate.csv"
    )
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_action: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_action.setdefault(str(row.get("action") or ""), []).append(row)
    for action_rows in by_action.values():
        action_rows.sort(key=lambda row: float(row["tokens"]))
    return by_action


def read_p0_physical_bridge_summary() -> list[dict[str, str]]:
    path = (
        ROOT.parent
        / "experiments"
        / "sglsim_large"
        / "results"
        / "strict_final_readiness"
        / "p0_physical_bridge_summary.csv"
    )
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_sglang_physical_restore_rows() -> list[dict[str, str]]:
    root = (
        ROOT.parent
        / "experiments"
        / "sglang_small"
        / "results"
        / "physical_restore_evidence"
    )
    if not root.exists():
        return []
    paths: list[Path] = []
    primary = root / "physical_restore_runs.csv"
    if primary.exists():
        paths.append(primary)
    paths.extend(
        path
        for path in sorted(root.glob("*/physical_restore_runs.csv"))
        if path != primary
    )
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def physical_copy_point(
    by_path: dict[str, list[dict[str, str]]], path: str, tokens: int
) -> dict[str, str] | None:
    for row in by_path.get(path, []):
        if int(float(row.get("tokens", "0"))) == tokens:
            return row
    return None


def physical_recovery_point(
    by_action: dict[str, list[dict[str, str]]], action: str, tokens: int
) -> dict[str, str] | None:
    for row in by_action.get(action, []):
        if int(float(row.get("tokens", "0"))) == tokens:
            return row
    return None


def select_result_rows(
    stem: str, template_rows: Iterable[dict[str, object]]
) -> tuple[list[dict[str, object]], str]:
    template = list(template_rows)
    write_csv(stem, template)
    measured = read_measured_rows(stem)
    if measured is not None:
        tiers = {
            str(row.get("evidence_tier") or "").strip().lower()
            for row in measured
        }
        sources = {
            str(row.get("source") or "").strip().lower()
            for row in measured
        }
        tiers.discard("")
        sources.discard("")
        if not tiers or tiers == {"final"}:
            return [dict(row) for row in measured], "measured"
        has_model = any("model" in item for item in tiers | sources)
        if has_model:
            return [dict(row) for row in measured], "model-based"
        has_trace = any("trace" in item for item in tiers | sources)
        if has_trace:
            return [dict(row) for row in measured], "trace-derived"
        has_sglsim = any("sglsim" in item for item in tiers | sources)
        has_sglang = any("sglang" in item for item in tiers | sources)
        if has_sglsim and has_sglang:
            return [dict(row) for row in measured], "mixed-preliminary"
        if has_sglsim:
            return [dict(row) for row in measured], "sglsim-preliminary"
        return [dict(row) for row in measured], "preliminary"
    return template, "template"


def result_evidence_status(stem: str) -> str:
    rows = read_measured_rows(stem)
    if not rows:
        return "template-only-not-measured"
    tiers = {
        str(row.get("evidence_tier") or "").strip().lower()
        for row in rows
    }
    sources = {
        str(row.get("source") or "").strip().lower()
        for row in rows
    }
    tiers.discard("")
    sources.discard("")
    if not tiers or tiers == {"final"}:
        return "final-measured"
    has_model = any("model" in item for item in tiers | sources)
    if has_model:
        return "model-based-preliminary"
    has_trace = any("trace" in item for item in tiers | sources)
    if has_trace:
        return "trace-derived-preliminary"
    has_sglsim = any("sglsim" in item for item in tiers | sources)
    has_sglang = any("sglang" in item for item in tiers | sources)
    if has_sglsim and has_sglang:
        return "mixed-preliminary"
    if has_sglsim:
        return "sglsim-preliminary"
    return "preliminary-measured"


def as_float(row: dict[str, object], key: str) -> float:
    value = row[key]
    if value == "":
        return float("nan")
    return float(value)


def optional_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def ci_values(rows: list[dict[str, object]], metric: str) -> np.ndarray:
    key = f"{metric}_ci95"
    return np.array([max(0.0, optional_float(row, key)) for row in rows])


def error_bar_style() -> dict[str, object]:
    return {"ecolor": "#334155", "elinewidth": 0.8, "capthick": 0.8}


def as_int(row: dict[str, object], key: str) -> int:
    return int(float(row[key]))


def mx_style(fill: str, stroke: str) -> str:
    return (
        "rounded=1;whiteSpace=wrap;html=1;fontSize=13;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.4;"
    )


def write_drawio(diagram: Diagram) -> None:
    ids = {box.key: f"n{i + 2}" for i, box in enumerate(diagram.boxes)}
    cells = [
        '<mxCell id="0"/>',
        '<mxCell id="1" parent="0"/>',
    ]
    for box in diagram.boxes:
        value = html.escape(box.label).replace("\n", "&lt;br&gt;")
        style = mx_style(box.fill, box.stroke)
        cells.append(
            f'<mxCell id="{ids[box.key]}" value="{value}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{box.x * 120:.0f}" '
            f'y="{box.y * 95:.0f}" width="{box.w * 120:.0f}" '
            f'height="{box.h * 95:.0f}" as="geometry"/></mxCell>'
        )
    edge_id = len(cells) + 2
    for edge in diagram.edges:
        value = html.escape(edge.label)
        dashed = "dashed=1;" if edge.dashed else ""
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
            f"jettySize=auto;html=1;fontSize=11;strokeColor={edge.color};"
            f"endArrow=block;endFill=1;{dashed}"
        )
        cells.append(
            f'<mxCell id="e{edge_id}" value="{value}" style="{style}" edge="1" '
            f'parent="1" source="{ids[edge.src]}" target="{ids[edge.dst]}">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>'
        )
        edge_id += 1
    xml = (
        '<mxfile host="app.diagrams.net" modified="2026-06-21T00:00:00.000Z" '
        'agent="Codex" version="24.0.0" type="device">'
        f'<diagram id="{diagram.stem}" name="{html.escape(diagram.title)}">'
        '<mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="1400" pageHeight="900" math="0" shadow="0"><root>'
        + "".join(cells)
        + "</root></mxGraphModel></diagram></mxfile>\n"
    )
    (SRC_DIR / f"{diagram.stem}.drawio").write_text(xml)


def render_diagram(diagram: Diagram) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.set_facecolor("white")
    lookup = {box.key: box for box in diagram.boxes}

    for edge in diagram.edges:
        src = lookup[edge.src]
        dst = lookup[edge.dst]
        x0 = src.x + src.w / 2
        y0 = src.y + src.h / 2
        x1 = dst.x + dst.w / 2
        y1 = dst.y + dst.h / 2
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.2,
                color=edge.color,
                linestyle="--" if edge.dashed else "-",
                shrinkA=18,
                shrinkB=18,
            ),
        )
        if edge.label:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + 0.15,
                edge.label,
                ha="center",
                va="center",
                fontsize=7.3,
                color=edge.color,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.8),
            )

    for box in diagram.boxes:
        rect = plt.Rectangle(
            (box.x, box.y),
            box.w,
            box.h,
            facecolor=box.fill,
            edgecolor=box.stroke,
            linewidth=1.3,
            joinstyle="round",
        )
        ax.add_patch(rect)
        ax.text(
            box.x + box.w / 2,
            box.y + box.h / 2,
            box.label,
            ha="center",
            va="center",
            fontsize=8.4,
            color=COLORS["ink"],
            linespacing=1.12,
        )

    max_x = max(box.x + box.w for box in diagram.boxes) + 0.4
    max_y = max(box.y + box.h for box in diagram.boxes) + 0.25
    min_y = min(box.y for box in diagram.boxes) - 0.25
    ax.set_xlim(-0.2, max_x)
    ax.set_ylim(max_y, min_y)
    ax.axis("off")
    save_plot(fig, diagram.stem)


def diagrams() -> list[Diagram]:
    blue = ("#EFF6FF", "#2563EB")
    green = ("#ECFDF5", "#059669")
    orange = ("#FFF7ED", "#EA580C")
    purple = ("#F5F3FF", "#7C3AED")
    red = ("#FEF2F2", "#DC2626")
    slate = ("#F8FAFC", "#475569")
    return [
        Diagram(
            "Fig. 4",
            "fig_system_overview",
            "System overview",
            (
                Box("wf", "Workflow\nmetadata", 0, 1, fill=green[0], stroke=green[1]),
                Box("req", "SGLang\nrequest", 2.2, 1, fill=blue[0], stroke=blue[1]),
                Box("radix", "Radix node\nAegis metadata", 4.4, 1, fill=blue[0], stroke=blue[1]),
                Box("planner", "Placement\nplanner", 6.6, 1, fill=purple[0], stroke=purple[1]),
                Box("gpu", "GPUReplica\npeer rank", 8.8, 0.25, fill=red[0], stroke=red[1]),
                Box("host", "HostBackup\nwrite-through", 8.8, 1.75, fill=orange[0], stroke=orange[1]),
                Box("frontier", "ACKed safe\nfrontier", 11.0, 1, fill=green[0], stroke=green[1]),
                Box("retry", "Failure retry\nrestore or recompute", 13.2, 1, fill=slate[0], stroke=slate[1]),
            ),
            (
                Edge("wf", "req"),
                Edge("req", "radix"),
                Edge("radix", "planner"),
                Edge("planner", "gpu", "P2P copy"),
                Edge("planner", "host", "D2H ACK"),
                Edge("gpu", "frontier", "peer ACK"),
                Edge("host", "frontier", "Host ACK"),
                Edge("frontier", "retry"),
            ),
            "AegisKV is policy and metadata on SGLang's KV lifecycle, not a separate KV manager.",
        ),
        Diagram(
            "Fig. 5",
            "fig_backup_data_path",
            "Backup data path",
            (
                Box("src", "Source DP rank\nactive KV", 0, 1, fill=blue[0], stroke=blue[1]),
                Box("peerq", "Peer-copy\nqueue", 2.4, 0.2, fill=purple[0], stroke=purple[1]),
                Box("peer", "Peer DP rank\nGPUReplica", 4.8, 0.2, fill=red[0], stroke=red[1]),
                Box("d2h", "D2H write\nqueue", 2.4, 1.8, fill=orange[0], stroke=orange[1]),
                Box("host", "Same worker\nHost memory", 4.8, 1.8, fill=orange[0], stroke=orange[1]),
                Box("ack", "ACK queues\nfrontier registry", 7.2, 1, fill=green[0], stroke=green[1]),
                Box("retry", "Retry rank\nrestore/recompute", 9.8, 1, fill=slate[0], stroke=slate[1]),
            ),
            (
                Edge("src", "peerq", "P2P enqueue"),
                Edge("peerq", "peer", "copy"),
                Edge("src", "d2h", "write-through"),
                Edge("d2h", "host", "D2H"),
                Edge("peer", "ack", "peer ACK"),
                Edge("host", "ack", "Host ACK"),
                Edge("ack", "retry", "frontier lookup"),
                Edge("host", "retry", "H2D restore"),
                Edge("peer", "retry", "hit/restore"),
            ),
            "GPUReplica and HostBackup differ in latency, capacity, and failure domain.",
        ),
        Diagram(
            "Fig. 5b",
            "fig_hostbackup_visibility_pipeline",
            "Fixed HostBackup visibility pipeline",
            (
                Box("gpu", "GPU active KV", 0, 1),
                Box("insert", "Radix insert", 2, 1),
                Box("d2h", "Fixed write-through\nD2H queue", 4, 1, fill=orange[0], stroke=orange[1]),
                Box("ack", "Host ACK", 6.2, 1, fill=green[0], stroke=green[1]),
                Box("dir", "HostBackup\ndirectory entry", 8.3, 1, fill=green[0], stroke=green[1]),
                Box("fail", "Source GPU\nfailure", 10.6, 0.2, fill=red[0], stroke=red[1]),
                Box("retry", "Retry on healthy\nDP rank", 12.6, 0.2, fill=slate[0], stroke=slate[1]),
                Box("restore", "Host restore +\nsuffix recompute", 12.6, 1.8, fill=orange[0], stroke=orange[1]),
                Box("peerq", "Peer-copy queue", 4, 2.5, fill=purple[0], stroke=purple[1]),
                Box("peer", "Peer GPUReplica\nACK", 6.2, 2.5, fill=red[0], stroke=red[1]),
            ),
            (
                Edge("gpu", "insert"),
                Edge("insert", "d2h"),
                Edge("d2h", "ack"),
                Edge("ack", "dir"),
                Edge("dir", "fail"),
                Edge("fail", "retry"),
                Edge("retry", "restore", "lookup failed-rank Host KV"),
                Edge("gpu", "peerq", "optional", dashed=True),
                Edge("peerq", "peer", "P2P ACK", dashed=True),
                Edge("peer", "retry", "fast path", dashed=True),
            ),
            "Baseline pays the same D2H but lacks cross-rank frontier visibility.",
        ),
        Diagram(
            "Fig. 6",
            "fig_decision_loop",
            "Online decision loop",
            (
                Box("observe", "Observe state\nworkflow/KV/queues", 0, 1, fill=green[0], stroke=green[1]),
                Box("budget", "Choose budget\nand placement", 2.5, 0.2, fill=purple[0], stroke=purple[1]),
                Box("copy", "Physical copy\nHost or peer GPU", 5.0, 0.2, fill=orange[0], stroke=orange[1]),
                Box("ack", "ACK and publish\nsafe frontier", 7.5, 1, fill=green[0], stroke=green[1]),
                Box("restore", "Failure-time\nrestore/recompute", 5.0, 1.9, fill=red[0], stroke=red[1]),
                Box("state", "State vector\nh,g,q,p2p,h2d,dup", 2.5, 1.9, fill=slate[0], stroke=slate[1]),
            ),
            (
                Edge("observe", "budget"),
                Edge("budget", "copy"),
                Edge("copy", "ack"),
                Edge("ack", "restore", "on failure", dashed=True),
                Edge("restore", "state"),
                Edge("state", "observe"),
                Edge("ack", "observe", "next step"),
            ),
            "The controller updates only ACKed safe frontiers; pending copies are not recoverable.",
        ),
        Diagram(
            "Fig. 7",
            "fig_failure_timeline",
            "Failure recovery timeline",
            (
                Box("normal", "Normal decode", 0, 1),
                Box("copy", "Backup copy", 2, 1, fill=orange[0], stroke=orange[1]),
                Box("ack", "ACK", 4, 1, fill=green[0], stroke=green[1]),
                Box("failure", "Failure", 6, 1, fill=red[0], stroke=red[1]),
                Box("snapshot", "Snapshot\nlive requests", 8, 1, fill=slate[0], stroke=slate[1]),
                Box("lookup", "Frontier\nlookup", 10, 1, fill=green[0], stroke=green[1]),
                Box("recover", "Restore or\nrecompute", 12, 1, fill=purple[0], stroke=purple[1]),
                Box("resume", "Resume\nworkflow", 14, 1, fill=blue[0], stroke=blue[1]),
            ),
            (
                Edge("normal", "copy"),
                Edge("copy", "ack"),
                Edge("ack", "failure"),
                Edge("failure", "snapshot"),
                Edge("snapshot", "lookup"),
                Edge("lookup", "recover"),
                Edge("recover", "resume"),
            ),
            "The recovery path compares no backup, Host, GPUReplica, and AegisKV segment choices.",
        ),
        Diagram(
            "Fig. 7b",
            "fig_radix_lock_ack_eviction",
            "Radix lock, ACK, and eviction state machine",
            (
                Box("unlocked", "Device-resident\nunlocked", 0, 1),
                Box("locked", "Active request\nlock_ref > 0", 2.2, 0.2, fill=blue[0], stroke=blue[1]),
                Box("pending", "Host copy\npending", 4.6, 0.2, fill=orange[0], stroke=orange[1]),
                Box("safe", "Host ACKed\nsafe frontier", 7.0, 0.2, fill=green[0], stroke=green[1]),
                Box("hostonly", "GPU evicted\nHost-only", 9.4, 1, fill=orange[0], stroke=orange[1]),
                Box("restored", "H2D restored\ndevice-resident", 7.0, 1.9, fill=blue[0], stroke=blue[1]),
                Box("unsafe", "Host evicted\nunsafe", 4.6, 1.9, fill=red[0], stroke=red[1]),
            ),
            (
                Edge("unlocked", "locked", "inc_lock_ref"),
                Edge("locked", "unlocked", "dec_lock_ref", dashed=True),
                Edge("locked", "pending", "write_backup"),
                Edge("pending", "safe", "writing_check ACK"),
                Edge("safe", "hostonly", "evict GPU"),
                Edge("hostonly", "restored", "H2D load"),
                Edge("safe", "unsafe", "evict_host"),
                Edge("pending", "unsafe", "failure before ACK", dashed=True),
            ),
            "Locked nodes are not evicted; pending copies never become failure-safe.",
        ),
    ]


def plot_fig1() -> None:
    stem = "fig_motivation_equal_size_loss"
    template_labels = [
        "terminated\nbranch",
        "running\nsubagent",
        "last subagent\nbefore fan-in",
        "main\nresume",
        "synthesis\ntail",
        "shared\nprefix",
    ]
    values = np.array([0.08, 0.55, 1.35, 1.55, 1.82, 2.15])
    template_rows = [{"segment": template_labels[i].replace("\n", " "), "delta_wct_norm": float(values[i])} for i in range(len(template_labels))]
    rows, source = select_result_rows(stem, template_rows)
    label_map = {
        "terminated branch": "terminated\nbranch",
        "running subagent": "running\nsubagent",
        "last subagent before fan-in": "last subagent\nbefore fan-in",
        "main resume": "main\nresume",
        "synthesis tail": "synthesis\ntail",
        "shared prefix": "shared\nprefix",
        "non-critical fan-out": "non-critical\nfan-out",
        "critical long-decode fan-out": "critical long-decode\nfan-out",
        "critical entry singleton": "critical entry\nsingleton",
        "critical fan-in singleton": "critical fan-in\nsingleton",
        "critical synthesis singleton": "critical synthesis\nsingleton",
    }
    labels = [
        label_map.get(str(row["segment"]), str(row["segment"]).replace(" ", "\n", 1))
        for row in rows
    ]
    values = np.array([as_float(row, "delta_wct_norm") for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(
        labels,
        values,
        yerr=ci_values(rows, "delta_wct_norm"),
        capsize=3,
        error_kw=error_bar_style(),
        color=[COLORS["ordinary"], COLORS["host"], COLORS["static"], COLORS["gpu"], COLORS["aegis"], COLORS["checkpoint"]],
    )
    style_axes(ax, "normalized WCT penalty")
    ax.set_title("Equal-size KV loss has non-uniform workflow impact", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig1b() -> None:
    stem = "fig_critical_backup_priority"
    budget = np.linspace(0.0, 1.0, 11)
    critical_first = 1.0 - 0.48 * (1.0 - np.exp(-3.2 * budget))
    size_first = 1.0 - 0.24 * (1.0 - np.exp(-1.8 * budget))
    random = 1.0 - 0.18 * (1.0 - np.exp(-1.2 * budget))
    noncritical_first = 1.0 - 0.08 * (1.0 - np.exp(-1.5 * budget))
    template_rows = []
    for policy, values in (
        ("critical-first", critical_first),
        ("size-first", size_first),
        ("random", random),
        ("non-critical-first", noncritical_first),
    ):
        for b, value in zip(budget, values):
            template_rows.append(
                {
                    "backup_budget_fraction": round(float(b), 2),
                    "placement_policy": policy,
                    "normalized_wct_after_failure": float(value),
                }
            )
    rows, source = select_result_rows(stem, template_rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.35))
    colors = {
        "critical-first": COLORS["aegis"],
        "size-first": COLORS["static"],
        "random": COLORS["host"],
        "non-critical-first": COLORS["ordinary"],
    }
    plotted_values: list[float] = []
    for policy in ("critical-first", "size-first", "random", "non-critical-first"):
        group = [
            row for row in rows if str(row["placement_policy"]) == policy
        ]
        group = sorted(group, key=lambda row: as_float(row, "backup_budget_fraction"))
        x_values = [as_float(row, "backup_budget_fraction") for row in group]
        y_values = [as_float(row, "normalized_wct_after_failure") for row in group]
        plotted_values.extend(y_values)
        yerr = ci_values(group, "normalized_wct_after_failure")
        err_style = error_bar_style() if np.any(yerr > 0) else {}
        ax.errorbar(
            x_values,
            y_values,
            yerr=yerr if np.any(yerr > 0) else None,
            marker="o",
            linewidth=2.0 if policy == "critical-first" else 1.5,
            capsize=2 if np.any(yerr > 0) else 0,
            label=policy,
            color=colors[policy],
            **err_style,
        )
    ax.set_xlabel("GPUReplica budget fraction", fontsize=9)
    style_axes(ax, "normalized failure-induced WCT penalty")
    lower = max(0.0, min(plotted_values or [0.48]) - 0.06)
    ax.set_ylim(lower, 1.04)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Critical-agent KV should be protected first", fontsize=11)
    add_data_badge(ax, source, x=0.02, y=0.08, ha="left", va="bottom")
    save_plot(fig, stem)


def plot_fig1c() -> None:
    stem = "fig_critical_agent_profile"
    template_rows = [
        {
            "stage_slot": "S0 singleton",
            "stage_id": 0,
            "stage_width": 1,
            "stage_local_agent_index": 0,
            "critical_agent": True,
            "decode_dominant": False,
            "mean_output_tokens": 42.0,
            "mean_latency_ms": 95.0,
            "stage_critical_path_share": 1.0,
        },
        {
            "stage_slot": "S1 fan-out long Critical",
            "stage_id": 1,
            "stage_width": 3,
            "stage_local_agent_index": 0,
            "critical_agent": True,
            "decode_dominant": True,
            "mean_output_tokens": 212.0,
            "mean_latency_ms": 323.0,
            "stage_critical_path_share": 1.0,
        },
        {
            "stage_slot": "S1 fan-out Non-Critical 1",
            "stage_id": 1,
            "stage_width": 3,
            "stage_local_agent_index": 1,
            "critical_agent": False,
            "decode_dominant": False,
            "mean_output_tokens": 41.0,
            "mean_latency_ms": 98.0,
            "stage_critical_path_share": 0.30,
        },
        {
            "stage_slot": "S1 fan-out Non-Critical 2",
            "stage_id": 1,
            "stage_width": 3,
            "stage_local_agent_index": 2,
            "critical_agent": False,
            "decode_dominant": False,
            "mean_output_tokens": 44.0,
            "mean_latency_ms": 100.0,
            "stage_critical_path_share": 0.31,
        },
        {
            "stage_slot": "S2 singleton",
            "stage_id": 2,
            "stage_width": 1,
            "stage_local_agent_index": 0,
            "critical_agent": True,
            "decode_dominant": False,
            "mean_output_tokens": 44.0,
            "mean_latency_ms": 99.0,
            "stage_critical_path_share": 1.0,
        },
    ]
    rows, source = select_result_rows(stem, template_rows)
    rows = sorted(
        rows,
        key=lambda row: (
            as_int(row, "stage_id"),
            as_int(row, "stage_local_agent_index"),
        ),
    )
    labels = [
        str(row["stage_slot"])
        .replace(" fan-out ", "\n")
        .replace(" singleton", "\nsingleton")
        .replace("Non-Critical ", "NonCrit ")
        for row in rows
    ]
    latency = np.array([as_float(row, "mean_latency_ms") for row in rows])
    output_tokens = np.array([as_float(row, "mean_output_tokens") for row in rows])
    critical = [
        str(row.get("critical_agent", "")).strip().lower() in {"true", "1", "yes"}
        for row in rows
    ]
    decode = [
        str(row.get("decode_dominant", "")).strip().lower() in {"true", "1", "yes"}
        for row in rows
    ]
    colors = [
        COLORS["gpu"] if is_decode else COLORS["aegis"] if is_critical else COLORS["ordinary"]
        for is_critical, is_decode in zip(critical, decode)
    ]
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(7.9, 3.85))
    yerr = ci_values(rows, "mean_latency_ms")
    ax1.bar(
        x,
        latency,
        yerr=yerr if np.any(yerr > 0) else None,
        capsize=2.5 if np.any(yerr > 0) else 0,
        error_kw=error_bar_style(),
        color=colors,
        alpha=0.92,
        label="latency",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.7)
    ax1.set_ylim(0, float(max(latency)) * 1.18)
    style_axes(ax1, "mean request latency (ms)")
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        output_tokens,
        color=COLORS["checkpoint"],
        marker="o",
        linewidth=1.8,
        label="output tokens",
    )
    ax2.set_ylabel("mean output tokens", fontsize=9)
    ax2.tick_params(axis="y", labelsize=8, colors="#334155")
    for idx, row in enumerate(rows):
        share = optional_float(row, "stage_critical_path_share")
        if decode[idx]:
            ax1.text(
                idx,
                latency[idx] + max(latency) * 0.025,
                f"share {share:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                color=COLORS["gpu"],
            )
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["aegis"], label="Critical singleton"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["gpu"], label="long-decode Critical"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["ordinary"], label="Non-Critical"),
        plt.Line2D([0], [0], color=COLORS["checkpoint"], marker="o", label="output tokens"),
    ]
    ax1.legend(
        handles=legend_items,
        fontsize=6.7,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        handlelength=1.4,
        columnspacing=1.0,
    )
    ax1.set_title("1-3-1-3-1 workload exposes long-decode Critical agents", fontsize=11, pad=28)
    add_data_badge(ax1, source, y=0.90)
    fig.subplots_adjust(top=0.73)
    save_plot(fig, stem)


def plot_fig2() -> None:
    stem = "fig_restore_recompute_boundary"
    tokens = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    recompute = 5.5 + tokens / 900
    host = 12 + tokens / 1850
    gpu = 3.0 + tokens / 3100 + np.maximum(tokens - 4096, 0) / 1200
    aegis = np.minimum.reduce([recompute, host, gpu])
    template_rows = []
    for t, r, h, g, a in zip(tokens, recompute, host, gpu, aegis):
        template_rows.append({"tokens": int(t), "recompute_ms": r, "host_restore_ms": h, "gpu_replica_ms": g, "aegis_lower_envelope_ms": a})
    rows, source = select_result_rows(stem, template_rows)
    tokens = np.array([as_float(row, "tokens") for row in rows])
    recompute = np.array([as_float(row, "recompute_ms") for row in rows])
    host = np.array([as_float(row, "host_restore_ms") for row in rows])
    gpu = np.array([as_float(row, "gpu_replica_ms") for row in rows])
    aegis = np.array([as_float(row, "aegis_lower_envelope_ms") for row in rows])
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(tokens, gpu, marker="o", label="GPUReplica restore", color=COLORS["gpu"])
    ax.plot(tokens, host, marker="o", label="Host restore", color=COLORS["host"])
    ax.plot(tokens, recompute, marker="o", label="recompute", color=COLORS["checkpoint"])
    ax.plot(tokens, aegis, lw=2.8, label="AegisKV choice", color=COLORS["aegis"])
    recovery_by_action = read_physical_recovery_aggregate()
    copy_by_path = read_physical_copy_aggregate()
    overlay_specs = [
        (
            recovery_by_action.get("host_restore_h2d") or copy_by_path.get("h2d_pinned", []),
            "physical Host restore",
            "#0F766E",
            "^",
        ),
        (
            recovery_by_action.get("gpu_replica_restore_p2p")
            or copy_by_path.get("p2p_gpu0_to_gpu1", []),
            "physical GPUReplica restore",
            "#7C2D12",
            "v",
        ),
    ]
    for copy_rows, label, color, marker in overlay_specs:
        if not copy_rows:
            continue
        copy_tokens = np.array([as_float(row, "tokens") for row in copy_rows])
        copy_ms = np.array([as_float(row, "p50_ms_mean") for row in copy_rows])
        copy_ci = np.array([as_float(row, "p50_ms_ci95") for row in copy_rows])
        ax.errorbar(
            copy_tokens,
            copy_ms,
            yerr=copy_ci,
            fmt=f"{marker}--",
            linewidth=1.1,
            markersize=4,
            capsize=2,
            color=color,
            alpha=0.86,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("segment tokens", fontsize=9)
    style_axes(ax, "recovery time (ms)")
    ax.legend(fontsize=7, ncol=3, frameon=False)
    ax.set_title("Restore vs recompute boundary", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig3() -> None:
    stem = "fig_static_gpu_replica_budget_gap"
    alpha = np.linspace(0, 1, 11)
    workloads = {
        "hot tails": 1.18 - 0.55 * np.exp(-((alpha - 0.65) ** 2) / 0.08) + 0.18 * alpha,
        "long prefixes": 1.05 - 0.28 * np.exp(-((alpha - 0.25) ** 2) / 0.05) + 0.38 * alpha,
        "fan-out skew": 1.25 - 0.45 * np.exp(-((alpha - 0.45) ** 2) / 0.06) + 0.10 * alpha,
    }
    template_rows = []
    for name, values in workloads.items():
        for a, v in zip(alpha, values):
            template_rows.append({"gpu_replica_budget_fraction": round(float(a), 2), "workload": name, "normalized_wct": float(v)})
    template_rows.append({"gpu_replica_budget_fraction": "dynamic", "workload": "AegisKV", "normalized_wct": 0.72})
    rows, source = select_result_rows(stem, template_rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    by_workload: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_workload.setdefault(str(row["workload"]), []).append(row)
    for name, group in by_workload.items():
        if name == "AegisKV":
            continue
        group = sorted(group, key=lambda row: as_float(row, "gpu_replica_budget_fraction"))
        ax.plot(
            [as_float(row, "gpu_replica_budget_fraction") for row in group],
            [as_float(row, "normalized_wct") for row in group],
            marker="o",
            label=name,
        )
    aegis_rows = by_workload.get("AegisKV", [])
    if aegis_rows:
        ax.axhline(as_float(aegis_rows[0], "normalized_wct"), color=COLORS["aegis"], lw=2.4, ls="--", label="dynamic AegisKV")
    ax.set_xlabel("GPUReplica HBM budget fraction", fontsize=9)
    style_axes(ax, "normalized cost")
    ax.set_ylim(0.932, 1.012)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    ax.set_title("Static GPUReplica budget is workload-dependent", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig3b() -> None:
    stem = "fig_alpha_workload_failure_policy"
    failure_rates = np.array([0.001, 0.003, 0.01, 0.03, 0.10])
    pressures = {
        "low critical pressure": 0.25,
        "medium critical pressure": 0.55,
        "high critical pressure": 0.85,
    }
    template_rows = []
    for label, pressure in pressures.items():
        alpha = np.clip(
            0.05
            + 0.58 * pressure
            + 0.22 * np.log10(1.0 + failure_rates * 100.0),
            0.04,
            0.82,
        )
        for rate, value in zip(failure_rates, alpha):
            template_rows.append(
                {
                    "failure_rate_per_window": float(rate),
                    "critical_pressure": label,
                    "alpha_backup_fraction": float(value),
                }
            )
    rows, source = select_result_rows(stem, template_rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    for label, color in (
        ("low critical pressure", COLORS["host"]),
        ("medium critical pressure", COLORS["static"]),
        ("high critical pressure", COLORS["aegis"]),
    ):
        group = [
            row for row in rows if str(row["critical_pressure"]) == label
        ]
        group = sorted(group, key=lambda row: as_float(row, "failure_rate_per_window"))
        ax.plot(
            [as_float(row, "failure_rate_per_window") for row in group],
            [as_float(row, "alpha_backup_fraction") for row in group],
            marker="o",
            label=label,
            color=color,
        )
    ax.set_xscale("log")
    ax.set_xlabel("failure rate per decision window", fontsize=9)
    style_axes(ax, r"chosen backup ratio $\alpha$")
    ax.set_ylim(0.0, 0.9)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(r"Adaptive $\alpha$ depends on workload and failure risk", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig3c() -> None:
    stem = "fig_online_alpha_adaptation"
    template_rows = []
    for request_class, pressure, alpha_values in (
        ("Non-Critical fan-out", 0.18, [0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Singleton Critical", 0.75, [0.65, 0.80, 0.85, 0.85, 0.90]),
        ("Long-decode Critical", 0.90, [0.65, 0.80, 0.85, 0.85, 0.90]),
    ):
        for failure_rate, alpha in zip((0.001, 0.003, 0.01, 0.03, 0.10), alpha_values):
            template_rows.append(
                {
                    "request_class": request_class,
                    "failure_rate_per_window": failure_rate,
                    "critical_pressure": pressure,
                    "policy_alpha": alpha,
                    "effective_gpu_replica_alpha": alpha,
                    "online_observed_alpha": alpha if failure_rate == 0.01 else "",
                    "observed_request_rows": 1,
                    "observed_mean_output_tokens": 48,
                    "observed_mean_latency_ms": 100,
                }
            )
    rows, source = select_result_rows(stem, template_rows)
    fig, ax = plt.subplots(figsize=(6.8, 3.45))
    def series_key(row: dict[str, object]) -> str:
        label = str(row.get("series_label") or "").strip()
        return label or str(row["request_class"])

    def series_style(label: str) -> tuple[str, str, str]:
        lowered = label.lower()
        if "non-critical" in lowered:
            return COLORS["ordinary"], "s", "--"
        if "critical agents" in lowered and "low load" in lowered:
            return COLORS["static"], "o", "-"
        if "critical agents" in lowered:
            return COLORS["aegis"], "o", "-"
        if "singleton" in lowered and "low load" in lowered:
            return COLORS["host"], "o", "-"
        if "singleton" in lowered:
            return COLORS["aegis"], "o", "-"
        if "long-decode" in lowered and "low load" in lowered:
            return COLORS["static"], "^", "-"
        return COLORS["gpu"], "^", "-"

    series_order = sorted(
        {series_key(row) for row in rows},
        key=lambda label: (
            "non-critical" not in label.lower(),
            "high load" in label.lower(),
            "long-decode" in label.lower(),
            label,
        ),
    )
    for series in series_order:
        group = sorted(
            [row for row in rows if series_key(row) == series],
            key=lambda row: as_float(row, "failure_rate_per_window"),
        )
        if not group:
            continue
        color, marker, linestyle = series_style(series)
        pressure = float(np.mean([as_float(row, "critical_pressure") for row in group]))
        if "non-critical" in series.lower():
            label = "Non-Critical fan-out (effective alpha=0)"
        else:
            label = f"{series} (pressure={pressure:.2f})"
        ax.plot(
            [as_float(row, "failure_rate_per_window") for row in group],
            [as_float(row, "effective_gpu_replica_alpha") for row in group],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        observed = [
            row
            for row in group
            if str(row.get("online_observed_alpha", "")).strip()
        ]
        if observed:
            ax.scatter(
                [as_float(row, "failure_rate_per_window") for row in observed],
                [as_float(row, "online_observed_alpha") for row in observed],
                s=74,
                facecolors="white",
                edgecolors=color,
                linewidths=1.8,
                zorder=5,
            )
    ax.set_xscale("log")
    ax.set_xlabel("failure rate per decision window", fontsize=9)
    style_axes(ax, r"effective GPUReplica budget $\alpha$")
    ax.set_ylim(-0.04, 0.98)
    ax.legend(
        fontsize=6.6,
        frameon=False,
        loc="center right",
        bbox_to_anchor=(0.98, 0.42),
    )
    ax.set_title("Online MDP alpha adapts to workload pressure and failure risk", fontsize=11)
    add_data_badge(ax, source, x=0.02, ha="left")
    save_plot(fig, stem)


def plot_fig3d() -> None:
    stem = "fig_staged_alpha_risk_tradeoff"
    template_rows = []
    for failure_rate, best_alpha in (
        (0.001, 0.0),
        (0.003, 0.0),
        (0.01, 0.2),
        (0.03, 0.5),
        (0.10, 0.8),
    ):
        for alpha in np.linspace(0, 1, 11):
            normalized = 1.0 + 0.010 * alpha - failure_rate * 0.28 * alpha
            template_rows.append(
                {
                    "failure_rate_per_window": failure_rate,
                    "alpha_backup_fraction": round(float(alpha), 1),
                    "normalized_expected_wct": float(normalized),
                    "normalized_expected_wct_ci95": 0.0,
                    "expected_wct_s": float(normalized),
                    "normal_backup_overhead_s": 0.0,
                    "expected_failure_recovery_s": 0.0,
                    "protected_critical_token_fraction": round(float(alpha), 1),
                    "optimal_alpha_for_rate": best_alpha,
                }
            )
    rows, source = select_result_rows(stem, template_rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.45))
    palette = {
        0.001: "#64748B",
        0.003: "#0F766E",
        0.01: "#2563EB",
        0.03: "#B45309",
        0.10: "#DC2626",
    }
    for failure_rate in sorted(
        {round(as_float(row, "failure_rate_per_window"), 3) for row in rows}
    ):
        group = sorted(
            [
                row
                for row in rows
                if round(as_float(row, "failure_rate_per_window"), 3) == failure_rate
            ],
            key=lambda row: as_float(row, "alpha_backup_fraction"),
        )
        if not group:
            continue
        xs = [as_float(row, "alpha_backup_fraction") for row in group]
        ys = [as_float(row, "normalized_expected_wct") for row in group]
        color = palette.get(failure_rate, COLORS["aegis"])
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.9,
            markersize=4.2,
            color=color,
            label=f"failure={failure_rate:g}",
        )
        optimal_alpha = optional_float(group[0], "optimal_alpha_for_rate")
        optimal_rows = [
            row
            for row in group
            if abs(as_float(row, "alpha_backup_fraction") - optimal_alpha) < 1e-9
        ]
        if optimal_rows:
            ax.scatter(
                [optimal_alpha],
                [as_float(optimal_rows[0], "normalized_expected_wct")],
                marker="*",
                s=95,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
            )
    y_values = [as_float(row, "normalized_expected_wct") for row in rows]
    y_min, y_max = min(y_values), max(y_values)
    margin = max((y_max - y_min) * 0.18, 0.002)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel(r"GPUReplica HBM ratio $\alpha$", fontsize=9)
    style_axes(ax, "normalized expected WCT")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="best")
    ax.set_title("1-3-1-3-1 alpha optimum shifts with failure risk", fontsize=11)
    add_data_badge(ax, source, x=0.02, ha="left")
    save_plot(fig, stem)


def plot_fig8() -> None:
    stem = "fig_overall_wct"
    template_policies = ["none", "periodic", "uniform\nHost", "uniform\nGPU", "request", "agent", "static\ncritical", "AegisKV"]
    values = [1.00, 0.84, 0.72, 0.70, 0.78, 0.68, 0.61, 0.47]
    template_rows = [{"policy": p.replace("\n", " "), "normalized_wct": v} for p, v in zip(template_policies, values)]
    rows, source = select_result_rows(stem, template_rows)
    policies = [str(row["policy"]).replace("uniform ", "uniform\n").replace("static ", "static\n") for row in rows]
    values = [as_float(row, "normalized_wct") for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 3.35))
    colors = [COLORS["none"], COLORS["checkpoint"], COLORS["host"], COLORS["gpu"], COLORS["ordinary"], COLORS["static"], "#0F766E", COLORS["aegis"]]
    has_failure_exposed = any(
        optional_float(row, "failure_exposed_restore_norm", 0.0) > 0
        for row in rows
    )
    if has_failure_exposed:
        x = np.arange(len(policies))
        restore_values = [
            optional_float(row, "failure_exposed_restore_norm", 0.0)
            for row in rows
        ]
        ax.bar(
            x - 0.18,
            values,
            0.36,
            yerr=ci_values(rows, "normalized_wct"),
            capsize=3,
            error_kw=error_bar_style(),
            label="total WCT",
            color="#93C5FD",
        )
        ax.bar(
            x + 0.18,
            restore_values,
            0.36,
            yerr=ci_values(rows, "failure_exposed_restore_norm"),
            capsize=3,
            error_kw=error_bar_style(),
            label="failure-exposed restore",
            color=COLORS["aegis"],
        )
        ax.set_xticks(x, policies)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    else:
        ax.bar(
            policies,
            values,
            yerr=ci_values(rows, "normalized_wct"),
            capsize=3,
            error_kw=error_bar_style(),
            color=[colors[i % len(colors)] for i in range(len(policies))],
        )
    style_axes(ax, "normalized cost")
    ax.set_ylim(0, 1.12)
    ax.set_title("Total WCT vs. failure-exposed recovery cost", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig9() -> None:
    stem = "fig_recovery_latency"
    policies = ["none", "Host", "GPUReplica", "AegisKV"]
    retry = np.array([100, 58, 43, 31])
    critical = np.array([100, 63, 49, 34])
    template_rows = [{"policy": p, "retry_latency_norm": float(r), "critical_agent_latency_norm": float(c)} for p, r, c in zip(policies, retry, critical)]
    rows, source = select_result_rows(stem, template_rows)
    policies = [str(row["policy"]) for row in rows]
    retry = np.array([as_float(row, "retry_latency_norm") for row in rows])
    critical = np.array([as_float(row, "critical_agent_latency_norm") for row in rows])
    x = np.arange(len(policies))
    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    ax.bar(
        x - 0.18,
        retry,
        0.36,
        yerr=ci_values(rows, "retry_latency_norm"),
        capsize=3,
        error_kw=error_bar_style(),
        label="retry latency",
        color=COLORS["host"],
    )
    ax.bar(
        x + 0.18,
        critical,
        0.36,
        yerr=ci_values(rows, "critical_agent_latency_norm"),
        capsize=3,
        error_kw=error_bar_style(),
        label="critical-agent latency",
        color=COLORS["aegis"],
    )
    ax.set_xticks(x, policies)
    style_axes(ax, "normalized latency")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Recovery latency and retry speedup", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig10() -> None:
    stem = "fig_recovery_path_breakdown"
    policies = ["Host only", "GPUReplica only", "Static critical", "AegisKV"]
    peer = np.array([0.00, 0.58, 0.38, 0.36])
    host = np.array([0.63, 0.00, 0.28, 0.42])
    ordinary = np.array([0.12, 0.10, 0.10, 0.08])
    recompute = 1 - peer - host - ordinary
    template_rows = []
    for i, p in enumerate(policies):
        template_rows.append({"policy": p, "peer_gpu": peer[i], "host_h2d": host[i], "ordinary_cache": ordinary[i], "recompute": recompute[i]})
    rows, source = select_result_rows(stem, template_rows)
    policies = [str(row["policy"]) for row in rows]
    display_policies = [
        policy.replace("uniform ", "uniform\n")
        .replace("online DP2 ", "online DP2\n")
        .replace("online DP4 ", "online DP4\n")
        for policy in policies
    ]
    peer = np.array([as_float(row, "peer_gpu") for row in rows])
    host = np.array([as_float(row, "host_h2d") for row in rows])
    ordinary = np.array([as_float(row, "ordinary_cache") for row in rows])
    recompute = np.array([as_float(row, "recompute") for row in rows])
    x = np.arange(len(policies))
    fig, ax = plt.subplots(figsize=(7.4, 3.55))
    bottom = np.zeros(len(policies))
    for label, vals, color in [
        ("peer_gpu", peer, COLORS["gpu"]),
        ("host_h2d", host, COLORS["host"]),
        ("ordinary_cache", ordinary, COLORS["ordinary"]),
        ("recompute", recompute, COLORS["checkpoint"]),
    ]:
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(x, display_policies)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylim(0, 1)
    style_axes(ax, "recovery path ratio")
    legend = ax.legend(fontsize=7, ncol=2, frameon=True, loc="upper left")
    legend.get_frame().set_facecolor("#FFFFFF")
    legend.get_frame().set_edgecolor("#E5E7EB")
    legend.get_frame().set_linewidth(0.6)
    ax.set_title("Recovery path breakdown", fontsize=11)
    recovery_by_action = read_physical_recovery_aggregate()
    copy_by_path = read_physical_copy_aggregate()
    h2d_8192 = physical_recovery_point(
        recovery_by_action, "host_restore_h2d", 8192
    ) or physical_copy_point(copy_by_path, "h2d_pinned", 8192)
    p2p_8192 = physical_recovery_point(
        recovery_by_action, "gpu_replica_restore_p2p", 8192
    ) or physical_copy_point(copy_by_path, "p2p_gpu0_to_gpu1", 8192)
    if h2d_8192 and p2p_8192:
        ax.text(
            0.99,
            0.05,
            f"physical restore @8192 tok: GPUReplica {as_float(p2p_8192, 'p50_ms_mean'):.2f} ms; Host {as_float(h2d_8192, 'p50_ms_mean'):.2f} ms",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="#CBD5E1", lw=0.6),
        )
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig10b() -> None:
    stem = "fig_shared_prefix_marginal"
    overlap = np.array([0, 15, 30, 50, 70, 90])
    dup_no = 1.00 + overlap / 70
    dup_marginal = 1.00 + overlap / 240
    coverage = 0.88 - 0.02 * np.sin(overlap / 90 * math.pi)
    template_rows = [{"shared_prefix_pct": int(o), "dup_no_dedup": float(a), "dup_marginal": float(b), "coverage_marginal": float(c)} for o, a, b, c in zip(overlap, dup_no, dup_marginal, coverage)]
    rows, source = select_result_rows(stem, template_rows)
    rows = sorted(rows, key=lambda row: as_float(row, "shared_prefix_pct"))
    overlap = np.array([as_float(row, "shared_prefix_pct") for row in rows])
    dup_no = np.array([as_float(row, "dup_no_dedup") for row in rows])
    dup_marginal = np.array([as_float(row, "dup_marginal") for row in rows])
    coverage = np.array([as_float(row, "coverage_marginal") for row in rows])
    fig, ax1 = plt.subplots(figsize=(6.4, 3.45))
    ax1.plot(overlap, dup_no, marker="o", label="no dedup bytes", color=COLORS["gpu"])
    ax1.plot(overlap, dup_marginal, marker="o", label="marginal bytes", color=COLORS["aegis"])
    ax1.set_xlabel("shared prefix ratio (%)", fontsize=9)
    style_axes(ax1, "normalized protected bytes")
    if source != "template":
        ax1.set_ylim(0.9, max(float(np.nanmax(dup_no)), float(np.nanmax(dup_marginal))) * 1.15)
    ax2 = ax1.twinx()
    ax2.plot(overlap, coverage, marker="s", label="recovery coverage", color=COLORS["host"])
    if source == "template":
        ax2.set_ylim(0.75, 0.95)
    else:
        cov_min = float(np.nanmin(coverage))
        cov_max = float(np.nanmax(coverage))
        pad = max(0.01, (cov_max - cov_min) * 0.2)
        ax2.set_ylim(max(0.0, cov_min - pad), min(1.02, cov_max + pad))
    ax2.set_ylabel("coverage", fontsize=9)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, frameon=False, loc="upper left")
    ax1.set_title("Shared-prefix marginal accounting", fontsize=11)
    add_data_badge(ax1, source)
    save_plot(fig, stem)


def plot_fig11() -> None:
    stem = "fig_normal_path_overhead"
    policies = ["none", "uniform Host", "uniform GPU", "AegisKV"]
    p99 = np.array([0, 5, 15, 6])
    write_amp = np.array([0, 1.9, 1.6, 1.18])
    template_rows = [{"policy": p, "p99_overhead_pct": float(o), "write_amplification": float(w)} for p, o, w in zip(policies, p99, write_amp)]
    rows, source = select_result_rows(stem, template_rows)
    label_map = {
        "SGLSim baseline": "Sim\nbase",
        "SGLSim AegisKV": "Sim\nAegis",
        "SGLang baseline": "SG\nbase",
        "SGLang HiCache WT": "SG\nHiC\nWT",
    }
    policies = [
        label_map.get(str(row["policy"]), str(row["policy"]).replace("uniform ", "uniform\n"))
        for row in rows
    ]
    p99 = np.array([as_float(row, "p99_overhead_pct") for row in rows])
    write_amp = np.array([as_float(row, "write_amplification") for row in rows])
    x = np.arange(len(policies))
    fig, ax1 = plt.subplots(figsize=(7.8, 3.65))
    ax1.bar(
        x - 0.18,
        p99,
        0.36,
        yerr=ci_values(rows, "p99_overhead_pct"),
        capsize=3,
        error_kw=error_bar_style(),
        label="p99 overhead",
        color=COLORS["host"],
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, fontsize=7)
    style_axes(ax1, "p99 overhead (%)")
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.18,
        write_amp,
        0.36,
        yerr=ci_values(rows, "write_amplification"),
        capsize=3,
        error_kw=error_bar_style(),
        label="write amp",
        color=COLORS["checkpoint"],
    )
    ax2.set_ylabel("write amplification", fontsize=9)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    ax1.set_title("Normal-path overhead", fontsize=11)
    add_data_badge(ax1, source)
    save_plot(fig, stem)


def plot_fig11b() -> None:
    stem = "fig_hostbackup_visibility"
    template_policies = ["baseline\ninvisible", "Host visible\nno GPU", "GPUReplica\nonly", "AegisKV"]
    overhead = np.array([0.0, 1.2, 7.8, 8.5])
    retry = np.array([100, 62, 54, 36])
    template_rows = [{"policy": p.replace("\n", " "), "p99_overhead_relative_pct": float(o), "retry_latency_norm": float(r)} for p, o, r in zip(template_policies, overhead, retry)]
    rows, source = select_result_rows(stem, template_rows)
    policies = []
    for row in rows:
        raw_label = str(row["policy"])
        label_map = {
            "baseline write-through invisible": "baseline\nwrite-through\ninvisible",
            "Host visible only": "Host visible\nonly",
            "GPUReplica only": "GPUReplica\nonly",
            "SGLang invisible control": "SGLang\ninvisible\ncontrol",
            "SGLang remote restore": "SGLang\nremote\nrestore",
            "SGLang remote_backup lookup disabled": "SGLang\nlookup\ndisabled",
            "SGLang remote_backup lookup enabled": "SGLang\nlookup\nenabled",
        }
        label = label_map.get(raw_label, raw_label)
        policies.append(label)
    overhead = np.array([as_float(row, "p99_overhead_relative_pct") for row in rows])
    retry = np.array([as_float(row, "retry_latency_norm") for row in rows])
    x = np.arange(len(policies))
    fig, ax1 = plt.subplots(figsize=(7.35, 3.55))
    ax1.bar(
        x - 0.18,
        overhead,
        0.36,
        yerr=ci_values(rows, "p99_overhead_relative_pct"),
        capsize=3,
        error_kw=error_bar_style(),
        label="extra p99 overhead",
        color=COLORS["ordinary"],
    )
    ax1.set_xticks(x, policies)
    style_axes(ax1, "extra p99 overhead (%)")
    ax1.set_ylim(0, max(6.0, float(np.nanmax(overhead)) * 1.2))
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.18,
        retry,
        0.36,
        yerr=ci_values(rows, "retry_latency_norm"),
        capsize=3,
        error_kw=error_bar_style(),
        label="retry latency",
        color=COLORS["aegis"],
    )
    ax2.set_ylabel("retry latency (normalized)", fontsize=9)
    ax2.set_ylim(0, max(1.2, float(np.nanmax(retry)) * 1.15))
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.98),
        ncol=2,
    )
    ax1.set_title("HostBackup visibility under fixed write-through", fontsize=11)
    add_data_badge(ax1, source)
    save_plot(fig, stem)


def plot_fig12() -> None:
    stem = "fig_ablation"
    template_labels = ["full", "no\nlifecycle", "no\nsegment", "no\nrestore", "no\npressure", "no\nhysteresis", "no critical\nrestore"]
    values = [0.47, 0.67, 0.61, 0.71, 0.64, 0.56, 0.59]
    template_rows = [{"variant": l.replace("\n", " "), "normalized_wct": v} for l, v in zip(template_labels, values)]
    rows, source = select_result_rows(stem, template_rows)
    label_map = {
        "AegisKV": "AegisKV",
        "no lifecycle": "no\nlifecycle",
        "no segment granularity": "no\nsegment",
        "no restore/recompute": "no\nrestore",
        "no pressure": "no\npressure",
        "no hysteresis": "no\nhysteresis",
        "no critical restore": "no critical\nrestore",
    }
    labels = [label_map.get(str(row["variant"]), str(row["variant"])) for row in rows]
    values = [as_float(row, "normalized_wct") for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.bar(
        labels,
        values,
        color=[COLORS["aegis"]] + [COLORS["static"]] * (len(labels) - 1),
    )
    style_axes(ax, "normalized WCT")
    if source == "template":
        ax.set_ylim(0, 0.8)
    else:
        lower = max(0.0, min(values) - 0.025)
        if min(values) >= 0.9:
            lower = min(0.985, lower)
        upper = max(values) + 0.02
        ax.set_ylim(lower, max(1.05, upper))
        ax.axhline(1.0, color="#64748B", linewidth=1.0, linestyle="--")
    ax.set_title("Ablation: policy signals matter", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig13() -> None:
    stem = "fig_budget_sensitivity"
    budgets = np.linspace(0, 1, 11)
    timings = ["pre-prefill", "decode", "fan-in", "summary"]
    centers = [0.15, 0.35, 0.55, 0.75]
    data = np.vstack([0.95 - 0.35 * np.exp(-((budgets - c) ** 2) / 0.045) + 0.18 * budgets for c in centers])
    template_rows = []
    for yi, timing in enumerate(timings):
        for b, v in zip(budgets, data[yi]):
            template_rows.append({"failure_timing": timing, "gpu_replica_budget_fraction": round(float(b), 2), "normalized_wct": float(v)})
    rows, source = select_result_rows(stem, template_rows)
    timings = list(dict.fromkeys(str(row["failure_timing"]) for row in rows))
    budgets = np.array(sorted({as_float(row, "gpu_replica_budget_fraction") for row in rows}))
    matrix = np.full((len(timings), len(budgets)), np.nan)
    timing_idx = {name: i for i, name in enumerate(timings)}
    budget_idx = {float(value): i for i, value in enumerate(budgets)}
    for row in rows:
        matrix[timing_idx[str(row["failure_timing"])]][budget_idx[as_float(row, "gpu_replica_budget_fraction")]] = as_float(row, "normalized_wct")
    data = matrix
    fig, ax = plt.subplots(figsize=(6.2, 3.45))
    im = ax.imshow(data, aspect="auto", cmap="viridis_r", origin="lower")
    ax.set_xticks(np.arange(len(budgets)), [f"{b:.1f}" for b in budgets], fontsize=7)
    ax.set_yticks(np.arange(len(timings)), timings)
    ax.set_xlabel("GPUReplica budget fraction", fontsize=9)
    ax.set_title("Budget sensitivity heatmap", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("normalized WCT", fontsize=8)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig14() -> None:
    stem = "fig_dp_scale"
    dp = np.array([2, 4, 8])
    wf_counts = [1, 4, 8, 16]
    template_rows = []
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    for wf in wf_counts:
        benefit = 1.0 - (0.12 + 0.12 * np.log2(dp)) * (1 - 0.025 * wf)
        benefit += np.maximum(wf - 8, 0) * 0.015
        for d, v in zip(dp, benefit):
            template_rows.append({"dp_size": int(d), "workflow_count": wf, "normalized_wct": float(v)})
    rows, source = select_result_rows(stem, template_rows)
    by_wf: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        series = str(row.get("series") or "").strip()
        if not series:
            series = f"{as_int(row, 'workflow_count')} workflows"
        by_wf.setdefault(series, []).append(row)
    for series, group in sorted(by_wf.items()):
        group = sorted(group, key=lambda row: as_float(row, "dp_size"))
        yerr = ci_values(group, "normalized_wct")
        ax.errorbar(
            [as_float(row, "dp_size") for row in group],
            [as_float(row, "normalized_wct") for row in group],
            yerr=yerr if np.any(yerr > 0) else None,
            marker="o",
            label=series,
            capsize=3,
            **error_bar_style(),
        )
    dp = np.array(sorted({as_int(row, "dp_size") for row in rows}))
    ax.set_xticks(dp)
    ax.set_xlabel("DP size", fontsize=9)
    style_axes(ax, "normalized WCT")
    values = [as_float(row, "normalized_wct") for row in rows]
    if values and min(values) >= 0.8 and max(values) <= 1.2:
        ax.set_ylim(min(0.95, min(values) - 0.01), max(1.02, max(values) + 0.01))
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    ax.set_title("DP scale and peer availability", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig14b() -> None:
    stem = "fig_peer_rank_selection"
    template_policies = ["random", "least\nloaded", "most\nlocal", "retry\naware", "load+locality\nAegis"]
    wct = np.array([0.82, 0.74, 0.71, 0.66, 0.57])
    overhead = np.array([12, 7, 18, 11, 8])
    template_rows = [{"policy": p.replace("\n", " "), "normalized_wct": float(v), "peer_p99_overhead_pct": float(o)} for p, v, o in zip(template_policies, wct, overhead)]
    rows, source = select_result_rows(stem, template_rows)
    policies = [
        str(row["policy"])
        .replace("non-peer route miss", "non-peer\nmiss")
        .replace("random healthy route", "random\nhealthy")
        .replace("AegisKV peer route", "AegisKV\npeer route")
        .replace("least-loaded", "least\nloaded")
        .replace("most-local", "most\nlocal")
        .replace("retry-route-aware", "retry-route\naware")
        .replace("AegisKV load+locality", "AegisKV\nload+locality")
        for row in rows
    ]
    wct = np.array([as_float(row, "normalized_wct") for row in rows])
    overhead = np.array([as_float(row, "peer_p99_overhead_pct") for row in rows])
    x = np.arange(len(policies))
    fig, ax1 = plt.subplots(figsize=(6.8, 3.45))
    ax1.bar(
        x - 0.18,
        wct,
        0.36,
        yerr=ci_values(rows, "normalized_wct"),
        capsize=3,
        error_kw=error_bar_style(),
        color=COLORS["aegis"],
        label="WCT",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, fontsize=8)
    style_axes(ax1, "normalized WCT")
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.18,
        overhead,
        0.36,
        yerr=ci_values(rows, "peer_p99_overhead_pct"),
        capsize=3,
        error_kw=error_bar_style(),
        color=COLORS["gpu"],
        label="p99",
    )
    ax2.set_ylabel("p99 recovery / recompute (%)", fontsize=9)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=True, loc="upper left")
    legend.get_frame().set_facecolor("#FFFFFF")
    legend.get_frame().set_edgecolor("#E5E7EB")
    legend.get_frame().set_linewidth(0.6)
    ax1.set_title("Peer-route locality cuts replay recovery", fontsize=11)
    add_data_badge(ax1, source)
    save_plot(fig, stem)


def plot_fig15() -> None:
    stem = "fig_simulator_calibration"
    policies = ["none", "Host", "GPUReplica", "AegisKV"]
    sglang = np.array([100, 64, 58, 41])
    sglsim = np.array([96, 61, 55, 39])
    template_rows = [{"policy": p, "sglang_norm": float(x), "sglsim_norm": float(y)} for p, x, y in zip(policies, sglang, sglsim)]
    rows, source = select_result_rows(stem, template_rows)
    policies = [str(row["policy"]) for row in rows]
    sglang = np.array([as_float(row, "sglang_norm") for row in rows])
    sglsim = np.array([as_float(row, "sglsim_norm") for row in rows])
    metrics = {str(row.get("metric", "")).strip() for row in rows}
    metrics.discard("")
    metric = next(iter(metrics)) if len(metrics) == 1 else "mixed"
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    scatter_colors = [COLORS["none"], COLORS["host"], COLORS["gpu"], COLORS["aegis"]]
    ax.scatter(sglang, sglsim, s=70, color=[scatter_colors[i % len(scatter_colors)] for i in range(len(policies))])
    observed_min = min(float(np.nanmin(sglang)), float(np.nanmin(sglsim)))
    observed_max = max(float(np.nanmax(sglang)), float(np.nanmax(sglsim)))
    if metric == "normal_path_p99_overhead_pct":
        pad = max(0.35, (observed_max - observed_min) * 0.18)
        lo = min(-0.05, observed_min - pad)
        hi = observed_max + pad
        xlabel = "SGLang p99 overhead (%)"
        ylabel = "SGLSim p99 overhead (%)"
        title = "Normal-path calibration anchor"
    else:
        pad = max(0.12, (observed_max - observed_min) * 0.16)
        lo = observed_min - pad
        hi = observed_max + pad
        xlabel = "SGLang anchor value"
        ylabel = "SGLSim anchor value"
        title = "SGLang/SGLSim calibration anchors"
    ax.plot([lo, hi], [lo, hi], color="#94A3B8", ls="--", lw=1)
    for p, x, y in zip(policies, sglang, sglsim):
        if metric == "normal_path_p99_overhead_pct" and p.lower() == "none":
            ax.text(x - 7.2, y - 1.6, p, fontsize=8)
        else:
            if metric == "normal_path_p99_overhead_pct":
                dx = 0.06
                dy = -0.18
                label = p
            else:
                span = max(hi - lo, 1e-6)
                dx = span * 0.018
                dy = -span * 0.035
                label = p.replace(" ", "\n", 1)
            ax.text(x + dx, y + dy, label, fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=9)
    style_axes(ax, ylabel)
    ax.set_title(title, fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig15b() -> None:
    stem = "fig_physical_recovery_copy_calibration"
    template_tokens = np.array([512, 1024, 2048, 4096, 8192])
    template_rows = []
    for path, scale in (
        ("Host restore H2D", 0.0059),
        ("Host backup D2H", 0.0057),
        ("GPUReplica restore P2P", 0.00053),
    ):
        for tokens in template_tokens:
            template_rows.append(
                {
                    "path": path,
                    "phase": "restore" if "restore" in path else "backup",
                    "tokens": int(tokens),
                    "p50_ms": float(tokens * scale),
                    "p50_ms_ci95": float(tokens * scale * 0.02),
                }
            )
    rows, source = select_result_rows(stem, template_rows)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["path"]), []).append(row)
    for path_rows in grouped.values():
        path_rows.sort(key=lambda row: as_float(row, "tokens"))

    fig, ax = plt.subplots(figsize=(6.2, 3.45))
    style = {
        "Host restore H2D": (COLORS["host"], "o"),
        "Host backup D2H": (COLORS["checkpoint"], "s"),
        "GPUReplica restore P2P": (COLORS["gpu"], "^"),
    }
    for path, path_rows in grouped.items():
        tokens = np.array([as_float(row, "tokens") for row in path_rows])
        p50 = np.array([as_float(row, "p50_ms") for row in path_rows])
        ci = np.array([optional_float(row, "p50_ms_ci95") for row in path_rows])
        color, marker = style.get(path, (COLORS["ink"], "o"))
        ax.errorbar(
            tokens,
            p50,
            yerr=ci,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            capsize=2.5,
            color=color,
            label=path,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("KV tokens copied", fontsize=9)
    style_axes(ax, "p50 copy latency (ms)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Physical recovery-copy calibration", fontsize=11)
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig15c() -> None:
    stem = "fig_recovery_model_calibration_error"
    template_tokens = np.array([512, 1024, 2048, 4096, 8192])
    template_rows = []
    for path, scale in (
        ("Host restore H2D", 0.0059),
        ("GPUReplica restore P2P", 0.00053),
    ):
        for tokens in template_tokens:
            template_rows.append(
                {
                    "path": path,
                    "tokens": int(tokens),
                    "measured_copy_ms": float(tokens * scale),
                    "modeled_copy_ms": float(tokens * scale * 1.02),
                    "abs_error_ms": float(tokens * scale * 0.02),
                    "rel_error_pct": 2.0,
                }
            )
    rows, source = select_result_rows(stem, template_rows)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["path"]), []).append(row)
    for path_rows in grouped.values():
        path_rows.sort(key=lambda row: as_float(row, "tokens"))

    fig, ax = plt.subplots(figsize=(6.25, 3.5))
    styles = {
        "Host restore H2D": (COLORS["host"], "o"),
        "GPUReplica restore P2P": (COLORS["gpu"], "^"),
    }
    max_rel_error = 0.0
    for path, path_rows in grouped.items():
        tokens = np.array([as_float(row, "tokens") for row in path_rows])
        measured = np.array([as_float(row, "measured_copy_ms") for row in path_rows])
        modeled = np.array([as_float(row, "modeled_copy_ms") for row in path_rows])
        max_rel_error = max(
            max_rel_error,
            max([optional_float(row, "rel_error_pct") for row in path_rows] or [0.0]),
        )
        color, marker = styles.get(path, (COLORS["ink"], "o"))
        ax.plot(
            tokens,
            measured,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            color=color,
            label=f"{path} measured",
        )
        ax.plot(
            tokens,
            modeled,
            marker="x",
            linewidth=1.6,
            markersize=5,
            linestyle="--",
            color=color,
            alpha=0.82,
            label=f"{path} model",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("KV tokens restored", fontsize=9)
    style_axes(ax, "copy component latency (ms)")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    ax.set_title("Recovery model calibration error", fontsize=11)
    ax.text(
        0.98,
        0.08,
        f"max copy-component error: {max_rel_error:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="#CBD5E1", lw=0.6),
    )
    add_data_badge(ax, source)
    save_plot(fig, stem)


def plot_fig15d() -> None:
    stem = "fig_sglang_sglsim_accuracy"
    template_rows = [
        {
            "metric": "prefill wall",
            "unit": "s",
            "sglang_value": 26.0,
            "sglsim_value": 24.15,
            "ratio": 0.929,
            "rel_error_pct": 7.1,
        },
        {
            "metric": "decode wall",
            "unit": "s",
            "sglang_value": 50.0,
            "sglsim_value": 52.41,
            "ratio": 1.048,
            "rel_error_pct": 4.8,
        },
        {
            "metric": "first decode throughput",
            "unit": "tok/s",
            "sglang_value": 199.49,
            "sglsim_value": 199.51,
            "ratio": 1.000,
            "rel_error_pct": 0.0,
        },
        {
            "metric": "e2e mean",
            "unit": "s",
            "sglang_value": 75.56,
            "sglsim_value": 78.08,
            "ratio": 1.033,
            "rel_error_pct": 3.3,
        },
    ]
    rows, source = select_result_rows(stem, template_rows)
    order = [
        "prefill wall",
        "decode wall",
        "first decode throughput",
        "e2e mean",
        "e2e p50",
        "e2e p90",
        "e2e p99",
        "cache hit tokens",
    ]
    rows = sorted(
        rows,
        key=lambda row: order.index(str(row["metric"]))
        if str(row["metric"]) in order
        else len(order),
    )
    labels = [
        str(row["metric"])
        .replace("first decode throughput", "decode\nthroughput")
        .replace("cache hit tokens", "cache-hit\ntokens")
        .replace("prefill wall", "prefill\nwall")
        .replace("decode wall", "decode\nwall")
        .replace("e2e ", "e2e\n")
        for row in rows
    ]
    ratios = np.array([as_float(row, "ratio") for row in rows])
    errors = np.array([as_float(row, "rel_error_pct") for row in rows])
    x = np.arange(len(rows))
    colors = [
        COLORS["aegis"] if abs(value - 1.0) <= 0.05 else COLORS["host"]
        for value in ratios
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.axhspan(0.90, 1.10, color="#DCFCE7", alpha=0.55, label="±10% band")
    ax.axhline(1.0, color="#334155", linewidth=1.0, linestyle="--")
    ax.plot(x, ratios, color=COLORS["ink"], linewidth=1.4, alpha=0.72)
    ax.scatter(x, ratios, s=48, color=colors, zorder=3)
    for xi, ratio, err in zip(x, ratios, errors):
        ax.text(
            xi,
            ratio + (0.018 if ratio <= 1.0 else -0.028),
            f"{err:.1f}%",
            ha="center",
            va="bottom" if ratio <= 1.0 else "top",
            fontsize=6.8,
            color="#334155",
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.86, 1.12)
    style_axes(ax, "SGLSim / SGLang ratio")
    ax.set_title("Real SGLang inference validates SGLSim timing", fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    add_data_badge(ax, source, x=0.02, y=0.98, ha="left")
    save_plot(fig, stem)


def plot_fig15e() -> None:
    stem = "fig_p0_physical_bridge_coverage"
    bridge_summary = read_p0_physical_bridge_summary()
    measured_rows: list[dict[str, object]] = []
    for row in bridge_summary:
        restore_rows = optional_float(row, "restore_rows")
        bridge_rows = optional_float(row, "bridge_eligible_rows")
        live_copy_rows = 0.0
        live_copy_fraction = live_copy_rows / restore_rows if restore_rows else 0.0
        bridge_fraction = bridge_rows / restore_rows if restore_rows else 0.0
        measured_rows.append(
            {
                "suite_label": str(row.get("suite_label") or ""),
                "figures": str(row.get("figures") or ""),
                "restore_rows": int(restore_rows),
                "bridge_eligible_rows": int(bridge_rows),
                "live_physical_tensor_copy_rows": int(live_copy_rows),
                "bridge_coverage_fraction": bridge_fraction,
                "live_copy_fraction": live_copy_fraction,
                "max_restored_tokens": int(optional_float(row, "max_restored_tokens")),
                "max_measurement_tokens": int(optional_float(row, "max_measurement_tokens")),
                "evidence_tier": "sglsim-physical-bridge-coverage",
                "source": "p0_physical_bridge_summary",
                "run_id": "strict_final_readiness_p0_bridge_20260702",
                "note": (
                    "Bridge rows map metadata-materialized SGLSim restore rows to "
                    "physical CUDA recovery-path calibration; live copy rows remain zero."
                ),
            }
        )
    if measured_rows:
        write_measured_csv(stem, measured_rows)

    template_rows = [
        {
            "suite_label": "policy sweep",
            "figures": "Fig. 8/9/10",
            "restore_rows": 405,
            "bridge_eligible_rows": 405,
            "live_physical_tensor_copy_rows": 0,
            "bridge_coverage_fraction": 1.0,
            "live_copy_fraction": 0.0,
            "max_restored_tokens": 3916,
            "max_measurement_tokens": 4096,
        },
        {
            "suite_label": "recovery action",
            "figures": "Fig. 10/14",
            "restore_rows": 288,
            "bridge_eligible_rows": 288,
            "live_physical_tensor_copy_rows": 0,
            "bridge_coverage_fraction": 1.0,
            "live_copy_fraction": 0.0,
            "max_restored_tokens": 2284,
            "max_measurement_tokens": 4096,
        },
    ]
    rows, source = select_result_rows(stem, template_rows)
    rows = sorted(rows, key=lambda row: str(row["suite_label"]))
    labels = [
        str(row["figures"]).replace("Fig. ", "Fig.\n")
        for row in rows
    ]
    restore = np.array([as_float(row, "restore_rows") for row in rows])
    bridge = np.array([as_float(row, "bridge_eligible_rows") for row in rows])
    live = np.array([as_float(row, "live_physical_tensor_copy_rows") for row in rows])
    bridge_frac = np.array([as_float(row, "bridge_coverage_fraction") for row in rows])
    live_frac = np.array([as_float(row, "live_copy_fraction") for row in rows])
    x = np.arange(len(rows))
    width = 0.24

    fig, ax1 = plt.subplots(figsize=(6.7, 3.55))
    ax1.bar(x - width, restore, width=width, color="#CBD5E1", label="restore rows")
    ax1.bar(x, bridge, width=width, color=COLORS["aegis"], label="physical bridge rows")
    ax1.bar(x + width, live, width=width, color=COLORS["gpu"], label="live physical copy rows")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    style_axes(ax1, "P0 restore rows")
    ax1.set_ylim(0, max(restore) * 1.18 if len(restore) else 1)
    for idx, row in enumerate(rows):
        max_restored = as_float(row, "max_restored_tokens")
        max_bucket = as_float(row, "max_measurement_tokens")
        ax1.text(
            idx,
            restore[idx] + max(restore) * 0.035,
            f"max {max_restored:.0f}->{max_bucket:.0f} tok",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#334155",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        bridge_frac * 100.0,
        color=COLORS["aegis"],
        marker="o",
        linewidth=2.1,
        label="bridge coverage",
    )
    ax2.plot(
        x,
        live_frac * 100.0,
        color=COLORS["gpu"],
        marker="x",
        linewidth=1.8,
        linestyle="--",
        label="live copy coverage",
    )
    ax2.set_ylim(-4, 108)
    ax2.set_ylabel("coverage (%)", fontsize=9)
    ax2.tick_params(axis="y", labelsize=8, colors="#334155")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#CBD5E1")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=6.8,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
    )
    ax1.set_title("Physical bridge covers P0 rows, live-copy gate remains open", fontsize=11, pad=25)
    add_data_badge(ax1, source, x=0.02, y=0.08, ha="left", va="bottom")
    fig.subplots_adjust(top=0.76)
    save_plot(fig, stem)


def plot_fig15f() -> None:
    stem = "fig_sglang_host_restore_materialization"
    state_order = [
        "host_frontier_not_visible_to_retry",
        "partial_async_materialized",
        "full_prefetch_materialized",
    ]
    state_labels = {
        "host_frontier_not_visible_to_retry": "hidden\ncontrol",
        "partial_async_materialized": "partial\nbest-effort",
        "full_prefetch_materialized": "full\nwait-complete",
    }
    state_roles = {
        "host_frontier_not_visible_to_retry": "control",
        "partial_async_materialized": "materialized",
        "full_prefetch_materialized": "materialized",
    }
    physical_rows = read_sglang_physical_restore_rows()
    measured_rows: list[dict[str, object]] = []
    for state in state_order:
        rows = [
            row
            for row in physical_rows
            if str(row.get("physical_restore_state") or "") == state
        ]
        measured_rows.append(
            {
                "state": state,
                "role": state_roles[state],
                "rows": len(rows),
                "retry_cached_tokens_after_accept": int(
                    sum(optional_float(row, "retry_cached_tokens_after_accept_total") for row in rows)
                ),
                "prefetch_io_completed_tokens": int(
                    sum(optional_float(row, "prefetch_io_completed_tokens_sum") for row in rows)
                ),
                "max_data_pages": int(
                    max((optional_float(row, "max_data_pages") for row in rows), default=0.0)
                ),
                "materialized": state != "host_frontier_not_visible_to_retry",
                "evidence_tier": "sglang-small-physical-restore-materialization",
                "source": "sglang_physical_restore_parser_summary",
                "run_id": "physical_restore_evidence_20260702",
                "note": (
                    "Parsed from real SGLang server.log and failure summaries; "
                    "Host restore materialization evidence, not P0 live physical_tensor_copy."
                ),
            }
        )
    if any(int(row["rows"]) > 0 for row in measured_rows):
        write_measured_csv(stem, measured_rows)

    template_rows = [
        {
            "state": "host_frontier_not_visible_to_retry",
            "role": "control",
            "rows": 4,
            "retry_cached_tokens_after_accept": 0,
            "prefetch_io_completed_tokens": 0,
            "max_data_pages": 0,
            "materialized": False,
        },
        {
            "state": "partial_async_materialized",
            "role": "materialized",
            "rows": 3,
            "retry_cached_tokens_after_accept": 18643,
            "prefetch_io_completed_tokens": 18601,
            "max_data_pages": 5137,
            "materialized": True,
        },
        {
            "state": "full_prefetch_materialized",
            "role": "materialized",
            "rows": 4,
            "retry_cached_tokens_after_accept": 20548,
            "prefetch_io_completed_tokens": 20548,
            "max_data_pages": 5137,
            "materialized": True,
        },
    ]
    rows, source = select_result_rows(stem, template_rows)
    rows = sorted(
        rows,
        key=lambda row: state_order.index(str(row["state"]))
        if str(row["state"]) in state_order
        else len(state_order),
    )
    labels = [state_labels.get(str(row["state"]), str(row["state"])) for row in rows]
    row_counts = np.array([as_float(row, "rows") for row in rows])
    cached_tokens = np.array(
        [as_float(row, "retry_cached_tokens_after_accept") for row in rows]
    )
    prefetch_tokens = np.array(
        [as_float(row, "prefetch_io_completed_tokens") for row in rows]
    )
    x = np.arange(len(rows))

    fig, ax1 = plt.subplots(figsize=(6.7, 3.55))
    colors = [
        COLORS["ordinary"]
        if str(row["role"]) == "control"
        else COLORS["host"]
        if str(row["state"]) == "partial_async_materialized"
        else COLORS["aegis"]
        for row in rows
    ]
    bars = ax1.bar(x, row_counts, width=0.46, color=colors, label="parser rows")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    style_axes(ax1, "real SGLang rows")
    ax1.set_ylim(0, max(row_counts) * 1.35 if len(row_counts) else 1)
    for rect, count in zip(bars, row_counts):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.12,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#334155",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        cached_tokens / 1000.0,
        color=COLORS["ink"],
        marker="o",
        linewidth=2.0,
        label="retry cached tokens",
    )
    ax2.plot(
        x,
        prefetch_tokens / 1000.0,
        color=COLORS["checkpoint"],
        marker="s",
        linewidth=1.7,
        linestyle="--",
        label="prefetch completed tokens",
    )
    ax2.set_ylabel("tokens (K)", fontsize=9)
    ax2.tick_params(axis="y", labelsize=8, colors="#334155")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#CBD5E1")
    ax2.set_ylim(0, max(cached_tokens.max(), prefetch_tokens.max()) / 1000.0 * 1.25 if len(rows) else 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=6.8,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.17),
    )
    ax1.set_title("Real SGLang Host restore materializes retry KV", fontsize=11, pad=25)
    add_data_badge(ax1, source, x=0.02, y=0.08, ha="left", va="bottom")
    fig.subplots_adjust(top=0.76)
    save_plot(fig, stem)


def generate_tables() -> None:
    tables = {
        "table1_state_vector.md": """# Table 1: State Vector

| Symbol | Meaning |
|---|---|
| `h_t` | Host backup hit / ACK ratio |
| `g_t` | GPUReplica ACK ratio |
| `q_t` | retry / scheduler queue pressure |
| `p2p_t` | P2P bandwidth pressure |
| `h2d_t` | H2D restore pressure |
| `d2h_wt_t` | fixed write-through D2H pressure |
| `host_vis_t` | fraction of ACKed Host KV visible to retry ranks |
| `w_t` | workflow criticality skew |
| `l_t` | lifecycle distribution |
| `loc_{p,t}` | peer KV locality for candidate rank `p` |
| `load_{p,t}` | peer queue/load pressure |
| `dup_t` | duplicate Host/GPUReplica protected bytes |
| `alpha_gpu_t` | peer GPU backup budget ratio |
| `r_t` | recent recovery success / saved WCT |
""",
        "table2_baselines.md": """# Table 2: Baselines

| Policy | Workflow | Lifecycle | Segment | Restore/recompute | Resource pressure |
|---|---|---|---|---|---|
| no backup | no | no | no | recompute | no |
| baseline write-through invisible | no | no | no | recompute | fixed D2H only |
| Host visible only | no | no | segment | restore/recompute | Host restore queue |
| uniform GPUReplica | no | no | segment | restore | peer budget only |
| request-level | weak | no | no | restore | coarse |
| agent-level | yes | weak | no | restore | coarse |
| static critical path | yes | weak | segment | restore | no |
| AegisKV | yes | yes | yes | yes | yes |
""",
        "table3_implementation_hooks.md": """# Table 3: Implementation Hooks

| Hook | SGLang path | Purpose |
|---|---|---|
| request metadata | tokenizer manager / `Req` | attach workflow state |
| radix match/insert/split | `HiRadixCache` / `RadixCache` | update segment metadata and shared SafeCover |
| fixed Host D2H | `HiRadixCache.write_backup` / `HiCacheController.write` | observe fixed write-through copy |
| Host ACK | `HiRadixCache.writing_check` / `ack_write_queue` | publish HostBackup directory entry |
| H2D/cross-rank restore | `HiCacheController.load` / restore protocol | recover failed-rank Host frontier on retry rank |
| GPU eviction | `HiRadixCache.evict` | preserve/invalidate GPU hit state |
| Host eviction | `evict_host` | invalidate Host frontier |
| peer ACK | peer copy manager / DP controller | mark GPUReplica frontier |
| peer demotion | peer replica manager | invalidate or downgrade GPUReplica frontier |
| failure retry | scheduler / DP controller | choose frontier and retry path |
""",
        "table4_correctness_tests.md": """# Table 4: Correctness Tests

| Test | Expected invariant | Current artifact |
|---|---|---|
| missing/pending HostBackup + failure | pending frontier rejected; retry recomputes | `test_aegis_restore_lookup_recomputes_when_host_backup_missing` |
| active `lock_ref` + eviction pressure | locked restored prefix is not evicted | `test_aegis_restored_locked_prefix_is_not_evicted_under_pressure` |
| divergent radix insert after restore | restored frontier token span still matches | `test_aegis_restored_prefix_survives_divergent_radix_insert` |
| unsupported pending restore action | no device KV is materialized before ACK | `test_aegis_pending_restore_action_does_not_materialize_device_cache` |
| Host unavailable + peer unavailable | unsafe frontier is not selected | `test_aegis_restore_lookup_recomputes_when_host_evicted_and_peer_unavailable` |
| peer demotion | GPUReplica frontier used only on peer rank | `test_aegis_restore_lookup_demotes_peer_gpu_when_retry_rank_is_not_peer` |
| request without Aegis metadata | no Aegis frontier or candidate retry is published | `test_aegis_baseline_request_without_metadata_does_not_publish_frontier` |
""",
    }
    for name, content in tables.items():
        (TABLE_DIR / name).write_text(content)


def make_manifest() -> list[dict[str, object]]:
    entries = []
    figure_specs = [
        ("Fig. 1", "fig_motivation_equal_size_loss", "result_template", "SGLang/SGLSim failure injection", "Equal-size KV loss has non-uniform WCT impact."),
        ("Fig. 1b", "fig_critical_backup_priority", "result_template", "SGLSim staged 1-3-1-3-1 backup sweep", "Critical-agent KV should be protected before non-critical KV."),
        ("Fig. 1c", "fig_critical_agent_profile", "result_template", "SGLSim staged 1-3-1-3-1 stage-barrier trace", "Long-decode fan-out agent is the Critical stage bottleneck."),
        ("Fig. 2", "fig_restore_recompute_boundary", "result_template", "SGLSim runtime replay calibrated", "Restore/recompute lower envelope."),
        ("Fig. 3", "fig_static_gpu_replica_budget_gap", "result_template", "SGLSim runtime replay calibrated", "Static GPUReplica budget is regime-dependent."),
        ("Fig. 3b", "fig_alpha_workload_failure_policy", "result_template", "offline MDP/policy sweep over staged workflow states", "Backup ratio alpha responds to workload pressure and failure risk."),
        ("Fig. 3c", "fig_online_alpha_adaptation", "result_template", "SGLSim staged online adaptive-alpha sweep", "Online controller adapts alpha to workload pressure and failure risk."),
        ("Fig. 3d", "fig_staged_alpha_risk_tradeoff", "result_template", "SGLSim staged 1-3-1-3-1 alpha-risk counterfactual", "Risk-adjusted alpha optimum shifts with failure rate on the staged workflow."),
        ("Fig. 4", "fig_system_overview", "mechanism_diagram", "generated bitmap + prompt", "AegisKV integrates with SGLang KV lifecycle."),
        ("Fig. 8", "fig_overall_wct", "result_template", "SGLSim runtime replay mixed", "Total WCT and failure-exposed recovery cost."),
        ("Fig. 9", "fig_recovery_latency", "result_template", "SGLang small + SGLSim large", "Retry and critical-agent latency."),
        ("Fig. 10", "fig_recovery_path_breakdown", "result_template", "SGLSim policy matrix", "Recovery path mix."),
        ("Fig. 10b", "fig_shared_prefix_marginal", "result_template", "SGLSim shared-prefix sweep", "Marginal SafeCover reduces duplicate backup."),
        ("Fig. 11", "fig_normal_path_overhead", "result_template", "SGLang no-failure overhead", "Normal-path overhead is bounded."),
        ("Fig. 11b", "fig_hostbackup_visibility", "result_template", "SGLang Host visible/invisible pair", "Visibility adds recovery without changing fixed D2H."),
        ("Fig. 12", "fig_ablation", "result_template", "SGLSim runtime replay calibrated", "Policy signals matter."),
        ("Fig. 13", "fig_budget_sensitivity", "result_template", "SGLSim replay regime budget sweep", "Budget optimum shifts."),
        ("Fig. 14", "fig_dp_scale", "result_template", "SGLSim direct DP sweep", "Peer availability improves then saturates."),
        ("Fig. 14b", "fig_peer_rank_selection", "result_template", "SGLSim peer-route replay", "Retry locality makes peer KV usable."),
        ("Fig. 15", "fig_simulator_calibration", "result_template", "paired SGLang/SGLSim runs", "SGLang/SGLSim calibration anchors."),
        ("Fig. 15b", "fig_physical_recovery_copy_calibration", "result_template", "torch/CUDA physical recovery-copy microbench", "Physical copy timing calibrates recovery model."),
        ("Fig. 15c", "fig_recovery_model_calibration_error", "result_template", "SGLSim recovery_model + torch/CUDA copy anchors", "Recovery model copy component matches physical copy anchors."),
        ("Fig. 15d", "fig_sglang_sglsim_accuracy", "result_template", "paired SGLang/SGLSim synthesis run", "Real SGLang inference validates base SGLSim timing."),
        ("Fig. 15e", "fig_p0_physical_bridge_coverage", "result_template", "P0 physical bridge coverage audit", "Physical calibration bridge covers P0 restore rows while live-copy gate remains open."),
        ("Fig. 15f", "fig_sglang_host_restore_materialization", "result_template", "real SGLang Host restore materialization parser", "Real SGLang Host restore rows materialize retry KV while controls stay hidden."),
    ]
    for fig, stem, kind, source, claim in figure_specs:
        entry = {
            "figure": fig,
            "stem": stem,
            "kind": kind,
            "source_needed_for_final": source,
            "claim": claim,
            "evidence_status": (
                "design-ready"
                if kind == "mechanism_diagram"
                else result_evidence_status(stem)
            ),
            "outputs": {},
        }
        if kind == "mechanism_diagram":
            entry["outputs"]["png"] = str(
                (PNG_DIR / "fig_system_overview_generated_v2_tight.png").relative_to(ROOT)
            )
            entry["outputs"]["generated_prompt"] = str(
                (SRC_DIR / "fig_system_overview_generated_v2.prompt.md").relative_to(ROOT)
            )
        else:
            entry["outputs"]["png"] = str((PNG_DIR / f"{stem}.png").relative_to(ROOT))
            entry["template_data"] = str((DATA_DIR / f"{stem}.csv").relative_to(ROOT))
            measured = read_measured_rows(stem)
            if measured:
                entry["measured_data"] = str((MEASURED_DIR / f"{stem}.csv").relative_to(ROOT))
                tiers = sorted(
                    {
                        str(row.get("evidence_tier") or "final").strip()
                        for row in measured
                        if str(row.get("evidence_tier") or "").strip()
                    }
                )
                entry["evidence_tier"] = ",".join(tiers) if tiers else "final"
        entries.append(entry)
    (ROOT / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    return entries


def write_readme(entries: list[dict[str, object]]) -> None:
    lines = [
        "# 图表资产",
        "",
        "本目录保存 AegisKV 论文草稿的图表资产。",
        "",
        "重要说明：结果图可能使用 preliminary SGLSim CSV 作为版式和内部审阅数据。重复 SGLSim replay 产生的 CSV 可以包含 run-level std/95% CI 列，部分柱状图会渲染这些区间。只有当对应 `evidence_tier` 变成 `final` 且 strict validator 通过后，它们才是最终 HPCA 定量证据。当前包只保留生成式 System overview 和动机/实验结果图，不再维护详细机制流程图。",
        "",
        "重新生成所有图表：",
        "",
        "```bash",
        "python benchmark/AegisKV/draft/figures/scripts/generate_all_figures.py",
        "```",
        "",
        "## 已生成图表",
        "",
        "| 图号 | 文件名 | 状态 | 最终证据来源 |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['figure']} | `{entry['stem']}` | {entry['evidence_status']} | {entry['source_needed_for_final']} |"
        )
    lines.extend(
        [
            "",
            "## 目录",
            "",
            "- `src/`: 生成式 system overview 的 prompt 记录。",
            "- `png/`: Markdown 审阅使用的渲染图。",
            "- `data/template/`: 结果图的 deterministic template CSV。",
            "- `data/measured/`: 可替换 template 的 measured 或 preliminary CSV。",
            "- `tables/`: 第 08 节中 Tables 1-4 的 Markdown 版本。",
            "- `gallery.md`: 所有图的单页预览。",
            "- `manifest.json`: 图表、数据来源和输出文件的机器可读索引。",
        ]
    )
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")


def write_gallery(entries: list[dict[str, object]]) -> None:
    lines = [
        "# AegisKV 图表预览",
        "",
        "标记为 `sglsim-preliminary` 的结果图使用 preliminary SGLSim CSV，只用于内部审阅。部分行包含重复 replay 的 run-level std/95% CI。写入最终定量结论前，必须用 `evidence_tier=final` 的 SGLang/SGLSim 测量替换这些数据。",
        "",
    ]
    for entry in entries:
        stem = entry["stem"]
        status = entry["evidence_status"]
        lines.extend(
            [
                f"## {entry['figure']}: `{stem}`",
                "",
                f"状态：`{status}`",
                "",
                f"![{stem}]({entry.get('outputs', {}).get('png', f'png/{stem}.png')})",
                "",
            ]
        )
    (ROOT / "gallery.md").write_text("\n".join(lines))


def main() -> int:
    ensure_dirs()
    plot_fig1()
    plot_fig1b()
    plot_fig1c()
    plot_fig2()
    plot_fig3()
    plot_fig3b()
    plot_fig3c()
    plot_fig3d()
    plot_fig8()
    plot_fig9()
    plot_fig10()
    plot_fig10b()
    plot_fig11()
    plot_fig11b()
    plot_fig12()
    plot_fig13()
    plot_fig14()
    plot_fig14b()
    plot_fig15()
    plot_fig15b()
    plot_fig15c()
    plot_fig15d()
    plot_fig15e()
    plot_fig15f()
    generate_tables()
    entries = make_manifest()
    write_readme(entries)
    write_gallery(entries)
    print(f"generated {len(entries)} figures under {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
