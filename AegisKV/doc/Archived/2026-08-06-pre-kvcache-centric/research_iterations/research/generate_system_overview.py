from __future__ import annotations

import html
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


OUT = Path(__file__).resolve().parents[1] / "system_overview.drawio"
W, H = 2600, 1700
FONT = "Noto Sans CJK SC"

INK = "#172B4D"
MUTED = "#5E6C84"
GRAY_FILL, GRAY_STROKE = "#F1F3F5", "#A5ADBA"
BLUE_FILL, BLUE_BODY, BLUE_STROKE = "#E8F1FB", "#FAFCFF", "#5B8DB8"
GREEN_FILL, GREEN_BODY, GREEN_STROKE = "#ECF8F1", "#FCFEFD", "#4B9B70"
YELLOW_FILL, YELLOW_BODY, YELLOW_STROKE = "#FFF6D8", "#FFFEF9", "#C9A227"
RED_FILL, RED_STROKE = "#FBE9E9", "#C94040"
PURPLE_FILL, PURPLE_STROKE = "#F2ECFA", "#8B6BB1"
ORANGE_FILL, ORANGE_STROKE = "#FFF4DF", "#D69E2E"


def esc(value: Any) -> str:
    return html.escape(str(value))


def card(
    title: str,
    lines: list[str],
    *,
    eyebrow: str = "",
    title_size: int = 16,
    body_size: int = 13,
    body_margin: int = 7,
) -> str:
    blocks: list[str] = []
    if eyebrow:
        blocks.append(
            f'<div style="font-size:11px;font-weight:700;color:{MUTED};">'
            f"{esc(eyebrow.upper())}</div>"
        )
    blocks.append(
        f'<div style="font-size:{title_size}px;font-weight:700;margin-top:3px;">'
        f"{esc(title)}</div>"
    )
    if lines:
        blocks.append(
            f'<div style="font-size:{body_size}px;margin-top:{body_margin}px;">'
            + "<br/>".join(esc(line) for line in lines)
            + "</div>"
        )
    return "".join(blocks)


def build() -> ET.ElementTree:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-08-05T10:05:00.000Z",
            "agent": "Codex",
            "version": "26.0.16",
            "type": "device",
        },
    )
    diagram = ET.SubElement(
        mxfile,
        "diagram",
        {"id": "aegiskv-three-layer", "name": "AegisKV Three-Layer Architecture"},
    )
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(W),
            "dy": str(H),
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(W),
            "pageHeight": str(H),
            "background": "#FFFFFF",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    absolute_origins: dict[str, tuple[float, float]] = {
        "0": (0.0, 0.0),
        "1": (0.0, 0.0),
    }

    def vertex(
        cid: str,
        value: str,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        parent: str = "1",
        fill: str = "#FFFFFF",
        stroke: str = GRAY_STROKE,
        font_color: str = INK,
        font_size: int = 13,
        bold: bool = False,
        rounded: bool = True,
        dashed: bool = False,
        align: str = "left",
        spacing: int = 12,
        stroke_width: float = 1.4,
        extra_style: str = "",
    ) -> None:
        style = (
            f"rounded={1 if rounded else 0};arcSize=12;whiteSpace=wrap;html=1;"
            f"overflow=hidden;align={align};verticalAlign=middle;spacing={spacing};"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"dashed={1 if dashed else 0};dashPattern=6 4;fontColor={font_color};"
            f"fontSize={font_size};fontStyle={1 if bold else 0};fontFamily={FONT};"
            f"{extra_style}"
        )
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cid,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )
        parent_x, parent_y = absolute_origins[parent]
        absolute_origins[cid] = (parent_x + x, parent_y + y)

    def group(
        cid: str,
        title: str,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        parent: str,
        header_fill: str,
        body_fill: str,
        stroke: str,
        start_size: int = 40,
        align: str = "left",
    ) -> None:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cid,
                "value": esc(title),
                "style": (
                    f"swimlane;startSize={start_size};horizontal=1;rounded=1;"
                    "arcSize=10;container=1;collapsible=0;recursiveResize=0;"
                    "whiteSpace=wrap;html=1;verticalAlign=middle;"
                    f"align={align};spacingLeft=12;fillColor={header_fill};"
                    f"swimlaneFillColor={body_fill};strokeColor={stroke};"
                    f"strokeWidth=1.6;fontSize=14;fontStyle=1;fontColor={INK};"
                    f"fontFamily={FONT};"
                ),
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )
        parent_x, parent_y = absolute_origins[parent]
        absolute_origins[cid] = (parent_x + x, parent_y + y)

    def edge(
        cid: str,
        source: str,
        target: str,
        *,
        parent: str = "1",
        color: str = BLUE_STROKE,
        dashed: bool = False,
        exit_x: float = 1.0,
        exit_y: float = 0.5,
        entry_x: float = 0.0,
        entry_y: float = 0.5,
        points: list[tuple[int, int]] | None = None,
        bidirectional: bool = False,
        width: float = 2.2,
    ) -> None:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cid,
                "value": "",
                "style": (
                    "edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=auto;html=1;"
                    f"strokeColor={color};strokeWidth={width};"
                    f"dashed={1 if dashed else 0};dashPattern=7 5;"
                    f"startArrow={'block' if bidirectional else 'none'};"
                    f"startFill={1 if bidirectional else 0};endArrow=block;endFill=1;"
                    f"exitX={exit_x};exitY={exit_y};exitDx=0;exitDy=0;exitPerimeter=1;"
                    f"entryX={entry_x};entryY={entry_y};entryDx=0;entryDy=0;entryPerimeter=1;"
                ),
                "edge": "1",
                "parent": parent,
                "source": source,
                "target": target,
            },
        )
        geometry = ET.SubElement(
            cell, "mxGeometry", {"relative": "1", "as": "geometry"}
        )
        if points:
            array = ET.SubElement(geometry, "Array", {"as": "points"})
            parent_x, parent_y = absolute_origins[parent]
            for px, py in points:
                ET.SubElement(
                    array,
                    "mxPoint",
                    {
                        "x": str(px - parent_x),
                        "y": str(py - parent_y),
                    },
                )

    vertex(
        "title",
        "AegisKV · Three-Layer System Architecture",
        350,
        0,
        1900,
        48,
        fill="none",
        stroke="none",
        font_size=28,
        bold=True,
        align="center",
        rounded=False,
        spacing=0,
        stroke_width=0,
    )
    vertex(
        "subtitle",
        "Slack-aware requests → causal route binding → certified finite-window MILP → DP×PP execution",
        450,
        52,
        1700,
        20,
        fill="none",
        stroke="none",
        font_color=MUTED,
        font_size=14,
        align="center",
        rounded=False,
        spacing=0,
        stroke_width=0,
    )
    group(
        "system-host",
        "AEGISKV SYSTEM · blue = request · orange = stage-aligned KV · dashed purple/green = certified control",
        40,
        85,
        2520,
        1560,
        parent="1",
        header_fill=GRAY_FILL,
        body_fill="#FFFFFF",
        stroke=GRAY_STROKE,
        start_size=42,
    )

    # ------------------------------------------------------------------
    # Layer 1: application workflow construction and Algorithm 1.
    # ------------------------------------------------------------------
    group(
        "application-layer",
        "LAYER 1 · APPLICATION · build the workflow, track progress, and attach per-request Slack",
        30,
        52,
        1830,
        310,
        parent="system-host",
        header_fill=YELLOW_FILL,
        body_fill="#FFFDF6",
        stroke=YELLOW_STROKE,
    )
    vertex(
        "workflow-input",
        card(
            "Agent / Workflow Handler",
            ["task-decomposition or role protocol", "construct request sequence and dependencies"],
            eyebrow="Application entry",
        ),
        30,
        75,
        390,
        145,
        parent="application-layer",
        fill="#FFFFFF",
        stroke=BLUE_STROKE,
        font_color="#274C77",
    )
    vertex(
        "workflow-builder",
        card(
            "Fixed Workflow DAG",
            ["node = one agent execution segment", "status: pending / running / completed"],
            eyebrow="Workflow builder",
        ),
        460,
        75,
        460,
        145,
        parent="application-layer",
        fill=BLUE_FILL,
        stroke=BLUE_STROKE,
        font_color="#274C77",
    )
    vertex(
        "algorithm-1-handler",
        card(
            "Algorithm 1 · Slack Handler",
            [
                "online: duration model + CPM forward / backward",
                "trace study: oracle Slack can bypass prediction",
                "publish Slack through one downstream contract",
            ],
            eyebrow="Application-side implementation",
        ),
        960,
        75,
        820,
        145,
        parent="application-layer",
        fill=YELLOW_FILL,
        stroke=YELLOW_STROKE,
        font_color="#5B4710",
    )
    vertex(
        "request-envelope",
        "REQUEST ENVELOPE ↓  workflow_id · dag_node_id · payload · slack_ms · slack_source · slack_version",
        960,
        238,
        820,
        42,
        parent="application-layer",
        fill="#FFFFFF",
        stroke=YELLOW_STROKE,
        font_color="#5B4710",
        font_size=13,
        bold=True,
        align="center",
        spacing=5,
    )
    edge("edge-app-builder", "workflow-input", "workflow-builder", parent="application-layer", color=BLUE_STROKE)
    edge("edge-builder-alg1", "workflow-builder", "algorithm-1-handler", parent="application-layer", color=YELLOW_STROKE)
    edge(
        "edge-alg1-envelope",
        "algorithm-1-handler",
        "request-envelope",
        parent="application-layer",
        color=YELLOW_STROKE,
        exit_x=0.5,
        exit_y=1.0,
        entry_x=0.5,
        entry_y=0.0,
    )

    # ------------------------------------------------------------------
    # Layer 2: causal execution routing plus a certified backup-route binding.
    # ------------------------------------------------------------------
    group(
        "router-layer",
        "LAYER 2 · ROUTER · observed state freezes execution e(r); certified exact plan binds one healthy backup b(r)",
        30,
        392,
        1830,
        245,
        parent="system-host",
        header_fill=PURPLE_FILL,
        body_fill="#FCFAFF",
        stroke=PURPLE_STROKE,
    )
    vertex(
        "execution-binding",
        card(
            "Execution target e(r)",
            ["causal normal-inference route", "example below: DP Rank 0"],
            eyebrow="Observed-state Top-1",
        ),
        90,
        90,
        300,
        105,
        parent="router-layer",
        fill=BLUE_FILL,
        stroke=BLUE_STROKE,
        font_color="#274C77",
        font_size=12,
    )
    vertex(
        "top2-scheduler",
        card(
            "Causal Router + Certified Plan",
            [
                "health / load / approximate Radix state observed now",
                "freeze e(r), candidates B(r), route_version",
                "load certified y[r,b]=1; future manifest is forbidden",
            ],
            eyebrow="Admission and route contract",
        ),
        620,
        70,
        590,
        145,
        parent="router-layer",
        fill=PURPLE_FILL,
        stroke=PURPLE_STROKE,
        font_color="#4A375C",
    )
    vertex(
        "backup-binding",
        card(
            "Backup target b(r)",
            ["one-hot MILP decision y[r,b]", "example below: DP Rank 3"],
            eyebrow="Certified immutable route",
        ),
        1425,
        90,
        300,
        105,
        parent="router-layer",
        fill=ORANGE_FILL,
        stroke=ORANGE_STROKE,
        font_color="#6B4A10",
        font_size=12,
    )
    edge(
        "edge-envelope-router",
        "request-envelope",
        "top2-scheduler",
        parent="system-host",
        color=PURPLE_STROKE,
        dashed=True,
        exit_x=0.5,
        exit_y=1.0,
        entry_x=0.5,
        entry_y=0.0,
        points=[(1440, 462), (1916, 462), (1916, 532), (985, 532)],
    )
    edge("edge-router-exec", "top2-scheduler", "execution-binding", parent="router-layer", color=BLUE_STROKE, exit_x=0.0, entry_x=1.0)
    edge("edge-router-backup", "top2-scheduler", "backup-binding", parent="router-layer", color=ORANGE_STROKE)

    # ------------------------------------------------------------------
    # Layer 3: four DP replicas, each with four PP ranks and local modules.
    # ------------------------------------------------------------------
    group(
        "inference-layer",
        "LAYER 3 · INFERENCE ENGINE · DP=4 × PP=4 processes; each DP owns one Connector policy and one atomic alpha/beta budget",
        30,
        665,
        1830,
        820,
        parent="system-host",
        header_fill=BLUE_FILL,
        body_fill="#FAFCFF",
        stroke=BLUE_STROKE,
    )
    vertex(
        "flow-legend",
        "CONTROL / DATA SPLIT · exact plan carries route + per-stage placements · runtime moves only PPp → PPp KV · green budget version is DP-wide",
        30,
        52,
        1770,
        35,
        parent="inference-layer",
        fill="#FFFFFF",
        stroke=GRAY_STROKE,
        font_color=MUTED,
        font_size=12,
        align="center",
        spacing=4,
    )

    dp_x = [30, 475, 920, 1365]
    dp_colors = [BLUE_STROKE, GRAY_STROKE, GRAY_STROKE, ORANGE_STROKE]
    dp_headers = [BLUE_FILL, GRAY_FILL, GRAY_FILL, ORANGE_FILL]
    dp_bodies = ["#FBFDFF", "#FCFDFE", "#FCFDFE", "#FFFDF8"]
    role_labels = [
        "EXAMPLE EXECUTION TARGET · e(r)",
        "AVAILABLE DP REPLICA",
        "AVAILABLE DP REPLICA",
        "EXAMPLE BACKUP TARGET · b(r)",
    ]
    for rank, x in enumerate(dp_x):
        gid = f"dp-{rank}"
        dp_abs_x = 70 + x
        group(
            gid,
            f"DP RANK {rank} · one replica / failure domain",
            x,
            140,
            420,
            620,
            parent="inference-layer",
            header_fill=dp_headers[rank],
            body_fill=dp_bodies[rank],
            stroke=dp_colors[rank],
            start_size=40,
        )
        vertex(
            f"dp-{rank}-role",
            role_labels[rank],
            20,
            50,
            380,
            28,
            parent=gid,
            fill=dp_headers[rank],
            stroke=dp_colors[rank],
            font_color=MUTED,
            font_size=11,
            bold=True,
            align="center",
            spacing=3,
        )
        for pp in range(4):
            vertex(
                f"dp-{rank}-pp-{pp}",
                f"PP{pp}\nS/W",
                20 + pp * 95,
                95,
                75,
                48,
                parent=gid,
                fill=RED_FILL,
                stroke=RED_STROKE,
                font_color="#7F1D1D",
                font_size=11,
                bold=True,
                align="center",
                spacing=3,
            )
            if pp:
                edge(
                    f"edge-dp-{rank}-pp-{pp-1}-{pp}",
                    f"dp-{rank}-pp-{pp-1}",
                    f"dp-{rank}-pp-{pp}",
                    parent=gid,
                    color=BLUE_STROKE,
                    width=1.6,
                )
        vertex(
            f"dp-{rank}-adapter",
            card(
                "Budget Publisher · Algorithm 3",
                [
                    f"H_{rank}, M_{rank} → atomic alpha_{rank} / beta_{rank} for PP0–PP3",
                    "K → K+1 only after 4/4 budget ACKs",
                ],
                eyebrow="Control plane · DP-owned budget",
            ),
            20,
            460,
            380,
            100,
            parent=gid,
            fill=GREEN_FILL,
            stroke=GREEN_STROKE,
            font_color="#276749",
            font_size=12,
            spacing=10,
        )
        vertex(
            f"dp-{rank}-connector",
            card(
                "Certified Connector · Algorithm 2",
                [
                    "b(r) + x[r,c,p,b,t] → HBM / Host / Recompute",
                    "PPp → PPp only; commit after 4/4 stage ACKs",
                ],
                eyebrow="Control plane · stage-local executor",
            ),
            20,
            165,
            380,
            100,
            parent=gid,
            fill=PURPLE_FILL,
            stroke=PURPLE_STROKE,
            font_color="#4A375C",
            font_size=12,
            spacing=10,
        )
        vertex(
            f"dp-{rank}-hbm",
            f"DATA PLANE · GPU HBM\nper-stage H_{rank} = alpha_{rank} × HBMmax_{rank}",
            20,
            290,
            180,
            68,
            parent=gid,
            fill=RED_FILL,
            stroke=RED_STROKE,
            font_color="#7F1D1D",
            font_size=12,
            bold=True,
            align="center",
            spacing=6,
        )
        vertex(
            f"dp-{rank}-dram",
            f"DATA PLANE · Host DRAM\nDP policy M_{rank} = beta_{rank} × Hostmax_{rank}",
            220,
            290,
            180,
            68,
            parent=gid,
            fill=GREEN_FILL,
            stroke=GREEN_STROKE,
            font_color="#276749",
            font_size=12,
            bold=True,
            align="center",
            spacing=6,
        )
        vertex(
            f"dp-{rank}-feedback",
            "ROLLING STATE K · resident stock · used link bytes · admitted / unmet utility · recovery p99 · SLO miss",
            20,
            380,
            380,
            58,
            parent=gid,
            fill="#FFFFFF",
            stroke=GREEN_STROKE,
            font_color="#276749",
            font_size=11,
            align="center",
            spacing=5,
        )
        vertex(
            f"dp-{rank}-scope",
            f"DP-OWNED alpha_{rank} / beta_{rank} across PP0–PP3 · node arbiter = physical Host cap only",
            20,
            580,
            380,
            38,
            parent=gid,
            fill=GREEN_FILL,
            stroke=GREEN_STROKE,
            font_color="#276749",
            font_size=11,
            bold=True,
            align="center",
            spacing=4,
        )
        edge(
            f"edge-dp-{rank}-adapter-connector",
            f"dp-{rank}-adapter",
            f"dp-{rank}-connector",
            parent=gid,
            color=GREEN_STROKE,
            dashed=True,
            exit_x=1.0,
            exit_y=0.5,
            entry_x=1.0,
            entry_y=0.5,
            points=[
                (dp_abs_x + 412, 1400),
                (dp_abs_x + 412, 1105),
            ],
            width=1.8,
        )
        edge(
            f"edge-dp-{rank}-connector-hbm",
            f"dp-{rank}-connector",
            f"dp-{rank}-hbm",
            parent=gid,
            color=ORANGE_STROKE,
            exit_x=0.237,
            exit_y=1.0,
            entry_x=0.5,
            entry_y=0.0,
            width=1.8,
        )
        edge(
            f"edge-dp-{rank}-connector-dram",
            f"dp-{rank}-connector",
            f"dp-{rank}-dram",
            parent=gid,
            color=ORANGE_STROKE,
            exit_x=0.763,
            exit_y=1.0,
            entry_x=0.5,
            entry_y=0.0,
            width=1.8,
        )
        edge(
            f"edge-dp-{rank}-feedback-adapter",
            f"dp-{rank}-feedback",
            f"dp-{rank}-adapter",
            parent=gid,
            color=GREEN_STROKE,
            dashed=True,
            exit_x=0.5,
            exit_y=1.0,
            entry_x=0.5,
            entry_y=0.0,
            width=1.6,
        )

    vertex(
        "kv-transfer-bus",
        "STAGE-ALIGNED KV · DP0.PPp → DP3.PPp for p=0..3 · contiguous prefix commits only after 4/4 stage ACKs",
        560,
        98,
        760,
        30,
        parent="inference-layer",
        fill=ORANGE_FILL,
        stroke=ORANGE_STROKE,
        font_color="#6B4A10",
        font_size=12,
        bold=True,
        align="center",
        spacing=4,
    )
    vertex(
        "failure-footer",
        "FAILURE / RETRY · use certified healthy b(r) · restore the verified all-stage contiguous prefix · recompute the remaining suffix",
        290,
        775,
        1250,
        28,
        parent="inference-layer",
        fill="#FFFFFF",
        stroke=GRAY_STROKE,
        font_color=MUTED,
        font_size=11,
        align="center",
        spacing=3,
    )

    # Router-to-engine bindings and the concrete data example.
    edge(
        "edge-exec-dp0",
        "execution-binding",
        "dp-0-pp-0",
        parent="system-host",
        color=BLUE_STROKE,
        exit_x=0.5,
        exit_y=1.0,
        entry_x=0.0,
        entry_y=0.5,
        points=[(310, 736), (55, 736), (55, 1009)],
        width=2.6,
    )
    edge(
        "edge-backup-dp3",
        "backup-binding",
        "dp-3-connector",
        parent="system-host",
        color=PURPLE_STROKE,
        dashed=True,
        exit_x=0.5,
        exit_y=1.0,
        entry_x=0.0,
        entry_y=0.5,
        points=[
            (1645, 736),
            (1916, 736),
            (1916, 884),
            (1429, 884),
            (1429, 1105),
        ],
        width=2.4,
    )
    edge(
        "edge-dp0-kv-bus",
        "dp-0-connector",
        "kv-transfer-bus",
        parent="inference-layer",
        color=ORANGE_STROKE,
        exit_x=1.0,
        exit_y=0.5,
        entry_x=0.0,
        entry_y=0.5,
        points=[(532, 1105), (532, 863)],
        width=2.6,
    )
    edge(
        "edge-kv-bus-dp3",
        "kv-transfer-bus",
        "dp-3-connector",
        parent="inference-layer",
        color=ORANGE_STROKE,
        exit_x=1.0,
        exit_y=0.5,
        entry_x=0.0,
        entry_y=0.5,
        points=[(1422, 863), (1422, 1105)],
        width=2.6,
    )

    # ------------------------------------------------------------------
    # Right-side implementation callouts.  These explain the three modules
    # without overloading the main deployment view.
    # ------------------------------------------------------------------
    group(
        "implementation-details",
        "CERTIFIED EXACT CONTROL · offline HiGHS → lightweight runtime plan",
        1900,
        52,
        590,
        1433,
        parent="system-host",
        header_fill=GRAY_FILL,
        body_fill="#FBFCFE",
        stroke=GRAY_STROKE,
        start_size=42,
    )
    group(
        "alg1-detail",
        "ALGORITHM 1 · APPLICATION SLACK HANDLER",
        20,
        58,
        550,
        320,
        parent="implementation-details",
        header_fill=YELLOW_FILL,
        body_fill=YELLOW_BODY,
        stroke=YELLOW_STROKE,
        start_size=38,
    )
    vertex(
        "alg1-detail-input",
        card(
            "Inputs",
            [
                "fixed DAG + progress + calibrated online duration distribution",
                "trace evaluation: oracle Slack may bypass this predictor",
            ],
        ),
        20,
        55,
        510,
        88,
        parent="alg1-detail",
        fill="#FFFFFF",
        stroke=YELLOW_STROKE,
        font_color="#5B4710",
        font_size=12,
        spacing=9,
    )
    vertex(
        "alg1-detail-compute",
        card(
            "Compute",
            ["sample K remaining-duration realizations", "CPM forward/backward → node Slack", "request Slack = conservative quantile"],
        ),
        20,
        150,
        510,
        105,
        parent="alg1-detail",
        fill=YELLOW_FILL,
        stroke=YELLOW_STROKE,
        font_color="#5B4710",
        font_size=12,
        spacing=9,
    )
    vertex(
        "alg1-detail-output",
        "OUTPUT · predicted_slack_ms · slack_quantile · slack_version",
        20,
        270,
        510,
        35,
        parent="alg1-detail",
        fill="#FFFFFF",
        stroke=YELLOW_STROKE,
        font_color="#5B4710",
        font_size=12,
        bold=True,
        align="center",
        spacing=5,
    )

    group(
        "router-detail",
        "ROUTER CONTRACT · CAUSAL ADMISSION ONLY",
        20,
        400,
        550,
        190,
        parent="implementation-details",
        header_fill=PURPLE_FILL,
        body_fill="#FFFCFF",
        stroke=PURPLE_STROKE,
        start_size=38,
    )
    vertex(
        "router-detail-body",
        card(
            "Observed state → immutable route version",
            [
                "choose e(r) from current health / load / cache state",
                "pass eligible backup candidates B(r) to the MILP",
                "accept certified b(r); never read a future failure manifest",
            ],
        ),
        20,
        55,
        510,
        110,
        parent="router-detail",
        fill="#FFFFFF",
        stroke=PURPLE_STROKE,
        font_color="#4A375C",
        font_size=12,
        spacing=9,
    )

    group(
        "alg2-detail",
        "ALGORITHM 2 · CERTIFIED ROUTE + STAGE PLACEMENT",
        20,
        612,
        550,
        400,
        parent="implementation-details",
        header_fill=PURPLE_FILL,
        body_fill="#FFFCFF",
        stroke=PURPLE_STROKE,
        start_size=38,
    )
    vertex(
        "alg2-detail-input",
        card(
            "Frozen finite-window inputs",
            [
                "Slack + causal e(r) + B(r) + chunks; hazard prior (not realized future)",
                "topology + resident/link stock + current ACKed H/M budget",
            ],
            title_size=15,
            body_size=11,
            body_margin=5,
        ),
        20,
        55,
        510,
        75,
        parent="alg2-detail",
        fill="#FFFFFF",
        stroke=PURPLE_STROKE,
        font_color="#4A375C",
        font_size=12,
        spacing=9,
    )
    vertex(
        "alg2-detail-vars",
        card(
            "Binary decisions",
            [
                "y[r,b]: exactly one immutable backup DP",
                "x[r,c,p,b,t]: one tier t ∈ {HBM, Host, Recompute}",
                "g[r,c,b,s]: exact all-PP-stage survival under scenario s",
            ],
        ),
        20,
        137,
        510,
        95,
        parent="alg2-detail",
        fill=PURPLE_FILL,
        stroke=PURPLE_STROKE,
        font_color="#4A375C",
        font_size=12,
        spacing=9,
    )
    vertex(
        "alg2-detail-compute",
        card(
            "Joint objective + hard constraints",
            [
                "min E[workflow loss] + lambda·CVaR_tau + copy/normal cost",
                "+ H/M budget cost + |delta H| / |delta M| switching cost",
                "stage HBM, DP Host, node Host, link BW, AND + contiguous prefix",
            ],
        ),
        20,
        239,
        510,
        95,
        parent="alg2-detail",
        fill="#FFFFFF",
        stroke=PURPLE_STROKE,
        font_color="#4A375C",
        font_size=12,
        spacing=9,
    )
    vertex(
        "alg2-detail-output",
        "OUTPUT · certified b(r) + x[r,c,p,b,t] · all-stage contiguous recoverable prefix",
        20,
        341,
        510,
        38,
        parent="alg2-detail",
        fill="#FFFFFF",
        stroke=PURPLE_STROKE,
        font_color="#4A375C",
        font_size=12,
        bold=True,
        align="center",
        spacing=5,
    )

    group(
        "alg3-detail",
        "ALGORITHM 3 · EXACT DP BUDGET + ATOMIC PUBLICATION",
        20,
        1034,
        550,
        360,
        parent="implementation-details",
        header_fill=GREEN_FILL,
        body_fill=GREEN_BODY,
        stroke=GREEN_STROKE,
        start_size=38,
    )
    vertex(
        "alg3-detail-input",
        card(
            "Causal rolling state at sealed window K",
            [
                "resident HBM/Host + used links + observed unavailable ranks",
                "offered/admitted/unmet utility + pressure + recovery p99 + SLO + ACKed version",
            ],
            title_size=15,
            body_size=11,
            body_margin=5,
        ),
        20,
        55,
        510,
        85,
        parent="alg3-detail",
        fill="#FFFFFF",
        stroke=GREEN_STROKE,
        font_color="#276749",
        font_size=12,
        spacing=9,
    )
    vertex(
        "alg3-detail-compute",
        card(
            "Integer budget decisions in the same MILP",
            [
                "H_n: one per-stage HBM token limit shared by PP0–PP3",
                "M_n: DP-owned Host policy limit; node capacity is separate",
                "alpha_n=H_n/HBMmax_n; beta_n=M_n/Hostmax_n",
            ],
            title_size=15,
            body_size=11,
            body_margin=5,
        ),
        20,
        147,
        510,
        95,
        parent="alg3-detail",
        fill=GREEN_FILL,
        stroke=GREEN_STROKE,
        font_color="#276749",
        font_size=12,
        spacing=9,
    )
    vertex(
        "alg3-detail-publication",
        card(
            "Publication + certification gate",
            [
                "HiGHS OPTIMAL + zero gap + primal=dual + independent validation",
                "decision SHA + replay; publish after 4/4 ACK",
                "route-only causal rolling reuses fixed H/M",
            ],
            title_size=15,
            body_size=11,
            body_margin=5,
        ),
        20,
        249,
        510,
        70,
        parent="alg3-detail",
        fill="#FFFFFF",
        stroke=GREEN_STROKE,
        font_color="#276749",
        font_size=11,
        spacing=8,
    )
    vertex(
        "alg3-detail-output",
        "PLAN · finite-instance optimum only · publish_new | reuse_current_fixed",
        20,
        326,
        510,
        28,
        parent="alg3-detail",
        fill="#FFFFFF",
        stroke=GREEN_STROKE,
        font_color="#276749",
        font_size=11,
        bold=True,
        align="center",
        spacing=5,
    )

    # Direct pointers from concrete components to their detailed descriptions.
    edge(
        "edge-alg1-callout",
        "algorithm-1-handler",
        "alg1-detail",
        parent="system-host",
        color=YELLOW_STROKE,
        dashed=True,
        exit_x=1.0,
        exit_y=0.5,
        entry_x=0.0,
        entry_y=0.35,
    )
    edge(
        "edge-router-callout",
        "top2-scheduler",
        "router-detail",
        parent="system-host",
        color=PURPLE_STROKE,
        dashed=True,
        exit_x=0.5,
        exit_y=0.0,
        entry_x=0.0,
        entry_y=0.5,
        points=[(985, 532), (1924, 532), (1924, 632)],
    )
    edge(
        "edge-connector-callout",
        "dp-3-connector",
        "alg2-detail",
        parent="system-host",
        color=PURPLE_STROKE,
        dashed=True,
        exit_x=1.0,
        exit_y=0.5,
        entry_x=0.0,
        entry_y=0.5,
        points=[(1922, 1105), (1922, 949)],
    )
    edge(
        "edge-adapter-callout",
        "dp-3-adapter",
        "alg3-detail",
        parent="system-host",
        color=GREEN_STROKE,
        dashed=True,
        exit_x=1.0,
        exit_y=0.5,
        entry_x=0.0,
        entry_y=0.5,
        points=[(1932, 1400), (1932, 1351)],
    )

    return ET.ElementTree(mxfile)


def indent_xml(element: ET.Element, level: int = 0) -> None:
    prefix = "\n" + level * "  "
    child_prefix = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_prefix
        for child in element:
            indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_prefix
        if element[-1].tail is not None:
            element[-1].tail = prefix
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = prefix


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tree = build()
    indent_xml(tree.getroot())
    tree.write(OUT, encoding="utf-8", xml_declaration=True)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
