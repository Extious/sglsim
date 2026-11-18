#!/usr/bin/env python3
"""Fail-closed structural audit for the main-paper AegisKV overview."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET


DEFAULT_DRAWIO = Path(__file__).with_name("aegiskv-paper-system-design-v2.drawio")
CONTROL_COLOR = "#1F4E79"
DATA_COLOR = "#C65911"
ALLOWED_COLORS = {
    "#FFFFFF",
    "#F8FAFC",
    "#EAF2F8",
    "#FFF7ED",
    "#0F172A",
    "#475569",
    "#64748B",
    CONTROL_COLOR,
    DATA_COLOR,
}
DATA_EDGES = {
    "edge-p2p": ("source-transfer", "peer-hbm"),
    "edge-hbm-recover": ("peer-hbm", "recovery-executor"),
    "edge-destage": ("peer-hbm", "dram-backing"),
    "edge-direct-dram": ("source-transfer", "dram-backing"),
    "edge-store-restore": ("store-transfer", "recovery-executor"),
}
REQUIRED_IDS = {
    "application-plane",
    "algorithm-1",
    "request-slack",
    "control-plane",
    "router",
    "aegis-controller",
    "kv-directory",
    "algorithm-2",
    "algorithm-3",
    "failure-control",
    "data-plane",
    "control-bus",
    "source-worker",
    "target-worker",
    "peer-hbm",
    "dram-backing",
    "recovery-executor",
    "store",
}
REQUIRED_TEXT = {
    "Algorithm 1",
    "Causal Slack",
    "Algorithm 2",
    "SlackLandlord",
    "Algorithm 3",
    "GuardedAIMD",
    "Peer-HBM quota alpha",
    "Store Master / KV Directory",
    "Failure:",
    "COMMITTED",
    "ADMIT_HBM",
    "ADMIT_DRAM",
    "BYPASS",
    "destage",
    "Store restore",
    "tensors bypass Controller",
}


def style_map(style: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in style.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


def audit(path: Path) -> None:
    root = ET.parse(path).getroot()
    model = root.find("./diagram/mxGraphModel")
    assert model is not None, "missing mxGraphModel"
    assert (model.get("pageWidth"), model.get("pageHeight")) == ("1600", "940")

    cells = root.findall(".//mxCell")
    ids = [cell.get("id") for cell in cells]
    assert all(ids), "every cell must have an id"
    assert len(ids) == len(set(ids)), "duplicate cell id"
    assert REQUIRED_IDS <= set(ids), f"missing required ids: {sorted(REQUIRED_IDS - set(ids))}"
    by_id = {cell.get("id"): cell for cell in cells}

    values = "\n".join(cell.get("value", "") for cell in cells)
    for phrase in REQUIRED_TEXT:
        assert phrase in values, f"missing visible phrase: {phrase}"
    assert "beta" not in values.lower(), "stale beta budget remains in the overview"
    assert not re.search(r"\b(?:GB/s|MB/s|ms|seconds?)\b", values), (
        "the architecture overview must not embed unverified measurements"
    )

    colors = {
        color.upper()
        for cell in cells
        for color in re.findall(r"#[0-9A-Fa-f]{6}", cell.get("style", ""))
    }
    assert colors <= ALLOWED_COLORS, f"unexpected palette entries: {sorted(colors - ALLOWED_COLORS)}"

    for cell in cells:
        styles = style_map(cell.get("style", ""))
        if "fontSize" in styles:
            assert float(styles["fontSize"]) >= 18, f"tiny text in {cell.get('id')}"
            assert styles.get("fontFamily") == "Noto Sans CJK SC", (
                f"non-authoritative font in {cell.get('id')}"
            )

    edges = [cell for cell in cells if cell.get("edge") == "1"]
    assert edges, "diagram has no edges"
    for edge in edges:
        geometry = edge.find("mxGeometry")
        assert geometry is not None and geometry.get("relative") == "1", (
            f"edge {edge.get('id')} lacks relative geometry"
        )
        assert not edge.get("value"), f"edge label is forbidden: {edge.get('id')}"
        styles = style_map(edge.get("style", ""))
        color = styles.get("strokeColor", "").upper()
        assert color in {CONTROL_COLOR, DATA_COLOR}, (
            f"edge {edge.get('id')} has a third semantic color: {color}"
        )
        expected = DATA_EDGES.get(edge.get("id"))
        if expected is None:
            assert color == CONTROL_COLOR, f"control edge {edge.get('id')} is not blue"
        else:
            assert color == DATA_COLOR, f"KV edge {edge.get('id')} is not orange"
            assert (edge.get("source"), edge.get("target")) == expected

        points = geometry.findall("./Array[@as='points']/mxPoint")
        for first, second in zip(points, points[1:]):
            assert first.get("x") == second.get("x") or first.get("y") == second.get("y"), (
                f"diagonal explicit segment in {edge.get('id')}"
            )

    assert set(DATA_EDGES) <= {edge.get("id") for edge in edges}
    assert len([edge for edge in edges if style_map(edge.get("style", "")).get("strokeColor", "").upper() == DATA_COLOR]) == len(DATA_EDGES)

    print(
        "PASS: compact three-plane overview; neutral palette; exactly two arrow "
        "colors; five direct KV paths; readable type; no measured claims."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drawio", nargs="?", type=Path, default=DEFAULT_DRAWIO)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    audit(arguments.drawio.resolve())
