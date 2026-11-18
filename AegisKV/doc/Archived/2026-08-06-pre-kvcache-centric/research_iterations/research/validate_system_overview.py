#!/usr/bin/env python3
"""Fail-closed structural checks for the AegisKV three-layer Draw.io."""

from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GROUP_IDS = {
    "system-host",
    "application-layer",
    "router-layer",
    "inference-layer",
    "implementation-details",
    "alg1-detail",
    "router-detail",
    "alg2-detail",
    "alg3-detail",
    "dp-0",
    "dp-1",
    "dp-2",
    "dp-3",
}

CONTENT_IDS = {
    "workflow-input",
    "workflow-builder",
    "algorithm-1-handler",
    "request-envelope",
    "top2-scheduler",
    "execution-binding",
    "backup-binding",
    "kv-transfer-bus",
    "failure-footer",
    "alg1-detail-compute",
    "router-detail-body",
    "alg2-detail-vars",
    "alg2-detail-compute",
    "alg3-detail-compute",
    "alg3-detail-publication",
}
for rank in range(4):
    CONTENT_IDS.update(
        {
            f"dp-{rank}-role",
            f"dp-{rank}-adapter",
            f"dp-{rank}-connector",
            f"dp-{rank}-hbm",
            f"dp-{rank}-dram",
            f"dp-{rank}-feedback",
            f"dp-{rank}-scope",
            *(f"dp-{rank}-pp-{pp}" for pp in range(4)),
        }
    )


def geometry(cell: ET.Element | None) -> tuple[float, float, float, float] | None:
    node = cell.find("mxGeometry") if cell is not None else None
    if node is None or node.get("relative") == "1":
        return None
    return tuple(float(node.get(key, "0")) for key in ("x", "y", "width", "height"))


def absolute_geometries(
    by_id: dict[str, ET.Element],
) -> dict[str, tuple[float, float, float, float] | None]:
    """Resolve nested Draw.io coordinates into page coordinates."""

    cache: dict[str, tuple[float, float, float, float] | None] = {}

    def resolve(cid: str) -> tuple[float, float, float, float] | None:
        if cid in cache:
            return cache[cid]
        cell = by_id.get(cid)
        own = geometry(cell)
        if own is None:
            cache[cid] = None
            return None
        x, y, width, height = own
        parent_id = cell.get("parent", "1") if cell is not None else "1"
        if parent_id not in {"0", "1"}:
            parent = resolve(parent_id)
            if parent is not None:
                x += parent[0]
                y += parent[1]
        result = (x, y, width, height)
        cache[cid] = result
        return result

    for cid in by_id:
        resolve(cid)
    return cache


def edge_route(
    edge: ET.Element,
    by_id: dict[str, ET.Element],
    absolute: dict[str, tuple[float, float, float, float] | None],
) -> list[tuple[float, float]]:
    """Approximate the orthogonal route used by the deterministic preview."""

    source = absolute.get(edge.get("source", ""))
    target = absolute.get(edge.get("target", ""))
    if source is None or target is None:
        return []
    styles = style_map(edge.get("style", ""))
    sx = source[0] + source[2] * float(styles.get("exitX", "0.5"))
    sy = source[1] + source[3] * float(styles.get("exitY", "0.5"))
    tx = target[0] + target[2] * float(styles.get("entryX", "0.5"))
    ty = target[1] + target[3] * float(styles.get("entryY", "0.5"))
    route: list[tuple[float, float]] = [(sx, sy)]
    node = edge.find("mxGeometry")
    points = node.find("Array[@as='points']") if node is not None else None
    if points is not None:
        parent = absolute.get(edge.get("parent", "1"))
        parent_x = parent[0] if parent is not None else 0.0
        parent_y = parent[1] if parent is not None else 0.0
        route.extend(
            (
                float(point.get("x", "0")) + parent_x,
                float(point.get("y", "0")) + parent_y,
            )
            for point in points.findall("mxPoint")
        )
    if len(route) == 1:
        if abs(tx - sx) >= abs(ty - sy):
            midpoint = (sx + tx) / 2
            route.extend([(midpoint, sy), (midpoint, ty)])
        else:
            midpoint = (sy + ty) / 2
            route.extend([(sx, midpoint), (tx, midpoint)])
    route.append((tx, ty))
    compact: list[tuple[float, float]] = []
    for point in route:
        if not compact or point != compact[-1]:
            compact.append(point)
    return compact


def segment_crosses_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: tuple[float, float, float, float],
    *,
    inset: float = 1.5,
) -> bool:
    """Return true only when a segment enters a card's interior."""

    ax, ay = start
    bx, by = end
    x, y, width, height = rect
    left, right = x + inset, x + width - inset
    top, bottom = y + inset, y + height - inset
    if left >= right or top >= bottom:
        return False
    if abs(ay - by) < 1e-6:
        return top < ay < bottom and max(min(ax, bx), left) < min(max(ax, bx), right)
    if abs(ax - bx) < 1e-6:
        return left < ax < right and max(min(ay, by), top) < min(max(ay, by), bottom)
    # Diagonal routes are invalid elsewhere, but Liang-Barsky keeps this audit
    # fail-closed if one slips into the XML.
    dx, dy = bx - ax, by - ay
    p = (-dx, dx, -dy, dy)
    q = (ax - left, right - ax, ay - top, bottom - ay)
    lower, upper = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-9:
            if qi < 0:
                return False
            continue
        ratio = qi / pi
        if pi < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return upper > 0 and lower < 1


def orthogonal_segments_intersect(
    left: tuple[tuple[float, float], tuple[float, float]],
    right: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    """Detect crossings and collinear overlaps between orthogonal segments."""

    (ax, ay), (bx, by) = left
    (cx, cy), (dx, dy) = right
    left_h = abs(ay - by) < 1e-6
    right_h = abs(cy - dy) < 1e-6
    left_v = abs(ax - bx) < 1e-6
    right_v = abs(cx - dx) < 1e-6
    if left_h and right_h:
        return abs(ay - cy) < 1e-6 and max(min(ax, bx), min(cx, dx)) < min(
            max(ax, bx), max(cx, dx)
        )
    if left_v and right_v:
        return abs(ax - cx) < 1e-6 and max(min(ay, by), min(cy, dy)) < min(
            max(ay, by), max(cy, dy)
        )
    if left_h and right_v:
        return min(ax, bx) < cx < max(ax, bx) and min(cy, dy) < ay < max(cy, dy)
    if left_v and right_h:
        return min(cx, dx) < ax < max(cx, dx) and min(ay, by) < cy < max(ay, by)
    return False


def style_map(style: str) -> dict[str, str]:
    return {
        key: value
        for item in style.split(";")
        if "=" in item
        for key, value in [item.split("=", 1)]
    }


def plain_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(value).split())


def accepted_evidence(path: Path) -> tuple[bool, str | None, list[str]]:
    if not path.is_file():
        return False, None, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, None, [f"final evidence JSON is invalid: {exc}"]
    accepted = bool(payload.get("gate_pass", False))
    run_id = str(payload.get("run_id") or path.stem)
    return accepted, run_id, ([] if accepted else ["final evidence gate_pass is false"])


def validate(drawio_path: Path, evidence_path: Path, *, require_measured: bool) -> dict[str, Any]:
    document = ET.parse(drawio_path).getroot()
    diagrams = document.findall("diagram")
    model = document.find(".//mxGraphModel")
    cells = document.findall(".//mxCell")
    by_id = {cell.get("id", ""): cell for cell in cells}
    vertices = [cell for cell in cells if cell.get("vertex") == "1"]
    edges = [cell for cell in cells if cell.get("edge") == "1"]
    absolute = absolute_geometries(by_id)
    issues: list[str] = []

    if len(diagrams) != 1:
        issues.append(f"expected one overview page, found {len(diagrams)}")
    ids = [cell.get("id", "") for cell in cells]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        issues.append(f"duplicate cell IDs: {duplicates}")

    page_size = None
    if model is None:
        issues.append("mxGraphModel is missing")
    else:
        page_size = (int(model.get("pageWidth", "0")), int(model.get("pageHeight", "0")))
        if page_size != (2600, 1700):
            issues.append(f"unexpected page size: {page_size}")
        if model.get("shadow") != "0":
            issues.append("overview must disable shadows")

    missing = sorted((GROUP_IDS | CONTENT_IDS) - set(by_id))
    if missing:
        issues.append(f"required cells are missing: {missing}")
    for cid in GROUP_IDS:
        cell = by_id.get(cid)
        if cell is not None and "swimlane" not in cell.get("style", ""):
            issues.append(f"{cid} is not a swimlane container")

    bad_edges = []
    labelled_edges = []
    non_orthogonal = []
    for edge in edges:
        edge_id = edge.get("id", "")
        node = edge.find("mxGeometry")
        if (
            edge.get("source") not in by_id
            or edge.get("target") not in by_id
            or node is None
            or node.get("relative") != "1"
        ):
            bad_edges.append(edge_id)
        if plain_text(edge.get("value", "")):
            labelled_edges.append(edge_id)
        if style_map(edge.get("style", "")).get("edgeStyle") != "orthogonalEdgeStyle":
            non_orthogonal.append(edge_id)
    if bad_edges:
        issues.append(f"invalid edge references/geometries: {bad_edges}")
    if labelled_edges:
        issues.append(f"edge labels are forbidden: {labelled_edges}")
    if non_orthogonal:
        issues.append(f"non-orthogonal edges: {non_orthogonal}")

    overflow = []
    for cell in vertices:
        parent_id = cell.get("parent", "")
        if parent_id in {"0", "1"}:
            continue
        child = geometry(cell)
        parent = geometry(by_id.get(parent_id))
        if child is None or parent is None:
            continue
        x, y, width, height = child
        _px, _py, parent_width, parent_height = parent
        if x < 0 or y < 0 or x + width > parent_width or y + height > parent_height:
            overflow.append((cell.get("id"), parent_id))
    if overflow:
        issues.append(f"child vertices overflow their containers: {overflow}")

    siblings: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for cell in vertices:
        cid = cell.get("id", "")
        box = geometry(cell)
        if box is not None and cid not in GROUP_IDS:
            siblings[cell.get("parent", "")].append((cid, box))
    overlaps = []
    for parent_id, items in siblings.items():
        for index, (left_id, left) in enumerate(items):
            lx, ly, lw, lh = left
            for right_id, right in items[index + 1 :]:
                rx, ry, rw, rh = right
                if min(lx + lw, rx + rw) > max(lx, rx) and min(ly + lh, ry + rh) > max(ly, ry):
                    overlaps.append((parent_id, left_id, right_id))
    if overlaps:
        issues.append(f"sibling content cards overlap: {overlaps}")

    content_group_overlaps: list[tuple[str, str, str]] = []
    groups_by_parent: dict[
        str, list[tuple[str, tuple[float, float, float, float]]]
    ] = defaultdict(list)
    for cell in vertices:
        cid = cell.get("id", "")
        box = geometry(cell)
        if cid in GROUP_IDS and box is not None:
            groups_by_parent[cell.get("parent", "")].append((cid, box))
    for parent_id, contents in siblings.items():
        for content_id, content in contents:
            cx, cy, cw, ch = content
            for group_id, group_box in groups_by_parent[parent_id]:
                gx, gy, gw, gh = group_box
                if min(cx + cw, gx + gw) > max(cx, gx) and min(
                    cy + ch, gy + gh
                ) > max(cy, gy):
                    content_group_overlaps.append((parent_id, content_id, group_id))
    if content_group_overlaps:
        issues.append(
            f"content cards overlap sibling containers: {content_group_overlaps}"
        )

    routes = {edge.get("id", ""): edge_route(edge, by_id, absolute) for edge in edges}
    out_of_page_edge_points: list[tuple[str, int, tuple[float, float]]] = []
    if page_size is not None:
        page_width, page_height = page_size
        for edge_id, route in routes.items():
            for point_index, (point_x, point_y) in enumerate(route):
                if not (0 <= point_x <= page_width and 0 <= point_y <= page_height):
                    out_of_page_edge_points.append(
                        (edge_id, point_index, (point_x, point_y))
                    )
    if out_of_page_edge_points:
        issues.append(f"edge routes leave the page: {out_of_page_edge_points}")

    diagonal_segments: list[tuple[str, int]] = []
    for edge_id, route in routes.items():
        for segment_index, ((ax, ay), (bx, by)) in enumerate(zip(route, route[1:])):
            if abs(ax - bx) > 1e-6 and abs(ay - by) > 1e-6:
                diagonal_segments.append((edge_id, segment_index))
    if diagonal_segments:
        issues.append(f"diagonal edge segments: {diagonal_segments}")

    collision_cards = {
        cell.get("id", ""): absolute.get(cell.get("id", ""))
        for cell in vertices
        if cell.get("id", "") not in GROUP_IDS | {"title", "subtitle"}
    }
    edge_card_collisions: list[tuple[str, str]] = []
    for edge in edges:
        edge_id = edge.get("id", "")
        endpoints = {edge.get("source", ""), edge.get("target", "")}
        route = routes[edge_id]
        for card_id, box in collision_cards.items():
            if card_id in endpoints or box is None:
                continue
            if any(
                segment_crosses_rect(start, end, box)
                for start, end in zip(route, route[1:])
            ):
                edge_card_collisions.append((edge_id, card_id))
    if edge_card_collisions:
        issues.append(f"edges cross content cards: {edge_card_collisions}")

    edge_edge_collisions: list[tuple[str, str]] = []
    for index, left_edge in enumerate(edges):
        left_id = left_edge.get("id", "")
        left_endpoints = {
            left_edge.get("source", ""),
            left_edge.get("target", ""),
        }
        left_segments = list(zip(routes[left_id], routes[left_id][1:]))
        for right_edge in edges[index + 1 :]:
            right_id = right_edge.get("id", "")
            right_endpoints = {
                right_edge.get("source", ""),
                right_edge.get("target", ""),
            }
            if left_endpoints & right_endpoints:
                continue
            right_segments = list(zip(routes[right_id], routes[right_id][1:]))
            if any(
                orthogonal_segments_intersect(left_segment, right_segment)
                for left_segment in left_segments
                for right_segment in right_segments
            ):
                edge_edge_collisions.append((left_id, right_id))
    if edge_edge_collisions:
        issues.append(f"unrelated edges intersect: {edge_edge_collisions}")

    tiny_fonts = []
    wrong_fonts = []
    for cell in vertices:
        styles = style_map(cell.get("style", ""))
        if float(styles.get("fontSize", "12")) < 11:
            tiny_fonts.append(cell.get("id"))
        if styles.get("fontFamily") != "Noto Sans CJK SC":
            wrong_fonts.append(cell.get("id"))
    if tiny_fonts:
        issues.append(f"fonts smaller than 11px: {tiny_fonts}")
    if wrong_fonts:
        issues.append(f"vertices outside the Noto Sans CJK style: {wrong_fonts}")

    all_text = " ".join(plain_text(cell.get("value", "")) for cell in cells)
    required_text = [
        "LAYER 1 · APPLICATION",
        "Agent / Workflow Handler",
        "Algorithm 1 · Slack Handler",
        "predicted_slack_ms",
        "trace evaluation: oracle Slack may bypass this predictor",
        "slack_source",
        "LAYER 2 · ROUTER",
        "Causal Router + Certified Plan",
        "Execution target e(r)",
        "Backup target b(r)",
        "future manifest is forbidden",
        "candidates B(r)",
        "LAYER 3 · INFERENCE ENGINE",
        "DP RANK 0",
        "DP RANK 1",
        "DP RANK 2",
        "DP RANK 3",
        "PP0",
        "PP1",
        "PP2",
        "PP3",
        "Certified Connector",
        "Budget Publisher",
        "CONTROL PLANE · STAGE-LOCAL EXECUTOR",
        "CONTROL PLANE · DP-OWNED BUDGET",
        "DATA PLANE · GPU HBM",
        "DATA PLANE · Host DRAM",
        "ROLLING STATE K",
        "DP-OWNED",
        "PPp → PPp only",
        "contiguous prefix commits only after 4/4 stage ACKs",
        "y[r,b]",
        "x[r,c,p,b,t]",
        "g[r,c,b,s]",
        "E[workflow loss]",
        "CVaR_tau",
        "H_n: one per-stage HBM token limit",
        "M_n: DP-owned Host policy limit",
        "HiGHS OPTIMAL",
        "zero gap",
        "reuse_current_fixed",
        "finite-instance optimum only",
    ]
    missing_text = [token for token in required_text if token not in all_text]
    if missing_text:
        issues.append(f"required overview content is missing: {missing_text}")

    measured, run_id, evidence_issues = accepted_evidence(evidence_path)
    issues.extend(evidence_issues)
    if require_measured and not measured:
        issues.append("measured evidence is required for final validation")

    return {
        "status": "pass" if not issues else "fail",
        "drawio": str(drawio_path),
        "page_size": page_size,
        "cell_count": len(cells),
        "vertex_count": len(vertices),
        "edge_count": len(edges),
        "edge_label_count": len(labelled_edges),
        "overlap_count": len(overlaps),
        "content_group_overlap_count": len(content_group_overlaps),
        "out_of_page_edge_point_count": len(out_of_page_edge_points),
        "diagonal_segment_count": len(diagonal_segments),
        "edge_card_collision_count": len(edge_card_collisions),
        "edge_edge_collision_count": len(edge_edge_collisions),
        "measured_evidence": measured,
        "accepted_run_id": run_id,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drawio", type=Path, required=True)
    parser.add_argument(
        "--final-evidence",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "presentation"
            / "experiments"
            / "qwen3_32b_dp4_pp4_final_summary.json"
        ),
    )
    parser.add_argument("--require-measured", action="store_true")
    args = parser.parse_args()
    result = validate(
        args.drawio.resolve(),
        args.final_evidence.resolve(),
        require_measured=args.require_measured,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
