#!/usr/bin/env python3
"""Render a deterministic visual-QA preview of system_overview.drawio.

This is deliberately a small subset renderer for the primitives emitted by
generate_system_overview.py.  It lets headless CI catch font fallback and
label-density problems even when the diagrams.net desktop CLI is unavailable.
"""

from __future__ import annotations

import argparse
import html
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# gpu10 only ships the wide monospace fallback.  Using it here is useful: if
# labels fit under this worst-case font, they also fit under Noto Sans CJK.
FONT_REGULAR = Path("/usr/share/fonts/dejavu/DejaVuSansMono.ttf")
FONT_BOLD = Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf")


def style_map(style: str) -> dict[str, str]:
    return {
        key: value
        for item in style.split(";")
        if "=" in item
        for key, value in [item.split("=", 1)]
    }


def geometry(cell: ET.Element) -> tuple[float, float, float, float] | None:
    node = cell.find("mxGeometry")
    if node is None or node.get("relative") == "1":
        return None
    return tuple(float(node.get(key, "0")) for key in ("x", "y", "width", "height"))


def absolute_geometry(
    cell: ET.Element,
    by_id: dict[str, ET.Element],
    cache: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    cid = cell.get("id", "")
    if cid in cache:
        return cache[cid]
    own = geometry(cell)
    if own is None:
        raise ValueError(f"cell {cid} lacks absolute geometry")
    x, y, width, height = own
    parent = cell.get("parent", "1")
    if parent not in {"0", "1"}:
        px, py, _pw, _ph = absolute_geometry(by_id[parent], by_id, cache)
        x += px
        y += py
    cache[cid] = (x, y, width, height)
    return cache[cid]


def plain_lines(value: str) -> list[str]:
    text = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return [
        html.unescape(line).strip()
        for line in text.splitlines()
        if html.unescape(line).strip()
    ]


def wrap_line(draw: ImageDraw.ImageDraw, line: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    if draw.textlength(line, font=font) <= width:
        return [line]
    words = line.split()
    if not words:
        return [line]
    wrapped: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            wrapped.append(current)
            current = word
    wrapped.append(current)
    return wrapped


def edge_route(
    edge: ET.Element,
    by_id: dict[str, ET.Element],
    cache: dict[str, tuple[float, float, float, float]],
) -> list[tuple[float, float]]:
    source = absolute_geometry(by_id[edge.get("source", "")], by_id, cache)
    target = absolute_geometry(by_id[edge.get("target", "")], by_id, cache)
    styles = style_map(edge.get("style", ""))
    sx = source[0] + source[2] * float(styles.get("exitX", "0.5"))
    sy = source[1] + source[3] * float(styles.get("exitY", "0.5"))
    tx = target[0] + target[2] * float(styles.get("entryX", "0.5"))
    ty = target[1] + target[3] * float(styles.get("entryY", "0.5"))
    route = [(sx, sy)]
    node = edge.find("mxGeometry/Array[@as='points']")
    if node is not None:
        route.extend(
            (float(point.get("x", "0")), float(point.get("y", "0")))
            for point in node.findall("mxPoint")
        )
    if len(route) == 1:
        if abs(tx - sx) >= abs(ty - sy):
            midpoint = (sx + tx) / 2
            route.extend([(midpoint, sy), (midpoint, ty)])
        else:
            midpoint = (sy + ty) / 2
            route.extend([(sx, midpoint), (tx, midpoint)])
    route.append((tx, ty))
    return route


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 10
    wing = 4.5
    back_x = end[0] - length * math.cos(angle)
    back_y = end[1] - length * math.sin(angle)
    left = (back_x + wing * math.sin(angle), back_y - wing * math.cos(angle))
    right = (back_x - wing * math.sin(angle), back_y + wing * math.cos(angle))
    draw.polygon([end, left, right], fill=color)


def render(drawio: Path, output: Path) -> None:
    document = ET.parse(drawio).getroot()
    model = document.find(".//mxGraphModel")
    if model is None:
        raise ValueError("mxGraphModel missing")
    width = int(model.get("pageWidth", "1800"))
    height = int(model.get("pageHeight", "1120"))
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    cells = document.findall(".//mxCell")
    by_id = {cell.get("id", ""): cell for cell in cells}
    vertices = [cell for cell in cells if cell.get("vertex") == "1"]
    groups = [cell for cell in vertices if "swimlane" in cell.get("style", "")]
    leaves = [cell for cell in vertices if cell not in groups]
    cache: dict[str, tuple[float, float, float, float]] = {}

    # Containers are nested, so paint outer groups before inner groups.
    def depth(cell: ET.Element) -> int:
        result = 0
        parent = cell.get("parent", "1")
        while parent not in {"0", "1"}:
            result += 1
            parent = by_id[parent].get("parent", "1")
        return result

    for cell in sorted(groups, key=depth):
        x, y, w, h = absolute_geometry(cell, by_id, cache)
        styles = style_map(cell.get("style", ""))
        header = int(float(styles.get("startSize", "36")))
        stroke = styles.get("strokeColor", "#A5ADBA")
        draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill=styles.get("swimlaneFillColor", "#FFFFFF"), outline=stroke, width=2)
        draw.rectangle((x + 1, y + 1, x + w - 1, y + header), fill=styles.get("fillColor", "#F1F3F5"))
        draw.line((x, y + header, x + w, y + header), fill=stroke, width=2)

    for edge in [cell for cell in cells if cell.get("edge") == "1"]:
        styles = style_map(edge.get("style", ""))
        color = styles.get("strokeColor", "#5B8DB8")
        route = edge_route(edge, by_id, cache)
        draw.line(route, fill=color, width=2, joint="curve")
        draw_arrow(draw, route[-2], route[-1], color)
        if styles.get("startArrow") == "block":
            draw_arrow(draw, route[1], route[0], color)

    for cell in leaves:
        x, y, w, h = absolute_geometry(cell, by_id, cache)
        styles = style_map(cell.get("style", ""))
        fill = styles.get("fillColor", "#FFFFFF")
        stroke = styles.get("strokeColor", "#A5ADBA")
        if fill != "none" or stroke != "none":
            draw.rounded_rectangle(
                (x, y, x + w, y + h),
                radius=8 if styles.get("rounded") == "1" else 0,
                fill=None if fill == "none" else fill,
                outline=None if stroke == "none" else stroke,
                width=2 if stroke != "none" else 1,
            )

    def paint_label(cell: ET.Element, *, group: bool) -> None:
        x, y, w, h = absolute_geometry(cell, by_id, cache)
        styles = style_map(cell.get("style", ""))
        size = max(9, int(float(styles.get("fontSize", "12"))))
        bold = styles.get("fontStyle") == "1" or group
        font = ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)
        color = styles.get("fontColor", "#172B4D")
        lines = plain_lines(cell.get("value", ""))
        if group:
            header = int(float(styles.get("startSize", "36")))
            lines = [" ".join(lines)]
            text_y = y + (header - size) / 2 - 1
            draw.text((x + 12, text_y), lines[0], fill=color, font=font)
            return
        spacing = int(float(styles.get("spacing", "10")))
        wrapped = [part for line in lines for part in wrap_line(draw, line, font, int(w - 2 * spacing))]
        line_height = size + 4
        total_height = len(wrapped) * line_height - 4
        text_y = y + max(spacing, (h - total_height) / 2)
        centered = styles.get("align") == "center"
        for line in wrapped:
            text_x = x + spacing
            if centered:
                text_x = x + (w - draw.textlength(line, font=font)) / 2
            draw.text((text_x, text_y), line, fill=color, font=font)
            text_y += line_height

    for cell in groups:
        paint_label(cell, group=True)
    for cell in leaves:
        paint_label(cell, group=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drawio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.drawio.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
