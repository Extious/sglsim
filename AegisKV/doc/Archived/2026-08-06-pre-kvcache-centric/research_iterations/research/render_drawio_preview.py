from __future__ import annotations

import html
import math
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def style_map(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in style.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def plain(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</div>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value).replace("\xa0", " ")
    return "\n".join(part.strip() for part in value.splitlines() if part.strip())


def wrap_visual(text: str, max_units: int) -> list[str]:
    lines: list[str] = []
    for source in text.splitlines():
        wrapped = textwrap.wrap(
            source,
            width=max_units,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def main(src: Path, dst: Path) -> None:
    root = ET.parse(src).getroot()
    cells = root.findall(".//mxCell")
    by_id = {cell.get("id", ""): cell for cell in cells}
    model = root.find(".//mxGraphModel")
    page_width = int(model.get("pageWidth", "2400")) if model is not None else 2400
    page_height = int(model.get("pageHeight", "1500")) if model is not None else 1500
    scale = 0.75
    background = model.get("background", "#F1F5F9") if model is not None else "#F1F5F9"
    image = Image.new(
        "RGB", (int(page_width * scale), int(page_height * scale)), background
    )
    draw = ImageDraw.Draw(image)
    regular = "/usr/share/fonts/abattis-cantarell/Cantarell-Regular.otf"
    bold = "/usr/share/fonts/abattis-cantarell/Cantarell-Bold.otf"

    def raw_geom(cell: ET.Element) -> tuple[float, float, float, float] | None:
        node = cell.find("mxGeometry")
        if node is None or node.get("relative") == "1":
            return None
        return tuple(float(node.get(key, "0")) for key in ("x", "y", "width", "height"))

    geometry_cache: dict[str, tuple[float, float, float, float] | None] = {}
    clipped_cells: list[str] = []

    def geom(cell: ET.Element) -> tuple[float, float, float, float] | None:
        cid = cell.get("id", "")
        if cid in geometry_cache:
            return geometry_cache[cid]
        own = raw_geom(cell)
        if own is None:
            geometry_cache[cid] = None
            return None
        x, y, width, height = own
        parent_id = cell.get("parent", "1")
        parent = by_id.get(parent_id)
        if parent is not None and parent_id not in {"0", "1"}:
            parent_box = geom(parent)
            if parent_box is not None:
                x += parent_box[0]
                y += parent_box[1]
        result = (x, y, width, height)
        geometry_cache[cid] = result
        return result

    def draw_vertex(cell: ET.Element) -> None:
        box = geom(cell)
        if box is None:
            return
        x, y, width, height = box
        x0, y0, x1, y1 = (int(v * scale) for v in (x, y, x + width, y + height))
        styles = style_map(cell.get("style", ""))
        fill = styles.get("fillColor", "#FFFFFF")
        stroke = styles.get("strokeColor", "#CBD5E1")
        if fill == "none":
            fill = None
        if stroke == "none":
            stroke = None
        radius = 8 if styles.get("rounded", "0") == "1" else 0
        stroke_width = max(int(round(float(styles.get("strokeWidth", "1")))), 1)
        is_swimlane = "swimlane" in cell.get("style", "")
        if is_swimlane:
            body_fill = styles.get("swimlaneFillColor", fill or "#FFFFFF")
            header_height = int(float(styles.get("startSize", "36")) * scale)
            draw.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=radius,
                fill=body_fill,
                outline=stroke,
                width=stroke_width,
            )
            draw.rectangle(
                (x0 + stroke_width, y0 + stroke_width, x1 - stroke_width, y0 + header_height),
                fill=fill,
            )
            if stroke:
                draw.line(
                    (x0, y0 + header_height, x1, y0 + header_height),
                    fill=stroke,
                    width=stroke_width,
                )
        elif radius:
            draw.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=radius,
                fill=fill,
                outline=stroke,
                width=stroke_width,
            )
        else:
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=fill,
                outline=stroke,
                width=stroke_width,
            )

        value = plain(cell.get("value", ""))
        if not value:
            return
        base_size = float(styles.get("fontSize", "12"))
        font_path = (
            bold
            if int(styles.get("fontStyle", "0")) & 1
            or "font-weight:700" in cell.get("value", "")
            else regular
        )
        font = ImageFont.truetype(font_path, max(int(round(base_size * scale)), 7))
        max_units = max(int(width / 7.2), 8)
        lines = wrap_visual(value, max_units)
        line_height = font.size + 2
        available_height = height * scale - 10
        if is_swimlane:
            available_height = float(styles.get("startSize", "36")) * scale - 6
        max_lines = max(int(available_height / line_height), 1)
        clipped = len(lines) > max_lines
        lines = lines[:max_lines]
        if clipped and lines:
            clipped_cells.append(cell.get("id", ""))
            lines[-1] = lines[-1][:-1] + "…"
        rendered = "\n".join(lines)
        text_color = styles.get("fontColor", "#0F172A")
        text_height = len(lines) * line_height
        text_y = y0 + 5
        if styles.get("verticalAlign") == "middle":
            region_height = (
                float(styles.get("startSize", "36")) * scale
                if is_swimlane
                else height * scale
            )
            text_y = y0 + max((region_height - text_height) / 2, 3)
        if styles.get("align") == "center":
            draw.multiline_text(
                ((x0 + x1) / 2, text_y),
                rendered,
                font=font,
                fill=text_color,
                spacing=2,
                align="center",
                anchor="ma",
            )
        else:
            draw.multiline_text(
                (x0 + 7, text_y),
                rendered,
                font=font,
                fill=text_color,
                spacing=2,
            )

    background_ids = {
        cell.get("id", "")
        for cell in cells
        if cell.get("vertex") == "1"
        and (
            cell.get("id") == "page-bg"
            or "swimlane" in cell.get("style", "")
            or "container=1" in cell.get("style", "")
        )
    }
    for cell in cells:
        if cell.get("id") in background_ids:
            draw_vertex(cell)

    edge_labels: list[tuple[str, float, float, str, int]] = []

    def arrowhead(
        tip: tuple[float, float],
        previous: tuple[float, float],
        color: str,
        *,
        filled: bool,
    ) -> None:
        dx = tip[0] - previous[0]
        dy = tip[1] - previous[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        size = 7
        bx, by = tip[0] - ux * size, tip[1] - uy * size
        px, py = -uy * size * 0.55, ux * size * 0.55
        points = [tip, (bx + px, by + py), (bx - px, by - py)]
        if filled:
            draw.polygon(points, fill=color)
        else:
            draw.line([points[1], points[0], points[2]], fill=color, width=1)

    def route_midpoint(route: list[tuple[float, float]]) -> tuple[float, float]:
        lengths = [
            math.hypot(bx - ax, by - ay)
            for (ax, ay), (bx, by) in zip(route, route[1:])
        ]
        remaining = sum(lengths) / 2
        for ((ax, ay), (bx, by)), segment in zip(zip(route, route[1:]), lengths):
            if remaining <= segment and segment > 0:
                ratio = remaining / segment
                return (ax + (bx - ax) * ratio, ay + (by - ay) * ratio)
            remaining -= segment
        return route[-1]

    # Approximate edge preview. The editable draw.io file remains authoritative.
    for cell in cells:
        if cell.get("edge") != "1":
            continue
        source = geom(by_id.get(cell.get("source", ""), ET.Element("none")))
        target = geom(by_id.get(cell.get("target", ""), ET.Element("none")))
        if source is None or target is None:
            continue
        styles = style_map(cell.get("style", ""))
        exit_x = float(styles.get("exitX", "0.5"))
        exit_y = float(styles.get("exitY", "0.5"))
        entry_x = float(styles.get("entryX", "0.5"))
        entry_y = float(styles.get("entryY", "0.5"))
        sx = source[0] + source[2] * exit_x
        sy = source[1] + source[3] * exit_y
        tx = target[0] + target[2] * entry_x
        ty = target[1] + target[3] * entry_y
        geometry = cell.find("mxGeometry")
        route: list[tuple[float, float]] = [(sx, sy)]
        if geometry is not None:
            array = geometry.find("Array[@as='points']")
            if array is not None:
                parent = geom(by_id.get(cell.get("parent", "1"), ET.Element("none")))
                parent_x = parent[0] if parent is not None else 0.0
                parent_y = parent[1] if parent is not None else 0.0
                route.extend(
                    (
                        float(point.get("x", "0")) + parent_x,
                        float(point.get("y", "0")) + parent_y,
                    )
                    for point in array.findall("mxPoint")
                )
        if len(route) == 1:
            if abs(tx - sx) >= abs(ty - sy):
                midpoint = (sx + tx) / 2
                route.extend([(midpoint, sy), (midpoint, ty)])
            else:
                midpoint = (sy + ty) / 2
                route.extend([(sx, midpoint), (tx, midpoint)])
        route.append((tx, ty))
        route_scaled = [(x * scale, y * scale) for x, y in route]
        color = styles.get("strokeColor", "#64748B")
        width = max(int(float(styles.get("strokeWidth", "1")) * scale), 1)
        draw.line(route_scaled, fill=color, width=width, joint="curve")
        if styles.get("endArrow", "block") != "none" and len(route_scaled) >= 2:
            arrowhead(
                route_scaled[-1],
                route_scaled[-2],
                color,
                filled=styles.get("endFill", "1") != "0",
            )
        if styles.get("startArrow", "none") != "none" and len(route_scaled) >= 2:
            arrowhead(
                route_scaled[0],
                route_scaled[1],
                color,
                filled=styles.get("startFill", "1") != "0",
            )
        value = plain(cell.get("value", ""))
        if value:
            label_x, label_y = route_midpoint(route_scaled)
            edge_labels.append(
                (
                    value.replace("\n", " "),
                    label_x,
                    label_y,
                    styles.get("fontColor", color),
                    int(float(styles.get("fontSize", "10"))),
                )
            )

    for cell in cells:
        if cell.get("vertex") == "1" and cell.get("id") not in background_ids:
            draw_vertex(cell)

    for value, x, y, color, size in edge_labels:
        font = ImageFont.truetype(bold, max(int(round(size * scale)), 7))
        bbox = draw.textbbox((0, 0), value, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        pad_x, pad_y = 4, 2
        draw.rounded_rectangle(
            (
                x - text_width / 2 - pad_x,
                y - text_height / 2 - pad_y,
                x + text_width / 2 + pad_x,
                y + text_height / 2 + pad_y,
            ),
            radius=3,
            fill="#FFFFFF",
            outline="#D5DAE0",
            width=1,
        )
        draw.text(
            (x, y - text_height / 2),
            value,
            font=font,
            fill=color,
            anchor="ma",
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst)
    print(dst)
    print(
        "clipped_cells=" + (",".join(clipped_cells) if clipped_cells else "none")
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
