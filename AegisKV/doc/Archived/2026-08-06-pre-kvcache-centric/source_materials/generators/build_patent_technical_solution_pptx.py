#!/usr/bin/env python3
"""Build the 5-slide AegisKV technical-solution section from the patent template."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


EMU_PER_INCH = 914400

RED = RGBColor(0xC0, 0x00, 0x00)
BLUE = RGBColor(0x33, 0x33, 0xFF)
DARK_BLUE = RGBColor(0x19, 0x3A, 0x5A)
MID_BLUE = RGBColor(0x2F, 0x75, 0xB5)
LIGHT_BLUE = RGBColor(0xE9, 0xF2, 0xFB)
PALE_BLUE = RGBColor(0xF5, 0xF9, 0xFD)
GREEN = RGBColor(0x31, 0x7A, 0x55)
LIGHT_GREEN = RGBColor(0xE9, 0xF5, 0xEE)
ORANGE = RGBColor(0xC6, 0x59, 0x11)
LIGHT_ORANGE = RGBColor(0xFC, 0xF0, 0xE6)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
BORDER = RGBColor(0xC8, 0xD3, 0xDF)
BLACK = RGBColor(0x20, 0x20, 0x20)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

CN_FONT = "微软雅黑"
LATIN_FONT = "Arial"
MATH_FONT = "Cambria Math"


def _in(value: float):
    return Inches(value)


def _style_paragraph(
    paragraph,
    *,
    size: float,
    color: RGBColor,
    bold: bool = False,
    font: str = CN_FONT,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    space_after: float = 2.5,
):
    paragraph.alignment = align
    paragraph.space_after = Pt(space_after)
    for run in paragraph.runs:
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color


def add_text(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    size: float = 16,
    color: RGBColor = BLACK,
    bold: bool = False,
    font: str = CN_FONT,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    margin: float = 0.04,
    space_after: float = 2.5,
):
    shape = slide.shapes.add_textbox(_in(x), _in(y), _in(w), _in(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = _in(margin)
    tf.margin_right = _in(margin)
    tf.margin_top = _in(margin)
    tf.margin_bottom = _in(margin)
    tf.vertical_anchor = valign
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        _style_paragraph(
            p,
            size=size,
            color=color,
            bold=bold,
            font=font,
            align=align,
            space_after=space_after,
        )
    return shape


def add_box(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: RGBColor = WHITE,
    line: RGBColor = BORDER,
    radius: bool = True,
):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, _in(x), _in(y), _in(w), _in(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.0)
    return shape


def add_box_text(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fill: RGBColor = WHITE,
    line: RGBColor = BORDER,
    size: float = 15,
    color: RGBColor = BLACK,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.CENTER,
    valign: MSO_ANCHOR = MSO_ANCHOR.MIDDLE,
    margin: float = 0.09,
    font: str = CN_FONT,
):
    shape = add_box(slide, x, y, w, h, fill=fill, line=line)
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = _in(margin)
    tf.margin_right = _in(margin)
    tf.margin_top = _in(margin)
    tf.margin_bottom = _in(margin)
    tf.vertical_anchor = valign
    lines = text.split("\n")
    for idx, line_text in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line_text
        _style_paragraph(
            p,
            size=size,
            color=color,
            bold=bold,
            font=font,
            align=align,
            space_after=2,
        )
    return shape


def add_arrow(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    direction: str = "right",
    color: RGBColor = MID_BLUE,
):
    shape_type = {
        "right": MSO_SHAPE.RIGHT_ARROW,
        "left": MSO_SHAPE.LEFT_ARROW,
        "up": MSO_SHAPE.UP_ARROW,
        "down": MSO_SHAPE.DOWN_ARROW,
    }[direction]
    shape = slide.shapes.add_shape(shape_type, _in(x), _in(y), _in(w), _in(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.color.rgb = color
    return shape


def add_title(slide, title: str, index: int):
    add_text(
        slide,
        0.55,
        0.14,
        12.1,
        0.52,
        title,
        size=27,
        color=RED,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
        margin=0,
    )
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, _in(0.56), _in(0.76), _in(12.18), _in(0.025)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    line.line.fill.background()
    add_text(
        slide,
        11.65,
        7.06,
        1.08,
        0.22,
        f"技术方案 {index}/5",
        size=8.5,
        color=GRAY,
        align=PP_ALIGN.RIGHT,
        margin=0,
    )


def add_label(
    slide,
    x: float,
    y: float,
    w: float,
    text: str,
    *,
    fill: RGBColor = MID_BLUE,
):
    return add_box_text(
        slide,
        x,
        y,
        w,
        0.34,
        text,
        fill=fill,
        line=fill,
        size=11,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0.02,
    )


def add_bottom_note(slide, text: str, *, fill: RGBColor = LIGHT_GRAY):
    add_box(slide, 0.60, 6.48, 12.10, 0.46, fill=fill, line=fill, radius=False)
    add_text(
        slide,
        0.77,
        6.56,
        11.76,
        0.27,
        text,
        size=11.5,
        color=DARK_BLUE,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        margin=0,
    )


def clear_placeholders(slide):
    for shape in list(slide.shapes):
        if shape.is_placeholder:
            slide.shapes._spTree.remove(shape._element)


def clear_all_shapes(slide):
    for shape in list(slide.shapes):
        slide.shapes._spTree.remove(shape._element)


def build_overview(slide):
    add_title(slide, "本Idea的技术方案（1）：总体架构与三层决策", 1)
    add_text(
        slide,
        0.72,
        0.91,
        11.9,
        0.34,
        "目标：在正常服务尾延迟约束下，优先保护对 workflow completion time 影响最大的已提交 KV。",
        size=15,
        color=DARK_BLUE,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )

    boxes = [
        (0.55, "运行时状态", "执行依赖图、queue/load\nrequest 与 KV 生命周期"),
        (3.72, "算法 1：Dynamic Criticality", "更新已观察关键路径\n估计当前 request 重要性"),
        (6.89, "算法 2：Placement", "比较恢复收益与备份代价\n选择 atom 与备份位置"),
        (10.06, "物理执行", "copy → ACK →\nsafe frontier 发布"),
    ]
    for idx, (x, heading, body) in enumerate(boxes):
        add_box(slide, x, 1.50, 2.70, 1.38, fill=PALE_BLUE, line=MID_BLUE)
        add_label(slide, x + 0.12, 1.62, 2.46, heading)
        add_text(
            slide,
            x + 0.14,
            2.08,
            2.42,
            0.62,
            body,
            size=13,
            color=BLACK,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
            margin=0.02,
        )
        if idx < len(boxes) - 1:
            add_arrow(slide, x + 2.75, 2.03, 0.35, 0.30)

    add_box(slide, 3.12, 3.28, 7.10, 0.96, fill=LIGHT_ORANGE, line=ORANGE)
    add_label(slide, 3.30, 3.42, 1.88, "算法 3：Alpha 控制", fill=ORANGE)
    add_text(
        slide,
        5.35,
        3.41,
        4.54,
        0.48,
        "慢时标调整 backup HBM 比例\n为算法 2 提供可用容量预算",
        size=13.5,
        color=BLACK,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        margin=0.01,
    )
    add_arrow(slide, 6.43, 2.92, 0.46, 0.30, direction="up", color=ORANGE)

    cards = [
        (
            0.72,
            "统一目标",
            "兼顾正常路径开销\n与故障后的 workflow WCT",
            LIGHT_BLUE,
            MID_BLUE,
        ),
        (
            4.55,
            "拓扑无关",
            "chain、branch、fan-out/fan-in\n或展开后的 iteration",
            LIGHT_GREEN,
            GREEN,
        ),
        (
            8.38,
            "三条恢复路径",
            "peer GPU、Host、recompute\n按最短有效路径择一",
            LIGHT_ORANGE,
            ORANGE,
        ),
    ]
    for x, heading, body, fill, line in cards:
        add_box(slide, x, 4.65, 3.45, 1.38, fill=fill, line=line)
        add_text(
            slide,
            x + 0.12,
            4.79,
            3.21,
            0.28,
            heading,
            size=15,
            color=line,
            bold=True,
            align=PP_ALIGN.CENTER,
            margin=0,
        )
        add_text(
            slide,
            x + 0.16,
            5.17,
            3.13,
            0.70,
            body,
            size=12.5,
            color=BLACK,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
            margin=0,
        )

    add_bottom_note(
        slide,
        "Criticality 只估计价值，Placement 决定放置，Alpha 决定预算；三层共享同一 workflow-level 目标。",
    )


def build_criticality(slide):
    add_title(slide, "本Idea的技术方案（2）：动态 DAG Criticality 估计与 KV 传播", 2)

    add_box(slide, 0.62, 1.02, 5.65, 4.98, fill=PALE_BLUE, line=MID_BLUE)
    add_label(slide, 0.82, 1.18, 2.45, "A. 已观察路径 + 剩余时间")
    add_text(
        slide,
        0.85,
        1.70,
        5.18,
        0.52,
        "关键路径计算与预测范围必须分开：",
        size=13.8,
        color=BLACK,
        margin=0,
    )
    add_box_text(
        slide,
        0.90,
        2.35,
        5.08,
        1.05,
        "已揭示 DAG（非预测）：用真实完成时间维护到当前 frontier 的最长已观察路径\n"
        "未完成节点（只预测耗时）：估计 running/ready call 的剩余执行与排队时间",
        fill=WHITE,
        line=BORDER,
        size=12.8,
        color=DARK_BLUE,
        bold=True,
    )
    add_text(
        slide,
        0.88,
        3.59,
        5.15,
        1.31,
        "• node/edge 揭示或 call 完成时，事件驱动地更新路径\n"
        "• 不预测完整未来 DAG；尚未出现的分支不纳入当前图\n"
        "• 结合路径位置、预计剩余时间和 queue/load 输出 Critical score\n"
        "• workflow 完成后用真实完整 DAG 校准在线估计与置信度",
        size=12.8,
        color=BLACK,
        margin=0,
        space_after=4,
    )
    add_box_text(
        slide,
        0.90,
        5.12,
        5.08,
        0.60,
        "参考：HEFT 的 DAG+估计代价；Decima 的事件驱动更新；Agentix/ATLAS 的 non-clairvoyant 动态 DAG。",
        fill=LIGHT_GREEN,
        line=GREEN,
        size=11.2,
        color=GREEN,
        bold=True,
    )

    add_arrow(slide, 6.38, 3.07, 0.48, 0.40)

    add_box(slide, 6.97, 1.02, 5.74, 4.98, fill=PALE_BLUE, line=MID_BLUE)
    add_label(slide, 7.17, 1.18, 2.30, "B. Request → KV atom")
    add_box_text(
        slide,
        7.26,
        1.70,
        5.14,
        0.72,
        "请求 metadata 携带：workflow/Agent 标识、Critical score、置信度与版本",
        fill=WHITE,
        line=BORDER,
        size=13.5,
        color=DARK_BLUE,
        bold=True,
    )
    add_text(
        slide,
        7.24,
        2.62,
        5.16,
        0.47,
        "radix extend / commit 时，将请求信息绑定到对应 KV atom：",
        size=13.5,
        color=BLACK,
        margin=0,
    )
    add_box_text(
        slide,
        7.50,
        3.06,
        4.62,
        0.64,
        "每个 atom 记录 owner、消费者列表、各消费者的 Critical score 与最新版本",
        fill=LIGHT_BLUE,
        line=MID_BLUE,
        size=13.2,
        color=DARK_BLUE,
        bold=True,
    )
    add_text(
        slide,
        7.23,
        3.91,
        5.21,
        1.50,
        "• 私有 atom：继承 owner request 的 score\n"
        "• shared prefix：保留 per-consumer score，不直接求和\n"
        "• 仅接受更新版本；atom eviction 时删除映射\n"
        "• 汇总后的 KV-level value signal 交给算法 2",
        size=13.5,
        color=BLACK,
        margin=0,
        space_after=5,
    )

    add_bottom_note(
        slide,
        "边界：在线得到的是“当前 Criticality 估计”，不是对最终完整关键路径的预知；Critical score 也不是故障概率。",
        fill=LIGHT_ORANGE,
    )


def build_placement(slide):
    add_title(slide, "本Idea的技术方案（3）：基于边际恢复收益的分层备份策略", 3)

    add_text(
        slide,
        0.72,
        0.91,
        11.90,
        0.35,
        "对每个已提交的 KV atom 比较可行位置；recompute 是无额外副本时的恢复路径，不是备份位置。",
        size=14,
        color=DARK_BLUE,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )

    candidates = [
        (0.65, "Peer GPU", "恢复快\n占用稀缺 HBM\n受 peer load/P2P 影响", LIGHT_BLUE, MID_BLUE),
        (3.55, "Host memory", "恢复较慢\n容量较大\n受 H2D/queue 影响", LIGHT_GREEN, GREEN),
        (6.45, "Remote tier（可选）", "容量更大\n恢复更慢\n受网络/存储影响", LIGHT_ORANGE, ORANGE),
        (9.35, "No extra copy", "零备份成本\n故障时 Host restore\n或 recompute", LIGHT_GRAY, GRAY),
    ]
    for x, heading, body, fill, line in candidates:
        add_box(slide, x, 1.40, 2.55, 1.35, fill=fill, line=line)
        add_text(
            slide,
            x + 0.10,
            1.54,
            2.35,
            0.26,
            heading,
            size=14,
            color=line,
            bold=True,
            align=PP_ALIGN.CENTER,
            margin=0,
        )
        add_text(
            slide,
            x + 0.12,
            1.88,
            2.31,
            0.70,
            body,
            size=11.5,
            color=BLACK,
            align=PP_ALIGN.CENTER,
            valign=MSO_ANCHOR.MIDDLE,
            margin=0,
        )

    add_box(slide, 0.65, 3.04, 7.45, 2.91, fill=PALE_BLUE, line=MID_BLUE)
    add_label(slide, 0.86, 3.20, 2.12, "收益—成本联合判断")
    add_box_text(
        slide,
        0.91,
        3.58,
        6.91,
        1.25,
        "Criticality：该 KV 丢失是否会拖慢整个 workflow\n"
        "Recovery benefit：从该位置恢复能节省多少等待、传输与重算时间\n"
        "Backup cost：复制耗时、容量占用以及对正常请求的可见干扰",
        fill=WHITE,
        line=BORDER,
        size=13.5,
        color=DARK_BLUE,
        bold=True,
    )
    add_box_text(
        slide,
        0.91,
        4.99,
        3.26,
        0.66,
        "优先保护：重要性高、恢复节省大、备份代价低",
        fill=LIGHT_GREEN,
        line=GREEN,
        size=12.8,
        color=GREEN,
        bold=True,
    )
    add_box_text(
        slide,
        4.53,
        4.99,
        3.29,
        0.66,
        "延后或不备份：价值低，或当前资源压力过高",
        fill=LIGHT_ORANGE,
        line=ORANGE,
        size=12.8,
        color=ORANGE,
        bold=True,
    )

    add_box(slide, 8.38, 3.04, 4.35, 2.91, fill=WHITE, line=RED)
    add_label(slide, 8.62, 3.20, 1.75, "约束与执行", fill=RED)
    add_text(
        slide,
        8.63,
        3.73,
        3.84,
        1.77,
        "1. 仅使用已 ACK 的 committed frontier\n"
        "2. owner 与 replica 跨 failure domain\n"
        "3. 每个 atom 至多选择一个额外位置\n"
        "4. 各位置不超过容量/带宽预算\n"
        "5. 按单位资源的综合收益排序并动态 admit/evict",
        size=12.4,
        color=BLACK,
        margin=0,
        space_after=4,
    )
    add_box_text(
        slide,
        8.70,
        5.38,
        3.70,
        0.39,
        "copy 完成并 ACK 后才发布 safe frontier",
        fill=LIGHT_ORANGE,
        line=ORANGE,
        size=11.2,
        color=ORANGE,
        bold=True,
    )

    add_bottom_note(
        slide,
        "shared prefix 的收益按 current-plan marginal WCT saving 计算，避免多个 atom 重复领取同一恢复收益。",
    )


def build_alpha(slide):
    add_title(slide, "本Idea的技术方案（4）：Alpha 的慢时标预算控制", 4)

    add_box(slide, 0.65, 1.10, 4.02, 4.89, fill=PALE_BLUE, line=MID_BLUE)
    add_label(slide, 0.88, 1.27, 1.77, "HBM 预算划分")
    add_text(
        slide,
        0.91,
        1.80,
        3.48,
        0.56,
        "Alpha 表示 GPU HBM 中为 backup 预留的比例",
        size=15,
        color=DARK_BLUE,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )
    total_x, total_y, total_w, total_h = 1.02, 2.58, 3.25, 0.96
    backup_w = total_w * 0.36
    inference_w = total_w - backup_w
    add_box_text(
        slide,
        total_x,
        total_y,
        backup_w,
        total_h,
        "Backup\npartition",
        fill=LIGHT_ORANGE,
        line=ORANGE,
        size=13,
        color=ORANGE,
        bold=True,
    )
    add_box_text(
        slide,
        total_x + backup_w,
        total_y,
        inference_w,
        total_h,
        "Inference\npartition",
        fill=LIGHT_BLUE,
        line=MID_BLUE,
        size=13,
        color=MID_BLUE,
        bold=True,
    )
    add_text(
        slide,
        0.93,
        3.88,
        3.46,
        1.45,
        "Alpha 过小：GPUReplica 覆盖不足\n"
        "Alpha 过大：压缩 active/prefix KV 空间\n"
        "因此 Alpha 不是越大越好，需要同时结算 normal-path cost 与 recovery saving。",
        size=13.1,
        color=BLACK,
        margin=0,
        space_after=5,
    )

    add_arrow(slide, 4.82, 3.12, 0.48, 0.38)

    add_box(slide, 5.39, 1.10, 7.34, 4.89, fill=WHITE, line=ORANGE)
    add_label(slide, 5.62, 1.27, 2.38, "候选比例搜索 + SLO gate", fill=ORANGE)
    add_text(
        slide,
        5.66,
        1.83,
        6.77,
        0.49,
        "只评估少量预设比例，避免在运行时求解复杂优化问题",
        size=15,
        color=DARK_BLUE,
        bold=True,
        align=PP_ALIGN.CENTER,
        margin=0,
    )

    labels = ["较低", "中低", "中等", "…", "较高"]
    for idx, label in enumerate(labels):
        x = 5.76 + idx * 1.31
        add_box_text(
            slide,
            x,
            2.47,
            1.05,
            0.55,
            label,
            fill=LIGHT_GRAY if label == "…" else PALE_BLUE,
            line=GRAY if label == "…" else MID_BLUE,
            size=12,
            color=BLACK,
            bold=True,
        )
        if idx < len(labels) - 1:
            add_arrow(slide, x + 1.08, 2.64, 0.18, 0.20, color=BORDER)

    add_box_text(
        slide,
        5.74,
        3.30,
        6.62,
        1.10,
        "对每个候选比例：先运行算法 2 得到可行的 KV 放置方案，\n"
        "再同时评估正常路径开销和故障后的 workflow 完成时间",
        fill=LIGHT_BLUE,
        line=MID_BLUE,
        size=13.5,
        color=DARK_BLUE,
        bold=True,
    )
    add_box_text(
        slide,
        5.74,
        4.61,
        6.62,
        0.83,
        "在满足正常服务 p99 SLO 的候选中选择整体代价最低者\n"
        "在线应用 rate limit、hysteresis 与 bounded correction",
        fill=LIGHT_GREEN,
        line=GREEN,
        size=13.2,
        color=GREEN,
        bold=True,
    )

    add_bottom_note(
        slide,
        "Alpha 在 control epoch 更新，不随单个请求抖动；复制和切换造成的开销也参与候选比较。",
        fill=LIGHT_ORANGE,
    )


def build_end_to_end(slide):
    add_title(slide, "本Idea的技术方案（5）：端到端执行闭环", 5)

    columns = [
        (
            0.62,
            "正常服务路径",
            MID_BLUE,
            LIGHT_BLUE,
            [
                ("1", "观测 workflow 执行状态与 queue/load"),
                ("2", "预测 Critical score，并随 request metadata 下传"),
                ("3", "radix commit 时建立 atom—consumer 映射"),
                ("4", "算法 2 选择候选；异步 copy、ACK 后发布"),
            ],
        ),
        (
            4.52,
            "故障恢复路径",
            RED,
            RGBColor(0xFB, 0xEB, 0xEB),
            [
                ("5", "检测 owner fail-stop 与失败 frontier"),
                ("6", "过滤未 ACK、版本过旧或同域 replica"),
                ("7", "在 peer GPU / Host / recompute 中择快"),
                ("8", "恢复 suffix 并继续 workflow 执行"),
            ],
        ),
        (
            8.42,
            "慢时标控制路径",
            ORANGE,
            LIGHT_ORANGE,
            [
                ("9", "聚合 normal-path p99 与 failure loss"),
                ("10", "对候选 Alpha 重算预算和 placement"),
                ("11", "经 SLO gate、hysteresis 后更新 Alpha"),
                ("12", "新预算仅作用于后续 admit/evict"),
            ],
        ),
    ]
    for x, heading, line, fill, steps in columns:
        add_box(slide, x, 1.12, 3.60, 4.73, fill=fill, line=line)
        add_label(slide, x + 0.20, 1.30, 3.20, heading, fill=line)
        y = 1.90
        for number, body in steps:
            circle = slide.shapes.add_shape(
                MSO_SHAPE.OVAL, _in(x + 0.22), _in(y), _in(0.43), _in(0.43)
            )
            circle.fill.solid()
            circle.fill.fore_color.rgb = line
            circle.line.color.rgb = line
            tf = circle.text_frame
            tf.clear()
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.text = number
            _style_paragraph(
                p,
                size=10.5 if len(number) > 1 else 12,
                color=WHITE,
                bold=True,
                align=PP_ALIGN.CENTER,
                space_after=0,
            )
            add_text(
                slide,
                x + 0.78,
                y - 0.01,
                2.54,
                0.50,
                body,
                size=12.3,
                color=BLACK,
                valign=MSO_ANCHOR.MIDDLE,
                margin=0,
            )
            y += 0.91

    add_arrow(slide, 4.26, 3.04, 0.22, 0.35)
    add_arrow(slide, 8.16, 3.04, 0.22, 0.35)

    add_box_text(
        slide,
        0.86,
        6.03,
        3.25,
        0.35,
        "不阻塞正常推理：copy 与 score 更新均可异步",
        fill=WHITE,
        line=MID_BLUE,
        size=10.8,
        color=MID_BLUE,
        bold=True,
    )
    add_box_text(
        slide,
        4.76,
        6.03,
        3.25,
        0.35,
        "只信任 committed safe frontier，避免恢复错误",
        fill=WHITE,
        line=RED,
        size=10.8,
        color=RED,
        bold=True,
    )
    add_box_text(
        slide,
        8.66,
        6.03,
        3.25,
        0.35,
        "控制变化有界，避免 HBM 分区频繁振荡",
        fill=WHITE,
        line=ORANGE,
        size=10.8,
        color=ORANGE,
        bold=True,
    )

    add_bottom_note(
        slide,
        "形成“状态估计 → 价值计算 → 预算内放置 → ACK 发布 → 故障择路 → 反馈调参”的完整可实施闭环。",
    )


def build(template: Path, output: Path):
    prs = Presentation(str(template))
    if len(prs.slides) < 3:
        raise ValueError("Template must contain the '本Idea的技术方案' guide slide.")

    template_index = 2
    template_slide = prs.slides[template_index]
    layout = template_slide.slide_layout
    if "技术方案" not in "\n".join(
        shape.text for shape in template_slide.shapes if hasattr(shape, "text")
    ):
        raise ValueError("Slide 3 is not the expected technical-solution guide page.")

    builders = [
        build_overview,
        build_criticality,
        build_placement,
        build_alpha,
        build_end_to_end,
    ]
    clear_all_shapes(template_slide)
    builders[0](template_slide)

    new_slide_ids = []
    for builder in builders[1:]:
        slide = prs.slides.add_slide(layout)
        clear_placeholders(slide)
        builder(slide)
        new_slide_ids.append(prs.slides._sldIdLst[-1])

    for slide_id in new_slide_ids:
        prs.slides._sldIdLst.remove(slide_id)
    for offset, slide_id in enumerate(new_slide_ids):
        prs.slides._sldIdLst.insert(template_index + 1 + offset, slide_id)

    prs.core_properties.title = "AegisKV 专利申请——技术方案"
    prs.core_properties.subject = (
        "Workflow-aware criticality prediction, tiered KV backup placement, "
        "and slow-timescale alpha control"
    )
    prs.core_properties.keywords = (
        "AegisKV, Critical Agent, KV cache, backup placement, fault recovery"
    )
    prs.core_properties.comments = (
        "Generated from AegisKV专利申请模板.pptx; the original template is not overwritten."
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    validate(output, expected_slides=len(prs.slides))


def validate(path: Path, expected_slides: int):
    with ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise ValueError(f"Corrupt PPTX member: {corrupt_member}")

    prs = Presentation(str(path))
    if len(prs.slides) != expected_slides:
        raise ValueError(
            f"Unexpected slide count: {len(prs.slides)} != {expected_slides}"
        )

    expected_titles = [
        "总体架构与三层决策",
        "动态 DAG Criticality 估计与 KV 传播",
        "基于边际恢复收益的分层备份策略",
        "Alpha 的慢时标预算控制",
        "端到端执行闭环",
    ]
    for offset, expected in enumerate(expected_titles, start=2):
        slide = prs.slides[offset]
        text = "\n".join(
            shape.text for shape in slide.shapes if hasattr(shape, "text")
        )
        if expected not in text:
            raise ValueError(f"Missing title on slide {offset + 1}: {expected}")
        if "撰写指引" in text:
            raise ValueError(f"Guide text remains on slide {offset + 1}")

        for shape in slide.shapes:
            if (
                shape.left < 0
                or shape.top < 0
                or shape.left + shape.width > prs.slide_width
                or shape.top + shape.height > prs.slide_height
            ):
                raise ValueError(
                    f"Out-of-bounds shape on slide {offset + 1}: {shape.name}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("benchmark/AegisKV/AegisKV专利申请模板.pptx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/AegisKV/AegisKV专利申请_技术方案.pptx"),
    )
    args = parser.parse_args()
    build(args.template.resolve(), args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
