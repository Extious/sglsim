#!/usr/bin/env python3
"""Build the AegisKV Chinese patent disclosure from the supplied Word template.

The generated document describes the current agent-aware fault-tolerance design and
avoids claiming unmeasured performance results.

Rebuild from the repository root with:

    UV_CACHE_DIR=/tmp/uv-cache uv run --with python-docx==1.2.0 \
      python AegisKV/patent/scripts/build_patent_disclosure.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import re
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


SCRIPT_DIR = Path(__file__).resolve().parent
PATENT_DIR = SCRIPT_DIR.parent
DEFAULT_TEMPLATE = PATENT_DIR / "template" / "AegisKV交底书模板.docx"
DEFAULT_OUTPUT = PATENT_DIR / "AegisKV专利交底书-严格模板版.docx"
REAL_CC_FIGURE = PATENT_DIR.parent / "figures" / "real-cc-parallel-subagent-duration.png"
MOTIVATION_RECOVERY_FIGURE = PATENT_DIR.parent / "figures" / "motivation-recovery-scaling.png"
MOTIVATION_POSITION_FIGURE = PATENT_DIR.parent / "figures" / "motivation-failure-position.png"
PRELIMINARY_FIGURE = PATENT_DIR.parent / "figures" / "preliminary-wct-kv-reuse.png"
LATEST_FIGURE_PREFIX = "latest-figure"
INVENTION_TITLE = (
    "AegisKV：一种面向智能体工作流的关键性感知 KV 备份与资源自适应方法、"
    "系统、设备及存储介质"
)

BODY_FONT_CN = "宋体"
HEADING_FONT_CN = "黑体"
BODY_FONT_LATIN = "Times New Roman"
PLACEHOLDER_RE = re.compile(r"(【待填写(?:：[^】]+)?】)")
MANUAL_HEADING_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\s+")


def set_run_font(run, *, size: float = 10.5, bold: bool | None = None,
                 cn_font: str = BODY_FONT_CN, color: RGBColor | None = None):
    run.font.name = BODY_FONT_LATIN
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cn_font)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    return run


def add_marked_text(paragraph, text: str, *, bold: bool = False,
                    size: float = 10.5, cn_font: str = BODY_FONT_CN):
    """Add text while rendering every unknown-field marker conspicuously."""
    for part in PLACEHOLDER_RE.split(text):
        if not part:
            continue
        run = paragraph.add_run(part)
        is_placeholder = bool(PLACEHOLDER_RE.fullmatch(part))
        set_run_font(
            run,
            size=size,
            bold=(True if is_placeholder else bold),
            cn_font=cn_font,
            color=(RGBColor(0xC0, 0x00, 0x00) if is_placeholder else None),
        )
        if is_placeholder:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return paragraph


def set_paragraph_format(paragraph, *, first_line: bool = True,
                         after: float = 4.0, before: float = 0.0,
                         line_spacing: float = 1.25):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line_spacing
    if first_line:
        fmt.first_line_indent = Pt(21)


def add_body(doc, text: str, *, bold_prefix: str | None = None,
             first_line: bool = True, after: float = 4.0):
    paragraph = doc.add_paragraph()
    set_paragraph_format(paragraph, first_line=first_line, after=after)
    paragraph.paragraph_format.keep_together = True
    if bold_prefix and text.startswith(bold_prefix):
        add_marked_text(paragraph, bold_prefix, bold=True)
        add_marked_text(paragraph, text[len(bold_prefix):])
    else:
        add_marked_text(paragraph, text)
    return paragraph


def add_bullet(doc, text: str, *, level: int = 0):
    paragraph = doc.add_paragraph()
    set_paragraph_format(paragraph, first_line=False, after=2.5)
    paragraph.paragraph_format.left_indent = Cm(0.65 + 0.55 * level)
    paragraph.paragraph_format.first_line_indent = Cm(-0.45)
    add_marked_text(paragraph, "• " + text)
    return paragraph


def add_step(doc, number: int | str, text: str):
    paragraph = doc.add_paragraph()
    set_paragraph_format(paragraph, first_line=False, after=3.0)
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.left_indent = Cm(0.75)
    paragraph.paragraph_format.first_line_indent = Cm(-0.75)
    add_marked_text(paragraph, f"S{number}. ", bold=True)
    add_marked_text(paragraph, text)
    return paragraph


def add_heading(
    doc,
    text: str,
    level: int = 1,
    *,
    numbered: bool = True,
    template_num_level: int | None = None,
):
    if numbered:
        # Heading styles in the Huawei template already carry multilevel
        # numbering.  Strip legacy literal prefixes such as ``4.2`` so Word
        # cannot display both the automatic and manual numbers.
        text = MANUAL_HEADING_NUMBER_RE.sub("", text)
    paragraph = doc.add_heading(level=level)
    if not numbered or template_num_level is not None:
        p_pr = paragraph._p.get_or_add_pPr()
        num_pr = OxmlElement("w:numPr")
        ilvl = OxmlElement("w:ilvl")
        ilvl.set(
            qn("w:val"),
            str(0 if template_num_level is None else template_num_level),
        )
        num_id = OxmlElement("w:numId")
        num_id.set(qn("w:val"), "0" if not numbered else "43")
        num_pr.append(ilvl)
        num_pr.append(num_id)
        p_pr.insert(0, num_pr)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    paragraph.paragraph_format.space_after = Pt(4)
    add_marked_text(
        paragraph,
        text,
        bold=True,
        size={1: 16, 2: 12, 3: 12}.get(level, 11),
        cn_font=HEADING_FONT_CN,
    )
    return paragraph


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_row_cant_split(row):
    """Keep a table row intact when Word lays out the document."""
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    tr_pr.append(cant_split)


def set_cell_margins(cell, top=70, start=90, bottom=70, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start),
                          ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_text(cell, text: str, *, bold: bool = False,
                  center: bool = False, size: float = 9.5):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    )
    set_paragraph_format(paragraph, first_line=False, after=0, line_spacing=1.05)
    add_marked_text(paragraph, text, bold=bold, size=size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)
    return cell


def add_table(doc, headers: list[str], rows: list[list[str]],
              *, widths_cm: list[float] | None = None,
              header_center: bool = True, font_size: float = 9.0,
              style_name: str = "Table Grid"):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = style_name
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header = table.rows[0]
    set_repeat_table_header(header)
    set_row_cant_split(header)
    for i, text in enumerate(headers):
        set_cell_text(header.cells[i], text, bold=True,
                      center=header_center, size=font_size)
        if widths_cm:
            header.cells[i].width = Cm(widths_cm[i])
    for row_data in rows:
        row = table.add_row()
        set_row_cant_split(row)
        for i, text in enumerate(row_data):
            set_cell_text(row.cells[i], text, center=False, size=font_size)
            if widths_cm:
                row.cells[i].width = Cm(widths_cm[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_caption(doc, text: str):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_format(paragraph, first_line=False, after=5)
    add_marked_text(paragraph, text, bold=True, size=9.5)
    return paragraph


def maybe_add_external_figure(doc, number: int, caption: str,
                              *, width_cm: float = 15.5):
    """Insert a separately maintained patent figure when its PNG exists.

    The editable source is kept in ``patent/figures`` by the parent workflow.
    Figures are required deliverables.  A formal build fails rather than
    silently producing a disclosure with a missing patent drawing.
    """
    image_path = (
        PATENT_DIR / "figures" / f"{LATEST_FIGURE_PREFIX}-{number}.png"
    )
    if not image_path.exists():
        raise FileNotFoundError(f"Required patent figure not found: {image_path}")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(image_path), width=Cm(width_cm))
    add_caption(doc, f"附图 {number}  {caption}")
    return True


def add_evidence_figure(doc, image_path: Path, caption: str,
                        *, width_cm: float = 15.5):
    """Insert a named evidence plot with a single Word caption."""
    if not image_path.exists():
        raise FileNotFoundError(f"Required evidence figure not found: {image_path}")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(image_path), width=Cm(width_cm))
    add_caption(doc, caption)
    return True


def add_page_number(paragraph):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, separate, end))
    set_run_font(run, size=9)


def clear_template_body(doc):
    body = doc._element.body
    section_properties = body.sectPr
    for child in list(body):
        if child is not section_properties:
            body.remove(child)


def configure_document(doc):
    # Preserve the supplied disclosure template's page geometry, styles,
    # numbering definitions, table styles, theme, and blank header/footer.
    # Visible document content follows the template body structure exactly.

    props = doc.core_properties
    props.title = INVENTION_TITLE
    props.subject = (
        "基于因果 Slack、分层惰性重定优先级和独立 HBM/DRAM 准入的 "
        "LLM 推理容错"
    )
    props.author = "AegisKV 项目组（发明人信息待确认）"
    props.last_modified_by = "AegisKV 项目组"
    props.revision = 1
    props.created = datetime(2026, 8, 16, tzinfo=timezone.utc)
    props.modified = datetime.now(timezone.utc)
    props.keywords = (
        "agentic workflow; causal Slack; lazy reprioritization; tier admission; "
        "KV cache; GuardedAIMD; GPU failure"
    )


def sanitize_package_metadata(document_path: Path):
    """Remove stale template metadata/links and request Word field refresh.

    ``python-docx`` preserves extended properties such as the template's page
    count, word count, company, proprietary custom properties, and the original
    author's local ``file:///C:/Users/...`` template path.  Rewriting the small
    OOXML parts avoids shipping misleading or identifying template residue
    without touching document content or styles.
    """
    app_name = "docProps/app.xml"
    custom_props_name = "docProps/custom.xml"
    settings_name = "word/settings.xml"
    settings_rels_name = "word/_rels/settings.xml.rels"
    app_ns = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    custom_props_ns = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
    package_rels_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ET.register_namespace("", app_ns)
    ET.register_namespace("", custom_props_ns)
    ET.register_namespace("", package_rels_ns)
    ET.register_namespace("w", w_ns)

    with zipfile.ZipFile(document_path, "r") as source:
        app_root = ET.fromstring(source.read(app_name))
        for local_name in ("Pages", "Words", "Characters", "CharactersWithSpaces",
                           "Lines", "Paragraphs", "TotalTime"):
            node = app_root.find(f"{{{app_ns}}}{local_name}")
            if node is not None:
                node.text = "0"
        template_name = app_root.find(f"{{{app_ns}}}Template")
        if template_name is not None:
            template_name.text = ""
        company = app_root.find(f"{{{app_ns}}}Company")
        if company is not None:
            company.text = ""

        settings_root = ET.fromstring(source.read(settings_name))
        attached_template = settings_root.find(f"{{{w_ns}}}attachedTemplate")
        if attached_template is not None:
            settings_root.remove(attached_template)
        update_fields = settings_root.find(f"{{{w_ns}}}updateFields")
        if update_fields is None:
            update_fields = ET.SubElement(settings_root, f"{{{w_ns}}}updateFields")
        update_fields.set(f"{{{w_ns}}}val", "true")

        settings_rels_root = ET.fromstring(source.read(settings_rels_name))
        for relationship in list(settings_rels_root):
            if relationship.get("Type", "").endswith("/attachedTemplate"):
                settings_rels_root.remove(relationship)

        custom_props_root = ET.fromstring(source.read(custom_props_name))
        for property_node in list(custom_props_root):
            custom_props_root.remove(property_node)

        with tempfile.NamedTemporaryFile(
            prefix=document_path.stem + ".", suffix=".docx",
            dir=document_path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED) as target:
                for item in source.infolist():
                    if item.filename == app_name:
                        data = ET.tostring(app_root, encoding="utf-8", xml_declaration=True)
                    elif item.filename == settings_name:
                        data = ET.tostring(settings_root, encoding="utf-8", xml_declaration=True)
                    elif item.filename == settings_rels_name:
                        data = ET.tostring(
                            settings_rels_root, encoding="utf-8", xml_declaration=True
                        )
                    elif item.filename == custom_props_name:
                        data = ET.tostring(
                            custom_props_root, encoding="utf-8", xml_declaration=True
                        )
                    else:
                        data = source.read(item.filename)
                    target.writestr(item, data)
            temporary_path.replace(document_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def build_cover(doc):
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(30)
    title.paragraph_format.space_after = Pt(18)
    add_marked_text(title, "专利技术交底书", bold=True, size=24,
                    cn_font=HEADING_FONT_CN)

    name = doc.add_paragraph()
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name.paragraph_format.space_after = Pt(24)
    add_marked_text(
        name,
        INVENTION_TITLE,
        bold=True,
        size=16,
        cn_font=HEADING_FONT_CN,
    )

    add_table(
        doc,
        ["项目", "内容"],
        [
            ["发明名称", INVENTION_TITLE],
            ["技术领域", "智能体大模型在线推理、GPU 容错、KVCache 分层存储与恢复"],
            ["发明人", "【待填写：发明人姓名及排序】"],
            ["技术联系人", "【待填写：姓名、电话、邮箱】"],
            ["高校主体", "【待填写：高校主体】"],
            ["华为合作主体", "【待填写：华为合作主体】"],
            ["知识产权归属/申请主体", "【待填写：权属、申请主体及合作约定】"],
            ["项目编号", "【待填写：华为项目编号/合同编号】"],
            ["最早构思日期", "【待填写：最早构思日期及证据位置】"],
            ["首次书面记录/内部披露", "【待填写：首次书面记录或内部披露日期】"],
            ["首次对外披露/计划公开", "【待填写：首次对外披露或计划公开日期】"],
            ["草案信息", "当前设计稿；正式交底/申请日期：【待填写：正式交底/申请日期】"],
        ],
        widths_cm=[4.0, 12.0],
        font_size=10,
    )

    warning = doc.add_table(rows=1, cols=1)
    warning.style = "Table Grid"
    warning.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_row_cant_split(warning.rows[0])
    set_cell_text(
        warning.cell(0, 0),
        "重要声明：本交底书围绕面向智能体工作流的大模型推理容错技术展开。建议保护对象是因果 Slack 跨层传递、请求级与 chunk 级优先级分解、双层惰性 victim 索引、HBM-first 写回式保护缓存、三态驻留、压力安全的 HBM 配额控制及故障恢复执行所构成的完整技术过程，而不是关键路径、Landlord、AIMD 或某个数学公式本身。新颖性、创造性和权利要求层级须由专利代理人结合正式专利检索复核。",
        bold=True,
        center=False,
        size=10,
    )

    doc.add_page_break()
    add_heading(doc, "文档结构", 1, numbered=False)
    for item in (
        "第 2 章：缩略语和关键术语",
        "第 3 章：相关技术背景以及最接近的现有技术",
        "第 4 章：本发明技术方案的详细阐述",
        "第 5 章：发散思维与等同替换",
        "第 6 章：在先公开的现有技术文献信息",
        "附录 A：专利交底书 Checklist",
    ):
        add_bullet(doc, item)


def build_section_2(doc):
    add_heading(doc, "缩略语和关键术语", 1)
    add_heading(doc, "2.1  缩略语", 2)
    add_table(
        doc,
        ["中文表述/中文术语", "完整英文表述", "英文缩略语"],
        [
            ["键值缓存", "Key-Value Cache", "KVCache / KV cache"],
            ["大语言模型", "Large Language Model", "LLM"],
            ["图形处理器", "Graphics Processing Unit", "GPU"],
            ["高带宽存储器", "High Bandwidth Memory", "HBM"],
            ["动态随机存取存储器", "Dynamic Random-Access Memory", "DRAM"],
            ["固态硬盘", "Solid-State Drive", "SSD"],
            ["数据并行", "Data Parallelism", "DP"],
            ["流水线并行", "Pipeline Parallelism", "PP"],
            ["确认消息", "Acknowledgment", "ACK"],
            ["工作流完成时间", "Workflow Completion Time", "WCT"],
            ["服务级目标", "Service-Level Objective", "SLO"],
            ["校验和", "Checksum", "可采用 CRC、哈希值或等价校验信息"],
        ],
        widths_cm=[4.5, 7.5, 4.0],
    )

    add_heading(doc, "2.2  关键术语", 2)
    definitions = [
        ("KVCache 分块（KV chunk）", "按连续 token 区间划分的逻辑 KVCache 保护单元。对于 PP 部署，同一逻辑分块在每个必要 PP stage 上分别具有与该 stage 所拥有模型层对应的 KV 张量分片（stage shard）。分块边界可固定，也可按传输大小、阶段进度或资源状态动态确定。"),
        ("stage shard 放置", "以（request version，logical chunk，PP stage）为索引的一份实际 KV 张量分片放置。不同 required stages 的 shard 可以分别位于对端 HBM、共享 DRAM 或持久层，不要求同一逻辑 chunk 的全部 shards 位于同一层级。"),
        ("请求版本（request version）", "标识一次逻辑推理请求当前可接受状态的单调版本。工具调用返回、会话历史变化、请求重启或显式取消均可产生新版本；旧版本的迟到传输和 ACK 不得推进新版本的恢复前沿。"),
        ("内容世代（content epoch）", "与 route/placement version 分离的不可变 KV 内容身份，由 request version、chunk/token range、模型版本、KV-layout hash 等共同确定。路由或备份 DP 改变时，仍存活的 DRAM/SSD 对象只有在内容世代和 hash 复核一致后才可重绑定到新 manifest，避免把旧字节伪装成新路由下生成的数据。"),
        ("放置版本（placement version）", "标识某一 KV 分块当前目标层级、目标位置、租约和校验约束的版本。Master 只有在确认版本一致时才接收 ACK。"),
        ("路由版本（route version）", "标识一次请求当前执行 DP、准入时冻结的唯一备份/恢复 DP b(r) 以及失效隔离 epoch 的版本，用于阻止故障前的旧路由继续提交状态；执行 DP 故障时只激活已冻结的 b(r)，不临时重选恢复目标。"),
        ("预算版本（budget version）", "标识 Master 向一个 DP 的全部 PP stages 原子发布的版本化 HBM admission/reservation envelope。新版本在必要 stages 全部确认之前不生效。"),
        ("版本化 KV 分块保护对象", "由 workflow/request 标识、请求版本、逻辑分块/token 区间、PP stage、模型与 KV 布局签名、放置版本、目标层级、目标位置、租约、校验信息和状态组成的可核验 shard 记录，并由 Master 聚合形成逻辑 chunk 的提交状态。它把普通缓存命中与可用于故障恢复的提交状态区分开。"),
        ("同 stage 跨 DP 传输", "源 DP 的第 p 个 PP stage 仅把其 stage-local KV 张量发送到另一 DP 的第 p 个 PP stage 或与该 stage 对应的共享存储对象，不把第 p 个 stage 的张量误放到第 q 个 stage。"),
        ("全 stage 确认（all-stage ACK）", "Master 针对一个请求版本和分块版本收齐所有必要 PP stages 的传输完成、版本匹配与校验通过确认。缺少任一必要 stage 的 ACK 时，该分块只能处于待提交状态。"),
        ("连续安全前沿（safe frontier）", "从请求起点开始、能够由同一有效恢复清单覆盖的最大连续 token 前缀。它是全部必要 PP stages 的共同连续前沿，而不是任一 stage 的最大值，也不是缓存中存在若干离散分块的数量。"),
        ("复合恢复清单（recovery manifest）", "Master 为恢复请求生成的版本化清单。对每个已提交逻辑 chunk，清单按 PP stage 分别指定一个有效的对端 HBM、共享 DRAM或持久层放置，并携带校验、租约和依赖顺序；同一逻辑 chunk 的不同 stage shards 可以位于不同层级。对未形成完整承诺的逻辑 chunk 及其后缀，清单给出 Recompute。"),
        ("N/Recompute 结果", "表示某一逻辑 chunk 没有为全部 required stages 建立完整、有效且连续的 HBM/DRAM/持久层放置集合，因此该逻辑 chunk 不得提交，safe frontier 不得越过它。N/Recompute 不是某个 stage shard 的物理存储层级；已产生但未组成完整承诺的局部副本只能作为 best-effort cache。"),
        ("机会式 HBM 比例 alpha", "按 DP 统一发布、允许用于保护副本的版本化 admission 上限。同一 DP 的全部 PP stages 使用相同 alpha 值，但分别按本 stage 的物理 HBM 容量和安全余量校验可用字节数。alpha 不是启动时切死并必须占满的物理分区；实际使用量还受正常推理负载和 slowdown guard 限制。发布较大的 alpha 同时表示系统承诺维持相应保护方案的可行 headroom，因而产生保护承诺的容量机会成本。"),
        ("保护承诺机会成本（reservation opportunity cost）", "为保证已提交或候选 peer-HBM 保护在 budget version 有效期内可被接纳、维持或恢复而保留可行 headroom 的代价。实现上可通过压缩正常请求的可准入容量、维护可回收安全余量或计入等价 shadow price 体现；即使 alpha 尚未被实际副本占满，也不能把提高上限视为零成本。"),
        ("统一控制服务（Master）", "维护请求、路由、放置和预算版本，聚合全部必要 stage 的 ACK，提交 logical chunk，维护连续 safe frontier 和恢复 manifest。Master 只承载元数据与控制决定，不转发大体积 KV tensor。"),
        ("同 stage 连接器（stage-aligned Connector）", "部署在每个（DP，PP stage）上的数据面组件。源 DP 的第 p 个 stage 通过该组件把本 stage 的 KV tensor 直接写入备份/恢复 DP 的第 p 个 stage，或写入清单中与第 p 个 stage 对应的共享存储对象。"),
        ("备份/恢复 DP", "Router 针对一个 request version 固定的单一完整健康 DP，记为 b(r)。该 DP 应与模型/KV layout 兼容、位于不同 GPU failure domain，并通过逐 stage peer-HBM headroom 和 retry service-capacity 检查；不存在合格目标时显式记录 degraded contract。peer-HBM shard 只写入 b(r) 的同编号 PP stage；执行 DP 失效后整请求重试到 b(r)，不在恢复时拼接多个不完整 DP。"),
        ("正常路径减速约束（normal-path slowdown guard）", "对备份写入、恢复读取、带宽和 HBM 占用施加的硬约束，用于限制保护流量对未受故障请求的排队、计算和传输干扰。该约束是保护资源准入条件，不以单独的公平性公式替代。"),
        ("因果 Slack（causal Slack）", "由工作流处理器根据当前已经揭示的 DAG、节点状态和剩余时长预测计算的请求延迟容忍时间。Slack 表示额外恢复延迟中可以被并行路径隐藏的部分，是本发明评价 failure-induced workflow WCT 损失和多层保护价值的直接输入；其预测版本和不确定性随请求下传。"),
        ("整请求跨 DP 重试", "执行 DP e(r) 的任一必要 GPU/PP stage 故障后，不拼接故障 stage 与其他 DP 的局部执行，而是把该逻辑请求路由到准入时冻结的唯一完整健康 b(r)；b(r) 恢复已提交连续前缀并仅重算其后的后缀，恢复时不重新竞选目标。"),
        ("GPU 故障", "本交底书优选实施例所处理的唯一故障场景，包括 GPU 失联、不可恢复设备错误或由运行时判定该 GPU 无法继续承载请求。一次仅有一个 GPU 不可用；若故障位于执行 DP，则该 DP 的对应 HBM 不可用并触发整请求恢复；若故障仅位于 b(r)，执行请求可继续，但相应 peer-HBM 承诺必须失效并重建。共享 DRAM 和持久层按部署配置继续可用。"),
    ]
    for term, definition in definitions:
        add_body(doc, f"{term}：{definition}", bold_prefix=f"{term}：")


def build_section_3(doc):
    add_heading(doc, "相关技术背景以及最接近的现有技术", 1)
    add_heading(doc, "3.1  背景技术", 2)
    add_body(doc, "大语言模型在线推理会在每个模型层为已处理 token 生成 Key/Value 状态。后续 decode 可直接读取这些 KVCache，而不必重复执行整个前缀。长上下文和多轮智能体工作流使单个请求已积累的 KVCache 持续增长；一旦执行 GPU 失效且 KVCache 仅位于该 GPU，本次请求通常需要在健康副本上重新排队并重算前缀。")
    add_body(doc, "生产级推理服务通常同时使用数据并行和流水线并行。一个 DP 副本可由多个 PP stages 组成，各 stage 只持有其模型层对应的 KV 张量。因此，“某些字节已经复制”并不等于“请求能够恢复”：只有目标副本的所有必要 stages 都具有版本一致、连续且布局兼容的 KV 状态，系统才能从相同 token 边界继续执行。")
    add_body(doc, "GPU HBM、共享 DRAM 和 SSD/持久存储在恢复速度、容量、传输开销和对正常推理的影响上不同。固定保留大量 HBM 会挤压正常 batch 和 KV 容量；完全不保护又会在 GPU 故障时产生较长的前缀重算。由此需要一个能够在运行时建立、验证、撤销和恢复 KVCache 承诺的多层协议，而不只是一个离线优先级公式。")
    add_body(doc, "在 agentic 工作流中，请求的重要程度会随已揭示 DAG、tool result 和剩余路径变化。位于关键路径或 Slack 较小的请求一旦重算，其延迟更容易传播到整个 workflow；Slack 较大的请求则可能隐藏同等恢复开销。因此，仅按 KV 大小、访问热度或到达顺序写回无法准确刻画故障恢复价值。")
    add_body(doc, "本发明面向 agentic workflow 的推理故障恢复：模型权重和计算进程可以重新加载或重新路由，而持续增长的 KVCache 是 GPU failure 后最昂贵的动态状态。系统并非无条件保存全部 KV，而是根据 causal Slack 和恢复代价选择最值得形成恢复承诺的分块，并限制保护动作对健康 workflow 的正常路径干扰。")

    add_heading(doc, "3.2  与本发明相关的现有技术", 2)
    add_heading(doc, "3.2.1  现有技术的技术方案", 3)
    prior_rows = [
        ["Mooncake（FAST 2025）", "使用 GPU HBM、CPU DRAM、SSD 和网络资源，提供分离式 KV 存储、全局缓存调度与直接传输。", "说明多层 KVCache 与全局元数据具有工程可行性。需重点核对其缓存对象、传输完成语义与本发明版本化恢复承诺之间的边界。"],
        ["IMPRESS（FAST 2025）", "以重要 token 识别、KV 重排和缓存替换管理 GPU/CPU/磁盘上的多层 prefix KV，目标是降低 TTFT。", "与本发明同样涉及多层 KV 存储；需核对其是否涉及故障恢复、required-stage ACK、共同连续前沿和版本化恢复清单。"],
        ["DejaVu（ICML 2024）", "提供 KV cache streaming 原语、状态复制和故障处理，使 worker 故障后能够重建 KV 状态。", "是最接近的 KV 流式容错论文之一。其 checkpoint/ACK、复制目标和恢复边界须与本发明的 stage-wise 多层清单、all-stage 提交及预算版本逐项比对。"],
        ["FailSafe / RaidServe（arXiv 2025；MLSys 2026）", "面向 TP worker 故障，公开 cyclic KV placement、混合 attention、负载感知路由、主动 KV 备份和按需权重恢复。", "与本发明均涉及主动 KV 保护和故障后路由；需重点核对并行拓扑、保护对象、提交条件、恢复前沿及备份资源控制。"],
        ["GhostServe（MLSys 2026）", "对流式 KV cache 计算纠删码 parity shard 并存入 host memory，设备故障后重建丢失 KV。", "与本发明均避免全量重算；需核对编码对象、checkpoint 一致性、PP-stage 完整性以及 host parity 与复合多层 manifest 的关系。"],
        ["LUMEN（arXiv 2026）", "把恢复协调拆为故障前 checkpoint placement、故障时 interrupted-request dispatch 和模型重载期间的 capacity restoration。", "与本发明均包含放置和重路由；需核对其 checkpoint holder、负载感知派发与本发明固定 b(r)、all-stage 连续提交协议的边界。"],
        ["Concordia（arXiv 2026）", "在持久 GPU kernel 中对 KV、scheduler 与通信等设备驻留执行状态进行增量 checkpoint 和恢复。", "与本发明均面向长时间 agent 状态；需核对其低层 checkpoint commit 与本发明 logical chunk、stage-wise 多层 manifest 和统一恢复前沿的边界。"],
        ["SYMPHONY（NSDI 2026）", "利用 advisory request、优先级与 GPU/分离式内存协作进行多层 KV 管理。", "与本发明均使用请求信号和多层资源；需核对其 cache availability 与本发明 GPU 故障恢复承诺、版本和 all-stage ACK 的区别。"],
        ["vLLM / PagedAttention（SOSP 2023）", "通过分页式 KVCache 内存管理减少碎片并支持高效请求调度。", "提供底层 KV block 管理基础；需核对分页对象是否具备本发明所要求的恢复版本、租约、stage ACK 与提交状态。"],
        ["SGLang（NeurIPS 2024）", "通过结构化语言程序运行时、RadixAttention/前缀复用等机制提高复杂 LLM 程序执行效率。", "提供 agentic/结构化程序与前缀缓存基础；需核对其缓存生命周期与 GPU 故障下跨 DP 恢复语义。"],
    ]
    add_table(
        doc,
        ["公开技术", "主要公开机制", "与本发明的关系（非法律结论）"],
        prior_rows,
        widths_cm=[3.2, 6.4, 6.4],
        font_size=8.7,
    )
    add_body(doc, "注：以上比较仅基于已查阅的公开论文，不替代专利数据库的新颖性/创造性检索，也不构成“未被任何专利公开”的法律结论。正式申请前应由专利代理人针对论文的引用文献、作者/机构专利族及同族申请继续检索。", bold_prefix="注：")
    add_body(doc, "相关标准/提案状态：截至本交底书设计基线，项目组尚未完成对正式标准、行业提案和开源协议规范的系统检索，也不据此声称不存在相关规范。专利代理人应把 CUDA/CXL、分离式 KV 存储、推理服务容错与缓存一致性相关规范纳入补充检索，并记录版本、发布日期和与本方案的逐项关系。", bold_prefix="相关标准/提案状态：")

    add_heading(doc, "3.2.2  现有技术的缺点", 3)
    shortcomings = [
        "现有分页 KV、分层缓存和 write-through/write-back 路径主要面向容量扩展、前缀复用、TTFT 或吞吐，通常不根据已揭示 workflow DAG 的 causal Slack 评价 GPU failure 后的 WCT 损失。",
        "普通 cache hit、热点副本或传输完成事件通常没有同时给出“请求版本一致、所有必要 PP stages 完成、前缀连续”这三个可恢复条件，故缓存存在性可能被误当作恢复承诺。",
        "PP stages 独立复制时，某 stage 可能领先于其他 stage。若直接采用最大已复制 token 数，恢复后会得到跨 stage 不一致的 KV 状态。",
        "对端 HBM、共享 DRAM、SSD 和重算往往由不同模块记录，故障时缺少一个可冻结、可查询、可校验的复合恢复清单。",
        "把 KV tensor 经由中心控制服务转发会形成数据带宽瓶颈；完全去中心化又难以建立一致的提交前沿和版本隔离。",
        "静态划分 HBM 保护区会在高负载时挤压正常推理；仅按单 stage 空闲量调整又可能让一个 DP 内的多个 PP stages 使用不一致的容量版本。",
        "仅有工作流排序仍不足以完成可靠恢复；系统还必须处理分块持续增长、租约过期、前缀空洞、迟到 ACK、版本隔离和故障后的确定性恢复。",
    ]
    for item in shortcomings:
        add_bullet(doc, item)


def build_section_4_problem_and_overview(doc):
    add_heading(doc, "4  本发明技术方案的详细阐述", 1)
    add_heading(doc, "4.1  本发明所要解决的技术问题", 2)
    problems = [
        "在不把工作流预测作为恢复正确性前提的条件下，如何可选地根据当前已揭示 DAG 计算 causal Slack，并把延迟传播敏感性作为保护价值信号；",
        "如何把持续增长的 KVCache 划分为可独立传输、可版本化校验且能推进恢复边界的分块保护对象；",
        "如何保证同一请求分块在全部必要 PP stages 上版本一致地完成，并只提交共同的连续前缀；",
        "如何统一描述对端 HBM、共享 DRAM、持久层三类物理恢复来源与 logical Recompute 结果，使恢复端能够按确定清单执行；",
        "如何让 Master 统一管理版本、租约、ACK 和恢复边界，同时避免 KV tensor 经过 Master 形成带宽瓶颈；",
        "如何按 DP 发布版本化 HBM admission/reservation envelope，并在全部 PP stages 之间原子生效、收缩或撤销，同时显式计入维持可行 headroom 的机会成本；",
        "如何在执行 DP 的 GPU 故障后隔离旧版本，把整请求迁移到准入时冻结的唯一完整备份/恢复 DP，恢复已提交前缀并仅重算后缀；",
        "如何使上述行为能够通过管理接口、日志、版本字段、ACK、数据路径和故障注入进行验证与侵权取证。",
    ]
    for problem in problems:
        add_bullet(doc, problem)

    add_heading(doc, "4.2  本发明的主要要点概述", 2)
    add_body(doc, "本发明是一种面向单 GPU failure 和 agentic workflow 的推理保护与恢复系统。系统以 causal Slack 表达不同请求对 workflow 完成时间的敏感性，以多层 KV 放置降低高价值请求的恢复开销，并以版本化提交协议保证恢复承诺真实可用。优选技术链如下：")
    overview_steps = [
        "Workflow Handler 维护当前已揭示 DAG，根据节点状态、剩余路径预测和不确定性计算每个活动请求的 causal Slack，并把 dag version、Slack 及预测置信信息随请求下传。Slack 小表示额外恢复时间更容易暴露到 workflow WCT。",
        "请求进入时，Router 针对该 request version 选择执行 DP e(r)，并从模型/layout 兼容、处于不同 failure domain 且通过逐 stage HBM headroom 与 retry service-capacity 检查的完整 DP 中固化单一 b(r) 和 route version；无合格目标时发布 degraded contract。Master 为请求创建 request version 和不可变 content epoch。",
        "每个 PP stage 在本地生成相同 token 区间的 stage-local KV chunk 元数据，并以不可变 content epoch、chunk/token range、stage id 和模型/KV-layout hash 形成内容键；route/manifest version 仅负责把该内容键绑定到当时的执行与恢复位置。",
        "Algorithm 2 根据 causal Slack、GPU failure 场景、恢复/重算时延、正常路径干扰和资源约束，为候选逻辑 chunk 的各 required-stage shards 分别选择对端 HBM、共享 DRAM 或 SSD，并产生带 placement version 与 lease 的目标清单；各 shard 的层级可以不同。若不能为每个 required stage 都形成至少一个有效放置，则该逻辑 chunk 的结果为 N/Recompute。",
        "数据面 Connector 只在同 PP stage 之间直传 KV tensor，或把该 stage 的 tensor 写入清单指定的共享/持久层；Master 不承载 tensor 数据。",
        "目标 stage 校验 request/placement/budget version、token range、布局签名与 checksum 后返回 ACK。Master 按预期 stage 集合聚合 ACK。",
        "只有全部必要 stages 均确认同一逻辑 chunk 各自的有效 HBM/DRAM/持久层 shard placement，且前驱逻辑 chunk 已提交，Master 才把该逻辑 chunk 置为 committed，并推进共同连续 safe frontier。",
        "Master 维护复合恢复 manifest。驱逐、租约过期、校验失败或 demotion 未完成时，先失效相关 manifest；一旦形成前缀空洞，safe frontier 必须回退。",
        "执行 DP 发生单 GPU 故障后，故障隔离 epoch 冻结旧 route/placement 写入但 pin 最后一个 committed manifest；Router 把整请求发送到预先固定的完整健康 b(r)。b(r) 按该只读 manifest 恢复连续前缀、重算剩余后缀并通过 all-stage barrier，随后 Master 才原子发布引用已校验 content key 的 successor route/manifest version。",
        "Algorithm 3 按 DP 统一发布机会式 alpha，使 peer-HBM 保护优先使用低负载时的安全余量。新 budget version 在所有必要 PP stages 确认前不生效；候选评价显式计入恢复价值、正常推理 headroom、保护承诺机会成本与版本变化代价。",
    ]
    for i, text in enumerate(overview_steps, 1):
        add_step(doc, i, text)

    add_body(doc, "三算法关系：Algorithm 1 产生 causal Slack；Algorithm 2 在固定 alpha 下联合选择 HBM、DRAM、SSD 或 Recompute；Algorithm 3 根据 Algorithm 2 的需求和正常负载反馈调整下一窗口 alpha。Slack 决定保护价值，版本核验、all-stage ACK、连续前沿和恢复清单保证即使预测存在误差也不会恢复错误状态。", bold_prefix="三算法关系：")

    add_table(
        doc,
        ["建议保护层级", "建议作为必要技术特征的组合", "可作为从属/并列保护的特征"],
        [
            ["方法主方案", "版本化 logical KV chunk 与 stage shards；各 required-stage shard 可独立选择 HBM/DRAM/持久层；每个 stage 均有有效放置并 all-stage ACK 后才提交；共同连续 safe frontier；故障后按 manifest 恢复前缀并重算后缀。", "全 stage 同 tier 仅为优选简化；租约、checksum、ACK 批处理、优先级输入、压缩、编码。"],
            ["系统主方案", "Master 元数据面与 Connector tensor 数据面分离；Master 维护版本/ACK/manifest/frontier，Connector 执行 stage-aligned 直传。", "Master 可集中、分片或复制；Connector 可基于 RDMA、PCIe、CXL、TCP 或等价传输。"],
            ["预算控制方案", "每 DP 的版本化 HBM admission/reservation envelope alpha 以 budget version 向全部 PP stages 原子发布；同一 DP 各 stage 使用同一 alpha，并按本 stage 物理容量与安全余量校验；目标计入保护承诺的容量机会成本和变化代价；收缩先 demote/失效再释放。", "比例可等同表示为字节、block 数或 token 容量及其映射规则；DRAM/SSD 由共享池容量、带宽和 lease 直接约束。"],
        ],
        widths_cm=[3.1, 7.5, 5.4],
        font_size=8.7,
    )


def build_section_4_architecture(doc):
    add_heading(doc, "4.3  本发明可应用的系统架构或场景", 2)
    add_body(doc, "优选部署是由多个完整 DP 副本组成的大模型在线推理集群，每个 DP 含一个或多个 PP stages。每个 stage 运行 Scheduler/worker 与 stage-aligned Connector。统一 AegisKV Master 逻辑上管理整个保护域的元数据；为避免单点，其物理实现可以是主备、复制状态机或分片服务，但对外提供单调版本和一致的提交决定。")

    add_table(
        doc,
        ["组件", "所在位置", "主要职责", "不承担的职责"],
        [
            ["Workflow Handler / Algorithm 1", "应用/编排层", "维护当前已揭示 workflow DAG；计算带版本和不确定性的 causal Slack 并随请求下传。", "不决定 KV 是否已经提交，不直接搬运 KV tensor。"],
            ["Router", "路由层", "针对 request version 固定执行 DP e(r) 和单一备份/恢复 DP b(r)，并发布 route version；执行 DP 故障后启用完整健康 b(r)。", "不按 PP stage 拼接多个半副本，不在分层放置过程中改变 b(r)。"],
            ["AegisKV Master / Algorithms 2/3", "控制/元数据面", "根据 Slack 和资源状态执行多层放置与 alpha 控制；维护保护对象、版本、lease、ACK、safe frontier 和 manifest。", "不作为 KV tensor 的中转数据通道。"],
            ["Scheduler/Worker", "每个 (DP, PP stage)", "执行模型层，产生 stage-local KV chunk，遵循 route/budget version，报告生命周期事件。", "不自行宣称请求级可恢复前沿。"],
            ["Stage-aligned Connector", "每个 PP stage", "源 p stage 到目标 p stage 或对应存储对象的直接传输；校验并 ACK。", "不得把 p stage 的 KV 写到 q stage。"],
            ["共享 DRAM / 持久层", "Store Node 或等价存储服务", "保存带版本和租约的 KV 对象；按 manifest 提供读取、校验、驱逐通知。", "缓存命中本身不等价于 committed 状态。"],
            ["Fault Monitor / Recovery Coordinator", "控制面", "检测 GPU failure，触发 fence，选择 manifest 并驱动整请求恢复。", "不接受故障 epoch 的迟到提交。"],
        ],
        widths_cm=[3.0, 3.0, 6.2, 3.8],
        font_size=8.2,
    )

    add_heading(doc, "4.3.1  附图说明", 3)
    for item in (
        "附图 1 为应用与路由层、统一 Master 控制面以及分布式 KV 数据面的总体架构示意图；",
        "附图 2 为跨 DP 的同 PP stage 版本化 KV chunk 保护事务及 all-stage ACK 示意图；",
        "附图 3 为同一 logical chunk 的 PP-stage shards 混合使用 peer HBM、共享 DRAM 和持久层的复合恢复清单示意图；",
        "附图 4 为单 GPU failure 后整请求重试到预先固定的完整备份/恢复 DP，并恢复连续前缀、重算后缀的流程图；",
        "附图 5 为每 DP 统一 alpha 的候选价值评价、保护承诺机会成本、变化代价、版本化发布、收缩、demotion 和 safe-frontier 回退流程图。",
    ):
        add_bullet(doc, item)

    add_body(doc, "附图 1 展示应用与路由层、统一 Master 元数据面以及分布式 KV 数据面的边界。图中的共享 DRAM 和持久层是 Connector 可直接选择的独立目标，不要求 KV 先经过 peer HBM。")
    maybe_add_external_figure(doc, 1, "AegisKV 系统总体架构")

    add_heading(doc, "4.3.2  保护对象与状态机", 3)
    add_table(
        doc,
        ["字段组", "最少包含内容", "用途"],
        [
            ["身份字段", "workflow_id、request_id、request_version、content_epoch", "隔离新旧逻辑请求，并把不可变 KV 内容身份与可变 route/placement version 分离。"],
            ["分块字段", "chunk_id、token_start、token_end、predecessor_chunk_id", "确定连续性和前驱依赖。"],
            ["拓扑字段", "source_dp、target_dp、pp_stage_id、required_stage_set", "保证同 stage 传输并确定 all-stage ACK 集合。"],
            ["兼容字段", "model_id、model_revision、KV layout signature/hash、dtype、shape", "阻止布局不兼容的 KV 被恢复，并允许路由变化后对仍存活对象做内容复核。"],
            ["放置字段", "tier、object/location id、placement_version、lease/expiry", "定位数据并支持撤销、驱逐和 demotion。"],
            ["校验字段", "byte length、checksum、transfer status、ACK bitmap", "核验数据完整性和 stage 完成集合。"],
            ["控制字段", "route_version、budget_version、published_alpha、committed_headroom 或等价 shadow price、failure epoch、state", "拒绝迟到消息、实现原子切换并记录保护承诺机会成本。"],
        ],
        widths_cm=[2.7, 7.7, 5.6],
        font_size=8.5,
    )
    add_body(doc, "状态机分为两级。单个 stage shard placement 优选状态依次为 PREPARED（已生成目标）→ TRANSFERRING（传输中）→ ACKED/VALID（该 stage 已校验）→ INVALID（租约、驱逐、故障 fence 或版本替换导致失效）；逻辑 chunk 只有在每个 required stage 都有一个版本一致的 VALID HBM/DRAM/持久层 placement，且前驱连续时才进入 COMMITTED。重复消息按对象标识与版本幂等处理；版本较旧的 ACK 记录为诊断事件但不得改变 frontier。")

    add_heading(doc, "4.3.3  分块提交协议", 3)
    commit_steps = [
        "Master 接收一个新 logical chunk 及其 required-stage shard 候选，验证它们属于同一 request version 和相同 token 区间。",
        "Master 依据 causal Slack、每个 stage 可用容量、传输带宽、预测恢复/重算时延、故障场景权重和正常路径 slowdown guard，为各 shard 分别选择 HBM、DRAM 或 SSD，产生共同的 logical-chunk placement version；各 shard 无需同 tier。候选方案按 workflow WCT 增量、尾部风险和保护成本整体评价，不把一次完整请求的恢复收益重复分配给每个 shard 或 chunk。",
        "各源 stage Connector 获取清单后，只向目标的相同 stage 或该 shard 对应的存储对象传输 stage-local tensor。",
        "目标校验对象身份、版本、布局、长度和 checksum；校验通过才发送包含 stage id 的 ACK。",
        "Master 将 ACK 与 required_stage_set 比对；若任一 required stage 缺少有效的 HBM/DRAM/持久层 placement、版本不一致或校验失败，则 logical chunk 保持 pending、重新放置或最终得到 N/Recompute，不得部分提交。",
        "当全部 required stages 分别确认其有效 HBM/DRAM/持久层 placement 且 predecessor logical chunk 已 committed 时，Master 提交该 logical chunk，并把 safe frontier 推进到其末端。",
        "若后续收到驱逐/lease 到期/损坏事件，Master 先使 manifest entry 失效；存在前缀空洞时回退到空洞前最后一个连续 committed chunk。",
    ]
    for i, text in enumerate(commit_steps, 1):
        add_step(doc, i, text)
    add_body(doc, "上述版本化提交事务见附图 2。该图中的 PP0–PP3 表示不同 stage 各自执行、校验和 ACK；KV 数据传输发生在不同 DP 的同编号 PP stage 之间，而不是从 PP0 传到 PP1。")
    maybe_add_external_figure(doc, 2, "跨 DP 的同 PP stage 版本化 KV chunk 保护事务与 all-stage ACK")

    add_heading(doc, "4.3.4  复合恢复清单与故障流程", 3)
    add_body(doc, "恢复 manifest 是提交状态的快照而不是临时 cache lookup。对于连续前缀中的每一 logical chunk，它按 required stage 分别给出一个有效 HBM/DRAM/持久层 shard placement 的源层级、对象位置、placement version、checksum 和读取顺序；同一 logical chunk 内各 stage 的层级可以不同。只有该集合完整时 logical chunk 才能出现在 committed 前缀中。对未形成完整集合的 logical chunk 及 safe frontier 之后部分，manifest 明确给出 N/Recompute。因而恢复端无需在故障期间临时猜测哪个副本可用。")
    add_body(doc, "附图 3 给出同一 logical chunk 的不同 stage shards 分别位于 peer HBM、共享 DRAM 和持久层的复合清单示例。")
    maybe_add_external_figure(doc, 3, "按 PP stage 组织的多层复合恢复清单")
    add_body(doc, "单 GPU failure 首先按故障对象分支处理。附图 4 展示执行 DP 的某个 GPU 故障这一主恢复分支。若故障仅位于预先固定的 b(r)，则 e(r) 上的请求继续执行；Master 立即使故障 stage 的 peer-HBM placement 和相关逻辑承诺失效，按连续性规则回退 frontier，并选择新的兼容健康 b(r) 发布新 route/manifest version。仍存活的 DRAM/SSD 对象不因路由版本变化而盲目复制或丢弃，只有在 content epoch、token range、layout hash 和 checksum 复核一致后才可重绑定到新 manifest。该分支仍属于单 GPU failure，不把两个 DP 的部分 PP pipeline 拼接为一次执行。")
    recovery_steps = [
        "Fault Monitor 判断某 GPU 不可继续服务，生成 failure epoch；Master 和 Router 冻结旧 route/placement version，拒绝其迟到写入。",
        "Router 启用该 request version 已固定且仍健康的 b(r)，同时 fence 旧写入并 pin 故障前的 committed manifest；本实施例不做 PP stage-level splicing。",
        "目标 DP 按 content epoch 和 model/KV-layout hash 请求 Master 返回不超过故障时刻、仍在 lease 内且校验有效的被 pin manifest；恢复完成后再原子发布 successor route/manifest version。",
        "各目标 PP stages 并行从其对应的 peer HBM、共享 DRAM 或持久层读取相同连续前缀，完成布局与 checksum 验证。",
        "Recovery Coordinator 仅在所有必要 stages 都恢复到同一 safe frontier 后允许请求进入执行队列。",
        "目标 DP 对 safe frontier 之后至故障前进度的 token 前缀执行重算，然后继续 decode；故障 WCT 统计应包含检测、fence、重路由、排队、读取、校验和后缀重算。",
    ]
    for i, text in enumerate(recovery_steps, 1):
        add_step(doc, i, text)
    add_body(doc, "上述单 GPU failure 下的整请求恢复过程见附图 4。")
    maybe_add_external_figure(doc, 4, "GPU failure 后的整请求跨 DP 恢复")

    add_heading(doc, "4.3.5  每 DP 机会式 alpha 的原子发布", 3)
    add_body(doc, "alpha 由 Master 的预算控制模块按 DP 管理，同一 DP 的全部 PP stages 观察同一已提交 budget version 和同一比例值，并各自按物理 HBM 容量计算候选上限、按安全 headroom 做最终校验。alpha 是 admission envelope，不代表启动时切出的永久物理分区，也不要求立即占满；但发布更大的 alpha 会形成维持相应 HBM 保护承诺可行性的控制面义务，因此必须计入 reservation opportunity cost。高负载、缺少高价值候选或 backup/restore 队列挤压正常执行时，alpha 可主动收缩乃至归零。")
    alpha_steps = [
        "Master 汇总各 PP stage 的安全 HBM headroom、正常请求排队、保护/恢复带宽、已有承诺和待保护需求，并在当前向量附近形成若干离散 alpha 候选；优选地每轮只改变一个 DP 或一个预定义的小型故障域组，避免对全部 DP 做指数级笛卡尔积枚举。对每个候选，放置模块在当前已观测窗口内评价可行的 chunk placement，并返回单 GPU failure 下的 workflow WCT 增量、尾部风险、实际保护成本以及维持该 alpha 保护承诺的 reservation opportunity cost。",
        "预算控制模块仅使用已经观测到的反馈预测下一控制窗口，选择预测目标较小且满足正常路径 slowdown、每 stage headroom 和变化速率约束的 alpha；目标另加入 budget-version 变化代价，避免为短暂波动频繁迁移 KV。机会成本防止把更大的未占满上限视为免费，从而使控制器能够在边际恢复收益不足时主动收缩 alpha。该选择不读取未来请求，也不声称未知未来上的全局最优。",
        "Master 向该 DP 的全部必要 stages 发送 PREPARE；每个 stage 验证自身可执行上限并返回 ACK 或 NACK。",
        "只有全部 stages ACK 时，Master 才发布 COMMIT，使新 alpha 对后续 placement 生效；否则继续使用旧版本。",
        "扩大预算只改变 admission 上限，不立即占满 HBM。收缩预算时，Master 先选择待迁移对象，将其 demote 至 DRAM/SSD并完成新位置 all-stage ACK。",
        "Master 原子切换 manifest 后释放旧 HBM；若迁移无法完成，则先把受影响对象置为 INVALID、回退 safe frontier，再释放空间，不得保留虚假的恢复承诺。",
        "为避免振荡，可使用 guard band、hysteresis 和 rate limit；这些参数不改变版本化原子发布语义。",
    ]
    for i, text in enumerate(alpha_steps, 1):
        add_step(doc, i, text)
    add_body(doc, "上述每 DP 统一 alpha 的版本化发布、收缩和回退顺序见附图 5。")
    maybe_add_external_figure(doc, 5, "每 DP 机会式 alpha 的原子发布、demotion 与回退")


def build_section_4_product_and_evidence(doc):
    add_heading(doc, "4.4  本发明涉及的产品、可能售卖形态及侵权取证方式", 2)
    add_heading(doc, "4.4.1  产品与售卖形态", 3)
    product_rows = [
        ["大模型推理引擎增强模块", "以软件许可证、容器镜像、插件或源代码交付，集成 Master、Connector、故障恢复与管理接口。"],
        ["KVCache 容错控制服务", "作为独立控制面/微服务售卖，向多个 vLLM、SGLang 或等价推理后端提供版本、放置、ACK 聚合和 manifest。"],
        ["云端托管推理服务", "以 API、按 token/实例/容量计费的服务提供 GPU failure 自动恢复，用户不直接接触内部组件。"],
        ["一体化推理设备或集群方案", "GPU 服务器、共享内存/存储节点、网络和 AegisKV 软件的组合交付。"],
        ["数据面加速组件", "Connector 与 RDMA/NIC/DPU/SmartNIC 或存储网关结合，提供同 stage 传输、checksum 和 ACK。"],
        ["可观测与合规模块", "提供 manifest、safe frontier、版本、恢复事件和正常路径 guard 的审计接口或控制台。"],
    ]
    add_table(doc, ["产品形态", "可能的售卖/交付方式"], product_rows,
              widths_cm=[5.0, 11.0], font_size=9)

    add_heading(doc, "4.4.2  侵权取证方式", 3)
    add_body(doc, "取证应在获得授权的测试环境、公开文档/接口或依法进行的证据保全中完成。单一性能现象只能作为线索；应优先获取能同时证明版本化对象、all-stage ACK、连续前沿和复合恢复清单的多源证据。")
    evidence_rows = [
        ["公开资料与产品界面", "产品手册、配置项、管理 API、控制台、日志字段中是否出现 request/chunk/placement/budget version、safe/committed frontier、stage ACK bitmap、manifest、demotion、lease 等对应概念。", "中：可证明功能承诺和字段，但需结合运行证据。"],
        ["授权黑盒故障注入", "在多 DP×PP 部署中终止某 GPU，记录请求是否整体重路由；监测恢复前缀的数据读取和后缀计算量；对不同 chunk/lease 状态重复测试。", "中：能观察整请求迁移与恢复/重算分界，但不能单独证明内部提交规则。"],
        ["网络与存储侧证据", "在合法授权下抓取 stage 编号、对象键、token range、版本和 ACK；核对源 DP.PPp 是否只向目标 DP.PPp 或对应存储对象发送 KV。", "高：可证明同 stage 传输和版本化 ACK。"],
        ["服务端审计日志/数据库", "检查 expected_stage_set、ACK bitmap、committed 状态、safe frontier 回退、manifest tier、failure epoch 和 stale write rejection。", "高：可直接证明控制面协议。"],
        ["二进制/镜像/源码鉴定", "经合法取得后分析 protobuf/schema、状态机、错误码、符号、配置和调用链，确认 Master 不转发 tensor、Connector 执行直传及预算两阶段发布。", "高：适合司法鉴定或证据保全。"],
        ["资源变化关联", "控制正常负载、HBM headroom 和保护需求，观察每 DP alpha/budget version 是否原子变化；在实际副本未占满时，较大 alpha 是否仍改变正常 admission、保留 headroom 或暴露 shadow-price/机会成本指标；收缩前是否发生 demotion 或 frontier 回退。", "辅助：需结合版本日志和目标函数审计字段，避免把普通缓存驱逐误判为本方案。"],
    ]
    add_table(doc, ["取证来源", "建议观察内容", "证明力与限制"], evidence_rows,
              widths_cm=[3.2, 8.4, 4.4], font_size=8.3)


def build_section_4_embodiments(doc):
    add_heading(doc, "4.5  本发明技术实现的具体实施例", 2)
    add_body(doc, "下述实施例用于说明可以实际执行的消息、状态和恢复顺序，不限定 DP/PP 数量、chunk 大小、传输协议或存储介质。实施例中的编号是说明性标识，不是实验结果。")

    add_heading(doc, "4.5.1  实施例一：DP=2、PP=4 的同 stage HBM 保护与连续提交", 3)
    add_body(doc, "部署包括 DP0 和 DP1，每个 DP 含 PP0、PP1、PP2、PP3 四个独立 Scheduler/worker；各 stage 只拥有其模型层范围的 KV。请求 R 在 DP0 执行；Router 在准入时冻结 DP0=e(R)、DP1=b(R) 并发布 route version 31，Master 创建 request version 17。")
    embodiment_1 = [
        "当请求 R 完成 token [0,512) 时，各 stage 上报 chunk C0 的 stage-local 元数据。Master 检查四个 stage 的 token range 一致，生成 placement version 91，目标为 DP1 的 peer HBM。",
        "DP0.PP0、PP1、PP2、PP3 的 Connector 分别只向 DP1.PP0、PP1、PP2、PP3 发送自己拥有的 KV tensor。每个目标 stage 校验 model revision、layout signature、request/placement/budget version、长度和 checksum。",
        "DP1 的四个 stages 分别返回 ACK(17, C0, 91, stage_id)。Master 的 expected_stage_set 为 {0,1,2,3}。若只收到 stage 0、1、2 的 ACK，C0 保持 pending，safe frontier 仍为 0。",
        "收齐 stage 3 且校验通过后，Master 把 C0 置为 committed，safe frontier 推进到 token 512。C1=[512,1024)、C2=[1024,1536) 以相同流程提交，frontier 依次推进且不可跳过中间分块。",
        "随后 DP0.PP2 发生 GPU failure。Fault Monitor 生成 failure epoch 8；Master fence route version 31 的新增写入和迟到 ACK，但 pin 其最后 committed manifest 供恢复只读；Router 只激活该版本已冻结的 b(R)=DP1，而不在故障后重新竞选目标。全部 stages 通过恢复 barrier 后，再原子发布以 DP1 为执行端的 successor route/manifest epoch。",
        "DP1 的四个 stages 读取同一 committed manifest。由于 C0-C2 已位于各自 stage 的 HBM，系统验证到相同 token 1536 后，从该处重算故障前尚未提交的后缀，再继续正常推理。任何 stage 未恢复到 1536 时均不得提前继续。",
    ]
    for i, text in enumerate(embodiment_1, 1):
        add_step(doc, i, text)
    add_body(doc, "本实施例的技术效果是：请求级可恢复性由四个 stage 的共同提交事实决定，避免仅凭最快 stage 或聚合字节数形成不一致恢复；故障后不需要重算已经提交的连续前缀。该效果是协议因果效果，不代表本文已经测得特定百分比加速。")

    add_heading(doc, "4.5.2  实施例二：HBM/DRAM/SSD 复合清单、前缀空洞与回退", 3)
    add_body(doc, "部署包括四个 DP，每个 DP 含四个 PP stages。请求 Q 在 DP0 执行，Router 针对本 request version 固定 DP1 为 b(Q)。为便于说明，本优选简单实施例让同一 logical chunk 的全部 stage shards 使用同一 tier：C0=[0,512) 的 PP0–PP3 shards 均位于 DP1 对应 stage HBM，C1=[512,1024) 的全部 shards 均位于共享 DRAM，C2=[1024,1536) 的全部 shards 均位于 SSD；C3=[1536,2048) 因未为全部 required stages 形成完整放置集合，逻辑结果为 N/Recompute。全 stage 同 tier 不是提交条件，也不限制保护范围。")
    embodiment_2 = [
        "Master 生成的 committed manifest 对 C0、C1、C2 按 stage 分别记录 tier、对象位置、placement version、checksum、lease 和读取源；C3 明确记录为 N/Recompute，因此 safe frontier 为 token 1536，而不是 2048。",
        "若 DP0 的 GPU 故障，Router 启用完整健康的 b(Q)=DP1。DP1.PPp 直接使用本 stage HBM 中的 C0，并分别从共享 DRAM 的 p-stage 对象和 SSD 的 p-stage 对象读取 C1、C2；读取可并行，但必须按 chunk 前驱关系统一验证。",
        "全部 stages 确认恢复到 1536 后，DP1 重算 C3 及其后的状态，并继续 decode。故障路径不把“SSD 有 C2”误解为可跳过缺失 C1。",
        "若 C1 的租约先到期，Store Node 向 Master 发出 eviction/expiry 事件。Master 立即使 C1 的 manifest entry 失效；即使 C2 仍存在，连续 safe frontier 也回退到 C0 末端 token 512。C2 可保留为 best-effort cache，但不再作为恢复承诺。",
        "若需把 C0 从 peer HBM demote 到 DRAM，Master 先创建新的 placement version，把四个 stage 的 C0 写入 DRAM并收齐 ACK，再原子切换 manifest，最后释放旧 HBM。任何中间失败都保留旧有效版本或显式失效，不能产生双版本歧义。",
    ]
    for i, text in enumerate(embodiment_2, 1):
        add_step(doc, i, text)
    add_body(doc, "附图 3 给出不采用上述简化的等价放置：同一 logical chunk j 的 PP0 shard 位于 b(r) 的 peer HBM，PP1/PP3 shards 位于共享 DRAM，PP2 shard 位于持久层。只要四个 required stages 各自都有一个有效、版本一致的 HBM/DRAM/持久层 placement 并完成 ACK，chunk j 即可提交；反之，即便其中三个 shards 已复制，logical chunk 的结果仍是 N/Recompute，局部副本不构成恢复承诺。")
    add_body(doc, "本实施例说明本发明保护的是“按 stage 组织、带连续性与版本语义的复合恢复清单”，而不是限定同一 logical chunk 的全部 shards 必须位于相同层级，也不是限定某一种存储介质。HBM、共享 DRAM 和 SSD 可以由等价的设备内存、CXL 内存、持久内存、NVMe-oF 或对象存储替换，只要保留对象版本、stage 对应、提交与失效语义。")
    add_body(doc, "备份侧故障补充例：若请求仍在 DP0 执行时 DP1.PP2 单独故障，Master 使 DP1.PP2 上的 peer-HBM shard 失效并据此回退承诺，但不停止 DP0 的正常执行。Router 选择新的完整健康 b(Q) 并发布新 route/manifest version；C1/C2 在共享 DRAM/SSD 中的对象经 content epoch、token range、layout hash 和 checksum 复核后可重绑定，无需因路由版本变化重新写入相同 KV 内容。")

    add_heading(doc, "4.5.3  实施例三：每 DP 机会式 alpha 的原子发布与收缩", 3)
    add_body(doc, "DP2 含四个 PP stages。Master 观察到四个 stages 在满足正常推理安全余量后，分别还可安全提供 10 GiB、8 GiB、12 GiB 和 6 GiB HBM。假设每个 stage 的可管理 HBM 容量为 24 GiB，则统一比例 alpha 不能超过最紧张 stage 对应的 25%；Master 可把 alpha=0、1/12、1/6、1/4 作为离散候选。即使 alpha=1/4 提供更大的 placement 可行域，其保护承诺机会成本也高于 alpha=1/6；若当前窗口的完整放置评价表明额外恢复收益不能覆盖该机会成本和版本变化代价，Master 可选择 alpha=1/6 并赋予 budget version 44，使每个 stage 的候选上限为 4 GiB。数值仅用于说明价值评价、同一比例与逐 stage 校验协议。")
    embodiment_3 = [
        "Master 向 DP2.PP0-PP3 发送 PREPARE(44, alpha=1/6)。每个 stage 按本地物理容量换算候选字节上限，再检查正在 admission 的正常请求、运行 batch、KV 占用和保留 guard；只有四个 stages 全部 ACK，Master 才发送 COMMIT(44)。",
        "若 PP3 返回 NACK，则 version 44 不生效，全部 stages 继续使用 version 43。任何携带 44 的 placement ACK 都不能被提交，避免部分 stages 已放大而部分仍采用旧预算。",
        "version 44 生效后，4 GiB 仍是 admission 上限而非必须立即占满的分区；但在该版本有效期间，控制器持续计入维持其保护承诺可行 headroom 的机会成本。placement 模块根据 causal Slack、预计可避免的重算、写入/恢复代价、故障场景和租约生命周期联合选择分块。Slack 预测误差可以改变保护价值，但不改变 ACK 或 frontier 的正确性判断。",
        "当正常请求负载上升时，Master 生成较小上限或零上限的 version 45。对于超过新上限的 committed HBM chunks，Master 先将其 demote 到共享 DRAM/SSD，完成新位置的 all-stage ACK 并切换 manifest，然后才释放 HBM。",
        "若 DRAM/SSD 容量不足，Master 按策略选择失效对象，先把对应 manifest entry 置为 INVALID 并回退 safe frontier，再释放 HBM；系统同时记录受影响 request/chunk、旧/新 budget version 和回退原因以供审计。",
        "为抑制频繁升降，可要求 headroom 连续多个控制窗口满足阈值、限制每次改变量并设置 guard band。这些控制参数改变调节速度，不改变“每 DP 统一、全 stage 原子、先切换承诺再释放”的核心机制。",
    ]
    for i, text in enumerate(embodiment_3, 1):
        add_step(doc, i, text)
    add_body(doc, "共享 DRAM 与 SSD 由 placement 模块按实际池容量、lease、队列和带宽直接约束；Algorithm 3 仅对稀缺且会直接影响正常推理的 HBM 维护机会式 alpha。")


def build_section_4_innovations(doc):
    add_heading(doc, "4.6  本发明的主要创新点及其对应的技术效果", 2)
    innovations = [
        ["1. 版本化 stage-shard 对象、all-stage ACK 与连续 safe frontier（核心必要）", "把分散在各 PP stage 和不同物理层级的 KV shard 聚合为可验证的 logical-chunk 恢复承诺；只有身份、版本和校验均通过的全部必要 stages 才推进连续前沿，避免普通 cache hit 或 partial copy 冒充可恢复状态。"],
        ["2. 冻结备份 DP、同 stage 数据通路与整请求确定恢复（核心必要）", "Router 在准入时冻结唯一完整 b(r)；Master 只发布元数据决定，Connector 在同编号 PP stage、共享 DRAM 和 SSD 间直传 KV。执行 GPU 故障后激活既定 b(r)，按 manifest 恢复前缀并重建后缀，不在故障时临时拼接或重选不完整 DP。"],
        ["3. 三类物理层级与 logical Recompute 的联合决策（重要实施）", "被保护 chunk 的每个必要 stage 在 HBM、DRAM、SSD 中恰选一层，否则整个 chunk 取 logical Recompute；联合考虑恢复 regret、保护资源成本和正常路径硬门禁，而不是把 Recompute 误作某个 shard 的第四种存储层级。"],
        ["4. 每 DP HBM admission envelope 的全 stage 原子发布、demotion 和回退（从属增强）", "Algorithm 3 用 placement value、预留机会成本和版本变化惩罚评价 alpha 候选；同一 DP 的 stages 原子确认同一版本，高负载时可主动收缩，且收缩不会静默破坏已有 recovery manifest。"],
        ["5. 已揭示 DAG 的 causal Slack 价值信号（可选从属）", "在不知道未来工具结果和最终请求长度的条件下，以 causal Slack 表达延迟传播敏感性并改善保护排序；Slack 不参与 ACK/frontier 正确性，也可由 SLO、业务等级或其他可审计价值信号替换。"],
    ]
    add_table(doc, ["本发明的主要创新点", "对应的有益效果"], innovations,
              widths_cm=[7.2, 8.8], font_size=8.8)
    add_body(doc, "创新点关系：创新点 1 与 2 构成可执行、可记录和可取证的协议/数据路径核心；创新点 3 与 4 是资源效率和动态容量控制实施；创新点 5 仅提供 agent-aware 价值信号，不是恢复正确性的前提。独立权利要求应优先围绕版本化对象、全 stage 提交、连续前沿、冻结恢复目标和确定恢复链，而不把既有 critical-path/Slack 概念或单个目标公式作为唯一发明核心。", bold_prefix="创新点关系：")


def build_section_5(doc):
    add_heading(doc, "5  发散思维：变形方案、应用扩展与规避设计", 1)
    add_body(doc, "以下变形均以保留“版本化分块—全 stage 提交—连续前沿—复合清单—确定恢复”这一发明构思为前提。是否写入独立或从属权利要求，应由专利代理人结合在先检索和单一性要求决定。")
    variants = [
        ["控制面形态", "统一 Master 可替换为主备、Raft/复制状态机、按模型/租户分片的逻辑 Master，或与 Router/存储元数据服务同进程部署；只要仍提供单调版本和唯一提交决定。"],
        ["存储层级", "“HBM/DRAM/SSD”可替换为其他 GPU/加速器设备内存、CXL 内存、持久内存、NVMe/NVMe-oF、远程内存、对象存储或其组合。"],
        ["保护编码", "完整副本可替换为压缩、量化、增量编码、去重、纠删码或奇偶校验；恢复清单应记录编码参数、组成片段和可恢复条件。"],
        ["分块方式", "固定 token 数可替换为按字节、层组、page、batch、时间或带宽自适应边界；连续性仍映射到同一逻辑 token 前缀。"],
        ["ACK 载体", "独立 ACK 消息可替换为批量 ACK、位图、日志提交索引、存储事务回执或 piggyback 字段；必须能确定每个必要 stage 对同一版本已完成。"],
        ["alpha 表示", "比例可替换为字节数、block 数、token 容量或按模型的配额；新值仍通过 budget version 全 stage 原子生效。"],
        ["并行拓扑", "PP 可与 TP、EP/MoE 或多节点混合；required_stage_set 可扩展为恢复所需 shard 集合，safe frontier 取所有必要 shard 的共同连续前沿。"],
        ["工作流价值输入", "causal Slack 可与业务等级、请求截止期、预测故障损失或租户策略组合；无 DAG 的退化实施例可使用 deadline，但 agent-aware 优选实施例以已揭示 DAG Slack 为主输入。"],
        ["故障检测", "驱动错误、心跳超时、进程退出、GPU reset、链路隔离或管理员指令均可生成 failure epoch；本交底书优选实施例只处理 GPU failure。"],
        ["售卖边界", "友商可能把 Master、Router、Connector 和 Store 拆成多个产品，或把字段改名。取证和权利要求应按功能关系覆盖，而不依赖模块商品名。"],
    ]
    add_table(doc, ["可能变形/规避方向", "等同或扩展实现"], variants,
              widths_cm=[4.2, 11.8], font_size=8.7)

    add_heading(doc, "5.1  可扩展应用", 2)
    for item in (
        "面向多轮聊天、代码智能体、深度研究和多智能体协作等长上下文在线推理，以 causal Slack 识别 failure delay 是否会传播到 workflow WCT。",
        "面向普通单请求 LLM serving，在没有 workflow DAG 的情况下仍可按 deadline、业务等级或恢复成本使用同一保护协议。",
        "面向多模型或多租户集群，以模型/KV layout signature 和租户 lease 隔离保护对象；Master 可按租户设定 quota。",
        "面向异构 GPU 集群，manifest 可记录恢复目标的布局转换或兼容版本；不兼容时显式选择 Recompute，不能强行恢复。",
    ):
        add_bullet(doc, item)

    add_heading(doc, "5.2  建议代理人重点评估的权利要求边界", 2)
    for item in (
        "是否以“all-stage ACK + 连续 safe frontier + 复合恢复 manifest”作为最小可区别的独立方法组合；",
        "同 stage 跨 DP 传输与 Master/tensor 数据面分离应放入独立系统权利要求还是高价值从属权利要求；",
        "机会式 alpha 的原子发布是否单独形成并列独立方案，以及如何覆盖比例/字节/block 等等同表示；",
        "Mooncake、DejaVu 及其作者/机构专利族是否已经公开相同的多 stage 提交或多层故障 manifest；",
        "中国、美国、欧洲申请对软件方法、系统、设备和计算机可读存储介质客体的不同撰写要求。",
    ):
        add_bullet(doc, item)


def build_section_6(doc):
    add_heading(doc, "6  在先公开的现有技术文献信息", 1)
    warning = doc.add_table(rows=1, cols=1)
    warning.style = "Table Grid"
    set_row_cant_split(warning.rows[0])
    set_cell_text(
        warning.cell(0, 0),
        "检索状态：下表仅列出目前可核验的代表性公开论文，未完成 CNIPA、WIPO Patentscope、Espacenet、Google Patents、USPTO 等专利数据库的系统检索。正式申请前必须补充检索论文引用链、作者/机构申请人、专利族、同族法律状态和可能的抵触申请。检索负责人为【待填写：专利代理人/检索负责人】，计划或实际完成日期为【待填写：检索完成日期】。",
        bold=True,
        center=False,
        size=9.5,
    )

    references = [
        ["1", "MOONCAKE；USENIX FAST 2025", "2025-02-25（会议论文集）", "USENIX 官方页面：https://www.usenix.org/conference/fast25/presentation/qin", "§3（架构与 Mooncake Store）、§4（调度），Fig. 2（总体架构）、Fig. 4（传输引擎）；重点核对多层 KVCache、全局元数据和热点副本。"],
        ["2", "IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference；USENIX FAST 2025", "2025-02-25（会议论文集）", "USENIX 官方页面：https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress", "§4.3（important token identification）、§4.4（prefix KV management）及 Fig. 6；重点核对多层 prefix KV、重排和缓存替换。"],
        ["3", "DejaVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving；ICML 2024 / PMLR 235", "2024-07（公开论文集）", "PMLR 官方页面：https://proceedings.mlr.press/v235/strati24a.html", "§4.1（DejaVuLib）、§4.2.3（Failure Handling），Fig. 7（streaming）、Fig. 10（failure and recovery）；重点核对复制目标、ACK/恢复边界和故障流程。"],
        ["4", "RaidServe: High-performance Resilient Serving；MLSys 2026（早期预印本题名 FailSafe）", "2026（会议论文）", "Stanford MAST 官方页面：https://mast.stanford.edu/pubs/raidserve/", "Cyclic KVCache Placement、Hybrid Attention、Fine-Grained Load-Aware Routing、proactive KVCache backup；重点核对 TP 拓扑、主动备份和恢复路由。"],
        ["5", "GhostServe: A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving；MLSys 2026", "2026（公开论文）", "arXiv：https://arxiv.org/abs/2605.00831；正式论文集由代理人归档", "host-memory parity、streaming checkpoint 和故障重建章节；重点核对编码粒度、checkpoint 一致性和恢复条件。"],
        ["6", "LUMEN: Coordinated Failure Recovery for Distributed LLM Serving；arXiv 2026", "2026（公开预印本）", "arXiv：https://arxiv.org/abs/2606.17787", "checkpoint placement、interrupted-request dispatch、capacity restoration；重点核对固定 holder、负载感知派发和恢复前沿。"],
        ["7", "Efficient Memory Management for Large Language Model Serving with PagedAttention；ACM SOSP 2023（vLLM）", "2023（ACM 会议论文集）", "ACM DOI：https://dl.acm.org/doi/10.1145/3600006.3613165", "PagedAttention 与 KV block 管理、调度和内存共享章节；重点核对分页 KVCache 对象与本发明版本化恢复对象的区别。"],
        ["8", "SGLang: Efficient Execution of Structured Language Model Programs；NeurIPS 2024", "2024（NeurIPS 论文集）", "NeurIPS DOI：https://doi.org/10.52202/079017-2000", "RadixAttention、runtime/compiler 与 structured LLM programs 章节；重点核对前缀复用、工作流执行和底层故障语义。"],
        ["9", "Concordia: JIT-Compiled Persistent-Kernel Checkpointing for Fault-Tolerant LLM Inference；arXiv 2026", "2026-06-22（预印本）", "arXiv：https://arxiv.org/abs/2606.23521", "GPU 驻留执行上下文、KV-block scanner、增量 checkpoint 与 CPU-visible log；重点核对提交记录、状态范围和恢复边界。"],
        ["10", "SYMPHONY: Enabling Compute-Memory Disaggregation in LLM Serving Systems；USENIX NSDI 2026", "2026-05（会议论文集）", "USENIX 官方页面：https://www.usenix.org/conference/nsdi26/presentation/agarwal", "advisory requests、priority-based KV cache management 与 cooperative memory management；重点核对优先级、多层管理和一致性语义。"],
    ]
    add_table(
        doc,
        ["序号", "期刊/会议及论文", "公开日期", "文档来源", "相关章节/图及检索重点"],
        references,
        widths_cm=[1.0, 4.6, 2.5, 3.8, 4.1],
        font_size=7.7,
    )
    add_body(doc, "补充检索关键词建议：KV cache checkpoint、KV cache replication、pipeline parallel failure recovery、stage-aligned transfer、committed prefix、recovery frontier、multi-tier KV cache、GPU inference fault tolerance、opportunistic HBM reservation，以及相应中英文同义词。")
    add_body(doc, "待补充专利文献：上述检索负责人应按计划完成日期补充文献号、申请人、优先权日、同族信息和与创新点 1–5 的逐项对比。")


def build_checklist(doc):
    doc.add_page_break()
    add_heading(doc, "附录 A  专利交底书 Checklist", 1)
    checklist = [
        ["1", "是否使用统一且有定义的专业术语，英文缩写是否给出全称？", "已覆盖", "第 2 章给出缩略语和关键术语；正文统一使用 request/placement/route/budget version、safe frontier、manifest、Master 与 Connector。"],
        ["2", "背景是否只描述已公开技术，且避免把内部迭代方案当作现有技术？", "已覆盖", "第 3 章只比较 10 项可核验的公开论文/预印本，并明确论文比较不能替代专利检索。"],
        ["3", "技术问题是否均能由本方案解决，是否避免夸大？", "已覆盖", "第 4.1 节问题与第 4.2–4.5 节机制逐项对应；未声称未测得的性能数值。"],
        ["4", "方案是否按组件、时机、场景、动作和状态完整描述？", "已覆盖", "给出组件职责、对象字段、状态机、提交/恢复/预算流程和异常回退。"],
        ["5", "是否提供多个完整实施例？", "已覆盖", "第 4.5 节包含 DP=2、PP=4 的同 stage HBM 提交、混合层级 manifest、alpha 原子发布三个实施例。"],
        ["6", "附图是否黑白、清晰、可编辑并与正文交互一致？", "已覆盖", "附图 1–5 均嵌入高分辨率黑白 PNG，并保留对应 Draw.io 可编辑源；提交前仍需按代理人制图规范统一部件编号。"],
        ["7", "创新点是否是技术特征且对应技术效果？", "已覆盖", "第 4.6 节列出 5 个创新点并区分必要、重要和从属关系。"],
        ["8", "是否说明产品形态和侵权取证？", "已覆盖", "第 4.4 节覆盖软件、控制服务、云服务、一体机、数据面组件及六类取证来源与限制。"],
        ["9", "是否提供在先文献信息并完成专利检索？", "部分完成", "已列 10 项可核验论文/预印本；专利数据库、标准/提案、作者/机构申请人和专利族检索仍须由代理人完成。"],
        ["10", "发明人、主体、联系人、时间证据和项目编号等信息是否完整？", "待完成", "正文共保留 12 个黄色高亮红字待填字段，正式提交前必须逐项补齐并复核一致性。"],
        ["11", "是否完整体现 agent-aware 三算法链？", "已覆盖", "Algorithm 1 计算 causal Slack；Algorithm 2 为被保护 chunk 的 stage shards 选择 HBM/DRAM/SSD，否则输出 logical Recompute；Algorithm 3 动态调整 per-DP alpha；ACK/frontier/manifest 保证恢复正确性。"],
        ["12", "是否无意把同一 logical chunk 限定为全 stage 同 tier？", "已规避", "第 2 章、4.2、4.3.3、4.3.4 和附图 3 均明确 shard 可分别位于 HBM/DRAM/持久层；全 stage 同 tier 仅为 4.5.2 的优选简单实施例。"],
        ["13", "是否完成实验与产品验证？", "未声称完成", "本交底书不使用旧实验或估算值证明技术效果；工程实现和测试记录应在后续补充，但不影响方案的逻辑可实施描述。"],
    ]
    add_table(doc, ["序号", "检查项", "状态", "本稿对应内容/待办"], checklist,
              widths_cm=[1.0, 6.1, 2.2, 6.7], font_size=8.3)

    add_heading(doc, "A.1  发明人贡献与权属记录", 2)
    add_body(doc, "发明人资格应结合拟申请权利要求判断，并记录每位候选发明人对技术构思的实质贡献。仅承担项目管理、按指令实现、常规测试或提供资源通常不能单独证明对发明构思作出贡献；最终名单和排序由合作双方及专利代理人根据适用法律确认。")
    add_table(
        doc,
        ["候选发明人", "对应创新点/独立技术贡献", "最早证据位置与日期", "所属主体及权属确认"],
        [["待由项目组填写", "建议映射至创新点 1–5，并说明是提出技术构思还是完成常规实现", "会议纪要、设计稿、代码提交或实验记录", "待由合作双方和代理人确认"]],
        widths_cm=[3.1, 6.1, 3.8, 3.0],
        font_size=8.5,
    )

    add_heading(doc, "正式提交前待办", 2)
    for item in (
        "补齐全部 12 个高亮待填字段，并由合作双方确认发明人贡献、申请主体、权属、关键时间证据和保密级别；",
        "由专利代理人完成专利数据库检索、专利族核对和创新点逐项对比；",
        "根据代理人意见把附图 1–5 重绘为标准专利附图，并统一部件编号；",
        "决定是否将机会式 alpha 作为并列独立方案，以及是否拆分方法/系统申请；",
        "由研发人员复核消息字段、状态转换、故障 fence、demotion 顺序和三项实施例的可实施性；",
        "正式申请前继续保密，避免论文投稿、开源提交或公开演示造成新颖性风险；公开时间与申请策略由项目负责人和专利代理人共同确认。",
    ):
        add_bullet(doc, item)


def _build_legacy_concise_disclosure(doc):
    """Build the review-oriented concise disclosure.

    The disclosure keeps the Huawei template's required Sections 2--6 and the
    five patent drawings, while centering the protocol features that are easier
    to identify and protect than an isolated scheduling formula.
    """
    add_heading(doc, "2  缩略语和关键术语", 1)
    add_table(
        doc,
        ["术语", "本文含义"],
        [
            ["KVCache / KV chunk", "请求执行产生的键值缓存；按连续 token 区间形成 logical chunk，并按 PP stage 分片。"],
            ["content epoch", "标识 KV 内容世代，防止旧内容被误认为当前请求状态。"],
            ["route/manifest version", "绑定执行 DP、单一备份/恢复 DP、分层位置和恢复清单的单调版本。"],
            ["all-stage ACK", "一个 logical chunk 的全部必要 PP stage 均完成身份、版本和校验确认。"],
            ["safe frontier", "从请求起点连续满足 all-stage ACK 的最大 chunk 边界；恢复不得越过该边界。"],
            ["alpha", "每个 DP 可用于机会式 peer-HBM 保护的统一准入上限；同一 DP 的全部 PP stages 使用相同 alpha 值。"],
            ["causal Slack", "依据当前已揭示 workflow DAG 估计的可容忍延迟；不使用未来真实输出长度。"],
        ],
        widths_cm=[4.2, 11.8],
        font_size=9.0,
    )

    add_heading(doc, "3  相关技术背景以及最接近的现有技术", 1)
    add_body(
        doc,
        "Agentic workflow 由多轮 prompt、thinking 和 tool call 组成。请求前缀会随轮次增长；一次 GPU failure 若只能从输入重新计算，会放大恢复时间，并可能沿依赖关系延长整个 workflow。与此同时，持续备份会消耗 HBM、主机内存、存储带宽和正常推理容量。",
    )
    add_body(
        doc,
        "Mooncake、vLLM 和 SGLang 已证明 KVCache 分页、复用、分层存储及直接传输的工程价值，但普通 cache hit、已传输字节数或局部 stage 副本并不自动构成故障后可执行的恢复点。本发明关注的技术缺口是：如何把分散于多个 PP stages 和存储层级的 KV 变成可验证、可回退的恢复承诺。",
    )
    add_body(
        doc,
        "Real CC trace 中的并行 subagent 执行时长存在显著组内差异：在 393 个 workflow 的 321 次并行 fan-out 中，最慢/最快分支时长比的中位数为 2.2 倍，P90 为 8.6 倍。fan-in 等待最慢分支，因此故障恢复价值必须考虑请求是否会延长当前最长路径。下列图同时展示该现象、长上下文重算开销和相同 KV 损失在不同 workflow 位置的 WCT 影响；它们用于说明技术问题，不构成完整系统的性能承诺。",
    )
    add_evidence_figure(
        doc,
        REAL_CC_FIGURE,
        "背景实验图  Real CC trace 中并行 subagent 的执行时长与 fan-out 内差异",
    )
    add_evidence_figure(
        doc,
        MOTIVATION_RECOVERY_FIGURE,
        "背景实验图  长上下文下的恢复开销趋势",
    )
    add_evidence_figure(
        doc,
        MOTIVATION_POSITION_FIGURE,
        "背景实验图  相同 KV 损失在不同 workflow 位置的 WCT 影响",
        width_cm=14.0,
    )
    add_table(
        doc,
        ["现有方向", "已有能力", "本发明补充的边界"],
        [
            ["vLLM / SGLang", "分页 KV、前缀复用、连续批处理", "不以 all-stage ACK 和连续前沿定义跨 DP 恢复承诺。"],
            ["Mooncake", "全局元数据、多层 KVCache 和直接传输", "缓存可用性不等同于版本化的 GPU 故障恢复点。"],
            ["容错复制/检查点", "复制 KV 或保存中间状态", "通常不同时约束 stage 身份、连续性、冻结恢复目标及正常路径干扰。"],
        ],
        widths_cm=[3.3, 5.1, 7.6],
        font_size=8.8,
    )

    add_heading(doc, "本发明技术方案的详细阐述", 1)
    add_heading(doc, "要解决的技术问题", 2)
    for item in (
        "把不断增长且跨 PP stage 分布的 KVCache 转化为可验证的连续恢复点；",
        "在 peer HBM、共享 DRAM、SSD 与 Recompute 之间选择保护方式，同时限制对健康请求的干扰；",
        "GPU failure 后在既定完整 DP 上恢复整个请求，避免临时拼接不完整的 stage 状态。",
    ):
        add_bullet(doc, item)

    add_heading(doc, "总体架构与职责边界", 2)
    add_body(
        doc,
        "Application 层的 Algorithm 1 计算 causal Slack；Router 在准入时冻结执行 DP、单一备份/恢复 DP 和 route version。统一 Master 保存元数据，运行 Algorithm 2 与 Algorithm 3，并发布版本化计划。Connector 只在数据面复制 KV、返回 stage ACK，不替代 Master 作全局提交决定。",
    )
    maybe_add_external_figure(doc, 1, "系统总体架构与控制/数据面边界")

    add_heading(doc, "4.3  版本化保护事务", 2)
    add_body(
        doc,
        "每个 logical chunk 由 request、turn、chunk、content epoch、token 区间和 layout signature 唯一标识。Master 为本次放置生成 route/manifest version；跨 DP 的同 PP stage Connector 执行复制。目标 stage 仅在身份、版本、长度和校验通过后返回 ACK。",
    )
    add_body(
        doc,
        "只有全部必要 stages 对同一版本返回 ACK，Master 才提交该 chunk。晚到 ACK、部分完成传输和过期版本均不能推进 safe frontier。租约到期后可清理未提交对象，因此故障时不会把孤儿数据误当作恢复状态。",
    )
    maybe_add_external_figure(doc, 2, "同 PP stage 复制与 all-stage ACK 提交")

    add_heading(doc, "4.4  多层放置与复合恢复清单", 2)
    add_body(
        doc,
        "Algorithm 2 在每个已完成 chunk 到来时比较恢复收益与保护成本。收益由发生目标 GPU failure 时可减少的 workflow 延迟估计；成本包括 D2H/P2P/存储传输、容量占用和对健康请求的正常路径干扰。硬约束覆盖层级容量、带宽、故障域隔离、版本一致性和备份 DP 的重试服务能力。",
    )
    add_body(
        doc,
        "一个被保护 logical chunk 的各 stage shard 可分别位于 peer HBM、共享 DRAM 或 SSD，并由同一 manifest 组合恢复。N/Recompute 不是某个 stage shard 的物理存储层级：只要某个必要 stage 没有可提交位置，整个 logical chunk 就保持未保护状态。",
    )
    maybe_add_external_figure(doc, 3, "多层 stage-shard manifest 与连续 safe frontier")

    add_heading(doc, "4.5  GPU 故障后的确定性恢复", 2)
    add_body(
        doc,
        "检测到执行侧单 GPU fail-stop 后，系统递增 failure epoch 并 fence 旧执行。Router 激活准入时冻结的完整备份 DP；Master 从 manifest 读取不超过 safe frontier 的最长连续前缀，各 matching stage 从 HBM、DRAM 或 SSD 恢复对应 shard。未覆盖后缀使用已记录 token 状态确定性重算，随后恢复采样/输出偏移并继续请求。",
    )
    add_body(
        doc,
        "恢复全过程使用同一 route/manifest version，不在故障时跨多个 DP 临时拼接 PP stages。若版本或校验不匹配，系统回退到更早 safe frontier 或完全 Recompute，而不是提交不完整状态。",
    )
    maybe_add_external_figure(doc, 4, "单 GPU failure 后的整请求恢复链")

    add_heading(doc, "4.6  三个算法的关系与动态 alpha", 2)
    add_table(
        doc,
        ["算法", "输入与输出", "作用"],
        [
            ["Algorithm 1", "已揭示 DAG、进度与时延估计 → causal Slack", "给出请求延迟传播敏感性；不是恢复正确性的前提。"],
            ["Algorithm 2", "chunk、Slack、资源状态与故障场景 → HBM/DRAM/SSD/N", "生成 stage-aware placement 与预期净收益。"],
            ["Algorithm 3", "上一窗口的放置价值、HBM headroom 与切换代价 → per-DP alpha", "慢时间尺度调整 peer-HBM admission envelope；不设置 beta。"],
        ],
        widths_cm=[3.0, 6.6, 6.4],
        font_size=8.8,
    )
    add_body(
        doc,
        "Algorithm 3 的新 alpha 先以 budget version 发布给该 DP 的全部 stages；全部确认后才生效。收缩 alpha 时，系统先把超额 committed chunks 降级到 DRAM/SSD并更新 manifest，再释放 HBM，防止预算调整静默破坏恢复承诺。",
    )
    maybe_add_external_figure(doc, 5, "per-DP alpha 的版本化发布与安全收缩")

    add_heading(doc, "4.7  可保护技术特征与技术效果", 2)
    add_table(
        doc,
        ["技术特征", "直接技术效果", "建议权利要求层级"],
        [
            ["版本化 logical chunk + all-stage ACK + 连续 safe frontier", "排除 partial copy、旧版本和不连续空洞，形成可验证恢复点。", "独立权利要求核心"],
            ["冻结恢复 DP + 同 stage 数据通路 + 复合 manifest", "保证故障后的恢复目标与 stage 身份确定，控制面和数据面可审计。", "独立/高价值从属"],
            ["多层放置与 logical Recompute", "在恢复代价和正常路径资源成本之间作受约束选择。", "从属实施"],
            ["版本化 per-DP alpha 与先迁移后释放", "机会式利用空闲 HBM，并维持既有恢复承诺。", "从属增强"],
            ["causal Slack 价值输入", "区分不同 workflow 请求的延迟传播影响。", "可替换的从属增强"],
        ],
        widths_cm=[6.0, 6.5, 3.5],
        font_size=8.5,
    )
    add_body(
        doc,
        "初步 SGLSim 回放观察到 WCT CDF 与 retry KV reuse 的改善信号。该实验尚未覆盖完整的 all-stage 提交、无 beta 的 alpha 控制和最终目标函数，故仅用于证明方案具有可实验检验性；正式申请或对外材料不得据此宣称完整系统已经达到图中数值。",
    )
    add_evidence_figure(
        doc,
        PRELIMINARY_FIGURE,
        "初步模拟结果  SGLSim 回放中的 WCT CDF 与 retry KV reuse（非正式性能结论）",
    )

    add_heading(doc, "4.8  优选实施例：DP=2、PP=4", 2)
    for number, text in enumerate((
        "Router 将请求路由到 DP0，并冻结 DP1 为完整备份 DP；Master 建立 route version。",
        "请求每完成一个 chunk，Algorithm 2 为 PP0--PP3 分别选择 HBM、DRAM、SSD 或整体 N/Recompute。",
        "四个 matching-stage Connector 执行传输并返回 ACK；Master 仅在 4/4 ACK 后提交 chunk，并按连续顺序推进 safe frontier。",
        "DP0 任一 GPU 故障后，系统激活 DP1，按 manifest 恢复连续前缀、重算后缀，并继续整个请求。",
    ), 1):
        add_step(doc, number, text)

    add_heading(doc, "5  发散思维：变形方案、应用扩展与规避设计", 1)
    add_body(
        doc,
        "在保持“版本化对象—全 stage 提交—连续前沿—确定恢复”关系的前提下，下列实现属于可替换方案：",
    )
    add_table(
        doc,
        ["变形方向", "可替换实现"],
        [
            ["Master 形态", "单进程、主备、复制状态机或按模型分片的逻辑统一权限。"],
            ["存储层级", "CXL、持久内存、远程内存、NVMe/NVMe-oF 或对象存储。"],
            ["保护表示", "完整副本、压缩/量化副本、增量编码或纠删码；manifest 记录恢复条件。"],
            ["并行拓扑", "PP 与 TP/EP 混合；required-stage 集合扩展为全部必要 shard 集合。"],
            ["价值信号", "causal Slack 可与 deadline、业务等级或故障损失预测组合。"],
            ["alpha 表示", "比例、字节、block 或 token 容量；仍需版本化原子生效。"],
        ],
        widths_cm=[4.1, 11.9],
        font_size=8.8,
    )
    add_body(
        doc,
        "建议代理人优先评估协议组合的独立可保护性，避免把 critical path、单一评分公式或特定模块商品名写成唯一必要特征。",
    )

    add_heading(doc, "6  在先公开的现有技术文献信息", 1)
    add_body(
        doc,
        "当前仅完成代表性论文梳理，尚未完成 CNIPA、WIPO、Espacenet、Google Patents 和 USPTO 的系统检索。检索负责人为【待填写：专利代理人/检索负责人】，完成日期为【待填写：检索完成日期】。",
    )
    add_table(
        doc,
        ["文献", "与本方案的关系", "正式检索重点"],
        [
            ["Mooncake, FAST 2025", "分离式 KV 存储、全局元数据和直接传输。", "缓存对象是否包含多 stage 原子恢复语义。"],
            ["DejaVu, ICML 2024", "KV 流式复制与故障处理。", "复制完成、ACK、恢复边界与目标选择。"],
            ["vLLM, SOSP 2023", "PagedAttention 与分页 KV 管理。", "分页对象与版本化恢复对象的区别。"],
            ["SGLang, NeurIPS 2024", "RadixAttention 与结构化 LLM 执行。", "前缀复用和多阶段故障语义。"],
            ["Agentix, NSDI 2026", "把 agent program 作为服务对象并优化其执行。", "workflow 语义输入与容错保护决策的边界。"],
            ["KVFlow, NeurIPS 2025", "基于 agent-step graph 的前缀缓存与预取。", "复用价值与故障恢复价值是否存在不同提交条件。"],
            ["Pythia, arXiv 2026", "利用 agent workflow 可预测性辅助 serving。", "预测信号是否涉及版本化恢复承诺。"],
            ["RaidServe, MLSys 2026", "弹性并行、KV 备份和故障恢复。", "备份目标、stage 一致性与正常路径约束。"],
            ["GhostServe, MLSys 2026", "shadow checkpointing 与轻量恢复。", "checkpoint 完整性、版本和恢复前沿。"],
            ["LUMEN, arXiv 2026", "协调 checkpoint、请求调度与容量恢复。", "跨请求协调与本方案 logical chunk 提交协议的差异。"],
            ["AnchorTP, DATE 2026", "状态保持的弹性张量并行恢复。", "并行拓扑变化与 PP-stage manifest 的边界。"],
            ["Tarragon / ReviveMoE, 2026", "面向 MoE 推理的硬件故障恢复。", "专家/并行状态与 KV 连续前沿的差异。"],
        ],
        widths_cm=[4.0, 6.2, 5.8],
        font_size=8.6,
    )

    doc.add_page_break()
    add_heading(doc, "附录 A  提交前 Checklist", 1)
    for item in (
        "补齐封面和第 6 章的全部 12 个高亮字段；",
        "由代理人完成专利族、优先权日和逐项技术特征检索；",
        "确认发明人贡献、申请主体、华为合作权属和最早证据；",
        "按代理人要求统一附图编号，但不要在图内重复文档题注；",
        "论文投稿、开源或公开演示前确认申请时序与保密要求；",
        "实验结果仅用于后续效果佐证，不把尚未验证的数值写入交底书。",
    ):
        add_bullet(doc, item)


def build_superseded_latest_design_disclosure(doc):
    """Build the latest HBM-first write-back disclosure in the template."""
    add_heading(doc, "缩略语和关键术语", 1)
    add_heading(doc, "缩略语", 2)
    add_table(
        doc,
        ["中文表述/中文术语", "完整英文表述", "英文缩略语"],
        [
            ["键值缓存", "Key-Value Cache", "KVCache / KV cache"],
            ["大语言模型", "Large Language Model", "LLM"],
            ["图形处理器", "Graphics Processing Unit", "GPU"],
            ["高带宽存储器", "High Bandwidth Memory", "HBM"],
            ["动态随机存取存储器", "Dynamic Random-Access Memory", "DRAM"],
            ["有向无环图", "Directed Acyclic Graph", "DAG"],
            ["关键路径法", "Critical Path Method", "CPM"],
            ["梯度提升决策树", "Gradient Boosting Decision Tree", "GBDT"],
            ["工作流完成时间", "Workflow Completion Time", "WCT"],
            ["数据/张量/流水线并行", "Data/Tensor/Pipeline Parallelism", "DP/TP/PP"],
            ["加性增、乘性减", "Additive Increase Multiplicative Decrease", "AIMD"],
        ],
        widths_cm=[4.5, 7.5, 4.0],
        font_size=8.8,
        style_name="Normal Table",
    )

    add_heading(doc, "关键术语", 2)
    terms = [
        ["因果 Slack", "只依据当前已揭示的残余 Workflow DAG、已观察进度和剩余时长预测计算的请求可容忍附加延迟；未知结构或缺失预测时取零。"],
        ["KV chunk", "按连续 token 区间封装的保护单元，包含恢复该区间所需的全部适用 TP/PP shards；open tail 和未提交 chunk 在故障后重算。"],
        ["恢复损失", "在当前时刻立即发生故障时，最快可用恢复路径超过请求剩余 Slack 的部分，用于统一评价正常期保护和故障期恢复顺序。"],
        ["分层惰性重定优先级", "把优先级分为请求级动态项和 chunk 级稳定项；Slack 变化只更新请求描述符及其当前 victim frontier，不扫描或搬迁该请求全部 KV。"],
        ["两级 victim 索引", "每个请求内部维护 suffix chunk 队列；每个资源池的全局最小堆至多暴露每个活动请求的一个当前可安全淘汰 frontier。"],
        ["写回式 Peer-HBM 保护缓存", "前台优先把 KV 批量复制到 Router 固定的目标 Worker HBM，目标进程接管幂等写回任务，并在服务空闲窗口把脏数据批量写入共享 DRAM Store。"],
        ["HBM_ONLY", "已提交的脏状态：目标 HBM 中存在可恢复副本，DRAM 写回尚未完成；普通容量压力不得直接淘汰。"],
        ["HBM_AND_DRAM", "已提交的干净状态：同一 chunk 在目标 HBM 和共享 DRAM 中均有兼容副本；可优先回收 HBM 而保留 DRAM。"],
        ["DRAM_ONLY", "目标 HBM 已回收但共享 DRAM 副本仍有效的状态，可跨兼容恢复目标加载。"],
        ["提交状态", "物理动作依次处于 PENDING、COMMITTED 或 ABORTED；只有全部 shards、route version、模型版本、KV ABI 和布局匹配的 COMMITTED 记录可用于恢复。"],
        ["GuardedAIMD", "仅调节每个 Peer-HBM target 的保护配额 alpha；压力出现时快速乘性收缩，连续稳定若干窗口后缓慢加性增长。"],
        ["silent replay", "故障后目标 Worker 恢复已提交 KV、重算未覆盖后缀并追赶到 Router 已提交的输出位置，在完成 fence 之前不向用户重复输出。"],
    ]
    add_table(doc, ["术语", "本文含义"], terms,
              widths_cm=[4.2, 11.8], font_size=8.7)

    add_heading(doc, "相关技术背景以及最接近的现有技术", 1)
    add_heading(doc, "背景技术", 2)
    add_body(
        doc,
        "智能体工作流由多轮 LLM request、tool call、分支和汇合构成，通常可表示为随执行逐步揭示的 DAG。应用关注的是整个工作流完成时间，而不是某一个请求的孤立延迟。旧 PPT 所示 coding 与 deep-research 场景说明：并行分支的持续时间和依赖位置不同，相同规模 KV 丢失造成的重算延迟可能被其他分支隐藏，也可能直接传播到 WCT。",
    )
    add_body(
        doc,
        "LLM 推理为每个已处理 token 保存逐层 Key/Value 状态。上下文越长，GPU fail-stop 后从输入重新计算已完成前缀的代价越大；但无差别同步复制全部 KV 又会占用稀缺 HBM、DRAM 和传输带宽，并干扰健康请求。因此需要同时回答三个系统问题：哪个请求更紧急、哪些连续 KV chunk 值得保护、保护缓存最多可以占用多少目标 HBM。",
    )
    add_body(
        doc,
        "本发明不把 Slack 变化等同于物理重排。工作流进度可以高频改变请求价值，而 HBM 到 DRAM 的写回和跨 Worker 复制具有真实数据移动成本。若每次 Slack 更新都重算并搬迁全部 chunk，会造成控制面扫描、复制抖动和前台干扰；若仅维护静态 HBM 副本，又不能在正常服务压力上升时安全回收。",
    )

    add_heading(doc, "与本发明相关的现有技术", 2)
    add_heading(doc, "现有技术的技术方案", 3)
    add_table(
        doc,
        ["公开方向", "主要公开能力", "需要进一步区分的技术边界"],
        [
            ["vLLM / SGLang", "PagedAttention、连续批处理、前缀复用和结构化 LLM 程序执行。", "缓存命中和程序级调度不当然等于由因果 Slack 驱动的故障保护、写回驻留状态和安全收缩。"],
            ["Mooncake", "分离式 KVCache、全局元数据、CPU DRAM/SSD 利用和直接传输。", "通用缓存调度不当然披露请求级/块级优先级分解、目标 HBM 写回任务和 clean-first 故障保护回收。"],
            ["DéjàVu", "KV-cache streaming、状态复制和故障处理。", "需比较其复制/恢复边界与本发明的 workflow Slack、惰性 victim 索引、三态写回缓存和 alpha 收缩。"],
            ["GhostServe", "以 host-memory parity shard 保护流式 KV 并在设备故障后重建。", "编码式 checkpoint 与本发明 HBM-first 完整副本、DRAM 写回及基于 Slack 的在线 victim 选择不同。"],
            ["Agentix", "将 agent program 作为一级服务对象并用程序上下文调度 LLM calls。", "程序级调度是否进入底层 GPU 故障保护、residency 状态和资源压力闭环需单独核对。"],
            ["Landlord", "对不同大小和取回成本的在线缓存对象维护 credit/rent 并选择 victim。", "经典单层文件缓存不直接解决 HBM/DRAM 写回状态、故障恢复损失、迁移成本和目标 HBM 配额。"],
        ],
        widths_cm=[3.0, 6.0, 7.0],
        font_size=8.4,
    )

    add_heading(doc, "现有技术的缺点", 3)
    for item in (
        "请求级缓存热度或大小不能反映故障延迟是否会沿智能体依赖传播到 WCT；",
        "逐 chunk 重新计算优先级容易在 Slack 高频变化时扫描整个请求或资源池，并把逻辑排序误实现为物理迁移；",
        "仅记录 HBM 或 DRAM 中存在字节，无法区分脏 HBM、已有 DRAM backing 的干净 HBM 以及只剩 DRAM 的状态；",
        "没有写回成本和 victim 保留损失的准入比较时，系统可能产生 HBM 与 DRAM 间的低价值往返；",
        "静态 HBM 保护区会挤压前台推理，而瞬时硬收缩又可能直接丢弃尚未写回的故障保护状态；",
        "物理复制、Controller commit、目标任务接管、请求结束和故障 fence 若缺少一致顺序，源 Worker 在窄窗口失效时可能留下不可验证状态。",
    ):
        add_bullet(doc, item)

    add_heading(doc, "本发明技术方案的详细阐述", 1)
    add_heading(doc, "本发明所要解决的技术问题", 2)
    for item in (
        "将当前已揭示的智能体工作流状态转换成可由推理控制层消费的请求剩余 Slack；",
        "在 Slack 变化时只更新元数据优先级，避免对请求全部 KV chunk 进行重排或迁移；",
        "在真实写回成本下在线接纳 Peer-HBM/DRAM 保护并选择满足连续前缀约束的 victim set；",
        "以三态 write-back residency 保证脏 HBM 不被压力丢弃，同时优先回收已有 DRAM backing 的干净 HBM；",
        "在前台显存压力变化时安全调整 Peer-HBM 配额，并在 GPU fail-stop 后按已提交状态完成恢复。",
    ):
        add_bullet(doc, item)

    add_heading(doc, "本发明的主要要点概述", 2)
    add_table(
        doc,
        ["算法/机制", "输入与输出", "技术作用"],
        [
            ["Algorithm 1：Causal Slack", "已揭示残余 DAG、执行摘要、剩余时长特征 → 每个 active branch 的 Slack", "把工作流级延迟传播关系转换成请求级、可解释的时间预算。"],
            ["Algorithm 2：SlackLandlord", "Slack、chunk、当前已提交前缀、资源状态 → bypass 或 admission/victim set", "通过分层惰性重定优先级和局部 credit/rent 管理 HBM/DRAM chunk；Slack 更新不搬数据。"],
            ["写回式 HBM 保护缓存", "HBM-first batch copy、目标任务、后台 Store write → 三态 residency", "把低时延故障恢复与可回收 DRAM backing 解耦，并显式计入写回成本。"],
            ["Algorithm 3：GuardedAIMD", "目标 Worker 的 serving HBM pressure 与稳定窗口 → alpha 配额", "压力时快速让位于前台推理，稳定时缓慢增加保护容量；只控制 HBM，不设置 DRAM 比例。"],
        ],
        widths_cm=[4.2, 6.3, 5.5],
        font_size=8.5,
    )
    add_body(
        doc,
        "三个算法共享同一个 fail-now recovery model：正常期用它形成 chunk 保护收益，故障期用剩余 Slack 排列恢复 due time。GBDT、CPM、Landlord 和 AIMD 分别属于可替换的基础构件；本发明的重点是上述跨层输入、惰性索引、写回状态机、资源保护和恢复 fence 的组合关系。",
    )

    add_heading(doc, "本发明可应用的系统架构或场景", 2)
    add_body(
        doc,
        "系统包括四类组件：（1）Workflow Runtime 维护已揭示 DAG 并运行 Algorithm 1；（2）Router 为每个 route version 绑定源 Worker、兼容的 failover target、模型与 KV 布局，并负责故障 fence；（3）Aegis Controller 作为 Store Master/KV Directory 的策略扩展，维护 SlackLandlord、residency、commit state 和 GuardedAIMD；（4）Worker/Connector seal KV chunk、执行 Peer-HBM/Store batch transfer、目标侧写回和恢复加载。Controller 传递元数据和决定，KV tensor 不经过 Controller。",
    )
    add_body(
        doc,
        "优选实验实施为 Qwen3-8B、TP=1、PP=1、DP=4：四个独立 Worker 分别使用一张 GPU。Router 为每个请求指定一个源 Worker及一个兼容目标 Worker；源 Worker 把 logical chunk 的全部物理 keys 一次批量复制到目标 HBM。该配置用于说明而不限制本发明；TP、PP、EP 或多节点部署只需把完整 chunk 扩展为恢复所需的全部 shards。",
    )
    maybe_add_external_figure(doc, 1, "AegisKV 三层架构及控制面、Peer-HBM 与 DRAM 数据路径")

    add_heading(doc, "本发明涉及的产品、可能的售卖形态以及侵权取证方式", 2)
    add_table(
        doc,
        ["产品/售卖形态", "可交付功能", "可依法取得的技术证据示例"],
        [
            ["LLM 推理平台或云服务", "工作流感知容错、跨 Worker KV 保护和故障恢复。", "公开 API/配置、故障演示、监控指标、资源配额变化和恢复结果。"],
            ["Aegis Controller 软件模块", "SlackLandlord、两级 victim 索引、三态 residency、提交与回收。", "管理接口、状态快照、事件日志、二进制符号、部署文档或经授权测试结果。"],
            ["Worker Connector/Agent", "Peer-HBM batch copy、目标任务接管、后台 Store write-back 和恢复 load。", "网络调用顺序、GPU/DRAM 流量、任务状态、Store key 生命周期与失败注入轨迹。"],
            ["软硬件一体机", "预装 Router、Store Master、Transfer Engine 和 HBM resource enforcer。", "产品手册、配置文件、性能计数器、故障恢复行为和采购交付物。"],
        ],
        widths_cm=[3.7, 6.0, 6.3],
        font_size=8.4,
    )
    add_body(
        doc,
        "取证应遵守合同、隐私、计算机系统访问和反不正当竞争等适用规则。单一性能现象通常不足以证明采用本方案，较强证据是同一产品中同时出现：请求 Slack metadata、每请求 frontier 与全局 victim heap、HBM_ONLY/HBM_AND_DRAM/DRAM_ONLY 转换、target-owned write-back、alpha 压力收缩以及故障恢复 fence。",
    )

    add_heading(doc, "本发明技术实现的具体实施例", 2)
    add_heading(doc, "实施例一：因果剩余时间预测与 Slack 跨层发布", 3)
    for number, text in enumerate((
        "Application 在 LLM request 或 tool call 返回时更新执行摘要，只把当前已经观察到的节点、边、状态和输出统计写入残余 DAG；尚未揭示的动态节点不进入本轮。",
        "GBDT 根据节点类型、已完成轮次、当前 prompt/output、工具类型、历史同类时长和负载等因果可见特征预测 agent node 的剩余时间；fork/join 等结构节点权重为零。",
        "对残余 DAG 生成拓扑序，前向计算 ES，反向计算 LS，并对 active branch i 计算 s_i=max(LS_i-ES_i,0)。单层 fan-out/join 时，该值退化为最长预测分支与本分支剩余时间之差。",
        "若下一 join 的残余结构不完整或关键 metadata 缺失，则 Valid=false 且 Slack 取零；随后把 workflow、agent、request、Slack 值、观测时间和版本附在下一次请求上。",
        "Controller 在请求执行期间仅按经过时间衰减剩余 Slack，不在 token 路径重建 DAG；新的 request/tool return 覆盖旧预测。",
    ), 1):
        add_step(doc, number, text)
    maybe_add_external_figure(doc, 2, "因果残余 DAG、GBDT、CPM 与请求 Slack 发布流程")

    add_heading(doc, "实施例二：分层惰性重定优先级与 SlackLandlord", 3)
    add_body(
        doc,
        "对请求 r 的 chunk c，将逻辑优先级表示为 P(r,c)=P_request(r)+P_chunk(r,c)。P_request 由剩余 Slack、请求当前恢复路径和 failure-now harm 决定；P_chunk 保存 ordinal、真实字节数、恢复贡献、提交 epoch 和稳定的 credit/rent 信息。Slack 变化只更新 P_request 及该请求当前 suffix frontier 的索引键。",
    )
    add_body(
        doc,
        "每个请求分别维护 HBM 和 DRAM 连续 prefix 及各自 suffix queue；每个资源池的全局最小堆至多暴露每个活动请求的一个当前可安全淘汰 tail。隐藏 chunk 不因每次 Slack 变化重新入堆，只有当前 frontier 被选中或前一 tail 移除后才按最新 epoch 懒刷新。因此一次普通 Slack 更新约为 O(log R)，其中 R 为活动请求数；释放 k 个实际 victim 只处理被暴露的 k 个 chunk 及少量 stale top。",
    )
    add_body(
        doc,
        "设请求当前已提交集合为 S，最快故障恢复时间为 T(S)，剩余 Slack 为 s，则恢复损失 J(S)=max(T(S)-s,0)。候选 x 的收益 A_x=max(J(S)-J(S+x),0)；HBM 候选还扣除预计 DRAM write-back 成本。空间不足时，Controller 从全局 heap 构造满足 suffix 和 residency 条件的 victim set Y，只有 A_x 大于删除 Y 的联合恢复损失且候选 Landlord threshold 高于各 victim key 才接纳，否则 bypass。Landlord 在此仅是单个目标 pool 的局部 victim 机制，不被描述为未知未来下的全局多层最优解。",
    )
    add_body(
        doc,
        "Slack update、请求优先级变化和 heap key 更新均为 metadata operation，不触发 HBM/DRAM copy。物理动作仅由新 chunk admission、实际容量压力、Algorithm 3 的 alpha 收缩、请求完成或后台清洗窗口触发。",
    )
    maybe_add_external_figure(doc, 3, "请求级/块级优先级分解、两级 victim 索引与惰性物理执行")

    add_heading(doc, "实施例三：HBM-first 写回式保护缓存与故障恢复", 3)
    for number, text in enumerate((
        "Worker seal 一个完整 logical chunk 后，Controller 先在 Router 固定的目标 Peer-HBM pool 内执行 Algorithm 2 admission；收益计算已经计入未来写回 DRAM 的成本。",
        "admission 通过后，源 Worker 把该 chunk 的全部 physical Store keys 作为一个 batch 写入目标 HBM slots。目标进程先持久接管幂等 write-back task，源 Worker 再 ACK HBM copy；版本和全部 shards 均匹配后形成可恢复的 HBM_ONLY。",
        "目标 cleaner 等待 HBM commit，立即为同 ordinal 的 DRAM write-back 建立 PENDING metadata，使请求结束 fence 能看到该后台动作；随后才等待 serving-idle window，并直接从 target-owned HBM slots 批量写入 Store。",
        "Store write 与 Controller completion 成功后，residency 原子变为 HBM_AND_DRAM。容量压力可删除其 HBM mapping并变为 DRAM_ONLY；对仍为 HBM_ONLY 的 dirty line，普通压力只能等待 cleaner，不能直接丢弃。",
        "若 HBM admission、物理复制或任务交接在 commit 前失败，源 buffers 尚存时执行一次前台 batch Store PUT，直接得到 DRAM_ONLY；失败或版本不匹配的 PENDING 记录变为 ABORTED。",
        "请求完成时，Router 等待已接纳的 PENDING write-back 进入 COMMITTED 或 ABORTED，再在同一终止 fence 中回收 Controller entries、Store keys 和 HBM slots。",
        "源 Worker fail-stop 后，Router fence 旧输出并激活 route-bound target。HBM_ONLY/HBM_AND_DRAM 从目标 HBM 批量恢复，DRAM_ONLY 从 Store 加载，未提交或未覆盖后缀重算；目标 silent replay 到已提交输出位置后才恢复输出权限。",
    ), 1):
        add_step(doc, number, text)
    maybe_add_external_figure(doc, 4, "HBM-first 写回、三态 residency、clean-first 回收与故障恢复")

    add_heading(doc, "实施例四：GuardedAIMD 的压力安全 HBM 配额控制", 3)
    add_body(
        doc,
        "对每个 Peer-HBM target p，设可由保护缓存使用的比例为 alpha_p，总 HBM 容量为 M_p，则 shadow-KV quota 为 alpha_p·M_p。Worker 的 primary allocator 和 serving reserve 始终具有最终优先权；Algorithm 3 不选择 chunk，也不控制 DRAM pool。",
    )
    add_body(
        doc,
        "每个控制窗口观察 primary headroom、allocation/reclaim stall 等 HBM pressure。若 q_p=1，则 alpha_p 立即更新为 max(alpha_min, gamma·alpha_p)，其中 0<gamma<1；若连续 K 个窗口无压力，则更新为 min(alpha_max, alpha_p+delta)；其他情况保持不变。一次窗口只遍历 targets，时间和状态均为 O(P)。",
    )
    add_body(
        doc,
        "当新 quota 低于当前 shadow usage 时，Local Resource Enforcer 先禁止新 HBM admission，并通知 Controller 执行 clean-first 回收：优先删除 HBM_AND_DRAM 的 suffix tail；剩余 HBM_ONLY 交给 cleaner 写回后再回收。若当前窗口仍不能达到水位，系统报告 quota_satisfied=false 并逐步收缩，绝不为瞬时满足 alpha 而丢弃 dirty data。",
    )
    maybe_add_external_figure(doc, 5, "GuardedAIMD 的 alpha 更新、clean-first 收缩与 dirty 保护")

    add_heading(doc, "本发明的主要创新点及其对应的技术效果", 2)
    innovations = [
        ["1. 因果 Slack 跨层契约与统一 fail-now recovery model（重要组合）", "正常期和故障期使用同一可解释时间量，按工作流延迟传播而不是孤立请求热度评价 KV；缺失信息时 zero-Slack fail-close。"],
        ["2. 请求级动态项与 chunk 级稳定项分解，以及每请求一个 frontier 的两级 lazy victim 索引（核心必要）", "Slack 变化只更新少量 metadata，避免扫描所有 resident chunks 或触发整个 KV 分布迁移，同时保持 suffix/连续前缀约束。"],
        ["3. HBM-first、target-owned task 与 HBM_ONLY/HBM_AND_DRAM/DRAM_ONLY 三态 write-back cache（核心必要）", "前台快速形成可恢复 HBM 副本，后台建立可跨目标使用的 DRAM backing；显式区分 dirty/clean，防止压力误删。"],
        ["4. 同时考虑新 chunk 收益、victim 联合损失和必要写回成本的 SlackLandlord admission（重要实施）", "只在净恢复价值为正时替换，并优先淘汰已有 DRAM backing 的低价值 HBM，减少无效往返和故障损失。"],
        ["5. 只控制 Peer-HBM 的 GuardedAIMD 与 clean-first 渐进收缩（从属增强）", "前台压力出现时快速释放保护额度，稳定时缓慢恢复；收缩水位不会直接破坏 dirty KV，DRAM 继续由 Store 管理。"],
        ["6. 目标任务接管、Controller commit、请求结束与 failure fence 的顺序约束（正确性增强）", "关闭源故障窗口；PENDING 永不用于恢复，COMMITTED 状态与实际物理副本、版本和恢复 activation 可审计地一致。"],
    ]
    add_table(doc, ["本发明的主要创新点", "对应的有益效果"], innovations,
              widths_cm=[7.2, 8.8], font_size=8.4)
    add_body(
        doc,
        "建议的独立保护主线是：接收工作流 Slack metadata；在 chunk seal 时通过两级惰性 victim 索引和写回成本进行 HBM-first admission；由目标进程接管写回任务并维护三态 residency；在压力窗口调节 HBM quota；在故障 fence 后仅使用已提交状态恢复。具体必要特征和权利要求层级仍须由代理人结合正式检索确定。",
    )

    add_heading(doc, "发散思维：", 1)
    add_heading(doc, "变形方案、应用扩展与规避设计", 2)
    add_table(
        doc,
        ["变形方向", "可替换或扩展实现"],
        [
            ["价值信号", "因果 Slack 可与 deadline、SLO、业务等级或预测故障损失组合；只要仍驱动请求级描述符和恢复 due time。"],
            ["优先级索引", "全局 indexed heap 可替换为 tournament tree、分桶队列或等价有序索引；仍只暴露每请求有限 frontier 并惰性刷新。"],
            ["存储层级", "共享 DRAM 可扩展到 CXL memory、持久内存、NVMe/NVMe-oF、远程内存或对象存储；最下层由 Store policy 管理。"],
            ["保护表示", "完整 KV 副本可采用压缩、量化、增量编码或纠删码；residency 与 commit 必须记录对应恢复条件。"],
            ["并行拓扑", "TP=1/PP=1/DP=4 可扩展到 TP、PP、EP/MoE 和多节点；logical chunk 的 commit 要求全部必要 shards 完整。"],
            ["写回调度", "固定 idle delay 可替换为带宽 token、I/O 优先级、队列长度或预测空闲窗口；target-owned、幂等和 request-completion 可见性保持。"],
            ["配额表示", "alpha 比例可替换为字节、page、block 或 token 容量；压力下降、稳定增长和 dirty-safe 收缩关系保持。"],
            ["控制器部署", "Controller 可单机、主备、复制状态机或按模型/租户分片；只要同一 pool 的 admission、capacity 和 victim metadata 顺序一致。"],
            ["离线规划", "固定已观测窗口可使用最小费用流等工具作为 oracle 或慢速诊断；不得把它表述为未知未来 workload 的在线全局最优保证。"],
        ],
        widths_cm=[4.0, 12.0],
        font_size=8.4,
    )

    add_heading(doc, "在先公开的现有技术文献信息", 1)
    add_body(
        doc,
        "本稿仅完成代表性一手论文核验，尚未完成 CNIPA、WIPO Patentscope、Espacenet、Google Patents 和 USPTO 的系统专利检索。检索负责人为【待填写：专利代理人/检索负责人】，完成日期为【待填写：检索完成日期】。正式申请前应继续检索论文引用链、作者和机构名下专利族、优先权日及抵触申请，并进行逐项技术特征对比。",
    )
    references = [
        ["1", "Mooncake；USENIX FAST 2025", "2025-02", "https://www.usenix.org/conference/fast25/presentation/qin", "分离式 KVCache、全局缓存调度、CPU DRAM/SSD/NIC 利用和直接传输。"],
        ["2", "DéjàVu；ICML/PMLR 2024", "2024-07", "https://proceedings.mlr.press/v235/strati24a.html", "KV-cache streaming、状态复制和 fault-tolerance。"],
        ["3", "vLLM / PagedAttention；SOSP 2023", "2023-10", "https://doi.org/10.1145/3600006.3613165", "分页 KV 管理、调度和内存共享。"],
        ["4", "SGLang；NeurIPS 2024", "2024", "https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html", "结构化 LLM program 执行和 RadixAttention。"],
        ["5", "Agentix；USENIX NSDI 2026", "2026-05", "https://www.usenix.org/conference/nsdi26/presentation/luo", "把 agent program 作为一级对象，并把程序上下文传入调度。"],
        ["6", "GhostServe；MLSys 2026", "2026", "https://proceedings.mlsys.org/paper_files/paper/2026/hash/d99e8e80a6c41e148db686918dd7eab3-Abstract-Conference.html", "host-memory parity shard、流式 checkpoint 与故障重建。"],
        ["7", "On-Line File Caching / Landlord；Algorithmica 2002", "2002", "https://www.cs.ucr.edu/~neal/Young02Online", "不同大小与取回成本对象的在线 credit/rent 缓存替换。"],
    ]
    add_table(
        doc,
        ["序号", "期刊/会议及文献", "公开日期", "文档来源", "与本方案相关内容"],
        references,
        widths_cm=[0.8, 3.5, 1.8, 5.1, 4.8],
        font_size=7.6,
    )

    doc.add_page_break()
    add_heading(doc, "专利交底书 Checklist", 1, numbered=False)
    checklist = [
        ["1", "模板第 2--6 章及创新点表是否完整？", "已覆盖", "使用原始 DOCX 的 styles、numbering、theme、section geometry 和表格样式生成。"],
        ["2", "是否完整描述三个算法及职责边界？", "已覆盖", "Algorithm 1 在 Application；Algorithm 2/3 在 Controller；Worker 执行物理 copy、cleaner 和 recovery。"],
        ["3", "是否区分逻辑重定优先级与物理迁移？", "已覆盖", "Slack update 只更新 request descriptor/frontier；物理动作仅由 admission、pressure、alpha contraction、completion 或 cleaner 触发。"],
        ["4", "是否使用最新 write-back residency？", "已覆盖", "正文和附图统一使用 HBM_ONLY、HBM_AND_DRAM、DRAM_ONLY，不保留旧 DRAM prerequisite/dependent-tier 语义。"],
        ["5", "Algorithm 3 是否仅控制 HBM alpha？", "已覆盖", "正文不设置 beta；DRAM 由 Store 管理；alpha 收缩 clean-first 且不丢 dirty line。"],
        ["6", "实施例是否说明 What/When/Where/How？", "已覆盖", "四个实施例分别说明 Slack、lazy index、target-owned write-back/recovery 和 GuardedAIMD。"],
        ["7", "是否提供黑白可编辑附图？", "已覆盖", "附图 1--5 由同一 draw.io 多页源生成，并嵌入无拉伸 PNG。"],
        ["8", "是否避免把未验证实验写成效果承诺？", "已覆盖", "旧模拟 WCT、L1/L2 和模型化恢复时间未写入新版正文。"],
        ["9", "现有技术是否采用可核验一手来源？", "部分完成", "已核验 7 项代表性论文页面；专利数据库和专利族检索仍待代理人完成。"],
        ["10", "发明人、主体、权属和时间字段是否完整？", "待完成", "保留 12 个黄色高亮字段，正式提交前逐项补齐并核对证据。"],
        ["11", "是否完成研发复核和代理人复核？", "待完成", "研发应复核状态转换与故障 fence；代理人应复核新颖性、创造性、单一性和权利要求范围。"],
    ]
    add_table(doc, ["序号", "检查项", "状态", "本稿对应内容/待办"], checklist,
              widths_cm=[0.9, 5.3, 2.0, 7.8], font_size=8.2)
    add_heading(doc, "正式提交前待办", 2, numbered=False)
    for item in (
        "补齐全部 12 个高亮字段，并由合作双方确认发明人贡献、申请主体、权属和关键日期；",
        "由专利代理人完成专利数据库、专利族、抵触申请和逐项技术特征检索；",
        "由研发人员逐项复核 request completion fence、target-owned task、residency eligibility 和 failure recovery；",
        "根据代理人制图规范统一附图部件编号，并在 Microsoft Word 中复核分页、字体、页码域和表格跨页；",
        "申请前继续保密，论文投稿、代码开源或公开演示时间须与申请策略协调。",
    ):
        add_bullet(doc, item)


def build_concise_disclosure(doc):
    """Fill only the sections and tables present in the supplied template."""

    # 2. 缩略语和关键术语
    add_heading(doc, "缩略语和关键术语", 1, template_num_level=0)
    add_heading(doc, "2.1缩略语：", 2, numbered=False)
    add_table(
        doc,
        ["中文表述/中文术语", "完整的英文表述/英文准确用语", "英文缩略语"],
        [
            ["键值缓存", "Key-Value Cache", "KVCache"],
            ["大语言模型", "Large Language Model", "LLM"],
            ["图形处理器", "Graphics Processing Unit", "GPU"],
            ["高带宽存储器", "High Bandwidth Memory", "HBM"],
            ["动态随机存取存储器", "Dynamic Random-Access Memory", "DRAM"],
            ["有向无环图", "Directed Acyclic Graph", "DAG"],
            ["关键路径法", "Critical Path Method", "CPM"],
            ["梯度提升决策树", "Gradient Boosting Decision Tree", "GBDT"],
            ["工作流完成时间", "Workflow Completion Time", "WCT"],
            ["数据并行", "Data Parallelism", "DP"],
            ["张量并行", "Tensor Parallelism", "TP"],
            ["流水线并行", "Pipeline Parallelism", "PP"],
            ["加性增、乘性减", "Additive Increase Multiplicative Decrease", "AIMD"],
        ],
        widths_cm=[4.5, 7.5, 4.0],
        font_size=8.8,
        style_name="Normal Table",
    )

    add_heading(
        doc,
        "2.2关键术语（如有涉及内部术语或自造词，请给出解释）：",
        2,
        numbered=False,
    )
    definitions = [
        (
            "残余工作流 DAG",
            "在当前时刻已经出现、但尚未完成的工作流节点和依赖关系；尚未运行到的动态分支不进入本轮计算。",
        ),
        (
            "因果 Slack",
            "某活动分支在不推迟整个工作流完成时间的前提下还可容忍的时间余量。其仅由当前可见信息计算；信息不完整时取零。",
        ),
        (
            "逻辑 KV 分块",
            "把一段连续 token 及恢复所需的全部 KV 数据视为一个不可拆分的备份、确认和恢复单元。",
        ),
        (
            "工作流关键性",
            "某个活动分支对整个工作流完成时间的影响程度。可容忍等待越少，关键性越高。",
        ),
        (
            "KV 备份优先级",
            "以立即故障时可避免的工作流恢复损失为收益，并按分块实际字节数归一化得到的接纳和淘汰依据。",
        ),
        (
            "两级候选索引",
            "每个资源池内，每个请求只暴露当前可安全释放的连续尾部分块；全局索引只比较这些尾部候选，其余分块暂不参与全局排序。",
        ),
        (
            "分层独立决策",
            "路由器先固定 Peer-HBM 目标；高速显存和主存分别使用本层容量、收益和淘汰候选判断是否接纳。结果可以是仅 HBM、仅 DRAM、两者都有或两者都没有。",
        ),
        (
            "源侧双路径传输",
            "源 Worker 分别通过 D2D/P2P 向目标 HBM 复制、通过 D2H 向 DRAM 传输。两条传输均由源 Worker 发起，不经过对方中转，并分别完成校验和确认。",
        ),
        (
            "AIMD 保护额度",
            "只规定每个目标 Worker 可用于 Peer-HBM 保护的额度：有前台压力时按比例快速降低，连续稳定时按固定步长缓慢增加；具体分块仍由存储控制模块选择。",
        ),
    ]
    for term, explanation in definitions:
        add_body(doc, f"{term}：{explanation}", bold_prefix=f"{term}：")

    # 3. 相关技术背景以及最接近的现有技术
    add_heading(
        doc,
        "相关技术背景以及最接近的现有技术",
        1,
        template_num_level=0,
    )
    add_heading(doc, "3.1　背景技术：", 2, numbered=False)
    add_body(
        doc,
        "智能体工作流通常由多轮大模型请求、工具调用以及动态产生的分支和汇合组成。工作流完成时间不等于单个请求时延之和；在 fan-out/join 结构中，较慢分支会延迟后续汇合，因此不同分支上的同等资源损失可能产生不同的工作流影响。此前 PPT 中关于并行子任务、失败位置与 WCT 差异的内容在本稿中仅作为这一问题背景，不采用其中旧版缓存层级或未经完整系统验证的性能数值。",
    )
    add_body(
        doc,
        "KVCache 可理解为大模型为已经读过的 token 保存的“已计算中间记忆”。它能避免重复计算，却会占用大量 GPU 显存；若承载它的 Worker 故障且没有有效副本，系统只能重算相应前缀。分布式推理系统因此会在 HBM、主机 DRAM 或远端存储之间保存和恢复 KVCache。",
    )
    add_body(
        doc,
        "用于保护 KVCache 的 HBM 与前台推理共享同一显存。保护过少会扩大故障后的恢复和重算，保护过多又会挤压模型执行、批处理及新请求所需空间。该矛盾要求控制面同时理解工作流紧迫度、分块恢复贡献、复制及后备建立成本、目标 Worker 压力和可验证的提交状态。",
    )

    add_heading(doc, "3.2  与本发明相关的现有技术", 2, numbered=False)
    add_heading(doc, "3.2.1  现有技术的技术方案", 3, numbered=False)
    add_body(
        doc,
        "第一类技术面向 GPU 内存和 KVCache 利用率。vLLM/PagedAttention 以分页方式减少显存碎片并共享缓存；SGLang/RadixAttention 以基数树复用公共前缀；H2O 等方法根据 token 的注意力贡献保留较重要的 KV。此类工作主要回答“怎样少占显存、怎样提高复用”。",
    )
    add_body(
        doc,
        "第二类技术面向应用和请求调度。Autellix 把智能体程序作为一等调度对象，利用程序上下文安排不同大模型调用；Parrot、SGLang 等系统也利用程序结构或语义变量提高执行效率。此类工作主要回答“先执行哪个调用”。",
    )
    add_body(
        doc,
        "第三类技术面向分离式服务和多层存储。DistServe 将 prefill 与 decode 分配到不同 GPU；Mooncake 使用 CPU DRAM、SSD 和网络构建分离式 KVCache 存储；LMCache 在多个推理引擎之间提供 KV 的保存、查找、传输和复用。此类工作主要回答“KV 放在哪里、怎样传得更快”。",
    )
    add_body(
        doc,
        "第四类技术面向迁移和容错。Llumnix 通过请求状态迁移改善多实例负载均衡，DéjàVu 通过 KV 流式传输和状态复制缩短故障恢复。典型流程是在故障后隔离旧实例，从已有副本恢复可用状态，并重新计算没有副本的部分。",
    )

    add_heading(doc, "3.2.2  现有技术的缺点", 3, numbered=False)
    for number, text in enumerate(
        (
            "Autellix 等工作流调度方法能够看到调用关系，却主要决定先执行哪个调用；vLLM、SGLang、H2O 等 KV 管理方法主要依据内存、复用或 token 重要性。两类方法之间缺少直接联系，因而不能回答：某一块 KV 丢失后，会让整个工作流多等待多久。",
            "Mooncake、LMCache、DéjàVu 等系统已经能够保存、传输或复制 KV，但通常分别关注数据通路、缓存复用或故障复制，尚不能依据同一 KV 对故障恢复的实际收益，分别判断高速显存副本和主存副本是否值得占用各自资源，也缺少统一、可验证的副本可用边界。",
            "vLLM、SGLang、DistServe、Llumnix 等系统会按负载分配或迁移计算资源，但用于容错的 KV 仍可能与前台推理争用显存。固定预留会浪费资源；若额度调节与实际淘汰没有分开，快速收缩又可能删除尚无主存后备的显存副本。",
        ),
        1,
    ):
        add_step(doc, number, text)

    # 4. 本发明技术方案的详细阐述
    add_heading(
        doc,
        "本发明技术方案的详细阐述",
        1,
        template_num_level=0,
    )
    add_heading(doc, "4.1  本发明所要解决的技术问题", 2, numbered=False)
    for number, text in enumerate(
        (
            "如何把请求在工作流中的位置、剩余依赖和立即故障时的恢复时间，换算成每个 KV 分块单位字节可避免的工作流恢复损失。",
            "如何在路由器已经固定 Peer-HBM 目标后，独立完成显存和主存准入，由源 Worker 发起两条直接传输，并以分层确认保证每个副本真实可用。",
            "如何只用 AIMD 调节显存保护额度，再由本地执行器和存储控制模块安全执行收缩，避免直接删除尚无主存后备的显存副本。",
        ),
        1,
    ):
        add_step(doc, number, text)

    add_heading(doc, "本发明的主要要点概述", 2, template_num_level=1)
    add_body(
        doc,
        "本发明围绕三个彼此配合的技术点展开：第一，计算工作流关键性感知的 KV 优先级；第二，依据该优先级作出分层备份决策并完成可验证传输；第三，用加性增加、乘性降低的方式平衡前台推理与显存保护。整体关系见附图 1。",
    )
    add_body(
        doc,
        "所谓 KVCache，是模型处理过输入后留下的“已计算中间记忆”。它丢失后通常可以重算，但重算对工作流的影响并不相同。本发明不追求复制全部 KV，而是优先保护丢失后会显著增加工作流完成时间的分块。",
    )
    add_body(
        doc,
        "保护点一——Critical-Aware 优先级计算：系统只使用当前已经出现的工作流节点和依赖，预测各活动分支的剩余时间，并计算可容忍等待余量。信息不完整时余量取零，避免把未知风险误判为低优先级。",
    )
    add_body(
        doc,
        "对请求 i，令 Tᵢ 为立即故障时从重算、DRAM 和 Peer-HBM 中选择最快路径的暂停时间，sᵢ 为剩余余量，则恢复损失 Jᵢ=max(Tᵢ-sᵢ,0)。候选分块 x 的收益 Aₓ=max(Jᵢ(加入前)-Jᵢ(加入后),0)，Aₓ/分块字节数越大，单位空间保护价值越高。",
    )
    add_body(
        doc,
        "每个资源池维护租值 Lᵣ；候选接纳后的排序值为 Hₓ=Lᵣ+Aₓ/bₓ。候选还须满足容量可行且 Aₓ 大于拟淘汰集合造成的联合损失。余量变化只修正该请求当前尾部候选的元数据，不搬运 KV；每层全局索引至多比较每个请求的一个当前尾部。",
    )
    add_body(
        doc,
        "保护点二——KV 备份决策与传输：Router 先为请求固定兼容的 Peer-HBM 目标。系统分别判断 HBM 候选和 DRAM 候选，任一层是否接纳都不以另一层的结果为前提，因而可形成仅 HBM、仅 DRAM、两者都有或两者都没有四种结果。",
    )
    add_body(
        doc,
        "每层分别使用自己的恢复收益、容量、租值和两级候选索引，独立比较候选收益与拟淘汰集合的联合损失。某层接纳后，系统先释放该层拟淘汰集合、分配最终空间并标为待确认，再异步传输；一层的未来传输成本不计入另一层的即时收益。",
    )
    add_body(
        doc,
        "若接纳 HBM，源 Worker 通过 D2D/P2P 将完整分块复制到路由绑定目标；若接纳 DRAM，源 Worker 通过 D2H 将完整分块直接传入 DRAM。两条传输均由源 Worker 发起，可并行或分别执行，不经过目标 HBM 中转，并各自独立确认。",
    )
    add_body(
        doc,
        "任何物理副本只有在全部必要分片、路由版本、模型版本和布局均通过校验后才可用于恢复；否则仅终止并释放该路径的空间，不影响另一条已确认路径。故障时先隔离旧执行方，只使用已确认兼容副本，其余部分重算。",
    )
    add_body(
        doc,
        "若路由绑定目标先于源 Worker 故障，Router 保持源的执行和输出权限，并把备用身份切换到另一兼容目标；旧目标 HBM 不会被重新标成新目标所有。已经确认且仍可访问的 DRAM 副本不受该 HBM 目标切换影响。",
    )
    add_body(
        doc,
        "保护点三——AIMD 资源平衡：AIMD 只更新每个目标 Worker 的显存保护额度。压力时乘性降低，连续稳定时加性增加。若新额度低于当前用量，本地执行器阻止新的 HBM 接纳并发出容量事件；存储控制模块再按优先级回收已有 DRAM 后备的显存尾部分块。",
    )
    add_body(
        doc,
        "只有对应 DRAM 副本已经确认的 HBM 分块才可被回收。若超额部分仅有 HBM 副本，本窗口报告额度尚未满足并保留该副本；源 Worker 仍存活且 DRAM 候选后来获准时，再由源 Worker 独立执行 D2H。AIMD 不选择分块，传输引擎独立管理队列和速率。",
    )

    add_heading(doc, "本发明可应用的系统架构或场景", 2, template_num_level=1)
    add_body(
        doc,
        "系统包括工作流运行时、路由器、备份控制器、源 Worker、目标 Worker 和共享存储。工作流运行时提供当前节点及依赖；备份控制器依次执行优先级计算、备份决策和额度调节；Worker 与共享存储直接传输 KV，控制器只传递决策和目录信息。",
    )
    add_body(
        doc,
        "源 Worker 将连续 token 及恢复所需的全部 KV 数据封装为一个逻辑分块。Router 固定兼容的目标 Worker，目标只接收获准的 Peer-HBM 副本；共享存储接收源 Worker 直接传入的 DRAM 副本。两层独立准入、独立分配空间并独立确认。",
    )
    add_body(
        doc,
        "该方法不依赖特定模型、并行方式或存储产品。对于 TP、PP、EP/MoE 或多节点部署，一个逻辑分块包含恢复所需的全部分片；只有全部分片完整且模型版本、布局和内容版本一致时，副本才被确认为可恢复。",
    )
    maybe_add_external_figure(
        doc,
        1,
        "三项核心机制及其端到端关系",
    )

    add_heading(
        doc,
        "本发明涉及的产品、可能的售卖形态以及侵权取证方式",
        2,
        template_num_level=1,
    )
    add_body(
        doc,
        "本发明可作为大模型推理平台、智能体服务平台或云服务中的 KV 容错功能交付，也可实现为独立控制器、Worker 插件、存储连接器或软硬件一体系统。",
    )
    add_body(
        doc,
        "在依法取得授权或公开信息的前提下，可从接口、日志和监控中识别三类证据：Slack、立即故障恢复损失与单位字节收益的对应关系；独立 HBM/DRAM 准入、源侧 D2D/P2P 与 D2H 以及各层确认顺序；前台压力触发的额度快速下降和稳定期缓慢上升。",
    )
    add_body(
        doc,
        "仅凭性能结果不能证明采用本发明。较强证据还应显示：Peer-HBM 目标在准入前已由 Router 固定；HBM 与 DRAM 分别准入；两条物理传输均从源 Worker 发起且各自确认；额度收缩与具体淘汰分属不同模块，且 HBM 回收受已确认 DRAM 副本保护。",
    )

    add_heading(doc, "4.5  本发明技术实现的具体实施例", 2, numbered=False)
    add_heading(doc, "4.5.1 本发明实施例一：", 3, numbered=False)
    add_body(
        doc,
        "典型场景一为两个并行智能体分支 A、B 在下一节点汇合。本实施例说明 Critical-Aware KV 优先级的计算，流程见附图 2。",
    )
    for number, text in enumerate(
        (
            "记录当前已完成节点、活动节点和已经出现的依赖，形成残余工作流 DAG；尚未出现的动态分支不参与本轮计算。",
            "用 GBDT 或其他预测器估计各活动分支的剩余时间，再用 CPM 计算因果 Slack。若 A、B 还需 12 秒和 20 秒，则 A 可等待 8 秒，B 不可等待。",
            "把 Slack、截止时间和汇合关系换算为请求关键性；Slack 越小，关键性越高。信息不足时 fail-close，把 Slack 取零，避免低估风险。",
            "比较立即故障时的最快恢复时间 Tᵢ 与剩余 Slack sᵢ，得到 Jᵢ=max(Tᵢ-sᵢ,0)；再计算候选加入前后的损失下降 Aₓ，并以 Aₓ/分块字节数表示单位空间保护价值。",
        ),
        1,
    ):
        add_step(doc, number, text)
    maybe_add_external_figure(
        doc,
        2,
        "工作流关键性感知的 KV 优先级计算",
    )

    add_heading(doc, "4.5.2 本发明实施例二", 3, numbered=False)
    add_body(
        doc,
        "典型场景二为长请求持续产生 KV 分块。本实施例说明分层备份决策、传输确认和故障恢复的完整流程，见附图 3 和附图 4。",
    )
    for number, text in enumerate(
        (
            "源 Worker 封装完整逻辑分块；Router 已为请求固定兼容的 Peer-HBM 目标。HBM 与 DRAM 分别维护容量、租值、每请求连续前沿和两级尾部候选索引。",
            "系统分别判断固定目标上的 HBM 候选和直接 DRAM 候选。每层以本层候选收益、可用容量和拟淘汰集合的联合损失作准入，可得到仅 HBM、仅 DRAM、两者都有或两者都没有四种结果。",
            "对每个获准层，系统先释放该层拟淘汰集合、分配最终空间并标为待确认。源 Worker 对 HBM 发起 D2D/P2P，对 DRAM 发起 D2H；两条路径互不等待，也不以目标 HBM 作为 DRAM 中转。",
            "HBM 与 DRAM 分别在全部必要分片、版本和布局通过校验后独立成为可恢复副本。一条路径失败时只终止并释放该路径，另一条已确认副本继续有效。",
            "源故障时，Router 先隔离旧执行方和输出权限，再激活健康的路由绑定目标。系统只采用完整、已确认且版本一致的连续副本；其余部分静默重算到已提交输出位置后再继续输出。",
            "请求结束时，系统等待两条待确认路径分别完成或明确失败，再统一清理目录、Store 对象和 HBM 空间；源 Worker 消失后不再由目标 HBM 派生新的 DRAM 副本。",
        ),
        1,
    ):
        add_step(doc, number, text)
    maybe_add_external_figure(
        doc,
        3,
        "HBM/DRAM 独立准入与两级候选索引",
    )
    maybe_add_external_figure(
        doc,
        4,
        "源侧 D2D/D2H 双路径传输与分层确认",
    )

    add_heading(doc, "4.5.3本发明实施例三", 3, numbered=False)
    add_body(
        doc,
        "典型场景三为新请求批量到达，前台推理突然需要更多显存。本实施例说明 AIMD 如何在保护和服务之间自动平衡，见附图 5。",
    )
    for number, text in enumerate(
        (
            "AIMD 每个窗口只读取各目标 Worker 的显存压力并更新保护额度；前台模型执行、批处理和新请求空间始终优先。AIMD 不选择分块，也不控制传输路径。",
            "检测到压力时，额度按比例乘性降低。本地执行器先阻止新的 Peer-HBM 接纳并发出容量变化事件；存储控制模块再按两级索引顺序回收已经有 DRAM 后备的 HBM 尾部分块。",
            "只有对应 DRAM 副本已经确认的 HBM 分块才可回收。若超额部分仅有 HBM 副本，本窗口报告额度尚未满足并保留；源 Worker 存活且 DRAM 后来获准时，再由源独立执行 D2H。",
            "连续多个窗口无压力时，额度按固定步长加性增加；尚未连续稳定则保持不变。增加额度只允许后续接纳更多 HBM，不会立即搬运或占满显存。",
        ),
        1,
    ):
        add_step(doc, number, text)
    maybe_add_external_figure(
        doc,
        5,
        "AIMD 额度更新与安全回收执行",
    )

    add_heading(
        doc,
        "4.6  本发明的主要创新点及其对应的技术效果",
        2,
        numbered=False,
    )
    add_table(
        doc,
        ["本发明的主要创新点", "对应的有益效果"],
        [
            [
                "必要创新点一：由因果 Slack、立即故障恢复损失和单位字节边际收益计算 KV 优先级。",
                "优先保护丢失后最可能拖慢整个工作流且单位空间收益高的 KV。",
            ],
            [
                "必要创新点二：独立 HBM/DRAM 准入、源侧 D2D/D2H 双路径传输和分层确认。",
                "允许任一层或两层形成副本，避免 HBM 到 DRAM 中转，并保证副本可验证。",
            ],
            [
                "必要创新点三：AIMD 只更新显存额度，并把本地阻止新接纳与存储侧实际回收分离。",
                "压力时快速退让，稳定时缓慢恢复，且不直接淘汰尚无 DRAM 后备的 HBM。",
            ],
        ],
        widths_cm=[8.0, 8.0],
        font_size=8.2,
    )
    add_body(
        doc,
        "建议独立权利要求主线：以因果 Slack 和立即故障恢复损失计算候选分块的单位字节收益；在 Router 固定目标后独立执行 HBM/DRAM 准入，由源 Worker 分别发起 D2D/P2P 和 D2H，并经完整校验和分层确认形成副本；AIMD 仅更新显存额度，具体安全回收由执行模块完成。",
        bold_prefix="建议独立权利要求主线：",
    )

    # 5. 发散思维
    add_heading(doc, "发散思维：", 1, template_num_level=0)
    add_body(
        doc,
        "关键性信号不限于 CPM Slack，也可采用截止时间、SLO、业务等级、故障概率或预测的工作流延迟增量；GBDT 可替换为其他只使用当前可见信息的预测方法。",
    )
    add_body(
        doc,
        "两级候选索引可由堆、分桶队列、锦标赛树或其他有序结构实现；每个请求可暴露一个或少量候选。等价实现仍应把频繁变化的优先级元数据与实际 KV 搬运分开。",
    )
    add_body(
        doc,
        "DRAM 可替换或扩展为 CXL 内存、持久内存、远程内存、NVMe 或对象存储；完整副本可采用压缩、量化、增量编码或纠删码，只要能验证恢复所需数据完整且版本一致。",
    )
    add_body(
        doc,
        "目标 Worker 可以位于同机 GPU、跨机 GPU 或独立故障域；模型可采用 TP、PP、DP、EP/MoE 等并行方式。确认条件相应扩展为恢复所需全部分片完整且版本一致。",
    )
    add_body(
        doc,
        "源侧 D2H 可依据带宽令牌、I/O 优先级或队列长度调度；保护额度可按字节、页、块或 token 表示。等价方案仍应保持 HBM 与 DRAM 独立准入、源侧直接传输以及各层物理完成确认。",
    )
    add_body(
        doc,
        "AIMD 的压力信号、降低比例、增加步长和稳定窗口数均可配置；控制器可单机、主备、复制或分片部署。等价实现仍应保持额度更新、阻止新接纳、实际分块回收和传输控制之间的职责分离。",
    )

    # 6. 在先公开的现有技术文献信息
    add_heading(
        doc,
        "在先公开的现有技术文献信息",
        1,
        template_num_level=0,
    )
    add_body(
        doc,
        "下表列出本稿已核验的代表性公开论文和作者页面，用于说明技术背景，不等同于完成专利新颖性或自由实施检索。正式申请前仍应由专利代理人检索 CNIPA、WIPO Patentscope、Espacenet、Google Patents 和 USPTO，并核对专利族、优先权日、抵触申请及逐项技术特征。",
    )
    add_body(doc, "专利交底书Checklist", first_line=False)
    add_table(
        doc,
        [
            "序号",
            "国别以及代码给出的文献号（对于专利）或期刊或标准名称（包括卷号或版本号）",
            "公开日期",
            "文档来源",
            "相关的段落和/或图号和/或页数和/或章节数",
        ],
        [
            ["1", "vLLM / PagedAttention，ACM SOSP 2023", "2023-10", "DOI: 10.1145/3600006.3613165", "论文关于分页 KVCache 管理、内存共享和调度的章节。"],
            ["2", "SGLang，NeurIPS 2024", "2024", "NeurIPS Proceedings", "论文关于结构化 LLM program 和 RadixAttention 的章节。"],
            ["3", "Autellix，arXiv:2502.13965", "2025-02", "arXiv", "论文关于以 agent program 为对象的调用调度及程序上下文。"],
            ["4", "Mooncake，USENIX FAST 2025", "2025-02", "USENIX", "论文关于分离式 KVCache、全局缓存调度和 Store 数据路径。"],
            ["5", "LMCache，arXiv:2510.09665", "2025-10", "arXiv", "论文关于跨引擎 KV 保存、查找、卸载、传输和复用。"],
            ["6", "DéjàVu，ICML 2024 / PMLR 235", "2024-07", "PMLR", "论文关于 KV-cache streaming、状态复制和 fault tolerance。"],
            ["7", "DistServe，USENIX OSDI 2024", "2024-07", "USENIX", "论文关于 prefill/decode 分离、资源分配和跨阶段通信。"],
        ],
        widths_cm=[0.8, 4.6, 1.7, 3.8, 5.1],
        font_size=6.4,
    )


def build_document(template: Path, output: Path):
    if not template.exists():
        raise FileNotFoundError(f"Template not found: {template}")
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = Document(str(template))
    clear_template_body(doc)
    configure_document(doc)
    build_concise_disclosure(doc)

    doc.save(str(output))
    sanitize_package_metadata(output)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_document(args.template.resolve(), args.output.resolve())
    print(result)


if __name__ == "__main__":
    main()
