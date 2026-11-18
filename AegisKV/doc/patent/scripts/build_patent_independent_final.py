#!/usr/bin/env python3
"""Build the four-figure, independent-tier final AegisKV disclosure.

The script intentionally starts from ``AegisKV专利交底书-latest.docx`` and
writes a new document.  It does not modify the source disclosure.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import struct
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph


SCRIPT_DIR = Path(__file__).resolve().parent
PATENT_DIR = SCRIPT_DIR.parent
SOURCE_DOCX = PATENT_DIR / "AegisKV专利交底书-latest.docx"
OUTPUT_DOCX = PATENT_DIR / "AegisKV专利交底书-独立分层最终版.docx"
SOURCE_DRAWIO = PATENT_DIR / "figures" / "aegiskv-patent-figures-latest.drawio"
OUTPUT_DRAWIO = (
    PATENT_DIR / "figures" / "aegiskv-patent-figures-independent-final.drawio"
)
FIGURE_DIR = PATENT_DIR / "figures" / "independent-final"
DRAWIO_CLI = Path(
    "/home/comp/zhanzhao/.codex/skills/drawio/scripts/drawio-cli"
)

FINAL_TITLE = (
    "AegisKV：一种工作流感知的租值型 KV 优先级、独立分层备份管理与"
    "显存资源平衡方法、系统、设备及存储介质"
)

# These compatibility glyphs are convenient in plain text, but common Chinese
# Word font stacks do not consistently contain them.  The generated DOCX uses
# ordinary letters with Word's native subscript/superscript formatting instead.
SUBSCRIPT_GLYPHS = {
    "ᵢ": "i",
    "ₓ": "x",
    "ᵣ": "r",
    "ᵧ": "y",
    "ₚ": "p",
}
SUPERSCRIPT_GLYPHS = {
    "ʳ": "r",
}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def set_cell_value(diagram, cell_id: str, value: str) -> None:
    for cell in diagram.iter("mxCell"):
        if cell.get("id") == cell_id:
            cell.set("value", value)
            return
    raise AssertionError(f"missing draw.io cell: {cell_id}")


def set_cell_geometry(diagram, cell_id: str, **values: str) -> None:
    for cell in diagram.iter("mxCell"):
        if cell.get("id") == cell_id:
            geometry = cell.find("mxGeometry")
            if geometry is None:
                raise AssertionError(f"draw.io cell has no geometry: {cell_id}")
            for key, value in values.items():
                geometry.set(key, value)
            return
    raise AssertionError(f"missing draw.io cell: {cell_id}")


def set_edge_terminals(diagram, cell_id: str, source: str, target: str) -> None:
    for cell in diagram.iter("mxCell"):
        if cell.get("id") == cell_id:
            assert cell.get("edge") == "1", f"draw.io cell is not an edge: {cell_id}"
            cell.set("source", source)
            cell.set("target", target)
            return
    raise AssertionError(f"missing draw.io edge: {cell_id}")


def build_four_page_drawio() -> None:
    root = ET.parse(SOURCE_DRAWIO).getroot()
    diagrams = root.findall("diagram")
    assert len(diagrams) == 5, "the source drawing must contain five pages"

    # Retain the overview, Slack, independent-admission, and AIMD pages.  The
    # dedicated transfer page is folded into prose/pseudocode so the final
    # disclosure contains exactly four figures, as requested.
    retained = [copy.deepcopy(diagrams[index]) for index in (0, 1, 2, 4)]
    for diagram in diagrams:
        root.remove(diagram)
    for index, diagram in enumerate(retained, 1):
        diagram.set("name", f"Figure {index}")
        root.append(diagram)

    overview, slack, placement, aimd = retained
    set_cell_value(
        overview,
        "f1-input",
        "工作流 Slack\n命中率 / 复用信息\nLRU / 最近访问时序",
    )
    set_cell_geometry(overview, "f1-input", y="175", height="135")
    set_cell_value(
        overview,
        "f1-p1",
        "A  工作流感知 KV 优先级\n故障恢复收益融入租值",
    )
    set_cell_value(
        overview,
        "f1-p2",
        "B  动态租值独立分层管理\nLazy Refresh\n准入 / 淘汰 / 提交",
    )
    set_cell_value(
        overview,
        "f1-p3",
        "C  AIMD 资源平衡\n压力时快降，稳定时慢增",
    )
    set_cell_value(
        overview,
        "f1-note",
        "租值优先级驱动两层管理；AIMD 只约束 Peer-HBM 额度，不改写优先级",
    )
    set_cell_value(
        overview,
        "f1-recovery",
        "故障恢复\n只使用已确认副本\n其余区间重算",
    )

    set_cell_value(slack, "f2-state", "工作流信息\n请求 Slack  sᵢ(t)")
    set_cell_value(slack, "f2-predict", "本层容错信息\nTbefore / Tafter\nr ∈ {Peer-HBM, DRAM}")
    set_cell_value(
        slack,
        "f2-slack",
        "工作流恢复收益\nWₓʳ = Δ[max(Tᵢʳ-sᵢ,0)]",
    )
    set_cell_value(
        slack,
        "f2-request",
        "传统缓存信息\n命中率 p_hit、复用代价\nLRU / 最近访问时序",
    )
    set_cell_value(
        slack,
        "f2-chunk",
        "复合收益\nGₓʳ = λwW + λhUhit + λlUlru",
    )
    set_cell_value(
        slack,
        "f2-score",
        "单位字节优先级\nPₓʳ = Gₓʳ / bₓ\n租值键 Hₓʳ = Lᵣ + Pₓʳ",
    )
    set_cell_geometry(slack, "f2-state", x="80", y="200", width="320", height="140")
    set_cell_geometry(slack, "f2-predict", x="500", y="200", width="360", height="140")
    set_cell_geometry(slack, "f2-slack", x="960", y="200", width="420", height="140")
    set_cell_geometry(slack, "f2-request", x="110", y="470", width="360", height="150")
    set_cell_geometry(slack, "f2-chunk", x="570", y="440", width="520", height="190")
    set_cell_geometry(slack, "f2-score", x="1170", y="455", width="390", height="175")
    set_edge_terminals(slack, "f2-e1", "f2-state", "f2-predict")
    set_edge_terminals(slack, "f2-e2", "f2-predict", "f2-slack")
    set_edge_terminals(slack, "f2-e3", "f2-slack", "f2-chunk")
    set_edge_terminals(slack, "f2-e4", "f2-request", "f2-chunk")
    set_edge_terminals(slack, "f2-e5", "f2-chunk", "f2-score")

    set_cell_value(placement, "f3-new", "已封闭 KV 候选 x")
    set_cell_value(
        placement,
        "f3-hbmq",
        "Peer-HBM 独立决策\nSlackLandlord(x,r)",
    )
    set_cell_value(
        placement,
        "f3-hbm",
        "BYPASS；或\n撤销 Y 并分配空间\nPENDING → D2D/P2P\nCOMMITTED / ABORTED",
    )
    set_cell_value(
        placement,
        "f3-dramq",
        "DRAM 独立决策\nSlackLandlord(x,r)",
    )
    set_cell_value(
        placement,
        "f3-dram",
        "BYPASS；或\n撤销 Y 并分配空间\nPENDING → D2H\nCOMMITTED / ABORTED",
    )
    set_cell_value(
        placement,
        "f3-none",
        "组合结果<br>HBM_ONLY<br>DRAM_ONLY<br>HBM_AND_DRAM<br>BYPASS",
    )
    set_cell_value(
        placement,
        "f3-note",
        "Slack、命中率或访问时序变化只使索引键失效；"
        "仅在检查堆顶或暴露新尾部时 Lazy Refresh，不扫描或搬运全部 KV",
    )
    set_cell_geometry(placement, "f3-note", y="510", height="90")
    set_cell_geometry(placement, "f3-hbmq", y="125", height="140")
    set_cell_geometry(placement, "f3-hbm", x="800", y="120", width="330", height="150")
    set_cell_geometry(placement, "f3-dramq", y="345", height="140")
    set_cell_geometry(placement, "f3-dram", x="800", y="340", width="330", height="150")
    set_cell_geometry(placement, "f3-none", x="1180", y="190", width="250", height="240")
    set_cell_value(
        placement,
        "f3-local",
        "第一级：每请求连续尾部\n只暴露一个安全候选\n隐藏分块保留原租值键",
    )
    set_cell_value(
        placement,
        "f3-global",
        "第二级：本池最小堆\n比较各请求暴露的尾部\n按版本惰性刷新陈旧键",
    )

    set_cell_value(
        aimd,
        "f5-store",
        "存储控制模块\n"
        "从安全尾部直接回收\n"
        "HBM_ONLY / HBM_AND_DRAM\n"
        "先撤销资格，再释放显存",
    )
    set_cell_geometry(aimd, "f5-store", x="1220", y="80", width="330", height="220")

    for page_index, diagram in enumerate(retained, 1):
        ids = [cell.get("id") for cell in diagram.iter("mxCell")]
        assert len(ids) == len(set(ids)), f"duplicate cell id on page {page_index}"
        for edge in diagram.findall('.//mxCell[@edge="1"]'):
            geometry = edge.find("mxGeometry")
            assert geometry is not None and geometry.get("relative") == "1", (
                f"edge {edge.get('id')} on page {page_index} lacks relative geometry"
            )

    OUTPUT_DRAWIO.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(OUTPUT_DRAWIO, encoding="utf-8", xml_declaration=True)


def export_figures() -> list[Path]:
    assert DRAWIO_CLI.exists(), f"missing draw.io CLI: {DRAWIO_CLI}"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for page_index in range(1, 5):
        output = FIGURE_DIR / f"figure-{page_index}.drawio.png"
        subprocess.run(
            [
                str(DRAWIO_CLI),
                "-x",
                "-f",
                "png",
                "-e",
                "-b",
                "20",
                "-s",
                "2",
                "-p",
                str(page_index),
                "-o",
                str(output),
                str(OUTPUT_DRAWIO),
            ],
            check=True,
        )
        assert output.exists() and output.stat().st_size > 0
        outputs.append(output)
    return outputs


def replace_paragraph_text(paragraph: Paragraph, text: str) -> None:
    old_rpr = None
    if paragraph.runs and paragraph.runs[0]._r.rPr is not None:
        old_rpr = copy.deepcopy(paragraph.runs[0]._r.rPr)
    for run in list(paragraph.runs):
        paragraph._p.remove(run._r)
    run = paragraph.add_run(text)
    if old_rpr is not None:
        run._r.insert(0, old_rpr)


def find_paragraph(document: Document, text: str) -> Paragraph:
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text == text]
    if len(matches) != 1:
        raise AssertionError(f"expected one paragraph {text!r}, found {len(matches)}")
    return matches[0]


def insert_paragraph_after(anchor: Paragraph, text: str = "") -> Paragraph:
    new_p = OxmlElement("w:p")
    anchor._p.addnext(new_p)
    paragraph = Paragraph(new_p, anchor._parent)
    paragraph.style = "Normal"
    if text:
        paragraph.add_run(text)
    return paragraph


def set_paragraph_box(paragraph: Paragraph) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), "F2F2F2")
    ppr.append(shading)
    borders = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "4")
        border.set(qn("w:color"), "808080")
        borders.append(border)
    ppr.append(borders)


def insert_pseudocode(anchor: Paragraph, title: str, lines: list[str]) -> Paragraph:
    paragraph = insert_paragraph_after(anchor)
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(7)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.keep_together = True
    set_paragraph_box(paragraph)

    title_run = paragraph.add_run(title)
    title_run.bold = True
    title_run.font.size = Pt(10)
    title_run.add_break()
    for index, line in enumerate(lines):
        run = paragraph.add_run(line)
        run.font.size = Pt(9.5)
        if index + 1 < len(lines):
            run.add_break()
    return paragraph


def copy_cell_format(source_cell, target_cell) -> None:
    source_tcpr = source_cell._tc.tcPr
    target_tcpr = target_cell._tc.get_or_add_tcPr()
    for child in list(target_tcpr):
        target_tcpr.remove(child)
    for child in source_tcpr:
        target_tcpr.append(copy.deepcopy(child))


def set_cell_text(cell, text: str, font_size: float | None = None) -> None:
    cell.text = text
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            if font_size is not None:
                run.font.size = Pt(font_size)


def add_table_row_like(table, values: list[str], font_size: float | None = None) -> None:
    template_cells = table.rows[-1].cells
    row = table.add_row()
    for source_cell, target_cell, value in zip(
        template_cells, row.cells, values, strict=True
    ):
        copy_cell_format(source_cell, target_cell)
        set_cell_text(target_cell, value, font_size)
    trpr = row._tr.get_or_add_trPr()
    if trpr.find(qn("w:cantSplit")) is None:
        trpr.append(OxmlElement("w:cantSplit"))


def remove_table_row(row) -> None:
    row._tr.getparent().remove(row._tr)


def replace_media(docx_path: Path, figures: list[Path]) -> None:
    with zipfile.ZipFile(docx_path) as archive:
        document_root = ET.fromstring(archive.read("word/document.xml"))
        rels_root = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        rel_targets = {rel.get("Id"): rel.get("Target") for rel in rels_root}
        a_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
        r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        media_targets = []
        for blip in document_root.iter(f"{{{a_ns}}}blip"):
            target = rel_targets[blip.get(f"{{{r_ns}}}embed")]
            if target.startswith("media/"):
                media_targets.append(f"word/{target}")
        assert len(media_targets) == 4, media_targets
        assert len(set(media_targets)) == 4, media_targets

        replacement = {
            media_target: figure.read_bytes()
            for media_target, figure in zip(media_targets, figures, strict=True)
        }
        with tempfile.NamedTemporaryFile(
            dir=docx_path.parent, suffix=".docx", delete=False
        ) as tmp_handle:
            tmp_path = Path(tmp_handle.name)
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as output:
                for info in archive.infolist():
                    data = replacement.get(info.filename, archive.read(info.filename))
                    output.writestr(info, data)
            shutil.move(tmp_path, docx_path)
        finally:
            tmp_path.unlink(missing_ok=True)


def enable_field_updates(document: Document) -> None:
    settings = document.settings._element
    update_fields = settings.find(qn("w:updateFields"))
    if update_fields is None:
        update_fields = OxmlElement("w:updateFields")
        settings.append(update_fields)
    update_fields.set(qn("w:val"), "true")


def split_script_text(text: str) -> list[tuple[str, str | None]]:
    """Replace compatibility glyphs and group text by Word vertical alignment."""
    segments: list[tuple[str, str | None]] = []
    characters: list[str] = []
    current_alignment: str | None = None
    for character in text:
        if character in SUBSCRIPT_GLYPHS:
            replacement = SUBSCRIPT_GLYPHS[character]
            alignment = "subscript"
        elif character in SUPERSCRIPT_GLYPHS:
            replacement = SUPERSCRIPT_GLYPHS[character]
            alignment = "superscript"
        else:
            replacement = character
            alignment = None
        if characters and alignment != current_alignment:
            segments.append(("".join(characters), current_alignment))
            characters = []
        characters.append(replacement)
        current_alignment = alignment
    if characters:
        segments.append(("".join(characters), current_alignment))
    return segments


def copy_run_properties(run, alignment: str | None):
    source_rpr = run.find(qn("w:rPr"))
    if source_rpr is None and alignment is None:
        return None
    rpr = (
        copy.deepcopy(source_rpr)
        if source_rpr is not None
        else OxmlElement("w:rPr")
    )
    if alignment is not None:
        for existing in list(rpr.findall(qn("w:vertAlign"))):
            rpr.remove(existing)
        vert_align = OxmlElement("w:vertAlign")
        vert_align.set(qn("w:val"), alignment)
        rpr.append(vert_align)
    return rpr


def make_run_fragment(run, child, alignment: str | None):
    fragment = OxmlElement("w:r")
    for key, value in run.attrib.items():
        fragment.set(key, value)
    rpr = copy_run_properties(run, alignment)
    if rpr is not None:
        fragment.append(rpr)
    fragment.append(child)
    return fragment


def normalize_unicode_script_runs(document: Document) -> Counter[str]:
    """Convert Unicode compatibility scripts to native Word run formatting."""
    converted: Counter[str] = Counter()
    for run in list(document.element.iter(qn("w:r"))):
        text_nodes = list(run.iter(qn("w:t")))
        if not any(
            any(
                character in SUBSCRIPT_GLYPHS
                or character in SUPERSCRIPT_GLYPHS
                for character in (text_node.text or "")
            )
            for text_node in text_nodes
        ):
            continue

        fragments = []
        for child in run:
            if child.tag == qn("w:rPr"):
                continue
            if child.tag != qn("w:t"):
                fragments.append(
                    make_run_fragment(run, copy.deepcopy(child), None)
                )
                continue
            for text, alignment in split_script_text(child.text or ""):
                text_node = copy.deepcopy(child)
                text_node.text = text
                if text[:1].isspace() or text[-1:].isspace():
                    text_node.set(XML_SPACE, "preserve")
                fragments.append(make_run_fragment(run, text_node, alignment))
                if alignment is not None:
                    converted[alignment] += len(text)

        parent = run.getparent()
        if parent is None:
            continue
        position = parent.index(run)
        for fragment in fragments:
            parent.insert(position, fragment)
            position += 1
        parent.remove(run)

    return converted


def build_document(figures: list[Path]) -> Counter[str]:
    document = Document(SOURCE_DOCX)
    properties = document.core_properties
    properties.title = FINAL_TITLE
    properties.subject = (
        "工作流 Slack、命中复用与 LRU 时序融合的租值型 KV 优先级，"
        "动态租值独立分层管理及 AIMD 显存资源平衡"
    )
    properties.author = "AegisKV 项目组"
    properties.last_modified_by = "AegisKV 项目组"
    properties.keywords = (
        "agentic workflow; Slack-aware KV priority; hit rate; LRU; "
        "SlackLandlord; Lazy Refresh; independent HBM/DRAM; AIMD; GPU failure"
    )
    properties.comments = ""

    # Abbreviation table: retain all existing entries, correct D2H, and define
    # the two transfer abbreviations used throughout the disclosure.
    abbreviation_table = document.tables[0]
    set_cell_text(
        abbreviation_table.rows[-1].cells[0], "设备到主机传输（如 HBM 到 DRAM）"
    )
    set_cell_text(abbreviation_table.rows[-1].cells[1], "Device-to-Host")
    set_cell_text(abbreviation_table.rows[-1].cells[2], "D2H")
    add_table_row_like(
        abbreviation_table,
        ["设备到设备传输", "Device-to-Device", "D2D"],
    )
    add_table_row_like(
        abbreviation_table,
        ["点到点直接传输", "Peer-to-Peer", "P2P"],
    )
    add_table_row_like(
        abbreviation_table,
        ["最近最少使用", "Least Recently Used", "LRU"],
    )
    for row in list(abbreviation_table.rows):
        if row.cells[2].text in {"CPM", "GBDT"}:
            remove_table_row(row)

    replacements = {
        "工作流 DAG：在当前时刻确定的完整的工作流节点和依赖关系；不涉及不确定的动态分支。": (
            "工作流 DAG：表示智能体请求、工具调用及其依赖关系。工作流运行时向 KV 管理器提供每个当前请求的 Slack；"
            "本发明不限定产生 Slack 所采用的执行时间预测模型或工作流分析算法。"
        ),
        "fan-out/join 结构：一个主智能体会创建多个子智能体，这个结构为fan-out，然后所有子智能体的结果回返回给主智能体，这个结构为join。": (
            "fan-out/join 结构：主智能体创建多个可并行执行的子智能体形成 fan-out；"
            "后续节点等待相关子智能体结果汇合形成 join。"
        ),
        "Slack 时间：某活动分支在不推迟整个工作流完成时间的前提下还可容忍的时间余量。其仅由当前可见信息计算；信息不完整时取零。": (
            "Slack：某活动请求在不推迟工作流预计完成时间的前提下可容忍的时间余量。"
            "同一智能体的不同轮请求分别携带当前 Slack；请求或工具调用返回后可以发布新值，信息不足时取零。"
        ),
        "KV chunk：把一个请求的连续 token 区间及恢复所需的全部 KV 或并行 shards 视为一个不可拆分的保护、提交和恢复单元。": (
            "逻辑 KV 分块：把一个请求的连续 token 区间及恢复该区间所需的全部 KV 并行分片"
            "视为一个不可拆分的保护、提交和恢复单元；未封闭的尾部分块不参与保护。"
        ),
        "分层惰性重定优先级：把经常变化的请求优先级与较稳定的分块属性分开保存。Slack 变化时只改少量索引元数据，不立即搬动 KV 数据。": (
            "惰性租值刷新（Lazy Refresh）：将经常变化的 Slack、命中统计和访问时序与分块的已保存租值键分开维护。"
            "相关信号变化时只更新版本；分块成为本池堆顶或新暴露尾部时才重算租值键，不扫描全池，也不触发 KV 搬运。"
        ),
        "两级 victim 索引：此处的victim是指在对KV进行备份时，由于容量限制准备被淘汰、用于腾出空间的 KV chunk。第一级记录各请求当前可淘汰的尾部分块，第二级从所有请求的这些候选中选择实际 victim；同一请求的其余分块暂不参与全局排序。": (
            "两级淘汰候选索引：第一级在每个资源池内为每个请求仅暴露当前可安全释放的连续尾部分块；"
            "第二级以本池最小堆比较各请求暴露的尾部候选。同一请求的其余分块保留原租值键和版本，成为新尾部时再惰性刷新。"
        ),
        "多级缓存分层独立管理：HBM 与 DRAM 分别维护池内候选和淘汰前沿。系统输出 ADMIT_HBM、ADMIT_DRAM 或 BYPASS；HBM 未接纳时仍可直接接纳到 DRAM。": (
            "独立分层管理：Peer-HBM 与 DRAM 分别维护容量、租金、连续前沿、PENDING 记录和两级淘汰索引，"
            "并使用相同形式的复合优先级独立返回接纳或绕过；任一层的结果和已有副本不替代另一层的本层判断。"
        ),
        "三态驻留：HBM_ONLY 表示只有目标 HBM 副本；HBM_AND_DRAM 表示 HBM 与 DRAM 副本都有效；DRAM_ONLY 表示只保留 DRAM backing。": (
            "独立分层驻留结果：HBM_ONLY 表示只有目标 HBM 副本；HBM_AND_DRAM 表示两层副本均有效；"
            "DRAM_ONLY 表示只有 DRAM 副本；两层均未接纳或均被回收时为 BYPASS。"
        ),
        "GuardedAIMD：只调节 Peer-HBM 保护额度的压力反馈控制：有正常推理显存压力时快速降低额度，最小到0，连续稳定时缓慢提高。": (
            "HBMQuotaAIMD：只调节每个目标 Worker 的 Peer-HBM 保护额度。前台有显存压力时乘性降低，"
            "连续稳定时加性增加；它不选择分块，也不要求回收前先建立 DRAM 副本。"
        ),
        "用于保护 KVCache 的 HBM 与前台推理共享同一显存。保护过少会扩大故障后的恢复和重算，保护过多又会挤压模型执行、批处理及新请求所需空间。该矛盾要求控制面同时理解工作流紧迫度、分块恢复贡献、复制及写回成本、目标 Worker 压力和可验证的提交状态。": (
            "用于保护 KVCache 的 Peer-HBM 与前台推理共享同一显存。保护过少会扩大故障后的恢复和重算，"
            "保护过多又会挤压模型执行、批处理及新请求所需空间。该矛盾要求控制面同时理解工作流紧迫度、"
            "逐层恢复贡献、复制带宽、目标 Worker 压力和可验证的提交状态。"
        ),
        "S1. Autellix 等工作流调度方法能够看到调用关系，却主要决定先执行哪个调用；vLLM、SGLang、H2O 等 KV 管理方法主要依据内存、复用或 token 重要性。两类方法之间缺少直接联系，因而不能回答：某一块 KV 丢失后，会让整个工作流多等待多久。": (
            "S1. 传统 KV 管理通常依据命中率、复用概率、对象大小、LRU 最近访问顺序或 token 局部重要性决定去留。"
            "这些信号能够描述缓存局部价值，却不能区分请求在工作流中的位置：两个命中率和最近访问时间相近的请求，"
            "在故障后可能对工作流完成时间造成完全不同的影响。"
        ),
        "S2. Mooncake、LMCache、DéjàVu 等系统已经能够保存、传输或复制 KV，但通常分别关注数据通路、缓存复用或故障复制，尚未把“是否备份、备份到哪一层、由谁负责后续传输、何时才可用于恢复”组成一个可验证的完整流程。": (
            "S2. Mooncake、LMCache、DéjàVu 等系统已经能够保存、传输或复制 KV，但同一智能体的多轮请求会快速改变 Slack，"
            "若每次变化都重排全部分块或迁移数据，控制开销和传输抖动会随 KV 规模增长。现有方案缺少将租金淘汰、"
            "独立分层、惰性刷新、源侧传输和物理完成提交统一起来的在线管理流程。"
        ),
        "S3. vLLM、SGLang、DistServe、Llumnix 等系统会按负载分配或迁移计算资源，但用于容错的 KV 副本仍可能与前台推理争用显存。固定预留浪费资源，过快释放又可能删掉最后一份可恢复副本，现有方案缺少兼顾前台压力和保护底线的弹性调节机制。": (
            "S3. vLLM、SGLang、DistServe、Llumnix 等系统会按负载分配或迁移计算资源，但容错副本仍可能与前台推理争用显存。"
            "固定预留浪费资源；若额度控制、接纳门禁和实际分块回收没有分工，压力收缩就难以同时保证前台优先、目录一致和确定性重算。"
        ),
        "S1. 如何把请求在工作流中的位置和剩余依赖，换算成每个 KV 分块的备份优先级，使有限资源优先保护丢失后最可能拖慢整个工作流的数据。": (
            "S1. 如何把请求的工作流 Slack 转换为故障恢复边际价值，并与命中率、复用收益、LRU 最近访问时序和分块字节数融合，"
            "形成可直接用于 KV 接纳和淘汰的租值型优先级。"
        ),
        "S2. 如何联合决定一个 KV 分块是否备份、备份到高速显存还是主存，并用明确的传输责任、完成确认和故障恢复顺序保证副本真实可用。": (
            "S2. 如何在 Slack、命中统计和访问时序快速变化时，以每层独立租金和两级尾部索引惰性刷新优先级，"
            "完成候选与淘汰集合比较、独立准入、源侧传输以及分层提交，而不扫描或搬运全部 KV。"
        ),
        "S3. 如何根据前台推理压力自动调节显存中的 KV 保护额度，在低压力时逐步增加保护、在高压力时快速退让，同时不破坏最后一份有效副本。": (
            "S3. 如何只用 AIMD 调节 Peer-HBM 保护额度，并由本地执行器和存储控制模块直接回收低价值 HBM 备份；"
            "回收不依赖 DRAM 副本，但必须先撤销恢复资格，缺失保护的区间在故障时确定性重算。"
        ),
        "本发明围绕三个彼此配合的技术点展开：第一，计算工作流关键性感知的 KV 优先级；第二，依据该优先级作出分层备份决策并完成可验证传输；第三，用加性增加、乘性降低的方式平衡前台推理与显存保护。整体关系见附图 1。": (
            "本发明包括三个相互配合的技术点：第一，把工作流 Slack 与命中率、复用代价和 LRU 时序信息融合为租值型 KV 优先级；"
            "第二，以该优先级执行动态租值驱动的独立分层管理，包括 Lazy Refresh、准入、淘汰、源侧传输和分层提交；"
            "第三，用 AIMD 平衡前台推理与 Peer-HBM 保护额度。整体关系见附图 1。"
        ),
        "保护点一——A: 工作流感知KV优先级计算": (
            "保护点一——工作流感知的租值型 KV 优先级"
        ),
        "系统只使用当前已经出现的工作流节点和依赖，预测各活动分支的剩余时间，并计算Slack时间。信息不完整时余量取零，避免把未知风险误判为低优先级。": (
            "工作流运行时为每个当前请求提供 Slack，并附带请求标识、观测时间和版本。"
            "同一智能体连续发出的不同请求分别使用各自的当前 Slack；新值到达时覆盖旧版本，执行期间可按经过时间递减。"
            "Slack 的具体计算方法不构成本保护点的限制。"
        ),
        "对请求 i，令 Tᵢ 为立即故障时从重算、DRAM 和 Peer-HBM 中选择最快路径的暂停时间，sᵢ 为剩余余量，则恢复损失 Jᵢ=max(Tᵢ-sᵢ,0)。候选分块 x 的收益 Aₓ=max(Jᵢ(加入前)-Jᵢ(加入后),0)，Aₓ/分块字节数越大，单位空间保护价值越高。": (
            "对资源层 r∈{Peer-HBM, DRAM}，令 Tᵢʳ(Sᵢʳ) 为只依据本层已确认连续保护前沿计算的立即故障恢复时间，"
            "其中包含本层加载、未保护后缀重算和静默重放。其 Slack 可见恢复损失为 Jᵢʳ(Sᵢʳ,t)=max(Tᵢʳ(Sᵢʳ)-sᵢ(t),0)。"
            "候选 x 的工作流恢复收益为 Wₓʳ=max(Jᵢʳ(加入前)-Jᵢʳ(加入后),0)。Slack 越小且本层副本越能缩短恢复，Wₓʳ 越大。"
        ),
        "每个资源池维护租值 Lᵣ；候选接纳后的排序值为 Hₓ=Lᵣ+Aₓ/bₓ。候选还须满足容量可行且 Aₓ 大于拟淘汰集合造成的联合损失。余量变化只修正该请求当前尾部候选的元数据，不搬运 KV；每层全局索引至多比较每个请求的一个当前尾部。": (
            "令 pₓ(t) 为候选命中或再次复用概率，Cₓʳ 为一次复用可节省的加载或重算代价，"
            "Uhitₓʳ=pₓ(t)·Cₓʳ；令 Ulruₓ=ψ(t-tlastₓ)，其中 ψ 随距最近访问时间增大而单调减小。"
            "本层复合收益为 Gₓʳ=λwWₓʳ+λhUhitₓʳ+λlUlruₓ，λw>0，λh、λl≥0。"
            "单位字节优先级 Pₓʳ=Gₓʳ/bₓ，租值键 Hₓʳ=Lᵣ+Pₓʳ。由此，传统命中率和 LRU 信息仍参与判断，"
            "而 Slack 通过故障恢复收益 Wₓʳ 使工作流中更关键的请求获得更高保护优先级。"
        ),
        "保护点二——B: KV 备份决策与传输": (
            "保护点二——动态租值驱动的独立分层 KV 管理"
        ),
        "Router 先为请求固定兼容的 Peer-HBM 目标。HBM 和 DRAM 分别依据本层的恢复收益、容量、租值及两级候选索引独立准入，一层的结果和未来传输成本不影响另一层，因而可形成仅 HBM、仅 DRAM、两者都有或两者都没有四种结果。": (
            "Router 先为每个路由版本固定满足模型版本、KV 格式、并行布局和故障域约束的 Peer-HBM 目标。"
            "Peer-HBM 与 DRAM 分别维护租金 Lᵣ、容量、每请求连续尾部和本池最小堆，并各自调用 SlackLandlord。"
            "另一层的接纳结果和已有副本不替代本层判断，因此可形成仅 HBM、仅 DRAM、两层都有或两层都没有四种结果。"
        ),
        "某层接纳后，系统释放该层拟淘汰集合、分配最终空间并标为待确认。源 Worker 通过 D2D/P2P 向目标 HBM 复制，通过 D2H 直接向 DRAM 传输；两条路径可并行或分别执行，不经过目标 HBM 中转，并各自独立确认。副本只有在必要分片、路由版本、模型版本和布局均校验通过后才能用于恢复；单条路径失败不影响另一条已确认路径。故障时先隔离旧执行方，再使用已确认的兼容副本恢复，其余部分重算。若目标 Worker 先故障，Router 保留源 Worker 的执行和输出权限并更换备用目标；旧目标 HBM 不转移所有权，已确认且仍可访问的 DRAM 副本继续有效。": (
            "Slack、命中统计、最近访问时间或连续前沿变化时，系统只更新相应版本。"
            "当某个尾部成为堆顶或因前一尾部被选中而新暴露时，系统才按最新信号执行 Lazy Refresh；隐藏分块保留原键。"
            "候选复合收益超过本层拟淘汰集合的联合损失且容量可满足时，系统撤销并释放淘汰集合，分配最终空间并建立 PENDING 记录。"
            "源 Worker 从仍存活的生产缓冲区"
            "通过 D2D/P2P 复制到目标 HBM，并通过 D2H 直接传入 DRAM；两条路径可并行或分别执行，不经过对方中转。"
            "只有全部必要分片、路由版本、模型版本、KV 格式和布局均校验通过且物理传输完成，本层记录才转为 COMMITTED；"
            "失败路径转为 ABORTED 并释放本层空间，不改变另一层。故障时先隔离旧执行方，只使用已确认副本，其余区间重算。"
        ),
        "保护点三——C: AIMD 资源平衡": (
            "保护点三——AIMD 资源平衡"
        ),
        "我们通过AIMD 动态调整每个目标 Worker 的显存保护额度：出现压力时按比例快速降低，连续稳定时按固定步长缓慢增加。若新额度低于当前用量，系统停止接纳新的 HBM 副本，并按优先级直接删除价值较低的 HBM 尾部分块，直至满足额度。删除仅影响备份副本；故障时，没有副本的部分通过重算恢复。AIMD 只调整额度，具体分块选择和删除由存储控制模块执行。": (
            "HBMQuotaAIMD 在每个控制窗口只更新目标 Worker 的 Peer-HBM 保护额度：有前台显存压力时乘性降低，"
            "连续稳定达到阈值时加性增加。新额度低于当前保护用量时，本地执行器立即阻止新的 HBM 接纳；"
            "存储控制模块再从本层两级索引选择低价值且可安全释放的 HBM_ONLY 或 HBM_AND_DRAM 尾部分块，"
            "先原子撤销其可恢复资格，再释放显存。该过程不要求已有 DRAM 副本；故障时缺失区间通过重算恢复。"
        ),
        "系统包括工作流运行时、路由器、备份控制器、源 Worker、目标 Worker 和共享存储。工作流运行时提供当前节点及依赖；备份控制器依次执行优先级计算、备份决策和额度调节；Worker 与共享存储直接传输 KV，控制器只传递决策和目录信息。": (
            "系统包括工作流运行时、Router、Aegis 控制器、KV Directory/Store Master、传输引擎、源 Worker、目标 Worker 和共享存储。"
            "工作流运行时发布带版本的 Slack；控制器把 Slack、命中率、复用代价和 LRU 时序融合成租值型优先级，"
            "再按资源层执行 Lazy Refresh 和 SlackLandlord 准入；传输引擎执行源侧 D2D/P2P 与 D2H；"
            "KV Directory 维护 PENDING、COMMITTED、ABORTED 和驻留状态；本地执行器实施 HBM 额度门禁。"
        ),
        "源 Worker 将连续 token 及恢复所需的全部 KV 数据封装为一个逻辑分块。Router 固定兼容的目标 Worker，目标只接收获准的 Peer-HBM 副本；共享存储接收源 Worker 直接传入的 DRAM 副本。两层独立准入、独立分配空间并独立确认。": (
            "源 Worker 将连续 token 及恢复所需全部并行分片封装为逻辑分块。Router 固定兼容目标，目标只接收获准的 Peer-HBM 副本；"
            "共享存储只接收获准的直接 DRAM 副本。两层独立准入、独立分配最终空间、独立确认，并分别保持连续保护前沿。"
        ),
        "该方法不依赖特定模型、并行方式或存储产品。对于 TP、PP、EP/MoE 或多节点部署，一个逻辑分块包含恢复所需的全部分片；只有全部分片完整且模型版本、布局和内容版本一致时，副本才被确认为可恢复。": (
            "该方法不依赖特定模型、并行方式或存储产品。对于 TP、PP、EP/MoE 或多节点部署，一个逻辑分块包含恢复所需的全部分片；"
            "只有全部分片完整且路由版本、模型版本、KV 格式、布局和内容版本一致时，本层副本才被确认为可恢复。"
            "源故障后，系统按同一 Slack 推导的恢复截止顺序处理就绪请求，并静默重放未保护区间。"
        ),
        "本发明可作为大模型推理平台、智能体服务平台或云服务中的 KV 容错功能交付，也可实现为独立控制器、Worker 插件、存储连接器或软硬件一体系统。": (
            "本发明可作为大模型推理平台、智能体服务平台或云服务中的 KV 容错功能交付，也可实现为独立控制器、Worker 插件、"
            "存储连接器或软硬件一体系统。相应计算设备包括处理器、存储器和通信接口；存储器或计算机可读存储介质中的指令"
            "由处理器执行时，实现本说明书所述复合优先级计算、惰性租值刷新、独立分层管理、传输提交、故障恢复和额度控制步骤。"
        ),
        "在依法取得授权或公开信息的前提下，可从接口、日志和监控中识别三类证据：Slack、立即故障恢复损失与单位字节收益的对应关系；独立 HBM/DRAM 准入、源侧 D2D/P2P 与 D2H 以及各层确认顺序；前台压力触发的额度快速下降和稳定期缓慢上升。": (
            "在依法取得授权或公开信息的前提下，可从接口、目录快照、传输日志和监控中识别三类证据：Slack、命中统计、LRU 时序、"
            "恢复收益与单位字节租值键的对应关系；陈旧键的惰性刷新以及 Peer-HBM/DRAM 独立准入、源侧 D2D/P2P 与 D2H、"
            "各层 PENDING/COMMITTED/ABORTED 顺序；"
            "前台压力触发的额度乘性下降、稳定期加性增加以及 HBM 尾部分块撤销和释放顺序。"
        ),
        "仅凭性能结果不能证明采用本发明。较强证据还应显示：Peer-HBM 目标在准入前已由 Router 固定；HBM 与 DRAM 分别准入；两条物理传输均从源 Worker 发起且各自确认；额度收缩与具体淘汰分属不同模块，且 HBM 回收受已确认 DRAM 副本保护。": (
            "仅凭性能结果不能证明采用本发明。较强证据还应显示：Peer-HBM 目标在准入前已由 Router 固定；两层收益和准入互不遮蔽；"
            "两条物理传输均从源 Worker 发起且各自提交；额度收缩与具体淘汰分属不同模块；HBM_ONLY 可在目录先撤销后直接回收，"
            "而不是经目标 HBM 转存到 DRAM。"
        ),
        "典型场景一为两个并行智能体分支 A、B 在下一节点汇合。本实施例说明 Critical-Aware KV 优先级的计算，流程见附图 2。": (
            "典型场景一为两个并行智能体分支 A、B 分别发出多轮请求并在后续节点汇合。"
            "本实施例说明如何把每轮请求的工作流 Slack 与命中率、复用收益和 LRU 时序融合为租值型 KV 优先级，见附图 2。"
        ),
        "S1. 记录当前已完成节点、活动节点和已经出现的依赖，形成残余工作流 DAG；尚未出现的动态分支不参与本轮计算。": (
            "S1. 工作流运行时为当前请求 i 提供 Slack sᵢ(t)、观测时间和版本；KV 管理器同时读取候选 x 的命中率估计 pₓ(t)、"
            "最近访问时间 tlastₓ、复用可节省代价 Cₓʳ、字节数 bₓ 和本层连续保护前沿 Sᵢʳ。"
        ),
        "S2. 用 GBDT 或其他预测器估计各活动分支的剩余时间，再用 CPM 计算因果 Slack。若 A、B 还需 12 秒和 20 秒，则 A 可等待 8 秒，B 不可等待。": (
            "S2. 对 r∈{Peer-HBM,DRAM}，分别估计加入 x 前后的立即故障恢复时间 Tᵢʳ，计算"
            "Jᵢʳ=max(Tᵢʳ-sᵢ(t),0) 以及工作流恢复收益 Wₓʳ=max(Jᵢʳ(加入前)-Jᵢʳ(加入后),0)。"
        ),
        "S3. 把 Slack、截止时间和汇合关系换算为请求关键性；Slack 越小，关键性越高。信息不足时 fail-close，把 Slack 取零，避免低估风险。": (
            "S3. 计算命中复用收益 Uhitₓʳ=pₓ(t)·Cₓʳ，并以随访问间隔增加而减小的函数 ψ 计算 Ulruₓ=ψ(t-tlastₓ)。"
            "按 Gₓʳ=λwWₓʳ+λhUhitₓʳ+λlUlruₓ 融合工作流与传统缓存信息。"
        ),
        "S4. 比较立即故障时的最快恢复时间 Tᵢ 与剩余 Slack sᵢ，得到 Jᵢ=max(Tᵢ-sᵢ,0)；再计算候选加入前后的损失下降 Aₓ，并以 Aₓ/分块字节数表示单位空间保护价值。": (
            "S4. 得到单位字节优先级 Pₓʳ=Gₓʳ/bₓ 和租值键 Hₓʳ=Lᵣ+Pₓʳ。Slack、命中统计或最近访问时间变化时只更新版本；"
            "当前尾部在实际参与比较时才按最新信号刷新，不因优先级变化搬运 KV。"
        ),
        "典型场景二为长请求持续产生 KV 分块。本实施例说明分层备份决策、传输确认和故障恢复的完整流程，见附图 3。": (
            "典型场景二为长请求持续产生已封闭逻辑 KV 分块，同时同一智能体的新一轮请求使 Slack 和命中统计持续变化。"
            "本实施例说明动态租值、Lazy Refresh、独立分层准入、源侧传输和分层提交的完整流程，见附图 3。"
        ),
        "S1. 源 Worker 封装完整逻辑分块；Router 已为请求固定兼容的 Peer-HBM 目标。HBM 与 DRAM 分别维护容量、租值、每请求连续前沿和两级尾部候选索引。": (
            "S1. 源 Worker 封装完整逻辑分块；Router 为当前路由版本固定兼容 Peer-HBM 目标。Peer-HBM 与 DRAM 分别维护容量、租金 Lᵣ、"
            "每请求连续尾部、本池最小堆、索引版本和提交记录。两个资源池互不读取对方的准入结果。"
        ),
        "S2. 系统分别判断固定目标上的 HBM 候选和直接 DRAM 候选。每层以本层候选收益、可用容量和拟淘汰集合的联合损失作准入，可得到仅 HBM、仅 DRAM、两者都有或两者都没有四种结果。": (
            "S2. 对每个资源层分别计算候选复合收益 Gₓʳ、单位字节优先级 Pₓʳ 和接纳阈值 Θₓʳ=Lᵣ+Pₓʳ，"
            "读取本层可用容量，并建立本池最小堆的临时视图。"
        ),
        "S3. 对每个获准层，系统先释放该层拟淘汰集合、分配最终空间并标为待确认。源 Worker 对 HBM 发起 D2D/P2P，对 DRAM 发起 D2H；两条路径互不等待，也不以目标 HBM 作为 DRAM 中转。": (
            "S3. 容量不足时重复检查本池最小堆，并跳过候选所属请求以保持其前缀依赖。若堆顶键版本陈旧，只刷新该尾部后重新比较；"
            "选中尾部后只暴露并刷新同请求的前一尾部。"
            "无法找到低于 Θₓʳ 的合法尾部，或候选复合收益不大于淘汰集合 Y 的联合损失时返回 BYPASS；否则撤销 Y、推进租金、"
            "分配候选最终空间并建立 PENDING 记录。"
        ),
        "S4. HBM 与 DRAM 分别在全部必要分片、版本和布局通过校验后独立成为可恢复副本。一条路径失败时只终止并释放该路径，另一条已确认副本继续有效。": (
            "S4. 源 Worker 从生产缓冲区对 HBM 发起 D2D/P2P、对 DRAM 发起直接 D2H；两条路径可并行或分别执行，"
            "互不作为中转，并由各自完成回调独立确认。"
        ),
        "S5. 源故障时，Router 先隔离旧执行方和输出权限，再激活健康的路由绑定目标。系统只采用完整、已确认且版本一致的连续副本；其余部分静默重算到已提交输出位置后再继续输出。": (
            "S5. 本层全部必要分片、路由版本、模型版本、KV 格式和布局通过校验后，PENDING 才转为 COMMITTED；"
            "失败路径转为 ABORTED 并只释放本层空间，另一层记录保持不变。组合两层提交状态得到 HBM_ONLY、DRAM_ONLY、"
            "HBM_AND_DRAM 或 BYPASS。"
        ),
        "S6. 请求结束时，系统等待两条待确认路径分别完成或明确失败，再统一清理目录、Store 对象和 HBM 空间；源 Worker 消失后不再由目标 HBM 派生新的 DRAM 副本。": (
            "S6. 源故障时，Router 先隔离旧执行方和输出权限，再冻结已提交连续前沿；系统按剩余 Slack 截止顺序处理请求，"
            "只从可访问的 COMMITTED HBM 或 DRAM 副本恢复，其余区间静默重算到已提交输出位置。请求结束时等待两条路径均提交或终止后再清理。"
        ),
        "典型场景三为新请求批量到达，前台推理突然需要更多显存。本实施例说明 AIMD 如何在保护和服务之间自动平衡，见附图 4。": (
            "典型场景三为新请求批量到达，前台推理突然需要更多显存。本实施例说明 HBMQuotaAIMD 如何只调整保护额度，"
            "并由两个执行模块完成直接安全回收，见附图 4。"
        ),
        "S1. AIMD 每个窗口只读取各目标 Worker 的显存压力并更新保护额度；前台模型执行、批处理和新请求空间始终优先。AIMD 不选择分块，也不控制传输路径。": (
            "S1. 每个控制窗口读取目标 Worker 的前台显存压力 qₚ，并更新保护比例 αₚ；前台模型执行、批处理和新请求空间始终优先。"
            "HBMQuotaAIMD 不选择分块，也不控制传输路径。"
        ),
        "S2. 检测到压力时，额度按比例乘性降低。本地执行器先阻止新的 Peer-HBM 接纳并发出容量变化事件；存储控制模块再按优先级回收备份的KV。": (
            "S2. qₚ=1 时令 αₚ←max(αₚmin,γ·αₚ)，其中 0<γ<1；本地执行器立即阻止超过新额度的 Peer-HBM 新接纳，"
            "并向存储控制模块发布容量变化事件。"
        ),
        "S3. 因为HBM和DRAM两层KV备份独立进行，所以当正常推理出现显存的压力时可以直接淘汰备份的KV不用额外决策是否offload到DRAM上。": (
            "S3. 存储控制模块从 HBM 本层两级索引选择低价值且可安全释放的连续尾部；HBM_ONLY 与 HBM_AND_DRAM 均可成为候选，"
            "不以 DRAM 副本存在为前提。系统先原子撤销目录中的可恢复资格，再释放显存；故障时没有其他副本的区间通过重算恢复。"
        ),
        "S4. 连续多个窗口无压力时，额度按固定步长加性增加；尚未连续稳定则保持不变。增加额度只允许后续接纳更多 HBM，不会立即搬运或占满显存。": (
            "S4. 连续 K 个窗口无压力时令 αₚ←min(αₚmax,αₚ+δ)，否则保持当前额度。额度增加只允许后续接纳更多 HBM，"
            "不会立即搬运或占满显存；若暂时没有安全尾部可回收，则继续拒绝新接纳并记录额度尚未满足。"
        ),
        "建议独立权利要求主线：以因果 Slack 和立即故障恢复损失计算候选分块的单位字节收益；在 Router 固定目标后独立执行 HBM/DRAM 准入，由源 Worker 分别发起 D2D/P2P 和 D2H，并经完整校验和分层确认形成副本；AIMD 仅更新显存额度，具体安全回收由执行模块完成。": (
            "建议权利要求布局：主独立权利要求以“工作流 Slack→故障恢复边际收益→与命中率、复用收益和 LRU 时序融合→"
            "单位字节租值型 KV 优先级→独立资源池中的 Lazy Refresh、候选/淘汰比较、源侧传输和分层提交”为主线；"
            "两级尾部索引、四种驻留结果和目标故障轮换作为从属层，AIMD 显存资源平衡作为补充从属机制。"
            "系统、设备和存储介质权利要求分别映射到工作流运行时、Router、控制器、目录、传输引擎和本地执行器。"
        ),
        "关键性信号不限于 CPM Slack，也可采用截止时间、SLO、业务等级、故障概率或预测的工作流延迟增量；GBDT 可替换为其他只使用当前可见信息的预测方法。": (
            "Slack 可由任意能够反映工作流依赖和剩余等待余量的模块提供，也可与截止时间、SLO、业务等级或故障概率共同使用。"
            "复合收益的融合方式可采用加权和、分段单调函数或词典序，但工作流恢复收益必须实际参与租值键的生成，而非仅用于日志或事后排序。"
        ),
        "源侧 D2H 可依据带宽令牌、I/O 优先级或队列长度调度；保护额度可按字节、页、块或 token 表示。等价方案仍应保持 HBM 与 DRAM 独立准入、源侧直接传输以及各层物理完成确认。": (
            "源侧 D2D/P2P 与 D2H 可依据各自或共享的带宽令牌、I/O 优先级和队列长度调度；保护额度可按字节、页、块或 token 表示。"
            "等价方案仍保持两层收益和准入互不遮蔽、源侧直接传输以及各层物理完成确认。"
        ),
        "AIMD 的压力信号、降低比例、增加步长和稳定窗口数均可配置；控制器可单机、主备、复制或分片部署。等价实现仍应保持额度更新、阻止新接纳、实际分块回收和传输控制之间的职责分离。": (
            "AIMD 的压力信号、降低比例、增加步长和稳定窗口数均可配置；控制器可单机、主备、复制或分片部署。"
            "等价实现仍保持额度更新、阻止新接纳、目录撤销、实际分块回收和传输控制之间的职责分离。"
        ),
        "附图 1  三项核心机制及其端到端关系": (
            "附图 1  工作流感知租值、独立分层管理与 AIMD 资源平衡"
        ),
        "附图 2  工作流关键性感知的 KV 优先级计算": (
            "附图 2  工作流 Slack 与传统缓存信息融合的租值型 KV 优先级"
        ),
        "附图 3  HBM/DRAM 独立准入与两级候选索引": (
            "附图 3  动态租值驱动的独立分层 KV 管理流程"
        ),
        "附图 4  AIMD 额度更新与安全回收执行": (
            "附图 4  AIMD 资源平衡与 Peer-HBM 安全回收"
        ),
    }

    for old_text, new_text in replacements.items():
        paragraph = find_paragraph(document, old_text)
        replace_paragraph_text(paragraph, new_text)

    # Add definitions that make the pseudocode and lifecycle terms self-contained.
    slack_term = find_paragraph(
        document,
        "Slack：某活动请求在不推迟工作流预计完成时间的前提下可容忍的时间余量。"
        "同一智能体的不同轮请求分别携带当前 Slack；请求或工具调用返回后可以发布新值，信息不足时取零。",
    )
    insert_paragraph_after(
        slack_term,
        "租值型 KV 优先级：把 Slack 所对应的故障恢复边际收益，与命中率、复用可节省代价、LRU 最近访问时序和实际字节数融合，"
        "再叠加资源池累计租金形成可用于接纳和淘汰的租值键。",
    )
    term_anchor = find_paragraph(
        document,
        "HBMQuotaAIMD：只调节每个目标 Worker 的 Peer-HBM 保护额度。前台有显存压力时乘性降低，"
        "连续稳定时加性增加；它不选择分块，也不要求回收前先建立 DRAM 副本。",
    )
    commit_term = insert_paragraph_after(
        term_anchor,
        "分层提交状态：PENDING 表示已分配最终空间但物理传输或校验尚未完成；COMMITTED 表示本层全部必要分片和版本已经确认，"
        "可用于恢复；ABORTED 表示本层路径终止且空间已释放。",
    )
    insert_paragraph_after(
        commit_term,
        "源侧双路径传输：获准的 Peer-HBM 副本由源 Worker 从生产缓冲区通过 D2D/P2P 发送到 Router 绑定目标；"
        "获准的 DRAM 副本由同一源 Worker 通过 D2H 直接发送到共享存储。两条路径不互为前置条件，也不相互中转。",
    )

    # Explain the shared signal that makes the three mechanisms one technical system.
    shared_signal_anchor = find_paragraph(
        document,
        "所谓 KVCache，是模型处理过输入后留下的“已计算中间记忆”。它丢失后通常可以重算，但重算对工作流的影响并不相同。本发明不追求复制全部 KV，而是优先保护丢失后会显著增加工作流完成时间的分块。",
    )
    insert_paragraph_after(
        shared_signal_anchor,
        "工作流运行时提供请求级 Slack；KV 管理器将 Slack 对应的故障恢复收益与命中率、复用代价和 LRU 时序信息融合为租值型优先级；"
        "Peer-HBM 与 DRAM 资源池使用该优先级执行惰性刷新、接纳和淘汰；AIMD 只改变 Peer-HBM 可用额度，"
        "不改写优先级、资源池租金或传输所有权。",
    )

    # Chinese pseudocode makes each innovation directly executable without adding
    # a fourth table or a fifth figure.
    alg1_anchor = find_paragraph(
        document,
        "S4. 得到单位字节优先级 Pₓʳ=Gₓʳ/bₓ 和租值键 Hₓʳ=Lᵣ+Pₓʳ。Slack、命中统计或最近访问时间变化时只更新版本；"
        "当前尾部在实际参与比较时才按最新信号刷新，不因优先级变化搬运 KV。",
    )
    insert_pseudocode(
        alg1_anchor,
        "中文伪代码 1：工作流感知的租值型 KV 优先级计算",
        [
            "输入：请求 i 的 Slack sᵢ(t)，候选 x，资源层 r，前沿 Sᵢʳ，字节数 bₓ，命中率 pₓ(t)，最近访问时间 tlastₓ，租金 Lᵣ。",
            "1）估计加入 x 前后的本层恢复时间 Tbefore、Tafter。",
            "2）Jbefore←max(Tbefore-sᵢ(t),0)，Jafter←max(Tafter-sᵢ(t),0)，Wₓʳ←max(Jbefore-Jafter,0)。",
            "3）Uhitₓʳ←pₓ(t)·Cₓʳ；Ulruₓ←ψ(t-tlastₓ)，其中 ψ 随访问间隔增大而减小。",
            "4）Gₓʳ←λwWₓʳ+λhUhitₓʳ+λlUlruₓ。",
            "5）Pₓʳ←Gₓʳ/bₓ；Hₓʳ←Lᵣ+Pₓʳ。",
            "6）返回优先级、租值键和信号版本；信号变化只标记索引键陈旧，不移动 KV。",
        ],
    )

    alg2_anchor = find_paragraph(
        document,
        "S6. 源故障时，Router 先隔离旧执行方和输出权限，再冻结已提交连续前沿；系统按剩余 Slack 截止顺序处理请求，"
        "只从可访问的 COMMITTED HBM 或 DRAM 副本恢复，其余区间静默重算到已提交输出位置。请求结束时等待两条路径均提交或终止后再清理。",
    )
    insert_pseudocode(
        alg2_anchor,
        "中文伪代码 2：动态租值驱动的独立分层 KV 管理",
        [
            "输入：已封闭分块 x=(c,r)、当前时刻 t、本层租金 Lᵣ、本池尾部最小堆 Qᵣ；对 Peer-HBM 和 DRAM 分别执行。",
            "1）计算 Gₓʳ、Pₓʳ、接纳阈值 Θₓʳ←Lᵣ+Pₓʳ；Y←∅；Q←Qᵣ 的临时视图；f←本层空闲字节。",
            "2）当 f<bₓ：读取 Q 的最小尾部 (y,Hᵧ)，跳过候选所属请求；若 y 的 Slack/命中/LRU/前沿版本陈旧，则 Lazy Refresh 后重新比较。",
            "3）若无合法 y 或 Hᵧ≥Θₓʳ，则本层 BYPASS；否则 Y←Y∪{y}，累计可释放字节。",
            "4）y 被选中后，只把同一请求新暴露的前一连续尾部按最新版本计算后插入 Q。",
            "5）容量满足后计算 Y 的复合联合损失 D_Yʳ；若 Gₓʳ≤D_Yʳ，则本层 BYPASS。",
            "6）原子撤销并释放 Y；Lᵣ←max(Lᵣ,max_{y∈Y}Hᵧ)；分配 x 的最终空间并记 PENDING。",
            "7）源 Worker 发起直接复制：Peer-HBM 使用 D2D/P2P，DRAM 使用 D2H。",
            "8）全部必要分片和版本校验通过：按完成时最新信号重算 Gₓʳ，令 Hₓʳ←Lᵣ+Gₓʳ/bₓ，记 COMMITTED 并加入本层索引；否则记 ABORTED 并释放本层空间。",
            "9）组合两层状态得到 HBM_ONLY、DRAM_ONLY、HBM_AND_DRAM 或 BYPASS；恢复只使用 COMMITTED 连续前沿。",
        ],
    )

    alg3_anchor = find_paragraph(
        document,
        "S4. 连续 K 个窗口无压力时令 αₚ←min(αₚmax,αₚ+δ)，否则保持当前额度。额度增加只允许后续接纳更多 HBM，"
        "不会立即搬运或占满显存；若暂时没有安全尾部可回收，则继续拒绝新接纳并记录额度尚未满足。",
    )
    insert_pseudocode(
        alg3_anchor,
        "中文伪代码 3：AIMD 资源平衡与直接安全回收",
        [
            "输入：每个目标 p 的压力 qₚ、保护比例 αₚ、稳定计数 zₚ；输出：下一窗口 HBM 额度。",
            "1）若 qₚ=1：αₚ←max(αₚmin,γ·αₚ)，zₚ←0。",
            "2）否则若 zₚ+1≥K：αₚ←min(αₚmax,αₚ+δ)，zₚ←0；否则 zₚ←zₚ+1。",
            "3）发布 quotaₚ=αₚ·Mₚ；若当前 HBM 保护用量超额，立即阻止新的 HBM 接纳。",
            "4）重复选择 HBM 本层最低价值安全尾部 y（可为 HBM_ONLY 或 HBM_AND_DRAM）。",
            "5）先在目录中撤销 y 的恢复资格，再释放物理显存；不等待或派生 DRAM 副本。",
            "6）达到额度后停止；若无安全尾部，则记录未满足并保持接纳门禁。",
        ],
    )

    # Rewrite the innovation table as claim-supporting technical features rather
    # than internal A/B/C labels.
    innovation_table = document.tables[1]
    innovation_rows = [
        [
            "必要创新点一：将工作流 Slack 转换为故障恢复边际收益，并与命中率、复用收益、LRU 时序和实际字节数融合，形成租值型 KV 优先级。",
            "在保留传统缓存信息的同时区分不同请求对工作流完成时间的影响，使容错资源优先保护工作流关键 KV。",
        ],
        [
            "必要创新点二：以每层独立租金、每请求连续尾部和本池最小堆执行 Lazy Refresh、候选与淘汰联合比较、源侧传输及分层提交。",
            "Slack 快速变化时无需扫描或搬运全部 KV；支持四种分层结果，并保证只有物理完成且校验通过的副本可恢复。",
        ],
        [
            "必要创新点三：AIMD 只更新 Peer-HBM 保护额度；本地执行器阻止新接纳，存储控制模块按租值顺序直接回收安全尾部。",
            "压力时快速归还显存、稳定时渐进恢复；不依赖先建 DRAM 副本，缺失保护区间可确定性重算。",
        ],
    ]
    for row, values in zip(innovation_table.rows[1:], innovation_rows, strict=True):
        for cell, value in zip(row.cells, values, strict=True):
            set_cell_text(cell, value, 9.5)

    # Add the closest algorithmic and fault-tolerance references that are needed
    # to distinguish the claimed combination.
    prior_art_table = document.tables[2]
    add_table_row_like(
        prior_art_table,
        [
            "8",
            "Young, On-Line File Caching，SODA 1998 / Algorithmica 2002",
            "1998 / 2002",
            "SODA / Algorithmica",
            "关于变长对象缓存中的 credit/rent 与 Landlord 淘汰机制；不包含工作流 Slack、独立多层副本或故障恢复提交。",
        ],
        8.5,
    )
    add_table_row_like(
        prior_art_table,
        [
            "9",
            "GhostServe，MLSys 2026",
            "2026",
            "MLSys Proceedings / arXiv:2605.00831",
            "关于流式 KV checkpoint、奇偶校验以及 restore/reconstruct/recompute；未采用动态工作流 Slack 驱动独立分层资源管理。",
        ],
        8.5,
    )
    add_table_row_like(
        prior_art_table,
        [
            "10",
            "Pythia，arXiv:2604.25899",
            "2026-04",
            "arXiv",
            "关于智能体工作流可预测性、置信度和回退；不涉及 KV 容错副本的逐层准入、提交和显存额度回收。",
        ],
        8.5,
    )

    # Keep all figures centered and preserve the requested four-figure count.
    figure_paragraphs = [
        paragraph
        for paragraph in document.paragraphs
        if paragraph._p.xpath(".//w:drawing")
    ]
    assert len(figure_paragraphs) == 4
    for paragraph in figure_paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
    assert len(document.inline_shapes) == len(figures)
    for shape, figure in zip(document.inline_shapes, figures, strict=True):
        width_px, height_px = png_size(figure)
        shape.height = int(shape.width * height_px / width_px)

    script_counts = normalize_unicode_script_runs(document)
    assert script_counts["subscript"] > 0
    assert script_counts["superscript"] > 0
    enable_field_updates(document)
    document.save(OUTPUT_DOCX)
    replace_media(OUTPUT_DOCX, figures)
    OUTPUT_DOCX.chmod(0o644)
    return script_counts


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def audit_outputs(
    figures: list[Path], expected_script_counts: Counter[str]
) -> None:
    assert OUTPUT_DOCX.exists()
    document = Document(OUTPUT_DOCX)
    all_text = "\n".join(
        [paragraph.text for paragraph in document.paragraphs]
        + [
            cell.text
            for table in document.tables
            for row in table.rows
            for cell in row.cells
        ]
    )
    required = [
        "保护点一——工作流感知的租值型 KV 优先级",
        "保护点二——动态租值驱动的独立分层 KV 管理",
        "保护点三——AIMD 资源平衡",
        "Jir(Sir,t)=max(Tir(Sir)-si(t),0)",
        "Gxr=λwWxr+λhUhitxr+λlUlrux",
        "Hxr=Lr+Pxr",
        "命中率",
        "LRU",
        "Lazy Refresh",
        "SlackLandlord",
        "HBM_ONLY 或 HBM_AND_DRAM",
        "先原子撤销其可恢复资格，再释放显存",
        "中文伪代码 1：工作流感知的租值型 KV 优先级计算",
        "中文伪代码 2：动态租值驱动的独立分层 KV 管理",
        "中文伪代码 3：AIMD 资源平衡与直接安全回收",
        "必要创新点一：将工作流 Slack 转换为故障恢复边际收益",
        "必要创新点二：以每层独立租金、每请求连续尾部和本池最小堆执行 Lazy Refresh",
        "必要创新点三：AIMD 只更新 Peer-HBM 保护额度",
    ]
    forbidden = [
        "因果",
        "Critical-Aware",
        "GBDT",
        "CPM",
        "写回式",
        "write-back",
        "回写",
        "GuardedAIMD",
        "不破坏最后一份有效副本",
        "HBM 回收受已确认 DRAM 副本保护",
        "不用额外决策是否offload",
        "结果回返回",
        "为了简洁",
        "这里表述",
        "本轮",
        "上一版",
        "最终版",
        "修订",
        "待确认",
        "另行确认",
    ]
    for text in required:
        assert text in all_text, f"final disclosure misses: {text}"
    for text in forbidden:
        assert text not in all_text, f"final disclosure retains stale text: {text}"
    for glyph in (*SUBSCRIPT_GLYPHS, *SUPERSCRIPT_GLYPHS):
        assert glyph not in all_text, f"compatibility script glyph remains: {glyph}"
    assert len(document.inline_shapes) == 4
    for shape, figure in zip(document.inline_shapes, figures, strict=True):
        width_px, height_px = png_size(figure)
        image_ratio = width_px / height_px
        box_ratio = shape.width / shape.height
        assert abs(image_ratio - box_ratio) / image_ratio < 0.005, (
            figure,
            image_ratio,
            box_ratio,
        )
    assert len(document.tables) == 3
    assert len(document.tables[1].rows) == 4
    assert len(document.tables[2].rows) == 11
    assert document.core_properties.title == FINAL_TITLE
    assert document.core_properties.author == "AegisKV 项目组"
    assert document.core_properties.comments in {None, ""}
    assert all(
        row._tr.get_or_add_trPr().find(qn("w:cantSplit")) is not None
        for table in document.tables
        for row in table.rows
    )

    with zipfile.ZipFile(OUTPUT_DOCX) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert not any(name.startswith("word/comments") for name in names)
        document_xml = archive.read("word/document.xml")
        assert document_xml.count(b"<w:drawing>") == 4
        assert b"file:///" not in document_xml.lower()
        document_root = ET.fromstring(document_xml)
        visible_text = "".join(
            text_node.text or ""
            for text_node in document_root.iter(qn("w:t"))
        )
        for glyph in (*SUBSCRIPT_GLYPHS, *SUPERSCRIPT_GLYPHS):
            assert glyph not in visible_text, (
                f"compatibility script glyph remains in OOXML: {glyph}"
            )
        native_script_counts: Counter[str] = Counter()
        for run in document_root.iter(qn("w:r")):
            rpr = run.find(qn("w:rPr"))
            if rpr is None:
                continue
            vert_align = rpr.find(qn("w:vertAlign"))
            if vert_align is None:
                continue
            alignment = vert_align.get(qn("w:val"))
            native_script_counts[alignment] += sum(
                len(text_node.text or "")
                for text_node in run.iter(qn("w:t"))
            )
        for alignment, expected_count in expected_script_counts.items():
            assert native_script_counts[alignment] >= expected_count, (
                alignment,
                expected_count,
                native_script_counts[alignment],
            )
        for revision_tag in (
            "w:ins",
            "w:del",
            "w:moveFrom",
            "w:moveTo",
            "w:commentRangeStart",
            "w:commentReference",
        ):
            assert document_root.find(f".//{qn(revision_tag)}") is None, revision_tag
        settings_root = ET.fromstring(archive.read("word/settings.xml"))
        assert settings_root.find(f".//{qn('w:trackRevisions')}") is None
        core_text = archive.read("docProps/core.xml").decode("utf-8")
        for text in forbidden:
            assert text.lower() not in core_text.lower(), (
                f"core properties retain stale text: {text}"
            )
    for figure in figures:
        width, height = png_size(figure)
        assert width >= 2300 and height >= 700, (figure, width, height)

    print(f"PASS: {OUTPUT_DOCX}")
    print(f"PASS: {OUTPUT_DRAWIO}")
    for figure in figures:
        print(
            f"PASS: {figure} {png_size(figure)[0]}x{png_size(figure)[1]} "
            f"sha256={hashlib.sha256(figure.read_bytes()).hexdigest()}"
        )


def main() -> None:
    assert SOURCE_DOCX.exists(), f"missing source DOCX: {SOURCE_DOCX}"
    assert SOURCE_DRAWIO.exists(), f"missing source drawing: {SOURCE_DRAWIO}"
    build_four_page_drawio()
    figures = export_figures()
    script_counts = build_document(figures)
    audit_outputs(figures, script_counts)


if __name__ == "__main__":
    main()
