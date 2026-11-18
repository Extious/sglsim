#!/usr/bin/env python3
"""Audit the strict-template latest-design AegisKV disclosure."""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


SCRIPT_DIR = Path(__file__).resolve().parent
PATENT_DIR = SCRIPT_DIR.parent
DEFAULT_DOCUMENT = PATENT_DIR / "AegisKV专利交底书-严格模板版.docx"
DEFAULT_TEMPLATE = PATENT_DIR / "template" / "AegisKV交底书模板.docx"
DRAWIO_SOURCE = PATENT_DIR / "figures" / "aegiskv-patent-figures-latest.drawio"
INVENTION_TITLE = (
    "AegisKV：一种面向智能体工作流的关键性感知 KV 备份与"
    "资源自适应方法、系统、设备及存储介质"
)

EXPECTED_HEADINGS = [
    ("Heading 1", "缩略语和关键术语"),
    ("Heading 2", "2.1缩略语："),
    ("Heading 2", "2.2关键术语（如有涉及内部术语或自造词，请给出解释）："),
    ("Heading 1", "相关技术背景以及最接近的现有技术"),
    ("Heading 2", "3.1　背景技术："),
    ("Heading 2", "3.2  与本发明相关的现有技术"),
    ("Heading 3", "3.2.1  现有技术的技术方案"),
    ("Heading 3", "3.2.2  现有技术的缺点"),
    ("Heading 1", "本发明技术方案的详细阐述"),
    ("Heading 2", "4.1  本发明所要解决的技术问题"),
    ("Heading 2", "本发明的主要要点概述"),
    ("Heading 2", "本发明可应用的系统架构或场景"),
    ("Heading 2", "本发明涉及的产品、可能的售卖形态以及侵权取证方式"),
    ("Heading 2", "4.5  本发明技术实现的具体实施例"),
    ("Heading 3", "4.5.1 本发明实施例一："),
    ("Heading 3", "4.5.2 本发明实施例二"),
    ("Heading 3", "4.5.3本发明实施例三"),
    ("Heading 2", "4.6  本发明的主要创新点及其对应的技术效果"),
    ("Heading 1", "发散思维："),
    ("Heading 1", "在先公开的现有技术文献信息"),
]

EXPECTED_NUMBERING = [
    (43, 0),
    (0, 0),
    (0, 0),
    (43, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (43, 0),
    (0, 0),
    (43, 1),
    (43, 1),
    (43, 1),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (43, 0),
    (43, 0),
]

ABBREVIATION_HEADER = [
    "中文表述/中文术语",
    "完整的英文表述/英文准确用语",
    "英文缩略语",
]
INNOVATION_HEADER = ["本发明的主要创新点", "对应的有益效果"]
PRIOR_ART_HEADER = [
    "序号",
    "国别以及代码给出的文献号（对于专利）或期刊或标准名称（包括卷号或版本号）",
    "公开日期",
    "文档来源",
    "相关的段落和/或图号和/或页数和/或章节数",
]

REQUIRED_TEXT = [
    "已计算中间记忆",
    "残余工作流 DAG",
    "因果 Slack",
    "GBDT",
    "CPM",
    "fail-close",
    "Critical-Aware",
    "工作流关键性",
    "KV 备份优先级",
    "Jᵢ=max(Tᵢ-sᵢ,0)",
    "Aₓ=max(Jᵢ(加入前)-Jᵢ(加入后),0)",
    "Aₓ/分块字节数",
    "Hₓ=Lᵣ+Aₓ/bₓ",
    "两级候选索引",
    "仅 HBM、仅 DRAM、两者都有或两者都没有",
    "任一层是否接纳都不以另一层的结果为前提",
    "两条传输均由源 Worker 发起",
    "D2D/P2P",
    "D2H",
    "不经过目标 HBM 中转",
    "各自独立确认",
    "一条路径失败时只终止并释放该路径",
    "完整、已确认且版本一致",
    "AIMD",
    "AIMD 不选择分块",
    "乘性降低",
    "加性增加",
    "额度尚未满足",
    "只有对应 DRAM 副本已经确认的 HBM 分块才可被回收",
    "旧目标 HBM 不会被重新标成新目标所有",
    "建议独立权利要求主线",
    "典型场景一",
    "典型场景二",
    "典型场景三",
    "专利交底书Checklist",
]

FORBIDDEN_STRUCTURE_OR_STALE_TEXT = [
    "专利技术交底书",
    "文档结构",
    "正式提交前待办",
    "变形方案、应用扩展与规避设计",
    "本发明实施例四",
    "附录 A",
    "内部评审稿",
    "【待填写",
    "GPU(g)、Host 和 Recompute",
    "GPU / Host / Recompute",
    "D / U / O",
    "alpha / beta",
    "alpha/beta",
    "beta_d",
    "dependent_tier_recovery_ms",
    "base_dram_chunk_id",
    "P(r,c)=P_request(r)+P_chunk(r,c)",
    "收益是否值得其空间与传输成本",
    "目标 Worker 可在源故障后继续完成该任务",
    "目标侧任务接管",
    "目标责任接管",
    "目标 Worker 先幂等登记",
    "登记成功只表示责任接管",
    "目标随后再独立判断 DRAM",
    "等待 HBM 数据通路安静",
    "目标可继续建立后备",
    "未来在目标侧建立 DRAM 后备",
    "责任接管后独立准入",
    "目标任务",
    "all-stage ACK",
    "safe frontier",
    "L1 reuse",
    "L2 reuse",
    "初步模拟结果",
    "WCT CDF",
    "写回",
    "回写",
    "write-back",
    "writeback",
    "HBM-first",
    "clean-first",
    "Algorithm 1",
    "Algorithm 2",
    "Algorithm 3",
    "SlackLandlord",
    "GuardedAIMD",
    "HBM_ONLY",
    "HBM_AND_DRAM",
    "DRAM_ONLY",
    "PENDING",
    "COMMITTED",
    "ABORTED",
    "victim",
]

FIGURE_CAPTIONS = [
    "三项核心机制及其端到端关系",
    "工作流关键性感知的 KV 优先级计算",
    "HBM/DRAM 独立准入与两级候选索引",
    "源侧 D2D/D2H 双路径传输与分层确认",
    "AIMD 额度更新与安全回收执行",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_xml(data: bytes) -> bytes:
    return ET.tostring(ET.fromstring(data), encoding="utf-8")


def png_properties(data: bytes) -> tuple[int, int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("embedded figure is not PNG")
    width, height = struct.unpack(">II", data[16:24])
    return width, height, data[25]


def all_paragraphs(document):
    yield from document.paragraphs
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def table_text(table) -> list[list[str]]:
    return [[cell.text for cell in row.cells] for row in table.rows]


def paragraphs_between(body_paragraphs, start_text: str, end_text: str):
    """Return body paragraphs strictly between two template headings."""
    starts = [i for i, paragraph in enumerate(body_paragraphs) if paragraph.text == start_text]
    ends = [i for i, paragraph in enumerate(body_paragraphs) if paragraph.text == end_text]
    assert len(starts) == len(ends) == 1, (
        f"cannot locate unique section boundary: {start_text!r} -> {end_text!r}"
    )
    assert starts[0] < ends[0]
    return body_paragraphs[starts[0] + 1 : ends[0]]


def numbered_steps(paragraphs) -> list[int]:
    steps = []
    for paragraph in paragraphs:
        match = re.match(r"^S(\d+)\.\s", paragraph.text)
        if match:
            steps.append(int(match.group(1)))
    return steps


def audit_template_geometry(document, template) -> None:
    assert len(document.sections) == len(template.sections) == 1
    section = document.sections[0]
    template_section = template.sections[0]
    for attribute in (
        "page_width",
        "page_height",
        "top_margin",
        "bottom_margin",
        "left_margin",
        "right_margin",
        "header_distance",
        "footer_distance",
    ):
        assert getattr(section, attribute) == getattr(template_section, attribute), (
            f"template section geometry changed: {attribute}"
        )
    assert all(not paragraph.text.strip() for paragraph in section.header.paragraphs)
    assert all(not paragraph.text.strip() for paragraph in section.footer.paragraphs)


def audit_drawio_source() -> None:
    assert DRAWIO_SOURCE.exists(), f"missing editable drawing: {DRAWIO_SOURCE}"
    source_text = DRAWIO_SOURCE.read_text(encoding="utf-8")
    colors = {color.upper() for color in re.findall(r"#[0-9A-Fa-f]{6}", source_text)}
    assert colors <= {"#000000", "#FFFFFF"}, (
        f"patent figures contain non-black/white colors: {sorted(colors)}"
    )
    for required in ("D2D/P2P", "D2H", "两层都没有", "各自确认"):
        assert required in source_text, f"drawing misses current-design text: {required}"
    for stale in ("责任接管", "目标独立判断 DRAM", "继续建立 DRAM 后备"):
        assert stale not in source_text, f"drawing retains stale design text: {stale}"
    root = ET.fromstring(source_text)
    diagrams = root.findall("diagram")
    assert [diagram.get("name") for diagram in diagrams] == [
        f"Figure {number}" for number in range(1, 6)
    ]
    for page_number, diagram in enumerate(diagrams, 1):
        for edge in diagram.findall('.//mxCell[@edge="1"]'):
            geometry = edge.find("mxGeometry")
            assert geometry is not None and geometry.get("relative") == "1", (
                f"edge {edge.get('id')} on page {page_number} lacks relative geometry"
            )


def audit(document_path: Path, template_path: Path) -> None:
    assert document_path.exists(), f"missing document: {document_path}"
    assert template_path.exists(), f"missing template: {template_path}"
    document = Document(str(document_path))
    template = Document(str(template_path))
    audit_template_geometry(document, template)
    audit_drawio_source()

    assert document.core_properties.title == INVENTION_TITLE
    assert document.core_properties.last_modified_by == "AegisKV 项目组"

    body_paragraphs = document.paragraphs
    paragraphs = list(all_paragraphs(document))
    full_text = "\n".join(paragraph.text for paragraph in paragraphs)
    nonempty_body = [paragraph for paragraph in body_paragraphs if paragraph.text.strip()]
    assert nonempty_body[0].text == "缩略语和关键术语", (
        "template body must begin directly with section 2"
    )

    actual_headings = [
        (paragraph.style.name, paragraph.text)
        for paragraph in body_paragraphs
        if paragraph.style.name.startswith("Heading")
    ]
    assert actual_headings == EXPECTED_HEADINGS, (
        "heading sequence differs from template:\n"
        f"actual={actual_headings}\nexpected={EXPECTED_HEADINGS}"
    )
    actual_numbering = []
    for paragraph in body_paragraphs:
        if not paragraph.style.name.startswith("Heading"):
            continue
        num_properties = paragraph._p.pPr.numPr
        actual_numbering.append(
            (num_properties.numId.val, num_properties.ilvl.val)
        )
    assert actual_numbering == EXPECTED_NUMBERING, (
        "heading numId/ilvl sequence differs from the template: "
        f"actual={actual_numbering}, expected={EXPECTED_NUMBERING}"
    )

    assert len(document.tables) == 3, (
        f"strict template requires exactly three tables, got {len(document.tables)}"
    )
    assert [table.style.name for table in document.tables] == [
        "Normal Table",
        "Table Grid",
        "Table Grid",
    ]
    tables = [table_text(table) for table in document.tables]
    assert tables[0][0] == ABBREVIATION_HEADER
    assert len(tables[0]) == 14, "abbreviation table must contain 13 defined terms"
    assert tables[1][0] == INNOVATION_HEADER
    assert len(tables[1]) == 4, "innovation table must contain the three protection pillars"
    assert tables[2][0] == PRIOR_ART_HEADER
    assert len(tables[2]) == 8, "prior-art table must contain seven representative sources"

    checklist = [
        paragraph
        for paragraph in body_paragraphs
        if paragraph.text == "专利交底书Checklist"
    ]
    assert len(checklist) == 1 and checklist[0].style.name == "Normal"
    for text in REQUIRED_TEXT:
        assert text in full_text, f"missing current-design invariant: {text}"
    for text in FORBIDDEN_STRUCTURE_OR_STALE_TEXT:
        assert text not in full_text, f"extra structure or stale formulation remains: {text}"

    overview = paragraphs_between(
        body_paragraphs,
        "本发明的主要要点概述",
        "本发明可应用的系统架构或场景",
    )
    embodiment_one = paragraphs_between(
        body_paragraphs,
        "4.5.1 本发明实施例一：",
        "4.5.2 本发明实施例二",
    )
    embodiment_two = paragraphs_between(
        body_paragraphs,
        "4.5.2 本发明实施例二",
        "4.5.3本发明实施例三",
    )
    embodiment_three = paragraphs_between(
        body_paragraphs,
        "4.5.3本发明实施例三",
        "4.6  本发明的主要创新点及其对应的技术效果",
    )
    assert numbered_steps(embodiment_one) == [1, 2, 3, 4]
    assert numbered_steps(embodiment_two) == [1, 2, 3, 4, 5, 6]
    assert numbered_steps(embodiment_three) == [1, 2, 3, 4]
    prior_art_gaps = paragraphs_between(
        body_paragraphs,
        "3.2.2  现有技术的缺点",
        "本发明技术方案的详细阐述",
    )
    assert numbered_steps(prior_art_gaps) == [1, 2, 3]
    for paragraph in overview + embodiment_one + embodiment_two + embodiment_three:
        text = paragraph.text.strip()
        if not text or text.startswith("附图 ") or paragraph._p.xpath(".//w:drawing"):
            continue
        assert len(text) <= 260, (
            f"reader-facing paragraph is too dense ({len(text)} chars): {text}"
        )
    for row in tables[1][1:]:
        assert all(len(cell) <= 80 for cell in row), (
            f"innovation table row is not concise enough: {row}"
        )
    assert full_text.count("本发明实施例") == 3, (
        "template permits exactly three embodiment headings"
    )

    for number, caption in enumerate(FIGURE_CAPTIONS, 1):
        assert f"附图 {number}  {caption}" in full_text
    assert len(document.inline_shapes) == 5, "expected exactly five patent figures"
    figure_paragraphs = [
        paragraph
        for paragraph in body_paragraphs
        if paragraph._p.xpath(".//w:drawing")
    ]
    assert len(figure_paragraphs) == 5
    for index, paragraph in enumerate(figure_paragraphs, 1):
        assert paragraph.alignment == WD_ALIGN_PARAGRAPH.CENTER
        assert paragraph.paragraph_format.keep_with_next is True
        body_index = body_paragraphs.index(paragraph)
        caption = body_paragraphs[body_index + 1]
        assert caption.text == f"附图 {index}  {FIGURE_CAPTIONS[index - 1]}"
        assert caption.alignment == WD_ALIGN_PARAGRAPH.CENTER

    rows = [row for table in document.tables for row in table.rows]
    assert all(
        row._tr.get_or_add_trPr().find(qn("w:cantSplit")) is not None
        for row in rows
    ), "a table row can split across pages"

    with zipfile.ZipFile(template_path) as template_archive, zipfile.ZipFile(
        document_path
    ) as archive:
        assert archive.testzip() is None, "corrupt DOCX ZIP entry"
        for part_name in (
            "word/styles.xml",
            "word/numbering.xml",
            "word/theme/theme1.xml",
            "word/fontTable.xml",
        ):
            assert canonical_xml(archive.read(part_name)) == canonical_xml(
                template_archive.read(part_name)
            ), f"template part changed semantically: {part_name}"

        media = sorted(
            name for name in archive.namelist() if name.startswith("word/media/")
        )
        assert media == [f"word/media/image{number}.png" for number in range(1, 6)]
        source_paths = [
            PATENT_DIR / "figures" / f"latest-figure-{number}.png"
            for number in range(1, 6)
        ]
        section = document.sections[0]
        usable_width = section.page_width - section.left_margin - section.right_margin
        for index, (media_name, source_path, shape) in enumerate(
            zip(media, source_paths, document.inline_shapes, strict=True), 1
        ):
            embedded = archive.read(media_name)
            source = source_path.read_bytes()
            assert sha256(embedded) == sha256(source), f"stale embedded figure {index}"
            width, height, color_type = png_properties(embedded)
            assert width >= 2900 and height >= 1500
            assert color_type == 2
            assert shape.width <= usable_width
            source_ratio = width / height
            inserted_ratio = shape.width / shape.height
            assert abs(source_ratio - inserted_ratio) / source_ratio < 0.01

        app_ns = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
        app_root = ET.fromstring(archive.read("docProps/app.xml"))
        for local_name in (
            "Pages",
            "Words",
            "Characters",
            "CharactersWithSpaces",
            "TotalTime",
        ):
            node = app_root.find(f"{{{app_ns}}}{local_name}")
            assert node is None or node.text in (None, "0")
        for local_name in ("Template", "Company"):
            node = app_root.find(f"{{{app_ns}}}{local_name}")
            assert node is None or not (node.text or "").strip()

        w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        settings_root = ET.fromstring(archive.read("word/settings.xml"))
        assert settings_root.find(f"{{{w_ns}}}attachedTemplate") is None
        update_fields = settings_root.find(f"{{{w_ns}}}updateFields")
        assert update_fields is not None and update_fields.get(f"{{{w_ns}}}val") == "true"

        for name in archive.namelist():
            if name.endswith((".xml", ".rels")):
                lowered = archive.read(name).lower()
                assert b"file:///" not in lowered and b"users\\" not in lowered

    print(
        "PASS: strict template structure retained exactly: sections 2-6, three "
        "embodiments, three template tables, one Checklist label, and no added "
        "cover/appendix/subsections. The three prior-art gaps, 4/6/4-step "
        "embodiments, three-pillar innovation table, and five concise editable "
        "black/white figures are present."
    )
    print(
        "NOTE: this script audits DOCX/OOXML structure and content. Pagination "
        "and font substitution require a separate rendered-page inspection."
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", nargs="?", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    audit(arguments.document.resolve(), arguments.template.resolve())
