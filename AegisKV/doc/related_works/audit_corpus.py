#!/usr/bin/env python3
"""Audit every local related-work PDF without copying paper text into artifacts.

The audit is deliberately file-level: two different PDF versions of the same
paper remain separate rows.  A later, human-curated review maps those versions
to canonical works and decides whether a mechanism is a direct baseline.

Run with::

    uv run --with pypdf --with pymupdf --with pillow \
      python AegisKV/related_works/audit_corpus.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pymupdf
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "audit"

KEYWORDS = {
    "fault": re.compile(r"\b(?:fault|failure|fail-stop|resilien\w*)\b", re.I),
    "checkpoint": re.compile(r"\bcheckpoint\w*\b", re.I),
    "recovery": re.compile(r"\b(?:recover\w*|restart\w*|restore\w*)\b", re.I),
    "replication": re.compile(r"\b(?:replica\w*|replication|redundan\w*)\b", re.I),
    "erasure": re.compile(r"\b(?:erasure|parity|reed.solomon)\b", re.I),
    "kv_cache": re.compile(r"\b(?:KV[ -]?cache|key[ -]?value cache)\b", re.I),
    "host_dram": re.compile(r"\b(?:host (?:DRAM|memory)|CPU (?:DRAM|memory))\b", re.I),
    "ssd": re.compile(r"\b(?:SSD|NVMe|persistent storage)\b", re.I),
    "workflow": re.compile(r"\b(?:workflow|agentic|multi-agent)\b", re.I),
    "slack": re.compile(r"\b(?:slack|critical path|deadline|SLO)\b", re.I),
}

SECTION_RE = re.compile(
    r"^(?:[A-Z]?\.?\d+(?:\.\d+){0,2}\.?\s+)?"
    r"(?:INTRODUCTION|BACKGROUND|MOTIVATION|DESIGN|SYSTEM|METHOD(?:OLOGY)?|"
    r"IMPLEMENTATION|EVALUATION|EXPERIMENTS?|DISCUSSION|LIMITATIONS?|"
    r"RELATED WORK|CONCLUSION)\b[^\n]{0,100}$",
    re.I | re.M,
)


@dataclass(frozen=True)
class PdfAuditRow:
    path: str
    sha256: str
    bytes: int
    pages: int
    encrypted: bool
    parseable_pages: int
    extracted_characters: int
    metadata_title: str
    keyword_pages: dict[str, list[int]]
    section_heading_pages: dict[str, list[int]]
    status: str


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def _page_texts(reader: PdfReader) -> Iterable[tuple[int, str]]:
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            yield page_no, page.extract_text() or ""
        except Exception:
            yield page_no, ""


def audit_pdf(path: Path) -> PdfAuditRow:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        reader = PdfReader(path)
        encrypted = bool(reader.is_encrypted)
        keyword_pages = {name: [] for name in KEYWORDS}
        headings: dict[str, list[int]] = {}
        chars = 0
        parseable = 0
        for page_no, text in _page_texts(reader):
            if text.strip():
                parseable += 1
            chars += len(text)
            for name, pattern in KEYWORDS.items():
                if pattern.search(text):
                    keyword_pages[name].append(page_no)
            for match in SECTION_RE.finditer(text):
                heading = _clean(match.group(0)).lower()
                headings.setdefault(heading, []).append(page_no)
        metadata_title = _clean((reader.metadata or {}).get("/Title", ""))
        status = "PASS" if len(reader.pages) > 0 and parseable > 0 else "FAIL"
        return PdfAuditRow(
            path=str(path.relative_to(ROOT)),
            sha256=digest,
            bytes=len(payload),
            pages=len(reader.pages),
            encrypted=encrypted,
            parseable_pages=parseable,
            extracted_characters=chars,
            metadata_title=metadata_title,
            keyword_pages=keyword_pages,
            section_heading_pages=headings,
            status=status,
        )
    except Exception as exc:
        return PdfAuditRow(
            path=str(path.relative_to(ROOT)),
            sha256=digest,
            bytes=len(payload),
            pages=0,
            encrypted=False,
            parseable_pages=0,
            extracted_characters=0,
            metadata_title="",
            keyword_pages={},
            section_heading_pages={},
            status=f"FAIL: {type(exc).__name__}: {_clean(exc)}",
        )


def _write_json(rows: list[PdfAuditRow]) -> None:
    payload = {
        "schema_version": 1,
        "scope": "every PDF recursively under AegisKV/related_works",
        "file_count": len(rows),
        "unique_sha256": len({row.sha256 for row in rows}),
        "total_pages": sum(row.pages for row in rows),
        "status_counts": dict(Counter(row.status for row in rows)),
        "rows": [asdict(row) for row in rows],
    }
    (OUTPUT / "corpus_file_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(rows: list[PdfAuditRow]) -> None:
    fields = [
        "path",
        "sha256",
        "bytes",
        "pages",
        "encrypted",
        "parseable_pages",
        "extracted_characters",
        "metadata_title",
        "fault_pages",
        "checkpoint_pages",
        "recovery_pages",
        "replication_pages",
        "erasure_pages",
        "kv_cache_pages",
        "workflow_pages",
        "slack_pages",
        "status",
    ]
    with (OUTPUT / "corpus_file_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "sha256": row.sha256,
                    "bytes": row.bytes,
                    "pages": row.pages,
                    "encrypted": row.encrypted,
                    "parseable_pages": row.parseable_pages,
                    "extracted_characters": row.extracted_characters,
                    "metadata_title": row.metadata_title,
                    **{
                        f"{name}_pages": ";".join(
                            str(page) for page in row.keyword_pages.get(name, [])
                        )
                        for name in (
                            "fault",
                            "checkpoint",
                            "recovery",
                            "replication",
                            "erasure",
                            "kv_cache",
                            "workflow",
                            "slack",
                        )
                    },
                    "status": row.status,
                }
            )


def _fit_label(draw: ImageDraw.ImageDraw, label: str, width: int) -> list[str]:
    words = re.split(r"(?<=[/_ -])", label)
    lines: list[str] = []
    current = ""
    for word in words:
        trial = current + word
        if current and draw.textlength(trial) > width:
            lines.append(current.rstrip())
            current = word.lstrip()
        else:
            current = trial
    if current:
        lines.append(current.rstrip())
    return lines[:3]


def _render_first_page(path: Path, target: Path) -> None:
    document = pymupdf.open(path)
    page = document[0]
    matrix = pymupdf.Matrix(0.85, 0.85)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    pixmap.save(target)
    document.close()


def _contact_sheets(paths: list[Path]) -> None:
    rendered = OUTPUT / "first_pages"
    rendered.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 300, 390
    label_h = 70
    columns, rows_per_sheet = 4, 4
    per_sheet = columns * rows_per_sheet
    font = ImageFont.load_default(size=12)
    for start in range(0, len(paths), per_sheet):
        subset = paths[start : start + per_sheet]
        sheet = Image.new(
            "RGB",
            (columns * thumb_w, rows_per_sheet * (thumb_h + label_h)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(subset):
            png = rendered / f"{start + offset + 1:03d}.png"
            _render_first_page(path, png)
            image = Image.open(png).convert("RGB")
            image.thumbnail((thumb_w - 12, thumb_h - 12))
            col = offset % columns
            row = offset // columns
            x = col * thumb_w + (thumb_w - image.width) // 2
            y = row * (thumb_h + label_h) + (thumb_h - image.height) // 2
            sheet.paste(image, (x, y))
            label = str(path.relative_to(ROOT))
            for line_no, line in enumerate(_fit_label(draw, label, thumb_w - 16)):
                draw.text(
                    (col * thumb_w + 8, row * (thumb_h + label_h) + thumb_h + 5 + 15 * line_no),
                    line,
                    fill="black",
                    font=font,
                )
        number = start // per_sheet + 1
        sheet.save(OUTPUT / f"first_pages_contact_sheet_{number:02d}.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in ROOT.rglob("*.pdf") if OUTPUT not in path.parents)
    rows = [audit_pdf(path) for path in paths]
    _write_json(rows)
    _write_csv(rows)
    _contact_sheets(paths)
    failures = [row for row in rows if row.status != "PASS"]
    print(
        json.dumps(
            {
                "files": len(rows),
                "unique_sha256": len({row.sha256 for row in rows}),
                "pages": sum(row.pages for row in rows),
                "failures": [asdict(row) for row in failures],
                "contact_sheets": len(list(OUTPUT.glob("first_pages_contact_sheet_*.png"))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
