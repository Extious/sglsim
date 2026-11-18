#!/usr/bin/env python3
"""Print bounded, page-labelled evidence from a local PDF for manual review."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


def _parse_pages(spec: str, total: int) -> list[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(page for page in pages if 1 <= page <= total)


def _compact(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"[ \t]+", " ", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", default="")
    parser.add_argument("--grep", action="append", default=[])
    parser.add_argument("--context", type=int, default=500)
    args = parser.parse_args()

    reader = PdfReader(args.pdf)
    selected = _parse_pages(args.pages, len(reader.pages)) if args.pages else []
    patterns = [re.compile(term, re.I) for term in args.grep]
    for page_no, page in enumerate(reader.pages, start=1):
        text = _compact(page.extract_text() or "")
        if page_no in selected:
            print(f"\n===== PHYSICAL PDF PAGE {page_no}/{len(reader.pages)} =====\n")
            print(text)
            continue
        for pattern in patterns:
            match = pattern.search(text)
            if match is None:
                continue
            start = max(0, match.start() - args.context)
            end = min(len(text), match.end() + args.context)
            print(
                f"\n===== PHYSICAL PDF PAGE {page_no}/{len(reader.pages)} "
                f"MATCH {pattern.pattern!r} =====\n"
            )
            print(text[start:end])
            break


if __name__ == "__main__":
    main()
