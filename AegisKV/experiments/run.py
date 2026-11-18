#!/usr/bin/env python3
"""Stable AegisKV experiment entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "python", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.main import main


if __name__ == "__main__":
    raise SystemExit(main())
