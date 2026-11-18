#!/usr/bin/env python3
"""Run the v3.1.1 domain-corrected lineage without overwriting v3.1."""

from pathlib import Path

import run_active_hbm_v31_paired_rerun as implementation


ROOT = Path(__file__).resolve().parent
implementation.SPEC_PATH = ROOT / "active_hbm_v311_paired_rerun_protocol.json"
implementation.RESULT_NAME = "active_hbm_v311_paired_rerun"


if __name__ == "__main__":
    raise SystemExit(implementation.main())
