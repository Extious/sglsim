"""Shared SGLang parity reference paths for dual-state (no-overlap / overlap) alignment."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "benchmark" / "synthesis_requests" / "results"
REF_NO_OVERLAP_NAME = "with_hicache_without_overlap_sglang"
REF_OVERLAP_NAME = "with_hicache_with_overlap_sglang"
REF_NO_OVERLAP = RESULTS / REF_NO_OVERLAP_NAME
REF_OVERLAP = RESULTS / REF_OVERLAP_NAME
LEGACY_REF_NAME = "with_hicache_sglang"
LEGACY_REF = RESULTS / LEGACY_REF_NAME
PARITY_REFERENCE_RUN_NAMES = {
    REF_NO_OVERLAP_NAME,
    REF_OVERLAP_NAME,
    LEGACY_REF_NAME,
}


def is_no_overlap_run(run_dir: Path) -> bool:
    cfg = run_dir / "simulator_config.json"
    if not cfg.is_file():
        return False
    try:
        payload = json.loads(cfg.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        return False
    return bool(scheduler.get("disable_overlap_schedule", False))


def ref_for_run(run_dir: Path) -> Path:
    """Pick the matching SGLang reference for a sim run directory."""
    if is_no_overlap_run(run_dir):
        return REF_NO_OVERLAP
    return REF_OVERLAP


def ref_by_name(name: str) -> Path:
    if name == REF_NO_OVERLAP_NAME:
        return REF_NO_OVERLAP
    if name == REF_OVERLAP_NAME:
        return REF_OVERLAP
    if name == LEGACY_REF_NAME:
        return LEGACY_REF
    return RESULTS / name
