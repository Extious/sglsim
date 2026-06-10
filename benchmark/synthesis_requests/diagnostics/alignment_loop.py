#!/usr/bin/env python3
"""One alignment-loop iteration: static audit, unit tests, latest-run latency gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIAG = REPO / "benchmark/synthesis_requests/diagnostics"
RESULTS = REPO / "benchmark/synthesis_requests/results"
sys.path.insert(0, str(DIAG))
from parity_refs import (  # noqa: E402
    PARITY_REFERENCE_RUN_NAMES,
    is_no_overlap_run,
    ref_for_run,
)


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd or REPO)


def _latest_sim_run(*, prefer_no_overlap: bool = True) -> Path | None:
    candidates = [
        p
        for p in RESULTS.iterdir()
        if p.is_dir()
        and p.name not in PARITY_REFERENCE_RUN_NAMES
        and (p / "request_details.csv").is_file()
    ]
    if not candidates:
        return None
    if prefer_no_overlap:
        no_overlap = [p for p in candidates if is_no_overlap_run(p)]
        if no_overlap:
            return max(no_overlap, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _suggest_next(failures: list[str]) -> str:
    text = " ".join(failures).lower()
    if "infer-service" in text or "makespan" in text:
        return "overlap/result_queue pipeline sleep and per-step scheduler overhead"
    if "p90" in text or "total" in text:
        return "tail_breakdown + predictor probe; check retract/stall in server.log"
    if "ratio" in text:
        return "prefill_adder/KV load_back admission and chunked merge timing"
    return "re-read alignment_notes.md delta matrix and pick highest-priority gap"


def main() -> int:
    rc = _run(["uv", "run", "python", str(DIAG / "behavior_audit.py")])
    if rc != 0:
        return rc

    rc = _run(
        ["uv", "run", "python", str(DIAG / "behavior_gate.py")],
    )
    if rc != 0:
        return rc

    rc = _run(
        [
            "uv",
            "run",
            "--extra",
            "test",
            "pytest",
            "test/unit/srt/test_prefill_adder_sglang_parity.py",
            "test/unit/srt/test_prefill_adder.py",
            "test/unit/srt/test_scheduler_chunked_prefill.py",
            "test/unit/srt/test_scheduler_last_batch.py",
            "test/unit/srt/test_overlap_schedule.py",
            "test/unit/kvcache",
            "-q",
        ],
    )
    if rc != 0:
        return rc

    latest = _latest_sim_run(prefer_no_overlap=True)
    if latest is None:
        print("alignment_loop: no sim run with request_details.csv — run benchmark next")
        print(
            "  uv run python benchmark/synthesis_requests/run.py "
            "--config-dir benchmark/synthesis_requests/config/parity_full_no_overlap "
            "--disable-overlap-schedule"
        )
        return 0

    ref = ref_for_run(latest)
    incomplete = sorted(
        (
            p.name
            for p in RESULTS.iterdir()
            if p.is_dir()
            and p.name not in PARITY_REFERENCE_RUN_NAMES
            and not (p / "request_details.csv").is_file()
            and (p / "simulator_config.json").is_file()
        ),
        reverse=True,
    )[:3]
    mode = "no-overlap" if is_no_overlap_run(latest) else "overlap"
    print(f"alignment_loop: diagnosing {latest.name} ({mode}) vs {ref.name}")
    if incomplete:
        print(
            "  note: newer incomplete run(s) exist (no request_details.csv): "
            + ", ".join(incomplete)
        )
        print("  re-run benchmark after Round 4 fixes to refresh latency gates")
    _run(
        [
            "uv",
            "run",
            "python",
            str(DIAG / "latency_budget.py"),
            str(latest),
            str(ref),
            "--strict",
        ],
    )
    _run(
        [
            "uv",
            "run",
            "python",
            str(DIAG / "loop_gate.py"),
            str(latest),
            str(ref),
        ],
    )

    from latency_budget import component_gates

    failures, _notes = component_gates(latest, ref)
    if failures:
        print("alignment_loop: next suggested focus:", _suggest_next(failures))
    else:
        print("alignment_loop: component gates passed on latest run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
