#!/usr/bin/env python3
"""Run the full parity diagnostic suite for one sim run vs its matching SGLang ref."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DIAG = Path(__file__).resolve().parent
REPO = DIAG.parents[2]
sys.path.insert(0, str(DIAG))
from parity_refs import REF_NO_OVERLAP, REF_OVERLAP, ref_for_run  # noqa: E402


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parity diagnostic bundle.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Sim run directory under benchmark/synthesis_requests/results/",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Override SGLang reference directory.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = REPO / "benchmark/synthesis_requests/results" / run_dir
    if not run_dir.is_dir():
        print(f"run_dir not found: {run_dir}", file=sys.stderr)
        return 1

    ref_dir = args.ref_dir or ref_for_run(run_dir)
    run_id = run_dir.name
    ref_id = ref_dir.name
    py = ["uv", "run", "python"]

    steps = [
        py + [str(DIAG / "baseline_report.py"), run_id, "--ref-run", ref_id],
        py + [str(DIAG / "latency_budget.py"), str(run_dir), str(ref_dir), "--strict"],
        py + [str(DIAG / "loop_gate.py"), str(run_dir), str(ref_dir)],
        py + [str(DIAG / "compare_results.py"), str(run_dir), str(ref_dir)],
        py
        + [
            str(DIAG / "e2e_makespan_decompose.py"),
            str(run_dir / "server.log"),
            str(ref_dir / "server.log"),
            str(run_dir / "request_details.csv"),
            str(ref_dir / "request_details.csv"),
        ],
        py + [str(DIAG / "tail_breakdown.py"), str(run_dir), str(ref_dir)],
    ]

    rc = 0
    for cmd in steps:
        step_rc = _run(cmd)
        rc = rc or step_rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
