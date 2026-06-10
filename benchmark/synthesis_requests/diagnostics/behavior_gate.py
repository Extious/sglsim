"""Primary acceptance gate: SGLang behavioral parity (not ref CSV fitting)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    audit = REPO / "benchmark/synthesis_requests/diagnostics/behavior_audit.py"
    result = subprocess.run(
        [sys.executable, str(audit)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print("FAIL: behavior_gate (see behavior_audit)")
        return 1
    print("PASS: behavior_gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
