#!/usr/bin/env python3
"""Audit whether the checked-in causal snapshot can run Algorithm 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sglsim.srt.simulation.aegis_recovery_snapshot import (
    audit_algorithm2_inputs,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = HERE / "results" / "causal_recovery_snapshot.json"
DEFAULT_OUTPUT = HERE / "results" / "algorithm2_input_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    result = audit_algorithm2_inputs(snapshot)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"safe_result={result['safe_result']}")
    print(f"blocking_input_count={len(result['blocking_inputs'])}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
