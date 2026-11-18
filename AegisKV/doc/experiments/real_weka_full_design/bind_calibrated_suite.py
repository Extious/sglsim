#!/usr/bin/env python3
"""Materialize one immutable calibrated-suite binding without running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AegisKV.experiments.real_weka_full_design.calibrated_suite_contract import (
    CONSUMER_KINDS,
    CalibratedConsumerInput,
    materialize_suite_binding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consumer-kind", choices=sorted(CONSUMER_KINDS), required=True
    )
    parser.add_argument("--consumer-manifest", type=Path, required=True)
    parser.add_argument("--consumer-payload", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw_matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if not isinstance(raw_matrix, dict):
        raise ValueError("matrix must contain one JSON object")
    output = materialize_suite_binding(
        source=CalibratedConsumerInput(
            consumer_kind=args.consumer_kind,
            consumer_manifest=args.consumer_manifest,
            consumer_payload=args.consumer_payload,
        ),
        matrix=raw_matrix,
        output_root=args.output_root,
    )
    print(output)


if __name__ == "__main__":
    main()
