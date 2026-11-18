#!/usr/bin/env python3
"""Build the immutable matched plan or audit a supplied topology inventory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AegisKV.experiments.physical.contract import (  # noqa: E402
    build_matched_replay_plan,
    canonical_json_bytes,
    load_physical_configs,
    preflight_inventory,
)


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--lineage", type=Path)
    plan_parser.add_argument("--output", type=Path)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--plan", type=Path, required=True)
    preflight_parser.add_argument("--inventory", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "plan":
        value = build_matched_replay_plan(lineage=args.lineage)
    else:
        value = preflight_inventory(
            inventory=_read_object(args.inventory),
            plan=_read_object(args.plan),
            configs=load_physical_configs(),
        )
    if args.output:
        _write_new(args.output.resolve(), value)
        print(args.output.resolve())
    else:
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
