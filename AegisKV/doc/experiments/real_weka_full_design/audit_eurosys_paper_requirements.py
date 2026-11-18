#!/usr/bin/env python3
"""Materialize a static EuroSys AegisKV paper-lineage readiness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    artifact_record,
)
from AegisKV.experiments.real_weka_full_design.eurosys_paper_readiness import (
    EuroSysPaperReadinessError,
    materialize_eurosys_paper_readiness,
    validate_eurosys_paper_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--evaluation-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        output = materialize_eurosys_paper_readiness(
            requirement_manifest_path=args.requirements,
            evaluation_source_path=args.evaluation_source,
            output_path=args.output,
        )
        readiness = validate_eurosys_paper_readiness(output)
    except (OSError, ValueError, EuroSysPaperReadinessError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "evidence_class": (
                        "sglsim_aegiskv_eurosys_paper_readiness_cli_failure_v1"
                    ),
                    "status": "FAIL_CLOSED",
                    "error": str(exc),
                    "cells": [],
                    "trace_loaded": False,
                    "model_fitted": False,
                    "replay_run": False,
                    "result_written": False,
                    "formal_ready": False,
                    "paper_ready": False,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": readiness["status"],
                "readiness_artifact": artifact_record(output),
                "planned_cell_count": 0,
                "trace_loaded": False,
                "model_fitted": False,
                "replay_run": False,
                "result_written": False,
                "formal_ready": False,
                "paper_ready": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if readiness["status"] == "READY_STATIC_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
