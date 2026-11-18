#!/usr/bin/env python3
"""Validate or bind one static multi-lineage AegisKV scale campaign."""

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
from AegisKV.experiments.real_weka_full_design.scale_campaign_contract import (
    SCALE_CAMPAIGN_BLOCKED_AUTHORIZATION,
    ScaleCampaignContractError,
    evaluate_scale_campaign_readiness,
    materialize_scale_campaign_binding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    try:
        readiness = evaluate_scale_campaign_readiness(args.requirements)
        if readiness["status"] != "READY_STATIC_ONLY":
            print(json.dumps(readiness, sort_keys=True, allow_nan=False))
            return 2
        output = materialize_scale_campaign_binding(
            requirement_manifest_path=args.requirements,
            output_root=args.output_root,
        )
    except (OSError, ValueError, ScaleCampaignContractError) as exc:
        failure = {
            "schema_version": 1,
            "evidence_class": "sglsim_aegiskv_scale_campaign_cli_failure_v1",
            "status": "FAIL_CLOSED",
            "error": str(exc),
            "cells": [],
            "authorization": dict(SCALE_CAMPAIGN_BLOCKED_AUTHORIZATION),
        }
        print(json.dumps(failure, sort_keys=True, allow_nan=False))
        return 2

    result = {
        "schema_version": 1,
        "status": "BOUND_STATIC_ONLY",
        "campaign_binding": artifact_record(output),
        "trace_loaded": False,
        "model_fitted": False,
        "replay_run": False,
        "result_written": False,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
