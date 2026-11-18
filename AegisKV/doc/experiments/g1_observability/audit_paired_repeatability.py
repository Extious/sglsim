#!/usr/bin/env python3
"""Audit repeated paired-bystander smokes without upgrading their evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_INPUTS = (
    HERE / "paired_results" / "paired_bystander_summary.json",
    HERE / "paired_results_repeat2" / "paired_bystander_summary.json",
)
DEFAULT_OUTPUT = HERE / "paired_repeatability_audit.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _progress_gap(summary: dict[str, Any]) -> int:
    progress = summary.get("pre_failure_bystander_progress", {})
    if not isinstance(progress, dict):
        return -1
    reuse = progress.get("paired-bystander-reuse", {})
    recompute = progress.get("paired-bystander-recompute", {})
    if not isinstance(reuse, dict) or not isinstance(recompute, dict):
        return -1
    return abs(
        int(reuse.get("generated_tokens_observed", 0))
        - int(recompute.get("generated_tokens_observed", 0))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = tuple(path.resolve() for path in (args.inputs or DEFAULT_INPUTS))
    if len(inputs) < 2:
        raise ValueError("repeatability audit requires at least two runs")

    rows: list[dict[str, Any]] = []
    for path in inputs:
        summary = _read(path)
        if summary.get("counterfactual_identification") != "NOT_IDENTIFIED":
            raise ValueError(f"{path} unexpectedly claims identified evidence")
        reuse = summary.get("reuse_retry", {})
        recompute = summary.get("recompute_retry", {})
        if not isinstance(reuse, dict) or not isinstance(recompute, dict):
            raise ValueError(f"{path} lacks retry projections")
        if int(reuse.get("restored_prefill_tokens", 0)) <= 0:
            raise ValueError(f"{path} reuse arm did not restore a prefix")
        if int(recompute.get("restored_prefill_tokens", 0)) != 0:
            raise ValueError(f"{path} recompute arm restored a prefix")
        rows.append(
            {
                "path": str(
                    path.relative_to(REPO_ROOT)
                    if path.is_relative_to(REPO_ROOT)
                    else path
                ),
                "snapshot_sha256": str(summary.get("snapshot_sha256") or ""),
                "bystander_latency_delta_s": float(
                    summary["observed_bystander_latency_delta_s"]
                ),
                "retry_latency_delta_s": float(
                    summary["observed_retry_latency_delta_s"]
                ),
                "pre_failure_generated_token_gap": _progress_gap(summary),
            }
        )

    bystander_deltas = [row["bystander_latency_delta_s"] for row in rows]
    retry_deltas = [row["retry_latency_delta_s"] for row in rows]
    bystander_signs = {
        1 if value > 0.0 else -1 if value < 0.0 else 0
        for value in bystander_deltas
    }
    retry_signs = {
        1 if value > 0.0 else -1 if value < 0.0 else 0
        for value in retry_deltas
    }
    output = {
        "schema_version": "aegiskv-paired-repeatability-audit-v1",
        "evidence_class": "sglsim_smoke_repeatability_falsification",
        "paper_ready": False,
        "runs": rows,
        "run_count": len(rows),
        "bystander_delta": {
            "mean_s": math.fsum(bystander_deltas) / len(bystander_deltas),
            "min_s": min(bystander_deltas),
            "max_s": max(bystander_deltas),
            "sign_stable": len(bystander_signs) == 1 and 0 not in bystander_signs,
        },
        "retry_delta": {
            "mean_s": math.fsum(retry_deltas) / len(retry_deltas),
            "min_s": min(retry_deltas),
            "max_s": max(retry_deltas),
            "sign_stable": len(retry_signs) == 1 and 0 not in retry_signs,
        },
        "max_pre_failure_generated_token_gap": max(
            int(row["pre_failure_generated_token_gap"]) for row in rows
        ),
        "counterfactual_identification": "FAILED",
        "g1_status": "NOT_PROVEN",
        "required_next_protocol": [
            "rank-crossover assignment",
            "identical replayed arrival trace and fault trigger",
            "matched pre-failure state or covariate-adjusted estimator",
            "multiple paired seeds with confidence interval",
            "hardware-calibrated or real-system outcomes",
        ],
    }
    args.output.resolve().write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"run_count={output['run_count']}")
    print(
        "bystander_sign_stable="
        f"{output['bystander_delta']['sign_stable']}"
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
