#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from sglsim.srt.simulation.recovery_model import (
    DEFAULT_GPU_SETUP_MS,
    DEFAULT_HOST_SETUP_MS,
    ENV_PHYSICAL_COPY_CSV,
    ENV_RECOVERY_MODEL,
    aegis_restore_overhead_ms,
)


ROOT = Path(__file__).resolve().parents[6]


ACTION_SPECS = {
    "host_restore_h2d": {
        "path": "Host restore H2D",
        "model_action": "host_restore",
        "setup_ms": DEFAULT_HOST_SETUP_MS,
    },
    "gpu_replica_restore_p2p": {
        "path": "GPUReplica restore P2P",
        "model_action": "gpu_replica",
        "setup_ms": DEFAULT_GPU_SETUP_MS,
    },
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare SGLSim physical-copy recovery model against CUDA copy anchors."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    old_model = os.environ.get(ENV_RECOVERY_MODEL)
    old_csv = os.environ.get(ENV_PHYSICAL_COPY_CSV)
    os.environ[ENV_RECOVERY_MODEL] = "physical_copy"
    os.environ[ENV_PHYSICAL_COPY_CSV] = str(args.input_csv)

    current_sha = git_sha()
    output_rows: list[dict[str, object]] = []
    try:
        for row in read_rows(args.input_csv):
            action = str(row.get("action") or "")
            spec = ACTION_SPECS.get(action)
            if spec is None:
                continue
            tokens = int(float(row.get("tokens") or 0))
            measured_copy_ms = float(row.get("p50_ms_mean") or 0.0)
            modeled_overhead_ms = aegis_restore_overhead_ms(
                action=str(spec["model_action"]),
                token_count=tokens,
                restored_tokens=tokens,
            )
            modeled_copy_ms = modeled_overhead_ms - float(spec["setup_ms"])
            abs_error_ms = abs(modeled_copy_ms - measured_copy_ms)
            rel_error_pct = (
                100.0 * abs_error_ms / measured_copy_ms
                if measured_copy_ms > 0.0
                else 0.0
            )
            output_rows.append(
                {
                    "path": spec["path"],
                    "action": action,
                    "tokens": tokens,
                    "measured_copy_ms": measured_copy_ms,
                    "modeled_copy_ms": modeled_copy_ms,
                    "abs_error_ms": abs_error_ms,
                    "rel_error_pct": rel_error_pct,
                    "model_setup_ms": spec["setup_ms"],
                    "modeled_restore_overhead_ms": modeled_overhead_ms,
                    "repeat_count": row.get("repeat_count", ""),
                    "evidence_tier": "sglsim-model-physical-copy-calibration",
                    "source": "sglsim_recovery_model_physical_copy_calibration",
                    "run_id": args.run_id,
                    "git_sha": current_sha,
                    "remote_host": row.get("hosts", ""),
                    "slurm_job_id": row.get("slurm_job_ids", ""),
                    "workload_id": "recovery_model_physical_copy_calibration",
                    "note": (
                        "SGLSim recovery_model physical_copy mode reads the same "
                        "torch/CUDA physical recovery path aggregate; plotted error "
                        "checks that the simulator copy component matches those "
                        "physical anchors before adding per-path setup cost."
                    ),
                }
            )
    finally:
        if old_model is None:
            os.environ.pop(ENV_RECOVERY_MODEL, None)
        else:
            os.environ[ENV_RECOVERY_MODEL] = old_model
        if old_csv is None:
            os.environ.pop(ENV_PHYSICAL_COPY_CSV, None)
        else:
            os.environ[ENV_PHYSICAL_COPY_CSV] = old_csv

    output_rows.sort(key=lambda item: (str(item["path"]), int(item["tokens"])))
    write_csv(args.output_csv, output_rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "input_csv": str(args.input_csv),
                "output_csv": str(args.output_csv),
                "rows": len(output_rows),
                "max_abs_error_ms": max(
                    (float(row["abs_error_ms"]) for row in output_rows),
                    default=0.0,
                ),
                "max_rel_error_pct": max(
                    (float(row["rel_error_pct"]) for row in output_rows),
                    default=0.0,
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
