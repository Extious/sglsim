#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]


ACTION_LABELS = {
    "host_restore_h2d": ("Host restore H2D", "restore"),
    "host_backup_write_d2h": ("Host backup D2H", "backup"),
    "gpu_replica_restore_p2p": ("GPUReplica restore P2P", "restore"),
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
        description="Build Fig. 15b physical recovery copy calibration rows."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    current_sha = git_sha()
    output_rows: list[dict[str, object]] = []
    for row in read_rows(args.input_csv):
        action = str(row.get("action") or "")
        if action not in ACTION_LABELS:
            continue
        path_label, phase = ACTION_LABELS[action]
        output_rows.append(
            {
                "path": path_label,
                "action": action,
                "phase": phase,
                "tokens": int(float(row.get("tokens") or 0)),
                "mib": row.get("mib", ""),
                "p50_ms": row.get("p50_ms_mean", ""),
                "p50_ms_ci95": row.get("p50_ms_ci95", ""),
                "p95_ms": row.get("p95_ms_mean", ""),
                "p95_ms_ci95": row.get("p95_ms_ci95", ""),
                "p50_gbps": row.get("p50_gbps_mean", ""),
                "p50_gbps_ci95": row.get("p50_gbps_ci95", ""),
                "bytes_per_token": row.get("bytes_per_token", ""),
                "repeat_count": row.get("repeat_count", ""),
                "evidence_tier": row.get(
                    "evidence_tier",
                    "sglang-small-physical-recovery-movement-smoke",
                ),
                "source": row.get(
                    "source",
                    "torch_cuda_physical_recovery_path_microbench_aggregate",
                ),
                "run_id": args.run_id,
                "git_sha": current_sha,
                "remote_host": row.get("hosts", ""),
                "slurm_job_id": row.get("slurm_job_ids", ""),
                "workload_id": "physical_recovery_copy_calibration",
                "note": (
                    "CUDA movement microbench shaped as AegisKV backup/restore "
                    "copy paths; used to calibrate SGLSim recovery timing, not "
                    "as end-to-end policy evidence."
                ),
            }
        )

    output_rows.sort(key=lambda item: (str(item["path"]), int(item["tokens"])))
    write_csv(args.output_csv, output_rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "input_csv": str(args.input_csv),
                "output_csv": str(args.output_csv),
                "rows": len(output_rows),
                "paths": sorted({str(row["path"]) for row in output_rows}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
