#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[6]


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


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_summary(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    row = rows[0]
    row["_summary_csv"] = str(path)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
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
        description="Derive Fig. 14 direct DP-scale CSV from SGLSim run summaries."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, action="append", required=True)
    parser.add_argument("--baseline-dp", type=int, default=2)
    args = parser.parse_args()

    source_rows = [read_summary(path) for path in args.summary_csv]
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        completed = _int(row.get("completed"))
        failed = _int(row.get("failed"))
        if completed <= 0 or failed != 0:
            raise ValueError(
                f"bad summary {row['_summary_csv']}: completed={completed} failed={failed}"
            )
        grouped[_int(row.get("dp_size"), 0)].append(row)

    if args.baseline_dp not in grouped:
        raise ValueError(f"baseline dp {args.baseline_dp} not present")
    baseline_values = [_float(row.get("duration")) for row in grouped[args.baseline_dp]]
    baseline_mean = mean(baseline_values)
    if baseline_mean <= 0:
        raise ValueError("baseline duration must be positive")

    current_sha = git_sha()
    output_rows: list[dict[str, object]] = []
    for dp_size in sorted(grouped):
        rows = grouped[dp_size]
        durations = [_float(row.get("duration")) for row in rows]
        normalized_values = [value / baseline_mean for value in durations]
        repeat_count = len(rows)
        normalized_std = stdev(normalized_values) if repeat_count > 1 else 0.0
        normalized_ci95 = (
            1.96 * normalized_std / math.sqrt(repeat_count)
            if repeat_count > 1
            else 0.0
        )
        first = rows[0]
        workflow_count = _int(first.get("num_requests"), 0)
        output_rows.append(
            {
                "dp_size": dp_size,
                "workflow_count": workflow_count,
                "normalized_wct": mean(normalized_values),
                "evidence_tier": "sglsim-runtime-replay-derived",
                "source": "sglsim_direct_dp_scale_sweep",
                "run_id": args.run_id,
                "git_sha": current_sha,
                "remote_host": "",
                "slurm_job_id": "",
                "model_path": first.get("model_path", ""),
                "workload_id": first.get("experiment_name", "dp_scale_sweep"),
                "tp_size": first.get("tp_size", 1),
                "failure_rank": 0,
                "repeat": f"mean-of-{repeat_count}",
                "note": (
                    "direct SGLSim hard-failover replay DP-size sweep; "
                    "normalized_wct uses run duration normalized to DP=2; "
                    "not final physical KV-copy evidence"
                ),
                "duration_s": mean(durations),
                "duration_s_std": stdev(durations) if repeat_count > 1 else 0.0,
                "duration_s_ci95": (
                    1.96 * stdev(durations) / math.sqrt(repeat_count)
                    if repeat_count > 1
                    else 0.0
                ),
                "mean_e2e_latency_ms": mean(
                    _float(row.get("mean_e2e_latency_ms")) for row in rows
                ),
                "replay_rows": sum(_int(row.get("aegis_failover_replay_count")) for row in rows),
                "hard_failure_replay_rows": sum(
                    _int(row.get("aegis_hard_failure_injected_replay_count"))
                    for row in rows
                ),
                "repeat_count": repeat_count,
                "repeat_run_ids": ";".join(row.get("run_dir", "") for row in rows),
            }
        )

    write_csv(args.output_csv, output_rows)
    summary = {
        "run_id": args.run_id,
        "output_csv": str(args.output_csv),
        "baseline_dp": args.baseline_dp,
        "dp_sizes": sorted(grouped),
        "rows": len(output_rows),
        "source_summary_csvs": [str(path) for path in args.summary_csv],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
