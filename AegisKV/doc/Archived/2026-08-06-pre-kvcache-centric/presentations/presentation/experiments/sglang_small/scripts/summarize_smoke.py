#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[idx])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in args.input_jsonl.read_text().splitlines()
        if line.strip()
    ]
    latencies = [float(row["latency_s"]) for row in rows if int(row["status"]) < 500]
    summary = {
        "count": len(rows),
        "success_count": len(latencies),
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
    }
    (args.output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (args.output_dir / "smoke_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # This is intentionally preliminary and must not be copied into
    # figures/data/measured without matching baseline runs.
    with (args.output_dir / "candidate_fig_normal_path_overhead.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["policy", "p99_overhead_pct", "write_amplification", "note"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "policy": "aegiskv_write_through_smoke",
                "p99_overhead_pct": "",
                "write_amplification": "",
                "note": "smoke only; requires baseline paired run before use",
            }
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
