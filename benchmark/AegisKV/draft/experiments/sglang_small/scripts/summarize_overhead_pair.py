#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(case_dir: Path) -> dict[str, float]:
    path = case_dir / "summary.json"
    if not path.exists():
        path = case_dir / "smoke_summary.json"
    return json.loads(path.read_text())


def pct_overhead(value: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--hicache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_summary(args.baseline_dir)
    hicache = load_summary(args.hicache_dir)

    p99_base = float(baseline["latency_p99_s"])
    p99_hicache = float(hicache["latency_p99_s"])
    p95_base = float(baseline["latency_p95_s"])
    p95_hicache = float(hicache["latency_p95_s"])
    mean_base = float(baseline["latency_mean_s"])
    mean_hicache = float(hicache["latency_mean_s"])

    pair = {
        "run_id": args.run_id,
        "baseline": baseline,
        "hicache_write_through": hicache,
        "p99_overhead_pct": pct_overhead(p99_hicache, p99_base),
        "p95_overhead_pct": pct_overhead(p95_hicache, p95_base),
        "mean_overhead_pct": pct_overhead(mean_hicache, mean_base),
        "note": "paired same-node DP=2 SGLang no-failure smoke; not full AegisKV policy evidence",
    }
    (args.output_dir / "paired_overhead_summary.json").write_text(
        json.dumps(pair, indent=2) + "\n"
    )

    with (args.output_dir / "paired_overhead_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "baseline_p99_s",
                "hicache_p99_s",
                "p99_overhead_pct",
                "baseline_p95_s",
                "hicache_p95_s",
                "p95_overhead_pct",
                "baseline_mean_s",
                "hicache_mean_s",
                "mean_overhead_pct",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": args.run_id,
                "baseline_p99_s": p99_base,
                "hicache_p99_s": p99_hicache,
                "p99_overhead_pct": pair["p99_overhead_pct"],
                "baseline_p95_s": p95_base,
                "hicache_p95_s": p95_hicache,
                "p95_overhead_pct": pair["p95_overhead_pct"],
                "baseline_mean_s": mean_base,
                "hicache_mean_s": mean_hicache,
                "mean_overhead_pct": pair["mean_overhead_pct"],
            }
        )

    with (args.output_dir / "candidate_fig_normal_path_overhead_sglang_pair.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "p99_overhead_pct",
                "write_amplification",
                "evidence_tier",
                "source",
                "run_id",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "policy": "baseline_no_hicache",
                "p99_overhead_pct": 0.0,
                "write_amplification": 1.0,
                "evidence_tier": "sglang-small-paired",
                "source": "sglang_dp2_overhead_pair",
                "run_id": args.run_id,
                "note": "baseline for paired no-failure overhead",
            }
        )
        writer.writerow(
            {
                "policy": "hicache_write_through",
                "p99_overhead_pct": pair["p99_overhead_pct"],
                "write_amplification": 1.0,
                "evidence_tier": "sglang-small-paired",
                "source": "sglang_dp2_overhead_pair",
                "run_id": args.run_id,
                "note": "fixed write-through HiCache path; paired with baseline in same Slurm job",
            }
        )

    print(json.dumps(pair, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
