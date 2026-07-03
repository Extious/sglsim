#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
DEFAULT_AGGREGATE_CSV = (
    ROOT
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "results"
    / "aggregate_overhead_pairs"
    / "candidate_fig_normal_path_overhead_sglang_aggregate.csv"
)
DEFAULT_FIG11_CSV = (
    ROOT
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_normal_path_overhead.csv"
)


POLICY_LABELS = {
    "baseline_no_hicache": "SGLang baseline",
    "hicache_write_through": "SGLang HiCache WT",
}


def read_csv(path: Path) -> list[dict[str, str]]:
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


def slurm_job_ids(run_id: str) -> str:
    parts = []
    for item in run_id.split("+"):
        if "_" in item:
            parts.append(item.rsplit("_", 1)[-1])
    return "+".join(parts)


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        policy = str(row.get("policy", ""))
        if policy not in POLICY_LABELS:
            continue
        run_id = str(row.get("run_id", ""))
        out.append(
            {
                "policy": POLICY_LABELS[policy],
                "p99_overhead_pct": row.get("p99_overhead_pct", ""),
                "write_amplification": row.get("write_amplification", "1.0"),
                "evidence_tier": row.get("evidence_tier", "sglang-small-paired"),
                "source": row.get(
                    "source", "sglang_dp2_overhead_pair_aggregate"
                ),
                "run_id": run_id,
                "git_sha": "",
                "remote_host": "hkbugpumanode",
                "slurm_job_id": slurm_job_ids(run_id),
                "model_path": "Qwen/Qwen3-8B",
                "workload_id": "dp2_overhead_pair",
                "dp_size": 2,
                "tp_size": 1,
                "failure_rank": "",
                "repeat": "aggregate",
                "note": row.get("note", ""),
            }
        )
    if len(out) != 2:
        raise ValueError("aggregate CSV must contain baseline and HiCache rows")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the current SGLang DP=2 overhead aggregate into Fig. 11 "
            "without regenerating other measured figure rows."
        )
    )
    parser.add_argument("--aggregate-csv", type=Path, default=DEFAULT_AGGREGATE_CSV)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_FIG11_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FIG11_CSV)
    args = parser.parse_args()

    existing = [dict(row) for row in read_csv(args.input_csv)]
    replacement = aggregate_rows(read_csv(args.aggregate_csv))
    retained = [
        row
        for row in existing
        if row.get("policy") not in {"SGLang baseline", "SGLang HiCache WT"}
    ]
    write_csv(args.output_csv, [*retained, *replacement])
    print(
        f"merged {len(replacement)} SGLang aggregate rows into {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
