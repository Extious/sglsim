#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
DEFAULT_FIG11B_CSV = (
    ROOT
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_hostbackup_visibility.csv"
)


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


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def convert_pair_rows(pair_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in pair_rows:
        policy = row.get("policy", "")
        if policy not in {
            "SGLang remote_backup lookup disabled",
            "SGLang remote_backup lookup enabled",
        }:
            continue
        out.append(
            {
                "policy": policy,
                "p99_overhead_relative_pct": 0.0,
                "retry_latency_norm": as_float(row, "retry_latency_norm"),
                "evidence_tier": row.get("evidence_tier", "sglang-small-paired-smoke"),
                "source": row.get("source", "sglang_dp2_failure_visibility_pair"),
                "run_id": row.get("run_id", ""),
                "git_sha": "",
                "remote_host": row.get("remote_host", "hkbugpumanode"),
                "slurm_job_id": row.get("slurm_job_id", ""),
                "model_path": row.get("model_path", "Qwen/Qwen3-8B"),
                "workload_id": row.get("workload_id", "dp2_failure_smoke_4req_512tok_8s"),
                "dp_size": row.get("dp_size", "2"),
                "tp_size": row.get("tp_size", "1"),
                "failure_rank": row.get("failure_rank", "0"),
                "hicache_storage_prefetch_policy": row.get(
                    "hicache_storage_prefetch_policy", ""
                ),
                "repeat_count": row.get("request_count", ""),
                "repeat_run_ids": row.get("run_id", ""),
                "repeat": "single-pair",
                "note": row.get("note", ""),
                "runtime_replay_rows": row.get("failover_response_count", ""),
                "runtime_recovery_tokens": row.get("pre_failover_backed_up_tokens", ""),
                "runtime_regimes": row.get("role", ""),
            }
        )
    if len(out) != 2:
        raise ValueError(
            "pair CSV must contain lookup-disabled and lookup-enabled SGLang rows"
        )
    return out


def convert_aggregate_rows(
    aggregate_rows: list[dict[str, str]],
    *,
    pair_class: str,
) -> list[dict[str, object]]:
    selected = [row for row in aggregate_rows if row.get("pair_class") == pair_class]
    if len(selected) != 1:
        raise ValueError(
            f"aggregate CSV must contain exactly one row for pair_class={pair_class!r}"
        )
    row = selected[0]
    pair_count = row.get("pair_count", "")
    run_ids = row.get("repeat_run_ids", "")
    invisible_jobs = row.get("invisible_job_ids", "")
    visible_jobs = row.get("visible_job_ids", "")
    policy = row.get("hicache_storage_prefetch_policy", "")
    common = {
        "p99_overhead_relative_pct": 0.0,
        "evidence_tier": row.get("evidence_tier", "sglang-small-paired-smoke"),
        "source": row.get("source", "sglang_failure_visibility_pair_aggregate"),
        "run_id": pair_class,
        "git_sha": "",
        "remote_host": "hkbugpumanode",
        "model_path": "Qwen/Qwen3-8B",
        "workload_id": "dp2_failure_smoke_4req_512tok_12s_same_write",
        "dp_size": 2,
        "tp_size": 1,
        "failure_rank": 0,
        "hicache_storage_prefetch_policy": policy,
        "repeat_count": pair_count,
        "repeat_run_ids": run_ids,
        "repeat": f"mean-of-{pair_count}-pairs" if pair_count else "aggregate",
        "p99_overhead_relative_pct_std": 0.0,
        "p99_overhead_relative_pct_ci95": 0.0,
        "runtime_recovery_tokens": row.get("visible_backed_up_tokens_mean", ""),
    }
    return [
        {
            **common,
            "policy": "SGLang remote_backup lookup disabled",
            "retry_latency_norm": 1.0,
            "retry_latency_norm_std": 0.0,
            "retry_latency_norm_ci95": 0.0,
            "slurm_job_id": invisible_jobs,
            "note": (
                "aggregate same-write-cost control rows: remote_backup writes "
                "are enabled but failover lookup is disabled"
            ),
            "runtime_regimes": "host_invisible_remote_backup_backend_lookup_disabled",
        },
        {
            **common,
            "policy": "SGLang remote_backup lookup enabled",
            "retry_latency_norm": as_float(row, "visible_retry_latency_norm_mean"),
            "retry_latency_norm_std": as_float(row, "visible_retry_latency_norm_std"),
            "retry_latency_norm_ci95": as_float(row, "visible_retry_latency_norm_ci95"),
            "slurm_job_id": visible_jobs,
            "note": (
                "aggregate same-write-cost visible rows: remote_backup writes "
                "and best-effort failover lookup are both enabled"
            ),
            "runtime_regimes": "host_visible_remote_backup_backend_lookup_enabled",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a same-write-cost SGLang failure pair into Fig. 11b measured data."
        )
    )
    parser.add_argument("--pair-csv", type=Path, default=None)
    parser.add_argument("--aggregate-csv", type=Path, default=None)
    parser.add_argument("--pair-class", default="same_write_best_effort")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_FIG11B_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FIG11B_CSV)
    args = parser.parse_args()

    if args.pair_csv is not None and args.aggregate_csv is not None:
        raise SystemExit("use only one of --pair-csv or --aggregate-csv")
    if args.aggregate_csv is not None:
        replacement = convert_aggregate_rows(
            read_csv(args.aggregate_csv),
            pair_class=args.pair_class,
        )
    elif args.pair_csv is not None:
        replacement = convert_pair_rows(read_csv(args.pair_csv))
    else:
        raise SystemExit("one of --pair-csv or --aggregate-csv is required")
    replacement_policies = {str(row["policy"]) for row in replacement}
    existing = [dict(row) for row in read_csv(args.input_csv)]
    retained = [
        row for row in existing if row.get("policy") not in replacement_policies
    ]
    write_csv(args.output_csv, [*retained, *replacement])
    print(f"merged {len(replacement)} same-write-cost SGLang rows into {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
