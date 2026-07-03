#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from derive_restore_boundary_from_replay import _bool, _float, git_sha, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frontier_tokens(row: dict[str, str]) -> float:
    for key in (
        "aegis_safe_frontier_tokens",
        "aegis_host_visible_tokens",
        "aegis_candidate_pre_failure_total_tokens",
    ):
        value = _float(row.get(key))
        if value > 0:
            return value
    return _float(row.get("input_length")) + _float(row.get("generated_tokens"))


def original_request_rows(run_dir: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(run_dir / "request_details.csv"):
        if _bool(row.get("aegis_failover_replay")):
            continue
        if _bool(row.get("aegis_cross_rank_retry")):
            continue
        if not str(row.get("agent_id") or "").strip():
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"no original AegisKV request rows in {run_dir}")
    return rows


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def group_key(run_dir: Path, row: dict[str, str]) -> tuple[str, str]:
    return str(run_dir), str(row.get("agent_id") or "")


def group_metrics(rows: list[dict[str, str]]) -> dict[str, float] | None:
    ordered = sorted(rows, key=lambda row: int(_float(row.get("aegis_worker_seq"))))
    if len(ordered) < 2:
        return None
    frontiers = [frontier_tokens(row) for row in ordered]
    if any(value <= 0 for value in frontiers):
        return None
    no_dedup = sum(frontiers)
    marginal = frontiers[0] + sum(
        max(frontiers[idx] - frontiers[idx - 1], 0.0)
        for idx in range(1, len(frontiers))
    )
    latest = max(frontiers)
    if no_dedup <= 0 or latest <= 0:
        return None
    shared_pct = 100.0 * (1.0 - marginal / no_dedup)
    return {
        "shared_prefix_pct": shared_pct,
        "dup_no_dedup": no_dedup / latest,
        "dup_marginal": marginal / latest,
        "coverage_marginal": min(1.0, marginal / latest),
        "turn_count": float(len(frontiers)),
        "latest_frontier_tokens": latest,
        "no_dedup_tokens": no_dedup,
        "marginal_tokens": marginal,
    }


def bucket_center(value: float, width: float) -> float:
    return math.floor(value / width) * width + width / 2.0


def summarize_bucket(
    *,
    center: float,
    metrics: list[dict[str, float]],
    base_meta: dict[str, object],
) -> dict[str, object]:
    dup_no = [row["dup_no_dedup"] for row in metrics]
    dup_marg = [row["dup_marginal"] for row in metrics]
    coverage = [row["coverage_marginal"] for row in metrics]
    shared = [row["shared_prefix_pct"] for row in metrics]
    return {
        "shared_prefix_pct": round(center, 2),
        "dup_no_dedup": mean(dup_no),
        "dup_marginal": mean(dup_marg),
        "coverage_marginal": mean(coverage),
        **base_meta,
        "repeat": f"mean-of-{len(metrics)}-agent-groups",
        "note": (
            "Observed multi-turn AegisKV request groups: no-dedup bytes sum "
            "each turn's safe frontier, marginal bytes protect only newly "
            "extended frontier tokens within the same agent; retry rows are "
            "excluded so cache-hit counters do not define shared-prefix ratio"
        ),
        "dup_no_dedup_std": metric_std(dup_no),
        "dup_no_dedup_ci95": metric_ci95(dup_no),
        "dup_marginal_std": metric_std(dup_marg),
        "dup_marginal_ci95": metric_ci95(dup_marg),
        "coverage_marginal_std": metric_std(coverage),
        "coverage_marginal_ci95": metric_ci95(coverage),
        "runtime_agent_groups": len(metrics),
        "runtime_mean_observed_shared_prefix_pct": mean(shared),
        "runtime_mean_turn_count": mean([row["turn_count"] for row in metrics]),
        "runtime_latest_frontier_tokens": int(
            sum(row["latest_frontier_tokens"] for row in metrics)
        ),
        "runtime_no_dedup_tokens": int(sum(row["no_dedup_tokens"] for row in metrics)),
        "runtime_marginal_tokens": int(sum(row["marginal_tokens"] for row in metrics)),
    }


def derive_rows(
    *,
    run_dirs: list[Path],
    run_id: str,
    bin_width: float,
    min_group_count: int,
    evidence_tier: str,
    source: str,
    workload_id: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    request_rows = 0
    for run_dir in run_dirs:
        rows = original_request_rows(run_dir)
        request_rows += len(rows)
        for row in rows:
            groups.setdefault(group_key(run_dir, row), []).append(row)

    metrics = [value for rows in groups.values() if (value := group_metrics(rows))]
    if not metrics:
        raise ValueError("no multi-turn agent groups found")

    buckets: dict[float, list[dict[str, float]]] = {}
    for item in metrics:
        buckets.setdefault(
            bucket_center(item["shared_prefix_pct"], bin_width), []
        ).append(item)

    run_dir_text = ";".join(str(path) for path in run_dirs)
    base_meta = {
        "evidence_tier": evidence_tier,
        "source": source,
        "run_id": run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": workload_id,
        "dp_size": 2,
        "tp_size": 1,
        "failure_rank": 0,
        "repeat_count": len(run_dirs),
        "repeat_run_ids": run_dir_text,
        "runtime_request_rows": request_rows,
        "runtime_total_agent_groups": len(metrics),
    }
    rows = [
        summarize_bucket(center=center, metrics=items, base_meta=base_meta)
        for center, items in sorted(buckets.items())
        if len(items) >= min_group_count
    ]
    if not rows:
        raise ValueError(
            f"no shared-prefix buckets have at least {min_group_count} groups"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 10b shared-prefix marginal accounting from original "
            "multi-turn request rows in SGLSim hard-failure matrices."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--bin-width", type=float, default=5.0)
    parser.add_argument("--min-group-count", type=int, default=5)
    parser.add_argument(
        "--evidence-tier",
        default="sglsim-runtime-request-derived",
        help=(
            "Evidence tier to write into the output CSV. Use final only after "
            "the sweep satisfies the final evidence matrix requirement."
        ),
    )
    parser.add_argument(
        "--source",
        default="sglsim_request_details_original_shared_prefix_marginal",
    )
    parser.add_argument(
        "--workload-id",
        default="aegiskv-sglsim-shared-prefix-marginal",
    )
    parser.add_argument(
        "--min-shared-prefix-span",
        type=float,
        default=0.0,
        help="Fail unless max(shared_prefix_pct)-min(shared_prefix_pct) reaches this span.",
    )
    args = parser.parse_args()

    rows = derive_rows(
        run_dirs=args.run_dir,
        run_id=args.run_id,
        bin_width=args.bin_width,
        min_group_count=args.min_group_count,
        evidence_tier=args.evidence_tier,
        source=args.source,
        workload_id=args.workload_id,
    )
    shared_values = [float(row["shared_prefix_pct"]) for row in rows]
    if args.min_shared_prefix_span > 0 and (
        max(shared_values) - min(shared_values)
    ) < args.min_shared_prefix_span:
        raise ValueError(
            "shared-prefix sweep span is too narrow: "
            f"{min(shared_values):.2f}..{max(shared_values):.2f} "
            f"< required span {args.min_shared_prefix_span:.2f}"
        )
    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "shared_prefix_pct": shared_values,
                "shared_prefix_span": max(shared_values) - min(shared_values),
                "agent_groups": sum(int(row["runtime_agent_groups"]) for row in rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
