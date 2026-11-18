#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


FIELDS = [
    "label",
    "run_dir",
    "experiment_name",
    "model_path",
    "dp_size",
    "tp_size",
    "pp_size",
    "num_requests",
    "gsp_num_turns",
    "max_concurrency",
    "completed",
    "failed",
    "duration",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "max_e2e_latency_ms",
    "total_input",
    "total_output",
    "aegis_host_visible_rate",
    "aegis_gpu_replica_rate",
    "aegis_safe_frontier_rate",
    "aegis_recompute_rate",
    "aegis_expected_recovery_ms_mean",
    "aegis_candidate_retry_count",
    "aegis_candidate_backed_up_rate",
    "aegis_candidate_recompute_rate",
    "aegis_candidate_peer_gpu_rate",
    "aegis_candidate_host_restore_rate",
    "aegis_candidate_ordinary_cache_rate",
    "aegis_candidate_mean_failover_overhead_ms",
    "aegis_failover_replay_count",
    "aegis_cross_rank_replay_count",
    "aegis_soft_failure_injected_replay_count",
    "aegis_hard_failure_injected_replay_count",
    "aegis_host_backup_lookup_hit_replay_count",
    "aegis_host_backup_lookup_tokens",
    "aegis_peer_gpu_lookup_hit_replay_count",
    "aegis_peer_gpu_lookup_tokens",
    "aegis_replay_restored_tokens",
    "aegis_replay_restored_prefill_tokens",
    "aegis_replay_recompute_tokens",
    "aegis_replay_mean_restore_sleep_ms",
    "note",
]


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_request_detail_rates(run_dir: Path) -> dict[str, float]:
    path = run_dir / "request_details.csv"
    if not path.exists():
        return {
            "aegis_host_visible_rate": 0.0,
            "aegis_gpu_replica_rate": 0.0,
            "aegis_safe_frontier_rate": 0.0,
            "aegis_recompute_rate": 0.0,
            "aegis_expected_recovery_ms_mean": 0.0,
            "aegis_candidate_retry_count": 0.0,
            "aegis_candidate_backed_up_rate": 0.0,
            "aegis_candidate_recompute_rate": 0.0,
            "aegis_candidate_peer_gpu_rate": 0.0,
            "aegis_candidate_host_restore_rate": 0.0,
            "aegis_candidate_ordinary_cache_rate": 0.0,
            "aegis_candidate_mean_failover_overhead_ms": 0.0,
            "aegis_failover_replay_count": 0.0,
            "aegis_cross_rank_replay_count": 0.0,
            "aegis_soft_failure_injected_replay_count": 0.0,
            "aegis_hard_failure_injected_replay_count": 0.0,
            "aegis_host_backup_lookup_hit_replay_count": 0.0,
            "aegis_host_backup_lookup_tokens": 0.0,
            "aegis_peer_gpu_lookup_hit_replay_count": 0.0,
            "aegis_peer_gpu_lookup_tokens": 0.0,
            "aegis_replay_restored_tokens": 0.0,
            "aegis_replay_restored_prefill_tokens": 0.0,
            "aegis_replay_recompute_tokens": 0.0,
            "aegis_replay_mean_restore_sleep_ms": 0.0,
        }
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total_tokens = sum(
        max(_float(row.get("input_length")), 0.0)
        + max(_float(row.get("generated_tokens")), 0.0)
        for row in rows
    )
    recovery_ms = [
        _float(row.get("aegis_expected_recovery_ms"))
        for row in rows
        if _float(row.get("aegis_expected_recovery_ms")) > 0
    ]
    candidate_rows = [
        row for row in rows if _bool(row.get("aegis_candidate_retry", False))
    ]
    candidate_total = sum(
        max(_float(row.get("aegis_candidate_pre_failure_total_tokens")), 0.0)
        for row in candidate_rows
    )
    candidate_overhead_ms = [
        _float(row.get("aegis_candidate_failover_overhead_s")) * 1000.0
        for row in candidate_rows
        if _float(row.get("aegis_candidate_failover_overhead_s")) > 0
    ]
    replay_rows = [
        row for row in rows if _bool(row.get("aegis_failover_replay", False))
    ]
    cross_rank_replay_rows = [
        row for row in replay_rows if _bool(row.get("aegis_cross_rank_retry", False))
    ]
    soft_failure_rows = [
        row
        for row in replay_rows
        if _bool(row.get("aegis_soft_failure_injected", False))
    ]
    hard_failure_rows = [
        row
        for row in replay_rows
        if _bool(row.get("aegis_hard_failure_injected", False))
    ]
    host_backup_lookup_rows = [
        row
        for row in replay_rows
        if _bool(row.get("aegis_host_backup_lookup_hit", False))
    ]
    peer_gpu_lookup_rows = [
        row
        for row in replay_rows
        if _bool(row.get("aegis_peer_gpu_lookup_hit", False))
    ]
    replay_sleep_ms = [
        _float(row.get("aegis_replay_restore_sleep_s")) * 1000.0
        for row in replay_rows
        if _float(row.get("aegis_replay_restore_sleep_s")) > 0
    ]

    def rate(column: str) -> float:
        if total_tokens <= 0:
            return 0.0
        return sum(max(_float(row.get(column)), 0.0) for row in rows) / total_tokens

    def candidate_rate(column: str, *, action: str | None = None) -> float:
        if candidate_total <= 0:
            return 0.0
        source = candidate_rows
        if action is not None:
            source = [
                row
                for row in candidate_rows
                if str(row.get("aegis_candidate_recovery_action", "")) == action
            ]
        return sum(max(_float(row.get(column)), 0.0) for row in source) / candidate_total

    return {
        "aegis_host_visible_rate": rate("aegis_host_visible_tokens"),
        "aegis_gpu_replica_rate": rate("aegis_gpu_replica_tokens"),
        "aegis_safe_frontier_rate": rate("aegis_safe_frontier_tokens"),
        "aegis_recompute_rate": rate("aegis_recompute_tokens"),
        "aegis_expected_recovery_ms_mean": mean(recovery_ms) if recovery_ms else 0.0,
        "aegis_candidate_retry_count": float(len(candidate_rows)),
        "aegis_candidate_backed_up_rate": candidate_rate(
            "aegis_candidate_backed_up_tokens"
        ),
        "aegis_candidate_recompute_rate": candidate_rate(
            "aegis_candidate_recompute_tokens"
        ),
        "aegis_candidate_peer_gpu_rate": candidate_rate(
            "aegis_candidate_backed_up_tokens",
            action="gpu_replica",
        ),
        "aegis_candidate_host_restore_rate": candidate_rate(
            "aegis_candidate_backed_up_tokens",
            action="host_restore",
        ),
        "aegis_candidate_ordinary_cache_rate": candidate_rate(
            "aegis_candidate_backed_up_tokens",
            action="ordinary_cache",
        ),
        "aegis_candidate_mean_failover_overhead_ms": (
            mean(candidate_overhead_ms) if candidate_overhead_ms else 0.0
        ),
        "aegis_failover_replay_count": float(len(replay_rows)),
        "aegis_cross_rank_replay_count": float(len(cross_rank_replay_rows)),
        "aegis_soft_failure_injected_replay_count": float(len(soft_failure_rows)),
        "aegis_hard_failure_injected_replay_count": float(len(hard_failure_rows)),
        "aegis_host_backup_lookup_hit_replay_count": float(
            len(host_backup_lookup_rows)
        ),
        "aegis_host_backup_lookup_tokens": sum(
            max(_float(row.get("aegis_host_backup_lookup_tokens")), 0.0)
            for row in replay_rows
        ),
        "aegis_peer_gpu_lookup_hit_replay_count": float(len(peer_gpu_lookup_rows)),
        "aegis_peer_gpu_lookup_tokens": sum(
            max(_float(row.get("aegis_peer_gpu_lookup_tokens")), 0.0)
            for row in replay_rows
        ),
        "aegis_replay_restored_tokens": sum(
            max(_float(row.get("aegis_replay_restored_tokens")), 0.0)
            for row in replay_rows
        ),
        "aegis_replay_restored_prefill_tokens": sum(
            max(_float(row.get("aegis_replay_restored_prefill_tokens")), 0.0)
            for row in replay_rows
        ),
        "aegis_replay_recompute_tokens": sum(
            max(_float(row.get("aegis_replay_recompute_tokens")), 0.0)
            for row in replay_rows
        ),
        "aegis_replay_mean_restore_sleep_ms": (
            mean(replay_sleep_ms) if replay_sleep_ms else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--label", default="sglsim_smoke")
    parser.add_argument(
        "--note",
        default="smoke only; do not copy into figures/data/measured",
    )
    args = parser.parse_args()

    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    config = json.loads((args.run_dir / "simulator_config.json").read_text())
    scheduler = config.get("scheduler") or {}
    workload = config.get("workload") or {}
    aegis_rates = _read_request_detail_rates(args.run_dir)
    row = {
        "label": args.label,
        "run_dir": str(args.run_dir),
        "experiment_name": config.get("experiment_name") or config.get("experiment", {}).get("name", ""),
        "model_path": config.get("model_path", ""),
        "dp_size": scheduler.get("dp_size", ""),
        "tp_size": scheduler.get("tp_size", ""),
        "pp_size": scheduler.get("pp_size", ""),
        "num_requests": workload.get("num_requests", ""),
        "gsp_num_turns": workload.get("gsp_num_turns", ""),
        "max_concurrency": workload.get("max_concurrency", ""),
        "completed": metrics.get("completed", ""),
        "failed": metrics.get("failed", ""),
        "duration": metrics.get("duration", ""),
        "mean_e2e_latency_ms": metrics.get("mean_e2e_latency_ms", ""),
        "median_e2e_latency_ms": metrics.get("median_e2e_latency_ms", ""),
        "p90_e2e_latency_ms": metrics.get("p90_e2e_latency_ms", ""),
        "max_e2e_latency_ms": metrics.get("max_e2e_latency_ms", ""),
        "total_input": metrics.get("total_input", ""),
        "total_output": metrics.get("total_output", ""),
        **aegis_rates,
        "note": args.note,
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)

    print(args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
