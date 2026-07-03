#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from derive_restore_boundary_from_replay import (
    fit_models,
    git_sha,
    replay_rows,
    row_pre_failure_tokens,
    selected_overhead_ms,
    write_csv,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def recompute_ms(model: dict[str, float], tokens: float) -> float:
    return model["recompute_intercept_ms"] + model["recompute_ms_per_token"] * tokens


def restore_ratios(
    model: dict[str, float],
    rows: list[dict[str, str]],
) -> list[float]:
    ratios = []
    for row in rows:
        baseline = recompute_ms(model, row_pre_failure_tokens(row))
        if baseline <= 0:
            continue
        ratios.append(selected_overhead_ms(row) / baseline)
    if not ratios:
        raise ValueError("no valid restore ratios")
    return ratios


def policy_row(
    *,
    policy: str,
    overhead_pct: float,
    ratios: list[float],
    base_meta: dict[str, object],
    note: str,
    regimes: str,
    replay_rows: int,
    recovery_tokens: int,
) -> dict[str, object]:
    return {
        "policy": policy,
        "p99_overhead_relative_pct": overhead_pct,
        "retry_latency_norm": mean(ratios),
        **base_meta,
        "repeat": f"mean-of-{len(ratios)}-replay-rows",
        "note": note,
        "p99_overhead_relative_pct_std": 0.0,
        "p99_overhead_relative_pct_ci95": 0.0,
        "retry_latency_norm_std": metric_std(ratios),
        "retry_latency_norm_ci95": metric_ci95(ratios),
        "runtime_replay_rows": replay_rows,
        "runtime_recovery_tokens": recovery_tokens,
        "runtime_regimes": regimes,
    }


def recovery_tokens(rows: list[dict[str, str]]) -> int:
    return int(sum(row_pre_failure_tokens(row) for row in rows))


def derive_rows(
    *,
    run_id: str,
    host_rows: list[dict[str, str]],
    gpu_rows: list[dict[str, str]],
    recompute_rows: list[dict[str, str]],
    run_dirs: list[Path],
) -> list[dict[str, object]]:
    model = fit_models(
        host_rows=host_rows,
        gpu_rows=gpu_rows,
        recompute_rows=recompute_rows,
    )
    host_ratios = restore_ratios(model, host_rows)
    gpu_ratios = restore_ratios(model, gpu_rows)
    recoverable_ratios = host_ratios + gpu_ratios
    baseline_ratios = [1.0 for _ in recoverable_ratios]

    base_meta = {
        "evidence_tier": "mixed-preliminary",
        "source": "sglsim_replay_retry_counterfactual_plus_policy_overhead_anchor",
        "run_id": run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": "aegiskv-sglsim-hostbackup-visibility",
        "dp_size": 2,
        "tp_size": 1,
        "failure_rank": 0,
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(str(path) for path in run_dirs),
        **model,
    }
    return [
        policy_row(
            policy="baseline write-through invisible",
            overhead_pct=0.0,
            ratios=baseline_ratios,
            base_meta=base_meta,
            note=(
                "Fixed write-through cost is the baseline; retry latency is "
                "same-row no-backup recompute for recoverable Host/GPU regimes"
            ),
            regimes="long_prefix_host;short_peer_gpu",
            replay_rows=len(host_rows) + len(gpu_rows),
            recovery_tokens=recovery_tokens(host_rows + gpu_rows),
        ),
        policy_row(
            policy="Host visible only",
            overhead_pct=0.0,
            ratios=host_ratios,
            base_meta=base_meta,
            note=(
                "Host visibility changes failure-time reachability without "
                "adding normal-path D2H beyond the fixed write-through baseline; "
                "retry latency uses long-prefix Host-visible replay rows"
            ),
            regimes="long_prefix_host",
            replay_rows=len(host_rows),
            recovery_tokens=recovery_tokens(host_rows),
        ),
        policy_row(
            policy="GPUReplica only",
            overhead_pct=4.9,
            ratios=gpu_ratios,
            base_meta=base_meta,
            note=(
                "Retry latency uses peer-visible hot-tail replay rows; p99 "
                "overhead is a preliminary policy-model peer-copy anchor, not "
                "final physical P2P evidence"
            ),
            regimes="short_peer_gpu",
            replay_rows=len(gpu_rows),
            recovery_tokens=recovery_tokens(gpu_rows),
        ),
        policy_row(
            policy="AegisKV",
            overhead_pct=1.425,
            ratios=recoverable_ratios,
            base_meta=base_meta,
            note=(
                "Retry latency aggregates Host-visible long-prefix and "
                "peer-visible hot-tail replay rows; p99 overhead is a "
                "preliminary selective-copy policy-model anchor"
            ),
            regimes="long_prefix_host;short_peer_gpu",
            replay_rows=len(host_rows) + len(gpu_rows),
            recovery_tokens=recovery_tokens(host_rows + gpu_rows),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 11b HostBackup visibility retry-latency rows from "
            "direct hard-failure replay regimes."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--host-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--gpu-run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--recompute-run-dir",
        type=Path,
        action="append",
        required=True,
    )
    args = parser.parse_args()

    host_rows = [
        row
        for row in replay_rows(args.host_run_dir)
        if str(row.get("aegis_replay_restore_action")) == "host_restore"
    ]
    gpu_rows = [
        row
        for row in replay_rows(args.gpu_run_dir)
        if str(row.get("aegis_replay_restore_action")) == "gpu_replica"
    ]
    recompute_rows = [
        row
        for row in replay_rows(args.recompute_run_dir)
        if str(row.get("aegis_replay_restore_action")) == "recompute"
    ]
    if not host_rows or not gpu_rows or not recompute_rows:
        raise ValueError("host, gpu, and recompute replay rows are required")

    run_dirs = [
        *args.host_run_dir,
        *args.gpu_run_dir,
        *args.recompute_run_dir,
    ]
    rows = derive_rows(
        run_id=args.run_id,
        host_rows=host_rows,
        gpu_rows=gpu_rows,
        recompute_rows=recompute_rows,
        run_dirs=run_dirs,
    )
    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "retry_latency_norm": {
                    row["policy"]: row["retry_latency_norm"] for row in rows
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
