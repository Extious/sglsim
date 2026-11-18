#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from derive_budget_gap_from_replay import (
    metric_ci95,
    metric_std,
    normalized_wct_for_overhead,
    parse_budget_points,
    static_restore_ms,
)
from derive_restore_boundary_from_replay import (
    _float,
    fit_models,
    git_sha,
    replay_rows,
    row_pre_failure_tokens,
    write_csv,
)


REGIMES = (
    ("hot-tail", "hot tails", "gpu_replica"),
    ("long-prefix", "long prefixes", "host_restore"),
    ("non-peer", "fan-out skew", "recompute"),
)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("no budget-sensitivity values to summarize")
    return {
        "normalized_wct": mean(values),
        "normalized_wct_std": metric_std(values),
        "normalized_wct_ci95": metric_ci95(values),
    }


def sensitivity_rows_for_regime(
    *,
    model: dict[str, float],
    rows: list[dict[str, str]],
    timing_label: str,
    workload: str,
    budgets: list[float],
    base_meta: dict[str, object],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    recovery_tokens = sum(row_pre_failure_tokens(row) for row in rows)
    peer_visible_tokens = sum(_float(row.get("aegis_peer_gpu_lookup_tokens")) for row in rows)
    for budget in budgets:
        ratios: list[float] = []
        restore_values: list[float] = []
        for row in rows:
            restore_ms = static_restore_ms(
                model,
                row,
                workload=workload,
                budget=budget,
            )
            ratio = normalized_wct_for_overhead(model, row, restore_ms)
            if ratio is not None:
                ratios.append(ratio)
                restore_values.append(restore_ms)
        summary = summarize(ratios)
        out.append(
            {
                "failure_timing": timing_label,
                "gpu_replica_budget_fraction": round(budget, 2),
                **summary,
                **base_meta,
                "repeat": f"mean-of-{len(rows)}-replay-rows",
                "note": (
                    "Replay-calibrated budget sensitivity over observed recovery "
                    "regimes. The failure_timing field is a current-regime proxy "
                    "for Fig. 13, not the final controlled timing sweep."
                ),
                "runtime_replay_rows": len(rows),
                "runtime_recovery_tokens": int(recovery_tokens),
                "runtime_peer_visible_tokens": int(peer_visible_tokens),
                "runtime_peer_visible_token_fraction": (
                    peer_visible_tokens / recovery_tokens if recovery_tokens > 0 else 0.0
                ),
                "runtime_mean_static_restore_ms": mean(restore_values),
                "runtime_workload_proxy": workload,
                **model,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 13 GPUReplica budget-sensitivity heatmap from direct "
            "hard-failure replay rows and the Fig. 2 calibrated recovery model."
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
    parser.add_argument(
        "--budget-points",
        type=parse_budget_points,
        default=parse_budget_points("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"),
    )
    args = parser.parse_args()

    rows_by_action = {
        "host_restore": [
            row
            for row in replay_rows(args.host_run_dir)
            if str(row.get("aegis_replay_restore_action")) == "host_restore"
        ],
        "gpu_replica": [
            row
            for row in replay_rows(args.gpu_run_dir)
            if str(row.get("aegis_replay_restore_action")) == "gpu_replica"
        ],
        "recompute": [
            row
            for row in replay_rows(args.recompute_run_dir)
            if str(row.get("aegis_replay_restore_action")) == "recompute"
        ],
    }
    if any(not rows for rows in rows_by_action.values()):
        raise ValueError(
            "host, gpu, and recompute replay rows are all required for Fig. 13"
        )

    model = fit_models(
        host_rows=rows_by_action["host_restore"],
        gpu_rows=rows_by_action["gpu_replica"],
        recompute_rows=rows_by_action["recompute"],
    )
    run_dirs = [
        *[str(path) for path in args.host_run_dir],
        *[str(path) for path in args.gpu_run_dir],
        *[str(path) for path in args.recompute_run_dir],
    ]
    base_meta = {
        "evidence_tier": "sglsim-runtime-replay-calibrated",
        "source": "sglsim_request_details_replay_budget_sensitivity_counterfactual",
        "run_id": args.run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": "aegiskv-sglsim-budget-sensitivity",
        "dp_size": 2,
        "tp_size": 1,
        "failure_rank": 0,
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(run_dirs),
    }

    rows: list[dict[str, object]] = []
    for label, workload, action in REGIMES:
        rows.extend(
            sensitivity_rows_for_regime(
                model=model,
                rows=rows_by_action[action],
                timing_label=label,
                workload=workload,
                budgets=args.budget_points,
                base_meta=base_meta,
            )
        )

    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "runtime_replay_rows": sum(len(rows) for rows in rows_by_action.values()),
                "regimes": [label for label, _, _ in REGIMES],
                "model": model,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
