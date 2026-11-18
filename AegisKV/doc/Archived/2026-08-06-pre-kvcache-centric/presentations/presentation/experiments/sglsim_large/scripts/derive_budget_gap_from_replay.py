#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from derive_restore_boundary_from_replay import (
    _float,
    fit_models,
    git_sha,
    replay_rows,
    row_pre_failure_tokens,
    selected_overhead_ms,
    write_csv,
)


def row_e2e_ms(row: dict[str, str]) -> float:
    for key in ("client_wall_e2e_latency_s", "e2e_latency_s", "total_latency_s"):
        value = _float(row.get(key))
        if value > 0:
            return value * 1000.0
    return 0.0


def recompute_ms(model: dict[str, float], tokens: float) -> float:
    return model["recompute_intercept_ms"] + model["recompute_ms_per_token"] * tokens


def host_ms(model: dict[str, float], tokens: float) -> float:
    return model["host_intercept_ms"] + model["host_ms_per_token"] * tokens


def gpu_plus_recompute_ms(
    model: dict[str, float],
    *,
    peer_tokens: float,
    suffix_tokens: float,
) -> float:
    if peer_tokens <= 0:
        return recompute_ms(model, suffix_tokens)
    return (
        model["gpu_intercept_ms"]
        + model["gpu_ms_per_token"] * peer_tokens
        + model["recompute_ms_per_token"] * max(suffix_tokens, 0.0)
    )


def gpu_plus_host_ms(
    model: dict[str, float],
    *,
    peer_tokens: float,
    host_tokens: float,
) -> float:
    if peer_tokens <= 0:
        return host_ms(model, host_tokens)
    value = model["gpu_intercept_ms"] + model["gpu_ms_per_token"] * peer_tokens
    if host_tokens > 0:
        value += model["host_intercept_ms"] + model["host_ms_per_token"] * host_tokens
    return value


def static_restore_ms(
    model: dict[str, float],
    row: dict[str, str],
    *,
    workload: str,
    budget: float,
) -> float:
    tokens = max(row_pre_failure_tokens(row), 0.0)
    peer_visible = max(_float(row.get("aegis_peer_gpu_lookup_tokens")), 0.0)
    peer_tokens = min(peer_visible, budget * tokens)
    suffix_tokens = max(tokens - peer_tokens, 0.0)

    if workload == "long prefixes":
        return gpu_plus_host_ms(
            model,
            peer_tokens=peer_tokens,
            host_tokens=suffix_tokens,
        )
    if workload in {"hot tails", "fan-out skew"}:
        return gpu_plus_recompute_ms(
            model,
            peer_tokens=peer_tokens,
            suffix_tokens=suffix_tokens,
        )
    raise ValueError(f"unknown workload: {workload}")


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("no values to summarize")
    return {
        "normalized_wct": mean(values),
        "normalized_wct_std": metric_std(values),
        "normalized_wct_ci95": metric_ci95(values),
    }


def normalized_wct_for_overhead(
    model: dict[str, float],
    row: dict[str, str],
    restore_overhead_ms: float,
) -> float | None:
    e2e = row_e2e_ms(row)
    if e2e <= 0:
        return None
    selected = selected_overhead_ms(row)
    tokens = row_pre_failure_tokens(row)
    no_backup = e2e - selected + recompute_ms(model, tokens)
    if no_backup <= 0:
        return None
    static_e2e = e2e - selected + restore_overhead_ms
    return static_e2e / no_backup


def budget_rows_for_workload(
    *,
    model: dict[str, float],
    rows: list[dict[str, str]],
    workload: str,
    budgets: list[float],
    base_meta: dict[str, object],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    recovery_tokens = sum(row_pre_failure_tokens(row) for row in rows)
    peer_visible_tokens = sum(_float(row.get("aegis_peer_gpu_lookup_tokens")) for row in rows)
    for budget in budgets:
        ratios: list[float] = []
        static_restore_values: list[float] = []
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
                static_restore_values.append(restore_ms)
        summary = summarize(ratios)
        out.append(
            {
                "gpu_replica_budget_fraction": round(budget, 2),
                "workload": workload,
                **summary,
                **base_meta,
                "repeat": f"mean-of-{len(rows)}-replay-rows",
                "note": (
                    "Static GPUReplica budget counterfactual keeps each replay "
                    "row's base latency and replaces selected recovery overhead "
                    "with peer-visible GPUReplica protection under the given "
                    "budget; long-prefix suffixes use Host restore, hot-tail and "
                    "fan-out suffixes recompute"
                ),
                "runtime_replay_rows": len(rows),
                "runtime_recovery_tokens": int(recovery_tokens),
                "runtime_peer_visible_tokens": int(peer_visible_tokens),
                "runtime_peer_visible_token_fraction": (
                    peer_visible_tokens / recovery_tokens if recovery_tokens > 0 else 0.0
                ),
                "runtime_mean_static_restore_ms": mean(static_restore_values),
                **model,
            }
        )
    return out


def dynamic_aegis_row(
    *,
    model: dict[str, float],
    regimes: dict[str, list[dict[str, str]]],
    base_meta: dict[str, object],
) -> dict[str, object]:
    regime_summaries: list[dict[str, object]] = []
    for label, rows in regimes.items():
        ratios: list[float] = []
        selected_values: list[float] = []
        recovery_tokens = 0.0
        for row in rows:
            selected = selected_overhead_ms(row)
            ratio = normalized_wct_for_overhead(model, row, selected)
            if ratio is not None:
                ratios.append(ratio)
                selected_values.append(selected)
                recovery_tokens += row_pre_failure_tokens(row)
        summary = summarize(ratios)
        regime_summaries.append(
            {
                "label": label,
                "replay_rows": len(ratios),
                "recovery_tokens": int(recovery_tokens),
                "normalized_wct": summary["normalized_wct"],
                "mean_selected_restore_ms": mean(selected_values),
            }
        )

    values = [float(row["normalized_wct"]) for row in regime_summaries]
    return {
        "gpu_replica_budget_fraction": "dynamic",
        "workload": "AegisKV",
        "normalized_wct": mean(values),
        "normalized_wct_std": metric_std(values),
        "normalized_wct_ci95": metric_ci95(values),
        **base_meta,
        "repeat": f"mean-of-{len(regime_summaries)}-regimes",
        "note": (
            "Dynamic AegisKV row is the equal-regime mean of observed hard-failure "
            "replay e2e divided by the same-row no-backup recompute counterfactual"
        ),
        "runtime_replay_rows": sum(int(row["replay_rows"]) for row in regime_summaries),
        "runtime_recovery_tokens": sum(
            int(row["recovery_tokens"]) for row in regime_summaries
        ),
        "runtime_regime_count": len(regime_summaries),
        "runtime_regimes": ";".join(str(row["label"]) for row in regime_summaries),
        "runtime_regime_summaries": json.dumps(regime_summaries, sort_keys=True),
        **model,
    }


def parse_budget_points(value: str) -> list[float]:
    points = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not points:
        raise argparse.ArgumentTypeError("budget points must not be empty")
    if any(point < 0 or point > 1 for point in points):
        raise argparse.ArgumentTypeError("budget points must be in [0, 1]")
    return sorted(dict.fromkeys(points))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 3 static GPUReplica budget-gap curves from direct "
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
        raise ValueError(
            "host, gpu, and recompute replay rows are all required for Fig. 3"
        )

    model = fit_models(
        host_rows=host_rows,
        gpu_rows=gpu_rows,
        recompute_rows=recompute_rows,
    )
    run_dirs = [
        *[str(path) for path in args.host_run_dir],
        *[str(path) for path in args.gpu_run_dir],
        *[str(path) for path in args.recompute_run_dir],
    ]
    base_meta = {
        "evidence_tier": "sglsim-runtime-replay-calibrated",
        "source": "sglsim_request_details_replay_static_gpu_budget_counterfactual",
        "run_id": args.run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": "aegiskv-sglsim-static-gpu-budget-gap",
        "dp_size": 2,
        "tp_size": 1,
        "failure_rank": 0,
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(run_dirs),
    }

    regimes = {
        "long prefixes": host_rows,
        "hot tails": gpu_rows,
        "fan-out skew": recompute_rows,
    }
    rows: list[dict[str, object]] = []
    rows.extend(
        budget_rows_for_workload(
            model=model,
            rows=gpu_rows,
            workload="hot tails",
            budgets=args.budget_points,
            base_meta=base_meta,
        )
    )
    rows.extend(
        budget_rows_for_workload(
            model=model,
            rows=host_rows,
            workload="long prefixes",
            budgets=args.budget_points,
            base_meta=base_meta,
        )
    )
    rows.extend(
        budget_rows_for_workload(
            model=model,
            rows=recompute_rows,
            workload="fan-out skew",
            budgets=args.budget_points,
            base_meta=base_meta,
        )
    )
    rows.append(dynamic_aegis_row(model=model, regimes=regimes, base_meta=base_meta))

    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "runtime_replay_rows": len(host_rows) + len(gpu_rows) + len(recompute_rows),
                "model": model,
                "dynamic_aegis_normalized_wct": rows[-1]["normalized_wct"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
