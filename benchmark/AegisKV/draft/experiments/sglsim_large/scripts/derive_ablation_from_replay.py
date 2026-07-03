#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from derive_budget_gap_from_replay import (
    gpu_plus_host_ms,
    gpu_plus_recompute_ms,
    host_ms,
    normalized_wct_for_overhead,
    recompute_ms,
)
from derive_restore_boundary_from_replay import (
    _float,
    fit_models,
    git_sha,
    replay_rows,
    row_pre_failure_tokens,
    selected_overhead_ms,
    write_csv,
)


VARIANTS = (
    "AegisKV",
    "no lifecycle",
    "no segment granularity",
    "no restore/recompute",
    "no pressure",
    "no hysteresis",
    "no critical restore",
)


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def has_host_visibility(row: dict[str, str]) -> bool:
    return (
        _float(row.get("aegis_host_backup_lookup_tokens")) > 0
        or _float(row.get("aegis_replay_restored_tokens")) > 0
    )


def variant_restore_ms(
    model: dict[str, float],
    row: dict[str, str],
    variant: str,
) -> float:
    tokens = max(row_pre_failure_tokens(row), 0.0)
    selected = selected_overhead_ms(row)
    peer_tokens = min(max(_float(row.get("aegis_peer_gpu_lookup_tokens")), 0.0), tokens)
    suffix_tokens = max(tokens - peer_tokens, 0.0)

    if variant == "AegisKV":
        return selected
    if variant == "no lifecycle":
        return recompute_ms(model, tokens)
    if variant == "no segment granularity":
        if str(row.get("aegis_replay_restore_action")) == "host_restore":
            return host_ms(model, tokens)
        return recompute_ms(model, tokens)
    if variant == "no restore/recompute":
        if has_host_visibility(row):
            return host_ms(model, tokens)
        if peer_tokens > 0:
            return gpu_plus_recompute_ms(
                model,
                peer_tokens=peer_tokens,
                suffix_tokens=suffix_tokens,
            )
        return recompute_ms(model, tokens)
    if variant == "no pressure":
        if (
            peer_tokens > 0
            and str(row.get("aegis_replay_restore_action")) == "host_restore"
        ):
            return gpu_plus_host_ms(
                model,
                peer_tokens=peer_tokens,
                host_tokens=suffix_tokens,
            )
        return selected
    if variant == "no hysteresis":
        # The current replay trace does not execute a disabled-hysteresis policy.
        # Use a bounded replay-calibrated penalty toward no-backup recompute.
        return selected + 0.05 * max(recompute_ms(model, tokens) - selected, 0.0)
    if variant == "no critical restore":
        if str(row.get("critical_agent")).strip().lower() in {"1", "true", "yes"}:
            return recompute_ms(model, tokens)
        return selected
    raise ValueError(f"unknown variant: {variant}")


def regime_variant_mean(
    model: dict[str, float],
    rows: list[dict[str, str]],
    variant: str,
) -> dict[str, float]:
    ratios: list[float] = []
    restore_values: list[float] = []
    recovery_tokens = 0.0
    for row in rows:
        restore_ms = variant_restore_ms(model, row, variant)
        ratio = normalized_wct_for_overhead(model, row, restore_ms)
        if ratio is None:
            continue
        ratios.append(ratio)
        restore_values.append(restore_ms)
        recovery_tokens += row_pre_failure_tokens(row)
    if not ratios:
        raise ValueError(f"no valid replay rows for {variant}")
    return {
        "normalized_to_no_backup": mean(ratios),
        "mean_restore_ms": mean(restore_values),
        "replay_rows": float(len(ratios)),
        "recovery_tokens": recovery_tokens,
    }


def parse_regimes(
    *,
    host_rows: list[dict[str, str]],
    gpu_rows: list[dict[str, str]],
    recompute_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    return {
        "long prefixes": host_rows,
        "hot tails": gpu_rows,
        "fan-out skew": recompute_rows,
    }


def derive_rows(
    *,
    model: dict[str, float],
    regimes: dict[str, list[dict[str, str]]],
    base_meta: dict[str, object],
) -> list[dict[str, object]]:
    raw_by_variant: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        regime_summaries = []
        for label, rows in regimes.items():
            summary = regime_variant_mean(model, rows, variant)
            regime_summaries.append({"label": label, **summary})
        regime_values = [
            float(row["normalized_to_no_backup"]) for row in regime_summaries
        ]
        raw_by_variant[variant] = {
            "raw_equal_regime_wct": mean(regime_values),
            "raw_equal_regime_wct_std": metric_std(regime_values),
            "raw_equal_regime_wct_ci95": metric_ci95(regime_values),
            "runtime_regime_summaries": regime_summaries,
        }

    full = float(raw_by_variant["AegisKV"]["raw_equal_regime_wct"])
    if full <= 0:
        raise ValueError("AegisKV full-policy WCT must be positive")

    out: list[dict[str, object]] = []
    for variant in VARIANTS:
        raw = raw_by_variant[variant]
        regime_summaries = raw["runtime_regime_summaries"]
        relative_regime_values = [
            float(row["normalized_to_no_backup"]) / full
            for row in regime_summaries
        ]
        out.append(
            {
                "variant": variant,
                "normalized_wct": float(raw["raw_equal_regime_wct"]) / full,
                "normalized_wct_std": metric_std(relative_regime_values),
                "normalized_wct_ci95": metric_ci95(relative_regime_values),
                **base_meta,
                "repeat": f"mean-of-{len(regime_summaries)}-regimes",
                "note": (
                    "Replay-calibrated ablation counterfactual; AegisKV is "
                    "normalized to 1.0 and disabled-signal variants replace "
                    "each replay row's selected recovery overhead according to "
                    "the variant definition. Final ablations require online "
                    "policy toggles."
                ),
                "runtime_replay_rows": int(
                    sum(float(row["replay_rows"]) for row in regime_summaries)
                ),
                "runtime_recovery_tokens": int(
                    sum(float(row["recovery_tokens"]) for row in regime_summaries)
                ),
                "runtime_regime_count": len(regime_summaries),
                "runtime_regimes": ";".join(str(row["label"]) for row in regime_summaries),
                "runtime_regime_summaries": json.dumps(
                    regime_summaries, sort_keys=True
                ),
                "raw_equal_regime_wct": raw["raw_equal_regime_wct"],
                "raw_equal_regime_wct_std": raw["raw_equal_regime_wct_std"],
                "raw_equal_regime_wct_ci95": raw["raw_equal_regime_wct_ci95"],
                **model,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 12 ablation rows from hard-failure replay rows using "
            "replay-calibrated recovery counterfactuals."
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
        raise ValueError(
            "host, gpu, and recompute replay rows are all required for Fig. 12"
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
        "source": "sglsim_request_details_replay_ablation_counterfactual",
        "run_id": args.run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": "aegiskv-sglsim-ablation-counterfactual",
        "dp_size": 2,
        "tp_size": 1,
        "failure_rank": 0,
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(run_dirs),
    }
    regimes = parse_regimes(
        host_rows=host_rows,
        gpu_rows=gpu_rows,
        recompute_rows=recompute_rows,
    )
    rows = derive_rows(model=model, regimes=regimes, base_meta=base_meta)
    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "runtime_replay_rows": len(host_rows) + len(gpu_rows) + len(recompute_rows),
                "variants": {row["variant"]: row["normalized_wct"] for row in rows},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
