#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
DEFAULT_FIG3_CSV = (
    ROOT
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_static_gpu_replica_budget_gap.csv"
)
DEFAULT_FIG13_CSV = (
    ROOT
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_budget_sensitivity.csv"
)

FIG3_SOURCE = "sglsim_agentic_online_budget_sweep"
FIG3_EVIDENCE_TIER = "sglsim-agentic-online-budget-sweep"
FIG13_SOURCE = "sglsim_agentic_online_budget_timing_sweep"
FIG13_EVIDENCE_TIER = "sglsim-agentic-online-budget-timing-sweep"

FIG3_WORKLOAD_PREFIXES = {
    "budget_hot_tail": "hot tails",
    "budget_long_prefix": "long prefixes",
    "budget_nonpeer": "fan-out skew",
}
FIG13_TIMING_PREFIXES = {
    "budget_timing_early": "early decode",
    "budget_hot_tail": "mid decode",
    "budget_timing_late": "late decode",
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> int:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
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
    return len(rows)


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _int(row: dict[str, str], key: str) -> int:
    return int(round(_float(row, key)))


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
        _ci95(values),
    )


def _budget_label(row: dict[str, str]) -> str:
    budget = _float(row, "aegis_gpu_budget_fraction")
    return f"{budget:.3f}"


def _scenario_prefix(scenario: str, prefixes: dict[str, str]) -> str | None:
    for prefix in prefixes:
        if scenario.startswith(prefix + "_"):
            return prefix
    return None


def _group_rows(
    rows: list[dict[str, str]],
    prefixes: dict[str, str],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scenario = str(row.get("scenario") or "")
        prefix = _scenario_prefix(scenario, prefixes)
        if prefix is None:
            continue
        grouped[(prefix, _budget_label(row))].append(row)
    return grouped


def _validate_group(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    prefixes: dict[str, str],
) -> None:
    budgets = sorted({budget for _, budget in grouped})
    if not budgets:
        raise ValueError("budget sweep rows are missing")
    for prefix in prefixes:
        missing = [budget for budget in budgets if not grouped.get((prefix, budget))]
        if missing:
            raise ValueError(f"{prefix} missing budget rows: {', '.join(missing)}")
        seed_sets = {
            budget: tuple(
                sorted(str(row.get("seed", "")).strip() for row in grouped[(prefix, budget)])
            )
            for budget in budgets
        }
        if any(not all(seed for seed in seeds) for seeds in seed_sets.values()):
            raise ValueError(f"{prefix} lacks seed provenance")
        if len(set(seed_sets.values())) != 1:
            raise ValueError(f"{prefix} seed sets differ by budget: {seed_sets}")
        for budget in budgets:
            for row in grouped[(prefix, budget)]:
                if _int(row, "failed") != 0:
                    raise ValueError(
                        f"{row.get('scenario')} seed={row.get('seed')} has failed rows"
                    )
                if _int(row, "replay_rows") <= 0:
                    raise ValueError(
                        f"{row.get('scenario')} seed={row.get('seed')} has no replay rows"
                    )


def _budget0_by_seed(rows: list[dict[str, str]]) -> dict[str, float]:
    baseline: dict[str, float] = {}
    for row in rows:
        seed = str(row.get("seed") or "")
        value = _float(row, "mean_e2e_ms")
        if not seed or value <= 0.0:
            raise ValueError("budget=0 rows require seed and positive mean_e2e_ms")
        baseline[seed] = value
    return baseline


def _normalized_values(
    rows: list[dict[str, str]],
    baseline_by_seed: dict[str, float],
) -> list[float]:
    values: list[float] = []
    for row in rows:
        seed = str(row.get("seed") or "")
        baseline = baseline_by_seed.get(seed, 0.0)
        value = _float(row, "mean_e2e_ms")
        if baseline <= 0.0 or value <= 0.0:
            raise ValueError(f"missing paired baseline for seed={seed}")
        values.append(value / baseline)
    return values


def _common_meta(
    rows: list[dict[str, str]],
    *,
    run_id: str,
    source: str,
    evidence_tier: str,
    note: str,
) -> dict[str, object]:
    first = rows[0]
    return {
        "evidence_tier": evidence_tier,
        "source": source,
        "run_id": run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": first.get("scenario", ""),
        "dp_size": first.get("dp_size", ""),
        "tp_size": 1,
        "failure_rank": 0,
        "repeat": f"mean-of-{len(rows)}-agentic-trials",
        "repeat_count": len(rows),
        "repeat_seeds": ";".join(row.get("seed", "") for row in rows),
        "repeat_run_ids": ";".join(row.get("source_run_dir", "") for row in rows),
        "runtime_replay_rows": sum(_int(row, "replay_rows") for row in rows),
        "runtime_failed_rows": sum(_int(row, "failed") for row in rows),
        "runtime_host_lookup_hits": sum(_int(row, "host_lookup_hits") for row in rows),
        "runtime_peer_lookup_hits": sum(_int(row, "peer_lookup_hits") for row in rows),
        "runtime_host_restore_actions": sum(
            _int(row, "replay_action_host_restore") for row in rows
        ),
        "runtime_peer_gpu_actions": sum(
            _int(row, "replay_action_peer_gpu_restore") for row in rows
        ),
        "runtime_recompute_actions": sum(
            _int(row, "replay_action_recompute") for row in rows
        ),
        "runtime_restored_tokens": sum(_int(row, "restored_tokens") for row in rows),
        "runtime_recompute_tokens": sum(_int(row, "recompute_tokens") for row in rows),
        "note": note,
    }


def derive_fig3_rows(rows: list[dict[str, str]], *, run_id: str) -> list[dict[str, object]]:
    grouped = _group_rows(rows, FIG3_WORKLOAD_PREFIXES)
    _validate_group(grouped, FIG3_WORKLOAD_PREFIXES)
    budgets = sorted({budget for _, budget in grouped}, key=float)
    if f"{0.0:.3f}" not in budgets:
        raise ValueError("Fig. 3 requires a budget=0 baseline")

    out: list[dict[str, object]] = []
    dynamic_values_by_seed: dict[str, list[float]] = defaultdict(list)
    for prefix, workload in FIG3_WORKLOAD_PREFIXES.items():
        baseline = _budget0_by_seed(grouped[(prefix, f"{0.0:.3f}")])
        for budget in budgets:
            scenario_rows = grouped[(prefix, budget)]
            normalized = _normalized_values(scenario_rows, baseline)
            mean_value, std_value, ci_value = _stats(normalized)
            for row, value in zip(scenario_rows, normalized, strict=True):
                dynamic_values_by_seed[str(row.get("seed") or "")].append(value)
            out.append(
                {
                    "gpu_replica_budget_fraction": float(budget),
                    "workload": workload,
                    "normalized_wct": mean_value,
                    "normalized_wct_std": std_value,
                    "normalized_wct_ci95": ci_value,
                    **_common_meta(
                        scenario_rows,
                        run_id=run_id,
                        source=FIG3_SOURCE,
                        evidence_tier=FIG3_EVIDENCE_TIER,
                        note=(
                            "Online SGLSim hard-restart GPUReplica budget sweep. "
                            "Rows are normalized to the same workload's budget=0 "
                            "paired seed."
                        ),
                    ),
                }
            )

    best_by_seed: list[float] = []
    for seed, values in sorted(dynamic_values_by_seed.items()):
        if not values:
            continue
        best_by_seed.append(min(values))
    mean_value, std_value, ci_value = _stats(best_by_seed)
    all_rows = [row for rows_for_key in grouped.values() for row in rows_for_key]
    out.append(
        {
            "gpu_replica_budget_fraction": "dynamic",
            "workload": "AegisKV",
            "normalized_wct": mean_value,
            "normalized_wct_std": std_value,
            "normalized_wct_ci95": ci_value,
            **_common_meta(
                all_rows,
                run_id=run_id,
                source=FIG3_SOURCE,
                evidence_tier=FIG3_EVIDENCE_TIER,
                note=(
                    "Dynamic AegisKV row is the per-seed best observed online "
                    "budget choice across the controlled workload sweep."
                ),
            ),
        }
    )
    return out


def derive_fig13_rows(rows: list[dict[str, str]], *, run_id: str) -> list[dict[str, object]]:
    grouped = _group_rows(rows, FIG13_TIMING_PREFIXES)
    _validate_group(grouped, FIG13_TIMING_PREFIXES)
    budgets = sorted({budget for _, budget in grouped}, key=float)
    if f"{0.0:.3f}" not in budgets:
        raise ValueError("Fig. 13 requires a budget=0 baseline")

    out: list[dict[str, object]] = []
    for prefix, timing in FIG13_TIMING_PREFIXES.items():
        baseline = _budget0_by_seed(grouped[(prefix, f"{0.0:.3f}")])
        for budget in budgets:
            scenario_rows = grouped[(prefix, budget)]
            normalized = _normalized_values(scenario_rows, baseline)
            mean_value, std_value, ci_value = _stats(normalized)
            out.append(
                {
                    "failure_timing": timing,
                    "gpu_replica_budget_fraction": float(budget),
                    "normalized_wct": mean_value,
                    "normalized_wct_std": std_value,
                    "normalized_wct_ci95": ci_value,
                    **_common_meta(
                        scenario_rows,
                        run_id=run_id,
                        source=FIG13_SOURCE,
                        evidence_tier=FIG13_EVIDENCE_TIER,
                        note=(
                            "Online SGLSim controlled failure-timing and "
                            "GPUReplica budget sweep, normalized to each "
                            "timing's budget=0 paired seed."
                        ),
                    ),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 3 and Fig. 13 from online SGLSim agentic GPUReplica "
            "budget sweeps."
        )
    )
    parser.add_argument("--paper-candidates-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fig3-output-csv", type=Path, default=DEFAULT_FIG3_CSV)
    parser.add_argument("--fig13-output-csv", type=Path, default=DEFAULT_FIG13_CSV)
    args = parser.parse_args()

    rows = read_csv(args.paper_candidates_csv)
    fig3_rows = derive_fig3_rows(rows, run_id=args.run_id)
    fig13_rows = derive_fig13_rows(rows, run_id=args.run_id)
    fig3_count = write_csv(args.fig3_output_csv, fig3_rows)
    fig13_count = write_csv(args.fig13_output_csv, fig13_rows)
    print(
        {
            "run_id": args.run_id,
            "fig3_output_csv": str(args.fig3_output_csv),
            "fig3_rows": fig3_count,
            "fig13_output_csv": str(args.fig13_output_csv),
            "fig13_rows": fig13_count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
