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
DEFAULT_OUTPUT = (
    ROOT
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_ablation.csv"
)

SOURCE = "sglsim_agentic_online_ablation_sweep"
EVIDENCE_TIER = "sglsim-agentic-online-ablation-sweep"

SCENARIO_TO_VARIANT = {
    "ablation_full_aegiskv": "AegisKV",
    "ablation_no_lifecycle": "no lifecycle",
    "ablation_no_segment_granularity": "no segment granularity",
    "ablation_no_restore_recompute": "no restore/recompute",
    "ablation_no_pressure": "no pressure",
    "ablation_no_hysteresis": "no hysteresis",
    "ablation_no_critical_restore": "no critical restore",
}

COMPARABLE_WORKLOAD_COLUMNS = (
    "dp_size",
    "workflow_count",
    "turns",
    "groups",
    "prompts_per_group",
    "system_prompt_len",
    "question_len",
    "output_len",
    "range_ratio",
    "max_concurrency",
    "critical_agent_ratio",
    "failover_replay",
    "failover_injection",
    "aegis_policy",
    "backup_kv",
)


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


def grouped_by_scenario(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scenario = str(row.get("scenario", ""))
        if scenario in SCENARIO_TO_VARIANT:
            out[scenario].append(row)
    return out


def _workload_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        str(row.get(column, "")).strip()
        for column in COMPARABLE_WORKLOAD_COLUMNS
    )


def _validate_comparable_workload(by_scenario: dict[str, list[dict[str, str]]]) -> None:
    signatures: dict[str, tuple[str, ...]] = {}
    seed_sets: dict[str, list[str]] = {}
    for scenario in SCENARIO_TO_VARIANT:
        rows = by_scenario.get(scenario, [])
        for row in rows:
            signature = _workload_signature(row)
            if not any(signature):
                raise ValueError(
                    "ablation rows lack workload signature columns; rerun the "
                    "agentic suite after refreshing paper-candidate schema"
                )
            previous = signatures.get(scenario)
            if previous is not None and previous != signature:
                raise ValueError(f"{scenario} has inconsistent workload signatures")
            signatures[scenario] = signature
        seed_sets[scenario] = sorted(str(row.get("seed", "")).strip() for row in rows)

    unique = {signature for signature in signatures.values()}
    if len(unique) != 1:
        details = {
            scenario: dict(zip(COMPARABLE_WORKLOAD_COLUMNS, signature, strict=True))
            for scenario, signature in signatures.items()
        }
        raise ValueError(f"ablation scenarios are not workload-comparable: {details}")

    if any(not all(seed for seed in seeds) for seeds in seed_sets.values()):
        raise ValueError("ablation rows lack seed columns")
    unique_seed_sets = {tuple(seeds) for seeds in seed_sets.values()}
    if len(unique_seed_sets) != 1:
        raise ValueError(f"ablation scenarios are not seed-comparable: {seed_sets}")


def derive_rows(rows: list[dict[str, str]], *, run_id: str) -> list[dict[str, object]]:
    by_scenario = grouped_by_scenario(rows)
    missing = sorted(
        scenario for scenario in SCENARIO_TO_VARIANT if not by_scenario.get(scenario)
    )
    if missing:
        raise ValueError("missing ablation scenario rows: " + ", ".join(missing))
    _validate_comparable_workload(by_scenario)

    baseline_values = [
        _float(row, "mean_e2e_ms")
        for row in by_scenario["ablation_full_aegiskv"]
        if _float(row, "mean_e2e_ms") > 0
    ]
    baseline_mean = statistics.mean(baseline_values)
    if baseline_mean <= 0:
        raise ValueError("ablation_full_aegiskv mean_e2e_ms must be positive")

    current_sha = git_sha()
    out: list[dict[str, object]] = []
    for scenario, variant in SCENARIO_TO_VARIANT.items():
        scenario_rows = by_scenario[scenario]
        normalized = [
            _float(row, "mean_e2e_ms") / baseline_mean
            for row in scenario_rows
            if _float(row, "mean_e2e_ms") > 0
        ]
        mean_value, std_value, ci_value = _stats(normalized)
        mean_e2e, mean_e2e_std, mean_e2e_ci = _stats(
            [_float(row, "mean_e2e_ms") for row in scenario_rows]
        )
        first = scenario_rows[0]
        replay_rows = sum(_int(row, "replay_rows") for row in scenario_rows)
        failed = sum(_int(row, "failed") for row in scenario_rows)
        host_actions = sum(
            _int(row, "replay_action_host_restore") for row in scenario_rows
        )
        peer_actions = sum(
            _int(row, "replay_action_peer_gpu_restore") for row in scenario_rows
        )
        recompute_actions = sum(
            _int(row, "replay_action_recompute") for row in scenario_rows
        )
        out.append(
            {
                "variant": variant,
                "normalized_wct": mean_value,
                "normalized_wct_std": std_value,
                "normalized_wct_ci95": ci_value,
                "evidence_tier": EVIDENCE_TIER,
                "source": SOURCE,
                "run_id": run_id,
                "git_sha": current_sha,
                "remote_host": "",
                "slurm_job_id": "",
                "model_path": "mock-model",
                "workload_id": scenario,
                "dp_size": first.get("dp_size", ""),
                "workflow_count": first.get("workflow_count", ""),
                "turns": first.get("turns", ""),
                "system_prompt_len": first.get("system_prompt_len", ""),
                "question_len": first.get("question_len", ""),
                "output_len": first.get("output_len", ""),
                "critical_agent_ratio": first.get("critical_agent_ratio", ""),
                "tp_size": 1,
                "failure_rank": 0,
                "aegis_policy": first.get("aegis_policy", ""),
                "aegis_ablation": first.get("aegis_ablation", ""),
                "retry_route_policy": first.get("failover_retry_policy", ""),
                "repeat": f"mean-of-{len(scenario_rows)}-agentic-trials",
                "repeat_count": len(scenario_rows),
                "repeat_seeds": ";".join(row.get("seed", "") for row in scenario_rows),
                "repeat_run_ids": ";".join(
                    row.get("source_run_dir", "") for row in scenario_rows
                ),
                "runtime_replay_rows": replay_rows,
                "runtime_failed_rows": failed,
                "runtime_host_restore_actions": host_actions,
                "runtime_peer_gpu_actions": peer_actions,
                "runtime_recompute_actions": recompute_actions,
                "mean_e2e_latency_ms": mean_e2e,
                "mean_e2e_latency_ms_std": mean_e2e_std,
                "mean_e2e_latency_ms_ci95": mean_e2e_ci,
                "note": (
                    "Online SGLSim ablation sweep with explicit "
                    "gsp_aegis_ablation toggles on the recovery path; rows are "
                    "metadata-level and physical-copy calibrated only when "
                    "paired with the recovery-action microbench."
                ),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project agentic workflow ablation sweep rows into Fig. 12."
    )
    parser.add_argument("--paper-candidates-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = derive_rows(read_csv(args.paper_candidates_csv), run_id=args.run_id)
    write_csv(args.output_csv, rows)
    print(f"wrote {len(rows)} online ablation rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
