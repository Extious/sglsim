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
    / "fig_peer_rank_selection.csv"
)

SOURCE = "sglsim_agentic_online_peer_route_sweep"
EVIDENCE_TIER = "sglsim-agentic-online-peer-route-sweep"

SCENARIO_TO_POLICY = {
    "peer_route_nonpeer_hot_tail": "non-peer route miss",
    "peer_route_random_hot_tail": "random healthy route",
    "peer_route_aegiskv_hot_tail": "AegisKV peer route",
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


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def grouped_by_scenario(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scenario = str(row.get("scenario", ""))
        if scenario in SCENARIO_TO_POLICY:
            out[scenario].append(row)
    return out


def _signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip() for column in COMPARABLE_WORKLOAD_COLUMNS)


def _validate_comparable_rows(by_scenario: dict[str, list[dict[str, str]]]) -> None:
    missing = sorted(
        scenario for scenario in SCENARIO_TO_POLICY if not by_scenario.get(scenario)
    )
    if missing:
        raise ValueError("missing peer-route scenario rows: " + ", ".join(missing))

    signatures: dict[str, tuple[str, ...]] = {}
    seed_sets: dict[str, tuple[str, ...]] = {}
    for scenario, rows in by_scenario.items():
        scenario_sigs = {_signature(row) for row in rows}
        if len(scenario_sigs) != 1 or not any(next(iter(scenario_sigs))):
            raise ValueError(f"{scenario} lacks a stable workload signature")
        signatures[scenario] = next(iter(scenario_sigs))
        seeds = tuple(sorted(str(row.get("seed", "")).strip() for row in rows))
        if not all(seeds):
            raise ValueError(f"{scenario} lacks seed provenance")
        seed_sets[scenario] = seeds
    if len(set(signatures.values())) != 1:
        raise ValueError(f"peer-route scenarios are not workload-comparable: {signatures}")
    if len(set(seed_sets.values())) != 1:
        raise ValueError(f"peer-route scenarios are not seed-comparable: {seed_sets}")


def _restore_overhead_ms(row: dict[str, str]) -> float:
    value = _float(row, "aegis_replay_restore_overhead_s") * 1000.0
    if value <= 0.0:
        value = _float(row, "aegis_replay_restore_sleep_s") * 1000.0
    return value


def _replay_rows_from_run(run_dir: str) -> list[dict[str, str]]:
    path = Path(run_dir) / "request_details.csv"
    rows = []
    for row in read_csv(path):
        if _bool(row.get("aegis_failover_replay")):
            rows.append(row)
    return rows


def _runtime_replay_overheads(rows: list[dict[str, str]]) -> list[float]:
    out = [_restore_overhead_ms(row) for row in rows]
    return [value for value in out if value > 0.0]


def derive_rows(rows: list[dict[str, str]], *, run_id: str) -> list[dict[str, object]]:
    by_scenario = grouped_by_scenario(rows)
    _validate_comparable_rows(by_scenario)

    baseline_values = [
        _float(row, "mean_e2e_ms") for row in by_scenario["peer_route_nonpeer_hot_tail"]
    ]
    baseline_mean = statistics.mean(value for value in baseline_values if value > 0.0)
    if baseline_mean <= 0.0:
        raise ValueError("peer_route_nonpeer_hot_tail mean_e2e_ms must be positive")

    runtime_rows_by_scenario: dict[str, list[dict[str, str]]] = {}
    overheads_by_scenario: dict[str, list[float]] = {}
    for scenario, scenario_rows in by_scenario.items():
        replay_rows: list[dict[str, str]] = []
        for row in scenario_rows:
            replay_rows.extend(_replay_rows_from_run(str(row.get("source_run_dir") or "")))
        if not replay_rows:
            raise ValueError(f"{scenario} has no replay rows")
        runtime_rows_by_scenario[scenario] = replay_rows
        overheads_by_scenario[scenario] = _runtime_replay_overheads(replay_rows)

    baseline_p99 = _percentile(
        overheads_by_scenario["peer_route_nonpeer_hot_tail"], 0.99
    )
    if baseline_p99 <= 0.0:
        raise ValueError("non-peer baseline p99 restore overhead must be positive")

    current_sha = git_sha()
    out: list[dict[str, object]] = []
    for scenario, policy in SCENARIO_TO_POLICY.items():
        scenario_rows = by_scenario[scenario]
        normalized = [
            _float(row, "mean_e2e_ms") / baseline_mean
            for row in scenario_rows
            if _float(row, "mean_e2e_ms") > 0.0
        ]
        replay_rows = runtime_rows_by_scenario[scenario]
        overheads = overheads_by_scenario[scenario]
        peer_hits = sum(1 for row in replay_rows if _bool(row.get("aegis_peer_gpu_lookup_hit")))
        host_hits = sum(1 for row in replay_rows if _bool(row.get("aegis_host_backup_lookup_hit")))
        recompute_actions = sum(
            1
            for row in replay_rows
            if str(row.get("aegis_replay_restore_action") or "") == "recompute"
        )
        gpu_actions = sum(
            1
            for row in replay_rows
            if str(row.get("aegis_replay_restore_action") or "") == "gpu_replica"
        )
        host_actions = sum(
            1
            for row in replay_rows
            if str(row.get("aegis_replay_restore_action") or "") == "host_restore"
        )
        first = scenario_rows[0]
        out.append(
            {
                "policy": policy,
                "normalized_wct": statistics.mean(normalized),
                "normalized_wct_std": statistics.stdev(normalized)
                if len(normalized) > 1
                else 0.0,
                "normalized_wct_ci95": _ci95(normalized),
                "peer_p99_overhead_pct": 100.0 * _percentile(overheads, 0.99) / baseline_p99,
                "peer_p99_overhead_pct_std": statistics.stdev(
                    [100.0 * value / baseline_p99 for value in overheads]
                )
                if len(overheads) > 1
                else 0.0,
                "peer_p99_overhead_pct_ci95": _ci95(
                    [100.0 * value / baseline_p99 for value in overheads]
                ),
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
                "repeat": f"mean-of-{len(scenario_rows)}-agentic-trials",
                "repeat_count": len(scenario_rows),
                "repeat_seeds": ";".join(row.get("seed", "") for row in scenario_rows),
                "repeat_run_ids": ";".join(
                    row.get("source_run_dir", "") for row in scenario_rows
                ),
                "runtime_replay_rows": len(replay_rows),
                "runtime_peer_lookup_hits": peer_hits,
                "runtime_route_hit_rate": peer_hits / len(replay_rows),
                "runtime_route_miss_rows": len(replay_rows) - peer_hits,
                "runtime_host_lookup_hits": host_hits,
                "runtime_action_gpu_replica": gpu_actions,
                "runtime_action_host_restore": host_actions,
                "runtime_action_recompute": recompute_actions,
                "runtime_mean_restore_overhead_ms": statistics.mean(overheads),
                "runtime_p99_restore_overhead_ms": _percentile(overheads, 0.99),
                "runtime_baseline_p99_recompute_overhead_ms": baseline_p99,
                "runtime_failed_rows": sum(_int(row, "failed") for row in scenario_rows),
                "note": (
                    "Online SGLSim peer-route sweep with implemented retry route "
                    "policies. Rows remain metadata-level with calibrated timing."
                ),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project agentic peer-route online suite rows into Fig. 14b."
    )
    parser.add_argument("--paper-candidates-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = derive_rows(read_csv(args.paper_candidates_csv), run_id=args.run_id)
    write_csv(args.output_csv, rows)
    print(f"wrote {len(rows)} online peer-route rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
