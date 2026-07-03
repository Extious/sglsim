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
DEFAULT_FIG9_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_recovery_latency.csv"
)

SOURCE = "sglsim_agentic_online_recovery_latency_sweep"
EVIDENCE_TIER = "sglsim-agentic-online-recovery-latency-sweep"

SCENARIO_TO_POLICY = {
    "policy_no_backup": "no backup",
    "policy_host_only": "uniform Host",
    "policy_gpu_replica": "uniform GPU",
    "policy_aegiskv": "AegisKV",
    "staged_13131_policy_no_backup": "no backup",
    "staged_13131_policy_host_only": "uniform Host",
    "staged_13131_policy_gpu_replica": "uniform GPU",
    "staged_13131_policy_aegiskv": "AegisKV",
    "staged_13131_policy_lite_no_backup": "no backup",
    "staged_13131_policy_lite_host_only": "uniform Host",
    "staged_13131_policy_lite_gpu_replica": "uniform GPU",
    "staged_13131_policy_lite_aegiskv": "AegisKV",
    "staged_13131_policy_stress_no_backup": "no backup",
    "staged_13131_policy_stress_host_only": "uniform Host",
    "staged_13131_policy_stress_gpu_replica": "uniform GPU",
    "staged_13131_policy_stress_aegiskv": "AegisKV",
}
REQUIRED_POLICIES = ("no backup", "uniform Host", "uniform GPU", "AegisKV")

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
    "failover_retry_rank_offset",
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


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in ("", None) else 0


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


def grouped_by_policy(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scenario = str(row.get("scenario", ""))
        policy = SCENARIO_TO_POLICY.get(scenario)
        if policy:
            out[policy].append(row)
    return out


def _workload_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip() for column in COMPARABLE_WORKLOAD_COLUMNS)


def _validate_comparable_workload(by_policy: dict[str, list[dict[str, str]]]) -> None:
    signatures: dict[str, tuple[str, ...]] = {}
    for policy in REQUIRED_POLICIES:
        scenario_rows = by_policy.get(policy, [])
        for row in scenario_rows:
            signature = _workload_signature(row)
            if not any(signature):
                raise ValueError(
                    "policy sweep rows lack workload signature columns; "
                    "rerun the agentic suite after refreshing paper-candidate schema"
                )
            previous = signatures.get(policy)
            if previous is not None and previous != signature:
                raise ValueError(f"{policy} has inconsistent workload signatures")
            signatures[policy] = signature
    unique = {signature for signature in signatures.values()}
    if len(unique) != 1:
        details = {
            policy: dict(zip(COMPARABLE_WORKLOAD_COLUMNS, signature, strict=True))
            for policy, signature in signatures.items()
        }
        raise ValueError(f"policy rows are not workload-comparable: {details}")

    seed_sets = {
        policy: sorted(str(row.get("seed", "")).strip() for row in scenario_rows)
        for policy, scenario_rows in by_policy.items()
    }
    if any(not all(seed for seed in seeds) for seeds in seed_sets.values()):
        raise ValueError("policy sweep rows lack seed columns")
    unique_seed_sets = {tuple(seeds) for seeds in seed_sets.values()}
    if len(unique_seed_sets) != 1:
        raise ValueError(f"policy rows are not seed-comparable: {seed_sets}")


def _positive_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = [_float(row, key) for row in rows]
    return [value for value in values if value > 0]


def derive_rows(rows: list[dict[str, str]], *, run_id: str) -> list[dict[str, object]]:
    by_policy = grouped_by_policy(rows)
    missing = sorted(policy for policy in REQUIRED_POLICIES if not by_policy.get(policy))
    if missing:
        raise ValueError("missing policy rows: " + ", ".join(missing))
    _validate_comparable_workload(by_policy)

    baseline_restore = statistics.mean(
        _positive_values(by_policy["no backup"], "mean_restore_sleep_ms")
    )
    baseline_p90 = statistics.mean(
        _positive_values(by_policy["no backup"], "p90_e2e_ms")
    )
    if baseline_restore <= 0 or baseline_p90 <= 0:
        raise ValueError("policy_no_backup baseline recovery metrics must be positive")

    current_sha = git_sha()
    out: list[dict[str, object]] = []
    for policy in REQUIRED_POLICIES:
        scenario_rows = by_policy[policy]
        retry_values = [
            _float(row, "mean_restore_sleep_ms") / baseline_restore
            for row in scenario_rows
            if _float(row, "mean_restore_sleep_ms") > 0
        ]
        critical_values = [
            _float(row, "p90_e2e_ms") / baseline_p90
            for row in scenario_rows
            if _float(row, "p90_e2e_ms") > 0
        ]
        retry_mean, retry_std, retry_ci = _stats(retry_values)
        critical_mean, critical_std, critical_ci = _stats(critical_values)
        first = scenario_rows[0]
        workload_id = ";".join(
            sorted({str(row.get("scenario", "")) for row in scenario_rows})
        )
        replay_rows = sum(_int(row, "replay_rows") for row in scenario_rows)
        failed_rows = sum(_int(row, "failed") for row in scenario_rows)
        out.append(
            {
                "policy": policy,
                "retry_latency_norm": retry_mean,
                "critical_agent_latency_norm": critical_mean,
                "retry_latency_norm_std": retry_std,
                "retry_latency_norm_ci95": retry_ci,
                "critical_agent_latency_norm_std": critical_std,
                "critical_agent_latency_norm_ci95": critical_ci,
                "evidence_tier": EVIDENCE_TIER,
                "source": SOURCE,
                "run_id": run_id,
                "git_sha": current_sha,
                "remote_host": "",
                "slurm_job_id": "",
                "model_path": "mock-model",
                "workload_id": workload_id,
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
                "runtime_replay_rows": replay_rows,
                "runtime_failed_rows": failed_rows,
                "runtime_actions": ";".join(
                    f"{name}={sum(_int(row, name) for row in scenario_rows)}"
                    for name in (
                        "replay_action_host_restore",
                        "replay_action_peer_gpu_restore",
                        "replay_action_recompute",
                        "replay_action_none",
                    )
                ),
                "note": (
                    "Online SGLSim policy sweep; retry latency is "
                    "mean restore sleep normalized to no-backup recompute, and "
                    "critical-agent latency is p90 end-to-end latency normalized "
                    "to the matched no-backup seed set."
                ),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project agentic workflow policy sweep rows into Fig. 9."
    )
    parser.add_argument("--paper-candidates-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FIG9_CSV)
    args = parser.parse_args()

    rows = derive_rows(read_csv(args.paper_candidates_csv), run_id=args.run_id)
    write_csv(args.output_csv, rows)
    print(f"wrote {len(rows)} online recovery-latency rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
