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
DEFAULT_FIG8_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_overall_wct.csv"
)

SOURCE = "sglsim_agentic_online_policy_sweep"
EVIDENCE_TIER = "sglsim-agentic-online-policy-sweep"

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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty values")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[max(0, min(index, len(ordered) - 1))]


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
        raise ValueError(
            "policy sweep rows lack seed columns; rerun the agentic suite after "
            "refreshing paper-candidate schema"
        )
    unique_seed_sets = {tuple(seeds) for seeds in seed_sets.values()}
    if len(unique_seed_sets) != 1:
        raise ValueError(f"policy rows are not seed-comparable: {seed_sets}")


def _request_details_path(row: dict[str, str]) -> Path:
    source_run_dir = str(row.get("source_run_dir") or "").strip()
    if not source_run_dir:
        raise ValueError(
            "policy sweep rows lack source_run_dir; rerun the agentic suite after "
            "refreshing paper-candidate schema"
        )
    path = Path(source_run_dir) / "request_details.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing request_details.csv for policy row: {path}")
    return path


def _row_time_s(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key, "")
        if value not in ("", None):
            return float(value)
    return 0.0


def _bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first_positive_time_s(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key, "")
        if value in ("", None):
            continue
        number = float(value)
        if number > 0:
            return number
    return 0.0


def _workflow_wcts_ms(row: dict[str, str]) -> list[float]:
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    with _request_details_path(row).open(newline="", encoding="utf-8") as f:
        for request in csv.DictReader(f):
            if str(request.get("status") or "").strip() != "completed":
                continue
            workflow_id = str(request.get("workflow_id") or "").strip()
            if not workflow_id:
                workflow_id = str(request.get("rid") or "").strip()
            if not workflow_id:
                continue
            start = _row_time_s(request, "created_time_s", "queue_start_s")
            finish = _row_time_s(request, "finish_time_s")
            if finish <= 0.0:
                finish = start + _row_time_s(
                    request, "client_wall_e2e_latency_s", "e2e_latency_s"
                )
            if finish <= start:
                continue
            starts[workflow_id] = min(starts.get(workflow_id, start), start)
            finishes[workflow_id] = max(finishes.get(workflow_id, finish), finish)
    return [
        (finishes[workflow_id] - starts[workflow_id]) * 1000.0
        for workflow_id in sorted(finishes)
        if workflow_id in starts and finishes[workflow_id] > starts[workflow_id]
    ]


def _trial_mean_workflow_wct_ms(row: dict[str, str]) -> float:
    values = _workflow_wcts_ms(row)
    if not values:
        raise ValueError(
            f"no completed workflow WCT samples in {_request_details_path(row)}"
        )
    return statistics.mean(values)


def _trial_failure_exposed_restore_ms(row: dict[str, str]) -> float:
    values: list[float] = []
    with _request_details_path(row).open(newline="", encoding="utf-8") as f:
        for request in csv.DictReader(f):
            if not _bool_text(request.get("aegis_failover_replay")):
                continue
            restore_s = _first_positive_time_s(
                request,
                "aegis_replay_restore_sleep_s",
                "aegis_replay_restore_overhead_s",
            )
            if restore_s > 0:
                values.append(restore_s * 1000.0)
    if values:
        return statistics.mean(values)
    return _float(row, "mean_restore_sleep_ms")


def derive_rows(rows: list[dict[str, str]], *, run_id: str) -> list[dict[str, object]]:
    by_policy = grouped_by_policy(rows)
    missing = sorted(policy for policy in REQUIRED_POLICIES if not by_policy.get(policy))
    if missing:
        raise ValueError("missing policy rows: " + ", ".join(missing))
    _validate_comparable_workload(by_policy)

    trial_wct_by_policy = {
        policy: [_trial_mean_workflow_wct_ms(row) for row in scenario_rows]
        for policy, scenario_rows in by_policy.items()
    }
    workflow_wct_by_policy = {
        policy: [
            wct
            for row in scenario_rows
            for wct in _workflow_wcts_ms(row)
        ]
        for policy, scenario_rows in by_policy.items()
    }
    failure_restore_by_policy = {
        policy: [
            value
            for row in scenario_rows
            if (value := _trial_failure_exposed_restore_ms(row)) > 0
        ]
        for policy, scenario_rows in by_policy.items()
    }
    baseline_values = trial_wct_by_policy["no backup"]
    baseline_mean = statistics.mean(value for value in baseline_values if value > 0)
    if baseline_mean <= 0:
        raise ValueError("no-backup workflow WCT must be positive")
    baseline_restore_values = failure_restore_by_policy["no backup"]
    baseline_restore_mean = (
        statistics.mean(baseline_restore_values) if baseline_restore_values else 0.0
    )

    current_sha = git_sha()
    out: list[dict[str, object]] = []
    for policy in REQUIRED_POLICIES:
        scenario_rows = by_policy[policy]
        normalized = [
            value / baseline_mean
            for value in trial_wct_by_policy[policy]
            if value > 0
        ]
        mean_value, std_value, ci_value = _stats(normalized)
        mean_wct, mean_wct_std, mean_wct_ci = _stats(trial_wct_by_policy[policy])
        restore_values = failure_restore_by_policy[policy]
        workflow_samples = workflow_wct_by_policy[policy]
        replay_rows = sum(int(float(row.get("replay_rows") or 0)) for row in scenario_rows)
        failed = sum(int(float(row.get("failed") or 0)) for row in scenario_rows)
        first = scenario_rows[0]
        workload_id = ";".join(
            sorted({str(row.get("scenario", "")) for row in scenario_rows})
        )
        result = {
            "policy": policy,
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
            "runtime_failed_rows": failed,
            "mean_workflow_wct_ms": mean_wct,
            "mean_workflow_wct_ms_std": mean_wct_std,
            "mean_workflow_wct_ms_ci95": mean_wct_ci,
            "p90_workflow_wct_ms": _percentile(workflow_samples, 0.9),
            "workflow_wct_sample_count": len(workflow_samples),
            "mean_e2e_latency_ms": statistics.mean(
                [_float(row, "mean_e2e_ms") for row in scenario_rows]
            ),
            "note": (
                "Online SGLSim 1-3-1-3-1 policy sweep; normalized_wct is "
                "computed from workflow-level completion time in "
                "request_details.csv, not per-request mean latency. "
                "failure_exposed_restore_norm is computed from failover replay "
                "restore sleep in the same request_details.csv files to expose "
                "the recovery component that total WCT can dilute. Rows remain "
                "metadata-level and physical-copy calibrated only when paired "
                "with the recovery-action microbench."
            ),
        }
        if baseline_restore_mean > 0 and restore_values:
            restore_mean, restore_std, restore_ci = _stats(restore_values)
            restore_norm = [
                value / baseline_restore_mean
                for value in restore_values
                if value > 0
            ]
            restore_norm_mean, restore_norm_std, restore_norm_ci = _stats(restore_norm)
            result.update(
                {
                    "failure_exposed_restore_ms": restore_mean,
                    "failure_exposed_restore_ms_std": restore_std,
                    "failure_exposed_restore_ms_ci95": restore_ci,
                    "failure_exposed_restore_norm": restore_norm_mean,
                    "failure_exposed_restore_norm_std": restore_norm_std,
                    "failure_exposed_restore_norm_ci95": restore_norm_ci,
                }
            )
        out.append(result)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project agentic workflow policy sweep rows into Fig. 8."
    )
    parser.add_argument("--paper-candidates-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FIG8_CSV)
    args = parser.parse_args()

    rows = derive_rows(read_csv(args.paper_candidates_csv), run_id=args.run_id)
    write_csv(args.output_csv, rows)
    print(f"wrote {len(rows)} online policy rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
