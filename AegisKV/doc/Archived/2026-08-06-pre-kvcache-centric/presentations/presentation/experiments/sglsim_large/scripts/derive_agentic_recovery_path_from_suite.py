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
DEFAULT_FIG10_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_recovery_path_breakdown.csv"
)

PATH_SOURCE = "sglsim_agentic_online_suite_recovery_path"
EVIDENCE_TIER = "sglsim-agentic-online-preliminary"


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


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


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


def _action_bucket(row: dict[str, str]) -> str:
    action = str(row.get("aegis_replay_restore_action") or "").strip().lower()
    if action in {"host", "host_h2d", "host_restore"}:
        return "host_restore"
    if action in {"gpu", "gpu_replica", "gpu_restore", "peer_gpu", "peer_gpu_restore"}:
        return "peer_gpu_restore"
    if action in {"recompute", "recompute_only"}:
        return "recompute"
    if action in {"", "none", "no_restore"}:
        return "none"
    return "other"


def _action_tokens(row: dict[str, str], bucket: str) -> int:
    if bucket in {"host_restore", "peer_gpu_restore"}:
        return int(round(_float(row, "aegis_replay_restored_tokens")))
    return int(
        round(
            _float(row, "aegis_replay_restored_tokens")
            + _float(row, "aegis_replay_recompute_tokens")
        )
    )


def _empty_trial(row: dict[str, str]) -> dict[str, object]:
    return {
        "scenario": row.get("scenario", ""),
        "dp_size": row.get("dp_size", ""),
        "workflow_count": row.get("workflow_count", ""),
        "source_run_dir": row.get("source_run_dir", ""),
        "replay_rows": int(round(_float(row, "replay_rows"))),
        "host_tokens": 0,
        "peer_tokens": 0,
        "recompute_tokens": 0,
        "none_tokens": 0,
        "other_tokens": 0,
        "host_actions": 0,
        "peer_actions": 0,
        "recompute_actions": 0,
        "none_actions": 0,
        "other_actions": 0,
    }


def _trial_from_summary_row(row: dict[str, str]) -> dict[str, object] | None:
    action_keys = [
        "replay_action_host_restore",
        "replay_action_peer_gpu_restore",
        "replay_action_recompute",
        "replay_action_none",
        "replay_action_other",
        "replay_tokens_host_restore",
        "replay_tokens_peer_gpu_restore",
        "replay_tokens_recompute",
        "replay_tokens_none",
        "replay_tokens_other",
    ]
    if not any(str(row.get(key, "")).strip() for key in action_keys):
        return None

    trial = _empty_trial(row)
    trial.update(
        {
            "host_tokens": int(round(_float(row, "replay_tokens_host_restore"))),
            "peer_tokens": int(round(_float(row, "replay_tokens_peer_gpu_restore"))),
            "recompute_tokens": int(
                round(
                    _float(row, "replay_tokens_recompute")
                    or _float(row, "recompute_tokens")
                )
            ),
            "none_tokens": int(round(_float(row, "replay_tokens_none"))),
            "other_tokens": int(round(_float(row, "replay_tokens_other"))),
            "host_actions": int(round(_float(row, "replay_action_host_restore"))),
            "peer_actions": int(round(_float(row, "replay_action_peer_gpu_restore"))),
            "recompute_actions": int(round(_float(row, "replay_action_recompute"))),
            "none_actions": int(round(_float(row, "replay_action_none"))),
            "other_actions": int(round(_float(row, "replay_action_other"))),
        }
    )
    return trial


def _trial_from_request_details(row: dict[str, str]) -> dict[str, object]:
    trial = _empty_trial(row)
    run_dir = Path(str(trial["source_run_dir"]))
    request_csv = run_dir / "request_details.csv"
    if not request_csv.exists():
        raise FileNotFoundError(f"missing request_details.csv under {run_dir}")

    replay_rows = 0
    with request_csv.open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            if not _bool(item.get("aegis_failover_replay")):
                continue
            replay_rows += 1
            bucket = _action_bucket(item)
            tokens = _action_tokens(item, bucket)
            trial["recompute_tokens"] = int(trial["recompute_tokens"]) + int(
                round(_float(item, "aegis_replay_recompute_tokens"))
            )
            if bucket == "host_restore":
                trial["host_actions"] = int(trial["host_actions"]) + 1
                trial["host_tokens"] = int(trial["host_tokens"]) + tokens
            elif bucket == "peer_gpu_restore":
                trial["peer_actions"] = int(trial["peer_actions"]) + 1
                trial["peer_tokens"] = int(trial["peer_tokens"]) + tokens
            elif bucket == "recompute":
                trial["recompute_actions"] = int(trial["recompute_actions"]) + 1
            elif bucket == "none":
                trial["none_actions"] = int(trial["none_actions"]) + 1
                trial["none_tokens"] = int(trial["none_tokens"]) + tokens
            else:
                trial["other_actions"] = int(trial["other_actions"]) + 1
                trial["other_tokens"] = int(trial["other_tokens"]) + tokens
    trial["replay_rows"] = replay_rows
    return trial


def _trial_from_row(row: dict[str, str]) -> dict[str, object]:
    return _trial_from_summary_row(row) or _trial_from_request_details(row)


def grouped_trials(rows: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        scenario = str(row.get("scenario", ""))
        if not scenario:
            continue
        out[scenario].append(_trial_from_row(row))
    return out


def _ratio(trial: dict[str, object], key: str) -> float:
    total = (
        int(trial["host_tokens"])
        + int(trial["peer_tokens"])
        + int(trial["recompute_tokens"])
    )
    if total <= 0:
        total = (
            int(trial["host_actions"])
            + int(trial["peer_actions"])
            + int(trial["recompute_actions"])
        )
        if total <= 0:
            return 0.0
        return int(trial[key.replace("_tokens", "_actions")]) / total
    return int(trial[key]) / total


def make_row(
    *,
    policy: str,
    scenario: str,
    trials: list[dict[str, object]],
    run_id: str,
    current_sha: str,
) -> dict[str, object]:
    peer_values = [_ratio(trial, "peer_tokens") for trial in trials]
    host_values = [_ratio(trial, "host_tokens") for trial in trials]
    recompute_values = [_ratio(trial, "recompute_tokens") for trial in trials]
    ordinary_values = [0.0 for _ in trials]
    peer_mean, peer_std, peer_ci = _stats(peer_values)
    host_mean, host_std, host_ci = _stats(host_values)
    ordinary_mean, ordinary_std, ordinary_ci = _stats(ordinary_values)
    recompute_mean, recompute_std, recompute_ci = _stats(recompute_values)
    first = trials[0]
    unknown_tokens = sum(
        int(trial["none_tokens"]) + int(trial["other_tokens"]) for trial in trials
    )
    unknown_actions = sum(
        int(trial["none_actions"]) + int(trial["other_actions"]) for trial in trials
    )
    return {
        "policy": policy,
        "peer_gpu": peer_mean,
        "host_h2d": host_mean,
        "ordinary_cache": ordinary_mean,
        "recompute": recompute_mean,
        "peer_gpu_std": peer_std,
        "host_h2d_std": host_std,
        "ordinary_cache_std": ordinary_std,
        "recompute_std": recompute_std,
        "peer_gpu_ci95": peer_ci,
        "host_h2d_ci95": host_ci,
        "ordinary_cache_ci95": ordinary_ci,
        "recompute_ci95": recompute_ci,
        "evidence_tier": EVIDENCE_TIER,
        "source": PATH_SOURCE,
        "run_id": run_id,
        "git_sha": current_sha,
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": scenario,
        "dp_size": first.get("dp_size", ""),
        "tp_size": 1,
        "failure_rank": 0,
        "repeat": f"mean-of-{len(trials)}-agentic-trials",
        "repeat_count": len(trials),
        "repeat_run_ids": ";".join(str(trial["source_run_dir"]) for trial in trials),
        "runtime_replay_rows": sum(int(trial["replay_rows"]) for trial in trials),
        "runtime_regime_count": 1,
        "runtime_regimes": scenario,
        "note": (
            "online SGLSim observed failover replay restore-action distribution; "
            "lookup hits are not counted as path choices"
            + (
                f"; ignored zero/unknown path actions={unknown_actions} "
                f"tokens={unknown_tokens}"
                if unknown_actions or unknown_tokens
                else ""
            )
        ),
        "runtime_regime_summaries": "",
    }


def derive_rows(rows: list[dict[str, str]], *, run_id: str) -> list[dict[str, object]]:
    by_scenario = grouped_trials(rows)
    labels = {
        "hard_failover_host": "online DP2 Host",
        "hard_failover_peer": "online DP4 peer",
        "hard_failover_peer_hot_tail": "online DP4 peer hot-tail",
        "hard_failover_nonpeer": "online DP4 non-peer",
    }
    selected = [scenario for scenario in labels if by_scenario.get(scenario)]
    if not selected:
        known = ", ".join(labels)
        raise ValueError(f"no supported scenario rows found; expected one of: {known}")

    current_sha = git_sha()
    return [
        make_row(
            policy=labels[scenario],
            scenario=scenario,
            trials=by_scenario[scenario],
            run_id=run_id,
            current_sha=current_sha,
        )
        for scenario in selected
    ]


def merge_rows(existing: list[dict[str, str]], replacement: list[dict[str, object]]) -> list[dict[str, object]]:
    retained = [
        row
        for row in existing
        if row.get("source") != PATH_SOURCE
    ]
    return [*retained, *replacement]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project agentic workflow online suite rows into Fig. 10 recovery path rows."
    )
    parser.add_argument("--paper-candidates-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_FIG10_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FIG10_CSV)
    args = parser.parse_args()

    replacement = derive_rows(read_csv(args.paper_candidates_csv), run_id=args.run_id)
    merged = merge_rows(read_csv(args.input_csv), replacement)
    write_csv(args.output_csv, merged)
    print(f"merged {len(replacement)} agentic recovery-path rows into {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
