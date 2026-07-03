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
DEFAULT_FIG14_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_dp_scale.csv"
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


def grouped_by_scenario(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("scenario", ""))].append(row)
    return out


def make_row(
    *,
    series: str,
    scenario: str,
    rows: list[dict[str, str]],
    baseline_mean_ms: float,
    run_id: str,
    current_sha: str,
) -> dict[str, object]:
    normalized = [_float(row, "mean_e2e_ms") / baseline_mean_ms for row in rows]
    mean_value, std_value, ci_value = _stats(normalized)
    first = rows[0]
    return {
        "dp_size": first.get("dp_size", ""),
        "workflow_count": first.get("workflow_count", ""),
        "series": series,
        "normalized_wct": mean_value,
        "normalized_wct_std": std_value,
        "normalized_wct_ci95": ci_value,
        "evidence_tier": "sglsim-agentic-online-preliminary",
        "source": "sglsim_agentic_online_suite_dp_scale",
        "run_id": run_id,
        "git_sha": current_sha,
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": scenario,
        "tp_size": 1,
        "failure_rank": 0,
        "repeat": f"mean-of-{len(rows)}-agentic-trials",
        "note": (
            "online SGLSim agentic workflow suite; normalized_wct uses "
            "trial mean_e2e_ms normalized to the DP2 HostBackup hard-failover "
            "baseline; still preliminary metadata simulation evidence"
        ),
        "duration_s": "",
        "duration_s_std": "",
        "duration_s_ci95": "",
        "mean_e2e_latency_ms": statistics.mean(
            _float(row, "mean_e2e_ms") for row in rows
        ),
        "replay_rows": sum(int(float(row.get("replay_rows") or 0)) for row in rows),
        "hard_failure_replay_rows": sum(
            int(float(row.get("replay_rows") or 0)) for row in rows
        ),
        "repeat_count": len(rows),
        "repeat_run_ids": ";".join(row.get("source_run_dir", "") for row in rows),
    }


def derive_rows(rows: list[dict[str, str]], *, run_id: str) -> list[dict[str, object]]:
    by_scenario = grouped_by_scenario(rows)
    required = {
        "hard_failover_host",
        "hard_failover_peer",
        "hard_failover_nonpeer",
    }
    missing = sorted(name for name in required if not by_scenario.get(name))
    if missing:
        raise ValueError("missing scenario rows: " + ", ".join(missing))

    baseline_rows = by_scenario["hard_failover_host"]
    baseline_mean = statistics.mean(_float(row, "mean_e2e_ms") for row in baseline_rows)
    if baseline_mean <= 0:
        raise ValueError("baseline mean_e2e_ms must be positive")

    current_sha = git_sha()
    return [
        make_row(
            series="agentic DP4 peer",
            scenario="hard_failover_host",
            rows=baseline_rows,
            baseline_mean_ms=baseline_mean,
            run_id=run_id,
            current_sha=current_sha,
        ),
        make_row(
            series="agentic DP4 peer",
            scenario="hard_failover_peer",
            rows=by_scenario["hard_failover_peer"],
            baseline_mean_ms=baseline_mean,
            run_id=run_id,
            current_sha=current_sha,
        ),
        make_row(
            series="agentic DP4 non-peer",
            scenario="hard_failover_host",
            rows=baseline_rows,
            baseline_mean_ms=baseline_mean,
            run_id=run_id,
            current_sha=current_sha,
        ),
        make_row(
            series="agentic DP4 non-peer",
            scenario="hard_failover_nonpeer",
            rows=by_scenario["hard_failover_nonpeer"],
            baseline_mean_ms=baseline_mean,
            run_id=run_id,
            current_sha=current_sha,
        ),
    ]


def merge_rows(existing: list[dict[str, str]], replacement: list[dict[str, object]]) -> list[dict[str, object]]:
    retained = [
        row
        for row in existing
        if row.get("source") != "sglsim_agentic_online_suite_dp_scale"
    ]
    return [*retained, *replacement]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project agentic workflow online suite rows into Fig. 14 DP-scale rows."
    )
    parser.add_argument("--paper-candidates-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_FIG14_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_FIG14_CSV)
    args = parser.parse_args()

    replacement = derive_rows(read_csv(args.paper_candidates_csv), run_id=args.run_id)
    merged = merge_rows(read_csv(args.input_csv), replacement)
    write_csv(args.output_csv, merged)
    print(f"merged {len(replacement)} agentic DP-scale rows into {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
