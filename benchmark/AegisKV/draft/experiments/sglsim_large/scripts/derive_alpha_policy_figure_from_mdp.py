#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "policies" / "aegiskv_alpha_mdp_policy.json"
DEFAULT_OUTPUT = (
    ROOT.parent.parent / "figures" / "data" / "measured" / "fig_alpha_workload_failure_policy.csv"
)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def pressure_label(value: str) -> str:
    return {
        "low": "low critical pressure",
        "medium": "medium critical pressure",
        "high": "high critical pressure",
    }.get(value, f"{value} critical pressure")


def load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"policy file is not a JSON object: {path}")
    states = payload.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError(f"policy file has no state rows: {path}")
    return payload


def derive_rows(policy: dict[str, Any], *, policy_path: Path) -> list[dict[str, object]]:
    training = policy.get("training") if isinstance(policy.get("training"), dict) else {}
    states = policy["states"]
    rows: list[dict[str, object]] = []
    for state in states:
        if not isinstance(state, dict):
            continue
        critical_pressure = str(state["critical_pressure"])
        rows.append(
            {
                "failure_rate_per_window": float(state["failure_rate_per_window"]),
                "critical_pressure": pressure_label(critical_pressure),
                "alpha_backup_fraction": float(state["alpha"]),
                "critical_pressure_value": float(state["critical_pressure_value"]),
                "value": float(state["value"]),
                "evidence_tier": "sglsim-model-policy-table",
                "source": str(policy.get("source", "aegiskv_alpha_mdp_value_iteration_v1")),
                "run_id": policy_path.stem,
                "git_sha": git_sha(),
                "remote_host": "",
                "slurm_job_id": "",
                "model_path": "mock-model",
                "workload_id": "staged-1-3-1-3-1-critical-pressure-grid",
                "dp_size": 4,
                "tp_size": 1,
                "failure_rank": "",
                "repeat": f"value-iteration-{training.get('iterations', '')}",
                "note": (
                    "Fig. 3b is derived from the offline value-iteration MDP "
                    "policy table used by AegisKV's online alpha controller; "
                    "it is model-based policy evidence, not a measured serving sweep."
                ),
                "gamma": training.get("gamma", ""),
                "policy_path": str(policy_path),
            }
        )
    if not rows:
        raise ValueError("policy file did not contain any usable state rows")
    rows.sort(
        key=lambda row: (
            float(row["critical_pressure_value"]),
            float(row["failure_rate_per_window"]),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive Fig. 3b alpha-policy rows from the trained AegisKV MDP policy table."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    rows = derive_rows(policy, policy_path=args.policy)
    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "evidence_tier": "sglsim-model-policy-table",
                "alpha_range": [
                    min(float(row["alpha_backup_fraction"]) for row in rows),
                    max(float(row["alpha_backup_fraction"]) for row in rows),
                ],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
