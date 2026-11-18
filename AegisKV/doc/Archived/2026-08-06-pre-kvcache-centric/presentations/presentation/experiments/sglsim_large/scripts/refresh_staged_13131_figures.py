#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
FIGURE_DATA_DIR = ROOT.parent.parent / "figures" / "data" / "measured"

DEFAULT_STAGE_RUN_DIR = (
    ROOT
    / "results"
    / "agentic_workflows"
    / "sglsim-staged-13131-fanout-stagebarrier-20260702"
    / "runs"
    / "trial_000"
    / "20260702_013041"
)
DEFAULT_STAGE_RUN_ID = "sglsim-staged-13131-fanout-stagebarrier-20260702"
DEFAULT_ALPHA_RUN_ROOT = (
    ROOT
    / "results"
    / "agentic_workflows"
    / "sglsim-staged-13131-alpha-online-sweep-20260702"
)
DEFAULT_POLICY = ROOT / "policies" / "aegiskv_alpha_mdp_policy.json"


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def _require_run_dir(path: Path) -> None:
    request_details = path / "request_details.csv"
    if not request_details.exists():
        raise FileNotFoundError(f"missing request_details.csv: {request_details}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh staged 1-3-1-3-1 motivation and alpha figure CSVs from "
            "observed SGLSim runs."
        )
    )
    parser.add_argument(
        "--stage-run-dir",
        type=Path,
        action="append",
        default=[DEFAULT_STAGE_RUN_DIR],
        help="Run directory containing request_details.csv for the staged 1-3-1-3-1 trace.",
    )
    parser.add_argument("--stage-run-id", default=DEFAULT_STAGE_RUN_ID)
    parser.add_argument("--alpha-run-root", type=Path, default=DEFAULT_ALPHA_RUN_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--min-workflows", type=int, default=8)
    parser.add_argument("--loss-tokens", type=int, default=512)
    parser.add_argument("--skip-alpha-online", action="store_true")
    parser.add_argument("--skip-mdp-policy", action="store_true")
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in args.stage_run_dir]
    for run_dir in run_dirs:
        _require_run_dir(run_dir)

    run_dir_args = []
    for run_dir in run_dirs:
        run_dir_args.extend(["--run-dir", str(run_dir)])

    _run(
        [
            sys.executable,
            _script("derive_equal_size_loss_from_stagebarrier.py"),
            "--run-id",
            args.stage_run_id,
            *run_dir_args,
            "--output-csv",
            str(FIGURE_DATA_DIR / "fig_motivation_equal_size_loss.csv"),
            "--loss-tokens",
            str(args.loss_tokens),
            "--min-workflows",
            str(args.min_workflows),
            "--evidence-tier",
            "sglsim-trace-derived-stagebarrier",
        ]
    )
    _run(
        [
            sys.executable,
            _script("derive_critical_backup_priority_from_workflows.py"),
            "--run-id",
            args.stage_run_id,
            *run_dir_args,
            "--output-csv",
            str(FIGURE_DATA_DIR / "fig_critical_backup_priority.csv"),
            "--min-workflows",
            str(args.min_workflows),
            "--evidence-tier",
            "sglsim-trace-derived",
        ]
    )
    _run(
        [
            sys.executable,
            _script("derive_critical_agent_profile_from_stagebarrier.py"),
            *[str(path) for path in run_dirs],
            "--output",
            str(FIGURE_DATA_DIR / "fig_critical_agent_profile.csv"),
            "--run-id",
            args.stage_run_id,
            "--evidence-tier",
            "sglsim-trace-derived-stagebarrier",
        ]
    )
    _run(
        [
            sys.executable,
            _script("derive_staged_alpha_risk_tradeoff_from_stagebarrier.py"),
            "--run-id",
            args.stage_run_id,
            *run_dir_args,
            "--output-csv",
            str(FIGURE_DATA_DIR / "fig_staged_alpha_risk_tradeoff.csv"),
            "--min-workflows",
            str(args.min_workflows),
            "--evidence-tier",
            "sglsim-trace-derived",
        ]
    )

    if not args.skip_mdp_policy:
        _run(
            [
                sys.executable,
                _script("derive_alpha_policy_figure_from_mdp.py"),
                "--policy",
                str(args.policy.resolve()),
                "--output-csv",
                str(FIGURE_DATA_DIR / "fig_alpha_workload_failure_policy.csv"),
            ]
        )

    if not args.skip_alpha_online:
        if not args.alpha_run_root.exists():
            raise FileNotFoundError(f"missing alpha run root: {args.alpha_run_root}")
        _run(
            [
                sys.executable,
                _script("derive_online_alpha_adaptation_from_trace.py"),
            ]
        )

    print(
        json.dumps(
            {
                "stage_run_id": args.stage_run_id,
                "stage_run_dirs": [str(path) for path in run_dirs],
                "alpha_run_root": str(args.alpha_run_root),
                "figure_data_dir": str(FIGURE_DATA_DIR),
                "refreshed": [
                    "fig_motivation_equal_size_loss",
                    "fig_critical_backup_priority",
                    "fig_critical_agent_profile",
                    "fig_staged_alpha_risk_tradeoff",
                    *([] if args.skip_mdp_policy else ["fig_alpha_workload_failure_policy"]),
                    *([] if args.skip_alpha_online else ["fig_online_alpha_adaptation"]),
                ],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
