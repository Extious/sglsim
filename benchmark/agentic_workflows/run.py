from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_BASE_CONFIG_DIR,
    DEFAULT_RESULTS_ROOT,
    get_scenario,
    materialize_synthesis_config,
    scenario_names,
    scenario_plan_row,
)
from .summary import summarize_run_dir, write_summary_csv, write_summary_json


def execute_scenario(
    *,
    scenario_name: str,
    output_root: Path,
    run_id: str | None = None,
    base_config_dir: Path = DEFAULT_BASE_CONFIG_DIR,
    runs: int | None = None,
    seed_base: int | None = None,
    start_port: int | None = None,
    log_level: str = "info",
    dry_run: bool = False,
    disable_overlap_schedule: bool | None = None,
) -> dict[str, Any]:
    scenario = get_scenario(scenario_name).with_overrides(
        runs=runs,
        seed=seed_base,
        start_port=start_port,
    )
    if disable_overlap_schedule is not None:
        from dataclasses import replace

        scenario = replace(
            scenario,
            disable_overlap_schedule=bool(disable_overlap_schedule),
        )

    resolved_run_id = run_id or (
        f"agentic-{scenario.name}-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    scenario_root = Path(output_root).resolve() / resolved_run_id
    plan_rows: list[dict[str, Any]] = []
    summaries = []

    for trial in range(int(scenario.runs)):
        config_dir = materialize_synthesis_config(
            scenario,
            base_config_dir=base_config_dir,
            output_dir=scenario_root,
            trial=trial,
        )
        row = scenario_plan_row(scenario, config_dir=config_dir, trial=trial)
        plan_rows.append(row)
        if dry_run:
            continue

        from benchmark.synthesis_requests.run import run_experiment

        trial_results_root = scenario_root / "runs" / f"trial_{trial:03d}"
        run_dir = run_experiment(
            config_dir,
            trial_results_root,
            log_level=log_level,
            disable_overlap_schedule=bool(scenario.disable_overlap_schedule),
        )
        summary = summarize_run_dir(run_dir, scenario=scenario.name)
        summaries.append(summary)
        row["run_dir"] = str(run_dir)

    if summaries:
        write_summary_csv(scenario_root / "agentic_workflow_summary.csv", summaries)

    manifest = {
        "run_id": resolved_run_id,
        "scenario": scenario.name,
        "dry_run": bool(dry_run),
        "runs": int(scenario.runs),
        "base_config_dir": str(Path(base_config_dir).resolve()),
        "output_root": str(scenario_root),
        "plan": plan_rows,
        "summary_csv": (
            str(scenario_root / "agentic_workflow_summary.csv")
            if summaries
            else ""
        ),
        "evidence_tier": "sglsim-agentic-online-preliminary",
    }
    write_summary_json(scenario_root / "agentic_workflow_manifest.json", manifest)
    return manifest


def _print_scenarios() -> None:
    for name in scenario_names():
        scenario = get_scenario(name)
        print(
            f"{name}: dp={scenario.dp_size} workflows={scenario.workflow_count} "
            f"turns={scenario.turns} failover={scenario.failover_replay}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SGLSim agentic workflow scenarios via synthesis_requests."
    )
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--scenario", default="smoke", choices=scenario_names())
    parser.add_argument("--base-config-dir", type=Path, default=DEFAULT_BASE_CONFIG_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--start-port", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    args = parser.parse_args(argv)

    if args.list_scenarios:
        _print_scenarios()
        return 0

    manifest = execute_scenario(
        scenario_name=args.scenario,
        output_root=args.results_root,
        run_id=args.run_id or None,
        base_config_dir=args.base_config_dir,
        runs=args.runs,
        seed_base=args.seed_base,
        start_port=args.start_port,
        log_level=args.log_level,
        dry_run=bool(args.dry_run),
        disable_overlap_schedule=(
            True if args.disable_overlap_schedule else None
        ),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
