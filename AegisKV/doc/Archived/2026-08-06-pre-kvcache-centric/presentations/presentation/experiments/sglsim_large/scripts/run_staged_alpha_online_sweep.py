#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.agentic_workflows.config import (
    DEFAULT_BASE_CONFIG_DIR,
    materialize_synthesis_config,
    scenario_plan_row,
)
from benchmark.agentic_workflows.config import get_scenario
from benchmark.agentic_workflows.summary import (
    summarize_run_dir,
    write_summary_csv,
    write_summary_json,
)
from benchmark.synthesis_requests.run import run_experiment


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = SGLSIM_LARGE_ROOT / "results" / "agentic_workflows"
DEFAULT_SWEEP_ID = "sglsim-staged-13131-alpha-online-sweep-20260702"


SWEEP_POINTS: tuple[dict[str, Any], ...] = (
    {
        "name": "lowload_fr0003",
        "load_level": "low load",
        "groups": 2,
        "max_concurrency": 6,
        "aegis_critical_pressure": 0.62,
        "failure_rate_per_window": 0.003,
        "seed": 8861,
        "start_port": 32620,
    },
    {
        "name": "lowload_fr001",
        "load_level": "low load",
        "groups": 2,
        "max_concurrency": 6,
        "aegis_critical_pressure": 0.62,
        "failure_rate_per_window": 0.01,
        "seed": 8862,
        "start_port": 32640,
    },
    {
        "name": "highload_fr001",
        "load_level": "high load",
        "groups": 3,
        "max_concurrency": 18,
        "aegis_critical_pressure": 0.86,
        "failure_rate_per_window": 0.01,
        "seed": 8872,
        "start_port": 32660,
    },
    {
        "name": "highload_fr003",
        "load_level": "high load",
        "groups": 3,
        "max_concurrency": 18,
        "aegis_critical_pressure": 0.86,
        "failure_rate_per_window": 0.03,
        "seed": 8873,
        "start_port": 32680,
    },
)


def execute_sweep(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    sweep_id: str = DEFAULT_SWEEP_ID,
    base_config_dir: Path = DEFAULT_BASE_CONFIG_DIR,
    log_level: str = "info",
    dry_run: bool = False,
) -> dict[str, Any]:
    base = get_scenario("staged_13131_alpha_online")
    sweep_root = Path(results_root).resolve() / sweep_id
    plan_rows: list[dict[str, Any]] = []
    summaries = []

    for index, point in enumerate(SWEEP_POINTS):
        scenario = replace(
            base,
            name=f"staged_13131_alpha_{point['name']}",
            description=(
                "1-3-1-3-1 online adaptive-alpha sweep point; "
                f"{point['load_level']} with failure_rate="
                f"{point['failure_rate_per_window']}."
            ),
            groups=int(point["groups"]),
            max_concurrency=int(point["max_concurrency"]),
            aegis_critical_pressure=float(point["aegis_critical_pressure"]),
            failure_rate_per_window=float(point["failure_rate_per_window"]),
            seed=int(point["seed"]),
            start_port=int(point["start_port"]),
            runs=1,
        )
        config_dir = materialize_synthesis_config(
            scenario,
            base_config_dir=base_config_dir,
            output_dir=sweep_root,
            trial=0,
        )
        plan_row = scenario_plan_row(scenario, config_dir=config_dir, trial=0)
        plan_row.update(
            {
                "sweep_point": point["name"],
                "sweep_index": index,
                "load_level": point["load_level"],
            }
        )
        plan_rows.append(plan_row)
        if dry_run:
            continue

        run_dir = run_experiment(
            config_dir,
            sweep_root / "runs" / str(point["name"]),
            log_level=log_level,
            disable_overlap_schedule=False,
        )
        summary = summarize_run_dir(run_dir, scenario=scenario.name)
        summaries.append(summary)
        plan_row["run_dir"] = str(run_dir)

    if summaries:
        write_summary_csv(sweep_root / "agentic_workflow_summary.csv", summaries)

    manifest = {
        "run_id": sweep_id,
        "scenario": "staged_13131_alpha_online_sweep",
        "dry_run": bool(dry_run),
        "base_scenario": "staged_13131_alpha_online",
        "base_config_dir": str(Path(base_config_dir).resolve()),
        "output_root": str(sweep_root),
        "sweep_points": list(SWEEP_POINTS),
        "plan": plan_rows,
        "summary_csv": (
            str(sweep_root / "agentic_workflow_summary.csv")
            if summaries
            else ""
        ),
        "evidence_tier": "sglsim-agentic-online-alpha-sweep",
    }
    write_summary_json(sweep_root / "agentic_workflow_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run 1-3-1-3-1 online adaptive-alpha SGLSim sweep."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--sweep-id", default=DEFAULT_SWEEP_ID)
    parser.add_argument("--base-config-dir", type=Path, default=DEFAULT_BASE_CONFIG_DIR)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = execute_sweep(
        results_root=args.results_root,
        sweep_id=args.sweep_id,
        base_config_dir=args.base_config_dir,
        log_level=args.log_level,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
