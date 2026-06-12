#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sglsim.fault_tolerance.config import load_experiment_config
from sglsim.fault_tolerance.simulator import compare_strategies
from sglsim.fault_tolerance.steady_state import sweep_concurrency

DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 32]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline fault-tolerance benchmark (none / host_backup / gpu_backup)."
    )
    parser.add_argument(
        "--sim-config",
        default="",
        help="Scenario directory or explicit sim JSON path.",
    )
    parser.add_argument(
        "--config-dir",
        default="",
        help="Experiment config directory (server/client/failure JSON).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--strategies",
        default="none,host_backup,gpu_backup",
        help="Comma-separated kv_backup strategies.",
    )
    parser.add_argument(
        "--concurrency-levels",
        default=",".join(str(v) for v in DEFAULT_CONCURRENCY),
        help="Comma-separated concurrency levels for steady-state sweep.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate academic-style comparison figures.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_dir = Path(args.config_dir) if args.config_dir else repo_root / "benchmark/fault_tolerance/config/scenario"
    sim_config = args.sim_config
    if sim_config and Path(sim_config).is_dir():
        config_dir = Path(sim_config)
        sim_config = ""

    config = load_experiment_config(config_dir, sim_config_path=sim_config)
    alias = {"none": "baseline", "baseline": "baseline"}
    strategies = []
    for item in args.strategies.split(","):
        name = item.strip()
        if not name:
            continue
        strategies.append(alias.get(name, name))

    concurrency_levels = [int(item.strip()) for item in args.concurrency_levels.split(",") if item.strip()]

    output_dir = Path(args.output_dir)
    summary = compare_strategies(config, strategies=strategies, output_dir=output_dir)
    sweep = sweep_concurrency(
        config,
        strategies=strategies,
        concurrency_levels=concurrency_levels,
        output_dir=output_dir / "steady_state",
    )
    print(json.dumps({"fault_injection": summary, "steady_state": sweep}, indent=2))

    if args.plot:
        plot_script = Path(__file__).resolve().parent / "plot_compare.py"
        for extra_args in (
            ["--results-dir", str(output_dir), "--output", str(output_dir / "fault_tolerance_compare")],
            [
                "--results-dir",
                str(output_dir / "steady_state"),
                "--output",
                str(output_dir / "throughput_cache_vs_concurrency"),
                "--mode",
                "concurrency",
            ],
            [
                "--results-dir",
                str(output_dir),
                "--output",
                str(output_dir / "cache_hit_compare"),
                "--mode",
                "cache",
            ],
        ):
            subprocess.run([sys.executable, str(plot_script), *extra_args], check=True)


if __name__ == "__main__":
    main()
