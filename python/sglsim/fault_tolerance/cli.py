from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from sglsim.fault_tolerance.config import Strategy, load_experiment_config
from sglsim.fault_tolerance.simulator import FaultToleranceSimulator, compare_strategies

DEFAULT_COMPARE: list[Strategy] = ["baseline", "host_backup", "gpu_backup"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only fault-tolerance simulations.")
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Directory containing server/client/failure JSON files.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for simulator outputs.")
    parser.add_argument(
        "--strategy",
        choices=["baseline", "remote_backup", "host_backup", "gpu_backup"],
        default=None,
        help="Override strategy from server.json.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run multiple strategies into subdirectories.",
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma-separated strategies for --compare.",
    )
    parser.add_argument(
        "--sim-config-path",
        default="",
        help="Optional simulator JSON for bandwidth / peer settings.",
    )
    args = parser.parse_args(argv)

    config = load_experiment_config(
        Path(args.config_dir),
        sim_config_path=args.sim_config_path,
    )
    output_dir = Path(args.output_dir)
    if args.compare:
        raw = args.strategies.strip() or ",".join(DEFAULT_COMPARE)
        strategies = cast(
            list[Strategy],
            [item.strip() for item in raw.split(",") if item.strip()],
        )
        summary = compare_strategies(
            config,
            strategies=strategies,
            output_dir=output_dir,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.strategy:
        config = config.with_strategy(cast(Strategy, args.strategy))
    result = FaultToleranceSimulator(config).run(output_dir=output_dir)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
