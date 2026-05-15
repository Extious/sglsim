from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from sglsim.fault_tolerance.config import Strategy, load_experiment_config
from sglsim.fault_tolerance.simulator import FaultToleranceSimulator, compare_strategies


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only fault-tolerance simulations.")
    parser.add_argument("--config-dir", required=True, help="Directory containing server/client/failure JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory for simulator outputs.")
    parser.add_argument(
        "--strategy",
        choices=["baseline", "remote_backup", "host_backup"],
        default=None,
        help="Override strategy from server.json.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline, remote_backup, and host_backup into subdirectories.",
    )
    args = parser.parse_args(argv)

    config = load_experiment_config(Path(args.config_dir))
    output_dir = Path(args.output_dir)
    if args.compare:
        summary = compare_strategies(
            config,
            strategies=["baseline", "remote_backup", "host_backup"],
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
