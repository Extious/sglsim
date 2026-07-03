from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_BASE_CONFIG_DIR, DEFAULT_RESULTS_ROOT, scenario_names
from .run import execute_scenario
from .summary import write_summary_json


DEFAULT_SUITE = ("smoke", "hard_failover_host", "hard_failover_peer", "hard_failover_nonpeer")


def execute_suite(
    *,
    scenarios: list[str],
    output_root: Path,
    suite_id: str | None = None,
    base_config_dir: Path = DEFAULT_BASE_CONFIG_DIR,
    runs: int | None = None,
    log_level: str = "info",
    dry_run: bool = False,
) -> dict[str, Any]:
    resolved_suite_id = suite_id or (
        "agentic-suite-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    suite_root = Path(output_root).resolve() / resolved_suite_id
    manifests: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        manifest = execute_scenario(
            scenario_name=scenario,
            output_root=suite_root,
            run_id=f"{index:02d}-{scenario}",
            base_config_dir=base_config_dir,
            runs=runs,
            log_level=log_level,
            dry_run=dry_run,
        )
        manifests.append(manifest)

    summary = {
        "suite_id": resolved_suite_id,
        "dry_run": bool(dry_run),
        "suite_root": str(suite_root),
        "scenarios": scenarios,
        "manifests": manifests,
    }
    write_summary_json(suite_root / "agentic_suite_manifest.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a suite of SGLSim agentic workflow scenarios."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=scenario_names(),
        help="Scenario to include; repeat to build a custom suite.",
    )
    parser.add_argument("--base-config-dir", type=Path, default=DEFAULT_BASE_CONFIG_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--suite-id", default="")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = execute_suite(
        scenarios=args.scenario or list(DEFAULT_SUITE),
        output_root=args.results_root,
        suite_id=args.suite_id or None,
        base_config_dir=args.base_config_dir,
        runs=args.runs,
        log_level=args.log_level,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
