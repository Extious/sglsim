from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PYTHON = REPO / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from benchmark.synthesis_requests.run import run_experiment  # noqa: E402
from benchmark.synthesis_requests.utils import DEFAULT_RESULTS_ROOT  # noqa: E402


DEFAULT_CONFIGS = [
    "benchmark/synthesis_requests/config/ft_nohicache_baseline",
    "benchmark/synthesis_requests/config/ft_nohicache_gpu_backup",
]

MULTI_AGENT_CONFIGS = [
    "benchmark/synthesis_requests/config/ma_ft_baseline",
    "benchmark/synthesis_requests/config/ma_ft_gpu_backup_all",
    "benchmark/synthesis_requests/config/ma_ft_gpu_backup_critical",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthesis fault-tolerance suite")
    parser.add_argument(
        "--config-dir",
        action="append",
        type=Path,
        help="Config directory to run. Defaults to baseline and gpu_backup.",
    )
    parser.add_argument(
        "--suite",
        choices=("synthesis", "multi-agent", "all"),
        default="synthesis",
        help="Named experiment suite when --config-dir is not provided.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
    )
    args = parser.parse_args(argv)

    if args.config_dir:
        configs = args.config_dir
    elif args.suite == "multi-agent":
        configs = [REPO / cfg for cfg in MULTI_AGENT_CONFIGS]
    elif args.suite == "all":
        configs = [REPO / cfg for cfg in DEFAULT_CONFIGS + MULTI_AGENT_CONFIGS]
    else:
        configs = [REPO / cfg for cfg in DEFAULT_CONFIGS]
    runs: dict[str, str] = {}
    for config_dir in configs:
        run_dir = run_experiment(
            Path(config_dir),
            args.results_root,
            log_level=args.log_level,
            disable_overlap_schedule=bool(args.disable_overlap_schedule),
        )
        name = Path(config_dir).name
        runs[name] = str(run_dir)

    manifest = args.results_root / "fault_tolerance_suite_latest.json"
    manifest.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(runs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
