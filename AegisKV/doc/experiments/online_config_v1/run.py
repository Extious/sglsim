#!/usr/bin/env python3
"""Run an online AegisKV experiment from three explicit JSON layers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from benchmark.synthesis_requests.run import run_experiment  # noqa: E402

try:  # Support both direct script execution and namespace-package imports.
    from .config import (  # type: ignore[import-not-found]
        load_experiment_config,
        materialize_runtime_config,
        write_campaign_snapshot,
    )
except ImportError:
    from config import (  # type: ignore[no-redef]
        load_experiment_config,
        materialize_runtime_config,
        write_campaign_snapshot,
    )


DEFAULT_CONFIG_DIR = _HERE / "configs" / "online-aegiskv-smoke"
DEFAULT_RESULTS_ROOT = _HERE / "results"


def _unique_campaign_dir(results_root: Path, experiment_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_name).strip("-.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(results_root) / f"{stamp}_{safe_name or 'experiment'}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def execute(
    config_dir: Path,
    results_root: Path,
    *,
    dry_run: bool = False,
    log_level: str = "info",
    disable_overlap_schedule: bool = False,
) -> Path:
    config = load_experiment_config(config_dir)
    campaign_dir = _unique_campaign_dir(results_root, config.name)
    effective_dir = materialize_runtime_config(
        config, campaign_dir / "effective_config"
    )
    write_campaign_snapshot(
        config,
        campaign_dir,
        status="materialized" if dry_run else "prepared",
    )
    if dry_run:
        return campaign_dir

    try:
        runtime_run_dir = run_experiment(
            effective_dir,
            campaign_dir / "runtime",
            log_level=log_level,
            disable_overlap_schedule=disable_overlap_schedule,
        )
    except Exception as exc:
        write_campaign_snapshot(
            config,
            campaign_dir,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    write_campaign_snapshot(
        config,
        campaign_dir,
        status="completed",
        runtime_run_dir=runtime_run_dir,
    )
    return campaign_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the online SGLSim AegisKV benchmark from request.json, "
            "failure.json, and deployment.json"
        )
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    args = parser.parse_args(argv)

    try:
        campaign_dir = execute(
            args.config_dir,
            args.results_root,
            dry_run=bool(args.dry_run),
            log_level=str(args.log_level),
            disable_overlap_schedule=bool(args.disable_overlap_schedule),
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    payload = {
        "status": "materialized" if args.dry_run else "completed",
        "campaign_dir": str(campaign_dir),
        "manifest": str(campaign_dir / "experiment_manifest.json"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
