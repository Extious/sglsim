#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[6]
EXP_DIR = ROOT / "benchmark" / "AegisKV" / "draft" / "experiments" / "sglsim_large"
DEFAULT_CONFIG_DIR = EXP_DIR / "config_failover_replay"
DEFAULT_RESULTS_ROOT = EXP_DIR / "results" / "agentic_workflows"

for path in (ROOT, ROOT / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark.agentic_workflows.suite import (  # noqa: E402
    DEFAULT_SUITE,
    execute_suite,
)
from benchmark.agentic_workflows.aggregate import aggregate_csv  # noqa: E402
from benchmark.agentic_workflows.combine import (  # noqa: E402
    combine_summary_files,
    find_summary_files,
)
from benchmark.agentic_workflows.paper import write_paper_candidates  # noqa: E402


def run_agentic_suite(
    *,
    scenarios: list[str],
    results_root: Path = DEFAULT_RESULTS_ROOT,
    suite_id: str | None = None,
    runs: int | None = None,
    dry_run: bool = False,
    log_level: str = "info",
) -> dict[str, Any]:
    resolved_suite_id = suite_id or (
        "aegiskv-agentic-suite-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    summary = execute_suite(
        scenarios=scenarios,
        output_root=results_root,
        suite_id=resolved_suite_id,
        base_config_dir=DEFAULT_CONFIG_DIR,
        runs=runs,
        log_level=log_level,
        dry_run=dry_run,
    )
    suite_root = Path(str(summary["suite_root"]))
    combined_csv = suite_root / "agentic_workflow_online_summary.csv"
    paper_csv = suite_root / "agentic_workflow_paper_candidates.csv"
    aggregate_csv_path = suite_root / "agentic_workflow_paper_aggregate.csv"

    summary_count = combine_summary_files(find_summary_files(suite_root), combined_csv)
    paper_count = (
        write_paper_candidates(combined_csv, paper_csv) if summary_count else 0
    )
    aggregate_count = aggregate_csv(paper_csv, aggregate_csv_path) if paper_count else 0
    summary.update(
        {
            "combined_summary_csv": str(combined_csv) if summary_count else "",
            "paper_candidate_csv": str(paper_csv) if paper_count else "",
            "paper_aggregate_csv": str(aggregate_csv_path) if aggregate_count else "",
            "summary_rows": int(summary_count),
            "paper_candidate_rows": int(paper_count),
            "paper_aggregate_rows": int(aggregate_count),
            "evidence_tier": "sglsim-agentic-online-preliminary",
            "note": (
                "Agentic workflow online suite wrapper. Rows remain preliminary "
                "until physical KV movement and final policy sweeps exist."
            ),
        }
    )
    (suite_root / "aegiskv_agentic_suite_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AegisKV agentic workflow suite via SGLSim synthesis_requests."
    )
    parser.add_argument("--scenario", action="append", help="Scenario to include.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--suite-id", default="")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = run_agentic_suite(
        scenarios=args.scenario or list(DEFAULT_SUITE),
        results_root=args.results_root,
        suite_id=args.suite_id or None,
        runs=args.runs,
        dry_run=bool(args.dry_run),
        log_level=args.log_level,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
