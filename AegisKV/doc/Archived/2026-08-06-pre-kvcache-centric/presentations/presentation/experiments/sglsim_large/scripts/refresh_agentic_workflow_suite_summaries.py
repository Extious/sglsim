#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[6]

for path in (ROOT, ROOT / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark.agentic_workflows.aggregate import aggregate_csv  # noqa: E402
from benchmark.agentic_workflows.combine import (  # noqa: E402
    combine_summary_files,
    find_summary_files,
)
from benchmark.agentic_workflows.paper import write_paper_candidates  # noqa: E402
from benchmark.agentic_workflows.summary import (  # noqa: E402
    summarize_run_dir,
    write_summary_csv,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def refresh_suite(suite_root: Path) -> dict[str, Any]:
    suite_root = Path(suite_root).resolve()
    manifest_path = suite_root / "agentic_suite_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = _read_json(manifest_path)

    scenario_summary_rows = 0
    for scenario_manifest in manifest.get("manifests") or []:
        scenario = str(scenario_manifest.get("scenario") or "")
        output_root = Path(str(scenario_manifest.get("output_root") or ""))
        if not scenario or not output_root:
            continue
        summaries = []
        for plan_row in scenario_manifest.get("plan") or []:
            run_dir = str(plan_row.get("run_dir") or "")
            if run_dir:
                summaries.append(summarize_run_dir(Path(run_dir), scenario=scenario))
        if summaries:
            write_summary_csv(output_root / "agentic_workflow_summary.csv", summaries)
            scenario_summary_rows += len(summaries)

    combined_csv = suite_root / "agentic_workflow_online_summary.csv"
    paper_csv = suite_root / "agentic_workflow_paper_candidates.csv"
    aggregate_csv_path = suite_root / "agentic_workflow_paper_aggregate.csv"
    summary_count = combine_summary_files(find_summary_files(suite_root), combined_csv)
    paper_count = write_paper_candidates(combined_csv, paper_csv) if summary_count else 0
    aggregate_count = aggregate_csv(paper_csv, aggregate_csv_path) if paper_count else 0

    summary_json_path = suite_root / "aegiskv_agentic_suite_summary.json"
    summary = _read_json(summary_json_path) if summary_json_path.exists() else manifest
    summary.update(
        {
            "combined_summary_csv": str(combined_csv) if summary_count else "",
            "paper_candidate_csv": str(paper_csv) if paper_count else "",
            "paper_aggregate_csv": str(aggregate_csv_path) if aggregate_count else "",
            "scenario_summary_rows": int(scenario_summary_rows),
            "summary_rows": int(summary_count),
            "paper_candidate_rows": int(paper_count),
            "paper_aggregate_rows": int(aggregate_count),
            "evidence_tier": "sglsim-agentic-online-preliminary",
            "refresh_note": (
                "summaries refreshed from existing request_details.csv files; "
                "no experiments were rerun"
            ),
        }
    )
    _write_json(summary_json_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh paper-facing CSV summaries for an existing agentic workflow suite."
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    args = parser.parse_args()

    print(json.dumps(refresh_suite(args.suite_root), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
