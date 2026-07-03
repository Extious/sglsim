#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[6]
EXP_DIR = ROOT / "benchmark" / "AegisKV" / "draft" / "experiments" / "sglsim_large"
FIGURE_CSV = (
    ROOT
    / "AegisKV"
    / "draft"
    / "figures"
    / "data"
    / "measured"
    / "fig_shared_prefix_marginal.csv"
)
DEFAULT_RESULTS_ROOT = EXP_DIR / "results" / "shared_prefix_sweep"
DEFAULT_SCENARIOS = (
    "shared_prefix_sweep_low",
    "shared_prefix_sweep_mid",
    "shared_prefix_sweep_high",
)

for path in (ROOT, ROOT / "python", Path(__file__).resolve().parent):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from derive_shared_prefix_marginal_from_replay import derive_rows  # noqa: E402
from derive_restore_boundary_from_replay import write_csv  # noqa: E402
from run_agentic_workflow_suite import run_agentic_suite  # noqa: E402


def _collect_run_dirs(summary: dict[str, Any]) -> list[Path]:
    run_dirs: list[Path] = []
    for manifest in summary.get("manifests", []):
        if not isinstance(manifest, dict):
            continue
        for row in manifest.get("plan", []):
            if not isinstance(row, dict):
                continue
            run_dir = str(row.get("run_dir") or "").strip()
            if run_dir:
                run_dirs.append(Path(run_dir))
    return run_dirs


def run_sweep(
    *,
    scenarios: list[str],
    suite_id: str,
    runs: int,
    results_root: Path,
    output_csv: Path | None,
    publish: bool,
    final_evidence: bool,
    min_group_count: int,
    min_shared_prefix_span: float,
    dry_run: bool,
    log_level: str,
) -> dict[str, Any]:
    summary = run_agentic_suite(
        scenarios=scenarios,
        results_root=results_root,
        suite_id=suite_id,
        runs=runs,
        dry_run=dry_run,
        log_level=log_level,
    )
    suite_root = Path(str(summary["suite_root"]))
    run_dirs = _collect_run_dirs(summary)
    output_csv = output_csv or (suite_root / "fig_shared_prefix_marginal.csv")
    result: dict[str, Any] = {
        "suite_id": suite_id,
        "suite_root": str(suite_root),
        "dry_run": bool(dry_run),
        "scenarios": scenarios,
        "runs": int(runs),
        "run_dirs": [str(path) for path in run_dirs],
        "output_csv": str(output_csv),
        "published_csv": str(FIGURE_CSV) if publish else "",
        "final_evidence": bool(final_evidence),
    }
    if dry_run:
        result["note"] = "dry run only; no shared-prefix CSV was derived"
        (suite_root / "shared_prefix_sweep_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return result
    if not run_dirs:
        raise ValueError("suite completed without run directories")

    evidence_tier = "final" if final_evidence else "sglsim-runtime-request-derived"
    rows = derive_rows(
        run_dirs=run_dirs,
        run_id=suite_id,
        bin_width=5.0,
        min_group_count=min_group_count,
        evidence_tier=evidence_tier,
        source="sglsim_agentic_online_shared_prefix_sweep",
        workload_id="aegiskv-sglsim-shared-prefix-broad-sweep",
    )
    shared_values = [float(row["shared_prefix_pct"]) for row in rows]
    span = max(shared_values) - min(shared_values)
    if span < min_shared_prefix_span:
        raise ValueError(
            "shared-prefix sweep span is too narrow: "
            f"{min(shared_values):.2f}..{max(shared_values):.2f} "
            f"< required span {min_shared_prefix_span:.2f}"
        )
    write_csv(output_csv, rows)
    if publish:
        FIGURE_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_csv, FIGURE_CSV)
    result.update(
        {
            "rows": len(rows),
            "shared_prefix_pct": shared_values,
            "shared_prefix_span": span,
            "runtime_agent_groups": sum(
                int(row["runtime_agent_groups"]) for row in rows
            ),
            "evidence_tier": evidence_tier,
        }
    )
    (suite_root / "shared_prefix_sweep_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run low/mid/high SGLSim shared-prefix scenarios and derive "
            "Fig. 10b marginal SafeCover rows."
        )
    )
    parser.add_argument("--scenario", action="append", help="Scenario to include.")
    parser.add_argument("--suite-id", default="")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--final-evidence", action="store_true")
    parser.add_argument("--min-group-count", type=int, default=3)
    parser.add_argument("--min-shared-prefix-span", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    suite_id = args.suite_id or (
        "sglsim-shared-prefix-sweep-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    result = run_sweep(
        scenarios=args.scenario or list(DEFAULT_SCENARIOS),
        suite_id=suite_id,
        runs=args.runs,
        results_root=args.results_root,
        output_csv=args.output_csv,
        publish=bool(args.publish),
        final_evidence=bool(args.final_evidence),
        min_group_count=args.min_group_count,
        min_shared_prefix_span=args.min_shared_prefix_span,
        dry_run=bool(args.dry_run),
        log_level=args.log_level,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
