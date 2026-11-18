#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
PYTHON_ROOT = REPO_ROOT / "python"
for path in (REPO_ROOT, PYTHON_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark.AegisKV.draft.experiments.sglsim_large.scripts.audit_live_physical_movement import (  # noqa: E402
    audit_suite,
    render_markdown,
    write_csv,
)


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = SGLSIM_LARGE_ROOT / "results" / "agentic_workflows"
DEFAULT_OUTPUT_DIR = SGLSIM_LARGE_ROOT / "results" / "strict_final_readiness"

DEFAULT_SUITES: tuple[tuple[str, str, str], ...] = (
    (
        "fig8_9_10_policy_sweep",
        "Fig. 8/9/10",
        "sglsim-staged-13131-policy-stress-v2-r3-20260702",
    ),
    (
        "fig10_14_recovery_action_suite",
        "Fig. 10/14",
        "sglsim-current-physical-recovery-action-agentic-r3-20260702",
    ),
)


def _audit_one(
    *,
    label: str,
    figures: str,
    suite_root: Path,
) -> dict[str, Any]:
    rows, totals = audit_suite(suite_root)
    output_csv = suite_root / "live_physical_movement_audit.csv"
    output_md = suite_root / "live_physical_movement_audit.md"
    write_csv(output_csv, rows)
    output_md.write_text(render_markdown(rows, totals), encoding="utf-8")
    return {
        "data_planes": totals["data_planes"],
        "figures": figures,
        "final_ready": bool(totals["final_ready"]),
        "label": label,
        "missing_data_plane_rows": int(totals["missing_data_plane_rows"]),
        "missing_timing_source_rows": int(totals["missing_timing_source_rows"]),
        "nonfinal_restore_rows": int(totals["nonfinal_restore_rows"]),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "physical_tensor_copy_rows": int(totals["physical_tensor_copy_rows"]),
        "replay_rows": int(totals["replay_rows"]),
        "request_csv_count": int(totals["request_csv_count"]),
        "restore_rows": int(totals["restore_rows"]),
        "schema_complete": bool(totals["schema_complete"]),
        "suite_root": str(suite_root),
        "timing_sources": totals["timing_sources"],
    }


def build_summary(suite_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_restore = sum(int(row["restore_rows"]) for row in suite_rows)
    total_physical = sum(int(row["physical_tensor_copy_rows"]) for row in suite_rows)
    total_nonfinal = sum(int(row["nonfinal_restore_rows"]) for row in suite_rows)
    total_missing_data_plane = sum(
        int(row["missing_data_plane_rows"]) for row in suite_rows
    )
    total_missing_timing_source = sum(
        int(row["missing_timing_source_rows"]) for row in suite_rows
    )
    return {
        "final_ready": (
            total_restore > 0
            and total_nonfinal == 0
            and total_missing_data_plane == 0
            and total_missing_timing_source == 0
        ),
        "missing_data_plane_rows": total_missing_data_plane,
        "missing_timing_source_rows": total_missing_timing_source,
        "nonfinal_restore_rows": total_nonfinal,
        "physical_tensor_copy_rows": total_physical,
        "restore_rows": total_restore,
        "schema_complete": (
            total_missing_data_plane == 0 and total_missing_timing_source == 0
        ),
        "suite_count": len(suite_rows),
    }


def render_summary_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# P0 Live Movement Audit Artifacts",
        "",
        "This report records the per-suite live movement audit files generated for the",
        "P0 strict-final data-plane gate. It does not convert metadata materialization",
        "rows into final physical tensor-copy evidence.",
        "",
        "## Summary",
        "",
        f"- restore rows: `{summary['restore_rows']}`",
        f"- physical tensor copy rows: `{summary['physical_tensor_copy_rows']}`",
        f"- non-final restore rows: `{summary['nonfinal_restore_rows']}`",
        f"- missing data-plane rows: `{summary['missing_data_plane_rows']}`",
        f"- missing timing-source rows: `{summary['missing_timing_source_rows']}`",
        f"- schema-complete: `{str(summary['schema_complete']).lower()}`",
        f"- final-ready: `{str(summary['final_ready']).lower()}`",
        "",
        "## Suites",
        "",
        "| Suite | Figures | Request CSVs | Replay rows | Restore rows | Physical tensor rows | Non-final rows | Data planes | Audit |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["suites"]:
        lines.append(
            "| {label} | {figures} | {csvs} | {replay} | {restore} | {physical} | {nonfinal} | `{planes}` | `{audit}` |".format(
                label=row["label"],
                figures=row["figures"],
                csvs=row["request_csv_count"],
                replay=row["replay_rows"],
                restore=row["restore_rows"],
                physical=row["physical_tensor_copy_rows"],
                nonfinal=row["nonfinal_restore_rows"],
                planes=json.dumps(row["data_planes"], sort_keys=True),
                audit=row["output_md"],
            )
        )
    lines.extend(
        [
            "",
            "## Strict-Final Boundary",
            "",
            "The P0 gate passes only when every restore row reports",
            "`aegis_replay_restore_data_plane=physical_tensor_copy` and carries a timing",
            "source. Current metadata-materialized rows remain hybrid/preliminary evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    specs: list[tuple[str, str, Path]] | None = None,
) -> dict[str, Any]:
    suite_specs = specs or [
        (label, figures, RESULTS_ROOT / root)
        for label, figures, root in DEFAULT_SUITES
    ]
    suite_rows: list[dict[str, Any]] = []
    for label, figures, suite_root in suite_specs:
        if not suite_root.exists():
            raise FileNotFoundError(f"missing suite root: {suite_root}")
        suite_rows.append(
            _audit_one(label=label, figures=figures, suite_root=suite_root)
        )

    payload = {
        "source": "write_p0_live_movement_audits",
        "summary": build_summary(suite_rows),
        "suites": suite_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "p0_live_movement_audits.json"
    output_md = output_dir / "p0_live_movement_audits.md"
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_summary_markdown(payload), encoding="utf-8")
    payload["output_json"] = str(output_json)
    payload["output_md"] = str(output_md)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write per-suite P0 live movement audit artifacts."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--suite",
        action="append",
        nargs=3,
        metavar=("LABEL", "FIGURES", "SUITE_ROOT"),
        help="Override or append suite as label figures suite_root.",
    )
    args = parser.parse_args(argv)
    specs = None
    if args.suite:
        specs = [(label, figures, Path(root)) for label, figures, root in args.suite]
    payload = run(output_dir=args.output_dir, specs=specs)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
