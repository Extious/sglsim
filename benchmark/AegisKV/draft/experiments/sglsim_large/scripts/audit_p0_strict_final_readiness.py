#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from benchmark.AegisKV.draft.experiments.sglsim_large.scripts.audit_live_physical_movement import audit_suite  # noqa: E402


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


def _suite_rows(
    *,
    label: str,
    figures: str,
    suite_root: Path,
) -> list[dict[str, Any]]:
    rows, totals = audit_suite(suite_root)
    data_planes = totals.get("data_planes", {})
    timing_sources = totals.get("timing_sources", {})
    if isinstance(data_planes, dict) and "legacy_missing" in data_planes:
        data_planes = {
            (
                "legacy_missing_nonfinal"
                if str(key) == "legacy_missing"
                else str(key)
            ): value
            for key, value in data_planes.items()
        }
    if isinstance(timing_sources, dict) and "legacy_missing" in timing_sources:
        timing_sources = {
            (
                "legacy_missing_nonfinal"
                if str(key) == "legacy_missing"
                else str(key)
            ): value
            for key, value in timing_sources.items()
        }
    return [
        {
            "suite_label": label,
            "figures": figures,
            "suite_root": str(suite_root),
            "request_csv_count": totals["request_csv_count"],
            "request_rows": totals["request_rows"],
            "replay_rows": totals["replay_rows"],
            "restore_rows": totals["restore_rows"],
            "physical_tensor_copy_rows": totals["physical_tensor_copy_rows"],
            "nonfinal_restore_rows": totals["nonfinal_restore_rows"],
            "missing_data_plane_rows": totals["missing_data_plane_rows"],
            "missing_timing_source_rows": totals["missing_timing_source_rows"],
            "schema_complete": bool(totals["schema_complete"]),
            "final_ready": bool(totals["final_ready"]),
            "data_planes": json.dumps(data_planes, sort_keys=True),
            "timing_sources": json.dumps(timing_sources, sort_keys=True),
            "next_required_evidence": (
                "replace metadata_materialization restore rows with "
                "physical_tensor_copy rows from the live retry data plane"
                if not bool(totals["final_ready"])
                else "strict-final live physical movement gate passes"
            ),
        }
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite_label",
        "figures",
        "suite_root",
        "request_csv_count",
        "request_rows",
        "replay_rows",
        "restore_rows",
        "physical_tensor_copy_rows",
        "nonfinal_restore_rows",
        "missing_data_plane_rows",
        "missing_timing_source_rows",
        "schema_complete",
        "final_ready",
        "data_planes",
        "timing_sources",
        "next_required_evidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    total_restore = sum(int(row["restore_rows"]) for row in rows)
    total_physical = sum(int(row["physical_tensor_copy_rows"]) for row in rows)
    total_nonfinal = sum(int(row["nonfinal_restore_rows"]) for row in rows)
    total_missing_data_plane = sum(int(row["missing_data_plane_rows"]) for row in rows)
    total_missing_timing_source = sum(
        int(row["missing_timing_source_rows"]) for row in rows
    )
    final_ready = (
        total_restore > 0
        and total_nonfinal == 0
        and total_missing_data_plane == 0
        and total_missing_timing_source == 0
    )
    lines = [
        "# P0 Strict-Final Readiness Audit",
        "",
        "This audit covers the current P0 online SGLSim suites used by Fig. 8/9/10/14.",
        "A row is strict-final only when restore actions are reported as ",
        "`aegis_replay_restore_data_plane=physical_tensor_copy`.",
        "",
        "## Summary",
        "",
        f"- restore rows: `{total_restore}`",
        f"- physical tensor copy rows: `{total_physical}`",
        f"- non-final restore rows: `{total_nonfinal}`",
        f"- missing data-plane rows: `{total_missing_data_plane}`",
        f"- missing timing-source rows: `{total_missing_timing_source}`",
        f"- strict-final-ready: `{str(final_ready).lower()}`",
        "",
        "## Suites",
        "",
        "| Suite | Figures | Request CSVs | Replay rows | Restore rows | Physical tensor rows | Non-final rows | Missing fields | Data planes | Timing sources |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {suite} | {figures} | {csvs} | {replay} | {restore} | {physical} | {nonfinal} | {missing} | `{planes}` | `{timing}` |".format(
                suite=row["suite_label"],
                figures=row["figures"],
                csvs=row["request_csv_count"],
                replay=row["replay_rows"],
                restore=row["restore_rows"],
                physical=row["physical_tensor_copy_rows"],
                nonfinal=row["nonfinal_restore_rows"],
                missing=int(row["missing_data_plane_rows"])
                + int(row["missing_timing_source_rows"]),
                planes=row["data_planes"],
                timing=row["timing_sources"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current P0 suites already exercise the online failover/retry policy path ",
            "with recovery timing calibrated by SGLang/torch physical copy measurements. ",
            "They are still not strict-final because the online retry path materializes ",
            "metadata rather than executing physical KV tensor movement.",
            "",
            "Next required step: connect the live retry data plane to physical Host H2D ",
            "and peer GPUReplica copy, then rerun this audit until all restore rows are ",
            "`physical_tensor_copy` and `strict-final-ready=true`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit current P0 AegisKV suites against the strict-final physical-movement gate."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--suite",
        action="append",
        nargs=3,
        metavar=("LABEL", "FIGURES", "SUITE_ROOT"),
        help="Override or append suite as label figures suite_root.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    specs = []
    if args.suite:
        specs = [
            (label, figures, Path(root))
            for label, figures, root in args.suite
        ]
    else:
        specs = [
            (label, figures, RESULTS_ROOT / root)
            for label, figures, root in DEFAULT_SUITES
        ]

    rows: list[dict[str, Any]] = []
    for label, figures, suite_root in specs:
        if not Path(suite_root).exists():
            raise FileNotFoundError(f"missing suite root: {suite_root}")
        rows.extend(
            _suite_rows(
                label=label,
                figures=figures,
                suite_root=Path(suite_root),
            )
        )

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "p0_strict_final_readiness.csv"
    md_path = output_dir / "p0_strict_final_readiness.md"
    write_csv(csv_path, rows)
    md_path.write_text(render_markdown(rows), encoding="utf-8")

    final_ready = all(bool(row["final_ready"]) for row in rows)
    print(
        json.dumps(
            {
                "output_csv": str(csv_path),
                "output_md": str(md_path),
                "suites": len(rows),
                "strict_final_ready": final_ready,
                "restore_rows": sum(int(row["restore_rows"]) for row in rows),
                "physical_tensor_copy_rows": sum(
                    int(row["physical_tensor_copy_rows"]) for row in rows
                ),
                "nonfinal_restore_rows": sum(
                    int(row["nonfinal_restore_rows"]) for row in rows
                ),
                "missing_data_plane_rows": sum(
                    int(row["missing_data_plane_rows"]) for row in rows
                ),
                "missing_timing_source_rows": sum(
                    int(row["missing_timing_source_rows"]) for row in rows
                ),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    if args.strict and not final_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
