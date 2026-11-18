#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


RESTORE_ACTIONS = {"host_restore", "gpu_replica"}
FINAL_DATA_PLANE = "physical_tensor_copy"


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _int(value: object) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_request_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    replay_rows = [row for row in rows if _bool(row.get("aegis_failover_replay"))]
    restore_rows = [
        row
        for row in replay_rows
        if str(row.get("aegis_replay_restore_action") or "").strip().lower()
        in RESTORE_ACTIONS
    ]
    missing_data_plane_rows = sum(
        1
        for row in restore_rows
        if not str(row.get("aegis_replay_restore_data_plane") or "").strip()
    )
    missing_timing_source_rows = sum(
        1
        for row in restore_rows
        if not str(row.get("aegis_replay_restore_timing_source") or "").strip()
    )
    data_planes = Counter(
        str(row.get("aegis_replay_restore_data_plane") or "legacy_missing")
        .strip()
        .lower()
        or "legacy_missing"
        for row in restore_rows
    )
    timing_sources = Counter(
        str(row.get("aegis_replay_restore_timing_source") or "legacy_missing")
        .strip()
        .lower()
        or "legacy_missing"
        for row in restore_rows
    )
    final_rows = data_planes.get(FINAL_DATA_PLANE, 0)
    nonfinal_rows = len(restore_rows) - final_rows
    schema_complete = (
        missing_data_plane_rows == 0
        and missing_timing_source_rows == 0
    )
    return {
        "request_csv": str(path),
        "run_dir": str(path.parent),
        "request_rows": len(rows),
        "replay_rows": len(replay_rows),
        "restore_rows": len(restore_rows),
        "physical_tensor_copy_rows": final_rows,
        "nonfinal_restore_rows": nonfinal_rows,
        "missing_data_plane_rows": missing_data_plane_rows,
        "missing_timing_source_rows": missing_timing_source_rows,
        "schema_complete": schema_complete,
        "data_planes": dict(sorted(data_planes.items())),
        "timing_sources": dict(sorted(timing_sources.items())),
        "final_ready": (
            nonfinal_rows == 0
            and len(restore_rows) > 0
            and schema_complete
        ),
    }


def audit_suite(suite_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_csvs = sorted(Path(suite_root).glob("**/request_details.csv"))
    rows = [audit_request_csv(path) for path in request_csvs]
    data_planes: Counter[str] = Counter()
    timing_sources: Counter[str] = Counter()
    for row in rows:
        data_planes.update(row["data_planes"])
        timing_sources.update(row["timing_sources"])
    totals = {
        "suite_root": str(suite_root),
        "request_csv_count": len(request_csvs),
        "request_rows": sum(int(row["request_rows"]) for row in rows),
        "replay_rows": sum(int(row["replay_rows"]) for row in rows),
        "restore_rows": sum(int(row["restore_rows"]) for row in rows),
        "physical_tensor_copy_rows": sum(
            int(row["physical_tensor_copy_rows"]) for row in rows
        ),
        "nonfinal_restore_rows": sum(int(row["nonfinal_restore_rows"]) for row in rows),
        "missing_data_plane_rows": sum(
            int(row["missing_data_plane_rows"]) for row in rows
        ),
        "missing_timing_source_rows": sum(
            int(row["missing_timing_source_rows"]) for row in rows
        ),
        "data_planes": dict(sorted(data_planes.items())),
        "timing_sources": dict(sorted(timing_sources.items())),
    }
    totals["final_ready"] = (
        totals["restore_rows"] > 0
        and totals["nonfinal_restore_rows"] == 0
        and totals["missing_data_plane_rows"] == 0
        and totals["missing_timing_source_rows"] == 0
    )
    totals["schema_complete"] = (
        totals["missing_data_plane_rows"] == 0
        and totals["missing_timing_source_rows"] == 0
    )
    return rows, totals


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_dir",
        "request_csv",
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["data_planes"] = json.dumps(row["data_planes"], sort_keys=True)
            out["timing_sources"] = json.dumps(row["timing_sources"], sort_keys=True)
            writer.writerow(out)


def render_markdown(rows: list[dict[str, Any]], totals: dict[str, Any]) -> str:
    lines = [
        "# Live Physical Movement Audit",
        "",
        f"- suite root: `{totals['suite_root']}`",
        f"- request CSVs: `{totals['request_csv_count']}`",
        f"- replay rows: `{totals['replay_rows']}`",
        f"- restore rows: `{totals['restore_rows']}`",
        f"- physical tensor copy rows: `{totals['physical_tensor_copy_rows']}`",
        f"- non-final restore rows: `{totals['nonfinal_restore_rows']}`",
        f"- missing data-plane rows: `{totals['missing_data_plane_rows']}`",
        f"- missing timing-source rows: `{totals['missing_timing_source_rows']}`",
        f"- schema-complete: `{str(totals['schema_complete']).lower()}`",
        f"- final-ready: `{str(totals['final_ready']).lower()}`",
        f"- data planes: `{json.dumps(totals['data_planes'], sort_keys=True)}`",
        f"- timing sources: `{json.dumps(totals['timing_sources'], sort_keys=True)}`",
        "",
        "## Per Run",
        "",
        "| Run | Replay rows | Restore rows | Physical tensor rows | Non-final rows | Missing fields | Data planes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {replay} | {restore} | {physical} | {nonfinal} | {missing} | `{planes}` |".format(
                run=Path(str(row["run_dir"])).name,
                replay=row["replay_rows"],
                restore=row["restore_rows"],
                physical=row["physical_tensor_copy_rows"],
                nonfinal=row["nonfinal_restore_rows"],
                missing=int(row["missing_data_plane_rows"])
                + int(row["missing_timing_source_rows"]),
                planes=json.dumps(row["data_planes"], sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Rows are final physical-movement evidence only when restore rows use ",
            "`aegis_replay_restore_data_plane=physical_tensor_copy` and every restore ",
            "row also carries a timing source. Metadata materialization with ",
            "calibrated sleep remains preliminary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether SGLSim failover replay rows use live physical KV movement."
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows, totals = audit_suite(args.suite_root)
    output_csv = args.output_csv or args.suite_root / "live_physical_movement_audit.csv"
    output_md = args.output_md or args.suite_root / "live_physical_movement_audit.md"
    write_csv(output_csv, rows)
    output_md.write_text(render_markdown(rows, totals), encoding="utf-8")
    print(
        json.dumps(
            {
                **totals,
                "output_csv": str(output_csv),
                "output_md": str(output_md),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    if args.strict and not bool(totals["final_ready"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
