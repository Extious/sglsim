"""Run-completeness checks shared by parity diagnostics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunIntegrity:
    row_count: int
    completed_rows: int
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    return int(float(raw))


def check_run_integrity(run_dir: Path, ref_dir: Path | None = None) -> RunIntegrity:
    csv_path = run_dir / "request_details.csv"
    metrics_path = run_dir / "metrics.json"
    failures: list[str] = []
    rows = _load_rows(csv_path)
    if not rows:
        failures.append("missing_or_empty_request_details")
        return RunIntegrity(row_count=0, completed_rows=0, failures=tuple(failures))

    bad_status = [
        row.get("rid", f"row{i}")
        for i, row in enumerate(rows)
        if row.get("status") not in (None, "", "completed")
    ]
    if bad_status:
        failures.append(
            "non_completed_status="
            + ",".join(str(rid) for rid in bad_status[:8])
        )

    short_output: list[str] = []
    for i, row in enumerate(rows):
        generated = _as_int(row.get("generated_tokens"))
        if generated is None:
            continue
        target = _as_int(row.get("target_output_tokens"))
        if target is None:
            target = _as_int(row.get("output_length"))
        if target is not None and generated < target:
            short_output.append(
                f"{row.get('rid', f'row{i}')}:{generated}/{target}"
            )
    if short_output:
        failures.append("short_generated_tokens=" + ",".join(short_output[:8]))

    completed_rows = sum(1 for row in rows if row.get("status") == "completed")
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_completed = _as_int(str(metrics.get("completed", "")))
        if metrics_completed is not None and metrics_completed != completed_rows:
            failures.append(
                f"metrics_completed_mismatch={metrics_completed}/{completed_rows}"
            )
        metrics_failed = _as_int(str(metrics.get("failed", "")))
        if metrics_failed is not None and metrics_failed != 0:
            failures.append(f"metrics_failed={metrics_failed}")
    else:
        failures.append("missing_metrics")

    if ref_dir is not None:
        ref_rows = _load_rows(ref_dir / "request_details.csv")
        if ref_rows and len(rows) != len(ref_rows):
            failures.append(f"row_count_mismatch={len(rows)}/{len(ref_rows)}")

    return RunIntegrity(
        row_count=len(rows),
        completed_rows=completed_rows,
        failures=tuple(failures),
    )
