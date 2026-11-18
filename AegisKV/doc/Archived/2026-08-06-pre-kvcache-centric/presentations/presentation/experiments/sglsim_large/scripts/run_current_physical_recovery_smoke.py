#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[6]
PYTHON_ROOT = ROOT / "python"
EXP_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "experiments"
SGLSIM_LARGE_DIR = EXP_DIR / "sglsim_large"
DEFAULT_RESULTS_ROOT = SGLSIM_LARGE_DIR / "results" / "agentic_workflows"
DEFAULT_PHYSICAL_COPY_CSV = (
    EXP_DIR
    / "sglang_small"
    / "results"
    / "physical_recovery_path_aggregate"
    / "physical_recovery_path_aggregate.csv"
)

for path in (ROOT, PYTHON_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark.AegisKV.draft.experiments.sglsim_large.scripts.audit_live_physical_movement import (  # noqa: E402
    audit_suite,
    render_markdown as render_audit_markdown,
    write_csv as write_audit_csv,
)
from benchmark.agentic_workflows.run import execute_scenario  # noqa: E402
from sglsim.srt.simulation.recovery_model import (  # noqa: E402
    ENV_PHYSICAL_COPY_CSV,
    ENV_RECOVERY_MODEL,
)


SCENARIO = "physical_copy_restore_smoke"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _counter_json(value: object) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return "{}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_summary_csvs(suite_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(suite_root.glob("**/agentic_workflow_summary.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _sum_float(rows: list[dict[str, str]], field: str) -> float:
    total = 0.0
    for row in rows:
        try:
            total += float(row.get(field) or 0.0)
        except ValueError:
            continue
    return total


def render_summary_markdown(payload: dict[str, Any]) -> str:
    audit = payload["audit_totals"]
    lines = [
        "# Current Physical-Recovery Smoke",
        "",
        "This run uses the current SGLSim online failover path with SGLang/torch "
        "physical recovery-path calibration enabled. It is a repeatable smoke for "
        "the recovery timing and request-detail instrumentation, not strict-final "
        "live KV tensor movement evidence.",
        "",
        "## Run",
        "",
        f"- run id: `{payload['run_id']}`",
        f"- scenario: `{payload['scenario']}`",
        f"- suite root: `{payload['suite_root']}`",
        f"- git sha: `{payload['git_sha']}`",
        f"- recovery model: `{payload['environment'][ENV_RECOVERY_MODEL]}`",
        f"- physical copy CSV: `{payload['environment'][ENV_PHYSICAL_COPY_CSV]}`",
        "",
        "## Result",
        "",
        f"- request CSVs: `{audit['request_csv_count']}`",
        f"- request rows: `{audit['request_rows']}`",
        f"- replay rows: `{audit['replay_rows']}`",
        f"- restore rows: `{audit['restore_rows']}`",
        f"- physical tensor copy rows: `{audit['physical_tensor_copy_rows']}`",
        f"- non-final restore rows: `{audit['nonfinal_restore_rows']}`",
        f"- final-ready: `{str(audit['final_ready']).lower()}`",
        f"- data planes: `{_counter_json(audit['data_planes'])}`",
        f"- timing sources: `{_counter_json(audit['timing_sources'])}`",
    ]

    summary = payload.get("agentic_summary", {})
    if summary:
        lines.extend(
            [
                f"- completed rows: `{summary.get('completed_rows', 0)}`",
                f"- failed rows: `{summary.get('failed_rows', 0)}`",
                f"- restored tokens: `{summary.get('restored_tokens', 0)}`",
                f"- mean restore sleep: `{summary.get('mean_restore_sleep_ms', 0.0):.3f} ms`",
            ]
        )

    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This smoke is expected to remain non-final until the online retry data "
            "plane reports `aegis_replay_restore_data_plane=physical_tensor_copy`. "
            "Rows with `metadata_materialization` plus `physical_recovery_path_csv` "
            "timing are calibrated SGLSim evidence, not live tensor-copy evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _agentic_summary(suite_root: Path) -> dict[str, Any]:
    rows = _read_summary_csvs(suite_root)
    replay_rows = _sum_float(rows, "replay_rows")
    weighted_restore_sleep_ms = 0.0
    for row in rows:
        try:
            weighted_restore_sleep_ms += float(row.get("mean_restore_sleep_ms") or 0.0) * float(
                row.get("replay_rows") or 0.0
            )
        except ValueError:
            continue
    mean_restore_sleep_ms = (
        weighted_restore_sleep_ms / replay_rows if replay_rows > 0.0 else 0.0
    )
    return {
        "summary_rows": len(rows),
        "completed_rows": int(_sum_float(rows, "completed")),
        "failed_rows": int(_sum_float(rows, "failed")),
        "failover_replay_rows": int(replay_rows),
        "restored_tokens": int(_sum_float(rows, "restored_tokens")),
        "mean_restore_sleep_ms": mean_restore_sleep_ms,
    }


def write_outputs(
    *,
    run_id: str,
    suite_root: Path,
    manifest: dict[str, Any],
    physical_copy_csv: Path,
) -> dict[str, Any]:
    rows, totals = audit_suite(suite_root)
    audit_csv = suite_root / "live_physical_movement_audit.csv"
    audit_md = suite_root / "live_physical_movement_audit.md"
    write_audit_csv(audit_csv, rows)
    audit_md.write_text(render_audit_markdown(rows, totals), encoding="utf-8")

    payload = {
        "run_id": run_id,
        "scenario": SCENARIO,
        "suite_root": str(suite_root),
        "manifest": manifest,
        "audit_csv": str(audit_csv),
        "audit_md": str(audit_md),
        "audit_totals": totals,
        "agentic_summary": _agentic_summary(suite_root),
        "environment": {
            ENV_RECOVERY_MODEL: "physical_copy",
            ENV_PHYSICAL_COPY_CSV: str(physical_copy_csv.resolve()),
        },
        "git_sha": git_sha(),
        "evidence_tier": "sglsim-current-physical-recovery-smoke-nonfinal",
    }
    summary_md = suite_root / "current_physical_recovery_smoke_summary.md"
    payload["summary_md"] = str(summary_md)
    _write_json(suite_root / "current_physical_recovery_smoke_summary.json", payload)
    summary_md.write_text(render_summary_markdown(payload), encoding="utf-8")
    return payload


def run_smoke(
    *,
    run_id: str,
    results_root: Path,
    physical_copy_csv: Path,
    runs: int,
    log_level: str,
    dry_run: bool,
) -> dict[str, Any]:
    if not physical_copy_csv.exists():
        raise FileNotFoundError(f"missing physical copy calibration CSV: {physical_copy_csv}")

    old_model = os.environ.get(ENV_RECOVERY_MODEL)
    old_csv = os.environ.get(ENV_PHYSICAL_COPY_CSV)
    os.environ[ENV_RECOVERY_MODEL] = "physical_copy"
    os.environ[ENV_PHYSICAL_COPY_CSV] = str(physical_copy_csv.resolve())

    try:
        manifest = execute_scenario(
            scenario_name=SCENARIO,
            output_root=results_root,
            run_id=run_id,
            runs=runs,
            log_level=log_level,
            dry_run=dry_run,
        )
        suite_root = Path(str(manifest["output_root"]))
        return write_outputs(
            run_id=run_id,
            suite_root=suite_root,
            manifest=manifest,
            physical_copy_csv=physical_copy_csv,
        )
    finally:
        if old_model is None:
            os.environ.pop(ENV_RECOVERY_MODEL, None)
        else:
            os.environ[ENV_RECOVERY_MODEL] = old_model
        if old_csv is None:
            os.environ.pop(ENV_PHYSICAL_COPY_CSV, None)
        else:
            os.environ[ENV_PHYSICAL_COPY_CSV] = old_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current SGLSim physical-copy-calibrated recovery smoke and "
            "write the live physical-movement audit."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--run-id",
        default="sglsim-current-physical-recovery-smoke-"
        + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--physical-copy-csv", type=Path, default=DEFAULT_PHYSICAL_COPY_CSV)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Only rewrite the audit and summary files for an existing run id.",
    )
    args = parser.parse_args()

    if args.refresh_existing:
        suite_root = Path(args.results_root).resolve() / args.run_id
        manifest_path = suite_root / "agentic_workflow_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"output_root": str(suite_root), "run_id": args.run_id}
        )
        payload = write_outputs(
            run_id=args.run_id,
            suite_root=suite_root,
            manifest=manifest,
            physical_copy_csv=args.physical_copy_csv,
        )
    else:
        payload = run_smoke(
            run_id=args.run_id,
            results_root=args.results_root,
            physical_copy_csv=args.physical_copy_csv,
            runs=args.runs,
            log_level=args.log_level,
            dry_run=bool(args.dry_run),
        )
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
