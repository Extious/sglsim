#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


EXP_DIR = Path(__file__).resolve().parents[1]
DRAFT_ROOT = EXP_DIR.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIT_DIR = EXP_DIR / "results" / "strict_final_recovery_matrix_audit"
DEFAULT_CHECK_SCRIPT = SCRIPT_DIR / "check_bound_slurm_jobs.py"
DEFAULT_SYNC_SCRIPT = DEFAULT_AUDIT_DIR / "sync_bound_jobs.sh"
DEFAULT_STATUS_JSON = DEFAULT_AUDIT_DIR / "strict_final_bound_slurm_jobs.json"
DEFAULT_OUTPUT_JSON = DEFAULT_AUDIT_DIR / "strict_final_wait_and_sync_summary.json"
DEFAULT_OUTPUT_MD = DEFAULT_AUDIT_DIR / "strict_final_wait_and_sync_summary.md"
DEFAULT_RECOVERY_AUDIT_SCRIPT = SCRIPT_DIR / "audit_strict_final_recovery_matrix.py"
DEFAULT_PACKAGE_VALIDATOR = DRAFT_ROOT / "scripts" / "validate_paper_outline.py"
NONERROR_STATUSES = {
    "pending",
    "ready_to_sync",
    "running",
    "synced_and_audited",
    "waiting",
}


def run_capture(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_record(
    name: str,
    command: list[str],
    *,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_capture,
    cwd: Path | None = None,
) -> dict[str, Any]:
    proc = runner(command) if cwd is None else runner(command)
    return {
        "command": shlex.join(command),
        "name": name,
        "returncode": proc.returncode,
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-40:]),
    }


def status_snapshot(status_json: Path) -> dict[str, Any]:
    payload = read_json(status_json)
    counts = payload.get("status_counts") if isinstance(payload.get("status_counts"), dict) else {}
    return {
        "checked_at": payload.get("checked_at", ""),
        "completed": counts.get("completed", 0),
        "failed": counts.get("failed", 0),
        "overall_status": payload.get("overall_status", ""),
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "total_bound_jobs": payload.get("total_bound_jobs", 0),
    }


def run_wait_and_sync(
    *,
    check_script: Path = DEFAULT_CHECK_SCRIPT,
    sync_script: Path = DEFAULT_SYNC_SCRIPT,
    recovery_audit_script: Path = DEFAULT_RECOVERY_AUDIT_SCRIPT,
    package_validator: Path = DEFAULT_PACKAGE_VALIDATOR,
    status_json: Path = DEFAULT_STATUS_JSON,
    max_polls: int = 1,
    poll_interval_s: float = 60.0,
    execute: bool = False,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_capture,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    poll_records: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    max_polls = max(1, max_polls)
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summary: dict[str, Any] = {
        "checked_at": checked_at,
        "execute": execute,
        "max_polls": max_polls,
        "poll_interval_s": poll_interval_s,
        "polls": poll_records,
        "source": "wait_and_sync_bound_jobs",
        "status": "unknown",
        "sync_commands": commands,
    }

    for index in range(1, max_polls + 1):
        check_command = [sys.executable, str(check_script)]
        check_record = command_record("check_bound_slurm_jobs", check_command, runner=runner)
        snapshot = status_snapshot(status_json)
        poll_records.append(
            {
                "check_returncode": check_record["returncode"],
                "poll": index,
                **snapshot,
            }
        )
        if snapshot["overall_status"] == "complete":
            if not execute:
                summary["status"] = "ready_to_sync"
                summary["next_step"] = "Re-run with --execute to run sync_bound_jobs.sh and audits."
                return summary
            for name, command in [
                ("sync_bound_jobs", ["bash", str(sync_script)]),
                (
                    "audit_strict_final_recovery_matrix",
                    [sys.executable, str(recovery_audit_script)],
                ),
                (
                    "validate_paper_outline",
                    [sys.executable, str(package_validator)],
                ),
            ]:
                record = command_record(name, command, runner=runner)
                commands.append(record)
                if record["returncode"] != 0:
                    summary["status"] = f"{name}_failed"
                    summary["next_step"] = f"Inspect {name} output and rerun this wrapper."
                    return summary
            summary["status"] = "synced_and_audited"
            summary["next_step"] = "Review strict-final recovery matrix audit output."
            return summary
        if snapshot["overall_status"] == "failed":
            summary["status"] = "slurm_failed"
            summary["next_step"] = "Inspect failed Slurm logs before syncing."
            return summary
        if index < max_polls:
            sleeper(poll_interval_s)

    latest = poll_records[-1] if poll_records else {}
    summary["status"] = str(latest.get("overall_status") or "waiting")
    summary["next_step"] = (
        "Wait for bound Slurm jobs to reach complete, then rerun this wrapper "
        "with --execute."
    )
    return summary


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strict-Final Wait And Sync",
        "",
        "This wrapper polls bound Slurm jobs and optionally runs the guarded sync/audit path.",
        "It does not claim new recovery evidence until sync and audits succeed.",
        "",
        f"- Checked at: `{payload.get('checked_at', '')}`",
        f"- Execute: `{str(payload.get('execute')).lower()}`",
        f"- Status: `{payload.get('status', '')}`",
        f"- Next step: {payload.get('next_step', '')}",
        "",
        "## Polls",
        "",
        "| Poll | Checked at | Slurm status | Pending | Running | Completed | Failed |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("polls", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {poll} | {checked_at} | {status} | {pending} | {running} | {completed} | {failed} |".format(
                poll=row.get("poll", ""),
                checked_at=row.get("checked_at", ""),
                status=row.get("overall_status", ""),
                pending=row.get("pending", 0),
                running=row.get("running", 0),
                completed=row.get("completed", 0),
                failed=row.get("failed", 0),
            )
        )
    if payload.get("sync_commands"):
        lines.extend(["", "## Commands", ""])
        for command in payload.get("sync_commands", []):
            if isinstance(command, dict):
                lines.append(
                    f"- `{command.get('name', '')}` returncode={command.get('returncode')}"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Poll bound strict-final Slurm jobs and optionally sync when complete."
    )
    parser.add_argument("--check-script", type=Path, default=DEFAULT_CHECK_SCRIPT)
    parser.add_argument("--sync-script", type=Path, default=DEFAULT_SYNC_SCRIPT)
    parser.add_argument("--recovery-audit-script", type=Path, default=DEFAULT_RECOVERY_AUDIT_SCRIPT)
    parser.add_argument(
        "--package-validator",
        type=Path,
        default=DEFAULT_PACKAGE_VALIDATOR,
    )
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-polls", type=int, default=1)
    parser.add_argument("--poll-interval-s", type=float, default=60.0)
    parser.add_argument(
        "--max-wait-minutes",
        type=float,
        default=None,
        help=(
            "Convenience wall-clock wait budget. When set, max-polls is derived "
            "from this value and --poll-interval-s."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run sync_bound_jobs.sh and audits if the bound Slurm jobs are complete.",
    )
    args = parser.parse_args()

    max_polls = args.max_polls
    if args.max_wait_minutes is not None:
        if args.max_wait_minutes <= 0:
            parser.error("--max-wait-minutes must be positive")
        if args.poll_interval_s <= 0:
            parser.error("--poll-interval-s must be positive when --max-wait-minutes is set")
        max_polls = max(1, int(math.ceil(args.max_wait_minutes * 60.0 / args.poll_interval_s)))

    payload = run_wait_and_sync(
        check_script=args.check_script,
        sync_script=args.sync_script,
        recovery_audit_script=args.recovery_audit_script,
        package_validator=args.package_validator,
        status_json=args.status_json,
        max_polls=max_polls,
        poll_interval_s=args.poll_interval_s,
        execute=args.execute,
    )
    write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if str(payload["status"]) in NONERROR_STATUSES else 2


if __name__ == "__main__":
    raise SystemExit(main())
