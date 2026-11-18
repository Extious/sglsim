#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = EXP_DIR / "results" / "strict_final_recovery_matrix_audit"
DEFAULT_BINDINGS = DEFAULT_AUDIT_DIR / "strict_final_recovery_job_bindings.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_AUDIT_DIR / "strict_final_bound_slurm_jobs.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_AUDIT_DIR / "strict_final_bound_slurm_jobs.json"
DEFAULT_OUTPUT_MD = DEFAULT_AUDIT_DIR / "strict_final_bound_slurm_jobs.md"

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bound_rows(path: Path) -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(path)
        if str(row.get("bound") or "").lower() == "true"
        and str(row.get("slurm_job_id") or "")
    ]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def remote_status_command(job_ids: list[str]) -> str:
    ids = ",".join(job_ids)
    squeue = (
        f"squeue -h -j {shlex.quote(ids)} "
        "-o '%i|%T|%M|%l|%D|%R'"
    )
    sacct = (
        f"sacct -n -P -j {shlex.quote(ids)} "
        "--format=JobID,JobName%40,State,ExitCode,Elapsed,Start,End 2>/dev/null "
        "|| true"
    )
    scontrol = " ".join(
        [
            "echo __SCONTROL__;",
            "for j in",
            " ".join(shlex.quote(job_id) for job_id in job_ids),
            "; do",
            "echo __JOB__$j;",
            "scontrol show job $j || true;",
            "done",
        ]
    )
    return f"{squeue}; echo __SACCT__; {sacct}; {scontrol}"


def parse_squeue(text: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 5)
        if len(parts) != 6:
            continue
        job_id, state, elapsed, time_limit, nodes, reason = parts
        out[job_id] = {
            "squeue_elapsed": elapsed,
            "squeue_nodes": nodes,
            "squeue_reason": reason,
            "squeue_state": state,
            "squeue_time_limit": time_limit,
        }
    return out


def parse_sacct(text: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        job_id, job_name, state, exit_code, elapsed, start, end = parts[:7]
        out[job_id] = {
            "sacct_elapsed": elapsed,
            "sacct_end": end,
            "sacct_exit_code": exit_code,
            "sacct_job_name": job_name,
            "sacct_start": start,
            "sacct_state": state,
        }
    return out


def parse_scontrol(text: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    current_job = ""
    current_tokens: list[str] = []

    def flush() -> None:
        if not current_job:
            return
        row: dict[str, str] = {}
        for token in current_tokens:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key in {
                "AllocTRES",
                "Command",
                "JobName",
                "JobState",
                "Partition",
                "Reason",
                "ReqTRES",
                "StartTime",
                "SubmitTime",
                "TimeLimit",
                "WorkDir",
            }:
                row[f"scontrol_{key}"] = value
        if row:
            out[current_job] = row

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("__JOB__"):
            flush()
            current_job = line[len("__JOB__") :].strip()
            current_tokens = []
            continue
        if current_job:
            current_tokens.extend(line.split())
    flush()
    return out


def split_remote_output(text: str) -> tuple[str, str, str]:
    sacct_marker = "__SACCT__"
    scontrol_marker = "__SCONTROL__"
    if sacct_marker not in text:
        return text, "", ""
    squeue_text, rest = text.split(sacct_marker, 1)
    if scontrol_marker not in rest:
        return squeue_text.strip(), rest.strip(), ""
    sacct_text, scontrol_text = rest.split(scontrol_marker, 1)
    return squeue_text.strip(), sacct_text.strip(), scontrol_text.strip()


def is_terminal_state(state: str) -> bool:
    return state.split()[0].strip().upper() in TERMINAL_STATES if state else False


def is_success(state: str, exit_code: str) -> bool:
    return state.strip().upper() == "COMPLETED" and exit_code.strip() == "0:0"


def merge_status(
    bindings: list[dict[str, str]],
    *,
    squeue_rows: dict[str, dict[str, str]],
    sacct_rows: dict[str, dict[str, str]],
    scontrol_rows: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    scontrol_rows = scontrol_rows or {}
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        job_id = str(binding.get("slurm_job_id") or "")
        squeue_row = squeue_rows.get(job_id, {})
        sacct_row = sacct_rows.get(job_id, {})
        scontrol_row = scontrol_rows.get(job_id, {})
        state = str(
            squeue_row.get("squeue_state")
            or sacct_row.get("sacct_state")
            or scontrol_row.get("scontrol_JobState")
            or ""
        )
        exit_code = str(sacct_row.get("sacct_exit_code") or "")
        terminal = is_terminal_state(state)
        rows.append(
            {
                "pair_id": binding.get("pair_id", ""),
                "role": binding.get("role", ""),
                "run_label": binding.get("run_label", ""),
                "slurm_job_id": job_id,
                **squeue_row,
                **sacct_row,
                **scontrol_row,
                "effective_state": state,
                "terminal": str(terminal).lower(),
                "successful": str(is_success(state, exit_code)).lower(),
            }
        )
    return rows


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "completed": 0,
        "failed": 0,
        "pending": 0,
        "running": 0,
        "terminal": 0,
        "unknown": 0,
    }
    for row in rows:
        state = str(row.get("effective_state") or "").upper()
        if not state:
            counts["unknown"] += 1
        elif state == "PENDING":
            counts["pending"] += 1
        elif state in {"RUNNING", "COMPLETING", "CONFIGURING"}:
            counts["running"] += 1
        elif is_success(state, str(row.get("sacct_exit_code") or "")):
            counts["completed"] += 1
            counts["terminal"] += 1
        elif is_terminal_state(state):
            counts["failed"] += 1
            counts["terminal"] += 1
        else:
            counts["unknown"] += 1
    return counts


def overall_status(rows: list[dict[str, Any]], total_bindings: int) -> str:
    if total_bindings == 0:
        return "unbound"
    counts = status_counts(rows)
    if counts["failed"] > 0:
        return "failed"
    if counts["completed"] == total_bindings:
        return "complete"
    if counts["running"] > 0:
        return "running"
    if counts["pending"] > 0:
        return "pending"
    return "unknown"


def collect_status(
    *,
    bindings_csv: Path,
    remote_host: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_command,
) -> dict[str, Any]:
    bindings = bound_rows(bindings_csv)
    job_ids = [str(row["slurm_job_id"]) for row in bindings]
    if not job_ids:
        rows: list[dict[str, Any]] = []
        return {
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": "",
            "overall_status": "unbound",
            "rows": rows,
            "source": "check_bound_slurm_jobs",
            "status_counts": status_counts(rows),
            "total_bound_jobs": 0,
        }
    remote_command = remote_status_command(job_ids)
    command = ["ssh", remote_host, remote_command]
    proc = runner(command)
    squeue_text, sacct_text, scontrol_text = split_remote_output(proc.stdout or "")
    rows = merge_status(
        bindings,
        squeue_rows=parse_squeue(squeue_text),
        sacct_rows=parse_sacct(sacct_text),
        scontrol_rows=parse_scontrol(scontrol_text),
    )
    return {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": shlex.join(command),
        "overall_status": overall_status(rows, len(bindings)),
        "remote_returncode": proc.returncode,
        "rows": rows,
        "source": "check_bound_slurm_jobs",
        "status_counts": status_counts(rows),
        "total_bound_jobs": len(bindings),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    counts = payload.get("status_counts", {})
    lines = [
        "# Strict-Final Bound Slurm Jobs",
        "",
        "This report tracks Slurm state for already-bound strict-final recovery jobs.",
        "It does not sync results or change recovery-matrix evidence.",
        "",
        f"- Checked at: `{payload.get('checked_at', '')}`",
        f"- Overall status: `{payload.get('overall_status', '')}`",
        f"- Bound jobs: `{payload.get('total_bound_jobs', 0)}`",
        (
            "- Counts: "
            f"pending={counts.get('pending', 0)}, "
            f"running={counts.get('running', 0)}, "
            f"completed={counts.get('completed', 0)}, "
            f"failed={counts.get('failed', 0)}, "
            f"unknown={counts.get('unknown', 0)}"
        ),
        "",
        "| Pair | Role | Job | State | Reason | Start | TRES | Exit | Elapsed |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {pair} | {role} | {job} | {state} | {reason} | {start} | {tres} | {exit_code} | {elapsed} |".format(
                pair=row.get("pair_id", ""),
                role=row.get("role", ""),
                job=row.get("slurm_job_id", ""),
                state=row.get("effective_state", ""),
                reason=row.get("squeue_reason", ""),
                start=row.get("scontrol_StartTime", ""),
                tres=row.get("scontrol_ReqTRES", ""),
                exit_code=row.get("sacct_exit_code", ""),
                elapsed=row.get("squeue_elapsed") or row.get("sacct_elapsed", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Slurm status for bound strict-final jobs."
    )
    parser.add_argument("--bindings-csv", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--remote-host", default="hkbugpumanode")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    payload = collect_status(
        bindings_csv=args.bindings_csv,
        remote_host=args.remote_host,
    )
    write_json(args.output_json, payload)
    write_csv(args.output_csv, payload.get("rows", []))
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["overall_status"] in {"complete", "pending", "running"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
