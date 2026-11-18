#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIT_DIR = EXP_DIR / "results" / "strict_final_recovery_matrix_audit"
DEFAULT_BIND_SCRIPT = SCRIPT_DIR / "bind_strict_final_job_ids.py"
DEFAULT_MANIFEST = DEFAULT_AUDIT_DIR / "strict_final_recovery_submission_manifest.json"
DEFAULT_PREFLIGHT_SCRIPT = SCRIPT_DIR / "check_strict_final_remote_reachability.py"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_submit_script(output_dir: Path, method: str) -> Path:
    if method == "alias":
        return output_dir / "submit_missing_alias.sh"
    if method == "direct-proxy":
        return output_dir / "submit_missing_direct_proxy.sh"
    raise ValueError(f"Unknown submit method: {method}")


def run_capture(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strict-Final Submit And Bind Summary",
        "",
        "This file records one wrapper invocation for missing strict-final jobs.",
        "It is operational metadata and does not claim new recovery evidence.",
        "",
        f"- Executed: `{payload['executed']}`",
        f"- Method: `{payload['method']}`",
        f"- Submit script: `{payload['submit_script']}`",
        f"- Submit log: `{payload['submit_log'] or ''}`",
        f"- Preflight return code: `{payload['preflight_returncode']}`",
        f"- Submit return code: `{payload['submit_returncode']}`",
        f"- Bind return code: `{payload['bind_returncode']}`",
        "",
        "## Commands",
        "",
        f"- Submit: `{payload['submit_command']}`",
        f"- Bind: `{payload['bind_command']}`",
        "",
    ]
    if not payload["executed"]:
        lines.extend(
            [
                "## Boundary",
                "",
                "This was a dry run. Add `--execute` to submit Slurm jobs.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_submit_and_bind(
    *,
    method: str,
    output_dir: Path,
    submit_script: Path,
    bind_script: Path,
    manifest_json: Path,
    preflight_script: Path,
    execute: bool,
    skip_preflight: bool,
    run_stamp: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    submit_command = ["bash", str(submit_script)]
    submit_log = output_dir / f"submit_missing_{method}_{run_stamp}.log"
    bind_command = [
        sys.executable,
        str(bind_script),
        "--submit-log",
        str(submit_log),
        "--manifest-json",
        str(manifest_json),
        "--output-dir",
        str(output_dir),
    ]
    preflight_command = [
        sys.executable,
        str(preflight_script),
        "--manifest-json",
        str(manifest_json),
        "--output-json",
        str(output_dir / "strict_final_remote_reachability.json"),
        "--output-md",
        str(output_dir / "strict_final_remote_reachability.md"),
    ]
    preflight_log = output_dir / f"preflight_missing_{method}_{run_stamp}.log"
    summary: dict[str, Any] = {
        "bind_command": shlex.join(bind_command),
        "bind_returncode": None,
        "executed": execute,
        "method": method,
        "manifest_json": str(manifest_json),
        "output_dir": str(output_dir),
        "preflight_command": shlex.join(preflight_command),
        "preflight_log": str(preflight_log) if execute and not skip_preflight else "",
        "preflight_returncode": None,
        "preflight_skipped": skip_preflight,
        "run_stamp": run_stamp,
        "source": "submit_and_bind_strict_final_missing_jobs",
        "submit_command": shlex.join(submit_command),
        "submit_log": str(submit_log) if execute else "",
        "submit_returncode": None,
        "submit_script": str(submit_script),
    }
    if not execute:
        return summary

    if not skip_preflight:
        preflight_proc = run_capture(preflight_command, cwd=output_dir)
        preflight_log.write_text(preflight_proc.stdout, encoding="utf-8")
        summary["preflight_returncode"] = preflight_proc.returncode
        if preflight_proc.returncode != 0:
            summary["status"] = "preflight_failed"
            return summary

    submit_proc = run_capture(submit_command, cwd=output_dir)
    submit_log.write_text(submit_proc.stdout, encoding="utf-8")
    summary["submit_returncode"] = submit_proc.returncode

    bind_proc = run_capture(bind_command, cwd=output_dir)
    summary["bind_returncode"] = bind_proc.returncode
    (output_dir / f"bind_missing_{method}_{run_stamp}.log").write_text(
        bind_proc.stdout,
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Submit missing strict-final Slurm jobs, save the submit log, "
            "and bind returned job ids. Dry-run by default."
        )
    )
    parser.add_argument("--method", choices=["direct-proxy", "alias"], default="direct-proxy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--submit-script", type=Path, default=None)
    parser.add_argument("--bind-script", type=Path, default=DEFAULT_BIND_SCRIPT)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight-script", type=Path, default=DEFAULT_PREFLIGHT_SCRIPT)
    parser.add_argument("--run-stamp", default="")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the submit script. Without this flag the command is only rendered.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip read-only SSH reachability checks before --execute.",
    )
    args = parser.parse_args()

    run_stamp = args.run_stamp or timestamp()
    submit_script = args.submit_script or default_submit_script(args.output_dir, args.method)
    summary = run_submit_and_bind(
        method=args.method,
        output_dir=args.output_dir,
        submit_script=submit_script,
        bind_script=args.bind_script,
        manifest_json=args.manifest_json,
        preflight_script=args.preflight_script,
        execute=args.execute,
        skip_preflight=args.skip_preflight,
        run_stamp=run_stamp,
    )
    write_json(args.output_dir / "strict_final_submit_and_bind_summary.json", summary)
    write_markdown(args.output_dir / "strict_final_submit_and_bind_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.execute:
        return 0
    return int(
        summary["preflight_returncode"]
        or summary["submit_returncode"]
        or summary["bind_returncode"]
        or 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
