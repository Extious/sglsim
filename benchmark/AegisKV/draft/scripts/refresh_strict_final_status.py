#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


DRAFT_ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = DRAFT_ROOT.parent
REPO_ROOT = DRAFT_ROOT.parents[2]
EXPERIMENTS = DRAFT_ROOT / "experiments"
SGLANG_SMALL_SCRIPTS = EXPERIMENTS / "sglang_small" / "scripts"
OUT_JSON = EXPERIMENTS / "strict_final_status_refresh.json"
OUT_MD = EXPERIMENTS / "strict_final_status_refresh.md"

DEFAULT_WAIT_SCRIPT = SGLANG_SMALL_SCRIPTS / "wait_and_sync_bound_jobs.py"
DEFAULT_DATAPLANE_REPORT_SCRIPT = (
    DRAFT_ROOT / "scripts" / "report_strict_final_dataplane_readiness.py"
)
DEFAULT_P0_BRIDGE_SCRIPT = (
    EXPERIMENTS
    / "sglsim_large"
    / "scripts"
    / "build_p0_physical_bridge.py"
)
DEFAULT_P0_READINESS_SCRIPT = (
    EXPERIMENTS
    / "sglsim_large"
    / "scripts"
    / "audit_p0_strict_final_readiness.py"
)
DEFAULT_P0_LIVE_MOVEMENT_AUDITS_SCRIPT = (
    EXPERIMENTS
    / "sglsim_large"
    / "scripts"
    / "write_p0_live_movement_audits.py"
)
DEFAULT_P0_CANDIDATE_VALIDATOR_SCRIPT = (
    EXPERIMENTS
    / "sglsim_large"
    / "scripts"
    / "validate_p0_physical_tensor_copy_candidates.py"
)
DEFAULT_P0_CANDIDATE_BUILDER_SCRIPT = (
    EXPERIMENTS
    / "sglsim_large"
    / "scripts"
    / "build_p0_physical_tensor_copy_candidates.py"
)
DEFAULT_P0_EVENT_TEMPLATE_SCRIPT = (
    EXPERIMENTS
    / "sglsim_large"
    / "scripts"
    / "write_p0_physical_tensor_copy_event_template.py"
)
DEFAULT_SYNC_TEXT_SCRIPT = DRAFT_ROOT / "scripts" / "sync_evidence_status_text.py"
DEFAULT_BUILD_DRAFT_SCRIPT = DRAFT_ROOT / "scripts" / "build_full_draft.py"
DEFAULT_GOAL_AUDIT_SCRIPT = DRAFT_ROOT / "scripts" / "audit_goal_completion.py"
DEFAULT_VALIDATE_DRAFT_SCRIPT = DRAFT_ROOT / "scripts" / "validate_draft_integrity.py"
DEFAULT_BEAMER_DIR = AEGIS_ROOT / "beamer"


def run_capture(command: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def stdout_tail(text: str, *, max_lines: int = 60, max_chars: int = 6000) -> str:
    tail = "\n".join(text.splitlines()[-max_lines:])
    if len(tail) <= max_chars:
        return tail
    return "...\n" + tail[-max_chars:]


def command_record(
    name: str,
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    runner: Callable[..., subprocess.CompletedProcess[str]] = run_capture,
) -> dict[str, Any]:
    proc = runner(command, cwd=cwd)
    return {
        "command": shlex.join(command),
        "cwd": str(cwd),
        "name": name,
        "returncode": proc.returncode,
        "stdout_tail": stdout_tail(proc.stdout or ""),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strict-Final Status Refresh",
        "",
        "This wrapper refreshes current Slurm status, synchronizes evidence-status text,",
        "regenerates draft/audit artifacts, and optionally rebuilds the beamer deck.",
        "It does not bypass the complete-only sync guard.",
        "",
        f"- Checked at: `{payload.get('checked_at', '')}`",
        f"- Execute sync path: `{str(payload.get('execute')).lower()}`",
        f"- Build beamer: `{str(payload.get('build_beamer')).lower()}`",
        f"- Status: `{payload.get('status', '')}`",
        f"- Next step: {payload.get('next_step', '')}",
        "",
        "## Commands",
        "",
        "| Step | Return code |",
        "|---|---:|",
    ]
    for command in payload.get("commands", []):
        if isinstance(command, dict):
            lines.append(f"| `{command.get('name', '')}` | {command.get('returncode', '')} |")
    lines.append("")
    return "\n".join(lines)


def write_summary_artifacts(
    payload: dict[str, Any],
    *,
    output_json: Path | None,
    output_md: Path | None,
) -> None:
    if output_json is not None:
        write_json(output_json, payload)
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")


def run_refresh(
    *,
    wait_script: Path = DEFAULT_WAIT_SCRIPT,
    p0_readiness_script: Path = DEFAULT_P0_READINESS_SCRIPT,
    p0_live_movement_audits_script: Path = DEFAULT_P0_LIVE_MOVEMENT_AUDITS_SCRIPT,
    p0_bridge_script: Path = DEFAULT_P0_BRIDGE_SCRIPT,
    p0_candidate_builder_script: Path = DEFAULT_P0_CANDIDATE_BUILDER_SCRIPT,
    p0_candidate_validator_script: Path = DEFAULT_P0_CANDIDATE_VALIDATOR_SCRIPT,
    p0_event_template_script: Path = DEFAULT_P0_EVENT_TEMPLATE_SCRIPT,
    dataplane_report_script: Path = DEFAULT_DATAPLANE_REPORT_SCRIPT,
    sync_text_script: Path = DEFAULT_SYNC_TEXT_SCRIPT,
    build_draft_script: Path = DEFAULT_BUILD_DRAFT_SCRIPT,
    goal_audit_script: Path = DEFAULT_GOAL_AUDIT_SCRIPT,
    validate_draft_script: Path = DEFAULT_VALIDATE_DRAFT_SCRIPT,
    beamer_dir: Path = DEFAULT_BEAMER_DIR,
    max_wait_minutes: float = 0.001,
    poll_interval_s: float = 60.0,
    execute: bool = False,
    build_beamer: bool = False,
    validate_draft: bool = True,
    p0_candidate_event_csvs: list[Path] | None = None,
    p0_candidate_event_jsonls: list[Path] | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = run_capture,
) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "build_beamer": build_beamer,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commands": commands,
        "execute": execute,
        "max_wait_minutes": max_wait_minutes,
        "poll_interval_s": poll_interval_s,
        "source": "refresh_strict_final_status",
        "status": "unknown",
        "validate_draft": validate_draft,
    }

    wait_command = [
        sys.executable,
        str(wait_script),
        "--max-wait-minutes",
        str(max_wait_minutes),
        "--poll-interval-s",
        str(poll_interval_s),
    ]
    if execute:
        wait_command.append("--execute")
    commands.append(command_record("wait_and_sync_bound_jobs", wait_command, runner=runner))
    if commands[-1]["returncode"] != 0:
        payload["status"] = "wait_and_sync_failed"
        payload["next_step"] = "Inspect wait_and_sync_bound_jobs output before refreshing derived artifacts."
        return payload

    candidate_event_csvs = list(p0_candidate_event_csvs or [])
    candidate_event_jsonls = list(p0_candidate_event_jsonls or [])
    for name, command in [
        (
            "audit_p0_strict_final_readiness",
            [sys.executable, str(p0_readiness_script)],
        ),
        (
            "write_p0_live_movement_audits",
            [sys.executable, str(p0_live_movement_audits_script)],
        ),
        ("build_p0_physical_bridge", [sys.executable, str(p0_bridge_script)]),
        *(
            [
                (
                    "build_p0_physical_tensor_copy_candidates",
                    [
                        sys.executable,
                        str(p0_candidate_builder_script),
                        *[
                            item
                            for path in candidate_event_csvs
                            for item in ("--event-csv", str(path))
                        ],
                        *[
                            item
                            for path in candidate_event_jsonls
                            for item in ("--event-jsonl", str(path))
                        ],
                    ],
                )
            ]
            if candidate_event_csvs or candidate_event_jsonls
            else []
        ),
        (
            "validate_p0_physical_tensor_copy_candidates",
            [sys.executable, str(p0_candidate_validator_script)],
        ),
        (
            "write_p0_physical_tensor_copy_event_template",
            [sys.executable, str(p0_event_template_script)],
        ),
        (
            "report_strict_final_dataplane_readiness",
            [sys.executable, str(dataplane_report_script)],
        ),
        ("sync_evidence_status_text", [sys.executable, str(sync_text_script)]),
        ("build_full_draft", [sys.executable, str(build_draft_script)]),
    ]:
        commands.append(command_record(name, command, runner=runner))
        if commands[-1]["returncode"] != 0:
            payload["status"] = f"{name}_failed"
            payload["next_step"] = f"Inspect {name} output before using refreshed artifacts."
            return payload

    if validate_draft:
        commands.append(
            command_record(
                "validate_draft_integrity",
                [sys.executable, str(validate_draft_script)],
                runner=runner,
            )
        )
        if commands[-1]["returncode"] != 0:
            payload["status"] = "validate_draft_integrity_failed"
            payload["next_step"] = "Fix draft integrity issues before presenting the refreshed status."
            return payload

    if build_beamer:
        commands.append(
            command_record(
                "build_beamer",
                ["make", "-B", "LATEX=lualatex"],
                cwd=beamer_dir,
                runner=runner,
            )
        )
        if commands[-1]["returncode"] != 0:
            payload["status"] = "build_beamer_failed"
            payload["next_step"] = "Inspect beamer build output before using presentation.pdf."
            return payload

    payload["status"] = "refreshed"
    payload["next_step"] = (
        "If Slurm status is complete, rerun with --execute to sync real SGLang results; "
        "otherwise keep waiting for the bound jobs."
    )
    write_summary_artifacts(payload, output_json=output_json, output_md=output_md)

    commands.append(
        command_record(
            "audit_goal_completion",
            [sys.executable, str(goal_audit_script)],
            runner=runner,
        )
    )
    if commands[-1]["returncode"] != 0:
        payload["status"] = "audit_goal_completion_failed"
        payload["next_step"] = "Inspect audit_goal_completion output before using refreshed artifacts."
        write_summary_artifacts(payload, output_json=output_json, output_md=output_md)
        return payload
    write_summary_artifacts(payload, output_json=output_json, output_md=output_md)

    commands.append(
        command_record(
            "audit_goal_completion_final",
            [sys.executable, str(goal_audit_script)],
            runner=runner,
        )
    )
    if commands[-1]["returncode"] != 0:
        payload["status"] = "audit_goal_completion_final_failed"
        payload["next_step"] = (
            "Inspect the final audit output; refresh summary was written but final "
            "audit did not complete."
        )
        commands.pop()
        write_summary_artifacts(payload, output_json=output_json, output_md=output_md)
        return payload
    commands.pop()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh AegisKV strict-final status artifacts without bypassing sync guards."
    )
    parser.add_argument("--wait-script", type=Path, default=DEFAULT_WAIT_SCRIPT)
    parser.add_argument(
        "--p0-readiness-script",
        type=Path,
        default=DEFAULT_P0_READINESS_SCRIPT,
    )
    parser.add_argument(
        "--p0-live-movement-audits-script",
        type=Path,
        default=DEFAULT_P0_LIVE_MOVEMENT_AUDITS_SCRIPT,
    )
    parser.add_argument("--p0-bridge-script", type=Path, default=DEFAULT_P0_BRIDGE_SCRIPT)
    parser.add_argument(
        "--p0-candidate-builder-script",
        type=Path,
        default=DEFAULT_P0_CANDIDATE_BUILDER_SCRIPT,
    )
    parser.add_argument(
        "--p0-candidate-validator-script",
        type=Path,
        default=DEFAULT_P0_CANDIDATE_VALIDATOR_SCRIPT,
    )
    parser.add_argument(
        "--p0-event-template-script",
        type=Path,
        default=DEFAULT_P0_EVENT_TEMPLATE_SCRIPT,
    )
    parser.add_argument(
        "--dataplane-report-script",
        type=Path,
        default=DEFAULT_DATAPLANE_REPORT_SCRIPT,
    )
    parser.add_argument("--sync-text-script", type=Path, default=DEFAULT_SYNC_TEXT_SCRIPT)
    parser.add_argument("--build-draft-script", type=Path, default=DEFAULT_BUILD_DRAFT_SCRIPT)
    parser.add_argument("--goal-audit-script", type=Path, default=DEFAULT_GOAL_AUDIT_SCRIPT)
    parser.add_argument("--validate-draft-script", type=Path, default=DEFAULT_VALIDATE_DRAFT_SCRIPT)
    parser.add_argument("--beamer-dir", type=Path, default=DEFAULT_BEAMER_DIR)
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUT_MD)
    parser.add_argument("--max-wait-minutes", type=float, default=0.001)
    parser.add_argument("--poll-interval-s", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--build-beamer", action="store_true")
    parser.add_argument("--skip-validate-draft", action="store_true")
    parser.add_argument("--p0-candidate-event-csv", type=Path, action="append", default=[])
    parser.add_argument("--p0-candidate-event-jsonl", type=Path, action="append", default=[])
    args = parser.parse_args()

    payload = run_refresh(
        wait_script=args.wait_script,
        p0_readiness_script=args.p0_readiness_script,
        p0_live_movement_audits_script=args.p0_live_movement_audits_script,
        p0_bridge_script=args.p0_bridge_script,
        p0_candidate_builder_script=args.p0_candidate_builder_script,
        p0_candidate_validator_script=args.p0_candidate_validator_script,
        p0_event_template_script=args.p0_event_template_script,
        dataplane_report_script=args.dataplane_report_script,
        sync_text_script=args.sync_text_script,
        build_draft_script=args.build_draft_script,
        goal_audit_script=args.goal_audit_script,
        validate_draft_script=args.validate_draft_script,
        beamer_dir=args.beamer_dir,
        max_wait_minutes=args.max_wait_minutes,
        poll_interval_s=args.poll_interval_s,
        execute=args.execute,
        build_beamer=args.build_beamer,
        validate_draft=not args.skip_validate_draft,
        p0_candidate_event_csvs=args.p0_candidate_event_csv,
        p0_candidate_event_jsonls=args.p0_candidate_event_jsonl,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    write_summary_artifacts(payload, output_json=args.output_json, output_md=args.output_md)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "refreshed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
