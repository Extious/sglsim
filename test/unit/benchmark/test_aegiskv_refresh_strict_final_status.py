from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "scripts"
    / "refresh_strict_final_status.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_refresh_strict_final_status",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_refresh_runs_status_sync_draft_audit_and_validation(tmp_path: Path) -> None:
    module = load_module()
    commands: list[tuple[str, Path]] = []

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append((" ".join(command), cwd))
        return subprocess.CompletedProcess(command, 0, "ok\n")

    payload = module.run_refresh(
        wait_script=tmp_path / "wait.py",
        sync_text_script=tmp_path / "sync_text.py",
        build_draft_script=tmp_path / "build_draft.py",
        goal_audit_script=tmp_path / "audit_goal.py",
        validate_draft_script=tmp_path / "validate.py",
        beamer_dir=tmp_path / "beamer",
        max_wait_minutes=0.1,
        poll_interval_s=2,
        execute=False,
        build_beamer=False,
        validate_draft=True,
        runner=fake_runner,
    )

    assert payload["status"] == "refreshed"
    assert [item["name"] for item in payload["commands"]] == [
        "wait_and_sync_bound_jobs",
        "audit_p0_strict_final_readiness",
        "write_p0_live_movement_audits",
        "build_p0_physical_bridge",
        "validate_p0_physical_tensor_copy_candidates",
        "write_p0_physical_tensor_copy_event_template",
        "report_strict_final_dataplane_readiness",
        "sync_evidence_status_text",
        "build_full_draft",
        "validate_draft_integrity",
        "audit_goal_completion",
    ]
    assert "--execute" not in commands[0][0]
    assert "make -B LATEX=lualatex" not in [item[0] for item in commands]


def test_refresh_can_pass_execute_and_build_beamer(tmp_path: Path) -> None:
    module = load_module()
    commands: list[tuple[str, Path]] = []

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append((" ".join(command), cwd))
        return subprocess.CompletedProcess(command, 0, "ok\n")

    payload = module.run_refresh(
        wait_script=tmp_path / "wait.py",
        sync_text_script=tmp_path / "sync_text.py",
        build_draft_script=tmp_path / "build_draft.py",
        goal_audit_script=tmp_path / "audit_goal.py",
        validate_draft_script=tmp_path / "validate.py",
        beamer_dir=tmp_path / "beamer",
        execute=True,
        build_beamer=True,
        validate_draft=False,
        runner=fake_runner,
    )

    assert payload["status"] == "refreshed"
    assert "--execute" in commands[0][0]
    assert [item["name"] for item in payload["commands"]][-2:] == [
        "build_beamer",
        "audit_goal_completion",
    ]
    assert ("make -B LATEX=lualatex", tmp_path / "beamer") in commands


def test_refresh_can_build_candidates_from_live_event_inputs(tmp_path: Path) -> None:
    module = load_module()
    commands: list[tuple[str, Path]] = []

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append((" ".join(command), cwd))
        return subprocess.CompletedProcess(command, 0, "ok\n")

    payload = module.run_refresh(
        wait_script=tmp_path / "wait.py",
        p0_candidate_builder_script=tmp_path / "build_candidates.py",
        sync_text_script=tmp_path / "sync_text.py",
        build_draft_script=tmp_path / "build_draft.py",
        goal_audit_script=tmp_path / "audit_goal.py",
        validate_draft_script=tmp_path / "validate.py",
        beamer_dir=tmp_path / "beamer",
        validate_draft=False,
        p0_candidate_event_csvs=[tmp_path / "events.csv"],
        p0_candidate_event_jsonls=[tmp_path / "events.jsonl"],
        runner=fake_runner,
    )

    names = [item["name"] for item in payload["commands"]]
    assert payload["status"] == "refreshed"
    assert names.index("build_p0_physical_bridge") < names.index(
        "build_p0_physical_tensor_copy_candidates"
    )
    assert names.index("build_p0_physical_tensor_copy_candidates") < names.index(
        "validate_p0_physical_tensor_copy_candidates"
    )
    rendered = "\n".join(command for command, _ in commands)
    assert f"{tmp_path / 'build_candidates.py'}" in rendered
    assert f"--event-csv {tmp_path / 'events.csv'}" in rendered
    assert f"--event-jsonl {tmp_path / 'events.jsonl'}" in rendered


def test_refresh_stops_after_failed_wait(tmp_path: Path) -> None:
    module = load_module()

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, "failed\n")

    payload = module.run_refresh(
        wait_script=tmp_path / "wait.py",
        sync_text_script=tmp_path / "sync_text.py",
        build_draft_script=tmp_path / "build_draft.py",
        goal_audit_script=tmp_path / "audit_goal.py",
        validate_draft_script=tmp_path / "validate.py",
        beamer_dir=tmp_path / "beamer",
        runner=fake_runner,
    )

    assert payload["status"] == "wait_and_sync_failed"
    assert [item["name"] for item in payload["commands"]] == ["wait_and_sync_bound_jobs"]


def test_refresh_writes_summary_before_final_audit(tmp_path: Path) -> None:
    module = load_module()
    output_json = tmp_path / "refresh.json"
    output_md = tmp_path / "refresh.md"
    commands: list[str] = []

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        rendered = " ".join(command)
        commands.append(rendered)
        return subprocess.CompletedProcess(command, 0, "ok\n")

    payload = module.run_refresh(
        wait_script=tmp_path / "wait.py",
        sync_text_script=tmp_path / "sync_text.py",
        build_draft_script=tmp_path / "build_draft.py",
        goal_audit_script=tmp_path / "audit_goal.py",
        validate_draft_script=tmp_path / "validate.py",
        beamer_dir=tmp_path / "beamer",
        validate_draft=False,
        output_json=output_json,
        output_md=output_md,
        runner=fake_runner,
    )

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "refreshed"
    assert commands.count(f"{sys.executable} {tmp_path / 'audit_goal.py'}") == 2
    assert [item["name"] for item in summary["commands"]] == [
        "wait_and_sync_bound_jobs",
        "audit_p0_strict_final_readiness",
        "write_p0_live_movement_audits",
        "build_p0_physical_bridge",
        "validate_p0_physical_tensor_copy_candidates",
        "write_p0_physical_tensor_copy_event_template",
        "report_strict_final_dataplane_readiness",
        "sync_evidence_status_text",
        "build_full_draft",
        "audit_goal_completion",
    ]
    assert "audit_goal_completion_final" not in output_md.read_text(encoding="utf-8")


def test_stdout_tail_truncates_long_single_lines() -> None:
    module = load_module()
    text = "prefix\n" + ("x" * 7000)

    tail = module.stdout_tail(text, max_lines=60, max_chars=100)

    assert tail.startswith("...\n")
    assert len(tail) <= 104
