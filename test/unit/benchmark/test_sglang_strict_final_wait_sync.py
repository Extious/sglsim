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
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "wait_and_sync_bound_jobs.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_strict_final_wait_sync",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_status(path: Path, status: str, *, pending: int, completed: int) -> None:
    path.write_text(
        json.dumps(
            {
                "checked_at": "2026-07-02T20:00:00+08:00",
                "overall_status": status,
                "status_counts": {
                    "completed": completed,
                    "failed": 0,
                    "pending": pending,
                    "running": 4 if status == "running" else 0,
                },
                "total_bound_jobs": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_pending_jobs_do_not_sync(tmp_path: Path) -> None:
    module = load_module()
    status_json = tmp_path / "status.json"

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        assert "check_bound_slurm_jobs.py" in command[-1]
        write_status(status_json, "pending", pending=4, completed=0)
        return subprocess.CompletedProcess(command, 0, "pending\n")

    payload = module.run_wait_and_sync(
        check_script=tmp_path / "check_bound_slurm_jobs.py",
        sync_script=tmp_path / "sync_bound_jobs.sh",
        recovery_audit_script=tmp_path / "audit_strict_final_recovery_matrix.py",
        goal_audit_script=tmp_path / "audit_goal_completion.py",
        status_json=status_json,
        max_polls=1,
        poll_interval_s=0,
        execute=True,
        runner=fake_runner,
        sleeper=lambda _seconds: None,
    )

    assert payload["status"] == "pending"
    assert payload["polls"][0]["pending"] == 4
    assert payload["sync_commands"] == []


def test_complete_jobs_dry_run_reports_ready(tmp_path: Path) -> None:
    module = load_module()
    status_json = tmp_path / "status.json"

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        write_status(status_json, "complete", pending=0, completed=4)
        return subprocess.CompletedProcess(command, 0, "complete\n")

    payload = module.run_wait_and_sync(
        check_script=tmp_path / "check_bound_slurm_jobs.py",
        sync_script=tmp_path / "sync_bound_jobs.sh",
        recovery_audit_script=tmp_path / "audit_strict_final_recovery_matrix.py",
        goal_audit_script=tmp_path / "audit_goal_completion.py",
        status_json=status_json,
        max_polls=1,
        poll_interval_s=0,
        execute=False,
        runner=fake_runner,
        sleeper=lambda _seconds: None,
    )

    assert payload["status"] == "ready_to_sync"
    assert "Re-run with --execute" in payload["next_step"]
    assert payload["sync_commands"] == []


def test_complete_jobs_execute_syncs_and_audits(tmp_path: Path) -> None:
    module = load_module()
    status_json = tmp_path / "status.json"
    commands: list[str] = []

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        rendered = " ".join(command)
        commands.append(rendered)
        if "check_bound_slurm_jobs.py" in rendered:
            write_status(status_json, "complete", pending=0, completed=4)
        return subprocess.CompletedProcess(command, 0, "ok\n")

    payload = module.run_wait_and_sync(
        check_script=tmp_path / "check_bound_slurm_jobs.py",
        sync_script=tmp_path / "sync_bound_jobs.sh",
        recovery_audit_script=tmp_path / "audit_strict_final_recovery_matrix.py",
        goal_audit_script=tmp_path / "audit_goal_completion.py",
        status_json=status_json,
        max_polls=1,
        poll_interval_s=0,
        execute=True,
        runner=fake_runner,
        sleeper=lambda _seconds: None,
    )

    assert payload["status"] == "synced_and_audited"
    assert [item["name"] for item in payload["sync_commands"]] == [
        "sync_bound_jobs",
        "audit_strict_final_recovery_matrix",
        "audit_goal_completion",
    ]
    assert any("sync_bound_jobs.sh" in command for command in commands)


def test_running_jobs_are_waiting_not_sync_failure(tmp_path: Path) -> None:
    module = load_module()
    status_json = tmp_path / "status.json"

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        write_status(status_json, "running", pending=0, completed=0)
        return subprocess.CompletedProcess(command, 0, "running\n")

    payload = module.run_wait_and_sync(
        check_script=tmp_path / "check_bound_slurm_jobs.py",
        sync_script=tmp_path / "sync_bound_jobs.sh",
        recovery_audit_script=tmp_path / "audit_strict_final_recovery_matrix.py",
        goal_audit_script=tmp_path / "audit_goal_completion.py",
        status_json=status_json,
        max_polls=1,
        poll_interval_s=0,
        execute=True,
        runner=fake_runner,
        sleeper=lambda _seconds: None,
    )

    assert payload["status"] == "running"
    assert payload["polls"][0]["running"] == 4
    assert payload["sync_commands"] == []
    assert "running" in module.NONERROR_STATUSES


def test_cli_max_wait_minutes_derives_max_polls(tmp_path: Path) -> None:
    status_json = tmp_path / "status.json"
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"
    check_script = tmp_path / "check.py"
    check_script.write_text(
        "\n".join(
            [
                "import json",
                f"path = {str(status_json)!r}",
                "open(path, 'w', encoding='utf-8').write(json.dumps({",
                "    'checked_at': '2026-07-02T20:00:00+08:00',",
                "    'overall_status': 'complete',",
                "    'status_counts': {'completed': 4, 'failed': 0, 'pending': 0, 'running': 0},",
                "    'total_bound_jobs': 4,",
                "}) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check-script",
            str(check_script),
            "--status-json",
            str(status_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--max-wait-minutes",
            "0.1",
            "--poll-interval-s",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert proc.returncode == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["max_polls"] == 3
    assert payload["status"] == "ready_to_sync"
    assert "| Poll | Checked at | Slurm status |" in output_md.read_text(
        encoding="utf-8"
    )
