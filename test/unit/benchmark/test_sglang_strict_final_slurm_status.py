from __future__ import annotations

import importlib.util
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
    / "check_bound_slurm_jobs.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_strict_final_slurm_status",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_squeue_and_sacct_rows() -> None:
    module = load_module()

    squeue = module.parse_squeue(
        "\n".join(
            [
                "35136|PENDING|0:00|1:00:00|1|Resources",
                "35137|RUNNING|2:03|1:00:00|1|node001",
            ]
        )
    )
    sacct = module.parse_sacct(
        "\n".join(
            [
                "35136|aegiskv|PENDING|0:0|00:00:00|Unknown|Unknown",
                "35137|aegiskv|RUNNING|0:0|00:02:03|2026-07-02T18:00:00|Unknown",
            ]
        )
    )

    assert squeue["35136"]["squeue_state"] == "PENDING"
    assert squeue["35136"]["squeue_reason"] == "Resources"
    assert sacct["35137"]["sacct_state"] == "RUNNING"
    assert sacct["35137"]["sacct_elapsed"] == "00:02:03"


def test_collect_status_reports_pending_bound_jobs(tmp_path: Path) -> None:
    module = load_module()
    bindings = tmp_path / "bindings.csv"
    bindings.write_text(
        "\n".join(
            [
                "pair_id,role,run_label,slurm_job_id,bound,source",
                "strict_final_same_write_wait_complete_r2,invisible,r2_i,35136,true,submit_log",
                "strict_final_same_write_wait_complete_r2,visible,r2_v,35137,true,submit_log",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        assert command[0:2] == ["ssh", "hkbugpumanode"]
        stdout = "\n".join(
            [
                "35136|PENDING|0:00|1:00:00|1|Resources",
                "35137|PENDING|0:00|1:00:00|1|Priority",
                "__SACCT__",
                "35136|aegiskv|PENDING|0:0|00:00:00|Unknown|Unknown",
                "35137|aegiskv|PENDING|0:0|00:00:00|Unknown|Unknown",
                "__SCONTROL__",
                "__JOB__35136",
                "JobId=35136 JobName=aegiskv JobState=PENDING Reason=Resources StartTime=2026-07-02T22:27:19 ReqTRES=cpu=1,gres/gpu:a100=2",
                "__JOB__35137",
                "JobId=35137 JobName=aegiskv JobState=PENDING Reason=Priority StartTime=2026-07-02T22:33:19 ReqTRES=cpu=1,gres/gpu:a100=2",
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout)

    payload = module.collect_status(
        bindings_csv=bindings,
        remote_host="hkbugpumanode",
        runner=fake_runner,
    )

    assert payload["overall_status"] == "pending"
    assert payload["total_bound_jobs"] == 2
    assert payload["status_counts"]["pending"] == 2
    assert payload["rows"][0]["squeue_reason"] == "Resources"
    assert payload["rows"][0]["scontrol_StartTime"] == "2026-07-02T22:27:19"
    assert payload["rows"][0]["scontrol_ReqTRES"] == "cpu=1,gres/gpu:a100=2"
    assert "squeue -h -j 35136,35137" in payload["command"]


def test_parse_scontrol_rows() -> None:
    module = load_module()

    rows = module.parse_scontrol(
        "\n".join(
            [
                "__JOB__35136",
                "JobId=35136 JobName=aegiskv-dp2-failure-smoke JobState=PENDING Reason=Resources",
                "StartTime=2026-07-02T22:27:19 ReqTRES=cpu=1,mem=1000000M,gres/gpu:a100=2",
                "__JOB__35137",
                "JobId=35137 JobName=aegiskv-dp2-failure-smoke JobState=RUNNING Reason=None",
                "StartTime=2026-07-02T22:33:19 ReqTRES=cpu=1,gres/gpu:a100=2",
            ]
        )
    )

    assert rows["35136"]["scontrol_JobState"] == "PENDING"
    assert rows["35136"]["scontrol_StartTime"] == "2026-07-02T22:27:19"
    assert rows["35136"]["scontrol_ReqTRES"].endswith("gres/gpu:a100=2")
    assert rows["35137"]["scontrol_JobState"] == "RUNNING"


def test_overall_status_complete_requires_completed_zero_exit() -> None:
    module = load_module()
    rows = [
        {
            "effective_state": "COMPLETED",
            "sacct_exit_code": "0:0",
        },
        {
            "effective_state": "COMPLETED",
            "sacct_exit_code": "0:0",
        },
    ]

    assert module.overall_status(rows, 2) == "complete"
    assert module.status_counts(rows)["completed"] == 2
