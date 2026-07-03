from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WRAPPER_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "submit_and_bind_strict_final_missing_jobs.py"
)
BIND_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "bind_strict_final_job_ids.py"
)


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_strict_final_submit_wrapper",
        WRAPPER_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path) -> None:
    manifest = {
        "jobs": [
            {
                "pair_id": "strict_final_same_write_wait_complete_r2",
                "invisible_run_label": "strict_final_wait_r2_invisible",
                "visible_run_label": "strict_final_wait_r2_visible",
                "sync_pair_direct_proxy_command": (
                    "python sync.py --invisible-job-id "
                    "<strict_final_wait_r2_invisible_job_id> "
                    "--visible-job-id <strict_final_wait_r2_visible_job_id>"
                ),
                "physical_restore_command": (
                    "python physical.py --result-dir "
                    "invisible=dp2_failure_smoke_<strict_final_wait_r2_invisible_job_id> "
                    "--result-dir visible=dp2_failure_smoke_<strict_final_wait_r2_visible_job_id>"
                ),
            }
        ]
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_submit_wrapper_dry_run_does_not_create_submit_log(tmp_path: Path) -> None:
    module = load_wrapper_module()
    submit_script = tmp_path / "submit.sh"
    manifest_json = tmp_path / "manifest.json"
    submit_script.write_text("echo should-not-run\n", encoding="utf-8")
    write_manifest(manifest_json)

    payload = module.run_submit_and_bind(
        method="direct-proxy",
        output_dir=tmp_path,
        submit_script=submit_script,
        bind_script=BIND_SCRIPT,
        manifest_json=manifest_json,
        preflight_script=tmp_path / "preflight.py",
        execute=False,
        skip_preflight=False,
        run_stamp="unit",
    )

    assert payload["executed"] is False
    assert payload["submit_log"] == ""
    assert payload["preflight_log"] == ""
    assert payload["submit_returncode"] is None
    assert payload["bind_returncode"] is None
    assert "bash" in payload["submit_command"]


def test_submit_wrapper_execute_saves_log_and_binds_jobs(tmp_path: Path) -> None:
    module = load_wrapper_module()
    submit_script = tmp_path / "submit.sh"
    manifest_json = tmp_path / "manifest.json"
    write_manifest(manifest_json)
    submit_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "echo 'submit strict_final_same_write_wait_complete_r2 invisible'",
                "echo 'Submitted batch job 70101'",
                "echo 'submit strict_final_same_write_wait_complete_r2 visible'",
                "echo 'Submitted batch job 70102'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run_submit_and_bind(
        method="direct-proxy",
        output_dir=tmp_path,
        submit_script=submit_script,
        bind_script=BIND_SCRIPT,
        manifest_json=manifest_json,
        preflight_script=tmp_path / "preflight.py",
        execute=True,
        skip_preflight=True,
        run_stamp="unit",
    )

    assert payload["executed"] is True
    assert payload["preflight_skipped"] is True
    assert payload["submit_returncode"] == 0
    assert payload["bind_returncode"] == 0
    assert (tmp_path / "submit_missing_direct-proxy_unit.log").exists()
    assert (tmp_path / "bind_missing_direct-proxy_unit.log").exists()

    with (tmp_path / "strict_final_recovery_job_bindings.csv").open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))
    assert [row["slurm_job_id"] for row in rows] == ["70101", "70102"]
    sync_text = (tmp_path / "sync_bound_jobs.sh").read_text(encoding="utf-8")
    assert "--invisible-job-id 70101" in sync_text
    assert "dp2_failure_smoke_70102" in sync_text


def test_submit_wrapper_stops_before_submit_on_preflight_failure(tmp_path: Path) -> None:
    module = load_wrapper_module()
    submit_script = tmp_path / "submit.sh"
    manifest_json = tmp_path / "manifest.json"
    preflight_script = tmp_path / "preflight.py"
    write_manifest(manifest_json)
    submit_script.write_text("echo should-not-submit\n", encoding="utf-8")
    preflight_script.write_text(
        "\n".join(
            [
                "import sys",
                "print('remote unreachable')",
                "sys.exit(2)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run_submit_and_bind(
        method="direct-proxy",
        output_dir=tmp_path,
        submit_script=submit_script,
        bind_script=BIND_SCRIPT,
        manifest_json=manifest_json,
        preflight_script=preflight_script,
        execute=True,
        skip_preflight=False,
        run_stamp="unit",
    )

    assert payload["status"] == "preflight_failed"
    assert payload["preflight_returncode"] == 2
    assert payload["submit_returncode"] is None
    assert payload["bind_returncode"] is None
    assert (tmp_path / "preflight_missing_direct-proxy_unit.log").exists()
    assert not (tmp_path / "submit_missing_direct-proxy_unit.log").exists()
