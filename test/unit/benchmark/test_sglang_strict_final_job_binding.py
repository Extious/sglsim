from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
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


def load_bind_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_strict_final_job_binding", BIND_SCRIPT
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
                "invisible_run_label": "strict_final_same_write_wait_complete_r2_host_invisible_lookup_disabled",
                "visible_run_label": "strict_final_same_write_wait_complete_r2_host_visible_lookup_enabled",
                "sync_pair_direct_proxy_command": (
                    "python sync.py --invisible-job-id "
                    "<strict_final_same_write_wait_complete_r2_host_invisible_lookup_disabled_job_id> "
                    "--visible-job-id "
                    "<strict_final_same_write_wait_complete_r2_host_visible_lookup_enabled_job_id>"
                ),
                "physical_restore_command": (
                    "python physical.py --result-dir "
                    "invisible=dp2_failure_smoke_<strict_final_same_write_wait_complete_r2_host_invisible_lookup_disabled_job_id> "
                    "--result-dir "
                    "visible=dp2_failure_smoke_<strict_final_same_write_wait_complete_r2_host_visible_lookup_enabled_job_id>"
                ),
            },
            {
                "pair_id": "strict_final_same_write_wait_complete_r3",
                "invisible_run_label": "strict_final_same_write_wait_complete_r3_host_invisible_lookup_disabled",
                "visible_run_label": "strict_final_same_write_wait_complete_r3_host_visible_lookup_enabled",
                "sync_pair_direct_proxy_command": (
                    "python sync.py --invisible-job-id "
                    "<strict_final_same_write_wait_complete_r3_host_invisible_lookup_disabled_job_id> "
                    "--visible-job-id "
                    "<strict_final_same_write_wait_complete_r3_host_visible_lookup_enabled_job_id>"
                ),
                "physical_restore_command": (
                    "python physical.py --result-dir "
                    "invisible=dp2_failure_smoke_<strict_final_same_write_wait_complete_r3_host_invisible_lookup_disabled_job_id> "
                    "--result-dir "
                    "visible=dp2_failure_smoke_<strict_final_same_write_wait_complete_r3_host_visible_lookup_enabled_job_id>"
                ),
            },
        ]
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_parse_submit_log_binds_jobs_and_replaces_placeholders(tmp_path: Path) -> None:
    module = load_bind_module()
    manifest_path = tmp_path / "manifest.json"
    submit_log = tmp_path / "submit.log"
    write_manifest(manifest_path)
    submit_log.write_text(
        "\n".join(
            [
                "Submitting strict-final missing jobs via direct proxy SSH",
                "submit strict_final_same_write_wait_complete_r2 invisible",
                "Submitted batch job 50001",
                "submit strict_final_same_write_wait_complete_r2 visible",
                "Submitted batch job 50002",
                "submit strict_final_same_write_wait_complete_r3 invisible",
                "Submitted batch job 50003",
                "submit strict_final_same_write_wait_complete_r3 visible",
                "Submitted batch job 50004",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = module.run_bind(manifest_path=manifest_path, submit_log=submit_log)
    script = module.render_sync_bound_script(
        payload["manifest"],
        payload["binding_rows"],
    )

    assert payload["bound_runs"] == 4
    assert payload["total_runs"] == 4
    assert "--invisible-job-id 50001" in script
    assert "--visible-job-id 50002" in script
    assert "dp2_failure_smoke_50003" in script
    assert "dp2_failure_smoke_50004" in script
    assert "check_bound_slurm_jobs.py" in script
    assert "overall_status" in script
    assert "wait for completion before syncing" in script
    assert "WARNING" not in script


def test_empty_binding_writes_template_with_warning(tmp_path: Path) -> None:
    module = load_bind_module()
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path)

    payload = module.run_bind(manifest_path=manifest_path)
    script = module.render_sync_bound_script(
        payload["manifest"],
        payload["binding_rows"],
    )
    markdown = module.render_markdown(
        payload["manifest"],
        payload["binding_rows"],
        submit_log=None,
        job_map_csv=None,
    )

    assert payload["bound_runs"] == 0
    assert payload["total_runs"] == 4
    assert "ERROR: sync_bound_jobs.sh still has unbound Slurm job ids." in script
    assert "exit 2" in script
    assert "WARNING: this pair still has unbound placeholders" in script
    assert "<strict_final_same_write_wait_complete_r2_host_invisible_lookup_disabled_job_id>" in script
    assert "Bound runs: 0/4" in markdown
