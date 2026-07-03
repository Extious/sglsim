from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PREPARE_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "prepare_strict_final_missing_jobs.py"
)


def load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_strict_final_missing_jobs", PREPARE_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_missing_csv(path: Path) -> None:
    fields = [
        "case_id",
        "pair_id",
        "repeat",
        "reason",
        "invisible_run_label",
        "visible_run_label",
        "invisible_submit_command",
        "visible_submit_command",
        "sync_pair_command",
        "physical_restore_command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "same_write_wait_complete",
                "pair_id": "strict_final_same_write_wait_complete_r2",
                "repeat": "2",
                "reason": "same_write_wait_complete accepted_pairs=1, required_repeats=3",
                "invisible_run_label": "strict_final_wait_r2_invisible",
                "visible_run_label": "strict_final_wait_r2_visible",
                "invisible_submit_command": (
                    "ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && "
                    "RUN_LABEL=strict_final_wait_r2_invisible "
                    "DISABLE_FAILOVER_REMOTE_LOOKUP=1 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'"
                ),
                "visible_submit_command": (
                    "ssh hkbugpumanode 'cd ~/aegiskv_sglang_small && "
                    "RUN_LABEL=strict_final_wait_r2_visible "
                    "DISABLE_FAILOVER_REMOTE_LOOKUP=0 sbatch slurm/aegiskv_dp2_failure_smoke.sbatch'"
                ),
                "sync_pair_command": "python sync_failure_pair_from_remote.py --invisible-job-id <i> --visible-job-id <v>",
                "physical_restore_command": "python extract_physical_restore_evidence.py --result-dir invisible=<i> --result-dir visible=<v>",
            }
        )


def test_prepare_manifest_builds_alias_and_direct_proxy_commands(tmp_path: Path) -> None:
    module = load_prepare_module()
    missing_csv = tmp_path / "missing.csv"
    write_missing_csv(missing_csv)

    rows = module.read_csv(missing_csv)
    manifest = module.build_manifest(
        rows,
        alias="hkbugpumanode",
        remote_user="25482351",
        remote_host="172.27.244.19",
        jump_user="zhanzhao",
        jump_host=module.DEFAULT_JUMP_HOST,
        identity_file="~/.ssh/id_rsa",
        timeout_s=20,
    )
    job = manifest["jobs"][0]

    assert manifest["job_count"] == 1
    assert job["pair_id"] == "strict_final_same_write_wait_complete_r2"
    assert job["case_id"] == "same_write_wait_complete"
    assert "ssh hkbugpumanode" in job["invisible_submit_alias"]
    assert "-F /dev/null" in job["invisible_submit_direct_proxy"]
    assert "ProxyCommand=" in job["invisible_submit_direct_proxy"]
    assert "25482351@172.27.244.19" in job["invisible_submit_direct_proxy"]
    assert "zhanzhao@faith.comp.hkbu.edu.hk" in job["invisible_submit_direct_proxy"]
    assert "DISABLE_FAILOVER_REMOTE_LOOKUP=1" in job["invisible_submit_direct_proxy"]
    assert "DISABLE_FAILOVER_REMOTE_LOOKUP=0" in job["visible_submit_direct_proxy"]
    assert "--remote-host 25482351@172.27.244.19" in job[
        "sync_pair_direct_proxy_command"
    ]
    assert "--rsync-rsh" in job["sync_pair_direct_proxy_command"]
    assert "ProxyCommand=" in job["sync_pair_direct_proxy_command"]


def test_prepare_rendered_helpers_preserve_evidence_boundary(tmp_path: Path) -> None:
    module = load_prepare_module()
    missing_csv = tmp_path / "missing.csv"
    write_missing_csv(missing_csv)
    manifest = module.build_manifest(
        module.read_csv(missing_csv),
        alias="hkbugpumanode",
        remote_user="25482351",
        remote_host="172.27.244.19",
        jump_user="zhanzhao",
        jump_host=module.DEFAULT_JUMP_HOST,
        identity_file="~/.ssh/id_rsa",
        timeout_s=20,
    )

    direct_script = module.render_submit_script(
        manifest,
        command_key="direct_proxy",
        title="Unit submit",
    )
    sync_template = module.render_sync_template(manifest)
    markdown = module.render_markdown(manifest)

    assert "Unit submit" in direct_script
    assert "strict_final_same_write_wait_complete_r2" in direct_script
    assert "Replace each <..._job_id> placeholder" in sync_template
    assert "sync_failure_pair_from_remote.py" in sync_template
    assert "--remote-host 25482351@172.27.244.19" in sync_template
    assert "--rsync-rsh" in sync_template
    assert "submit_and_bind_strict_final_missing_jobs.py" in markdown
    assert "does not claim new evidence" in markdown
    assert "recovery matrix remains" in markdown
