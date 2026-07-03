from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PLAN_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "plan_strict_final_recovery_matrix.py"
)


def load_plan_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_strict_final_recovery_plan", PLAN_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plan_contains_same_write_visible_and_invisible_runs() -> None:
    module = load_plan_module()
    plan = module.build_plan(
        repeats=1,
        base_port=41000,
        base_remote_backup_port=42000,
        remote_host="unit-host",
        remote_workdir="/tmp/aegiskv_sglang_small",
    )

    runs = plan["runs"]
    pairs = plan["pairs"]
    assert len(runs) == 4
    assert len(pairs) == 2
    assert {pair["case_id"] for pair in pairs} == {
        "same_write_best_effort",
        "same_write_wait_complete",
    }
    assert {run["port"] for run in runs} == {41000, 41001, 41002, 41003}
    assert {run["remote_backup_port"] for run in runs} == {
        42000,
        42001,
        42002,
        42003,
    }

    best_effort = [
        run for run in runs if run["prefetch_policy"] == "best_effort"
    ]
    assert {run["role"] for run in best_effort} == {
        "host_invisible_lookup_disabled",
        "host_visible_lookup_enabled",
    }
    invisible = next(
        run for run in best_effort if run["role"] == "host_invisible_lookup_disabled"
    )
    visible = next(
        run for run in best_effort if run["role"] == "host_visible_lookup_enabled"
    )
    assert invisible["env"]["ENABLE_REMOTE_BACKUP"] == "1"
    assert visible["env"]["ENABLE_REMOTE_BACKUP"] == "1"
    assert invisible["env"]["DISABLE_FAILOVER_REMOTE_LOOKUP"] == "1"
    assert visible["env"]["DISABLE_FAILOVER_REMOTE_LOOKUP"] == "0"
    assert invisible["env"]["HICACHE_STORAGE_PREFETCH_POLICY"] == "best_effort"


def test_submit_and_postprocess_commands_reuse_existing_harness() -> None:
    module = load_plan_module()
    plan = module.build_plan(
        repeats=1,
        remote_host="unit-host",
        remote_workdir="/tmp/aegiskv_sglang_small",
    )

    submit_commands = "\n".join(run["submit_command"] for run in plan["runs"])
    assert "ssh unit-host" in submit_commands
    assert "sbatch slurm/aegiskv_dp2_failure_smoke.sbatch" in submit_commands
    assert "ENABLE_REMOTE_BACKUP=1" in submit_commands
    assert "DISABLE_FAILOVER_REMOTE_LOOKUP=1" in submit_commands
    assert "DISABLE_FAILOVER_REMOTE_LOOKUP=0" in submit_commands
    assert "READY_TIMEOUT_S=2400" in submit_commands
    assert "REQUEST_TIMEOUT_S=900" in submit_commands
    assert "HICACHE_STORAGE_PREFETCH_POLICY=best_effort" in submit_commands
    assert "HICACHE_STORAGE_PREFETCH_POLICY=wait_complete" in submit_commands

    best_effort_pair = next(
        pair for pair in plan["pairs"] if pair["case_id"] == "same_write_best_effort"
    )
    assert "sync_failure_pair_from_remote.py" in best_effort_pair["sync_pair_command"]
    assert "--same-write-cost-control" in best_effort_pair["sync_pair_command"]
    assert "--prefetch-policy best_effort" in best_effort_pair["sync_pair_command"]
    assert (
        "extract_physical_restore_evidence.py"
        in best_effort_pair["physical_restore_command"]
    )
    assert "partial_async_materialized" in ";".join(
        best_effort_pair["acceptance_criteria"]
    )
    assert "aggregate_failure_visibility_pairs.py" in plan["postprocess_commands"][
        "aggregate_failure_visibility_pairs"
    ]
    assert "do not by themselves turn SGLSim restore rows" in plan[
        "strict_final_boundary"
    ]


def test_markdown_and_csv_rows_keep_boundary_visible() -> None:
    module = load_plan_module()
    plan = module.build_plan(repeats=1)

    markdown = module.render_markdown(plan)
    run_rows = module.flatten_run_rows(plan)
    pair_rows = module.flatten_pair_rows(plan)

    assert "# Strict-Final Real-System Recovery Matrix" in markdown
    assert "must not be represented as SGLSim physical tensor" in markdown
    assert len(run_rows) == 4
    assert len(pair_rows) == 2
    assert "env_enable_remote_backup" in run_rows[0]
    assert any(
        "visible_retry_latency_norm_vs_invisible < 1.0"
        in row["acceptance_criteria"]
        for row in pair_rows
    )
