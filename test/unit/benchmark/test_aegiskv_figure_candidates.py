import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AEGIS_ROOT = ROOT / "benchmark" / "AegisKV"
DP_SCALE_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_dp_scale_from_summaries.py"
)
CALIBRATION_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_calibration_from_overhead.py"
)
RECOVERY_PATH_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_recovery_path_from_replay.py"
)
RECOVERY_LATENCY_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_recovery_latency_from_replay.py"
)
RESTORE_BOUNDARY_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_restore_boundary_from_replay.py"
)
OVERALL_WCT_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_overall_wct_from_replay.py"
)
BUDGET_GAP_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_budget_gap_from_replay.py"
)
BUDGET_SENSITIVITY_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_budget_sensitivity_from_replay.py"
)
ABLATION_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_ablation_from_replay.py"
)
SHARED_PREFIX_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_shared_prefix_marginal_from_replay.py"
)
HOST_VISIBILITY_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_hostbackup_visibility_from_replay.py"
)
PEER_ROUTE_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_peer_route_selection_from_replay.py"
)
POLICY_MATRIX_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_policy_matrix_from_replay.py"
)
AGENTIC_POLICY_MATRIX_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_agentic_policy_matrix_from_suite.py"
)
AGENTIC_RECOVERY_LATENCY_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_agentic_recovery_latency_from_suite.py"
)
AGENTIC_PEER_ROUTE_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_agentic_peer_route_selection_from_suite.py"
)
AGENTIC_ABLATION_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_agentic_ablation_from_suite.py"
)
AGENTIC_BUDGET_SWEEP_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_agentic_budget_sweep_from_suite.py"
)
FAILURE_PAIR_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "summarize_failure_pair.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agentic_policy_matrix_derives_fig8_rows_from_online_suite(tmp_path):
    module = load_module(
        AGENTIC_POLICY_MATRIX_SCRIPT,
        "aegiskv_agentic_policy_matrix",
    )

    def write_request_details(run_dir, *, workflow_wct_ms, restore_ms):
        run_dir.mkdir(parents=True)
        fields = [
            "rid",
            "status",
            "created_time_s",
            "finish_time_s",
            "workflow_id",
            "aegis_failover_replay",
            "aegis_replay_restore_sleep_s",
            "aegis_replay_restore_overhead_s",
        ]
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for workflow_index in range(2):
                start_s = 10.0 * workflow_index
                workflow_id = f"wf-{workflow_index}"
                writer.writerow(
                    {
                        "rid": f"{workflow_id}-short",
                        "status": "completed",
                        "created_time_s": start_s,
                        "finish_time_s": start_s + 0.010,
                        "workflow_id": workflow_id,
                        "aegis_failover_replay": "false",
                        "aegis_replay_restore_sleep_s": "",
                        "aegis_replay_restore_overhead_s": "",
                    }
                )
                writer.writerow(
                    {
                        "rid": f"{workflow_id}-tail",
                        "status": "completed",
                        "created_time_s": start_s,
                        "finish_time_s": start_s + workflow_wct_ms / 1000.0,
                        "workflow_id": workflow_id,
                        "aegis_failover_replay": "false",
                        "aegis_replay_restore_sleep_s": "",
                        "aegis_replay_restore_overhead_s": "",
                    }
                )
            writer.writerow(
                {
                    "rid": "replay-0",
                    "status": "failed",
                    "created_time_s": "",
                    "finish_time_s": "",
                    "workflow_id": "",
                    "aegis_failover_replay": "true",
                    "aegis_replay_restore_sleep_s": str(restore_ms / 1000.0),
                    "aegis_replay_restore_overhead_s": str((restore_ms + 1.0) / 1000.0),
                }
            )

    rows = []
    for scenario, workflow_wct_ms, restore_ms in [
        ("policy_no_backup", 100.0, 20.0),
        ("policy_host_only", 80.0, 15.0),
        ("policy_gpu_replica", 70.0, 12.0),
        ("policy_aegiskv", 60.0, 10.0),
    ]:
        for trial in range(3):
            run_dir = tmp_path / scenario / f"trial_{trial}"
            write_request_details(
                run_dir,
                workflow_wct_ms=workflow_wct_ms + trial,
                restore_ms=restore_ms + trial,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "seed": str(8100 + trial),
                    "dp_size": "2",
                    "workflow_count": "32",
                    "mean_e2e_ms": "999.0",
                    "turns": "3",
                    "groups": "8",
                    "prompts_per_group": "4",
                    "system_prompt_len": "512",
                    "question_len": "256",
                    "output_len": "64",
                    "range_ratio": "0.75",
                    "max_concurrency": "1",
                    "critical_agent_ratio": "0.25",
                    "failover_replay": "True",
                    "failover_injection": "hard-restart",
                    "failover_retry_rank_offset": "1",
                    "replay_rows": "24",
                    "failed": "0",
                    "source_run_dir": str(run_dir),
                }
            )

    derived = module.derive_rows(rows, run_id="unit-policy")
    by_policy = {row["policy"]: row for row in derived}

    assert set(by_policy) == {
        "no backup",
        "uniform Host",
        "uniform GPU",
        "AegisKV",
    }
    assert by_policy["no backup"]["normalized_wct"] == 1.0
    assert by_policy["AegisKV"]["normalized_wct"] < by_policy["uniform GPU"][
        "normalized_wct"
    ]
    assert by_policy["AegisKV"]["source"] == "sglsim_agentic_online_policy_sweep"
    assert by_policy["AegisKV"]["repeat_count"] == 3
    assert by_policy["AegisKV"]["runtime_replay_rows"] == 72
    assert abs(by_policy["AegisKV"]["mean_workflow_wct_ms"] - 61.0) < 1e-9
    assert by_policy["no backup"]["failure_exposed_restore_norm"] == 1.0
    assert by_policy["AegisKV"]["failure_exposed_restore_norm"] < by_policy[
        "uniform GPU"
    ]["failure_exposed_restore_norm"]
    assert abs(by_policy["AegisKV"]["failure_exposed_restore_ms"] - 11.0) < 1e-9
    assert by_policy["AegisKV"]["workflow_wct_sample_count"] == 6
    assert "workflow-level completion time" in by_policy["AegisKV"]["note"]
    assert "failure_exposed_restore_norm" in by_policy["AegisKV"]["note"]


def test_agentic_recovery_latency_derives_fig9_rows_from_online_suite():
    module = load_module(
        AGENTIC_RECOVERY_LATENCY_SCRIPT,
        "aegiskv_agentic_recovery_latency",
    )
    rows = []
    for scenario, restore_ms, p90_ms in [
        ("policy_no_backup", 20.0, 120.0),
        ("policy_host_only", 15.0, 105.0),
        ("policy_gpu_replica", 12.0, 100.0),
        ("policy_aegiskv", 10.0, 92.0),
    ]:
        for trial in range(3):
            rows.append(
                {
                    "scenario": scenario,
                    "seed": str(8100 + trial),
                    "dp_size": "4",
                    "workflow_count": "32",
                    "mean_restore_sleep_ms": str(restore_ms + trial),
                    "p90_e2e_ms": str(p90_ms + trial),
                    "turns": "3",
                    "groups": "8",
                    "prompts_per_group": "4",
                    "system_prompt_len": "512",
                    "question_len": "256",
                    "output_len": "64",
                    "range_ratio": "0.75",
                    "max_concurrency": "1",
                    "critical_agent_ratio": "0.25",
                    "failover_replay": "True",
                    "failover_injection": "hard-restart",
                    "failover_retry_rank_offset": "1",
                    "replay_rows": "24",
                    "failed": "0",
                    "replay_action_host_restore": "16" if scenario == "policy_aegiskv" else "0",
                    "replay_action_peer_gpu_restore": "8" if scenario == "policy_aegiskv" else "0",
                    "replay_action_recompute": "24" if scenario == "policy_no_backup" else "0",
                    "replay_action_none": "0",
                    "source_run_dir": f"/runs/{scenario}/{trial}",
                }
            )

    derived = module.derive_rows(rows, run_id="unit-recovery-latency")
    by_policy = {row["policy"]: row for row in derived}

    assert set(by_policy) == {
        "no backup",
        "uniform Host",
        "uniform GPU",
        "AegisKV",
    }
    assert by_policy["no backup"]["retry_latency_norm"] == 1.0
    assert by_policy["AegisKV"]["retry_latency_norm"] < by_policy["uniform GPU"][
        "retry_latency_norm"
    ]
    assert by_policy["AegisKV"]["critical_agent_latency_norm"] < 1.0
    assert (
        by_policy["AegisKV"]["source"]
        == "sglsim_agentic_online_recovery_latency_sweep"
    )
    assert by_policy["AegisKV"]["repeat_count"] == 3
    assert by_policy["AegisKV"]["runtime_replay_rows"] == 72


def test_agentic_ablation_derives_fig12_rows_from_online_suite():
    module = load_module(
        AGENTIC_ABLATION_SCRIPT,
        "aegiskv_agentic_ablation",
    )
    scenario_means = [
        ("ablation_full_aegiskv", "", 100.0),
        ("ablation_no_lifecycle", "no_lifecycle", 110.0),
        ("ablation_no_segment_granularity", "no_segment_granularity", 106.0),
        ("ablation_no_restore_recompute", "no_restore_recompute", 108.0),
        ("ablation_no_pressure", "no_pressure", 104.0),
        ("ablation_no_hysteresis", "no_hysteresis", 103.0),
        ("ablation_no_critical_restore", "no_critical_restore", 111.0),
        ("ablation_static_alpha", "static_alpha", 102.0),
        ("ablation_size_only", "size_only_backup", 105.0),
        ("ablation_no_peer_locality", "no_peer_locality", 107.0),
        ("ablation_no_marginal_accounting", "no_marginal_accounting", 101.0),
    ]
    rows = []
    for scenario, ablation, mean_ms in scenario_means:
        for trial in range(3):
            rows.append(
                {
                    "scenario": scenario,
                    "seed": str(8600 + trial),
                    "dp_size": "4",
                    "workflow_count": "32",
                    "mean_e2e_ms": str(mean_ms + trial),
                    "turns": "3",
                    "groups": "8",
                    "prompts_per_group": "4",
                    "system_prompt_len": "512",
                    "question_len": "256",
                    "output_len": "64",
                    "range_ratio": "0.75",
                    "max_concurrency": "1",
                    "critical_agent_ratio": "0.25",
                    "failover_replay": "True",
                    "failover_injection": "hard-restart",
                    "failover_retry_policy": "peer",
                    "aegis_policy": "aegiskv",
                    "aegis_ablation": ablation,
                    "backup_kv": "True",
                    "replay_rows": "24",
                    "failed": "0",
                    "replay_action_host_restore": "10",
                    "replay_action_peer_gpu_restore": "8",
                    "replay_action_recompute": "6",
                    "source_run_dir": f"/runs/{scenario}/{trial}",
                }
            )

    derived = module.derive_rows(rows, run_id="unit-ablation")
    by_variant = {row["variant"]: row for row in derived}

    assert set(by_variant) == {
        "AegisKV",
        "no lifecycle",
        "no segment granularity",
        "no restore/recompute",
        "no pressure",
        "no hysteresis",
        "no critical restore",
        "static alpha",
        "size-only backup",
        "no peer locality",
        "no marginal accounting",
    }
    assert by_variant["AegisKV"]["normalized_wct"] == 1.0
    assert by_variant["no lifecycle"]["normalized_wct"] > 1.0
    assert by_variant["AegisKV"]["source"] == "sglsim_agentic_online_ablation_sweep"
    assert by_variant["AegisKV"]["repeat_count"] == 3
    assert by_variant["AegisKV"]["runtime_replay_rows"] == 72


def test_agentic_budget_sweep_derives_fig3_and_fig13_rows_from_online_suite():
    module = load_module(
        AGENTIC_BUDGET_SWEEP_SCRIPT,
        "aegiskv_agentic_budget_sweep",
    )
    budgets = [0.0, 0.1, 0.2, 0.5, 1.0]
    prefix_specs = {
        "budget_hot_tail": (100.0, -4.0, "peer", 1, 0),
        "budget_long_prefix": (100.0, 2.0, "peer", 1, 0),
        "budget_nonpeer": (100.0, 0.0, "non_peer", 2, 0),
        "budget_timing_early": (100.0, -1.0, "peer", 1, 1),
        "budget_timing_late": (100.0, -3.0, "peer", 1, 64),
    }
    rows = []
    for prefix, (base_ms, slope_ms, retry_policy, retry_offset, failure_after) in prefix_specs.items():
        for budget in budgets:
            scenario = f"{prefix}_b{int(round(budget * 100)):03d}"
            for trial in range(3):
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": str(8700 + trial),
                        "dp_size": "4",
                        "workflow_count": "32",
                        "mean_e2e_ms": str(base_ms + slope_ms * budget + trial),
                        "turns": "3",
                        "groups": "8",
                        "prompts_per_group": "4",
                        "system_prompt_len": "512" if prefix == "budget_long_prefix" else "64",
                        "question_len": "256" if prefix == "budget_long_prefix" else "32",
                        "output_len": "64",
                        "range_ratio": "0.75",
                        "max_concurrency": "1",
                        "critical_agent_ratio": "0.25",
                        "failover_replay": "True",
                        "failover_injection": "hard-restart",
                        "failover_retry_rank_offset": str(retry_offset),
                        "failover_retry_policy": retry_policy,
                        "aegis_policy": "aegiskv",
                        "aegis_gpu_budget_fraction": str(budget),
                        "failure_after_output_tokens": str(failure_after),
                        "backup_kv": "True",
                        "replay_rows": "24",
                        "failed": "0",
                        "host_lookup_hits": "24",
                        "peer_lookup_hits": "24" if retry_policy == "peer" and budget > 0 else "0",
                        "restored_tokens": "128",
                        "recompute_tokens": "8",
                        "replay_action_host_restore": "4",
                        "replay_action_peer_gpu_restore": "16" if budget > 0 else "0",
                        "replay_action_recompute": "4",
                        "source_run_dir": f"/runs/{scenario}/{trial}",
                    }
                )

    fig3_rows = module.derive_fig3_rows(rows, run_id="unit-budget")
    fig13_rows = module.derive_fig13_rows(rows, run_id="unit-budget")

    fig3_by_key = {
        (row["workload"], row["gpu_replica_budget_fraction"]): row
        for row in fig3_rows
    }
    assert fig3_by_key[("hot tails", 0.0)]["normalized_wct"] == 1.0
    assert fig3_by_key[("hot tails", 1.0)]["normalized_wct"] < 1.0
    assert fig3_by_key[("fan-out skew", 1.0)]["normalized_wct"] == 1.0
    assert ("AegisKV", "dynamic") in fig3_by_key
    assert fig3_by_key[("AegisKV", "dynamic")]["source"] == (
        "sglsim_agentic_online_budget_sweep"
    )

    fig13_by_key = {
        (row["failure_timing"], row["gpu_replica_budget_fraction"]): row
        for row in fig13_rows
    }
    assert fig13_by_key[("early decode", 0.0)]["normalized_wct"] == 1.0
    assert fig13_by_key[("late decode", 1.0)]["normalized_wct"] < 1.0
    assert fig13_by_key[("mid decode", 1.0)]["source"] == (
        "sglsim_agentic_online_budget_timing_sweep"
    )
    assert len(fig13_rows) == 15


def test_agentic_policy_matrix_rejects_mismatched_workload():
    module = load_module(
        AGENTIC_POLICY_MATRIX_SCRIPT,
        "aegiskv_agentic_policy_matrix_mismatch",
    )
    rows = []
    for scenario in [
        "policy_no_backup",
        "policy_host_only",
        "policy_gpu_replica",
        "policy_aegiskv",
    ]:
        rows.append(
            {
                "scenario": scenario,
                "seed": "8100",
                "dp_size": "4",
                "workflow_count": "32",
                "turns": "3",
                "groups": "8",
                "prompts_per_group": "4",
                "system_prompt_len": "64" if scenario == "policy_gpu_replica" else "512",
                "question_len": "32" if scenario == "policy_gpu_replica" else "256",
                "output_len": "64",
                "range_ratio": "0.75",
                "max_concurrency": "1",
                "critical_agent_ratio": "0.25",
                "failover_replay": "True",
                "failover_injection": "hard-restart",
                "failover_retry_rank_offset": "1",
                "mean_e2e_ms": "100",
                "replay_rows": "24",
                "failed": "0",
                "source_run_dir": f"/runs/{scenario}",
            }
        )

    try:
        module.derive_rows(rows, run_id="unit-policy")
        assert False, "expected workload-comparability check to fail"
    except ValueError as exc:
        assert "not workload-comparable" in str(exc)


def test_agentic_policy_matrix_rejects_mismatched_seeds():
    module = load_module(
        AGENTIC_POLICY_MATRIX_SCRIPT,
        "aegiskv_agentic_policy_matrix_seed_mismatch",
    )
    rows = []
    for index, scenario in enumerate(
        [
            "policy_no_backup",
            "policy_host_only",
            "policy_gpu_replica",
            "policy_aegiskv",
        ]
    ):
        rows.append(
            {
                "scenario": scenario,
                "seed": str(8100 + index),
                "dp_size": "4",
                "workflow_count": "32",
                "turns": "3",
                "groups": "8",
                "prompts_per_group": "4",
                "system_prompt_len": "512",
                "question_len": "256",
                "output_len": "64",
                "range_ratio": "0.75",
                "max_concurrency": "1",
                "critical_agent_ratio": "0.25",
                "failover_replay": "True",
                "failover_injection": "hard-restart",
                "failover_retry_rank_offset": "1",
                "mean_e2e_ms": "100",
                "replay_rows": "24",
                "failed": "0",
                "source_run_dir": f"/runs/{scenario}",
            }
        )

    try:
        module.derive_rows(rows, run_id="unit-policy")
        assert False, "expected seed-comparability check to fail"
    except ValueError as exc:
        assert "not seed-comparable" in str(exc)


def write_summary(path, *, dp_size, duration, run_dir):
    fields = [
        "run_dir",
        "experiment_name",
        "model_path",
        "dp_size",
        "tp_size",
        "num_requests",
        "completed",
        "failed",
        "duration",
        "mean_e2e_latency_ms",
        "aegis_failover_replay_count",
        "aegis_hard_failure_injected_replay_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "run_dir": run_dir,
                "experiment_name": "unit-dp-scale",
                "model_path": "mock-model",
                "dp_size": dp_size,
                "tp_size": 1,
                "num_requests": 32,
                "completed": 120,
                "failed": 0,
                "duration": duration,
                "mean_e2e_latency_ms": duration * 2.0,
                "aegis_failover_replay_count": 24,
                "aegis_hard_failure_injected_replay_count": 24,
            }
        )


def test_direct_dp_scale_summary_derivation(tmp_path):
    summaries = []
    for idx, (dp_size, duration) in enumerate(
        [(2, 10.0), (2, 12.0), (4, 9.0), (8, 8.0)]
    ):
        path = tmp_path / f"summary_{idx}.csv"
        write_summary(path, dp_size=dp_size, duration=duration, run_dir=f"run-{idx}")
        summaries.extend(["--summary-csv", str(path)])

    output = tmp_path / "fig_dp_scale.csv"
    subprocess.run(
        [
            sys.executable,
            str(DP_SCALE_SCRIPT),
            "--run-id",
            "unit-direct-dp",
            "--output-csv",
            str(output),
            *summaries,
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = {int(row["dp_size"]): row for row in csv.DictReader(f)}

    assert set(rows) == {2, 4, 8}
    assert float(rows[2]["normalized_wct"]) == 1.0
    assert float(rows[4]["normalized_wct"]) == 9.0 / 11.0
    assert float(rows[8]["normalized_wct"]) == 8.0 / 11.0
    assert rows[2]["repeat"] == "mean-of-2"
    assert rows[4]["repeat"] == "mean-of-1"


def test_calibration_anchor_uses_sglang_and_sglsim_overhead_rows(tmp_path):
    input_csv = tmp_path / "fig_normal_path_overhead.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "policy",
            "p99_overhead_pct",
            "p99_overhead_pct_std",
            "p99_overhead_pct_ci95",
            "slurm_job_id",
            "model_path",
            "repeat",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"policy": "none", "p99_overhead_pct": 0.0},
                {"policy": "AegisKV", "p99_overhead_pct": 5.0},
                {
                    "policy": "SGLang baseline",
                    "p99_overhead_pct": 0.0,
                    "slurm_job_id": "1+2",
                    "model_path": "Qwen/Qwen3-8B",
                    "repeat": "aggregate",
                },
                {
                    "policy": "SGLang HiCache WT",
                    "p99_overhead_pct": 0.5,
                    "slurm_job_id": "1+2",
                    "model_path": "Qwen/Qwen3-8B",
                    "repeat": "aggregate",
                },
            ]
        )
    output = tmp_path / "fig_simulator_calibration.csv"
    visibility_csv = tmp_path / "fig_hostbackup_visibility.csv"
    with visibility_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "policy",
            "retry_latency_norm",
            "retry_latency_norm_std",
            "retry_latency_norm_ci95",
            "run_id",
            "remote_host",
            "slurm_job_id",
            "model_path",
            "dp_size",
            "tp_size",
            "failure_rank",
            "repeat",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "policy": "baseline write-through invisible",
                    "retry_latency_norm": 1.0,
                    "run_id": "sglsim-control",
                },
                {
                    "policy": "Host visible only",
                    "retry_latency_norm": 0.7,
                    "run_id": "sglsim-visible",
                },
                {
                    "policy": "SGLang remote_backup lookup disabled",
                    "retry_latency_norm": 1.0,
                    "run_id": "sglang-control",
                    "slurm_job_id": "3",
                    "model_path": "Qwen/Qwen3-8B",
                    "dp_size": 2,
                    "tp_size": 1,
                    "failure_rank": 0,
                    "repeat": "aggregate",
                },
                {
                    "policy": "SGLang remote_backup lookup enabled",
                    "retry_latency_norm": 0.8,
                    "run_id": "sglang-visible",
                    "slurm_job_id": "4",
                    "model_path": "Qwen/Qwen3-8B",
                    "dp_size": 2,
                    "tp_size": 1,
                    "failure_rank": 0,
                    "repeat": "aggregate",
                },
            ]
        )

    subprocess.run(
        [
            sys.executable,
            str(CALIBRATION_SCRIPT),
            "--run-id",
            "unit-calibration",
            "--input-csv",
            str(input_csv),
            "--hostbackup-visibility-csv",
            str(visibility_csv),
            "--output-csv",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = {row["policy"]: row for row in csv.DictReader(f)}

    assert {"baseline", "protected path"}.issubset(set(rows))
    assert rows["protected path"]["metric"] == "normal_path_p99_overhead_pct"
    assert float(rows["protected path"]["sglang_norm"]) == 0.5
    assert float(rows["protected path"]["sglsim_norm"]) == 5.0
    assert rows["protected path"]["evidence_tier"] == "mixed-preliminary"
    assert rows["Host-visible retry"]["metric"] == "retry_latency_norm"


def test_sglang_failure_pair_summary_preserves_frontier_delta(tmp_path):
    def write_failure_summary(result_dir, *, p99_s, backed_up_tokens, output_tokens):
        result_dir.mkdir()
        payload = {
            "count": 4,
            "success_count": 4,
            "failover_response_count": 4,
            "captured_request_count": 4,
            "dispatched_request_count": 4,
            "response_pre_failover_output_tokens": output_tokens,
            "response_pre_failover_backed_up_tokens": backed_up_tokens,
            "failure_injection_status": 200,
            "recovery_status": 200,
            "latency_mean_s": p99_s - 0.2,
            "latency_p95_s": p99_s - 0.1,
            "latency_p99_s": p99_s,
        }
        (result_dir / "failure_summary_checked.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    invisible_dir = tmp_path / "dp2_failure_smoke_1"
    visible_dir = tmp_path / "dp2_failure_smoke_2"
    write_failure_summary(
        invisible_dir, p99_s=10.0, backed_up_tokens=0, output_tokens=100
    )
    write_failure_summary(
        visible_dir, p99_s=11.0, backed_up_tokens=500, output_tokens=125
    )

    output_dir = tmp_path / "pair"
    subprocess.run(
        [
            sys.executable,
            str(FAILURE_PAIR_SCRIPT),
            "--invisible-dir",
            str(invisible_dir),
            "--visible-dir",
            str(visible_dir),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "unit-failure-pair",
            "--invisible-job-id",
            "1",
            "--visible-job-id",
            "2",
        ],
        check=True,
        cwd=ROOT,
    )

    summary = json.loads(
        (output_dir / "failure_visibility_pair_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["frontier_token_delta"] == 500
    assert summary["visible_retry_latency_norm_vs_invisible"] == 1.1
    assert summary["visible_backed_up_tokens_per_request"] == 125.0
    assert summary["visible_backed_up_tokens_per_output_token"] == 4.0

    with (
        output_dir / "candidate_fig_hostbackup_visibility_sglang_pair.csv"
    ).open(newline="", encoding="utf-8") as f:
        rows = {row["role"]: row for row in csv.DictReader(f)}

    assert set(rows) == {
        "host_invisible_no_storage_backend",
        "host_visible_remote_backup_backend",
    }
    assert rows["host_visible_remote_backup_backend"][
        "normal_path_overhead_measured"
    ] == "false"
    assert float(rows["host_visible_remote_backup_backend"]["retry_latency_norm"]) == 1.1


def test_direct_recovery_path_derivation_replaces_only_aegiskv_row(tmp_path):
    input_csv = tmp_path / "fig_recovery_path_breakdown.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "policy",
            "peer_gpu",
            "host_h2d",
            "ordinary_cache",
            "recompute",
            "evidence_tier",
            "source",
            "run_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "policy": "uniform Host",
                    "peer_gpu": 0.0,
                    "host_h2d": 0.7,
                    "ordinary_cache": 0.2,
                    "recompute": 0.1,
                    "source": "model-baseline",
                },
                {
                    "policy": "AegisKV",
                    "peer_gpu": 0.25,
                    "host_h2d": 0.25,
                    "ordinary_cache": 0.25,
                    "recompute": 0.25,
                    "source": "old-model",
                },
            ]
        )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "request_details.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "request_id",
            "aegis_failover_replay",
            "aegis_replay_restore_action",
            "aegis_replay_restored_tokens",
            "aegis_replay_recompute_tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "request_id": "host-row",
                    "aegis_failover_replay": "true",
                    "aegis_replay_restore_action": "host_restore",
                    "aegis_replay_restored_tokens": "30",
                    "aegis_replay_recompute_tokens": "0",
                },
                {
                    "request_id": "gpu-row",
                    "aegis_failover_replay": "true",
                    "aegis_replay_restore_action": "gpu_replica",
                    "aegis_replay_restored_tokens": "10",
                    "aegis_replay_recompute_tokens": "5",
                },
                {
                    "request_id": "non-replay-row",
                    "aegis_failover_replay": "false",
                    "aegis_replay_restore_action": "host_restore",
                    "aegis_replay_restored_tokens": "999",
                    "aegis_replay_recompute_tokens": "999",
                },
            ]
        )

    output = tmp_path / "out.csv"
    subprocess.run(
        [
            sys.executable,
            str(RECOVERY_PATH_SCRIPT),
            "--run-id",
            "unit-recovery-path",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = {row["policy"]: row for row in csv.DictReader(f)}

    assert rows["uniform Host"]["source"] == "model-baseline"
    aegis = rows["AegisKV"]
    assert float(aegis["peer_gpu"]) == 10.0 / 45.0
    assert float(aegis["host_h2d"]) == 30.0 / 45.0
    assert float(aegis["ordinary_cache"]) == 0.0
    assert float(aegis["recompute"]) == 5.0 / 45.0
    assert aegis["evidence_tier"] == "sglsim-runtime-replay-derived"
    assert aegis["source"] == "sglsim_request_details_replay_recovery_path"
    assert aegis["run_id"] == "unit-recovery-path"
    assert aegis["runtime_replay_rows"] == "2"
    assert aegis["runtime_recovery_tokens"] == "45"


def test_direct_recovery_path_derivation_can_average_regimes(tmp_path):
    input_csv = tmp_path / "fig_recovery_path_breakdown.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "policy",
            "peer_gpu",
            "host_h2d",
            "ordinary_cache",
            "recompute",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "policy": "AegisKV",
                "peer_gpu": 0.0,
                "host_h2d": 1.0,
                "ordinary_cache": 0.0,
                "recompute": 0.0,
            }
        )

    def write_request_details(run_dir, rows):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "aegis_replay_restore_action",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    host_dir = tmp_path / "host"
    peer_dir = tmp_path / "peer"
    write_request_details(
        host_dir,
        [
            {
                "aegis_failover_replay": "true",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restored_tokens": "90",
                "aegis_replay_recompute_tokens": "0",
            }
        ],
    )
    write_request_details(
        peer_dir,
        [
            {
                "aegis_failover_replay": "true",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restored_tokens": "10",
                "aegis_replay_recompute_tokens": "30",
            }
        ],
    )

    output = tmp_path / "out.csv"
    subprocess.run(
        [
            sys.executable,
            str(RECOVERY_PATH_SCRIPT),
            "--run-id",
            "unit-regime-recovery-path",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output),
            "--regime",
            f"long_host={host_dir}",
            "--regime",
            f"short_peer={peer_dir}",
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["policy"] == "AegisKV"
    assert float(row["host_h2d"]) == 0.5
    assert float(row["peer_gpu"]) == 0.125
    assert float(row["recompute"]) == 0.375
    assert row["runtime_replay_rows"] == "2"
    assert row["runtime_recovery_tokens"] == "130"
    assert row["runtime_regime_count"] == "2"
    assert row["runtime_regimes"] == "long_host;short_peer"


def test_direct_recovery_latency_derivation_can_average_regimes(tmp_path):
    input_csv = tmp_path / "fig_recovery_latency.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "policy",
            "retry_latency_norm",
            "critical_agent_latency_norm",
            "source",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "policy": "none",
                    "retry_latency_norm": 1.0,
                    "critical_agent_latency_norm": 1.0,
                    "source": "model-baseline",
                },
                {
                    "policy": "AegisKV",
                    "retry_latency_norm": 0.1,
                    "critical_agent_latency_norm": 0.1,
                    "source": "old-model",
                },
            ]
        )

    def write_request_details(run_dir, *, selected_ms, tokens=100):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restore_sleep_s",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "aegis_failover_replay": "true",
                    "aegis_candidate_pre_failure_total_tokens": str(tokens),
                    "aegis_replay_restore_overhead_s": str(selected_ms / 1000.0),
                    "aegis_replay_restore_sleep_s": str(selected_ms / 1000.0),
                    "aegis_replay_restored_tokens": "50",
                    "aegis_replay_recompute_tokens": "50",
                }
            )

    # Same-length recompute baseline for 100 tokens is 5.08 ms.
    fast_dir = tmp_path / "fast"
    slow_dir = tmp_path / "slow"
    write_request_details(fast_dir, selected_ms=2.54)
    write_request_details(slow_dir, selected_ms=5.08)

    output = tmp_path / "out.csv"
    subprocess.run(
        [
            sys.executable,
            str(RECOVERY_LATENCY_SCRIPT),
            "--run-id",
            "unit-regime-recovery-latency",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output),
            "--regime",
            f"fast={fast_dir}",
            "--regime",
            f"slow={slow_dir}",
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = {row["policy"]: row for row in csv.DictReader(f)}

    assert rows["none"]["source"] == "model-baseline"
    aegis = rows["AegisKV"]
    assert float(aegis["retry_latency_norm"]) == 0.75
    assert float(aegis["critical_agent_latency_norm"]) == 0.75
    assert aegis["source"] == "sglsim_request_details_replay_recovery_latency"
    assert aegis["runtime_replay_rows"] == "2"
    assert aegis["runtime_recovery_tokens"] == "200"
    assert aegis["runtime_regime_count"] == "2"
    assert aegis["runtime_regimes"] == "fast;slow"


def test_restore_boundary_derivation_calibrates_from_replay_rows(tmp_path):
    def write_request_details(run_dir, rows):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_replay_restore_action",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    recompute_dir = tmp_path / "recompute"
    host_dir = tmp_path / "host"
    gpu_dir = tmp_path / "gpu"
    write_request_details(
        recompute_dir,
        [
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.005",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "100",
            },
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.006",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "200",
            },
        ],
    )
    write_request_details(
        host_dir,
        [
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.009",
                "aegis_replay_restored_tokens": "100",
                "aegis_replay_recompute_tokens": "0",
            },
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.011",
                "aegis_replay_restored_tokens": "200",
                "aegis_replay_recompute_tokens": "0",
            },
        ],
    )
    write_request_details(
        gpu_dir,
        [
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.0029",
                "aegis_replay_restored_tokens": "20",
                "aegis_replay_recompute_tokens": "80",
            },
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.0038",
                "aegis_replay_restored_tokens": "40",
                "aegis_replay_recompute_tokens": "160",
            },
        ],
    )

    output = tmp_path / "fig_restore_recompute_boundary.csv"
    subprocess.run(
        [
            sys.executable,
            str(RESTORE_BOUNDARY_SCRIPT),
            "--run-id",
            "unit-boundary",
            "--output-csv",
            str(output),
            "--host-run-dir",
            str(host_dir),
            "--gpu-run-dir",
            str(gpu_dir),
            "--recompute-run-dir",
            str(recompute_dir),
            "--token-points",
            "100",
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["source"] == "sglsim_request_details_replay_restore_boundary"
    assert float(row["recompute_ms"]) == 5.0
    assert float(row["host_restore_ms"]) == 9.0
    assert abs(float(row["gpu_replica_ms"]) - 2.9) < 1e-9
    assert abs(float(row["aegis_lower_envelope_ms"]) - 2.9) < 1e-9
    assert float(row["gpu_observed_coverage"]) == 0.2
    assert row["runtime_replay_rows"] == "6"


def test_budget_gap_derivation_uses_peer_visible_budget_counterfactual(tmp_path):
    def write_request_details(run_dir, rows):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "client_wall_e2e_latency_s",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_replay_restore_action",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
                "aegis_peer_gpu_lookup_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    recompute_dir = tmp_path / "recompute"
    host_dir = tmp_path / "host"
    gpu_dir = tmp_path / "gpu"
    write_request_details(
        recompute_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "1000",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.014",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "1000",
                "aegis_peer_gpu_lookup_tokens": "0",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "2000",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.024",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "2000",
                "aegis_peer_gpu_lookup_tokens": "0",
            },
        ],
    )
    write_request_details(
        host_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "1000",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.010",
                "aegis_replay_restored_tokens": "1000",
                "aegis_replay_recompute_tokens": "0",
                "aegis_peer_gpu_lookup_tokens": "100",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "2000",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.012",
                "aegis_replay_restored_tokens": "2000",
                "aegis_replay_recompute_tokens": "0",
                "aegis_peer_gpu_lookup_tokens": "200",
            },
        ],
    )
    write_request_details(
        gpu_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.003",
                "aegis_replay_restored_tokens": "20",
                "aegis_replay_recompute_tokens": "80",
                "aegis_peer_gpu_lookup_tokens": "20",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.004",
                "aegis_replay_restored_tokens": "40",
                "aegis_replay_recompute_tokens": "160",
                "aegis_peer_gpu_lookup_tokens": "40",
            },
        ],
    )

    output = tmp_path / "fig_static_gpu_replica_budget_gap.csv"
    subprocess.run(
        [
            sys.executable,
            str(BUDGET_GAP_SCRIPT),
            "--run-id",
            "unit-budget-gap",
            "--output-csv",
            str(output),
            "--host-run-dir",
            str(host_dir),
            "--gpu-run-dir",
            str(gpu_dir),
            "--recompute-run-dir",
            str(recompute_dir),
            "--budget-points",
            "0,1",
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 7
    keyed = {
        (row["workload"], row["gpu_replica_budget_fraction"]): row for row in rows
    }
    assert float(keyed[("hot tails", "0.0")]["normalized_wct"]) == 1.0
    assert float(keyed[("hot tails", "1.0")]["normalized_wct"]) < 1.0
    assert float(keyed[("fan-out skew", "1.0")]["normalized_wct"]) == 1.0
    assert float(keyed[("long prefixes", "0.0")]["normalized_wct"]) < 1.0

    aegis = keyed[("AegisKV", "dynamic")]
    assert aegis["evidence_tier"] == "sglsim-runtime-replay-calibrated"
    assert aegis["source"] == "sglsim_request_details_replay_static_gpu_budget_counterfactual"
    assert aegis["runtime_replay_rows"] == "6"
    assert aegis["runtime_regime_count"] == "3"


def test_budget_sensitivity_derivation_uses_replay_regime_proxy(tmp_path):
    def write_request_details(run_dir, rows):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "client_wall_e2e_latency_s",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_replay_restore_action",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
                "aegis_peer_gpu_lookup_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    recompute_dir = tmp_path / "recompute"
    host_dir = tmp_path / "host"
    gpu_dir = tmp_path / "gpu"
    write_request_details(
        recompute_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "1000",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.014",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "1000",
                "aegis_peer_gpu_lookup_tokens": "0",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "2000",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.024",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "2000",
                "aegis_peer_gpu_lookup_tokens": "0",
            },
        ],
    )
    write_request_details(
        host_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "1000",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.010",
                "aegis_replay_restored_tokens": "1000",
                "aegis_replay_recompute_tokens": "0",
                "aegis_peer_gpu_lookup_tokens": "100",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "2000",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.012",
                "aegis_replay_restored_tokens": "2000",
                "aegis_replay_recompute_tokens": "0",
                "aegis_peer_gpu_lookup_tokens": "200",
            },
        ],
    )
    write_request_details(
        gpu_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.003",
                "aegis_replay_restored_tokens": "20",
                "aegis_replay_recompute_tokens": "80",
                "aegis_peer_gpu_lookup_tokens": "20",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.004",
                "aegis_replay_restored_tokens": "40",
                "aegis_replay_recompute_tokens": "160",
                "aegis_peer_gpu_lookup_tokens": "40",
            },
        ],
    )

    output = tmp_path / "fig_budget_sensitivity.csv"
    subprocess.run(
        [
            sys.executable,
            str(BUDGET_SENSITIVITY_SCRIPT),
            "--run-id",
            "unit-budget-sensitivity",
            "--output-csv",
            str(output),
            "--host-run-dir",
            str(host_dir),
            "--gpu-run-dir",
            str(gpu_dir),
            "--recompute-run-dir",
            str(recompute_dir),
            "--budget-points",
            "0,1",
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 6
    assert {row["failure_timing"] for row in rows} == {
        "hot-tail",
        "long-prefix",
        "non-peer",
    }
    keyed = {
        (row["failure_timing"], row["gpu_replica_budget_fraction"]): row
        for row in rows
    }
    assert float(keyed[("hot-tail", "1.0")]["normalized_wct"]) < 1.0
    assert float(keyed[("non-peer", "1.0")]["normalized_wct"]) == 1.0
    assert keyed[("hot-tail", "1.0")]["evidence_tier"] == (
        "sglsim-runtime-replay-calibrated"
    )
    assert keyed[("hot-tail", "1.0")]["source"] == (
        "sglsim_request_details_replay_budget_sensitivity_counterfactual"
    )


def test_ablation_derivation_normalizes_to_full_aegiskv(tmp_path):
    def write_request_details(run_dir, rows):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "client_wall_e2e_latency_s",
                "critical_agent",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_replay_restore_action",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
                "aegis_peer_gpu_lookup_tokens",
                "aegis_host_backup_lookup_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    recompute_dir = tmp_path / "recompute"
    host_dir = tmp_path / "host"
    gpu_dir = tmp_path / "gpu"
    write_request_details(
        recompute_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "critical_agent": "true",
                "aegis_candidate_pre_failure_total_tokens": "1000",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.014",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "1000",
                "aegis_peer_gpu_lookup_tokens": "0",
                "aegis_host_backup_lookup_tokens": "1000",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "critical_agent": "true",
                "aegis_candidate_pre_failure_total_tokens": "2000",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.024",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "2000",
                "aegis_peer_gpu_lookup_tokens": "0",
                "aegis_host_backup_lookup_tokens": "2000",
            },
        ],
    )
    write_request_details(
        host_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "critical_agent": "true",
                "aegis_candidate_pre_failure_total_tokens": "1000",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.010",
                "aegis_replay_restored_tokens": "1000",
                "aegis_replay_recompute_tokens": "0",
                "aegis_peer_gpu_lookup_tokens": "100",
                "aegis_host_backup_lookup_tokens": "1000",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "critical_agent": "true",
                "aegis_candidate_pre_failure_total_tokens": "2000",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.012",
                "aegis_replay_restored_tokens": "2000",
                "aegis_replay_recompute_tokens": "0",
                "aegis_peer_gpu_lookup_tokens": "200",
                "aegis_host_backup_lookup_tokens": "2000",
            },
        ],
    )
    write_request_details(
        gpu_dir,
        [
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "critical_agent": "true",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.003",
                "aegis_replay_restored_tokens": "20",
                "aegis_replay_recompute_tokens": "80",
                "aegis_peer_gpu_lookup_tokens": "20",
                "aegis_host_backup_lookup_tokens": "100",
            },
            {
                "aegis_failover_replay": "true",
                "client_wall_e2e_latency_s": "0.100",
                "critical_agent": "true",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.004",
                "aegis_replay_restored_tokens": "40",
                "aegis_replay_recompute_tokens": "160",
                "aegis_peer_gpu_lookup_tokens": "40",
                "aegis_host_backup_lookup_tokens": "200",
            },
        ],
    )

    output = tmp_path / "fig_ablation.csv"
    subprocess.run(
        [
            sys.executable,
            str(ABLATION_SCRIPT),
            "--run-id",
            "unit-ablation",
            "--output-csv",
            str(output),
            "--host-run-dir",
            str(host_dir),
            "--gpu-run-dir",
            str(gpu_dir),
            "--recompute-run-dir",
            str(recompute_dir),
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = {row["variant"]: row for row in csv.DictReader(f)}

    assert float(rows["AegisKV"]["normalized_wct"]) == 1.0
    assert float(rows["no lifecycle"]["normalized_wct"]) > 1.0
    assert (
        float(rows["no critical restore"]["normalized_wct"])
        == float(rows["no lifecycle"]["normalized_wct"])
    )
    assert rows["AegisKV"]["source"] == "sglsim_request_details_replay_ablation_counterfactual"
    assert rows["AegisKV"]["runtime_replay_rows"] == "6"
    assert rows["AegisKV"]["runtime_regime_count"] == "3"


def test_shared_prefix_marginal_derivation_excludes_retry_cache_hits(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fields = [
        "rid",
        "agent_id",
        "aegis_worker_seq",
        "aegis_failover_replay",
        "aegis_cross_rank_retry",
        "aegis_safe_frontier_tokens",
        "cache_hit_tokens",
        "input_length",
        "generated_tokens",
    ]
    with (run_dir / "request_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "rid": "a-t0",
                    "agent_id": "a",
                    "aegis_worker_seq": "0",
                    "aegis_failover_replay": "false",
                    "aegis_cross_rank_retry": "false",
                    "aegis_safe_frontier_tokens": "100",
                    "cache_hit_tokens": "0",
                },
                {
                    "rid": "a-t1",
                    "agent_id": "a",
                    "aegis_worker_seq": "1",
                    "aegis_failover_replay": "false",
                    "aegis_cross_rank_retry": "false",
                    "aegis_safe_frontier_tokens": "150",
                    "cache_hit_tokens": "0",
                },
                {
                    "rid": "a-t2",
                    "agent_id": "a",
                    "aegis_worker_seq": "2",
                    "aegis_failover_replay": "false",
                    "aegis_cross_rank_retry": "false",
                    "aegis_safe_frontier_tokens": "200",
                    "cache_hit_tokens": "0",
                },
                {
                    "rid": "a-t2-retry",
                    "agent_id": "a",
                    "aegis_worker_seq": "2",
                    "aegis_failover_replay": "true",
                    "aegis_cross_rank_retry": "true",
                    "aegis_safe_frontier_tokens": "999",
                    "cache_hit_tokens": "999",
                },
                {
                    "rid": "b-t0",
                    "agent_id": "b",
                    "aegis_worker_seq": "0",
                    "aegis_failover_replay": "false",
                    "aegis_cross_rank_retry": "false",
                    "aegis_safe_frontier_tokens": "100",
                    "cache_hit_tokens": "0",
                },
                {
                    "rid": "b-t1",
                    "agent_id": "b",
                    "aegis_worker_seq": "1",
                    "aegis_failover_replay": "false",
                    "aegis_cross_rank_retry": "false",
                    "aegis_safe_frontier_tokens": "100",
                    "cache_hit_tokens": "0",
                },
                {
                    "rid": "b-t2",
                    "agent_id": "b",
                    "aegis_worker_seq": "2",
                    "aegis_failover_replay": "false",
                    "aegis_cross_rank_retry": "false",
                    "aegis_safe_frontier_tokens": "200",
                    "cache_hit_tokens": "0",
                },
            ]
        )

    output = tmp_path / "fig_shared_prefix_marginal.csv"
    subprocess.run(
        [
            sys.executable,
            str(SHARED_PREFIX_SCRIPT),
            "--run-id",
            "unit-shared-prefix",
            "--output-csv",
            str(output),
            "--run-dir",
            str(run_dir),
            "--bin-width",
            "100",
            "--min-group-count",
            "1",
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["source"] == "sglsim_request_details_original_shared_prefix_marginal"
    assert float(row["dup_no_dedup"]) == (2.25 + 2.0) / 2.0
    assert float(row["dup_marginal"]) == 1.0
    assert float(row["coverage_marginal"]) == 1.0
    assert row["runtime_agent_groups"] == "2"
    assert row["runtime_request_rows"] == "6"


def test_hostbackup_visibility_derivation_uses_recoverable_replay_regimes(tmp_path):
    def write_request_details(run_dir, rows):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_replay_restore_action",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    recompute_dir = tmp_path / "recompute"
    host_dir = tmp_path / "host"
    gpu_dir = tmp_path / "gpu"
    write_request_details(
        recompute_dir,
        [
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.005",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "100",
            },
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.006",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "200",
            },
        ],
    )
    write_request_details(
        host_dir,
        [
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.009",
                "aegis_replay_restored_tokens": "100",
                "aegis_replay_recompute_tokens": "0",
            },
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.011",
                "aegis_replay_restored_tokens": "200",
                "aegis_replay_recompute_tokens": "0",
            },
        ],
    )
    write_request_details(
        gpu_dir,
        [
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.0029",
                "aegis_replay_restored_tokens": "20",
                "aegis_replay_recompute_tokens": "80",
            },
            {
                "aegis_failover_replay": "true",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.0038",
                "aegis_replay_restored_tokens": "40",
                "aegis_replay_recompute_tokens": "160",
            },
        ],
    )

    output = tmp_path / "fig_hostbackup_visibility.csv"
    subprocess.run(
        [
            sys.executable,
            str(HOST_VISIBILITY_SCRIPT),
            "--run-id",
            "unit-host-visibility",
            "--output-csv",
            str(output),
            "--host-run-dir",
            str(host_dir),
            "--gpu-run-dir",
            str(gpu_dir),
            "--recompute-run-dir",
            str(recompute_dir),
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = {row["policy"]: row for row in csv.DictReader(f)}

    assert float(rows["baseline write-through invisible"]["retry_latency_norm"]) == 1.0
    assert rows["Host visible only"]["runtime_regimes"] == "long_prefix_host"
    assert rows["GPUReplica only"]["runtime_regimes"] == "short_peer_gpu"
    assert rows["AegisKV"]["runtime_regimes"] == "long_prefix_host;short_peer_gpu"
    assert rows["AegisKV"]["source"] == "sglsim_replay_retry_counterfactual_plus_policy_overhead_anchor"
    assert rows["AegisKV"]["evidence_tier"] == "mixed-preliminary"


def test_overall_wct_derivation_uses_replay_counterfactual(tmp_path):
    input_csv = tmp_path / "fig_overall_wct.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["policy", "normalized_wct", "source"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"policy": "none", "normalized_wct": 1.0, "source": "model"},
                {"policy": "AegisKV", "normalized_wct": 0.1, "source": "old"},
            ]
        )

    def write_request_details(run_dir, *, e2e_ms, selected_ms, tokens):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "client_wall_e2e_latency_s",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "aegis_failover_replay": "true",
                    "client_wall_e2e_latency_s": str(e2e_ms / 1000.0),
                    "aegis_candidate_pre_failure_total_tokens": str(tokens),
                    "aegis_replay_restore_overhead_s": str(selected_ms / 1000.0),
                    "aegis_replay_restored_tokens": str(tokens),
                    "aegis_replay_recompute_tokens": "0",
                }
            )

    # Same-length recompute baseline for 100 tokens is 5.08 ms.
    fast_dir = tmp_path / "fast"
    slow_dir = tmp_path / "slow"
    write_request_details(fast_dir, e2e_ms=100.0, selected_ms=2.54, tokens=100)
    write_request_details(slow_dir, e2e_ms=100.0, selected_ms=5.08, tokens=100)

    output = tmp_path / "out.csv"
    subprocess.run(
        [
            sys.executable,
            str(OVERALL_WCT_SCRIPT),
            "--run-id",
            "unit-overall-wct",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output),
            "--regime",
            f"fast={fast_dir}",
            "--regime",
            f"slow={slow_dir}",
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = {row["policy"]: row for row in csv.DictReader(f)}

    assert rows["none"]["source"] == "model"
    aegis = rows["AegisKV"]
    expected_fast = 100.0 / (100.0 - 2.54 + 5.08)
    expected_slow = 1.0
    assert float(aegis["normalized_wct"]) == (expected_fast + expected_slow) / 2.0
    assert aegis["source"] == "sglsim_request_details_replay_overall_wct_counterfactual"
    assert aegis["runtime_replay_rows"] == "2"
    assert aegis["runtime_recovery_tokens"] == "200"
    assert aegis["runtime_regime_count"] == "2"


def test_peer_route_selection_derivation_matches_requests(tmp_path):
    def write_request_details(run_dir, *, action, overheads_ms, e2e_ms):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "rid",
                "aegis_failover_replay",
                "e2e_latency_s",
                "aegis_replay_restore_action",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restore_sleep_s",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_peer_gpu_lookup_hit",
                "aegis_peer_gpu_lookup_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for idx, overhead_ms in enumerate(overheads_ms):
                writer.writerow(
                    {
                        "rid": f"req-{idx}",
                        "aegis_failover_replay": "true",
                        "e2e_latency_s": str(e2e_ms[idx] / 1000.0),
                        "aegis_replay_restore_action": action,
                        "aegis_replay_restore_overhead_s": str(overhead_ms / 1000.0),
                        "aegis_replay_restore_sleep_s": str(overhead_ms / 1000.0),
                        "aegis_candidate_pre_failure_total_tokens": "128",
                        "aegis_peer_gpu_lookup_hit": str(action == "gpu_replica"),
                        "aegis_peer_gpu_lookup_tokens": "64"
                        if action == "gpu_replica"
                        else "0",
                    }
                )

    peer_dir = tmp_path / "peer"
    nonpeer_dir = tmp_path / "nonpeer"
    write_request_details(
        peer_dir,
        action="gpu_replica",
        overheads_ms=[2.0, 3.0],
        e2e_ms=[100.0, 200.0],
    )
    write_request_details(
        nonpeer_dir,
        action="recompute",
        overheads_ms=[4.0, 6.0],
        e2e_ms=[100.0, 200.0],
    )

    output = tmp_path / "fig_peer_rank_selection.csv"
    subprocess.run(
        [
            sys.executable,
            str(PEER_ROUTE_SCRIPT),
            "--run-id",
            "unit-peer-route",
            "--output-csv",
            str(output),
            "--peer-run-dir",
            str(peer_dir),
            "--nonpeer-run-dir",
            str(nonpeer_dir),
            "--dp-size",
            "4",
        ],
        check=True,
        cwd=ROOT,
    )

    with output.open(newline="", encoding="utf-8") as f:
        rows = {row["policy"]: row for row in csv.DictReader(f)}

    assert set(rows) == {
        "non-peer route miss",
        "random healthy route",
        "AegisKV peer route",
    }
    assert float(rows["non-peer route miss"]["normalized_wct"]) == 1.0
    assert float(rows["non-peer route miss"]["peer_p99_overhead_pct"]) == 100.0
    assert (
        float(rows["AegisKV peer route"]["normalized_wct"])
        < float(rows["random healthy route"]["normalized_wct"])
        < 1.0
    )
    assert (
        float(rows["AegisKV peer route"]["peer_p99_overhead_pct"])
        < float(rows["random healthy route"]["peer_p99_overhead_pct"])
        < 100.0
    )
    assert rows["AegisKV peer route"]["runtime_replay_rows"] == "2"
    assert rows["AegisKV peer route"]["evidence_tier"] == "sglsim-runtime-replay-derived"


def test_agentic_peer_route_selection_derives_online_rows(tmp_path):
    module = load_module(
        AGENTIC_PEER_ROUTE_SCRIPT,
        "aegiskv_agentic_peer_route",
    )

    def write_run(run_dir, *, action, peer_hit, overhead_ms):
        run_dir.mkdir(parents=True)
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "rid",
                "aegis_failover_replay",
                "aegis_peer_gpu_lookup_hit",
                "aegis_host_backup_lookup_hit",
                "aegis_replay_restore_action",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restore_sleep_s",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "rid": "req-retry",
                    "aegis_failover_replay": "true",
                    "aegis_peer_gpu_lookup_hit": str(peer_hit),
                    "aegis_host_backup_lookup_hit": "false",
                    "aegis_replay_restore_action": action,
                    "aegis_replay_restore_overhead_s": str(overhead_ms / 1000.0),
                    "aegis_replay_restore_sleep_s": str(overhead_ms / 1000.0),
                }
            )

    rows = []
    scenario_specs = [
        ("peer_route_nonpeer_hot_tail", "non_peer", "recompute", False, 100.0, 6.0),
        ("peer_route_random_hot_tail", "random_healthy", "recompute", False, 95.0, 5.0),
        ("peer_route_aegiskv_hot_tail", "peer", "gpu_replica", True, 90.0, 3.0),
    ]
    for scenario, route_policy, action, peer_hit, mean_ms, overhead_ms in scenario_specs:
        for trial in range(2):
            run_dir = tmp_path / scenario / f"trial_{trial}"
            write_run(
                run_dir,
                action=action,
                peer_hit=peer_hit,
                overhead_ms=overhead_ms + trial,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "seed": str(8500 + trial),
                    "dp_size": "4",
                    "workflow_count": "32",
                    "turns": "3",
                    "groups": "8",
                    "prompts_per_group": "4",
                    "system_prompt_len": "64",
                    "question_len": "32",
                    "output_len": "64",
                    "range_ratio": "0.75",
                    "max_concurrency": "1",
                    "critical_agent_ratio": "0.25",
                    "failover_replay": "True",
                    "failover_injection": "hard-restart",
                    "failover_retry_rank_offset": "1",
                    "failover_retry_policy": route_policy,
                    "mean_e2e_ms": str(mean_ms + trial),
                    "failed": "0",
                    "source_run_dir": str(run_dir),
                }
            )

    derived = module.derive_rows(rows, run_id="unit-peer-route-online")
    by_policy = {row["policy"]: row for row in derived}

    assert set(by_policy) == {
        "non-peer route miss",
        "random healthy route",
        "AegisKV peer route",
    }
    assert by_policy["non-peer route miss"]["normalized_wct"] == 1.0
    assert by_policy["AegisKV peer route"]["normalized_wct"] < by_policy[
        "random healthy route"
    ]["normalized_wct"]
    assert by_policy["AegisKV peer route"]["runtime_route_hit_rate"] == 1.0
    assert by_policy["random healthy route"]["source"] == (
        "sglsim_agentic_online_peer_route_sweep"
    )


def test_policy_matrix_derivation_replaces_synthesis_baselines(tmp_path):
    def write_request_details(run_dir, rows):
        run_dir.mkdir()
        with (run_dir / "request_details.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "aegis_failover_replay",
                "e2e_latency_s",
                "aegis_candidate_pre_failure_total_tokens",
                "aegis_replay_restore_action",
                "aegis_replay_restore_overhead_s",
                "aegis_replay_restore_sleep_s",
                "aegis_replay_restored_tokens",
                "aegis_replay_recompute_tokens",
                "aegis_host_backup_lookup_tokens",
                "aegis_peer_gpu_lookup_tokens",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({"aegis_failover_replay": "true", **row})

    host_dir = tmp_path / "host"
    gpu_dir = tmp_path / "gpu"
    recompute_dir = tmp_path / "recompute"
    write_request_details(
        host_dir,
        [
            {
                "e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.009",
                "aegis_replay_restore_sleep_s": "0.009",
                "aegis_replay_restored_tokens": "100",
                "aegis_replay_recompute_tokens": "0",
                "aegis_host_backup_lookup_tokens": "100",
                "aegis_peer_gpu_lookup_tokens": "20",
            },
            {
                "e2e_latency_s": "0.200",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_overhead_s": "0.010",
                "aegis_replay_restore_sleep_s": "0.010",
                "aegis_replay_restored_tokens": "200",
                "aegis_replay_recompute_tokens": "0",
                "aegis_host_backup_lookup_tokens": "200",
                "aegis_peer_gpu_lookup_tokens": "40",
            },
        ],
    )
    write_request_details(
        gpu_dir,
        [
            {
                "e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.006",
                "aegis_replay_restore_sleep_s": "0.006",
                "aegis_replay_restored_tokens": "40",
                "aegis_replay_recompute_tokens": "60",
                "aegis_host_backup_lookup_tokens": "100",
                "aegis_peer_gpu_lookup_tokens": "40",
            },
            {
                "e2e_latency_s": "0.200",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_overhead_s": "0.008",
                "aegis_replay_restore_sleep_s": "0.008",
                "aegis_replay_restored_tokens": "80",
                "aegis_replay_recompute_tokens": "120",
                "aegis_host_backup_lookup_tokens": "200",
                "aegis_peer_gpu_lookup_tokens": "80",
            },
        ],
    )
    write_request_details(
        recompute_dir,
        [
            {
                "e2e_latency_s": "0.100",
                "aegis_candidate_pre_failure_total_tokens": "100",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.005",
                "aegis_replay_restore_sleep_s": "0.005",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "100",
                "aegis_host_backup_lookup_tokens": "100",
                "aegis_peer_gpu_lookup_tokens": "0",
            },
            {
                "e2e_latency_s": "0.200",
                "aegis_candidate_pre_failure_total_tokens": "200",
                "aegis_replay_restore_action": "recompute",
                "aegis_replay_restore_overhead_s": "0.007",
                "aegis_replay_restore_sleep_s": "0.007",
                "aegis_replay_restored_tokens": "0",
                "aegis_replay_recompute_tokens": "200",
                "aegis_host_backup_lookup_tokens": "200",
                "aegis_peer_gpu_lookup_tokens": "0",
            },
        ],
    )

    fig8 = tmp_path / "fig8.csv"
    fig9 = tmp_path / "fig9.csv"
    fig10 = tmp_path / "fig10.csv"
    subprocess.run(
        [
            sys.executable,
            str(POLICY_MATRIX_SCRIPT),
            "--run-id",
            "unit-policy-matrix",
            "--fig8-csv",
            str(fig8),
            "--fig9-csv",
            str(fig9),
            "--fig10-csv",
            str(fig10),
            "--host-run-dir",
            str(host_dir),
            "--gpu-run-dir",
            str(gpu_dir),
            "--recompute-run-dir",
            str(recompute_dir),
        ],
        check=True,
        cwd=ROOT,
    )

    with fig8.open(newline="", encoding="utf-8") as f:
        fig8_rows = {row["policy"]: row for row in csv.DictReader(f)}
    with fig9.open(newline="", encoding="utf-8") as f:
        fig9_rows = {row["policy"]: row for row in csv.DictReader(f)}
    with fig10.open(newline="", encoding="utf-8") as f:
        fig10_rows = {row["policy"]: row for row in csv.DictReader(f)}

    assert set(fig8_rows) == {"no backup", "uniform Host", "uniform GPU", "AegisKV"}
    assert set(fig9_rows) == set(fig8_rows)
    assert set(fig10_rows) == set(fig8_rows)
    assert float(fig8_rows["no backup"]["normalized_wct"]) == 1.0
    assert fig8_rows["uniform Host"]["source"] == "sglsim_multiregime_replay_policy_counterfactual"
    assert fig8_rows["uniform Host"]["evidence_tier"] == "sglsim-runtime-replay-calibrated"
    assert float(fig9_rows["no backup"]["retry_latency_norm"]) == 1.0
    assert float(fig10_rows["no backup"]["recompute"]) == 1.0
    assert float(fig10_rows["uniform Host"]["host_h2d"]) == 1.0
    assert float(fig10_rows["uniform GPU"]["peer_gpu"]) > 0.0
