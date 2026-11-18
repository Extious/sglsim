from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

from benchmark.agentic_workflows.config import (
    DEFAULT_BASE_CONFIG_DIR,
    get_scenario,
    materialize_synthesis_config,
    SCENARIOS,
)
from benchmark.agentic_workflows.aggregate import aggregate_csv, aggregate_rows
from benchmark.agentic_workflows.run import execute_scenario, main as online_main
from benchmark.agentic_workflows.combine import combine_summary_files
from benchmark.agentic_workflows.paper import write_paper_candidates
from benchmark.agentic_workflows.summary import summarize_run_dir, write_summary_csv
from benchmark.agentic_workflows.failure import failure_plan_from_scenario
from benchmark.agentic_workflows.online import build_named_online_plan, main as online_plan_main
from benchmark.agentic_workflows.workload import (
    agent_role_for_stage,
    summarize_workflow_shape,
)


ROOT = Path(__file__).resolve().parents[3]
AEGIS_ROOT = ROOT / "benchmark" / "AegisKV"
AGENTIC_SUITE_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "run_agentic_workflow_suite.py"
)
AGENTIC_DP_SCALE_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_agentic_dp_scale_from_suite.py"
)
AGENTIC_RECOVERY_PATH_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_agentic_recovery_path_from_suite.py"
)
AGENTIC_REFRESH_SUITE_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "refresh_agentic_workflow_suite_summaries.py"
)
AGENTIC_ABLATION_SCRIPT = (
    AEGIS_ROOT
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "derive_agentic_ablation_from_suite.py"
)


def load_agentic_suite_module():
    spec = importlib.util.spec_from_file_location("aegiskv_agentic_suite", AGENTIC_SUITE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_agentic_dp_scale_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_agentic_dp_scale", AGENTIC_DP_SCALE_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_agentic_recovery_path_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_agentic_recovery_path", AGENTIC_RECOVERY_PATH_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_agentic_refresh_suite_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_agentic_refresh_suite", AGENTIC_REFRESH_SUITE_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_agentic_ablation_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_agentic_ablation", AGENTIC_ABLATION_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_materialize_synthesis_config_projects_agentic_scenario(tmp_path):
    scenario = get_scenario("hard_failover_nonpeer")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=2,
    )

    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))
    server = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))

    assert client["gsp_num_groups"] == scenario.groups
    assert client["gsp_num_turns"] == scenario.turns
    assert client["gsp_failover_replay"] is True
    assert client["gsp_failover_retry_rank_offset"] == 2
    assert client["gsp_failover_retry_policy"] == scenario.failover_retry_policy
    assert client["gsp_aegis_ablation"] == scenario.aegis_ablation
    assert client["seed"] == scenario.seed + 2
    assert server["scheduler"]["dp_size"] == 4
    assert server["port"] == scenario.start_port + 2
    assert server["experiment_name"] == "agentic-hard_failover_nonpeer-t002"


def test_materialize_budget_sweep_projects_budget_and_failure_timing(tmp_path):
    scenario = get_scenario("budget_timing_late_b100")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=1,
    )

    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))

    assert client["gsp_aegis_gpu_budget_fraction"] == 1.0
    assert client["gsp_failure_after_output_tokens"] == 64
    assert client["gsp_failover_retry_policy"] == "peer"
    assert client["seed"] == scenario.seed + 1


def test_materialize_staged_13131_scenario_projects_shape(tmp_path):
    scenario = get_scenario("staged_13131_motivation")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=0,
    )

    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))

    assert scenario.workflow_shape == (1, 3, 1, 3, 1)
    assert scenario.workflow_count == scenario.groups * 9
    assert scenario.max_concurrency >= scenario.groups * max(scenario.workflow_shape)
    assert client["gsp_workflow_shape"] == [1, 3, 1, 3, 1]
    assert client["gsp_prompts_per_group"] == 9
    assert client["max_concurrency"] == scenario.max_concurrency
    assert client["gsp_stage_long_decode_agent_index"] == 0
    assert client["gsp_long_decode_output_multiplier"] == 5.0
    assert client["gsp_failure_after_output_tokens"] == 32
    assert client["gsp_aegis_policy"] == "aegiskv"


def test_motivation_main_subagent_dp8_failure_plan(tmp_path):
    scenario = get_scenario("motivation_main_subagent_dp8_failure")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=0,
    )

    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))
    server = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))
    workload = summarize_workflow_shape(scenario)
    failure = failure_plan_from_scenario(scenario)
    plan = build_named_online_plan(
        "motivation_main_subagent_dp8_failure",
        output_dir=tmp_path / "plan",
        trial=0,
    )

    assert server["scheduler"]["dp_size"] == 8
    assert server["cache"]["l1_capacity_pages"] == 393216
    assert server["context_length"] == 196608
    assert client["gsp_failover_replay"] is True
    assert client["gsp_failover_replay_critical_only"] is False
    assert client["gsp_failure_after_output_tokens"] == 128
    assert client["gsp_failure_injection_count"] == 24
    assert client["gsp_failure_target_stage_id"] == 4
    assert client["gsp_failure_target_stage_local_agent_index"] == 0
    assert client["gsp_failure_target_turn_idx"] == 0
    assert client["gsp_backup_kv"] is False
    assert client["gsp_stage_context_lens"] == [40960, 40960, 81920, 40960, 122880]
    assert workload.workflow_shape == "1-3-1-3-1"
    assert workload.workflow_count == scenario.groups
    assert workload.total_agentic_requests == scenario.workflow_count
    assert workload.main_agents_per_workflow == 3
    assert workload.subagents_per_workflow == 6
    assert workload.estimated_first_turn_prompt_tokens == 40960
    assert workload.estimated_final_turn_context_tokens == 122880
    assert failure.controlled_by_injection_count
    assert failure.injection_count == 24
    assert failure.target_stage_id == 4
    assert failure.target_stage_local_agent_index == 0
    assert failure.target_turn_idx == 0
    assert failure.retry_policy == "peer"
    assert agent_role_for_stage(stage_width=1) == "main_agent"
    assert agent_role_for_stage(stage_width=3) == "subagent"
    assert "request_details.csv" in plan.result_artifacts
    assert plan.synthesis_entrypoint == "benchmark.synthesis_requests.run.run_experiment"


def test_online_plan_cli_writes_plan_json(tmp_path, capsys):
    rc = online_plan_main(
        [
            "--scenario",
            "motivation_main_subagent_dp8_failure",
            "--output-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert (tmp_path / "online_experiment_plan.json").exists()
    assert payload["scenario"] == "motivation_main_subagent_dp8_failure"
    assert payload["failure"]["injection_count"] == 24
    assert payload["failure"]["target_stage_id"] == 4


def test_dp_failure_sweep_state_reserves_per_batch_workflow_once():
    from benchmark.synthesis_requests.client import DpFailureSweepState

    state = DpFailureSweepState(injections_per_batch=4, batch_size=32)
    assert state.try_reserve("wf-0", stage_id=1, stage_local_agent_index=1) is True
    assert state.try_reserve("wf-0", stage_id=1, stage_local_agent_index=1) is False
    assert state.try_reserve("wf-31", stage_id=1, stage_local_agent_index=2) is True
    assert state.try_reserve("wf-32", stage_id=3, stage_local_agent_index=0) is True
    assert state.try_reserve("wf-63", stage_id=4, stage_local_agent_index=0) is True
    assert state.completed() is False
    assert state.try_reserve("wf-1", stage_id=2, stage_local_agent_index=0) is True
    assert state.try_reserve("wf-2", stage_id=3, stage_local_agent_index=1) is True
    assert state.try_reserve("wf-33", stage_id=3, stage_local_agent_index=2) is True
    assert state.try_reserve("wf-34", stage_id=0, stage_local_agent_index=0) is True
    assert state.completed() is True
    assert state.try_reserve("wf-3", stage_id=1, stage_local_agent_index=0) is False


def test_motivation_dp_sweep_scenario_materialize(tmp_path):
    scenario = get_scenario("motivation_main_subagent_dp8_dp_sweep")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=0,
    )
    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))

    assert scenario.groups == 64
    assert scenario.max_workflow_concurrency == 32
    assert scenario.dp_failure_sweep is True
    assert scenario.dp_failure_sweep_injections_per_batch == 4
    assert client["max_concurrency"] is None
    assert client["gsp_max_workflow_concurrency"] == 32
    assert client["gsp_dp_failure_sweep"] is True
    assert client["gsp_dp_failure_sweep_injections_per_batch"] == 4
    assert client["gsp_dp_failure_sweep_batch_size"] == 32
    assert client["gsp_failure_injection_count"] is None
    assert client["gsp_failover_replay"] is True


def test_motivation_dp_sweep_round2_scenario_materialize(tmp_path):
    scenario = get_scenario("motivation_main_subagent_dp8_dp_sweep_round2")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=0,
    )
    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))

    assert scenario.dp_failure_sweep is True
    assert scenario.dp_failure_sweep_round1_enabled is False
    assert client["gsp_dp_failure_sweep_round1_enabled"] is False
    assert len(client["gsp_dp_failure_sweep_precovered_agent_slots"]) == 8
    assert len(client["gsp_dp_failure_sweep_precovered_workflows"]) == 8


def test_motivation_agent_sweep_scenario_materialize(tmp_path):
    scenario = get_scenario("motivation_main_subagent_dp8_agent_sweep")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=0,
    )
    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))

    assert scenario.groups == 24
    assert scenario.max_workflow_concurrency == 24
    assert scenario.agent_slot_failure_sweep is True
    assert client["max_concurrency"] is None
    assert client["gsp_max_workflow_concurrency"] == 24
    assert client["gsp_agent_slot_failure_sweep"] is True
    assert client["gsp_failure_injection_count"] is None
    assert client["gsp_failover_replay"] is True


def test_motivation_main_and_subagent_failure_scenarios_are_comparable():
    names = [
        "motivation_main_subagent_dp8_no_failure",
        "motivation_main_subagent_dp8_failure",
        "motivation_main_subagent_dp8_subagent_failure",
        "motivation_main_subagent_dp8_noncritical_subagent_failure",
        "motivation_main_subagent_dp8_s1_a2_failure",
        "motivation_main_subagent_dp8_s3_a0_failure",
        "motivation_main_subagent_dp8_s3_a1_failure",
        "motivation_main_subagent_dp8_s3_a2_failure",
    ]
    scenarios = [get_scenario(name) for name in names]
    signatures = {
        (
            scenario.groups,
            scenario.prompts_per_group,
            scenario.turns,
            scenario.workflow_shape,
            scenario.system_prompt_len,
            scenario.question_len,
            scenario.output_len,
            scenario.dp_size,
            scenario.seed,
            scenario.stage_context_lens,
        )
        for scenario in scenarios
    }

    assert len(signatures) == 1
    assert scenarios[0].groups == 24
    assert scenarios[0].max_concurrency == 24
    assert scenarios[0].max_workflow_concurrency == 24
    assert scenarios[0].seed == 9251
    assert scenarios[0].failover_replay is False
    assert scenarios[0].failure_injection_count == 0
    assert scenarios[1].failure_injection_count == 24
    assert scenarios[1].failure_target_stage_id == 4
    assert scenarios[1].failure_target_stage_local_agent_index == 0
    assert scenarios[2].failure_target_stage_id == 1
    assert scenarios[2].failure_target_stage_local_agent_index == 0
    assert scenarios[3].failure_target_stage_id == 1
    assert scenarios[3].failure_target_stage_local_agent_index == 1
    subagent_targets = {
        (
            scenario.failure_target_stage_id,
            scenario.failure_target_stage_local_agent_index,
        )
        for scenario in scenarios[2:]
    }
    assert subagent_targets == {
        (1, 0),
        (1, 1),
        (1, 2),
        (3, 0),
        (3, 1),
        (3, 2),
    }


def test_online_alpha_dp8_scenarios_cover_load_and_failure_states(tmp_path):
    names = [
        "online_alpha_dp8_baseload_lowrisk",
        "online_alpha_dp8_baseload_highrisk",
        "online_alpha_dp8_highload_lowrisk",
        "online_alpha_dp8_highload_highrisk",
    ]
    scenarios = [get_scenario(name) for name in names]

    assert {scenario.groups for scenario in scenarios} == {2, 4}
    assert {scenario.max_concurrency for scenario in scenarios} == {6, 12}
    assert {scenario.failure_rate_per_window for scenario in scenarios} == {0.003, 0.03}
    assert {scenario.aegis_critical_pressure for scenario in scenarios} == {0.62, 0.86}
    assert all(scenario.dp_size == 8 for scenario in scenarios)
    assert all(scenario.aegis_policy == "aegiskv" for scenario in scenarios)
    assert all(scenario.alpha_policy_path for scenario in scenarios)
    assert all(
        scenario.stage_context_lens == (40960, 40960, 81920, 40960, 122880)
        for scenario in scenarios
    )
    assert all(scenario.context_length == 196608 for scenario in scenarios)

    config_dir = materialize_synthesis_config(
        scenarios[0],
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=0,
    )
    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))
    assert client["gsp_alpha_policy_path"] == scenarios[0].alpha_policy_path
    assert client["gsp_failure_rate_per_window"] == 0.003
    assert client["gsp_aegis_critical_pressure"] == 0.62


def test_staged_13131_stress_policy_scenarios_are_comparable_and_soft(tmp_path):
    policy_names = [
        "staged_13131_policy_stress_no_backup",
        "staged_13131_policy_stress_host_only",
        "staged_13131_policy_stress_gpu_replica",
        "staged_13131_policy_stress_aegiskv",
    ]
    signatures = {
        (
            SCENARIOS[name].groups,
            SCENARIOS[name].prompts_per_group,
            SCENARIOS[name].turns,
            SCENARIOS[name].system_prompt_len,
            SCENARIOS[name].question_len,
            SCENARIOS[name].output_len,
            SCENARIOS[name].dp_size,
            SCENARIOS[name].critical_agent_ratio,
            SCENARIOS[name].failover_injection,
            SCENARIOS[name].failover_retry_rank_offset,
            SCENARIOS[name].seed,
            SCENARIOS[name].failure_after_output_tokens,
            SCENARIOS[name].workflow_shape,
            SCENARIOS[name].stage_context_lens,
            SCENARIOS[name].long_decode_output_multiplier,
        )
        for name in policy_names
    }
    policies = {SCENARIOS[name].aegis_policy for name in policy_names}

    assert len(signatures) == 1
    assert policies == {"no_backup", "host_only", "gpu_replica", "aegiskv"}
    assert SCENARIOS["staged_13131_policy_stress_no_backup"].backup_kv is False
    assert SCENARIOS["staged_13131_policy_stress_aegiskv"].backup_kv is True
    assert SCENARIOS["staged_13131_policy_stress_aegiskv"].failover_injection == "soft"
    assert SCENARIOS["staged_13131_policy_stress_aegiskv"].failure_after_output_tokens == 128

    scenario = get_scenario("staged_13131_policy_stress_aegiskv")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=1,
    )
    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))

    assert client["gsp_failover_injection"] == "soft"
    assert client["gsp_failure_after_output_tokens"] == 128
    assert client["gsp_long_decode_output_multiplier"] == 8.0
    assert client["seed"] == scenario.seed + 1


def test_staged_13131_lite_policy_scenarios_pin_failure_timing(tmp_path):
    policy_names = [
        "staged_13131_policy_lite_no_backup",
        "staged_13131_policy_lite_host_only",
        "staged_13131_policy_lite_gpu_replica",
        "staged_13131_policy_lite_aegiskv",
    ]
    signatures = {
        (
            SCENARIOS[name].groups,
            SCENARIOS[name].prompts_per_group,
            SCENARIOS[name].turns,
            SCENARIOS[name].system_prompt_len,
            SCENARIOS[name].question_len,
            SCENARIOS[name].output_len,
            SCENARIOS[name].dp_size,
            SCENARIOS[name].critical_agent_ratio,
            SCENARIOS[name].failover_injection,
            SCENARIOS[name].failover_retry_rank_offset,
            SCENARIOS[name].seed,
            SCENARIOS[name].failure_after_output_tokens,
            SCENARIOS[name].workflow_shape,
            SCENARIOS[name].stage_context_lens,
            SCENARIOS[name].long_decode_output_multiplier,
        )
        for name in policy_names
    }
    policies = {SCENARIOS[name].aegis_policy for name in policy_names}

    assert len(signatures) == 1
    assert policies == {"no_backup", "host_only", "gpu_replica", "aegiskv"}
    assert SCENARIOS["staged_13131_policy_lite_no_backup"].backup_kv is False
    assert SCENARIOS["staged_13131_policy_lite_aegiskv"].backup_kv is True
    assert SCENARIOS["staged_13131_policy_lite_aegiskv"].failure_after_output_tokens == 64

    scenario = get_scenario("staged_13131_policy_lite_aegiskv")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=2,
    )
    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))

    assert client["gsp_failure_after_output_tokens"] == 64
    assert client["gsp_long_decode_output_multiplier"] == 5.0
    assert client["seed"] == scenario.seed + 2


def test_staged_13131_100k_context_policy_scenarios_are_comparable(tmp_path):
    policy_names = [
        "staged_13131_100k_context_no_backup",
        "staged_13131_100k_context_host_only",
        "staged_13131_100k_context_gpu_replica",
        "staged_13131_100k_context_aegiskv",
    ]
    signatures = {
        (
            SCENARIOS[name].groups,
            SCENARIOS[name].prompts_per_group,
            SCENARIOS[name].turns,
            SCENARIOS[name].system_prompt_len,
            SCENARIOS[name].question_len,
            SCENARIOS[name].output_len,
            SCENARIOS[name].dp_size,
            SCENARIOS[name].critical_agent_ratio,
            SCENARIOS[name].failover_injection,
            SCENARIOS[name].failover_retry_rank_offset,
            SCENARIOS[name].seed,
            SCENARIOS[name].failure_after_output_tokens,
            SCENARIOS[name].workflow_shape,
            SCENARIOS[name].long_decode_output_multiplier,
        )
        for name in policy_names
    }
    policies = {SCENARIOS[name].aegis_policy for name in policy_names}

    assert len(signatures) == 1
    assert policies == {"no_backup", "host_only", "gpu_replica", "aegiskv"}
    scenario = get_scenario("staged_13131_100k_context_aegiskv")
    assert scenario.workflow_shape == (1, 3, 1, 3, 1)
    assert scenario.workflow_count == 18
    assert scenario.dp_size == 8
    assert scenario.max_concurrency == 6
    assert scenario.stage_context_lens == (40960, 40960, 81920, 40960, 122880)
    assert scenario.estimated_first_turn_prompt_tokens == 40960
    assert scenario.estimated_final_turn_context_tokens == 122880
    assert scenario.estimated_long_decode_output_tokens == 1024
    assert scenario.backup_kv is True
    assert scenario.failure_rate_per_window == 0.01
    assert scenario.alpha_policy_path

    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=0,
    )
    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))

    assert client["gsp_workflow_shape"] == [1, 3, 1, 3, 1]
    assert client["gsp_stage_context_lens"] == [40960, 40960, 81920, 40960, 122880]
    assert client["gsp_system_prompt_len"] == 98304
    assert client["gsp_question_len"] == 1024
    assert client["gsp_output_len"] == 128
    assert client["gsp_long_decode_output_multiplier"] == 8.0
    assert client["gsp_failover_injection"] == "soft"
    assert client["gsp_failure_after_output_tokens"] == 128
    assert client["gsp_aegis_policy"] == "aegiskv"
    assert client["gsp_alpha_policy_path"] == scenario.alpha_policy_path


def test_alpha_tradeoff_dp8_scenarios_cover_load_failure_and_alpha(tmp_path):
    names = {
        name
        for name in SCENARIOS
        if name.startswith("alpha_tradeoff_dp8_")
    }

    assert len(names) == 48
    alphas = {round(float(SCENARIOS[name].aegis_gpu_budget_fraction), 2) for name in names}
    injection_counts = {SCENARIOS[name].failure_injection_count for name in names}
    group_counts = {SCENARIOS[name].groups for name in names}
    ports = {SCENARIOS[name].start_port for name in names}

    assert alphas == {
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    }
    assert injection_counts == {1, 3}
    assert group_counts == {2, 4}
    assert len(ports) == len(names)

    scenario = get_scenario("alpha_tradeoff_dp8_highload_fail3_a030")
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=DEFAULT_BASE_CONFIG_DIR,
        output_dir=tmp_path,
        trial=0,
    )
    client = json.loads((config_dir / "client.json").read_text(encoding="utf-8"))
    server = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))

    assert scenario.dp_size == 8
    assert scenario.aegis_policy == "gpu_replica"
    assert scenario.backup_kv is True
    assert scenario.replay_critical_only is True
    assert scenario.failure_injection_count == 3
    assert scenario.groups == 4
    assert scenario.max_concurrency == 12
    assert scenario.estimated_final_turn_context_tokens == 122880
    assert client["gsp_stage_context_lens"] == [40960, 40960, 81920, 40960, 122880]
    assert client["gsp_aegis_gpu_budget_fraction"] == 0.3
    assert client["gsp_failure_injection_count"] == 3
    assert client["gsp_failover_replay_critical_only"] is True
    assert server["scheduler"]["dp_size"] == 8
    assert server["cache"]["aegis_total_l1_capacity_pages"] == 786432
    assert server["cache"]["l1_capacity_pages"] == 550502
    assert server["context_length"] == 196608


def test_hot_tail_peer_scenario_uses_short_prefixes():
    scenario = get_scenario("hard_failover_peer_hot_tail")

    assert scenario.dp_size == 4
    assert scenario.failover_retry_rank_offset == 1
    assert scenario.system_prompt_len < 128
    assert scenario.question_len < 128
    assert scenario.output_len >= 64


def test_policy_sweep_scenarios_share_workload_shape():
    policy_names = [
        "policy_no_backup",
        "policy_host_only",
        "policy_gpu_replica",
        "policy_aegiskv",
    ]
    signatures = {
        (
            SCENARIOS[name].groups,
            SCENARIOS[name].prompts_per_group,
            SCENARIOS[name].turns,
            SCENARIOS[name].system_prompt_len,
            SCENARIOS[name].question_len,
            SCENARIOS[name].output_len,
            SCENARIOS[name].dp_size,
            SCENARIOS[name].critical_agent_ratio,
            SCENARIOS[name].failover_injection,
            SCENARIOS[name].failover_retry_rank_offset,
            SCENARIOS[name].seed,
        )
        for name in policy_names
    }

    assert len(signatures) == 1


def test_peer_route_scenarios_share_workload_shape_and_vary_route_policy():
    scenario_names = [
        "peer_route_nonpeer_hot_tail",
        "peer_route_random_hot_tail",
        "peer_route_aegiskv_hot_tail",
    ]
    signatures = {
        (
            SCENARIOS[name].groups,
            SCENARIOS[name].prompts_per_group,
            SCENARIOS[name].turns,
            SCENARIOS[name].system_prompt_len,
            SCENARIOS[name].question_len,
            SCENARIOS[name].output_len,
            SCENARIOS[name].dp_size,
            SCENARIOS[name].critical_agent_ratio,
            SCENARIOS[name].failover_injection,
            SCENARIOS[name].seed,
        )
        for name in scenario_names
    }
    policies = {SCENARIOS[name].failover_retry_policy for name in scenario_names}

    assert len(signatures) == 1
    assert policies == {"non_peer", "random_healthy", "peer"}


def test_ablation_scenarios_share_workload_shape_and_vary_ablation():
    scenario_names = [
        "ablation_full_aegiskv",
        "ablation_no_lifecycle",
        "ablation_no_segment_granularity",
        "ablation_no_restore_recompute",
        "ablation_no_pressure",
        "ablation_no_hysteresis",
        "ablation_no_critical_restore",
        "ablation_static_alpha",
        "ablation_size_only",
        "ablation_no_peer_locality",
        "ablation_no_marginal_accounting",
    ]
    signatures = {
        (
            SCENARIOS[name].groups,
            SCENARIOS[name].prompts_per_group,
            SCENARIOS[name].turns,
            SCENARIOS[name].system_prompt_len,
            SCENARIOS[name].question_len,
            SCENARIOS[name].output_len,
            SCENARIOS[name].dp_size,
            SCENARIOS[name].critical_agent_ratio,
            SCENARIOS[name].failover_injection,
            SCENARIOS[name].seed,
        )
        for name in scenario_names
    }
    ablations = {SCENARIOS[name].aegis_ablation for name in scenario_names}
    failure_after = {SCENARIOS[name].failure_after_output_tokens for name in scenario_names}

    assert len(signatures) == 1
    assert failure_after == {64}
    assert ablations == {
        "",
        "no_lifecycle",
        "no_segment_granularity",
        "no_restore_recompute",
        "no_pressure",
        "no_hysteresis",
        "no_critical_restore",
        "static_alpha",
        "size_only_backup",
        "no_peer_locality",
        "no_marginal_accounting",
    }


def test_budget_sweep_scenarios_cover_workloads_timings_and_budgets():
    budget_suffixes = {"b000", "b010", "b020", "b050", "b100"}
    for prefix in ("budget_hot_tail", "budget_long_prefix", "budget_nonpeer"):
        names = {name for name in SCENARIOS if name.startswith(prefix + "_")}
        assert names == {f"{prefix}_{suffix}" for suffix in budget_suffixes}
    for prefix in ("budget_timing_early", "budget_timing_late"):
        names = {name for name in SCENARIOS if name.startswith(prefix + "_")}
        assert names == {f"{prefix}_{suffix}" for suffix in budget_suffixes}

    assert SCENARIOS["budget_nonpeer_b100"].failover_retry_policy == "non_peer"
    assert SCENARIOS["budget_timing_early_b000"].failure_after_output_tokens == 1
    assert SCENARIOS["budget_timing_late_b000"].failure_after_output_tokens == 64


def test_online_dry_run_writes_configs_and_manifest(tmp_path):
    manifest = execute_scenario(
        scenario_name="smoke",
        output_root=tmp_path,
        run_id="unit-dry",
        runs=2,
        dry_run=True,
    )

    assert manifest["dry_run"] is True
    assert len(manifest["plan"]) == 2
    for row in manifest["plan"]:
        assert Path(row["config_dir"], "client.json").exists()
        assert Path(row["config_dir"], "server.json").exists()
        assert Path(row["online_plan"]).exists()
    assert (tmp_path / "unit-dry" / "agentic_workflow_manifest.json").exists()


def test_online_lists_scenarios(capsys):
    assert online_main(["--list-scenarios"]) == 0
    captured = capsys.readouterr()
    assert "hard_failover_host" in captured.out
    assert "smoke" in captured.out


def _write_fake_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "completed": 3,
                "failed": 0,
                "mean_e2e_latency_ms": 10.0,
                "p90_e2e_latency_ms": 12.0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "simulator_config.json").write_text(
        json.dumps(
            {
                "experiment": {"name": "unit-agentic"},
                "scheduler": {"dp_size": 2},
                "workload": {
                    "seed": 1234,
                    "num_requests": 2,
                    "gsp_num_turns": 2,
                    "gsp_num_groups": 1,
                    "gsp_prompts_per_group": 2,
                    "gsp_system_prompt_len": 64,
                    "gsp_question_len": 16,
                    "gsp_output_len": 8,
                    "gsp_range_ratio": 0.75,
                    "max_concurrency": 1,
                    "gsp_critical_agent_ratio": 0.5,
                    "gsp_failover_replay": True,
                    "gsp_failover_injection": "hard-restart",
                    "gsp_failover_retry_rank_offset": 1,
                    "gsp_failover_retry_policy": "peer",
                    "gsp_aegis_policy": "host_only",
                    "gsp_aegis_ablation": "no_lifecycle",
                    "gsp_aegis_gpu_budget_fraction": 0.5,
                    "gsp_failure_after_output_tokens": 4,
                    "gsp_backup_kv": True,
                },
            }
        ),
        encoding="utf-8",
    )
    fields = [
        "rid",
        "critical_agent",
        "client_wall_e2e_latency_s",
        "aegis_candidate_retry",
        "aegis_failover_replay",
        "aegis_cross_rank_retry",
        "aegis_hard_failure_injected",
        "aegis_host_backup_lookup_hit",
        "aegis_peer_gpu_lookup_hit",
        "aegis_replay_restored_tokens",
        "aegis_replay_restored_prefill_tokens",
        "aegis_replay_recompute_tokens",
        "aegis_replay_restore_action",
        "aegis_host_backup_lookup_tokens",
        "aegis_peer_gpu_lookup_tokens",
        "aegis_replay_restore_sleep_s",
    ]
    rows = [
        {
            "rid": "1-t0",
            "critical_agent": "True",
            "client_wall_e2e_latency_s": "0.010",
            "aegis_candidate_retry": "True",
            "aegis_failover_replay": "False",
            "aegis_cross_rank_retry": "False",
            "aegis_hard_failure_injected": "False",
            "aegis_host_backup_lookup_hit": "False",
            "aegis_peer_gpu_lookup_hit": "False",
            "aegis_replay_restored_tokens": "0",
            "aegis_replay_restored_prefill_tokens": "0",
            "aegis_replay_recompute_tokens": "0",
            "aegis_replay_restore_action": "none",
            "aegis_host_backup_lookup_tokens": "0",
            "aegis_peer_gpu_lookup_tokens": "0",
            "aegis_replay_restore_sleep_s": "0",
        },
        {
            "rid": "1-t0-retry",
            "critical_agent": "True",
            "client_wall_e2e_latency_s": "0.020",
            "aegis_candidate_retry": "False",
            "aegis_failover_replay": "True",
            "aegis_cross_rank_retry": "True",
            "aegis_hard_failure_injected": "True",
            "aegis_host_backup_lookup_hit": "True",
            "aegis_peer_gpu_lookup_hit": "False",
            "aegis_replay_restored_tokens": "128",
            "aegis_replay_restored_prefill_tokens": "120",
            "aegis_replay_recompute_tokens": "8",
            "aegis_replay_restore_action": "host_restore",
            "aegis_host_backup_lookup_tokens": "128",
            "aegis_peer_gpu_lookup_tokens": "0",
            "aegis_replay_restore_sleep_s": "0.003",
        },
        {
            "rid": "2-t0",
            "critical_agent": "False",
            "client_wall_e2e_latency_s": "0.015",
            "aegis_candidate_retry": "False",
            "aegis_failover_replay": "False",
            "aegis_cross_rank_retry": "False",
            "aegis_hard_failure_injected": "False",
            "aegis_host_backup_lookup_hit": "False",
            "aegis_peer_gpu_lookup_hit": "False",
            "aegis_replay_restored_tokens": "0",
            "aegis_replay_restored_prefill_tokens": "0",
            "aegis_replay_recompute_tokens": "0",
            "aegis_replay_restore_action": "none",
            "aegis_host_backup_lookup_tokens": "0",
            "aegis_peer_gpu_lookup_tokens": "0",
            "aegis_replay_restore_sleep_s": "0",
        },
    ]
    with (run_dir / "request_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_run_dir_extracts_aegis_replay_metrics(tmp_path):
    run_dir = tmp_path / "run"
    _write_fake_run(run_dir)

    summary = summarize_run_dir(run_dir, scenario="unit")

    assert summary.scenario == "unit"
    assert summary.seed == 1234
    assert summary.completed == 3
    assert summary.workflow_count == 1
    assert summary.total_agentic_requests == 2
    assert summary.system_prompt_len == 64
    assert summary.question_len == 16
    assert summary.aegis_policy == "host_only"
    assert summary.aegis_ablation == "no_lifecycle"
    assert summary.aegis_gpu_budget_fraction == 0.5
    assert summary.failure_after_output_tokens == 4
    assert summary.failover_retry_policy == "peer"
    assert summary.backup_kv == "True"
    assert summary.replay_rows == 1
    assert summary.candidate_retry_rows == 1
    assert summary.cross_rank_replay_rows == 1
    assert summary.host_lookup_hits == 1
    assert summary.restored_tokens == 128
    assert summary.recompute_tokens == 8
    assert summary.replay_action_host_restore == 1
    assert summary.replay_tokens_host_restore == 128
    assert summary.replay_tokens_recompute == 8
    assert summary.replay_action_peer_gpu_restore == 0
    assert summary.mean_restore_sleep_ms == 3.0


def test_plot_scripts_combine_and_project_summary_rows(tmp_path):
    run_dir = tmp_path / "run"
    _write_fake_run(run_dir)
    summary = summarize_run_dir(run_dir, scenario="unit")
    summary_csv = tmp_path / "summaries" / "agentic_workflow_summary.csv"
    write_summary_csv(summary_csv, [summary])

    combined_csv = tmp_path / "combined.csv"
    assert combine_summary_files([summary_csv], combined_csv) == 1
    paper_csv = tmp_path / "paper.csv"
    assert write_paper_candidates(combined_csv, paper_csv) == 1

    rows = list(csv.DictReader(paper_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["scenario"] == "unit"
    assert rows[0]["seed"] == "1234"
    assert rows[0]["evidence_tier"] == "sglsim-agentic-online-preliminary"
    assert rows[0]["source_run_dir"] == str(run_dir)
    assert rows[0]["workflow_count"] == "1"
    assert rows[0]["total_agentic_requests"] == "2"
    assert rows[0]["system_prompt_len"] == "64"
    assert rows[0]["failover_retry_policy"] == "peer"
    assert rows[0]["aegis_policy"] == "host_only"
    assert rows[0]["aegis_ablation"] == "no_lifecycle"
    assert rows[0]["aegis_gpu_budget_fraction"] == "0.5"
    assert rows[0]["failure_after_output_tokens"] == "4"
    assert rows[0]["replay_action_host_restore"] == "1"
    assert rows[0]["replay_tokens_host_restore"] == "128"


def test_aggregate_paper_candidates_computes_ci(tmp_path):
    rows = [
        {
            "scenario": "unit",
            "dp_size": "2",
            "workflow_count": "4",
            "total_agentic_requests": "8",
            "turns": "2",
            "completed": "8",
            "failed": "0",
            "replay_rows": "1",
            "host_lookup_hits": "1",
            "peer_lookup_hits": "0",
            "restored_tokens": "100",
            "recompute_tokens": "0",
            "mean_e2e_ms": "10",
            "p90_e2e_ms": "12",
            "mean_restore_sleep_ms": "3",
            "evidence_tier": "sglsim-agentic-online-preliminary",
            "source_run_dir": "run-a",
        },
        {
            "scenario": "unit",
            "dp_size": "2",
            "workflow_count": "4",
            "total_agentic_requests": "8",
            "turns": "2",
            "completed": "8",
            "failed": "0",
            "replay_rows": "1",
            "host_lookup_hits": "1",
            "peer_lookup_hits": "0",
            "restored_tokens": "140",
            "recompute_tokens": "0",
            "mean_e2e_ms": "14",
            "p90_e2e_ms": "16",
            "mean_restore_sleep_ms": "5",
            "evidence_tier": "sglsim-agentic-online-preliminary",
            "source_run_dir": "run-b",
        },
    ]

    aggregate = aggregate_rows(rows)

    assert len(aggregate) == 1
    assert aggregate[0]["trials"] == 2
    assert aggregate[0]["completed_total"] == 16
    assert aggregate[0]["restored_tokens_mean"] == 120
    assert aggregate[0]["replay_tokens_host_restore_total"] == 0
    assert float(aggregate[0]["mean_e2e_ms_ci95"]) > 0

    paper_csv = tmp_path / "paper.csv"
    with paper_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    aggregate_csv_path = tmp_path / "aggregate.csv"
    assert aggregate_csv(paper_csv, aggregate_csv_path) == 1


def test_agentic_dp_scale_derives_peer_and_nonpeer_series():
    module = load_agentic_dp_scale_module()
    rows = [
        {
            "scenario": "hard_failover_host",
            "dp_size": "2",
            "workflow_count": "32",
            "mean_e2e_ms": "100",
            "replay_rows": "24",
            "source_run_dir": "host-a",
        },
        {
            "scenario": "hard_failover_host",
            "dp_size": "2",
            "workflow_count": "32",
            "mean_e2e_ms": "110",
            "replay_rows": "24",
            "source_run_dir": "host-b",
        },
        {
            "scenario": "hard_failover_peer",
            "dp_size": "4",
            "workflow_count": "32",
            "mean_e2e_ms": "99",
            "replay_rows": "24",
            "source_run_dir": "peer-a",
        },
        {
            "scenario": "hard_failover_nonpeer",
            "dp_size": "4",
            "workflow_count": "32",
            "mean_e2e_ms": "108",
            "replay_rows": "24",
            "source_run_dir": "nonpeer-a",
        },
    ]

    derived = module.derive_rows(rows, run_id="unit")

    assert [row["series"] for row in derived] == [
        "agentic DP4 peer",
        "agentic DP4 peer",
        "agentic DP4 non-peer",
        "agentic DP4 non-peer",
    ]
    peer_dp4 = derived[1]
    nonpeer_dp4 = derived[3]
    assert peer_dp4["dp_size"] == "4"
    assert peer_dp4["normalized_wct"] == 99 / 105
    assert nonpeer_dp4["normalized_wct"] == 108 / 105
    assert peer_dp4["evidence_tier"] == "sglsim-agentic-online-preliminary"


def test_agentic_recovery_path_derives_observed_actions_from_run_dirs(tmp_path):
    module = load_agentic_recovery_path_module()
    scenario_rows = []
    for scenario, dp_size in [
        ("hard_failover_host", "2"),
        ("hard_failover_peer", "4"),
        ("hard_failover_nonpeer", "4"),
    ]:
        run_dir = tmp_path / scenario
        _write_fake_run(run_dir)
        scenario_rows.append(
            {
                "scenario": scenario,
                "dp_size": dp_size,
                "workflow_count": "2",
                "replay_rows": "1",
                "source_run_dir": str(run_dir),
            }
        )

    derived = module.derive_rows(scenario_rows, run_id="unit")

    assert [row["policy"] for row in derived] == [
        "online DP2 Host",
        "online DP4 peer",
        "online DP4 non-peer",
    ]
    assert all(row["host_h2d"] == 128 / 136 for row in derived)
    assert all(row["peer_gpu"] == 0.0 for row in derived)
    assert all(row["recompute"] == 8 / 136 for row in derived)
    assert derived[0]["runtime_replay_rows"] == 1
    assert derived[0]["source"] == "sglsim_agentic_online_suite_recovery_path"


def test_agentic_recovery_path_allows_single_hot_tail_suite(tmp_path):
    module = load_agentic_recovery_path_module()
    run_dir = tmp_path / "hot-tail"
    _write_fake_run(run_dir)

    derived = module.derive_rows(
        [
            {
                "scenario": "hard_failover_peer_hot_tail",
                "dp_size": "4",
                "workflow_count": "2",
                "replay_rows": "1",
                "source_run_dir": str(run_dir),
            }
        ],
        run_id="unit-hot-tail",
    )

    assert len(derived) == 1
    assert derived[0]["policy"] == "online DP4 peer hot-tail"


def test_agentic_recovery_path_merge_replaces_prior_suite_rows():
    module = load_agentic_recovery_path_module()
    existing = [
        {
            "policy": "online DP2 Host",
            "source": "sglsim_agentic_online_suite_recovery_path",
            "run_id": "suite-a",
        },
        {"policy": "AegisKV", "source": "counterfactual", "run_id": "base"},
    ]
    replacement = [
        {
            "policy": "online DP4 peer hot-tail",
            "source": "sglsim_agentic_online_suite_recovery_path",
            "run_id": "suite-b",
        }
    ]

    merged = module.merge_rows(existing, replacement)

    assert [row["policy"] for row in merged] == [
        "AegisKV",
        "online DP4 peer hot-tail",
    ]


def test_refresh_agentic_suite_summaries_from_existing_runs(tmp_path):
    module = load_agentic_refresh_suite_module()
    suite_root = tmp_path / "suite"
    scenario_root = suite_root / "00-hard_failover_host"
    run_dir = scenario_root / "runs" / "trial_000" / "run"
    _write_fake_run(run_dir)
    manifest = {
        "suite_id": "unit-suite",
        "suite_root": str(suite_root),
        "scenarios": ["hard_failover_host"],
        "manifests": [
            {
                "scenario": "hard_failover_host",
                "output_root": str(scenario_root),
                "plan": [{"run_dir": str(run_dir)}],
            }
        ],
    }
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / "agentic_suite_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    summary = module.refresh_suite(suite_root)

    assert summary["scenario_summary_rows"] == 1
    paper_csv = Path(summary["paper_candidate_csv"])
    aggregate_csv_path = Path(summary["paper_aggregate_csv"])
    paper_rows = list(csv.DictReader(paper_csv.open(newline="", encoding="utf-8")))
    aggregate_rows_out = list(
        csv.DictReader(aggregate_csv_path.open(newline="", encoding="utf-8"))
    )
    assert paper_rows[0]["replay_action_host_restore"] == "1"
    assert paper_rows[0]["replay_tokens_host_restore"] == "128"
    assert aggregate_rows_out[0]["replay_action_host_restore_total"] == "1"


def test_aegiskv_agentic_suite_script_dry_run(tmp_path):
    module = load_agentic_suite_module()
    summary = module.run_agentic_suite(
        scenarios=["smoke"],
        results_root=tmp_path,
        suite_id="unit-suite",
        runs=1,
        dry_run=True,
        log_level="warning",
    )

    assert summary["dry_run"] is True
    assert summary["summary_rows"] == 0
    assert summary["paper_aggregate_rows"] == 0
    suite_root = tmp_path / "unit-suite"
    assert (suite_root / "agentic_suite_manifest.json").exists()
    assert (suite_root / "aegiskv_agentic_suite_summary.json").exists()
    config_dir = (
        suite_root
        / "00-smoke"
        / "configs"
        / "smoke"
        / "trial_000"
    )
    assert (config_dir / "client.json").exists()
    assert (config_dir / "server.json").exists()
