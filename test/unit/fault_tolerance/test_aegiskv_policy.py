from __future__ import annotations

from sglsim.fault_tolerance.aegiskv import (
    AegisKVPolicyConfig,
    BackupLocation,
    apply_backup_policy,
    is_critical_agent_id,
    score_segment,
)
from sglsim.fault_tolerance.aegiskv_simulator import (
    MultiAgentWorkflowConfig,
    ROUND1_SUBAGENTS,
    ROUND2_SUBAGENTS,
    build_two_round_coding_workflow,
    simulate_run,
    RunSpec,
)


def test_two_round_workflow_contains_design_critical_segments():
    segments = build_two_round_coding_workflow("wf-test", MultiAgentWorkflowConfig())
    locals_ = {segment.segment_id.rsplit(":", 1)[1]: segment for segment in segments}

    assert {"P1", "P6", "S15", "S21", "P8"}.issubset(locals_)
    assert locals_["S15"].lifecycle_state.value == "ToolBlocked"
    assert locals_["S21"].role == "regression_verifier"
    assert locals_["P8"].sink_distance == 0


def test_workflow_uses_o0_and_s1_through_s6_agents():
    segments = build_two_round_coding_workflow("wf-test", MultiAgentWorkflowConfig())
    agent_ids = {segment.agent_id for segment in segments}
    assert agent_ids == {"O0", *ROUND1_SUBAGENTS, *ROUND2_SUBAGENTS}


def test_critical_agents_have_heavy_decode():
    cfg = MultiAgentWorkflowConfig()
    segments = build_two_round_coding_workflow("wf-test", cfg)
    by_agent: dict[str, list[int]] = {}
    for segment in segments:
        by_agent.setdefault(segment.agent_id, []).append(segment.output_tokens)
    assert max(by_agent["S2"]) == cfg.output_tokens_critical
    assert max(by_agent["S5"]) == cfg.output_tokens_critical
    assert max(by_agent["O0"]) == cfg.output_tokens_main
    assert by_agent["S1"][0] == cfg.output_tokens_default


def test_three_policies_backup_scope():
    segments = build_two_round_coding_workflow("wf-test", MultiAgentWorkflowConfig())

    baseline = apply_backup_policy(segments, AegisKVPolicyConfig(policy="baseline"))
    assert sum(1 for segment in baseline if segment.is_backed_up) == 0

    naive = apply_backup_policy(segments, AegisKVPolicyConfig(policy="naive_backup"))
    naive_count = sum(1 for segment in naive if segment.is_backed_up)
    assert naive_count > 0

    aegis = apply_backup_policy(segments, AegisKVPolicyConfig(policy="aegiskv"))
    aegis_count = sum(1 for segment in aegis if segment.is_backed_up)
    assert 0 < aegis_count < naive_count
    for segment in aegis:
        if segment.is_backed_up:
            assert segment.is_critical_agent


def test_aegiskv_critical_failure_beats_baseline_on_recompute():
    base = dict(
        experiment="strategy",
        failure_site="S5_ToolBlocked",
        repeat_idx=0,
        seed=100,
        backup_budget_ratio=1.0,
        peer_backup_lag_tokens=8,
        num_workflows=8,
    )
    baseline = simulate_run(RunSpec(strategy="baseline", **base)).run
    aegis = simulate_run(RunSpec(strategy="aegiskv", **base)).run
    naive = simulate_run(RunSpec(strategy="naive_backup", **base)).run

    assert aegis["total_recomputed_tokens"] < baseline["total_recomputed_tokens"]
    assert aegis["backed_up_tokens"] < naive["backed_up_tokens"]
    assert aegis["total_recomputed_tokens"] <= naive["total_recomputed_tokens"] * 1.15


def test_simulate_run_emits_workflow_and_timeline_artifacts():
    result = simulate_run(
        RunSpec(
            experiment="strategy",
            strategy="aegiskv",
            failure_site="S5_ToolBlocked",
            repeat_idx=0,
            seed=100,
            backup_budget_ratio=1.0,
            peer_backup_lag_tokens=8,
            num_workflows=4,
        )
    )
    assert len(result.workflows) == 4
    assert len(result.agent_timelines) == 4 * 9
    assert any(row["is_failure_target"] == 1 for row in result.agent_timelines)
    assert is_critical_agent_id("S2")
    assert is_critical_agent_id("S5")
