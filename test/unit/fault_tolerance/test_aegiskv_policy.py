from __future__ import annotations

from sglsim.fault_tolerance.aegiskv import (
    AegisKVPolicyConfig,
    BackupLocation,
    apply_backup_policy,
    score_segment,
)
from sglsim.fault_tolerance.aegiskv_simulator import (
    MultiAgentWorkflowConfig,
    build_two_round_coding_workflow,
    simulate_run,
    RunSpec,
)


def test_two_round_workflow_contains_design_critical_segments():
    segments = build_two_round_coding_workflow("wf-test", MultiAgentWorkflowConfig())
    locals_ = {segment.segment_id.rsplit(":", 1)[1]: segment for segment in segments}

    assert {"S1", "S6", "S15", "S21", "S8"}.issubset(locals_)
    assert locals_["S15"].lifecycle_state.value == "ToolBlocked"
    assert locals_["S21"].role == "regression_verifier"
    assert locals_["S8"].sink_distance == 0


def test_aegiskv_scores_tool_blocked_and_final_resume_above_summarized_tail():
    segments = build_two_round_coding_workflow("wf-test", MultiAgentWorkflowConfig())
    by_local = {segment.segment_id.rsplit(":", 1)[1]: segment for segment in segments}

    assert score_segment(by_local["S21"]) > score_segment(by_local["S11"])
    assert score_segment(by_local["S8"]) > score_segment(by_local["S12"])
    assert score_segment(by_local["S6"]) > score_segment(by_local["S18"])


def test_segment_policy_beats_whole_agent_backup_under_same_budget():
    base = dict(
        experiment="strategy",
        failure_site="F2_ToolBlocked",
        repeat_idx=0,
        seed=100,
        backup_budget_ratio=0.5,
        peer_backup_lag_tokens=8,
        num_workflows=8,
    )
    agent = simulate_run(RunSpec(strategy="gpu_backup_agent", **base)).run
    aegis = simulate_run(RunSpec(strategy="aegiskv_segment", **base)).run

    assert aegis["total_recomputed_tokens"] < agent["total_recomputed_tokens"]
    assert (
        aegis["critical_segment_survival_rate"]
        > agent["critical_segment_survival_rate"]
    )


def test_agent_policy_is_coarse_grained_by_agent_group():
    segments = build_two_round_coding_workflow("wf-test", MultiAgentWorkflowConfig())
    protected = apply_backup_policy(
        segments,
        AegisKVPolicyConfig(policy="gpu_backup_agent", backup_budget_ratio=0.25),
    )
    backed_agents = {
        segment.agent_id
        for segment in protected
        if segment.backup_location == BackupLocation.GPU_REPLICA
    }
    for agent_id in backed_agents:
        agent_segments = [segment for segment in protected if segment.agent_id == agent_id]
        assert all(
            segment.backup_location == BackupLocation.GPU_REPLICA
            for segment in agent_segments
        )
