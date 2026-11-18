from __future__ import annotations

import json

import pytest

from sglsim.srt.simulation.aegis_exact_contract import (
    CertifiedDPBudgetPlan,
    CertifiedExactWindowPlan,
    CertifiedRoutePlan,
    CertifiedStageChunkPlacement,
    load_certified_exact_window_plan,
)
from sglsim.srt.simulation.aegis_exact_optimizer import (
    ExactAdmissionState,
    ExactObjectiveWeights,
    ExactTopology,
)
from sglsim.srt.simulation.aegis_exact_rolling import (
    ExactRollingState,
    ExactWindowRequest,
    build_bootstrap_exact_instance,
    solve_causal_exact_rolling_window,
)


def _topology() -> ExactTopology:
    return ExactTopology(
        pp_size=2,
        dp_to_node=(0, 1),
        hbm_capacity_tokens=((64, 64), (64, 64)),
        host_policy_max_tokens=(128, 128),
        node_host_capacity_tokens=((0, 128), (1, 128)),
        link_bandwidth_bytes=((0, 1, 4096), (1, 0, 4096)),
        hbm_failure_domains=(("node:0", "node:0"), ("node:1", "node:1")),
        host_failure_domains=((0, "node:0"), (1, "node:1")),
    )


def test_bootstrap_instance_uses_separate_hazard_prior_and_chunk_slack() -> None:
    instance = build_bootstrap_exact_instance(
        instance_id="rolling-unit",
        requests=(
            ExactWindowRequest(
                request_id="r0",
                workflow_id="w0",
                source_dp_rank=0,
                token_count=6,
                slack_s=0.002,
                critical_weight=1.0,
            ),
        ),
        topology=_topology(),
        weights=ExactObjectiveWeights(),
        failure_hazard_probability=0.2,
        failure_scope="node",
        chunk_tokens=4,
        recompute_setup_s=0.001,
        recompute_s_per_token=0.001,
        protected_restore_setup_s=0.001,
        protected_restore_s_per_token=0.0,
        previous_hbm_budget_tokens=(0, 0),
        previous_host_budget_tokens=(0, 0),
        time_limit_s=10.0,
        deterministic_seed=7,
    )

    assert [scenario.probability for scenario in instance.scenarios] == [0.8, 0.2]
    assert [chunk.token_count for chunk in instance.chunks] == [4, 2]
    assert all(chunk.backup_candidates == (1,) for chunk in instance.chunks)
    failure_losses = [
        dict(chunk.unprotected_extra_loss_s)["failure:node:0"]
        for chunk in instance.chunks
    ]
    protected_losses = [
        dict(chunk.protected_extra_loss_s)["failure:node:0"]
        for chunk in instance.chunks
    ]
    assert sum(failure_losses) == pytest.approx(0.005)
    assert sum(protected_losses) == pytest.approx(0.0)


def test_bootstrap_keeps_admission_availability_separate_from_hazard_prior() -> None:
    state = ExactAdmissionState(
        state_id="manifest-known-diagnostic",
        valid_from_logical_time_s=2.0,
        valid_until_logical_time_s=3.0,
        unavailable_dp_ranks=(),
        information_mode="future_manifest_oracle",
    )
    instance = build_bootstrap_exact_instance(
        instance_id="rolling-oracle-label-unit",
        requests=(
            ExactWindowRequest(
                request_id="r0",
                workflow_id="w0",
                source_dp_rank=0,
                token_count=4,
                slack_s=0.0,
                critical_weight=1.0,
                admission_time_s=2.5,
                admission_state_id=state.state_id,
            ),
        ),
        topology=_topology(),
        weights=ExactObjectiveWeights(),
        failure_hazard_probability=0.2,
        failure_scope="node",
        chunk_tokens=4,
        recompute_setup_s=0.001,
        recompute_s_per_token=0.001,
        protected_restore_setup_s=0.001,
        protected_restore_s_per_token=0.0,
        previous_hbm_budget_tokens=(0, 0),
        previous_host_budget_tokens=(0, 0),
        time_limit_s=10.0,
        deterministic_seed=7,
        admission_states=(state,),
    )

    assert [scenario.probability for scenario in instance.scenarios] == [0.8, 0.2]
    assert instance.admission_states == (state,)
    assert instance.chunks[0].admission_state_id == state.state_id
    assert instance.chunks[0].admission_time_s == 2.5


def test_bootstrap_rejects_request_outside_observed_admission_interval() -> None:
    state = ExactAdmissionState(
        state_id="short-state",
        valid_from_logical_time_s=2.0,
        valid_until_logical_time_s=3.0,
        unavailable_dp_ranks=(),
    )
    with pytest.raises(ValueError, match="outside state"):
        build_bootstrap_exact_instance(
            instance_id="rolling-bad-time-unit",
            requests=(
                ExactWindowRequest(
                    request_id="r0",
                    workflow_id="w0",
                    source_dp_rank=0,
                    token_count=4,
                    slack_s=0.0,
                    critical_weight=1.0,
                    admission_time_s=3.0,
                    admission_state_id=state.state_id,
                ),
            ),
            topology=_topology(),
            weights=ExactObjectiveWeights(),
            failure_hazard_probability=0.2,
            failure_scope="node",
            chunk_tokens=4,
            recompute_setup_s=0.001,
            recompute_s_per_token=0.001,
            protected_restore_setup_s=0.001,
            protected_restore_s_per_token=0.0,
            previous_hbm_budget_tokens=(0, 0),
            previous_host_budget_tokens=(0, 0),
            time_limit_s=10.0,
            deterministic_seed=7,
            admission_states=(state,),
        )


def test_causal_rolling_solve_freezes_current_budget_and_persists_state() -> None:
    solved = solve_causal_exact_rolling_window(
        instance_id="causal-rolling-unit",
        requests=(
            ExactWindowRequest(
                request_id="r0",
                workflow_id="w0",
                source_dp_rank=0,
                token_count=4,
                slack_s=0.0,
                critical_weight=1.0,
            ),
        ),
        rolling_state=ExactRollingState(
            logical_time_s=4.5,
            window_id=3,
            route_version=4,
            unavailable_dp_ranks=(),
            resident_hbm_tokens=((0, 0), (2, 2)),
            resident_host_tokens_by_dp=(0, 0),
            resident_host_tokens_by_node=((0, 0), (1, 0)),
            used_link_bandwidth_bytes=(),
            current_hbm_budget_tokens=(16, 16),
            current_host_budget_tokens=(32, 32),
        ),
        topology=_topology(),
        weights=ExactObjectiveWeights(),
        failure_hazard_probability=0.2,
        failure_scope="node",
        chunk_tokens=4,
        recompute_setup_s=0.001,
        recompute_s_per_token=0.001,
        protected_restore_setup_s=0.001,
        protected_restore_s_per_token=0.0,
        time_limit_s=10.0,
        deterministic_seed=7,
    )

    assert solved.plan.budget_publication_mode == "reuse_current_fixed"
    assert solved.plan.window_id == 3
    assert solved.plan.route_version == 4
    route = solved.plan.route_for("r0")
    assert route is not None
    assert route.admission_time_s == 4.5
    assert route.admission_state_id == "causal-window-3-unavailable-none"
    assert route.admission_information_mode == "causal_observed"
    assert solved.replay_decision_sha256 == (
        solved.result.certificate.decision_sha256
    )


def test_runtime_loader_accepts_full_solver_artifact(tmp_path) -> None:
    plan = CertifiedExactWindowPlan(
        instance_id="artifact-unit",
        instance_sha256="instance",
        decision_sha256="decision",
        solver="unit",
        solver_version="1",
        solver_status="OPTIMAL",
        objective=1.0,
        dual_bound=1.0,
        mip_gap=0.0,
        solve_time_s=0.1,
        numerical_tolerance=1e-9,
        validation_errors=(),
        dp_size=2,
        pp_size=2,
        window_id=1,
        route_version=2,
        valid_from_logical_time=0.0,
        routes=(
            CertifiedRoutePlan(
                request_id="r0",
                workflow_id="w0",
                execution_dp_rank=0,
                backup_dp_rank=1,
            ),
        ),
        budgets=tuple(
            CertifiedDPBudgetPlan(
                dp_rank=rank,
                hbm_budget_tokens_per_stage=4,
                host_budget_tokens=0,
                alpha=0.5,
                beta=0.0,
            )
            for rank in range(2)
        ),
        placements=tuple(
            CertifiedStageChunkPlacement(
                request_id="r0",
                chunk_id=0,
                pp_rank=stage,
                backup_dp_rank=1,
                tier="HBM",
                token_count=4,
            )
            for stage in range(2)
        ),
    )
    path = tmp_path / "exact.json"
    path.write_text(
        json.dumps({"certified_exact_window_plan": plan.to_dict()}),
        encoding="utf-8",
    )

    assert load_certified_exact_window_plan(path) == plan
