from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from sglsim.srt.simulation.aegis_exact_optimizer import (
    ExactAdmissionState,
    ExactChunk,
    ExactObjectiveWeights,
    ExactOptimizationInstance,
    ExactScenario,
    ExactSolverCertificate,
    ExactTopology,
    build_certified_exact_window_plan,
    solve_exact_placement,
)
from sglsim.srt.simulation.aegis_exact_contract import (
    CertifiedExactWindowPlan,
)


def _instance(*, two_chunks: bool = False) -> ExactOptimizationInstance:
    topology = ExactTopology(
        pp_size=2,
        dp_to_node=(0, 1, 2),
        hbm_capacity_tokens=((0, 0), (4, 4), (4, 4)),
        host_policy_max_tokens=(0, 8, 8),
        node_host_capacity_tokens=((0, 0), (1, 8), (2, 8)),
        link_bandwidth_bytes=((0, 1, 128), (0, 2, 128)),
        hbm_failure_domains=(
            ("node0/pp0", "node0/pp1"),
            ("node1", "node1"),
            ("node2", "node2"),
        ),
        host_failure_domains=((0, "node0"), (1, "node1"), (2, "node2")),
    )
    scenarios = (
        ExactScenario(
            scenario_id="healthy",
            probability=0.5,
            failed_domains=(),
            base_workflow_loss_s=(("w0", 0.0),),
        ),
        ExactScenario(
            scenario_id="node1_failed",
            probability=0.5,
            failed_domains=("node1",),
            base_workflow_loss_s=(("w0", 0.0),),
        ),
    )
    chunks = [
        ExactChunk(
            request_id="r0",
            workflow_id="w0",
            chunk_id=0,
            source_dp_rank=0,
            token_count=4,
            backup_candidates=(1, 2),
            unprotected_extra_loss_s=(("healthy", 10.0), ("node1_failed", 10.0)),
            protected_extra_loss_s=(("healthy", 1.0), ("node1_failed", 1.0)),
        )
    ]
    if two_chunks:
        chunks.append(
            ExactChunk(
                request_id="r0",
                workflow_id="w0",
                chunk_id=1,
                source_dp_rank=0,
                token_count=4,
                backup_candidates=(1, 2),
                unprotected_extra_loss_s=(
                    ("healthy", 8.0),
                    ("node1_failed", 8.0),
                ),
                protected_extra_loss_s=(
                    ("healthy", 2.0),
                    ("node1_failed", 2.0),
                ),
            )
        )
    return ExactOptimizationInstance(
        instance_id="tiny-two" if two_chunks else "tiny-one",
        topology=topology,
        scenarios=scenarios,
        chunks=tuple(chunks),
        weights=ExactObjectiveWeights(
            expected_workflow_loss=1.0,
            cvar=0.0,
            hbm_normal_cost_per_token=0.1,
            host_normal_cost_per_token=0.2,
            hbm_budget_cost_per_token=0.01,
            host_budget_cost_per_token=0.01,
            hbm_copy_bytes_per_token=1,
            host_copy_bytes_per_token=1,
        ),
        previous_hbm_budget_tokens=(0, 0, 0),
        previous_host_budget_tokens=(0, 0, 0),
        time_limit_s=10.0,
        deterministic_seed=17,
    )


def _bruteforce_one_chunk(instance: ExactOptimizationInstance) -> float:
    chunk = instance.chunks[0]
    best = float("inf")
    for backup_dp in chunk.backup_candidates:
        for tiers in itertools.product(("HBM", "HOST", "RECOMPUTE"), repeat=2):
            hbm_by_stage = [
                chunk.token_count if tier == "HBM" else 0 for tier in tiers
            ]
            host = sum(chunk.token_count for tier in tiers if tier == "HOST")
            hbm_budget = max(hbm_by_stage)
            if any(
                used > instance.topology.hbm_capacity_tokens[backup_dp][pp]
                for pp, used in enumerate(hbm_by_stage)
            ):
                continue
            if host > instance.topology.host_policy_max_tokens[backup_dp]:
                continue
            node = instance.topology.dp_to_node[backup_dp]
            if host > instance.topology.node_host_capacity(node):
                continue
            copy_bytes = sum(
                chunk.token_count for tier in tiers if tier in {"HBM", "HOST"}
            )
            if copy_bytes > instance.topology.link_capacity(0, backup_dp):
                continue
            normal = sum(
                chunk.token_count
                * (
                    instance.weights.hbm_normal_cost_per_token
                    if tier == "HBM"
                    else instance.weights.host_normal_cost_per_token
                    if tier == "HOST"
                    else 0.0
                )
                for tier in tiers
            )
            budget_cost = (
                hbm_budget * instance.weights.hbm_budget_cost_per_token
                + host * instance.weights.host_budget_cost_per_token
            )
            expected_loss = 0.0
            for scenario in instance.scenarios:
                failed = set(scenario.failed_domains)
                protected = all(
                    (
                        tier == "HBM"
                        and instance.topology.hbm_failure_domains[backup_dp][pp]
                        not in failed
                    )
                    or (
                        tier == "HOST"
                        and instance.topology.host_failure_domain(node) not in failed
                    )
                    for pp, tier in enumerate(tiers)
                )
                losses = dict(
                    chunk.protected_extra_loss_s
                    if protected
                    else chunk.unprotected_extra_loss_s
                )
                expected_loss += scenario.probability * losses[scenario.scenario_id]
            best = min(best, normal + budget_cost + expected_loss)
    return best


def test_milp_matches_exhaustive_enumeration_and_returns_optimality_certificate() -> None:
    instance = _instance()
    result = solve_exact_placement(instance).require_global_optimum()

    assert result.certificate.status == "OPTIMAL"
    assert result.certificate.mip_gap == 0.0
    assert result.certificate.objective == pytest.approx(
        _bruteforce_one_chunk(instance), abs=1e-8
    )
    assert dict(result.backup_dp_by_request) == {"r0": 2}
    assert [placement.tier for placement in result.placements] == ["HBM", "HBM"]
    assert result.budgets[2].hbm_budget_tokens_per_stage == 4
    assert result.budgets[2].alpha == 1.0
    assert result.certificate.validation_errors == ()


def test_failed_backup_domain_cannot_produce_all_stage_recoverable_prefix() -> None:
    result = solve_exact_placement(_instance()).require_global_optimum()
    prefixes = {
        (scenario, request): tokens
        for scenario, request, tokens in result.recoverable_prefix_by_scenario
    }

    assert prefixes[("healthy", "r0")] == 4
    assert prefixes[("node1_failed", "r0")] == 4
    assert dict(result.backup_dp_by_request)["r0"] == 2


def test_two_chunk_solution_keeps_recoverable_prefix_contiguous() -> None:
    instance = _instance(two_chunks=True)
    topology = replace(
        instance.topology,
        host_policy_max_tokens=(0, 0, 0),
        node_host_capacity_tokens=((0, 0), (1, 0), (2, 0)),
    )
    result = solve_exact_placement(
        replace(instance, topology=topology)
    ).require_global_optimum()
    prefixes = {
        (scenario, request): tokens
        for scenario, request, tokens in result.recoverable_prefix_by_scenario
    }

    assert prefixes[("healthy", "r0")] == 4
    assert prefixes[("node1_failed", "r0")] == 4
    chunk_tiers = {
        (placement.chunk_id, placement.pp_rank): placement.tier
        for placement in result.placements
    }
    assert chunk_tiers[(0, 0)] == "HBM"
    assert chunk_tiers[(0, 1)] == "HBM"
    assert chunk_tiers[(1, 0)] == "RECOMPUTE"
    assert chunk_tiers[(1, 1)] == "RECOMPUTE"


def test_solver_replay_is_deterministic_and_hashes_instance_and_decision() -> None:
    instance = _instance()
    first = solve_exact_placement(instance).require_global_optimum()
    second = solve_exact_placement(instance).require_global_optimum()

    assert first.certificate.instance_sha256 == instance.sha256
    assert second.certificate.instance_sha256 == instance.sha256
    assert first.certificate.decision_sha256 == second.certificate.decision_sha256
    assert first.certificate.objective == second.certificate.objective


def test_causal_admission_state_forbids_an_observed_unavailable_backup() -> None:
    instance = _instance()
    state = ExactAdmissionState(
        state_id="dp2-down",
        valid_from_logical_time_s=4.0,
        valid_until_logical_time_s=8.0,
        unavailable_dp_ranks=(2,),
        information_mode="causal_observed",
    )
    chunks = tuple(
        replace(
            chunk,
            admission_time_s=5.0,
            admission_state_id=state.state_id,
        )
        for chunk in instance.chunks
    )
    stateful = replace(
        instance,
        chunks=chunks,
        admission_states=(state,),
    )

    result = solve_exact_placement(stateful).require_global_optimum()

    assert dict(result.backup_dp_by_request) == {"r0": 1}
    plan = build_certified_exact_window_plan(
        stateful,
        result,
        window_id=2,
        route_version=3,
        valid_from_logical_time=4.0,
    )
    route = plan.route_for("r0")
    assert route is not None
    assert route.admission_state_id == "dp2-down"
    assert route.admission_time_s == 5.0
    assert route.observed_unavailable_dp_ranks == (2,)
    assert route.admission_information_mode == "causal_observed"
    assert plan.uses_future_manifest_oracle is False


def test_admission_state_rejects_unavailable_execution_or_all_backups() -> None:
    instance = _instance()
    execution_down = ExactAdmissionState(
        state_id="source-down",
        valid_from_logical_time_s=0.0,
        valid_until_logical_time_s=None,
        unavailable_dp_ranks=(0,),
    )
    with pytest.raises(ValueError, match="execution DP is unavailable"):
        replace(
            instance,
            admission_states=(execution_down,),
            chunks=tuple(
                replace(chunk, admission_state_id="source-down")
                for chunk in instance.chunks
            ),
        )

    backups_down = ExactAdmissionState(
        state_id="backups-down",
        valid_from_logical_time_s=0.0,
        valid_until_logical_time_s=None,
        unavailable_dp_ranks=(1, 2),
    )
    with pytest.raises(ValueError, match="no available backup DP"):
        replace(
            instance,
            admission_states=(backups_down,),
            chunks=tuple(
                replace(chunk, admission_state_id="backups-down")
                for chunk in instance.chunks
            ),
        )


def test_rolling_resident_stock_and_fixed_budgets_are_hard_constraints() -> None:
    instance = _instance()
    state = ExactAdmissionState(
        state_id="dp1-down",
        valid_from_logical_time_s=0.0,
        valid_until_logical_time_s=None,
        unavailable_dp_ranks=(1,),
    )
    topology = replace(
        instance.topology,
        host_policy_max_tokens=(0, 0, 0),
        node_host_capacity_tokens=((0, 0), (1, 0), (2, 0)),
    )
    rolling = replace(
        instance,
        topology=topology,
        chunks=tuple(
            replace(chunk, admission_state_id=state.state_id)
            for chunk in instance.chunks
        ),
        admission_states=(state,),
        resident_hbm_tokens=((0, 0), (0, 0), (2, 2)),
        resident_host_tokens_by_dp=(0, 0, 0),
        resident_host_tokens_by_node=((0, 0), (1, 0), (2, 0)),
        fixed_hbm_budget_tokens=(0, 0, 4),
        fixed_host_budget_tokens=(0, 0, 0),
    )

    result = solve_exact_placement(rolling).require_global_optimum()

    assert dict(result.backup_dp_by_request) == {"r0": 2}
    assert {placement.tier for placement in result.placements} == {"RECOMPUTE"}
    assert result.budgets[2].hbm_budget_tokens_per_stage == 4
    assert result.certificate.validation_errors == ()


def test_rolling_node_stock_must_equal_dp_attributed_host_stock() -> None:
    instance = _instance()
    with pytest.raises(ValueError, match="must equal DP-attributed stock"):
        replace(
            instance,
            resident_host_tokens_by_dp=(0, 0, 2),
            resident_host_tokens_by_node=((0, 0), (1, 0), (2, 1)),
        )


def test_certified_result_builds_solver_free_runtime_wire_contract() -> None:
    instance = _instance()
    result = solve_exact_placement(instance).require_global_optimum()

    plan = build_certified_exact_window_plan(
        instance,
        result,
        window_id=1,
        route_version=2,
        valid_from_logical_time=3.5,
    )
    restored = CertifiedExactWindowPlan.from_dict(plan.to_dict())

    assert restored == plan
    assert restored.certified_optimal is True
    assert restored.route_for("r0").backup_dp_rank == 2
    assert [
        row.tier for row in restored.placements_for("r0", pp_rank=0)
    ] == ["HBM"]
    assert restored.solver_certificate_id.endswith(
        result.certificate.decision_sha256
    )


def test_nonoptimal_certificate_cannot_claim_global_optimum() -> None:
    certificate = ExactSolverCertificate(
        solver="test",
        solver_version="test",
        status="TIME_OR_ITERATION_LIMIT",
        status_code=1,
        message="time limit",
        objective=1.0,
        dual_bound=0.5,
        mip_gap=0.5,
        node_count=1,
        solve_time_s=1.0,
        instance_sha256="instance",
        decision_sha256="decision",
        deterministic_seed=0,
        numerical_tolerance=1e-8,
        validation_errors=(),
        solver_options=(),
    )

    assert certificate.certified_optimal is False


def test_node_physical_capacity_is_validated_independently_from_dp_beta() -> None:
    with pytest.raises(ValueError, match="node 2"):
        ExactTopology(
            pp_size=1,
            dp_to_node=(0, 2),
            hbm_capacity_tokens=((1,), (1,)),
            host_policy_max_tokens=(1, 1),
            node_host_capacity_tokens=((0, 1),),
            link_bandwidth_bytes=((0, 1, 1),),
            hbm_failure_domains=(("d0",), ("d1",)),
            host_failure_domains=((0, "n0"),),
        )
