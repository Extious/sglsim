from sglsim.srt.managers.data_parallel_controller import DataParallelController
from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.aegis_placement import (
    AegisClusterTopology,
    AegisNodeHostPool,
    AegisPersistentPool,
    AegisTopologyAwarePlanner,
)
from sglsim.srt.simulation.aegis_budget import (
    DPBudgetPublisher,
    NodeHostCapacityArbiter,
)
from sglsim.srt.simulation.aegis_exact_contract import (
    CertifiedDPBudgetPlan,
    CertifiedExactWindowPlan,
    CertifiedRoutePlan,
    CertifiedStageChunkPlacement,
)
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.aegis_residual_dag import (
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
)
from sglsim.srt.utils.network import _QUEUES
from dataclasses import replace
from queue import Queue
import json

import pytest


def _dp4_pp4_controller() -> DataParallelController:
    ConfigManager._raw = {
        "scheduler": {
            "dp_size": 4,
            "pp_size": 4,
            "load_balance_method": "cache_aware",
        },
        "aegis_l1_l2": {
            "enabled": True,
            # Controller tests use one-token synthetic leases; alignment
            # semantics are covered directly in test_aegis_placement.py.
            "host_alignment_tokens": 1,
            "topology": {
                "dp_rank_to_node": [0, 1, 2, 3],
                "pp_stages_per_node": 4,
                "gpu_backup_requires_direct_retry_route": True,
                "host_pool_scope": "node_shared_process_external",
            },
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    ConfigManager._hicache = None
    server_args = ServerArgs(dp_size=4, pp_size=4)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    return DataParallelController(
        server_args,
        port_args,
        scheduler_cmd_queues=[Queue() for _ in range(4)],
    )


def _replace_node_host_capacity(
    controller: DataParallelController,
    *,
    node_id: int,
    capacity_tokens: int,
) -> None:
    dp_ranks = tuple(
        dp_rank
        for dp_rank in range(len(controller.workers))
        if controller._aegis_topology.node_for_rank(dp_rank) == node_id
    )
    arbiter = NodeHostCapacityArbiter(
        node_id=node_id,
        physical_capacity_tokens=capacity_tokens,
        allowed_dp_ranks=dp_ranks,
    )
    for dp_rank in dp_ranks:
        current = controller._aegis_budget_publishers[dp_rank].current
        assert current is not None
        snapshot = replace(
            current,
            beta=1.0,
            host_policy_limit_tokens=capacity_tokens,
        )
        publisher = DPBudgetPublisher(
            dp_rank=dp_rank,
            pp_size=int(controller.server_args.pp_size),
        )
        publisher.publish(snapshot)
        controller._aegis_budget_publishers[dp_rank] = publisher
        arbiter.publish_dp_budget(snapshot)
    controller._aegis_node_capacity_arbiters[node_id] = arbiter


def test_current_design_controller_fails_unproven_slack_closed_to_zero() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_design_semantics = DESIGN_SEMANTICS
    sim = {"workflow_slack_ms": 5000.0}

    assert controller._effective_aegis_slack_ms(sim) == 0.0
    assert sim["aegis_slack_fail_closed"] is True
    assert sim["aegis_slack_update_physical_movement"] is False


def test_current_design_controller_consumes_visible_residual_dag_ticket() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_design_semantics = DESIGN_SEMANTICS
    graph = VisibleResidualDag(
        workflow_id="workflow",
        graph_version="v1",
        observed_at_ms=0.0,
        nodes=(
            ResidualDagNode("short", "agent", 2.0, active=True, agent_id="a"),
            ResidualDagNode("long", "agent", 5.0, active=True, agent_id="b"),
        ),
        edges=(),
        next_join_subgraph_complete=True,
    )
    sim = build_residual_dag_slack_tickets(graph)["short"].to_simulation_fields()

    assert controller._effective_aegis_slack_ms(sim) == 3.0
    assert sim["aegis_slack_fail_closed"] is False
    assert sim["aegis_slack_consumption_reason"] == "visible_residual_dag_cpm"


def test_aegis_events_are_unbounded_and_incrementally_journaled(
    tmp_path,
    monkeypatch,
) -> None:
    journal = tmp_path / "aegis-events.jsonl"
    monkeypatch.setenv("SGLSIM_AEGIS_EVENT_JOURNAL_PATH", str(journal))
    controller = _dp4_pp4_controller()
    try:
        first = controller._append_aegis_event("unit_event", {"value": 1})
        second = controller._append_aegis_event("unit_event", {"value": 2})
        status = controller.aegis_status()
    finally:
        controller.close_event_journal()

    assert isinstance(controller._aegis_events, list)
    assert [first["sequence"], second["sequence"]] == [1, 2]
    persisted = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    assert persisted == [first, second]
    assert status["event_history"]["mode"] == "unbounded_in_memory"
    assert status["event_history"]["journal"]["records_written"] == 2


def test_round_robin_distributes_to_dp_workers():
    ConfigManager._raw = {
        "scheduler": {"dp_size": 2, "load_balance_method": "round_robin"},
    }
    ConfigManager._scheduler = None
    server_args = ServerArgs(dp_size=2)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(server_args, port_args)

    for i in range(4):
        req = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(rid=f"r{i}", input_len=2, max_new_tokens=1)
        )
        controller.dispatch(req)

    q0 = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(0)]
    q1 = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(1)]
    assert len(q0) == 2
    assert len(q1) == 2


def test_soft_fail_supports_multiple_ranks_but_preserves_one_healthy_worker():
    controller = _dp4_pp4_controller()

    assert controller.soft_fail_rank(1)
    assert controller.soft_fail_rank(2)
    assert controller.soft_fail_rank(3)
    assert controller.soft_failed() == [1, 2, 3]
    assert not controller.soft_fail_rank(0)

    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="surviving-rank", input_len=2, max_new_tokens=1)
    )
    assert controller.dispatch(req) == 0

    assert controller.recover_soft_failed_rank(2)
    assert controller.soft_failed() == [1, 3]


def test_routing_key_affinity_keeps_same_key_on_one_dp_worker():
    ConfigManager._raw = {
        "scheduler": {
            "dp_size": 4,
            "load_balance_method": "round_robin",
            "enable_routing_key_affinity": True,
        },
    }
    ConfigManager._scheduler = None
    server_args = ServerArgs(dp_size=4)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(server_args, port_args)

    for i in range(3):
        req = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid=f"group-a-{i}",
                input_len=2,
                max_new_tokens=1,
                routing_key="group-a",
            )
        )
        controller.dispatch(req)

    target = 0
    for dp_rank in range(4):
        q = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(dp_rank)]
        assert len(q) == (3 if dp_rank == target else 0)
    assert all(
        msg.sampling_params.custom_params["simulation"]["assigned_dp_rank"] == target
        for msg in _QUEUES[port_args.scheduler_input_ipc_name_for_dp(target)]
    )


def test_routing_key_affinity_spreads_many_groups_across_workers():
    ConfigManager._raw = {
        "scheduler": {
            "dp_size": 4,
            "load_balance_method": "round_robin",
            "enable_routing_key_affinity": True,
        },
    }
    ConfigManager._scheduler = None
    server_args = ServerArgs(dp_size=4)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(server_args, port_args)

    for i in range(16):
        req = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid=f"group-{i}",
                input_len=2,
                max_new_tokens=1,
                routing_key=f"group-{i}",
            )
        )
        controller.dispatch(req)

    for dp_rank in range(4):
        q = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(dp_rank)]
        assert len(q) == 4
        assert all(
            msg.sampling_params.custom_params["simulation"]["assigned_dp_rank"] == dp_rank
            for msg in q
        )


def test_cache_aware_load_balance_method_reuses_matching_prefix_worker():
    ConfigManager._raw = {
        "scheduler": {"dp_size": 4, "load_balance_method": "cache_aware"},
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    ConfigManager._hicache = None
    server_args = ServerArgs(dp_size=4)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(server_args, port_args)

    requests = [
        GenerateReqInput(
            rid="hot-0",
            input_len=8,
            max_new_tokens=1,
            routing_key="hot",
        ),
        GenerateReqInput(
            rid="cold-0",
            input_len=8,
            max_new_tokens=1,
            routing_key="cold",
        ),
        GenerateReqInput(
            rid="hot-1",
            input_len=8,
            max_new_tokens=1,
            routing_key="hot",
        ),
    ]
    for req in requests:
        controller.dispatch(TokenizedGenerateReqInput.from_generate(req))

    q0 = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(0)]
    q1 = _QUEUES[port_args.scheduler_input_ipc_name_for_dp(1)]
    assert [msg.rid for msg in q0] == ["hot-0", "hot-1"]
    assert [msg.rid for msg in q1] == ["cold-0"]
    assert all(
        msg.sampling_params.custom_params["simulation"][
            "aegis_cache_load_aware_routing"
        ]
        for msg in [*q0, *q1]
    )


def test_recovered_rank_rehomes_execution_key_once_then_returns_to_cache_aware():
    ConfigManager._raw = {
        "scheduler": {
            "dp_size": 4,
            "pp_size": 4,
            "load_balance_method": "cache_aware",
        },
        "aegis_l1_l2": {
            "enabled": True,
            "rehome_after_recovery": True,
            "host_alignment_tokens": 1,
            "topology": {
                "dp_rank_to_node": [0, 1, 2, 3],
                "pp_stages_per_node": 4,
            },
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    ConfigManager._hicache = None
    server_args = ServerArgs(dp_size=4, pp_size=4)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(
        server_args,
        port_args,
        scheduler_cmd_queues=[Queue() for _ in range(4)],
    )

    def request(rid: str) -> TokenizedGenerateReqInput:
        req = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(rid=rid, input_len=32, max_new_tokens=1)
        )
        req.sampling_params.custom_params["simulation"].update(
            {
                "aegis_policy": "aegiskv_no_backup",
                "workflow_id": "workflow",
                "agent_id": "main",
            }
        )
        return req

    assert controller.dispatch(request("before-failure")) == 0
    assert controller.soft_fail_rank(0)
    controller._apply_gpu_recovery_l1_clear(
        {
            "request_id": "clear-0",
            "dp_rank": 0,
            "cache_result": {},
            "backup_result": {},
            "rank_state": {},
        }
    )
    assert controller.dispatch(request("during-failure")) == 1
    assert controller.recover_soft_failed_rank(0)

    rehomed = request("after-recovery")
    assert controller.dispatch(rehomed) == 0
    assert rehomed.sampling_params.custom_params["simulation"][
        "aegis_rehome_after_recovery"
    ] is True
    assert "agent:workflow:main" not in controller._aegis_pending_rehome


def test_pipeline_gpu_l1_clear_waits_for_every_stage_before_recovery():
    controller = _dp4_pp4_controller()
    assert controller.soft_fail_rank(2)
    request_id, waiter = controller.begin_rank_l1_clear(
        2, request_id="clear-dp2"
    )

    for pp_rank in range(4):
        controller._apply_gpu_recovery_l1_clear(
            {
                "type": "gpu_recovery_l1_clear_ack",
                "request_id": request_id,
                "dp_rank": 2,
                "pp_rank": pp_rank,
                "tp_rank": 0,
                "cache_result": {
                    "cleared_device_tokens": pp_rank + 1,
                    "cleared_device_pages": pp_rank + 2,
                },
                "backup_result": {
                    "dp_rank": 2,
                    "cleared_l1_backup_tokens": 10 + pp_rank,
                    "affected_backup_entries": 1,
                    "preserved_l2_backup_tokens": 20 + pp_rank,
                    "updated_hints": [
                        {
                            "backup_key": f"backup::pp{pp_rank}",
                            "pp_rank": pp_rank,
                        }
                    ],
                },
                "rank_state": {"stage_marker": pp_rank},
                "cleared_time": 100.0 + pp_rank,
            }
        )
        assert waiter.is_set() is (pp_rank == 3)
        if pp_rank < 3:
            assert controller.finish_rank_l1_clear(request_id) is None

    result = controller.finish_rank_l1_clear(request_id)
    assert result is not None
    assert result["all_stages_cleared"] is True
    assert result["cleared_stage_count"] == 4
    assert result["required_stage_count"] == 4
    assert [item["pp_rank"] for item in result["stage_results"]] == [0, 1, 2, 3]
    assert result["cache_result"]["cleared_device_tokens"] == 10
    assert result["cache_result"]["cleared_device_pages"] == 14
    assert result["backup_result"]["cleared_l1_backup_tokens"] == 46
    assert result["backup_result"]["affected_backup_entries"] == 4
    assert result["backup_result"]["preserved_l2_backup_tokens"] == 86
    assert len(result["backup_result"]["updated_hints"]) == 4
    assert result["rank_state"]["stages"] == {
        str(pp_rank): {"stage_marker": pp_rank} for pp_rank in range(4)
    }
    assert result["cleared_time"] == 103.0


def test_matched_route_plan_constrains_execution_but_keeps_backup_scoring():
    controller = _dp4_pp4_controller()
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="matched-route", input_len=512, max_new_tokens=16)
    )
    sim = req.sampling_params.custom_params["simulation"]
    sim.update(
        {
            "aegis_policy": "aegiskv_l1_l2",
            "backup_kv": True,
            "aegis_route_plan_dp_rank": 2,
            "aegis_critical_probability": 0.9,
            "workflow_slack_ms": 0.0,
            "workflow_id": "workflow",
            "agent_id": "main",
        }
    )

    assert controller.dispatch(req) == 2
    assert sim["aegis_route_plan_applied"] is True
    assert sim["aegis_backup_dp_rank"] in {0, 1, 3}


def test_matched_retry_route_plan_overrides_dynamic_recovery_rank():
    controller = _dp4_pp4_controller()
    assert controller.soft_fail_rank(1)
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="matched-retry", input_len=512, max_new_tokens=16)
    )
    sim = req.sampling_params.custom_params["simulation"]
    sim.update(
        {
            "aegis_policy": "aegiskv_no_backup",
            "failover_replay": True,
            "failed_dp_rank": 1,
            "pre_failure_total_tokens": 528,
            "aegis_backup_key": "matched-retry-source",
            "aegis_route_plan_retry_dp_rank": 3,
        }
    )

    assert controller.dispatch(req) == 3
    assert sim["aegis_route_plan_retry_applied"] is True
    assert sim["restore_action"] == "recompute"


def test_dp4_pp4_backup_plan_carries_gpu_mapping_and_host_paths() -> None:
    controller = _dp4_pp4_controller()
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="topology-plan", input_len=512, max_new_tokens=16)
    )
    sim = req.sampling_params.custom_params.setdefault("simulation", {})
    sim.update(
        {
            "aegis_policy": "aegiskv_l1_l2",
            "backup_kv": True,
            "aegis_backup_dp_rank": 1,
            "aegis_critical_probability": 0.9,
            "workflow_slack_ms": 0.0,
            "aegis_failure_scope": "node",
        }
    )

    controller._dispatch_l1_l2_backup_plan(
        req,
        source_dp_rank=0,
        cache_key="topology-plan",
        requested_tokens=528,
    )
    op, plan = controller._scheduler_cmd_queues[1].get_nowait()
    stage_plans = [plan]
    for _ in range(3):
        next_op, next_plan = controller._scheduler_cmd_queues[1].get_nowait()
        assert next_op == op
        stage_plans.append(next_plan)

    assert op == "aegis_l1_l2_backup_plan"
    assert sim["aegis_backup_key"] == plan["request_backup_key"]
    assert plan["failure_scope"] == "node"
    assert plan["gpu_candidate"]["reusable_dp_ranks"] == (1,)
    assert len(plan["gpu_candidate"]["pp_gpu_mapping"]) == 4
    assert plan["host_candidate"]["host_pool_node_id"] == 1
    assert plan["host_candidate"]["backup_path"] == ("d2h", "h2h")
    assert plan["host_candidate"]["recovery_path"] == ("h2d",)
    assert [stage["target_pp_rank"] for stage in stage_plans] == [0, 1, 2, 3]
    assert {
        stage["dp_budget_snapshot"]["budget_version"]
        for stage in stage_plans
    } == {1}
    expected_alpha = (
        plan["dp_budget_snapshot"]["hbm_limit_tokens_per_stage"][0]
        / controller._aegis_gpu_ledger.total_cache_tokens
    )
    expected_beta = (
        plan["dp_budget_snapshot"]["host_policy_limit_tokens"]
        / controller._aegis_node_capacity_arbiters[1].physical_capacity_tokens
    )
    assert {
        stage["dp_budget_snapshot"]["alpha"] for stage in stage_plans
    } == {expected_alpha}
    assert {
        stage["dp_budget_snapshot"]["beta"] for stage in stage_plans
    } == {expected_beta}


def test_controller_installs_and_dispatches_certified_exact_window() -> None:
    controller = _dp4_pp4_controller()
    for publisher in controller._aegis_budget_publishers.values():
        current = publisher.current
        assert current is not None
        for pp_rank in range(4):
            publisher.acknowledge(
                pp_rank=pp_rank,
                budget_version=current.budget_version,
            )

    hbm_reference = controller._aegis_gpu_ledger.total_cache_tokens
    exact = CertifiedExactWindowPlan(
        instance_id="controller-exact-window",
        instance_sha256="instance-sha",
        decision_sha256="decision-sha",
        solver="scipy.optimize.milp/HiGHS",
        solver_version="test",
        solver_status="OPTIMAL",
        objective=1.0,
        dual_bound=1.0,
        mip_gap=0.0,
        solve_time_s=0.1,
        numerical_tolerance=1e-8,
        validation_errors=(),
        dp_size=4,
        pp_size=4,
        window_id=1,
        route_version=2,
        valid_from_logical_time=0.0,
        routes=(
            CertifiedRoutePlan(
                request_id="exact-r0",
                workflow_id="exact-workflow",
                execution_dp_rank=0,
                backup_dp_rank=1,
            ),
        ),
        budgets=tuple(
            CertifiedDPBudgetPlan(
                dp_rank=dp_rank,
                hbm_budget_tokens_per_stage=(4 if dp_rank == 1 else 0),
                host_budget_tokens=0,
                alpha=(4 / hbm_reference if dp_rank == 1 else 0.0),
                beta=0.0,
            )
            for dp_rank in range(4)
        ),
        placements=tuple(
            CertifiedStageChunkPlacement(
                request_id="exact-r0",
                chunk_id=0,
                pp_rank=pp_rank,
                backup_dp_rank=1,
                tier="HBM",
                token_count=4,
            )
            for pp_rank in range(4)
        ),
    )

    installed = controller.install_certified_exact_window(exact)
    assert installed["budget_versions"] == {0: 2, 1: 2, 2: 2, 3: 2}
    for command_queue in controller._scheduler_cmd_queues:
        op, snapshot = command_queue.get_nowait()
        assert op == "aegis_publish_dp_budget"
        assert snapshot["budget_version"] == 2

    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="exact-r0", input_len=3, max_new_tokens=1)
    )
    sim = req.sampling_params.custom_params["simulation"]
    sim.update(
        {
            "aegis_policy": "aegiskv_l1_l2",
            "backup_kv": True,
            "critical_agent": True,
            "workflow_id": "exact-workflow",
            "agent_id": "main",
            "worker_seq": 1,
            "aegis_failure_scope": "node",
        }
    )

    assert controller.dispatch(req) == 0
    assert sim["aegis_backup_dp_rank"] == 1
    assert sim["aegis_placement_mode"] == "certified_exact"
    stage_plans = []
    for _ in range(4):
        op, stage_plan = controller._scheduler_cmd_queues[1].get_nowait()
        assert op == "aegis_l1_l2_backup_plan"
        stage_plans.append(stage_plan)
    assert [plan["target_pp_rank"] for plan in stage_plans] == [0, 1, 2, 3]
    assert all(plan["budget_version"] == 2 for plan in stage_plans)
    assert all(
        plan["placement_mode"] == "certified_exact" for plan in stage_plans
    )
    assert all(
        plan["exact_chunk_placements"]
        == [{"chunk_id": 0, "token_count": 4, "tier": "HBM"}]
        for plan in stage_plans
    )
    status = controller.aegis_status()
    assert status["certified_exact_window"]["certified_optimal"] is True
    assert status["exact_deployment_violation_count"] == 0


def test_certified_exact_route_fails_closed_when_admission_state_changed() -> None:
    controller = _dp4_pp4_controller()
    for publisher in controller._aegis_budget_publishers.values():
        current = publisher.current
        assert current is not None
        for pp_rank in range(4):
            publisher.acknowledge(
                pp_rank=pp_rank,
                budget_version=current.budget_version,
            )
    hbm_reference = controller._aegis_gpu_ledger.total_cache_tokens
    exact = CertifiedExactWindowPlan(
        instance_id="admission-state-unit",
        instance_sha256="instance-state-sha",
        decision_sha256="decision-state-sha",
        solver="unit",
        solver_version="1",
        solver_status="OPTIMAL",
        objective=1.0,
        dual_bound=1.0,
        mip_gap=0.0,
        solve_time_s=0.1,
        numerical_tolerance=1e-8,
        validation_errors=(),
        dp_size=4,
        pp_size=4,
        window_id=1,
        route_version=2,
        valid_from_logical_time=0.0,
        routes=(
            CertifiedRoutePlan(
                request_id="state-r0",
                workflow_id="state-workflow",
                execution_dp_rank=0,
                backup_dp_rank=1,
                admission_state_id="all-available",
                observed_unavailable_dp_ranks=(),
                admission_information_mode="causal_observed",
            ),
        ),
        budgets=tuple(
            CertifiedDPBudgetPlan(
                dp_rank=dp_rank,
                hbm_budget_tokens_per_stage=(4 if dp_rank == 1 else 0),
                host_budget_tokens=0,
                alpha=(4 / hbm_reference if dp_rank == 1 else 0.0),
                beta=0.0,
            )
            for dp_rank in range(4)
        ),
        placements=tuple(
            CertifiedStageChunkPlacement(
                request_id="state-r0",
                chunk_id=0,
                pp_rank=pp_rank,
                backup_dp_rank=1,
                tier="HBM",
                token_count=4,
            )
            for pp_rank in range(4)
        ),
    )
    controller.install_certified_exact_window(exact)
    assert controller.soft_fail_rank(3) is True
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="state-r0", input_len=3, max_new_tokens=1)
    )
    req.sampling_params.custom_params["simulation"].update(
        {
            "aegis_policy": "aegiskv_l1_l2",
            "backup_kv": True,
            "critical_agent": True,
            "workflow_id": "state-workflow",
            "agent_id": "main",
            "worker_seq": 1,
        }
    )

    with pytest.raises(RuntimeError, match="admission state does not match"):
        controller.dispatch(req)

    status = controller.aegis_status()
    assert status["exact_deployment_violation_count"] == 1
    assert status["exact_deployment_violations"][0]["kind"] == (
        "admission_state_mismatch"
    )


def test_route_only_exact_windows_reuse_an_acknowledged_fixed_budget() -> None:
    controller = _dp4_pp4_controller()
    for publisher in controller._aegis_budget_publishers.values():
        current = publisher.current
        assert current is not None
        for pp_rank in range(4):
            publisher.acknowledge(
                pp_rank=pp_rank,
                budget_version=current.budget_version,
            )

    def route_only_plan(window_id: int, rid: str) -> CertifiedExactWindowPlan:
        budgets = []
        for dp_rank in range(4):
            current = controller._aegis_budget_publishers[dp_rank].current
            assert current is not None
            budgets.append(
                CertifiedDPBudgetPlan(
                    dp_rank=dp_rank,
                    hbm_budget_tokens_per_stage=current.hbm_limit_tokens_per_stage[0],
                    host_budget_tokens=current.host_policy_limit_tokens,
                    alpha=current.alpha,
                    beta=current.beta,
                )
            )
        return CertifiedExactWindowPlan(
            instance_id=f"route-only-{window_id}",
            instance_sha256=f"instance-{window_id}",
            decision_sha256=f"decision-{window_id}",
            solver="unit",
            solver_version="1",
            solver_status="OPTIMAL",
            objective=1.0,
            dual_bound=1.0,
            mip_gap=0.0,
            solve_time_s=0.1,
            numerical_tolerance=1e-8,
            validation_errors=(),
            dp_size=4,
            pp_size=4,
            window_id=window_id,
            route_version=window_id + 1,
            valid_from_logical_time=float(window_id),
            routes=(
                CertifiedRoutePlan(
                    request_id=rid,
                    workflow_id="route-only-workflow",
                    execution_dp_rank=0,
                    backup_dp_rank=1,
                    admission_time_s=float(window_id),
                ),
            ),
            budgets=tuple(budgets),
            placements=tuple(
                CertifiedStageChunkPlacement(
                    request_id=rid,
                    chunk_id=0,
                    pp_rank=pp_rank,
                    backup_dp_rank=1,
                    tier="RECOMPUTE",
                    token_count=4,
                )
                for pp_rank in range(4)
            ),
            budget_publication_mode="reuse_current_fixed",
        )

    first = controller.install_certified_exact_window(
        route_only_plan(1, "route-only-r0")
    )
    second = controller.install_certified_exact_window(
        route_only_plan(2, "route-only-r1")
    )

    assert first["budget_reused"] is True
    assert second["budget_reused"] is True
    assert first["budget_versions"] == second["budget_versions"]
    assert all(queue.empty() for queue in controller._scheduler_cmd_queues)


def test_exact_rolling_snapshot_separates_stock_and_interval_link_flow() -> None:
    controller = _dp4_pp4_controller()
    controller._append_aegis_event(
        "backup_stage_ack",
        {
            "source_dp_rank": 0,
            "target_dp_rank": 1,
            "hbm_tokens": 3,
            "host_tokens": 2,
        },
    )
    # The synthetic controller clock is wall based, so use the event's own
    # timestamp as the causal interval probe.
    event_time = float(controller._aegis_events[-1]["time"])
    assert controller.soft_fail_rank(3) is True

    snapshot = controller.exact_rolling_state_snapshot(
        logical_time_s=event_time,
        link_window_duration_s=max(event_time + 1.0, 1.0),
        kv_bytes_per_token_per_stage=10,
    )

    assert snapshot["unavailable_dp_ranks"] == (3,)
    assert snapshot["used_link_bandwidth_bytes"] == ((0, 1, 50),)
    assert snapshot["resident_hbm_tokens"] == tuple(
        (0, 0, 0, 0) for _ in range(4)
    )
    assert snapshot["resident_host_tokens_by_dp"] == (0, 0, 0, 0)
    assert snapshot["semantics"]["resident"] == "stock_at_route_admission"


def test_causal_execution_preview_uses_normal_route_fallback_when_plan_is_down() -> None:
    controller = _dp4_pp4_controller()
    assert controller.soft_fail_rank(3) is True
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="preview-r0", input_len=3, max_new_tokens=1)
    )
    req.sampling_params.custom_params["simulation"].update(
        {
            "aegis_policy": "aegiskv_l1_l2",
            "backup_kv": True,
            "critical_agent": True,
            "aegis_route_plan_dp_rank": 3,
        }
    )

    preview = controller.preview_causal_execution_rank(req)

    assert preview in {0, 1, 2}
    assert preview != 3


def test_backup_target_filter_rejects_same_node_for_node_failure() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_topology = AegisClusterTopology.from_dict(
        {
            "dp_rank_to_node": [0, 0, 1, 2],
            "pp_stages_per_node": 4,
        },
        dp_size=4,
        pp_size=4,
    )
    controller._aegis_placement_planner = AegisTopologyAwarePlanner(
        topology=controller._aegis_topology,
        costs=controller._aegis_path_costs,
    )
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="node-filter", input_len=512, max_new_tokens=16)
    )
    sim = req.sampling_params.custom_params.setdefault("simulation", {})
    sim.update(
        {
            "aegis_policy": "aegiskv_l1_only",
            "critical_agent": True,
            "workflow_slack_ms": 0.0,
            "aegis_failure_scope": "node",
        }
    )

    selected = controller._choose_aegis_backup_rank(
        req=req,
        sim=sim,
        source_dp_rank=0,
        candidate_ranks=[1, 2, 3],
    )

    assert selected == 2
    same_node = next(
        row for row in sim["aegis_backup_target_scores"] if row["rank"] == 1
    ) if any(
        row["rank"] == 1 for row in sim["aegis_backup_target_scores"]
    ) else None
    assert same_node is None


def test_backup_target_score_avoids_high_gpu_pressure() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_rank_states[1] = {"normal_l1_pressure": 1.0}
    controller._aegis_rank_states[2] = {"normal_l1_pressure": 0.0}
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="pressure", input_len=1024, max_new_tokens=32)
    )
    sim = req.sampling_params.custom_params.setdefault("simulation", {})
    sim.update(
        {
            "aegis_policy": "aegiskv_l1_only",
            "critical_agent": True,
            "workflow_slack_ms": 0.0,
            "aegis_failure_scope": "node",
        }
    )

    selected = controller._choose_aegis_backup_rank(
        req=req,
        sim=sim,
        source_dp_rank=0,
        candidate_ranks=[1, 2],
    )

    assert selected == 2
    scores = {row["rank"]: row for row in sim["aegis_backup_target_scores"]}
    assert scores[2]["objective"] > scores[1]["objective"]


def test_retry_uses_gpu_only_on_exact_target_but_host_on_any_rank() -> None:
    controller = _dp4_pp4_controller()
    hint = {
        "ack": True,
        "target_dp_rank": 1,
        "l1_tokens": 600,
        "l2_tokens": 400,
        "failure_scope": "node",
        "gpu_candidate": {
            "valid": True,
            "reusable_dp_ranks": [1],
        },
        "host_candidate": {
            "valid": True,
            "host_pool_node_id": 1,
        },
    }

    direct = controller._aegis_retry_option(
        hint=hint,
        retry_dp_rank=1,
        requested_tokens=1000,
        failed_dp_rank=0,
    )
    remote = controller._aegis_retry_option(
        hint=hint,
        retry_dp_rank=2,
        requested_tokens=1000,
        failed_dp_rank=0,
    )

    assert direct["l1_tokens"] == 600
    assert direct["l2_tokens"] == 400
    assert direct["gpu_direct_reuse"] is True
    assert direct["host_restore_path"] == ("h2d",)
    assert remote["l1_tokens"] == 0
    assert remote["l2_tokens"] == 400
    assert remote["gpu_direct_reuse"] is False
    assert remote["host_restore_path"] == ("h2h", "h2d")
    assert remote["recompute_tokens"] == 600


def test_retry_never_counts_peer_gpu_tokens_after_route_changes() -> None:
    controller = _dp4_pp4_controller()
    option = controller._aegis_retry_option(
        hint={
            "ack": True,
            "target_dp_rank": 1,
            "l1_tokens": 1000,
            "l2_tokens": 0,
            "failure_scope": "gpu",
            "gpu_candidate": {
                "valid": True,
                "reusable_dp_ranks": [1],
            },
            "host_candidate": {},
        },
        retry_dp_rank=2,
        requested_tokens=1000,
        failed_dp_rank=0,
    )

    assert option["l1_tokens"] == 0
    assert option["restored_tokens"] == 0
    assert option["recompute_tokens"] == 1000
    assert option["action"] == "recompute"


def test_l3_lease_is_committed_and_retry_reuses_persistent_tokens() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_l3_pool = AegisPersistentPool(
        capacity_tokens=1024,
        manifest_quorum=2,
    )
    payload = controller._sync_aegis_l3_lease(
        {
            "rid": "persistent",
            "backup_key": "persistent",
            "target_dp_rank": 1,
            "version": 2,
            "l1_tokens": 128,
            "l2_tokens": 256,
            "recompute_tokens": 640,
            "l3_candidate": {
                "valid": True,
                "utility": 640.0,
                "recovery_path": ["l3_read", "h2d"],
            },
        }
    )

    assert payload["l3_tokens"] == 640
    assert payload["recompute_tokens"] == 0
    assert payload["safe_frontier_tokens"] == 1024
    assert payload["recovery_action"] == "l1_l2_l3_backup"
    assert payload["persistent_pool_lease"]["manifest_quorum"] == 2

    option = controller._aegis_retry_option(
        hint={"ack": True, **payload},
        retry_dp_rank=2,
        requested_tokens=1024,
        failed_dp_rank=0,
    )
    assert option["l1_tokens"] == 0
    assert option["l2_tokens"] == 0
    assert option["l3_tokens"] == 640
    assert option["l3_restore_path"] == ("l3_read", "h2d")
    assert option["action"] == "l3_backup"


def test_persistent_eviction_revokes_router_hint_before_retry() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_l3_pool = AegisPersistentPool(
        capacity_tokens=512,
        manifest_quorum=2,
    )
    low = controller._sync_aegis_l3_lease(
        {
            "rid": "low-l3",
            "backup_key": "low-l3",
            "target_dp_rank": 1,
            "version": 1,
            "l1_tokens": 0,
            "l2_tokens": 0,
            "recompute_tokens": 512,
            "l3_candidate": {"valid": True, "utility": 51.2},
        }
    )
    controller._aegis_recovery_hints["low-l3"] = {"ack": True, **low}

    high = controller._sync_aegis_l3_lease(
        {
            "rid": "high-l3",
            "backup_key": "high-l3",
            "target_dp_rank": 2,
            "version": 1,
            "l1_tokens": 0,
            "l2_tokens": 0,
            "recompute_tokens": 512,
            "l3_candidate": {"valid": True, "utility": 512.0},
        }
    )

    revoked = controller.recovery_hint("low-l3")
    assert high["persistent_pool_evicted_backup_keys"] == ["low-l3"]
    assert revoked["l3_tokens"] == 0
    assert revoked["recompute_tokens"] == 512
    assert revoked["persistent_pool_lease"] is None


def test_host_ack_creates_process_external_node_lease() -> None:
    controller = _dp4_pp4_controller()
    payload = controller._sync_aegis_host_lease(
        {
            "rid": "host-lease",
            "backup_key": "host-lease",
            "version": 1,
            "l1_tokens": 0,
            "l2_tokens": 1,
            "recompute_tokens": 0,
            "safe_frontier_tokens": 1,
            "failure_scope": "node",
            "host_candidate": {
                "valid": True,
                "host_pool_node_id": 1,
                "source_node_id": 0,
                "utility": 128.0,
            },
        }
    )

    assert payload["host_pool_lease"]["node_id"] == 1
    assert payload["host_pool_lease"]["token_count"] == 1
    assert payload["host_capacity_lease"]["request"]["dp_rank"] == 1
    assert payload["host_budget_owner"] == "dp_controller"
    assert payload["beta_scope"] == "dp_replica_policy"
    assert payload["node_host_capacity_owner"] == "node_physical_arbiter"
    assert payload["host_pool_state"]["scope"] == "node_shared_process_external"
    assert controller._aegis_host_pools[1].get("host-lease") is not None


def test_shared_node_arbiter_prevents_cross_dp_physical_oversubscription() -> None:
    ConfigManager._raw = {
        "scheduler": {
            "dp_size": 4,
            "pp_size": 4,
            "load_balance_method": "cache_aware",
        },
        "cache": {
            "page_size": 1,
            "l1_capacity_pages": 100,
            "l2_capacity_pages": 100,
        },
        "aegis_l1_l2": {
            "enabled": True,
            "initial_beta": 0.8,
            "host_alignment_tokens": 1,
            "topology": {
                "dp_rank_to_node": [0, 0, 1, 1],
                "pp_stages_per_node": 4,
                "host_pool_scope": "node_shared_process_external",
            },
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    ConfigManager._hicache = None
    server_args = ServerArgs(dp_size=4, pp_size=4)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    controller = DataParallelController(
        server_args,
        port_args,
        scheduler_cmd_queues=[Queue() for _ in range(4)],
    )

    def host_ack(key: str, target_dp_rank: int, tokens: int):
        return controller._sync_aegis_host_lease(
            {
                "rid": key,
                "backup_key": key,
                "target_dp_rank": target_dp_rank,
                "version": 1,
                "budget_version": 1,
                "l1_tokens": 0,
                "l2_tokens": tokens,
                "recompute_tokens": 0,
                "safe_frontier_tokens": tokens,
                "recovery_action": "l2_backup",
                "failure_scope": "node",
                "workflow_id": "shared-node-workflow",
                "host_candidate": {
                    "valid": True,
                    "host_pool_node_id": 0,
                    "source_node_id": 1,
                    "utility": float(tokens),
                },
            }
        )

    first = host_ack("dp0-70", 0, 70)
    denied = host_ack("dp1-40", 1, 40)

    assert first["l2_tokens"] == 70
    assert denied["l2_tokens"] == 0
    assert denied["recompute_tokens"] == 40
    assert denied["host_capacity_admission_reason"] == "node_physical_capacity"
    assert denied["dp_host_policy_limit_tokens"] == 80
    assert controller._aegis_budget_publishers[0].current.beta == 0.8
    assert controller._aegis_budget_publishers[1].current.beta == 0.8
    arbiter = controller._aegis_node_capacity_arbiters[0]
    assert arbiter.resident_tokens == 70
    assert arbiter.dp_resident_tokens(0) == 70
    assert arbiter.dp_resident_tokens(1) == 0


def test_host_ack_scores_utility_per_aligned_allocation() -> None:
    controller = _dp4_pp4_controller()
    _replace_node_host_capacity(
        controller,
        node_id=1,
        capacity_tokens=1024,
    )
    controller._aegis_host_pools[1] = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=1024,
        protection_ratio=1.0,
        pinned_quota_tokens=0,
        alignment_tokens=256,
    )

    payload = controller._sync_aegis_host_lease(
        {
            "rid": "aligned-density",
            "backup_key": "aligned-density",
            "version": 1,
            "l1_tokens": 0,
            "l2_tokens": 1,
            "recompute_tokens": 0,
            "safe_frontier_tokens": 1,
            "failure_scope": "node",
            "host_candidate": {
                "valid": True,
                "host_pool_node_id": 1,
                "source_node_id": 0,
                "utility": 128.0,
            },
        }
    )

    assert payload["host_pool_lease"]["allocated_tokens"] == 256
    assert payload["host_pool_utility_density_per_allocated_token"] == 0.5


def test_shared_host_eviction_downgrades_router_hint_and_reconciles_rank() -> None:
    controller = _dp4_pp4_controller()
    _replace_node_host_capacity(
        controller,
        node_id=1,
        capacity_tokens=1024,
    )
    controller._aegis_host_pools[1] = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=512,
        protection_ratio=1.0,
        pinned_quota_tokens=0,
    )
    low = controller._sync_aegis_host_lease(
        {
            "rid": "low",
            "backup_key": "low",
            "target_dp_rank": 1,
            "version": 1,
            "l1_tokens": 0,
            "l2_tokens": 512,
            "recompute_tokens": 0,
            "safe_frontier_tokens": 512,
            "recovery_action": "l2_backup",
            "failure_scope": "node",
            "host_candidate": {
                "valid": True,
                "host_pool_node_id": 1,
                "source_node_id": 0,
                "utility": 512.0,
            },
        }
    )
    controller._aegis_recovery_hints["low"] = low

    high = controller._sync_aegis_host_lease(
        {
            "rid": "high",
            "backup_key": "high",
            "target_dp_rank": 1,
            "version": 1,
            "l1_tokens": 0,
            "l2_tokens": 512,
            "recompute_tokens": 0,
            "safe_frontier_tokens": 512,
            "recovery_action": "l2_backup",
            "failure_scope": "node",
            "host_candidate": {
                "valid": True,
                "host_pool_node_id": 1,
                "source_node_id": 0,
                "utility": 1024.0,
            },
        }
    )

    downgraded = controller.recovery_hint("low")
    assert high["host_pool_evicted_backup_keys"] == ["low"]
    assert downgraded["l2_tokens"] == 0
    assert downgraded["recompute_tokens"] == 512
    assert downgraded["host_pool_lease"] is None
    arbiter = controller._aegis_node_capacity_arbiters[1]
    assert arbiter.dp_resident_tokens(1) == 512
    assert "low" not in arbiter.snapshot()["leases"]
    op, command = controller._scheduler_cmd_queues[1].get_nowait()
    assert op == "aegis_l1_l2_invalidate_host_lease"
    assert command == {
        "backup_key": "low",
        "expected_version": 1,
        "host_pool_node_id": 1,
    }


def test_shared_host_rejection_downgrades_current_ack_and_reconciles_rank() -> None:
    controller = _dp4_pp4_controller()
    _replace_node_host_capacity(
        controller,
        node_id=1,
        capacity_tokens=256,
    )
    controller._aegis_host_pools[1] = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=256,
        protection_ratio=1.0,
        pinned_quota_tokens=0,
    )
    existing = controller._aegis_host_pools[1].admit(
        backup_key="higher",
        token_count=256,
        utility_density=10.0,
        source_node_id=0,
        failure_scope="node",
    )
    assert existing.admitted

    payload = controller._sync_aegis_host_lease(
        {
            "rid": "lower",
            "backup_key": "lower",
            "target_dp_rank": 1,
            "version": 3,
            "l1_tokens": 0,
            "l2_tokens": 256,
            "recompute_tokens": 0,
            "safe_frontier_tokens": 256,
            "recovery_action": "l2_backup",
            "failure_scope": "node",
            "host_candidate": {
                "valid": True,
                "host_pool_node_id": 1,
                "source_node_id": 0,
                "utility": 1.0,
            },
        }
    )

    assert payload["l2_tokens"] == 0
    assert payload["recompute_tokens"] == 256
    assert payload["host_pool_reconciliation_pending"] is True
    op, command = controller._scheduler_cmd_queues[1].get_nowait()
    assert op == "aegis_l1_l2_invalidate_host_lease"
    assert command["backup_key"] == "lower"
    assert command["expected_version"] == 3


def test_router_ignores_stale_ack_before_touching_newer_host_lease() -> None:
    controller = _dp4_pp4_controller()
    current = controller._sync_aegis_host_lease(
        {
            "rid": "same",
            "backup_key": "same",
            "target_dp_rank": 1,
            "version": 2,
            "l1_tokens": 0,
            "l2_tokens": 1,
            "recompute_tokens": 0,
            "safe_frontier_tokens": 1,
            "recovery_action": "l2_backup",
            "failure_scope": "node",
            "host_candidate": {
                "valid": True,
                "host_pool_node_id": 1,
                "source_node_id": 0,
                "utility": 10.0,
            },
        }
    )
    controller._aegis_recovery_hints["same"] = current

    controller.apply_aegis_rank_event(
        {
            "type": "aegis_l1_l2_backup_ack",
            "payload": {
                **current,
                "version": 1,
                "l2_tokens": 100,
            },
            "updated_hints": [],
            "rank_state": {},
        }
    )

    assert controller.recovery_hint("same")["version"] == 2
    assert controller._aegis_host_pools[1].get("same").version == 2
    assert controller._scheduler_cmd_queues[1].empty()


def test_algorithm2_lineage_supersedes_old_cross_rank_frontier_and_ack() -> None:
    controller = _dp4_pp4_controller()

    def request(rid: str, worker_seq: int, backup_rank: int):
        req = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(rid=rid, input_len=32, max_new_tokens=1)
        )
        sim = req.sampling_params.custom_params["simulation"]
        sim.update(
            {
                "aegis_policy": "aegiskv_l1_l2",
                "backup_kv": True,
                "aegis_backup_key": rid,
                "aegis_backup_dp_rank": backup_rank,
                "workflow_id": "workflow-a",
                "agent_id": "main",
                "worker_seq": worker_seq,
            }
        )
        return req, sim

    old_req, old_sim = request("old-frontier", 8, 1)
    assert controller._track_aegis_frontier_lineage(old_req, old_sim) is True
    controller._aegis_recovery_hints["old-frontier"] = {
        "backup_key": "old-frontier",
        "version": 1,
        "target_dp_rank": 1,
    }
    admission = controller._aegis_host_pools[1].admit(
        backup_key="old-frontier",
        token_count=1,
        utility_density=1.0,
        source_node_id=0,
        failure_scope="node",
        version=1,
    )
    assert admission.admitted is True

    new_req, new_sim = request("new-frontier", 9, 2)
    assert controller._track_aegis_frontier_lineage(new_req, new_sim) is True
    assert new_sim["aegis_superseded_backup_key"] == "old-frontier"
    assert controller.recovery_hint("old-frontier") is None
    assert controller._aegis_host_pools[1].get("old-frontier") is None

    controller._aegis_recovery_hints["new-frontier"] = {
        "backup_key": "new-frontier",
        "version": 1,
        "target_dp_rank": 2,
        "l1_tokens": 128,
    }
    resolved, resolved_key, chain = controller._resolve_lineage_recovery_hint(
        "old-frontier"
    )
    assert resolved is not None
    assert resolved_key == "new-frontier"
    assert chain == ("old-frontier", "new-frontier")
    commands = [
        controller._scheduler_cmd_queues[1].get_nowait()
        for _ in range(4)
    ]
    assert all(op == "aegis_l1_l2_supersede_backup" for op, _ in commands)
    assert [command["pp_rank"] for _, command in commands] == [0, 1, 2, 3]
    assert [command["backup_key"] for _, command in commands] == [
        f"old-frontier::pp{pp_rank}" for pp_rank in range(4)
    ]
    assert all(
        command["request_backup_key"] == "old-frontier"
        and command["superseded_by"] == "new-frontier"
        and command["required_pp_size"] == 4
        for _, command in commands
    )

    for pp_rank in range(4):
        controller.apply_aegis_rank_event(
            {
                "type": "aegis_l1_l2_supersede_ack",
                "dp_rank": 1,
                "pp_rank": pp_rank,
                "request_backup_key": "old-frontier",
                "stage_backup_key": f"old-frontier::pp{pp_rank}",
                "required_pp_size": 4,
                "result": {
                    "backup_key": f"old-frontier::pp{pp_rank}",
                    "expected_version": 1,
                    "superseded_by": "new-frontier",
                    "released": True,
                    "released_l1_tokens": 2,
                    "released_l2_tokens": 3,
                },
                "rank_state": {"pp_rank": pp_rank, "entry_count": 0},
            }
        )
        aggregate_events = [
            event
            for event in controller.aegis_status()["events"]
            if event["kind"] == "backup_frontier_supersede_applied"
        ]
        assert len(aggregate_events) == (1 if pp_rank == 3 else 0)

    aggregate = aggregate_events[0]
    assert aggregate["all_stages_applied"] is True
    assert aggregate["applied_stage_count"] == 4
    assert aggregate["released_l1_tokens"] == 8
    assert aggregate["released_l2_tokens"] == 12
    assert aggregate["stage_backup_keys"] == [
        f"old-frontier::pp{pp_rank}" for pp_rank in range(4)
    ]
    assert sorted(controller._aegis_rank_states[1]["stages"]) == [
        "0",
        "1",
        "2",
        "3",
    ]

    # A delayed ACK for the retired request must not recreate shared leases or
    # a Router-visible recovery hint.
    controller.apply_aegis_rank_event(
        {
            "type": "aegis_l1_l2_backup_ack",
            "payload": {
                "rid": "old-frontier",
                "backup_key": "old-frontier",
                "version": 1,
                "target_dp_rank": 1,
                "l2_tokens": 1,
            },
            "updated_hints": [],
            "rank_state": {},
        }
    )
    assert controller.recovery_hint("old-frontier") is None
    assert controller._aegis_host_pools[1].get("old-frontier") is None

    stale_req, stale_sim = request("stale-frontier", 8, 3)
    assert controller._track_aegis_frontier_lineage(stale_req, stale_sim) is False
    assert stale_sim["aegis_lineage_plan_stale"] is True


def test_algorithm2_lineage_refreshes_router_selected_backup_target() -> None:
    controller = _dp4_pp4_controller()
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="router-selected", input_len=32, max_new_tokens=1)
    )
    sim = req.sampling_params.custom_params["simulation"]
    sim.update(
        {
            "aegis_policy": "aegiskv_l1_l2",
            "backup_kv": True,
            "workflow_id": "workflow-refresh",
            "agent_id": "main",
            "worker_seq": 1,
        }
    )

    assert controller._track_aegis_frontier_lineage(req, sim) is True
    assert controller._aegis_lineage_latest[("workflow-refresh", "main")][
        "target_dp_rank"
    ] == -1

    sim["aegis_backup_dp_rank"] = 3
    controller._refresh_aegis_lineage_target(req, sim)

    assert controller._aegis_lineage_latest[("workflow-refresh", "main")][
        "target_dp_rank"
    ] == 3


def test_no_backup_retry_reuses_surviving_normal_lineage_prefix() -> None:
    controller = _dp4_pp4_controller()
    cache_key = "agent:workflow-a:main"
    controller._aegis_gpu_ledger.record_inference(
        rid="newer-turn",
        dp_rank=0,
        cache_key=cache_key,
        token_count=256,
    )
    req = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="old-turn-retry", input_len=128, max_new_tokens=128)
    )
    sim = req.sampling_params.custom_params["simulation"]
    sim.update(
        {
            "aegis_policy": "aegiskv_no_backup",
            "failover_replay": True,
            "failed_dp_rank": 2,
            "pre_failure_total_tokens": 256,
            "aegis_backup_key": "old-turn",
            "workflow_id": "workflow-a",
            "agent_id": "main",
            "aegis_route_plan_retry_dp_rank": 0,
        }
    )

    assert controller._choose_l1_l2_retry_rank(req, sim) == 0
    assert sim["restore_action"] == "live_l1_reuse"
    assert sim["restore_tokens"] == 256
    assert sim["restore_recompute_tokens"] == 0
    assert sim["aegis_retry_used_hint"] is False
    assert sim["aegis_retry_live_prefix_tokens"] == 256


def _pipeline_stage_ack_event(pp_rank: int, frontier: int) -> dict:
    total = 128
    return {
        "type": "aegis_l1_l2_backup_ack",
        "payload": {
            "rid": "stage-request",
            "backup_key": f"stage-request::pp{pp_rank}",
            "source_dp_rank": 0,
            "target_dp_rank": 1,
            "version": 1,
            "ack": True,
            "l1_tokens": frontier,
            "l2_tokens": 0,
            "recompute_tokens": total - frontier,
            "safe_frontier_tokens": frontier,
            "recovery_action": "l1_backup",
            "sequence": pp_rank + 1,
            "updated_time": 1.0 + pp_rank,
            "failure_scope": "node",
            "gpu_candidate": {
                "valid": True,
                "backup_path": ("gpu_rdma",),
                "recovery_path": ("direct_hbm_reuse",),
                "reusable_dp_ranks": (1,),
            },
        },
        "rank_state": {
            "normal_l1_pressure": 0.1 * pp_rank,
            "normal_l2_pressure": 0.05 * pp_rank,
        },
        "dp_rank": 1,
        "pp_rank": pp_rank,
        "tp_rank": 0,
        "request_backup_key": "stage-request",
        "stage_backup_key": f"stage-request::pp{pp_rank}",
        "source_pp_rank": pp_rank,
        "target_pp_rank": pp_rank,
        "required_pp_size": 4,
        "route_version": 3,
        "budget_version": 1,
    }


def _pipeline_group_commit_event(pp_rank: int, frontier: int) -> dict:
    event = _pipeline_stage_ack_event(pp_rank, frontier)
    event["type"] = "aegis_l1_l2_group_commit_ack"
    event["payload"].update(
        {
            "token_count": 128,
            "materialization_scope": "committed_prefix",
            "source_committed_token_end": 128,
            "group_commit_verified": True,
            "copy_completion_evidence": "sglsim_modeled_group_commit",
            "execution_id": "stage-request-execution-1",
            "block_generation": 0,
        }
    )
    return event


def _install_target_rotation_fixture(
    controller: DataParallelController,
    *,
    backup_key: str = "rotation-request",
) -> None:
    stage_payloads = []
    for pp_rank in range(4):
        stage_key = f"{backup_key}::pp{pp_rank}"
        stage_payload = {
            "rid": backup_key,
            "backup_key": stage_key,
            "request_backup_key": backup_key,
            "pp_rank": pp_rank,
            "source_dp_rank": 0,
            "target_dp_rank": 1,
            "version": 1,
            "route_version": 3,
            "budget_version": 1,
            "token_count": 128,
            "l1_tokens": 64,
            "l2_tokens": 64,
            "hbm_pending_tokens": 64,
            "dram_pending_tokens": 0,
            "reserved_l1_tokens": 128,
            "reserved_l2_tokens": 64,
            "safe_frontier_tokens": 64,
            "recompute_tokens": 64,
            "materialization_scope": "committed_prefix",
            "source_committed_token_end": 128,
            "group_commit_verified": True,
            "copy_completion_evidence": "unit-independent-tier-commit",
            "execution_id": "rotation-execution-1",
            "block_generation": 0,
            "design_semantics": DESIGN_SEMANTICS,
            "failure_scope": "node",
            "gpu_candidate": {
                "valid": True,
                "target_dp_rank": 1,
                "reusable_dp_ranks": [1],
            },
            "host_candidate": {
                "valid": True,
                "host_pool_node_id": 1,
                "target_dp_rank": 1,
            },
            "tier_committed_source_frontiers": {
                "hbm": 64,
                "dram": 64,
            },
        }
        stage_payloads.append(stage_payload)
        controller._aegis_stage_reservation_plans[
            (backup_key, 1, pp_rank)
        ] = {
            **stage_payload,
            "allowed_tiers": ["l1", "l2"],
            "required_pp_size": 4,
            "source_pp_rank": pp_rank,
            "target_pp_rank": pp_rank,
            "dp_budget_snapshot": controller._aegis_budget_publishers[
                1
            ].current.to_dict(),
        }
    controller._aegis_stage_reservation_plans[(backup_key, 2, 0)] = {
        **controller._aegis_stage_reservation_plans[(backup_key, 1, 0)],
        "version": 2,
    }
    controller._aegis_recovery_hints[backup_key] = {
        **stage_payloads[0],
        "backup_key": backup_key,
        "stage_backup_key": None,
        "all_stages_committed": True,
        "required_stage_count": 4,
        "stage_payloads": stage_payloads,
        "stage_hbm_frontiers": [64, 64, 64, 64],
        "stage_dram_frontiers": [64, 64, 64, 64],
    }


def test_target_before_source_failure_rotates_same_route_hbm_only() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_committed_copy_protocol = True
    controller._aegis_design_semantics = DESIGN_SEMANTICS
    _install_target_rotation_fixture(controller)
    assert controller.mark_rank_hard_failed(1)

    result = controller.invalidate_aegis_physical_failure(
        failed_dp_ranks=(1,),
        reason="unit_target_failure",
    )

    assert len(result["target_rotations"]) == 1
    rotation = result["target_rotations"][0]
    assert rotation["source_dp_rank"] == 0
    assert rotation["previous_target_dp_rank"] == 1
    assert rotation["replacement_target_dp_rank"] == 2
    assert rotation["route_version"] == 3
    assert rotation["route_version_changed"] is False
    assert rotation["dram_preserved"] is True
    hint = controller.recovery_hint("rotation-request")
    assert hint is not None
    assert hint["target_dp_rank"] == 2
    assert hint["route_version"] == 3
    assert hint["l1_tokens"] == 0
    assert hint["l2_tokens"] == 64
    assert hint["reserved_l2_tokens"] == 64
    assert hint["tier_committed_source_frontiers"] == {
        "hbm": 0,
        "dram": 64,
    }
    assert all(
        stage["target_dp_rank"] == 2 and stage["l1_tokens"] == 0
        for stage in hint["stage_payloads"]
    )

    rotated_plans = []
    while not controller._scheduler_cmd_queues[2].empty():
        rotated_plans.append(controller._scheduler_cmd_queues[2].get_nowait())
    assert len(rotated_plans) == 4
    assert all(op == "aegis_l1_l2_backup_plan" for op, _plan in rotated_plans)
    assert all(
        plan["target_dp_rank"] == 2
        and plan["source_dp_rank"] == 0
        and plan["route_version"] == 3
        and plan["version"] == 1
        and plan["allowed_tiers"] == ["l1"]
        and plan["target_rotation_hbm_only"] is True
        for _op, plan in rotated_plans
    )
    # A different version is not captured by this rotation.
    assert controller._aegis_stage_reservation_plans[
        ("rotation-request", 2, 0)
    ]["target_dp_rank"] == 1

    # Replacement reservation ACKs update only HBM metadata; the committed
    # source-owned DRAM prefix remains the recovery-safe tier.
    for pp_rank in range(4):
        ack = _pipeline_stage_ack_event(pp_rank, 0)
        ack["request_backup_key"] = "rotation-request"
        ack["stage_backup_key"] = f"rotation-request::pp{pp_rank}"
        ack["payload"].update(
            {
                "rid": "rotation-request",
                "backup_key": f"rotation-request::pp{pp_rank}",
                "target_dp_rank": 2,
                "version": 1,
                "hbm_pending_tokens": 128,
                "reserved_l1_tokens": 128,
                "design_semantics": DESIGN_SEMANTICS,
            }
        )
        ack["dp_rank"] = 2
        controller.apply_aegis_rank_event(ack)
    merged = controller.recovery_hint("rotation-request")
    assert merged is not None
    assert merged["target_dp_rank"] == 2
    assert merged["route_version"] == 3
    assert merged["l1_tokens"] == 0
    assert merged["l2_tokens"] == 64
    assert merged["reserved_l1_tokens"] == 128
    assert merged["reserved_l2_tokens"] == 64
    assert merged["materialization_scope"] == "committed_prefix"
    assert merged["group_commit_verified"] is True

    stale_ack = _pipeline_stage_ack_event(0, 128)
    stale_ack["request_backup_key"] = "rotation-request"
    stale_ack["stage_backup_key"] = "rotation-request::pp0"
    stale_ack["payload"].update(
        {
            "rid": "rotation-request",
            "backup_key": "rotation-request::pp0",
            "target_dp_rank": 1,
            "version": 1,
            "design_semantics": DESIGN_SEMANTICS,
        }
    )
    controller.apply_aegis_rank_event(stale_ack)
    after_stale = controller.recovery_hint("rotation-request")
    assert after_stale is not None
    assert after_stale["target_dp_rank"] == 2
    assert after_stale["l1_tokens"] == 0
    assert after_stale["l2_tokens"] == 64
    assert controller.aegis_status()["events"][-1]["kind"] == (
        "stale_rotated_pipeline_ack_ignored"
    )

    # The source Scheduler still names its admission-time target.  Router
    # fencing resolves the event to the replacement without changing version.
    controller.apply_aegis_rank_event(
        {
            "type": "aegis_source_chunk_committed",
            "rid": "rotation-request",
            "request_backup_key": "rotation-request",
            "source_dp_rank": 0,
            "target_dp_rank": 1,
            "source_pp_rank": 2,
            "target_pp_rank": 2,
            "required_pp_size": 4,
            "version": 1,
            "route_version": 3,
            "budget_version": 1,
            "source_committed_token_end": 128,
            "reserved_token_count": 128,
            "execution_id": "rotation-execution-1",
            "block_generation": 0,
        }
    )
    op, command = controller._scheduler_cmd_queues[2].get_nowait()
    assert op == "aegis_materialize_committed_prefix"
    assert command["target_dp_rank"] == 2
    assert command["route_version"] == 3
    assert controller._scheduler_cmd_queues[1].empty()

    completion = controller.complete_aegis_request(
        backup_key="rotation-request",
        expected_version=1,
        target_dp_rank=1,
    )
    assert completion["dispatched"] is True
    assert completion["requested_target_dp_rank"] == 1
    assert completion["target_dp_rank"] == 2
    completion_commands = []
    while not controller._scheduler_cmd_queues[2].empty():
        completion_commands.append(
            controller._scheduler_cmd_queues[2].get_nowait()
        )
    assert len(completion_commands) == 4
    assert all(
        op == "aegis_l1_l2_complete_request"
        for op, _payload in completion_commands
    )

    duplicate = controller.invalidate_aegis_physical_failure(
        failed_dp_ranks=(1,),
        reason="duplicate_unit_target_failure",
    )
    assert duplicate["target_rotations"] == []


def test_target_failure_without_compatible_replacement_is_degraded() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_design_semantics = DESIGN_SEMANTICS
    _install_target_rotation_fixture(controller, backup_key="degraded-request")
    for rank in (1, 2, 3):
        assert controller.mark_rank_hard_failed(rank)

    result = controller.invalidate_aegis_physical_failure(
        failed_dp_ranks=(1,),
        reason="unit_no_standby",
    )

    assert len(result["target_rotations"]) == 1
    rotation = result["target_rotations"][0]
    assert rotation["degraded"] is True
    assert rotation["replacement_target_dp_rank"] is None
    assert rotation["route_version"] == 3
    hint = controller.recovery_hint("degraded-request")
    assert hint is not None
    assert hint["route_degraded_no_standby"] is True
    assert hint["target_dp_rank"] == 1
    assert hint["l1_tokens"] == 0
    assert hint["l2_tokens"] == 64
    assert all(queue.empty() for queue in controller._scheduler_cmd_queues)


def test_source_chunk_commit_forwards_only_to_aligned_target_stage() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_committed_copy_protocol = True

    controller.apply_aegis_rank_event(
        {
            "type": "aegis_source_chunk_committed",
            "rid": "source-commit",
            "request_backup_key": "source-object",
            "source_dp_rank": 0,
            "target_dp_rank": 1,
            "source_pp_rank": 2,
            "target_pp_rank": 2,
            "required_pp_size": 4,
            "version": 3,
            "route_version": 4,
            "budget_version": 5,
            "source_committed_token_end": 64,
            "reserved_token_count": 128,
            "execution_id": "source-execution-1",
            "block_generation": 7,
            "observed_time": 10.0,
        }
    )

    op, command = controller._scheduler_cmd_queues[1].get_nowait()
    assert op == "aegis_materialize_committed_prefix"
    assert command["request_backup_key"] == "source-object"
    assert command["stage_backup_key"] == "source-object::pp2"
    assert command["source_committed_token_end"] == 64
    assert command["execution_id"] == "source-execution-1"
    assert controller._scheduler_cmd_queues[0].empty()
    events = controller.aegis_status()["events"]
    assert events[-1]["kind"] == "source_chunk_committed"


def _physical_tier_notification_event(
    *,
    group_id: str = "physical-hbm-group-1",
    token_end: int = 64,
) -> dict:
    return {
        "type": "aegis_source_chunk_committed",
        "rid": "physical-source-commit",
        "request_backup_key": "physical-source-object",
        "source_dp_rank": 0,
        "target_dp_rank": 1,
        "source_pp_rank": 2,
        "target_pp_rank": 2,
        "required_pp_size": 4,
        "version": 3,
        "route_version": 4,
        "budget_version": 5,
        "source_committed_token_end": 128,
        "reserved_token_count": 128,
        "execution_id": "physical-source-execution-1",
        "block_generation": 7,
        "observed_time": 10.0,
        "group_id": group_id,
        "tier": "gpu_aegis",
        "state": "COMMITTED",
        "success": True,
        "token_start": max(token_end - 64, 0),
        "token_end": token_end,
        "expected_shards": 4,
        "acked_shards": 4,
    }


def test_physical_tier_commit_dispatch_is_single_tier_and_idempotent() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_committed_copy_protocol = True
    controller._aegis_design_semantics = DESIGN_SEMANTICS
    event = _physical_tier_notification_event()

    assert controller.commit_aegis_physical_tier(
        event, completed_tier="hbm"
    )

    op, command = controller._scheduler_cmd_queues[1].get_nowait()
    assert op == "aegis_materialize_committed_prefix"
    assert command["completed_tiers"] == ["hbm"]
    assert command["physical_group_id"] == "physical-hbm-group-1"
    assert command["physical_group_commit_verified"] is True
    # The source event may cover 128 tokens, but this physical group proves
    # only its own first 64-token range.
    assert command["source_committed_token_end"] == 64
    assert controller.commit_aegis_physical_tier(
        event, completed_tier="hbm"
    ) is False
    assert controller._scheduler_cmd_queues[1].empty()
    assert controller.aegis_status()["events"][-1]["kind"] == (
        "physical_tier_commit_dispatched"
    )


def test_physical_tier_commit_rejects_conflict_and_incomplete_group() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_committed_copy_protocol = True
    controller._aegis_design_semantics = DESIGN_SEMANTICS
    event = _physical_tier_notification_event()
    assert controller.commit_aegis_physical_tier(
        event, completed_tier="hbm"
    )
    controller._scheduler_cmd_queues[1].get_nowait()

    with pytest.raises(ValueError, match="conflicting commit evidence"):
        controller.commit_aegis_physical_tier(
            {**event, "token_end": 128},
            completed_tier="hbm",
        )

    incomplete = _physical_tier_notification_event(
        group_id="physical-dram-group-incomplete"
    )
    incomplete.update(
        {
            "tier": "node_dram",
            "state": "ABORTED",
            "success": False,
            "acked_shards": 3,
        }
    )
    assert controller.commit_aegis_physical_tier(
        incomplete, completed_tier="dram"
    ) is False
    assert controller._scheduler_cmd_queues[1].empty()
    assert controller.aegis_status()["events"][-1]["kind"] == (
        "physical_tier_commit_rejected"
    )


def test_physical_tier_commit_does_not_resurrect_after_target_failure() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_committed_copy_protocol = True
    controller._aegis_design_semantics = DESIGN_SEMANTICS
    event = _physical_tier_notification_event(
        group_id="physical-hbm-group-target-loss"
    )
    controller.status[1] = False

    assert controller.commit_aegis_physical_tier(
        event, completed_tier="hbm"
    ) is False
    assert controller._scheduler_cmd_queues[1].empty()

    # Once the callback observed a dead destination, replaying the same group
    # after repair cannot manufacture a lost HBM copy.
    controller.status[1] = True
    assert controller.commit_aegis_physical_tier(
        event, completed_tier="hbm"
    ) is False
    assert controller._scheduler_cmd_queues[1].empty()


def test_committed_copy_mode_never_recovers_from_reservation_ack() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_committed_copy_protocol = True
    reservation = {
        "ack": True,
        "backup_key": "reserved-only",
        "source_dp_rank": 0,
        "target_dp_rank": 1,
        "l1_tokens": 128,
        "safe_frontier_tokens": 128,
        "materialization_scope": "metadata_reservation_only",
        "gpu_candidate": {"reusable_dp_ranks": [1]},
    }

    option = controller._aegis_retry_option(
        hint=reservation,
        retry_dp_rank=1,
        requested_tokens=128,
        failed_dp_rank=0,
        cache_key="missing",
    )

    assert option["restored_tokens"] == 0
    assert option["recompute_tokens"] == 128
    assert option["action"] == "recompute"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("source_committed_token_end", None),
        ("source_committed_token_end", "not-an-int"),
        ("safe_frontier_tokens", -1),
        ("safe_frontier_tokens", 129),
        ("execution_id", ""),
        ("block_generation", -1),
        ("block_generation", "not-an-int"),
    ],
)
def test_committed_copy_mode_fails_closed_on_malformed_commit_proof(
    field: str, invalid_value: object
) -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_committed_copy_protocol = True
    committed = {
        "ack": True,
        "backup_key": "committed",
        "source_dp_rank": 0,
        "target_dp_rank": 1,
        "l1_tokens": 128,
        "l2_tokens": 0,
        "safe_frontier_tokens": 128,
        "materialization_scope": "committed_prefix",
        "source_committed_token_end": 128,
        "group_commit_verified": True,
        "execution_id": "execution-1",
        "block_generation": 0,
        "gpu_candidate": {"reusable_dp_ranks": [1]},
    }
    committed[field] = invalid_value

    option = controller._aegis_retry_option(
        hint=committed,
        retry_dp_rank=1,
        requested_tokens=128,
        failed_dp_rank=0,
        cache_key="missing",
    )

    assert option["restored_tokens"] == 0
    assert option["recompute_tokens"] == 128
    assert option["action"] == "recompute"


def test_pipeline_group_commit_retains_latest_stage_state_across_updates() -> None:
    controller = _dp4_pp4_controller()
    controller._aegis_committed_copy_protocol = True

    for pp_rank in range(4):
        controller.apply_aegis_rank_event(
            _pipeline_group_commit_event(pp_rank, 64)
        )
    first = controller.recovery_hint("stage-request")
    assert first is not None
    assert first["materialization_scope"] == "committed_prefix"
    assert first["group_commit_verified"] is True
    assert first["safe_frontier_tokens"] == 64

    # Faster stages can advance independently.  The controller must retain
    # their latest ACKs so the slow stage's later update can advance the group
    # frontier without requiring the faster stages to retransmit.
    for pp_rank in range(3):
        controller.apply_aegis_rank_event(
            _pipeline_group_commit_event(pp_rank, 128)
        )
        assert controller.recovery_hint("stage-request")[
            "safe_frontier_tokens"
        ] == 64
    controller.apply_aegis_rank_event(_pipeline_group_commit_event(3, 128))

    final = controller.recovery_hint("stage-request")
    assert final is not None
    assert final["safe_frontier_tokens"] == 128
    assert final["source_committed_token_end"] == 128
    assert final["recovery_hint"]["restored_tokens_by_stage"] == [
        128,
        128,
        128,
        128,
    ]


def test_pipeline_barrier_never_mixes_versions_or_execution_ids() -> None:
    controller = _dp4_pp4_controller()
    for pp_rank in range(3):
        controller.apply_aegis_rank_event(
            _pipeline_stage_ack_event(pp_rank, 64)
        )

    for pp_rank in range(4):
        current = _pipeline_stage_ack_event(pp_rank, 96)
        current["payload"]["version"] = 2
        controller.apply_aegis_rank_event(current)
    assert controller.recovery_hint("stage-request")["version"] == 2

    # The final v1 stage cannot complete the earlier three-stage partial set
    # after a newer version has become authoritative.
    controller.apply_aegis_rank_event(_pipeline_stage_ack_event(3, 128))
    assert controller.recovery_hint("stage-request")["version"] == 2

    isolated = _dp4_pp4_controller()
    isolated._aegis_committed_copy_protocol = True
    for pp_rank in range(3):
        isolated.apply_aegis_rank_event(
            _pipeline_group_commit_event(pp_rank, 64)
        )
    other_execution = _pipeline_group_commit_event(3, 64)
    other_execution["payload"]["execution_id"] = "other-execution"
    isolated.apply_aegis_rank_event(other_execution)
    assert isolated.recovery_hint("stage-request") is None


def test_pipeline_ack_barrier_publishes_only_after_all_four_stages() -> None:
    controller = _dp4_pp4_controller()
    frontiers = (128, 96, 112, 64)

    for pp_rank, frontier in enumerate(frontiers[:3]):
        controller.apply_aegis_rank_event(
            _pipeline_stage_ack_event(pp_rank, frontier)
        )
        assert controller.recovery_hint("stage-request") is None

    controller.apply_aegis_rank_event(_pipeline_stage_ack_event(3, frontiers[3]))
    hint = controller.recovery_hint("stage-request")

    assert hint is not None
    assert hint["all_stages_committed"] is True
    assert hint["committed_stage_count"] == 4
    assert hint["safe_frontier_tokens"] == 64
    assert hint["recovery_hint"]["bottleneck_pp_rank"] == 3
    assert hint["recovery_hint"]["restored_tokens_by_stage"] == [
        128,
        96,
        112,
        64,
    ]
    assert [
        ack["backup_stage"]["pp_rank"] for ack in hint["stage_acks"]
    ] == [0, 1, 2, 3]


def test_pipeline_ack_barrier_rejects_cross_stage_connector_ack() -> None:
    controller = _dp4_pp4_controller()
    event = _pipeline_stage_ack_event(2, 64)
    event["source_pp_rank"] = 1

    with pytest.raises(ValueError, match="same-stage Connector"):
        controller.apply_aegis_rank_event(event)


def test_pipeline_retry_uses_all_stage_bottleneck_and_records_stage_arrays() -> None:
    controller = _dp4_pp4_controller()
    for pp_rank, frontier in enumerate((128, 96, 112, 64)):
        controller.apply_aegis_rank_event(
            _pipeline_stage_ack_event(pp_rank, frontier)
        )
    assert controller.soft_fail_rank(0)
    retry = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(rid="stage-request-retry", input_len=64, max_new_tokens=64)
    )
    sim = retry.sampling_params.custom_params["simulation"]
    sim.update(
        {
            "aegis_policy": "aegiskv_l1_l2",
            "failover_replay": True,
            "aegis_backup_key": "stage-request",
            "failed_dp_rank": 0,
            "pre_failure_total_tokens": 128,
            "aegis_route_plan_retry_dp_rank": 1,
        }
    )

    assert controller._choose_l1_l2_retry_rank(retry, sim) == 1
    assert sim["restore_tokens"] == 64
    assert sim["restore_recompute_tokens"] == 64
    assert sim["restore_action"] == "stage_aligned_restore"
    assert sim["aegis_retry_restored_tokens_by_stage"] == [128, 96, 112, 64]
    assert sim["aegis_retry_recomputed_tokens_by_stage"] == [0, 32, 16, 64]
    assert sim["aegis_retry_bottleneck_pp_rank"] == 3

    non_backup = controller._aegis_retry_option(
        hint=controller.recovery_hint("stage-request"),
        retry_dp_rank=2,
        requested_tokens=128,
        failed_dp_rank=0,
        cache_key="missing",
    )
    assert non_backup["restored_tokens"] == 0
    assert non_backup["recompute_tokens"] == 128
