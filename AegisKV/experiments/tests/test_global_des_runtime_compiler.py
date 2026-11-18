from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from AegisKV.experiments.src.global_des_config import config_hash, load_suite
from AegisKV.experiments.src.global_des_runtime_compiler import (
    RuntimeBindings,
    RuntimeNotReadyError,
    audit_p1_p12_execution,
    compile_deployment,
    compile_failure_timeline,
    cell_runtime_readiness,
    deployment_layer_readiness,
    failure_layer_readiness,
)
from sglsim.srt.managers.aegis_independent_tier_manager import (
    AegisIndependentTierRankManager,
)
from sglsim.srt.managers.aegis_l1_l2 import (
    AegisBackupPlan,
    resolve_aegis_per_node_dram_quota,
)
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.global_des import PhysicalGpuId
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    PhysicalGpuSpec,
)


MODEL_REVISIONS = {
    "qwen3-8b": "b968826d9c46dd6066d109eabc6255188de91218",
    "qwen3-32b": "9216db5781bf21249d130ec9da846c4624c16137",
}


def _modeled_bindings(**updates: object) -> RuntimeBindings:
    binding = RuntimeBindings(
        model_revisions=MODEL_REVISIONS,
        simulator_revision="3" * 40,
        hardware_timing_class="modeled",
        hardware_timing_source="configs_v3 modeled A100/link profile",
        hardware_timing_sha256="4" * 64,
        allow_modeled_hardware=True,
        failure_target_selector="dense_first_v1",
        repair_delay_policy="serial_sum_v1",
    )
    return replace(binding, **updates)


def _frozen_modeled_bindings(**updates: object) -> RuntimeBindings:
    binding = RuntimeBindings(
        simulator_revision="3" * 64,
        failure_target_selector="dense_first_v1",
        repair_delay_policy="serial_sum_v1",
    )
    return replace(binding, **updates)


def _first_axes(group: str, **updates: object) -> tuple[object, dict[str, object]]:
    selected = load_suite().group_map[group]
    axes = dict(next(selected.iter_cells())["axes"])
    axes.update(updates)
    return selected, axes


def _active_plan(
    group: object,
    axes: dict[str, object],
    deployment: object,
    *,
    source_rank: int = 0,
) -> dict[str, object]:
    failure = group.failure.resolved
    scenario = next(
        (
            str(axes[name])
            for name in ("failure_scenario", "fault_domain", "failure_severity")
            if name in failure["axes"]
        ),
        str(failure["fixed"]["fault_model"]["central_target"]),
    )
    scope, count_text = scenario.rsplit("_", 1)
    count = int(count_text)
    source_node = next(
        spec.gpu.node_id
        for spec in deployment.topology.gpus
        if spec.gpu.dp_rank == source_rank
    )
    trigger = failure["fixed"]["trigger"]
    normalization = trigger.get("scale_normalization")
    if normalization is None:
        offset_ns = int(trigger["active_offset_ns"])
        offset_rule = {
            "rule": "fixed_active_offset_v1",
            "active_offset_ns": offset_ns,
        }
    else:
        reference_gpus = int(normalization["reference_node_count"]) * int(
            normalization["reference_gpus_per_node"]
        )
        selected_gpus = deployment.gpu_count
        reference_offset = int(normalization["reference_active_offset_ns"])
        offset_ns = (
            reference_offset * reference_gpus + selected_gpus // 2
        ) // selected_gpus
        offset_rule = {
            "rule": normalization["rule"],
            "active_offset_ns": offset_ns,
        }
    if scope == "node":
        targets = [{"failure_scope": "node", "node_id": source_node}]
    else:
        targets = [
            {
                "failure_scope": scope,
                "dp_rank": (source_rank + index) % deployment.dp_size,
            }
            for index in range(count)
        ]
    plan: dict[str, object] = {
        "schema": "aegiskv.policy-independent-failure-plan.v1",
        "construction": "pre_execution_request_universe_projection_v1",
        "request_universe_sha256": "a" * 64,
        "executed_request_set_sha256": "b" * 64,
        "seed": int(axes["seed"]),
        "scenario": scenario,
        "physical_topology_sha256": config_hash(deployment.topology.as_dict()),
        "anchor_selector": "minimum_arrival_ns_then_request_id_v1",
        "anchor_request_id": "anchor-request",
        "anchor_arrival_ns": 0,
        "anchor_source_dp_rank": source_rank,
        "anchor_source_node_id": source_node,
        "target_selector": trigger["target_selector"][scope],
        "planned_targets": targets,
        "active_offset_rule": offset_rule,
        "active_offset_ns": offset_ns,
        "policy_inputs_excluded": list(
            trigger["shadow_plan"]["excluded_inputs"]
        ),
    }
    plan["shadow_plan_sha256"] = config_hash(plan)
    return plan


def test_frozen_suite_audits_all_2134_p1_p12_cells_fail_closed() -> None:
    audit = audit_p1_p12_execution(load_suite())

    assert audit.planned_cells == 2134
    assert audit.executable_cells == 0
    assert audit.blocked_cells == 2134
    assert tuple(group.group for group in audit.groups) == tuple(
        f"P{index}" for index in range(1, 13)
    )
    assert sum(group.planned_cells for group in audit.groups) == 2134
    codes = {
        code for group in audit.groups for code in group.error_cell_counts
    }
    assert {
        "simulator_revision_missing",
        "failure_target_selector_missing",
        "repair_delay_policy_missing",
    } <= codes
    assert audit.as_dict()["scope"] == "P1-P12"


def test_honest_frozen_modeled_provenance_compiles_all_frozen_models() -> None:
    audit = audit_p1_p12_execution(
        load_suite(), _frozen_modeled_bindings()
    )

    assert audit.planned_cells == 2134
    assert audit.executable_cells == 2134
    assert audit.blocked_cells == 0
    assert {
        group.group: group.executable_cells for group in audit.groups
    } == {
        "P1": 160,
        "P2": 240,
        "P3": 640,
        "P4": 40,
        "P5": 60,
        "P6": 200,
        "P7": 240,
        "P8": 160,
        "P9": 160,
        "P10": 120,
        "P11": 60,
        "P12": 54,
    }
    assert {
        code for group in audit.groups for code in group.error_cell_counts
    } == set()


def test_complete_test_bindings_make_every_p1_p12_cell_compilable() -> None:
    audit = audit_p1_p12_execution(load_suite(), _modeled_bindings())
    executable_by_group = {
        group.group: group.executable_cells for group in audit.groups
    }

    assert audit.planned_cells == 2134
    assert audit.executable_cells == 2134
    assert audit.blocked_cells == 0
    assert executable_by_group == {
        "P1": 160,
        "P2": 240,
        "P3": 640,
        "P4": 40,
        "P5": 60,
        "P6": 200,
        "P7": 240,
        "P8": 160,
        "P9": 160,
        "P10": 120,
        "P11": 60,
        "P12": 54,
    }
    remaining = {
        code for group in audit.groups for code in group.error_cell_counts
    }
    assert remaining == set()


def test_modeled_binding_is_explicit_and_compiles_dense_tp4_topology() -> None:
    group, axes = _first_axes(
        "P1", policy="recompute", model="qwen3-32b", fault_domain="worker_1"
    )
    compiled = compile_deployment(
        group.deployment.resolved, axes, _modeled_bindings()
    )

    assert compiled.node_count == 4
    assert compiled.dp_size == 8
    assert compiled.pp_size == 1
    assert compiled.tp_size == 4
    assert len(compiled.topology.gpus) == 32
    assert len(compiled.topology.nodes) == 4
    assert all(len(compiled.topology.group(dp, 0)) == 4 for dp in range(8))
    assert {
        (spec.gpu.dp_rank, spec.gpu.pp_rank, spec.gpu.tp_rank)
        for spec in compiled.topology.gpus
    } == {(dp, 0, tp) for dp in range(8) for tp in range(4)}
    assert compiled.provenance["evidence_class"] == "modeled_not_calibrated"
    assert compiled.policy.tiers == ()
    assert compiled.simulator_config_overlay["communication"][
        "kv_bytes_per_token"
    ] == 262_144 // 4


def test_global_des_starts_with_every_gpu_node_and_link_resource_addressable() -> None:
    group, axes = _first_axes("P1", policy="recompute", model="qwen3-8b")
    compiled = compile_deployment(
        group.deployment.resolved, axes, _modeled_bindings()
    )
    des = compiled.new_global_des()
    keys = {resource.key for resource in des.resource_busy_until_ns}

    assert len(compiled.topology.gpus) == 32
    assert len(compiled.topology.nodes) == 4
    assert "fabric:cluster-fabric-0:lane0" in keys
    assert "nvlink:node0:lane0" in keys
    assert "dram-read:node0:lane0" in keys
    assert "dram-write:node0:lane0" in keys
    assert any(key.startswith("gpu-d2d-read:node0/gpu0/dp0/pp0/tp0") for key in keys)
    assert any(key.startswith("gpu-h2d-write:node0/gpu0/dp0/pp0/tp0") for key in keys)
    assert des.gpu_hbm == {}
    assert des.node_dram == {}

    serialized = compiled.as_dict()
    bindings = serialized["rank_resource_bindings"]
    assert len(bindings) == len(compiled.topology.gpus)
    first = next(
        row
        for row in bindings
        if row["gpu"]["dp_rank"] == 0
        and row["gpu"]["pp_rank"] == 0
        and row["gpu"]["tp_rank"] == 0
    )
    assert first["node_dram_id"] == "node0"
    assert first["node_dram"]["capacity_bytes"] == 1_099_511_627_776
    assert "fabric:cluster-fabric-0:lane0" in first["transfer_resources"]
    assert "dram-read:node0:lane0" in first["transfer_resources"]
    assert "gpu-d2d-read:node0/gpu0/dp0/pp0/tp0:lane0" in first[
        "transfer_resources"
    ]


def test_cross_node_bandwidth_axis_compiles_into_copy_fabric() -> None:
    group, axes = _first_axes("P7", cross_node_bandwidth_gb_s=10)
    compiled = compile_deployment(
        group.deployment.resolved, axes, _modeled_bindings()
    )

    assert compiled.copy_fabric.bandwidths.fabric_bytes_per_s == 10_000_000_000
    assert compiled.copy_fabric.bandwidths.pcie_bytes_per_s == 25_000_000_000
    assert compiled.copy_fabric.setup.cross_node_d2d_ns == 15_000
    assert compiled.copy_fabric.setup.cross_node_d2h_ns == 16_000
    assert compiled.copy_fabric.bandwidths.dram_read_bytes_per_s == 40_000_000_000
    assert compiled.copy_fabric.setup.same_node_h2d_ns == 3_000
    assert compiled.copy_fabric.setup.cross_node_h2d_ns == 16_000
    serialized = compiled.as_dict()["copy_fabric"]
    assert serialized["bandwidth_bytes_per_s"]["fabric_bytes_per_s"] == 10_000_000_000
    assert serialized["setup_ns"]["cross_node_h2d_ns"] == 16_000


def test_hbm_only_failure_cell_accepts_integrated_physical_recovery_path() -> None:
    group, axes = _first_axes(
        "P1",
        policy="neighbor_replica",
        model="qwen3-8b",
        fault_domain="worker_1",
    )
    report = cell_runtime_readiness(group, axes, _modeled_bindings())

    assert report.ready
    assert all(
        error.code != "physical_recovery_transfer_binding_missing"
        for error in report.errors
    )


def test_selected_dram_budget_compiles_to_exact_node_shared_runtime_quota() -> None:
    group, axes = _first_axes(
        "P10",
        model="qwen3-8b",
        dram_budget_gib_per_node=128,
    )
    compiled = compile_deployment(
        group.deployment.resolved, axes, _modeled_bindings()
    )
    runtime = compiled.simulator_config_overlay["aegis_l1_l2"]
    quota_bytes = 128 * 2**30
    quota_tokens, tail = divmod(quota_bytes, 147_456)

    assert runtime["require_explicit_per_node_dram_quota"] is True
    assert runtime["per_node_aegis_dram_quota_bytes"] == quota_bytes
    assert runtime["per_node_aegis_dram_quota_tokens"] == quota_tokens
    assert runtime["kv_bytes_per_logical_token"] == 147_456
    assert compiled.dram_aegis_budget_tokens_per_node == quota_tokens
    assert compiled.dram_aegis_unallocatable_tail_bytes == tail

    parsed = resolve_aegis_per_node_dram_quota(
        runtime,
        fallback_tokens=1,
    )
    assert parsed.explicit
    assert parsed.as_dict()["effective_quota_bytes"] == quota_bytes - tail
    broken = dict(runtime)
    broken["per_node_aegis_dram_quota_tokens"] += 1
    with pytest.raises(ValueError, match="must equal floor"):
        resolve_aegis_per_node_dram_quota(broken, fallback_tokens=1)
    non_integer = dict(runtime)
    non_integer["per_node_aegis_dram_quota_bytes"] = float(quota_bytes)
    with pytest.raises(ValueError, match="must be a positive integer"):
        resolve_aegis_per_node_dram_quota(non_integer, fallback_tokens=1)


def test_compiled_fabric_binds_to_started_physical_cluster_before_jobs() -> None:
    group, axes = _first_axes("P1", policy="neighbor_replica", model="qwen3-8b")
    compiled = compile_deployment(
        group.deployment.resolved, axes, _modeled_bindings()
    )

    class ResourceBridge:
        topology = compiled.topology

    class AegisBridge:
        jobs: tuple[object, ...] = ()
        fabric = None

    class HiCacheBridge:
        transfers: dict[str, object] = {}
        fabric = None

    class Cluster:
        global_des = compiled.new_global_des()
        resource_bridge = ResourceBridge()
        aegis_transfer_bridge = AegisBridge()
        hicache_transfer_bridge = HiCacheBridge()

    result = compiled.bind_started_cluster(Cluster())

    assert result["copy_fabric_bound"] is True
    assert Cluster.aegis_transfer_bridge.fabric is compiled.copy_fabric
    assert Cluster.hicache_transfer_bridge.fabric is compiled.copy_fabric
    assert result["resource_count"] == len(compiled.transfer_resources.resources)


def test_compiled_dram_policy_rejects_cluster_quota_drift() -> None:
    group, axes = _first_axes("P1", policy="aegiskv", model="qwen3-8b")
    compiled = compile_deployment(
        group.deployment.resolved, axes, _modeled_bindings()
    )

    class ResourceBridge:
        topology = compiled.topology

    class AegisBridge:
        jobs: tuple[object, ...] = ()
        fabric = None

    class HiCacheBridge:
        transfers: dict[str, object] = {}
        fabric = None

    class Quota:
        def __init__(self, tokens: int) -> None:
            self.tokens = tokens

        def as_dict(self) -> dict[str, object]:
            return {
                "quota_bytes": compiled.dram_aegis_budget_bytes_per_node,
                "quota_tokens": self.tokens,
                "explicit": True,
            }

    class Cluster:
        global_des = compiled.new_global_des()
        resource_bridge = ResourceBridge()
        aegis_transfer_bridge = AegisBridge()
        hicache_transfer_bridge = HiCacheBridge()
        aegis_per_node_dram_quota = Quota(
            compiled.dram_aegis_budget_tokens_per_node + 1
        )

    with pytest.raises(ValueError, match="quota differs"):
        compiled.bind_started_cluster(Cluster())
    Cluster.aegis_per_node_dram_quota = Quota(
        compiled.dram_aegis_budget_tokens_per_node
    )
    assert compiled.bind_started_cluster(Cluster())[
        "per_node_dram_quota_bound"
    ] is True


def test_worker4_scenario_becomes_four_same_time_hard_fail_stop_events() -> None:
    group, axes = _first_axes(
        "P8", failure_severity="worker_4", policy="recompute", model="qwen3-8b"
    )
    bindings = _modeled_bindings(failure_target_selector="seed_mod_dense_order_v1")
    deployment = compile_deployment(group.deployment.resolved, axes, bindings)
    timeline = compile_failure_timeline(
        group.failure.resolved,
        axes,
        deployment,
        bindings,
        active_exposure_plan=_active_plan(group, axes, deployment),
        anchor_active_at_ns=0,
    )

    assert len(timeline.events) == 4
    assert {event.inject_at_ns for event in timeline.events} == {1_000_000}
    assert {event.failure_scope for event in timeline.events} == {"worker"}
    assert len({event.dp_rank for event in timeline.events}) == 4
    # 30 s repair + 100 ms replacement + 50 ms routing outage.
    assert {event.duration_ns for event in timeline.events} == {30_150_000_000}
    assert all(event.as_dict()["failure_mode"] == "hard_fail_stop" for event in timeline.events)
    for event in timeline.events:
        target = event.as_dict()["resolved_physical_target"]
        assert len(target["physical_gpu_ids"]) == 1
        assert target["service_unavailable_dp_ranks"] == [event.dp_rank]
        assert target["scheduler_process_ids"] == [f"dp{event.dp_rank}/pp0"]


class _HardCluster:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def schedule_fail_stop(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))

    def schedule_soft_failure(self, **_kwargs: object) -> None:  # pragma: no cover
        raise AssertionError("soft failure must never be selected")


def test_node_failure_installer_calls_only_hard_cluster_api() -> None:
    group, axes = _first_axes(
        "P1", fault_domain="node_1", policy="recompute", model="qwen3-8b"
    )
    bindings = _modeled_bindings()
    deployment = compile_deployment(group.deployment.resolved, axes, bindings)
    timeline = compile_failure_timeline(
        group.failure.resolved,
        axes,
        deployment,
        bindings,
        active_exposure_plan=_active_plan(group, axes, deployment),
        anchor_active_at_ns=0,
    )
    cluster = _HardCluster()

    assert timeline.install(cluster) == 1
    assert cluster.calls[0]["failure_scope"] == "node"
    assert cluster.calls[0]["node_id"] == "node0"
    assert cluster.calls[0]["payload"]["soft_metadata_failure_forbidden"] is True
    resolved = cluster.calls[0]["payload"]["resolved_physical_target"]
    assert resolved["node_dram_ids"] == ["node0"]
    assert len(resolved["physical_gpu_ids"]) == 8
    assert all(gpu_id.startswith("node0/") for gpu_id in resolved["physical_gpu_ids"])
    assert "dp_rank" not in cluster.calls[0]


def test_soft_only_runtime_is_rejected() -> None:
    group, axes = _first_axes(
        "P1", fault_domain="worker_1", policy="recompute", model="qwen3-8b"
    )
    bindings = _modeled_bindings()
    deployment = compile_deployment(group.deployment.resolved, axes, bindings)
    timeline = compile_failure_timeline(
        group.failure.resolved,
        axes,
        deployment,
        bindings,
        active_exposure_plan=_active_plan(group, axes, deployment),
        anchor_active_at_ns=0,
    )

    class SoftOnly:
        def schedule_soft_failure(self, **_kwargs: object) -> None:
            pass

    with pytest.raises(TypeError, match="refusing legacy soft metadata failure"):
        timeline.install(SoftOnly())


def test_no_failure_axis_emits_empty_timeline() -> None:
    group, axes = _first_axes(
        "P2", failure_scenario="no_failure", policy="recompute"
    )
    bindings = _modeled_bindings(
        failure_target_selector="", repair_delay_policy=""
    )
    deployment = compile_deployment(group.deployment.resolved, axes, bindings)
    readiness = failure_layer_readiness(
        group.failure.resolved, axes, deployment, bindings
    )
    timeline = compile_failure_timeline(
        group.failure.resolved, axes, deployment, bindings
    )

    assert readiness.ready
    assert timeline.events == ()
    assert timeline.scenario == "no_failure"


def test_missing_simulator_lineage_is_structured_with_frozen_32b_revision() -> None:
    group, axes = _first_axes("P1", model="qwen3-32b")
    readiness = deployment_layer_readiness(
        group.deployment.resolved, axes, RuntimeBindings()
    )
    by_code = {error.code: error for error in readiness.errors}

    assert readiness.ready is False
    assert "model_revision_missing" not in by_code
    assert by_code["simulator_revision_missing"].layer == "deployment"
    assert "hardware_timing_provenance_missing" not in by_code
    with pytest.raises(RuntimeNotReadyError) as caught:
        compile_deployment(group.deployment.resolved, axes, RuntimeBindings())
    assert caught.value.as_dict() == readiness.as_dict()


def test_modeled_hardware_requires_explicit_authorization() -> None:
    group, axes = _first_axes("P1")
    bindings = _modeled_bindings(allow_modeled_hardware=False)
    errors = deployment_layer_readiness(
        group.deployment.resolved, axes, bindings
    ).errors

    assert any(error.code == "modeled_hardware_not_authorized" for error in errors)


def test_p3_formula_keeps_pre_injection_physical_gpu_exposure_constant() -> None:
    group, axes = _first_axes("P3", policy="recompute", node_count=1)
    bindings = _modeled_bindings()
    one_node = compile_deployment(group.deployment.resolved, axes, bindings)
    one_timeline = compile_failure_timeline(
        group.failure.resolved,
        axes,
        one_node,
        bindings,
        active_exposure_plan=_active_plan(group, axes, one_node),
        anchor_active_at_ns=0,
    )
    axes["node_count"] = 8
    eight_nodes = compile_deployment(group.deployment.resolved, axes, bindings)
    eight_timeline = compile_failure_timeline(
        group.failure.resolved,
        axes,
        eight_nodes,
        bindings,
        active_exposure_plan=_active_plan(group, axes, eight_nodes),
        anchor_active_at_ns=0,
    )

    assert one_timeline.events[0].inject_at_ns == 4_000_000
    assert eight_timeline.events[0].inject_at_ns == 500_000
    assert (
        one_timeline.metadata["trigger_normalization"][
            "conditioned_cluster_gpu_time_exposure_ns"
        ]
        == eight_timeline.metadata["trigger_normalization"][
            "conditioned_cluster_gpu_time_exposure_ns"
        ]
        == 32_000_000
    )


def test_no_algorithm_1_compiles_and_enforces_zero_ordering_slack() -> None:
    group, axes = _first_axes("P4", algorithm_variant="no_algorithm_1")
    compiled = compile_deployment(
        group.deployment.resolved, axes, _modeled_bindings()
    )
    assert compiled.simulator_config_overlay["aegis_l1_l2"][
        "disable_slack_ordering"
    ] is True

    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config={
            "design_semantics": DESIGN_SEMANTICS,
            "committed_copy_protocol": True,
            "disable_slack_ordering": True,
        },
        hbm_total_tokens=128,
        dram_total_tokens=128,
        now_fn=lambda: 0.0,
    )
    ack = manager.place(
        AegisBackupPlan(
            rid="ablation",
            backup_key="ablation",
            source_dp_rank=0,
            target_dp_rank=1,
            token_count=16,
            critical_probability=1.0,
            slack_ms=10_000.0,
            allowed_tiers=(),
        )
    )
    assert manager.controller.request_profile("ablation").slack_ms == 0.0
    assert ack.placement_mode == "cost_landlord_no_slack"


def test_missing_deployment_axis_is_structured_instead_of_key_error() -> None:
    group, axes = _first_axes("P1")
    axes.pop("model")
    readiness = deployment_layer_readiness(
        group.deployment.resolved, axes, _modeled_bindings()
    )

    assert any(
        error.code == "axis_value_missing"
        and error.config_path == "#/axes/model"
        for error in readiness.errors
    )


def test_tp4_single_gpu_failure_fails_closed_instead_of_erasing_all_shards() -> None:
    group, axes = _first_axes("P1", policy="recompute", model="qwen3-32b")
    bindings = _modeled_bindings()
    deployment = compile_deployment(group.deployment.resolved, axes, bindings)
    failure = deepcopy(group.failure.resolved)
    failure["axes"] = {"fault_domain": ["gpu_1"]}
    axes["fault_domain"] = "gpu_1"
    errors = failure_layer_readiness(
        failure, axes, deployment, bindings
    ).errors

    assert any(
        error.code == "single_gpu_fail_stop_not_supported_for_tp"
        for error in errors
    )


def test_pp2_process_and_gpu_failure_fail_closed_instead_of_widening_to_worker() -> None:
    group, axes = _first_axes("P1", policy="recompute", model="qwen3-8b")
    bindings = _modeled_bindings()
    deployment = replace(
        compile_deployment(group.deployment.resolved, axes, bindings),
        pp_size=2,
    )
    for scenario, expected_code in (
        ("process_1", "single_process_fail_stop_not_supported_for_pp"),
        ("gpu_1", "single_gpu_fail_stop_not_supported_for_pp"),
    ):
        failure = deepcopy(group.failure.resolved)
        failure["axes"] = {"fault_domain": [scenario]}
        selected_axes = {**axes, "fault_domain": scenario}
        errors = failure_layer_readiness(
            failure, selected_axes, deployment, bindings
        ).errors
        assert any(error.code == expected_code for error in errors)


def test_failure_compiler_rejects_cross_node_partial_tp_target() -> None:
    group, axes = _first_axes(
        "P1", fault_domain="node_1", policy="recompute", model="qwen3-8b"
    )
    deployment = compile_deployment(
        group.deployment.resolved, axes, _modeled_bindings()
    )
    cross_node = OfflinePhysicalTopology(
        gpus=(
            PhysicalGpuSpec(PhysicalGpuId("node0", 0, 0, 0, 0), 1_000),
            PhysicalGpuSpec(PhysicalGpuId("node1", 0, 0, 0, 1), 1_000),
            PhysicalGpuSpec(PhysicalGpuId("node2", 0, 1, 0, 0), 1_000),
            PhysicalGpuSpec(PhysicalGpuId("node2", 1, 1, 0, 1), 1_000),
        ),
        nodes=tuple(NodeDramSpec(f"node{index}", 10_000) for index in range(3)),
    )
    deployment = replace(
        deployment,
        node_count=3,
        dp_size=2,
        pp_size=1,
        tp_size=2,
        topology=cross_node,
    )

    readiness = failure_layer_readiness(
        group.failure.resolved,
        axes,
        deployment,
        _modeled_bindings(),
    )
    assert readiness.ready is False
    assert any(
        error.code == "failure_target_not_representable"
        for error in readiness.errors
    )

    with pytest.raises(RuntimeNotReadyError) as caught:
        compile_failure_timeline(
            group.failure.resolved,
            axes,
            deployment,
            _modeled_bindings(),
        )

    assert caught.value.errors[0].code == "failure_target_not_representable"
    assert "would widen node failure" in caught.value.errors[0].message
