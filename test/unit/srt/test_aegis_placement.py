from __future__ import annotations

import pytest

from sglsim.srt.simulation.aegis_placement import (
    AegisClusterTopology,
    AegisNodeHostBudgetController,
    AegisNodeHostPool,
    AegisPathCostModel,
    AegisPersistentPool,
    AegisPlacementRequest,
    AegisTopologyAwarePlanner,
)


def _topology() -> AegisClusterTopology:
    return AegisClusterTopology.from_dict(
        {
            "dp_rank_to_node": [0, 1, 2, 3],
            "pp_stages_per_node": 4,
            "gpu_backup_requires_direct_retry_route": True,
            "host_pool_scope": "node_shared_process_external",
        },
        dp_size=4,
        pp_size=4,
    )


def _request(*, scope: str = "node") -> AegisPlacementRequest:
    return AegisPlacementRequest(
        rid="req-0",
        token_count=1024,
        source_dp_rank=0,
        critical_probability=0.9,
        slack_lcb_ms=0.0,
        failure_scope=scope,
    )


def test_pp4_gpu_copy_maps_each_stage_to_the_same_stage_on_target_replica() -> None:
    candidate = AegisTopologyAwarePlanner(
        topology=_topology(), costs=AegisPathCostModel()
    ).gpu_candidate(_request(), target_dp_rank=2)

    assert candidate.valid is True
    assert candidate.direct_retry_route_required is True
    assert candidate.reusable_dp_ranks == (2,)
    assert [row["pp_rank"] for row in candidate.pp_gpu_mapping] == [0, 1, 2, 3]
    assert candidate.pp_gpu_mapping[3] == {
        "pp_rank": 3,
        "source_gpu": "node0/gpu3",
        "target_gpu": "node2/gpu3",
    }
    assert candidate.backup_path == ("gpu_rdma",)
    assert candidate.recovery_path == ("direct_hbm_reuse",)


def test_gpu_copy_in_failed_node_domain_is_rejected() -> None:
    topology = AegisClusterTopology.from_dict(
        {"dp_rank_to_node": [0, 0, 1, 2], "pp_stages_per_node": 4},
        dp_size=4,
        pp_size=4,
    )
    candidate = AegisTopologyAwarePlanner(
        topology=topology, costs=AegisPathCostModel()
    ).gpu_candidate(_request(scope="node"), target_dp_rank=1)

    assert candidate.valid is False
    assert candidate.failure_scope_survives is False
    assert candidate.utility == float("-inf")


def test_failure_hazard_scales_expected_deadline_benefit() -> None:
    certain = AegisTopologyAwarePlanner(
        topology=_topology(),
        costs=AegisPathCostModel(
            failure_hazard_probability=1.0,
            normal_path_cost_weight=0.0,
            pressure_cost_weight=0.0,
        ),
    ).gpu_candidate(_request(), target_dp_rank=2)
    historical = AegisTopologyAwarePlanner(
        topology=_topology(),
        costs=AegisPathCostModel(
            failure_hazard_probability=0.01,
            normal_path_cost_weight=0.0,
            pressure_cost_weight=0.0,
        ),
    ).gpu_candidate(_request(), target_dp_rank=2)

    assert historical.deadline_benefit_ms == pytest.approx(
        certain.deadline_benefit_ms * 0.01,
        abs=1e-6,
    )
    assert historical.utility == historical.deadline_benefit_ms


def test_remote_target_host_backup_is_d2h_h2h_then_local_h2d() -> None:
    candidate = AegisTopologyAwarePlanner(
        topology=_topology(), costs=AegisPathCostModel()
    ).host_candidate(
        _request(), host_pool_node_id=1, retry_dp_rank=1
    )

    assert candidate.valid is True
    assert candidate.backup_path == ("d2h", "h2h")
    assert candidate.recovery_path == ("h2d",)
    assert candidate.host_pool_node_id == 1
    assert candidate.direct_retry_route_required is False


def test_host_copy_on_another_node_requires_h2h_plus_h2d_for_remote_retry() -> None:
    candidate = AegisTopologyAwarePlanner(
        topology=_topology(), costs=AegisPathCostModel()
    ).host_candidate(
        _request(), host_pool_node_id=1, retry_dp_rank=2
    )

    assert candidate.valid is True
    assert candidate.recovery_path == ("h2h", "h2d")
    assert candidate.recovery_ms > AegisTopologyAwarePlanner(
        topology=_topology(), costs=AegisPathCostModel()
    ).host_candidate(
        _request(), host_pool_node_id=1, retry_dp_rank=1
    ).recovery_ms


def test_source_host_copy_is_not_counted_for_node_failure() -> None:
    candidate = AegisTopologyAwarePlanner(
        topology=_topology(), costs=AegisPathCostModel()
    ).host_candidate(
        _request(scope="node"), host_pool_node_id=0, retry_dp_rank=1
    )

    assert candidate.valid is False
    assert candidate.failure_scope_survives is False
    assert candidate.deadline_benefit_ms == 0.0


def test_shared_host_pool_promotes_cache_payload_and_bounds_pinned_quota() -> None:
    pool = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=4096,
        protection_ratio=0.5,
        pinned_quota_tokens=512,
    )

    first = pool.admit(
        backup_key="cached",
        token_count=768,
        cache_resident_tokens=512,
        utility_density=2.0,
        source_node_id=0,
        failure_scope="node",
    )
    second = pool.admit(
        backup_key="new",
        token_count=256,
        utility_density=1.0,
        source_node_id=0,
        failure_scope="node",
    )

    assert first.admitted is True
    assert first.lease is not None
    assert first.lease.cache_reused_tokens == 512
    assert first.lease.copied_tokens == 256
    assert first.lease.pinned_tokens == 512
    assert first.lease.pageable_tokens == 256
    assert second.lease is not None
    assert second.lease.pinned_tokens == 0
    assert second.lease.pageable_tokens == 256
    assert pool.snapshot()["scope"] == "node_shared_process_external"


def test_shared_host_pool_replaces_lower_value_lease_but_not_higher_value() -> None:
    pool = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=1024,
        protection_ratio=0.5,
        pinned_quota_tokens=0,
    )
    low = pool.admit(
        backup_key="low",
        token_count=512,
        utility_density=0.2,
        source_node_id=0,
        failure_scope="node",
    )
    high = pool.admit(
        backup_key="high",
        token_count=512,
        utility_density=1.0,
        source_node_id=0,
        failure_scope="node",
    )
    rejected = pool.admit(
        backup_key="lower",
        token_count=512,
        utility_density=0.1,
        source_node_id=0,
        failure_scope="node",
    )

    assert low.admitted is True
    assert high.admitted is True
    assert high.evicted_backup_keys == ("low",)
    assert tuple(lease.version for lease in high.evicted_leases) == (1,)
    assert pool.get("low") is None
    assert pool.get("high") is not None
    assert rejected.admitted is False
    assert pool.get("high") is not None


def test_shared_host_pool_release_is_version_guarded() -> None:
    pool = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=1024,
        protection_ratio=1.0,
        pinned_quota_tokens=0,
    )
    admission = pool.admit(
        backup_key="same",
        token_count=128,
        utility_density=1.0,
        source_node_id=0,
        failure_scope="node",
        version=2,
    )

    assert admission.admitted
    assert pool.release("same", max_version=1) is None
    assert pool.get("same") is not None
    assert pool.release("same", max_version=2) is not None


def test_shared_host_pool_rejects_stale_replacement_version() -> None:
    pool = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=1024,
        protection_ratio=1.0,
        pinned_quota_tokens=0,
    )
    current = pool.admit(
        backup_key="same",
        token_count=128,
        utility_density=1.0,
        source_node_id=0,
        failure_scope="node",
        version=2,
    )
    stale = pool.admit(
        backup_key="same",
        token_count=256,
        utility_density=10.0,
        source_node_id=0,
        failure_scope="node",
        version=1,
    )

    assert current.admitted
    assert not stale.admitted
    assert "stale" in stale.reason
    assert pool.get("same").version == 2
    assert pool.get("same").token_count == 128


def test_shared_host_pool_charges_aligned_capacity_and_fragmentation() -> None:
    pool = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=1024,
        protection_ratio=1.0,
        pinned_quota_tokens=256,
        alignment_tokens=256,
    )

    admission = pool.admit(
        backup_key="unaligned",
        token_count=257,
        utility_density=1.0,
        source_node_id=0,
        failure_scope="node",
    )

    assert admission.admitted
    assert admission.lease is not None
    assert admission.lease.token_count == 257
    assert admission.lease.allocated_tokens == 512
    assert admission.lease.internal_fragment_tokens == 255
    assert admission.lease.pinned_tokens == 256
    assert admission.lease.pageable_tokens == 256
    assert pool.allocation_tokens(0) == 0
    assert pool.allocation_tokens(1) == 256
    assert pool.allocation_tokens(257) == 512
    snapshot = pool.snapshot()
    assert snapshot["protected_used_tokens"] == 512
    assert snapshot["internal_fragment_tokens"] == 255
    assert snapshot["alignment_tokens"] == 256

    partially_pinned = AegisNodeHostPool(
        node_id=2,
        capacity_tokens=1024,
        protection_ratio=1.0,
        pinned_quota_tokens=300,
        alignment_tokens=256,
    ).admit(
        backup_key="aligned-pinned",
        token_count=257,
        utility_density=1.0,
        source_node_id=0,
        failure_scope="node",
    )
    assert partially_pinned.lease is not None
    assert partially_pinned.lease.pinned_tokens == 256
    assert partially_pinned.lease.pageable_tokens == 256


def test_shared_host_pool_region_limit_replaces_only_lower_utility() -> None:
    pool = AegisNodeHostPool(
        node_id=1,
        capacity_tokens=4096,
        protection_ratio=1.0,
        pinned_quota_tokens=0,
        alignment_tokens=256,
        max_regions=1,
    )
    first = pool.admit(
        backup_key="low",
        token_count=128,
        utility_density=0.2,
        source_node_id=0,
        failure_scope="node",
    )
    replacement = pool.admit(
        backup_key="high",
        token_count=128,
        utility_density=1.0,
        source_node_id=0,
        failure_scope="node",
    )
    rejected = pool.admit(
        backup_key="lower",
        token_count=128,
        utility_density=0.1,
        source_node_id=0,
        failure_scope="node",
    )

    assert first.admitted
    assert replacement.admitted
    assert replacement.evicted_backup_keys == ("low",)
    assert not rejected.admitted
    assert rejected.reason == "node Host region limit exhausted"
    assert pool.get("high") is not None
    assert pool.snapshot()["max_regions"] == 1


def test_algorithm2_prefers_positive_recovery_utility_over_recompute() -> None:
    planner = AegisTopologyAwarePlanner(
        topology=_topology(),
        costs=AegisPathCostModel(recompute_ms_per_token=0.2),
    )
    request = _request()
    host = planner.host_candidate(
        request, host_pool_node_id=1, retry_dp_rank=1
    )
    recompute = planner.recompute_candidate(request, retry_dp_rank=1)

    assert host.utility > 0.0
    assert planner.choose([host, recompute]).tier == "host"


def test_l3_candidate_requires_storage_and_committed_manifest_path() -> None:
    planner = AegisTopologyAwarePlanner(
        topology=_topology(),
        costs=AegisPathCostModel(recompute_ms_per_token=0.2),
    )

    available = planner.l3_candidate(
        _request(),
        retry_dp_rank=3,
        storage_available=True,
        manifest_quorum=2,
    )
    unavailable = planner.l3_candidate(
        _request(),
        retry_dp_rank=3,
        storage_available=False,
        manifest_quorum=2,
    )

    assert available.valid is True
    assert available.backup_path == ("d2h", "l3_write")
    assert available.recovery_path == ("l3_read", "h2d")
    assert available.utility > 0.0
    assert unavailable.valid is False


def test_persistent_pool_manifest_version_and_utility_eviction() -> None:
    pool = AegisPersistentPool(capacity_tokens=512, manifest_quorum=2)
    low = pool.admit(
        backup_key="low",
        token_count=512,
        utility_density=0.1,
        version=1,
    )
    high = pool.admit(
        backup_key="high",
        token_count=512,
        utility_density=1.0,
        version=3,
    )

    assert low.admitted is True
    assert high.admitted is True
    assert [lease.backup_key for lease in high.evicted_leases] == ["low"]
    assert high.lease is not None
    assert high.lease.committed is True
    assert high.lease.manifest_quorum == 2
    assert high.lease.manifest_version == 3
    assert pool.release("high", max_version=2) is None
    assert pool.get("high") is not None


def test_node_host_beta_uses_adaptive_overflow_window() -> None:
    controller = AegisNodeHostBudgetController(
        node_id=1,
        capacity_tokens=1000,
        initial_beta=0.25,
        window_min_requests=2,
        window_max_requests=10,
        window_target_ms=1000.0,
        overflow_trigger_tokens=100,
        demand_ewma=1.0,
        publish_smoothing=1.0,
        hysteresis=0.0,
        safety_margin=0.0,
        burst_reserve_tokens=0,
    )
    started = controller._window_started

    assert controller.observe(
        demand_tokens=100,
        admitted_tokens=100,
        unmet_tokens=0,
        normal_pressure=0.0,
        protected_used_tokens=100,
        now=started + 0.1,
    ) is None
    update = controller.observe(
        demand_tokens=200,
        admitted_tokens=0,
        unmet_tokens=200,
        normal_pressure=0.0,
        protected_used_tokens=100,
        now=started + 0.2,
    )

    assert update is not None
    assert update["trigger"] == "overflow_shock"
    assert update["D"] == 300
    assert update["U"] == 100
    assert update["O"] == 200
    assert controller.beta == 1.0
