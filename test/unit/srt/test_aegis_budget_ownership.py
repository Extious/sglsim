from __future__ import annotations

import pytest

from sglsim.srt.managers.pipeline_parallel import DPBudgetSnapshot
from sglsim.srt.simulation.aegis_budget import (
    DPBudgetPublisher,
    HostAdmissionRequest,
    NodeHostCapacityArbiter,
    TierWindowState,
)


def _budget(
    dp_rank: int,
    *,
    version: int = 1,
    window: int = 0,
    host_limit: int = 80,
) -> DPBudgetSnapshot:
    return DPBudgetSnapshot(
        dp_rank=dp_rank,
        budget_version=version,
        window_id=window,
        alpha=0.25,
        beta=host_limit / 100.0,
        hbm_limit_tokens_per_stage=(25, 25, 25, 25),
        host_policy_limit_tokens=host_limit,
        valid_from_logical_time=float(window),
        objective_id=f"obj-{dp_rank}-{version}",
        solver_certificate_id=f"cert-{dp_rank}-{version}",
    )


def _request(
    key: str,
    *,
    dp_rank: int,
    pp_rank: int = 0,
    version: int = 1,
    tokens: int,
    utility: float = 1.0,
) -> HostAdmissionRequest:
    return HostAdmissionRequest(
        lease_key=key,
        request_id=f"request-{key}",
        workflow_id="workflow-0",
        chunk_id=0,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        budget_version=version,
        token_count=tokens,
        utility=utility,
    )


def test_one_dp_budget_version_is_shared_and_acknowledged_by_all_four_stages() -> None:
    publisher = DPBudgetPublisher(dp_rank=2, pp_size=4)
    first = _budget(2)
    publisher.publish(first)

    assert [publisher.view_for_stage(pp) for pp in range(4)] == [first] * 4
    assert publisher.acknowledge(pp_rank=0, budget_version=1) is False
    assert publisher.acknowledge(pp_rank=1, budget_version=1) is False
    assert publisher.acknowledge(pp_rank=2, budget_version=1) is False
    assert publisher.acknowledge(pp_rank=3, budget_version=1) is True
    assert publisher.acknowledged_pp_ranks == (0, 1, 2, 3)

    second = _budget(2, version=2, window=1, host_limit=70)
    publisher.publish(second)
    assert publisher.view_for_stage(0) is second
    assert publisher.acknowledged_pp_ranks == ()


def test_budget_acknowledgements_can_be_reset_for_worker_generation_replay() -> None:
    publisher = DPBudgetPublisher(dp_rank=2, pp_size=4)
    publisher.publish(_budget(2))
    for pp_rank in range(4):
        publisher.acknowledge(pp_rank=pp_rank, budget_version=1)

    publisher.reset_acknowledgements()

    assert publisher.current == _budget(2)
    assert publisher.acknowledged_pp_ranks == ()


def test_budget_publisher_rejects_partial_or_stale_publication() -> None:
    publisher = DPBudgetPublisher(dp_rank=0, pp_size=4)
    publisher.publish(_budget(0))
    publisher.acknowledge(pp_rank=0, budget_version=1)

    with pytest.raises(RuntimeError, match="every PP stage"):
        publisher.publish(_budget(0, version=2, window=1))
    with pytest.raises(ValueError, match="stale or future"):
        publisher.acknowledge(pp_rank=1, budget_version=2)


def test_budget_snapshot_wire_round_trip_is_stable() -> None:
    budget = _budget(2, version=3, window=2, host_limit=70)

    restored = DPBudgetSnapshot.from_dict(budget.to_dict())

    assert restored == budget
    assert isinstance(restored.hbm_limit_tokens_per_stage, tuple)


def test_node_arbiter_distinguishes_dp_beta_from_physical_node_capacity() -> None:
    arbiter = NodeHostCapacityArbiter(
        node_id=7,
        physical_capacity_tokens=100,
        allowed_dp_ranks=(0, 1),
    )
    arbiter.publish_dp_budget(_budget(0, host_limit=80))
    arbiter.publish_dp_budget(_budget(1, host_limit=80))

    first = arbiter.admit(_request("dp0-70", dp_rank=0, tokens=70))
    dp_budget_denial = arbiter.admit(
        _request("dp0-20", dp_rank=0, tokens=20, utility=4.0)
    )
    node_denial = arbiter.admit(
        _request("dp1-40", dp_rank=1, tokens=40, utility=6.0)
    )

    assert first.admitted is True
    assert dp_budget_denial.admitted is False
    assert dp_budget_denial.reason == "dp_policy_budget"
    assert dp_budget_denial.unmet_tokens == 20
    assert dp_budget_denial.unmet_utility == 4.0
    assert node_denial.admitted is False
    assert node_denial.reason == "node_physical_capacity"
    assert node_denial.dp_policy_limit_tokens == 80
    assert arbiter.resident_tokens == 70
    assert arbiter.dp_resident_tokens(0) == 70
    assert arbiter.dp_resident_tokens(1) == 0


def test_release_is_version_guarded_and_frees_physical_capacity() -> None:
    arbiter = NodeHostCapacityArbiter(
        node_id=0,
        physical_capacity_tokens=100,
        allowed_dp_ranks=(0, 1),
    )
    arbiter.publish_dp_budget(_budget(0, host_limit=80))
    arbiter.publish_dp_budget(_budget(1, host_limit=80))
    lease = arbiter.admit(_request("a", dp_rank=0, tokens=70)).lease
    assert lease is not None

    with pytest.raises(ValueError, match="version"):
        arbiter.release("a", expected_lease_version=lease.lease_version + 1)
    assert arbiter.release(
        "a", expected_lease_version=lease.lease_version
    ) == lease
    admitted = arbiter.admit(_request("b", dp_rank=1, tokens=80))
    assert admitted.admitted is True
    assert arbiter.resident_tokens == 80


def test_stale_budget_and_budget_below_residency_are_rejected() -> None:
    arbiter = NodeHostCapacityArbiter(
        node_id=0,
        physical_capacity_tokens=100,
        allowed_dp_ranks=(0,),
    )
    arbiter.publish_dp_budget(_budget(0, host_limit=80))
    arbiter.admit(_request("a", dp_rank=0, tokens=70))

    with pytest.raises(ValueError, match="stale or future"):
        arbiter.admit(_request("stale", dp_rank=0, version=2, tokens=1))
    with pytest.raises(RuntimeError, match="lower than current"):
        arbiter.publish_dp_budget(
            _budget(0, version=2, window=1, host_limit=60)
        )


def test_window_state_keeps_stock_flow_and_unmet_metrics_separate() -> None:
    state = TierWindowState(
        dp_rank=0,
        window_id=3,
        budget_version=4,
        unique_offered=100,
        attempted_by_tier=(("HBM", 80), ("HOST", 20)),
        admitted_by_tier=(("HBM", 60), ("HOST", 10)),
        resident_start_by_tier=(("HBM", 40), ("HOST", 30)),
        resident_end_by_tier=(("HBM", 70), ("HOST", 25)),
        released_by_tier=(("HBM", 20), ("HOST", 10)),
        evicted_by_tier=(("HBM", 10), ("HOST", 5)),
        unmet_tokens_by_tier=(("HBM", 20), ("HOST", 10)),
        unmet_utility_by_tier=(("HBM", 7.0), ("HOST", 2.0)),
        normal_pressure=0.4,
        recovery_p99_s=0.08,
        slo_miss_rate=0.02,
    )

    assert dict(state.attempted_by_tier)["HBM"] == 80
    assert dict(state.resident_end_by_tier)["HBM"] == 70
    assert dict(state.unmet_tokens_by_tier)["HBM"] == 20
    assert not hasattr(state, "D")
    assert not hasattr(state, "U")
    assert not hasattr(state, "O")
    assert not hasattr(state, "Z")
