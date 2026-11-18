import math

import pytest

from sglsim.srt.simulation.aegis_peer_reserve import (
    AegisPeerHbmLedger,
    AegisPeerLedgerMode,
    AegisPeerReserveEnvelope,
    theorem_peer_budget_bytes,
)


def _envelope() -> AegisPeerReserveEnvelope:
    return AegisPeerReserveEnvelope(
        primary_growth_rate_upper_bytes_per_s=100.0,
        reclaim_terminal_upper_s=0.2,
        allocation_burst_upper_bytes=5,
        fragmentation_upper_bytes=7,
        envelope_version="reserve-v1",
        source="allocator_contract",
    )


def test_reserve_is_the_ceiling_of_growth_burst_and_fragmentation() -> None:
    envelope = _envelope()

    assert envelope.required_reserve_bytes == 32
    assert theorem_peer_budget_bytes(
        kv_capacity_bytes=1000,
        primary_occupied_bytes=600,
        alpha=0.5,
        envelope=envelope,
    ) == 368


def test_policy_fraction_can_be_tighter_than_physical_headroom() -> None:
    assert theorem_peer_budget_bytes(
        kv_capacity_bytes=1000,
        primary_occupied_bytes=100,
        alpha=0.25,
        envelope=_envelope(),
    ) == 250


def test_missing_hard_envelope_disables_theorem_peer_budget() -> None:
    assert theorem_peer_budget_bytes(
        kv_capacity_bytes=1000,
        primary_occupied_bytes=100,
        alpha=1.0,
        envelope=None,
    ) == 0


def test_invalid_or_future_envelopes_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        AegisPeerReserveEnvelope(
            primary_growth_rate_upper_bytes_per_s=math.inf,
            reclaim_terminal_upper_s=1.0,
            allocation_burst_upper_bytes=0,
            fragmentation_upper_bytes=0,
            envelope_version="bad",
            source="unit-test",
        )

    with pytest.raises(ValueError, match="future information"):
        AegisPeerReserveEnvelope(
            primary_growth_rate_upper_bytes_per_s=1.0,
            reclaim_terminal_upper_s=1.0,
            allocation_burst_upper_bytes=0,
            fragmentation_upper_bytes=0,
            envelope_version="bad",
            source="trace_oracle",
            uses_future_information=True,
        )


def test_budget_rejects_invalid_capacity_or_alpha() -> None:
    with pytest.raises(ValueError, match="primary occupancy exceeds"):
        theorem_peer_budget_bytes(
            kv_capacity_bytes=10,
            primary_occupied_bytes=11,
            alpha=0.5,
            envelope=_envelope(),
        )
    with pytest.raises(ValueError, match="alpha"):
        theorem_peer_budget_bytes(
            kv_capacity_bytes=10,
            primary_occupied_bytes=1,
            alpha=1.1,
            envelope=_envelope(),
        )


def test_ledger_consumes_and_replenishes_reserve_without_overcommit() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=600,
        peer_bytes=300,
        required_reserve_bytes=100,
    )

    ledger.grow_primary(60)
    assert ledger.mode is AegisPeerLedgerMode.DRAINING
    assert ledger.accounted_bytes == 1000
    with pytest.raises(RuntimeError, match="disabled"):
        ledger.admit_peer(1)

    ledger.terminal_reclaim_peer(60)
    assert ledger.mode is AegisPeerLedgerMode.NORMAL
    assert ledger.accounted_bytes == 1000


def test_primary_growth_beyond_remaining_reserve_is_atomic_refusal() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=600,
        peer_bytes=300,
        required_reserve_bytes=100,
    )

    with pytest.raises(RuntimeError, match="remaining reserve"):
        ledger.grow_primary(101)

    assert ledger.primary_bytes == 600
    assert ledger.remaining_reserve_bytes == 100
    assert ledger.accounted_bytes == 1000


def test_zero_copy_and_d2d_adoption_account_for_different_peak_bytes() -> None:
    zero_copy = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=600,
        peer_bytes=300,
        required_reserve_bytes=100,
    )
    zero_copy.adopt_zero_copy(40)
    assert (zero_copy.primary_bytes, zero_copy.peer_bytes) == (640, 260)
    assert zero_copy.mode is AegisPeerLedgerMode.NORMAL

    d2d = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=600,
        peer_bytes=250,
        required_reserve_bytes=100,
    )
    d2d.begin_d2d_adoption(40)
    assert d2d.d2d_pending_bytes == 40
    assert d2d.remaining_reserve_bytes == 100
    assert d2d.mode is AegisPeerLedgerMode.DRAINING
    d2d.finish_d2d_adoption(40)
    assert (d2d.primary_bytes, d2d.peer_bytes) == (640, 210)
    assert d2d.mode is AegisPeerLedgerMode.NORMAL


def test_d2d_adoption_cannot_spend_primary_safety_reserve() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=600,
        peer_bytes=300,
        required_reserve_bytes=100,
    )

    with pytest.raises(RuntimeError, match="protected primary reserve"):
        ledger.begin_d2d_adoption(1)

    assert ledger.primary_bytes == 600
    assert ledger.peer_bytes == 300
    assert ledger.remaining_reserve_bytes == 100
    assert ledger.d2d_pending_bytes == 0


def test_pending_d2d_blocks_new_peer_admission_without_debiting_reserve() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=500,
        peer_bytes=200,
        required_reserve_bytes=100,
        desired_peer_cap_bytes=400,
    )

    ledger.begin_d2d_adoption(50)
    assert ledger.remaining_reserve_bytes == 100
    with pytest.raises(RuntimeError, match="disabled"):
        ledger.admit_peer(1)


def test_pending_d2d_is_a_subset_of_both_accounted_copies() -> None:
    with pytest.raises(ValueError, match="source or destination occupancy"):
        AegisPeerHbmLedger(
            capacity_bytes=1000,
            primary_bytes=20,
            peer_bytes=100,
            required_reserve_bytes=0,
            d2d_pending_bytes=21,
        )


def test_capacity_shrink_requires_prior_reclaim() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=600,
        peer_bytes=300,
        required_reserve_bytes=100,
    )

    with pytest.raises(RuntimeError, match="capacity shrink"):
        ledger.shrink_capacity(950)
    assert ledger.capacity_bytes == 1000

    ledger.terminal_reclaim_peer(50)
    ledger.shrink_capacity(950)
    assert ledger.capacity_bytes == 950
    assert ledger.accounted_bytes == 950


def test_lowering_desired_cap_drains_without_freeing_actual_bytes() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=500,
        peer_bytes=300,
        required_reserve_bytes=100,
        desired_peer_cap_bytes=400,
    )

    ledger.set_desired_peer_cap(200)

    assert ledger.peer_bytes == 300
    assert ledger.peer_cap_overdraft_bytes == 100
    assert ledger.accounted_bytes == 900
    assert ledger.mode is AegisPeerLedgerMode.DRAINING
    with pytest.raises(RuntimeError, match="disabled"):
        ledger.admit_peer(1)


def test_terminal_reclaim_clears_cap_overdraft_only_after_actual_release() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=500,
        peer_bytes=300,
        required_reserve_bytes=100,
        desired_peer_cap_bytes=200,
    )

    ledger.terminal_reclaim_peer(60)
    assert ledger.peer_cap_overdraft_bytes == 40
    assert ledger.mode is AegisPeerLedgerMode.DRAINING

    ledger.terminal_reclaim_peer(40)
    assert ledger.peer_bytes == 200
    assert ledger.peer_cap_overdraft_bytes == 0
    assert ledger.mode is AegisPeerLedgerMode.NORMAL


def test_desired_cap_blocks_admission_even_with_physical_headroom() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=400,
        peer_bytes=200,
        required_reserve_bytes=100,
        desired_peer_cap_bytes=220,
    )

    ledger.admit_peer(20)
    assert ledger.peer_bytes == 220
    with pytest.raises(RuntimeError, match="desired peer cap"):
        ledger.admit_peer(1)


def test_capacity_shrink_tightens_but_never_materializes_policy_cap() -> None:
    ledger = AegisPeerHbmLedger(
        capacity_bytes=1000,
        primary_bytes=500,
        peer_bytes=100,
        required_reserve_bytes=100,
        desired_peer_cap_bytes=900,
    )

    ledger.shrink_capacity(800)

    assert ledger.capacity_bytes == 800
    assert ledger.desired_peer_cap_bytes == 800
    assert ledger.peer_bytes == 100
    assert ledger.mode is AegisPeerLedgerMode.NORMAL
