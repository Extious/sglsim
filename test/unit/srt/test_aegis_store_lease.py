import pytest

from sglsim.srt.simulation.aegis_store_lease import (
    ControllerLeaseView,
    StoreLeaseQuota,
    StoreLeaseState,
    StoreMedium,
    StoreObjectIdentity,
    StoreObjectState,
    StoreTierLeaseMaster,
    controller_safe_expiry_ms,
)


def _identity(name: str = "group") -> StoreObjectIdentity:
    return StoreObjectIdentity(
        object_id=name,
        version=7,
        commit_digest=f"digest-{name}",
        placement_set_digest=f"placement-{name}",
    )


def _master(
    *,
    medium: StoreMedium = StoreMedium.SSD,
    size: int = 100,
    survival_cap: int = 1000,
    dram_cap: int = 1000,
) -> tuple[StoreTierLeaseMaster, StoreObjectIdentity]:
    master = StoreTierLeaseMaster(
        quota=StoreLeaseQuota(
            survival_bytes=survival_cap,
            dram_bytes=dram_cap,
        )
    )
    identity = _identity()
    master.register_object(
        StoreObjectState(
            identity=identity,
            actual_bytes=size,
            medium=medium,
            survives_failure_domains=frozenset({"source-a"}),
        )
    )
    return master, identity


def _acquire(
    master: StoreTierLeaseMaster,
    identity: StoreObjectIdentity,
    *,
    action_id: str,
    min_medium: StoreMedium,
    now_ms: int = 10,
    ttl_ms: int = 100,
) -> object:
    return master.acquire(
        action_id=action_id,
        identity=identity,
        expected_actual_bytes=100,
        failure_domain="source-a",
        min_medium=min_medium,
        ttl_ms=ttl_ms,
        placement_timeout_ms=50,
        now_ms=now_ms,
        authority_epoch=master.authority_epoch,
    )


def test_existing_ssd_object_activates_and_charges_actual_bytes() -> None:
    master, identity = _master()
    record = _acquire(
        master,
        identity,
        action_id="keep-ssd",
        min_medium=StoreMedium.SSD,
    )

    assert record.state is StoreLeaseState.ACTIVE
    assert master.used_quota == StoreLeaseQuota(100, 0)
    assert master.deterministic_protection(action_id="keep-ssd", now_ms=20)


def test_dram_promotion_is_pending_and_has_zero_value_until_commit() -> None:
    master, identity = _master(medium=StoreMedium.SSD)
    record = _acquire(
        master,
        identity,
        action_id="promote",
        min_medium=StoreMedium.DRAM,
    )

    assert record.state is StoreLeaseState.PENDING
    assert master.used_quota == StoreLeaseQuota(100, 100)
    assert not master.deterministic_protection(action_id="promote", now_ms=20)

    ack = master.commit_placement(
        action_id="promote",
        completed_medium=StoreMedium.DRAM,
        now_ms=30,
        authority_epoch=1,
    )
    assert ack.actual_bytes == 100
    assert master.deterministic_protection(action_id="promote", now_ms=31)


def test_pending_reservation_prevents_dram_overcommit() -> None:
    master, identity = _master(dram_cap=100)
    _acquire(
        master,
        identity,
        action_id="first",
        min_medium=StoreMedium.DRAM,
    )
    second = _identity("second")
    master.register_object(
        StoreObjectState(
            second,
            actual_bytes=100,
            medium=StoreMedium.SSD,
            survives_failure_domains=frozenset({"source-a"}),
        )
    )

    with pytest.raises(MemoryError, match="DRAM quota"):
        master.acquire(
            action_id="second",
            identity=second,
            expected_actual_bytes=100,
            failure_domain="source-a",
            min_medium=StoreMedium.DRAM,
            ttl_ms=100,
            placement_timeout_ms=50,
            now_ms=10,
            authority_epoch=1,
        )
    assert "second" not in master.records


def test_same_object_leases_share_bytes_and_strongest_tier_wins() -> None:
    master, identity = _master()
    _acquire(
        master,
        identity,
        action_id="ssd",
        min_medium=StoreMedium.SSD,
    )
    _acquire(
        master,
        identity,
        action_id="dram",
        min_medium=StoreMedium.DRAM,
    )

    assert master.used_quota == StoreLeaseQuota(100, 100)
    master.release(action_id="dram", authority_epoch=1)
    assert master.used_quota == StoreLeaseQuota(100, 0)


def test_same_physical_placement_can_protect_two_source_domains_once() -> None:
    master = StoreTierLeaseMaster(
        quota=StoreLeaseQuota(survival_bytes=100, dram_bytes=0)
    )
    identity = _identity()
    master.register_object(
        StoreObjectState(
            identity=identity,
            actual_bytes=100,
            medium=StoreMedium.SSD,
            survives_failure_domains=frozenset({"source-a", "source-b"}),
        )
    )
    for action_id, domain in (("a", "source-a"), ("b", "source-b")):
        master.acquire(
            action_id=action_id,
            identity=identity,
            expected_actual_bytes=100,
            failure_domain=domain,
            min_medium=StoreMedium.SSD,
            ttl_ms=100,
            placement_timeout_ms=50,
            now_ms=10,
            authority_epoch=1,
        )

    assert master.used_quota == StoreLeaseQuota(100, 0)


def test_logically_equal_content_in_distinct_placements_is_charged_twice() -> None:
    master = StoreTierLeaseMaster(
        quota=StoreLeaseQuota(survival_bytes=200, dram_bytes=0)
    )
    first = StoreObjectIdentity("group", 7, "same-content", "replica-set-a")
    second = StoreObjectIdentity("group", 7, "same-content", "replica-set-b")
    for identity, domain in ((first, "source-a"), (second, "source-b")):
        master.register_object(
            StoreObjectState(
                identity,
                100,
                StoreMedium.SSD,
                frozenset({domain}),
            )
        )
        master.acquire(
            action_id=domain,
            identity=identity,
            expected_actual_bytes=100,
            failure_domain=domain,
            min_medium=StoreMedium.SSD,
            ttl_ms=100,
            placement_timeout_ms=50,
            now_ms=10,
            authority_epoch=1,
        )

    assert master.used_quota == StoreLeaseQuota(200, 0)


def test_placement_must_survive_the_named_source_failure_domain() -> None:
    master, identity = _master()
    with pytest.raises(ValueError, match="does not survive"):
        master.acquire(
            action_id="wrong-domain",
            identity=identity,
            expected_actual_bytes=100,
            failure_domain="source-b",
            min_medium=StoreMedium.SSD,
            ttl_ms=100,
            placement_timeout_ms=50,
            now_ms=10,
            authority_epoch=1,
        )
    assert master.used_quota == StoreLeaseQuota(0, 0)


def test_byte_or_identity_mismatch_is_rejected_without_charge() -> None:
    master, identity = _master()
    with pytest.raises(ValueError, match="byte estimate"):
        master.acquire(
            action_id="stale-size",
            identity=identity,
            expected_actual_bytes=99,
            failure_domain="source-a",
            min_medium=StoreMedium.SSD,
            ttl_ms=100,
            placement_timeout_ms=50,
            now_ms=10,
            authority_epoch=1,
        )
    with pytest.raises(ValueError, match="complete immutable"):
        master.acquire(
            action_id="stale-version",
            identity=StoreObjectIdentity(
                "group", 8, "digest-group", "placement-group"
            ),
            expected_actual_bytes=100,
            failure_domain="source-a",
            min_medium=StoreMedium.SSD,
            ttl_ms=100,
            placement_timeout_ms=50,
            now_ms=10,
            authority_epoch=1,
        )
    assert master.used_quota == StoreLeaseQuota(0, 0)


def test_action_replay_is_idempotent_but_parameter_change_is_rejected() -> None:
    master, identity = _master()
    first = _acquire(
        master,
        identity,
        action_id="same",
        min_medium=StoreMedium.SSD,
    )
    second = _acquire(
        master,
        identity,
        action_id="same",
        min_medium=StoreMedium.SSD,
    )
    assert first is second

    with pytest.raises(ValueError, match="changed immutable request"):
        _acquire(
            master,
            identity,
            action_id="same",
            min_medium=StoreMedium.DRAM,
        )


def test_active_lease_blocks_demotion_but_release_allows_it() -> None:
    master, identity = _master(medium=StoreMedium.DRAM)
    _acquire(
        master,
        identity,
        action_id="keep-dram",
        min_medium=StoreMedium.DRAM,
    )
    with pytest.raises(PermissionError, match="blocks"):
        master.set_object_medium(
            identity=identity,
            new_medium=StoreMedium.SSD,
            now_ms=20,
            authority_epoch=1,
        )

    master.release(action_id="keep-dram", authority_epoch=1)
    master.set_object_medium(
        identity=identity,
        new_medium=StoreMedium.SSD,
        now_ms=20,
        authority_epoch=1,
    )
    assert master.objects[identity].medium is StoreMedium.SSD


def test_expiry_releases_quota_and_allows_demotion() -> None:
    master, identity = _master(medium=StoreMedium.DRAM)
    _acquire(
        master,
        identity,
        action_id="short",
        min_medium=StoreMedium.DRAM,
        now_ms=10,
        ttl_ms=20,
    )
    assert not master.deterministic_protection(action_id="short", now_ms=30)
    assert master.records["short"].state is StoreLeaseState.EXPIRED
    assert master.used_quota == StoreLeaseQuota(0, 0)
    master.set_object_medium(
        identity=identity,
        new_medium=StoreMedium.SSD,
        now_ms=30,
        authority_epoch=1,
    )


def test_pending_timeout_has_no_value_and_releases_reservation() -> None:
    master, identity = _master()
    _acquire(
        master,
        identity,
        action_id="late-promotion",
        min_medium=StoreMedium.DRAM,
    )
    master.expire_due(now_ms=60)
    assert master.records["late-promotion"].state is StoreLeaseState.FAILED
    assert master.used_quota == StoreLeaseQuota(0, 0)


def test_controller_uses_local_send_time_not_master_absolute_clock() -> None:
    safe = controller_safe_expiry_ms(
        send_ms=1_000,
        receive_ms=1_025,
        ttl_ms=100,
        clock_guard_ms=5,
    )
    assert safe == 1_095

    master, identity = _master()
    _acquire(
        master,
        identity,
        action_id="local-clock",
        min_medium=StoreMedium.SSD,
    )
    ack = master.ack(action_id="local-clock", now_ms=20)
    view = ControllerLeaseView(ack=ack, safe_expiry_ms=safe)
    assert view.is_usable_at(1_094)
    assert view.covers_horizon(1_094)
    assert not view.covers_horizon(1_095)


def test_slow_ack_has_no_safe_future_interval() -> None:
    assert (
        controller_safe_expiry_ms(
            send_ms=100,
            receive_ms=200,
            ttl_ms=100,
        )
        is None
    )


def test_failover_fences_old_epoch_preserves_active_and_drops_pending() -> None:
    master, identity = _master(medium=StoreMedium.DRAM)
    _acquire(
        master,
        identity,
        action_id="active",
        min_medium=StoreMedium.SSD,
        now_ms=10,
        ttl_ms=50,
    )
    pending_identity = _identity("pending")
    master.register_object(
        StoreObjectState(
            pending_identity,
            100,
            StoreMedium.SSD,
            frozenset({"source-a"}),
        )
    )
    _acquire(
        master,
        pending_identity,
        action_id="pending",
        min_medium=StoreMedium.DRAM,
        now_ms=10,
        ttl_ms=50,
    )

    master.failover(new_authority_epoch=2, now_ms=40)
    assert master.records["active"].master_expiry_ms == 90
    assert master.records["active"].authority_epoch == 2
    assert master.records["pending"].state is StoreLeaseState.FAILED
    assert master.deterministic_protection(action_id="active", now_ms=89)
    with pytest.raises(PermissionError, match="stale"):
        master.release(action_id="active", authority_epoch=1)


def test_renewal_extends_but_never_shortens_master_expiry() -> None:
    master, identity = _master()
    _acquire(
        master,
        identity,
        action_id="renew",
        min_medium=StoreMedium.SSD,
        now_ms=10,
        ttl_ms=100,
    )
    master.renew(
        action_id="renew",
        ttl_ms=20,
        now_ms=20,
        authority_epoch=1,
    )
    assert master.records["renew"].master_expiry_ms == 110
    master.renew(
        action_id="renew",
        ttl_ms=100,
        now_ms=30,
        authority_epoch=1,
    )
    assert master.records["renew"].master_expiry_ms == 130
