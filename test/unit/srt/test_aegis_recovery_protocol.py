from __future__ import annotations

from dataclasses import replace

import pytest

from sglsim.srt.simulation.aegis_recovery_protocol import (
    DurableContinuationState,
    DurableInputObject,
    OutputLogEntry,
    ProtocolViolation,
    RecoveryContinuationContract,
    RouterOutputJournal,
    ShadowCatalog,
    ShadowState,
    ShardCommitAck,
    StoreCommitCatalog,
    StoreCommitRecord,
    StoreComponentAck,
    StoreComponentRef,
    TokenGroup,
    build_output_replay_plan,
    compute_input_manifest_digest,
    output_log_genesis_digest,
)


RANKS = ("tp0", "tp1")
INITIAL_SAMPLING_POSITION = 10
INITIAL_STATE_DIGEST = "initial-state-sha256"
INPUT_OBJECTS = (
    DurableInputObject(
        object_id="durable-token-log",
        content_digest="input-sha256",
        schema_version="tokens-v1",
        storage_failure_domain="router-quorum",
    ),
)


def _output_entries(
    *token_ids: int,
    input_objects: tuple[DurableInputObject, ...] = INPUT_OBJECTS,
) -> tuple[OutputLogEntry, ...]:
    tail = output_log_genesis_digest(
        input_manifest_digest=compute_input_manifest_digest(input_objects),
        initial_state_digest=INITIAL_STATE_DIGEST,
        initial_sampling_position=INITIAL_SAMPLING_POSITION,
    )
    entries = []
    for sequence, token_id in enumerate(token_ids, start=1):
        entry = OutputLogEntry.committed(
            sequence=sequence,
            token_id=token_id,
            next_sampling_position=INITIAL_SAMPLING_POSITION + sequence,
            previous_entry_digest=tail,
            next_state_digest=f"state-{sequence}-sha256",
        )
        entries.append(entry)
        tail = entry.entry_digest
    return tuple(entries)


def _continuation_contract(
    *,
    entries: tuple[OutputLogEntry, ...] = (),
    input_objects: tuple[DurableInputObject, ...] = INPUT_OBJECTS,
) -> RecoveryContinuationContract:
    final_state_digest = (
        entries[-1].next_state_digest if entries else INITIAL_STATE_DIGEST
    )
    tail_digest = (
        entries[-1].entry_digest
        if entries
        else output_log_genesis_digest(
            input_manifest_digest=compute_input_manifest_digest(input_objects),
            initial_state_digest=INITIAL_STATE_DIGEST,
            initial_sampling_position=INITIAL_SAMPLING_POSITION,
        )
    )
    return RecoveryContinuationContract(
        source_failure_domain="worker-group:source",
        target_compatibility_hash="compat-v1",
        target_prefix_hash="prefix-v1",
        input_objects=input_objects,
        state=DurableContinuationState(
            schema_version="continuation-v1",
            state_digest=final_state_digest,
            compatibility_hash="compat-v1",
            prefix_hash="prefix-v1",
            next_output_sequence=len(entries) + 1,
            next_sampling_position=INITIAL_SAMPLING_POSITION + len(entries),
            output_log_tail_digest=tail_digest,
        ),
    )


def _acks(
    token_start: int,
    token_end: int,
    *,
    creation_epoch: int = 1,
    action_id: str = "action-1",
    digest: str | None = None,
) -> tuple[ShardCommitAck, ...]:
    content_digest = digest or f"digest-{token_start}-{token_end}"
    return tuple(
        ShardCommitAck(
            rank_id=rank_id,
            token_start=token_start,
            token_end=token_end,
            content_digest=content_digest,
            shard_digest=f"{rank_id}-bytes-{token_start}-{token_end}",
            source_execution="source-execution-1",
            source_block_generation=7 + rank_offset,
            destination_handle=f"{rank_id}-page-{token_start}-{token_end}",
            destination_generation=70 + rank_offset + token_start,
            action_id=action_id,
            creation_epoch=creation_epoch,
        )
        for rank_offset, rank_id in enumerate(RANKS)
    )


def _catalog_with_two_groups() -> ShadowCatalog:
    catalog = ShadowCatalog(issuer_epoch=1, required_ranks=RANKS)
    catalog.commit_group(group_id="g0", acks=_acks(0, 4))
    catalog.commit_group(group_id="g1", acks=_acks(4, 8))
    return catalog


def _store_components(
    token_start: int,
    token_end: int,
    *,
    group_id: str,
    action_id: str,
    creation_epoch: int = 1,
) -> tuple[StoreComponentAck, ...]:
    content_digest = f"prefix-{token_start}-{token_end}"
    components = []
    for offset, role in enumerate(("tp0:k", "tp0:v", "tp1:k", "tp1:v")):
        components.append(
            StoreComponentAck(
                group_id=group_id,
                component_role=role,
                object_key=f"kv/{group_id}/{role}",
                object_checksum=f"checksum-{group_id}-{role}",
                token_start=token_start,
                token_end=token_end,
                compatibility_hash="compat-v1",
                content_digest=content_digest,
                source_execution="source-execution-1",
                source_block_generation=20 + offset,
                action_id=action_id,
                creation_epoch=creation_epoch,
            )
        )
    return tuple(components)


def _store_record(
    components: tuple[StoreComponentAck, ...],
) -> StoreCommitRecord:
    first = components[0]
    return StoreCommitRecord(
        group_id=first.group_id,
        record_key=f"commit/{first.group_id}",
        record_checksum=f"commit-checksum-{first.group_id}",
        token_start=first.token_start,
        token_end=first.token_end,
        compatibility_hash=first.compatibility_hash,
        content_digest=first.content_digest,
        source_execution=first.source_execution,
        action_id=first.action_id,
        creation_epoch=first.creation_epoch,
        components=tuple(
            StoreComponentRef(
                component_role=component.component_role,
                object_key=component.object_key,
                object_checksum=component.object_checksum,
                source_block_generation=component.source_block_generation,
            )
            for component in components
        ),
    )


def test_group_commit_requires_all_ranks_one_identity_and_no_prefix_hole() -> None:
    catalog = ShadowCatalog(issuer_epoch=1, required_ranks=RANKS)

    with pytest.raises(ProtocolViolation, match="all required ranks"):
        catalog.commit_group(group_id="missing-rank", acks=_acks(0, 4)[:1])

    mismatched = list(_acks(0, 4))
    mismatched[1] = ShardCommitAck(
        **{
            **mismatched[1].__dict__,
            "content_digest": "different-content",
        }
    )
    with pytest.raises(ProtocolViolation, match="identities disagree"):
        catalog.commit_group(group_id="mismatched", acks=mismatched)

    with pytest.raises(ProtocolViolation, match="prefix hole"):
        catalog.commit_group(group_id="hole", acks=_acks(4, 8))

    catalog.commit_group(group_id="g0", acks=_acks(0, 4))
    assert catalog.group_states("g0") == (
        ShadowState.COMMITTED,
        ShadowState.COMMITTED,
    )


def test_group_commit_allows_rank_local_generations_and_byte_digests() -> None:
    catalog = ShadowCatalog(issuer_epoch=1, required_ranks=RANKS)
    acks = _acks(0, 4)

    assert acks[0].source_block_generation != acks[1].source_block_generation
    assert acks[0].destination_generation != acks[1].destination_generation
    assert acks[0].destination_handle != acks[1].destination_handle
    assert acks[0].shard_digest != acks[1].shard_digest
    catalog.commit_group(group_id="g0", acks=acks)
    assert catalog.group_states("g0") == (
        ShadowState.COMMITTED,
        ShadowState.COMMITTED,
    )


def test_new_issuer_can_pin_old_certificate_but_stale_leader_cannot_commit() -> None:
    catalog = ShadowCatalog(issuer_epoch=1, required_ranks=RANKS)
    catalog.commit_group(group_id="g0", acks=_acks(0, 4))
    catalog.advance_issuer_epoch(2)

    with pytest.raises(ProtocolViolation, match="stale leader"):
        catalog.commit_group(
            group_id="stale-g1",
            acks=_acks(4, 8, creation_epoch=1),
        )

    pin_set = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=2,
        durable_continuation_token_end=4,
    )
    assert pin_set.group_ids == ("g0",)
    assert pin_set.pinned_token_end == 4
    assert catalog.group_states("g0") == (
        ShadowState.RECOVERY_PINNED,
        ShadowState.RECOVERY_PINNED,
    )


def test_partial_freeze_prepare_rolls_back_and_freezes_old_route_commits() -> None:
    catalog = _catalog_with_two_groups()

    pin_set = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
        fail_prepare_ranks=("tp1",),
    )

    assert pin_set.group_ids == ()
    assert pin_set.pinned_token_end == 0
    assert catalog.group_states("g0") == (
        ShadowState.COMMITTED,
        ShadowState.COMMITTED,
    )
    with pytest.raises(ProtocolViolation, match="after freeze"):
        catalog.commit_group(group_id="late-g2", acks=_acks(8, 12))


def test_recovery_pin_prevents_reclaim_until_adopt_or_abort() -> None:
    catalog = _catalog_with_two_groups()
    pin_set = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
    )

    with pytest.raises(ProtocolViolation, match="cannot be reclaimed"):
        catalog.reclaim("g0", issuer_epoch=1)

    route_permit = catalog.adopt(pin_set, issuer_epoch=1)
    assert catalog.is_routable(route_permit)
    assert catalog.group_states("g0") == (
        ShadowState.ADOPTED,
        ShadowState.ADOPTED,
    )
    with pytest.raises(ProtocolViolation, match="primary state"):
        catalog.reclaim("g0", issuer_epoch=1)

    aborted = _catalog_with_two_groups()
    aborted_pin = aborted.freeze_and_pin(
        recovery_execution="recovery-2",
        route_version=4,
        issuer_epoch=1,
        durable_continuation_token_end=8,
    )
    aborted.abort_recovery(aborted_pin, issuer_epoch=1)
    aborted.reclaim("g0", issuer_epoch=1)
    assert aborted.group_states("g0") == (
        ShadowState.REVOKED,
        ShadowState.REVOKED,
    )


def test_local_adoption_prepare_is_not_routable_until_all_rank_commit() -> None:
    catalog = _catalog_with_two_groups()
    pin_set = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
    )
    generations = catalog.snapshot_destination_generations(pin_set)

    prepared = catalog.prepare_adoption(
        pin_set,
        issuer_epoch=1,
        observed_destination_generations=generations,
    )
    assert catalog.group_states("g0") == (
        ShadowState.ADOPT_PREPARED,
        ShadowState.ADOPT_PREPARED,
    )
    assert not catalog.is_routable(None)

    permit = catalog.commit_adoption(prepared, issuer_epoch=1)
    assert catalog.is_routable(permit)
    assert permit.adopted_token_end == 8


def test_empty_failed_pin_set_cannot_mint_peer_route_permit() -> None:
    catalog = _catalog_with_two_groups()
    pin_set = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
        fail_prepare_ranks=("tp0",),
    )

    with pytest.raises(ProtocolViolation, match="empty peer prefix"):
        catalog.prepare_adoption(
            pin_set,
            issuer_epoch=1,
            observed_destination_generations={},
        )
    assert not catalog.is_routable(None)


def test_partial_adoption_prepare_rolls_back_and_remains_pinned() -> None:
    catalog = _catalog_with_two_groups()
    pin_set = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
    )

    with pytest.raises(ProtocolViolation, match="failed during adoption"):
        catalog.prepare_adoption(
            pin_set,
            issuer_epoch=1,
            observed_destination_generations=catalog.snapshot_destination_generations(
                pin_set
            ),
            fail_prepare_ranks=("tp1",),
        )

    assert catalog.group_states("g0") == (
        ShadowState.RECOVERY_PINNED,
        ShadowState.RECOVERY_PINNED,
    )
    assert catalog.group_states("g1") == (
        ShadowState.RECOVERY_PINNED,
        ShadowState.RECOVERY_PINNED,
    )


def test_adoption_rechecks_destination_generation_and_fails_closed() -> None:
    catalog = _catalog_with_two_groups()
    pin_set = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
    )
    generations = catalog.snapshot_destination_generations(pin_set)
    generations[("g1", "tp0")] += 1

    with pytest.raises(ProtocolViolation, match="generation changed"):
        catalog.prepare_adoption(
            pin_set,
            issuer_epoch=1,
            observed_destination_generations=generations,
        )

    assert catalog.group_states("g0") == (
        ShadowState.RECOVERY_PINNED,
        ShadowState.RECOVERY_PINNED,
    )
    assert catalog.group_states("g1") == (
        ShadowState.RECOVERY_PINNED,
        ShadowState.RECOVERY_PINNED,
    )


def test_aborted_adoption_prepare_retains_pin_and_never_mints_route_permit() -> None:
    catalog = _catalog_with_two_groups()
    pin_set = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
    )
    prepared = catalog.prepare_adoption(
        pin_set,
        issuer_epoch=1,
        observed_destination_generations=catalog.snapshot_destination_generations(
            pin_set
        ),
    )

    catalog.abort_adoption(prepared, issuer_epoch=1)
    assert not catalog.is_routable(None)
    assert catalog.group_states("g1") == (
        ShadowState.RECOVERY_PINNED,
        ShadowState.RECOVERY_PINNED,
    )


def test_new_leader_must_rebind_existing_pin_before_finishing_recovery() -> None:
    catalog = _catalog_with_two_groups()
    old_pin = catalog.freeze_and_pin(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
    )
    catalog.advance_issuer_epoch(2)

    with pytest.raises(ProtocolViolation, match="another issuer"):
        catalog.adopt(old_pin, issuer_epoch=2)

    forged_subset = replace(
        old_pin,
        group_ids=("g0",),
        pinned_token_end=4,
    )
    with pytest.raises(ProtocolViolation, match="cannot be reconstructed"):
        catalog.rebind_pin_set(forged_subset, issuer_epoch=2)

    rebound = catalog.rebind_pin_set(old_pin, issuer_epoch=2)
    assert rebound.issuer_epoch == 2
    with pytest.raises(ProtocolViolation, match="another issuer"):
        catalog.adopt(old_pin, issuer_epoch=2)

    catalog.adopt(rebound, issuer_epoch=2)
    assert catalog.group_states("g1") == (
        ShadowState.ADOPTED,
        ShadowState.ADOPTED,
    )


def test_store_components_are_not_visible_without_marker_last_commit() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    components = _store_components(0, 4, group_id="g0", action_id="store-a0")
    for component in components:
        catalog.complete_component(component, issuer_epoch=1)

    assert catalog.visible_group_ids() == ()

    catalog.publish_group(_store_record(components), issuer_epoch=1)
    assert catalog.visible_group_ids() == ("g0",)


def test_store_partial_batch_completion_cannot_publish_group() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    components = _store_components(0, 4, group_id="g0", action_id="store-a0")
    for component in components[:-1]:
        catalog.complete_component(component, issuer_epoch=1)

    with pytest.raises(ProtocolViolation, match="incomplete"):
        catalog.publish_group(_store_record(components), issuer_epoch=1)
    assert catalog.visible_group_ids() == ()


def test_store_commit_rejects_mixed_checksum_or_generation() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    components = _store_components(0, 4, group_id="g0", action_id="store-a0")
    for component in components:
        catalog.complete_component(component, issuer_epoch=1)

    refs = list(_store_record(components).components)
    refs[1] = replace(refs[1], object_checksum="wrong-checksum")
    mismatched = replace(_store_record(components), components=tuple(refs))
    with pytest.raises(ProtocolViolation, match="does not match"):
        catalog.publish_group(mismatched, issuer_epoch=1)


def test_store_marker_with_missing_member_fails_closed_at_recovery() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    components = _store_components(0, 4, group_id="g0", action_id="store-a0")
    for component in components:
        catalog.complete_component(component, issuer_epoch=1)
    catalog.publish_group(_store_record(components), issuer_epoch=1)

    # Model an eviction/corruption race after marker publication but before
    # recovery obtains the group read lease.  The marker does not manufacture
    # missing bytes: the usable prefix truncates to zero.
    catalog.evict_object(components[-1].object_key)
    lease = catalog.acquire_read_prefix(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=4,
        target_compatibility_hash="compat-v1",
    )
    assert lease.group_ids == ()
    assert lease.leased_token_end == 0


def test_store_read_lease_returns_only_longest_complete_continuous_prefix() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    first = _store_components(0, 4, group_id="g0", action_id="store-a0")
    second = _store_components(4, 8, group_id="g1", action_id="store-a1")
    for component in first + second:
        catalog.complete_component(component, issuer_epoch=1)
    catalog.publish_group(_store_record(first), issuer_epoch=1)
    catalog.publish_group(_store_record(second), issuer_epoch=1)
    catalog.evict_object(second[0].object_key)

    lease = catalog.acquire_read_prefix(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
        target_compatibility_hash="compat-v1",
    )
    assert lease.group_ids == ("g0",)
    assert lease.leased_token_end == 4
    assert set(lease.object_keys) == {
        "commit/g0",
        *(component.object_key for component in first),
    }


def test_store_read_lease_respects_durable_and_compatibility_boundaries() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    first = _store_components(0, 4, group_id="g0", action_id="store-a0")
    second = _store_components(4, 8, group_id="g1", action_id="store-a1")
    for component in first + second:
        catalog.complete_component(component, issuer_epoch=1)
    catalog.publish_group(_store_record(first), issuer_epoch=1)
    catalog.publish_group(_store_record(second), issuer_epoch=1)

    boundary_lease = catalog.acquire_read_prefix(
        recovery_execution="recovery-boundary",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=6,
        target_compatibility_hash="compat-v1",
    )
    assert boundary_lease.group_ids == ("g0",)
    assert boundary_lease.leased_token_end == 4

    incompatible_lease = catalog.acquire_read_prefix(
        recovery_execution="recovery-incompatible",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=8,
        target_compatibility_hash="compat-v2",
    )
    assert incompatible_lease.group_ids == ()
    assert incompatible_lease.leased_token_end == 0


def test_store_read_lease_blocks_eviction_until_terminal_release() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    components = _store_components(0, 4, group_id="g0", action_id="store-a0")
    for component in components:
        catalog.complete_component(component, issuer_epoch=1)
    catalog.publish_group(_store_record(components), issuer_epoch=1)
    lease = catalog.acquire_read_prefix(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=4,
        target_compatibility_hash="compat-v1",
    )

    with pytest.raises(ProtocolViolation, match="read-leased"):
        catalog.evict_object(components[0].object_key)
    with pytest.raises(ProtocolViolation, match="read-leased"):
        catalog.evict_object("commit/g0")

    catalog.release_read_lease(lease, issuer_epoch=1)
    catalog.evict_object(components[0].object_key)


def test_new_controller_rebinds_exact_store_read_lease_without_unpinning() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    components = _store_components(0, 4, group_id="g0", action_id="store-a0")
    for component in components:
        catalog.complete_component(component, issuer_epoch=1)
    catalog.publish_group(_store_record(components), issuer_epoch=1)
    old_lease = catalog.acquire_read_prefix(
        recovery_execution="recovery-1",
        route_version=3,
        issuer_epoch=1,
        durable_continuation_token_end=4,
        target_compatibility_hash="compat-v1",
    )
    catalog.advance_issuer_epoch(2)

    with pytest.raises(ProtocolViolation, match="stale issuer"):
        catalog.release_read_lease(old_lease, issuer_epoch=1)
    rebound = catalog.rebind_read_lease(old_lease, issuer_epoch=2)
    assert rebound.issuer_epoch == 2
    assert rebound.group_ids == old_lease.group_ids
    assert rebound.object_keys == old_lease.object_keys
    with pytest.raises(ProtocolViolation, match="read-leased"):
        catalog.evict_object("commit/g0")
    with pytest.raises(ProtocolViolation, match="stale or forged"):
        catalog.release_read_lease(old_lease, issuer_epoch=2)

    catalog.release_read_lease(rebound, issuer_epoch=2)
    catalog.evict_object("commit/g0")


def test_store_late_old_leader_completion_and_marker_are_fenced() -> None:
    catalog = StoreCommitCatalog(issuer_epoch=1)
    components = _store_components(0, 4, group_id="g0", action_id="store-a0")
    catalog.advance_issuer_epoch(2)

    with pytest.raises(ProtocolViolation, match="stale issuer"):
        catalog.complete_component(components[0], issuer_epoch=1)
    with pytest.raises(ProtocolViolation, match="stale issuer"):
        catalog.publish_group(_store_record(components), issuer_epoch=1)


def test_durable_boundary_excludes_group_crossing_into_speculative_output() -> None:
    entries = _output_entries(101, 102, 103)
    plan = build_output_replay_plan(
        durable_input_token_end=8,
        committed_groups=(
            TokenGroup(0, 4),
            TokenGroup(4, 8),
            TokenGroup(8, 12),
        ),
        output_entries=entries,
        last_externally_published_sequence=3,
        continuation_contract=_continuation_contract(entries=entries),
        initial_sampling_position=INITIAL_SAMPLING_POSITION,
        initial_state_digest=INITIAL_STATE_DIGEST,
    )

    assert plan.durable_continuation_token_end == 11
    assert plan.usable_kv_token_end == 8
    assert plan.replay_output_sequences == (1, 2, 3)


def test_replay_plan_reconstructs_input_and_durable_output_gap_once() -> None:
    entries = _output_entries(101, 102, 103)
    plan = build_output_replay_plan(
        durable_input_token_end=6,
        committed_groups=(TokenGroup(0, 4),),
        output_entries=entries,
        last_externally_published_sequence=1,
        continuation_contract=_continuation_contract(entries=entries),
        initial_sampling_position=INITIAL_SAMPLING_POSITION,
        initial_state_digest=INITIAL_STATE_DIGEST,
    )

    assert plan.replay_input_token_range == (4, 6)
    assert plan.replay_output_sequences == (1, 2, 3)
    assert plan.pending_outbox_sequences == (2, 3)
    assert plan.next_output_sequence == 4
    assert plan.next_sampling_position == 13


@pytest.mark.parametrize(
    ("committed_groups", "usable_end", "replay_inputs", "replay_outputs"),
    [
        ((), 0, (0, 4), (1, 2, 3)),
        ((TokenGroup(0, 4),), 4, (4, 4), (1, 2, 3)),
        (
            (TokenGroup(0, 4), TokenGroup(4, 6)),
            6,
            (4, 4),
            (3,),
        ),
        (
            (TokenGroup(0, 4), TokenGroup(4, 7)),
            7,
            (4, 4),
            (),
        ),
    ],
)
def test_replay_boundary_uses_half_open_prompt_plus_accepted_outputs(
    committed_groups: tuple[TokenGroup, ...],
    usable_end: int,
    replay_inputs: tuple[int, int],
    replay_outputs: tuple[int, ...],
) -> None:
    entries = _output_entries(101, 102, 103)
    plan = build_output_replay_plan(
        durable_input_token_end=4,
        committed_groups=committed_groups,
        output_entries=entries,
        last_externally_published_sequence=3,
        continuation_contract=_continuation_contract(entries=entries),
        initial_sampling_position=INITIAL_SAMPLING_POSITION,
        initial_state_digest=INITIAL_STATE_DIGEST,
    )

    assert plan.durable_continuation_token_end == 7
    assert plan.usable_kv_token_end == usable_end
    assert plan.replay_input_token_range == replay_inputs
    assert plan.replay_output_sequences == replay_outputs
    assert (
        (replay_inputs[1] - replay_inputs[0]) + len(replay_outputs)
        == plan.durable_continuation_token_end - usable_end
    )


@pytest.mark.parametrize(
    ("entries", "published", "message"),
    [
        (
            (
                _output_entries(101)[0],
                OutputLogEntry.committed(
                    sequence=3,
                    token_id=103,
                    next_sampling_position=12,
                    previous_entry_digest=_output_entries(101)[0].entry_digest,
                    next_state_digest="state-3-sha256",
                ),
            ),
            1,
            "gap-free",
        ),
        (_output_entries(101), 2, "exceeds durable"),
    ],
)
def test_replay_rejects_invalid_wal_or_outbox_pointer(
    entries: tuple[OutputLogEntry, ...],
    published: int,
    message: str,
) -> None:
    contract_entries = entries if tuple(
        entry.sequence for entry in entries
    ) == tuple(range(1, len(entries) + 1)) else ()
    with pytest.raises(ProtocolViolation, match=message):
        build_output_replay_plan(
            durable_input_token_end=4,
            committed_groups=(TokenGroup(0, 4),),
            output_entries=entries,
            last_externally_published_sequence=published,
            continuation_contract=_continuation_contract(
                entries=contract_entries
            ),
            initial_sampling_position=INITIAL_SAMPLING_POSITION,
            initial_state_digest=INITIAL_STATE_DIGEST,
        )


@pytest.mark.parametrize(
    ("object_update", "message"),
    [
        (
            {"storage_failure_domain": "worker-group:source"},
            "does not survive source failure",
        ),
        ({"resolvable": False}, "not target-resolvable"),
        ({"target_authorized": False}, "not target-resolvable"),
        ({"canonical_bytes_verified": False}, "bytes are unverified"),
    ],
)
def test_replay_requires_failure_surviving_canonical_input_objects(
    object_update: dict[str, object], message: str
) -> None:
    invalid_object = replace(INPUT_OBJECTS[0], **object_update)
    input_objects = (invalid_object,)
    entries = _output_entries(101, input_objects=input_objects)
    contract = _continuation_contract(
        entries=entries, input_objects=input_objects
    )

    with pytest.raises(ProtocolViolation, match=message):
        build_output_replay_plan(
            durable_input_token_end=4,
            committed_groups=(TokenGroup(0, 4),),
            output_entries=entries,
            last_externally_published_sequence=1,
            continuation_contract=contract,
            initial_sampling_position=INITIAL_SAMPLING_POSITION,
            initial_state_digest=INITIAL_STATE_DIGEST,
        )


@pytest.mark.parametrize(
    ("state_update", "message"),
    [
        ({"state_complete": False}, "incomplete or unsupported"),
        ({"target_supported": False}, "incomplete or unsupported"),
        ({"authoritative_hypotheses": 2}, "one authoritative hypothesis"),
        ({"includes_speculative_draft_state": True}, "cannot be authority"),
        ({"serialized_state_verified": False}, "incomplete or unsupported"),
        ({"compatibility_hash": "other"}, "compatibility hash mismatch"),
        ({"prefix_hash": "other"}, "prefix hash mismatch"),
        ({"next_output_sequence": 3}, "durable output sequence"),
        ({"next_sampling_position": 12}, "sampling position"),
        ({"output_log_tail_digest": "other"}, "output log tail"),
        ({"state_digest": "other"}, "final output transaction"),
    ],
)
def test_replay_rejects_incomplete_or_misaligned_continuation_state(
    state_update: dict[str, object], message: str
) -> None:
    entries = _output_entries(101)
    contract = _continuation_contract(entries=entries)
    contract = replace(contract, state=replace(contract.state, **state_update))

    with pytest.raises(ProtocolViolation, match=message):
        build_output_replay_plan(
            durable_input_token_end=4,
            committed_groups=(TokenGroup(0, 4),),
            output_entries=entries,
            last_externally_published_sequence=1,
            continuation_contract=contract,
            initial_sampling_position=INITIAL_SAMPLING_POSITION,
            initial_state_digest=INITIAL_STATE_DIGEST,
        )


def test_replay_plan_carries_the_validated_continuation_authority() -> None:
    contract = _continuation_contract()

    plan = build_output_replay_plan(
        durable_input_token_end=4,
        committed_groups=(TokenGroup(0, 4),),
        output_entries=(),
        last_externally_published_sequence=0,
        continuation_contract=contract,
        initial_sampling_position=INITIAL_SAMPLING_POSITION,
        initial_state_digest=INITIAL_STATE_DIGEST,
    )

    assert plan.next_output_sequence == 1
    assert plan.next_sampling_position == INITIAL_SAMPLING_POSITION
    assert plan.continuation_schema_version == "continuation-v1"
    assert plan.continuation_state_digest == INITIAL_STATE_DIGEST


def _router_journal() -> RouterOutputJournal:
    return RouterOutputJournal(
        current_execution="exec-1",
        current_route_version=1,
        input_manifest_digest=compute_input_manifest_digest(INPUT_OBJECTS),
        initial_state_digest=INITIAL_STATE_DIGEST,
        initial_sampling_position=INITIAL_SAMPLING_POSITION,
    )


def test_router_journal_commits_before_publish_and_preserves_outbox_on_fence() -> None:
    journal = _router_journal()
    with pytest.raises(ProtocolViolation, match="before durable commit"):
        journal.publish_next(1)
    first = journal.commit_output(
        worker_execution="exec-1",
        route_version=1,
        sequence=1,
        token_id=101,
        next_sampling_position=11,
        durable_next_state_payload=b"state-after-1",
    )

    assert journal.last_externally_published_sequence == 0
    journal.publish_next(1)
    journal.fence_worker(new_execution="exec-2", new_route_version=2)

    with pytest.raises(ProtocolViolation, match="stale worker"):
        journal.commit_output(
            worker_execution="exec-1",
            route_version=1,
            sequence=2,
            token_id=102,
            next_sampling_position=12,
            durable_next_state_payload=b"stale-state",
        )
    second = journal.commit_output(
        worker_execution="exec-2",
        route_version=2,
        sequence=2,
        token_id=102,
        next_sampling_position=12,
        durable_next_state_payload=b"state-after-2",
    )

    assert journal.last_externally_published_sequence == 1
    assert second.previous_entry_digest == first.entry_digest
    assert tuple(entry.sequence for entry in journal.entries) == (1, 2)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"sequence": 2}, "sequence is not next"),
        ({"next_sampling_position": 12}, "position is not next"),
        ({"durable_next_state_payload": b""}, "state is not durable"),
    ],
)
def test_router_journal_rejects_torn_or_nonnext_commit_atomically(
    update: dict[str, object], message: str
) -> None:
    journal = _router_journal()
    args = {
        "worker_execution": "exec-1",
        "route_version": 1,
        "sequence": 1,
        "token_id": 101,
        "next_sampling_position": 11,
        "durable_next_state_payload": b"state-after-1",
    }
    args.update(update)

    with pytest.raises(ProtocolViolation, match=message):
        journal.commit_output(**args)

    assert journal.entries == ()
    assert journal.last_externally_published_sequence == 0


def test_replay_rejects_self_valid_but_disconnected_log_entry() -> None:
    entries = _output_entries(101)
    disconnected = OutputLogEntry.committed(
        sequence=2,
        token_id=102,
        next_sampling_position=12,
        previous_entry_digest=entries[0].previous_entry_digest,
        next_state_digest="state-2-sha256",
    )
    broken = (entries[0], disconnected)

    with pytest.raises(ProtocolViolation, match="chain is broken"):
        build_output_replay_plan(
            durable_input_token_end=4,
            committed_groups=(TokenGroup(0, 4),),
            output_entries=broken,
            last_externally_published_sequence=1,
            continuation_contract=_continuation_contract(entries=broken),
            initial_sampling_position=INITIAL_SAMPLING_POSITION,
            initial_state_digest=INITIAL_STATE_DIGEST,
        )


def test_replay_rejects_sampling_position_jump() -> None:
    first = _output_entries(101)[0]
    jumped = OutputLogEntry.committed(
        sequence=2,
        token_id=102,
        next_sampling_position=13,
        previous_entry_digest=first.entry_digest,
        next_state_digest="state-2-sha256",
    )
    entries = (first, jumped)

    with pytest.raises(ProtocolViolation, match="advance by one"):
        build_output_replay_plan(
            durable_input_token_end=4,
            committed_groups=(TokenGroup(0, 4),),
            output_entries=entries,
            last_externally_published_sequence=1,
            continuation_contract=_continuation_contract(entries=entries),
            initial_sampling_position=INITIAL_SAMPLING_POSITION,
            initial_state_digest=INITIAL_STATE_DIGEST,
        )
