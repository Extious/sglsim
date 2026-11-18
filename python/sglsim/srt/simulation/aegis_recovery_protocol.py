"""Executable semantics for AegisKV recovery-state handoff.

This module is intentionally independent from the online SGLang-parity
runtime.  It checks the protocol contract that sits between a completed
protection copy and physical recovery:

* a copy is useful to the healthy-path policy only after all-rank group commit;
* Store objects use a marker-last commit record because the underlying
  multi-object put completes each key independently; group routing/leases do
  not by themselves make a partial K/V or multi-rank write atomically visible;
* a failure plan uses a two-phase all-rank ``freeze and pin`` handoff, so
  commit/reclaim races cannot create use-after-free;
* peer pages remain private through local adoption prepare; only an all-rank
  adoption commit creates a route permit, so locally written request mappings
  cannot make a partially adopted request schedulable;
* a controller creation epoch names an immutable commit certificate, whereas
  the current issuer epoch fences mutations after leader failover; and
* Router output records atomically bind each accepted token to its next
  continuation-state digest in an input-rooted hash chain before publication;
  recovery never consumes KV past that durable boundary and replays the
  durable token-log gap exactly once; and
* replay authority requires failure-surviving canonical inputs plus a complete,
  versioned single-sequence continuation state.  A digest, source-local
  reference, incompatible processor state, or partial beam state fails closed.

Passing these semantics is not evidence that tensor bytes moved, allocator
pages were adopted, or a hard-killed worker resumed.  Those remain physical
implementation obligations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class ProtocolViolation(ValueError):
    """A transition would violate the recovery protocol."""


def _stable_digest(label: str, *parts: object) -> str:
    """Length-prefix fields so protocol digests cannot be delimiter-ambiguous."""

    digest = hashlib.sha256()
    for part in (label, *parts):
        encoded = part if isinstance(part, bytes) else str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class ShadowState(str, Enum):
    COMMITTED = "committed"
    PIN_PREPARED = "pin_prepared"
    RECOVERY_PINNED = "recovery_pinned"
    ADOPT_PREPARED = "adopt_prepared"
    ADOPTED = "adopted"
    REVOKED = "revoked"


@dataclass(frozen=True)
class ShardCommitAck:
    rank_id: str
    token_start: int
    token_end: int
    # Shared logical token-prefix identity.  Actual TP/PP shard bytes differ.
    content_digest: str
    # Rank-local checksum of the materialized KV bytes.
    shard_digest: str
    source_execution: str
    # The source generation is covered by the read pin until this ACK.  The
    # destination handle/generation names the private target allocation that
    # must still be owned when adoption is prepared.  All three fields are
    # rank-local certificates and are never compared across ranks.
    source_block_generation: int
    destination_handle: str
    destination_generation: int
    action_id: str
    creation_epoch: int
    checksum_ok: bool = True
    authorized: bool = True
    isolated_placement: bool = True

    def __post_init__(self) -> None:
        if not self.rank_id or not self.content_digest or not self.shard_digest:
            raise ProtocolViolation("rank and digests must be nonempty")
        if (
            not self.source_execution
            or not self.destination_handle
            or not self.action_id
        ):
            raise ProtocolViolation("lifecycle identifiers must be nonempty")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ProtocolViolation("token range must be positive and ordered")
        if (
            self.source_block_generation < 0
            or self.destination_generation < 0
            or self.creation_epoch < 0
        ):
            raise ProtocolViolation("generation and epoch must be nonnegative")

    @property
    def group_identity(self) -> tuple[object, ...]:
        return (
            self.token_start,
            self.token_end,
            self.content_digest,
            self.source_execution,
            self.action_id,
            self.creation_epoch,
        )


@dataclass
class _RankShadow:
    ack: ShardCommitAck
    state: ShadowState = ShadowState.COMMITTED
    recovery_execution: str | None = None
    recovery_route_version: int | None = None
    pin_issuer_epoch: int | None = None


@dataclass
class _CommittedGroup:
    group_id: str
    required_ranks: tuple[str, ...]
    ranks: dict[str, _RankShadow]

    @property
    def token_start(self) -> int:
        return next(iter(self.ranks.values())).ack.token_start

    @property
    def token_end(self) -> int:
        return next(iter(self.ranks.values())).ack.token_end

    @property
    def creation_epoch(self) -> int:
        return next(iter(self.ranks.values())).ack.creation_epoch


@dataclass(frozen=True)
class RecoveryPinSet:
    recovery_execution: str
    route_version: int
    issuer_epoch: int
    group_ids: tuple[str, ...]
    pinned_token_end: int


@dataclass(frozen=True)
class RecoveryAdoptionPrepare:
    """All-rank local mappings exist, but the request is still quarantined."""

    recovery_execution: str
    route_version: int
    issuer_epoch: int
    group_ids: tuple[str, ...]
    prepared_token_end: int


@dataclass(frozen=True)
class RecoveryRoutePermit:
    """Capability returned only after all-rank adoption commit."""

    recovery_execution: str
    route_version: int
    issuer_epoch: int
    group_ids: tuple[str, ...]
    adopted_token_end: int


class ShadowCatalog:
    """Small target-local model of group commit, failover, and recovery pins."""

    def __init__(self, *, issuer_epoch: int, required_ranks: Iterable[str]):
        ranks = tuple(sorted(set(required_ranks)))
        if issuer_epoch < 0:
            raise ProtocolViolation("issuer epoch must be nonnegative")
        if not ranks:
            raise ProtocolViolation("required ranks must be nonempty")
        self._issuer_epoch = issuer_epoch
        self._required_ranks = ranks
        self._groups: dict[str, _CommittedGroup] = {}
        self._protection_frozen = False
        self._prepared_adoption: RecoveryAdoptionPrepare | None = None
        self._route_permit: RecoveryRoutePermit | None = None

    @property
    def issuer_epoch(self) -> int:
        return self._issuer_epoch

    def advance_issuer_epoch(self, new_epoch: int) -> None:
        if new_epoch <= self._issuer_epoch:
            raise ProtocolViolation("issuer epoch must increase")
        self._issuer_epoch = new_epoch

    def commit_group(
        self,
        *,
        group_id: str,
        acks: Iterable[ShardCommitAck],
    ) -> None:
        if self._protection_frozen:
            raise ProtocolViolation("old-route commit arrived after freeze")
        if not group_id or group_id in self._groups:
            raise ProtocolViolation("group ID must be nonempty and unique")
        ack_list = tuple(acks)
        by_rank = {ack.rank_id: ack for ack in ack_list}
        if len(by_rank) != len(ack_list):
            raise ProtocolViolation("duplicate rank ACK")
        if tuple(sorted(by_rank)) != self._required_ranks:
            raise ProtocolViolation("all required ranks must ACK")
        identities = {ack.group_identity for ack in ack_list}
        if len(identities) != 1:
            raise ProtocolViolation("rank ACK identities disagree")
        if any(
            not ack.checksum_ok
            or not ack.authorized
            or not ack.isolated_placement
            for ack in ack_list
        ):
            raise ProtocolViolation("commit validation failed")
        if ack_list[0].creation_epoch != self._issuer_epoch:
            raise ProtocolViolation("stale leader cannot commit a new group")

        previous_end = max(
            (
                group.token_end
                for group in self._groups.values()
                if all(
                    rank.state
                    not in {ShadowState.REVOKED, ShadowState.ADOPTED}
                    for rank in group.ranks.values()
                )
            ),
            default=0,
        )
        if ack_list[0].token_start != previous_end:
            raise ProtocolViolation("group would create a prefix hole")

        self._groups[group_id] = _CommittedGroup(
            group_id=group_id,
            required_ranks=self._required_ranks,
            ranks={
                rank_id: _RankShadow(ack=by_rank[rank_id])
                for rank_id in self._required_ranks
            },
        )

    def freeze_and_pin(
        self,
        *,
        recovery_execution: str,
        route_version: int,
        issuer_epoch: int,
        durable_continuation_token_end: int,
        fail_prepare_ranks: Iterable[str] = (),
    ) -> RecoveryPinSet:
        self._require_current_issuer(issuer_epoch)
        if not recovery_execution or route_version < 0:
            raise ProtocolViolation("recovery lifecycle must be valid")
        if durable_continuation_token_end < 0:
            raise ProtocolViolation("durable boundary must be nonnegative")

        self._protection_frozen = True
        failed_ranks = set(fail_prepare_ranks)
        pinned_groups: list[str] = []
        pinned_end = 0

        for group in sorted(self._groups.values(), key=lambda item: item.token_start):
            if group.token_start != pinned_end:
                break
            if group.token_end > durable_continuation_token_end:
                break
            if any(
                rank.state is not ShadowState.COMMITTED
                for rank in group.ranks.values()
            ):
                break

            prepared: list[_RankShadow] = []
            prepare_failed = False
            for rank_id in group.required_ranks:
                rank = group.ranks[rank_id]
                if rank_id in failed_ranks:
                    prepare_failed = True
                    break
                rank.state = ShadowState.PIN_PREPARED
                rank.recovery_execution = recovery_execution
                rank.recovery_route_version = route_version
                rank.pin_issuer_epoch = issuer_epoch
                prepared.append(rank)

            if prepare_failed:
                for rank in prepared:
                    self._clear_pin(rank, state=ShadowState.COMMITTED)
                break

            for rank in prepared:
                rank.state = ShadowState.RECOVERY_PINNED
            pinned_groups.append(group.group_id)
            pinned_end = group.token_end

        return RecoveryPinSet(
            recovery_execution=recovery_execution,
            route_version=route_version,
            issuer_epoch=issuer_epoch,
            group_ids=tuple(pinned_groups),
            pinned_token_end=pinned_end,
        )

    def prepare_adoption(
        self,
        pin_set: RecoveryPinSet,
        *,
        issuer_epoch: int,
        observed_destination_generations: dict[tuple[str, str], int],
        fail_prepare_ranks: Iterable[str] = (),
    ) -> RecoveryAdoptionPrepare:
        """Prepare local request mappings without making the request routable.

        A real Worker may either bind the private shadow pages directly or copy
        them into freshly allocated request pages.  In both cases the local
        mapping is quarantined until ``commit_adoption``.  Destination
        generations are rechecked because a transfer ACK alone cannot prove
        that the target allocation has not since been recycled.
        """

        self._require_pin_set(pin_set, issuer_epoch=issuer_epoch)
        if not pin_set.group_ids:
            raise ProtocolViolation("cannot adopt an empty peer prefix")
        if self._prepared_adoption is not None or self._route_permit is not None:
            raise ProtocolViolation("adoption lifecycle already exists")

        expected_keys = {
            (group_id, rank_id)
            for group_id in pin_set.group_ids
            for rank_id in self._required_ranks
        }
        if set(observed_destination_generations) != expected_keys:
            raise ProtocolViolation("adoption must validate every destination handle")

        failed_ranks = set(fail_prepare_ranks)
        prepared_ranks: list[_RankShadow] = []
        try:
            for group_id in pin_set.group_ids:
                group = self._groups[group_id]
                for rank_id in group.required_ranks:
                    rank = group.ranks[rank_id]
                    if rank_id in failed_ranks:
                        raise ProtocolViolation("rank failed during adoption prepare")
                    observed = observed_destination_generations[(group_id, rank_id)]
                    if observed != rank.ack.destination_generation:
                        raise ProtocolViolation("destination generation changed")
                    rank.state = ShadowState.ADOPT_PREPARED
                    prepared_ranks.append(rank)
        except ProtocolViolation:
            for rank in prepared_ranks:
                rank.state = ShadowState.RECOVERY_PINNED
            raise

        prepared = RecoveryAdoptionPrepare(
            recovery_execution=pin_set.recovery_execution,
            route_version=pin_set.route_version,
            issuer_epoch=issuer_epoch,
            group_ids=pin_set.group_ids,
            prepared_token_end=pin_set.pinned_token_end,
        )
        self._prepared_adoption = prepared
        return prepared

    def commit_adoption(
        self,
        prepared: RecoveryAdoptionPrepare,
        *,
        issuer_epoch: int,
    ) -> RecoveryRoutePermit:
        """Atomically create routing authority after every rank prepared."""

        self._require_prepared_adoption(prepared, issuer_epoch=issuer_epoch)
        permit = RecoveryRoutePermit(
            recovery_execution=prepared.recovery_execution,
            route_version=prepared.route_version,
            issuer_epoch=issuer_epoch,
            group_ids=prepared.group_ids,
            adopted_token_end=prepared.prepared_token_end,
        )
        for group_id in prepared.group_ids:
            for rank in self._groups[group_id].ranks.values():
                self._clear_pin(rank, state=ShadowState.ADOPTED)
        self._prepared_adoption = None
        self._route_permit = permit
        return permit

    def abort_adoption(
        self,
        prepared: RecoveryAdoptionPrepare,
        *,
        issuer_epoch: int,
    ) -> None:
        """Roll local mappings back while retaining the recovery pin."""

        self._require_prepared_adoption(prepared, issuer_epoch=issuer_epoch)
        for group_id in prepared.group_ids:
            for rank in self._groups[group_id].ranks.values():
                rank.state = ShadowState.RECOVERY_PINNED
        self._prepared_adoption = None

    def is_routable(self, permit: RecoveryRoutePermit | None) -> bool:
        """Return whether an exact, committed route capability is live."""

        if permit is None or permit != self._route_permit:
            return False
        return all(
            rank.state is ShadowState.ADOPTED
            for group_id in permit.group_ids
            for rank in self._groups[group_id].ranks.values()
        )

    def snapshot_destination_generations(
        self, pin_set: RecoveryPinSet
    ) -> dict[tuple[str, str], int]:
        """Test/reference helper for an unchanged physical allocator snapshot."""

        return {
            (group_id, rank_id): self._groups[group_id]
            .ranks[rank_id]
            .ack.destination_generation
            for group_id in pin_set.group_ids
            for rank_id in self._required_ranks
        }

    def adopt(
        self, pin_set: RecoveryPinSet, *, issuer_epoch: int
    ) -> RecoveryRoutePermit:
        """Single-process shorthand for prepare plus all-rank commit.

        Runtime adapters must expose the two phases separately.  This helper
        keeps older semantic callers concise while preserving the route-permit
        linearization point.
        """

        prepared = self.prepare_adoption(
            pin_set,
            issuer_epoch=issuer_epoch,
            observed_destination_generations=self.snapshot_destination_generations(
                pin_set
            ),
        )
        return self.commit_adoption(prepared, issuer_epoch=issuer_epoch)
        for group_id in pin_set.group_ids:
            for rank in self._groups[group_id].ranks.values():
                self._clear_pin(rank, state=ShadowState.ADOPTED)

    def rebind_pin_set(
        self, pin_set: RecoveryPinSet, *, issuer_epoch: int
    ) -> RecoveryPinSet:
        """Transfer an existing recovery pin to a newly elected issuer."""

        self._require_current_issuer(issuer_epoch)
        if pin_set.issuer_epoch >= issuer_epoch:
            raise ProtocolViolation("pin rebind requires a newer issuer epoch")
        self._validate_pin_set_contents(
            pin_set,
            expected_pin_issuer_epoch=pin_set.issuer_epoch,
            error="old pin set cannot be reconstructed",
        )
        for group_id in pin_set.group_ids:
            for rank in self._groups[group_id].ranks.values():
                rank.pin_issuer_epoch = issuer_epoch
        return RecoveryPinSet(
            recovery_execution=pin_set.recovery_execution,
            route_version=pin_set.route_version,
            issuer_epoch=issuer_epoch,
            group_ids=pin_set.group_ids,
            pinned_token_end=pin_set.pinned_token_end,
        )

    def abort_recovery(
        self, pin_set: RecoveryPinSet, *, issuer_epoch: int
    ) -> None:
        self._require_pin_set(pin_set, issuer_epoch=issuer_epoch)
        for group_id in pin_set.group_ids:
            for rank in self._groups[group_id].ranks.values():
                self._clear_pin(rank, state=ShadowState.COMMITTED)

    def reclaim(self, group_id: str, *, issuer_epoch: int) -> None:
        self._require_current_issuer(issuer_epoch)
        group = self._groups[group_id]
        if any(
            rank.state
            in {
                ShadowState.PIN_PREPARED,
                ShadowState.RECOVERY_PINNED,
                ShadowState.ADOPT_PREPARED,
            }
            for rank in group.ranks.values()
        ):
            raise ProtocolViolation("recovery-pinned group cannot be reclaimed")
        if any(rank.state is ShadowState.ADOPTED for rank in group.ranks.values()):
            raise ProtocolViolation("adopted pages are primary state")
        for rank in group.ranks.values():
            rank.state = ShadowState.REVOKED

    def group_states(self, group_id: str) -> tuple[ShadowState, ...]:
        group = self._groups[group_id]
        return tuple(group.ranks[rank_id].state for rank_id in group.required_ranks)

    def _require_current_issuer(self, issuer_epoch: int) -> None:
        if issuer_epoch != self._issuer_epoch:
            raise ProtocolViolation("stale issuer epoch")

    def _require_pin_set(
        self, pin_set: RecoveryPinSet, *, issuer_epoch: int
    ) -> None:
        self._require_current_issuer(issuer_epoch)
        if pin_set.issuer_epoch != issuer_epoch:
            raise ProtocolViolation("pin set belongs to another issuer epoch")
        self._validate_pin_set_contents(
            pin_set,
            expected_pin_issuer_epoch=issuer_epoch,
            error="pin set no longer owns all ranks",
        )

    def _require_prepared_adoption(
        self,
        prepared: RecoveryAdoptionPrepare,
        *,
        issuer_epoch: int,
    ) -> None:
        self._require_current_issuer(issuer_epoch)
        if prepared != self._prepared_adoption or prepared.issuer_epoch != issuer_epoch:
            raise ProtocolViolation("adoption prepare is stale or forged")
        try:
            groups = [self._groups[group_id] for group_id in prepared.group_ids]
        except KeyError as exc:
            raise ProtocolViolation("adoption prepare lost a group") from exc
        expected_start = 0
        for group in groups:
            if group.token_start != expected_start:
                raise ProtocolViolation("adoption prepare is not a continuous prefix")
            if any(
                rank.state is not ShadowState.ADOPT_PREPARED
                or rank.recovery_execution != prepared.recovery_execution
                or rank.recovery_route_version != prepared.route_version
                or rank.pin_issuer_epoch != issuer_epoch
                for rank in group.ranks.values()
            ):
                raise ProtocolViolation("not every rank owns the adoption prepare")
            expected_start = group.token_end
        if prepared.prepared_token_end != expected_start:
            raise ProtocolViolation("adoption prepare boundary is inconsistent")

    def _validate_pin_set_contents(
        self,
        pin_set: RecoveryPinSet,
        *,
        expected_pin_issuer_epoch: int,
        error: str,
    ) -> None:
        if len(set(pin_set.group_ids)) != len(pin_set.group_ids):
            raise ProtocolViolation(error)
        try:
            groups = [self._groups[group_id] for group_id in pin_set.group_ids]
        except KeyError as exc:
            raise ProtocolViolation(error) from exc
        expected_start = 0
        for group in groups:
            if group.token_start != expected_start:
                raise ProtocolViolation(error)
            if any(
                rank.state is not ShadowState.RECOVERY_PINNED
                or rank.recovery_execution != pin_set.recovery_execution
                or rank.recovery_route_version != pin_set.route_version
                or rank.pin_issuer_epoch != expected_pin_issuer_epoch
                for rank in group.ranks.values()
            ):
                raise ProtocolViolation(error)
            expected_start = group.token_end
        if pin_set.pinned_token_end != expected_start:
            raise ProtocolViolation(error)

        owned_groups = tuple(
            group.group_id
            for group in sorted(
                self._groups.values(), key=lambda item: item.token_start
            )
            if all(
                rank.state is ShadowState.RECOVERY_PINNED
                and rank.recovery_execution == pin_set.recovery_execution
                and rank.recovery_route_version == pin_set.route_version
                and rank.pin_issuer_epoch == expected_pin_issuer_epoch
                for rank in group.ranks.values()
            )
        )
        if owned_groups != pin_set.group_ids:
            raise ProtocolViolation(error)

    @staticmethod
    def _clear_pin(rank: _RankShadow, *, state: ShadowState) -> None:
        rank.state = state
        rank.recovery_execution = None
        rank.recovery_route_version = None
        rank.pin_issuer_epoch = None


@dataclass(frozen=True)
class StoreComponentAck:
    """One immutable Store object completed by the existing object protocol."""

    group_id: str
    component_role: str
    object_key: str
    object_checksum: str
    token_start: int
    token_end: int
    compatibility_hash: str
    content_digest: str
    source_execution: str
    source_block_generation: int
    action_id: str
    creation_epoch: int
    complete: bool = True
    authorized: bool = True
    source_failure_isolated: bool = True

    def __post_init__(self) -> None:
        if not all(
            (
                self.group_id,
                self.component_role,
                self.object_key,
                self.object_checksum,
                self.compatibility_hash,
                self.content_digest,
                self.source_execution,
                self.action_id,
            )
        ):
            raise ProtocolViolation("Store component identity must be nonempty")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ProtocolViolation("Store component token range is invalid")
        if self.source_block_generation < 0 or self.creation_epoch < 0:
            raise ProtocolViolation("Store component generation/epoch is invalid")


@dataclass(frozen=True)
class StoreComponentRef:
    """The exact member list authenticated by a Store commit record."""

    component_role: str
    object_key: str
    object_checksum: str
    source_block_generation: int

    def __post_init__(self) -> None:
        if not self.component_role or not self.object_key or not self.object_checksum:
            raise ProtocolViolation("Store component reference must be complete")
        if self.source_block_generation < 0:
            raise ProtocolViolation("Store component generation is invalid")


@dataclass(frozen=True)
class StoreCommitRecord:
    """Small marker written only after every required byte object is complete.

    This is deliberately called a commit record, not a manifest: it is an
    ordinary immutable Store object scoped to one token group.  Readers never
    infer recoverability by enumerating component keys.
    """

    group_id: str
    record_key: str
    record_checksum: str
    token_start: int
    token_end: int
    compatibility_hash: str
    content_digest: str
    source_execution: str
    action_id: str
    creation_epoch: int
    components: tuple[StoreComponentRef, ...]
    complete: bool = True
    authorized: bool = True
    source_failure_isolated: bool = True

    def __post_init__(self) -> None:
        if not all(
            (
                self.group_id,
                self.record_key,
                self.record_checksum,
                self.compatibility_hash,
                self.content_digest,
                self.source_execution,
                self.action_id,
            )
        ):
            raise ProtocolViolation("Store commit-record identity must be nonempty")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ProtocolViolation("Store commit-record token range is invalid")
        if self.creation_epoch < 0 or not self.components:
            raise ProtocolViolation("Store commit record requires epoch and members")
        roles = [component.component_role for component in self.components]
        keys = [component.object_key for component in self.components]
        if len(set(roles)) != len(roles) or len(set(keys)) != len(keys):
            raise ProtocolViolation("Store commit record has duplicate members")


@dataclass(frozen=True)
class StoreReadLease:
    recovery_execution: str
    route_version: int
    issuer_epoch: int
    group_ids: tuple[str, ...]
    object_keys: tuple[str, ...]
    leased_token_end: int


class StoreCommitCatalog:
    """Reference semantics for marker-last Store commit and group read lease.

    Mooncake already supplies per-object two-phase put, checksums, group routing,
    coalesced group lease refresh, and group eviction.  This model adds only the
    Aegis rule that a recovery-visible group has a final commit record naming
    every required object.  Missing/corrupt members truncate the usable prefix;
    they can never yield a partial group.
    """

    def __init__(self, *, issuer_epoch: int):
        if issuer_epoch < 0:
            raise ProtocolViolation("issuer epoch must be nonnegative")
        self._issuer_epoch = issuer_epoch
        self._components: dict[str, StoreComponentAck] = {}
        self._records: dict[str, StoreCommitRecord] = {}
        self._record_groups_by_key: dict[str, str] = {}
        self._read_leases: dict[
            tuple[str, int, int], StoreReadLease
        ] = {}
        self._leased_key_refcounts: dict[str, int] = {}

    def advance_issuer_epoch(self, new_epoch: int) -> None:
        if new_epoch <= self._issuer_epoch:
            raise ProtocolViolation("issuer epoch must increase")
        self._issuer_epoch = new_epoch

    def complete_component(
        self, ack: StoreComponentAck, *, issuer_epoch: int
    ) -> None:
        self._require_current_issuer(issuer_epoch)
        if ack.creation_epoch != issuer_epoch:
            raise ProtocolViolation("stale leader cannot complete Store component")
        if not ack.complete or not ack.authorized or not ack.source_failure_isolated:
            raise ProtocolViolation("Store component completion validation failed")
        previous = self._components.get(ack.object_key)
        if previous is not None:
            if previous != ack:
                raise ProtocolViolation("immutable Store object key collision")
            return
        self._components[ack.object_key] = ack

    def publish_group(
        self, record: StoreCommitRecord, *, issuer_epoch: int
    ) -> None:
        """Publish the marker after validating all independently completed keys."""

        self._require_current_issuer(issuer_epoch)
        if record.creation_epoch != issuer_epoch:
            raise ProtocolViolation("stale leader cannot publish Store group")
        if (
            not record.complete
            or not record.authorized
            or not record.source_failure_isolated
        ):
            raise ProtocolViolation("Store commit-record validation failed")

        previous = self._records.get(record.group_id)
        if previous is not None:
            if previous != record:
                raise ProtocolViolation("Store group ID is immutable")
            return
        if (
            record.record_key in self._components
            or record.record_key in self._record_groups_by_key
        ):
            raise ProtocolViolation("Store commit-record key collision")

        expected_roles = {component.component_role for component in record.components}
        actual_roles = {
            component.component_role
            for component in self._components.values()
            if component.group_id == record.group_id
        }
        if actual_roles != expected_roles:
            raise ProtocolViolation(
                "Store group is incomplete or has unexpected members"
            )

        for ref in record.components:
            component = self._components.get(ref.object_key)
            if component is None:
                raise ProtocolViolation("Store group member is missing")
            if (
                component.group_id != record.group_id
                or component.component_role != ref.component_role
                or component.object_checksum != ref.object_checksum
                or component.source_block_generation != ref.source_block_generation
                or component.token_start != record.token_start
                or component.token_end != record.token_end
                or component.compatibility_hash != record.compatibility_hash
                or component.content_digest != record.content_digest
                or component.source_execution != record.source_execution
                or component.action_id != record.action_id
                or component.creation_epoch != record.creation_epoch
            ):
                raise ProtocolViolation(
                    "Store group member does not match commit record"
                )

        previous_end = max(
            (existing.token_end for existing in self._records.values()), default=0
        )
        if record.token_start != previous_end:
            raise ProtocolViolation("Store group would create a prefix hole")

        self._records[record.group_id] = record
        self._record_groups_by_key[record.record_key] = record.group_id

    def acquire_read_prefix(
        self,
        *,
        recovery_execution: str,
        route_version: int,
        issuer_epoch: int,
        durable_continuation_token_end: int,
        target_compatibility_hash: str,
    ) -> StoreReadLease:
        """Atomically validate and lease the longest marker-backed prefix."""

        self._require_current_issuer(issuer_epoch)
        if not recovery_execution or route_version < 0:
            raise ProtocolViolation("recovery lifecycle must be valid")
        if durable_continuation_token_end < 0 or not target_compatibility_hash:
            raise ProtocolViolation("Store recovery boundary/compatibility is invalid")
        lease_id = (recovery_execution, route_version, issuer_epoch)
        if lease_id in self._read_leases:
            raise ProtocolViolation("Store read lease already exists")

        leased_groups: list[str] = []
        leased_keys: list[str] = []
        leased_end = 0
        for record in sorted(self._records.values(), key=lambda item: item.token_start):
            if record.token_start != leased_end:
                break
            if record.token_end > durable_continuation_token_end:
                break
            if record.compatibility_hash != target_compatibility_hash:
                break

            group_keys: list[str] = [record.record_key]
            valid = True
            for ref in record.components:
                component = self._components.get(ref.object_key)
                if component is None or (
                    component.group_id != record.group_id
                    or component.component_role != ref.component_role
                    or component.object_checksum != ref.object_checksum
                    or component.source_block_generation
                    != ref.source_block_generation
                ):
                    valid = False
                    break
                group_keys.append(component.object_key)
            if not valid:
                break

            leased_groups.append(record.group_id)
            leased_keys.extend(group_keys)
            leased_end = record.token_end

        lease = StoreReadLease(
            recovery_execution=recovery_execution,
            route_version=route_version,
            issuer_epoch=issuer_epoch,
            group_ids=tuple(leased_groups),
            object_keys=tuple(leased_keys),
            leased_token_end=leased_end,
        )
        for key in lease.object_keys:
            self._leased_key_refcounts[key] = self._leased_key_refcounts.get(key, 0) + 1
        self._read_leases[lease_id] = lease
        return lease

    def rebind_read_lease(
        self, lease: StoreReadLease, *, issuer_epoch: int
    ) -> StoreReadLease:
        """Transfer one exact group lease to the current Controller issuer.

        Rebinding changes only mutation authority.  The atomically acquired
        object-reference set and its refcounts stay unchanged, so neither an
        old leader nor a partial reconstruction can release a subset while a
        recovery load is in flight.
        """

        self._require_current_issuer(issuer_epoch)
        if lease.issuer_epoch >= issuer_epoch:
            raise ProtocolViolation("read-lease rebind requires a newer issuer")
        old_id = (
            lease.recovery_execution,
            lease.route_version,
            lease.issuer_epoch,
        )
        if self._read_leases.get(old_id) != lease:
            raise ProtocolViolation("old Store read lease cannot be reconstructed")
        new_id = (
            lease.recovery_execution,
            lease.route_version,
            issuer_epoch,
        )
        if new_id in self._read_leases:
            raise ProtocolViolation("new issuer already has this Store read lease")
        rebound = StoreReadLease(
            recovery_execution=lease.recovery_execution,
            route_version=lease.route_version,
            issuer_epoch=issuer_epoch,
            group_ids=lease.group_ids,
            object_keys=lease.object_keys,
            leased_token_end=lease.leased_token_end,
        )
        del self._read_leases[old_id]
        self._read_leases[new_id] = rebound
        return rebound

    def release_read_lease(self, lease: StoreReadLease, *, issuer_epoch: int) -> None:
        self._require_current_issuer(issuer_epoch)
        lease_id = (
            lease.recovery_execution,
            lease.route_version,
            lease.issuer_epoch,
        )
        if self._read_leases.get(lease_id) != lease:
            raise ProtocolViolation("Store read lease is stale or forged")
        del self._read_leases[lease_id]
        for key in lease.object_keys:
            count = self._leased_key_refcounts[key] - 1
            if count:
                self._leased_key_refcounts[key] = count
            else:
                del self._leased_key_refcounts[key]

    def evict_object(self, object_key: str) -> None:
        """Model eviction/race; recovery remains safe by truncating the prefix."""

        if self._leased_key_refcounts.get(object_key, 0):
            raise ProtocolViolation("read-leased Store object cannot be evicted")
        if object_key in self._components:
            del self._components[object_key]
            return
        group_id = self._record_groups_by_key.pop(object_key, None)
        if group_id is None:
            raise ProtocolViolation("unknown Store object")
        del self._records[group_id]

    def visible_group_ids(self) -> tuple[str, ...]:
        return tuple(
            record.group_id
            for record in sorted(
                self._records.values(), key=lambda item: item.token_start
            )
        )

    def _require_current_issuer(self, issuer_epoch: int) -> None:
        if issuer_epoch != self._issuer_epoch:
            raise ProtocolViolation("stale issuer epoch")


@dataclass(frozen=True)
class TokenGroup:
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ProtocolViolation("token group must be positive and ordered")


@dataclass(frozen=True)
class OutputLogEntry:
    sequence: int
    token_id: int
    next_sampling_position: int
    previous_entry_digest: str
    next_state_digest: str
    entry_digest: str

    @classmethod
    def committed(
        cls,
        *,
        sequence: int,
        token_id: int,
        next_sampling_position: int,
        previous_entry_digest: str,
        next_state_digest: str,
    ) -> "OutputLogEntry":
        entry_digest = cls.compute_digest(
            sequence=sequence,
            token_id=token_id,
            next_sampling_position=next_sampling_position,
            previous_entry_digest=previous_entry_digest,
            next_state_digest=next_state_digest,
        )
        return cls(
            sequence=sequence,
            token_id=token_id,
            next_sampling_position=next_sampling_position,
            previous_entry_digest=previous_entry_digest,
            next_state_digest=next_state_digest,
            entry_digest=entry_digest,
        )

    @staticmethod
    def compute_digest(
        *,
        sequence: int,
        token_id: int,
        next_sampling_position: int,
        previous_entry_digest: str,
        next_state_digest: str,
    ) -> str:
        return _stable_digest(
            "aegis-router-output-v1",
            sequence,
            token_id,
            next_sampling_position,
            previous_entry_digest,
            next_state_digest,
        )

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
            or isinstance(self.token_id, bool)
            or not isinstance(self.token_id, int)
            or self.token_id < 0
            or isinstance(self.next_sampling_position, bool)
            or not isinstance(self.next_sampling_position, int)
            or self.next_sampling_position < 0
        ):
            raise ProtocolViolation("output sequence/token/position is invalid")
        if not all(
            (
                self.previous_entry_digest,
                self.next_state_digest,
                self.entry_digest,
            )
        ):
            raise ProtocolViolation("output log digest chain is incomplete")
        expected = self.compute_digest(
            sequence=self.sequence,
            token_id=self.token_id,
            next_sampling_position=self.next_sampling_position,
            previous_entry_digest=self.previous_entry_digest,
            next_state_digest=self.next_state_digest,
        )
        if self.entry_digest != expected:
            raise ProtocolViolation("output log entry digest mismatch")


@dataclass(frozen=True)
class DurableInputObject:
    """One canonical request input that survives the declared source failure."""

    object_id: str
    content_digest: str
    schema_version: str
    storage_failure_domain: str
    resolvable: bool = True
    target_authorized: bool = True
    canonical_bytes_verified: bool = True

    def __post_init__(self) -> None:
        if not all(
            (
                self.object_id,
                self.content_digest,
                self.schema_version,
                self.storage_failure_domain,
            )
        ):
            raise ProtocolViolation("canonical input identity must be nonempty")


def compute_input_manifest_digest(
    input_objects: Iterable[DurableInputObject],
) -> str:
    """Bind the ordered canonical-object manifest used to reconstruct input."""

    objects = tuple(input_objects)
    if not objects:
        raise ProtocolViolation("canonical input objects must be durable")
    fields: list[object] = []
    for item in objects:
        fields.extend(
            (
                item.object_id,
                item.content_digest,
                item.schema_version,
                item.storage_failure_domain,
            )
        )
    return _stable_digest("aegis-input-manifest-v1", *fields)


def output_log_genesis_digest(
    *,
    input_manifest_digest: str,
    initial_state_digest: str,
    initial_sampling_position: int,
) -> str:
    if (
        not input_manifest_digest
        or not initial_state_digest
        or isinstance(initial_sampling_position, bool)
        or not isinstance(initial_sampling_position, int)
        or initial_sampling_position < 0
    ):
        raise ProtocolViolation("output log genesis is incomplete")
    return _stable_digest(
        "aegis-router-output-genesis-v1",
        input_manifest_digest,
        initial_state_digest,
        initial_sampling_position,
    )


@dataclass(frozen=True)
class DurableContinuationState:
    """Versioned authority for the first new output after durable replay."""

    schema_version: str
    state_digest: str
    compatibility_hash: str
    prefix_hash: str
    next_output_sequence: int
    next_sampling_position: int
    output_log_tail_digest: str
    sampling_semantics: str = "sglang-position-v1"
    state_complete: bool = True
    target_supported: bool = True
    serialized_state_verified: bool = True
    authoritative_hypotheses: int = 1
    includes_speculative_draft_state: bool = False

    def __post_init__(self) -> None:
        if not all(
            (
                self.schema_version,
                self.state_digest,
                self.compatibility_hash,
                self.prefix_hash,
                self.output_log_tail_digest,
                self.sampling_semantics,
            )
        ):
            raise ProtocolViolation("continuation identity must be nonempty")
        if self.next_output_sequence <= 0 or self.next_sampling_position < 0:
            raise ProtocolViolation("continuation sequence/position is invalid")
        if (
            isinstance(self.authoritative_hypotheses, bool)
            or not isinstance(self.authoritative_hypotheses, int)
            or self.authoritative_hypotheses <= 0
        ):
            raise ProtocolViolation(
                "authoritative hypothesis count must be a positive integer"
            )


@dataclass(frozen=True)
class RecoveryContinuationContract:
    """Inputs and target checks required before continuation can be adopted."""

    source_failure_domain: str
    target_compatibility_hash: str
    target_prefix_hash: str
    input_objects: tuple[DurableInputObject, ...]
    state: DurableContinuationState

    def __post_init__(self) -> None:
        if not all(
            (
                self.source_failure_domain,
                self.target_compatibility_hash,
                self.target_prefix_hash,
            )
        ):
            raise ProtocolViolation("continuation contract identity is incomplete")
        if not self.input_objects:
            raise ProtocolViolation("canonical input objects must be durable")
        object_ids = [item.object_id for item in self.input_objects]
        if len(set(object_ids)) != len(object_ids):
            raise ProtocolViolation("canonical input object IDs must be unique")

    @property
    def input_manifest_digest(self) -> str:
        return compute_input_manifest_digest(self.input_objects)

    def validate_boundary(
        self,
        *,
        durable_output_sequence: int,
        next_sampling_position: int,
        output_log_tail_digest: str,
        final_entry_state_digest: str | None,
    ) -> None:
        for item in self.input_objects:
            if item.storage_failure_domain == self.source_failure_domain:
                raise ProtocolViolation(
                    "canonical input object does not survive source failure"
                )
            if not item.resolvable or not item.target_authorized:
                raise ProtocolViolation(
                    "canonical input object is not target-resolvable/authorized"
                )
            if not item.canonical_bytes_verified:
                raise ProtocolViolation("canonical input bytes are unverified")

        state = self.state
        if (
            not state.state_complete
            or not state.target_supported
            or not state.serialized_state_verified
        ):
            raise ProtocolViolation(
                "continuation state is incomplete or unsupported"
            )
        if state.authoritative_hypotheses != 1:
            raise ProtocolViolation(
                "E1 continuation requires one authoritative hypothesis"
            )
        if state.includes_speculative_draft_state:
            raise ProtocolViolation("speculative draft state cannot be authority")
        if state.compatibility_hash != self.target_compatibility_hash:
            raise ProtocolViolation("target compatibility hash mismatch")
        if state.prefix_hash != self.target_prefix_hash:
            raise ProtocolViolation("target prefix hash mismatch")
        if state.next_output_sequence != durable_output_sequence + 1:
            raise ProtocolViolation(
                "continuation state is not aligned to durable output sequence"
            )
        if state.next_sampling_position != next_sampling_position:
            raise ProtocolViolation(
                "continuation state is not aligned to sampling position"
            )
        if state.output_log_tail_digest != output_log_tail_digest:
            raise ProtocolViolation(
                "continuation state is not bound to output log tail"
            )
        if (
            final_entry_state_digest is not None
            and state.state_digest != final_entry_state_digest
        ):
            raise ProtocolViolation(
                "continuation state does not match final output transaction"
            )


@dataclass(frozen=True)
class OutputReplayPlan:
    durable_continuation_token_end: int
    usable_kv_token_end: int
    replay_input_token_range: tuple[int, int]
    replay_output_sequences: tuple[int, ...]
    pending_outbox_sequences: tuple[int, ...]
    next_output_sequence: int
    next_sampling_position: int
    continuation_schema_version: str
    continuation_state_digest: str


def build_output_replay_plan(
    *,
    durable_input_token_end: int,
    committed_groups: Iterable[TokenGroup],
    output_entries: Iterable[OutputLogEntry],
    last_externally_published_sequence: int,
    continuation_contract: RecoveryContinuationContract,
    initial_sampling_position: int,
    initial_state_digest: str,
) -> OutputReplayPlan:
    """Validate continuation authority and plan one silent replay of the gap.

    Token positions are half open.  If the canonical prompt/input occupies
    ``[0, P)`` and the Router has durably accepted ``n`` output tokens, the
    continuation input boundary is ``B = P + n``: accepted output token ``k``
    is the input at position ``P + k - 1`` for the next decode.  A committed KV
    prefix ending at ``K <= B`` therefore needs prompt replay ``[K, P)`` when
    ``K < P`` and output-sequence replay ``K - P + 1 .. n`` when ``K >= P``.
    No group crossing ``B`` is usable, even if its bytes are committed.
    """

    if (
        durable_input_token_end < 0
        or initial_sampling_position < 0
        or not initial_state_digest
    ):
        raise ProtocolViolation(
            "input boundary and initial continuation state are invalid"
        )
    entries = tuple(sorted(output_entries, key=lambda item: item.sequence))
    expected_sequences = tuple(range(1, len(entries) + 1))
    if tuple(entry.sequence for entry in entries) != expected_sequences:
        raise ProtocolViolation("durable output log must be gap-free")
    log_tail = output_log_genesis_digest(
        input_manifest_digest=continuation_contract.input_manifest_digest,
        initial_state_digest=initial_state_digest,
        initial_sampling_position=initial_sampling_position,
    )
    next_sampling_position = initial_sampling_position
    for entry in entries:
        if entry.previous_entry_digest != log_tail:
            raise ProtocolViolation("durable output log chain is broken")
        if entry.next_sampling_position != next_sampling_position + 1:
            raise ProtocolViolation(
                "durable output sampling positions must advance by one"
            )
        log_tail = entry.entry_digest
        next_sampling_position = entry.next_sampling_position
    durable_output_sequence = len(entries)
    if not 0 <= last_externally_published_sequence <= durable_output_sequence:
        raise ProtocolViolation("outbox pointer exceeds durable output")

    # Half-open canonical-input boundary after all Router-accepted outputs.
    durable_token_end = durable_input_token_end + durable_output_sequence
    usable_kv_end = 0
    for group in sorted(committed_groups, key=lambda item: item.token_start):
        if group.token_start != usable_kv_end:
            break
        if group.token_end > durable_token_end:
            break
        usable_kv_end = group.token_end

    # Prompt/input suffix and accepted-output suffix are disjoint replay
    # sources; together they cover exactly [usable_kv_end, durable_token_end).
    replay_input_start = min(usable_kv_end, durable_input_token_end)
    replay_input_range = (replay_input_start, durable_input_token_end)
    first_output_sequence = max(
        1,
        usable_kv_end - durable_input_token_end + 1,
    )
    replay_outputs = tuple(
        range(first_output_sequence, durable_output_sequence + 1)
    )
    pending_outbox = tuple(
        range(last_externally_published_sequence + 1, durable_output_sequence + 1)
    )
    continuation_contract.validate_boundary(
        durable_output_sequence=durable_output_sequence,
        next_sampling_position=next_sampling_position,
        output_log_tail_digest=log_tail,
        final_entry_state_digest=(
            entries[-1].next_state_digest if entries else initial_state_digest
        ),
    )

    return OutputReplayPlan(
        durable_continuation_token_end=durable_token_end,
        usable_kv_token_end=usable_kv_end,
        replay_input_token_range=replay_input_range,
        replay_output_sequences=replay_outputs,
        pending_outbox_sequences=pending_outbox,
        next_output_sequence=durable_output_sequence + 1,
        next_sampling_position=next_sampling_position,
        continuation_schema_version=(
            continuation_contract.state.schema_version
        ),
        continuation_state_digest=continuation_contract.state.state_digest,
    )


@dataclass
class RouterOutputJournal:
    """Atomic Router-side token/state commit before surviving-stream publish.

    The transition is deliberately scoped to E1: the Router and its stream do
    not fail.  A storage adapter must make one ``commit_output`` transition
    durable as a unit, or discard a torn tail by checksum/chain validation.
    """

    current_execution: str
    current_route_version: int
    input_manifest_digest: str
    initial_state_digest: str
    initial_sampling_position: int
    _entries: list[OutputLogEntry] = field(default_factory=list, init=False)
    _last_published_sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if (
            not self.current_execution
            or isinstance(self.current_route_version, bool)
            or not isinstance(self.current_route_version, int)
            or self.current_route_version < 0
        ):
            raise ProtocolViolation("Router output fence is invalid")
        output_log_genesis_digest(
            input_manifest_digest=self.input_manifest_digest,
            initial_state_digest=self.initial_state_digest,
            initial_sampling_position=self.initial_sampling_position,
        )

    @property
    def entries(self) -> tuple[OutputLogEntry, ...]:
        return tuple(self._entries)

    @property
    def last_externally_published_sequence(self) -> int:
        return self._last_published_sequence

    @property
    def tail_digest(self) -> str:
        if self._entries:
            return self._entries[-1].entry_digest
        return output_log_genesis_digest(
            input_manifest_digest=self.input_manifest_digest,
            initial_state_digest=self.initial_state_digest,
            initial_sampling_position=self.initial_sampling_position,
        )

    @property
    def current_state_digest(self) -> str:
        if self._entries:
            return self._entries[-1].next_state_digest
        return self.initial_state_digest

    @property
    def next_sampling_position(self) -> int:
        if self._entries:
            return self._entries[-1].next_sampling_position
        return self.initial_sampling_position

    def commit_output(
        self,
        *,
        worker_execution: str,
        route_version: int,
        sequence: int,
        token_id: int,
        next_sampling_position: int,
        durable_next_state_payload: bytes,
    ) -> OutputLogEntry:
        if (
            worker_execution != self.current_execution
            or route_version != self.current_route_version
        ):
            raise ProtocolViolation("stale worker output fence")
        if sequence != len(self._entries) + 1:
            raise ProtocolViolation("Router output sequence is not next")
        if next_sampling_position != self.next_sampling_position + 1:
            raise ProtocolViolation("Router sampling position is not next")
        if not durable_next_state_payload:
            raise ProtocolViolation("next continuation state is not durable")
        next_state_digest = _stable_digest(
            "aegis-continuation-state-v1", durable_next_state_payload
        )
        entry = OutputLogEntry.committed(
            sequence=sequence,
            token_id=token_id,
            next_sampling_position=next_sampling_position,
            previous_entry_digest=self.tail_digest,
            next_state_digest=next_state_digest,
        )
        # One reference transition: no publish-visible intermediate state.
        self._entries.append(entry)
        return entry

    def publish_next(self, sequence: int) -> None:
        if sequence != self._last_published_sequence + 1:
            raise ProtocolViolation("outbox publication is not next")
        if sequence > len(self._entries):
            raise ProtocolViolation("cannot publish output before durable commit")
        self._last_published_sequence = sequence

    def fence_worker(
        self, *, new_execution: str, new_route_version: int
    ) -> None:
        if not new_execution or new_execution == self.current_execution:
            raise ProtocolViolation("new execution fence is invalid")
        if new_route_version <= self.current_route_version:
            raise ProtocolViolation("route version must advance")
        self.current_execution = new_execution
        self.current_route_version = new_route_version
