"""Reference state machine for AegisKV Store protection leases.

This is a theorem/protocol model, not a replacement Store implementation.
Ordinary Store objects remain best-effort cache/offload state.  A protection
lease is the narrow extra contract that lets Algorithm 2 count an immutable,
complete object at a minimum medium for a bounded horizon.

The model makes five safety rules executable:

* the Store Master, not the Controller, charges the object's actual bytes;
* PENDING placement reserves quota but has zero deterministic recovery value;
* ACTIVE is published only after the requested medium is complete;
* multiple leases for the same immutable *physical replica set* share byte
  charges, with the strongest requested medium determining the aggregate
  reservation; logically equal content in different placements never aliases;
* an acknowledged ACTIVE record survives a Master failover.  The new leader
  fences the old epoch and conservatively restarts the full TTL, which can
  retain bytes too long but cannot shorten the acknowledged interval.

Master timestamps are deliberately absent from the Controller contract.  A
Controller derives a local conservative expiry from its own monotonic send
time and the granted TTL; see ``controller_safe_expiry_ms``.

The production API lets the Master choose/reserve the qualifying placement and
returns its digest.  For compactness, this reference starts after that choice:
``StoreObjectIdentity.placement_set_digest`` is the Master-selected set, and
``actual_bytes`` is the target-medium footprint of its commit record and named
members.  A promotion's transient source read pin/traffic is a separate action
resource; its staged destination is already represented by PENDING quota.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, IntEnum


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class StoreMedium(IntEnum):
    ABSENT = 0
    SSD = 1
    DRAM = 2


class StoreLeaseState(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StoreObjectIdentity:
    object_id: str
    version: int
    commit_digest: str
    placement_set_digest: str

    def __post_init__(self) -> None:
        if (
            not self.object_id
            or not self.commit_digest
            or not self.placement_set_digest
        ):
            raise ValueError(
                "object, commit, and physical placement identity must be nonempty"
            )
        _nonnegative_int("version", self.version)


@dataclass
class StoreObjectState:
    identity: StoreObjectIdentity
    # Sum of the physical replica bytes that this placement contract pins or
    # reserves (commit record plus all required K/V/layout members), not
    # logical payload length.
    actual_bytes: int
    medium: StoreMedium
    survives_failure_domains: frozenset[str]
    complete: bool = True

    def __post_init__(self) -> None:
        _positive_int("actual_bytes", self.actual_bytes)
        self.medium = StoreMedium(self.medium)
        if not self.survives_failure_domains or any(
            not domain for domain in self.survives_failure_domains
        ):
            raise ValueError("placement must declare surviving failure domains")
        if self.complete and self.medium is StoreMedium.ABSENT:
            raise ValueError("a complete object cannot be absent")


@dataclass(frozen=True)
class StoreLeaseQuota:
    survival_bytes: int
    dram_bytes: int

    def __post_init__(self) -> None:
        _nonnegative_int("survival_bytes", self.survival_bytes)
        _nonnegative_int("dram_bytes", self.dram_bytes)


@dataclass
class StoreLeaseRecord:
    action_id: str
    identity: StoreObjectIdentity
    actual_bytes: int
    failure_domain: str
    min_medium: StoreMedium
    initial_ttl_ms: int
    ttl_ms: int
    placement_timeout_ms: int
    pending_deadline_ms: int
    authority_epoch: int
    state: StoreLeaseState
    master_expiry_ms: int | None = None

    def request_key(self) -> tuple[object, ...]:
        return (
            self.identity,
            self.actual_bytes,
            self.failure_domain,
            self.min_medium,
            self.initial_ttl_ms,
            self.placement_timeout_ms,
        )


@dataclass(frozen=True)
class StoreLeaseAck:
    action_id: str
    identity: StoreObjectIdentity
    actual_bytes: int
    failure_domain: str
    min_medium: StoreMedium
    ttl_ms: int
    authority_epoch: int


@dataclass(frozen=True)
class ControllerLeaseView:
    ack: StoreLeaseAck
    safe_expiry_ms: int

    def is_usable_at(self, controller_now_ms: int) -> bool:
        _nonnegative_int("controller_now_ms", controller_now_ms)
        return controller_now_ms < self.safe_expiry_ms

    def covers_horizon(self, horizon_ms: int) -> bool:
        _nonnegative_int("horizon_ms", horizon_ms)
        return horizon_ms < self.safe_expiry_ms


def controller_safe_expiry_ms(
    *,
    send_ms: int,
    receive_ms: int,
    ttl_ms: int,
    clock_guard_ms: int = 0,
) -> int | None:
    """Return a local lower bound on an acknowledged lease's expiry.

    The Master activates after the request is sent and before the ACK is
    received.  Therefore, without comparing clocks, expiry is no earlier than
    ``send + ttl`` on the Controller timeline.  ``clock_guard_ms`` covers a
    separately established local clock-rate/timestamp error.  If the round
    trip has already consumed the safe interval, the ACK has no usable future
    horizon and ``None`` is returned.
    """

    send = _nonnegative_int("send_ms", send_ms)
    receive = _nonnegative_int("receive_ms", receive_ms)
    ttl = _positive_int("ttl_ms", ttl_ms)
    guard = _nonnegative_int("clock_guard_ms", clock_guard_ms)
    if receive < send:
        raise ValueError("receive_ms must not precede send_ms")
    safe_expiry = send + ttl - guard
    if safe_expiry <= receive:
        return None
    return safe_expiry


class StoreTierLeaseMaster:
    """Small fail-closed model of the Store Master lease authority."""

    def __init__(
        self,
        *,
        quota: StoreLeaseQuota,
        authority_epoch: int = 1,
    ) -> None:
        _positive_int("authority_epoch", authority_epoch)
        self.quota = quota
        self.authority_epoch = authority_epoch
        self.objects: dict[StoreObjectIdentity, StoreObjectState] = {}
        self.records: dict[str, StoreLeaseRecord] = {}

    def register_object(self, state: StoreObjectState) -> None:
        if state.identity in self.objects:
            raise ValueError("immutable Store object already registered")
        self.objects[state.identity] = state

    @property
    def used_quota(self) -> StoreLeaseQuota:
        strongest: dict[StoreObjectIdentity, StoreLeaseRecord] = {}
        for record in self.records.values():
            if record.state not in {
                StoreLeaseState.PENDING,
                StoreLeaseState.ACTIVE,
            }:
                continue
            previous = strongest.get(record.identity)
            if previous is None or record.min_medium > previous.min_medium:
                strongest[record.identity] = record

        survival = sum(record.actual_bytes for record in strongest.values())
        dram = sum(
            record.actual_bytes
            for record in strongest.values()
            if record.min_medium >= StoreMedium.DRAM
        )
        return StoreLeaseQuota(survival_bytes=survival, dram_bytes=dram)

    def _check_epoch(self, authority_epoch: int) -> None:
        if authority_epoch != self.authority_epoch:
            raise PermissionError("stale Store Master authority epoch")

    def _check_quota(self) -> None:
        used = self.used_quota
        if used.survival_bytes > self.quota.survival_bytes:
            raise MemoryError("Store protection survival quota exceeded")
        if used.dram_bytes > self.quota.dram_bytes:
            raise MemoryError("Store protection DRAM quota exceeded")

    def acquire(
        self,
        *,
        action_id: str,
        identity: StoreObjectIdentity,
        expected_actual_bytes: int,
        failure_domain: str,
        min_medium: StoreMedium,
        ttl_ms: int,
        placement_timeout_ms: int,
        now_ms: int,
        authority_epoch: int,
    ) -> StoreLeaseRecord:
        self._check_epoch(authority_epoch)
        if not action_id:
            raise ValueError("action_id must be nonempty")
        expected = _positive_int("expected_actual_bytes", expected_actual_bytes)
        if not failure_domain:
            raise ValueError("failure_domain must be nonempty")
        ttl = _positive_int("ttl_ms", ttl_ms)
        placement_timeout = _positive_int(
            "placement_timeout_ms", placement_timeout_ms
        )
        now = _nonnegative_int("now_ms", now_ms)
        required_medium = StoreMedium(min_medium)
        if required_medium is StoreMedium.ABSENT:
            raise ValueError("a protection lease requires SSD or DRAM")

        existing = self.records.get(action_id)
        requested_key = (
            identity,
            expected,
            failure_domain,
            required_medium,
            ttl,
            placement_timeout,
        )
        if existing is not None:
            if existing.request_key() != requested_key:
                raise ValueError("action_id replay changed immutable request")
            return existing

        obj = self.objects.get(identity)
        if obj is None or not obj.complete:
            raise ValueError("lease requires a complete immutable Store object")
        if failure_domain not in obj.survives_failure_domains:
            raise ValueError("physical placement does not survive failure domain")
        if obj.actual_bytes != expected:
            raise ValueError("Controller byte estimate does not match Store bytes")

        initial_state = (
            StoreLeaseState.ACTIVE
            if obj.medium >= required_medium
            else StoreLeaseState.PENDING
        )
        record = StoreLeaseRecord(
            action_id=action_id,
            identity=identity,
            actual_bytes=obj.actual_bytes,
            failure_domain=failure_domain,
            min_medium=required_medium,
            initial_ttl_ms=ttl,
            ttl_ms=ttl,
            placement_timeout_ms=placement_timeout,
            pending_deadline_ms=now + placement_timeout,
            authority_epoch=self.authority_epoch,
            state=initial_state,
            master_expiry_ms=now + ttl
            if initial_state is StoreLeaseState.ACTIVE
            else None,
        )
        self.records[action_id] = record
        try:
            self._check_quota()
        except Exception:
            del self.records[action_id]
            raise
        return record

    def commit_placement(
        self,
        *,
        action_id: str,
        completed_medium: StoreMedium,
        now_ms: int,
        authority_epoch: int,
    ) -> StoreLeaseAck:
        self._check_epoch(authority_epoch)
        now = _nonnegative_int("now_ms", now_ms)
        record = self.records[action_id]
        if record.state is not StoreLeaseState.PENDING:
            raise ValueError("only a pending lease can commit placement")
        if now >= record.pending_deadline_ms:
            record.state = StoreLeaseState.FAILED
            self._check_quota()
            raise TimeoutError("placement completed after its terminal deadline")

        medium = StoreMedium(completed_medium)
        obj = self.objects[record.identity]
        if not obj.complete or medium < record.min_medium:
            record.state = StoreLeaseState.FAILED
            self._check_quota()
            raise ValueError("placement did not complete at the requested medium")

        obj.medium = max(obj.medium, medium)
        record.state = StoreLeaseState.ACTIVE
        record.master_expiry_ms = now + record.ttl_ms
        record.authority_epoch = self.authority_epoch
        return self.ack(action_id=action_id, now_ms=now)

    def ack(self, *, action_id: str, now_ms: int) -> StoreLeaseAck:
        now = _nonnegative_int("now_ms", now_ms)
        self.expire_due(now_ms=now)
        record = self.records[action_id]
        if record.state is not StoreLeaseState.ACTIVE:
            raise ValueError("only an active lease has a protection ACK")
        obj = self.objects[record.identity]
        if not obj.complete or obj.medium < record.min_medium:
            raise RuntimeError("active lease lost its promised placement")
        return StoreLeaseAck(
            action_id=record.action_id,
            identity=record.identity,
            actual_bytes=record.actual_bytes,
            failure_domain=record.failure_domain,
            min_medium=record.min_medium,
            ttl_ms=record.ttl_ms,
            authority_epoch=record.authority_epoch,
        )

    def renew(
        self,
        *,
        action_id: str,
        ttl_ms: int,
        now_ms: int,
        authority_epoch: int,
    ) -> StoreLeaseAck:
        self._check_epoch(authority_epoch)
        now = _nonnegative_int("now_ms", now_ms)
        ttl = _positive_int("ttl_ms", ttl_ms)
        self.expire_due(now_ms=now)
        record = self.records[action_id]
        if record.state is not StoreLeaseState.ACTIVE:
            raise ValueError("only an active lease can be renewed")
        record.ttl_ms = ttl
        record.master_expiry_ms = max(
            int(record.master_expiry_ms), now + ttl
        )
        record.authority_epoch = self.authority_epoch
        return self.ack(action_id=action_id, now_ms=now)

    def release(
        self,
        *,
        action_id: str,
        authority_epoch: int,
    ) -> None:
        self._check_epoch(authority_epoch)
        record = self.records[action_id]
        if record.state in {StoreLeaseState.PENDING, StoreLeaseState.ACTIVE}:
            record.state = StoreLeaseState.RELEASED

    def expire_due(self, *, now_ms: int) -> None:
        now = _nonnegative_int("now_ms", now_ms)
        for record in self.records.values():
            if (
                record.state is StoreLeaseState.PENDING
                and now >= record.pending_deadline_ms
            ):
                record.state = StoreLeaseState.FAILED
            elif (
                record.state is StoreLeaseState.ACTIVE
                and record.master_expiry_ms is not None
                and now >= record.master_expiry_ms
            ):
                record.state = StoreLeaseState.EXPIRED

    def deterministic_protection(
        self,
        *,
        action_id: str,
        now_ms: int,
    ) -> bool:
        try:
            self.ack(action_id=action_id, now_ms=now_ms)
        except (KeyError, RuntimeError, ValueError):
            return False
        return True

    def set_object_medium(
        self,
        *,
        identity: StoreObjectIdentity,
        new_medium: StoreMedium,
        now_ms: int,
        authority_epoch: int,
    ) -> None:
        """Apply an eviction/demotion only if ACTIVE leases permit it."""

        self._check_epoch(authority_epoch)
        self.expire_due(now_ms=now_ms)
        medium = StoreMedium(new_medium)
        active_min = StoreMedium.ABSENT
        for record in self.records.values():
            if (
                record.identity == identity
                and record.state is StoreLeaseState.ACTIVE
            ):
                active_min = max(active_min, record.min_medium)
        if medium < active_min:
            raise PermissionError("active tier lease blocks eviction or demotion")

        obj = self.objects[identity]
        obj.medium = medium
        obj.complete = medium is not StoreMedium.ABSENT
        if medium is StoreMedium.ABSENT:
            for record in self.records.values():
                if (
                    record.identity == identity
                    and record.state is StoreLeaseState.PENDING
                ):
                    record.state = StoreLeaseState.FAILED

    def failover(self, *, new_authority_epoch: int, now_ms: int) -> None:
        """Fence the old leader and conservatively preserve durable ACTIVE leases.

        PENDING placement has never produced a protection ACK and may fail
        closed.  Each ACTIVE record is rebound to the new epoch and receives a
        full TTL from takeover.  A production implementation must durably
        replay the same records/quota and may use a stronger cluster-time
        scheme instead; silently dropping them is not conforming behavior.
        """

        new_epoch = _positive_int("new_authority_epoch", new_authority_epoch)
        now = _nonnegative_int("now_ms", now_ms)
        if new_epoch <= self.authority_epoch:
            raise ValueError("new authority epoch must increase")

        for record in self.records.values():
            if record.state is StoreLeaseState.PENDING:
                record.state = StoreLeaseState.FAILED
            elif record.state is StoreLeaseState.ACTIVE:
                obj = self.objects[record.identity]
                if not obj.complete or obj.medium < record.min_medium:
                    raise RuntimeError("cannot restore an acknowledged tier lease")
                record.master_expiry_ms = now + record.ttl_ms
                record.authority_epoch = new_epoch
        self.authority_epoch = new_epoch
        self._check_quota()

    def snapshot_records(self) -> tuple[StoreLeaseRecord, ...]:
        """Return detached records representing the state that HA must persist."""

        return tuple(replace(record) for record in self.records.values())
