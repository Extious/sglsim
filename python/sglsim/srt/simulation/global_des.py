"""Globally ordered virtual-time primitives for offline SGLSim experiments.

The :class:`GlobalDES` kernel owns one authoritative nanosecond cursor and one
stable ``(timestamp, priority, sequence)`` event heap.  A DP/PP/TP rank may
schedule work independently, but no rank-local clock is ever max-synchronized
into another rank.  Concurrent work therefore completes at the maximum of its
completion times instead of being accidentally serialized.

This module is intentionally policy-free.  It provides the physical resource
accounting, KV-copy state, transfer reservations, and audit trail needed by a
future ``OfflineSchedulerCluster`` adapter.  It has no third-party dependency
and does not import the online scheduler runtime.
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


def _nonnegative_int(name: str, value: int) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


class EventPriority(IntEnum):
    """Default deterministic ordering for events at the same timestamp."""

    FAILURE = 0
    TRANSFER_START = 1
    TRANSFER_COMPLETION = 2
    COMPUTE_COMPLETION = 3
    REQUEST_ARRIVAL = 4
    CONTROL = 5


@dataclass(frozen=True, order=True)
class PhysicalGpuId:
    """Identity of one physical GPU in a parallel SGLang deployment."""

    node_id: str
    local_gpu_id: int
    dp_rank: int
    pp_rank: int
    tp_rank: int

    def __post_init__(self) -> None:
        if not str(self.node_id):
            raise ValueError("node_id must be nonempty")
        for name in ("local_gpu_id", "dp_rank", "pp_rank", "tp_rank"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def key(self) -> str:
        return (
            f"{self.node_id}/gpu{self.local_gpu_id}/dp{self.dp_rank}/"
            f"pp{self.pp_rank}/tp{self.tp_rank}"
        )

    def as_dict(self) -> dict[str, str | int]:
        return {
            "node_id": self.node_id,
            "local_gpu_id": self.local_gpu_id,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
        }


class HbmAccount(str, Enum):
    WEIGHTS = "weights"
    RUNTIME = "runtime"
    PRIMARY_KV = "primary_kv"
    PREFIX_KV = "prefix_kv"
    RECOVERY_PENDING = "recovery_pending"
    AEGIS_PENDING = "aegis_pending"
    AEGIS_COMMITTED = "aegis_committed"


class DramAccount(str, Enum):
    RUNTIME = "runtime"
    HICACHE_PRIMARY = "hicache_primary"
    AEGIS_PENDING = "aegis_pending"
    AEGIS_COMMITTED = "aegis_committed"


@dataclass
class GpuHbmLedger:
    """Byte-exact HBM accounting for exactly one physical GPU."""

    capacity_bytes: int
    weights: int = 0
    runtime: int = 0
    primary_kv: int = 0
    prefix_kv: int = 0
    recovery_pending: int = 0
    aegis_pending: int = 0
    aegis_committed: int = 0
    healthy: bool = True

    def __post_init__(self) -> None:
        self.capacity_bytes = _nonnegative_int("HBM capacity_bytes", self.capacity_bytes)
        self.validate()

    @property
    def used_bytes(self) -> int:
        return sum(int(getattr(self, account.value)) for account in HbmAccount)

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def validate(self) -> None:
        for account in HbmAccount:
            _nonnegative_int(f"HBM {account.value}", getattr(self, account.value))
        if self.used_bytes > self.capacity_bytes:
            raise ValueError(
                f"HBM usage {self.used_bytes} exceeds capacity {self.capacity_bytes}"
            )

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "capacity_bytes": self.capacity_bytes,
            **{account.value: int(getattr(self, account.value)) for account in HbmAccount},
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "healthy": bool(self.healthy),
        }


@dataclass
class NodeDramLedger:
    """Byte-exact host DRAM accounting for one node failure domain."""

    capacity_bytes: int
    runtime: int = 0
    hicache_primary: int = 0
    aegis_pending: int = 0
    aegis_committed: int = 0
    healthy: bool = True

    def __post_init__(self) -> None:
        self.capacity_bytes = _nonnegative_int(
            "DRAM capacity_bytes", self.capacity_bytes
        )
        self.validate()

    @property
    def used_bytes(self) -> int:
        return sum(int(getattr(self, account.value)) for account in DramAccount)

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def validate(self) -> None:
        for account in DramAccount:
            _nonnegative_int(f"DRAM {account.value}", getattr(self, account.value))
        if self.used_bytes > self.capacity_bytes:
            raise ValueError(
                f"DRAM usage {self.used_bytes} exceeds capacity {self.capacity_bytes}"
            )

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "capacity_bytes": self.capacity_bytes,
            **{account.value: int(getattr(self, account.value)) for account in DramAccount},
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "healthy": bool(self.healthy),
        }


class StorageTier(str, Enum):
    GPU_PRIMARY = "gpu_primary"
    GPU_PREFIX = "gpu_prefix"
    GPU_AEGIS = "gpu_aegis"
    NODE_DRAM = "node_dram"
    NODE_HICACHE_PRIMARY = "node_hicache_primary"


class KvCopyState(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    LOST = "LOST"


class LocationAccounting(str, Enum):
    """Whether an extent location owns or merely aliases ledger bytes."""

    INTERNAL = "internal"
    EXTERNAL = "external"


class ExtentAccountingAuditMode(str, Enum):
    """KV-ledger conservation policy used by :meth:`audit_invariants`."""

    STRICT = "strict"
    EXTERNAL_AGGREGATE = "external_aggregate"


@dataclass(frozen=True, order=True)
class KvLocation:
    """One physical residency location for a KV extent."""

    tier: StorageTier
    gpu: PhysicalGpuId | None = None
    node_id: str | None = None

    def __post_init__(self) -> None:
        tier = StorageTier(self.tier)
        object.__setattr__(self, "tier", tier)
        if tier in {StorageTier.NODE_DRAM, StorageTier.NODE_HICACHE_PRIMARY}:
            if not self.node_id or self.gpu is not None:
                raise ValueError(f"{tier.value} requires node_id and forbids gpu")
        elif self.gpu is None or self.node_id is not None:
            raise ValueError(f"{tier.value} requires gpu and forbids node_id")

    @classmethod
    def gpu_primary(cls, gpu: PhysicalGpuId) -> "KvLocation":
        return cls(StorageTier.GPU_PRIMARY, gpu=gpu)

    @classmethod
    def gpu_prefix(cls, gpu: PhysicalGpuId) -> "KvLocation":
        return cls(StorageTier.GPU_PREFIX, gpu=gpu)

    @classmethod
    def gpu_aegis(cls, gpu: PhysicalGpuId) -> "KvLocation":
        return cls(StorageTier.GPU_AEGIS, gpu=gpu)

    @classmethod
    def node_dram(cls, node_id: str) -> "KvLocation":
        return cls(StorageTier.NODE_DRAM, node_id=str(node_id))

    @classmethod
    def node_hicache_primary(cls, node_id: str) -> "KvLocation":
        return cls(StorageTier.NODE_HICACHE_PRIMARY, node_id=str(node_id))

    @property
    def key(self) -> str:
        if self.gpu is not None:
            return f"{self.tier.value}:{self.gpu.key}"
        return f"{self.tier.value}:{self.node_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "gpu": None if self.gpu is None else self.gpu.as_dict(),
            "node_id": self.node_id,
        }


@dataclass
class KvExtent:
    """An indivisible KV byte range and the state of each physical copy."""

    extent_id: str
    request_id: str
    size_bytes: int
    token_start: int = 0
    token_end: int = 0
    route_version: int = 1
    ordinal: int = 0
    model_revision: str = "unknown"
    kv_abi: str = "unknown"
    parallel_layout: str = "tp1-pp1"
    pp_rank: int = 0
    tp_rank: int = 0
    shard_count: int = 1
    source_failure_domain: str = "unknown"
    workflow_id: str = ""
    locations: dict[KvLocation, KvCopyState] = field(default_factory=dict)
    location_accounting: dict[KvLocation, LocationAccounting] = field(
        default_factory=dict
    )
    external_account_keys: dict[KvLocation, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.extent_id or not self.request_id:
            raise ValueError("extent_id and request_id must be nonempty")
        self.size_bytes = _nonnegative_int("extent size_bytes", self.size_bytes)
        if self.size_bytes == 0:
            raise ValueError("extent size_bytes must be positive")
        self.token_start = _nonnegative_int("token_start", self.token_start)
        self.token_end = _nonnegative_int("token_end", self.token_end)
        if self.token_end and self.token_end <= self.token_start:
            raise ValueError("token_end must exceed token_start")
        self.route_version = _nonnegative_int("route_version", self.route_version)
        if self.route_version == 0:
            raise ValueError("route_version must be positive")
        self.ordinal = _nonnegative_int("ordinal", self.ordinal)
        self.pp_rank = _nonnegative_int("pp_rank", self.pp_rank)
        self.tp_rank = _nonnegative_int("tp_rank", self.tp_rank)
        self.shard_count = _nonnegative_int("shard_count", self.shard_count)
        if self.shard_count == 0:
            raise ValueError("shard_count must be positive")
        for name in (
            "model_revision",
            "kv_abi",
            "parallel_layout",
            "source_failure_domain",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must be nonempty")
        if not self.workflow_id:
            self.workflow_id = self.request_id
        extra_accounting = set(self.location_accounting) - set(self.locations)
        if extra_accounting:
            raise ValueError("location_accounting contains unknown locations")
        extra_external_keys = set(self.external_account_keys) - set(self.locations)
        if extra_external_keys:
            raise ValueError("external_account_keys contains unknown locations")
        for location in self.locations:
            accounting = LocationAccounting(
                self.location_accounting.get(
                    location, LocationAccounting.INTERNAL
                )
            )
            self.location_accounting[location] = accounting
            if accounting is LocationAccounting.EXTERNAL:
                key = str(
                    self.external_account_keys.get(
                        location, f"extent:{self.extent_id}"
                    )
                )
                if not key:
                    raise ValueError("external_account_key must be nonempty")
                self.external_account_keys[location] = key
            elif location in self.external_account_keys:
                raise ValueError("internal location cannot have external_account_key")

    def snapshot(self) -> dict[str, Any]:
        return {
            "extent_id": self.extent_id,
            "request_id": self.request_id,
            "size_bytes": self.size_bytes,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "route_version": self.route_version,
            "ordinal": self.ordinal,
            "model_revision": self.model_revision,
            "kv_abi": self.kv_abi,
            "parallel_layout": self.parallel_layout,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "shard_count": self.shard_count,
            "source_failure_domain": self.source_failure_domain,
            "workflow_id": self.workflow_id,
            "locations": [
                {
                    "location": location.as_dict(),
                    "state": state.value,
                    "accounting": self.location_accounting.get(
                        location, LocationAccounting.INTERNAL
                    ).value,
                    "external_account_key": self.external_account_keys.get(location),
                }
                for location, state in sorted(
                    self.locations.items(), key=lambda item: item[0].key
                )
            ],
        }


@dataclass(frozen=True, order=True)
class TransferResourceId:
    """A serial resource used by a transfer (copy engine, NIC, or link)."""

    kind: str
    owner: str
    lane: int = 0

    def __post_init__(self) -> None:
        if not self.kind or not self.owner:
            raise ValueError("transfer resource kind and owner must be nonempty")
        if int(self.lane) < 0:
            raise ValueError("transfer resource lane must be nonnegative")

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.owner}:lane{self.lane}"


@dataclass(frozen=True)
class ResourceReservation:
    ready_ns: int
    start_ns: int
    end_ns: int
    resources: tuple[TransferResourceId, ...]


class TransferState(str, Enum):
    QUEUED = "QUEUED"
    # Compatibility alias for callers of the first global-DES draft.
    PENDING = "QUEUED"
    IN_FLIGHT = "IN_FLIGHT"
    ACKED = "ACKED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class TransferDestinationMode(str, Enum):
    """How a transfer accounts for and materializes its destination.

    ``MATERIALIZED_COPY`` preserves the original GlobalDES contract: a
    destination location is installed as PENDING and owns its ledger bytes.
    ``EXTERNAL_PREALLOCATED_SINK`` reserves byte-exact receive capacity in the
    target GPU's RECOVERY_PENDING account without fabricating an extent
    location owned by the scheduler.  ``OWNERSHIP_ACTIVATION`` exposes an
    already resident same-GPU Aegis copy and therefore reserves neither copy
    resources nor additional HBM capacity.
    """

    MATERIALIZED_COPY = "MATERIALIZED_COPY"
    EXTERNAL_PREALLOCATED_SINK = "EXTERNAL_PREALLOCATED_SINK"
    OWNERSHIP_ACTIVATION = "OWNERSHIP_ACTIVATION"


class CommitGroupState(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


@dataclass
class TierCommitGroup:
    """Atomic multi-shard commit boundary for one independent Aegis tier."""

    group_id: str
    tier: StorageTier
    route_version: int
    model_revision: str
    kv_abi: str
    parallel_layout: str
    expected_shard_keys: tuple[str, ...]
    shard_transfers: dict[str, str] = field(default_factory=dict)
    acked_shards: set[str] = field(default_factory=set)
    state: CommitGroupState = CommitGroupState.PENDING

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("commit group_id must be nonempty")
        self.tier = StorageTier(self.tier)
        if self.tier not in {StorageTier.GPU_AEGIS, StorageTier.NODE_DRAM}:
            raise ValueError("commit groups are only valid for Aegis HBM or DRAM")
        self.route_version = _nonnegative_int(
            "group route_version", self.route_version
        )
        if self.route_version == 0:
            raise ValueError("group route_version must be positive")
        for name in ("model_revision", "kv_abi", "parallel_layout"):
            if not str(getattr(self, name)):
                raise ValueError(f"group {name} must be nonempty")
        self.expected_shard_keys = tuple(str(key) for key in self.expected_shard_keys)
        if not self.expected_shard_keys or any(
            not key for key in self.expected_shard_keys
        ):
            raise ValueError("expected_shard_keys must be nonempty strings")
        if len(set(self.expected_shard_keys)) != len(self.expected_shard_keys):
            raise ValueError("expected_shard_keys must not contain duplicates")

    def snapshot(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "tier": self.tier.value,
            "route_version": self.route_version,
            "model_revision": self.model_revision,
            "kv_abi": self.kv_abi,
            "parallel_layout": self.parallel_layout,
            "expected_shard_keys": list(self.expected_shard_keys),
            "shard_transfers": dict(sorted(self.shard_transfers.items())),
            "acked_shards": sorted(self.acked_shards),
            "state": self.state.value,
        }


@dataclass
class RecoveryLoadGroup:
    """Atomic PP x TP recovery boundary for one failover replay.

    One logical PP/TP shard may require several source extents, so unlike a
    :class:`TierCommitGroup`, each shard key can bind multiple transfers.  A
    group must be sealed after every expected shard has at least one bound
    transfer.  No transfer becomes COMMITTED until every bound transfer ACKs.
    """

    group_id: str
    retry_request_id: str
    request_backup_key: str
    route_version: int
    expected_shard_keys: tuple[str, ...]
    shard_transfers: dict[str, list[str]] = field(default_factory=dict)
    acked_transfers: set[str] = field(default_factory=set)
    adopted_shards: set[str] = field(default_factory=set)
    sealed: bool = False
    state: CommitGroupState = CommitGroupState.PENDING
    adopted_at_ns: int | None = None
    discarded: bool = False
    discard_reason: str | None = None
    discarded_at_ns: int | None = None

    def __post_init__(self) -> None:
        for name in ("group_id", "retry_request_id", "request_backup_key"):
            if not str(getattr(self, name)):
                raise ValueError(f"recovery {name} must be nonempty")
        self.route_version = _nonnegative_int(
            "recovery route_version", self.route_version
        )
        if self.route_version == 0:
            raise ValueError("recovery route_version must be positive")
        self.expected_shard_keys = tuple(
            str(key) for key in self.expected_shard_keys
        )
        if not self.expected_shard_keys or any(
            not key for key in self.expected_shard_keys
        ):
            raise ValueError("recovery expected_shard_keys must be nonempty strings")
        if len(set(self.expected_shard_keys)) != len(self.expected_shard_keys):
            raise ValueError("recovery expected_shard_keys must not contain duplicates")

    @property
    def transfer_ids(self) -> tuple[str, ...]:
        return tuple(
            transfer_id
            for shard_key in self.expected_shard_keys
            for transfer_id in self.shard_transfers.get(shard_key, ())
        )

    def snapshot(self) -> dict[str, Any]:
        abandoned_shards = (
            sorted(set(self.expected_shard_keys) - self.adopted_shards)
            if self.discarded
            else []
        )
        discard_scope = (
            None
            if not self.discarded
            else (
                "all"
                if not self.adopted_shards
                else "unadopted_remainder"
            )
        )
        return {
            "group_id": self.group_id,
            "retry_request_id": self.retry_request_id,
            "request_backup_key": self.request_backup_key,
            "route_version": self.route_version,
            "expected_shard_keys": list(self.expected_shard_keys),
            "shard_transfers": {
                key: list(value)
                for key, value in sorted(self.shard_transfers.items())
            },
            "acked_transfers": sorted(self.acked_transfers),
            "adopted_shards": sorted(self.adopted_shards),
            "sealed": bool(self.sealed),
            "state": self.state.value,
            "fully_adopted": self.adopted_shards
            == set(self.expected_shard_keys),
            "adopted_at_ns": self.adopted_at_ns,
            "discarded": bool(self.discarded),
            "discard_scope": discard_scope,
            "abandoned_shards": abandoned_shards,
            "discard_reason": self.discard_reason,
            "terminal_reason": self.discard_reason,
            "discarded_at_ns": self.discarded_at_ns,
        }


@dataclass
class TransferRecord:
    transfer_id: str
    extent_id: str
    source: KvLocation
    destination: KvLocation
    size_bytes: int
    ready_ns: int
    start_ns: int
    end_ns: int
    resources: tuple[TransferResourceId, ...]
    start_event_id: str
    completion_event_id: str
    state: TransferState = TransferState.QUEUED
    remove_source_on_complete: bool = False
    group_id: str | None = None
    shard_key: str | None = None
    destination_mode: TransferDestinationMode = (
        TransferDestinationMode.MATERIALIZED_COPY
    )
    token_start: int = 0
    token_end: int = 0
    recovery_group_id: str | None = None
    recovery_shard_key: str | None = None
    recovery_capacity_hold_active: bool = False
    recovery_capacity_hold_lost: bool = False
    recovery_adopted: bool = False
    recovery_discarded: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "extent_id": self.extent_id,
            "source": self.source.as_dict(),
            "destination": self.destination.as_dict(),
            "size_bytes": self.size_bytes,
            "ready_ns": self.ready_ns,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "resources": [resource.key for resource in self.resources],
            "start_event_id": self.start_event_id,
            "completion_event_id": self.completion_event_id,
            "state": self.state.value,
            "remove_source_on_complete": self.remove_source_on_complete,
            "group_id": self.group_id,
            "shard_key": self.shard_key,
            "destination_mode": self.destination_mode.value,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "recovery_group_id": self.recovery_group_id,
            "recovery_shard_key": self.recovery_shard_key,
            "recovery_capacity_hold_active": bool(
                self.recovery_capacity_hold_active
            ),
            "recovery_capacity_hold_lost": bool(
                self.recovery_capacity_hold_lost
            ),
            "recovery_adopted": bool(self.recovery_adopted),
            "recovery_discarded": bool(self.recovery_discarded),
        }


EventCallback = Callable[["GlobalDES", "GlobalEvent"], None]


@dataclass(order=True, frozen=True)
class GlobalEvent:
    """One event ordered only by time, priority, then insertion sequence."""

    timestamp_ns: int
    priority: int
    sequence: int
    event_id: str = field(compare=False)
    kind: str = field(compare=False)
    payload: Any = field(default=None, compare=False)
    callback: EventCallback | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class JournalEntry:
    timestamp_ns: int
    event_id: str
    kind: str
    entity: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "event_id": self.event_id,
            "kind": self.kind,
            "entity": self.entity,
            "before": copy.deepcopy(dict(self.before)),
            "after": copy.deepcopy(dict(self.after)),
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


class GlobalDES:
    """Single-cursor, stable-heap DES with physical KV resource ledgers."""

    def __init__(self, *, start_ns: int = 0) -> None:
        self.now_ns = _nonnegative_int("start_ns", start_ns)
        self._events: list[GlobalEvent] = []
        # Keep an explicit lifecycle for every event ever accepted by this
        # kernel.  The heap alone cannot distinguish an unknown event from one
        # that has already been dispatched, which is essential for fail-closed
        # cancellation and for post-run audit evidence.
        self._events_by_id: dict[str, GlobalEvent] = {}
        self._event_states: dict[str, str] = {}
        self._event_sequence = 0
        self._operation_sequence = 0
        self._transfer_sequence = 0
        self._current_event_id: str | None = None

        self.gpu_hbm: dict[PhysicalGpuId, GpuHbmLedger] = {}
        self.node_dram: dict[str, NodeDramLedger] = {}
        self.extents: dict[str, KvExtent] = {}
        self.resource_busy_until_ns: dict[TransferResourceId, int] = {}
        self.transfers: dict[str, TransferRecord] = {}
        self.commit_groups: dict[str, TierCommitGroup] = {}
        self.recovery_load_groups: dict[str, RecoveryLoadGroup] = {}

        self._journal: list[JournalEntry] = []
        self._gpu_timeseries: list[dict[str, Any]] = []
        self._dram_timeseries: list[dict[str, Any]] = []
        self._extent_timeseries: list[dict[str, Any]] = []
        self._transfer_timeseries: list[dict[str, Any]] = []
        self._group_timeseries: list[dict[str, Any]] = []
        self._recovery_group_timeseries: list[dict[str, Any]] = []
        self._resource_timeseries: list[dict[str, Any]] = []

    @property
    def journal(self) -> tuple[JournalEntry, ...]:
        return tuple(self._journal)

    def _operation_id(self) -> str:
        if self._current_event_id is not None:
            return self._current_event_id
        event_id = f"op-{self._operation_sequence:012d}"
        self._operation_sequence += 1
        return event_id

    def _append_journal(
        self,
        *,
        event_id: str,
        kind: str,
        entity: str,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._journal.append(
            JournalEntry(
                timestamp_ns=self.now_ns,
                event_id=str(event_id),
                kind=str(kind),
                entity=str(entity),
                before=MappingProxyType(copy.deepcopy(dict(before))),
                after=MappingProxyType(copy.deepcopy(dict(after))),
                metadata=MappingProxyType(copy.deepcopy(dict(metadata or {}))),
            )
        )

    def schedule_at(
        self,
        timestamp_ns: int,
        *,
        kind: str,
        priority: EventPriority | int = EventPriority.CONTROL,
        payload: Any = None,
        callback: EventCallback | None = None,
    ) -> GlobalEvent:
        """Insert an event without advancing this or any rank-local clock."""

        timestamp = _nonnegative_int("event timestamp_ns", timestamp_ns)
        if timestamp < self.now_ns:
            raise ValueError(
                f"cannot schedule {kind!r} at {timestamp}; DES is at {self.now_ns}"
            )
        if not str(kind):
            raise ValueError("event kind must be nonempty")
        event = GlobalEvent(
            timestamp_ns=timestamp,
            priority=int(priority),
            sequence=self._event_sequence,
            event_id=f"evt-{self._event_sequence:012d}",
            kind=str(kind),
            payload=payload,
            callback=callback,
        )
        self._event_sequence += 1
        heapq.heappush(self._events, event)
        self._events_by_id[event.event_id] = event
        self._event_states[event.event_id] = "queued"
        self._append_journal(
            event_id=event.event_id,
            kind="event_scheduled",
            entity=event.kind,
            before={"queued": False},
            after={
                "queued": True,
                "timestamp_ns": timestamp,
                "priority": event.priority,
                "sequence": event.sequence,
            },
        )
        return event

    def cancel_event(
        self,
        event_id: str,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Cancel one event that has not started dispatching.

        Returns ``True`` for the first cancellation and ``False`` when the
        same cancellation is replayed.  Unknown IDs and events that have
        started dispatching fail closed.  A cancelled event is physically
        removed from the authoritative heap, so it can neither run its
        callback nor advance :attr:`now_ns`.
        """

        normalized_event_id = str(event_id)
        if not normalized_event_id:
            raise ValueError("event_id must be nonempty")
        if reason is not None and not str(reason):
            raise ValueError("reason must be nonempty when provided")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping when provided")

        event = self._events_by_id.get(normalized_event_id)
        if event is None:
            raise KeyError(f"unknown event_id {normalized_event_id!r}")
        state = self._event_states[normalized_event_id]
        if state == "cancelled":
            return False
        if state != "queued":
            raise RuntimeError(
                f"cannot cancel event {normalized_event_id!r} in state {state!r}"
            )

        try:
            self._events.remove(event)
        except ValueError as exc:  # Defensive lifecycle/heap consistency gate.
            raise RuntimeError(
                f"queued event {normalized_event_id!r} is absent from the heap"
            ) from exc
        heapq.heapify(self._events)
        self._event_states[normalized_event_id] = "cancelled"
        self._append_journal(
            event_id=normalized_event_id,
            kind="event_cancelled",
            entity=event.kind,
            before={"queued": True, "status": "queued"},
            after={"queued": False, "status": "cancelled"},
            metadata={
                "reason": None if reason is None else str(reason),
                "scheduled_timestamp_ns": event.timestamp_ns,
                "priority": event.priority,
                "sequence": event.sequence,
                "cancellation_metadata": copy.deepcopy(dict(metadata or {})),
            },
        )
        return True

    def schedule_after(
        self,
        delay_ns: int,
        *,
        kind: str,
        priority: EventPriority | int = EventPriority.CONTROL,
        payload: Any = None,
        callback: EventCallback | None = None,
    ) -> GlobalEvent:
        return self.schedule_at(
            self.now_ns + _nonnegative_int("delay_ns", delay_ns),
            kind=kind,
            priority=priority,
            payload=payload,
            callback=callback,
        )

    def has_events(self) -> bool:
        return bool(self._events)

    def peek_next(self) -> GlobalEvent:
        if not self._events:
            raise IndexError("global DES event queue is empty")
        return self._events[0]

    def pop_next(self) -> GlobalEvent:
        """Advance to and execute the earliest globally ordered event."""

        if not self._events:
            raise IndexError("global DES event queue is empty")
        event = heapq.heappop(self._events)
        state = self._event_states.get(event.event_id)
        if state != "queued":
            raise RuntimeError(
                f"event heap contains {event.event_id!r} in state {state!r}"
            )
        self._event_states[event.event_id] = "dispatching"
        before_now = self.now_ns
        if event.timestamp_ns < before_now:  # Defensive heap/cursor invariant.
            raise RuntimeError("event heap attempted to move the DES clock backwards")
        self.now_ns = event.timestamp_ns
        previous_event_id = self._current_event_id
        self._current_event_id = event.event_id
        error: BaseException | None = None
        try:
            if event.callback is not None:
                event.callback(self, event)
        except BaseException as exc:  # Journal the failed dispatch, then preserve it.
            error = exc
        finally:
            final_state = "error" if error is not None else "completed"
            self._event_states[event.event_id] = final_state
            self._append_journal(
                event_id=event.event_id,
                kind="event_dispatched",
                entity=event.kind,
                before={"now_ns": before_now, "queued": True},
                after={
                    "now_ns": self.now_ns,
                    "queued": False,
                    "status": final_state,
                },
            )
            self._current_event_id = previous_event_id
        if error is not None:
            raise error
        return event

    def run(self, *, until_ns: int | None = None) -> list[GlobalEvent]:
        """Run all events, or only events at/before ``until_ns``."""

        limit = None if until_ns is None else _nonnegative_int("until_ns", until_ns)
        if limit is not None and limit < self.now_ns:
            raise ValueError("until_ns cannot precede the current DES time")
        dispatched: list[GlobalEvent] = []
        while self._events and (limit is None or self._events[0].timestamp_ns <= limit):
            dispatched.append(self.pop_next())
        return dispatched

    def register_gpu(
        self,
        gpu: PhysicalGpuId,
        *,
        capacity_bytes: int,
        weights_bytes: int = 0,
        runtime_bytes: int = 0,
    ) -> GpuHbmLedger:
        if gpu in self.gpu_hbm:
            raise ValueError(f"GPU {gpu.key} is already registered")
        ledger = GpuHbmLedger(
            capacity_bytes=capacity_bytes,
            weights=_nonnegative_int("weights_bytes", weights_bytes),
            runtime=_nonnegative_int("runtime_bytes", runtime_bytes),
        )
        self.gpu_hbm[gpu] = ledger
        event_id = self._operation_id()
        after = ledger.snapshot()
        self._append_journal(
            event_id=event_id,
            kind="gpu_registered",
            entity=gpu.key,
            before={},
            after=after,
        )
        self._record_gpu(gpu, event_id, "gpu_registered")
        return ledger

    def register_node_dram(
        self,
        node_id: str,
        *,
        capacity_bytes: int,
        runtime_bytes: int = 0,
        hicache_primary_bytes: int = 0,
    ) -> NodeDramLedger:
        node = str(node_id)
        if not node:
            raise ValueError("node_id must be nonempty")
        if node in self.node_dram:
            raise ValueError(f"DRAM node {node!r} is already registered")
        ledger = NodeDramLedger(
            capacity_bytes=capacity_bytes,
            runtime=_nonnegative_int("runtime_bytes", runtime_bytes),
            hicache_primary=_nonnegative_int(
                "hicache_primary_bytes", hicache_primary_bytes
            ),
        )
        self.node_dram[node] = ledger
        event_id = self._operation_id()
        after = ledger.snapshot()
        self._append_journal(
            event_id=event_id,
            kind="dram_registered",
            entity=node,
            before={},
            after=after,
        )
        self._record_dram(node, event_id, "dram_registered")
        return ledger

    def adjust_gpu_hbm(
        self,
        gpu: PhysicalGpuId,
        account: HbmAccount | str,
        delta_bytes: int,
        *,
        reason: str = "accounting",
        event_id: str | None = None,
    ) -> GpuHbmLedger:
        ledger = self._require_gpu(gpu)
        parsed_account = HbmAccount(account)
        delta = int(delta_bytes)
        if delta > 0 and not ledger.healthy:
            raise ValueError(f"cannot allocate HBM on unhealthy GPU {gpu.key}")
        before = ledger.snapshot()
        candidate = int(getattr(ledger, parsed_account.value)) + delta
        if candidate < 0:
            raise ValueError(
                f"HBM {parsed_account.value} would become negative on {gpu.key}"
            )
        candidate_used = ledger.used_bytes + delta
        if candidate_used > ledger.capacity_bytes:
            raise ValueError(
                f"HBM usage {candidate_used} exceeds capacity "
                f"{ledger.capacity_bytes} on {gpu.key}"
            )
        setattr(ledger, parsed_account.value, candidate)
        ledger.validate()
        op_id = event_id or self._operation_id()
        after = ledger.snapshot()
        self._append_journal(
            event_id=op_id,
            kind="gpu_hbm_adjusted",
            entity=gpu.key,
            before=before,
            after=after,
            metadata={
                "account": parsed_account.value,
                "delta_bytes": delta,
                "reason": str(reason),
            },
        )
        self._record_gpu(gpu, op_id, reason)
        return ledger

    def adjust_node_dram(
        self,
        node_id: str,
        account: DramAccount | str,
        delta_bytes: int,
        *,
        reason: str = "accounting",
        event_id: str | None = None,
    ) -> NodeDramLedger:
        node = str(node_id)
        ledger = self._require_dram(node)
        parsed_account = DramAccount(account)
        delta = int(delta_bytes)
        if delta > 0 and not ledger.healthy:
            raise ValueError(f"cannot allocate DRAM on unhealthy node {node}")
        before = ledger.snapshot()
        candidate = int(getattr(ledger, parsed_account.value)) + delta
        if candidate < 0:
            raise ValueError(
                f"DRAM {parsed_account.value} would become negative on {node}"
            )
        candidate_used = ledger.used_bytes + delta
        if candidate_used > ledger.capacity_bytes:
            raise ValueError(
                f"DRAM usage {candidate_used} exceeds capacity "
                f"{ledger.capacity_bytes} on {node}"
            )
        setattr(ledger, parsed_account.value, candidate)
        ledger.validate()
        op_id = event_id or self._operation_id()
        after = ledger.snapshot()
        self._append_journal(
            event_id=op_id,
            kind="node_dram_adjusted",
            entity=node,
            before=before,
            after=after,
            metadata={
                "account": parsed_account.value,
                "delta_bytes": delta,
                "reason": str(reason),
            },
        )
        self._record_dram(node, op_id, reason)
        return ledger

    def register_extent(
        self,
        *,
        extent_id: str,
        request_id: str,
        size_bytes: int,
        token_start: int = 0,
        token_end: int = 0,
        route_version: int = 1,
        ordinal: int = 0,
        model_revision: str = "unknown",
        kv_abi: str = "unknown",
        parallel_layout: str = "tp1-pp1",
        pp_rank: int = 0,
        tp_rank: int = 0,
        shard_count: int = 1,
        source_failure_domain: str = "unknown",
        workflow_id: str = "",
    ) -> KvExtent:
        if extent_id in self.extents:
            raise ValueError(f"extent {extent_id!r} is already registered")
        extent = KvExtent(
            extent_id=str(extent_id),
            request_id=str(request_id),
            size_bytes=size_bytes,
            token_start=token_start,
            token_end=token_end,
            route_version=route_version,
            ordinal=ordinal,
            model_revision=model_revision,
            kv_abi=kv_abi,
            parallel_layout=parallel_layout,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
            shard_count=shard_count,
            source_failure_domain=source_failure_domain,
            workflow_id=workflow_id,
        )
        self.extents[extent.extent_id] = extent
        event_id = self._operation_id()
        snapshot = extent.snapshot()
        self._append_journal(
            event_id=event_id,
            kind="extent_registered",
            entity=extent.extent_id,
            before={},
            after=snapshot,
        )
        self._record_extent(extent, event_id, "extent_registered")
        return extent

    def register_commit_group(
        self,
        *,
        group_id: str,
        tier: StorageTier,
        route_version: int,
        model_revision: str,
        kv_abi: str,
        parallel_layout: str,
        expected_shard_keys: Sequence[str],
    ) -> TierCommitGroup:
        """Register one tier-local, fail-closed multi-shard commit group."""

        identity = str(group_id)
        if identity in self.commit_groups:
            raise ValueError(f"commit group {identity!r} is already registered")
        group = TierCommitGroup(
            group_id=identity,
            tier=tier,
            route_version=route_version,
            model_revision=str(model_revision),
            kv_abi=str(kv_abi),
            parallel_layout=str(parallel_layout),
            expected_shard_keys=tuple(expected_shard_keys),
        )
        self.commit_groups[group.group_id] = group
        event_id = self._operation_id()
        self._append_journal(
            event_id=event_id,
            kind="commit_group_registered",
            entity=group.group_id,
            before={},
            after=group.snapshot(),
        )
        self._record_group(group, event_id, "commit_group_registered")
        return group

    def register_recovery_load_group(
        self,
        *,
        group_id: str,
        retry_request_id: str,
        request_backup_key: str,
        route_version: int,
        expected_shard_keys: Sequence[str],
    ) -> RecoveryLoadGroup:
        """Register an unsealed atomic recovery group.

        The caller binds one or more transfers for every PP/TP shard and then
        calls :meth:`seal_recovery_load_group`.  Keeping registration and
        sealing separate lets a planner fail closed when admission of any
        shard fails.
        """

        identity = str(group_id)
        if identity in self.recovery_load_groups or identity in self.commit_groups:
            raise ValueError(f"group {identity!r} is already registered")
        group = RecoveryLoadGroup(
            group_id=identity,
            retry_request_id=str(retry_request_id),
            request_backup_key=str(request_backup_key),
            route_version=route_version,
            expected_shard_keys=tuple(expected_shard_keys),
        )
        self.recovery_load_groups[group.group_id] = group
        event_id = self._operation_id()
        self._append_journal(
            event_id=event_id,
            kind="recovery_load_group_registered",
            entity=group.group_id,
            before={},
            after=group.snapshot(),
        )
        self._record_recovery_group(
            group, event_id, "recovery_load_group_registered"
        )
        return group

    def seal_recovery_load_group(self, group_id: str) -> RecoveryLoadGroup:
        """Seal a recovery group after all expected PP/TP shards are bound.

        Missing shards abort the entire group and release every admitted
        recovery-capacity hold before raising.
        """

        group = self._require_recovery_group(group_id)
        if group.state is not CommitGroupState.PENDING:
            raise ValueError(
                f"recovery group {group.group_id} is {group.state.value}, not PENDING"
            )
        if group.sealed:
            return group
        missing = tuple(
            key
            for key in group.expected_shard_keys
            if not group.shard_transfers.get(key)
        )
        if missing:
            self.abort_recovery_load_group(
                group.group_id,
                reason=f"missing_shards:{','.join(missing)}",
            )
            raise ValueError(
                f"recovery group {group.group_id} is missing shards: "
                f"{', '.join(missing)}"
            )
        event_id = self._operation_id()
        before = group.snapshot()
        group.sealed = True
        self._append_journal(
            event_id=event_id,
            kind="recovery_load_group_sealed",
            entity=group.group_id,
            before=before,
            after=group.snapshot(),
        )
        self._record_recovery_group(
            group, event_id, "recovery_load_group_sealed"
        )
        if group.acked_transfers == set(group.transfer_ids):
            self._commit_recovery_load_group(group, event_id=event_id)
        return group

    def adopt_recovery_load_shard(
        self,
        group_id: str,
        shard_key: str,
    ) -> RecoveryLoadGroup:
        """Atomically hand one committed recovery shard to its Scheduler.

        Adoption is intentionally separate from transfer commit.  A committed
        external sink still owns ``RECOVERY_PENDING`` capacity until the
        corresponding Scheduler stage has successfully prepared the replay;
        only then can its allocator snapshot take over those bytes.  Same-GPU
        ownership activation releases the byte-identical Aegis source copy at
        the same boundary and therefore never creates a second HBM peak.
        """

        return self.adopt_recovery_load_shards(group_id, (shard_key,))

    def recovery_load_shards_adoption_error(
        self,
        group_id: str,
        shard_keys: Sequence[str] | None = None,
    ) -> str | None:
        """Return a fail-closed dry-run error for Scheduler ownership handoff."""

        group = self.recovery_load_groups.get(str(group_id))
        if group is None:
            return "unknown_recovery_group"
        requested = tuple(
            group.expected_shard_keys
            if shard_keys is None
            else dict.fromkeys(str(key) for key in shard_keys)
        )
        if not requested or any(not key for key in requested):
            return "empty_shard_set"
        if not set(requested).issubset(set(group.expected_shard_keys)):
            return "unknown_shard"
        if group.state is not CommitGroupState.COMMITTED:
            return f"group_{group.state.value.lower()}"
        if group.discarded:
            return "group_discarded"

        external_required: dict[PhysicalGpuId, int] = {}
        activation_sources: set[tuple[str, KvLocation]] = set()
        for shard_key in requested:
            if shard_key in group.adopted_shards:
                continue
            transfer_ids = group.shard_transfers.get(shard_key, ())
            if not transfer_ids:
                return f"empty_shard:{shard_key}"
            for transfer_id in transfer_ids:
                transfer = self.transfers.get(transfer_id)
                if transfer is None:
                    return f"missing_transfer:{transfer_id}"
                if transfer.state is not TransferState.COMMITTED:
                    return f"transfer_not_committed:{transfer_id}"
                if transfer.recovery_adopted or transfer.recovery_discarded:
                    return f"transfer_terminal_ownership:{transfer_id}"
                if not self._location_is_healthy(transfer.destination):
                    return f"destination_unhealthy:{transfer_id}"
                if (
                    transfer.destination_mode
                    is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                ):
                    if (
                        not transfer.recovery_capacity_hold_active
                        or transfer.recovery_capacity_hold_lost
                    ):
                        return f"capacity_hold_unavailable:{transfer_id}"
                    assert transfer.destination.gpu is not None
                    external_required[transfer.destination.gpu] = (
                        external_required.get(transfer.destination.gpu, 0)
                        + transfer.size_bytes
                    )
                elif (
                    transfer.destination_mode
                    is TransferDestinationMode.OWNERSHIP_ACTIVATION
                ):
                    extent = self.extents.get(transfer.extent_id)
                    if extent is None:
                        return f"missing_extent:{transfer.extent_id}"
                    if (
                        transfer.size_bytes != extent.size_bytes
                        or transfer.token_start != extent.token_start
                        or transfer.token_end != extent.token_end
                    ):
                        return f"partial_ownership_activation:{transfer_id}"
                    source_key = (extent.extent_id, transfer.source)
                    if source_key in activation_sources:
                        return f"duplicate_activation_source:{transfer.extent_id}"
                    activation_sources.add(source_key)
                    if (
                        extent.locations.get(transfer.source)
                        is not KvCopyState.COMMITTED
                        or not self._location_is_healthy(transfer.source)
                    ):
                        return f"activation_source_unavailable:{transfer_id}"
                else:
                    return f"invalid_destination_mode:{transfer_id}"
        for gpu, required in external_required.items():
            ledger = self._require_gpu(gpu)
            if not ledger.healthy or ledger.recovery_pending < required:
                return f"capacity_hold_underflow:{gpu.key}"
        return None

    def adopt_recovery_load_shards(
        self,
        group_id: str,
        shard_keys: Sequence[str],
    ) -> RecoveryLoadGroup:
        """Atomically adopt a nonempty set of PP/TP shards.

        The method validates every transfer before mutating any ledger.  It is
        idempotent for shards already adopted by the same group.  Ownership
        activation currently requires a complete source extent: a partial
        activation fails closed so aggregate extent accounting can never be
        silently under-released.
        """

        group = self._require_recovery_group(group_id)
        requested = tuple(dict.fromkeys(str(key) for key in shard_keys))
        if not requested or any(not key for key in requested):
            raise ValueError("recovery adoption requires nonempty shard keys")
        unknown = sorted(set(requested) - set(group.expected_shard_keys))
        if unknown:
            raise ValueError(
                f"recovery group {group.group_id} has no shards: {', '.join(unknown)}"
            )
        if group.state is not CommitGroupState.COMMITTED:
            raise ValueError(
                f"recovery group {group.group_id} is {group.state.value}, not COMMITTED"
            )
        if group.discarded:
            raise ValueError(f"recovery group {group.group_id} was discarded")

        pending = tuple(
            key for key in requested if key not in group.adopted_shards
        )
        if not pending:
            return group
        validation_error = self.recovery_load_shards_adoption_error(
            group.group_id, pending
        )
        if validation_error is not None:
            raise RuntimeError(
                f"recovery group {group.group_id} is not adoptable: "
                f"{validation_error}"
            )

        transfers: list[TransferRecord] = []
        activation_sources: set[tuple[str, KvLocation]] = set()
        external_required: dict[PhysicalGpuId, int] = {}
        for shard_key in pending:
            transfer_ids = group.shard_transfers.get(shard_key, ())
            if not transfer_ids:
                raise RuntimeError(
                    f"recovery group {group.group_id} shard {shard_key} is empty"
                )
            for transfer_id in transfer_ids:
                transfer = self.transfers[transfer_id]
                if transfer.state is not TransferState.COMMITTED:
                    raise RuntimeError(
                        f"recovery transfer {transfer_id} is not COMMITTED"
                    )
                if transfer.recovery_adopted:
                    raise RuntimeError(
                        f"recovery transfer {transfer_id} was adopted outside its shard"
                    )
                if transfer.recovery_discarded:
                    raise RuntimeError(
                        f"recovery transfer {transfer_id} was discarded"
                    )
                if not self._location_is_healthy(transfer.destination):
                    raise RuntimeError(
                        f"recovery transfer {transfer_id} destination is unhealthy"
                    )
                if (
                    transfer.destination_mode
                    is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                ):
                    if (
                        not transfer.recovery_capacity_hold_active
                        or transfer.recovery_capacity_hold_lost
                    ):
                        raise RuntimeError(
                            f"recovery transfer {transfer_id} has no live capacity hold"
                        )
                    assert transfer.destination.gpu is not None
                    external_required[transfer.destination.gpu] = (
                        external_required.get(transfer.destination.gpu, 0)
                        + transfer.size_bytes
                    )
                else:
                    if (
                        transfer.destination_mode
                        is not TransferDestinationMode.OWNERSHIP_ACTIVATION
                    ):
                        raise RuntimeError(
                            f"recovery transfer {transfer_id} has invalid destination mode"
                        )
                    extent = self._require_extent(transfer.extent_id)
                    if (
                        transfer.size_bytes != extent.size_bytes
                        or transfer.token_start != extent.token_start
                        or transfer.token_end != extent.token_end
                    ):
                        raise RuntimeError(
                            "partial OWNERSHIP_ACTIVATION cannot be adopted; "
                            "use deterministic recompute fallback"
                        )
                    source_key = (extent.extent_id, transfer.source)
                    if source_key in activation_sources:
                        raise RuntimeError(
                            f"recovery group reuses activation source {extent.extent_id}"
                        )
                    activation_sources.add(source_key)
                    if (
                        extent.locations.get(transfer.source)
                        is not KvCopyState.COMMITTED
                        or not self._location_is_healthy(transfer.source)
                    ):
                        raise RuntimeError(
                            f"recovery activation source {extent.extent_id} is unavailable"
                        )
                transfers.append(transfer)

        for gpu, required in external_required.items():
            ledger = self._require_gpu(gpu)
            if not ledger.healthy or ledger.recovery_pending < required:
                raise RuntimeError(
                    f"GPU {gpu.key} cannot hand off {required} recovery bytes"
                )

        event_id = self._operation_id()
        group_before = group.snapshot()
        for transfer in transfers:
            transfer_before = transfer.snapshot()
            if (
                transfer.destination_mode
                is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
            ):
                self._release_recovery_capacity_hold(
                    transfer,
                    event_id=event_id,
                    reason="recovery_shard_adopted",
                )
            else:
                self.release_extent_location(
                    transfer.extent_id,
                    transfer.source,
                    reason="recovery_ownership_adopted",
                )
            transfer.recovery_adopted = True
            self._append_journal(
                event_id=event_id,
                kind="recovery_transfer_adopted",
                entity=transfer.transfer_id,
                before=transfer_before,
                after=transfer.snapshot(),
                metadata={
                    "group_id": group.group_id,
                    "shard_key": transfer.recovery_shard_key,
                },
            )
            self._record_transfer(
                transfer, event_id, "recovery_transfer_adopted"
            )

        group.adopted_shards.update(pending)
        if group.adopted_shards == set(group.expected_shard_keys):
            group.adopted_at_ns = self.now_ns
        self._append_journal(
            event_id=event_id,
            kind="recovery_load_shards_adopted",
            entity=group.group_id,
            before=group_before,
            after=group.snapshot(),
            metadata={"shard_keys": list(pending)},
        )
        self._record_recovery_group(
            group, event_id, "recovery_load_shards_adopted"
        )
        return group

    def discard_recovery_load_group(
        self,
        group_id: str,
        *,
        reason: str,
    ) -> RecoveryLoadGroup:
        """Release unadopted recovery holds before a recompute fallback.

        A pending group is first aborted, which releases every active receive
        hold.  A committed group can be discarded only before any Scheduler
        shard has adopted it.  Protection-owned activation sources are left in
        place; external receive holds are released byte-exactly.
        """

        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("discard reason must be nonempty")
        group = self._require_recovery_group(group_id)
        if group.discarded:
            return group
        if group.adopted_shards:
            raise ValueError(
                f"cannot discard partially adopted recovery group {group.group_id}"
            )
        if group.state is CommitGroupState.PENDING:
            self.abort_recovery_load_group(
                group.group_id,
                reason=f"discarded:{normalized_reason}",
            )

        event_id = self._operation_id()
        before = group.snapshot()
        for transfer_id in group.transfer_ids:
            transfer = self.transfers[transfer_id]
            transfer_before = transfer.snapshot()
            if transfer.recovery_capacity_hold_active:
                self._release_recovery_capacity_hold(
                    transfer,
                    event_id=event_id,
                    reason="recovery_group_discarded",
                )
            transfer.recovery_discarded = True
            self._append_journal(
                event_id=event_id,
                kind="recovery_transfer_discarded",
                entity=transfer.transfer_id,
                before=transfer_before,
                after=transfer.snapshot(),
                metadata={"group_id": group.group_id, "reason": normalized_reason},
            )
            self._record_transfer(
                transfer, event_id, "recovery_transfer_discarded"
            )
        group.discarded = True
        group.discard_reason = normalized_reason
        group.discarded_at_ns = self.now_ns
        self._append_journal(
            event_id=event_id,
            kind="recovery_load_group_discarded",
            entity=group.group_id,
            before=before,
            after=group.snapshot(),
            metadata={"reason": normalized_reason},
        )
        self._record_recovery_group(
            group, event_id, "recovery_load_group_discarded"
        )
        return group

    def abandon_partially_adopted_recovery_group(
        self,
        group_id: str,
        *,
        reason: str,
    ) -> RecoveryLoadGroup:
        """Terminally discard only the shards not yet owned by a Scheduler.

        This operation is reserved for a target failure after at least one PP
        stage adopted its shard.  Adopted shards remain immutable historical
        evidence; every remaining external receive hold is released and every
        unadopted transfer is marked discarded.  No protection extent is
        materialized or removed by this operation.
        """

        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("partial-abandon reason must be nonempty")
        group = self._require_recovery_group(group_id)
        if group.discarded:
            return group
        if group.state is not CommitGroupState.COMMITTED:
            raise ValueError(
                f"partially adopted group {group.group_id} is not COMMITTED"
            )
        expected = set(group.expected_shard_keys)
        if not group.adopted_shards or group.adopted_shards == expected:
            raise ValueError(
                f"recovery group {group.group_id} is not partially adopted"
            )

        event_id = self._operation_id()
        before = group.snapshot()
        for shard_key in group.expected_shard_keys:
            if shard_key in group.adopted_shards:
                continue
            for transfer_id in group.shard_transfers.get(shard_key, ()):
                transfer = self.transfers[transfer_id]
                transfer_before = transfer.snapshot()
                if transfer.recovery_capacity_hold_active:
                    self._release_recovery_capacity_hold(
                        transfer,
                        event_id=event_id,
                        reason="partial_recovery_group_abandoned",
                    )
                transfer.recovery_discarded = True
                self._append_journal(
                    event_id=event_id,
                    kind="recovery_transfer_discarded_after_partial_adoption",
                    entity=transfer.transfer_id,
                    before=transfer_before,
                    after=transfer.snapshot(),
                    metadata={
                        "group_id": group.group_id,
                        "shard_key": shard_key,
                        "reason": normalized_reason,
                    },
                )
                self._record_transfer(
                    transfer,
                    event_id,
                    "recovery_transfer_discarded_after_partial_adoption",
                )
        group.discarded = True
        group.discard_reason = normalized_reason
        group.discarded_at_ns = self.now_ns
        self._append_journal(
            event_id=event_id,
            kind="partially_adopted_recovery_group_abandoned",
            entity=group.group_id,
            before=before,
            after=group.snapshot(),
            metadata={"reason": normalized_reason},
        )
        self._record_recovery_group(
            group, event_id, "partially_adopted_recovery_group_abandoned"
        )
        return group

    def place_extent(
        self,
        extent_id: str,
        location: KvLocation,
        *,
        state: KvCopyState = KvCopyState.COMMITTED,
        reason: str = "extent_placed",
        event_id: str | None = None,
    ) -> None:
        extent = self._require_extent(extent_id)
        self._validate_location(location, require_healthy=True)
        parsed_state = KvCopyState(state)
        if parsed_state in {KvCopyState.ABORTED, KvCopyState.LOST}:
            raise ValueError("a newly placed extent must be PENDING or COMMITTED")
        existing_state = extent.locations.get(location)
        if existing_state in {KvCopyState.PENDING, KvCopyState.COMMITTED}:
            raise ValueError(
                f"extent {extent.extent_id!r} already has an active location "
                f"{location.key}"
            )
        op_id = event_id or self._operation_id()
        before = extent.snapshot()
        self._adjust_location(
            location,
            parsed_state,
            extent.size_bytes,
            reason=reason,
            event_id=op_id,
        )
        extent.locations[location] = parsed_state
        extent.location_accounting[location] = LocationAccounting.INTERNAL
        extent.external_account_keys.pop(location, None)
        after = extent.snapshot()
        self._append_journal(
            event_id=op_id,
            kind="extent_location_placed",
            entity=extent.extent_id,
            before=before,
            after=after,
            metadata={"location": location.key, "state": parsed_state.value},
        )
        self._record_extent(extent, op_id, reason)

    def alias_external_source_location(
        self,
        extent_id: str,
        location: KvLocation,
        *,
        external_account_key: str | None = None,
        reason: str = "external_source_alias",
    ) -> None:
        """Attach a committed source alias without changing aggregate ledgers.

        ``OfflineResourceBridge`` may account scheduler-owned primary KV before
        it materializes per-migration extents.  This seam links an extent to
        those existing bytes instead of double-counting them.  External aliases
        are source-only and are supported for GPU primary KV and node HiCache
        primary KV.
        """

        extent = self._require_extent(extent_id)
        self._validate_location(location, require_healthy=True)
        if location.tier not in {
            StorageTier.GPU_PRIMARY,
            StorageTier.NODE_HICACHE_PRIMARY,
        }:
            raise ValueError(
                "external source aliases require GPU_PRIMARY or "
                "NODE_HICACHE_PRIMARY"
            )
        if extent.locations.get(location) in {
            KvCopyState.PENDING,
            KvCopyState.COMMITTED,
        }:
            raise ValueError(
                f"extent {extent.extent_id!r} already has an active location "
                f"{location.key}"
            )
        account_key = (
            f"extent:{extent.extent_id}"
            if external_account_key is None
            else str(external_account_key)
        )
        if not account_key:
            raise ValueError("external_account_key must be nonempty")
        shared_identity: tuple[Any, ...] | None = None
        shared_live = False
        for other in self.extents.values():
            if (
                location in other.locations
                and other.location_accounting.get(location)
                is LocationAccounting.EXTERNAL
                and other.external_account_keys.get(location) == account_key
            ):
                identity = self._external_source_identity(other)
                if shared_identity is None:
                    shared_identity = identity
                if identity != self._external_source_identity(extent):
                    raise ValueError(
                        "external_account_key size/identity conflict"
                    )
                if other.locations.get(location) in {
                    KvCopyState.PENDING,
                    KvCopyState.COMMITTED,
                }:
                    shared_live = True

        observed = self._external_aggregate_observed_bytes(location)
        internal_live = self._internal_live_bytes(location)
        external_required = self._external_live_bytes(location) + (
            0 if shared_live else extent.size_bytes
        )
        aggregate_available = observed - internal_live
        if aggregate_available < external_required:
            raise ValueError(
                f"external live bytes {external_required} exceed aggregate ledger "
                f"bytes {max(aggregate_available, 0)} at {location.key}"
            )

        event_id = self._operation_id()
        before = extent.snapshot()
        extent.locations[location] = KvCopyState.COMMITTED
        extent.location_accounting[location] = LocationAccounting.EXTERNAL
        extent.external_account_keys[location] = account_key
        self._append_journal(
            event_id=event_id,
            kind="external_source_location_aliased",
            entity=extent.extent_id,
            before=before,
            after=extent.snapshot(),
            metadata={
                "location": location.key,
                "accounting": LocationAccounting.EXTERNAL.value,
                "external_account_key": account_key,
                "aggregate_observed_bytes": observed,
                "internal_live_bytes": internal_live,
                "aggregate_available_bytes": aggregate_available,
                "external_live_bytes": external_required,
                "reason": str(reason),
            },
        )
        self._record_extent(extent, event_id, reason)

    def register_external_source_location(
        self,
        extent_id: str,
        location: KvLocation,
        *,
        external_account_key: str | None = None,
        reason: str = "external_source_alias",
    ) -> None:
        """Compatibility spelling for :meth:`alias_external_source_location`."""

        self.alias_external_source_location(
            extent_id,
            location,
            external_account_key=external_account_key,
            reason=reason,
        )

    def release_extent_location(
        self,
        extent_id: str,
        location: KvLocation,
        *,
        reason: str = "extent_released",
    ) -> KvCopyState:
        """Release one physical copy and its accounted capacity.

        Any queued or in-flight transfer using the location is aborted first,
        so a later completion event cannot resurrect a released copy.
        """

        extent = self._require_extent(extent_id)
        if location not in extent.locations:
            raise KeyError(
                f"extent {extent.extent_id!r} has no location {location.key}"
            )
        event_id = self._operation_id()
        active_states = {
            TransferState.QUEUED,
            TransferState.IN_FLIGHT,
            TransferState.ACKED,
        }
        for transfer in tuple(self.transfers.values()):
            if (
                transfer.extent_id == extent.extent_id
                and transfer.state in active_states
                and location in {transfer.source, transfer.destination}
            ):
                self.abort_transfer(
                    transfer.transfer_id,
                    reason=f"{reason}:location_released",
                    event_id=event_id,
                )
        state = extent.locations[location]
        accounting = extent.location_accounting.get(
            location, LocationAccounting.INTERNAL
        )
        before = extent.snapshot()
        if (
            state in {KvCopyState.PENDING, KvCopyState.COMMITTED}
            and accounting is LocationAccounting.INTERNAL
        ):
            self._remove_location_accounting(
                location,
                state,
                extent.size_bytes,
                event_id=event_id,
                reason=reason,
            )
        del extent.locations[location]
        extent.location_accounting.pop(location, None)
        extent.external_account_keys.pop(location, None)
        self._append_journal(
            event_id=event_id,
            kind="extent_location_released",
            entity=extent.extent_id,
            before=before,
            after=extent.snapshot(),
            metadata={
                "location": location.key,
                "old_state": state.value,
                "accounting": accounting.value,
            },
        )
        self._record_extent(extent, event_id, reason)
        return state

    def remove_extent_location(
        self,
        extent_id: str,
        location: KvLocation,
        *,
        reason: str = "extent_removed",
    ) -> KvCopyState:
        """Compatibility spelling for :meth:`release_extent_location`."""

        return self.release_extent_location(extent_id, location, reason=reason)

    def reserve_transfer(
        self,
        *,
        extent_id: str,
        source: KvLocation,
        destination: KvLocation,
        resources: Sequence[TransferResourceId],
        duration_ns: int,
        ready_ns: int | None = None,
        remove_source_on_complete: bool = False,
        group_id: str | None = None,
        shard_key: str | None = None,
        destination_mode: TransferDestinationMode | str = (
            TransferDestinationMode.MATERIALIZED_COPY
        ),
        transfer_size_bytes: int | None = None,
        token_start: int | None = None,
        token_end: int | None = None,
        recovery_group_id: str | None = None,
        recovery_shard_key: str | None = None,
    ) -> TransferRecord:
        """Reserve all resources together and create one transfer lifecycle.

        The returned interval is ``start=max(ready, busy_until[resource...])``.
        All listed serial resources become busy through ``end``.  This is a
        deterministic first-order contention model; a later network backend
        can replace duration calculation without changing the event contract.

        The default destination mode is fully backward compatible and creates
        an extent-owned PENDING location.  Recovery callers may instead bind a
        byte/token subspan to an external scheduler-owned receive sink or to a
        zero-copy same-GPU ownership activation.
        """

        extent = self._require_extent(extent_id)
        mode = TransferDestinationMode(destination_mode)
        transfer_size, transfer_token_start, transfer_token_end = (
            self._resolve_transfer_span(
                extent,
                transfer_size_bytes=transfer_size_bytes,
                token_start=token_start,
                token_end=token_end,
            )
        )
        self._validate_location(source, require_healthy=True)
        self._validate_location(destination, require_healthy=True)
        if extent.locations.get(source) is not KvCopyState.COMMITTED:
            raise ValueError("transfer source must hold a COMMITTED extent copy")
        if extent.locations.get(destination) in {
            KvCopyState.PENDING,
            KvCopyState.COMMITTED,
        }:
            raise ValueError("transfer destination already has an active extent copy")
        self._validate_transfer_path(source, destination)
        self._validate_destination_mode(
            mode,
            extent=extent,
            source=source,
            destination=destination,
            transfer_size_bytes=transfer_size,
            resources=resources,
            duration_ns=duration_ns,
            remove_source_on_complete=remove_source_on_complete,
        )
        if (group_id is None) != (shard_key is None):
            raise ValueError("group_id and shard_key must be provided together")
        if (recovery_group_id is None) != (recovery_shard_key is None):
            raise ValueError(
                "recovery_group_id and recovery_shard_key must be provided together"
            )
        if group_id is not None and recovery_group_id is not None:
            raise ValueError("a transfer cannot bind both commit-group kinds")
        group: TierCommitGroup | None = None
        group_before: dict[str, Any] | None = None
        parsed_shard_key: str | None = None
        if group_id is not None:
            group = self._require_group(group_id)
            parsed_shard_key = str(shard_key)
            self._validate_group_binding(
                group,
                extent=extent,
                destination=destination,
                shard_key=parsed_shard_key,
            )
            group_before = group.snapshot()
        recovery_group: RecoveryLoadGroup | None = None
        recovery_group_before: dict[str, Any] | None = None
        parsed_recovery_shard_key: str | None = None
        if recovery_group_id is not None:
            recovery_group = self._require_recovery_group(recovery_group_id)
            parsed_recovery_shard_key = str(recovery_shard_key)
            self._validate_recovery_group_binding(
                recovery_group,
                extent=extent,
                destination_mode=mode,
                shard_key=parsed_recovery_shard_key,
            )
            recovery_group_before = recovery_group.snapshot()

        duration = _nonnegative_int("duration_ns", duration_ns)
        ready = self.now_ns if ready_ns is None else _nonnegative_int(
            "ready_ns", ready_ns
        )
        if ready < self.now_ns:
            raise ValueError("transfer ready_ns cannot precede current DES time")
        resource_tuple = tuple(resources)
        if (
            mode is not TransferDestinationMode.OWNERSHIP_ACTIVATION
            and not resource_tuple
        ):
            raise ValueError("a transfer must reserve at least one resource")
        if len(set(resource_tuple)) != len(resource_tuple):
            raise ValueError("transfer resources must be unique")
        for resource in resource_tuple:
            if not isinstance(resource, TransferResourceId):
                raise TypeError("resources must contain TransferResourceId values")

        start = (
            ready
            if not resource_tuple
            else max(
                ready,
                *(
                    self.resource_busy_until_ns.get(resource, 0)
                    for resource in resource_tuple
                ),
            )
        )
        end = start + duration
        transfer_id = f"transfer-{self._transfer_sequence:012d}"
        self._transfer_sequence += 1
        op_id = self._operation_id()

        # Capacity admission happens before changing any resource calendar.
        capacity_hold_active = False
        if mode is TransferDestinationMode.MATERIALIZED_COPY:
            self._ensure_location_capacity(
                destination, KvCopyState.PENDING, transfer_size
            )
            self.place_extent(
                extent.extent_id,
                destination,
                state=KvCopyState.PENDING,
                reason="transfer_reserved",
                event_id=op_id,
            )
        elif mode is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK:
            assert destination.gpu is not None
            self.adjust_gpu_hbm(
                destination.gpu,
                HbmAccount.RECOVERY_PENDING,
                transfer_size,
                reason="external_recovery_sink_reserved",
                event_id=op_id,
            )
            capacity_hold_active = True

        for resource in resource_tuple:
            before_busy = self.resource_busy_until_ns.get(resource, 0)
            self.resource_busy_until_ns[resource] = end
            self._append_journal(
                event_id=op_id,
                kind="transfer_resource_reserved",
                entity=resource.key,
                before={"busy_until_ns": before_busy},
                after={"busy_until_ns": end},
                metadata={
                    "transfer_id": transfer_id,
                    "ready_ns": ready,
                    "start_ns": start,
                    "end_ns": end,
                },
            )
            self._resource_timeseries.append(
                {
                    "timestamp_ns": self.now_ns,
                    "event_id": op_id,
                    "resource": resource.key,
                    "busy_until_ns": end,
                    "transfer_id": transfer_id,
                }
            )

        start_event = self.schedule_at(
            start,
            kind="transfer_start",
            priority=EventPriority.TRANSFER_START,
            payload={"transfer_id": transfer_id},
            callback=lambda simulator, _event: simulator._start_transfer(transfer_id),
        )
        completion = self.schedule_at(
            end,
            kind="transfer_completion",
            priority=EventPriority.TRANSFER_COMPLETION,
            payload={"transfer_id": transfer_id},
            callback=lambda simulator, _event: simulator._complete_transfer(transfer_id),
        )
        record = TransferRecord(
            transfer_id=transfer_id,
            extent_id=extent.extent_id,
            source=source,
            destination=destination,
            size_bytes=transfer_size,
            ready_ns=ready,
            start_ns=start,
            end_ns=end,
            resources=resource_tuple,
            start_event_id=start_event.event_id,
            completion_event_id=completion.event_id,
            remove_source_on_complete=bool(remove_source_on_complete),
            group_id=None if group is None else group.group_id,
            shard_key=parsed_shard_key,
            destination_mode=mode,
            token_start=transfer_token_start,
            token_end=transfer_token_end,
            recovery_group_id=(
                None if recovery_group is None else recovery_group.group_id
            ),
            recovery_shard_key=parsed_recovery_shard_key,
            recovery_capacity_hold_active=capacity_hold_active,
        )
        self.transfers[transfer_id] = record
        if group is not None:
            assert parsed_shard_key is not None and group_before is not None
            group.shard_transfers[parsed_shard_key] = transfer_id
            self._append_journal(
                event_id=op_id,
                kind="commit_group_shard_bound",
                entity=group.group_id,
                before=group_before,
                after=group.snapshot(),
                metadata={
                    "shard_key": parsed_shard_key,
                    "transfer_id": transfer_id,
                },
            )
            self._record_group(group, op_id, "commit_group_shard_bound")
        if recovery_group is not None:
            assert (
                parsed_recovery_shard_key is not None
                and recovery_group_before is not None
            )
            recovery_group.shard_transfers.setdefault(
                parsed_recovery_shard_key, []
            ).append(transfer_id)
            self._append_journal(
                event_id=op_id,
                kind="recovery_load_group_transfer_bound",
                entity=recovery_group.group_id,
                before=recovery_group_before,
                after=recovery_group.snapshot(),
                metadata={
                    "shard_key": parsed_recovery_shard_key,
                    "transfer_id": transfer_id,
                },
            )
            self._record_recovery_group(
                recovery_group, op_id, "recovery_load_group_transfer_bound"
            )
        self._append_journal(
            event_id=op_id,
            kind="transfer_reserved",
            entity=transfer_id,
            before={},
            after=record.snapshot(),
        )
        self._record_transfer(record, op_id, "transfer_reserved")
        return record

    def reserve_resources(
        self,
        resources: Sequence[TransferResourceId],
        *,
        duration_ns: int,
        ready_ns: int | None = None,
    ) -> ResourceReservation:
        """Reserve a multi-resource interval without creating a KV transfer."""

        resource_tuple = tuple(resources)
        if not resource_tuple or len(set(resource_tuple)) != len(resource_tuple):
            raise ValueError("resources must be a nonempty unique sequence")
        if any(not isinstance(resource, TransferResourceId) for resource in resource_tuple):
            raise TypeError("resources must contain TransferResourceId values")
        ready = self.now_ns if ready_ns is None else _nonnegative_int(
            "ready_ns", ready_ns
        )
        if ready < self.now_ns:
            raise ValueError("ready_ns cannot precede current DES time")
        duration = _nonnegative_int("duration_ns", duration_ns)
        start = max(
            ready,
            *(self.resource_busy_until_ns.get(resource, 0) for resource in resource_tuple),
        )
        end = start + duration
        event_id = self._operation_id()
        for resource in resource_tuple:
            before = self.resource_busy_until_ns.get(resource, 0)
            self.resource_busy_until_ns[resource] = end
            self._append_journal(
                event_id=event_id,
                kind="resource_reserved",
                entity=resource.key,
                before={"busy_until_ns": before},
                after={"busy_until_ns": end},
                metadata={"ready_ns": ready, "start_ns": start, "end_ns": end},
            )
            self._resource_timeseries.append(
                {
                    "timestamp_ns": self.now_ns,
                    "event_id": event_id,
                    "resource": resource.key,
                    "busy_until_ns": end,
                }
            )
        return ResourceReservation(ready, start, end, resource_tuple)

    def abort_transfer(
        self,
        transfer_id: str,
        *,
        reason: str = "cancelled",
        event_id: str | None = None,
    ) -> TransferRecord:
        """Abort an active transfer and release its destination reservation.

        The method is idempotent for an already-aborted transfer.  Resource
        calendars remain reserved because later reservations may already have
        been placed behind this interval; only byte capacity is released.
        """

        transfer = self.transfers[str(transfer_id)]
        if transfer.recovery_group_id is not None:
            recovery_group = self._require_recovery_group(
                transfer.recovery_group_id
            )
            if recovery_group.state is CommitGroupState.PENDING:
                self.abort_recovery_load_group(
                    recovery_group.group_id,
                    reason=(
                        f"shard {transfer.recovery_shard_key}: {reason}"
                    ),
                    event_id=event_id,
                )
                return transfer
        if transfer.group_id is not None:
            group = self._require_group(transfer.group_id)
            if group.state is CommitGroupState.PENDING:
                self.abort_commit_group(
                    group.group_id,
                    reason=f"shard {transfer.shard_key}: {reason}",
                    event_id=event_id,
                )
                return transfer
        return self._abort_transfer_only(
            transfer,
            reason=reason,
            event_id=event_id or self._operation_id(),
        )

    def _abort_transfer_only(
        self,
        transfer: TransferRecord,
        *,
        reason: str,
        event_id: str,
    ) -> TransferRecord:
        if transfer.state is TransferState.ABORTED:
            return transfer
        if transfer.state is TransferState.COMMITTED:
            raise ValueError(f"cannot abort committed transfer {transfer.transfer_id}")
        extent = self._require_extent(transfer.extent_id)
        destination_state = extent.locations.get(transfer.destination)
        if (
            transfer.destination_mode
            is TransferDestinationMode.MATERIALIZED_COPY
            and destination_state is KvCopyState.PENDING
        ):
            extent_before = extent.snapshot()
            if extent.location_accounting.get(
                transfer.destination, LocationAccounting.INTERNAL
            ) is LocationAccounting.INTERNAL:
                self._remove_location_accounting(
                    transfer.destination,
                    KvCopyState.PENDING,
                    transfer.size_bytes,
                    event_id=event_id,
                    reason="transfer_aborted",
                )
            extent.locations[transfer.destination] = KvCopyState.ABORTED
            self._append_journal(
                event_id=event_id,
                kind="extent_transfer_aborted",
                entity=extent.extent_id,
                before=extent_before,
                after=extent.snapshot(),
                metadata={
                    "transfer_id": transfer.transfer_id,
                    "reason": str(reason),
                },
            )
            self._record_extent(extent, event_id, "transfer_aborted")
        if transfer.recovery_capacity_hold_active:
            self._release_recovery_capacity_hold(
                transfer,
                event_id=event_id,
                reason="transfer_aborted",
            )

        before = transfer.snapshot()
        transfer.state = TransferState.ABORTED
        self._append_journal(
            event_id=event_id,
            kind="transfer_aborted",
            entity=transfer.transfer_id,
            before=before,
            after=transfer.snapshot(),
            metadata={"reason": str(reason)},
        )
        self._record_transfer(transfer, event_id, f"transfer_aborted:{reason}")
        return transfer

    def abort_commit_group(
        self,
        group_id: str,
        *,
        reason: str = "cancelled",
        event_id: str | None = None,
    ) -> TierCommitGroup:
        """Abort one tier-local group without touching independent tiers."""

        group = self._require_group(group_id)
        if group.state is CommitGroupState.ABORTED:
            return group
        if group.state is CommitGroupState.COMMITTED:
            raise ValueError(f"cannot abort committed group {group.group_id}")
        op_id = event_id or self._operation_id()
        before = group.snapshot()
        for shard_key in group.expected_shard_keys:
            transfer_id = group.shard_transfers.get(shard_key)
            if transfer_id is None:
                continue
            transfer = self.transfers[transfer_id]
            if transfer.state in {
                TransferState.QUEUED,
                TransferState.IN_FLIGHT,
                TransferState.ACKED,
            }:
                self._abort_transfer_only(
                    transfer,
                    reason=f"group_aborted:{reason}",
                    event_id=op_id,
                )
        group.state = CommitGroupState.ABORTED
        self._append_journal(
            event_id=op_id,
            kind="commit_group_aborted",
            entity=group.group_id,
            before=before,
            after=group.snapshot(),
            metadata={"reason": str(reason)},
        )
        self._record_group(group, op_id, f"commit_group_aborted:{reason}")
        return group

    def abort_recovery_load_group(
        self,
        group_id: str,
        *,
        reason: str = "cancelled",
        event_id: str | None = None,
    ) -> RecoveryLoadGroup:
        """Abort every active transfer in one PP/TP recovery boundary."""

        group = self._require_recovery_group(group_id)
        if group.state is CommitGroupState.ABORTED:
            return group
        if group.state is CommitGroupState.COMMITTED:
            raise ValueError(f"cannot abort committed recovery group {group.group_id}")
        op_id = event_id or self._operation_id()
        before = group.snapshot()
        for transfer_id in group.transfer_ids:
            transfer = self.transfers[transfer_id]
            if transfer.state in {
                TransferState.QUEUED,
                TransferState.IN_FLIGHT,
                TransferState.ACKED,
            }:
                self._abort_transfer_only(
                    transfer,
                    reason=f"recovery_group_aborted:{reason}",
                    event_id=op_id,
                )
        group.state = CommitGroupState.ABORTED
        self._append_journal(
            event_id=op_id,
            kind="recovery_load_group_aborted",
            entity=group.group_id,
            before=before,
            after=group.snapshot(),
            metadata={"reason": str(reason)},
        )
        self._record_recovery_group(
            group, op_id, f"recovery_load_group_aborted:{reason}"
        )
        return group

    def fail_gpu(self, gpu: PhysicalGpuId) -> tuple[str, ...]:
        """Fail one GPU while leaving every node DRAM ledger untouched."""

        return self._fail_gpu(gpu, event_id=self._operation_id())

    def _fail_gpu(
        self, gpu: PhysicalGpuId, *, event_id: str
    ) -> tuple[str, ...]:
        ledger = self._require_gpu(gpu)
        if not ledger.healthy:
            return ()
        before = ledger.snapshot()
        for account in HbmAccount:
            setattr(ledger, account.value, 0)
        ledger.healthy = False
        self._append_journal(
            event_id=event_id,
            kind="gpu_failed",
            entity=gpu.key,
            before=before,
            after=ledger.snapshot(),
        )
        self._record_gpu(gpu, event_id, "gpu_failed")

        affected: list[str] = []
        for extent in self.extents.values():
            for location in tuple(extent.locations):
                old_state = extent.locations[location]
                if location.gpu != gpu or old_state not in {
                    KvCopyState.PENDING,
                    KvCopyState.COMMITTED,
                }:
                    continue
                extent_before = extent.snapshot()
                extent.locations[location] = (
                    KvCopyState.ABORTED
                    if old_state is KvCopyState.PENDING
                    else KvCopyState.LOST
                )
                affected.append(extent.extent_id)
                self._append_journal(
                    event_id=event_id,
                    kind="extent_copy_lost",
                    entity=extent.extent_id,
                    before=extent_before,
                    after=extent.snapshot(),
                    metadata={"location": location.key, "old_state": old_state.value},
                )
                self._record_extent(extent, event_id, "gpu_failed")

        for transfer in tuple(self.transfers.values()):
            if transfer.state not in {
                TransferState.QUEUED,
                TransferState.IN_FLIGHT,
                TransferState.ACKED,
            }:
                continue
            if transfer.source.gpu == gpu or transfer.destination.gpu == gpu:
                endpoint = "source" if transfer.source.gpu == gpu else "destination"
                self.abort_transfer(
                    transfer.transfer_id,
                    reason=f"{endpoint}_gpu_failed",
                    event_id=event_id,
                )
        for transfer in self.transfers.values():
            if (
                transfer.state is TransferState.COMMITTED
                and transfer.destination_mode
                is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                and transfer.destination.gpu == gpu
                and transfer.recovery_capacity_hold_active
            ):
                before_transfer = transfer.snapshot()
                transfer.recovery_capacity_hold_active = False
                transfer.recovery_capacity_hold_lost = True
                self._append_journal(
                    event_id=event_id,
                    kind="recovery_capacity_hold_lost",
                    entity=transfer.transfer_id,
                    before=before_transfer,
                    after=transfer.snapshot(),
                    metadata={"gpu": gpu.key},
                )
                self._record_transfer(
                    transfer, event_id, "recovery_capacity_hold_lost"
                )
        return tuple(sorted(set(affected)))

    def repair_gpu(
        self,
        gpu: PhysicalGpuId,
        *,
        weights_bytes: int = 0,
        runtime_bytes: int = 0,
    ) -> GpuHbmLedger:
        """Make a failed GPU allocatable again after model/runtime reload."""

        ledger = self._require_gpu(gpu)
        if ledger.healthy:
            raise ValueError(f"GPU {gpu.key} is already healthy")
        weights = _nonnegative_int("weights_bytes", weights_bytes)
        runtime = _nonnegative_int("runtime_bytes", runtime_bytes)
        if weights + runtime > ledger.capacity_bytes:
            raise ValueError("repaired GPU baseline exceeds HBM capacity")
        event_id = self._operation_id()
        before = ledger.snapshot()
        for account in HbmAccount:
            setattr(ledger, account.value, 0)
        ledger.weights = weights
        ledger.runtime = runtime
        ledger.healthy = True
        ledger.validate()
        self._append_journal(
            event_id=event_id,
            kind="gpu_repaired",
            entity=gpu.key,
            before=before,
            after=ledger.snapshot(),
        )
        self._record_gpu(gpu, event_id, "gpu_repaired")
        return ledger

    def fail_node(self, node_id: str) -> tuple[str, ...]:
        """Fail all GPUs and the DRAM pool of exactly one physical node."""

        node = str(node_id)
        dram = self._require_dram(node)
        event_id = self._operation_id()
        affected: list[str] = []
        for gpu in sorted(self.gpu_hbm):
            if gpu.node_id == node:
                affected.extend(self._fail_gpu(gpu, event_id=event_id))

        if dram.healthy:
            before = dram.snapshot()
            for account in DramAccount:
                setattr(dram, account.value, 0)
            dram.healthy = False
            self._append_journal(
                event_id=event_id,
                kind="node_dram_failed",
                entity=node,
                before=before,
                after=dram.snapshot(),
            )
            self._record_dram(node, event_id, "node_dram_failed")

            for extent in self.extents.values():
                for location in tuple(extent.locations):
                    old_state = extent.locations[location]
                    if location.node_id != node or old_state not in {
                        KvCopyState.PENDING,
                        KvCopyState.COMMITTED,
                    }:
                        continue
                    extent_before = extent.snapshot()
                    extent.locations[location] = (
                        KvCopyState.ABORTED
                        if old_state is KvCopyState.PENDING
                        else KvCopyState.LOST
                    )
                    affected.append(extent.extent_id)
                    self._append_journal(
                        event_id=event_id,
                        kind="extent_copy_lost",
                        entity=extent.extent_id,
                        before=extent_before,
                        after=extent.snapshot(),
                        metadata={
                            "location": location.key,
                            "old_state": old_state.value,
                        },
                    )
                    self._record_extent(extent, event_id, "node_dram_failed")

        for transfer in tuple(self.transfers.values()):
            if transfer.state not in {
                TransferState.QUEUED,
                TransferState.IN_FLIGHT,
                TransferState.ACKED,
            }:
                continue
            source_node = self._location_node(transfer.source)
            destination_node = self._location_node(transfer.destination)
            if node in {source_node, destination_node}:
                endpoint = "source" if source_node == node else "destination"
                self.abort_transfer(
                    transfer.transfer_id,
                    reason=f"{endpoint}_node_failed",
                    event_id=event_id,
                )
        return tuple(sorted(set(affected)))

    def repair_node_dram(
        self,
        node_id: str,
        *,
        runtime_bytes: int = 0,
        hicache_primary_bytes: int = 0,
    ) -> NodeDramLedger:
        """Make one failed node DRAM pool available with an empty Aegis tier."""

        node = str(node_id)
        ledger = self._require_dram(node)
        if ledger.healthy:
            raise ValueError(f"DRAM node {node!r} is already healthy")
        runtime = _nonnegative_int("runtime_bytes", runtime_bytes)
        hicache = _nonnegative_int(
            "hicache_primary_bytes", hicache_primary_bytes
        )
        if runtime + hicache > ledger.capacity_bytes:
            raise ValueError("repaired DRAM baseline exceeds capacity")
        event_id = self._operation_id()
        before = ledger.snapshot()
        for account in DramAccount:
            setattr(ledger, account.value, 0)
        ledger.runtime = runtime
        ledger.hicache_primary = hicache
        ledger.healthy = True
        ledger.validate()
        self._append_journal(
            event_id=event_id,
            kind="node_dram_repaired",
            entity=node,
            before=before,
            after=ledger.snapshot(),
        )
        self._record_dram(node, event_id, "node_dram_repaired")
        return ledger

    def snapshot(self) -> dict[str, Any]:
        """Return one JSON-compatible full-state snapshot."""

        return {
            "now_ns": self.now_ns,
            "queued_events": len(self._events),
            "gpus": {
                gpu.key: {**gpu.as_dict(), **ledger.snapshot()}
                for gpu, ledger in sorted(self.gpu_hbm.items())
            },
            "node_dram": {
                node: ledger.snapshot()
                for node, ledger in sorted(self.node_dram.items())
            },
            "extents": {
                extent_id: extent.snapshot()
                for extent_id, extent in sorted(self.extents.items())
            },
            "resources": {
                resource.key: busy_until
                for resource, busy_until in sorted(self.resource_busy_until_ns.items())
            },
            "transfers": {
                transfer_id: transfer.snapshot()
                for transfer_id, transfer in sorted(self.transfers.items())
            },
            "commit_groups": {
                group_id: group.snapshot()
                for group_id, group in sorted(self.commit_groups.items())
            },
            "recovery_load_groups": {
                group_id: group.snapshot()
                for group_id, group in sorted(self.recovery_load_groups.items())
            },
        }

    def export_timeseries(self) -> dict[str, list[dict[str, Any]]]:
        """Return defensive in-memory tables suitable for JSONL/Parquet writers."""

        return copy.deepcopy(
            {
                "gpu_hbm": self._gpu_timeseries,
                "node_dram": self._dram_timeseries,
                "kv_extents": self._extent_timeseries,
                "transfers": self._transfer_timeseries,
                "commit_groups": self._group_timeseries,
                "recovery_load_groups": self._recovery_group_timeseries,
                "resources": self._resource_timeseries,
                "journal": [entry.as_dict() for entry in self._journal],
            }
        )

    def audit_invariants(
        self,
        *,
        raise_on_error: bool = False,
        strict_extent_accounting: bool = False,
        extent_accounting_mode: ExtentAccountingAuditMode | str | None = None,
    ) -> list[str]:
        """Audit structural invariants and optionally exact KV conservation.

        Strict extent accounting is opt-in because scheduler integration may
        initially feed aggregate primary/prefix usage before it emits one
        :class:`KvExtent` per allocation.
        """

        if extent_accounting_mode is None:
            accounting_mode = (
                ExtentAccountingAuditMode.STRICT
                if strict_extent_accounting
                else None
            )
        else:
            accounting_mode = ExtentAccountingAuditMode(extent_accounting_mode)
            if (
                strict_extent_accounting
                and accounting_mode is not ExtentAccountingAuditMode.STRICT
            ):
                raise ValueError(
                    "strict_extent_accounting conflicts with "
                    "extent_accounting_mode"
                )

        errors: list[str] = []
        if self.now_ns < 0:
            errors.append("global clock is negative")
        queued_event_ids = [event.event_id for event in self._events]
        queued_event_id_set = set(queued_event_ids)
        if len(queued_event_ids) != len(queued_event_id_set):
            errors.append("event heap contains duplicate event IDs")
        for event in self._events:
            if event.timestamp_ns < self.now_ns:
                errors.append(f"queued event {event.event_id} precedes global clock")
            if self._events_by_id.get(event.event_id) != event:
                errors.append(f"queued event {event.event_id} lacks canonical record")
            if self._event_states.get(event.event_id) != "queued":
                errors.append(
                    f"queued event {event.event_id} has lifecycle state "
                    f"{self._event_states.get(event.event_id)!r}"
                )
        cancellation_evidence = {
            entry.event_id
            for entry in self._journal
            if entry.kind == "event_cancelled"
        }
        for event_id, event in self._events_by_id.items():
            state = self._event_states.get(event_id)
            if state not in {
                "queued",
                "dispatching",
                "completed",
                "error",
                "cancelled",
            }:
                errors.append(f"event {event_id} has invalid lifecycle state {state!r}")
                continue
            if state == "queued" and event_id not in queued_event_id_set:
                errors.append(f"queued event {event_id} is absent from event heap")
            if state != "queued" and event_id in queued_event_id_set:
                errors.append(
                    f"non-queued event {event_id} remains in event heap"
                )
            if state == "dispatching" and event_id != self._current_event_id:
                errors.append(f"event {event_id} is stranded in dispatching state")
            if state == "cancelled" and event_id not in cancellation_evidence:
                errors.append(f"cancelled event {event_id} lacks journal evidence")
            if state != "cancelled" and event_id in cancellation_evidence:
                errors.append(
                    f"non-cancelled event {event_id} has cancellation evidence"
                )
            if event.event_id != event_id:
                errors.append(f"event registry key mismatch for {event_id}")
        for gpu, ledger in self.gpu_hbm.items():
            try:
                ledger.validate()
            except ValueError as exc:
                errors.append(f"GPU {gpu.key}: {exc}")
            if not ledger.healthy and ledger.used_bytes:
                errors.append(f"failed GPU {gpu.key} retains volatile HBM usage")
        for node, ledger in self.node_dram.items():
            try:
                ledger.validate()
            except ValueError as exc:
                errors.append(f"DRAM {node}: {exc}")
            if not ledger.healthy and ledger.used_bytes:
                errors.append(f"failed DRAM node {node} retains volatile usage")
        external_identities: dict[
            tuple[KvLocation, str], tuple[Any, ...]
        ] = {}
        for extent in self.extents.values():
            for location, state in extent.locations.items():
                try:
                    self._validate_location(location)
                    KvCopyState(state)
                    accounting = LocationAccounting(
                        extent.location_accounting[location]
                    )
                    if (
                        accounting is LocationAccounting.EXTERNAL
                        and location.tier
                        not in {
                            StorageTier.GPU_PRIMARY,
                            StorageTier.NODE_HICACHE_PRIMARY,
                        }
                    ):
                        raise ValueError(
                            "external accounting is only valid for primary sources"
                        )
                    if (
                        accounting is LocationAccounting.EXTERNAL
                        and state is KvCopyState.PENDING
                    ):
                        raise ValueError("external source alias cannot be PENDING")
                    if accounting is LocationAccounting.EXTERNAL:
                        account_key = extent.external_account_keys[location]
                        if not account_key:
                            raise ValueError("external_account_key is empty")
                        identity_key = (location, account_key)
                        identity = self._external_source_identity(extent)
                        previous = external_identities.setdefault(
                            identity_key, identity
                        )
                        if previous != identity:
                            raise ValueError(
                                "external_account_key size/identity conflict"
                            )
                    elif location in extent.external_account_keys:
                        raise ValueError(
                            "internal location has external_account_key"
                        )
                except KeyError:
                    errors.append(
                        f"extent {extent.extent_id}: location has no accounting owner"
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(f"extent {extent.extent_id}: {exc}")
            extra_accounting = set(extent.location_accounting) - set(extent.locations)
            if extra_accounting:
                errors.append(
                    f"extent {extent.extent_id}: accounting has stale locations"
                )
            extra_external_keys = set(extent.external_account_keys) - set(
                extent.locations
            )
            if extra_external_keys:
                errors.append(
                    f"extent {extent.extent_id}: external keys have stale locations"
                )
        for transfer in self.transfers.values():
            if not transfer.ready_ns <= transfer.start_ns <= transfer.end_ns:
                errors.append(f"transfer {transfer.transfer_id} has invalid time order")
            for resource in transfer.resources:
                if self.resource_busy_until_ns.get(resource, -1) < transfer.end_ns:
                    errors.append(
                        f"resource {resource.key} ends before {transfer.transfer_id}"
                    )
            extent = self.extents.get(transfer.extent_id)
            if extent is None:
                errors.append(f"transfer {transfer.transfer_id} has no extent")
                continue
            destination_state = extent.locations.get(transfer.destination)
            active = transfer.state in {
                TransferState.QUEUED,
                TransferState.IN_FLIGHT,
                TransferState.ACKED,
            }
            if (
                transfer.destination_mode
                is TransferDestinationMode.MATERIALIZED_COPY
            ):
                if active and destination_state is not KvCopyState.PENDING:
                    errors.append(
                        f"active transfer {transfer.transfer_id} has no pending destination"
                    )
                if (
                    transfer.state is TransferState.COMMITTED
                    and destination_state
                    not in {KvCopyState.COMMITTED, KvCopyState.LOST, None}
                ):
                    errors.append(
                        f"committed transfer {transfer.transfer_id} has invalid destination state"
                    )
                if transfer.size_bytes != extent.size_bytes:
                    errors.append(
                        f"materialized transfer {transfer.transfer_id} is partial"
                    )
            else:
                if destination_state in {
                    KvCopyState.PENDING,
                    KvCopyState.COMMITTED,
                }:
                    errors.append(
                        f"external transfer {transfer.transfer_id} fabricated a destination"
                    )
                if (
                    transfer.destination_mode
                    is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                ):
                    if (
                        transfer.state in {
                            TransferState.QUEUED,
                            TransferState.IN_FLIGHT,
                            TransferState.ACKED,
                        }
                        and not transfer.recovery_capacity_hold_active
                    ):
                        errors.append(
                            f"external sink {transfer.transfer_id} lacks its capacity hold"
                        )
                    if (
                        transfer.state is TransferState.COMMITTED
                        and not transfer.recovery_capacity_hold_active
                        and not transfer.recovery_capacity_hold_lost
                        and not transfer.recovery_adopted
                        and not transfer.recovery_discarded
                    ):
                        errors.append(
                            f"committed external sink {transfer.transfer_id} has "
                            "no held/lost/adopted/discarded capacity state"
                        )
                    if (
                        transfer.recovery_capacity_hold_active
                        and transfer.recovery_capacity_hold_lost
                    ):
                        errors.append(
                            f"external sink {transfer.transfer_id} has a live and "
                            "lost capacity hold"
                        )
                    if (
                        transfer.recovery_capacity_hold_lost
                        and self._location_is_healthy(transfer.destination)
                    ):
                        errors.append(
                            f"external sink {transfer.transfer_id} marks capacity "
                            "lost on a healthy destination"
                        )
                elif transfer.recovery_capacity_hold_active:
                    errors.append(
                        f"ownership activation {transfer.transfer_id} has a capacity hold"
                    )
                terminal_recovery_states = sum(
                    bool(value)
                    for value in (
                        transfer.recovery_capacity_hold_active,
                        transfer.recovery_capacity_hold_lost,
                        transfer.recovery_adopted,
                        transfer.recovery_discarded,
                    )
                )
                if (
                    transfer.state is TransferState.COMMITTED
                    and transfer.destination_mode
                    is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                    and terminal_recovery_states != 1
                ):
                    errors.append(
                        f"committed recovery transfer {transfer.transfer_id} has "
                        f"{terminal_recovery_states} ownership states"
                    )
                if transfer.recovery_adopted and transfer.recovery_discarded:
                    errors.append(
                        f"recovery transfer {transfer.transfer_id} is adopted and discarded"
                    )
                if (
                    transfer.destination_mode
                    is TransferDestinationMode.OWNERSHIP_ACTIVATION
                    and (
                        transfer.resources
                        or transfer.start_ns != transfer.end_ns
                        or transfer.source.gpu != transfer.destination.gpu
                    )
                ):
                    errors.append(
                        f"ownership activation {transfer.transfer_id} is not zero-copy"
                    )
                if (
                    transfer.destination_mode
                    is TransferDestinationMode.OWNERSHIP_ACTIVATION
                    and transfer.recovery_adopted
                    and extent.locations.get(transfer.source)
                    in {KvCopyState.PENDING, KvCopyState.COMMITTED}
                ):
                    errors.append(
                        f"adopted activation {transfer.transfer_id} retained its source"
                    )
            if (
                transfer.state is TransferState.ACKED
                and transfer.group_id is None
                and transfer.recovery_group_id is None
            ):
                errors.append(f"ungrouped transfer {transfer.transfer_id} is ACKED")
            if (transfer.group_id is None) != (transfer.shard_key is None):
                errors.append(
                    f"transfer {transfer.transfer_id} has incomplete group binding"
                )
            if (transfer.recovery_group_id is None) != (
                transfer.recovery_shard_key is None
            ):
                errors.append(
                    f"transfer {transfer.transfer_id} has incomplete recovery binding"
                )
            if (
                transfer.group_id is not None
                and transfer.recovery_group_id is not None
            ):
                errors.append(
                    f"transfer {transfer.transfer_id} binds two commit groups"
                )
        for group in self.commit_groups.values():
            errors.extend(self._audit_commit_group(group))
        for group in self.recovery_load_groups.values():
            errors.extend(self._audit_recovery_load_group(group))
        if accounting_mode is not None:
            errors.extend(
                self._audit_extent_account_conservation(accounting_mode)
            )
        previous_timestamp = -1
        for index, entry in enumerate(self._journal):
            if not entry.event_id:
                errors.append(f"journal entry {index} has no event_id")
            if entry.before is None or entry.after is None:
                errors.append(f"journal entry {index} lacks before/after state")
            if entry.timestamp_ns < previous_timestamp:
                errors.append(f"journal entry {index} moves backwards in time")
            previous_timestamp = entry.timestamp_ns
        if errors and raise_on_error:
            raise AssertionError("; ".join(errors))
        return errors

    def _audit_commit_group(self, group: TierCommitGroup) -> list[str]:
        errors: list[str] = []
        expected = set(group.expected_shard_keys)
        bound = set(group.shard_transfers)
        if not bound.issubset(expected):
            errors.append(f"group {group.group_id} binds unexpected shard keys")
        if not group.acked_shards.issubset(expected):
            errors.append(f"group {group.group_id} ACKs unexpected shard keys")
        state_acked: set[str] = set()
        for shard_key, transfer_id in group.shard_transfers.items():
            transfer = self.transfers.get(transfer_id)
            if transfer is None:
                errors.append(
                    f"group {group.group_id} shard {shard_key} has no transfer"
                )
                continue
            if transfer.group_id != group.group_id or transfer.shard_key != shard_key:
                errors.append(
                    f"group {group.group_id} shard {shard_key} binding is inconsistent"
                )
            extent = self.extents.get(transfer.extent_id)
            if extent is None:
                continue
            identity = (
                extent.route_version,
                extent.model_revision,
                extent.kv_abi,
                extent.parallel_layout,
            )
            expected_identity = (
                group.route_version,
                group.model_revision,
                group.kv_abi,
                group.parallel_layout,
            )
            if (
                identity != expected_identity
                or transfer.destination.tier is not group.tier
                or extent.shard_count != len(group.expected_shard_keys)
            ):
                errors.append(
                    f"group {group.group_id} shard {shard_key} identity mismatch"
                )
            if transfer.state is TransferState.ACKED:
                state_acked.add(shard_key)

        if group.state is CommitGroupState.PENDING:
            if state_acked != group.acked_shards:
                errors.append(f"group {group.group_id} ACK set disagrees with transfers")
            if group.acked_shards == expected:
                errors.append(f"group {group.group_id} missed atomic final commit")
            for transfer_id in group.shard_transfers.values():
                if self.transfers[transfer_id].state in {
                    TransferState.COMMITTED,
                    TransferState.ABORTED,
                }:
                    errors.append(
                        f"pending group {group.group_id} has terminal shard transfer"
                    )
        elif group.state is CommitGroupState.COMMITTED:
            if bound != expected or group.acked_shards != expected:
                errors.append(f"committed group {group.group_id} is missing shards")
            for transfer_id in group.shard_transfers.values():
                if self.transfers[transfer_id].state is not TransferState.COMMITTED:
                    errors.append(
                        f"committed group {group.group_id} has uncommitted transfer"
                    )
        else:
            for transfer_id in group.shard_transfers.values():
                transfer = self.transfers[transfer_id]
                if transfer.state in {
                    TransferState.QUEUED,
                    TransferState.IN_FLIGHT,
                    TransferState.ACKED,
                }:
                    errors.append(f"aborted group {group.group_id} has active transfer")
                extent = self.extents.get(transfer.extent_id)
                if (
                    extent is not None
                    and extent.locations.get(transfer.destination)
                    is KvCopyState.PENDING
                ):
                    errors.append(f"aborted group {group.group_id} retains pending bytes")
        return errors

    def _audit_recovery_load_group(
        self, group: RecoveryLoadGroup
    ) -> list[str]:
        errors: list[str] = []
        expected = set(group.expected_shard_keys)
        bound = set(group.shard_transfers)
        if not group.adopted_shards.issubset(expected):
            errors.append(
                f"recovery group {group.group_id} adopted unexpected shards"
            )
        if group.adopted_shards and group.state is not CommitGroupState.COMMITTED:
            errors.append(
                f"non-committed recovery group {group.group_id} adopted shards"
            )
        if group.discarded and group.adopted_shards == expected:
            errors.append(
                f"fully adopted recovery group {group.group_id} is discarded"
            )
        fully_adopted = group.adopted_shards == expected
        if fully_adopted != (group.adopted_at_ns is not None):
            errors.append(
                f"recovery group {group.group_id} has inconsistent adoption time"
            )
        if group.discarded != (group.discarded_at_ns is not None):
            errors.append(
                f"recovery group {group.group_id} has inconsistent discard time"
            )
        if group.discarded != (group.discard_reason is not None):
            errors.append(
                f"recovery group {group.group_id} has inconsistent discard reason"
            )
        if not bound.issubset(expected):
            errors.append(
                f"recovery group {group.group_id} binds unexpected shard keys"
            )
        all_transfer_ids = list(group.transfer_ids)
        if len(all_transfer_ids) != len(set(all_transfer_ids)):
            errors.append(
                f"recovery group {group.group_id} binds a transfer more than once"
            )
        if not group.acked_transfers.issubset(set(all_transfer_ids)):
            errors.append(
                f"recovery group {group.group_id} ACKs unknown transfers"
            )
        state_acked: set[str] = set()
        for shard_key, transfer_ids in group.shard_transfers.items():
            if not transfer_ids:
                errors.append(
                    f"recovery group {group.group_id} shard {shard_key} is empty"
                )
            for transfer_id in transfer_ids:
                transfer = self.transfers.get(transfer_id)
                if transfer is None:
                    errors.append(
                        f"recovery group {group.group_id} has no transfer "
                        f"{transfer_id}"
                    )
                    continue
                if (
                    transfer.recovery_group_id != group.group_id
                    or transfer.recovery_shard_key != shard_key
                ):
                    errors.append(
                        f"recovery group {group.group_id} transfer {transfer_id} "
                        "binding is inconsistent"
                    )
                extent = self.extents.get(transfer.extent_id)
                if extent is not None and extent.route_version != group.route_version:
                    errors.append(
                        f"recovery group {group.group_id} transfer {transfer_id} "
                        "route version mismatch"
                    )
                if transfer.state is TransferState.ACKED:
                    state_acked.add(transfer_id)

        if group.state is CommitGroupState.PENDING:
            if state_acked != group.acked_transfers:
                errors.append(
                    f"recovery group {group.group_id} ACK set disagrees with transfers"
                )
            if group.sealed and bound != expected:
                errors.append(
                    f"sealed recovery group {group.group_id} is missing shards"
                )
            if (
                group.sealed
                and group.acked_transfers == set(all_transfer_ids)
            ):
                errors.append(
                    f"recovery group {group.group_id} missed atomic final commit"
                )
            for transfer_id in all_transfer_ids:
                if self.transfers[transfer_id].state in {
                    TransferState.COMMITTED,
                    TransferState.ABORTED,
                }:
                    errors.append(
                        f"pending recovery group {group.group_id} has terminal transfer"
                    )
        elif group.state is CommitGroupState.COMMITTED:
            if not group.sealed or bound != expected:
                errors.append(
                    f"committed recovery group {group.group_id} is incomplete"
                )
            if group.acked_transfers != set(all_transfer_ids):
                errors.append(
                    f"committed recovery group {group.group_id} is missing ACKs"
                )
            for transfer_id in all_transfer_ids:
                transfer = self.transfers[transfer_id]
                if transfer.state is not TransferState.COMMITTED:
                    errors.append(
                        f"committed recovery group {group.group_id} has "
                        f"uncommitted transfer {transfer_id}"
                    )
                shard_adopted = transfer.recovery_shard_key in group.adopted_shards
                if transfer.recovery_adopted != shard_adopted:
                    errors.append(
                        f"recovery group {group.group_id} transfer {transfer_id} "
                        "adoption disagrees with its shard"
                    )
                should_be_discarded = bool(
                    group.discarded and not shard_adopted
                )
                if transfer.recovery_discarded != should_be_discarded:
                    errors.append(
                        f"recovery group {group.group_id} transfer {transfer_id} "
                        "discard state disagrees with its group"
                    )
        else:
            for transfer_id in all_transfer_ids:
                transfer = self.transfers[transfer_id]
                if transfer.state in {
                    TransferState.QUEUED,
                    TransferState.IN_FLIGHT,
                    TransferState.ACKED,
                }:
                    errors.append(
                        f"aborted recovery group {group.group_id} has active transfer"
                    )
                if transfer.recovery_capacity_hold_active:
                    errors.append(
                        f"aborted recovery group {group.group_id} retains capacity"
                    )
        return errors

    def _audit_extent_account_conservation(
        self, mode: ExtentAccountingAuditMode
    ) -> list[str]:
        internal_gpu = {
            gpu: {
                HbmAccount.PRIMARY_KV: 0,
                HbmAccount.PREFIX_KV: 0,
                HbmAccount.RECOVERY_PENDING: 0,
                HbmAccount.AEGIS_PENDING: 0,
                HbmAccount.AEGIS_COMMITTED: 0,
            }
            for gpu in self.gpu_hbm
        }
        internal_dram = {
            node: {
                DramAccount.HICACHE_PRIMARY: 0,
                DramAccount.AEGIS_PENDING: 0,
                DramAccount.AEGIS_COMMITTED: 0,
            }
            for node in self.node_dram
        }
        external_gpu_primary: dict[
            PhysicalGpuId, dict[str, tuple[Any, ...]]
        ] = {gpu: {} for gpu in self.gpu_hbm}
        external_dram_hicache: dict[
            str, dict[str, tuple[Any, ...]]
        ] = {node: {} for node in self.node_dram}
        for extent in self.extents.values():
            for location, state in extent.locations.items():
                if state not in {KvCopyState.PENDING, KvCopyState.COMMITTED}:
                    continue
                accounting = extent.location_accounting.get(
                    location, LocationAccounting.INTERNAL
                )
                if accounting is LocationAccounting.EXTERNAL:
                    account_key = extent.external_account_keys[location]
                    identity = self._external_source_identity(extent)
                    if location.tier is StorageTier.GPU_PRIMARY:
                        assert location.gpu is not None
                        external_gpu_primary[location.gpu].setdefault(
                            account_key, identity
                        )
                    elif location.tier is StorageTier.NODE_HICACHE_PRIMARY:
                        assert location.node_id is not None
                        external_dram_hicache[location.node_id].setdefault(
                            account_key, identity
                        )
                    continue
                if location.tier is StorageTier.GPU_PRIMARY:
                    assert location.gpu is not None
                    internal_gpu[location.gpu][HbmAccount.PRIMARY_KV] += extent.size_bytes
                elif location.tier is StorageTier.GPU_PREFIX:
                    assert location.gpu is not None
                    internal_gpu[location.gpu][HbmAccount.PREFIX_KV] += extent.size_bytes
                elif location.tier is StorageTier.GPU_AEGIS:
                    assert location.gpu is not None
                    account = (
                        HbmAccount.AEGIS_PENDING
                        if state is KvCopyState.PENDING
                        else HbmAccount.AEGIS_COMMITTED
                    )
                    internal_gpu[location.gpu][account] += extent.size_bytes
                elif location.tier is StorageTier.NODE_HICACHE_PRIMARY:
                    assert location.node_id is not None
                    internal_dram[location.node_id][
                        DramAccount.HICACHE_PRIMARY
                    ] += extent.size_bytes
                else:
                    assert location.node_id is not None
                    account = (
                        DramAccount.AEGIS_PENDING
                        if state is KvCopyState.PENDING
                        else DramAccount.AEGIS_COMMITTED
                    )
                    internal_dram[location.node_id][account] += extent.size_bytes

        for transfer in self.transfers.values():
            if not transfer.recovery_capacity_hold_active:
                continue
            if (
                transfer.destination_mode
                is not TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                or transfer.destination.gpu is None
            ):
                continue
            internal_gpu[transfer.destination.gpu][
                HbmAccount.RECOVERY_PENDING
            ] += transfer.size_bytes

        errors: list[str] = []
        for gpu, accounts in internal_gpu.items():
            ledger = self.gpu_hbm[gpu]
            for account, expected in accounts.items():
                external = (
                    sum(
                        int(identity[0])
                        for identity in external_gpu_primary[gpu].values()
                    )
                    if account is HbmAccount.PRIMARY_KV
                    else 0
                )
                required = expected + external
                observed = int(getattr(ledger, account.value))
                aggregate_account = account is HbmAccount.PRIMARY_KV
                valid = (
                    observed >= required
                    if mode is ExtentAccountingAuditMode.EXTERNAL_AGGREGATE
                    and aggregate_account
                    else observed == required
                )
                if not valid:
                    relation = ">=" if (
                        mode is ExtentAccountingAuditMode.EXTERNAL_AGGREGATE
                        and aggregate_account
                    ) else "=="
                    errors.append(
                        f"GPU {gpu.key} {account.value}: ledger={observed}, "
                        f"required {relation} internal({expected}) + "
                        f"external({external})"
                    )
        for node, accounts in internal_dram.items():
            ledger = self.node_dram[node]
            for account, expected in accounts.items():
                external = (
                    sum(
                        int(identity[0])
                        for identity in external_dram_hicache[node].values()
                    )
                    if account is DramAccount.HICACHE_PRIMARY
                    else 0
                )
                required = expected + external
                observed = int(getattr(ledger, account.value))
                aggregate_account = account is DramAccount.HICACHE_PRIMARY
                valid = (
                    observed >= required
                    if mode is ExtentAccountingAuditMode.EXTERNAL_AGGREGATE
                    and aggregate_account
                    else observed == required
                )
                if not valid:
                    relation = ">=" if (
                        mode is ExtentAccountingAuditMode.EXTERNAL_AGGREGATE
                        and aggregate_account
                    ) else "=="
                    errors.append(
                        f"DRAM {node} {account.value}: ledger={observed}, "
                        f"required {relation} internal({expected}) + "
                        f"external({external})"
                    )
        return errors

    def _start_transfer(self, transfer_id: str) -> None:
        transfer = self.transfers[transfer_id]
        if transfer.state is TransferState.ABORTED:
            event_id = self._current_event_id or self._operation_id()
            snapshot = transfer.snapshot()
            self._append_journal(
                event_id=event_id,
                kind="transfer_start_skipped",
                entity=transfer.transfer_id,
                before=snapshot,
                after=snapshot,
                metadata={"reason": "already_aborted"},
            )
            self._record_transfer(transfer, event_id, "transfer_start_skipped")
            return
        if transfer.state is not TransferState.QUEUED:
            raise RuntimeError(f"transfer {transfer_id} is not queued")
        endpoint_error = self._transfer_endpoint_error(transfer)
        if endpoint_error is not None:
            self.abort_transfer(
                transfer.transfer_id,
                reason=f"start_rejected:{endpoint_error}",
                event_id=self._current_event_id,
            )
            return
        event_id = self._current_event_id or self._operation_id()
        before = transfer.snapshot()
        transfer.state = TransferState.IN_FLIGHT
        self._append_journal(
            event_id=event_id,
            kind="transfer_started",
            entity=transfer.transfer_id,
            before=before,
            after=transfer.snapshot(),
        )
        self._record_transfer(transfer, event_id, "transfer_started")

    def _complete_transfer(self, transfer_id: str) -> None:
        transfer = self.transfers[transfer_id]
        event_id = self._current_event_id or self._operation_id()
        if transfer.state is TransferState.ABORTED:
            snapshot = transfer.snapshot()
            self._append_journal(
                event_id=event_id,
                kind="transfer_completion_skipped",
                entity=transfer.transfer_id,
                before=snapshot,
                after=snapshot,
                metadata={"reason": "already_aborted"},
            )
            self._record_transfer(transfer, event_id, "transfer_completion_skipped")
            return
        if transfer.state is not TransferState.IN_FLIGHT:
            raise RuntimeError(f"transfer {transfer_id} is not in flight")
        extent = self._require_extent(transfer.extent_id)
        endpoint_error = self._transfer_endpoint_error(transfer)
        if endpoint_error is not None:
            self.abort_transfer(
                transfer.transfer_id,
                reason=f"completion_rejected:{endpoint_error}",
                event_id=event_id,
            )
            return

        if transfer.recovery_group_id is not None:
            self._ack_recovery_group_transfer(transfer, event_id=event_id)
            return
        if transfer.group_id is not None:
            self._ack_group_transfer(transfer, event_id=event_id)
            return

        if (
            transfer.destination_mode
            is TransferDestinationMode.MATERIALIZED_COPY
        ):
            extent_before = extent.snapshot()
            self._transition_location_to_committed(
                transfer.destination,
                transfer.size_bytes,
                event_id=event_id,
            )
            extent.locations[transfer.destination] = KvCopyState.COMMITTED
            if transfer.remove_source_on_complete:
                if extent.location_accounting.get(
                    transfer.source, LocationAccounting.INTERNAL
                ) is LocationAccounting.INTERNAL:
                    self._remove_location_accounting(
                        transfer.source,
                        KvCopyState.COMMITTED,
                        transfer.size_bytes,
                        event_id=event_id,
                        reason="transfer_source_removed",
                    )
                del extent.locations[transfer.source]
                extent.location_accounting.pop(transfer.source, None)
                extent.external_account_keys.pop(transfer.source, None)
            self._append_journal(
                event_id=event_id,
                kind="extent_transfer_committed",
                entity=extent.extent_id,
                before=extent_before,
                after=extent.snapshot(),
                metadata={"transfer_id": transfer.transfer_id},
            )
            self._record_extent(extent, event_id, "transfer_committed")

        transfer_before = transfer.snapshot()
        transfer.state = TransferState.COMMITTED
        self._append_journal(
            event_id=event_id,
            kind="transfer_committed",
            entity=transfer.transfer_id,
            before=transfer_before,
            after=transfer.snapshot(),
        )
        self._record_transfer(transfer, event_id, "transfer_committed")

    def _ack_group_transfer(
        self, transfer: TransferRecord, *, event_id: str
    ) -> None:
        assert transfer.group_id is not None and transfer.shard_key is not None
        group = self._require_group(transfer.group_id)
        if group.state is not CommitGroupState.PENDING:
            self.abort_transfer(
                transfer.transfer_id,
                reason=f"group_not_pending:{group.state.value}",
                event_id=event_id,
            )
            return
        if group.shard_transfers.get(transfer.shard_key) != transfer.transfer_id:
            self.abort_commit_group(
                group.group_id,
                reason="shard_binding_mismatch",
                event_id=event_id,
            )
            return

        transfer_before = transfer.snapshot()
        transfer.state = TransferState.ACKED
        self._append_journal(
            event_id=event_id,
            kind="transfer_acked",
            entity=transfer.transfer_id,
            before=transfer_before,
            after=transfer.snapshot(),
            metadata={
                "group_id": group.group_id,
                "shard_key": transfer.shard_key,
            },
        )
        self._record_transfer(transfer, event_id, "transfer_acked")

        group_before = group.snapshot()
        group.acked_shards.add(transfer.shard_key)
        self._append_journal(
            event_id=event_id,
            kind="commit_group_shard_acked",
            entity=group.group_id,
            before=group_before,
            after=group.snapshot(),
            metadata={
                "shard_key": transfer.shard_key,
                "transfer_id": transfer.transfer_id,
            },
        )
        self._record_group(group, event_id, "commit_group_shard_acked")

        if group.acked_shards == set(group.expected_shard_keys):
            error = self._group_commit_error(group)
            if error is not None:
                self.abort_commit_group(
                    group.group_id,
                    reason=f"commit_validation_failed:{error}",
                    event_id=event_id,
                )
                return
            self._commit_group(group, event_id=event_id)

    def _ack_recovery_group_transfer(
        self, transfer: TransferRecord, *, event_id: str
    ) -> None:
        assert (
            transfer.recovery_group_id is not None
            and transfer.recovery_shard_key is not None
        )
        group = self._require_recovery_group(transfer.recovery_group_id)
        if group.state is not CommitGroupState.PENDING:
            self.abort_transfer(
                transfer.transfer_id,
                reason=f"recovery_group_not_pending:{group.state.value}",
                event_id=event_id,
            )
            return
        if transfer.transfer_id not in group.shard_transfers.get(
            transfer.recovery_shard_key, ()
        ):
            self.abort_recovery_load_group(
                group.group_id,
                reason="transfer_binding_mismatch",
                event_id=event_id,
            )
            return

        transfer_before = transfer.snapshot()
        transfer.state = TransferState.ACKED
        self._append_journal(
            event_id=event_id,
            kind="recovery_transfer_acked",
            entity=transfer.transfer_id,
            before=transfer_before,
            after=transfer.snapshot(),
            metadata={
                "group_id": group.group_id,
                "shard_key": transfer.recovery_shard_key,
            },
        )
        self._record_transfer(transfer, event_id, "recovery_transfer_acked")

        group_before = group.snapshot()
        group.acked_transfers.add(transfer.transfer_id)
        self._append_journal(
            event_id=event_id,
            kind="recovery_load_group_transfer_acked",
            entity=group.group_id,
            before=group_before,
            after=group.snapshot(),
            metadata={
                "shard_key": transfer.recovery_shard_key,
                "transfer_id": transfer.transfer_id,
            },
        )
        self._record_recovery_group(
            group, event_id, "recovery_load_group_transfer_acked"
        )

        if group.sealed and group.acked_transfers == set(group.transfer_ids):
            error = self._recovery_group_commit_error(group)
            if error is not None:
                self.abort_recovery_load_group(
                    group.group_id,
                    reason=f"commit_validation_failed:{error}",
                    event_id=event_id,
                )
                return
            self._commit_recovery_load_group(group, event_id=event_id)

    def _recovery_group_commit_error(
        self, group: RecoveryLoadGroup
    ) -> str | None:
        if not group.sealed:
            return "group_not_sealed"
        expected = set(group.expected_shard_keys)
        if set(group.shard_transfers) != expected:
            return "missing_shard_transfer"
        transfer_ids = set(group.transfer_ids)
        if group.acked_transfers != transfer_ids:
            return "missing_transfer_ack"
        for transfer_id in group.transfer_ids:
            transfer = self.transfers[transfer_id]
            if transfer.state is not TransferState.ACKED:
                return f"transfer_{transfer_id}_not_acked"
            endpoint_error = self._transfer_endpoint_error(transfer)
            if endpoint_error is not None:
                return f"transfer_{transfer_id}_{endpoint_error}"
            if (
                transfer.destination_mode
                is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                and not transfer.recovery_capacity_hold_active
            ):
                return f"transfer_{transfer_id}_missing_capacity_hold"
        return None

    def _commit_recovery_load_group(
        self, group: RecoveryLoadGroup, *, event_id: str
    ) -> None:
        """Commit all recovery records in the final ACK's global event."""

        error = self._recovery_group_commit_error(group)
        if error is not None:
            raise RuntimeError(
                f"recovery group {group.group_id} cannot commit: {error}"
            )
        group_before = group.snapshot()
        for transfer_id in group.transfer_ids:
            transfer = self.transfers[transfer_id]
            transfer_before = transfer.snapshot()
            transfer.state = TransferState.COMMITTED
            self._append_journal(
                event_id=event_id,
                kind="recovery_transfer_group_committed",
                entity=transfer.transfer_id,
                before=transfer_before,
                after=transfer.snapshot(),
                metadata={
                    "group_id": group.group_id,
                    "shard_key": transfer.recovery_shard_key,
                },
            )
            self._record_transfer(
                transfer, event_id, "recovery_transfer_group_committed"
            )
        group.state = CommitGroupState.COMMITTED
        self._append_journal(
            event_id=event_id,
            kind="recovery_load_group_committed",
            entity=group.group_id,
            before=group_before,
            after=group.snapshot(),
        )
        self._record_recovery_group(
            group, event_id, "recovery_load_group_committed"
        )

    def _group_commit_error(self, group: TierCommitGroup) -> str | None:
        expected = set(group.expected_shard_keys)
        if set(group.shard_transfers) != expected:
            return "missing_shard_transfer"
        if group.acked_shards != expected:
            return "missing_shard_ack"

        gpu_pending_required: dict[PhysicalGpuId, int] = {}
        dram_pending_required: dict[str, int] = {}
        for shard_key in group.expected_shard_keys:
            transfer = self.transfers[group.shard_transfers[shard_key]]
            if transfer.state is not TransferState.ACKED:
                return f"shard_{shard_key}_not_acked"
            endpoint_error = self._transfer_endpoint_error(transfer)
            if endpoint_error is not None:
                return f"shard_{shard_key}_{endpoint_error}"
            extent = self._require_extent(transfer.extent_id)
            if not self._group_identity_matches(group, extent, transfer.destination):
                return f"shard_{shard_key}_identity_mismatch"
            if group.tier is StorageTier.GPU_AEGIS:
                assert transfer.destination.gpu is not None
                gpu = transfer.destination.gpu
                gpu_pending_required[gpu] = (
                    gpu_pending_required.get(gpu, 0) + extent.size_bytes
                )
            else:
                assert transfer.destination.node_id is not None
                node = transfer.destination.node_id
                dram_pending_required[node] = (
                    dram_pending_required.get(node, 0) + extent.size_bytes
                )
        for gpu, required in gpu_pending_required.items():
            if self._require_gpu(gpu).aegis_pending < required:
                return f"pending_hbm_underflow:{gpu.key}"
        for node, required in dram_pending_required.items():
            if self._require_dram(node).aegis_pending < required:
                return f"pending_dram_underflow:{node}"
        return None

    def _commit_group(self, group: TierCommitGroup, *, event_id: str) -> None:
        """Commit every ACKed shard in the final ACK's global event."""

        error = self._group_commit_error(group)
        if error is not None:
            raise RuntimeError(f"group {group.group_id} cannot commit: {error}")
        group_before = group.snapshot()
        for shard_key in group.expected_shard_keys:
            transfer = self.transfers[group.shard_transfers[shard_key]]
            extent = self._require_extent(transfer.extent_id)
            extent_before = extent.snapshot()
            self._transition_location_to_committed(
                transfer.destination,
                extent.size_bytes,
                event_id=event_id,
            )
            extent.locations[transfer.destination] = KvCopyState.COMMITTED
            if transfer.remove_source_on_complete:
                if extent.location_accounting.get(
                    transfer.source, LocationAccounting.INTERNAL
                ) is LocationAccounting.INTERNAL:
                    self._remove_location_accounting(
                        transfer.source,
                        KvCopyState.COMMITTED,
                        extent.size_bytes,
                        event_id=event_id,
                        reason="group_transfer_source_removed",
                    )
                del extent.locations[transfer.source]
                extent.location_accounting.pop(transfer.source, None)
                extent.external_account_keys.pop(transfer.source, None)
            self._append_journal(
                event_id=event_id,
                kind="extent_group_transfer_committed",
                entity=extent.extent_id,
                before=extent_before,
                after=extent.snapshot(),
                metadata={
                    "group_id": group.group_id,
                    "shard_key": shard_key,
                    "transfer_id": transfer.transfer_id,
                },
            )
            self._record_extent(extent, event_id, "group_transfer_committed")

            transfer_before = transfer.snapshot()
            transfer.state = TransferState.COMMITTED
            self._append_journal(
                event_id=event_id,
                kind="transfer_group_committed",
                entity=transfer.transfer_id,
                before=transfer_before,
                after=transfer.snapshot(),
                metadata={"group_id": group.group_id, "shard_key": shard_key},
            )
            self._record_transfer(transfer, event_id, "transfer_group_committed")

        group.state = CommitGroupState.COMMITTED
        self._append_journal(
            event_id=event_id,
            kind="commit_group_committed",
            entity=group.group_id,
            before=group_before,
            after=group.snapshot(),
        )
        self._record_group(group, event_id, "commit_group_committed")

    def _transition_location_to_committed(
        self,
        location: KvLocation,
        size_bytes: int,
        *,
        event_id: str,
    ) -> None:
        if location.tier is StorageTier.GPU_AEGIS:
            assert location.gpu is not None
            ledger = self._require_gpu(location.gpu)
            before = ledger.snapshot()
            if ledger.aegis_pending < size_bytes:
                raise RuntimeError("GPU Aegis pending accounting underflow")
            ledger.aegis_pending -= size_bytes
            ledger.aegis_committed += size_bytes
            ledger.validate()
            self._append_journal(
                event_id=event_id,
                kind="gpu_hbm_transfer_committed",
                entity=location.gpu.key,
                before=before,
                after=ledger.snapshot(),
                metadata={"size_bytes": size_bytes},
            )
            self._record_gpu(location.gpu, event_id, "transfer_committed")
            return
        if location.tier is StorageTier.NODE_DRAM:
            assert location.node_id is not None
            ledger = self._require_dram(location.node_id)
            before = ledger.snapshot()
            if ledger.aegis_pending < size_bytes:
                raise RuntimeError("DRAM Aegis pending accounting underflow")
            ledger.aegis_pending -= size_bytes
            ledger.aegis_committed += size_bytes
            ledger.validate()
            self._append_journal(
                event_id=event_id,
                kind="node_dram_transfer_committed",
                entity=location.node_id,
                before=before,
                after=ledger.snapshot(),
                metadata={"size_bytes": size_bytes},
            )
            self._record_dram(location.node_id, event_id, "transfer_committed")
            return
        if location.tier in {
            StorageTier.GPU_PRIMARY,
            StorageTier.GPU_PREFIX,
            StorageTier.NODE_HICACHE_PRIMARY,
        }:
            # Recovery/HiCache destinations reserve their final account at
            # QUEUED time, so PENDING -> COMMITTED is metadata-only here.
            return
        raise ValueError("unsupported transfer destination")

    def _adjust_location(
        self,
        location: KvLocation,
        state: KvCopyState,
        delta_bytes: int,
        *,
        reason: str,
        event_id: str,
    ) -> None:
        if state not in {KvCopyState.PENDING, KvCopyState.COMMITTED}:
            return
        if location.tier is StorageTier.GPU_PRIMARY:
            assert location.gpu is not None
            self.adjust_gpu_hbm(
                location.gpu,
                HbmAccount.PRIMARY_KV,
                delta_bytes,
                reason=reason,
                event_id=event_id,
            )
        elif location.tier is StorageTier.GPU_PREFIX:
            assert location.gpu is not None
            self.adjust_gpu_hbm(
                location.gpu,
                HbmAccount.PREFIX_KV,
                delta_bytes,
                reason=reason,
                event_id=event_id,
            )
        elif location.tier is StorageTier.GPU_AEGIS:
            assert location.gpu is not None
            account = (
                HbmAccount.AEGIS_PENDING
                if state is KvCopyState.PENDING
                else HbmAccount.AEGIS_COMMITTED
            )
            self.adjust_gpu_hbm(
                location.gpu,
                account,
                delta_bytes,
                reason=reason,
                event_id=event_id,
            )
        elif location.tier is StorageTier.NODE_DRAM:
            assert location.node_id is not None
            account = (
                DramAccount.AEGIS_PENDING
                if state is KvCopyState.PENDING
                else DramAccount.AEGIS_COMMITTED
            )
            self.adjust_node_dram(
                location.node_id,
                account,
                delta_bytes,
                reason=reason,
                event_id=event_id,
            )
        else:
            assert location.tier is StorageTier.NODE_HICACHE_PRIMARY
            assert location.node_id is not None
            self.adjust_node_dram(
                location.node_id,
                DramAccount.HICACHE_PRIMARY,
                delta_bytes,
                reason=reason,
                event_id=event_id,
            )

    def _remove_location_accounting(
        self,
        location: KvLocation,
        state: KvCopyState,
        size_bytes: int,
        *,
        event_id: str,
        reason: str,
    ) -> None:
        self._adjust_location(
            location,
            state,
            -size_bytes,
            reason=reason,
            event_id=event_id,
        )

    def _release_recovery_capacity_hold(
        self,
        transfer: TransferRecord,
        *,
        event_id: str,
        reason: str,
    ) -> None:
        if not transfer.recovery_capacity_hold_active:
            return
        if (
            transfer.destination_mode
            is not TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
            or transfer.destination.gpu is None
        ):
            raise RuntimeError("invalid recovery-capacity hold owner")
        gpu = transfer.destination.gpu
        ledger = self._require_gpu(gpu)
        if ledger.healthy:
            if ledger.recovery_pending < transfer.size_bytes:
                raise RuntimeError("GPU recovery pending accounting underflow")
            self.adjust_gpu_hbm(
                gpu,
                HbmAccount.RECOVERY_PENDING,
                -transfer.size_bytes,
                reason=reason,
                event_id=event_id,
            )
        elif ledger.recovery_pending:
            raise RuntimeError("failed GPU retained recovery pending bytes")
        transfer.recovery_capacity_hold_active = False

    def _ensure_location_capacity(
        self,
        location: KvLocation,
        state: KvCopyState,
        size_bytes: int,
    ) -> None:
        if location.gpu is not None:
            ledger = self._require_gpu(location.gpu)
        else:
            assert location.node_id is not None
            ledger = self._require_dram(location.node_id)
        if ledger.free_bytes < size_bytes:
            raise ValueError(
                f"{location.key} has {ledger.free_bytes} free bytes, needs {size_bytes}"
            )
        del state  # The selected account does not change total capacity admission.

    def _validate_location(
        self, location: KvLocation, *, require_healthy: bool = False
    ) -> None:
        if not isinstance(location, KvLocation):
            raise TypeError("location must be a KvLocation")
        if location.gpu is not None:
            ledger = self._require_gpu(location.gpu)
            if require_healthy and not ledger.healthy:
                raise ValueError(f"GPU destination/source {location.gpu.key} is unhealthy")
        else:
            assert location.node_id is not None
            ledger = self._require_dram(location.node_id)
            if require_healthy and not ledger.healthy:
                raise ValueError(f"DRAM node {location.node_id!r} is unhealthy")

    def _validate_transfer_path(
        self, source: KvLocation, destination: KvLocation
    ) -> None:
        if (
            source.tier is StorageTier.GPU_AEGIS
            and destination.tier
            in {StorageTier.NODE_DRAM, StorageTier.NODE_HICACHE_PRIMARY}
        ):
            raise ValueError(
                "GPU_AEGIS to node DRAM destage is forbidden by current design"
            )
        recovery_sources = {
            StorageTier.GPU_AEGIS,
            StorageTier.NODE_DRAM,
            StorageTier.NODE_HICACHE_PRIMARY,
        }
        if destination.tier in {
            StorageTier.GPU_PRIMARY,
            StorageTier.GPU_PREFIX,
        }:
            if source.tier not in recovery_sources:
                raise ValueError(
                    "recovery load requires a DRAM or peer-HBM committed source"
                )
            return
        if destination.tier in {
            StorageTier.GPU_AEGIS,
            StorageTier.NODE_DRAM,
            StorageTier.NODE_HICACHE_PRIMARY,
        }:
            if source.tier not in {
                StorageTier.GPU_PRIMARY,
                StorageTier.GPU_PREFIX,
            }:
                raise ValueError("protection/offload source must be primary GPU KV")
            return
        raise ValueError("unsupported transfer path")

    @staticmethod
    def _resolve_transfer_span(
        extent: KvExtent,
        *,
        transfer_size_bytes: int | None,
        token_start: int | None,
        token_end: int | None,
    ) -> tuple[int, int, int]:
        if (token_start is None) != (token_end is None):
            raise ValueError("token_start and token_end must be provided together")
        extent_has_span = extent.token_end > extent.token_start
        if token_start is None:
            start = extent.token_start
            end = extent.token_end
        else:
            start = _nonnegative_int("transfer token_start", token_start)
            end = _nonnegative_int("transfer token_end", token_end)
            if end <= start:
                raise ValueError("transfer token_end must exceed token_start")
            if not extent_has_span:
                raise ValueError(
                    "partial token span requires extent token_start/token_end metadata"
                )
            if start < extent.token_start or end > extent.token_end:
                raise ValueError("transfer token span is outside the source extent")

        if transfer_size_bytes is None:
            if token_start is None or (
                start == extent.token_start and end == extent.token_end
            ):
                size = extent.size_bytes
            else:
                extent_tokens = extent.token_end - extent.token_start
                span_tokens = end - start
                numerator = extent.size_bytes * span_tokens
                if numerator % extent_tokens:
                    raise ValueError("transfer token span has non-integral byte size")
                size = numerator // extent_tokens
        else:
            size = _nonnegative_int(
                "transfer_size_bytes", transfer_size_bytes
            )
            if size == 0:
                raise ValueError("transfer_size_bytes must be positive")
            if size > extent.size_bytes:
                raise ValueError("transfer_size_bytes exceeds source extent size")
            if token_start is not None:
                extent_tokens = extent.token_end - extent.token_start
                span_tokens = end - start
                if size * extent_tokens != extent.size_bytes * span_tokens:
                    raise ValueError(
                        "transfer_size_bytes does not match the token subspan"
                    )
            elif size != extent.size_bytes:
                raise ValueError(
                    "partial transfer_size_bytes requires a token subspan"
                )
        return size, start, end

    @staticmethod
    def _validate_destination_mode(
        mode: TransferDestinationMode,
        *,
        extent: KvExtent,
        source: KvLocation,
        destination: KvLocation,
        transfer_size_bytes: int,
        resources: Sequence[TransferResourceId],
        duration_ns: int,
        remove_source_on_complete: bool,
    ) -> None:
        if mode is TransferDestinationMode.MATERIALIZED_COPY:
            if transfer_size_bytes != extent.size_bytes:
                raise ValueError(
                    "MATERIALIZED_COPY requires the complete source extent"
                )
            return
        if remove_source_on_complete:
            raise ValueError(
                "external recovery modes cannot remove the protection source"
            )
        if destination.tier not in {
            StorageTier.GPU_PRIMARY,
            StorageTier.GPU_PREFIX,
        } or destination.gpu is None:
            raise ValueError("external recovery destination must be a GPU KV tier")
        if mode is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK:
            if source == destination:
                raise ValueError("external recovery copy requires distinct endpoints")
            return
        assert mode is TransferDestinationMode.OWNERSHIP_ACTIVATION
        if (
            source.tier is not StorageTier.GPU_AEGIS
            or source.gpu is None
            or source.gpu != destination.gpu
        ):
            raise ValueError(
                "OWNERSHIP_ACTIVATION requires same-GPU GPU_AEGIS source"
            )
        if tuple(resources):
            raise ValueError("OWNERSHIP_ACTIVATION cannot reserve copy resources")
        if _nonnegative_int("duration_ns", duration_ns) != 0:
            raise ValueError("OWNERSHIP_ACTIVATION duration must be zero")

    def _validate_group_binding(
        self,
        group: TierCommitGroup,
        *,
        extent: KvExtent,
        destination: KvLocation,
        shard_key: str,
    ) -> None:
        if group.state is not CommitGroupState.PENDING:
            raise ValueError(
                f"commit group {group.group_id} is {group.state.value}, not PENDING"
            )
        if shard_key not in group.expected_shard_keys:
            raise ValueError(
                f"shard {shard_key!r} is not expected by group {group.group_id}"
            )
        if shard_key in group.shard_transfers:
            raise ValueError(
                f"duplicate shard {shard_key!r} in group {group.group_id}"
            )
        if any(
            self.transfers[transfer_id].extent_id == extent.extent_id
            for transfer_id in group.shard_transfers.values()
        ):
            raise ValueError(
                f"extent {extent.extent_id!r} is already bound to group {group.group_id}"
            )
        if extent.shard_count != len(group.expected_shard_keys):
            raise ValueError(
                "extent shard_count does not match group expected shard count"
            )
        if not self._group_identity_matches(group, extent, destination):
            raise ValueError(
                "commit group identity/layout/version/tier mismatch"
            )

    def _validate_recovery_group_binding(
        self,
        group: RecoveryLoadGroup,
        *,
        extent: KvExtent,
        destination_mode: TransferDestinationMode,
        shard_key: str,
    ) -> None:
        if group.state is not CommitGroupState.PENDING:
            raise ValueError(
                f"recovery group {group.group_id} is {group.state.value}, not PENDING"
            )
        if group.sealed:
            raise ValueError(f"recovery group {group.group_id} is already sealed")
        if shard_key not in group.expected_shard_keys:
            raise ValueError(
                f"shard {shard_key!r} is not expected by recovery group "
                f"{group.group_id}"
            )
        if extent.route_version != group.route_version:
            raise ValueError("recovery group route version mismatch")
        if destination_mode not in {
            TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK,
            TransferDestinationMode.OWNERSHIP_ACTIVATION,
        }:
            raise ValueError(
                "recovery groups require an external sink or ownership activation"
            )

    @staticmethod
    def _group_identity_matches(
        group: TierCommitGroup,
        extent: KvExtent,
        destination: KvLocation,
    ) -> bool:
        return (
            destination.tier is group.tier
            and extent.route_version == group.route_version
            and extent.model_revision == group.model_revision
            and extent.kv_abi == group.kv_abi
            and extent.parallel_layout == group.parallel_layout
        )

    def _external_live_bytes(self, location: KvLocation) -> int:
        groups: dict[str, tuple[Any, ...]] = {}
        for extent in self.extents.values():
            if (
                extent.locations.get(location)
                not in {KvCopyState.PENDING, KvCopyState.COMMITTED}
                or extent.location_accounting.get(location)
                is not LocationAccounting.EXTERNAL
            ):
                continue
            key = extent.external_account_keys[location]
            identity = self._external_source_identity(extent)
            previous = groups.setdefault(key, identity)
            if previous != identity:
                raise ValueError("external_account_key size/identity conflict")
        return sum(int(identity[0]) for identity in groups.values())

    @staticmethod
    def _external_source_identity(extent: KvExtent) -> tuple[Any, ...]:
        """Identity of one scheduler-owned production-buffer byte range."""

        return (
            extent.size_bytes,
            extent.request_id,
            extent.workflow_id,
            extent.token_start,
            extent.token_end,
            extent.route_version,
            extent.model_revision,
            extent.kv_abi,
            extent.parallel_layout,
            extent.pp_rank,
            extent.tp_rank,
            extent.shard_count,
        )

    def _internal_live_bytes(self, location: KvLocation) -> int:
        return sum(
            extent.size_bytes
            for extent in self.extents.values()
            if extent.locations.get(location)
            in {KvCopyState.PENDING, KvCopyState.COMMITTED}
            and extent.location_accounting.get(
                location, LocationAccounting.INTERNAL
            )
            is LocationAccounting.INTERNAL
        )

    def _external_aggregate_observed_bytes(self, location: KvLocation) -> int:
        if location.tier is StorageTier.GPU_PRIMARY:
            assert location.gpu is not None
            return self._require_gpu(location.gpu).primary_kv
        if location.tier is StorageTier.NODE_HICACHE_PRIMARY:
            assert location.node_id is not None
            return self._require_dram(location.node_id).hicache_primary
        raise ValueError("location is not an externally accounted source tier")

    def _location_is_healthy(self, location: KvLocation) -> bool:
        if location.gpu is not None:
            return self._require_gpu(location.gpu).healthy
        assert location.node_id is not None
        return self._require_dram(location.node_id).healthy

    def _location_node(self, location: KvLocation) -> str:
        if location.gpu is not None:
            return location.gpu.node_id
        assert location.node_id is not None
        return location.node_id

    def _transfer_endpoint_error(self, transfer: TransferRecord) -> str | None:
        extent = self._require_extent(transfer.extent_id)
        if not self._location_is_healthy(transfer.source):
            return "source_unhealthy"
        if not self._location_is_healthy(transfer.destination):
            return "destination_unhealthy"
        if extent.locations.get(transfer.source) is not KvCopyState.COMMITTED:
            return "source_copy_not_committed"
        if (
            transfer.destination_mode
            is TransferDestinationMode.MATERIALIZED_COPY
        ):
            if extent.locations.get(transfer.destination) is not KvCopyState.PENDING:
                return "destination_copy_not_pending"
        else:
            if extent.locations.get(transfer.destination) in {
                KvCopyState.PENDING,
                KvCopyState.COMMITTED,
            }:
                return "external_destination_was_materialized"
            if (
                transfer.destination_mode
                is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                and not transfer.recovery_capacity_hold_active
            ):
                return "destination_capacity_hold_missing"
        return None

    def _require_gpu(self, gpu: PhysicalGpuId) -> GpuHbmLedger:
        try:
            return self.gpu_hbm[gpu]
        except KeyError as exc:
            raise KeyError(f"GPU {gpu.key} is not registered") from exc

    def _require_dram(self, node_id: str) -> NodeDramLedger:
        try:
            return self.node_dram[node_id]
        except KeyError as exc:
            raise KeyError(f"DRAM node {node_id!r} is not registered") from exc

    def _require_extent(self, extent_id: str) -> KvExtent:
        try:
            return self.extents[str(extent_id)]
        except KeyError as exc:
            raise KeyError(f"extent {extent_id!r} is not registered") from exc

    def _require_group(self, group_id: str) -> TierCommitGroup:
        try:
            return self.commit_groups[str(group_id)]
        except KeyError as exc:
            raise KeyError(f"commit group {group_id!r} is not registered") from exc

    def _require_recovery_group(self, group_id: str) -> RecoveryLoadGroup:
        try:
            return self.recovery_load_groups[str(group_id)]
        except KeyError as exc:
            raise KeyError(
                f"recovery load group {group_id!r} is not registered"
            ) from exc

    def _record_gpu(self, gpu: PhysicalGpuId, event_id: str, reason: str) -> None:
        self._gpu_timeseries.append(
            {
                "timestamp_ns": self.now_ns,
                "event_id": event_id,
                "reason": str(reason),
                **gpu.as_dict(),
                **self.gpu_hbm[gpu].snapshot(),
            }
        )

    def _record_dram(self, node_id: str, event_id: str, reason: str) -> None:
        self._dram_timeseries.append(
            {
                "timestamp_ns": self.now_ns,
                "event_id": event_id,
                "reason": str(reason),
                "node_id": node_id,
                **self.node_dram[node_id].snapshot(),
            }
        )

    def _record_extent(
        self, extent: KvExtent, event_id: str, reason: str
    ) -> None:
        self._extent_timeseries.append(
            {
                "timestamp_ns": self.now_ns,
                "event_id": event_id,
                "reason": str(reason),
                **extent.snapshot(),
            }
        )

    def _record_transfer(
        self, transfer: TransferRecord, event_id: str, reason: str
    ) -> None:
        self._transfer_timeseries.append(
            {
                "timestamp_ns": self.now_ns,
                "event_id": event_id,
                "reason": str(reason),
                **transfer.snapshot(),
            }
        )

    def _record_group(
        self, group: TierCommitGroup, event_id: str, reason: str
    ) -> None:
        self._group_timeseries.append(
            {
                "timestamp_ns": self.now_ns,
                "event_id": event_id,
                "reason": str(reason),
                **group.snapshot(),
            }
        )

    def _record_recovery_group(
        self, group: RecoveryLoadGroup, event_id: str, reason: str
    ) -> None:
        self._recovery_group_timeseries.append(
            {
                "timestamp_ns": self.now_ns,
                "event_id": event_id,
                "reason": str(reason),
                **group.snapshot(),
            }
        )


__all__ = [
    "CommitGroupState",
    "DramAccount",
    "EventPriority",
    "ExtentAccountingAuditMode",
    "GlobalDES",
    "GlobalEvent",
    "GpuHbmLedger",
    "HbmAccount",
    "JournalEntry",
    "KvCopyState",
    "KvExtent",
    "KvLocation",
    "LocationAccounting",
    "NodeDramLedger",
    "PhysicalGpuId",
    "RecoveryLoadGroup",
    "ResourceReservation",
    "StorageTier",
    "TierCommitGroup",
    "TransferRecord",
    "TransferDestinationMode",
    "TransferResourceId",
    "TransferState",
]
