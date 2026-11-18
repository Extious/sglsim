"""Physical AegisKV copy planning for the offline global-DES runtime.

The online Scheduler publishes a *cumulative* post-forward source frontier.
This bridge turns it into non-overlapping physical TP-shard extents and places
source-owned peer-HBM and host-DRAM copies on :class:`GlobalDES`.  It is a
deliberately narrow adapter: admission hints are inputs, while byte capacity,
copy-resource contention, failure ordering, and atomic tier commit are owned
by the global event kernel.

The two protection tiers never copy through one another.  Both originate at
the Scheduler-owned ``GPU_PRIMARY`` production range, form separate commit
groups, and may independently commit or abort.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from sglsim.srt.simulation.global_des import (
    CommitGroupState,
    EventPriority,
    GlobalDES,
    KvCopyState,
    KvLocation,
    PhysicalGpuId,
    StorageTier,
    TransferDestinationMode,
    TransferResourceId,
    TransferState,
)
from sglsim.srt.simulation.offline_resource_bridge import OfflinePhysicalTopology


OFFLINE_AEGIS_TRANSFER_BRIDGE_SCHEMA = (
    "sglsim.offline.aegis-physical-transfer-bridge.v1"
)
_NOTIFY_PRIORITY = int(EventPriority.CONTROL) + 100


def _int_value(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    is_text = isinstance(value, str)
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if is_text:
        canonical = str(value).strip()
        if canonical != str(parsed):
            raise ValueError(f"{name} must be a canonical integer")
    elif parsed != value:
        raise ValueError(f"{name} must be an integer")
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _positive_int(name: str, value: Any) -> int:
    return _int_value(name, value, minimum=1)


def _required_text(name: str, value: Any) -> str:
    parsed = str(value or "")
    if not parsed:
        raise ValueError(f"{name} must be nonempty")
    return parsed


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:24]}"


class CopyPathKind(str, Enum):
    SAME_NODE_D2D_NVLINK = "same_node_d2d_nvlink"
    CROSS_NODE_D2D_FABRIC = "cross_node_d2d_fabric"
    SAME_NODE_SOURCE_D2H = "same_node_source_d2h"
    CROSS_NODE_SOURCE_D2H_FABRIC = "cross_node_source_d2h_fabric"
    PEER_HBM_OWNERSHIP_ACTIVATION = "peer_hbm_ownership_activation"
    SAME_NODE_RECOVERY_D2D = "same_node_recovery_d2d"
    CROSS_NODE_RECOVERY_D2D_FABRIC = "cross_node_recovery_d2d_fabric"
    SAME_NODE_DRAM_H2D = "same_node_dram_h2d"
    CROSS_NODE_DRAM_H2D_FABRIC = "cross_node_dram_h2d_fabric"


@dataclass(frozen=True)
class CopyFabricBandwidths:
    """Sustained byte/s rates used to find one path's bottleneck."""

    gpu_copy_bytes_per_s: int = 100_000_000_000
    nvlink_bytes_per_s: int = 200_000_000_000
    pcie_bytes_per_s: int = 32_000_000_000
    nic_bytes_per_s: int = 25_000_000_000
    fabric_bytes_per_s: int = 25_000_000_000
    dram_write_bytes_per_s: int = 50_000_000_000
    dram_read_bytes_per_s: int = 50_000_000_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))


@dataclass(frozen=True)
class CopyFabricSetupLatencies:
    """Fixed setup cost for each physical copy path, in nanoseconds."""

    same_node_d2d_ns: int = 2_000
    cross_node_d2d_ns: int = 8_000
    same_node_d2h_ns: int = 3_000
    cross_node_d2h_ns: int = 10_000
    same_node_h2d_ns: int = 3_000
    cross_node_h2d_ns: int = 10_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _int_value(name, getattr(self, name), minimum=0)
            )


@dataclass(frozen=True)
class CopyFabricConfig:
    """Physical copy-fabric parameters shared by every planned transfer."""

    bandwidths: CopyFabricBandwidths = field(default_factory=CopyFabricBandwidths)
    setup: CopyFabricSetupLatencies = field(
        default_factory=CopyFabricSetupLatencies
    )
    fabric_id: str = "cluster-fabric-0"

    def __post_init__(self) -> None:
        if not isinstance(self.bandwidths, CopyFabricBandwidths):
            raise TypeError("bandwidths must be CopyFabricBandwidths")
        if not isinstance(self.setup, CopyFabricSetupLatencies):
            raise TypeError("setup must be CopyFabricSetupLatencies")
        object.__setattr__(
            self, "fabric_id", _required_text("fabric_id", self.fabric_id)
        )


@dataclass(frozen=True)
class AegisReservationHint:
    """The tier-local capacity reservation made before source production."""

    target_dp_rank: int
    hbm_reserved_tokens: int
    dram_reserved_tokens: int
    host_pool_node_id: str = ""
    version: int = 1
    route_version: int = 1
    request_backup_key: str = ""
    previous_target_dp_rank: int | None = None
    failed_target_dp_ranks: tuple[int, ...] = ()
    target_rotation_generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_dp_rank",
            _int_value("target_dp_rank", self.target_dp_rank),
        )
        object.__setattr__(
            self,
            "hbm_reserved_tokens",
            _int_value("hbm_reserved_tokens", self.hbm_reserved_tokens),
        )
        object.__setattr__(
            self,
            "dram_reserved_tokens",
            _int_value("dram_reserved_tokens", self.dram_reserved_tokens),
        )
        object.__setattr__(self, "version", _positive_int("version", self.version))
        object.__setattr__(
            self,
            "route_version",
            _positive_int("route_version", self.route_version),
        )
        object.__setattr__(self, "host_pool_node_id", str(self.host_pool_node_id))
        object.__setattr__(self, "request_backup_key", str(self.request_backup_key))
        if self.previous_target_dp_rank is not None:
            object.__setattr__(
                self,
                "previous_target_dp_rank",
                _int_value(
                    "previous_target_dp_rank", self.previous_target_dp_rank
                ),
            )
        object.__setattr__(
            self,
            "failed_target_dp_ranks",
            tuple(sorted({int(rank) for rank in self.failed_target_dp_ranks})),
        )
        object.__setattr__(
            self,
            "target_rotation_generation",
            _int_value(
                "target_rotation_generation", self.target_rotation_generation
            ),
        )
        if self.target_rotation_generation < 0:
            raise ValueError("target_rotation_generation must be nonnegative")
        if self.previous_target_dp_rank is not None:
            if self.previous_target_dp_rank == self.target_dp_rank:
                raise ValueError("target rotation must change the target DP")
            if self.previous_target_dp_rank not in self.failed_target_dp_ranks:
                raise ValueError(
                    "previous target must be fenced in failed_target_dp_ranks"
                )
            if self.target_dp_rank in self.failed_target_dp_ranks:
                raise ValueError("replacement target is already fenced")
            if self.target_rotation_generation <= 0:
                raise ValueError("target rotation requires a positive generation")
        if self.dram_reserved_tokens and not self.host_pool_node_id:
            raise ValueError(
                "host_pool_node_id is required when DRAM tokens are reserved"
            )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        event: Mapping[str, Any],
    ) -> "AegisReservationHint":
        if not isinstance(raw, Mapping):
            raise TypeError("reservation_hint must be a mapping")

        def reserved_tokens(
            explicit_key: str,
            committed_key: str,
            pending_key: str,
        ) -> Any:
            if explicit_key in raw:
                return raw[explicit_key]
            if committed_key in raw or pending_key in raw:
                return int(raw.get(committed_key, 0)) + int(raw.get(pending_key, 0))
            return 0

        host_candidate = raw.get("host_candidate")
        nested_host = (
            host_candidate.get("host_pool_node_id", "")
            if isinstance(host_candidate, Mapping)
            else ""
        )
        rotation = raw.get("hbm_target_rotation")
        rotation = dict(rotation) if isinstance(rotation, Mapping) else {}
        failed_targets = rotation.get(
            "failed_target_dp_ranks", raw.get("failed_target_dp_ranks", ())
        )
        if isinstance(failed_targets, (str, bytes)):
            failed_targets = ()
        return cls(
            target_dp_rank=raw.get(
                "target_dp_rank", event.get("target_dp_rank", -1)
            ),
            hbm_reserved_tokens=reserved_tokens(
                "reserved_l1_tokens", "l1_tokens", "hbm_pending_tokens"
            ),
            dram_reserved_tokens=reserved_tokens(
                "reserved_l2_tokens", "l2_tokens", "dram_pending_tokens"
            ),
            host_pool_node_id=str(
                raw.get("host_pool_node_id", nested_host)
            ),
            version=raw.get("version", event.get("version", 1)),
            route_version=raw.get(
                "route_version", event.get("route_version", 1)
            ),
            request_backup_key=str(
                raw.get(
                    "request_backup_key",
                    raw.get("backup_key", event.get("request_backup_key", "")),
                )
            ),
            previous_target_dp_rank=(
                rotation.get(
                    "previous_target_dp_rank",
                    raw.get("previous_target_dp_rank"),
                )
                if rotation
                else raw.get("previous_target_dp_rank")
            ),
            failed_target_dp_ranks=tuple(failed_targets or ()),
            target_rotation_generation=rotation.get(
                "generation", raw.get("target_rotation_generation", 0)
            ),
        )


@dataclass(frozen=True)
class PhysicalTransferJob:
    transfer_id: str
    group_id: str
    extent_id: str
    tier: StorageTier
    shard_key: str
    tp_rank: int
    token_start: int
    token_end: int
    size_bytes: int
    path_kind: CopyPathKind
    source: KvLocation
    destination: KvLocation
    ready_ns: int
    start_ns: int
    end_ns: int
    setup_ns: int
    bottleneck_bytes_per_s: int
    resources: tuple[TransferResourceId, ...]

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def queue_delay_ns(self) -> int:
        return self.start_ns - self.ready_ns

    def as_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "group_id": self.group_id,
            "extent_id": self.extent_id,
            "tier": self.tier.value,
            "shard_key": self.shard_key,
            "tp_rank": self.tp_rank,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "size_bytes": self.size_bytes,
            "path_kind": self.path_kind.value,
            "source": self.source.as_dict(),
            "destination": self.destination.as_dict(),
            "ready_ns": self.ready_ns,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "queue_delay_ns": self.queue_delay_ns,
            "setup_ns": self.setup_ns,
            "bottleneck_bytes_per_s": self.bottleneck_bytes_per_s,
            "resources": [resource.key for resource in self.resources],
        }


@dataclass(frozen=True)
class PhysicalRecoveryJob:
    """One replay-time ownership activation or contended recovery load."""

    recovery_id: str
    transfer_id: str
    group_id: str
    retry_request_id: str
    request_backup_key: str
    extent_id: str
    source_tier: StorageTier
    path_kind: CopyPathKind
    pp_rank: int
    tp_rank: int
    token_start: int
    token_end: int
    size_bytes: int
    source: KvLocation
    destination: KvLocation
    ready_ns: int
    start_ns: int
    end_ns: int
    setup_ns: int
    bottleneck_bytes_per_s: int
    resources: tuple[TransferResourceId, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "recovery_id": self.recovery_id,
            "transfer_id": self.transfer_id,
            "group_id": self.group_id,
            "retry_request_id": self.retry_request_id,
            "request_backup_key": self.request_backup_key,
            "extent_id": self.extent_id,
            "source_tier": self.source_tier.value,
            "path_kind": self.path_kind.value,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "size_bytes": self.size_bytes,
            "source": self.source.as_dict(),
            "destination": self.destination.as_dict(),
            "ready_ns": self.ready_ns,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.end_ns - self.start_ns,
            "queue_delay_ns": self.start_ns - self.ready_ns,
            "setup_ns": self.setup_ns,
            "bottleneck_bytes_per_s": self.bottleneck_bytes_per_s,
            "resources": [resource.key for resource in self.resources],
        }


@dataclass(frozen=True)
class TierNotification:
    group_id: str
    tier: StorageTier
    state: CommitGroupState
    success: bool
    request_backup_key: str
    version: int
    route_version: int
    token_start: int
    token_end: int
    expected_shards: int
    acked_shards: int
    timestamp_ns: int
    reason: str
    transfer_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "tier": self.tier.value,
            "state": self.state.value,
            "success": self.success,
            "request_backup_key": self.request_backup_key,
            "version": self.version,
            "route_version": self.route_version,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "expected_shards": self.expected_shards,
            "acked_shards": self.acked_shards,
            "timestamp_ns": self.timestamp_ns,
            "reason": self.reason,
            "transfer_ids": list(self.transfer_ids),
        }


TierNotificationCallback = Callable[[TierNotification], None]


@dataclass(frozen=True)
class TierGroupPlan:
    group_id: str
    tier: StorageTier
    token_start: int
    token_end: int
    transfer_ids: tuple[str, ...]
    notify_at_ns: int
    admitted: bool
    admission_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "tier": self.tier.value,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "transfer_ids": list(self.transfer_ids),
            "notify_at_ns": self.notify_at_ns,
            "admitted": self.admitted,
            "admission_error": self.admission_error,
        }


@dataclass(frozen=True)
class _TierGroupOwner:
    """Version-safe logical owner for one physical tier commit group."""

    group_id: str
    request_backup_key: str
    version: int
    route_version: int
    source_dp_rank: int
    target_dp_rank: int
    pp_rank: int
    tier: StorageTier
    token_start: int
    token_end: int


@dataclass(frozen=True)
class ChunkTransferPlan:
    request_backup_key: str
    source_dp_rank: int
    target_dp_rank: int
    pp_rank: int
    version: int
    route_version: int
    source_token_start: int
    source_token_end: int
    duplicate: bool
    extent_ids: tuple[str, ...] = ()
    tier_groups: tuple[TierGroupPlan, ...] = ()

    @property
    def transfer_ids(self) -> tuple[str, ...]:
        return tuple(
            transfer_id
            for group in self.tier_groups
            for transfer_id in group.transfer_ids
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_backup_key": self.request_backup_key,
            "source_dp_rank": self.source_dp_rank,
            "target_dp_rank": self.target_dp_rank,
            "pp_rank": self.pp_rank,
            "version": self.version,
            "route_version": self.route_version,
            "source_token_start": self.source_token_start,
            "source_token_end": self.source_token_end,
            "duplicate": self.duplicate,
            "extent_ids": list(self.extent_ids),
            "tier_groups": [group.as_dict() for group in self.tier_groups],
            "transfer_ids": list(self.transfer_ids),
        }


@dataclass(frozen=True)
class _FrontierState:
    version: int
    route_version: int
    source_token_end: int
    hbm_source_token_end: int
    dram_source_token_end: int
    execution_id: str
    block_generation: int
    target_dp_rank: int
    host_pool_node_id: str
    hbm_reserved_tokens: int
    dram_reserved_tokens: int
    model_revision: str
    kv_abi: str
    parallel_layout: str
    target_rotation_generation: int = 0


@dataclass(frozen=True)
class _PathReservation:
    kind: CopyPathKind
    resources: tuple[TransferResourceId, ...]
    setup_ns: int
    bottleneck_bytes_per_s: int

    def duration_ns(self, size_bytes: int) -> int:
        size = _positive_int("size_bytes", size_bytes)
        transfer_ns = (
            size * 1_000_000_000 + self.bottleneck_bytes_per_s - 1
        ) // self.bottleneck_bytes_per_s
        return self.setup_ns + transfer_ns


@dataclass(frozen=True)
class _RecoveryTransferSpec:
    recovery_id: str
    shard_key: str
    source_job: PhysicalTransferJob
    pp_rank: int
    tp_rank: int
    token_start: int
    token_end: int
    size_bytes: int
    source: KvLocation
    destination: KvLocation
    path: _PathReservation
    destination_mode: TransferDestinationMode


class OfflineAegisTransferBridge:
    """Translate Scheduler frontier events into contended physical KV copies."""

    def __init__(
        self,
        des: GlobalDES,
        topology: OfflinePhysicalTopology,
        *,
        fabric: CopyFabricConfig | None = None,
    ) -> None:
        if not isinstance(des, GlobalDES):
            raise TypeError("des must be GlobalDES")
        if not isinstance(topology, OfflinePhysicalTopology):
            raise TypeError("topology must be OfflinePhysicalTopology")
        self.des = des
        self.topology = topology
        self.fabric = fabric or CopyFabricConfig()
        self._frontiers: dict[tuple[str, int, int], _FrontierState] = {}
        self._jobs: list[PhysicalTransferJob] = []
        self._recovery_jobs: list[PhysicalRecoveryJob] = []
        self._recovery_event_ids: dict[str, tuple[str, str]] = {}
        self._recovery_outcomes: dict[str, str] = {}
        self._notifications: list[TierNotification] = []
        self._audit: list[dict[str, Any]] = []
        self._group_owners: dict[str, _TierGroupOwner] = {}
        self._released_group_destinations: set[str] = set()
        self._group_failure_reasons: dict[str, str] = {}
        self._notification_groups: set[str] = set()
        self._validate_registered_topology()

    @property
    def jobs(self) -> tuple[PhysicalTransferJob, ...]:
        return tuple(self._jobs)

    @property
    def tier_notifications(self) -> tuple[TierNotification, ...]:
        return tuple(self._notifications)

    @property
    def recovery_jobs(self) -> tuple[PhysicalRecoveryJob, ...]:
        return tuple(self._recovery_jobs)

    @property
    def audit(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._audit))

    def _refresh_recovery_outcomes(self) -> None:
        """Project canonical GlobalDES states onto the v1 compatibility view."""

        compatibility = {
            TransferState.QUEUED: "QUEUED",
            TransferState.IN_FLIGHT: "IN_FLIGHT",
            TransferState.ACKED: "IN_FLIGHT",
            TransferState.COMMITTED: "COMPLETED",
            TransferState.ABORTED: "ABORTED",
        }
        for job in self._recovery_jobs:
            transfer = self.des.transfers.get(job.transfer_id)
            if transfer is not None:
                self._recovery_outcomes[job.recovery_id] = compatibility[
                    transfer.state
                ]

    def export(self) -> dict[str, Any]:
        """Return a defensive, JSON-compatible physical-copy artifact.

        The authoritative transfer and commit-group state remains in
        :class:`GlobalDES`; this view simply keeps the bridge plan, tier
        notifications, and audit trail together for an offline experiment.
        """

        group_ids = sorted(self._notification_groups)
        self._refresh_recovery_outcomes()
        return copy.deepcopy(
            {
                "schema": OFFLINE_AEGIS_TRANSFER_BRIDGE_SCHEMA,
                "frontiers": [
                    {
                        "request_backup_key": key[0],
                        "source_dp_rank": key[1],
                        "pp_rank": key[2],
                        **{
                            field_name: getattr(state, field_name)
                            for field_name in state.__dataclass_fields__
                        },
                    }
                    for key, state in sorted(self._frontiers.items())
                ],
                "jobs": [job.as_dict() for job in self._jobs],
                "recovery_jobs": [
                    job.as_dict() for job in self._recovery_jobs
                ],
                "recovery_outcomes": dict(sorted(self._recovery_outcomes.items())),
                "groups": [
                    self.des.commit_groups[group_id].snapshot()
                    for group_id in group_ids
                    if group_id in self.des.commit_groups
                ],
                "group_owners": [
                    {
                        **{
                            field_name: getattr(owner, field_name)
                            for field_name in owner.__dataclass_fields__
                            if field_name != "tier"
                        },
                        "tier": owner.tier.value,
                        "released": owner.group_id
                        in self._released_group_destinations,
                    }
                    for owner in sorted(
                        self._group_owners.values(),
                        key=lambda item: (
                            item.request_backup_key,
                            item.version,
                            item.pp_rank,
                            item.token_start,
                            item.tier.value,
                        ),
                    )
                ],
                "notifications": [
                    notification.as_dict()
                    for notification in self._notifications
                ],
                "audit": self._audit,
            }
        )

    @staticmethod
    def _root_backup_key(backup_key: str) -> tuple[str, int | None]:
        """Normalize a PP-local manager key to its request-level identity."""

        root, marker, suffix = str(backup_key).rpartition("::pp")
        if marker and root and suffix.isdigit():
            return root, int(suffix)
        return str(backup_key), None

    def request_groups_terminal(self, request_backup_key: str, version: int) -> bool:
        """Return whether every registered tier group for a version is terminal."""

        key = str(request_backup_key)
        for owner in self._group_owners.values():
            if owner.request_backup_key != key or owner.version != int(version):
                continue
            group = self.des.commit_groups.get(owner.group_id)
            if group is not None and group.state is CommitGroupState.PENDING:
                return False
        # No protection group is a valid BYPASS completion.
        return True

    def materialized_source_frontier_token_end(
        self,
        *,
        request_backup_key: str,
        source_dp_rank: int,
        pp_rank: int,
        version: int,
        route_version: int,
        execution_id: str,
    ) -> int:
        """Return the already-materialized frontier for one exact lineage.

        The offline worker can report a terminal, token-granular frontier just
        before SGLang's finished-request cache path releases its final partial
        page.  The cluster uses this read-only view when projecting that event
        onto the post-response physical allocator: a previously copied partial
        page remains valid, while a never-materialized tail must be recomputed.
        """

        key = (
            str(request_backup_key),
            _int_value("source_dp_rank", source_dp_rank),
            _int_value("pp_rank", pp_rank),
        )
        previous = self._frontiers.get(key)
        if previous is None:
            return 0
        if (
            previous.version != _positive_int("version", version)
            or previous.route_version
            != _positive_int("route_version", route_version)
            or previous.execution_id != _required_text(
                "execution_id", execution_id
            )
        ):
            return 0
        return int(previous.source_token_end)

    def release_owned_copies(
        self,
        *,
        request_backup_key: str,
        version: int,
        tier: StorageTier | str | None = None,
        target_dp_rank: int | None = None,
        pp_rank: int | None = None,
        ordinal: int | None = None,
        token_start: int | None = None,
        token_end: int | None = None,
        reason: str,
    ) -> tuple[dict[str, Any], ...]:
        """Release matching peer-HBM or Store-DRAM destinations exactly once.

        The logical manager owns eviction policy; this bridge owns byte-exact
        physical locations.  A release therefore selects a versioned,
        tier-local chunk and never infers a cross-tier migration.  Releasing a
        pending destination first aborts its own commit group through
        :meth:`GlobalDES.release_extent_location`.
        """

        root_key, key_pp_rank = self._root_backup_key(request_backup_key)
        selected_pp_rank = key_pp_rank if pp_rank is None else int(pp_rank)
        selected_tier = None if tier is None else StorageTier(tier)
        if selected_tier not in {
            None,
            StorageTier.GPU_AEGIS,
            StorageTier.NODE_DRAM,
        }:
            raise ValueError("physical Aegis release must target HBM or DRAM")
        selected_version = _positive_int("version", version)
        selected_target = (
            None
            if target_dp_rank is None
            else _int_value("target_dp_rank", target_dp_rank)
        )
        start = None if token_start is None else _int_value("token_start", token_start)
        end = None if token_end is None else _positive_int("token_end", token_end)
        if (start is None) != (end is None):
            raise ValueError("token_start and token_end must be supplied together")
        if start is not None and end <= start:
            raise ValueError("physical Aegis release token range is invalid")

        rows: list[dict[str, Any]] = []
        jobs_by_group: dict[str, list[PhysicalTransferJob]] = {}
        for job in self._jobs:
            jobs_by_group.setdefault(job.group_id, []).append(job)
        for owner in sorted(
            self._group_owners.values(),
            key=lambda item: (item.pp_rank, item.token_start, item.tier.value),
        ):
            if (
                owner.request_backup_key != root_key
                or owner.version != selected_version
                or (
                    selected_tier is not None
                    and owner.tier is not selected_tier
                )
                or (
                    selected_target is not None
                    and owner.target_dp_rank != selected_target
                )
                or (
                    selected_pp_rank is not None
                    and owner.pp_rank != selected_pp_rank
                )
                or (
                    start is not None
                    and (
                        owner.token_start != start
                        or owner.token_end != end
                    )
                )
            ):
                continue
            if owner.group_id in self._released_group_destinations:
                continue
            released_locations = 0
            released_bytes = 0
            old_states: list[str] = []
            for job in jobs_by_group.get(owner.group_id, ()):
                extent = self.des.extents.get(job.extent_id)
                if extent is None or job.destination not in extent.locations:
                    continue
                old_state = self.des.release_extent_location(
                    job.extent_id,
                    job.destination,
                    reason=str(reason),
                )
                released_locations += 1
                released_bytes += int(job.size_bytes)
                old_states.append(old_state.value)
            self._released_group_destinations.add(owner.group_id)
            row = {
                "group_id": owner.group_id,
                "request_backup_key": owner.request_backup_key,
                "version": owner.version,
                "route_version": owner.route_version,
                "source_dp_rank": owner.source_dp_rank,
                "target_dp_rank": owner.target_dp_rank,
                "pp_rank": owner.pp_rank,
                "tier": owner.tier.value,
                "ordinal": ordinal,
                "token_start": owner.token_start,
                "token_end": owner.token_end,
                "released_locations": released_locations,
                "released_bytes": released_bytes,
                "old_states": old_states,
                "reason": str(reason),
                "timestamp_ns": self.des.now_ns,
            }
            rows.append(row)
            self._record("physical_copy_released", **row)
        return tuple(rows)

    def apply_release_intents(
        self, intents: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        """Apply rank-manager lifecycle releases to physical destinations."""

        rows: list[dict[str, Any]] = []
        tier_aliases = {
            "HBM": StorageTier.GPU_AEGIS,
            "DRAM": StorageTier.NODE_DRAM,
            "gpu_aegis": StorageTier.GPU_AEGIS,
            "node_dram": StorageTier.NODE_DRAM,
            "hbm": StorageTier.GPU_AEGIS,
            "dram": StorageTier.NODE_DRAM,
        }
        for raw in intents:
            if not isinstance(raw, Mapping):
                continue
            raw_tier = raw.get("tier")
            selected_tier = None
            if raw_tier is not None:
                selected_tier = tier_aliases.get(str(raw_tier))
                if selected_tier is None:
                    raise ValueError(f"unknown physical release tier {raw_tier!r}")
            rows.extend(
                self.release_owned_copies(
                    request_backup_key=_required_text(
                        "request_backup_key", raw.get("request_backup_key")
                    ),
                    version=_positive_int("version", raw.get("version")),
                    tier=selected_tier,
                    target_dp_rank=(
                        None
                        if raw.get("target_dp_rank") is None
                        else _int_value(
                            "target_dp_rank", raw.get("target_dp_rank")
                        )
                    ),
                    pp_rank=(
                        None
                        if raw.get("pp_rank") is None
                        else _int_value("pp_rank", raw.get("pp_rank"))
                    ),
                    ordinal=(
                        None
                        if raw.get("ordinal") is None
                        else _int_value("ordinal", raw.get("ordinal"))
                    ),
                    token_start=raw.get("token_start"),
                    token_end=raw.get("token_end"),
                    reason=str(raw.get("reason") or "logical_owner_released"),
                )
            )
        return tuple(rows)

    def _record(self, kind: str, **fields: Any) -> None:
        self._audit.append(
            {
                "schema": OFFLINE_AEGIS_TRANSFER_BRIDGE_SCHEMA,
                "sequence": len(self._audit),
                "timestamp_ns": self.des.now_ns,
                "kind": str(kind),
                **copy.deepcopy(fields),
            }
        )

    def _validate_registered_topology(self) -> None:
        expected_gpus = {spec.gpu for spec in self.topology.gpus}
        expected_nodes = {spec.node_id for spec in self.topology.nodes}
        missing_gpus = expected_gpus - set(self.des.gpu_hbm)
        missing_nodes = expected_nodes - set(self.des.node_dram)
        if missing_gpus or missing_nodes:
            raise ValueError(
                "GlobalDES physical ledgers must be registered before the Aegis "
                f"bridge (missing_gpus={len(missing_gpus)}, "
                f"missing_nodes={sorted(missing_nodes)})"
            )

    def _resolve_node(self, value: str) -> str:
        known = tuple(sorted(node.node_id for node in self.topology.nodes))
        raw = str(value)
        if raw in known:
            return raw
        try:
            index = int(raw)
        except (TypeError, ValueError):
            index = -1
        if str(index) == raw and index >= 0:
            for candidate in (f"node-{index}", f"node{index}"):
                if candidate in known:
                    return candidate
            if index < len(known):
                return known[index]
        raise ValueError(f"unknown host_pool_node_id {raw!r}; known={list(known)}")

    @staticmethod
    def _normalize_kv_bytes(
        raw: Mapping[int | str, int], expected_tp_ranks: Sequence[int]
    ) -> dict[int, int]:
        if not isinstance(raw, Mapping):
            raise TypeError("kv_bytes_per_token_by_tp_rank must be a mapping")
        parsed: dict[int, int] = {}
        for key, value in raw.items():
            rank = _int_value("TP-rank key", key)
            if rank in parsed:
                raise ValueError(f"duplicate TP-rank byte entry {rank}")
            parsed[rank] = _positive_int(
                f"kv_bytes_per_token_by_tp_rank[{rank}]", value
            )
        expected = set(int(rank) for rank in expected_tp_ranks)
        if set(parsed) != expected:
            raise ValueError(
                "KV byte table must exactly cover physical TP ranks; "
                f"expected={sorted(expected)}, got={sorted(parsed)}"
            )
        return parsed

    @staticmethod
    def _parse_event(event: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise TypeError("source_chunk_committed event must be a mapping")
        event_type = str(event.get("type", "aegis_source_chunk_committed"))
        if event_type not in {
            "aegis_source_chunk_committed",
            "source_chunk_committed",
        }:
            raise ValueError(f"unexpected source frontier event type {event_type!r}")
        source_pp = _int_value("source_pp_rank", event.get("source_pp_rank", 0))
        target_pp = _int_value(
            "target_pp_rank", event.get("target_pp_rank", source_pp)
        )
        if source_pp != target_pp:
            raise ValueError("source and target Aegis copies must use the same PP stage")
        source_end = _int_value(
            "source_committed_token_end",
            event.get("source_committed_token_end", -1),
        )
        reserved = _int_value(
            "reserved_token_count", event.get("reserved_token_count", source_end)
        )
        if source_end > reserved:
            raise ValueError("source frontier exceeds its admission-time reservation")
        fallback_chunk = max(source_end, reserved, 1)
        prefix_chunk_tokens = _positive_int(
            "prefix_chunk_tokens",
            event.get(
                "prefix_chunk_tokens",
                event.get("chunk_tokens", fallback_chunk),
            ),
        )
        decode_chunk_tokens = _positive_int(
            "decode_chunk_tokens",
            event.get(
                "decode_chunk_tokens",
                event.get("chunk_tokens", fallback_chunk),
            ),
        )
        return {
            "request_backup_key": _required_text(
                "request_backup_key",
                event.get("request_backup_key", event.get("backup_key", "")),
            ),
            "rid": str(event.get("rid", "")),
            "source_dp_rank": _int_value(
                "source_dp_rank", event.get("source_dp_rank", -1)
            ),
            "target_dp_rank": _int_value(
                "target_dp_rank", event.get("target_dp_rank", -1)
            ),
            "pp_rank": source_pp,
            "required_pp_size": _positive_int(
                "required_pp_size", event.get("required_pp_size", 1)
            ),
            "version": _positive_int("version", event.get("version", 1)),
            "route_version": _positive_int(
                "route_version", event.get("route_version", 1)
            ),
            "source_token_end": source_end,
            "reserved_token_count": reserved,
            "prefix_chunk_tokens": prefix_chunk_tokens,
            "decode_chunk_tokens": decode_chunk_tokens,
            "execution_id": _required_text(
                "execution_id", event.get("execution_id", "")
            ),
            "block_generation": _int_value(
                "block_generation", event.get("block_generation", 0)
            ),
            "ready_ns": _int_value(
                "timestamp_ns", event.get("timestamp_ns", 0)
            )
            if "timestamp_ns" in event
            else None,
        }

    def consume_source_chunk_committed(
        self,
        event: Mapping[str, Any],
        reservation_hint: Mapping[str, Any] | AegisReservationHint,
        *,
        kv_bytes_per_token_by_tp_rank: Mapping[int | str, int],
        model_revision: str,
        kv_abi: str,
        parallel_layout: str,
        on_tier_notification: TierNotificationCallback | None = None,
    ) -> ChunkTransferPlan:
        """Plan physical copies for one cumulative Scheduler source frontier."""

        parsed = self._parse_event(event)
        hint = (
            reservation_hint
            if isinstance(reservation_hint, AegisReservationHint)
            else AegisReservationHint.from_mapping(reservation_hint, event=event)
        )
        model = _required_text("model_revision", model_revision)
        abi = _required_text("kv_abi", kv_abi)
        layout = _required_text("parallel_layout", parallel_layout)
        source_event_target = int(parsed["target_dp_rank"])
        target_rotated = hint.target_dp_rank != source_event_target
        if target_rotated:
            if not (
                hint.previous_target_dp_rank is not None
                and source_event_target in hint.failed_target_dp_ranks
                and hint.previous_target_dp_rank in hint.failed_target_dp_ranks
                and hint.target_rotation_generation > 0
            ):
                raise ValueError("reservation target DP differs from source event")
            parsed["target_dp_rank"] = hint.target_dp_rank
        if hint.version != parsed["version"]:
            raise ValueError("reservation version differs from source event")
        if hint.route_version != parsed["route_version"]:
            raise ValueError("reservation route_version differs from source event")
        if (
            hint.request_backup_key
            and hint.request_backup_key != parsed["request_backup_key"]
        ):
            raise ValueError("reservation backup key differs from source event")
        if parsed["source_dp_rank"] == parsed["target_dp_rank"]:
            raise ValueError("Aegis target DP must differ from source DP")

        source_group = self.topology.group(
            parsed["source_dp_rank"], parsed["pp_rank"]
        )
        target_group = self.topology.group(
            parsed["target_dp_rank"], parsed["pp_rank"]
        )
        if len(source_group) != len(target_group):
            raise ValueError("source and target physical TP groups differ in size")
        pp_count = len(
            {
                pp
                for dp, pp in self.topology.rank_groups
                if dp == parsed["source_dp_rank"]
            }
        )
        if parsed["required_pp_size"] != pp_count:
            raise ValueError(
                "source event PP size differs from physical source topology"
            )
        target_pp_count = len(
            {
                pp
                for dp, pp in self.topology.rank_groups
                if dp == parsed["target_dp_rank"]
            }
        )
        if target_pp_count != pp_count:
            raise ValueError("source and target DP replicas have different PP size")
        expected_tp = [spec.gpu.tp_rank for spec in source_group]
        if [spec.gpu.tp_rank for spec in target_group] != expected_tp:
            raise ValueError("source and target TP-rank coordinates differ")
        kv_bytes = self._normalize_kv_bytes(
            kv_bytes_per_token_by_tp_rank, expected_tp
        )
        host_node = (
            self._resolve_node(hint.host_pool_node_id)
            if hint.dram_reserved_tokens
            else ""
        )
        ready_ns = (
            self.des.now_ns if parsed["ready_ns"] is None else parsed["ready_ns"]
        )
        if ready_ns < self.des.now_ns:
            raise ValueError("source frontier timestamp precedes the global DES cursor")

        key = (
            parsed["request_backup_key"],
            parsed["source_dp_rank"],
            parsed["pp_rank"],
        )
        previous = self._frontiers.get(key)
        frontier = self._validate_frontier(
            previous,
            parsed=parsed,
            target_dp_rank=hint.target_dp_rank,
            host_node=host_node,
            hbm_reserved_tokens=hint.hbm_reserved_tokens,
            dram_reserved_tokens=hint.dram_reserved_tokens,
            model_revision=model,
            kv_abi=abi,
            parallel_layout=layout,
            previous_target_dp_rank=hint.previous_target_dp_rank,
            failed_target_dp_ranks=hint.failed_target_dp_ranks,
            target_rotation_generation=hint.target_rotation_generation,
        )
        source_start = int(frontier["source_start"])
        hbm_start = int(frontier["hbm_start"])
        dram_start = int(frontier["dram_start"])
        rotation_applied = bool(frontier["rotation_applied"])
        if rotation_applied:
            assert previous is not None
            released = self.release_owned_copies(
                request_backup_key=parsed["request_backup_key"],
                version=parsed["version"],
                tier=StorageTier.GPU_AEGIS,
                target_dp_rank=previous.target_dp_rank,
                pp_rank=parsed["pp_rank"],
                reason="bound_hbm_target_failed",
            )
            self._record(
                "standby_target_rotated",
                request_backup_key=parsed["request_backup_key"],
                version=parsed["version"],
                route_version=parsed["route_version"],
                source_dp_rank=parsed["source_dp_rank"],
                previous_target_dp_rank=previous.target_dp_rank,
                replacement_target_dp_rank=hint.target_dp_rank,
                failed_target_dp_ranks=list(hint.failed_target_dp_ranks),
                generation=hint.target_rotation_generation,
                released_hbm_group_ids=[row["group_id"] for row in released],
                dram_frontier_preserved=dram_start,
                route_version_changed=False,
            )
        elif target_rotated:
            self._record(
                "stale_source_target_rewritten",
                request_backup_key=parsed["request_backup_key"],
                version=parsed["version"],
                route_version=parsed["route_version"],
                source_event_target_dp_rank=source_event_target,
                replacement_target_dp_rank=hint.target_dp_rank,
                generation=hint.target_rotation_generation,
            )

        hbm_done = (
            hint.hbm_reserved_tokens <= 0
            or min(parsed["source_token_end"], hint.hbm_reserved_tokens)
            <= hbm_start
        )
        dram_done = (
            hint.dram_reserved_tokens <= 0
            or min(parsed["source_token_end"], hint.dram_reserved_tokens)
            <= dram_start
        )
        if previous is not None and hbm_done and dram_done:
            if rotation_applied:
                self._frontiers[key] = replace(
                    previous,
                    source_token_end=max(
                        previous.source_token_end,
                        int(parsed["source_token_end"]),
                    ),
                    hbm_source_token_end=0,
                    target_dp_rank=hint.target_dp_rank,
                    hbm_reserved_tokens=hint.hbm_reserved_tokens,
                    target_rotation_generation=(
                        hint.target_rotation_generation
                    ),
                )
            self._record(
                "duplicate_frontier_ignored",
                request_backup_key=parsed["request_backup_key"],
                source_dp_rank=parsed["source_dp_rank"],
                pp_rank=parsed["pp_rank"],
                version=parsed["version"],
                source_token_end=parsed["source_token_end"],
            )
            return ChunkTransferPlan(
                request_backup_key=parsed["request_backup_key"],
                source_dp_rank=parsed["source_dp_rank"],
                target_dp_rank=parsed["target_dp_rank"],
                pp_rank=parsed["pp_rank"],
                version=parsed["version"],
                route_version=parsed["route_version"],
                source_token_start=source_start,
                source_token_end=parsed["source_token_end"],
                duplicate=True,
            )

        active_starts = [
            start
            for start, reserved in (
                (hbm_start, hint.hbm_reserved_tokens),
                (dram_start, hint.dram_reserved_tokens),
            )
            if reserved > start and parsed["source_token_end"] > start
        ]
        schedule_start = min(active_starts, default=parsed["source_token_end"])
        boundaries = {schedule_start, parsed["source_token_end"]}
        boundaries.update(
            start
            for start in (hbm_start, dram_start)
            if schedule_start < start < parsed["source_token_end"]
        )
        prefix_end = min(
            int(parsed["prefix_chunk_tokens"]),
            int(parsed["reserved_token_count"]),
        )
        if schedule_start < prefix_end < parsed["source_token_end"]:
            boundaries.add(prefix_end)
        decode_chunk = int(parsed["decode_chunk_tokens"])
        boundary = prefix_end
        if boundary < schedule_start:
            boundary += (
                (schedule_start - boundary + decode_chunk - 1) // decode_chunk
            ) * decode_chunk
        while boundary < parsed["source_token_end"]:
            if schedule_start < boundary:
                boundaries.add(boundary)
            boundary += decode_chunk
        for limit in (hint.hbm_reserved_tokens, hint.dram_reserved_tokens):
            if schedule_start < limit < parsed["source_token_end"]:
                boundaries.add(limit)
        ordered = sorted(boundaries)
        extent_ids: list[str] = []
        group_plans: list[TierGroupPlan] = []
        for token_start, token_end in zip(ordered, ordered[1:]):
            if token_end <= token_start:
                continue
            tiers: list[StorageTier] = []
            if hbm_start <= token_start and token_end <= hint.hbm_reserved_tokens:
                tiers.append(StorageTier.GPU_AEGIS)
            if dram_start <= token_start and token_end <= hint.dram_reserved_tokens:
                tiers.append(StorageTier.NODE_DRAM)
            if not tiers:
                self._record(
                    "source_range_unprotected",
                    request_backup_key=parsed["request_backup_key"],
                    token_start=token_start,
                    token_end=token_end,
                    reason="outside_tier_reservations",
                )
                continue

            extents = self._register_source_extents(
                parsed=parsed,
                token_start=token_start,
                token_end=token_end,
                source_group=source_group,
                kv_bytes=kv_bytes,
                model_revision=model,
                kv_abi=abi,
                parallel_layout=layout,
            )
            extent_ids.extend(extent_id for _gpu, extent_id in extents)
            for tier in tiers:
                group_plans.append(
                    self._schedule_tier_group(
                        parsed=parsed,
                        tier=tier,
                        token_start=token_start,
                        token_end=token_end,
                        extents=extents,
                        target_group=target_group,
                        host_node=host_node,
                        ready_ns=ready_ns,
                        model_revision=model,
                        kv_abi=abi,
                        parallel_layout=layout,
                        on_tier_notification=on_tier_notification,
                    )
                )

        # Publish the new cumulative frontier only after all physical source
        # aliases and tier-local plans were created successfully.  A malformed
        # input can therefore be corrected and retried instead of becoming an
        # idempotent-looking but silently lost update.
        self._frontiers[key] = _FrontierState(
            version=parsed["version"],
            route_version=parsed["route_version"],
            source_token_end=parsed["source_token_end"],
            hbm_source_token_end=max(
                hbm_start,
                min(parsed["source_token_end"], hint.hbm_reserved_tokens),
            ),
            dram_source_token_end=max(
                dram_start,
                min(parsed["source_token_end"], hint.dram_reserved_tokens),
            ),
            execution_id=parsed["execution_id"],
            block_generation=parsed["block_generation"],
            target_dp_rank=hint.target_dp_rank,
            host_pool_node_id=host_node,
            hbm_reserved_tokens=hint.hbm_reserved_tokens,
            dram_reserved_tokens=hint.dram_reserved_tokens,
            model_revision=model,
            kv_abi=abi,
            parallel_layout=layout,
            target_rotation_generation=hint.target_rotation_generation,
        )
        self._record(
            "source_frontier_materialized",
            request_backup_key=parsed["request_backup_key"],
            source_dp_rank=parsed["source_dp_rank"],
            target_dp_rank=parsed["target_dp_rank"],
            pp_rank=parsed["pp_rank"],
            version=parsed["version"],
            route_version=parsed["route_version"],
            source_token_start=schedule_start,
            source_token_end=parsed["source_token_end"],
            extent_ids=list(extent_ids),
            group_ids=[group.group_id for group in group_plans],
        )
        return ChunkTransferPlan(
            request_backup_key=parsed["request_backup_key"],
            source_dp_rank=parsed["source_dp_rank"],
            target_dp_rank=parsed["target_dp_rank"],
            pp_rank=parsed["pp_rank"],
            version=parsed["version"],
            route_version=parsed["route_version"],
            source_token_start=schedule_start,
            source_token_end=parsed["source_token_end"],
            duplicate=False,
            extent_ids=tuple(extent_ids),
            tier_groups=tuple(group_plans),
        )

    def _validate_frontier(
        self,
        previous: _FrontierState | None,
        *,
        parsed: Mapping[str, Any],
        target_dp_rank: int,
        host_node: str,
        hbm_reserved_tokens: int,
        dram_reserved_tokens: int,
        model_revision: str,
        kv_abi: str,
        parallel_layout: str,
        previous_target_dp_rank: int | None,
        failed_target_dp_ranks: tuple[int, ...],
        target_rotation_generation: int,
    ) -> dict[str, int | bool]:
        if previous is None:
            return {
                "source_start": 0,
                "hbm_start": 0,
                "dram_start": 0,
                "rotation_applied": False,
            }
        if parsed["version"] < previous.version:
            raise ValueError("source frontier version regressed")
        if parsed["route_version"] < previous.route_version:
            raise ValueError("source frontier route_version regressed")
        for name, value in {
            "model_revision": model_revision,
            "kv_abi": kv_abi,
            "parallel_layout": parallel_layout,
        }.items():
            if value != getattr(previous, name):
                raise ValueError(f"source frontier {name} changed across versions")
        if parsed["version"] > previous.version:
            return {
                "source_start": 0,
                "hbm_start": 0,
                "dram_start": 0,
                "rotation_applied": False,
            }
        fields = {
            "route_version": parsed["route_version"],
            "execution_id": parsed["execution_id"],
            "block_generation": parsed["block_generation"],
            "host_pool_node_id": host_node,
        }
        for name, value in fields.items():
            if value != getattr(previous, name):
                raise ValueError(f"source frontier {name} changed within one version")
        rotation_applied = target_dp_rank != previous.target_dp_rank
        if rotation_applied:
            if not (
                previous_target_dp_rank == previous.target_dp_rank
                and previous.target_dp_rank in failed_target_dp_ranks
                and target_dp_rank not in failed_target_dp_ranks
                and target_rotation_generation
                > previous.target_rotation_generation
            ):
                raise ValueError(
                    "source frontier target changed without a newer fenced rotation"
                )
        elif target_rotation_generation < previous.target_rotation_generation:
            raise ValueError("source frontier target rotation generation regressed")
        rotation_hbm_activated = (
            not rotation_applied
            and target_rotation_generation > 0
            and target_rotation_generation == previous.target_rotation_generation
            and previous.hbm_reserved_tokens == 0
            and hbm_reserved_tokens > 0
            and previous_target_dp_rank in failed_target_dp_ranks
        )
        # A target or host failure can independently invalidate one protection
        # tier while the source Scheduler is still publishing committed chunks.
        # That is a monotonic contraction of the same version, not a new route
        # and never a target-owned destage.  HBM may re-expand only when the
        # Router has atomically fenced the old target and bound a replacement;
        # DRAM remains independently monotonic within the route version.
        for name, value in {
            "hbm_reserved_tokens": hbm_reserved_tokens,
            "dram_reserved_tokens": dram_reserved_tokens,
        }.items():
            previous_value = int(getattr(previous, name))
            if int(value) > previous_value and not (
                name == "hbm_reserved_tokens"
                and (rotation_applied or rotation_hbm_activated)
            ):
                raise ValueError(
                    f"source frontier {name} expanded within one version"
                )
            if int(value) < previous_value:
                self._record(
                    "tier_reservation_contracted",
                    request_backup_key=parsed["request_backup_key"],
                    version=parsed["version"],
                    route_version=parsed["route_version"],
                    tier=(
                        StorageTier.GPU_AEGIS.value
                        if name == "hbm_reserved_tokens"
                        else StorageTier.NODE_DRAM.value
                    ),
                    previous_reserved_tokens=previous_value,
                    reserved_tokens=int(value),
                    source_token_end=parsed["source_token_end"],
                    timestamp_ns=self.des.now_ns,
                )
        if rotation_hbm_activated:
            self._record(
                "rotation_hbm_reservation_activated",
                request_backup_key=parsed["request_backup_key"],
                version=parsed["version"],
                route_version=parsed["route_version"],
                target_dp_rank=target_dp_rank,
                generation=target_rotation_generation,
                reserved_tokens=hbm_reserved_tokens,
                source_token_end=parsed["source_token_end"],
                timestamp_ns=self.des.now_ns,
            )
        if parsed["source_token_end"] < previous.source_token_end:
            raise ValueError("source committed frontier regressed")
        return {
            "source_start": previous.source_token_end,
            "hbm_start": (
                0 if rotation_applied else previous.hbm_source_token_end
            ),
            "dram_start": previous.dram_source_token_end,
            "rotation_applied": rotation_applied,
        }

    def _register_source_extents(
        self,
        *,
        parsed: Mapping[str, Any],
        token_start: int,
        token_end: int,
        source_group: Sequence[Any],
        kv_bytes: Mapping[int, int],
        model_revision: str,
        kv_abi: str,
        parallel_layout: str,
    ) -> tuple[tuple[PhysicalGpuId, str], ...]:
        result: list[tuple[PhysicalGpuId, str]] = []
        for spec in source_group:
            gpu = spec.gpu
            extent_id = _stable_id(
                "aegis-extent",
                parsed["request_backup_key"],
                parsed["version"],
                parsed["route_version"],
                parsed["execution_id"],
                parsed["pp_rank"],
                gpu.tp_rank,
                token_start,
                token_end,
            )
            size_bytes = (token_end - token_start) * kv_bytes[gpu.tp_rank]
            source = KvLocation.gpu_primary(gpu)
            account_key = _stable_id(
                "scheduler-primary-range",
                gpu.key,
                parsed["request_backup_key"],
                parsed["execution_id"],
                token_start,
                token_end,
            )
            existing = self.des.extents.get(extent_id)
            if existing is None:
                self.des.register_extent(
                    extent_id=extent_id,
                    request_id=parsed["request_backup_key"],
                    size_bytes=size_bytes,
                    token_start=token_start,
                    token_end=token_end,
                    route_version=parsed["route_version"],
                    ordinal=token_start,
                    model_revision=model_revision,
                    kv_abi=kv_abi,
                    parallel_layout=parallel_layout,
                    pp_rank=parsed["pp_rank"],
                    tp_rank=gpu.tp_rank,
                    shard_count=len(source_group),
                    source_failure_domain=f"node:{gpu.node_id}",
                    workflow_id=parsed["execution_id"],
                )
                self.des.alias_external_source_location(
                    extent_id,
                    source,
                    external_account_key=account_key,
                    reason="offline_scheduler_primary_source_range",
                )
                audit_kind = "source_extent_aliased"
            else:
                identity = {
                    "request_id": parsed["request_backup_key"],
                    "size_bytes": size_bytes,
                    "token_start": token_start,
                    "token_end": token_end,
                    "route_version": parsed["route_version"],
                    "ordinal": token_start,
                    "model_revision": model_revision,
                    "kv_abi": kv_abi,
                    "parallel_layout": parallel_layout,
                    "pp_rank": parsed["pp_rank"],
                    "tp_rank": gpu.tp_rank,
                    "shard_count": len(source_group),
                    "source_failure_domain": f"node:{gpu.node_id}",
                    "workflow_id": parsed["execution_id"],
                }
                conflicts = [
                    name
                    for name, value in identity.items()
                    if getattr(existing, name) != value
                ]
                if conflicts:
                    raise ValueError(
                        "source extent identity changed during target rotation: "
                        + ",".join(conflicts)
                    )
                if (
                    existing.locations.get(source) is not KvCopyState.COMMITTED
                    or existing.external_account_keys.get(source) != account_key
                ):
                    raise ValueError(
                        "target rotation lost the source-owned production buffer"
                    )
                audit_kind = "source_extent_reused_for_target_rotation"
            result.append((gpu, extent_id))
            self._record(
                audit_kind,
                extent_id=extent_id,
                source=source.key,
                external_account_key=account_key,
                token_start=token_start,
                token_end=token_end,
                size_bytes=size_bytes,
            )
        return tuple(result)

    def _schedule_tier_group(
        self,
        *,
        parsed: Mapping[str, Any],
        tier: StorageTier,
        token_start: int,
        token_end: int,
        extents: Sequence[tuple[PhysicalGpuId, str]],
        target_group: Sequence[Any],
        host_node: str,
        ready_ns: int,
        model_revision: str,
        kv_abi: str,
        parallel_layout: str,
        on_tier_notification: TierNotificationCallback | None,
    ) -> TierGroupPlan:
        tier_name = "hbm" if tier is StorageTier.GPU_AEGIS else "dram"
        group_id = _stable_id(
            f"aegis-{tier_name}-group",
            parsed["request_backup_key"],
            parsed["version"],
            parsed["route_version"],
            parsed["execution_id"],
            parsed["pp_rank"],
            (
                parsed["target_dp_rank"]
                if tier is StorageTier.GPU_AEGIS
                else "source-owned-dram"
            ),
            token_start,
            token_end,
        )
        owner = _TierGroupOwner(
            group_id=group_id,
            request_backup_key=str(parsed["request_backup_key"]),
            version=int(parsed["version"]),
            route_version=int(parsed["route_version"]),
            source_dp_rank=int(parsed["source_dp_rank"]),
            target_dp_rank=int(parsed["target_dp_rank"]),
            pp_rank=int(parsed["pp_rank"]),
            tier=tier,
            token_start=int(token_start),
            token_end=int(token_end),
        )
        existing_owner = self._group_owners.get(group_id)
        if existing_owner is not None and existing_owner != owner:
            raise ValueError(f"conflicting physical group owner {group_id}")
        self._group_owners[group_id] = owner
        shard_keys = tuple(
            f"pp{parsed['pp_rank']}:tp{source_gpu.tp_rank}"
            for source_gpu, _extent_id in extents
        )
        self.des.register_commit_group(
            group_id=group_id,
            tier=tier,
            route_version=parsed["route_version"],
            model_revision=model_revision,
            kv_abi=kv_abi,
            parallel_layout=parallel_layout,
            expected_shard_keys=shard_keys,
        )

        destinations: list[KvLocation] = []
        target_by_tp = {spec.gpu.tp_rank: spec.gpu for spec in target_group}
        for source_gpu, _extent_id in extents:
            if tier is StorageTier.GPU_AEGIS:
                destinations.append(
                    KvLocation.gpu_aegis(target_by_tp[source_gpu.tp_rank])
                )
            else:
                destinations.append(KvLocation.node_dram(host_node))

        preflight_error = self._tier_preflight_error(extents, destinations)
        if preflight_error:
            self._group_failure_reasons[group_id] = preflight_error
            self.des.abort_commit_group(group_id, reason=preflight_error)
            self._schedule_notification(
                group_id=group_id,
                tier=tier,
                parsed=parsed,
                token_start=token_start,
                token_end=token_end,
                notify_at_ns=self.des.now_ns,
                transfer_ids=(),
                callback=on_tier_notification,
            )
            self._record(
                "tier_admission_rejected",
                group_id=group_id,
                tier=tier.value,
                token_start=token_start,
                token_end=token_end,
                reason=preflight_error,
            )
            return TierGroupPlan(
                group_id=group_id,
                tier=tier,
                token_start=token_start,
                token_end=token_end,
                transfer_ids=(),
                notify_at_ns=self.des.now_ns,
                admitted=False,
                admission_error=preflight_error,
            )

        jobs: list[PhysicalTransferJob] = []
        error = ""
        for (source_gpu, extent_id), destination, shard_key in zip(
            extents, destinations, shard_keys
        ):
            source = KvLocation.gpu_primary(source_gpu)
            extent = self.des.extents[extent_id]
            path = self._copy_path(source_gpu, destination)
            try:
                transfer = self.des.reserve_transfer(
                    extent_id=extent_id,
                    source=source,
                    destination=destination,
                    resources=path.resources,
                    duration_ns=path.duration_ns(extent.size_bytes),
                    ready_ns=ready_ns,
                    group_id=group_id,
                    shard_key=shard_key,
                )
            except Exception as exc:
                error = f"reservation_failed:{type(exc).__name__}:{exc}"
                self._group_failure_reasons[group_id] = error
                self.des.abort_commit_group(group_id, reason=error)
                break
            job = PhysicalTransferJob(
                transfer_id=transfer.transfer_id,
                group_id=group_id,
                extent_id=extent_id,
                tier=tier,
                shard_key=shard_key,
                tp_rank=source_gpu.tp_rank,
                token_start=token_start,
                token_end=token_end,
                size_bytes=extent.size_bytes,
                path_kind=path.kind,
                source=source,
                destination=destination,
                ready_ns=transfer.ready_ns,
                start_ns=transfer.start_ns,
                end_ns=transfer.end_ns,
                setup_ns=path.setup_ns,
                bottleneck_bytes_per_s=path.bottleneck_bytes_per_s,
                resources=path.resources,
            )
            jobs.append(job)
            self._jobs.append(job)
            self._record("transfer_job_reserved", **job.as_dict())

        transfer_ids = tuple(job.transfer_id for job in jobs)
        notify_at = max((job.end_ns for job in jobs), default=self.des.now_ns)
        self._schedule_notification(
            group_id=group_id,
            tier=tier,
            parsed=parsed,
            token_start=token_start,
            token_end=token_end,
            notify_at_ns=notify_at,
            transfer_ids=transfer_ids,
            callback=on_tier_notification,
        )
        return TierGroupPlan(
            group_id=group_id,
            tier=tier,
            token_start=token_start,
            token_end=token_end,
            transfer_ids=transfer_ids,
            notify_at_ns=notify_at,
            admitted=not error,
            admission_error=error,
        )

    def _tier_preflight_error(
        self,
        extents: Sequence[tuple[PhysicalGpuId, str]],
        destinations: Sequence[KvLocation],
    ) -> str:
        needed_by_gpu: dict[PhysicalGpuId, int] = {}
        needed_by_node: dict[str, int] = {}
        for (source_gpu, extent_id), destination in zip(extents, destinations):
            source_ledger = self.des.gpu_hbm[source_gpu]
            if not source_ledger.healthy:
                return f"source_gpu_unhealthy:{source_gpu.key}"
            extent = self.des.extents[extent_id]
            if (
                extent.locations.get(KvLocation.gpu_primary(source_gpu))
                is not KvCopyState.COMMITTED
            ):
                return f"source_not_committed:{extent_id}"
            if destination.gpu is not None:
                ledger = self.des.gpu_hbm[destination.gpu]
                if not ledger.healthy:
                    return f"destination_gpu_unhealthy:{destination.gpu.key}"
                needed_by_gpu[destination.gpu] = (
                    needed_by_gpu.get(destination.gpu, 0) + extent.size_bytes
                )
            else:
                assert destination.node_id is not None
                ledger = self.des.node_dram[destination.node_id]
                if not ledger.healthy:
                    return f"destination_node_unhealthy:{destination.node_id}"
                needed_by_node[destination.node_id] = (
                    needed_by_node.get(destination.node_id, 0) + extent.size_bytes
                )
        for gpu, needed in needed_by_gpu.items():
            free = self.des.gpu_hbm[gpu].free_bytes
            if free < needed:
                return f"hbm_capacity:{gpu.key}:free={free}:needed={needed}"
        for node, needed in needed_by_node.items():
            free = self.des.node_dram[node].free_bytes
            if free < needed:
                return f"dram_capacity:{node}:free={free}:needed={needed}"
        return ""

    def _copy_path(
        self, source_gpu: PhysicalGpuId, destination: KvLocation
    ) -> _PathReservation:
        bw = self.fabric.bandwidths
        setup = self.fabric.setup
        if destination.tier is StorageTier.GPU_AEGIS:
            assert destination.gpu is not None
            target = destination.gpu
            if source_gpu.node_id == target.node_id:
                return _PathReservation(
                    kind=CopyPathKind.SAME_NODE_D2D_NVLINK,
                    resources=(
                        TransferResourceId("gpu-d2d-read", source_gpu.key),
                        TransferResourceId("nvlink", source_gpu.node_id),
                        TransferResourceId("gpu-d2d-write", target.key),
                    ),
                    setup_ns=setup.same_node_d2d_ns,
                    bottleneck_bytes_per_s=min(
                        bw.gpu_copy_bytes_per_s, bw.nvlink_bytes_per_s
                    ),
                )
            return _PathReservation(
                kind=CopyPathKind.CROSS_NODE_D2D_FABRIC,
                resources=(
                    TransferResourceId("gpu-d2d-read", source_gpu.key),
                    TransferResourceId("pcie-egress", source_gpu.node_id),
                    TransferResourceId("nic-egress", source_gpu.node_id),
                    TransferResourceId("fabric", self.fabric.fabric_id),
                    TransferResourceId("nic-ingress", target.node_id),
                    TransferResourceId("pcie-ingress", target.node_id),
                    TransferResourceId("gpu-d2d-write", target.key),
                ),
                setup_ns=setup.cross_node_d2d_ns,
                bottleneck_bytes_per_s=min(
                    bw.gpu_copy_bytes_per_s,
                    bw.pcie_bytes_per_s,
                    bw.nic_bytes_per_s,
                    bw.fabric_bytes_per_s,
                ),
            )
        if destination.tier is not StorageTier.NODE_DRAM:
            raise ValueError("offline Aegis copies only target GPU_AEGIS or NODE_DRAM")
        assert destination.node_id is not None
        if source_gpu.node_id == destination.node_id:
            return _PathReservation(
                kind=CopyPathKind.SAME_NODE_SOURCE_D2H,
                resources=(
                    TransferResourceId("gpu-d2h-read", source_gpu.key),
                    TransferResourceId("pcie-d2h", source_gpu.node_id),
                    TransferResourceId("dram-write", destination.node_id),
                ),
                setup_ns=setup.same_node_d2h_ns,
                bottleneck_bytes_per_s=min(
                    bw.gpu_copy_bytes_per_s,
                    bw.pcie_bytes_per_s,
                    bw.dram_write_bytes_per_s,
                ),
            )
        return _PathReservation(
            kind=CopyPathKind.CROSS_NODE_SOURCE_D2H_FABRIC,
            resources=(
                TransferResourceId("gpu-d2h-read", source_gpu.key),
                TransferResourceId("pcie-egress", source_gpu.node_id),
                TransferResourceId("nic-egress", source_gpu.node_id),
                TransferResourceId("fabric", self.fabric.fabric_id),
                TransferResourceId("nic-ingress", destination.node_id),
                TransferResourceId("pcie-ingress", destination.node_id),
                TransferResourceId("dram-write", destination.node_id),
            ),
            setup_ns=setup.cross_node_d2h_ns,
            bottleneck_bytes_per_s=min(
                bw.gpu_copy_bytes_per_s,
                bw.pcie_bytes_per_s,
                bw.nic_bytes_per_s,
                bw.fabric_bytes_per_s,
                bw.dram_write_bytes_per_s,
            ),
        )

    def _recovery_path(
        self,
        source: KvLocation,
        destination: KvLocation,
    ) -> _PathReservation:
        """Return the physical resource path for one failover load.

        A same-GPU peer-HBM recovery is an ownership activation: the already
        resident pages are made visible to the retry Scheduler and no byte
        transfer is fabricated.  Every other path reserves all serial copy,
        PCIe, NIC, fabric, and DRAM-read resources on the global calendar.
        """

        if destination.tier is not StorageTier.GPU_PRIMARY:
            raise ValueError("recovery destination must be GPU_PRIMARY")
        assert destination.gpu is not None
        target = destination.gpu
        bw = self.fabric.bandwidths
        setup = self.fabric.setup

        if source.tier is StorageTier.GPU_AEGIS:
            assert source.gpu is not None
            peer = source.gpu
            if peer == target:
                return _PathReservation(
                    kind=CopyPathKind.PEER_HBM_OWNERSHIP_ACTIVATION,
                    resources=(),
                    setup_ns=0,
                    bottleneck_bytes_per_s=bw.gpu_copy_bytes_per_s,
                )
            if peer.node_id == target.node_id:
                return _PathReservation(
                    kind=CopyPathKind.SAME_NODE_RECOVERY_D2D,
                    resources=(
                        TransferResourceId("gpu-d2d-read", peer.key),
                        TransferResourceId("nvlink", peer.node_id),
                        TransferResourceId("gpu-d2d-write", target.key),
                    ),
                    setup_ns=setup.same_node_d2d_ns,
                    bottleneck_bytes_per_s=min(
                        bw.gpu_copy_bytes_per_s,
                        bw.nvlink_bytes_per_s,
                    ),
                )
            return _PathReservation(
                kind=CopyPathKind.CROSS_NODE_RECOVERY_D2D_FABRIC,
                resources=(
                    TransferResourceId("gpu-d2d-read", peer.key),
                    TransferResourceId("pcie-egress", peer.node_id),
                    TransferResourceId("nic-egress", peer.node_id),
                    TransferResourceId("fabric", self.fabric.fabric_id),
                    TransferResourceId("nic-ingress", target.node_id),
                    TransferResourceId("pcie-ingress", target.node_id),
                    TransferResourceId("gpu-d2d-write", target.key),
                ),
                setup_ns=setup.cross_node_d2d_ns,
                bottleneck_bytes_per_s=min(
                    bw.gpu_copy_bytes_per_s,
                    bw.pcie_bytes_per_s,
                    bw.nic_bytes_per_s,
                    bw.fabric_bytes_per_s,
                ),
            )

        if source.tier is not StorageTier.NODE_DRAM:
            raise ValueError("recovery source must be GPU_AEGIS or NODE_DRAM")
        assert source.node_id is not None
        host_node = source.node_id
        if host_node == target.node_id:
            return _PathReservation(
                kind=CopyPathKind.SAME_NODE_DRAM_H2D,
                resources=(
                    TransferResourceId("dram-read", host_node),
                    TransferResourceId("pcie-h2d", host_node),
                    TransferResourceId("gpu-h2d-write", target.key),
                ),
                setup_ns=setup.same_node_h2d_ns,
                bottleneck_bytes_per_s=min(
                    bw.dram_read_bytes_per_s,
                    bw.pcie_bytes_per_s,
                    bw.gpu_copy_bytes_per_s,
                ),
            )
        return _PathReservation(
            kind=CopyPathKind.CROSS_NODE_DRAM_H2D_FABRIC,
            resources=(
                TransferResourceId("dram-read", host_node),
                TransferResourceId("pcie-egress", host_node),
                TransferResourceId("nic-egress", host_node),
                TransferResourceId("fabric", self.fabric.fabric_id),
                TransferResourceId("nic-ingress", target.node_id),
                TransferResourceId("pcie-ingress", target.node_id),
                TransferResourceId("gpu-h2d-write", target.key),
            ),
            setup_ns=setup.cross_node_h2d_ns,
            bottleneck_bytes_per_s=min(
                bw.dram_read_bytes_per_s,
                bw.pcie_bytes_per_s,
                bw.nic_bytes_per_s,
                bw.fabric_bytes_per_s,
                bw.gpu_copy_bytes_per_s,
            ),
        )

    @staticmethod
    def _normalize_recovery_tier_tokens(
        raw: Mapping[StorageTier | str, int],
    ) -> dict[StorageTier, int]:
        if not isinstance(raw, Mapping):
            raise TypeError("tier_tokens must be a mapping")
        result = {
            StorageTier.GPU_AEGIS: 0,
            StorageTier.NODE_DRAM: 0,
        }
        aliases = {
            "hbm": StorageTier.GPU_AEGIS,
            "l1": StorageTier.GPU_AEGIS,
            "gpu_aegis": StorageTier.GPU_AEGIS,
            "dram": StorageTier.NODE_DRAM,
            "l2": StorageTier.NODE_DRAM,
            "node_dram": StorageTier.NODE_DRAM,
        }
        for key, value in raw.items():
            if isinstance(key, StorageTier):
                tier = key
            else:
                try:
                    tier = aliases[str(key).strip().lower()]
                except KeyError as exc:
                    raise ValueError(f"unsupported recovery tier {key!r}") from exc
            if tier not in result:
                raise ValueError(f"unsupported recovery tier {tier.value!r}")
            if result[tier]:
                raise ValueError(f"duplicate recovery token entry for {tier.value}")
            result[tier] = _int_value(
                f"tier_tokens[{tier.value}]", value, minimum=0
            )
        selected = [tier for tier, tokens in result.items() if tokens > 0]
        if len(selected) > 1:
            raise ValueError(
                "current independent-tier semantics select HBM or DRAM, not both"
            )
        return result

    def plan_failover_recovery(
        self,
        *,
        retry_request_id: str,
        request_backup_key: str,
        retry_dp_rank: int,
        tier_tokens: Mapping[StorageTier | str, int],
        ready_ns: int | None = None,
        route_version: int | None = None,
    ) -> dict[str, Any]:
        """Reserve replay-time recovery paths on the authoritative DES heap.

        Protection extents remain source-owned.  Every non-zero-copy receive
        reserves RECOVERY_PENDING bytes on its physical target GPU, but does
        not fabricate a destination extent owned by the Scheduler.  A
        committed hold remains pending until the later Scheduler-adoption
        integration atomically hands those bytes to primary/prefix accounting.
        """

        retry_id = _required_text("retry_request_id", retry_request_id)
        backup_key = _required_text("request_backup_key", request_backup_key)
        retry_rank = _int_value("retry_dp_rank", retry_dp_rank)
        if any(job.retry_request_id == retry_id for job in self._recovery_jobs):
            raise ValueError(f"retry request {retry_id!r} already has a recovery plan")
        requested_by_tier = self._normalize_recovery_tier_tokens(tier_tokens)
        selected = [
            (tier, tokens)
            for tier, tokens in requested_by_tier.items()
            if tokens > 0
        ]
        ready = self.des.now_ns if ready_ns is None else _int_value(
            "ready_ns", ready_ns
        )
        if ready < self.des.now_ns:
            raise ValueError("recovery ready_ns precedes the global DES cursor")
        retry_groups = tuple(
            (pp_rank, self.topology.group(retry_rank, pp_rank))
            for dp_rank, pp_rank in self.topology.rank_groups
            if dp_rank == retry_rank
        )
        if not retry_groups:
            raise ValueError(f"unknown retry DP rank {retry_rank}")
        if not selected:
            plan = {
                "schema": OFFLINE_AEGIS_TRANSFER_BRIDGE_SCHEMA,
                "retry_request_id": retry_id,
                "request_backup_key": backup_key,
                "retry_dp_rank": retry_rank,
                "selected_tier": "recompute",
                "requested_tokens": 0,
                "ready_ns": ready,
                "max_end_ns": ready,
                "jobs": [],
            }
            self._record("recovery_plan_recompute_only", **plan)
            return copy.deepcopy(plan)

        tier, requested_tokens = selected[0]
        candidates = []
        for job in self._jobs:
            extent = self.des.extents[job.extent_id]
            group = self.des.commit_groups.get(job.group_id)
            if (
                job.tier is tier
                and extent.request_id == backup_key
                and group is not None
                and group.state is CommitGroupState.COMMITTED
                and extent.locations.get(job.destination)
                is KvCopyState.COMMITTED
                and (route_version is None or extent.route_version == int(route_version))
            ):
                candidates.append(job)
        if not candidates:
            raise ValueError(
                f"no committed {tier.value} protection copy for {backup_key!r}"
            )
        if route_version is None:
            latest_route = max(
                self.des.extents[job.extent_id].route_version for job in candidates
            )
            candidates = [
                job
                for job in candidates
                if self.des.extents[job.extent_id].route_version == latest_route
            ]
        else:
            latest_route = int(route_version)

        # Coverage is validated for every PP/TP shard before reserving a
        # single resource interval.  A malformed/stale recovery hint therefore
        # fails without leaving a partially installed physical plan.
        for pp_rank, retry_group in retry_groups:
            for target_spec in retry_group:
                tp_rank = target_spec.gpu.tp_rank
                spans = sorted(
                    (
                        (job.token_start, job.token_end)
                        for job in candidates
                        if self.des.extents[job.extent_id].pp_rank == pp_rank
                        and job.tp_rank == tp_rank
                    )
                )
                frontier = 0
                for token_start, token_end in spans:
                    if frontier >= requested_tokens:
                        break
                    if token_end <= frontier:
                        continue
                    if token_start > frontier:
                        break
                    frontier = min(token_end, requested_tokens)
                if frontier < requested_tokens:
                    raise ValueError(
                        "committed recovery frontier does not cover retry shard "
                        f"pp={pp_rank}, tp={tp_rank}: {frontier} < {requested_tokens}"
                    )

        expected_shard_keys = tuple(
            f"pp{pp_rank}/tp{target_spec.gpu.tp_rank}"
            for pp_rank, retry_group in retry_groups
            for target_spec in retry_group
        )
        specs: list[_RecoveryTransferSpec] = []
        for pp_rank, retry_group in retry_groups:
            for target_spec in retry_group:
                tp_rank = target_spec.gpu.tp_rank
                shard_key = f"pp{pp_rank}/tp{tp_rank}"
                shard_candidates = sorted(
                    (
                        job
                        for job in candidates
                        if self.des.extents[job.extent_id].pp_rank == pp_rank
                        and job.tp_rank == tp_rank
                    ),
                    key=lambda job: (job.token_start, job.token_end, job.extent_id),
                )
                frontier = 0
                for source_job in shard_candidates:
                    if frontier >= requested_tokens:
                        break
                    if source_job.token_end <= frontier:
                        continue
                    if source_job.token_start > frontier:
                        break
                    token_start = frontier
                    token_end = min(source_job.token_end, requested_tokens)
                    span_tokens = source_job.token_end - source_job.token_start
                    if span_tokens <= 0 or source_job.size_bytes % span_tokens:
                        raise ValueError(
                            f"extent {source_job.extent_id} has non-integral bytes/token"
                        )
                    size_bytes = (
                        token_end - token_start
                    ) * (source_job.size_bytes // span_tokens)
                    source = source_job.destination
                    destination = KvLocation.gpu_primary(target_spec.gpu)
                    path = self._recovery_path(source, destination)
                    recovery_id = _stable_id(
                        "aegis-recovery",
                        retry_id,
                        backup_key,
                        tier.value,
                        source_job.extent_id,
                        destination.key,
                        token_start,
                        token_end,
                    )
                    specs.append(
                        _RecoveryTransferSpec(
                            recovery_id=recovery_id,
                            shard_key=shard_key,
                            source_job=source_job,
                            pp_rank=pp_rank,
                            tp_rank=tp_rank,
                            token_start=token_start,
                            token_end=token_end,
                            size_bytes=size_bytes,
                            source=source,
                            destination=destination,
                            path=path,
                            destination_mode=(
                                TransferDestinationMode.OWNERSHIP_ACTIVATION
                                if path.kind
                                is CopyPathKind.PEER_HBM_OWNERSHIP_ACTIVATION
                                else TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
                            ),
                        )
                    )
                    frontier = token_end
                assert frontier >= requested_tokens

        # External scheduler-owned receive pages still need byte-exact HBM
        # admission.  Preflight the aggregate per target GPU so a capacity
        # error cannot leave a partially installed recovery group.
        required_holds: dict[PhysicalGpuId, int] = {}
        for spec in specs:
            if (
                spec.destination_mode
                is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
            ):
                assert spec.destination.gpu is not None
                required_holds[spec.destination.gpu] = (
                    required_holds.get(spec.destination.gpu, 0)
                    + spec.size_bytes
                )
        for gpu, required_bytes in required_holds.items():
            ledger = self.des.gpu_hbm[gpu]
            if not ledger.healthy:
                raise ValueError(f"recovery destination GPU {gpu.key} is unhealthy")
            if ledger.free_bytes < required_bytes:
                raise ValueError(
                    f"recovery destination {gpu.key} has {ledger.free_bytes} "
                    f"free bytes, needs {required_bytes} RECOVERY_PENDING bytes"
                )

        recovery_group_id = _stable_id(
            "aegis-recovery-group",
            retry_id,
            backup_key,
            tier.value,
            latest_route,
            requested_tokens,
        )
        recovery_group = self.des.register_recovery_load_group(
            group_id=recovery_group_id,
            retry_request_id=retry_id,
            request_backup_key=backup_key,
            route_version=latest_route,
            expected_shard_keys=expected_shard_keys,
        )
        planned: list[PhysicalRecoveryJob] = []
        try:
            for spec in specs:
                duration_ns = (
                    0
                    if spec.destination_mode
                    is TransferDestinationMode.OWNERSHIP_ACTIVATION
                    else spec.path.duration_ns(spec.size_bytes)
                )
                transfer = self.des.reserve_transfer(
                    extent_id=spec.source_job.extent_id,
                    source=spec.source,
                    destination=spec.destination,
                    resources=spec.path.resources,
                    duration_ns=duration_ns,
                    ready_ns=ready,
                    destination_mode=spec.destination_mode,
                    transfer_size_bytes=spec.size_bytes,
                    token_start=spec.token_start,
                    token_end=spec.token_end,
                    recovery_group_id=recovery_group.group_id,
                    recovery_shard_key=spec.shard_key,
                )
                job = PhysicalRecoveryJob(
                    recovery_id=spec.recovery_id,
                    transfer_id=transfer.transfer_id,
                    group_id=recovery_group.group_id,
                    retry_request_id=retry_id,
                    request_backup_key=backup_key,
                    extent_id=spec.source_job.extent_id,
                    source_tier=tier,
                    path_kind=spec.path.kind,
                    pp_rank=spec.pp_rank,
                    tp_rank=spec.tp_rank,
                    token_start=spec.token_start,
                    token_end=spec.token_end,
                    size_bytes=spec.size_bytes,
                    source=spec.source,
                    destination=spec.destination,
                    ready_ns=transfer.ready_ns,
                    start_ns=transfer.start_ns,
                    end_ns=transfer.end_ns,
                    setup_ns=spec.path.setup_ns,
                    bottleneck_bytes_per_s=(
                        spec.path.bottleneck_bytes_per_s
                    ),
                    resources=spec.path.resources,
                )
                planned.append(job)
            self.des.seal_recovery_load_group(recovery_group.group_id)
        except Exception:
            if recovery_group.state is CommitGroupState.PENDING:
                self.des.abort_recovery_load_group(
                    recovery_group.group_id,
                    reason="planner_admission_failed",
                )
            raise

        for job in planned:
            self._recovery_jobs.append(job)
            self._recovery_outcomes[job.recovery_id] = "QUEUED"
            transfer = self.des.transfers[job.transfer_id]
            self._recovery_event_ids[job.recovery_id] = (
                transfer.start_event_id,
                transfer.completion_event_id,
            )
            self._record("recovery_job_reserved", **job.as_dict())

        max_end = max((job.end_ns for job in planned), default=ready)
        plan = {
            "schema": OFFLINE_AEGIS_TRANSFER_BRIDGE_SCHEMA,
            "retry_request_id": retry_id,
            "request_backup_key": backup_key,
            "retry_dp_rank": retry_rank,
            "selected_tier": tier.value,
            "requested_tokens": requested_tokens,
            "route_version": latest_route,
            "group_id": recovery_group.group_id,
            "ready_ns": ready,
            "max_end_ns": max_end,
            "jobs": [job.as_dict() for job in planned],
        }
        self._record("recovery_plan_reserved", **plan)
        return copy.deepcopy(plan)

    def _schedule_notification(
        self,
        *,
        group_id: str,
        tier: StorageTier,
        parsed: Mapping[str, Any],
        token_start: int,
        token_end: int,
        notify_at_ns: int,
        transfer_ids: tuple[str, ...],
        callback: TierNotificationCallback | None,
    ) -> None:
        if group_id in self._notification_groups:
            raise RuntimeError(f"duplicate notification schedule for {group_id}")
        self._notification_groups.add(group_id)

        def notify(simulator: GlobalDES, _event: Any) -> None:
            group = simulator.commit_groups[group_id]
            if group.state is CommitGroupState.PENDING:
                reason = "projected_completion_without_atomic_group_commit"
                self._group_failure_reasons[group_id] = reason
                simulator.abort_commit_group(group_id, reason=reason)
            reason = self._group_failure_reasons.get(
                group_id,
                (
                    "all_tp_shards_acked"
                    if group.state is CommitGroupState.COMMITTED
                    else "commit_group_aborted"
                ),
            )
            notification = TierNotification(
                group_id=group_id,
                tier=tier,
                state=group.state,
                success=group.state is CommitGroupState.COMMITTED,
                request_backup_key=parsed["request_backup_key"],
                version=parsed["version"],
                route_version=parsed["route_version"],
                token_start=token_start,
                token_end=token_end,
                expected_shards=len(group.expected_shard_keys),
                acked_shards=len(group.acked_shards),
                timestamp_ns=simulator.now_ns,
                reason=reason,
                transfer_ids=transfer_ids,
            )
            self._notifications.append(notification)
            self._record("tier_notification", **notification.as_dict())
            if callback is not None:
                callback(notification)

        self.des.schedule_at(
            notify_at_ns,
            kind="aegis_tier_commit_notification",
            priority=_NOTIFY_PRIORITY,
            payload={"group_id": group_id, "tier": tier.value},
            callback=notify,
        )

    def audit_invariants(self, *, raise_on_error: bool = False) -> list[str]:
        """Audit bridge-owned edges, job records, and frontier monotonicity."""

        self._refresh_recovery_outcomes()
        errors: list[str] = []
        seen_transfer_ids: set[str] = set()
        for job in self._jobs:
            if job.transfer_id in seen_transfer_ids:
                errors.append(f"duplicate bridge transfer {job.transfer_id}")
            seen_transfer_ids.add(job.transfer_id)
            transfer = self.des.transfers.get(job.transfer_id)
            if transfer is None:
                errors.append(f"missing GlobalDES transfer {job.transfer_id}")
                continue
            if transfer.source.tier is not StorageTier.GPU_PRIMARY:
                errors.append(f"{job.transfer_id} is not source-owned GPU primary")
            if job.destination.tier is not job.tier:
                errors.append(
                    f"{job.transfer_id} destination tier differs from its "
                    "independent owner tier"
                )
            if (
                transfer.source.tier is StorageTier.GPU_AEGIS
                and transfer.destination.tier is StorageTier.NODE_DRAM
            ):
                errors.append(f"{job.transfer_id} contains forbidden HBM-to-DRAM edge")
            if job.end_ns < job.start_ns or job.start_ns < job.ready_ns:
                errors.append(f"{job.transfer_id} has an invalid time interval")
            if transfer.group_id != job.group_id:
                errors.append(f"{job.transfer_id} group identity differs")
            owner = self._group_owners.get(job.group_id)
            if owner is None:
                errors.append(f"{job.transfer_id} lacks a physical group owner")
            elif (
                owner.tier is not job.tier
                or owner.token_start != job.token_start
                or owner.token_end != job.token_end
            ):
                errors.append(f"{job.transfer_id} differs from its physical owner")
        for group_id, owner in self._group_owners.items():
            group = self.des.commit_groups.get(group_id)
            if group is None:
                errors.append(f"physical owner {group_id} lacks a commit group")
            elif group.tier is not owner.tier:
                errors.append(f"physical owner {group_id} crosses protection tiers")
        for group_id in self._released_group_destinations:
            if group_id not in self._group_owners:
                errors.append(f"released physical group {group_id} has no owner")
                continue
            for job in self._jobs:
                if job.group_id != group_id:
                    continue
                extent = self.des.extents.get(job.extent_id)
                if extent is not None and job.destination in extent.locations:
                    errors.append(
                        f"released physical group {group_id} retained destination "
                        f"{job.destination.key}"
                    )
        seen_recovery_ids: set[str] = set()
        for job in self._recovery_jobs:
            if job.recovery_id in seen_recovery_ids:
                errors.append(f"duplicate recovery job {job.recovery_id}")
            seen_recovery_ids.add(job.recovery_id)
            if job.source_tier not in {
                StorageTier.GPU_AEGIS,
                StorageTier.NODE_DRAM,
            }:
                errors.append(
                    f"{job.recovery_id} has illegal source tier "
                    f"{job.source_tier.value}"
                )
            if job.source.tier is not job.source_tier:
                errors.append(f"{job.recovery_id} source tier differs from record")
            if job.destination.tier is not StorageTier.GPU_PRIMARY:
                errors.append(f"{job.recovery_id} does not target primary GPU KV")
            transfer = self.des.transfers.get(job.transfer_id)
            if transfer is None:
                errors.append(
                    f"{job.recovery_id} lacks GlobalDES transfer {job.transfer_id}"
                )
                continue
            if (
                transfer.recovery_group_id != job.group_id
                or transfer.extent_id != job.extent_id
                or transfer.source != job.source
                or transfer.destination != job.destination
                or transfer.size_bytes != job.size_bytes
                or transfer.token_start != job.token_start
                or transfer.token_end != job.token_end
            ):
                errors.append(f"{job.recovery_id} differs from GlobalDES transfer")
            if job.start_ns < job.ready_ns or job.end_ns < job.start_ns:
                errors.append(f"{job.recovery_id} has an invalid time interval")
            activation = (
                job.path_kind
                is CopyPathKind.PEER_HBM_OWNERSHIP_ACTIVATION
            )
            if activation and (job.resources or job.end_ns != job.start_ns):
                errors.append(
                    f"{job.recovery_id} fabricates a copy for ownership activation"
                )
            if not activation and not job.resources:
                errors.append(f"{job.recovery_id} has no contended resources")
            expected_mode = (
                TransferDestinationMode.OWNERSHIP_ACTIVATION
                if activation
                else TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
            )
            if transfer.destination_mode is not expected_mode:
                errors.append(f"{job.recovery_id} has the wrong destination mode")
            extent = self.des.extents[job.extent_id]
            if extent.locations.get(job.destination) in {
                KvCopyState.PENDING,
                KvCopyState.COMMITTED,
            }:
                errors.append(f"{job.recovery_id} fabricated a destination extent")
            for resource in job.resources:
                if self.des.resource_busy_until_ns.get(resource, -1) < job.end_ns:
                    errors.append(
                        f"resource {resource.key} ends before {job.recovery_id}"
                    )
            event_ids = self._recovery_event_ids.get(job.recovery_id)
            if event_ids is None or len(set(event_ids)) != 2:
                errors.append(f"{job.recovery_id} lacks distinct start/end events")
            elif event_ids != (
                transfer.start_event_id,
                transfer.completion_event_id,
            ):
                errors.append(
                    f"{job.recovery_id} compatibility events differ from GlobalDES"
                )
            if self._recovery_outcomes.get(job.recovery_id) not in {
                "QUEUED",
                "IN_FLIGHT",
                "COMPLETED",
                "ABORTED",
            }:
                errors.append(f"{job.recovery_id} has invalid lifecycle state")
        notified_group_ids = [item.group_id for item in self._notifications]
        if len(notified_group_ids) != len(set(notified_group_ids)):
            errors.append("duplicate tier notification group")
        errors.extend(
            self.des.audit_invariants(extent_accounting_mode="external_aggregate")
        )
        if raise_on_error and errors:
            raise RuntimeError("; ".join(errors))
        return errors


__all__ = [
    "AegisReservationHint",
    "ChunkTransferPlan",
    "CopyFabricBandwidths",
    "CopyFabricConfig",
    "CopyFabricSetupLatencies",
    "CopyPathKind",
    "OFFLINE_AEGIS_TRANSFER_BRIDGE_SCHEMA",
    "OfflineAegisTransferBridge",
    "PhysicalRecoveryJob",
    "PhysicalTransferJob",
    "TierGroupPlan",
    "TierNotification",
    "TierNotificationCallback",
]
