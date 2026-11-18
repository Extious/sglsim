"""Project offline Scheduler KV snapshots onto physical DES resources.

The offline Scheduler deliberately reports one symmetric TP-rank view.  This
module is the boundary that turns that view into explicit ``(dp, pp, tp)``
GPU ledgers and node-local host-DRAM ledgers.  It owns only the primary/prefix
HiCache accounts; Aegis pending/committed accounts remain owned by
``GlobalDES`` transfer operations.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from sglsim.srt.simulation.global_des import (
    DramAccount,
    GlobalDES,
    HbmAccount,
    KvCopyState,
    PhysicalGpuId,
    StorageTier,
)


OFFLINE_KV_RESOURCE_SNAPSHOT_SCHEMA = "sglsim.offline.rank-kv-resource.v1"
OFFLINE_RESOURCE_BRIDGE_SCHEMA = "sglsim.offline.physical-resource-bridge.v1"
_SINGLE_TP_VIEW_SCOPE = "one-symmetric-tp-rank-view"
_AUXILIARY_REQUEST_OWNER_PREFIXES = (
    "prefill-floor-",
    "prefill-retry-",
    "retract-",
    "retract-last-",
)


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if parsed < 0 or parsed != value:
        raise ValueError(f"{name} must be a nonnegative integer")
    return parsed


@dataclass(frozen=True)
class PhysicalGpuSpec:
    """Static HBM baseline for one physical GPU."""

    gpu: PhysicalGpuId
    hbm_capacity_bytes: int
    weights_bytes: int = 0
    runtime_bytes: int = 0

    def __post_init__(self) -> None:
        capacity = _nonnegative_int(
            "hbm_capacity_bytes", self.hbm_capacity_bytes
        )
        weights = _nonnegative_int("weights_bytes", self.weights_bytes)
        runtime = _nonnegative_int("runtime_bytes", self.runtime_bytes)
        if weights + runtime > capacity:
            raise ValueError(
                f"GPU {self.gpu.key} weights/runtime baseline exceeds HBM capacity"
            )
        object.__setattr__(self, "hbm_capacity_bytes", capacity)
        object.__setattr__(self, "weights_bytes", weights)
        object.__setattr__(self, "runtime_bytes", runtime)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu.as_dict(),
            "hbm_capacity_bytes": self.hbm_capacity_bytes,
            "weights_bytes": self.weights_bytes,
            "runtime_bytes": self.runtime_bytes,
        }


@dataclass(frozen=True)
class NodeDramSpec:
    """Static host-DRAM baseline for one physical node."""

    node_id: str
    capacity_bytes: int
    runtime_bytes: int = 0

    def __post_init__(self) -> None:
        node_id = str(self.node_id)
        if not node_id:
            raise ValueError("node_id must be nonempty")
        capacity = _nonnegative_int("capacity_bytes", self.capacity_bytes)
        runtime = _nonnegative_int("runtime_bytes", self.runtime_bytes)
        if runtime > capacity:
            raise ValueError(
                f"node {node_id} runtime baseline exceeds DRAM capacity"
            )
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "capacity_bytes", capacity)
        object.__setattr__(self, "runtime_bytes", runtime)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "capacity_bytes": self.capacity_bytes,
            "runtime_bytes": self.runtime_bytes,
        }


@dataclass(frozen=True)
class OfflinePhysicalTopology:
    """Complete physical placement for all Scheduler TP groups."""

    gpus: tuple[PhysicalGpuSpec, ...]
    nodes: tuple[NodeDramSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpus", tuple(self.gpus))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        self._validate()

    def _validate(self) -> None:
        if not self.gpus:
            raise ValueError("physical topology must contain at least one GPU")
        if not self.nodes:
            raise ValueError("physical topology must contain at least one DRAM node")

        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("physical topology has duplicate node_id entries")
        known_nodes = set(node_ids)

        identities = [spec.gpu for spec in self.gpus]
        if len(set(identities)) != len(identities):
            raise ValueError("physical topology has duplicate PhysicalGpuId entries")
        slots = [(spec.gpu.node_id, spec.gpu.local_gpu_id) for spec in self.gpus]
        if len(set(slots)) != len(slots):
            raise ValueError("two logical ranks occupy the same physical GPU slot")
        rank_coordinates = [
            (spec.gpu.dp_rank, spec.gpu.pp_rank, spec.gpu.tp_rank)
            for spec in self.gpus
        ]
        if len(set(rank_coordinates)) != len(rank_coordinates):
            raise ValueError("physical topology has duplicate (dp, pp, tp) ranks")
        for spec in self.gpus:
            if spec.gpu.node_id not in known_nodes:
                raise ValueError(
                    f"GPU {spec.gpu.key} refers to unknown node {spec.gpu.node_id!r}"
                )

        groups: dict[tuple[int, int], list[int]] = {}
        for spec in self.gpus:
            key = (spec.gpu.dp_rank, spec.gpu.pp_rank)
            groups.setdefault(key, []).append(spec.gpu.tp_rank)
        for key, tp_ranks in groups.items():
            expected = list(range(len(tp_ranks)))
            if sorted(tp_ranks) != expected:
                raise ValueError(
                    f"TP ranks for dp={key[0]}, pp={key[1]} must be contiguous "
                    f"from zero; got {sorted(tp_ranks)}"
                )

        dp_ranks = sorted({dp for dp, _pp in groups})
        if dp_ranks != list(range(len(dp_ranks))):
            raise ValueError(f"DP ranks must be contiguous from zero; got {dp_ranks}")
        for dp_rank in dp_ranks:
            pp_ranks = sorted(pp for dp, pp in groups if dp == dp_rank)
            if pp_ranks != list(range(len(pp_ranks))):
                raise ValueError(
                    f"PP ranks for dp={dp_rank} must be contiguous from zero; "
                    f"got {pp_ranks}"
                )

    def group(self, dp_rank: int, pp_rank: int) -> tuple[PhysicalGpuSpec, ...]:
        group = tuple(
            sorted(
                (
                    spec
                    for spec in self.gpus
                    if spec.gpu.dp_rank == int(dp_rank)
                    and spec.gpu.pp_rank == int(pp_rank)
                ),
                key=lambda spec: spec.gpu.tp_rank,
            )
        )
        if not group:
            raise ValueError(
                f"no physical TP group for dp={int(dp_rank)}, pp={int(pp_rank)}"
            )
        return group

    @property
    def rank_groups(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted({(spec.gpu.dp_rank, spec.gpu.pp_rank) for spec in self.gpus})
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gpus": [spec.as_dict() for spec in sorted(self.gpus, key=lambda x: x.gpu)],
            "nodes": [
                spec.as_dict()
                for spec in sorted(self.nodes, key=lambda x: x.node_id)
            ],
        }


@dataclass(frozen=True)
class _RankProjection:
    timestamp_ns: int
    scheduler_tp_rank: int
    tp_group_size: int
    primary_bytes_per_tp_rank: int
    prefix_bytes_per_tp_rank: int
    l1_capacity_bytes_per_tp_rank: int
    l2_used_bytes_per_tp_rank: int
    l2_capacity_bytes_per_tp_rank: int
    l2_used_by_node: tuple[tuple[str, int], ...]
    l2_capacity_by_node: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "scheduler_tp_rank": self.scheduler_tp_rank,
            "tp_group_size": self.tp_group_size,
            "primary_bytes_per_tp_rank": self.primary_bytes_per_tp_rank,
            "prefix_bytes_per_tp_rank": self.prefix_bytes_per_tp_rank,
            "l1_capacity_bytes_per_tp_rank": self.l1_capacity_bytes_per_tp_rank,
            "l2_used_bytes_per_tp_rank": self.l2_used_bytes_per_tp_rank,
            "l2_capacity_bytes_per_tp_rank": self.l2_capacity_bytes_per_tp_rank,
            "l2_used_by_node": dict(self.l2_used_by_node),
            "l2_capacity_by_node": dict(self.l2_capacity_by_node),
        }


class OfflineResourceBridge:
    """Replace rank-local Scheduler views in physical HBM/DRAM ledgers."""

    def __init__(
        self,
        des: GlobalDES,
        topology: OfflinePhysicalTopology,
        *,
        clock_tolerance_ns: int = 1,
    ) -> None:
        self.des = des
        self.topology = topology
        self.clock_tolerance_ns = _nonnegative_int(
            "clock_tolerance_ns", clock_tolerance_ns
        )
        self._latest: dict[tuple[int, int], _RankProjection] = {}
        self._rank_steps: list[dict[str, Any]] = []
        self._failed_dp_ranks: set[int] = set()
        self._ever_failed_dp_ranks: set[int] = set()
        self._failed_process_groups: set[tuple[int, int]] = set()
        self._ever_failed_process_groups: set[tuple[int, int]] = set()
        self._failed_gpus: set[PhysicalGpuId] = set()
        self._ever_failed_gpus: set[PhysicalGpuId] = set()
        self._failed_nodes: set[str] = set()
        self._ever_failed_nodes: set[str] = set()
        self._node_affected_dp_ranks: dict[str, tuple[int, ...]] = {}
        self._lifecycle_events: list[dict[str, Any]] = []

        if self.des.gpu_hbm or self.des.node_dram:
            raise ValueError(
                "OfflineResourceBridge requires an unregistered physical ledger"
            )

        for node in sorted(topology.nodes, key=lambda item: item.node_id):
            self.des.register_node_dram(
                node.node_id,
                capacity_bytes=node.capacity_bytes,
                runtime_bytes=node.runtime_bytes,
            )
        for spec in sorted(topology.gpus, key=lambda item: item.gpu):
            self.des.register_gpu(
                spec.gpu,
                capacity_bytes=spec.hbm_capacity_bytes,
                weights_bytes=spec.weights_bytes,
                runtime_bytes=spec.runtime_bytes,
            )

    @property
    def rank_steps(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._rank_steps))

    @property
    def failed_dp_ranks(self) -> tuple[int, ...]:
        return tuple(sorted(self._failed_dp_ranks))

    @property
    def failed_process_groups(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._failed_process_groups))

    @property
    def failed_gpus(self) -> tuple[PhysicalGpuId, ...]:
        return tuple(sorted(self._failed_gpus))

    @property
    def failed_nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self._failed_nodes))

    @property
    def node_affected_dp_ranks(self) -> dict[str, tuple[int, ...]]:
        return copy.deepcopy(
            {
                node_id: self._node_affected_dp_ranks[node_id]
                for node_id in sorted(self._failed_nodes)
            }
        )

    def _dp_groups(
        self, dp_rank: int
    ) -> tuple[tuple[tuple[int, int], tuple[PhysicalGpuSpec, ...]], ...]:
        parsed_dp = _nonnegative_int("dp_rank", dp_rank)
        groups = tuple(
            (key, self.topology.group(*key))
            for key in self.topology.rank_groups
            if key[0] == parsed_dp
        )
        if not groups:
            raise ValueError(f"unknown physical DP rank {parsed_dp}")
        return groups

    def _node_specs(self, node_id: str) -> tuple[PhysicalGpuSpec, ...]:
        node = str(node_id)
        if not node or node not in {spec.node_id for spec in self.topology.nodes}:
            raise ValueError(f"unknown physical node {node!r}")
        return tuple(
            sorted(
                (spec for spec in self.topology.gpus if spec.gpu.node_id == node),
                key=lambda spec: spec.gpu,
            )
        )

    def _validate_node_failure_placement(self, node_id: str) -> None:
        """Fail closed when one symmetric Scheduler TP view spans nodes.

        The bridge deliberately consumes one symmetric TP-rank snapshot.  A
        partial TP-group failure would require a per-TP Scheduler snapshot to
        distinguish the surviving shards, which this schema does not provide.
        Dense placements keep every ``(dp, pp)`` TP group on one node; reject a
        non-dense split before mutating any physical ledger.
        """

        for key in self.topology.rank_groups:
            group = self.topology.group(*key)
            group_nodes = {spec.gpu.node_id for spec in group}
            if node_id in group_nodes and len(group_nodes) != 1:
                raise ValueError(
                    f"node fail-stop cannot represent cross-node TP group "
                    f"dp={key[0]}, pp={key[1]} with symmetric rank snapshots"
                )

    @staticmethod
    def _event_metadata(
        event_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if event_metadata is not None and not isinstance(event_metadata, Mapping):
            raise ValueError("event_metadata must be a mapping")
        return {} if event_metadata is None else copy.deepcopy(dict(event_metadata))

    def _empty_projection(
        self,
        group: tuple[PhysicalGpuSpec, ...],
    ) -> _RankProjection:
        return _RankProjection(
            timestamp_ns=self.des.now_ns,
            scheduler_tp_rank=0,
            tp_group_size=len(group),
            primary_bytes_per_tp_rank=0,
            prefix_bytes_per_tp_rank=0,
            l1_capacity_bytes_per_tp_rank=0,
            l2_used_bytes_per_tp_rank=0,
            l2_capacity_bytes_per_tp_rank=0,
            l2_used_by_node=(),
            l2_capacity_by_node=(),
        )

    def _append_lifecycle_rank_step(
        self,
        *,
        key: tuple[int, int],
        group: tuple[PhysicalGpuSpec, ...],
        projection: _RankProjection,
        phase: str,
        event_metadata: Mapping[str, Any],
    ) -> None:
        step = {
            "schema": OFFLINE_RESOURCE_BRIDGE_SCHEMA,
            "sequence": len(self._rank_steps),
            "timestamp_ns": self.des.now_ns,
            "des_now_ns": self.des.now_ns,
            "phase": str(phase),
            "event_metadata": copy.deepcopy(dict(event_metadata)),
            "dp_rank": key[0],
            "pp_rank": key[1],
            "scheduler_tp_rank": projection.scheduler_tp_rank,
            "physical_gpus": [spec.gpu.as_dict() for spec in group],
            "primary_pages_per_tp_rank": 0,
            "prefix_pages_per_tp_rank": 0,
            "primary_bytes_per_tp_rank": projection.primary_bytes_per_tp_rank,
            "prefix_bytes_per_tp_rank": projection.prefix_bytes_per_tp_rank,
            "l1_used_bytes_per_tp_rank": (
                projection.primary_bytes_per_tp_rank
                + projection.prefix_bytes_per_tp_rank
            ),
            "physical_primary_bytes_total": (
                projection.primary_bytes_per_tp_rank * len(group)
            ),
            "physical_prefix_bytes_total": (
                projection.prefix_bytes_per_tp_rank * len(group)
            ),
            "l2_used_bytes_per_tp_rank": projection.l2_used_bytes_per_tp_rank,
            "l2_used_by_node": dict(projection.l2_used_by_node),
            "running_request_ids": [],
            "waiting_request_ids": [],
            "gpu_hbm_after": {
                spec.gpu.key: self.des.gpu_hbm[spec.gpu].snapshot() for spec in group
            },
            "node_dram_after": {
                node.node_id: self.des.node_dram[node.node_id].snapshot()
                for node in self.topology.nodes
            },
        }
        self._rank_steps.append(step)

    def _append_lifecycle_event(
        self,
        *,
        action: str,
        status: str,
        dp_rank: int,
        reason: str,
        metadata: Mapping[str, Any],
        gpus: tuple[PhysicalGpuSpec, ...],
        affected_extents: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        event = {
            "schema": OFFLINE_RESOURCE_BRIDGE_SCHEMA,
            "sequence": len(self._lifecycle_events),
            "timestamp_ns": self.des.now_ns,
            "action": str(action),
            "status": str(status),
            "dp_rank": int(dp_rank),
            "reason": str(reason),
            "event_metadata": copy.deepcopy(dict(metadata)),
            "physical_gpus": [spec.gpu.as_dict() for spec in gpus],
            "affected_extents": list(affected_extents),
        }
        self._lifecycle_events.append(copy.deepcopy(event))
        return copy.deepcopy(event)

    def _append_node_lifecycle_event(
        self,
        *,
        action: str,
        status: str,
        node_id: str,
        reason: str,
        metadata: Mapping[str, Any],
        gpus: tuple[PhysicalGpuSpec, ...],
        affected_dp_ranks: tuple[int, ...],
        affected_extents: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        event = {
            "schema": OFFLINE_RESOURCE_BRIDGE_SCHEMA,
            "sequence": len(self._lifecycle_events),
            "timestamp_ns": self.des.now_ns,
            "action": str(action),
            "status": str(status),
            "failure_domain": "node",
            "node_id": str(node_id),
            "affected_dp_ranks": list(affected_dp_ranks),
            "reason": str(reason),
            "event_metadata": copy.deepcopy(dict(metadata)),
            "physical_gpus": [spec.gpu.as_dict() for spec in gpus],
            "affected_extents": list(affected_extents),
        }
        self._lifecycle_events.append(copy.deepcopy(event))
        return copy.deepcopy(event)

    def _physical_gpu_spec(self, gpu: PhysicalGpuId) -> PhysicalGpuSpec:
        matches = tuple(spec for spec in self.topology.gpus if spec.gpu == gpu)
        if len(matches) != 1:
            raise ValueError(f"unknown physical GPU {gpu.key}")
        return matches[0]

    def _invalidate_group_ordinary_l1(
        self,
        key: tuple[int, int],
        group: tuple[PhysicalGpuSpec, ...],
        *,
        reason: str,
    ) -> tuple[str, ...]:
        """Invalidate a TP group's ordinary KV without erasing survivor Aegis.

        A single missing TP shard makes the complete model-parallel KV group
        unusable.  Surviving GPUs remain healthy and retain independent Aegis
        copies, but ordinary primary/prefix bytes and their external aliases
        must be removed from the whole group before the Scheduler process can
        later restart with empty volatile state.
        """

        group_gpus = {spec.gpu for spec in group}
        released: set[str] = set()
        for extent in tuple(self.des.extents.values()):
            for location, state in tuple(extent.locations.items()):
                if (
                    location.gpu not in group_gpus
                    or location.tier
                    not in {StorageTier.GPU_PRIMARY, StorageTier.GPU_PREFIX}
                    or state not in {KvCopyState.PENDING, KvCopyState.COMMITTED}
                ):
                    continue
                self.des.release_extent_location(
                    extent.extent_id,
                    location,
                    reason=f"{reason}:tp_group_incomplete",
                )
                released.add(extent.extent_id)

        for spec in group:
            ledger = self.des.gpu_hbm[spec.gpu]
            if not ledger.healthy:
                continue
            for account in (HbmAccount.PRIMARY_KV, HbmAccount.PREFIX_KV):
                current = int(getattr(ledger, account.value))
                if current:
                    self.des.adjust_gpu_hbm(
                        spec.gpu,
                        account,
                        -current,
                        reason=f"{reason}:tp_group_incomplete",
                    )

        previous = self._latest.get(key, self._empty_projection(group))
        failed_projection = replace(
            previous,
            timestamp_ns=self.des.now_ns,
            primary_bytes_per_tp_rank=0,
            prefix_bytes_per_tp_rank=0,
        )
        self._latest[key] = failed_projection
        self._append_lifecycle_rank_step(
            key=key,
            group=group,
            projection=failed_projection,
            phase="tp_group_ordinary_l1_invalidated",
            event_metadata={"reason": str(reason)},
        )
        return tuple(sorted(released))

    def fail_process_group(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        reason: str = "offline_scheduler_process_failure",
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fail one ``(dp, pp)`` process and its complete physical TP group."""

        key = (_nonnegative_int("dp_rank", dp_rank), _nonnegative_int("pp_rank", pp_rank))
        group = self.topology.group(*key)
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("failure reason must be nonempty")
        metadata = self._event_metadata(event_metadata)
        if key[0] in self._failed_dp_ranks:
            raise ValueError(f"cannot fail process {key} during DP-rank failure")
        failed_node_overlap = sorted(
            {spec.gpu.node_id for spec in group} & self._failed_nodes
        )
        if failed_node_overlap:
            raise ValueError(
                f"cannot fail process {key} during node failure on "
                f"{failed_node_overlap}"
            )
        failed_gpu_overlap = sorted(
            spec.gpu.key for spec in group if spec.gpu in self._failed_gpus
        )
        if failed_gpu_overlap:
            raise ValueError(
                f"cannot fail process {key} with failed GPUs {failed_gpu_overlap}"
            )
        if key in self._failed_process_groups:
            return self._append_lifecycle_event(
                action="fail",
                status="already_failed",
                dp_rank=key[0],
                reason=reason_name,
                metadata={**metadata, "failure_domain": "process", "pp_rank": key[1]},
                gpus=group,
            )

        affected: set[str] = set()
        for spec in group:
            affected.update(self.des.fail_gpu(spec.gpu))
        self._failed_process_groups.add(key)
        self._ever_failed_process_groups.add(key)
        previous = self._latest.get(key, self._empty_projection(group))
        failed_projection = replace(
            previous,
            timestamp_ns=self.des.now_ns,
            primary_bytes_per_tp_rank=0,
            prefix_bytes_per_tp_rank=0,
        )
        self._latest[key] = failed_projection
        self._append_lifecycle_rank_step(
            key=key,
            group=group,
            projection=failed_projection,
            phase="scheduler_process_failed",
            event_metadata={**metadata, "reason": reason_name},
        )
        event = self._append_lifecycle_event(
            action="fail",
            status="failed",
            dp_rank=key[0],
            reason=reason_name,
            metadata={**metadata, "failure_domain": "process", "pp_rank": key[1]},
            gpus=group,
            affected_extents=tuple(sorted(affected)),
        )
        event["failure_domain"] = "process"
        event["pp_rank"] = key[1]
        event["scheduler_process_id"] = f"dp{key[0]}/pp{key[1]}"
        self._lifecycle_events[-1] = copy.deepcopy(event)
        return event

    def repair_process_group(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        reason: str = "offline_scheduler_process_repair",
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reload one failed process group with empty ordinary/Aegis KV."""

        key = (_nonnegative_int("dp_rank", dp_rank), _nonnegative_int("pp_rank", pp_rank))
        group = self.topology.group(*key)
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("repair reason must be nonempty")
        metadata = self._event_metadata(event_metadata)
        if key not in self._failed_process_groups:
            if key not in self._ever_failed_process_groups:
                raise ValueError(f"cannot repair never-failed process {key}")
            event = self._append_lifecycle_event(
                action="repair",
                status="already_repaired",
                dp_rank=key[0],
                reason=reason_name,
                metadata={**metadata, "failure_domain": "process", "pp_rank": key[1]},
                gpus=group,
            )
            event["failure_domain"] = "process"
            event["pp_rank"] = key[1]
            self._lifecycle_events[-1] = copy.deepcopy(event)
            return event
        if any(self.des.gpu_hbm[spec.gpu].healthy for spec in group):
            raise ValueError(f"process {key} has a partially repaired TP group")

        for spec in group:
            self.des.repair_gpu(
                spec.gpu,
                weights_bytes=spec.weights_bytes,
                runtime_bytes=spec.runtime_bytes,
            )
        self._failed_process_groups.remove(key)
        projection = self._latest.get(key, self._empty_projection(group))
        self._append_lifecycle_rank_step(
            key=key,
            group=group,
            projection=projection,
            phase="scheduler_process_repaired_empty",
            event_metadata={**metadata, "reason": reason_name},
        )
        event = self._append_lifecycle_event(
            action="repair",
            status="repaired",
            dp_rank=key[0],
            reason=reason_name,
            metadata={**metadata, "failure_domain": "process", "pp_rank": key[1]},
            gpus=group,
        )
        event["failure_domain"] = "process"
        event["pp_rank"] = key[1]
        event["scheduler_process_id"] = f"dp{key[0]}/pp{key[1]}"
        self._lifecycle_events[-1] = copy.deepcopy(event)
        return event

    def fail_gpu(
        self,
        gpu: PhysicalGpuId,
        *,
        reason: str = "offline_physical_gpu_failure",
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fail exactly one GPU and invalidate its now-incomplete TP group."""

        spec = self._physical_gpu_spec(gpu)
        key = (gpu.dp_rank, gpu.pp_rank)
        group = self.topology.group(*key)
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("failure reason must be nonempty")
        metadata = self._event_metadata(event_metadata)
        if key[0] in self._failed_dp_ranks:
            raise ValueError(f"cannot fail GPU {gpu.key} during DP-rank failure")
        if key in self._failed_process_groups:
            raise ValueError(f"cannot fail GPU {gpu.key} during process failure")
        if gpu.node_id in self._failed_nodes:
            raise ValueError(f"cannot fail GPU {gpu.key} during node failure")
        if gpu in self._failed_gpus:
            event = self._append_lifecycle_event(
                action="fail",
                status="already_failed",
                dp_rank=gpu.dp_rank,
                reason=reason_name,
                metadata={**metadata, "failure_domain": "gpu"},
                gpus=(spec,),
            )
            event["failure_domain"] = "gpu"
            event["physical_gpu_id"] = gpu.key
            self._lifecycle_events[-1] = copy.deepcopy(event)
            return event

        affected = set(self.des.fail_gpu(gpu))
        affected.update(
            self._invalidate_group_ordinary_l1(
                key,
                group,
                reason=reason_name,
            )
        )
        self._failed_gpus.add(gpu)
        self._ever_failed_gpus.add(gpu)
        event = self._append_lifecycle_event(
            action="fail",
            status="failed",
            dp_rank=gpu.dp_rank,
            reason=reason_name,
            metadata={**metadata, "failure_domain": "gpu"},
            gpus=(spec,),
            affected_extents=tuple(sorted(affected)),
        )
        event["failure_domain"] = "gpu"
        event["physical_gpu_id"] = gpu.key
        event["incomplete_tp_group"] = {"dp_rank": gpu.dp_rank, "pp_rank": gpu.pp_rank}
        event["recovery_source_forbidden"] = True
        self._lifecycle_events[-1] = copy.deepcopy(event)
        return event

    def repair_gpu(
        self,
        gpu: PhysicalGpuId,
        *,
        reason: str = "offline_physical_gpu_repair",
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reload exactly one GPU; lost KV remains lost and is not resurrected."""

        spec = self._physical_gpu_spec(gpu)
        key = (gpu.dp_rank, gpu.pp_rank)
        group = self.topology.group(*key)
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("repair reason must be nonempty")
        metadata = self._event_metadata(event_metadata)
        if gpu not in self._failed_gpus:
            if gpu not in self._ever_failed_gpus:
                raise ValueError(f"cannot repair never-failed GPU {gpu.key}")
            event = self._append_lifecycle_event(
                action="repair",
                status="already_repaired",
                dp_rank=gpu.dp_rank,
                reason=reason_name,
                metadata={**metadata, "failure_domain": "gpu"},
                gpus=(spec,),
            )
            event["failure_domain"] = "gpu"
            event["physical_gpu_id"] = gpu.key
            self._lifecycle_events[-1] = copy.deepcopy(event)
            return event
        if self.des.gpu_hbm[gpu].healthy:
            raise ValueError(f"failed GPU {gpu.key} is already physically healthy")
        self.des.repair_gpu(
            gpu,
            weights_bytes=spec.weights_bytes,
            runtime_bytes=spec.runtime_bytes,
        )
        self._failed_gpus.remove(gpu)
        projection = self._latest.get(key, self._empty_projection(group))
        self._append_lifecycle_rank_step(
            key=key,
            group=group,
            projection=projection,
            phase="physical_gpu_repaired_empty",
            event_metadata={**metadata, "reason": reason_name},
        )
        event = self._append_lifecycle_event(
            action="repair",
            status="repaired",
            dp_rank=gpu.dp_rank,
            reason=reason_name,
            metadata={**metadata, "failure_domain": "gpu"},
            gpus=(spec,),
        )
        event["failure_domain"] = "gpu"
        event["physical_gpu_id"] = gpu.key
        event["repair_restored_empty_volatile_state"] = True
        self._lifecycle_events[-1] = copy.deepcopy(event)
        return event

    def fail_dp_rank(
        self,
        dp_rank: int,
        *,
        reason: str = "offline_dp_rank_failure",
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fail every PP/TP GPU of one DP rank without touching host DRAM.

        A repeated failure of an already-failed DP rank is an idempotent
        ``already_failed`` event.  The first failure zeroes the bridge's
        expected L1 projection but deliberately retains every L2 projection.
        """

        groups = self._dp_groups(dp_rank)
        parsed_dp = groups[0][0][0]
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("failure reason must be nonempty")
        metadata = self._event_metadata(event_metadata)
        all_gpus = tuple(spec for _key, group in groups for spec in group)
        process_overlap = sorted(
            key for key, _group in groups if key in self._failed_process_groups
        )
        gpu_overlap = sorted(
            spec.gpu.key for spec in all_gpus if spec.gpu in self._failed_gpus
        )
        if process_overlap or gpu_overlap:
            raise ValueError(
                f"cannot fail DP rank {parsed_dp} during narrower failure: "
                f"processes={process_overlap}, gpus={gpu_overlap}"
            )
        failed_node_overlap = sorted(
            {spec.gpu.node_id for spec in all_gpus} & self._failed_nodes
        )
        if failed_node_overlap:
            raise ValueError(
                f"cannot fail DP rank {parsed_dp} while node failure is active on "
                f"{failed_node_overlap}"
            )
        if parsed_dp in self._failed_dp_ranks:
            return self._append_lifecycle_event(
                action="fail",
                status="already_failed",
                dp_rank=parsed_dp,
                reason=reason_name,
                metadata=metadata,
                gpus=all_gpus,
            )

        affected: set[str] = set()
        for spec in all_gpus:
            affected.update(self.des.fail_gpu(spec.gpu))

        self._failed_dp_ranks.add(parsed_dp)
        self._ever_failed_dp_ranks.add(parsed_dp)
        lifecycle_metadata = {
            **metadata,
            "reason": reason_name,
            "affected_extents": sorted(affected),
        }
        for key, group in groups:
            previous = self._latest.get(key, self._empty_projection(group))
            failed_projection = replace(
                previous,
                timestamp_ns=self.des.now_ns,
                primary_bytes_per_tp_rank=0,
                prefix_bytes_per_tp_rank=0,
            )
            self._latest[key] = failed_projection
            self._append_lifecycle_rank_step(
                key=key,
                group=group,
                projection=failed_projection,
                phase="dp_rank_failed",
                event_metadata=lifecycle_metadata,
            )
        return self._append_lifecycle_event(
            action="fail",
            status="failed",
            dp_rank=parsed_dp,
            reason=reason_name,
            metadata=metadata,
            gpus=all_gpus,
            affected_extents=tuple(sorted(affected)),
        )

    def repair_dp_rank(
        self,
        dp_rank: int,
        *,
        reason: str = "offline_dp_rank_repair",
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reload topology baselines without resurrecting lost L1/Aegis KV.

        Repairing a DP rank that has never failed is invalid.  Repeating a
        completed repair is an idempotent ``already_repaired`` event.
        """

        groups = self._dp_groups(dp_rank)
        parsed_dp = groups[0][0][0]
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("repair reason must be nonempty")
        metadata = self._event_metadata(event_metadata)
        all_gpus = tuple(spec for _key, group in groups for spec in group)
        failed_node_overlap = sorted(
            {spec.gpu.node_id for spec in all_gpus} & self._failed_nodes
        )
        if failed_node_overlap:
            raise ValueError(
                f"cannot repair DP rank {parsed_dp} while node failure is active on "
                f"{failed_node_overlap}"
            )
        if parsed_dp not in self._failed_dp_ranks:
            if parsed_dp not in self._ever_failed_dp_ranks:
                raise ValueError(f"cannot repair never-failed DP rank {parsed_dp}")
            return self._append_lifecycle_event(
                action="repair",
                status="already_repaired",
                dp_rank=parsed_dp,
                reason=reason_name,
                metadata=metadata,
                gpus=all_gpus,
            )
        if any(self.des.gpu_hbm[spec.gpu].healthy for spec in all_gpus):
            raise ValueError(
                f"DP rank {parsed_dp} has a partially repaired physical GPU set"
            )

        for spec in all_gpus:
            self.des.repair_gpu(
                spec.gpu,
                weights_bytes=spec.weights_bytes,
                runtime_bytes=spec.runtime_bytes,
            )
        self._failed_dp_ranks.remove(parsed_dp)
        lifecycle_metadata = {**metadata, "reason": reason_name}
        for key, group in groups:
            projection = self._latest.get(key, self._empty_projection(group))
            # Failure already zeroed L1; repair changes only model/runtime
            # baselines and keeps durable L2 bytes in place.
            self._append_lifecycle_rank_step(
                key=key,
                group=group,
                projection=projection,
                phase="dp_rank_repaired",
                event_metadata=lifecycle_metadata,
            )
        return self._append_lifecycle_event(
            action="repair",
            status="repaired",
            dp_rank=parsed_dp,
            reason=reason_name,
            metadata=metadata,
            gpus=all_gpus,
        )

    def fail_node(
        self,
        node_id: str,
        *,
        reason: str = "offline_node_failure",
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically fail one dense-placement node failure domain.

        Every local GPU and the node-local DRAM ledger become unavailable.
        Their primary/HiCache and independent Aegis copies are volatile and
        are therefore lost.  Other nodes' projections and ledgers are left
        untouched.
        """

        node = str(node_id)
        local_gpus = self._node_specs(node)
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("failure reason must be nonempty")
        metadata = self._event_metadata(event_metadata)
        affected_dp_ranks = tuple(
            sorted({spec.gpu.dp_rank for spec in local_gpus})
        )
        if node in self._failed_nodes:
            return self._append_node_lifecycle_event(
                action="fail",
                status="already_failed",
                node_id=node,
                reason=reason_name,
                metadata=metadata,
                gpus=local_gpus,
                affected_dp_ranks=self._node_affected_dp_ranks[node],
            )

        # Validate every representability and overlapping-lifecycle condition
        # before GlobalDES performs the all-local-resource mutation.
        self._validate_node_failure_placement(node)
        overlapping_dp_failures = sorted(
            set(affected_dp_ranks) & self._failed_dp_ranks
        )
        if overlapping_dp_failures:
            raise ValueError(
                "cannot fail node while DP-rank failure is active for "
                f"{overlapping_dp_failures}"
            )
        overlapping_processes = sorted(
            key
            for key in self._failed_process_groups
            if any(
                spec.gpu.node_id == node
                for spec in self.topology.group(*key)
            )
        )
        overlapping_gpus = sorted(
            gpu.key for gpu in self._failed_gpus if gpu.node_id == node
        )
        if overlapping_processes or overlapping_gpus:
            raise ValueError(
                f"cannot fail node during narrower failure: "
                f"processes={overlapping_processes}, gpus={overlapping_gpus}"
            )

        affected_extents = self.des.fail_node(node)
        self._failed_nodes.add(node)
        self._ever_failed_nodes.add(node)
        self._node_affected_dp_ranks[node] = affected_dp_ranks

        lifecycle_metadata = {
            **metadata,
            "reason": reason_name,
            "node_id": node,
            "affected_dp_ranks": list(affected_dp_ranks),
            "affected_extents": list(affected_extents),
        }
        for key in self.topology.rank_groups:
            group = self.topology.group(*key)
            if not group or group[0].gpu.node_id != node:
                continue
            previous = self._latest.get(key, self._empty_projection(group))
            failed_projection = replace(
                previous,
                timestamp_ns=self.des.now_ns,
                primary_bytes_per_tp_rank=0,
                prefix_bytes_per_tp_rank=0,
                l1_capacity_bytes_per_tp_rank=0,
                l2_used_bytes_per_tp_rank=0,
                l2_capacity_bytes_per_tp_rank=0,
                l2_used_by_node=((node, 0),),
                l2_capacity_by_node=((node, 0),),
            )
            self._latest[key] = failed_projection
            self._append_lifecycle_rank_step(
                key=key,
                group=group,
                projection=failed_projection,
                phase="node_failed",
                event_metadata=lifecycle_metadata,
            )
        return self._append_node_lifecycle_event(
            action="fail",
            status="failed",
            node_id=node,
            reason=reason_name,
            metadata=metadata,
            gpus=local_gpus,
            affected_dp_ranks=affected_dp_ranks,
            affected_extents=affected_extents,
        )

    def repair_node(
        self,
        node_id: str,
        *,
        reason: str = "offline_node_repair",
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Restore resource baselines without resurrecting any node-local KV."""

        node = str(node_id)
        local_gpus = self._node_specs(node)
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("repair reason must be nonempty")
        metadata = self._event_metadata(event_metadata)
        topology_dp_ranks = tuple(
            sorted({spec.gpu.dp_rank for spec in local_gpus})
        )
        if node not in self._failed_nodes:
            if node not in self._ever_failed_nodes:
                raise ValueError(f"cannot repair never-failed node {node!r}")
            return self._append_node_lifecycle_event(
                action="repair",
                status="already_repaired",
                node_id=node,
                reason=reason_name,
                metadata=metadata,
                gpus=local_gpus,
                affected_dp_ranks=topology_dp_ranks,
            )

        overlapping_dp_failures = sorted(
            set(self._node_affected_dp_ranks[node]) & self._failed_dp_ranks
        )
        if overlapping_dp_failures:
            raise ValueError(
                "cannot repair node while DP-rank failure is active for "
                f"{overlapping_dp_failures}"
            )
        if self.des.node_dram[node].healthy:
            raise ValueError(f"failed node {node!r} has healthy DRAM")
        healthy_local = [
            spec.gpu.key
            for spec in local_gpus
            if self.des.gpu_hbm[spec.gpu].healthy
        ]
        if healthy_local:
            raise ValueError(
                f"failed node {node!r} has partially repaired GPUs {healthy_local}"
            )

        # All arguments were validated above.  The topology baselines were
        # already capacity-checked at construction, so these operations cannot
        # fail midway under a valid GlobalDES ledger.
        for spec in local_gpus:
            self.des.repair_gpu(
                spec.gpu,
                weights_bytes=spec.weights_bytes,
                runtime_bytes=spec.runtime_bytes,
            )
        node_spec = next(spec for spec in self.topology.nodes if spec.node_id == node)
        self.des.repair_node_dram(
            node,
            runtime_bytes=node_spec.runtime_bytes,
            hicache_primary_bytes=0,
        )
        affected_dp_ranks = self._node_affected_dp_ranks.pop(node)
        self._failed_nodes.remove(node)

        lifecycle_metadata = {
            **metadata,
            "reason": reason_name,
            "node_id": node,
            "affected_dp_ranks": list(affected_dp_ranks),
        }
        for key in self.topology.rank_groups:
            group = self.topology.group(*key)
            if not group or group[0].gpu.node_id != node:
                continue
            projection = self._latest.get(key, self._empty_projection(group))
            self._append_lifecycle_rank_step(
                key=key,
                group=group,
                projection=projection,
                phase="node_repaired_empty",
                event_metadata=lifecycle_metadata,
            )
        return self._append_node_lifecycle_event(
            action="repair",
            status="repaired",
            node_id=node,
            reason=reason_name,
            metadata=metadata,
            gpus=local_gpus,
            affected_dp_ranks=affected_dp_ranks,
        )

    @staticmethod
    def _validate_pool(
        pool: Any,
        *,
        name: str,
        page_size_tokens: int,
        kv_bytes_per_token: int,
    ) -> dict[str, Any]:
        if not isinstance(pool, Mapping):
            raise ValueError(f"{name} must be a mapping")
        values = {
            field: _nonnegative_int(f"{name}.{field}", pool.get(field))
            for field in (
                "capacity_pages",
                "used_pages",
                "free_pages",
                "capacity_tokens",
                "used_tokens",
                "free_tokens",
                "capacity_bytes",
                "used_bytes",
                "free_bytes",
            )
        }
        if values["used_pages"] + values["free_pages"] != values["capacity_pages"]:
            raise ValueError(f"{name} page accounting is inconsistent")
        for prefix in ("capacity", "used", "free"):
            expected_tokens = values[f"{prefix}_pages"] * page_size_tokens
            expected_bytes = expected_tokens * kv_bytes_per_token
            if values[f"{prefix}_tokens"] != expected_tokens:
                raise ValueError(f"{name} {prefix}_tokens is inconsistent")
            if values[f"{prefix}_bytes"] != expected_bytes:
                raise ValueError(f"{name} {prefix}_bytes is inconsistent")

        owners = pool.get("owner_pages")
        if not isinstance(owners, Mapping):
            raise ValueError(f"{name}.owner_pages must be a mapping")
        owner_pages: dict[str, int] = {}
        for owner, pages in owners.items():
            owner_name = str(owner)
            if not owner_name:
                raise ValueError(f"{name}.owner_pages has an empty owner")
            owner_pages[owner_name] = _nonnegative_int(
                f"{name}.owner_pages[{owner_name!r}]", pages
            )
        if sum(owner_pages.values()) != values["used_pages"]:
            raise ValueError(f"{name}.owner_pages does not conserve used_pages")
        values["owner_pages"] = owner_pages
        return values

    @staticmethod
    def _owner_is_active(owner: str, active_request_ids: set[str]) -> bool:
        if owner in active_request_ids:
            return True
        return any(
            owner.startswith(prefix) and request_id == owner[len(prefix) :]
            for prefix in _AUXILIARY_REQUEST_OWNER_PREFIXES
            for request_id in active_request_ids
        )

    def _parse_snapshot(
        self, snapshot: Mapping[str, Any]
    ) -> tuple[
        tuple[int, int],
        tuple[PhysicalGpuSpec, ...],
        _RankProjection,
        dict[str, Any],
    ]:
        if not isinstance(snapshot, Mapping):
            raise ValueError("KV resource snapshot must be a mapping")
        if snapshot.get("schema") != OFFLINE_KV_RESOURCE_SNAPSHOT_SCHEMA:
            raise ValueError(
                "unsupported KV resource snapshot schema: "
                f"{snapshot.get('schema')!r}"
            )
        if snapshot.get("scope") != _SINGLE_TP_VIEW_SCOPE:
            raise ValueError(
                "snapshot must be a one-symmetric-tp-rank-view, not a TP total"
            )

        try:
            timestamp_s = float(snapshot.get("timestamp_s"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "snapshot.timestamp_s must be finite and nonnegative"
            ) from exc
        if timestamp_s < 0 or not math.isfinite(timestamp_s):
            raise ValueError("snapshot.timestamp_s must be finite and nonnegative")
        timestamp_ns = round(timestamp_s * 1_000_000_000)
        if abs(timestamp_ns - self.des.now_ns) > self.clock_tolerance_ns:
            raise ValueError(
                f"snapshot time {timestamp_ns} ns does not match global DES time "
                f"{self.des.now_ns} ns"
            )

        dp_rank = _nonnegative_int("snapshot.dp_rank", snapshot.get("dp_rank"))
        pp_rank = _nonnegative_int("snapshot.pp_rank", snapshot.get("pp_rank"))
        scheduler_tp_rank = _nonnegative_int(
            "snapshot.scheduler_tp_rank", snapshot.get("scheduler_tp_rank")
        )
        logical_tp_size = _nonnegative_int(
            "snapshot.logical_tp_group_size",
            snapshot.get("logical_tp_group_size"),
        )
        if logical_tp_size == 0:
            raise ValueError("snapshot.logical_tp_group_size must be positive")
        group = self.topology.group(dp_rank, pp_rank)
        if logical_tp_size != len(group):
            raise ValueError(
                f"snapshot TP size {logical_tp_size} does not match physical "
                f"group size {len(group)} for dp={dp_rank}, pp={pp_rank}"
            )
        if scheduler_tp_rank >= logical_tp_size:
            raise ValueError("snapshot.scheduler_tp_rank lies outside its TP group")
        key = (dp_rank, pp_rank)
        previous = self._latest.get(key)
        if previous is not None:
            if scheduler_tp_rank != previous.scheduler_tp_rank:
                raise ValueError(
                    "representative scheduler_tp_rank changed for one DP/PP group"
                )
            if timestamp_ns < previous.timestamp_ns:
                raise ValueError("rank snapshot time moved backwards")

        page_size = _nonnegative_int(
            "snapshot.page_size_tokens", snapshot.get("page_size_tokens")
        )
        kv_bytes = _nonnegative_int(
            "snapshot.kv_bytes_per_token_per_tp_rank",
            snapshot.get("kv_bytes_per_token_per_tp_rank"),
        )
        if page_size == 0 or kv_bytes == 0:
            raise ValueError("snapshot page size and KV bytes/token must be positive")
        l1 = self._validate_pool(
            snapshot.get("l1_device"),
            name="snapshot.l1_device",
            page_size_tokens=page_size,
            kv_bytes_per_token=kv_bytes,
        )
        l2 = self._validate_pool(
            snapshot.get("l2_host"),
            name="snapshot.l2_host",
            page_size_tokens=page_size,
            kv_bytes_per_token=kv_bytes,
        )

        running = snapshot.get("running_request_ids")
        waiting = snapshot.get("waiting_request_ids")
        if not isinstance(running, list) or not isinstance(waiting, list):
            raise ValueError(
                "running_request_ids and waiting_request_ids must be lists"
            )
        if any(not isinstance(rid, str) for rid in (*running, *waiting)):
            raise ValueError("active request IDs must be strings")
        active_request_ids = set((*running, *waiting))
        if "" in active_request_ids:
            raise ValueError("active request IDs must be nonempty")

        primary_pages = sum(
            pages
            for owner, pages in l1["owner_pages"].items()
            if self._owner_is_active(owner, active_request_ids)
        )
        prefix_pages = l1["used_pages"] - primary_pages
        bytes_per_page = page_size * kv_bytes
        primary_bytes = primary_pages * bytes_per_page
        prefix_bytes = prefix_pages * bytes_per_page
        if primary_bytes + prefix_bytes != l1["used_bytes"]:
            raise ValueError("L1 owner classification does not conserve bytes")

        l2_used_by_node: dict[str, int] = {}
        l2_capacity_by_node: dict[str, int] = {}
        for spec in group:
            node_id = spec.gpu.node_id
            l2_used_by_node[node_id] = (
                l2_used_by_node.get(node_id, 0) + l2["used_bytes"]
            )
            l2_capacity_by_node[node_id] = (
                l2_capacity_by_node.get(node_id, 0) + l2["capacity_bytes"]
            )
            if (
                spec.weights_bytes
                + spec.runtime_bytes
                + l1["capacity_bytes"]
                > spec.hbm_capacity_bytes
            ):
                raise ValueError(
                    f"Scheduler L1 capacity cannot fit on physical GPU {spec.gpu.key}"
                )

        projection = _RankProjection(
            timestamp_ns=timestamp_ns,
            scheduler_tp_rank=scheduler_tp_rank,
            tp_group_size=logical_tp_size,
            primary_bytes_per_tp_rank=primary_bytes,
            prefix_bytes_per_tp_rank=prefix_bytes,
            l1_capacity_bytes_per_tp_rank=l1["capacity_bytes"],
            l2_used_bytes_per_tp_rank=l2["used_bytes"],
            l2_capacity_bytes_per_tp_rank=l2["capacity_bytes"],
            l2_used_by_node=tuple(sorted(l2_used_by_node.items())),
            l2_capacity_by_node=tuple(sorted(l2_capacity_by_node.items())),
        )
        parsed = {
            "page_size_tokens": page_size,
            "kv_bytes_per_token_per_tp_rank": kv_bytes,
            "l1": l1,
            "l2": l2,
            "running_request_ids": sorted(set(map(str, running))),
            "waiting_request_ids": sorted(set(map(str, waiting))),
            "primary_pages_per_tp_rank": primary_pages,
            "prefix_pages_per_tp_rank": prefix_pages,
        }
        return key, group, projection, parsed

    def _desired_dram(
        self,
        replacement_key: tuple[int, int],
        replacement: _RankProjection,
        *,
        capacity: bool = False,
    ) -> dict[str, int]:
        totals = {node.node_id: 0 for node in self.topology.nodes}
        projections = dict(self._latest)
        projections[replacement_key] = replacement
        attribute = "l2_capacity_by_node" if capacity else "l2_used_by_node"
        for projection in projections.values():
            for node_id, num_bytes in getattr(projection, attribute):
                totals[node_id] += num_bytes
        return totals

    def consume_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        phase: str,
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically replace one DP/PP Scheduler projection.

        ``used_bytes`` in the input is already per TP rank.  Each GPU receives
        that value once; no GPU receives the TP-group total.  Host L2 bytes are
        likewise counted once per physical TP shard and accumulated only into
        the shard's node-local DRAM ledger.
        """

        phase_name = str(phase)
        if not phase_name:
            raise ValueError("phase must be nonempty")
        metadata = self._event_metadata(event_metadata)
        key, group, projection, parsed = self._parse_snapshot(snapshot)
        failed_group_nodes = sorted(
            {spec.gpu.node_id for spec in group} & self._failed_nodes
        )
        if failed_group_nodes:
            raise ValueError(
                f"snapshot cannot allocate on failed node(s) {failed_group_nodes} "
                f"for dp={key[0]}, pp={key[1]}"
            )
        if key[0] in self._failed_dp_ranks:
            if (
                projection.primary_bytes_per_tp_rank
                or projection.prefix_bytes_per_tp_rank
            ):
                raise ValueError(
                    f"nonzero L1 snapshot for failed DP rank {key[0]} is forbidden"
                )
            previous = self._latest.get(key)
            if previous is not None and (
                projection.l2_used_bytes_per_tp_rank
                != previous.l2_used_bytes_per_tp_rank
                or projection.l2_capacity_bytes_per_tp_rank
                != previous.l2_capacity_bytes_per_tp_rank
                or projection.l2_used_by_node != previous.l2_used_by_node
                or projection.l2_capacity_by_node != previous.l2_capacity_by_node
            ):
                raise ValueError(
                    f"snapshot cannot change durable L2 while DP rank {key[0]} "
                    "is failed"
                )
        narrower_failure = key in self._failed_process_groups or any(
            spec.gpu in self._failed_gpus for spec in group
        )
        if narrower_failure:
            if (
                projection.primary_bytes_per_tp_rank
                or projection.prefix_bytes_per_tp_rank
            ):
                raise ValueError(
                    f"nonzero L1 snapshot for unavailable TP group "
                    f"dp={key[0]}, pp={key[1]} is forbidden"
                )
            previous = self._latest.get(key)
            if previous is not None and (
                projection.l2_used_bytes_per_tp_rank
                != previous.l2_used_bytes_per_tp_rank
                or projection.l2_capacity_bytes_per_tp_rank
                != previous.l2_capacity_bytes_per_tp_rank
                or projection.l2_used_by_node != previous.l2_used_by_node
                or projection.l2_capacity_by_node != previous.l2_capacity_by_node
            ):
                raise ValueError(
                    f"snapshot cannot change durable L2 while TP group "
                    f"dp={key[0]}, pp={key[1]} is unavailable"
                )
        desired_dram = self._desired_dram(key, projection)
        desired_dram_capacity = self._desired_dram(
            key, projection, capacity=True
        )

        # Validate the complete replacement before touching any ledger.
        for spec in group:
            ledger = self.des.gpu_hbm[spec.gpu]
            dynamic = (
                projection.primary_bytes_per_tp_rank
                + projection.prefix_bytes_per_tp_rank
            )
            other = ledger.used_bytes - ledger.primary_kv - ledger.prefix_kv
            if dynamic and not ledger.healthy:
                raise ValueError(f"cannot project KV onto unhealthy GPU {spec.gpu.key}")
            if other + dynamic > ledger.capacity_bytes:
                raise ValueError(
                    f"physical HBM capacity exceeded on GPU {spec.gpu.key}"
                )
        for node_spec in self.topology.nodes:
            ledger = self.des.node_dram[node_spec.node_id]
            desired = desired_dram[node_spec.node_id]
            other = ledger.used_bytes - ledger.hicache_primary
            if desired and not ledger.healthy:
                raise ValueError(
                    f"cannot project HiCache L2 onto unhealthy node {node_spec.node_id}"
                )
            if other + desired > ledger.capacity_bytes:
                raise ValueError(
                    f"physical DRAM capacity exceeded on node {node_spec.node_id}"
                )
            if (
                node_spec.runtime_bytes
                + desired_dram_capacity[node_spec.node_id]
                > node_spec.capacity_bytes
            ):
                raise ValueError(
                    f"configured L2 capacity exceeds physical DRAM on node "
                    f"{node_spec.node_id}"
                )

        reason = f"offline_scheduler_snapshot:{phase_name}"
        for spec in group:
            ledger = self.des.gpu_hbm[spec.gpu]
            targets = {
                HbmAccount.PRIMARY_KV: projection.primary_bytes_per_tp_rank,
                HbmAccount.PREFIX_KV: projection.prefix_bytes_per_tp_rank,
            }
            deltas = {
                account: target - int(getattr(ledger, account.value))
                for account, target in targets.items()
            }
            # Free before allocating so a primary->prefix reclassification is
            # safe even when the ledger is exactly at capacity.
            for sign in (-1, 1):
                for account in (HbmAccount.PRIMARY_KV, HbmAccount.PREFIX_KV):
                    delta = deltas[account]
                    if delta and (delta < 0) == (sign < 0):
                        self.des.adjust_gpu_hbm(
                            spec.gpu, account, delta, reason=reason
                        )

        for node_spec in self.topology.nodes:
            ledger = self.des.node_dram[node_spec.node_id]
            delta = desired_dram[node_spec.node_id] - ledger.hicache_primary
            if delta:
                self.des.adjust_node_dram(
                    node_spec.node_id,
                    DramAccount.HICACHE_PRIMARY,
                    delta,
                    reason=reason,
                )

        self._latest[key] = projection
        step = {
            "schema": OFFLINE_RESOURCE_BRIDGE_SCHEMA,
            "sequence": len(self._rank_steps),
            "timestamp_ns": projection.timestamp_ns,
            "des_now_ns": self.des.now_ns,
            "phase": phase_name,
            "event_metadata": metadata,
            "dp_rank": key[0],
            "pp_rank": key[1],
            "scheduler_tp_rank": projection.scheduler_tp_rank,
            "physical_gpus": [spec.gpu.as_dict() for spec in group],
            "primary_pages_per_tp_rank": parsed["primary_pages_per_tp_rank"],
            "prefix_pages_per_tp_rank": parsed["prefix_pages_per_tp_rank"],
            "primary_bytes_per_tp_rank": projection.primary_bytes_per_tp_rank,
            "prefix_bytes_per_tp_rank": projection.prefix_bytes_per_tp_rank,
            "l1_used_bytes_per_tp_rank": parsed["l1"]["used_bytes"],
            "physical_primary_bytes_total": (
                projection.primary_bytes_per_tp_rank * len(group)
            ),
            "physical_prefix_bytes_total": (
                projection.prefix_bytes_per_tp_rank * len(group)
            ),
            "l2_used_bytes_per_tp_rank": projection.l2_used_bytes_per_tp_rank,
            "l2_used_by_node": dict(projection.l2_used_by_node),
            "running_request_ids": parsed["running_request_ids"],
            "waiting_request_ids": parsed["waiting_request_ids"],
            "gpu_hbm_after": {
                spec.gpu.key: self.des.gpu_hbm[spec.gpu].snapshot() for spec in group
            },
            "node_dram_after": {
                node.node_id: self.des.node_dram[node.node_id].snapshot()
                for node in self.topology.nodes
            },
        }
        self._rank_steps.append(copy.deepcopy(step))
        return copy.deepcopy(step)

    def release_provisional_hicache_d2h(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        bytes_per_tp_rank: int,
        reason: str,
        intent_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Remove uncommitted D2H host holds after a hard worker loss.

        The Scheduler allocator reserves L2 pages before publishing an
        ordinary HiCache D2H intent.  A hard fail-stop terminates that worker,
        so it cannot consume the ABORTED ACK and publish a corrected snapshot.
        This parent-side adoption rollback changes only the ordinary
        ``hicache_primary`` projection; committed host copies and every Aegis
        account remain untouched.
        """

        key = (_nonnegative_int("dp_rank", dp_rank), _nonnegative_int("pp_rank", pp_rank))
        group = self.topology.group(*key)
        previous = self._latest.get(key)
        if previous is None:
            raise ValueError(
                f"no Scheduler projection for provisional HiCache rollback {key}"
            )
        release = _nonnegative_int("bytes_per_tp_rank", bytes_per_tp_rank)
        if release <= 0:
            raise ValueError("provisional HiCache rollback must release bytes")
        if release > previous.l2_used_bytes_per_tp_rank:
            raise ValueError(
                f"provisional HiCache rollback exceeds L2 use for {key}: "
                f"{release} > {previous.l2_used_bytes_per_tp_rank}"
            )
        by_node = dict(previous.l2_used_by_node)
        for spec in group:
            available = int(by_node.get(spec.gpu.node_id, 0))
            if available < release:
                raise ValueError(
                    "provisional HiCache rollback exceeds node-local L2 use "
                    f"for {spec.gpu.node_id}: {release} > {available}"
                )
            by_node[spec.gpu.node_id] = available - release
        replacement = replace(
            previous,
            timestamp_ns=self.des.now_ns,
            l2_used_bytes_per_tp_rank=(
                previous.l2_used_bytes_per_tp_rank - release
            ),
            l2_used_by_node=tuple(sorted(by_node.items())),
        )
        desired_dram = self._desired_dram(key, replacement)
        for node in self.topology.nodes:
            ledger = self.des.node_dram[node.node_id]
            delta = desired_dram[node.node_id] - ledger.hicache_primary
            if delta:
                if delta > 0:
                    raise RuntimeError(
                        "provisional HiCache rollback attempted to allocate DRAM"
                    )
                self.des.adjust_node_dram(
                    node.node_id,
                    DramAccount.HICACHE_PRIMARY,
                    delta,
                    reason=str(reason),
                )
        self._latest[key] = replacement
        metadata = {
            "reason": str(reason),
            "intent_ids": sorted(str(value) for value in intent_ids),
            "released_bytes_per_tp_rank": release,
        }
        self._append_lifecycle_rank_step(
            key=key,
            group=group,
            projection=replacement,
            phase="ordinary_hicache_d2h_aborted",
            event_metadata=metadata,
        )
        return {
            "dp_rank": key[0],
            "pp_rank": key[1],
            "released_bytes_per_tp_rank": release,
            "physical_released_bytes": release * len(group),
            "intent_ids": metadata["intent_ids"],
            "node_dram_after": {
                node.node_id: self.des.node_dram[node.node_id].snapshot()
                for node in self.topology.nodes
            },
        }

    def export(self) -> dict[str, Any]:
        """Return a defensive, JSON-compatible bridge artifact."""

        return copy.deepcopy(
            {
                "schema": OFFLINE_RESOURCE_BRIDGE_SCHEMA,
                "topology": self.topology.as_dict(),
                "rank_steps": self._rank_steps,
                "failed_dp_ranks": sorted(self._failed_dp_ranks),
                "failed_process_groups": [
                    {"dp_rank": dp_rank, "pp_rank": pp_rank}
                    for dp_rank, pp_rank in sorted(self._failed_process_groups)
                ],
                "failed_physical_gpu_ids": [
                    gpu.key for gpu in sorted(self._failed_gpus)
                ],
                "failed_nodes": sorted(self._failed_nodes),
                "affected_dp_ranks_by_failed_node": {
                    node_id: list(self._node_affected_dp_ranks[node_id])
                    for node_id in sorted(self._failed_nodes)
                },
                "node_lifecycle_events": [
                    event
                    for event in self._lifecycle_events
                    if event.get("failure_domain") == "node"
                ],
                "lifecycle_events": self._lifecycle_events,
                "latest_rank_views": {
                    f"dp{dp}/pp{pp}": projection.as_dict()
                    for (dp, pp), projection in sorted(self._latest.items())
                },
                "physical_resources": self.des.snapshot(),
            }
        )

    def audit(self, *, raise_on_error: bool = False) -> list[str]:
        """Audit topology, replacement accounting, and byte conservation."""

        errors = list(self.des.audit_invariants())
        expected_gpu: dict[PhysicalGpuId, tuple[int, int]] = {
            spec.gpu: (0, 0) for spec in self.topology.gpus
        }
        expected_dram = {node.node_id: 0 for node in self.topology.nodes}
        expected_dram_capacity = {node.node_id: 0 for node in self.topology.nodes}

        known_nodes = {node.node_id for node in self.topology.nodes}
        if not self._failed_nodes <= known_nodes:
            errors.append("failed node set contains a node outside the topology")
        if set(self._node_affected_dp_ranks) != self._failed_nodes:
            errors.append("failed-node affected-DP map is not lifecycle-consistent")
        for node_id in sorted(self._failed_nodes):
            expected_affected = tuple(
                sorted(
                    {
                        spec.gpu.dp_rank
                        for spec in self.topology.gpus
                        if spec.gpu.node_id == node_id
                    }
                )
            )
            if self._node_affected_dp_ranks.get(node_id) != expected_affected:
                errors.append(
                    f"failed node {node_id} has incorrect affected DP ranks"
                )
            node_ledger = self.des.node_dram.get(node_id)
            if node_ledger is not None and node_ledger.healthy:
                errors.append(f"failed node {node_id} has healthy DRAM")
            for spec in self.topology.gpus:
                if (
                    spec.gpu.node_id == node_id
                    and self.des.gpu_hbm[spec.gpu].healthy
                ):
                    errors.append(
                        f"failed node {node_id} has healthy GPU {spec.gpu.key}"
                    )

        for dp_rank in self._failed_dp_ranks:
            for _key, group in self._dp_groups(dp_rank):
                for spec in group:
                    if self.des.gpu_hbm[spec.gpu].healthy:
                        errors.append(
                            f"failed DP rank {dp_rank} has healthy GPU {spec.gpu.key}"
                        )

        known_process_groups = set(self.topology.rank_groups)
        if not self._failed_process_groups <= known_process_groups:
            errors.append("failed process set contains a group outside the topology")
        for key in sorted(self._failed_process_groups):
            for spec in self.topology.group(*key):
                if self.des.gpu_hbm[spec.gpu].healthy:
                    errors.append(
                        f"failed process dp{key[0]}/pp{key[1]} has healthy GPU "
                        f"{spec.gpu.key}"
                    )

        known_gpus = {spec.gpu for spec in self.topology.gpus}
        if not self._failed_gpus <= known_gpus:
            errors.append("failed GPU set contains an ID outside the topology")
        for gpu in sorted(self._failed_gpus):
            if self.des.gpu_hbm[gpu].healthy:
                errors.append(f"failed physical GPU {gpu.key} is healthy")
            key = (gpu.dp_rank, gpu.pp_rank)
            projection = self._latest.get(key)
            if projection is not None and (
                projection.primary_bytes_per_tp_rank
                or projection.prefix_bytes_per_tp_rank
            ):
                errors.append(
                    f"incomplete TP group dp{gpu.dp_rank}/pp{gpu.pp_rank} "
                    "retains ordinary L1 projection"
                )

        for key, projection in self._latest.items():
            group = self.topology.group(*key)
            if projection.tp_group_size != len(group):
                errors.append(f"dp{key[0]}/pp{key[1]} TP size changed")
            for spec in group:
                expected_gpu[spec.gpu] = (
                    projection.primary_bytes_per_tp_rank,
                    projection.prefix_bytes_per_tp_rank,
                )
            for node_id, value in projection.l2_used_by_node:
                expected_dram[node_id] += value
            for node_id, value in projection.l2_capacity_by_node:
                expected_dram_capacity[node_id] += value
            failed_group_nodes = {
                spec.gpu.node_id for spec in group
            } & self._failed_nodes
            if failed_group_nodes and (
                projection.primary_bytes_per_tp_rank
                or projection.prefix_bytes_per_tp_rank
            ):
                errors.append(
                    f"failed node projection retains L1 for dp{key[0]}/pp{key[1]}"
                )
            for node_id in failed_group_nodes:
                if dict(projection.l2_used_by_node).get(node_id, 0):
                    errors.append(
                        f"failed node {node_id} projection retains HiCache L2"
                    )

        actual_primary = 0
        actual_prefix = 0
        expected_primary = 0
        expected_prefix = 0
        expected_gpu_specs = {spec.gpu: spec for spec in self.topology.gpus}
        if set(self.des.gpu_hbm) != set(expected_gpu_specs):
            errors.append("registered GPU set does not match physical topology")
        for gpu, (primary, prefix) in expected_gpu.items():
            ledger = self.des.gpu_hbm.get(gpu)
            if ledger is None:
                errors.append(f"topology GPU {gpu.key} is not registered")
                continue
            if ledger.capacity_bytes != expected_gpu_specs[gpu].hbm_capacity_bytes:
                errors.append(f"GPU {gpu.key} HBM capacity differs from topology")
            if ledger.healthy and (
                ledger.weights != expected_gpu_specs[gpu].weights_bytes
                or ledger.runtime != expected_gpu_specs[gpu].runtime_bytes
            ):
                errors.append(f"GPU {gpu.key} baseline differs from topology")
            expected_primary += primary
            expected_prefix += prefix
            actual_primary += ledger.primary_kv
            actual_prefix += ledger.prefix_kv
            if ledger.primary_kv != primary or ledger.prefix_kv != prefix:
                errors.append(
                    f"GPU {gpu.key} primary/prefix does not match latest rank view"
                )
        if actual_primary != expected_primary or actual_prefix != expected_prefix:
            errors.append("physical GPU primary/prefix byte conservation failed")

        actual_dram = 0
        expected_node_specs = {node.node_id: node for node in self.topology.nodes}
        if set(self.des.node_dram) != set(expected_node_specs):
            errors.append("registered DRAM node set does not match physical topology")
        for node in self.topology.nodes:
            ledger = self.des.node_dram.get(node.node_id)
            if ledger is None:
                errors.append(f"topology node {node.node_id} is not registered")
                continue
            if ledger.capacity_bytes != node.capacity_bytes:
                errors.append(
                    f"node {node.node_id} DRAM capacity differs from topology"
                )
            actual_dram += ledger.hicache_primary
            if ledger.hicache_primary != expected_dram[node.node_id]:
                errors.append(
                    f"node {node.node_id} HiCache bytes do not match rank projections"
                )
            if (
                node.runtime_bytes + expected_dram_capacity[node.node_id]
                > node.capacity_bytes
            ):
                errors.append(
                    f"node {node.node_id} configured L2 capacity exceeds DRAM"
                )
            if node.node_id in self._failed_nodes and ledger.used_bytes != 0:
                errors.append(
                    f"failed node {node.node_id} DRAM ledger is not empty"
                )
        if actual_dram != sum(expected_dram.values()):
            errors.append("node-local HiCache DRAM byte conservation failed")

        latest_step_time: dict[tuple[int, int], int] = {}
        for step in self._rank_steps:
            key = (int(step["dp_rank"]), int(step["pp_rank"]))
            timestamp = int(step["timestamp_ns"])
            if timestamp < latest_step_time.get(key, -1):
                errors.append(
                    f"rank step time moved backwards for dp{key[0]}/pp{key[1]}"
                )
            latest_step_time[key] = timestamp
            physical_count = len(step["physical_gpus"])
            if step["physical_primary_bytes_total"] != (
                step["primary_bytes_per_tp_rank"] * physical_count
            ):
                errors.append(
                    f"rank step {step['sequence']} primary bytes are not conserved"
                )
            if step["physical_prefix_bytes_total"] != (
                step["prefix_bytes_per_tp_rank"] * physical_count
            ):
                errors.append(
                    f"rank step {step['sequence']} prefix bytes are not conserved"
                )
            if step["l1_used_bytes_per_tp_rank"] != (
                step["primary_bytes_per_tp_rank"]
                + step["prefix_bytes_per_tp_rank"]
            ):
                errors.append(
                    f"rank step {step['sequence']} L1 bytes are not conserved"
                )
            if sum(step["l2_used_by_node"].values()) != (
                step["l2_used_bytes_per_tp_rank"] * physical_count
            ):
                errors.append(
                    f"rank step {step['sequence']} L2 bytes are not conserved"
                )

        lifecycle_failed_nodes: set[str] = set()
        for sequence, event in enumerate(self._lifecycle_events):
            if event.get("sequence") != sequence:
                errors.append("resource lifecycle event sequence is not contiguous")
            if event.get("failure_domain") != "node":
                continue
            node_id = str(event.get("node_id", ""))
            if node_id not in known_nodes:
                errors.append(f"node lifecycle references unknown node {node_id!r}")
                continue
            expected_affected = sorted(
                {
                    spec.gpu.dp_rank
                    for spec in self.topology.gpus
                    if spec.gpu.node_id == node_id
                }
            )
            if event.get("affected_dp_ranks") != expected_affected:
                errors.append(
                    f"node lifecycle for {node_id} has incorrect affected DP ranks"
                )
            action_status = (event.get("action"), event.get("status"))
            if action_status == ("fail", "failed"):
                lifecycle_failed_nodes.add(node_id)
            elif action_status == ("fail", "already_failed"):
                if node_id not in lifecycle_failed_nodes:
                    errors.append(
                        f"node {node_id} already_failed event has no active failure"
                    )
            elif action_status == ("repair", "repaired"):
                if node_id not in lifecycle_failed_nodes:
                    errors.append(
                        f"node {node_id} repair event has no active failure"
                    )
                lifecycle_failed_nodes.discard(node_id)
            elif action_status == ("repair", "already_repaired"):
                if node_id in lifecycle_failed_nodes:
                    errors.append(
                        f"node {node_id} already_repaired event is still failed"
                    )
            else:
                errors.append(
                    f"node lifecycle has unsupported action/status {action_status}"
                )
        if lifecycle_failed_nodes != self._failed_nodes:
            errors.append("node lifecycle history disagrees with failed-node state")

        if raise_on_error and errors:
            raise RuntimeError("; ".join(errors))
        return errors


__all__ = [
    "NodeDramSpec",
    "OfflinePhysicalTopology",
    "OfflineResourceBridge",
    "PhysicalGpuSpec",
]
