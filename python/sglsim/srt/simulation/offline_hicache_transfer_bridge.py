"""Canonical ordinary SGLang HiCache L1<->L2 transfers for offline DES.

The Scheduler process owns one symmetric TP-rank allocator view.  It publishes
logical intents at the exact pinned SGLang control-flow points.  This parent
adapter expands each intent over the physical ``(dp, pp, tp)`` group and
reserves the same copy-engine/PCIe/DRAM resources used by AegisKV transfers.

Capacity remains owned by Scheduler allocator snapshots: D2H host pages and
H2D device pages are provisionally allocated before the intent is emitted.
Consequently this bridge never writes Aegis accounts and never adds a second
ordinary-KV copy to the HBM/DRAM ledgers.  Its lifecycle is nevertheless
authoritative for recoverability: only a group-wide COMMITTED ACK makes D2H
host metadata visible, and H2D compute may start only after the same barrier.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from sglsim.srt.mem_cache.hicache import OFFLINE_HICACHE_INTENT_SCHEMA
from sglsim.srt.simulation.global_des import (
    EventPriority,
    GlobalDES,
    PhysicalGpuId,
    TransferResourceId,
)
from sglsim.srt.simulation.offline_aegis_transfer_bridge import CopyFabricConfig
from sglsim.srt.simulation.offline_resource_bridge import OfflinePhysicalTopology


OFFLINE_HICACHE_TRANSFER_BRIDGE_SCHEMA = (
    "sglsim.offline.hicache-physical-transfer-bridge.v1"
)


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _int_field(raw: Mapping[str, Any], name: str, *, minimum: int = 0) -> int:
    value = raw.get(name)
    if isinstance(value, bool):
        raise ValueError(f"ordinary HiCache {name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"ordinary HiCache {name} must be an integer") from exc
    if parsed != value or parsed < minimum:
        raise ValueError(f"ordinary HiCache {name} must be >= {minimum}")
    return parsed


class OrdinaryHiCacheDirection(str, Enum):
    D2H = "d2h"
    H2D = "h2d"


class OrdinaryHiCacheTransferState(str, Enum):
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    ACKED = "ACKED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class OrdinaryHiCacheIntent:
    intent_id: str
    direction: OrdinaryHiCacheDirection
    owner: str
    radix_node_id: int
    host_page_ids: tuple[int, ...]
    device_page_ids: tuple[int, ...]
    page_size_tokens: int
    page_count: int
    token_count: int
    kv_bytes_per_token_per_tp_rank: int
    size_bytes_per_tp_rank: int
    dp_rank: int
    pp_rank: int
    scheduler_tp_rank: int
    worker_incarnation: int
    operation_sequence: int
    predicted_duration_s: float
    blocks_compute: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OrdinaryHiCacheIntent":
        if not isinstance(raw, Mapping):
            raise TypeError("ordinary HiCache intent must be a mapping")
        if raw.get("schema") != OFFLINE_HICACHE_INTENT_SCHEMA:
            raise ValueError(
                f"unsupported ordinary HiCache intent schema {raw.get('schema')!r}"
            )
        intent_id = str(raw.get("intent_id") or "")
        owner = str(raw.get("owner") or "")
        if not intent_id or not owner:
            raise ValueError("ordinary HiCache intent ID/owner must be nonempty")
        direction = OrdinaryHiCacheDirection(str(raw.get("direction") or ""))
        if str(raw.get("state") or "") != "PENDING":
            raise ValueError("new ordinary HiCache intent must be PENDING")
        host_pages = tuple(int(page) for page in raw.get("host_page_ids", ()))
        device_pages = tuple(int(page) for page in raw.get("device_page_ids", ()))
        if not host_pages or len(host_pages) != len(device_pages):
            raise ValueError("ordinary HiCache page vectors must be nonempty/equal")
        if any(page < 0 for page in (*host_pages, *device_pages)):
            raise ValueError("ordinary HiCache page IDs must be nonnegative")
        if len(set(host_pages)) != len(host_pages):
            raise ValueError("ordinary HiCache host pages must be unique")
        if len(set(device_pages)) != len(device_pages):
            raise ValueError("ordinary HiCache device pages must be unique")
        page_count = _int_field(raw, "page_count", minimum=1)
        page_size = _int_field(raw, "page_size_tokens", minimum=1)
        token_count = _int_field(raw, "token_count", minimum=1)
        kv_bytes = _int_field(
            raw, "kv_bytes_per_token_per_tp_rank", minimum=1
        )
        size_bytes = _int_field(raw, "size_bytes_per_tp_rank", minimum=1)
        if page_count != len(host_pages):
            raise ValueError("ordinary HiCache page_count is inconsistent")
        if token_count != page_count * page_size:
            raise ValueError("ordinary HiCache token_count is inconsistent")
        if size_bytes != token_count * kv_bytes:
            raise ValueError("ordinary HiCache byte count is inconsistent")
        try:
            predicted = float(raw.get("predicted_duration_s", 0.0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("predicted_duration_s must be nonnegative") from exc
        if not math.isfinite(predicted) or predicted < 0.0:
            raise ValueError("predicted_duration_s must be finite/nonnegative")
        if raw.get("logical_tp_scope") != "one-symmetric-tp-rank-view":
            raise ValueError("ordinary HiCache intent must be a logical TP view")
        if raw.get("requires_parent_tp_expansion") is not True:
            raise ValueError("ordinary HiCache intent must require TP expansion")
        blocks_compute = raw.get("blocks_compute", False)
        if not isinstance(blocks_compute, bool):
            raise ValueError("ordinary HiCache blocks_compute must be boolean")
        if direction is OrdinaryHiCacheDirection.H2D and not blocks_compute:
            raise ValueError("ordinary HiCache H2D must block compute until ACK")
        return cls(
            intent_id=intent_id,
            direction=direction,
            owner=owner,
            radix_node_id=_int_field(raw, "radix_node_id"),
            host_page_ids=host_pages,
            device_page_ids=device_pages,
            page_size_tokens=page_size,
            page_count=page_count,
            token_count=token_count,
            kv_bytes_per_token_per_tp_rank=kv_bytes,
            size_bytes_per_tp_rank=size_bytes,
            dp_rank=_int_field(raw, "dp_rank"),
            pp_rank=_int_field(raw, "pp_rank"),
            scheduler_tp_rank=_int_field(raw, "scheduler_tp_rank"),
            worker_incarnation=_int_field(raw, "worker_incarnation"),
            operation_sequence=_int_field(raw, "operation_sequence"),
            predicted_duration_s=predicted,
            blocks_compute=blocks_compute,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OFFLINE_HICACHE_INTENT_SCHEMA,
            "intent_id": self.intent_id,
            "direction": self.direction.value,
            "owner": self.owner,
            "radix_node_id": self.radix_node_id,
            "host_page_ids": list(self.host_page_ids),
            "device_page_ids": list(self.device_page_ids),
            "page_size_tokens": self.page_size_tokens,
            "page_count": self.page_count,
            "token_count": self.token_count,
            "kv_bytes_per_token_per_tp_rank": (
                self.kv_bytes_per_token_per_tp_rank
            ),
            "size_bytes_per_tp_rank": self.size_bytes_per_tp_rank,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "scheduler_tp_rank": self.scheduler_tp_rank,
            "worker_incarnation": self.worker_incarnation,
            "operation_sequence": self.operation_sequence,
            "predicted_duration_s": self.predicted_duration_s,
            "blocks_compute": self.blocks_compute,
            "logical_tp_scope": "one-symmetric-tp-rank-view",
            "requires_parent_tp_expansion": True,
        }


@dataclass
class PhysicalOrdinaryHiCacheTransfer:
    transfer_id: str
    intent_id: str
    direction: OrdinaryHiCacheDirection
    owner: str
    gpu: PhysicalGpuId
    node_id: str
    tp_rank: int
    page_count: int
    token_count: int
    size_bytes: int
    ready_ns: int
    start_ns: int
    end_ns: int
    resources: tuple[TransferResourceId, ...]
    state: OrdinaryHiCacheTransferState = OrdinaryHiCacheTransferState.PENDING
    start_event_id: str = ""
    completion_event_id: str = ""
    failure_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        gpu_location = {
            "tier": "gpu_prefix",
            "gpu": self.gpu.as_dict(),
            "node_id": None,
        }
        host_location = {
            "tier": "node_hicache_primary",
            "gpu": None,
            "node_id": self.node_id,
        }
        source, destination = (
            (gpu_location, host_location)
            if self.direction is OrdinaryHiCacheDirection.D2H
            else (host_location, gpu_location)
        )
        return {
            "transfer_id": self.transfer_id,
            "intent_id": self.intent_id,
            "direction": self.direction.value,
            "transfer_class": "ordinary_hicache",
            "path_kind": f"ordinary_same_node_{self.direction.value}",
            "path": f"ordinary_same_node_{self.direction.value}",
            "owner": self.owner,
            "gpu": self.gpu.as_dict(),
            "node_id": self.node_id,
            "tp_rank": self.tp_rank,
            "page_count": self.page_count,
            "token_count": self.token_count,
            "size_bytes": self.size_bytes,
            "ready_ns": self.ready_ns,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.end_ns - self.start_ns,
            "queue_delay_ns": self.start_ns - self.ready_ns,
            "resources": [resource.key for resource in self.resources],
            "source": source,
            "destination": destination,
            "state": self.state.value,
            "start_event_id": self.start_event_id,
            "completion_event_id": self.completion_event_id,
            "failure_reason": self.failure_reason,
            "capacity_accounting": "external_scheduler_allocator_snapshot",
        }


@dataclass
class OrdinaryHiCacheTransferGroup:
    intent: OrdinaryHiCacheIntent
    expected_transfer_ids: tuple[str, ...]
    transfer_ids: list[str] = field(default_factory=list)
    state: OrdinaryHiCacheTransferState = OrdinaryHiCacheTransferState.PENDING
    terminal_ns: int | None = None
    failure_reason: str = ""
    callback_delivered: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.as_dict(),
            "expected_transfer_ids": list(self.expected_transfer_ids),
            "transfer_ids": list(self.transfer_ids),
            "state": self.state.value,
            "terminal_ns": self.terminal_ns,
            "failure_reason": self.failure_reason,
            "callback_delivered": self.callback_delivered,
            "ack": {
                "intent_id": self.intent.intent_id,
                "state": self.state.value,
            },
        }


GroupTerminalCallback = Callable[[dict[str, Any]], None]


class OfflineHiCacheTransferBridge:
    """Expand logical ordinary HiCache operations into physical TP copies."""

    def __init__(
        self,
        des: GlobalDES,
        topology: OfflinePhysicalTopology,
        *,
        fabric: CopyFabricConfig | None = None,
        on_group_terminal: GroupTerminalCallback | None = None,
    ) -> None:
        if not isinstance(des, GlobalDES):
            raise TypeError("des must be GlobalDES")
        if not isinstance(topology, OfflinePhysicalTopology):
            raise TypeError("topology must be OfflinePhysicalTopology")
        self.des = des
        self.topology = topology
        self.fabric = fabric or CopyFabricConfig()
        self.on_group_terminal = on_group_terminal
        self.groups: dict[str, OrdinaryHiCacheTransferGroup] = {}
        self.transfers: dict[str, PhysicalOrdinaryHiCacheTransfer] = {}
        self.audit_rows: list[dict[str, Any]] = []

    def _resources(
        self,
        direction: OrdinaryHiCacheDirection,
        gpu: PhysicalGpuId,
    ) -> tuple[tuple[TransferResourceId, ...], int, int]:
        bandwidths = self.fabric.bandwidths
        setup = self.fabric.setup
        if direction is OrdinaryHiCacheDirection.D2H:
            return (
                (
                    TransferResourceId("gpu-d2h-read", gpu.key),
                    TransferResourceId("pcie-d2h", gpu.node_id),
                    TransferResourceId("dram-write", gpu.node_id),
                ),
                setup.same_node_d2h_ns,
                min(
                    bandwidths.gpu_copy_bytes_per_s,
                    bandwidths.pcie_bytes_per_s,
                    bandwidths.dram_write_bytes_per_s,
                ),
            )
        return (
            (
                TransferResourceId("dram-read", gpu.node_id),
                TransferResourceId("pcie-h2d", gpu.node_id),
                TransferResourceId("gpu-h2d-write", gpu.key),
            ),
            setup.same_node_h2d_ns,
            min(
                bandwidths.dram_read_bytes_per_s,
                bandwidths.pcie_bytes_per_s,
                bandwidths.gpu_copy_bytes_per_s,
            ),
        )

    @staticmethod
    def _duration_ns(size_bytes: int, setup_ns: int, bandwidth: int) -> int:
        return setup_ns + (
            int(size_bytes) * 1_000_000_000 + int(bandwidth) - 1
        ) // int(bandwidth)

    def _validate_external_capacity(
        self,
        intent: OrdinaryHiCacheIntent,
        gpu: PhysicalGpuId,
    ) -> None:
        gpu_ledger = self.des.gpu_hbm.get(gpu)
        node_ledger = self.des.node_dram.get(gpu.node_id)
        if gpu_ledger is None or node_ledger is None:
            raise ValueError("ordinary HiCache intent targets unregistered topology")
        if not gpu_ledger.healthy or not node_ledger.healthy:
            raise ValueError("ordinary HiCache intent targets unhealthy resource")
        # The Scheduler allocator already holds the destination and the source.
        # These are conservative aggregate checks; exact page IDs remain in the
        # worker intent and are reconciled by the next allocator snapshot.
        if intent.direction is OrdinaryHiCacheDirection.D2H:
            if node_ledger.hicache_primary < intent.size_bytes_per_tp_rank:
                raise ValueError(
                    "D2H provisional host capacity is absent from DRAM snapshot"
                )
            if (
                gpu_ledger.primary_kv + gpu_ledger.prefix_kv
                < intent.size_bytes_per_tp_rank
            ):
                raise ValueError("D2H source capacity is absent from HBM snapshot")
        else:
            if node_ledger.hicache_primary < intent.size_bytes_per_tp_rank:
                raise ValueError("H2D source capacity is absent from DRAM snapshot")
            if (
                gpu_ledger.primary_kv + gpu_ledger.prefix_kv
                < intent.size_bytes_per_tp_rank
            ):
                raise ValueError(
                    "H2D provisional destination is absent from HBM snapshot"
                )

    def submit_intent(
        self,
        raw_intent: Mapping[str, Any],
        *,
        ready_ns: int | None = None,
    ) -> OrdinaryHiCacheTransferGroup:
        intent = OrdinaryHiCacheIntent.from_mapping(raw_intent)
        existing = self.groups.get(intent.intent_id)
        if existing is not None:
            if existing.intent != intent:
                raise ValueError(
                    f"ordinary HiCache intent {intent.intent_id} changed on replay"
                )
            return existing
        group_specs = self.topology.group(intent.dp_rank, intent.pp_rank)
        if intent.scheduler_tp_rank >= len(group_specs):
            raise ValueError("Scheduler TP rank lies outside physical TP group")
        resolved_ready = self.des.now_ns if ready_ns is None else int(ready_ns)
        if resolved_ready < self.des.now_ns:
            raise ValueError("ordinary HiCache ready_ns precedes DES cursor")
        transfer_ids = tuple(
            _stable_id("hicache-xfer", intent.intent_id, spec.gpu.key)
            for spec in group_specs
        )
        group = OrdinaryHiCacheTransferGroup(
            intent=intent,
            expected_transfer_ids=transfer_ids,
        )
        self.groups[intent.intent_id] = group
        try:
            for spec, transfer_id in zip(group_specs, transfer_ids, strict=True):
                self._validate_external_capacity(intent, spec.gpu)
                resources, setup_ns, bandwidth = self._resources(
                    intent.direction, spec.gpu
                )
                reservation = self.des.reserve_resources(
                    resources,
                    duration_ns=self._duration_ns(
                        intent.size_bytes_per_tp_rank,
                        setup_ns,
                        bandwidth,
                    ),
                    ready_ns=resolved_ready,
                )
                transfer = PhysicalOrdinaryHiCacheTransfer(
                    transfer_id=transfer_id,
                    intent_id=intent.intent_id,
                    direction=intent.direction,
                    owner=intent.owner,
                    gpu=spec.gpu,
                    node_id=spec.gpu.node_id,
                    tp_rank=spec.gpu.tp_rank,
                    page_count=intent.page_count,
                    token_count=intent.token_count,
                    size_bytes=intent.size_bytes_per_tp_rank,
                    ready_ns=reservation.ready_ns,
                    start_ns=reservation.start_ns,
                    end_ns=reservation.end_ns,
                    resources=reservation.resources,
                )
                self.transfers[transfer_id] = transfer
                group.transfer_ids.append(transfer_id)
                start_event = self.des.schedule_at(
                    transfer.start_ns,
                    kind="ordinary_hicache_transfer_start",
                    priority=EventPriority.TRANSFER_START,
                    payload={"transfer_id": transfer_id},
                    callback=lambda simulator, event, tid=transfer_id: (
                        self._start_transfer(tid)
                    ),
                )
                completion_event = self.des.schedule_at(
                    transfer.end_ns,
                    kind="ordinary_hicache_transfer_completion",
                    priority=EventPriority.TRANSFER_COMPLETION,
                    payload={"transfer_id": transfer_id},
                    callback=lambda simulator, event, tid=transfer_id: (
                        self._complete_transfer(tid)
                    ),
                )
                transfer.start_event_id = start_event.event_id
                transfer.completion_event_id = completion_event.event_id
        except BaseException as exc:
            self.abort_intent(
                intent.intent_id,
                reason=f"admission_failed:{type(exc).__name__}:{exc}",
            )
            raise
        if tuple(group.transfer_ids) != group.expected_transfer_ids:
            self.abort_intent(intent.intent_id, reason="TP expansion incomplete")
            raise RuntimeError("ordinary HiCache TP expansion is incomplete")
        self.audit_rows.append(
            {
                "timestamp_ns": self.des.now_ns,
                "event": "intent_submitted",
                "intent_id": intent.intent_id,
                "direction": intent.direction.value,
                "physical_tp_shards": len(group.transfer_ids),
                "transfer_ids": list(group.transfer_ids),
            }
        )
        return group

    def _endpoint_healthy(self, transfer: PhysicalOrdinaryHiCacheTransfer) -> bool:
        gpu = self.des.gpu_hbm.get(transfer.gpu)
        node = self.des.node_dram.get(transfer.node_id)
        return bool(gpu and node and gpu.healthy and node.healthy)

    def _start_transfer(self, transfer_id: str) -> None:
        transfer = self.transfers[transfer_id]
        if transfer.state is OrdinaryHiCacheTransferState.ABORTED:
            return
        if transfer.state is not OrdinaryHiCacheTransferState.PENDING:
            raise RuntimeError(f"ordinary HiCache transfer {transfer_id} started twice")
        if not self._endpoint_healthy(transfer):
            self.abort_intent(
                transfer.intent_id,
                reason=f"endpoint_unhealthy_at_start:{transfer.gpu.key}",
            )
            return
        transfer.state = OrdinaryHiCacheTransferState.IN_FLIGHT

    def _complete_transfer(self, transfer_id: str) -> None:
        transfer = self.transfers[transfer_id]
        if transfer.state is OrdinaryHiCacheTransferState.ABORTED:
            return
        if transfer.state not in {
            OrdinaryHiCacheTransferState.PENDING,
            OrdinaryHiCacheTransferState.IN_FLIGHT,
        }:
            raise RuntimeError(
                f"ordinary HiCache transfer {transfer_id} completed twice"
            )
        if not self._endpoint_healthy(transfer):
            self.abort_intent(
                transfer.intent_id,
                reason=f"endpoint_unhealthy_at_completion:{transfer.gpu.key}",
            )
            return
        transfer.state = OrdinaryHiCacheTransferState.ACKED
        group = self.groups[transfer.intent_id]
        if all(
            self.transfers[tid].state
            is OrdinaryHiCacheTransferState.ACKED
            for tid in group.expected_transfer_ids
        ):
            # One parent event owns the all-TP visibility boundary.  No shard
            # is recoverable merely because its DMA completed earlier.
            for transfer_id in group.expected_transfer_ids:
                self.transfers[
                    transfer_id
                ].state = OrdinaryHiCacheTransferState.COMMITTED
            group.state = OrdinaryHiCacheTransferState.COMMITTED
            group.terminal_ns = self.des.now_ns
            self._notify_terminal(group)

    def abort_intent(
        self,
        intent_id: str,
        *,
        reason: str,
    ) -> OrdinaryHiCacheTransferGroup:
        group = self.groups[str(intent_id)]
        if group.state is OrdinaryHiCacheTransferState.ABORTED:
            return group
        if group.state is OrdinaryHiCacheTransferState.COMMITTED:
            raise ValueError("cannot abort committed ordinary HiCache intent")
        group.state = OrdinaryHiCacheTransferState.ABORTED
        group.terminal_ns = self.des.now_ns
        group.failure_reason = str(reason)
        for transfer_id in group.transfer_ids:
            transfer = self.transfers[transfer_id]
            if transfer.state in {
                OrdinaryHiCacheTransferState.PENDING,
                OrdinaryHiCacheTransferState.IN_FLIGHT,
                OrdinaryHiCacheTransferState.ACKED,
            }:
                transfer.state = OrdinaryHiCacheTransferState.ABORTED
                transfer.failure_reason = str(reason)
        self._notify_terminal(group)
        return group

    def fail_gpu(self, gpu: PhysicalGpuId, *, reason: str = "gpu_failed") -> int:
        affected = {
            transfer.intent_id
            for transfer in self.transfers.values()
            if transfer.gpu == gpu
            and transfer.state
            in {
                OrdinaryHiCacheTransferState.PENDING,
                OrdinaryHiCacheTransferState.IN_FLIGHT,
                OrdinaryHiCacheTransferState.ACKED,
            }
        }
        for intent_id in sorted(affected):
            self.abort_intent(intent_id, reason=f"{reason}:{gpu.key}")
        return len(affected)

    def fail_node(self, node_id: str, *, reason: str = "node_failed") -> int:
        node = str(node_id)
        affected = {
            transfer.intent_id
            for transfer in self.transfers.values()
            if transfer.node_id == node
            and transfer.state
            in {
                OrdinaryHiCacheTransferState.PENDING,
                OrdinaryHiCacheTransferState.IN_FLIGHT,
                OrdinaryHiCacheTransferState.ACKED,
            }
        }
        for intent_id in sorted(affected):
            self.abort_intent(intent_id, reason=f"{reason}:{node}")
        return len(affected)

    def _notify_terminal(self, group: OrdinaryHiCacheTransferGroup) -> None:
        if group.callback_delivered:
            return
        group.callback_delivered = True
        row = {
            "intent_id": group.intent.intent_id,
            "state": group.state.value,
            "direction": group.intent.direction.value,
            "dp_rank": group.intent.dp_rank,
            "pp_rank": group.intent.pp_rank,
            "scheduler_tp_rank": group.intent.scheduler_tp_rank,
            "worker_incarnation": group.intent.worker_incarnation,
            "blocks_compute": group.intent.blocks_compute,
            "terminal_ns": group.terminal_ns,
            "failure_reason": group.failure_reason,
            "transfer_ids": list(group.transfer_ids),
        }
        self.audit_rows.append(
            {
                "timestamp_ns": self.des.now_ns,
                "event": "intent_terminal",
                **copy.deepcopy(row),
            }
        )
        if self.on_group_terminal is not None:
            self.on_group_terminal(copy.deepcopy(row))

    def audit(self, *, require_terminal: bool = True) -> list[str]:
        errors: list[str] = []
        bound = {
            transfer_id
            for group in self.groups.values()
            for transfer_id in group.transfer_ids
        }
        orphan = sorted(set(self.transfers) - bound)
        if orphan:
            errors.append(f"ordinary HiCache orphan transfers: {orphan}")
        for intent_id, group in self.groups.items():
            if (
                group.state is not OrdinaryHiCacheTransferState.ABORTED
                and tuple(group.transfer_ids) != group.expected_transfer_ids
            ):
                errors.append(f"ordinary HiCache {intent_id} TP group incomplete")
            states = [self.transfers[tid].state for tid in group.transfer_ids]
            if group.state is OrdinaryHiCacheTransferState.COMMITTED and any(
                state is not OrdinaryHiCacheTransferState.COMMITTED
                for state in states
            ):
                errors.append(f"ordinary HiCache {intent_id} false commit")
            if group.state is OrdinaryHiCacheTransferState.ABORTED and any(
                state
                in {
                    OrdinaryHiCacheTransferState.PENDING,
                    OrdinaryHiCacheTransferState.IN_FLIGHT,
                    OrdinaryHiCacheTransferState.ACKED,
                }
                for state in states
            ):
                errors.append(f"ordinary HiCache {intent_id} abort left active shard")
            if require_terminal and group.state in {
                OrdinaryHiCacheTransferState.PENDING,
                OrdinaryHiCacheTransferState.IN_FLIGHT,
            }:
                errors.append(f"ordinary HiCache {intent_id} remains pending")
            if group.state in {
                OrdinaryHiCacheTransferState.COMMITTED,
                OrdinaryHiCacheTransferState.ABORTED,
            } and not group.callback_delivered:
                errors.append(f"ordinary HiCache {intent_id} missed terminal ACK")
        return errors

    def export(self, *, require_terminal: bool = True) -> dict[str, Any]:
        """Export the canonical ordinary-transfer journal.

        Final artifacts are fail-closed by default: a group with any
        PENDING/IN_FLIGHT/ACKED shard is reported through ``audit_errors``.
        Callers taking an explicitly intermediate diagnostic snapshot may set
        ``require_terminal=False`` and must not publish it as a completed cell.
        """

        return copy.deepcopy(
            {
                "schema": OFFLINE_HICACHE_TRANSFER_BRIDGE_SCHEMA,
                "accounting": {
                    "ordinary_hbm_accounts": ["primary_kv", "prefix_kv"],
                    "ordinary_dram_account": "hicache_primary",
                    "aegis_accounts_used": False,
                    "capacity_owner": "Scheduler allocator snapshots",
                    "recoverability_owner": "ordinary HiCache intent ACK groups",
                    "logical_tp_expansion": "one Scheduler view expanded to every physical TP shard",
                },
                "groups": [
                    self.groups[key].as_dict() for key in sorted(self.groups)
                ],
                "transfers": [
                    self.transfers[key].as_dict() for key in sorted(self.transfers)
                ],
                "audit_rows": copy.deepcopy(self.audit_rows),
                "audit_errors": self.audit(require_terminal=require_terminal),
            }
        )
