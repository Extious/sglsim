from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sglsim.srt.communication.types import CommPath, CommunicationRequest
from sglsim.srt.mem_cache.cache_operation import CacheOperation
from sglsim.srt.mem_cache.memory_pool import DeviceKVPool
from sglsim.srt.mem_cache.memory_pool_host import HostKVPool
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.state import StateManager


@dataclass
class CacheRecord:
    req_id: str
    page_id: int
    tier: str


OFFLINE_HICACHE_INTENT_SCHEMA = "sglsim.offline.hicache-intent.v1"


class HiCacheTransferDirection(str, Enum):
    D2H = "d2h"
    H2D = "h2d"


class HiCacheIntentState(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


@dataclass
class OfflineHiCacheIntent:
    """One logical-TP ordinary HiCache L1<->L2 transfer intent.

    SGLSim owns one symmetric Scheduler metadata pool per ``(dp, pp)`` stage.
    The parent offline bridge expands this logical record onto every physical
    TP GPU.  The worker never claims that this record is a physical transfer.
    """

    intent_id: str
    direction: HiCacheTransferDirection
    owner: str
    radix_node_id: int
    host_page_ids: tuple[int, ...]
    device_page_ids: tuple[int, ...]
    page_size_tokens: int
    kv_bytes_per_token_per_tp_rank: int
    dp_rank: int
    pp_rank: int
    scheduler_tp_rank: int
    worker_incarnation: int
    operation_sequence: int
    predicted_duration_s: float
    blocks_compute: bool
    state: HiCacheIntentState = HiCacheIntentState.PENDING

    @property
    def page_count(self) -> int:
        return len(self.host_page_ids)

    @property
    def token_count(self) -> int:
        return self.page_count * self.page_size_tokens

    @property
    def size_bytes_per_tp_rank(self) -> int:
        return self.token_count * self.kv_bytes_per_token_per_tp_rank

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
            "logical_tp_scope": "one-symmetric-tp-rank-view",
            "requires_parent_tp_expansion": True,
            "operation_sequence": self.operation_sequence,
            "predicted_duration_s": self.predicted_duration_s,
            "blocks_compute": bool(self.blocks_compute),
            "state": self.state.value,
        }


class HiCacheController:
    """Simulated HiCacheController aligned with sglang.srt.managers.cache_controller."""

    def __init__(
        self,
        mem_pool_device: DeviceKVPool,
        mem_pool_host: HostKVPool | None,
        *,
        write_policy: str = "write_through_selective",
        page_size: int = 16,
    ) -> None:
        if write_policy not in ("write_through", "write_through_selective", "write_back"):
            raise ValueError(f"Invalid write policy: {write_policy}")
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.write_queue: list[CacheOperation] = []
        self.load_queue: list[CacheOperation] = []
        self.l3_backup_queue: list[int] = []
        self._node_host_pages: dict[int, list[int]] = {}
        self.pending_backup_dur: float = 0.0
        self.pending_write_pages: int = 0
        self._d2h_transfer_s: float = 0.0
        self._h2d_transfer_s: float = 0.0
        self._last_comm_source: str = "none"
        self._load_consumer_index: int = 0
        self._offline_des_identity: tuple[int, int, int] | None = None
        self._offline_kv_bytes_per_token: int = 0
        self._offline_intent_sequence: int = 0
        self._offline_worker_incarnation: int = 0
        self._offline_intents: dict[str, OfflineHiCacheIntent] = {}
        self._offline_intent_outbox: list[str] = []

    @property
    def hierarchical_enabled(self) -> bool:
        return self.mem_pool_host is not None and self.mem_pool_host.capacity_pages > 0

    @property
    def last_comm_source(self) -> str:
        return self._last_comm_source

    @property
    def offline_des_enabled(self) -> bool:
        return self._offline_des_identity is not None

    def enable_offline_des(
        self,
        *,
        dp_rank: int,
        pp_rank: int,
        scheduler_tp_rank: int,
        kv_bytes_per_token_per_tp_rank: int,
        worker_incarnation: int = 0,
    ) -> None:
        """Enable intent/ACK mode for one offline worker only.

        The online wall-clock runtime never calls this method.  Keeping the
        switch instance-local prevents the global virtual clock from leaking
        into the online scheduler path.
        """

        identity = (int(dp_rank), int(pp_rank), int(scheduler_tp_rank))
        if any(value < 0 for value in identity):
            raise ValueError("offline HiCache rank identity must be nonnegative")
        kv_bytes = int(kv_bytes_per_token_per_tp_rank)
        if kv_bytes <= 0:
            raise ValueError("offline HiCache KV bytes/token must be positive")
        incarnation = int(worker_incarnation)
        if incarnation < 0:
            raise ValueError("offline HiCache worker incarnation must be nonnegative")
        if self._offline_des_identity not in (None, identity):
            raise RuntimeError("offline HiCache identity cannot change")
        self._offline_des_identity = identity
        self._offline_kv_bytes_per_token = kv_bytes
        self._offline_worker_incarnation = incarnation

    def _new_offline_intent(
        self,
        *,
        direction: HiCacheTransferDirection,
        owner: str,
        node_id: int,
        host_page_ids: list[int],
        device_page_ids: list[int],
        predicted_duration_s: float,
        blocks_compute: bool,
    ) -> OfflineHiCacheIntent | None:
        if not self.offline_des_enabled:
            return None
        assert self._offline_des_identity is not None
        sequence = self._offline_intent_sequence
        self._offline_intent_sequence += 1
        dp_rank, pp_rank, scheduler_tp_rank = self._offline_des_identity
        intent_id = (
            f"hicache-dp{dp_rank}-pp{pp_rank}-tp{scheduler_tp_rank}-"
            f"i{self._offline_worker_incarnation}-"
            f"{sequence:012d}"
        )
        intent = OfflineHiCacheIntent(
            intent_id=intent_id,
            direction=direction,
            owner=str(owner),
            radix_node_id=int(node_id),
            host_page_ids=tuple(int(page) for page in host_page_ids),
            device_page_ids=tuple(int(page) for page in device_page_ids),
            page_size_tokens=int(self.page_size),
            kv_bytes_per_token_per_tp_rank=self._offline_kv_bytes_per_token,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            scheduler_tp_rank=scheduler_tp_rank,
            worker_incarnation=self._offline_worker_incarnation,
            operation_sequence=sequence,
            predicted_duration_s=max(float(predicted_duration_s), 0.0),
            blocks_compute=bool(blocks_compute),
        )
        self._offline_intents[intent_id] = intent
        self._offline_intent_outbox.append(intent_id)
        return intent

    def drain_offline_intents(self) -> list[dict[str, Any]]:
        intent_ids = self._offline_intent_outbox
        self._offline_intent_outbox = []
        return [self._offline_intents[intent_id].as_dict() for intent_id in intent_ids]

    def offline_intent(self, intent_id: str) -> OfflineHiCacheIntent:
        try:
            return self._offline_intents[str(intent_id)]
        except KeyError as exc:
            raise KeyError(f"unknown offline HiCache intent {intent_id!r}") from exc

    def mark_offline_intent_terminal(
        self,
        intent_id: str,
        *,
        committed: bool,
    ) -> OfflineHiCacheIntent:
        intent = self.offline_intent(intent_id)
        target = (
            HiCacheIntentState.COMMITTED
            if committed
            else HiCacheIntentState.ABORTED
        )
        if intent.state is target:
            return intent
        if intent.state is not HiCacheIntentState.PENDING:
            raise RuntimeError(
                f"offline HiCache intent {intent.intent_id} is already "
                f"{intent.state.value}"
            )
        intent.state = target
        return intent

    def offline_intent_audit(self) -> list[str]:
        return [
            f"ordinary HiCache intent {intent.intent_id} remains PENDING"
            for intent in self._offline_intents.values()
            if intent.state is HiCacheIntentState.PENDING
        ]

    def write(
        self,
        device_page_ids: list[int],
        *,
        node_id: int,
        owner: str,
        host_evict_fn: Callable[[int], None] | None = None,
        offline_dependency: bool = False,
    ) -> tuple[list[int] | None, float]:
        if not self.hierarchical_enabled or not device_page_ids:
            return None, 0.0
        host_pool = self.mem_pool_host
        assert host_pool is not None
        num_pages = len(device_page_ids)
        host_pages = host_pool.alloc(num_pages, owner=owner)
        if host_pages is None and host_evict_fn is not None:
            host_evict_fn(num_pages)
            host_pages = host_pool.alloc(num_pages, owner=owner)
        if host_pages is None:
            return None, 0.0
        host_page_ids = [page.page_id for page in host_pages]
        dur = self._predict_transfer_duration(
            CommPath.D2H,
            host_page_ids=host_page_ids,
            device_page_ids=list(device_page_ids),
        )
        intent = self._new_offline_intent(
            direction=HiCacheTransferDirection.D2H,
            owner=owner,
            node_id=node_id,
            host_page_ids=host_page_ids,
            device_page_ids=list(device_page_ids),
            predicted_duration_s=dur,
            blocks_compute=bool(offline_dependency),
        )
        # Online Pipeline v2 continues to consume this duration exactly as it
        # did before.  Offline GlobalDES reserves the physical copy resources
        # explicitly, so adding the aggregate duration there would double count.
        if not self.offline_des_enabled:
            self.pending_backup_dur += dur
        self.pending_write_pages += num_pages
        if not self.offline_des_enabled:
            self._d2h_transfer_s += dur
        self.write_queue.append(
            CacheOperation(
                host_page_ids=host_page_ids,
                device_page_ids=list(device_page_ids),
                node_ids=[node_id],
                intent_id=(None if intent is None else intent.intent_id),
                owner=owner,
            )
        )
        self._node_host_pages[node_id] = list(host_page_ids)
        return host_page_ids, dur

    def take_pending_d2h(self, amount: float) -> float:
        taken = min(self.pending_backup_dur, amount)
        self.pending_backup_dur -= taken
        return taken

    def load(
        self,
        host_page_ids: list[int],
        *,
        node_id: int,
        owner: str,
    ) -> list[int] | None:
        if not host_page_ids:
            return None
        device_pages = self.mem_pool_device.alloc(len(host_page_ids), owner=owner)
        if device_pages is None:
            return None
        device_page_ids = [page.page_id for page in device_pages]
        self.load_queue.append(
            CacheOperation(
                host_page_ids=list(host_page_ids),
                device_page_ids=device_page_ids,
                node_ids=[node_id],
                owner=owner,
            )
        )
        if self.offline_des_enabled:
            duration = self._predict_transfer_duration(
                CommPath.H2D,
                host_page_ids=list(host_page_ids),
                device_page_ids=device_page_ids,
            )
            intent = self._new_offline_intent(
                direction=HiCacheTransferDirection.H2D,
                owner=owner,
                node_id=node_id,
                host_page_ids=list(host_page_ids),
                device_page_ids=device_page_ids,
                predicted_duration_s=duration,
                blocks_compute=True,
            )
            self.load_queue[-1].intent_id = (
                None if intent is None else intent.intent_id
            )
        return device_page_ids

    def consume_backup_overlap(self, forward_dur: float) -> float:
        overlapped = min(self.pending_backup_dur, forward_dur)
        self.pending_backup_dur -= overlapped
        return self.pending_backup_dur

    def consume_backup_serial(self) -> float:
        dur = self.pending_backup_dur
        self.pending_backup_dur = 0.0
        return dur

    def pop_write_pages_since_check(self) -> int:
        pages = self.pending_write_pages
        self.pending_write_pages = 0
        return pages

    def pop_d2h_transfer_s(self) -> float:
        dur = self._d2h_transfer_s
        self._d2h_transfer_s = 0.0
        return dur

    def pop_h2d_transfer_s(self) -> float:
        dur = self._h2d_transfer_s
        self._h2d_transfer_s = 0.0
        return dur

    def drain_load_queue(self) -> tuple[int, float]:
        if not self.load_queue:
            return 0, 0.0
        op = CacheOperation.merge_ops(self.load_queue)
        self.load_queue.clear()
        if self.offline_des_enabled:
            # Intent creation already predicted and published each logical
            # operation.  The parent DES owns timing and the H2D ACK gate.
            return len(op.host_page_ids), 0.0
        duration = self._predict_transfer_duration(
            CommPath.H2D,
            host_page_ids=op.host_page_ids,
            device_page_ids=op.device_page_ids,
        )
        self._h2d_transfer_s += duration
        StateManager.inc_hicache_l2_load_dur(duration)
        return len(op.host_page_ids), duration

    def start_loading(self) -> int:
        """Return a monotonically increasing loading consumer index."""
        self._load_consumer_index += 1
        return self._load_consumer_index

    def drain_l3_backup_queue(self) -> tuple[int, float]:
        if not self.l3_backup_queue:
            return 0, 0.0
        page_ids = list(self.l3_backup_queue)
        self.l3_backup_queue.clear()
        pred = ConfigManager.get_communication_predictor().predict(
            CommunicationRequest(
                path=CommPath.DISK_WRITE,
                page_ids=tuple(page_ids),
                page_size=self.page_size,
            )
        )
        duration = pred.duration_s
        self._last_comm_source = pred.source
        self.pending_backup_dur += duration
        return len(page_ids), duration

    def predict_disk_read_duration(self, page_ids: list[int]) -> float:
        if not page_ids:
            return 0.0
        pred = ConfigManager.get_communication_predictor().predict(
            CommunicationRequest(
                path=CommPath.DISK_READ,
                page_ids=tuple(page_ids),
                page_size=self.page_size,
            )
        )
        self._last_comm_source = pred.source
        return pred.duration_s

    def host_pages_for_node(self, node_id: int) -> list[int]:
        return list(self._node_host_pages.get(node_id, []))

    def _predict_transfer_duration(
        self,
        path: CommPath,
        *,
        host_page_ids: list[int],
        device_page_ids: list[int],
    ) -> float:
        pred = ConfigManager.get_communication_predictor().predict(
            CommunicationRequest(
                path=path,
                host_page_ids=tuple(host_page_ids),
                device_page_ids=tuple(device_page_ids),
                page_size=self.page_size,
            )
        )
        self._last_comm_source = pred.source
        return pred.duration_s
