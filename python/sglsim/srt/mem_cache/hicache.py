from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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

    @property
    def hierarchical_enabled(self) -> bool:
        return self.mem_pool_host is not None and self.mem_pool_host.capacity_pages > 0

    @property
    def last_comm_source(self) -> str:
        return self._last_comm_source

    def write(
        self,
        device_page_ids: list[int],
        *,
        node_id: int,
        owner: str,
        host_evict_fn: Callable[[int], None] | None = None,
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
        self.pending_backup_dur += dur
        self.pending_write_pages += num_pages
        self._d2h_transfer_s += dur
        self.write_queue.append(
            CacheOperation(
                host_page_ids=host_page_ids,
                device_page_ids=list(device_page_ids),
                node_ids=[node_id],
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
            )
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
