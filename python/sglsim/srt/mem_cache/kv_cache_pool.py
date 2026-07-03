from __future__ import annotations

import logging

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.chunked_prefill import split_prefill_length
from sglsim.srt.managers.schedule_batch import Req
from sglsim.srt.mem_cache.hiradix_cache import HiRadixCache
from sglsim.srt.mem_cache.transfer import TransferModel

DEFAULT_TRANSFER_MODEL = TransferModel(
    fixed_latency_s=0.0,
    bandwidth_bytes_per_s=1.0,
)

logger = logging.getLogger(__name__)


class KVCachePool:
    """KV cache orchestration facade aligned with sglang HiRadixCache + HiCacheController."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._cache = HiRadixCache(
            page_size=self.config.page_size,
            l1_capacity_pages=self.config.l1_capacity_pages,
            l2_capacity_pages=self.config.l2_capacity_pages,
            l3_capacity_pages=self.config.l3_capacity_pages,
            write_policy=self.config.hicache_write_policy,
            write_through_threshold=self.config.write_through_threshold,
            load_back_threshold=self.config.load_back_threshold,
            prefetch_threshold=self.config.prefetch_threshold,
            prefetch_stop_policy=self.config.prefetch_stop_policy,
            prefetch_capacity_limit_ratio=self.config.prefetch_capacity_limit_ratio,
            storage_backend=self.config.hicache_storage_backend,
        )
        self.cache_prefix_tokens = 0
        self.metrics = {
            "cache_prefix_tokens": 0,
            "prefetch_pages": 0,
            "l2_write_pages": 0,
            "l3_write_pages": 0,
            "l3_prefetch_tokens": 0,
            "load_completed_tokens": 0,
            "d2h_transfer_s": 0.0,
            "h2d_transfer_s": 0.0,
            "comm_source": "none",
        }
        self.transfer_model = DEFAULT_TRANSFER_MODEL
        self._last_check_result: dict[str, int] = {}

    @property
    def radix(self):
        return self._cache.tree

    @property
    def hicache(self):
        return self._cache.cache_controller

    @property
    def allocator(self):
        return self._cache.mem_pool_device._allocator

    @property
    def hierarchical_enabled(self) -> bool:
        return self._cache.hierarchical_enabled

    def kv_token_usage(self) -> tuple[int, int]:
        return self._cache.kv_token_usage()

    def kv_token_usage_details(
        self, *, logical_tokens: int | None = None
    ) -> tuple[int, int, int, int]:
        return self._cache.kv_token_usage_details(logical_tokens=logical_tokens)

    @staticmethod
    def logical_kv_tokens(reqs) -> int:
        return sum(req.total_length for req in reqs)

    def evictable_size(self) -> int:
        return self._cache.evictable_size()

    def available_and_evictable_tokens(self) -> int:
        return self._cache.available_and_evictable_tokens()

    def available_device_tokens(self) -> int:
        return (
            self._cache.mem_pool_device.available_size * self._cache.page_size
        )

    def available_and_evictable_pages(self) -> int:
        return self._cache.available_and_evictable_pages()

    @staticmethod
    def _prefill_match_ids(req: Req) -> list[int]:
        # SGLang leaves the last token uncached for the extend path so logits
        # are computed even when the whole fill sequence exists in radix.
        max_prefix_len = max(req.prefill_length - 1, 0)
        return req.fill_ids[:max_prefix_len]

    def evict_from_tree_cache(self, num_tokens: int, *, owner: str = "evict") -> None:
        self._cache.evict_from_tree_cache(num_tokens, owner=owner)

    def matched_prefix_length(self, req: Req) -> int:
        """Read-only device-resident radix match length for the request input.

        Mirrors SGLang ``Req.init_next_round_input``: ``prefix_indices`` only
        contains device-resident KV. Host hits are tracked separately via
        ``host_hit_length`` and may become device prefix only after load_back.
        """
        return self._cache.match_prefix(self._prefill_match_ids(req)).device_length

    def refresh_matched_prefix(self, req: Req) -> int:
        """Refresh device and host prefix metadata like SGLang Req.init_next_round_input."""
        token_ids = self._prefill_match_ids(req)
        match = self._cache.match_prefix(token_ids)
        max_cached = max(req.prefill_length - 1, 0)
        device_len = min(match.device_length, max_cached)
        host_len = min(match.host_length, max(max_cached - device_len, 0))
        req.cached_length = device_len
        req.host_hit_length = host_len

        last_host_node_id = None
        if host_len > 0:
            node = match.last_node or self._cache.tree.root_node
            while (
                node is not self._cache.tree.root_node
                and not node.backuped
            ):
                node = node.parent or self._cache.tree.root_node
            if node.backuped:
                last_host_node_id = node.id
        req.last_host_node_id = last_host_node_id
        return device_len

    def device_resident_length(self, req: Req) -> int:
        """Read-only device-resident match length for the request input.

        Used after ``load_back`` to find how much of the prefix is actually on the
        device (robust to a partial load under memory pressure)."""
        return self._cache.match_prefix(self._prefill_match_ids(req)).device_length

    def init_load_back(self, req: Req, *, mem_quota: int | None = None) -> int:
        # Use full fill_ids (not _prefill_match_ids which excludes the last token)
        # because host KV for the last prefill token can still be loaded back.
        # Mirrors SGLang: init_load_back operates on req.fill_ids.
        host_hit, last_host = self._cache.init_load_back(
            req.fill_ids,
            mem_quota=mem_quota,
        )
        req.host_hit_length = host_hit
        req.last_host_node_id = last_host.id if last_host is not None else None
        return host_hit

    def load_back(self, req: Req, *, mem_quota: int | None = None) -> int:
        node = self._cache.tree.get_node(req.last_host_node_id)
        if node is None:
            return 0
        return self._cache.load_back(req.rid, node, mem_quota=mem_quota)

    def ready_to_load_host_cache(self) -> int:
        return self._cache.ready_to_load_host_cache()

    def prefetch_from_storage(self, req: Req) -> str:
        status = self._cache.prefetch_from_storage(req.rid, req.input_ids)
        req.prefetch_status = status
        return status

    def check_prefetch_progress(self, req: Req) -> bool:
        return self._cache.check_prefetch_progress(req.rid)

    def prepare_prefill(self, req: Req) -> None:
        token_ids = self._prefill_match_ids(req)
        match = self._cache.match_prefix(token_ids)
        max_cached = max(req.prefill_length - 1, 0)
        new_cached, last_node_id = self._cache.prepare_prefill(
            req.rid,
            token_ids,
            min(req.cached_length, max_cached),
            last_node_id=req.last_node_id,
        )
        prior_cached = req.cached_length
        new_cached = min(new_cached, max_cached)
        req.cached_length = new_cached
        req.cache_protected_len = new_cached
        if not req.cache_hit_recorded:
            # Record cache hit discovered at admission time.  When
            # _refresh_req_next_round_input already recorded a hit the
            # delta guards against double-counting (only additional tokens
            # beyond the refresh-provided match are counted).
            newly_matched = max(
                0,
                min(max(new_cached, match.device_length), max_cached) - prior_cached,
            )
            req.cache_hit_tokens = newly_matched
            req.cache_hit_log_pending = req.cache_hit_tokens
            req.cache_hit_recorded = True
            self.cache_prefix_tokens += req.cache_hit_tokens
            self.metrics["cache_prefix_tokens"] = self.cache_prefix_tokens
        else:
            req.cache_hit_log_pending = 0
        # _cache.prepare_prefill always locks (inc_lock_ref) the node corresponding
        # to the returned last_node_id, moving the lock off any prior node. Track it
        # unconditionally so a later cache_finished_req dec_lock_ref's the SAME node.
        # (Guarding on `is None` left a retracted+re-admitted req pointing at a stale
        # node, dec'ing an unlocked node -> "Lock reference counter is already zero".)
        req.last_node_id = last_node_id

    def materialize_restored_prefix(self, req: Req, restored_tokens: int) -> int:
        max_cached = max(req.prefill_length - 1, 0)
        target = min(max(restored_tokens, 0), max_cached)
        if target <= 0:
            req.aegis_restored_prefill_tokens = 0
            return 0

        restored, last_node_id = self._cache.materialize_device_prefix(
            req.rid,
            self._prefill_match_ids(req),
            target,
            last_node_id=req.last_node_id,
        )
        restored = min(restored, max_cached)
        if restored <= 0:
            req.aegis_restored_prefill_tokens = 0
            return 0

        req.cached_length = max(req.cached_length, restored)
        req.kv_committed_length = max(req.kv_committed_length, restored)
        req.cache_protected_len = max(req.cache_protected_len, restored)
        req.last_node_id = last_node_id
        req.aegis_restored_prefill_tokens = restored

        previous_hit = req.cache_hit_tokens if req.cache_hit_recorded else 0
        if restored > previous_hit:
            delta = restored - previous_hit
            self.cache_prefix_tokens += delta
            self.metrics["cache_prefix_tokens"] = self.cache_prefix_tokens
        req.cache_hit_tokens = max(req.cache_hit_tokens, restored)
        req.cache_hit_log_pending = req.cache_hit_tokens
        req.cache_hit_recorded = True
        req.prefill_plan_synced = True
        return restored

    def sync_retract_prefix(self, req: Req, *, chunk_size: int | None) -> None:
        """Reconcile req cache pointers after retract; prefix stays in radix tree."""
        match = self._cache.match_prefix(self._prefill_match_ids(req))
        req.cached_length = match.device_length
        req.kv_committed_length = match.device_length
        req.cache_protected_len = 0
        # SGLang reset_for_retract clears req.last_node. This sync is read-only;
        # the next admission reacquires the radix lock via prepare_prefill.
        req.last_node_id = None
        uncached = max(req.prefill_length - match.device_length, 0)
        if uncached <= 0:
            req.prefill_chunks_remaining = []
        else:
            req.prefill_chunks_remaining = list(
                split_prefill_length(uncached, chunk_size)
            )

    def extend_prefill(self, req: Req, extend_tokens: int) -> None:
        defer_tree = True
        kv_len, protected, last_node_id = self._cache.extend_prefill(
            req.rid,
            req.fill_ids,
            extend_tokens,
            kv_committed_length=req.kv_committed_length,
            cache_protected_len=req.cache_protected_len,
            last_node_id=req.last_node_id,
            token_count=req.prefill_length,
            defer_tree_insert=defer_tree,
            cached_length=req.cached_length,
            prefill_alloc_page_ids=req.prefill_alloc_page_ids,
        )
        req.kv_committed_length = kv_len
        req.cache_protected_len = protected
        req.last_node_id = last_node_id

    def _release_deferred_prefill_pages(self, req: Req) -> None:
        if not req.prefill_alloc_page_ids:
            return
        unique_ids = sorted(set(req.prefill_alloc_page_ids))
        pages = [self._cache.mem_pool_device.page(page_id) for page_id in unique_ids]
        self._cache.mem_pool_device.free(pages)
        req.prefill_alloc_page_ids.clear()

    def release_deferred_prefill_pages(self, req: Req) -> None:
        self._release_deferred_prefill_pages(req)

    def release_prefill_lock(self, req: Req) -> int:
        released = self._cache.release_lock_ref(req.last_node_id)
        req.cache_protected_len = 0
        req.last_node_id = None
        return released

    def cache_unfinished_req(self, req: Req, *, chunked: bool = False) -> None:
        if not req.prefill_alloc_page_ids and req.kv_committed_length <= req.cached_length:
            return
        kv_len, protected, last_node_id = self._cache.cache_unfinished_req(
            req.rid,
            req.fill_ids,
            kv_committed_length=req.kv_committed_length,
            cache_protected_len=req.cache_protected_len,
            last_node_id=req.last_node_id,
            cached_length=req.cached_length,
            prefill_alloc_page_ids=req.prefill_alloc_page_ids,
            chunked=chunked,
        )
        req.kv_committed_length = kv_len
        req.cache_protected_len = protected
        req.last_node_id = last_node_id
        if last_node_id is None and protected < req.cached_length:
            req.cached_length = protected
        else:
            req.cached_length = max(req.cached_length, protected)
        req.prefill_alloc_page_ids.clear()

    def finalize_prefill_in_tree(self, req: Req) -> None:
        self.cache_unfinished_req(req)

    def extend_decode(self, req: Req, extend_tokens: int = 1) -> None:
        """Allocate KV for the next decode token (before simulated forward).

        Mirrors SGLang ``prepare_for_decode`` → ``alloc_for_decode``.
        """
        if extend_tokens <= 0:
            return
        page = self._cache.alloc_for_decode(
            kv_committed_length=req.kv_committed_length,
            owner=req.rid,
        )
        if page is not None:
            req.decode_pages.append(page)
        req.kv_committed_length += extend_tokens

    def finish_request(self, req: Req) -> int:
        if req.kv_released:
            return 0
        req.kv_released = True
        if req.prefill_alloc_page_ids:
            self.cache_unfinished_req(req)
        self._release_deferred_prefill_pages(req)
        if req.output_ids:
            released = self._cache.cache_finished_req_with_existing_pages(
                req.rid,
                req.fill_ids,
                kv_committed_length=req.kv_committed_length,
                cache_protected_len=req.cache_protected_len,
                last_node_id=req.last_node_id,
                decode_pages=req.decode_pages,
            )
            req.decode_pages.clear()
            return released
        if req.decode_pages:
            self._cache.mem_pool_device.free(req.decode_pages)
            req.decode_pages.clear()
        return self._cache.cache_finished_req(req.rid, last_node_id=req.last_node_id)

    def check_hicache_events(self, *, remain_dur: float = 0.0) -> dict[str, int]:
        result = self._cache.check_hicache_events(remain_dur=remain_dur)
        self.metrics["l2_write_pages"] += result["l2_write_pages"]
        self.metrics["prefetch_pages"] += result["prefetch_pages"]
        self.metrics["l3_write_pages"] += result["l3_write_pages"]
        self.metrics["l3_prefetch_tokens"] += result.get("l3_prefetch_tokens", 0)
        self.metrics["load_completed_tokens"] += result.get("load_completed_tokens", 0)
        self.metrics["d2h_transfer_s"] += self.hicache.pop_d2h_transfer_s()
        self.metrics["h2d_transfer_s"] += self.hicache.pop_h2d_transfer_s()
        self.metrics["comm_source"] = self.hicache.last_comm_source
        return result

    def consume_backup_overlap(self, forward_dur: float) -> float:
        return self.hicache.consume_backup_overlap(forward_dur)

    def consume_backup_serial(self) -> float:
        return self.hicache.consume_backup_serial()

    def flush_write_through_acks(self) -> int:
        return self._cache.flush_write_through_acks()

    def write_to_l3(self, req: Req) -> None:
        if self.config.l3_capacity_pages == 0:
            return
        self._cache.write_to_l3(req.rid, req.input_ids)

    def prefetch(self, req: Req) -> None:
        host_hit = self.init_load_back(req)
        if host_hit > 0:
            self.load_back(req)

    def release_request(self, req: Req) -> int:
        """Release request KV without inserting generated tokens into radix.

        SGLang ``ScheduleBatch.retract_decode`` calls
        ``release_kv_cache(..., is_insert=False)`` so the scheduler can reclaim
        decode KV immediately. Only true completion uses ``finish_request`` and
        inserts the finished sequence into the prefix tree.
        """
        if req.kv_released:
            return 0
        req.kv_released = True
        self._release_deferred_prefill_pages(req)
        if req.decode_pages:
            self._cache.mem_pool_device.free(req.decode_pages)
            req.decode_pages.clear()
        return self._cache.cache_finished_req(req.rid, last_node_id=req.last_node_id)
