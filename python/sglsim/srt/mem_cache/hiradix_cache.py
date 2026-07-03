from __future__ import annotations

import logging
from math import ceil

from sglsim.srt.mem_cache.common import extend_alloc_tokens
from sglsim.srt.mem_cache.hicache import HiCacheController
from sglsim.srt.mem_cache.hicache_storage import (
    MockHiCacheStorage,
    PrefetchOperation,
    calc_prefetch_pages,
)
from sglsim.srt.mem_cache.memory_pool import DeviceKVPool
from sglsim.srt.mem_cache.memory_pool_host import HostKVPool
from sglsim.srt.mem_cache.page import KVPage
from sglsim.srt.mem_cache.radix_cache import PrefixMatch, RadixCache, TreeNode, _page_align
from sglsim.srt.communication.kv_bytes import kv_page_bytes_all_layers, resolve_kv_byte_info
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.state import StateManager

logger = logging.getLogger(__name__)


class HiRadixCache:
    """HiRadix cache aligned with sglang.srt.mem_cache.hiradix_cache."""

    def __init__(
        self,
        *,
        page_size: int,
        l1_capacity_pages: int,
        l2_capacity_pages: int,
        l3_capacity_pages: int = 0,
        write_policy: str = "write_through_selective",
        write_through_threshold: int = 2,
        load_back_threshold: int = 10,
        prefetch_threshold: int = 256,
        prefetch_stop_policy: str = "best_effort",
        prefetch_capacity_limit_ratio: float = 0.8,
        storage_backend: str | None = None,
    ) -> None:
        self.page_size = page_size
        self.write_through_threshold = (
            1 if write_policy == "write_through" else write_through_threshold
        )
        self.load_back_threshold = load_back_threshold
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_stop_policy = prefetch_stop_policy
        self.prefetch_capacity_limit_ratio = prefetch_capacity_limit_ratio
        self.tree = RadixCache(page_size=page_size)
        self.mem_pool_device = DeviceKVPool(l1_capacity_pages, page_size)
        self.mem_pool_host = (
            HostKVPool(l2_capacity_pages, page_size)
            if l2_capacity_pages > 0
            else None
        )
        self.cache_controller = HiCacheController(
            self.mem_pool_device,
            self.mem_pool_host,
            write_policy=write_policy,
            page_size=page_size,
        )
        self.enable_storage = (
            l3_capacity_pages > 0 and storage_backend is not None
        )
        self.storage: MockHiCacheStorage | None = (
            MockHiCacheStorage(capacity_pages=l3_capacity_pages, page_size=page_size)
            if self.enable_storage
            else None
        )
        self.ongoing_load_back: dict[int, TreeNode] = {}
        self.ongoing_write_through: dict[int, TreeNode] = {}
        self._write_through_lock_counts: dict[int, int] = {}
        self.ongoing_prefetch: dict[str, tuple[TreeNode, list[int], list[int], PrefetchOperation]] = {}
        self._chunked_prefetch: PrefetchOperation | None = None
        self._loads_drained = False

    @property
    def hierarchical_enabled(self) -> bool:
        return self.cache_controller.hierarchical_enabled

    def match_prefix(self, token_ids: list[int]) -> PrefixMatch:
        return self.tree.match_prefix(token_ids)

    def prepare_prefill(
        self,
        req_id: str,
        token_ids: list[int],
        cached_length: int,
        *,
        last_node_id: int | None,
    ) -> tuple[int, int | None]:
        match = self.tree.match_prefix(token_ids)
        if self.cache_controller.write_policy == "write_through_selective":
            for node in self.tree.collect_nodes_on_path(token_ids):
                if node.value is not None and not node.backuped:
                    self._inc_hit_count(node, req_id)
        # SGLang's req.prefix_indices contains only device-resident KV. Host
        # hits are tracked separately and become prefix KV only after load_back
        # completes. Do not lock or return a host-only tail here.
        device_cached = min(match.device_length, len(token_ids))
        # Preserve a pre-set cached_length (e.g. from retract recovery) as long
        # as at least one device-resident page remains — the gap can be filled
        # via load_back.  If the entire prefix is host-only then cached_length
        # must be 0 so the scheduler treats this as an uncached prefill.
        if device_cached > 0:
            new_cached = max(cached_length, device_cached)
        else:
            new_cached = 0
        if device_cached > 0:
            device_match = self.tree.match_prefix(token_ids[:device_cached])
            last_node = device_match.last_node or self.tree.root_node
        else:
            last_node = self.tree.root_node
        old_node = self.tree.get_node(last_node_id)
        if old_node is None:
            # Either a fresh request (last_node_id is None) or the previously locked
            # node was evicted under pressure (e.g. a retracted victim awaiting
            # re-admission). Either way, lock the freshly matched prefix node and
            # report it, so the caller tracks exactly the node we locked.
            self.tree.inc_lock_ref(last_node)
            return new_cached, last_node.id
        if last_node.id != old_node.id:
            self.tree.inc_lock_ref(last_node)
            if old_node.lock_ref > self._write_through_lock_counts.get(old_node.id, 0):
                try:
                    self.tree.dec_lock_ref(old_node)
                except RuntimeError:
                    logger.warning(
                        "Skip prepare_prefill lock release for %s: stale lock chain (node=%s lock_ref=%d)",
                        req_id,
                        old_node.id,
                        old_node.lock_ref,
                    )
            return new_cached, last_node.id
        if old_node.lock_ref == 0 and old_node is not self.tree.root_node:
            self.tree.inc_lock_ref(old_node)
        return new_cached, last_node_id

    def materialize_device_prefix(
        self,
        req_id: str,
        token_ids: list[int],
        restored_tokens: int,
        *,
        last_node_id: int | None,
    ) -> tuple[int, int | None]:
        """Materialize restored KV as a device-resident radix prefix.

        This is the metadata equivalent of copying protected KV pages into the
        retry rank before the replay request is admitted. It intentionally does
        not model tensor contents; it only allocates target-rank page metadata,
        inserts the restored prefix into radix, and locks the prefix for the
        request's lifecycle.
        """
        target = min(max(restored_tokens, 0), len(token_ids))
        target = _page_align(target, self.page_size)
        if target <= 0:
            return 0, last_node_id

        aligned_ids = token_ids[:target]
        match = self.tree.match_prefix(aligned_ids)
        device_covered = min(match.device_length, target)
        page_ids = list(match.page_ids[:device_covered])
        missing_tokens = target - device_covered
        allocated_pages: list[KVPage] = []

        if missing_tokens > 0:
            self.evict_from_tree_cache(missing_tokens, owner=req_id)
            allocated_pages = self._alloc_device_pages(
                ceil(missing_tokens / self.page_size),
                owner=req_id,
            )
            added = 0
            for page in allocated_pages:
                remaining = missing_tokens - added
                covered_page = min(self.page_size, remaining)
                page_ids.extend([page.page_id] * covered_page)
                added += covered_page

        result = self.tree.insert(aligned_ids, page_ids)
        self._free_replaced_page_ids(result.replaced_page_ids)

        old_node = self.tree.get_node(last_node_id)
        if old_node is not None and old_node.id != result.last_node.id:
            self._dec_request_lock_ref_tolerant(
                old_node,
                req_id=req_id,
                context="materialize_device_prefix",
            )
        if old_node is None or old_node.id != result.last_node.id:
            self.tree.inc_lock_ref(result.last_node)
        elif result.last_node.lock_ref == 0 and result.last_node is not self.tree.root_node:
            self.tree.inc_lock_ref(result.last_node)

        if allocated_pages:
            self._free_redundant_allocations(
                allocated_pages,
                cache_protected_len=device_covered,
                prefix_len=result.prefix_len,
                covered=device_covered,
                extend_end=target,
            )

        return target, result.last_node.id

    def _extend_prefill_bounds(
        self,
        extend_tokens: int,
        *,
        kv_committed_length: int,
        total_tokens: int,
    ) -> tuple[int, int] | None:
        """Return (logical_end, extend_end) or None when no page allocation is needed."""
        if extend_tokens <= 0:
            return None
        logical_end = min(kv_committed_length + extend_tokens, total_tokens)
        if logical_end <= kv_committed_length:
            return None
        extend_end = _page_align(logical_end, self.page_size)
        if extend_end <= kv_committed_length:
            return logical_end, kv_committed_length
        return logical_end, extend_end

    def extend_prefill_alloc_only(
        self,
        req_id: str,
        extend_tokens: int,
        *,
        token_ids: list[int],
        kv_committed_length: int,
        cache_protected_len: int,
        last_node_id: int | None,
        token_count: int,
        cached_length: int,
        prefill_alloc_page_ids: list[int],
    ) -> tuple[int, int, int | None, list[int]]:
        """Allocate KV pages for a prefill chunk without radix match/insert.

        Mirrors the decode ``alloc_for_decode`` fast path: chunked prefill only
        needs device pages until the request finishes prefilling, at which point
        ``finalize_prefill_in_tree`` performs a single radix insert.
        """
        bounds = self._extend_prefill_bounds(
            extend_tokens,
            kv_committed_length=kv_committed_length,
            total_tokens=token_count,
        )
        if bounds is None:
            return kv_committed_length, cache_protected_len, last_node_id, prefill_alloc_page_ids
        logical_end, extend_end = bounds
        if extend_end <= kv_committed_length:
            return logical_end, cache_protected_len, last_node_id, prefill_alloc_page_ids

        # SGLang alloc_extend uses req.prefix_indices as the allocation prefix.
        # In the simulator, a blocked/retracted chunk can return to the waiting
        # queue with the device prefix present in radix but without an active
        # request lock. Refresh the device-resident prefix here so alloc-only
        # prefill allocates only the missing suffix, not the whole committed
        # prefix again.
        committed_ids = token_ids[: min(kv_committed_length, len(token_ids))]
        current_match = self.tree.match_prefix(committed_ids) if committed_ids else None
        device_prefix_len = current_match.device_length if current_match else 0
        device_covered = max(
            cache_protected_len + len(prefill_alloc_page_ids),
            device_prefix_len,
        )
        new_tokens = extend_end - device_covered
        if new_tokens <= 0:
            return logical_end, cache_protected_len, last_node_id, prefill_alloc_page_ids

        num_tokens = extend_alloc_tokens(new_tokens, self.page_size)
        self.evict_from_tree_cache(num_tokens, owner=req_id)
        num_pages = ceil(new_tokens / self.page_size)
        allocated_pages = self._alloc_device_pages(num_pages, owner=req_id)
        updated_ids = list(prefill_alloc_page_ids)
        added = 0
        for page in allocated_pages:
            remaining = new_tokens - added
            covered_page = min(self.page_size, remaining)
            updated_ids.extend([page.page_id] * covered_page)
            added += covered_page
        return logical_end, cache_protected_len, last_node_id, updated_ids

    def cache_unfinished_req(
        self,
        req_id: str,
        token_ids: list[int],
        *,
        kv_committed_length: int,
        cache_protected_len: int,
        last_node_id: int | None,
        cached_length: int,
        prefill_alloc_page_ids: list[int],
        chunked: bool = False,
    ) -> tuple[int, int, int | None]:
        """Insert in-flight prefill KV into the radix tree (sglang cache_unfinished_req).

        Called after each chunked prefill forward so device pages join the radix
        tree incrementally instead of staying in deferred alloc-only state until
        prefill completes.
        """
        aligned_len = _page_align(kv_committed_length, self.page_size)
        if aligned_len <= 0:
            return kv_committed_length, cache_protected_len, last_node_id
        if not prefill_alloc_page_ids:
            # Overlap stash_chunked_request may run after process_batch_result
            # already flushed deferred pages into the radix tree.
            return kv_committed_length, cache_protected_len, last_node_id
        if aligned_len <= cache_protected_len:
            return kv_committed_length, cache_protected_len, last_node_id

        committed_len = kv_committed_length
        aligned_ids = token_ids[:aligned_len]
        # Align with sglang ``cache_unfinished_req``: the prefix comes from the
        # KV already materialized in radix / req_to_token. In overlap retries,
        # req.cache_protected_len can lag behind pages that were stashed by an
        # earlier chunk, so refresh it from the current device-resident match
        # before deciding how many deferred pages are missing.
        current_match = self.tree.match_prefix(aligned_ids)
        tree_prefix_len = max(cache_protected_len, current_match.device_length)
        prefix_len = min(tree_prefix_len, aligned_len)
        suffix_len = aligned_len - prefix_len
        if suffix_len <= 0:
            new_last_node = current_match.last_node
            if new_last_node is not None and new_last_node.id != last_node_id:
                old_node = self.tree.get_node(last_node_id)
                self._dec_request_lock_ref_tolerant(
                    old_node,
                    req_id=req_id,
                    context="cache_unfinished_req",
                )
                self.tree.inc_lock_ref(new_last_node)
            return committed_len, prefix_len, (
                new_last_node.id if new_last_node is not None else last_node_id
            )

        suffix_page_ids = prefill_alloc_page_ids[:suffix_len]
        if len(suffix_page_ids) < suffix_len:
            missing = suffix_len - len(suffix_page_ids)
            self.evict_from_tree_cache(missing, owner=req_id)
            try:
                pages = self._alloc_device_pages(
                    ceil(missing / self.page_size),
                    owner=req_id,
                )
            except RuntimeError:
                logger.warning(
                    "Skip cache_unfinished_req for %s: missing deferred pages "
                    "(need=%d have=%d available=%d evictable=%d)",
                    req_id,
                    suffix_len,
                    len(suffix_page_ids),
                    self.mem_pool_device.available_size * self.page_size,
                    self.tree.evictable_size(),
                )
                return prefix_len, prefix_len, last_node_id
            added = 0
            for page in pages:
                remaining = missing - added
                covered_page = min(self.page_size, remaining)
                suffix_page_ids.extend([page.page_id] * covered_page)
                added += covered_page

        old_node = self.tree.get_node(last_node_id)
        if prefix_len > 0:
            match = current_match
            if match.device_length < prefix_len:
                match = self.tree.match_prefix(aligned_ids[:prefix_len])
            if match.device_length < prefix_len:
                logger.warning(
                    "Skip cache_unfinished_req for %s: device prefix missing (need=%d have=%d)",
                    req_id,
                    prefix_len,
                    match.device_length,
                )
                free_ids = self.mem_pool_device._allocator._free_page_ids
                pages = [
                    self.mem_pool_device.page(page_id)
                    for page_id in sorted(set(suffix_page_ids))
                    if page_id not in free_ids
                ]
                self.mem_pool_device.free(pages)
                return min(committed_len, match.device_length), match.device_length, None
            prefix_page_ids = list(match.page_ids[:prefix_len])
        else:
            prefix_page_ids = []
        # SGLang inserts the full fill_ids key. A suffix-only fast path can
        # overwrite an existing child under a shared prefix when two requests
        # extend the same next token, breaking active request locks.
        result = self.tree.insert(aligned_ids, prefix_page_ids + suffix_page_ids)
        self._free_replaced_page_ids(result.replaced_page_ids)

        unique_page_ids: list[int] = []
        seen: set[int] = set()
        for page_id in suffix_page_ids:
            if page_id not in seen:
                unique_page_ids.append(page_id)
                seen.add(page_id)
        protected_len = self.tree.prefix_length_to_node(result.last_node)
        if unique_page_ids:
            allocated_pages = [self.mem_pool_device.page(pid) for pid in unique_page_ids]
            self._free_redundant_allocations(
                allocated_pages,
                cache_protected_len=cache_protected_len,
                prefix_len=result.prefix_len,
                covered=prefix_len,
                extend_end=aligned_len,
            )

        self._dec_request_lock_ref_tolerant(
            old_node,
            req_id=req_id,
            context="cache_unfinished_req",
        )
        self.tree.inc_lock_ref(result.last_node)

        self._record_insert_hits_for_key(aligned_ids, req_id, chunked=chunked)

        return committed_len, protected_len, result.last_node.id

    def finalize_prefill_in_tree(
        self,
        req_id: str,
        token_ids: list[int],
        *,
        kv_committed_length: int,
        cache_protected_len: int,
        last_node_id: int | None,
        cached_length: int,
        prefill_alloc_page_ids: list[int],
    ) -> tuple[int, int, int | None]:
        """Insert deferred prefill KV into the radix tree in one pass."""
        return self.cache_unfinished_req(
            req_id,
            token_ids,
            kv_committed_length=kv_committed_length,
            cache_protected_len=cache_protected_len,
            last_node_id=last_node_id,
            cached_length=cached_length,
            prefill_alloc_page_ids=prefill_alloc_page_ids,
        )

    def cache_finished_req(
        self,
        req_id: str,
        *,
        last_node_id: int | None,
    ) -> int:
        """Release radix lock on request completion."""
        node = self.tree.get_node(last_node_id)
        released_tokens = 0
        if (
            node is not None
            and node.lock_ref > self._write_through_lock_counts.get(node.id, 0)
        ):
            try:
                released_tokens = self.tree.dec_lock_ref(node)
            except RuntimeError:
                # Best-effort release on teardown path; do not crash worker on
                # stale ancestor lock counters after blocked/retract retries.
                logger.warning(
                    "Skip cache_finished_req lock release for %s: stale lock chain (node=%s lock_ref=%d)",
                    req_id,
                    node.id,
                    node.lock_ref,
                )
        self.ongoing_prefetch.pop(req_id, None)
        return released_tokens

    def cache_finished_req_with_existing_pages(
        self,
        req_id: str,
        token_ids: list[int],
        *,
        kv_committed_length: int,
        cache_protected_len: int,
        last_node_id: int | None,
        decode_pages: list[KVPage],
    ) -> int:
        """Insert finished decode KV using pages already owned by the request.

        Mirrors SGLang ``RadixCache.cache_finished_req``: generated KV is already
        allocated in the request token pool, so finishing must insert those indices
        into radix and release duplicates. It must not allocate another L1 copy.
        """
        token_count = min(kv_committed_length, len(token_ids))
        aligned_len = _page_align(token_count, self.page_size)
        retained_decode_page_ids: set[int] = set()

        old_node = self.tree.get_node(last_node_id)
        old_node_len = (
            self.tree.prefix_length_to_node(old_node)
            if old_node is not None
            else 0
        )
        prefix_len = min(max(cache_protected_len, old_node_len), aligned_len)

        if aligned_len > 0 and prefix_len <= aligned_len:
            prefix_page_ids: list[int] = []
            if prefix_len > 0:
                prefix_match = self.tree.match_prefix(token_ids[:prefix_len])
                if prefix_match.device_length >= prefix_len:
                    prefix_page_ids = list(prefix_match.page_ids[:prefix_len])
                else:
                    logger.warning(
                        "Skipping finished radix insert for %s: device prefix missing (need=%d have=%d)",
                        req_id,
                        prefix_len,
                        prefix_match.device_length,
                    )
                    self._free_decode_pages(decode_pages, retained_decode_page_ids)
                    return self.cache_finished_req(
                        req_id,
                        last_node_id=last_node_id,
                    )

            suffix_len = aligned_len - prefix_len
            decode_page_ids = self._expand_decode_page_ids(decode_pages)
            if len(decode_page_ids) < suffix_len:
                logger.warning(
                    "Truncate finished radix insert for %s: missing decode page ids (need=%d have=%d)",
                    req_id,
                    suffix_len,
                    len(decode_page_ids),
                )
                suffix_len = len(decode_page_ids)
                aligned_len = prefix_len + suffix_len

            if aligned_len > 0:
                suffix_page_ids = decode_page_ids[:suffix_len]
                result = self.tree.insert(
                    token_ids[:aligned_len],
                    prefix_page_ids + suffix_page_ids,
                )
                self._free_replaced_page_ids(result.replaced_page_ids)
                duplicate_suffix_len = min(
                    max(result.prefix_len - prefix_len, 0),
                    suffix_len,
                )
                retained_decode_page_ids = set(
                    suffix_page_ids[duplicate_suffix_len:suffix_len]
                )

                self._record_insert_hits_for_key(token_ids[:aligned_len], req_id)

        self._free_decode_pages(decode_pages, retained_decode_page_ids)
        return self.cache_finished_req(req_id, last_node_id=last_node_id)

    def _expand_decode_page_ids(self, decode_pages: list[KVPage]) -> list[int]:
        page_ids: list[int] = []
        for page in decode_pages:
            page_ids.extend([page.page_id] * self.page_size)
        return page_ids

    def _free_decode_pages(
        self,
        decode_pages: list[KVPage],
        retained_page_ids: set[int],
    ) -> None:
        if not decode_pages:
            return
        free_ids = self.mem_pool_device._allocator._free_page_ids
        to_free = [
            page
            for page in decode_pages
            if page.page_id not in retained_page_ids
            and page.page_id not in free_ids
        ]
        if to_free:
            self.mem_pool_device.free(to_free)

    def release_lock_ref(self, last_node_id: int | None) -> int:
        """Release one lock_ref without touching request lifecycle side effects."""
        node = self.tree.get_node(last_node_id)
        if node is None:
            return 0
        if node.lock_ref <= self._write_through_lock_counts.get(node.id, 0):
            return 0
        try:
            return self.tree.dec_lock_ref(node)
        except RuntimeError:
            logger.warning(
                "Skip release_lock_ref for node=%s: stale lock chain (lock_ref=%d)",
                node.id,
                node.lock_ref,
            )
            return 0

    def extend_prefill(
        self,
        req_id: str,
        token_ids: list[int],
        extend_tokens: int,
        *,
        kv_committed_length: int,
        cache_protected_len: int,
        last_node_id: int | None,
        token_count: int | None = None,
        defer_tree_insert: bool = False,
        cached_length: int = 0,
        prefill_alloc_page_ids: list[int] | None = None,
    ) -> tuple[int, int, int | None]:
        if extend_tokens <= 0:
            return kv_committed_length, cache_protected_len, last_node_id

        total_tokens = token_count if token_count is not None else len(token_ids)
        if defer_tree_insert:
            logical_end, protected, node_id, updated_ids = self.extend_prefill_alloc_only(
                req_id,
                extend_tokens,
                token_ids=token_ids,
                kv_committed_length=kv_committed_length,
                cache_protected_len=cache_protected_len,
                last_node_id=last_node_id,
                token_count=total_tokens,
                cached_length=cached_length,
                prefill_alloc_page_ids=list(prefill_alloc_page_ids or []),
            )
            if prefill_alloc_page_ids is not None:
                prefill_alloc_page_ids[:] = updated_ids
            return logical_end, protected, node_id

        bounds = self._extend_prefill_bounds(
            extend_tokens,
            kv_committed_length=kv_committed_length,
            total_tokens=total_tokens,
        )
        if bounds is None:
            return kv_committed_length, cache_protected_len, last_node_id
        logical_end, extend_end = bounds
        if extend_end <= kv_committed_length:
            return logical_end, cache_protected_len, last_node_id

        aligned_ids = token_ids[:extend_end]
        match = self.tree.match_prefix(aligned_ids)
        device_covered = min(match.device_length, extend_end)
        new_tokens = extend_end - device_covered
        allocated_pages: list[KVPage] = []

        if new_tokens > 0:
            num_tokens = extend_alloc_tokens(new_tokens, self.page_size)
            self.evict_from_tree_cache(num_tokens, owner=req_id)
            num_pages = ceil(new_tokens / self.page_size)
            allocated_pages = self._alloc_device_pages(num_pages, owner=req_id)
            new_page_ids: list[int] = []
            for page in allocated_pages:
                remaining = new_tokens - len(new_page_ids)
                covered_page = min(self.page_size, remaining)
                new_page_ids.extend([page.page_id] * covered_page)
            full_page_ids = list(match.page_ids[:device_covered]) + new_page_ids
        else:
            new_page_ids = []
            full_page_ids = list(match.page_ids[:extend_end])

        result = self.tree.insert(aligned_ids, full_page_ids)
        self._free_replaced_page_ids(result.replaced_page_ids)
        if allocated_pages:
            self._free_redundant_allocations(
                allocated_pages,
                cache_protected_len=cache_protected_len,
                prefix_len=result.prefix_len,
                covered=device_covered,
                extend_end=extend_end,
            )

        old_node = self.tree.get_node(last_node_id)
        if (
            old_node is not None
            and old_node.lock_ref > self._write_through_lock_counts.get(old_node.id, 0)
        ):
            try:
                self.tree.dec_lock_ref(old_node)
            except RuntimeError:
                logger.warning(
                    "Skip extend_prefill lock release for %s: stale lock chain (node=%s lock_ref=%d)",
                    req_id,
                    old_node.id,
                    old_node.lock_ref,
                )
        self.tree.inc_lock_ref(result.last_node)

        self._record_insert_hits_for_key(aligned_ids, req_id)

        return logical_end, result.prefix_len, result.last_node.id

    def alloc_for_decode(
        self,
        *,
        kv_committed_length: int,
        owner: str,
    ) -> KVPage | None:
        """O(1) decode KV allocation (SGLang ``alloc_for_decode``).

        Allocates device pages without radix ``match_prefix``/``insert``.
        When ``page_size > 1``, a new page is allocated only on page boundaries;
        otherwise the caller reuses the current partial page.
        """
        if self.page_size > 1 and kv_committed_length % self.page_size != 0:
            return None
        num_tokens = self.page_size if self.page_size > 1 else 1
        self.evict_from_tree_cache(num_tokens, owner=owner)
        return self._alloc_device_pages(1, owner=owner)[0]

    def _inc_hit_count(
        self,
        node: TreeNode,
        req_id: str,
        *,
        chunked: bool = False,
    ) -> None:
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1
        if not node.backuped and node.hit_count >= self.write_through_threshold:
            self.write_backup(node, req_id)

    def _inc_write_through_lock(self, node: TreeNode) -> None:
        """Lightweight write-through lock for simulator hot path.

        SGLang uses full-path inc_lock_ref/dec_lock_ref around async write ACK.
        In pure-Python radix with page_size=1 this becomes O(depth) per touched
        node and dominates wall-clock latency. For simulator fidelity we only
        need to prevent this node from being evicted before ACK, so we protect
        the node itself (O(1)) and keep evictable/protected accounting in sync.
        """
        if node.lock_ref == 0:
            self.tree.evictable_size_ -= len(node.key)
            self.tree.protected_size_ += len(node.key)
        node.lock_ref += 1
        self._write_through_lock_counts[node.id] = (
            self._write_through_lock_counts.get(node.id, 0) + 1
        )
        self.tree._update_leaf_status(node)
        self.tree._update_host_leaf_status(node)

    def _dec_write_through_lock(self, node: TreeNode) -> None:
        if node.lock_ref <= 0:
            return
        count = self._write_through_lock_counts.get(node.id, 0)
        if count <= 0:
            return
        if count == 1:
            self._write_through_lock_counts.pop(node.id, None)
        else:
            self._write_through_lock_counts[node.id] = count - 1
        if node.lock_ref == 1:
            self.tree.evictable_size_ += len(node.key)
            self.tree.protected_size_ -= len(node.key)
        node.lock_ref -= 1
        self.tree._update_leaf_status(node)
        self.tree._update_host_leaf_status(node)

    def write_backup(self, node: TreeNode, req_id: str, *, write_back: bool = False) -> float:
        if not self.hierarchical_enabled or node.value is None or node.backuped:
            return 0.0
        unique_device = list(dict.fromkeys(node.value))
        before = self.cache_controller.pending_backup_dur
        host_ids, _ = self.cache_controller.write(
            unique_device,
            node_id=node.id,
            owner=req_id,
            host_evict_fn=self.evict_host,
        )
        if host_ids is None:
            return 0.0
        self.tree.set_host_value(node, self._expand_host_ids(node, host_ids))
        self.ongoing_write_through[node.id] = node
        # SGLSim writes host metadata synchronously and keeps only transfer
        # duration pending for wall-clock accounting. Do not mix that pending
        # write-through state into request lock_ref; request locks are path-level
        # while a write-through ACK is node-local, and sharing the counter corrupts
        # protected/evictable accounting under full no-overlap pressure.
        added = self.cache_controller.pending_backup_dur - before
        if write_back:
            return self.cache_controller.take_pending_d2h(added)
        return added

    def writing_check(self, *, write_back: bool = False) -> int:
        if write_back:
            while self.cache_controller.write_queue:
                op = self.cache_controller.write_queue.pop(0)
                for node_id in op.node_ids:
                    node = self.ongoing_write_through.pop(node_id, None)
                    if node is not None and self.enable_storage:
                        self.write_backup_storage(node)
            while self.ongoing_write_through:
                node_id = next(iter(self.ongoing_write_through))
                self.ongoing_write_through.pop(node_id, None)
            return 0
        if not self.ongoing_write_through:
            return 0
        finish_count = 0
        if self.cache_controller.write_queue:
            while self.cache_controller.write_queue:
                op = self.cache_controller.write_queue.pop(0)
                for node_id in op.node_ids:
                    node = self.ongoing_write_through.pop(node_id, None)
                    if node is None:
                        continue
                    if self.enable_storage:
                        self.write_backup_storage(node)
                    finish_count += 1
            return finish_count

        while self.ongoing_write_through:
            node_id = next(iter(self.ongoing_write_through))
            node = self.ongoing_write_through.pop(node_id)
            if self.enable_storage:
                self.write_backup_storage(node)
            finish_count += 1
        return finish_count

    def flush_write_through_acks(self) -> int:
        """Eagerly flush completed write-through metadata acks."""
        return self.writing_check(write_back=False)

    def write_backup_storage(self, node: TreeNode) -> None:
        if not self.enable_storage or self.storage is None or node.host_value is None:
            return
        unique_host = list(dict.fromkeys(node.host_value))
        nodes = self.tree.collect_nodes_on_path(node.key)
        token_ids: list[int] = []
        for path_node in nodes:
            if path_node is node:
                token_ids.extend(node.key)
                break
            token_ids.extend(path_node.key)
        if not token_ids:
            token_ids = list(node.key)
        if self.storage.batch_set(token_ids, unique_host):
            node.storage_acked_len = len(node.key)

    def init_load_back(
        self,
        token_ids: list[int],
        *,
        mem_quota: int | None = None,
    ) -> tuple[int, TreeNode | None]:
        if not self.hierarchical_enabled:
            return 0, None
        match = self.tree.match_prefix(token_ids)
        if match.matched_length <= 0:
            return 0, None
        node = match.last_node or self.tree.root_node
        host_hit_length = 0
        last_host_node = node
        while node.evicted:
            if not node.backuped or node.host_value is None:
                return 0, None
            host_hit_length += len(node.host_value)
            node = node.parent or self.tree.root_node
        while not last_host_node.backuped and last_host_node is not self.tree.root_node:
            last_host_node = last_host_node.parent or self.tree.root_node
        if host_hit_length < self.load_back_threshold:
            return 0, None
        return host_hit_length, last_host_node

    def load_back(
        self,
        req_id: str,
        last_host_node: TreeNode,
        *,
        mem_quota: int | None = None,
    ) -> int:
        if not self.hierarchical_enabled:
            return 0
        nodes_to_load: list[TreeNode] = []
        node = last_host_node
        host_tokens = 0
        while node.evicted and node is not self.tree.root_node:
            if not node.backuped or node.host_value is None:
                return 0
            nodes_to_load.insert(0, node)
            host_tokens += len(node.host_value)
            node = node.parent or self.tree.root_node
        ancestor = node
        delta = self.tree.inc_lock_ref(ancestor)
        if host_tokens < self.load_back_threshold:
            self.tree.dec_lock_ref(ancestor)
            return 0
        if mem_quota is not None and host_tokens > mem_quota + delta:
            self.tree.dec_lock_ref(ancestor)
            return 0

        host_page_ids: list[int] = []
        for load_node in nodes_to_load:
            host_page_ids.extend(dict.fromkeys(load_node.host_value or []))
        device_ids = self.cache_controller.load(
            host_page_ids,
            node_id=last_host_node.id,
            owner=req_id,
        )
        self.tree.dec_lock_ref(ancestor)
        if device_ids is None:
            self.evict_from_tree_cache(host_tokens, owner=req_id)
            self.tree.inc_lock_ref(ancestor)
            device_ids = self.cache_controller.load(
                host_page_ids,
                node_id=last_host_node.id,
                owner=req_id,
            )
            self.tree.dec_lock_ref(ancestor)
        if device_ids is None:
            return 0

        offset = 0
        updates: list[tuple[TreeNode, list[int]]] = []
        for load_node in nodes_to_load:
            count = len(dict.fromkeys(load_node.host_value or []))
            chunk = device_ids[offset : offset + count]
            offset += count
            expanded = self._expand_device_ids(load_node, chunk)
            updates.append((load_node, expanded))

        # Restore device residency (sync mock; see SGLang load_back for async lock path).
        pool_pages = self.mem_pool_device._allocator._pages_by_id
        for load_node, expanded in updates:
            if load_node.value is not None:
                unique_ids = list(dict.fromkeys(load_node.value))
                freed = [
                    pool_pages[page_id]
                    for page_id in unique_ids
                    if page_id in pool_pages
                ]
                if freed:
                    self.mem_pool_device.free(freed)
            load_node.value = expanded
            self.tree._update_leaf_status(load_node)
        # Mirror sglang load_back: evictable_size_ += len(device_indices) after restore.
        # Sync mock: no inc/dec on last_host_node (restore is atomic; async path uses
        # ongoing_load_back + loading_check in SGLang).
        self.tree.evictable_size_ += len(device_ids)
        self.ongoing_load_back[last_host_node.id] = last_host_node
        self.tree.inc_lock_ref(last_host_node)
        return host_tokens

    def _assert_evictable_invariant(self, *, context: str = "") -> None:
        self.tree.assert_evictable_invariant(context=context)

    def loading_check(self) -> int:
        if not self._loads_drained or not self.ongoing_load_back:
            return 0
        completed_tokens = 0
        for last_node in list(self.ongoing_load_back.values()):
            if last_node.lock_ref > 0:
                self.tree.dec_lock_ref(last_node)
            completed_tokens += len(last_node.host_value or [])
        self.ongoing_load_back.clear()
        if self.tree.evictable_size_ < 0 or self.tree.protected_size_ < 0:
            self._assert_evictable_invariant(context="loading_check")
        self._loads_drained = False
        return completed_tokens

    def ready_to_load_host_cache(self) -> int:
        return self.cache_controller.start_loading()

    def prefetch(self, req_id: str, token_ids: list[int]) -> None:
        host_hit, last_host = self.init_load_back(token_ids)
        if host_hit <= 0 or last_host is None:
            return
        self.load_back(req_id, last_host)

    def prefetch_from_storage(self, req_id: str, token_ids: list[int]) -> str:
        if not self.enable_storage or self.storage is None or self.mem_pool_host is None:
            return "skipped"
        prefetch_length = _page_align(len(token_ids), self.page_size)
        if prefetch_length < self.prefetch_threshold:
            return "skipped"
        cap_limit = int(
            self.mem_pool_host.capacity_pages
            * self.page_size
            * self.prefetch_capacity_limit_ratio
        )
        used_tokens = (
            self.mem_pool_host.capacity_pages - self.mem_pool_host.available_size
        ) * self.page_size
        if used_tokens + prefetch_length > cap_limit:
            return "skipped"
        hit_tokens = self.storage.storage_hit_query(token_ids)
        if hit_tokens < self.prefetch_threshold:
            return "skipped"

        match = self.tree.match_prefix(token_ids)
        last_host_node = match.last_node or self.tree.root_node
        while not last_host_node.backuped and last_host_node is not self.tree.root_node:
            last_host_node = last_host_node.parent or self.tree.root_node
        last_host_node.protect_host()

        num_pages = hit_tokens // self.page_size
        host_pages = self.mem_pool_host.alloc(num_pages, owner=req_id)
        if host_pages is None:
            self.evict_host(num_pages)
            host_pages = self.mem_pool_host.alloc(num_pages, owner=req_id)
        if host_pages is None:
            last_host_node.release_host()
            return "skipped"

        host_page_ids = [page.page_id for page in host_pages]
        op = PrefetchOperation(
            req_id=req_id,
            token_ids=list(token_ids[:hit_tokens]),
            host_page_ids=host_page_ids,
            target_tokens=hit_tokens,
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            list(token_ids[:hit_tokens]),
            host_page_ids,
            op,
        )
        return "dispatched"

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True
        op = self.ongoing_prefetch[req_id][3]
        if self.prefetch_stop_policy == "wait_complete":
            return op.terminated and op.completed_tokens >= op.target_tokens
        return True

    def handle_prefetch_operation(self, remain_dur: float) -> int:
        if not self.enable_storage or self.storage is None:
            return 0
        platform = ConfigManager.get_platform_config()
        sched = ConfigManager.get_scheduler_config()
        raw = getattr(ConfigManager, "_raw", None) or {}
        model_path = str(raw.get("model_path", "mock-model"))
        kv_info = resolve_kv_byte_info(model_path=model_path, sched=sched)
        page_bytes = kv_page_bytes_all_layers(self.page_size, kv_info)
        bw = platform.disk_read_bandwidth_gb * 1e9
        completed_tokens = 0

        if self._chunked_prefetch is not None:
            op = self._chunked_prefetch
            remaining_pages = max(
                (op.target_tokens - op.completed_tokens) // self.page_size,
                0,
            )
            done_pages, prefetch_dur = calc_prefetch_pages(
                remaining_pages,
                page_bytes,
                remain_dur,
                bw,
            )
            if done_pages < remaining_pages:
                op.completed_tokens += done_pages * self.page_size
                dur = self.cache_controller.predict_disk_read_duration(
                    op.host_page_ids[:done_pages]
                )
                StateManager.inc_hicache_l2_load_dur(dur)
                return done_pages * self.page_size
            op.completed_tokens = op.target_tokens
            op.mark_terminate()
            dur = self.cache_controller.predict_disk_read_duration(op.host_page_ids)
            StateManager.inc_hicache_l2_load_dur(dur)
            self._finalize_prefetch(op.req_id)
            self._chunked_prefetch = None
            remain_dur -= prefetch_dur
            completed_tokens += op.completed_tokens

        for req_id, (last_node, token_ids, host_page_ids, op) in list(
            self.ongoing_prefetch.items()
        ):
            if op.terminated or remain_dur <= 0:
                continue
            required_pages = op.target_tokens // self.page_size
            done_pages, prefetch_dur = calc_prefetch_pages(
                required_pages,
                page_bytes,
                remain_dur,
                bw,
            )
            if done_pages < required_pages:
                op.completed_tokens = done_pages * self.page_size
                dur = self.cache_controller.predict_disk_read_duration(
                    host_page_ids[:done_pages]
                )
                StateManager.inc_hicache_l2_load_dur(dur)
                self._chunked_prefetch = op
                remain_dur = 0
                completed_tokens += op.completed_tokens
                continue
            op.completed_tokens = op.target_tokens
            op.mark_terminate()
            dur = self.cache_controller.predict_disk_read_duration(host_page_ids)
            StateManager.inc_hicache_l2_load_dur(dur)
            self._finalize_prefetch(req_id)
            remain_dur -= prefetch_dur
            completed_tokens += op.completed_tokens
        return completed_tokens

    def _finalize_prefetch(self, req_id: str) -> None:
        entry = self.ongoing_prefetch.pop(req_id, None)
        if entry is None:
            return
        last_node, token_ids, host_page_ids, op = entry
        aligned = token_ids[: op.completed_tokens]
        if not aligned:
            last_node.release_host()
            return
        page_ids_per_token: list[int] = []
        unique = list(dict.fromkeys(host_page_ids))
        idx = 0
        last = None
        for _ in range(len(aligned)):
            if idx < len(unique):
                page_ids_per_token.append(unique[idx])
                last = unique[idx]
            else:
                page_ids_per_token.append(last if last is not None else unique[-1])
            if (len(page_ids_per_token) % self.page_size) == 0:
                idx += 1
        self.tree.insert(aligned, page_ids_per_token)
        last_node.release_host()

    def prefill_pages_needed(self, prefill_length: int) -> int:
        return ceil(prefill_length / self.page_size)

    def evictable_size(self) -> int:
        return self.tree.evictable_size()

    def evictable_pages(self) -> int:
        return self.tree.evictable_pages()

    def available_and_evictable_tokens(self) -> int:
        return self.mem_pool_device.available_size * self.page_size + self.tree.evictable_size()

    def available_and_evictable_pages(self) -> int:
        return ceil(self.available_and_evictable_tokens() / self.page_size)

    def evict_from_tree_cache(self, num_tokens: int, *, owner: str = "evict") -> None:
        if num_tokens <= 0:
            return
        pool_pages = self.mem_pool_device._allocator._pages_by_id
        policy = self.cache_controller.write_policy
        reconciled = False
        for _ in range(512):
            available_tokens = self.mem_pool_device.available_size * self.page_size
            if available_tokens >= num_tokens:
                return
            if not reconciled and self.tree.evictable_size() <= 0:
                self.tree.reconcile_accounting()
                reconciled = True
                if self.tree.evictable_size() <= 0:
                    return
            before_evictable = self.tree.evictable_size()
            deficit = num_tokens - available_tokens
            freed = self._evict_device_pages(deficit, pool_pages, policy, owner)
            if freed:
                self.mem_pool_device.free(freed)
                continue
            if not freed or self.tree.evictable_size() >= before_evictable:
                if reconciled:
                    return
                self.tree.reconcile_accounting()
                reconciled = True

    def _evict_device_pages(
        self,
        deficit: int,
        pool_pages: dict[int, KVPage],
        policy: str,
        owner: str,
    ) -> list[KVPage]:
        if policy == "write_back":
            return self.tree.evict_write_back(
                deficit,
                pool_pages,
                backup_fn=lambda node: self.write_backup(node, owner, write_back=True),
            )
        write_through = policy in ("write_through", "write_through_selective")
        return self.tree.evict(deficit, pool_pages, write_through=write_through)

    def evict_host(self, num_pages: int) -> None:
        if self.mem_pool_host is None or num_pages <= 0:
            return
        if self.mem_pool_host.available_size >= num_pages:
            return
        deficit = num_pages - self.mem_pool_host.available_size
        pool_pages = self.mem_pool_host._allocator._pages_by_id
        freed = self.tree.evict_host(deficit, pool_pages)
        if freed:
            self.mem_pool_host.free(freed)

    def write_to_l3(self, req_id: str, token_ids: list[int]) -> None:
        if not self.enable_storage:
            return
        for node in self.tree.collect_nodes_on_path(token_ids):
            if node.host_value is None:
                continue
            if node.storage_acked_len >= len(node.key):
                continue
            for page_id in dict.fromkeys(node.host_value):
                self.cache_controller.l3_backup_queue.append(page_id)

    def finish_request(self, req_id: str, last_node_id: int | None) -> int:
        return self.cache_finished_req(req_id, last_node_id=last_node_id)

    def check_hicache_events(self, *, remain_dur: float = 0.0) -> dict[str, int]:
        if (
            not self.ongoing_write_through
            and not self.ongoing_prefetch
            and self._chunked_prefetch is None
            and not self.cache_controller.load_queue
            and not self.cache_controller.l3_backup_queue
            and self.cache_controller.pending_write_pages == 0
            and not self.ongoing_load_back
        ):
            return {
                "l2_write_pages": 0,
                "prefetch_pages": 0,
                "l3_write_pages": 0,
                "l3_prefetch_tokens": 0,
                "load_completed_tokens": 0,
            }
        self.writing_check()
        prefetch_tokens = self.handle_prefetch_operation(remain_dur)
        load_pages, _ = self.cache_controller.drain_load_queue()
        self._loads_drained = True
        load_completed = self.loading_check()
        l3_pages, _ = self.cache_controller.drain_l3_backup_queue()
        write_pages = self.cache_controller.pop_write_pages_since_check()
        return {
            "l2_write_pages": write_pages,
            "prefetch_pages": load_pages,
            "l3_write_pages": l3_pages,
            "l3_prefetch_tokens": prefetch_tokens,
            "load_completed_tokens": load_completed,
        }

    def kv_token_usage_details(
        self, *, logical_tokens: int | None = None
    ) -> tuple[int, int, int, int]:
        l1_total = self.mem_pool_device.capacity_pages * self.page_size
        available = self.mem_pool_device.available_size * self.page_size
        evictable = self.tree.evictable_size()
        num_used = l1_total - (available + evictable)
        num_used = max(0, min(num_used, l1_total))
        if logical_tokens is not None:
            # Match SGLang log semantics: #token / capacity from running-seq lengths.
            num_used = max(0, min(int(logical_tokens), l1_total))
        return num_used, l1_total, available, evictable

    def kv_token_usage(self) -> tuple[int, int]:
        num_used, l1_total, _, _ = self.kv_token_usage_details()
        return num_used, l1_total

    def _dec_request_lock_ref_tolerant(
        self,
        node: TreeNode | None,
        *,
        req_id: str,
        context: str,
    ) -> int:
        """Release one request lock along a possibly stale radix path.

        SGLang's request lock normally releases with a strict full-path
        ``dec_lock_ref``. In SGLSim, deferred chunked-prefill insertion can
        split or replace the leaf stored in ``req.last_node_id`` before the next
        stash. The leaf may then have no request lock while ancestors still do.
        Release only refs above write-through protection, preserving async
        backup locks and keeping evictable/protected accounting consistent.
        """
        if node is None:
            return 0

        released = 0
        skipped = False
        cur = node
        while cur is not self.tree.root_node:
            write_locks = self._write_through_lock_counts.get(cur.id, 0)
            if cur.lock_ref <= write_locks:
                skipped = True
                cur = cur.parent or self.tree.root_node
                continue
            if cur.lock_ref == 1 and cur.value is not None:
                self.tree.evictable_size_ += len(cur.key)
                self.tree.protected_size_ -= len(cur.key)
                released += len(cur.key)
            cur.lock_ref -= 1
            self.tree._update_leaf_status(cur)
            self.tree._update_host_leaf_status(cur)
            cur = cur.parent or self.tree.root_node

        if skipped:
            logger.debug(
                "%s released stale lock path for %s from node=%s released=%d",
                context,
                req_id,
                node.id,
                released,
            )
        return released

    def _alloc_device_pages(self, num_pages: int, owner: str) -> list[KVPage]:
        pages = self.mem_pool_device.alloc(num_pages, owner=owner)
        if pages is not None:
            return pages
        raise RuntimeError("insufficient L1 KV cache pages")

    def _free_redundant_allocations(
        self,
        allocated_pages: list[KVPage],
        *,
        cache_protected_len: int,
        prefix_len: int,
        covered: int,
        extend_end: int,
    ) -> None:
        overlap_start = max(covered, cache_protected_len)
        overlap_end = min(prefix_len, extend_end)
        redundant_tokens = overlap_end - overlap_start
        if redundant_tokens <= 0:
            return
        num_pages = min(
            ceil(redundant_tokens / self.page_size),
            len(allocated_pages),
        )
        if num_pages <= 0:
            return
        free_ids = self.mem_pool_device._allocator._free_page_ids
        to_free = [
            page
            for page in allocated_pages[:num_pages]
            if page.page_id not in free_ids
        ]
        if to_free:
            self.mem_pool_device.free(to_free)

    def _free_replaced_page_ids(self, page_ids: list[int]) -> None:
        if not page_ids:
            return
        free_ids = self.mem_pool_device._allocator._free_page_ids
        pages = [
            self.mem_pool_device.page(page_id)
            for page_id in dict.fromkeys(page_ids)
            if page_id not in free_ids
        ]
        if pages:
            self.mem_pool_device.free(pages)

    @staticmethod
    def _expand_host_ids(node: TreeNode, unique_host: list[int]) -> list[int]:
        if not node.value:
            return unique_host
        expanded: list[int] = []
        unique_idx = 0
        last_id = None
        for dev_id in node.value:
            if dev_id != last_id:
                if unique_idx < len(unique_host):
                    host_id = unique_host[unique_idx]
                    unique_idx += 1
                else:
                    host_id = unique_host[-1]
                last_id = dev_id
            expanded.append(host_id)
        return expanded

    def _record_insert_hit(
        self,
        node: TreeNode,
        req_id: str,
        *,
        chunked: bool = False,
    ) -> None:
        if not self.hierarchical_enabled or node.value is None:
            return
        if self.cache_controller.write_policy in (
            "write_through",
            "write_through_selective",
        ):
            self._inc_hit_count(node, req_id, chunked=chunked)

    def _record_insert_hits_for_key(
        self,
        token_ids: list[int],
        req_id: str,
        *,
        chunked: bool = False,
    ) -> None:
        if chunked:
            return
        for node in self.tree.collect_nodes_on_path(token_ids):
            self._record_insert_hit(node, req_id, chunked=False)

    @staticmethod
    def _expand_device_ids(node: TreeNode, unique_device: list[int]) -> list[int]:
        if not node.host_value:
            return unique_device
        expanded: list[int] = []
        unique_idx = 0
        last_id = None
        for host_id in node.host_value:
            if host_id != last_id:
                if unique_idx < len(unique_device):
                    dev_id = unique_device[unique_idx]
                    unique_idx += 1
                else:
                    dev_id = unique_device[-1]
                last_id = host_id
            expanded.append(dev_id)
        return expanded
