from __future__ import annotations

from sglsim.srt.mem_cache.allocator import KVCacheAllocator
from sglsim.srt.mem_cache.page import KVPage


class DeviceKVPool:
    """GPU (L1) KV page pool, aligned with sglang.srt.mem_cache.memory_pool."""

    def __init__(self, capacity_pages: int, page_size: int) -> None:
        self.page_size = page_size
        self._allocator = KVCacheAllocator(capacity_pages, page_size)

    @property
    def capacity_pages(self) -> int:
        return self._allocator.capacity_pages

    @property
    def available_size(self) -> int:
        return self._allocator.num_free_pages

    @property
    def used_pages(self) -> int:
        return self._allocator.used_pages

    def alloc(self, num_pages: int, owner: str) -> list[KVPage] | None:
        if num_pages == 0:
            return []
        if num_pages > self._allocator.num_free_pages:
            return None
        return self._allocator.alloc(num_pages, owner=owner)

    def free(self, pages: list[KVPage]) -> None:
        if pages:
            self._allocator.free(pages)

    def clear(self) -> int:
        """Release every device page after a simulated GPU-memory loss.

        Returns the number of pages that were resident before the clear.  The
        host pool is a separate allocator and is deliberately unaffected.
        """

        pages = [
            self._allocator._pages_by_id[page_id]
            for page_id in range(self._allocator.capacity_pages)
            if page_id not in self._allocator._free_page_ids
        ]
        self.free(pages)
        return len(pages)

    def page(self, page_id: int) -> KVPage:
        return self._allocator._pages_by_id[page_id]
