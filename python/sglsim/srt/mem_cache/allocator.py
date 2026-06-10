from collections import deque
from collections.abc import Iterable

from sglsim.srt.mem_cache.page import KVPage


class KVCacheAllocator:
    def __init__(self, capacity_pages: int, page_size: int):
        if capacity_pages < 0:
            raise ValueError("capacity_pages must be non-negative")
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        pages = [
            KVPage(page_id=page_id, page_size=page_size)
            for page_id in range(capacity_pages)
        ]
        self._pages_by_id = {page.page_id: page for page in pages}
        self._owned_page_ids = set(self._pages_by_id)
        self._free_page_ids = set(self._owned_page_ids)
        self._free_pages = deque(pages)

    @property
    def num_free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def capacity_pages(self) -> int:
        return len(self._owned_page_ids)

    @property
    def used_pages(self) -> int:
        return self.capacity_pages - self.num_free_pages

    def alloc(self, num_pages: int, owner: str) -> list[KVPage]:
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        if num_pages > self.num_free_pages:
            raise RuntimeError("insufficient KV cache pages")

        pages = []
        for _ in range(num_pages):
            page = self._free_pages.popleft()
            self._free_page_ids.remove(page.page_id)
            page.owner = owner
            pages.append(page)
        return pages

    def free(self, pages: Iterable[KVPage]) -> None:
        pages = list(pages)
        page_ids = [page.page_id for page in pages]

        if len(set(page_ids)) != len(page_ids):
            raise ValueError("cannot free the same KV cache page more than once")
        for page in pages:
            page_id = page.page_id
            if page_id not in self._owned_page_ids:
                raise ValueError("KV cache page does not belong to this allocator")
            if self._pages_by_id[page_id] is not page:
                raise ValueError("KV cache page object does not belong to this allocator")
            if page_id in self._free_page_ids:
                raise ValueError("KV cache page is already free")

        for page in pages:
            page.owner = None
            self._free_page_ids.add(page.page_id)
            self._free_pages.append(page)
