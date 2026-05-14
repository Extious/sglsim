from __future__ import annotations


class PageAllocator:
    def __init__(self, num_pages: int):
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        self.num_pages = num_pages
        self._free_pages = list(range(num_pages))
        self._allocated: set[int] = set()

    @property
    def free_count(self) -> int:
        return len(self._free_pages)

    @property
    def allocated_count(self) -> int:
        return len(self._allocated)

    def allocate(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self._free_pages):
            raise RuntimeError(
                f"Insufficient free pages: requested {count}, available {len(self._free_pages)}"
            )
        pages = self._free_pages[:count]
        del self._free_pages[:count]
        self._allocated.update(pages)
        return pages

    def release(self, pages: list[int]) -> None:
        returned = []
        for page in pages:
            if page < 0 or page >= self.num_pages:
                raise ValueError(f"Invalid page id: {page}")
            if page in self._allocated:
                self._allocated.remove(page)
                returned.append(page)
        self._free_pages = sorted(set(self._free_pages).union(returned))

