from __future__ import annotations

from dataclasses import dataclass, field


def _page_align(length: int, page_size: int) -> int:
    if page_size <= 1:
        return length
    return (length // page_size) * page_size


@dataclass
class PrefetchOperation:
    req_id: str
    token_ids: list[int]
    host_page_ids: list[int] = field(default_factory=list)
    completed_tokens: int = 0
    target_tokens: int = 0
    terminated: bool = False

    def mark_terminate(self) -> None:
        self.terminated = True


class MockHiCacheStorage:
    """In-memory L3 index aligned with sglang storage backend semantics."""

    def __init__(self, *, capacity_pages: int, page_size: int) -> None:
        self.capacity_pages = max(capacity_pages, 0)
        self.page_size = page_size
        self._entries: dict[tuple[int, ...], list[int]] = {}
        self._used_pages = 0

    @property
    def used_pages(self) -> int:
        return self._used_pages

    def storage_hit_query(self, token_ids: list[int]) -> int:
        aligned = _page_align(len(token_ids), self.page_size)
        if aligned <= 0:
            return 0
        key = tuple(token_ids[:aligned])
        if key not in self._entries:
            return 0
        return aligned

    def batch_set(self, token_ids: list[int], page_ids: list[int]) -> bool:
        aligned = _page_align(len(token_ids), self.page_size)
        if aligned <= 0 or not page_ids:
            return False
        num_pages = len(dict.fromkeys(page_ids))
        key = tuple(token_ids[:aligned])
        if key in self._entries:
            old_pages = len(dict.fromkeys(self._entries[key]))
            self._used_pages -= old_pages
        if self._used_pages + num_pages > self.capacity_pages:
            return False
        self._entries[key] = list(page_ids)
        self._used_pages += num_pages
        return True

    def get_stats(self) -> dict[str, int]:
        return {
            "capacity_pages": self.capacity_pages,
            "used_pages": self._used_pages,
            "entries": len(self._entries),
        }


def calc_prefetch_pages(
    required_pages: int,
    page_size_bytes: int,
    max_dur: float,
    bandwidth_bytes_per_s: float,
) -> tuple[int, float]:
    if required_pages <= 0 or page_size_bytes <= 0 or bandwidth_bytes_per_s <= 0:
        return 0, 0.0
    full_dur = required_pages * page_size_bytes / bandwidth_bytes_per_s
    if full_dur <= max_dur:
        return required_pages, full_dur
    completed = max(int(max_dur * bandwidth_bytes_per_s / page_size_bytes), 1)
    completed = min(completed, required_pages)
    return completed, max_dur
