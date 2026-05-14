from pathlib import Path

from sglsim.srt.mem_cache.allocator import PageAllocator
from sglsim.srt.mem_cache.hicache import (
    HiCacheStorage,
    estimate_transfer_seconds,
    record_hicache_l2_backup,
    record_hicache_l2_load,
)
from sglsim.srt.mem_cache.radix_cache import PrefixCache
from sglsim.srt.simulator.state import SimulatorState, set_active_state


def test_page_allocator_allocates_and_releases_pages():
    allocator = PageAllocator(num_pages=4)

    pages = allocator.allocate(2)
    assert pages == [0, 1]
    assert allocator.free_count == 2

    allocator.release([0])
    assert allocator.allocate(2) == [0, 2]


def test_page_allocator_rejects_over_allocation():
    allocator = PageAllocator(num_pages=1)
    allocator.allocate(1)

    try:
        allocator.allocate(1)
    except RuntimeError as exc:
        assert "Insufficient free pages" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_prefix_cache_returns_longest_prefix_match():
    cache = PrefixCache()
    cache.insert([1, 2, 3], [10, 11, 12])
    cache.insert([1, 2, 4], [10, 11, 13])

    tokens, pages = cache.match([1, 2, 3, 9])

    assert tokens == [1, 2, 3]
    assert pages == [10, 11, 12]

    tokens, pages = cache.match([1, 2, 8])
    assert tokens == [1, 2]
    assert pages == [10, 11]


def test_hicache_key_persistence(tmp_path: Path):
    path = tmp_path / "keys.txt"
    storage = HiCacheStorage(storage_file_path=str(path))

    assert not storage.exists("a")
    assert storage.set("a")
    assert storage.exists("a")
    assert storage.batch_exists(["a", "b"], extra_info=None) == 1

    restored = HiCacheStorage(storage_file_path=str(path))
    assert restored.exists("a")
    assert restored.clear()
    assert not restored.exists("a")


def test_hicache_transfer_estimator_and_state_recording():
    assert estimate_transfer_seconds(num_tokens=4, bytes_per_token=2, bandwidth_bytes=4) == 2
    assert estimate_transfer_seconds(num_tokens=4, bytes_per_token=2, bandwidth_bytes=0) == 0

    state = SimulatorState()
    set_active_state(state)
    try:
        record_hicache_l2_load(num_tokens=4, bytes_per_token=2, bandwidth_bytes=4)
        record_hicache_l2_backup(num_tokens=6, bytes_per_token=2, bandwidth_bytes=4)
    finally:
        set_active_state(None)

    assert state.pop_hicache_l2_load_dur() == 2
    assert state.pop_hicache_l2_backup_dur() == 3

