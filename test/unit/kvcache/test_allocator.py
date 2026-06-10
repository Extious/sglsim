import pytest

from sglsim.srt.mem_cache.allocator import KVCacheAllocator
from sglsim.srt.mem_cache.page import KVPage


def test_kv_cache_allocator_allocates_and_reuses_pages():
    allocator = KVCacheAllocator(capacity_pages=2, page_size=16)

    pages = allocator.alloc(num_pages=2, owner="r0")
    allocator.free(pages[:1])
    reused = allocator.alloc(num_pages=1, owner="r1")

    assert len(pages) == 2
    assert reused[0].owner == "r1"
    assert allocator.num_free_pages == 0


def test_kv_cache_allocator_raises_when_insufficient_pages():
    allocator = KVCacheAllocator(capacity_pages=1, page_size=16)

    with pytest.raises(RuntimeError):
        allocator.alloc(num_pages=2, owner="r0")


def test_kv_cache_allocator_zero_page_alloc_is_noop():
    allocator = KVCacheAllocator(capacity_pages=2, page_size=16)

    pages = allocator.alloc(num_pages=0, owner="r0")

    assert pages == []
    assert allocator.num_free_pages == 2


def test_kv_cache_allocator_rejects_double_free_without_increasing_free_count():
    allocator = KVCacheAllocator(capacity_pages=1, page_size=16)
    pages = allocator.alloc(num_pages=1, owner="r0")

    allocator.free(pages)
    free_count = allocator.num_free_pages
    with pytest.raises(ValueError):
        allocator.free(pages)

    assert allocator.num_free_pages == free_count


def test_kv_cache_allocator_rejects_foreign_page_without_increasing_free_count():
    allocator = KVCacheAllocator(capacity_pages=1, page_size=16)
    foreign_page = KVPage(page_id=99, page_size=16, owner="r1")
    free_count = allocator.num_free_pages

    with pytest.raises(ValueError):
        allocator.free([foreign_page])

    assert allocator.num_free_pages == free_count


def test_kv_cache_allocator_rejects_forged_in_range_page_without_corruption():
    allocator = KVCacheAllocator(capacity_pages=1, page_size=16)
    pages = allocator.alloc(num_pages=1, owner="r0")
    forged = KVPage(page_id=0, page_size=16, owner="manual")

    with pytest.raises(ValueError):
        allocator.free([forged])

    assert allocator.num_free_pages == 0

    allocator.free(pages)
    reused = allocator.alloc(num_pages=1, owner="r1")

    assert reused == pages
    assert reused[0].owner == "r1"
    assert allocator.num_free_pages == 0


def test_kv_cache_allocator_reallocates_page_after_single_free():
    allocator = KVCacheAllocator(capacity_pages=1, page_size=16)
    pages = allocator.alloc(num_pages=1, owner="r0")

    allocator.free(pages)
    reused = allocator.alloc(num_pages=1, owner="r1")

    assert reused == pages
    assert reused[0].owner == "r1"
    assert allocator.num_free_pages == 0


@pytest.mark.parametrize(
    ("capacity_pages", "page_size"),
    [
        (-1, 16),
        (1, 0),
        (1, -1),
    ],
)
def test_kv_cache_allocator_rejects_invalid_configuration(
    capacity_pages, page_size
):
    with pytest.raises(ValueError):
        KVCacheAllocator(capacity_pages=capacity_pages, page_size=page_size)
