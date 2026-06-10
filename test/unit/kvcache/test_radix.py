import pytest

from sglsim.srt.mem_cache.radix_cache import RadixCache


def test_radix_cache_matches_longest_prefix():
    cache = RadixCache()
    cache.insert([1, 2, 3], page_ids=[10, 11, 12])

    match = cache.match_prefix([1, 2, 4])

    assert match.matched_length == 2
    assert match.page_ids == [10, 11]


def test_radix_cache_no_match_returns_empty_prefix():
    cache = RadixCache()
    cache.insert([1, 2, 3], page_ids=[10, 11, 12])

    match = cache.match_prefix([4, 5, 6])

    assert match.matched_length == 0
    assert match.page_ids == []


def test_radix_cache_exact_match_returns_full_page_ids():
    cache = RadixCache()
    cache.insert([1, 2, 3], page_ids=[10, 11, 12])

    match = cache.match_prefix([1, 2, 3])

    assert match.matched_length == 3
    assert match.page_ids == [10, 11, 12]


def test_radix_cache_rejects_page_ids_shorter_than_token_ids():
    cache = RadixCache()

    with pytest.raises(ValueError):
        cache.insert([1, 2, 3], page_ids=[10, 11])
