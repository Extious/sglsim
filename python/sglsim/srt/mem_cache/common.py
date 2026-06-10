from __future__ import annotations

from math import ceil


def extend_alloc_tokens(extend_tokens: int, page_size: int, batch_size: int = 1) -> int:
    """Budget tokens for extend alloc (SGLang over-estimate with page overhead)."""
    return extend_tokens + batch_size * page_size


def evict_from_tree_cache(
    cache: object,
    num_tokens: int,
) -> None:
    if num_tokens <= 0:
        return
    cache.evict_from_tree_cache(num_tokens)
