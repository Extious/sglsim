from __future__ import annotations

from typing import Any, Optional


def _int_attr(obj: Any, name: str, default: int = 0) -> int:
    return int(getattr(obj, name, default) or 0)


def apply_cache_source_breakdown(req: Any) -> None:
    """Populate per-source cache counters on a request."""
    prefix_indices = getattr(req, "prefix_indices", None)
    prefix_len = len(prefix_indices) if prefix_indices is not None else 0
    host_total = _int_attr(req, "host_hit_length")
    storage_completed = _int_attr(req, "storage_hit_length")
    storage_reused = min(host_total, storage_completed)
    host_portion = max(0, host_total - storage_reused)
    device_portion = max(0, prefix_len - host_total)

    req.cached_tokens_device = device_portion
    req.cached_tokens_host = host_portion
    req.cached_tokens_storage = storage_completed
    req.reused_tokens_device = device_portion
    req.reused_tokens_host = host_portion
    req.reused_tokens_storage = storage_reused


def record_remote_prefetch_stats(
    req: Any, *, completed_tokens: int, expected_tokens: int
) -> None:
    """Preserve remote prefetch counters across scheduler admission retries."""
    completed = max(0, int(completed_tokens or 0))
    expected = max(0, int(expected_tokens or 0))
    changed = False

    if completed > _int_attr(req, "storage_hit_length"):
        req.storage_hit_length = completed
        changed = True
    if expected > _int_attr(req, "storage_query_tokens"):
        req.storage_query_tokens = expected
        changed = True

    if changed:
        req._cache_breakdown_computed = False


def build_cached_tokens_details(
    req: Any,
    *,
    enable_hicache_storage: bool,
    storage_backend_type: str = "none",
) -> Optional[dict]:
    device = _int_attr(req, "cached_tokens_device")
    host = _int_attr(req, "cached_tokens_host")
    storage = _int_attr(req, "cached_tokens_storage")
    storage_query = _int_attr(req, "storage_query_tokens")

    has_detailed_cache = device > 0 or host > 0 or storage > 0
    has_storage_query = enable_hicache_storage and storage_query > 0
    if has_detailed_cache or has_storage_query:
        details = {
            "device": device,
            "host": host,
            "reused_device": _int_attr(req, "reused_tokens_device"),
            "reused_host": _int_attr(req, "reused_tokens_host"),
        }
        if enable_hicache_storage:
            details["storage"] = storage
            details["storage_query"] = storage_query
            details["reused_storage"] = _int_attr(req, "reused_tokens_storage")
            details["storage_backend"] = storage_backend_type
        return details

    cached_tokens = _int_attr(req, "cached_tokens")
    if cached_tokens > 0:
        return {
            "device": cached_tokens,
            "host": 0,
            "reused_device": cached_tokens,
            "reused_host": 0,
        }

    return None

__all__ = [
    "apply_cache_source_breakdown",
    "build_cached_tokens_details",
    "record_remote_prefetch_stats",
]
