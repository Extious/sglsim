from __future__ import annotations


def should_count_backup_hit(write_policy: str, *, chunked: bool) -> bool:
    """Return whether an inserted HiCache segment should count toward backup hits."""
    if not chunked:
        return True
    return write_policy in {"write_through", "write_through_selective"}
