from __future__ import annotations

import numpy as np


def split_contiguous_page_segments(
    host_page_ids: list[int],
    device_page_ids: list[int],
) -> list[int]:
    if not host_page_ids:
        return []
    if len(host_page_ids) != len(device_page_ids):
        raise ValueError("host_page_ids and device_page_ids length mismatch")

    host = np.asarray(host_page_ids, dtype=np.int64)
    dev = np.asarray(device_page_ids, dtype=np.int64)
    if len(host) == 1:
        return [1]

    cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
    cut = np.flatnonzero(~cont) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(host)]
    return [int(ends[i] - starts[i]) for i in range(len(starts))]


def split_contiguous_page_segments_from_ids(page_ids: list[int]) -> list[int]:
    if not page_ids:
        return []
    pages = np.asarray(page_ids, dtype=np.int64)
    if len(pages) == 1:
        return [1]
    cont = np.diff(pages) == 1
    cut = np.flatnonzero(~cont) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(pages)]
    return [int(ends[i] - starts[i]) for i in range(len(starts))]
