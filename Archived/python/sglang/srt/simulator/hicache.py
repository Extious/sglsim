from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np

from sglang.srt.simulator.state import get_active_state


class SimulatorHiCacheStorage:
    def __init__(self, storage_file_path: str = "/tmp/sglang_simulator/hicache_storage_keys.txt"):
        self.storage: set[str] = set()
        self.storage_file_path = storage_file_path
        os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)
        if os.path.exists(self.storage_file_path):
            with open(self.storage_file_path) as f:
                for line in f:
                    key = line.strip()
                    if key:
                        self.storage.add(key)

    def register_mem_pool_host(self, mem_pool_host):
        return None

    def get(self, key: str, target_location: Optional[Any] = None, target_sizes: Optional[Any] = None):
        del target_location, target_sizes
        return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ):
        del target_locations, target_sizes
        return [None for _ in keys]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        del value, target_location, target_sizes
        if key in self.storage:
            return True
        self.storage.add(key)
        with open(self.storage_file_path, "a") as f:
            f.write(key + "\n")
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        extra_info=None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        del values, extra_info, target_locations, target_sizes
        for key in keys:
            if not self.set(key):
                return False
        return True

    def exists(self, key: str) -> bool:
        return key in self.storage

    def batch_exists(self, keys: List[str], extra_info=None) -> int:
        del extra_info
        for i, key in enumerate(keys):
            if key not in self.storage:
                return i
        return len(keys)

    def clear(self) -> bool:
        self.storage.clear()
        with open(self.storage_file_path, "w"):
            pass
        return True

    def get_stats(self):
        return None


def _to_numpy_indices(indices) -> np.ndarray:
    if hasattr(indices, "detach"):
        indices = indices.detach()
    if hasattr(indices, "cpu"):
        indices = indices.cpu()
    if hasattr(indices, "numpy"):
        return np.asarray(indices.numpy(), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def estimate_segment_transfer_seconds(
    host_indices,
    device_indices,
    bytes_per_token: int,
    bandwidth_bytes: Optional[float],
) -> float:
    if bandwidth_bytes is None or bandwidth_bytes <= 0:
        return 0.0
    host = _to_numpy_indices(host_indices)
    device = _to_numpy_indices(device_indices)
    num_indices = len(host)
    if num_indices == 0:
        return 0.0
    if num_indices == 1:
        return bytes_per_token / bandwidth_bytes
    contiguous = (np.diff(host) == 1) & (np.diff(device) == 1)
    cut = np.flatnonzero(~contiguous) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, num_indices]
    segment_lens = (ends - starts).astype(np.float64)
    size_bytes = segment_lens * float(bytes_per_token)
    return float(np.sum(size_bytes / bandwidth_bytes))


def record_hicache_l2_load(
    host_indices,
    device_indices,
    bytes_per_token: int,
    bandwidth_bytes: Optional[float],
) -> None:
    state = get_active_state()
    if state is not None:
        state.inc_hicache_l2_load_dur(
            estimate_segment_transfer_seconds(
                host_indices, device_indices, bytes_per_token, bandwidth_bytes
            )
        )


def record_hicache_l2_backup(
    host_indices,
    device_indices,
    bytes_per_token: int,
    bandwidth_bytes: Optional[float],
) -> None:
    state = get_active_state()
    if state is not None:
        state.inc_hicache_l2_backup_dur(
            estimate_segment_transfer_seconds(
                host_indices, device_indices, bytes_per_token, bandwidth_bytes
            )
        )
