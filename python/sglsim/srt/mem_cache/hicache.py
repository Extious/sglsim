from __future__ import annotations

import os
from typing import Any, Optional

from sglsim.srt.simulator.state import get_active_state


class HiCacheStorage:
    def __init__(self, storage_file_path: str = "/tmp/sglsim/hicache_storage_keys.txt"):
        self.storage_file_path = storage_file_path
        self.storage: set[str] = set()
        os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)
        if os.path.exists(self.storage_file_path):
            with open(self.storage_file_path) as f:
                for line in f:
                    key = line.strip()
                    if key:
                        self.storage.add(key)

    def register_mem_pool_host(self, mem_pool_host):
        return None

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ):
        del key, target_location, target_sizes
        return None

    def batch_get(
        self,
        keys: list[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ):
        del target_locations, target_sizes
        return [self.get(key) for key in keys]

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
        keys: list[str],
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

    def batch_exists(self, keys: list[str], extra_info=None) -> int:
        del extra_info
        for index, key in enumerate(keys):
            if key not in self.storage:
                return index
        return len(keys)

    def clear(self) -> bool:
        self.storage.clear()
        with open(self.storage_file_path, "w"):
            pass
        return True

    def get_stats(self):
        return None


SimulatorHiCacheStorage = HiCacheStorage


def estimate_transfer_seconds(
    *,
    num_tokens: int,
    bytes_per_token: int | float,
    bandwidth_bytes: Optional[float],
) -> float:
    if num_tokens <= 0 or bandwidth_bytes is None or bandwidth_bytes <= 0:
        return 0.0
    return float(num_tokens) * float(bytes_per_token) / float(bandwidth_bytes)


def record_hicache_l2_load(
    *,
    num_tokens: int,
    bytes_per_token: int | float,
    bandwidth_bytes: Optional[float],
) -> None:
    state = get_active_state()
    if state is not None:
        state.inc_hicache_l2_load_dur(
            estimate_transfer_seconds(
                num_tokens=num_tokens,
                bytes_per_token=bytes_per_token,
                bandwidth_bytes=bandwidth_bytes,
            )
        )


def record_hicache_l2_backup(
    *,
    num_tokens: int,
    bytes_per_token: int | float,
    bandwidth_bytes: Optional[float],
) -> None:
    state = get_active_state()
    if state is not None:
        state.inc_hicache_l2_backup_dur(
            estimate_transfer_seconds(
                num_tokens=num_tokens,
                bytes_per_token=bytes_per_token,
                bandwidth_bytes=bandwidth_bytes,
            )
        )

