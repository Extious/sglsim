from __future__ import annotations

from types import SimpleNamespace

import torch

from sglsim.srt.simulator.native.cache_controller_patch import (
    estimate_host_transfer_seconds,
    estimate_storage_transfer_seconds,
)


def test_host_transfer_seconds_is_size_over_bandwidth():
    assert estimate_host_transfer_seconds(100, 10, 1000) == 1.0


def test_storage_transfer_seconds_zero_for_no_tokens():
    assert estimate_storage_transfer_seconds(0, 10, 1000) == 0.0


def test_segmented_host_transfer_is_slower_for_scattered_indices():
    import torch

    from sglsim.srt.simulator.native import cache_controller_patch

    host_pool = SimpleNamespace(size_per_token=100, layer_num=2)
    contiguous = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    scattered = torch.tensor([0, 1, 10, 11], dtype=torch.int64)
    contiguous_dur = cache_controller_patch._segmented_host_transfer_seconds(
        contiguous, contiguous, 10, cat="H2D"
    )
    scattered_dur = cache_controller_patch._segmented_host_transfer_seconds(
        scattered, scattered, 10, cat="H2D"
    )
    assert scattered_dur > contiguous_dur


def test_simulated_host_pool_transfer_records_load_and_backup_durations():
    from sglsim.srt.simulator.native import cache_controller_patch
    from sglsim.srt.simulator.state import SimulatorState, set_active_state

    cache_controller_patch.install(SimpleNamespace())
    state = SimulatorState()
    set_active_state(state)
    try:
        host_pool = SimpleNamespace(size_per_token=10, layer_num=2)
        host_indices = torch.tensor([0, 1, 2, 6], dtype=torch.int64)
        device_indices = torch.tensor([10, 11, 12, 20], dtype=torch.int64)

        cache_controller_patch.simulated_load_to_device_per_layer(
            host_pool, None, host_indices, device_indices, 0, "kernel"
        )
        cache_controller_patch.simulated_backup_from_device_all_layer(
            host_pool, None, host_indices, device_indices, "kernel"
        )

        assert state.hicache_l2_load_dur > 0
        assert state.hicache_l2_backup_dur > 0
    finally:
        set_active_state(None)


def test_native_host_pool_uses_small_cpu_metadata_without_cuda_register(monkeypatch):
    from sglsim.srt.mem_cache import memory_pool_host
    from sglsim.srt.simulator.native import cache_controller_patch

    class HugeMemory:
        available = 10**12

    def fail_cuda_register():
        raise AssertionError("native simulator must not call cudaHostRegister")

    requested_dims = []

    def fake_allocate(self, dims, dtype, device):
        requested_dims.append(tuple(int(x) for x in dims))
        assert max(requested_dims[-1]) <= 20
        return torch.empty(dims, dtype=dtype, device=device)

    monkeypatch.setattr(memory_pool_host.psutil, "virtual_memory", lambda: HugeMemory())
    monkeypatch.setattr(memory_pool_host.HostTensorAllocator, "allocate", fake_allocate)
    monkeypatch.setattr(torch.cuda, "cudart", fail_cuda_register, raising=False)

    cache_controller_patch.install(SimpleNamespace())
    device_pool = SimpleNamespace(
        store_dtype=torch.float16,
        size=16,
        start_layer=0,
        end_layer=1,
        head_num=1,
        head_dim=1,
        layer_num=1,
        device="cpu",
    )

    host_pool = memory_pool_host.MHATokenToKVPoolHost(
        device_pool,
        host_to_device_ratio=1.1,
        host_size=60,
        page_size=4,
        layout="layer_first",
        pin_memory=True,
        device="cpu",
    )

    assert host_pool.size == 20
    assert host_pool.available_size() == 20
    if requested_dims:
        assert max(requested_dims[0]) <= 20
