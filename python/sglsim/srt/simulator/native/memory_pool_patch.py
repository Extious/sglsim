from __future__ import annotations

from sglsim.srt.simulator.native.cpu_allocator_kernels import (
    alloc_decode_cpu,
    alloc_extend_cpu,
)

_INSTALLED = False


class IndexableWrapper:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def install(config) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from sglsim.srt.mem_cache import allocator

    allocator.alloc_extend_kernel = IndexableWrapper(alloc_extend_cpu)
    allocator.alloc_decode_kernel = IndexableWrapper(alloc_decode_cpu)


def is_installed() -> bool:
    return _INSTALLED

