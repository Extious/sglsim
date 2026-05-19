from __future__ import annotations

import sys
import types


def test_sgl_kernel_patch_bypasses_architecture_specific_ops(monkeypatch):
    monkeypatch.setitem(sys.modules, "sgl_kernel", types.ModuleType("sgl_kernel"))
    module = types.ModuleType("sgl_kernel.load_utils")

    def _raise(*args, **kwargs):
        raise RuntimeError("should be bypassed")

    module._load_architecture_specific_ops = _raise
    monkeypatch.setitem(sys.modules, "sgl_kernel.load_utils", module)

    from sglsim.srt.simulator.native import sgl_kernel_patch

    sgl_kernel_patch._INSTALLED = False
    sgl_kernel_patch.install()
    module._load_architecture_specific_ops()
    assert sgl_kernel_patch.is_installed() is True
