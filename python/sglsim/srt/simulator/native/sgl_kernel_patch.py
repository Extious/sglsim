from __future__ import annotations

import importlib

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        load_utils = importlib.import_module("sgl_kernel.load_utils")
    except ImportError:
        return

    def _noop_load_architecture_specific_ops(*args, **kwargs):
        return None

    load_utils._load_architecture_specific_ops = _noop_load_architecture_specific_ops
    _INSTALLED = True


def is_installed() -> bool:
    return _INSTALLED
