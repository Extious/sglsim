from __future__ import annotations


def _missing_kernel(*args, **kwargs):
    raise NotImplementedError(
        "sglsim provides only CPU simulation shims for selected sgl_kernel APIs; "
        "this GPU kernel is outside the native simulator boundary."
    )


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return _missing_kernel

