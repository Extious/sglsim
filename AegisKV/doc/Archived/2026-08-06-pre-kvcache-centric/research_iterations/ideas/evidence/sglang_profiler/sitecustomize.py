"""Runtime-only compatibility shim for the pinned SGLang profiler branch.

The profiler branch at fdebc938f7 predates upstream fix 14c0a31829 and applies
``huggingface_hub.dataclasses.strict`` to a non-dataclass
``Cohere2MoeConfig``.  On the locally installed huggingface_hub 1.10.2 this
prevents the server argument module from importing.  Removing the newly added
``strict`` symbol makes that branch take its existing "older
huggingface_hub" fallback, without modifying either checkout or environment.
"""

try:
    import huggingface_hub.dataclasses as _hf_dataclasses

    if hasattr(_hf_dataclasses, "strict"):
        delattr(_hf_dataclasses, "strict")
except Exception:
    # This file is only an import-time compatibility aid.  The real failure,
    # if any, should remain visible from the SGLang process.
    pass
