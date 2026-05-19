from __future__ import annotations

import os

__version__ = "0.1.0"

# sglsim vendors SGLang runtime code but replaces GPU inference and device
# communication with CPU simulation. Force SGLang's platform probes onto the CPU
# path before any vendored module imports torch.
if os.getenv("SGLSIM_FORCE_CPU_SIM", "1") == "1":
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
