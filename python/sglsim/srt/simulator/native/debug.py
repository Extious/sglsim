from __future__ import annotations

import os
import sys
import time


def trace(message: str) -> None:
    if os.getenv("SGLSIM_NATIVE_TRACE", "0") != "1":
        return
    print(f"[SGLSIM_NATIVE_TRACE {time.time():.3f}] {message}", file=sys.stderr, flush=True)
