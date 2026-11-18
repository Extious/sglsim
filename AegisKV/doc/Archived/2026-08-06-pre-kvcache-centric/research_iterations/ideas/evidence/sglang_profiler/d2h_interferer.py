#!/usr/bin/env python3
"""Generate bounded, chunked, pinned GPU-to-host traffic on one CUDA device."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--buffer-mib", type=int, default=256)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.buffer_mib % args.chunk_mib:
        raise ValueError("buffer-mib must be divisible by chunk-mib")

    element_size = torch.empty((), dtype=torch.uint8).element_size()
    total_elements = args.buffer_mib * 1024 * 1024 // element_size
    chunk_elements = args.chunk_mib * 1024 * 1024 // element_size
    device = torch.empty(total_elements, dtype=torch.uint8, device="cuda")
    host = torch.empty(total_elements, dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.Stream()

    # Fault pages and warm both allocations before taking measurements.
    device.fill_(1)
    with torch.cuda.stream(stream):
        host[:chunk_elements].copy_(device[:chunk_elements], non_blocking=True)
    stream.synchronize()

    start = time.perf_counter()
    copies = 0
    while time.perf_counter() - start < args.duration_s:
        for offset in range(0, total_elements, chunk_elements):
            with torch.cuda.stream(stream):
                host[offset : offset + chunk_elements].copy_(
                    device[offset : offset + chunk_elements], non_blocking=True
                )
            # One outstanding chunk at a time gives a bounded, preemptible
            # copy stream while still creating sustained D2H pressure.
            stream.synchronize()
            copies += 1
            if time.perf_counter() - start >= args.duration_s:
                break
    elapsed = time.perf_counter() - start
    bytes_copied = copies * args.chunk_mib * 1024 * 1024
    result = {
        "gpu": torch.cuda.get_device_name(),
        "duration_s": elapsed,
        "buffer_mib": args.buffer_mib,
        "chunk_mib": args.chunk_mib,
        "copies": copies,
        "bytes_copied": bytes_copied,
        "throughput_GBps_decimal": bytes_copied / elapsed / 1e9,
        "copy_direction": "D2H",
        "host_memory": "pinned",
        "synchronization": "per chunk",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
