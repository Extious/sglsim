#!/usr/bin/env python3
"""Convert torch CUDA allocator snapshots into self-contained browser HTML."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from torch.cuda._memory_viz import segment_plot, trace_plot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    snapshots = sorted(args.root.rglob("*-memory*.pickle"))
    if not snapshots:
        raise SystemExit(f"no memory snapshots found under {args.root}")
    for snapshot_path in snapshots:
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        stem = snapshot_path.name.removesuffix(".pickle")
        trace_path = snapshot_path.with_name(stem + "-memory-timeline.html")
        segment_path = snapshot_path.with_name(stem + "-allocator-history.html")
        trace_path.write_text(trace_plot(snapshot), encoding="utf-8")
        segment_path.write_text(segment_plot(snapshot), encoding="utf-8")
        print(trace_path)
        print(segment_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
