#!/usr/bin/env python3
"""Extract stage-scoped scheduling evidence from SGLang torch traces.

Durations in Chrome traces are microseconds.  The script deliberately reports
observed GPU-event gaps, not "free PCIe/HBM bandwidth": an empty trace interval
does not prove that an uninstrumented engine or shared memory fabric is idle.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    import orjson
except ImportError:  # pragma: no cover - standard-library fallback
    orjson = None


HARDWARE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_NAMES = {
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cudaLaunchCooperativeKernel",
}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min_us": min(values) if values else None,
        "median_us": median(values) if values else None,
        "p95_us": percentile(values, 0.95),
        "max_us": max(values) if values else None,
    }


def union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def clipped_interval(
    event: dict[str, Any], start: float, end: float
) -> tuple[float, float] | None:
    event_start = float(event.get("ts", 0))
    event_end = event_start + float(event.get("dur", 0))
    left, right = max(start, event_start), min(end, event_end)
    return (left, right) if right > left else None


def gaps(
    merged: list[tuple[float, float]], start: float, end: float
) -> list[float]:
    result: list[float] = []
    cursor = start
    for left, right in merged:
        if left > cursor:
            result.append(left - cursor)
        cursor = max(cursor, right)
    if cursor < end:
        result.append(end - cursor)
    return result


def overlap_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def add_event_total(
    totals: dict[str, dict[str, float | int]],
    event: dict[str, Any],
    start: float,
    end: float,
) -> None:
    clipped = clipped_interval(event, start, end)
    if clipped is None:
        return
    name = str(event.get("name", "unknown"))
    item = totals.setdefault(name, {"count": 0, "duration_us": 0.0})
    item["count"] += 1
    item["duration_us"] += clipped[1] - clipped[0]


def ranked_event_totals(
    totals: dict[str, dict[str, float | int]], limit: int = 15
) -> list[dict[str, float | int | str]]:
    total_duration = sum(float(item["duration_us"]) for item in totals.values())
    rows = [
        {
            "name": name,
            "count": int(item["count"]),
            "duration_us": float(item["duration_us"]),
            "share_pct": (
                100.0 * float(item["duration_us"]) / total_duration
                if total_duration > 0
                else 0.0
            ),
        }
        for name, item in totals.items()
    ]
    return sorted(
        rows,
        key=lambda item: (float(item["duration_us"]), int(item["count"])),
        reverse=True,
    )[:limit]


def load_trace(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def select_step(event: dict[str, Any], wanted: str) -> bool:
    name = str(event.get("name", ""))
    if wanted == "prefill":
        return name.startswith("step[EXTEND")
    return name.startswith("step[DECODE")


def analyze(path: Path, wanted: str) -> dict[str, Any]:
    trace = load_trace(path)
    events = trace["traceEvents"]
    gpu_steps = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "gpu_user_annotation"
        and select_step(event, wanted)
    ]
    cpu_steps = {
        event.get("args", {}).get("External id"): event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and select_step(event, wanted)
    }
    hardware = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") in HARDWARE_CATEGORIES
        and float(event.get("dur", 0)) > 0
    ]
    runtime = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "cuda_runtime"
    ]
    cpu_annotations = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "user_annotation"
    ]
    cpu_ops = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "cpu_op"
    ]

    records: list[dict[str, Any]] = []
    all_idle: list[float] = []
    all_launch_gaps: list[float] = []
    kernel_totals: dict[str, dict[str, float | int]] = {}
    cpu_annotation_totals: dict[str, dict[str, float | int]] = {}
    cpu_op_totals: dict[str, dict[str, float | int]] = {}
    cuda_runtime_totals: dict[str, dict[str, float | int]] = {}
    nccl_totals: dict[str, dict[str, float | int]] = {}
    for step_index, step in enumerate(sorted(gpu_steps, key=lambda item: item["ts"])):
        start = float(step["ts"])
        end = start + float(step["dur"])
        step_hardware = [
            event
            for event in hardware
            if float(event.get("ts", 0)) < end
            and float(event.get("ts", 0)) + float(event.get("dur", 0)) > start
        ]
        kernel_intervals = union(
            interval
            for event in step_hardware
            if event.get("cat") == "kernel"
            if (interval := clipped_interval(event, start, end)) is not None
        )
        hardware_intervals = union(
            interval
            for event in step_hardware
            if (interval := clipped_interval(event, start, end)) is not None
        )
        memcpy_events = [
            event for event in step_hardware if event.get("cat") == "gpu_memcpy"
        ]
        memcpy_intervals = union(
            interval
            for event in memcpy_events
            if (interval := clipped_interval(event, start, end)) is not None
        )
        idle = gaps(hardware_intervals, start, end)
        all_idle.extend(idle)

        external_id = step.get("args", {}).get("External id")
        cpu_step = cpu_steps.get(external_id)
        if cpu_step is None:
            cpu_start, cpu_end = start, end
        else:
            cpu_start = float(cpu_step["ts"])
            cpu_end = cpu_start + float(cpu_step["dur"])
        launches = sorted(
            (
                event
                for event in runtime
                if event.get("name") in LAUNCH_NAMES
                and cpu_start <= float(event.get("ts", 0)) < cpu_end
            ),
            key=lambda item: item["ts"],
        )
        launch_gaps = [
            max(
                0.0,
                float(current["ts"])
                - (
                    float(previous["ts"])
                    + float(previous.get("dur", 0))
                ),
            )
            for previous, current in zip(launches, launches[1:])
        ]
        all_launch_gaps.extend(launch_gaps)
        for event in step_hardware:
            if event.get("cat") == "kernel":
                add_event_total(kernel_totals, event, start, end)
                if "nccl" in str(event.get("name", "")).lower():
                    add_event_total(nccl_totals, event, start, end)
        for event in cpu_annotations:
            add_event_total(cpu_annotation_totals, event, cpu_start, cpu_end)
        for event in cpu_ops:
            add_event_total(cpu_op_totals, event, cpu_start, cpu_end)
        for event in runtime:
            add_event_total(cuda_runtime_totals, event, cpu_start, cpu_end)

        memcpy_by_kind: dict[str, dict[str, float | int]] = {}
        for event in memcpy_events:
            kind = str(event.get("name", "unknown"))
            item = memcpy_by_kind.setdefault(
                kind, {"count": 0, "bytes": 0, "duration_us": 0.0}
            )
            item["count"] += 1
            item["bytes"] += int(event.get("args", {}).get("bytes", 0) or 0)
            item["duration_us"] += float(event.get("dur", 0))

        kernel_busy = sum(right - left for left, right in kernel_intervals)
        hardware_busy = sum(right - left for left, right in hardware_intervals)
        duration = end - start
        records.append(
            {
                "step_index": step_index,
                "name": step["name"],
                "external_id": external_id,
                "duration_us": duration,
                "hardware_first_to_last_span_us": (
                    hardware_intervals[-1][1] - hardware_intervals[0][0]
                    if hardware_intervals
                    else 0.0
                ),
                "kernel_union_busy_us": kernel_busy,
                "kernel_union_occupancy_pct": 100.0 * kernel_busy / duration,
                "all_hardware_union_busy_us": hardware_busy,
                "all_hardware_union_occupancy_pct": 100.0
                * hardware_busy
                / duration,
                "kernel_launches": sum(
                    1 for event in step_hardware if event.get("cat") == "kernel"
                ),
                "memcpy": memcpy_by_kind,
                "memcpy_kernel_overlap_us": overlap_duration(
                    memcpy_intervals, kernel_intervals
                ),
                "observed_hardware_idle_gaps_us": summary(idle),
                "observed_idle_gaps_ge_10us": sum(value >= 10 for value in idle),
                "observed_idle_gaps_ge_50us": sum(value >= 50 for value in idle),
                "observed_idle_gaps_ge_100us": sum(value >= 100 for value in idle),
                "observed_idle_gaps_ge_500us": sum(value >= 500 for value in idle),
                "cpu_kernel_launch_calls": len(launches),
                "cpu_launch_issue_gaps_us": summary(launch_gaps),
            }
        )

    def aggregate(field: str) -> dict[str, float | int | None]:
        return summary([float(record[field]) for record in records])

    memcpy_totals: dict[str, dict[str, float | int]] = {}
    for record in records:
        for kind, values in record["memcpy"].items():
            item = memcpy_totals.setdefault(
                kind, {"count": 0, "bytes": 0, "duration_us": 0.0}
            )
            for key in item:
                item[key] += values[key]

    nccl_hardware_events = [
        event
        for event in events
        if "nccl" in str(event.get("name", "")).lower()
        and event.get("ph") == "X"
        and event.get("cat") in HARDWARE_CATEGORIES
    ]
    return {
        "trace": str(path),
        "wanted_stage": wanted,
        "trace_schema_version": trace.get("schemaVersion"),
        "device": (trace.get("deviceProperties") or [{}])[0].get("name"),
        "selected_step_count": len(records),
        "aggregate": {
            "step_duration_us": aggregate("duration_us"),
            "hardware_first_to_last_span_us": aggregate(
                "hardware_first_to_last_span_us"
            ),
            "kernel_union_busy_us": aggregate("kernel_union_busy_us"),
            "kernel_union_occupancy_pct": aggregate(
                "kernel_union_occupancy_pct"
            ),
            "all_hardware_union_busy_us": aggregate(
                "all_hardware_union_busy_us"
            ),
            "all_hardware_union_occupancy_pct": aggregate(
                "all_hardware_union_occupancy_pct"
            ),
            "observed_hardware_idle_gaps_us": summary(all_idle),
            "observed_idle_gap_threshold_counts": {
                "ge_10us": sum(value >= 10 for value in all_idle),
                "ge_50us": sum(value >= 50 for value in all_idle),
                "ge_100us": sum(value >= 100 for value in all_idle),
                "ge_500us": sum(value >= 500 for value in all_idle),
                "ge_1000us": sum(value >= 1000 for value in all_idle),
            },
            "cpu_launch_issue_gaps_us": summary(all_launch_gaps),
            "memcpy": memcpy_totals,
            "nccl_hardware_event_count_in_trace": len(nccl_hardware_events),
        },
        "top_kernels": ranked_event_totals(kernel_totals),
        "top_sglang_cpu_annotations": ranked_event_totals(
            cpu_annotation_totals
        ),
        "top_cpu_ops": ranked_event_totals(cpu_op_totals),
        "top_cuda_runtime_calls": ranked_event_totals(cuda_runtime_totals),
        "top_nccl_kernels": ranked_event_totals(nccl_totals),
        "steps": records,
        "interpretation_boundary": (
            "These are intervals with no recorded CUDA kernel/memcpy/memset. "
            "They do not prove PCIe copy-engine, HBM, NIC, or system-wide "
            "bandwidth availability; CUPTI counters or an interference "
            "experiment is required before calling them safe backup windows."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "prefill": analyze(args.prefill, "prefill"),
        "decode": analyze(args.decode, "decode"),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
