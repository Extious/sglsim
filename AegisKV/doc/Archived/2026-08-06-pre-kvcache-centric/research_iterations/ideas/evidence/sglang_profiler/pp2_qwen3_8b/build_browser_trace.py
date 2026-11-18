#!/usr/bin/env python3
"""Merge the selected PP=2 traces and whole-device counters for Perfetto.

The raw torch.profiler files remain authoritative. This helper normalizes
their per-process clock bases, combines one prefill+decode interval from each
PP rank, and adds sampled GPU/CPU/PCIe counters. Split stage traces remain
accepted for diagnostic replay, but the formal artifact contract requires the
two automatic combined rank traces.
The result is one standard compressed Chrome trace that can be imported into
Perfetto or Chrome tracing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import analyze_run


def _one_path(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one trace matching {root / pattern}, found {paths}"
        )
    return paths[0]


def selected_trace_paths(run_dir: Path) -> list[tuple[str, int, Path]]:
    profile_root = run_dir / "profile" / "traces"
    selected: list[tuple[str, int, Path]] = []
    directory = profile_root / "combined"
    combined_paths = [
        sorted(directory.glob(f"*PP-{rank}.trace.json.gz"))
        for rank in (0, 1)
    ]
    if all(len(paths) == 1 for paths in combined_paths):
        return [
            ("combined", rank, combined_paths[rank][0]) for rank in (0, 1)
        ]
    for stage, suffix in (("prefill", "EXTEND"), ("decode", "DECODE")):
        for rank in (0, 1):
            selected.append(
                (
                    stage,
                    rank,
                    _one_path(
                        directory,
                        f"*PP-{rank}-{suffix}.trace.json.gz",
                    ),
                )
            )
    return selected


def _load_trace(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("traceEvents"), list):
        raise RuntimeError(f"{path} has no traceEvents list")
    if not isinstance(payload.get("baseTimeNanoseconds"), (int, float)):
        raise RuntimeError(f"{path} has no numeric baseTimeNanoseconds")
    return payload


def _pid_namespaces(
    loaded: list[tuple[str, int, Path, dict[str, Any]]],
) -> dict[tuple[str, int], dict[Any, int]]:
    """Create stable, disjoint Perfetto process IDs for all four sources.

    Kineto traces reuse small process IDs for CUDA devices/streams and emit
    metadata for every visible device. Concatenating stage/rank JSON files
    without remapping lets their metadata overwrite one another in Perfetto.
    """
    source_keys = [(stage, rank) for stage, rank, _path, _payload in loaded]
    observed: dict[tuple[str, int], set[Any]] = {
        key: set() for key in source_keys
    }
    for stage, rank, _path, payload in loaded:
        for event in payload["traceEvents"]:
            if "pid" in event:
                observed[(stage, rank)].add(event["pid"])
    result: dict[tuple[str, int], dict[Any, int]] = {}
    for source_index, key in enumerate(source_keys):
        result[key] = {
            original: 8_000_000 + source_index * 100_000 + index
            for index, original in enumerate(
                sorted(
                    observed[key],
                    key=lambda value: (type(value).__name__, str(value)),
                )
            )
        }
    return result


def _flow_id(stage: str, rank: int, original: Any) -> int | str:
    """Namespace Chrome flow IDs, which are otherwise global across files."""
    stage_index = {"combined": 1, "prefill": 2, "decode": 3}[stage]
    if isinstance(original, int) and not isinstance(original, bool):
        return rank * 1_000_000_000_000 + stage_index * 100_000_000_000 + original
    return f"aegiskv:{stage}:pp{rank}:{original}"


def _counter_event(
    *,
    pid: int,
    timestamp_us: float,
    name: str,
    values: dict[str, float],
) -> dict[str, Any]:
    return {
        "ph": "C",
        "cat": "aegiskv.resource",
        "name": name,
        "pid": pid,
        "tid": 0,
        "ts": timestamp_us,
        "args": values,
    }


def _metadata_events(pid: int, name: str, sort_index: int) -> list[dict[str, Any]]:
    return [
        {
            "ph": "M",
            "name": "process_name",
            "pid": pid,
            "tid": 0,
            "args": {"name": name},
        },
        {
            "ph": "M",
            "name": "process_sort_index",
            "pid": pid,
            "tid": 0,
            "args": {"sort_index": sort_index},
        },
    ]


def _resource_events(
    run_dir: Path,
    *,
    merged_base_ns: float,
    intervals: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    interval = analyze_run.profile_interval(intervals)
    if interval is None:
        raise RuntimeError(
            "steady_load/profile_prefill/profile_decode phase markers are missing"
        )
    start_wall, end_wall = interval

    events: list[dict[str, Any]] = []
    gpu_pids = {"0": 9_800_000, "1": 9_800_001}
    for gpu, pid in gpu_pids.items():
        events.extend(
            _metadata_events(pid, f"GPU {gpu} whole-device telemetry", 900 + int(gpu))
        )
    cpu_pid = 9_799_999
    phase_pid = 9_799_998
    events.extend(_metadata_events(cpu_pid, "SGLang process-tree telemetry", 902))
    events.extend(_metadata_events(phase_pid, "AegisKV workload phases", 899))

    for phase in ("steady_load", "profile_prefill", "profile_decode"):
        if phase not in intervals:
            continue
        phase_start, phase_end = intervals[phase]
        events.append(
            {
                "ph": "X",
                "cat": "aegiskv.phase",
                "name": phase,
                "pid": phase_pid,
                "tid": 0,
                "ts": phase_start * 1_000_000.0 - merged_base_ns / 1000.0,
                "dur": (phase_end - phase_start) * 1_000_000.0,
                "args": {"wall_start_s": phase_start, "wall_end_s": phase_end},
            }
        )

    resources = analyze_run.read_resources(run_dir / "resource_samples.csv")
    cpu_seen_timestamps: set[float] = set()
    gpu_columns = {
        "gpu_util_pct": "utilization.gpu",
        "memory_controller_util_pct": "utilization.memory",
        "hbm_used_mib": "memory.used",
        "hbm_total_mib": "memory.total",
        "power_w": "power.draw",
        "temperature_c": "temperature.gpu",
        "sm_clock_mhz": "clocks.current.sm",
        "memory_clock_mhz": "clocks.current.memory",
        "gpu_compute_processes": "gpu_compute_process_count",
        "foreign_gpu_processes": "foreign_gpu_process_count",
    }
    for row in resources:
        wall_ts = float(row["sample_wall_ts"])
        if not start_wall <= wall_ts <= end_wall:
            continue
        gpu = str(row["index"])
        if gpu not in gpu_pids:
            continue
        trace_ts = wall_ts * 1_000_000.0 - merged_base_ns / 1000.0
        values = {
            name: converted
            for name, column in gpu_columns.items()
            if (converted := analyze_run.number(row.get(column))) is not None
        }
        if values:
            events.append(
                _counter_event(
                    pid=gpu_pids[gpu],
                    timestamp_us=trace_ts,
                    name="GPU telemetry",
                    values=values,
                )
            )
        if wall_ts not in cpu_seen_timestamps:
            cpu_seen_timestamps.add(wall_ts)
            cpu_values = {}
            for name, column, divisor in (
                ("cpu_pct", "server_tree_cpu_pct", 1.0),
                ("rss_mib", "server_tree_rss_kib", 1024.0),
                ("processes", "server_tree_processes", 1.0),
            ):
                converted = analyze_run.number(row.get(column))
                if converted is not None:
                    cpu_values[name] = converted / divisor
            if cpu_values:
                events.append(
                    _counter_event(
                        pid=cpu_pid,
                        timestamp_us=trace_ts,
                        name="SGLang process tree",
                        values=cpu_values,
                    )
                )

    pcie = analyze_run.read_pcie(run_dir / "pcie_throughput_dmon.csv")
    for row in pcie:
        wall_ts = float(row["sample_wall_ts"])
        gpu = str(row["gpu"])
        if not start_wall <= wall_ts <= end_wall or gpu not in gpu_pids:
            continue
        values = {
            name: float(value)
            for name, value in (
                ("pcie_rx_mb_s", row.get("rxpci_mb_s")),
                ("pcie_tx_mb_s", row.get("txpci_mb_s")),
            )
            if value is not None
        }
        if values:
            events.append(
                _counter_event(
                    pid=gpu_pids[gpu],
                    timestamp_us=wall_ts * 1_000_000.0
                    - merged_base_ns / 1000.0,
                    name="PCIe throughput",
                    values=values,
                )
            )
    return events


def build_combined_trace(run_dir: Path, output_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    source_entries = selected_trace_paths(run_dir)
    loaded = [
        (stage, rank, path, _load_trace(path))
        for stage, rank, path in source_entries
    ]
    merged_base_ns = min(
        float(payload["baseTimeNanoseconds"]) for _, _, _, payload in loaded
    )
    pid_namespaces = _pid_namespaces(loaded)

    events: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    device_properties: dict[tuple[Any, Any], dict[str, Any]] = {}
    for stage, rank, path, payload in loaded:
        source_key = (stage, rank)
        source_base_ns = float(payload["baseTimeNanoseconds"])
        offset_us = (source_base_ns - merged_base_ns) / 1000.0
        source_events = payload["traceEvents"]
        for event in source_events:
            if "pid" in event:
                event["pid"] = pid_namespaces[source_key][event["pid"]]
            if event.get("ph") in {"s", "t", "f"} and "id" in event:
                event["id"] = _flow_id(stage, rank, event["id"])
            timestamp = event.get("ts")
            if isinstance(timestamp, (int, float)):
                event["ts"] = float(timestamp) + offset_us
        events.extend(source_events)
        namespace_values = list(pid_namespaces[source_key].values())
        source_manifest.append(
            {
                "stage": stage,
                "pp_rank": rank,
                "path": str(path.relative_to(run_dir)),
                "events": len(source_events),
                "base_time_nanoseconds": source_base_ns,
                "clock_offset_us": offset_us,
                "pid_namespace_size": len(pid_namespaces[source_key]),
                "pid_namespace_min": min(namespace_values),
                "pid_namespace_max": max(namespace_values),
                "flow_id_namespace": f"{stage}:pp{rank}",
            }
        )
        for device in payload.get("deviceProperties", []):
            key = (device.get("id"), device.get("name"))
            device_properties[key] = device

    intervals = analyze_run.read_markers(
        run_dir / "profile" / "phase_markers.jsonl"
    )
    resource_events = _resource_events(
        run_dir, merged_base_ns=merged_base_ns, intervals=intervals
    )
    events.extend(resource_events)
    events.sort(
        key=lambda event: (
            0 if event.get("ph") == "M" else 1,
            float(event.get("ts", 0.0)),
        )
    )

    merged = {
        "schemaVersion": 1,
        "displayTimeUnit": "ms",
        "baseTimeNanoseconds": int(merged_base_ns),
        "record_shapes": 1,
        "with_stack": 0,
        "traceName": "AegisKV Qwen3-8B PP=2 prefill+decode+resources",
        "deviceProperties": list(device_properties.values()),
        "sourceTraces": source_manifest,
        "traceEvents": events,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(merged, handle, ensure_ascii=False, separators=(",", ":"))

    manifest = {
        "output": str(output_path.relative_to(run_dir)),
        "source_traces": source_manifest,
        "source_trace_count": len(source_manifest),
        "source_event_count": sum(item["events"] for item in source_manifest),
        "resource_event_count": len(resource_events),
        "total_event_count": len(events),
        "base_time_nanoseconds": int(merged_base_ns),
        "phase_intervals": intervals,
    }
    (output_path.parent / "browser_trace_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output or (
        run_dir
        / "browser_trace"
        / "pp2-prefill-decode-resources.trace.json.gz"
    )
    manifest = build_combined_trace(run_dir, output.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
