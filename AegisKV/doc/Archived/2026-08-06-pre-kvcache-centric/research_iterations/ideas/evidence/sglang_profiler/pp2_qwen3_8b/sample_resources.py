#!/usr/bin/env python3
"""Sample GPU and SGLang process resources during a profiling job."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


GPU_FIELDS = [
    "timestamp",
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "pcie.link.gen.current",
    "pcie.link.width.current",
]


def descendants(root_pid: int) -> set[int]:
    ppid_to_children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            closing = stat.rfind(")")
            fields = stat[closing + 2 :].split()
            ppid = int(fields[1])
            ppid_to_children.setdefault(ppid, []).append(int(entry.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    found = {root_pid}
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for child in ppid_to_children.get(parent, []):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def process_snapshot(root_pid: int) -> tuple[int, int, int, set[int]]:
    """Return CPU ticks, RSS KiB, process count, and PIDs for the tree."""
    pids = descendants(root_pid)
    if not pids:
        return 0, 0, 0, set()
    total_ticks = rss_kib = count = 0
    page_kib = os.sysconf("SC_PAGE_SIZE") // 1024
    for pid in pids:
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
            closing = stat.rfind(")")
            fields = stat[closing + 2 :].split()
            # /proc/PID/stat fields 14 and 15 are utime/stime.  The sliced
            # array starts at field 3, hence indices 11 and 12.
            total_ticks += int(fields[11]) + int(fields[12])
            statm = (Path("/proc") / str(pid) / "statm").read_text(
                encoding="utf-8"
            )
            rss_pages = int(statm.split()[1])
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
        count += 1
        rss_kib += rss_pages * page_kib
    return total_ticks, rss_kib, count, pids


def total_memory_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    raise RuntimeError("/proc/meminfo has no MemTotal")


def query_gpus() -> list[list[str]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(GPU_FIELDS),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [[field.strip() for field in line.split(",")] for line in proc.stdout.splitlines()]


def query_compute_apps() -> dict[str, list[dict[str, str]]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result: dict[str, list[dict[str, str]]] = {}
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            continue
        pid, uuid, process_name, used_memory = fields
        result.setdefault(uuid, []).append(
            {
                "pid": pid,
                "process_name": process_name,
                "used_memory_mib": used_memory,
            }
        )
    return result


def app_memory_mib(app: dict[str, str]) -> float:
    try:
        return float(app["used_memory_mib"])
    except (KeyError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_wall_ts",
        *GPU_FIELDS,
        "server_tree_cpu_pct",
        "server_tree_mem_pct",
        "server_tree_rss_kib",
        "server_tree_processes",
        "gpu_compute_process_count",
        "gpu_compute_memory_mib",
        "foreign_gpu_process_count",
        "foreign_gpu_processes",
    ]
    clock_ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
    host_memory_kib = total_memory_kib()
    previous_ticks: int | None = None
    previous_mono: float | None = None
    compute_apps: dict[str, list[dict[str, str]]] = {}
    last_compute_query_mono: float | None = None
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        while not args.stop_file.exists():
            sample_ts = time.time()
            sample_mono = time.monotonic()
            try:
                ticks, rss, count, server_pids = process_snapshot(args.server_pid)
                if previous_ticks is None or previous_mono is None:
                    cpu = None
                else:
                    elapsed_cpu_window = max(sample_mono - previous_mono, 1e-9)
                    cpu = (
                        max(0, ticks - previous_ticks)
                        / clock_ticks_per_second
                        / elapsed_cpu_window
                        * 100.0
                    )
                previous_ticks = ticks
                previous_mono = sample_mono
                mem = rss / max(host_memory_kib, 1) * 100.0
                if (
                    last_compute_query_mono is None
                    or sample_mono - last_compute_query_mono >= 1.0
                ):
                    compute_apps = query_compute_apps()
                    last_compute_query_mono = sample_mono
                for gpu_values in query_gpus():
                    gpu_record = dict(zip(GPU_FIELDS, gpu_values))
                    apps = compute_apps.get(gpu_record["uuid"], [])
                    foreign = [
                        app
                        for app in apps
                        if int(app["pid"]) not in server_pids
                    ]
                    row = {
                        "sample_wall_ts": f"{sample_ts:.6f}",
                        "server_tree_cpu_pct": (
                            "" if cpu is None else f"{cpu:.3f}"
                        ),
                        "server_tree_mem_pct": f"{mem:.3f}",
                        "server_tree_rss_kib": rss,
                        "server_tree_processes": count,
                        "gpu_compute_process_count": len(apps),
                        "gpu_compute_memory_mib": sum(
                            app_memory_mib(app) for app in apps
                        ),
                        "foreign_gpu_process_count": len(foreign),
                        "foreign_gpu_processes": "; ".join(
                            f"{app['pid']}:{app['process_name']}:"
                            f"{app['used_memory_mib']}MiB"
                            for app in foreign
                        ),
                    }
                    row.update(gpu_record)
                    writer.writerow(row)
                handle.flush()
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                print(f"resource sample failed at {sample_ts}: {exc!r}", flush=True)
            elapsed = time.time() - sample_ts
            time.sleep(max(0.01, args.interval - elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
