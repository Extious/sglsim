#!/usr/bin/env python3
"""Build a report and self-contained HTML dashboard for one PP=2 run."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import html
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Callable


GPU_COLORS = {"0": "#2563EB", "1": "#16A34A"}
EXPECTED_SGLANG_COMMIT = "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"
EXPECTED_MODEL_MARKER = "models--Qwen--Qwen3-8B"


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * pct) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "p50": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def number(value: Any) -> float | None:
    try:
        if value in (None, "", "[N/A]", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    converted = number(value)
    if converted is None:
        return "n/a"
    return f"{converted:.{digits}f}{suffix}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_markers(path: Path) -> dict[str, tuple[float, float]]:
    starts: dict[str, float] = {}
    intervals: dict[str, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        phase = str(item.get("phase"))
        if item.get("event") == "phase_start":
            starts[phase] = float(item["timestamp"])
        elif item.get("event") == "phase_end" and phase in starts:
            intervals[phase] = (starts[phase], float(item["timestamp"]))
    return intervals


def read_resources(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            parsed["sample_wall_ts"] = float(row["sample_wall_ts"])
            rows.append(parsed)
    return rows


def read_pcie(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(line for line in handle if line.strip())
        header: list[str] | None = None
        for raw in reader:
            values = [item.strip() for item in raw]
            if values[0].startswith("#Date"):
                header = [values[0].lstrip("#"), *values[1:]]
                continue
            if header is None or len(values) != len(header):
                continue
            item = dict(zip(header, values))
            timestamp = dt.datetime.strptime(
                f"{item['Date']} {item['Time']}", "%Y%m%d %H:%M:%S"
            ).timestamp()
            rows.append(
                {
                    "sample_wall_ts": timestamp,
                    "gpu": item["gpu"],
                    "rxpci_mb_s": number(item["rxpci"]),
                    "txpci_mb_s": number(item["txpci"]),
                }
            )
    return rows


def per_phase_resource_summary(
    resources: list[dict[str, Any]],
    pcie: list[dict[str, Any]],
    intervals: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = {
        "gpu_util_pct": "utilization.gpu",
        "memory_controller_util_pct": "utilization.memory",
        "memory_used_mib": "memory.used",
        "memory_total_mib": "memory.total",
        "power_w": "power.draw",
        "power_limit_w": "power.limit",
        "temperature_c": "temperature.gpu",
        "sm_clock_mhz": "clocks.current.sm",
        "memory_clock_mhz": "clocks.current.memory",
        "pcie_link_generation": "pcie.link.gen.current",
        "pcie_link_width": "pcie.link.width.current",
        "server_tree_cpu_pct": "server_tree_cpu_pct",
        "server_tree_rss_kib": "server_tree_rss_kib",
        "gpu_compute_process_count": "gpu_compute_process_count",
        "gpu_compute_memory_mib": "gpu_compute_memory_mib",
        "foreign_gpu_process_count": "foreign_gpu_process_count",
    }
    for phase, (start, end) in intervals.items():
        phase_rows = [
            row for row in resources if start <= row["sample_wall_ts"] <= end
        ]
        by_gpu: dict[str, Any] = {}
        for gpu in sorted({str(row["index"]) for row in phase_rows}):
            gpu_rows = [row for row in phase_rows if str(row["index"]) == gpu]
            by_gpu[gpu] = {
                key: summarize(
                    [
                        converted
                        for row in gpu_rows
                        if (converted := number(row.get(column))) is not None
                    ]
                )
                for key, column in metrics.items()
            }
        pcie_rows = [
            row for row in pcie if start <= row["sample_wall_ts"] <= end
        ]
        for gpu in sorted({str(row["gpu"]) for row in pcie_rows}):
            by_gpu.setdefault(gpu, {})
            gpu_pcie = [row for row in pcie_rows if str(row["gpu"]) == gpu]
            by_gpu[gpu]["pcie_rx_mb_s"] = summarize(
                [
                    float(row["rxpci_mb_s"])
                    for row in gpu_pcie
                    if row["rxpci_mb_s"] is not None
                ]
            )
            by_gpu[gpu]["pcie_tx_mb_s"] = summarize(
                [
                    float(row["txpci_mb_s"])
                    for row in gpu_pcie
                    if row["txpci_mb_s"] is not None
                ]
            )
        result[phase] = {
            "start_ts": start,
            "end_ts": end,
            "duration_s": end - start,
            "resource_samples": len(phase_rows),
            "pcie_samples": len(pcie_rows),
            "gpus": by_gpu,
        }
    return result


def stage_row(stage: dict[str, Any]) -> dict[str, Any]:
    aggregate = stage["aggregate"]

    def us_to_ms(value: Any) -> float | None:
        converted = number(value)
        return converted / 1000.0 if converted is not None else None

    return {
        "trace": stage["trace"],
        "stage": stage["wanted_stage"],
        "device": stage.get("device"),
        "steps": stage["selected_step_count"],
        "step_ms_p50": us_to_ms(aggregate["step_duration_us"]["median_us"]),
        "step_ms_p95": us_to_ms(aggregate["step_duration_us"]["p95_us"]),
        "kernel_busy_ms_p50": us_to_ms(
            aggregate["kernel_union_busy_us"]["median_us"]
        ),
        "kernel_occupancy_p50": aggregate["kernel_union_occupancy_pct"][
            "median_us"
        ],
        "kernel_occupancy_p95": aggregate["kernel_union_occupancy_pct"]["p95_us"],
        "idle_gap_us_p50": aggregate["observed_hardware_idle_gaps_us"][
            "median_us"
        ],
        "idle_gap_us_p95": aggregate["observed_hardware_idle_gaps_us"]["p95_us"],
        "nccl_events": aggregate["nccl_hardware_event_count_in_trace"],
        "memcpy": aggregate["memcpy"],
        "top_kernels": stage.get("top_kernels", []),
        "top_sglang_cpu_annotations": stage.get(
            "top_sglang_cpu_annotations", []
        ),
        "top_cpu_ops": stage.get("top_cpu_ops", []),
        "top_cuda_runtime_calls": stage.get("top_cuda_runtime_calls", []),
        "top_nccl_kernels": stage.get("top_nccl_kernels", []),
    }


def load_stage_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in (0, 1):
        payload = read_json(run_dir / f"stage_timeline_pp{rank}.json")
        for stage_name in ("prefill", "decode"):
            row = stage_row(payload[stage_name])
            row["pp_rank"] = rank
            rows.append(row)
    return rows


def relative_files(run_dir: Path, pattern: str) -> list[str]:
    return [
        str(path.relative_to(run_dir))
        for path in sorted(run_dir.rglob(pattern))
        if path.is_file()
    ]


def selected_raw_trace_files(run_dir: Path) -> list[str]:
    selected: list[str] = []
    directory = run_dir / "profile" / "traces" / "combined"
    for rank in (0, 1):
        paths = sorted(directory.glob(f"*PP-{rank}.trace.json.gz"))
        if len(paths) == 1:
            selected.append(str(paths[0].relative_to(run_dir)))
    return selected


def queue_summary(workload: dict[str, Any], key: str) -> dict[str, Any]:
    return workload[key]["workload"]["queue_time_s"]


def slight_queue_gate(
    queue: dict[str, Any], e2e: dict[str, Any]
) -> tuple[bool, str]:
    queue_mean = number(queue.get("mean"))
    queue_p95 = number(queue.get("p95"))
    e2e_p50 = number(e2e.get("p50"))
    if queue_mean is None or queue_p95 is None or e2e_p50 is None:
        return False, "scheduler queue/e2e metrics are missing"
    if queue.get("source") != "prometheus_histogram_delta":
        return False, "scheduler queue is not sourced from Prometheus deltas"
    if queue_mean < 0.002:
        return False, "mean scheduler queue is below 2 ms"
    if queue_p95 > max(0.050, 0.25 * e2e_p50):
        return False, "queue p95 exceeds the slight-queue bound"
    return True, "scheduler queue is visible and bounded"


def phase_load_gate(phase: dict[str, Any]) -> tuple[bool, str]:
    successes = int(phase.get("successes") or 0)
    if successes != int(phase.get("requests") or 0):
        return False, "phase contains failed requests"
    if int(phase.get("total_retractions") or 0) != 0:
        return False, "phase contains scheduler retractions"
    queue_count = phase.get("queue_time_s", {}).get("count")
    if queue_count is None or not successes <= int(queue_count) <= successes + 1:
        return False, "scheduler queue histogram count does not match requests"
    return slight_queue_gate(
        phase["queue_time_s"], phase["server_e2e_latency_s"]
    )


def profile_interval(
    intervals: dict[str, tuple[float, float]]
) -> tuple[float, float] | None:
    selected = [
        interval
        for phase, interval in intervals.items()
        if phase in {"steady_load", "profile_prefill", "profile_decode"}
    ]
    if not selected:
        return None
    return min(item[0] for item in selected), max(item[1] for item in selected)


def polyline_svg(
    rows: list[dict[str, Any]],
    *,
    intervals: dict[str, tuple[float, float]],
    value_fn: Callable[[dict[str, Any]], float | None],
    gpu_key: str,
    y_min: float,
    y_max: float,
    title: str,
    unit: str,
) -> str:
    interval = profile_interval(intervals)
    if interval is None:
        return "<p>No profile interval.</p>"
    start, end = interval
    selected = [row for row in rows if start <= row["sample_wall_ts"] <= end]
    width, height, left, right, top, bottom = 940, 245, 55, 20, 28, 35
    plot_w, plot_h = width - left - right, height - top - bottom

    def x_of(timestamp: float) -> float:
        return left + (timestamp - start) / max(end - start, 1e-9) * plot_w

    def y_of(value: float) -> float:
        clipped = min(y_max, max(y_min, value))
        return top + (y_max - clipped) / max(y_max - y_min, 1e-9) * plot_h

    lines = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#FFFFFF"/>',
        f'<text x="{left}" y="18" font-size="14" font-weight="700" fill="#0F172A">{html.escape(title)}</text>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = y_of(value)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#E2E8F0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
            f'fill="#64748B">{value:.0f}{html.escape(unit)}</text>'
        )
    phase_colors = {
        "steady_load": "#FEF3C7",
        "profile_prefill": "#DBEAFE",
        "profile_decode": "#DCFCE7",
    }
    for phase, (phase_start, phase_end) in intervals.items():
        if phase not in phase_colors:
            continue
        x1, x2 = x_of(phase_start), x_of(phase_end)
        lines.append(
            f'<rect x="{x1:.1f}" y="{top}" width="{max(1.0,x2-x1):.1f}" '
            f'height="{plot_h}" fill="{phase_colors[phase]}" opacity="0.42"/>'
        )
        lines.append(
            f'<text x="{(x1+x2)/2:.1f}" y="{height-10}" text-anchor="middle" '
            f'font-size="10" fill="#475569">{html.escape(phase.removeprefix("profile_"))}</text>'
        )
    for gpu in sorted({str(row[gpu_key]) for row in selected}):
        points: list[str] = []
        for row in selected:
            if str(row[gpu_key]) != gpu:
                continue
            value = value_fn(row)
            if value is None:
                continue
            points.append(f"{x_of(row['sample_wall_ts']):.1f},{y_of(value):.1f}")
        if points:
            color = GPU_COLORS.get(gpu, "#7C3AED")
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            lines.append(
                f'<text x="{width-right-75}" y="{18+int(gpu)*13}" font-size="10" '
                f'fill="{color}">GPU {html.escape(gpu)}</text>'
            )
    lines.append("</svg>")
    return "".join(lines)


def load_gate(workload: dict[str, Any]) -> tuple[bool, str]:
    selected = number(workload.get("selected_rate_rps"))
    if selected is None:
        return False, "selected offered rate is missing"
    candidates = [
        item
        for item in workload.get("rate_calibration", [])
        if abs(float(item["offered_rate_rps"]) - selected) < 1e-9
    ]
    if len(candidates) != 1:
        return False, "selected offered-rate point is missing or duplicated"
    point = candidates[0]
    point_ok, point_reason = phase_load_gate(point)
    if not point_ok:
        return False, f"selected offered-rate point: {point_reason}"
    throughputs = [
        number(item.get("token_throughput_tps"))
        for item in workload["rate_calibration"]
        if int(item.get("errors") or 0) == 0
    ]
    observed = [value for value in throughputs if value is not None]
    selected_throughput = number(point.get("token_throughput_tps"))
    if (
        selected_throughput is None
        or not observed
        or selected_throughput < 0.90 * max(observed)
    ):
        return False, "selected offered load is below 90% of calibrated peak throughput"
    steady_ok, steady_reason = phase_load_gate(workload["steady_load"])
    if not steady_ok:
        return False, f"formal steady load: {steady_reason}"
    return True, "throughput is near saturation and scheduler queue is visible but bounded"


def short_operation(value: str, limit: int = 72) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def top_operation(
    row: dict[str, Any], key: str, *, include_share: bool = False
) -> str:
    values = row.get(key) or []
    if not values:
        return "n/a"
    item = values[0]
    name = short_operation(str(item.get("name", "unknown")))
    if include_share:
        return f"{name} ({fmt(item.get('share_pct'),1,'%')})"
    return name


def memcpy_totals(row: dict[str, Any]) -> tuple[int, float]:
    values = row.get("memcpy") or {}
    return (
        sum(int(item.get("bytes") or 0) for item in values.values()),
        sum(float(item.get("duration_us") or 0.0) for item in values.values()),
    )


def analysis_findings(summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    load = summary["load_gate"]
    findings.append(f"Load gate {load['status']}: {load['reason']}.")
    formal = summary.get("formal_load_gates", {})
    if formal:
        findings.append(
            "Formal steady-load gate: "
            f"{formal['steady_load']['status']} "
            f"({formal['steady_load']['reason']})."
        )

    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in summary["stage_rows"]:
        by_stage.setdefault(str(row["stage"]), []).append(row)
    for stage_name in ("prefill", "decode"):
        rows = sorted(by_stage.get(stage_name, []), key=lambda row: row["pp_rank"])
        if len(rows) != 2:
            continue
        step_values = [number(row["step_ms_p50"]) for row in rows]
        occupancy = [number(row["kernel_occupancy_p50"]) for row in rows]
        if any(value is None for value in step_values + occupancy):
            continue
        numeric_steps = [float(value) for value in step_values if value is not None]
        numeric_occupancy = [
            float(value) for value in occupancy if value is not None
        ]
        balance_ratio = max(numeric_steps) / max(min(numeric_steps), 1e-9)
        nccl_events = sum(int(row["nccl_events"]) for row in rows)
        memcpy_bytes = sum(memcpy_totals(row)[0] for row in rows)
        findings.append(
            f"{stage_name}: PP0/PP1 median steps are "
            f"{numeric_steps[0]:.3f}/{numeric_steps[1]:.3f} ms "
            f"(slow/fast ratio {balance_ratio:.2f}x), kernel-union occupancy "
            f"{numeric_occupancy[0]:.1f}/{numeric_occupancy[1]:.1f}%, with "
            f"{nccl_events} NCCL hardware events and {memcpy_bytes} traced memcpy bytes."
        )

    for phase, label in (
        ("steady_load", "Unprofiled steady-load phase"),
        ("profile_prefill", "Profiled prefill phase"),
        ("profile_decode", "Profiled decode phase"),
    ):
        phase_data = summary["phase_resources"].get(phase, {})
        gpu_data = phase_data.get("gpus", {})
        if set(gpu_data) != {"0", "1"}:
            continue

        def gpu_metric(rank: int, metric_name: str, statistic: str) -> float | None:
            return number(
                gpu_data.get(str(rank), {})
                .get(metric_name, {})
                .get(statistic)
            )

        util = [
            gpu_metric(rank, "gpu_util_pct", "p50") for rank in (0, 1)
        ]
        mem_util = [
            gpu_metric(rank, "memory_controller_util_pct", "p50")
            for rank in (0, 1)
        ]
        hbm_used = [
            gpu_metric(rank, "memory_used_mib", "p50")
            for rank in (0, 1)
        ]
        hbm_total = [
            gpu_metric(rank, "memory_total_mib", "p50")
            for rank in (0, 1)
        ]
        power = [
            gpu_metric(rank, "power_w", "p50") for rank in (0, 1)
        ]
        pcie = [
            max(
                gpu_metric(rank, "pcie_rx_mb_s", "p95") or 0.0,
                gpu_metric(rank, "pcie_tx_mb_s", "p95") or 0.0,
            )
            for rank in (0, 1)
        ]
        cpu = gpu_metric(0, "server_tree_cpu_pct", "p50")
        foreign = [
            gpu_metric(rank, "foreign_gpu_process_count", "max")
            for rank in (0, 1)
        ]
        headroom = [
            (
                max(0.0, float(total) - float(used))
                if total is not None and used is not None
                else None
            )
            for total, used in zip(hbm_total, hbm_used)
        ]
        findings.append(
            f"{label}: GPU-util p50 {fmt(util[0],1,'%')}/"
            f"{fmt(util[1],1,'%')}, memory-controller p50 "
            f"{fmt(mem_util[0],1,'%')}/{fmt(mem_util[1],1,'%')}, "
            f"HBM headroom p50 {fmt(headroom[0],0,' MiB')}/"
            f"{fmt(headroom[1],0,' MiB')}, power p50 "
            f"{fmt(power[0],1,' W')}/{fmt(power[1],1,' W')}, "
            f"PCIe max-direction p95 {fmt(pcie[0],1,' MB/s')}/"
            f"{fmt(pcie[1],1,' MB/s')}, process-tree CPU p50 "
            f"{fmt(cpu,1,'%')}, foreign GPU processes max "
            f"{fmt(foreign[0],0)}/{fmt(foreign[1],0)}."
        )
    findings.append(
        "Kernel-union gaps and low GPU-util samples are attribution evidence, "
        "not proof of free HBM/copy-engine/PCIe bandwidth; backup credits still "
        "require path telemetry and foreground SLO guards."
    )
    return findings


def evidence_issues(summary: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    failed_contract = [
        item["name"]
        for item in summary["deployment_contract"]["checks"]
        if not item["passed"]
    ]
    if failed_contract:
        issues.append(
            "deployment contract failed: " + ", ".join(failed_contract)
        )
    if len(summary["selected_raw_trace_files"]) != 2:
        issues.append(
            "need exactly two selected raw traces, one automatic combined "
            "prefill/decode interval per PP rank"
        )
    if len(summary["browser_trace_files"]) != 1:
        issues.append("need exactly one combined browser trace")
    manifest = summary.get("browser_trace_manifest") or {}
    if manifest.get("source_trace_count") != 2:
        issues.append(
            "combined browser trace does not prove both PP-rank sources"
        )
    source_traces = manifest.get("source_traces") or []
    namespace_ranges: dict[int, tuple[int, int]] = {}
    for rank in (0, 1):
        matching_sources = [
            item
            for item in source_traces
            if str(item.get("stage")) == "combined"
            and int(item.get("pp_rank", -1)) == rank
        ]
        ranges = {
            (
                int(item.get("pid_namespace_min", -1)),
                int(item.get("pid_namespace_max", -1)),
            )
            for item in matching_sources
        }
        if len(matching_sources) == 1 and len(ranges) == 1:
            namespace_ranges[rank] = next(iter(ranges))
    ordered_ranges = sorted(namespace_ranges.values())
    if (
        set(namespace_ranges) != {0, 1}
        or any(
            previous[1] >= following[0]
            for previous, following in zip(ordered_ranges, ordered_ranges[1:])
        )
    ):
        issues.append(
            "combined browser trace lacks two disjoint PP-rank PID namespaces"
        )
    if {
        str(item.get("flow_id_namespace", "")) for item in source_traces
    } != {"combined:pp0", "combined:pp1"}:
        issues.append("combined browser trace lacks per-source flow-ID namespaces")
    if int(manifest.get("resource_event_count") or 0) <= 0:
        issues.append("combined browser trace has no whole-device resource counters")
    if len(summary["memory_html_files"]) < 4:
        issues.append("need at least four CUDA allocator browser HTML files")
    if len(summary["triage_files"]) != 2:
        issues.append("need exactly two combined PP-rank triage reports")
    if len(summary["stage_rows"]) != 4 or any(
        int(row.get("steps") or 0) <= 0 for row in summary["stage_rows"]
    ):
        issues.append("stage analysis must contain non-empty prefill/decode rows for PP0/PP1")
    for key in ("prefill_profile", "decode_profile"):
        phase_workload = summary["workload"][key]["workload"]
        if int(phase_workload.get("successes") or 0) != int(
            phase_workload.get("requests") or 0
        ):
            issues.append(f"{key} contains failed requests")
        if int(phase_workload.get("total_retractions") or 0) != 0:
            issues.append(f"{key} contains scheduler retractions")
    steady_ok, steady_reason = phase_load_gate(summary["workload"]["steady_load"])
    if not steady_ok:
        issues.append(f"steady_load gate failed: {steady_reason}")
    for phase in ("steady_load", "profile_prefill", "profile_decode"):
        phase_data = summary["phase_resources"].get(phase)
        if not phase_data:
            issues.append(f"{phase} resource interval is missing")
            continue
        if set(phase_data.get("gpus", {})) != {"0", "1"}:
            issues.append(f"{phase} must contain telemetry for both GPUs")
        if int(phase_data.get("resource_samples") or 0) < 2:
            issues.append(f"{phase} has too few 200-ms GPU resource samples")
        if int(phase_data.get("pcie_samples") or 0) < 2:
            issues.append(f"{phase} has too few 1-s PCIe samples")
        foreign_metric_counts = [
            int(values.get("foreign_gpu_process_count", {}).get("count") or 0)
            for values in phase_data.get("gpus", {}).values()
        ]
        if (
            len(foreign_metric_counts) != 2
            or min(foreign_metric_counts) < 2
        ):
            issues.append(f"{phase} lacks foreign-GPU-process telemetry")
        foreign_max = max(
            number(values.get("foreign_gpu_process_count", {}).get("max"))
            or 0.0
            for values in phase_data.get("gpus", {}).values()
        )
        if foreign_max > 0:
            issues.append(f"{phase} contains a foreign GPU process")
    load_ok, load_reason = load_gate(summary["workload"])
    if not load_ok:
        issues.append(f"load gate failed: {load_reason}")
    for key in ("job_id", "node", "partition", "sglang_commit"):
        if str(summary.get(key, "")).strip().lower() in {"", "unknown"}:
            issues.append(f"run identity field {key} is missing")
    return issues


def render_report(summary: dict[str, Any]) -> str:
    workload = summary["workload"]
    steady_queue = workload["steady_load"]["queue_time_s"]
    prefill_queue = queue_summary(workload, "prefill_profile")
    decode_queue = queue_summary(workload, "decode_profile")
    lines = [
        "# Qwen3-8B PP=2 SGLang profiler report",
        "",
        "## Run identity",
        "",
        f"- Slurm job: `{summary['job_id']}`",
        f"- Node: `{summary['node']}`",
        f"- Partition: `{summary['partition']}`",
        f"- SGLang profiler commit: `{summary['sglang_commit']}`",
        "- Topology: one node, two RTX 4090, TP=1, PP=2, Qwen3-8B.",
        "- Observability mode: CUDA graph disabled so CPU call stacks and individual kernels remain attributable.",
        "",
        "## Deployment contract",
        "",
        "| Check | Observed | Expected | Status |",
        "| --- | --- | --- | ---: |",
    ]
    for item in summary["deployment_contract"]["checks"]:
        observed = short_operation(str(item["observed"]), 90)
        expected = short_operation(str(item["expected"]), 90)
        lines.append(
            f"| {item['name']} | `{observed}` | `{expected}` | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
        "",
        "## Load point",
        "",
        f"- Selected steady offered rate: **{fmt(workload['selected_rate_rps'],3,' req/s')}** "
        f"(`{workload['selection_reason']}`).",
        f"- Profiler request concurrency: prefill **{workload['selected_concurrency']}**, "
        f"decode **{workload['decode_concurrency']}**.",
        f"- Load gate: **{summary['load_gate']['status']}** "
        f"({summary['load_gate']['reason']}).",
        f"- Formal steady-load gate: **{summary['formal_load_gates']['steady_load']['status']}** "
        f"({summary['formal_load_gates']['steady_load']['reason']}).",
        f"- Steady scheduler queue mean / p95 upper bound: "
        f"**{fmt(steady_queue['mean'],3,' s')} / {fmt(steady_queue['p95'],3,' s')}**.",
        f"- Profiled prefill queue mean / p95 upper bound: "
        f"**{fmt(prefill_queue['mean'],3,' s')} / {fmt(prefill_queue['p95'],3,' s')}**.",
        f"- Profiled decode queue mean / p95 upper bound: "
        f"**{fmt(decode_queue['mean'],3,' s')} / {fmt(decode_queue['p95'],3,' s')}**.",
        "- The steady phase is the normal-path utilization result; profiled-phase "
        "queue values include torch.profiler overhead and are reported as diagnostics.",
        "",
        "## Resource interpretation",
        "",
        *[f"- {finding}" for finding in summary["analysis_findings"]],
        "",
        "## PP-rank torch.profiler timeline",
        "",
        "| Stage | PP rank | steps | step p50 | kernel union p50 | occupancy p50 | gap p95 | NCCL events |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["stage_rows"]:
        lines.append(
            f"| {row['stage']} | {row['pp_rank']} | {row['steps']} | "
            f"{fmt(row['step_ms_p50'],3,' ms')} | "
            f"{fmt(row['kernel_busy_ms_p50'],3,' ms')} | "
            f"{fmt(row['kernel_occupancy_p50'],2,'%')} | "
            f"{fmt(row['idle_gap_us_p95'],2,' us')} | {row['nccl_events']} |"
        )
    lines.extend(
        [
            "",
            "## Operation hot spots",
            "",
            "| Stage | PP rank | top GPU kernel | top SGLang CPU annotation | top ATen op | memcpy | NCCL |",
            "| --- | ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in summary["stage_rows"]:
        memcpy_bytes, memcpy_duration_us = memcpy_totals(row)
        lines.append(
            f"| {row['stage']} | {row['pp_rank']} | "
            f"{top_operation(row,'top_kernels',include_share=True)} | "
            f"{top_operation(row,'top_sglang_cpu_annotations')} | "
            f"{top_operation(row,'top_cpu_ops')} | "
            f"{memcpy_bytes} B / {fmt(memcpy_duration_us,1,' us')} | "
            f"{row['nccl_events']} |"
        )
    lines.extend(
        [
            "",
            "## Whole-device resource telemetry",
            "",
            "| Phase | GPU | GPU util p50/p95 | HBM used p50 | memory-controller p50 | power p50 | PCIe RX p95 | PCIe TX p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase in ("steady_load", "profile_prefill", "profile_decode"):
        phase_data = summary["phase_resources"].get(phase, {})
        for gpu, values in phase_data.get("gpus", {}).items():
            lines.append(
                f"| {phase} | {gpu} | "
                f"{fmt(values.get('gpu_util_pct',{}).get('p50'),1,'%')} / "
                f"{fmt(values.get('gpu_util_pct',{}).get('p95'),1,'%')} | "
                f"{fmt(values.get('memory_used_mib',{}).get('p50'),0,' MiB')} | "
                f"{fmt(values.get('memory_controller_util_pct',{}).get('p50'),1,'%')} | "
                f"{fmt(values.get('power_w',{}).get('p50'),1,' W')} | "
                f"{fmt(values.get('pcie_rx_mb_s',{}).get('p95'),1,' MB/s')} | "
                f"{fmt(values.get('pcie_tx_mb_s',{}).get('p95'),1,' MB/s')} |"
            )
    lines.extend(
        [
            "",
            "## Browser artifacts",
            "",
            "- Open `index.html` directly for the telemetry dashboard and artifact index.",
            "- Import `browser_trace/pp2-prefill-decode-resources.trace.json.gz` once into Perfetto UI or Chrome tracing. It combines both PP ranks, the selected chunked-prefill/decode intervals, and whole-device GPU/CPU/PCIe counters.",
            "- The individual raw `*.trace.json.gz` files remain the authoritative unmerged torch.profiler traces.",
            "- Open each `*-memory-timeline.html` or `*-allocator-history.html` directly for CUDA allocator history.",
            "",
            "## Evidence gate",
            "",
            f"- Status: **{summary['evidence_grade']}**.",
            *(
                [f"- Missing/failed: {issue}" for issue in summary["evidence_issues"]]
                if summary["evidence_issues"]
                else ["- All declared trace, load, telemetry, allocator, and identity checks passed."]
            ),
            "",
            "## Interpretation boundary",
            "",
            "A torch trace proves events issued by the profiled SGLang processes. An empty trace interval does not prove spare HBM, copy-engine, PCIe, or network bandwidth. The 1-second PCIe dmon samples and 200-ms GPU telemetry are whole-device observations, while the trace gives per-operation attribution. Backup admission should use both and retain foreground SLO feedback.",
            "",
        ]
    )
    return "\n".join(lines)


def html_table_stage(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['stage'])}</td><td>{row['pp_rank']}</td>"
            f"<td>{row['steps']}</td><td>{fmt(row['step_ms_p50'],3,' ms')}</td>"
            f"<td>{fmt(row['kernel_occupancy_p50'],1,'%')}</td>"
            f"<td>{fmt(row['idle_gap_us_p95'],1,' us')}</td>"
            f"<td>{row['nccl_events']}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Stage</th><th>PP rank</th><th>Steps</th>"
        "<th>Step p50</th><th>Kernel union</th><th>Gap p95</th>"
        "<th>NCCL events</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def html_table_hotspots(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        memcpy_bytes, memcpy_duration_us = memcpy_totals(row)
        body.append(
            "<tr>"
            f"<td>{html.escape(row['stage'])}</td><td>{row['pp_rank']}</td>"
            f"<td>{html.escape(top_operation(row,'top_kernels',include_share=True))}</td>"
            f"<td>{html.escape(top_operation(row,'top_sglang_cpu_annotations'))}</td>"
            f"<td>{html.escape(top_operation(row,'top_cpu_ops'))}</td>"
            f"<td>{memcpy_bytes} B / {fmt(memcpy_duration_us,1,' us')}</td>"
            f"<td>{row['nccl_events']}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Stage</th><th>PP rank</th><th>Top GPU kernel</th>"
        "<th>Top SGLang CPU annotation</th><th>Top ATen op</th><th>Memcpy</th>"
        "<th>NCCL</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def html_table_contract(contract: dict[str, Any]) -> str:
    rows = []
    for item in contract["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['name']))}</td>"
            f"<td><code>{html.escape(short_operation(str(item['observed']),90))}</code></td>"
            f"<td><code>{html.escape(short_operation(str(item['expected']),90))}</code></td>"
            f"<td>{status}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Check</th><th>Observed</th><th>Expected</th>"
        "<th>Status</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def artifact_links(paths: list[str]) -> str:
    return "".join(
        f'<li><a href="{html.escape(path)}">{html.escape(path)}</a></li>'
        for path in paths
    )


def render_html(
    summary: dict[str, Any],
    resources: list[dict[str, Any]],
    pcie: list[dict[str, Any]],
    intervals: dict[str, tuple[float, float]],
) -> str:
    workload = summary["workload"]
    util_svg = polyline_svg(
        resources,
        intervals=intervals,
        value_fn=lambda row: number(row.get("utilization.gpu")),
        gpu_key="index",
        y_min=0,
        y_max=100,
        title="GPU utilization: steady normal path and profiled phases",
        unit="%",
    )
    mem_max = max(
        [
            number(row.get("memory.total")) or 0.0
            for row in resources
        ]
        or [24_576.0]
    )
    memory_svg = polyline_svg(
        resources,
        intervals=intervals,
        value_fn=lambda row: number(row.get("memory.used")),
        gpu_key="index",
        y_min=0,
        y_max=max(mem_max, 1.0),
        title="HBM allocation footprint",
        unit=" MiB",
    )
    pcie_max = max(
        [
            max(number(row.get("rxpci_mb_s")) or 0, number(row.get("txpci_mb_s")) or 0)
            for row in pcie
        ]
        or [1.0]
    )
    pcie_svg = polyline_svg(
        pcie,
        intervals=intervals,
        value_fn=lambda row: (
            max(number(row.get("rxpci_mb_s")) or 0, number(row.get("txpci_mb_s")) or 0)
        ),
        gpu_key="gpu",
        y_min=0,
        y_max=max(1.0, pcie_max * 1.05),
        title="PCIe throughput, max(RX, TX) per GPU",
        unit=" MB/s",
    )
    steady_queue = workload["steady_load"]["queue_time_s"]
    steady = workload["steady_load"]
    trace_links = artifact_links(summary["trace_files"])
    memory_links = artifact_links(summary["memory_html_files"])
    triage_links = artifact_links(summary["triage_files"])
    browser_trace_links = artifact_links(summary["browser_trace_files"])
    issue_html = (
        "<ul>"
        + "".join(
            f"<li>{html.escape(issue)}</li>" for issue in summary["evidence_issues"]
        )
        + "</ul>"
        if summary["evidence_issues"]
        else "<p>All declared trace, load, telemetry, allocator, and identity checks passed.</p>"
    )
    findings_html = "".join(
        f"<li>{html.escape(finding)}</li>"
        for finding in summary["analysis_findings"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AegisKV Qwen3-8B PP=2 profiler</title>
<style>
:root {{ color-scheme: light; --ink:#0f172a; --muted:#475569; --line:#cbd5e1; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,sans-serif; color:var(--ink); background:#f1f5f9; }}
main {{ max-width:1180px; margin:auto; padding:32px 24px 64px; }}
h1 {{ margin:0 0 8px; font-size:30px; }}
h2 {{ margin-top:32px; font-size:21px; }}
p,li {{ line-height:1.55; color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
.card {{ background:white; border:1px solid var(--line); border-radius:14px; padding:16px; }}
.label {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#7c3aed; font-weight:800; }}
.value {{ font-size:24px; font-weight:850; margin-top:5px; }}
.charts {{ display:grid; grid-template-columns:1fr; gap:16px; }}
svg {{ width:100%; height:auto; border:1px solid var(--line); border-radius:14px; }}
table {{ width:100%; border-collapse:collapse; background:white; border-radius:12px; overflow:hidden; }}
th,td {{ padding:10px 12px; border-bottom:1px solid #e2e8f0; text-align:left; font-size:13px; }}
th {{ background:#e2e8f0; }}
.artifact-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
.note {{ border-left:4px solid #a16207; background:#fef3c7; padding:14px 16px; border-radius:8px; }}
code {{ font-family:ui-monospace,SFMono-Regular,monospace; }}
@media(max-width:800px) {{ .cards,.artifact-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>Qwen3-8B · PP=2 · 2×RTX 4090 profiler</h1>
<p>Slurm <code>{html.escape(summary['job_id'])}</code> on
<code>{html.escape(summary['node'])}</code>, SGLang profiler commit
<code>{html.escape(summary['sglang_commit'][:12])}</code>. CUDA graphs are disabled
for complete function and kernel attribution.</p>
<div class="cards">
  <div class="card"><div class="label">steady offered load</div><div class="value">{fmt(workload['selected_rate_rps'],3,' req/s')}</div><p>{html.escape(workload['selection_reason'])}</p></div>
  <div class="card"><div class="label">steady throughput</div><div class="value">{fmt(steady['request_throughput_rps'],3,' req/s')}</div><p>unprofiled normal path</p></div>
  <div class="card"><div class="label">queue mean / p95 bound</div><div class="value">{fmt(steady_queue['mean'],3,' s')} / {fmt(steady_queue['p95'],3,' s')}</div><p>Prometheus histogram delta</p></div>
  <div class="card"><div class="label">evidence gate</div><div class="value">{html.escape(summary['evidence_grade'])}</div><p>trace + allocator + telemetry + analysis</p></div>
</div>
<h2>Deployment contract: {html.escape(summary['deployment_contract']['status'])}</h2>
{html_table_contract(summary['deployment_contract'])}
<h2>Resource interpretation</h2>
<ul>{findings_html}</ul>
<h2>Resource utilization timeline</h2>
<div class="charts">{util_svg}{memory_svg}{pcie_svg}</div>
<h2>torch.profiler stage summary</h2>
{html_table_stage(summary['stage_rows'])}
<h2>Operation hot spots</h2>
{html_table_hotspots(summary['stage_rows'])}
<h2>Browser artifacts</h2>
<div class="artifact-grid">
  <div class="card"><div class="label">One-file PP2 browser trace</div><p>Import once into <a href="https://ui.perfetto.dev">Perfetto UI</a> to see both PP ranks, prefill/decode, and resource counters.</p><ul>{browser_trace_links}</ul></div>
  <div class="card"><div class="label">CUDA allocator HTML</div><p>Open directly in this browser.</p><ul>{memory_links}</ul></div>
  <div class="card"><div class="label">Kernel/function triage</div><p>Top kernels, Python scopes, and overlap candidates.</p><ul>{triage_links}</ul></div>
</div>
<details><summary>Individual authoritative torch.profiler traces</summary><ul>{trace_links}</ul></details>
<h2>Evidence gate: {html.escape(summary['evidence_grade'])}</h2>
{issue_html}
<h2>How to read the two levels</h2>
<p class="note"><strong>torch.profiler</strong> attributes CPU calls, CUDA runtime,
kernels, memcpy, and NCCL to each PP rank. Whole-device telemetry supplies GPU
utilization, HBM footprint, power, and PCIe throughput. Trace gaps alone are not
evidence of free copy-engine, HBM, PCIe, or network bandwidth.</p>
</main></body></html>"""


def parse_environment(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) >= 2:
        result["node"] = lines[1].strip()
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def deployment_contract(
    environment_path: Path,
    environment: dict[str, str],
    server_info: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    raw_environment = environment_path.read_text(
        encoding="utf-8", errors="replace"
    )
    gpu_rows = re.findall(
        r"^GPU\s+\d+:\s+(.+?)\s+\(UUID:\s*([^)]+)\)",
        raw_environment,
        flags=re.MULTILINE,
    )
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, expected: Any, passed: bool) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
            }
        )

    gpu_names = [name for name, _uuid in gpu_rows]
    gpu_uuids = [uuid for _name, uuid in gpu_rows]
    check("gpu_count", len(gpu_rows), 2, len(gpu_rows) == 2)
    check(
        "gpu_model",
        gpu_names,
        "two RTX 4090",
        len(gpu_names) == 2 and all("RTX 4090" in name for name in gpu_names),
    )
    check(
        "gpu_uuid_unique",
        gpu_uuids,
        "two unique UUIDs",
        len(gpu_uuids) == 2 and len(set(gpu_uuids)) == 2,
    )
    check(
        "sglang_commit",
        environment.get("sglang_commit"),
        EXPECTED_SGLANG_COMMIT,
        environment.get("sglang_commit") == EXPECTED_SGLANG_COMMIT,
    )
    check(
        "sglang_profile_v2",
        environment.get("sglang_profile_v2"),
        "0 (legacy start_step/num_steps path)",
        environment.get("sglang_profile_v2") == "0",
    )

    expected_server_values = {
        "tp_size": 1,
        "pp_size": 2,
        "context_length": 16384,
        "chunked_prefill_size": 1024,
        "max_running_requests": 32,
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "enable_metrics": True,
        "enable_request_time_stats_logging": True,
        "attention_backend": "triton",
        "sampling_backend": "pytorch",
    }
    for key, expected in expected_server_values.items():
        observed = server_info.get(key)
        check(key, observed, expected, observed == expected)
    observed_fraction = number(server_info.get("mem_fraction_static"))
    check(
        "mem_fraction_static",
        observed_fraction,
        0.80,
        observed_fraction is not None and abs(observed_fraction - 0.80) < 1e-9,
    )
    model_path = str(server_info.get("model_path", ""))
    check(
        "server_model",
        model_path,
        EXPECTED_MODEL_MARKER,
        EXPECTED_MODEL_MARKER in model_path,
    )
    model_entries = models.get("data") if isinstance(models, dict) else None
    advertised_models = [
        str(item.get("id") or item.get("root") or "")
        for item in (model_entries or [])
        if isinstance(item, dict)
    ]
    check(
        "advertised_model",
        advertised_models,
        EXPECTED_MODEL_MARKER,
        len(advertised_models) == 1
        and EXPECTED_MODEL_MARKER in advertised_models[0],
    )
    return {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
        "gpu_names": gpu_names,
        "gpu_uuids": gpu_uuids,
    }


def artifact_contract(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, expected: Any, passed: bool) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
            }
        )

    check(
        "evidence_gate",
        summary.get("evidence_grade"),
        "complete",
        summary.get("evidence_grade") == "complete",
    )
    expected_counts = {
        "selected_raw_traces": (len(summary["selected_raw_trace_files"]), 2),
        "combined_browser_traces": (len(summary["browser_trace_files"]), 1),
        "stage_rows": (len(summary["stage_rows"]), 4),
        "stage_json_files": (len(relative_files(run_dir, "stage_timeline_pp*.json")), 2),
        "triage_markdown_files": (len(summary["triage_files"]), 2),
    }
    for name, (observed, expected) in expected_counts.items():
        check(name, observed, expected, observed == expected)

    memory_html_count = len(summary["memory_html_files"])
    check(
        "memory_html_files",
        memory_html_count,
        "at least four",
        memory_html_count >= 4,
    )
    required_paths = [
        "environment.txt",
        "gpu_processes_before_launch.txt",
        "live_server_info.json",
        "live_models.json",
        "metrics_after.prom",
        "resource_samples.csv",
        "pcie_throughput_dmon.csv",
        "profile/workload_summary.json",
        "profile/phase_markers.jsonl",
        "browser_trace/browser_trace_manifest.json",
        "browser_trace/pp2-prefill-decode-resources.trace.json.gz",
        "analysis_summary.json",
        "REPORT.md",
        "index.html",
    ]
    missing = [
        relative
        for relative in required_paths
        if not (run_dir / relative).is_file()
        or (run_dir / relative).stat().st_size == 0
    ]
    check("required_nonempty_files", missing, "none missing", not missing)
    preflight_path = run_dir / "gpu_processes_before_launch.txt"
    preflight_text = (
        preflight_path.read_text(encoding="utf-8", errors="replace").strip()
        if preflight_path.is_file()
        else "missing"
    )
    check(
        "gpu_preflight_empty",
        preflight_text,
        "status=empty",
        preflight_text == "status=empty",
    )
    listed_artifacts = (
        summary["selected_raw_trace_files"]
        + summary["browser_trace_files"]
        + summary["triage_files"]
        + summary["memory_html_files"]
        + relative_files(run_dir, "stage_timeline_pp*.json")
    )
    missing_listed_artifacts = [
        relative
        for relative in listed_artifacts
        if not (run_dir / relative).is_file()
        or (run_dir / relative).stat().st_size == 0
    ]
    check(
        "listed_artifacts_nonempty",
        missing_listed_artifacts,
        "all listed artifacts exist and are non-empty",
        not missing_listed_artifacts,
    )

    undersized_traces = [
        path
        for path in summary["selected_raw_trace_files"]
        + summary["browser_trace_files"]
        if (run_dir / path).stat().st_size < 64 * 1024
    ]
    check(
        "trace_minimum_size",
        undersized_traces,
        "all selected/combined traces >= 64 KiB",
        not undersized_traces,
    )
    trace_headers_missing = []
    for relative in summary["selected_raw_trace_files"] + summary["browser_trace_files"]:
        with gzip.open(run_dir / relative, "rb") as handle:
            header = handle.read(512 * 1024)
        if b"traceEvents" not in header:
            trace_headers_missing.append(relative)
    check(
        "trace_json_headers",
        trace_headers_missing,
        "all gzip traces expose traceEvents",
        not trace_headers_missing,
    )

    workload = summary["workload"]
    workload_observed = {
        "chunked_prefill_size": workload.get("chunked_prefill_size"),
        "capture_mode": workload["prefill_profile"].get("capture_mode"),
        "prefill_input_len": workload["prefill_profile"]["workload"].get("input_len"),
        "decode_input_len": workload["decode_profile"]["workload"].get("input_len"),
        "decode_output_len": workload["decode_profile"]["workload"].get("output_len"),
    }
    workload_expected = {
        "chunked_prefill_size": 1024,
        "capture_mode": "single_arm_forward_count_automatic",
        "prefill_input_len": 4096,
        "decode_input_len": 128,
        "decode_output_len": 128,
    }
    check(
        "workload_shape",
        workload_observed,
        workload_expected,
        workload_observed == workload_expected,
    )

    index_text = (run_dir / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )
    index_markers = [
        "<!doctype html>",
        "One-file PP2 browser trace",
        "Resource utilization timeline",
        "Operation hot spots",
        "Deployment contract",
    ]
    missing_index_markers = [
        marker for marker in index_markers if marker not in index_text
    ]
    check(
        "dashboard_content",
        missing_index_markers,
        "all required dashboard sections",
        not missing_index_markers,
    )
    return {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    workload = read_json(run_dir / "profile" / "workload_summary.json")
    intervals = read_markers(run_dir / "profile" / "phase_markers.jsonl")
    resources = read_resources(run_dir / "resource_samples.csv")
    pcie = read_pcie(run_dir / "pcie_throughput_dmon.csv")
    environment_path = run_dir / "environment.txt"
    environment = parse_environment(environment_path)
    server_info = read_json(run_dir / "live_server_info.json")
    models = read_json(run_dir / "live_models.json")
    stage_rows = load_stage_rows(run_dir)
    browser_manifest_path = (
        run_dir / "browser_trace" / "browser_trace_manifest.json"
    )
    browser_manifest = (
        read_json(browser_manifest_path) if browser_manifest_path.exists() else {}
    )
    summary: dict[str, Any] = {
        "job_id": environment.get("slurm_job_id", run_dir.name),
        "node": environment.get("node", "unknown"),
        "partition": environment.get("slurm_partition", "unknown"),
        "sglang_commit": environment.get("sglang_commit", "unknown"),
        "deployment_contract": deployment_contract(
            environment_path, environment, server_info, models
        ),
        "workload": workload,
        "intervals": intervals,
        "phase_resources": per_phase_resource_summary(resources, pcie, intervals),
        "stage_rows": stage_rows,
        "trace_files": relative_files(run_dir, "*.trace.json.gz"),
        "selected_raw_trace_files": selected_raw_trace_files(run_dir),
        "browser_trace_files": relative_files(
            run_dir, "pp2-prefill-decode-resources.trace.json.gz"
        ),
        "browser_trace_manifest": browser_manifest,
        "memory_html_files": relative_files(run_dir, "*-memory-timeline.html")
        + relative_files(run_dir, "*-allocator-history.html"),
        "triage_files": relative_files(run_dir, "triage_*.md"),
    }
    load_ok, load_reason = load_gate(workload)
    summary["load_gate"] = {
        "status": "pass" if load_ok else "fail",
        "reason": load_reason,
    }
    phase_ok, phase_reason = phase_load_gate(workload["steady_load"])
    summary["formal_load_gates"] = {
        "steady_load": {
            "status": "pass" if phase_ok else "fail",
            "reason": phase_reason,
        }
    }
    summary["analysis_findings"] = analysis_findings(summary)
    summary["evidence_issues"] = evidence_issues(summary)
    summary["evidence_grade"] = (
        "complete" if not summary["evidence_issues"] else "incomplete"
    )
    summary_path = run_dir / "analysis_summary.json"
    report_path = run_dir / "REPORT.md"
    index_path = run_dir / "index.html"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(summary), encoding="utf-8")
    index_path.write_text(
        render_html(summary, resources, pcie, intervals),
        encoding="utf-8",
    )
    summary["artifact_contract"] = artifact_contract(run_dir, summary)
    if summary["artifact_contract"]["status"] != "pass":
        summary["evidence_grade"] = "incomplete"
        summary["evidence_issues"].append("final artifact contract failed")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(summary), encoding="utf-8")
    index_path.write_text(
        render_html(summary, resources, pcie, intervals),
        encoding="utf-8",
    )
    (run_dir / "FINAL_VALIDATION.json").write_text(
        json.dumps(summary["artifact_contract"], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(report_path)
    print(index_path)
    if summary["artifact_contract"]["status"] != "pass":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
