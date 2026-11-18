#!/usr/bin/env python3
"""Calibrate and capture a saturated Qwen3-8B PP=2 SGLang workload.

The script intentionally uses only the Python standard library so it can run in
the same environment as the SGLang server without installing client packages.
It records SGLang's scheduler-side queue_time (enabled by --enable-metrics), the
client wall-clock latency, and explicit phase markers used by the resource
telemetry analyzer.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * pct) - 1))
    return ordered[index]


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        status = int(response.status)
    if not raw:
        return {"http_status": status}
    text = raw.decode("utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # The profiler branch's /start_profile endpoint deliberately returns
        # text/plain ("Start profiling.") while /generate returns JSON.
        return {"http_status": status, "body": text.strip()}
    if not isinstance(parsed, dict):
        raise TypeError(f"expected a JSON object from {url}, got {type(parsed).__name__}")
    return parsed


def get_json(url: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def get_text(url: str, timeout: float) -> str:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")


def get_status(url: str, timeout: float) -> int:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        response.read()
        return int(response.status)


async def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status = await asyncio.to_thread(
                get_status, base_url.rstrip("/") + "/health", 10.0
            )
            if status == 200:
                return
        except Exception as exc:  # pragma: no cover - live-node path
            last_error = exc
            await asyncio.sleep(2.0)
    raise TimeoutError(f"server not healthy within {timeout_s}s: {last_error!r}")


QUEUE_METRIC_RE = re.compile(
    r"^sglang:queue_time_seconds_(sum|count|bucket)"
    r"\{(?P<labels>[^}]*)\}\s+(?P<value>[-+0-9.eE]+)$"
)
PROM_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


def parse_queue_histogram(text: str) -> dict[str, dict[str, Any]]:
    """Parse the scheduler's authoritative queue histogram by PP rank."""
    result: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        match = QUEUE_METRIC_RE.match(line.strip())
        if not match:
            continue
        kind = match.group(1)
        labels = dict(PROM_LABEL_RE.findall(match.group("labels")))
        rank = labels.get("pp_rank")
        if rank not in {"0", "1"}:
            continue
        value = float(match.group("value"))
        rank_data = result.setdefault(
            rank, {"sum": 0.0, "count": 0.0, "buckets": {}}
        )
        if kind == "bucket":
            le_raw = labels.get("le", "+Inf")
            le = math.inf if le_raw == "+Inf" else float(le_raw)
            rank_data["buckets"][le] = value
        else:
            rank_data[kind] = value
    return result


async def queue_histogram_snapshot(base_url: str) -> dict[str, dict[str, Any]]:
    text = await asyncio.to_thread(
        get_text, base_url.rstrip("/") + "/metrics", 30.0
    )
    snapshot = parse_queue_histogram(text)
    if set(snapshot) != {"0", "1"}:
        raise RuntimeError(
            f"expected queue histograms for PP ranks 0 and 1, got {sorted(snapshot)}"
        )
    return snapshot


def histogram_quantile_upper_bound(
    buckets: dict[float, float], count: float, quantile: float
) -> float | None:
    if count <= 0:
        return None
    target = quantile * count
    for upper, observed in sorted(buckets.items()):
        if observed >= target:
            return None if math.isinf(upper) else upper
    return None


def queue_histogram_delta(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return exact mean and conservative histogram-quantile upper bounds."""
    per_rank: dict[str, dict[str, Any]] = {}
    for rank in ("0", "1"):
        start = before[rank]
        end = after[rank]
        count = float(end["count"]) - float(start["count"])
        total = float(end["sum"]) - float(start["sum"])
        bucket_delta = {
            upper: float(end["buckets"].get(upper, 0.0))
            - float(start["buckets"].get(upper, 0.0))
            for upper in set(start["buckets"]) | set(end["buckets"])
        }
        if count <= 0 or total < -1e-9 or any(value < -1e-9 for value in bucket_delta.values()):
            raise RuntimeError(
                f"invalid queue histogram delta for PP{rank}: "
                f"count={count}, sum={total}"
            )
        per_rank[rank] = {
            "count": int(round(count)),
            "sum": total,
            "mean": total / count if count else None,
            "p50": histogram_quantile_upper_bound(bucket_delta, count, 0.50),
            "p95": histogram_quantile_upper_bound(bucket_delta, count, 0.95),
        }
    expected_count = min(int(item["count"]) for item in per_rank.values())
    p50_values = [item["p50"] for item in per_rank.values()]
    p95_values = [item["p95"] for item in per_rank.values()]
    return {
        "source": "prometheus_histogram_delta",
        "count": expected_count,
        "mean": max(float(item["mean"]) for item in per_rank.values()),
        "p50": (
            max(float(value) for value in p50_values)
            if all(value is not None for value in p50_values)
            else None
        ),
        "p95": (
            max(float(value) for value in p95_values)
            if all(value is not None for value in p95_values)
            else None
        ),
        "min": None,
        "max": None,
        "quantiles_are_upper_bounds": True,
        "per_pp_rank": per_rank,
    }


def make_input_ids(seed: int, request_index: int, input_len: int) -> list[int]:
    # The sequence begins with a per-request header and then uses a deterministic
    # random body. This avoids accidental radix-cache hits without invalid token
    # IDs for Qwen3.
    rng = random.Random(seed * 1_000_003 + request_index * 97)
    header = [1000 + ((request_index * 37 + offset) % 7000) for offset in range(16)]
    body_len = max(0, input_len - len(header))
    return (header + [rng.randrange(1000, 9000) for _ in range(body_len)])[:input_len]


async def one_request(
    *,
    base_url: str,
    seed: int,
    request_index: int,
    input_len: int,
    output_len: int,
    timeout_s: float,
    launch_delay_s: float,
) -> dict[str, Any]:
    if launch_delay_s > 0:
        await asyncio.sleep(launch_delay_s)
    payload = {
        "input_ids": make_input_ids(seed, request_index, input_len),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": False,
    }
    client_start = time.time()
    monotonic_start = time.perf_counter()
    try:
        body = await asyncio.to_thread(
            post_json,
            base_url.rstrip("/") + "/generate",
            payload,
            timeout_s,
        )
        elapsed = time.perf_counter() - monotonic_start
        meta = body.get("meta_info", {})
        return {
            "request_index": request_index,
            "ok": True,
            "client_start_ts": client_start,
            "client_end_ts": time.time(),
            "client_latency_s": elapsed,
            "server_e2e_latency_s": meta.get("e2e_latency"),
            "queue_time_s": meta.get("queue_time"),
            "forward_entry_time": meta.get("forward_entry_time"),
            "prefill_finished_time": meta.get("prefill_finished_time"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "num_retractions": meta.get("num_retractions"),
            "finish_reason": meta.get("finish_reason"),
        }
    except Exception as exc:  # pragma: no cover - live-node path
        return {
            "request_index": request_index,
            "ok": False,
            "client_start_ts": client_start,
            "client_end_ts": time.time(),
            "client_latency_s": time.perf_counter() - monotonic_start,
            "error": repr(exc),
        }


async def run_wave(
    *,
    label: str,
    base_url: str,
    concurrency: int,
    waves: int,
    seed: int,
    input_len: int,
    output_len: int,
    timeout_s: float,
    launch_jitter_ms: float,
    marker_path: Path,
    request_count: int | None = None,
    arrival_rate_rps: float | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    queue_before = await queue_histogram_snapshot(base_url)
    phase_start = time.time()
    append_marker(
        marker_path,
        {
            "event": "phase_start",
            "phase": label,
            "timestamp": phase_start,
            "concurrency": concurrency,
            "waves": waves,
            "input_len": input_len,
            "output_len": output_len,
            "offered_rate_rps": arrival_rate_rps,
        },
    )
    if arrival_rate_rps is not None:
        if arrival_rate_rps <= 0:
            raise ValueError("arrival_rate_rps must be positive")
        total_requests = int(request_count or 0)
        if total_requests <= 0:
            raise ValueError("request_count must be positive for an offered-load wave")
        tasks = [
            one_request(
                base_url=base_url,
                seed=seed,
                request_index=request_index,
                input_len=input_len,
                output_len=output_len,
                timeout_s=timeout_s,
                launch_delay_s=request_index / arrival_rate_rps,
            )
            for request_index in range(total_requests)
        ]
        records.extend(await asyncio.gather(*tasks))
    else:
        for wave_index in range(waves):
            tasks = [
                one_request(
                    base_url=base_url,
                    seed=seed + wave_index * 10_000,
                    request_index=wave_index * concurrency + request_index,
                    input_len=input_len,
                    output_len=output_len,
                    timeout_s=timeout_s,
                    launch_delay_s=(request_index * launch_jitter_ms / 1000.0),
                )
                for request_index in range(concurrency)
            ]
            records.extend(await asyncio.gather(*tasks))
    phase_end = time.time()
    append_marker(
        marker_path,
        {
            "event": "phase_end",
            "phase": label,
            "timestamp": phase_end,
        },
    )
    queue_after = await queue_histogram_snapshot(base_url)
    successful = [record for record in records if record["ok"]]
    client = [float(record["client_latency_s"]) for record in successful]
    e2e = [
        float(record["server_e2e_latency_s"])
        for record in successful
        if record.get("server_e2e_latency_s") is not None
    ]
    queue = [
        float(record["queue_time_s"])
        for record in successful
        if record.get("queue_time_s") is not None
    ]
    total_tokens = sum(
        int(record.get("prompt_tokens") or 0)
        + int(record.get("completion_tokens") or 0)
        for record in successful
    )
    total_retractions = sum(
        int(record.get("num_retractions") or 0) for record in successful
    )
    duration = max(phase_end - phase_start, 1e-9)
    scheduler_queue = queue_histogram_delta(queue_before, queue_after)
    return {
        "label": label,
        "concurrency": concurrency,
        "waves": waves,
        "input_len": input_len,
        "output_len": output_len,
        "phase_start_ts": phase_start,
        "phase_end_ts": phase_end,
        "duration_s": duration,
        "offered_rate_rps": arrival_rate_rps,
        "requests": len(records),
        "successes": len(successful),
        "errors": len(records) - len(successful),
        "total_retractions": total_retractions,
        "request_throughput_rps": len(successful) / duration,
        "token_throughput_tps": total_tokens / duration,
        "client_latency_s": stats(client),
        "server_e2e_latency_s": stats(e2e),
        "queue_time_s": scheduler_queue,
        "http_response_queue_time_s": stats(queue),
        "records": records,
    }


def append_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def print_phase_summary(phase: dict[str, Any]) -> None:
    queue = phase["queue_time_s"]

    def numeric(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.6f}"

    print(
        f"{phase['label']}: requests={phase['successes']}/{phase['requests']} "
        f"throughput={phase['request_throughput_rps']:.3f} req/s "
        f"tokens={phase['token_throughput_tps']:.1f} tok/s "
        f"queue_mean={numeric(queue['mean'])}s "
        f"queue_p95_upper={numeric(queue['p95'])}s",
        flush=True,
    )


async def arm_profile(
    *,
    base_url: str,
    output_dir: Path,
    profile_id: str,
    prefix: str,
    stage: str,
    num_steps: int | None,
    start_step: int | None = None,
    activities: list[str] | None = None,
    profile_stages: list[str] | None = None,
    profile_by_stage: bool = True,
    with_stack: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_dir": str(output_dir.resolve()),
        "start_step": start_step,
        "num_steps": num_steps,
        "activities": activities or ["CPU", "GPU"],
        "profile_by_stage": profile_by_stage,
        "with_stack": with_stack,
        "record_shapes": True,
        "profile_id": profile_id,
        "merge_profiles": False,
        "profile_prefix": prefix,
        "profile_stages": profile_stages or [stage],
    }
    return await asyncio.to_thread(
        post_json,
        base_url.rstrip("/") + "/start_profile",
        payload,
        60.0,
    )


async def wait_for_trace_count(
    directory: Path, expected: int, timeout_s: float
) -> list[str]:
    deadline = time.monotonic() + timeout_s
    last_sizes: dict[str, int] = {}
    last_incomplete: list[str] = []
    stable_rounds = 0
    while time.monotonic() < deadline:
        paths = sorted(directory.glob("*.trace.json.gz"))
        sizes = {str(path): path.stat().st_size for path in paths}
        if len(paths) >= expected and sizes == last_sizes and all(s > 0 for s in sizes.values()):
            stable_rounds += 1
            if stable_rounds >= 2:
                last_incomplete = [
                    str(path) for path in paths if not gzip_stream_is_complete(path)
                ]
                if not last_incomplete:
                    return [str(path) for path in paths]
                stable_rounds = 0
        else:
            stable_rounds = 0
        last_sizes = sizes
        await asyncio.sleep(2.0)
    raise TimeoutError(
        f"expected {expected} stable complete gzip traces in {directory}, "
        f"observed={last_sizes}, incomplete={last_incomplete}"
    )


def gzip_stream_is_complete(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(8 * 1024 * 1024):
                pass
    except (EOFError, OSError):
        return False
    return True


async def wait_for_memory_snapshot_count(
    directory: Path, expected: int, timeout_s: float
) -> list[str]:
    deadline = time.monotonic() + timeout_s
    last_sizes: dict[str, int] = {}
    stable_rounds = 0
    while time.monotonic() < deadline:
        paths = sorted(directory.glob("*-memory-*.pickle"))
        sizes = {str(path): path.stat().st_size for path in paths}
        if (
            len(paths) >= expected
            and sizes == last_sizes
            and all(size > 0 for size in sizes.values())
        ):
            stable_rounds += 1
            if stable_rounds >= 2:
                return [str(path) for path in paths]
        else:
            stable_rounds = 0
        last_sizes = sizes
        await asyncio.sleep(2.0)
    raise TimeoutError(
        f"expected {expected} stable allocator snapshots in {directory}, "
        f"observed {last_sizes}"
    )


def choose_concurrency(calibration: list[dict[str, Any]]) -> tuple[int, str]:
    # "Near saturation with slight queueing" requires both throughput near the
    # best observed calibration point and visible-but-bounded scheduler delay.
    # This avoids mistaking a lightly loaded request that happens to wait 2 ms
    # for a saturated deployment.
    usable = [
        item
        for item in calibration
        if int(item.get("successes") or 0) > 0
        and int(item.get("errors") or 0) == 0
        and int(item.get("total_retractions") or 0) == 0
    ]
    if not usable:
        raise RuntimeError("no successful calibration points")
    max_throughput = max(float(item["token_throughput_tps"]) for item in usable)
    for item in calibration:
        if item not in usable:
            continue
        queue_p50 = item["queue_time_s"]["p50"]
        queue_p95 = item["queue_time_s"]["p95"]
        e2e_p50 = item["server_e2e_latency_s"]["p50"]
        throughput = float(item["token_throughput_tps"])
        if (
            queue_p50 is not None
            and queue_p95 is not None
            and e2e_p50 is not None
            and throughput >= 0.90 * max_throughput
            and queue_p50 >= 0.002
            and queue_p95 <= max(0.010, 0.25 * e2e_p50)
        ):
            return (
                int(item["concurrency"]),
                "first_near_saturation_slight_queue_point",
            )
    fallback = max(usable, key=lambda item: int(item["concurrency"]))
    return int(fallback["concurrency"]), "largest_calibrated_load_fallback"


def queue_is_slight(
    queue: dict[str, Any],
    e2e: dict[str, Any],
    *,
    minimum_mean_s: float = 0.002,
) -> bool:
    queue_mean = queue.get("mean")
    queue_p95 = queue.get("p95")
    e2e_p50 = e2e.get("p50")
    return bool(
        queue_mean is not None
        and queue_p95 is not None
        and e2e_p50 is not None
        and float(queue_mean) >= minimum_mean_s
        and float(queue_p95) <= max(0.050, 0.25 * float(e2e_p50))
    )


def choose_offered_rate(
    calibration: list[dict[str, Any]],
) -> tuple[float, str]:
    usable = [
        item
        for item in calibration
        if int(item.get("successes") or 0) == int(item.get("requests") or 0)
        and int(item.get("errors") or 0) == 0
        and int(item.get("total_retractions") or 0) == 0
    ]
    if not usable:
        raise RuntimeError("no successful offered-load calibration points")
    max_throughput = max(float(item["token_throughput_tps"]) for item in usable)
    for item in usable:
        if (
            float(item["token_throughput_tps"]) >= 0.90 * max_throughput
            and queue_is_slight(
                item["queue_time_s"],
                item["server_e2e_latency_s"],
                # Leave margin for normal run-to-run drift: 63437 selected a
                # 4.8-ms calibration point whose formal repeat fell to 1.7 ms.
                minimum_mean_s=0.008,
            )
        ):
            return (
                float(item["offered_rate_rps"]),
                "first_near_saturation_bounded_queue_rate",
            )
    fallback = min(
        usable,
        key=lambda item: (
            abs(float(item["token_throughput_tps"]) - 0.95 * max_throughput),
            float(item["offered_rate_rps"]),
        ),
    )
    return float(fallback["offered_rate_rps"]), "closest_clean_rate_fallback"


async def main_async(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = args.output_dir / "phase_markers.jsonl"
    marker_path.unlink(missing_ok=True)
    await wait_for_server(args.url, args.server_timeout)

    warmup = await run_wave(
        label="warmup",
        base_url=args.url,
        concurrency=4,
        waves=1,
        seed=args.seed,
        input_len=512,
        output_len=16,
        timeout_s=args.request_timeout,
        launch_jitter_ms=0.5,
        marker_path=marker_path,
    )
    print_phase_summary(warmup)

    calibration: list[dict[str, Any]] = []
    for index, concurrency in enumerate(args.calibration_concurrency):
        phase = await run_wave(
            label=f"calibration_c{concurrency}",
            base_url=args.url,
            concurrency=concurrency,
            waves=args.calibration_waves,
            seed=args.seed + 100_000 + index * 10_000,
            input_len=args.calibration_input_len,
            output_len=args.calibration_output_len,
            timeout_s=args.request_timeout,
            launch_jitter_ms=args.launch_jitter_ms,
            marker_path=marker_path,
        )
        calibration.append(phase)
        print_phase_summary(phase)

    peak_request_rate = max(
        float(item["request_throughput_rps"])
        for item in calibration
        if int(item.get("errors") or 0) == 0
    )
    rate_calibration: list[dict[str, Any]] = []
    for index, factor in enumerate(args.offered_rate_factors):
        offered_rate = peak_request_rate * factor
        request_count = max(
            1, math.ceil(offered_rate * args.offered_rate_duration)
        )
        phase = await run_wave(
            label=f"rate_calibration_{factor:.3f}",
            base_url=args.url,
            concurrency=args.profile_concurrency,
            waves=1,
            seed=args.seed + 150_000 + index * 10_000,
            input_len=args.calibration_input_len,
            output_len=args.calibration_output_len,
            timeout_s=args.request_timeout,
            launch_jitter_ms=0.0,
            marker_path=marker_path,
            request_count=request_count,
            arrival_rate_rps=offered_rate,
        )
        rate_calibration.append(phase)
        print_phase_summary(phase)

    selected_rate, selection_reason = choose_offered_rate(rate_calibration)
    selected_concurrency = args.profile_concurrency
    append_marker(
        marker_path,
        {
            "event": "selection",
            "phase": "calibration",
            "timestamp": time.time(),
            "selected_concurrency": selected_concurrency,
            "selected_rate_rps": selected_rate,
            "reason": selection_reason,
        },
    )

    steady_request_count = max(
        1, math.ceil(selected_rate * args.steady_load_duration)
    )
    steady_load = await run_wave(
        label="steady_load",
        base_url=args.url,
        concurrency=selected_concurrency,
        waves=1,
        seed=args.seed + 175_000,
        input_len=args.calibration_input_len,
        output_len=args.calibration_output_len,
        timeout_s=args.request_timeout,
        launch_jitter_ms=0.0,
        marker_path=marker_path,
        request_count=steady_request_count,
        arrival_rate_rps=selected_rate,
    )
    print_phase_summary(steady_load)

    # Collect allocator state before Kineto exists. Formal runs 63329 and 63437
    # showed that creating another native recorder after a stack-unwinding
    # Kineto result can trigger delayed ProfilerResult GC and segfault PP0.
    memory_trace_dir = args.output_dir / "traces" / "memory"
    memory_arm = await arm_profile(
        base_url=args.url,
        output_dir=memory_trace_dir,
        profile_id=f"{args.profile_id}-memory",
        prefix=f"{args.profile_prefix}-memory",
        stage="prefill",
        num_steps=args.memory_profile_steps,
        activities=["MEM"],
        profile_stages=["prefill", "decode"],
        with_stack=False,
    )
    memory_concurrency = min(
        selected_concurrency, args.memory_profile_concurrency
    )
    memory = await run_wave(
        label="profile_memory",
        base_url=args.url,
        concurrency=memory_concurrency,
        waves=1,
        seed=args.seed + 400_000,
        input_len=args.prefill_input_len,
        output_len=args.prefill_output_len,
        timeout_s=args.request_timeout,
        launch_jitter_ms=args.launch_jitter_ms,
        marker_path=marker_path,
    )
    print_phase_summary(memory)
    memory_snapshots = await wait_for_memory_snapshot_count(
        memory_trace_dir, expected=4, timeout_s=args.trace_timeout
    )

    # Arm Kineto exactly once, but let each scheduler rank stop and export its
    # own combined interval after the same number of local forwards.  A manual
    # /stop_profile stalls for PP=2 in the pinned legacy profiler path: PP0
    # enters the synchronous stop/export path and does not return before the
    # control request reaches PP1.  Forward-count stops are evaluated locally
    # by both schedulers, so neither rank depends on a second stop control
    # message.  The 100-step interval has a shape-derived lower bound: the
    # first wave uses at most 16 * (4096 / 1024) = 64 prefill steps plus 16
    # decode steps, and the second wave uses at most 16 short-prompt prefill
    # steps.  It therefore includes at least four stable second-wave decode
    # steps.  There is deliberately no second
    # Kineto arm: runs 63329/63437 showed that a later native ProfilerResult
    # lifetime can crash during unwinder cleanup.
    combined_trace_dir = args.output_dir / "traces" / "combined"
    combined_arm = await arm_profile(
        base_url=args.url,
        output_dir=combined_trace_dir,
        profile_id=f"{args.profile_id}-combined",
        prefix=f"{args.profile_prefix}-combined",
        stage="combined",
        num_steps=args.combined_profile_steps,
        start_step=1,
        activities=["CPU", "GPU"],
        profile_stages=[],
        profile_by_stage=False,
        with_stack=False,
    )
    prefill = await run_wave(
        label="profile_prefill",
        base_url=args.url,
        concurrency=selected_concurrency,
        waves=1,
        seed=args.seed + 200_000,
        input_len=args.prefill_input_len,
        output_len=args.prefill_output_len,
        timeout_s=args.request_timeout,
        launch_jitter_ms=args.launch_jitter_ms,
        marker_path=marker_path,
    )
    print_phase_summary(prefill)
    decode_concurrency = max(selected_concurrency, args.min_decode_concurrency)
    decode = await run_wave(
        label="profile_decode",
        base_url=args.url,
        concurrency=decode_concurrency,
        waves=1,
        seed=args.seed + 300_000,
        input_len=args.decode_input_len,
        output_len=args.decode_output_len,
        timeout_s=args.request_timeout,
        launch_jitter_ms=args.launch_jitter_ms,
        marker_path=marker_path,
    )
    print_phase_summary(decode)
    combined_traces = await wait_for_trace_count(
        combined_trace_dir, expected=2, timeout_s=args.trace_timeout
    )

    result = {
        "schema_version": 2,
        "url": args.url,
        "profile_id": args.profile_id,
        "profile_prefix": args.profile_prefix,
        "selected_concurrency": selected_concurrency,
        "selected_rate_rps": selected_rate,
        "decode_concurrency": decode_concurrency,
        "selection_reason": selection_reason,
        "chunked_prefill_size": args.chunked_prefill_size,
        "warmup": warmup,
        "calibration": calibration,
        "rate_calibration": rate_calibration,
        "steady_load": steady_load,
        "prefill_profile": {
            "capture_mode": "single_arm_forward_count_automatic",
            "arm_response": combined_arm,
            "profile_steps": args.combined_profile_steps,
            "trace_files": combined_traces,
            "workload": prefill,
        },
        "decode_profile": {
            "capture_mode": "single_arm_forward_count_automatic",
            "arm_response": combined_arm,
            "profile_steps": args.combined_profile_steps,
            "trace_files": combined_traces,
            "workload": decode,
        },
        "memory_profile": {
            "arm_response": memory_arm,
            "profile_steps": args.memory_profile_steps,
            "snapshot_files": memory_snapshots,
            "workload": memory,
        },
    }
    (args.output_dir / "workload_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if (
        warmup["errors"]
        or any(item["errors"] for item in calibration)
        or any(item["errors"] for item in rate_calibration)
        or steady_load["errors"]
    ):
        return 2
    if prefill["errors"] or decode["errors"] or memory["errors"]:
        return 3
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30188")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--profile-prefix", default="aegiskv-qwen3-8b-pp2")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--calibration-concurrency",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 12, 16, 24, 32],
    )
    parser.add_argument("--calibration-waves", type=int, default=1)
    parser.add_argument("--calibration-input-len", type=int, default=2048)
    parser.add_argument("--calibration-output-len", type=int, default=64)
    parser.add_argument(
        "--offered-rate-factors",
        type=float,
        nargs="+",
        default=[0.75, 0.85, 0.90, 0.95, 1.00, 1.05],
    )
    parser.add_argument("--offered-rate-duration", type=float, default=12.0)
    parser.add_argument("--steady-load-duration", type=float, default=20.0)
    parser.add_argument("--profile-concurrency", type=int, default=16)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--prefill-input-len", type=int, default=4096)
    parser.add_argument("--prefill-output-len", type=int, default=16)
    parser.add_argument("--decode-input-len", type=int, default=128)
    parser.add_argument("--decode-output-len", type=int, default=128)
    parser.add_argument("--min-decode-concurrency", type=int, default=8)
    parser.add_argument("--combined-profile-steps", type=int, default=100)
    parser.add_argument("--memory-profile-steps", type=int, default=4)
    parser.add_argument("--memory-profile-concurrency", type=int, default=8)
    parser.add_argument("--launch-jitter-ms", type=float, default=0.5)
    parser.add_argument("--server-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--trace-timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    try:
        return asyncio.run(main_async(parse_args()))
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"profile workload failed: {exc!r}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
