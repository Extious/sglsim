from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


_SIM_PREFIX_RE = re.compile(
    r"^\[(?P<sim_time>\d+(?:\.\d+)?)s(?: DP(?P<dp>\d+))?\]\s*(?P<message>.*)$"
)
_WALL_PREFIX_RE = re.compile(
    r"^\[(?P<wall_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)(?: DP(?P<dp>\d+))?\]\s*(?P<message>.*)$"
)
_INT_FIELD_RE = re.compile(r"#(?P<name>[\w-]+):\s*(?P<value>-?\d+)")
_FLOAT_FIELD_RE = re.compile(r"(?P<name>token usage|input throughput \(token/s\)|gen throughput \(token/s\)):\s*(?P<value>-?\d+(?:\.\d+)?)")
_HTTP_RE = re.compile(
    r'INFO:\s+\S+\s+-\s+"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/\d(?:\.\d)?"\s+(?P<status_code>\d+)'
)
_RETRACT_RE = re.compile(
    r"#retracted_reqs:\s*(?P<retracted>\d+),\s*"
    r"#new_tokens_gained:\s*(?P<tokens>\d+),\s*"
    r"#new_token_ratio:\s*(?P<before>\d+(?:\.\d+)?)\s*->\s*(?P<after>\d+(?:\.\d+)?)"
)
_MAX_TOTAL_RE = re.compile(
    r"max_total_num_tokens=(?P<max_total>\d+),\s*"
    r"chunked_prefill_size=(?P<chunked>\d+),\s*"
    r"max_prefill_tokens=(?P<max_prefill>\d+),\s*"
    r"max_running_requests=(?P<max_running>\d+),\s*"
    r"context_len=(?P<context>\d+),\s*"
    r"available_gpu_mem=(?P<available>\d+(?:\.\d+)?) GB"
)

_FIELD_NAMES = {
    "new-seq": "new_seqs",
    "new-token": "new_tokens",
    "cached-token": "cached_tokens",
    "running-req": "running_reqs",
    "queue-req": "queue_reqs",
    "logical-token": "logical_tokens",
    "avail-token": "available_tokens",
    "evictable-token": "evictable_tokens",
    "token": "tokens",
}
_FLOAT_NAMES = {
    "token usage": "token_usage",
    "input throughput (token/s)": "input_throughput",
    "gen throughput (token/s)": "gen_throughput",
}


@dataclass(frozen=True)
class LogEvent:
    event_type: str
    message: str
    sim_time_s: float | None = None
    wall_time: str | None = None
    dp_rank: int | None = None
    fields: dict[str, Any] | None = None


def _parse_prefix(line: str) -> tuple[float | None, str | None, int | None, str] | None:
    sim_match = _SIM_PREFIX_RE.match(line)
    if sim_match:
        dp = sim_match.group("dp")
        return (
            float(sim_match.group("sim_time")),
            None,
            int(dp) if dp is not None else None,
            sim_match.group("message"),
        )
    wall_match = _WALL_PREFIX_RE.match(line)
    if wall_match:
        dp = wall_match.group("dp")
        return (
            None,
            wall_match.group("wall_time"),
            int(dp) if dp is not None else None,
            wall_match.group("message"),
        )
    return None


def _parse_batch_fields(message: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for match in _INT_FIELD_RE.finditer(message):
        name = _FIELD_NAMES.get(match.group("name"))
        if name is not None:
            fields[name] = int(match.group("value"))
    for match in _FLOAT_FIELD_RE.finditer(message):
        name = _FLOAT_NAMES[match.group("name")]
        fields[name] = float(match.group("value"))
    if "cuda graph: True" in message:
        fields["cuda_graph"] = True
    elif "cuda graph: False" in message:
        fields["cuda_graph"] = False
    return fields


def _parse_event_type(message: str) -> str | None:
    if message.startswith("Prefill batch"):
        return "prefill"
    if message.startswith("Decode batch"):
        return "decode"
    if message.startswith("KV cache pool is full. Retract requests."):
        return "kv_retract"
    if _HTTP_RE.search(message):
        return "http_access"
    if _MAX_TOTAL_RE.search(message):
        return "capacity"
    return None


def _parse_fields(event_type: str, message: str) -> dict[str, Any]:
    if event_type in {"prefill", "decode"}:
        return _parse_batch_fields(message)
    if event_type == "kv_retract":
        match = _RETRACT_RE.search(message)
        if match is None:
            return {}
        return {
            "retracted_reqs": int(match.group("retracted")),
            "new_tokens_gained": int(match.group("tokens")),
            "new_token_ratio_before": float(match.group("before")),
            "new_token_ratio_after": float(match.group("after")),
        }
    if event_type == "http_access":
        match = _HTTP_RE.search(message)
        if match is None:
            return {}
        return {
            "method": match.group("method"),
            "path": match.group("path"),
            "status_code": int(match.group("status_code")),
        }
    if event_type == "capacity":
        match = _MAX_TOTAL_RE.search(message)
        if match is None:
            return {}
        return {
            "max_total_num_tokens": int(match.group("max_total")),
            "chunked_prefill_size": int(match.group("chunked")),
            "max_prefill_tokens": int(match.group("max_prefill")),
            "max_running_requests": int(match.group("max_running")),
            "context_len": int(match.group("context")),
            "available_gpu_mem_gb": float(match.group("available")),
        }
    return {}


def parse_server_log_lines(lines: Iterable[str]) -> list[LogEvent]:
    events: list[LogEvent] = []
    for line in lines:
        parsed = _parse_prefix(line.strip())
        if parsed is None:
            continue
        sim_time_s, wall_time, dp_rank, message = parsed
        event_type = _parse_event_type(message)
        if event_type is None:
            continue
        events.append(
            LogEvent(
                event_type=event_type,
                message=message,
                sim_time_s=sim_time_s,
                wall_time=wall_time,
                dp_rank=dp_rank,
                fields=_parse_fields(event_type, message),
            )
        )
    return events


def parse_server_log(path: Path) -> list[LogEvent]:
    return parse_server_log_lines(path.read_text(encoding="utf-8", errors="replace").splitlines())


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_events(events: list[LogEvent]) -> dict[str, Any]:
    prefill = [event for event in events if event.event_type == "prefill"]
    decode = [event for event in events if event.event_type == "decode"]
    retract = [event for event in events if event.event_type == "kv_retract"]
    http = [event for event in events if event.event_type == "http_access"]
    token_usage = [
        float(event.fields.get("token_usage", 0.0))
        for event in events
        if event.fields and "token_usage" in event.fields
    ]
    decode_throughput = [
        float(event.fields["gen_throughput"])
        for event in decode
        if event.fields and "gen_throughput" in event.fields
    ]
    return {
        "prefill_count": len(prefill),
        "decode_count": len(decode),
        "kv_retract_count": len(retract),
        "http_access_count": len(http),
        "max_token_usage": max(token_usage, default=0.0),
        "decode_throughput_p50": float(median(decode_throughput)) if decode_throughput else 0.0,
        "decode_throughput_p90": _percentile(decode_throughput, 0.90),
        "decode_throughput_p99": _percentile(decode_throughput, 0.99),
    }


def _read_request_details(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {str(row.get("rid", "")): row for row in csv.DictReader(f)}


def _first_present_field(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    for field in candidates:
        if field in row and row[field] not in ("", None):
            return field
    raise KeyError(f"Missing any of fields: {', '.join(candidates)}")


def _float_field(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", None) else 0.0


def _mean_abs_error(lhs: list[float], rhs: list[float]) -> float:
    if not lhs:
        return 0.0
    return sum(abs(a - b) for a, b in zip(lhs, rhs, strict=True)) / len(lhs)


def summarize_trace_comparison(real_csv: Path, sim_csv: Path) -> dict[str, Any]:
    real_rows = _read_request_details(real_csv)
    sim_rows = _read_request_details(sim_csv)
    common_rids = sorted(rid for rid in real_rows if rid and rid in sim_rows)
    if not common_rids:
        return {"request_count": 0}

    first_real = real_rows[common_rids[0]]
    first_sim = sim_rows[common_rids[0]]
    real_latency_field = _first_present_field(
        first_real,
        ("e2e_latency_s", "inference_time_s", "total_latency_s"),
    )
    sim_latency_field = _first_present_field(
        first_sim,
        ("logical_total_latency_s", "total_latency_s", "inference_time_s"),
    )
    real_latencies = [
        _float_field(real_rows[rid], real_latency_field) for rid in common_rids
    ]
    sim_latencies = [_float_field(sim_rows[rid], sim_latency_field) for rid in common_rids]
    real_cache_hits = [
        _float_field(real_rows[rid], "cache_hit_tokens") for rid in common_rids
    ]
    sim_cache_hits = [
        _float_field(sim_rows[rid], "cache_hit_tokens") for rid in common_rids
    ]
    return {
        "request_count": len(common_rids),
        "real_latency_field": real_latency_field,
        "sim_latency_field": sim_latency_field,
        "real_latency_p50_s": float(median(real_latencies)),
        "sim_latency_p50_s": float(median(sim_latencies)),
        "real_latency_p90_s": _percentile(real_latencies, 0.90),
        "sim_latency_p90_s": _percentile(sim_latencies, 0.90),
        "real_latency_max_s": max(real_latencies),
        "sim_latency_max_s": max(sim_latencies),
        "latency_mae_s": _mean_abs_error(real_latencies, sim_latencies),
        "cache_hit_mae_tokens": _mean_abs_error(real_cache_hits, sim_cache_hits),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze synthesis_requests server logs")
    parser.add_argument("log_path", type=Path, nargs="?")
    parser.add_argument("--real-request-details", type=Path)
    parser.add_argument("--sim-request-details", type=Path)
    args = parser.parse_args(argv)
    if args.real_request_details and args.sim_request_details:
        print(
            json.dumps(
                summarize_trace_comparison(
                    args.real_request_details,
                    args.sim_request_details,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.log_path is None:
        parser.error("log_path or both request detail CSVs are required")
    events = parse_server_log(args.log_path)
    print(json.dumps(summarize_events(events), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
