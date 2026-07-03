#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READINESS_DIR = SGLSIM_LARGE_ROOT / "results" / "strict_final_readiness"
DEFAULT_REQUIRED_CSV = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.required.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.live.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.live.json"
DEFAULT_OUTPUT_MD = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.live.md"

EVENT_SCHEMA_VERSION = "p0-live-copy-event-v1"
FINAL_DATA_PLANE = "physical_tensor_copy"
COPY_BACKEND = "torch.cuda.copy_"
# Qwen/Qwen3-8B-style bf16 KV estimate:
# 2 tensors (K,V) * 36 layers * 8 KV heads * 128 head dim * 2 bytes.
DEFAULT_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2

EVENT_FIELDS = [
    "suite_label",
    "request_csv",
    "rid",
    "event_schema_version",
    "event_kind",
    "restore_action",
    "restored_tokens",
    "aegis_replay_restore_data_plane",
    "aegis_replay_restore_timing_source",
    "physical_action",
    "physical_copy_verified",
    "copy_latency_ms",
    "copy_bytes",
    "source_artifact",
    "src_rank",
    "dst_rank",
    "src_device",
    "dst_device",
    "copy_backend",
    "copy_started_at_ns",
    "copy_ended_at_ns",
    "measurement_repeats",
    "warmup_repeats",
    "bytes_per_token",
    "copy_path_available",
    "copy_path_note",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _int(value: object) -> int:
    try:
        if value in ("", None):
            return 0
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * 0.95
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def copy_bytes_for_row(row: dict[str, Any], *, bytes_per_token: int) -> int:
    restored_tokens = _int(row.get("restored_tokens"))
    if restored_tokens <= 0:
        return 0
    return restored_tokens * int(bytes_per_token)


def select_required_rows(
    rows: list[dict[str, str]],
    *,
    event_kind: str,
    limit: int | None,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if event_kind == "all" or str(row.get("event_kind") or "") == event_kind
    ]
    if limit is not None:
        return selected[:limit]
    return selected


def copy_plan(
    row: dict[str, Any],
    *,
    host_target_device: int,
    peer_source_device: int,
    peer_target_device: int,
) -> dict[str, str | bool]:
    event_kind = str(row.get("event_kind") or "").strip()
    if event_kind == "live_host_h2d_copy":
        return {
            "src_rank": "",
            "dst_rank": str(host_target_device),
            "src_device": "cpu:pinned",
            "dst_device": f"cuda:{host_target_device}",
            "requires_peer_access": False,
            "copy_path_note": "pinned host to GPU restore copy",
        }
    if event_kind == "live_peer_p2p_copy":
        return {
            "src_rank": str(peer_source_device),
            "dst_rank": str(peer_target_device),
            "src_device": f"cuda:{peer_source_device}",
            "dst_device": f"cuda:{peer_target_device}",
            "requires_peer_access": True,
            "copy_path_note": "peer GPU to GPUReplica restore copy",
        }
    return {
        "src_rank": "",
        "dst_rank": "",
        "src_device": "",
        "dst_device": "",
        "requires_peer_access": False,
        "copy_path_note": f"unsupported event kind: {event_kind}",
    }


def build_event_row(
    required: dict[str, Any],
    *,
    latency_ms: float,
    copy_bytes: int,
    source_artifact: Path,
    started_at_ns: int,
    ended_at_ns: int,
    repeats: int,
    warmup: int,
    bytes_per_token: int,
    plan: dict[str, str | bool],
    verified: bool = True,
    note: str = "",
) -> dict[str, Any]:
    event_kind = str(required.get("event_kind") or "").strip()
    return {
        "suite_label": str(required.get("suite_label") or "").strip(),
        "request_csv": str(required.get("request_csv") or "").strip(),
        "rid": str(required.get("rid") or "").strip(),
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "event_kind": event_kind,
        "restore_action": str(required.get("restore_action") or "").strip(),
        "restored_tokens": _int(required.get("restored_tokens")),
        "aegis_replay_restore_data_plane": FINAL_DATA_PLANE,
        "aegis_replay_restore_timing_source": event_kind,
        "physical_action": str(required.get("physical_action") or "").strip(),
        "physical_copy_verified": "true" if verified else "false",
        "copy_latency_ms": f"{latency_ms:.6f}" if latency_ms > 0 else "",
        "copy_bytes": int(copy_bytes) if copy_bytes > 0 else "",
        "source_artifact": str(source_artifact.resolve()) if source_artifact else "",
        "src_rank": str(plan.get("src_rank") or ""),
        "dst_rank": str(plan.get("dst_rank") or ""),
        "src_device": str(plan.get("src_device") or ""),
        "dst_device": str(plan.get("dst_device") or ""),
        "copy_backend": COPY_BACKEND if verified else "",
        "copy_started_at_ns": int(started_at_ns) if started_at_ns else "",
        "copy_ended_at_ns": int(ended_at_ns) if ended_at_ns else "",
        "measurement_repeats": int(repeats),
        "warmup_repeats": int(warmup),
        "bytes_per_token": int(bytes_per_token),
        "copy_path_available": "true" if verified else "false",
        "copy_path_note": note or str(plan.get("copy_path_note") or ""),
    }


def cuda_environment(torch: Any) -> dict[str, Any]:
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
        }
        for index in range(device_count)
    ]
    peer_access: dict[str, bool] = {}
    for source in range(device_count):
        for target in range(device_count):
            if source == target:
                continue
            peer_access[f"{source}->{target}"] = bool(
                torch.cuda.can_device_access_peer(source, target)
            )
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": device_count,
        "devices": devices,
        "peer_access": peer_access,
        "torch_version": getattr(torch, "__version__", ""),
    }


def require_torch_cuda() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError("PyTorch is required for live CUDA copy collection") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot collect live copy events")
    return torch


def _wall_timed_copy(
    torch: Any,
    fn: Any,
    *,
    devices: tuple[int, ...],
    repeats: int,
    warmup: int,
) -> tuple[float, int, int]:
    samples: list[float] = []
    started_at_ns = 0
    ended_at_ns = 0
    for index in range(warmup + repeats):
        for device in devices:
            torch.cuda.synchronize(device)
        start = time.time_ns()
        fn()
        for device in devices:
            torch.cuda.synchronize(device)
        end = time.time_ns()
        if index >= warmup:
            if not started_at_ns:
                started_at_ns = start
            ended_at_ns = end
            samples.append((end - start) / 1_000_000.0)
    return _mean(samples), started_at_ns, ended_at_ns


def measure_host_h2d(
    torch: Any,
    *,
    byte_count: int,
    target_device: int,
    repeats: int,
    warmup: int,
    fill_value: int,
) -> tuple[float, int, int]:
    device = f"cuda:{target_device}"
    with torch.cuda.device(target_device):
        source = torch.empty(byte_count, dtype=torch.uint8, device="cpu", pin_memory=True)
        target = torch.empty(byte_count, dtype=torch.uint8, device=device)
        source.fill_(fill_value)
        target.fill_(0)
        latency_ms, started, ended = _wall_timed_copy(
            torch,
            lambda: target.copy_(source, non_blocking=True),
            devices=(target_device,),
            repeats=repeats,
            warmup=warmup,
        )
        if byte_count > 0 and (
            int(target[0].item()) != fill_value
            or int(target[byte_count - 1].item()) != fill_value
        ):
            raise RuntimeError("Host H2D copy verification failed")
    return latency_ms, started, ended


def measure_peer_p2p(
    torch: Any,
    *,
    byte_count: int,
    source_device: int,
    target_device: int,
    repeats: int,
    warmup: int,
    fill_value: int,
) -> tuple[float, int, int]:
    if source_device == target_device:
        raise RuntimeError("P2P source and target devices must differ")
    if not torch.cuda.can_device_access_peer(source_device, target_device):
        raise RuntimeError(
            f"CUDA peer access is unavailable for {source_device}->{target_device}"
        )
    with torch.cuda.device(source_device):
        source = torch.empty(byte_count, dtype=torch.uint8, device=f"cuda:{source_device}")
        source.fill_(fill_value)
    with torch.cuda.device(target_device):
        target = torch.empty(byte_count, dtype=torch.uint8, device=f"cuda:{target_device}")
        target.fill_(0)
        latency_ms, started, ended = _wall_timed_copy(
            torch,
            lambda: target.copy_(source, non_blocking=True),
            devices=(source_device, target_device),
            repeats=repeats,
            warmup=warmup,
        )
        if byte_count > 0 and (
            int(target[0].item()) != fill_value
            or int(target[byte_count - 1].item()) != fill_value
        ):
            raise RuntimeError("Peer P2P copy verification failed")
    return latency_ms, started, ended


def summarize(
    *,
    required_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    event_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    output_csv: Path,
    environment: dict[str, Any],
    bytes_per_token: int,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    latencies_by_kind: dict[str, list[float]] = {}
    bytes_by_kind: dict[str, int] = {}
    for row in event_rows:
        event_kind = str(row.get("event_kind") or "")
        by_kind[event_kind] = by_kind.get(event_kind, 0) + 1
        latencies_by_kind.setdefault(event_kind, []).append(float(row["copy_latency_ms"]))
        bytes_by_kind[event_kind] = bytes_by_kind.get(event_kind, 0) + _int(row.get("copy_bytes"))

    skipped_by_reason: dict[str, int] = {}
    for row in skipped_rows:
        reason = str(row.get("skip_reason") or "unknown")
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1

    latency_summary = {
        kind: {
            "mean_ms": _mean(values),
            "p95_ms": _p95(values),
            "rows": len(values),
        }
        for kind, values in sorted(latencies_by_kind.items())
    }
    return {
        "bytes_per_token": int(bytes_per_token),
        "copy_backend": COPY_BACKEND,
        "environment": environment,
        "event_rows": len(event_rows),
        "event_rows_by_kind": dict(sorted(by_kind.items())),
        "latency_by_kind": latency_summary,
        "output_csv": str(output_csv),
        "required_rows": len(required_rows),
        "selected_rows": len(selected_rows),
        "skipped_rows": len(skipped_rows),
        "skipped_rows_by_reason": dict(sorted(skipped_by_reason.items())),
        "strict_final_candidate_ready": len(event_rows) == len(required_rows),
        "total_copy_bytes_by_kind": dict(sorted(bytes_by_kind.items())),
        "warmup_repeats": int(warmup),
        "measurement_repeats": int(repeats),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    env = payload.get("environment") or {}
    lines = [
        "# P0 Live Physical Tensor-Copy Events",
        "",
        "This report records measured CUDA copy events for the P0 required-event",
        "manifest. It only emits rows whose copy path was executed and verified.",
        "",
        f"- output CSV: `{payload['output_csv']}`",
        f"- required rows: `{payload['required_rows']}`",
        f"- selected rows: `{payload['selected_rows']}`",
        f"- emitted live event rows: `{payload['event_rows']}`",
        f"- skipped rows: `{payload['skipped_rows']}`",
        f"- event rows by kind: `{payload['event_rows_by_kind']}`",
        f"- skipped rows by reason: `{payload['skipped_rows_by_reason']}`",
        f"- bytes per token: `{payload['bytes_per_token']}`",
        f"- copy backend: `{payload['copy_backend']}`",
        "",
        "## CUDA Environment",
        "",
        f"- torch version: `{env.get('torch_version', '')}`",
        f"- cuda available: `{str(env.get('cuda_available', False)).lower()}`",
        f"- device count: `{env.get('device_count', 0)}`",
        f"- peer access: `{env.get('peer_access', {})}`",
        "",
        "## Latency Summary",
        "",
    ]
    latency = payload.get("latency_by_kind") or {}
    if latency:
        for kind, item in latency.items():
            lines.append(
                f"- `{kind}`: rows `{item['rows']}`, mean `{item['mean_ms']:.6f}` ms, p95 `{item['p95_ms']:.6f}` ms"
            )
    else:
        lines.append("- No verified live-copy rows were emitted.")
    lines.append("")
    return "\n".join(lines)


def collect_events(
    *,
    required_csv: Path,
    output_csv: Path,
    event_kind: str,
    limit: int | None,
    bytes_per_token: int,
    repeats: int,
    warmup: int,
    host_target_device: int,
    peer_source_device: int,
    peer_target_device: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    torch = require_torch_cuda()
    required_rows = _read_csv(required_csv)
    selected_rows = select_required_rows(required_rows, event_kind=event_kind, limit=limit)
    environment = cuda_environment(torch)

    event_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for index, required in enumerate(selected_rows, start=1):
        plan = copy_plan(
            required,
            host_target_device=host_target_device,
            peer_source_device=peer_source_device,
            peer_target_device=peer_target_device,
        )
        copy_bytes = copy_bytes_for_row(required, bytes_per_token=bytes_per_token)
        if copy_bytes <= 0:
            skipped_rows.append(
                {
                    "row_index": index,
                    "rid": required.get("rid", ""),
                    "event_kind": required.get("event_kind", ""),
                    "skip_reason": "non_positive_copy_bytes",
                }
            )
            continue

        try:
            fill_value = (index % 251) + 1
            if required.get("event_kind") == "live_host_h2d_copy":
                latency_ms, started, ended = measure_host_h2d(
                    torch,
                    byte_count=copy_bytes,
                    target_device=host_target_device,
                    repeats=repeats,
                    warmup=warmup,
                    fill_value=fill_value,
                )
            elif required.get("event_kind") == "live_peer_p2p_copy":
                latency_ms, started, ended = measure_peer_p2p(
                    torch,
                    byte_count=copy_bytes,
                    source_device=peer_source_device,
                    target_device=peer_target_device,
                    repeats=repeats,
                    warmup=warmup,
                    fill_value=fill_value,
                )
            else:
                raise RuntimeError(f"unsupported event kind: {required.get('event_kind')}")
        except RuntimeError as exc:
            skipped_rows.append(
                {
                    "row_index": index,
                    "rid": required.get("rid", ""),
                    "event_kind": required.get("event_kind", ""),
                    "copy_bytes": copy_bytes,
                    "skip_reason": str(exc),
                }
            )
            continue

        event_rows.append(
            build_event_row(
                required,
                latency_ms=latency_ms,
                copy_bytes=copy_bytes,
                source_artifact=output_csv,
                started_at_ns=started,
                ended_at_ns=ended,
                repeats=repeats,
                warmup=warmup,
                bytes_per_token=bytes_per_token,
                plan=plan,
                note=str(plan.get("copy_path_note") or ""),
            )
        )
    return event_rows, skipped_rows, environment, required_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect measured CUDA live-copy events for P0 strict-final evidence."
    )
    parser.add_argument("--required-csv", type=Path, default=DEFAULT_REQUIRED_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--event-kind",
        choices=("all", "live_host_h2d_copy", "live_peer_p2p_copy"),
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bytes-per-token", type=int, default=DEFAULT_BYTES_PER_TOKEN)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--host-target-device", type=int, default=0)
    parser.add_argument("--peer-source-device", type=int, default=0)
    parser.add_argument("--peer-target-device", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive when provided")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.bytes_per_token <= 0:
        raise SystemExit("--bytes-per-token must be positive")

    event_rows, skipped_rows, environment, required_rows = collect_events(
        required_csv=args.required_csv,
        output_csv=args.output_csv,
        event_kind=args.event_kind,
        limit=args.limit,
        bytes_per_token=args.bytes_per_token,
        repeats=args.repeats,
        warmup=args.warmup,
        host_target_device=args.host_target_device,
        peer_source_device=args.peer_source_device,
        peer_target_device=args.peer_target_device,
    )
    selected_rows = select_required_rows(
        required_rows,
        event_kind=args.event_kind,
        limit=args.limit,
    )
    _write_csv(args.output_csv, event_rows)
    payload = summarize(
        required_rows=required_rows,
        selected_rows=selected_rows,
        event_rows=event_rows,
        skipped_rows=skipped_rows,
        output_csv=args.output_csv,
        environment=environment,
        bytes_per_token=args.bytes_per_token,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    payload["required_csv"] = str(args.required_csv)
    payload["event_kind_filter"] = args.event_kind
    payload["limit"] = args.limit
    payload["skipped_sample"] = skipped_rows[:10]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and (skipped_rows or len(event_rows) != len(selected_rows)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
