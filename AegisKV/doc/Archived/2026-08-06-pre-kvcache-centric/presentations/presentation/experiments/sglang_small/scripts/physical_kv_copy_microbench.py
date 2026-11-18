#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_TOKEN_COUNTS = (128, 256, 512, 1024, 2048, 4096, 8192)
# Qwen/Qwen3-8B-style bf16 KV estimate:
# 2 tensors (K,V) * 36 layers * 8 KV heads * 128 head dim * 2 bytes.
DEFAULT_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    return 1.96 * _stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _gbps(byte_count: int, ms: float) -> float:
    if byte_count <= 0 or ms <= 0:
        return 0.0
    return (byte_count / 1e9) / (ms / 1000.0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_samples(
    *,
    path_name: str,
    token_count: int,
    byte_count: int,
    samples_ms: list[float],
    available: bool,
    note: str = "",
) -> dict[str, Any]:
    p50_ms = _percentile(samples_ms, 0.50)
    p95_ms = _percentile(samples_ms, 0.95)
    return {
        "path": path_name,
        "tokens": int(token_count),
        "bytes": int(byte_count),
        "mib": byte_count / (1024.0 * 1024.0),
        "available": str(bool(available)).lower(),
        "samples": len(samples_ms),
        "mean_ms": _mean(samples_ms),
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "ci95_ms": _ci95(samples_ms),
        "mean_gbps": _gbps(byte_count, _mean(samples_ms)),
        "p50_gbps": _gbps(byte_count, p50_ms),
        "p95_gbps": _gbps(byte_count, p95_ms),
        "note": note,
    }


def _cuda_timed_copy(torch: Any, fn, repeats: int, warmup: int) -> list[float]:
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for index in range(warmup + repeats):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        if index >= warmup:
            samples.append(elapsed_ms)
    return samples


def _bench_one_size(
    torch: Any,
    *,
    token_count: int,
    bytes_per_token: int,
    repeats: int,
    warmup: int,
    dtype: str,
) -> list[dict[str, Any]]:
    byte_count = int(token_count) * int(bytes_per_token)
    element_size = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    element_count = max(1, math.ceil(byte_count / element_size))
    effective_bytes = element_count * element_size

    rows: list[dict[str, Any]] = []
    dtype_obj = getattr(torch, dtype)
    with torch.cuda.device(0):
        src0 = torch.empty(element_count, device="cuda:0", dtype=dtype_obj)
        dst0 = torch.empty_like(src0)
        cpu_pinned = torch.empty(element_count, device="cpu", dtype=dtype_obj, pin_memory=True)
        cpu_pinned.fill_(1)
        src0.fill_(2)
        dst0.fill_(0)

        rows.append(
            summarize_samples(
                path_name="h2d_pinned",
                token_count=token_count,
                byte_count=effective_bytes,
                samples_ms=_cuda_timed_copy(
                    torch,
                    lambda: dst0.copy_(cpu_pinned, non_blocking=True),
                    repeats,
                    warmup,
                ),
                available=True,
                note="pinned host to GPU0 copy",
            )
        )
        rows.append(
            summarize_samples(
                path_name="d2h_pinned",
                token_count=token_count,
                byte_count=effective_bytes,
                samples_ms=_cuda_timed_copy(
                    torch,
                    lambda: cpu_pinned.copy_(src0, non_blocking=True),
                    repeats,
                    warmup,
                ),
                available=True,
                note="GPU0 to pinned host copy",
            )
        )
        rows.append(
            summarize_samples(
                path_name="d2d_local",
                token_count=token_count,
                byte_count=effective_bytes,
                samples_ms=_cuda_timed_copy(
                    torch,
                    lambda: dst0.copy_(src0, non_blocking=True),
                    repeats,
                    warmup,
                ),
                available=True,
                note="GPU0 local device-to-device copy",
            )
        )

    if torch.cuda.device_count() >= 2:
        src1 = torch.empty(element_count, device="cuda:1", dtype=dtype_obj)
        src1.fill_(0)
        can_peer = bool(torch.cuda.can_device_access_peer(1, 0))
        samples = _cuda_timed_copy(
            torch,
            lambda: src1.copy_(src0, non_blocking=True),
            repeats,
            warmup,
        )
        rows.append(
            summarize_samples(
                path_name="p2p_gpu0_to_gpu1",
                token_count=token_count,
                byte_count=effective_bytes,
                samples_ms=samples,
                available=True,
                note=f"GPU0 to GPU1 copy; cuda_can_access_peer={str(can_peer).lower()}",
            )
        )
    else:
        rows.append(
            summarize_samples(
                path_name="p2p_gpu0_to_gpu1",
                token_count=token_count,
                byte_count=effective_bytes,
                samples_ms=[],
                available=False,
                note="requires at least two visible CUDA devices",
            )
        )
    return rows


def run_microbench(
    *,
    token_counts: list[int],
    bytes_per_token: int,
    repeats: int,
    warmup: int,
    dtype: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": str(torch.__version__),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "dtype": dtype,
        "bytes_per_token": int(bytes_per_token),
        "token_counts": token_counts,
        "repeats": int(repeats),
        "warmup": int(warmup),
        "evidence_tier": "sglang-small-physical-copy-smoke",
        "note": (
            "Real CUDA copy microbench for KV-sized tensors. This calibrates "
            "physical movement costs but is not full SGLang end-to-end policy evidence."
        ),
    }
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
    meta["devices"] = devices

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not available inside this allocation")

    rows: list[dict[str, Any]] = []
    for token_count in token_counts:
        rows.extend(
            _bench_one_size(
                torch,
                token_count=token_count,
                bytes_per_token=bytes_per_token,
                repeats=repeats,
                warmup=warmup,
                dtype=dtype,
            )
        )
    for row in rows:
        row.update(
            {
                "bytes_per_token": int(bytes_per_token),
                "dtype": dtype,
                "slurm_job_id": meta["slurm_job_id"],
                "host": meta["host"],
                "evidence_tier": meta["evidence_tier"],
                "source": "torch_cuda_physical_kv_copy_microbench",
            }
        )
    return rows, meta


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        "# Physical KV Copy Microbenchmark",
        "",
        "This is a small-scale CUDA copy calibration artifact for AegisKV.",
        "It validates physical movement costs separately from metadata-only SGLSim restoration.",
        "",
        "## Metadata",
        "",
        f"- host: `{meta.get('host', '')}`",
        f"- Slurm job: `{meta.get('slurm_job_id', '')}`",
        f"- CUDA_VISIBLE_DEVICES: `{meta.get('cuda_visible_devices', '')}`",
        f"- torch: `{meta.get('torch_version', '')}`",
        f"- devices: `{meta.get('torch_cuda_device_count', 0)}`",
        f"- bytes per token assumption: `{meta.get('bytes_per_token', '')}`",
        f"- repeats: `{meta.get('repeats', '')}`; warmup: `{meta.get('warmup', '')}`",
        "",
        "## Results",
        "",
        "| Path | Tokens | MiB | p50 ms | p95 ms | p50 GB/s | Available |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {path} | {tokens} | {mib:.1f} | {p50:.3f} | {p95:.3f} | {gbps:.2f} | {avail} |".format(
                path=row.get("path", ""),
                tokens=int(row.get("tokens") or 0),
                mib=float(row.get("mib") or 0.0),
                p50=float(row.get("p50_ms") or 0.0),
                p95=float(row.get("p95_ms") or 0.0),
                gbps=float(row.get("p50_gbps") or 0.0),
                avail=row.get("available", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `h2d_pinned` and `d2h_pinned` calibrate HostBackup restore/write-through movement.",
            "- `p2p_gpu0_to_gpu1` calibrates peer GPUReplica movement when two CUDA devices are visible.",
            "- This artifact is physical copy evidence, but it is still a microbenchmark, not final end-to-end AegisKV policy evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_token_counts(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("token count list cannot be empty")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("token counts must be positive")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark physical CUDA KV-sized H2D/D2H/P2P copies."
    )
    parser.add_argument(
        "--token-counts",
        type=parse_token_counts,
        default=list(DEFAULT_TOKEN_COUNTS),
        help="Comma-separated token counts to benchmark.",
    )
    parser.add_argument("--bytes-per-token", type=int, default=DEFAULT_BYTES_PER_TOKEN)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, meta = run_microbench(
        token_counts=list(args.token_counts),
        bytes_per_token=int(args.bytes_per_token),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        dtype=str(args.dtype),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "physical_kv_copy_microbench.csv", rows)
    (args.output_dir / "physical_kv_copy_microbench.json").write_text(
        json.dumps({"metadata": meta, "rows": rows}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "physical_kv_copy_microbench.md").write_text(
        render_markdown(rows, meta),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
