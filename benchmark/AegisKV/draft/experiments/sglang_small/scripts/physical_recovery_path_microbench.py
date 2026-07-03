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
from typing import Any, Callable


DEFAULT_TOKEN_COUNTS = (512, 1024, 2048, 4096, 8192)
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
    action: str,
    recovery_path: str,
    phase: str,
    copy_path: str,
    token_count: int,
    byte_count: int,
    samples_ms: list[float],
    available: bool,
    source_location: str,
    target_location: str,
    source_device: int | str = "",
    target_device: int | str = "",
    note: str = "",
) -> dict[str, Any]:
    p50_ms = _percentile(samples_ms, 0.50)
    p95_ms = _percentile(samples_ms, 0.95)
    return {
        "action": action,
        "recovery_path": recovery_path,
        "phase": phase,
        "copy_path": copy_path,
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
        "source_location": source_location,
        "target_location": target_location,
        "source_device": source_device,
        "target_device": target_device,
        "is_restore_action": str(phase == "restore").lower(),
        "note": note,
    }


def _cuda_timed_copy(
    torch: Any,
    fn: Callable[[], None],
    *,
    repeats: int,
    warmup: int,
    timing_device: int,
) -> list[float]:
    samples: list[float] = []
    with torch.cuda.device(timing_device):
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


def _summarize_unavailable(
    *,
    action: str,
    recovery_path: str,
    phase: str,
    copy_path: str,
    token_count: int,
    byte_count: int,
    source_location: str,
    target_location: str,
    source_device: int | str = "",
    target_device: int | str = "",
    note: str,
) -> dict[str, Any]:
    return summarize_samples(
        action=action,
        recovery_path=recovery_path,
        phase=phase,
        copy_path=copy_path,
        token_count=token_count,
        byte_count=byte_count,
        samples_ms=[],
        available=False,
        source_location=source_location,
        target_location=target_location,
        source_device=source_device,
        target_device=target_device,
        note=note,
    )


def _measure_action(
    torch: Any,
    *,
    action: str,
    recovery_path: str,
    phase: str,
    copy_path: str,
    token_count: int,
    byte_count: int,
    repeats: int,
    warmup: int,
    timing_device: int,
    fn: Callable[[], None],
    source_location: str,
    target_location: str,
    source_device: int | str = "",
    target_device: int | str = "",
    note: str = "",
) -> dict[str, Any]:
    try:
        samples = _cuda_timed_copy(
            torch,
            fn,
            repeats=repeats,
            warmup=warmup,
            timing_device=timing_device,
        )
        return summarize_samples(
            action=action,
            recovery_path=recovery_path,
            phase=phase,
            copy_path=copy_path,
            token_count=token_count,
            byte_count=byte_count,
            samples_ms=samples,
            available=True,
            source_location=source_location,
            target_location=target_location,
            source_device=source_device,
            target_device=target_device,
            note=note,
        )
    except RuntimeError as exc:
        return _summarize_unavailable(
            action=action,
            recovery_path=recovery_path,
            phase=phase,
            copy_path=copy_path,
            token_count=token_count,
            byte_count=byte_count,
            source_location=source_location,
            target_location=target_location,
            source_device=source_device,
            target_device=target_device,
            note=f"copy failed: {exc}",
        )


def _bench_one_size(
    torch: Any,
    *,
    token_count: int,
    bytes_per_token: int,
    repeats: int,
    warmup: int,
    dtype: str,
    source_device: int,
    target_device: int,
) -> list[dict[str, Any]]:
    byte_count = int(token_count) * int(bytes_per_token)
    dtype_obj = getattr(torch, dtype)
    element_size = torch.empty((), dtype=dtype_obj).element_size()
    element_count = max(1, math.ceil(byte_count / element_size))
    effective_bytes = element_count * element_size

    rows: list[dict[str, Any]] = []
    with torch.cuda.device(source_device):
        source_kv = torch.empty(element_count, device=f"cuda:{source_device}", dtype=dtype_obj)
        source_kv.fill_(2)
        host_backup = torch.empty(element_count, device="cpu", dtype=dtype_obj, pin_memory=True)
        host_backup.fill_(1)

    with torch.cuda.device(target_device):
        retry_kv = torch.empty(element_count, device=f"cuda:{target_device}", dtype=dtype_obj)
        retry_kv.fill_(0)
        local_stage = torch.empty_like(retry_kv)
        local_stage.fill_(3)

    rows.append(
        _measure_action(
            torch,
            action="host_backup_write_d2h",
            recovery_path="HostBackup",
            phase="backup",
            copy_path="d2h_pinned",
            token_count=token_count,
            byte_count=effective_bytes,
            repeats=repeats,
            warmup=warmup,
            timing_device=source_device,
            fn=lambda: host_backup.copy_(source_kv, non_blocking=True),
            source_location="failed_rank_gpu_kv",
            target_location="pinned_host_backup",
            source_device=source_device,
            target_device="host",
            note="pre-failure write-through copy from active KV to pinned HostBackup",
        )
    )
    rows.append(
        _measure_action(
            torch,
            action="host_restore_h2d",
            recovery_path="HostBackup",
            phase="restore",
            copy_path="h2d_pinned",
            token_count=token_count,
            byte_count=effective_bytes,
            repeats=repeats,
            warmup=warmup,
            timing_device=target_device,
            fn=lambda: retry_kv.copy_(host_backup, non_blocking=True),
            source_location="pinned_host_backup",
            target_location="retry_rank_gpu_kv",
            source_device="host",
            target_device=target_device,
            note="recovery-shaped HostBackup restore into retry KV buffer",
        )
    )
    rows.append(
        _measure_action(
            torch,
            action="local_retry_buffer_copy_d2d",
            recovery_path="LocalReplay",
            phase="restore",
            copy_path="d2d_local",
            token_count=token_count,
            byte_count=effective_bytes,
            repeats=repeats,
            warmup=warmup,
            timing_device=target_device,
            fn=lambda: retry_kv.copy_(local_stage, non_blocking=True),
            source_location="retry_rank_local_stage",
            target_location="retry_rank_gpu_kv",
            source_device=target_device,
            target_device=target_device,
            note="local retry-rank materialization baseline; not a remote survival path",
        )
    )

    if torch.cuda.device_count() >= 2 and source_device != target_device:
        can_peer = bool(torch.cuda.can_device_access_peer(target_device, source_device))
        peer_note = (
            "recovery-shaped peer GPUReplica movement into retry KV buffer; "
            f"cuda_can_access_peer={str(can_peer).lower()}"
        )
        rows.append(
            _measure_action(
                torch,
                action="gpu_replica_write_p2p",
                recovery_path="GPUReplica",
                phase="backup",
                copy_path=f"p2p_gpu{source_device}_to_gpu{target_device}",
                token_count=token_count,
                byte_count=effective_bytes,
                repeats=repeats,
                warmup=warmup,
                timing_device=target_device,
                fn=lambda: retry_kv.copy_(source_kv, non_blocking=True),
                source_location="active_rank_gpu_kv",
                target_location="peer_gpu_replica",
                source_device=source_device,
                target_device=target_device,
                note=(
                    "pre-failure replica update to surviving peer GPU; "
                    f"cuda_can_access_peer={str(can_peer).lower()}"
                ),
            )
        )
        rows.append(
            _measure_action(
                torch,
                action="gpu_replica_restore_p2p",
                recovery_path="GPUReplica",
                phase="restore",
                copy_path=f"p2p_gpu{source_device}_to_gpu{target_device}",
                token_count=token_count,
                byte_count=effective_bytes,
                repeats=repeats,
                warmup=warmup,
                timing_device=target_device,
                fn=lambda: retry_kv.copy_(source_kv, non_blocking=True),
                source_location="surviving_peer_gpu_replica",
                target_location="retry_rank_gpu_kv",
                source_device=source_device,
                target_device=target_device,
                note=peer_note,
            )
        )
    else:
        for action, phase, source_location, target_location in (
            (
                "gpu_replica_write_p2p",
                "backup",
                "active_rank_gpu_kv",
                "peer_gpu_replica",
            ),
            (
                "gpu_replica_restore_p2p",
                "restore",
                "surviving_peer_gpu_replica",
                "retry_rank_gpu_kv",
            ),
        ):
            rows.append(
                _summarize_unavailable(
                    action=action,
                    recovery_path="GPUReplica",
                    phase=phase,
                    copy_path=f"p2p_gpu{source_device}_to_gpu{target_device}",
                    token_count=token_count,
                    byte_count=effective_bytes,
                    source_location=source_location,
                    target_location=target_location,
                    source_device=source_device,
                    target_device=target_device,
                    note="requires at least two distinct visible CUDA devices",
                )
            )

    del source_kv, host_backup, retry_kv, local_stage
    torch.cuda.empty_cache()
    return rows


def run_microbench(
    *,
    token_counts: list[int],
    bytes_per_token: int,
    repeats: int,
    warmup: int,
    dtype: str,
    source_device: int,
    target_device: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not available inside this allocation")

    device_count = int(torch.cuda.device_count())
    resolved_target = int(target_device) if target_device is not None else (1 if device_count >= 2 else 0)
    if source_device < 0 or source_device >= device_count:
        raise ValueError(f"source device {source_device} is outside visible CUDA range")
    if resolved_target < 0 or resolved_target >= device_count:
        raise ValueError(f"target device {resolved_target} is outside visible CUDA range")

    peer_access: list[dict[str, Any]] = []
    for dst in range(device_count):
        for src in range(device_count):
            peer_access.append(
                {
                    "source": src,
                    "target": dst,
                    "can_access_peer": bool(torch.cuda.can_device_access_peer(dst, src)),
                }
            )

    devices = []
    for index in range(device_count):
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

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": str(torch.__version__),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": device_count,
        "dtype": dtype,
        "bytes_per_token": int(bytes_per_token),
        "token_counts": token_counts,
        "repeats": int(repeats),
        "warmup": int(warmup),
        "source_device": int(source_device),
        "target_device": int(resolved_target),
        "devices": devices,
        "peer_access": peer_access,
        "evidence_tier": "sglang-small-physical-recovery-movement-smoke",
        "note": (
            "Real CUDA copy microbench shaped as AegisKV backup/restore actions. "
            "It writes into retry-rank KV buffers, but remains an action-level "
            "microbenchmark rather than full SGLang end-to-end failure recovery."
        ),
    }

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
                source_device=source_device,
                target_device=resolved_target,
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
                "source": "torch_cuda_physical_recovery_path_microbench",
            }
        )
    return rows, meta


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        "# Physical Recovery Path Microbenchmark",
        "",
        "This is a small-scale CUDA movement artifact shaped like AegisKV recovery actions.",
        "It validates real tensor movement into retry KV buffers, not just raw bandwidth.",
        "",
        "## Metadata",
        "",
        f"- host: `{meta.get('host', '')}`",
        f"- Slurm job: `{meta.get('slurm_job_id', '')}`",
        f"- CUDA_VISIBLE_DEVICES: `{meta.get('cuda_visible_devices', '')}`",
        f"- torch: `{meta.get('torch_version', '')}`",
        f"- devices: `{meta.get('torch_cuda_device_count', 0)}`",
        f"- source/target devices: `{meta.get('source_device', '')}` -> `{meta.get('target_device', '')}`",
        f"- bytes per token assumption: `{meta.get('bytes_per_token', '')}`",
        f"- repeats: `{meta.get('repeats', '')}`; warmup: `{meta.get('warmup', '')}`",
        "",
        "## Results",
        "",
        "| Action | Tokens | Phase | Copy path | MiB | p50 ms | p95 ms | p50 GB/s | Available |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {action} | {tokens} | {phase} | {copy_path} | {mib:.1f} | {p50:.3f} | {p95:.3f} | {gbps:.2f} | {avail} |".format(
                action=row.get("action", ""),
                tokens=int(row.get("tokens") or 0),
                phase=row.get("phase", ""),
                copy_path=row.get("copy_path", ""),
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
            "- `host_restore_h2d` is the physical HostBackup restore action into the retry KV buffer.",
            "- `gpu_replica_restore_p2p` is the physical GPUReplica restore action into the retry KV buffer.",
            "- `host_backup_write_d2h` and `gpu_replica_write_p2p` capture steady-state backup movement costs.",
            "- This artifact is stronger than raw copy calibration, but still not final end-to-end AegisKV policy evidence.",
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
        description="Benchmark physical CUDA movement shaped as AegisKV recovery actions."
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
    parser.add_argument("--source-device", type=int, default=0)
    parser.add_argument("--target-device", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, meta = run_microbench(
        token_counts=list(args.token_counts),
        bytes_per_token=int(args.bytes_per_token),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        dtype=str(args.dtype),
        source_device=int(args.source_device),
        target_device=args.target_device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "physical_recovery_path_microbench.csv", rows)
    (args.output_dir / "physical_recovery_path_microbench.json").write_text(
        json.dumps({"metadata": meta, "rows": rows}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "physical_recovery_path_microbench.md").write_text(
        render_markdown(rows, meta),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
