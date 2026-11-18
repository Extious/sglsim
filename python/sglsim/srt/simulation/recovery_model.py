from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_RECOMPUTE_SETUP_MS = 4.0
DEFAULT_RECOMPUTE_MS_PER_TOKEN = 0.0108
DEFAULT_HOST_SETUP_MS = 7.5
DEFAULT_HOST_MS_PER_TOKEN = 0.0036
DEFAULT_GPU_SETUP_MS = 2.4
DEFAULT_GPU_MS_PER_TOKEN = 0.0019

ENV_RECOVERY_MODEL = "SGLSIM_AEGIS_RECOVERY_MODEL"
ENV_PHYSICAL_COPY_CSV = "SGLSIM_AEGIS_PHYSICAL_COPY_CSV"


@dataclass(frozen=True)
class CopyPoint:
    tokens: int
    p50_ms: float


def _default_restore_overhead_ms(
    *, action: str, token_count: int, restored_tokens: int
) -> float:
    token_count = max(int(token_count), 0)
    restored_tokens = max(int(restored_tokens), 0)
    recompute_tokens = max(token_count - restored_tokens, 0)
    if action == "recompute" or restored_tokens <= 0:
        return DEFAULT_RECOMPUTE_SETUP_MS + token_count * DEFAULT_RECOMPUTE_MS_PER_TOKEN
    if action in {"gpu_replica", "l1_backup"}:
        return (
            DEFAULT_GPU_SETUP_MS
            + restored_tokens * DEFAULT_GPU_MS_PER_TOKEN
            + recompute_tokens * DEFAULT_RECOMPUTE_MS_PER_TOKEN
        )
    if action == "l1_l2_backup":
        # The rank-local controller supplies the exact combined timing.  This
        # fallback conservatively uses the L2 path when called independently.
        action = "l2_backup"
    return (
        DEFAULT_HOST_SETUP_MS
        + restored_tokens * DEFAULT_HOST_MS_PER_TOKEN
        + recompute_tokens * DEFAULT_RECOMPUTE_MS_PER_TOKEN
    )


def _interpolated_copy_ms(points: list[CopyPoint], tokens: int) -> float:
    tokens = max(int(tokens), 0)
    if tokens <= 0 or not points:
        return 0.0
    if len(points) == 1:
        point = points[0]
        return point.p50_ms * tokens / max(point.tokens, 1)
    points = sorted(points, key=lambda point: point.tokens)
    if tokens <= points[0].tokens:
        return points[0].p50_ms * tokens / max(points[0].tokens, 1)
    for left, right in zip(points, points[1:]):
        if tokens <= right.tokens:
            span = max(right.tokens - left.tokens, 1)
            ratio = (tokens - left.tokens) / span
            return left.p50_ms + ratio * (right.p50_ms - left.p50_ms)
    left, right = points[-2], points[-1]
    slope = (right.p50_ms - left.p50_ms) / max(right.tokens - left.tokens, 1)
    return right.p50_ms + (tokens - right.tokens) * slope


@lru_cache(maxsize=8)
def _load_physical_copy_table(csv_path: str) -> dict[str, tuple[CopyPoint, ...]]:
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    by_path: dict[str, list[CopyPoint]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            action = str(row.get("action") or "")
            if action:
                if action == "host_restore_h2d":
                    path_name = "h2d_pinned"
                elif action == "gpu_replica_restore_p2p":
                    path_name = "p2p_gpu0_to_gpu1"
                else:
                    continue
            else:
                path_name = str(row.get("path") or row.get("copy_path") or "")
            if path_name not in {"h2d_pinned", "p2p_gpu0_to_gpu1"}:
                continue
            try:
                point = CopyPoint(
                    tokens=int(float(row["tokens"])),
                    p50_ms=float(row["p50_ms_mean"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            by_path.setdefault(path_name, []).append(point)
    return {
        path_name: tuple(sorted(points, key=lambda point: point.tokens))
        for path_name, points in by_path.items()
    }


@lru_cache(maxsize=8)
def _physical_copy_table_kind(csv_path: str) -> str:
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return "missing_physical_copy_csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return "empty_physical_copy_csv"
        fields = set(reader.fieldnames)
        if "action" in fields:
            return "physical_recovery_path_csv"
        if "path" in fields:
            return "physical_copy_csv"
    return "unknown_physical_copy_csv"


def aegis_restore_timing_source() -> str:
    model = os.getenv(ENV_RECOVERY_MODEL, "default").strip().lower()
    if model not in {"physical_copy", "physical-copy"}:
        return "default_coefficients"
    table_path = os.getenv(ENV_PHYSICAL_COPY_CSV, "").strip()
    if not table_path:
        return "physical_copy_model_missing_csv"
    return _physical_copy_table_kind(table_path)


def aegis_restore_data_plane(action: str) -> str:
    action = str(action or "").strip().lower()
    if action in {
        "host_restore",
        "gpu_replica",
        "l1_backup",
        "l2_backup",
        "l1_l2_backup",
    }:
        return "metadata_materialization"
    if action == "recompute":
        return "recompute"
    return action or "none"


def aegis_restore_overhead_ms(
    *, action: str, token_count: int, restored_tokens: int
) -> float:
    model = os.getenv(ENV_RECOVERY_MODEL, "default").strip().lower()
    if model not in {"physical_copy", "physical-copy"}:
        return _default_restore_overhead_ms(
            action=action,
            token_count=token_count,
            restored_tokens=restored_tokens,
        )

    token_count = max(int(token_count), 0)
    restored_tokens = max(int(restored_tokens), 0)
    recompute_tokens = max(token_count - restored_tokens, 0)
    if action == "recompute" or restored_tokens <= 0:
        return _default_restore_overhead_ms(
            action="recompute",
            token_count=token_count,
            restored_tokens=0,
        )

    table_path = os.getenv(ENV_PHYSICAL_COPY_CSV, "").strip()
    table = _load_physical_copy_table(table_path) if table_path else {}
    if not table:
        return _default_restore_overhead_ms(
            action=action,
            token_count=token_count,
            restored_tokens=restored_tokens,
        )

    copy_path = (
        "p2p_gpu0_to_gpu1"
        if action in {"gpu_replica", "l1_backup"}
        else "h2d_pinned"
    )
    copy_points = list(table.get(copy_path, ()))
    if not copy_points:
        return _default_restore_overhead_ms(
            action=action,
            token_count=token_count,
            restored_tokens=restored_tokens,
        )

    setup_ms = (
        DEFAULT_GPU_SETUP_MS
        if action in {"gpu_replica", "l1_backup"}
        else DEFAULT_HOST_SETUP_MS
    )
    return (
        setup_ms
        + _interpolated_copy_ms(copy_points, restored_tokens)
        + recompute_tokens * DEFAULT_RECOMPUTE_MS_PER_TOKEN
    )


def aegis_restore_overhead_s(
    *, action: str, token_count: int, restored_tokens: int
) -> float:
    return aegis_restore_overhead_ms(
        action=action,
        token_count=token_count,
        restored_tokens=restored_tokens,
    ) / 1000.0
