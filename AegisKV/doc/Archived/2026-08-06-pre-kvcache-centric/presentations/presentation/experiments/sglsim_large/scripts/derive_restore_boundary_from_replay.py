#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[6]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
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


def replay_rows(run_dirs: list[Path]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for run_dir in run_dirs:
        rows = read_csv(run_dir / "request_details.csv")
        out.extend(row for row in rows if _bool(row.get("aegis_failover_replay")))
    if not out:
        raise ValueError("no replay rows found")
    return out


def row_pre_failure_tokens(row: dict[str, str]) -> float:
    tokens = _float(row.get("aegis_candidate_pre_failure_total_tokens"))
    if tokens > 0:
        return tokens
    return _float(row.get("input_length")) + _float(
        row.get("aegis_candidate_pre_failure_output_tokens")
    )


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("linear fit needs at least two points")
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_bar = mean(xs)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom <= 0:
        raise ValueError("linear fit has zero x variance")
    slope = sum((x - x_bar) * (y - y_bar) for x, y in points) / denom
    intercept = y_bar - slope * x_bar
    return intercept, slope


def selected_overhead_ms(row: dict[str, str]) -> float:
    value = _float(row.get("aegis_replay_restore_overhead_s")) * 1000.0
    if value <= 0:
        value = _float(row.get("aegis_replay_restore_sleep_s")) * 1000.0
    return value


def fit_models(
    *,
    host_rows: list[dict[str, str]],
    gpu_rows: list[dict[str, str]],
    recompute_rows: list[dict[str, str]],
) -> dict[str, float]:
    recompute_points = [
        (row_pre_failure_tokens(row), selected_overhead_ms(row))
        for row in recompute_rows
    ]
    recompute_intercept, recompute_slope = linear_fit(recompute_points)

    host_points = []
    for row in host_rows:
        restored = _float(row.get("aegis_replay_restored_tokens"))
        recompute = _float(row.get("aegis_replay_recompute_tokens"))
        y = selected_overhead_ms(row) - recompute_slope * recompute
        if restored > 0:
            host_points.append((restored, y))
    host_intercept, host_slope = linear_fit(host_points)

    gpu_points = []
    gpu_coverages = []
    for row in gpu_rows:
        restored = _float(row.get("aegis_replay_restored_tokens"))
        recompute = _float(row.get("aegis_replay_recompute_tokens"))
        total = max(row_pre_failure_tokens(row), restored + recompute)
        y = selected_overhead_ms(row) - recompute_slope * recompute
        if restored > 0:
            gpu_points.append((restored, y))
            if total > 0:
                gpu_coverages.append(restored / total)
    gpu_intercept, gpu_slope = linear_fit(gpu_points)
    gpu_coverage = mean(gpu_coverages) if gpu_coverages else 1.0

    return {
        "recompute_intercept_ms": recompute_intercept,
        "recompute_ms_per_token": recompute_slope,
        "host_intercept_ms": host_intercept,
        "host_ms_per_token": host_slope,
        "gpu_intercept_ms": gpu_intercept,
        "gpu_ms_per_token": gpu_slope,
        "gpu_observed_coverage": gpu_coverage,
        "host_rows": float(len(host_rows)),
        "gpu_rows": float(len(gpu_rows)),
        "recompute_rows": float(len(recompute_rows)),
    }


def parse_token_points(value: str) -> list[int]:
    points = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not points:
        raise argparse.ArgumentTypeError("token points must not be empty")
    if any(point <= 0 for point in points):
        raise argparse.ArgumentTypeError("token points must be positive")
    return sorted(dict.fromkeys(points))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 2 restore/recompute boundary from direct replay "
            "restore-overhead rows."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--host-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--gpu-run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--recompute-run-dir", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--token-points",
        type=parse_token_points,
        default=parse_token_points("128,256,512,1024,2048,4096,8192,16384"),
    )
    args = parser.parse_args()

    host_rows = [
        row
        for row in replay_rows(args.host_run_dir)
        if str(row.get("aegis_replay_restore_action")) == "host_restore"
    ]
    gpu_rows = [
        row
        for row in replay_rows(args.gpu_run_dir)
        if str(row.get("aegis_replay_restore_action")) == "gpu_replica"
    ]
    recompute_rows = [
        row
        for row in replay_rows(args.recompute_run_dir)
        if str(row.get("aegis_replay_restore_action")) == "recompute"
    ]
    if not host_rows or not gpu_rows or not recompute_rows:
        raise ValueError(
            "host, gpu, and recompute replay rows are all required for calibration"
        )

    model = fit_models(
        host_rows=host_rows,
        gpu_rows=gpu_rows,
        recompute_rows=recompute_rows,
    )
    current_sha = git_sha()
    source_dirs = [
        *[str(path) for path in args.host_run_dir],
        *[str(path) for path in args.gpu_run_dir],
        *[str(path) for path in args.recompute_run_dir],
    ]
    rows: list[dict[str, object]] = []
    for tokens in args.token_points:
        recompute = model["recompute_intercept_ms"] + (
            model["recompute_ms_per_token"] * tokens
        )
        host = model["host_intercept_ms"] + model["host_ms_per_token"] * tokens
        gpu_restored = model["gpu_observed_coverage"] * tokens
        gpu_recompute = max(tokens - gpu_restored, 0.0)
        gpu = (
            model["gpu_intercept_ms"]
            + model["gpu_ms_per_token"] * gpu_restored
            + model["recompute_ms_per_token"] * gpu_recompute
        )
        rows.append(
            {
                "tokens": tokens,
                "recompute_ms": recompute,
                "host_restore_ms": host,
                "gpu_replica_ms": gpu,
                "aegis_lower_envelope_ms": min(recompute, host, gpu),
                "evidence_tier": "sglsim-runtime-replay-calibrated",
                "source": "sglsim_request_details_replay_restore_boundary",
                "run_id": args.run_id,
                "git_sha": current_sha,
                "repeat": f"calibrated-from-{len(source_dirs)}-runs",
                "repeat_count": len(source_dirs),
                "repeat_run_ids": ";".join(source_dirs),
                "note": (
                    "Curves are calibrated from direct hard-failure replay "
                    "restore-overhead rows; GPUReplica curve uses observed "
                    "peer-tail coverage plus suffix recompute"
                ),
                **model,
                "runtime_replay_rows": len(host_rows) + len(gpu_rows) + len(recompute_rows),
                "runtime_regimes": "long_prefix_host;short_peer_gpu;short_nonpeer_recompute",
            }
        )

    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "model": model,
                "rows": len(rows),
                "runtime_replay_rows": len(host_rows)
                + len(gpu_rows)
                + len(recompute_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
