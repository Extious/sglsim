#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)")
DECODE_RE = re.compile(
    r"Decode batch, #running-req: (\d+).*gen throughput \(token/s\): ([\d.]+)"
)


@dataclass(frozen=True)
class PhaseMetrics:
    prefill_wall_s: float
    prefill_batches: int
    decode_wall_s: float
    decode_logs: int
    first_decode_thr_tok_s: float


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


def parse_ts(line: str) -> datetime | None:
    match = TS_RE.match(line)
    if not match:
        return None
    text = match.group(1)
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in text else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(text, fmt)


def phase_metrics(log_path: Path, dp: int = 0) -> PhaseMetrics:
    events: list[tuple[datetime, str, int, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if f" DP{dp}]" not in line:
            continue
        ts = parse_ts(line)
        if ts is None:
            continue
        if "Prefill batch" in line:
            events.append((ts, "pre", -1, 0.0))
            continue
        match = DECODE_RE.search(line)
        if match:
            running = int(match.group(1))
            throughput = float(match.group(2))
            if throughput > 10.0:
                events.append((ts, "dec", running, throughput))
    dec10_idx = next(
        (i for i, event in enumerate(events) if event[1] == "dec" and event[2] == 10),
        None,
    )
    if dec10_idx is None:
        return PhaseMetrics(0.0, 0, 0.0, 0, 0.0)
    prefill_phase = events[:dec10_idx]
    decode_phase = events[dec10_idx : dec10_idx + 26]
    prefill_wall = (
        (prefill_phase[-1][0] - prefill_phase[0][0]).total_seconds()
        if len(prefill_phase) > 1
        else 0.0
    )
    decode_wall = (
        (decode_phase[-1][0] - decode_phase[0][0]).total_seconds()
        if len(decode_phase) > 1
        else 0.0
    )
    return PhaseMetrics(
        prefill_wall_s=prefill_wall,
        prefill_batches=sum(1 for event in prefill_phase if event[1] == "pre"),
        decode_wall_s=decode_wall,
        decode_logs=sum(1 for event in decode_phase if event[1] == "dec"),
        first_decode_thr_tok_s=events[dec10_idx][3],
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def values(rows: list[dict[str, str]], column: str) -> list[float]:
    output = []
    for row in rows:
        value = row.get(column)
        if value in ("", None):
            continue
        output.append(float(value))
    return output


def percentile(items: list[float], pct: float) -> float:
    if not items:
        return 0.0
    ordered = sorted(items)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive SGLang/SGLSim synthesis parity accuracy anchors."
    )
    parser.add_argument("--sglsim-dir", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    sim_rows = read_rows(args.sglsim_dir / "request_details.csv")
    sglang_rows = read_rows(args.sglang_dir / "request_details.csv")
    sim_phase = phase_metrics(args.sglsim_dir / "server.log")
    sglang_phase = phase_metrics(args.sglang_dir / "server.log")
    sglang_metrics_path = args.sglang_dir / "metrics.json"
    sglang_metrics = (
        json.loads(sglang_metrics_path.read_text(encoding="utf-8"))
        if sglang_metrics_path.exists()
        else {}
    )

    sim_e2e = values(sim_rows, "e2e_latency_s")
    sglang_e2e = values(sglang_rows, "e2e_latency_s")
    sim_cache_hit = sum(values(sim_rows, "cache_hit_tokens"))
    sglang_cache_hit = sum(values(sglang_rows, "cache_hit_tokens"))
    current_sha = git_sha()

    raw_metrics = [
        ("prefill wall", "s", sim_phase.prefill_wall_s, sglang_phase.prefill_wall_s),
        ("decode wall", "s", sim_phase.decode_wall_s, sglang_phase.decode_wall_s),
        (
            "first decode throughput",
            "tok/s",
            sim_phase.first_decode_thr_tok_s,
            sglang_phase.first_decode_thr_tok_s,
        ),
        ("e2e mean", "s", statistics.mean(sim_e2e), statistics.mean(sglang_e2e)),
        ("e2e p50", "s", percentile(sim_e2e, 50), percentile(sglang_e2e, 50)),
        ("e2e p90", "s", percentile(sim_e2e, 90), percentile(sglang_e2e, 90)),
        ("e2e p99", "s", percentile(sim_e2e, 99), percentile(sglang_e2e, 99)),
        ("cache hit tokens", "tokens", sim_cache_hit, sglang_cache_hit),
    ]

    output_rows: list[dict[str, object]] = []
    for metric, unit, sim_value, sglang_value in raw_metrics:
        ratio = sim_value / sglang_value if sglang_value else 0.0
        output_rows.append(
            {
                "metric": metric,
                "unit": unit,
                "sglang_value": sglang_value,
                "sglsim_value": sim_value,
                "ratio": ratio,
                "rel_error_pct": abs(ratio - 1.0) * 100.0,
                "within_10pct": abs(ratio - 1.0) <= 0.10,
                "sglsim_run": str(args.sglsim_dir),
                "sglang_run": str(args.sglang_dir),
                "sglang_completed": sglang_metrics.get("completed", len(sglang_rows)),
                "sglsim_completed": len(sim_rows),
                "evidence_tier": "sglang-sglsim-synthesis-parity",
                "source": "synthesis_nohicache_fa3_parity",
                "run_id": args.run_id,
                "git_sha": current_sha,
                "workload_id": "synthesis_nohicache_fa3",
                "note": (
                    "Direct SGLang fa3 no-HiCache inference run compared with "
                    "the matching SGLSim wall-clock run. The figure reports "
                    "SGLSim/SGLang ratios; it is an accuracy anchor for the "
                    "base serving simulator rather than AegisKV failure-policy "
                    "end-to-end evidence."
                ),
            }
        )
    write_csv(args.output_csv, output_rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(output_rows),
                "max_rel_error_pct": max(
                    float(row["rel_error_pct"]) for row in output_rows
                ),
                "within_10pct": sum(
                    1 for row in output_rows if bool(row["within_10pct"])
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
