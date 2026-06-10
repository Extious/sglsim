"""Decompose sim vs real makespan into prefill / decode / queue phases per wave."""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)")
_PREFILL_RE = re.compile(r"Prefill batch")
_DECODE_RE = re.compile(
    r"Decode batch, #running-req: (\d+).*gen throughput \(token/s\): ([\d.]+)"
)


@dataclass
class PhaseMetrics:
    prefill_wall_s: float
    prefill_batches: int
    decode_wall_s: float
    decode_logs: int
    first_decode_thr: float


def _parse_ts(line: str) -> datetime | None:
    match = _TS_RE.match(line)
    if not match:
        return None
    text = match.group(1)
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in text else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(text, fmt)


def _phase_metrics(log_path: Path, dp: int = 0) -> PhaseMetrics:
    events: list[tuple[datetime, str, int, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if f" DP{dp}]" not in line:
            continue
        ts = _parse_ts(line)
        if ts is None:
            continue
        if _PREFILL_RE.search(line):
            events.append((ts, "pre", -1, 0.0))
            continue
        match = _DECODE_RE.search(line)
        if match:
            running = int(match.group(1))
            thr = float(match.group(2))
            if thr > 10:
                events.append((ts, "dec", running, thr))
    if not events:
        return PhaseMetrics(0.0, 0, 0.0, 0, 0.0)

    dec10_idx = next((i for i, e in enumerate(events) if e[1] == "dec" and e[2] == 10), None)
    if dec10_idx is None:
        return PhaseMetrics(0.0, 0, 0.0, 0, 0.0)

    prefill_phase = events[:dec10_idx]
    prefill_wall = (
        (prefill_phase[-1][0] - prefill_phase[0][0]).total_seconds()
        if len(prefill_phase) > 1
        else 0.0
    )
    decode_phase = events[dec10_idx : dec10_idx + 26]
    decode_wall = (
        (decode_phase[-1][0] - decode_phase[0][0]).total_seconds()
        if len(decode_phase) > 1
        else 0.0
    )
    return PhaseMetrics(
        prefill_wall_s=prefill_wall,
        prefill_batches=sum(1 for e in prefill_phase if e[1] == "pre"),
        decode_wall_s=decode_wall,
        decode_logs=sum(1 for e in decode_phase if e[1] == "dec"),
        first_decode_thr=events[dec10_idx][3],
    )


def _csv_summary(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    e2e = [float(r["e2e_latency_s"]) for r in rows if r.get("e2e_latency_s")]
    wait = [float(r["waiting_time_s"]) for r in rows if r.get("waiting_time_s")]
    infer = [float(r["inference_time_s"]) for r in rows if r.get("inference_time_s")]
    logical = [
        float(r["logical_service_time_s"])
        for r in rows
        if r.get("logical_service_time_s")
    ]
    return {
        "e2e_mean": statistics.mean(e2e) if e2e else 0.0,
        "wait_mean": statistics.mean(wait) if wait else 0.0,
        "infer_mean": statistics.mean(infer) if infer else 0.0,
        "logical_service_mean": statistics.mean(logical) if logical else 0.0,
    }


def main() -> None:
    sim_log = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "benchmark/synthesis_requests/logs/server.log"
    real_log = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else REPO / "benchmark/synthesis_requests/results/without_hicache/server.log"
    )
    sim_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    real_csv = Path(sys.argv[4]) if len(sys.argv) > 4 else real_log.parent / "request_details.csv"
    real_metrics = real_log.parent / "metrics.json"

    print(f"sim log : {sim_log}")
    print(f"real log: {real_log}\n")

    print(f"{'phase':<22} {'sim DP0':>12} {'real DP0':>12} {'ratio':>8}")
    sim_p = _phase_metrics(sim_log, dp=0)
    real_p = _phase_metrics(real_log, dp=0)
    for name, s_val, r_val in (
        ("prefill_wall_s", sim_p.prefill_wall_s, real_p.prefill_wall_s),
        ("prefill_batches", float(sim_p.prefill_batches), float(real_p.prefill_batches)),
        ("decode_wall_s", sim_p.decode_wall_s, real_p.decode_wall_s),
        ("decode_logs", float(sim_p.decode_logs), float(real_p.decode_logs)),
        ("first_decode_thr", sim_p.first_decode_thr, real_p.first_decode_thr),
    ):
        ratio = s_val / r_val if r_val else float("nan")
        print(f"{name:<22} {s_val:12.2f} {r_val:12.2f} {ratio:8.3f}")

    if sim_csv and sim_csv.is_file():
        sim_row = _csv_summary(sim_csv)
        real_row = _csv_summary(real_csv) if real_csv.is_file() else {}
        print("\n=== request_details.csv ===")
        print(f"{'metric':<22} {'sim':>12} {'real':>12} {'ratio':>8}")
        for key in ("e2e_mean", "wait_mean", "infer_mean", "logical_service_mean"):
            s_val = sim_row.get(key, 0.0)
            r_val = real_row.get(key, 0.0)
            ratio = s_val / r_val if r_val else float("nan")
            print(f"{key:<22} {s_val:12.2f} {r_val:12.2f} {ratio:8.3f}")

    if real_metrics.is_file():
        metrics = json.loads(real_metrics.read_text(encoding="utf-8"))
        print(f"\nreal makespan(s): {metrics.get('duration', 0):.1f}")


if __name__ == "__main__":
    main()
