"""Strict latency gate vs SGLang parity reference (default: dual-state ref via parity_refs).

Primary acceptance is behavior_gate.py (SGLang parity). Use sanity_vs_ref.py
for wide informational comparison. Do not add sim-only logic to pass this gate.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG))
from run_integrity import check_run_integrity  # noqa: E402

RATIO_LO, RATIO_HI = 0.9, 1.1
WAIT_MAX_S = 2.0
MAKESPAN_RATIO_MAX = 1.2


def _col(path: Path, name: str) -> list[float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [float(r[name]) for r in rows if r.get(name) not in (None, "")]


def _stats(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    n = len(ordered)
    mean = statistics.mean(ordered)
    p50 = ordered[n // 2]
    p90 = ordered[int(0.9 * (n - 1))]
    return mean, p50, p90


def _thrash_ok(sim_dir: Path) -> tuple[bool, str]:
    log = sim_dir / "server.log"
    if not log.is_file():
        log = REPO / "benchmark/synthesis_requests/logs/server.log"
    if not log.is_file():
        return False, "missing server.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    retract = len(re.findall(r"Retract requests", text, re.I))
    if retract > 0:
        return False, f"retract_events={retract}"
    if "Scheduler stalled" in text:
        return False, "scheduler_stalled"
    if text.count("Scheduler busy-wait") > 20:
        return False, "excessive_busy_wait"
    if re.search(r"evictable_size_\s+mismatch|evictable=-", text):
        return False, "evictable_accounting"
    return True, "ok"


def check(sim_dir: Path, ref_dir: Path) -> int:
    sim_csv = sim_dir / "request_details.csv"
    ref_csv = ref_dir / "request_details.csv"
    if not sim_csv.is_file() or not ref_csv.is_file():
        print("FAIL: missing request_details.csv")
        return 1
    integrity = check_run_integrity(sim_dir, ref_dir=ref_dir)
    if not integrity.ok:
        print("FAIL: run_integrity")
        for failure in integrity.failures:
            print(f"  - {failure}")
        return 1

    ref_e2e = _col(ref_csv, "e2e_latency_s")
    ref_mean, ref_p50, ref_p90 = _stats(ref_e2e)

    total = _col(sim_csv, "total_latency_s")
    client = _col(sim_csv, "client_wall_e2e_latency_s")
    wait = _col(sim_csv, "waiting_time_s")
    infer = _col(sim_csv, "inference_time_s")
    svc = _col(sim_csv, "logical_service_time_s")

    ref_quantiles = {
        "mean": ref_mean,
        "p50": ref_p50,
        "p90": ref_p90,
    }

    failures: list[str] = []
    notes: list[str] = []

    tm, tp50, tp90 = _stats(total)
    total_mean_ratio = tm / ref_mean if ref_mean > 0 else 0.0
    if not RATIO_LO <= total_mean_ratio <= RATIO_HI:
        failures.append(
            f"total_latency.mean ratio={total_mean_ratio:.3f} "
            f"(sim={tm:.2f}s ref={ref_mean:.2f}s, target 0.9-1.1)"
        )

    if statistics.mean(wait) >= WAIT_MAX_S:
        notes.append(f"waiting_mean={statistics.mean(wait):.2f}s (>={WAIT_MAX_S})")

    if svc:
        gap = statistics.mean([max(i - s, 0.0) for i, s in zip(infer, svc)])
        notes.append(f"infer_minus_svc_mean={gap:.2f}s (diagnostic)")
    im, ip50, ip90 = _stats(infer)
    notes.append(
        "forward ratios diagnostic="
        f"{im/ref_mean:.3f}/{ip50/ref_p50:.3f}/{ip90/ref_p90:.3f}"
    )
    notes.append(
        "total quantiles diagnostic="
        f"{tm/ref_mean:.3f}/{tp50/ref_p50:.3f}/{tp90/ref_p90:.3f}"
    )
    cm, cp50, cp90 = _stats(client)
    notes.append(
        "client ratios diagnostic="
        f"{cm/ref_mean:.3f}/{cp50/ref_p50:.3f}/{cp90/ref_p90:.3f}"
    )

    def _duration(run_dir: Path) -> float:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            return float(json.loads(metrics_path.read_text())["duration"])
        created = _col(run_dir / "request_details.csv", "created_time_s")
        finish = _col(run_dir / "request_details.csv", "finish_time_s")
        if created and finish:
            return max(finish) - min(created)
        return 0.0

    ref_dur = _duration(ref_dir)
    sim_dur = _duration(sim_dir)
    if sim_dur > MAKESPAN_RATIO_MAX * ref_dur:
        failures.append(f"makespan ratio={sim_dur/ref_dur:.3f} (max {MAKESPAN_RATIO_MAX})")

    thrash_ok, thrash_msg = _thrash_ok(sim_dir)
    if not thrash_ok:
        failures.append(f"thrash: {thrash_msg}")

    print(f"sim={sim_dir.name} ref={ref_dir.name}")
    print(f"  ref e2e mean/p50/p90: {ref_mean:.2f} / {ref_p50:.2f} / {ref_p90:.2f}")
    im, ip50, ip90 = _stats(infer)
    print(
        f"  sim forward mean/p50/p90 ratio: "
        f"{im/ref_mean:.3f} / {ip50/ref_p50:.3f} / {ip90/ref_p90:.3f}"
    )
    tm, _, _ = _stats(total)
    cm, _, _ = _stats(client)
    _, tp50, tp90 = _stats(total)
    _, cp50, cp90 = _stats(client)
    print(
        f"  sim total  mean/p50/p90 ratio: "
        f"{tm/ref_mean:.3f} / {tp50/ref_p50:.3f} / {tp90/ref_p90:.3f}"
    )
    print(
        f"  sim client mean/p50/p90 ratio: "
        f"{cm/ref_mean:.3f} / {cp50/ref_p50:.3f} / {cp90/ref_p90:.3f}"
    )
    print(f"  waiting mean: {statistics.mean(wait):.2f}s")
    if svc:
        gap = statistics.mean([max(i - s, 0.0) for i, s in zip(infer, svc)])
        print(f"  infer-svc mean: {gap:.2f}s")
    print(f"  makespan: {sim_dur:.1f}s / {ref_dur:.1f}s = {sim_dur/ref_dur:.3f}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    if notes:
        print("notes:")
        for n in notes:
            print(f"  - {n}")

    print("PASS: loop_gate")
    return 0


def main() -> None:
    diag = Path(__file__).resolve().parent
    sys.path.insert(0, str(diag))
    from parity_refs import REF_NO_OVERLAP, ref_for_run

    sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "benchmark/synthesis_requests/results/20260601_120846"
    )
    if len(sys.argv) > 2:
        ref_dir = Path(sys.argv[2])
    elif sim_dir.is_dir():
        ref_dir = ref_for_run(sim_dir)
    else:
        ref_dir = REF_NO_OVERLAP
    sys.exit(check(sim_dir, ref_dir))


if __name__ == "__main__":
    main()
