"""Optional sanity check vs with_hicache_sglang (wide bands; does not fail the merge gate)."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

RATIO_LO, RATIO_HI = 0.8, 1.2
MAKESPAN_RATIO_MAX = 1.3


def _col(path: Path, name: str) -> list[float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [float(r[name]) for r in rows if r.get(name) not in (None, "")]


def _stats(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    n = len(ordered)
    return statistics.mean(ordered), ordered[n // 2], ordered[int(0.9 * (n - 1))]


def check(sim_dir: Path, ref_dir: Path) -> int:
    sim_csv = sim_dir / "request_details.csv"
    ref_csv = ref_dir / "request_details.csv"
    if not sim_csv.is_file() or not ref_csv.is_file():
        print("WARN: missing CSV for sanity_vs_ref")
        return 0

    ref_e2e = _col(ref_csv, "e2e_latency_s")
    ref_mean, ref_p50, ref_p90 = _stats(ref_e2e)
    total = _col(sim_csv, "total_latency_s")
    tm, tp50, tp90 = _stats(total)

    failures: list[str] = []
    for name, val, ref_q in (
        ("mean", tm, ref_mean),
        ("p50", tp50, ref_p50),
        ("p90", tp90, ref_p90),
    ):
        ratio = val / ref_q if ref_q > 0 else 0.0
        if not RATIO_LO <= ratio <= RATIO_HI:
            failures.append(
                f"total.{name} ratio={ratio:.3f} (sim={val:.2f}s ref={ref_q:.2f}s)"
            )

    def _duration(run_dir: Path) -> float:
        mp = run_dir / "metrics.json"
        if mp.is_file():
            return float(json.loads(mp.read_text())["duration"])
        return 0.0

    ref_dur = _duration(ref_dir)
    sim_dur = _duration(sim_dir)
    if ref_dur > 0 and sim_dur > MAKESPAN_RATIO_MAX * ref_dur:
        failures.append(f"makespan ratio={sim_dur/ref_dur:.3f}")

    print(f"sanity_vs_ref sim={sim_dir.name} ref={ref_dir.name}")
    print(
        f"  total mean/p50/p90 ratio: "
        f"{tm/ref_mean:.3f} / {tp50/ref_p50:.3f} / {tp90/ref_p90:.3f}"
    )
    if failures:
        print("SANITY WARN (informational, does not block behavior_gate):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("SANITY OK")
    return 0


def main() -> int:
    sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "benchmark/synthesis_requests/results"
    )
    ref_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        REPO / "benchmark/synthesis_requests/results/with_hicache_sglang"
    )
    return check(sim_dir, ref_dir)


if __name__ == "__main__":
    sys.exit(main())
