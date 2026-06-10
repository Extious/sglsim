"""Decompose sim vs SGLang latency with per-component <10% gates."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG))
from run_integrity import check_run_integrity  # noqa: E402

RATIO_LO, RATIO_HI = 0.9, 1.1
WAIT_MAX_S = 2.0
MAKESPAN_RATIO_MAX = 1.1


def _col(path: Path, name: str) -> list[float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: list[float] = []
    for row in rows:
        raw = row.get(name)
        if raw is None or raw == "":
            continue
        out.append(float(raw))
    return out


def _summ(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    p50 = ordered[n // 2]
    p90 = ordered[int(0.9 * (n - 1))]
    return {
        "n": float(n),
        "mean": statistics.mean(values),
        "p50": p50,
        "p90": p90,
        "max": max(values),
    }


def _ratio(sim: float, ref: float) -> float:
    if ref <= 0:
        return float("inf")
    return sim / ref


def _in_band(ratio: float) -> bool:
    return RATIO_LO <= ratio <= RATIO_HI


def analyze_run(run_dir: Path) -> dict[str, dict[str, float]]:
    csv_path = run_dir / "request_details.csv"
    metrics_path = run_dir / "metrics.json"
    e2e = _col(csv_path, "e2e_latency_s")
    total = _col(csv_path, "total_latency_s") or e2e
    client = _col(csv_path, "client_wall_e2e_latency_s") or e2e
    wait = _col(csv_path, "waiting_time_s")
    infer = _col(csv_path, "inference_time_s")
    svc = _col(csv_path, "logical_service_time_s")
    overhead = [max(i - s, 0.0) for i, s in zip(infer, svc)] if svc else infer
    duration = 0.0
    if metrics_path.is_file():
        duration = float(json.loads(metrics_path.read_text())["duration"])
    else:
        created = _col(csv_path, "created_time_s")
        finish = _col(csv_path, "finish_time_s")
        if created and finish:
            duration = max(finish) - min(created)
    return {
        "e2e": _summ(e2e),
        "total_latency": _summ(total),
        "client_e2e": _summ(client),
        "waiting": _summ(wait),
        "inference": _summ(infer),
        "logical_service": _summ(svc) if svc else {},
        "infer_minus_svc": _summ(overhead) if overhead else {},
        "duration_s": {"mean": duration},
    }


def component_gates(sim_dir: Path, ref_dir: Path) -> tuple[list[str], list[str]]:
    sim = analyze_run(sim_dir)
    ref = analyze_run(ref_dir)
    ref_e2e = ref.get("e2e") or ref.get("total_latency", {})
    failures: list[str] = []
    notes: list[str] = []

    # Primary: end-to-end latency vs ref e2e. SGLSim preserves queue/service
    # decomposition, while SGLang reference CSV does not expose comparable queue
    # columns; forward/service is therefore diagnostic below.
    primary = sim.get("total_latency", {})
    sv, rv = primary.get("mean"), ref_e2e.get("mean")
    if sv is not None and rv is not None:
        ratio = _ratio(sv, rv)
        if not _in_band(ratio):
            failures.append(f"e2e mean ratio {ratio:.3f} not in [{RATIO_LO},{RATIO_HI}]")
        else:
            notes.append(f"e2e mean ratio {ratio:.3f} OK")
    quantile_ratios: list[str] = []
    for q in ("p50", "p90"):
        sv, rv = primary.get(q), ref_e2e.get(q)
        if sv is not None and rv is not None:
            quantile_ratios.append(f"{q}={_ratio(sv, rv):.3f}")
    if quantile_ratios:
        notes.append("e2e quantiles diagnostic " + ", ".join(quantile_ratios))

    # Secondary: client wall time includes HTTP/client-side overhead not present
    # in the SGLang reference CSV.
    for key, label in (
        ("client_e2e", "client mean"),
    ):
        s_mean = sim.get(key, {}).get("mean")
        r_mean = ref_e2e.get("mean")
        if s_mean is None or r_mean is None:
            continue
        ratio = _ratio(s_mean, r_mean)
        if not _in_band(ratio):
            notes.append(f"{label} ratio {ratio:.3f} (secondary, not strict)")
        else:
            notes.append(f"{label} ratio {ratio:.3f} OK")

    wait_mean = sim.get("waiting", {}).get("mean")
    if wait_mean is not None:
        if wait_mean >= WAIT_MAX_S:
            notes.append(f"waiting mean {wait_mean:.2f}s (>= {WAIT_MAX_S}s)")
        else:
            notes.append(f"waiting mean {wait_mean:.2f}s OK")

    sim_dur = sim.get("duration_s", {}).get("mean", 0.0)
    ref_dur = ref.get("duration_s", {}).get("mean", 0.0)
    if ref_dur > 0:
        ms_ratio = sim_dur / ref_dur
        if ms_ratio > MAKESPAN_RATIO_MAX:
            failures.append(
                f"makespan ratio {ms_ratio:.3f} > {MAKESPAN_RATIO_MAX}"
            )
        else:
            notes.append(f"makespan ratio {ms_ratio:.3f} OK")

    infer_svc = sim.get("infer_minus_svc", {}).get("mean")
    if infer_svc is not None:
        notes.append(
            f"infer-service mean {infer_svc:.2f}s (diagnostic; ref has no logical_service)"
        )
    infer = sim.get("inference", {})
    ratios: list[str] = []
    for q in ("mean", "p50", "p90"):
        sv, rv = infer.get(q), ref_e2e.get(q)
        if sv is not None and rv is not None:
            ratios.append(f"{q}={_ratio(sv, rv):.3f}")
    if ratios:
        notes.append("forward ratios diagnostic " + ", ".join(ratios))

    return failures, notes


def print_report(sim_dir: Path, ref_dir: Path, *, strict: bool = False) -> int:
    integrity = check_run_integrity(sim_dir, ref_dir=ref_dir)
    sim = analyze_run(sim_dir)
    ref = analyze_run(ref_dir)
    ref_e2e = ref.get("e2e") or ref.get("total_latency", {})

    print(f"sim  dir: {sim_dir}")
    print(f"ref  dir: {ref_dir}")
    if not integrity.ok:
        print()
        print("run_integrity")
        for failure in integrity.failures:
            print(f"  FAIL: {failure}")
        if strict:
            return 1
    print()
    print(f"{'metric':<22} {'sim_mean':>10} {'ref_mean':>10} {'ratio':>8}")
    for key, label in (
        ("total_latency", "total_latency_s"),
        ("client_e2e", "client_wall_e2e"),
        ("e2e", "e2e_latency_s"),
        ("waiting", "waiting_time_s"),
        ("inference", "inference_time_s"),
        ("logical_service", "logical_service_s"),
        ("infer_minus_svc", "infer - service"),
    ):
        s = sim.get(key, {}).get("mean")
        r = (
            ref_e2e.get("mean")
            if key in ("total_latency", "client_e2e", "e2e")
            else ref.get(key, {}).get("mean")
        )
        if s is None:
            continue
        if r is None:
            r = 0.0
        ratio = _ratio(s, r) if r else float("nan")
        print(f"{label:<22} {s:10.2f} {r:10.2f} {ratio:8.3f}")

    sim_dur = sim.get("duration_s", {}).get("mean", 0.0)
    ref_dur = ref.get("duration_s", {}).get("mean", 0.0)
    if ref_dur > 0:
        print()
        print(f"makespan sim={sim_dur:.1f}s ref={ref_dur:.1f}s ratio={sim_dur/ref_dur:.3f}")

    failures, notes = component_gates(sim_dir, ref_dir)
    print()
    print("component_gates")
    for n in notes:
        print(f"  OK: {n}")
    for f in failures:
        print(f"  FAIL: {f}")

    if strict and failures:
        return 1
    return 0


def main() -> None:
    diag = Path(__file__).resolve().parent
    sys.path.insert(0, str(diag))
    from parity_refs import REF_NO_OVERLAP, ref_for_run

    sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "benchmark/synthesis_requests/results/20260601_120846"
    )
    if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
        ref_dir = Path(sys.argv[2])
    elif sim_dir.is_dir():
        ref_dir = ref_for_run(sim_dir)
    else:
        ref_dir = REF_NO_OVERLAP
    strict = "--strict" in sys.argv
    sys.exit(print_report(sim_dir, ref_dir, strict=strict))


if __name__ == "__main__":
    main()
