"""Compare SGLSim run metrics against real SGLang traces (fa3 baseline)."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Accounting notes (aligned 2026-06-01):
# - sim cache_hit_tokens: device-resident radix prefix at first prepare_prefill
# - sim load_back_tokens: host tokens queued at prefill admit (load_back return)
# - sglang CSV cache_hit_tokens: usage.prompt_tokens_details.cached_tokens (total)
# - sglang CSV load_back_tokens: usage.cached_tokens_details.host component
# Compare sums as directional signals; exact parity requires matching HTTP usage API.


def _load_e2e(path: Path) -> list[float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [float(r["e2e_latency_s"]) for r in rows if r.get("e2e_latency_s")]


def _load_int_col(path: Path, col: str) -> list[int]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: list[int] = []
    for row in rows:
        raw = row.get(col)
        if raw is None or raw == "":
            continue
        out.append(int(float(raw)))
    return out


def _summ(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    p90 = ordered[int(0.9 * (n - 1))]
    return {
        "n": n,
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "p90": p90,
        "max": max(values),
    }


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "benchmark/synthesis_requests/results/20260531_142847"
    )
    real_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        REPO / "benchmark/synthesis_requests/results/without_hicache"
    )

    sim_e2e = _summ(_load_e2e(sim_dir / "request_details.csv"))
    real_e2e = _summ(_load_e2e(real_dir / "request_details.csv"))
    real_metrics = _load_metrics(real_dir / "metrics.json")

    sim_load_back = _load_int_col(sim_dir / "request_details.csv", "load_back_tokens")
    real_load_back = _load_int_col(real_dir / "request_details.csv", "load_back_tokens")
    sim_cache_hit = _load_int_col(sim_dir / "request_details.csv", "cache_hit_tokens")
    real_cache_hit = _load_int_col(real_dir / "request_details.csv", "cache_hit_tokens")

    print(f"sim   dir: {sim_dir}")
    print(f"real  dir: {real_dir}")
    print()
    print(f"{'metric':<22} {'sim':>12} {'real(fa3)':>12} {'ratio':>8}")
    for key in ("mean", "std", "min", "p90", "max"):
        s = sim_e2e[key]
        r = real_e2e[key]
        ratio = s / r if r else float("nan")
        print(f"e2e_{key + '_s':<20} {s:12.2f} {r:12.2f} {ratio:8.3f}")
    print()
    if sim_load_back or real_load_back:
        sim_lb = sum(sim_load_back)
        real_lb = sum(real_load_back)
        print(
            f"load_back_tokens(sum): sim={sim_lb} real={real_lb} "
            "(real CSV may omit column -> 0; check server.log for host H2D events)"
        )
    if sim_cache_hit or real_cache_hit:
        print(
            f"cache_hit_tokens(sum): sim={sum(sim_cache_hit)} real={sum(real_cache_hit)} "
            "(sim=device-resident; real=total cached per usage API)"
        )

    sim_log = REPO / "benchmark/synthesis_requests/logs/server.log"
    if not sim_log.is_file():
        sim_log = sim_dir / "server.log"
    real_log = real_dir / "server.log"
    if sim_log.is_file() or real_log.is_file():
        import importlib.util

        thrash_path = Path(__file__).resolve().parent / "thrash_metrics.py"
        spec = importlib.util.spec_from_file_location("thrash_metrics", thrash_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        analyze_server_log = mod.analyze_server_log

        print()
        print("=== server.log thrash indicators ===")
        for label, log_path in (("sim", sim_log), ("real", real_log)):
            if log_path.is_file():
                m = analyze_server_log(log_path)
                print(
                    f"  {label}: retract={m['retract_events']} "
                    f"prefill_new_tok={m['prefill_new_token_sum']} "
                    f"zero_cached_frac={m['zero_cached_fraction']:.2f} "
                    f"usage_p90={m['token_usage_p90']:.2f}"
                )
    print()
    print(
        f"real makespan(s): {real_metrics.get('duration', 0):.1f}  "
        f"output_tok/s: {real_metrics.get('output_throughput', 0):.1f}"
    )
    print(
        "Note: sim #token in sglang_projected logs uses physical KV pool occupancy "
        "(shared radix pages counted once); DP round_robin may shift per-DP totals."
    )


if __name__ == "__main__":
    main()
