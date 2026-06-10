"""Compare real SGLang decode throughput with vs without HiCache (log-based probe).

Use existing server.log traces to quantify the decode 1.54x gap before deciding
whether aiconfigurator needs a HiCache-specific decode model.

For kernel-level confirmation, re-run real SGLang with only
``--enable-hierarchical-cache`` toggled and capture nsys/torch profiler on a
steady decode window (bs=10, #queue-req=0).
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_DECODE_RE = re.compile(
    r"\[([^\]]+)\] Decode batch, #running-req: (\d+), #token: (\d+).*"
    r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)


def _parse_decode_lines(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _DECODE_RE.search(line)
        if not match:
            continue
        ts, bs, total, thr, queue = (
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
            float(match.group(4)),
            int(match.group(5)),
        )
        rows.append(
            {
                "ts": ts,
                "bs": bs,
                "total": total,
                "kv_per_req": total // max(bs, 1),
                "throughput": thr,
                "queue": queue,
            }
        )
    return rows


def _steady_decode(rows: list[dict], *, min_thr: float = 50.0) -> list[dict]:
    return [
        r
        for r in rows
        if r["throughput"] >= min_thr and r["queue"] == 0 and r["bs"] >= 8
    ]


def _summ(rows: list[dict]) -> dict:
    if not rows:
        return {}
    thr = [r["throughput"] for r in rows]
    kv = [r["kv_per_req"] for r in rows]
    return {
        "n": len(rows),
        "thr_mean": statistics.mean(thr),
        "thr_min": min(thr),
        "thr_max": max(thr),
        "kv_mean": statistics.mean(kv),
    }


def main() -> None:
    no_hicache = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO / "benchmark/synthesis_requests/results/without_hicache_sglang/server.log"
    )
    with_hicache = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else REPO / "benchmark/synthesis_requests/results/with_hicache_sglang/server.log"
    )
    sim_hicache = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else REPO / "benchmark/synthesis_requests/results/with_hicache_sglsim/server.log"
    )

    no_rows = _steady_decode(_parse_decode_lines(no_hicache))
    hi_rows = _steady_decode(_parse_decode_lines(with_hicache))
    sim_rows = _steady_decode(_parse_decode_lines(sim_hicache))

    no_s = _summ(no_rows)
    hi_s = _summ(hi_rows)
    sim_s = _summ(sim_rows)

    print(f"no-hicache log : {no_hicache}")
    print(f"with-hicache   : {with_hicache}")
    print(f"sim hicache    : {sim_hicache}")
    print()
    print(f"{'label':<16} {'n':>6} {'thr_mean':>10} {'kv_mean':>10}")
    for label, s in (
        ("real no-hicache", no_s),
        ("real with-hicache", hi_s),
        ("sim with-hicache", sim_s),
    ):
        if not s:
            print(f"{label:<16} {'—':>6}")
            continue
        print(
            f"{label:<16} {s['n']:>6} {s['thr_mean']:>10.1f} {s['kv_mean']:>10.0f}"
        )

    if no_s and hi_s:
        ratio = hi_s["thr_mean"] / no_s["thr_mean"]
        step_no = 10 / no_s["thr_mean"] if no_s["thr_mean"] else 0
        step_hi = 10 / hi_s["thr_mean"] if hi_s["thr_mean"] else 0
        print()
        print(f"real with/no decode throughput ratio: {ratio:.3f}x")
        print(f"implied decode step (bs=10): no={step_no*1000:.1f}ms hi={step_hi*1000:.1f}ms")
        print(
            "e2e decode savings (1024 steps): "
            f"{(step_no - step_hi) * 1024:.1f}s"
        )
        print()
        print("Interpretation:")
        print("- fa3 decode kernel path is identical with/without HiCache (page_table).")
        print("- sim aiconfigurator matches real no-hicache (~199 tok/s) but not with-hicache (~305).")
        print("- gap is NOT explained by sim scheduling/KV transfer logic; needs GPU profiler.")
        print("- Do NOT add hicache decode_scale hack until nsys confirms kernel/runtime effect.")


if __name__ == "__main__":
    main()
