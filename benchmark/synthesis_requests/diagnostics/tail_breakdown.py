"""Break down sim tail latency by turn/waiting and cross-check server.log."""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_TURN_RE = re.compile(r"-t(\d+)$")
_DP_RE = re.compile(r"DP(\d+)")
_QUEUE_RE = re.compile(r"#queue-req: (\d+)")
_USAGE_RE = re.compile(r"token usage: ([\d.]+)")


def _parse_turn(rid: str) -> int:
    match = _TURN_RE.search(rid)
    return int(match.group(1)) if match else -1


def _load_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _turn_stats(rows: list[dict], col: str) -> dict[int, list[float]]:
    by_turn: dict[int, list[float]] = {}
    for row in rows:
        turn = _parse_turn(row.get("rid", ""))
        raw = row.get(col)
        if raw in (None, ""):
            continue
        by_turn.setdefault(turn, []).append(float(raw))
    return by_turn


def _summ(values: list[float]) -> str:
    if not values:
        return "n=0"
    ordered = sorted(values)
    n = len(ordered)
    p90 = ordered[int(0.9 * (n - 1))]
    return (
        f"n={n} mean={statistics.mean(values):.2f}s "
        f"p50={ordered[n // 2]:.2f}s p90={p90:.2f}s max={max(values):.2f}s"
    )


def _log_dp_usage(log_path: Path) -> dict[int, dict[str, float]]:
    """Per-DP max queue depth and token usage from server.log."""
    stats: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        dp_match = _DP_RE.search(line)
        if not dp_match:
            continue
        dp = int(dp_match.group(1))
        entry = stats.setdefault(dp, {"max_queue": 0.0, "max_usage": 0.0})
        q_match = _QUEUE_RE.search(line)
        if q_match:
            entry["max_queue"] = max(entry["max_queue"], float(q_match.group(1)))
        u_match = _USAGE_RE.search(line)
        if u_match:
            entry["max_usage"] = max(entry["max_usage"], float(u_match.group(1)))
    return stats


def print_report(sim_dir: Path, ref_dir: Path, *, top_n: int = 12) -> None:
    sim_csv = sim_dir / "request_details.csv"
    ref_csv = ref_dir / "request_details.csv"
    sim_rows = _load_rows(sim_csv)
    ref_rows = _load_rows(ref_csv)

    print(f"sim  dir: {sim_dir}")
    print(f"ref  dir: {ref_dir}")
    print()

    for col, label in (
        ("waiting_time_s", "waiting"),
        ("total_latency_s", "total"),
        ("client_wall_e2e_latency_s", "client_e2e"),
    ):
        print(f"=== sim by turn ({label}) ===")
        for turn in sorted(_turn_stats(sim_rows, col)):
            print(f"  t{turn}: {_summ(_turn_stats(sim_rows, col)[turn])}")
        print(f"=== ref by turn (e2e_latency_s) ===")
        ref_by_turn = _turn_stats(ref_rows, "e2e_latency_s")
        for turn in sorted(ref_by_turn):
            print(f"  t{turn}: {_summ(ref_by_turn[turn])}")
        print()

    ranked = sorted(
        sim_rows,
        key=lambda r: float(r.get("waiting_time_s") or 0),
        reverse=True,
    )[:top_n]
    print(f"=== top {top_n} by waiting_time_s ===")
    print(f"{'rid':<10} {'wait':>8} {'infer':>8} {'total':>8} {'svc':>8}")
    for row in ranked:
        svc = row.get("logical_service_time_s", "")
        print(
            f"{row['rid']:<10} "
            f"{float(row['waiting_time_s']):8.2f} "
            f"{float(row['inference_time_s']):8.2f} "
            f"{float(row['total_latency_s']):8.2f} "
            f"{float(svc or 0):8.2f}"
        )
    print()

    log_path = sim_dir / "server.log"
    if not log_path.is_file():
        log_path = REPO / "benchmark/synthesis_requests/logs/server.log"
    if log_path.is_file():
        print(f"=== server.log DP peaks ({log_path.name}) ===")
        for dp, vals in sorted(_log_dp_usage(log_path).items()):
            print(
                f"  DP{dp}: max_queue={vals['max_queue']:.0f} "
                f"max_usage={vals['max_usage']:.3f}"
            )
    else:
        print("=== server.log: missing ===")


def main() -> None:
    diag = Path(__file__).resolve().parent
    sys.path.insert(0, str(diag))
    from parity_refs import REF_NO_OVERLAP, ref_for_run

    sim_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "benchmark/synthesis_requests/results/20260602_193524"
    )
    if len(sys.argv) > 2:
        ref_dir = Path(sys.argv[2])
    elif sim_dir.is_dir():
        ref_dir = ref_for_run(sim_dir)
    else:
        ref_dir = REF_NO_OVERLAP
    print_report(sim_dir, ref_dir)


if __name__ == "__main__":
    main()
