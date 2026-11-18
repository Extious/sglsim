#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from derive_restore_boundary_from_replay import (
    _bool,
    _float,
    git_sha,
    read_csv,
    write_csv,
)


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def overhead_ms(row: dict[str, str]) -> float:
    value = _float(row.get("aegis_replay_restore_overhead_s")) * 1000.0
    if value <= 0:
        value = _float(row.get("aegis_replay_restore_sleep_s")) * 1000.0
    return value


def e2e_ms(row: dict[str, str]) -> float:
    for key in ("client_wall_e2e_latency_s", "e2e_latency_s", "total_latency_s"):
        value = _float(row.get(key))
        if value > 0:
            return value * 1000.0
    return 0.0


def matrix_run_dirs(matrix_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for matrix_dir in matrix_dirs:
        summary = json.loads((matrix_dir / "matrix_summary.json").read_text())
        out.extend(Path(path) for path in summary["run_dirs"])
    return out


def load_recovery_rows(
    run_dirs: list[Path],
    *,
    expected_action: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for trial, run_dir in enumerate(run_dirs):
        for row in read_csv(run_dir / "request_details.csv"):
            if not _bool(row.get("aegis_failover_replay")):
                continue
            action = str(row.get("aegis_replay_restore_action") or "")
            if action != expected_action:
                continue
            rid = str(row.get("rid") or row.get("request_id") or "")
            if not rid:
                rid = (
                    str(row.get("aegis_failed_source_dp_rank") or "")
                    + ":"
                    + str(row.get("aegis_worker_seq") or "")
                    + ":"
                    + str(row.get("aegis_candidate_pre_failure_total_tokens") or "")
                )
            row = dict(row)
            row["_match_key"] = f"{trial}:{rid}"
            row["_source_run_dir"] = str(run_dir)
            rows.append(row)
    if not rows:
        raise ValueError(f"no {expected_action} recovery rows found")
    return rows


def pair_rows(
    *,
    peer_rows: list[dict[str, str]],
    nonpeer_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    nonpeer_by_key = {row["_match_key"]: row for row in nonpeer_rows}
    pairs = [
        (peer, nonpeer_by_key[peer["_match_key"]])
        for peer in peer_rows
        if peer["_match_key"] in nonpeer_by_key
    ]
    if not pairs:
        raise ValueError("peer and non-peer replay rows did not match by request id")
    return pairs


def summarize_wct(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("no WCT values to summarize")
    return {
        "normalized_wct": mean(values),
        "normalized_wct_std": metric_std(values),
        "normalized_wct_ci95": metric_ci95(values),
    }


def summarize_overhead(
    values_ms: list[float],
    *,
    baseline_p99_ms: float,
) -> dict[str, float]:
    values_pct = [100.0 * value / baseline_p99_ms for value in values_ms]
    return {
        "peer_p99_overhead_pct": 100.0
        * percentile(values_ms, 0.99)
        / baseline_p99_ms,
        "peer_p99_overhead_pct_std": metric_std(values_pct),
        "peer_p99_overhead_pct_ci95": metric_ci95(values_pct),
        "runtime_mean_restore_overhead_ms": mean(values_ms),
        "runtime_p99_restore_overhead_ms": percentile(values_ms, 0.99),
    }


def make_row(
    *,
    policy: str,
    wct_values: list[float],
    overhead_values_ms: list[float],
    baseline_p99_ms: float,
    base_meta: dict[str, object],
    route_hit_rate: float,
    matched_rows: int,
) -> dict[str, object]:
    return {
        "policy": policy,
        **summarize_wct(wct_values),
        **summarize_overhead(overhead_values_ms, baseline_p99_ms=baseline_p99_ms),
        **base_meta,
        "repeat": f"mean-of-{matched_rows}-matched-replay-requests",
        "runtime_route_hit_rate": route_hit_rate,
        "runtime_replay_rows": matched_rows,
        "runtime_baseline_p99_recompute_overhead_ms": baseline_p99_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 14b peer-route selection rows from matched SGLSim "
            "hard-failover replay request_details.csv files."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--peer-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--nonpeer-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--peer-matrix-dir", type=Path, action="append", default=[])
    parser.add_argument("--nonpeer-matrix-dir", type=Path, action="append", default=[])
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--model-path", default="mock-model")
    args = parser.parse_args()

    peer_run_dirs = [
        *args.peer_run_dir,
        *matrix_run_dirs(args.peer_matrix_dir),
    ]
    nonpeer_run_dirs = [
        *args.nonpeer_run_dir,
        *matrix_run_dirs(args.nonpeer_matrix_dir),
    ]
    if not peer_run_dirs or not nonpeer_run_dirs:
        raise ValueError("peer and non-peer run dirs are both required")
    if args.dp_size <= 1:
        raise ValueError("--dp-size must be greater than one")

    peer_rows = load_recovery_rows(peer_run_dirs, expected_action="gpu_replica")
    nonpeer_rows = load_recovery_rows(nonpeer_run_dirs, expected_action="recompute")
    pairs = pair_rows(peer_rows=peer_rows, nonpeer_rows=nonpeer_rows)

    peer_overheads: list[float] = []
    nonpeer_overheads: list[float] = []
    peer_wct: list[float] = []
    nonpeer_wct: list[float] = []
    random_wct: list[float] = []
    random_overheads: list[float] = []
    random_hit_probability = 1.0 / float(args.dp_size - 1)

    for peer, nonpeer in pairs:
        peer_overhead = overhead_ms(peer)
        nonpeer_overhead = overhead_ms(nonpeer)
        baseline_e2e = e2e_ms(nonpeer)
        if peer_overhead <= 0 or nonpeer_overhead <= 0 or baseline_e2e <= 0:
            continue
        base_without_restore = baseline_e2e - nonpeer_overhead
        if base_without_restore <= 0:
            continue

        peer_static_e2e = base_without_restore + peer_overhead
        random_static_e2e = (
            random_hit_probability * peer_static_e2e
            + (1.0 - random_hit_probability) * baseline_e2e
        )
        random_overhead = (
            random_hit_probability * peer_overhead
            + (1.0 - random_hit_probability) * nonpeer_overhead
        )

        peer_overheads.append(peer_overhead)
        nonpeer_overheads.append(nonpeer_overhead)
        random_overheads.append(random_overhead)
        peer_wct.append(peer_static_e2e / baseline_e2e)
        nonpeer_wct.append(1.0)
        random_wct.append(random_static_e2e / baseline_e2e)

    if not peer_wct:
        raise ValueError("matched rows had no positive latency/overhead values")

    run_dirs = [str(path) for path in [*peer_run_dirs, *nonpeer_run_dirs]]
    base_meta = {
        "evidence_tier": "sglsim-runtime-replay-derived",
        "source": "sglsim_dp4_peer_route_matched_request_details",
        "run_id": args.run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": args.model_path,
        "workload_id": "aegiskv-sglsim-dp4-short-tail-peer-route",
        "dp_size": args.dp_size,
        "tp_size": 1,
        "failure_rank": "all",
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(run_dirs),
        "note": (
            "Matched DP4 short-tail hard-failover replay. Non-peer route is the "
            "observed recompute baseline; random healthy route is a direct "
            "1/(dp_size-1) mixture of matched peer-hit and route-miss rows; "
            "AegisKV peer route is the recorded peer-rank hit."
        ),
    }
    baseline_p99 = percentile(nonpeer_overheads, 0.99)
    rows = [
        make_row(
            policy="non-peer route miss",
            wct_values=nonpeer_wct,
            overhead_values_ms=nonpeer_overheads,
            baseline_p99_ms=baseline_p99,
            base_meta=base_meta,
            route_hit_rate=0.0,
            matched_rows=len(peer_wct),
        ),
        make_row(
            policy="random healthy route",
            wct_values=random_wct,
            overhead_values_ms=random_overheads,
            baseline_p99_ms=baseline_p99,
            base_meta=base_meta,
            route_hit_rate=random_hit_probability,
            matched_rows=len(peer_wct),
        ),
        make_row(
            policy="AegisKV peer route",
            wct_values=peer_wct,
            overhead_values_ms=peer_overheads,
            baseline_p99_ms=baseline_p99,
            base_meta=base_meta,
            route_hit_rate=1.0,
            matched_rows=len(peer_wct),
        ),
    ]

    write_csv(args.output_csv, rows)
    summary = {
        "run_id": args.run_id,
        "output_csv": str(args.output_csv),
        "matched_recovery_rows": len(peer_wct),
        "peer_run_dirs": [str(path) for path in peer_run_dirs],
        "nonpeer_run_dirs": [str(path) for path in nonpeer_run_dirs],
        "random_hit_probability": random_hit_probability,
        "baseline_p99_recompute_overhead_ms": baseline_p99,
        "policies": [
            {
                "policy": row["policy"],
                "normalized_wct": row["normalized_wct"],
                "peer_p99_overhead_pct": row["peer_p99_overhead_pct"],
                "runtime_mean_restore_overhead_ms": row[
                    "runtime_mean_restore_overhead_ms"
                ],
            }
            for row in rows
        ],
    }
    if args.summary_md:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Fig. 14b Peer-Route Replay Derivation",
            "",
            f"- run_id: `{args.run_id}`",
            f"- matched recovery rows: `{len(peer_wct)}`",
            f"- DP size: `{args.dp_size}`",
            f"- random healthy route hit probability: `{random_hit_probability:.4f}`",
            f"- baseline p99 recompute overhead: `{baseline_p99:.4f} ms`",
            "",
            "| policy | normalized WCT | p99 overhead vs recompute | mean restore overhead |",
            "|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['policy']} | {float(row['normalized_wct']):.4f} | "
                f"{float(row['peer_p99_overhead_pct']):.2f}% | "
                f"{float(row['runtime_mean_restore_overhead_ms']):.4f} ms |"
            )
        lines.extend(
            [
                "",
                "This is a matched-request replay derivation, not a final physical "
                "P2P KV-copy policy sweep.",
                "",
            ]
        )
        args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
