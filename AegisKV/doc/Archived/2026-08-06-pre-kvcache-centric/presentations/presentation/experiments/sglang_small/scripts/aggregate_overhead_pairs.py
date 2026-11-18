#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "results"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def server_controls(path: Path) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {
        "has_log": bool(text),
        "disable_cuda_graph": "disable_cuda_graph=True" in text,
        "disable_piecewise_cuda_graph": "disable_piecewise_cuda_graph=True" in text,
        "hicache_enabled": "enable_hierarchical_cache=True" in text,
        "hicache_disabled": "enable_hierarchical_cache=False" in text,
    }


def classify_run(run_dir: Path, min_requests: int) -> tuple[dict, list[str]]:
    summary_path = run_dir / "paired_overhead_summary.json"
    if not summary_path.exists():
        return {}, ["missing paired_overhead_summary.json"]

    summary = read_json(summary_path)
    problems: list[str] = []

    baseline = summary.get("baseline", {})
    hicache = summary.get("hicache_write_through", {})
    for label, payload in (("baseline", baseline), ("hicache_write_through", hicache)):
        count = int(payload.get("count") or 0)
        success = int(payload.get("success_count") or 0)
        if count < min_requests:
            problems.append(f"{label} count {count} < min_requests {min_requests}")
        if count != success:
            problems.append(f"{label} success_count {success} != count {count}")

    baseline_ctrl = server_controls(run_dir / "baseline_no_hicache" / "server.log")
    hicache_ctrl = server_controls(run_dir / "hicache_write_through" / "server.log")
    if not baseline_ctrl["has_log"]:
        problems.append("missing baseline server.log")
    if not hicache_ctrl["has_log"]:
        problems.append("missing hicache server.log")
    if not baseline_ctrl["hicache_disabled"]:
        problems.append("baseline did not log enable_hierarchical_cache=False")
    if not hicache_ctrl["hicache_enabled"]:
        problems.append("hicache did not log enable_hierarchical_cache=True")
    for label, ctrl in (("baseline", baseline_ctrl), ("hicache", hicache_ctrl)):
        if not ctrl["disable_cuda_graph"]:
            problems.append(f"{label} did not log disable_cuda_graph=True")
        if not ctrl["disable_piecewise_cuda_graph"]:
            problems.append(f"{label} did not log disable_piecewise_cuda_graph=True")

    return summary, problems


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metric(items: list[dict], key: str) -> dict[str, float]:
    values = [float(item[key]) for item in items]
    return {
        f"{key}_mean": mean(values),
        f"{key}_min": min(values),
        f"{key}_max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "aggregate_overhead_pairs",
    )
    parser.add_argument("--min-requests", type=int, default=8)
    args = parser.parse_args()

    run_rows: list[dict[str, object]] = []
    included: list[dict] = []

    for run_dir in sorted(args.results_root.glob("dp2_overhead_pair_*")):
        if not run_dir.is_dir():
            continue
        summary, problems = classify_run(run_dir, args.min_requests)
        included_flag = bool(summary) and not problems
        if included_flag:
            included.append(summary)
        run_rows.append(
            {
                "run_id": run_dir.name,
                "included": str(included_flag).lower(),
                "p99_overhead_pct": summary.get("p99_overhead_pct", ""),
                "p95_overhead_pct": summary.get("p95_overhead_pct", ""),
                "mean_overhead_pct": summary.get("mean_overhead_pct", ""),
                "baseline_count": summary.get("baseline", {}).get("count", ""),
                "hicache_count": summary.get("hicache_write_through", {}).get("count", ""),
                "problems": "; ".join(problems),
            }
        )

    write_csv(args.output_dir / "overhead_pair_runs.csv", run_rows)

    aggregate = {
        "included_runs": [item.get("run_id") for item in included],
        "included_count": len(included),
        "min_requests": args.min_requests,
    }
    if included:
        aggregate.update(summarize_metric(included, "p99_overhead_pct"))
        aggregate.update(summarize_metric(included, "p95_overhead_pct"))
        aggregate.update(summarize_metric(included, "mean_overhead_pct"))
        candidate_rows = [
            {
                "policy": "baseline_no_hicache",
                "p99_overhead_pct": 0.0,
                "write_amplification": 1.0,
                "evidence_tier": "sglang-small-paired",
                "source": "sglang_dp2_overhead_pair_aggregate",
                "run_id": "+".join(str(item.get("run_id")) for item in included),
                "note": "aggregate baseline for paired no-failure overhead",
            },
            {
                "policy": "hicache_write_through",
                "p99_overhead_pct": aggregate["p99_overhead_pct_mean"],
                "write_amplification": 1.0,
                "evidence_tier": "sglang-small-paired",
                "source": "sglang_dp2_overhead_pair_aggregate",
                "run_id": "+".join(str(item.get("run_id")) for item in included),
                "note": "matched DP=2 SGLang paired overhead aggregate; not final AegisKV evidence",
            },
        ]
        write_csv(
            args.output_dir / "candidate_fig_normal_path_overhead_sglang_aggregate.csv",
            candidate_rows,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overhead_pair_aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))
    return 0 if included else 1


if __name__ == "__main__":
    raise SystemExit(main())
