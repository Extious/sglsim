#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "failure_visibility_pair_aggregate"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    return 1.96 * _stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _is_same_write(row: dict[str, Any]) -> bool:
    invisible = _float(row.get("invisible_pre_failover_backed_up_tokens"))
    visible = _float(row.get("visible_pre_failover_backed_up_tokens"))
    return invisible > 0.0 and visible > 0.0


def _prefetch_policy(row: dict[str, Any]) -> str:
    policy = str(row.get("hicache_storage_prefetch_policy") or "").strip()
    if policy:
        return policy
    text = " ".join(
        str(row.get(name) or "")
        for name in ("run_id", "summary_json", "pair_dir", "note")
    ).lower()
    if "best_effort" in text:
        return "best_effort"
    if "same_write" in text:
        return "wait_complete"
    return ""


def _pair_class(row: dict[str, Any]) -> str:
    if _is_same_write(row):
        policy = _prefetch_policy(row) or "unknown"
        return f"same_write_{policy}"
    return "frontier_vs_no_storage"


def collect_pair_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("failure_visibility_pair*/failure_visibility_pair_summary.json")):
        row = _read_json(path)
        if not row:
            continue
        row["summary_json"] = str(path)
        row["pair_dir"] = str(path.parent)
        row["hicache_storage_prefetch_policy"] = _prefetch_policy(row)
        row["pair_class"] = _pair_class(row)
        row["same_write_cost"] = str(_is_same_write(row)).lower()
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("pair_class") or ""),
            str(row.get("hicache_storage_prefetch_policy") or ""),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (pair_class, prefetch_policy), group in sorted(grouped.items()):
        norms = [_float(row.get("visible_retry_latency_norm_vs_invisible")) for row in group]
        deltas = [_float(row.get("visible_latency_delta_pct_vs_invisible")) for row in group]
        frontier_deltas = [_float(row.get("frontier_token_delta")) for row in group]
        visible_tokens = [_float(row.get("visible_pre_failover_backed_up_tokens")) for row in group]
        invisible_p99 = [_float(row.get("invisible_latency_p99_s")) for row in group]
        visible_p99 = [_float(row.get("visible_latency_p99_s")) for row in group]
        out.append(
            {
                "pair_class": pair_class,
                "hicache_storage_prefetch_policy": prefetch_policy,
                "pair_count": len(group),
                "visible_retry_latency_norm_mean": _mean(norms),
                "visible_retry_latency_norm_std": _stdev(norms),
                "visible_retry_latency_norm_ci95": _ci95(norms),
                "visible_latency_delta_pct_mean": _mean(deltas),
                "frontier_token_delta_mean": _mean(frontier_deltas),
                "visible_backed_up_tokens_mean": _mean(visible_tokens),
                "invisible_latency_p99_s_mean": _mean(invisible_p99),
                "visible_latency_p99_s_mean": _mean(visible_p99),
                "improving_pairs": sum(1 for value in norms if value < 1.0),
                "repeat_run_ids": ";".join(str(row.get("run_id") or "") for row in group),
                "invisible_job_ids": ";".join(
                    str(row.get("invisible_slurm_job_id") or "") for row in group
                ),
                "visible_job_ids": ";".join(
                    str(row.get("visible_slurm_job_id") or "") for row in group
                ),
                "evidence_tier": "sglang-small-paired-smoke",
                "source": "sglang_failure_visibility_pair_aggregate",
                "note": (
                    "aggregate of small-scale paired SGLang failure visibility runs; "
                    "not final AegisKV policy evidence"
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def render_markdown(pair_rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> str:
    lines = [
        "# SGLang Failure Visibility Pair Aggregate",
        "",
        "This report aggregates small-scale paired SGLang failure visibility runs.",
        "It is a calibration artifact, not final AegisKV policy evidence.",
        "",
        "## Aggregate",
        "",
        "| Class | Prefetch policy | Pairs | Visible p99/control p99 | 95% CI | Improving pairs | Jobs |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate:
        lines.append(
            "| {cls} | {policy} | {count} | {norm:.3f}x | {ci:.3f} | {improve} | {jobs} -> {visible} |".format(
                cls=row.get("pair_class", ""),
                policy=row.get("hicache_storage_prefetch_policy", ""),
                count=int(row.get("pair_count") or 0),
                norm=_float(row.get("visible_retry_latency_norm_mean")),
                ci=_float(row.get("visible_retry_latency_norm_ci95")),
                improve=int(row.get("improving_pairs") or 0),
                jobs=row.get("invisible_job_ids", ""),
                visible=row.get("visible_job_ids", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Pairs",
            "",
            "| Run id | Class | Policy | Invisible job | Visible job | Norm | Delta | Frontier delta |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            "| {run} | {cls} | {policy} | {inv} | {vis} | {norm:.3f}x | {delta:.1f}% | {frontier:,.0f} |".format(
                run=row.get("run_id", ""),
                cls=row.get("pair_class", ""),
                policy=row.get("hicache_storage_prefetch_policy", ""),
                inv=row.get("invisible_slurm_job_id", ""),
                vis=row.get("visible_slurm_job_id", ""),
                norm=_float(row.get("visible_retry_latency_norm_vs_invisible")),
                delta=_float(row.get("visible_latency_delta_pct_vs_invisible")),
                frontier=_float(row.get("frontier_token_delta")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate SGLang failure visibility pair summaries."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    pair_rows = collect_pair_rows(args.results_dir)
    aggregate = aggregate_rows(pair_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "failure_visibility_pair_rows.csv", pair_rows)
    write_csv(args.output_dir / "failure_visibility_pair_aggregate.csv", aggregate)
    (args.output_dir / "failure_visibility_pair_aggregate.md").write_text(
        render_markdown(pair_rows, aggregate),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pair_rows": len(pair_rows),
                "aggregate_rows": len(aggregate),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
