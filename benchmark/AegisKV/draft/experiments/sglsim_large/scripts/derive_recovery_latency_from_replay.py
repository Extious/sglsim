#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[6]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def recompute_baseline_ms(tokens: float) -> float:
    return 4.0 + max(tokens, 0.0) * 0.0108


def replay_rows(run_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(run_dir / "request_details.csv")
    out = [row for row in rows if _bool(row.get("aegis_failover_replay"))]
    if not out:
        raise ValueError(f"no replay rows in {run_dir}")
    return out


def row_pre_failure_tokens(row: dict[str, str]) -> float:
    tokens = _float(row.get("aegis_candidate_pre_failure_total_tokens"))
    if tokens > 0:
        return tokens
    return _float(row.get("input_length")) + _float(
        row.get("aegis_candidate_pre_failure_output_tokens")
    )


def run_latency(run_dir: Path) -> dict[str, float]:
    ratios: list[float] = []
    overhead_ms: list[float] = []
    sleep_ms: list[float] = []
    baseline_ms: list[float] = []
    recovery_tokens = 0.0
    for row in replay_rows(run_dir):
        tokens = row_pre_failure_tokens(row)
        baseline = recompute_baseline_ms(tokens)
        selected = _float(row.get("aegis_replay_restore_overhead_s")) * 1000.0
        if selected <= 0:
            selected = _float(row.get("aegis_replay_restore_sleep_s")) * 1000.0
        if baseline <= 0 or selected < 0:
            continue
        ratios.append(selected / baseline)
        overhead_ms.append(selected)
        sleep_ms.append(_float(row.get("aegis_replay_restore_sleep_s")) * 1000.0)
        baseline_ms.append(baseline)
        recovery_tokens += _float(row.get("aegis_replay_restored_tokens"))
        recovery_tokens += _float(row.get("aegis_replay_recompute_tokens"))

    if not ratios:
        raise ValueError(f"no valid replay latency rows in {run_dir}")
    return {
        "retry_latency_norm": mean(ratios),
        "critical_agent_latency_norm": percentile(ratios, 95),
        "mean_restore_overhead_ms": mean(overhead_ms),
        "p95_restore_overhead_ms": percentile(overhead_ms, 95),
        "mean_restore_sleep_ms": mean(sleep_ms),
        "mean_recompute_baseline_ms": mean(baseline_ms),
        "replay_rows": float(len(ratios)),
        "recovery_tokens": recovery_tokens,
    }


def metric_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def parse_regime(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--regime entries must be formatted as LABEL=RUN_DIR"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("regime label must not be empty")
    return label, Path(path)


def aggregate_run_dirs(
    run_dirs: list[Path],
) -> tuple[dict[str, float], dict[str, object], list[dict[str, object]]]:
    summaries = [run_latency(path) for path in run_dirs]
    metrics = ("retry_latency_norm", "critical_agent_latency_norm")
    aggregate = {metric: metric_mean([row[metric] for row in summaries]) for metric in metrics}
    aggregate.update(
        {
            f"{metric}_std": metric_std([row[metric] for row in summaries])
            for metric in metrics
        }
    )
    aggregate.update(
        {
            f"{metric}_ci95": metric_ci95([row[metric] for row in summaries])
            for metric in metrics
        }
    )
    totals = {
        "runtime_replay_rows": int(sum(row["replay_rows"] for row in summaries)),
        "runtime_recovery_tokens": int(sum(row["recovery_tokens"] for row in summaries)),
        "runtime_mean_restore_overhead_ms": metric_mean(
            [row["mean_restore_overhead_ms"] for row in summaries]
        ),
        "runtime_mean_restore_sleep_ms": metric_mean(
            [row["mean_restore_sleep_ms"] for row in summaries]
        ),
        "runtime_mean_recompute_baseline_ms": metric_mean(
            [row["mean_recompute_baseline_ms"] for row in summaries]
        ),
    }
    return aggregate, totals, []


def aggregate_regimes(
    regimes: list[tuple[str, Path]],
) -> tuple[dict[str, float], dict[str, object], list[dict[str, object]]]:
    by_regime: dict[str, list[Path]] = {}
    for label, path in regimes:
        by_regime.setdefault(label, []).append(path)
    if not by_regime:
        raise ValueError("no regimes supplied")

    regime_summaries: list[dict[str, object]] = []
    for label, paths in by_regime.items():
        runs = [run_latency(path) for path in paths]
        summary: dict[str, object] = {
            "label": label,
            "runs": len(paths),
            "replay_rows": int(sum(row["replay_rows"] for row in runs)),
            "recovery_tokens": int(sum(row["recovery_tokens"] for row in runs)),
            "retry_latency_norm": metric_mean(
                [row["retry_latency_norm"] for row in runs]
            ),
            "critical_agent_latency_norm": metric_mean(
                [row["critical_agent_latency_norm"] for row in runs]
            ),
            "mean_restore_overhead_ms": metric_mean(
                [row["mean_restore_overhead_ms"] for row in runs]
            ),
            "mean_restore_sleep_ms": metric_mean(
                [row["mean_restore_sleep_ms"] for row in runs]
            ),
            "mean_recompute_baseline_ms": metric_mean(
                [row["mean_recompute_baseline_ms"] for row in runs]
            ),
        }
        regime_summaries.append(summary)

    metrics = ("retry_latency_norm", "critical_agent_latency_norm")
    aggregate = {
        metric: metric_mean([float(row[metric]) for row in regime_summaries])
        for metric in metrics
    }
    aggregate.update(
        {
            f"{metric}_std": metric_std(
                [float(row[metric]) for row in regime_summaries]
            )
            for metric in metrics
        }
    )
    aggregate.update(
        {
            f"{metric}_ci95": metric_ci95(
                [float(row[metric]) for row in regime_summaries]
            )
            for metric in metrics
        }
    )
    totals = {
        "runtime_replay_rows": int(sum(int(row["replay_rows"]) for row in regime_summaries)),
        "runtime_recovery_tokens": int(
            sum(int(row["recovery_tokens"]) for row in regime_summaries)
        ),
        "runtime_regime_count": len(regime_summaries),
        "runtime_regimes": ";".join(str(row["label"]) for row in regime_summaries),
        "runtime_regime_summaries": json.dumps(regime_summaries, sort_keys=True),
        "runtime_mean_restore_overhead_ms": metric_mean(
            [float(row["mean_restore_overhead_ms"]) for row in regime_summaries]
        ),
        "runtime_mean_restore_sleep_ms": metric_mean(
            [float(row["mean_restore_sleep_ms"]) for row in regime_summaries]
        ),
        "runtime_mean_recompute_baseline_ms": metric_mean(
            [float(row["mean_recompute_baseline_ms"]) for row in regime_summaries]
        ),
    }
    return aggregate, totals, regime_summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace Fig. 9 AegisKV row with direct replay restore-latency "
            "ratios relative to same-length recompute."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--regime",
        type=parse_regime,
        action="append",
        default=[],
        help=(
            "Optional equal-regime aggregation input formatted as LABEL=RUN_DIR. "
            "When supplied, each label is averaged across its runs first and the "
            "AegisKV row is the equal-weight mean across labels."
        ),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = [dict(row) for row in read_csv(args.input_csv)]
    if args.regime:
        aggregate, totals, _ = aggregate_regimes(args.regime)
        run_dirs = [path for _, path in args.regime]
        repeat = (
            f"mean-of-{totals['runtime_regime_count']}-regimes/"
            f"{len(run_dirs)}-runs"
        )
        note = (
            "AegisKV recovery latency is an equal-regime mean of direct runtime "
            "replay restore-overhead ratios relative to same-length recompute; "
            "other policy rows remain preliminary model baselines"
        )
    else:
        if not args.run_dir:
            raise ValueError("at least one --run-dir or --regime is required")
        aggregate, totals, _ = aggregate_run_dirs(args.run_dir)
        run_dirs = list(args.run_dir)
        repeat = f"mean-of-{len(run_dirs)}"
        note = (
            "AegisKV recovery latency is derived from direct runtime replay "
            "restore-overhead ratios relative to same-length recompute; other "
            "policy rows remain preliminary model baselines"
        )

    current_sha = git_sha()
    replaced = False
    for row in rows:
        if row.get("policy") == "AegisKV":
            row.update(aggregate)
            row.update(
                {
                    "evidence_tier": "sglsim-runtime-replay-derived",
                    "source": "sglsim_request_details_replay_recovery_latency",
                    "run_id": args.run_id,
                    "git_sha": current_sha,
                    "repeat": repeat,
                    "note": note,
                    "repeat_count": len(run_dirs),
                    "repeat_run_ids": ";".join(str(path) for path in run_dirs),
                    **totals,
                }
            )
            replaced = True
            break
    if not replaced:
        raise ValueError("input CSV has no AegisKV row")

    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "run_dirs": [str(path) for path in run_dirs],
                "aegis_row": aggregate,
                **totals,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
