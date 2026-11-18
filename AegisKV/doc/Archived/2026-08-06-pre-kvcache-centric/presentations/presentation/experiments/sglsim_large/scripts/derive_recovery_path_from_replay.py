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


def run_shares(run_dir: Path) -> dict[str, float]:
    rows = read_csv(run_dir / "request_details.csv")
    replay_rows = [row for row in rows if _bool(row.get("aegis_failover_replay"))]
    if not replay_rows:
        raise ValueError(f"no replay rows in {run_dir}")

    totals = {"peer_gpu": 0.0, "host_h2d": 0.0, "ordinary_cache": 0.0, "recompute": 0.0}
    for row in replay_rows:
        restored = max(_float(row.get("aegis_replay_restored_tokens")), 0.0)
        recompute = max(_float(row.get("aegis_replay_recompute_tokens")), 0.0)
        action = str(row.get("aegis_replay_restore_action", ""))
        if action == "gpu_replica":
            totals["peer_gpu"] += restored
        elif action == "host_restore":
            totals["host_h2d"] += restored
        elif action == "ordinary_cache":
            totals["ordinary_cache"] += restored
        elif restored > 0:
            totals["host_h2d"] += restored
        totals["recompute"] += recompute

    denominator = sum(totals.values())
    if denominator <= 0:
        raise ValueError(f"zero replay recovery tokens in {run_dir}")
    shares = {key: value / denominator for key, value in totals.items()}
    shares["replay_rows"] = float(len(replay_rows))
    shares["recovery_tokens"] = denominator
    return shares


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
) -> tuple[dict[str, float], dict[str, float]]:
    shares = [run_shares(path) for path in run_dirs]
    metrics = ("peer_gpu", "host_h2d", "ordinary_cache", "recompute")
    aggregate = {metric: metric_mean([row[metric] for row in shares]) for metric in metrics}
    aggregate.update(
        {
            f"{metric}_std": metric_std([row[metric] for row in shares])
            for metric in metrics
        }
    )
    aggregate.update(
        {
            f"{metric}_ci95": metric_ci95([row[metric] for row in shares])
            for metric in metrics
        }
    )
    totals = {
        "runtime_replay_rows": int(sum(row["replay_rows"] for row in shares)),
        "runtime_recovery_tokens": int(sum(row["recovery_tokens"] for row in shares)),
    }
    return aggregate, totals


def aggregate_regimes(
    regimes: list[tuple[str, Path]],
) -> tuple[dict[str, float], dict[str, object]]:
    by_regime: dict[str, list[Path]] = {}
    for label, path in regimes:
        by_regime.setdefault(label, []).append(path)
    if not by_regime:
        raise ValueError("no regimes supplied")

    metrics = ("peer_gpu", "host_h2d", "ordinary_cache", "recompute")
    summaries: list[dict[str, object]] = []
    for label, paths in by_regime.items():
        shares = [run_shares(path) for path in paths]
        summary: dict[str, object] = {
            "label": label,
            "runs": len(paths),
            "replay_rows": int(sum(row["replay_rows"] for row in shares)),
            "recovery_tokens": int(sum(row["recovery_tokens"] for row in shares)),
        }
        for metric in metrics:
            summary[metric] = metric_mean([row[metric] for row in shares])
        summaries.append(summary)

    aggregate = {
        metric: metric_mean([float(row[metric]) for row in summaries])
        for metric in metrics
    }
    aggregate.update(
        {
            f"{metric}_std": metric_std([float(row[metric]) for row in summaries])
            for metric in metrics
        }
    )
    aggregate.update(
        {
            f"{metric}_ci95": metric_ci95([float(row[metric]) for row in summaries])
            for metric in metrics
        }
    )
    totals = {
        "runtime_replay_rows": int(sum(int(row["replay_rows"]) for row in summaries)),
        "runtime_recovery_tokens": int(
            sum(int(row["recovery_tokens"]) for row in summaries)
        ),
        "runtime_regime_count": len(summaries),
        "runtime_regimes": ";".join(str(row["label"]) for row in summaries),
        "runtime_regime_summaries": json.dumps(summaries, sort_keys=True),
    }
    return aggregate, totals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace Fig. 10 AegisKV row with direct replay recovery-path counters."
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
        aggregate, totals = aggregate_regimes(args.regime)
        run_dirs = [path for _, path in args.regime]
        repeat = (
            f"mean-of-{totals['runtime_regime_count']}-regimes/"
            f"{len(run_dirs)}-runs"
        )
        note = (
            "AegisKV recovery path shares are equal-regime means from direct "
            "runtime replay counters; other policy rows remain preliminary "
            "model baselines"
        )
    else:
        if not args.run_dir:
            raise ValueError("at least one --run-dir or --regime is required")
        aggregate, totals = aggregate_run_dirs(args.run_dir)
        run_dirs = list(args.run_dir)
        repeat = f"mean-of-{len(run_dirs)}"
        note = (
            "AegisKV recovery path shares are direct runtime replay "
            "counters from request_details; other policy rows remain "
            "preliminary model baselines"
        )

    current_sha = git_sha()
    replaced = False
    for row in rows:
        if row.get("policy") == "AegisKV":
            row.update(aggregate)
            row.update(
                {
                    "evidence_tier": "sglsim-runtime-replay-derived",
                    "source": "sglsim_request_details_replay_recovery_path",
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
