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


def row_e2e_ms(row: dict[str, str]) -> float:
    for key in ("client_wall_e2e_latency_s", "e2e_latency_s", "total_latency_s"):
        value = _float(row.get(key))
        if value > 0:
            return value * 1000.0
    return 0.0


def selected_restore_ms(row: dict[str, str]) -> float:
    selected = _float(row.get("aegis_replay_restore_overhead_s")) * 1000.0
    if selected <= 0:
        selected = _float(row.get("aegis_replay_restore_sleep_s")) * 1000.0
    return max(selected, 0.0)


def run_wct(run_dir: Path) -> dict[str, float]:
    ratios: list[float] = []
    actual_ms: list[float] = []
    counterfactual_ms: list[float] = []
    selected_ms: list[float] = []
    recompute_ms: list[float] = []
    recovery_tokens = 0.0
    for row in replay_rows(run_dir):
        e2e = row_e2e_ms(row)
        if e2e <= 0:
            continue
        selected = selected_restore_ms(row)
        recompute = recompute_baseline_ms(row_pre_failure_tokens(row))
        no_backup = e2e - selected + recompute
        if no_backup <= 0:
            continue
        ratios.append(e2e / no_backup)
        actual_ms.append(e2e)
        counterfactual_ms.append(no_backup)
        selected_ms.append(selected)
        recompute_ms.append(recompute)
        recovery_tokens += _float(row.get("aegis_replay_restored_tokens"))
        recovery_tokens += _float(row.get("aegis_replay_recompute_tokens"))

    if not ratios:
        raise ValueError(f"no valid replay WCT rows in {run_dir}")
    return {
        "normalized_wct": mean(ratios),
        "mean_actual_e2e_ms": mean(actual_ms),
        "mean_no_backup_counterfactual_ms": mean(counterfactual_ms),
        "mean_selected_restore_ms": mean(selected_ms),
        "mean_recompute_counterfactual_ms": mean(recompute_ms),
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


def aggregate_regimes(
    regimes: list[tuple[str, Path]],
) -> tuple[dict[str, float], dict[str, object]]:
    by_regime: dict[str, list[Path]] = {}
    for label, path in regimes:
        by_regime.setdefault(label, []).append(path)
    if not by_regime:
        raise ValueError("no regimes supplied")

    summaries: list[dict[str, object]] = []
    for label, paths in by_regime.items():
        runs = [run_wct(path) for path in paths]
        summaries.append(
            {
                "label": label,
                "runs": len(paths),
                "replay_rows": int(sum(row["replay_rows"] for row in runs)),
                "recovery_tokens": int(sum(row["recovery_tokens"] for row in runs)),
                "normalized_wct": metric_mean([row["normalized_wct"] for row in runs]),
                "mean_actual_e2e_ms": metric_mean(
                    [row["mean_actual_e2e_ms"] for row in runs]
                ),
                "mean_no_backup_counterfactual_ms": metric_mean(
                    [row["mean_no_backup_counterfactual_ms"] for row in runs]
                ),
                "mean_selected_restore_ms": metric_mean(
                    [row["mean_selected_restore_ms"] for row in runs]
                ),
                "mean_recompute_counterfactual_ms": metric_mean(
                    [row["mean_recompute_counterfactual_ms"] for row in runs]
                ),
            }
        )

    values = [float(row["normalized_wct"]) for row in summaries]
    aggregate = {
        "normalized_wct": metric_mean(values),
        "normalized_wct_std": metric_std(values),
        "normalized_wct_ci95": metric_ci95(values),
    }
    totals = {
        "runtime_replay_rows": int(sum(int(row["replay_rows"]) for row in summaries)),
        "runtime_recovery_tokens": int(sum(int(row["recovery_tokens"]) for row in summaries)),
        "runtime_regime_count": len(summaries),
        "runtime_regimes": ";".join(str(row["label"]) for row in summaries),
        "runtime_regime_summaries": json.dumps(summaries, sort_keys=True),
        "runtime_mean_actual_e2e_ms": metric_mean(
            [float(row["mean_actual_e2e_ms"]) for row in summaries]
        ),
        "runtime_mean_no_backup_counterfactual_ms": metric_mean(
            [float(row["mean_no_backup_counterfactual_ms"]) for row in summaries]
        ),
        "runtime_mean_selected_restore_ms": metric_mean(
            [float(row["mean_selected_restore_ms"]) for row in summaries]
        ),
        "runtime_mean_recompute_counterfactual_ms": metric_mean(
            [float(row["mean_recompute_counterfactual_ms"]) for row in summaries]
        ),
    }
    return aggregate, totals


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace Fig. 8 AegisKV row with direct replay e2e WCT ratios "
            "against same-row no-backup recompute counterfactuals."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--regime", type=parse_regime, action="append", required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = [dict(row) for row in read_csv(args.input_csv)]
    aggregate, totals = aggregate_regimes(args.regime)
    run_dirs = [path for _, path in args.regime]
    current_sha = git_sha()
    replaced = False
    for row in rows:
        if row.get("policy") == "AegisKV":
            row.update(aggregate)
            row.update(
                {
                    "evidence_tier": "sglsim-runtime-replay-derived",
                    "source": "sglsim_request_details_replay_overall_wct_counterfactual",
                    "run_id": args.run_id,
                    "git_sha": current_sha,
                    "repeat": (
                        f"mean-of-{totals['runtime_regime_count']}-regimes/"
                        f"{len(run_dirs)}-runs"
                    ),
                    "note": (
                        "AegisKV WCT row is an equal-regime mean of observed "
                        "hard-failure replay e2e divided by a same-row no-backup "
                        "recompute counterfactual; other policy rows remain "
                        "preliminary model baselines"
                    ),
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
