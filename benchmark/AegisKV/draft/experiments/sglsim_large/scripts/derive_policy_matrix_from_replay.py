#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from derive_restore_boundary_from_replay import (
    _float,
    fit_models,
    git_sha,
    replay_rows,
    row_pre_failure_tokens,
    selected_overhead_ms,
    write_csv,
)


POLICIES = ("no backup", "uniform Host", "uniform GPU", "AegisKV")


def metric_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def row_e2e_ms(row: dict[str, str]) -> float:
    for key in ("client_wall_e2e_latency_s", "e2e_latency_s", "total_latency_s"):
        value = _float(row.get(key))
        if value > 0:
            return value * 1000.0
    return 0.0


def recompute_ms(model: dict[str, float], tokens: float) -> float:
    return model["recompute_intercept_ms"] + model["recompute_ms_per_token"] * tokens


def host_ms(model: dict[str, float], tokens: float) -> float:
    if tokens <= 0:
        return 0.0
    return model["host_intercept_ms"] + model["host_ms_per_token"] * tokens


def gpu_plus_recompute_ms(
    model: dict[str, float],
    *,
    peer_tokens: float,
    suffix_tokens: float,
) -> float:
    if peer_tokens <= 0:
        return recompute_ms(model, suffix_tokens)
    return (
        model["gpu_intercept_ms"]
        + model["gpu_ms_per_token"] * peer_tokens
        + model["recompute_ms_per_token"] * max(suffix_tokens, 0.0)
    )


def host_plus_recompute_ms(
    model: dict[str, float],
    *,
    host_tokens: float,
    suffix_tokens: float,
) -> float:
    if host_tokens <= 0:
        return recompute_ms(model, suffix_tokens)
    value = host_ms(model, host_tokens)
    if suffix_tokens > 0:
        value += model["recompute_ms_per_token"] * suffix_tokens
    return value


def host_visible_tokens(row: dict[str, str], tokens: float) -> float:
    visible = _float(row.get("aegis_host_backup_lookup_tokens"))
    if visible <= 0:
        visible = _float(row.get("aegis_replay_restored_tokens"))
    return min(max(visible, 0.0), tokens)


def peer_visible_tokens(row: dict[str, str], tokens: float) -> float:
    visible = _float(row.get("aegis_peer_gpu_lookup_tokens"))
    return min(max(visible, 0.0), tokens)


def policy_overhead_ms(
    model: dict[str, float],
    row: dict[str, str],
    policy: str,
) -> float:
    tokens = max(row_pre_failure_tokens(row), 0.0)
    if policy == "no backup":
        return recompute_ms(model, tokens)
    if policy == "uniform Host":
        host_tokens = host_visible_tokens(row, tokens)
        return host_plus_recompute_ms(
            model,
            host_tokens=host_tokens,
            suffix_tokens=max(tokens - host_tokens, 0.0),
        )
    if policy == "uniform GPU":
        peer_tokens = peer_visible_tokens(row, tokens)
        return gpu_plus_recompute_ms(
            model,
            peer_tokens=peer_tokens,
            suffix_tokens=max(tokens - peer_tokens, 0.0),
        )
    if policy == "AegisKV":
        return selected_overhead_ms(row)
    raise ValueError(f"unknown policy: {policy}")


def policy_path_tokens(row: dict[str, str], policy: str) -> dict[str, float]:
    tokens = max(row_pre_failure_tokens(row), 0.0)
    if policy == "no backup":
        return {"peer_gpu": 0.0, "host_h2d": 0.0, "ordinary_cache": 0.0, "recompute": tokens}
    if policy == "uniform Host":
        host_tokens = host_visible_tokens(row, tokens)
        return {
            "peer_gpu": 0.0,
            "host_h2d": host_tokens,
            "ordinary_cache": 0.0,
            "recompute": max(tokens - host_tokens, 0.0),
        }
    if policy == "uniform GPU":
        peer_tokens = peer_visible_tokens(row, tokens)
        return {
            "peer_gpu": peer_tokens,
            "host_h2d": 0.0,
            "ordinary_cache": 0.0,
            "recompute": max(tokens - peer_tokens, 0.0),
        }
    if policy == "AegisKV":
        action = str(row.get("aegis_replay_restore_action") or "")
        restored = max(_float(row.get("aegis_replay_restored_tokens")), 0.0)
        recompute = max(_float(row.get("aegis_replay_recompute_tokens")), 0.0)
        out = {"peer_gpu": 0.0, "host_h2d": 0.0, "ordinary_cache": 0.0, "recompute": recompute}
        if action == "gpu_replica":
            out["peer_gpu"] = restored
        elif action == "ordinary_cache":
            out["ordinary_cache"] = restored
        elif restored > 0:
            out["host_h2d"] = restored
        return out
    raise ValueError(f"unknown policy: {policy}")


def collect_action_rows(
    *,
    host_run_dirs: list[Path],
    gpu_run_dirs: list[Path],
    recompute_run_dirs: list[Path],
) -> dict[str, list[dict[str, str]]]:
    rows_by_action = {
        "host_restore": [
            row
            for row in replay_rows(host_run_dirs)
            if str(row.get("aegis_replay_restore_action")) == "host_restore"
        ],
        "gpu_replica": [
            row
            for row in replay_rows(gpu_run_dirs)
            if str(row.get("aegis_replay_restore_action")) == "gpu_replica"
        ],
        "recompute": [
            row
            for row in replay_rows(recompute_run_dirs)
            if str(row.get("aegis_replay_restore_action")) == "recompute"
        ],
    }
    if any(not rows for rows in rows_by_action.values()):
        raise ValueError("host, gpu, and recompute replay rows are all required")
    return rows_by_action


def matrix_run_dirs(matrix_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for matrix_dir in matrix_dirs:
        summary = json.loads((matrix_dir / "matrix_summary.json").read_text())
        out.extend(Path(path) for path in summary["run_dirs"])
    return out


def row_wct_ratio(
    model: dict[str, float],
    row: dict[str, str],
    policy: str,
) -> float | None:
    e2e = row_e2e_ms(row)
    if e2e <= 0:
        return None
    selected = selected_overhead_ms(row)
    tokens = row_pre_failure_tokens(row)
    no_backup_e2e = e2e - selected + recompute_ms(model, tokens)
    if no_backup_e2e <= 0:
        return None
    policy_e2e = e2e - selected + policy_overhead_ms(model, row, policy)
    return policy_e2e / no_backup_e2e


def summarize_regime_values(
    *,
    model: dict[str, float],
    regimes: dict[str, list[dict[str, str]]],
    policy: str,
    value_fn,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    for label, rows in regimes.items():
        values = [value for row in rows if (value := value_fn(model, row, policy)) is not None]
        if not values:
            raise ValueError(f"no values for {policy} in regime {label}")
        summaries.append(
            {
                "label": label,
                "rows": len(values),
                "mean": metric_mean(values),
                "p95": percentile(values, 95),
            }
        )
    means = [float(row["mean"]) for row in summaries]
    p95s = [float(row["p95"]) for row in summaries]
    return (
        {
            "mean": metric_mean(means),
            "mean_std": metric_std(means),
            "mean_ci95": metric_ci95(means),
            "p95": metric_mean(p95s),
            "p95_std": metric_std(p95s),
            "p95_ci95": metric_ci95(p95s),
        },
        summaries,
    )


def latency_ratio(
    model: dict[str, float],
    row: dict[str, str],
    policy: str,
) -> float | None:
    tokens = row_pre_failure_tokens(row)
    baseline = recompute_ms(model, tokens)
    if baseline <= 0:
        return None
    return policy_overhead_ms(model, row, policy) / baseline


def aggregate_path_shares(
    regimes: dict[str, list[dict[str, str]]],
    policy: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    metrics = ("peer_gpu", "host_h2d", "ordinary_cache", "recompute")
    summaries: list[dict[str, object]] = []
    for label, rows in regimes.items():
        totals = {metric: 0.0 for metric in metrics}
        for row in rows:
            row_tokens = policy_path_tokens(row, policy)
            for metric in metrics:
                totals[metric] += row_tokens[metric]
        denominator = sum(totals.values())
        if denominator <= 0:
            raise ValueError(f"zero recovery tokens for {policy} in {label}")
        summaries.append(
            {
                "label": label,
                "rows": len(rows),
                "tokens": denominator,
                **{metric: totals[metric] / denominator for metric in metrics},
            }
        )
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
    return aggregate, summaries


def base_meta(
    *,
    run_id: str,
    model_path: str,
    run_dirs: list[Path],
    regimes: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    return {
        "evidence_tier": "sglsim-runtime-replay-calibrated",
        "source": "sglsim_multiregime_replay_policy_counterfactual",
        "run_id": run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": model_path,
        "workload_id": "aegiskv-sglsim-multiregime-policy-counterfactual",
        "dp_size": "mixed",
        "tp_size": 1,
        "failure_rank": "all",
        "repeat": f"mean-of-{len(regimes)}-regimes/{len(run_dirs)}-runs",
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(str(path) for path in run_dirs),
        "runtime_replay_rows": sum(len(rows) for rows in regimes.values()),
        "runtime_regime_count": len(regimes),
        "runtime_regimes": ";".join(regimes.keys()),
    }


def make_fig8_rows(
    *,
    model: dict[str, float],
    regimes: dict[str, list[dict[str, str]]],
    meta: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        summary, regime_summaries = summarize_regime_values(
            model=model,
            regimes=regimes,
            policy=policy,
            value_fn=row_wct_ratio,
        )
        rows.append(
            {
                "policy": policy,
                "normalized_wct": summary["mean"],
                "normalized_wct_std": summary["mean_std"],
                "normalized_wct_ci95": summary["mean_ci95"],
                **meta,
                "note": (
                    "All rows are equal-regime hard-failover replay "
                    "counterfactuals. No backup recomputes lost tokens; uniform "
                    "Host restores Host-visible tokens; uniform GPU restores "
                    "peer-visible tokens and recomputes suffixes; AegisKV uses "
                    "the observed runtime lower-envelope replay plan."
                ),
                "runtime_regime_summaries": json.dumps(regime_summaries, sort_keys=True),
            }
        )
    return rows


def make_fig9_rows(
    *,
    model: dict[str, float],
    regimes: dict[str, list[dict[str, str]]],
    meta: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        summary, regime_summaries = summarize_regime_values(
            model=model,
            regimes=regimes,
            policy=policy,
            value_fn=latency_ratio,
        )
        rows.append(
            {
                "policy": policy,
                "retry_latency_norm": summary["mean"],
                "critical_agent_latency_norm": summary["p95"],
                "retry_latency_norm_std": summary["mean_std"],
                "retry_latency_norm_ci95": summary["mean_ci95"],
                "critical_agent_latency_norm_std": summary["p95_std"],
                "critical_agent_latency_norm_ci95": summary["p95_ci95"],
                **meta,
                "note": (
                    "All rows are equal-regime hard-failover replay recovery "
                    "overhead counterfactuals normalized to same-length "
                    "recompute; critical-agent latency is the equal-regime mean "
                    "of per-regime p95 ratios."
                ),
                "runtime_regime_summaries": json.dumps(regime_summaries, sort_keys=True),
            }
        )
    return rows


def make_fig10_rows(
    *,
    regimes: dict[str, list[dict[str, str]]],
    meta: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        shares, regime_summaries = aggregate_path_shares(regimes, policy)
        rows.append(
            {
                "policy": policy,
                **shares,
                **meta,
                "note": (
                    "All rows are equal-regime hard-failover replay path-share "
                    "counterfactuals over recovered tokens."
                ),
                "runtime_regime_summaries": json.dumps(regime_summaries, sort_keys=True),
            }
        )
    return rows


def write_summary(path: Path, *, run_id: str, rows_by_fig: dict[str, list[dict[str, object]]]) -> None:
    lines = [
        "# Fig. 8/9/10 Multi-Policy Replay Counterfactual",
        "",
        f"- run_id: `{run_id}`",
        "- source: `sglsim_multiregime_replay_policy_counterfactual`",
        "",
    ]
    for figure, rows in rows_by_fig.items():
        lines.extend([f"## {figure}", "", "| policy | primary metric | secondary |", "|---|---:|---:|"])
        for row in rows:
            if figure == "Fig. 8":
                primary = float(row["normalized_wct"])
                secondary = float(row.get("normalized_wct_ci95", 0.0))
            elif figure == "Fig. 9":
                primary = float(row["retry_latency_norm"])
                secondary = float(row["critical_agent_latency_norm"])
            else:
                primary = float(row["recompute"])
                secondary = float(row["host_h2d"]) + float(row["peer_gpu"])
            lines.append(f"| {row['policy']} | {primary:.4f} | {secondary:.4f} |")
        lines.append("")
    lines.append(
        "These are replay-calibrated counterfactual rows, not final physical KV-copy policy sweeps."
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 8/9/10 policy rows from the same multi-regime "
            "hard-failover replay data instead of the older synthesis policy model."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fig8-csv", type=Path, required=True)
    parser.add_argument("--fig9-csv", type=Path, required=True)
    parser.add_argument("--fig10-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--host-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--gpu-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--recompute-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--host-matrix-dir", type=Path, action="append", default=[])
    parser.add_argument("--gpu-matrix-dir", type=Path, action="append", default=[])
    parser.add_argument("--recompute-matrix-dir", type=Path, action="append", default=[])
    parser.add_argument("--model-path", default="mock-model")
    args = parser.parse_args()

    host_run_dirs = [*args.host_run_dir, *matrix_run_dirs(args.host_matrix_dir)]
    gpu_run_dirs = [*args.gpu_run_dir, *matrix_run_dirs(args.gpu_matrix_dir)]
    recompute_run_dirs = [
        *args.recompute_run_dir,
        *matrix_run_dirs(args.recompute_matrix_dir),
    ]
    if not host_run_dirs or not gpu_run_dirs or not recompute_run_dirs:
        raise ValueError("host, gpu, and recompute run dirs are all required")

    rows_by_action = collect_action_rows(
        host_run_dirs=host_run_dirs,
        gpu_run_dirs=gpu_run_dirs,
        recompute_run_dirs=recompute_run_dirs,
    )
    model = fit_models(
        host_rows=rows_by_action["host_restore"],
        gpu_rows=rows_by_action["gpu_replica"],
        recompute_rows=rows_by_action["recompute"],
    )
    regimes = {
        "long_prefix_host": rows_by_action["host_restore"],
        "short_peer_gpu": rows_by_action["gpu_replica"],
        "short_nonpeer_recompute": rows_by_action["recompute"],
    }
    run_dirs = [*host_run_dirs, *gpu_run_dirs, *recompute_run_dirs]
    meta = base_meta(
        run_id=args.run_id,
        model_path=args.model_path,
        run_dirs=run_dirs,
        regimes=regimes,
    )
    fig8_rows = make_fig8_rows(model=model, regimes=regimes, meta=meta)
    fig9_rows = make_fig9_rows(model=model, regimes=regimes, meta=meta)
    fig10_rows = make_fig10_rows(regimes=regimes, meta=meta)

    write_csv(args.fig8_csv, fig8_rows)
    write_csv(args.fig9_csv, fig9_rows)
    write_csv(args.fig10_csv, fig10_rows)
    rows_by_fig = {"Fig. 8": fig8_rows, "Fig. 9": fig9_rows, "Fig. 10": fig10_rows}
    if args.summary_md:
        write_summary(args.summary_md, run_id=args.run_id, rows_by_fig=rows_by_fig)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "fig8_csv": str(args.fig8_csv),
                "fig9_csv": str(args.fig9_csv),
                "fig10_csv": str(args.fig10_csv),
                "runtime_replay_rows": meta["runtime_replay_rows"],
                "model": model,
                "rows": {figure: len(rows) for figure, rows in rows_by_fig.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
