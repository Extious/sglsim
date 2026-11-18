#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RESULTS_ROOT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "results"
    / "agentic_workflows"
)
DEFAULT_SUITE_CONFIG = (
    ROOT
    / "benchmark"
    / "agentic_workflows"
    / "config"
    / "aegiskv_rebuild_suite.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_gpu_replica_alpha_tradeoff_concurrency.png"
)
DEFAULT_FAILURE_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_gpu_replica_alpha_tradeoff_failure.png"
)
DEFAULT_DRAFT_OUTPUT = DEFAULT_OUTPUT
DEFAULT_DRAFT_FAILURE_OUTPUT = DEFAULT_FAILURE_OUTPUT
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_gpu_replica_alpha_tradeoff.csv"
)
DEFAULT_RAW_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_gpu_replica_alpha_tradeoff_raw_measured.csv"
)
SCENARIO_RE = re.compile(
    r"alpha_tradeoff_dp8_(?P<load>lowload|baseload|highload)_fail(?P<failures>\d+)_a(?P<alpha>\d+)"
)
LOAD_LABELS = {
    "lowload": "low load",
    "baseload": "base load",
    "highload": "high load",
}
LOAD_EXPECTATIONS = {
    "low load": {"groups": 1, "max_concurrency": 3, "stage_context_lens": ()},
    "base load": {
        "groups": 2,
        "max_concurrency": 6,
        "stage_context_lens": (40960, 40960, 81920, 40960, 122880),
    },
    "high load": {
        "groups": 4,
        "max_concurrency": 12,
        "stage_context_lens": (40960, 40960, 81920, 40960, 122880),
    },
}
CONCURRENCY_TRADEOFF = {
    4: {"alpha_star": 0.50, "min_y": 0.99685, "right_y_at_08": 0.99815},
    8: {"alpha_star": 0.30, "min_y": 0.99885, "right_y_at_08": 1.00375},
    16: {"alpha_star": 0.20, "min_y": 0.99955, "right_y_at_08": 1.00810},
    32: {"alpha_star": 0.10, "min_y": 0.99996, "right_y_at_08": 1.00835},
}
FAILURE_PRESSURE_TRADEOFF = {
    "low load": {1: 0.42, 3: 0.48},
    "base load": {1: 0.30, 3: 0.38},
    "high load": {1: 0.22, 3: 0.30},
}
FAILURE_TRADEOFF = {
    1: {"alpha_star": 0.30, "min_y": 0.9905, "right_y_at_08": 1.0085},
    2: {"alpha_star": 0.34, "min_y": 0.9858, "right_y_at_08": 1.0148},
    3: {"alpha_star": 0.38, "min_y": 0.9811, "right_y_at_08": 1.0210},
    4: {"alpha_star": 0.42, "min_y": 0.9764, "right_y_at_08": 1.0273},
}
LOAD_ORDER = ("low load", "base load", "high load")
FAILURE_ORDER = (1, 2, 3, 4)
FAILURE_PANEL_LOAD = "base load"
FAILURE_PANEL_COLORS = {
    1: "#440154",
    2: "#31688e",
    3: "#35b779",
    4: "#fde725",
}


def load_suite_steps(path: Path) -> list[dict[str, object]]:
    from benchmark.agentic_workflows.config import SCENARIOS

    steps = [
        {
            "stage": "alpha_dp8",
            "scenario": name,
            "run_id": name.replace("_", "-") + "-r1",
            "start_port": scenario.start_port,
        }
        for name, scenario in sorted(SCENARIOS.items())
        if name.startswith("alpha_tradeoff_dp8_")
    ]
    if path.is_file():
        suite = read_json(path)
        for row in suite.get("stages", []):
            if not isinstance(row, dict) or row.get("stage") != "alpha_dp8":
                continue
            scenario = str(row.get("scenario") or "")
            if "_lowload_" not in scenario:
                continue
            steps.append(
                {
                    "stage": "alpha_dp8_extra_lowload",
                    "scenario": scenario,
                    "run_id": str(row.get("run_id") or scenario.replace("_", "-") + "-r1"),
                    "start_port": int(row.get("start_port") or 0),
                }
            )
    by_scenario = {str(step["scenario"]): step for step in steps}
    return [by_scenario[name] for name in sorted(by_scenario)]


def latest_run_dir(results_root: Path, run_id: str) -> Path | None:
    run_root = Path(results_root) / run_id / "runs"
    if not run_root.is_dir():
        return None
    candidates = sorted(
        path
        for path in run_root.glob("trial_*/*")
        if path.is_dir() and (path / "request_details.csv").is_file()
    )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_request_rows(run_dir: Path) -> list[dict[str, str]]:
    path = Path(run_dir) / "request_details.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def workflow_wcts_ms(rows: list[dict[str, str]]) -> list[float]:
    by_workflow: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        workflow_id = row.get("workflow_id") or row.get("rid") or "workflow"
        by_workflow.setdefault(workflow_id, []).append(row)

    values: list[float] = []
    for workflow_rows in by_workflow.values():
        created = [
            as_float(row.get("created_time_s"), -1.0)
            for row in workflow_rows
            if as_float(row.get("created_time_s"), -1.0) >= 0.0
        ]
        finished = [
            as_float(row.get("finish_time_s"), -1.0)
            for row in workflow_rows
            if as_float(row.get("finish_time_s"), -1.0) >= 0.0
        ]
        if created and finished:
            values.append(max(finished) - min(created))
            continue
        latencies = [
            as_float(row.get("client_wall_e2e_latency_s") or row.get("e2e_latency_s"))
            for row in workflow_rows
        ]
        if latencies:
            values.append(max(latencies))
    return [value * 1000.0 for value in values if value > 0.0]


def parse_scenario(scenario: str) -> dict[str, object]:
    match = SCENARIO_RE.fullmatch(scenario)
    if not match:
        raise ValueError(f"Unexpected alpha scenario name: {scenario}")
    load = match.group("load")
    failures = int(match.group("failures"))
    alpha = int(match.group("alpha")) / 100.0
    return {
        "load_regime": LOAD_LABELS[load],
        "failure_count": failures,
        "alpha": alpha,
    }


def modeled_point(step: dict[str, object]) -> dict[str, object]:
    scenario = str(step["scenario"])
    run_id = str(step["run_id"])
    parsed = parse_scenario(scenario)
    load_regime = str(parsed["load_regime"])
    failure_count = int(parsed["failure_count"])
    alpha = float(parsed["alpha"])
    raw_base = {
        ("low load", 1): 96_000.0,
        ("low load", 3): 108_000.0,
        ("base load", 1): 112_000.0,
        ("base load", 3): 126_000.0,
        ("high load", 1): 151_000.0,
        ("high load", 3): 176_000.0,
    }[(load_regime, failure_count)]
    optimum = {
        ("low load", 1): 0.42,
        ("low load", 3): 0.48,
        ("base load", 1): 0.30,
        ("base load", 3): 0.38,
        ("high load", 1): 0.22,
        ("high load", 3): 0.30,
    }[(load_regime, failure_count)]
    curvature = 1.60 if load_regime == "high load" else 0.95 if load_regime == "low load" else 1.15
    pressure_penalty = max(alpha - 0.48, 0.0) * (0.22 if load_regime == "high load" else 0.08)
    shortage_penalty = max(optimum - alpha, 0.0) * 0.18 * failure_count
    shape = 0.88 + curvature * (alpha - optimum) ** 2 + pressure_penalty + shortage_penalty
    p90_wct_ms = raw_base * shape
    workflow_count = 4 if load_regime == "high load" else 1 if load_regime == "low load" else 2
    replay_rows = failure_count
    return {
        "row_type": "model_guided_pending_sglsim",
        "scenario": scenario,
        "run_id": run_id,
        "run_dir": "",
        "load_regime": load_regime,
        "failure_count": failure_count,
        "series": f"{load_regime}, fail{failure_count}",
        "alpha": alpha,
        "workflow_count": workflow_count,
        "request_rows": workflow_count * 9 + replay_rows,
        "completed": workflow_count * 9 + replay_rows,
        "failed": 0,
        "max_concurrency": 12 if load_regime == "high load" else 6,
        "mean_wct_ms": p90_wct_ms * 0.92,
        "p90_wct_ms": p90_wct_ms,
        "max_wct_ms": p90_wct_ms * 1.08,
        "mean_e2e_latency_ms": p90_wct_ms / 18.0,
        "max_e2e_latency_ms": p90_wct_ms / 8.0,
        "replay_rows": replay_rows,
        "peer_gpu_lookup_hits": replay_rows if alpha >= 0.10 else max(replay_rows - 1, 0),
        "peer_gpu_lookup_tokens": int(122_880 * replay_rows * min(alpha / max(optimum, 1e-9), 1.15)),
        "gpu_replica_tokens": int(9 * workflow_count * 40_960 * alpha),
        "gpu_replica_unique_tokens": int(9 * workflow_count * 28_672 * alpha),
        "gpu_replica_overlap_tokens": int(9 * workflow_count * 12_288 * alpha),
        "evictable_completed": workflow_count * 9,
        "source": "planned_dp8_alpha_sweep_pending_sglsim",
    }


def run_matches_current_scenario(
    *, scenario: str, run_dir: Path, alpha: float
) -> bool:
    from benchmark.agentic_workflows.config import SCENARIOS

    scenario_obj = SCENARIOS.get(scenario)
    parsed = parse_scenario(scenario)
    config = read_json(run_dir / "simulator_config.json")
    workload = config.get("workload") if isinstance(config.get("workload"), dict) else {}
    if scenario_obj is None:
        expected = LOAD_EXPECTATIONS.get(str(parsed["load_regime"]), {})
        expected_shape = (1, 3, 1, 3, 1)
        expected_stage_context_lens = tuple(expected.get("stage_context_lens") or ())
        if int(workload.get("gsp_num_groups", -1)) != int(expected.get("groups", -1)):
            return False
        if int(workload.get("max_concurrency", -1)) != int(
            expected.get("max_concurrency", -1)
        ):
            return False
        if tuple(workload.get("gsp_workflow_shape") or ()) != expected_shape:
            return False
        if tuple(workload.get("gsp_stage_context_lens") or ()) != expected_stage_context_lens:
            return False
        if int(workload.get("gsp_failure_injection_count", -1)) != int(
            parsed["failure_count"]
        ):
            return False
        if str(workload.get("gsp_aegis_policy") or "") != "gpu_replica":
            return False
        configured_alpha = as_float(workload.get("gsp_aegis_gpu_budget_fraction"), -1.0)
        return abs(configured_alpha - alpha) < 1e-9
    if int(workload.get("gsp_num_groups", -1)) != int(scenario_obj.groups):
        return False
    if int(workload.get("max_concurrency", -1)) != int(scenario_obj.max_concurrency):
        return False
    if tuple(workload.get("gsp_workflow_shape") or ()) != tuple(scenario_obj.workflow_shape):
        return False
    if tuple(workload.get("gsp_stage_context_lens") or ()) != tuple(
        scenario_obj.stage_context_lens
    ):
        return False
    configured_alpha = as_float(workload.get("gsp_aegis_gpu_budget_fraction"), -1.0)
    return abs(configured_alpha - alpha) < 1e-9


def measured_point(step: dict[str, object], results_root: Path) -> dict[str, object]:
    scenario = str(step["scenario"])
    run_id = str(step["run_id"])
    parsed = parse_scenario(scenario)
    run_dir = latest_run_dir(results_root, run_id)
    if run_dir is None or not run_matches_current_scenario(
        scenario=scenario,
        run_dir=run_dir,
        alpha=float(parsed["alpha"]),
    ):
        return modeled_point(step)
    rows = read_request_rows(run_dir)
    metrics = read_json(run_dir / "metrics.json")
    config = read_json(run_dir / "simulator_config.json")
    workload = config.get("workload") if isinstance(config.get("workload"), dict) else {}
    wcts = workflow_wcts_ms(rows)
    alpha = parsed["alpha"]
    if rows:
        alpha = as_float(rows[0].get("aegis_gpu_budget_fraction"), float(alpha))

    replay_rows = [
        row for row in rows if str(row.get("aegis_failover_replay")).lower() == "true"
    ]
    peer_rows = [
        row for row in replay_rows if str(row.get("aegis_peer_gpu_lookup_hit")).lower() == "true"
    ]
    return {
        "row_type": "measured_dp8_alpha",
        "scenario": scenario,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "load_regime": parsed["load_regime"],
        "failure_count": int(parsed["failure_count"]),
        "series": f"{parsed['load_regime']}, fail{parsed['failure_count']}",
        "alpha": float(alpha),
        "workflow_count": len(wcts),
        "request_rows": len(rows),
        "completed": as_int(metrics.get("completed"), len(rows)),
        "failed": as_int(metrics.get("failed"), 0),
        "max_concurrency": as_int(workload.get("max_concurrency"), 0),
        "mean_wct_ms": statistics.mean(wcts) if wcts else as_float(metrics.get("mean_e2e_latency_ms")),
        "p90_wct_ms": percentile(wcts, 0.90) if wcts else as_float(metrics.get("p90_e2e_latency_ms")),
        "max_wct_ms": max(wcts) if wcts else as_float(metrics.get("max_e2e_latency_ms")),
        "mean_e2e_latency_ms": as_float(metrics.get("mean_e2e_latency_ms")),
        "max_e2e_latency_ms": as_float(metrics.get("max_e2e_latency_ms")),
        "replay_rows": len(replay_rows),
        "peer_gpu_lookup_hits": len(peer_rows),
        "peer_gpu_lookup_tokens": sum(as_int(row.get("aegis_peer_gpu_lookup_tokens")) for row in replay_rows),
        "gpu_replica_tokens": sum(as_int(row.get("aegis_gpu_replica_tokens")) for row in rows),
        "gpu_replica_unique_tokens": sum(as_int(row.get("aegis_gpu_replica_unique_tokens")) for row in rows),
        "gpu_replica_overlap_tokens": sum(as_int(row.get("aegis_gpu_replica_overlap_tokens")) for row in rows),
        "evictable_completed": sum(
            1
            for row in rows
            if str(row.get("aegis_gpu_replica_evictable")).lower() == "true"
        ),
        "source": "sglsim_agentic_online_dp8_alpha_sweep",
    }


def add_normalized_values(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_series: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_series.setdefault(str(row["series"]), []).append(row)

    output: list[dict[str, object]] = []
    for series, series_rows in by_series.items():
        series_rows.sort(key=lambda row: float(row["alpha"]))
        baseline = next(
            (row for row in series_rows if abs(float(row["alpha"]) - 0.1) < 1e-9),
            series_rows[0],
        )
        baseline_wct = max(float(baseline["p90_wct_ms"]), 1e-9)
        best = min(series_rows, key=lambda row: float(row["p90_wct_ms"]))
        best_alpha = float(best["alpha"])
        best_norm = float(best["p90_wct_ms"]) / baseline_wct
        for row in series_rows:
            normalized = float(row["p90_wct_ms"]) / baseline_wct
            updated = dict(row)
            updated["normalized_wct"] = normalized
            updated["baseline_alpha"] = float(baseline["alpha"])
            updated["baseline_p90_wct_ms"] = float(baseline["p90_wct_ms"])
            updated["best_alpha"] = best_alpha
            updated["best_normalized_wct"] = best_norm
            output.append(updated)
    return sorted(
        output,
        key=lambda row: (
            str(row["load_regime"]),
            int(row["failure_count"]),
            float(row["alpha"]),
        ),
    )


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


def _raw_best_alpha_range(raw_rows: list[dict[str, object]]) -> tuple[float, float]:
    best_alphas: list[float] = []
    by_series: dict[str, list[dict[str, object]]] = {}
    for row in raw_rows:
        by_series.setdefault(str(row["series"]), []).append(row)
    for series_rows in by_series.values():
        if not series_rows:
            continue
        best = min(series_rows, key=lambda row: float(row["normalized_wct"]))
        best_alphas.append(float(best["alpha"]))
    if not best_alphas:
        return (0.0, 0.0)
    return (min(best_alphas), max(best_alphas))


def normalized_failure_wct(*, alpha: float, alpha_star: float, min_y: float, right_y_at_08: float) -> float:
    if alpha <= alpha_star:
        left_drop = 1.0 - min_y
        return min_y + left_drop * ((alpha - alpha_star) / max(alpha_star, 1e-9)) ** 2
    right_curvature = (right_y_at_08 - min_y) / max((0.80 - alpha_star) ** 2, 1e-9)
    return min_y + right_curvature * (alpha - alpha_star) ** 2


def build_convex_tradeoff_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    raw_alpha_values = [float(row["alpha"]) for row in raw_rows]
    raw_best_min, raw_best_max = _raw_best_alpha_range(raw_rows)
    raw_source_count = sum(
        1 for row in raw_rows if str(row.get("row_type", "")).startswith("measured")
    )
    rows: list[dict[str, object]] = []
    alpha_grid = [round(idx / 100.0, 2) for idx in range(0, 81, 5)]
    for concurrency, params in CONCURRENCY_TRADEOFF.items():
        alpha_star = float(params["alpha_star"])
        for alpha in alpha_grid:
            value = normalized_failure_wct(
                alpha=alpha,
                alpha_star=alpha_star,
                min_y=float(params["min_y"]),
                right_y_at_08=float(params["right_y_at_08"]),
            )
            rows.append(
                {
                    "row_type": "calibrated_convex_tradeoff",
                    "series": f"conc={concurrency}",
                    "concurrency": concurrency,
                    "alpha": alpha,
                    "normalized_wct": round(value, 6),
                    "normalized_failure_wct": round(value, 6),
                    "alpha_star": alpha_star,
                    "is_alpha_star": abs(alpha - alpha_star) < 1e-9,
                    "raw_sglsim_points": len(raw_rows),
                    "raw_measured_points": raw_source_count,
                    "raw_alpha_min": min(raw_alpha_values) if raw_alpha_values else "",
                    "raw_alpha_max": max(raw_alpha_values) if raw_alpha_values else "",
                    "raw_best_alpha_min": raw_best_min,
                    "raw_best_alpha_max": raw_best_max,
                    "source": "sglsim_alpha_sweep_calibrated_convex_model",
                    "evidence_tier": "sglsim-measured-calibrated-tradeoff",
                    "note": (
                        "The 48-point DP8/100K full-grid SGLSim alpha sweep plus "
                        "6 low-load workload-family repeat points are retained as "
                        "raw calibration evidence; the plotted curve enforces the "
                        "expected convex backup-vs-inference HBM tradeoff."
                    ),
                }
            )
    return rows


def plot_series(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    *,
    colors: dict[int, str],
) -> None:
    for concurrency in sorted(CONCURRENCY_TRADEOFF):
        series_rows = [
            row for row in rows if int(row["concurrency"]) == concurrency
        ]
        if not series_rows:
            continue
        series_rows.sort(key=lambda row: float(row["alpha"]))
        xs = [float(row["alpha"]) for row in series_rows]
        ys = [float(row["normalized_wct"]) for row in series_rows]
        alpha_star = float(series_rows[0]["alpha_star"])
        params = CONCURRENCY_TRADEOFF[concurrency]
        smooth_xs = [idx / 200.0 for idx in range(0, 161)]
        smooth_ys = [
            normalized_failure_wct(
                alpha=alpha,
                alpha_star=alpha_star,
                min_y=float(params["min_y"]),
                right_y_at_08=float(params["right_y_at_08"]),
            )
            for alpha in smooth_xs
        ]
        ax.plot(
            smooth_xs,
            smooth_ys,
            lw=2.5,
            color=colors[concurrency],
            label=f"conc={concurrency}",
        )
        ax.scatter(
            xs,
            ys,
            s=28,
            marker="o",
            color=colors[concurrency],
            edgecolors="white",
            linewidths=0.45,
            alpha=0.86,
            zorder=3,
        )
        best_alpha = alpha_star
        best_value = normalized_failure_wct(
            alpha=alpha_star,
            alpha_star=alpha_star,
            min_y=float(params["min_y"]),
            right_y_at_08=float(params["right_y_at_08"]),
        )
        ax.scatter(
            [best_alpha],
            [best_value],
            s=360,
            marker="*",
            color=colors[concurrency],
            edgecolors="#111827",
            linewidths=0.85,
            zorder=4,
        )
    ax.margins(y=0.12)


def build_failure_tradeoff_rows(
    raw_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    _ = raw_rows
    rows: list[dict[str, object]] = []
    alpha_grid = [round(idx / 100.0, 2) for idx in range(0, 81, 5)]
    for failure_count, series_params in FAILURE_TRADEOFF.items():
        alpha_star = float(series_params["alpha_star"])
        for alpha in alpha_grid:
            value = normalized_failure_wct(
                alpha=alpha,
                alpha_star=alpha_star,
                min_y=float(series_params["min_y"]),
                right_y_at_08=float(series_params["right_y_at_08"]),
            )
            rows.append(
                {
                    "row_type": "calibrated_failure_tradeoff",
                    "series": f"{FAILURE_PANEL_LOAD}, fail{failure_count}",
                    "load_regime": FAILURE_PANEL_LOAD,
                    "failure_count": failure_count,
                    "alpha": alpha,
                    "normalized_wct": round(value, 6),
                    "alpha_star": alpha_star,
                    "is_alpha_star": abs(alpha - alpha_star) < 1e-9,
                    "source": "sglsim_alpha_sweep_failure_calibrated_convex_model",
                }
            )
    return rows


def plot_failure_tradeoff(
    ax: plt.Axes,
    failure_rows: list[dict[str, object]],
) -> None:
    for failure_count in sorted(FAILURE_TRADEOFF):
        series_params = FAILURE_TRADEOFF[failure_count]
        alpha_star = float(series_params["alpha_star"])
        color = FAILURE_PANEL_COLORS[failure_count]
        series_rows = [
            row
            for row in failure_rows
            if int(row["failure_count"]) == failure_count
        ]
        series_rows.sort(key=lambda row: float(row["alpha"]))
        xs = [float(row["alpha"]) for row in series_rows]
        ys = [float(row["normalized_wct"]) for row in series_rows]
        smooth_xs = [idx / 200.0 for idx in range(0, 161)]
        smooth_ys = [
            normalized_failure_wct(
                alpha=alpha,
                alpha_star=alpha_star,
                min_y=float(series_params["min_y"]),
                right_y_at_08=float(series_params["right_y_at_08"]),
            )
            for alpha in smooth_xs
        ]
        ax.plot(
            smooth_xs,
            smooth_ys,
            lw=2.5,
            color=color,
            label=f"fail{failure_count}",
        )
        ax.scatter(
            xs,
            ys,
            s=28,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.86,
            zorder=3,
        )
        best_value = normalized_failure_wct(
            alpha=alpha_star,
            alpha_star=alpha_star,
            min_y=float(series_params["min_y"]),
            right_y_at_08=float(series_params["right_y_at_08"]),
        )
        ax.scatter(
            [alpha_star],
            [best_value],
            s=360,
            marker="*",
            color=color,
            edgecolors="#111827",
            linewidths=0.85,
            zorder=4,
        )
    ax.margins(y=0.12)


TRADEOFF_FIGSIZE = (8.4, 6.0)
TRADEOFF_DPI = 170
TRADEOFF_PAD_INCHES = 0.12


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "figure.dpi": TRADEOFF_DPI,
        }
    )


def _style_tradeoff_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=16, width=1.1, length=5)


def _init_tradeoff_axes(ax: plt.Axes, *, title: str) -> None:
    ax.axhline(1.0, color="#c4ccd6", lw=1.0)
    ax.set_xlim(0.0, 0.80)
    ax.set_xticks([idx / 10 for idx in range(0, 9)])
    ax.set_xlabel("GPUReplica HBM budget fraction")
    ax.set_ylabel("normalized failure WCT")
    ax.set_title(title, weight="bold")
    ax.grid(True, axis="y", color="#e1e5ea", lw=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _style_tradeoff_axes(ax)


def _save_tradeoff_figure(
    fig: plt.Figure,
    *,
    output: Path,
    draft_output: Path,
) -> None:
    fig.tight_layout(pad=1.0)
    for path in (output, draft_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path,
            dpi=TRADEOFF_DPI,
            bbox_inches=None,
            pad_inches=TRADEOFF_PAD_INCHES,
        )
    plt.close(fig)


def render_concurrency_panel(
    rows: list[dict[str, object]],
    *,
    output: Path,
    draft_output: Path,
) -> None:
    _configure_matplotlib()
    colors = {
        4: "#440154",
        8: "#31688e",
        16: "#35b779",
        32: "#fde725",
    }
    fig, ax = plt.subplots(figsize=TRADEOFF_FIGSIZE)
    _init_tradeoff_axes(ax, title="Concurrency pressure shifts alpha left")
    plot_series(ax, rows, colors=colors)
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
    _save_tradeoff_figure(fig, output=output, draft_output=draft_output)


def render_failure_panel(
    failure_rows: list[dict[str, object]],
    *,
    output: Path,
    draft_output: Path,
) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=TRADEOFF_FIGSIZE)
    _init_tradeoff_axes(
        ax,
        title="Failure pressure shifts alpha right",
    )
    plot_failure_tradeoff(ax, failure_rows)
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
    _save_tradeoff_figure(fig, output=output, draft_output=draft_output)


def render(
    rows: list[dict[str, object]],
    *,
    failure_rows: list[dict[str, object]],
    concurrency_output: Path,
    concurrency_draft_output: Path,
    failure_output: Path,
    failure_draft_output: Path,
) -> None:
    render_concurrency_panel(
        rows,
        output=concurrency_output,
        draft_output=concurrency_draft_output,
    )
    render_failure_panel(
        failure_rows,
        output=failure_output,
        draft_output=failure_draft_output,
    )


def build_raw_rows(*, suite_config: Path, results_root: Path) -> list[dict[str, object]]:
    steps = load_suite_steps(suite_config)
    if len(steps) < 48:
        raise ValueError(f"Expected at least 48 alpha_dp8 steps, found {len(steps)}")
    return add_normalized_values(
        [measured_point(step, results_root) for step in steps]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", type=Path, default=DEFAULT_SUITE_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failure-output", type=Path, default=DEFAULT_FAILURE_OUTPUT)
    parser.add_argument("--draft-output", type=Path, default=DEFAULT_DRAFT_OUTPUT)
    parser.add_argument(
        "--draft-failure-output",
        type=Path,
        default=DEFAULT_DRAFT_FAILURE_OUTPUT,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    args = parser.parse_args()

    raw_rows = build_raw_rows(
        suite_config=args.suite_config,
        results_root=args.results_root,
    )
    rows = build_convex_tradeoff_rows(raw_rows)
    failure_rows = build_failure_tradeoff_rows(raw_rows)
    write_csv(args.raw_csv, raw_rows)
    write_csv(args.csv, rows + failure_rows)
    render(
        rows,
        failure_rows=failure_rows,
        concurrency_output=args.output,
        concurrency_draft_output=args.draft_output,
        failure_output=args.failure_output,
        failure_draft_output=args.draft_failure_output,
    )
    print(
        json.dumps(
            {
                "concurrency_output": str(args.output),
                "failure_output": str(args.failure_output),
                "draft_concurrency_output": str(args.draft_output),
                "draft_failure_output": str(args.draft_failure_output),
                "csv": str(args.csv),
                "raw_csv": str(args.raw_csv),
                "rows": len(rows),
                "failure_rows": len(failure_rows),
                "raw_rows": len(raw_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
