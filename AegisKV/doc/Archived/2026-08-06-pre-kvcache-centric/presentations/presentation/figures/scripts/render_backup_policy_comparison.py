#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_backup_policy_comparison.png"
)
DEFAULT_DRAFT_OUTPUT = DEFAULT_OUTPUT
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_backup_policy_comparison.csv"
)

POLICY_LABELS = {
    "no_backup": "No backup",
    "host_only": "HostBackup",
    "gpu_replica": "GPUReplica",
    "aegiskv": "AegisKV (Fix alpha)",
}
POLICY_ORDER = ["no_backup", "host_only", "gpu_replica", "aegiskv"]
POLICY_COLORS = {
    "no_backup": "#6B7280",
    "host_only": "#2563EB",
    "gpu_replica": "#D97706",
    "aegiskv": "#059669",
}


@dataclass(frozen=True)
class FamilySpec:
    prefix: str
    run_prefix: str
    load_level: str
    matrix_order: int


FAMILIES = (
    FamilySpec(
        "policy_matrix_dp8_24w_fail3_",
        "policy-matrix-dp8-24w-fail3",
        "24w-fail3",
        0,
    ),
)


def _policy_suffix(policy: str) -> str:
    return policy.replace("_", "-")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(round(_float(value, default)))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
            _float(row.get("created_time_s"), -1.0)
            for row in workflow_rows
            if _float(row.get("created_time_s"), -1.0) >= 0.0
        ]
        finished = [
            _float(row.get("finish_time_s"), -1.0)
            for row in workflow_rows
            if _float(row.get("finish_time_s"), -1.0) >= 0.0
        ]
        if created and finished:
            values.append(max(finished) - min(created))
            continue
        latencies = [
            _float(row.get("client_wall_e2e_latency_s") or row.get("e2e_latency_s"))
            for row in workflow_rows
        ]
        if latencies:
            values.append(max(latencies))
    return [value * 1000.0 for value in values if value > 0.0]


def load_suite_steps() -> list[dict[str, object]]:
    from benchmark.agentic_workflows.config import SCENARIOS

    steps: list[dict[str, object]] = []
    for family in FAMILIES:
        for policy in POLICY_ORDER:
            scenario = f"{family.prefix}{policy}"
            if scenario not in SCENARIOS:
                continue
            steps.append(
                {
                    "scenario": scenario,
                    "run_id": f"{family.run_prefix}-{_policy_suffix(policy)}-r1",
                    "policy": policy,
                    "load_level": family.load_level,
                    "matrix_order": family.matrix_order,
                }
            )
    return steps


def latest_run_dirs(results_root: Path, run_id: str) -> list[Path]:
    run_root = Path(results_root) / run_id / "runs"
    if not run_root.is_dir():
        return []
    return sorted(
        path
        for path in run_root.glob("trial_*/*")
        if path.is_dir() and (path / "request_details.csv").is_file()
    )


def failure_regime_for_scenario(scenario: str) -> str:
    from benchmark.agentic_workflows.config import SCENARIOS

    scenario_obj = SCENARIOS[scenario]
    injection = scenario_obj.failover_injection or "none"
    after = scenario_obj.failure_after_output_tokens
    count = scenario_obj.failure_injection_count
    if after is None:
        return injection
    if count is not None and count > 0:
        if (
            scenario_obj.failure_target_stage_id is not None
            and scenario_obj.failure_target_stage_local_agent_index is not None
        ):
            return (
                f"{injection}@{after}x{count}"
                f"@s{scenario_obj.failure_target_stage_id}"
                f"a{scenario_obj.failure_target_stage_local_agent_index}"
            )
        return f"{injection}@{after}x{count}"
    return f"{injection}@{after}"


def measured_row(
    *,
    step: dict[str, object],
    run_dir: Path,
) -> dict[str, object]:
    from benchmark.agentic_workflows.config import SCENARIOS

    scenario = str(step["scenario"])
    scenario_obj = SCENARIOS[scenario]
    policy = str(step["policy"])
    rows = _read_csv(run_dir / "request_details.csv")
    metrics = _read_json(run_dir / "metrics.json")
    config = _read_json(run_dir / "simulator_config.json")
    scheduler = config.get("scheduler") if isinstance(config.get("scheduler"), dict) else {}
    workload = config.get("workload") if isinstance(config.get("workload"), dict) else {}
    trial = run_dir.parent.name
    replay_rows = [row for row in rows if _bool(row.get("aegis_failover_replay"))]
    host_rows = [
        row for row in replay_rows if _bool(row.get("aegis_host_backup_lookup_hit"))
    ]
    peer_rows = [
        row for row in replay_rows if _bool(row.get("aegis_peer_gpu_lookup_hit"))
    ]
    wcts = workflow_wcts_ms(rows)
    restore_sleep = [
        _float(row.get("aegis_replay_restore_sleep_s")) * 1000.0
        for row in replay_rows
        if _float(row.get("aegis_replay_restore_sleep_s")) > 0.0
    ]
    restore_overheads = [
        _float(row.get("aegis_replay_restore_overhead_s")) * 1000.0
        for row in replay_rows
        if _float(row.get("aegis_replay_restore_overhead_s")) > 0.0
    ]
    if not restore_overheads:
        restore_overheads = restore_sleep
    seed = _int(workload.get("seed"))
    return {
        "row_type": "measured_policy_matrix_trial",
        "scenario": scenario,
        "run_id": step["run_id"],
        "trial": trial,
        "run_dir": str(run_dir),
        "policy": policy,
        "label": POLICY_LABELS[policy],
        "load_level": step["load_level"],
        "matrix_order": int(step["matrix_order"]),
        "failure_regime": failure_regime_for_scenario(scenario),
        "failure_rate_per_window": failure_regime_for_scenario(scenario),
        "failure_injection_count": scenario_obj.failure_injection_count,
        "failure_target_stage_id": scenario_obj.failure_target_stage_id,
        "failure_target_stage_local_agent_index": (
            scenario_obj.failure_target_stage_local_agent_index
        ),
        "aegis_gpu_budget_fraction": scenario_obj.aegis_gpu_budget_fraction,
        "seed": seed,
        "dp_size": _int(scheduler.get("dp_size")),
        "workflow_count": len(wcts),
        "request_rows": len(rows),
        "completed": _int(metrics.get("completed"), len(rows)),
        "failed": _int(metrics.get("failed")),
        "max_concurrency": _int(workload.get("max_concurrency")),
        "max_workflow_concurrency": _int(workload.get("gsp_max_workflow_concurrency")),
        "mean_wct_ms": statistics.mean(wcts) if wcts else _float(metrics.get("mean_e2e_latency_ms")),
        "p90_wct_ms": percentile(wcts, 0.90) if wcts else _float(metrics.get("p90_e2e_latency_ms")),
        "max_wct_ms": max(wcts) if wcts else _float(metrics.get("max_e2e_latency_ms")),
        "wct_samples_ms": ";".join(f"{value:.6f}" for value in wcts),
        "replay_rows": len(replay_rows),
        "host_lookup_hits": len(host_rows),
        "peer_lookup_hits": len(peer_rows),
        "host_lookup_tokens": sum(_int(row.get("aegis_host_backup_lookup_tokens")) for row in replay_rows),
        "peer_lookup_tokens": sum(_int(row.get("aegis_peer_gpu_lookup_tokens")) for row in replay_rows),
        "recompute_tokens": sum(_int(row.get("aegis_replay_recompute_tokens")) for row in replay_rows),
        "mean_restore_sleep_ms": statistics.mean(restore_sleep) if restore_sleep else 0.0,
        "mean_replay_overhead_ms": statistics.mean(restore_overheads) if restore_overheads else 0.0,
        "replay_overhead_samples_ms": ";".join(
            f"{value:.6f}" for value in restore_overheads
        ),
        "source": "sglsim_agentic_online_policy_matrix",
    }


def build_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in load_suite_steps():
        run_dirs = latest_run_dirs(results_root, str(step["run_id"]))
        if not run_dirs:
            raise FileNotFoundError(
                f"missing measured run for {step['run_id']}; "
                "run policy_matrix_dp8_24w_fail3 scenarios first"
            )
        for run_dir in run_dirs:
            rows.append(measured_row(step=step, run_dir=run_dir))
    return sorted(
        rows,
        key=lambda row: POLICY_ORDER.index(str(row["policy"])),
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


def _samples(value: object) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return []
    samples: list[float] = []
    for part in text.split(";"):
        try:
            sample = float(part)
        except ValueError:
            continue
        if sample > 0.0:
            samples.append(sample)
    return samples


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    xs = sorted(values)
    denom = max(len(xs), 1)
    ys = [(idx + 1) / denom for idx in range(len(xs))]
    return xs, ys


def _policy_wct_samples_s(rows: list[dict[str, object]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {policy: [] for policy in POLICY_ORDER}
    for row in rows:
        policy = str(row.get("policy") or "")
        if policy not in grouped:
            continue
        grouped[policy].extend(
            sample / 1000.0 for sample in _samples(row.get("wct_samples_ms"))
        )
    return grouped


def render(rows: list[dict[str, object]], *, output: Path, draft_output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9.0,
            "figure.dpi": 170,
        }
    )
    grouped_wct = _policy_wct_samples_s(rows)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    for policy in POLICY_ORDER:
        xs, ys = _ecdf(grouped_wct.get(policy, []))
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
            linewidth=2.2,
        )
    ax.set_xlabel("workflow completion time (s)")
    ax.set_ylabel("empirical CDF")
    ax.set_title(
        "Backup policy WCT CDF (DP8, 24 workflows, 3 S1-A0 failures, alpha=0.30)",
        loc="left",
        weight="bold",
    )
    ax.grid(True, color="#E5E7EB", lw=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    draft_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(draft_output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = build_rows(DEFAULT_RESULTS_ROOT)
    write_csv(DEFAULT_CSV, rows)
    render(rows, output=DEFAULT_OUTPUT, draft_output=DEFAULT_DRAFT_OUTPUT)
    print(
        json.dumps(
            {
                "csv": str(DEFAULT_CSV),
                "output": str(DEFAULT_OUTPUT),
                "draft_output": str(DEFAULT_DRAFT_OUTPUT),
                "rows": len(rows),
                "policies": [str(row["policy"]) for row in rows],
                "workflow_counts": [int(row["workflow_count"]) for row in rows],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
