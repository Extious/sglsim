from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .plot_wct_cdf import (
    POLICY_STYLE,
    load_policy_results,
    recompute_penalty_label,
)


METRICS = ("mean", "p50", "p90", "p99")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _policy_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for policy, payload in results.items():
        retry = payload["retry_summary"]
        wct = payload["wct_summary_s"]
        rows.append(
            {
                "policy": policy,
                "label": payload["policy_label"],
                **{f"wct_{metric}_s": wct[metric] for metric in METRICS},
                "retry_count": retry["count"],
                "restored_tokens": retry["restored_tokens"],
                "recompute_tokens": retry["recompute_tokens"],
                "capacity_limited_tokens": retry.get(
                    "capacity_limited_tokens", 0
                ),
                "restored_fraction": retry["restored_fraction"],
                "retry_mean_latency_s": retry["mean_latency_s"],
                "retry_p90_latency_s": retry["p90_latency_s"],
            }
        )
    return rows


def _paired_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    aegis = {
        str(row["workflow_id"]): row for row in results["aegiskv"]["workflows"]
    }
    baseline = {
        str(row["workflow_id"]): row for row in results["no_backup"]["workflows"]
    }
    return [
        {
            "workflow_id": workflow_id,
            "aegiskv_wct_s": float(aegis[workflow_id]["wct_s"]),
            "no_backup_wct_s": float(baseline[workflow_id]["wct_s"]),
            "delta_s": float(aegis[workflow_id]["wct_s"])
            - float(baseline[workflow_id]["wct_s"]),
            "aegiskv_retry_count": int(aegis[workflow_id]["retry_count"]),
            "no_backup_retry_count": int(baseline[workflow_id]["retry_count"]),
        }
        for workflow_id in sorted(aegis)
    ]


def _rank_alpha_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    updates = payload.get("rank_alpha_updates")
    if not isinstance(updates, list):
        updates = payload.get("alpha_beta_updates", [])
    for update in updates:
        rows.append(
            {
                "sequence": int(update["sequence"]),
                "dp_rank": int(update["dp_rank"]),
                "window_id": int(update["window_id"]),
                "trigger": str(update.get("trigger", "unknown")),
                "old_alpha": float(update["old_alpha"]),
                "new_alpha": float(update["new_alpha"]),
            }
        )
    return rows


def _node_beta_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for update in payload.get("node_beta_updates", []):
        rows.append(
            {
                "sequence": int(update["sequence"]),
                "node_id": int(update["node_id"]),
                "window_id": int(update["window_id"]),
                "trigger": str(update.get("trigger", "unknown")),
                "D": int(update.get("D", 0)),
                "U": int(update.get("U", 0)),
                "O": int(update.get("O", 0)),
                "old_beta": float(update["old_beta"]),
                "new_beta": float(update["new_beta"]),
            }
        )
    return rows


def _failure_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, payload in results.items():
        for index, event in enumerate(payload.get("failure_events", []), start=1):
            clear = event.get("l1_clear", {})
            rows.append(
                {
                    "policy": policy,
                    "policy_label": payload["policy_label"],
                    "failure_index": index,
                    "event_id": event.get("event_id", f"failure-{index:02d}"),
                    "dp_rank": int(event.get("dp_rank", -1)),
                    "minimum_duration_s": float(event.get("duration_s", 0.0)),
                    "actual_outage_s": float(event.get("actual_outage_s", 0.0)),
                    "recovery_control_latency_s": float(
                        event.get("recovery_control_latency_s", 0.0)
                    ),
                    "cleared_device_tokens": int(
                        clear.get("cleared_device_tokens", 0)
                    ),
                    "cleared_l1_backup_tokens": int(
                        clear.get("cleared_l1_backup_tokens", 0)
                    ),
                    "preserved_l2_backup_tokens": int(
                        clear.get("preserved_l2_backup_tokens", 0)
                    ),
                    "preserved_host_tokens": int(
                        clear.get("preserved_host_tokens", 0)
                    ),
                }
            )
    return rows


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = 10_000,
    seed: int = 20260722,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def _reduction_percent(candidate: float, baseline: float) -> float:
    """Return positive values when ``candidate`` is lower than ``baseline``."""

    if baseline == 0:
        return 0.0
    return round(100.0 * (baseline - candidate) / baseline, 6)


def _render_failure_analysis(
    *,
    results: dict[str, dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    penalty_label: str,
    output: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    policies = list(POLICY_STYLE)
    x = list(range(len(policies)))

    ax = axes[0, 0]
    for policy, (label, color, linestyle) in POLICY_STYLE.items():
        rows = [row for row in failure_rows if row["policy"] == policy]
        rows.sort(key=lambda row: int(row["failure_index"]))
        ax.plot(
            [int(row["failure_index"]) for row in rows],
            [float(row["actual_outage_s"]) for row in rows],
            marker="o",
            markersize=4,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            label=label,
        )
    minimum = max(
        (float(row["minimum_duration_s"]) for row in failure_rows),
        default=0.0,
    )
    ax.axhline(
        minimum,
        color="#111827",
        linewidth=1.0,
        linestyle="--",
        label=f"minimum hold ({minimum:g}s)",
    )
    ax.set_xlabel("Failure event")
    ax.set_ylabel("Actual rank outage (s)")
    ax.set_title("(a) Safe recovery extends the minimum hold")
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    width = 0.24
    clear_metrics = (
        ("cleared_device_tokens", "GPU L1 cleared", "#D55E00"),
        ("cleared_l1_backup_tokens", "Backup L1 invalidated", "#E69F00"),
        ("preserved_l2_backup_tokens", "Backup L2 preserved", "#009E73"),
    )
    for metric_index, (metric, label, color) in enumerate(clear_metrics):
        values = [
            sum(
                int(row[metric])
                for row in failure_rows
                if row["policy"] == policy
            )
            for policy in policies
        ]
        ax.bar(
            [value + (metric_index - 1) * width for value in x],
            values,
            width=width,
            label=label,
            color=color,
        )
    ax.set_xticks(x, [POLICY_STYLE[p][0] for p in policies])
    ax.set_ylabel("Tokens across injected failures")
    ax.set_title("(b) Failed-replica loss and surviving backup state")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    latency_width = 0.34
    means = [float(results[p]["retry_summary"]["mean_latency_s"]) for p in policies]
    p90s = [float(results[p]["retry_summary"]["p90_latency_s"]) for p in policies]
    ax.bar(
        [value - latency_width / 2 for value in x],
        means,
        width=latency_width,
        color="#56B4E9",
        label="Mean",
    )
    ax.bar(
        [value + latency_width / 2 for value in x],
        p90s,
        width=latency_width,
        color="#0072B2",
        label="P90",
    )
    ax.set_xticks(x, [POLICY_STYLE[p][0] for p in policies])
    ax.set_ylabel("Retry request latency (s)")
    ax.set_title("(c) End-to-end retry latency")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    action_styles = (
        ("l1_backup", "GPU", "#0072B2"),
        ("l2_backup", "Host", "#009E73"),
        ("l3_backup", "L3", "#E69F00"),
        ("l1_l2_backup", "GPU + Host", "#56B4E9"),
        ("l1_l3_backup", "GPU + L3", "#CC79A7"),
        ("l2_l3_backup", "Host + L3", "#F0E442"),
        ("l1_l2_l3_backup", "GPU + Host + L3", "#8B6BB1"),
        ("recompute", "Recompute", "#6B7280"),
    )
    bottoms = [0] * len(policies)
    for action, label, color in action_styles:
        values = [
            int(results[p]["retry_summary"]["recovery_actions"].get(action, 0))
            for p in policies
        ]
        ax.bar(x, values, bottom=bottoms, color=color, label=label)
        bottoms = [bottoms[index] + values[index] for index in x]
    ax.set_xticks(x, [POLICY_STYLE[p][0] for p in policies])
    ax.set_ylabel("Retry count")
    ax.set_title("(d) Recovery action selected for affected requests")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)

    replay = results["aegiskv"]["replay"]
    evidence = results["aegiskv"].get("evidence_tier")
    profile = "formal" if evidence == "sglsim-online-measured" else "quick"
    penalty_suffix = f"; {penalty_label}" if penalty_label else ""
    deployment = results["aegiskv"]
    fig.suptitle(
        f"AegisKV {profile} recovery mechanics: Real CC fan-out/fan-in, "
        f"{deployment.get('model', 'unknown')}, "
        f"DP={deployment.get('dp_size', '?')}/"
        f"PP={deployment.get('pp_size', '?')}, "
        f"context <= {int(replay['context_length'])}"
        f"{penalty_suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def render(results_dir: Path, output: Path) -> Path:
    results = load_policy_results(results_dir)
    penalty_label = recompute_penalty_label(results_dir)
    policy_rows = _policy_rows(results)
    paired_rows = _paired_rows(results)
    rank_alpha_rows = _rank_alpha_rows(results["aegiskv"])
    node_beta_rows = _node_beta_rows(results["aegiskv"])
    failure_rows = _failure_rows(results)
    paired_deltas = [float(row["delta_s"]) for row in paired_rows]
    paired_ci = _bootstrap_mean_ci(paired_deltas)

    _write_csv(
        results_dir / "policy_summary.csv",
        tuple(policy_rows[0]),
        policy_rows,
    )
    _write_csv(
        results_dir / "paired_wct_delta.csv",
        tuple(paired_rows[0]),
        paired_rows,
    )
    if rank_alpha_rows:
        _write_csv(
            results_dir / "rank_alpha_updates.csv",
            tuple(rank_alpha_rows[0]),
            rank_alpha_rows,
        )
    if node_beta_rows:
        _write_csv(
            results_dir / "node_beta_updates.csv",
            tuple(node_beta_rows[0]),
            node_beta_rows,
        )
    if failure_rows:
        _write_csv(
            results_dir / "failure_event_summary.csv",
            tuple(failure_rows[0]),
            failure_rows,
        )
    aegis_wct = results["aegiskv"]["wct_summary_s"]
    baseline_wct = results["no_backup"]["wct_summary_s"]
    aegis_retry = results["aegiskv"]["retry_summary"]
    baseline_retry = results["no_backup"]["retry_summary"]
    aegis_mean = float(aegis_wct["mean"])
    baseline_mean = float(baseline_wct["mean"])
    statistical_summary = {
        "workflow_count": len(paired_rows),
        "aegiskv_mean_wct_s": aegis_mean,
        "no_backup_mean_wct_s": baseline_mean,
        "mean_wct_reduction_percent": _reduction_percent(
            aegis_mean, baseline_mean
        ),
        "wct_comparison": {
            metric: {
                "aegiskv_s": float(aegis_wct[metric]),
                "no_backup_s": float(baseline_wct[metric]),
                "reduction_percent": _reduction_percent(
                    float(aegis_wct[metric]), float(baseline_wct[metric])
                ),
            }
            for metric in (*METRICS, "max")
        },
        "retry_comparison": {
            "aegiskv_count": int(aegis_retry["count"]),
            "no_backup_count": int(baseline_retry["count"]),
            "mean_latency_s": {
                "aegiskv": float(aegis_retry["mean_latency_s"]),
                "no_backup": float(baseline_retry["mean_latency_s"]),
                "reduction_percent": _reduction_percent(
                    float(aegis_retry["mean_latency_s"]),
                    float(baseline_retry["mean_latency_s"]),
                ),
            },
            "p90_latency_s": {
                "aegiskv": float(aegis_retry["p90_latency_s"]),
                "no_backup": float(baseline_retry["p90_latency_s"]),
                "reduction_percent": _reduction_percent(
                    float(aegis_retry["p90_latency_s"]),
                    float(baseline_retry["p90_latency_s"]),
                ),
            },
        },
        "token_recovery": {
            "aegiskv_restored_tokens": int(aegis_retry["restored_tokens"]),
            "aegiskv_recompute_tokens": int(aegis_retry["recompute_tokens"]),
            "no_backup_restored_tokens": int(baseline_retry["restored_tokens"]),
            "no_backup_recompute_tokens": int(baseline_retry["recompute_tokens"]),
            "aegiskv_restored_fraction": float(aegis_retry["restored_fraction"]),
        },
        "paired_mean_delta_s": round(mean(paired_deltas), 6),
        "paired_bootstrap_95ci_s": [round(value, 6) for value in paired_ci],
        "aegiskv_workflow_wins": sum(value < 0 for value in paired_deltas),
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20260722,
    }
    (results_dir / "statistical_summary.json").write_text(
        json.dumps(statistical_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))

    ax = axes[0, 0]
    x = list(range(len(POLICY_STYLE)))
    width = 0.19
    metric_colors = ("#0072B2", "#56B4E9", "#E69F00", "#D55E00")
    for metric_index, (metric, color) in enumerate(zip(METRICS, metric_colors)):
        values = [float(results[p]["wct_summary_s"][metric]) for p in POLICY_STYLE]
        ax.bar(
            [value + (metric_index - 1.5) * width for value in x],
            values,
            width=width,
            label=metric.upper(),
            color=color,
        )
    ax.set_xticks(x, [POLICY_STYLE[p][0] for p in POLICY_STYLE])
    ax.set_ylabel("Workflow completion time (s)")
    ax.set_title("(a) WCT distribution statistics")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    restored = [results[p]["retry_summary"]["restored_tokens"] for p in POLICY_STYLE]
    recomputed = [results[p]["retry_summary"]["recompute_tokens"] for p in POLICY_STYLE]
    capacity_limited = [
        results[p]["retry_summary"].get("capacity_limited_tokens", 0)
        for p in POLICY_STYLE
    ]
    direct_recompute = [
        max(recomputed[index] - capacity_limited[index], 0)
        for index in range(len(recomputed))
    ]
    ax.bar(x, restored, color="#009E73", label="Restored")
    ax.bar(
        x,
        capacity_limited,
        bottom=restored,
        color="#E69F00",
        label="Capacity-limited",
    )
    ax.bar(
        x,
        direct_recompute,
        bottom=[restored[i] + capacity_limited[i] for i in x],
        color="#CC79A7",
        label="Direct recompute",
    )
    for idx, policy in enumerate(POLICY_STYLE):
        fraction = 100.0 * float(results[policy]["retry_summary"]["restored_fraction"])
        ax.text(
            idx,
            restored[idx] + recomputed[idx],
            f"{fraction:.1f}% reused",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x, [POLICY_STYLE[p][0] for p in POLICY_STYLE])
    ax.set_ylabel("Retry-prefix tokens")
    ax.set_title("(b) Recovery work under fixed failures")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rank_alpha_rows:
        by_rank[int(row["dp_rank"])].append(row)
    for rows in by_rank.values():
        rows.sort(key=lambda row: int(row["sequence"]))
        ax.plot(
            [row["sequence"] for row in rows],
            [row["new_alpha"] for row in rows],
            color="#0072B2",
            alpha=0.32,
            linewidth=1.0,
        )
    by_node: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in node_beta_rows:
        by_node[int(row["node_id"])].append(row)
    for rows in by_node.values():
        rows.sort(key=lambda row: int(row["sequence"]))
        ax.plot(
            [row["sequence"] for row in rows],
            [row["new_beta"] for row in rows],
            color="#E69F00",
            alpha=0.32,
            linewidth=1.0,
        )
    ax.set_xlabel("Router event sequence")
    ax.set_ylabel("Published partition ratio")
    ax.set_ylim(bottom=0.0)
    ax.set_title(
        "(c) Algorithm 3: DP-replica alpha and node-shared beta"
    )
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.legend(
        handles=(
            Line2D([0], [0], color="#0072B2", label=r"$\alpha_d$ (GPU)"),
            Line2D([0], [0], color="#E69F00", label=r"$\beta_n$ (Host)"),
        ),
        frameon=False,
    )

    ax = axes[1, 1]
    ordered = sorted(paired_rows, key=lambda row: float(row["delta_s"]))
    deltas = [float(row["delta_s"]) for row in ordered]
    colors = ["#009E73" if value < 0 else "#D55E00" for value in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors, width=0.8)
    paired_mean = mean(deltas)
    ax.axhline(0.0, color="#111827", linewidth=0.9)
    ax.axhline(
        paired_mean,
        color="#0072B2",
        linestyle="--",
        linewidth=1.4,
        label=(
            f"mean = {paired_mean:+.2f}s, "
            f"95% CI [{paired_ci[0]:+.2f}, {paired_ci[1]:+.2f}]"
        ),
    )
    ax.set_xlabel("Workflow (sorted by paired delta)")
    ax.set_ylabel("AegisKV - No Backup WCT (s)")
    ax.set_title("(d) Paired workflow-level WCT change")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)

    replay = results["aegiskv"]["replay"]
    evidence = results["aegiskv"].get("evidence_tier")
    profile = "formal" if evidence == "sglsim-online-measured" else "quick"
    penalty_suffix = f"; {penalty_label}" if penalty_label else ""
    deployment = results["aegiskv"]
    failure_durations = {
        float(event.get("duration_s", 0.0))
        for event in deployment.get("failure_events", [])
    }
    duration_label = (
        f"{next(iter(failure_durations)):g}-s"
        if len(failure_durations) == 1
        else "manifest-duration"
    )
    fig.suptitle(
        f"AegisKV {profile} experiment: Real CC fan-out/fan-in, "
        f"{deployment.get('model', 'unknown')}, "
        f"DP={deployment.get('dp_size', '?')}/"
        f"PP={deployment.get('pp_size', '?')}, "
        f"context <= {int(replay['context_length'])}, "
        f"{deployment['failure_count_requested']} fixed {duration_label} failures"
        f"{penalty_suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    _render_failure_analysis(
        results=results,
        failure_rows=failure_rows,
        penalty_label=penalty_label,
        output=output.with_name("failure_recovery_analysis.png"),
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot AegisKV internal diagnostics.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.results_dir / "internal_analysis.png")
    render(args.results_dir, output)
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_name('failure_recovery_analysis.png')}")
    print(f"wrote {output.with_name('failure_recovery_analysis.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
