#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PRESENTATION_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    PRESENTATION_ROOT
    / "experiments"
    / "sglsim_large"
    / "results"
    / "agentic_workflows"
)
SUMMARY_CSV = (
    PRESENTATION_ROOT
    / "figures"
    / "data"
    / "measured"
    / "fig_backup_policy_comparison.csv"
)
OUTPUT = PRESENTATION_ROOT / "Figures" / "fig_backup_policy_comparison_internal.png"
DETAIL_CSV = (
    PRESENTATION_ROOT
    / "figures"
    / "data"
    / "measured"
    / "fig_backup_policy_comparison_internal.csv"
)
PAIRED_CSV = (
    PRESENTATION_ROOT
    / "figures"
    / "data"
    / "measured"
    / "fig_backup_policy_comparison_paired_wct.csv"
)

POLICIES = ("no_backup", "host_only", "gpu_replica", "aegiskv")
LABELS = {
    "no_backup": "No backup",
    "host_only": "HostBackup",
    "gpu_replica": "GPUReplica",
    "aegiskv": "AegisKV\n(fixed alpha)",
}
COLORS = {
    "no_backup": "#6B7280",
    "host_only": "#2563EB",
    "gpu_replica": "#D97706",
    "aegiskv": "#059669",
}


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    return int(round(_float(value, default)))


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _request_details_path(summary: dict[str, str]) -> Path:
    trial_root = (
        RESULTS_ROOT
        / summary["run_id"]
        / "runs"
        / summary["trial"]
    )
    candidates = sorted(trial_root.glob("*/request_details.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one request_details.csv below {trial_root}, found {candidates}"
        )
    return candidates[0]


def _workflow_wcts(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["workflow_id"])].append(row)
    result: dict[str, float] = {}
    for workflow_id, workflow_rows in grouped.items():
        created = [_float(row["created_time_s"]) for row in workflow_rows]
        finished = [_float(row["finish_time_s"]) for row in workflow_rows]
        result[workflow_id] = max(finished) - min(created)
    return result


def _policy_detail(summary: dict[str, str]) -> dict[str, Any]:
    policy = summary["policy"]
    rows = _read_csv(_request_details_path(summary))
    replay_rows = [row for row in rows if _bool(row.get("aegis_failover_replay"))]
    workflow_wcts = _workflow_wcts(rows)
    wcts = list(workflow_wcts.values())

    host_restored = 0
    gpu_restored = 0
    for row in replay_rows:
        restored = _int(row.get("aegis_replay_restored_tokens"))
        action = str(row.get("aegis_replay_restore_action") or "").lower()
        if "host" in action or "l2" in action:
            host_restored += restored
        elif "gpu" in action or "replica" in action or "l1" in action:
            gpu_restored += restored

    normal_rows = [row for row in rows if row not in replay_rows]
    replay_overheads = [
        1000.0 * _float(row.get("aegis_replay_restore_overhead_s"))
        for row in replay_rows
    ]
    return {
        "policy": policy,
        "label": LABELS[policy].replace("\n", " "),
        "workflow_count": len(wcts),
        "request_rows": len(rows),
        "mean_wct_s": statistics.mean(wcts),
        "p50_wct_s": _percentile(wcts, 0.50),
        "p90_wct_s": _percentile(wcts, 0.90),
        "p99_wct_s": _percentile(wcts, 0.99),
        "max_wct_s": max(wcts),
        "paired_win_count_vs_no_backup": 0,
        "paired_mean_delta_s_vs_no_backup": 0.0,
        "host_restored_tokens": host_restored,
        "gpu_restored_tokens": gpu_restored,
        "recompute_tokens": sum(
            _int(row.get("aegis_replay_recompute_tokens")) for row in replay_rows
        ),
        "host_lookup_tokens": sum(
            _int(row.get("aegis_host_backup_lookup_tokens")) for row in replay_rows
        ),
        "peer_lookup_tokens": sum(
            _int(row.get("aegis_peer_gpu_lookup_tokens")) for row in replay_rows
        ),
        "replay_count": len(replay_rows),
        "mean_replay_overhead_ms": statistics.mean(replay_overheads),
        "replay_overheads_ms": replay_overheads,
        "mean_normal_waiting_s": statistics.mean(
            _float(row.get("waiting_time_s")) for row in normal_rows
        ),
        "mean_normal_e2e_s": statistics.mean(
            _float(row.get("client_wall_e2e_latency_s")) for row in normal_rows
        ),
        "workflow_wcts": workflow_wcts,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_details() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = {row["policy"]: row for row in _read_csv(SUMMARY_CSV)}
    if set(summaries) != set(POLICIES):
        raise ValueError(f"unexpected policy set in {SUMMARY_CSV}: {sorted(summaries)}")
    details = [_policy_detail(summaries[policy]) for policy in POLICIES]
    baseline = details[0]["workflow_wcts"]
    paired_rows: list[dict[str, Any]] = []
    for detail in details[1:]:
        policy = str(detail["policy"])
        current = detail["workflow_wcts"]
        if set(current) != set(baseline):
            raise ValueError(f"workflow IDs differ between no_backup and {policy}")
        deltas = []
        for workflow_id in sorted(baseline):
            delta = float(current[workflow_id]) - float(baseline[workflow_id])
            deltas.append(delta)
            paired_rows.append(
                {
                    "policy": policy,
                    "label": LABELS[policy].replace("\n", " "),
                    "workflow_id": workflow_id,
                    "policy_wct_s": current[workflow_id],
                    "no_backup_wct_s": baseline[workflow_id],
                    "delta_s": delta,
                }
            )
        detail["paired_win_count_vs_no_backup"] = sum(delta < 0 for delta in deltas)
        detail["paired_mean_delta_s_vs_no_backup"] = statistics.mean(deltas)
    return details, paired_rows


def render(details: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> None:
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
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
    x = np.arange(len(POLICIES))

    ax = axes[0, 0]
    metrics = (
        ("mean_wct_s", "Mean", "#0072B2"),
        ("p50_wct_s", "P50", "#56B4E9"),
        ("p90_wct_s", "P90", "#E69F00"),
        ("p99_wct_s", "P99", "#D55E00"),
    )
    width = 0.19
    for index, (field, label, color) in enumerate(metrics):
        ax.bar(
            x + (index - 1.5) * width,
            [float(detail[field]) for detail in details],
            width=width,
            color=color,
            label=label,
        )
    ax.set_xticks(x, [LABELS[policy] for policy in POLICIES])
    ax.set_ylabel("Workflow completion time (s)")
    ax.set_title("(a) WCT distribution")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    host = [int(detail["host_restored_tokens"]) for detail in details]
    gpu = [int(detail["gpu_restored_tokens"]) for detail in details]
    recompute = [int(detail["recompute_tokens"]) for detail in details]
    ax.bar(x, host, color="#2563EB", label="Actually restored from Host")
    ax.bar(x, gpu, bottom=host, color="#D97706", label="Actually restored from GPU")
    ax.bar(
        x,
        recompute,
        bottom=[host[i] + gpu[i] for i in range(len(x))],
        color="#CC79A7",
        label="Recomputed",
    )
    for index, detail in enumerate(details):
        total = host[index] + gpu[index] + recompute[index]
        reused = 100.0 * (host[index] + gpu[index]) / max(total, 1)
        ax.text(
            index,
            total * 0.94,
            f"{reused:.1f}% reused",
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            weight="bold",
        )
    ax.set_xticks(x, [LABELS[policy] for policy in POLICIES])
    ax.set_ylabel("Retry-prefix tokens")
    ax.set_title("(b) Actual recovery work across 3 failures")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
    )

    ax = axes[1, 0]
    means = [float(detail["mean_replay_overhead_ms"]) for detail in details]
    ax.bar(x, means, color=[COLORS[policy] for policy in POLICIES], alpha=0.82)
    for index, detail in enumerate(details):
        samples = list(detail["replay_overheads_ms"])
        offsets = np.linspace(-0.10, 0.10, len(samples))
        ax.scatter(
            index + offsets,
            samples,
            color="#111827",
            s=20,
            zorder=3,
            label="Failure replay" if index == 0 else None,
        )
        ax.text(index, means[index] + 10, f"{means[index]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, [LABELS[policy] for policy in POLICIES])
    ax.set_ylabel("Recovery overhead (ms)")
    ax.set_title("(c) Per-failure recovery overhead")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    paired_by_policy = {
        policy: [
            float(row["delta_s"])
            for row in paired_rows
            if row["policy"] == policy
        ]
        for policy in POLICIES[1:]
    }
    positions = np.arange(1, len(POLICIES))
    box = ax.boxplot(
        [paired_by_policy[policy] for policy in POLICIES[1:]],
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.4},
    )
    for patch, policy in zip(box["boxes"], POLICIES[1:]):
        patch.set_facecolor(COLORS[policy])
        patch.set_alpha(0.35)
    for index, policy in enumerate(POLICIES[1:], start=1):
        values = paired_by_policy[policy]
        jitter = np.linspace(-0.16, 0.16, len(values))
        ax.scatter(
            index + jitter,
            values,
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.35,
            s=22,
            alpha=0.85,
        )
        mean_delta = statistics.mean(values)
        wins = sum(value < 0 for value in values)
        ax.text(
            index,
            max(values) + 2.0,
            f"mean {mean_delta:+.1f}s\n{wins}/24 wins",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    all_deltas = [float(row["delta_s"]) for row in paired_rows]
    ax.set_ylim(min(all_deltas) - 4.0, max(all_deltas) + 14.0)
    ax.set_xticks(positions, [LABELS[policy] for policy in POLICIES[1:]])
    ax.set_ylabel("Policy - No Backup WCT (s)")
    ax.set_title("(d) Paired workflow-level change")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)

    fig.suptitle(
        "Backup policy internal comparison (DP8, 24 workflows, "
        "3 soft failures, fixed alpha=0.30)",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Legacy single-seed/single-trial diagnostic; generated shared-prefix workload. "
        "This is not the current dynamic alpha/beta controller.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), h_pad=3.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    details, paired_rows = load_details()
    detail_fields = [
        field
        for field in details[0]
        if field not in {"replay_overheads_ms", "workflow_wcts"}
    ]
    _write_csv(
        DETAIL_CSV,
        detail_fields,
        [{field: detail[field] for field in detail_fields} for detail in details],
    )
    _write_csv(
        PAIRED_CSV,
        ["policy", "label", "workflow_id", "policy_wct_s", "no_backup_wct_s", "delta_s"],
        paired_rows,
    )
    render(details, paired_rows)
    print(f"wrote {OUTPUT}")
    print(f"wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {DETAIL_CSV}")
    print(f"wrote {PAIRED_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
