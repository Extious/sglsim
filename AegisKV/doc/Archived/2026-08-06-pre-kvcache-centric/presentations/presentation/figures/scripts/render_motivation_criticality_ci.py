#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
MEASURED_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "figures" / "data" / "measured"
BEAMER_FIG_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "Figures"
DRAFT_PNG_DIR = BEAMER_FIG_DIR

PROFILE_CSV = MEASURED_DIR / "fig_critical_agent_profile.csv"
OUT_CSV = MEASURED_DIR / "fig_motivation_criticality_ci.csv"
OUT_PNG = DRAFT_PNG_DIR / "fig_motivation_criticality_ci.png"
OUT_BEAMER = BEAMER_FIG_DIR / "fig_motivation_criticality_ci.png"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _profile_run_dirs() -> list[Path]:
    rows = _read_csv(PROFILE_CSV)
    run_dirs = sorted(
        {
            Path(item)
            for row in rows
            for item in str(row.get("repeat_run_ids") or "").split(";")
            if item
        }
    )
    return [
        path
        for path in run_dirs
        if path.is_dir() and (path / "request_details.csv").is_file()
    ]


def _class_name(row: dict[str, str]) -> str:
    stage_width = int(_float(row.get("stage_width")))
    local_idx = int(_float(row.get("stage_local_agent_index")))
    if stage_width == 1:
        return "Singleton Critical"
    if local_idx == 0 and _bool_text(row.get("critical_agent")):
        return "Long Critical fan-out"
    return "Non-Critical fan-out"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return 1.96 * math.sqrt(variance) / math.sqrt(len(values))


def _build_rows() -> list[dict[str, str]]:
    raw_by_workflow: dict[tuple[str, str], dict[str, list[float]]] = {}
    raw_rows = 0
    for run_dir in _profile_run_dirs():
        for row in _read_csv(run_dir / "request_details.csv"):
            if row.get("status") != "completed":
                continue
            if _bool_text(row.get("aegis_cross_rank_retry")):
                continue
            workflow_id = str(row.get("workflow_id") or "")
            if not workflow_id:
                continue
            class_name = _class_name(row)
            key = (workflow_id, class_name)
            raw_by_workflow.setdefault(key, {"output_tokens": [], "latency_ms": []})
            raw_by_workflow[key]["output_tokens"].append(_float(row.get("output_length")))
            raw_by_workflow[key]["latency_ms"].append(_float(row.get("total_latency_s")) * 1000.0)
            raw_rows += 1

    rows: list[dict[str, str]] = []
    per_class: dict[str, dict[str, list[float]]] = {}
    for (workflow_id, class_name), metrics in sorted(raw_by_workflow.items()):
        output_mean = _mean(metrics["output_tokens"])
        latency_mean = _mean(metrics["latency_ms"])
        per_class.setdefault(class_name, {"output_tokens": [], "latency_ms": []})
        per_class[class_name]["output_tokens"].append(output_mean)
        per_class[class_name]["latency_ms"].append(latency_mean)
        rows.append(
            {
                "row_type": "workflow_class_mean",
                "workflow_id": workflow_id,
                "agent_class": class_name,
                "workflow_mean_output_tokens": f"{output_mean:.6f}",
                "workflow_mean_latency_ms": f"{latency_mean:.6f}",
                "sample_rows": str(len(metrics["output_tokens"])),
                "source": "sglsim_agentic_online_motivation_profile",
                "evidence_tier": "sglsim-dp8-profile-workflow-ci",
                "note": "One workflow-level sample for confidence intervals.",
            }
        )

    summary_rows: list[dict[str, str]] = []
    for class_name in (
        "Singleton Critical",
        "Long Critical fan-out",
        "Non-Critical fan-out",
    ):
        metrics = per_class.get(class_name, {"output_tokens": [], "latency_ms": []})
        summary_rows.append(
            {
                "row_type": "class_summary",
                "workflow_id": "",
                "agent_class": class_name,
                "workflow_mean_output_tokens": f"{_mean(metrics['output_tokens']):.6f}",
                "output_tokens_ci95": f"{_ci95(metrics['output_tokens']):.6f}",
                "workflow_mean_latency_ms": f"{_mean(metrics['latency_ms']):.6f}",
                "latency_ms_ci95": f"{_ci95(metrics['latency_ms']):.6f}",
                "workflow_groups": str(len(metrics["output_tokens"])),
                "sample_rows": str(raw_rows),
                "source": "sglsim_agentic_online_motivation_profile",
                "evidence_tier": "sglsim-dp8-profile-workflow-ci",
                "note": "Class summary over workflow-level means from the DP8 motivation profile run.",
            }
        )

    by_class = {row["agent_class"]: row for row in summary_rows}
    long = by_class.get("Long Critical fan-out", {})
    non = by_class.get("Non-Critical fan-out", {})
    long_output = _float(long.get("workflow_mean_output_tokens"))
    non_output = _float(non.get("workflow_mean_output_tokens"))
    long_latency = _float(long.get("workflow_mean_latency_ms"))
    non_latency = _float(non.get("workflow_mean_latency_ms"))
    summary_rows.append(
        {
            "row_type": "criticality_ratio_summary",
            "workflow_id": "",
            "agent_class": "Long Critical fan-out / Non-Critical fan-out",
            "workflow_mean_output_tokens": f"{long_output / max(non_output, 1e-9):.6f}",
            "workflow_mean_latency_ms": f"{long_latency / max(non_latency, 1e-9):.6f}",
            "workflow_groups": str(
                min(
                    int(_float(long.get("workflow_groups"))),
                    int(_float(non.get("workflow_groups"))),
                )
            ),
            "sample_rows": str(raw_rows),
            "source": "sglsim_agentic_online_motivation_profile",
            "evidence_tier": "sglsim-dp8-profile-workflow-ci",
            "note": "Ratio above 1 means long Critical fan-out agents dominate Non-Critical fan-out agents.",
        }
    )
    return rows + summary_rows


def _write_csv(rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("row_type") == "class_summary"]


def render(rows: list[dict[str, str]]) -> None:
    summaries = _summary_rows(rows)
    labels = [row["agent_class"] for row in summaries]
    colors = ["#6B7280", "#D97706", "#94A3B8"]
    x = np.arange(len(labels), dtype=float)

    output_values = [_float(row.get("workflow_mean_output_tokens")) for row in summaries]
    output_ci = [_float(row.get("output_tokens_ci95")) for row in summaries]
    latency_values = [_float(row.get("workflow_mean_latency_ms")) for row in summaries]
    latency_ci = [_float(row.get("latency_ms_ci95")) for row in summaries]
    workflow_rows = [
        row for row in rows if row.get("row_type") == "workflow_class_mean"
    ]

    def grouped(metric: str) -> list[list[float]]:
        values: dict[str, list[float]] = {label: [] for label in labels}
        for row in workflow_rows:
            cls = row.get("agent_class") or ""
            if cls in values:
                values[cls].append(_float(row.get(metric)))
        return [values[label] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), gridspec_kw={"wspace": 0.28})
    for ax, values, errors, samples, ylabel, title in (
        (
            axes[0],
            output_values,
            output_ci,
            grouped("workflow_mean_output_tokens"),
            "Workflow-mean output tokens",
            "(a) Decode pressure by agent class",
        ),
        (
            axes[1],
            latency_values,
            latency_ci,
            grouped("workflow_mean_latency_ms"),
            "Workflow-mean request latency (ms)",
            "(b) Latency exposure by agent class",
        ),
    ):
        for idx, (class_samples, color) in enumerate(zip(samples, colors, strict=True)):
            if class_samples:
                jitter = np.linspace(-0.075, 0.075, len(class_samples))
                ax.scatter(
                    np.full(len(class_samples), x[idx]) + jitter,
                    class_samples,
                    s=28,
                    color=color,
                    alpha=0.34,
                    edgecolors="none",
                    zorder=2,
                )
            ax.errorbar(
                x[idx],
                values[idx],
                yerr=errors[idx],
                fmt="D",
                ms=6.0,
                color="#111827",
                markerfacecolor=color,
                markeredgecolor="#111827",
                ecolor="#111827",
                elinewidth=1.1,
                capsize=5,
                zorder=4,
            )
            ax.text(
                x[idx],
                values[idx] + max(values) * 0.040,
                f"{values[idx]:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.7,
            )
        ax.plot(x, values, color="#111827", linewidth=0.8, alpha=0.55, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Singleton\nCritical", "Long Critical\nfan-out", "Non-Critical\nfan-out"],
            fontsize=8.8,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold", fontsize=11)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(value + error for value, error in zip(values, errors, strict=True)) * 1.22)
        ax.set_xlim(-0.40, len(labels) - 0.60)

    ratio = next(
        row
        for row in rows
        if row.get("row_type") == "criticality_ratio_summary"
    )
    fig.suptitle(
        "Motivation repeat profile: Critical fan-out agents dominate the tail",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=12.4,
        weight="bold",
    )
    fig.text(
        0.98,
        -0.02,
        (
            "DP8 motivation profile, 8 workflow groups; 95% CI over workflow-level "
            "means. Long Critical / Non-Critical ratio: "
            f"output {float(ratio['workflow_mean_output_tokens']):.1f}x, "
            f"latency {float(ratio['workflow_mean_latency_ms']):.1f}x."
        ),
        ha="right",
        va="top",
        fontsize=8.5,
        color="#475569",
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_BEAMER.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    fig.savefig(OUT_BEAMER, bbox_inches="tight", dpi=180)
    plt.close(fig)


def main() -> int:
    rows = _build_rows()
    _write_csv(rows)
    render(rows)
    print(OUT_CSV)
    print(OUT_PNG)
    print(OUT_BEAMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
