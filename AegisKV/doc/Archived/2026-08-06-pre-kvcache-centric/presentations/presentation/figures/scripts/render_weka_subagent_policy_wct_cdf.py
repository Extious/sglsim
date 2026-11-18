#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SUITE_DIR = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "demo"
    / "results"
    / "wct_cdf"
    / "20260722-013820"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "presentation"
    / "Figures"
    / "fig_weka_subagent_policy_wct_cdf.png"
)
DEFAULT_DRAFT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "presentation"
    / "figures"
    / "png"
    / "fig_weka_subagent_policy_wct_cdf.png"
)
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "presentation"
    / "figures"
    / "data"
    / "measured"
    / "fig_weka_subagent_policy_wct_cdf.csv"
)

POLICY_SPECS = (
    ("aegiskv", "AegisKV", "#0072B2"),
    ("no_backup", "No Backup", "#6B7280"),
)
RETRY_TOKEN_SPECS = (
    ("l1_reuse_tokens", "L1 reuse", "#059669"),
    ("l2_reuse_tokens", "L2 reuse", "#2563EB"),
    ("recompute_tokens", "Recompute", "#6B7280"),
)
COLOR_GRID = "#E5E7EB"
COLOR_MUTED = "#64748B"


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
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


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(values)
    return ordered, [(idx + 1) / len(ordered) for idx in range(len(ordered))]


def _retry_token_breakdown(payload: dict) -> dict[str, int]:
    retry_events = [
        event
        for workflow in payload.get("workflows", [])
        for event in workflow.get("retry_events", [])
    ]
    totals = {
        "retry_count": len(retry_events),
        "l1_reuse_tokens": 0,
        "l2_reuse_tokens": 0,
        "mixed_l1_l2_reuse_tokens": 0,
        "recompute_tokens": 0,
    }
    for event in retry_events:
        action = str(event.get("recovery_action") or "recompute")
        restored = max(int(event.get("restored_tokens", 0) or 0), 0)
        recompute = max(int(event.get("recompute_tokens", 0) or 0), 0)
        if action == "l1_backup":
            totals["l1_reuse_tokens"] += restored
        elif action == "l2_backup":
            totals["l2_reuse_tokens"] += restored
        elif action == "l1_l2_backup":
            # The persisted retry event contains only the combined restored
            # count.  Do not invent an L1/L2 split when exact attribution is
            # unavailable.
            totals["mixed_l1_l2_reuse_tokens"] += restored
        elif restored:
            raise ValueError(
                f"restored tokens use unsupported recovery action {action!r}"
            )
        totals["recompute_tokens"] += recompute

    summary = payload.get("retry_summary", {})
    expected_count = int(summary.get("count", len(retry_events)) or 0)
    expected_restored = int(summary.get("restored_tokens", 0) or 0)
    expected_recompute = int(summary.get("recompute_tokens", 0) or 0)
    measured_restored = (
        totals["l1_reuse_tokens"]
        + totals["l2_reuse_tokens"]
        + totals["mixed_l1_l2_reuse_tokens"]
    )
    if expected_count != totals["retry_count"]:
        raise ValueError(
            f"retry count mismatch: events={totals['retry_count']} summary={expected_count}"
        )
    if expected_restored != measured_restored:
        raise ValueError(
            f"restored-token mismatch: events={measured_restored} "
            f"summary={expected_restored}"
        )
    if expected_recompute != totals["recompute_tokens"]:
        raise ValueError(
            f"recompute-token mismatch: events={totals['recompute_tokens']} "
            f"summary={expected_recompute}"
        )
    totals["retry_tokens_total"] = measured_restored + totals["recompute_tokens"]
    return totals


def collect_measured_rows(suite_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for suffix, label, _color in POLICY_SPECS:
        result_path = suite_dir / f"{suffix}.json"
        payload = _read_json(result_path)
        workflow_rows = payload.get("workflows", [])
        if not workflow_rows:
            raise ValueError(f"no workflow samples in {result_path}")
        token_summary = _retry_token_breakdown(payload)
        if token_summary["mixed_l1_l2_reuse_tokens"]:
            raise ValueError(
                f"{result_path} contains mixed L1/L2 restores without an exact "
                "per-tier token split"
            )
        for workflow in workflow_rows:
            rows.append(
                {
                    "policy_suffix": suffix,
                    "policy_label": label,
                    "source_result": str(result_path),
                    "workflow_id": str(workflow.get("workflow_id") or ""),
                    "wct_s": round(float(workflow.get("wct_s", 0.0)), 6),
                }
            )
        wct_summary = payload.get("wct_summary_s", {})
        rows.append(
            {
                "policy_suffix": suffix,
                "policy_label": label,
                "source_result": str(result_path),
                "workflow_id": "__summary__",
                "wct_s": "",
                "workflow_count": len(workflow_rows),
                "mean_wct_s": wct_summary.get("mean", ""),
                "p50_wct_s": wct_summary.get("p50", ""),
                "p90_wct_s": wct_summary.get("p90", ""),
                "p99_wct_s": wct_summary.get("p99", ""),
                "max_wct_s": wct_summary.get("max", ""),
                **token_summary,
                "evidence_tier": payload.get("evidence_tier", ""),
                "failure_manifest_path": payload.get("failure_manifest_path", ""),
            }
        )
    return rows


def _curve_rows(measured_rows: list[dict[str, object]]) -> dict[str, list[float]]:
    return {
        suffix: [
            float(row["wct_s"])
            for row in measured_rows
            if row.get("policy_suffix") == suffix
            and row.get("workflow_id") != "__summary__"
        ]
        for suffix, _label, _color in POLICY_SPECS
    }


def _summary_rows(measured_rows: list[dict[str, object]]) -> dict[str, dict]:
    return {
        str(row["policy_suffix"]): row
        for row in measured_rows
        if row.get("workflow_id") == "__summary__"
    }


def render_figure(
    *,
    measured_rows: list[dict[str, object]],
    output_path: Path,
    draft_output_path: Path,
) -> None:
    curves = _curve_rows(measured_rows)
    summaries = _summary_rows(measured_rows)
    if any(not curves[suffix] for suffix, _label, _color in POLICY_SPECS):
        raise ValueError("missing WCT samples for one or more policies")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 300,
        }
    )
    fig, (ax, ax_retry) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.45),
        gridspec_kw={"width_ratios": (1.35, 1.0)},
    )

    for suffix, label, color in POLICY_SPECS:
        xs, ys = _ecdf(curves[suffix])
        ax.step(xs, ys, where="post", color=color, linewidth=2.4, label=label)
    ax.set_xlabel("Workflow completion time (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=11.5)
    ax.set_title(
        "(a) Workflow completion time",
        loc="left",
        weight="bold",
        fontsize=13,
        color=COLOR_MUTED,
        pad=8,
    )
    ax.spines[["top", "right"]].set_visible(False)

    x_positions = list(range(len(POLICY_SPECS)))
    bottoms = [0.0 for _ in POLICY_SPECS]
    for key, label, color in RETRY_TOKEN_SPECS:
        percentages: list[float] = []
        counts: list[int] = []
        for suffix, _policy_label, _policy_color in POLICY_SPECS:
            summary = summaries[suffix]
            count = int(summary[key])
            total = int(summary["retry_tokens_total"])
            counts.append(count)
            percentages.append(100.0 * count / total if total else 0.0)
        ax_retry.bar(
            x_positions,
            percentages,
            bottom=bottoms,
            width=0.62,
            color=color,
            label=label,
            zorder=2,
        )
        for x_pos, bottom, percentage, count in zip(
            x_positions, bottoms, percentages, counts
        ):
            if percentage < 7.0:
                continue
            ax_retry.text(
                x_pos,
                bottom + percentage / 2.0,
                f"{count:,}\n({percentage:.1f}%)",
                ha="center",
                va="center",
                color="white",
                fontsize=10.5,
                weight="bold",
                linespacing=1.15,
                zorder=3,
            )
        bottoms = [bottom + value for bottom, value in zip(bottoms, percentages)]

    tick_labels = [
        f"{label}\n{int(summaries[suffix]['retry_count'])} retries"
        for suffix, label, _color in POLICY_SPECS
    ]
    ax_retry.set_xticks(x_positions, tick_labels)
    ax_retry.set_ylabel("Retry-token source (%)")
    ax_retry.set_ylim(0.0, 112.0)
    ax_retry.set_yticks(range(0, 101, 20))
    ax_retry.grid(axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax_retry.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.2,
        fontsize=10.5,
    )
    ax_retry.set_title(
        "(b) Retry-token reuse",
        loc="left",
        weight="bold",
        fontsize=13,
        color=COLOR_MUTED,
        pad=8,
    )
    ax_retry.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Real CC trace / Qwen3-8B / context <= 1,024 / DP=8 / "
        "20 workflows / 2 GPU failures",
        x=0.06,
        y=0.992,
        ha="left",
        fontsize=12.5,
        weight="bold",
        color=COLOR_MUTED,
    )
    fig.text(
        0.5,
        0.025,
        "Preliminary scaled-token smoke; fixed time/rank manifest, unmatched "
        "victims (AegisKV: 24 retries; No Backup: 27 retries).",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLOR_MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.22, wspace=0.30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draft_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, pad_inches=0.04)
    fig.savefig(draft_output_path, dpi=300, pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render AegisKV versus No Backup WCT CDF with measured L1/L2 "
            "retry-token reuse."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draft-output", type=Path, default=DEFAULT_DRAFT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    measured_rows = collect_measured_rows(args.suite_dir)
    _write_csv(measured_rows, args.csv)
    render_figure(
        measured_rows=measured_rows,
        output_path=args.output,
        draft_output_path=args.draft_output,
    )
    print(f"wrote {args.csv}")
    print(f"wrote {args.output}")
    print(f"wrote {args.draft_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
