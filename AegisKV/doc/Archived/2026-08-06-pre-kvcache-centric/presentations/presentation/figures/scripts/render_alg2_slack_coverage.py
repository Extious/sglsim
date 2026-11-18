#!/usr/bin/env python3
"""Render the exact slack transformation over measured recovery anchors.

The L1 Backup and L2 Backup paths use physical CUDA movement measurements.
The Recompute path is the current SGLSim coefficient model.  The transformation
from retry time to visible workflow delay is analytical: [R - slack]_+.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CALIBRATION = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_real_system_recovery_calibration.csv"
)
DEFAULT_MOVEMENT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglang_small"
    / "results"
    / "physical_recovery_path_aggregate"
    / "physical_recovery_path_aggregate.csv"
)
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_alg2_slack_coverage.csv"
)
DEFAULT_PNG = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_alg2_slack_coverage.png"
)

COLORS = {
    "L1 Backup": "#2676b8",
    "L2 Backup": "#2f8f65",
    "Recompute": "#c86535",
}

DISPLAY_LABELS = {
    "L1 Backup": "一级备份",
    "L2 Backup": "二级备份",
    "Recompute": "重算",
}


def configure_chinese_font() -> str:
    result = subprocess.run(
        ["kpsewhich", "FandolHei-Regular.otf"],
        check=False,
        capture_output=True,
        text=True,
    )
    font_path = Path(result.stdout.strip()) if result.stdout.strip() else None
    if font_path is not None and font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        return font_manager.FontProperties(fname=font_path).get_name()
    return "DejaVu Sans"


def load_paths(calibration_path: Path, movement_path: Path) -> pd.DataFrame:
    calibration = pd.read_csv(calibration_path)
    anchors = calibration[
        calibration["row_type"].isin(
            ["path_latency_anchor", "sglsim_recompute_prediction"]
        )
    ].copy()
    anchors["strategy"] = anchors["path_label"].map(
        {
            "Peer GPU restore": "L1 Backup",
            "Host restore": "L2 Backup",
            "Recompute": "Recompute",
        }
    )
    anchors["retry_ms"] = anchors["physical_anchor_restore_ms"].fillna(
        anchors["sglsim_prediction_ms"]
    )
    anchors = anchors[["strategy", "tokens", "retry_ms", "evidence_tier"]]

    movement = pd.read_csv(movement_path)
    writes = movement[movement["phase"] == "backup"].copy()
    writes["strategy"] = writes["recovery_path"].map(
        {"GPUReplica": "L1 Backup", "HostBackup": "L2 Backup"}
    )
    writes = writes[["strategy", "tokens", "p50_ms_mean"]].rename(
        columns={"p50_ms_mean": "backup_write_ms"}
    )
    anchors = anchors.merge(writes, on=["strategy", "tokens"], how="left")
    anchors["backup_write_ms"] = anchors["backup_write_ms"].fillna(0.0)
    return anchors.sort_values(["tokens", "strategy"]).reset_index(drop=True)


def build_rows(paths: pd.DataFrame, focus_tokens: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in paths.itertuples():
        rows.append(
            {
                "row_type": "path_anchor",
                "strategy": item.strategy,
                "tokens": int(item.tokens),
                "retry_ms": float(item.retry_ms),
                "backup_write_ms": float(item.backup_write_ms),
                "slack_ms": "",
                "visible_wct_penalty_ms": "",
                "benefit_vs_recompute_ms": "",
                "evidence_tier": item.evidence_tier,
                "source": (
                    "SGLang physical CUDA movement"
                    if item.strategy != "Recompute"
                    else "SGLSim recompute coefficient model"
                ),
            }
        )

    focus = paths[paths["tokens"] == focus_tokens].set_index("strategy")
    recompute_ms = float(focus.loc["Recompute", "retry_ms"])
    slack_grid = np.linspace(0.0, recompute_ms * 1.15, 80)
    for strategy, item in focus.iterrows():
        retry_ms = float(item["retry_ms"])
        for slack_ms in slack_grid:
            penalty = max(retry_ms - float(slack_ms), 0.0)
            baseline_penalty = max(recompute_ms - float(slack_ms), 0.0)
            rows.append(
                {
                    "row_type": "slack_curve",
                    "strategy": strategy,
                    "tokens": focus_tokens,
                    "retry_ms": retry_ms,
                    "backup_write_ms": float(item["backup_write_ms"]),
                    "slack_ms": float(slack_ms),
                    "visible_wct_penalty_ms": penalty,
                    "benefit_vs_recompute_ms": max(
                        baseline_penalty - penalty, 0.0
                    ),
                    "evidence_tier": item["evidence_tier"],
                    "source": "exact [retry - slack]_+ transformation",
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(paths: pd.DataFrame, focus_tokens: int, output: Path) -> None:
    focus = paths[paths["tokens"] == focus_tokens].set_index("strategy")
    recompute_ms = float(focus.loc["Recompute", "retry_ms"])
    slack = np.linspace(0.0, recompute_ms * 1.15, 300)
    strategies = ["L1 Backup", "L2 Backup", "Recompute"]

    plt.rcParams.update(
        {
            "font.family": configure_chinese_font(),
            "font.size": 9,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55))

    for strategy in strategies:
        subset = paths[paths["strategy"] == strategy]
        axes[0].plot(
            subset["tokens"],
            subset["retry_ms"],
            marker="o",
            linewidth=2,
            color=COLORS[strategy],
            label=DISPLAY_LABELS[strategy],
        )
    axes[0].set_xlabel("已恢复的 KV 词元数")
    axes[0].set_ylabel("重试路径时延（毫秒）")
    axes[0].set_title("(a) 实测与建模的重试路径")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    for strategy in strategies:
        retry_ms = float(focus.loc[strategy, "retry_ms"])
        axes[1].plot(
            slack,
            np.maximum(retry_ms - slack, 0.0),
            linewidth=2,
            color=COLORS[strategy],
            label=DISPLAY_LABELS[strategy],
        )
        axes[1].axvline(
            retry_ms,
            color=COLORS[strategy],
            linestyle=":",
            linewidth=1,
            alpha=0.75,
        )
    axes[1].set_xlabel(r"请求松弛时间 $\sigma$（毫秒）")
    axes[1].set_ylabel(r"可见惩罚 $[R-\sigma]_+$（毫秒）")
    axes[1].set_title(f"(b) 精确松弛覆盖，{focus_tokens:,} 个词元")
    axes[1].grid(alpha=0.22)

    backup_strategies = ["L1 Backup", "L2 Backup"]
    x = np.arange(len(backup_strategies))
    write_cost = [
        float(focus.loc[strategy, "backup_write_ms"])
        for strategy in backup_strategies
    ]
    benefit = [
        recompute_ms - float(focus.loc[strategy, "retry_ms"])
        for strategy in backup_strategies
    ]
    width = 0.35
    axes[2].bar(
        x - width / 2,
        write_cost,
        width,
        color=[COLORS[strategy] for strategy in backup_strategies],
        alpha=0.55,
        label="备份写入成本",
    )
    axes[2].bar(
        x + width / 2,
        benefit,
        width,
        color=[COLORS[strategy] for strategy in backup_strategies],
        label=r"$\sigma=0$ 时的收益",
    )
    axes[2].set_xticks(x, [DISPLAY_LABELS[item] for item in backup_strategies])
    axes[2].set_ylabel("时延（毫秒）")
    axes[2].set_title("(c) 零松弛边界")
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(frameon=False, fontsize=8)
    for container in axes[2].containers:
        axes[2].bar_label(container, fmt="%.1f", padding=2, fontsize=8)

    fig.suptitle(
        "松弛时间将每条恢复路径转换为工作流可见时延",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--movement", type=Path, default=DEFAULT_MOVEMENT)
    parser.add_argument("--focus-tokens", type=int, default=4096)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()

    paths = load_paths(args.calibration, args.movement)
    rows = build_rows(paths, args.focus_tokens)
    write_csv(rows, args.csv)
    render(paths, args.focus_tokens, args.png)

    focus = paths[paths["tokens"] == args.focus_tokens].set_index("strategy")
    recompute = float(focus.loc["Recompute", "retry_ms"])
    for strategy in ["L1 Backup", "L2 Backup"]:
        retry = float(focus.loc[strategy, "retry_ms"])
        write = float(focus.loc[strategy, "backup_write_ms"])
        print(
            f"{strategy}: write={write:.3f} ms, retry={retry:.3f} ms, "
            f"zero-slack benefit={recompute - retry:.3f} ms"
        )


if __name__ == "__main__":
    main()
