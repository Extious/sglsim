#!/usr/bin/env python3
"""Derive load-only per-node L1 Backup headroom for one DP8 replay.

The calculation uses only normal serving load: each request conservatively
holds its final KV-token footprint from creation to completion on its assigned
DP rank.  It does not use backup priority, replica placement, or backup demand.

    alpha_n = [1 - peak_serving_load_n / M_n - safety_margin]_+.

This is a measured replay illustration of load skew, not an end-to-end alpha
actuation sweep and not a strict-final policy claim.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "results"
    / "agentic_workflows"
    / "online-alpha-dp8-highload-highrisk-r1"
    / "runs"
    / "trial_000"
    / "20260704_034137"
    / "request_details.csv"
)
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_alg3_per_node_alpha.csv"
)
DEFAULT_PNG = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_alg3_per_node_alpha.png"
)


def configure_chinese_font() -> None:
    """Register the same Fandol family used by the XeLaTeX presentation."""
    result = subprocess.run(
        ["kpsewhich", "FandolHei-Regular.otf"],
        check=True,
        capture_output=True,
        text=True,
    )
    font_path = result.stdout.strip()
    if not font_path:
        raise RuntimeError("找不到 FandolHei-Regular.otf")
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.unicode_minus": False,
        }
    )


def peak_serving_tokens(frame: pd.DataFrame, rank: int) -> int:
    selected = frame[frame["aegis_assigned_dp_rank"] == rank]
    events: list[tuple[float, int, int]] = []
    for row in selected.itertuples():
        tokens = int(row.input_length + row.output_length)
        events.append((float(row.created_time_s), 1, tokens))
        events.append((float(row.finish_time_s), -1, tokens))
    # At identical timestamps a completed request releases capacity first.
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    peak = 0
    for _, direction, token_count in events:
        current += direction * token_count
        peak = max(peak, current)
    return peak


def analyze(
    frame: pd.DataFrame,
    safety_margin_fraction: float,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    dp_size = int(frame["aegis_assigned_dp_rank"].max() + 1)
    positive = frame[
        (frame["aegis_gpu_replica_quota_tokens"] > 0)
        & (frame["aegis_gpu_budget_fraction"] > 0)
    ]
    capacity_samples = (
        positive["aegis_gpu_replica_quota_tokens"]
        / positive["aegis_gpu_budget_fraction"]
    )
    total_cache_tokens = int(round(float(capacity_samples.median())))

    rows: list[dict[str, object]] = []
    for rank in range(dp_size):
        source = frame[frame["aegis_assigned_dp_rank"] == rank]
        peak = peak_serving_tokens(frame, rank)
        serving_load_fraction = peak / max(total_cache_tokens, 1)
        alpha_headroom = max(
            0.0,
            1.0 - serving_load_fraction - safety_margin_fraction,
        )
        rows.append(
            {
                "dp_rank": rank,
                "request_rows": int(len(source)),
                "input_tokens": int(source["input_length"].sum()),
                "peak_serving_kv_tokens": peak,
                "total_cache_tokens": total_cache_tokens,
                "peak_serving_load_fraction": serving_load_fraction,
                "safety_margin_fraction": safety_margin_fraction,
                "alpha_load_headroom": alpha_headroom,
                "evidence_tier": "preliminary measured load replay",
                "source": "online-alpha-dp8-highload-highrisk-r1 request_details.csv",
            }
        )

    alpha_values = np.array(
        [float(row["alpha_load_headroom"]) for row in rows]
    )
    uniform_safe_alpha = float(alpha_values.min())
    per_node_capacity = float(alpha_values.sum() * total_cache_tokens)
    uniform_capacity = float(uniform_safe_alpha * total_cache_tokens * dp_size)
    summary = {
        "dp_size": float(dp_size),
        "total_cache_tokens_per_node": float(total_cache_tokens),
        "safety_margin_fraction": safety_margin_fraction,
        "uniform_safe_alpha": uniform_safe_alpha,
        "per_node_backup_capacity_tokens": per_node_capacity,
        "uniform_backup_capacity_tokens": uniform_capacity,
        "per_node_capacity_gain": (
            per_node_capacity / uniform_capacity - 1.0
            if uniform_capacity > 0
            else 0.0
        ),
    }
    return rows, summary


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(
    rows: list[dict[str, object]],
    summary: dict[str, float],
    output: Path,
) -> None:
    frame = pd.DataFrame(rows)
    ranks = frame["dp_rank"].to_numpy()
    load = frame["peak_serving_load_fraction"].to_numpy()
    alpha = frame["alpha_load_headroom"].to_numpy()
    uniform_safe_alpha = summary["uniform_safe_alpha"]

    configure_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 3.7))
    axes[0].bar(
        ranks,
        load * 100.0,
        color="#4c78a8",
        label="峰值正常 L1 负载",
    )
    axes[0].set_xlabel("执行 DP 节点")
    axes[0].set_ylabel("可管理 L1 容量占比（%）")
    axes[0].set_title("(a) 各节点峰值正常服务负载")
    axes[0].set_xticks(ranks)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(
        ranks,
        alpha * 100.0,
        color="#2f8f65",
        label=r"负载导出的逐节点 $\alpha_n$",
    )
    axes[1].axhline(
        uniform_safe_alpha * 100.0,
        color="#c86535",
        linestyle="--",
        linewidth=2,
        label="统一安全比例",
    )
    axes[1].set_xlabel("执行 DP 节点")
    axes[1].set_ylabel("可用于 L1 备份的比例（%）")
    axes[1].set_title("(b) 负载感知的安全备份余量")
    axes[1].set_xticks(ranks)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(
        0.98,
        0.96,
        (
            "逐节点配置比统一安全比例\n"
            f"多提供 {summary['per_node_capacity_gain'] * 100:.1f}% 备份容量"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )

    fig.suptitle(
        "DP8 回放：正常 L1 负载偏斜产生不同安全备份余量",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--safety-margin-fraction", type=float, default=0.05)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    rows, summary = analyze(frame, args.safety_margin_fraction)
    write_csv(rows, args.csv)
    render(rows, summary, args.png)
    print(
        "uniform safe alpha={:.4f}; per-node headroom adds {:.1f}% capacity".format(
            summary["uniform_safe_alpha"],
            summary["per_node_capacity_gain"] * 100.0,
        )
    )
    for row in rows:
        print(
            "rank {dp_rank}: peak_serving={peak_serving_kv_tokens}, "
            "load={load:.4f}, alpha={alpha:.4f}".format(
                load=float(row["peak_serving_load_fraction"]),
                alpha=float(row["alpha_load_headroom"]),
                **row,
            )
        )


if __name__ == "__main__":
    main()
