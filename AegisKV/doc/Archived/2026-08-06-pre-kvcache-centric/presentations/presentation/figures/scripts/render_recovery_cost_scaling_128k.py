#!/usr/bin/env python3
"""Render full-context recovery paths up to each model's context ceiling.

This path-isolation figure deliberately removes L1/L2 backup-capacity limits:
L1 assumes the complete failed-request KV is GPU-resident, L2 assumes the
complete KV is Host-resident, and No Backup recomputes the complete context.
The context axis remains bounded by the configured model limit (128K for the
current Qwen3 experiments).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sglsim.srt.spec.accelerator.base import AcceleratorInfo
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.communication.kv_bytes import resolve_kv_byte_info
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.simulation.recovery_model import (
    ENV_PHYSICAL_COPY_CSV,
    ENV_RECOVERY_MODEL,
    aegis_restore_overhead_s,
)
from sglsim.srt.time_predictor.aiconfigurator import AIConfiguratorTimePredictor


PRESENTATION_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PRESENTATION_ROOT / "Figures" / "fig_recovery_cost_scaling_128k.png"
DATA_CSV = (
    PRESENTATION_ROOT
    / "figures"
    / "data"
    / "analytic"
    / "fig_recovery_cost_scaling_128k.csv"
)
PHYSICAL_RECOVERY_CSV = (
    PRESENTATION_ROOT
    / "experiments"
    / "sglang_small"
    / "results"
    / "physical_recovery_path_aggregate"
    / "physical_recovery_path_aggregate.csv"
)

TOKEN_TICKS = np.asarray([1024 * (2**power) for power in range(8)], dtype=int)
DISPLAY_TICKS = np.asarray([1024, 4096, 16384, 65536, 131072], dtype=int)
CHUNK_TOKENS = 8192
RETRY_OVERHEAD_S = 0.0024


@dataclass(frozen=True)
class Deployment:
    name: str
    family: str
    model_path: str
    layers: int
    kv_heads: int
    head_dim: int
    tp_size: int
    pp_size: int
    max_context_tokens: int
    ep_size: int = 1

    @property
    def device_count(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def parallelism_label(self) -> str:
        ep = f" / EP={self.ep_size}" if self.ep_size > 1 else ""
        return f"TP={self.tp_size} / PP={self.pp_size}{ep}"


DEPLOYMENTS = (
    Deployment(
        name="Qwen3-8B",
        family="dense",
        model_path="Qwen/Qwen3-8B",
        layers=36,
        kv_heads=8,
        head_dim=128,
        tp_size=4,
        pp_size=2,
        max_context_tokens=131_072,
    ),
    Deployment(
        name="Qwen3-32B",
        family="dense",
        model_path="Qwen/Qwen3-32B",
        layers=64,
        kv_heads=8,
        head_dim=128,
        tp_size=4,
        pp_size=2,
        max_context_tokens=131_072,
    ),
    Deployment(
        name="Qwen3-235B-A22B",
        family="MoE",
        model_path="Qwen/Qwen3-235B-A22B",
        layers=94,
        kv_heads=4,
        head_dim=128,
        tp_size=8,
        pp_size=4,
        max_context_tokens=131_072,
        ep_size=8,
    ),
)

PATHS = (
    ("No Backup: AIConfigurator full Prefill", "no_backup", "#6B7280", ":", "o"),
    (r"L1 reuse: SGLSim GPU P2P restore", "l1_reuse", "#059669", "-", "s"),
    (r"L2 reuse: SGLSim Host$\rightarrow$GPU restore", "l2_reuse", "#2563EB", "-.", "D"),
)


def _scheduler_config(deployment: Deployment) -> SchedulerSimConfig:
    return SchedulerSimConfig(
        tp_size=deployment.tp_size,
        pp_size=deployment.pp_size,
        ep_size=deployment.ep_size,
        dp_size=1,
        backend_name="sglang",
        backend_version="0.5.10",
    )


def _make_predictor(deployment: Deployment) -> AIConfiguratorTimePredictor:
    return AIConfiguratorTimePredictor(
        ModelInfo.from_model_path(
            deployment.model_path,
            context_len=deployment.max_context_tokens,
        ),
        AcceleratorInfo(name="a100_sxm", hbm_capacity_gb=80.0),
        _scheduler_config(deployment),
        {
            "database_mode": "SILICON",
            "prefill_scale_factor": 1.05,
            "decode_scale_factor": 0.9,
            "forward_scale": 1.0,
        },
    )


def _predict_prefill_suffix_s(
    predictor: AIConfiguratorTimePredictor,
    *,
    restored_prefix: int,
    target_context: int,
    cache: dict[tuple[int, int], float],
) -> float:
    """Sum exact 8K static_ctx points from the A100 SILICON database."""
    if target_context <= restored_prefix:
        return 0.0
    total_s = 0.0
    prefix = restored_prefix
    while prefix < target_context:
        extend = min(CHUNK_TOKENS, target_context - prefix)
        key = (prefix, extend)
        latency_s = cache.get(key)
        if latency_s is None:
            runtime = predictor._runtime_config_cls(
                batch_size=1,
                isl=prefix + extend,
                prefix=prefix,
                osl=1,
            )
            summary = predictor._session.run_static(runtime, mode="static_ctx")
            if summary.check_oom():
                raise RuntimeError(
                    "AIConfigurator reported OOM for "
                    f"prefix={prefix}, extend={extend}, target={target_context}"
                )
            latency_s = (
                sum(summary.get_context_latency_dict().values())
                / 1000.0
                * predictor.prefill_scale_factor
                * predictor.forward_scale
            )
            cache[key] = latency_s
        total_s += latency_s
        prefix += extend
    return total_s


def _calibration_bytes_per_token() -> int:
    """Return the KV payload unit used by the SGLSim physical-copy table."""
    ratios: set[int] = set()
    with PHYSICAL_RECOVERY_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("action") not in {
                "gpu_replica_restore_p2p",
                "host_restore_h2d",
            }:
                continue
            tokens = int(row["tokens"])
            payload_bytes = int(row["bytes"])
            if tokens > 0 and payload_bytes % tokens == 0:
                ratios.add(payload_bytes // tokens)
    if len(ratios) != 1:
        raise ValueError(
            "physical recovery calibration must have one bytes/token unit; "
            f"found {sorted(ratios)}"
        )
    return ratios.pop()


def _rank_kv_bytes_per_token(deployment: Deployment) -> int:
    """Use SGLSim's model/parallelism-aware KV-byte estimator."""
    info = resolve_kv_byte_info(
        model_path=deployment.model_path,
        sched=_scheduler_config(deployment),
    )
    return int(info.bytes_per_token_all_layers)


def _sglsim_restore_s(
    path: str,
    *,
    context_tokens: int,
    rank_kv_bytes_per_token: int,
    calibration_bytes_per_token: int,
) -> tuple[float, int, int]:
    """Predict restore from model-specific bytes using SGLSim's copy model.

    The measured SGLSim recovery table is indexed in Qwen3-8B-equivalent
    tokens.  Convert the actual per-rank payload to that calibrated unit before
    calling the runtime predictor.  This preserves the measured P2P/H2D paths
    while allowing model architecture and TP/PP layout to change the result.
    """
    if context_tokens <= 0:
        return 0.0, 0, 0
    payload_bytes = int(context_tokens * rank_kv_bytes_per_token)
    equivalent_tokens = max(
        1,
        int(round(payload_bytes / calibration_bytes_per_token)),
    )
    if path == "l1_reuse":
        action = "l1_backup"
    elif path == "l2_reuse":
        action = "l2_backup"
    else:
        raise ValueError(f"unsupported SGLSim restore path: {path}")
    return (
        aegis_restore_overhead_s(
            action=action,
            token_count=equivalent_tokens,
            restored_tokens=equivalent_tokens,
        ),
        payload_bytes,
        equivalent_tokens,
    )


def _format_context(tokens: float) -> str:
    if tokens >= 1024**2:
        return f"{tokens / 1024**2:g}M"
    return f"{tokens / 1024:g}K"


def _format_time(seconds: float) -> str:
    if seconds < 0.010:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 120.0:
        return f"{seconds:.1f} s"
    return f"{seconds / 60.0:.1f} min"


def build_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    calibration_bpt = _calibration_bytes_per_token()
    for deployment in DEPLOYMENTS:
        predictor = _make_predictor(deployment)
        rank_kv_bpt = _rank_kv_bytes_per_token(deployment)
        latency_cache: dict[tuple[int, int], float] = {}
        for context in TOKEN_TICKS[TOKEN_TICKS <= deployment.max_context_tokens]:
            context_i = int(context)
            for label, path, _color, _linestyle, _marker in PATHS:
                l1_reused = 0
                l2_reused = 0
                l1_restore_s = 0.0
                l2_restore_s = 0.0
                restore_payload_bytes = 0
                calibration_equivalent_tokens = 0
                if path == "l1_reuse":
                    l1_reused = context_i
                    (
                        l1_restore_s,
                        restore_payload_bytes,
                        calibration_equivalent_tokens,
                    ) = _sglsim_restore_s(
                        path,
                        context_tokens=l1_reused,
                        rank_kv_bytes_per_token=rank_kv_bpt,
                        calibration_bytes_per_token=calibration_bpt,
                    )
                elif path == "l2_reuse":
                    l2_reused = context_i
                    (
                        l2_restore_s,
                        restore_payload_bytes,
                        calibration_equivalent_tokens,
                    ) = _sglsim_restore_s(
                        path,
                        context_tokens=l2_reused,
                        rank_kv_bytes_per_token=rank_kv_bpt,
                        calibration_bytes_per_token=calibration_bpt,
                    )
                restored = l1_reused + l2_reused
                prefill_s = _predict_prefill_suffix_s(
                    predictor,
                    restored_prefix=restored,
                    target_context=context_i,
                    cache=latency_cache,
                )
                rows.append(
                    {
                        "model": deployment.name,
                        "model_family": deployment.family,
                        "accelerator": "A100 80GB",
                        "tp_size": deployment.tp_size,
                        "pp_size": deployment.pp_size,
                        "ep_size": deployment.ep_size,
                        "device_count": deployment.device_count,
                        "recovery_path": label,
                        "path_key": path,
                        "context_tokens": context_i,
                        "configured_context_limit_tokens": deployment.max_context_tokens,
                        "backup_capacity_limit_applied": 0,
                        "l1_capacity_tokens": "",
                        "l2_capacity_tokens": "",
                        "alpha": "",
                        "beta": "",
                        "l1_reused_tokens": l1_reused,
                        "l2_reused_tokens": l2_reused,
                        "recomputed_tokens": context_i - restored,
                        "rank_kv_bytes_per_token": rank_kv_bpt,
                        "restore_payload_bytes": restore_payload_bytes,
                        "calibration_bytes_per_token": calibration_bpt,
                        "calibration_equivalent_tokens": calibration_equivalent_tokens,
                        "aiconfigurator_prefill_s": prefill_s,
                        "l1_restore_s": l1_restore_s,
                        "l2_restore_s": l2_restore_s,
                        "h2d_s": l2_restore_s,
                        "retry_overhead_s": (
                            RETRY_OVERHEAD_S if path == "no_backup" else 0.0
                        ),
                        "recovery_time_s": (
                            (RETRY_OVERHEAD_S if path == "no_backup" else 0.0)
                            + l1_restore_s
                            + l2_restore_s
                            + prefill_s
                        ),
                        "sglsim_restore_timing_model": "physical_recovery_path_csv",
                        "sglsim_restore_calibration_csv": str(PHYSICAL_RECOVERY_CSV),
                        "aiconfigurator_system": "a100_sxm",
                        "aiconfigurator_backend": "sglang-0.5.10",
                        "aiconfigurator_database_mode": "SILICON",
                        "chunk_tokens": CHUNK_TOKENS,
                        "aiconfigurator_oom": 0,
                    }
                )
    return rows


def write_data(rows: list[dict[str, float | int | str]]) -> None:
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DATA_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict[str, float | int | str]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.6), sharey=True)
    global_max_recovery_s = max(float(row["recovery_time_s"]) for row in rows)
    endpoint_offsets = {
        "no_backup": 18,
        "l1_reuse": -10,
        "l2_reuse": 10,
    }

    for panel_idx, (ax, deployment) in enumerate(zip(axes, DEPLOYMENTS)):
        model_rows = [row for row in rows if row["model"] == deployment.name]
        rank_kv_bpt = int(model_rows[0]["rank_kv_bytes_per_token"])
        for label, path, color, linestyle, marker in PATHS:
            path_rows = [row for row in model_rows if row["path_key"] == path]
            x = np.asarray([int(row["context_tokens"]) for row in path_rows])
            y = np.asarray([float(row["recovery_time_s"]) for row in path_rows])
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=3.1,
                marker=marker,
                markersize=6.5,
                label=label,
            )
            ax.annotate(
                _format_time(float(y[-1])),
                xy=(x[-1], y[-1]),
                xytext=(-7, endpoint_offsets[path]),
                textcoords="offset points",
                ha="right",
                va=(
                    "bottom"
                    if endpoint_offsets[path] > 0
                    else "top"
                    if endpoint_offsets[path] < 0
                    else "center"
                ),
                fontsize=12.5,
                color=color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 0.5,
                },
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        model_ticks = DISPLAY_TICKS[
            DISPLAY_TICKS <= deployment.max_context_tokens
        ]
        ax.set_xticks(model_ticks)
        ax.set_xticklabels([_format_context(value) for value in model_ticks])
        ax.set_xlim(TOKEN_TICKS[0], deployment.max_context_tokens * 1.12)
        ax.set_ylim(1.5e-3, global_max_recovery_s * 2.2)
        ax.set_xlabel("Failed-request context (tokens)")
        ax.grid(color="#E5E7EB", linewidth=0.85, which="both")
        ax.set_title(
            f"({chr(ord('a') + panel_idx)}) {deployment.name} ({deployment.family})\n"
            f"A100 x {deployment.device_count} | {deployment.parallelism_label} | context <= 128K",
            weight="bold",
            fontsize=13.5,
            pad=11,
        )
        ax.text(
            0.025,
            0.965,
            "AIConfigurator SILICON\n"
            "No Backup: 8K chunked Prefill\n"
            "L1/L2: SGLSim physical-copy model\n"
            f"Per-rank KV: {rank_kv_bpt / 1024:.1f} KiB/token",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.5,
            linespacing=1.25,
            bbox={
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "alpha": 0.92,
                "boxstyle": "round,pad=0.35",
            },
        )

    axes[0].set_ylabel("Recovery time (s, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        frameon=False,
        columnspacing=2.0,
        handlelength=3.0,
    )
    fig.suptitle(
        "AIConfigurator Recovery Simulation on A100 80GB GPUs",
        fontsize=21,
        weight="bold",
        y=0.975,
    )
    fig.subplots_adjust(left=0.065, right=0.988, bottom=0.105, top=0.775, wspace=0.15)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    # Pin the figure to the checked-in SGLSim A100 recovery calibration rather
    # than inheriting an ambient coefficient-mode environment.
    os.environ[ENV_RECOVERY_MODEL] = "physical_copy"
    os.environ[ENV_PHYSICAL_COPY_CSV] = str(PHYSICAL_RECOVERY_CSV)
    rows = build_rows()
    write_data(rows)
    render(rows)
    print(f"wrote {OUTPUT}")
    print(f"wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {DATA_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
