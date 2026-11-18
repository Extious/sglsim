#!/usr/bin/env python3
"""Render one-column long-context recovery-component predictions.

The main axes show model-aware full-prefix recomputation from the public
AIConfigurator predictor used by SGLSim.  A linear inset shows the current
Aegis failover planner's default direct-HBM, local-DRAM, and SSD path costs.

The two sources must not be conflated: recomputation is a SILICON-database
configuration prediction, while the tier curves are uncalibrated simulator
defaults.  None is an end-to-end hardware recovery measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from sglsim.srt.simulation.aegis_placement import AegisPathCostModel
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.spec.accelerator.base import AcceleratorInfo
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.time_predictor.aiconfigurator import AIConfiguratorTimePredictor
from sglsim.srt.time_predictor.base import ScheduleBatch, ScheduleRequest


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
DEFAULT_OUTPUT = ROOT / "figures" / "motivation-recovery-scaling.png"
DEFAULT_DATA = ROOT / "figures" / "data" / "motivation-recovery-scaling.csv"

CONTEXT_LIMIT = 131_072
CONTEXTS = np.asarray([16_384, 32_768, 65_536, 131_072], dtype=int)
DISPLAY_TICK_LABELS = ["16K", "32K", "64K", "128K"]
CHUNK_TOKENS = 8192

AICONFIGURATOR_DATABASE_MODE = "SILICON"
AICONFIGURATOR_PREFILL_SCALE = 1.05
AICONFIGURATOR_DECODE_SCALE = 0.9
AICONFIGURATOR_FORWARD_SCALE = 1.0
AICONFIGURATOR_BACKEND = "sglang-0.5.10"

FIGURE_SCOPE = "isolated_long_context_recovery_component_time"
RUNTIME_BOUNDARY = "not_end_to_end_recovery_and_not_physical_pp_run_verified"


@dataclass(frozen=True)
class Deployment:
    name: str
    model_path: str
    tp_size: int
    pp_size: int
    ep_size: int = 1
    color: str = "#0072B2"
    marker: str = "o"

    @property
    def key(self) -> str:
        return f"{self.name}-tp{self.tp_size}-pp{self.pp_size}-ep{self.ep_size}"

    @property
    def compact_label(self) -> str:
        topology = f"TP{self.tp_size}\u00d7PP{self.pp_size}"
        if self.ep_size > 1:
            topology += f"\u00d7EP{self.ep_size}"
        return f"{self.name.removeprefix('Qwen3-')}  {topology}"


DEPLOYMENTS = (
    Deployment(
        name="Qwen3-8B",
        model_path="Qwen/Qwen3-8B",
        tp_size=4,
        pp_size=2,
        color="#0072B2",
        marker="o",
    ),
    Deployment(
        name="Qwen3-32B",
        model_path="Qwen/Qwen3-32B",
        tp_size=4,
        pp_size=2,
        color="#D55E00",
        marker="s",
    ),
    Deployment(
        name="Qwen3-235B-A22B",
        model_path="Qwen/Qwen3-235B-A22B",
        tp_size=8,
        pp_size=4,
        ep_size=8,
        color="#009E73",
        marker="^",
    ),
)

TIER_PATHS = (
    (
        "peer_hbm",
        "Direct peer HBM",
        ("direct_hbm_reuse",),
        "#228833",
        (0, (5, 2)),
        "uncalibrated_aegis_path_cost_default",
    ),
    (
        "local_dram",
        "Local DRAM",
        ("h2d",),
        "#4477AA",
        (0, (4, 1.5, 1, 1.5)),
        "uncalibrated_aegis_path_cost_default",
    ),
    (
        "ssd",
        "SSD",
        ("l3_read", "h2d"),
        "#AA3377",
        (0, (1.2, 1.5)),
        "uncalibrated_aegis_path_cost_default",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE.resolve()))
    except ValueError:
        return str(resolved)


def _source_path(obj: object) -> Path:
    source = inspect.getsourcefile(obj)
    if not source:
        raise RuntimeError(f"cannot resolve source file for {obj!r}")
    return Path(source).resolve()


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
            context_len=CONTEXT_LIMIT,
        ),
        AcceleratorInfo(name="a100_sxm", hbm_capacity_gb=80.0),
        _scheduler_config(deployment),
        {
            "database_mode": AICONFIGURATOR_DATABASE_MODE,
            "prefill_scale_factor": AICONFIGURATOR_PREFILL_SCALE,
            "decode_scale_factor": AICONFIGURATOR_DECODE_SCALE,
            "forward_scale": AICONFIGURATOR_FORWARD_SCALE,
        },
    )


def _predict_full_prefix_recompute_s(
    predictor: AIConfiguratorTimePredictor,
    context_tokens: int,
) -> tuple[float, tuple[float, ...]]:
    """Teacher-force the failed prefix through the public predictor API."""
    chunk_latencies: list[float] = []
    prefix = 0
    while prefix < context_tokens:
        extend = min(CHUNK_TOKENS, context_tokens - prefix)
        latency_s = predictor.predict_infer_time(
            ScheduleBatch(
                reqs=[
                    ScheduleRequest(
                        extend_length=extend,
                        past_kv_length=prefix,
                    )
                ]
            )
        )
        if latency_s <= 0.0:
            raise RuntimeError(
                "AIConfigurator returned non-positive prefill latency for "
                f"prefix={prefix}, extend={extend}, context={context_tokens}"
            )
        chunk_latencies.append(float(latency_s))
        prefix += extend
    return sum(chunk_latencies), tuple(chunk_latencies)


def _common_metadata() -> dict[str, object]:
    predictor_source = _source_path(AIConfiguratorTimePredictor)
    path_model_source = _source_path(AegisPathCostModel)
    generator_source = Path(__file__).resolve()
    return {
        "figure_scope": FIGURE_SCOPE,
        "runtime_boundary": RUNTIME_BOUNDARY,
        "predictor_source": _display_path(predictor_source),
        "predictor_source_sha256": _sha256(predictor_source),
        "path_model_source": _display_path(path_model_source),
        "path_model_source_sha256": _sha256(path_model_source),
        "generator_sha256": _sha256(generator_source),
        "aiconfigurator_system": "a100_sxm",
        "aiconfigurator_backend": AICONFIGURATOR_BACKEND,
        "aiconfigurator_database_mode": AICONFIGURATOR_DATABASE_MODE,
        "aiconfigurator_prefill_scale_factor": AICONFIGURATOR_PREFILL_SCALE,
        "aiconfigurator_decode_scale_factor": AICONFIGURATOR_DECODE_SCALE,
        "aiconfigurator_forward_scale": AICONFIGURATOR_FORWARD_SCALE,
        "configured_context_limit_tokens": CONTEXT_LIMIT,
    }


def build_rows() -> list[dict[str, object]]:
    metadata = _common_metadata()
    path_model = AegisPathCostModel()
    path_defaults = asdict(path_model)
    rows: list[dict[str, object]] = []

    for deployment in DEPLOYMENTS:
        predictor = _make_predictor(deployment)
        for context in CONTEXTS:
            context_i = int(context)
            recompute_s, chunks = _predict_full_prefix_recompute_s(
                predictor,
                context_i,
            )
            rows.append(
                {
                    "deployment": deployment.key,
                    "model": deployment.name,
                    "model_path": deployment.model_path,
                    "accelerator": "A100 80GB",
                    "tp_size": deployment.tp_size,
                    "pp_size": deployment.pp_size,
                    "ep_size": deployment.ep_size,
                    "context_tokens": context_i,
                    "path_key": "recompute",
                    "path_label": "Full recompute",
                    "path_components": "aiconfigurator_prefill",
                    "component_time_s": recompute_s,
                    "chunk_count": len(chunks),
                    "chunk_latencies_s": ";".join(
                        f"{value:.12g}" for value in chunks
                    ),
                    "chunk_tokens": CHUNK_TOKENS,
                    "timing_source": (
                        "AIConfigurator SILICON public predictor; 8K chunks"
                    ),
                    "evidence_tier": (
                        "silicon_database_configuration_prediction"
                    ),
                    "model_specific": 1,
                    "local_dram_assumption": "",
                    **path_defaults,
                    **metadata,
                }
            )

    for context in CONTEXTS:
        context_i = int(context)
        for (
            path_key,
            path_label,
            components,
            _color,
            _linestyle,
            evidence_tier,
        ) in TIER_PATHS:
            rows.append(
                {
                    "deployment": "shared-aegis-tier-default",
                    "model": "model-independent token-cost default",
                    "model_path": "",
                    "accelerator": "",
                    "tp_size": "",
                    "pp_size": "",
                    "ep_size": "",
                    "context_tokens": context_i,
                    "path_key": path_key,
                    "path_label": path_label,
                    "path_components": "+".join(components),
                    "component_time_s": (
                        path_model.path_ms(components, context_i) / 1000.0
                    ),
                    "chunk_count": 0,
                    "chunk_latencies_s": "",
                    "chunk_tokens": "",
                    "timing_source": "AegisPathCostModel default coefficients",
                    "evidence_tier": evidence_tier,
                    "model_specific": 0,
                    "local_dram_assumption": (
                        "DRAM pool is local to retry node"
                        if path_key == "local_dram"
                        else ""
                    ),
                    **path_defaults,
                    **metadata,
                }
            )
    return rows


def write_data(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selected(
    rows: list[dict[str, object]],
    *,
    path_key: str,
    deployment: str | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["path_key"] == path_key
        and (deployment is None or row["deployment"] == deployment)
    ]


def render(rows: list[dict[str, object]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 8.1,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.34, 2.82))

    for deployment in DEPLOYMENTS:
        selected = _selected(
            rows,
            path_key="recompute",
            deployment=deployment.key,
        )
        x = [int(row["context_tokens"]) for row in selected]
        y = [float(row["component_time_s"]) for row in selected]
        ax.plot(
            x,
            y,
            color=deployment.color,
            marker=deployment.marker,
            markersize=3.7,
            markeredgewidth=0.55,
            linewidth=1.65,
            zorder=3,
        )
        topology = f"TP{deployment.tp_size}\u00d7PP{deployment.pp_size}"
        if deployment.ep_size > 1:
            topology += f"\u00d7EP{deployment.ep_size}"
        ax.annotate(
            (
                f"{deployment.name.removeprefix('Qwen3-')}"
                f"\n{topology}"
                f"\n{y[-1]:.1f} s"
            ),
            xy=(x[-1], y[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=5.8,
            fontweight="semibold",
            color=deployment.color,
            ha="left",
            va="center",
            linespacing=0.86,
            annotation_clip=False,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(CONTEXTS)
    ax.set_xticklabels(DISPLAY_TICK_LABELS)
    # Keep the data axis faithful to the 16K--128K domain.  A dedicated figure
    # gutter to its right carries the three endpoint labels, aligned to the
    # same left edge and completely outside the curve field.
    ax.set_xlim(14_500, 145_000)
    ax.set_ylim(0.0, 64.5)
    ax.set_xlabel("Failed prefix (tokens)", labelpad=2)
    ax.set_ylabel("Modeled component time (s)", labelpad=2)
    ax.grid(
        True,
        which="major",
        color="#D9DEE7",
        linewidth=0.5,
        alpha=0.82,
    )
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # The inset is part of the same chart and retains a linear seconds axis.
    # It exists only to make sub-second tier defaults visible beside the
    # tens-of-seconds AIConfigurator predictions.
    inset = ax.inset_axes([0.055, 0.59, 0.62, 0.33])
    path_handles: list[Line2D] = []
    for path_key, path_label, _components, color, linestyle, _tier in TIER_PATHS:
        selected = _selected(rows, path_key=path_key)
        inset.plot(
            [int(row["context_tokens"]) for row in selected],
            [float(row["component_time_s"]) for row in selected],
            color=color,
            linestyle=linestyle,
            linewidth=1.35,
            marker="o",
            markersize=2.5,
            markeredgewidth=0.35,
            zorder=3,
        )
        legend_label = path_label
        if path_key == "peer_hbm":
            legend_label = "Direct HBM · 0.2 ms"
        path_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                linewidth=1.25,
                label=legend_label,
            )
        )

    inset.set_xscale("log", base=2)
    inset.set_xticks(CONTEXTS)
    inset.set_xticklabels(DISPLAY_TICK_LABELS, fontsize=5.9)
    inset.set_xlim(14_500, 145_000)
    inset.set_ylim(-0.025, 1.25)
    inset.set_yticks([0.0, 0.4, 0.8, 1.2])
    inset.yaxis.tick_right()
    inset.tick_params(
        axis="y",
        labelleft=False,
        labelright=True,
        labelsize=5.9,
        pad=1.2,
    )
    inset.tick_params(axis="x", pad=1.2)
    inset.grid(True, color="#DFE4EA", linewidth=0.38, alpha=0.9)
    inset.set_axisbelow(True)
    inset.set_title(
        "Reuse paths (s)",
        fontsize=6.3,
        loc="left",
        pad=1.5,
    )
    inset.legend(
        handles=path_handles,
        loc="upper left",
        bbox_to_anchor=(0.018, 0.975),
        frameon=False,
        fontsize=5.8,
        handlelength=1.55,
        borderaxespad=0.0,
        labelspacing=0.13,
    )
    for spine in inset.spines.values():
        spine.set_color("#667085")
        spine.set_linewidth(0.65)

    fig.subplots_adjust(left=0.19, right=0.75, bottom=0.19, top=0.975)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=420)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    rows = build_rows()
    write_data(rows, args.data)
    render(rows, args.output)
    print(args.output)
    print(args.output.with_suffix(".pdf"))
    print(args.data)
    for deployment in DEPLOYMENTS:
        point = next(
            row
            for row in rows
            if row["deployment"] == deployment.key
            and row["path_key"] == "recompute"
            and int(row["context_tokens"]) == 131_072
        )
        print(f"{deployment.name} recompute @ 128K: {float(point['component_time_s']):.6f}s")
    for path_key, path_label, _components, _color, _style, _tier in TIER_PATHS:
        point = next(
            row
            for row in rows
            if row["path_key"] == path_key
            and int(row["context_tokens"]) == 131_072
        )
        print(f"{path_label} @ 128K: {float(point['component_time_s']):.6f}s")


if __name__ == "__main__":
    main()
