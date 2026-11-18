#!/usr/bin/env python3
"""Repackage frozen targeted-v2 data with a collision-free S4 layout.

This script never changes simulated values.  It verifies the complete frozen
input checksum tree, copies the plotting sources, and atomically publishes a
versioned figure-only bundle.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

import run_active_hbm_v313_s4_s5_targeted_v2 as runner
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results/active_hbm_v313_s4_s5_targeted_v2"
OUTPUT_NAME = "active_hbm_v313_s4_s5_targeted_v2_figures_v4"


def verify_checksum_tree(path: Path) -> bool:
    for line in (path / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if core.sha256(path / name) != digest:
            return False
    return True


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_s4(
    path: Path,
    alg1: list[dict[str, str]],
    alg2: list[dict[str, str]],
    alg3: list[dict[str, str]],
) -> None:
    runner.v1run.style()
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.95),
        gridspec_kw={"width_ratios": [0.82, 1.35, 1.0]},
    )

    # Algorithm 1: priority should direct protected bytes to critical branches.
    x = np.arange(2)
    y = [float(row["critical_hbm_copy_share_mean"]) for row in alg1]
    e = [float(row["critical_hbm_copy_share_ci95_half_width"]) for row in alg1]
    axes[0].errorbar(x, y, yerr=e, fmt="o", color="#d62728", capsize=3, markersize=5)
    axes[0].set_xticks(x, ["Full", "-Alg. 1"])
    axes[0].set_ylabel("Critical-branch share of\nPeer-HBM copy bytes")
    axes[0].set_ylim(0, max(0.5, max(y) * 1.25))
    axes[0].set_title("Algorithm 1\nslack priority")
    axes[0].grid(axis="y", alpha=0.25)

    # Algorithm 2: paired bars share one compact benefit-class tick.  Hatching
    # identifies the ablation, while color encodes the resulting residency.
    benefits = ["both_benefit", "hbm_benefit", "dram_benefit", "neither_benefit"]
    benefit_labels = ["Both", "HBM", "DRAM", "Neither"]
    outcomes = ["HBM_AND_DRAM", "HBM_ONLY", "DRAM_ONLY", "BYPASS"]
    outcome_labels = ["HBM + DRAM", "HBM only", "DRAM only", "Bypass"]
    colors = ["#1b9e77", "#66a61e", "#7570b3", "#bdbdbd"]
    group_centers = np.arange(len(benefits)) * 1.75
    offsets = (-0.32, 0.32)
    policies = ("Full AegisKV", "-Algorithm 2")
    for center, benefit in zip(group_centers, benefits):
        for offset, policy in zip(offsets, policies):
            bottom = 0.0
            for outcome, color in zip(outcomes, colors):
                row = next(
                    item for item in alg2
                    if item["policy"] == policy
                    and item["benefit_class"] == benefit
                    and item["outcome"] == outcome
                )
                height = float(row["fraction_mean"])
                axes[1].bar(
                    center + offset, height, bottom=bottom, width=0.58,
                    color=color, edgecolor="#444444" if policy == "-Algorithm 2" else "none",
                    linewidth=0.45, hatch="///" if policy == "-Algorithm 2" else None,
                )
                bottom += height
    axes[1].set_xticks(group_centers, benefit_labels)
    axes[1].set_xlabel("Eligible tier benefit")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Residency fraction")
    axes[1].set_title("Algorithm 2\ntier independence")
    # Algorithm 3: target-level AIMD response under the shared schedule.
    for policy, color, marker in (
        ("Full AegisKV", "#7b3294", "o"),
        ("-Algorithm 3", "#555555", "s"),
    ):
        selected = [row for row in alg3 if row["policy"] == policy]
        t = np.asarray([float(row["time_s"]) for row in selected])
        mean = np.asarray([float(row["alpha_mean"]) for row in selected])
        half = np.asarray([float(row["alpha_ci95_half_width"]) for row in selected])
        axes[2].plot(t, mean, color=color, marker=marker, markersize=3, label=policy)
        axes[2].fill_between(t, mean - half, mean + half, color=color, alpha=0.13)
    axes[2].axvspan(90, 180, color="#d62728", alpha=0.06)
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel(r"Peer-HBM $\alpha$")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("Algorithm 3\nAIMD response")
    axes[2].legend(frameon=False, loc="best")
    axes[2].grid(axis="y", alpha=0.25)

    legend_handles = [
        Patch(facecolor="white", edgecolor="#777777", label="Full"),
        Patch(facecolor="white", edgecolor="#444444", hatch="///", label="-Alg. 2"),
        *[Patch(facecolor=color, label=label) for color, label in zip(colors, outcome_labels)],
    ]
    fig.legend(
        handles=legend_handles, frameon=False, ncol=6, loc="lower center",
        bbox_to_anchor=(0.5, 0.005), columnspacing=1.1, handlelength=1.2,
        fontsize=7.0,
    )
    fig.suptitle("S4 mechanism-targeted ablations (3 paired seeds)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.91), w_pad=1.0)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not verify_checksum_tree(INPUT):
        raise RuntimeError("frozen targeted-v2 input checksum failed")
    output = ROOT / "results" / OUTPUT_NAME
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage = ROOT / "results" / f".{OUTPUT_NAME}.stage-{os.getpid()}"
    (stage / "figures").mkdir(parents=True)

    sources = (
        "s4_alg1_plot_source.csv", "s4_alg1_seed.csv",
        "s4_alg2_plot_source.csv", "s4_alg2_seed.csv",
        "s4_alg3_plot_source.csv", "s4_alg3_timeline_seed.csv",
        "s4_paired_wct_ci.csv", "s5_plot_source.csv", "s5_timeline_seed.csv",
    )
    for name in sources:
        shutil.copy2(INPUT / name, stage / name)
    shutil.copy2(INPUT / "figures/s5_safe_recovery_timeline.pdf", stage / "figures/s5_safe_recovery_timeline.pdf")
    shutil.copy2(INPUT / "figures/s5_safe_recovery_timeline.png", stage / "figures/s5_safe_recovery_timeline.png")

    plot_s4(
        stage / "figures/s4_targeted_ablation",
        read_csv(stage / "s4_alg1_plot_source.csv"),
        read_csv(stage / "s4_alg2_plot_source.csv"),
        read_csv(stage / "s4_alg3_plot_source.csv"),
    )

    render_checks: dict[str, bool] = {}
    for stem in ("s4_targeted_ablation", "s5_safe_recovery_timeline"):
        with fitz.open(stage / "figures" / f"{stem}.pdf") as document:
            render_checks[f"{stem}_single_page_vector_pdf"] = len(document) == 1
        with Image.open(stage / "figures" / f"{stem}.png") as image:
            render_checks[f"{stem}_png_width_at_least_1800"] = image.width >= 1800

    frozen_audit_sha = core.sha256(INPUT / "audit.json")
    frozen_manifest_sha = core.sha256(INPUT / "manifest.json")
    checks = {
        "frozen_input_checksum_tree_pass": True,
        "frozen_input_audit_pass": json.loads((INPUT / "audit.json").read_text())["status"] == "PASS",
        "source_csvs_byte_identical": all(core.sha256(stage / name) == core.sha256(INPUT / name) for name in sources),
        "s5_figures_byte_identical": all(
            core.sha256(stage / "figures" / f"s5_safe_recovery_timeline.{suffix}")
            == core.sha256(INPUT / "figures" / f"s5_safe_recovery_timeline.{suffix}")
            for suffix in ("pdf", "png")
        ),
        **render_checks,
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "frozen_input": str(INPUT),
        "frozen_input_audit_sha256": frozen_audit_sha,
        "frozen_input_manifest_sha256": frozen_manifest_sha,
        "no_simulated_data_or_statistic_changes": True,
        "layout_change": "S4 Algorithm-2 benefit groups use compact group ticks; policy hatch and residency colors share one figure-level legend.",
        "source_sha256": {name: core.sha256(stage / name) for name in sources},
        "figure_sha256": {
            str(path.relative_to(stage)): core.sha256(path)
            for path in sorted((stage / "figures").iterdir())
        },
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": audit["status"],
        "paper_ready": False,
        "frozen_input": str(INPUT),
        "frozen_input_audit_sha256": frozen_audit_sha,
        "no_data_change": True,
        "invalidated_layout_predecessors": [
            {
                "path": "results/active_hbm_v313_s4_s5_targeted_v2_figures_v2",
                "reason": "Manual PDF inspection found the Algorithm-2 policy legend overlapping its panel title.",
            },
            {
                "path": "results/active_hbm_v313_s4_s5_targeted_v2_figures_v3",
                "reason": "Layout was clean, but automated PDF inspection found a 6.6pt legend below the preregistered 7pt minimum.",
            },
        ],
        "S4": {
            "pdf": "figures/s4_targeted_ablation.pdf",
            "png": "figures/s4_targeted_ablation.png",
            "sources": [name for name in sources if name.startswith("s4_")],
        },
        "S5": {
            "pdf": "figures/s5_safe_recovery_timeline.pdf",
            "png": "figures/s5_safe_recovery_timeline.png",
            "sources": [name for name in sources if name.startswith("s5_")],
        },
        "evidence_boundary": "Three-seed mechanism-targeted modeled DES; not paper-ready or physical A100 evidence.",
        "packager_sha256": core.sha256(Path(__file__)),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in artifacts.items())
    )
    os.replace(stage, output)
    print(json.dumps({
        "output": str(output),
        "status": audit["status"],
        "audit_sha256": core.sha256(output / "audit.json"),
        "manifest_sha256": core.sha256(output / "manifest.json"),
        "s4_pdf_sha256": core.sha256(output / "figures/s4_targeted_ablation.pdf"),
        "s4_png_sha256": core.sha256(output / "figures/s4_targeted_ablation.png"),
        "s5_pdf_sha256": core.sha256(output / "figures/s5_safe_recovery_timeline.pdf"),
        "s5_png_sha256": core.sha256(output / "figures/s5_safe_recovery_timeline.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
