#!/usr/bin/env python3
"""Package frozen S4 PASS and S5 FAIL-diagnostic figures without data changes."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import run_active_hbm_v313_s4_s5_pilot as runner
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results/active_hbm_v313_s4_s5_preliminary_pilot"
OUTPUT_NAME = "active_hbm_v313_s4_s5_audit_bound_figures_v2"


def verify_checksum_tree(path: Path) -> bool:
    for line in (path / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if core.sha256(path / name) != digest:
            return False
    return True


def plot_s5(path: Path, rows: list[dict[str, str]]) -> None:
    runner.style()
    t = np.asarray([float(row["time_s"]) for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.15), sharex=True)
    spans = [(0, 60, "#2ca02c"), (60, 120, "#d62728"), (120, 240, "#2ca02c")]
    for ax in axes:
        for left, right, color in spans:
            ax.axvspan(left, right, color=color, alpha=0.07)
        ax.grid(axis="y", alpha=0.25)

    alpha = np.asarray([float(row["alpha_mean"]) for row in rows])
    alpha_ci = np.asarray([float(row["alpha_ci95_half_width"]) for row in rows])
    axes[0].plot(t, alpha, color="#7b3294", marker="o", label=r"$\alpha$")
    axes[0].fill_between(t, alpha-alpha_ci, alpha+alpha_ci, color="#7b3294", alpha=0.16)
    axes[0].set_ylabel(r"Peer-HBM $\alpha$")
    axes[0].set_ylim(bottom=0)
    for x, label in ((30, "Low"), (90, "High"), (180, "Recovery")):
        axes[0].text(
            x, 0.96, label, transform=axes[0].get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, color="#444444",
        )
    quota_ax = axes[0].twinx()
    quota_ax.plot(t, 40.0 * alpha, color="#c51b7d", linestyle="--", linewidth=1.1)
    quota_ax.set_ylabel(r"Quota $40\alpha$ (GiB/target)")
    quota_ax.spines["top"].set_visible(False)

    offered = np.asarray([float(row["offered_active_requests_per_gpu_mean"]) for row in rows])
    realized = np.asarray([float(row["realized_active_requests_per_gpu_mean"]) for row in rows])
    realized_ci = np.asarray([float(row["realized_active_requests_per_gpu_ci95_half_width"]) for row in rows])
    axes[1].step(t, offered, where="post", color="#555555", linestyle="--", label="Offered load")
    axes[1].plot(t, realized, color="#ff7f0e", marker="o", label="Realized active")
    axes[1].fill_between(t, realized-realized_ci, realized+realized_ci, color="#ff7f0e", alpha=0.16)
    axes[1].set_ylabel("Requests/GPU")
    pressure_ax = axes[1].twinx()
    pressure = np.asarray([float(row["pressure_fraction_mean"]) for row in rows])
    pressure_ax.plot(t, pressure, color="#d62728", marker="s", linewidth=1.1, label="Pressure")
    pressure_ax.set_ylabel("Pressure fraction")
    pressure_ax.set_ylim(-0.03, 1.03)
    pressure_ax.spines["top"].set_visible(False)
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = pressure_ax.get_legend_handles_labels()
    axes[1].legend(
        h1+h2, l1+l2, frameon=False, ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
    )

    occupancy = [
        ("serving_hbm_gib_per_gpu", "Serving HBM", "#1f77b4"),
        ("peer_hbm_backup_gib_per_gpu", "Peer-HBM backup", "#2ca02c"),
        ("dram_gib_per_gpu", "DRAM", "#9467bd"),
    ]
    for field, label, color in occupancy:
        mean = np.asarray([float(row[f"{field}_mean"]) for row in rows])
        half = np.asarray([float(row[f"{field}_ci95_half_width"]) for row in rows])
        axes[2].plot(t, mean, marker="o", color=color, label=label)
        axes[2].fill_between(t, mean-half, mean+half, color=color, alpha=0.13)
    axes[2].set_ylabel("Occupancy (GiB/GPU)")
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].legend(
        frameon=False, ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
    )
    axes[2].set_xlim(0, 240)
    fig.suptitle("S5 AIMD and physical occupancy timeline (3 seeds)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97), h_pad=1.3)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not verify_checksum_tree(INPUT):
        raise RuntimeError("frozen input checksum failed")
    input_audit = json.loads((INPUT / "audit.json").read_text())
    output = ROOT / "results" / OUTPUT_NAME
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage = ROOT / "results" / f".{OUTPUT_NAME}.stage-{os.getpid()}"
    (stage / "figures").mkdir(parents=True)
    for name in ("s4_plot_source.csv", "s5_plot_source.csv", "s4_seed_metrics.csv", "s5_control_timeline_seed.csv"):
        shutil.copy2(INPUT / name, stage / name)
    s4_rows = list(csv.DictReader((stage / "s4_plot_source.csv").open()))
    s5_rows = list(csv.DictReader((stage / "s5_plot_source.csv").open()))
    runner.plot_s4(stage / "figures/s4_algorithm_ablation", s4_rows)
    plot_s5(stage / "figures/s5_aimd_occupancy_timeline", s5_rows)

    render_checks = {}
    for stem in ("s4_algorithm_ablation", "s5_aimd_occupancy_timeline"):
        pdf = stage / "figures" / f"{stem}.pdf"
        png = stage / "figures" / f"{stem}.png"
        with fitz.open(pdf) as doc:
            render_checks[f"{stem}_vector_pdf_single_page"] = len(doc) == 1
        with Image.open(png) as image:
            render_checks[f"{stem}_png_width_at_least_1800"] = image.width >= 1800

    common_checks = input_audit["checks"]["lineage_and_tests"]
    s4_checks = {
        **input_audit["checks"]["S4"],
        "vector_pdf_single_page": render_checks["s4_algorithm_ablation_vector_pdf_single_page"],
        "png_width_at_least_1800": render_checks["s4_algorithm_ablation_png_width_at_least_1800"],
        "frozen_input_checksum_pass": True,
        "lineage_and_tests_pass": all(common_checks.values()),
    }
    s5_checks = {
        **input_audit["checks"]["S5"],
        "vector_pdf_single_page": render_checks["s5_aimd_occupancy_timeline_vector_pdf_single_page"],
        "png_width_at_least_1800": render_checks["s5_aimd_occupancy_timeline_png_width_at_least_1800"],
        "frozen_input_checksum_pass": True,
        "lineage_and_tests_pass": all(common_checks.values()),
    }
    s4_status = "PASS" if all(s4_checks.values()) else "FAIL"
    s5_status = "PASS" if all(s5_checks.values()) else "FAIL_DIAGNOSTIC"
    input_audit_sha = core.sha256(INPUT / "audit.json")
    s4_audit = {
        "schema_version": 1, "status": s4_status, "paper_ready": False,
        "checks": s4_checks,
        "failed_checks": [key for key, passed in s4_checks.items() if not passed],
        "frozen_input": str(INPUT), "frozen_input_audit_sha256": input_audit_sha,
        "source_csv_sha256": core.sha256(stage / "s4_plot_source.csv"),
        "pdf_sha256": core.sha256(stage / "figures/s4_algorithm_ablation.pdf"),
        "png_sha256": core.sha256(stage / "figures/s4_algorithm_ablation.png"),
    }
    s5_audit = {
        "schema_version": 1, "status": s5_status, "paper_ready": False,
        "checks": s5_checks,
        "failed_checks": [key for key, passed in s5_checks.items() if not passed],
        "diagnosis": (
            "The frozen low-high-recovery schedule drove alpha down in every seed, "
            "but residual per-target pressure prevented a recovery-phase mean-alpha "
            "increase in all three seeds (seed 2026082602 did not recover). This is "
            "retained as a negative mechanism result; no schedule or seed was changed."
        ),
        "frozen_input": str(INPUT), "frozen_input_audit_sha256": input_audit_sha,
        "source_csv_sha256": core.sha256(stage / "s5_plot_source.csv"),
        "pdf_sha256": core.sha256(stage / "figures/s5_aimd_occupancy_timeline.pdf"),
        "png_sha256": core.sha256(stage / "figures/s5_aimd_occupancy_timeline.png"),
    }
    (stage / "s4_audit.json").write_text(json.dumps(s4_audit, indent=2, sort_keys=True) + "\n")
    (stage / "s5_audit.json").write_text(json.dumps(s5_audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "S4_PASS_S5_FAIL_DIAGNOSTIC",
        "paper_ready": False,
        "frozen_input": str(INPUT),
        "frozen_input_audit_sha256": input_audit_sha,
        "no_data_or_schedule_change": True,
        "S4": {
            "status": s4_status, "pdf": "figures/s4_algorithm_ablation.pdf",
            "png": "figures/s4_algorithm_ablation.png", "source": "s4_plot_source.csv",
            "audit": "s4_audit.json",
        },
        "S5": {
            "status": s5_status, "pdf": "figures/s5_aimd_occupancy_timeline.pdf",
            "png": "figures/s5_aimd_occupancy_timeline.png", "source": "s5_plot_source.csv",
            "audit": "s5_audit.json",
        },
        "evidence_boundary": "Three-seed preliminary modeled DES; not paper-ready or physical A100 evidence.",
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
        "output": str(output), "status": manifest["status"],
        "s4_audit_sha256": core.sha256(output / "s4_audit.json"),
        "s5_audit_sha256": core.sha256(output / "s5_audit.json"),
        "manifest_sha256": core.sha256(output / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
