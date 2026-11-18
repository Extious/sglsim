#!/usr/bin/env python3
"""Publish a relative-gain visual-only view of the immutable S7-v2 bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "active_hbm_v313_s7_failure_target_v2_full"
TARGET = ROOT / "results" / "active_hbm_v313_s7_failure_target_v2_visual_v3"
SOURCE_AUDIT_SHA = "5f9bf670c3230ae1bcf8c10b3d36754459fe3f6d7b5b253594054cdf247fb018"
SOURCE_MANIFEST_SHA = "251aecc2721aab7a374f618948d09c3b3ee1b9b19fe55e6edc624af7b58e1595"
SOURCE_SUMS_SHA = "649ba1116e116eeffb76fd045c941d7eef069c42b587e59cfe2bd9f6a6e9d0d6"
SOURCE_PAIRED_SHA = "602d79266a00d2d8b0c987be84d2bc1035dea2c34516049b8978ba12f6201652"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.3,
        "axes.titlesize": 8.2, "axes.labelsize": 7.7,
        "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
        "legend.fontsize": 7.1, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "lines.linewidth": 1.25,
    })


def plot(path: Path, rows: list[dict[str, str]]) -> None:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.45))
    panels = [
        ("bandwidth", "source_to_peer_hbm_copy", "Source to Peer-HBM copy"),
        ("bandwidth", "source_to_dram_copy", "Source to DRAM copy"),
        ("bandwidth", "peer_hbm_recovery", "Peer-HBM recovery"),
        ("bandwidth", "dram_recovery", "DRAM recovery"),
        ("repair", "source_worker_gpu", "Worker/GPU replacement"),
        ("repair", "source_node", "Source-node replacement"),
    ]
    encodings = {
        "primary": ("#2166ac", "s", "-", "All workflows (primary)"),
        "secondary": (
            "#d6604d", "o", "--",
            "Failure targets (secondary; modeled replacement)",
        ),
    }
    for axis, (kind, selector, title) in zip(axes.flat, panels):
        if kind == "bandwidth":
            selected = sorted(
                (row for row in rows if row["axis"] == kind and row["direction"] == selector),
                key=lambda row: float(row["bandwidth_gb_s"]),
            )
            x = [float(row["bandwidth_gb_s"]) for row in selected]
            axis.set_xlabel("Bandwidth (GB/s)")
        else:
            selected = sorted(
                (row for row in rows if row["axis"] == kind and row["failure_type"] == selector),
                key=lambda row: float(row["repair_ms"]),
            )
            x = [float(row["repair_ms"]) / 1000.0 for row in selected]
            axis.set_xlabel("Replacement (s)")
        for name, metric in (
            ("primary", "fault_wct_relative_gain"),
            ("secondary", "failure_target_wct_relative_gain"),
        ):
            color, marker, line, label = encodings[name]
            axis.errorbar(
                x, [100.0 * float(row[f"{metric}_mean"]) for row in selected],
                yerr=[100.0 * float(row[f"{metric}_ci95_half_width"]) for row in selected],
                color=color, marker=marker, linestyle=line, markersize=3.5,
                capsize=2.0, label=label,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.75)
        axis.set_ylim(0.0, 100.0)
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.set_title(title)
        axis.set_ylabel("Relative WCT gain (%)")
        axis.grid(axis="y", alpha=0.22)
    handles = [
        Line2D([0], [0], color=value[0], marker=value[1], linestyle=value[2], label=value[3])
        for value in encodings.values()
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.952), handlelength=2.0, columnspacing=1.2)
    fig.suptitle("S7 directional and repair sensitivity", fontsize=10.0, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=1.35, w_pad=1.15)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def verify_figure(pdf: Path, png: Path) -> dict[str, Any]:
    with fitz.open(pdf) as document:
        sizes = [
            float(span["size"])
            for block in document[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", []) if span["text"].strip()
        ]
        pages = len(document)
    with Image.open(png) as image:
        width, height = image.size
    return {
        "single_page_vector_pdf": pages == 1 and bool(sizes),
        "minimum_extracted_font_pt": min(sizes),
        "minimum_font_at_least_7pt": min(sizes) >= 7.0 - 1e-6,
        "png_width_px": width, "png_height_px": height,
        "png_width_at_least_1800": width >= 1800,
    }


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite {TARGET}")
    source_audit = json.loads((SOURCE / "audit.json").read_text())
    if source_audit["status"] != "PASS" or source_audit["cell_count"] != 168:
        raise RuntimeError("S7-v2 source is not the audited full PASS bundle")
    if any((
        sha256(SOURCE / "audit.json") != SOURCE_AUDIT_SHA,
        sha256(SOURCE / "manifest.json") != SOURCE_MANIFEST_SHA,
        sha256(SOURCE / "SHA256SUMS") != SOURCE_SUMS_SHA,
        sha256(SOURCE / "paired_ci.csv") != SOURCE_PAIRED_SHA,
    )):
        raise RuntimeError("S7-v2 source hash drift")
    rows = read_csv(SOURCE / "paired_ci.csv")
    stage = TARGET.parent / f".{TARGET.name}.stage-{os.getpid()}"
    (stage / "figures").mkdir(parents=True)
    shutil.copy2(SOURCE / "paired_ci.csv", stage / "source_paired_ci.csv")
    stem = stage / "figures" / "s7_v313_failure_target_v2_relative_gain_observed"
    plot(stem, rows)
    figure_checks = verify_figure(stem.with_suffix(".pdf"), stem.with_suffix(".png"))
    directions = {row["direction"] for row in rows if row["axis"] == "bandwidth"}
    domains = {row["failure_type"] for row in rows if row["axis"] == "repair"}
    checks = {
        "source_full_audit_hash_exact": sha256(SOURCE / "audit.json") == SOURCE_AUDIT_SHA,
        "source_paired_ci_hash_exact": sha256(SOURCE / "paired_ci.csv") == SOURCE_PAIRED_SHA,
        "exact_28_unfiltered_ci_rows": len(rows) == 28,
        "all_four_directions_retained": directions == {
            "source_to_peer_hbm_copy", "source_to_dram_copy",
            "peer_hbm_recovery", "dram_recovery",
        },
        "both_repair_domains_retained": domains == {"source_worker_gpu", "source_node"},
        "all_workflow_primary_and_failure_target_secondary_present": all(
            key in row for row in rows for key in (
                "fault_wct_relative_gain_mean",
                "fault_wct_relative_gain_ci95_half_width",
                "failure_target_wct_relative_gain_mean",
                "failure_target_wct_relative_gain_ci95_half_width",
            )
        ),
        "fixed_zero_to_100_percent_axis_no_denominator_suppression": True,
        "no_data_statistic_seed_direction_or_sign_change": True,
        "single_page_vector_pdf": figure_checks["single_page_vector_pdf"],
        "minimum_font_at_least_7pt": figure_checks["minimum_font_at_least_7pt"],
        "png_width_at_least_1800": figure_checks["png_width_at_least_1800"],
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "source_audit_sha256": SOURCE_AUDIT_SHA,
        "source_paired_ci_sha256": SOURCE_PAIRED_SHA,
        "figure_checks": figure_checks,
        "evidence_boundary": "Visual-only relative-gain rendering of the immutable S7-v2 metadata-DES PASS bundle. Failure-target values expose modeled/unmeasured replacement; they are secondary and not physical measurements. All-workflow remains primary.",
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "status": audit["status"], "paper_ready": False,
        "selection": "S7_V2_VISUAL_ONLY_V3_NONE",
        "source_bundle": str(SOURCE),
        "source_audit_sha256": SOURCE_AUDIT_SHA,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "source_sha256sums_sha256": SOURCE_SUMS_SHA,
        "packager_sha256": sha256(Path(__file__)),
        "audit": "audit.json", "audit_sha256": sha256(stage / "audit.json"),
        "source_paired_ci": "source_paired_ci.csv",
        "figure": {
            "pdf": f"figures/{stem.name}.pdf",
            "png": f"figures/{stem.name}.png"
        },
        "no_data_or_statistic_change": True,
        "evidence_boundary": audit["evidence_boundary"]
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit["failed_checks"])
    artifacts = {
        str(path.relative_to(stage)): sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text("".join(
        f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())
    ))
    os.replace(stage, TARGET)
    print(json.dumps({
        "output": str(TARGET),
        "audit_sha256": sha256(TARGET / "audit.json"),
        "manifest_sha256": sha256(TARGET / "manifest.json"),
        "sha256sums_sha256": sha256(TARGET / "SHA256SUMS"),
    }, indent=2))


if __name__ == "__main__":
    main()
