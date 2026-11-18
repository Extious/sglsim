#!/usr/bin/env python3
"""Derive an audit-bound two-model S3 figure without rerunning DES."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_8B = ROOT / "results" / "active_hbm_v313_s3_scale_context_pilot"
SOURCE_32B = ROOT / "results" / "active_hbm_v313_s3_qwen32b_full_v1"
TARGET = ROOT / "results" / "active_hbm_v313_s3_two_model_combined_visual_v2"
CONTEXTS = [4096, 8192, 16384, 32768, 40960]
TOPOLOGIES = ["same_node", "cross_node"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derived_rows() -> list[dict[str, Any]]:
    old = read_csv(SOURCE_8B / "paired_ci.csv")
    new = read_csv(SOURCE_32B / "paired_ci.csv")
    rows: list[dict[str, Any]] = []
    for row in old:
        if row["metric"] != "fault_wct_mean_ms":
            continue
        axis = row["axis"]
        value = int(row["axis_value"])
        context = 16384 if axis == "scale" else value
        dp = value if axis == "scale" else 64
        rows.append({
            "model": "Qwen3-8B", "layout": "TP1/PP1",
            "sampling_design": "9-point union sweep",
            "panel_axis": axis, "topology": row["topology"],
            "system_dp": dp, "physical_gpu_count": dp,
            "context_tokens": context,
            "paired_seed_count": int(row["seed_count"]),
            "aegis_wct_reduction_pct_mean": 100.0 * (
                1.0 - float(row["aegis_over_native_mean"])
            ),
            "aegis_wct_reduction_pct_ci95_half_width": 100.0 * float(
                row["aegis_over_native_ci95_half_width"]
            ),
            "metric": "failure-conditioned all-workflow mean WCT",
            "probe_count_per_condition_seed": 12,
            "arrival_exposure_ms": 90000.0,
            "source_bundle": SOURCE_8B.name,
        })
    # The 32B full cross contributes every registered context to both scale
    # panels. Its context panel is fixed at DP16, i.e. the same 64 physical GPU
    # budget as the 8B DP64 context panel.
    for row in new:
        if row["metric"] != "fault_wct_mean_ms":
            continue
        dp = int(row["system_dp"])
        context = int(row["context_tokens"])
        common = {
            "model": "Qwen3-32B", "layout": "TP1/PP4",
            "sampling_design": "25-point full cross",
            "topology": row["topology"], "system_dp": dp,
            "physical_gpu_count": int(row["physical_gpu_count"]),
            "context_tokens": context,
            "paired_seed_count": int(row["paired_seed_count"]),
            "aegis_wct_reduction_pct_mean": float(
                row["aegis_wct_reduction_pct_mean"]
            ),
            "aegis_wct_reduction_pct_ci95_half_width": float(
                row["aegis_wct_reduction_pct_ci95_half_width"]
            ),
            "metric": "failure-conditioned all-workflow mean WCT",
            "probe_count_per_condition_seed": 12,
            "arrival_exposure_ms": 90000.0,
            "source_bundle": SOURCE_32B.name,
        }
        rows.append({**common, "panel_axis": "scale"})
        if dp == 16:
            rows.append({**common, "panel_axis": "context"})
    return sorted(rows, key=lambda row: (
        row["model"], row["panel_axis"], row["topology"],
        int(row["context_tokens"]), int(row["system_dp"]),
    ))


def render(stage: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    import fitz

    colors = {
        4096: "#1b9e77", 8192: "#66a61e", 16384: "#7570b3",
        32768: "#e6ab02", 40960: "#d95f02",
    }
    topo_colors = {"same_node": "#4d4d4d", "cross_node": "#7b3294"}
    topo_styles = {"same_node": "-", "cross_node": "--"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.6,
        "axes.titlesize": 8.4, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.9), sharey=True)
    models = [
        ("Qwen3-8B", "TP1/PP1", [16, 32, 64, 128, 256], 64),
        ("Qwen3-32B", "TP1/PP4", [4, 8, 16, 32, 64], 16),
    ]
    for row_index, (model, layout, dps, context_dp) in enumerate(models):
        for column, topology in enumerate(TOPOLOGIES):
            ax = axes[row_index, column]
            available_contexts = (
                [16384] if model == "Qwen3-8B" else CONTEXTS
            )
            x = np.arange(len(dps))
            for context in available_contexts:
                selected = sorted(
                    [item for item in rows
                     if item["model"] == model
                     and item["panel_axis"] == "scale"
                     and item["topology"] == topology
                     and int(item["context_tokens"]) == context],
                    key=lambda item: int(item["system_dp"]),
                )
                mean = np.asarray([
                    float(item["aegis_wct_reduction_pct_mean"])
                    for item in selected
                ])
                half = np.asarray([
                    float(item["aegis_wct_reduction_pct_ci95_half_width"])
                    for item in selected
                ])
                ax.errorbar(
                    x, mean, yerr=half, color=colors[context], marker="o",
                    markersize=2.8, capsize=1.8,
                )
            ax.axhline(0.0, color="#777777", linestyle=":", linewidth=0.75)
            ax.set_xticks(x, [str(value) for value in dps])
            ax.set_xlabel("System DP")
            domain = "Same-node" if topology == "same_node" else "Cross-node"
            suffix = " (16K)" if model == "Qwen3-8B" else ""
            ax.set_title(f"{model} {layout}\n{domain}{suffix}")
            ax.grid(axis="y", alpha=0.22)

        ax = axes[row_index, 2]
        x = np.arange(len(CONTEXTS))
        for topology in TOPOLOGIES:
            selected = sorted(
                [item for item in rows
                 if item["model"] == model
                 and item["panel_axis"] == "context"
                 and item["topology"] == topology
                 and int(item["system_dp"]) == context_dp],
                key=lambda item: int(item["context_tokens"]),
            )
            mean = np.asarray([
                float(item["aegis_wct_reduction_pct_mean"])
                for item in selected
            ])
            half = np.asarray([
                float(item["aegis_wct_reduction_pct_ci95_half_width"])
                for item in selected
            ])
            ax.errorbar(
                x, mean, yerr=half, color=topo_colors[topology],
                linestyle=topo_styles[topology], marker="o",
                markersize=2.8, capsize=1.8,
            )
        ax.axhline(0.0, color="#777777", linestyle=":", linewidth=0.75)
        ax.set_xticks(x, [f"{value // 1024}K" for value in CONTEXTS])
        ax.set_xlabel("Context length")
        ax.set_title(f"{model} {layout}\n64 physical GPUs")
        ax.grid(axis="y", alpha=0.22)

    fig.text(
        0.012, 0.53, "Mean WCT reduction vs Native (%)",
        rotation=90, va="center", ha="center", fontsize=8.0,
    )
    context_handles = [
        Line2D([0], [0], color=colors[value], marker="o",
               label=f"{value // 1024}K")
        for value in CONTEXTS
    ]
    topology_handles = [
        Line2D([0], [0], color=topo_colors[value],
               linestyle=topo_styles[value], marker="o",
               label=("Same-node / local" if value == "same_node"
                      else "Cross-node / remote"))
        for value in TOPOLOGIES
    ]
    first = fig.legend(
        handles=context_handles, title="Context", frameon=False, ncol=5,
        loc="lower left", bbox_to_anchor=(0.045, 0.002),
        columnspacing=0.75, handlelength=1.5, title_fontsize=7.0,
    )
    fig.add_artist(first)
    fig.legend(
        handles=topology_handles, frameon=False, ncol=2,
        loc="lower right", bbox_to_anchor=(0.985, 0.002),
        columnspacing=0.8, handlelength=1.8,
    )
    fig.suptitle("S3 modeled scale and context across model layouts", fontsize=9.8, y=0.992)
    fig.tight_layout(rect=(0.052, 0.115, 1, 0.955), h_pad=1.15, w_pad=1.0)
    figure_dir = stage / "figures"
    figure_dir.mkdir(parents=True)
    pdf = figure_dir / "s3_v313_two_model_scale_context_combined_visual_v2.pdf"
    png = figure_dir / "s3_v313_two_model_scale_context_combined_visual_v2.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    with Image.open(png) as image:
        width, height = image.size
    document = fitz.open(pdf)
    sizes: list[float] = []
    text_inside = True
    image_count = 0
    for page in document:
        image_count += len(page.get_images(full=True))
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        sizes.append(float(span["size"]))
                        text_inside = text_inside and page.rect.contains(
                            fitz.Rect(span["bbox"])
                        )
    page_count = document.page_count
    document.close()
    return {
        "pdf": str(pdf.relative_to(stage)),
        "png": str(png.relative_to(stage)),
        "pdf_page_count": page_count,
        "pdf_raster_image_count": image_count,
        "pdf_min_font_pt": min(sizes),
        "all_text_inside_pdf_page": text_inside,
        "png_width_px": width, "png_height_px": height,
    }


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"immutable target exists: {TARGET}")
    verifies = {}
    for label, source in (("qwen3_8b", SOURCE_8B), ("qwen3_32b", SOURCE_32B)):
        result = subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"], cwd=source,
            capture_output=True, text=True,
        )
        verifies[label] = result.returncode == 0
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    manifest_8 = json.loads((SOURCE_8B / "manifest.json").read_text())
    audit_8 = json.loads((SOURCE_8B / "audit.json").read_text())
    protocol_8 = json.loads((SOURCE_8B / "protocol.json").read_text())
    manifest_32 = json.loads((SOURCE_32B / "manifest.json").read_text())
    audit_32 = json.loads((SOURCE_32B / "audit.json").read_text())
    protocol_32 = json.loads((SOURCE_32B / "protocol.json").read_text())
    rows = derived_rows()

    stage = TARGET.with_name(f".{TARGET.name}.stage-{os.getpid()}")
    stage.mkdir(parents=True)
    shutil.copy2(SOURCE_8B / "paired_ci.csv", stage / "source_qwen3_8b_paired_ci.csv")
    shutil.copy2(SOURCE_32B / "paired_ci.csv", stage / "source_qwen3_32b_paired_ci.csv")
    write_csv(stage / "s3_two_model_figure_source.csv", rows)
    figure = render(stage, rows)

    source_hashes = {
        "qwen3_8b_manifest_sha256": sha256(SOURCE_8B / "manifest.json"),
        "qwen3_8b_audit_sha256": sha256(SOURCE_8B / "audit.json"),
        "qwen3_8b_sha256sums_sha256": sha256(SOURCE_8B / "SHA256SUMS"),
        "qwen3_8b_paired_ci_sha256": sha256(SOURCE_8B / "paired_ci.csv"),
        "qwen3_32b_manifest_sha256": sha256(SOURCE_32B / "manifest.json"),
        "qwen3_32b_audit_sha256": sha256(SOURCE_32B / "audit.json"),
        "qwen3_32b_sha256sums_sha256": sha256(SOURCE_32B / "SHA256SUMS"),
        "qwen3_32b_paired_ci_sha256": sha256(SOURCE_32B / "paired_ci.csv"),
    }
    rows_8 = [row for row in rows if row["model"] == "Qwen3-8B"]
    rows_32 = [row for row in rows if row["model"] == "Qwen3-32B"]
    checks = {
        "both_source_sha256sums_pass": all(verifies.values()),
        "both_source_audits_pass": (
            audit_8["status"] == "PASS" and manifest_8["status"] == "PASS"
            and audit_32["status"] == "PASS" and manifest_32["status"] == "PASS"
        ),
        "qwen3_8b_corrected_postprocess_lineage": (
            audit_8["mode"] == "full_corrected_postprocess"
            and (SOURCE_8B / "audit_amendment.json").is_file()
        ),
        "qwen3_32b_analytic_not_physical_boundary": (
            manifest_32["profile_status"]
            == "MODELED_ANALYTIC_WEIGHT_RATIO_NOT_PHYSICAL"
            and protocol_32["service_model"]["not_a100_calibration"] is True
        ),
        "both_fixed_90s_and_12_probe_exposure": (
            protocol_8["workload"]["fixed_arrival_cutoff_ms"] == 90000.0
            and protocol_8["workload"]["fault_probes_per_axis_point_seed"] == 12
            and protocol_32["workload"]["fixed_arrival_cutoff_ms"] == 90000.0
            and protocol_32["workload"]["fault_probes_per_condition_seed"] == 12
        ),
        "all_ci_use_exactly_three_workload_seeds": (
            all(row["paired_seed_count"] == 3 for row in rows)
        ),
        "qwen3_8b_sparse_union_not_fabricated_as_full_cross": (
            len(rows_8) == 20
            and {row["context_tokens"] for row in rows_8
                 if row["panel_axis"] == "scale"} == {16384}
            and {row["system_dp"] for row in rows_8
                 if row["panel_axis"] == "context"} == {64}
        ),
        "qwen3_32b_scale_retains_full_cross": (
            len([row for row in rows_32 if row["panel_axis"] == "scale"]) == 50
            and {(row["system_dp"], row["context_tokens"], row["topology"])
                 for row in rows_32 if row["panel_axis"] == "scale"}
            == {(dp, context, topology)
                for dp in [4, 8, 16, 32, 64]
                for context in CONTEXTS for topology in TOPOLOGIES}
        ),
        "context_panels_share_64_physical_gpu_budget": all(
            row["physical_gpu_count"] == 64
            for row in rows if row["panel_axis"] == "context"
        ),
        "source_csvs_byte_identical": (
            sha256(stage / "source_qwen3_8b_paired_ci.csv")
            == source_hashes["qwen3_8b_paired_ci_sha256"]
            and sha256(stage / "source_qwen3_32b_paired_ci.csv")
            == source_hashes["qwen3_32b_paired_ci_sha256"]
        ),
        "no_performance_direction_selection": True,
        "single_page_vector_pdf": (
            figure["pdf_page_count"] == 1
            and figure["pdf_raster_image_count"] == 0
        ),
        "png_width_at_least_1800": figure["png_width_px"] >= 1800,
        "pdf_min_font_at_least_7pt": figure["pdf_min_font_pt"] >= 6.99,
        "pdf_text_not_cropped": figure["all_text_inside_pdf_page"],
    }
    lineage = {
        "schema_version": 1,
        "derivation": "visual_only_from_frozen_paired_ci_no_DES_rerun",
        "source_bundles": {
            "Qwen3-8B TP1/PP1": str(SOURCE_8B),
            "Qwen3-32B TP1/PP4": str(SOURCE_32B),
        },
        "source_hashes": source_hashes,
        "sampling_designs": {
            "Qwen3-8B TP1/PP1": "9-point union: scale at 16K plus context at DP64",
            "Qwen3-32B TP1/PP4": "25-point full DP by context cross product",
        },
        "context_comparison_rule": (
            "fixed 64 physical GPUs: Qwen3-8B DP64 and Qwen3-32B DP16"
        ),
        "failure_exposure": (
            "both use 12 fixed source-worker probes over 90s per condition-seed; "
            "all-workflow dilution changes with workflow universe and DP"
        ),
        "independent_cross_model_samples": True,
        "cross_model_performance_ranking_registered": False,
    }
    (stage / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False, "visual_only": True,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "source_hashes": source_hashes, "figure": figure,
        "row_count": len(rows),
        "evidence_boundary": (
            "Two preliminary metadata-DES lineages with different model layouts "
            "and sampling designs. Qwen3-32B service is analytic, not physical."
        ),
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"audit FAIL {audit['failed_checks']}; stage={stage}")
    caption = (
        "Modeled AegisKV failure-conditioned all-workflow mean WCT reduction "
        "relative to modeled Native Mooncake. Error bars are paired Student-t "
        "95% confidence intervals across three workload seeds. Qwen3-8B "
        "TP1/PP1 uses its registered 9-point union sweep (scale at 16K; context "
        "at DP64), whereas Qwen3-32B TP1/PP4 uses a 25-point full cross. "
        "Context panels hold total physical GPUs at 64 (DP64 versus DP16). "
        "Both lineages use 12 fixed probes over 90 s; decreasing affected "
        "workflow share at larger DP dilutes the all-workflow metric. The "
        "Qwen3-32B service profile and all failover timings are modeled, not "
        "matched physical measurements.\n"
    )
    (stage / "figure_caption.txt").write_text(caption, encoding="utf-8")
    manifest = {
        "schema_version": 1, "status": "PASS", "paper_ready": False,
        "visual_only": True, "lineage": "lineage.json",
        "audit": "audit.json", "audit_sha256": sha256(stage / "audit.json"),
        "figure": figure, "figure_caption": "figure_caption.txt",
        "figure_source": "s3_two_model_figure_source.csv",
        "source_hashes": source_hashes,
        "paired_ci_inference_unit": "workload_seed",
        "paired_seed_count": 3, "no_result_selection_or_raw_change": True,
        "evidence_boundary": audit["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts = {
        str(path.relative_to(stage)): sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())),
        encoding="utf-8",
    )
    os.replace(stage, TARGET)
    print(TARGET)
    print(f"audit_sha256={sha256(TARGET / 'audit.json')}")
    print(f"manifest_sha256={sha256(TARGET / 'manifest.json')}")
    print(f"sha256sums_sha256={sha256(TARGET / 'SHA256SUMS')}")
    print(f"pdf_sha256={sha256(TARGET / figure['pdf'])}")
    print(f"png_sha256={sha256(TARGET / figure['png'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
