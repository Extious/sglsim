#!/usr/bin/env python3
"""Render an audit-bound S1 visual-only v2 from the frozen ten-seed bundle."""

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
SOURCE = ROOT / "results" / "active_hbm_v313_s1_six_architecture_10seed_v1"
TARGET = ROOT / "results" / "active_hbm_v313_s1_six_architecture_10seed_visual_v2"
NAMES = [
    "Native Mooncake", "Static", "DejaVu", "KevlarFlow",
    "GhostServe", "AegisKV",
]
PAIRED_NAMES = NAMES[1:]
DOMAINS = ["source_worker_gpu", "source_node"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render(
    stage: Path,
    cdf_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    import fitz

    colors = {
        "Native Mooncake": "#222222", "Static": "#8c564b",
        "DejaVu": "#1f77b4", "KevlarFlow": "#9467bd",
        "GhostServe": "#2ca02c", "AegisKV": "#d62728",
    }
    styles = {
        "Native Mooncake": "-", "Static": "--", "DejaVu": "-.",
        "KevlarFlow": ":", "GhostServe": (0, (5, 2)),
        "AegisKV": "-",
    }
    domain_titles = {
        "source_worker_gpu": "Worker / GPU failure",
        "source_node": "Source-node failure",
    }
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 5.25),
        gridspec_kw={"height_ratios": [1.0, 1.08]},
    )

    # Absolute pooled ECDFs are plotted at their frozen x/y coordinates.  There
    # is deliberately no jitter or architecture-specific horizontal offset.
    for column, domain in enumerate(DOMAINS):
        axis = axes[0, column]
        for name in NAMES:
            rows = [
                row for row in cdf_rows
                if row["display_policy"] == name
                and row["failure_domain"] == domain
            ]
            axis.plot(
                np.asarray([float(row["wct_ms"]) for row in rows]) / 1000.0,
                [float(row["ecdf"]) for row in rows],
                color=colors[name], linestyle=styles[name],
                linewidth=1.35 if name != "AegisKV" else 1.7,
                label=name,
            )
        axis.set_title(domain_titles[domain])
        axis.set_xlabel("Workflow completion time (s)")
        axis.set_ylim(0.0, 1.01)
        axis.grid(True, alpha=0.22, linewidth=0.6)
    axes[0, 0].set_ylabel("Empirical CDF")
    axes[0, 1].tick_params(labelleft=False)

    metric_styles = {
        "mean": {"label": "Mean WCT", "marker": "o", "offset": -0.13},
        "p99": {"label": "p99 WCT", "marker": "s", "offset": 0.13},
    }
    y_base = np.arange(len(PAIRED_NAMES), dtype=float)
    for column, domain in enumerate(DOMAINS):
        axis = axes[1, column]
        rows_by_name = {
            row["display_policy"]: row for row in paired_rows
            if row["failure_domain"] == domain
        }
        for metric, style in metric_styles.items():
            for index, name in enumerate(PAIRED_NAMES):
                row = rows_by_name[name]
                center = float(row[f"paired_{metric}_gain_pct_mean"])
                low = float(row[f"paired_{metric}_gain_pct_ci95_low"])
                high = float(row[f"paired_{metric}_gain_pct_ci95_high"])
                axis.errorbar(
                    center, y_base[index] + style["offset"],
                    xerr=np.asarray([[center - low], [high - center]]),
                    fmt=style["marker"], color=colors[name],
                    markerfacecolor=(colors[name] if metric == "mean" else "white"),
                    markeredgewidth=0.9, markersize=4.2,
                    elinewidth=1.0, capsize=2.2, capthick=0.9,
                    zorder=3,
                )
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8,
                     zorder=0)
        axis.set_title(f"{domain_titles[domain]}: paired gain")
        axis.set_xlabel("WCT reduction vs Native Mooncake (%)")
        axis.set_yticks(y_base)
        axis.set_yticklabels(PAIRED_NAMES if column == 0 else [])
        axis.set_ylim(len(PAIRED_NAMES) - 0.48, -0.52)
        axis.grid(True, axis="x", alpha=0.22, linewidth=0.6)
        if domain == "source_worker_gpu":
            axis.set_xlim(-0.55, 10.75)
        else:
            axis.set_xlim(-0.115, 0.55)
    axes[1, 0].set_ylabel("Architecture")

    architecture_handles = [
        Line2D([0], [0], color=colors[name], linestyle=styles[name],
               linewidth=1.6, label=name)
        for name in NAMES
    ]
    fig.legend(
        handles=architecture_handles, loc="upper center", ncol=6,
        frameon=False, bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.85, handlelength=2.0,
    )
    metric_handles = [
        Line2D([0], [0], color="#333333", marker=style["marker"],
               markerfacecolor=("#333333" if metric == "mean" else "white"),
               linestyle="none", markersize=4.5, label=style["label"])
        for metric, style in metric_styles.items()
    ]
    fig.legend(
        handles=metric_handles, loc="center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.493), columnspacing=1.4, handletextpad=0.35,
    )
    fig.subplots_adjust(
        left=0.135, right=0.985, bottom=0.105, top=0.905,
        wspace=0.20, hspace=0.66,
    )

    figure_dir = stage / "figures"
    figure_dir.mkdir(parents=True)
    pdf = figure_dir / "s1_v313_six_architecture_10seed_observed_visual_v2.pdf"
    png = figure_dir / "s1_v313_six_architecture_10seed_observed_visual_v2.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    width, height = Image.open(png).size
    document = fitz.open(pdf)
    font_sizes: list[float] = []
    text_inside_page = True
    raster_image_count = 0
    for page in document:
        raster_image_count += len(page.get_images(full=True))
        page_rect = page.rect
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        font_sizes.append(float(span["size"]))
                        span_rect = fitz.Rect(span["bbox"])
                        text_inside_page = text_inside_page and page_rect.contains(
                            span_rect
                        )
    page_count = document.page_count
    document.close()
    return {
        "pdf": str(pdf.relative_to(stage)),
        "png": str(png.relative_to(stage)),
        "png_width_px": width,
        "png_height_px": height,
        "pdf_page_count": page_count,
        "pdf_min_font_pt": min(font_sizes),
        "pdf_raster_image_count": raster_image_count,
        "all_text_inside_pdf_page": text_inside_page,
        "cdf_horizontal_offset_applied": False,
    }


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"immutable target exists: {TARGET}")
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    source_verify = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=SOURCE,
        capture_output=True, text=True,
    )
    if source_verify.returncode != 0:
        raise RuntimeError(source_verify.stdout + source_verify.stderr)

    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    source_audit = json.loads((SOURCE / "audit.json").read_text())
    cdf_rows = load_csv(SOURCE / "cdf_render_source.csv")
    paired_rows = load_csv(SOURCE / "paired_vs_native_ci.csv")

    stage = TARGET.with_name(f".{TARGET.name}.stage-{os.getpid()}")
    stage.mkdir(parents=True)
    shutil.copy2(SOURCE / "cdf_render_source.csv", stage / "cdf_render_source.csv")
    shutil.copy2(
        SOURCE / "paired_vs_native_ci.csv", stage / "paired_vs_native_ci.csv"
    )
    figure = render(stage, cdf_rows, paired_rows)

    cdf_groups = {
        (row["display_policy"], row["failure_domain"]) for row in cdf_rows
    }
    paired_groups = {
        (row["display_policy"], row["failure_domain"]) for row in paired_rows
    }
    paired_counts = {int(row["paired_seed_count"]) for row in paired_rows}
    source_hashes = {
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "source_audit_sha256": sha256(SOURCE / "audit.json"),
        "source_sha256sums_sha256": sha256(SOURCE / "SHA256SUMS"),
        "cdf_render_source_sha256": sha256(SOURCE / "cdf_render_source.csv"),
        "paired_vs_native_ci_sha256": sha256(
            SOURCE / "paired_vs_native_ci.csv"
        ),
    }
    checks = {
        "source_sha256sums_all_pass": source_verify.returncode == 0,
        "source_bundle_status_pass": (
            source_manifest["status"] == "PASS"
            and source_audit["status"] == "PASS"
        ),
        "source_is_ten_seed_full_lineage": (
            source_manifest["mode"] == "full"
            and source_manifest["seed_count"] == 10
            and source_manifest["old_single_seed_lineage"]
            == "INVALIDATED_ORACLE_SLACK_DIAGNOSTIC_ONLY"
        ),
        "exact_six_cdf_architectures_two_domains": cdf_groups == {
            (name, domain) for name in NAMES for domain in DOMAINS
        },
        "exact_five_paired_architectures_two_domains": paired_groups == {
            (name, domain) for name in PAIRED_NAMES for domain in DOMAINS
        },
        "paired_ci_inference_unit_exactly_ten_workload_seeds": (
            paired_counts == {10}
            and source_manifest["paired_ci_inference_unit"] == "workload_seed"
        ),
        "mean_and_p99_paired_ci_present": all(
            all(row.get(field, "") != "" for field in (
                "paired_mean_gain_pct_mean", "paired_mean_gain_pct_ci95_low",
                "paired_mean_gain_pct_ci95_high", "paired_p99_gain_pct_mean",
                "paired_p99_gain_pct_ci95_low", "paired_p99_gain_pct_ci95_high",
            ))
            for row in paired_rows
        ),
        "frozen_cdf_source_byte_identical": sha256(
            stage / "cdf_render_source.csv"
        ) == source_hashes["cdf_render_source_sha256"],
        "frozen_paired_ci_source_byte_identical": sha256(
            stage / "paired_vs_native_ci.csv"
        ) == source_hashes["paired_vs_native_ci_sha256"],
        "no_cdf_horizontal_offset": not figure["cdf_horizontal_offset_applied"],
        "single_page_vector_pdf": (
            figure["pdf_page_count"] == 1
            and figure["pdf_raster_image_count"] == 0
        ),
        "png_width_at_least_1800": figure["png_width_px"] >= 1800,
        "pdf_min_font_at_least_7pt": figure["pdf_min_font_pt"] >= 6.99,
        "pdf_text_not_cropped": figure["all_text_inside_pdf_page"],
        "no_performance_direction_or_seed_selection": source_manifest[
            "no_result_seed_architecture_or_domain_selection"
        ],
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "visual_only": True,
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "source_bundle": str(SOURCE),
        "source_hashes": source_hashes,
        "figure": figure,
        "inference_unit": "10 paired workload seeds",
        "pooled_ecdf_role": "descriptive",
        "evidence_boundary": source_manifest["evidence_boundary"],
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"audit FAIL {audit['failed_checks']}; stage={stage}")

    caption = (
        "Failure-conditioned workflow completion time for six modeled "
        "architectures. Top: absolute pooled ECDFs over all retained "
        "probe-workflow observations; exact overlap is left unjittered. "
        "Bottom: paired mean and p99 WCT reduction relative to Native "
        "Mooncake, with Student-t 95% confidence intervals across 10 "
        "workload seeds. Worker-domain adapters cluster near 6% mean "
        "reduction; AegisKV exceeds GhostServe by only about 0.013 percentage "
        "points. Under source-node failure, AegisKV's mean reduction is about "
        "0.323%. Timings and architecture adapters are modeled/unmeasured.\n"
    )
    (stage / "figure_caption.txt").write_text(caption, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "paper_ready": False,
        "visual_only": True,
        "source_bundle": str(SOURCE),
        "source_hashes": source_hashes,
        "audit": "audit.json",
        "audit_sha256": sha256(stage / "audit.json"),
        "figure": figure,
        "figure_caption": "figure_caption.txt",
        "display_policy_order": NAMES,
        "failure_domains": DOMAINS,
        "paired_ci_inference_unit": "workload_seed",
        "paired_seed_count": 10,
        "pooled_cdf_is_descriptive": True,
        "cdf_horizontal_offset_applied": False,
        "no_result_selection_or_data_change": True,
        "evidence_boundary": source_manifest["evidence_boundary"],
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
        "".join(
            f"{value}  {name}\n" for name, value in sorted(artifacts.items())
        ),
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
