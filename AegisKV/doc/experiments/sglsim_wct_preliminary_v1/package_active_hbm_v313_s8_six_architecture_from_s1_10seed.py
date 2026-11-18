#!/usr/bin/env python3
"""Derive an audit-bound S8 failure-domain visual from frozen S1 results."""

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
TARGET = ROOT / "results" / "active_hbm_v313_s8_six_architecture_from_s1_10seed_v2"
NAMES = [
    "Native Mooncake", "Static", "DejaVu", "KevlarFlow",
    "GhostServe", "AegisKV",
]
PAIRED_NAMES = NAMES[1:]
DOMAINS = ["source_worker_gpu", "source_node"]
DOMAIN_LABELS = {
    "source_worker_gpu": "Worker / GPU",
    "source_node": "Source-node",
}


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
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derive_sources(
    architecture: list[dict[str, str]], paired: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    native_rows: list[dict[str, Any]] = []
    for row in architecture:
        if row["display_policy"] != "Native Mooncake":
            continue
        native_rows.append({
            "display_policy": row["display_policy"],
            "failure_domain": row["failure_domain"],
            "seed_count": int(row["seed_count"]),
            "mean_wct_s": float(row["wct_mean_ms_mean"]) / 1000.0,
            "mean_wct_ci95_low_s": float(row["wct_mean_ms_ci95_low"]) / 1000.0,
            "mean_wct_ci95_high_s": float(row["wct_mean_ms_ci95_high"]) / 1000.0,
            "p99_wct_s": float(row["wct_p99_ms_mean"]) / 1000.0,
            "p99_wct_ci95_low_s": float(row["wct_p99_ms_ci95_low"]) / 1000.0,
            "p99_wct_ci95_high_s": float(row["wct_p99_ms_ci95_high"]) / 1000.0,
            "ci_inference_unit": "workload_seed",
        })
    paired_rows: list[dict[str, Any]] = []
    for row in paired:
        paired_rows.append({
            "display_policy": row["display_policy"],
            "baseline": "Native Mooncake",
            "failure_domain": row["failure_domain"],
            "paired_seed_count": int(row["paired_seed_count"]),
            "mean_wct_reduction_pct": float(row["paired_mean_gain_pct_mean"]),
            "mean_wct_reduction_ci95_low_pct": float(
                row["paired_mean_gain_pct_ci95_low"]
            ),
            "mean_wct_reduction_ci95_high_pct": float(
                row["paired_mean_gain_pct_ci95_high"]
            ),
            "p99_wct_reduction_pct": float(row["paired_p99_gain_pct_mean"]),
            "p99_wct_reduction_ci95_low_pct": float(
                row["paired_p99_gain_pct_ci95_low"]
            ),
            "p99_wct_reduction_ci95_high_pct": float(
                row["paired_p99_gain_pct_ci95_high"]
            ),
            "ci_inference_unit": "workload_seed",
        })
    return native_rows, paired_rows


def render(
    stage: Path,
    native_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    import fitz

    colors = {
        "Static": "#8c564b", "DejaVu": "#1f77b4",
        "KevlarFlow": "#9467bd", "GhostServe": "#2ca02c",
        "AegisKV": "#d62728",
    }
    metrics = {
        "mean": {"label": "Mean WCT", "marker": "o", "offset": -0.13},
        "p99": {"label": "p99 WCT", "marker": "s", "offset": 0.13},
    }
    plt.rcParams.update({
        "font.size": 8.0, "axes.labelsize": 8.3, "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.4, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 3.35),
        gridspec_kw={"width_ratios": [0.75, 1.40, 1.40]},
    )

    axis = axes[0]
    native_by_domain = {row["failure_domain"]: row for row in native_rows}
    native_y = np.arange(len(DOMAINS), dtype=float)
    for metric, style in metrics.items():
        for index, domain in enumerate(DOMAINS):
            row = native_by_domain[domain]
            center = float(row[f"{metric}_wct_s"])
            low = float(row[f"{metric}_wct_ci95_low_s"])
            high = float(row[f"{metric}_wct_ci95_high_s"])
            axis.errorbar(
                center, native_y[index] + style["offset"],
                xerr=np.asarray([[center - low], [high - center]]),
                fmt=style["marker"], color="#222222",
                markerfacecolor=("#222222" if metric == "mean" else "white"),
                markeredgewidth=0.9, markersize=4.4,
                elinewidth=1.0, capsize=2.2, capthick=0.9,
            )
    axis.set_title("Native absolute WCT")
    axis.set_xlabel("WCT (s)")
    axis.set_yticks(native_y)
    axis.set_yticklabels([DOMAIN_LABELS[domain] for domain in DOMAINS])
    axis.set_ylim(len(DOMAINS) - 0.48, -0.48)
    axis.set_xlim(185.0, 468.0)
    axis.grid(True, axis="x", alpha=0.22, linewidth=0.6)

    for column, domain in enumerate(DOMAINS, start=1):
        axis = axes[column]
        by_name = {
            row["display_policy"]: row for row in paired_rows
            if row["failure_domain"] == domain
        }
        paired_y = np.arange(len(PAIRED_NAMES), dtype=float)
        for metric, style in metrics.items():
            for index, name in enumerate(PAIRED_NAMES):
                row = by_name[name]
                center = float(row[f"{metric}_wct_reduction_pct"])
                low = float(row[f"{metric}_wct_reduction_ci95_low_pct"])
                high = float(row[f"{metric}_wct_reduction_ci95_high_pct"])
                axis.errorbar(
                    center, paired_y[index] + style["offset"],
                    xerr=np.asarray([[center - low], [high - center]]),
                    fmt=style["marker"], color=colors[name],
                    markerfacecolor=(colors[name] if metric == "mean" else "white"),
                    markeredgewidth=0.9, markersize=4.3,
                    elinewidth=1.0, capsize=2.2, capthick=0.9,
                    zorder=3,
                )
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8,
                     zorder=0)
        axis.set_title(f"{DOMAIN_LABELS[domain]} failure")
        axis.set_xlabel("WCT reduction vs Native (%)")
        axis.set_yticks(paired_y)
        axis.set_yticklabels(PAIRED_NAMES if column == 1 else [])
        axis.set_ylim(len(PAIRED_NAMES) - 0.48, -0.52)
        axis.grid(True, axis="x", alpha=0.22, linewidth=0.6)
        axis.set_xlim(-0.55, 10.75) if domain == "source_worker_gpu" else axis.set_xlim(-0.115, 0.70)
    metric_handles = [
        Line2D(
            [0], [0], color="#333333", marker=style["marker"],
            markerfacecolor=("#333333" if metric == "mean" else "white"),
            linestyle="none", markersize=4.7, label=style["label"],
        )
        for metric, style in metrics.items()
    ]
    fig.legend(
        handles=metric_handles, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.995), columnspacing=1.5, handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.13, right=0.988, bottom=0.18, top=0.82, wspace=0.62,
    )

    figure_dir = stage / "figures"
    figure_dir.mkdir(parents=True)
    pdf = figure_dir / "s8_v313_six_architecture_failure_domain_10seed_visual_v2.pdf"
    png = figure_dir / "s8_v313_six_architecture_failure_domain_10seed_visual_v2.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    width, height = Image.open(png).size
    document = fitz.open(pdf)
    font_sizes: list[float] = []
    text_inside = True
    image_count = 0
    for page in document:
        image_count += len(page.get_images(full=True))
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        font_sizes.append(float(span["size"]))
                        text_inside = text_inside and page.rect.contains(
                            fitz.Rect(span["bbox"])
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
        "pdf_raster_image_count": image_count,
        "all_text_inside_pdf_page": text_inside,
    }


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"immutable target exists: {TARGET}")
    source_verify = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=SOURCE,
        capture_output=True, text=True,
    )
    if source_verify.returncode != 0:
        raise RuntimeError(source_verify.stdout + source_verify.stderr)

    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    source_audit = json.loads((SOURCE / "audit.json").read_text())
    architecture = read_csv(SOURCE / "architecture_ci.csv")
    paired = read_csv(SOURCE / "paired_vs_native_ci.csv")
    native_rows, paired_rows = derive_sources(architecture, paired)

    stage = TARGET.with_name(f".{TARGET.name}.stage-{os.getpid()}")
    stage.mkdir(parents=True)
    shutil.copy2(SOURCE / "architecture_ci.csv", stage / "source_architecture_ci.csv")
    shutil.copy2(
        SOURCE / "paired_vs_native_ci.csv",
        stage / "source_paired_vs_native_ci.csv",
    )
    write_csv(stage / "s8_native_absolute_wct_ci.csv", native_rows)
    write_csv(stage / "s8_paired_wct_reduction_ci.csv", paired_rows)
    figure = render(stage, native_rows, paired_rows)

    source_hashes = {
        "manifest_sha256": sha256(SOURCE / "manifest.json"),
        "audit_sha256": sha256(SOURCE / "audit.json"),
        "sha256sums_sha256": sha256(SOURCE / "SHA256SUMS"),
        "architecture_ci_sha256": sha256(SOURCE / "architecture_ci.csv"),
        "paired_vs_native_ci_sha256": sha256(
            SOURCE / "paired_vs_native_ci.csv"
        ),
    }
    native_groups = {
        (row["display_policy"], row["failure_domain"])
        for row in native_rows
    }
    paired_groups = {
        (row["display_policy"], row["failure_domain"])
        for row in paired_rows
    }
    node_rows = {
        row["display_policy"]: row for row in paired_rows
        if row["failure_domain"] == "source_node"
    }
    checks = {
        "source_sha256sums_all_pass": source_verify.returncode == 0,
        "source_s1_full_audit_pass": (
            source_manifest["status"] == "PASS"
            and source_manifest["mode"] == "full"
            and source_audit["status"] == "PASS"
        ),
        "same_s1_bundle_not_independent_samples": True,
        "exact_six_display_names_in_source": source_manifest[
            "display_policy_order"
        ] == NAMES,
        "exact_native_two_domain_rows": native_groups == {
            ("Native Mooncake", domain) for domain in DOMAINS
        },
        "exact_five_paired_architectures_two_domains": paired_groups == {
            (name, domain) for name in PAIRED_NAMES for domain in DOMAINS
        },
        "ci_inference_unit_ten_workload_seeds": (
            {row["seed_count"] for row in native_rows} == {10}
            and {row["paired_seed_count"] for row in paired_rows} == {10}
            and all(row["ci_inference_unit"] == "workload_seed"
                    for row in native_rows + paired_rows)
        ),
        "dejavu_node_negative_retained": (
            node_rows["DejaVu"]["mean_wct_reduction_pct"] < 0.0
            and node_rows["DejaVu"]["p99_wct_reduction_pct"] < 0.0
        ),
        "kevlarflow_node_negative_retained": (
            node_rows["KevlarFlow"]["mean_wct_reduction_pct"] < 0.0
            and node_rows["KevlarFlow"]["p99_wct_reduction_pct"] < 0.0
        ),
        "aegiskv_weak_node_advantage_retained": (
            0.0 < node_rows["AegisKV"]["mean_wct_reduction_pct"] < 1.0
        ),
        "source_architecture_ci_byte_identical": sha256(
            stage / "source_architecture_ci.csv"
        ) == source_hashes["architecture_ci_sha256"],
        "source_paired_ci_byte_identical": sha256(
            stage / "source_paired_vs_native_ci.csv"
        ) == source_hashes["paired_vs_native_ci_sha256"],
        "single_page_vector_pdf": (
            figure["pdf_page_count"] == 1
            and figure["pdf_raster_image_count"] == 0
        ),
        "png_width_at_least_1800": figure["png_width_px"] >= 1800,
        "pdf_min_font_at_least_7pt": figure["pdf_min_font_pt"] >= 6.99,
        "pdf_text_not_cropped": figure["all_text_inside_pdf_page"],
        "no_performance_direction_seed_or_domain_selection": source_manifest[
            "no_result_seed_architecture_or_domain_selection"
        ],
    }
    lineage = {
        "schema_version": 1,
        "derivation": "visual_and_unit_conversion_only_no_DES_rerun",
        "source_bundle": str(SOURCE),
        "source_hashes": source_hashes,
        "source_campaign": "S1 six-architecture ten-seed full",
        "same_sample_as_s1": True,
        "independent_s8_sample": False,
        "paired_ci_inference_unit": "workload_seed",
        "paired_seed_count": 10,
        "derivation_rules": {
            "native_absolute_seconds": "architecture_ci milliseconds / 1000",
            "paired_reduction_percent": "paired_vs_native_ci values unchanged",
        },
    }
    (stage / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "visual_only_derivation": True,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "source_bundle": str(SOURCE),
        "source_hashes": source_hashes,
        "renderer": Path(__file__).name,
        "renderer_sha256": sha256(Path(__file__)),
        "figure": figure,
        "evidence_boundary": source_manifest["evidence_boundary"],
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"audit FAIL {audit['failed_checks']}; stage={stage}")

    caption = (
        "Failure-domain comparison derived from the same ten-seed S1 "
        "simulation bundle, not an independent S8 sample. Native Mooncake's "
        "absolute mean and p99 WCT anchor the paired panels. Points show "
        "paired WCT reduction relative to Native; bars are Student-t 95% "
        "confidence intervals across 10 workload seeds. DejaVu and KevlarFlow "
        "retain their negative source-node reductions, while AegisKV's modeled "
        "source-node mean reduction is only about 0.323%. Positive reduction "
        "means lower WCT. Hardware and failover timings are modeled/unmeasured.\n"
    )
    (stage / "figure_caption.txt").write_text(caption, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "paper_ready": False,
        "visual_only_derivation": True,
        "source_bundle": str(SOURCE),
        "same_sample_as_s1": True,
        "independent_s8_sample": False,
        "source_hashes": source_hashes,
        "lineage": "lineage.json",
        "audit": "audit.json",
        "audit_sha256": sha256(stage / "audit.json"),
        "renderer": Path(__file__).name,
        "renderer_sha256": sha256(Path(__file__)),
        "figure": figure,
        "figure_caption": "figure_caption.txt",
        "native_absolute_source": "s8_native_absolute_wct_ci.csv",
        "paired_reduction_source": "s8_paired_wct_reduction_ci.csv",
        "display_policy_order": NAMES,
        "failure_domains": DOMAINS,
        "paired_ci_inference_unit": "workload_seed",
        "paired_seed_count": 10,
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
            f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())
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
