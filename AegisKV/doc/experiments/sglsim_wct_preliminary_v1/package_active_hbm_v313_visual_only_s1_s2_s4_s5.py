#!/usr/bin/env python3
"""Re-render frozen S1/S2/S4/S5 data without evidence prose in panels.

This packager changes figure text and layout only.  It verifies both frozen
input checksum trees, regenerates plots from their existing CSV/NPZ sources,
checks that regenerated/copy source CSVs are byte-identical to their frozen
predecessors, and atomically publishes a separate immutable bundle.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import fitz
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from PIL import Image

import package_active_hbm_v313_s4_s5_targeted_v2_figures_v2 as s45plot
import plot_active_hbm_v313_s1_s2_pilot as s12plot
import run_active_hbm_v313_s4_s5_targeted_v2 as s45run
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
S12_INPUT = ROOT / "results/active_hbm_v313_paper_preview_s1_s2_pilot"
S45_INPUT = ROOT / "results/active_hbm_v313_s4_s5_targeted_v2_figures_v4"
OUTPUT_NAME = "active_hbm_v313_visual_only_s1_s2_s4_s5_v1"

S12_SOURCES = (
    "s1_cold_fault_ecdf_source.csv",
    "s2_fixed_c16_rate_source.csv",
)
S45_SOURCES = (
    "s4_alg1_plot_source.csv", "s4_alg1_seed.csv",
    "s4_alg2_plot_source.csv", "s4_alg2_seed.csv",
    "s4_alg3_plot_source.csv", "s4_alg3_timeline_seed.csv",
    "s4_paired_wct_ci.csv", "s5_plot_source.csv", "s5_timeline_seed.csv",
)
FIGURES = (
    "s1_architecture_fault_ecdf",
    "s2_failure_rate_sensitivity",
    "s4_mechanism_ablations",
    "s5_aimd_physical_occupancy",
)


def verify_checksum_tree(path: Path) -> bool:
    for line in (path / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if core.sha256(path / name) != digest:
            return False
    return True


@contextmanager
def replace_suptitle(replacement: str) -> Iterator[None]:
    """Replace exactly one evidence-bearing suptitle without changing data."""

    original = Figure.suptitle
    calls = {"count": 0}

    def neutral(self: Figure, _text: str, *args, **kwargs):
        calls["count"] += 1
        return original(self, replacement, *args, **kwargs)

    Figure.suptitle = neutral
    try:
        yield
    finally:
        Figure.suptitle = original
    if calls["count"] != 1:
        raise AssertionError(f"expected one suptitle replacement, got {calls['count']}")


def pdf_text_and_min_font(path: Path) -> tuple[str, float, bool]:
    texts: list[str] = []
    sizes: list[float] = []
    with fitz.open(path) as document:
        one_page = len(document) == 1
        for page in document:
            texts.append(page.get_text())
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["text"].strip():
                            sizes.append(float(span["size"]))
    return "\n".join(texts), min(sizes), one_page


def numeric_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Restore numeric CSV fields expected by the frozen S5 plot function."""

    converted: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = {}
        for key, value in row.items():
            try:
                item[key] = float(value)
            except ValueError:
                item[key] = value
        converted.append(item)
    return converted


def main() -> None:
    if not verify_checksum_tree(S12_INPUT):
        raise RuntimeError("frozen S1/S2 checksum tree failed")
    if not verify_checksum_tree(S45_INPUT):
        raise RuntimeError("frozen S4/S5 checksum tree failed")

    output = ROOT / "results" / OUTPUT_NAME
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage = ROOT / "results" / f".{OUTPUT_NAME}.stage-{os.getpid()}"
    (stage / "figures").mkdir(parents=True)

    # S1/S2 regenerate from the same frozen raw inputs used by the predecessor.
    # Their source CSVs must match the predecessor byte-for-byte.
    s12plot.style()
    with replace_suptitle("Architecture fault response"):
        s12plot.plot_s1(stage)
    with replace_suptitle("Failure-rate sensitivity"):
        s12plot.plot_s2(stage)
    os.replace(stage / "s1_cold_fault_ecdf.pdf", stage / "figures/s1_architecture_fault_ecdf.pdf")
    os.replace(stage / "s1_cold_fault_ecdf.png", stage / "figures/s1_architecture_fault_ecdf.png")
    os.replace(stage / "s2_fixed_c16_rate_paired.pdf", stage / "figures/s2_failure_rate_sensitivity.pdf")
    os.replace(stage / "s2_fixed_c16_rate_paired.png", stage / "figures/s2_failure_rate_sensitivity.png")

    # S4/S5 consume the already published aggregate CSVs.  Seed CSVs are copied
    # for lineage, but no statistic is recomputed or changed.
    for name in S45_SOURCES:
        shutil.copy2(S45_INPUT / name, stage / name)
    with replace_suptitle("Mechanism ablations"):
        s45plot.plot_s4(
            stage / "figures/s4_mechanism_ablations",
            s45plot.read_csv(stage / "s4_alg1_plot_source.csv"),
            s45plot.read_csv(stage / "s4_alg2_plot_source.csv"),
            s45plot.read_csv(stage / "s4_alg3_plot_source.csv"),
        )
    with replace_suptitle("AIMD and physical occupancy"):
        s45run.plot_s5(
            stage / "figures/s5_aimd_physical_occupancy",
            numeric_rows(s45plot.read_csv(stage / "s5_plot_source.csv")),
        )

    forbidden = ("pilot", "preliminary", "seed", "sha", "hash", "evidence")
    figure_checks: dict[str, bool] = {}
    figure_stats: dict[str, dict[str, object]] = {}
    for stem in FIGURES:
        pdf = stage / "figures" / f"{stem}.pdf"
        png = stage / "figures" / f"{stem}.png"
        text, min_font, one_page = pdf_text_and_min_font(pdf)
        with Image.open(png) as image:
            width, height = image.size
        figure_checks[f"{stem}_single_page_vector_pdf"] = one_page
        figure_checks[f"{stem}_png_width_at_least_1800"] = width >= 1800
        figure_checks[f"{stem}_minimum_font_at_least_7pt"] = min_font >= 6.99
        figure_checks[f"{stem}_no_evidence_prose_inside_panel"] = not any(
            re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE)
            for token in forbidden
        )
        figure_stats[stem] = {
            "pdf_sha256": core.sha256(pdf),
            "png_sha256": core.sha256(png),
            "png_width_px": width,
            "png_height_px": height,
            "minimum_extracted_font_pt": min_font,
            "pdf_text": " ".join(text.split()),
        }

    source_checks = {
        "s12_sources_byte_identical": all(
            core.sha256(stage / name) == core.sha256(S12_INPUT / name)
            for name in S12_SOURCES
        ),
        "s45_sources_byte_identical": all(
            core.sha256(stage / name) == core.sha256(S45_INPUT / name)
            for name in S45_SOURCES
        ),
    }
    checks = {
        "frozen_s12_checksum_tree_pass": True,
        "frozen_s45_checksum_tree_pass": True,
        "frozen_s12_audit_pass": json.loads((S12_INPUT / "audit.json").read_text())["status"] == "PASS",
        "frozen_s45_audit_pass": json.loads((S45_INPUT / "audit.json").read_text())["status"] == "PASS",
        "no_simulated_data_or_statistic_changes": True,
        **source_checks,
        **figure_checks,
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "visual_only": True,
        "manual_visual_inspection": "required after atomic publication",
        "figure_stats": figure_stats,
        "frozen_inputs": {
            "S1_S2": {
                "path": str(S12_INPUT),
                "audit_sha256": core.sha256(S12_INPUT / "audit.json"),
                "manifest_sha256": core.sha256(S12_INPUT / "manifest.json"),
                "sha256sums_sha256": core.sha256(S12_INPUT / "SHA256SUMS"),
            },
            "S4_S5": {
                "path": str(S45_INPUT),
                "audit_sha256": core.sha256(S45_INPUT / "audit.json"),
                "manifest_sha256": core.sha256(S45_INPUT / "manifest.json"),
                "sha256sums_sha256": core.sha256(S45_INPUT / "SHA256SUMS"),
            },
        },
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "status": audit["status"],
        "paper_ready": False,
        "visual_only": True,
        "no_data_change": True,
        "figures": {
            "S1": "figures/s1_architecture_fault_ecdf",
            "S2": "figures/s2_failure_rate_sensitivity",
            "S4": "figures/s4_mechanism_ablations",
            "S5": "figures/s5_aimd_physical_occupancy",
        },
        "title_changes": {
            "S1": "Architecture fault response",
            "S2": "Failure-rate sensitivity",
            "S4": "Mechanism ablations",
            "S5": "AIMD and physical occupancy",
        },
        "panel_text_policy": (
            "No pilot, preliminary, seed, checksum, hash, or evidence-boundary prose "
            "inside any figure; those qualifiers remain in this manifest and source audits."
        ),
        "frozen_inputs": audit["frozen_inputs"],
        "evidence_boundary": (
            "Visual-only re-render of frozen modeled-DES data. S1 is a one-seed "
            "cold-start cohort; S2 is fixed-c16 with one workload seed and 2,800 "
            "90-second windows per rate; S4/S5 use three paired seeds. These are "
            "not physical A100 or paper-ready evidence."
        ),
        "packager_sha256": core.sha256(Path(__file__)),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if audit["status"] != "PASS":
        raise RuntimeError(audit["failed_checks"])
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
        "sha256sums_sha256": core.sha256(output / "SHA256SUMS"),
        "figure_sha256": {
            str(path.relative_to(output)): core.sha256(path)
            for path in sorted((output / "figures").iterdir())
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
