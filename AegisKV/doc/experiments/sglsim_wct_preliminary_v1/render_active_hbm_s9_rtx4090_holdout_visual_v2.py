#!/usr/bin/env python3
"""Visual-only S9 rerender from the frozen job141411 independent-holdout bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "active_hbm_s9_rtx4090_four_direction_holdout_141411_v1"
OUTPUT = ROOT / "results" / "active_hbm_s9_rtx4090_four_direction_holdout_141411_visual_v2"
PATHS = ["source_to_peer_hbm", "source_to_dram", "peer_hbm_recovery", "dram_recovery"]
LABELS = {
    "source_to_peer_hbm": "Source → Peer HBM",
    "source_to_dram": "Source → DRAM",
    "peer_hbm_recovery": "Peer HBM recovery",
    "dram_recovery": "DRAM recovery",
}
PAYLOADS = [67108864, 268435456, 536870912, 1073741824]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"fail-close: output exists: {OUTPUT}")
    check = os.system(f"cd '{SOURCE}' && sha256sum -c SHA256SUMS >/tmp/s9-v2-source-sha.log")
    if check != 0:
        raise SystemExit("fail-close: source SHA256SUMS mismatch")
    cells = rows(SOURCE / "data" / "holdout_errors_by_path_payload.csv")
    holdout = rows(SOURCE / "data" / "holdout_samples.csv")
    source_audit = json.loads((SOURCE / "audit.json").read_text())
    if len(cells) != 16 or len(holdout) != 80 or source_audit["audit_status"] != "PASS":
        raise SystemExit("fail-close: frozen source completeness mismatch")

    plt.rcParams.update({
        "font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.0, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[2.1, 1.0])
    x = np.arange(4); xlabels = ["64", "256", "512", "1024"]
    grouped = defaultdict(list)
    for row in holdout:
        grouped[(row["path"], int(row["payload_bytes"]))].append(float(row["observed_transfer_ms"]))
    legend_handles = None
    for col, path in enumerate(PATHS):
        top = fig.add_subplot(grid[0, col]); bottom = fig.add_subplot(grid[1, col], sharex=top)
        selected = sorted((row for row in cells if row["path"] == path), key=lambda row: int(row["payload_bytes"]))
        mean = np.array([float(row["observed_mean_ms"]) for row in selected])
        lo = np.array([float(row["observed_mean_ci95_low_ms"]) for row in selected])
        hi = np.array([float(row["observed_mean_ci95_high_ms"]) for row in selected])
        pred = np.array([float(row["predicted_transfer_ms"]) for row in selected])
        mape = np.array([float(row["mape_percent"]) for row in selected])
        for xpos, payload in zip(x, PAYLOADS):
            raw = grouped[(path, payload)]
            top.scatter(xpos + np.linspace(-0.07, 0.07, len(raw)), raw, s=9, color="#0072B2", alpha=0.35, linewidths=0)
        observed = top.errorbar(x, mean, yerr=[mean-lo, hi-mean], color="#0072B2", marker="o", markersize=3.4,
                                linewidth=1.25, capsize=2.0, label="Physical holdout mean")
        model, = top.plot(x, pred, color="#D55E00", marker="s", markersize=3.2, linewidth=1.25,
                          label="Frozen simulator curve")
        top.set_title(LABELS[path], pad=4); top.grid(axis="y", alpha=0.22, linewidth=0.5); top.tick_params(labelbottom=False)
        if path != "peer_hbm_recovery":
            top.set_yscale("log")
            top.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
            top.yaxis.set_minor_formatter(FuncFormatter(lambda _value, _pos: ""))
        if col == 0: top.set_ylabel("Transfer time (ms)")
        bottom.plot(x, mape, color="#4C78A8", marker="o", markersize=3.2, linewidth=1.1)
        gate = bottom.axhline(10.0, color="#333333", linestyle="--", linewidth=0.8)
        bottom.fill_between([-0.35, 3.35], 0, 10, color="#009E73", alpha=0.06, linewidth=0)
        bottom.set_ylim(0, max(12.0, float(mape.max())*1.18)); bottom.set_xlim(-0.35, 3.35)
        bottom.set_xticks(x, xlabels); bottom.set_xlabel("Payload (MiB)"); bottom.grid(axis="y", alpha=0.22, linewidth=0.5)
        if col == 0: bottom.set_ylabel("Cell MAPE (%)")
        if legend_handles is None: legend_handles = [observed, model, gate]
    fig.legend(legend_handles, ["Physical holdout mean ± 95% CI", "Frozen simulator curve", "10% reference gate"],
               loc="outside upper center", ncol=3, frameon=False, borderaxespad=0.2)

    stage = Path(tempfile.mkdtemp(prefix=".s9-visual-v2-stage-", dir=str(OUTPUT.parent)))
    try:
        (stage / "figures").mkdir(); (stage / "source").mkdir()
        pdf = stage / "figures" / "s9_rtx4090_four_direction_independent_holdout_visual_v2.pdf"
        png = stage / "figures" / "s9_rtx4090_four_direction_independent_holdout_visual_v2.png"
        fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=300, bbox_inches="tight"); plt.close(fig)
        for name in ["audit.json", "manifest.json", "SHA256SUMS"]:
            shutil.copy2(SOURCE / name, stage / "source" / name)
        for name in ["holdout_samples.csv", "holdout_errors_by_path_payload.csv", "fitted_rtx4090_service_curves.json"]:
            shutil.copy2(SOURCE / "data" / name, stage / "source" / name)
        doc = fitz.open(pdf); spans=[]
        for block in doc[0].get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans.extend(span for span in line.get("spans", []) if str(span.get("text", "")).strip())
        min_font = min(float(span["size"]) for span in spans)
        visual_checks = {
            "source_bundle_sha_pass": True, "source_rows_80_16": len(holdout)==80 and len(cells)==16,
            "source_acceptance_fail_retained": source_audit["calibration_acceptance"] == "FAIL",
            "single_page_vector_pdf": len(doc)==1 and bool(spans), "minimum_font_at_least_7pt": min_font >= 7.0,
            "png_at_least_1800px": plt.imread(png).shape[1] >= 1800,
            "all_four_paths_and_payloads_retained": {row["path"] for row in cells} == set(PATHS) and {int(row["payload_bytes"]) for row in cells} == set(PAYLOADS),
        }
        audit = {"status": "PASS" if all(visual_checks.values()) else "FAIL", "checks": visual_checks,
                 "minimum_font_pt": min_font, "data_mutation": "none", "source_audit_sha256": sha(SOURCE/"audit.json"),
                 "calibration_acceptance": "FAIL", "failed_cell": source_audit["failed_cells"]}
        write_json(stage / "audit.json", audit)
        manifest = {"schema_version": 1, "status": "PASS_VISUAL_ONLY_INDEPENDENT_HOLDOUT_ACCEPTANCE_FAIL",
                    "source_bundle": str(SOURCE), "source_manifest_sha256": sha(SOURCE/"manifest.json"),
                    "source_audit_sha256": sha(SOURCE/"audit.json"), "source_holdout_csv_sha256": sha(SOURCE/"data/holdout_samples.csv"),
                    "data_mutation": "none", "holdout_samples": 80, "cells": 16, "passing_cells": 15,
                    "overall_mape_percent": source_audit["overall_holdout_mape_percent"], "a100_profile_mutated": False,
                    "figure_pdf": str(pdf.relative_to(stage)), "figure_png": str(png.relative_to(stage)),
                    "evidence_boundary": "Visual-only rendering of valid independent RTX4090 holdout job141411; all-cell 10 percent gate remains FAIL and A100 main profile is unchanged."}
        write_json(stage / "manifest.json", manifest)
        files=sorted(p for p in stage.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
        (stage/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(stage)}\n" for p in files))
        if not all(visual_checks.values()): raise RuntimeError(audit)
        os.replace(stage, OUTPUT)
    except Exception:
        print(f"stage preserved: {stage}")
        raise
    final_pdf = OUTPUT / "figures" / pdf.name
    final_png = OUTPUT / "figures" / png.name
    print(json.dumps({"output": str(OUTPUT), "minimum_font_pt": min_font, "pdf_sha256": sha(final_pdf), "png_sha256": sha(final_png)}, indent=2))


if __name__ == "__main__": main()
