#!/usr/bin/env python3
"""Recompute S3 postprocessing from the immutable complete FAIL stage.

The DES raw files are never altered.  This finalizer exists solely because the
original audit expected 40 paired rows instead of the dimensionally required
80.  It rechecks raw WCT arrays, regenerates CI tables and the figure, records
the amendment, and atomically publishes a separate PASS bundle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import run_active_hbm_v313_s3_scale_context as runner
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results/active_hbm_v313_s3_scale_context_pilot.invalidated_audit_count_v1"
TARGET = ROOT / "results/active_hbm_v313_s3_scale_context_pilot"
SPEC_PATH = ROOT / "active_hbm_v313_s3_scale_context_protocol.json"
FROZEN_RUNNER = ROOT / "run_active_hbm_v313_s3_scale_context_simulation_v1_frozen.py"
CORRECTED_RUNNER = ROOT / "run_active_hbm_v313_s3_scale_context.py"


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tree_hash(path: Path) -> tuple[str, dict[str, str]]:
    rows = {
        str(item.relative_to(path)): core.sha256(item)
        for item in sorted(path.rglob("*")) if item.is_file()
    }
    return digest_json(rows), rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-7)


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite {TARGET}")
    source_audit = json.loads((INPUT / "audit.json").read_text())
    source_manifest = json.loads((INPUT / "manifest.json").read_text())
    if source_audit["failed_checks"] != ["exact_40_paired_metric_rows"]:
        raise RuntimeError("input is not the expected single-check FAIL stage")
    if core.sha256(FROZEN_RUNNER) != source_manifest["runner_sha256"]:
        raise RuntimeError("frozen simulation runner does not match input manifest")
    input_tree_sha, input_tree_files = tree_hash(INPUT)

    stage = TARGET.parent / f".{TARGET.name}.corrected-stage-{os.getpid()}"
    shutil.copytree(INPUT, stage)
    try:
        (stage / "figures").mkdir(exist_ok=True)
        (stage / "provenance").mkdir()
        shutil.copy2(stage / "audit.json", stage / "diagnostic_source_audit.json")
        shutil.copy2(stage / "manifest.json", stage / "diagnostic_source_manifest.json")
        shutil.copy2(stage / "aggregate_ci.csv", stage / "diagnostic_source_aggregate_ci.csv")
        shutil.copy2(stage / "paired_ci.csv", stage / "diagnostic_source_paired_ci.csv")
        shutil.copy2(FROZEN_RUNNER, stage / "provenance/simulation_runner.py")
        shutil.copy2(CORRECTED_RUNNER, stage / "provenance/corrected_postprocess_runner.py")
        shutil.copy2(Path(__file__), stage / "provenance/finalizer.py")

        negative = subprocess.run(
            ["python", str(ROOT / "test_active_hbm_v313_s3_postprocess.py")],
            cwd=str(ROOT), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        (stage / "test_logs/test_active_hbm_v313_s3_postprocess.log").write_text(
            negative.stdout, encoding="utf-8"
        )
        negative_pass = negative.returncode == 0 and "RESULT 9/9 PASS" in negative.stdout

        cells = read_csv(stage / "cell_metrics.csv")
        raw_metric_exact = True
        raw_hash_exact = True
        raw_names = set()
        for cell in cells:
            raw_path = stage / cell["raw_npz"]
            raw_names.add(cell["raw_npz"])
            raw_hash_exact &= (
                raw_path.exists()
                and core.sha256(raw_path) == cell["raw_npz_sha256"]
            )
            with np.load(raw_path) as raw:
                healthy = np.asarray(raw["healthy_wct_ms"], dtype=float)
                fault = np.asarray(raw["worker_fault_wct_ms"], dtype=float)
                shared = np.asarray(raw["capacity_shared_used_bytes"], dtype=np.int64)
                raw_metric_exact &= (
                    len(healthy) == int(cell["workflow_count"])
                    and fault.shape[0] == int(cell["fault_probe_count"])
                    and fault.size == int(cell["fault_observation_count"])
                    and close(np.mean(healthy), cell["healthy_wct_mean_ms"])
                    and close(core.quantile(healthy.tolist(), 0.99), cell["healthy_wct_p99_ms"])
                    and close(np.mean(fault), cell["fault_wct_mean_ms"])
                    and close(core.quantile(fault.ravel().tolist(), 0.99), cell["fault_wct_p99_ms"])
                    and int(np.max(shared)) == int(cell["capacity_max_shared_used_bytes"])
                )

        aggregate_rows, paired_rows = runner.aggregate(cells)
        runner.write_csv(stage / "aggregate_ci.csv", aggregate_rows)
        runner.write_csv(stage / "paired_ci.csv", paired_rows)
        spec = json.loads((stage / "protocol.json").read_text())
        stem = spec["figure_grammar"]["filename_stem"]
        figure_path = stage / "figures" / stem
        runner.plot_s3(figure_path, aggregate_rows)
        figure_checks = runner.verify_figure(
            figure_path.with_suffix(".pdf"), figure_path.with_suffix(".png"),
            float(spec["figure_grammar"]["minimum_font_pt"]),
            int(spec["figure_grammar"]["png_minimum_width_px"]),
        )

        expected_metrics = {
            "healthy_wct_mean_ms", "healthy_wct_p99_ms",
            "fault_wct_mean_ms", "fault_wct_p99_ms",
        }
        paired_keys = {
            (row["axis"], row["axis_value"], row["topology"], row["metric"])
            for row in paired_rows
        }
        amendment = {
            "schema_version": 1,
            "status": "AUDIT_EXPECTATION_CORRECTED_WITHOUT_DES_RERUN",
            "immutable_input": str(INPUT),
            "immutable_input_tree_sha256": input_tree_sha,
            "immutable_input_file_count": len(input_tree_files),
            "source_failed_check": "exact_40_paired_metric_rows",
            "derivation": {
                "scale_display_points": 5,
                "context_display_points": 5,
                "shared_center_execution_note": "one DES center cell contributes to both preregistered display axes",
                "topology_count": 2,
                "paired_metric_count": 4,
                "formula": "(5 scale + 5 context) * 2 topologies * 4 paired metrics",
                "correct_expected_rows": 80,
            },
            "simulation_runner_sha256": core.sha256(FROZEN_RUNNER),
            "corrected_postprocess_runner_sha256": core.sha256(CORRECTED_RUNNER),
            "finalizer_sha256": core.sha256(Path(__file__)),
            "no_raw_cell_changed_or_selected": True,
        }
        (stage / "audit_amendment.json").write_text(
            json.dumps(amendment, indent=2, sort_keys=True) + "\n"
        )

        checks = dict(source_audit["checks"])
        checks.pop("exact_40_paired_metric_rows")
        checks.update({
            "postprocess_negative_regression_9_of_9_pass": negative_pass,
            "immutable_fail_stage_tree_bound": bool(input_tree_sha),
            "exact_108_unique_raw_cells": len(cells) == 108
            and len(raw_names) == 108,
            "all_raw_npz_hashes_match_immutable_cell_manifest": raw_hash_exact,
            "raw_WCT_and_capacity_metrics_recomputed_exactly": raw_metric_exact,
            "exact_40_aggregate_rows": len(aggregate_rows) == 40,
            "exact_80_paired_metric_rows_derived_from_registered_schema": (
                len(paired_rows) == 80
                and len(paired_keys) == 80
                and {row["metric"] for row in paired_rows} == expected_metrics
            ),
            "all_paired_rows_use_three_seeds": all(
                int(row["seed_count"]) == 3 for row in paired_rows
            ),
            "figure_single_page_vector_pdf": figure_checks["single_page_vector_pdf"],
            "figure_png_width_at_least_1800": figure_checks[
                "png_width_at_least_registered"
            ],
            "figure_minimum_font_at_least_7pt": figure_checks[
                "minimum_font_at_least_registered"
            ],
            "old_FAIL_stage_preserved_and_not_published": INPUT.exists()
            and source_audit["status"] == "FAIL",
            "no_DES_cell_rerun_or_selection_in_correction": True,
        })
        audit = {
            "schema_version": 2,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "mode": "full_corrected_postprocess",
            "paper_ready": False,
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "cell_count": len(cells),
            "aggregate_row_count": len(aggregate_rows),
            "paired_row_count": len(paired_rows),
            "protocol_sha256": core.sha256(stage / "protocol.json"),
            "capacity_protocol_sha256": core.sha256(stage / "capacity_protocol.json"),
            "cell_metrics_sha256": core.sha256(stage / "cell_metrics.csv"),
            "aggregate_ci_sha256": core.sha256(stage / "aggregate_ci.csv"),
            "paired_ci_sha256": core.sha256(stage / "paired_ci.csv"),
            "mechanism_counts_sha256": core.sha256(stage / "mechanism_counts.csv"),
            "audit_amendment_sha256": core.sha256(stage / "audit_amendment.json"),
            "immutable_input_tree_sha256": input_tree_sha,
            "figure_checks": figure_checks,
            "evidence_boundary": spec["evidence_boundary"],
        }
        (stage / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "schema_version": 2,
            "status": audit["status"],
            "paper_ready": False,
            "selection": "S3",
            "protocol": "protocol.json",
            "protocol_sha256": audit["protocol_sha256"],
            "simulation_runner": "provenance/simulation_runner.py",
            "simulation_runner_sha256": core.sha256(stage / "provenance/simulation_runner.py"),
            "corrected_postprocess_runner": "provenance/corrected_postprocess_runner.py",
            "corrected_postprocess_runner_sha256": core.sha256(
                stage / "provenance/corrected_postprocess_runner.py"
            ),
            "audit": "audit.json",
            "audit_sha256": core.sha256(stage / "audit.json"),
            "audit_amendment": "audit_amendment.json",
            "immutable_failed_stage": str(INPUT),
            "immutable_failed_stage_tree_sha256": input_tree_sha,
            "cell_metrics": "cell_metrics.csv",
            "cell_summary": "cell_summary.jsonl",
            "mechanism_counts": "mechanism_counts.csv",
            "aggregate_ci": "aggregate_ci.csv",
            "paired_ci": "paired_ci.csv",
            "figure": {
                "pdf": f"figures/{stem}.pdf",
                "png": f"figures/{stem}.png",
            },
            "invalidated_runs_preserved": [
                "results/active_hbm_v313_s3_scale_context_pilot.invalidated_timeout_v1",
                "results/active_hbm_v313_s3_scale_context_pilot.invalidated_audit_count_v1",
            ],
            "old_s3_preserved": "results/s3_preliminary_v1",
            "evidence_boundary": spec["evidence_boundary"],
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if audit["status"] != "PASS":
            raise RuntimeError(f"corrected audit failed: {audit['failed_checks']}")
        artifacts = {
            str(path.relative_to(stage)): core.sha256(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        }
        (stage / "SHA256SUMS").write_text(
            "".join(
                f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())
            )
        )
        os.replace(stage, TARGET)
        print(json.dumps({
            "output": str(TARGET), "status": audit["status"],
            "audit_sha256": core.sha256(TARGET / "audit.json"),
            "manifest_sha256": core.sha256(TARGET / "manifest.json"),
            "amendment_sha256": core.sha256(TARGET / "audit_amendment.json"),
            "sha256sums_sha256": core.sha256(TARGET / "SHA256SUMS"),
            "figure_pdf_sha256": core.sha256(TARGET / f"figures/{stem}.pdf"),
            "figure_png_sha256": core.sha256(TARGET / f"figures/{stem}.png"),
        }, indent=2))
    except Exception:
        raise


if __name__ == "__main__":
    main()
