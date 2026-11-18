#!/usr/bin/env python3
"""Build the preregistered RTX4090 four-direction independent-holdout S9 bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "active_hbm_s9_rtx4090_four_direction_holdout_protocol.json"
EVAL_PROTOCOL = ROOT / "active_hbm_s9_rtx4090_four_direction_holdout_141411_evaluation_protocol.json"
DEFAULT_OUT = ROOT / "results" / "active_hbm_s9_rtx4090_four_direction_holdout_141411_v1"
PATHS = ["source_to_peer_hbm", "source_to_dram", "peer_hbm_recovery", "dram_recovery"]
PATH_LABELS = {
    "source_to_peer_hbm": "Source → Peer HBM",
    "source_to_dram": "Source → DRAM",
    "peer_hbm_recovery": "Peer HBM recovery",
    "dram_recovery": "DRAM recovery",
}
PAYLOADS = [67108864, 268435456, 536870912, 1073741824]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def exact_bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    """Seed-free exhaustive ordinary bootstrap for n=5 (5^5 means)."""
    array = np.asarray(values, dtype=float)
    means = np.fromiter(
        (array[list(indexes)].mean() for indexes in itertools.product(range(len(array)), repeat=len(array))),
        dtype=float,
        count=len(array) ** len(array),
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def verify_checksum_manifest(root: Path, manifest: Path) -> tuple[int, int, str]:
    proc = subprocess.run(
        ["sha256sum", "-c", str(manifest)], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    failed = sum("FAILED" in line for line in lines)
    return len(lines), failed, proc.stdout


def fit_curves(training: list[dict]) -> dict[str, dict]:
    curves: dict[str, dict] = {}
    for path in PATHS:
        rows = [row for row in training if row["path"] == path]
        x = np.asarray([[1.0, row["payload_bytes"] / 2**30] for row in rows], dtype=float)
        y = np.asarray([row["transfer_ms"] for row in rows], dtype=float)
        coefficients, residual_norm = nnls(x, y)
        setup_ms, ms_per_gib = map(float, coefficients)
        predictions = x @ coefficients
        ss_res = float(np.square(y - predictions).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        curves[path] = {
            "path": path,
            "estimator": "scipy.optimize.nnls",
            "training_sample_count": len(rows),
            "setup_ms": setup_ms,
            "ms_per_gib": ms_per_gib,
            "inverse_bandwidth_ms_per_byte": ms_per_gib / 2**30,
            "effective_asymptotic_gib_s": 1000.0 / ms_per_gib,
            "residual_norm_ms": float(residual_norm),
            "training_r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        }
    return curves


def predict_ms(curve: dict, payload_bytes: int) -> float:
    return float(curve["setup_ms"] + curve["ms_per_gib"] * payload_bytes / 2**30)


def make_figure(cell_rows: list[dict], holdout_rows: list[dict], curves: dict, figure_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.0, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[2.1, 1.0])
    x = np.arange(len(PAYLOADS))
    labels = ["64", "256", "512", "1024"]
    obs_color = "#0072B2"
    sim_color = "#D55E00"
    error_color = "#4C78A8"
    handles = None
    for column, path in enumerate(PATHS):
        top = fig.add_subplot(grid[0, column])
        bottom = fig.add_subplot(grid[1, column], sharex=top)
        cells = sorted((row for row in cell_rows if row["path"] == path), key=lambda row: row["payload_bytes"])
        means = np.asarray([row["observed_mean_ms"] for row in cells])
        lows = np.asarray([row["observed_mean_ci95_low_ms"] for row in cells])
        highs = np.asarray([row["observed_mean_ci95_high_ms"] for row in cells])
        predicted = np.asarray([row["predicted_transfer_ms"] for row in cells])
        err = np.asarray([row["mape_percent"] for row in cells])
        raw_by_payload = defaultdict(list)
        for row in holdout_rows:
            if row["path"] == path:
                raw_by_payload[row["payload_bytes"]].append(row["observed_transfer_ms"])
        for xpos, payload in zip(x, PAYLOADS):
            jitter = np.linspace(-0.07, 0.07, len(raw_by_payload[payload]))
            top.scatter(xpos + jitter, raw_by_payload[payload], s=9, color=obs_color, alpha=0.35, linewidths=0, zorder=2)
        obs = top.errorbar(
            x, means, yerr=[means - lows, highs - means], color=obs_color,
            marker="o", markersize=3.4, linewidth=1.25, capsize=2.0, label="Physical holdout mean",
        )
        sim, = top.plot(x, predicted, color=sim_color, marker="s", markersize=3.2, linewidth=1.25, label="Frozen simulator curve")
        top.set_title(PATH_LABELS[path], pad=4)
        top.set_yscale("log")
        top.grid(axis="y", alpha=0.22, linewidth=0.5)
        top.tick_params(labelbottom=False)
        if column == 0:
            top.set_ylabel("Transfer time (ms, log)")
        bottom.plot(x, err, color=error_color, marker="o", markersize=3.2, linewidth=1.1)
        bottom.axhline(10.0, color="#333333", linestyle="--", linewidth=0.8)
        bottom.fill_between([-0.35, 3.35], 0, 10, color="#009E73", alpha=0.06, linewidth=0)
        bottom.set_ylim(0, max(12.0, float(err.max()) * 1.18))
        bottom.set_xlim(-0.35, 3.35)
        bottom.set_xticks(x, labels)
        bottom.set_xlabel("Payload (MiB)")
        bottom.grid(axis="y", alpha=0.22, linewidth=0.5)
        if column == 0:
            bottom.set_ylabel("Cell MAPE (%)")
        if handles is None:
            handles = [obs, sim, plt.Line2D([], [], color="#333333", linestyle="--", linewidth=0.8)]
    fig.legend(handles, ["Physical holdout mean ± 95% CI", "Frozen simulator curve", "10% reference gate"],
               loc="outside upper center", ncol=3, frameon=False, borderaxespad=0.2)
    pdf = figure_dir / "s9_rtx4090_four_direction_independent_holdout_141411.pdf"
    png = figure_dir / "s9_rtx4090_four_direction_independent_holdout_141411.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"fail-close: output already exists: {output}")
    prereg = json.loads(PREREG.read_text())
    evaluation = json.loads(EVAL_PROTOCOL.read_text())
    train_path = Path(evaluation["fit_input"]["path"])
    holdout_path = Path(evaluation["holdout_input"]["calibration_path"])
    contract_path = Path(evaluation["holdout_input"]["contract_path"])
    gate_path = Path(evaluation["holdout_input"]["gate_path"])
    checksum_path = Path(evaluation["holdout_input"]["checksums_path"])
    train_doc = json.loads(train_path.read_text())
    holdout_doc = json.loads(holdout_path.read_text())
    contract = json.loads(contract_path.read_text())
    gate = json.loads(gate_path.read_text())
    training = train_doc["samples"]
    holdout = holdout_doc["samples"]
    holdout_root = holdout_path.parent
    checksum_count, checksum_failed, checksum_log = verify_checksum_manifest(holdout_root, checksum_path)

    train_probe_hashes = {row["probe_artifact"]["sha256"] for row in training}
    holdout_probe_hashes = {row["probe_artifact"]["sha256"] for row in holdout}
    holdout_identities = [
        (row["sequence"], row["repeat_ordinal"], row["path"], row["payload_bytes"], row["probe_artifact"]["sha256"])
        for row in holdout
    ]
    holdout_identity_sha = canonical_sha(holdout_identities)
    cell_counts = Counter((row["path"], row["payload_bytes"]) for row in holdout)
    prereg_mtime_ns = PREREG.stat().st_mtime_ns
    holdout_mtime_ns = holdout_path.stat().st_mtime_ns
    match_checks = {
        "preregistered_protocol_hash_exact": sha256(PREREG) == evaluation["preregistered_protocol"]["sha256"],
        "protocol_frozen_before_holdout": prereg_mtime_ns < holdout_mtime_ns,
        "fit_input_hash_exact": sha256(train_path) == evaluation["fit_input"]["sha256"],
        "holdout_calibration_hash_exact": sha256(holdout_path) == evaluation["holdout_input"]["calibration_sha256"],
        "holdout_contract_hash_exact": sha256(contract_path) == evaluation["holdout_input"]["contract_sha256"],
        "holdout_gate_hash_exact": sha256(gate_path) == evaluation["holdout_input"]["gate_sha256"],
        "holdout_checksum_manifest_hash_exact": sha256(checksum_path) == evaluation["holdout_input"]["checksums_sha256"],
        "physical_checksum_manifest_pass": checksum_count > 0 and checksum_failed == 0,
        "physical_gate_pass": gate["status"] == "PASS" and gate["calibration_ready"] is True,
        "new_job_and_run_lineage": str(gate["slurm_job_id"]) == "141411" and gate["lineage_id"] == evaluation["holdout_input"]["run_id"] and prereg["training_source"]["source_slurm_job_id"] != str(gate["slurm_job_id"]),
        "source_independent_no_old_result_reuse": contract["staged_calibration_extension"]["old_result_reused"] is False and not (train_probe_hashes & holdout_probe_hashes),
        "hardware_two_rtx4090_one_node": contract["hardware"]["gpu_model"] == "NVIDIA GeForce RTX 4090" and contract["hardware"]["gpus_per_node"] == 2 and contract["hardware"]["node_count"] == 1,
        "layout_and_dtype_exact": holdout_doc["layout"] == {"tp_size": 1, "pp_size": 1, "system_dp_size": 2} and gate["dtype"] == "float16",
        "four_physical_path_classes_exact": holdout_doc["physical_path_classes"] == {key: value["physical_path_class"] for key, value in prereg["path_mapping"].items()},
        "path_set_exact": set(holdout_doc["paths"]) == set(PATHS),
        "payload_set_exact": holdout_doc["payload_sizes"] == PAYLOADS,
        "exact_16_cells_5_repeats": len(cell_counts) == 16 and set(cell_counts.values()) == {5},
        "exact_80_holdout_samples": len(holdout) == 80,
        "holdout_sample_identity_hash_exact": holdout_identity_sha == evaluation["holdout_input"]["sample_identity_sha256"],
        "train_holdout_probe_hash_disjoint": len(train_probe_hashes & holdout_probe_hashes) == 0,
        "fit_only_training_80": len(training) == 80 and evaluation["fit_input"]["role"] == "FIT_ONLY",
    }
    if not all(match_checks.values()):
        failed = [name for name, passed in match_checks.items() if not passed]
        raise SystemExit("fail-close input mismatch: " + ", ".join(failed))

    curves = fit_curves(training)
    training_rows: list[dict] = []
    for row in training:
        curve = curves[row["path"]]
        training_rows.append({
            "role": "FIT_ONLY", "source_job_id": "141298", "run_id": prereg["training_source"]["physical_run_id"],
            "sequence": row["sequence"], "repeat_ordinal": row["repeat_ordinal"], "path": row["path"],
            "payload_bytes": row["payload_bytes"], "payload_mib": row["payload_bytes"] / 2**20,
            "observed_transfer_ms": row["transfer_ms"], "fitted_transfer_ms": predict_ms(curve, row["payload_bytes"]),
            "probe_artifact_sha256": row["probe_artifact"]["sha256"], "payload_checksum_sha256": row["checksum_sha256"],
        })
    holdout_rows: list[dict] = []
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in holdout:
        predicted = predict_ms(curves[row["path"]], row["payload_bytes"])
        observed = float(row["transfer_ms"])
        out = {
            "role": "HOLDOUT_ONLY", "source_job_id": "141411", "run_id": gate["lineage_id"],
            "sequence": row["sequence"], "repeat_ordinal": row["repeat_ordinal"], "path": row["path"],
            "physical_path_class": row["physical_path_class"], "payload_bytes": row["payload_bytes"],
            "payload_mib": row["payload_bytes"] / 2**20, "observed_transfer_ms": observed,
            "predicted_transfer_ms": predicted, "absolute_percentage_error": abs(predicted - observed) / observed * 100,
            "signed_percentage_error": (predicted - observed) / observed * 100,
            "observed_gib_s": row["effective_gib_per_s"],
            "predicted_gib_s": (row["payload_bytes"] / 2**30) / (predicted / 1000),
            "probe_artifact_sha256": row["probe_artifact"]["sha256"], "payload_checksum_sha256": row["checksum_sha256"],
        }
        holdout_rows.append(out)
        grouped[(row["path"], row["payload_bytes"])].append(out)
    cell_rows: list[dict] = []
    for path in PATHS:
        for payload in PAYLOADS:
            rows = grouped[(path, payload)]
            observed = [row["observed_transfer_ms"] for row in rows]
            ape = [row["absolute_percentage_error"] for row in rows]
            signed = [row["signed_percentage_error"] for row in rows]
            low, high = exact_bootstrap_mean_ci(observed)
            predicted = rows[0]["predicted_transfer_ms"]
            mape = float(np.mean(ape))
            cell_rows.append({
                "path": path, "payload_bytes": payload, "payload_mib": payload / 2**20, "repeat_count": len(rows),
                "raw_observed_transfer_ms_json": json.dumps(observed, separators=(",", ":")),
                "observed_mean_ms": float(np.mean(observed)), "observed_median_ms": float(np.median(observed)),
                "observed_mean_ci95_low_ms": low, "observed_mean_ci95_high_ms": high,
                "predicted_transfer_ms": predicted, "mape_percent": mape,
                "median_absolute_percentage_error_percent": float(np.median(ape)),
                "p95_absolute_percentage_error_percent": float(np.quantile(ape, 0.95)),
                "signed_mean_percentage_error_percent": float(np.mean(signed)),
                "observed_effective_gib_s_from_mean": (payload / 2**30) / (np.mean(observed) / 1000),
                "predicted_effective_gib_s": (payload / 2**30) / (predicted / 1000),
                "acceptance_threshold_percent": 10.0, "cell_acceptance_pass": mape <= 10.0,
            })
    acceptance_pass = all(row["cell_acceptance_pass"] for row in cell_rows)
    direction_summary = []
    for path in PATHS:
        rows = [row for row in holdout_rows if row["path"] == path]
        direction_summary.append({
            "path": path, "sample_count": len(rows),
            "mape_percent": float(np.mean([row["absolute_percentage_error"] for row in rows])),
            "signed_mean_percentage_error_percent": float(np.mean([row["signed_percentage_error"] for row in rows])),
            "max_cell_mape_percent": max(row["mape_percent"] for row in cell_rows if row["path"] == path),
        })

    stage = Path(tempfile.mkdtemp(prefix="s9-141411-stage-", dir=str(output.parent)))
    try:
        data_dir = stage / "data"; figure_dir = stage / "figures"; lineage_dir = stage / "lineage"
        data_dir.mkdir(parents=True); figure_dir.mkdir(); lineage_dir.mkdir()
        (stage / "physical-checksum-verification.log").write_text(checksum_log)
        shutil.copy2(PREREG, lineage_dir / PREREG.name)
        shutil.copy2(EVAL_PROTOCOL, lineage_dir / EVAL_PROTOCOL.name)
        write_csv(data_dir / "training_samples.csv", training_rows, list(training_rows[0]))
        write_csv(data_dir / "holdout_samples.csv", holdout_rows, list(holdout_rows[0]))
        write_csv(data_dir / "holdout_errors_by_path_payload.csv", cell_rows, list(cell_rows[0]))
        write_csv(data_dir / "direction_summary.csv", direction_summary, list(direction_summary[0]))
        curves_doc = {
            "schema_version": 1, "namespace": "rtx4090_calibration", "fit_source_job_id": "141298",
            "fit_source_sha256": sha256(train_path), "holdout_source_used_for_fit": False,
            "formula": evaluation["fit"]["formula"], "curves": curves,
            "a100_main_profile_mutated": False,
        }
        write_json(data_dir / "fitted_rtx4090_service_curves.json", curves_doc)
        pdf, png = make_figure(cell_rows, holdout_rows, curves, figure_dir)
        overall_mape = float(np.mean([row["absolute_percentage_error"] for row in holdout_rows]))
        failed_cells = [
            {"path": row["path"], "payload_mib": row["payload_mib"], "mape_percent": row["mape_percent"]}
            for row in cell_rows if not row["cell_acceptance_pass"]
        ]
        integrity_checks = dict(match_checks)
        integrity_checks.update({
            "four_independent_fitted_curves": set(curves) == set(PATHS),
            "positive_inverse_bandwidth_all_paths": all(curve["inverse_bandwidth_ms_per_byte"] > 0 for curve in curves.values()),
            "nonnegative_setup_all_paths": all(curve["setup_ms"] >= 0 for curve in curves.values()),
            "holdout_never_used_for_fit": curves_doc["holdout_source_used_for_fit"] is False,
            "exact_80_training_output_rows": len(training_rows) == 80,
            "exact_80_holdout_output_rows": len(holdout_rows) == 80,
            "exact_16_error_rows": len(cell_rows) == 16,
            "no_sample_exclusion": all(len(grouped[(path, payload)]) == 5 for path in PATHS for payload in PAYLOADS),
            "a100_profile_not_mutated": curves_doc["a100_main_profile_mutated"] is False,
        })
        audit = {
            "schema_version": 1,
            "audit_status": "PASS" if all(integrity_checks.values()) else "FAIL",
            "calibration_acceptance": "PASS" if acceptance_pass else "FAIL",
            "calibration_acceptance_rule": evaluation["acceptance"]["rule"],
            "checks": integrity_checks,
            "check_count": len(integrity_checks), "passed_check_count": sum(integrity_checks.values()),
            "input_match_status": "PASS",
            "holdout_sample_count": len(holdout_rows), "holdout_cell_count": len(cell_rows),
            "overall_holdout_mape_percent": overall_mape,
            "failed_cell_count": len(failed_cells), "failed_cells": failed_cells,
            "direction_summary": direction_summary,
            "uncertainty_deviation": evaluation["uncertainty"]["deviation_note"],
            "evidence_boundary": evaluation["evidence_boundary"],
        }
        write_json(stage / "audit.json", audit)
        manifest = {
            "schema_version": 1, "bundle_id": evaluation["evaluation_id"],
            "status": "VALID_INDEPENDENT_HOLDOUT_ACCEPTANCE_PASS" if acceptance_pass else "VALID_INDEPENDENT_HOLDOUT_ACCEPTANCE_FAIL",
            "paper_ready": False, "physical_hardware": "2x RTX4090 single-node",
            "simulator_namespace": "rtx4090_calibration", "main_a100_profile_unchanged": True,
            "fit_job": "141298", "holdout_job": "141411", "fit_sample_count": 80,
            "holdout_sample_count": 80, "holdout_cell_count": 16,
            "preregistered_protocol_sha256": sha256(PREREG), "evaluation_protocol_sha256": sha256(EVAL_PROTOCOL),
            "fit_input_sha256": sha256(train_path), "holdout_input_sha256": sha256(holdout_path),
            "figure_pdf": str(pdf.relative_to(stage)), "figure_png": str(png.relative_to(stage)),
            "source_files": [
                "data/training_samples.csv", "data/fitted_rtx4090_service_curves.json",
                "data/holdout_samples.csv", "data/holdout_errors_by_path_payload.csv", "data/direction_summary.csv",
            ],
            "evidence_boundary": evaluation["evidence_boundary"],
            "caption": "Frozen direction-specific RTX4090 service curves fitted only on job141298 are compared with independent job141411 physical repeats. Points and intervals use all five repeats per path/payload cell. The preregistered 10% all-cell MAPE gate is reported without refitting or exclusions; this is calibration-only evidence and does not calibrate the A100 40 GiB main simulation profile.",
        }
        write_json(stage / "manifest.json", manifest)
        files = sorted(path for path in stage.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
        (stage / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(stage)}\n" for path in files))
        if not all(integrity_checks.values()):
            raise RuntimeError("postprocess integrity audit failed")
        os.replace(stage, output)
    except Exception:
        print(f"staging preserved at {stage}")
        raise
    print(json.dumps({
        "output": str(output), "audit_status": "PASS", "calibration_acceptance": "PASS" if acceptance_pass else "FAIL",
        "overall_mape_percent": overall_mape, "failed_cells": failed_cells,
    }, indent=2))


if __name__ == "__main__":
    main()
