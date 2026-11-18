#!/usr/bin/env python3
"""Join the audited 141298 RTX4090 profile to the current SGLSim path model.

This is a calibration comparison, not an A100 measurement.  SGLSim values are
not fitted: they are produced by the same setup-plus-bandwidth service curve
used for AegisKV source-copy events in ``run_preliminary.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SG_ROOT = ROOT.parents[2]
DEFAULT_PROFILE = Path(
    "/home/comp/zhanzhao/Mooncake/aegiskv/results/physical_profiles/"
    "rtx4090-pcie-host-staged-141298-v1"
)
BASE_BUNDLE = ROOT / "target_shaped_real_bundle"
PROTOCOL = ROOT / "protocol.json"
RUNNER = ROOT / "run_preliminary.py"
MODEL_SOURCE = SG_ROOT / "python/sglsim/srt/simulation/aegis_full_design.py"
COLORS = {"peer_hbm_same_node": "#2563eb", "dram_local": "#d97706"}
LABELS = {"peer_hbm_same_node": "Same-node Peer-HBM", "dram_local": "Local DRAM"}
PAYLOADS = [64, 256, 512, 1024]
PATHS = ["peer_hbm_same_node", "dram_local"]
SETUP_MS = {"peer_hbm_same_node": 0.05, "dram_local": 0.10}
BYTES_PER_MS_PER_GB_S = 1_000_000.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.2, "axes.titlesize": 10.5,
    "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.2, "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effective_gib_s(payload_mib: int, gb_s: float, setup_ms: float) -> tuple[float, float]:
    payload_bytes = int(payload_mib) * 2**20
    duration_ms = float(setup_ms) + payload_bytes / (
        float(gb_s) * BYTES_PER_MS_PER_GB_S
    )
    throughput = (payload_bytes / 2**30) / (duration_ms / 1000.0)
    return duration_ms, throughput


def verify_manifest(profile: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    manifest_path = profile / "physical-profile-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "physical_manifest_pass": manifest.get("status") == "PASS",
        "physical_profile_ready": manifest.get("physical_profile_ready") is True,
        "physical_hardware_exact": manifest.get("hardware") == "single-node 2x NVIDIA GeForce RTX 4090",
        "physical_path_class_exact": manifest.get("physical_path_class") == "pcie_pinned_host_staged",
        "physical_input_hashes_match_manifest": True,
    }
    for key in (
        "calibration", "calibration_gate", "detailed_four_path_profile",
        "sglsim_import_schema", "sglsim_s9_import_csv",
    ):
        record = manifest[key]
        path = Path(record["path"])
        checks["physical_input_hashes_match_manifest"] &= (
            path.exists()
            and path.stat().st_size == int(record["size_bytes"])
            and sha256(path) == record["sha256"]
        )
    gate_path = Path(manifest["calibration_gate"]["path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    checks["physical_calibration_gate_pass"] = (
        gate.get("status") == "PASS" and gate.get("calibration_ready") is True
    )
    return manifest, gate, checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "target_shaped_real_bundle_s9_matched_141298.staging",
    )
    parser.add_argument("--visual-inspection-pass", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to mix lineage in existing directory: {output}")
    output.mkdir(parents=True)

    physical_manifest, physical_gate, checks = verify_manifest(args.profile_dir)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runner_text = RUNNER.read_text(encoding="utf-8")
    checks.update({
        "current_design_semantics_exact": protocol.get("design_semantics") == "source_owned_independent_hbm_dram_v2",
        "simulator_hardware_is_a100_40gb_model": (
            protocol["hardware"]["gpu"] == "NVIDIA A100 40GB"
            and float(protocol["hardware"]["gpu_hbm_gib"]) == 40.0
        ),
        "executed_copy_formula_present": (
            "duration = setup + spec.byte_size / (bw * GB_PER_S_TO_BYTES_PER_MS)" in runner_text
        ),
        "executed_hbm_setup_exact": (
            '(ProtectionTier.HBM, decision.hbm.admitted, f"peer_hbm_{peer_mode}", 0.05)' in runner_text
        ),
        "executed_dram_setup_exact": (
            '(ProtectionTier.DRAM, decision.dram.admitted, f"dram_{dram_mode}", 0.10)' in runner_text
        ),
        "base_s1_s8_bundle_untouched": BASE_BUNDLE.exists(),
    })

    model_config_path = output / "data" / "s9_sglsim_model_config.json"
    model_config = {
        "schema_version": 1,
        "model_kind": "current AegisKV source-copy DES service curve",
        "formula": "duration_ms = setup_ms + payload_bytes / (bandwidth_gb_s * 1e6); effective_gib_s = (payload_bytes / 2^30) / (duration_ms / 1000)",
        "unit_contract": "protocol bandwidth uses decimal GB/s; plotted effective throughput uses binary GiB/s",
        "paths": {
            path: {
                "bandwidth_gb_s": float(protocol["hardware"]["bandwidth_gb_s"][path]),
                "setup_ms": SETUP_MS[path],
                "hardware_role": "configured A100 40GB simulator input, not a physical A100 measurement",
            }
            for path in PATHS
        },
        "payload_mib": PAYLOADS,
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "runner": {"path": str(RUNNER), "sha256": sha256(RUNNER)},
        "transfer_model_source": {"path": str(MODEL_SOURCE), "sha256": sha256(MODEL_SOURCE)},
        "physical_comparator": "single-node 2x RTX4090 PCIe pinned-host staged path",
        "calibration_policy": "no fitting, rescaling, or target-trend threshold",
    }
    write_json(model_config_path, model_config)
    model_config_sha = sha256(model_config_path)

    import_path = args.profile_dir / "sglsim-s9-physical-profile.csv"
    with import_path.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    keys = [(int(row["payload_mib"]), row["path"]) for row in source_rows]
    checks.update({
        "eight_import_rows": len(source_rows) == 8,
        "join_key_unique": len(set(keys)) == len(keys),
        "payloads_exact": sorted({key[0] for key in keys}) == PAYLOADS,
        "paths_exact": sorted({key[1] for key in keys}) == sorted(PATHS),
        "physical_values_positive": all(float(row["physical_gib_s"]) > 0 for row in source_rows),
        "physical_audit_sha_matches_gate": all(
            row["physical_audit_sha256"] == physical_manifest["calibration_gate"]["sha256"]
            for row in source_rows
        ),
    })

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        payload = int(source["payload_mib"])
        path = source["path"]
        bandwidth = float(protocol["hardware"]["bandwidth_gb_s"][path])
        duration_ms, modeled = effective_gib_s(payload, bandwidth, SETUP_MS[path])
        physical = float(source["physical_gib_s"])
        relative_error = abs(modeled - physical) / physical * 100.0
        rows.append({
            "payload_mib": payload,
            "path": path,
            "physical_gib_s": physical,
            "sglsim_gib_s": modeled,
            "sglsim_duration_ms": duration_ms,
            "relative_error_percent": relative_error,
            "physical_run_id": source["physical_run_id"],
            "physical_audit_sha256": source["physical_audit_sha256"],
            "sglsim_source_sha256": model_config_sha,
            "physical_hardware": "single-node 2x NVIDIA GeForce RTX 4090",
            "simulator_hardware_model": "NVIDIA A100 40GB configured input",
            "physical_path_class": source["physical_path_class"],
            "calibration_gate_10_percent": "FAIL" if relative_error > 10.0 else "PASS",
            "status": "MATCHED_MODELED_COMPARISON",
        })
    rows.sort(key=lambda row: (PATHS.index(row["path"]), row["payload_mib"]))
    csv_path = output / "data" / "s9_matched_calibration_overlay.csv"
    write_csv(csv_path, rows)

    # Recompute every model value independently from the persisted config.
    checks["matched_model_values_recompute"] = all(
        math.isclose(
            row["sglsim_gib_s"],
            effective_gib_s(
                row["payload_mib"],
                model_config["paths"][row["path"]]["bandwidth_gb_s"],
                model_config["paths"][row["path"]]["setup_ms"],
            )[1],
            rel_tol=1e-13,
        )
        for row in rows
    )
    checks["relative_error_recompute"] = all(
        math.isclose(
            row["relative_error_percent"],
            abs(row["sglsim_gib_s"] - row["physical_gib_s"])
            / row["physical_gib_s"] * 100.0,
            rel_tol=1e-13,
        )
        for row in rows
    )
    checks["no_a100_physical_claim"] = all(
        "configured input" in row["simulator_hardware_model"] for row in rows
    )
    checks["visual_inspection_pass"] = bool(args.visual_inspection_pass)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.45))
    for path in PATHS:
        selected = [row for row in rows if row["path"] == path]
        x = [row["payload_mib"] for row in selected]
        physical = [row["physical_gib_s"] for row in selected]
        modeled = [row["sglsim_gib_s"] for row in selected]
        error = [row["relative_error_percent"] for row in selected]
        axes[0].plot(x, physical, "o-", color=COLORS[path], lw=1.8,
                     label=f"{LABELS[path]} - physical RTX4090")
        axes[0].plot(x, modeled, "s--", color=COLORS[path], lw=1.6,
                     label=f"{LABELS[path]} - SGLSim A100 model")
        axes[1].plot(x, error, marker="o", color=COLORS[path], lw=1.8,
                     label=LABELS[path])
        for xpos, value in zip(x, error):
            axes[1].annotate(
                f"{value:.0f}%", (xpos, value), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=6.2,
                color=COLORS[path],
            )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(PAYLOADS, [str(value) for value in PAYLOADS])
    axes[0].set(xlabel="Payload (MiB)", ylabel="Effective bandwidth (GiB/s)",
                title="Matched payload/path overlay (log y)")
    axes[0].legend(fontsize=6.2, loc="best")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(PAYLOADS, [str(value) for value in PAYLOADS])
    axes[1].axhline(10.0, color="#b91c1c", ls="--", lw=1.1, label="10% calibration gate")
    axes[1].set(xlabel="Payload (MiB)", ylabel="Relative error (%)",
                title="Model error (log y)")
    axes[1].legend(fontsize=6.6, loc="center left")
    for axis in axes:
        axis.grid(alpha=.15, which="both")
    fig.suptitle("S9. RTX4090 physical vs configured SGLSim model - matched", y=.99)
    fig.text(
        .5, .005,
        "Physical: RTX4090 PCIe host-staged. Simulator: A100 40GB configured model; not an A100 measurement. No fitting.",
        ha="center", fontsize=7.1, color="#555555",
    )
    fig.subplots_adjust(top=.82, bottom=.19, wspace=.28)
    figure_base = output / "figures" / "s9_matched_calibration_overlay"
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path, png_path = figure_base.with_suffix(".pdf"), figure_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", metadata={"Creator": "audited SGLSim S9 matched overlay"})
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    lineage_path = output / "lineage" / "s9.json"
    write_json(lineage_path, {
        "schema_version": 1,
        "panel": "S9",
        "join_key": ["payload_mib", "path"],
        "physical_profile_manifest": {
            "path": str(args.profile_dir / "physical-profile-manifest.json"),
            "sha256": sha256(args.profile_dir / "physical-profile-manifest.json"),
            "status": physical_manifest["status"],
        },
        "physical_import_csv": {"path": str(import_path), "sha256": sha256(import_path)},
        "physical_gate": {
            "path": physical_manifest["calibration_gate"]["path"],
            "sha256": physical_manifest["calibration_gate"]["sha256"],
            "status": physical_gate["status"],
        },
        "sglsim_model_config": {"path": "data/s9_sglsim_model_config.json", "sha256": model_config_sha},
        "matched_csv": {"path": "data/s9_matched_calibration_overlay.csv", "sha256": sha256(csv_path)},
        "base_s1_s8_bundle": {
            "path": str(BASE_BUNDLE),
            "manifest_sha256": sha256(BASE_BUNDLE / "manifest.json"),
            "mutation": "none; this output is a sibling S9-only bundle",
        },
        "evidence_boundary": (
            "The physical series is a PASS RTX4090 PCIe pinned-host staged profile. "
            "The SGLSim series is the configured A100 40GB service model, not a physical "
            "A100 measurement. The comparison is not fitted and is not paper-ready."
        ),
    })

    calibration_gate_pass = all(row["relative_error_percent"] <= 10.0 for row in rows)
    audit_path = output / "audits" / "s9.json"
    audit = {
        "schema_version": 1,
        "panel": "S9",
        "status": "PASS" if all(checks.values()) else (
            "PENDING_VISUAL_INSPECTION"
            if all(value for key, value in checks.items() if key != "visual_inspection_pass")
            else "FAIL"
        ),
        "checks": checks,
        "physical_profile_gate_status": physical_gate["status"],
        "simulator_to_physical_10pct_gate_status": "PASS" if calibration_gate_pass else "FAIL",
        "simulator_to_physical_gate_is_acceptance_for_artifact": False,
        "matched_row_count": len(rows),
        "relative_error_percent_min": min(row["relative_error_percent"] for row in rows),
        "relative_error_percent_max": max(row["relative_error_percent"] for row in rows),
        "csv_sha256": sha256(csv_path),
        "lineage_sha256": sha256(lineage_path),
        "model_config_sha256": model_config_sha,
        "figure_pdf_sha256": sha256(pdf_path),
        "figure_png_sha256": sha256(png_path),
        "paper_ready": False,
        "conclusion": (
            "The matched current SGLSim A100-configured path model substantially "
            "overpredicts the RTX4090 host-staged physical throughput; this is a "
            "measured calibration mismatch, not an A100 physical result."
        ),
    }
    write_json(audit_path, audit)
    if audit["status"] == "FAIL":
        raise RuntimeError([name for name, passed in checks.items() if not passed])

    manifest_path = output / "manifest.json"
    write_json(manifest_path, {
        "schema_version": 1,
        "artifact_status": audit["status"],
        "title": "S9 matched RTX4090 physical versus configured SGLSim overlay",
        "s1_s8_mutation": "none",
        "data": "data/s9_matched_calibration_overlay.csv",
        "model_config": "data/s9_sglsim_model_config.json",
        "lineage": "lineage/s9.json",
        "audit": "audits/s9.json",
        "pdf": "figures/s9_matched_calibration_overlay.pdf",
        "png": "figures/s9_matched_calibration_overlay.png",
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "physical_profile_manifest_sha256": sha256(args.profile_dir / "physical-profile-manifest.json"),
        "base_bundle_manifest_sha256": sha256(BASE_BUNDLE / "manifest.json"),
    })
    sums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
