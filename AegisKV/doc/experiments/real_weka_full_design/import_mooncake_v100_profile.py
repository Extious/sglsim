#!/usr/bin/env python3
"""Create a fail-closed SGLsim config from one pinned Mooncake V100 profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for search_root in (REPO_ROOT, REPO_ROOT / "python"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS

from AegisKV.experiments.real_weka_full_design.run import (
    V100_AIMD_PRESSURE_CALIBRATION_EVIDENCE,
    V100_AIMD_PRESSURE_MODE,
    V100_COMPATIBILITY_IDENTITY,
    V100_HARDWARE_LABEL,
    V100_PROFILE_METRICS,
    V100_PROFILE_SEMANTICS,
    V100_RUNTIME_CONSUMPTION_CONTRACT,
    calibrated_runtime_config,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def import_profile(
    *,
    template_path: Path,
    profile_path: Path,
    source_manifest_path: Path,
    source_verification_path: Path,
    pilot_runtime_gate_path: Path,
    aimd_pressure_calibration_path: Path | None = None,
    context_audit_path: Path,
    selected_scale_label: str,
    maximum_calibrated_context_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    template_path = template_path.resolve()
    profile_path = profile_path.resolve()
    source_manifest_path = source_manifest_path.resolve()
    source_verification_path = source_verification_path.resolve()
    pilot_runtime_gate_path = pilot_runtime_gate_path.resolve()
    aimd_pressure_calibration_path = (
        aimd_pressure_calibration_path.resolve()
        if aimd_pressure_calibration_path is not None
        else None
    )
    context_audit_path = context_audit_path.resolve()
    if maximum_calibrated_context_tokens <= 0:
        raise ValueError("maximum calibrated context tokens must be positive")
    template = _object(template_path)
    profile = _object(profile_path)
    manifest = _object(source_manifest_path)
    verification = _object(source_verification_path)
    context_audit = _object(context_audit_path)
    if template.get("design_semantics") != DESIGN_SEMANTICS:
        raise ValueError("template does not use the current AegisKV design")
    if template.get("calibration", {}).get("status") == "READY":
        raise ValueError("refusing to replace a READY template calibration")
    metrics = profile.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(V100_PROFILE_METRICS):
        raise ValueError("Mooncake profile lacks metrics")
    for name in V100_PROFILE_METRICS:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Mooncake profile metric {name} is invalid") from exc
        if not math.isfinite(value) or (
            value < 0.0
            if name == "recompute_fixed_overhead_ms"
            else value <= 0.0
        ):
            raise ValueError(f"Mooncake profile metric {name} is invalid")
    manifest_ref = _artifact(source_manifest_path)
    verification_ref = _artifact(source_verification_path)
    pilot_runtime_gate_ref = _artifact(pilot_runtime_gate_path)
    profile_ref = _artifact(profile_path)
    context_audit_ref = _artifact(context_audit_path)
    config_manifest_ref = profile.get("config_manifest")
    config_verification_ref = profile.get("config_verification")
    if not isinstance(config_manifest_ref, dict) or not isinstance(
        config_verification_ref, dict
    ):
        raise ValueError("Mooncake profile lacks its verified config lineage")
    compatibility = profile.get("compatibility_identity")
    if not (
        profile.get("schema_version") == 2
        and profile.get("evidence_class") == "physical_measurement"
        and profile.get("status") == "passed"
        and profile.get("failures") == []
        and profile.get("profile_semantics") == V100_PROFILE_SEMANTICS
        and profile.get("hardware_label") == V100_HARDWARE_LABEL
        and compatibility == V100_COMPATIBILITY_IDENTITY
        and profile.get("runtime_consumption_contract")
        == V100_RUNTIME_CONSUMPTION_CONTRACT
        and profile.get("source_manifest", {}).get("sha256")
        == manifest_ref["sha256"]
        and profile.get("source_manifest", {}).get("entries_sha256")
        == manifest.get("entries_sha256")
        and manifest.get("status") == "PINNED"
        and verification.get("status") == "PASS"
        and verification.get("failures") == []
        and verification.get("manifest", {}).get("sha256")
        == manifest_ref["sha256"]
        and verification.get("entries_sha256") == manifest.get("entries_sha256")
    ):
        raise ValueError("Mooncake profile provenance contract did not pass")
    if maximum_calibrated_context_tokens > int(compatibility["max_model_len"]):
        raise ValueError("calibrated context exceeds the physical deployment")
    physical_layout = {
        "tp_size": int(profile["layout"]["tp_size"]),
        "pp_size": int(profile["layout"]["pp_size"]),
        "dp_size": int(profile["layout"]["dp_size"]),
    }
    template_layout = {
        "tp_size": int(template.get("cluster", {}).get("tp_size", 0)),
        "pp_size": int(template.get("cluster", {}).get("pp_size", 0)),
        "dp_size": int(template.get("cluster", {}).get("dp_size", 0)),
    }
    if template_layout != physical_layout:
        raise ValueError(
            "V100 primary simulation layout must match the physical profile; "
            "cross-layout sensitivity requires its own calibrated profile"
        )
    aimd_pressure = template.get("aimd", {})
    if not (
        aimd_pressure.get("pressure_mode")
        == "active_source_request_count_proxy"
        and aimd_pressure.get("pressure_calibrated") is False
        and int(aimd_pressure.get("pressure_active_source_requests", 0)) > 0
    ):
        raise ValueError(
            "V100 template must explicitly disclose its uncalibrated "
            "active-source-request AIMD pressure proxy"
        )
    pressure_calibration_ref: dict[str, Any] | None = None
    pressure_calibration: dict[str, Any] | None = None
    if aimd_pressure_calibration_path is not None:
        pressure_calibration_ref = _artifact(aimd_pressure_calibration_path)
        pressure_calibration = _object(aimd_pressure_calibration_path)
        if not (
            pressure_calibration.get("schema_version") == 1
            and pressure_calibration.get("evidence_class")
            == V100_AIMD_PRESSURE_CALIBRATION_EVIDENCE
            and pressure_calibration.get("status") == "PASS"
            and pressure_calibration.get("failures") == []
            and pressure_calibration.get("hardware_label") == V100_HARDWARE_LABEL
            and pressure_calibration.get("pressure_mode")
            == V100_AIMD_PRESSURE_MODE
            and pressure_calibration.get(
                "sglsim_quantitative_aimd_pressure_authorized"
            )
            is True
            and pressure_calibration.get("formal_scale_submission_authorized")
            is False
            and pressure_calibration.get("paper_performance_claim_authorized")
            is False
            and pressure_calibration.get("paper_ready") is False
            and isinstance(pressure_calibration.get("pool_models"), dict)
            and len(pressure_calibration["pool_models"]) == 4
        ):
            raise ValueError("AIMD pressure calibration is not a PASS artifact")
    selected_rows = [
        row
        for row in context_audit.get("scale_points", [])
        if isinstance(row, dict) and row.get("label") == selected_scale_label
    ]
    audit_model = context_audit.get("model")
    audit_trace = context_audit.get("trace")
    expected_trace = template.get("dataset", {}).get("expected_full_profile")
    if not (
        context_audit.get("status") == "PASS"
        and context_audit.get("design_semantics") == DESIGN_SEMANTICS
        and isinstance(audit_model, dict)
        and isinstance(audit_trace, dict)
        and {
            name: int(audit_trace[name])
            for name in ("request_count", "input_tokens", "output_tokens")
        }
        == {
            name: int(expected_trace[name])
            for name in ("request_count", "input_tokens", "output_tokens")
        }
        and audit_model.get("revision")
        == f"Qwen/Qwen3-8B@{profile.get('model_revision')}"
        and int(audit_model.get("kv_bytes_per_token_all_layers", 0))
        == int(template.get("model", {}).get("kv_bytes_per_token", -1))
        and len(selected_rows) == 1
    ):
        raise ValueError("context projection audit contract did not pass")
    selected_scale = selected_rows[0]
    if not (
        selected_scale.get("readiness")
        == "ELIGIBLE_AFTER_VALID_PHYSICAL_PROFILE"
        and selected_scale.get("role")
        == "preregistered_primary_after_physical_profile_gate"
        and selected_scale.get("performance_results_present") is False
        and int(selected_scale.get("requests_above_model_context", -1)) == 0
        and int(
            selected_scale.get(
                "requests_above_physical_calibration_context", -1
            )
        )
        == 0
        and int(selected_scale["context_tokens"]["max"])
        <= maximum_calibrated_context_tokens
    ):
        raise ValueError("selected context scale is not primary-gate eligible")
    imported_model = {
        **template["model"],
        "revision": str(profile["model"]),
        "model_snapshot_revision": str(profile["model_revision"]),
        "serving_dtype": str(compatibility["serving_dtype"]),
        "kv_cache_dtype": str(compatibility["kv_cache_dtype"]),
        "kv_abi": str(compatibility["kv_abi"]),
        "block_size": int(compatibility["block_size"]),
        "maximum_context_tokens": int(audit_model["maximum_context_tokens"]),
        "mapping_note": (
            "Real-Weka topology/timing with a preregistered token projection; "
            "KV identity is the physical Qwen3-8B FP16 vLLM deployment."
        ),
    }
    token_projection = {
        "status": "PASS",
        "selected_scale_label": selected_scale_label,
        "factor": float(selected_scale["factor"]),
        "source_trace_sha256": str(audit_trace["sha256"]),
        "maximum_projected_context_tokens": int(
            selected_scale["context_tokens"]["max"]
        ),
        "context_audit": context_audit_ref,
    }
    calibration = {
        "status": "READY",
        "bundle_semantics": V100_PROFILE_SEMANTICS,
        "hardware_label": V100_HARDWARE_LABEL,
        "physical_profile": profile_ref,
        "source_manifest": manifest_ref,
        "source_verification": verification_ref,
        "config_manifest": dict(config_manifest_ref),
        "config_verification": dict(config_verification_ref),
        "pilot_runtime_gate": pilot_runtime_gate_ref,
        "compatibility_identity": dict(compatibility),
        "peer_hbm_protection_setup_ms": 0.0,
        "direct_dram_protection_setup_ms": 0.0,
        "hbm_recovery_setup_ms": 0.0,
        "dram_recovery_setup_ms": 0.0,
        "maximum_calibrated_context_tokens": maximum_calibrated_context_tokens,
        "source_layout": physical_layout,
        "simulation_layout": dict(template["cluster"]),
        "layout_extrapolation": False,
    }
    if pressure_calibration_ref is not None:
        calibration["aimd_pressure_calibration"] = pressure_calibration_ref
    imported_aimd = dict(template["aimd"])
    if pressure_calibration is not None:
        imported_aimd.update(
            {
                "pressure_mode": V100_AIMD_PRESSURE_MODE,
                "pressure_calibrated": True,
                "physical_pressure_pool_models": pressure_calibration[
                    "pool_models"
                ],
                "pressure_note": (
                    "Same-campaign physical CUDA-headroom OR successful "
                    "Peer-HBM transfer-duration model; artifact identity is "
                    "verified by the independent consumer."
                ),
            }
        )
        imported_aimd.pop("pressure_active_source_requests", None)
    imported = {
        **template,
        "experiment_id": (
            f"real-weka-full-aegiskv-v100-{profile_ref['sha256'][:12]}"
        ),
        "model": imported_model,
        "workload": {**template["workload"], "token_projection": token_projection},
        "aimd": imported_aimd,
        "calibration": calibration,
    }
    # Re-run the independent consumer-side gate before writing any output.
    runtime, provenance = calibrated_runtime_config(imported)
    import_manifest = {
        "schema_version": 1,
        "evidence_class": "sglsim_v100_physical_profile_import",
        "status": "PASS",
        "design_semantics": DESIGN_SEMANTICS,
        "hardware_label": V100_HARDWARE_LABEL,
        "template": _artifact(template_path),
        "physical_profile": profile_ref,
        "source_manifest": manifest_ref,
        "source_verification": verification_ref,
        "config_manifest": dict(config_manifest_ref),
        "config_verification": dict(config_verification_ref),
        "pilot_runtime_gate": pilot_runtime_gate_ref,
        "context_projection_audit": context_audit_ref,
        "selected_context_scale": dict(selected_scale),
        "runtime_config": {
            "hbm_copy_bytes_per_ms": runtime.hbm_copy.bandwidth_bytes_per_ms,
            "dram_copy_bytes_per_ms": runtime.dram_copy.bandwidth_bytes_per_ms,
            "hbm_load_ms_per_token": runtime.hbm_load.ms_per_token,
            "dram_load_ms_per_token": runtime.dram_load.ms_per_token,
            "recompute_ms_per_token": runtime.recompute.ms_per_token,
            "recompute_fixed_overhead_ms": runtime.recompute.setup_ms,
        },
        "runtime_consumption_contract": dict(
            profile["runtime_consumption_contract"]
        ),
        "calibration_provenance": provenance,
        "runtime_alignment": {
            "four_condition_physical_gate_passed": True,
            "source_owned_hbm_dram_paths_independently_verified": True,
            "hbm_dram_timing_overlap_required": False,
            "config_manifest_and_verification_lineage_verified": True,
            "runtime_admission_decision_wire_values_verified": True,
            "four_gpu_clean_telemetry_verified": True,
            "aimd_alpha_response_and_direct_reclaim_exercised": True,
            "independent_peer_hbm_and_dram_recovery_exercised": True,
            "all_four_independent_residency_outcomes_exercised": True,
            "hbm_only_direct_reclaim_without_dram_prerequisite_exercised": True,
            "pilot_campaign_id": provenance["pilot_runtime_gate"][
                "campaign_id"
            ],
            "quantitative_aimd_simulation_authorized": bool(
                provenance["aimd_pressure"][
                    "sglsim_quantitative_aimd_pressure_authorized"
                ]
            ),
            "quantitative_aimd_claim_authorized": False,
            "paper_performance_claim_authorized": False,
        },
        "aimd_pressure_model": {
            "mode": provenance["aimd_pressure"]["mode"],
            "calibrated": provenance["aimd_pressure"]["calibrated"],
            "artifact": provenance["aimd_pressure"].get("artifact"),
            "active_source_request_threshold": provenance["aimd_pressure"].get(
                "active_source_request_threshold"
            ),
            "quantitative_aimd_simulation_authorized": bool(
                provenance["aimd_pressure"][
                    "sglsim_quantitative_aimd_pressure_authorized"
                ]
            ),
            "quantitative_aimd_claim_authorized": False,
        },
        "evidence_boundary": (
            "This imports physical coefficients into a metadata-only simulation; "
            "it is not physical end-to-end Mooncake evidence. "
            + (
                "The AIMD pressure input is calibrated from same-campaign "
                "pre-control CUDA residency and Peer-HBM duration samples; this "
                "authorizes quantitative pressure behavior in same-layout "
                "metadata replay, but not a paper performance claim."
                if pressure_calibration is not None
                else "The AIMD pressure input remains an uncalibrated "
                "active-source-request proxy. Passing the four-condition "
                "physical runtime gate authorizes diagnostic replay only, not "
                "a paper performance or quantitative AIMD claim."
            )
        ),
        "paper_ready": False,
        "formal_ready": False,
    }
    return imported, import_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--physical-profile", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--pilot-runtime-gate", type=Path, required=True)
    parser.add_argument("--aimd-pressure-calibration", type=Path)
    parser.add_argument("--context-audit", type=Path, required=True)
    parser.add_argument("--selected-scale-label", required=True)
    parser.add_argument("--maximum-calibrated-context-tokens", type=int, default=4096)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_config.resolve(), args.output_manifest.resolve())
    if any(path.exists() for path in outputs):
        raise FileExistsError("V100 import outputs are immutable")
    config, manifest = import_profile(
        template_path=args.template,
        profile_path=args.physical_profile,
        source_manifest_path=args.source_manifest,
        source_verification_path=args.source_verification,
        pilot_runtime_gate_path=args.pilot_runtime_gate,
        aimd_pressure_calibration_path=args.aimd_pressure_calibration,
        context_audit_path=args.context_audit,
        selected_scale_label=args.selected_scale_label,
        maximum_calibrated_context_tokens=args.maximum_calibrated_context_tokens,
    )
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest["output_config"] = _artifact(outputs[0])
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
