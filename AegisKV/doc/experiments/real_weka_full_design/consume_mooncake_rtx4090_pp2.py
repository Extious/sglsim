#!/usr/bin/env python3
"""Validate an RTX4090 PP2 import; never launch SGLSim replay or figures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
for search_root in (REPO_ROOT, REPO_ROOT / "python"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from sglsim.srt.simulation.aegis_full_design import (
    AegisFullDesignConfig,
    CalibratedTransferPath,
)
from sglsim.srt.simulation.aegis_independent_tiers import RecoveryTiming

from AegisKV.experiments.real_weka_full_design.rtx4090_pp2_contract import (
    DESIGN_SEMANTICS,
    HARDWARE_LABEL,
    IMPORT_EVIDENCE_CLASS,
    InputPaths,
    PROFILE_EVIDENCE_CLASS,
    PROFILE_METRICS,
    PROFILE_SEMANTICS,
    RUNTIME_CONSUMPTION_CONTRACT,
    TCP_DIAGNOSTIC_ACK,
    build_profile,
    load_object,
    sha256_file,
)


def _artifact_path(record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
        raise ValueError(f"{label} artifact record is incomplete")
    path = Path(str(record["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"{label} artifact hash did not verify")
    if "size_bytes" in record and path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{label} artifact size did not verify")
    return path


def validate_import(
    *, profile_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the normalized profile from the pinned physical inputs."""

    profile_path = profile_path.resolve()
    manifest_path = manifest_path.resolve()
    profile = load_object(profile_path)
    manifest = load_object(manifest_path)
    output_profile = manifest.get("output_profile")
    verified_profile_path = _artifact_path(
        output_profile, label="RTX4090 imported profile"
    )
    profile_file_sha256 = sha256_file(profile_path)
    expected_directory_name = f"rtx4090_pp2_{profile_file_sha256[:16]}"
    if not (
        verified_profile_path == profile_path
        and profile_path.parent == manifest_path.parent
        and profile_path.parent.name == expected_directory_name
        and manifest.get("schema_version") == 1
        and manifest.get("evidence_class") == IMPORT_EVIDENCE_CLASS
        and manifest.get("status") == "PASS"
        and manifest.get("design_semantics") == DESIGN_SEMANTICS
        and manifest.get("profile_semantics") == PROFILE_SEMANTICS
        and manifest.get("hardware_label") == HARDWARE_LABEL
        and manifest.get("output_directory_name") == expected_directory_name
        and manifest.get("output_directory_identity_sha256")
        == profile_file_sha256
        and manifest.get("lineage_input_sha256")
        == profile.get("lineage_input_sha256")
        and manifest.get("runtime_consumption_contract")
        == RUNTIME_CONSUMPTION_CONTRACT
        and manifest.get("transport_contract")
        == profile.get("transport_observation")
        and manifest.get("recovery_transport_contract")
        == profile.get("recovery_transport_observation")
        and manifest.get("p2p_topology_diagnostic")
        == profile.get("p2p_topology_diagnostic")
        and manifest.get("evidence_role")
        == "preliminary_targeted_physical_diagnostic"
        and manifest.get("default_replay_authorized") is False
        and manifest.get("formal_replay_authorized") is False
        and manifest.get("figure_generation_authorized") is False
        and manifest.get("paper_performance_claim_authorized") is False
        and manifest.get("mixed_v100_or_a100_lineage") is False
        and manifest.get("formal_ready") is False
        and manifest.get("paper_ready") is False
    ):
        raise ValueError("RTX4090 import manifest contract did not pass")
    if not (
        profile.get("schema_version") == 1
        and profile.get("evidence_class") == PROFILE_EVIDENCE_CLASS
        and profile.get("status") == "PASS"
        and profile.get("design_semantics") == DESIGN_SEMANTICS
        and profile.get("profile_semantics") == PROFILE_SEMANTICS
        and profile.get("hardware_label") == HARDWARE_LABEL
        and profile.get("runtime_consumption_contract")
        == RUNTIME_CONSUMPTION_CONTRACT
        and profile.get("transport_observation", {}).get("transfer_protocol")
        == "tcp"
        and profile.get("transport_observation", {}).get("physical_path")
        == "registered_gpu_memory_over_mooncake_tcp"
        and profile.get("transport_observation", {}).get(
            "direct_cuda_p2p_ready"
        )
        is False
        and profile.get("transport_observation", {}).get(
            "exact_d2d_p2p_contract_ready"
        )
        is False
        and profile.get("transport_observation", {}).get(
            "simulator_rate_default_eligible"
        )
        is False
        and profile.get("recovery_transport_observation", {}).get(
            "transfer_protocol"
        )
        == "tcp"
        and profile.get("recovery_transport_observation", {}).get(
            "source_owned_transfers_only"
        )
        is True
        and profile.get("runtime_authorization", {}).get(
            "formal_replay_authorized"
        )
        is False
        and profile.get("formal_ready") is False
        and profile.get("paper_ready") is False
    ):
        raise ValueError("RTX4090 normalized profile contract did not pass")

    input_records = manifest.get("lineage_inputs")
    if not isinstance(input_records, dict) or input_records != profile.get(
        "lineage_inputs"
    ):
        raise ValueError("RTX4090 import/profile lineage inputs differ")
    try:
        input_paths = InputPaths(
            **{
                name: _artifact_path(record, label=f"RTX4090 input {name}")
                for name, record in input_records.items()
            }
        )
    except TypeError as exc:
        raise ValueError("RTX4090 lineage input field set changed") from exc
    rebuilt = build_profile(input_paths)
    if rebuilt != profile:
        raise ValueError("RTX4090 profile is not reproducible from its inputs")
    return profile, manifest


def materialize_diagnostic_runtime(
    *,
    profile_path: Path,
    manifest_path: Path,
    acknowledgement: str | None = None,
) -> tuple[AegisFullDesignConfig, dict[str, Any]]:
    """Construct metadata timing objects only after a transport acknowledgement.

    This does not run a trace or authorize a performance claim.  The explicit
    acknowledgement prevents registered-HBM-over-TCP rates from silently
    becoming a default CUDA-P2P simulator calibration.
    """

    profile, manifest = validate_import(
        profile_path=profile_path, manifest_path=manifest_path
    )
    if acknowledgement != TCP_DIAGNOSTIC_ACK:
        raise ValueError(
            "RTX4090 registered-HBM-over-TCP rates are diagnostic-only; "
            f"explicit acknowledgement {TCP_DIAGNOSTIC_ACK!r} is required"
        )
    metrics = profile.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(PROFILE_METRICS):
        raise ValueError("RTX4090 profile metric set changed")
    normalized: dict[str, float] = {}
    for name in PROFILE_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value) or (
            value < 0.0
            if name == "recompute_fixed_overhead_ms"
            else value <= 0.0
        ):
            raise ValueError(f"RTX4090 metric {name} is invalid")
        normalized[name] = value

    capacity = profile["physical_capacities"]
    params = profile["runtime_parameters"]
    setup = profile["setup_ms"]
    kv_bytes = int(params["kv_bytes_per_token"])
    runtime = AegisFullDesignConfig(
        hbm_capacity_bytes=int(capacity["hbm_bytes_per_logical_target"]),
        dram_capacity_bytes=int(capacity["dram_bytes_per_logical_target"]),
        kv_bytes_per_token=kv_bytes,
        hbm_copy=CalibratedTransferPath(
            setup_ms=float(setup["peer_hbm_protection"]),
            bandwidth_bytes_per_ms=normalized[
                "peer_hbm_protection_effective_bytes_per_ms"
            ],
            provenance=str(profile_path.resolve()),
        ),
        dram_copy=CalibratedTransferPath(
            setup_ms=float(setup["direct_dram_protection"]),
            bandwidth_bytes_per_ms=normalized[
                "direct_dram_protection_effective_bytes_per_ms"
            ],
            provenance=str(profile_path.resolve()),
        ),
        recompute=RecoveryTiming(
            setup_ms=normalized["recompute_fixed_overhead_ms"],
            ms_per_token=normalized["recompute_ms_per_token"],
        ),
        hbm_load=RecoveryTiming(
            setup_ms=float(setup["peer_hbm_recovery"]),
            ms_per_token=kv_bytes
            / normalized["peer_hbm_recovery_effective_bytes_per_ms"],
        ),
        dram_load=RecoveryTiming(
            setup_ms=float(setup["dram_recovery"]),
            ms_per_token=kv_bytes
            / normalized["dram_recovery_effective_bytes_per_ms"],
        ),
        prefix_chunk_tokens=int(params["prefix_chunk_tokens"]),
        decode_chunk_tokens=int(params["decode_chunk_tokens"]),
        pp_size=int(params["pp_size"]),
        tp_size=int(params["tp_size"]),
        initial_hbm_alpha=float(params["initial_hbm_alpha"]),
        minimum_hbm_alpha=float(params["minimum_hbm_alpha"]),
        maximum_hbm_alpha=float(params["maximum_hbm_alpha"]),
        aimd_decrease_factor=float(params["aimd_decrease_factor"]),
        aimd_increase_step=float(params["aimd_increase_step"]),
        aimd_stable_windows=int(params["aimd_stable_windows"]),
        random_salt=f"rtx4090-pp2-diagnostic-{profile['lineage_input_sha256']}",
        event_log_limit=int(params["event_log_limit"]),
    )
    provenance = {
        "status": "VALIDATED_DIAGNOSTIC_RUNTIME_OBJECT_ONLY",
        "hardware_label": HARDWARE_LABEL,
        "profile_semantics": PROFILE_SEMANTICS,
        "acknowledgement": acknowledgement,
        "transfer_protocol": "tcp",
        "physical_path": "registered_gpu_memory_over_mooncake_tcp",
        "protection_rates_used_only_for_copy_pressure": True,
        "recovery_rates_used_only_for_tier_recovery_ms": True,
        "cross_tier_charging": False,
        "replay_performed": False,
        "replay_authorized": False,
        "formal_ready": False,
        "paper_ready": False,
        "import_manifest_sha256": sha256_file(manifest_path.resolve()),
        "lineage_input_sha256": manifest["lineage_input_sha256"],
    }
    return runtime, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--materialize-diagnostic-runtime",
        action="store_true",
        help="construct a timing object only; this never runs replay",
    )
    parser.add_argument("--acknowledgement")
    args = parser.parse_args()
    if args.materialize_diagnostic_runtime:
        runtime, provenance = materialize_diagnostic_runtime(
            profile_path=args.profile,
            manifest_path=args.manifest,
            acknowledgement=args.acknowledgement,
        )
        print(
            json.dumps(
                {"runtime": asdict(runtime), "provenance": provenance},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        profile, _manifest = validate_import(
            profile_path=args.profile, manifest_path=args.manifest
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "hardware_label": profile["hardware_label"],
                    "profile_semantics": profile["profile_semantics"],
                    "transfer_protocol": profile["transport_observation"][
                        "transfer_protocol"
                    ],
                    "default_replay_authorized": False,
                    "formal_ready": False,
                    "paper_ready": False,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
