from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    CAMPAIGN_ID,
    DESIGN_SEMANTICS,
    HARDWARE_LABEL,
    artifact_record,
)
from AegisKV.experiments.real_weka_full_design.preflight_mooncake_a100_dp2 import (
    preflight_and_write,
)
from sglsim.srt.simulation.aegis_mooncake_bridge import (
    COMPONENT_EVENT_FIELD_MAPPING,
    COMPONENT_PROFILE_EVIDENCE_CLASS,
    MOONCAKE_PATH_MAPPING,
    MooncakeA100BridgeInputs,
    MooncakeBridgeContractError,
    build_aegis_output_contract_from_mooncake,
    validate_mooncake_a100_output_bridge,
)
from sglsim.srt.simulation.aegis_output_contract import (
    COMPONENT_COST_MODEL_SCHEMA,
)
from test.unit.srt.test_aegis_a100_dp2_contract import (
    _fixture as _a100_fixture,
)
from test.unit.srt.test_aegis_a100_dp2_contract import (
    _write_json,
)


def _effectiveness_row(
    *, copies: int, copy_bytes: int, hits: int, hit_bytes: int
) -> dict[str, Any]:
    defined = hits > 0
    return {
        "successful_protection_copy_count": copies,
        "successful_protection_copy_bytes": copy_bytes,
        "recovery_hit_count": hits,
        "recovery_hit_bytes": hit_bytes,
        "amplification_defined": defined,
        "successful_copies_per_hit": copies / hits if defined else None,
        "successful_bytes_per_hit": copy_bytes / hits if defined else None,
    }


def _effectiveness_block() -> dict[str, Any]:
    peer = _effectiveness_row(
        copies=1, copy_bytes=4096, hits=1, hit_bytes=4096
    )
    dram = _effectiveness_row(
        copies=1, copy_bytes=4096, hits=1, hit_bytes=4096
    )
    return {
        "evidence_scope": "same_strict_run_gate_events",
        "paths": {
            "source_to_peer_hbm": peer,
            "source_to_store_dram": dram,
        },
        "all_protection_paths": _effectiveness_row(
            copies=2, copy_bytes=8192, hits=2, hit_bytes=8192
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _failed_condition(gate_path: Path) -> tuple[Path, Path]:
    gate = _load_json(gate_path)
    for raw in gate["condition_profiles"]:
        profile_path = Path(raw["path"]).resolve()
        profile = _load_json(profile_path)
        if (
            profile["treatment"] == "aegiskv"
            and profile["failure_method"] == "sigkill_process_group"
        ):
            return profile_path, Path(
                profile["input_lineage"]["strict_run_audit"]["path"]
            ).resolve()
    raise AssertionError("failed AegisKV fixture condition is missing")


def _rehash_condition_and_gate(
    *, profile_path: Path, strict_path: Path, gate_path: Path
) -> None:
    profile = _load_json(profile_path)
    strict_ref = artifact_record(strict_path)
    profile["input_lineage"]["strict_run_audit"] = strict_ref
    profile["artifacts"]["strict_audit"] = strict_ref
    _write_json(profile_path, profile)
    gate = _load_json(gate_path)
    for reference in gate["condition_profiles"]:
        if Path(reference["path"]).resolve() == profile_path:
            reference.update(artifact_record(profile_path))
            break
    else:
        raise AssertionError("failed profile is absent from fixture gate")
    _write_json(gate_path, gate)


def _install_current_strict_population(
    *, profile_path: Path, strict_path: Path, gate_path: Path
) -> None:
    strict = _load_json(strict_path)
    measurements = strict.setdefault("measurements", {})
    data_plane = measurements.setdefault("aegiskv_data_plane", {})
    data_plane.update(
        {
            "protection_effectiveness": _effectiveness_block(),
            "recovery_loads": {
                "peer_hbm": {
                    "planned_recovery_event_count": 1,
                    "planned_recovery_request_count": 1,
                    "planned_loaded_bytes": 4096,
                },
                "dram": {
                    "planned_recovery_event_count": 1,
                    "planned_recovery_request_count": 1,
                    "planned_loaded_bytes": 4096,
                },
            },
        }
    )
    strict.update(
        {
            "schema_version": 1,
            "evidence_class": "strict_audit_report",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "treatment": "aegiskv",
            "measurements": measurements,
        }
    )
    _write_json(strict_path, strict)
    _rehash_condition_and_gate(
        profile_path=profile_path,
        strict_path=strict_path,
        gate_path=gate_path,
    )


def _component_profile(
    path: Path,
    *,
    source_manifest: Path,
    config_manifest: Path,
    config_verification: Path,
    physical_profile: Path,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "evidence_class": COMPONENT_PROFILE_EVIDENCE_CLASS,
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "cost_model_schema": COMPONENT_COST_MODEL_SCHEMA,
            "source_manifest": artifact_record(source_manifest),
            "config_manifest": artifact_record(config_manifest),
            "config_verification": artifact_record(config_verification),
            "physical_profile": artifact_record(physical_profile),
            "protection_paths": {
                "peer_hbm_protection": {
                    "available": True,
                    "source_event": "hbm_physical_copy",
                    "event_field_mapping": COMPONENT_EVENT_FIELD_MAPPING,
                    "cost_model_schema": COMPONENT_COST_MODEL_SCHEMA,
                    "components": {
                        "target_allocation": {
                            "per_transfer_setup_ms": 0.25,
                            "first_transfer_warmup_ms": 0.0,
                            "steady_state_bytes_per_ms": None,
                        },
                        "source_pack": {
                            "per_transfer_setup_ms": 0.25,
                            "first_transfer_warmup_ms": 0.0,
                            "steady_state_bytes_per_ms": None,
                        },
                        "engine_transfer": {
                            "per_transfer_setup_ms": 0.5,
                            "first_transfer_warmup_ms": 3.0,
                            "steady_state_bytes_per_ms": 50.0,
                        },
                        "target_finish": {
                            "per_transfer_setup_ms": 0.0,
                            "first_transfer_warmup_ms": 0.0,
                            "steady_state_bytes_per_ms": None,
                        },
                    },
                },
                "direct_dram_protection": {
                    "available": False,
                    "source_event": "dram_physical_put",
                    "unavailable_reason": (
                        "dram_physical_put_monolithic_wall_duration_only"
                    ),
                },
            },
        },
    )


def _bridge_fixture(
    tmp_path: Path, *, component_profile: bool = True
) -> tuple[MooncakeA100BridgeInputs, dict[str, Path]]:
    a100_inputs, _ = _a100_fixture(tmp_path / "producer")
    condition_path, strict_path = _failed_condition(
        a100_inputs.mechanism_gate
    )
    _install_current_strict_population(
        profile_path=condition_path,
        strict_path=strict_path,
        gate_path=a100_inputs.mechanism_gate,
    )
    output_dir = preflight_and_write(
        inputs=a100_inputs, output_root=tmp_path / "a100-preflight"
    )
    component_path = None
    if component_profile:
        component_path = _component_profile(
            tmp_path / "component-profile.json",
            source_manifest=a100_inputs.source_manifest,
            config_manifest=a100_inputs.config_manifest,
            config_verification=a100_inputs.config_verification,
            physical_profile=a100_inputs.physical_profile,
        )
    bridge_inputs = MooncakeA100BridgeInputs(
        preflight_manifest=output_dir / "preflight-manifest.json",
        physical_profile=a100_inputs.physical_profile,
        mechanism_gate=a100_inputs.mechanism_gate,
        failed_aegiskv_condition_profile=condition_path,
        component_profile=component_path,
    )
    return bridge_inputs, {
        "preflight": bridge_inputs.preflight_manifest,
        "source_manifest": a100_inputs.source_manifest,
        "config_manifest": a100_inputs.config_manifest,
        "physical_profile": a100_inputs.physical_profile,
        "mechanism_gate": a100_inputs.mechanism_gate,
        "condition_profile": condition_path,
        "strict_audit": strict_path,
        "component_profile": component_path,
    }


def _runtime_snapshot() -> dict[str, Any]:
    def path(
        path_id: str,
        tier: str,
        copy_kind: str,
        service_ms: float,
    ) -> dict[str, Any]:
        return {
            "path_id": path_id,
            "tier": tier,
            "copy_kind": copy_kind,
            "attempted_copies": 1,
            "attempted_bytes": 4096,
            "successful_copies": 1,
            "successful_bytes": 4096,
            "failed_copies": 0,
            "cancelled_copies": 0,
            "pending_copies": 0,
            "first_transfer_warmup_count": 1,
            "healthy_period_attempted_copies": 1,
            "healthy_period_attempted_bytes": 4096,
            "healthy_period_first_transfer_warmup_count": 1,
            "healthy_period_transfer_service_ms": service_ms,
        }

    def churn() -> dict[str, Any]:
        return {
            "admission_attempt_count": 1,
            "admission_success_count": 1,
            "admission_rejection_count": 0,
            "admitted_bytes": 4096,
            "eviction_count": 0,
            "evicted_bytes": 0,
            "removal_count": 0,
            "removed_bytes": 0,
            "churn_event_count": 1,
            "churn_bytes": 4096,
            "rejection_reason_counts": {},
            "eviction_reason_counts": {},
            "removal_reason_counts": {},
        }

    return {
        "design_semantics": DESIGN_SEMANTICS,
        "pending_transfer_jobs": 0,
        "protection_paths": {
            "source_to_peer_hbm": path(
                "source_to_peer_hbm", "HBM", "source_d2d", 85.92
            ),
            "source_to_store_dram": path(
                "source_to_store_dram", "DRAM", "source_d2h", 169.84
            ),
        },
        "pool_churn": {"HBM": churn(), "DRAM": churn()},
        "recovery_estimate_action_counts": {"PEER_HBM": 1, "DRAM": 1},
        "recovery_execution_action_counts": {"PEER_HBM": 1, "DRAM": 1},
    }


def _effect_pair() -> dict[str, Any]:
    return {
        "pair_id": "same-lineage-pair-0",
        "baseline_whole_workload_wct_ms": 1000.0,
        "aegis_whole_workload_wct_ms": 1100.0,
        "baseline_failure_local_ms": 200.0,
        "aegis_failure_local_ms": 100.0,
        "included": True,
        "performance_direction_used_for_inclusion": False,
    }


def test_current_mooncake_bridge_maps_lineage_population_and_components(
    tmp_path: Path,
) -> None:
    inputs, paths = _bridge_fixture(tmp_path)
    result = validate_mooncake_a100_output_bridge(inputs)
    manifest = result.manifest()

    assert manifest["accepted_preflight_evidence_class"] == (
        "sglsim_a100_dp2_consumer_preflight_v2"
    )
    assert manifest["strict_run_audit"] == {
        "sha256": result.strict_run_audit_sha256,
        "role": "condition_local_runtime_audit_only",
        "accepted_as_preflight": False,
    }
    assert manifest["physical_path_mapping"] == MOONCAKE_PATH_MAPPING
    population = manifest["same_condition_population"]
    assert population["source"].endswith("protection_effectiveness")
    assert population["protection_effectiveness"]["evidence_scope"] == (
        "same_strict_run_gate_events"
    )
    assert result.recovery_hit_counts == {"peer_hbm": 1, "dram": 1}
    assert result.protection_population["source_to_peer_hbm"][
        "successful_bytes"
    ] == 4096
    assert result.component_availability["source_to_peer_hbm"][
        "available"
    ] is True
    assert result.component_availability["source_to_store_dram"] == {
        "available": False,
        "source_path": "direct_dram_protection",
        "reason": "dram_physical_put_monolithic_wall_duration_only",
    }
    assert result.lineage_id == manifest["lineage_id"]
    assert manifest["lineage_derivation"]["preflight_sha256"] == (
        result.preflight_sha256
    )
    assert manifest["lineage_derivation"]["source_manifest_sha256"] == (
        artifact_record(paths["source_manifest"])["sha256"]
    )
    assert manifest["lineage_derivation"]["strict_run_audit_sha256"] == (
        result.strict_run_audit_sha256
    )
    assert result.component_availability["source_to_peer_hbm"][
        "event_field_mapping"
    ]["source_pack"] == [
        "source_pack_allocation_ms",
        "source_pack_duration_ms",
    ]

    output = build_aegis_output_contract_from_mooncake(
        inputs=inputs,
        policy="aegiskv",
        runtime_snapshots={"0": _runtime_snapshot()},
        effect_pairs=[_effect_pair()],
    )
    assert output["metadata_only"] is True
    assert output["authorization"]["performance_claim_authorized"] is False
    assert output["transfer_service_components"]["source_to_peer_hbm"][
        "available"
    ] is True
    assert output["transfer_service_components"]["source_to_store_dram"] == {
        "available": False,
        "cost_model_schema": COMPONENT_COST_MODEL_SCHEMA,
        "components": None,
    }
    assert len(output["remaining_profile_dependent_fields"]) == 12
    assert all(
        field.startswith("source_to_store_dram.")
        for field in output["remaining_profile_dependent_fields"]
    )
    claim = output["joint_effect_contract"]["claim_gate"]
    assert claim["failure_local_benefit"] is True
    assert claim["whole_workload_regression"] is True
    assert claim["general_advantage_supported"] is False
    assert not (tmp_path / "replay.json").exists()
    assert not (tmp_path / "performance.json").exists()


def test_strict_audit_cannot_substitute_for_independent_preflight(
    tmp_path: Path,
) -> None:
    inputs, paths = _bridge_fixture(tmp_path)
    crossed = replace(inputs, preflight_manifest=paths["strict_audit"])
    with pytest.raises(
        MooncakeBridgeContractError, match="exact SGLSim A100 consumer PASS preflight"
    ):
        validate_mooncake_a100_output_bridge(crossed)


def test_old_strict_audit_without_same_run_population_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, paths = _bridge_fixture(tmp_path)
    strict = _load_json(paths["strict_audit"])
    strict["measurements"]["aegiskv_data_plane"].pop(
        "protection_effectiveness"
    )
    _write_json(paths["strict_audit"], strict)
    _rehash_condition_and_gate(
        profile_path=paths["condition_profile"],
        strict_path=paths["strict_audit"],
        gate_path=paths["mechanism_gate"],
    )
    # Recreate just the immutable SGLSim preflight with the updated real paths.
    original_preflight = _load_json(paths["preflight"])
    original_preflight["input_artifacts"]["mechanism_gate"] = artifact_record(
        paths["mechanism_gate"]
    )
    for index, raw in enumerate(original_preflight["condition_profiles"]):
        if Path(raw["path"]).resolve() == paths["condition_profile"]:
            original_preflight["condition_profiles"][index] = artifact_record(
                paths["condition_profile"]
            )
    for index, raw in enumerate(original_preflight["condition_strict_audits"]):
        if Path(raw["path"]).resolve() == paths["strict_audit"]:
            original_preflight["condition_strict_audits"][index] = artifact_record(
                paths["strict_audit"]
            )
    _refresh_preflight_identity(original_preflight)
    replacement = _write_json(
        tmp_path / "old-strict-preflight.json", original_preflight
    )
    crossed = replace(inputs, preflight_manifest=replacement)
    with pytest.raises(
        MooncakeBridgeContractError, match="protection effectiveness"
    ):
        validate_mooncake_a100_output_bridge(crossed)


def _refresh_preflight_identity(preflight: dict[str, Any]) -> None:
    from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
        canonical_sha256,
    )

    identity = canonical_sha256(
        {
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "wire_values": preflight["runtime_admission_decision_wire_values"],
            "input_sha256": {
                name: row["sha256"]
                for name, row in sorted(preflight["input_artifacts"].items())
            },
            "condition_profile_sha256": [
                row["sha256"] for row in preflight["condition_profiles"]
            ],
        }
    )
    preflight["lineage_identity_sha256"] = identity
    preflight["output_directory_identity_sha256"] = identity
    preflight["output_directory_name"] = f"a100_dp2_preflight_{identity[:16]}"


def test_condition_profile_cannot_cross_strict_population(
    tmp_path: Path,
) -> None:
    inputs, paths = _bridge_fixture(tmp_path)
    strict = _load_json(paths["strict_audit"])
    block = strict["measurements"]["aegiskv_data_plane"][
        "protection_effectiveness"
    ]
    block["paths"]["source_to_peer_hbm"] = _effectiveness_row(
        copies=1, copy_bytes=4096, hits=0, hit_bytes=0
    )
    block["all_protection_paths"] = _effectiveness_row(
        copies=2, copy_bytes=8192, hits=1, hit_bytes=4096
    )
    peer_recovery = strict["measurements"]["aegiskv_data_plane"][
        "recovery_loads"
    ]["peer_hbm"]
    peer_recovery["planned_recovery_event_count"] = 0
    peer_recovery["planned_recovery_request_count"] = 0
    peer_recovery["planned_loaded_bytes"] = 0
    _write_json(paths["strict_audit"], strict)
    _rehash_condition_and_gate(
        profile_path=paths["condition_profile"],
        strict_path=paths["strict_audit"],
        gate_path=paths["mechanism_gate"],
    )
    preflight = _load_json(paths["preflight"])
    preflight["input_artifacts"]["mechanism_gate"] = artifact_record(
        paths["mechanism_gate"]
    )
    for raw in preflight["condition_profiles"]:
        if Path(raw["path"]).resolve() == paths["condition_profile"]:
            raw.update(artifact_record(paths["condition_profile"]))
    for raw in preflight["condition_strict_audits"]:
        if Path(raw["path"]).resolve() == paths["strict_audit"]:
            raw.update(artifact_record(paths["strict_audit"]))
    _refresh_preflight_identity(preflight)
    replacement = _write_json(tmp_path / "crossed-preflight.json", preflight)
    crossed = replace(inputs, preflight_manifest=replacement)
    with pytest.raises(
        MooncakeBridgeContractError,
        match="peer_hbm_recovery crosses its strict-audit population",
    ):
        validate_mooncake_a100_output_bridge(crossed)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "source_manifest",
        "config_manifest",
        "physical_profile",
        "mechanism_gate",
        "strict_audit",
    ],
)
def test_bound_artifact_hash_tamper_fails_closed(
    tmp_path: Path, artifact_name: str
) -> None:
    inputs, paths = _bridge_fixture(tmp_path)
    path = paths[artifact_name]
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(MooncakeBridgeContractError, match="hash mismatch"):
        validate_mooncake_a100_output_bridge(inputs)


def test_partial_component_profile_is_rejected(tmp_path: Path) -> None:
    inputs, paths = _bridge_fixture(tmp_path)
    component_path = paths["component_profile"]
    assert component_path is not None
    component = _load_json(component_path)
    component["protection_paths"]["peer_hbm_protection"]["components"].pop(
        "target_finish"
    )
    _write_json(component_path, component)
    with pytest.raises(MooncakeBridgeContractError, match="partial"):
        validate_mooncake_a100_output_bridge(inputs)


def test_paired_effects_remain_separate_and_required(tmp_path: Path) -> None:
    inputs, _ = _bridge_fixture(tmp_path, component_profile=False)
    result = validate_mooncake_a100_output_bridge(inputs)
    with pytest.raises(
        MooncakeBridgeContractError, match="paired effects remain separately required"
    ):
        result.output_contract_kwargs(
            policy="aegiskv",
            runtime_snapshots={"0": _runtime_snapshot()},
            effect_pairs=[],
        )
