"""Fail-closed Mooncake A100-DP2 bridge for AegisKV output accounting.

The bridge consumes already-validated, immutable artifacts.  It performs no
trace loading, model fitting, replay, figure generation, or physical I/O.  In
particular, a condition-local ``strict_run_audit`` is validated as lineage but
is never accepted as the SGLSim A100 consumer preflight artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sglsim.srt.simulation.aegis_output_contract import (
    COMPONENT_COST_MODEL_SCHEMA,
    COMPONENT_NAMES,
    build_aegis_output_contract,
)


BRIDGE_SCHEMA = "sglsim_mooncake_a100_output_bridge_v1"
PREFLIGHT_EVIDENCE_CLASS = "sglsim_a100_dp2_consumer_preflight_v2"
PHYSICAL_PROFILE_EVIDENCE_CLASS = "physical_measurement"
MECHANISM_GATE_EVIDENCE_CLASS = (
    "a100_dp2_current_design_four_condition_mechanism_gate"
)
CONDITION_PROFILE_EVIDENCE_CLASS = (
    "aegiskv_current_design_condition_mechanism_profile"
)
COMPONENT_PROFILE_EVIDENCE_CLASS = (
    "mooncake_a100_transfer_component_profile_v1"
)
CAMPAIGN_ID = "qwen3-a100-dp2-20260824-v1"
HARDWARE_LABEL = "A100-1node-TP1-PP1-systemDP2"
DESIGN_SEMANTICS = "source_owned_independent_hbm_dram_v2"
PHYSICAL_PROFILE_SEMANTICS = "source_owned_four_physical_paths_v1"
WIRE_VALUES = {
    "BYPASS": 0,
    "ADMIT_DRAM": 1,
    "INVALID": 2,
    "ADMIT_HBM": 3,
}
TRANSFER_DURATION_FORMULA = (
    "per_transfer_setup_ms + "
    "(first_transfer_warmup_ms if calibration_transfer_ordinal == 0 else 0) + "
    "bytes / steady_state_bytes_per_ms"
)
TRANSFER_PARAMETERS = (
    "per_transfer_setup_ms",
    "first_transfer_warmup_ms",
    "steady_state_bytes_per_ms",
)
MOONCAKE_PATH_MAPPING = {
    "peer_hbm_protection": {
        "role": "protection",
        "runtime_path_id": "source_to_peer_hbm",
        "tier": "HBM",
        "copy_kind": "source_d2d",
        "event": "hbm_physical_copy",
    },
    "direct_dram_protection": {
        "role": "protection",
        "runtime_path_id": "source_to_store_dram",
        "tier": "DRAM",
        "copy_kind": "source_d2h",
        "event": "dram_physical_put",
    },
    "peer_hbm_recovery": {
        "role": "recovery",
        "recovery_hit_key": "peer_hbm",
        "event": "hbm_physical_load",
    },
    "dram_recovery": {
        "role": "recovery",
        "recovery_hit_key": "dram",
        "event": "dram_physical_load",
    },
}
COMPONENT_EVENT_FIELD_MAPPING = {
    "target_allocation": ["target_allocation_duration_ms"],
    "source_pack": ["source_pack_allocation_ms", "source_pack_duration_ms"],
    "engine_transfer": ["engine_transfer_duration_ms"],
    "target_finish": ["target_finish_duration_ms"],
}
PREFLIGHT_INPUT_NAMES = {
    "source_manifest",
    "runtime_binding",
    "model_submission_gate",
    "config_manifest",
    "config_verification",
    "physical_profile",
    "physical_profile_audit",
    "aimd_stall_calibration",
    "mechanism_gate",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MooncakeBridgeContractError(ValueError):
    """Raised when Mooncake evidence cannot enter output accounting."""


@dataclass(frozen=True)
class MooncakeA100BridgeInputs:
    """Physical and condition evidence for one same-population bridge."""

    preflight_manifest: Path
    physical_profile: Path
    mechanism_gate: Path
    failed_aegiskv_condition_profile: Path
    component_profile: Path | None = None

    def resolved(self) -> "MooncakeA100BridgeInputs":
        return MooncakeA100BridgeInputs(
            preflight_manifest=self.preflight_manifest.resolve(),
            physical_profile=self.physical_profile.resolve(),
            mechanism_gate=self.mechanism_gate.resolve(),
            failed_aegiskv_condition_profile=(
                self.failed_aegiskv_condition_profile.resolve()
            ),
            component_profile=(
                None
                if self.component_profile is None
                else self.component_profile.resolve()
            ),
        )


@dataclass(frozen=True)
class MooncakeA100BridgeResult:
    """Validated bridge values; no simulator runtime or result is created."""

    lineage_id: str
    preflight_sha256: str
    preflight_lineage_id: str
    source_manifest_sha256: str
    config_manifest_sha256: str
    config_verification_sha256: str
    physical_profile_sha256: str
    mechanism_gate_sha256: str
    condition_profile_sha256: str
    condition_id: str
    physical_path_profiles: Mapping[str, Mapping[str, Any]]
    protection_population: Mapping[str, Mapping[str, Any]]
    recovery_hit_counts: Mapping[str, int]
    protection_effectiveness: Mapping[str, Any]
    component_profiles: Mapping[str, Mapping[str, Any]]
    component_availability: Mapping[str, Mapping[str, Any]]
    strict_run_audit_sha256: str
    component_profile_sha256: str | None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "schema": BRIDGE_SCHEMA,
            "status": "PASS",
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "lineage_id": self.lineage_id,
            "lineage_derivation": {
                "canonical_json_sha256": True,
                "preflight_sha256": self.preflight_sha256,
                "preflight_lineage_id": self.preflight_lineage_id,
                "source_manifest_sha256": self.source_manifest_sha256,
                "config_manifest_sha256": self.config_manifest_sha256,
                "config_verification_sha256": (
                    self.config_verification_sha256
                ),
                "physical_profile_sha256": self.physical_profile_sha256,
                "mechanism_gate_sha256": self.mechanism_gate_sha256,
                "condition_profile_sha256": self.condition_profile_sha256,
                "strict_run_audit_sha256": self.strict_run_audit_sha256,
                "component_profile_sha256": self.component_profile_sha256,
            },
            "accepted_preflight_evidence_class": PREFLIGHT_EVIDENCE_CLASS,
            "strict_run_audit": {
                "sha256": self.strict_run_audit_sha256,
                "role": "condition_local_runtime_audit_only",
                "accepted_as_preflight": False,
            },
            "physical_path_mapping": {
                key: dict(value) for key, value in MOONCAKE_PATH_MAPPING.items()
            },
            "physical_path_profiles": {
                key: dict(value)
                for key, value in self.physical_path_profiles.items()
            },
            "same_condition_population": {
                "condition_id": self.condition_id,
                "condition_profile_sha256": self.condition_profile_sha256,
                "strict_run_audit_sha256": self.strict_run_audit_sha256,
                "source": (
                    "strict_run_audit.measurements.aegiskv_data_plane."
                    "protection_effectiveness"
                ),
                "protection_effectiveness": self.protection_effectiveness,
                "protection": {
                    key: dict(value)
                    for key, value in self.protection_population.items()
                },
                "recovery_hit_counts": dict(self.recovery_hit_counts),
            },
            "component_availability": {
                key: dict(value)
                for key, value in self.component_availability.items()
            },
            "paired_effects_required_separately": True,
            "authorization": {
                "replay_authorized": False,
                "figure_generation_authorized": False,
                "performance_claim_authorized": False,
                "formal_ready": False,
                "paper_ready": False,
            },
            "evidence_boundary": (
                "Static Mooncake-to-SGLSim accounting bridge only. It validates "
                "lineage and mappings but produces no replay or performance result."
            ),
        }

    def output_contract_kwargs(
        self,
        *,
        policy: str,
        runtime_snapshots: Mapping[str, Mapping[str, Any]],
        effect_pairs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if policy != "aegiskv":
            raise MooncakeBridgeContractError(
                "the A100 bridge condition must use the aegiskv policy"
            )
        if not effect_pairs:
            raise MooncakeBridgeContractError(
                "paired effects remain separately required"
            )
        return {
            "policy": policy,
            "runtime_snapshots": runtime_snapshots,
            "effect_pairs": effect_pairs,
            "recovery_hit_evidence": {
                "evidence_kind": "physical_observed",
                "lineage_id": self.lineage_id,
                "hardware_label": HARDWARE_LABEL,
                "preflight_sha256": self.preflight_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "config_manifest_sha256": self.config_manifest_sha256,
                "config_verification_sha256": (
                    self.config_verification_sha256
                ),
                "physical_profile_sha256": self.physical_profile_sha256,
                "recovery_hit_artifact_sha256": (
                    self.strict_run_audit_sha256
                ),
                "mechanism_gate_sha256": self.mechanism_gate_sha256,
                "mechanism_gate_status": "PASS",
                "same_lineage": True,
                "simulator_rate_default_eligible": True,
                "component_breakdown_declared": bool(
                    self.component_profiles
                ),
                "runtime_counter_source": None,
                "hit_counts": dict(self.recovery_hit_counts),
                "observed_successful_protection": {
                    key: dict(value)
                    for key, value in self.protection_population.items()
                },
            },
            "expected_lineage_id": self.lineage_id,
            "expected_hardware_label": HARDWARE_LABEL,
            "component_profiles": (
                dict(self.component_profiles)
                if self.component_profiles
                else None
            ),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MooncakeBridgeContractError(message)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MooncakeBridgeContractError(f"{label} must be an object")
    return value


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MooncakeBridgeContractError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MooncakeBridgeContractError(f"{label} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: Any, *, label: str) -> str:
    parsed = str(value)
    _require(
        _SHA256_RE.fullmatch(parsed) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return parsed


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MooncakeBridgeContractError(f"{label} must be a nonnegative integer")
    _require(value >= 0, f"{label} must be a nonnegative integer")
    return value


def _number(value: Any, *, label: str, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MooncakeBridgeContractError(f"{label} must be finite") from exc
    _require(math.isfinite(parsed), f"{label} must be finite")
    _require(
        parsed > 0.0 if positive else parsed >= 0.0,
        f"{label} must be {'positive' if positive else 'nonnegative'}",
    )
    return parsed


def _artifact(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    raw = _mapping(value, label=label)
    path_text = str(raw.get("path") or "")
    _require(bool(path_text), f"{label} path is missing")
    path = Path(path_text)
    _require(path.is_absolute(), f"{label} path must be absolute")
    resolved = path.resolve()
    _require(resolved.is_file(), f"{label} artifact is missing")
    if expected_path is not None:
        _require(
            resolved == expected_path.resolve(),
            f"{label} references the wrong artifact",
        )
    observed_sha = _sha256(resolved)
    _require(
        _digest(raw.get("sha256"), label=f"{label} sha256") == observed_sha,
        f"{label} hash mismatch",
    )
    if "size_bytes" in raw:
        _require(
            _nonnegative_int(raw["size_bytes"], label=f"{label} size_bytes")
            == resolved.stat().st_size,
            f"{label} size mismatch",
        )
    return {
        "path": str(resolved),
        "sha256": observed_sha,
        "size_bytes": resolved.stat().st_size,
    }


def _same_artifact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("path") == right.get("path")
        and left.get("sha256") == right.get("sha256")
    )


def _validate_preflight(
    path: Path,
    *,
    physical_profile_path: Path,
    mechanism_gate_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    value = _load(path, label="A100 preflight manifest")
    _require(
        value.get("schema_version") == 1
        and value.get("evidence_class") == PREFLIGHT_EVIDENCE_CLASS
        and value.get("status") == "PASS"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("hardware_label") == HARDWARE_LABEL
        and value.get("design_semantics") == DESIGN_SEMANTICS
        and value.get("physical_profile_semantics")
        == PHYSICAL_PROFILE_SEMANTICS
        and value.get("runtime_admission_decision_wire_values") == WIRE_VALUES
        and value.get("transfer_cost_model_schema")
        == COMPONENT_COST_MODEL_SCHEMA,
        "only the exact SGLSim A100 consumer PASS preflight is accepted",
    )
    authorization = _mapping(
        value.get("authorization"), label="preflight authorization"
    )
    for key in (
        "trace_loading_authorized",
        "model_fitting_authorized",
        "replay_authorized",
        "figure_generation_authorized",
        "performance_claim_authorized",
        "formal_ready",
        "paper_ready",
    ):
        _require(
            authorization.get(key) is False,
            f"preflight authorization {key} must remain false",
        )
    performance_window = _mapping(
        value.get("performance_window_contract"),
        label="preflight performance window",
    )
    _require(
        performance_window.get("post_workload_aimd_observation_s") == 25.0
        and performance_window.get(
            "observation_tail_excluded_from_performance_window"
        )
        is True,
        "preflight does not exclude the AIMD observation tail",
    )
    raw_inputs = _mapping(value.get("input_artifacts"), label="preflight inputs")
    _require(
        set(raw_inputs) == PREFLIGHT_INPUT_NAMES,
        "preflight input artifact set is not exact",
    )
    inputs = {
        name: _artifact(raw_inputs[name], label=f"preflight input {name}")
        for name in sorted(PREFLIGHT_INPUT_NAMES)
    }
    _require(
        Path(inputs["physical_profile"]["path"]) == physical_profile_path
        and Path(inputs["mechanism_gate"]["path"]) == mechanism_gate_path,
        "preflight binds different physical-profile or mechanism-gate paths",
    )
    raw_profiles = value.get("condition_profiles")
    raw_strict = value.get("condition_strict_audits")
    _require(
        isinstance(raw_profiles, list)
        and len(raw_profiles) == 4
        and isinstance(raw_strict, list)
        and len(raw_strict) == 4,
        "preflight must bind exactly four condition profiles and strict audits",
    )
    profiles = [
        _artifact(raw, label=f"preflight condition profile {index}")
        for index, raw in enumerate(raw_profiles)
    ]
    strict_audits = [
        _artifact(raw, label=f"preflight strict audit {index}")
        for index, raw in enumerate(raw_strict)
    ]
    _artifact(value.get("normalized_profile"), label="preflight normalized profile")
    preflight_lineage = _canonical_sha256(
        {
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "wire_values": WIRE_VALUES,
            "input_sha256": {
                name: inputs[name]["sha256"] for name in sorted(inputs)
            },
            "condition_profile_sha256": [
                profile["sha256"] for profile in profiles
            ],
        }
    )
    _require(
        value.get("lineage_identity_sha256") == preflight_lineage
        and value.get("output_directory_identity_sha256") == preflight_lineage
        and value.get("output_directory_name")
        == f"a100_dp2_preflight_{preflight_lineage[:16]}",
        "preflight lineage identity is not reproducible",
    )
    return value, inputs, profiles, strict_audits, preflight_lineage


def _validate_physical_profile(
    path: Path,
    *,
    inputs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = _load(path, label="Mooncake physical profile")
    _require(
        value.get("schema_version") == 2
        and value.get("evidence_class") == PHYSICAL_PROFILE_EVIDENCE_CLASS
        and value.get("status") == "passed"
        and value.get("profile_semantics") == PHYSICAL_PROFILE_SEMANTICS
        and value.get("design_semantics") == DESIGN_SEMANTICS
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("hardware_label") == HARDWARE_LABEL
        and value.get("transfer_cost_model_schema")
        == COMPONENT_COST_MODEL_SCHEMA
        and value.get("failures") == []
        and value.get("formal_ready") is False
        and value.get("paper_ready") is False,
        "Mooncake A100 schema-v2 physical profile did not pass",
    )
    for name in ("source_manifest", "config_manifest", "config_verification"):
        observed = _artifact(
            value.get(name),
            label=f"physical profile {name}",
            expected_path=Path(inputs[name]["path"]),
        )
        _require(
            observed["sha256"] == inputs[name]["sha256"],
            f"physical profile crosses {name} lineage",
        )
    metrics = _mapping(value.get("metrics"), label="physical profile metrics")
    measurements = _mapping(
        value.get("transfer_measurements"),
        label="physical profile transfer measurements",
    )
    _require(
        set(measurements) == set(MOONCAKE_PATH_MAPPING),
        "physical profile path set is not exact",
    )
    normalized: dict[str, dict[str, Any]] = {}
    for source_name, mapping in MOONCAKE_PATH_MAPPING.items():
        coefficients = {
            parameter: _number(
                metrics.get(f"{source_name}_{parameter}"),
                label=f"physical profile {source_name}.{parameter}",
                positive=parameter == "steady_state_bytes_per_ms",
            )
            for parameter in TRANSFER_PARAMETERS
        }
        measurement = _mapping(
            measurements[source_name],
            label=f"physical measurement {source_name}",
        )
        cost_model = _mapping(
            measurement.get("cost_model"),
            label=f"physical cost model {source_name}",
        )
        _require(
            measurement.get("event") == mapping["event"]
            and cost_model.get("schema") == COMPONENT_COST_MODEL_SCHEMA,
            f"physical measurement {source_name} identity mismatch",
        )
        for parameter, expected in coefficients.items():
            observed = _number(
                cost_model.get(parameter),
                label=f"physical cost model {source_name}.{parameter}",
                positive=parameter == "steady_state_bytes_per_ms",
            )
            _require(
                math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12),
                f"physical measurement {source_name} differs from metrics",
            )
        normalized[source_name] = {**mapping, **coefficients}
    contract = _mapping(
        value.get("runtime_consumption_contract"),
        label="physical runtime consumption contract",
    )
    _require(
        contract.get("transfer_cost_model_schema")
        == COMPONENT_COST_MODEL_SCHEMA
        and contract.get("transfer_duration_formula")
        == TRANSFER_DURATION_FORMULA,
        "physical runtime consumption formula is incompatible",
    )
    expected_sections = {
        "peer_hbm_protection": "healthy_peer_hbm_protection",
        "direct_dram_protection": "healthy_direct_dram_protection",
        "peer_hbm_recovery": "algorithm2_peer_hbm_recovery",
        "dram_recovery": "algorithm2_dram_recovery",
    }
    for source_name, section_name in expected_sections.items():
        section = _mapping(
            contract.get(section_name),
            label=f"runtime consumption {section_name}",
        )
        _require(
            section
            == {
                parameter: f"metrics.{source_name}_{parameter}"
                for parameter in sorted(TRANSFER_PARAMETERS)
            },
            f"runtime consumption mapping {section_name} is incompatible",
        )
    return value, normalized


def _validate_gate(
    path: Path,
    *,
    inputs: Mapping[str, Mapping[str, Any]],
    preflight_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value = _load(path, label="Mooncake mechanism gate")
    _require(
        value.get("schema_version") == 1
        and value.get("evidence_class") == MECHANISM_GATE_EVIDENCE_CLASS
        and value.get("status") == "PASS"
        and value.get("failures") == []
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("hardware_label") == HARDWARE_LABEL
        and value.get("design_semantics") == DESIGN_SEMANTICS
        and value.get("pilot_performance_claim_authorized") is False
        and value.get("formal_scale_submission_authorized") is False
        and value.get("paper_ready") is False,
        "Mooncake four-condition mechanism gate did not pass",
    )
    for name in (
        "source_manifest",
        "config_manifest",
        "config_verification",
        "physical_profile",
        "aimd_stall_calibration",
    ):
        observed = _artifact(
            value.get(name),
            label=f"mechanism gate {name}",
            expected_path=Path(inputs[name]["path"]),
        )
        _require(
            observed["sha256"] == inputs[name]["sha256"],
            f"mechanism gate crosses {name} lineage",
        )
    raw_profiles = value.get("condition_profiles")
    _require(
        isinstance(raw_profiles, list) and len(raw_profiles) == 4,
        "mechanism gate must hash exactly four condition profiles",
    )
    gate_profiles = [
        _artifact(raw, label=f"mechanism gate condition profile {index}")
        for index, raw in enumerate(raw_profiles)
    ]
    _require(
        {(row["path"], row["sha256"]) for row in gate_profiles}
        == {(row["path"], row["sha256"]) for row in preflight_profiles},
        "mechanism gate and preflight bind different condition profiles",
    )
    checks = value.get("checks")
    _require(isinstance(checks, list) and bool(checks), "mechanism checks are absent")
    names: set[str] = set()
    for raw in checks:
        check = _mapping(raw, label="mechanism check")
        name = str(check.get("name") or "")
        _require(
            bool(name) and name not in names and check.get("passed") is True,
            "mechanism gate contains a duplicate or failing check",
        )
        names.add(name)
    return value


def _validate_condition_population(
    condition_path: Path,
    *,
    inputs: Mapping[str, Mapping[str, Any]],
    preflight_profiles: Sequence[Mapping[str, Any]],
    strict_audits: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, int],
    dict[str, Any],
    str,
]:
    condition_sha = _sha256(condition_path)
    _require(
        (str(condition_path), condition_sha)
        in {(row["path"], row["sha256"]) for row in preflight_profiles},
        "selected condition profile is absent from the PASS preflight/gate",
    )
    value = _load(condition_path, label="Mooncake condition profile")
    _require(
        value.get("schema_version") == 1
        and value.get("evidence_class") == CONDITION_PROFILE_EVIDENCE_CLASS
        and value.get("status") == "PASS"
        and value.get("failures") == []
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("hardware_label") == HARDWARE_LABEL
        and value.get("design_semantics") == DESIGN_SEMANTICS
        and value.get("treatment") == "aegiskv"
        and value.get("failure_method") == "sigkill_process_group",
        "bridge requires the failed AegisKV condition profile",
    )
    _require(
        bool(str(value.get("condition_id") or "")),
        "failed AegisKV condition_id must be nonempty",
    )
    lineage = _mapping(value.get("input_lineage"), label="condition lineage")
    for name in (
        "source_manifest",
        "config_manifest",
        "config_verification",
        "physical_profile",
        "aimd_stall_calibration",
    ):
        observed = _artifact(
            lineage.get(name),
            label=f"condition lineage {name}",
            expected_path=Path(inputs[name]["path"]),
        )
        _require(
            observed["sha256"] == inputs[name]["sha256"],
            f"condition profile crosses {name} lineage",
        )
    strict = _artifact(
        lineage.get("strict_run_audit"), label="condition strict_run_audit"
    )
    _require(
        (strict["path"], strict["sha256"])
        in {(row["path"], row["sha256"]) for row in strict_audits},
        "condition strict_run_audit is absent from preflight lineage",
    )
    strict_value = _load(Path(strict["path"]), label="condition strict audit")
    _require(
        strict_value.get("schema_version") == 1
        and strict_value.get("evidence_class") == "strict_audit_report"
        and strict_value.get("status") == "PASS"
        and strict_value.get("failures") == []
        and strict_value.get("campaign_id") == CAMPAIGN_ID
        and strict_value.get("treatment") == "aegiskv",
        "condition strict audit did not pass for the same AegisKV campaign",
    )
    strict_measurements = _mapping(
        strict_value.get("measurements"), label="strict audit measurements"
    )
    data_plane = _mapping(
        strict_measurements.get("aegiskv_data_plane"),
        label="strict audit AegisKV data plane",
    )
    effectiveness = _mapping(
        data_plane.get("protection_effectiveness"),
        label="strict audit protection effectiveness",
    )
    _require(
        effectiveness.get("evidence_scope") == "same_strict_run_gate_events",
        "strict audit protection effectiveness has the wrong evidence scope",
    )
    raw_effective_paths = _mapping(
        effectiveness.get("paths"),
        label="strict audit protection effectiveness paths",
    )
    _require(
        set(raw_effective_paths)
        == {"source_to_peer_hbm", "source_to_store_dram"},
        "strict audit protection effectiveness path set is not exact",
    )

    effectiveness_fields = {
        "successful_protection_copy_count",
        "successful_protection_copy_bytes",
        "recovery_hit_count",
        "recovery_hit_bytes",
        "amplification_defined",
        "successful_copies_per_hit",
        "successful_bytes_per_hit",
    }

    def validate_effectiveness_row(raw: Any, *, label: str) -> dict[str, Any]:
        row = _mapping(raw, label=label)
        _require(
            set(row) == effectiveness_fields,
            f"{label} schema is not exact",
        )
        copies = _nonnegative_int(
            row["successful_protection_copy_count"],
            label=f"{label}.successful_protection_copy_count",
        )
        copy_bytes = _nonnegative_int(
            row["successful_protection_copy_bytes"],
            label=f"{label}.successful_protection_copy_bytes",
        )
        hits = _nonnegative_int(
            row["recovery_hit_count"],
            label=f"{label}.recovery_hit_count",
        )
        hit_bytes = _nonnegative_int(
            row["recovery_hit_bytes"],
            label=f"{label}.recovery_hit_bytes",
        )
        _require(
            bool(copies) == bool(copy_bytes)
            and bool(hits) == bool(hit_bytes)
            and (hits == 0 or copies > 0),
            f"{label} crosses or corrupts the same-run population",
        )
        defined = hits > 0
        _require(
            row["amplification_defined"] is defined,
            f"{label}.amplification_defined is inconsistent",
        )
        if defined:
            copies_per_hit = _number(
                row["successful_copies_per_hit"],
                label=f"{label}.successful_copies_per_hit",
            )
            bytes_per_hit = _number(
                row["successful_bytes_per_hit"],
                label=f"{label}.successful_bytes_per_hit",
            )
            _require(
                math.isclose(
                    copies_per_hit,
                    copies / hits,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    bytes_per_hit,
                    copy_bytes / hits,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{label} amplification does not reproduce",
            )
        else:
            _require(
                row["successful_copies_per_hit"] is None
                and row["successful_bytes_per_hit"] is None,
                f"{label} must leave zero-hit amplification unavailable",
            )
            copies_per_hit = None
            bytes_per_hit = None
        return {
            "successful_protection_copy_count": copies,
            "successful_protection_copy_bytes": copy_bytes,
            "recovery_hit_count": hits,
            "recovery_hit_bytes": hit_bytes,
            "amplification_defined": defined,
            "successful_copies_per_hit": copies_per_hit,
            "successful_bytes_per_hit": bytes_per_hit,
        }

    effective_paths = {
        path_id: validate_effectiveness_row(
            raw_effective_paths[path_id],
            label=f"strict audit protection effectiveness {path_id}",
        )
        for path_id in ("source_to_peer_hbm", "source_to_store_dram")
    }
    recovery_loads = _mapping(
        data_plane.get("recovery_loads"),
        label="strict audit recovery loads",
    )
    for tier, path_id in (
        ("peer_hbm", "source_to_peer_hbm"),
        ("dram", "source_to_store_dram"),
    ):
        summary = _mapping(
            recovery_loads.get(tier),
            label=f"strict audit recovery loads {tier}",
        )
        planned_events = _nonnegative_int(
            summary.get("planned_recovery_event_count"),
            label=f"strict audit recovery loads {tier}.planned_recovery_event_count",
        )
        _nonnegative_int(
            summary.get("planned_recovery_request_count"),
            label=(
                f"strict audit recovery loads {tier}."
                "planned_recovery_request_count"
            ),
        )
        planned_bytes = _nonnegative_int(
            summary.get("planned_loaded_bytes"),
            label=f"strict audit recovery loads {tier}.planned_loaded_bytes",
        )
        _require(
            planned_events == effective_paths[path_id]["recovery_hit_count"]
            and planned_bytes
            == effective_paths[path_id]["recovery_hit_bytes"],
            f"strict audit {tier} recovery summary crosses effectiveness",
        )
    all_paths = validate_effectiveness_row(
        effectiveness.get("all_protection_paths"),
        label="strict audit protection effectiveness all_protection_paths",
    )
    for field in (
        "successful_protection_copy_count",
        "successful_protection_copy_bytes",
        "recovery_hit_count",
        "recovery_hit_bytes",
    ):
        _require(
            all_paths[field] == sum(row[field] for row in effective_paths.values()),
            f"strict audit aggregate {field} does not equal its two paths",
        )
    total_hits = int(all_paths["recovery_hit_count"])
    _require(
        all_paths["amplification_defined"] is (total_hits > 0),
        "strict audit aggregate amplification state is inconsistent",
    )
    if total_hits > 0:
        _require(
            math.isclose(
                float(all_paths["successful_copies_per_hit"]),
                int(all_paths["successful_protection_copy_count"]) / total_hits,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(all_paths["successful_bytes_per_hit"]),
                int(all_paths["successful_protection_copy_bytes"]) / total_hits,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "strict audit aggregate amplification does not reproduce",
        )

    workload_events = _mapping(
        value.get("workload_window_events"), label="condition workload events"
    )
    source_paths = _mapping(
        workload_events.get("source_owned_paths"),
        label="condition source-owned paths",
    )
    _require(
        set(source_paths) == set(MOONCAKE_PATH_MAPPING),
        "condition source-owned path set is not exact",
    )
    protection: dict[str, dict[str, Any]] = {}
    hits: dict[str, int] = {}
    for source_name, mapping in MOONCAKE_PATH_MAPPING.items():
        row = _mapping(source_paths[source_name], label=f"condition {source_name}")
        count = _nonnegative_int(row.get("count"), label=f"{source_name}.count")
        byte_count = _nonnegative_int(
            row.get("bytes"), label=f"{source_name}.bytes"
        )
        _require(
            bool(count) == bool(byte_count),
            f"condition {source_name} count and bytes disagree",
        )
        _require(
            (count == 0 and row.get("physical") is False)
            or (count > 0 and row.get("physical") is True),
            f"condition {source_name} physical flag disagrees with observations",
        )
        if mapping["role"] == "protection":
            runtime_path = str(mapping["runtime_path_id"])
            strict_row = effective_paths[runtime_path]
            _require(
                count == strict_row["successful_protection_copy_count"]
                and byte_count
                == strict_row["successful_protection_copy_bytes"],
                f"condition {source_name} crosses its strict-audit population",
            )
            protection[runtime_path] = {
                "tier": mapping["tier"],
                "copy_kind": mapping["copy_kind"],
                "successful_copies": count,
                "successful_bytes": byte_count,
            }
        else:
            runtime_path = (
                "source_to_peer_hbm"
                if mapping["recovery_hit_key"] == "peer_hbm"
                else "source_to_store_dram"
            )
            strict_row = effective_paths[runtime_path]
            _require(
                count == strict_row["recovery_hit_count"]
                and byte_count == strict_row["recovery_hit_bytes"],
                f"condition {source_name} crosses its strict-audit population",
            )
            hits[str(mapping["recovery_hit_key"])] = count
    normalized_effectiveness = {
        "evidence_scope": "same_strict_run_gate_events",
        "paths": effective_paths,
        "all_protection_paths": all_paths,
    }
    return value, protection, hits, normalized_effectiveness, strict["sha256"]


def _validate_component_profile(
    path: Path | None,
    *,
    inputs: Mapping[str, Mapping[str, Any]],
    lineage_id: str,
    physical_paths: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    str | None,
]:
    availability = {
        "source_to_peer_hbm": {
            "available": False,
            "source_path": "peer_hbm_protection",
            "reason": "producer_component_profile_not_supplied",
        },
        "source_to_store_dram": {
            "available": False,
            "source_path": "direct_dram_protection",
            "reason": "dram_physical_put_monolithic_wall_duration_only",
        },
    }
    if path is None:
        return {}, availability, None
    value = _load(path, label="Mooncake component profile")
    _require(
        value.get("schema_version") == 1
        and value.get("evidence_class") == COMPONENT_PROFILE_EVIDENCE_CLASS
        and value.get("status") == "PASS"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("hardware_label") == HARDWARE_LABEL
        and value.get("design_semantics") == DESIGN_SEMANTICS
        and value.get("cost_model_schema") == COMPONENT_COST_MODEL_SCHEMA
        and value.get("failures") == [],
        "Mooncake component profile did not pass",
    )
    for name in (
        "source_manifest",
        "config_manifest",
        "config_verification",
        "physical_profile",
    ):
        observed = _artifact(
            value.get(name),
            label=f"component profile {name}",
            expected_path=Path(inputs[name]["path"]),
        )
        _require(
            observed["sha256"] == inputs[name]["sha256"],
            f"component profile crosses {name} lineage",
        )
    rows = _mapping(value.get("protection_paths"), label="component paths")
    _require(
        set(rows) == {"peer_hbm_protection", "direct_dram_protection"},
        "component profile protection path set is not exact",
    )
    normalized: dict[str, dict[str, Any]] = {}
    for source_name in ("peer_hbm_protection", "direct_dram_protection"):
        row = _mapping(rows[source_name], label=f"component path {source_name}")
        runtime_path = str(MOONCAKE_PATH_MAPPING[source_name]["runtime_path_id"])
        source_event = str(MOONCAKE_PATH_MAPPING[source_name]["event"])
        if row.get("available") is not True:
            _require(
                row.get("available") is False
                and row.get("source_event") == source_event
                and bool(str(row.get("unavailable_reason") or ""))
                and "components" not in row,
                f"unavailable component path {source_name} is not explicit",
            )
            availability[runtime_path] = {
                "available": False,
                "source_path": source_name,
                "reason": str(row["unavailable_reason"]),
            }
            continue
        _require(
            row.get("source_event") == source_event
            and row.get("event_field_mapping")
            == COMPONENT_EVENT_FIELD_MAPPING
            and row.get("cost_model_schema") == COMPONENT_COST_MODEL_SCHEMA,
            f"component source mapping {source_name} is incompatible",
        )
        components = _mapping(
            row.get("components"), label=f"component values {source_name}"
        )
        _require(
            set(components) == set(COMPONENT_NAMES),
            f"component path {source_name} is partial",
        )
        normalized_components: dict[str, dict[str, Any]] = {}
        setup_total = 0.0
        warmup_total = 0.0
        slope_total = 0.0
        for component in COMPONENT_NAMES:
            values = _mapping(
                components[component],
                label=f"component {source_name}.{component}",
            )
            _require(
                set(values) == set(TRANSFER_PARAMETERS),
                f"component {source_name}.{component} schema mismatch",
            )
            setup = _number(
                values["per_transfer_setup_ms"],
                label=f"{source_name}.{component}.per_transfer_setup_ms",
            )
            warmup = _number(
                values["first_transfer_warmup_ms"],
                label=f"{source_name}.{component}.first_transfer_warmup_ms",
            )
            rate_raw = values["steady_state_bytes_per_ms"]
            rate = (
                None
                if rate_raw is None
                else _number(
                    rate_raw,
                    label=f"{source_name}.{component}.steady_state_bytes_per_ms",
                    positive=True,
                )
            )
            setup_total += setup
            warmup_total += warmup
            slope_total += 0.0 if rate is None else 1.0 / rate
            normalized_components[component] = {
                "per_transfer_setup_ms": setup,
                "first_transfer_warmup_ms": warmup,
                "steady_state_bytes_per_ms": rate,
            }
        aggregate = physical_paths[source_name]
        expected_slope = 1.0 / float(aggregate["steady_state_bytes_per_ms"])
        _require(
            math.isclose(
                setup_total,
                float(aggregate["per_transfer_setup_ms"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                warmup_total,
                float(aggregate["first_transfer_warmup_ms"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                slope_total, expected_slope, rel_tol=0.0, abs_tol=1e-12
            ),
            f"component path {source_name} does not sum to its physical profile",
        )
        normalized[runtime_path] = {
            "lineage_id": lineage_id,
            "hardware_label": HARDWARE_LABEL,
            "profile_sha256": inputs["physical_profile"]["sha256"],
            "cost_model_schema": COMPONENT_COST_MODEL_SCHEMA,
            "components": normalized_components,
        }
        availability[runtime_path] = {
            "available": True,
            "source_path": source_name,
            "source_event": source_event,
            "event_field_mapping": dict(COMPONENT_EVENT_FIELD_MAPPING),
        }
    _require(bool(normalized), "component profile declares no available path")
    return normalized, availability, _sha256(path)


def validate_mooncake_a100_output_bridge(
    raw_inputs: MooncakeA100BridgeInputs,
) -> MooncakeA100BridgeResult:
    """Validate and normalize one A100 failed-condition accounting bridge."""

    inputs = raw_inputs.resolved()
    for name in (
        "preflight_manifest",
        "physical_profile",
        "mechanism_gate",
        "failed_aegiskv_condition_profile",
    ):
        _require(Path(getattr(inputs, name)).is_file(), f"missing bridge input {name}")
    if inputs.component_profile is not None:
        _require(inputs.component_profile.is_file(), "missing component profile")
    (
        _preflight,
        preflight_inputs,
        preflight_profiles,
        strict_audits,
        preflight_lineage,
    ) = _validate_preflight(
        inputs.preflight_manifest,
        physical_profile_path=inputs.physical_profile,
        mechanism_gate_path=inputs.mechanism_gate,
    )
    _physical, physical_paths = _validate_physical_profile(
        inputs.physical_profile, inputs=preflight_inputs
    )
    _validate_gate(
        inputs.mechanism_gate,
        inputs=preflight_inputs,
        preflight_profiles=preflight_profiles,
    )
    condition, protection, hits, effectiveness, strict_sha = (
        _validate_condition_population(
            inputs.failed_aegiskv_condition_profile,
            inputs=preflight_inputs,
            preflight_profiles=preflight_profiles,
            strict_audits=strict_audits,
        )
    )
    preflight_sha = _sha256(inputs.preflight_manifest)
    condition_sha = _sha256(inputs.failed_aegiskv_condition_profile)
    component_sha = (
        None
        if inputs.component_profile is None
        else _sha256(inputs.component_profile)
    )
    bridge_lineage = _canonical_sha256(
        {
            "schema": BRIDGE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "preflight_sha256": preflight_sha,
            "preflight_lineage_id": preflight_lineage,
            "source_manifest_sha256": preflight_inputs["source_manifest"][
                "sha256"
            ],
            "config_manifest_sha256": preflight_inputs["config_manifest"][
                "sha256"
            ],
            "config_verification_sha256": preflight_inputs[
                "config_verification"
            ]["sha256"],
            "physical_profile_sha256": preflight_inputs["physical_profile"][
                "sha256"
            ],
            "mechanism_gate_sha256": preflight_inputs["mechanism_gate"][
                "sha256"
            ],
            "condition_profile_sha256": condition_sha,
            "strict_run_audit_sha256": strict_sha,
            "component_profile_sha256": component_sha,
        }
    )
    component_profiles, availability, verified_component_sha = (
        _validate_component_profile(
            inputs.component_profile,
            inputs=preflight_inputs,
            lineage_id=bridge_lineage,
            physical_paths=physical_paths,
        )
    )
    _require(
        verified_component_sha == component_sha,
        "component profile changed during bridge validation",
    )
    return MooncakeA100BridgeResult(
        lineage_id=bridge_lineage,
        preflight_sha256=preflight_sha,
        preflight_lineage_id=preflight_lineage,
        source_manifest_sha256=preflight_inputs["source_manifest"]["sha256"],
        config_manifest_sha256=preflight_inputs["config_manifest"]["sha256"],
        config_verification_sha256=preflight_inputs["config_verification"][
            "sha256"
        ],
        physical_profile_sha256=preflight_inputs["physical_profile"]["sha256"],
        mechanism_gate_sha256=preflight_inputs["mechanism_gate"]["sha256"],
        condition_profile_sha256=condition_sha,
        condition_id=str(condition.get("condition_id") or ""),
        physical_path_profiles=physical_paths,
        protection_population=protection,
        recovery_hit_counts=hits,
        protection_effectiveness=effectiveness,
        component_profiles=component_profiles,
        component_availability=availability,
        strict_run_audit_sha256=strict_sha,
        component_profile_sha256=component_sha,
    )


def build_aegis_output_contract_from_mooncake(
    *,
    inputs: MooncakeA100BridgeInputs,
    policy: str,
    runtime_snapshots: Mapping[str, Mapping[str, Any]],
    effect_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the bridge, then build an in-memory metadata report."""

    bridge = validate_mooncake_a100_output_bridge(inputs)
    return build_aegis_output_contract(
        **bridge.output_contract_kwargs(
            policy=policy,
            runtime_snapshots=runtime_snapshots,
            effect_pairs=effect_pairs,
        )
    )


__all__ = [
    "BRIDGE_SCHEMA",
    "COMPONENT_EVENT_FIELD_MAPPING",
    "COMPONENT_PROFILE_EVIDENCE_CLASS",
    "MOONCAKE_PATH_MAPPING",
    "MooncakeA100BridgeInputs",
    "MooncakeA100BridgeResult",
    "MooncakeBridgeContractError",
    "build_aegis_output_contract_from_mooncake",
    "validate_mooncake_a100_output_bridge",
]
