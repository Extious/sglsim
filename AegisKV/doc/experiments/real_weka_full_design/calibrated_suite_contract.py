#!/usr/bin/env python3
"""Static, fail-closed binding for future calibrated AegisKV suites.

The contract accepts exactly one independently validated hardware lineage:
either the A100-DP2 consumer preflight or the existing V100-DP4 schema-v2
import.  It binds a deterministic experiment matrix to that lineage but never
loads a trace, creates a simulator runtime, runs a replay, or emits performance
metrics.  A real runner must invoke :func:`preflight_suite_runner` before any
of those operations and must use the hash-derived output directory.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    A100_CONSUMER_COMPATIBILITY_IDENTITY,
    A100DP2InputPaths,
    CONFIG_LAYOUT as A100_CONFIG_LAYOUT,
    DESIGN_SEMANTICS,
    MODEL as A100_MODEL,
    PREFLIGHT_EVIDENCE_CLASS as A100_PREFLIGHT_EVIDENCE_CLASS,
    artifact_record,
    canonical_sha256,
    load_object,
    validate_a100_dp2_bundle,
)
from AegisKV.experiments.real_weka_full_design.run import (
    V100_HARDWARE_LABEL,
    V100_PROFILE_SEMANTICS,
    preflight_v100_config,
)


A100_CONSUMER_KIND = "a100_dp2_consumer_preflight_v2"
V100_CONSUMER_KIND = "v100_dp4_schema_v2_import"
CONSUMER_KINDS = {A100_CONSUMER_KIND, V100_CONSUMER_KIND}
SUITE_BINDING_EVIDENCE_CLASS = "sglsim_aegiskv_calibrated_suite_binding_v1"
SUITE_RUNNER_PREFLIGHT_EVIDENCE_CLASS = (
    "sglsim_aegiskv_calibrated_suite_runner_preflight_v1"
)
SUITE_BINDING_STATUS = "BOUND_STATIC_ONLY"
MATRIX_SCHEMA_VERSION = 1
MAX_PLANNED_CELLS = 100_000
MATRIX_FIELDS = {
    "schema_version",
    "model_scales",
    "system_dp_sizes",
    "concurrency_levels",
    "failure_rates",
    "capacity_points",
    "seeds",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")

A100_STATIC_AUTHORIZATION = {
    "suite_binding_authorized": True,
    "trace_loading_authorized": False,
    "model_fitting_authorized": False,
    "replay_authorized": False,
    "figure_generation_authorized": False,
    "performance_claim_authorized": False,
    "formal_ready": False,
    "paper_ready": False,
}

SUITE_STATIC_AUTHORIZATION = {
    "suite_binding_materialized": True,
    "static_runner_preflight_authorized": True,
    "trace_loading_authorized": False,
    "model_fitting_authorized": False,
    "replay_authorized": False,
    "figure_generation_authorized": False,
    "performance_result_authorized": False,
    "performance_claim_authorized": False,
    "formal_ready": False,
    "paper_ready": False,
}


class CalibratedSuiteContractError(ValueError):
    """Raised when a consumer lineage or suite matrix fails closed."""


@dataclass(frozen=True)
class CalibratedConsumerInput:
    """Paths for one validated consumer and its normalized payload."""

    consumer_kind: str
    consumer_manifest: Path
    consumer_payload: Path

    def resolved(self) -> "CalibratedConsumerInput":
        return CalibratedConsumerInput(
            consumer_kind=str(self.consumer_kind),
            consumer_manifest=Path(self.consumer_manifest).resolve(),
            consumer_payload=Path(self.consumer_payload).resolve(),
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibratedSuiteContractError(message)


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return load_object(path)
    except (OSError, ValueError) as exc:
        raise CalibratedSuiteContractError(
            f"cannot load {label} JSON object {path}: {exc}"
        ) from exc


def _verify_artifact(
    value: Any, *, label: str, expected_path: Path | None = None
) -> Path:
    _require(isinstance(value, Mapping), f"{label} artifact reference is absent")
    reference = dict(value)
    path = Path(str(reference.get("path") or "")).resolve()
    _require(path.is_file(), f"{label} artifact is missing: {path}")
    expected_sha = str(reference.get("sha256") or "")
    _require(_SHA256_RE.fullmatch(expected_sha) is not None, f"{label} sha invalid")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(observed == expected_sha, f"{label} artifact hash mismatch")
    size = reference.get("size_bytes")
    _require(
        isinstance(size, int)
        and not isinstance(size, bool)
        and size == path.stat().st_size,
        f"{label} artifact size mismatch",
    )
    if expected_path is not None:
        _require(path == expected_path.resolve(), f"{label} artifact path mismatch")
    return path


def _exact_static_authorization(value: Any, *, label: str) -> None:
    _require(
        value == A100_STATIC_AUTHORIZATION,
        f"{label} does not have the exact static-only authorization boundary",
    )


def _a100_consumer_descriptor(
    source: CalibratedConsumerInput,
) -> dict[str, Any]:
    manifest = _load(source.consumer_manifest, label="A100 consumer manifest")
    evidence_class = str(manifest.get("evidence_class") or "")
    if evidence_class != A100_PREFLIGHT_EVIDENCE_CLASS:
        if evidence_class == "sglsim_v100_physical_profile_import":
            raise CalibratedSuiteContractError(
                "A100 and V100 consumer lineages cannot cross"
            )
        lowered_evidence = evidence_class.lower()
        if "clean" in lowered_evidence or "submission_gate" in lowered_evidence:
            raise CalibratedSuiteContractError(
                "a Mooncake clean submission/readiness gate is not the "
                "independent SGLSim A100 consumer preflight"
            )
        raise CalibratedSuiteContractError(
            "A100 suite input is not the independent SGLSim consumer preflight"
        )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("status") == "PASS"
        and manifest.get("design_semantics") == DESIGN_SEMANTICS
        and manifest.get("consumer_compatibility_identity")
        == A100_CONSUMER_COMPATIBILITY_IDENTITY,
        "A100 consumer manifest identity is invalid",
    )
    _exact_static_authorization(
        manifest.get("authorization"), label="A100 consumer manifest"
    )

    normalized_path = _verify_artifact(
        manifest.get("normalized_profile"),
        label="A100 normalized profile",
        expected_path=source.consumer_payload,
    )
    normalized = _load(normalized_path, label="A100 normalized profile")
    _require(
        normalized.get("schema_version") == 2
        and normalized.get("profile_schema_class") == "A100DP2PhysicalProfile"
        and normalized.get("hardware_label")
        == A100_CONSUMER_COMPATIBILITY_IDENTITY["hardware_label"]
        and normalized.get("design_semantics") == DESIGN_SEMANTICS
        and normalized.get("consumer_compatibility_identity")
        == A100_CONSUMER_COMPATIBILITY_IDENTITY
        and normalized.get("runtime_admission_decision_wire_values")
        == A100_CONSUMER_COMPATIBILITY_IDENTITY[
            "runtime_admission_decision_wire_values"
        ]
        and normalized.get("authorization") == A100_STATIC_AUTHORIZATION,
        "A100 normalized physical profile identity is invalid",
    )

    raw_inputs = manifest.get("input_artifacts")
    _require(isinstance(raw_inputs, Mapping), "A100 consumer inputs are absent")
    field_names = set(A100DP2InputPaths.__dataclass_fields__)
    _require(
        set(raw_inputs) == field_names,
        "A100 consumer input artifact set is not exact",
    )
    input_paths: dict[str, Path] = {}
    for name in sorted(field_names):
        input_paths[name] = _verify_artifact(
            raw_inputs[name], label=f"A100 consumer {name}"
        )

    # Re-run the complete A100 producer validation from original artifacts.
    # The preflight manifest is never accepted on its outer PASS alone.
    replayed = validate_a100_dp2_bundle(A100DP2InputPaths(**input_paths))
    expected_manifest = replayed.manifest()
    _require(
        set(manifest)
        == set(expected_manifest)
        | {
            "output_directory_name",
            "output_directory_identity_sha256",
            "normalized_profile",
        },
        "A100 consumer manifest field set is not exact",
    )
    for key, expected in expected_manifest.items():
        _require(
            manifest.get(key) == expected,
            f"A100 consumer manifest is not reproducible at {key}",
        )
    _require(
        manifest.get("output_directory_identity_sha256")
        == replayed.lineage_identity_sha256
        and manifest.get("output_directory_name")
        == f"a100_dp2_preflight_{replayed.lineage_identity_sha256[:16]}",
        "A100 consumer output identity is not lineage-derived",
    )
    _require(
        set(normalized)
        == {
            "schema_version",
            "profile_schema_class",
            "hardware_label",
            "design_semantics",
            "physical_profile_semantics",
            "runtime_admission_decision_wire_values",
            "consumer_compatibility_identity",
            "model",
            "dtype",
            "layout",
            "model_revision",
            "transfer_cost_model_schema",
            "runtime_consumption_contract",
            "metrics",
            "source_profile",
            "authorization",
        }
        and normalized.get("metrics") == expected_manifest["profile_metrics"]
        and normalized.get("source_profile")
        == expected_manifest["input_artifacts"]["physical_profile"]
        and normalized.get("model_revision")
        == expected_manifest["model_revision"],
        "A100 normalized profile differs from the replayed consumer",
    )

    config = _load(input_paths["config_manifest"], label="A100 config manifest")
    model_config_sha256 = str(config.get("model_config_sha256") or "")
    _require(
        _SHA256_RE.fullmatch(model_config_sha256) is not None,
        "A100 model config SHA-256 is absent from the validated config lineage",
    )
    capacity = dict(
        dict(config.get("configuration") or {}).get("capacity") or {}
    )
    peer_hbm_bytes = capacity.get("peer_hbm_bytes_per_target")
    dram_bytes = capacity.get("dram_bytes")
    _require(
        isinstance(peer_hbm_bytes, int)
        and not isinstance(peer_hbm_bytes, bool)
        and peer_hbm_bytes > 0
        and isinstance(dram_bytes, int)
        and not isinstance(dram_bytes, bool)
        and dram_bytes > 0,
        "A100 calibrated capacity anchor is invalid",
    )

    calibration_artifacts = {
        name: dict(raw_inputs[name])
        for name in (
            "source_manifest",
            "runtime_binding",
            "model_submission_gate",
            "config_manifest",
            "config_verification",
            "physical_profile",
            "physical_profile_audit",
            "aimd_stall_calibration",
            "mechanism_gate",
        )
    }
    calibration_identity = canonical_sha256(
        {
            "consumer_kind": A100_CONSUMER_KIND,
            "lineage_identity_sha256": replayed.lineage_identity_sha256,
            "artifact_sha256": {
                name: value["sha256"]
                for name, value in calibration_artifacts.items()
            },
        }
    )
    return {
        "consumer_kind": A100_CONSUMER_KIND,
        "consumer_manifest": artifact_record(source.consumer_manifest),
        "consumer_payload": artifact_record(source.consumer_payload),
        "hardware_label": expected_manifest["hardware_label"],
        "design_semantics": DESIGN_SEMANTICS,
        "profile_semantics": expected_manifest["physical_profile_semantics"],
        "profile_schema_version": 2,
        "profile_schema_class": "A100DP2PhysicalProfile",
        "model": {
            "name": A100_MODEL,
            "scale_label": "Qwen3-8B",
            "revision": expected_manifest["model_revision"],
            "config_sha256": model_config_sha256,
            "dtype": "fp16",
        },
        "layout": {
            "tp_size": A100_CONFIG_LAYOUT["tp_size"],
            "pp_size": A100_CONFIG_LAYOUT["pp_size"],
            "system_dp_size": A100_CONFIG_LAYOUT["system_dp_size"],
            "worker_count": A100_CONFIG_LAYOUT["worker_count"],
        },
        "capacity_anchor": {
            "peer_hbm_bytes_per_target": peer_hbm_bytes,
            "store_dram_bytes_total": dram_bytes,
        },
        "compatibility_identity": _json_clone(
            A100_CONSUMER_COMPATIBILITY_IDENTITY
        ),
        "profile_sha256": calibration_artifacts["physical_profile"]["sha256"],
        "mechanism_gate_sha256": calibration_artifacts["mechanism_gate"][
            "sha256"
        ],
        "lineage_identity_sha256": replayed.lineage_identity_sha256,
        "calibration_identity_sha256": calibration_identity,
        "calibration_artifacts": calibration_artifacts,
        "condition_profiles": [
            dict(value) for value in replayed.condition_profile_artifacts
        ],
        "condition_strict_audits": [
            dict(value) for value in replayed.strict_audit_artifacts
        ],
        "consumer_execution_authorized": False,
    }


def _v100_consumer_descriptor(
    source: CalibratedConsumerInput,
) -> dict[str, Any]:
    try:
        config, _runtime, provenance, _projection, binding = preflight_v100_config(
            source.consumer_payload, source.consumer_manifest
        )
    except (OSError, ValueError) as exc:
        raise CalibratedSuiteContractError(
            f"V100 schema-v2 consumer preflight failed: {exc}"
        ) from exc
    calibration = config.get("calibration")
    model = config.get("model")
    cluster = config.get("cluster")
    capacity = config.get("capacity")
    _require(
        isinstance(calibration, Mapping)
        and isinstance(model, Mapping)
        and isinstance(cluster, Mapping)
        and isinstance(capacity, Mapping),
        "V100 imported config lacks model/layout/capacity calibration",
    )
    _require(
        model.get("revision") == A100_MODEL
        and model.get("serving_dtype") == "half"
        and model.get("kv_cache_dtype") == "fp16"
        and calibration.get("bundle_semantics") == V100_PROFILE_SEMANTICS
        and calibration.get("hardware_label") == V100_HARDWARE_LABEL,
        "V100 imported model or physical-profile identity is invalid",
    )
    peer_hbm_bytes = capacity.get("hbm_bytes_per_target")
    dram_bytes = capacity.get("dram_bytes_total")
    _require(
        isinstance(peer_hbm_bytes, int)
        and not isinstance(peer_hbm_bytes, bool)
        and peer_hbm_bytes > 0
        and isinstance(dram_bytes, int)
        and not isinstance(dram_bytes, bool)
        and dram_bytes > 0,
        "V100 calibrated capacity anchor is invalid",
    )
    artifact_names = (
        "physical_profile",
        "source_manifest",
        "source_verification",
        "config_manifest",
        "config_verification",
        "pilot_runtime_gate",
    )
    calibration_artifacts: dict[str, dict[str, Any]] = {}
    for name in artifact_names:
        value = calibration.get(name)
        _verify_artifact(value, label=f"V100 {name}")
        calibration_artifacts[name] = dict(value)
    optional_pressure = calibration.get("aimd_pressure_calibration")
    if optional_pressure is not None:
        _verify_artifact(optional_pressure, label="V100 AIMD pressure calibration")
        calibration_artifacts["aimd_pressure_calibration"] = dict(optional_pressure)

    config_manifest = _load(
        Path(calibration_artifacts["config_manifest"]["path"]),
        label="V100 config manifest",
    )
    model_config_sha256 = str(config_manifest.get("model_config_sha256") or "")
    _require(
        _SHA256_RE.fullmatch(model_config_sha256) is not None,
        "V100 model config SHA-256 is absent from the validated config lineage",
    )

    calibration_identity = canonical_sha256(
        {
            "consumer_kind": V100_CONSUMER_KIND,
            "artifact_sha256": {
                name: value["sha256"]
                for name, value in calibration_artifacts.items()
            },
            "runtime_consumption_contract": provenance[
                "runtime_consumption_contract"
            ],
        }
    )
    layout = {
        "tp_size": int(cluster["tp_size"]),
        "pp_size": int(cluster["pp_size"]),
        "system_dp_size": int(cluster["dp_size"]),
        "worker_count": int(cluster["dp_size"]),
    }
    _require(
        layout == {
            "tp_size": 1,
            "pp_size": 1,
            "system_dp_size": 4,
            "worker_count": 4,
        },
        "V100 suite input is not the exact DP4 physical layout",
    )
    return {
        "consumer_kind": V100_CONSUMER_KIND,
        "consumer_manifest": artifact_record(source.consumer_manifest),
        "consumer_payload": artifact_record(source.consumer_payload),
        "hardware_label": V100_HARDWARE_LABEL,
        "design_semantics": DESIGN_SEMANTICS,
        "profile_semantics": V100_PROFILE_SEMANTICS,
        "profile_schema_version": 2,
        "profile_schema_class": "V100DP4SchemaV2Import",
        "model": {
            "name": str(model["revision"]),
            "scale_label": "Qwen3-8B",
            "revision": str(model["model_snapshot_revision"]),
            "config_sha256": model_config_sha256,
            "dtype": "fp16",
        },
        "layout": layout,
        "capacity_anchor": {
            "peer_hbm_bytes_per_target": peer_hbm_bytes,
            "store_dram_bytes_total": dram_bytes,
        },
        "compatibility_identity": _json_clone(
            calibration["compatibility_identity"]
        ),
        "profile_sha256": binding["physical_profile_sha256"],
        "mechanism_gate_sha256": calibration_artifacts[
            "pilot_runtime_gate"
        ]["sha256"],
        "lineage_identity_sha256": canonical_sha256(
            {
                "source_manifest_sha256": binding["source_manifest_sha256"],
                "config_manifest_sha256": binding["config_manifest_sha256"],
                "config_verification_sha256": binding[
                    "config_verification_sha256"
                ],
                "physical_profile_sha256": binding["physical_profile_sha256"],
            }
        ),
        "calibration_identity_sha256": calibration_identity,
        "calibration_artifacts": calibration_artifacts,
        "condition_profiles": [],
        "condition_strict_audits": [],
        "consumer_execution_authorized": False,
    }


def validate_consumer_input(source: CalibratedConsumerInput) -> dict[str, Any]:
    """Re-hash and replay the selected hardware consumer in isolation."""

    source = source.resolved()
    _require(
        source.consumer_kind in CONSUMER_KINDS,
        f"unknown calibrated consumer kind: {source.consumer_kind!r}",
    )
    _require(
        source.consumer_manifest.is_file(),
        f"consumer manifest is missing: {source.consumer_manifest}",
    )
    _require(
        source.consumer_payload.is_file(),
        f"consumer payload is missing: {source.consumer_payload}",
    )
    if source.consumer_kind == A100_CONSUMER_KIND:
        return _a100_consumer_descriptor(source)
    manifest = _load(source.consumer_manifest, label="V100 import manifest")
    _require(
        manifest.get("evidence_class") == "sglsim_v100_physical_profile_import",
        "A100 and V100 consumer lineages cannot cross",
    )
    return _v100_consumer_descriptor(source)


def _strict_int_sequence(
    value: Any, *, label: str, minimum: int
) -> list[int]:
    _require(isinstance(value, list) and bool(value), f"{label} must be nonempty")
    result: list[int] = []
    for item in value:
        _require(
            isinstance(item, int)
            and not isinstance(item, bool)
            and item >= minimum,
            f"{label} contains an invalid integer",
        )
        result.append(item)
    _require(result == sorted(set(result)), f"{label} must be unique and sorted")
    return result


def _strict_rate_sequence(value: Any) -> list[float]:
    _require(
        isinstance(value, list) and bool(value),
        "failure_rates must be nonempty",
    )
    result: list[float] = []
    for item in value:
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and 0.0 <= float(item) <= 1.0,
            "failure_rates contains a value outside [0, 1]",
        )
        result.append(float(item))
    _require(
        result == sorted(set(result)),
        "failure_rates must be unique and sorted",
    )
    return result


def _validated_matrix(
    raw: Mapping[str, Any], *, source: Mapping[str, Any]
) -> dict[str, Any]:
    _require(set(raw) == MATRIX_FIELDS, "suite matrix field set is not exact")
    _require(
        raw.get("schema_version") == MATRIX_SCHEMA_VERSION,
        "suite matrix schema version is invalid",
    )
    expected_model = {
        "model": source["model"]["name"],
        "scale_label": source["model"]["scale_label"],
        "model_revision": source["model"]["revision"],
    }
    _require(
        raw.get("model_scales") == [expected_model],
        "suite model scale/revision exceeds or crosses the physical lineage",
    )
    expected_dp = int(source["layout"]["system_dp_size"])
    _require(
        raw.get("system_dp_sizes") == [expected_dp],
        "suite system-DP cannot extrapolate beyond the physical lineage",
    )
    concurrency = _strict_int_sequence(
        raw.get("concurrency_levels"), label="concurrency_levels", minimum=1
    )
    failure_rates = _strict_rate_sequence(raw.get("failure_rates"))
    seeds = _strict_int_sequence(raw.get("seeds"), label="seeds", minimum=0)

    raw_capacities = raw.get("capacity_points")
    _require(
        isinstance(raw_capacities, list) and bool(raw_capacities),
        "capacity_points must be nonempty",
    )
    capacities: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    anchor = source["capacity_anchor"]
    for raw_point in raw_capacities:
        _require(
            isinstance(raw_point, Mapping)
            and set(raw_point)
            == {"label", "peer_hbm_fraction", "store_dram_fraction"},
            "capacity point field set is not exact",
        )
        label = str(raw_point.get("label") or "")
        _require(
            _LABEL_RE.fullmatch(label) is not None and label not in seen_labels,
            "capacity point label is invalid or duplicated",
        )
        seen_labels.add(label)
        fractions: dict[str, float] = {}
        for field in ("peer_hbm_fraction", "store_dram_fraction"):
            value = raw_point.get(field)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0,
                f"capacity point {label} {field} is outside [0, 1]",
            )
            fractions[field] = float(value)
        capacities.append(
            {
                "label": label,
                **fractions,
                "peer_hbm_bytes_per_target": int(
                    anchor["peer_hbm_bytes_per_target"]
                    * fractions["peer_hbm_fraction"]
                ),
                "store_dram_bytes_total": int(
                    anchor["store_dram_bytes_total"]
                    * fractions["store_dram_fraction"]
                ),
            }
        )
    _require(
        [point["label"] for point in capacities]
        == sorted(point["label"] for point in capacities),
        "capacity_points must be sorted by label",
    )
    cell_count = (
        len(concurrency) * len(failure_rates) * len(capacities) * len(seeds)
    )
    _require(
        0 < cell_count <= MAX_PLANNED_CELLS,
        f"suite matrix cardinality exceeds {MAX_PLANNED_CELLS}",
    )
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "model_scales": [expected_model],
        "system_dp_sizes": [expected_dp],
        "concurrency_levels": concurrency,
        "failure_rates": failure_rates,
        "capacity_points": capacities,
        "seeds": seeds,
        "cell_count": cell_count,
    }


def build_suite_binding(
    *, source: CalibratedConsumerInput, matrix: Mapping[str, Any]
) -> dict[str, Any]:
    """Build an in-memory static binding after replaying the consumer gate."""

    descriptor = validate_consumer_input(source)
    normalized_matrix = _validated_matrix(matrix, source=descriptor)
    identity_payload = {
        "schema_version": 1,
        "evidence_class": SUITE_BINDING_EVIDENCE_CLASS,
        "consumer_kind": descriptor["consumer_kind"],
        "consumer_manifest_sha256": descriptor["consumer_manifest"]["sha256"],
        "consumer_payload_sha256": descriptor["consumer_payload"]["sha256"],
        "profile_sha256": descriptor["profile_sha256"],
        "calibration_identity_sha256": descriptor[
            "calibration_identity_sha256"
        ],
        "matrix": normalized_matrix,
    }
    binding_sha = canonical_sha256(identity_payload)
    binding_directory_name = f"calibrated_suite_binding_{binding_sha[:16]}"
    runner_output_name = f"calibrated_suite_run_{binding_sha[:16]}"
    return {
        "schema_version": 1,
        "evidence_class": SUITE_BINDING_EVIDENCE_CLASS,
        "status": SUITE_BINDING_STATUS,
        "suite_binding_identity_sha256": binding_sha,
        "suite_id": f"aegiskv-calibrated-suite-{binding_sha[:16]}",
        "binding_directory_name": binding_directory_name,
        "design_semantics": DESIGN_SEMANTICS,
        "source_consumer": descriptor,
        "matrix": normalized_matrix,
        "runner_contract": {
            "schema_version": 1,
            "consumer_preflight_before_trace_output_or_model_fit": True,
            "suite_binding_rehash_before_runner_construction": True,
            "hash_derived_output_directory": runner_output_name,
            "immutable_output_refuse_overwrite": True,
            "cross_hardware_lineage_allowed": False,
            "cross_model_or_system_dp_extrapolation_allowed": False,
            "physical_profile_sha256": descriptor["profile_sha256"],
            "calibration_identity_sha256": descriptor[
                "calibration_identity_sha256"
            ],
            "matrix_axis_semantics": {
                "model_scales": "exact_consumer_model_and_revision_only",
                "system_dp_sizes": "exact_physically_validated_system_dp_only",
                "concurrency_levels": "planned_max_inflight_requests",
                "failure_rates": (
                    "planned_per_workflow_failure_probability_axis;"
                    "schedule_requires_seeded_runner_materialization"
                ),
                "capacity_points": (
                    "fractions_of_same_lineage_calibrated_hbm_and_dram_bytes"
                ),
                "seeds": "deterministic_matrix_and_failure_schedule_seed",
            },
            "planned_cell_state": "PLANNED_NOT_RUN",
        },
        "authorization": dict(SUITE_STATIC_AUTHORIZATION),
        "result_state": {
            "execution_status": "NOT_RUN",
            "performance_results_present": False,
            "result_artifacts": [],
        },
        "evidence_boundary": (
            "Hash-bound metadata-only suite plan. It contains no trace, fitted "
            "model, replay, figure, performance metric, physical KV movement, "
            "or performance claim."
        ),
    }


def _matrix_spec_from_binding(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": matrix["schema_version"],
        "model_scales": _json_clone(matrix["model_scales"]),
        "system_dp_sizes": list(matrix["system_dp_sizes"]),
        "concurrency_levels": list(matrix["concurrency_levels"]),
        "failure_rates": list(matrix["failure_rates"]),
        "capacity_points": [
            {
                "label": point["label"],
                "peer_hbm_fraction": point["peer_hbm_fraction"],
                "store_dram_fraction": point["store_dram_fraction"],
            }
            for point in matrix["capacity_points"]
        ],
        "seeds": list(matrix["seeds"]),
    }


def validate_suite_binding(path: Path) -> dict[str, Any]:
    """Rebuild and byte-semantically verify one immutable suite binding."""

    path = Path(path).resolve()
    binding = _load(path, label="calibrated suite binding")
    _require(
        binding.get("schema_version") == 1
        and binding.get("evidence_class") == SUITE_BINDING_EVIDENCE_CLASS
        and binding.get("status") == SUITE_BINDING_STATUS
        and binding.get("authorization") == SUITE_STATIC_AUTHORIZATION
        and binding.get("result_state")
        == {
            "execution_status": "NOT_RUN",
            "performance_results_present": False,
            "result_artifacts": [],
        },
        "suite binding is not static-only",
    )
    source = binding.get("source_consumer")
    _require(isinstance(source, Mapping), "suite binding source consumer is absent")
    matrix = binding.get("matrix")
    _require(
        isinstance(matrix, Mapping)
        and MATRIX_FIELDS < set(matrix)
        and set(matrix) == MATRIX_FIELDS | {"cell_count"},
        "suite binding matrix is incomplete",
    )
    manifest_path = _verify_artifact(
        source.get("consumer_manifest"), label="suite consumer manifest"
    )
    payload_path = _verify_artifact(
        source.get("consumer_payload"), label="suite consumer payload"
    )
    expected = build_suite_binding(
        source=CalibratedConsumerInput(
            consumer_kind=str(source.get("consumer_kind") or ""),
            consumer_manifest=manifest_path,
            consumer_payload=payload_path,
        ),
        matrix=_matrix_spec_from_binding(matrix),
    )
    _require(binding == expected, "suite binding is not exactly reproducible")
    return binding


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_suite_binding(
    *,
    source: CalibratedConsumerInput,
    matrix: Mapping[str, Any],
    output_root: Path,
) -> Path:
    """Write only a hash-derived immutable binding after all validation."""

    binding = build_suite_binding(source=source, matrix=matrix)
    output_root = Path(output_root).resolve()
    directory = output_root / binding["binding_directory_name"]
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite immutable suite: {directory}")

    # The validation/build above is the ordering fence.  No path is created
    # before both the physical consumer and the matrix have passed.
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = output_root / (
        f".{binding['binding_directory_name']}.tmp-{os.getpid()}"
    )
    if temporary_directory.exists():
        raise FileExistsError(temporary_directory)
    temporary_directory.mkdir()
    temporary_path = temporary_directory / "suite-binding.json"
    payload = (
        json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        _write_new(temporary_path, payload)
        validate_suite_binding(temporary_path)
        os.replace(temporary_directory, directory)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        if temporary_directory.exists():
            temporary_directory.rmdir()
    return directory / "suite-binding.json"


def _planned_cells(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix = binding["matrix"]
    cells: list[dict[str, Any]] = []
    for concurrency, failure_rate, capacity, seed in itertools.product(
        matrix["concurrency_levels"],
        matrix["failure_rates"],
        matrix["capacity_points"],
        matrix["seeds"],
    ):
        axes = {
            "model": matrix["model_scales"][0],
            "system_dp_size": matrix["system_dp_sizes"][0],
            "concurrency": concurrency,
            "failure_rate": failure_rate,
            "capacity": capacity,
            "seed": seed,
        }
        cells.append(
            {
                "cell_id": canonical_sha256(
                    {
                        "suite_binding_identity_sha256": binding[
                            "suite_binding_identity_sha256"
                        ],
                        "axes": axes,
                    }
                )[:20],
                "axes": _json_clone(axes),
                "state": "PLANNED_NOT_RUN",
            }
        )
    return cells


def preflight_suite_runner(path: Path) -> dict[str, Any]:
    """Return a no-I/O runner plan after fully revalidating the binding."""

    path = Path(path).resolve()
    binding = validate_suite_binding(path)
    cells = _planned_cells(binding)
    _require(
        len(cells) == binding["matrix"]["cell_count"],
        "suite runner cell cardinality changed",
    )
    return {
        "schema_version": 1,
        "evidence_class": SUITE_RUNNER_PREFLIGHT_EVIDENCE_CLASS,
        "status": "PASS_STATIC_ONLY",
        "suite_binding": artifact_record(path),
        "suite_binding_identity_sha256": binding[
            "suite_binding_identity_sha256"
        ],
        "physical_profile_sha256": binding["source_consumer"][
            "profile_sha256"
        ],
        "calibration_identity_sha256": binding["source_consumer"][
            "calibration_identity_sha256"
        ],
        "output_directory_name": binding["runner_contract"][
            "hash_derived_output_directory"
        ],
        "cell_count": len(cells),
        "cells": cells,
        "authorization": {
            "trace_loading_authorized": False,
            "model_fitting_authorized": False,
            "replay_authorized": False,
            "result_output_authorized": False,
            "performance_claim_authorized": False,
        },
        "evidence_boundary": (
            "Static runner preflight only; every cell is PLANNED_NOT_RUN and "
            "contains no performance fields."
        ),
    }
