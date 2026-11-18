#!/usr/bin/env python3
"""Fail-closed contract for the isolated Mooncake A100-DP2 lineage.

This module is intentionally independent of ``run.py`` and the V100 profile
consumer.  It validates the A100 producer artifacts needed before an A100
trace could be loaded, an output directory could be created, or a timing model
could be fitted.  It does not construct a simulator runtime or authorize a
replay, figure, performance result, or paper claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


CAMPAIGN_ID = "qwen3-a100-dp2-20260824-v1"
HARDWARE_LABEL = "A100-1node-TP1-PP1-systemDP2"
MODEL = "Qwen/Qwen3-8B"
DESIGN_SEMANTICS = "source_owned_independent_hbm_dram_v2"
PHYSICAL_PROFILE_SEMANTICS = "source_owned_four_physical_paths_v1"
PHYSICAL_PROFILE_EVIDENCE_CLASS = "physical_measurement"
CONDITION_PROFILE_EVIDENCE_CLASS = (
    "aegiskv_current_design_condition_mechanism_profile"
)
MECHANISM_GATE_EVIDENCE_CLASS = (
    "a100_dp2_current_design_four_condition_mechanism_gate"
)
PREFLIGHT_EVIDENCE_CLASS = "sglsim_a100_dp2_consumer_preflight_v2"

WIRE_VALUES = {
    "BYPASS": 0,
    "ADMIT_DRAM": 1,
    "INVALID": 2,
    "ADMIT_HBM": 3,
}
PHYSICAL_LAYOUT = {"tp_size": 1, "pp_size": 1, "dp_size": 2}
CONFIG_LAYOUT = {
    "model": MODEL,
    "serving_dtype": "half",
    "kv_cache_dtype": "fp16",
    "tp_size": 1,
    "pp_size": 1,
    "system_dp_size": 2,
    "worker_count": 2,
}
HARDWARE_PROFILE = {
    "gpu_count": 2,
    "gpu_capability": [8, 0],
    "gpu_name_substring": "A100",
    "gpu_tres_per_node": "gpu:a100:2",
    "slurm_partition": "short",
    "physical_interconnect": "runtime_validated_cuda_p2p",
}
EXPECTED_CONDITIONS = {
    ("no-protection", "none"),
    ("no-protection", "sigkill_process_group"),
    ("aegiskv", "none"),
    ("aegiskv", "sigkill_process_group"),
}
EXPECTED_TRANSFER_TREATMENTS = {
    "aegiskv",
    "aegiskv-no-hbm",
    "aegiskv-hbm-reject",
}
TRANSFER_REPEATS_PER_TREATMENT = 2
TRANSFER_AUDIT_COUNT = (
    len(EXPECTED_TRANSFER_TREATMENTS) * TRANSFER_REPEATS_PER_TREATMENT
)
TRANSFER_PATH_EVENTS = {
    "peer_hbm_protection": "hbm_physical_copy",
    "direct_dram_protection": "dram_physical_put",
    "peer_hbm_recovery": "hbm_physical_load",
    "dram_recovery": "dram_physical_load",
}
TRANSFER_PATH_TREATMENTS = {
    "peer_hbm_protection": "aegiskv",
    "direct_dram_protection": "aegiskv-no-hbm",
    "peer_hbm_recovery": "aegiskv",
    "dram_recovery": "aegiskv-no-hbm",
}
TRANSFER_TREATMENT_REQUIRED_EVENTS = {
    "aegiskv": {"hbm_physical_copy", "hbm_physical_load"},
    "aegiskv-no-hbm": {"dram_physical_put", "dram_physical_load"},
    "aegiskv-hbm-reject": {"dram_physical_put", "dram_physical_load"},
}
TRANSFER_COST_MODEL_SCHEMA = "affine_setup_first_transfer_warmup_v1"
TRANSFER_DURATION_FORMULA = (
    "per_transfer_setup_ms + "
    "(first_transfer_warmup_ms if calibration_transfer_ordinal == 0 else 0) + "
    "bytes / steady_state_bytes_per_ms"
)
TRANSFER_COST_COMPONENTS = {
    "per_transfer_setup_ms",
    "first_transfer_warmup_ms",
    "steady_state_bytes_per_ms",
}
PROFILE_METRICS = {
    "recompute_ms_per_token",
    "recompute_fixed_overhead_ms",
    *{
        f"{path}_{component}"
        for path in TRANSFER_PATH_EVENTS
        for component in TRANSFER_COST_COMPONENTS
    },
}
RUNTIME_CONSUMPTION_CONTRACT = {
    "transfer_cost_model_schema": TRANSFER_COST_MODEL_SCHEMA,
    "transfer_duration_formula": TRANSFER_DURATION_FORMULA,
    "healthy_peer_hbm_protection": {
        component: f"metrics.peer_hbm_protection_{component}"
        for component in sorted(TRANSFER_COST_COMPONENTS)
    },
    "healthy_direct_dram_protection": {
        component: f"metrics.direct_dram_protection_{component}"
        for component in sorted(TRANSFER_COST_COMPONENTS)
    },
    "algorithm2_peer_hbm_recovery": {
        component: f"metrics.peer_hbm_recovery_{component}"
        for component in sorted(TRANSFER_COST_COMPONENTS)
    },
    "algorithm2_dram_recovery": {
        component: f"metrics.dram_recovery_{component}"
        for component in sorted(TRANSFER_COST_COMPONENTS)
    },
    "recompute_slope": "metrics.recompute_ms_per_token",
    "recompute_fixed_overhead": "metrics.recompute_fixed_overhead_ms",
    "physical_calibration_identifiability": {
        "repeats_per_treatment": TRANSFER_REPEATS_PER_TREATMENT,
        "total_transfer_runs": TRANSFER_AUDIT_COUNT,
        "stream_identity": "slurm_job_run",
        "ordinal_scope": "per_run_per_event_actual_issue_order",
        "path_calibration_treatments": TRANSFER_PATH_TREATMENTS,
        "chunk_or_worker_log_is_stream": False,
    },
}

ACTIVATION_REVISION = "v44-affine-six-run-calibration"
RUNTIME_BINDING_EVIDENCE_CLASS = "a100_dp2_runtime_binding_v46"
RUNTIME_IDENTITY_EVIDENCE_CLASS = "a100_dp2_runtime_identity_snapshot_v46"
MODEL_SUBMISSION_GATE_EVIDENCE_CLASS = "a100_dp2_model_submission_gate_v46"
MODEL_ATTEMPT_AUDIT_EVIDENCE_CLASS = (
    "a100_dp2_model_execution_attempt_audit_v47"
)
PHYSICAL_PROFILE_AUDIT_EVIDENCE_CLASS = (
    "a100_dp2_physical_profile_build_audit_v54"
)
TRANSFER_SERIES_EVIDENCE_CLASS = (
    "a100_dp2_transfer_calibration_series_ready_v53"
)
TRANSFER_ATTEMPT_AUDIT_EVIDENCE_CLASS = "a100_dp2_transfer_attempt_audit_v53"
RECOMPUTE_ATTEMPT_AUDIT_EVIDENCE_CLASS = (
    "a100_dp2_recompute_attempt_audit_v51"
)
PHYSICAL_CALIBRATION_GATE_EVIDENCE_CLASS = (
    "a100_dp2_physical_calibration_submission_gate_v50"
)
AIMD_STALL_CALIBRATION_EVIDENCE_CLASS = (
    "a100_dp2_aimd_local_stall_calibration_v55"
)
EXPECTED_CLEAN_GATES = {
    "bidirectional_cuda_p2p_and_integrity",
    "mooncake_cuda_engine_import",
    "pinned_vllm_v2_runtime",
    "pre_run_source_runtime_identity",
    "qwen3_8b_snapshot_available",
    "slurm_node_exclusive",
    "two_a100_80gb_sm80_devices",
    "zero_foreign_gpu_processes",
}
GPU_TELEMETRY_CONTRACT = {
    "device_ids": [0, 1],
    "select": "pucmet",
    "sample_interval_seconds": 2,
    "timestamp_options": "DT",
    "format": "csv",
    "per_gpu_activity_required": True,
}
GPU_TELEMETRY_COLUMNS = (
    "#Date",
    "Time",
    "gpu",
    "pwr",
    "gtemp",
    "mtemp",
    "sm",
    "mem",
    "enc",
    "dec",
    "jpg",
    "ofa",
    "mclk",
    "pclk",
    "fb",
    "bar1",
    "ccpm",
    "sbecc",
    "dbecc",
    "pci",
    "rxpci",
    "txpci",
)
GPU_TELEMETRY_UNITS = (
    "#YYYYMMDD",
    "HH:MM:SS",
    "Idx",
    "W",
    "C",
    "C",
    "%",
    "%",
    "%",
    "%",
    "%",
    "%",
    "MHz",
    "MHz",
    "MB",
    "MB",
    "MB",
    "errs",
    "errs",
    "errs",
    "MB/s",
    "MB/s",
)
RUNTIME_ARTIFACT_NAMES = {
    "python_executable",
    "vllm_package",
    "v2_model_runner",
    "torch_package",
    "torch_c_extension",
    "mooncake_engine",
}

# These names are A100 consumer requirements.  They do not import, rename, or
# change the V100 gate's required-name set in run.py.
A100_REQUIRED_MECHANISM_CHECKS = {
    "runtime_admission_decision_wire_values_verified",
    "single_source_config_physical_profile_lineage",
    "independent_admit_hbm_admit_dram_and_bypass_exercised",
    "hbm_reject_then_independent_dram_admit_exercised",
    "all_four_independent_residency_outcomes_exercised",
    "source_owned_peer_hbm_and_direct_dram_protection_physical",
    "hbm_and_dram_paths_confirm_independently",
    "no_target_owned_or_hbm_to_dram_runtime_semantics",
    "failed_aegiskv_physical_peer_hbm_recovery",
    "failed_aegiskv_physical_dram_recovery",
    "target_local_stall_contracts_alpha_and_hbm_quota",
    "stable_windows_additively_increase_alpha",
    "aimd_pause_reclaim_satisfy_resume_contract",
    "controller_deletes_hbm_only_tail_without_dram_prerequisite",
    "physically_calibrated_local_stall_threshold_applied",
    "all_lineage_artifacts_exist_and_match_bytes",
}

# Immutable identity consumed by the static calibrated-suite binder.  This is
# deliberately A100-only: accepting it never changes or aliases the V100
# schema, and the binder must still replay the complete validation below from
# the original producer artifacts.
A100_CONSUMER_COMPATIBILITY_IDENTITY = {
    "profile_schema_class": "A100DP2PhysicalProfile",
    "hardware_label": HARDWARE_LABEL,
    "model": MODEL,
    "model_scale": "Qwen3-8B",
    "dtype": "fp16",
    "layout": CONFIG_LAYOUT,
    "design_semantics": DESIGN_SEMANTICS,
    "physical_profile_semantics": PHYSICAL_PROFILE_SEMANTICS,
    "runtime_admission_decision_wire_values": WIRE_VALUES,
    "required_mechanism_checks": sorted(A100_REQUIRED_MECHANISM_CHECKS),
    "source_owned_paths": {
        "peer_hbm_protection": "source_worker_d2d_p2p",
        "direct_dram_protection": "source_worker_d2h",
        "peer_hbm_recovery": "survivor_worker_hbm_load",
        "dram_recovery": "survivor_worker_store_load",
        "target_owned_or_hbm_to_dram_path_allowed": False,
    },
}

_AIMD_CONFIG_FIELDS = {
    "total_hbm_bytes",
    "interval_s",
    "pressure_free_bytes",
    "local_stall_ms",
    "initial_alpha",
    "min_alpha",
    "max_alpha",
    "decrease_factor",
    "increase_step",
    "stable_window_threshold",
}
_AIMD_ZERO_INTEGER_FIELDS = {
    "window_count",
    "pressure_window_count",
    "stable_window_count",
    "memory_pressure_window_count",
    "local_stall_pressure_window_count",
    "alpha_transition_count",
    "reclaim_count",
    "reclaim_bytes",
    "hbm_only_reclaim_count",
}
_AIMD_TRAJECTORY_FIELDS = {
    "observed_at_ms",
    "pressure",
    "pressure_reasons",
    "alpha",
    "quota_bytes",
    "allocated_bytes",
    "stable_windows",
    "cuda_free_bytes",
    "reclaim_count",
    "reclaim_bytes",
    "hbm_only_reclaim_count",
}
_AIMD_POOL_SUMMARY_FIELDS = {
    "window_count",
    "pressure_window_count",
    "stable_window_count",
    "memory_pressure_window_count",
    "local_stall_pressure_window_count",
    "initial_alpha",
    "final_alpha",
    "observed_min_alpha",
    "observed_max_alpha",
    "alpha_transition_count",
    "min_cuda_free_bytes",
    "peak_quota_utilization",
    "reclaim_count",
    "reclaim_bytes",
    "hbm_only_reclaim_count",
    "trajectory",
}
AIMD_CONSUMPTION_CONTRACT = {
    "deployment_config_path": "deployment.hbm_quota_aimd",
    "strict_measurement_path": "measurements.aegiskv_data_plane.aimd",
    "frozen_fields": sorted(_AIMD_CONFIG_FIELDS),
    "pool_count_source": "config.layout.worker_count",
    "expected_pool_count": CONFIG_LAYOUT["worker_count"],
    "aegiskv_condition_requirement": "enabled_complete_per_pool_trajectory",
    "no_protection_requirement": "absent_or_explicit_disabled_zero",
    "pressure_sources": [
        "cuda_free_bytes",
        "peer_hbm_allocation_reclaim_stall",
    ],
    "controller_law": (
        "target_local_multiplicative_decrease_additive_increase_"
        "stable_window_quota_v1"
    ),
    "raw_local_operation_samples_remain_in_hashed_strict_audit": True,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


class A100DP2ContractError(ValueError):
    """Raised whenever an A100 artifact fails the independent contract."""


@dataclass(frozen=True)
class A100DP2InputPaths:
    """Complete high-level input set for one A100-DP2 preflight."""

    source_manifest: Path
    runtime_binding: Path
    model_submission_gate: Path
    config_manifest: Path
    config_verification: Path
    physical_profile: Path
    physical_profile_audit: Path
    aimd_stall_calibration: Path
    mechanism_gate: Path

    def resolved(self) -> "A100DP2InputPaths":
        return A100DP2InputPaths(
            **{
                name: Path(getattr(self, name)).resolve()
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class A100DP2PhysicalProfile:
    """A100-only normalized view of the schema-v2 physical coefficients."""

    artifact_path: Path
    artifact_sha256: str
    model_revision: str
    recompute_ms_per_token: float
    recompute_fixed_overhead_ms: float
    peer_hbm_protection_per_transfer_setup_ms: float
    peer_hbm_protection_first_transfer_warmup_ms: float
    peer_hbm_protection_steady_state_bytes_per_ms: float
    direct_dram_protection_per_transfer_setup_ms: float
    direct_dram_protection_first_transfer_warmup_ms: float
    direct_dram_protection_steady_state_bytes_per_ms: float
    peer_hbm_recovery_per_transfer_setup_ms: float
    peer_hbm_recovery_first_transfer_warmup_ms: float
    peer_hbm_recovery_steady_state_bytes_per_ms: float
    dram_recovery_per_transfer_setup_ms: float
    dram_recovery_first_transfer_warmup_ms: float
    dram_recovery_steady_state_bytes_per_ms: float

    @property
    def metrics(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in sorted(PROFILE_METRICS)
        }


@dataclass(frozen=True)
class A100DP2PreflightResult:
    """Validated preflight metadata; deliberately contains no runtime object."""

    inputs: A100DP2InputPaths
    profile: A100DP2PhysicalProfile
    input_artifacts: Mapping[str, Mapping[str, Any]]
    source_entries_sha256: str
    condition_profile_artifacts: tuple[Mapping[str, Any], ...]
    strict_audit_artifacts: tuple[Mapping[str, Any], ...]
    aimd_local_stall_ms: float
    aimd_runtime_config: Mapping[str, Any]
    lineage_identity_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "evidence_class": PREFLIGHT_EVIDENCE_CLASS,
            "status": "PASS",
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "hardware_profile": HARDWARE_PROFILE,
            "model": MODEL,
            "dtype": "fp16",
            "layout": CONFIG_LAYOUT,
            "design_semantics": DESIGN_SEMANTICS,
            "physical_profile_semantics": PHYSICAL_PROFILE_SEMANTICS,
            "runtime_admission_decision_wire_values": WIRE_VALUES,
            "profile_schema_class": "A100DP2PhysicalProfile",
            "consumer_compatibility_identity": json.loads(
                json.dumps(A100_CONSUMER_COMPATIBILITY_IDENTITY)
            ),
            "transfer_cost_model_schema": TRANSFER_COST_MODEL_SCHEMA,
            "runtime_consumption_contract": RUNTIME_CONSUMPTION_CONTRACT,
            "profile_metrics": self.profile.metrics,
            "model_revision": self.profile.model_revision,
            "source_entries_sha256": self.source_entries_sha256,
            "lineage_identity_sha256": self.lineage_identity_sha256,
            "input_artifacts": {
                key: dict(value) for key, value in self.input_artifacts.items()
            },
            "condition_profiles": [
                dict(value) for value in self.condition_profile_artifacts
            ],
            "condition_strict_audits": [
                dict(value) for value in self.strict_audit_artifacts
            ],
            "aimd_local_stall_ms": self.aimd_local_stall_ms,
            "aimd_runtime_config": dict(self.aimd_runtime_config),
            "aimd_consumption_contract": AIMD_CONSUMPTION_CONTRACT,
            "performance_window_contract": {
                "post_workload_aimd_observation_s": 25.0,
                "observation_tail_excluded_from_performance_window": True,
            },
            "authorization": {
                "suite_binding_authorized": True,
                "trace_loading_authorized": False,
                "model_fitting_authorized": False,
                "replay_authorized": False,
                "figure_generation_authorized": False,
                "performance_claim_authorized": False,
                "formal_ready": False,
                "paper_ready": False,
            },
            "evidence_boundary": (
                "A100-DP2 artifact-consumption preflight only. A PASS validates "
                "physical and mechanism lineage but does not run or authorize "
                "SGLSim replay, fitting, figures, or a performance claim."
            ),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A100DP2ContractError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A100DP2ContractError(f"{path} must contain one JSON object")
    return value


def artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise A100DP2ContractError(f"artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A100DP2ContractError(message)


def _artifact_path(
    value: Any, *, label: str, expected_path: Path | None = None
) -> Path:
    _require(isinstance(value, Mapping), f"{label} artifact reference is absent")
    reference = _mapping(value)
    path = Path(str(reference.get("path") or "")).resolve()
    _require(path.is_file(), f"{label} artifact is missing: {path}")
    expected_sha = str(reference.get("sha256") or "")
    _require(_SHA256_RE.fullmatch(expected_sha) is not None, f"{label} sha invalid")
    _require(sha256_file(path) == expected_sha, f"{label} artifact hash mismatch")
    if "size_bytes" in reference:
        _require(
            path.stat().st_size == _integer(reference.get("size_bytes"), -1),
            f"{label} artifact size mismatch",
        )
    if expected_path is not None:
        _require(path == expected_path.resolve(), f"{label} artifact path mismatch")
    return path


def _reference_sha(value: Any, *, label: str) -> str:
    _artifact_path(value, label=label)
    return str(_mapping(value).get("sha256") or "")


def _require_contained(path: Path, root: Path, *, label: str) -> None:
    _require(
        path == root or root in path.parents,
        f"{label} escaped the isolated execution root",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise A100DP2ContractError(
                f"invalid JSONL row {path}:{line_number}"
            ) from exc
        _require(isinstance(value, dict), f"non-object JSONL row {path}:{line_number}")
        rows.append(value)
    return rows


def _validate_source_manifest(
    manifest_path: Path, verification_path: Path
) -> tuple[dict[str, Any], str]:
    manifest = load_object(manifest_path)
    entries = manifest.get("entries")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("evidence_class")
        == "aegiskv_source_and_native_binary_manifest"
        and manifest.get("status") == "PINNED"
        and manifest.get("inventory_policy") == "exact_selected_file_set"
        and isinstance(entries, list)
        and bool(entries)
        and all(isinstance(item, Mapping) for item in entries)
        and manifest.get("entry_count") == len(entries),
        "A100 source manifest is not a nonempty exact PINNED manifest",
    )
    entries_sha = canonical_sha256(entries)
    _require(
        entries_sha == manifest.get("entries_sha256"),
        "A100 source manifest canonical entries digest mismatch",
    )
    repository_root = Path(str(manifest.get("repository_root") or "")).resolve()
    _require(repository_root.is_dir(), "A100 source repository root is missing")
    observed_relatives: set[str] = set()
    for index, raw in enumerate(entries):
        entry = _mapping(raw)
        relative = str(entry.get("relative_path") or "")
        _require(
            relative and relative not in observed_relatives,
            "duplicate source entry",
        )
        observed_relatives.add(relative)
        relative_path = Path(relative)
        _require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"invalid source entry path at index {index}",
        )
        path = (repository_root / relative_path).resolve()
        _require(
            repository_root in path.parents and path.is_file(),
            f"source entry is missing: {relative}",
        )
        _require(
            sha256_file(path) == str(entry.get("sha256") or ""),
            f"source entry hash mismatch: {relative}",
        )
        _require(
            path.stat().st_size == _integer(entry.get("size_bytes"), -1),
            f"source entry size mismatch: {relative}",
        )

    verification = load_object(verification_path)
    verified_manifest = _mapping(verification.get("manifest"))
    _require(
        verification.get("evidence_class")
        == "aegiskv_source_manifest_verification"
        and verification.get("status") == "PASS"
        and verification.get("failures") == []
        and verified_manifest.get("sha256") == sha256_file(manifest_path)
        and verification.get("entries_sha256") == entries_sha,
        "A100 source manifest verification did not pass",
    )
    _artifact_path(
        verified_manifest,
        label="source verification manifest",
        expected_path=manifest_path,
    )
    return manifest, entries_sha


def _validate_runtime_binding(
    path: Path,
    *,
    source_manifest_path: Path,
    source_verification_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    binding = load_object(path)
    runtime_identity = _mapping(binding.get("runtime_identity"))
    runtime_artifacts = _mapping(runtime_identity.get("artifacts"))
    execution_root = Path(str(binding.get("execution_root") or "")).resolve()
    clean_job_id = binding.get("clean_job_id")
    _require(
        binding.get("schema_version") == 1
        and binding.get("evidence_class") == RUNTIME_BINDING_EVIDENCE_CLASS
        and binding.get("status") == "PASS"
        and binding.get("failures") == []
        and binding.get("campaign_id") == CAMPAIGN_ID
        and binding.get("hardware_label") == HARDWARE_LABEL
        and binding.get("design_semantics") == DESIGN_SEMANTICS
        and binding.get("activation_revision") == ACTIVATION_REVISION
        and isinstance(clean_job_id, int)
        and not isinstance(clean_job_id, bool)
        and clean_job_id > 0
        and binding.get("model_execution_probe_submission_authorized") is False
        and binding.get("physical_calibration_submission_authorized") is False
        and binding.get("mechanism_submission_authorized") is False
        and binding.get("runtime_or_physical_result") is False
        and binding.get("formal_ready") is False
        and binding.get("paper_ready") is False
        and execution_root.is_dir()
        and (
            path.resolve() == execution_root
            or execution_root in path.resolve().parents
        )
        and runtime_identity.get("execution_root") == str(execution_root)
        and set(runtime_artifacts) == RUNTIME_ARTIFACT_NAMES
        and _mapping(runtime_identity.get("versions"))
        == {
            "vllm": "1.2.2+aegiskv.sm89",
            "torch": "2.10.0+cu128",
            "cuda": "12.8",
        },
        "A100 v46 runtime binding is invalid",
    )

    source_activation_path = _artifact_path(
        binding.get("source_activation"), label="runtime-binding source activation"
    )
    _require_contained(
        source_activation_path,
        execution_root,
        label="runtime-binding source activation",
    )
    source_activation = load_object(source_activation_path)
    _require(
        source_activation.get("evidence_class")
        == "a100_dp2_v19_source_activation_gate"
        and source_activation.get("status") == "PASS"
        and source_activation.get("failures") == []
        and source_activation.get("campaign_id") == CAMPAIGN_ID
        and source_activation.get("hardware_label") == HARDWARE_LABEL
        and source_activation.get("design_semantics") == DESIGN_SEMANTICS
        and source_activation.get("activation_revision") == ACTIVATION_REVISION
        and source_activation.get("jobs_submitted") == [],
        "A100 v46 source activation is not current-design source-only PASS evidence",
    )
    _artifact_path(
        source_activation.get("source_manifest"),
        label="source-activation source manifest",
        expected_path=source_manifest_path,
    )
    _artifact_path(
        source_activation.get("source_verification"),
        label="source-activation source verification",
        expected_path=source_verification_path,
    )

    clean_submission_path = _artifact_path(
        binding.get("clean_preflight_submission"),
        label="runtime-binding clean submission",
    )
    _require_contained(
        clean_submission_path,
        execution_root,
        label="runtime-binding clean submission",
    )
    clean_submission = load_object(clean_submission_path)
    clean_lineage = _mapping(clean_submission.get("lineage"))
    clean_gate_ref = _mapping(clean_lineage.get("submission_gate"))
    expected_clean_comment = (
        "aegis-a100-v45-clean-"
        f"{str(clean_gate_ref.get('sha256') or '')[:12]}"
    )
    _require(
        clean_submission.get("evidence_class")
        == "a100_dp2_clean_preflight_submission_v45"
        and clean_submission.get("status") == "QUEUED_NO_RUNTIME"
        and clean_submission.get("campaign_id") == CAMPAIGN_ID
        and clean_submission.get("hardware_label") == HARDWARE_LABEL
        and clean_submission.get("design_semantics") == DESIGN_SEMANTICS
        and clean_submission.get("clean_job_id") == clean_job_id
        and clean_submission.get("slurm_comment") == expected_clean_comment
        and clean_submission.get("clean_preflight_submission_authorized") is True
        and clean_submission.get("model_execution_probe_submission_authorized")
        is False
        and clean_submission.get("runtime_or_physical_result") is False
        and set(clean_lineage) >= {"submission_gate", "clean_launcher"},
        "A100 v46 clean-preflight submission lineage is invalid",
    )
    for name in ("submission_gate", "clean_launcher"):
        lineage_path = _artifact_path(
            clean_lineage[name], label=f"clean-preflight lineage {name}"
        )
        _require_contained(
            lineage_path,
            execution_root,
            label=f"clean-preflight lineage {name}",
        )

    clean_path = _artifact_path(
        binding.get("clean_preflight"), label="runtime-binding clean preflight"
    )
    _require_contained(
        clean_path, execution_root, label="runtime-binding clean preflight"
    )
    clean = load_object(clean_path)
    clean_artifacts = _mapping(clean.get("artifacts"))
    _require(
        clean.get("evidence_class") == "a100_dp2_clean_gpu_preflight_audit"
        and clean.get("status") == "PASS"
        and clean.get("failures") == []
        and clean.get("campaign_id") == CAMPAIGN_ID
        and clean.get("hardware_label") == HARDWARE_LABEL
        and str(clean.get("slurm_job_id")) == str(clean_job_id)
        and set(_mapping(clean.get("gates"))) == EXPECTED_CLEAN_GATES
        and all(value is True for value in _mapping(clean.get("gates")).values())
        and bool(clean_artifacts),
        "A100 v46 clean-preflight audit is invalid",
    )
    for name, value in clean_artifacts.items():
        artifact_path = _artifact_path(
            value, label=f"clean-preflight artifact {name}"
        )
        _require_contained(
            artifact_path,
            execution_root,
            label=f"clean-preflight artifact {name}",
        )

    scheduler_path = _artifact_path(
        binding.get("clean_scheduler_final"),
        label="runtime-binding clean scheduler final",
    )
    _require_contained(
        scheduler_path,
        execution_root,
        label="runtime-binding clean scheduler final",
    )
    scheduler_rows = [
        line.strip()
        for line in scheduler_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scheduler_fields = scheduler_rows[0].split("|") if len(scheduler_rows) == 1 else []
    elapsed_seconds = (
        _integer(scheduler_fields[3], -1) if len(scheduler_fields) == 6 else -1
    )
    _require(
        len(scheduler_fields) == 6
        and scheduler_fields[0] == str(clean_job_id)
        and scheduler_fields[1] == "COMPLETED"
        and scheduler_fields[2] == "0:0"
        and 1 <= elapsed_seconds <= 600
        and scheduler_fields[4] not in {"", "None assigned", "Unknown"}
        and any(
            token in scheduler_fields[5]
            for token in ("gres/gpu=2", "gres/gpu:a100=2")
        ),
        "A100 v46 clean scheduler evidence is invalid",
    )

    runtime_snapshot_path = _artifact_path(
        binding.get("runtime_identity_snapshot"),
        label="runtime-binding identity snapshot",
    )
    _require_contained(
        runtime_snapshot_path,
        execution_root,
        label="runtime-binding identity snapshot",
    )
    runtime_snapshot = load_object(runtime_snapshot_path)
    _require(
        runtime_snapshot.get("evidence_class") == RUNTIME_IDENTITY_EVIDENCE_CLASS
        and runtime_snapshot.get("status") == "PASS"
        and runtime_snapshot.get("campaign_id") == CAMPAIGN_ID
        and runtime_snapshot.get("hardware_label") == HARDWARE_LABEL
        and runtime_snapshot.get("execution_root") == str(execution_root)
        and runtime_snapshot.get("runtime_identity") == runtime_identity
        and runtime_snapshot.get("gpu_or_model_execution_performed") is False
        and runtime_snapshot.get("runtime_or_physical_result") is False
        and runtime_snapshot.get("formal_ready") is False
        and runtime_snapshot.get("paper_ready") is False,
        "A100 v46 runtime identity snapshot is invalid",
    )
    for name, value in runtime_artifacts.items():
        artifact_path = _artifact_path(value, label=f"runtime identity {name}")
        _require(
            artifact_path == execution_root or execution_root in artifact_path.parents,
            f"runtime identity {name} escaped the isolated execution root",
        )
    invocation_raw = Path(
        str(runtime_identity.get("python_invocation_path") or "")
    )
    symlink_target = str(runtime_identity.get("python_symlink_target") or "")
    invocation_path = invocation_raw.resolve()
    _require(
        invocation_raw.is_absolute()
        and ".." not in invocation_raw.parts
        and invocation_raw.is_symlink()
        and symlink_target
        and not Path(symlink_target).is_absolute()
        and os.readlink(invocation_raw) == symlink_target
        and invocation_path == _artifact_path(
            runtime_artifacts["python_executable"], label="runtime Python"
        )
        and (
            invocation_raw == execution_root
            or execution_root in invocation_raw.parents
        )
        and (
            invocation_path == execution_root
            or execution_root in invocation_path.parents
        ),
        "A100 v46 runtime Python is not execution-root-local",
    )

    overlay = _mapping(binding.get("model_probe_overlay"))
    _require(
        set(overlay) == {"model_launcher", "model_coordinator"},
        "A100 v46 model-probe overlay is incomplete",
    )
    launcher_path = _artifact_path(
        overlay.get("model_launcher"), label="v46 model launcher"
    )
    coordinator_path = _artifact_path(
        overlay.get("model_coordinator"), label="v46 model coordinator"
    )
    _require_contained(
        launcher_path, execution_root, label="v46 model launcher"
    )
    _require_contained(
        coordinator_path, execution_root, label="v46 model coordinator"
    )
    launcher_text = launcher_path.read_text(encoding="utf-8")
    _require(
        all(
            token in launcher_text
            for token in (
                "nvidia-smi dmon",
                "--id=0,1",
                "--select=pucmet",
                "--delay=2",
                "--options=DT",
                "--format=csv",
                "--filename=",
                "gpu-telemetry.csv",
                "VLLM_USE_V2_MODEL_RUNNER=1",
                "VLLM_LOGGING_LEVEL=INFO",
            )
        )
        and "pucvmet" not in launcher_text,
        "A100 v46 model launcher lacks the exact two-device CSV telemetry contract",
    )
    return binding, runtime_identity


def _validate_model_submission_gate(
    path: Path,
    *,
    runtime_binding_path: Path,
    runtime_binding: Mapping[str, Any],
) -> dict[str, Any]:
    gate = load_object(path)
    _require(
        gate.get("schema_version") == 1
        and gate.get("evidence_class") == MODEL_SUBMISSION_GATE_EVIDENCE_CLASS
        and gate.get("status") == "PASS"
        and gate.get("failures") == []
        and gate.get("campaign_id") == CAMPAIGN_ID
        and gate.get("hardware_label") == HARDWARE_LABEL
        and gate.get("design_semantics") == DESIGN_SEMANTICS
        and gate.get("activation_revision") == ACTIVATION_REVISION
        and gate.get("execution_root") == runtime_binding.get("execution_root")
        and gate.get("clean_job_id") == runtime_binding.get("clean_job_id")
        and gate.get("model_execution_probe_submission_authorized") is True
        and gate.get("maximum_inflight_model_probes") == 1
        and gate.get("maximum_job_runtime_seconds") == 1200
        and gate.get("physical_calibration_submission_authorized") is False
        and gate.get("mechanism_submission_authorized") is False
        and gate.get("formal_scale_submission_authorized") is False
        and gate.get("runtime_or_physical_result") is False
        and gate.get("jobs_submitted") == []
        and gate.get("formal_ready") is False
        and gate.get("paper_ready") is False,
        "A100 v46 model-submission gate is invalid",
    )
    _artifact_path(
        gate.get("runtime_binding"),
        label="model-submission runtime binding",
        expected_path=runtime_binding_path,
    )
    scheduler_path = _artifact_path(
        gate.get("scheduler_final_artifact"),
        label="model-submission scheduler final",
        expected_path=_artifact_path(
            runtime_binding.get("clean_scheduler_final"),
            label="runtime-binding clean scheduler final",
        ),
    )
    scheduler = _mapping(gate.get("scheduler_final"))
    scheduler_rows = [
        line.strip()
        for line in scheduler_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scheduler_fields = (
        scheduler_rows[0].split("|") if len(scheduler_rows) == 1 else []
    )
    elapsed_seconds = (
        _integer(scheduler_fields[3], -1) if len(scheduler_fields) == 6 else -1
    )
    _require(
        scheduler.get("status") == "PASS"
        and scheduler.get("failures") == []
        and scheduler.get("job_id") == str(runtime_binding.get("clean_job_id"))
        and scheduler.get("state") == "COMPLETED"
        and scheduler.get("exit_code") == "0:0"
        and scheduler.get("elapsed_seconds") == elapsed_seconds
        and scheduler.get("node_list")
        == (scheduler_fields[4] if len(scheduler_fields) == 6 else None)
        and scheduler.get("alloc_tres")
        == (scheduler_fields[5] if len(scheduler_fields) == 6 else None)
        and scheduler.get("raw")
        == (scheduler_rows[0] if len(scheduler_rows) == 1 else None),
        "A100 v46 model-submission scheduler evidence is not reproducible",
    )
    execution_root = Path(str(runtime_binding.get("execution_root") or "")).resolve()
    builder_path = _artifact_path(
        gate.get("gate_builder"), label="v46 model-submission gate builder"
    )
    _require(
        builder_path.name == "build_model_submission_gate_v46.py",
        "A100 v46 model-submission gate builder is invalid",
    )
    _require_contained(
        builder_path,
        execution_root,
        label="v46 model-submission gate builder",
    )
    for name in (
        "source_activation",
        "clean_preflight_submission",
        "clean_preflight",
        "runtime_identity_snapshot",
    ):
        _require(
            gate.get(name) == runtime_binding.get(name),
            f"A100 v46 model-submission {name} crossed runtime lineage",
        )
        _artifact_path(gate.get(name), label=f"model-submission {name}")
    overlay = _mapping(runtime_binding.get("model_probe_overlay"))
    for gate_name, overlay_name in (
        ("model_launcher", "model_launcher"),
        ("model_coordinator", "model_coordinator"),
    ):
        _require(
            gate.get(gate_name) == overlay.get(overlay_name),
            f"A100 v46 {gate_name} crossed runtime lineage",
        )
        _artifact_path(gate.get(gate_name), label=f"v46 {gate_name}")
    return gate


def _validate_model_probe(
    path: Path,
    *,
    runtime_binding_path: Path,
    runtime_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    probe = load_object(path)
    workers = probe.get("worker_results")
    revision = str(probe.get("model_revision") or "")
    model_config_sha = str(probe.get("model_config_sha256") or "")
    observed_runtime_identity = probe.get("runtime_identity")
    lineage = _mapping(probe.get("input_lineage"))
    _require(
        probe.get("evidence_class")
        == "a100_dp2_qwen3_8b_vllm_v2_model_execution_probe"
        and probe.get("status") == "PASS"
        and probe.get("failures") == []
        and probe.get("campaign_id") == CAMPAIGN_ID
        and probe.get("hardware_label") == HARDWARE_LABEL
        and probe.get("model_id") == MODEL
        and probe.get("tp_size") == 1
        and probe.get("pp_size") == 1
        and probe.get("system_dp_size") == 2
        and probe.get("model_runner_v2_observed") is True
        and _REVISION_RE.fullmatch(revision) is not None
        and _SHA256_RE.fullmatch(model_config_sha) is not None
        and observed_runtime_identity == runtime_identity
        and probe.get("formal_ready") is False
        and probe.get("paper_ready") is False
        and isinstance(workers, list)
        and len(workers) == 2,
        "A100 model execution probe identity is invalid",
    )
    _artifact_path(
        lineage.get("runtime_binding"),
        label="model-probe runtime binding",
        expected_path=runtime_binding_path,
    )
    ranks: set[int] = set()
    devices: set[str] = set()
    for index, raw in enumerate(workers):
        worker = _mapping(raw)
        ranks.add(_integer(worker.get("logical_rank"), -1))
        devices.add(str(worker.get("cuda_visible_devices") or ""))
        _require(
            worker.get("evidence_class")
            == "a100_qwen3_8b_vllm_v2_worker_execution"
            and worker.get("status") == "PASS"
            and worker.get("gpu_capability") == [8, 0]
            and "A100" in str(worker.get("gpu_name") or "")
            and worker.get("model_id") == MODEL
            and worker.get("model_revision") == revision
            and worker.get("model_config_sha256") == model_config_sha
            and worker.get("model_runner_v2_requested") is True
            and worker.get("model_runner_v2_engine_config_selected") is True
            and worker.get("model_runner_v2_log_marker_observed") is True
            and worker.get("model_runner_v2_observed") is True
            and worker.get("runtime_identity") == runtime_identity,
            f"A100 model worker {index} identity/V2 observation is invalid",
        )
    _require(ranks == {0, 1}, "A100 model probe lacks two unique logical workers")
    _require(
        len(devices) == 2 and "" not in devices,
        "A100 model probe lacks two unique CUDA devices",
    )
    return probe, revision


def _validate_config(
    manifest_path: Path,
    verification_path: Path,
    source_manifest_path: Path,
    source_verification_path: Path,
    runtime_binding_path: Path,
    runtime_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Path]:
    manifest = load_object(manifest_path)
    configuration = manifest.get("configuration")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("evidence_class")
        == "aegiskv_physical_campaign_config_manifest"
        and manifest.get("status") == "PINNED"
        and manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("design_semantics") == DESIGN_SEMANTICS
        and manifest.get("hardware_label") == HARDWARE_LABEL
        and manifest.get("model") == MODEL
        and manifest.get("model_runner_v2_observed") is True
        and manifest.get("layout") == CONFIG_LAYOUT
        and manifest.get("hardware_profile") == HARDWARE_PROFILE
        and isinstance(configuration, Mapping)
        and canonical_sha256(configuration) == manifest.get("configuration_sha256")
        and manifest.get("jobs_submitted") == []
        and manifest.get("formal_ready") is False
        and manifest.get("paper_ready") is False,
        "A100 config manifest identity is invalid",
    )
    _artifact_path(
        manifest.get("source_manifest"),
        label="config source manifest",
        expected_path=source_manifest_path,
    )
    _artifact_path(
        manifest.get("source_verification"),
        label="config source verification",
        expected_path=source_verification_path,
    )
    config_paths = _mapping(configuration).get("source_owned_paths")
    _require(
        config_paths
        == {
            "peer_hbm_protection": "source_worker_d2d_p2p",
            "direct_dram_protection": "source_worker_d2h",
            "peer_hbm_recovery": "survivor_worker_hbm_load",
            "dram_recovery": "survivor_worker_store_load",
            "target_owned_or_hbm_to_dram_path_allowed": False,
        },
        "A100 config does not use independent source-owned HBM/DRAM paths",
    )
    configuration_map = _mapping(configuration)
    capacity = _mapping(configuration_map.get("capacity"))
    aimd = _mapping(configuration_map.get("aimd"))
    required_aimd_fields = _AIMD_CONFIG_FIELDS - {
        "total_hbm_bytes",
        "local_stall_ms",
    }
    aimd_numbers = {
        name: _number(aimd.get(name))
        for name in (
            "interval_s",
            "initial_alpha",
            "min_alpha",
            "max_alpha",
            "decrease_factor",
            "increase_step",
        )
    }
    _require(
        required_aimd_fields <= set(aimd)
        and isinstance(capacity.get("peer_hbm_bytes_per_target"), int)
        and not isinstance(capacity.get("peer_hbm_bytes_per_target"), bool)
        and _integer(capacity.get("peer_hbm_bytes_per_target"), 0) > 0
        and isinstance(aimd.get("pressure_free_bytes"), int)
        and not isinstance(aimd.get("pressure_free_bytes"), bool)
        and _integer(aimd.get("pressure_free_bytes"), 0) > 0
        and all(
            isinstance(aimd.get(name), (int, float))
            and not isinstance(aimd.get(name), bool)
            and math.isfinite(value)
            for name, value in aimd_numbers.items()
        )
        and aimd_numbers["interval_s"] == 5.0
        and 0.0
        <= aimd_numbers["min_alpha"]
        <= aimd_numbers["initial_alpha"]
        <= aimd_numbers["max_alpha"]
        <= 1.0
        and 0.0 < aimd_numbers["decrease_factor"] < 1.0
        and aimd_numbers["increase_step"] > 0.0
        and isinstance(aimd.get("stable_window_threshold"), int)
        and not isinstance(aimd.get("stable_window_threshold"), bool)
        and _integer(aimd.get("stable_window_threshold"), -1) == 3,
        "A100 config AIMD observation contract changed",
    )
    model_probe_path = _artifact_path(
        manifest.get("model_execution_probe"), label="A100 model execution probe"
    )
    _probe, revision = _validate_model_probe(
        model_probe_path,
        runtime_binding_path=runtime_binding_path,
        runtime_identity=runtime_identity,
    )
    _require(
        manifest.get("model_revision") == revision,
        "A100 config/model-probe revision mismatch",
    )

    verification = load_object(verification_path)
    _require(
        verification.get("schema_version") == 1
        and verification.get("evidence_class")
        == "a100_dp2_config_manifest_verification"
        and verification.get("status") == "PASS"
        and verification.get("failures") == []
        and verification.get("campaign_id") == CAMPAIGN_ID
        and verification.get("hardware_label") == HARDWARE_LABEL
        and verification.get("design_semantics") == DESIGN_SEMANTICS
        and verification.get("jobs_submitted") == []
        and verification.get("formal_ready") is False
        and verification.get("paper_ready") is False,
        "A100 config verification did not pass",
    )
    _artifact_path(
        verification.get("manifest"),
        label="verified A100 config manifest",
        expected_path=manifest_path,
    )
    return manifest, revision, model_probe_path


def _normalize_metrics(metrics: Any) -> dict[str, float]:
    _require(isinstance(metrics, Mapping), "A100 physical metrics are absent")
    _require(set(metrics) == PROFILE_METRICS, "A100 physical metric set changed")
    normalized = {name: _number(metrics[name]) for name in PROFILE_METRICS}
    for name, value in normalized.items():
        nonnegative = name == "recompute_fixed_overhead_ms" or name.endswith(
            ("_per_transfer_setup_ms", "_first_transfer_warmup_ms")
        )
        valid = math.isfinite(value) and (
            value >= 0.0 if nonnegative else value > 0.0
        )
        _require(valid, f"A100 physical metric is invalid: {name}")
    return normalized


def _validate_gpu_telemetry_csv(path: Path, *, label: str) -> dict[str, Any]:
    """Independently validate exact two-device ``nvidia-smi dmon`` CSV.

    The producer summary is deliberately not trusted for per-device activity.
    Every complete frame must contain GPU 0 and GPU 1, and each GPU must have
    multiple samples with independently positive SM utilization and FB usage.
    This rejects the historical whitespace ``pucvmet`` output as well as an
    aggregate-only nonzero summary that hides an inactive device.
    """

    try:
        rows = [
            [field.strip() for field in row]
            for row in csv.reader(path.read_text(encoding="utf-8").splitlines())
            if row
        ]
    except OSError as exc:
        raise A100DP2ContractError(f"cannot read {label}: {exc}") from exc
    _require(
        len(rows) >= 6
        and tuple(rows[0]) == GPU_TELEMETRY_COLUMNS
        and tuple(rows[1]) == GPU_TELEMETRY_UNITS,
        f"{label} is not exact --select=pucmet --format=csv dmon telemetry",
    )

    frames: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    index = 2
    while index < len(rows):
        fields = rows[index]
        if tuple(fields) == GPU_TELEMETRY_COLUMNS:
            _require(
                index + 1 < len(rows)
                and tuple(rows[index + 1]) == GPU_TELEMETRY_UNITS,
                f"{label} has an unpaired repeated dmon header",
            )
            index += 2
            continue
        _require(
            tuple(fields) != GPU_TELEMETRY_UNITS
            and len(fields) == len(GPU_TELEMETRY_COLUMNS),
            f"{label} contains a non-CSV or incomplete dmon row",
        )
        record = dict(zip(GPU_TELEMETRY_COLUMNS, fields, strict=True))
        date = str(record["#Date"])
        clock = str(record["Time"])
        gpu_id = str(record["gpu"])
        _require(
            re.fullmatch(r"\d{8}", date) is not None
            and re.fullmatch(r"\d{2}:\d{2}:\d{2}", clock) is not None
            and gpu_id in {"0", "1"},
            f"{label} has an invalid timestamp or GPU identity",
        )
        numeric: dict[str, float] = {}
        for name in ("pwr", "gtemp", "sm", "mem", "fb", "pci", "rxpci", "txpci"):
            value = _number(record.get(name))
            _require(
                math.isfinite(value) and value >= 0.0,
                f"{label} has invalid {name} for GPU {gpu_id}",
            )
            numeric[name] = value
        timestamp = f"{date} {clock}"
        _require(
            gpu_id not in frames[timestamp],
            f"{label} has duplicate GPU {gpu_id} in one frame",
        )
        frames[timestamp][gpu_id] = (numeric["sm"], numeric["fb"])
        index += 1

    ordered = sorted(frames.items())
    _require(
        len(ordered) >= 2
        and all(set(frame) == {"0", "1"} for _timestamp, frame in ordered),
        f"{label} lacks multiple complete frames for each A100",
    )
    timestamps = [
        datetime.strptime(timestamp, "%Y%m%d %H:%M:%S")
        for timestamp, _frame in ordered
    ]
    _require(
        all(
            0.0 < (current - previous).total_seconds() <= 30.0
            for previous, current in zip(
                timestamps[:-1], timestamps[1:], strict=True
            )
        ),
        f"{label} has invalid or unordered dmon frame timestamps",
    )
    per_gpu: dict[str, dict[str, float | int]] = {}
    for gpu_id in ("0", "1"):
        samples = [frame[gpu_id] for _timestamp, frame in ordered]
        peak_sm = max(value[0] for value in samples)
        peak_fb = max(value[1] for value in samples)
        _require(
            len(samples) >= 2 and peak_sm > 0.0 and peak_fb > 0.0,
            f"{label} lacks independent SM/FB activity for GPU {gpu_id}",
        )
        per_gpu[gpu_id] = {
            "frame_count": len(samples),
            "peak_gpu_utilization_pct": peak_sm,
            "peak_fb_memory_mb": peak_fb,
        }
    return {"complete_frame_count": len(ordered), "per_gpu": per_gpu}


def _validate_producer_per_gpu_telemetry(
    telemetry: Any, *, recomputed: Mapping[str, Any], label: str
) -> None:
    """Require the producer's per-device audit and reproduce every value."""

    declared = _mapping(telemetry)
    devices = _mapping(declared.get("devices"))
    recomputed_devices = _mapping(recomputed.get("per_gpu"))
    _require(
        declared.get("status") == "PASS"
        and declared.get("failures") == []
        and declared.get("observed_gpu_ids") == ["0", "1"]
        and _integer(declared.get("complete_frame_count"), 0) >= 2
        and _integer(declared.get("complete_frame_count"), -1)
        == _integer(recomputed.get("complete_frame_count"), -2)
        and set(devices) == {"0", "1"}
        and set(recomputed_devices) == {"0", "1"},
        f"{label} producer per_gpu_telemetry check did not pass",
    )
    for gpu_id in ("0", "1"):
        device = _mapping(devices[gpu_id])
        independently_observed = _mapping(recomputed_devices[gpu_id])
        declared_sm = _number(device.get("peak_gpu_utilization_pct"), 0.0)
        declared_fb = _number(device.get("peak_fb_memory_mb"), 0.0)
        _require(
            declared_sm > 0.0
            and declared_fb > 0.0
            and math.isclose(
                declared_sm,
                _number(
                    independently_observed.get("peak_gpu_utilization_pct")
                ),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                declared_fb,
                _number(independently_observed.get("peak_fb_memory_mb")),
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            f"{label} producer per_gpu_runtime_activity mismatch for GPU {gpu_id}",
        )


def _validate_declared_gpu_telemetry(
    telemetry: Any,
    *,
    raw_reference: Any,
    label: str,
    require_workload_window: bool,
) -> dict[str, Any]:
    declared = _mapping(telemetry)
    raw_path = _artifact_path(raw_reference, label=f"{label} raw CSV")
    declared_artifact = _mapping(declared.get("artifact"))
    _require(
        declared.get("status") == "PASS"
        and declared.get("failures") == []
        and declared.get("expected_device_count") == 2
        and declared.get("gpu_ids") == ["0", "1"]
        and _integer(declared.get("complete_frame_count"), 0) >= 2
        and _number(declared.get("peak_gpu_utilization_pct"), 0.0) > 0.0
        and _number(declared.get("peak_total_fb_memory_mb"), 0.0) > 0.0
        and declared_artifact.get("path") == str(raw_path)
        and declared_artifact.get("sha256") == sha256_file(raw_path),
        f"{label} declared telemetry is invalid or crossed",
    )
    if require_workload_window:
        workload = _mapping(declared.get("workload_window"))
        _require(
            workload.get("status") == "PASS"
            and workload.get("failures") == []
            and _integer(workload.get("complete_frame_count"), 0) >= 2
            and _number(workload.get("duration_s"), 0.0) > 0.0,
            f"{label} workload telemetry window is invalid",
        )
    return _validate_gpu_telemetry_csv(raw_path, label=label)


def _validate_physical_profile(
    path: Path,
    *,
    source_path: Path,
    config_path: Path,
    config_verification_path: Path,
    model_revision: str,
) -> tuple[dict[str, Any], A100DP2PhysicalProfile]:
    profile = load_object(path)
    metrics = _normalize_metrics(profile.get("metrics"))
    compatibility = _mapping(profile.get("compatibility_identity"))
    _require(
        profile.get("schema_version") == 2
        and profile.get("evidence_class") == PHYSICAL_PROFILE_EVIDENCE_CLASS
        and profile.get("status") == "passed"
        and profile.get("profile_semantics") == PHYSICAL_PROFILE_SEMANTICS
        and profile.get("design_semantics") == DESIGN_SEMANTICS
        and profile.get("campaign_id") == CAMPAIGN_ID
        and profile.get("hardware_label") == HARDWARE_LABEL
        and profile.get("hardware_profile") == HARDWARE_PROFILE
        and profile.get("layout") == PHYSICAL_LAYOUT
        and profile.get("model") == MODEL
        and profile.get("model_revision") == model_revision
        and compatibility.get("serving_dtype") == "half"
        and compatibility.get("kv_cache_dtype") == "fp16"
        and compatibility.get("layout") == PHYSICAL_LAYOUT
        and profile.get("runtime_consumption_contract")
        == RUNTIME_CONSUMPTION_CONTRACT
        and profile.get("failures") == []
        and profile.get("formal_ready") is False
        and profile.get("paper_ready") is False,
        "A100 schema-v2 physical profile contract did not pass",
    )
    _artifact_path(
        profile.get("source_manifest"),
        label="physical-profile source manifest",
        expected_path=source_path,
    )
    _artifact_path(
        profile.get("config_manifest"),
        label="physical-profile config manifest",
        expected_path=config_path,
    )
    _artifact_path(
        profile.get("config_verification"),
        label="physical-profile config verification",
        expected_path=config_verification_path,
    )
    normalized = A100DP2PhysicalProfile(
        artifact_path=path,
        artifact_sha256=sha256_file(path),
        model_revision=model_revision,
        **metrics,
    )
    return profile, normalized


def _run_derived_calibration_stream_id(run_id: str) -> str:
    return f"slurm-job:{run_id}"


def _validate_run_transfer_records(
    records: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    run_id: str,
    stream_id: str,
) -> None:
    """Prove that one physical run contributes exactly one transport stream."""

    expected_events = TRANSFER_TREATMENT_REQUIRED_EVENTS[treatment]
    transfer_events = set(TRANSFER_PATH_EVENTS.values())
    observed_events: set[str] = set()
    by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        event = str(record.get("event") or "")
        if event not in transfer_events:
            continue
        observed_events.add(event)
        by_event[event].append(record)
    _require(
        expected_events <= observed_events,
        f"A100 condition-local transfer run lacks required events: {treatment}",
    )
    for event, event_records in by_event.items():
        ordinals: list[int] = []
        for record in event_records:
            record_run_id = str(record.get("calibration_run_id") or "").strip()
            record_stream_id = str(
                record.get("calibration_stream_id") or ""
            ).strip()
            ordinal = _integer(record.get("calibration_transfer_ordinal"), -1)
            _require(
                record_run_id == run_id and record_stream_id == stream_id,
                f"A100 transfer event stream is not run-derived: {treatment}/{event}",
            )
            _require(
                ordinal >= 0,
                f"invalid affine physical calibration sample: {event}",
            )
            ordinals.append(ordinal)
        _require(
            ordinals == list(range(len(ordinals))),
            f"A100 calibration ordinals do not follow issue order: {treatment}/{event}",
        )


def _affine_transfer_cost_model(
    records: Sequence[Mapping[str, Any]],
    event: str,
    *,
    expected_stream_ids: set[str],
) -> dict[str, Any]:
    """Fit setup, first-transfer warm-up, and steady-state slope.

    Only successful physical samples are eligible.  Every eligible record must
    carry the one stream identity derived from their physical Slurm run and a
    transfer ordinal in actual issue order.  A request/chunk/worker identity is
    intentionally not accepted as a stream.  Ordinal-zero samples estimate
    the additional one-time warm-up after fitting an affine per-transfer model
    to later samples with at least two payload sizes.
    """

    loaded = event.endswith("_load")
    byte_field = "loaded_bytes" if loaded else "bytes"
    samples: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        if record.get("event") != event:
            continue
        success = (
            record.get("loaded") is True
            if loaded
            else record.get("transfer_succeeded") is True
        )
        if not success:
            continue
        byte_count = _integer(record.get(byte_field), 0)
        duration_ms = _number(record.get("duration_ms"))
        stream_id = str(record.get("calibration_stream_id") or "").strip()
        ordinal = _integer(record.get("calibration_transfer_ordinal"), -1)
        _require(
            byte_count > 0
            and math.isfinite(duration_ms)
            and duration_ms > 0.0
            and bool(stream_id)
            and ordinal >= 0
            and (not loaded or byte_count == _integer(record.get("bytes"), -1)),
            f"invalid affine physical calibration sample: {event}",
        )
        identity = (stream_id, ordinal)
        _require(
            identity not in seen,
            f"duplicate affine physical calibration sample: {event} {identity}",
        )
        seen.add(identity)
        samples.append(
            {
                "stream_id": stream_id,
                "ordinal": ordinal,
                "bytes": byte_count,
                "duration_ms": duration_ms,
            }
        )
    _require(bool(samples), f"missing physical calibration samples: {event}")
    streams = {str(sample["stream_id"]) for sample in samples}
    warmup = [sample for sample in samples if sample["ordinal"] == 0]
    steady = [sample for sample in samples if sample["ordinal"] > 0]
    warmup_streams = {str(sample["stream_id"]) for sample in warmup}
    steady_streams = {str(sample["stream_id"]) for sample in steady}
    distinct_payloads = sorted({int(sample["bytes"]) for sample in steady})
    _require(
        streams == expected_stream_ids
        and len(streams) == TRANSFER_REPEATS_PER_TREATMENT
        and warmup_streams == streams
        and steady_streams == streams
        and len(warmup) >= 2
        and len(steady) >= 4
        and len(distinct_payloads) >= 2,
        f"affine physical calibration lacks warm-up/steady-state coverage: {event}",
    )
    for stream_id in sorted(streams):
        stream_ordinals = [
            int(sample["ordinal"])
            for sample in samples
            if sample["stream_id"] == stream_id
        ]
        _require(
            stream_ordinals == list(range(len(stream_ordinals))),
            f"affine physical calibration stream is not one ordered run: {event}",
        )

    xs = [float(sample["bytes"]) for sample in steady]
    ys = [float(sample["duration_ms"]) for sample in steady]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    _require(denominator > 0.0, f"affine physical fit is singular: {event}")
    slope_ms_per_byte = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys, strict=True)
    ) / denominator
    per_transfer_setup_ms = mean_y - slope_ms_per_byte * mean_x
    predictions = [
        per_transfer_setup_ms + slope_ms_per_byte * value for value in xs
    ]
    residual_sum_squares = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(ys, predictions, strict=True)
    )
    total_sum_squares = sum((value - mean_y) ** 2 for value in ys)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else 1.0
    )
    _require(
        math.isfinite(slope_ms_per_byte)
        and slope_ms_per_byte > 0.0
        and math.isfinite(per_transfer_setup_ms)
        and per_transfer_setup_ms >= 0.0
        and math.isfinite(r_squared)
        and r_squared >= 0.80,
        f"affine physical fit is invalid: {event}",
    )
    warmup_residuals = [
        float(sample["duration_ms"])
        - (
            per_transfer_setup_ms
            + slope_ms_per_byte * float(sample["bytes"])
        )
        for sample in warmup
    ]
    first_transfer_warmup_ms = statistics.median(warmup_residuals)
    _require(
        math.isfinite(first_transfer_warmup_ms)
        and first_transfer_warmup_ms >= 0.0,
        f"first-transfer warm-up fit is invalid: {event}",
    )
    return {
        "schema": TRANSFER_COST_MODEL_SCHEMA,
        "duration_formula": TRANSFER_DURATION_FORMULA,
        "event": event,
        "sample_count": len(samples),
        "stream_count": len(streams),
        "stream_ids": sorted(streams),
        "warmup_sample_count": len(warmup),
        "steady_state_sample_count": len(steady),
        "distinct_steady_state_payload_bytes": distinct_payloads,
        "fit_r_squared": r_squared,
        "per_transfer_setup_ms": per_transfer_setup_ms,
        "first_transfer_warmup_ms": first_transfer_warmup_ms,
        "steady_state_bytes_per_ms": 1.0 / slope_ms_per_byte,
    }


def _validate_physical_calibrations(
    profile: Mapping[str, Any],
    *,
    source_path: Path,
    config_path: Path,
    config_verification_path: Path,
    config: Mapping[str, Any],
    normalized: A100DP2PhysicalProfile,
    runtime_binding_path: Path,
    model_submission_gate_path: Path,
    model_probe_path: Path,
) -> tuple[Path, Path, list[Path], list[Path]]:
    artifacts = _mapping(profile.get("source_artifacts"))
    recompute_path = _artifact_path(
        artifacts.get("recompute_audit"), label="recompute calibration"
    )
    recompute = load_object(recompute_path)
    fit = _mapping(recompute.get("fit"))
    _require(
        recompute.get("evidence_class") == "a100_dp2_recompute_calibration_gate"
        and recompute.get("status") == "PASS"
        and recompute.get("failures") == []
        and recompute.get("campaign_id") == CAMPAIGN_ID
        and recompute.get("hardware_label") == HARDWARE_LABEL
        and recompute.get("layout") == PHYSICAL_LAYOUT
        and recompute.get("calibration_ready") is True
        and math.isclose(
            _number(fit.get("recompute_ms_per_token")),
            normalized.recompute_ms_per_token,
        )
        and math.isclose(
            _number(fit.get("fixed_overhead_ms")),
            normalized.recompute_fixed_overhead_ms,
        ),
        "A100 recompute calibration is missing or inconsistent",
    )
    recompute_profile_path = _artifact_path(
        artifacts.get("recompute_profile"), label="recompute physical profile"
    )
    recompute_profile = load_object(recompute_profile_path)
    recompute_source_path = recompute_profile_path.parent / "source-manifest.json"
    _require(
        recompute_profile.get("evidence_class")
        == "recompute_latency_physical_measurement"
        and recompute_profile.get("status") == "passed"
        and recompute_profile.get("model") == MODEL
        and recompute_profile.get("model_revision") == normalized.model_revision
        and recompute_profile.get("fit") == fit,
        "A100 recompute physical measurement is invalid",
    )
    _require(
        recompute_source_path.is_file()
        and sha256_file(recompute_source_path) == sha256_file(source_path),
        "A100 recompute calibration crossed source lineage",
    )

    peer_path = _artifact_path(
        artifacts.get("registered_gpu_copy_audit"),
        label="Peer-HBM registered-copy calibration",
    )
    peer = load_object(peer_path)
    _require(
        peer.get("evidence_class") == "a100_dp2_registered_gpu_copy_gate"
        and peer.get("status") == "PASS"
        and peer.get("failures") == []
        and peer.get("campaign_id") == CAMPAIGN_ID
        and peer.get("hardware_label") == HARDWARE_LABEL
        and peer.get("design_semantics") == DESIGN_SEMANTICS
        and _integer(peer.get("payload_bytes"), 0) > 0,
        "A100 Peer-HBM registered-copy calibration did not pass",
    )
    config_peer_ref = config.get("registered_gpu_copy_gate")
    _require(
        _reference_sha(config_peer_ref, label="config registered-copy gate")
        == sha256_file(peer_path),
        "A100 config and physical profile use different registered-copy gates",
    )
    peer_upstream = _mapping(peer.get("upstream"))
    _artifact_path(
        peer_upstream.get("runtime_binding"),
        label="registered-copy runtime binding",
        expected_path=runtime_binding_path,
    )
    _artifact_path(
        peer_upstream.get("model_execution_probe"),
        label="registered-copy model probe",
        expected_path=model_probe_path,
    )
    model_audit_path = _artifact_path(
        peer_upstream.get("model_execution_audit"),
        label="registered-copy model execution audit",
    )
    model_audit = load_object(model_audit_path)
    model_details = _mapping(model_audit.get("details"))
    _require(
        model_audit.get("schema_version") == 1
        and model_audit.get("evidence_class")
        == MODEL_ATTEMPT_AUDIT_EVIDENCE_CLASS
        and model_audit.get("auditor_revision")
        == "dynamic-v46-runtime-binding"
        and model_audit.get("status") == "PASS"
        and model_audit.get("failures") == []
        and model_audit.get("campaign_id") == CAMPAIGN_ID
        and model_audit.get("hardware_label") == HARDWARE_LABEL
        and model_audit.get("design_semantics") == DESIGN_SEMANTICS
        and model_audit.get("clean_model_probe_passed") is True
        and model_audit.get("registered_copy_submission_authorized") is True
        and model_audit.get("physical_calibration_submission_authorized") is False
        and model_audit.get("mechanism_submission_authorized") is False
        and model_audit.get("paired_campaign_submission_authorized") is False
        and model_audit.get("formal_ready") is False
        and model_audit.get("paper_ready") is False,
        "A100 model execution attempt audit did not pass",
    )
    _artifact_path(
        model_audit.get("runtime_binding"),
        label="model-audit runtime binding",
        expected_path=runtime_binding_path,
    )
    _artifact_path(
        model_audit.get("model_submission_gate"),
        label="model-audit submission gate",
        expected_path=model_submission_gate_path,
    )
    _artifact_path(
        model_details.get("model_result"),
        label="model-audit model result",
        expected_path=model_probe_path,
    )
    model_telemetry = _mapping(model_details.get("gpu_telemetry"))
    recomputed_model_telemetry = _validate_declared_gpu_telemetry(
        model_telemetry,
        raw_reference=model_telemetry.get("artifact"),
        label="model-probe GPU telemetry",
        require_workload_window=False,
    )
    _validate_producer_per_gpu_telemetry(
        model_details.get("per_gpu_telemetry"),
        recomputed=recomputed_model_telemetry,
        label="model-probe",
    )

    peer_artifacts = _mapping(peer.get("artifacts"))
    _validate_declared_gpu_telemetry(
        peer.get("gpu_telemetry"),
        raw_reference=peer_artifacts.get("gpu_telemetry"),
        label="registered-copy GPU telemetry",
        require_workload_window=False,
    )

    raw_transfer = artifacts.get("transfer_audits")
    raw_generic = artifacts.get("generic_transfer_audits")
    raw_gate_logs = artifacts.get("gate_logs")
    _require(
        isinstance(raw_transfer, list)
        and len(raw_transfer) == TRANSFER_AUDIT_COUNT
        and isinstance(raw_generic, list)
        and len(raw_generic) == TRANSFER_AUDIT_COUNT
        and isinstance(raw_gate_logs, list)
        and len(raw_gate_logs) >= TRANSFER_AUDIT_COUNT,
        "A100 transfer calibration requires exactly two runs per treatment",
    )
    transfer_paths = [
        _artifact_path(value, label=f"transfer calibration {index}")
        for index, value in enumerate(raw_transfer)
    ]
    generic_paths = [
        _artifact_path(value, label=f"generic transfer calibration {index}")
        for index, value in enumerate(raw_generic)
    ]
    _require(
        len(set(transfer_paths)) == TRANSFER_AUDIT_COUNT
        and len(set(generic_paths)) == TRANSFER_AUDIT_COUNT,
        "A100 transfer calibration repeats are not independent artifacts",
    )

    expected_lineage = {
        "source_manifest": source_path,
        "config_manifest": config_path,
        "config_verification": config_verification_path,
        "runtime_binding": runtime_binding_path,
        # The frozen Mooncake transfer wire retains this historical field
        # spelling, but its referenced artifact must be the current v46 model
        # submission gate validated above.
        "model_retry_gate": model_submission_gate_path,
        "recompute_profile": recompute_profile_path,
    }
    generic_by_path: dict[Path, Mapping[str, Any]] = {}
    generic_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    run_streams_by_treatment: dict[str, set[str]] = defaultdict(set)
    records_by_treatment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    measurement_gate_paths: set[Path] = set()
    seen_run_dirs: set[Path] = set()
    seen_run_ids: set[str] = set()
    seen_stream_ids: set[str] = set()

    def validate_lineage(raw: Any, *, label: str) -> None:
        lineage = _mapping(raw)
        _require(
            set(lineage) == set(expected_lineage),
            f"{label} calibration lineage field set changed",
        )
        for name, expected_path in expected_lineage.items():
            _artifact_path(
                lineage[name],
                label=f"{label} {name}",
                expected_path=expected_path,
            )

    for path in generic_paths:
        generic = load_object(path)
        treatment = str(generic.get("treatment") or "")
        repeat_index = _integer(generic.get("calibration_repeat_index"), -1)
        run_id = str(generic.get("calibration_run_id") or "").strip()
        stream_id = str(generic.get("calibration_stream_id") or "").strip()
        measurement = _mapping(generic.get("measurement"))
        run_dir = Path(str(generic.get("run_dir") or "")).resolve()
        local_source = run_dir / "source-manifest.json"
        run_id_in_dir = bool(
            run_id
            and re.search(
                rf"(?:^|[^0-9]){re.escape(run_id)}(?:[^0-9]|$)", run_dir.name
            )
        )
        _require(
            generic.get("evidence_class") == "physical_transfer_calibration_audit"
            and generic.get("status") == "PASS"
            and generic.get("failures") == []
            and treatment in EXPECTED_TRANSFER_TREATMENTS
            and repeat_index in range(TRANSFER_REPEATS_PER_TREATMENT)
            and run_id.isdecimal()
            and int(run_id) > 0
            and run_id_in_dir
            and stream_id == _run_derived_calibration_stream_id(run_id)
            and generic.get("condition_local") is True
            and measurement.get("hardware_profile") == HARDWARE_PROFILE
            and measurement.get("calibration_repeat_index") == repeat_index
            and measurement.get("calibration_run_id") == run_id
            and measurement.get("calibration_stream_id") == stream_id
            and measurement.get("condition_local") is True
            and run_dir.is_dir()
            and local_source.is_file()
            and sha256_file(local_source) == sha256_file(source_path),
            f"generic transfer calibration is invalid: {treatment}/{repeat_index}",
        )
        validate_lineage(
            generic.get("input_lineage"), label=f"generic {treatment}/{repeat_index}"
        )
        _require(
            measurement.get("input_lineage") == generic.get("input_lineage"),
            f"generic transfer measurement crossed lineage: {treatment}/{repeat_index}",
        )
        raw_local_gate_paths = measurement.get("gate_log_paths")
        _require(
            isinstance(raw_local_gate_paths, list) and bool(raw_local_gate_paths),
            f"generic transfer calibration lacks gate logs: {treatment}/{repeat_index}",
        )
        local_gate_paths: list[Path] = []
        for raw_gate_path in raw_local_gate_paths:
            gate_path = Path(str(raw_gate_path)).resolve()
            try:
                gate_path.relative_to(run_dir)
            except ValueError as exc:
                raise A100DP2ContractError(
                    "A100 calibration gate log is not condition-local: "
                    f"{treatment}/{repeat_index}"
                ) from exc
            _require(
                gate_path.is_file(),
                f"A100 calibration gate log is missing: {treatment}/{repeat_index}",
            )
            local_gate_paths.append(gate_path)
        _require(
            len(local_gate_paths) == len(set(local_gate_paths)),
            f"duplicate run-local gate log: {treatment}/{repeat_index}",
        )
        rows = [row for gate_path in local_gate_paths for row in _read_jsonl(gate_path)]
        _validate_run_transfer_records(
            rows,
            treatment=treatment,
            run_id=run_id,
            stream_id=stream_id,
        )
        key = (treatment, repeat_index)
        _require(
            key not in generic_by_key
            and run_dir not in seen_run_dirs
            and run_id not in seen_run_ids
            and stream_id not in seen_stream_ids,
            "A100 transfer calibration repeats reused a run or stream identity",
        )
        generic_by_key[key] = generic
        generic_by_path[path] = generic
        seen_run_dirs.add(run_dir)
        seen_run_ids.add(run_id)
        seen_stream_ids.add(stream_id)
        run_streams_by_treatment[treatment].add(stream_id)
        records_by_treatment[treatment].extend(rows)
        measurement_gate_paths.update(local_gate_paths)

    expected_repeat_keys = {
        (treatment, repeat_index)
        for treatment in EXPECTED_TRANSFER_TREATMENTS
        for repeat_index in range(TRANSFER_REPEATS_PER_TREATMENT)
    }
    _require(
        set(generic_by_key) == expected_repeat_keys,
        "A100 transfer calibration does not contain exactly two repeats per treatment",
    )

    peer_signatures: set[str] = set()
    wrapper_keys: set[tuple[str, int]] = set()
    used_generic_paths: set[Path] = set()
    for path in transfer_paths:
        wrapper = load_object(path)
        treatment = str(wrapper.get("treatment") or "")
        repeat_index = _integer(wrapper.get("calibration_repeat_index"), -1)
        run_id = str(wrapper.get("calibration_run_id") or "").strip()
        stream_id = str(wrapper.get("calibration_stream_id") or "").strip()
        key = (treatment, repeat_index)
        measurement = _mapping(wrapper.get("measurement"))
        generic_path = _artifact_path(
            wrapper.get("generic_audit"),
            label=f"generic transfer audit {treatment}/{repeat_index}",
        )
        generic = generic_by_path.get(generic_path)
        _require(
            wrapper.get("evidence_class") == "a100_dp2_transfer_calibration_gate"
            and wrapper.get("status") == "PASS"
            and wrapper.get("failures") == []
            and wrapper.get("campaign_id") == CAMPAIGN_ID
            and wrapper.get("hardware_label") == HARDWARE_LABEL
            and wrapper.get("calibration_ready") is True
            and wrapper.get("condition_local") is True
            and key in generic_by_key
            and key not in wrapper_keys
            and generic is not None
            and generic_path not in used_generic_paths
            and run_id == generic.get("calibration_run_id")
            and stream_id == generic.get("calibration_stream_id")
            and measurement == generic.get("measurement")
            and wrapper.get("input_lineage") == generic.get("input_lineage"),
            f"A100 transfer calibration is invalid: {treatment}/{repeat_index}",
        )
        validate_lineage(
            wrapper.get("input_lineage"), label=f"wrapper {treatment}/{repeat_index}"
        )
        signature = _mapping(measurement.get("transport_signature"))
        peer_signature = _mapping(signature.get("peer_hbm"))
        dram_signature = _mapping(signature.get("dram"))
        _require(
            peer_signature.get("protocol") == "nvlink_intra"
            and peer_signature.get("physical_interconnect")
            == "runtime_validated_cuda_p2p"
            and dram_signature.get("protocol") == "tcp",
            f"A100 split physical transport signature is invalid: {treatment}",
        )
        wrapper_keys.add(key)
        used_generic_paths.add(generic_path)
        peer_signatures.add(json.dumps(peer_signature, sort_keys=True))
    _require(
        wrapper_keys == expected_repeat_keys
        and used_generic_paths == set(generic_paths)
        and len(peer_signatures) == 1,
        "A100 transfer treatments, repeats, or Peer-HBM engine identity changed",
    )

    gate_paths = [
        _artifact_path(value, label=f"physical gate log {index}")
        for index, value in enumerate(raw_gate_logs)
    ]
    _require(
        len(gate_paths) == len(set(gate_paths))
        and set(gate_paths) == measurement_gate_paths,
        "A100 physical gate-log set crossed a condition-local run boundary",
    )
    measurements = _mapping(profile.get("transfer_measurements"))
    _require(
        set(measurements) == set(TRANSFER_PATH_EVENTS),
        "A100 affine transfer measurement set is incomplete",
    )
    exact_fields = {
        "schema",
        "duration_formula",
        "event",
        "sample_count",
        "stream_count",
        "stream_ids",
        "warmup_sample_count",
        "steady_state_sample_count",
        "distinct_steady_state_payload_bytes",
    }
    numeric_fields = {
        "fit_r_squared",
        "per_transfer_setup_ms",
        "first_transfer_warmup_ms",
        "steady_state_bytes_per_ms",
    }
    for path_name, event in TRANSFER_PATH_EVENTS.items():
        calibration_treatment = TRANSFER_PATH_TREATMENTS[path_name]
        expected_stream_ids = run_streams_by_treatment[calibration_treatment]
        recomputed = _affine_transfer_cost_model(
            records_by_treatment[calibration_treatment],
            event,
            expected_stream_ids=expected_stream_ids,
        )
        measurement = _mapping(measurements[path_name])
        declared = _mapping(measurement.get("cost_model"))
        _require(
            measurement.get("event") == event
            and measurement.get("calibration_treatment") == calibration_treatment
            and measurement.get("calibration_repeat_count")
            == TRANSFER_REPEATS_PER_TREATMENT
            and measurement.get("calibration_stream_ids")
            == sorted(expected_stream_ids)
            and all(declared.get(field) == recomputed[field] for field in exact_fields)
            and all(
                math.isclose(
                    _number(declared.get(field)),
                    _number(recomputed[field]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for field in numeric_fields
            ),
            f"A100 affine transfer cost model is not reproducible: {path_name}",
        )
        for component in TRANSFER_COST_COMPONENTS:
            metric_name = f"{path_name}_{component}"
            _require(
                math.isclose(
                    float(getattr(normalized, metric_name)),
                    _number(recomputed[component]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"A100 affine transfer metric is not reproducible: {metric_name}",
            )
    return recompute_path, peer_path, transfer_paths, gate_paths


def _validate_physical_audit(
    path: Path,
    *,
    source_path: Path,
    source_verification_path: Path,
    config_path: Path,
    config_verification_path: Path,
    physical_profile_path: Path,
    runtime_binding_path: Path,
    model_probe_path: Path,
    registered_copy_path: Path,
    recompute_inner_path: Path,
    transfer_inner_paths: Sequence[Path],
) -> dict[str, Any]:
    audit = load_object(path)
    _require(
        audit.get("schema_version") == 1
        and audit.get("evidence_class") == PHYSICAL_PROFILE_AUDIT_EVIDENCE_CLASS
        and audit.get("builder_revision") == "v53-series-and-v51-lineage"
        and audit.get("status") == "PASS"
        and audit.get("campaign_id") == CAMPAIGN_ID
        and audit.get("hardware_label") == HARDWARE_LABEL
        and audit.get("design_semantics") == DESIGN_SEMANTICS
        and str(audit.get("calibration_node") or "")
        and audit.get("physical_profile_build_authorized") is True
        and audit.get("runtime_or_physical_result") is True
        and audit.get("mechanism_submission_authorized") is False
        and audit.get("paired_campaign_submission_authorized") is False
        and audit.get("formal_ready") is False
        and audit.get("paper_ready") is False,
        "A100 v54 physical-profile audit did not pass",
    )
    for key, expected in (
        ("source_manifest", source_path),
        ("config_manifest", config_path),
        ("config_verification", config_verification_path),
        ("physical_profile", physical_profile_path),
    ):
        _artifact_path(audit.get(key), label=f"v54 {key}", expected_path=expected)

    series_path = _artifact_path(
        audit.get("transfer_series_ready_v53"), label="v54 transfer series"
    )
    _artifact_path(
        audit.get("transfer_series_ready"),
        label="v54 compatibility transfer series",
        expected_path=series_path,
    )
    series = load_object(series_path)
    raw_attempt_refs = _mapping(series.get("transfer_attempt_audits"))
    attempt_ref_shape_valid = set(raw_attempt_refs) == EXPECTED_TRANSFER_TREATMENTS
    if attempt_ref_shape_valid:
        attempt_ref_shape_valid = all(
            isinstance(raw_attempt_refs[treatment], list)
            and len(raw_attempt_refs[treatment]) == TRANSFER_REPEATS_PER_TREATMENT
            for treatment in EXPECTED_TRANSFER_TREATMENTS
        )
    _require(
        series.get("evidence_class") == TRANSFER_SERIES_EVIDENCE_CLASS
        and series.get("status") == "PASS"
        and series.get("campaign_id") == CAMPAIGN_ID
        and series.get("hardware_label") == HARDWARE_LABEL
        and series.get("design_semantics") == DESIGN_SEMANTICS
        and series.get("physical_profile_build_authorized") is True
        and series.get("mechanism_submission_authorized") is False
        and series.get("runtime_or_physical_result") is True
        and set(series.get("treatments") or []) == EXPECTED_TRANSFER_TREATMENTS
        and series.get("repeats_per_treatment") == TRANSFER_REPEATS_PER_TREATMENT
        and attempt_ref_shape_valid,
        "A100 v53 transfer-calibration series is invalid",
    )
    transfer_attempt_paths = {
        treatment: [
            _artifact_path(
                value,
                label=f"v53 transfer attempt {treatment}/{repeat_index}",
            )
            for repeat_index, value in enumerate(raw_attempt_refs[treatment])
        ]
        for treatment in sorted(EXPECTED_TRANSFER_TREATMENTS)
    }
    flattened_attempt_paths = [
        path
        for treatment_paths in transfer_attempt_paths.values()
        for path in treatment_paths
    ]
    raw_audit_attempt_refs = audit.get("transfer_attempt_audits")
    _require(
        isinstance(raw_audit_attempt_refs, list)
        and len(raw_audit_attempt_refs) == TRANSFER_AUDIT_COUNT,
        "A100 v54 audit transfer-repeat cardinality is invalid",
    )
    audit_attempt_paths = {
        _artifact_path(value, label=f"v54 transfer attempt {index}")
        for index, value in enumerate(raw_audit_attempt_refs)
    }
    _require(
        len(set(flattened_attempt_paths)) == TRANSFER_AUDIT_COUNT
        and len(audit_attempt_paths) == TRANSFER_AUDIT_COUNT
        and audit_attempt_paths == set(flattened_attempt_paths),
        "A100 v54 audit and v53 series bind different transfer attempts",
    )

    recompute_compatibility_refs: set[Path] = set()
    recompute_v51_refs: set[Path] = set()
    nodes: set[str] = set()
    observed_inner_paths: set[Path] = set()
    attempt_run_ids: set[str] = set()
    attempt_stream_ids: set[str] = set()
    for treatment, treatment_paths in transfer_attempt_paths.items():
        for repeat_index, attempt_path in enumerate(treatment_paths):
            attempt = load_object(attempt_path)
            run_id = str(attempt.get("calibration_run_id") or "").strip()
            stream_id = str(attempt.get("calibration_stream_id") or "").strip()
            inner_path = _artifact_path(
                attempt.get("transfer_inner_audit"),
                label=f"v53 {treatment}/{repeat_index} inner calibration",
            )
            inner = load_object(inner_path)
            _require(
                attempt.get("schema_version") == 1
                and attempt.get("evidence_class")
                == TRANSFER_ATTEMPT_AUDIT_EVIDENCE_CLASS
                and attempt.get("auditor_revision")
                == "dynamic-v52-submission-v51-lineage"
                and attempt.get("status") == "PASS"
                and attempt.get("failures") == []
                and attempt.get("campaign_id") == CAMPAIGN_ID
                and attempt.get("hardware_label") == HARDWARE_LABEL
                and attempt.get("design_semantics") == DESIGN_SEMANTICS
                and attempt.get("treatment") == treatment
                and attempt.get("calibration_repeat_index") == repeat_index
                and run_id == inner.get("calibration_run_id")
                and stream_id == inner.get("calibration_stream_id")
                and stream_id == _run_derived_calibration_stream_id(run_id)
                and attempt.get("condition_local") is True
                and inner.get("condition_local") is True
                and attempt.get("transfer_calibration_complete") is True
                and attempt.get("physical_profile_build_authorized") is False
                and attempt.get("mechanism_submission_authorized") is False
                and attempt.get("paired_campaign_submission_authorized") is False
                and attempt.get("formal_ready") is False
                and attempt.get("paper_ready") is False,
                f"A100 v53 transfer attempt is invalid: {treatment}/{repeat_index}",
            )
            _require(
                run_id not in attempt_run_ids and stream_id not in attempt_stream_ids,
                "A100 v53 transfer attempts reused a physical run stream",
            )
            attempt_run_ids.add(run_id)
            attempt_stream_ids.add(stream_id)
            nodes.add(str(attempt.get("calibration_node") or ""))
            recompute_compatibility_refs.add(
                _artifact_path(
                    attempt.get("recompute_attempt_audit"),
                    label=f"v53 {treatment}/{repeat_index} compatibility recompute",
                )
            )
            recompute_v51_refs.add(
                _artifact_path(
                    attempt.get("recompute_attempt_audit_v51"),
                    label=f"v53 {treatment}/{repeat_index} v51 recompute",
                )
            )
            observed_inner_paths.add(inner_path)
    _require(
        nodes == {str(audit.get("calibration_node"))}
        and len(recompute_compatibility_refs) == 1
        and len(recompute_v51_refs) == 1
        and len(attempt_run_ids) == TRANSFER_AUDIT_COUNT
        and len(attempt_stream_ids) == TRANSFER_AUDIT_COUNT
        and observed_inner_paths == set(transfer_inner_paths),
        "A100 v53 transfer attempts crossed node/recompute/inner lineage",
    )

    recompute_compatibility_path = _artifact_path(
        audit.get("recompute_attempt_audit"),
        label="v54 compatibility recompute attempt",
    )
    series_compatibility_path = _artifact_path(
        series.get("recompute_compatibility_view_v52"),
        label="v53 recompute compatibility view",
    )
    recompute_attempt_path = _artifact_path(
        series.get("recompute_attempt_audit_v51"),
        label="v53 original recompute attempt",
    )
    _require(
        recompute_compatibility_refs
        == {recompute_compatibility_path, series_compatibility_path}
        and recompute_v51_refs == {recompute_attempt_path},
        "A100 v54/v53 transfer chain uses different recompute attempts",
    )
    compatibility = load_object(recompute_compatibility_path)
    _require(
        compatibility.get("evidence_class")
        == "a100_dp2_recompute_attempt_audit_v35"
        and compatibility.get("compatibility_view_revision")
        == "v52-for-frozen-transfer-launcher"
        and compatibility.get("status") == "PASS"
        and compatibility.get("failures") == []
        and compatibility.get("calibration_node")
        == audit.get("calibration_node")
        and compatibility.get("transfer_calibration_submission_authorized")
        is True
        and compatibility.get("mechanism_submission_authorized") is False,
        "A100 v52 recompute compatibility view is invalid",
    )
    _artifact_path(
        compatibility.get("compatibility_view_of"),
        label="v52 original recompute attempt",
        expected_path=recompute_attempt_path,
    )
    recompute_attempt = load_object(recompute_attempt_path)
    _require(
        recompute_attempt.get("schema_version") == 1
        and recompute_attempt.get("evidence_class")
        == RECOMPUTE_ATTEMPT_AUDIT_EVIDENCE_CLASS
        and recompute_attempt.get("auditor_revision")
        == "dynamic-v50-submission-and-gate"
        and recompute_attempt.get("status") == "PASS"
        and recompute_attempt.get("failures") == []
        and recompute_attempt.get("campaign_id") == CAMPAIGN_ID
        and recompute_attempt.get("hardware_label") == HARDWARE_LABEL
        and recompute_attempt.get("design_semantics") == DESIGN_SEMANTICS
        and recompute_attempt.get("calibration_node")
        == audit.get("calibration_node")
        and recompute_attempt.get("transfer_calibration_submission_authorized")
        is True
        and recompute_attempt.get("mechanism_submission_authorized") is False
        and recompute_attempt.get("paired_campaign_submission_authorized") is False
        and recompute_attempt.get("formal_ready") is False
        and recompute_attempt.get("paper_ready") is False,
        "A100 v51 recompute attempt audit is invalid",
    )
    _artifact_path(
        recompute_attempt.get("recompute_inner_audit"),
        label="v51 recompute inner audit",
        expected_path=recompute_inner_path,
    )
    physical_gate_path = _artifact_path(
        recompute_attempt.get("physical_calibration_submission_gate"),
        label="v51 physical calibration gate",
    )
    physical_gate = load_object(physical_gate_path)
    _require(
        physical_gate.get("schema_version") == 1
        and physical_gate.get("evidence_class")
        == PHYSICAL_CALIBRATION_GATE_EVIDENCE_CLASS
        and physical_gate.get("gate_revision") == "dynamic-v46-v49-lineage"
        and physical_gate.get("status") == "PASS"
        and physical_gate.get("failures") == []
        and physical_gate.get("campaign_id") == CAMPAIGN_ID
        and physical_gate.get("hardware_label") == HARDWARE_LABEL
        and physical_gate.get("design_semantics") == DESIGN_SEMANTICS
        and physical_gate.get("calibration_node") == audit.get("calibration_node")
        and physical_gate.get("recompute_calibration_submission_authorized")
        is True
        and physical_gate.get("transfer_calibration_submission_authorized")
        is False
        and physical_gate.get("mechanism_submission_authorized") is False
        and physical_gate.get("jobs_submitted") == []
        and physical_gate.get("runtime_or_physical_result") is False
        and physical_gate.get("formal_ready") is False
        and physical_gate.get("paper_ready") is False,
        "A100 v50 physical-calibration lineage gate is invalid",
    )
    for name, expected in (
        ("source_manifest", source_path),
        ("config_manifest", config_path),
        ("config_verification", config_verification_path),
        ("runtime_binding", runtime_binding_path),
        ("model_execution_probe", model_probe_path),
        ("registered_copy_inner_audit", registered_copy_path),
    ):
        _artifact_path(
            physical_gate.get(name), label=f"v50 {name}", expected_path=expected
        )
    source_activation_path = _artifact_path(
        physical_gate.get("source_activation"), label="v50 source activation"
    )
    runtime_binding = load_object(runtime_binding_path)
    _artifact_path(
        runtime_binding.get("source_activation"),
        label="v46 source activation",
        expected_path=source_activation_path,
    )
    source_activation = load_object(source_activation_path)
    _artifact_path(
        source_activation.get("source_verification"),
        label="v50 source verification",
        expected_path=source_verification_path,
    )
    return audit


def _validate_aimd_calibration(
    path: Path,
    *,
    physical_audit_path: Path,
    physical_profile_path: Path,
    source_path: Path,
    config_path: Path,
    config_verification_path: Path,
    physical_gate_paths: Sequence[Path],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    calibration = load_object(path)
    threshold = _mapping(calibration.get("threshold"))
    observation = _mapping(calibration.get("observation_contract"))
    _require(
        calibration.get("schema_version") == 1
        and calibration.get("evidence_class")
        == AIMD_STALL_CALIBRATION_EVIDENCE_CLASS
        and calibration.get("calibration_revision")
        == "v54-physical-profile-lineage"
        and calibration.get("status") == "PASS"
        and calibration.get("campaign_id") == CAMPAIGN_ID
        and calibration.get("hardware_label") == HARDWARE_LABEL
        and calibration.get("design_semantics") == DESIGN_SEMANTICS
        and calibration.get("performance_direction_used_for_selection") is False
        and calibration.get("mechanism_submission_authorized") is False
        and calibration.get("paired_campaign_submission_authorized") is False
        and calibration.get("formal_scale_submission_authorized") is False
        and calibration.get("paper_ready") is False
        and calibration.get("failures") == []
        and threshold.get("environment_variable") == "AEGIS_AIMD_LOCAL_STALL_MS"
        and threshold.get("estimator")
        == "median_successful_target_local_hbm_operation_duration_ms"
        and _integer(threshold.get("sample_count"), 0) >= 2
        and _integer(threshold.get("source_log_count"), 0) >= 2
        and _number(observation.get("aimd_interval_s")) == 5.0
        and _integer(observation.get("stable_window_threshold"), -1) == 3
        and _number(observation.get("post_workload_observation_s")) == 25.0
        and observation.get("tail_formula")
        == "interval_s * (stable_window_threshold + 2)",
        "A100 v55 physical AIMD calibration contract did not pass",
    )
    for key, expected in (
        ("physical_profile_audit", physical_audit_path),
        ("physical_profile", physical_profile_path),
        ("source_manifest", source_path),
        ("config_manifest", config_path),
        ("config_verification", config_verification_path),
    ):
        _artifact_path(
            calibration.get(key), label=f"v55 {key}", expected_path=expected
        )
    builder_path = _artifact_path(
        calibration.get("builder_and_verifier"), label="v55 builder and verifier"
    )
    _require(
        builder_path.name == "build_aimd_stall_calibration_v55.py",
        "v55 calibration was not produced by the campaign-local builder",
    )
    calibration_gate_paths = [
        _artifact_path(value, label=f"v55 gate log {index}")
        for index, value in enumerate(calibration.get("gate_logs") or [])
    ]
    _require(
        set(calibration_gate_paths) == set(physical_gate_paths),
        "v55 calibration and physical profile use different gate logs",
    )
    samples: list[float] = []
    paths_with_samples: set[Path] = set()
    operation_counts: Counter[str] = Counter()
    for gate_path in calibration_gate_paths:
        for record in _read_jsonl(gate_path):
            duration = _number(record.get("duration_ms"))
            operation = str(record.get("operation") or "")
            if (
                record.get("event") == "hbm_local_resource_operation"
                and record.get("target_local") is True
                and record.get("success") is True
                and operation in {"allocation", "reclaim"}
                and math.isfinite(duration)
                and duration > 0.0
                and _integer(record.get("key_count"), 0) > 0
                and _integer(record.get("bytes"), 0) > 0
            ):
                samples.append(duration)
                paths_with_samples.add(gate_path)
                operation_counts[operation] += 1
    _require(
        len(samples) >= 2 and len(paths_with_samples) >= 2,
        "v55 calibration lacks two independent physical HBM operation streams",
    )
    median = float(statistics.median(samples))
    _require(
        math.isclose(_number(threshold.get("local_stall_ms")), median)
        and math.isclose(_number(threshold.get("median_duration_ms")), median)
        and threshold.get("operation_counts") == dict(operation_counts),
        "v55 physical median AIMD threshold is not reproducible",
    )
    config_aimd = _mapping(_mapping(config.get("configuration")).get("aimd"))
    _require(
        _number(config_aimd.get("interval_s"))
        * (_integer(config_aimd.get("stable_window_threshold")) + 2)
        == 25.0,
        "A100 config does not independently derive the 25-second AIMD tail",
    )
    return calibration, median


def _strict_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum,
        f"{label} is not an integer >= {minimum}",
    )
    return value


def _strict_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (minimum is None or float(value) >= minimum),
        f"{label} is not a valid finite number",
    )
    return float(value)


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _aimd_quota_bytes(total_bytes: int, alpha: float) -> int:
    """Mirror Mooncake's stable floating-point quota conversion."""

    quota = int(total_bytes) * float(alpha)
    nearest = round(quota)
    tolerance = 1e-12 * max(1.0, abs(quota))
    return int(nearest if abs(quota - nearest) <= tolerance else math.floor(quota))


def _expected_aimd_runtime_config(
    config_manifest: Mapping[str, Any], *, local_stall_ms: float
) -> dict[str, Any]:
    configuration = _mapping(config_manifest.get("configuration"))
    capacity = _mapping(configuration.get("capacity"))
    aimd = _mapping(configuration.get("aimd"))
    expected = {
        "total_hbm_bytes": _integer(
            capacity.get("peer_hbm_bytes_per_target"), -1
        ),
        "interval_s": _number(aimd.get("interval_s")),
        "pressure_free_bytes": _integer(aimd.get("pressure_free_bytes"), -1),
        "local_stall_ms": local_stall_ms,
        "initial_alpha": _number(aimd.get("initial_alpha")),
        "min_alpha": _number(aimd.get("min_alpha")),
        "max_alpha": _number(aimd.get("max_alpha")),
        "decrease_factor": _number(aimd.get("decrease_factor")),
        "increase_step": _number(aimd.get("increase_step")),
        "stable_window_threshold": _integer(
            aimd.get("stable_window_threshold"), -1
        ),
    }
    _require(
        set(expected) == _AIMD_CONFIG_FIELDS
        and expected["total_hbm_bytes"] > 0
        and expected["pressure_free_bytes"] > 0
        and expected["interval_s"] > 0.0
        and expected["local_stall_ms"] > 0.0
        and 0.0
        <= expected["min_alpha"]
        <= expected["initial_alpha"]
        <= expected["max_alpha"]
        <= 1.0
        and 0.0 < expected["decrease_factor"] < 1.0
        and expected["increase_step"] > 0.0
        and expected["stable_window_threshold"] > 0,
        "A100 config cannot form a complete deployment hbm_quota_aimd",
    )
    return expected


def _validate_disabled_strict_aimd(
    value: Any, *, profile_aimd: Mapping[str, Any], pair: tuple[str, str]
) -> None:
    _require(
        _integer(profile_aimd.get("window_count"), -1) == 0
        and not _mapping(profile_aimd.get("pools")),
        f"A100 no-protection condition {pair} has AIMD profile activity",
    )
    # Mooncake's strict audit omits aegiskv_data_plane.aimd when the connector
    # is no-protection.  If a future producer emits the disabled block, accept
    # only its explicit all-zero form; never infer zero from a malformed block.
    if value is None:
        return
    aimd = _mapping(value)
    _require(
        aimd.get("enabled") is False
        and all(
            _strict_integer(
                aimd.get(name), label=f"condition {pair} disabled AIMD {name}"
            )
            == 0
            for name in _AIMD_ZERO_INTEGER_FIELDS
        )
        and _same_number(aimd.get("pressure_window_fraction"), 0.0)
        and aimd.get("final_alpha_mean") is None
        and aimd.get("observed_min_alpha") is None
        and aimd.get("observed_max_alpha") is None
        and _same_number(aimd.get("peak_quota_utilization"), 0.0)
        and aimd.get("pools") == {},
        f"A100 no-protection condition {pair} has nonzero strict AIMD activity",
    )


def _validate_strict_aimd(
    profile: Mapping[str, Any],
    strict_audit: Mapping[str, Any],
    *,
    pair: tuple[str, str],
    config_manifest: Mapping[str, Any],
    local_stall_ms: float,
) -> None:
    """Reproduce the r12 per-pool AIMD summary and controller law.

    The condition profile and strict audit are independently hashed inputs.
    Their two views of the same gate events must agree.  The strict trajectory
    intentionally contains no raw local-operation durations, so the PASS
    strict audit remains the authority for deriving local-stall pressure from
    those samples; SGLSim independently checks every emitted reason, state
    transition, quota, pool aggregate, and matching condition-profile field.
    """

    measurements = _mapping(strict_audit.get("measurements"))
    data_plane = _mapping(measurements.get("aegiskv_data_plane"))
    raw_aimd = data_plane.get("aimd") if "aimd" in data_plane else None
    profile_aimd = _event_section(profile, "aimd")
    if pair[0] == "no-protection":
        _validate_disabled_strict_aimd(
            raw_aimd, profile_aimd=profile_aimd, pair=pair
        )
        return

    _require(
        pair[0] == "aegiskv" and isinstance(raw_aimd, Mapping),
        f"A100 condition {pair} lacks strict-audit AIMD measurements",
    )
    aimd = _mapping(raw_aimd)
    expected_config = _expected_aimd_runtime_config(
        config_manifest, local_stall_ms=local_stall_ms
    )
    _require(
        aimd.get("enabled") is True
        and aimd.get("config") == expected_config,
        f"A100 condition {pair} strict AIMD config differs from frozen inputs",
    )
    pools = _mapping(aimd.get("pools"))
    profile_pools = _mapping(profile_aimd.get("pools"))
    _require(
        _strict_integer(
            aimd.get("pool_count"), label=f"condition {pair} AIMD pool_count"
        )
        == CONFIG_LAYOUT["worker_count"]
        and len(pools) == CONFIG_LAYOUT["worker_count"]
        and set(pools) == set(profile_pools)
        and "" not in pools,
        f"A100 condition {pair} AIMD pools do not match DP2 Router topology",
    )

    aggregate_ints: Counter[str] = Counter()
    final_alphas: list[float] = []
    observed_alphas: list[float] = []
    aggregate_peak_utilization = 0.0
    profile_alpha_decreases = 0
    profile_alpha_increases = 0
    profile_quota_contractions = 0
    for pool_id in sorted(pools):
        summary = _mapping(pools[pool_id])
        profile_pool = _mapping(profile_pools[pool_id])
        trajectory = summary.get("trajectory")
        timeline = profile_pool.get("timeline")
        _require(
            set(summary) == _AIMD_POOL_SUMMARY_FIELDS
            and isinstance(trajectory, list)
            and bool(trajectory)
            and isinstance(timeline, list)
            and len(timeline) == len(trajectory),
            f"A100 condition {pair} pool {pool_id} trajectory is incomplete",
        )

        previous_at = -math.inf
        previous_alpha = float(expected_config["initial_alpha"])
        previous_stable = 0
        pool_alphas: list[float] = []
        pool_cuda_free: list[int] = []
        pool_peak_utilization = 0.0
        pool_counts: Counter[str] = Counter()
        previous_profile_alpha: float | None = None
        previous_profile_quota: int | None = None
        for index, (raw_event, raw_profile_event) in enumerate(
            zip(trajectory, timeline, strict=True)
        ):
            event = _mapping(raw_event)
            profile_event = _mapping(raw_profile_event)
            label = f"condition {pair} pool {pool_id}[{index}]"
            _require(
                set(event) == _AIMD_TRAJECTORY_FIELDS,
                f"{label} strict trajectory schema changed",
            )
            observed_at = _strict_number(
                event.get("observed_at_ms"), label=f"{label} observed_at_ms"
            )
            alpha = _strict_number(event.get("alpha"), label=f"{label} alpha")
            quota_bytes = _strict_integer(
                event.get("quota_bytes"), label=f"{label} quota_bytes"
            )
            allocated_bytes = _strict_integer(
                event.get("allocated_bytes"), label=f"{label} allocated_bytes"
            )
            stable_windows = _strict_integer(
                event.get("stable_windows"), label=f"{label} stable_windows"
            )
            cuda_free_bytes = _strict_integer(
                event.get("cuda_free_bytes"), label=f"{label} cuda_free_bytes"
            )
            reclaim_count = _strict_integer(
                event.get("reclaim_count"), label=f"{label} reclaim_count"
            )
            reclaim_bytes = _strict_integer(
                event.get("reclaim_bytes"), label=f"{label} reclaim_bytes"
            )
            hbm_only_reclaim_count = _strict_integer(
                event.get("hbm_only_reclaim_count"),
                label=f"{label} hbm_only_reclaim_count",
            )
            reasons = event.get("pressure_reasons")
            pressure = event.get("pressure")
            _require(
                isinstance(pressure, bool)
                and isinstance(reasons, list)
                and reasons
                == [
                    name
                    for name in (
                        "cuda_free_bytes",
                        "peer_hbm_allocation_reclaim_stall",
                    )
                    if name in reasons
                ]
                and len(reasons) == len(set(reasons)),
                f"{label} pressure reasons are invalid",
            )
            memory_pressure = (
                cuda_free_bytes < expected_config["pressure_free_bytes"]
            )
            local_pressure = "peer_hbm_allocation_reclaim_stall" in reasons
            expected_reasons = []
            if memory_pressure:
                expected_reasons.append("cuda_free_bytes")
            if local_pressure:
                expected_reasons.append("peer_hbm_allocation_reclaim_stall")
            _require(
                observed_at > previous_at
                and reasons == expected_reasons
                and pressure is bool(memory_pressure or local_pressure),
                f"{label} pressure was not derived from target-local evidence",
            )

            expected_stable = previous_stable
            if pressure:
                expected_alpha = max(
                    expected_config["min_alpha"],
                    expected_config["decrease_factor"] * previous_alpha,
                )
                expected_stable = 0
            else:
                expected_stable += 1
                expected_alpha = previous_alpha
                if expected_stable >= expected_config["stable_window_threshold"]:
                    expected_alpha = min(
                        expected_config["max_alpha"],
                        previous_alpha + expected_config["increase_step"],
                    )
                    expected_stable = 0
            expected_quota = _aimd_quota_bytes(
                expected_config["total_hbm_bytes"], expected_alpha
            )
            _require(
                _same_number(alpha, expected_alpha)
                and stable_windows == expected_stable
                and quota_bytes == expected_quota
                and allocated_bytes <= quota_bytes
                and hbm_only_reclaim_count <= reclaim_count
                and (reclaim_count > 0 or reclaim_bytes == 0),
                f"{label} violates the target-local AIMD/quota law",
            )

            required_profile_fields = {
                "observed_at_ms",
                "pressure",
                "memory_pressure",
                "local_stall_pressure",
                "local_stall_threshold_ms",
                "pressure_reasons",
                "alpha",
                "quota_bytes",
                "allocated_bytes_before_control",
                "allocated_bytes",
                "cuda_free_bytes",
                "cuda_total_bytes",
                "stable_windows",
                "quota_satisfied",
                "reclaimed_slot_count",
                "new_hbm_admissions_paused_before_reclaim",
                "new_hbm_admissions_resumed_after_reclaim",
            }
            _require(
                required_profile_fields <= set(profile_event)
                and _same_number(profile_event.get("observed_at_ms"), observed_at)
                and profile_event.get("pressure") is pressure
                and profile_event.get("memory_pressure") is memory_pressure
                and profile_event.get("local_stall_pressure") is local_pressure
                and _same_number(
                    profile_event.get("local_stall_threshold_ms"), local_stall_ms
                )
                and profile_event.get("pressure_reasons") == reasons
                and _same_number(profile_event.get("alpha"), alpha)
                and _same_number(profile_event.get("quota_bytes"), quota_bytes)
                and _strict_number(
                    profile_event.get("allocated_bytes_before_control"),
                    label=f"{label} allocated_bytes_before_control",
                    minimum=0.0,
                )
                >= 0.0
                and _same_number(
                    profile_event.get("allocated_bytes"), allocated_bytes
                )
                and _same_number(
                    profile_event.get("cuda_free_bytes"), cuda_free_bytes
                )
                and _strict_number(
                    profile_event.get("cuda_total_bytes"),
                    label=f"{label} cuda_total_bytes",
                    minimum=max(1.0, float(cuda_free_bytes)),
                )
                >= cuda_free_bytes
                and _integer(profile_event.get("stable_windows"), -1)
                == stable_windows
                and profile_event.get("quota_satisfied") is True
                and _integer(profile_event.get("reclaimed_slot_count"), -1)
                == reclaim_count
                and profile_event.get(
                    "new_hbm_admissions_paused_before_reclaim"
                )
                is pressure
                and profile_event.get(
                    "new_hbm_admissions_resumed_after_reclaim"
                )
                is pressure,
                f"{label} strict/profile AIMD views disagree",
            )

            if previous_profile_alpha is not None:
                profile_alpha_decreases += int(alpha < previous_profile_alpha)
                profile_alpha_increases += int(alpha > previous_profile_alpha)
            if previous_profile_quota is not None:
                profile_quota_contractions += int(quota_bytes < previous_profile_quota)
            previous_profile_alpha = alpha
            previous_profile_quota = quota_bytes
            previous_at = observed_at
            pool_counts["pressure_window_count"] += int(pressure)
            pool_counts["memory_pressure_window_count"] += int(memory_pressure)
            pool_counts["local_stall_pressure_window_count"] += int(local_pressure)
            pool_counts["alpha_transition_count"] += int(
                not _same_number(alpha, previous_alpha)
            )
            pool_counts["reclaim_count"] += reclaim_count
            pool_counts["reclaim_bytes"] += reclaim_bytes
            pool_counts["hbm_only_reclaim_count"] += hbm_only_reclaim_count
            pool_alphas.append(alpha)
            pool_cuda_free.append(cuda_free_bytes)
            utilization = allocated_bytes / quota_bytes if quota_bytes else 0.0
            pool_peak_utilization = max(pool_peak_utilization, utilization)
            previous_alpha = alpha
            previous_stable = stable_windows

        window_count = len(trajectory)
        pool_counts["window_count"] = window_count
        pool_counts["stable_window_count"] = (
            window_count - pool_counts["pressure_window_count"]
        )
        integer_summary_fields = {
            "window_count",
            "pressure_window_count",
            "stable_window_count",
            "memory_pressure_window_count",
            "local_stall_pressure_window_count",
            "alpha_transition_count",
            "reclaim_count",
            "reclaim_bytes",
            "hbm_only_reclaim_count",
        }
        _require(
            all(
                _strict_integer(
                    summary.get(name),
                    label=f"condition {pair} pool {pool_id} {name}",
                )
                == pool_counts[name]
                for name in integer_summary_fields
            )
            and _same_number(
                summary.get("initial_alpha"), expected_config["initial_alpha"]
            )
            and _same_number(summary.get("final_alpha"), pool_alphas[-1])
            and _same_number(summary.get("observed_min_alpha"), min(pool_alphas))
            and _same_number(summary.get("observed_max_alpha"), max(pool_alphas))
            and _strict_integer(
                summary.get("min_cuda_free_bytes"),
                label=f"condition {pair} pool {pool_id} min_cuda_free_bytes",
            )
            == min(pool_cuda_free)
            and _same_number(
                summary.get("peak_quota_utilization"), pool_peak_utilization
            ),
            f"A100 condition {pair} pool {pool_id} strict AIMD summary drifted",
        )
        _require(
            _integer(profile_pool.get("window_count"), -1) == window_count
            and _same_number(profile_pool.get("initial_alpha"), pool_alphas[0])
            and _same_number(profile_pool.get("final_alpha"), pool_alphas[-1])
            and _same_number(profile_pool.get("minimum_alpha"), min(pool_alphas))
            and _same_number(profile_pool.get("maximum_alpha"), max(pool_alphas))
            and profile_pool.get("local_stall_threshold_ms_values")
            == [local_stall_ms],
            f"A100 condition {pair} pool {pool_id} profile summary drifted",
        )
        for name in integer_summary_fields:
            aggregate_ints[name] += pool_counts[name]
        final_alphas.append(pool_alphas[-1])
        observed_alphas.extend(pool_alphas)
        aggregate_peak_utilization = max(
            aggregate_peak_utilization, pool_peak_utilization
        )

    aggregate_expectations = {
        "pool_count": len(pools),
        **{name: aggregate_ints[name] for name in _AIMD_ZERO_INTEGER_FIELDS},
    }
    _require(
        all(
            _strict_integer(
                aimd.get(name), label=f"condition {pair} AIMD {name}"
            )
            == expected
            for name, expected in aggregate_expectations.items()
        )
        and _same_number(
            aimd.get("pressure_window_fraction"),
            aggregate_ints["pressure_window_count"]
            / aggregate_ints["window_count"],
        )
        and _same_number(
            aimd.get("final_alpha_mean"), sum(final_alphas) / len(final_alphas)
        )
        and _same_number(aimd.get("observed_min_alpha"), min(observed_alphas))
        and _same_number(aimd.get("observed_max_alpha"), max(observed_alphas))
        and _same_number(
            aimd.get("peak_quota_utilization"), aggregate_peak_utilization
        ),
        f"A100 condition {pair} aggregate strict AIMD summary drifted",
    )
    reclaim_samples = profile_aimd.get("reclaim_samples")
    _require(
        _integer(profile_aimd.get("window_count"), -1)
        == aggregate_ints["window_count"]
        and _integer(
            profile_aimd.get(
                "target_local_allocation_or_reclaim_stall_window_count"
            ),
            -1,
        )
        == aggregate_ints["local_stall_pressure_window_count"]
        and _integer(profile_aimd.get("malformed_window_count"), -1) == 0
        and _integer(profile_aimd.get("alpha_decrease_count"), -1)
        == profile_alpha_decreases
        and _integer(profile_aimd.get("alpha_increase_count"), -1)
        == profile_alpha_increases
        and _integer(profile_aimd.get("quota_contraction_count"), -1)
        == profile_quota_contractions
        and _integer(profile_aimd.get("quota_unsatisfied_window_count"), -1) == 0
        and isinstance(reclaim_samples, list)
        and len(reclaim_samples) == aggregate_ints["reclaim_count"]
        and sum(
            _mapping(sample).get("residency") == "HBM_ONLY"
            for sample in reclaim_samples
        )
        == aggregate_ints["hbm_only_reclaim_count"],
        f"A100 condition {pair} profile/strict AIMD population differs",
    )


def _event_section(profile: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(_mapping(profile.get("workload_window_events")).get(name))


def _source_path(profile: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(_event_section(profile, "source_owned_paths").get(name))


def _positive_physical_path(
    profile: Mapping[str, Any],
    name: str,
    *,
    owner: str,
    protocol: str,
    interconnect: str,
) -> bool:
    path = _source_path(profile, name)
    protocols = {
        str(key)
        for key, value in _mapping(path.get("protocol_counts")).items()
        if _integer(value) > 0
    }
    interconnects = {
        str(key)
        for key, value in _mapping(path.get("physical_interconnect_counts")).items()
        if _integer(value) > 0
    }
    return (
        _integer(path.get("count")) > 0
        and _integer(path.get("bytes")) > 0
        and _number(_mapping(path.get("duration_ms")).get("minimum"), 0.0) > 0.0
        and path.get("physical") is True
        and path.get("initiated_by_owner") is True
        and path.get("owner") == owner
        and protocols == {protocol}
        and interconnects == {interconnect}
    )


def _zero_physical_path(profile: Mapping[str, Any], name: str) -> bool:
    path = _source_path(profile, name)
    return _integer(path.get("count")) == 0 and _integer(path.get("bytes")) == 0


def _aggregate_counts(
    profiles: Iterable[Mapping[str, Any]], section: str
) -> Counter[str]:
    result: Counter[str] = Counter()
    for profile in profiles:
        for name, value in _event_section(profile, section).items():
            result[str(name)] += _integer(value)
    return result


def _validate_profile_artifacts(profile: Mapping[str, Any]) -> None:
    artifacts = _mapping(profile.get("artifacts"))
    _require(bool(artifacts), "condition profile artifact set is absent")
    for name, raw in artifacts.items():
        if isinstance(raw, list):
            for index, value in enumerate(raw):
                _artifact_path(value, label=f"condition {name}[{index}]")
        elif isinstance(raw, Mapping):
            _artifact_path(raw, label=f"condition {name}")


def _validate_condition_profiles(
    gate: Mapping[str, Any],
    *,
    config_manifest: Mapping[str, Any],
    source_sha: str,
    config_sha: str,
    config_verification_sha: str,
    physical_profile_sha: str,
    aimd_calibration_sha: str,
    aimd_threshold: float,
) -> tuple[
    list[dict[str, Any]],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    raw_refs = gate.get("condition_profiles")
    _require(
        isinstance(raw_refs, list) and len(raw_refs) == 4,
        "A100 mechanism gate must bind exactly four condition profiles",
    )
    profile_artifacts = tuple(
        artifact_record(
            _artifact_path(value, label=f"condition profile {index}")
        )
        for index, value in enumerate(raw_refs)
    )
    profiles = [load_object(Path(record["path"])) for record in profile_artifacts]
    pairs = [
        (str(value.get("treatment") or ""), str(value.get("failure_method") or ""))
        for value in profiles
    ]
    _require(
        len(set(pairs)) == 4 and set(pairs) == EXPECTED_CONDITIONS,
        "A100 mechanism profiles do not cover the exact four conditions",
    )
    strict_artifacts: list[Mapping[str, Any]] = []
    lineage_identities: set[tuple[str, ...]] = set()
    strict_hashes: set[str] = set()
    for profile, pair in zip(profiles, pairs, strict=True):
        _require(
            profile.get("schema_version") == 1
            and profile.get("evidence_class") == CONDITION_PROFILE_EVIDENCE_CLASS
            and profile.get("status") == "PASS"
            and profile.get("failures") == []
            and profile.get("campaign_id") == CAMPAIGN_ID
            and str(profile.get("condition_id") or "")
            and profile.get("design_semantics") == DESIGN_SEMANTICS
            and profile.get("hardware_label") == HARDWARE_LABEL
            and profile.get("layout") == CONFIG_LAYOUT
            and profile.get("runtime_admission_decision_wire_values") == WIRE_VALUES,
            f"A100 condition profile identity is invalid: {pair}",
        )
        _validate_profile_artifacts(profile)
        lineage = _mapping(profile.get("input_lineage"))
        lineage_hashes: list[str] = []
        expected_hashes = {
            "source_manifest": source_sha,
            "config_manifest": config_sha,
            "config_verification": config_verification_sha,
            "physical_profile": physical_profile_sha,
            "aimd_stall_calibration": aimd_calibration_sha,
        }
        for name, expected_sha in expected_hashes.items():
            observed_sha = _reference_sha(
                lineage.get(name), label=f"condition {pair} {name}"
            )
            _require(
                observed_sha == expected_sha,
                f"A100 condition {pair} has crossed {name} lineage",
            )
            lineage_hashes.append(observed_sha)
        strict_path = _artifact_path(
            lineage.get("strict_run_audit"),
            label=f"condition {pair} strict run audit",
        )
        strict = load_object(strict_path)
        _require(
            strict.get("evidence_class") == "strict_audit_report"
            and strict.get("status") == "PASS"
            and strict.get("failures") == [],
            f"A100 condition {pair} strict audit did not pass",
        )
        strict_record = artifact_record(strict_path)
        strict_artifacts.append(strict_record)
        strict_hashes.add(str(strict_record["sha256"]))
        lineage_identities.add(tuple(lineage_hashes))
        _validate_strict_aimd(
            profile,
            strict,
            pair=pair,
            config_manifest=config_manifest,
            local_stall_ms=aimd_threshold,
        )

        workload = _mapping(profile.get("workload"))
        observation = _mapping(profile.get("mechanism_observation_window"))
        profile_artifact_map = _mapping(profile.get("artifacts"))
        _require(
            _reference_sha(
                profile_artifact_map.get("strict_audit"),
                label=f"condition {pair} strict artifact",
            )
            == strict_record["sha256"],
            f"A100 condition {pair} has two different strict audits",
        )
        summary_path = _artifact_path(
            profile_artifact_map.get("summary"),
            label=f"condition {pair} performance summary",
        )
        summary = load_object(summary_path)
        started = _number(workload.get("started_at_ms"))
        finished = _number(workload.get("finished_at_ms"))
        observation_started = _number(observation.get("started_at_ms"))
        observation_finished = _number(observation.get("finished_at_ms"))
        required_ms = _number(observation.get("required_duration_ms"))
        _require(
            math.isfinite(started)
            and math.isfinite(finished)
            and started < finished <= observation_started
            and observation.get("complete") is True
            and required_ms == 25000.0
            and observation_finished - observation_started >= required_ms,
            f"A100 condition {pair} counts the AIMD tail in its performance window",
        )
        _require(
            summary.get("campaign_id") == CAMPAIGN_ID
            and summary.get("campaign_condition_id") == profile.get("condition_id")
            and _number(summary.get("started_at_ms")) == started
            and _number(summary.get("finished_at_ms")) == finished
            and math.isclose(
                _number(summary.get("elapsed_ms")), finished - started
            ),
            f"A100 condition {pair} summary/performance window identity changed",
        )
        if "performance_window" in profile:
            _require(
                profile.get("performance_window") == workload,
                f"A100 condition {pair} has a second overlapping performance window",
            )
        calibration = _mapping(profile.get("aimd_local_stall_calibration"))
        _require(
            math.isclose(_number(calibration.get("local_stall_ms")), aimd_threshold)
            and calibration.get("deployed") is True
            and calibration.get("runtime_windows_match") is True,
            f"A100 condition {pair} did not apply the physical AIMD threshold",
        )
        telemetry = _mapping(profile.get("gpu_telemetry"))
        _require(
            _integer(telemetry.get("foreign_gpu_process_count"), -1) == 0
            and telemetry.get("post_run_gpu_process_cleanup") is True,
            f"A100 condition {pair} two-GPU telemetry is invalid",
        )
        _validate_declared_gpu_telemetry(
            telemetry,
            raw_reference=profile_artifact_map.get("gpu_telemetry"),
            label=f"condition {pair} GPU telemetry",
            require_workload_window=True,
        )
    _require(
        len(lineage_identities) == 1,
        "A100 four conditions do not share one "
        "source/config/profile/calibration lineage",
    )
    _require(len(strict_hashes) == 4, "A100 condition strict audits are not unique")
    return profiles, profile_artifacts, tuple(strict_artifacts)


def _validate_mechanism_semantics(profiles: Sequence[Mapping[str, Any]]) -> None:
    by_pair = {
        (str(profile["treatment"]), str(profile["failure_method"])): profile
        for profile in profiles
    }
    baseline = [
        by_pair[("no-protection", failure)]
        for failure in ("none", "sigkill_process_group")
    ]
    for profile in baseline:
        _require(
            not any(
                _integer(value) > 0
                for value in _event_section(profile, "admission_decisions").values()
            )
            and not any(
                _integer(value) > 0
                for value in _event_section(
                    profile, "independent_tier_outcomes"
                ).values()
            )
            and _integer(_event_section(profile, "aimd").get("window_count")) == 0
            and all(
                _zero_physical_path(profile, name)
                for name in (
                    "peer_hbm_protection",
                    "direct_dram_protection",
                    "peer_hbm_recovery",
                    "dram_recovery",
                )
            ),
            "A100 no-protection condition contains AegisKV mechanism activity",
        )

    aegis = [
        by_pair[("aegiskv", failure)]
        for failure in ("none", "sigkill_process_group")
    ]
    decisions = _aggregate_counts(aegis, "admission_decisions")
    outcomes = _aggregate_counts(aegis, "independent_tier_outcomes")
    _require(
        decisions["ADMIT_HBM"] > 0
        and decisions["ADMIT_DRAM"] > 0
        and decisions["BYPASS"] > 0
        and decisions["INVALID"] == 0
        and decisions["UNEXPECTED_TIER_DECISION"] == 0
        and decisions["HBM_REJECT_THEN_ADMIT_DRAM"] > 0,
        "A100 independent admission/wire outcomes were not fully exercised",
    )
    _require(
        all(
            outcomes[name] > 0
            for name in ("HBM_ONLY", "DRAM_ONLY", "HBM_AND_DRAM", "BYPASS")
        ),
        "A100 did not exercise all four independent residency outcomes",
    )
    for profile in aegis:
        pair = (profile["treatment"], profile["failure_method"])
        _require(
            _positive_physical_path(
                profile,
                "peer_hbm_protection",
                owner="source_worker_d2d_p2p",
                protocol="nvlink_intra",
                interconnect="runtime_validated_cuda_p2p",
            )
            and _positive_physical_path(
                profile,
                "direct_dram_protection",
                owner="source_worker_d2h",
                protocol="tcp",
                interconnect="host_dram_store",
            ),
            f"A100 condition {pair} lacks physical source-owned HBM D2D/DRAM D2H",
        )
        confirmations = _event_section(profile, "path_confirmations")
        _require(
            _integer(confirmations.get("hbm_independent_ack_count")) > 0
            and _integer(confirmations.get("dram_independent_ack_count")) > 0
            and _integer(confirmations.get("hbm_and_dram_independent_ack_count")) > 0,
            f"A100 condition {pair} lacks independent HBM/DRAM ACKs",
        )
        forbidden = _event_section(profile, "forbidden_semantics")
        _require(
            all(
                _integer(forbidden.get(name)) == 0
                for name in (
                    "target_owned_task_count",
                    "hbm_to_dram_transfer_count",
                    "background_destage_count",
                    "cross_tier_transfer_count",
                )
            ),
            f"A100 condition {pair} used target-owned or HBM-to-DRAM semantics",
        )

    healthy = by_pair[("aegiskv", "none")]
    failed = by_pair[("aegiskv", "sigkill_process_group")]
    _require(
        _zero_physical_path(healthy, "peer_hbm_recovery")
        and _zero_physical_path(healthy, "dram_recovery"),
        "healthy A100 AegisKV condition contains recovery loads",
    )
    _require(
        _positive_physical_path(
            failed,
            "peer_hbm_recovery",
            owner="survivor_worker_hbm_load",
            protocol="nvlink_intra",
            interconnect="runtime_validated_cuda_p2p",
        )
        and _positive_physical_path(
            failed,
            "dram_recovery",
            owner="survivor_worker_store_load",
            protocol="tcp",
            interconnect="host_dram_store",
        ),
        "failed A100 condition lacks both physical recovery paths",
    )

    alpha_decreases = 0
    alpha_increases = 0
    quota_contractions = 0
    local_stalls = 0
    valid_reclaim = False
    for profile in aegis:
        aimd = _event_section(profile, "aimd")
        alpha_decreases += _integer(aimd.get("alpha_decrease_count"))
        alpha_increases += _integer(aimd.get("alpha_increase_count"))
        quota_contractions += _integer(aimd.get("quota_contraction_count"))
        local_stalls += _integer(
            aimd.get("target_local_allocation_or_reclaim_stall_window_count")
        )
        _require(
            aimd.get("new_hbm_admissions_paused_before_reclaim") is True
            and aimd.get("new_hbm_admissions_resumed_after_reclaim") is True
            and _integer(aimd.get("quota_unsatisfied_window_count")) == 0,
            "A100 AIMD pause/reclaim/satisfy/resume contract did not pass",
        )
        for raw in aimd.get("reclaim_samples") or []:
            sample = _mapping(raw)
            freed = sample.get("freed")
            if (
                sample.get("residency") == "HBM_ONLY"
                and sample.get("controller_selected") is True
                and sample.get("dram_backing_required") is False
                and sample.get("physical_copy_deleted") is True
                and isinstance(freed, list)
                and bool(freed)
                and all(value is True or _integer(value, -1) >= 0 for value in freed)
            ):
                valid_reclaim = True
    _require(
        local_stalls > 0 and alpha_decreases > 0 and quota_contractions > 0,
        "A100 AIMD did not contract alpha and quota on physical local stalls",
    )
    _require(alpha_increases > 0, "A100 AIMD did not additively increase alpha")
    _require(
        valid_reclaim,
        "A100 controller did not directly reclaim an HBM_ONLY tail without DRAM",
    )


def _validate_mechanism_gate(
    path: Path,
    *,
    config_manifest: Mapping[str, Any],
    source_sha: str,
    config_sha: str,
    config_verification_sha: str,
    physical_profile_sha: str,
    aimd_calibration_sha: str,
    aimd_threshold: float,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    gate = load_object(path)
    _require(
        gate.get("schema_version") == 1
        and gate.get("evidence_class") == MECHANISM_GATE_EVIDENCE_CLASS
        and gate.get("status") == "PASS"
        and gate.get("failures") == []
        and gate.get("campaign_id") == CAMPAIGN_ID
        and gate.get("hardware_label") == HARDWARE_LABEL
        and gate.get("design_semantics") == DESIGN_SEMANTICS
        and gate.get("pilot_performance_claim_authorized") is False
        and gate.get("formal_scale_submission_authorized") is False
        and gate.get("paper_ready") is False,
        "A100 four-condition mechanism gate did not pass",
    )
    for name, expected_sha in (
        ("source_manifest", source_sha),
        ("config_manifest", config_sha),
        ("config_verification", config_verification_sha),
        ("physical_profile", physical_profile_sha),
        ("aimd_stall_calibration", aimd_calibration_sha),
    ):
        _require(
            _reference_sha(gate.get(name), label=f"mechanism gate {name}")
            == expected_sha,
            f"A100 mechanism gate crossed {name} lineage",
        )
    checks = gate.get("checks")
    _require(isinstance(checks, list), "A100 mechanism gate checks are absent")
    check_map: dict[str, bool] = {}
    for raw in checks:
        check = _mapping(raw)
        name = str(check.get("name") or "")
        _require(name and name not in check_map, "duplicate A100 mechanism check")
        check_map[name] = check.get("passed") is True
    _require(
        A100_REQUIRED_MECHANISM_CHECKS <= set(check_map)
        and all(check_map[name] for name in A100_REQUIRED_MECHANISM_CHECKS),
        "A100 mechanism gate lacks a required passing check",
    )
    profiles, profile_artifacts, strict_artifacts = _validate_condition_profiles(
        gate,
        config_manifest=config_manifest,
        source_sha=source_sha,
        config_sha=config_sha,
        config_verification_sha=config_verification_sha,
        physical_profile_sha=physical_profile_sha,
        aimd_calibration_sha=aimd_calibration_sha,
        aimd_threshold=aimd_threshold,
    )
    _validate_mechanism_semantics(profiles)
    return profile_artifacts, strict_artifacts


def validate_a100_dp2_bundle(paths: A100DP2InputPaths) -> A100DP2PreflightResult:
    """Validate all A100 artifacts without creating output or loading a trace."""

    inputs = paths.resolved()
    for name in inputs.__dataclass_fields__:
        path = getattr(inputs, name)
        _require(
            path.is_file(),
            f"required A100 producer artifact is missing: {name}={path}",
        )

    physical_raw = load_object(inputs.physical_profile)
    source_artifacts = _mapping(physical_raw.get("source_artifacts"))
    source_verification_path = _artifact_path(
        source_artifacts.get("source_verification"),
        label="physical-profile source verification",
    )
    _source, entries_sha = _validate_source_manifest(
        inputs.source_manifest, source_verification_path
    )
    runtime_binding, runtime_identity = _validate_runtime_binding(
        inputs.runtime_binding,
        source_manifest_path=inputs.source_manifest,
        source_verification_path=source_verification_path,
    )
    _validate_model_submission_gate(
        inputs.model_submission_gate,
        runtime_binding_path=inputs.runtime_binding,
        runtime_binding=runtime_binding,
    )
    config, revision, model_probe_path = _validate_config(
        inputs.config_manifest,
        inputs.config_verification,
        inputs.source_manifest,
        source_verification_path,
        inputs.runtime_binding,
        runtime_identity,
    )
    profile_raw, normalized = _validate_physical_profile(
        inputs.physical_profile,
        source_path=inputs.source_manifest,
        config_path=inputs.config_manifest,
        config_verification_path=inputs.config_verification,
        model_revision=revision,
    )
    recompute_path, registered_copy_path, transfer_inner_paths, gate_paths = (
        _validate_physical_calibrations(
            profile_raw,
            source_path=inputs.source_manifest,
            config_path=inputs.config_manifest,
            config_verification_path=inputs.config_verification,
            config=config,
            normalized=normalized,
            runtime_binding_path=inputs.runtime_binding,
            model_submission_gate_path=inputs.model_submission_gate,
            model_probe_path=model_probe_path,
        )
    )
    _validate_physical_audit(
        inputs.physical_profile_audit,
        source_path=inputs.source_manifest,
        source_verification_path=source_verification_path,
        config_path=inputs.config_manifest,
        config_verification_path=inputs.config_verification,
        physical_profile_path=inputs.physical_profile,
        runtime_binding_path=inputs.runtime_binding,
        model_probe_path=model_probe_path,
        registered_copy_path=registered_copy_path,
        recompute_inner_path=recompute_path,
        transfer_inner_paths=transfer_inner_paths,
    )
    _aimd, aimd_threshold = _validate_aimd_calibration(
        inputs.aimd_stall_calibration,
        physical_audit_path=inputs.physical_profile_audit,
        physical_profile_path=inputs.physical_profile,
        source_path=inputs.source_manifest,
        config_path=inputs.config_manifest,
        config_verification_path=inputs.config_verification,
        physical_gate_paths=gate_paths,
        config=config,
    )
    source_sha = sha256_file(inputs.source_manifest)
    config_sha = sha256_file(inputs.config_manifest)
    config_verification_sha = sha256_file(inputs.config_verification)
    physical_profile_sha = sha256_file(inputs.physical_profile)
    aimd_sha = sha256_file(inputs.aimd_stall_calibration)
    profile_artifacts, strict_artifacts = _validate_mechanism_gate(
        inputs.mechanism_gate,
        config_manifest=config,
        source_sha=source_sha,
        config_sha=config_sha,
        config_verification_sha=config_verification_sha,
        physical_profile_sha=physical_profile_sha,
        aimd_calibration_sha=aimd_sha,
        aimd_threshold=aimd_threshold,
    )
    input_artifacts = {
        name: artifact_record(getattr(inputs, name))
        for name in inputs.__dataclass_fields__
    }
    lineage_identity = canonical_sha256(
        {
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "wire_values": WIRE_VALUES,
            "input_sha256": {
                name: value["sha256"] for name, value in input_artifacts.items()
            },
            "condition_profile_sha256": [
                value["sha256"] for value in profile_artifacts
            ],
        }
    )
    return A100DP2PreflightResult(
        inputs=inputs,
        profile=normalized,
        input_artifacts=input_artifacts,
        source_entries_sha256=entries_sha,
        condition_profile_artifacts=profile_artifacts,
        strict_audit_artifacts=strict_artifacts,
        aimd_local_stall_ms=aimd_threshold,
        aimd_runtime_config=_expected_aimd_runtime_config(
            config, local_stall_ms=aimd_threshold
        ),
        lineage_identity_sha256=lineage_identity,
    )
