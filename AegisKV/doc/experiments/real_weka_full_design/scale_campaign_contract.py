#!/usr/bin/env python3
"""Fail-closed, multi-lineage planning for calibrated AegisKV scale studies.

This module composes independently validated single-lineage calibrated-suite
bindings.  It never infers a model or hardware list, never reuses a physical
profile across model revisions or layouts, and never loads traces, fits a
model, runs a replay, or emits a performance result.  Missing physical
lineages are returned as machine-readable readiness gaps.  Complete inputs
produce only immutable ``PLANNED_NOT_RUN`` cells.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    DESIGN_SEMANTICS,
    artifact_record,
    canonical_sha256,
    load_object,
)
from AegisKV.experiments.real_weka_full_design.calibrated_suite_contract import (
    SUITE_BINDING_EVIDENCE_CLASS,
    CalibratedSuiteContractError,
    preflight_suite_runner,
    validate_suite_binding,
)


REQUIREMENTS_EVIDENCE_CLASS = (
    "sglsim_aegiskv_scale_campaign_requirements_v1"
)
READINESS_EVIDENCE_CLASS = "sglsim_aegiskv_scale_campaign_readiness_v1"
CAMPAIGN_BINDING_EVIDENCE_CLASS = (
    "sglsim_aegiskv_multi_lineage_scale_campaign_binding_v1"
)
CAMPAIGN_RUNNER_PREFLIGHT_EVIDENCE_CLASS = (
    "sglsim_aegiskv_multi_lineage_scale_campaign_runner_preflight_v1"
)
CAMPAIGN_BINDING_STATUS = "BOUND_STATIC_ONLY"
MAX_SCALE_CAMPAIGN_REQUIREMENTS = 64
MAX_SCALE_CAMPAIGN_CELLS = 1_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")

_MANIFEST_FIELDS = {
    "schema_version",
    "evidence_class",
    "campaign_label",
    "limits",
    "requirements",
}
_LIMIT_FIELDS = {"max_requirements", "max_total_cells"}
_REQUIREMENT_FIELDS = {
    "requirement_id",
    "consumer_kind",
    "hardware_label",
    "design_semantics",
    "model",
    "layout",
    "physical_profile",
    "calibration",
    "lineage_identity_sha256",
    "workload_axes",
    "paper_axes",
    "suite_binding",
}
_LEGACY_REQUIREMENT_FIELDS = _REQUIREMENT_FIELDS - {"paper_axes"}
_MODEL_FIELDS = {"name", "scale_label", "revision", "config_sha256", "dtype"}
_LEGACY_MODEL_FIELDS = _MODEL_FIELDS - {"config_sha256"}
_LAYOUT_FIELDS = {"tp_size", "pp_size", "system_dp_size", "worker_count"}
_PROFILE_FIELDS = {"schema_version", "schema_class", "semantics", "sha256"}
_CALIBRATION_FIELDS = {"identity_sha256", "mechanism_gate_sha256"}
_WORKLOAD_AXIS_FIELDS = {
    "concurrency_levels",
    "failure_rates",
    "capacity_points",
    "seeds",
}
_PAPER_AXIS_FIELDS = {
    "context_cap_tokens",
    "token_projection_grid",
    "workload_shapes",
    "phase_matrix",
}
_TOKEN_PROJECTION_FIELDS = {"scale", "role"}
_WORKLOAD_SHAPE_FIELDS = {"workflow_count", "concurrency"}
_PHASE_FIELDS = {
    "phase",
    "policy_count",
    "seed_count",
    "shape_count",
    "run_count",
}
_CAPACITY_FIELDS = {
    "label",
    "peer_hbm_fraction",
    "store_dram_fraction",
}
_ARTIFACT_FIELDS = {"path", "sha256", "size_bytes"}

SCALE_CAMPAIGN_STATIC_AUTHORIZATION = {
    "campaign_binding_materialized": True,
    "static_runner_preflight_authorized": True,
    "runnable_cells_authorized": False,
    "trace_loading_authorized": False,
    "model_fitting_authorized": False,
    "replay_authorized": False,
    "result_output_authorized": False,
    "performance_result_authorized": False,
    "performance_claim_authorized": False,
    "formal_ready": False,
    "paper_ready": False,
}

SCALE_CAMPAIGN_BLOCKED_AUTHORIZATION = {
    "campaign_binding_materialization_authorized": False,
    "static_runner_preflight_authorized": False,
    "runnable_cells_authorized": False,
    "trace_loading_authorized": False,
    "model_fitting_authorized": False,
    "replay_authorized": False,
    "result_output_authorized": False,
    "performance_result_authorized": False,
    "performance_claim_authorized": False,
    "formal_ready": False,
    "paper_ready": False,
}


class ScaleCampaignContractError(ValueError):
    """Raised when a scale-campaign declaration fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScaleCampaignContractError(message)


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return load_object(path)
    except (OSError, ValueError) as exc:
        raise ScaleCampaignContractError(
            f"cannot load {label} JSON object {path}: {exc}"
        ) from exc


def _strict_positive_int(value: Any, *, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return int(value)


def _strict_string(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and bool(value) and len(value) <= 512,
        f"{label} must be a nonempty bounded string",
    )
    return value


def _strict_sha(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be an exact SHA-256",
    )
    return value


def _optional_string(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _strict_string(value, label=label)


def _optional_sha(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _strict_sha(value, label=label)


def _strict_int_axis(value: Any, *, label: str, minimum: int) -> list[int]:
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


def _strict_rate_axis(value: Any) -> list[float]:
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


def _normalize_artifact_reference(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} artifact reference is invalid")
    _require(set(value) == _ARTIFACT_FIELDS, f"{label} artifact fields are not exact")
    path_text = _strict_string(value.get("path"), label=f"{label} path")
    sha = _strict_sha(value.get("sha256"), label=f"{label} sha256")
    size = value.get("size_bytes")
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label} size_bytes is invalid",
    )
    return {
        "path": str(Path(path_text).resolve()),
        "sha256": sha,
        "size_bytes": int(size),
    }


def _verify_artifact_reference(value: Mapping[str, Any], *, label: str) -> Path:
    reference = _normalize_artifact_reference(value, label=label)
    path = Path(reference["path"])
    _require(path.is_file(), f"{label} artifact is missing: {path}")
    observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(observed_sha == reference["sha256"], f"{label} artifact hash mismatch")
    _require(
        path.stat().st_size == reference["size_bytes"],
        f"{label} artifact size mismatch",
    )
    return path


def _normalize_workload_axes(
    value: Any,
) -> tuple[dict[str, Any], int | None]:
    _require(
        isinstance(value, Mapping) and set(value) == _WORKLOAD_AXIS_FIELDS,
        "workload_axes field set is not exact",
    )
    concurrency = (
        None
        if value.get("concurrency_levels") is None
        else _strict_int_axis(
            value.get("concurrency_levels"),
            label="concurrency_levels",
            minimum=1,
        )
    )
    failure_rates = (
        None
        if value.get("failure_rates") is None
        else _strict_rate_axis(value.get("failure_rates"))
    )
    seeds = (
        None
        if value.get("seeds") is None
        else _strict_int_axis(value.get("seeds"), label="seeds", minimum=0)
    )
    raw_capacities = value.get("capacity_points")
    capacities: list[dict[str, Any]] | None = None
    if raw_capacities is not None:
        _require(
            isinstance(raw_capacities, list) and bool(raw_capacities),
            "capacity_points must be nonempty",
        )
        capacities = []
        seen_labels: set[str] = set()
        for raw in raw_capacities:
            _require(
                isinstance(raw, Mapping) and set(raw) == _CAPACITY_FIELDS,
                "capacity point field set is not exact",
            )
            label = str(raw.get("label") or "")
            _require(
                _IDENTIFIER_RE.fullmatch(label) is not None
                and label not in seen_labels,
                "capacity point label is invalid or duplicated",
            )
            seen_labels.add(label)
            point: dict[str, Any] = {"label": label}
            for field in ("peer_hbm_fraction", "store_dram_fraction"):
                raw_fraction = raw.get(field)
                _require(
                    isinstance(raw_fraction, (int, float))
                    and not isinstance(raw_fraction, bool)
                    and math.isfinite(float(raw_fraction))
                    and 0.0 <= float(raw_fraction) <= 1.0,
                    f"capacity point {label} {field} is outside [0, 1]",
                )
                point[field] = float(raw_fraction)
            capacities.append(point)
        _require(
            [point["label"] for point in capacities]
            == sorted(point["label"] for point in capacities),
            "capacity_points must be sorted by label",
        )
    complete = all(
        axis is not None
        for axis in (concurrency, failure_rates, capacities, seeds)
    )
    cell_count = None
    if complete:
        cell_count = (
            len(concurrency)
            * len(failure_rates)
            * len(capacities)
            * len(seeds)
        )
    return (
        {
            "concurrency_levels": concurrency,
            "failure_rates": failure_rates,
            "capacity_points": capacities,
            "seeds": seeds,
        },
        cell_count,
    )


def _normalize_paper_axes(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    _require(
        isinstance(value, Mapping) and set(value) == _PAPER_AXIS_FIELDS,
        "paper_axes field set is not exact",
    )
    context_cap = _strict_positive_int(
        value.get("context_cap_tokens"), label="paper context_cap_tokens"
    )
    raw_grid = value.get("token_projection_grid")
    _require(
        isinstance(raw_grid, list) and bool(raw_grid),
        "paper token_projection_grid must be nonempty",
    )
    grid: list[dict[str, Any]] = []
    for raw in raw_grid:
        _require(
            isinstance(raw, Mapping) and set(raw) == _TOKEN_PROJECTION_FIELDS,
            "paper token projection field set is not exact",
        )
        scale = raw.get("scale")
        _require(
            isinstance(scale, (int, float))
            and not isinstance(scale, bool)
            and math.isfinite(float(scale))
            and 0.0 < float(scale) <= 1.0,
            "paper token projection scale is invalid",
        )
        role = _strict_string(raw.get("role"), label="paper projection role")
        _require(
            _IDENTIFIER_RE.fullmatch(role) is not None,
            "paper projection role is invalid",
        )
        grid.append({"scale": float(scale), "role": role})
    _require(
        [point["scale"] for point in grid]
        == sorted({point["scale"] for point in grid}),
        "paper token projection scales must be unique and sorted",
    )

    raw_shapes = value.get("workload_shapes")
    _require(
        isinstance(raw_shapes, list) and bool(raw_shapes),
        "paper workload_shapes must be nonempty",
    )
    shapes: list[dict[str, int]] = []
    for raw in raw_shapes:
        _require(
            isinstance(raw, Mapping) and set(raw) == _WORKLOAD_SHAPE_FIELDS,
            "paper workload shape field set is not exact",
        )
        shapes.append(
            {
                "workflow_count": _strict_positive_int(
                    raw.get("workflow_count"), label="paper workflow_count"
                ),
                "concurrency": _strict_positive_int(
                    raw.get("concurrency"), label="paper concurrency"
                ),
            }
        )
    shape_keys = [
        (shape["workflow_count"], shape["concurrency"]) for shape in shapes
    ]
    _require(
        shape_keys == sorted(set(shape_keys)),
        "paper workload shapes must be unique and sorted",
    )

    raw_phases = value.get("phase_matrix")
    _require(
        isinstance(raw_phases, list) and bool(raw_phases),
        "paper phase_matrix must be nonempty",
    )
    phases: list[dict[str, Any]] = []
    seen_phases: set[str] = set()
    for raw in raw_phases:
        _require(
            isinstance(raw, Mapping) and set(raw) == _PHASE_FIELDS,
            "paper phase field set is not exact",
        )
        phase = _strict_string(raw.get("phase"), label="paper phase")
        _require(
            _IDENTIFIER_RE.fullmatch(phase) is not None
            and phase not in seen_phases,
            "paper phase is invalid or duplicated",
        )
        seen_phases.add(phase)
        normalized = {
            "phase": phase,
            "policy_count": _strict_positive_int(
                raw.get("policy_count"), label=f"paper {phase} policy_count"
            ),
            "seed_count": _strict_positive_int(
                raw.get("seed_count"), label=f"paper {phase} seed_count"
            ),
            "shape_count": _strict_positive_int(
                raw.get("shape_count"), label=f"paper {phase} shape_count"
            ),
            "run_count": _strict_positive_int(
                raw.get("run_count"), label=f"paper {phase} run_count"
            ),
        }
        _require(
            normalized["run_count"]
            == 2
            * normalized["policy_count"]
            * normalized["seed_count"]
            * normalized["shape_count"],
            f"paper {phase} run count does not encode healthy/failed pairs",
        )
        phases.append(normalized)
    return {
        "context_cap_tokens": context_cap,
        "token_projection_grid": grid,
        "workload_shapes": shapes,
        "phase_matrix": phases,
    }


def _normalize_model(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and (
            set(value) == _MODEL_FIELDS
            or set(value) == _LEGACY_MODEL_FIELDS
        ),
        "requirement model field set is not exact",
    )
    return {
        "name": _strict_string(value.get("name"), label="model name"),
        "scale_label": _strict_string(
            value.get("scale_label"), label="model scale_label"
        ),
        "revision": _optional_string(value.get("revision"), label="model revision"),
        "config_sha256": _optional_sha(
            value.get("config_sha256"), label="model config_sha256"
        ),
        "dtype": _strict_string(value.get("dtype"), label="model dtype"),
    }


def _normalize_layout(value: Any) -> dict[str, int]:
    _require(
        isinstance(value, Mapping) and set(value) == _LAYOUT_FIELDS,
        "requirement layout field set is not exact",
    )
    return {
        field: _strict_positive_int(value.get(field), label=f"layout {field}")
        for field in sorted(_LAYOUT_FIELDS)
    }


def _normalize_profile(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == _PROFILE_FIELDS,
        "physical_profile field set is not exact",
    )
    return {
        "schema_version": _strict_positive_int(
            value.get("schema_version"), label="physical profile schema_version"
        ),
        "schema_class": _optional_string(
            value.get("schema_class"), label="physical profile schema_class"
        ),
        "semantics": _strict_string(
            value.get("semantics"), label="physical profile semantics"
        ),
        "sha256": _optional_sha(
            value.get("sha256"), label="physical profile sha256"
        ),
    }


def _normalize_calibration(value: Any) -> dict[str, str | None]:
    _require(
        isinstance(value, Mapping) and set(value) == _CALIBRATION_FIELDS,
        "calibration field set is not exact",
    )
    return {
        "identity_sha256": _optional_sha(
            value.get("identity_sha256"), label="calibration identity_sha256"
        ),
        "mechanism_gate_sha256": _optional_sha(
            value.get("mechanism_gate_sha256"),
            label="calibration mechanism_gate_sha256",
        ),
    }


def _normalize_requirement(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and (
            set(value) == _REQUIREMENT_FIELDS
            or set(value) == _LEGACY_REQUIREMENT_FIELDS
        ),
        "scale-campaign requirement field set is not exact",
    )
    requirement_id = str(value.get("requirement_id") or "")
    _require(
        _IDENTIFIER_RE.fullmatch(requirement_id) is not None,
        "requirement_id is invalid",
    )
    design_semantics = _strict_string(
        value.get("design_semantics"), label=f"{requirement_id} design_semantics"
    )
    _require(
        design_semantics == DESIGN_SEMANTICS,
        f"requirement {requirement_id} uses non-current design semantics",
    )
    axes, cell_count = _normalize_workload_axes(value.get("workload_axes"))
    paper_axes = _normalize_paper_axes(value.get("paper_axes"))
    if paper_axes is not None and axes["concurrency_levels"] is not None:
        shape_concurrency = sorted(
            {shape["concurrency"] for shape in paper_axes["workload_shapes"]}
        )
        _require(
            axes["concurrency_levels"] == shape_concurrency,
            f"requirement {requirement_id} paper workload shapes disagree "
            "with declared concurrency levels",
        )
    if cell_count is not None and paper_axes is not None:
        cell_count = (
            len(paper_axes["workload_shapes"])
            * len(axes["failure_rates"])
            * len(axes["capacity_points"])
            * len(axes["seeds"])
            * len(paper_axes["token_projection_grid"])
        )
    suite_binding = value.get("suite_binding")
    if suite_binding is not None:
        suite_binding = _normalize_artifact_reference(
            suite_binding, label=f"requirement {requirement_id} suite binding"
        )
    return {
        "requirement_id": requirement_id,
        "consumer_kind": _strict_string(
            value.get("consumer_kind"), label=f"{requirement_id} consumer_kind"
        ),
        "hardware_label": _optional_string(
            value.get("hardware_label"), label=f"{requirement_id} hardware_label"
        ),
        "design_semantics": design_semantics,
        "model": _normalize_model(value.get("model")),
        "layout": _normalize_layout(value.get("layout")),
        "physical_profile": _normalize_profile(value.get("physical_profile")),
        "calibration": _normalize_calibration(value.get("calibration")),
        "lineage_identity_sha256": _optional_sha(
            value.get("lineage_identity_sha256"),
            label=f"{requirement_id} lineage_identity_sha256",
        ),
        "workload_axes": axes,
        "paper_axes": paper_axes,
        "declared_cell_count": cell_count,
        "suite_binding": suite_binding,
    }


def _requirement_gaps(requirement: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return unresolved identities without inventing placeholder values."""

    requirement_id = requirement["requirement_id"]
    missing = (
        ("hardware_label", requirement["hardware_label"], "MISSING_HARDWARE_LABEL"),
        (
            "model.revision",
            requirement["model"]["revision"],
            "MISSING_MODEL_REVISION",
        ),
        (
            "model.config_sha256",
            requirement["model"]["config_sha256"],
            "MISSING_MODEL_CONFIG_SHA256",
        ),
        (
            "physical_profile.schema_class",
            requirement["physical_profile"]["schema_class"],
            "MISSING_PHYSICAL_PROFILE_SCHEMA_CLASS",
        ),
        (
            "physical_profile.sha256",
            requirement["physical_profile"]["sha256"],
            "MISSING_PHYSICAL_PROFILE_SHA256",
        ),
        (
            "calibration.identity_sha256",
            requirement["calibration"]["identity_sha256"],
            "MISSING_CALIBRATION_IDENTITY_SHA256",
        ),
        (
            "calibration.mechanism_gate_sha256",
            requirement["calibration"]["mechanism_gate_sha256"],
            "MISSING_MECHANISM_GATE_SHA256",
        ),
        (
            "lineage_identity_sha256",
            requirement["lineage_identity_sha256"],
            "MISSING_LINEAGE_IDENTITY_SHA256",
        ),
        (
            "workload_axes.concurrency_levels",
            requirement["workload_axes"]["concurrency_levels"],
            "MISSING_CONCURRENCY_LEVELS",
        ),
        (
            "workload_axes.failure_rates",
            requirement["workload_axes"]["failure_rates"],
            "MISSING_FAILURE_RATES",
        ),
        (
            "workload_axes.capacity_points",
            requirement["workload_axes"]["capacity_points"],
            "MISSING_CAPACITY_POINTS",
        ),
        (
            "workload_axes.seeds",
            requirement["workload_axes"]["seeds"],
            "MISSING_SEEDS",
        ),
    )
    return [
        {
            "requirement_id": requirement_id,
            "code": code,
            "field": field,
            "message": f"{field} is not yet bound to a measured lineage",
        }
        for field, observed, code in missing
        if observed is None
    ]


def _load_requirement_manifest(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    manifest = _load(path, label="scale-campaign requirement manifest")
    _require(
        set(manifest) == _MANIFEST_FIELDS,
        "scale-campaign requirement manifest field set is not exact",
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("evidence_class") == REQUIREMENTS_EVIDENCE_CLASS,
        "scale-campaign requirement manifest identity is invalid",
    )
    campaign_label = str(manifest.get("campaign_label") or "")
    _require(
        _IDENTIFIER_RE.fullmatch(campaign_label) is not None,
        "campaign_label is invalid",
    )
    limits = manifest.get("limits")
    _require(
        isinstance(limits, Mapping) and set(limits) == _LIMIT_FIELDS,
        "scale-campaign limits field set is not exact",
    )
    max_requirements = _strict_positive_int(
        limits.get("max_requirements"), label="limits max_requirements"
    )
    max_total_cells = _strict_positive_int(
        limits.get("max_total_cells"), label="limits max_total_cells"
    )
    _require(
        max_requirements <= MAX_SCALE_CAMPAIGN_REQUIREMENTS,
        f"max_requirements exceeds hard limit {MAX_SCALE_CAMPAIGN_REQUIREMENTS}",
    )
    _require(
        max_total_cells <= MAX_SCALE_CAMPAIGN_CELLS,
        f"max_total_cells exceeds hard limit {MAX_SCALE_CAMPAIGN_CELLS}",
    )
    raw_requirements = manifest.get("requirements")
    _require(
        isinstance(raw_requirements, list) and bool(raw_requirements),
        "scale-campaign requirements must be nonempty",
    )
    _require(
        len(raw_requirements) <= max_requirements,
        "scale-campaign requirement cardinality exceeds declared limit",
    )
    requirements = [_normalize_requirement(value) for value in raw_requirements]
    ids = [value["requirement_id"] for value in requirements]
    _require(len(ids) == len(set(ids)), "duplicate requirement_id is forbidden")
    lineage_ids = [
        value["lineage_identity_sha256"]
        for value in requirements
        if value["lineage_identity_sha256"] is not None
    ]
    _require(
        len(lineage_ids) == len(set(lineage_ids)),
        "duplicate declared lineage requirement is forbidden",
    )
    total_cells = sum(
        value["declared_cell_count"] or 0 for value in requirements
    )
    _require(
        total_cells <= max_total_cells,
        "scale-campaign cell cardinality exceeds declared limit",
    )
    return manifest, sorted(requirements, key=lambda value: value["requirement_id"])


def _axes_from_suite_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    matrix = binding["matrix"]
    return {
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


def _match_requirement(
    requirement: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    runner: Mapping[str, Any],
) -> None:
    requirement_id = requirement["requirement_id"]
    source = binding["source_consumer"]
    _require(
        requirement["consumer_kind"] == source["consumer_kind"],
        f"requirement {requirement_id} consumer kind crosses suite lineage",
    )
    _require(
        requirement["hardware_label"] == source["hardware_label"],
        f"requirement {requirement_id} hardware does not match suite binding",
    )
    _require(
        requirement["design_semantics"] == source["design_semantics"],
        f"requirement {requirement_id} design semantics do not match suite binding",
    )
    observed_model = source["model"]
    _require(
        requirement["model"]["revision"] == observed_model["revision"],
        f"requirement {requirement_id} model revision does not match suite binding",
    )
    _require(
        requirement["model"] == observed_model,
        f"requirement {requirement_id} model identity does not match suite binding",
    )
    observed_layout = source["layout"]
    layout_labels = {
        "tp_size": "TP",
        "pp_size": "PP",
        "system_dp_size": "system-DP",
        "worker_count": "worker-count",
    }
    for field, label in layout_labels.items():
        _require(
            requirement["layout"][field] == observed_layout[field],
            f"requirement {requirement_id} {label} layout does not match suite binding",
        )
    observed_profile = {
        "schema_version": source["profile_schema_version"],
        "schema_class": source["profile_schema_class"],
        "semantics": source["profile_semantics"],
        "sha256": source["profile_sha256"],
    }
    _require(
        requirement["physical_profile"] == observed_profile,
        f"requirement {requirement_id} physical profile identity does not match",
    )
    observed_calibration = {
        "identity_sha256": source["calibration_identity_sha256"],
        "mechanism_gate_sha256": source["mechanism_gate_sha256"],
    }
    _require(
        requirement["calibration"] == observed_calibration,
        f"requirement {requirement_id} calibration identity does not match",
    )
    _require(
        requirement["lineage_identity_sha256"]
        == source["lineage_identity_sha256"],
        f"requirement {requirement_id} lineage identity does not match",
    )
    _require(
        requirement["workload_axes"] == _axes_from_suite_binding(binding),
        f"requirement {requirement_id} workload axes do not match suite binding",
    )
    paper_axes = requirement["paper_axes"]
    if paper_axes is None:
        expected_cells = binding["matrix"]["cell_count"]
    else:
        shapes_by_concurrency: dict[int, int] = {}
        for shape in paper_axes["workload_shapes"]:
            concurrency = shape["concurrency"]
            shapes_by_concurrency[concurrency] = (
                shapes_by_concurrency.get(concurrency, 0) + 1
            )
        _require(
            set(shapes_by_concurrency)
            == set(requirement["workload_axes"]["concurrency_levels"]),
            f"requirement {requirement_id} paper shapes do not cover each "
            "calibrated concurrency",
        )
        expected_cells = sum(
            shapes_by_concurrency[int(local["axes"]["concurrency"])]
            * len(paper_axes["token_projection_grid"])
            for local in runner["cells"]
        )
    _require(
        binding["matrix"]["cell_count"] == runner["cell_count"]
        and requirement["declared_cell_count"] == expected_cells,
        f"requirement {requirement_id} cell cardinality does not match suite binding",
    )
    _require(
        source.get("consumer_execution_authorized") is False,
        f"requirement {requirement_id} consumer unexpectedly authorizes execution",
    )
    _require(
        runner.get("authorization")
        == {
            "trace_loading_authorized": False,
            "model_fitting_authorized": False,
            "replay_authorized": False,
            "result_output_authorized": False,
            "performance_claim_authorized": False,
        },
        f"requirement {requirement_id} suite runner is not static-only",
    )


def _evaluate(
    requirement_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requirement_manifest_path = Path(requirement_manifest_path).resolve()
    manifest, requirements = _load_requirement_manifest(requirement_manifest_path)
    manifest_record = artifact_record(requirement_manifest_path)
    gaps: list[dict[str, Any]] = []
    bound: list[dict[str, Any]] = []
    seen_binding_identities: set[str] = set()

    for requirement in requirements:
        requirement_id = requirement["requirement_id"]
        identity_gaps = _requirement_gaps(requirement)
        if identity_gaps:
            gaps.extend(identity_gaps)
            if requirement["suite_binding"] is None:
                gaps.append(
                    {
                        "requirement_id": requirement_id,
                        "code": "MISSING_SUITE_BINDING_REFERENCE",
                        "message": (
                            "an independently validated calibrated-suite binding "
                            "has not been supplied"
                        ),
                    }
                )
            continue
        reference = requirement["suite_binding"]
        if reference is None:
            gaps.append(
                {
                    "requirement_id": requirement_id,
                    "code": "MISSING_SUITE_BINDING_REFERENCE",
                    "message": (
                        "an independently validated calibrated-suite binding "
                        "has not been supplied"
                    ),
                }
            )
            continue
        binding_path = Path(reference["path"])
        if not binding_path.is_file():
            gaps.append(
                {
                    "requirement_id": requirement_id,
                    "code": "SUITE_BINDING_ARTIFACT_MISSING",
                    "path": str(binding_path),
                    "message": "the declared calibrated-suite binding is absent",
                }
            )
            continue

        binding_path = _verify_artifact_reference(
            reference, label=f"requirement {requirement_id} suite binding"
        )
        try:
            binding = validate_suite_binding(binding_path)
            runner = preflight_suite_runner(binding_path)
        except (CalibratedSuiteContractError, OSError, ValueError) as exc:
            raise ScaleCampaignContractError(
                f"requirement {requirement_id} calibrated-suite revalidation "
                f"failed: {exc}"
            ) from exc
        _require(
            binding.get("evidence_class") == SUITE_BINDING_EVIDENCE_CLASS,
            f"requirement {requirement_id} does not reference a calibrated suite",
        )
        binding_identity = str(binding["suite_binding_identity_sha256"])
        _require(
            binding_identity not in seen_binding_identities,
            "one calibrated-suite binding cannot be reused by multiple requirements",
        )
        seen_binding_identities.add(binding_identity)
        _match_requirement(requirement, binding=binding, runner=runner)
        bound.append(
            {
                "requirement": requirement,
                "suite_binding": binding,
                "suite_runner": runner,
                "suite_binding_artifact": artifact_record(binding_path),
            }
        )

    ready_ids = [value["requirement"]["requirement_id"] for value in bound]
    blocked_ids = sorted(
        {str(value["requirement_id"]) for value in gaps}
    )
    blocked = bool(gaps)
    readiness = {
        "schema_version": 1,
        "evidence_class": READINESS_EVIDENCE_CLASS,
        "status": "BLOCKED_MISSING_LINEAGES" if blocked else "READY_STATIC_ONLY",
        "campaign_label": manifest["campaign_label"],
        "requirement_manifest": manifest_record,
        "requirement_count": len(requirements),
        "ready_requirement_ids": ready_ids,
        "blocked_requirement_ids": blocked_ids,
        "gaps": gaps,
        "planned_cell_count": (
            0
            if blocked
            else sum(value["declared_cell_count"] or 0 for value in requirements)
        ),
        "cells": [],
        "authorization": (
            dict(SCALE_CAMPAIGN_BLOCKED_AUTHORIZATION)
            if blocked
            else {
                **SCALE_CAMPAIGN_BLOCKED_AUTHORIZATION,
                "campaign_binding_materialization_authorized": True,
            }
        ),
        "evidence_boundary": (
            "Readiness validation only. Missing lineages create no cells or "
            "output; complete readiness permits only immutable static campaign "
            "binding, never trace, replay, result, formal, or paper use."
        ),
    }
    return readiness, bound


def evaluate_scale_campaign_readiness(
    requirement_manifest_path: Path,
) -> dict[str, Any]:
    """Return machine-readable lineage gaps without creating output."""

    readiness, _bound = _evaluate(requirement_manifest_path)
    return readiness


def load_scale_campaign_requirements(
    requirement_manifest_path: Path,
) -> list[dict[str, Any]]:
    """Return the normalized declarations without validating any suite input."""

    _manifest, requirements = _load_requirement_manifest(
        Path(requirement_manifest_path).resolve()
    )
    return _clone(requirements)


def _global_cells(bound: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    seen_global_ids: set[str] = set()
    for value in bound:
        requirement = value["requirement"]
        binding = value["suite_binding"]
        runner = value["suite_runner"]
        seen_local_ids: set[str] = set()
        paper_axes = requirement.get("paper_axes")
        expanded_locals: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for local in runner["cells"]:
            if paper_axes is None:
                expanded_locals.append((local, _clone(local["axes"])))
                continue
            matching_shapes = [
                shape
                for shape in paper_axes["workload_shapes"]
                if shape["concurrency"] == local["axes"]["concurrency"]
            ]
            _require(
                bool(matching_shapes),
                f"requirement {requirement['requirement_id']} has no paper "
                "shape for a calibrated concurrency",
            )
            for shape in matching_shapes:
                for projection in paper_axes["token_projection_grid"]:
                    axes = _clone(local["axes"])
                    axes["workflow_shape"] = _clone(shape)
                    axes["token_projection"] = _clone(projection)
                    axes["context_cap_tokens"] = paper_axes[
                        "context_cap_tokens"
                    ]
                    expanded_locals.append((local, axes))

        for local, expanded_axes in expanded_locals:
            local_id = str(local["cell_id"])
            if paper_axes is None:
                _require(
                    local_id not in seen_local_ids,
                    f"requirement {requirement['requirement_id']} repeats a "
                    "suite cell",
                )
                seen_local_ids.add(local_id)
            identity_payload = {
                "requirement_id": requirement["requirement_id"],
                "lineage_identity_sha256": requirement[
                    "lineage_identity_sha256"
                ],
                "suite_binding_identity_sha256": binding[
                    "suite_binding_identity_sha256"
                ],
                "suite_cell_id": local_id,
                "axes": expanded_axes,
            }
            global_id = canonical_sha256(identity_payload)
            _require(global_id not in seen_global_ids, "duplicate global cell detected")
            seen_global_ids.add(global_id)
            cells.append(
                {
                    "cell_id": global_id,
                    "requirement_id": requirement["requirement_id"],
                    "lineage_identity_sha256": requirement[
                        "lineage_identity_sha256"
                    ],
                    "suite_binding_identity_sha256": binding[
                        "suite_binding_identity_sha256"
                    ],
                    "suite_cell_id": local_id,
                    "axes": expanded_axes,
                    "state": "PLANNED_NOT_RUN",
                }
            )
    return cells


def build_scale_campaign_binding(
    requirement_manifest_path: Path,
) -> dict[str, Any]:
    """Build a complete in-memory campaign after revalidating every lineage."""

    readiness, bound = _evaluate(requirement_manifest_path)
    _require(
        readiness["status"] == "READY_STATIC_ONLY",
        "scale campaign is blocked by missing calibrated lineages",
    )
    manifest = _load(
        Path(requirement_manifest_path).resolve(),
        label="scale-campaign requirement manifest",
    )
    cells = _global_cells(bound)
    _require(
        len(cells) == readiness["planned_cell_count"],
        "global campaign cell cardinality changed",
    )
    requirements_payload: list[dict[str, Any]] = []
    for value in bound:
        requirement = _clone(value["requirement"])
        requirement.pop("suite_binding")
        requirements_payload.append(
            {
                **requirement,
                "suite_binding": value["suite_binding_artifact"],
                "suite_binding_identity_sha256": value["suite_binding"][
                    "suite_binding_identity_sha256"
                ],
            }
        )
    identity_payload = {
        "schema_version": 1,
        "evidence_class": CAMPAIGN_BINDING_EVIDENCE_CLASS,
        "requirement_manifest_sha256": readiness["requirement_manifest"]["sha256"],
        "requirements": [
            {
                "requirement_id": value["requirement_id"],
                "lineage_identity_sha256": value["lineage_identity_sha256"],
                "suite_binding_sha256": value["suite_binding"]["sha256"],
                "suite_binding_identity_sha256": value[
                    "suite_binding_identity_sha256"
                ],
            }
            for value in requirements_payload
        ],
        "cell_ids": [cell["cell_id"] for cell in cells],
    }
    campaign_identity = canonical_sha256(identity_payload)
    return {
        "schema_version": 1,
        "evidence_class": CAMPAIGN_BINDING_EVIDENCE_CLASS,
        "status": CAMPAIGN_BINDING_STATUS,
        "campaign_binding_identity_sha256": campaign_identity,
        "campaign_id": f"aegiskv-scale-campaign-{campaign_identity[:16]}",
        "campaign_label": manifest["campaign_label"],
        "binding_directory_name": f"scale_campaign_binding_{campaign_identity[:16]}",
        "design_semantics": DESIGN_SEMANTICS,
        "requirement_manifest": readiness["requirement_manifest"],
        "limits": _clone(manifest["limits"]),
        "requirements": requirements_payload,
        "cell_count": len(cells),
        "cells": cells,
        "runner_contract": {
            "schema_version": 1,
            "all_suite_bindings_revalidated_before_cell_construction": True,
            "campaign_binding_rehash_before_trace_model_or_output": True,
            "hash_derived_output_directory": (
                f"scale_campaign_run_{campaign_identity[:16]}"
            ),
            "immutable_output_refuse_overwrite": True,
            "cross_hardware_model_revision_or_layout_reuse_allowed": False,
            "lineage_identity_in_every_cell": True,
            "planned_cell_state": "PLANNED_NOT_RUN",
        },
        "authorization": dict(SCALE_CAMPAIGN_STATIC_AUTHORIZATION),
        "result_state": {
            "execution_status": "NOT_RUN",
            "performance_results_present": False,
            "result_artifacts": [],
        },
        "evidence_boundary": (
            "Multi-lineage large-model/config readiness plan only. Every model "
            "revision and TP/PP/system-DP layout retains its own validated "
            "physical/calibration lineage. No extrapolation, trace, replay, "
            "figure, performance result, formal claim, or paper claim exists."
        ),
    }


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


def materialize_scale_campaign_binding(
    *, requirement_manifest_path: Path, output_root: Path
) -> Path:
    """Atomically write one complete, immutable, static campaign binding."""

    binding = build_scale_campaign_binding(requirement_manifest_path)
    output_root = Path(output_root).resolve()
    directory = output_root / binding["binding_directory_name"]
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite immutable campaign: {directory}")

    # No output path is created until every lower suite and global invariant
    # has passed.  This is the trace/model/output ordering fence for the CLI.
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = output_root / (
        f".{binding['binding_directory_name']}.tmp-{os.getpid()}"
    )
    if temporary_directory.exists():
        raise FileExistsError(temporary_directory)
    temporary_directory.mkdir()
    binding_path = temporary_directory / "scale-campaign-binding.json"
    payload = (
        json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        _write_new(binding_path, payload)
        validate_scale_campaign_binding(binding_path)
        os.replace(temporary_directory, directory)
    finally:
        if binding_path.exists():
            binding_path.unlink()
        if temporary_directory.exists():
            temporary_directory.rmdir()
    return directory / "scale-campaign-binding.json"


def validate_scale_campaign_binding(path: Path) -> dict[str, Any]:
    """Rebuild one campaign from its requirement manifest and suite inputs."""

    path = Path(path).resolve()
    binding = _load(path, label="scale-campaign binding")
    _require(
        binding.get("schema_version") == 1
        and binding.get("evidence_class") == CAMPAIGN_BINDING_EVIDENCE_CLASS
        and binding.get("status") == CAMPAIGN_BINDING_STATUS
        and binding.get("authorization") == SCALE_CAMPAIGN_STATIC_AUTHORIZATION
        and binding.get("result_state")
        == {
            "execution_status": "NOT_RUN",
            "performance_results_present": False,
            "result_artifacts": [],
        },
        "scale-campaign binding is not static-only",
    )
    requirement_manifest_path = _verify_artifact_reference(
        binding.get("requirement_manifest"), label="campaign requirement manifest"
    )
    expected = build_scale_campaign_binding(requirement_manifest_path)
    _require(binding == expected, "scale-campaign binding is not reproducible")
    return binding


def preflight_scale_campaign_runner(path: Path) -> dict[str, Any]:
    """Return an in-memory static plan after revalidating all lineages again."""

    path = Path(path).resolve()
    binding = validate_scale_campaign_binding(path)
    cells = _clone(binding["cells"])
    _require(
        len(cells) == binding["cell_count"]
        and len({cell["cell_id"] for cell in cells}) == len(cells)
        and all(cell["state"] == "PLANNED_NOT_RUN" for cell in cells),
        "scale-campaign runner cells are invalid",
    )
    return {
        "schema_version": 1,
        "evidence_class": CAMPAIGN_RUNNER_PREFLIGHT_EVIDENCE_CLASS,
        "status": "PASS_STATIC_ONLY",
        "campaign_binding": artifact_record(path),
        "campaign_binding_identity_sha256": binding[
            "campaign_binding_identity_sha256"
        ],
        "output_directory_name": binding["runner_contract"][
            "hash_derived_output_directory"
        ],
        "cell_count": len(cells),
        "cells": cells,
        "authorization": {
            "runnable_cells_authorized": False,
            "trace_loading_authorized": False,
            "model_fitting_authorized": False,
            "replay_authorized": False,
            "result_output_authorized": False,
            "performance_result_authorized": False,
            "performance_claim_authorized": False,
            "formal_ready": False,
            "paper_ready": False,
        },
        "evidence_boundary": (
            "Static multi-lineage runner preflight only. Every cell is "
            "PLANNED_NOT_RUN and contains no performance fields."
        ),
    }
