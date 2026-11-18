#!/usr/bin/env python3
"""Static EuroSys evaluation-matrix alignment and lineage readiness.

This module consumes the generic declarative multi-lineage requirement
manifest.  It compares its paper axes with the checked paper source and then
delegates all physical-lineage checks to ``scale_campaign_contract``.  It
never creates campaign cells, loads a trace, fits a model, or runs a replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    artifact_record,
    canonical_sha256,
)
from AegisKV.experiments.real_weka_full_design.scale_campaign_contract import (
    ScaleCampaignContractError,
    evaluate_scale_campaign_readiness,
    load_scale_campaign_requirements,
)


PAPER_READINESS_EVIDENCE_CLASS = (
    "sglsim_aegiskv_eurosys_paper_campaign_readiness_v1"
)
PAPER_BLOCKED_AUTHORIZATION = {
    "readiness_artifact_authorized": True,
    "campaign_binding_materialization_authorized": False,
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


class EuroSysPaperReadinessError(ValueError):
    """Raised when the paper declaration or readiness artifact is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EuroSysPaperReadinessError(message)


def _read_text(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EuroSysPaperReadinessError(
            f"cannot read evaluation source {path}: {exc}"
        ) from exc


def _model_paragraphs(text: str, scale_label: str) -> list[str]:
    return [
        paragraph
        for paragraph in re.split(r"\n\s*\n", text)
        if scale_label.lower() in paragraph.lower()
    ]


def _layout_pattern(layout: Mapping[str, int]) -> re.Pattern[str]:
    return re.compile(
        rf"TP\s*=\s*{layout['tp_size']}.*?"
        rf"PP\s*=\s*{layout['pp_size']}.*?"
        rf"(?:system(?:-level)?[-\s]*)?DP\s*=\s*"
        rf"{layout['system_dp_size']}",
        flags=re.IGNORECASE | re.DOTALL,
    )


def _paper_declarations(text: str) -> dict[str, Any]:
    projection_paragraphs = [
        paragraph
        for paragraph in re.split(r"\n\s*\n", text)
        if "token projection" in paragraph.lower()
    ]
    projection_text = "\n".join(projection_paragraphs)
    projection_scales = sorted(
        {
            float(value)
            for value in re.findall(
                r"(?<![\d.])0\.\d+(?![\d.])", projection_text
            )
        }
    )
    workload_shapes = sorted(
        {
            (int(workflows), int(concurrency))
            for workflows, concurrency in re.findall(
                r"\$\(\s*(\d+)\s*,\s*(\d+)\s*\)\$", text
            )
        }
    )
    phase_pattern = re.compile(
        r"^\s*(Pilot|Central|Scale|Ablations)\s*&\s*(\d+)\s*&\s*"
        r"(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*\\\\\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    phase_matrix = [
        {
            "phase": phase.lower(),
            "policy_count": int(policy_count),
            "seed_count": int(seed_count),
            "shape_count": int(shape_count),
            "run_count": int(run_count),
        }
        for phase, policy_count, seed_count, shape_count, run_count in (
            match.groups() for match in phase_pattern.finditer(text)
        )
    ]
    return {
        "token_projection_scales": projection_scales,
        "workload_shapes": [
            {"workflow_count": workflows, "concurrency": concurrency}
            for workflows, concurrency in workload_shapes
        ],
        "phase_matrix": phase_matrix,
    }


def _require_shared_paper_axes(requirements: list[dict[str, Any]]) -> None:
    paper_axes = [requirement.get("paper_axes") for requirement in requirements]
    _require(
        all(value is not None for value in paper_axes),
        "every EuroSys requirement must declare paper_axes",
    )
    reference = paper_axes[0]
    for requirement, observed in zip(requirements[1:], paper_axes[1:]):
        for field in ("token_projection_grid", "workload_shapes", "phase_matrix"):
            _require(
                observed[field] == reference[field],
                f"requirement {requirement['requirement_id']} does not retain "
                f"the shared paper {field}",
            )


def _align_requirement_to_paper(
    requirement: Mapping[str, Any], text: str
) -> dict[str, Any]:
    requirement_id = str(requirement["requirement_id"])
    model = requirement["model"]
    layout = requirement["layout"]
    paragraphs = _model_paragraphs(text, str(model["scale_label"]))
    joined = "\n".join(paragraphs)
    checks = {
        "model_scale_declared": bool(paragraphs),
        "dtype_declared": bool(
            paragraphs
            and re.search(
                rf"\b{re.escape(str(model['dtype']))}\b",
                joined,
                flags=re.IGNORECASE,
            )
        ),
        "layout_declared": bool(
            paragraphs and _layout_pattern(layout).search(joined)
        ),
        "context_cap_declared": bool(
            paragraphs
            and (
                f"{requirement['paper_axes']['context_cap_tokens']:,}" in joined
                or str(requirement["paper_axes"]["context_cap_tokens"]) in joined
            )
        ),
    }
    hardware_label = requirement.get("hardware_label")
    if hardware_label is None:
        checks["hardware_declared"] = False
    else:
        hardware_token = str(hardware_label).split("-", maxsplit=1)[0]
        checks["hardware_declared"] = bool(
            paragraphs
            and re.search(
                rf"\b{re.escape(hardware_token)}\b",
                joined,
                flags=re.IGNORECASE,
            )
        )
    if layout["worker_count"] != layout["system_dp_size"]:
        checks["scheduler_process_count_declared"] = bool(
            paragraphs
            and re.search(
                rf"\b{layout['worker_count']}\s+Scheduler\s+processes\b",
                joined,
                flags=re.IGNORECASE,
            )
        )
    else:
        checks["scheduler_process_count_declared"] = True

    gaps = [
        {
            "requirement_id": requirement_id,
            "code": f"PAPER_{name.upper()}_MISSING",
            "field": name,
            "message": (
                f"05-evaluation.tex does not declare the requirement's {name}"
            ),
        }
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "requirement_id": requirement_id,
        "checks": checks,
        "declared_in_evaluation_source": not gaps,
        "gaps": gaps,
    }


def build_eurosys_paper_readiness(
    *, requirement_manifest_path: Path, evaluation_source_path: Path
) -> dict[str, Any]:
    """Build a no-cells paper/readiness audit from declarative inputs."""

    requirement_manifest_path = Path(requirement_manifest_path).resolve()
    evaluation_source_path = Path(evaluation_source_path).resolve()
    try:
        generic = evaluate_scale_campaign_readiness(requirement_manifest_path)
        requirements = load_scale_campaign_requirements(requirement_manifest_path)
    except (ScaleCampaignContractError, OSError, ValueError) as exc:
        raise EuroSysPaperReadinessError(
            f"generic scale-campaign requirement validation failed: {exc}"
        ) from exc
    _require_shared_paper_axes(requirements)
    text = _read_text(evaluation_source_path)
    observed = _paper_declarations(text)
    reference_axes = requirements[0]["paper_axes"]
    expected_scales = [
        point["scale"] for point in reference_axes["token_projection_grid"]
    ]
    expected_shapes = reference_axes["workload_shapes"]
    expected_phases = reference_axes["phase_matrix"]
    global_checks = {
        "token_projection_grid_matches": (
            observed["token_projection_scales"] == expected_scales
        ),
        "workload_shapes_match": observed["workload_shapes"] == expected_shapes,
        "phase_matrix_matches": observed["phase_matrix"] == expected_phases,
    }
    global_gaps = [
        {
            "requirement_id": None,
            "code": f"PAPER_{name.upper()}_MISMATCH",
            "field": name,
            "message": "declarative paper axes disagree with 05-evaluation.tex",
        }
        for name, passed in global_checks.items()
        if not passed
    ]
    requirement_alignment = [
        _align_requirement_to_paper(requirement, text)
        for requirement in requirements
    ]
    paper_gaps = global_gaps + [
        gap for alignment in requirement_alignment for gap in alignment["gaps"]
    ]
    all_gaps = [
        {"scope": "physical_lineage", **gap} for gap in generic["gaps"]
    ] + [{"scope": "paper_source", **gap} for gap in paper_gaps]
    blocked = generic["status"] != "READY_STATIC_ONLY" or bool(paper_gaps)
    implementation_directory = Path(__file__).resolve().parent
    validator_implementation = {
        "paper_readiness": artifact_record(Path(__file__).resolve()),
        "scale_campaign_contract": artifact_record(
            implementation_directory / "scale_campaign_contract.py"
        ),
        "calibrated_suite_contract": artifact_record(
            implementation_directory / "calibrated_suite_contract.py"
        ),
    }
    identity_payload = {
        "schema_version": 1,
        "evidence_class": PAPER_READINESS_EVIDENCE_CLASS,
        "requirement_manifest_sha256": artifact_record(
            requirement_manifest_path
        )["sha256"],
        "evaluation_source_sha256": artifact_record(evaluation_source_path)[
            "sha256"
        ],
        "generic_status": generic["status"],
        "validator_sha256": {
            name: value["sha256"]
            for name, value in validator_implementation.items()
        },
        "gap_codes": [
            [gap.get("requirement_id"), gap["scope"], gap["code"]]
            for gap in all_gaps
        ],
    }
    return {
        "schema_version": 1,
        "evidence_class": PAPER_READINESS_EVIDENCE_CLASS,
        "status": (
            "BLOCKED_PAPER_REQUIREMENTS" if blocked else "READY_STATIC_ONLY"
        ),
        "readiness_identity_sha256": canonical_sha256(identity_payload),
        "requirement_manifest": artifact_record(requirement_manifest_path),
        "evaluation_source": artifact_record(evaluation_source_path),
        "validator_implementation": validator_implementation,
        "generic_scale_campaign_readiness": generic,
        "paper_declarations_observed": observed,
        "paper_global_checks": global_checks,
        "requirement_alignment": requirement_alignment,
        "bound_requirement_ids": list(generic["ready_requirement_ids"]),
        "missing_requirement_ids": list(generic["blocked_requirement_ids"]),
        "gaps": all_gaps,
        "planned_cell_count": 0,
        "cells": [],
        "authorization": dict(PAPER_BLOCKED_AUTHORIZATION),
        "result_state": {
            "execution_status": "NOT_RUN",
            "performance_results_present": False,
            "result_artifacts": [],
        },
        "evidence_boundary": (
            "Paper-matrix declaration and physical-lineage readiness only. "
            "A missing lineage or paper-source mismatch creates no cells and "
            "authorizes no trace, replay, result, formal, or paper claim."
        ),
    }


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_eurosys_paper_readiness(
    *,
    requirement_manifest_path: Path,
    evaluation_source_path: Path,
    output_path: Path,
) -> Path:
    readiness = build_eurosys_paper_readiness(
        requirement_manifest_path=requirement_manifest_path,
        evaluation_source_path=evaluation_source_path,
    )
    payload = (
        json.dumps(readiness, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output_path = Path(output_path).resolve()
    _write_new(output_path, payload)
    validate_eurosys_paper_readiness(output_path)
    return output_path


def validate_eurosys_paper_readiness(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EuroSysPaperReadinessError(
            f"cannot load paper readiness artifact {path}: {exc}"
        ) from exc
    _require(isinstance(observed, dict), "paper readiness artifact is not an object")
    for field in ("requirement_manifest", "evaluation_source"):
        reference = observed.get(field)
        _require(isinstance(reference, Mapping), f"{field} reference is missing")
        artifact = Path(str(reference.get("path") or "")).resolve()
        _require(artifact.is_file(), f"{field} artifact is missing")
        _require(
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            == reference.get("sha256")
            and artifact.stat().st_size == reference.get("size_bytes"),
            f"{field} artifact hash/size mismatch",
        )
    expected = build_eurosys_paper_readiness(
        requirement_manifest_path=Path(observed["requirement_manifest"]["path"]),
        evaluation_source_path=Path(observed["evaluation_source"]["path"]),
    )
    _require(observed == expected, "paper readiness artifact is not reproducible")
    return observed
