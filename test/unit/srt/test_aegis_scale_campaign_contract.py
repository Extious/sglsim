from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    DESIGN_SEMANTICS,
    artifact_record,
)
from AegisKV.experiments.real_weka_full_design.calibrated_suite_contract import (
    A100_CONSUMER_KIND,
    V100_CONSUMER_KIND,
    CalibratedConsumerInput,
    materialize_suite_binding,
    validate_suite_binding,
)
from AegisKV.experiments.real_weka_full_design.scale_campaign_contract import (
    CAMPAIGN_BINDING_EVIDENCE_CLASS,
    REQUIREMENTS_EVIDENCE_CLASS,
    SCALE_CAMPAIGN_STATIC_AUTHORIZATION,
    ScaleCampaignContractError,
    _global_cells,
    build_scale_campaign_binding,
    evaluate_scale_campaign_readiness,
    materialize_scale_campaign_binding,
    preflight_scale_campaign_runner,
    validate_scale_campaign_binding,
)
from test.unit.srt.test_aegis_a100_dp2_contract import (
    _a100_suite_matrix,
    _materialized_a100_consumer,
)
from test.unit.srt.test_aegis_real_weka_calibration import (
    _materialized_v100_import,
    _v100_calibrated_suite_matrix,
)


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "AegisKV/experiments/real_weka_full_design/bind_scale_campaign.py"
)


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _raw_axes(binding: Mapping[str, Any]) -> dict[str, Any]:
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


def _requirement(
    *, requirement_id: str, binding_path: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    source = binding["source_consumer"]
    return {
        "requirement_id": requirement_id,
        "consumer_kind": source["consumer_kind"],
        "hardware_label": source["hardware_label"],
        "design_semantics": DESIGN_SEMANTICS,
        "model": deepcopy(source["model"]),
        "layout": deepcopy(source["layout"]),
        "physical_profile": {
            "schema_version": source["profile_schema_version"],
            "schema_class": source["profile_schema_class"],
            "semantics": source["profile_semantics"],
            "sha256": source["profile_sha256"],
        },
        "calibration": {
            "identity_sha256": source["calibration_identity_sha256"],
            "mechanism_gate_sha256": source["mechanism_gate_sha256"],
        },
        "lineage_identity_sha256": source["lineage_identity_sha256"],
        "workload_axes": _raw_axes(binding),
        "suite_binding": artifact_record(binding_path),
    }


def _manifest(
    requirements: list[dict[str, Any]], *, max_total_cells: int = 10_000
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_class": REQUIREMENTS_EVIDENCE_CLASS,
        "campaign_label": "declared-multi-lineage-scale-readiness",
        "limits": {
            "max_requirements": 16,
            "max_total_cells": max_total_cells,
        },
        "requirements": requirements,
    }


@pytest.fixture(scope="module")
def calibrated_lineages(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("aegis-scale-campaign-lineages")

    _inputs, a100_manifest, a100_profile = _materialized_a100_consumer(
        root / "a100"
    )
    a100_binding_path = materialize_suite_binding(
        source=CalibratedConsumerInput(
            consumer_kind=A100_CONSUMER_KIND,
            consumer_manifest=a100_manifest,
            consumer_payload=a100_profile,
        ),
        matrix=_a100_suite_matrix(),
        output_root=root / "a100-suite",
    )
    a100_binding = validate_suite_binding(a100_binding_path)

    v100_root = root / "v100"
    v100_root.mkdir()
    v100_config_path, v100_manifest_path, v100_config, _import_manifest = (
        _materialized_v100_import(v100_root, calibrated_pressure=True)
    )
    v100_binding_path = materialize_suite_binding(
        source=CalibratedConsumerInput(
            consumer_kind=V100_CONSUMER_KIND,
            consumer_manifest=v100_manifest_path,
            consumer_payload=v100_config_path,
        ),
        matrix=_v100_calibrated_suite_matrix(v100_config),
        output_root=root / "v100-suite",
    )
    v100_binding = validate_suite_binding(v100_binding_path)

    return {
        "a100_path": a100_binding_path,
        "v100_path": v100_binding_path,
        "a100": a100_binding,
        "v100": v100_binding,
        "a100_requirement": _requirement(
            requirement_id="a100-qwen3-8b-tp1-pp1-dp2",
            binding_path=a100_binding_path,
            binding=a100_binding,
        ),
        "v100_requirement": _requirement(
            requirement_id="v100-qwen3-8b-tp1-pp1-dp4",
            binding_path=v100_binding_path,
            binding=v100_binding,
        ),
    }


def _valid_requirements(lineages: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        deepcopy(lineages["a100_requirement"]),
        deepcopy(lineages["v100_requirement"]),
    ]


def test_valid_multi_lineage_campaign_is_static_and_globally_unique(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    manifest_path = _write_json(
        tmp_path / "requirements.json",
        _manifest(_valid_requirements(calibrated_lineages)),
    )
    readiness = evaluate_scale_campaign_readiness(manifest_path)
    assert readiness["status"] == "READY_STATIC_ONLY"
    assert readiness["gaps"] == []
    assert readiness["cells"] == []

    binding_path = materialize_scale_campaign_binding(
        requirement_manifest_path=manifest_path,
        output_root=tmp_path / "campaign-output",
    )
    binding = validate_scale_campaign_binding(binding_path)
    assert binding["evidence_class"] == CAMPAIGN_BINDING_EVIDENCE_CLASS
    assert binding["status"] == "BOUND_STATIC_ONLY"
    assert binding["cell_count"] == 36
    assert len(binding["requirements"]) == 2
    assert len({cell["cell_id"] for cell in binding["cells"]}) == 36
    assert len({cell["lineage_identity_sha256"] for cell in binding["cells"]}) == 2
    assert all(
        len(cell["cell_id"]) == 64
        and cell["state"] == "PLANNED_NOT_RUN"
        and set(cell)
        == {
            "cell_id",
            "requirement_id",
            "lineage_identity_sha256",
            "suite_binding_identity_sha256",
            "suite_cell_id",
            "axes",
            "state",
        }
        for cell in binding["cells"]
    )
    assert binding["authorization"] == SCALE_CAMPAIGN_STATIC_AUTHORIZATION
    assert binding["result_state"] == {
        "execution_status": "NOT_RUN",
        "performance_results_present": False,
        "result_artifacts": [],
    }

    runner = preflight_scale_campaign_runner(binding_path)
    assert runner["status"] == "PASS_STATIC_ONLY"
    assert runner["cell_count"] == 36
    assert runner["authorization"]["runnable_cells_authorized"] is False
    assert runner["authorization"]["replay_authorized"] is False
    assert runner["authorization"]["performance_result_authorized"] is False
    assert runner["authorization"]["formal_ready"] is False
    assert runner["authorization"]["paper_ready"] is False
    assert not (binding_path.parent / "replay.json").exists()
    assert not (binding_path.parent / "performance.json").exists()
    with pytest.raises(FileExistsError):
        materialize_scale_campaign_binding(
            requirement_manifest_path=manifest_path,
            output_root=tmp_path / "campaign-output",
        )


def test_complete_declarative_paper_axes_expand_only_planned_cells(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    requirement = deepcopy(calibrated_lineages["a100_requirement"])
    requirement["paper_axes"] = {
        "context_cap_tokens": 40960,
        "token_projection_grid": [
            {"scale": 0.004, "role": "primary"},
            {"scale": 0.008, "role": "sensitivity"},
        ],
        "workload_shapes": [
            {"workflow_count": 15, "concurrency": 1},
            {"workflow_count": 15, "concurrency": 4},
            {"workflow_count": 30, "concurrency": 4},
            {"workflow_count": 60, "concurrency": 16},
        ],
        "phase_matrix": [
            {
                "phase": "static-test",
                "policy_count": 1,
                "seed_count": 1,
                "shape_count": 1,
                "run_count": 2,
            }
        ],
    }
    manifest_path = _write_json(
        tmp_path / "requirements.json", _manifest([requirement])
    )
    readiness = evaluate_scale_campaign_readiness(manifest_path)
    assert readiness["status"] == "READY_STATIC_ONLY"
    assert readiness["planned_cell_count"] == 64
    binding = build_scale_campaign_binding(manifest_path)
    assert binding["cell_count"] == 64
    assert len({cell["cell_id"] for cell in binding["cells"]}) == 64
    assert all(cell["state"] == "PLANNED_NOT_RUN" for cell in binding["cells"])
    assert {cell["axes"]["token_projection"]["scale"] for cell in binding["cells"]} == {
        0.004,
        0.008,
    }
    assert {cell["axes"]["workflow_shape"]["workflow_count"] for cell in binding["cells"]} == {
        15,
        30,
        60,
    }
    assert binding["authorization"]["replay_authorized"] is False
    assert binding["authorization"]["performance_result_authorized"] is False


def test_missing_lineage_returns_gap_and_creates_no_cells_or_output(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    requirements[1]["suite_binding"] = None
    manifest_path = _write_json(
        tmp_path / "requirements.json", _manifest(requirements)
    )
    readiness = evaluate_scale_campaign_readiness(manifest_path)
    assert readiness["status"] == "BLOCKED_MISSING_LINEAGES"
    assert readiness["planned_cell_count"] == 0
    assert readiness["cells"] == []
    assert readiness["gaps"] == [
        {
            "requirement_id": "v100-qwen3-8b-tp1-pp1-dp4",
            "code": "MISSING_SUITE_BINDING_REFERENCE",
            "message": (
                "an independently validated calibrated-suite binding "
                "has not been supplied"
            ),
        }
    ]
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(
        ScaleCampaignContractError,
        match="blocked by missing calibrated lineages",
    ):
        materialize_scale_campaign_binding(
            requirement_manifest_path=manifest_path,
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("field", "replacement", "pattern"),
    [
        ("name", "Qwen/Qwen3-32B", "model identity"),
        ("revision", "f" * 40, "model revision"),
    ],
)
def test_requirement_rejects_wrong_model_or_revision(
    tmp_path: Path,
    calibrated_lineages: Mapping[str, Any],
    field: str,
    replacement: str,
    pattern: str,
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    requirements[0]["model"][field] = replacement
    manifest_path = _write_json(
        tmp_path / f"wrong-{field}.json", _manifest(requirements)
    )
    with pytest.raises(ScaleCampaignContractError, match=pattern):
        build_scale_campaign_binding(manifest_path)


@pytest.mark.parametrize(
    ("field", "pattern"),
    [
        ("tp_size", "TP layout"),
        ("pp_size", "PP layout"),
        ("system_dp_size", "system-DP layout"),
    ],
)
def test_requirement_rejects_wrong_tp_pp_or_system_dp(
    tmp_path: Path,
    calibrated_lineages: Mapping[str, Any],
    field: str,
    pattern: str,
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    requirements[0]["layout"][field] += 1
    manifest_path = _write_json(
        tmp_path / f"wrong-{field}.json", _manifest(requirements)
    )
    with pytest.raises(ScaleCampaignContractError, match=pattern):
        build_scale_campaign_binding(manifest_path)


def test_requirement_rejects_cross_hardware_binding(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    requirements[0]["hardware_label"] = requirements[1]["hardware_label"]
    manifest_path = _write_json(
        tmp_path / "cross-hardware.json", _manifest(requirements)
    )
    with pytest.raises(ScaleCampaignContractError, match="hardware does not match"):
        build_scale_campaign_binding(manifest_path)


@pytest.mark.parametrize(
    ("section", "field", "pattern"),
    [
        ("physical_profile", "sha256", "physical profile identity"),
        ("calibration", "identity_sha256", "calibration identity"),
        (None, "lineage_identity_sha256", "lineage identity"),
    ],
)
def test_requirement_rejects_profile_calibration_or_lineage_mismatch(
    tmp_path: Path,
    calibrated_lineages: Mapping[str, Any],
    section: str | None,
    field: str,
    pattern: str,
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    target = requirements[0] if section is None else requirements[0][section]
    target[field] = "f" * 64
    manifest_path = _write_json(
        tmp_path / f"wrong-{field}.json", _manifest(requirements)
    )
    with pytest.raises(ScaleCampaignContractError, match=pattern):
        build_scale_campaign_binding(manifest_path)


def test_one_suite_binding_cannot_be_reused_across_requirements(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    requirements[1]["suite_binding"] = deepcopy(requirements[0]["suite_binding"])
    manifest_path = _write_json(
        tmp_path / "reused-binding.json", _manifest(requirements)
    )
    with pytest.raises(ScaleCampaignContractError, match="cannot be reused"):
        build_scale_campaign_binding(manifest_path)


def test_duplicate_requirement_id_is_rejected_before_any_binding_output(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    requirements[1]["requirement_id"] = requirements[0]["requirement_id"]
    manifest_path = _write_json(
        tmp_path / "duplicate-requirement.json", _manifest(requirements)
    )
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(ScaleCampaignContractError, match="duplicate requirement_id"):
        materialize_scale_campaign_binding(
            requirement_manifest_path=manifest_path,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_outer_suite_pass_tamper_is_revalidated_and_rejected(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    source_path = Path(requirements[0]["suite_binding"]["path"])
    tampered_path = tmp_path / "outer-pass-suite-binding.json"
    tampered = json.loads(source_path.read_text(encoding="utf-8"))
    tampered["status"] = "PASS"
    _write_json(tampered_path, tampered)
    requirements[0]["suite_binding"] = artifact_record(tampered_path)
    manifest_path = _write_json(
        tmp_path / "requirements.json", _manifest(requirements)
    )
    with pytest.raises(
        ScaleCampaignContractError,
        match="calibrated-suite revalidation failed.*not static-only",
    ):
        build_scale_campaign_binding(manifest_path)


def test_global_cardinality_limit_fails_before_output(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    requirements = _valid_requirements(calibrated_lineages)
    manifest_path = _write_json(
        tmp_path / "over-cardinality.json",
        _manifest(requirements, max_total_cells=35),
    )
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(ScaleCampaignContractError, match="cell cardinality"):
        materialize_scale_campaign_binding(
            requirement_manifest_path=manifest_path,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_duplicate_local_cell_is_rejected_by_global_constructor() -> None:
    local = {
        "cell_id": "local-1",
        "axes": {"seed": 1},
        "state": "PLANNED_NOT_RUN",
    }
    bound = [
        {
            "requirement": {
                "requirement_id": "lineage-a",
                "lineage_identity_sha256": "a" * 64,
            },
            "suite_binding": {"suite_binding_identity_sha256": "b" * 64},
            "suite_runner": {"cells": [local, deepcopy(local)]},
        }
    ]
    with pytest.raises(ScaleCampaignContractError, match="repeats a suite cell"):
        _global_cells(bound)


def test_campaign_outer_pass_tamper_is_rejected(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    manifest_path = _write_json(
        tmp_path / "requirements.json",
        _manifest(_valid_requirements(calibrated_lineages)),
    )
    binding = build_scale_campaign_binding(manifest_path)
    binding["status"] = "PASS"
    binding_path = _write_json(tmp_path / "tampered-campaign.json", binding)
    with pytest.raises(ScaleCampaignContractError, match="not static-only"):
        preflight_scale_campaign_runner(binding_path)


def test_cli_reports_missing_lineage_before_creating_output(
    tmp_path: Path, calibrated_lineages: Mapping[str, Any]
) -> None:
    requirements = [deepcopy(calibrated_lineages["a100_requirement"])]
    requirements[0]["suite_binding"] = None
    manifest_path = _write_json(
        tmp_path / "requirements.json", _manifest(requirements)
    )
    output_root = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--requirements",
            str(manifest_path),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    result = json.loads(completed.stdout)
    assert result["status"] == "BLOCKED_MISSING_LINEAGES"
    assert result["cells"] == []
    assert result["authorization"]["trace_loading_authorized"] is False
    assert result["authorization"]["replay_authorized"] is False
    assert result["authorization"]["performance_result_authorized"] is False
    assert not output_root.exists()
