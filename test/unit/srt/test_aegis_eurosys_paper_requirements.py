from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    artifact_record,
)
from AegisKV.experiments.real_weka_full_design.eurosys_paper_readiness import (
    PAPER_BLOCKED_AUTHORIZATION,
    EuroSysPaperReadinessError,
    build_eurosys_paper_readiness,
    materialize_eurosys_paper_readiness,
    validate_eurosys_paper_readiness,
)
from AegisKV.experiments.real_weka_full_design.scale_campaign_contract import (
    REQUIREMENTS_EVIDENCE_CLASS,
    ScaleCampaignContractError,
    evaluate_scale_campaign_readiness,
    load_scale_campaign_requirements,
)
from test.unit.srt.test_aegis_scale_campaign_contract import calibrated_lineages


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "AegisKV/experiments/real_weka_full_design"
REQUIREMENTS = PAPER_ROOT / "eurosys_paper_requirements.json"
READINESS = PAPER_ROOT / "eurosys-paper-readiness.json"
EVALUATION = Path(
    "/home/comp/zhanzhao/Mooncake/aegiskv/paper/sections/05-evaluation.tex"
)
SCRIPT = PAPER_ROOT / "audit_eurosys_paper_requirements.py"
EVALUATION_SHA256 = (
    "232dd00ff775801166fb40f2e7a17617cc654efe0273c2d44dbdabc4bc68f7d8"
)


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_paper_manifest_matches_current_evaluation_matrix_exactly() -> None:
    assert hashlib.sha256(EVALUATION.read_bytes()).hexdigest() == EVALUATION_SHA256
    requirements = load_scale_campaign_requirements(REQUIREMENTS)
    assert len(requirements) == 3
    by_id = {value["requirement_id"]: value for value in requirements}

    a100 = by_id["a100-qwen3-8b-fp16-tp1-pp1-dp2"]
    assert a100["hardware_label"] == "A100-1node-TP1-PP1-systemDP2"
    assert a100["model"]["scale_label"] == "Qwen3-8B"
    assert a100["model"]["dtype"] == "fp16"
    assert a100["layout"] == {
        "tp_size": 1,
        "pp_size": 1,
        "system_dp_size": 2,
        "worker_count": 2,
    }

    v100 = by_id["v100s-qwen3-8b-fp16-tp1-pp1-dp4"]
    assert v100["hardware_label"] == "V100S-1node-TP1-PP1-systemDP4"
    assert v100["model"]["scale_label"] == "Qwen3-8B"
    assert v100["layout"] == {
        "tp_size": 1,
        "pp_size": 1,
        "system_dp_size": 4,
        "worker_count": 4,
    }

    qwen32 = by_id["qwen3-32b-bf16-tp1-pp4-dp4"]
    assert qwen32["hardware_label"] is None
    assert qwen32["model"]["scale_label"] == "Qwen3-32B"
    assert qwen32["model"]["dtype"] == "bf16"
    assert qwen32["layout"] == {
        "tp_size": 1,
        "pp_size": 4,
        "system_dp_size": 4,
        "worker_count": 16,
    }
    assert qwen32["paper_axes"]["context_cap_tokens"] == 32768

    expected_grid = [0.004, 0.008, 0.016, 0.032]
    expected_shapes = [
        {"workflow_count": 15, "concurrency": 4},
        {"workflow_count": 30, "concurrency": 4},
        {"workflow_count": 60, "concurrency": 1},
        {"workflow_count": 60, "concurrency": 4},
        {"workflow_count": 60, "concurrency": 8},
    ]
    expected_phases = [
        {
            "phase": "pilot",
            "policy_count": 2,
            "seed_count": 1,
            "shape_count": 1,
            "run_count": 4,
        },
        {
            "phase": "central",
            "policy_count": 7,
            "seed_count": 5,
            "shape_count": 1,
            "run_count": 70,
        },
        {
            "phase": "scale",
            "policy_count": 2,
            "seed_count": 3,
            "shape_count": 5,
            "run_count": 60,
        },
        {
            "phase": "ablations",
            "policy_count": 5,
            "seed_count": 5,
            "shape_count": 1,
            "run_count": 50,
        },
    ]
    for requirement in requirements:
        axes = requirement["paper_axes"]
        assert [point["scale"] for point in axes["token_projection_grid"]] == (
            expected_grid
        )
        assert axes["workload_shapes"] == expected_shapes
        assert axes["phase_matrix"] == expected_phases


def test_current_paper_readiness_is_machine_readable_and_fully_blocked() -> None:
    readiness = build_eurosys_paper_readiness(
        requirement_manifest_path=REQUIREMENTS,
        evaluation_source_path=EVALUATION,
    )
    assert readiness["status"] == "BLOCKED_PAPER_REQUIREMENTS"
    assert readiness["paper_global_checks"] == {
        "token_projection_grid_matches": True,
        "workload_shapes_match": True,
        "phase_matrix_matches": True,
    }
    assert readiness["bound_requirement_ids"] == []
    assert readiness["missing_requirement_ids"] == [
        "a100-qwen3-8b-fp16-tp1-pp1-dp2",
        "qwen3-32b-bf16-tp1-pp4-dp4",
        "v100s-qwen3-8b-fp16-tp1-pp1-dp4",
    ]
    assert readiness["planned_cell_count"] == 0
    assert readiness["cells"] == []
    assert readiness["authorization"] == PAPER_BLOCKED_AUTHORIZATION
    assert readiness["result_state"] == {
        "execution_status": "NOT_RUN",
        "performance_results_present": False,
        "result_artifacts": [],
    }
    assert all(
        readiness["authorization"][field] is False
        for field in (
            "campaign_binding_materialization_authorized",
            "runnable_cells_authorized",
            "trace_loading_authorized",
            "model_fitting_authorized",
            "replay_authorized",
            "result_output_authorized",
            "performance_result_authorized",
            "performance_claim_authorized",
            "formal_ready",
            "paper_ready",
        )
    )
    alignments = {
        value["requirement_id"]: value
        for value in readiness["requirement_alignment"]
    }
    assert alignments["v100s-qwen3-8b-fp16-tp1-pp1-dp4"][
        "declared_in_evaluation_source"
    ] is True
    assert alignments["a100-qwen3-8b-fp16-tp1-pp1-dp2"][
        "declared_in_evaluation_source"
    ] is True
    assert alignments["qwen3-32b-bf16-tp1-pp4-dp4"]["checks"][
        "scheduler_process_count_declared"
    ] is True
    assert alignments["qwen3-32b-bf16-tp1-pp4-dp4"]["checks"][
        "hardware_declared"
    ] is False
    assert all(
        alignments["a100-qwen3-8b-fp16-tp1-pp1-dp2"]["checks"].values()
    )


def test_projection_grid_drift_is_retained_as_paper_gap(tmp_path: Path) -> None:
    manifest = json.loads(REQUIREMENTS.read_text(encoding="utf-8"))
    for requirement in manifest["requirements"]:
        requirement["paper_axes"]["token_projection_grid"][-1]["scale"] = 0.064
    path = _write_json(tmp_path / "requirements.json", manifest)
    readiness = build_eurosys_paper_readiness(
        requirement_manifest_path=path, evaluation_source_path=EVALUATION
    )
    assert readiness["paper_global_checks"]["token_projection_grid_matches"] is False
    assert any(
        gap["code"] == "PAPER_TOKEN_PROJECTION_GRID_MATCHES_MISMATCH"
        for gap in readiness["gaps"]
    )
    assert readiness["cells"] == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hardware", "hardware does not match"),
        ("model", "model identity does not match"),
        ("config", "model identity does not match"),
    ],
)
def test_a100_binding_cannot_be_reused_across_paper_hardware_or_model(
    tmp_path: Path,
    calibrated_lineages: Mapping[str, Any],
    mutation: str,
    message: str,
) -> None:
    requirement = deepcopy(calibrated_lineages["a100_requirement"])
    requirement["requirement_id"] = f"cross-lineage-{mutation}"
    if mutation == "hardware":
        requirement["hardware_label"] = "V100S-1node-TP1-PP1-systemDP4"
    elif mutation == "model":
        requirement["model"]["name"] = "Qwen/Qwen3-32B"
        requirement["model"]["scale_label"] = "Qwen3-32B"
        requirement["model"]["dtype"] = "bf16"
    else:
        requirement["model"]["config_sha256"] = "d" * 64
    manifest = {
        "schema_version": 1,
        "evidence_class": REQUIREMENTS_EVIDENCE_CLASS,
        "campaign_label": "cross-lineage-negative",
        "limits": {"max_requirements": 1, "max_total_cells": 1000},
        "requirements": [requirement],
    }
    path = _write_json(tmp_path / "requirements.json", manifest)
    with pytest.raises(ScaleCampaignContractError, match=message):
        evaluate_scale_campaign_readiness(path)


def test_checked_in_readiness_is_reproducible_and_cli_stays_static(
    tmp_path: Path,
) -> None:
    checked = validate_eurosys_paper_readiness(READINESS)
    assert checked["status"] == "BLOCKED_PAPER_REQUIREMENTS"
    assert checked["cells"] == []

    output = tmp_path / "paper-readiness.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--requirements",
            str(REQUIREMENTS),
            "--evaluation-source",
            str(EVALUATION),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "BLOCKED_PAPER_REQUIREMENTS"
    assert payload["planned_cell_count"] == 0
    assert payload["trace_loaded"] is False
    assert payload["model_fitted"] is False
    assert payload["replay_run"] is False
    assert payload["result_written"] is False
    assert payload["formal_ready"] is False
    assert payload["paper_ready"] is False
    assert payload["readiness_artifact"] == artifact_record(output)
    assert validate_eurosys_paper_readiness(output)["cells"] == []
    assert not (tmp_path / "replay.json").exists()
    assert not (tmp_path / "performance.json").exists()
    with pytest.raises(FileExistsError):
        materialize_eurosys_paper_readiness(
            requirement_manifest_path=REQUIREMENTS,
            evaluation_source_path=EVALUATION,
            output_path=output,
        )


def test_paper_readiness_rejects_tampered_outer_artifact(tmp_path: Path) -> None:
    path = tmp_path / "paper-readiness.json"
    materialize_eurosys_paper_readiness(
        requirement_manifest_path=REQUIREMENTS,
        evaluation_source_path=EVALUATION,
        output_path=path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "READY_STATIC_ONLY"
    _write_json(tmp_path / "tampered.json", payload)
    with pytest.raises(
        EuroSysPaperReadinessError, match="not reproducible"
    ):
        validate_eurosys_paper_readiness(tmp_path / "tampered.json")
