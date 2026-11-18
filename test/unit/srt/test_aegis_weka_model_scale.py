from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from AegisKV.experiments.real_weka_model_scale.audit_model_scale import audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    return path


def _fixture(tmp_path: Path, *, with_profile: bool = False) -> Path:
    design = tmp_path / "current_design.md"
    design.write_text("source-owned independent HBM and DRAM\n")
    model = _write(
        tmp_path / "model.json",
        {
            "model_type": "qwen3",
            "max_position_embeddings": 40960,
            "num_hidden_layers": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "torch_dtype": "bfloat16",
        },
    )
    context = _write(
        tmp_path / "context.json",
        {
            "status": "PASS",
            "design_semantics": "source_owned_independent_hbm_dram_v2",
            "formal_ready": False,
            "paper_ready": False,
            "trace": {"sha256": "a" * 64},
            "model": {"maximum_context_tokens": 40960},
            "scale_points": [
                {
                    "label": "primary",
                    "factor": 0.004,
                    "request_count": 3,
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "context_tokens": {"max": 3987},
                    "requests_above_model_context": 0,
                },
                {
                    "label": "sensitivity",
                    "factor": 0.032,
                    "request_count": 3,
                    "input_tokens": 800,
                    "output_tokens": 80,
                    "context_tokens": {"max": 31892},
                    "requests_above_model_context": 0,
                },
            ],
        },
    )
    layout = {"tp_size": 1, "pp_size": 4, "dp_size": 4}
    compatibility = {
        "serving_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "kv_abi": "qwen3-32b-bf16-block16-tp1-pp4-v1",
        "block_size": 16,
        "max_model_len": 32768,
        "logical_chunk_tokens": {"prefix": 4096, "decode": 256},
        "layout": layout,
    }
    profile = tmp_path / "profiles" / "profile.json"
    if with_profile:
        _write(
            profile,
            {
                "schema_version": 2,
                "evidence_class": "physical_measurement",
                "status": "passed",
                "profile_semantics": "source_owned_four_physical_paths_v1",
                "model": "Qwen/Qwen3-32B",
                "model_revision": "Qwen/Qwen3-32B@unit",
                "model_config_sha256": _sha256(model),
                "layout": layout,
                "compatibility_identity": compatibility,
                "metrics": {
                    "recompute_ms_per_token": 1.0,
                    "recompute_fixed_overhead_ms": 0.0,
                    "peer_hbm_protection_effective_bytes_per_ms": 2.0,
                    "direct_dram_protection_effective_bytes_per_ms": 3.0,
                    "peer_hbm_recovery_effective_bytes_per_ms": 4.0,
                    "dram_recovery_effective_bytes_per_ms": 5.0,
                },
            },
        )
    return _write(
        tmp_path / "config.json",
        {
            "design_semantics": "source_owned_independent_hbm_dram_v2",
            "design": {"path": str(design), "sha256": _sha256(design)},
            "context_audit": {
                "path": str(context),
                "sha256": _sha256(context),
                "trace_sha256": "a" * 64,
            },
            "model": {
                "name": "Qwen/Qwen3-32B",
                "revision": "Qwen/Qwen3-32B@unit",
                "config_path": str(model),
                "config_sha256": _sha256(model),
                "kv_bytes_per_token_all_layers": 262144,
            },
            "layout": layout,
            "expected_scheduler_processes": 16,
            "model_specific_profile": {
                "path": str(profile),
                "required_compatibility_identity": compatibility,
            },
            "scale_points": [
                {"label": "primary", "role": "primary"},
                {"label": "sensitivity", "role": "sensitivity"},
            ],
        },
    )


def test_qwen3_32b_preflight_blocks_without_own_profile(tmp_path: Path) -> None:
    result = audit(_fixture(tmp_path))
    assert result["status"] == "PASS"
    assert result["matrix_readiness"] == "BLOCKED_MODEL_SPECIFIC_PROFILE"
    assert result["simulation_execution_authorized"] is False
    assert result["model"]["kv_bytes_per_token_all_layers"] == 262144
    assert result["layout"]["scheduler_processes"] == 16
    assert result["layout"]["layers_per_pipeline_stage"] == 16
    assert result["layout"]["kv_bytes_per_token_per_pipeline_stage"] == 65536
    assert {
        row["readiness"] for row in result["scale_points"]
    } == {"BLOCKED_MODEL_SPECIFIC_PROFILE"}


def test_qwen3_32b_preflight_authorizes_only_matching_profile(tmp_path: Path) -> None:
    result = audit(_fixture(tmp_path, with_profile=True))
    assert result["model_specific_profile"]["verified"] is True
    assert result["matrix_readiness"] == "READY_FOR_CURRENT_DESIGN_SIMULATION"
    assert result["simulation_execution_authorized"] is True
    assert all(
        row["readiness"] == "READY_FOR_CURRENT_DESIGN_SIMULATION"
        for row in result["scale_points"]
    )
    assert result["performance_results_present"] is False


def test_qwen3_32b_preflight_rejects_model_config_drift(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["model"]["config_sha256"] = "0" * 64
    _write(config_path, config)
    with pytest.raises(ValueError, match="model config sha256 mismatch"):
        audit(config_path)


def test_qwen3_32b_preflight_rejects_cross_model_profile(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path, with_profile=True)
    config = json.loads(config_path.read_text())
    profile_path = Path(config["model_specific_profile"]["path"])
    profile = json.loads(profile_path.read_text())
    profile["model"] = "Qwen/Qwen3-8B"
    _write(profile_path, profile)
    with pytest.raises(ValueError, match="profile identity mismatch"):
        audit(config_path)
