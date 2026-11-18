from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from AegisKV.experiments.real_weka_context_scale.audit_context_scale import audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    return path


def _fixture(tmp_path: Path) -> Path:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"requests":[{"in":100,"out":10},{"in":800,"out":100}]}\n'
        '{"requests":[{"in":1800,"out":200}]}\n',
        encoding="utf-8",
    )
    model = _write(
        tmp_path / "model.json",
        {
            "model_type": "unit",
            "max_position_embeddings": 1024,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "torch_dtype": "bfloat16",
        },
    )
    return _write(
        tmp_path / "config.json",
        {
            "design_semantics": "source_owned_independent_hbm_dram_v2",
            "trace": {
                "path": str(trace),
                "sha256": _sha256(trace),
                "expected_full_profile": {
                    "request_count": 3,
                    "input_tokens": 2700,
                    "output_tokens": 310,
                },
            },
            "model": {
                "revision": "unit",
                "config_path": str(model),
                "config_sha256": _sha256(model),
                "kv_bytes_per_token_all_layers": 64,
            },
            "physical_calibration_contract": {
                "maximum_calibrated_context_tokens": 512
            },
            "scale_points": [
                {"label": "raw", "factor": 1.0, "role": "diagnostic"},
                {"label": "half", "factor": 0.5, "role": "sensitivity"},
                {"label": "quarter", "factor": 0.25, "role": "primary"},
            ],
        },
    )


def test_context_audit_separates_model_invalid_and_calibration_valid(
    tmp_path: Path,
) -> None:
    result = audit(_fixture(tmp_path))
    by_label = {row["label"]: row for row in result["scale_points"]}
    assert result["status"] == "PASS"
    assert result["unscaled_model_context_violation"]["request_count"] == 1
    assert by_label["raw"]["readiness"] == "INVALID_MODEL_CONTEXT"
    assert by_label["half"]["readiness"] == (
        "MODEL_VALID_CONTEXT_EXTRAPOLATION_ONLY"
    )
    assert by_label["quarter"]["readiness"] == (
        "ELIGIBLE_AFTER_VALID_PHYSICAL_PROFILE"
    )
    assert by_label["quarter"]["performance_results_present"] is False


def test_context_audit_rejects_trace_drift(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["trace"]["sha256"] = "0" * 64
    _write(config_path, config)
    with pytest.raises(ValueError, match="trace sha256 mismatch"):
        audit(config_path)


def test_context_audit_rejects_incorrect_kv_footprint(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["model"]["kv_bytes_per_token_all_layers"] = 63
    _write(config_path, config)
    with pytest.raises(ValueError, match="pinned KV bytes"):
        audit(config_path)
