#!/usr/bin/env python3
"""Fail-closed preflight for a larger-model Real-Weka SGLsim campaign.

This audit deliberately does not synthesize missing timing coefficients.  A
model/layout-specific profile must bind compute, all four independent AegisKV
paths, KV geometry, and the exact deployment layout before the simulator may
produce a performance result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


DESIGN_SEMANTICS = "source_owned_independent_hbm_dram_v2"
PROFILE_SEMANTICS = "source_owned_four_physical_paths_v1"
REQUIRED_PROFILE_METRICS = {
    "recompute_ms_per_token",
    "recompute_fixed_overhead_ms",
    "peer_hbm_protection_effective_bytes_per_ms",
    "direct_dram_protection_effective_bytes_per_ms",
    "peer_hbm_recovery_effective_bytes_per_ms",
    "dram_recovery_effective_bytes_per_ms",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _dtype_bytes(raw: object) -> int:
    dtype = str(raw or "").lower()
    if dtype in {"bfloat16", "bf16", "float16", "fp16", "half"}:
        return 2
    if dtype in {"float32", "fp32"}:
        return 4
    raise ValueError(f"unsupported KV dtype: {raw!r}")


def _model_contract(model_config: Mapping[str, Any]) -> dict[str, Any]:
    layers = _positive_int(model_config.get("num_hidden_layers"), name="layers")
    kv_heads = _positive_int(
        model_config.get("num_key_value_heads"), name="KV heads"
    )
    head_dim = _positive_int(model_config.get("head_dim"), name="head dim")
    maximum_context = _positive_int(
        model_config.get("max_position_embeddings"), name="model context"
    )
    dtype = str(model_config.get("torch_dtype") or "")
    bytes_per_value = _dtype_bytes(dtype)
    return {
        "model_type": str(model_config.get("model_type") or ""),
        "maximum_context_tokens": maximum_context,
        "num_hidden_layers": layers,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "kv_dtype": dtype,
        "kv_bytes_per_token_all_layers": (
            2 * layers * kv_heads * head_dim * bytes_per_value
        ),
    }


def _layout(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ValueError("layout must be an object")
    layout = {
        "tp_size": _positive_int(raw.get("tp_size"), name="tp_size"),
        "pp_size": _positive_int(raw.get("pp_size"), name="pp_size"),
        "dp_size": _positive_int(raw.get("dp_size"), name="dp_size"),
    }
    return layout


def _validate_profile(
    profile_path: Path,
    *,
    model_record: Mapping[str, Any],
    model_config_sha256: str,
    layout: Mapping[str, int],
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    profile = _object(profile_path)
    expected_model_revision = str(model_record["revision"])
    if not (
        profile.get("schema_version") == 2
        and profile.get("evidence_class") == "physical_measurement"
        and profile.get("status") == "passed"
        and profile.get("profile_semantics") == PROFILE_SEMANTICS
        and profile.get("model") == model_record.get("name")
        and profile.get("model_revision") == expected_model_revision
        and profile.get("model_config_sha256") == model_config_sha256
        and profile.get("layout") == dict(layout)
        and profile.get("compatibility_identity") == dict(compatibility)
    ):
        raise ValueError("model-specific profile identity mismatch")
    metrics = profile.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != REQUIRED_PROFILE_METRICS:
        raise ValueError("model-specific profile metric set mismatch")
    for name in REQUIRED_PROFILE_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(f"non-finite profile metric: {name}")
        if name == "recompute_fixed_overhead_ms":
            if value < 0.0:
                raise ValueError(f"negative profile metric: {name}")
        elif value <= 0.0:
            raise ValueError(f"non-positive profile metric: {name}")
    return profile


def audit(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    auditor_path = Path(__file__).resolve()
    config = _object(config_path)
    if config.get("design_semantics") != DESIGN_SEMANTICS:
        raise ValueError("model-scale audit does not use the current design")

    design_record = config.get("design")
    context_record = config.get("context_audit")
    model_record = config.get("model")
    profile_contract = config.get("model_specific_profile")
    if not all(
        isinstance(value, Mapping)
        for value in (
            design_record,
            context_record,
            model_record,
            profile_contract,
        )
    ):
        raise ValueError("model-scale config is incomplete")

    design_path = Path(str(design_record["path"])).resolve()
    context_path = Path(str(context_record["path"])).resolve()
    model_path = Path(str(model_record["config_path"])).resolve()
    for path in (design_path, context_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path, expected, label in (
        (design_path, design_record.get("sha256"), "design"),
        (context_path, context_record.get("sha256"), "context audit"),
        (model_path, model_record.get("config_sha256"), "model config"),
    ):
        if _sha256(path) != str(expected or ""):
            raise ValueError(f"{label} sha256 mismatch")

    context_audit = _object(context_path)
    if not (
        context_audit.get("status") == "PASS"
        and context_audit.get("design_semantics") == DESIGN_SEMANTICS
        and context_audit.get("performance_results_present") in {None, False}
        and context_audit.get("formal_ready") is False
        and context_audit.get("paper_ready") is False
        and context_audit.get("trace", {}).get("sha256")
        == context_record.get("trace_sha256")
    ):
        raise ValueError("context audit is not a current-design planning artifact")

    model = _model_contract(_object(model_path))
    expected_kv_bytes = _positive_int(
        model_record.get("kv_bytes_per_token_all_layers"),
        name="pinned KV bytes",
    )
    if model["kv_bytes_per_token_all_layers"] != expected_kv_bytes:
        raise ValueError("pinned KV bytes do not match the model config")
    if model["maximum_context_tokens"] != int(
        context_audit.get("model", {}).get("maximum_context_tokens", -1)
    ):
        raise ValueError(
            "context audit cannot be reused for a different model context limit"
        )

    layout = _layout(config.get("layout"))
    if model["num_hidden_layers"] % layout["pp_size"] != 0:
        raise ValueError("model layers are not evenly divisible by pp_size")
    expected_workers = (
        layout["tp_size"] * layout["pp_size"] * layout["dp_size"]
    )
    if expected_workers != _positive_int(
        config.get("expected_scheduler_processes"),
        name="expected scheduler processes",
    ):
        raise ValueError("scheduler process count does not match DP/PP/TP layout")

    compatibility = profile_contract.get("required_compatibility_identity")
    if not isinstance(compatibility, Mapping):
        raise ValueError("required compatibility identity is missing")
    if compatibility.get("layout") != layout:
        raise ValueError("profile compatibility layout mismatch")
    maximum_experiment_context = _positive_int(
        compatibility.get("max_model_len"), name="profile max_model_len"
    )
    if maximum_experiment_context > model["maximum_context_tokens"]:
        raise ValueError("profile context exceeds the model context limit")

    profile_path = Path(str(profile_contract["path"])).resolve()
    profile: dict[str, Any] | None = None
    if profile_path.is_file():
        profile = _validate_profile(
            profile_path,
            model_record=model_record,
            model_config_sha256=_sha256(model_path),
            layout=layout,
            compatibility=compatibility,
        )

    scale_rows = context_audit.get("scale_points")
    if not isinstance(scale_rows, list):
        raise ValueError("context audit has no scale points")
    by_label = {
        str(row.get("label")): row
        for row in scale_rows
        if isinstance(row, Mapping)
    }
    requested = config.get("scale_points")
    if not isinstance(requested, list) or not requested:
        raise ValueError("at least one model-scale point is required")
    seen_labels: set[str] = set()
    selected: list[dict[str, Any]] = []
    primary_ready = False
    for raw in requested:
        if not isinstance(raw, Mapping):
            raise ValueError("model-scale point must be an object")
        label = str(raw.get("label") or "")
        role = str(raw.get("role") or "")
        if not label or label in seen_labels or label not in by_label:
            raise ValueError(f"invalid or duplicate scale label: {label}")
        seen_labels.add(label)
        context_row = by_label[label]
        maximum_context = int(context_row.get("context_tokens", {}).get("max", -1))
        above_model = int(context_row.get("requests_above_model_context", -1))
        if above_model > 0:
            readiness = "INVALID_MODEL_CONTEXT"
        elif profile is None:
            readiness = "BLOCKED_MODEL_SPECIFIC_PROFILE"
        elif maximum_context > maximum_experiment_context:
            readiness = "MODEL_VALID_CONTEXT_EXTRAPOLATION_ONLY"
        else:
            readiness = "READY_FOR_CURRENT_DESIGN_SIMULATION"
        primary_ready = primary_ready or (
            role == "primary"
            and readiness == "READY_FOR_CURRENT_DESIGN_SIMULATION"
        )
        selected.append(
            {
                "label": label,
                "role": role,
                "factor": float(context_row["factor"]),
                "request_count": int(context_row["request_count"]),
                "input_tokens": int(context_row["input_tokens"]),
                "output_tokens": int(context_row["output_tokens"]),
                "maximum_context_tokens": maximum_context,
                "readiness": readiness,
                "performance_results_present": False,
            }
        )
    if sum(row["role"] == "primary" for row in selected) != 1:
        raise ValueError("exactly one preregistered primary scale is required")

    matrix_readiness = (
        "READY_FOR_CURRENT_DESIGN_SIMULATION"
        if primary_ready
        else "BLOCKED_MODEL_SPECIFIC_PROFILE"
    )
    return {
        "schema_version": 1,
        "evidence_class": "real_weka_larger_model_preflight_not_performance_result",
        "status": "PASS",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "design_semantics": DESIGN_SEMANTICS,
        "design": {
            "path": str(design_path),
            "sha256": _sha256(design_path),
        },
        "auditor": {"path": str(auditor_path), "sha256": _sha256(auditor_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "context_audit": {
            "path": str(context_path),
            "sha256": _sha256(context_path),
            "trace_sha256": context_record["trace_sha256"],
        },
        "model": {
            "name": str(model_record["name"]),
            "revision": str(model_record["revision"]),
            "config_path": str(model_path),
            "config_sha256": _sha256(model_path),
            **model,
        },
        "layout": {
            **layout,
            "scheduler_processes": expected_workers,
            "layers_per_pipeline_stage": (
                model["num_hidden_layers"] // layout["pp_size"]
            ),
            "kv_bytes_per_token_per_pipeline_stage": (
                model["kv_bytes_per_token_all_layers"] // layout["pp_size"]
            ),
        },
        "model_specific_profile": {
            "path": str(profile_path),
            "exists": profile is not None,
            "verified": profile is not None,
            "required_schema_version": 2,
            "required_profile_semantics": PROFILE_SEMANTICS,
            "required_metrics": sorted(REQUIRED_PROFILE_METRICS),
            "required_compatibility_identity": dict(compatibility),
        },
        "scale_points": selected,
        "matrix_readiness": matrix_readiness,
        "simulation_execution_authorized": primary_ready,
        "performance_results_present": False,
        "formal_ready": False,
        "paper_ready": False,
        "evidence_boundary": (
            "Model geometry, topology, context feasibility, and profile identity "
            "only. A missing model-specific profile blocks simulation; this "
            "artifact is not a performance, physical, formal, or paper result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
