#!/usr/bin/env python3
"""Audit whether a real-Weka token projection is executable by one model.

The materialized trace is compact JSONL with request records whose ``in`` and
``out`` fields are adjacent.  A byte-regex scan avoids materializing very large
``hash_ids`` arrays while still failing closed on the pinned request/token
totals and trace hash.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
from typing import Any, Iterable


DESIGN_SEMANTICS = "source_owned_independent_hbm_dram_v2"
REQUEST_PATTERN = re.compile(br'"in":([0-9]+),"out":([0-9]+)')


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


def _quantile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sample")
    index = min(int((len(values) - 1) * fraction), len(values) - 1)
    return values[index]


def _distribution(values: Iterable[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    return {
        "p50": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "p999": _quantile(ordered, 0.999),
        "max": ordered[-1],
    }


def _scan_requests(path: Path) -> tuple[list[int], list[int]]:
    prompts: list[int] = []
    outputs: list[int] = []
    with path.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as mapped:
        for match in REQUEST_PATTERN.finditer(mapped):
            prompts.append(int(match.group(1)))
            outputs.append(int(match.group(2)))
    if not prompts:
        raise ValueError("trace contains no compact in/out request records")
    return prompts, outputs


def _dtype_bytes(raw: object) -> int:
    dtype = str(raw or "").lower()
    if dtype in {"bfloat16", "bf16", "float16", "fp16"}:
        return 2
    if dtype in {"float32", "fp32"}:
        return 4
    raise ValueError(f"unsupported KV dtype: {raw!r}")


def _model_contract(config: dict[str, Any]) -> dict[str, Any]:
    required = (
        "max_position_embeddings",
        "num_hidden_layers",
        "num_key_value_heads",
        "head_dim",
    )
    if any(
        isinstance(config.get(name), bool) or int(config.get(name, 0)) <= 0
        for name in required
    ):
        raise ValueError("model config lacks positive KV/context dimensions")
    dtype_bytes = _dtype_bytes(config.get("torch_dtype"))
    kv_bytes = (
        2
        * int(config["num_hidden_layers"])
        * int(config["num_key_value_heads"])
        * int(config["head_dim"])
        * dtype_bytes
    )
    return {
        "model_type": str(config.get("model_type") or ""),
        "maximum_context_tokens": int(config["max_position_embeddings"]),
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_key_value_heads": int(config["num_key_value_heads"]),
        "head_dim": int(config["head_dim"]),
        "kv_dtype": str(config.get("torch_dtype") or ""),
        "kv_bytes_per_token_all_layers": kv_bytes,
    }


def _scaled_tokens(value: int, factor: float, *, minimum: int) -> int:
    return max(int(math.ceil(int(value) * float(factor))), int(minimum))


def audit(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    auditor_path = Path(__file__).resolve()
    config = _object(config_path)
    if config.get("design_semantics") != DESIGN_SEMANTICS:
        raise ValueError("context audit does not use the current AegisKV design")
    trace_record = config.get("trace")
    model_record = config.get("model")
    calibration = config.get("physical_calibration_contract")
    if not all(
        isinstance(value, dict)
        for value in (trace_record, model_record, calibration)
    ):
        raise ValueError("context audit config is incomplete")
    trace_path = Path(str(trace_record["path"])).resolve()
    model_path = Path(str(model_record["config_path"])).resolve()
    for path in (trace_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(trace_path) != str(trace_record.get("sha256") or ""):
        raise ValueError("trace sha256 mismatch")
    if _sha256(model_path) != str(model_record.get("config_sha256") or ""):
        raise ValueError("model config sha256 mismatch")
    model = _model_contract(_object(model_path))
    if model["kv_bytes_per_token_all_layers"] != int(
        model_record["kv_bytes_per_token_all_layers"]
    ):
        raise ValueError("pinned KV bytes do not match the model config")
    prompts, outputs = _scan_requests(trace_path)
    expected = trace_record.get("expected_full_profile")
    observed = {
        "request_count": len(prompts),
        "input_tokens": sum(prompts),
        "output_tokens": sum(outputs),
    }
    if expected != observed:
        raise ValueError(f"trace request/token profile mismatch: {observed}")
    contexts = [prompt + output for prompt, output in zip(prompts, outputs)]
    maximum_model = int(model["maximum_context_tokens"])
    maximum_calibrated = int(calibration["maximum_calibrated_context_tokens"])
    if maximum_calibrated <= 0 or maximum_calibrated > maximum_model:
        raise ValueError("physical context calibration range is invalid")

    scale_results: list[dict[str, Any]] = []
    raw_scales = config.get("scale_points")
    if not isinstance(raw_scales, list) or not raw_scales:
        raise ValueError("at least one context scale point is required")
    seen_factors: set[float] = set()
    for raw in raw_scales:
        if not isinstance(raw, dict):
            raise ValueError("context scale point must be an object")
        factor = float(raw["factor"])
        if not math.isfinite(factor) or not 0.0 < factor <= 1.0:
            raise ValueError("context scale factor must be in (0, 1]")
        if factor in seen_factors:
            raise ValueError("context scale factors must be unique")
        seen_factors.add(factor)
        scaled_prompts = [
            _scaled_tokens(value, factor, minimum=1) for value in prompts
        ]
        scaled_outputs = [
            _scaled_tokens(value, factor, minimum=0) for value in outputs
        ]
        scaled_contexts = [
            prompt + output
            for prompt, output in zip(scaled_prompts, scaled_outputs)
        ]
        above_model = sum(value > maximum_model for value in scaled_contexts)
        above_calibrated = sum(
            value > maximum_calibrated for value in scaled_contexts
        )
        if above_model:
            readiness = "INVALID_MODEL_CONTEXT"
        elif above_calibrated:
            readiness = "MODEL_VALID_CONTEXT_EXTRAPOLATION_ONLY"
        else:
            readiness = "ELIGIBLE_AFTER_VALID_PHYSICAL_PROFILE"
        scale_results.append(
            {
                "label": str(raw["label"]),
                "factor": factor,
                "role": str(raw["role"]),
                "readiness": readiness,
                "request_count": len(scaled_contexts),
                "input_tokens": sum(scaled_prompts),
                "output_tokens": sum(scaled_outputs),
                "context_tokens": _distribution(scaled_contexts),
                "requests_above_model_context": above_model,
                "fraction_above_model_context": above_model / len(contexts),
                "requests_above_physical_calibration_context": above_calibrated,
                "fraction_above_physical_calibration_context": above_calibrated
                / len(contexts),
                "performance_results_present": False,
            }
        )

    return {
        "schema_version": 1,
        "evidence_class": "real_weka_context_projection_audit_not_performance_result",
        "status": "PASS",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "design_semantics": DESIGN_SEMANTICS,
        "auditor": {
            "path": str(auditor_path),
            "sha256": _sha256(auditor_path),
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "trace": {
            "path": str(trace_path),
            "sha256": _sha256(trace_path),
            **observed,
            "prompt_tokens": _distribution(prompts),
            "context_tokens": _distribution(contexts),
        },
        "model": {
            "revision": str(model_record["revision"]),
            "config_path": str(model_path),
            "config_sha256": _sha256(model_path),
            **model,
        },
        "physical_calibration_contract": dict(calibration),
        "unscaled_model_context_violation": {
            "request_count": sum(value > maximum_model for value in contexts),
            "fraction": sum(value > maximum_model for value in contexts)
            / len(contexts),
        },
        "scale_points": scale_results,
        "selection_policy": (
            "The scale grid and roles are fixed before any replacement V100 "
            "performance run. All scale outcomes must be retained; the primary "
            "candidate cannot be changed after observing treatment results."
        ),
        "formal_ready": False,
        "paper_ready": False,
        "evidence_boundary": (
            "Token/model/calibration-range feasibility only. Scaling preserves "
            "workflow topology but changes token volume, so every scaled result "
            "must be labeled as a projection, not an unmodified Weka replay."
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
