#!/usr/bin/env python3
"""One-pass Weka preprocessing for the large-scale simulation campaign.

The 1.8 GiB source is read exactly once.  The output retains one compact row
per LLM request and aggregate prefix-block reuse counts; it intentionally does
not duplicate the variable-length ``hash_ids`` arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.agentic_workflows import weka_full_trace as wft
from sglsim.srt.simulation.aegis_gbdt import AegisGBDTConfig


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"
DEFAULT_OUTPUT = ROOT / "cache" / "weka_compact_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _raw_requests_with_ids(raw: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    workflow_id = str(raw["id"])
    short = workflow_id[:8]
    output: list[tuple[str, dict[str, Any]]] = []
    # This deliberately mirrors parse_weka_workflow so the trace-grounded
    # Algorithm-1 tickets and the raw hash_ids share the same request identity.
    for stage_id, (kind, items) in enumerate(
        wft._causal_stages(raw["requests"]), start=1
    ):
        if kind == "main":
            for index, row in enumerate(items):
                output.append((f"{short}-p{stage_id}-m{index}", row))
        else:
            for branch_index, group in enumerate(items):
                for turn_index, row in enumerate(group.get("requests") or ()):
                    output.append(
                        (
                            f"{short}-p{stage_id}-s{branch_index}-t{turn_index}",
                            row,
                        )
                    )
    return output


def _reuse_by_request(
    rows: list[tuple[str, dict[str, Any]]], *, block_size: int
) -> dict[str, dict[str, int]]:
    indexed = [(index, rid, row) for index, (rid, row) in enumerate(rows)]
    indexed.sort(key=lambda item: (float(item[2].get("t", 0.0)), item[0]))
    seen: set[int] = set()
    output: dict[str, dict[str, int]] = {}
    cursor = 0
    while cursor < len(indexed):
        arrival = float(indexed[cursor][2].get("t", 0.0))
        end = cursor + 1
        while end < len(indexed) and math.isclose(
            float(indexed[end][2].get("t", 0.0)),
            arrival,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            end += 1
        for _index, rid, row in indexed[cursor:end]:
            hashes = tuple(int(value) for value in row.get("hash_ids") or ())
            prefix_reuse_blocks = 0
            for value in hashes:
                if value not in seen:
                    break
                prefix_reuse_blocks += 1
            prompt_tokens = max(int(row.get("in", 0)), 0)
            output[rid] = {
                "hash_block_count": len(hashes),
                "prefix_reuse_block_count": prefix_reuse_blocks,
                "input_reuse_tokens": min(
                    prompt_tokens, prefix_reuse_blocks * int(block_size)
                ),
            }
        for _index, _rid, row in indexed[cursor:end]:
            seen.update(int(value) for value in row.get("hash_ids") or ())
        cursor = end
    return output


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    positive = actual > 1e-9
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "count": int(len(actual)),
        "mae_ms": float(np.mean(np.abs(error))) if len(actual) else 0.0,
        "mape": (
            float(np.mean(np.abs(error[positive]) / actual[positive]))
            if np.any(positive)
            else 0.0
        ),
        "r2": (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0.0
            else 0.0
        ),
    }


def prepare(protocol_path: Path, output_dir: Path, *, force: bool = False) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = Path(protocol["source_trace"]["path"])
    if not source.is_absolute():
        source = Path.cwd() / source
    source = source.resolve()
    observed_sha = _sha256(source)
    if observed_sha != protocol["source_trace"]["sha256"]:
        raise ValueError("Weka source SHA-256 does not match the frozen protocol")
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"compact trace already exists: {output_dir}; use --force only before campaign execution"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    workflows: list[wft.WekaWorkflow] = []
    raw_by_rid: dict[str, dict[str, int]] = {}
    workflow_reuse: list[dict[str, Any]] = []
    block_sizes: set[int] = set()
    hash_scopes: set[str] = set()
    with source.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = orjson.loads(line)
            workflow = wft.parse_weka_workflow(raw)
            workflows.append(workflow)
            rows = _raw_requests_with_ids(raw)
            parsed_ids = {request.rid for request in workflow.requests}
            raw_ids = {rid for rid, _row in rows}
            if parsed_ids != raw_ids:
                raise AssertionError(
                    f"raw/parser request identity mismatch in workflow {workflow.workflow_id}"
                )
            block_size = int(raw.get("block_size", 0))
            if block_size <= 0:
                raise ValueError(f"workflow {workflow.workflow_id} lacks block_size")
            block_sizes.add(block_size)
            hash_scopes.add(str(raw.get("hash_id_scope") or ""))
            reuse = _reuse_by_request(rows, block_size=block_size)
            raw_by_rid.update(reuse)
            prompt = sum(max(int(row.get("in", 0)), 0) for _rid, row in rows)
            reused = sum(value["input_reuse_tokens"] for value in reuse.values())
            workflow_reuse.append(
                {
                    "workflow_id": workflow.workflow_id,
                    "request_count": len(rows),
                    "input_tokens": prompt,
                    "input_reuse_tokens": reused,
                    "input_reuse_fraction": reused / prompt if prompt else 0.0,
                    "requests_with_reuse": sum(
                        value["input_reuse_tokens"] > 0 for value in reuse.values()
                    ),
                }
            )
            if line_number % 50 == 0:
                print(f"parsed {line_number} workflows", flush=True)

    if len(workflows) != int(protocol["source_trace"]["workflow_count"]):
        raise AssertionError("workflow count differs from frozen protocol")
    requests = [request for workflow in workflows for request in workflow.requests]
    if len(requests) != int(protocol["source_trace"]["request_count"]):
        raise AssertionError("request count differs from frozen protocol")
    if len(raw_by_rid) != len(requests):
        raise AssertionError("raw reuse rows do not cover every parsed request")
    if block_sizes != {64} or hash_scopes != {"local"}:
        raise AssertionError(
            f"unexpected reuse contract: block_sizes={block_sizes}, scopes={hash_scopes}"
        )

    samples = wft.build_causal_samples(workflows)
    tickets, crossfit = wft.crossfit_causal_slack(
        samples,
        folds=5,
        split_salt="aegiskv-large-v2-crossfit",
        interval_coverage=0.90,
        gbdt_config=AegisGBDTConfig(
            estimators=48,
            learning_rate=0.06,
            max_depth=3,
            min_leaf=48,
            max_bins=32,
        ),
        blend_tau_rounds=2.0,
    )
    tickets = wft.zero_slack_tickets_for_unmodeled_requests(workflows, tickets)
    actual_remaining_by_rid = {
        row.publish_for_request_id: float(row.actual_remaining_ms)
        for row in samples
        if row.publish_for_request_id
    }

    workflow_index = {
        workflow.workflow_id: index for index, workflow in enumerate(workflows)
    }
    rid = np.asarray([request.rid for request in requests], dtype="U64")
    arrays: dict[str, np.ndarray] = {
        "request_id": rid,
        "workflow_index": np.asarray(
            [workflow_index[request.workflow_id] for request in requests],
            dtype=np.int16,
        ),
        "stage_id": np.asarray([request.stage_id for request in requests], dtype=np.int16),
        "role": np.asarray([request.role == "subagent" for request in requests], dtype=np.int8),
        "arrival_ms": np.asarray([request.arrival_ms for request in requests], dtype=np.float64),
        "api_time_ms": np.asarray([request.api_time_ms for request in requests], dtype=np.float32),
        "prompt_tokens_raw": np.asarray([request.prompt_tokens for request in requests], dtype=np.int32),
        "output_tokens_raw": np.asarray([request.output_tokens for request in requests], dtype=np.int32),
        "hash_block_count": np.asarray(
            [raw_by_rid[request.rid]["hash_block_count"] for request in requests],
            dtype=np.int32,
        ),
        "prefix_reuse_block_count": np.asarray(
            [raw_by_rid[request.rid]["prefix_reuse_block_count"] for request in requests],
            dtype=np.int32,
        ),
        "input_reuse_tokens_raw": np.asarray(
            [raw_by_rid[request.rid]["input_reuse_tokens"] for request in requests],
            dtype=np.int32,
        ),
        "actual_slack_ms": np.asarray(
            [tickets[request.rid].actual_slack_ms for request in requests],
            dtype=np.float32,
        ),
        "predicted_slack_ms": np.asarray(
            [tickets[request.rid].slack_q50_ms for request in requests],
            dtype=np.float32,
        ),
        "slack_lcb_ms": np.asarray(
            [tickets[request.rid].slack_lcb_ms for request in requests],
            dtype=np.float32,
        ),
        "actual_remaining_ms": np.asarray(
            [actual_remaining_by_rid.get(request.rid, 0.0) for request in requests],
            dtype=np.float32,
        ),
        "predicted_remaining_ms": np.asarray(
            [tickets[request.rid].remaining_q50_ms for request in requests],
            dtype=np.float32,
        ),
        "remaining_lower_ms": np.asarray(
            [tickets[request.rid].remaining_interval_lower_ms for request in requests],
            dtype=np.float32,
        ),
        "remaining_upper_ms": np.asarray(
            [tickets[request.rid].remaining_interval_upper_ms for request in requests],
            dtype=np.float32,
        ),
        "modeled_by_algorithm1": np.asarray(
            [tickets[request.rid].fold >= 0 for request in requests], dtype=np.int8
        ),
    }
    array_manifest: dict[str, Any] = {}
    for name, array in arrays.items():
        path = output_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        array_manifest[name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }

    workflow_path = output_dir / "workflow_reuse.json"
    workflow_path.write_text(
        json.dumps(workflow_reuse, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    modeled = arrays["modeled_by_algorithm1"].astype(bool)
    actual_remaining = arrays["actual_remaining_ms"][modeled]
    predicted_remaining = arrays["predicted_remaining_ms"][modeled]
    actual_slack = arrays["actual_slack_ms"][modeled]
    predicted_slack = arrays["predicted_slack_ms"][modeled]
    remaining_diag = _regression_metrics(actual_remaining, predicted_remaining)
    remaining_diag["interval_coverage"] = float(
        np.mean(
            (arrays["remaining_lower_ms"][modeled] <= actual_remaining)
            & (actual_remaining <= arrays["remaining_upper_ms"][modeled])
        )
    )
    slack_diag = _regression_metrics(actual_slack, predicted_slack)
    slack_diag["lcb_coverage"] = float(
        np.mean(arrays["slack_lcb_ms"][modeled] <= actual_slack + 1e-6)
    )
    input_total = int(np.sum(arrays["prompt_tokens_raw"], dtype=np.int64))
    reuse_total = int(np.sum(arrays["input_reuse_tokens_raw"], dtype=np.int64))
    manifest = {
        "schema_version": 1,
        "evidence_class": "trace_grounded_compact_workload_v1",
        "campaign_id": protocol["campaign_id"],
        "source": {
            "path": str(source),
            "sha256": observed_sha,
            "size_bytes": source.stat().st_size,
            "workflow_count": len(workflows),
            "request_count": len(requests),
            "input_tokens": input_total,
            "output_tokens": int(np.sum(arrays["output_tokens_raw"], dtype=np.int64)),
        },
        "dag": wft.dataset_manifest(
            workflows, source_path=source, source_sha256=observed_sha
        ),
        "reuse": {
            "semantics": "workload prefix-block reuse from native Weka hash_ids; not protection-copy deduplication",
            "hash_id_scope": "local",
            "block_size_tokens": 64,
            "input_reuse_tokens": reuse_total,
            "input_reuse_fraction": reuse_total / input_total,
            "request_fraction_with_reuse": float(
                np.mean(arrays["input_reuse_tokens_raw"] > 0)
            ),
            "request_reuse_fraction_distribution": _percentiles(
                arrays["input_reuse_tokens_raw"]
                / np.maximum(arrays["prompt_tokens_raw"], 1)
            ),
            "workflow_reuse_fraction_distribution": _percentiles(
                [row["input_reuse_fraction"] for row in workflow_reuse]
            ),
            "controller_cross_request_copy_deduplication": False,
        },
        "algorithm1": {
            "evidence_class": "workflow_disjoint_trace_grounded_crossfit",
            "crossfit": crossfit,
            "remaining": remaining_diag,
            "slack": slack_diag,
            "unmodeled_requests_fail_closed_to_zero_slack": int(np.sum(~modeled)),
        },
        "arrays": array_manifest,
        "workflow_reuse": {
            "path": workflow_path.name,
            "sha256": _sha256(workflow_path),
            "count": len(workflow_reuse),
        },
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.protocol, args.output_dir, force=args.force)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output_dir),
                "workflow_count": manifest["source"]["workflow_count"],
                "request_count": manifest["source"]["request_count"],
                "input_reuse_fraction": manifest["reuse"]["input_reuse_fraction"],
                "algorithm1_remaining_mae_ms": manifest["algorithm1"]["remaining"]["mae_ms"],
                "algorithm1_slack_mae_ms": manifest["algorithm1"]["slack"]["mae_ms"],
                "canonical_sha256": manifest["canonical_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
