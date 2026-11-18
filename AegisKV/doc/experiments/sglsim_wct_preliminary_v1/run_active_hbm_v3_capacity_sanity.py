#!/usr/bin/env python3
"""Validate the frozen active-serving-HBM v3 capacity equations."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PROTOCOL_PATH = ROOT / "active_hbm_v3_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def serving_reservation_bytes(protocol: dict, prompt_tokens: int) -> int:
    """Return dispatch reservation without accepting a realized-output input."""

    shared = protocol["shared_capacity"]
    model = protocol["frozen_model"]
    return (
        int(prompt_tokens) + int(shared["request_max_output_tokens"])
    ) * int(model["kv_bytes_per_token"])


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    target = ROOT / "results" / "active_hbm_v3_capacity_sanity"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    model = protocol["frozen_model"]
    runtime = protocol["runtime_reserve"]
    shared = protocol["shared_capacity"]
    total = int(protocol["hardware"]["total_hbm_bytes"])
    static = int(total * float(runtime["mem_fraction_static"]))
    reserve = total - static
    weights = int(model["weight_bytes_per_tp1_pp1_rank"])
    budget = static - weights
    kv = (
        2 * int(model["layers"]) * int(model["kv_heads"])
        * int(model["head_dim"]) * int(model["kv_dtype_bytes"])
    )
    index = json.loads(Path(model["safetensors_index_path"]).read_text(encoding="utf-8"))
    initial_backup_limit = int(protocol["backup_projection"]["aegiskv_initial_peer_backup_limit_bytes"])
    max_engine_lanes = int(shared["max_engine_lanes_per_gpu"])
    max_output = int(shared["request_max_output_tokens"])
    workload_source = json.loads(
        (ROOT / shared["request_max_output_tokens_source_path"]).read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for context in protocol["sanity_matrix"]["context_tokens"]:
        for realized_output in protocol["sanity_matrix"][
            "realized_output_tokens_for_oracle_invariance_test"
        ]:
            # The diagnostic varies final output length while deliberately
            # keeping dispatch reservation fixed at request_max_output_tokens.
            serving = serving_reservation_bytes(protocol, int(context))
            prompt_backup = int(context) * kv
            no_backup_lanes = min(max_engine_lanes, budget // serving)
            initial_backup_lanes = max(
                lanes
                for lanes in range(max_engine_lanes + 1)
                if lanes * serving + min(
                    initial_backup_limit, lanes * prompt_backup
                ) <= budget
            )
            for offered in protocol["sanity_matrix"]["offered_active_requests_per_gpu"]:
                dispatched_no_backup = min(int(offered), int(no_backup_lanes))
                dispatched_initial_backup = min(int(offered), int(initial_backup_lanes))
                no_backup_used = dispatched_no_backup * serving
                projected_backup = min(
                    initial_backup_limit,
                    dispatched_initial_backup * prompt_backup,
                )
                initial_total = dispatched_initial_backup * serving + projected_backup
                rows.append({
                    "context_tokens": context,
                    "trace_realized_output_tokens_diagnostic_only": realized_output,
                    "request_max_output_tokens_used_at_dispatch": max_output,
                    "offered_active_requests_per_gpu": offered,
                    "request_serving_reservation_bytes": serving,
                    "request_serving_reservation_gib": serving / 1024**3,
                    "shared_budget_bytes": budget,
                    "shared_budget_gib": budget / 1024**3,
                    "max_lanes_no_peer_backup": no_backup_lanes,
                    "max_lanes_with_initial_peer_backup_projection": initial_backup_lanes,
                    "dispatched_no_backup": dispatched_no_backup,
                    "dispatched_with_initial_backup": dispatched_initial_backup,
                    "no_backup_active_serving_bytes": no_backup_used,
                    "initial_active_serving_bytes": dispatched_initial_backup * serving,
                    "initial_projected_peer_backup_bytes": projected_backup,
                    "initial_shared_used_bytes": initial_total,
                    "capacity_invariant_pass": initial_total <= budget,
                    "offered_requires_memory_throttle_no_backup": int(offered) > no_backup_lanes,
                    "offered_requires_memory_throttle_initial_backup": int(offered) > initial_backup_lanes,
                })
    csv_path = stage / "capacity_sanity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checks = {
        "frozen_config_hash_matches": sha256(Path(model["config_path"])) == model["config_sha256"],
        "frozen_safetensors_index_hash_matches": sha256(Path(model["safetensors_index_path"])) == model["safetensors_index_sha256"],
        "weight_bytes_match_safetensors_metadata": int(index["metadata"]["total_size"]) == weights,
        "kv_formula_matches_147456_bytes_per_token": kv == int(model["kv_bytes_per_token"]) == 147456,
        "static_hbm_formula_exact": static == int(runtime["static_hbm_bytes"]),
        "runtime_reserve_formula_exact": reserve == int(runtime["runtime_reserve_bytes"]),
        "shared_budget_formula_exact": budget == int(shared["serving_plus_peer_backup_budget_bytes"]),
        "max_output_source_hash_matches": sha256(
            ROOT / shared["request_max_output_tokens_source_path"]
        ) == shared["request_max_output_tokens_source_sha256"],
        "scheduler_default_source_hash_matches": sha256(
            REPO / runtime["source_path"]
        ) == runtime["source_sha256"],
        "capacity_formula_source_hash_matches": sha256(
            REPO / runtime["capacity_formula_source_path"]
        ) == runtime["capacity_formula_source_sha256"],
        "max_output_value_is_frozen_workload_max": max_output == max(
            int(value) for value in workload_source["workload"]["output_tokens"]
        ) == 4096,
        "fixed_allocations_fit_total_hbm": weights + reserve < total,
        "every_request_individually_fits_shared_budget": all(
            int(row["request_serving_reservation_bytes"]) <= budget for row in rows
        ),
        "every_enumerated_dispatch_respects_capacity": all(row["capacity_invariant_pass"] for row in rows),
        "some_registered_cells_require_memory_throttle": any(
            row["offered_requires_memory_throttle_no_backup"] for row in rows
        ) and any(row["offered_requires_memory_throttle_initial_backup"] for row in rows),
        "initial_peer_backup_never_increases_feasible_lanes": all(
            int(row["max_lanes_with_initial_peer_backup_projection"])
            <= int(row["max_lanes_no_peer_backup"])
            for row in rows
        ),
        "dispatch_uses_no_realized_output_oracle": all(
            len({
                (
                    int(row["request_serving_reservation_bytes"]),
                    int(row["dispatched_no_backup"]),
                    int(row["dispatched_with_initial_backup"]),
                )
                for row in rows
                if int(row["context_tokens"]) == int(context)
                and int(row["offered_active_requests_per_gpu"]) == int(offered)
            }) == 1
            for context in protocol["sanity_matrix"]["context_tokens"]
            for offered in protocol["sanity_matrix"]["offered_active_requests_per_gpu"]
        ) and protocol["no_oracle_gate"]["dispatch_uses_no_realized_output_oracle"] is True,
        "no_performance_threshold_registered": protocol["registered_gates"]["no_performance_win_threshold"] is True,
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "row_count": len(rows),
        "formula_values": {
            "total_hbm_bytes": total,
            "weight_bytes": weights,
            "runtime_reserve_bytes": reserve,
            "serving_plus_peer_backup_budget_bytes": budget,
            "kv_bytes_per_token": kv,
        },
        "paper_ready": False,
        "performance_result": False,
        "evidence_boundary": protocol["evidence_boundary"],
    }
    audit_path = stage / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {
        "audit.json": sha256(audit_path),
        "capacity_sanity.csv": sha256(csv_path),
    }
    manifest = {
        "schema_version": 1,
        "status": audit["status"],
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "target": str(target),
        "artifacts": artifacts,
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        + f"{sha256(manifest_path)}  manifest.json\n",
        encoding="utf-8",
    )
    if audit["status"] != "PASS":
        raise RuntimeError([name for name, passed in checks.items() if not passed])
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={sha256(target / 'audit.json')}")
    print(f"manifest_sha256={sha256(target / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
