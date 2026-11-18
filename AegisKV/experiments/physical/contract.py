"""Immutable plan binding and fail-closed inventory validation.

This module never submits a job, sends a signal, starts a server, or labels a
simulation as physical evidence.  It only binds a deterministic subset of the
audited simulation lineage and checks a caller-supplied hardware inventory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


PHYSICAL_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PHYSICAL_ROOT.parent
CONFIG_ROOT = PHYSICAL_ROOT / "configs"
LAYER_FILES = ("request.json", "failure.json", "deployment.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_physical_configs(config_root: Path = CONFIG_ROOT) -> dict[str, dict[str, Any]]:
    config_root = Path(config_root)
    loaded: dict[str, dict[str, Any]] = {}
    schemas = {
        "request": "aegiskv.physical-request.v1",
        "failure": "aegiskv.physical-failure.v1",
        "deployment": "aegiskv.physical-deployment.v1",
    }
    for filename in LAYER_FILES:
        layer = filename.removesuffix(".json")
        value = _read_json(config_root / filename)
        _require(value.get("layer") == layer, f"{filename}: layer mismatch")
        _require(value.get("schema_version") == schemas[layer], f"{filename}: schema mismatch")
        loaded[layer] = value

    request = loaded["request"]
    deployment = loaded["deployment"]
    selection = request["selection"]
    _require(selection["independent_seed_count_per_group"] >= 3, "at least three matched seeds required")
    _require(sum(selection["expected_cells_by_group"].values()) == selection["expected_cell_count"], "cell-count contract mismatch")
    _require(selection["retain_all_non_seed_axes"] is True, "physical plan may not prune non-seed axes")
    _require(request["replay_contract"]["resampling_allowed"] is False, "matched replay may not resample")
    topology = deployment["topology"]
    _require(topology["node_count"] == 4 and topology["gpus_per_node"] == 8, "physical topology must remain 4x8")
    _require(deployment["models"]["qwen3-8b"]["tp_size"] == 1, "Qwen3-8B must use TP1")
    _require(deployment["models"]["qwen3-32b"]["tp_size"] == 4, "Qwen3-32B must use TP4")
    _require(deployment["authorization"]["automatic_job_submission"] is False, "automatic submission is forbidden")
    _require(
        deployment["runtime_identity"]["current_design_sha256"]
        == request["source_lineage"]["current_design_sha256"],
        "design hash differs across physical layers",
    )
    return loaded


def _verify_hash(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"missing {label}: {path}")
    _require(bool(SHA256_RE.fullmatch(str(expected))), f"invalid expected hash for {label}")
    actual = sha256_file(path)
    _require(actual == expected, f"{label} SHA-256 mismatch: {actual}")


def verify_source_lineage(
    lineage: Path, request_config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    lineage = Path(lineage).resolve()
    binding = request_config["source_lineage"]
    expected_path = (EXPERIMENT_ROOT / binding["relative_path"]).resolve()
    _require(lineage == expected_path, f"lineage path is not the preregistered source: {lineage}")
    _verify_hash(lineage / "manifest.json", binding["manifest_sha256"], "manifest")
    _verify_hash(lineage / "audit.json", binding["audit_sha256"], "audit")
    _verify_hash(lineage / "summary.csv", binding["summary_sha256"], "summary")
    _verify_hash(
        lineage / "config_snapshot" / "current_design.md",
        binding["current_design_sha256"],
        "current design",
    )
    manifest = _read_json(lineage / "manifest.json")
    audit = _read_json(lineage / "audit.json")
    _require(manifest.get("schema_version") == "aegiskv.experiment-lineage.v2", "lineage schema mismatch")
    _require(manifest.get("completed_cell_count") == manifest.get("expected_cell_count") == 613, "source lineage is incomplete")
    _require(manifest.get("metadata_only") is True and manifest.get("physical_result") is False, "source lineage evidence label mismatch")
    _require(audit.get("status") == binding["required_audit_status"], "source audit status mismatch")
    _require(audit.get("semantic_and_reproducibility_status") == "PASS", "source semantic audit did not pass")
    _require(audit.get("failed_checks") == [], "source audit contains failed checks")
    for relative, expected in manifest.get("config_sha256", {}).items():
        _verify_hash(lineage / "config_snapshot" / relative, expected, f"snapshotted config {relative}")
    for relative, expected in manifest.get("code_sha256", {}).items():
        _verify_hash(lineage / "code_snapshot" / relative, expected, f"snapshotted code {relative}")
    return manifest, audit


def _clean_csv_value(value: str) -> Any:
    if value == "":
        return None
    if value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _artifact_binding(lineage: Path, relative: str) -> dict[str, Any]:
    path = lineage / relative
    _require(path.is_file(), f"source artifact missing: {relative}")
    return {
        "relative_path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_matched_replay_plan(
    *,
    lineage: Path | None = None,
    config_root: Path = CONFIG_ROOT,
) -> dict[str, Any]:
    configs = load_physical_configs(config_root)
    request = configs["request"]
    if lineage is None:
        lineage = EXPERIMENT_ROOT / request["source_lineage"]["relative_path"]
    lineage = Path(lineage).resolve()
    manifest, audit = verify_source_lineage(lineage, request)

    with (lineage / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selection = request["selection"]
    groups = list(selection["groups"])
    seeds_by_group: dict[str, list[int]] = {}
    for group in groups:
        seeds = sorted({int(row["seed"]) for row in rows if row["group"] == group})
        count = int(selection["independent_seed_count_per_group"])
        _require(len(seeds) >= count, f"{group}: insufficient source seeds")
        seeds_by_group[group] = seeds[:count]

    selected = [
        row
        for row in rows
        if row["group"] in groups and int(row["seed"]) in seeds_by_group[row["group"]]
    ]
    counts = Counter(row["group"] for row in selected)
    _require(dict(counts) == selection["expected_cells_by_group"], f"matched cell counts changed: {dict(counts)}")
    _require(len(selected) == selection["expected_cell_count"], "matched replay cell count changed")
    _require(len({row["cell_id"] for row in selected}) == len(selected), "duplicate source cell IDs")

    input_fields = (
        "cell_id",
        "group",
        "seed",
        "model",
        "policy",
        "failure_scenario",
        "failed_workers",
        "failed_worker_count",
        "failure_time_ms",
        "scheduled_failure_count",
        "repair_time_ms",
        "load_multiplier",
        "context_tokens",
        "effective_bandwidth_GB_per_s",
        "workload_sha256",
    )
    cells: list[dict[str, Any]] = []
    artifact_cache: dict[str, dict[str, Any]] = {}
    for row in sorted(selected, key=lambda value: (groups.index(value["group"]), int(value["seed"]), value["cell_id"])):
        raw_relative = row["raw_artifact"]
        shadow_relative = row["shadow_artifact"]
        for relative in (raw_relative, shadow_relative):
            if relative not in artifact_cache:
                artifact_cache[relative] = _artifact_binding(lineage, relative)
        cells.append(
            {
                "inputs": {field: _clean_csv_value(row[field]) for field in input_fields},
                "simulation_reference": {
                    "raw_artifact": artifact_cache[raw_relative],
                    "shadow_artifact": artifact_cache[shadow_relative],
                },
            }
        )

    physical_config_hashes = {
        filename: sha256_file(Path(config_root) / filename) for filename in LAYER_FILES
    }
    physical_code_hashes = {
        name: sha256_file(PHYSICAL_ROOT / name)
        for name in ("contract.py", "build_plan.py")
    }
    plan: dict[str, Any] = {
        "schema_version": "aegiskv.physical-matched-replay-plan.v1",
        "source_lineage": {
            "relative_path": request["source_lineage"]["relative_path"],
            "manifest_sha256": request["source_lineage"]["manifest_sha256"],
            "audit_sha256": request["source_lineage"]["audit_sha256"],
            "summary_sha256": request["source_lineage"]["summary_sha256"],
            "metadata_only": manifest["metadata_only"],
            "source_audit_status": audit["status"],
        },
        "physical_config_sha256": physical_config_hashes,
        "physical_code_sha256": physical_code_hashes,
        "selection": {
            "seed_rule": selection["seed_rule"],
            "seeds_by_group": seeds_by_group,
            "cell_count": len(cells),
            "cells_by_group": dict(counts),
        },
        "cells": cells,
        "authorization": {
            "static_plan_materialized": True,
            "calibration_collection_authorized": False,
            "matched_replay_authorized": False,
            "performance_result_authorized": False,
            "paper_ready": False,
            "job_submission_performed": False,
        },
        "evidence_boundary": "static plan only; valid same-topology inventory and calibrated profile remain required",
    }
    plan["plan_id"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    return plan


def verify_plan(plan: Mapping[str, Any], configs: Mapping[str, Mapping[str, Any]]) -> None:
    _require(plan.get("schema_version") == "aegiskv.physical-matched-replay-plan.v1", "plan schema mismatch")
    candidate = deepcopy(dict(plan))
    plan_id = candidate.pop("plan_id", None)
    _require(plan_id == hashlib.sha256(canonical_json_bytes(candidate)).hexdigest(), "plan identity mismatch")
    _require(plan["selection"]["cell_count"] == configs["request"]["selection"]["expected_cell_count"], "plan cell count mismatch")
    _require(plan["authorization"]["matched_replay_authorized"] is False, "static plan cannot authorize replay")
    for filename, expected in plan["physical_config_sha256"].items():
        _verify_hash(CONFIG_ROOT / filename, expected, f"physical config {filename}")
    for filename, expected in plan["physical_code_sha256"].items():
        _verify_hash(PHYSICAL_ROOT / filename, expected, f"physical code {filename}")


def preflight_inventory(
    *,
    inventory: Mapping[str, Any],
    plan: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    configs = dict(configs or load_physical_configs())
    verify_plan(plan, configs)
    topology = configs["deployment"]["topology"]
    failures: list[str] = []

    def check(condition: bool, code: str) -> None:
        if not condition:
            failures.append(code)

    check(inventory.get("schema_version") == "aegiskv.physical-inventory.v1", "inventory_schema")
    allocation = inventory.get("allocation") or {}
    check(allocation.get("exclusive") is True, "allocation_not_exclusive")
    check(allocation.get("operator_authorized") is True, "operator_authorization_missing")
    nodes = inventory.get("nodes") or []
    check(len(nodes) == int(topology["node_count"]), "node_count_mismatch")
    hostnames = [str(node.get("hostname", "")) for node in nodes]
    check(len(hostnames) == len(set(hostnames)) and all(hostnames), "node_identity_invalid")
    gpu_uuids: list[str] = []
    for node in nodes:
        hostname = str(node.get("hostname", "unknown"))
        check(float(node.get("host_memory_gib", 0)) >= float(topology["host_memory_gib_min_per_node"]), f"{hostname}:host_memory_insufficient")
        check(float(node.get("aegiskv_reserved_host_memory_gib", 0)) >= float(topology["host_memory_gib_reserved_for_aegiskv_min_per_node"]), f"{hostname}:aegiskv_host_reservation_insufficient")
        check(float(node.get("network_line_rate_gbit_per_s", 0)) >= float(topology["network_line_rate_gbit_per_s_min"]), f"{hostname}:network_line_rate_insufficient")
        check(node.get("all_pairs_intra_node_p2p") is True, f"{hostname}:p2p_incomplete")
        check(node.get("zero_foreign_gpu_processes") is True, f"{hostname}:foreign_gpu_process")
        gpus = node.get("gpus") or []
        check(len(gpus) == int(topology["gpus_per_node"]), f"{hostname}:gpu_count_mismatch")
        for gpu in gpus:
            name = str(gpu.get("name", ""))
            memory_mib = int(gpu.get("memory_mib", 0))
            for required in topology["gpu_name_required_substrings"]:
                check(required.lower() in name.lower(), f"{hostname}:gpu_name_mismatch")
            check(int(topology["gpu_memory_mib_min"]) <= memory_mib <= int(topology["gpu_memory_mib_max"]), f"{hostname}:gpu_memory_mismatch")
            check(str(gpu.get("compute_capability")) == topology["compute_capability"], f"{hostname}:compute_capability_mismatch")
            gpu_uuids.append(str(gpu.get("uuid", "")))
    check(len(gpu_uuids) == len(set(gpu_uuids)) and all(gpu_uuids), "gpu_uuid_invalid")
    fabric = inventory.get("fabric") or {}
    check(fabric.get("all_pairs_inter_node_rdma") is True, "inter_node_rdma_incomplete")
    runtime = inventory.get("runtime_identity") or {}
    for field in ("sglang_git_sha", "mooncake_git_sha", "container_digest"):
        check(bool(SHA256_RE.fullmatch(str(runtime.get(field, "")).removeprefix("sha256:"))), f"runtime_identity:{field}")
    check(runtime.get("current_design_sha256") == configs["deployment"]["runtime_identity"]["current_design_sha256"], "runtime_identity:current_design")
    revisions = runtime.get("model_snapshot_revisions") or {}
    for model in configs["deployment"]["models"]:
        check(bool(SHA256_RE.fullmatch(str(revisions.get(model, "")))), f"runtime_identity:model_revision:{model}")

    unique_failures = sorted(set(failures))
    ready = not unique_failures
    inventory_sha = hashlib.sha256(canonical_json_bytes(dict(inventory))).hexdigest()
    return {
        "schema_version": "aegiskv.physical-inventory-preflight.v1",
        "status": "READY_FOR_CALIBRATION" if ready else "BLOCKED_TOPOLOGY_OR_IDENTITY",
        "plan_id": plan["plan_id"],
        "inventory_sha256": inventory_sha,
        "failures": unique_failures,
        "authorization": {
            "calibration_collection_authorized": ready,
            "matched_replay_authorized": False,
            "performance_result_authorized": False,
            "paper_ready": False,
            "job_submission_performed": False,
        },
        "next_gate": "same-topology calibrated profile with holdout error audit",
    }
