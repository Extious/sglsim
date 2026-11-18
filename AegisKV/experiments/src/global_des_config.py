"""Fail-closed loader for the planned AegisKV global-DES experiment suite.

This module validates configuration only.  It neither launches the simulator nor
creates result artifacts, and a valid suite remains ``planned_unexecuted``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = EXPERIMENT_ROOT / "configs_v3"

SUITE_SCHEMA = "aegiskv.global-des-suite.v3"
LAYER_SCHEMA = "aegiskv.global-des-layer.v3"
PLANNED_STATUS = "planned_unexecuted"
LAYER_NAMES = ("request", "failure", "deployment")
LAYER_FILENAMES = tuple(f"{name}.json" for name in LAYER_NAMES)
GROUP_IDS = tuple(f"P{index}" for index in range(13))
GIB = 1024**3
GB = 10**9

TEN_SEEDS = tuple(range(2026082900, 2026082910))
FIVE_SEEDS = TEN_SEEDS[:5]
THREE_SEEDS = TEN_SEEDS[:3]

EXPECTED_CELLS = {
    "P0": 12,
    "P1": 160,
    "P2": 240,
    "P3": 640,
    "P4": 40,
    "P5": 60,
    "P6": 200,
    "P7": 240,
    "P8": 160,
    "P9": 160,
    "P10": 120,
    "P11": 60,
    "P12": 54,
}
EXPECTED_TOTAL_CELLS = 2146

P0_SCENARIOS = (
    "independent_dp_overlap",
    "qwen3_8b_tp1_shard",
    "qwen3_32b_tp4_shard",
    "synthetic_pp2_shard",
    "same_node_peer_hbm_path",
    "cross_node_peer_hbm_path",
    "source_to_node_dram_path",
    "node_dram_to_recovery_gpu_path",
    "failure_in_pending_window",
    "failure_after_commit_source_first",
    "target_first_and_node_failure",
    "stable_same_timestamp_tie_hash",
)

REQUIRED_INVARIANTS = (
    "global_event_order",
    "deterministic_event_hash",
    "per_gpu_hbm_capacity",
    "per_node_dram_capacity",
    "kv_extent_conservation",
    "legal_state_transition",
    "all_shards_ack_before_commit",
    "no_hbm_to_dram_edge",
    "independent_hbm_dram_admission",
    "non_lockstep_dp_overlap",
)

AXIS_OWNERSHIP = {
    "request": (
        "contract_scenario",
        "seed",
        "load_multiplier",
        "context_tokens",
        "load_phase",
        "reuse_level",
        "dag_stratum",
    ),
    "failure": (
        "failure_scenario",
        "fault_domain",
        "repair_horizon_s",
        "failure_severity",
    ),
    "deployment": (
        "policy",
        "model",
        "node_count",
        "hbm_budget_gib_per_gpu",
        "dram_budget_gib_per_node",
        "replica_placement",
        "cross_node_bandwidth_gb_s",
        "algorithm_variant",
        "verification_chunk_tokens",
    ),
}

EXPECTED_GROUP_AXES: dict[str, dict[str, dict[str, tuple[Any, ...]]]] = {
    "P0": {
        "request": {"contract_scenario": P0_SCENARIOS, "seed": TEN_SEEDS[:1]},
        "failure": {},
        "deployment": {},
    },
    "P1": {
        "request": {"seed": TEN_SEEDS},
        "failure": {"fault_domain": ("worker_1", "node_1")},
        "deployment": {
            "policy": ("recompute", "host_checkpoint", "neighbor_replica", "aegiskv"),
            "model": ("qwen3-8b", "qwen3-32b"),
        },
    },
    "P2": {
        "request": {"seed": TEN_SEEDS, "load_multiplier": (0.5, 1.0, 1.5, 2.0)},
        "failure": {"failure_scenario": ("no_failure", "worker_1", "node_1")},
        "deployment": {"policy": ("recompute", "aegiskv")},
    },
    "P3": {
        "request": {"seed": TEN_SEEDS, "context_tokens": (8192, 16384, 32768, 40960)},
        "failure": {},
        "deployment": {
            "node_count": (1, 2, 4, 8),
            "model": ("qwen3-8b", "qwen3-32b"),
            "policy": ("recompute", "aegiskv"),
        },
    },
    "P4": {
        "request": {"seed": TEN_SEEDS},
        "failure": {},
        "deployment": {
            "algorithm_variant": (
                "full",
                "no_algorithm_1",
                "static_placement",
                "no_algorithm_3",
            )
        },
    },
    "P5": {
        "request": {"seed": FIVE_SEEDS, "load_phase": ("low", "high", "recovery")},
        "failure": {},
        "deployment": {"hbm_budget_gib_per_gpu": (2, 4, 6, 8)},
    },
    "P6": {
        "request": {"seed": TEN_SEEDS, "reuse_level": (0.0, 0.25, 0.5, 0.75, 1.0)},
        "failure": {},
        "deployment": {
            "replica_placement": ("same_node_peer_hbm", "cross_node_peer_hbm"),
            "policy": ("recompute", "aegiskv"),
        },
    },
    "P7": {
        "request": {"seed": TEN_SEEDS},
        "failure": {
            "repair_horizon_s": (1, 5, 15, 30),
            "fault_domain": ("worker_1", "node_1"),
        },
        "deployment": {"cross_node_bandwidth_gb_s": (10, 20, 25)},
    },
    "P8": {
        "request": {"seed": TEN_SEEDS},
        "failure": {"failure_severity": ("worker_1", "worker_2", "worker_4", "node_1")},
        "deployment": {
            "model": ("qwen3-8b", "qwen3-32b"),
            "policy": ("recompute", "aegiskv"),
        },
    },
    "P9": {
        "request": {"seed": TEN_SEEDS, "load_multiplier": (0.5, 1.0, 1.5, 2.0)},
        "failure": {},
        "deployment": {
            "model": ("qwen3-8b", "qwen3-32b"),
            "policy": ("no_protection_control", "aegiskv"),
        },
    },
    "P10": {
        "request": {"seed": FIVE_SEEDS},
        "failure": {},
        "deployment": {
            "hbm_budget_gib_per_gpu": (2, 4, 6, 8),
            "dram_budget_gib_per_node": (128, 256, 512),
            "model": ("qwen3-8b", "qwen3-32b"),
        },
    },
    "P11": {
        "request": {"seed": TEN_SEEDS, "dag_stratum": ("small", "medium", "long")},
        "failure": {},
        "deployment": {"policy": ("recompute", "aegiskv")},
    },
    "P12": {
        "request": {"seed": THREE_SEEDS},
        "failure": {},
        "deployment": {
            "model": ("qwen3-8b", "qwen3-32b"),
            "cross_node_bandwidth_gb_s": (10, 20, 25),
            "verification_chunk_tokens": (1024, 4096, 8192),
        },
    },
}

SUITE_TOP_LEVEL_KEYS = {
    "schema_version",
    "suite_id",
    "status",
    "layers",
    "groups",
    "axis_ownership",
    "expected_total_cells",
    "p0_gate",
}
LAYER_TOP_LEVEL_KEYS = {"schema_version", "layer", "group", "extends", "fixed", "axes"}
COMMON_LAYER_TOP_LEVEL_KEYS = LAYER_TOP_LEVEL_KEYS - {"extends"}


class ConfigValidationError(ValueError):
    """Raised when a v3 configuration fails a closed validation gate."""


def _fail(message: str) -> None:
    raise ConfigValidationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    """Read a strict JSON object (duplicate keys and non-finite numbers fail)."""

    def reject_constant(value: str) -> None:
        _fail(f"non-finite JSON number {value} in {path}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigValidationError(f"cannot read strict JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"top-level JSON object required: {path}")
    return value


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def config_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
    *,
    required: set[str] | None = None,
) -> None:
    keys = set(value)
    unknown = keys - allowed
    missing = (allowed if required is None else required) - keys
    _require(not unknown, f"{context}: unknown top-level fields {sorted(unknown)}")
    _require(not missing, f"{context}: missing top-level fields {sorted(missing)}")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@dataclass(frozen=True)
class LayerConfig:
    layer: str
    group: str
    path: Path
    raw: dict[str, Any]
    resolved: dict[str, Any]

    @property
    def fixed(self) -> dict[str, Any]:
        return self.resolved["fixed"]

    @property
    def axes(self) -> dict[str, list[Any]]:
        return self.resolved["axes"]


@dataclass(frozen=True)
class GroupConfig:
    group: str
    request: LayerConfig
    failure: LayerConfig
    deployment: LayerConfig

    @property
    def layers(self) -> tuple[LayerConfig, LayerConfig, LayerConfig]:
        return (self.request, self.failure, self.deployment)

    @property
    def axis_items(self) -> tuple[tuple[str, list[Any]], ...]:
        items: list[tuple[str, list[Any]]] = []
        for layer in self.layers:
            items.extend(layer.axes.items())
        return tuple(items)

    @property
    def cell_count(self) -> int:
        return math.prod(len(values) for _, values in self.axis_items)

    def iter_cells(self) -> Iterator[dict[str, Any]]:
        names = [name for name, _ in self.axis_items]
        values = [axis for _, axis in self.axis_items]
        for ordinal, combination in enumerate(itertools.product(*values)):
            yield {
                "group": self.group,
                "cell_ordinal": ordinal,
                "axes": dict(zip(names, combination)),
            }


@dataclass(frozen=True)
class SuiteConfig:
    root: Path
    suite: dict[str, Any]
    groups: tuple[GroupConfig, ...]

    @property
    def total_cells(self) -> int:
        return sum(group.cell_count for group in self.groups)

    @property
    def group_map(self) -> dict[str, GroupConfig]:
        return {group.group: group for group in self.groups}

    def manifest_plan(self) -> dict[str, Any]:
        """Return plan metadata without implying that any cell was executed."""

        return {
            "schema_version": SUITE_SCHEMA,
            "suite_id": self.suite["suite_id"],
            "status": PLANNED_STATUS,
            "executed_cells": 0,
            "planned_cells": self.total_cells,
            "group_cells": {group.group: group.cell_count for group in self.groups},
            "resolved_config_sha256": config_hash(
                {
                    group.group: {
                        layer.layer: layer.resolved for layer in group.layers
                    }
                    for group in self.groups
                }
            ),
        }


@dataclass(frozen=True)
class GpuPlacement:
    node_id: int
    local_gpu_id: int
    dp_rank: int
    pp_rank: int
    tp_rank: int


def _validate_layer_envelope(
    value: dict[str, Any],
    *,
    path: Path,
    expected_layer: str,
    expected_group: str,
    common: bool,
) -> None:
    allowed = COMMON_LAYER_TOP_LEVEL_KEYS if common else LAYER_TOP_LEVEL_KEYS
    _exact_keys(value, allowed, str(path))
    _require(value["schema_version"] == LAYER_SCHEMA, f"{path}: layer schema mismatch")
    _require(value["layer"] == expected_layer, f"{path}: layer mismatch")
    _require(value["group"] == expected_group, f"{path}: group mismatch")
    _require(isinstance(value["fixed"], dict), f"{path}: fixed must be an object")
    _require(isinstance(value["axes"], dict), f"{path}: axes must be an object")
    if common:
        _require(not value["axes"], f"{path}: common axes would hide experiment multiplicity")


def _load_layer(root: Path, group: str, layer: str) -> LayerConfig:
    path = (root / group / f"{layer}.json").resolve()
    raw = read_json(path)
    _validate_layer_envelope(
        raw,
        path=path,
        expected_layer=layer,
        expected_group=group,
        common=False,
    )
    expected_extends = f"../common/{layer}.json"
    _require(raw["extends"] == expected_extends, f"{path}: extends must be {expected_extends!r}")
    parent_path = (path.parent / raw["extends"]).resolve()
    expected_parent = (root / "common" / f"{layer}.json").resolve()
    _require(parent_path == expected_parent, f"{path}: extends escapes the canonical common layer")
    parent = read_json(parent_path)
    _validate_layer_envelope(
        parent,
        path=parent_path,
        expected_layer=layer,
        expected_group="common",
        common=True,
    )
    resolved = _deep_merge(parent, raw)
    _require(set(resolved) == COMMON_LAYER_TOP_LEVEL_KEYS, f"{path}: invalid resolved envelope")
    return LayerConfig(layer=layer, group=group, path=path, raw=raw, resolved=resolved)


def _validate_exact_file_topology(root: Path) -> None:
    _require(root.is_dir(), f"configuration root does not exist: {root}")
    root_json = {path.name for path in root.glob("*.json")}
    _require(root_json == {"suite.json"}, f"hidden root JSON layer/config found: {sorted(root_json)}")
    expected_directories = {"common", *GROUP_IDS}
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    _require(
        actual_directories == expected_directories,
        f"configuration directories must be common plus P0-P12: {sorted(actual_directories)}",
    )
    for directory_name in sorted(expected_directories):
        directory = root / directory_name
        files = {path.name for path in directory.iterdir() if path.is_file()}
        _require(
            files == set(LAYER_FILENAMES),
            f"{directory}: exactly request/failure/deployment JSON files are required; found {sorted(files)}",
        )


def load_suite(config_root: Path = CONFIG_ROOT) -> SuiteConfig:
    root = Path(config_root).resolve()
    _validate_exact_file_topology(root)
    suite = read_json(root / "suite.json")
    groups = tuple(
        GroupConfig(
            group=group,
            request=_load_layer(root, group, "request"),
            failure=_load_layer(root, group, "failure"),
            deployment=_load_layer(root, group, "deployment"),
        )
        for group in GROUP_IDS
    )
    loaded = SuiteConfig(root=root, suite=suite, groups=groups)
    validate_suite(loaded)
    return loaded


def _axis_tuple(value: Mapping[str, Sequence[Any]]) -> dict[str, tuple[Any, ...]]:
    return {str(key): tuple(values) for key, values in value.items()}


def _validate_suite_envelope(loaded: SuiteConfig) -> None:
    suite = loaded.suite
    _exact_keys(suite, SUITE_TOP_LEVEL_KEYS, "suite.json")
    _require(suite["schema_version"] == SUITE_SCHEMA, "suite schema mismatch")
    _require(suite["suite_id"] == "aegiskv-paper-global-des-v3", "suite id mismatch")
    _require(suite["status"] == PLANNED_STATUS, "suite must remain planned_unexecuted")
    _require(tuple(suite["layers"]) == LAYER_NAMES, "exactly three layers are permitted")
    configured_ownership = {
        layer: tuple(axes) for layer, axes in suite["axis_ownership"].items()
    }
    _require(configured_ownership == AXIS_OWNERSHIP, "axis ownership contract mismatch")
    group_records = suite["groups"]
    _require(isinstance(group_records, list), "suite groups must be a list")
    _require(len(group_records) == len(GROUP_IDS), "suite must contain P0-P12 exactly once")
    for expected_group, record in zip(GROUP_IDS, group_records):
        _require(isinstance(record, dict), f"{expected_group}: group record must be an object")
        _exact_keys(record, {"id", "expected_cells", "status"}, f"suite group {expected_group}")
        _require(record["id"] == expected_group, f"suite group order/id mismatch at {expected_group}")
        _require(record["status"] == PLANNED_STATUS, f"{expected_group}: cannot claim execution")
        _require(
            record["expected_cells"] == EXPECTED_CELLS[expected_group],
            f"{expected_group}: preregistered cell count mismatch",
        )
    _require(suite["expected_total_cells"] == EXPECTED_TOTAL_CELLS, "suite total must be 2146")


def _validate_p0_gate(loaded: SuiteConfig) -> None:
    gate = loaded.suite["p0_gate"]
    _require(isinstance(gate, dict), "p0_gate must be an object")
    _exact_keys(
        gate,
        {"mode", "gate_group", "blocks_groups", "required_scenarios", "required_invariants"},
        "p0_gate",
    )
    _require(gate["mode"] == "all_cells_and_invariants_pass", "P0 gate must fail closed")
    _require(gate["gate_group"] == "P0", "P0 must be the gate group")
    _require(tuple(gate["blocks_groups"]) == GROUP_IDS[1:], "P0 must block every P1-P12 group")
    _require(tuple(gate["required_scenarios"]) == P0_SCENARIOS, "P0 scenario gate mismatch")
    _require(tuple(gate["required_invariants"]) == REQUIRED_INVARIANTS, "P0 invariant gate mismatch")
    p0 = loaded.group_map["P0"]
    _require(p0.cell_count == 12, "P0 gate requires exactly 12 cells")
    _require(tuple(p0.request.axes["contract_scenario"]) == P0_SCENARIOS, "P0 scenarios incomplete")
    _require(tuple(p0.request.axes["seed"]) == TEN_SEEDS[:1], "P0 must use one fixed seed")


def _validate_axes(loaded: SuiteConfig) -> None:
    for group in loaded.groups:
        expected_by_layer = EXPECTED_GROUP_AXES[group.group]
        seen_axes: set[str] = set()
        for layer in group.layers:
            actual = _axis_tuple(layer.axes)
            _require(
                actual == expected_by_layer[layer.layer],
                f"{group.group} {layer.layer}: axes differ from the frozen paper matrix",
            )
            for axis, values in layer.axes.items():
                _require(axis not in seen_axes, f"{group.group}: duplicate axis {axis}")
                _require(axis in AXIS_OWNERSHIP[layer.layer], f"{group.group}: {axis} owned by wrong layer")
                _require(isinstance(values, list) and values, f"{group.group}: empty/non-list axis {axis}")
                fingerprints = [canonical_json_bytes(value) for value in values]
                _require(len(fingerprints) == len(set(fingerprints)), f"{group.group}: duplicate values in {axis}")
                seen_axes.add(axis)
        _require(group.cell_count == EXPECTED_CELLS[group.group], f"{group.group}: cell product mismatch")
    _require(loaded.total_cells == EXPECTED_TOTAL_CELLS, "mechanical suite total is not 2146")


def _validate_request(group: GroupConfig) -> None:
    fixed = group.request.fixed
    _require(
        set(fixed) == {"workload", "arrival", "context", "seed", "artifact_lineage"},
        f"{group.group} request: fixed contract fields mismatch",
    )
    workload = fixed["workload"]
    for key in ("kind", "dataset_id", "request_universe", "selection"):
        _require(isinstance(workload.get(key), str) and workload[key], f"{group.group}: workload.{key} missing")
    arrival = fixed["arrival"]
    _require(arrival.get("clock") == "global_virtual_time", f"{group.group}: requests must use virtual time")
    _require(arrival.get("timestamp_unit") == "ns", f"{group.group}: request time unit must be ns")
    _require(arrival.get("closed_loop") is False, f"{group.group}: hidden wall-clock closed loop forbidden")
    context = fixed["context"]
    maximum_context = int(context["maximum_tokens"])
    _require(0 < int(context["central_tokens"]) <= maximum_context, f"{group.group}: invalid context")
    for value in group.request.axes.get("context_tokens", [context["central_tokens"]]):
        _require(0 < int(value) <= maximum_context, f"{group.group}: context exceeds model limit")
    seed_contract = fixed["seed"]
    _require(seed_contract.get("implicit_seed_forbidden") is True, f"{group.group}: implicit seed forbidden")
    _require(all(type(seed) is int for seed in group.request.axes["seed"]), f"{group.group}: integer seeds required")
    lineage = fixed["artifact_lineage"]
    _require(isinstance(lineage.get("input_artifacts"), list) and lineage["input_artifacts"], f"{group.group}: input lineage missing")
    for key in (
        "request_universe_hash_required",
        "input_sha256_required",
        "resolved_three_layer_snapshot_required",
        "simulator_git_revision_required",
        "model_revision_required",
        "immutable_run_id_required",
    ):
        _require(lineage.get(key) is True, f"{group.group}: artifact_lineage.{key} must be true")


def _validate_failure(group: GroupConfig) -> None:
    fixed = group.failure.fixed
    _require(
        set(fixed) == {"fault_model", "domain", "trigger", "repair", "paired_plan"},
        f"{group.group} failure: fixed contract fields mismatch",
    )
    fault = fixed["fault_model"]
    _require(fault.get("type") == "fail_stop", f"{group.group}: only fail-stop is allowed")
    _require(type(fault.get("enabled")) is bool, f"{group.group}: fault enabled must be boolean")
    _require(fault.get("fence_before_recovery") is True, f"{group.group}: route fence is mandatory")
    domain = fixed["domain"]
    _require(
        set(domain.get("supported", [])) == {"worker", "gpu", "process", "node"},
        f"{group.group}: incomplete failure domains",
    )
    _require(domain.get("same_node_peer_survives_node_failure") is False, f"{group.group}: node failure semantics invalid")
    trigger = fixed["trigger"]
    _require(trigger.get("short_run_poisson_generation") is False, f"{group.group}: short-run Poisson forbidden")
    mode = str(trigger.get("mode") or "")
    if group.group == "P0":
        _require(
            mode == "scenario_defined_deterministic_virtual_time",
            "P0: deterministic contract trigger mode drift",
        )
    elif fault["enabled"] is True:
        _require(
            mode == "policy_independent_active_exposure_v1",
            f"{group.group}: formal failures require the active-exposure trigger",
        )
    elif group.group in {"P5", "P9"}:
        _require(mode == "disabled", f"{group.group}: disabled fault trigger drift")
    elif group.group == "P12":
        _require(
            mode == "deterministic_analytical_verification",
            "P12: analytical trigger mode drift",
        )
    if mode == "policy_independent_active_exposure_v1":
        _require(
            int(trigger.get("active_offset_ns", 0)) > 0,
            f"{group.group}: active exposure offset must be positive",
        )
        _require(
            trigger.get("anchor_selector")
            == "minimum_arrival_ns_then_request_id_v1",
            f"{group.group}: anchor selector drift",
        )
        _require(
            trigger.get("activation_signal")
            == "actual_scheduler_stage_ownership_v1",
            f"{group.group}: activation signal drift",
        )
        _require(
            trigger.get("target_selector")
            == {
                "worker": "anchor_source_then_dense_order_v1",
                "gpu": "anchor_source_then_dense_order_v1",
                "process": "anchor_source_then_dense_order_v1",
                "node": "node_containing_anchor_source_v1",
            },
            f"{group.group}: active failure target selector drift",
        )
        shadow = trigger.get("shadow_plan")
        _require(
            isinstance(shadow, Mapping),
            f"{group.group}: shadow plan contract missing",
        )
        _require(
            set(shadow)
            == {
                "construction",
                "hash",
                "same_plan_required_across_policies",
                "policy_outcome_selection_forbidden",
                "included_inputs",
                "excluded_inputs",
            },
            f"{group.group}: shadow plan fields drift",
        )
        _require(
            shadow["construction"]
            == "pre_execution_request_universe_projection_v1",
            f"{group.group}: shadow plan construction drift",
        )
        _require(
            shadow["hash"] == "sha256_canonical_json_v1",
            f"{group.group}: shadow plan hash rule drift",
        )
        _require(
            shadow["same_plan_required_across_policies"] is True
            and shadow["policy_outcome_selection_forbidden"] is True,
            f"{group.group}: policy-independent shadow plan is mandatory",
        )
        _require(
            "deployment.policy" in shadow["excluded_inputs"]
            and "runtime_completion_time" in shadow["excluded_inputs"]
            and "runtime_failure_outcome" in shadow["excluded_inputs"],
            f"{group.group}: policy/outcome exclusions are incomplete",
        )
    normalization = trigger.get("scale_normalization")
    if group.group == "P3":
        _require(isinstance(normalization, Mapping), "P3: failure scale normalization formula missing")
        _require(
            set(normalization)
            == {
                "rule",
                "reference_node_count",
                "reference_gpus_per_node",
                "reference_active_offset_ns",
                "selected_gpu_count_source",
                "trigger_offset_formula",
                "exposure_window",
                "rounding",
            },
            "P3: failure scale normalization contract drift",
        )
        _require(
            normalization["rule"]
            == "constant_conditioned_cluster_gpu_time_exposure_v2",
            "P3: failure exposure rule mismatch",
        )
        _require(int(normalization["reference_node_count"]) == 4, "P3: failure reference node count drift")
        _require(int(normalization["reference_gpus_per_node"]) == 8, "P3: failure reference GPU count drift")
        _require(
            int(normalization["reference_active_offset_ns"])
            == int(trigger["active_offset_ns"]),
            "P3: failure reference offset differs from central active offset",
        )
        _require(
            normalization["selected_gpu_count_source"]
            == "compiled_physical_topology",
            "P3: failure normalization must use physical GPUs",
        )
        _require(
            normalization["trigger_offset_formula"]
            == "reference_active_offset_ns * reference_gpu_count / selected_gpu_count",
            "P3: failure active-offset formula drift",
        )
        _require(
            normalization["exposure_window"]
            == "after_anchor_scheduler_ownership_before_injection",
            "P3: failure exposure window drift",
        )
        _require(
            normalization["rounding"] == "nearest_integer_ns_half_up",
            "P3: failure trigger rounding drift",
        )
    else:
        _require(normalization is None, f"{group.group}: undeclared failure scale normalization")
    repair = fixed["repair"]
    _require(float(repair["central_horizon_s"]) >= 0, f"{group.group}: negative repair horizon")
    for horizon in group.failure.axes.get("repair_horizon_s", [repair["central_horizon_s"]]):
        _require(float(horizon) >= 0, f"{group.group}: negative repair horizon axis")
    paired = fixed["paired_plan"]
    _require(type(paired.get("required")) is bool, f"{group.group}: paired plan declaration required")
    if paired["required"]:
        _require(paired.get("same_seed") is True, f"{group.group}: paired seed mismatch allowed")
        _require(paired.get("same_request_universe") is True, f"{group.group}: paired universe mismatch allowed")
        _require(paired.get("shadow_schedule_hash_required") is True, f"{group.group}: shadow hash required")
        if fault["enabled"] is True:
            _require(
                paired.get("same_trigger_rule") is True
                and paired.get("same_anchor_request") is True
                and paired.get("same_active_offset") is True
                and paired.get("shadow_plan_sha256_required") is True,
                f"{group.group}: active-exposure paired plan contract incomplete",
            )
            _require(
                paired.get("same_failure_timestamp") is False,
                f"{group.group}: causal active trigger cannot require equal wall timestamps",
            )
    if group.group in {"P5", "P9", "P12"}:
        _require(fault["enabled"] is False, f"{group.group}: preregistered no-failure group enabled faults")


def _selected_axis_or_fixed(
    group: GroupConfig,
    axis: str,
    fixed_value: Any,
) -> tuple[Any, ...]:
    return tuple(group.deployment.axes.get(axis, [fixed_value]))


def generate_placement(
    deployment_fixed: Mapping[str, Any],
    *,
    model_name: str,
    node_count: int,
) -> tuple[GpuPlacement, ...]:
    """Mechanically expand the declared physical placement generator."""

    cluster = deployment_fixed["cluster"]
    model = deployment_fixed["models"][model_name]
    gpus_per_node = int(cluster["gpus_per_node"])
    tp_size = int(model["tp_size"])
    pp_size = int(model["pp_size"])
    try:
        dp_size = int(model["dp_size_by_node_count"][str(node_count)])
    except KeyError as exc:
        raise ConfigValidationError(
            f"{model_name}: no DP size declared for {node_count} nodes"
        ) from exc
    placements: list[GpuPlacement] = []
    for dp_rank in range(dp_size):
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                linear_rank = ((dp_rank * pp_size) + pp_rank) * tp_size + tp_rank
                placements.append(
                    GpuPlacement(
                        node_id=linear_rank // gpus_per_node,
                        local_gpu_id=linear_rank % gpus_per_node,
                        dp_rank=dp_rank,
                        pp_rank=pp_rank,
                        tp_rank=tp_rank,
                    )
                )
    return tuple(placements)


def build_offline_physical_topology(
    deployment_fixed: Mapping[str, Any],
    *,
    model_name: str,
    node_count: int,
):
    """Build the runtime physical-ledger topology from deployment JSON.

    The import is intentionally local so configuration validation remains a
    lightweight operation.  Every generated TP/PP shard receives exactly one
    weight shard; dynamic primary/prefix KV is supplied later by the real
    Offline Scheduler resource snapshots.
    """

    from sglsim.srt.simulation.global_des import PhysicalGpuId
    from sglsim.srt.simulation.offline_resource_bridge import (
        NodeDramSpec,
        OfflinePhysicalTopology,
        PhysicalGpuSpec,
    )

    cluster = deployment_fixed["cluster"]
    models = deployment_fixed["models"]
    _require(model_name in models, f"unknown model {model_name!r}")
    resolved_nodes = int(node_count)
    _require(resolved_nodes > 0, "node_count must be positive")
    model = models[model_name]
    shard_count = int(model["tp_size"]) * int(model["pp_size"])
    total_weight_bytes = int(model["weight_bytes_total"])
    _require(
        total_weight_bytes % shard_count == 0,
        f"{model_name}: weight bytes do not divide TP/PP shards",
    )
    weight_bytes_per_gpu = total_weight_bytes // shard_count
    hbm_capacity = int(cluster["gpu_hbm_capacity_bytes"])
    runtime_hbm = int(
        deployment_fixed["gpu_hbm"]["runtime_reserve_bytes_per_gpu"]
    )
    placements = generate_placement(
        deployment_fixed,
        model_name=model_name,
        node_count=resolved_nodes,
    )
    gpu_specs = tuple(
        PhysicalGpuSpec(
            gpu=PhysicalGpuId(
                node_id=f"node{placement.node_id}",
                local_gpu_id=placement.local_gpu_id,
                dp_rank=placement.dp_rank,
                pp_rank=placement.pp_rank,
                tp_rank=placement.tp_rank,
            ),
            hbm_capacity_bytes=hbm_capacity,
            weights_bytes=weight_bytes_per_gpu,
            runtime_bytes=runtime_hbm,
        )
        for placement in placements
    )
    dram_capacity = int(cluster["host_dram_capacity_bytes_per_node"])
    dram_runtime = int(
        deployment_fixed["node_dram"]["runtime_reserve_bytes_per_node"]
    )
    nodes = tuple(
        NodeDramSpec(
            node_id=f"node{node_id}",
            capacity_bytes=dram_capacity,
            runtime_bytes=dram_runtime,
        )
        for node_id in range(resolved_nodes)
    )
    return OfflinePhysicalTopology(gpus=gpu_specs, nodes=nodes)


def _validate_models_and_placement(group: GroupConfig) -> None:
    fixed = group.deployment.fixed
    cluster = fixed["cluster"]
    models = fixed["models"]
    _require(set(models) == {"qwen3-8b", "qwen3-32b"}, f"{group.group}: model set mismatch")
    expected_model = {
        "qwen3-8b": {"layers": 36, "kv_heads": 8, "kv_bytes": 147456, "tp": 1, "pp": 1},
        "qwen3-32b": {"layers": 64, "kv_heads": 8, "kv_bytes": 262144, "tp": 4, "pp": 1},
    }
    for model_name, geometry in models.items():
        expected = expected_model[model_name]
        calculated_kv = (
            int(geometry["num_layers"])
            * 2
            * int(geometry["num_kv_heads"])
            * int(geometry["head_dim"])
            * int(geometry["kv_dtype_bytes"])
        )
        _require(calculated_kv == int(geometry["kv_bytes_per_token_total"]), f"{group.group}: {model_name} KV geometry mismatch")
        _require(int(geometry["kv_bytes_per_token_total"]) == expected["kv_bytes"], f"{group.group}: {model_name} KV bytes drift")
        _require(int(geometry["num_layers"]) == expected["layers"], f"{group.group}: {model_name} layer count drift")
        _require(int(geometry["num_kv_heads"]) == expected["kv_heads"], f"{group.group}: {model_name} KV head count drift")
        _require(int(geometry["tp_size"]) == expected["tp"], f"{group.group}: {model_name} TP mismatch")
        _require(int(geometry["pp_size"]) == expected["pp"], f"{group.group}: {model_name} PP mismatch")
        _require(int(geometry["num_layers"]) % int(geometry["pp_size"]) == 0, f"{group.group}: PP layer sharding is fractional")
        _require(int(geometry["num_kv_heads"]) % int(geometry["tp_size"]) == 0, f"{group.group}: TP KV-head sharding is fractional")
        _require(
            int(geometry["parameter_count"]) * int(geometry["weight_dtype_bytes"])
            == int(geometry["weight_bytes_total"]),
            f"{group.group}: {model_name} weight geometry mismatch",
        )
        for node_count in (1, 2, 4, 8):
            expected_dp = node_count * int(cluster["gpus_per_node"]) // (
                int(geometry["tp_size"]) * int(geometry["pp_size"])
            )
            _require(
                int(geometry["dp_size_by_node_count"][str(node_count)]) == expected_dp,
                f"{group.group}: {model_name} DP size mismatch at {node_count} nodes",
            )
    placement = fixed["placement"]
    _require(placement["kind"] == "dense_cartesian_rank_map_v1", f"{group.group}: placement generator mismatch")
    _require(tuple(placement["rank_order"]) == ("dp_rank", "pp_rank", "tp_rank"), f"{group.group}: rank order mismatch")
    _require(tuple(placement["gpu_order"]) == ("node_id", "local_gpu_id"), f"{group.group}: GPU order mismatch")
    _require(
        placement["linear_rank_formula"] == "((dp_rank * pp_size) + pp_rank) * tp_size + tp_rank"
        and placement["node_formula"] == "linear_rank // gpus_per_node"
        and placement["local_gpu_formula"] == "linear_rank % gpus_per_node",
        f"{group.group}: placement formula drift",
    )
    _require(placement["require_unique_gpu_per_rank_tuple"] is True, f"{group.group}: unique placement not required")
    _require(placement["require_all_gpus_assigned"] is True, f"{group.group}: partial placement allowed")
    model_names = _selected_axis_or_fixed(group, "model", fixed["central_model"])
    node_counts = _selected_axis_or_fixed(group, "node_count", cluster["central_node_count"])
    for model_name, node_count_value in itertools.product(model_names, node_counts):
        _require(model_name in models, f"{group.group}: unknown model {model_name}")
        node_count = int(node_count_value)
        placements = generate_placement(fixed, model_name=str(model_name), node_count=node_count)
        capacity = node_count * int(cluster["gpus_per_node"])
        physical_ids = {(entry.node_id, entry.local_gpu_id) for entry in placements}
        rank_ids = {(entry.dp_rank, entry.pp_rank, entry.tp_rank) for entry in placements}
        _require(len(placements) == capacity, f"{group.group}: placement does not fill physical capacity")
        _require(len(physical_ids) == capacity, f"{group.group}: GPU placement collision")
        _require(len(rank_ids) == capacity, f"{group.group}: rank tuple collision")
        _require(all(0 <= entry.node_id < node_count for entry in placements), f"{group.group}: placement exceeds node capacity")
        _require(
            all(0 <= entry.local_gpu_id < int(cluster["gpus_per_node"]) for entry in placements),
            f"{group.group}: placement exceeds local GPU capacity",
        )


def _validate_memory(group: GroupConfig) -> None:
    fixed = group.deployment.fixed
    cluster = fixed["cluster"]
    hbm = fixed["gpu_hbm"]
    required_hbm_accounts = (
        "weights",
        "runtime_reserve",
        "primary_kv",
        "radix_prefix_kv",
        "aegis_pending",
        "aegis_committed",
        "free",
    )
    _require(tuple(hbm["ledger_accounts"]) == required_hbm_accounts, f"{group.group}: HBM ledger mismatch")
    _require(hbm["free_is_derived"] is True and hbm["overcommit_forbidden"] is True, f"{group.group}: HBM must fail closed")
    model_names = _selected_axis_or_fixed(group, "model", fixed["central_model"])
    hbm_budgets = _selected_axis_or_fixed(group, "hbm_budget_gib_per_gpu", hbm["central_aegis_budget_gib_per_gpu"])
    hbm_capacity = int(cluster["gpu_hbm_capacity_bytes"])
    for model_name, budget in itertools.product(model_names, hbm_budgets):
        model = fixed["models"][model_name]
        shards = int(model["tp_size"]) * int(model["pp_size"])
        _require(int(model["weight_bytes_total"]) % shards == 0, f"{group.group}: fractional weight shard")
        accounted = (
            int(model["weight_bytes_total"]) // shards
            + int(hbm["runtime_reserve_bytes_per_gpu"])
            + int(hbm["primary_kv_budget_bytes_per_gpu"])
            + int(hbm["radix_prefix_budget_bytes_per_gpu"])
            + int(budget) * GIB
        )
        _require(accounted <= hbm_capacity, f"{group.group}: per-GPU HBM ledger exceeds capacity for {model_name}/{budget}GiB")
    dram = fixed["node_dram"]
    required_dram_accounts = (
        "runtime_reserve",
        "hicache_primary",
        "aegis_pending",
        "aegis_committed",
        "free",
    )
    _require(tuple(dram["ledger_accounts"]) == required_dram_accounts, f"{group.group}: DRAM ledger mismatch")
    _require(dram["pool_scope"] == "per_node", f"{group.group}: DRAM must be per-node")
    _require(dram["global_aggregate_pool_forbidden"] is True, f"{group.group}: aggregate DRAM pool allowed")
    _require(dram["free_is_derived"] is True and dram["overcommit_forbidden"] is True, f"{group.group}: DRAM must fail closed")
    central_quota_bytes = int(dram["central_aegis_budget_gib_per_node"]) * GIB
    _require(
        int(dram["per_node_aegis_dram_quota_bytes"]) == central_quota_bytes,
        f"{group.group}: central per-node Aegis DRAM byte quota drift",
    )
    central_tokens = dram["per_node_aegis_dram_quota_tokens"]
    _require(
        set(central_tokens) == set(fixed["models"]),
        f"{group.group}: per-model Aegis DRAM token quota incomplete",
    )
    for model_name, model in fixed["models"].items():
        expected_tokens = central_quota_bytes // int(model["kv_bytes_per_token_total"])
        _require(
            int(central_tokens[model_name]) == expected_tokens,
            f"{group.group}: {model_name} per-node Aegis DRAM token quota mismatch",
        )
    _require(
        dram["quota_binding_rule"]
        == "selected_gib_to_bytes_then_floor_by_logical_kv_bytes_v1",
        f"{group.group}: Aegis DRAM quota binding rule drift",
    )
    dram_budgets = _selected_axis_or_fixed(group, "dram_budget_gib_per_node", dram["central_aegis_budget_gib_per_node"])
    host_capacity = int(cluster["host_dram_capacity_bytes_per_node"])
    for budget in dram_budgets:
        accounted = (
            int(dram["runtime_reserve_bytes_per_node"])
            + int(dram["hicache_primary_budget_bytes_per_node"])
            + int(budget) * GIB
        )
        _require(accounted <= host_capacity, f"{group.group}: per-node DRAM ledger exceeds host capacity at {budget}GiB")


def _validate_copy_resources(group: GroupConfig) -> None:
    fixed = group.deployment.fixed
    resources = fixed["copy_resources"]
    engines = resources["gpu_copy_engines"]
    for key in ("d2d_per_gpu", "d2h_per_gpu", "h2d_per_gpu"):
        _require(type(engines[key]) is int and engines[key] > 0, f"{group.group}: invalid copy engine {key}")
    _require(engines["full_duplex"] is False, f"{group.group}: undeclared full-duplex copy overlap")
    dram_lanes = resources["dram_lanes_per_node"]
    _require(int(dram_lanes["read"]) > 0 and int(dram_lanes["write"]) > 0, f"{group.group}: missing DRAM lanes")
    _require(int(dram_lanes["bandwidth_bytes_per_s"]) > 0, f"{group.group}: invalid DRAM bandwidth")
    _require(dram_lanes["contention"] == "exclusive_reservation_calendar", f"{group.group}: DRAM contention unspecified")
    links = resources["links"]
    _require(set(links) == {"pcie", "nvlink_nvswitch", "nic_rdma", "cross_node_fabric"}, f"{group.group}: link set mismatch")
    for link_name, link in links.items():
        _require(int(link["peak_bandwidth_bytes_per_s"]) > 0, f"{group.group}: nonpositive {link_name} bandwidth")
        _require(int(link["setup_ns"]) >= 0, f"{group.group}: negative {link_name} setup")
        _require(link["contention"] == "exclusive_reservation_calendar", f"{group.group}: {link_name} contention unspecified")
    cross_node = links["cross_node_fabric"]
    effective_values = _selected_axis_or_fixed(
        group,
        "cross_node_bandwidth_gb_s",
        int(cross_node["central_effective_bandwidth_bytes_per_s"]) / GB,
    )
    physical_limit = min(
        int(cross_node["peak_bandwidth_bytes_per_s"]),
        int(links["nic_rdma"]["peak_bandwidth_bytes_per_s"]),
        int(links["pcie"]["peak_bandwidth_bytes_per_s"]),
    )
    for bandwidth in effective_values:
        effective = float(bandwidth) * GB
        _require(effective > 0, f"{group.group}: nonpositive effective link bandwidth")
        _require(effective <= physical_limit, f"{group.group}: effective link bandwidth exceeds path capacity")
    required_paths = {
        "same_node_peer_hbm",
        "cross_node_peer_hbm",
        "source_to_node_dram",
        "node_dram_to_recovery_gpu",
    }
    _require(set(resources["path_resources"]) == required_paths, f"{group.group}: copy path set mismatch")
    _require(all(resources["path_resources"].values()), f"{group.group}: empty copy path resource list")


def _validate_aegis_and_des(group: GroupConfig, p0_gate: Mapping[str, Any]) -> None:
    fixed = group.deployment.fixed
    aegis = fixed["aegiskv"]
    _require(aegis["ownership"] == "source_owned", f"{group.group}: protection must be source-owned")
    admission = aegis["tier_admission"]
    _require(admission["peer_hbm"] == "independent", f"{group.group}: HBM admission is not independent")
    _require(admission["node_dram"] == "independent", f"{group.group}: DRAM admission is not independent")
    _require(admission["same_source_production_buffer"] is True, f"{group.group}: DRAM source semantics invalid")
    allowed_edges = tuple(aegis["allowed_copy_edges"])
    forbidden_edges = set(aegis["forbidden_copy_edges"])
    _require("source_primary_to_peer_hbm" in allowed_edges, f"{group.group}: HBM protection edge missing")
    _require("source_primary_to_node_dram" in allowed_edges, f"{group.group}: DRAM protection edge missing")
    _require("peer_hbm_to_node_dram" not in allowed_edges, f"{group.group}: HBM-to-DRAM edge is forbidden")
    _require("peer_hbm_to_node_dram" in forbidden_edges, f"{group.group}: HBM-to-DRAM prohibition missing")
    _require("node_dram_to_peer_hbm" in forbidden_edges, f"{group.group}: cross-tier writeback prohibition missing")
    _require(aegis["all_pp_tp_shards_ack_before_commit"] is True, f"{group.group}: partial commit allowed")
    _require(int(aegis["prefix_chunk_tokens"]) == 4096, f"{group.group}: prefix chunk must be 4096")
    _require(int(aegis["sealed_decode_chunk_tokens"]) == 256, f"{group.group}: decode chunk must be 256")
    aimd = aegis["aimd"]
    _require(0 < float(aimd["minimum_alpha"]) <= float(aimd["initial_alpha"]) <= float(aimd["maximum_alpha"]) <= 1, f"{group.group}: AIMD alpha bounds invalid")
    _require(float(aimd["additive_increase"]) > 0, f"{group.group}: AIMD increase invalid")
    _require(0 < float(aimd["multiplicative_decrease"]) < 1, f"{group.group}: AIMD decrease invalid")
    _require(aimd["quota_applies_only_to_aegis_hbm"] is True, f"{group.group}: AIMD quota scope invalid")
    variants = aegis.get("algorithm_variants")
    if group.group == "P4":
        _require(isinstance(variants, Mapping), "P4: algorithm variant contract missing")
        no_algorithm_1 = variants.get("no_algorithm_1", {})
        _require(no_algorithm_1.get("slack_ordering") is False, "P4: Algorithm 1 is not disabled")
        _require(no_algorithm_1.get("dynamic_placement") is True, "P4: Algorithm 2 changed in Algorithm 1 ablation")
        _require(no_algorithm_1.get("aimd") is True, "P4: Algorithm 3 changed in Algorithm 1 ablation")
        _require(
            no_algorithm_1.get("runtime_adapter")
            == "aegis_l1_l2.disable_slack_ordering"
            and no_algorithm_1.get("runtime_value") is True
            and float(no_algorithm_1.get("ordering_slack_ms", -1)) == 0.0,
            "P4: no_algorithm_1 runtime adapter is incomplete",
        )
    des = fixed["global_des"]
    _require(des["kernel"] == "single_authoritative_min_heap", f"{group.group}: global DES kernel mismatch")
    _require(des["time_unit"] == "ns", f"{group.group}: DES unit must be ns")
    _require(tuple(des["event_sort_key"]) == ("timestamp_ns", "priority", "sequence"), f"{group.group}: tie order mismatch")
    _require(len(des["priority_order"]) == len(set(des["priority_order"])), f"{group.group}: duplicate event priorities")
    _require(des["sequence_is_strictly_monotonic"] is True, f"{group.group}: event sequence not stable")
    _require(des["lockstep_rank_clock_forbidden"] is True, f"{group.group}: lockstep DP clock allowed")
    _require(des["resource_available_at_calendars"] is True, f"{group.group}: resource calendars missing")
    _require(tuple(des["required_invariants"]) == REQUIRED_INVARIANTS, f"{group.group}: DES invariants mismatch")
    _require(tuple(p0_gate["required_invariants"]) == tuple(des["required_invariants"]), f"{group.group}: P0/DES invariant drift")
    output = fixed["output_contract"]
    _require(str(output["evidence_class"]).startswith("planned_"), f"{group.group}: output claims execution")
    _require(output["execution_claim_permitted"] is False, f"{group.group}: planned config permits execution claim")
    required_artifacts = {
        "events.jsonl",
        "rank_steps.jsonl",
        "gpu_hbm_timeseries.jsonl",
        "node_dram_timeseries.jsonl",
        "kv_extents.jsonl",
        "transfers.jsonl",
        "requests.jsonl",
        "failures.jsonl",
        "audit.json",
        "manifest.json",
    }
    _require(set(output["required_artifacts"]) == required_artifacts, f"{group.group}: output contract incomplete")
    _require(output["derived_plot_tables_may_not_replace_raw_events"] is True, f"{group.group}: derived plots may replace raw evidence")


def _validate_provenance(group: GroupConfig) -> None:
    provenance = group.deployment.fixed["provenance"]
    _require(
        set(provenance) == {"hardware_timing", "simulator_revision", "model_revisions"},
        f"{group.group}: deployment provenance fields mismatch",
    )
    hardware = provenance["hardware_timing"]
    _require(hardware["evidence_class"] == "modeled_not_calibrated", f"{group.group}: hardware evidence class drift")
    _require(hardware["calibration_claim_permitted"] is False, f"{group.group}: modeled hardware claims calibration")
    _require(hardware["modeled_execution_authorized"] is True, f"{group.group}: modeled execution is not explicit")
    _require(
        hardware["assumption_id"] == "a100-sxm4-40gb-nvlink-pcie-rdma-modeled-v1",
        f"{group.group}: hardware assumption identity drift",
    )
    _require(
        hardware["source"] == "configs_v3/common/deployment.json#fixed/cluster+copy_resources",
        f"{group.group}: hardware assumption source drift",
    )
    _require(
        hardware["integrity_rule"]
        == "canonical_json_sha256_of_cluster_and_copy_resources_v1",
        f"{group.group}: hardware assumption integrity rule drift",
    )
    simulator = provenance["simulator_revision"]
    _require(simulator["required"] is True, f"{group.group}: simulator revision is optional")
    _require(
        simulator["binding_strategy"] == "runtime_implementation_content_sha256_v1"
        and simulator["hardcoded_dirty_tree_revision_forbidden"] is True,
        f"{group.group}: simulator revision binding is not dirty-tree safe",
    )
    models = provenance["model_revisions"]
    _require(models["required"] is True and models["missing_revision_fail_closed"] is True, f"{group.group}: model revision binding is not fail-closed")
    _require(
        models["binding_strategy"] == "frozen_json_then_cli_exact_snapshot_v1",
        f"{group.group}: model revision binding strategy drift",
    )
    revisions = models["revisions"]
    _require(set(revisions) == {"qwen3-8b", "qwen3-32b"}, f"{group.group}: model revision map incomplete")
    _require(
        revisions["qwen3-8b"] == "b968826d9c46dd6066d109eabc6255188de91218",
        f"{group.group}: Qwen3-8B frozen revision drift",
    )
    _require(
        revisions["qwen3-32b"]
        == "9216db5781bf21249d130ec9da846c4624c16137",
        f"{group.group}: Qwen3-32B frozen revision drift",
    )
    revision_sources = models["revision_sources"]
    _require(
        revision_sources
        == {
            "qwen3-8b": "https://huggingface.co/Qwen/Qwen3-8B/commits/b968826d9c46dd6066d109eabc6255188de91218",
            "qwen3-32b": "https://huggingface.co/Qwen/Qwen3-32B/commits/9216db5781bf21249d130ec9da846c4624c16137",
        },
        f"{group.group}: model revision source drift",
    )


def _validate_deployment(group: GroupConfig, p0_gate: Mapping[str, Any]) -> None:
    fixed = group.deployment.fixed
    _require(
        set(fixed)
        == {
            "cluster",
            "models",
            "central_model",
            "placement",
            "gpu_hbm",
            "node_dram",
            "copy_resources",
            "aegiskv",
            "global_des",
            "provenance",
            "output_contract",
        },
        f"{group.group} deployment: fixed contract fields mismatch",
    )
    cluster = fixed["cluster"]
    _require(int(cluster["central_node_count"]) == 4, f"{group.group}: central node count drift")
    _require(int(cluster["gpus_per_node"]) == 8, f"{group.group}: GPUs per node drift")
    _require(int(cluster["gpu_hbm_capacity_bytes"]) == 40 * GIB, f"{group.group}: A100 HBM capacity mismatch")
    _require(int(cluster["host_dram_capacity_bytes_per_node"]) > 0, f"{group.group}: host DRAM capacity missing")
    _validate_models_and_placement(group)
    _validate_memory(group)
    _validate_copy_resources(group)
    _validate_aegis_and_des(group, p0_gate)
    _validate_provenance(group)


def validate_suite(loaded: SuiteConfig) -> None:
    """Validate schema, ownership, topology, resources, plan size, and P0 gate."""

    _validate_suite_envelope(loaded)
    _validate_p0_gate(loaded)
    _validate_axes(loaded)
    for group in loaded.groups:
        _validate_request(group)
        _validate_failure(group)
        _validate_deployment(group, loaded.suite["p0_gate"])


__all__ = [
    "AXIS_OWNERSHIP",
    "CONFIG_ROOT",
    "ConfigValidationError",
    "EXPECTED_CELLS",
    "EXPECTED_TOTAL_CELLS",
    "GROUP_IDS",
    "GpuPlacement",
    "GroupConfig",
    "LayerConfig",
    "P0_SCENARIOS",
    "SuiteConfig",
    "config_hash",
    "generate_placement",
    "load_suite",
    "read_json",
    "validate_suite",
]
