"""Load, resolve, validate, and snapshot the three experiment layers."""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
SOURCE_REGISTRY_PATH = EXPERIMENT_ROOT / "research" / "source_registry.json"
PARAMETER_PROVENANCE_PATH = (
    EXPERIMENT_ROOT / "research" / "parameter_provenance.json"
)
LAYER_FILENAMES = ("request.json", "failure.json", "deployment.json")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


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
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def resolve_json(path: Path, *, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic config inheritance: {path}")
    value = read_json(path)
    parent = value.get("extends")
    if parent is None:
        return value
    parent_path = (path.parent / str(parent)).resolve()
    base = resolve_json(parent_path, seen=(*seen, path))
    resolved = _deep_merge(base, value)
    resolved["resolved_from"] = [str(parent_path), str(path)]
    return resolved


@dataclass(frozen=True)
class GroupConfig:
    group: str
    request: dict[str, Any]
    failure: dict[str, Any]
    deployment: dict[str, Any]
    source_paths: tuple[Path, Path, Path]

    @property
    def layers(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (self.request, self.failure, self.deployment)

    @property
    def axes(self) -> dict[str, list[Any]]:
        values: dict[str, list[Any]] = {}
        for layer in self.layers:
            for name, axis_values in (layer.get("axes") or {}).items():
                values[str(name)] = list(axis_values)
        return values


@dataclass(frozen=True)
class SuiteConfig:
    suite: dict[str, Any]
    groups: tuple[GroupConfig, ...]
    sources: dict[str, Any]
    parameter_provenance: dict[str, Any]


def load_group(group: str, config_root: Path = CONFIG_ROOT) -> GroupConfig:
    paths = tuple((Path(config_root) / group / name).resolve() for name in LAYER_FILENAMES)
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing layer configs for {group}: {missing}")
    layers = tuple(resolve_json(path) for path in paths)
    return GroupConfig(str(group), layers[0], layers[1], layers[2], paths)  # type: ignore[arg-type]


def load_suite(config_root: Path = CONFIG_ROOT) -> SuiteConfig:
    config_root = Path(config_root)
    suite = read_json(config_root / "suite.json")
    registry = read_json(SOURCE_REGISTRY_PATH).get("sources")
    if not isinstance(registry, dict):
        raise ValueError("source registry must contain a sources object")
    parameter_provenance = read_json(PARAMETER_PROVENANCE_PATH)
    groups = tuple(load_group(str(group), config_root) for group in suite["groups"])
    loaded = SuiteConfig(
        suite=suite,
        groups=groups,
        sources=registry,
        parameter_provenance=parameter_provenance,
    )
    validate_suite(loaded)
    return loaded


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_suite(loaded: SuiteConfig) -> None:
    suite = loaded.suite
    _require(suite.get("schema_version") == "aegiskv.experiment-suite.v2", "suite schema mismatch")
    ownership = suite.get("axis_ownership") or {}
    known_sources = set(loaded.sources)
    provenance = loaded.parameter_provenance
    _require(
        provenance.get("schema_version") == "aegiskv.parameter-provenance.v1",
        "parameter provenance schema mismatch",
    )
    evidence_classes = set(provenance.get("evidence_classes") or [])
    parameters = provenance.get("parameters") or {}
    _require(isinstance(parameters, dict) and bool(parameters), "parameter provenance missing")
    for parameter_name, record in parameters.items():
        _require(isinstance(record, dict), f"invalid provenance record {parameter_name}")
        _require(
            record.get("evidence_class") in evidence_classes,
            f"unknown evidence class for {parameter_name}",
        )
        record_sources = set(map(str, record.get("sources") or []))
        _require(bool(record_sources), f"sources missing for {parameter_name}")
        _require(
            not (record_sources - known_sources),
            f"unknown provenance sources for {parameter_name}: {sorted(record_sources - known_sources)}",
        )
        _require(bool(record.get("config_paths")), f"config paths missing for {parameter_name}")
        _require(bool(record.get("scope")), f"scope missing for {parameter_name}")
    for blocker in provenance.get("paper_readiness_blockers") or []:
        _require(blocker in parameters, f"unknown paper-readiness blocker {blocker}")
        _require(
            parameters[blocker]["evidence_class"] == "modeled_uncalibrated",
            f"paper-readiness blocker must be modeled_uncalibrated: {blocker}",
        )
    seen_groups: set[str] = set()
    paper_minimum_seeds = (
        suite.get("profiles", {}).get("paper", {}).get("minimum_distinct_seeds", {})
    )
    for group in loaded.groups:
        _require(group.group not in seen_groups, f"duplicate group {group.group}")
        seen_groups.add(group.group)
        seen_axes: set[str] = set()
        for expected_layer, layer in zip(("request", "failure", "deployment"), group.layers):
            _require(layer.get("group") == group.group, f"{group.group} {expected_layer}: group mismatch")
            _require(layer.get("layer") == expected_layer, f"{group.group}: layer mismatch")
            sources = layer.get("sources") or []
            _require(bool(sources), f"{group.group} {expected_layer}: sources required")
            unknown = set(map(str, sources)) - known_sources
            _require(not unknown, f"{group.group} {expected_layer}: unknown sources {sorted(unknown)}")
            axes = layer.get("axes") or {}
            for axis, values in axes.items():
                _require(axis not in seen_axes, f"{group.group}: duplicate axis owner {axis}")
                _require(axis in set(ownership[expected_layer]), f"{group.group}: {axis} owned by wrong layer")
                _require(isinstance(values, list) and bool(values), f"{group.group}: empty axis {axis}")
                seen_axes.add(axis)

        workload = group.request["workload"]
        _require(
            workload["profile"]
            in {"tracelab_session", "controlled_fork_join", "weka_agentx_causal"},
            f"{group.group}: unknown workload",
        )
        contexts = group.request.get("axes", {}).get("context_tokens", [workload["context_tokens"]])
        _require(all(0 < int(value) <= 40960 for value in contexts), f"{group.group}: context exceeds Qwen3 limit")
        seeds = list(group.request.get("axes", {}).get("seed", []))
        _require(len(seeds) == len(set(seeds)), f"{group.group}: duplicate seeds")
        _require(
            len(seeds) >= int(paper_minimum_seeds.get(group.group, 1)),
            f"{group.group}: insufficient paper seeds",
        )
        if workload["profile"] == "weka_agentx_causal":
            predictor = workload["causal_slack_predictor"]
            _require(
                workload["arrival_model"]
                == "recorded_intra_workflow_fixed_schedule_plus_fixed_workflow_starts",
                f"{group.group}: Weka replay must use the declared fixed schedule",
            )
            _require(
                predictor["future_information"] is False,
                f"{group.group}: future information is forbidden",
            )
            _require(
                predictor["calibration_unit"]
                == "workflow_max_absolute_residual",
                f"{group.group}: request-level calibration is not independent",
            )
            _require(
                float(predictor["interval_coverage"]) >= 0.95,
                f"{group.group}: Weka simultaneous interval target is too low",
            )
            _require(
                predictor["unmodeled_request_behavior"] == "fail_close_zero",
                f"{group.group}: unavailable Slack must fail close",
            )
            _require(
                workload["sampling_mode"]
                == "disjoint_seed_partition_without_replacement",
                f"{group.group}: Weka seed samples must be disjoint",
            )

        failure = group.failure["fault_model"]
        hazard = group.failure["hazard_weighting"]
        _require(failure["type"] == "fail_stop", f"{group.group}: unsupported failure type")
        _require(hazard["generated_as_poisson_in_short_run"] is False, f"{group.group}: short-run field Poisson forbidden")
        _require(float(hazard["central_mttf_gpu_hours"]) >= 1000, f"{group.group}: implausibly short field MTTF")
        _require(
            float(hazard["central_mttf_gpu_hours"]) == 9600.0,
            f"{group.group}: A100 MTTF must preserve the cited 400-day central value",
        )

        deployment = group.deployment
        models = deployment["models"]
        _require(models["qwen3-8b"]["kv_bytes_per_token"] == 147456, "Qwen3-8B KV bytes mismatch")
        _require(models["qwen3-32b"]["kv_bytes_per_token"] == 262144, "Qwen3-32B KV bytes mismatch")
        _require(models["qwen3-32b"]["tp_size"] == 4, "Qwen3-32B must use TP4 on A100-40GB")
        expected_prefix_chunk = 1024 if group.group == "S9" else 4096
        _require(
            int(deployment["protection"]["prefix_chunk_tokens"])
            == expected_prefix_chunk,
            f"{group.group}: prefix chunk does not match the current-design contract",
        )
        _require(
            int(deployment["protection"]["decode_chunk_tokens"]) == 256,
            f"{group.group}: decode chunk does not match the current-design contract",
        )
        _require(deployment["cluster"]["aegiskv_host_dram_gib_per_node"] <= deployment["cluster"]["host_memory_gib_per_node"], "AegisKV DRAM exceeds host memory")
        _require(deployment["evidence"]["physical_result"] is False, "metadata DES mislabeled physical")
        _require(
            float(deployment["cluster"]["effective_cross_node_bandwidth_GB_per_s"])
            <= float(deployment["cluster"]["network_line_rate_Gbit_per_s"]) / 8.0,
            "effective cross-node bandwidth exceeds line rate",
        )
        policies = deployment["policies"]
        for policy in deployment["axes"].get("policy", []):
            _require(policy in policies, f"{group.group}: unknown policy {policy}")


def matrix_rows(group: GroupConfig, suite: Mapping[str, Any], profile: str) -> list[dict[str, Any]]:
    axes = group.axes
    profile_config = suite["profiles"][profile]
    limit = profile_config.get("max_axis_values")
    names = sorted(axes)
    values: list[list[Any]] = []
    for name in names:
        rows = list(axes[name])
        if limit is not None and name != "policy":
            rows = rows[: int(limit)]
        values.append(rows)
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def expected_cell_count(loaded: SuiteConfig, profile: str) -> int:
    return sum(len(matrix_rows(group, loaded.suite, profile)) for group in loaded.groups)


def iter_config_files(config_root: Path = CONFIG_ROOT) -> Iterable[Path]:
    yield Path(config_root) / "suite.json"
    yield Path(config_root) / "common" / "request.json"
    yield Path(config_root) / "common" / "failure.json"
    yield Path(config_root) / "common" / "deployment.json"
    for group in read_json(Path(config_root) / "suite.json")["groups"]:
        for filename in LAYER_FILENAMES:
            yield Path(config_root) / str(group) / filename


__all__ = [
    "CONFIG_ROOT",
    "EXPERIMENT_ROOT",
    "PARAMETER_PROVENANCE_PATH",
    "GroupConfig",
    "SuiteConfig",
    "canonical_json_bytes",
    "expected_cell_count",
    "iter_config_files",
    "load_group",
    "load_suite",
    "matrix_rows",
    "sha256_bytes",
    "sha256_file",
    "validate_suite",
]
