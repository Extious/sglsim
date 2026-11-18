"""Validate the request/failure/deployment configuration for S1--S9.

The three files are experiment authoring surfaces.  Frozen result CSVs are
declared by deployment.json so rendering has a content-addressed input trail.
They are not relabeled as a fresh simulator or physical run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "aegiskv.s1-s9-three-layer.v1"
DESIGN_SEMANTICS = "source_owned_independent_hbm_dram_v2"
GROUPS = tuple(f"S{i}" for i in range(1, 10))
SOURCE_FILENAMES = ("request.json", "failure.json", "deployment.json")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG_ROOT = HERE / "configs"

_COMMON_KEYS = {"schema_version", "group", "layer", "description"}
_ALLOWED_KEYS = {
    "request": _COMMON_KEYS
    | {"workload", "axes", "matrix_note"},
    "failure": _COMMON_KEYS
    | {"fault_model", "axes", "components", "matrix_note"},
    "deployment": _COMMON_KEYS
    | {
        "topology",
        "policy",
        "axes",
        "planned_cells",
        "unique_cells",
        "reuse_from",
        "execution",
        "evidence",
        "matrix_components",
        "matrix_note",
    },
}


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing configuration: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def _validate_layer(raw: Mapping[str, Any], *, group: str, layer: str) -> None:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{group}/{layer}.json must declare schema_version={SCHEMA_VERSION!r}"
        )
    if raw.get("group") != group:
        raise ValueError(f"{group}/{layer}.json has mismatched group={raw.get('group')!r}")
    if raw.get("layer") != layer:
        raise ValueError(f"{group}/{layer}.json has mismatched layer={raw.get('layer')!r}")
    unknown = sorted(set(raw) - _ALLOWED_KEYS[layer])
    if unknown:
        raise ValueError(f"unknown {group}/{layer} keys: {unknown}")


def _axis_product(axes: Any, *, owner: str) -> int:
    if axes is None:
        return 1
    if not isinstance(axes, dict):
        raise ValueError(f"{owner}.axes must be an object")
    result = 1
    for name, values in axes.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{owner}.axes contains an invalid name")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{owner}.axes.{name} must be a non-empty list")
        result *= len(values)
    return result


def _failure_factor(raw: Mapping[str, Any], *, group: str) -> int:
    components = raw.get("components")
    if components is None:
        return _axis_product(raw.get("axes", {}), owner=f"{group}/failure")
    if raw.get("axes"):
        raise ValueError(f"{group}/failure cannot combine axes and components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"{group}/failure.components must be a non-empty list")
    total = 0
    names: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError(f"{group}/failure.components[{index}] must be an object")
        name = component.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"{group}/failure component names must be unique strings")
        names.add(name)
        total += _axis_product(
            component.get("axes", {}),
            owner=f"{group}/failure.components[{name}]",
        )
    return total


def _resolve_repo_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be repository-relative: {value}")
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the repository: {value}") from exc
    return resolved


@dataclass(frozen=True)
class GroupConfig:
    group: str
    config_dir: Path
    request: dict[str, Any]
    failure: dict[str, Any]
    deployment: dict[str, Any]
    source_csv: Path
    source_audit: Path
    source_lineage: Path | None
    source_rows: int
    audit: dict[str, Any]

    @property
    def planned_cells(self) -> int:
        return int(self.deployment["planned_cells"])

    @property
    def unique_cells(self) -> int:
        return int(self.deployment["unique_cells"])

    @property
    def config_sha256(self) -> dict[str, str]:
        return {
            name: sha256_file(self.config_dir / name) for name in SOURCE_FILENAMES
        }

    @property
    def evidence_status(self) -> str:
        for key in ("audit_status", "status"):
            value = self.audit.get(key)
            if isinstance(value, str):
                return value
        return "UNKNOWN"


def load_suite(config_root: Path = DEFAULT_CONFIG_ROOT) -> dict[str, Any]:
    suite = _read_json_object(Path(config_root) / "suite.json")
    if suite.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("suite.json has the wrong schema_version")
    if suite.get("design_semantics") != DESIGN_SEMANTICS:
        raise ValueError(
            f"suite.json must use design_semantics={DESIGN_SEMANTICS!r}"
        )
    if tuple(suite.get("groups", ())) != GROUPS:
        raise ValueError(f"suite.json groups must be exactly {list(GROUPS)}")
    return suite


def load_group(config_root: Path, group: str) -> GroupConfig:
    if group not in GROUPS:
        raise ValueError(f"unknown experiment group: {group}")
    config_dir = Path(config_root) / group
    request = _read_json_object(config_dir / "request.json")
    failure = _read_json_object(config_dir / "failure.json")
    deployment = _read_json_object(config_dir / "deployment.json")
    _validate_layer(request, group=group, layer="request")
    _validate_layer(failure, group=group, layer="failure")
    _validate_layer(deployment, group=group, layer="deployment")

    execution = deployment.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"{group}/deployment.execution must be an object")
    if execution.get("mode") != "consume_frozen_audited_result":
        raise ValueError(
            f"{group}/deployment.execution.mode must preserve frozen-result provenance"
        )
    source_csv = _resolve_repo_path(
        execution.get("source_csv"), field=f"{group}.execution.source_csv"
    )
    source_audit = _resolve_repo_path(
        execution.get("source_audit"), field=f"{group}.execution.source_audit"
    )
    lineage_raw = execution.get("source_lineage")
    source_lineage = (
        _resolve_repo_path(lineage_raw, field=f"{group}.execution.source_lineage")
        if lineage_raw
        else None
    )
    for path in (source_csv, source_audit, source_lineage):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{group} declared input does not exist: {path}")

    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = execution.get("expected_rows")
    if not isinstance(expected_rows, int) or expected_rows < 1:
        raise ValueError(f"{group}.execution.expected_rows must be a positive integer")
    if len(rows) != expected_rows:
        raise ValueError(
            f"{group} source row count changed: expected {expected_rows}, got {len(rows)}"
        )

    audit = _read_json_object(source_audit)
    expected_data_sha = audit.get("data_sha256")
    if expected_data_sha and expected_data_sha != sha256_file(source_csv):
        raise ValueError(f"{group} source CSV no longer matches its audit hash")

    planned_cells = deployment.get("planned_cells")
    unique_cells = deployment.get("unique_cells")
    if not isinstance(planned_cells, int) or planned_cells < 1:
        raise ValueError(f"{group}.planned_cells must be a positive integer")
    if not isinstance(unique_cells, int) or not 0 <= unique_cells <= planned_cells:
        raise ValueError(f"{group}.unique_cells must be within [0, planned_cells]")

    matrix_components = deployment.get("matrix_components")
    if matrix_components is not None:
        if not isinstance(matrix_components, list) or not matrix_components:
            raise ValueError(f"{group}.matrix_components must be a non-empty list")
        component_names: set[str] = set()
        computed_cells = 0
        for index, component in enumerate(matrix_components):
            if not isinstance(component, dict):
                raise ValueError(f"{group}.matrix_components[{index}] must be an object")
            name = component.get("name")
            cells = component.get("cells")
            formula = component.get("formula")
            if not isinstance(name, str) or not name or name in component_names:
                raise ValueError(f"{group}.matrix_components names must be unique strings")
            if not isinstance(cells, int) or cells < 1:
                raise ValueError(f"{group}.matrix_components[{name}].cells must be positive")
            if not isinstance(formula, str) or not formula:
                raise ValueError(f"{group}.matrix_components[{name}].formula is required")
            component_names.add(name)
            computed_cells += cells
    else:
        request_factor = _axis_product(request.get("axes", {}), owner=f"{group}/request")
        failure_factor = _failure_factor(failure, group=group)
        deployment_factor = _axis_product(
            deployment.get("axes", {}), owner=f"{group}/deployment"
        )
        computed_cells = request_factor * failure_factor * deployment_factor
    if computed_cells != planned_cells:
        raise ValueError(
            f"{group} matrix computes {computed_cells} cells, declared {planned_cells}"
        )

    evidence = deployment.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"{group}/deployment.evidence must be an object")
    if evidence.get("paper_ready") is not False:
        raise ValueError(f"{group} must not promote preliminary evidence to paper_ready")
    semantics = deployment.get("policy", {}).get("design_semantics")
    if semantics != DESIGN_SEMANTICS:
        raise ValueError(f"{group} has stale AegisKV design semantics: {semantics!r}")

    return GroupConfig(
        group=group,
        config_dir=config_dir,
        request=request,
        failure=failure,
        deployment=deployment,
        source_csv=source_csv,
        source_audit=source_audit,
        source_lineage=source_lineage,
        source_rows=len(rows),
        audit=audit,
    )


def load_all(config_root: Path = DEFAULT_CONFIG_ROOT) -> tuple[dict[str, Any], list[GroupConfig]]:
    suite = load_suite(config_root)
    groups = [load_group(config_root, group) for group in GROUPS]
    nominal = sum(group.planned_cells for group in groups)
    unique = sum(group.unique_cells for group in groups)
    if nominal != suite.get("nominal_cells_before_reuse"):
        raise ValueError(
            f"suite nominal cell mismatch: configs={nominal}, suite={suite.get('nominal_cells_before_reuse')}"
        )
    if unique != suite.get("minimum_unique_cells"):
        raise ValueError(
            f"suite unique cell mismatch: configs={unique}, suite={suite.get('minimum_unique_cells')}"
        )
    return suite, groups


def group_manifest(group: GroupConfig) -> dict[str, Any]:
    execution = group.deployment["execution"]
    evidence = group.deployment["evidence"]
    value: dict[str, Any] = {
        "group": group.group,
        "planned_cells": group.planned_cells,
        "unique_cells": group.unique_cells,
        "reuse_from": group.deployment.get("reuse_from"),
        "config_sha256": group.config_sha256,
        "source_csv": str(group.source_csv.relative_to(REPO_ROOT)),
        "source_csv_sha256": sha256_file(group.source_csv),
        "source_rows": group.source_rows,
        "source_audit": str(group.source_audit.relative_to(REPO_ROOT)),
        "source_audit_sha256": sha256_file(group.source_audit),
        "audit_status": group.evidence_status,
        "paper_ready": False,
        "evidence_class": evidence.get("class"),
        "known_gaps": list(evidence.get("known_gaps", [])),
        "plot_kind": execution.get("plot_kind"),
    }
    if group.source_lineage is not None:
        value["source_lineage"] = str(group.source_lineage.relative_to(REPO_ROOT))
        value["source_lineage_sha256"] = sha256_file(group.source_lineage)
    acceptance = group.audit.get("calibration_acceptance")
    if acceptance is not None:
        value["calibration_acceptance"] = acceptance
    return value


def validate_no_duplicate_axes(groups: Iterable[GroupConfig]) -> None:
    """Fail when the same matrix control is owned by more than one layer."""

    for group in groups:
        owners: dict[str, str] = {}
        layer_axes = {
            "request": group.request.get("axes", {}),
            "deployment": group.deployment.get("axes", {}),
        }
        if group.failure.get("components") is None:
            layer_axes["failure"] = group.failure.get("axes", {})
        for owner, axes in layer_axes.items():
            for name in axes:
                previous = owners.setdefault(name, owner)
                if previous != owner:
                    raise ValueError(
                        f"{group.group} axis {name!r} is owned by both {previous} and {owner}"
                    )
