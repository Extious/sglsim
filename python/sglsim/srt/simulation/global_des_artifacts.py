"""Atomic raw-evidence writer for one offline global-DES experiment cell.

The writer deliberately emits JSON/JSONL only.  Its outputs are modeled
simulation evidence, never measurements from real hardware.  It accepts an
already completed :class:`~sglsim.srt.simulation.global_des.GlobalDES`, audits
the selected byte-accounting contract, serializes every table canonically, and
publishes the cell directory with one final rename.

``manifest.json`` inventories every *other* file in the directory.  A manifest
cannot contain the SHA256 of its own final bytes without a recursive definition,
so it carries a deterministic ``canonical_payload_sha256`` computed over the
manifest payload before that field is inserted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sglsim.srt.simulation.global_des import GlobalDES


MANIFEST_SCHEMA = "sglsim.global-des-cell-manifest.v1"
EVIDENCE_CLASS = "modeled_offline_global_virtual_clock_simulation"
DEFAULT_ENGINE = "sglsim_global_des"

_REQUIRED_CONFIG_LAYERS = frozenset({"request", "failure", "deployment"})
_REQUIRED_FILES = (
    "events.jsonl",
    "rank_steps.jsonl",
    "gpu_hbm_timeseries.jsonl",
    "node_dram_timeseries.jsonl",
    "kv_extents.jsonl",
    "transfers.jsonl",
    "requests.jsonl",
    "failures.jsonl",
    "audit.json",
)
_TRANSFER_GROUP_FILE = "transfer_groups.jsonl"
_TRANSFER_RESOURCE_FILE = "transfer_resources.jsonl"
_RECOVERY_TRANSFER_FILE = "recovery_transfers.jsonl"
_FILE_SCHEMAS = {
    "events.jsonl": "sglsim.global-des.events.v1",
    "rank_steps.jsonl": "sglsim.global-des.rank-steps.v1",
    "gpu_hbm_timeseries.jsonl": "sglsim.global-des.gpu-hbm-timeseries.v1",
    "node_dram_timeseries.jsonl": "sglsim.global-des.node-dram-timeseries.v1",
    "kv_extents.jsonl": "sglsim.global-des.kv-extents.v1",
    "transfers.jsonl": "sglsim.global-des.transfers.v1",
    "requests.jsonl": "sglsim.global-des.requests.v1",
    "failures.jsonl": "sglsim.global-des.failures.v1",
    "audit.json": "sglsim.global-des.audit.v1",
    _TRANSFER_GROUP_FILE: "sglsim.global-des.transfer-groups.v1",
    _TRANSFER_RESOURCE_FILE: "sglsim.global-des.transfer-resources.v1",
    _RECOVERY_TRANSFER_FILE: "sglsim.global-des.recovery-transfers.v1",
}


class GlobalDESArtifactError(RuntimeError):
    """Base class for fail-closed artifact publication errors."""


class ArtifactInvariantError(GlobalDESArtifactError):
    """Raised when the DES state is not safe to publish as successful evidence."""


class ArtifactSerializationError(GlobalDESArtifactError):
    """Raised when an input is not canonical, finite JSON data."""


class ArtifactTargetExistsError(GlobalDESArtifactError):
    """Raised before writing when the requested cell path already exists."""


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode one JSON value deterministically and reject non-finite numbers."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactSerializationError(
            f"artifact contains a non-canonical JSON value: {exc}"
        ) from exc
    return (encoded + "\n").encode("utf-8")


def _reject_nonfinite(value: Any, *, path: str = "$") -> None:
    """Give a useful path for NaN/Inf instead of relying only on json.dumps."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactSerializationError(f"non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArtifactSerializationError(
                    f"JSON object key at {path} is not a string: {key!r}"
                )
            _reject_nonfinite(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, path=f"{path}[{index}]")


def _copy_json_object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactSerializationError(f"{name} must be a JSON object")
    copied = copy.deepcopy(dict(value))
    _reject_nonfinite(copied, path=name)
    # This also rejects values such as Path, Enum, bytes, and arbitrary objects.
    _canonical_json_bytes(copied)
    return copied


def _materialize_rows(
    rows: Iterable[Mapping[str, Any]], *, name: str
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ArtifactSerializationError(f"{name}[{index}] must be a JSON object")
        copied = copy.deepcopy(dict(row))
        _reject_nonfinite(copied, path=f"{name}[{index}]")
        _canonical_json_bytes(copied)
        materialized.append(copied)
    return materialized


_TIME_KEYS = (
    "timestamp_ns",
    "start_ns",
    "arrival_ns",
    "failure_ns",
    "scheduled_ns",
    "completion_ns",
    "logical_time_ns",
)
_IDENTITY_KEYS = (
    "event_id",
    "step_id",
    "request_id",
    "failure_id",
    "transfer_id",
    "recovery_id",
    "extent_id",
    "group_id",
    "node_id",
    "dp_rank",
    "pp_rank",
    "tp_rank",
)


def _stable_table_sort(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give non-journal tables a deterministic temporal/identity order."""

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        time_value: int | float = 0
        for field in _TIME_KEYS:
            candidate = row.get(field)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                time_value = candidate
                break
        identities = tuple(str(row.get(field, "")) for field in _IDENTITY_KEYS)
        return (time_value, *identities, _canonical_json_bytes(row))

    return sorted(rows, key=key)


def _prepare_events(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve authoritative same-timestamp journal order and before/after."""

    events = _materialize_rows(rows, name="events")
    for index, row in enumerate(events):
        if "timestamp_ns" not in row:
            raise ArtifactSerializationError(
                f"events[{index}] is missing timestamp_ns"
            )
        if not isinstance(row.get("before"), Mapping) or not isinstance(
            row.get("after"), Mapping
        ):
            raise ArtifactSerializationError(
                f"events[{index}] must preserve object-valued before and after"
            )
    # Python's sort is stable.  The journal's insertion order is the DES's
    # authoritative ordering for all side effects sharing a timestamp.
    return sorted(events, key=lambda row: row["timestamp_ns"])


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_code_hash(lineage: Mapping[str, Any]) -> str:
    for key in ("code_hash", "simulator_git_revision"):
        value = lineage.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ArtifactSerializationError(
        "lineage_metadata requires nonempty code_hash or simulator_git_revision"
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_global_des_artifacts(
    cell_dir: str | os.PathLike[str],
    *,
    des: GlobalDES,
    rank_steps: Iterable[Mapping[str, Any]],
    requests: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    resolved_config: Mapping[str, Any],
    lineage_metadata: Mapping[str, Any],
    audit_details: Mapping[str, Any] | None = None,
    extent_accounting_mode: str = "strict",
    supplemental_audit_errors: Iterable[str] = (),
    recovery_transfers: Iterable[Mapping[str, Any]] = (),
    ordinary_hicache_transfers: Mapping[str, Any] | None = None,
) -> Path:
    """Publish all raw evidence for one successful global-virtual-time cell.

    ``resolved_config`` must contain exactly the request, failure, and
    deployment layers.  ``lineage_metadata`` must identify the code revision;
    it may override the default engine name with a nonempty ``engine`` string.

    The target path must not exist.  All validation and serialization happens
    before a sibling temporary directory is created.  Publication is one
    directory rename, so a failed audit or write never leaves a success
    manifest at ``cell_dir``.
    """

    if not isinstance(des, GlobalDES):
        raise TypeError("des must be a GlobalDES instance")

    target = Path(cell_dir)
    if os.path.lexists(os.fspath(target)):
        raise ArtifactTargetExistsError(
            f"cell directory already exists and will not be overwritten: {target}"
        )

    config = _copy_json_object(resolved_config, name="resolved_config")
    if set(config) != _REQUIRED_CONFIG_LAYERS:
        raise ArtifactSerializationError(
            "resolved_config must contain exactly request, failure, and deployment"
        )
    lineage = _copy_json_object(lineage_metadata, name="lineage_metadata")
    details = (
        {}
        if audit_details is None
        else _copy_json_object(audit_details, name="audit_details")
    )
    code_hash = _resolve_code_hash(lineage)
    engine_value = lineage.get("engine", DEFAULT_ENGINE)
    if not isinstance(engine_value, str) or not engine_value.strip():
        raise ArtifactSerializationError("lineage_metadata.engine must be nonempty")
    engine = engine_value.strip()
    config_sha256 = _sha256(_canonical_json_bytes(config))

    rank_step_rows = _stable_table_sort(
        _materialize_rows(rank_steps, name="rank_steps")
    )
    request_rows = _stable_table_sort(_materialize_rows(requests, name="requests"))
    failure_rows = _stable_table_sort(_materialize_rows(failures, name="failures"))

    mode = str(extent_accounting_mode)
    if mode not in {"strict", "external_aggregate", "scheduler_aggregate"}:
        raise ArtifactSerializationError(
            "extent_accounting_mode must be strict, external_aggregate, or "
            "scheduler_aggregate"
        )
    ordinary_artifact = (
        {}
        if ordinary_hicache_transfers is None
        else _copy_json_object(
            ordinary_hicache_transfers,
            name="ordinary_hicache_transfers",
        )
    )
    if ordinary_artifact and ordinary_artifact.get("schema") != (
        "sglsim.offline.hicache-physical-transfer-bridge.v1"
    ):
        raise ArtifactSerializationError(
            "ordinary_hicache_transfers has an unsupported schema"
        )
    ordinary_errors = [
        str(error) for error in ordinary_artifact.get("audit_errors", ())
    ]
    extra_errors = [
        *(str(error) for error in supplemental_audit_errors),
        *ordinary_errors,
    ]
    if any(not error for error in extra_errors):
        raise ArtifactSerializationError(
            "supplemental_audit_errors must contain nonempty strings"
        )
    # Real Scheduler snapshots own primary/prefix allocator aggregates and do
    # not emit one KvExtent per retained radix page.  In that integration mode
    # OfflineResourceBridge performs the byte-exact projection audit (passed
    # through supplemental_audit_errors); GlobalDES still checks every
    # structural/failure/transfer invariant without inventing aggregate
    # extents after the run.
    des_errors = (
        des.audit_invariants()
        if mode == "scheduler_aggregate"
        else des.audit_invariants(extent_accounting_mode=mode)
    )
    audit_errors = [*des_errors, *extra_errors]
    if audit_errors:
        raise ArtifactInvariantError(
            "global DES invariant audit failed: " + "; ".join(audit_errors)
        )

    exported = des.export_timeseries()
    events = _prepare_events(exported.get("journal", ()))
    gpu_hbm = _stable_table_sort(
        _materialize_rows(exported.get("gpu_hbm", ()), name="gpu_hbm_timeseries")
    )
    node_dram = _stable_table_sort(
        _materialize_rows(
            exported.get("node_dram", ()), name="node_dram_timeseries"
        )
    )
    kv_extents = _stable_table_sort(
        _materialize_rows(exported.get("kv_extents", ()), name="kv_extents")
    )
    global_transfer_rows = _materialize_rows(
        exported.get("transfers", ()), name="transfers"
    )
    for row in global_transfer_rows:
        row.setdefault("transfer_class", "global_des_kv")
    ordinary_transfer_rows = _materialize_rows(
        ordinary_artifact.get("transfers", ()),
        name="ordinary_hicache_transfers.transfers",
    )
    for row in ordinary_transfer_rows:
        row["transfer_class"] = "ordinary_hicache"
    transfers = _stable_table_sort(
        [*global_transfer_rows, *ordinary_transfer_rows]
    )
    # Protection and recovery use different canonical group tables in the
    # kernel.  Export both: omitting ``recovery_load_groups`` would make a
    # successful PP x TP recovery impossible to audit from immutable evidence.
    protection_group_rows = _materialize_rows(
        exported.get("transfer_groups", exported.get("commit_groups", ())),
        name="transfer_groups.protection",
    )
    for row in protection_group_rows:
        row.setdefault("group_class", "aegis_protection")
    recovery_group_rows = _materialize_rows(
        exported.get("recovery_load_groups", ()),
        name="transfer_groups.recovery_load",
    )
    for row in recovery_group_rows:
        row["group_class"] = "aegis_recovery_load"
    ordinary_group_rows = _materialize_rows(
        ordinary_artifact.get("groups", ()),
        name="ordinary_hicache_transfers.groups",
    )
    for row in ordinary_group_rows:
        row["group_class"] = "ordinary_hicache"
        # Promote the nested logical ID for deterministic table consumers.
        intent = row.get("intent")
        if isinstance(intent, Mapping):
            row.setdefault("group_id", str(intent.get("intent_id") or ""))
    transfer_groups = _stable_table_sort(
        [
            *protection_group_rows,
            *recovery_group_rows,
            *ordinary_group_rows,
        ]
    )
    transfer_resources = _stable_table_sort(
        _materialize_rows(
            exported.get("resources", ()), name="transfer_resources"
        )
    )
    recovery_transfer_rows = _stable_table_sort(
        _materialize_rows(
            recovery_transfers, name="recovery_transfers"
        )
    )
    ordinary_audit_rows = _stable_table_sort(
        _materialize_rows(
            ordinary_artifact.get("audit_rows", ()),
            name="ordinary_hicache_transfers.audit_rows",
        )
    )
    if ordinary_artifact:
        details["ordinary_hicache"] = {
            "schema": ordinary_artifact.get("schema"),
            "accounting": copy.deepcopy(ordinary_artifact.get("accounting", {})),
            "audit_rows": ordinary_audit_rows,
        }

    audit = {
        "schema": _FILE_SCHEMAS["audit.json"],
        "evidence_class": EVIDENCE_CLASS,
        "engine": engine,
        "code_hash": code_hash,
        "config_sha256": config_sha256,
        "strict_extent_accounting": mode == "strict",
        "extent_accounting_mode": mode,
        "scheduler_aggregate_accounting": mode == "scheduler_aggregate",
        "passed": True,
        "errors": [],
        "global_now_ns": des.now_ns,
        "queued_event_count": des.snapshot()["queued_events"],
        "details": details,
    }

    jsonl_tables: dict[str, Sequence[Mapping[str, Any]]] = {
        "events.jsonl": events,
        "rank_steps.jsonl": rank_step_rows,
        "gpu_hbm_timeseries.jsonl": gpu_hbm,
        "node_dram_timeseries.jsonl": node_dram,
        "kv_extents.jsonl": kv_extents,
        "transfers.jsonl": transfers,
        "requests.jsonl": request_rows,
        "failures.jsonl": failure_rows,
        _TRANSFER_GROUP_FILE: transfer_groups,
        _TRANSFER_RESOURCE_FILE: transfer_resources,
        _RECOVERY_TRANSFER_FILE: recovery_transfer_rows,
    }
    payloads = {name: _jsonl_bytes(rows) for name, rows in jsonl_tables.items()}
    payloads["audit.json"] = _canonical_json_bytes(audit)

    common_descriptor = {
        "evidence_class": EVIDENCE_CLASS,
        "engine": engine,
        "code_hash": code_hash,
        "config_sha256": config_sha256,
    }
    files: dict[str, dict[str, Any]] = {}
    optional_files = (
        _TRANSFER_GROUP_FILE,
        _TRANSFER_RESOURCE_FILE,
        _RECOVERY_TRANSFER_FILE,
    )
    for name in (*_REQUIRED_FILES, *optional_files):
        row_count = 1 if name == "audit.json" else len(jsonl_tables[name])
        files[name] = {
            "sha256": _sha256(payloads[name]),
            "row_count": row_count,
            "schema": _FILE_SCHEMAS[name],
            **common_descriptor,
        }

    manifest_payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "success",
        **common_descriptor,
        "resolved_config": config,
        "lineage": lineage,
        "simulation": {
            "clock": "global_virtual_time",
            "timestamp_unit": "ns",
            "now_ns": des.now_ns,
            "real_hardware_measurement": False,
        },
        "files": files,
        "optional_files": list(optional_files),
    }
    manifest_payload["canonical_payload_sha256"] = _sha256(
        _canonical_json_bytes(manifest_payload)
    )
    manifest_bytes = _canonical_json_bytes(manifest_payload)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(
        prefix=f".{target.name}.tmp-", dir=os.fspath(target.parent)
    )
    temp_path = Path(temporary.name)
    try:
        for name in (*_REQUIRED_FILES, *optional_files):
            _write_bytes(temp_path / name, payloads[name])
        # A manifest is written last even inside the unpublished temporary cell.
        _write_bytes(temp_path / "manifest.json", manifest_bytes)
        if os.path.lexists(os.fspath(target)):
            raise ArtifactTargetExistsError(
                f"cell directory appeared during publication: {target}"
            )
        os.rename(temp_path, target)
    finally:
        # TemporaryDirectory removes only the exact sibling directory it made;
        # after a successful rename that old path no longer exists.
        temporary.cleanup()

    return target


__all__ = [
    "ArtifactInvariantError",
    "ArtifactSerializationError",
    "ArtifactTargetExistsError",
    "EVIDENCE_CLASS",
    "GlobalDESArtifactError",
    "MANIFEST_SCHEMA",
    "write_global_des_artifacts",
]
