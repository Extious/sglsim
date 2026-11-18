"""Fail-closed aggregation and plotting for the AegisKV P1--P12 DES suite.

The module intentionally reads only immutable scheduler-run artifacts.  A plot
is emitted only after every source manifest and raw file has passed its SHA256
check.  Registered P1--P12 figures additionally require the *entire* frozen
axis product and all seed/baseline pairs; an incomplete smoke run can therefore
produce mechanism diagnostics, but never a partially populated paper figure.

No missing value is replaced by zero, interpolated, or borrowed from the old
S1--S9 613-cell lineage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


PLOT_SCHEMA = "aegiskv.global-des-suite-plots.v1"
FIGURE_SCHEMA = "aegiskv.global-des-suite-figure.v1"
SUITE_SCHEMA = "aegiskv.global-des-scheduler-suite-run.v1"
CELL_SCHEMA = "sglsim.global-des-cell-manifest.v1"
AUDIT_SCHEMA = "sglsim.global-des.audit.v1"
GROUPS = tuple(f"P{index}" for index in range(1, 13))
GIB = 1024**3


class PlotEvidenceError(RuntimeError):
    """The supplied evidence is corrupt, ambiguous, or outside the contract."""


class FigureDataError(RuntimeError):
    """A complete cell grid lacks a metric required by its registered figure."""


def _canonical_bytes(value: Any) -> bytes:
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlotEvidenceError(f"cannot read canonical JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlotEvidenceError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PlotEvidenceError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlotEvidenceError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _axis_token(value: Any) -> tuple[str, str]:
    """Keep JSON type in cell identities (``1`` must not equal ``1.0``)."""

    return (type(value).__name__, json.dumps(value, sort_keys=True))


def _identity(group: str, axes: Mapping[str, Any]) -> tuple[Any, ...]:
    return (group, *((key, _axis_token(value)) for key, value in sorted(axes.items())))


@dataclass
class CellEvidence:
    run_dir: Path
    cell_dir: Path
    suite_record: dict[str, Any]
    manifest: dict[str, Any]
    audit: dict[str, Any]
    group: str
    axes: dict[str, Any]
    claims: dict[str, Any]
    _tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def cell_id(self) -> str:
        return str(self.manifest["lineage"]["cell_id"])

    @property
    def request_universe_hash(self) -> str:
        return str(self.manifest["lineage"].get("request_universe_hash", ""))

    @property
    def compiled_requests_hash(self) -> str:
        return str(self.manifest["lineage"].get("compiled_requests_hash", ""))

    def rows(self, name: str) -> list[dict[str, Any]]:
        if name not in self.manifest.get("files", {}):
            raise FigureDataError(f"{self.cell_id} has no declared {name}")
        if name not in self._tables:
            if not name.endswith(".jsonl"):
                raise FigureDataError(f"{name} is not a JSONL table")
            self._tables[name] = _load_jsonl(self.cell_dir / name)
        return self._tables[name]


@dataclass(frozen=True)
class FrozenGroup:
    group: str
    axes: dict[str, tuple[Any, ...]]
    expected_cells: int

    def expected_identities(self) -> set[tuple[Any, ...]]:
        keys = tuple(sorted(self.axes))
        return {
            _identity(self.group, dict(zip(keys, values)))
            for values in itertools.product(*(self.axes[key] for key in keys))
        }


@dataclass
class EvidenceCollection:
    cells: list[CellEvidence]
    source_runs: list[dict[str, Any]]
    frozen_groups: dict[str, FrozenGroup]
    suite_id: str

    def by_group(self, group: str) -> list[CellEvidence]:
        return [cell for cell in self.cells if cell.group == group]


def load_frozen_groups(config_root: Path) -> tuple[str, dict[str, FrozenGroup]]:
    suite = _load_json(config_root / "suite.json")
    suite_id = str(suite.get("suite_id", ""))
    if suite.get("schema_version") != "aegiskv.global-des-suite.v3" or not suite_id:
        raise PlotEvidenceError("plotter requires the frozen global-DES v3 suite")
    expected_counts = {
        str(row["id"]): int(row["expected_cells"])
        for row in suite.get("groups", [])
        if isinstance(row, dict) and "id" in row and "expected_cells" in row
    }
    frozen: dict[str, FrozenGroup] = {}
    for group in GROUPS:
        axes: dict[str, tuple[Any, ...]] = {}
        for layer in ("request", "failure", "deployment"):
            payload = _load_json(config_root / group / f"{layer}.json")
            if payload.get("group") != group:
                raise PlotEvidenceError(f"{group}/{layer}.json has wrong group")
            layer_axes = payload.get("axes", {})
            if not isinstance(layer_axes, dict):
                raise PlotEvidenceError(f"{group}/{layer}.json axes must be an object")
            for key, values in layer_axes.items():
                if key in axes:
                    raise PlotEvidenceError(f"axis {group}.{key} has multiple owners")
                if not isinstance(values, list) or not values:
                    raise PlotEvidenceError(f"axis {group}.{key} must be nonempty")
                axes[str(key)] = tuple(values)
        product = math.prod(len(values) for values in axes.values())
        expected = expected_counts.get(group)
        if expected != product:
            raise PlotEvidenceError(
                f"frozen {group} count mismatch: suite={expected}, axes={product}"
            )
        frozen[group] = FrozenGroup(group, axes, product)
    return suite_id, frozen


def _verify_cell(
    run_dir: Path,
    record: Mapping[str, Any],
    *,
    expected_suite_id: str,
) -> CellEvidence:
    group = str(record.get("group", ""))
    cell_id = str(record.get("cell_id", ""))
    if group not in GROUPS or not cell_id:
        raise PlotEvidenceError(f"invalid suite cell record in {run_dir}")
    if record.get("status") != "executed":
        raise PlotEvidenceError(f"non-executed suite cell cannot be plotted: {cell_id}")
    cell_dir = run_dir / group / cell_id
    manifest_path = cell_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PlotEvidenceError(f"missing cell manifest: {manifest_path}")
    recorded_path = Path(str(record.get("manifest", ""))).resolve()
    if recorded_path != manifest_path.resolve():
        raise PlotEvidenceError(
            f"suite manifest path escapes/cross-links the run: {cell_id}"
        )
    manifest = _load_json(manifest_path)
    payload = dict(manifest)
    claimed_payload_hash = str(payload.pop("canonical_payload_sha256", ""))
    actual_payload_hash = _sha256_bytes(_canonical_bytes(payload))
    if not claimed_payload_hash or claimed_payload_hash != actual_payload_hash:
        raise PlotEvidenceError(f"cell manifest payload hash mismatch: {cell_id}")
    if record.get("manifest_payload_sha256") != claimed_payload_hash:
        raise PlotEvidenceError(f"suite/cell manifest hash mismatch: {cell_id}")
    if manifest.get("schema") != CELL_SCHEMA:
        raise PlotEvidenceError(f"unsupported cell manifest schema: {cell_id}")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, dict):
        raise PlotEvidenceError(f"missing lineage: {cell_id}")
    if (
        lineage.get("cell_id") != cell_id
        or lineage.get("group") != group
        or lineage.get("suite_id") != expected_suite_id
    ):
        raise PlotEvidenceError(f"cell lineage identity mismatch: {cell_id}")
    axes = lineage.get("axes")
    claims = lineage.get("claims")
    if not isinstance(axes, dict) or not isinstance(claims, dict):
        raise PlotEvidenceError(f"cell axes/claims are absent: {cell_id}")
    files = manifest.get("files")
    if not isinstance(files, dict) or "audit.json" not in files:
        raise PlotEvidenceError(f"cell file inventory is incomplete: {cell_id}")
    for name, file_record in files.items():
        if Path(name).name != name or not isinstance(file_record, dict):
            raise PlotEvidenceError(f"unsafe file inventory entry in {cell_id}: {name}")
        path = cell_dir / name
        if not path.is_file():
            raise PlotEvidenceError(f"missing declared raw file: {path}")
        if _sha256_file(path) != file_record.get("sha256"):
            raise PlotEvidenceError(f"raw file SHA256 mismatch: {path}")
        if name.endswith(".jsonl"):
            with path.open("rb") as handle:
                row_count = sum(1 for line in handle if line.strip())
            if row_count != file_record.get("row_count"):
                raise PlotEvidenceError(f"raw row count mismatch: {path}")
    audit = _load_json(cell_dir / "audit.json")
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("passed") is not True:
        raise PlotEvidenceError(f"cell audit is not a passing DES audit: {cell_id}")
    details = audit.get("details")
    if not isinstance(details, dict) or details.get("real_scheduler_integration") is not True:
        raise PlotEvidenceError(f"cell lacks real-Scheduler integration evidence: {cell_id}")
    if details.get("cell_id") not in (None, cell_id):
        raise PlotEvidenceError(f"audit/lineage cell id disagrees: {cell_id}")
    if details.get("axes") not in (None, axes):
        raise PlotEvidenceError(f"audit/lineage axes disagree: {cell_id}")
    if details.get("claims") != claims:
        raise PlotEvidenceError(f"audit/lineage claims disagree: {cell_id}")
    return CellEvidence(
        run_dir=run_dir,
        cell_dir=cell_dir,
        suite_record=dict(record),
        manifest=manifest,
        audit=audit,
        group=group,
        axes=dict(axes),
        claims=dict(claims),
    )


def collect_evidence(
    run_dirs: Sequence[Path],
    *,
    config_root: Path,
) -> EvidenceCollection:
    if not run_dirs:
        raise PlotEvidenceError("at least one immutable run directory is required")
    suite_id, frozen = load_frozen_groups(config_root)
    cells: list[CellEvidence] = []
    source_runs: list[dict[str, Any]] = []
    seen_runs: set[Path] = set()
    seen_cells: dict[tuple[Any, ...], CellEvidence] = {}
    for supplied in run_dirs:
        run_dir = supplied.resolve()
        if run_dir in seen_runs:
            raise PlotEvidenceError(f"duplicate run directory: {run_dir}")
        seen_runs.add(run_dir)
        suite_path = run_dir / "suite_manifest.json"
        suite = _load_json(suite_path)
        if suite.get("schema") != SUITE_SCHEMA or suite.get("complete") is not True:
            raise PlotEvidenceError(f"run is not a complete immutable publication: {run_dir}")
        if suite.get("suite_id") != suite_id:
            raise PlotEvidenceError(f"suite id mismatch: {run_dir}")
        records = suite.get("cells")
        if not isinstance(records, list) or len(records) != suite.get("processed_cells"):
            raise PlotEvidenceError(f"suite cell inventory/count mismatch: {run_dir}")
        run_cells: list[str] = []
        for record in records:
            if not isinstance(record, dict):
                raise PlotEvidenceError(f"invalid suite cell record: {run_dir}")
            cell = _verify_cell(run_dir, record, expected_suite_id=suite_id)
            spec = frozen[cell.group]
            if set(cell.axes) != set(spec.axes):
                raise PlotEvidenceError(
                    f"{cell.cell_id} axes differ from frozen {cell.group} axes"
                )
            for key, value in cell.axes.items():
                if _axis_token(value) not in {_axis_token(v) for v in spec.axes[key]}:
                    raise PlotEvidenceError(
                        f"{cell.cell_id} has out-of-plan axis {key}={value!r}"
                    )
            identity = _identity(cell.group, cell.axes)
            if identity in seen_cells:
                other = seen_cells[identity]
                raise PlotEvidenceError(
                    "duplicate cell identity across immutable runs: "
                    f"{other.cell_id} and {cell.cell_id}"
                )
            seen_cells[identity] = cell
            cells.append(cell)
            run_cells.append(cell.cell_id)
        source_runs.append(
            {
                "path": str(run_dir),
                "suite_manifest_sha256": _sha256_file(suite_path),
                "claims": dict(suite.get("claims") or {}),
                "cell_ids": sorted(run_cells),
            }
        )
    return EvidenceCollection(cells, source_runs, frozen, suite_id)


def _pairing_contract(group: str) -> tuple[str | None, str | None]:
    return {
        "P1": ("policy", "recompute"),
        "P2": ("policy", "recompute"),
        "P3": ("policy", "recompute"),
        "P4": ("algorithm_variant", "full"),
        "P5": (None, None),
        "P6": ("policy", "recompute"),
        "P7": (None, None),
        "P8": ("policy", "recompute"),
        "P9": ("policy", "no_protection_control"),
        "P10": (None, None),
        "P11": ("policy", "recompute"),
        "P12": (None, None),
    }[group]


def group_readiness(
    collection: EvidenceCollection, group: str
) -> tuple[bool, list[str]]:
    cells = collection.by_group(group)
    spec = collection.frozen_groups[group]
    blockers: list[str] = []
    expected = spec.expected_identities()
    observed = {_identity(group, cell.axes) for cell in cells}
    missing = expected - observed
    if missing:
        blockers.append(
            f"missing_cells:{len(missing)}/{spec.expected_cells}; "
            f"observed={len(observed)}"
        )
    extras = observed - expected
    if extras:
        blockers.append(f"out_of_plan_cells:{len(extras)}")
    axis, baseline = _pairing_contract(group)
    if axis is not None:
        buckets: dict[tuple[Any, ...], dict[str, CellEvidence]] = defaultdict(dict)
        for cell in cells:
            pair_axes = {key: value for key, value in cell.axes.items() if key != axis}
            buckets[_identity(group, pair_axes)][str(cell.axes[axis])] = cell
        treatments = tuple(str(value) for value in spec.axes[axis])
        for pair_key, bucket in sorted(buckets.items(), key=lambda item: str(item[0])):
            if baseline not in bucket:
                blockers.append(f"missing_baseline:{baseline}:{pair_key}")
                continue
            for treatment in treatments:
                if treatment not in bucket:
                    blockers.append(f"missing_pair:{treatment}:{pair_key}")
                    continue
                left = bucket[baseline]
                right = bucket[treatment]
                if (
                    not left.request_universe_hash
                    or left.request_universe_hash != right.request_universe_hash
                ):
                    blockers.append(
                        f"request_universe_pair_mismatch:{left.cell_id}:{right.cell_id}"
                    )
                if (
                    not left.compiled_requests_hash
                    or left.compiled_requests_hash != right.compiled_requests_hash
                ):
                    blockers.append(
                        f"compiled_requests_pair_mismatch:{left.cell_id}:{right.cell_id}"
                    )
    # Every frozen seed must be represented; this is stated separately so a
    # reviewer can distinguish absent seeds from another absent axis.
    seed_values = spec.axes.get("seed", ())
    observed_seeds = {_axis_token(cell.axes.get("seed")) for cell in cells}
    missing_seeds = [seed for seed in seed_values if _axis_token(seed) not in observed_seeds]
    if missing_seeds:
        blockers.append("missing_seeds:" + ",".join(str(seed) for seed in missing_seeds))
    # Workload identity is invariant across policy, failure, topology, and
    # resource axes.  Request-owned axes form the bucket key.  P5/P6/P10 also
    # intentionally share a universe across their phase/reuse/capacity sweep.
    universe_axes = {
        "P1": ("seed",),
        "P2": ("seed", "load_multiplier"),
        "P3": ("seed", "context_tokens"),
        "P4": ("seed",),
        "P5": ("seed",),
        "P6": ("seed",),
        "P7": ("seed",),
        "P8": ("seed",),
        "P9": ("seed", "load_multiplier"),
        "P10": ("seed",),
        "P11": ("seed", "dag_stratum"),
        "P12": (),
    }[group]
    universe_buckets: dict[tuple[Any, ...], list[CellEvidence]] = defaultdict(list)
    for cell in cells:
        if universe_axes:
            universe_buckets[
                _identity(group, {key: cell.axes[key] for key in universe_axes})
            ].append(cell)
    for bucket, bucket_cells in universe_buckets.items():
        hashes = {cell.request_universe_hash for cell in bucket_cells}
        if "" in hashes or len(hashes) != 1:
            blockers.append(f"request_universe_sweep_mismatch:{bucket}")
    return not blockers, blockers


def _number(value: Any, *, name: str, cell: CellEvidence) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigureDataError(f"{cell.cell_id} missing numeric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise FigureDataError(f"{cell.cell_id} has non-finite {name}")
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise FigureDataError("percentile of empty evidence is undefined")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _workflow_wct(cell: CellEvidence) -> dict[str, float]:
    bounds: dict[str, list[float]] = {}
    for row in cell.rows("requests.jsonl"):
        if str(row.get("status", "")) != "completed":
            continue
        stats = row.get("request_stats")
        if not isinstance(stats, dict):
            raise FigureDataError(f"{cell.cell_id} request lacks request_stats")
        workflow = str(row.get("workflow") or row.get("session") or "")
        if not workflow:
            raise FigureDataError(f"{cell.cell_id} request lacks workflow identity")
        created = _number(stats.get("created_time", row.get("arrival_ns", 0) / 1e9), name="created_time", cell=cell)
        completed = _number(stats.get("last_event_time"), name="last_event_time", cell=cell)
        if completed < created:
            raise FigureDataError(f"{cell.cell_id} has negative workflow duration")
        if workflow not in bounds:
            bounds[workflow] = [created, completed]
        else:
            bounds[workflow][0] = min(bounds[workflow][0], created)
            bounds[workflow][1] = max(bounds[workflow][1], completed)
    if not bounds:
        raise FigureDataError(f"{cell.cell_id} has no completed workflow")
    return {key: value[1] - value[0] for key, value in sorted(bounds.items())}


def _integral(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str], value_fields: Sequence[str], end_ns: int) -> float:
    grouped: dict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        timestamp = row.get("timestamp_ns")
        if not isinstance(timestamp, int):
            continue
        key = tuple(row.get(field) for field in key_fields)
        value = sum(float(row.get(field, 0)) for field in value_fields)
        grouped[key].append((timestamp, value))
    total_byte_ns = 0.0
    for samples in grouped.values():
        samples.sort()
        for index, (start, value) in enumerate(samples):
            stop = samples[index + 1][0] if index + 1 < len(samples) else end_ns
            if stop > start:
                total_byte_ns += value * (stop - start)
    return total_byte_ns / 1e6  # byte-ms


def _cell_summary(cell: CellEvidence) -> dict[str, Any]:
    workflows = _workflow_wct(cell)
    wcts = list(workflows.values())
    requests = cell.rows("requests.jsonl")
    completed = [row for row in requests if str(row.get("status", "")) == "completed"]
    first_tokens: list[float] = []
    per_output: list[float] = []
    output_tokens = 0
    starts: list[float] = []
    ends: list[float] = []
    protected_tokens = 0
    possible_tokens = 0
    recompute_tokens = 0
    replay_tokens = 0
    nonzero_slack = 0
    slack_count = 0
    for row in completed:
        stats = row.get("request_stats") or {}
        latencies = stats.get("gen_token_latencies") or []
        if latencies:
            first_tokens.append(float(latencies[0]))
            if len(latencies) > 1:
                per_output.extend(float(value) for value in latencies[1:])
        output_tokens += int(stats.get("generated_tokens", row.get("output_tokens", 0)))
        starts.append(float(stats.get("created_time", row.get("arrival_ns", 0) / 1e9)))
        ends.append(_number(stats.get("last_event_time"), name="last_event_time", cell=cell))
        protected_tokens += max(int(stats.get("aegis_safe_frontier_tokens", 0)), 0)
        possible_tokens += max(
            int(stats.get("input_length", row.get("input_tokens", 0)))
            + int(stats.get("generated_tokens", row.get("output_tokens", 0))),
            0,
        )
        recompute_tokens += max(int(stats.get("aegis_replay_recompute_tokens", 0)), 0)
        replay_tokens += max(int(stats.get("aegis_replay_restored_tokens", 0)), 0) + max(
            int(stats.get("aegis_replay_recompute_tokens", 0)), 0
        )
        simulation = row.get("simulation") or {}
        if simulation.get("aegis_slack_usable") is True:
            slack_count += 1
            if float(simulation.get("workflow_slack_ms", 0)) > 0:
                nonzero_slack += 1
    makespan = max(ends) - min(starts)
    if makespan <= 0:
        raise FigureDataError(f"{cell.cell_id} has non-positive makespan")
    transfer_rows = cell.rows("transfers.jsonl")
    final_transfers: dict[str, Mapping[str, Any]] = {}
    for row in transfer_rows:
        transfer_id = str(row.get("transfer_id", ""))
        if transfer_id:
            final_transfers[transfer_id] = row
    moved_bytes = sum(
        max(int(row.get("size_bytes", 0)), 0)
        for row in final_transfers.values()
        if str(row.get("transfer_class", "global_des_kv"))
        != "ordinary_hicache"
        and not str(row.get("recovery_group_id") or "")
    )
    recovery_bytes = sum(
        max(int(row.get("size_bytes", 0)), 0)
        for row in final_transfers.values()
        if str(row.get("transfer_class", "global_des_kv"))
        != "ordinary_hicache"
        and bool(str(row.get("recovery_group_id") or ""))
    )
    ordinary_hicache_bytes = sum(
        max(int(row.get("size_bytes", 0)), 0)
        for row in final_transfers.values()
        if str(row.get("transfer_class", "")) == "ordinary_hicache"
    )
    end_ns = max(
        [int(max(ends) * 1e9)]
        + [int(row.get("timestamp_ns", 0)) for row in cell.rows("gpu_hbm_timeseries.jsonl")]
        + [int(row.get("timestamp_ns", 0)) for row in cell.rows("node_dram_timeseries.jsonl")]
    )
    hbm_byte_ms = _integral(
        cell.rows("gpu_hbm_timeseries.jsonl"),
        ("node_id", "local_gpu_id", "dp_rank", "pp_rank", "tp_rank"),
        ("aegis_pending", "aegis_committed"),
        end_ns,
    )
    dram_byte_ms = _integral(
        cell.rows("node_dram_timeseries.jsonl"),
        ("node_id",),
        ("aegis_pending", "aegis_committed"),
        end_ns,
    )
    metrics = (cell.audit.get("details") or {}).get("metrics") or {}
    failures = metrics.get("failures") or {}
    return {
        **cell.axes,
        "cell_id": cell.cell_id,
        "workflow_count": len(wcts),
        "mean_wct_s": statistics.fmean(wcts),
        "p95_wct_s": _percentile(wcts, 0.95),
        "p99_wct_s": _percentile(wcts, 0.99),
        "mean_ttft_s": statistics.fmean(first_tokens) if first_tokens else None,
        "mean_tpot_s": statistics.fmean(per_output) if per_output else None,
        "throughput_tokens_s": output_tokens / makespan,
        "protection_bytes": moved_bytes,
        "recovery_bytes": recovery_bytes,
        "ordinary_hicache_bytes": ordinary_hicache_bytes,
        "protection_bytes_per_request": moved_bytes / len(completed),
        "hbm_aegis_byte_ms": hbm_byte_ms,
        "dram_aegis_byte_ms": dram_byte_ms,
        "protected_token_fraction": protected_tokens / possible_tokens if possible_tokens else None,
        "recompute_token_fraction": recompute_tokens / replay_tokens if replay_tokens else None,
        "slack_nonzero_fraction": nonzero_slack / slack_count if slack_count else None,
        "affected_original_requests": int(failures.get("affected_original_requests", 0)),
        "workflow_wct": workflows,
    }


def _paired_seed_rows(
    cells: Sequence[CellEvidence],
    *,
    axis: str,
    baseline: str,
) -> list[dict[str, Any]]:
    summaries = {_identity(cell.group, cell.axes): _cell_summary(cell) for cell in cells}
    index = {_identity(cell.group, cell.axes): cell for cell in cells}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        treatment = str(cell.axes[axis])
        if treatment == baseline:
            continue
        base_axes = dict(cell.axes)
        base_axes[axis] = baseline
        baseline_cell = index.get(_identity(cell.group, base_axes))
        if baseline_cell is None:
            raise FigureDataError(f"missing {baseline} pair for {cell.cell_id}")
        left = summaries[_identity(cell.group, baseline_cell.axes)]
        right = summaries[_identity(cell.group, cell.axes)]
        left_wct = left["workflow_wct"]
        right_wct = right["workflow_wct"]
        if set(left_wct) != set(right_wct):
            raise FigureDataError(
                f"workflow identities differ in pair {baseline_cell.cell_id}/{cell.cell_id}"
            )
        effects = [right_wct[key] - left_wct[key] for key in sorted(left_wct)]
        row = {
            **{key: value for key, value in cell.axes.items() if key != axis},
            axis: treatment,
            "baseline": baseline,
            "baseline_cell_id": baseline_cell.cell_id,
            "treatment_cell_id": cell.cell_id,
            "paired_workflows": len(effects),
            "mean_effect_s": statistics.fmean(effects),
            "p95_effect_s": _percentile(effects, 0.95),
        }
        rows.append(row)
    return rows


def _source_claims(cells: Sequence[CellEvidence]) -> dict[str, Any]:
    if not cells:
        return {"paper_ready": False, "reason": "no_source_cells"}
    return {
        "paper_ready": all(cell.claims.get("paper_ready") is True for cell in cells),
        "modeled_simulation": all(
            cell.claims.get("modeled_simulation") is True for cell in cells
        ),
        "real_hardware_measurement": all(
            cell.claims.get("real_hardware_measurement") is True for cell in cells
        ),
        "preliminary": any(cell.claims.get("preliminary") is True for cell in cells),
        "source_claims": [
            {"cell_id": cell.cell_id, "claims": cell.claims} for cell in cells
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise FigureDataError(f"refusing to emit empty figure CSV: {path.name}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _finish_figure(fig: Any, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    _mpl().close(fig)


def _figure_manifest(
    output: Path,
    figure_id: str,
    cells: Sequence[CellEvidence],
    *,
    figure_kind: str,
    note: str,
) -> dict[str, Any]:
    base = output / figure_id
    files = {}
    for suffix in ("csv", "png", "pdf"):
        path = base.with_suffix(f".{suffix}")
        if not path.is_file() or path.stat().st_size == 0:
            raise FigureDataError(f"figure output missing: {path}")
        files[path.name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    payload = {
        "schema": FIGURE_SCHEMA,
        "plotter_code_sha256": _sha256_file(Path(__file__)),
        "figure_id": figure_id,
        "figure_kind": figure_kind,
        "note": note,
        "claims": _source_claims(cells),
        "source_cells": [
            {
                "cell_id": cell.cell_id,
                "group": cell.group,
                "axes": cell.axes,
                "manifest_payload_sha256": cell.manifest["canonical_payload_sha256"],
            }
            for cell in cells
        ],
        "files": files,
    }
    payload["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    manifest_path = output / f"{figure_id}.manifest.json"
    manifest_path.write_bytes(_canonical_bytes(payload))
    return payload


def _simple_line_plot(
    rows: Sequence[Mapping[str, Any]],
    *,
    x: str,
    y: str,
    series: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Any:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[series])].append(row)
    for label, values in sorted(groups.items()):
        try:
            values = sorted(values, key=lambda row: float(row[x]))
        except (TypeError, ValueError):
            values = sorted(values, key=lambda row: str(row[x]))
        ax.plot([row[x] for row in values], [row[y] for row in values], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return fig


def _render_p1_ecdf(output: Path, cells: Sequence[CellEvidence]) -> tuple[str, str]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if int(_cell_summary(cell)["affected_original_requests"]) <= 0:
            raise FigureDataError(f"P1 failure did not affect an active request: {cell.cell_id}")
        for workflow, wct in _workflow_wct(cell).items():
            rows.append({**cell.axes, "cell_id": cell.cell_id, "workflow": workflow, "wct_s": wct})
    rows.sort(key=lambda row: (str(row["model"]), str(row["fault_domain"]), str(row["policy"]), float(row["wct_s"])))
    counters: dict[tuple[str, str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        totals[(str(row["model"]), str(row["fault_domain"]), str(row["policy"]))] += 1
    for row in rows:
        key = (str(row["model"]), str(row["fault_domain"]), str(row["policy"]))
        counters[key] += 1
        row["ecdf"] = counters[key] / totals[key]
    _write_csv(output / "p1_wct_ecdf.csv", rows)
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False, sharey=True)
    models = sorted({str(row["model"]) for row in rows})
    faults = sorted({str(row["fault_domain"]) for row in rows})
    for ax, (model, fault) in zip(axes.flat, itertools.product(models, faults)):
        for policy in sorted({str(row["policy"]) for row in rows}):
            subset = [row for row in rows if row["model"] == model and row["fault_domain"] == fault and row["policy"] == policy]
            if subset:
                ax.step([row["wct_s"] for row in subset], [row["ecdf"] for row in subset], where="post", label=policy)
        ax.set_title(f"{model} / {fault}")
        ax.set_xlabel("Absolute workflow WCT (s)")
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    _finish_figure(fig, output / "p1_wct_ecdf")
    return "p1_wct_ecdf", "Absolute workflow WCT; no curve shifting"


def _render_forest(
    output: Path,
    cells: Sequence[CellEvidence],
    *,
    group: str,
    axis: str,
    baseline: str,
    figure_id: str,
) -> tuple[str, str]:
    seed_rows = _paired_seed_rows(cells, axis=axis, baseline=baseline)
    _write_csv(output / f"{figure_id}.csv", seed_rows)
    dimensions = [key for key in sorted(cells[0].axes) if key not in {"seed", axis}]
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in seed_rows:
        key = tuple([row[axis], *(row[name] for name in dimensions)])
        buckets[key].append(float(row["mean_effect_s"]))
    labels: list[str] = []
    means: list[float] = []
    errors: list[float] = []
    for key, values in sorted(buckets.items(), key=lambda item: str(item[0])):
        labels.append(" / ".join(str(value) for value in key))
        means.append(statistics.fmean(values))
        errors.append(1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, max(4.0, len(labels) * 0.28)))
    positions = list(range(len(labels)))
    ax.errorbar(means, positions, xerr=errors, fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(positions, labels, fontsize=7)
    ax.set_xlabel(f"Paired {axis} - {baseline} workflow WCT (s); 95% seed CI")
    ax.set_title(f"{group} paired effect forest")
    ax.grid(axis="x", alpha=0.25)
    _finish_figure(fig, output / figure_id)
    return figure_id, "Seed-paired mean effects; CI is across frozen seeds"


def _render_group_summary(
    output: Path,
    cells: Sequence[CellEvidence],
    *,
    group: str,
) -> tuple[str, str]:
    """Render registered P2--P11 panels from exact per-cell summaries."""

    summaries = [_cell_summary(cell) for cell in cells]
    figure_id = {
        "P2": "p2_load_failure_context",
        "P3": "p3_context_scale",
        "P4": "p4_algorithm_ablation",
        "P5": "p5_alpha_budget_timeline",
        "P6": "p6_prefix_reuse",
        "P7": "p7_path_bandwidth_repair",
        "P8": "p8_fault_count_domain",
        "P9": "p9_healthy_overhead",
        "P10": "p10_capacity_heatmap",
        "P11": "p11_observed_dag_coverage",
    }[group]
    if group in {"P2", "P3", "P4", "P6", "P8", "P9", "P11"}:
        axis, baseline = _pairing_contract(group)
        assert axis and baseline
        paired = _paired_seed_rows(cells, axis=axis, baseline=baseline)
        by_cell = {row["cell_id"]: row for row in summaries}
        for row in paired:
            treatment = by_cell[row["treatment_cell_id"]]
            row.update(
                {
                    "mean_wct_s": treatment["mean_wct_s"],
                    "p95_wct_s": treatment["p95_wct_s"],
                    "protection_bytes": treatment["protection_bytes"],
                    "hbm_aegis_byte_ms": treatment["hbm_aegis_byte_ms"],
                    "dram_aegis_byte_ms": treatment["dram_aegis_byte_ms"],
                    "protected_token_fraction": treatment["protected_token_fraction"],
                    "slack_nonzero_fraction": treatment["slack_nonzero_fraction"],
                }
            )
        rows = paired
    else:
        rows = summaries
    if group == "P5":
        # A paper control-loop timeline requires actual sampled controller
        # state, never the configured initial alpha repeated as observations.
        timeline_rows: list[dict[str, Any]] = []
        for cell in cells:
            for row in cell.rows("rank_steps.jsonl"):
                candidates = {**(row.get("event_metadata") or {}), **row}
                if all(key in candidates for key in ("aegis_alpha", "aegis_quota_bytes", "aegis_used_bytes", "aegis_pressure")):
                    timeline_rows.append(
                        {
                            **cell.axes,
                            "cell_id": cell.cell_id,
                            "timestamp_ns": candidates.get("timestamp_ns", row.get("des_now_ns")),
                            "alpha": candidates["aegis_alpha"],
                            "quota_bytes": candidates["aegis_quota_bytes"],
                            "used_bytes": candidates["aegis_used_bytes"],
                            "pressure": candidates["aegis_pressure"],
                        }
                    )
        if not timeline_rows:
            raise FigureDataError("P5 lacks sampled alpha/quota/used/pressure controller state")
        rows = timeline_rows
    _write_csv(output / f"{figure_id}.csv", rows)
    plt = _mpl()
    if group in {"P2", "P10"}:
        x_name, y_name = (
            ("load_multiplier", "failure_scenario")
            if group == "P2"
            else ("hbm_budget_gib_per_gpu", "dram_budget_gib_per_node")
        )
        xs = sorted({row[x_name] for row in rows}, key=float)
        ys = sorted({row[y_name] for row in rows}, key=str)
        grid = []
        for y_value in ys:
            line = []
            for x_value in xs:
                values = [float(row["mean_effect_s" if group == "P2" else "mean_wct_s"]) for row in rows if row[x_name] == x_value and row[y_name] == y_value]
                if not values:
                    raise FigureDataError(f"{group} incomplete heatmap bin")
                line.append(statistics.fmean(values))
            grid.append(line)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        image = ax.imshow(grid, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(len(xs)), [str(value) for value in xs])
        ax.set_yticks(range(len(ys)), [str(value) for value in ys])
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{group} {'paired WCT effect' if group == 'P2' else 'absolute WCT capacity map'}")
        fig.colorbar(image, ax=ax, label="seconds")
    elif group == "P5":
        fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        for budget in sorted({row["hbm_budget_gib_per_gpu"] for row in rows}, key=float):
            subset = sorted([row for row in rows if row["hbm_budget_gib_per_gpu"] == budget], key=lambda row: int(row["timestamp_ns"]))
            x = [int(row["timestamp_ns"]) / 1e9 for row in subset]
            for ax, field_name in zip(axes, ("alpha", "pressure", "quota_bytes", "used_bytes")):
                ax.plot(x, [float(row[field_name]) for row in subset], label=f"{budget} GiB")
                ax.set_ylabel(field_name)
                ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Independent phase-local virtual time (s)")
        axes[0].legend(ncol=4, fontsize=7)
        fig.suptitle("P5 controller observations (phases are not concatenated)")
    else:
        x_candidates = {
            "P3": "node_count",
            "P4": "algorithm_variant",
            "P6": "reuse_level",
            "P7": "cross_node_bandwidth_gb_s",
            "P8": "failure_severity",
            "P9": "load_multiplier",
            "P11": "dag_stratum",
        }
        x_name = x_candidates[group]
        metric = "mean_effect_s" if "mean_effect_s" in rows[0] else "mean_wct_s"
        bucket_rows: list[dict[str, Any]] = []
        paired_axis, _ = _pairing_contract(group)
        dimensions = [
            key
            for key in sorted(cells[0].axes)
            if key not in {"seed", x_name, paired_axis}
        ]
        buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in rows:
            label = "/".join(str(row[key]) for key in dimensions) or group
            buckets[(row[x_name], label)].append(float(row[metric]))
        for (x_value, label), values in buckets.items():
            bucket_rows.append({x_name: x_value, "series": label, metric: statistics.fmean(values)})
        fig = _simple_line_plot(
            bucket_rows,
            x=x_name,
            y=metric,
            series="series",
            title=f"{group} registered aggregate (paired where applicable)",
            ylabel="seconds",
        )
    _finish_figure(fig, output / figure_id)
    return figure_id, "Complete frozen grid; exact seed/policy pairing"


def _render_p12(output: Path, cells: Sequence[CellEvidence]) -> tuple[str, str]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        checks = (((cell.audit.get("details") or {}).get("metrics") or {}).get("analytical_vs_des") or {}).get("checks")
        if not isinstance(checks, list) or not checks:
            raise FigureDataError(f"{cell.cell_id} lacks analytical-vs-DES checks")
        for check in checks:
            analytical = _number(check.get("analytical_duration_ns"), name="analytical_duration_ns", cell=cell)
            observed = _number(check.get("des_duration_ns"), name="des_duration_ns", cell=cell)
            rows.append(
                {
                    **cell.axes,
                    "cell_id": cell.cell_id,
                    "phase": check.get("phase"),
                    "transfer_id": check.get("transfer_id"),
                    "analytical_duration_ns": analytical,
                    "des_duration_ns": observed,
                    "absolute_error_ns": abs(observed - analytical),
                    "relative_error": abs(observed - analytical) / analytical if analytical else None,
                }
            )
    _write_csv(output / "p12_analytical_parity.csv", rows)
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = [row["analytical_duration_ns"] / 1e6 for row in rows]
    y = [row["des_duration_ns"] / 1e6 for row in rows]
    axes[0].scatter(x, y, s=15, alpha=0.7)
    bound = max(x + y)
    axes[0].plot([0, bound], [0, bound], "k--", linewidth=0.8)
    axes[0].set_xlabel("Analytical duration (ms)")
    axes[0].set_ylabel("DES duration (ms)")
    axes[0].set_title("y=x parity")
    axes[1].hist([row["absolute_error_ns"] for row in rows], bins=min(20, max(1, len(rows))))
    axes[1].set_xlabel("Absolute error (ns)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Modeled internal error")
    _finish_figure(fig, output / "p12_analytical_parity")
    return "p12_analytical_parity", "Internal formula/DES parity; not hardware calibration"


def _gpu_label(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('node_id')}/gpu{row.get('local_gpu_id')}"
        f"/dp{row.get('dp_rank')}/pp{row.get('pp_rank')}/tp{row.get('tp_rank')}"
    )


def _diagnostic_dp_rank(output: Path, cell: CellEvidence) -> tuple[str, str]:
    """Render every Scheduler DP/PP actor on the authoritative global time line."""

    rows: list[dict[str, Any]] = []
    for row in cell.rows("rank_steps.jsonl"):
        timestamp_ns = row.get("timestamp_ns", row.get("des_now_ns"))
        if not isinstance(timestamp_ns, int):
            raise FigureDataError(
                f"{cell.cell_id} rank step has no integer global timestamp"
            )
        running = sorted(str(value) for value in row.get("running_request_ids", ()))
        waiting = sorted(str(value) for value in row.get("waiting_request_ids", ()))
        rows.append(
            {
                "cell_id": cell.cell_id,
                "timestamp_ns": timestamp_ns,
                "rank": f"dp{row.get('dp_rank')}/pp{row.get('pp_rank', 0)}",
                "dp_rank": int(row.get("dp_rank", -1)),
                "pp_rank": int(row.get("pp_rank", 0)),
                "phase": str(row.get("phase", "")),
                "running_count": len(running),
                "waiting_count": len(waiting),
                "active_count": len(set(running).union(waiting)),
                "running_request_ids": running,
                "waiting_request_ids": waiting,
                "primary_kv_gib": float(
                    row.get("physical_primary_bytes_total", 0)
                )
                / GIB,
                "prefix_kv_gib": float(
                    row.get("physical_prefix_bytes_total", 0)
                )
                / GIB,
            }
        )
    if not rows:
        raise FigureDataError(f"{cell.cell_id} has no per-rank state timeline")
    _write_csv(output / "diagnostic_per_dp_rank_state.csv", rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["rank"])].append(row)
    plt = _mpl()
    fig, (request_ax, kv_ax) = plt.subplots(
        2, 1, figsize=(13.5, 8.2), sharex=True
    )
    for label, values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["timestamp_ns"]))
        x = [int(row["timestamp_ns"]) / 1e9 for row in values]
        request_ax.step(
            x,
            [int(row["active_count"]) for row in values],
            where="post",
            linewidth=0.8,
            alpha=0.8,
            label=label,
        )
        kv_ax.step(
            x,
            [
                float(row["primary_kv_gib"])
                + float(row["prefix_kv_gib"])
                for row in values
            ],
            where="post",
            linewidth=0.8,
            alpha=0.8,
            label=label,
        )
    request_ax.set_title(
        f"{cell.cell_id}: request state of every DP/PP Scheduler actor"
    )
    request_ax.set_ylabel("Running or waiting requests")
    request_ax.grid(alpha=0.25)
    request_ax.legend(fontsize=5.0, ncol=4, loc="upper right")
    kv_ax.set_title("Scheduler-owned primary + radix-prefix KV by DP/PP")
    kv_ax.set_xlabel("Global virtual time (s)")
    kv_ax.set_ylabel("KV metadata occupancy (GiB)")
    kv_ax.grid(alpha=0.25)
    _finish_figure(fig, output / "diagnostic_per_dp_rank_state")
    return (
        "diagnostic_per_dp_rank_state",
        "Every DP/PP Scheduler actor: running/waiting request count and allocator-owned KV over global virtual time",
    )


def _diagnostic_hbm(output: Path, cell: CellEvidence) -> tuple[str, str]:
    rows = []
    for row in cell.rows("gpu_hbm_timeseries.jsonl"):
        rows.append(
            {
                "cell_id": cell.cell_id,
                "timestamp_ns": row["timestamp_ns"],
                "gpu": _gpu_label(row),
                "weights_gib": float(row.get("weights", 0)) / GIB,
                "runtime_gib": float(row.get("runtime", 0)) / GIB,
                "primary_kv_gib": float(row.get("primary_kv", 0)) / GIB,
                "prefix_kv_gib": float(row.get("prefix_kv", 0)) / GIB,
                "recovery_pending_gib": float(
                    row.get("recovery_pending", 0)
                )
                / GIB,
                "aegis_pending_gib": float(row.get("aegis_pending", 0)) / GIB,
                "aegis_committed_gib": float(row.get("aegis_committed", 0)) / GIB,
                "used_gib": float(row.get("used_bytes", 0)) / GIB,
                "capacity_gib": float(row.get("capacity_bytes", 0)) / GIB,
                "healthy": bool(row.get("healthy")),
            }
        )
    _write_csv(output / "diagnostic_per_gpu_hbm.csv", rows)
    plt = _mpl()
    fig, (ax, aegis_ax, composition_ax) = plt.subplots(
        3,
        1,
        figsize=(13.5, 12.0),
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.15)},
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["gpu"])].append(row)
    for label, values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["timestamp_ns"]))
        ax.step(
            [int(row["timestamp_ns"]) / 1e9 for row in values],
            [float(row["used_gib"]) for row in values],
            where="post",
            linewidth=0.8,
            alpha=0.75,
            label=label,
        )
        aegis_ax.step(
            [int(row["timestamp_ns"]) / 1e9 for row in values],
            [
                float(row["recovery_pending_gib"])
                + float(row["aegis_pending_gib"])
                + float(row["aegis_committed_gib"])
                for row in values
            ],
            where="post",
            linewidth=0.8,
            alpha=0.75,
            label=label,
        )
    ax.set_title(f"{cell.cell_id}: full used HBM on every physical GPU")
    ax.set_xlabel("Global virtual time (s)")
    ax.set_ylabel("Total used HBM (GiB)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=4.8, ncol=4, loc="upper right")
    aegis_ax.set_title("Aegis protection plus recovery-reservation HBM")
    aegis_ax.set_xlabel("Global virtual time (s)")
    aegis_ax.set_ylabel("Recovery + Aegis KV (GiB)")
    aegis_ax.grid(alpha=0.25)

    # Aegis-only curves answer when protection bytes exist, but they must not
    # be mistaken for total GPU occupancy.  For every physical GPU, show the
    # exact component vector at that GPU's maximum observed used-HBM snapshot.
    # Selecting one real snapshot (rather than independently peaking each
    # component) preserves an additive, capacity-comparable composition.
    peak_rows: list[dict[str, Any]] = []
    component_fields = (
        ("weights_gib", "weights"),
        ("runtime_gib", "runtime"),
        ("primary_kv_gib", "primary KV"),
        ("prefix_kv_gib", "prefix KV"),
        ("recovery_pending_gib", "recovery pending"),
        ("aegis_pending_gib", "Aegis pending"),
        ("aegis_committed_gib", "Aegis committed"),
    )
    for label, values in sorted(grouped.items()):
        peak = max(
            values,
            key=lambda row: (
                sum(float(row[field]) for field, _name in component_fields),
                int(row["timestamp_ns"]),
            ),
        )
        peak_rows.append({**peak, "gpu": label})
    positions = list(range(len(peak_rows)))
    bottoms = [0.0] * len(peak_rows)
    colors = (
        "#4c78a8",
        "#9ecae9",
        "#f58518",
        "#54a24b",
        "#72b7b2",
        "#e45756",
        "#b279a2",
    )
    for (field, name), color in zip(component_fields, colors):
        values = [float(row[field]) for row in peak_rows]
        composition_ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.82,
            label=name,
            color=color,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    composition_ax.set_xticks(
        positions,
        [str(row["gpu"]) for row in peak_rows],
        rotation=75,
        ha="right",
        fontsize=5.6,
    )
    composition_ax.set_ylabel("Used HBM at per-GPU peak (GiB)")
    composition_ax.set_xlabel("Physical GPU = node / local GPU / DP / PP / TP")
    composition_ax.set_title("Full HBM composition at each GPU's peak-used snapshot")
    composition_ax.grid(axis="y", alpha=0.25)
    composition_ax.legend(fontsize=6.2, ncol=3, loc="upper right")
    _finish_figure(fig, output / "diagnostic_per_gpu_hbm")
    return (
        "diagnostic_per_gpu_hbm",
        "Every physical DP/PP/TP GPU; full-used and Aegis/recovery timelines plus additive peak composition",
    )


def _diagnostic_dram(output: Path, cell: CellEvidence) -> tuple[str, str]:
    rows = []
    for row in cell.rows("node_dram_timeseries.jsonl"):
        rows.append(
            {
                "cell_id": cell.cell_id,
                "timestamp_ns": row["timestamp_ns"],
                "node_id": row["node_id"],
                "runtime_gib": float(row.get("runtime", 0)) / GIB,
                "hicache_primary_gib": float(row.get("hicache_primary", 0)) / GIB,
                "aegis_pending_gib": float(row.get("aegis_pending", 0)) / GIB,
                "aegis_committed_gib": float(row.get("aegis_committed", 0)) / GIB,
                "used_gib": float(row.get("used_bytes", 0)) / GIB,
                "capacity_gib": float(row.get("capacity_bytes", 0)) / GIB,
                "healthy": bool(row.get("healthy")),
            }
        )
    _write_csv(output / "diagnostic_per_node_dram.csv", rows)
    plt = _mpl()
    fig, (ax, aegis_ax, composition_ax) = plt.subplots(
        3,
        1,
        figsize=(10.5, 10.2),
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.0)},
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["node_id"])].append(row)
    for node_id, values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["timestamp_ns"]))
        ax.step(
            [int(row["timestamp_ns"]) / 1e9 for row in values],
            [float(row["used_gib"]) for row in values],
            where="post",
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            label=node_id,
        )
        aegis_ax.step(
            [int(row["timestamp_ns"]) / 1e9 for row in values],
            [
                float(row["aegis_pending_gib"])
                + float(row["aegis_committed_gib"])
                for row in values
            ],
            where="post",
            marker="o",
            markersize=2.0,
            linewidth=1.0,
            label=node_id,
        )
    ax.set_title(f"{cell.cell_id}: full used DRAM in every node pool")
    ax.set_xlabel("Global virtual time (s)")
    ax.set_ylabel("Total used DRAM (GiB)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    aegis_ax.set_title("Aegis pending + committed DRAM")
    aegis_ax.set_xlabel("Global virtual time (s)")
    aegis_ax.set_ylabel("Aegis DRAM (GiB)")
    aegis_ax.grid(alpha=0.25)

    component_fields = (
        ("runtime_gib", "runtime"),
        ("hicache_primary_gib", "HiCache primary"),
        ("aegis_pending_gib", "Aegis pending"),
        ("aegis_committed_gib", "Aegis committed"),
    )
    peak_rows: list[dict[str, Any]] = []
    for node_id, values in sorted(grouped.items()):
        peak = max(
            values,
            key=lambda row: (
                sum(float(row[field]) for field, _name in component_fields),
                int(row["timestamp_ns"]),
            ),
        )
        peak_rows.append({**peak, "node_id": node_id})
    positions = list(range(len(peak_rows)))
    bottoms = [0.0] * len(peak_rows)
    colors = ("#9ecae9", "#f58518", "#e45756", "#b279a2")
    for (field, name), color in zip(component_fields, colors):
        values = [float(row[field]) for row in peak_rows]
        composition_ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.7,
            label=name,
            color=color,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    composition_ax.set_xticks(
        positions,
        [str(row["node_id"]) for row in peak_rows],
    )
    composition_ax.set_xlabel("Physical node DRAM pool")
    composition_ax.set_ylabel("Used DRAM at per-node peak (GiB)")
    composition_ax.set_title("Full DRAM composition at each node's peak-used snapshot")
    composition_ax.grid(axis="y", alpha=0.25)
    composition_ax.legend(fontsize=7, ncol=2)
    _finish_figure(fig, output / "diagnostic_per_node_dram")
    return (
        "diagnostic_per_node_dram",
        "Every node-local DRAM pool; full-used and Aegis timelines plus additive peak composition",
    )


def _diagnostic_gantt(output: Path, cell: CellEvidence) -> tuple[str, str]:
    transfers: dict[str, dict[str, Any]] = {}
    for row in cell.rows("transfers.jsonl"):
        transfer_id = str(row.get("transfer_id", ""))
        if transfer_id:
            transfers.setdefault(transfer_id, {}).update(row)
    recovery_name = "recovery_transfers.jsonl"
    if recovery_name in cell.manifest.get("files", {}):
        for row in cell.rows(recovery_name):
            transfer_id = str(row.get("recovery_id") or row.get("transfer_id") or "")
            if transfer_id:
                transfers.setdefault(transfer_id, {}).update(row)
                transfers[transfer_id]["recovery"] = True
    rows = []
    for transfer_id, row in sorted(transfers.items()):
        start = row.get("start_ns")
        end = row.get("end_ns")
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise FigureDataError(f"{cell.cell_id} transfer interval invalid: {transfer_id}")
        source = row.get("source") or {}
        destination = row.get("destination") or {}
        rows.append(
            {
                "cell_id": cell.cell_id,
                "transfer_id": transfer_id,
                "start_ns": start,
                "end_ns": end,
                "duration_ns": end - start,
                "path": row.get("path_kind") or f"{source.get('tier')}->{destination.get('tier')}",
                "transfer_class": row.get("transfer_class", "global_des_kv"),
                "direction": row.get("direction"),
                "size_bytes": row.get("size_bytes"),
                "resources": row.get("resources", []),
                "state": row.get("state") or row.get("outcome"),
                "recovery": bool(row.get("recovery"))
                or bool(str(row.get("recovery_group_id") or "")),
            }
        )
    if not rows:
        raise FigureDataError(f"{cell.cell_id} has no KV transfer for a Gantt")
    _write_csv(output / "diagnostic_kv_transfer_gantt.csv", rows)
    plt = _mpl()
    height = min(14.0, max(4.5, len(rows) * 0.12))
    fig, ax = plt.subplots(figsize=(11, height))
    paths = sorted({str(row["path"]) for row in rows})
    colors = {path: plt.get_cmap("tab10")(index % 10) for index, path in enumerate(paths)}
    for index, row in enumerate(rows):
        ax.barh(index, row["duration_ns"] / 1e9, left=row["start_ns"] / 1e9, color=colors[str(row["path"])], height=0.7)
    # Avoid an unreadable 100+ label wall; the CSV remains the exact complete
    # Gantt source and the raster displays every interval.
    step = max(1, math.ceil(len(rows) / 25))
    ticks = list(range(0, len(rows), step))
    ax.set_yticks(ticks, [str(rows[index]["transfer_id"]) for index in ticks], fontsize=6)
    ax.set_xlabel("Global virtual time (s)")
    ax.set_ylabel("Physical KV transfer")
    ax.set_title(f"{cell.cell_id}: KV migration Gantt ({len(rows)} jobs)")
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=color, linewidth=6, label=path) for path, color in colors.items()]
    ax.legend(handles=handles, fontsize=6, loc="upper right")
    _finish_figure(fig, output / "diagnostic_kv_transfer_gantt")
    return (
        "diagnostic_kv_transfer_gantt",
        "All declared ordinary HiCache, Aegis protection, and recovery transfer intervals",
    )


def _diagnostic_failure(output: Path, cell: CellEvidence) -> tuple[str, str]:
    failures = cell.rows("failures.jsonl")
    if not failures:
        raise FigureDataError(f"{cell.cell_id} has no failure timeline")
    rows = []
    for row in failures:
        start = _number(row.get("injected_offset_s"), name="injected_offset_s", cell=cell)
        recovered = _number(row.get("recovered_offset_s"), name="recovered_offset_s", cell=cell)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "event_id": row.get("event_id"),
                "failure_scope": row.get("failure_scope"),
                "start_s": start,
                "recovered_s": recovered,
                "duration_s": recovered - start,
                "affected_dp_ranks": row.get("affected_dp_ranks", []),
                "affected_request_ids": row.get("affected_request_ids", []),
                "terminated_stages": len(row.get("terminated_stages", [])),
                "restarted_stages": len(row.get("restarted_stages", [])),
            }
        )
    _write_csv(output / "diagnostic_failure_timeline.csv", rows)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for index, row in enumerate(rows):
        ax.barh(index, row["duration_s"], left=row["start_s"], color="#d95f02", alpha=0.7)
        ax.scatter([row["start_s"], row["recovered_s"]], [index, index], color=["#b2182b", "#1b7837"], zorder=3)
    ax.set_yticks(range(len(rows)), [str(row["event_id"]) for row in rows], fontsize=7)
    ax.set_xlabel("Global virtual time (s)")
    ax.set_title(f"{cell.cell_id}: fail-stop and empty-state repair")
    ax.grid(axis="x", alpha=0.25)
    _finish_figure(fig, output / "diagnostic_failure_timeline")
    return "diagnostic_failure_timeline", "Failure and repair offsets from the authoritative failure journal"


def generate_plots(
    collection: EvidenceCollection,
    output_dir: Path,
    *,
    representative_group: str | None = None,
) -> dict[str, Any]:
    target = output_dir.resolve()
    if os.path.lexists(target):
        raise PlotEvidenceError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    figures: list[dict[str, Any]] = []
    blockers: dict[str, list[str]] = {}
    try:
        for group in GROUPS:
            ready, reasons = group_readiness(collection, group)
            if not ready:
                blockers[group] = reasons
                continue
            cells = collection.by_group(group)
            try:
                if group == "P1":
                    rendered = [_render_p1_ecdf(temp, cells), _render_forest(temp, cells, group=group, axis="policy", baseline="recompute", figure_id="p1_paired_forest")]
                elif group == "P12":
                    rendered = [_render_p12(temp, cells)]
                else:
                    rendered = [_render_group_summary(temp, cells, group=group)]
                for figure_id, note in rendered:
                    figures.append(_figure_manifest(temp, figure_id, cells, figure_kind="registered_paper_figure", note=note))
            except FigureDataError as exc:
                blockers.setdefault(group, []).append(f"required_metric_missing:{exc}")

        candidates = collection.cells
        if representative_group:
            if representative_group not in GROUPS:
                raise PlotEvidenceError(f"unknown representative group: {representative_group}")
            candidates = collection.by_group(representative_group)
        if candidates:
            representative = sorted(candidates, key=lambda cell: (cell.group, cell.cell_id))[0]
            for renderer in (
                _diagnostic_dp_rank,
                _diagnostic_hbm,
                _diagnostic_dram,
                _diagnostic_gantt,
                _diagnostic_failure,
            ):
                try:
                    figure_id, note = renderer(temp, representative)
                except FigureDataError as exc:
                    blockers.setdefault("diagnostics", []).append(str(exc))
                    continue
                figures.append(_figure_manifest(temp, figure_id, [representative], figure_kind="representative_mechanism_diagnostic", note=note))
        else:
            blockers.setdefault("diagnostics", []).append("no_source_cell")

        claims = _source_claims(collection.cells)
        payload = {
            "schema": PLOT_SCHEMA,
            "plotter_code_sha256": _sha256_file(Path(__file__)),
            "suite_id": collection.suite_id,
            "claims": claims,
            "source_runs": collection.source_runs,
            "source_cell_count": len(collection.cells),
            "registered_figures_complete": not any(group in blockers for group in GROUPS),
            "figures": [
                {
                    "figure_id": figure["figure_id"],
                    "figure_kind": figure["figure_kind"],
                    "manifest": f"{figure['figure_id']}.manifest.json",
                    "manifest_payload_sha256": figure["canonical_payload_sha256"],
                    "paper_ready": figure["claims"]["paper_ready"],
                }
                for figure in figures
            ],
            "blockers": blockers,
            "prohibitions": {
                "missing_value_fill": False,
                "interpolation": False,
                "historical_613_cell_reuse": False,
                "partial_registered_figure": False,
            },
        }
        payload["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(payload))
        (temp / "plot_manifest.json").write_bytes(_canonical_bytes(payload))
        os.rename(temp, target)
        return payload
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    experiment_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Verify immutable global-DES runs and render fail-closed P1--P12 figures"
    )
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=experiment_root / "configs_v3",
    )
    parser.add_argument("--representative-group", choices=GROUPS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    collection = collect_evidence(args.run_dir, config_root=args.config_root)
    manifest = generate_plots(
        collection,
        args.output_dir,
        representative_group=args.representative_group,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CellEvidence",
    "EvidenceCollection",
    "FigureDataError",
    "PlotEvidenceError",
    "collect_evidence",
    "generate_plots",
    "group_readiness",
    "load_frozen_groups",
    "main",
]
