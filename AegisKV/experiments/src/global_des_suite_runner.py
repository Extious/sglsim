"""P1--P12 orchestration for the real-Scheduler global virtual-time path.

This runner intentionally does not reuse the legacy aggregate AegisKV DES.
Every executed request enters :class:`OfflineSchedulerCluster`, whose real
SGLSim/SGLang-mirrored Scheduler workers share the single ``GlobalDES`` heap.
The runner owns only configuration resolution, request compilation, hard
failure/replay orchestration, lineage, metrics, and atomic artifact publication.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sglsim.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.aegis_residual_dag import (
    RESIDUAL_DAG_SLACK_SEMANTICS,
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
    resolve_slack_for_design,
)
from sglsim.srt.simulation.global_des_artifacts import (
    ArtifactTargetExistsError,
    write_global_des_artifacts,
)
from sglsim.srt.simulation.offline_aegis_transfer_bridge import (
    CopyFabricBandwidths,
    CopyFabricConfig,
    CopyFabricSetupLatencies,
    OfflineAegisTransferBridge,
)
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster
from sglsim.srt.simulation.offline_router_ledger import (
    CONTINUATION_EVIDENCE_CLASS,
    REPLAY_CONTRACT_SCHEMA,
    ROUTER_LEDGER_SCHEMA,
    execution_index as router_execution_index,
)

from .global_des_causal_workload import (
    CONTROLLED_OVERLAY_MODE,
    CausalWorkloadError,
    enrich_causal_workload,
)
from .build_weka_observed_workload import (
    ARTIFACT_SCHEMA as WEKA_OBSERVED_ARTIFACT_SCHEMA,
    REQUEST_SCHEMA as WEKA_OBSERVED_REQUEST_SCHEMA,
    TOKEN_IDENTITY_SCHEMA as WEKA_TOKEN_IDENTITY_SCHEMA,
    WekaObservedBuildError,
    audit_artifact as audit_weka_observed_artifact,
)
from .global_des_config import (
    CONFIG_ROOT,
    GROUP_IDS,
    GroupConfig,
    SuiteConfig,
    build_offline_physical_topology,
    canonical_json_bytes,
    config_hash,
    load_suite,
)
from .global_des_suite_metrics import (
    MODELED_EVIDENCE_CLASS,
    build_scheduler_cell_metrics,
    evidence_claims,
)
from .global_des_workload import (
    CompiledWorkload,
    TokenIdentity,
    WorkloadNotReadyError,
    compile_workload,
    deterministic_token_ids,
)


RUN_SCHEMA = "aegiskv.global-des-scheduler-suite-run.v1"
CELL_ENGINE = "sglsim_offline_scheduler_global_des"
DEFAULT_P0_SUMMARY = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "20260829T121349Z_global_des_p0_contract_v5"
    / "P0"
    / "summary.json"
)
SMOKE_GROUPS = ("P1", "P4", "P11", "P12")
GIB = 1024**3
GB = 10**9


class SuiteExecutionError(RuntimeError):
    """Base class for fail-closed suite orchestration failures."""


class SuiteReadinessError(SuiteExecutionError):
    """Raised before a cell launches when required lineage/input is absent."""

    def __init__(self, code: str, message: str, *, details: Any = None):
        self.code = str(code)
        self.details = details
        super().__init__(f"{self.code}: {message}")


class ResumeValidationError(SuiteExecutionError):
    """Raised when an existing cell is incomplete, stale, or corrupt."""


@dataclass(frozen=True)
class CellSpec:
    group: str
    ordinal: int
    axes: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    cell_id: str


@dataclass(frozen=True)
class PreparedWorkload:
    records: tuple[dict[str, Any], ...]
    request_universe_hash: str
    compiled_requests_hash: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionOptions:
    model_paths: Mapping[str, str]
    model_revisions: Mapping[str, str]
    smoke: bool = False
    max_requests_per_cell: int | None = None
    max_events: int = 1_000_000
    calibration_lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CellExecution:
    path: Path
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    replay_ids: tuple[str, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteReadinessError(
            "json_unreadable", f"cannot read {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SuiteReadinessError("json_not_object", f"object required: {path}")
    return value


def validate_p0_summary(path: Path) -> dict[str, Any]:
    """Verify the frozen twelve-cell P0 contract gate without widening claims."""

    summary = _strict_json(Path(path))
    claims = summary.get("claims")
    cells = summary.get("cells")
    if summary.get("schema") != "aegiskv.global-des-p0-summary.v1":
        raise SuiteReadinessError("p0_schema_mismatch", "unexpected P0 schema")
    if summary.get("status") != "PASS" or not isinstance(claims, Mapping):
        raise SuiteReadinessError("p0_gate_failed", "P0 status is not PASS")
    if claims.get("configured_p0_contract_gate_passed") is not True:
        raise SuiteReadinessError(
            "p0_gate_failed", "configured P0 contract gate is not true"
        )
    if not isinstance(cells, list) or len(cells) != 12:
        raise SuiteReadinessError(
            "p0_cell_count_mismatch", "P0 must contain twelve cells"
        )
    if any(cell.get("status") != "PASS" for cell in cells):
        raise SuiteReadinessError("p0_cell_failed", "one or more P0 cells failed")
    # The referenced P0 is explicitly contract-only.  The first P1--P12 cell
    # produced by this runner is the Scheduler-integration evidence; never
    # rewrite the older summary to claim it already covered this path.
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256_file(Path(path)),
        "status": "PASS",
        "contract_cells": 12,
        "contract_only": bool(claims.get("contract_execution_only", True)),
        "source_full_scheduler_integration": bool(
            claims.get("full_scheduler_integration", False)
        ),
    }


def _resolved_three_layers(group: GroupConfig, axes: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in group.layers:
        selected = {
            name: copy.deepcopy(axes[name])
            for name in layer.axes
            if name in axes
        }
        result[layer.layer] = {
            "schema_version": layer.resolved["schema_version"],
            "layer": layer.layer,
            "group": group.group,
            "fixed": copy.deepcopy(layer.fixed),
            "selected_axes": selected,
        }
    return result


def _cell_id(group: str, ordinal: int, axes: Mapping[str, Any]) -> str:
    suffix = config_hash(
        {"group": str(group), "cell_ordinal": int(ordinal), "axes": dict(axes)}
    )[:16]
    return f"{group}-{int(ordinal):04d}-{suffix}"


def enumerate_cells(loaded: SuiteConfig) -> tuple[CellSpec, ...]:
    cells: list[CellSpec] = []
    for group in loaded.groups:
        if group.group == "P0":
            continue
        for row in group.iter_cells():
            axes = dict(row["axes"])
            ordinal = int(row["cell_ordinal"])
            cells.append(
                CellSpec(
                    group=group.group,
                    ordinal=ordinal,
                    axes=axes,
                    resolved_config=_resolved_three_layers(group, axes),
                    cell_id=_cell_id(group.group, ordinal, axes),
                )
            )
    return tuple(cells)


def select_cells(
    loaded: SuiteConfig,
    *,
    groups: Sequence[str] = (),
    selectors: Sequence[str] = (),
    smoke: bool = False,
    all_cells: bool = False,
) -> tuple[CellSpec, ...]:
    """Select stable cells; an unqualified 2,134-cell launch is forbidden."""

    all_planned = enumerate_cells(loaded)
    group_set = {str(group) for group in groups}
    unknown_groups = group_set - set(GROUP_IDS[1:])
    if unknown_groups:
        raise ValueError(f"unknown/non-performance groups: {sorted(unknown_groups)}")
    by_key = {(cell.group, cell.ordinal): cell for cell in all_planned}
    explicit: list[CellSpec] = []
    for raw in selectors:
        try:
            group, ordinal_text = str(raw).split(":", 1)
            key = group, int(ordinal_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"cell selector must be GROUP:ORDINAL, got {raw!r}") from exc
        try:
            explicit.append(by_key[key])
        except KeyError as exc:
            raise ValueError(f"unknown cell selector {raw!r}") from exc
    if explicit:
        selected = explicit
    elif smoke:
        chosen_groups = tuple(groups) if groups else SMOKE_GROUPS
        selected = [by_key[(group, 0)] for group in chosen_groups]
    elif group_set:
        selected = [cell for cell in all_planned if cell.group in group_set]
    elif all_cells:
        selected = list(all_planned)
    else:
        raise ValueError(
            "refusing implicit full-suite launch; use --cells, --groups, --smoke, or --all"
        )
    deduplicated: dict[tuple[str, int], CellSpec] = {}
    for cell in selected:
        deduplicated[(cell.group, cell.ordinal)] = cell
    return tuple(deduplicated.values())


def _compiled_to_prepared(compiled: CompiledWorkload) -> PreparedWorkload:
    return PreparedWorkload(
        records=tuple(record.as_dict() for record in compiled.records),
        request_universe_hash=compiled.request_universe_hash,
        compiled_requests_hash=compiled.compiled_requests_hash,
        metadata={
            **dict(compiled.metadata),
            "dataset_audit": compiled.dataset_audit.as_dict(),
        },
    )


def _prepare_p11_workload(cell: CellSpec, experiment_root: Path) -> PreparedWorkload:
    fixed = cell.resolved_config["request"]["fixed"]
    raw_artifacts = fixed["artifact_lineage"]["input_artifacts"]
    if len(raw_artifacts) != 1:
        raise SuiteReadinessError(
            "p11_artifact_ambiguous", "P11 requires exactly one observed DAG artifact"
        )
    raw_path = Path(str(raw_artifacts[0]))
    if not raw_path.is_absolute():
        if raw_path.parts[:2] == ("AegisKV", "experiments"):
            raw_path = experiment_root.parents[1] / raw_path
        else:
            raw_path = experiment_root / raw_path
    root = raw_path.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SuiteReadinessError(
            "p11_observed_dag_missing",
            f"compiled observed DAG manifest is absent: {manifest_path}",
        )
    manifest = _strict_json(manifest_path)
    if manifest.get("schema") != WEKA_OBSERVED_ARTIFACT_SCHEMA:
        raise SuiteReadinessError(
            "p11_observed_dag_schema_mismatch", "unexpected P11 workload schema"
        )
    try:
        artifact_audit = audit_weka_observed_artifact(root)
    except WekaObservedBuildError as exc:
        raise SuiteReadinessError(
            "p11_observed_dag_audit_failed", str(exc)
        ) from exc
    configured_source_sha = str(
        fixed["workload"]["source_artifact"]["sha256"]
    )
    if str(manifest.get("source", {}).get("source_sha256") or "") != configured_source_sha:
        raise SuiteReadinessError(
            "p11_source_hash_mismatch",
            "compiled P11 source hash differs from the frozen request JSON",
        )
    requests_name = str(manifest.get("requests_file") or "requests.jsonl")
    requests_path = root / requests_name
    expected_sha = str(manifest.get("requests_sha256") or "")
    if not requests_path.is_file() or not expected_sha:
        raise SuiteReadinessError(
            "p11_requests_missing", "P11 manifest lacks pinned requests JSONL"
        )
    actual_sha = _sha256_file(requests_path)
    if actual_sha != expected_sha:
        raise SuiteReadinessError(
            "p11_requests_hash_mismatch",
            f"P11 requests SHA-256 mismatch: {actual_sha}",
        )
    selected_seed = int(cell.axes["seed"])
    selected_stratum = str(cell.axes["dag_stratum"])
    partition = next(
        (
            row
            for row in manifest.get("partitions") or ()
            if int(row.get("seed", -1)) == selected_seed
            and str(row.get("dag_stratum")) == selected_stratum
        ),
        None,
    )
    if not isinstance(partition, Mapping):
        raise SuiteReadinessError(
            "p11_partition_missing",
            f"no frozen P11 partition for seed={selected_seed}, "
            f"stratum={selected_stratum}",
        )
    records: list[dict[str, Any]] = []
    with requests_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SuiteReadinessError(
                    "p11_requests_json_invalid",
                    f"invalid JSON at {requests_path}:{line_number}",
                ) from exc
            if not isinstance(row, dict):
                raise SuiteReadinessError(
                    "p11_request_not_object", f"object required at line {line_number}"
                )
            if row.get("schema") != WEKA_OBSERVED_REQUEST_SCHEMA:
                raise SuiteReadinessError(
                    "p11_request_schema_mismatch",
                    f"unexpected P11 row schema at line {line_number}",
                )
            if (
                int(row.get("seed", -1)) != selected_seed
                or str(row.get("dag_stratum")) != selected_stratum
            ):
                continue
            records.append(row)
    if not records:
        raise SuiteReadinessError(
            "p11_partition_empty",
            f"no records for seed={selected_seed}, stratum={selected_stratum}",
        )
    expected_workflows = tuple(sorted(map(str, partition.get("workflow_ids") or ())))
    observed_workflows = tuple(sorted({str(row["workflow"]) for row in records}))
    if observed_workflows != expected_workflows:
        raise SuiteReadinessError(
            "p11_partition_workflow_mismatch",
            "selected P11 records differ from the manifest workflow partition",
        )
    if len(records) != int(partition.get("request_count", -1)):
        raise SuiteReadinessError(
            "p11_partition_request_count_mismatch",
            "selected P11 request count differs from the manifest partition",
        )
    workflows_per_partition = int(
        manifest["dag_stratification"]["workflows_per_seed_per_stratum"]
    )
    if workflows_per_partition != 1 or len(observed_workflows) != 1:
        raise SuiteReadinessError(
            "p11_atomic_workflow_contract_mismatch",
            "P11 currently requires exactly one complete workflow per seed/stratum",
        )
    universe_hash = str(manifest.get("request_universe_hash") or "")
    if not universe_hash:
        raise SuiteReadinessError(
            "p11_universe_hash_missing", "P11 request universe hash is required"
        )
    return PreparedWorkload(
        records=tuple(records),
        request_universe_hash=universe_hash,
        compiled_requests_hash=_sha256_bytes(canonical_json_bytes(records)),
        metadata={
            "kind": "weka_agentx_observed_dag",
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "requests_sha256": actual_sha,
            "artifact_audit": artifact_audit,
            "seed": selected_seed,
            "dag_stratum": selected_stratum,
            "workflow_ids": list(observed_workflows),
            "request_count": len(records),
            # A WCT observation is meaningful only for the complete workflow.
            # The smoke path therefore executes this whole partition instead
            # of applying the generic first-N request truncation.
            "atomic_workflow_replay": True,
            "smoke_request_truncation_permitted": False,
            "selected_ticket_coverage": {
                "modeled_requests": sum(
                    row["simulation"].get("weka_ticket_modeled") is True
                    for row in records
                ),
                "unmodeled_fail_close_zero_requests": sum(
                    row["simulation"].get("weka_ticket_modeled") is not True
                    for row in records
                ),
                "nonzero_slack_requests": sum(
                    float(row["simulation"].get("workflow_slack_ms", 0.0)) > 0.0
                    for row in records
                ),
            },
            "predictor_lineage_sha256": manifest.get("predictor_lineage_sha256"),
            "causal_predictor_is_hardware_calibrated": False,
            "paper_ready": False,
        },
    )


def _prepare_p12_workload(cell: CellSpec) -> PreparedWorkload:
    tokens = int(cell.axes["verification_chunk_tokens"])
    seed = int(cell.axes["seed"])
    records = tuple(
        {
            "rid": f"p12-{cell.cell_id}-{index}",
            "workflow": f"p12-verification-{cell.cell_id}",
            "session": cell.cell_id,
            "provider": "analytical_contract",
            "round_index": index,
            "arrival_ns": index * 1_000,
            "input_tokens": tokens,
            # P12 verifies one complete sealed prefix chunk.  Adding a decode
            # token would create a partial second chunk and test a different
            # recovery-composition contract.
            "output_tokens": 0,
            "cached_prefix_tokens": 0,
            "prefix_key": f"p12-{seed}-{index}",
            "source_round_pk": index,
            "source_round_id": f"closed-form-{index}",
        }
        for index in range(2)
    )
    return PreparedWorkload(
        records=records,
        request_universe_hash=_sha256_bytes(
            canonical_json_bytes(
                {
                    "kind": "closed_form_transfer_cases",
                    "seed": seed,
                    "tokens": tokens,
                }
            )
        ),
        compiled_requests_hash=_sha256_bytes(canonical_json_bytes(records)),
        metadata={
            "kind": "analytical_transfer_verification",
            "closed_form_cases": len(records),
            "hardware_calibration": "missing_modeled_preliminary",
        },
    )


def _p12_analytical_slack_fields(
    *, rid: str, workflow_id: str, observed_at_ms: float
) -> dict[str, Any]:
    """Build a causal ticket for the explicitly analytical P12 workload.

    The longer peer is part of P12's declared fork/join contract, rather than
    inferred from a completed trace.  Consequently the ticket exercises the
    current-design admission path without smuggling future information into
    P1--P11.  P12 remains labelled modeled/preliminary in every artifact.
    """

    peer_id = f"{rid}/analytical-peer"
    graph = VisibleResidualDag(
        workflow_id=str(workflow_id),
        graph_version="p12-analytical-visible-fork-join-v1",
        observed_at_ms=float(observed_at_ms),
        nodes=(
            ResidualDagNode("fork", "control", 0.0),
            ResidualDagNode(
                str(rid), "agent", 0.0, active=True, agent_id="transfer-under-test"
            ),
            ResidualDagNode(
                peer_id, "agent", 1.0, active=True, agent_id="visible-peer"
            ),
            ResidualDagNode("join", "control", 0.0),
        ),
        edges=(
            ("fork", str(rid)),
            ("fork", peer_id),
            (str(rid), "join"),
            (peer_id, "join"),
        ),
        next_join_subgraph_complete=True,
        metadata_complete=True,
        uses_future_information=False,
    )
    return build_residual_dag_slack_tickets(graph)[str(rid)].to_simulation_fields()


def prepare_workload(
    cell: CellSpec,
    *,
    experiment_root: Path,
) -> PreparedWorkload:
    if cell.group in {f"P{index}" for index in range(1, 11)}:
        try:
            group_request = {
                "schema_version": "aegiskv.global-des-layer.v3",
                "layer": "request",
                "group": cell.group,
                "fixed": copy.deepcopy(cell.resolved_config["request"]["fixed"]),
                "axes": {
                    key: [value]
                    for key, value in cell.resolved_config["request"][
                        "selected_axes"
                    ].items()
                },
            }
            compiled = compile_workload(
                group_request,
                cell.axes,
                experiment_root=experiment_root,
            )
        except WorkloadNotReadyError as exc:
            raise SuiteReadinessError(
                "request_compiler_not_ready", str(exc), details=exc.as_dict()
            ) from exc
        prepared = _compiled_to_prepared(compiled)
        workload_config = cell.resolved_config["request"]["fixed"]["workload"]
        causal_contract = workload_config.get("causal_slack_contract")
        if not isinstance(causal_contract, Mapping):
            raise SuiteReadinessError(
                "causal_slack_contract_missing",
                f"{cell.group} request layer lacks a causal Slack contract",
            )
        context_tokens = int(
            cell.axes.get(
                "context_tokens",
                cell.resolved_config["request"]["fixed"]["context"][
                    "central_tokens"
                ],
            )
        )
        try:
            enriched, causal_audit = enrich_causal_workload(
                prepared.records,
                group=cell.group,
                seed=int(cell.axes["seed"]),
                context_tokens=context_tokens,
                contract=causal_contract,
            )
        except CausalWorkloadError as exc:
            raise SuiteReadinessError(
                "causal_slack_contract_invalid",
                str(exc),
                details={"group": cell.group},
            ) from exc
        universe_hash = prepared.request_universe_hash
        if causal_audit["mode"] == CONTROLLED_OVERLAY_MODE:
            universe_hash = _sha256_bytes(
                canonical_json_bytes(
                    {
                        "base_request_universe_hash": universe_hash,
                        "causal_contract": causal_contract,
                        "seed": int(cell.axes["seed"]),
                        "overlay_request_ids": [
                            str(row["rid"])
                            for row in enriched
                            if row.get("provider")
                            == "controlled_visible_fork_join"
                        ],
                    }
                )
            )
        return PreparedWorkload(
            records=enriched,
            request_universe_hash=universe_hash,
            compiled_requests_hash=_sha256_bytes(canonical_json_bytes(enriched)),
            metadata={
                **dict(prepared.metadata),
                "base_compiled_requests_hash": prepared.compiled_requests_hash,
                "causal_workload": causal_audit,
            },
        )
    if cell.group == "P11":
        return _prepare_p11_workload(cell, experiment_root)
    if cell.group == "P12":
        return _prepare_p12_workload(cell)
    raise SuiteReadinessError("unsupported_group", f"cannot execute {cell.group}")


def _selected_model(cell: CellSpec) -> str:
    deployment = cell.resolved_config["deployment"]
    return str(
        deployment["selected_axes"].get(
            "model", deployment["fixed"]["central_model"]
        )
    )


def _selected_nodes(cell: CellSpec) -> int:
    deployment = cell.resolved_config["deployment"]
    return int(
        deployment["selected_axes"].get(
            "node_count", deployment["fixed"]["cluster"]["central_node_count"]
        )
    )


def _selected_policy(cell: CellSpec) -> str:
    deployment = cell.resolved_config["deployment"]
    return str(
        deployment["selected_axes"].get(
            "policy", deployment["fixed"]["aegiskv"]["central_policy"]
        )
    )


def _policy_runtime_name(policy: str) -> tuple[str, bool]:
    normalized = str(policy).lower().replace("-", "_")
    mapping = {
        "recompute": ("aegiskv_no_backup", False),
        "no_protection_control": ("aegiskv_no_backup", False),
        "host_checkpoint": ("aegiskv_l2_only", True),
        "neighbor_replica": ("aegiskv_l1_only", True),
        "aegiskv": ("aegiskv_l1_l2", True),
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise SuiteReadinessError(
            "unsupported_policy", f"no Scheduler mapping for policy {policy!r}"
        ) from exc


def _model_runtime_lineage(
    cell: CellSpec, options: ExecutionOptions
) -> tuple[str, str, str]:
    model = _selected_model(cell)
    model_path = str(options.model_paths.get(model) or "")
    revision = str(options.model_revisions.get(model) or "")
    if not model_path:
        raise SuiteReadinessError(
            "model_path_missing", f"runtime path is required for {model}"
        )
    if not revision:
        raise SuiteReadinessError(
            "model_revision_missing", f"immutable model revision is required for {model}"
        )
    return model, model_path, revision


def _aegis_budget_bytes(cell: CellSpec) -> tuple[int, int]:
    deployment = cell.resolved_config["deployment"]
    axes = deployment["selected_axes"]
    fixed = deployment["fixed"]
    hbm = int(
        axes.get(
            "hbm_budget_gib_per_gpu",
            fixed["gpu_hbm"]["central_aegis_budget_gib_per_gpu"],
        )
    ) * GIB
    dram = int(
        axes.get(
            "dram_budget_gib_per_node",
            fixed["node_dram"]["central_aegis_budget_gib_per_node"],
        )
    ) * GIB
    return hbm, dram


def _runtime_config(
    cell: CellSpec,
    *,
    model_path: str,
) -> tuple[dict[str, Any], Any]:
    deployment = cell.resolved_config["deployment"]["fixed"]
    model_name = _selected_model(cell)
    node_count = _selected_nodes(cell)
    model = deployment["models"][model_name]
    topology = build_offline_physical_topology(
        deployment,
        model_name=model_name,
        node_count=node_count,
    )
    dp_size = int(model["dp_size_by_node_count"][str(node_count)])
    tp_size = int(model["tp_size"])
    pp_size = int(model["pp_size"])
    kv_per_rank = max(
        int(model["kv_bytes_per_token_total"]) // (tp_size * pp_size), 1
    )
    page_size = 16
    hbm_aegis, dram_aegis = _aegis_budget_bytes(cell)
    logical_kv_bytes_per_token = int(model["kv_bytes_per_token_total"])
    dram_aegis_tokens, dram_aegis_tail_bytes = divmod(
        dram_aegis, logical_kv_bytes_per_token
    )
    algorithm_variant = str(
        cell.resolved_config["deployment"]["selected_axes"].get(
            "algorithm_variant", "full"
        )
    )
    primary_and_prefix = int(
        deployment["gpu_hbm"]["primary_kv_budget_bytes_per_gpu"]
    ) + int(deployment["gpu_hbm"]["radix_prefix_budget_bytes_per_gpu"])
    l1_tokens = max((primary_and_prefix + hbm_aegis) // kv_per_rank, 4096)
    l1_pages = max(l1_tokens // page_size, 64)
    per_node_dp = max(dp_size // node_count, 1)
    l2_tokens_per_scheduler = max(
        (
            int(deployment["node_dram"]["hicache_primary_budget_bytes_per_node"])
            + dram_aegis
        )
        // (per_node_dp * kv_per_rank),
        4096,
    )
    l2_pages = max(l2_tokens_per_scheduler // page_size, 64)
    rank_nodes = [
        topology.group(dp_rank, 0)[0].gpu.node_id for dp_rank in range(dp_size)
    ]
    node_index = {
        node.node_id: index for index, node in enumerate(topology.nodes)
    }
    policy_name, protected = _policy_runtime_name(_selected_policy(cell))
    hbm_fraction = min(max(hbm_aegis / max(primary_and_prefix + hbm_aegis, 1), 0.0), 1.0)
    dram_total = int(deployment["node_dram"]["hicache_primary_budget_bytes_per_node"]) + dram_aegis
    dram_fraction = min(max(dram_aegis / max(dram_total, 1), 0.0), 1.0)
    if policy_name == "aegiskv_l2_only":
        hbm_fraction = 0.0
    if policy_name == "aegiskv_l1_only":
        dram_fraction = 0.0
    runtime = {
        "model_path": model_path,
        "scheduler": {
            "dp_size": dp_size,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "load_balance_method": "round_robin",
            "max_prefill_tokens": min(
                int(cell.resolved_config["request"]["fixed"]["context"]["maximum_tokens"]),
                16384,
            ),
            "chunked_prefill_size": int(deployment["aegiskv"]["prefix_chunk_tokens"]),
            "max_running_requests": 64,
        },
        "platform": {
            "accelerator": {
                "name": "a100_sxm",
                "hbm_capacity_gb": int(deployment["cluster"]["gpu_hbm_capacity_bytes"]) / GIB,
            }
        },
        "cache": {
            "page_size": page_size,
            "l1_capacity_pages": l1_pages,
            "aegis_total_l1_capacity_pages": l1_pages,
            "l2_capacity_pages": l2_pages,
        },
        "communication": {"kv_bytes_per_token": kv_per_rank},
        "aegis_l1_l2": {
            "enabled": protected,
            "committed_copy_protocol": protected,
            "design_semantics": DESIGN_SEMANTICS,
            "chunk_tokens": int(deployment["aegiskv"]["prefix_chunk_tokens"]),
            "prefix_chunk_tokens": int(deployment["aegiskv"]["prefix_chunk_tokens"]),
            "decode_chunk_tokens": int(deployment["aegiskv"]["sealed_decode_chunk_tokens"]),
            "host_alignment_tokens": 1,
            "require_explicit_per_node_dram_quota": True,
            "per_node_aegis_dram_quota_bytes": dram_aegis,
            "per_node_aegis_dram_quota_tokens": dram_aegis_tokens,
            "kv_bytes_per_logical_token": logical_kv_bytes_per_token,
            "disable_slack_ordering": algorithm_variant == "no_algorithm_1",
            "initial_alpha": hbm_fraction,
            "initial_beta": dram_fraction,
            "topology": {
                "dp_rank_to_node": [node_index[node] for node in rank_nodes],
                "pp_stages_per_node": pp_size,
            },
        },
        "experiment_runtime_projection": {
            "policy": _selected_policy(cell),
            "scheduler_policy": policy_name,
            "model": model_name,
            "node_count": node_count,
            "dp_size": dp_size,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "kv_bytes_per_token_per_tp_pp_rank": kv_per_rank,
            "hbm_aegis_budget_bytes_per_gpu": hbm_aegis,
            "dram_aegis_budget_bytes_per_node": dram_aegis,
            "dram_aegis_budget_tokens_per_node": dram_aegis_tokens,
            "dram_aegis_unallocatable_tail_bytes_per_node": (
                dram_aegis_tail_bytes
            ),
            "algorithm_variant": algorithm_variant,
            "disable_slack_ordering": algorithm_variant == "no_algorithm_1",
        },
    }
    return runtime, topology


def _copy_fabric(cell: CellSpec) -> CopyFabricConfig:
    deployment = cell.resolved_config["deployment"]
    fixed = deployment["fixed"]["copy_resources"]
    links = fixed["links"]
    dram = fixed["dram_lanes_per_node"]
    effective_fabric = int(
        float(
            deployment["selected_axes"].get(
                "cross_node_bandwidth_gb_s",
                int(links["cross_node_fabric"]["central_effective_bandwidth_bytes_per_s"]) / GB,
            )
        )
        * GB
    )
    pcie_setup = int(links["pcie"]["setup_ns"])
    nic_setup = int(links["nic_rdma"]["setup_ns"])
    fabric_setup = int(links["cross_node_fabric"]["setup_ns"])
    dram_setup = int(dram["setup_ns"])
    return CopyFabricConfig(
        bandwidths=CopyFabricBandwidths(
            gpu_copy_bytes_per_s=int(links["nvlink_nvswitch"]["peak_bandwidth_bytes_per_s"]),
            nvlink_bytes_per_s=int(links["nvlink_nvswitch"]["peak_bandwidth_bytes_per_s"]),
            pcie_bytes_per_s=int(links["pcie"]["peak_bandwidth_bytes_per_s"]),
            nic_bytes_per_s=int(links["nic_rdma"]["peak_bandwidth_bytes_per_s"]),
            fabric_bytes_per_s=effective_fabric,
            dram_write_bytes_per_s=int(dram["bandwidth_bytes_per_s"]),
            dram_read_bytes_per_s=int(dram["bandwidth_bytes_per_s"]),
        ),
        setup=CopyFabricSetupLatencies(
            same_node_d2d_ns=int(links["nvlink_nvswitch"]["setup_ns"]),
            cross_node_d2d_ns=2 * pcie_setup + 2 * nic_setup + fabric_setup,
            same_node_d2h_ns=pcie_setup + dram_setup,
            cross_node_d2h_ns=(
                2 * pcie_setup + 2 * nic_setup + fabric_setup + dram_setup
            ),
            same_node_h2d_ns=dram_setup + pcie_setup,
            cross_node_h2d_ns=(
                dram_setup + 2 * pcie_setup + 2 * nic_setup + fabric_setup
            ),
        ),
        fabric_id="cluster-fabric-0",
    )


def _fallback_token_ids(rid: str, count: int, *, vocab_size: int = 151_936) -> list[int]:
    result: list[int] = []
    for index in range(max(int(count), 0)):
        digest = hashlib.sha256(f"{rid}:{index}".encode("utf-8")).digest()
        result.append(int.from_bytes(digest[:4], "big") % vocab_size)
    return result


def _record_token_ids(record: Mapping[str, Any]) -> list[int]:
    count = int(record["input_tokens"])
    explicit = record.get("input_ids")
    if isinstance(explicit, list):
        values = [int(value) for value in explicit]
        if len(values) != count:
            raise SuiteReadinessError(
                "request_token_count_mismatch",
                f"{record.get('rid')}: input_ids length differs from input_tokens",
            )
        return values
    raw_identity = record.get("input_token_identity")
    if isinstance(raw_identity, Mapping):
        if raw_identity.get("schema") == WEKA_TOKEN_IDENTITY_SCHEMA:
            block_size = int(raw_identity.get("block_size", 0))
            block_ids = raw_identity.get("block_hash_ids")
            namespace = str(raw_identity.get("block_namespace_sha256") or "")
            fallback = str(raw_identity.get("fallback_stream_sha256") or "")
            if (
                block_size <= 0
                or not isinstance(block_ids, list)
                or len(namespace) != 64
                or len(fallback) != 64
                or int(raw_identity.get("token_count", -1)) != count
            ):
                raise SuiteReadinessError(
                    "p11_token_identity_invalid",
                    f"{record.get('rid')}: invalid Weka block token identity",
                )
            values: list[int] = []
            for index in range(count):
                block_index, block_offset = divmod(index, block_size)
                if block_index < len(block_ids):
                    identity = (
                        f"{namespace}:{int(block_ids[block_index])}:{block_offset}"
                    )
                else:
                    identity = f"{fallback}:{index}"
                digest = hashlib.sha256(identity.encode("utf-8")).digest()
                values.append(int.from_bytes(digest[:4], "big") % 151_936)
            return values
        identity = TokenIdentity(
            prefix_stream_sha256=str(raw_identity["prefix_stream_sha256"]),
            prefix_tokens=int(raw_identity["prefix_tokens"]),
            suffix_stream_sha256=str(raw_identity["suffix_stream_sha256"]),
            suffix_tokens=int(raw_identity["suffix_tokens"]),
        )
        values = list(deterministic_token_ids(identity, vocab_size=151_936))
        if len(values) != count:
            raise SuiteReadinessError(
                "request_token_identity_mismatch",
                f"{record.get('rid')}: token identity length differs from input_tokens",
            )
        return values
    return _fallback_token_ids(str(record["rid"]), count)


def _failure_scope(cell: CellSpec) -> str:
    failure = cell.resolved_config["failure"]
    selected = failure["selected_axes"]
    raw = str(
        selected.get(
            "fault_domain",
            selected.get(
                "failure_severity",
                selected.get(
                    "failure_scenario",
                    failure["fixed"]["fault_model"]["central_domain"],
                ),
            ),
        )
    ).lower()
    if raw.startswith("node"):
        return "node"
    if raw.startswith("gpu"):
        return "gpu"
    if raw.startswith("process"):
        return "process"
    return "worker"


def _algorithm_request_fields(cell: CellSpec, source_rank: int, dp_size: int) -> dict[str, Any]:
    variant = str(
        cell.resolved_config["deployment"]["selected_axes"].get(
            "algorithm_variant", "full"
        )
    )
    if variant == "full":
        return {}
    if variant == "no_algorithm_1":
        # Algorithm 1 is Slack ordering.  Its preregistered ablation is a
        # runtime ordering adapter, not the historical request-level
        # ``size_only_backup`` shortcut (which also changes eligibility and
        # backup lifecycle semantics).
        return {}
    if variant == "static_placement":
        return {"aegis_backup_route_plan_dp_rank": (source_rank + 1) % dp_size}
    if variant == "no_algorithm_3":
        return {
            "aegis_online_alpha_adapter": False,
            "aegis_aimd_disabled": True,
        }
    raise SuiteReadinessError(
        "unsupported_algorithm_variant", f"unknown P4 variant {variant!r}"
    )


def _placement_request_fields(
    cell: CellSpec, source_rank: int, dp_size: int
) -> dict[str, Any]:
    placement = cell.resolved_config["deployment"]["selected_axes"].get(
        "replica_placement"
    )
    if placement is None:
        return {}
    model_name = _selected_model(cell)
    deployment = cell.resolved_config["deployment"]["fixed"]
    model = deployment["models"][model_name]
    dp_per_node = int(deployment["cluster"]["gpus_per_node"]) // (
        int(model["tp_size"]) * int(model["pp_size"])
    )
    if dp_per_node < 1:
        raise SuiteReadinessError(
            "replica_placement_impossible", "deployment has no complete DP per node"
        )
    node_start = (source_rank // dp_per_node) * dp_per_node
    if placement == "same_node_peer_hbm":
        if dp_per_node < 2:
            raise SuiteReadinessError(
                "same_node_replica_impossible", "same-node placement needs two DPs per node"
            )
        target = node_start + ((source_rank - node_start + 1) % dp_per_node)
    elif placement == "cross_node_peer_hbm":
        if dp_size <= dp_per_node:
            raise SuiteReadinessError(
                "cross_node_replica_impossible", "cross-node placement needs two nodes"
            )
        target = (source_rank + dp_per_node) % dp_size
    else:
        raise SuiteReadinessError(
            "unsupported_replica_placement", f"unknown placement {placement!r}"
        )
    return {
        "aegis_backup_route_plan_dp_rank": int(target),
        "configured_replica_placement": str(placement),
    }


def _validate_protected_request_slack(
    cell: CellSpec, *, rid: str, simulation: Mapping[str, Any]
) -> None:
    """Fail closed when a current-design protected request lacks causal Slack."""

    _, protected = _policy_runtime_name(_selected_policy(cell))
    if not protected:
        return
    variant = str(
        cell.resolved_config["deployment"]["selected_axes"].get(
            "algorithm_variant", "full"
        )
    )
    if variant == "no_algorithm_1":
        # This preregistered ablation intentionally removes Slack ordering.
        return
    _, reason = resolve_slack_for_design(
        simulation,
        design_semantics=DESIGN_SEMANTICS,
    )
    if reason != "visible_residual_dag_cpm" and cell.group == "P11":
        # P11 includes observed requests for which no admission-time predictor
        # output exists (for example singleton stages).  A complete explicit
        # unusable ticket is Algorithm-1 fail-close-to-zero.  Protection stays
        # enabled; only Slack ordering receives zero.  Missing fields, future
        # information, or a nonzero unusable ticket still block the cell.
        required = {
            "aegis_slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
            "aegis_slack_uses_future_information": False,
            "aegis_slack_residual_dag_visible": True,
            "aegis_slack_metadata_only": True,
            "aegis_slack_usable": False,
            "aegis_slack_reason": "weka_agentx_fail_close_zero",
            "weka_ticket_modeled": False,
            "weka_ticket_publish_causal": True,
        }
        explicit_fail_close = all(
            simulation.get(field) == expected for field, expected in required.items()
        )
        try:
            zero_slack = float(simulation.get("workflow_slack_ms", -1.0)) == 0.0
        except (TypeError, ValueError):
            zero_slack = False
        identities_complete = all(
            str(simulation.get(field) or "").strip()
            for field in (
                "aegis_slack_graph_version",
                "aegis_slack_node_id",
                "aegis_slack_agent_id",
                "aegis_slack_input_provenance_sha256",
            )
        )
        if explicit_fail_close and zero_slack and identities_complete:
            return
    if reason != "visible_residual_dag_cpm":
        raise SuiteReadinessError(
            "causal_slack_input_missing",
            f"protected request {rid!r} lacks a valid admission-time "
            f"{RESIDUAL_DAG_SLACK_SEMANTICS} ticket ({reason})",
            details={
                "rid": str(rid),
                "group": cell.group,
                "algorithm_variant": variant,
                "reason": reason,
                "future_information_must_be_false": True,
            },
        )


def validate_prepared_workload_readiness(
    cell: CellSpec, workload: PreparedWorkload
) -> None:
    """Validate policy inputs without constructing workers or token tensors."""

    for record in workload.records:
        rid = str(record.get("rid") or "")
        simulation: dict[str, Any] = {}
        if cell.group == "P12":
            simulation.update(
                _p12_analytical_slack_fields(
                    rid=rid,
                    workflow_id=str(record.get("workflow") or f"workflow/{rid}"),
                    observed_at_ms=int(record.get("arrival_ns", 0)) / 1e6,
                )
            )
        if isinstance(record.get("simulation"), Mapping):
            simulation.update(copy.deepcopy(dict(record["simulation"])))
        _validate_protected_request_slack(cell, rid=rid, simulation=simulation)


def _materialize_requests(
    cell: CellSpec,
    workload: PreparedWorkload,
    *,
    dp_size: int,
    smoke_limit: int | None,
) -> tuple[list[TokenizedGenerateReqInput], list[dict[str, Any]]]:
    records = list(workload.records)
    if smoke_limit is not None and not (
        cell.group == "P11"
        and workload.metadata.get("atomic_workflow_replay") is True
    ):
        records = records[: max(int(smoke_limit), 0)]
    if not records:
        raise SuiteReadinessError("request_set_empty", f"{cell.cell_id} has no requests")
    policy_name, protected = _policy_runtime_name(_selected_policy(cell))
    scope = _failure_scope(cell)
    requests: list[TokenizedGenerateReqInput] = []
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        rid = str(record.get("rid") or "")
        if not rid:
            raise SuiteReadinessError("request_id_missing", "compiled request has no rid")
        source_rank = int.from_bytes(
            hashlib.sha256(rid.encode("utf-8")).digest()[:8], "big"
        ) % dp_size
        simulation = {
            "server_created_time": int(record.get("arrival_ns", 0)) / 1e9,
            "created_time": int(record.get("arrival_ns", 0)) / 1e9,
            "force_dp_rank": source_rank,
            "aegis_policy": policy_name,
            "backup_kv": protected,
            "critical_agent": protected,
            "aegis_critical_probability": 1.0 if protected else 0.0,
            "workflow_slack_ms": 0.0,
            "aegis_failure_scope": scope,
            "workflow_id": str(record.get("workflow") or f"workflow/{rid}"),
            "agent_id": str(
                record.get("agent_id") or record.get("provider") or "request"
            ),
            "worker_seq": int(
                record.get(
                    "worker_seq",
                    int(record.get("round_index", index)) + 1,
                )
            ),
            "prefix_key": str(record.get("prefix_key") or ""),
            **_algorithm_request_fields(cell, source_rank, dp_size),
            **_placement_request_fields(cell, source_rank, dp_size),
        }
        if cell.group == "P12":
            simulation.update(
                _p12_analytical_slack_fields(
                    rid=rid,
                    workflow_id=str(record.get("workflow") or f"workflow/{rid}"),
                    observed_at_ms=int(record.get("arrival_ns", 0)) / 1e6,
                )
            )
        if protected:
            simulation.pop("force_dp_rank", None)
            simulation["aegis_route_plan_dp_rank"] = source_rank
        extra_simulation = record.get("simulation")
        if isinstance(extra_simulation, Mapping):
            simulation.update(copy.deepcopy(dict(extra_simulation)))
        _validate_protected_request_slack(cell, rid=rid, simulation=simulation)
        request = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid=rid,
                input_ids=_record_token_ids(record),
                max_new_tokens=int(record.get("output_tokens", 1)),
                routing_key=str(record.get("prefix_key") or "") or None,
                simulation=simulation,
            )
        )
        requests.append(request)
        rows.append(
            {
                **copy.deepcopy(dict(record)),
                "schema": "aegiskv.global-des-scheduler-request.v1",
                "cell_id": cell.cell_id,
                "group": cell.group,
                "planned_dp_rank": source_rank,
                "aegis_policy": policy_name,
                "simulation": copy.deepcopy(simulation),
                "failover_replay": False,
                "status": "scheduled",
            }
        )
    return requests, rows


def _failure_count(cell: CellSpec) -> int:
    failure = cell.resolved_config["failure"]
    selected = failure["selected_axes"]
    for name in ("failure_severity", "fault_domain", "failure_scenario"):
        raw = str(selected.get(name, ""))
        if raw.startswith("worker_"):
            try:
                return max(int(raw.rsplit("_", 1)[1]), 1)
            except ValueError:
                return 1
    return 1


def _failure_enabled(cell: CellSpec) -> bool:
    failure = cell.resolved_config["failure"]
    if failure["fixed"]["fault_model"]["enabled"] is not True:
        return False
    selected = failure["selected_axes"]
    return str(selected.get("failure_scenario", "")).lower() != "no_failure"


def _failure_scenario_name(cell: CellSpec) -> str:
    failure = cell.resolved_config["failure"]
    selected = failure["selected_axes"]
    for name in ("failure_scenario", "fault_domain", "failure_severity"):
        if name in selected:
            return str(selected[name])
    return str(failure["fixed"]["fault_model"].get("central_target") or "none")


def _active_exposure_offset(
    cell: CellSpec, *, physical_gpu_count: int
) -> tuple[int, dict[str, Any]]:
    trigger = cell.resolved_config["failure"]["fixed"]["trigger"]
    normalization = trigger.get("scale_normalization")
    if normalization is None:
        value = int(trigger["active_offset_ns"])
        return value, {
            "rule": "fixed_active_offset_v1",
            "active_offset_ns": value,
        }
    selected_gpus = int(physical_gpu_count)
    if selected_gpus <= 0:
        raise SuiteReadinessError(
            "failure_scale_topology_empty",
            "P3 active-exposure normalization requires physical GPUs",
        )
    reference_gpus = int(normalization["reference_node_count"]) * int(
        normalization["reference_gpus_per_node"]
    )
    reference_offset = int(normalization["reference_active_offset_ns"])
    numerator = reference_offset * reference_gpus
    value = (numerator + selected_gpus // 2) // selected_gpus
    return value, {
        "rule": str(normalization["rule"]),
        "reference_gpu_count": reference_gpus,
        "reference_active_offset_ns": reference_offset,
        "selected_gpu_count": selected_gpus,
        "conditioned_cluster_gpu_time_exposure_ns": selected_gpus * value,
        "exposure_window": str(normalization["exposure_window"]),
        "rounding": str(normalization["rounding"]),
        "active_offset_ns": value,
    }


def _build_active_exposure_plan(
    cell: CellSpec,
    workload: PreparedWorkload,
    request_rows: Sequence[Mapping[str, Any]],
    topology: Any,
) -> dict[str, Any] | None:
    trigger = cell.resolved_config["failure"]["fixed"]["trigger"]
    if not _failure_enabled(cell) or trigger.get("mode") != (
        "policy_independent_active_exposure_v1"
    ):
        return None
    if not request_rows:
        raise SuiteReadinessError(
            "active_failure_request_set_empty",
            "active-exposure failure has no executed requests",
        )
    anchor = min(
        request_rows,
        key=lambda row: (int(row.get("arrival_ns", 0)), str(row.get("rid") or "")),
    )
    anchor_rid = str(anchor.get("rid") or "")
    source_rank = int(anchor.get("planned_dp_rank", -1))
    if not anchor_rid or source_rank < 0:
        raise SuiteReadinessError(
            "active_failure_anchor_invalid",
            "pre-execution anchor has no request id or planned source rank",
        )
    source_nodes = sorted(
        {
            str(spec.gpu.node_id)
            for spec in topology.gpus
            if int(spec.gpu.dp_rank) == source_rank
        }
    )
    if len(source_nodes) != 1:
        raise SuiteReadinessError(
            "active_failure_anchor_node_ambiguous",
            "anchor DP rank must map to exactly one physical node",
            details={"dp_rank": source_rank, "nodes": source_nodes},
        )
    scenario = _failure_scenario_name(cell)
    scope = _failure_scope(cell)
    count = _failure_count(cell)
    if scope == "node":
        planned_targets = [
            {"failure_scope": "node", "node_id": source_nodes[0]}
        ]
        target_selector = str(trigger["target_selector"]["node"])
    else:
        if count >= int(max(spec.gpu.dp_rank for spec in topology.gpus) + 1):
            raise SuiteReadinessError(
                "failure_removes_all_dp",
                "active-exposure failure must leave a healthy DP rank",
            )
        dp_size = max(int(spec.gpu.dp_rank) for spec in topology.gpus) + 1
        planned_targets = [
            {"failure_scope": scope, "dp_rank": (source_rank + index) % dp_size}
            for index in range(count)
        ]
        target_selector = str(trigger["target_selector"][scope])
    active_offset_ns, offset_rule = _active_exposure_offset(
        cell, physical_gpu_count=len(topology.gpus)
    )
    executed_request_projection = [
        {
            "rid": str(row["rid"]),
            "arrival_ns": int(row.get("arrival_ns", 0)),
            "input_tokens": int(row.get("input_tokens", 0)),
            "output_tokens": int(row.get("output_tokens", 0)),
            "planned_dp_rank": int(row["planned_dp_rank"]),
        }
        for row in request_rows
    ]
    shadow = trigger["shadow_plan"]
    plan: dict[str, Any] = {
        "schema": "aegiskv.policy-independent-failure-plan.v1",
        "construction": str(shadow["construction"]),
        "request_universe_sha256": str(workload.request_universe_hash),
        "executed_request_set_sha256": _sha256_bytes(
            canonical_json_bytes(executed_request_projection)
        ),
        "seed": int(cell.axes["seed"]),
        "scenario": scenario,
        "physical_topology_sha256": config_hash(topology.as_dict()),
        "anchor_selector": str(trigger["anchor_selector"]),
        "anchor_request_id": anchor_rid,
        "anchor_arrival_ns": int(anchor.get("arrival_ns", 0)),
        "anchor_source_dp_rank": source_rank,
        "anchor_source_node_id": source_nodes[0],
        "target_selector": target_selector,
        "planned_targets": planned_targets,
        "active_offset_rule": offset_rule,
        "active_offset_ns": active_offset_ns,
        "policy_inputs_excluded": list(shadow["excluded_inputs"]),
    }
    plan["shadow_plan_sha256"] = config_hash(plan)
    return plan


def _anchor_owned_stage_keys(
    cluster: OfflineSchedulerCluster, plan: Mapping[str, Any]
) -> tuple[tuple[int, int, int], ...]:
    anchor = str(plan["anchor_request_id"])
    return tuple(
        sorted(
            key
            for key, request_ids in cluster._stage_owned_request_ids.items()
            if anchor in request_ids
        )
    )


def _wait_for_active_anchor(
    cluster: OfflineSchedulerCluster,
    plan: Mapping[str, Any],
    *,
    max_events: int,
) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
    events_used = 0
    expected_rank = int(plan["anchor_source_dp_rank"])
    while True:
        owned = _anchor_owned_stage_keys(cluster, plan)
        if owned:
            ranks = {key[0] for key in owned}
            if ranks != {expected_rank}:
                raise SuiteExecutionError(
                    "active failure anchor route differs from the pre-execution plan: "
                    f"expected DP {expected_rank}, observed {sorted(ranks)}"
                )
            return cluster.global_des.now_ns, events_used, owned
        if events_used >= max_events:
            raise SuiteExecutionError(
                "cell exceeded max_events before the planned failure anchor became active"
            )
        if not cluster.coordinator.has_events():
            raise SuiteExecutionError(
                "event heap drained before the planned failure anchor became active"
            )
        cluster.process_next_event()
        events_used += 1


def _schedule_failures(
    cluster: OfflineSchedulerCluster,
    cell: CellSpec,
    *,
    compiled_timeline: Any | None = None,
) -> tuple[str, ...]:
    if compiled_timeline is not None:
        events = tuple(compiled_timeline.events)
        compiled_timeline.install(cluster)
        return tuple(str(event.event_id) for event in events)
    if not _failure_enabled(cell):
        return ()
    fixed = cell.resolved_config["failure"]["fixed"]
    selected = cell.resolved_config["failure"]["selected_axes"]
    trigger = fixed["trigger"]
    offset_ns = int(trigger["central_time_ns"])
    normalization = trigger.get("scale_normalization")
    if normalization is not None:
        selected_gpu_count = len(cluster.resource_bridge.topology.gpus)
        if selected_gpu_count <= 0:
            raise SuiteReadinessError(
                "failure_scale_topology_empty",
                "P3 failure normalization requires physical GPUs",
            )
        reference_gpu_count = int(
            normalization["reference_node_count"]
        ) * int(normalization["reference_gpus_per_node"])
        numerator = (
            int(normalization["reference_trigger_time_ns"])
            * reference_gpu_count
        )
        # Frozen rule: nearest integer nanosecond, half up.
        offset_ns = (numerator + selected_gpu_count // 2) // selected_gpu_count
    offset_s = offset_ns / 1e9
    horizon_s = float(
        selected.get("repair_horizon_s", fixed["repair"]["central_horizon_s"])
    )
    # Runner binding is the same fail-closed policy used by the compiler.
    # Replacement and routing outage are serially exposed in the configured
    # hard-resource unavailability interval.
    duration_s = horizon_s + (
        float(fixed["repair"]["replacement_route_delay_ms"])
        + float(fixed["repair"]["routing_outage_ms"])
    ) / 1000.0
    if duration_s <= 0:
        raise SuiteReadinessError(
            "repair_horizon_invalid", "hard fail-stop duration must be positive"
        )
    scope = _failure_scope(cell)
    count = _failure_count(cell)
    ids: list[str] = []
    if scope == "node":
        node_ids = [node.node_id for node in cluster.resource_bridge.topology.nodes]
        if len(node_ids) < 2:
            raise SuiteReadinessError(
                "node_failure_no_survivor", "node failure requires at least two nodes"
            )
        targets: Iterable[tuple[int | None, str | None]] = ((None, node_ids[1]),)
    else:
        if count >= cluster.dp_size:
            raise SuiteReadinessError(
                "failure_removes_all_dp", "hard failure must leave a healthy DP rank"
            )
        targets = ((rank, None) for rank in range(1, count + 1))
    for index, (dp_rank, node_id) in enumerate(targets):
        event_id = f"{cell.cell_id}-failure-{index}"
        cluster.schedule_fail_stop(
            event_id=event_id,
            offset_s=offset_s,
            duration_s=duration_s,
            failure_scope=scope,
            dp_rank=dp_rank,
            node_id=node_id,
            payload={
                "cell_id": cell.cell_id,
                "configured_failure_scope": scope,
                "route_fence_required": True,
                "failure_target_selector": "contract_test_dense_second_v1",
                "repair_delay_policy": "serial_sum_v1",
                "soft_metadata_failure_forbidden": True,
                "trigger_time_ns": offset_ns,
            },
        )
        ids.append(event_id)
    return tuple(ids)


def _schedule_active_plan_without_compiler(
    cluster: OfflineSchedulerCluster,
    cell: CellSpec,
    plan: Mapping[str, Any],
    *,
    anchor_active_at_ns: int,
) -> tuple[str, ...]:
    """Contract-test adapter for runtime overrides.

    Formal cells bind the same plan through ``compile_failure_timeline``.  A
    runtime override has no compiled deployment, so this path preserves the
    identical causal trigger/target contract and records why the compiler was
    unavailable.
    """

    fixed = cell.resolved_config["failure"]["fixed"]
    selected = cell.resolved_config["failure"]["selected_axes"]
    horizon_s = float(
        selected.get("repair_horizon_s", fixed["repair"]["central_horizon_s"])
    )
    duration_s = horizon_s + (
        float(fixed["repair"]["replacement_route_delay_ms"])
        + float(fixed["repair"]["routing_outage_ms"])
    ) / 1000.0
    if duration_s <= 0:
        raise SuiteReadinessError(
            "repair_horizon_invalid", "hard fail-stop duration must be positive"
        )
    inject_at_ns = int(anchor_active_at_ns) + int(plan["active_offset_ns"])
    ids: list[str] = []
    for index, target in enumerate(plan["planned_targets"]):
        event_id = f"{cell.cell_id}-failure-{index}"
        cluster.schedule_fail_stop(
            event_id=event_id,
            offset_s=inject_at_ns / 1e9,
            duration_s=duration_s,
            failure_scope=str(target["failure_scope"]),
            dp_rank=(
                None if "dp_rank" not in target else int(target["dp_rank"])
            ),
            node_id=(
                None if "node_id" not in target else str(target["node_id"])
            ),
            payload={
                "cell_id": cell.cell_id,
                "configured_failure_scope": str(target["failure_scope"]),
                "route_fence_required": True,
                "failure_target_selector": str(plan["target_selector"]),
                "repair_delay_policy": "serial_sum_v1",
                "soft_metadata_failure_forbidden": True,
                "trigger_mode": "policy_independent_active_exposure_v1",
                "trigger_time_ns": inject_at_ns,
                "anchor_active_at_ns": int(anchor_active_at_ns),
                "anchor_request_id": str(plan["anchor_request_id"]),
                "anchor_source_dp_rank": int(plan["anchor_source_dp_rank"]),
                "anchor_source_node_id": str(plan["anchor_source_node_id"]),
                "shadow_plan_sha256": str(plan["shadow_plan_sha256"]),
                "active_exposure_plan": copy.deepcopy(dict(plan)),
                "contract_test_runtime_override": True,
            },
        )
        ids.append(event_id)
    return tuple(ids)


def _clone_failover_replay(
    source: TokenizedGenerateReqInput,
    *,
    event_id: str,
    failed_dp_rank: int,
    replay_at_s: float,
    replay_spec: Mapping[str, Any],
) -> TokenizedGenerateReqInput:
    source_sim = copy.deepcopy(
        source.sampling_params.custom_params.get("simulation", {})
    )
    source_sim.pop("force_dp_rank", None)
    # A retry is a new execution route.  Do not leak source-Scheduler restore,
    # lookup, or already-sent protection-plan state into it.  Physical
    # re-protection/job activation is intentionally outside this Router-output
    # closure and remains owned by the physical recovery task.
    for key in tuple(source_sim):
        if key.startswith(
            (
                "restore_",
                "aegis_replay_",
                "aegis_retry_",
                "peer_gpu_lookup_",
                "host_backup_lookup_",
                "l3_backup_lookup_",
            )
        ) or key in {
            "aegis_backup_plan_sent",
            "aegis_backup_plan_acknowledged",
            "aegis_backup_plan_id",
            "aegis_candidate_retry",
            "aegis_candidate_failed_dp_rank",
            "aegis_candidate_pre_failure_output_tokens",
            "aegis_candidate_pre_failure_total_tokens",
            "aegis_candidate_backed_up_tokens",
            "aegis_candidate_recompute_tokens",
            "aegis_candidate_recovery_action",
            "aegis_candidate_failover_overhead_s",
        }:
            source_sim.pop(key, None)
    backup_key = str(source_sim.get("aegis_backup_key") or source.rid)
    committed = int(replay_spec["committed_output_position"])
    remaining = int(replay_spec["remaining_max_new_tokens"])
    logical_id = str(replay_spec["logical_request_id"])
    next_route_version = int(replay_spec["next_route_version"])
    source_sim.update(
        {
            "server_created_time": replay_at_s,
            "created_time": replay_at_s,
            "aegis_backup_key": backup_key,
            "failover_replay": True,
            "soft_failure_injected": False,
            "hard_failure_injected": True,
            "failure_event_id": event_id,
            "failed_dp_rank": int(failed_dp_rank),
            "pre_failure_total_tokens": int(
                replay_spec["pre_failure_total_tokens"]
            ),
            "backup_kv": False,
            "aegis_logical_request_id": logical_id,
            "aegis_replay_of": str(source.rid),
            "aegis_route_version": next_route_version,
            "aegis_recovery_source_route_version": int(
                replay_spec["failed_route_version"]
            ),
            "aegis_router_replay_contract": REPLAY_CONTRACT_SCHEMA,
            "aegis_committed_output_position": committed,
            "aegis_remaining_max_new_tokens": remaining,
            "aegis_metadata_continuation_digest": str(
                replay_spec["continuation_state_digest"]
            ),
            "aegis_metadata_compatibility_hash": str(
                replay_spec["continuation_compatibility_hash"]
            ),
            "aegis_metadata_prefix_hash": str(
                replay_spec["continuation_prefix_hash"]
            ),
            "aegis_continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
            "synthetic_output_token_id": int(
                replay_spec["synthetic_output_token_id"]
            ),
            "aegis_reprotection_scope": (
                "router_output_route_chain_with_physical_recovery_admission_gate"
            ),
        }
    )
    return TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid=f"{source.rid}.replay.{event_id}",
            input_ids=[int(token) for token in replay_spec["replay_input_ids"]],
            max_new_tokens=remaining,
            routing_key=(
                str(replay_spec["routing_key"])
                if replay_spec.get("routing_key") is not None
                else None
            ),
            simulation=source_sim,
        )
    )


def _request_stats(result: Any) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for snapshot in result.rank_snapshots.values():
        for rid, stats in snapshot.get("request_stats", {}).items():
            merged[str(rid)] = copy.deepcopy(dict(stats))
    return merged


def _finalize_request_rows(
    rows: list[dict[str, Any]],
    result: Any,
    *,
    replay_source_by_id: Mapping[str, str],
) -> list[dict[str, Any]]:
    stats = _request_stats(result)
    router_executions = router_execution_index(
        getattr(result, "router_output_ledger", {})
    )
    killed: dict[str, str] = {}
    for failure in result.failure_events:
        event_id = str(failure.get("event_id") or "")
        for rid in failure.get("affected_request_ids", ()):
            killed[str(rid)] = event_id
    finalized: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row["rid"])
        request_stats = stats.get(rid)
        router_execution = router_executions.get(rid, {})
        if router_execution.get("state") == "completed":
            status = "completed"
        elif request_stats is not None:
            status = str(request_stats.get("status", "completed"))
        elif rid in killed:
            status = "failed_superseded_by_replay"
        else:
            status = str(row.get("status", "unknown"))
        finalized.append(
            {
                **row,
                "status": status,
                "request_stats": request_stats or {},
                "failure_event_id": killed.get(rid),
                "replay_of": replay_source_by_id.get(rid),
                "router_output": router_execution,
                "logical_request_id": (
                    router_execution.get("logical_request_id")
                    or row.get("logical_request_id")
                    or rid
                ),
                "route_version": (
                    router_execution.get("route_version")
                    if router_execution
                    else row.get("route_version", 1)
                ),
            }
        )
    return finalized


def _validate_terminal_request_closure(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless every original/replay chain reaches a terminal row."""

    terminal_statuses = {"completed", "failed_superseded_by_replay"}
    by_rid: dict[str, dict[str, Any]] = {}
    successors: dict[str, list[str]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        rid = str(row.get("rid") or "")
        if not rid:
            raise SuiteExecutionError(
                "request terminal audit found an empty request id"
            )
        if rid in by_rid:
            raise SuiteExecutionError(
                f"request terminal audit found duplicate rid {rid!r}"
            )
        by_rid[rid] = row
        replay_of = str(row.get("replay_of") or "")
        if bool(row.get("failover_replay", False)):
            if not replay_of:
                raise SuiteExecutionError(
                    f"failover replay {rid!r} has no replay_of lineage"
                )
            successors.setdefault(replay_of, []).append(rid)
        elif replay_of:
            raise SuiteExecutionError(
                f"original request {rid!r} unexpectedly has replay_of lineage"
            )

    nonterminal = {
        rid: str(row.get("status", "unknown"))
        for rid, row in by_rid.items()
        if str(row.get("status", "unknown")) not in terminal_statuses
    }
    if nonterminal:
        preview = ", ".join(
            f"{rid}={status}" for rid, status in sorted(nonterminal.items())[:8]
        )
        raise SuiteExecutionError(
            "request terminal audit rejected nonterminal rows: " + preview
        )

    superseded = {
        rid
        for rid, row in by_rid.items()
        if str(row.get("status")) == "failed_superseded_by_replay"
    }
    for rid in sorted(superseded):
        if not str(by_rid[rid].get("failure_event_id") or ""):
            raise SuiteExecutionError(
                f"superseded request {rid!r} has no failure_event_id"
            )
        if not successors.get(rid):
            raise SuiteExecutionError(
                f"superseded request {rid!r} has no replay successor"
            )

    originals = sorted(
        rid
        for rid, row in by_rid.items()
        if not bool(row.get("failover_replay", False))
    )
    if not originals:
        raise SuiteExecutionError("request terminal audit found no originals")

    terminal_leaf_by_original: dict[str, str] = {}
    reachable: set[str] = set()
    for original in originals:
        current = original
        visited: set[str] = set()
        reachable.add(current)
        while str(by_rid[current].get("status")) == (
            "failed_superseded_by_replay"
        ):
            if current in visited:
                raise SuiteExecutionError(
                    f"request replay lineage contains a cycle at {current!r}"
                )
            visited.add(current)
            next_ids = sorted(successors.get(current, ()))
            if len(next_ids) != 1:
                raise SuiteExecutionError(
                    f"request {current!r} has {len(next_ids)} replay successors"
                )
            current = next_ids[0]
            if current not in by_rid:
                raise SuiteExecutionError(
                    f"request replay lineage references missing rid {current!r}"
                )
            reachable.add(current)
        if str(by_rid[current].get("status")) != "completed":
            raise SuiteExecutionError(
                f"request replay chain for {original!r} does not end completed"
            )
        terminal_leaf_by_original[original] = current

    orphan_replays = sorted(
        rid
        for rid, row in by_rid.items()
        if bool(row.get("failover_replay", False))
        and str(row.get("replay_of") or "") not in by_rid
    )
    if orphan_replays:
        raise SuiteExecutionError(
            "request terminal audit found orphan replays: "
            + ", ".join(orphan_replays[:8])
        )
    unreachable = sorted(set(by_rid) - reachable)
    if unreachable:
        raise SuiteExecutionError(
            "request terminal audit found replay rows outside every original "
            "chain: "
            + ", ".join(unreachable[:8])
        )
    return {
        "status": "PASS",
        "row_count": len(by_rid),
        "original_count": len(originals),
        "replay_count": len(by_rid) - len(originals),
        "completed_leaf_count": len(terminal_leaf_by_original),
        "superseded_count": len(superseded),
        "nonterminal_count": 0,
        "terminal_leaf_by_original": dict(sorted(terminal_leaf_by_original.items())),
    }


def _validate_router_output_conservation(
    export: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate the artifact-facing logical output chains."""

    if str(export.get("schema") or "") != ROUTER_LEDGER_SCHEMA:
        raise SuiteExecutionError("Router output ledger schema is missing or unsupported")
    logical_requests = export.get("logical_requests")
    if not isinstance(logical_requests, list):
        raise SuiteExecutionError("Router output ledger has no logical request list")
    total_budget = 0
    total_committed = 0
    replay_count = 0
    route_chains: dict[str, list[int]] = {}
    for logical in logical_requests:
        if not isinstance(logical, Mapping):
            raise SuiteExecutionError("Router output ledger logical row is malformed")
        logical_id = str(logical.get("logical_request_id") or "")
        budget = int(logical.get("output_budget", -1))
        committed = int(logical.get("committed_output_position", -1))
        published = int(logical.get("published_output_position", -1))
        if not logical_id or budget < 0:
            raise SuiteExecutionError("Router logical request identity/budget is invalid")
        if logical.get("state") != "completed":
            raise SuiteExecutionError(
                f"Router logical request {logical_id!r} is not completed"
            )
        if committed != budget or published != budget:
            raise SuiteExecutionError(
                f"Router logical output conservation failed for {logical_id!r}"
            )
        entries = logical.get("output_entries")
        if not isinstance(entries, list):
            raise SuiteExecutionError(f"Router entries missing for {logical_id!r}")
        sequences = [int(entry["sequence"]) for entry in entries]
        if sequences != list(range(1, budget + 1)):
            raise SuiteExecutionError(
                f"Router output has a gap/duplicate for {logical_id!r}"
            )
        executions = logical.get("executions")
        if not isinstance(executions, list) or not executions:
            raise SuiteExecutionError(
                f"Router execution chain missing for {logical_id!r}"
            )
        routes = [int(item["route_version"]) for item in executions]
        if routes != list(range(routes[0], routes[0] + len(routes))):
            raise SuiteExecutionError(
                f"Router route versions are discontinuous for {logical_id!r}"
            )
        owned = [
            int(sequence)
            for execution in executions
            for sequence in execution.get("committed_sequences", ())
        ]
        if owned != list(range(1, budget + 1)):
            raise SuiteExecutionError(
                f"Router execution outputs are not conserved for {logical_id!r}"
            )
        if len(executions) > 1 and logical.get("prompt_identity_mode") != "token_ids":
            raise SuiteExecutionError(
                f"formal replay lacks canonical prompt token IDs for {logical_id!r}"
            )
        for index, execution in enumerate(executions[1:], 1):
            replay_count += 1
            predecessor = executions[index - 1]
            base = int(execution["base_output_sequence"])
            if not bool(execution.get("strict_replay_contract", False)):
                raise SuiteExecutionError(
                    f"Router replay contract is non-strict for {logical_id!r}"
                )
            if execution.get("predecessor_execution_id") != predecessor.get(
                "execution_id"
            ):
                raise SuiteExecutionError(
                    f"Router replay predecessor mismatch for {logical_id!r}"
                )
            if base != int(
                predecessor.get("failure_committed_output_position", -1)
            ):
                raise SuiteExecutionError(
                    f"Router replay does not start at committed fence for {logical_id!r}"
                )
            if int(execution["requested_new_tokens"]) != budget - base:
                raise SuiteExecutionError(
                    f"Router replay max_new_tokens is not N-k for {logical_id!r}"
                )
            if execution.get("committed_sequences") and sorted(
                execution.get("expected_ready_ranks", ())
            ) != sorted(execution.get("ready_ranks", ())):
                raise SuiteExecutionError(
                    f"Router replay lacks all-rank ready ACK for {logical_id!r}"
                )
        route_chains[logical_id] = routes
        total_budget += budget
        total_committed += committed
    if total_budget != total_committed:
        raise SuiteExecutionError("Router suite-level logical output is not conserved")
    return {
        "schema": "aegiskv.router-output-conservation-audit.v1",
        "status": "PASS",
        "logical_request_count": len(logical_requests),
        "replay_count": replay_count,
        "logical_output_budget_tokens": total_budget,
        "logical_committed_and_published_tokens": total_committed,
        "route_chains": dict(sorted(route_chains.items())),
        "sequence_gap_count": 0,
        "sequence_duplicate_count": 0,
        "accepted_stale_output_count": 0,
        "continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
        "real_logits_or_rng_continuation_claimed": False,
        "physical_recovery_job_activation_gate_implemented": True,
        "physical_recovery_job_activation_gate_in_scope": bool(
            export.get("physical_recovery_job_activation_gate_in_scope", False)
        ),
    }


def _validate_physical_recovery_admission_gate(
    router_export: Mapping[str, Any],
    physical_aegis_export: Mapping[str, Any],
) -> dict[str, Any]:
    """Require every output-producing recovery admission to cross its gate.

    The Router-wide boolean is insufficient when a cell contains several
    replays: one correctly gated replay must not mask another replay whose
    Scheduler output bypassed adoption/fallback evidence.  This audit joins
    the parent-owned admission table to each execution's immutable PP0 ACK.
    """

    raw_gate = physical_aegis_export.get("recovery_admission_gate", {})
    if not isinstance(raw_gate, Mapping):
        raise SuiteExecutionError("physical recovery admission export is malformed")
    admissions = raw_gate.get("admissions", ())
    if not isinstance(admissions, (list, tuple)):
        raise SuiteExecutionError("physical recovery admission table is malformed")
    pending_holds = raw_gate.get("pending_holds", ())
    if not isinstance(pending_holds, (list, tuple)):
        raise SuiteExecutionError("physical recovery pending-hold table is malformed")
    if pending_holds:
        raise SuiteExecutionError("physical recovery admission export retains orphan holds")

    executions = router_execution_index(router_export)
    terminal_states = {"ADOPTED", "FALLBACK_RECOMPUTE"}
    checked: list[dict[str, Any]] = []
    for raw in admissions:
        if not isinstance(raw, Mapping):
            raise SuiteExecutionError("physical recovery admission row is malformed")
        state = str(raw.get("state") or "")
        if state not in terminal_states:
            continue
        request_id = str(raw.get("request_id") or "")
        if not request_id or request_id not in executions:
            raise SuiteExecutionError(
                f"physical recovery admission has no Router execution: {request_id!r}"
            )
        execution = executions[request_id]
        ack = execution.get("ready_ack_evidence")
        if not isinstance(ack, Mapping) or not bool(
            ack.get("physical_recovery_job_gate_in_scope", False)
        ):
            raise SuiteExecutionError(
                f"Router replay bypassed physical recovery gate: {request_id!r}"
            )
        gate = ack.get("physical_recovery_gate")
        if not isinstance(gate, Mapping) or str(gate.get("state") or "") != state:
            raise SuiteExecutionError(
                f"Router/admission recovery gate state mismatch: {request_id!r}"
            )
        if str(gate.get("group_id") or "") != str(raw.get("group_id") or ""):
            raise SuiteExecutionError(
                f"Router/admission recovery group mismatch: {request_id!r}"
            )
        checked.append(
            {
                "request_id": request_id,
                "state": state,
                "group_id": str(raw.get("group_id") or ""),
                "route_version": int(raw.get("execution_route_version", 0)),
            }
        )

    observed = bool(
        router_export.get(
            "physical_recovery_job_activation_gate_in_scope", False
        )
    )
    if checked and not observed:
        raise SuiteExecutionError(
            "per-replay recovery gates exist but Router aggregate flag is false"
        )
    return {
        "schema": "aegiskv.physical-recovery-admission-gate-audit.v1",
        "status": "PASS",
        "implemented": True,
        "observed": observed,
        "terminal_admission_count": len(checked),
        "checked_executions": checked,
        "pending_hold_count": 0,
        "admission_export": copy.deepcopy(dict(raw_gate)),
    }


def _drive_with_hard_failure_replay(
    cluster: OfflineSchedulerCluster,
    *,
    cell: CellSpec,
    requests: Sequence[TokenizedGenerateReqInput],
    request_rows: list[dict[str, Any]],
    failure_ids: Sequence[str],
    max_events: int,
    requests_already_scheduled: bool = False,
    initial_events_used: int = 0,
) -> tuple[Any, list[dict[str, Any]], dict[str, str]]:
    expected = {str(request.rid) for request in requests}
    row_by_rid = {str(row["rid"]): row for row in request_rows}
    replay_source_by_id: dict[str, str] = {}
    if not requests_already_scheduled:
        for request in requests:
            arrival_s = float(
                request.sampling_params.custom_params["simulation"][
                    "server_created_time"
                ]
            )
            cluster.schedule_request_arrival(request, at_s=arrival_s)
    failure_set = {str(value) for value in failure_ids}
    injected: set[str] = set()
    events_used = int(initial_events_used)
    while injected != failure_set:
        if events_used >= max_events:
            raise SuiteExecutionError(
                "cell exceeded max_events before all hard failures injected"
            )
        if not cluster.coordinator.has_events():
            raise SuiteExecutionError(
                f"event heap drained before failures injected: {sorted(failure_set - injected)}"
            )
        event = cluster.process_next_event()
        events_used += 1
        if event.kind != "hard_failure_inject":
            continue
        payload = dict(event.payload or {})
        event_id = str(payload.get("event_id") or "")
        if event_id not in failure_set:
            continue
        injected.add(event_id)
        active = cluster._active_failures.get(event_id)
        if not isinstance(active, dict):
            raise SuiteExecutionError(
                f"hard failure {event_id} produced no active fail-stop record"
            )
        routing_outage_s = float(
            cell.resolved_config["failure"]["fixed"]["repair"][
                "routing_outage_ms"
            ]
        ) / 1000.0
        replay_at = cluster.coordinator.now_s + routing_outage_s
        failed_ranks = tuple(int(value) for value in active["affected_dp_ranks"])
        for original_rid in active.get("affected_request_ids", ()):
            original_id = str(original_rid)
            original = cluster._request_catalog.get(original_id)
            if original is None:
                raise SuiteExecutionError(
                    f"failure {event_id} lost request catalog entry {original_id}"
                )
            original_sim = original.sampling_params.custom_params.get(
                "simulation", {}
            )
            failed_rank = int(
                original_sim.get(
                    "failed_dp_rank",
                    failed_ranks[0] if failed_ranks else -1,
                )
            )
            replay_spec = cluster.router_output_ledger.replay_spec(
                original_id,
                event_id=event_id,
            )
            if not bool(replay_spec["replay_required"]):
                # The Router had already committed and published N outputs.
                # A new execution would violate exactly-once completion.
                continue
            replay = _clone_failover_replay(
                original,
                event_id=event_id,
                failed_dp_rank=failed_rank,
                replay_at_s=replay_at,
                replay_spec=replay_spec,
            )
            expected.discard(original_id)
            expected.add(str(replay.rid))
            replay_source_by_id[str(replay.rid)] = original_id
            cluster.schedule_request_arrival(replay, at_s=replay_at)
            source_row = row_by_rid[original_id]
            replay_row = {
                **copy.deepcopy(source_row),
                "rid": str(replay.rid),
                "arrival_ns": int(round(replay_at * 1e9)),
                "failover_replay": True,
                "replay_of": original_id,
                "failure_event_id": event_id,
                "planned_dp_rank": None,
                "status": "scheduled",
                "logical_request_id": str(
                    replay_spec["logical_request_id"]
                ),
                "route_version": int(replay_spec["next_route_version"]),
                "committed_output_position_at_replay": int(
                    replay_spec["committed_output_position"]
                ),
                "remaining_max_new_tokens": int(
                    replay_spec["remaining_max_new_tokens"]
                ),
                "pre_failure_total_tokens": int(
                    replay_spec["pre_failure_total_tokens"]
                ),
                "continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
            }
            request_rows.append(replay_row)
            row_by_rid[str(replay.rid)] = replay_row
    result = cluster.run_until_complete(
        expected_request_ids=expected,
        max_events=max(max_events - events_used, 1),
    )
    return result, request_rows, replay_source_by_id


def _validate_active_failure_exposure(
    *,
    plan: Mapping[str, Any] | None,
    anchor_active_at_ns: int | None,
    failures: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if plan is None:
        return {
            "schema": "aegiskv.active-failure-exposure-audit.v1",
            "status": "NOT_APPLICABLE",
            "required": False,
        }
    if anchor_active_at_ns is None:
        raise SuiteExecutionError(
            "active-exposure plan has no causal Scheduler activation timestamp"
        )
    original_ids = {
        str(row["rid"])
        for row in request_rows
        if not bool(row.get("failover_replay", False))
    }
    affected_originals = sorted(
        original_ids.intersection(
            str(rid)
            for failure in failures
            for rid in failure.get("affected_request_ids", ())
        )
    )
    anchor = str(plan["anchor_request_id"])
    if not failures:
        raise SuiteExecutionError(
            "formal active-exposure cell completed without an injected failure"
        )
    if not affected_originals:
        raise SuiteExecutionError(
            "formal failure hit no active original request"
        )
    if anchor not in affected_originals:
        raise SuiteExecutionError(
            "formal failure did not hit the pre-execution anchor request"
        )
    expected_inject_ns = int(anchor_active_at_ns) + int(
        plan["active_offset_ns"]
    )
    observed_targets: list[dict[str, Any]] = []
    for failure in failures:
        if str(failure.get("shadow_plan_sha256") or "") != str(
            plan["shadow_plan_sha256"]
        ):
            raise SuiteExecutionError(
                "injected failure lost the pre-execution shadow plan hash"
            )
        observed_ns = int(
            round(float(failure.get("injected_offset_s", -1.0)) * 1e9)
        )
        if observed_ns != expected_inject_ns:
            raise SuiteExecutionError(
                "active failure timestamp differs from anchor activation plus "
                f"the planned offset ({observed_ns} != {expected_inject_ns})"
            )
        scope = str(failure.get("failure_scope") or "")
        target = {"failure_scope": scope}
        if scope == "node":
            target["node_id"] = str(failure.get("node_id") or "")
        else:
            target["dp_rank"] = int(failure.get("dp_rank", -1))
        observed_targets.append(target)
    if observed_targets != list(plan["planned_targets"]):
        raise SuiteExecutionError(
            "injected physical targets differ from the pre-execution plan"
        )
    return {
        "schema": "aegiskv.active-failure-exposure-audit.v1",
        "status": "PASS",
        "required": True,
        "shadow_plan_sha256": str(plan["shadow_plan_sha256"]),
        "anchor_request_id": anchor,
        "anchor_source_dp_rank": int(plan["anchor_source_dp_rank"]),
        "anchor_source_node_id": str(plan["anchor_source_node_id"]),
        "anchor_active_at_ns": int(anchor_active_at_ns),
        "active_offset_ns": int(plan["active_offset_ns"]),
        "planned_trigger_time_ns": expected_inject_ns,
        "planned_targets": copy.deepcopy(list(plan["planned_targets"])),
        "affected_original_request_ids": affected_originals,
        "policy_outcome_selection_forbidden": True,
    }


def _implementation_hash() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    inputs = (
        Path(__file__).resolve(),
        Path(__file__).with_name("global_des_suite_metrics.py"),
        Path(__file__).with_name("global_des_workload.py"),
        Path(__file__).with_name("global_des_config.py"),
        Path(__file__).with_name("global_des_runtime_compiler.py"),
        Path(__file__).with_name("global_des_causal_workload.py"),
        repo_root / "python/sglsim/srt/managers/aegis_l1_l2.py",
        repo_root / "python/sglsim/srt/managers/aegis_independent_tier_manager.py",
        repo_root / "python/sglsim/srt/managers/data_parallel_controller.py",
        repo_root / "python/sglsim/srt/managers/scheduler.py",
        repo_root / "python/sglsim/srt/mem_cache/cache_operation.py",
        repo_root / "python/sglsim/srt/mem_cache/hicache.py",
        repo_root / "python/sglsim/srt/mem_cache/hiradix_cache.py",
        repo_root / "python/sglsim/srt/mem_cache/kv_cache_pool.py",
        repo_root / "python/sglsim/srt/mem_cache/radix_cache.py",
        repo_root / "python/sglsim/srt/simulation/aegis_independent_tiers.py",
        repo_root / "python/sglsim/srt/simulation/global_des.py",
        repo_root / "python/sglsim/srt/simulation/global_des_artifacts.py",
        repo_root / "python/sglsim/srt/simulation/offline_cluster.py",
        repo_root / "python/sglsim/srt/simulation/offline_coordinator.py",
        repo_root / "python/sglsim/srt/simulation/offline_failure_domain.py",
        repo_root / "python/sglsim/srt/simulation/offline_hicache_transfer_bridge.py",
        repo_root / "python/sglsim/srt/simulation/offline_worker.py",
        repo_root / "python/sglsim/srt/simulation/offline_resource_bridge.py",
        repo_root / "python/sglsim/srt/simulation/offline_aegis_transfer_bridge.py",
        repo_root / "python/sglsim/srt/simulation/offline_router_ledger.py",
        *sorted(CONFIG_ROOT.rglob("*.json")),
    )
    digest = hashlib.sha256()
    for path in inputs:
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def execute_prepared_cell(
    cell: CellSpec,
    workload: PreparedWorkload,
    cell_dir: Path,
    *,
    options: ExecutionOptions,
    p0_lineage: Mapping[str, Any],
    runtime_config_override: Mapping[str, Any] | None = None,
    physical_topology_override: Any | None = None,
) -> CellExecution:
    """Execute and atomically publish one cell through real Scheduler workers.

    Runtime/topology overrides exist for contract tests only.  They are fully
    captured in the cell audit and do not bypass the Scheduler/GlobalDES path.
    """

    validate_prepared_workload_readiness(cell, workload)
    model_name, model_path, model_revision = _model_runtime_lineage(cell, options)
    implementation_revision = _implementation_hash()
    compiled_deployment = None
    compiled_failure_timeline = None
    compiler_bindings = None
    compiler_projection: dict[str, Any]
    if runtime_config_override is None:
        # Import locally to keep runner/bootstrap imports acyclic and to make
        # the compiler an execution-time fail-closed gate rather than a module
        # import side effect.
        from .global_des_runtime_compiler import (
            RuntimeBindings,
            compile_deployment,
            compile_failure_timeline,
        )

        def _compiler_layer(layer_name: str) -> dict[str, Any]:
            layer = copy.deepcopy(dict(cell.resolved_config[layer_name]))
            selected_axes = dict(layer.pop("selected_axes", {}))
            layer["axes"] = {
                name: [copy.deepcopy(value)]
                for name, value in selected_axes.items()
            }
            return layer

        compiler_bindings = RuntimeBindings(
            model_revisions={model_name: model_revision},
            simulator_revision=implementation_revision,
            failure_target_selector="active_anchor_plan_v1",
            repair_delay_policy="serial_sum_v1",
        )
        compiled_deployment = compile_deployment(
            _compiler_layer("deployment"),
            cell.axes,
            compiler_bindings,
        )
        active_trigger = (
            cell.resolved_config["failure"]["fixed"]["trigger"].get("mode")
            == "policy_independent_active_exposure_v1"
            and _failure_enabled(cell)
        )
        if not active_trigger:
            compiled_failure_timeline = compile_failure_timeline(
                _compiler_layer("failure"),
                cell.axes,
                compiled_deployment,
                compiler_bindings,
            )
        runtime_config, topology = _runtime_config(cell, model_path=model_path)
        if topology.as_dict() != compiled_deployment.topology.as_dict():
            raise SuiteReadinessError(
                "compiled_topology_projection_mismatch",
                "runner topology differs from the deployment compiler",
            )
        compiled_aegis = compiled_deployment.simulator_config_overlay[
            "aegis_l1_l2"
        ]
        projected_aegis = runtime_config["aegis_l1_l2"]
        for field_name in (
            "require_explicit_per_node_dram_quota",
            "per_node_aegis_dram_quota_bytes",
            "per_node_aegis_dram_quota_tokens",
            "kv_bytes_per_logical_token",
            "disable_slack_ordering",
        ):
            if projected_aegis.get(field_name) != compiled_aegis.get(field_name):
                raise SuiteReadinessError(
                    "compiled_runtime_projection_mismatch",
                    f"runner aegis_l1_l2.{field_name} differs from compiler",
                )
        topology = compiled_deployment.topology
        runtime_config["runtime_provenance"] = copy.deepcopy(
            dict(compiled_deployment.simulator_config_overlay["runtime_provenance"])
        )
        compiler_projection = {
            "deployment": compiled_deployment.as_dict(),
            "failure_timeline": (
                None
                if compiled_failure_timeline is None
                else compiled_failure_timeline.as_dict()
            ),
            "runtime_bindings": {
                "simulator_revision_strategy": (
                    "runtime_implementation_content_sha256_v1"
                ),
                "failure_target_selector": "active_anchor_plan_v1",
                "repair_delay_policy": "serial_sum_v1",
            },
        }
    else:
        runtime_config = copy.deepcopy(dict(runtime_config_override))
        topology = physical_topology_override
        if topology is None:
            raise ValueError("physical_topology_override is required with runtime override")
        compiler_projection = {
            "contract_test_runtime_override": True,
            "deployment": None,
            "failure_timeline": None,
        }
    if not bool(runtime_config.setdefault("aegis_l1_l2", {}).get("enabled", False)):
        # The latest independent-tier manager is instantiated only for an
        # active protection policy.  Baselines remain Scheduler-real but must
        # not accidentally opt into a partial latest-design protocol.
        runtime_config["aegis_l1_l2"].pop("design_semantics", None)
        runtime_config["aegis_l1_l2"].pop("committed_copy_protocol", None)
    runtime_config.setdefault("aegis_l1_l2", {})["model_revision"] = model_revision
    runtime_config["aegis_l1_l2"].setdefault("kv_abi", "sglsim-bf16-kv-v1")
    runtime_config["aegis_l1_l2"].setdefault(
        "parallel_layout",
        (
            f"dp{runtime_config['scheduler']['dp_size']}"
            f"-pp{runtime_config['scheduler'].get('pp_size', 1)}"
            f"-tp{runtime_config['scheduler'].get('tp_size', 1)}"
        ),
    )
    smoke_limit = options.max_requests_per_cell if options.smoke else None
    requests, request_rows = _materialize_requests(
        cell,
        workload,
        dp_size=int(runtime_config["scheduler"]["dp_size"]),
        smoke_limit=smoke_limit,
    )
    active_exposure_plan = _build_active_exposure_plan(
        cell, workload, request_rows, topology
    )
    compiler_projection["active_exposure_plan"] = copy.deepcopy(
        active_exposure_plan
    )
    anchor_active_at_ns: int | None = None
    anchor_owned_stage_keys: tuple[tuple[int, int, int], ...] = ()
    cell_dir = Path(cell_dir)
    cell_dir.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(os.fspath(cell_dir)):
        raise ArtifactTargetExistsError(f"cell path already exists: {cell_dir}")

    with tempfile.TemporaryDirectory(
        prefix=f".{cell.cell_id}.runtime-", dir=cell_dir.parent
    ) as temporary:
        runtime_path = Path(temporary) / "sim-config.json"
        runtime_path.write_bytes(canonical_json_bytes(runtime_config))
        server_args = ServerArgs(
            model_path=model_path,
            dp_size=int(runtime_config["scheduler"]["dp_size"]),
            tp_size=int(runtime_config["scheduler"].get("tp_size", 1)),
            pp_size=int(runtime_config["scheduler"].get("pp_size", 1)),
            streaming_arrival=True,
            sim_config_path=str(runtime_path),
        )
        cluster_kwargs = (
            compiled_deployment.offline_cluster_kwargs()
            if compiled_deployment is not None
            else {"physical_topology": topology}
        )
        with OfflineSchedulerCluster(server_args, **cluster_kwargs) as cluster:
            if compiled_deployment is not None:
                compiled_deployment.bind_started_cluster(cluster)
            else:
                if cluster.hicache_transfer_bridge is None:
                    raise SuiteReadinessError(
                        "ordinary_hicache_adapter_missing",
                        "formal global-DES runner requires ordinary HiCache adapter",
                    )
                cluster.hicache_transfer_bridge.fabric = _copy_fabric(cell)
                if cluster.aegis_transfer_bridge is not None:
                    cluster.aegis_transfer_bridge = OfflineAegisTransferBridge(
                        cluster.global_des,
                        topology,
                        fabric=_copy_fabric(cell),
                    )
            if active_exposure_plan is not None:
                for request in requests:
                    at_s = float(
                        request.sampling_params.custom_params["simulation"][
                            "server_created_time"
                        ]
                    )
                    cluster.schedule_request_arrival(request, at_s=at_s)
                (
                    anchor_active_at_ns,
                    activation_events_used,
                    anchor_owned_stage_keys,
                ) = _wait_for_active_anchor(
                    cluster,
                    active_exposure_plan,
                    max_events=options.max_events,
                )
                if compiled_deployment is not None:
                    assert compiler_bindings is not None
                    compiled_failure_timeline = compile_failure_timeline(
                        _compiler_layer("failure"),
                        cell.axes,
                        compiled_deployment,
                        compiler_bindings,
                        active_exposure_plan=active_exposure_plan,
                        anchor_active_at_ns=anchor_active_at_ns,
                    )
                    compiler_projection["failure_timeline"] = (
                        compiled_failure_timeline.as_dict()
                    )
                    failure_ids = _schedule_failures(
                        cluster,
                        cell,
                        compiled_timeline=compiled_failure_timeline,
                    )
                else:
                    failure_ids = _schedule_active_plan_without_compiler(
                        cluster,
                        cell,
                        active_exposure_plan,
                        anchor_active_at_ns=anchor_active_at_ns,
                    )
                result, request_rows, replay_source_by_id = (
                    _drive_with_hard_failure_replay(
                        cluster,
                        cell=cell,
                        requests=requests,
                        request_rows=request_rows,
                        failure_ids=failure_ids,
                        max_events=options.max_events,
                        requests_already_scheduled=True,
                        initial_events_used=activation_events_used,
                    )
                )
            else:
                failure_ids = _schedule_failures(
                    cluster,
                    cell,
                    compiled_timeline=compiled_failure_timeline,
                )
                if failure_ids:
                    result, request_rows, replay_source_by_id = (
                        _drive_with_hard_failure_replay(
                            cluster,
                            cell=cell,
                            requests=requests,
                            request_rows=request_rows,
                            failure_ids=failure_ids,
                            max_events=options.max_events,
                        )
                    )
                else:
                    for request in requests:
                        at_s = float(
                            request.sampling_params.custom_params["simulation"][
                                "server_created_time"
                            ]
                        )
                        cluster.schedule_request_arrival(request, at_s=at_s)
                    result = cluster.run_until_complete(
                        expected_request_ids={str(request.rid) for request in requests},
                        max_events=options.max_events,
                    )
                    replay_source_by_id = {}
            des = cluster.global_des
            bridge_export = (
                cluster.resource_bridge.export()
                if cluster.resource_bridge is not None
                else {}
            )

    if result.physical_hicache_transfers.get("schema") != (
        "sglsim.offline.hicache-physical-transfer-bridge.v1"
    ):
        raise SuiteReadinessError(
            "ordinary_hicache_adapter_missing",
            "completed cell has no canonical ordinary HiCache transfer export",
        )

    compiler_projection["active_exposure_runtime_binding"] = (
        None
        if active_exposure_plan is None
        else {
            "anchor_active_at_ns": anchor_active_at_ns,
            "anchor_owned_stage_keys": [
                list(key) for key in anchor_owned_stage_keys
            ],
            "planned_trigger_time_ns": (
                None
                if anchor_active_at_ns is None
                else int(anchor_active_at_ns)
                + int(active_exposure_plan["active_offset_ns"])
            ),
        }
    )
    active_failure_exposure = _validate_active_failure_exposure(
        plan=active_exposure_plan,
        anchor_active_at_ns=anchor_active_at_ns,
        failures=result.failure_events,
        request_rows=request_rows,
    )

    finalized_requests = _finalize_request_rows(
        request_rows,
        result,
        replay_source_by_id=replay_source_by_id,
    )
    terminal_request_closure = _validate_terminal_request_closure(
        finalized_requests
    )
    router_output_conservation = _validate_router_output_conservation(
        result.router_output_ledger
    )
    physical_recovery_admission_gate = (
        _validate_physical_recovery_admission_gate(
            result.router_output_ledger,
            result.physical_aegis_transfers,
        )
    )
    metrics = build_scheduler_cell_metrics(
        result,
        group=cell.group,
        request_rows=finalized_requests,
        smoke=options.smoke,
        calibration_lineage=options.calibration_lineage,
    )
    code_hash = implementation_revision
    claims = evidence_claims(
        cell.group,
        smoke=options.smoke,
        calibration_lineage=options.calibration_lineage,
    )
    atomic_workflow_smoke = bool(
        options.smoke
        and workload.metadata.get("atomic_workflow_replay") is True
        and workload.metadata.get("smoke_request_truncation_permitted") is False
    )
    if atomic_workflow_smoke:
        # This is still a smoke-selected cell from the 60-cell P11 matrix, but
        # it is not a request-prefix subset: the complete workflow is present.
        claims = {
            **claims,
            "smoke_request_subset": False,
            "smoke_atomic_workflow_subset": True,
            "paper_ready_reason": "modeled_smoke_complete_atomic_workflow",
        }
        metrics["claims"] = dict(claims)
    lineage = {
        "engine": CELL_ENGINE,
        "code_hash": code_hash,
        "suite_id": "aegiskv-paper-global-des-v3",
        "group": cell.group,
        "cell_id": cell.cell_id,
        "cell_ordinal": cell.ordinal,
        "axes": dict(cell.axes),
        "model": model_name,
        "model_path": model_path,
        "model_revision": model_revision,
        "request_universe_hash": workload.request_universe_hash,
        "compiled_requests_hash": workload.compiled_requests_hash,
        "failure_shadow_plan_sha256": (
            None
            if active_exposure_plan is None
            else active_exposure_plan["shadow_plan_sha256"]
        ),
        "runtime_config_sha256": _sha256_bytes(canonical_json_bytes(runtime_config)),
        "runtime_compiler_projection_sha256": _sha256_bytes(
            canonical_json_bytes(compiler_projection)
        ),
        "p0_contract_gate": dict(p0_lineage),
        "execution_scope": (
            "smoke_complete_atomic_workflow"
            if atomic_workflow_smoke
            else "smoke_request_prefix"
            if options.smoke
            else "complete_compiled_cell"
        ),
        "claims": claims,
    }
    write_global_des_artifacts(
        cell_dir,
        des=des,
        rank_steps=bridge_export.get("rank_steps", ()),
        requests=finalized_requests,
        failures=result.failure_events,
        resolved_config=cell.resolved_config,
        lineage_metadata=lineage,
        audit_details={
            "schema": "aegiskv.global-des-scheduler-cell-audit.v1",
            "status": "PASS",
            "cell_id": cell.cell_id,
            "axes": dict(cell.axes),
            "claims": claims,
            "workload": dict(workload.metadata),
            "runtime_config": runtime_config,
            "runtime_compiler": compiler_projection,
            "policy_independent_failure_plan": copy.deepcopy(
                active_exposure_plan
            ),
            "active_failure_exposure": active_failure_exposure,
            "metrics": metrics,
            "request_terminal_closure": terminal_request_closure,
            "router_output_conservation": router_output_conservation,
            "router_output_ledger": result.router_output_ledger,
            "physical_recovery_admission_gate": (
                physical_recovery_admission_gate
            ),
            # The Scheduler reports token-granular causal frontiers before its
            # finished-request cache path may release a final partial page.
            # Preserve every physical clamp in the immutable audit so a reader
            # can distinguish copied tokens from the explicit recompute tail.
            "physical_source_frontier_adjustments": list(
                result.physical_aegis_transfers.get(
                    "source_frontier_adjustments", ()
                )
            ),
            "real_scheduler_integration": True,
            "causal_rank_work_barrier": (
                "conservative_dispatch_lower_bound_until_worker_response_v1"
            ),
            "hard_failure_api": "OfflineSchedulerCluster.schedule_fail_stop",
            "replay_is_explicit_new_request": True,
            "killed_original_request_is_never_counted_complete": True,
            "committed_output_position_owned_by_parent_router": True,
            "commit_before_detokenizer_publish": True,
            "silent_replay_input_is_prompt_plus_committed_synthetic_outputs": True,
            "continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
            "real_logits_or_rng_continuation_claimed": False,
            "physical_recovery_job_activation_gate_implemented": True,
            "physical_recovery_job_activation_gate_in_scope": bool(
                physical_recovery_admission_gate["observed"]
            ),
            "scheduler_aggregate_accounting": True,
            "post_run_synthetic_extent_materialization": False,
        },
        extent_accounting_mode="scheduler_aggregate",
        supplemental_audit_errors=result.resource_audit_errors,
        recovery_transfers=result.physical_aegis_transfers.get(
            "recovery_jobs", ()
        ),
        ordinary_hicache_transfers=result.physical_hicache_transfers,
    )
    manifest = _strict_json(cell_dir / "manifest.json")
    return CellExecution(
        path=cell_dir,
        manifest=manifest,
        metrics=metrics,
        replay_ids=tuple(sorted(replay_source_by_id)),
    )


def validate_resumable_cell(
    cell_dir: Path,
    *,
    expected_config_sha256: str,
    expected_code_hash: str,
) -> dict[str, Any]:
    """Validate every manifest-listed byte before a resume skip."""

    root = Path(cell_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ResumeValidationError(
            f"existing cell is incomplete (manifest missing): {root}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeValidationError(f"cannot read cell manifest {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ResumeValidationError(f"cell manifest is not an object: {manifest_path}")
    if manifest.get("config_sha256") != expected_config_sha256:
        raise ResumeValidationError(f"resolved config drift for existing cell {root}")
    if manifest.get("code_hash") != expected_code_hash:
        raise ResumeValidationError(f"implementation drift for existing cell {root}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ResumeValidationError(f"manifest has no file inventory: {root}")
    for name, record in files.items():
        if not isinstance(record, Mapping):
            raise ResumeValidationError(f"bad file record {name!r} in {root}")
        path = root / str(name)
        if not path.is_file():
            raise ResumeValidationError(f"manifest-listed file is missing: {path}")
        actual = _sha256_file(path)
        if actual != record.get("sha256"):
            raise ResumeValidationError(f"manifest-listed file is corrupt: {path}")
    return manifest


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(value))
    fd, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_selected_cells(
    *,
    loaded: SuiteConfig,
    cells: Sequence[CellSpec],
    run_dir: Path,
    options: ExecutionOptions,
    p0_summary: Path,
    resume: bool,
    experiment_root: Path,
) -> Path:
    """Execute selected cells sequentially and maintain a resumable run index."""

    p0_lineage = validate_p0_summary(p0_summary)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    code_hash = _implementation_hash()
    records: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, 1):
        target = root / cell.group / cell.cell_id
        expected_config = config_hash(cell.resolved_config)
        if target.exists():
            if not resume:
                raise ArtifactTargetExistsError(
                    f"cell exists; pass --resume only after validation: {target}"
                )
            manifest = validate_resumable_cell(
                target,
                expected_config_sha256=expected_config,
                expected_code_hash=code_hash,
            )
            records.append(
                {
                    "cell_id": cell.cell_id,
                    "group": cell.group,
                    "ordinal": cell.ordinal,
                    "status": "validated_resume_skip",
                    "manifest": str(target / "manifest.json"),
                    "manifest_payload_sha256": manifest.get(
                        "canonical_payload_sha256"
                    ),
                }
            )
        else:
            workload = prepare_workload(cell, experiment_root=experiment_root)
            execution = execute_prepared_cell(
                cell,
                workload,
                target,
                options=options,
                p0_lineage=p0_lineage,
            )
            records.append(
                {
                    "cell_id": cell.cell_id,
                    "group": cell.group,
                    "ordinal": cell.ordinal,
                    "status": "executed",
                    "manifest": str(execution.path / "manifest.json"),
                    "manifest_payload_sha256": execution.manifest.get(
                        "canonical_payload_sha256"
                    ),
                    "replay_ids": list(execution.replay_ids),
                }
            )
        run_manifest = {
            "schema": RUN_SCHEMA,
            "suite_id": loaded.suite["suite_id"],
            "configured_suite_status": loaded.suite["status"],
            "execution_engine": CELL_ENGINE,
            "code_hash": code_hash,
            "selected_cells": len(cells),
            "processed_cells": len(records),
            "complete": len(records) == len(cells),
            "smoke": options.smoke,
            "p0_contract_gate": p0_lineage,
            "claims": {
                "modeled_simulation": True,
                "real_hardware_measurement": False,
                "paper_ready": False,
                "full_2146_cell_suite_executed": False,
                "P0_executed_by_this_runner": False,
                "P1_P12_selected_cells_executed": len(records),
            },
            "cells": records,
        }
        _atomic_json(root / "suite_manifest.json", run_manifest)
        print(
            f"[{index}/{len(cells)}] {cell.cell_id}: {records[-1]['status']}",
            flush=True,
        )
    return root


def _parse_assignments(values: Sequence[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        try:
            name, value = str(raw).split("=", 1)
        except ValueError as exc:
            raise ValueError(f"{option} requires MODEL=VALUE, got {raw!r}") from exc
        if not name or not value or name in result:
            raise ValueError(f"invalid/duplicate {option} assignment {raw!r}")
        result[name] = value
    return result


def _calibration_lineage(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise SuiteReadinessError(
            "calibration_manifest_missing", f"calibration manifest absent: {resolved}"
        )
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _default_run_dir(results_root: Path, *, smoke: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "global_des_scheduler_smoke" if smoke else "global_des_scheduler_suite"
    candidate = Path(results_root) / f"{stamp}_{suffix}"
    index = 0
    while candidate.exists():
        index += 1
        candidate = Path(results_root) / f"{stamp}_{suffix}_{index}"
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected AegisKV P1-P12 cells through the real Scheduler global DES"
    )
    parser.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--p0-summary", type=Path, default=DEFAULT_P0_SUMMARY)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--cells", nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true", dest="all_cells")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate selection/P0/workload readiness without launching workers")
    parser.add_argument("--max-requests-per-cell", type=int)
    parser.add_argument("--max-events", type=int, default=1_000_000)
    parser.add_argument("--model-path", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--model-revision", action="append", default=[], metavar="MODEL=REVISION")
    parser.add_argument("--calibration-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_requests_per_cell is not None and not args.smoke:
        raise SystemExit("--max-requests-per-cell is permitted only with --smoke")
    if args.smoke and args.max_requests_per_cell is None:
        args.max_requests_per_cell = 2
    loaded = load_suite(args.config_root)
    cells = select_cells(
        loaded,
        groups=args.groups,
        selectors=args.cells,
        smoke=args.smoke,
        all_cells=args.all_cells,
    )
    p0 = validate_p0_summary(args.p0_summary)
    model_paths = _parse_assignments(args.model_path, option="--model-path")
    model_revisions = _parse_assignments(
        args.model_revision, option="--model-revision"
    )
    if args.check:
        readiness: list[dict[str, Any]] = []
        for cell in cells:
            try:
                workload = prepare_workload(
                    cell, experiment_root=Path(__file__).resolve().parents[1]
                )
                validate_prepared_workload_readiness(cell, workload)
                model_name, model_path, revision = _model_runtime_lineage(
                    cell,
                    ExecutionOptions(
                        model_paths=model_paths,
                        model_revisions=model_revisions,
                    ),
                )
                status = {
                    "cell_id": cell.cell_id,
                    "ready": True,
                    "requests": len(workload.records),
                    "model": model_name,
                    "model_path": model_path,
                    "model_revision": revision,
                }
            except SuiteReadinessError as exc:
                status = {
                    "cell_id": cell.cell_id,
                    "ready": False,
                    "code": exc.code,
                    "error": str(exc),
                    "details": exc.details,
                }
            readiness.append(status)
        print(
            json.dumps(
                {"p0": p0, "cells": readiness},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if all(row["ready"] for row in readiness) else 2
    options = ExecutionOptions(
        model_paths=model_paths,
        model_revisions=model_revisions,
        smoke=bool(args.smoke),
        max_requests_per_cell=args.max_requests_per_cell,
        max_events=max(int(args.max_events), 1),
        calibration_lineage=_calibration_lineage(args.calibration_manifest),
    )
    run_dir = args.run_dir or _default_run_dir(args.results_root, smoke=args.smoke)
    run_selected_cells(
        loaded=loaded,
        cells=cells,
        run_dir=run_dir,
        options=options,
        p0_summary=args.p0_summary,
        resume=bool(args.resume),
        experiment_root=Path(__file__).resolve().parents[1],
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_ENGINE",
    "DEFAULT_P0_SUMMARY",
    "RUN_SCHEMA",
    "CellExecution",
    "CellSpec",
    "ExecutionOptions",
    "PreparedWorkload",
    "ResumeValidationError",
    "SuiteExecutionError",
    "SuiteReadinessError",
    "enumerate_cells",
    "execute_prepared_cell",
    "main",
    "prepare_workload",
    "run_selected_cells",
    "select_cells",
    "validate_p0_summary",
    "validate_prepared_workload_readiness",
    "validate_resumable_cell",
]
