"""Compile the pinned Weka/AgentX corpus into the P11 observed-DAG artifact.

The compiler has three deliberately separate responsibilities:

* parse the complete pinned corpus and freeze an equal-cardinality DAG-size
  stratification over workflows with an observed parallel stage;
* train/evaluate the workflow-disjoint causal remaining-time predictor on the
  complete eligible universe; and
* publish only admission-time metadata to request rows.  Labels such as actual
  remaining time or actual Slack never enter the simulation payload.

The produced artifact is modeled, not hardware-calibrated, and is never marked
paper-ready by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sglsim.srt.simulation.aegis_gbdt import AegisGBDTConfig
from sglsim.srt.simulation.aegis_residual_dag import (
    RESIDUAL_DAG_SLACK_SEMANTICS,
)

from .global_des_config import EXPERIMENT_ROOT, canonical_json_bytes, load_suite
from .request.weka_causal import (
    CausalSlackTicket,
    WekaRequest,
    WekaWorkflow,
    build_causal_samples,
    crossfit_causal_slack,
    dataset_manifest,
    default_full_trace_path,
    load_weka_workflows,
    sha256_file,
    zero_slack_tickets_for_unmodeled_requests,
)


ARTIFACT_SCHEMA = "aegiskv.weka-agentx-causal-workload.v1"
REQUEST_SCHEMA = "aegiskv.weka-agentx-causal-request.v1"
TOKEN_IDENTITY_SCHEMA = "aegiskv.weka-hash-block-token-identity.v1"
SOURCE_SHA256 = (
    "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"
)
DEFAULT_OUTPUT_ROOT = (
    EXPERIMENT_ROOT / "data" / "observed" / "weka_agentx_causal_v1"
)


class WekaObservedBuildError(RuntimeError):
    """Fail-closed P11 compilation error."""


@dataclass(frozen=True)
class StratumRule:
    name: str
    minimum_requests_inclusive: int
    maximum_requests_inclusive: int
    expected_workflows: int

    def contains(self, request_count: int) -> bool:
        return (
            int(self.minimum_requests_inclusive)
            <= int(request_count)
            <= int(self.maximum_requests_inclusive)
        )


@dataclass(frozen=True)
class BuildSpec:
    dataset_id: str
    source_sha256: str
    source_size_bytes: int
    source_workflow_count: int
    source_request_count: int
    eligible_parallel_workflows: int
    seeds: tuple[int, ...]
    partition_salt: str
    workflows_per_seed_per_stratum: int
    strata: tuple[StratumRule, ...]
    context_tokens: int
    output_cap_tokens: int
    folds: int
    split_salt: str
    interval_coverage: float
    blend_tau_rounds: float
    gbdt: AegisGBDTConfig

    @classmethod
    def from_suite(cls) -> "BuildSpec":
        layer = load_suite().group_map["P11"].request
        workload = layer.fixed["workload"]
        source = workload["source_artifact"]
        stratification = workload["dag_stratification"]
        predictor = workload["causal_slack_predictor"]
        return cls(
            dataset_id=str(workload["dataset_id"]),
            source_sha256=str(source["sha256"]),
            source_size_bytes=int(source["size_bytes"]),
            source_workflow_count=int(source["workflow_count"]),
            source_request_count=int(source["request_count"]),
            eligible_parallel_workflows=int(
                stratification["eligible_parallel_workflows"]
            ),
            seeds=tuple(int(value) for value in layer.axes["seed"]),
            partition_salt=str(stratification["partition_salt"]),
            workflows_per_seed_per_stratum=int(
                stratification["workflows_per_seed_per_stratum"]
            ),
            strata=tuple(
                StratumRule(
                    name=str(row["name"]),
                    minimum_requests_inclusive=int(
                        row["minimum_requests_inclusive"]
                    ),
                    maximum_requests_inclusive=int(
                        row["maximum_requests_inclusive"]
                    ),
                    expected_workflows=int(row["expected_workflows"]),
                )
                for row in stratification["strata"]
            ),
            context_tokens=int(workload["context_projection"]["context_tokens"]),
            output_cap_tokens=int(
                workload["context_projection"]["output_cap_tokens"]
            ),
            folds=int(predictor["folds"]),
            split_salt=str(predictor["split_salt"]),
            interval_coverage=float(predictor["interval_coverage"]),
            blend_tau_rounds=float(predictor["blend_tau_rounds"]),
            gbdt=AegisGBDTConfig(**dict(predictor["gbdt"])),
        )


def _canonical_json(value: object) -> bytes:
    return canonical_json_bytes(value)


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _stable_rank(salt: str, *parts: object) -> tuple[int, tuple[str, ...]]:
    text = ":".join((str(salt), *(str(part) for part in parts)))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big"), tuple(str(part) for part in parts)


def _has_observed_parallel_stage(workflow: WekaWorkflow) -> bool:
    return any(len(stage.branches) > 1 for stage in workflow.stages)


def stratify_workflows(
    workflows: Sequence[WekaWorkflow], spec: BuildSpec
) -> dict[str, tuple[WekaWorkflow, ...]]:
    """Apply the frozen, exhaustive 42/42/42 request-count strata."""

    eligible = [row for row in workflows if _has_observed_parallel_stage(row)]
    if len(eligible) != spec.eligible_parallel_workflows:
        raise WekaObservedBuildError(
            "eligible parallel-workflow count differs from frozen config: "
            f"{len(eligible)} != {spec.eligible_parallel_workflows}"
        )
    output: dict[str, tuple[WekaWorkflow, ...]] = {}
    assigned: set[str] = set()
    for rule in spec.strata:
        rows = tuple(
            sorted(
                (
                    workflow
                    for workflow in eligible
                    if rule.contains(len(workflow.requests))
                ),
                key=lambda workflow: (len(workflow.requests), workflow.workflow_id),
            )
        )
        if len(rows) != rule.expected_workflows:
            raise WekaObservedBuildError(
                f"stratum {rule.name!r} has {len(rows)} workflows, expected "
                f"{rule.expected_workflows}"
            )
        overlap = assigned & {row.workflow_id for row in rows}
        if overlap:
            raise WekaObservedBuildError(
                f"workflow assigned to multiple strata: {sorted(overlap)}"
            )
        assigned.update(row.workflow_id for row in rows)
        output[rule.name] = rows
    eligible_ids = {row.workflow_id for row in eligible}
    if assigned != eligible_ids:
        missing = sorted(eligible_ids - assigned)
        raise WekaObservedBuildError(
            f"frozen strata do not exhaust the eligible universe: {missing[:5]}"
        )
    return output


def assign_seed_partitions(
    strata: Mapping[str, Sequence[WekaWorkflow]], spec: BuildSpec
) -> dict[tuple[int, str], tuple[WekaWorkflow, ...]]:
    """Choose deterministic, seed-disjoint workflows within every stratum."""

    required = len(spec.seeds) * spec.workflows_per_seed_per_stratum
    output: dict[tuple[int, str], tuple[WekaWorkflow, ...]] = {}
    globally_selected: set[str] = set()
    for rule in spec.strata:
        candidates = tuple(strata[rule.name])
        if len(candidates) < required:
            raise WekaObservedBuildError(
                f"stratum {rule.name!r} has {len(candidates)} workflows but "
                f"seed-disjoint sampling requires {required}"
            )
        ranked = sorted(
            candidates,
            key=lambda workflow: _stable_rank(
                spec.partition_salt, rule.name, workflow.workflow_id
            ),
        )
        for partition, seed in enumerate(spec.seeds):
            start = partition * spec.workflows_per_seed_per_stratum
            selected = tuple(
                ranked[start : start + spec.workflows_per_seed_per_stratum]
            )
            ids = {row.workflow_id for row in selected}
            if globally_selected & ids:
                raise WekaObservedBuildError(
                    "seed partition reused a workflow: "
                    f"{sorted(globally_selected & ids)}"
                )
            globally_selected.update(ids)
            output[(int(seed), rule.name)] = selected
    return output


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        count += 1
    return count


def _project_request(
    request: WekaRequest,
    *,
    block_size: int,
    context_tokens: int,
    output_cap_tokens: int,
    prior_hash_ids: Sequence[int],
) -> tuple[int, int, int, tuple[int, ...]]:
    """Head-clip while preserving the observed hash-block prefix identity."""

    output = min(
        max(int(request.output_tokens), 0),
        int(output_cap_tokens),
        max(int(context_tokens) - 1, 0),
    )
    prompt_budget = max(int(context_tokens) - output, 1)
    prompt = min(max(int(request.prompt_tokens), 1), prompt_budget)
    raw_common_blocks = _longest_common_prefix(prior_hash_ids, request.hash_ids)
    prefix = min(raw_common_blocks * int(block_size), prompt)
    required_blocks = math.ceil(prompt / max(int(block_size), 1))
    block_ids = tuple(int(value) for value in request.hash_ids[:required_blocks])
    return prompt, prefix, output, block_ids


def _ticket_provenance_payload(
    *,
    ticket: CausalSlackTicket,
    normalized_published_at_ms: float,
    modeled: bool,
) -> dict[str, Any]:
    """Return only causally publishable values; no actual label is included."""

    return {
        "schema": "aegiskv.weka-visible-residual-ticket-input.v1",
        "request_id": ticket.request_id,
        "workflow_id": ticket.workflow_id,
        "stage_id": int(ticket.stage_id),
        "agent_id": ticket.agent_id,
        "published_at_ms": round(float(normalized_published_at_ms), 6),
        "modeled": bool(modeled),
        "fold": int(ticket.fold),
        "predictor_sha256": str(ticket.predictor_sha256),
        "slack_lcb_ms": round(float(ticket.slack_lcb_ms), 6),
        "remaining_q50_ms": round(float(ticket.remaining_q50_ms), 6),
        "remaining_interval_lower_ms": round(
            float(ticket.remaining_interval_lower_ms), 6
        ),
        "remaining_interval_upper_ms": round(
            float(ticket.remaining_interval_upper_ms), 6
        ),
        "future_information": False,
    }


def visible_residual_ticket_fields(
    *,
    ticket: CausalSlackTicket,
    request_arrival_ms: float,
    workflow_start_ms: float,
) -> dict[str, Any]:
    """Convert a causal cross-fit ticket to the current-design ticket fields.

    For a modeled request, the current branch's conformal upper bound is the
    conservative path weight.  ``slack_lcb`` is its CPM total float against the
    dominant peer lower envelope, so ``ES=0``, ``LS=slack_lcb`` and
    ``critical_path=upper+slack_lcb`` are algebraically consistent.  The
    actual remaining time and actual Slack are intentionally absent.

    Missing/non-causal predictions retain a structurally complete metadata
    ticket but set ``usable=false`` and zero Slack.  That is Algorithm-1
    fail-close, not a request drop and not a no-protection policy substitution.
    """

    published_causally = float(ticket.published_at_ms) <= float(request_arrival_ms) + 1e-9
    modeled = int(ticket.fold) >= 0 and published_causally
    published = max(float(ticket.published_at_ms) - float(workflow_start_ms), 0.0)
    provenance_payload = _ticket_provenance_payload(
        ticket=ticket,
        normalized_published_at_ms=published,
        modeled=modeled,
    )
    provenance = _hash_json(provenance_payload)
    if modeled:
        slack = round(max(float(ticket.slack_lcb_ms), 0.0), 6)
        upper = round(max(float(ticket.remaining_interval_upper_ms), 0.0), 6)
        return {
            "workflow_slack_ms": slack,
            "aegis_slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
            "aegis_slack_uses_future_information": False,
            "aegis_slack_residual_dag_visible": True,
            "aegis_slack_metadata_only": True,
            "aegis_slack_usable": True,
            "aegis_slack_kind": "CPM_CONSERVATIVE_INTERVAL_TOTAL_FLOAT",
            "aegis_slack_reason": "visible_residual_dag_cpm",
            "aegis_slack_graph_version": (
                f"weka-causal-fold-{ticket.fold}-{ticket.predictor_sha256[:16]}"
            ),
            "aegis_slack_observed_at_ms": round(published, 6),
            "aegis_slack_node_id": ticket.request_id,
            "aegis_slack_agent_id": ticket.agent_id,
            "aegis_slack_es_ms": 0.0,
            "aegis_slack_ls_ms": slack,
            "aegis_slack_critical_path_ms": round(upper + slack, 6),
            "aegis_slack_input_provenance_sha256": provenance,
            "weka_ticket_modeled": True,
            "weka_crossfit_fold": int(ticket.fold),
            "weka_predictor_sha256": str(ticket.predictor_sha256),
            "weka_ticket_publish_causal": True,
            "weka_ticket_derivation": (
                "workflow_disjoint_crossfit_conformal_lcb_to_visible_cpm_v1"
            ),
        }
    return {
        "workflow_slack_ms": 0.0,
        "aegis_slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
        "aegis_slack_uses_future_information": False,
        "aegis_slack_residual_dag_visible": True,
        "aegis_slack_metadata_only": True,
        "aegis_slack_usable": False,
        "aegis_slack_kind": "UNKNOWN",
        "aegis_slack_reason": "weka_agentx_fail_close_zero",
        "aegis_slack_graph_version": "weka-causal-fail-close-zero",
        "aegis_slack_observed_at_ms": round(published, 6),
        "aegis_slack_node_id": ticket.request_id,
        "aegis_slack_agent_id": ticket.agent_id,
        "aegis_slack_es_ms": 0.0,
        "aegis_slack_ls_ms": 0.0,
        "aegis_slack_critical_path_ms": 0.0,
        "aegis_slack_input_provenance_sha256": provenance,
        "weka_ticket_modeled": False,
        "weka_crossfit_fold": -1,
        "weka_predictor_sha256": "fail-close-zero",
        "weka_ticket_publish_causal": bool(published_causally),
        "weka_ticket_derivation": "explicit_unmodeled_fail_close_zero_v1",
    }


def _branch_sequences(workflow: WekaWorkflow) -> dict[str, int]:
    output: dict[str, int] = {}
    for stage in workflow.stages:
        for branch in stage.branches:
            for sequence, request in enumerate(
                sorted(branch.requests, key=lambda row: (row.arrival_ms, row.rid)),
                start=1,
            ):
                output[request.rid] = sequence
    return output


def _request_rows(
    assignments: Mapping[tuple[int, str], Sequence[WekaWorkflow]],
    *,
    tickets: Mapping[str, CausalSlackTicket],
    spec: BuildSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_rids: set[str] = set()
    for (seed, stratum), workflows in sorted(assignments.items()):
        for workflow in sorted(workflows, key=lambda row: row.workflow_id):
            branch_sequence = _branch_sequences(workflow)
            prior_hashes: dict[str, tuple[int, ...]] = {}
            ordered = sorted(
                workflow.requests,
                key=lambda row: (row.arrival_ms, row.stage_id, row.agent_id, row.rid),
            )
            for round_index, request in enumerate(ordered):
                if request.rid in seen_rids:
                    raise WekaObservedBuildError(
                        f"request ID is not globally unique: {request.rid}"
                    )
                seen_rids.add(request.rid)
                prior = prior_hashes.get(request.agent_id, ())
                prompt, prefix, output, block_ids = _project_request(
                    request,
                    block_size=workflow.block_size,
                    context_tokens=spec.context_tokens,
                    output_cap_tokens=spec.output_cap_tokens,
                    prior_hash_ids=prior,
                )
                prior_hashes[request.agent_id] = request.hash_ids
                ticket = tickets[request.rid]
                simulation = visible_residual_ticket_fields(
                    ticket=ticket,
                    request_arrival_ms=request.arrival_ms,
                    workflow_start_ms=workflow.start_ms,
                )
                token_identity = {
                    "schema": TOKEN_IDENTITY_SCHEMA,
                    "block_size": int(workflow.block_size),
                    "block_hash_ids": list(block_ids),
                    # Weka hash_ids are workflow-local cache-block identities
                    # (many workflows restart them at zero), not global content
                    # hashes.  Namespace them to forbid fictitious cross-
                    # workflow prefix hits while preserving observed sharing
                    # inside one workflow.
                    "block_namespace_sha256": hashlib.sha256(
                        f"weka-workflow-blocks:{workflow.workflow_id}".encode()
                    ).hexdigest(),
                    "token_count": int(prompt),
                    "fallback_stream_sha256": hashlib.sha256(
                        (
                            f"weka-fallback:{workflow.workflow_id}:"
                            f"{request.agent_id}:{request.rid}"
                        ).encode()
                    ).hexdigest(),
                }
                rows.append(
                    {
                        "schema": REQUEST_SCHEMA,
                        "rid": request.rid,
                        "workflow": workflow.workflow_id,
                        "session": workflow.workflow_id,
                        "provider": "weka_agentx",
                        "agent_id": request.agent_id,
                        "agent_role": request.role,
                        "stage_id": int(request.stage_id),
                        "worker_seq": int(branch_sequence[request.rid]),
                        "round_index": int(round_index),
                        "arrival_ns": int(
                            round(
                                max(request.arrival_ms - workflow.start_ms, 0.0)
                                * 1_000_000.0
                            )
                        ),
                        "input_tokens": int(prompt),
                        "output_tokens": int(output),
                        "cached_prefix_tokens": int(prefix),
                        "prefix_key": (
                            f"weka:{workflow.workflow_id}:{request.agent_id}"
                        ),
                        "input_token_identity": token_identity,
                        "source_round_pk": int(round_index),
                        "source_round_id": request.rid,
                        "seed": int(seed),
                        "dag_stratum": str(stratum),
                        "simulation": simulation,
                    }
                )
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["dag_stratum"]),
            int(row["arrival_ns"]),
            str(row["rid"]),
        )
    )
    return rows


def _universe_payload(
    assignments: Mapping[tuple[int, str], Sequence[WekaWorkflow]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "aegiskv.weka-agentx-request-universe.v1",
        "partitions": [
            {
                "seed": int(seed),
                "dag_stratum": str(stratum),
                "workflow_ids": sorted(row.workflow_id for row in workflows),
            }
            for (seed, stratum), workflows in sorted(assignments.items())
        ],
        "request_identities": [
            {
                "seed": int(row["seed"]),
                "dag_stratum": str(row["dag_stratum"]),
                "workflow": str(row["workflow"]),
                "rid": str(row["rid"]),
            }
            for row in rows
        ],
    }


def compile_artifact_payload(
    workflows: Sequence[WekaWorkflow],
    *,
    spec: BuildSpec,
    source_path: Path,
    source_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compile in memory; publication is handled separately and atomically."""

    if len(workflows) != spec.source_workflow_count:
        raise WekaObservedBuildError(
            f"source workflow count mismatch: {len(workflows)} != "
            f"{spec.source_workflow_count}"
        )
    source_requests = sum(len(row.requests) for row in workflows)
    if source_requests != spec.source_request_count:
        raise WekaObservedBuildError(
            f"source request count mismatch: {source_requests} != "
            f"{spec.source_request_count}"
        )
    strata = stratify_workflows(workflows, spec)
    assignments = assign_seed_partitions(strata, spec)
    eligible = tuple(
        workflow for rule in spec.strata for workflow in strata[rule.name]
    )
    samples = build_causal_samples(eligible)
    raw_tickets, predictor_metrics = crossfit_causal_slack(
        samples,
        folds=spec.folds,
        split_salt=spec.split_salt,
        interval_coverage=spec.interval_coverage,
        blend_tau_rounds=spec.blend_tau_rounds,
        gbdt_config=spec.gbdt,
    )
    tickets = zero_slack_tickets_for_unmodeled_requests(eligible, raw_tickets)
    rows = _request_rows(assignments, tickets=tickets, spec=spec)
    universe_payload = _universe_payload(assignments, rows)
    selected_workflow_ids = {
        workflow.workflow_id
        for workflows_for_partition in assignments.values()
        for workflow in workflows_for_partition
    }
    selected_ticket_counts = Counter(
        "modeled" if row["simulation"]["weka_ticket_modeled"] else "unmodeled"
        for row in rows
    )
    selected_nonzero = sum(
        float(row["simulation"]["workflow_slack_ms"]) > 0.0 for row in rows
    )
    predictor_lineage = {
        **predictor_metrics,
        "feature_values_visible_at_request_admission_only": True,
        "actual_remaining_or_slack_in_request_payload": False,
        "hardware_calibrated": False,
    }
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_id": "weka_agentx_causal_v1",
        "requests_file": "requests.jsonl",
        "dataset_id": spec.dataset_id,
        "source": {
            **dataset_manifest(
                workflows,
                source_path=source_path,
                source_sha256=source_sha256,
            ),
            "path_relative_to_repository": str(
                source_path.resolve().relative_to(EXPERIMENT_ROOT.parents[1])
            ),
        },
        "eligibility": {
            "rule": "contains_observed_parallel_stage",
            "eligible_workflow_count": len(eligible),
            "eligible_request_count": sum(len(row.requests) for row in eligible),
        },
        "dag_stratification": {
            "unit": "requests_per_workflow",
            "method": "equal_cardinality_rank_tertiles_frozen_boundaries_v1",
            "strata": [asdict(rule) for rule in spec.strata],
            "partition_salt": spec.partition_salt,
            "workflows_per_seed_per_stratum": (
                spec.workflows_per_seed_per_stratum
            ),
            "seed_count": len(spec.seeds),
            "selection_without_replacement": True,
            "selected_workflow_count": len(selected_workflow_ids),
            "unselected_eligible_workflow_count": len(eligible)
            - len(selected_workflow_ids),
        },
        "partitions": [
            {
                "seed": int(seed),
                "dag_stratum": str(stratum),
                "workflow_ids": sorted(row.workflow_id for row in workflows_for_seed),
                "request_count": sum(
                    row["seed"] == seed and row["dag_stratum"] == stratum
                    for row in rows
                ),
            }
            for (seed, stratum), workflows_for_seed in sorted(assignments.items())
        ],
        "context_projection": {
            "method": "head_preserving_clip_with_observed_hash_blocks_v1",
            "context_tokens": spec.context_tokens,
            "output_cap_tokens": spec.output_cap_tokens,
            "prefix_semantics": (
                "longest_observed_hash_block_prefix_with_prior_same_agent_request"
            ),
            "token_identity_schema": TOKEN_IDENTITY_SCHEMA,
        },
        "request_count": len(rows),
        "selected_ticket_coverage": {
            "modeled_requests": int(selected_ticket_counts["modeled"]),
            "unmodeled_fail_close_zero_requests": int(
                selected_ticket_counts["unmodeled"]
            ),
            "nonzero_slack_requests": int(selected_nonzero),
            "modeled_fraction": (
                selected_ticket_counts["modeled"] / len(rows) if rows else 0.0
            ),
        },
        "request_universe_hash": _hash_json(universe_payload),
        "predictor_lineage": predictor_lineage,
        "predictor_lineage_sha256": _hash_json(predictor_lineage),
        "evidence": {
            "class": "modeled_observed_workload_not_hardware_calibrated",
            "future_information_used_by_policy": False,
            "unmodeled_behavior": "explicit_fail_close_zero",
            "hardware_calibrated": False,
            "paper_ready": False,
        },
    }
    return rows, manifest


def _write_requests(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for row in rows:
            payload = _canonical_json(row)
            stream.write(payload)
            digest.update(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def publish_artifact(
    output_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one immutable directory via a same-parent atomic rename."""

    output_root = Path(output_root).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        existing = audit_artifact(output_root)
        expected_universe = str(manifest["request_universe_hash"])
        if existing["request_universe_hash"] == expected_universe:
            return existing
        raise WekaObservedBuildError(
            f"refusing to overwrite different observed artifact: {output_root}"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent)
        )
    )
    try:
        requests_sha256 = _write_requests(temporary / "requests.jsonl", rows)
        final_manifest = {
            **dict(manifest),
            "requests_sha256": requests_sha256,
        }
        _write_json(temporary / "manifest.json", final_manifest)
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return audit_artifact(output_root)


def audit_artifact(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    requests_path = root / "requests.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WekaObservedBuildError(f"invalid manifest: {manifest_path}: {exc}") from exc
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise WekaObservedBuildError("unexpected observed-workload schema")
    expected_sha = str(manifest.get("requests_sha256") or "")
    actual_sha = sha256_file(requests_path)
    if not expected_sha or actual_sha != expected_sha:
        raise WekaObservedBuildError(
            f"requests hash mismatch: {actual_sha} != {expected_sha}"
        )
    counts: Counter[tuple[int, str]] = Counter()
    workflows_by_partition: dict[tuple[int, str], set[str]] = {}
    modeled = 0
    unmodeled = 0
    nonzero = 0
    request_count = 0
    seen_rids: set[str] = set()
    with requests_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WekaObservedBuildError(
                    f"invalid request JSON at line {line_number}"
                ) from exc
            if row.get("schema") != REQUEST_SCHEMA:
                raise WekaObservedBuildError(
                    f"unexpected request schema at line {line_number}"
                )
            rid = str(row.get("rid") or "")
            if not rid or rid in seen_rids:
                raise WekaObservedBuildError(f"missing/duplicate rid {rid!r}")
            seen_rids.add(rid)
            key = (int(row["seed"]), str(row["dag_stratum"]))
            counts[key] += 1
            workflows_by_partition.setdefault(key, set()).add(str(row["workflow"]))
            simulation = row.get("simulation")
            if not isinstance(simulation, Mapping):
                raise WekaObservedBuildError(f"simulation ticket missing for {rid}")
            if simulation.get("aegis_slack_uses_future_information") is not False:
                raise WekaObservedBuildError(f"future-information ticket for {rid}")
            if simulation.get("aegis_slack_semantics") != RESIDUAL_DAG_SLACK_SEMANTICS:
                raise WekaObservedBuildError(f"ticket semantics mismatch for {rid}")
            if simulation.get("weka_ticket_modeled") is True:
                modeled += 1
                if simulation.get("aegis_slack_usable") is not True:
                    raise WekaObservedBuildError(f"modeled ticket is unusable for {rid}")
            else:
                unmodeled += 1
                if (
                    simulation.get("aegis_slack_usable") is not False
                    or float(simulation.get("workflow_slack_ms", -1.0)) != 0.0
                    or simulation.get("aegis_slack_reason")
                    != "weka_agentx_fail_close_zero"
                ):
                    raise WekaObservedBuildError(
                        f"unmodeled ticket does not fail close for {rid}"
                    )
            nonzero += float(simulation.get("workflow_slack_ms", 0.0)) > 0.0
            identity = row.get("input_token_identity")
            if (
                not isinstance(identity, Mapping)
                or identity.get("schema") != TOKEN_IDENTITY_SCHEMA
                or int(identity.get("token_count", -1)) != int(row["input_tokens"])
                or len(str(identity.get("block_namespace_sha256") or "")) != 64
            ):
                raise WekaObservedBuildError(f"token identity mismatch for {rid}")
            request_count += 1
    if request_count != int(manifest.get("request_count", -1)):
        raise WekaObservedBuildError("manifest request count mismatch")
    manifest_partitions = {
        (int(row["seed"]), str(row["dag_stratum"])): row
        for row in manifest.get("partitions") or ()
    }
    if set(counts) != set(manifest_partitions):
        raise WekaObservedBuildError("request partitions differ from manifest")
    selected_workflows: set[str] = set()
    for key, row in manifest_partitions.items():
        if counts[key] != int(row["request_count"]):
            raise WekaObservedBuildError(f"partition request count mismatch: {key}")
        expected_workflows = set(map(str, row["workflow_ids"]))
        if workflows_by_partition[key] != expected_workflows:
            raise WekaObservedBuildError(f"partition workflows mismatch: {key}")
        if selected_workflows & expected_workflows:
            raise WekaObservedBuildError("workflow reused across seed partitions")
        selected_workflows.update(expected_workflows)
    coverage = manifest["selected_ticket_coverage"]
    if (
        modeled != int(coverage["modeled_requests"])
        or unmodeled != int(coverage["unmodeled_fail_close_zero_requests"])
        or nonzero != int(coverage["nonzero_slack_requests"])
    ):
        raise WekaObservedBuildError("ticket coverage differs from manifest")
    return {
        "path": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "requests_sha256": actual_sha,
        "request_universe_hash": str(manifest["request_universe_hash"]),
        "request_count": request_count,
        "workflow_count": len(selected_workflows),
        "partition_count": len(counts),
        "modeled_requests": modeled,
        "unmodeled_fail_close_zero_requests": unmodeled,
        "nonzero_slack_requests": nonzero,
    }


def build_observed_artifact(
    *,
    source_path: Path = default_full_trace_path(),
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    spec: BuildSpec | None = None,
) -> dict[str, Any]:
    spec = spec or BuildSpec.from_suite()
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise WekaObservedBuildError(f"pinned Weka source is absent: {source_path}")
    actual_size = source_path.stat().st_size
    if actual_size != spec.source_size_bytes:
        raise WekaObservedBuildError(
            f"source size mismatch: {actual_size} != {spec.source_size_bytes}"
        )
    actual_sha = sha256_file(source_path)
    if actual_sha != spec.source_sha256 or actual_sha != SOURCE_SHA256:
        raise WekaObservedBuildError(f"source SHA-256 mismatch: {actual_sha}")
    workflows = tuple(load_weka_workflows(source_path))
    rows, manifest = compile_artifact_payload(
        workflows,
        spec=spec,
        source_path=source_path,
        source_sha256=actual_sha,
    )
    return publish_artifact(output_root, rows=rows, manifest=manifest)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_full_trace_path())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="verify an existing immutable artifact without reading the source corpus",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = (
        audit_artifact(args.output)
        if args.audit_only
        else build_observed_artifact(source_path=args.source, output_root=args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_SCHEMA",
    "BuildSpec",
    "DEFAULT_OUTPUT_ROOT",
    "REQUEST_SCHEMA",
    "SOURCE_SHA256",
    "StratumRule",
    "TOKEN_IDENTITY_SCHEMA",
    "WekaObservedBuildError",
    "assign_seed_partitions",
    "audit_artifact",
    "build_observed_artifact",
    "compile_artifact_payload",
    "publish_artifact",
    "stratify_workflows",
    "visible_residual_ticket_fields",
]
