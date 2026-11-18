"""TraceLab and controlled fork/join request construction."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from sglsim.srt.simulation.aegis_gbdt import AegisGBDTConfig
from sglsim.srt.simulation.aegis_complete import CompleteRequest
from sglsim.srt.simulation.aegis_residual_dag import RESIDUAL_DAG_SLACK_SEMANTICS

from ..config import sha256_file
from .weka_causal import (
    CausalSlackTicket,
    WekaWorkflow,
    build_causal_samples,
    crossfit_causal_slack,
    dataset_manifest,
    load_weka_workflows,
    zero_slack_tickets_for_unmodeled_requests,
)


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _slack_evidence(request_id: str, workflow_id: str, slack_ms: float, seed: int) -> dict[str, Any]:
    identity = f"{workflow_id}:{request_id}:{seed}:{slack_ms:.6f}"
    return {
        "workflow_slack_ms": float(slack_ms),
        "aegis_slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
        "aegis_slack_uses_future_information": False,
        "aegis_slack_residual_dag_visible": True,
        "aegis_slack_metadata_only": True,
        "aegis_slack_usable": True,
        "aegis_slack_reason": "visible_sequential_session" if slack_ms == 0 else "visible_controlled_fork_join",
        "aegis_slack_graph_version": f"request-layer-v2-s{seed}",
        "aegis_slack_node_id": request_id,
        "aegis_slack_agent_id": workflow_id,
        "aegis_slack_input_provenance_sha256": hashlib.sha256(identity.encode()).hexdigest(),
    }


@dataclass(frozen=True)
class WorkloadRow:
    request_id: str
    workflow_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    cached_prefix_tokens: int
    service_ms: float
    prefill_service_ms: float
    slack_ms: float
    provider: str
    raw_input_tokens: int
    raw_output_tokens: int
    affinity_key: str = ""
    slack_evidence: Mapping[str, Any] | None = None

    @property
    def reuse_fraction(self) -> float:
        return self.cached_prefix_tokens / max(self.prompt_tokens, 1)


@dataclass(frozen=True)
class Workload:
    rows: tuple[WorkloadRow, ...]
    metadata: dict[str, Any]

    def complete_requests(self, worker_ids: Sequence[str], *, seed: int) -> tuple[CompleteRequest, ...]:
        if not worker_ids:
            raise ValueError("worker_ids are required")
        requests = []
        for row in self.rows:
            affinity_key = (
                row.affinity_key
                or (row.request_id if row.provider == "controlled" else row.workflow_id)
            )
            worker = worker_ids[_stable_int(affinity_key) % len(worker_ids)]
            evidence = (
                dict(row.slack_evidence)
                if row.slack_evidence is not None
                else _slack_evidence(
                    row.request_id, row.workflow_id, row.slack_ms, seed
                )
            )
            requests.append(
                CompleteRequest(
                    request_id=row.request_id,
                    workflow_id=row.workflow_id,
                    arrival_ms=row.arrival_ms,
                    prompt_tokens=row.prompt_tokens,
                    output_tokens=row.output_tokens,
                    service_ms=row.service_ms,
                    prefill_service_ms=row.prefill_service_ms,
                    slack_ms=row.slack_ms,
                    slack_evidence=evidence,
                    preferred_source=worker,
                )
            )
        return tuple(requests)


def _session_bin(round_count: int) -> str | None:
    if round_count < 4:
        return None
    if round_count <= 16:
        return "4-16"
    if round_count <= 64:
        return "17-64"
    if round_count <= 256:
        return "65-256"
    return ">256"


def _select_sessions(con: Any, *, seed: int, per_bin: int) -> list[tuple[str, str, int]]:
    summaries = con.execute(
        "select provider, session_id, count(*) n from rounds group by 1,2"
    ).fetchall()
    strata: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
    for provider, session_id, count in summaries:
        bin_name = _session_bin(int(count))
        if bin_name is None:
            continue
        strata.setdefault((str(provider), bin_name), []).append(
            (str(provider), str(session_id), int(count))
        )
    selected: list[tuple[str, str, int]] = []
    for key, rows in sorted(strata.items()):
        ranked = sorted(rows, key=lambda row: _stable_int(f"{seed}:{row[0]}:{row[1]}"))
        if len(ranked) < per_bin:
            raise RuntimeError(f"TraceLab stratum {key} has only {len(ranked)} sessions")
        selected.extend(ranked[:per_bin])
    return selected


def _fetch_rounds(con: Any, sessions: Sequence[tuple[str, str, int]]) -> list[tuple[Any, ...]]:
    clauses = []
    params: list[str] = []
    for provider, session_id, _count in sessions:
        clauses.append("(r.provider = ? and r.session_id = ?)")
        params.extend((provider, session_id))
    query = f"""
        select r.provider, r.session_id, r.round_index, r.round_pk,
               r.input_tokens_total, r.prefix_tokens, r.newly_append_tokens,
               r.output_tokens, min(t.timestamp) as first_event_at
        from rounds r left join timing_events t using(round_pk)
        where {' or '.join(clauses)}
        group by all
        order by r.provider, r.session_id, r.round_index
    """
    return con.execute(query, params).fetchall()


def _project_tokens(
    raw_input: int,
    raw_prefix: int,
    raw_output: int,
    *,
    context_cap: int,
    source_median: int,
    output_cap: int,
) -> tuple[int, int, int]:
    output = min(max(int(raw_output), 0), int(output_cap), max(context_cap - 1, 0))
    input_budget = max(int(context_cap) - output, 1)
    scale = (0.5 * float(context_cap)) / max(int(source_median), 1)
    prompt = min(max(int(round(max(raw_input, 1) * scale)), 1), input_budget)
    prefix_fraction = min(max(int(raw_prefix), 0) / max(int(raw_input), 1), 1.0)
    prefix = min(int(round(prompt * prefix_fraction)), prompt)
    return prompt, prefix, output


def _tracelab_workload(
    request_config: Mapping[str, Any],
    model: Mapping[str, Any],
    axes: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    experiment_root: Path,
) -> Workload:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("TraceLab paper workload requires the benchmark extra (duckdb)") from exc

    workload = request_config["workload"]
    db_path = experiment_root / workload["artifact"]
    if not db_path.is_file():
        raise FileNotFoundError(f"pinned TraceLab artifact missing: {db_path}")
    actual_sha = sha256_file(db_path)
    if actual_sha != workload["sha256"]:
        raise RuntimeError(f"TraceLab SHA-256 mismatch: {actual_sha}")

    seed = int(axes["seed"])
    per_bin = int(profile.get("sessions_per_provider_bin", workload["sessions_per_provider_bin"]))
    max_rounds = max(
        int(profile.get("max_rounds_per_session", workload["max_rounds_per_session"])),
        int(workload.get("minimum_rounds_per_session", 1)),
    )
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        sessions = _select_sessions(con, seed=seed, per_bin=per_bin)
        raw_rows = _fetch_rounds(con, sessions)
    finally:
        con.close()

    by_session: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
    for row in raw_rows:
        by_session.setdefault((str(row[0]), str(row[1])), []).append(row)

    rng = random.Random(seed)
    load_multiplier = float(axes.get("load_multiplier", 1.0))
    session_rate = float(workload["session_arrival_rate_per_s"]) * load_multiplier
    context_cap = int(axes.get("context_tokens", workload["context_tokens"]))
    context_cap = min(context_cap, int(model["max_context_tokens"]))
    service = model["service_time"]
    session_start_ms = 0.0
    rows: list[WorkloadRow] = []
    session_records = sorted(
        by_session.items(), key=lambda item: _stable_int(f"order:{seed}:{item[0][0]}:{item[0][1]}")
    )
    for session_index, ((provider, session_id), session_rows) in enumerate(session_records):
        if session_index:
            session_start_ms += rng.expovariate(max(session_rate, 1e-9)) * 1000.0
        ordered = sorted(session_rows, key=lambda row: int(row[2]))
        if len(ordered) > max_rounds:
            start = _stable_int(f"window:{seed}:{provider}:{session_id}") % (len(ordered) - max_rounds + 1)
            ordered = ordered[start : start + max_rounds]
        timestamps = [row[8] for row in ordered]
        relative_ms = 0.0
        workflow_id = f"{provider}:{session_id}"
        previous_timestamp: datetime | None = None
        for row in ordered:
            timestamp = row[8]
            if previous_timestamp is not None and timestamp is not None:
                gap_ms = max((timestamp - previous_timestamp).total_seconds() * 1000.0, 0.0)
                relative_ms += min(gap_ms, float(workload["within_session_gap_cap_ms"])) * float(
                    workload["within_session_gap_scale"]
                )
            previous_timestamp = timestamp or previous_timestamp
            prompt, prefix, output = _project_tokens(
                int(row[4] or 0),
                int(row[5] or 0),
                int(row[7] or 0),
                context_cap=context_cap,
                source_median=int(workload["source_median_input_tokens"]),
                output_cap=int(workload["output_cap_tokens"]),
            )
            uncached = max(prompt - prefix, 0)
            prefill_service_ms = (
                float(service["setup_ms"])
                + uncached * float(service["prefill_ms_per_uncached_token"])
            )
            service_ms = prefill_service_ms + output * float(service["decode_ms_per_token"])
            request_id = f"tl-{provider}-{hashlib.sha1(session_id.encode()).hexdigest()[:10]}-{int(row[2])}"
            rows.append(
                WorkloadRow(
                    request_id=request_id,
                    workflow_id=workflow_id,
                    arrival_ms=session_start_ms + relative_ms,
                    prompt_tokens=prompt,
                    output_tokens=output,
                    cached_prefix_tokens=prefix,
                    service_ms=max(service_ms, 0.001),
                    prefill_service_ms=min(prefill_service_ms, max(service_ms, 0.001)),
                    slack_ms=0.0,
                    provider=provider,
                    raw_input_tokens=int(row[4] or 0),
                    raw_output_tokens=int(row[7] or 0),
                )
            )
    rows.sort(key=lambda row: (row.arrival_ms, row.request_id))
    return Workload(
        tuple(rows),
        {
            "profile": "tracelab_session",
            "dataset_sha256": actual_sha,
            "selected_sessions": len(session_records),
            "selected_rounds": len(rows),
            "sampling_unit": "provider_session",
            "context_projection": workload["context_projection"],
            "context_cap_tokens": context_cap,
            "maximum_projected_prompt_tokens": max(
                (row.prompt_tokens for row in rows), default=0
            ),
            "maximum_projected_total_tokens": max(
                (row.prompt_tokens + row.output_tokens for row in rows), default=0
            ),
            "load_multiplier": load_multiplier,
            "seed": seed,
        },
    )


def _controlled_fork_join(
    request_config: Mapping[str, Any],
    model: Mapping[str, Any],
    axes: Mapping[str, Any],
) -> Workload:
    workload = request_config["workload"]
    seed = int(axes["seed"])
    rng = random.Random(seed)
    workflow_count = int(workload["workflow_count"])
    branches = int(workload["branches_per_workflow"])
    context = min(int(axes.get("context_tokens", workload["context_tokens"])), int(model["max_context_tokens"]))
    load_multiplier = float(axes.get("load_multiplier", 1.0))
    rate = float(workload["session_arrival_rate_per_s"]) * load_multiplier
    service = model["service_time"]
    arrival_ms = 0.0
    rows: list[WorkloadRow] = []
    for workflow_index in range(workflow_count):
        if workflow_index:
            arrival_ms += rng.expovariate(max(rate, 1e-9)) * 1000.0
        raw_specs = []
        for branch in range(branches):
            prompt = max(int(context * rng.uniform(0.25, 0.95)), 1)
            reuse = rng.uniform(0.25, 0.95)
            prefix = int(prompt * reuse)
            output = min(rng.randint(64, 512), max(context - prompt, 0))
            prefill_service_ms = (
                float(service["setup_ms"])
                + (prompt - prefix) * float(service["prefill_ms_per_uncached_token"])
            )
            service_ms = prefill_service_ms + output * float(service["decode_ms_per_token"])
            raw_specs.append((branch, prompt, prefix, output, service_ms, prefill_service_ms))
        critical_ms = max(spec[4] for spec in raw_specs)
        workflow_id = f"forkjoin-{seed}-{workflow_index}"
        for branch, prompt, prefix, output, service_ms, prefill_service_ms in raw_specs:
            rows.append(
                WorkloadRow(
                    request_id=f"fj-{seed}-{workflow_index}-{branch}",
                    workflow_id=workflow_id,
                    arrival_ms=arrival_ms,
                    prompt_tokens=prompt,
                    output_tokens=output,
                    cached_prefix_tokens=prefix,
                    service_ms=service_ms,
                    prefill_service_ms=prefill_service_ms,
                    slack_ms=max(critical_ms - service_ms, 0.0),
                    provider="controlled",
                    raw_input_tokens=prompt,
                    raw_output_tokens=output,
                )
            )
    return Workload(
        tuple(rows),
        {
            "profile": "controlled_fork_join",
            "workflow_count": workflow_count,
            "branches_per_workflow": branches,
            "context_cap_tokens": context,
            "maximum_projected_prompt_tokens": max(
                (row.prompt_tokens for row in rows), default=0
            ),
            "maximum_projected_total_tokens": max(
                (row.prompt_tokens + row.output_tokens for row in rows), default=0
            ),
            "load_multiplier": load_multiplier,
            "seed": seed,
            "topology_evidence": "controlled_not_observed_trace",
        },
    )


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        count += 1
    return count


@lru_cache(maxsize=4)
def _weka_bundle(
    path_text: str,
    expected_sha256: str,
    folds: int,
    split_salt: str,
    interval_coverage: float,
    blend_tau_rounds: float,
    gbdt_json: str,
) -> tuple[
    tuple[WekaWorkflow, ...],
    dict[str, CausalSlackTicket],
    dict[str, Any],
    dict[str, Any],
]:
    path = Path(path_text)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Weka AgentX SHA-256 mismatch: {actual_sha256}"
        )
    workflows = tuple(load_weka_workflows(path))
    predictor = json.loads(gbdt_json)
    tickets, metrics = crossfit_causal_slack(
        build_causal_samples(workflows),
        folds=int(folds),
        split_salt=str(split_salt),
        interval_coverage=float(interval_coverage),
        blend_tau_rounds=float(blend_tau_rounds),
        gbdt_config=AegisGBDTConfig(**predictor),
    )
    tickets = zero_slack_tickets_for_unmodeled_requests(workflows, tickets)
    manifest = dataset_manifest(
        workflows, source_path=path, source_sha256=actual_sha256
    )
    return workflows, tickets, metrics, manifest


def _weka_slack_evidence(
    *,
    ticket: CausalSlackTicket,
    request_arrival_ms: float,
    dataset_sha256: str,
) -> dict[str, Any]:
    causal = ticket.published_at_ms <= request_arrival_ms + 1e-9
    modeled = ticket.fold >= 0 and causal
    identity = (
        f"{dataset_sha256}:{ticket.workflow_id}:{ticket.request_id}:"
        f"{ticket.published_at_ms:.9f}:{ticket.predictor_sha256}"
    )
    return {
        "aegis_slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
        "aegis_slack_uses_future_information": False,
        "aegis_slack_residual_dag_visible": True,
        "aegis_slack_metadata_only": True,
        "aegis_slack_usable": bool(modeled),
        "aegis_slack_reason": (
            "weka_agentx_workflow_disjoint_crossfit_lcb"
            if modeled
            else "weka_agentx_fail_close_zero"
        ),
        "aegis_slack_graph_version": (
            f"weka-causal-fold-{ticket.fold}-{ticket.predictor_sha256[:16]}"
            if modeled
            else "weka-causal-fail-close-zero"
        ),
        "aegis_slack_node_id": ticket.request_id,
        "aegis_slack_agent_id": ticket.agent_id,
        "aegis_slack_observed_at_ms": float(ticket.published_at_ms),
        "aegis_slack_input_provenance_sha256": hashlib.sha256(
            identity.encode()
        ).hexdigest(),
        "weka_crossfit_fold": int(ticket.fold),
        "weka_predictor_sha256": ticket.predictor_sha256,
        "weka_ticket_publish_causal": bool(causal),
    }


def _weka_agentx_workload(
    request_config: Mapping[str, Any],
    model: Mapping[str, Any],
    axes: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    experiment_root: Path,
) -> Workload:
    workload = request_config["workload"]
    artifact = (experiment_root / str(workload["artifact"])).resolve()
    predictor = workload["causal_slack_predictor"]
    gbdt = dict(predictor["gbdt"])
    workflows, tickets, metrics, source_manifest = _weka_bundle(
        str(artifact),
        str(workload["sha256"]),
        int(predictor["folds"]),
        str(predictor["split_salt"]),
        float(predictor["interval_coverage"]),
        float(predictor["blend_tau_rounds"]),
        json.dumps(gbdt, sort_keys=True, separators=(",", ":")),
    )
    maximum_requests = int(workload["maximum_requests_per_workflow"])
    eligible = [
        workflow
        for workflow in workflows
        if any(len(stage.branches) > 1 for stage in workflow.stages)
        and int(workload["minimum_requests_per_workflow"])
        <= len(workflow.requests)
        <= maximum_requests
    ]
    seed = int(axes["seed"])
    selected_count = int(
        profile.get("weka_workflows", workload["workflows_per_seed"])
    )
    sampling_mode = str(workload["sampling_mode"])
    if sampling_mode != "disjoint_seed_partition_without_replacement":
        raise ValueError(f"unknown Weka sampling mode {sampling_mode!r}")
    ranked = sorted(
        eligible,
        key=lambda workflow: (
            _stable_int(
                f"{workload['sampling_partition_salt']}:{workflow.workflow_id}"
            ),
            workflow.workflow_id,
        ),
    )
    partition = seed - int(workload["sampling_seed_base"])
    if partition < 0:
        raise ValueError("Weka sampling seed precedes sampling_seed_base")
    selection_start = partition * selected_count
    selection_end = selection_start + selected_count
    if len(ranked) < selection_end:
        raise RuntimeError(
            f"Weka eligibility leaves {len(ranked)} workflows, disjoint partition "
            f"requires {selection_end}"
        )
    selected = ranked[selection_start:selection_end]
    context_cap = min(
        int(axes.get("context_tokens", workload["context_tokens"])),
        int(model["max_context_tokens"]),
    )
    fixed_start_interval_ms = float(workload["fixed_workflow_start_interval_ms"])
    service = model["service_time"]
    rows: list[WorkloadRow] = []
    causal_ticket_count = 0
    nonzero_slack_count = 0
    for workflow_index, workflow in enumerate(selected):
        workflow_start_ms = workflow_index * fixed_start_interval_ms
        source_start_ms = workflow.start_ms
        prior_hashes: dict[str, tuple[int, ...]] = {}
        ordered = sorted(
            workflow.requests, key=lambda row: (row.arrival_ms, row.rid)
        )
        for request in ordered:
            branch_identity = (
                request.source_agent_id
                if request.source_agent_id
                else f"{request.role}-root"
            )
            affinity_key = f"{workflow.workflow_id}:{branch_identity}"
            prior = prior_hashes.get(affinity_key, ())
            prefix_blocks = _longest_common_prefix(prior, request.hash_ids)
            raw_prefix_tokens = min(
                prefix_blocks * int(workflow.block_size),
                request.prompt_tokens,
            )
            prior_hashes[affinity_key] = request.hash_ids
            prompt, prefix, output = _project_tokens(
                request.prompt_tokens,
                raw_prefix_tokens,
                request.output_tokens,
                context_cap=context_cap,
                source_median=int(workload["source_median_input_tokens"]),
                output_cap=int(workload["output_cap_tokens"]),
            )
            uncached = max(prompt - prefix, 0)
            prefill_service_ms = (
                float(service["setup_ms"])
                + uncached * float(service["prefill_ms_per_uncached_token"])
            )
            service_ms = (
                prefill_service_ms
                + output * float(service["decode_ms_per_token"])
            )
            ticket = tickets[request.rid]
            # Ticket times and request arrivals share the source-workflow time
            # base.  A non-causal or unmodeled ticket is already fail-closed.
            evidence = _weka_slack_evidence(
                ticket=ticket,
                request_arrival_ms=request.arrival_ms,
                dataset_sha256=str(workload["sha256"]),
            )
            slack_ms = (
                float(ticket.slack_lcb_ms)
                if evidence["aegis_slack_usable"]
                else 0.0
            )
            causal_ticket_count += int(bool(evidence["aegis_slack_usable"]))
            nonzero_slack_count += int(slack_ms > 1e-9)
            rows.append(
                WorkloadRow(
                    request_id=request.rid,
                    workflow_id=workflow.workflow_id,
                    arrival_ms=(
                        workflow_start_ms + request.arrival_ms - source_start_ms
                    ),
                    prompt_tokens=prompt,
                    output_tokens=output,
                    cached_prefix_tokens=prefix,
                    service_ms=max(service_ms, 0.001),
                    prefill_service_ms=min(
                        prefill_service_ms, max(service_ms, 0.001)
                    ),
                    slack_ms=slack_ms,
                    provider="weka_agentx",
                    raw_input_tokens=request.prompt_tokens,
                    raw_output_tokens=request.output_tokens,
                    affinity_key=affinity_key,
                    slack_evidence=evidence,
                )
            )
    rows.sort(key=lambda row: (row.arrival_ms, row.request_id))
    predictor_metrics = {
        key: value for key, value in metrics.items() if key != "predictors"
    }
    predictor_metrics["predictor_sha256_by_fold"] = {
        fold: value["sha256"]
        for fold, value in metrics["predictors"].items()
    }
    return Workload(
        tuple(rows),
        {
            "profile": "weka_agentx_causal",
            "dataset_sha256": workload["sha256"],
            "dataset_manifest": source_manifest,
            "eligible_workflow_count": len(eligible),
            "selected_workflows": len(selected),
            "selected_workflow_ids": [row.workflow_id for row in selected],
            "selected_requests": len(rows),
            "sampling_unit": "weka_agentx_workflow",
            "sampling_mode": sampling_mode,
            "sampling_partition": partition,
            "sampling_without_replacement": True,
            "arrival_model": "recorded_intra_workflow_fixed_schedule_plus_fixed_workflow_starts",
            "fixed_workflow_start_interval_ms": fixed_start_interval_ms,
            "source_api_duration_used_as_service_time": False,
            "context_projection": workload["context_projection"],
            "context_cap_tokens": context_cap,
            "maximum_projected_prompt_tokens": max(
                (row.prompt_tokens for row in rows), default=0
            ),
            "maximum_projected_total_tokens": max(
                (row.prompt_tokens + row.output_tokens for row in rows), default=0
            ),
            "prefix_semantics": "longest_hash_block_prefix_with_prior_same_agent_request",
            "causal_ticket_count": causal_ticket_count,
            "nonzero_slack_ticket_count": nonzero_slack_count,
            "slack_predictor": predictor_metrics,
            "seed": seed,
        },
    )


def build_workload(
    request_config: Mapping[str, Any],
    model: Mapping[str, Any],
    axes: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    experiment_root: Path,
) -> Workload:
    workload_profile = request_config["workload"]["profile"]
    if workload_profile == "tracelab_session":
        return _tracelab_workload(
            request_config,
            model,
            axes,
            profile=profile,
            experiment_root=experiment_root,
        )
    if workload_profile == "controlled_fork_join":
        return _controlled_fork_join(request_config, model, axes)
    if workload_profile == "weka_agentx_causal":
        return _weka_agentx_workload(
            request_config,
            model,
            axes,
            profile=profile,
            experiment_root=experiment_root,
        )
    raise ValueError(f"unknown workload profile {workload_profile!r}")


__all__ = ["Workload", "WorkloadRow", "build_workload"]
