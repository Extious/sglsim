"""Admission-visible causal Slack inputs for the global-DES paper suite.

TraceLab provider sessions are sequential request streams, not observed
multi-agent DAGs.  They therefore receive a valid *zero*-Slack singleton CPM
ticket.  P4 additionally appends a small, explicitly modeled fork/join
foreground so that Algorithm 1 can be exercised without pretending that the
TraceLab rows contain an observed branch topology.

No simulated completion timestamp, failure outcome, or post-hoc service time
is consumed here.  Every positive ticket is computed from the request-layer
fork/join contract and its frozen admission-time predictor.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from sglsim.srt.simulation.aegis_residual_dag import (
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
)


CAUSAL_WORKLOAD_SCHEMA = "aegiskv.global-des-causal-workload.v1"
SEQUENTIAL_ZERO_MODE = "visible_sequential_singleton_zero_v1"
CONTROLLED_OVERLAY_MODE = (
    "tracelab_background_plus_controlled_visible_fork_join_v1"
)


class CausalWorkloadError(ValueError):
    """Raised before execution when the causal-workload contract is invalid."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_nonnegative(name: str, value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CausalWorkloadError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise CausalWorkloadError(f"{name} must be finite and nonnegative")
    return parsed


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or int(value) <= 0:
        raise CausalWorkloadError(f"{name} must be a positive integer")
    return int(value)


def _ticket_fields(
    *,
    rid: str,
    workflow_id: str,
    agent_id: str,
    observed_at_ms: float,
    graph_version: str,
    predicted_remaining_ms: float,
) -> dict[str, Any]:
    graph = VisibleResidualDag(
        workflow_id=str(workflow_id),
        graph_version=str(graph_version),
        observed_at_ms=float(observed_at_ms),
        nodes=(
            ResidualDagNode("ingress", "control", 0.0),
            ResidualDagNode(
                str(rid),
                "agent",
                float(predicted_remaining_ms),
                active=True,
                agent_id=str(agent_id),
            ),
            ResidualDagNode("join", "control", 0.0),
        ),
        edges=(("ingress", str(rid)), (str(rid), "join")),
        next_join_subgraph_complete=True,
        metadata_complete=True,
        uses_future_information=False,
    )
    return build_residual_dag_slack_tickets(graph)[str(rid)].to_simulation_fields()


def attach_sequential_zero_slack(
    records: Iterable[Mapping[str, Any]],
    *,
    group: str,
    contract: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Attach a real CPM ticket whose topology proves exact zero Slack.

    A singleton residual frontier has no parallel peer and therefore has zero
    total float regardless of its latency prediction.  We use the configured
    constant only to keep the residual DAG structurally complete; it cannot
    create positive Slack or change scheduling time.
    """

    if str(contract.get("mode")) not in {
        SEQUENTIAL_ZERO_MODE,
        CONTROLLED_OVERLAY_MODE,
    }:
        raise CausalWorkloadError("unsupported causal Slack contract mode")
    if contract.get("future_information") is not False:
        raise CausalWorkloadError("causal Slack contract must forbid future information")
    if contract.get("missing_metadata") != "fail_close_zero":
        raise CausalWorkloadError("missing causal metadata must fail close to zero")
    background = contract.get("background", {})
    if not isinstance(background, Mapping):
        raise CausalWorkloadError("causal Slack background contract must be an object")
    if background.get("topology") != "single_visible_agent_to_next_join":
        raise CausalWorkloadError("sequential background topology is not frozen")
    if background.get("positive_slack_forbidden") is not True:
        raise CausalWorkloadError("sequential background must forbid positive Slack")
    prediction_ms = _finite_nonnegative(
        "background.predicted_remaining_ms",
        background.get("predicted_remaining_ms", 1.0),
    )
    version = str(background.get("graph_version") or "")
    if not version:
        raise CausalWorkloadError("sequential background graph_version is required")

    output: list[dict[str, Any]] = []
    for raw in records:
        row = copy.deepcopy(dict(raw))
        rid = str(row.get("rid") or "")
        if not rid:
            raise CausalWorkloadError("request record is missing rid")
        workflow_id = str(row.get("workflow") or f"workflow/{rid}")
        provider = str(row.get("provider") or "request")
        observed_at_ms = int(row.get("arrival_ns", 0)) / 1_000_000.0
        simulation = dict(row.get("simulation") or {})
        simulation.update(
            _ticket_fields(
                rid=rid,
                workflow_id=workflow_id,
                agent_id=provider,
                observed_at_ms=observed_at_ms,
                graph_version=f"{version}:{group}:{rid}",
                predicted_remaining_ms=prediction_ms,
            )
        )
        if float(simulation["workflow_slack_ms"]) != 0.0:
            raise AssertionError("sequential singleton unexpectedly produced positive Slack")
        simulation.update(
            {
                "aegis_slack_reason": "visible_sequential_singleton_exact_zero",
                "aegis_slack_workload_evidence": "observed_arrivals_modeled_topology",
                "aegis_slack_positive_value_source": "none",
            }
        )
        row["simulation"] = simulation
        output.append(row)
    return tuple(output), {
        "mode": SEQUENTIAL_ZERO_MODE,
        "record_count": len(output),
        "nonzero_ticket_count": 0,
        "uses_future_information": False,
        "graph_version": version,
    }


def _controlled_predictor(contract: Mapping[str, Any]) -> tuple[float, float, float]:
    predictor = contract.get("admission_predictor")
    if not isinstance(predictor, Mapping):
        raise CausalWorkloadError("controlled overlay admission_predictor is required")
    if predictor.get("kind") != "frozen_linear_uncached_prefill_plus_decode_v1":
        raise CausalWorkloadError("unsupported controlled admission predictor")
    if predictor.get("uses_simulated_completion") is not False:
        raise CausalWorkloadError("admission predictor may not use simulated completion")
    if predictor.get("hardware_calibrated") is not False:
        raise CausalWorkloadError(
            "controlled predictor must remain explicitly modeled and uncalibrated"
        )
    return (
        _finite_nonnegative("predictor.setup_ms", predictor.get("setup_ms")),
        _finite_nonnegative(
            "predictor.prefill_ms_per_uncached_token",
            predictor.get("prefill_ms_per_uncached_token"),
        ),
        _finite_nonnegative(
            "predictor.decode_ms_per_token",
            predictor.get("decode_ms_per_token"),
        ),
    )


def build_controlled_fork_join_overlay(
    *,
    group: str,
    seed: int,
    context_tokens: int,
    contract: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Build an admission-visible modeled fork/join foreground for P4."""

    overlay = contract.get("overlay")
    if not isinstance(overlay, Mapping):
        raise CausalWorkloadError("controlled overlay specification is required")
    if overlay.get("evidence_class") != "modeled_mechanism_workload_not_observed_trace":
        raise CausalWorkloadError("controlled overlay evidence class must be explicit")
    workflow_count = _positive_int("overlay.workflow_count", overlay.get("workflow_count"))
    branches = _positive_int("overlay.branches_per_workflow", overlay.get("branches_per_workflow"))
    if branches < 2:
        raise CausalWorkloadError("controlled fork/join requires at least two branches")
    interval_ms = _finite_nonnegative(
        "overlay.workflow_interval_ms", overlay.get("workflow_interval_ms")
    )
    fractions = overlay.get("prompt_fractions")
    if not isinstance(fractions, Sequence) or isinstance(fractions, (str, bytes)):
        raise CausalWorkloadError("overlay.prompt_fractions must be an array")
    prompt_fractions = tuple(float(value) for value in fractions)
    if len(prompt_fractions) != branches or any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in prompt_fractions
    ):
        raise CausalWorkloadError("one prompt fraction in (0,1] is required per branch")
    output_tokens = overlay.get("output_tokens")
    if not isinstance(output_tokens, Sequence) or isinstance(output_tokens, (str, bytes)):
        raise CausalWorkloadError("overlay.output_tokens must be an array")
    outputs = tuple(int(value) for value in output_tokens)
    if len(outputs) != branches or any(value < 0 for value in outputs):
        raise CausalWorkloadError("one nonnegative output size is required per branch")
    prefix_fraction = _finite_nonnegative(
        "overlay.cached_prefix_fraction", overlay.get("cached_prefix_fraction")
    )
    if prefix_fraction > 1.0:
        raise CausalWorkloadError("cached_prefix_fraction cannot exceed one")
    setup_ms, prefill_ms, decode_ms = _controlled_predictor(contract)
    graph_version = str(overlay.get("graph_version") or "")
    if not graph_version:
        raise CausalWorkloadError("controlled overlay graph_version is required")

    records: list[dict[str, Any]] = []
    nonzero = 0
    for workflow_index in range(workflow_count):
        workflow_id = f"p4-controlled-fj-{seed}-{workflow_index}"
        # A seed-dependent rotation changes which branch is critical while
        # preserving the exact same request universe across policy variants.
        rotation = int.from_bytes(
            hashlib.sha256(f"{seed}:{workflow_index}".encode()).digest()[:4],
            "big",
        ) % branches
        specs: list[tuple[str, int, int, int, float]] = []
        for branch in range(branches):
            shape = (branch + rotation) % branches
            prompt = max(int(round(int(context_tokens) * prompt_fractions[shape])), 1)
            output = min(outputs[shape], max(int(context_tokens) - prompt, 0))
            cached = min(int(round(prompt * prefix_fraction)), prompt)
            predicted = setup_ms + max(prompt - cached, 0) * prefill_ms + output * decode_ms
            rid = f"p4-fj-{seed}-{workflow_index}-{branch}"
            specs.append((rid, prompt, output, cached, predicted))

        nodes = [ResidualDagNode("fork", "control", 0.0)]
        nodes.extend(
            ResidualDagNode(
                rid,
                "agent",
                predicted,
                active=True,
                agent_id=f"branch-{branch}",
            )
            for branch, (rid, _prompt, _output, _cached, predicted) in enumerate(specs)
        )
        nodes.append(ResidualDagNode("join", "control", 0.0))
        edges = [
            edge
            for rid, _prompt, _output, _cached, _predicted in specs
            for edge in (("fork", rid), (rid, "join"))
        ]
        arrival_ns = int(round(workflow_index * interval_ms * 1_000_000.0))
        graph = VisibleResidualDag(
            workflow_id=workflow_id,
            graph_version=f"{graph_version}:{seed}:{workflow_index}",
            observed_at_ms=arrival_ns / 1_000_000.0,
            nodes=tuple(nodes),
            edges=tuple(edges),
            next_join_subgraph_complete=True,
            metadata_complete=True,
            uses_future_information=False,
        )
        tickets = build_residual_dag_slack_tickets(graph)
        for branch, (rid, prompt, output, cached, predicted) in enumerate(specs):
            simulation = tickets[rid].to_simulation_fields()
            simulation.update(
                {
                    "aegis_slack_reason": "controlled_visible_fork_join_modeled_predictor",
                    "aegis_slack_workload_evidence": overlay["evidence_class"],
                    "aegis_slack_positive_value_source": "frozen_admission_predictor",
                    "aegis_slack_predicted_remaining_ms": predicted,
                    "controlled_fork_join_branch": branch,
                }
            )
            nonzero += int(float(simulation["workflow_slack_ms"]) > 0.0)
            records.append(
                {
                    "rid": rid,
                    "workflow": workflow_id,
                    "session": workflow_id,
                    "provider": "controlled_visible_fork_join",
                    "agent_id": f"{workflow_id}/branch-{branch}",
                    "worker_seq": 1,
                    "round_index": branch,
                    "arrival_ns": arrival_ns,
                    "input_tokens": prompt,
                    "output_tokens": output,
                    "cached_prefix_tokens": cached,
                    "prefix_key": f"{workflow_id}/branch-{branch}",
                    "source_round_pk": 10_000_000_000 + workflow_index * branches + branch,
                    "source_round_id": f"controlled-{seed}-{workflow_index}-{branch}",
                    "simulation": simulation,
                }
            )
    predictor_lineage = {
        "setup_ms": setup_ms,
        "prefill_ms_per_uncached_token": prefill_ms,
        "decode_ms_per_token": decode_ms,
        "hardware_calibrated": False,
        "uses_simulated_completion": False,
    }
    return tuple(records), {
        "mode": CONTROLLED_OVERLAY_MODE,
        "workflow_count": workflow_count,
        "branch_count": len(records),
        "nonzero_ticket_count": nonzero,
        "uses_future_information": False,
        "evidence_class": overlay["evidence_class"],
        "predictor_lineage": predictor_lineage,
        "predictor_sha256": _canonical_sha256(predictor_lineage),
    }


def enrich_causal_workload(
    records: Iterable[Mapping[str, Any]],
    *,
    group: str,
    seed: int,
    context_tokens: int,
    contract: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Apply the configured causal input contract without post-hoc data."""

    background, background_audit = attach_sequential_zero_slack(
        records,
        group=group,
        contract=contract,
    )
    mode = str(contract.get("mode"))
    if mode == SEQUENTIAL_ZERO_MODE:
        return background, {
            "schema": CAUSAL_WORKLOAD_SCHEMA,
            "mode": mode,
            "background": background_audit,
            "overlay": None,
        }
    if mode != CONTROLLED_OVERLAY_MODE:
        raise CausalWorkloadError(f"unsupported causal workload mode {mode!r}")
    overlay, overlay_audit = build_controlled_fork_join_overlay(
        group=group,
        seed=int(seed),
        context_tokens=int(context_tokens),
        contract=contract,
    )
    # Foreground first makes a bounded smoke exercise the positive-Slack path;
    # full cells still contain the complete frozen TraceLab background.
    combined = tuple(overlay) + tuple(background)
    return combined, {
        "schema": CAUSAL_WORKLOAD_SCHEMA,
        "mode": mode,
        "background": background_audit,
        "overlay": overlay_audit,
        "combined_record_count": len(combined),
    }


__all__ = [
    "CAUSAL_WORKLOAD_SCHEMA",
    "CONTROLLED_OVERLAY_MODE",
    "CausalWorkloadError",
    "SEQUENTIAL_ZERO_MODE",
    "attach_sequential_zero_slack",
    "build_controlled_fork_join_overlay",
    "enrich_causal_workload",
]
