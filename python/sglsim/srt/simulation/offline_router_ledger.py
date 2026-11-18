"""Router-owned output fencing for the offline Scheduler/GlobalDES path.

This module closes the *metadata simulator* continuation loop.  SGLSim does
not execute model logits or preserve a real sampler RNG state; consequently
the durable continuation payload below is explicitly limited to deterministic
synthetic token IDs and sampling positions.  It must not be cited as evidence
of bit-exact continuation for a real model runtime.

The Router is the only owner of the logical request, route version, durable
output journal, and externally published position.  Scheduler executions only
report monotonically increasing generated-token counts.  A report is converted
to durable synthetic output entries before the corresponding detokenizer
message is allowed to leave the parent process.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from sglsim.srt.managers.io_struct import TokenizedGenerateReqInput
from sglsim.srt.simulation.aegis_recovery_protocol import (
    DurableContinuationState,
    DurableInputObject,
    ProtocolViolation,
    RouterOutputJournal,
    compute_input_manifest_digest,
)


ROUTER_LEDGER_SCHEMA = "sglsim.offline.router-output-ledger.v1"
REPLAY_CONTRACT_SCHEMA = "sglsim.metadata-output-replay.v1"
CONTINUATION_SCHEMA = "sglsim.metadata-continuation.v1"
CONTINUATION_EVIDENCE_CLASS = (
    "metadata_only_synthetic_token_continuation_not_real_logits_or_rng"
)
READY_ACK_MODE = (
    "scheduler_terminal_physical_pp_roundtrip_or_logical_pp_tp_aggregate_v1"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _digest(label: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_bytes(value))
    return digest.hexdigest()


def _simulation(request: TokenizedGenerateReqInput) -> dict[str, Any]:
    value = request.sampling_params.custom_params.setdefault("simulation", {})
    if not isinstance(value, dict):
        raise ProtocolViolation("request simulation metadata must be an object")
    return value


def _prompt_identity(
    request: TokenizedGenerateReqInput,
) -> tuple[tuple[int, ...], int, str]:
    prompt_count = int(
        request.input_token_count
        if request.input_token_count is not None
        else len(request.input_ids)
    )
    if prompt_count < 0:
        raise ProtocolViolation("canonical prompt length is negative")
    if request.input_ids:
        if len(request.input_ids) != prompt_count:
            raise ProtocolViolation("canonical prompt IDs/length disagree")
        return tuple(int(token) for token in request.input_ids), prompt_count, "token_ids"
    # Some low-level metadata tests construct length-only requests.  Preserve
    # their shape with an explicit placeholder identity.  Formal suite replay
    # requires token_ids and audits this field fail-closed.
    return tuple(0 for _ in range(prompt_count)), prompt_count, "length_only_placeholder"


@dataclass
class _Execution:
    execution_id: str
    route_version: int
    predecessor_execution_id: str | None
    base_output_sequence: int
    requested_new_tokens: int
    strict_replay_contract: bool
    state: str
    observed_generated_tokens: int = 0
    assigned_dp_rank: int | None = None
    expected_ready_ranks: tuple[str, ...] = ()
    ready_ranks: set[str] = field(default_factory=set)
    ready_ack_mode: str | None = None
    ready_ack_evidence: dict[str, Any] = field(default_factory=dict)
    committed_sequences: list[int] = field(default_factory=list)
    failure_event_id: str | None = None
    failure_committed_output_position: int | None = None
    failure_continuation_digest: str | None = None


@dataclass
class _LogicalRequest:
    logical_request_id: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_count: int
    prompt_identity_mode: str
    output_budget: int
    synthetic_output_token_id: int
    routing_key: str | None
    compatibility_hash: str
    input_object: DurableInputObject
    journal: RouterOutputJournal
    executions: list[_Execution]
    current_execution_id: str
    state: str = "active"
    failures: list[dict[str, Any]] = field(default_factory=list)
    stale_output_rejections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def current(self) -> _Execution:
        for execution in reversed(self.executions):
            if execution.execution_id == self.current_execution_id:
                return execution
        raise ProtocolViolation("logical request has no current execution")


class OfflineRouterLedger:
    """Durable logical-request/output authority owned by the offline parent."""

    def __init__(
        self,
        *,
        model_revision: str,
        kv_abi: str,
        parallel_layout: str,
    ) -> None:
        if not model_revision or not kv_abi or not parallel_layout:
            raise ProtocolViolation("offline Router compatibility identity is incomplete")
        self.compatibility_hash = _digest(
            "sglsim.offline.compatibility.v1",
            {
                "model_revision": str(model_revision),
                "kv_abi": str(kv_abi),
                "parallel_layout": str(parallel_layout),
                "continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
            },
        )
        self._logical: dict[str, _LogicalRequest] = {}
        self._execution_to_logical: dict[str, str] = {}
        self._physical_recovery_gate_observed = False

    def has_execution(self, execution_id: str) -> bool:
        return str(execution_id) in self._execution_to_logical

    def _lookup_execution(self, execution_id: str) -> tuple[_LogicalRequest, _Execution]:
        logical_id = self._execution_to_logical.get(str(execution_id))
        if logical_id is None:
            raise ProtocolViolation(
                f"Router has no registered execution {execution_id!r}"
            )
        logical = self._logical[logical_id]
        for execution in logical.executions:
            if execution.execution_id == execution_id:
                return logical, execution
        raise ProtocolViolation("Router execution index is corrupt")

    def register_dispatch(self, request: TokenizedGenerateReqInput) -> dict[str, Any]:
        """Register an original or bind one strict replay before dispatch."""

        execution_id = str(request.rid)
        if not execution_id or execution_id in self._execution_to_logical:
            raise ProtocolViolation(
                f"duplicate/empty Router execution ID {execution_id!r}"
            )
        sim = _simulation(request)
        is_replay = bool(sim.get("failover_replay", False))
        strict_replay = (
            str(sim.get("aegis_router_replay_contract") or "")
            == REPLAY_CONTRACT_SCHEMA
        )
        if is_replay and strict_replay:
            return self._bind_strict_replay(request)

        # Legacy tests may submit a retry directly without an earlier Router
        # fence.  Keep that request runnable, but give it an independent
        # logical identity and mark it non-authoritative for formal replay.
        logical_id = str(sim.get("aegis_logical_request_id") or execution_id)
        if logical_id in self._logical:
            raise ProtocolViolation(
                "non-strict replay cannot attach to an existing logical request"
            )
        prompt_ids, prompt_count, prompt_mode = _prompt_identity(request)
        output_budget = int(request.sampling_params.max_new_tokens)
        if output_budget < 0:
            raise ProtocolViolation("logical output budget is negative")
        route_version = int(sim.get("aegis_route_version", 1))
        if route_version <= 0:
            raise ProtocolViolation("initial route version must be positive")
        synthetic_token = int(sim.get("synthetic_output_token_id", 0))
        prompt_digest = _digest(
            "sglsim.offline.canonical-input.v1",
            {
                "logical_request_id": logical_id,
                "prompt_token_ids": list(prompt_ids),
                "prompt_token_count": prompt_count,
                "prompt_identity_mode": prompt_mode,
                "routing_key": request.routing_key,
            },
        )
        input_object = DurableInputObject(
            object_id=f"router-input/{logical_id}",
            content_digest=prompt_digest,
            schema_version="sglsim.offline.canonical-input.v1",
            storage_failure_domain="router-durable-control-plane",
        )
        manifest_digest = compute_input_manifest_digest((input_object,))
        initial_state_payload = {
            "schema": CONTINUATION_SCHEMA,
            "logical_request_id": logical_id,
            "prompt_token_count": prompt_count,
            "next_output_sequence": 1,
            "next_sampling_position": prompt_count,
            "evidence_class": CONTINUATION_EVIDENCE_CLASS,
        }
        initial_state_digest = _digest(
            "sglsim.metadata-continuation-state.v1", initial_state_payload
        )
        journal = RouterOutputJournal(
            current_execution=execution_id,
            current_route_version=route_version,
            input_manifest_digest=manifest_digest,
            initial_state_digest=initial_state_digest,
            initial_sampling_position=prompt_count,
        )
        execution = _Execution(
            execution_id=execution_id,
            route_version=route_version,
            predecessor_execution_id=None,
            base_output_sequence=0,
            requested_new_tokens=output_budget,
            strict_replay_contract=not is_replay,
            state="active",
        )
        logical = _LogicalRequest(
            logical_request_id=logical_id,
            prompt_token_ids=prompt_ids,
            prompt_token_count=prompt_count,
            prompt_identity_mode=prompt_mode,
            output_budget=output_budget,
            synthetic_output_token_id=synthetic_token,
            routing_key=request.routing_key,
            compatibility_hash=self.compatibility_hash,
            input_object=input_object,
            journal=journal,
            executions=[execution],
            current_execution_id=execution_id,
        )
        self._logical[logical_id] = logical
        self._execution_to_logical[execution_id] = logical_id
        sim.setdefault("aegis_logical_request_id", logical_id)
        sim.setdefault("aegis_route_version", route_version)
        sim.setdefault("aegis_continuation_evidence_class", CONTINUATION_EVIDENCE_CLASS)
        sim.setdefault("aegis_router_output_ledger_schema", ROUTER_LEDGER_SCHEMA)
        return self.execution_record(execution_id)

    def _bind_strict_replay(
        self, request: TokenizedGenerateReqInput
    ) -> dict[str, Any]:
        execution_id = str(request.rid)
        sim = _simulation(request)
        logical_id = str(sim.get("aegis_logical_request_id") or "")
        if not logical_id or logical_id not in self._logical:
            raise ProtocolViolation("strict replay has no fenced logical request")
        logical = self._logical[logical_id]
        predecessor = logical.current
        if logical.state != "fenced" or predecessor.state != "fenced":
            raise ProtocolViolation("strict replay may only bind after Router fence")
        committed = len(logical.journal.entries)
        remaining = logical.output_budget - committed
        if remaining <= 0:
            raise ProtocolViolation("completed logical request must not be replayed")
        route_version = int(sim.get("aegis_route_version", 0))
        if route_version != predecessor.route_version + 1:
            raise ProtocolViolation("strict replay route version is not predecessor + 1")
        if int(sim.get("aegis_committed_output_position", -1)) != committed:
            raise ProtocolViolation("strict replay committed boundary mismatch")
        if int(request.sampling_params.max_new_tokens) != remaining:
            raise ProtocolViolation("strict replay remaining max_new_tokens mismatch")
        if int(sim.get("synthetic_output_token_id", 0)) != (
            logical.synthetic_output_token_id
        ):
            raise ProtocolViolation("strict replay synthetic token identity mismatch")
        expected_input = logical.prompt_token_ids + tuple(
            entry.token_id for entry in logical.journal.entries
        )
        if tuple(int(token) for token in request.input_ids) != expected_input:
            raise ProtocolViolation("strict replay input is not prompt + accepted outputs")
        expected_continuation = self.continuation_state(logical_id)
        if (
            str(sim.get("aegis_metadata_compatibility_hash") or "")
            != expected_continuation.compatibility_hash
        ):
            raise ProtocolViolation("strict replay compatibility hash mismatch")
        if (
            str(sim.get("aegis_metadata_prefix_hash") or "")
            != expected_continuation.prefix_hash
        ):
            raise ProtocolViolation("strict replay prefix hash mismatch")
        if (
            str(sim.get("aegis_metadata_continuation_digest") or "")
            != expected_continuation.state_digest
        ):
            raise ProtocolViolation("strict replay continuation digest mismatch")
        if int(sim.get("pre_failure_total_tokens", -1)) != (
            logical.prompt_token_count + committed
        ):
            raise ProtocolViolation("strict replay pre_failure_total_tokens mismatch")
        logical.journal.fence_worker(
            new_execution=execution_id,
            new_route_version=route_version,
        )
        execution = _Execution(
            execution_id=execution_id,
            route_version=route_version,
            predecessor_execution_id=predecessor.execution_id,
            base_output_sequence=committed,
            requested_new_tokens=remaining,
            strict_replay_contract=True,
            state="recovering",
            failure_event_id=str(sim.get("failure_event_id") or "") or None,
            failure_committed_output_position=committed,
            failure_continuation_digest=expected_continuation.state_digest,
        )
        logical.executions.append(execution)
        logical.current_execution_id = execution_id
        logical.state = "recovering"
        self._execution_to_logical[execution_id] = logical_id
        return self.execution_record(execution_id)

    def set_assignment(
        self,
        execution_id: str,
        *,
        dp_rank: int,
        pp_size: int,
        logical_tp_size: int,
        configured_pp_size: int | None = None,
        logical_pp: bool = False,
    ) -> None:
        logical, execution = self._lookup_execution(execution_id)
        del logical
        if pp_size <= 0 or logical_tp_size <= 0:
            raise ProtocolViolation("replay rank topology is invalid")
        execution.assigned_dp_rank = int(dp_rank)
        resolved_configured_pp = int(configured_pp_size or pp_size)
        if logical_pp:
            if int(pp_size) != 1 or resolved_configured_pp <= 0:
                raise ProtocolViolation("logical PP aggregate topology is invalid")
            execution.expected_ready_ranks = (
                f"dp{int(dp_rank)}.pp_aggregate[0:{resolved_configured_pp})"
                f".tp_aggregate[0:{int(logical_tp_size)})",
            )
        else:
            execution.expected_ready_ranks = tuple(
                f"dp{int(dp_rank)}.pp{pp_rank}"
                f".tp_aggregate[0:{int(logical_tp_size)})"
                for pp_rank in range(int(pp_size))
            )

    def confirm_dispatched_route(
        self,
        execution_id: str,
        *,
        route_version: int,
    ) -> None:
        """Bind the controller-selected initial route before any output.

        Certified exact routing is resolved inside the parent DP controller
        during dispatch.  It may replace the provisional version carried by an
        original request.  A recovery route is already fenced by the Router
        and therefore may not be rewritten here.
        """

        logical, execution = self._lookup_execution(execution_id)
        resolved = int(route_version)
        if resolved <= 0:
            raise ProtocolViolation("dispatched route version must be positive")
        if resolved == execution.route_version:
            return
        if (
            execution.predecessor_execution_id is not None
            or len(logical.journal.entries) != 0
            or execution.state != "active"
            or len(logical.executions) != 1
        ):
            raise ProtocolViolation("controller rewrote a fenced recovery route")
        execution.route_version = resolved
        logical.journal.current_route_version = resolved

    def acknowledge_scheduler_boundary(
        self,
        execution_id: str,
        *,
        response_dp_rank: int,
        response_pp_rank: int,
        response_tp_rank: int,
        physical_recovery_gate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Activate replay at the PP0 result boundary, or fail closed.

        PP0 can emit client output only after the request has completed the
        existing pipeline envelope round-trip.  TP is intentionally represented
        by one Scheduler process in SGLSim, so its ACK is a logical aggregate.
        When supplied by ``OfflineSchedulerCluster``, the optional physical
        gate evidence proves that Scheduler admission followed either complete
        PP/TP ownership adoption or an explicit recompute fallback.  Callers
        without a physical bridge preserve the previous ledger behavior.
        """

        logical, execution = self._lookup_execution(execution_id)
        if execution.assigned_dp_rank is None:
            raise ProtocolViolation("output arrived before route assignment")
        if int(response_dp_rank) != execution.assigned_dp_rank:
            raise ProtocolViolation("output arrived from the wrong DP rank")
        if int(response_pp_rank) != 0 or int(response_tp_rank) != 0:
            raise ProtocolViolation("output requires the PP0/TP-aggregate boundary")
        gate_state: str | None = None
        if physical_recovery_gate is not None:
            if not isinstance(physical_recovery_gate, Mapping):
                raise ProtocolViolation("physical recovery gate must be a mapping")
            gate_state = str(physical_recovery_gate.get("state") or "")
            if gate_state not in {"ADOPTED", "FALLBACK_RECOMPUTE"}:
                raise ProtocolViolation(
                    "physical recovery gate is not terminal at Scheduler output"
                )
            if gate_state == "ADOPTED":
                canonical = physical_recovery_gate.get("canonical_group")
                if (
                    not isinstance(canonical, Mapping)
                    or canonical.get("state") != "COMMITTED"
                    or not bool(canonical.get("fully_adopted", False))
                ):
                    raise ProtocolViolation(
                        "physical recovery gate lacks a fully adopted commit group"
                    )
            elif not str(physical_recovery_gate.get("fallback_reason") or ""):
                raise ProtocolViolation(
                    "physical recovery recompute fallback has no reason"
                )
            self._physical_recovery_gate_observed = True
        if execution.state == "active":
            return self.execution_record(execution_id)
        if execution.state != "recovering" or logical.state != "recovering":
            raise ProtocolViolation("output arrived outside an active/recovering route")
        if not execution.expected_ready_ranks:
            raise ProtocolViolation("replay activation has no expected rank set")
        execution.ready_ranks.update(execution.expected_ready_ranks)
        execution.ready_ack_mode = READY_ACK_MODE
        execution.ready_ack_evidence = {
            "response_dp_rank": int(response_dp_rank),
            "response_pp_rank": int(response_pp_rank),
            "response_tp_rank": int(response_tp_rank),
            "pipeline_boundary": "pp0_client_output_after_pipeline_roundtrip",
            "parallel_boundary": (
                "single_scheduler_logical_pp_tp_aggregate"
                if any("pp_aggregate" in key for key in execution.expected_ready_ranks)
                else "physical_pp_roundtrip_logical_tp_aggregate"
            ),
            "physical_recovery_job_gate_in_scope": physical_recovery_gate
            is not None,
            "physical_recovery_gate_state": gate_state,
            "physical_recovery_gate": (
                None
                if physical_recovery_gate is None
                else dict(physical_recovery_gate)
            ),
        }
        if execution.ready_ranks != set(execution.expected_ready_ranks):
            raise ProtocolViolation("not all replay ranks acknowledged readiness")
        execution.state = "active"
        logical.state = "active"
        return self.execution_record(execution_id)

    def observe_scheduler_message(
        self,
        message: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit/publish all newly reported synthetic tokens before forwarding."""

        execution_id = str(message.get("rid") or "")
        logical, execution = self._lookup_execution(execution_id)
        if (
            execution_id != logical.current_execution_id
            or execution.route_version != logical.journal.current_route_version
            or execution.state != "active"
        ):
            rejection = {
                "execution_id": execution_id,
                "reported_generated_tokens": int(message.get("generated_tokens", 0)),
                "current_execution_id": logical.current_execution_id,
                "current_route_version": logical.journal.current_route_version,
                "reason": "stale_or_not_activated_output_fence",
            }
            logical.stale_output_rejections.append(rejection)
            raise ProtocolViolation("stale worker output fence")
        reported = int(message.get("generated_tokens", 0))
        if reported < execution.observed_generated_tokens:
            raise ProtocolViolation("Scheduler generated-token count regressed")
        previous_observed = execution.observed_generated_tokens
        absolute = execution.base_output_sequence + reported
        if reported > execution.requested_new_tokens or absolute > logical.output_budget:
            raise ProtocolViolation("Scheduler output exceeds logical output budget")
        for generated_index in range(execution.observed_generated_tokens + 1, reported + 1):
            sequence = execution.base_output_sequence + generated_index
            state_payload = _canonical_bytes(
                {
                    "schema": CONTINUATION_SCHEMA,
                    "evidence_class": CONTINUATION_EVIDENCE_CLASS,
                    "logical_request_id": logical.logical_request_id,
                    "worker_execution": execution.execution_id,
                    "route_version": execution.route_version,
                    "committed_output_sequence": sequence,
                    "synthetic_token_id": logical.synthetic_output_token_id,
                    "next_sampling_position": logical.prompt_token_count + sequence,
                }
            )
            logical.journal.commit_output(
                worker_execution=execution.execution_id,
                route_version=execution.route_version,
                sequence=sequence,
                token_id=logical.synthetic_output_token_id,
                next_sampling_position=logical.prompt_token_count + sequence,
                durable_next_state_payload=state_payload,
            )
            # Parent publication is synchronous: no detokenizer-visible state
            # exists between this durable commit and advancing the outbox.
            logical.journal.publish_next(sequence)
            execution.committed_sequences.append(sequence)
        execution.observed_generated_tokens = reported
        terminal = str(message.get("op") or "") == "finalize"
        if terminal:
            if str(message.get("status") or "") != "completed":
                raise ProtocolViolation("strict Router finalize is not completed")
            if absolute != logical.output_budget:
                raise ProtocolViolation("completed execution violates logical output budget")
            execution.state = "completed"
            logical.state = "completed"
        return {
            "router_ledger_schema": ROUTER_LEDGER_SCHEMA,
            "router_logical_request_id": logical.logical_request_id,
            "router_execution_id": execution.execution_id,
            "router_route_version": execution.route_version,
            "router_execution_base_output_sequence": execution.base_output_sequence,
            "router_newly_published_sequence_start": (
                execution.base_output_sequence + previous_observed + 1
                if reported > previous_observed
                else None
            ),
            "router_newly_published_sequence_end": (
                absolute if reported > previous_observed else None
            ),
            "router_newly_published_token_count": reported - previous_observed,
            "router_committed_output_position": len(logical.journal.entries),
            "router_published_output_position": (
                logical.journal.last_externally_published_sequence
            ),
            "router_output_budget": logical.output_budget,
            "router_continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
        }

    def continuation_state(self, logical_request_id: str) -> DurableContinuationState:
        logical = self._logical[str(logical_request_id)]
        committed_tokens = [entry.token_id for entry in logical.journal.entries]
        prefix_hash = _digest(
            "sglsim.offline.continuation-prefix.v1",
            {
                "prompt_token_ids": list(logical.prompt_token_ids),
                "committed_output_token_ids": committed_tokens,
                "prompt_identity_mode": logical.prompt_identity_mode,
            },
        )
        return DurableContinuationState(
            schema_version=CONTINUATION_SCHEMA,
            state_digest=logical.journal.current_state_digest,
            compatibility_hash=logical.compatibility_hash,
            prefix_hash=prefix_hash,
            next_output_sequence=len(logical.journal.entries) + 1,
            next_sampling_position=logical.journal.next_sampling_position,
            output_log_tail_digest=logical.journal.tail_digest,
            sampling_semantics=(
                "sglsim-synthetic-position-v1;not-real-logits-or-rng"
            ),
        )

    def fence_execution(self, execution_id: str, *, event_id: str) -> dict[str, Any]:
        """Freeze the exact committed/published boundary for one failure."""

        logical, execution = self._lookup_execution(execution_id)
        committed = len(logical.journal.entries)
        published = logical.journal.last_externally_published_sequence
        if committed != published:
            raise ProtocolViolation("offline synchronous outbox has an unpublished gap")
        continuation = self.continuation_state(logical.logical_request_id)
        remaining = logical.output_budget - committed
        replay_required = remaining > 0
        if execution.state == "completed":
            if replay_required:
                raise ProtocolViolation("completed execution has missing logical outputs")
        elif execution.state not in {"active", "recovering"}:
            raise ProtocolViolation("execution cannot be fenced twice")
        if replay_required:
            execution.state = "fenced"
            logical.state = "fenced"
        boundary = {
            "schema": "sglsim.offline.router-failure-boundary.v1",
            "event_id": str(event_id),
            "logical_request_id": logical.logical_request_id,
            "failed_execution_id": execution.execution_id,
            "failed_route_version": execution.route_version,
            "next_route_version": execution.route_version + 1,
            "committed_output_position": committed,
            "published_output_position": published,
            "original_prompt_tokens": logical.prompt_token_count,
            "original_output_budget": logical.output_budget,
            "pre_failure_total_tokens": logical.prompt_token_count + committed,
            "remaining_max_new_tokens": remaining,
            "replay_required": replay_required,
            "continuation_schema_version": continuation.schema_version,
            "continuation_state_digest": continuation.state_digest,
            "continuation_compatibility_hash": continuation.compatibility_hash,
            "continuation_prefix_hash": continuation.prefix_hash,
            "output_log_tail_digest": continuation.output_log_tail_digest,
            "continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
        }
        execution.failure_event_id = str(event_id)
        execution.failure_committed_output_position = committed
        execution.failure_continuation_digest = continuation.state_digest
        logical.failures.append(dict(boundary))
        return boundary

    def replay_spec(self, execution_id: str, *, event_id: str) -> dict[str, Any]:
        logical, execution = self._lookup_execution(execution_id)
        matches = [
            item
            for item in logical.failures
            if item["event_id"] == str(event_id)
            and item["failed_execution_id"] == execution.execution_id
        ]
        if len(matches) != 1:
            raise ProtocolViolation("failure has no unique Router replay boundary")
        boundary = matches[0]
        committed_ids = [entry.token_id for entry in logical.journal.entries]
        committed = int(boundary["committed_output_position"])
        if len(committed_ids) != committed:
            raise ProtocolViolation("Router output advanced after execution fence")
        return {
            **dict(boundary),
            "replay_contract_schema": REPLAY_CONTRACT_SCHEMA,
            "replay_input_ids": list(logical.prompt_token_ids) + committed_ids,
            "replay_input_token_count": logical.prompt_token_count + committed,
            "synthetic_output_token_id": logical.synthetic_output_token_id,
            "routing_key": logical.routing_key,
            "prompt_identity_mode": logical.prompt_identity_mode,
        }

    def execution_record(self, execution_id: str) -> dict[str, Any]:
        logical, execution = self._lookup_execution(execution_id)
        return {
            "execution_id": execution.execution_id,
            "logical_request_id": logical.logical_request_id,
            "route_version": execution.route_version,
            "predecessor_execution_id": execution.predecessor_execution_id,
            "base_output_sequence": execution.base_output_sequence,
            "requested_new_tokens": execution.requested_new_tokens,
            "observed_generated_tokens": execution.observed_generated_tokens,
            "strict_replay_contract": execution.strict_replay_contract,
            "state": execution.state,
            "assigned_dp_rank": execution.assigned_dp_rank,
            "expected_ready_ranks": list(execution.expected_ready_ranks),
            "ready_ranks": sorted(execution.ready_ranks),
            "ready_ack_mode": execution.ready_ack_mode,
            "ready_ack_evidence": dict(execution.ready_ack_evidence),
            "committed_sequences": list(execution.committed_sequences),
            "failure_event_id": execution.failure_event_id,
            "failure_committed_output_position": (
                execution.failure_committed_output_position
            ),
            "failure_continuation_digest": execution.failure_continuation_digest,
        }

    def audit(self, *, require_complete: bool) -> dict[str, Any]:
        errors: list[str] = []
        logical_rows: list[dict[str, Any]] = []
        for logical_id in sorted(self._logical):
            logical = self._logical[logical_id]
            entries = logical.journal.entries
            sequences = [entry.sequence for entry in entries]
            expected_sequences = list(range(1, len(entries) + 1))
            if sequences != expected_sequences:
                errors.append(f"{logical_id}: durable output sequence has a gap/duplicate")
            if logical.journal.last_externally_published_sequence != len(entries):
                errors.append(f"{logical_id}: committed/published output positions differ")
            route_versions = [item.route_version for item in logical.executions]
            if route_versions != list(
                range(route_versions[0], route_versions[0] + len(route_versions))
            ):
                errors.append(f"{logical_id}: route versions are not contiguous")
            owned_sequences = [
                sequence
                for execution in logical.executions
                for sequence in execution.committed_sequences
            ]
            if owned_sequences != expected_sequences:
                errors.append(f"{logical_id}: execution output ownership is not conserved")
            for index, execution in enumerate(logical.executions[1:], 1):
                predecessor = logical.executions[index - 1]
                if not execution.strict_replay_contract:
                    errors.append(f"{logical_id}: replay lacks strict metadata contract")
                if execution.predecessor_execution_id != predecessor.execution_id:
                    errors.append(f"{logical_id}: replay predecessor is discontinuous")
                if execution.base_output_sequence != (
                    predecessor.failure_committed_output_position
                ):
                    errors.append(f"{logical_id}: replay starts outside failure fence")
                if execution.requested_new_tokens != (
                    logical.output_budget - execution.base_output_sequence
                ):
                    errors.append(f"{logical_id}: replay budget is not N-k")
                if (
                    execution.committed_sequences
                    and set(execution.expected_ready_ranks) != execution.ready_ranks
                ):
                    errors.append(f"{logical_id}: replay activated without all-rank ACK")
            if require_complete and logical.state != "completed":
                errors.append(f"{logical_id}: logical request is not completed")
            if logical.state == "completed" and len(entries) != logical.output_budget:
                errors.append(f"{logical_id}: completed logical output is not conserved")
            logical_rows.append(
                {
                    "logical_request_id": logical_id,
                    "state": logical.state,
                    "output_budget": logical.output_budget,
                    "committed_output_tokens": len(entries),
                    "published_output_tokens": (
                        logical.journal.last_externally_published_sequence
                    ),
                    "route_versions": route_versions,
                    "execution_count": len(logical.executions),
                    "stale_output_rejection_count": len(
                        logical.stale_output_rejections
                    ),
                }
            )
        result = {
            "schema": "sglsim.offline.router-output-audit.v1",
            "status": "PASS" if not errors else "FAIL",
            "require_complete": bool(require_complete),
            "logical_request_count": len(self._logical),
            "logical_requests": logical_rows,
            "errors": errors,
            "continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
        }
        if errors:
            raise ProtocolViolation("offline Router output audit failed: " + "; ".join(errors))
        return result

    def export(self) -> dict[str, Any]:
        logical_rows: list[dict[str, Any]] = []
        for logical_id in sorted(self._logical):
            logical = self._logical[logical_id]
            continuation = self.continuation_state(logical_id)
            logical_rows.append(
                {
                    "logical_request_id": logical.logical_request_id,
                    "state": logical.state,
                    "prompt_token_count": logical.prompt_token_count,
                    "prompt_identity_mode": logical.prompt_identity_mode,
                    "output_budget": logical.output_budget,
                    "synthetic_output_token_id": logical.synthetic_output_token_id,
                    "compatibility_hash": logical.compatibility_hash,
                    "input_manifest_digest": logical.journal.input_manifest_digest,
                    "current_execution_id": logical.current_execution_id,
                    "current_route_version": logical.journal.current_route_version,
                    "committed_output_position": len(logical.journal.entries),
                    "published_output_position": (
                        logical.journal.last_externally_published_sequence
                    ),
                    "output_entries": [
                        {
                            "sequence": entry.sequence,
                            "token_id": entry.token_id,
                            "next_sampling_position": entry.next_sampling_position,
                            "previous_entry_digest": entry.previous_entry_digest,
                            "next_state_digest": entry.next_state_digest,
                            "entry_digest": entry.entry_digest,
                        }
                        for entry in logical.journal.entries
                    ],
                    "continuation": {
                        "schema_version": continuation.schema_version,
                        "state_digest": continuation.state_digest,
                        "prefix_hash": continuation.prefix_hash,
                        "next_output_sequence": continuation.next_output_sequence,
                        "next_sampling_position": continuation.next_sampling_position,
                        "output_log_tail_digest": continuation.output_log_tail_digest,
                        "sampling_semantics": continuation.sampling_semantics,
                    },
                    "executions": [
                        self.execution_record(item.execution_id)
                        for item in logical.executions
                    ],
                    "failures": [dict(item) for item in logical.failures],
                    "stale_output_rejections": [
                        dict(item) for item in logical.stale_output_rejections
                    ],
                }
            )
        return {
            "schema": ROUTER_LEDGER_SCHEMA,
            "continuation_schema": CONTINUATION_SCHEMA,
            "replay_contract_schema": REPLAY_CONTRACT_SCHEMA,
            "continuation_evidence_class": CONTINUATION_EVIDENCE_CLASS,
            "ready_ack_mode": READY_ACK_MODE,
            "physical_recovery_job_activation_gate_in_scope": bool(
                self._physical_recovery_gate_observed
            ),
            "logical_requests": logical_rows,
        }


def execution_index(export: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a copied execution index for artifact enrichment."""

    indexed: dict[str, dict[str, Any]] = {}
    for logical in export.get("logical_requests", ()):
        if not isinstance(logical, Mapping):
            continue
        for execution in logical.get("executions", ()):
            if not isinstance(execution, Mapping):
                continue
            rid = str(execution.get("execution_id") or "")
            if rid:
                indexed[rid] = dict(execution)
    return indexed


def logical_rows(export: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Typed iteration helper used by strict suite audits."""

    for item in export.get("logical_requests", ()):
        if isinstance(item, Mapping):
            yield item
