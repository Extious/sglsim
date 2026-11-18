"""Build auditable finite rolling-window instances for AegisKV.

This module is offline-only input construction.  The runtime consumes only
``CertifiedExactWindowPlan`` and never imports this module or SciPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from sglsim.srt.simulation.aegis_exact_contract import CertifiedExactWindowPlan
from sglsim.srt.simulation.aegis_exact_optimizer import (
    DEFAULT_ADMISSION_STATE,
    DEFAULT_ADMISSION_STATE_ID,
    ExactAdmissionState,
    ExactChunk,
    ExactObjectiveWeights,
    ExactOptimizationInstance,
    ExactOptimizationResult,
    ExactScenario,
    ExactTopology,
    build_certified_exact_window_plan,
    solve_exact_placement,
)


@dataclass(frozen=True)
class ExactWindowRequest:
    request_id: str
    workflow_id: str
    source_dp_rank: int
    token_count: int
    slack_s: float
    critical_weight: float
    admission_time_s: float = 0.0
    admission_state_id: str = DEFAULT_ADMISSION_STATE_ID

    def __post_init__(self) -> None:
        if not self.request_id or not self.workflow_id:
            raise ValueError("exact-window request identity must be nonempty")
        if self.source_dp_rank < 0 or self.token_count <= 0:
            raise ValueError("exact-window request route/tokens are invalid")
        if self.slack_s < 0.0:
            raise ValueError("exact-window request slack must be nonnegative")
        if not 0.0 <= self.critical_weight <= 1.0:
            raise ValueError("critical_weight must be in [0, 1]")
        if not math.isfinite(self.admission_time_s) or self.admission_time_s < 0.0:
            raise ValueError("admission_time_s must be finite and nonnegative")
        if not self.admission_state_id:
            raise ValueError("admission_state_id must be nonempty")


@dataclass(frozen=True)
class ExactRollingState:
    """Causal cross-window stock and already-published Algorithm-3 budget."""

    logical_time_s: float
    window_id: int
    route_version: int
    unavailable_dp_ranks: tuple[int, ...]
    resident_hbm_tokens: tuple[tuple[int, ...], ...]
    resident_host_tokens_by_dp: tuple[int, ...]
    resident_host_tokens_by_node: tuple[tuple[int, int], ...]
    used_link_bandwidth_bytes: tuple[tuple[int, int, int], ...]
    current_hbm_budget_tokens: tuple[int, ...]
    current_host_budget_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.logical_time_s) or self.logical_time_s < 0.0:
            raise ValueError("rolling logical_time_s must be finite/nonnegative")
        if self.window_id < 0 or self.route_version <= 0:
            raise ValueError("rolling window/route versions are invalid")
        if len(set(self.unavailable_dp_ranks)) != len(
            self.unavailable_dp_ranks
        ):
            raise ValueError("rolling unavailable DP ranks must be unique")


@dataclass(frozen=True)
class CertifiedCausalRollingSolve:
    instance: ExactOptimizationInstance
    result: ExactOptimizationResult
    plan: CertifiedExactWindowPlan
    replay_decision_sha256: str
    replay_objective: float


def solve_causal_exact_rolling_window(
    *,
    instance_id: str,
    requests: tuple[ExactWindowRequest, ...],
    rolling_state: ExactRollingState,
    topology: ExactTopology,
    weights: ExactObjectiveWeights,
    failure_hazard_probability: float,
    failure_scope: str,
    chunk_tokens: int,
    recompute_setup_s: float,
    recompute_s_per_token: float,
    protected_restore_setup_s: float,
    protected_restore_s_per_token: float,
    time_limit_s: float,
    deterministic_seed: int,
    deterministic_replay: bool = True,
) -> CertifiedCausalRollingSolve:
    """Certify one state-matched finite rolling window.

    Only state already observed at ``logical_time_s`` is admitted here.  The
    realized future failure manifest is not an input.  Both alpha and beta are
    fixed to the currently ACKed integer budgets, so the resulting plan can
    atomically advance Router/Connector versions without publishing a new
    Algorithm-3 budget at every request arrival.
    """

    if not requests:
        raise ValueError("causal rolling solve requires requests")
    if len(rolling_state.current_hbm_budget_tokens) != topology.dp_size:
        raise ValueError("rolling HBM budgets do not match topology")
    if len(rolling_state.current_host_budget_tokens) != topology.dp_size:
        raise ValueError("rolling Host budgets do not match topology")
    state_id = (
        f"causal-window-{rolling_state.window_id}-unavailable-"
        + (
            "none"
            if not rolling_state.unavailable_dp_ranks
            else "-".join(str(rank) for rank in rolling_state.unavailable_dp_ranks)
        )
    )
    admission_state = ExactAdmissionState(
        state_id=state_id,
        valid_from_logical_time_s=rolling_state.logical_time_s,
        valid_until_logical_time_s=None,
        unavailable_dp_ranks=tuple(sorted(rolling_state.unavailable_dp_ranks)),
        information_mode="causal_observed",
    )
    causal_requests = tuple(
        replace(
            request,
            admission_time_s=rolling_state.logical_time_s,
            admission_state_id=state_id,
        )
        for request in requests
    )
    instance = build_bootstrap_exact_instance(
        instance_id=instance_id,
        requests=causal_requests,
        topology=topology,
        weights=weights,
        failure_hazard_probability=failure_hazard_probability,
        failure_scope=failure_scope,
        chunk_tokens=chunk_tokens,
        recompute_setup_s=recompute_setup_s,
        recompute_s_per_token=recompute_s_per_token,
        protected_restore_setup_s=protected_restore_setup_s,
        protected_restore_s_per_token=protected_restore_s_per_token,
        previous_hbm_budget_tokens=(
            rolling_state.current_hbm_budget_tokens
        ),
        previous_host_budget_tokens=(
            rolling_state.current_host_budget_tokens
        ),
        time_limit_s=time_limit_s,
        deterministic_seed=deterministic_seed,
        admission_states=(admission_state,),
        resident_hbm_tokens=rolling_state.resident_hbm_tokens,
        resident_host_tokens_by_dp=rolling_state.resident_host_tokens_by_dp,
        resident_host_tokens_by_node=rolling_state.resident_host_tokens_by_node,
        used_link_bandwidth_bytes=rolling_state.used_link_bandwidth_bytes,
        fixed_hbm_budget_tokens=rolling_state.current_hbm_budget_tokens,
        fixed_host_budget_tokens=rolling_state.current_host_budget_tokens,
    )
    result = solve_exact_placement(instance).require_global_optimum()
    replay = (
        solve_exact_placement(instance).require_global_optimum()
        if deterministic_replay
        else result
    )
    if (
        result.certificate.decision_sha256
        != replay.certificate.decision_sha256
        or result.certificate.objective != replay.certificate.objective
    ):
        raise RuntimeError(
            "causal rolling exact replay changed decision digest/objective"
        )
    if replay.certificate.decision_sha256 is None:
        raise RuntimeError("causal rolling replay has no decision digest")
    plan = build_certified_exact_window_plan(
        instance,
        result,
        window_id=rolling_state.window_id,
        route_version=rolling_state.route_version,
        valid_from_logical_time=rolling_state.logical_time_s,
    )
    if plan.budget_publication_mode != "reuse_current_fixed":
        raise RuntimeError("causal route-only solve did not freeze current budgets")
    return CertifiedCausalRollingSolve(
        instance=instance,
        result=result,
        plan=plan,
        replay_decision_sha256=replay.certificate.decision_sha256,
        replay_objective=float(replay.certificate.objective),
    )


def _source_failure_domain(
    topology: ExactTopology,
    source_dp_rank: int,
    *,
    failure_scope: str,
) -> str:
    if failure_scope == "node":
        return topology.host_failure_domain(
            topology.dp_to_node[source_dp_rank]
        )
    if failure_scope == "dp":
        return topology.hbm_failure_domains[source_dp_rank][0]
    raise ValueError("failure_scope must be 'node' or 'dp'")


def build_bootstrap_exact_instance(
    *,
    instance_id: str,
    requests: tuple[ExactWindowRequest, ...],
    topology: ExactTopology,
    weights: ExactObjectiveWeights,
    failure_hazard_probability: float,
    failure_scope: str,
    chunk_tokens: int,
    recompute_setup_s: float,
    recompute_s_per_token: float,
    protected_restore_setup_s: float,
    protected_restore_s_per_token: float,
    previous_hbm_budget_tokens: tuple[int, ...],
    previous_host_budget_tokens: tuple[int, ...],
    time_limit_s: float,
    deterministic_seed: int,
    admission_states: tuple[ExactAdmissionState, ...] = (
        DEFAULT_ADMISSION_STATE,
    ),
    resident_hbm_tokens: tuple[tuple[int, ...], ...] = (),
    resident_host_tokens_by_dp: tuple[int, ...] = (),
    resident_host_tokens_by_node: tuple[tuple[int, int], ...] = (),
    used_link_bandwidth_bytes: tuple[tuple[int, int, int], ...] = (),
    fixed_hbm_budget_tokens: tuple[int, ...] | None = None,
    fixed_host_budget_tokens: tuple[int, ...] | None = None,
) -> ExactOptimizationInstance:
    """Construct a zero-residency finite window from frozen trace requests.

    ``failure_hazard_probability`` is the Algorithm 2 scenario prior.  It is
    intentionally independent of the benchmark's realized injection rate and
    of ``admission_states``.  Admission state contains only availability known
    when Router freezes the request route.  A state labeled
    ``future_manifest_oracle`` is diagnostic-only and must not be deployed.
    The healthy scenario plus one scenario per distinct source failure domain
    form a finite empirical distribution.
    """

    if not requests:
        raise ValueError("bootstrap exact window must contain requests")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("bootstrap exact window request IDs must be unique")
    if not admission_states:
        raise ValueError("bootstrap exact window admission_states must be nonempty")
    admission_state_map = {
        state.state_id: state for state in admission_states
    }
    if len(admission_state_map) != len(admission_states):
        raise ValueError("bootstrap admission state IDs must be unique")
    if not 0.0 <= failure_hazard_probability <= 1.0:
        raise ValueError("failure hazard probability must be in [0, 1]")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    for value in (
        recompute_setup_s,
        recompute_s_per_token,
        protected_restore_setup_s,
        protected_restore_s_per_token,
    ):
        if value < 0.0:
            raise ValueError("recovery cost coefficients must be nonnegative")

    workflows = tuple(sorted({request.workflow_id for request in requests}))
    source_domains = tuple(
        sorted(
            {
                _source_failure_domain(
                    topology,
                    request.source_dp_rank,
                    failure_scope=failure_scope,
                )
                for request in requests
            }
        )
    )
    failure_probability = (
        failure_hazard_probability / len(source_domains)
        if source_domains
        else 0.0
    )
    zero_workflow_loss = tuple((workflow_id, 0.0) for workflow_id in workflows)
    scenarios = [
        ExactScenario(
            scenario_id="healthy",
            probability=1.0 - failure_hazard_probability,
            failed_domains=(),
            base_workflow_loss_s=zero_workflow_loss,
        )
    ]
    scenarios.extend(
        ExactScenario(
            scenario_id=f"failure:{domain}",
            probability=failure_probability,
            failed_domains=(domain,),
            base_workflow_loss_s=zero_workflow_loss,
        )
        for domain in source_domains
    )

    chunks: list[ExactChunk] = []
    for request in sorted(requests, key=lambda value: value.request_id):
        try:
            admission_state = admission_state_map[request.admission_state_id]
        except KeyError as exc:
            raise ValueError(
                f"request {request.request_id} references unknown admission state "
                f"{request.admission_state_id!r}"
            ) from exc
        if not admission_state.contains(request.admission_time_s):
            raise ValueError(
                f"request {request.request_id} admission time lies outside state "
                f"{request.admission_state_id!r}"
            )
        unavailable = set(admission_state.unavailable_dp_ranks)
        if request.source_dp_rank in unavailable:
            raise ValueError(
                f"request {request.request_id} execution DP is unavailable at "
                "route admission"
            )
        source_domain = _source_failure_domain(
            topology,
            request.source_dp_rank,
            failure_scope=failure_scope,
        )
        candidates = tuple(
            dp_rank
            for dp_rank in range(topology.dp_size)
            if dp_rank != request.source_dp_rank
            and (
                failure_scope != "node"
                or topology.host_failure_domain(topology.dp_to_node[dp_rank])
                != source_domain
            )
        )
        if not candidates:
            raise ValueError(
                f"request {request.request_id} has no failure-independent backup DP"
            )
        if not any(candidate not in unavailable for candidate in candidates):
            raise ValueError(
                f"request {request.request_id} has no currently available, "
                "failure-independent backup DP"
            )

        total_unprotected = request.critical_weight * max(
            recompute_setup_s
            + request.token_count * recompute_s_per_token
            - request.slack_s,
            0.0,
        )
        total_protected = request.critical_weight * max(
            protected_restore_setup_s
            + request.token_count * protected_restore_s_per_token
            - request.slack_s,
            0.0,
        )
        # Preserve additivity while applying setup/slack once per request.
        chunk_count = int(math.ceil(request.token_count / chunk_tokens))
        remaining = request.token_count
        for chunk_id in range(chunk_count):
            tokens = min(chunk_tokens, remaining)
            remaining -= tokens
            fraction = tokens / request.token_count
            unprotected_losses: list[tuple[str, float]] = []
            protected_losses: list[tuple[str, float]] = []
            for scenario in scenarios:
                source_failed = source_domain in scenario.failed_domains
                unprotected_losses.append(
                    (
                        scenario.scenario_id,
                        total_unprotected * fraction if source_failed else 0.0,
                    )
                )
                protected_losses.append(
                    (
                        scenario.scenario_id,
                        total_protected * fraction if source_failed else 0.0,
                    )
                )
            chunks.append(
                ExactChunk(
                    request_id=request.request_id,
                    workflow_id=request.workflow_id,
                    chunk_id=chunk_id,
                    source_dp_rank=request.source_dp_rank,
                    token_count=tokens,
                    backup_candidates=candidates,
                    unprotected_extra_loss_s=tuple(unprotected_losses),
                    protected_extra_loss_s=tuple(protected_losses),
                    admission_time_s=request.admission_time_s,
                    admission_state_id=request.admission_state_id,
                )
            )

    return ExactOptimizationInstance(
        instance_id=instance_id,
        topology=topology,
        scenarios=tuple(scenarios),
        chunks=tuple(chunks),
        weights=weights,
        previous_hbm_budget_tokens=previous_hbm_budget_tokens,
        previous_host_budget_tokens=previous_host_budget_tokens,
        admission_states=admission_states,
        resident_hbm_tokens=resident_hbm_tokens,
        resident_host_tokens_by_dp=resident_host_tokens_by_dp,
        resident_host_tokens_by_node=resident_host_tokens_by_node,
        used_link_bandwidth_bytes=used_link_bandwidth_bytes,
        fixed_hbm_budget_tokens=fixed_hbm_budget_tokens,
        fixed_host_budget_tokens=fixed_host_budget_tokens,
        time_limit_s=time_limit_s,
        deterministic_seed=deterministic_seed,
    )
