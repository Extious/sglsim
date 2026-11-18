"""Certified finite-horizon MILP foundation for AegisKV Algorithms 2/3.

The solver in this module is offline-only.  It chooses one backup DP per
request, a stage-local HBM/Host/Recompute outcome for every chunk, and one
DP-owned HBM/Host policy budget.  Scenario-specific recoverability is an exact
all-PP-stage AND with a contiguous-prefix constraint.

Only a HiGHS OPTIMAL result with zero reported gap and an independently
validated solution is exposed as ``certified_optimal``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from sglsim.srt.simulation.aegis_exact_contract import (
    CertifiedDPBudgetPlan,
    CertifiedExactWindowPlan,
    CertifiedRoutePlan,
    CertifiedStageChunkPlacement,
)


TIERS = ("HBM", "HOST", "RECOMPUTE")
ADMISSION_INFORMATION_MODES = (
    "causal_observed",
    "topology_robust",
    "future_manifest_oracle",
)
DEFAULT_ADMISSION_STATE_ID = "all_available"


def _pairs_to_dict(pairs: Iterable[tuple[object, object]]) -> dict:
    return dict(pairs)


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExactScenario:
    scenario_id: str
    probability: float
    failed_domains: tuple[str, ...]
    base_workflow_loss_s: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be nonempty")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("scenario probability must be in [0, 1]")
        if len(set(self.failed_domains)) != len(self.failed_domains):
            raise ValueError("failed_domains must be unique")
        losses = _pairs_to_dict(self.base_workflow_loss_s)
        if len(losses) != len(self.base_workflow_loss_s):
            raise ValueError("base_workflow_loss_s has duplicate workflows")
        if any(float(value) < 0.0 for value in losses.values()):
            raise ValueError("base workflow losses must be nonnegative")


@dataclass(frozen=True)
class ExactAdmissionState:
    """Availability known when Router freezes ``e(r), b(r)``.

    This is intentionally separate from :class:`ExactScenario`.  Scenarios are
    the Algorithm 2 hazard prior used by the expected-loss/CVaR objective;
    admission state is information already observed by the causal Router.  A
    state derived from a realized future manifest must carry the explicit
    ``future_manifest_oracle`` label and is ineligible for deployed evidence.
    """

    state_id: str
    valid_from_logical_time_s: float
    valid_until_logical_time_s: float | None
    unavailable_dp_ranks: tuple[int, ...]
    information_mode: str = "causal_observed"

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("admission state_id must be nonempty")
        if not math.isfinite(self.valid_from_logical_time_s):
            raise ValueError("admission state start must be finite")
        if self.valid_from_logical_time_s < 0.0:
            raise ValueError("admission state start must be nonnegative")
        if self.valid_until_logical_time_s is not None:
            if not math.isfinite(self.valid_until_logical_time_s):
                raise ValueError("admission state end must be finite when set")
            if self.valid_until_logical_time_s <= self.valid_from_logical_time_s:
                raise ValueError("admission state interval must have positive width")
        if len(set(self.unavailable_dp_ranks)) != len(
            self.unavailable_dp_ranks
        ):
            raise ValueError("unavailable DP ranks must be unique")
        if any(rank < 0 for rank in self.unavailable_dp_ranks):
            raise ValueError("unavailable DP ranks must be nonnegative")
        if self.information_mode not in ADMISSION_INFORMATION_MODES:
            raise ValueError(
                "admission information_mode must be one of "
                f"{ADMISSION_INFORMATION_MODES}"
            )

    def contains(self, logical_time_s: float) -> bool:
        value = float(logical_time_s)
        return value >= self.valid_from_logical_time_s and (
            self.valid_until_logical_time_s is None
            or value < self.valid_until_logical_time_s
        )


DEFAULT_ADMISSION_STATE = ExactAdmissionState(
    state_id=DEFAULT_ADMISSION_STATE_ID,
    valid_from_logical_time_s=0.0,
    valid_until_logical_time_s=None,
    unavailable_dp_ranks=(),
    information_mode="causal_observed",
)


@dataclass(frozen=True)
class ExactChunk:
    request_id: str
    workflow_id: str
    chunk_id: int
    source_dp_rank: int
    token_count: int
    backup_candidates: tuple[int, ...]
    unprotected_extra_loss_s: tuple[tuple[str, float], ...]
    protected_extra_loss_s: tuple[tuple[str, float], ...]
    admission_time_s: float = 0.0
    admission_state_id: str = DEFAULT_ADMISSION_STATE_ID

    def __post_init__(self) -> None:
        if not self.request_id or not self.workflow_id:
            raise ValueError("request_id and workflow_id must be nonempty")
        if self.chunk_id < 0:
            raise ValueError("chunk_id must be nonnegative")
        if self.source_dp_rank < 0:
            raise ValueError("source_dp_rank must be nonnegative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not self.backup_candidates:
            raise ValueError("backup_candidates must be nonempty")
        if len(set(self.backup_candidates)) != len(self.backup_candidates):
            raise ValueError("backup_candidates must be unique")
        if self.source_dp_rank in self.backup_candidates:
            raise ValueError("source DP cannot be a backup candidate")
        if not math.isfinite(self.admission_time_s) or self.admission_time_s < 0.0:
            raise ValueError("chunk admission_time_s must be finite and nonnegative")
        if not self.admission_state_id:
            raise ValueError("chunk admission_state_id must be nonempty")
        unprotected = _pairs_to_dict(self.unprotected_extra_loss_s)
        protected = _pairs_to_dict(self.protected_extra_loss_s)
        if len(unprotected) != len(self.unprotected_extra_loss_s):
            raise ValueError("duplicate unprotected scenario loss")
        if len(protected) != len(self.protected_extra_loss_s):
            raise ValueError("duplicate protected scenario loss")
        if set(unprotected) != set(protected):
            raise ValueError("protected/unprotected scenarios must match")
        for scenario_id in unprotected:
            raw = float(unprotected[scenario_id])
            covered = float(protected[scenario_id])
            if raw < 0.0 or covered < 0.0:
                raise ValueError("scenario losses must be nonnegative")
            if covered > raw:
                raise ValueError(
                    "protected scenario loss cannot exceed unprotected loss"
                )


@dataclass(frozen=True)
class ExactTopology:
    """Finite physical capacities and failure domains.

    Tuple positions are DP ranks unless a key is explicitly present.  HBM
    capacities/domains are stage-local.  Host policy capacity is DP-owned while
    node Host capacity is physical and shared.
    """

    pp_size: int
    dp_to_node: tuple[int, ...]
    hbm_capacity_tokens: tuple[tuple[int, ...], ...]
    host_policy_max_tokens: tuple[int, ...]
    node_host_capacity_tokens: tuple[tuple[int, int], ...]
    link_bandwidth_bytes: tuple[tuple[int, int, int], ...]
    hbm_failure_domains: tuple[tuple[str, ...], ...]
    host_failure_domains: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if self.pp_size <= 0:
            raise ValueError("pp_size must be positive")
        dp_size = len(self.dp_to_node)
        if dp_size == 0:
            raise ValueError("dp_to_node must be nonempty")
        if len(self.hbm_capacity_tokens) != dp_size:
            raise ValueError("hbm capacity must have one row per DP")
        if len(self.host_policy_max_tokens) != dp_size:
            raise ValueError("host policy capacity must have one value per DP")
        if len(self.hbm_failure_domains) != dp_size:
            raise ValueError("HBM failure domains must have one row per DP")
        for dp_rank in range(dp_size):
            if len(self.hbm_capacity_tokens[dp_rank]) != self.pp_size:
                raise ValueError("HBM capacity row does not match pp_size")
            if len(self.hbm_failure_domains[dp_rank]) != self.pp_size:
                raise ValueError("HBM failure-domain row does not match pp_size")
            if any(value < 0 for value in self.hbm_capacity_tokens[dp_rank]):
                raise ValueError("HBM capacities must be nonnegative")
            if self.host_policy_max_tokens[dp_rank] < 0:
                raise ValueError("Host policy capacities must be nonnegative")
            if any(not value for value in self.hbm_failure_domains[dp_rank]):
                raise ValueError("HBM failure domains must be nonempty")
        node_capacity = _pairs_to_dict(self.node_host_capacity_tokens)
        host_domains = _pairs_to_dict(self.host_failure_domains)
        if len(node_capacity) != len(self.node_host_capacity_tokens):
            raise ValueError("duplicate node Host capacity")
        if len(host_domains) != len(self.host_failure_domains):
            raise ValueError("duplicate Host failure domain")
        for node_id in set(self.dp_to_node):
            if node_id not in node_capacity or node_id not in host_domains:
                raise ValueError(f"node {node_id} lacks Host capacity/domain")
        if any(int(value) < 0 for value in node_capacity.values()):
            raise ValueError("node Host capacities must be nonnegative")
        links: set[tuple[int, int]] = set()
        for source, target, capacity in self.link_bandwidth_bytes:
            if (source, target) in links:
                raise ValueError("duplicate directed link capacity")
            if not 0 <= source < dp_size or not 0 <= target < dp_size:
                raise ValueError("link DP rank is outside topology")
            if capacity < 0:
                raise ValueError("link capacity must be nonnegative")
            links.add((source, target))

    @property
    def dp_size(self) -> int:
        return len(self.dp_to_node)

    def node_host_capacity(self, node_id: int) -> int:
        return int(_pairs_to_dict(self.node_host_capacity_tokens)[node_id])

    def host_failure_domain(self, node_id: int) -> str:
        return str(_pairs_to_dict(self.host_failure_domains)[node_id])

    def link_capacity(self, source_dp: int, target_dp: int) -> int:
        links = {
            (int(source), int(target)): int(capacity)
            for source, target, capacity in self.link_bandwidth_bytes
        }
        try:
            return links[(source_dp, target_dp)]
        except KeyError as exc:
            raise ValueError(
                f"missing link capacity for DP{source_dp}->DP{target_dp}"
            ) from exc


@dataclass(frozen=True)
class ExactObjectiveWeights:
    expected_workflow_loss: float = 1.0
    cvar: float = 0.0
    cvar_tau: float = 0.95
    hbm_normal_cost_per_token: float = 0.0
    host_normal_cost_per_token: float = 0.0
    copy_bandwidth_cost_per_byte: float = 0.0
    hbm_budget_cost_per_token: float = 0.0
    host_budget_cost_per_token: float = 0.0
    hbm_switch_cost_per_token: float = 0.0
    host_switch_cost_per_token: float = 0.0
    hbm_copy_bytes_per_token: int = 1
    host_copy_bytes_per_token: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "cvar_tau":
                if not 0.0 < float(value) < 1.0:
                    raise ValueError("cvar_tau must be in (0, 1)")
            elif float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.hbm_copy_bytes_per_token <= 0:
            raise ValueError("hbm_copy_bytes_per_token must be positive")
        if self.host_copy_bytes_per_token <= 0:
            raise ValueError("host_copy_bytes_per_token must be positive")


@dataclass(frozen=True)
class ExactOptimizationInstance:
    instance_id: str
    topology: ExactTopology
    scenarios: tuple[ExactScenario, ...]
    chunks: tuple[ExactChunk, ...]
    weights: ExactObjectiveWeights
    previous_hbm_budget_tokens: tuple[int, ...]
    previous_host_budget_tokens: tuple[int, ...]
    admission_states: tuple[ExactAdmissionState, ...] = (
        DEFAULT_ADMISSION_STATE,
    )
    # Rolling-state stock is distinct from offered/admitted flow.  Empty
    # tuples are a backward-compatible zero-stock representation.
    resident_hbm_tokens: tuple[tuple[int, ...], ...] = ()
    resident_host_tokens_by_dp: tuple[int, ...] = ()
    resident_host_tokens_by_node: tuple[tuple[int, int], ...] = ()
    used_link_bandwidth_bytes: tuple[tuple[int, int, int], ...] = ()
    # A route-only rolling solve may optimize placement under an already
    # published Algorithm-3 budget.  When set, these bounds are equalities.
    fixed_hbm_budget_tokens: tuple[int, ...] | None = None
    fixed_host_budget_tokens: tuple[int, ...] | None = None
    time_limit_s: float = 300.0
    numerical_tolerance: float = 1e-8
    deterministic_seed: int = 0

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must be nonempty")
        if not self.scenarios or not self.chunks:
            raise ValueError("scenarios and chunks must be nonempty")
        if self.time_limit_s <= 0.0 or self.numerical_tolerance <= 0.0:
            raise ValueError("time limit and numerical tolerance must be positive")
        if len(self.previous_hbm_budget_tokens) != self.topology.dp_size:
            raise ValueError("previous HBM budgets must have one value per DP")
        if len(self.previous_host_budget_tokens) != self.topology.dp_size:
            raise ValueError("previous Host budgets must have one value per DP")
        if any(value < 0 for value in self.previous_hbm_budget_tokens):
            raise ValueError("previous HBM budgets must be nonnegative")
        if any(value < 0 for value in self.previous_host_budget_tokens):
            raise ValueError("previous Host budgets must be nonnegative")
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario IDs must be unique")
        if not math.isclose(
            sum(scenario.probability for scenario in self.scenarios),
            1.0,
            rel_tol=0.0,
            abs_tol=self.numerical_tolerance,
        ):
            raise ValueError("scenario probabilities must sum to one")
        scenario_set = set(scenario_ids)
        if not self.admission_states:
            raise ValueError("admission_states must be nonempty")
        admission_state_map = {
            state.state_id: state for state in self.admission_states
        }
        if len(admission_state_map) != len(self.admission_states):
            raise ValueError("admission state IDs must be unique")
        for state in self.admission_states:
            if any(
                rank >= self.topology.dp_size
                for rank in state.unavailable_dp_ranks
            ):
                raise ValueError("admission state DP rank is outside topology")

        if self.resident_hbm_tokens and len(self.resident_hbm_tokens) != (
            self.topology.dp_size
        ):
            raise ValueError("resident HBM stock must have one row per DP")
        for dp_rank, row in enumerate(self.resident_hbm_tokens):
            if len(row) != self.topology.pp_size:
                raise ValueError("resident HBM row does not match pp_size")
            for pp_rank, value in enumerate(row):
                if value < 0:
                    raise ValueError("resident HBM stock must be nonnegative")
                if value > self.topology.hbm_capacity_tokens[dp_rank][pp_rank]:
                    raise ValueError("resident HBM stock exceeds physical capacity")

        if self.resident_host_tokens_by_dp and len(
            self.resident_host_tokens_by_dp
        ) != self.topology.dp_size:
            raise ValueError("resident Host stock must have one value per DP")
        resolved_host_by_dp = self.resolved_resident_host_tokens_by_dp
        for dp_rank, value in enumerate(resolved_host_by_dp):
            if value < 0:
                raise ValueError("resident Host stock must be nonnegative")
            if value > self.topology.host_policy_max_tokens[dp_rank]:
                raise ValueError("resident Host stock exceeds DP policy maximum")

        node_stock = _pairs_to_dict(self.resident_host_tokens_by_node)
        if len(node_stock) != len(self.resident_host_tokens_by_node):
            raise ValueError("resident Host node stock contains duplicate nodes")
        topology_nodes = set(self.topology.dp_to_node)
        if node_stock and set(node_stock) != topology_nodes:
            raise ValueError("resident Host node stock must cover topology nodes")
        resolved_node_stock = self.resolved_resident_host_tokens_by_node
        for node_id, value in resolved_node_stock:
            if value < 0:
                raise ValueError("resident Host node stock must be nonnegative")
            if value > self.topology.node_host_capacity(node_id):
                raise ValueError("resident Host stock exceeds node capacity")
            attributed = sum(
                resolved_host_by_dp[dp_rank]
                for dp_rank, mapped_node in enumerate(self.topology.dp_to_node)
                if mapped_node == node_id
            )
            if value != attributed:
                raise ValueError(
                    "node Host resident stock must equal DP-attributed stock"
                )

        link_stock: set[tuple[int, int]] = set()
        for source, target, used in self.used_link_bandwidth_bytes:
            key = (int(source), int(target))
            if key in link_stock:
                raise ValueError("used link bandwidth contains duplicate links")
            link_stock.add(key)
            if used < 0:
                raise ValueError("used link bandwidth must be nonnegative")
            if used > self.topology.link_capacity(*key):
                raise ValueError("used link bandwidth exceeds window capacity")

        for name, fixed, physical in (
            (
                "fixed HBM",
                self.fixed_hbm_budget_tokens,
                tuple(
                    min(self.topology.hbm_capacity_tokens[rank])
                    for rank in range(self.topology.dp_size)
                ),
            ),
            (
                "fixed Host",
                self.fixed_host_budget_tokens,
                self.topology.host_policy_max_tokens,
            ),
        ):
            if fixed is None:
                continue
            if len(fixed) != self.topology.dp_size:
                raise ValueError(f"{name} budgets must have one value per DP")
            if any(value < 0 for value in fixed):
                raise ValueError(f"{name} budgets must be nonnegative")
            if any(value > physical[rank] for rank, value in enumerate(fixed)):
                raise ValueError(f"{name} budget exceeds physical/policy maximum")
        if self.fixed_hbm_budget_tokens is not None:
            for dp_rank, fixed in enumerate(self.fixed_hbm_budget_tokens):
                if fixed < max(self.resolved_resident_hbm_tokens[dp_rank]):
                    raise ValueError("fixed HBM budget is below resident stock")
        if self.fixed_host_budget_tokens is not None:
            for dp_rank, fixed in enumerate(self.fixed_host_budget_tokens):
                if fixed < resolved_host_by_dp[dp_rank]:
                    raise ValueError("fixed Host budget is below resident stock")
        request_contract: dict[str, tuple] = {}
        request_chunk_ids: dict[str, list[int]] = {}
        for chunk in self.chunks:
            if chunk.source_dp_rank >= self.topology.dp_size:
                raise ValueError("chunk source DP is outside topology")
            if any(
                candidate < 0 or candidate >= self.topology.dp_size
                for candidate in chunk.backup_candidates
            ):
                raise ValueError("chunk backup candidate is outside topology")
            if set(dict(chunk.unprotected_extra_loss_s)) != scenario_set:
                raise ValueError("chunk losses do not cover every scenario")
            try:
                admission_state = admission_state_map[chunk.admission_state_id]
            except KeyError as exc:
                raise ValueError(
                    f"chunk references unknown admission state "
                    f"{chunk.admission_state_id!r}"
                ) from exc
            if not admission_state.contains(chunk.admission_time_s):
                raise ValueError(
                    "chunk admission time lies outside its observed state interval"
                )
            unavailable = set(admission_state.unavailable_dp_ranks)
            if chunk.source_dp_rank in unavailable:
                raise ValueError(
                    "chunk execution DP is unavailable at route admission"
                )
            if not any(
                candidate not in unavailable
                for candidate in chunk.backup_candidates
            ):
                raise ValueError(
                    f"request {chunk.request_id} has no available backup DP at "
                    "route admission"
                )
            contract = (
                chunk.workflow_id,
                chunk.source_dp_rank,
                chunk.backup_candidates,
                chunk.admission_time_s,
                chunk.admission_state_id,
            )
            previous = request_contract.setdefault(chunk.request_id, contract)
            if previous != contract:
                raise ValueError("all chunks of one request must share route contract")
            request_chunk_ids.setdefault(chunk.request_id, []).append(chunk.chunk_id)
            for candidate in chunk.backup_candidates:
                self.topology.link_capacity(chunk.source_dp_rank, candidate)
        for request_id, chunk_ids in request_chunk_ids.items():
            ordered = sorted(chunk_ids)
            if len(set(ordered)) != len(ordered):
                raise ValueError(f"duplicate chunk ID for request {request_id}")
            if ordered != list(range(ordered[0], ordered[-1] + 1)):
                raise ValueError(f"noncontiguous chunk IDs for request {request_id}")

    @property
    def sha256(self) -> str:
        return _sha256_json(asdict(self))

    @property
    def resolved_resident_hbm_tokens(self) -> tuple[tuple[int, ...], ...]:
        if self.resident_hbm_tokens:
            return self.resident_hbm_tokens
        return tuple(
            tuple(0 for _ in range(self.topology.pp_size))
            for _ in range(self.topology.dp_size)
        )

    @property
    def resolved_resident_host_tokens_by_dp(self) -> tuple[int, ...]:
        if self.resident_host_tokens_by_dp:
            return self.resident_host_tokens_by_dp
        return tuple(0 for _ in range(self.topology.dp_size))

    @property
    def resolved_resident_host_tokens_by_node(
        self,
    ) -> tuple[tuple[int, int], ...]:
        if self.resident_host_tokens_by_node:
            return self.resident_host_tokens_by_node
        by_dp = self.resolved_resident_host_tokens_by_dp
        return tuple(
            (
                node_id,
                sum(
                    by_dp[dp_rank]
                    for dp_rank, mapped_node in enumerate(self.topology.dp_to_node)
                    if mapped_node == node_id
                ),
            )
            for node_id in sorted(set(self.topology.dp_to_node))
        )

    def used_link_bytes(self, source_dp: int, target_dp: int) -> int:
        return int(
            {
                (int(source), int(target)): int(used)
                for source, target, used in self.used_link_bandwidth_bytes
            }.get((int(source_dp), int(target_dp)), 0)
        )


@dataclass(frozen=True)
class ExactStagePlacement:
    request_id: str
    chunk_id: int
    pp_rank: int
    backup_dp_rank: int
    tier: str
    token_count: int


@dataclass(frozen=True)
class ExactBudgetDecision:
    dp_rank: int
    hbm_budget_tokens_per_stage: int
    host_budget_tokens: int
    alpha: float
    beta: float


@dataclass(frozen=True)
class ExactSolverCertificate:
    solver: str
    solver_version: str
    status: str
    status_code: int
    message: str
    objective: float | None
    dual_bound: float | None
    mip_gap: float | None
    node_count: int | None
    solve_time_s: float
    instance_sha256: str
    decision_sha256: str | None
    deterministic_seed: int
    numerical_tolerance: float
    validation_errors: tuple[str, ...]
    solver_options: tuple[tuple[str, str], ...] = ()

    @property
    def certified_optimal(self) -> bool:
        if self.status != "OPTIMAL" or self.objective is None:
            return False
        if self.dual_bound is None or self.mip_gap is None:
            return False
        scale = max(abs(self.objective), 1.0)
        return (
            self.mip_gap <= self.numerical_tolerance
            and abs(self.objective - self.dual_bound)
            <= self.numerical_tolerance * scale
            and not self.validation_errors
            and self.decision_sha256 is not None
        )


@dataclass(frozen=True)
class ExactOptimizationResult:
    backup_dp_by_request: tuple[tuple[str, int], ...]
    placements: tuple[ExactStagePlacement, ...]
    budgets: tuple[ExactBudgetDecision, ...]
    recoverable_prefix_by_scenario: tuple[tuple[str, str, int], ...]
    workflow_loss_by_scenario: tuple[tuple[str, str, float], ...]
    certificate: ExactSolverCertificate

    def require_global_optimum(self) -> ExactOptimizationResult:
        if not self.certificate.certified_optimal:
            raise RuntimeError(
                "finite MILP is not certified globally optimal: "
                f"status={self.certificate.status}, gap={self.certificate.mip_gap}, "
                f"errors={self.certificate.validation_errors}"
            )
        return self


class _VariableBook:
    def __init__(self) -> None:
        self.keys: list[tuple] = []
        self.index: dict[tuple, int] = {}
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.objective: list[float] = []

    def add(
        self,
        key: tuple,
        *,
        lower: float,
        upper: float,
        integer: bool,
        objective: float = 0.0,
    ) -> int:
        if key in self.index:
            raise KeyError(key)
        index = len(self.keys)
        self.keys.append(key)
        self.index[key] = index
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.integrality.append(1 if integer else 0)
        self.objective.append(float(objective))
        return index

    def __getitem__(self, key: tuple) -> int:
        return self.index[key]


class _ConstraintBook:
    def __init__(self) -> None:
        self.rows: list[dict[int, float]] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, row: dict[int, float], *, lower: float, upper: float) -> None:
        compact = {index: value for index, value in row.items() if value != 0.0}
        self.rows.append(compact)
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def linear_constraint(self, variable_count: int) -> LinearConstraint:
        row_indices: list[int] = []
        column_indices: list[int] = []
        data: list[float] = []
        for row_index, row in enumerate(self.rows):
            for column_index, value in row.items():
                row_indices.append(row_index)
                column_indices.append(column_index)
                data.append(value)
        matrix = coo_matrix(
            (data, (row_indices, column_indices)),
            shape=(len(self.rows), variable_count),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


@dataclass(frozen=True)
class _Model:
    variables: _VariableBook
    constraints: _ConstraintBook
    requests: tuple[str, ...]
    workflows: tuple[str, ...]


def _build_model(instance: ExactOptimizationInstance) -> _Model:
    variables = _VariableBook()
    constraints = _ConstraintBook()
    topology = instance.topology
    weights = instance.weights
    requests = tuple(sorted({chunk.request_id for chunk in instance.chunks}))
    workflows = tuple(sorted({chunk.workflow_id for chunk in instance.chunks}))
    chunks_by_request: dict[str, list[int]] = {}
    request_candidates: dict[str, tuple[int, ...]] = {}
    admission_state_by_request: dict[str, ExactAdmissionState] = {}
    admission_state_map = {
        state.state_id: state for state in instance.admission_states
    }
    for chunk_index, chunk in enumerate(instance.chunks):
        chunks_by_request.setdefault(chunk.request_id, []).append(chunk_index)
        request_candidates[chunk.request_id] = chunk.backup_candidates
        admission_state_by_request[chunk.request_id] = admission_state_map[
            chunk.admission_state_id
        ]

    for request_id in requests:
        unavailable = set(
            admission_state_by_request[request_id].unavailable_dp_ranks
        )
        for backup_dp in request_candidates[request_id]:
            variables.add(
                ("y", request_id, backup_dp),
                lower=0.0,
                # Availability here is causal admission information, not the
                # future-failure scenario distribution used by the objective.
                upper=0.0 if backup_dp in unavailable else 1.0,
                integer=True,
            )

    for chunk_index, chunk in enumerate(instance.chunks):
        for pp_rank in range(topology.pp_size):
            for backup_dp in chunk.backup_candidates:
                for tier in TIERS:
                    normal_cost = 0.0
                    if tier == "HBM":
                        normal_cost = (
                            chunk.token_count * weights.hbm_normal_cost_per_token
                            + chunk.token_count
                            * weights.hbm_copy_bytes_per_token
                            * weights.copy_bandwidth_cost_per_byte
                        )
                    elif tier == "HOST":
                        normal_cost = (
                            chunk.token_count * weights.host_normal_cost_per_token
                            + chunk.token_count
                            * weights.host_copy_bytes_per_token
                            * weights.copy_bandwidth_cost_per_byte
                        )
                    variables.add(
                        ("x", chunk_index, pp_rank, backup_dp, tier),
                        lower=0.0,
                        upper=1.0,
                        integer=True,
                        objective=normal_cost,
                    )
        for backup_dp in chunk.backup_candidates:
            for scenario_index, _scenario in enumerate(instance.scenarios):
                variables.add(
                    ("g", chunk_index, backup_dp, scenario_index),
                    lower=0.0,
                    upper=1.0,
                    integer=True,
                )

    for dp_rank in range(topology.dp_size):
        hbm_max = min(topology.hbm_capacity_tokens[dp_rank])
        host_max = topology.host_policy_max_tokens[dp_rank]
        resident_hbm_floor = max(
            instance.resolved_resident_hbm_tokens[dp_rank]
        )
        resident_host_floor = instance.resolved_resident_host_tokens_by_dp[
            dp_rank
        ]
        fixed_hbm = (
            None
            if instance.fixed_hbm_budget_tokens is None
            else instance.fixed_hbm_budget_tokens[dp_rank]
        )
        fixed_host = (
            None
            if instance.fixed_host_budget_tokens is None
            else instance.fixed_host_budget_tokens[dp_rank]
        )
        variables.add(
            ("H", dp_rank),
            lower=(resident_hbm_floor if fixed_hbm is None else fixed_hbm),
            upper=(hbm_max if fixed_hbm is None else fixed_hbm),
            integer=True,
            objective=weights.hbm_budget_cost_per_token,
        )
        variables.add(
            ("M", dp_rank),
            lower=(resident_host_floor if fixed_host is None else fixed_host),
            upper=(host_max if fixed_host is None else fixed_host),
            integer=True,
            objective=weights.host_budget_cost_per_token,
        )
        variables.add(
            ("dH", dp_rank),
            lower=0.0,
            upper=max(hbm_max, instance.previous_hbm_budget_tokens[dp_rank]),
            integer=True,
            objective=weights.hbm_switch_cost_per_token,
        )
        variables.add(
            ("dM", dp_rank),
            lower=0.0,
            upper=max(host_max, instance.previous_host_budget_tokens[dp_rank]),
            integer=True,
            objective=weights.host_switch_cost_per_token,
        )

    for workflow_id in workflows:
        for scenario_index, scenario in enumerate(instance.scenarios):
            variables.add(
                ("L", workflow_id, scenario_index),
                lower=0.0,
                upper=np.inf,
                integer=False,
                objective=(
                    weights.expected_workflow_loss * scenario.probability
                ),
            )
    variables.add(
        ("eta",),
        lower=0.0,
        upper=np.inf,
        integer=False,
        objective=weights.cvar,
    )
    for scenario_index, scenario in enumerate(instance.scenarios):
        variables.add(
            ("xi", scenario_index),
            lower=0.0,
            upper=np.inf,
            integer=False,
            objective=(
                weights.cvar
                * scenario.probability
                / (1.0 - weights.cvar_tau)
            ),
        )

    # One immutable backup DP per request.
    for request_id in requests:
        constraints.add(
            {
                variables[("y", request_id, backup_dp)]: 1.0
                for backup_dp in request_candidates[request_id]
            },
            lower=1.0,
            upper=1.0,
        )

    # Every selected request/chunk/stage has exactly one explicit tier.
    for chunk_index, chunk in enumerate(instance.chunks):
        for pp_rank in range(topology.pp_size):
            for backup_dp in chunk.backup_candidates:
                row = {
                    variables[("x", chunk_index, pp_rank, backup_dp, tier)]: 1.0
                    for tier in TIERS
                }
                row[variables[("y", chunk.request_id, backup_dp)]] = -1.0
                constraints.add(row, lower=0.0, upper=0.0)

    # One DP-owned HBM budget applied to every stage, and one DP Host budget.
    for dp_rank in range(topology.dp_size):
        for pp_rank in range(topology.pp_size):
            row = {variables[("H", dp_rank)]: -1.0}
            for chunk_index, chunk in enumerate(instance.chunks):
                if dp_rank not in chunk.backup_candidates:
                    continue
                row[variables[("x", chunk_index, pp_rank, dp_rank, "HBM")]] = (
                    float(chunk.token_count)
                )
            constraints.add(
                row,
                lower=-np.inf,
                upper=-float(
                    instance.resolved_resident_hbm_tokens[dp_rank][pp_rank]
                ),
            )
        host_row = {variables[("M", dp_rank)]: -1.0}
        for chunk_index, chunk in enumerate(instance.chunks):
            if dp_rank not in chunk.backup_candidates:
                continue
            for pp_rank in range(topology.pp_size):
                host_row[
                    variables[("x", chunk_index, pp_rank, dp_rank, "HOST")]
                ] = float(chunk.token_count)
        constraints.add(
            host_row,
            lower=-np.inf,
            upper=-float(
                instance.resolved_resident_host_tokens_by_dp[dp_rank]
            ),
        )

    # Node physical Host capacity is separate from DP beta policy capacity.
    for node_id in sorted(set(topology.dp_to_node)):
        row: dict[int, float] = {}
        for dp_rank, mapped_node in enumerate(topology.dp_to_node):
            if mapped_node != node_id:
                continue
            for chunk_index, chunk in enumerate(instance.chunks):
                if dp_rank not in chunk.backup_candidates:
                    continue
                for pp_rank in range(topology.pp_size):
                    row[
                        variables[("x", chunk_index, pp_rank, dp_rank, "HOST")]
                    ] = float(chunk.token_count)
        constraints.add(
            row,
            lower=-np.inf,
            upper=float(
                topology.node_host_capacity(node_id)
                - dict(instance.resolved_resident_host_tokens_by_node)[node_id]
            ),
        )

    # Directed copy bandwidth for HBM and Host placement.
    for source_dp, backup_dp, capacity in topology.link_bandwidth_bytes:
        row: dict[int, float] = {}
        for chunk_index, chunk in enumerate(instance.chunks):
            if chunk.source_dp_rank != source_dp:
                continue
            if backup_dp not in chunk.backup_candidates:
                continue
            for pp_rank in range(topology.pp_size):
                row[
                    variables[("x", chunk_index, pp_rank, backup_dp, "HBM")]
                ] = float(
                    chunk.token_count * weights.hbm_copy_bytes_per_token
                )
                row[
                    variables[("x", chunk_index, pp_rank, backup_dp, "HOST")]
                ] = float(
                    chunk.token_count * weights.host_copy_bytes_per_token
                )
        constraints.add(
            row,
            lower=-np.inf,
            upper=float(
                capacity - instance.used_link_bytes(source_dp, backup_dp)
            ),
        )

    # Absolute budget switching cost.
    for dp_rank in range(topology.dp_size):
        previous_h = float(instance.previous_hbm_budget_tokens[dp_rank])
        previous_m = float(instance.previous_host_budget_tokens[dp_rank])
        constraints.add(
            {
                variables[("H", dp_rank)]: 1.0,
                variables[("dH", dp_rank)]: -1.0,
            },
            lower=-np.inf,
            upper=previous_h,
        )
        constraints.add(
            {
                variables[("H", dp_rank)]: -1.0,
                variables[("dH", dp_rank)]: -1.0,
            },
            lower=-np.inf,
            upper=-previous_h,
        )
        constraints.add(
            {
                variables[("M", dp_rank)]: 1.0,
                variables[("dM", dp_rank)]: -1.0,
            },
            lower=-np.inf,
            upper=previous_m,
        )
        constraints.add(
            {
                variables[("M", dp_rank)]: -1.0,
                variables[("dM", dp_rank)]: -1.0,
            },
            lower=-np.inf,
            upper=-previous_m,
        )

    # Exact all-stage survival AND for each scenario.
    for chunk_index, chunk in enumerate(instance.chunks):
        for backup_dp in chunk.backup_candidates:
            backup_node = topology.dp_to_node[backup_dp]
            host_domain = topology.host_failure_domain(backup_node)
            for scenario_index, scenario in enumerate(instance.scenarios):
                failed = set(scenario.failed_domains)
                g_index = variables[("g", chunk_index, backup_dp, scenario_index)]
                survival_rows: list[dict[int, float]] = []
                for pp_rank in range(topology.pp_size):
                    hbm_survives = (
                        topology.hbm_failure_domains[backup_dp][pp_rank]
                        not in failed
                    )
                    host_survives = host_domain not in failed
                    survival = {
                        variables[("x", chunk_index, pp_rank, backup_dp, "HBM")]: (
                            1.0 if hbm_survives else 0.0
                        ),
                        variables[("x", chunk_index, pp_rank, backup_dp, "HOST")]: (
                            1.0 if host_survives else 0.0
                        ),
                    }
                    survival_rows.append(survival)
                    upper_row = {g_index: 1.0}
                    for index, coefficient in survival.items():
                        upper_row[index] = upper_row.get(index, 0.0) - coefficient
                    constraints.add(upper_row, lower=-np.inf, upper=0.0)
                lower_row = {g_index: 1.0}
                for survival in survival_rows:
                    for index, coefficient in survival.items():
                        lower_row[index] = lower_row.get(index, 0.0) - coefficient
                constraints.add(
                    lower_row,
                    lower=-(topology.pp_size - 1),
                    upper=np.inf,
                )

    # A later chunk cannot be declared recoverable when an earlier chunk is not.
    for request_id, indices in chunks_by_request.items():
        ordered = sorted(indices, key=lambda index: instance.chunks[index].chunk_id)
        for previous_index, next_index in zip(ordered, ordered[1:]):
            for backup_dp in request_candidates[request_id]:
                for scenario_index, _scenario in enumerate(instance.scenarios):
                    constraints.add(
                        {
                            variables[("g", next_index, backup_dp, scenario_index)]: 1.0,
                            variables[("g", previous_index, backup_dp, scenario_index)]: -1.0,
                        },
                        lower=-np.inf,
                        upper=0.0,
                    )

    # Scenario workflow loss equals base + unprotected loss - protected benefit.
    for workflow_id in workflows:
        workflow_chunk_indices = [
            index
            for index, chunk in enumerate(instance.chunks)
            if chunk.workflow_id == workflow_id
        ]
        for scenario_index, scenario in enumerate(instance.scenarios):
            row = {variables[("L", workflow_id, scenario_index)]: 1.0}
            rhs = float(dict(scenario.base_workflow_loss_s).get(workflow_id, 0.0))
            for chunk_index in workflow_chunk_indices:
                chunk = instance.chunks[chunk_index]
                unprotected = float(
                    dict(chunk.unprotected_extra_loss_s)[scenario.scenario_id]
                )
                protected = float(
                    dict(chunk.protected_extra_loss_s)[scenario.scenario_id]
                )
                rhs += unprotected
                benefit = unprotected - protected
                for backup_dp in chunk.backup_candidates:
                    row[
                        variables[("g", chunk_index, backup_dp, scenario_index)]
                    ] = benefit
            constraints.add(row, lower=rhs, upper=rhs)

    # CVaR of total scenario workflow loss.
    for scenario_index, _scenario in enumerate(instance.scenarios):
        row = {
            variables[("xi", scenario_index)]: 1.0,
            variables[("eta",)]: 1.0,
        }
        for workflow_id in workflows:
            row[variables[("L", workflow_id, scenario_index)]] = -1.0
        constraints.add(row, lower=0.0, upper=np.inf)

    return _Model(
        variables=variables,
        constraints=constraints,
        requests=requests,
        workflows=workflows,
    )


def _selected_value(solution: np.ndarray, index: int) -> bool:
    return bool(solution[index] >= 0.5)


def _extract_decisions(
    instance: ExactOptimizationInstance,
    model: _Model,
    solution: np.ndarray,
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[ExactStagePlacement, ...],
    tuple[ExactBudgetDecision, ...],
    tuple[tuple[str, str, int], ...],
    tuple[tuple[str, str, float], ...],
]:
    variables = model.variables
    backup_by_request: list[tuple[str, int]] = []
    request_candidates = {
        chunk.request_id: chunk.backup_candidates for chunk in instance.chunks
    }
    for request_id in model.requests:
        selected = [
            backup_dp
            for backup_dp in request_candidates[request_id]
            if _selected_value(solution, variables[("y", request_id, backup_dp)])
        ]
        if len(selected) == 1:
            backup_by_request.append((request_id, selected[0]))

    placements: list[ExactStagePlacement] = []
    selected_routes = dict(backup_by_request)
    for chunk_index, chunk in enumerate(instance.chunks):
        backup_dp = selected_routes.get(chunk.request_id)
        if backup_dp is None:
            continue
        for pp_rank in range(instance.topology.pp_size):
            tiers = [
                tier
                for tier in TIERS
                if _selected_value(
                    solution,
                    variables[("x", chunk_index, pp_rank, backup_dp, tier)],
                )
            ]
            if len(tiers) == 1:
                placements.append(
                    ExactStagePlacement(
                        request_id=chunk.request_id,
                        chunk_id=chunk.chunk_id,
                        pp_rank=pp_rank,
                        backup_dp_rank=backup_dp,
                        tier=tiers[0],
                        token_count=chunk.token_count,
                    )
                )

    budgets: list[ExactBudgetDecision] = []
    for dp_rank in range(instance.topology.dp_size):
        hbm = int(round(solution[variables[("H", dp_rank)]]))
        host = int(round(solution[variables[("M", dp_rank)]]))
        hbm_max = min(instance.topology.hbm_capacity_tokens[dp_rank])
        host_max = instance.topology.host_policy_max_tokens[dp_rank]
        budgets.append(
            ExactBudgetDecision(
                dp_rank=dp_rank,
                hbm_budget_tokens_per_stage=hbm,
                host_budget_tokens=host,
                alpha=hbm / max(hbm_max, 1),
                beta=host / max(host_max, 1),
            )
        )

    recoverable: list[tuple[str, str, int]] = []
    chunk_indices_by_request: dict[str, list[int]] = {}
    for chunk_index, chunk in enumerate(instance.chunks):
        chunk_indices_by_request.setdefault(chunk.request_id, []).append(chunk_index)
    for scenario_index, scenario in enumerate(instance.scenarios):
        for request_id in model.requests:
            backup_dp = selected_routes.get(request_id)
            prefix = 0
            if backup_dp is not None:
                ordered = sorted(
                    chunk_indices_by_request[request_id],
                    key=lambda index: instance.chunks[index].chunk_id,
                )
                for chunk_index in ordered:
                    if not _selected_value(
                        solution,
                        variables[("g", chunk_index, backup_dp, scenario_index)],
                    ):
                        break
                    prefix += instance.chunks[chunk_index].token_count
            recoverable.append((scenario.scenario_id, request_id, prefix))

    workflow_losses = tuple(
        (
            scenario.scenario_id,
            workflow_id,
            float(solution[variables[("L", workflow_id, scenario_index)]]),
        )
        for scenario_index, scenario in enumerate(instance.scenarios)
        for workflow_id in model.workflows
    )
    return (
        tuple(backup_by_request),
        tuple(placements),
        tuple(budgets),
        tuple(recoverable),
        workflow_losses,
    )


def _validate_decisions(
    instance: ExactOptimizationInstance,
    backup_by_request: tuple[tuple[str, int], ...],
    placements: tuple[ExactStagePlacement, ...],
    budgets: tuple[ExactBudgetDecision, ...],
    recoverable_prefix_by_scenario: tuple[tuple[str, str, int], ...],
    workflow_loss_by_scenario: tuple[tuple[str, str, float], ...],
    solver_objective: float | None,
) -> tuple[str, ...]:
    errors: list[str] = []
    routes = dict(backup_by_request)
    expected_requests = {chunk.request_id for chunk in instance.chunks}
    if set(routes) != expected_requests:
        errors.append("route decision missing or duplicated request")
    admission_state_map = {
        state.state_id: state for state in instance.admission_states
    }
    first_chunk_by_request: dict[str, ExactChunk] = {}
    for chunk in instance.chunks:
        first_chunk_by_request.setdefault(chunk.request_id, chunk)
    for request_id, backup_dp in routes.items():
        chunk = first_chunk_by_request.get(request_id)
        if chunk is None:
            continue
        state = admission_state_map[chunk.admission_state_id]
        if backup_dp in state.unavailable_dp_ranks:
            errors.append(
                "route selected unavailable backup at admission: "
                f"request={request_id} state={state.state_id} DP{backup_dp}"
            )
    chunk_map = {
        (chunk.request_id, chunk.chunk_id): chunk for chunk in instance.chunks
    }
    placement_map: dict[tuple[str, int, int], ExactStagePlacement] = {}
    for placement in placements:
        key = (placement.request_id, placement.chunk_id, placement.pp_rank)
        if key in placement_map:
            errors.append(f"duplicate placement {key}")
        placement_map[key] = placement
        chunk = chunk_map.get((placement.request_id, placement.chunk_id))
        if chunk is None:
            errors.append(f"placement references unknown chunk {key}")
            continue
        if placement.backup_dp_rank != routes.get(placement.request_id):
            errors.append(f"placement violates immutable route {key}")
        if placement.tier not in TIERS:
            errors.append(f"invalid tier {placement.tier}")
        if placement.token_count != chunk.token_count:
            errors.append(f"placement token count mismatch {key}")
    for chunk in instance.chunks:
        for pp_rank in range(instance.topology.pp_size):
            if (chunk.request_id, chunk.chunk_id, pp_rank) not in placement_map:
                errors.append(
                    f"missing placement {(chunk.request_id, chunk.chunk_id, pp_rank)}"
                )

    budget_map = {budget.dp_rank: budget for budget in budgets}
    if set(budget_map) != set(range(instance.topology.dp_size)):
        errors.append("budget decision missing or duplicated DP")
    for dp_rank, budget in budget_map.items():
        hbm_max = min(instance.topology.hbm_capacity_tokens[dp_rank])
        host_max = instance.topology.host_policy_max_tokens[dp_rank]
        if not 0 <= budget.hbm_budget_tokens_per_stage <= hbm_max:
            errors.append(f"DP{dp_rank} alpha budget outside physical bounds")
        if not 0 <= budget.host_budget_tokens <= host_max:
            errors.append(f"DP{dp_rank} beta budget outside policy bounds")
        expected_alpha = budget.hbm_budget_tokens_per_stage / max(hbm_max, 1)
        expected_beta = budget.host_budget_tokens / max(host_max, 1)
        if not math.isclose(
            budget.alpha,
            expected_alpha,
            rel_tol=0.0,
            abs_tol=instance.numerical_tolerance,
        ):
            errors.append(f"DP{dp_rank} alpha normalization mismatch")
        if not math.isclose(
            budget.beta,
            expected_beta,
            rel_tol=0.0,
            abs_tol=instance.numerical_tolerance,
        ):
            errors.append(f"DP{dp_rank} beta normalization mismatch")
    hbm_used: dict[tuple[int, int], int] = {
        (dp_rank, pp_rank): int(value)
        for dp_rank, row in enumerate(instance.resolved_resident_hbm_tokens)
        for pp_rank, value in enumerate(row)
    }
    host_used: dict[int, int] = {
        dp_rank: int(value)
        for dp_rank, value in enumerate(
            instance.resolved_resident_host_tokens_by_dp
        )
    }
    node_host_used: dict[int, int] = {
        int(node_id): int(value)
        for node_id, value in instance.resolved_resident_host_tokens_by_node
    }
    link_used: dict[tuple[int, int], int] = {
        (int(source), int(target)): int(used)
        for source, target, used in instance.used_link_bandwidth_bytes
    }
    for placement in placements:
        chunk = chunk_map[(placement.request_id, placement.chunk_id)]
        backup_dp = placement.backup_dp_rank
        if placement.tier == "HBM":
            hbm_used[(backup_dp, placement.pp_rank)] = (
                hbm_used.get((backup_dp, placement.pp_rank), 0)
                + placement.token_count
            )
            link_used[(chunk.source_dp_rank, backup_dp)] = (
                link_used.get((chunk.source_dp_rank, backup_dp), 0)
                + placement.token_count
                * instance.weights.hbm_copy_bytes_per_token
            )
        elif placement.tier == "HOST":
            host_used[backup_dp] = host_used.get(backup_dp, 0) + placement.token_count
            node_id = instance.topology.dp_to_node[backup_dp]
            node_host_used[node_id] = (
                node_host_used.get(node_id, 0) + placement.token_count
            )
            link_used[(chunk.source_dp_rank, backup_dp)] = (
                link_used.get((chunk.source_dp_rank, backup_dp), 0)
                + placement.token_count
                * instance.weights.host_copy_bytes_per_token
            )
    for (dp_rank, pp_rank), used in hbm_used.items():
        budget = budget_map.get(dp_rank)
        if budget is None or used > budget.hbm_budget_tokens_per_stage:
            errors.append(f"DP{dp_rank}.PP{pp_rank} exceeds alpha budget")
        if used > instance.topology.hbm_capacity_tokens[dp_rank][pp_rank]:
            errors.append(f"DP{dp_rank}.PP{pp_rank} exceeds physical HBM")
    for dp_rank, used in host_used.items():
        budget = budget_map.get(dp_rank)
        if budget is None or used > budget.host_budget_tokens:
            errors.append(f"DP{dp_rank} exceeds beta budget")
    for node_id, used in node_host_used.items():
        if used > instance.topology.node_host_capacity(node_id):
            errors.append(f"node {node_id} exceeds physical Host capacity")
    for (source_dp, backup_dp), used in link_used.items():
        if used > instance.topology.link_capacity(source_dp, backup_dp):
            errors.append(f"link DP{source_dp}->DP{backup_dp} exceeds bandwidth")
    if instance.fixed_hbm_budget_tokens is not None:
        for dp_rank, expected in enumerate(instance.fixed_hbm_budget_tokens):
            budget = budget_map.get(dp_rank)
            if budget is None or budget.hbm_budget_tokens_per_stage != expected:
                errors.append(f"DP{dp_rank} violates fixed alpha budget")
    if instance.fixed_host_budget_tokens is not None:
        for dp_rank, expected in enumerate(instance.fixed_host_budget_tokens):
            budget = budget_map.get(dp_rank)
            if budget is None or budget.host_budget_tokens != expected:
                errors.append(f"DP{dp_rank} violates fixed beta budget")

    # Independently rebuild scenario survival and contiguous safe prefixes from
    # the selected tiers/failure domains; do not trust the solver's g variables.
    recoverable_map = {
        (scenario_id, request_id): int(tokens)
        for scenario_id, request_id, tokens in recoverable_prefix_by_scenario
    }
    expected_recoverable_keys = {
        (scenario.scenario_id, request_id)
        for scenario in instance.scenarios
        for request_id in expected_requests
    }
    if set(recoverable_map) != expected_recoverable_keys:
        errors.append("recoverable-prefix output keys are incomplete or duplicated")
    chunks_by_request: dict[str, list[ExactChunk]] = {}
    for chunk in instance.chunks:
        chunks_by_request.setdefault(chunk.request_id, []).append(chunk)
    chunk_survival: dict[tuple[str, str, int], bool] = {}
    for scenario in instance.scenarios:
        failed = set(scenario.failed_domains)
        for request_id, request_chunks in chunks_by_request.items():
            backup_dp = routes.get(request_id)
            prefix = 0
            prefix_open = backup_dp is not None
            for chunk in sorted(request_chunks, key=lambda value: value.chunk_id):
                survives = prefix_open
                if backup_dp is None:
                    survives = False
                else:
                    node_id = instance.topology.dp_to_node[backup_dp]
                    host_domain = instance.topology.host_failure_domain(node_id)
                    for pp_rank in range(instance.topology.pp_size):
                        placement = placement_map.get(
                            (request_id, chunk.chunk_id, pp_rank)
                        )
                        if placement is None:
                            survives = False
                            break
                        if placement.tier == "HBM":
                            survives = survives and (
                                instance.topology.hbm_failure_domains[backup_dp][
                                    pp_rank
                                ]
                                not in failed
                            )
                        elif placement.tier == "HOST":
                            survives = survives and host_domain not in failed
                        else:
                            survives = False
                        if not survives:
                            break
                chunk_survival[
                    (scenario.scenario_id, request_id, chunk.chunk_id)
                ] = survives
                if survives:
                    prefix += chunk.token_count
                else:
                    prefix_open = False
            if recoverable_map.get((scenario.scenario_id, request_id)) != prefix:
                errors.append(
                    "recoverable prefix mismatch for "
                    f"scenario={scenario.scenario_id} request={request_id}"
                )

    # Rebuild workflow losses from the same scenario/prefix contract.
    workflow_loss_map = {
        (scenario_id, workflow_id): float(loss)
        for scenario_id, workflow_id, loss in workflow_loss_by_scenario
    }
    workflows = {chunk.workflow_id for chunk in instance.chunks}
    expected_loss_keys = {
        (scenario.scenario_id, workflow_id)
        for scenario in instance.scenarios
        for workflow_id in workflows
    }
    if set(workflow_loss_map) != expected_loss_keys:
        errors.append("workflow-loss output keys are incomplete or duplicated")
    rebuilt_loss_map: dict[tuple[str, str], float] = {}
    for scenario in instance.scenarios:
        base = dict(scenario.base_workflow_loss_s)
        for workflow_id in workflows:
            loss = float(base.get(workflow_id, 0.0))
            for chunk in instance.chunks:
                if chunk.workflow_id != workflow_id:
                    continue
                protected = chunk_survival.get(
                    (scenario.scenario_id, chunk.request_id, chunk.chunk_id),
                    False,
                )
                values = dict(
                    chunk.protected_extra_loss_s
                    if protected
                    else chunk.unprotected_extra_loss_s
                )
                loss += float(values[scenario.scenario_id])
            rebuilt_loss_map[(scenario.scenario_id, workflow_id)] = loss
            observed = workflow_loss_map.get((scenario.scenario_id, workflow_id))
            if observed is None or not math.isclose(
                observed,
                loss,
                rel_tol=0.0,
                abs_tol=instance.numerical_tolerance,
            ):
                errors.append(
                    "workflow loss mismatch for "
                    f"scenario={scenario.scenario_id} workflow={workflow_id}"
                )

    # Independently recompute the complete objective, including empirical CVaR.
    weights = instance.weights
    objective = 0.0
    scenario_total_losses: list[tuple[float, float]] = []
    for scenario in instance.scenarios:
        total_loss = sum(
            rebuilt_loss_map[(scenario.scenario_id, workflow_id)]
            for workflow_id in workflows
        )
        scenario_total_losses.append((scenario.probability, total_loss))
        objective += (
            weights.expected_workflow_loss * scenario.probability * total_loss
        )
    if weights.cvar > 0.0:
        eta_candidates = {0.0, *(loss for _probability, loss in scenario_total_losses)}
        cvar_value = min(
            eta
            + sum(
                probability * max(loss - eta, 0.0)
                for probability, loss in scenario_total_losses
            )
            / (1.0 - weights.cvar_tau)
            for eta in eta_candidates
        )
        objective += weights.cvar * cvar_value
    for placement in placements:
        if placement.tier == "HBM":
            objective += (
                placement.token_count * weights.hbm_normal_cost_per_token
                + placement.token_count
                * weights.hbm_copy_bytes_per_token
                * weights.copy_bandwidth_cost_per_byte
            )
        elif placement.tier == "HOST":
            objective += (
                placement.token_count * weights.host_normal_cost_per_token
                + placement.token_count
                * weights.host_copy_bytes_per_token
                * weights.copy_bandwidth_cost_per_byte
            )
    for dp_rank, budget in budget_map.items():
        objective += (
            budget.hbm_budget_tokens_per_stage
            * weights.hbm_budget_cost_per_token
            + budget.host_budget_tokens * weights.host_budget_cost_per_token
            + abs(
                budget.hbm_budget_tokens_per_stage
                - instance.previous_hbm_budget_tokens[dp_rank]
            )
            * weights.hbm_switch_cost_per_token
            + abs(
                budget.host_budget_tokens
                - instance.previous_host_budget_tokens[dp_rank]
            )
            * weights.host_switch_cost_per_token
        )
    if solver_objective is None or not math.isclose(
        objective,
        solver_objective,
        rel_tol=0.0,
        abs_tol=instance.numerical_tolerance
        * max(abs(objective), abs(solver_objective or 0.0), 1.0),
    ):
        errors.append(
            f"objective mismatch: independently={objective} solver={solver_objective}"
        )
    return tuple(sorted(set(errors)))


def solve_exact_placement(
    instance: ExactOptimizationInstance,
) -> ExactOptimizationResult:
    """Solve and certify one finite AegisKV placement/budget instance."""

    model = _build_model(instance)
    variables = model.variables
    options = {
        "disp": False,
        "presolve": True,
        "time_limit": float(instance.time_limit_s),
        "mip_rel_gap": 0.0,
        "random_seed": int(instance.deterministic_seed),
        "threads": 1,
    }
    started = time.perf_counter()
    # SciPy forwards these two native HiGHS options but warns because they are
    # not in SciPy's short allow-list.  Keep them explicit for deterministic,
    # single-thread experiment isolation and suppress only that wrapper warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        result = milp(
            c=np.asarray(variables.objective, dtype=float),
            integrality=np.asarray(variables.integrality, dtype=np.int32),
            bounds=Bounds(
                np.asarray(variables.lower, dtype=float),
                np.asarray(variables.upper, dtype=float),
            ),
            constraints=model.constraints.linear_constraint(len(variables.keys)),
            options=options,
        )
    solve_time = time.perf_counter() - started
    status_names = {
        0: "OPTIMAL",
        1: "TIME_OR_ITERATION_LIMIT",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "SOLVER_ERROR",
    }
    status = status_names.get(int(result.status), f"UNKNOWN_{result.status}")

    backup_by_request: tuple[tuple[str, int], ...] = ()
    placements: tuple[ExactStagePlacement, ...] = ()
    budgets: tuple[ExactBudgetDecision, ...] = ()
    recoverable: tuple[tuple[str, str, int], ...] = ()
    workflow_losses: tuple[tuple[str, str, float], ...] = ()
    validation_errors: tuple[str, ...] = ()
    decision_sha256: str | None = None
    if result.x is not None:
        (
            backup_by_request,
            placements,
            budgets,
            recoverable,
            workflow_losses,
        ) = _extract_decisions(instance, model, np.asarray(result.x, dtype=float))
        validation_errors = _validate_decisions(
            instance,
            backup_by_request,
            placements,
            budgets,
            recoverable,
            workflow_losses,
            float(result.fun) if result.fun is not None else None,
        )
        decision_sha256 = _sha256_json(
            {
                "backup_dp_by_request": backup_by_request,
                "placements": [asdict(value) for value in placements],
                "budgets": [asdict(value) for value in budgets],
                "recoverable_prefix_by_scenario": recoverable,
                "workflow_loss_by_scenario": workflow_losses,
            }
        )

    certificate = ExactSolverCertificate(
        solver="scipy.optimize.milp/HiGHS",
        solver_version=f"scipy-{scipy.__version__}",
        status=status,
        status_code=int(result.status),
        message=str(result.message),
        objective=float(result.fun) if result.fun is not None else None,
        dual_bound=(
            float(result.mip_dual_bound)
            if getattr(result, "mip_dual_bound", None) is not None
            else None
        ),
        mip_gap=(
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        node_count=(
            int(result.mip_node_count)
            if getattr(result, "mip_node_count", None) is not None
            else None
        ),
        solve_time_s=solve_time,
        instance_sha256=instance.sha256,
        decision_sha256=decision_sha256,
        deterministic_seed=instance.deterministic_seed,
        numerical_tolerance=instance.numerical_tolerance,
        validation_errors=validation_errors,
        solver_options=tuple(
            sorted((str(key), str(value)) for key, value in options.items())
        ),
    )
    return ExactOptimizationResult(
        backup_dp_by_request=backup_by_request,
        placements=placements,
        budgets=budgets,
        recoverable_prefix_by_scenario=recoverable,
        workflow_loss_by_scenario=workflow_losses,
        certificate=certificate,
    )


def build_certified_exact_window_plan(
    instance: ExactOptimizationInstance,
    result: ExactOptimizationResult,
    *,
    window_id: int,
    route_version: int,
    valid_from_logical_time: float,
) -> CertifiedExactWindowPlan:
    """Convert a certified offline solve into the solver-free runtime wire plan."""

    result.require_global_optimum()
    certificate = result.certificate
    if certificate.decision_sha256 is None:
        raise RuntimeError("certified exact result has no decision digest")
    chunk_by_request: dict[str, ExactChunk] = {}
    for chunk in instance.chunks:
        chunk_by_request.setdefault(chunk.request_id, chunk)
    admission_state_map = {
        state.state_id: state for state in instance.admission_states
    }
    routes = tuple(
        CertifiedRoutePlan(
            request_id=request_id,
            workflow_id=chunk_by_request[request_id].workflow_id,
            execution_dp_rank=chunk_by_request[request_id].source_dp_rank,
            backup_dp_rank=backup_dp_rank,
            admission_time_s=chunk_by_request[request_id].admission_time_s,
            admission_state_id=(
                chunk_by_request[request_id].admission_state_id
            ),
            observed_unavailable_dp_ranks=admission_state_map[
                chunk_by_request[request_id].admission_state_id
            ].unavailable_dp_ranks,
            admission_information_mode=admission_state_map[
                chunk_by_request[request_id].admission_state_id
            ].information_mode,
        )
        for request_id, backup_dp_rank in sorted(
            result.backup_dp_by_request
        )
    )
    return CertifiedExactWindowPlan(
        instance_id=instance.instance_id,
        instance_sha256=certificate.instance_sha256,
        decision_sha256=certificate.decision_sha256,
        solver=certificate.solver,
        solver_version=certificate.solver_version,
        solver_status=certificate.status,
        objective=certificate.objective,
        dual_bound=certificate.dual_bound,
        mip_gap=certificate.mip_gap,
        solve_time_s=certificate.solve_time_s,
        numerical_tolerance=certificate.numerical_tolerance,
        validation_errors=certificate.validation_errors,
        dp_size=instance.topology.dp_size,
        pp_size=instance.topology.pp_size,
        window_id=int(window_id),
        route_version=int(route_version),
        valid_from_logical_time=float(valid_from_logical_time),
        routes=routes,
        budgets=tuple(
            CertifiedDPBudgetPlan(
                dp_rank=budget.dp_rank,
                hbm_budget_tokens_per_stage=(
                    budget.hbm_budget_tokens_per_stage
                ),
                host_budget_tokens=budget.host_budget_tokens,
                alpha=budget.alpha,
                beta=budget.beta,
            )
            for budget in sorted(result.budgets, key=lambda value: value.dp_rank)
        ),
        placements=tuple(
            CertifiedStageChunkPlacement(
                request_id=placement.request_id,
                chunk_id=placement.chunk_id,
                pp_rank=placement.pp_rank,
                backup_dp_rank=placement.backup_dp_rank,
                tier=placement.tier,
                token_count=placement.token_count,
            )
            for placement in sorted(
                result.placements,
                key=lambda value: (
                    value.request_id,
                    value.chunk_id,
                    value.pp_rank,
                ),
            )
        ),
        budget_publication_mode=(
            "reuse_current_fixed"
            if instance.fixed_hbm_budget_tokens is not None
            and instance.fixed_host_budget_tokens is not None
            else "publish_new"
        ),
    ).require_certified_optimal()
