"""Lightweight wire contract for certified AegisKV rolling windows.

This module intentionally has no solver dependency.  Offline code may import
SciPy/HiGHS to build the contract; online and offline runtime controllers only
validate and execute the frozen decisions carried here.
"""

from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXACT_PLACEMENT_TIERS = ("HBM", "HOST", "RECOMPUTE")
EXACT_ADMISSION_INFORMATION_MODES = (
    "causal_observed",
    "topology_robust",
    "future_manifest_oracle",
)
EXACT_BUDGET_PUBLICATION_MODES = (
    "publish_new",
    "reuse_current_fixed",
)


def _plan_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept either a bare wire plan or an offline solver artifact.

    Runtime code deliberately consumes only the lightweight certified plan.
    The surrounding artifact may additionally retain the full finite instance,
    solver result, and build provenance for paper-grade reproducibility.
    """

    for key in ("certified_exact_window_plan", "plan"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            return nested
    return raw


@dataclass(frozen=True)
class CertifiedRoutePlan:
    request_id: str
    workflow_id: str
    execution_dp_rank: int
    backup_dp_rank: int
    admission_time_s: float = 0.0
    admission_state_id: str = "all_available"
    observed_unavailable_dp_ranks: tuple[int, ...] = ()
    admission_information_mode: str = "causal_observed"

    def __post_init__(self) -> None:
        if not self.request_id or not self.workflow_id:
            raise ValueError("exact route request/workflow IDs must be nonempty")
        if self.execution_dp_rank < 0 or self.backup_dp_rank < 0:
            raise ValueError("exact route DP ranks must be nonnegative")
        if self.execution_dp_rank == self.backup_dp_rank:
            raise ValueError("exact route execution and backup DP must differ")
        if not math.isfinite(self.admission_time_s) or self.admission_time_s < 0.0:
            raise ValueError("exact route admission time must be finite/nonnegative")
        if not self.admission_state_id:
            raise ValueError("exact route admission state must be nonempty")
        if len(set(self.observed_unavailable_dp_ranks)) != len(
            self.observed_unavailable_dp_ranks
        ):
            raise ValueError("exact route unavailable DP ranks must be unique")
        if any(rank < 0 for rank in self.observed_unavailable_dp_ranks):
            raise ValueError("exact route unavailable DP ranks must be nonnegative")
        if self.execution_dp_rank in self.observed_unavailable_dp_ranks:
            raise ValueError("exact route execution DP was unavailable at admission")
        if self.backup_dp_rank in self.observed_unavailable_dp_ranks:
            raise ValueError("exact route backup DP was unavailable at admission")
        if self.admission_information_mode not in (
            EXACT_ADMISSION_INFORMATION_MODES
        ):
            raise ValueError("unknown exact route admission information mode")


@dataclass(frozen=True)
class CertifiedStageChunkPlacement:
    request_id: str
    chunk_id: int
    pp_rank: int
    backup_dp_rank: int
    tier: str
    token_count: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("exact placement request_id must be nonempty")
        if self.chunk_id < 0 or self.pp_rank < 0 or self.backup_dp_rank < 0:
            raise ValueError("exact placement indices must be nonnegative")
        if self.tier not in EXACT_PLACEMENT_TIERS:
            raise ValueError(f"unknown exact placement tier {self.tier!r}")
        if self.token_count <= 0:
            raise ValueError("exact placement token_count must be positive")


@dataclass(frozen=True)
class CertifiedDPBudgetPlan:
    dp_rank: int
    hbm_budget_tokens_per_stage: int
    host_budget_tokens: int
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.dp_rank < 0:
            raise ValueError("exact budget DP rank must be nonnegative")
        if self.hbm_budget_tokens_per_stage < 0 or self.host_budget_tokens < 0:
            raise ValueError("exact budget token limits must be nonnegative")
        if not 0.0 <= self.alpha <= 1.0 or not 0.0 <= self.beta <= 1.0:
            raise ValueError("exact alpha/beta must be in [0, 1]")


@dataclass(frozen=True)
class CertifiedExactWindowPlan:
    instance_id: str
    instance_sha256: str
    decision_sha256: str
    solver: str
    solver_version: str
    solver_status: str
    objective: float | None
    dual_bound: float | None
    mip_gap: float | None
    solve_time_s: float
    numerical_tolerance: float
    validation_errors: tuple[str, ...]
    dp_size: int
    pp_size: int
    window_id: int
    route_version: int
    valid_from_logical_time: float
    routes: tuple[CertifiedRoutePlan, ...]
    budgets: tuple[CertifiedDPBudgetPlan, ...]
    placements: tuple[CertifiedStageChunkPlacement, ...]
    budget_publication_mode: str = "publish_new"
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.instance_id or not self.instance_sha256:
            raise ValueError("exact window instance identity must be nonempty")
        if not self.decision_sha256 or not self.solver or not self.solver_version:
            raise ValueError("exact window solver identity must be nonempty")
        if self.dp_size <= 0 or self.pp_size <= 0:
            raise ValueError("exact window topology sizes must be positive")
        if self.window_id < 0 or self.route_version <= 0:
            raise ValueError("exact window/route versions are invalid")
        if self.valid_from_logical_time < 0.0:
            raise ValueError("exact window logical time must be nonnegative")
        if self.solve_time_s < 0.0 or self.numerical_tolerance <= 0.0:
            raise ValueError("exact solve time/tolerance is invalid")
        if self.budget_publication_mode not in EXACT_BUDGET_PUBLICATION_MODES:
            raise ValueError("unknown exact budget publication mode")
        route_map = {route.request_id: route for route in self.routes}
        if len(route_map) != len(self.routes) or not route_map:
            raise ValueError("exact window routes must be nonempty and unique")
        if any(
            route.execution_dp_rank >= self.dp_size
            or route.backup_dp_rank >= self.dp_size
            or any(
                rank >= self.dp_size
                for rank in route.observed_unavailable_dp_ranks
            )
            for route in self.routes
        ):
            raise ValueError("exact route is outside the DP topology")
        budget_map = {budget.dp_rank: budget for budget in self.budgets}
        if set(budget_map) != set(range(self.dp_size)):
            raise ValueError("exact window must carry one budget per DP")

        placement_map: dict[
            tuple[str, int, int], CertifiedStageChunkPlacement
        ] = {}
        chunk_tokens: dict[tuple[str, int], int] = {}
        chunk_ids_by_request_stage: dict[tuple[str, int], list[int]] = {}
        for placement in self.placements:
            route = route_map.get(placement.request_id)
            if route is None:
                raise ValueError("exact placement references an unknown request")
            if placement.backup_dp_rank != route.backup_dp_rank:
                raise ValueError("exact placement violates immutable backup route")
            if placement.pp_rank >= self.pp_size:
                raise ValueError("exact placement is outside the PP topology")
            key = (
                placement.request_id,
                placement.chunk_id,
                placement.pp_rank,
            )
            if key in placement_map:
                raise ValueError(f"duplicate exact placement {key}")
            placement_map[key] = placement
            chunk_key = (placement.request_id, placement.chunk_id)
            previous_tokens = chunk_tokens.setdefault(
                chunk_key, placement.token_count
            )
            if previous_tokens != placement.token_count:
                raise ValueError("exact chunk token count differs across PP stages")
            chunk_ids_by_request_stage.setdefault(
                (placement.request_id, placement.pp_rank), []
            ).append(placement.chunk_id)

        for request_id in route_map:
            reference: tuple[int, ...] | None = None
            for pp_rank in range(self.pp_size):
                chunk_ids = tuple(
                    sorted(
                        chunk_ids_by_request_stage.get(
                            (request_id, pp_rank), []
                        )
                    )
                )
                if not chunk_ids or chunk_ids != tuple(range(len(chunk_ids))):
                    raise ValueError(
                        "every exact PP stage must cover contiguous chunks from zero"
                    )
                if reference is None:
                    reference = chunk_ids
                elif chunk_ids != reference:
                    raise ValueError(
                        "exact PP stages must cover the same request chunks"
                    )

    @property
    def certified_optimal(self) -> bool:
        if self.solver_status != "OPTIMAL" or self.objective is None:
            return False
        if self.dual_bound is None or self.mip_gap is None:
            return False
        scale = max(abs(self.objective), 1.0)
        return (
            self.mip_gap <= self.numerical_tolerance
            and math.isclose(
                self.objective,
                self.dual_bound,
                rel_tol=0.0,
                abs_tol=self.numerical_tolerance * scale,
            )
            and not self.validation_errors
        )

    @property
    def solver_certificate_id(self) -> str:
        return f"{self.solver}:{self.decision_sha256}"

    @property
    def uses_future_manifest_oracle(self) -> bool:
        return any(
            route.admission_information_mode == "future_manifest_oracle"
            for route in self.routes
        )

    def require_certified_optimal(self) -> "CertifiedExactWindowPlan":
        if not self.certified_optimal:
            raise RuntimeError(
                "exact window is not certified globally optimal: "
                f"status={self.solver_status}, gap={self.mip_gap}, "
                f"errors={self.validation_errors}"
            )
        return self

    def route_for(self, request_id: str) -> CertifiedRoutePlan | None:
        return next(
            (route for route in self.routes if route.request_id == request_id),
            None,
        )

    def placements_for(
        self, request_id: str, pp_rank: int
    ) -> tuple[CertifiedStageChunkPlacement, ...]:
        return tuple(
            sorted(
                (
                    placement
                    for placement in self.placements
                    if placement.request_id == request_id
                    and placement.pp_rank == int(pp_rank)
                ),
                key=lambda placement: placement.chunk_id,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CertifiedExactWindowPlan":
        raw = _plan_payload(raw)
        return cls(
            instance_id=str(raw.get("instance_id") or ""),
            instance_sha256=str(raw.get("instance_sha256") or ""),
            decision_sha256=str(raw.get("decision_sha256") or ""),
            solver=str(raw.get("solver") or ""),
            solver_version=str(raw.get("solver_version") or ""),
            solver_status=str(raw.get("solver_status") or ""),
            objective=(
                None
                if raw.get("objective") is None
                else float(raw["objective"])
            ),
            dual_bound=(
                None
                if raw.get("dual_bound") is None
                else float(raw["dual_bound"])
            ),
            mip_gap=(
                None if raw.get("mip_gap") is None else float(raw["mip_gap"])
            ),
            solve_time_s=float(raw.get("solve_time_s", 0.0)),
            numerical_tolerance=float(
                raw.get("numerical_tolerance", 1e-8)
            ),
            validation_errors=tuple(
                str(value) for value in raw.get("validation_errors", ())
            ),
            dp_size=int(raw.get("dp_size", 0)),
            pp_size=int(raw.get("pp_size", 0)),
            window_id=int(raw.get("window_id", -1)),
            route_version=int(raw.get("route_version", 0)),
            valid_from_logical_time=float(
                raw.get("valid_from_logical_time", 0.0)
            ),
            routes=tuple(
                CertifiedRoutePlan(
                    request_id=str(value.get("request_id") or ""),
                    workflow_id=str(value.get("workflow_id") or ""),
                    execution_dp_rank=int(value.get("execution_dp_rank", -1)),
                    backup_dp_rank=int(value.get("backup_dp_rank", -1)),
                    admission_time_s=float(value.get("admission_time_s", 0.0)),
                    admission_state_id=str(
                        value.get("admission_state_id") or "all_available"
                    ),
                    observed_unavailable_dp_ranks=tuple(
                        int(rank)
                        for rank in value.get(
                            "observed_unavailable_dp_ranks", ()
                        )
                    ),
                    admission_information_mode=str(
                        value.get("admission_information_mode")
                        or "causal_observed"
                    ),
                )
                for value in raw.get("routes", ())
            ),
            budgets=tuple(
                CertifiedDPBudgetPlan(**dict(value))
                for value in raw.get("budgets", ())
            ),
            placements=tuple(
                CertifiedStageChunkPlacement(**dict(value))
                for value in raw.get("placements", ())
            ),
            budget_publication_mode=str(
                raw.get("budget_publication_mode") or "publish_new"
            ),
            schema_version=int(raw.get("schema_version", 1)),
        )


def load_certified_exact_window_plan(
    path: str | Path,
) -> CertifiedExactWindowPlan:
    """Load and validate a certified runtime plan without importing a solver."""

    resolved = Path(path).expanduser().resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("certified exact window artifact must be a JSON object")
    return CertifiedExactWindowPlan.from_dict(raw).require_certified_optimal()
