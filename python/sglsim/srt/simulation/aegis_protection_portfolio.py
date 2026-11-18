"""Reference selector for AegisKV's normal-time protection portfolio.

This module is deliberately outside the online SGLang-parity runtime.  It
connects the single-failure-domain EDF semantics in
``aegis_recovery_laxity`` to the part of Algorithm 2 that is easy to describe
incorrectly in prose:

* one protection action may help jobs in several fail-now source domains;
* its resource demand is charged once against typed, shared budgets;
* action-gated peer/DRAM/SSD edges are composed by the serial prefix
  shortest-path reference for every selected set, preserving
  complementarities inside that operation grammar;
* prefix prerequisites are admitted as one dependency-closed bundle, so a
  zero-value first chunk cannot strand a valuable longer prefix;
* the objective is the parameter-free robust pair
  ``(worst domain maximum tardiness, sum of domain workflow overruns)``;
* the online reference rule uses lexicographic gain per dominant resource
  share and a deterministic victim-cover proposal when capacity is full;
  without modeled exchange headroom, a proposal whose victim-only phase is
  worse than the same-horizon baseline is rejected rather than pretending the
  replacement is atomic; and
* exhaustive enumeration certifies a small, static committed-action snapshot;
  it refuses nonzero copy lead times rather than pretending to be the paper's
  separate time-expanded/future-aware oracle.

It is a semantic oracle and falsification harness, not a general parallel
Recovery Planner and not evidence that real KV bytes, leases, copy completion,
or recovery-lane isolation are implemented.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass

from sglsim.srt.simulation.aegis_prefix_recovery import PrefixRecoveryProblem
from sglsim.srt.simulation.aegis_recovery_laxity import (
    RecoveryJob,
    RecoveryLaxityInstance,
    RecoveryRisk,
    RecoveryRiskGain,
    RecoverySchedule,
    RecoveryTarget,
)


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ResourceBudget:
    resource_id: str
    capacity: float

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id must be nonempty")
        _positive_finite("resource capacity", self.capacity)


@dataclass(frozen=True)
class ResourceDemand:
    resource_id: str
    amount: float

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id must be nonempty")
        _positive_finite("resource demand", self.amount)


@dataclass(frozen=True)
class ScenarioRecoveryProblem:
    """A state-dependent serial prefix plan for one scenario job."""

    job_id: str
    problem: PrefixRecoveryProblem

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("recovery problem job_id must be nonempty")


@dataclass(frozen=True)
class FailureScenario:
    failure_domain: str
    targets: tuple[RecoveryTarget, ...]
    jobs: tuple[RecoveryJob, ...]
    recovery_problems: tuple[ScenarioRecoveryProblem, ...] = ()

    def __post_init__(self) -> None:
        if not self.failure_domain:
            raise ValueError("failure_domain must be nonempty")
        # Reuse the single-domain model as the schema validator.
        RecoveryLaxityInstance(
            failure_domain=self.failure_domain,
            targets=self.targets,
            jobs=self.jobs,
            actions=(),
        )
        job_by_id = {job.job_id: job for job in self.jobs}
        problem_job_ids = [item.job_id for item in self.recovery_problems]
        if len(set(problem_job_ids)) != len(problem_job_ids):
            raise ValueError("recovery problems must have unique job IDs")
        for item in self.recovery_problems:
            job = job_by_id.get(item.job_id)
            if job is None:
                raise ValueError(
                    f"recovery problem references unknown job {item.job_id}"
                )
            baseline = item.problem.solve().total_service_upper_s
            if not math.isclose(
                baseline,
                job.unprotected_service_s,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "empty prefix plan must equal the job's unprotected "
                    "service"
                )


@dataclass(frozen=True)
class ScenarioEffect:
    """Legacy fixed-state service result for a job without a prefix problem.

    New composition-sensitive cases should use ``ScenarioRecoveryProblem`` and
    action-gated prefix edges.  This scalar adapter remains useful for actions
    whose complete post-action service is independent of every other action.
    """

    failure_domain: str
    job_id: str
    protected_service_s: float

    def __post_init__(self) -> None:
        if not self.failure_domain or not self.job_id:
            raise ValueError("effect identifiers must be nonempty")
        if (
            not math.isfinite(float(self.protected_service_s))
            or float(self.protected_service_s) < 0.0
        ):
            raise ValueError(
                "protected_service_s must be finite and nonnegative"
            )


@dataclass(frozen=True)
class PortfolioAction:
    """One incremental action or the terminal endpoint of a prefix bundle.

    For an action with predecessors, ``completion_offset_s`` is an absolute
    upper bound from the current decision origin to completion of its whole
    dependency closure, not merely the last chunk's standalone duration.
    ``valid_for_s`` is the remaining effect/source lifetime on the same axis.
    Every closure member's validity must cover that terminal horizon.  Once
    expired, an effect is absent from a horizon evaluation, but its resource
    demand remains charged until an explicit release/reclaim transition.
    """

    action_id: str
    demands: tuple[ResourceDemand, ...]
    effects: tuple[ScenarioEffect, ...]
    predecessors: tuple[str, ...] = ()
    releasable: bool = True
    completion_offset_s: float = 0.0
    valid_for_s: float | None = None

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be nonempty")
        if not self.demands:
            raise ValueError("action must consume at least one resource")
        resource_ids = [demand.resource_id for demand in self.demands]
        effect_ids = [
            (effect.failure_domain, effect.job_id) for effect in self.effects
        ]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("action demands must have unique resource IDs")
        if len(set(effect_ids)) != len(effect_ids):
            raise ValueError("action effects must have unique domain/job IDs")
        if len(set(self.predecessors)) != len(self.predecessors):
            raise ValueError("action predecessors must be unique")
        if self.action_id in self.predecessors:
            raise ValueError("action cannot depend on itself")
        if (
            not math.isfinite(float(self.completion_offset_s))
            or float(self.completion_offset_s) < 0.0
        ):
            raise ValueError(
                "completion_offset_s must be finite and nonnegative"
            )
        if self.valid_for_s is not None:
            if (
                not math.isfinite(float(self.valid_for_s))
                or float(self.valid_for_s) < 0.0
            ):
                raise ValueError("valid_for_s must be finite and nonnegative")
        if self.completion_offset_s > 0.0 and self.valid_for_s is None:
            raise ValueError(
                "a nonzero completion offset requires a validity bound"
            )


@dataclass(frozen=True)
class PortfolioOutcome:
    selected_actions: tuple[str, ...]
    risk: RecoveryRisk
    domain_schedules: tuple[tuple[str, RecoverySchedule], ...]


@dataclass(frozen=True)
class PortfolioGreedyStep:
    step: int
    added_action_id: str
    added_action_ids: tuple[str, ...]
    removed_action_ids: tuple[str, ...]
    completion_offset_s: float
    dominant_share: float
    maximum_tardiness_density: float
    workflow_overrun_density: float
    risk_before: tuple[float, float]
    risk_after: tuple[float, float]
    used_resources_after: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PortfolioGreedyResult:
    selected_actions: tuple[str, ...]
    outcome: PortfolioOutcome
    used_resources: tuple[tuple[str, float], ...]
    steps: tuple[PortfolioGreedyStep, ...]


@dataclass(frozen=True)
class PortfolioExactCertificate:
    status: str
    objective: tuple[float, float]
    best_bound: tuple[float, float]
    gap: float
    solve_time_s: float
    enumerated_subsets: int
    feasible_subsets: int
    instance_sha256: str


@dataclass(frozen=True)
class PortfolioExactResult:
    selected_actions: tuple[str, ...]
    outcome: PortfolioOutcome
    used_resources: tuple[tuple[str, float], ...]
    certificate: PortfolioExactCertificate


@dataclass(frozen=True)
class _Proposal:
    action: PortfolioAction
    additions: frozenset[str]
    victims: frozenset[str]
    selected_after: frozenset[str]
    outcome_before: PortfolioOutcome
    outcome_after: PortfolioOutcome
    gain: RecoveryRiskGain
    dominant_share: float


@dataclass(frozen=True)
class ProtectionPortfolio:
    scenarios: tuple[FailureScenario, ...]
    budgets: tuple[ResourceBudget, ...]
    actions: tuple[PortfolioAction, ...]

    def __post_init__(self) -> None:
        if not self.scenarios or not self.budgets:
            raise ValueError("scenarios and budgets must be nonempty")
        domains = [scenario.failure_domain for scenario in self.scenarios]
        resources = [budget.resource_id for budget in self.budgets]
        action_ids = [action.action_id for action in self.actions]
        if len(set(domains)) != len(domains):
            raise ValueError("failure domains must be unique")
        if len(set(resources)) != len(resources):
            raise ValueError("resource budgets must be unique")
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action IDs must be unique")

        budget_ids = set(resources)
        action_id_set = set(action_ids)
        jobs_by_domain = {
            scenario.failure_domain: {job.job_id: job for job in scenario.jobs}
            for scenario in self.scenarios
        }
        planned_jobs_by_domain = {
            scenario.failure_domain: {
                item.job_id for item in scenario.recovery_problems
            }
            for scenario in self.scenarios
        }
        prefix_action_ids = {
            action_id
            for scenario in self.scenarios
            for item in scenario.recovery_problems
            for action_id in item.problem.action_ids
        }
        unknown_prefix_actions = prefix_action_ids - action_id_set
        if unknown_prefix_actions:
            raise ValueError(
                "prefix recovery problems reference unknown actions: "
                f"{sorted(unknown_prefix_actions)}"
            )
        for action in self.actions:
            if (
                not action.effects
                and action.action_id not in prefix_action_ids
            ):
                raise ValueError(
                    f"action {action.action_id} has neither a scalar effect "
                    "nor an action-gated prefix edge"
                )
            unknown_resources = {
                demand.resource_id for demand in action.demands
            } - budget_ids
            if unknown_resources:
                raise ValueError(
                    f"action {action.action_id} references unknown resources: "
                    f"{sorted(unknown_resources)}"
                )
            unknown_predecessors = set(action.predecessors) - action_id_set
            if unknown_predecessors:
                raise ValueError(
                    f"action {action.action_id} references unknown predecessors: "
                    f"{sorted(unknown_predecessors)}"
                )
            for effect in action.effects:
                if effect.failure_domain not in jobs_by_domain:
                    raise ValueError(
                        f"action {action.action_id} references unknown domain "
                        f"{effect.failure_domain}"
                    )
                job = jobs_by_domain[effect.failure_domain].get(effect.job_id)
                if job is None:
                    raise ValueError(
                        f"action {action.action_id} references unknown job "
                        f"{effect.failure_domain}/{effect.job_id}"
                    )
                if (
                    effect.protected_service_s
                    > job.unprotected_service_s + 1e-12
                ):
                    raise ValueError(
                        "protection cannot increase recovery service"
                    )
                if (
                    effect.job_id
                    in planned_jobs_by_domain[effect.failure_domain]
                ):
                    raise ValueError(
                        "a job cannot mix scalar effects with a prefix "
                        "recovery problem"
                    )
        self._assert_acyclic_dependencies()

    @property
    def budget_by_id(self) -> dict[str, ResourceBudget]:
        return {budget.resource_id: budget for budget in self.budgets}

    @property
    def action_by_id(self) -> dict[str, PortfolioAction]:
        return {action.action_id: action for action in self.actions}

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _assert_acyclic_dependencies(self) -> None:
        by_id = self.action_by_id
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(action_id: str) -> None:
            if action_id in visited:
                return
            if action_id in visiting:
                raise ValueError("action dependency graph must be acyclic")
            visiting.add(action_id)
            for predecessor in by_id[action_id].predecessors:
                visit(predecessor)
            visiting.remove(action_id)
            visited.add(action_id)

        for action_id in sorted(by_id):
            visit(action_id)

    def _validate_selected(self, selected: frozenset[str]) -> None:
        unknown = selected - self.action_by_id.keys()
        if unknown:
            raise ValueError(f"unknown selected actions: {sorted(unknown)}")
        for action_id in selected:
            missing = set(self.action_by_id[action_id].predecessors) - selected
            if missing:
                raise ValueError(
                    f"selected action {action_id} is missing predecessors: "
                    f"{sorted(missing)}"
                )

    def used_resources(
        self, selected: frozenset[str]
    ) -> tuple[tuple[str, float], ...]:
        self._validate_selected(selected)
        used = {budget.resource_id: 0.0 for budget in self.budgets}
        for action_id in selected:
            for demand in self.action_by_id[action_id].demands:
                used[demand.resource_id] += demand.amount
        return tuple(sorted(used.items()))

    def fits(self, selected: frozenset[str]) -> bool:
        used = dict(self.used_resources(selected))
        return all(
            used[resource_id] <= budget.capacity + 1e-12
            for resource_id, budget in self.budget_by_id.items()
        )

    def dominant_share(self, action_id: str) -> float:
        try:
            action = self.action_by_id[action_id]
        except KeyError as exc:
            raise ValueError(f"unknown action: {action_id}") from exc
        return max(
            demand.amount / self.budget_by_id[demand.resource_id].capacity
            for demand in action.demands
        )

    def dominant_share_for(self, action_ids: frozenset[str]) -> float:
        """Return the dominant incremental share of an atomic action bundle."""

        if not action_ids:
            raise ValueError("action bundle must be nonempty")
        unknown = action_ids - self.action_by_id.keys()
        if unknown:
            raise ValueError(f"unknown bundled actions: {sorted(unknown)}")
        demand_by_resource = {
            budget.resource_id: 0.0 for budget in self.budgets
        }
        for action_id in action_ids:
            for demand in self.action_by_id[action_id].demands:
                demand_by_resource[demand.resource_id] += demand.amount
        return max(
            demand_by_resource[resource_id] / budget.capacity
            for resource_id, budget in self.budget_by_id.items()
        )

    def _dependency_closure(self, root: str) -> frozenset[str]:
        if root not in self.action_by_id:
            raise ValueError(f"unknown action: {root}")
        closure = {root}
        pending = [root]
        while pending:
            action_id = pending.pop()
            for predecessor in self.action_by_id[action_id].predecessors:
                if predecessor not in closure:
                    closure.add(predecessor)
                    pending.append(predecessor)
        return frozenset(closure)

    def evaluate(
        self,
        selected: frozenset[str],
        *,
        fail_offset_s: float = 0.0,
    ) -> PortfolioOutcome:
        """Evaluate a frozen job snapshot on one shared absolute time axis.

        ``RecoveryJob.slack_lcb_s`` is the ticket expiry relative to the
        decision origin.  Moving the hypothetical failure to
        ``fail_offset_s`` delays target/job availability by the same amount
        and uses ``max(expiry, fail_offset)`` as the failure-time due date.
        Thus copy lead time consumes positive remaining Slack but an expired
        ticket clamps to zero Slack instead of accumulating negative age.
        Unless an external horizon contract also fixes job
        liveness, prefix growth, and service evolution, this is a deliberately
        labeled myopic snapshot score, not an exact prediction of
        ``R(S; tau_a)`` for the future live workload.
        """

        self._validate_selected(selected)
        if (
            not math.isfinite(float(fail_offset_s))
            or float(fail_offset_s) < 0.0
        ):
            raise ValueError("fail_offset_s must be finite and nonnegative")
        domain_schedules: list[tuple[str, RecoverySchedule]] = []
        for scenario in sorted(
            self.scenarios, key=lambda item: item.failure_domain
        ):
            valid_selected = frozenset(
                action.action_id
                for action in self.actions
                if action.action_id in selected
                and (
                    action.valid_for_s is None
                    or fail_offset_s <= action.valid_for_s + 1e-12
                )
            )
            problem_by_job = {
                item.job_id: item.problem
                for item in scenario.recovery_problems
            }
            scalar_service = {
                job.job_id: job.unprotected_service_s for job in scenario.jobs
            }
            for action in self.actions:
                if action.action_id not in valid_selected:
                    continue
                for effect in action.effects:
                    if effect.failure_domain == scenario.failure_domain:
                        scalar_service[effect.job_id] = min(
                            scalar_service[effect.job_id],
                            effect.protected_service_s,
                        )
            resolved_jobs: list[RecoveryJob] = []
            for job in scenario.jobs:
                problem = problem_by_job.get(job.job_id)
                service = scalar_service[job.job_id]
                if problem is not None:
                    service = problem.solve(
                        enabled_action_ids=(
                            valid_selected & problem.action_ids
                        )
                    ).total_service_upper_s
                resolved_jobs.append(
                    RecoveryJob(
                        job.job_id,
                        job.workflow_id,
                        job.target_id,
                        max(job.slack_lcb_s, fail_offset_s),
                        service,
                        job.release_offset_s + fail_offset_s,
                    )
                )
            instance = RecoveryLaxityInstance(
                failure_domain=scenario.failure_domain,
                targets=tuple(
                    RecoveryTarget(
                        target.target_id,
                        target.availability_offset_s + fail_offset_s,
                    )
                    for target in scenario.targets
                ),
                jobs=tuple(resolved_jobs),
                actions=(),
            )
            schedule = instance.schedule(frozenset())
            domain_schedules.append((scenario.failure_domain, schedule))

        risk = RecoveryRisk(
            maximum_tardiness_s=max(
                schedule.risk.maximum_tardiness_s
                for _domain, schedule in domain_schedules
            ),
            total_workflow_overrun_s=sum(
                schedule.risk.total_workflow_overrun_s
                for _domain, schedule in domain_schedules
            ),
        )
        return PortfolioOutcome(
            selected_actions=tuple(sorted(selected)),
            risk=risk,
            domain_schedules=tuple(domain_schedules),
        )

    def _release_closure(
        self, selected: frozenset[str], root: str
    ) -> frozenset[str]:
        closure = {root}
        changed = True
        while changed:
            changed = False
            for action_id in selected - closure:
                if set(self.action_by_id[action_id].predecessors) & closure:
                    closure.add(action_id)
                    changed = True
        return frozenset(closure)

    def _cover_victims(
        self,
        selected: frozenset[str],
        additions: frozenset[str],
        required_selected: frozenset[str],
        *,
        fail_offset_s: float,
    ) -> frozenset[str] | None:
        working = selected
        victims: frozenset[str] = frozenset()
        while not self.fits(working | additions):
            used = dict(self.used_resources(working))
            demand = {budget.resource_id: 0.0 for budget in self.budgets}
            for action_id in additions:
                for item in self.action_by_id[action_id].demands:
                    demand[item.resource_id] += item.amount
            deficit = {
                resource_id: max(
                    0.0,
                    used[resource_id]
                    + demand.get(resource_id, 0.0)
                    - budget.capacity,
                )
                for resource_id, budget in self.budget_by_id.items()
            }
            before = self.evaluate(
                working, fail_offset_s=fail_offset_s
            ).risk
            candidates: list[
                tuple[tuple[float, float, tuple[str, ...]], frozenset[str]]
            ] = []
            for root in sorted(working):
                group = self._release_closure(working, root)
                if group & required_selected:
                    continue
                if any(
                    not self.action_by_id[action_id].releasable
                    for action_id in group
                ):
                    continue
                after_selected = working - group
                after = self.evaluate(
                    after_selected, fail_offset_s=fail_offset_s
                ).risk
                before_used = dict(self.used_resources(working))
                after_used = dict(self.used_resources(after_selected))
                cover = sum(
                    min(
                        before_used[resource_id] - after_used[resource_id],
                        deficit[resource_id],
                    )
                    / budget.capacity
                    for resource_id, budget in self.budget_by_id.items()
                )
                if cover <= 1e-12:
                    continue
                candidates.append(
                    (
                        (
                            (
                                after.maximum_tardiness_s
                                - before.maximum_tardiness_s
                            )
                            / cover,
                            (
                                after.total_workflow_overrun_s
                                - before.total_workflow_overrun_s
                            )
                            / cover,
                            tuple(sorted(group)),
                        ),
                        group,
                    )
                )
            if not candidates:
                return None
            _key, chosen = min(candidates, key=lambda row: row[0])
            working = working - chosen
            victims = victims | chosen
        return victims

    def _proposal(
        self,
        selected: frozenset[str],
        action: PortfolioAction,
        *,
        allow_swaps: bool,
    ) -> _Proposal | None:
        closure = self._dependency_closure(action.action_id)
        additions = closure - selected
        required_selected = closure & selected
        completion_offset_s = max(
            self.action_by_id[action_id].completion_offset_s
            for action_id in additions
        )
        if any(
            bundled.valid_for_s is not None
            and completion_offset_s > bundled.valid_for_s + 1e-12
            for bundled in (
                self.action_by_id[action_id] for action_id in closure
            )
        ):
            return None
        fail_offset_s = completion_offset_s
        before = self.evaluate(selected, fail_offset_s=fail_offset_s)
        with_action = selected | additions
        victims: frozenset[str] = frozenset()
        if not self.fits(with_action):
            if not allow_swaps:
                return None
            covered = self._cover_victims(
                selected,
                additions,
                required_selected,
                fail_offset_s=fail_offset_s,
            )
            if covered is None:
                return None
            victims = covered
            with_action = (selected - victims) | additions
        try:
            self._validate_selected(with_action)
        except ValueError:
            return None
        if not self.fits(with_action):
            return None
        if victims:
            # A terminal set difference is not a physical atomic swap.  This
            # compact reference has no make-before-break exchange headroom, so
            # it admits only victim sets whose break-before-make phase does not
            # worsen the robust objective at the common commit horizon.
            # Production additionally enumerates bounded reclaim/commit/abort
            # endpoints; partial candidate bytes always retain zero value.
            transient = self.evaluate(
                selected - victims, fail_offset_s=fail_offset_s
            )
            if transient.risk.key > before.risk.key:
                return None
        after = self.evaluate(with_action, fail_offset_s=fail_offset_s)
        gain = before.risk.gain_to(after.risk)
        if not gain.is_positive():
            return None
        return _Proposal(
            action=action,
            additions=additions,
            victims=victims,
            selected_after=with_action,
            outcome_before=before,
            outcome_after=after,
            gain=gain,
            dominant_share=self.dominant_share_for(additions),
        )

    def greedy(
        self,
        *,
        initial_selected: frozenset[str] = frozenset(),
        allow_swaps: bool = True,
    ) -> PortfolioGreedyResult:
        self._validate_selected(initial_selected)
        if not self.fits(initial_selected):
            raise ValueError("initial_selected exceeds a resource budget")
        selected = initial_selected
        current = self.evaluate(selected)
        steps: list[PortfolioGreedyStep] = []
        visited = {selected}
        while True:
            proposals: list[
                tuple[tuple[float, float, int, str], _Proposal]
            ] = []
            for action in self.actions:
                if action.action_id in selected:
                    continue
                proposal = self._proposal(
                    selected,
                    action,
                    allow_swaps=allow_swaps,
                )
                if proposal is None or proposal.selected_after in visited:
                    continue
                proposals.append(
                    (
                        (
                            -proposal.gain.maximum_tardiness_reduction_s
                            / proposal.dominant_share,
                            -proposal.gain.workflow_overrun_reduction_s
                            / proposal.dominant_share,
                            len(proposal.victims),
                            action.action_id,
                        ),
                        proposal,
                    )
                )
            if not proposals:
                break
            _key, chosen = min(proposals, key=lambda row: row[0])
            selected = chosen.selected_after
            current = self.evaluate(selected)
            visited.add(selected)
            steps.append(
                PortfolioGreedyStep(
                    step=len(steps) + 1,
                    added_action_id=chosen.action.action_id,
                    added_action_ids=tuple(sorted(chosen.additions)),
                    removed_action_ids=tuple(sorted(chosen.victims)),
                    completion_offset_s=max(
                        self.action_by_id[action_id].completion_offset_s
                        for action_id in chosen.additions
                    ),
                    dominant_share=chosen.dominant_share,
                    maximum_tardiness_density=(
                        chosen.gain.maximum_tardiness_reduction_s
                        / chosen.dominant_share
                    ),
                    workflow_overrun_density=(
                        chosen.gain.workflow_overrun_reduction_s
                        / chosen.dominant_share
                    ),
                    risk_before=chosen.outcome_before.risk.key,
                    risk_after=chosen.outcome_after.risk.key,
                    used_resources_after=self.used_resources(selected),
                )
            )
        return PortfolioGreedyResult(
            selected_actions=tuple(sorted(selected)),
            outcome=current,
            used_resources=self.used_resources(selected),
            steps=tuple(steps),
        )

    def solve_exact(self, *, max_actions: int = 20) -> PortfolioExactResult:
        if any(action.completion_offset_s > 0.0 for action in self.actions):
            raise ValueError(
                "the static exact oracle requires already-committed actions "
                "with completion_offset_s == 0"
            )
        if len(self.actions) > max_actions:
            raise ValueError(
                f"exact enumeration is limited to {max_actions} actions"
            )
        started = time.perf_counter()
        action_ids = tuple(sorted(self.action_by_id))
        best_key: tuple[object, ...] | None = None
        best_selected: frozenset[str] | None = None
        best_outcome: PortfolioOutcome | None = None
        enumerated = 0
        feasible = 0
        for count in range(len(action_ids) + 1):
            for subset in itertools.combinations(action_ids, count):
                enumerated += 1
                selected = frozenset(subset)
                try:
                    self._validate_selected(selected)
                except ValueError:
                    continue
                if not self.fits(selected):
                    continue
                feasible += 1
                outcome = self.evaluate(selected)
                used = dict(self.used_resources(selected))
                normalized_use = sum(
                    used[resource_id] / budget.capacity
                    for resource_id, budget in self.budget_by_id.items()
                )
                key = (
                    outcome.risk.maximum_tardiness_s,
                    outcome.risk.total_workflow_overrun_s,
                    normalized_use,
                    tuple(sorted(selected)),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_selected = selected
                    best_outcome = outcome
        if best_selected is None or best_outcome is None:
            raise RuntimeError("instance has no feasible action subset")
        objective = best_outcome.risk.key
        return PortfolioExactResult(
            selected_actions=tuple(sorted(best_selected)),
            outcome=best_outcome,
            used_resources=self.used_resources(best_selected),
            certificate=PortfolioExactCertificate(
                status="OPTIMAL",
                objective=objective,
                best_bound=objective,
                gap=0.0,
                solve_time_s=max(time.perf_counter() - started, 0.0),
                enumerated_subsets=enumerated,
                feasible_subsets=feasible,
                instance_sha256=self.sha256,
            ),
        )
