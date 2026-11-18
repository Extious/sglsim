"""Small exact model for AegisKV's workflow-harm objective.

This module is deliberately separate from the online SGLang-parity runtime.  It
is a falsification/oracle model for tiny protection windows: every policy sees
the same actions and resource budget, and every selected set is measured by the
same joint shadow schedule.  The model is not a claim that a single-server
queue is a faithful GPU serving simulator; calibrated SGLSim and real-system
experiments remain necessary for that claim.

The exact scheduler jointly chooses a retry target for requests interrupted by
one failure domain and interleaves recovery jobs with the target's existing
queue.  Existing (bystander) jobs retain their FIFO order.  Workflow completion
is the maximum active-branch completion plus its residual tail, so benefits are
not summed independently across requests in the same workflow.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal


PolicyName = Literal[
    "load_size",
    "criticality",
    "request_slack",
    "aegis_joint",
]


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class Harm:
    """Lexicographic workflow harm: incremental SLO tardiness, then WCT."""

    tardiness_s: float = 0.0
    wct_s: float = 0.0

    def __post_init__(self) -> None:
        _finite_nonnegative("tardiness_s", self.tardiness_s)
        _finite_nonnegative("wct_s", self.wct_s)

    @property
    def key(self) -> tuple[float, float]:
        return (self.tardiness_s, self.wct_s)

    def scaled(self, weight: float) -> Harm:
        _finite_nonnegative("weight", weight)
        return Harm(self.tardiness_s * weight, self.wct_s * weight)

    def plus(self, other: Harm) -> Harm:
        return Harm(
            self.tardiness_s + other.tardiness_s,
            self.wct_s + other.wct_s,
        )

    def gain_to(self, after: Harm) -> LexicographicGain:
        return LexicographicGain(
            tardiness_reduction_s=self.tardiness_s - after.tardiness_s,
            wct_reduction_s=self.wct_s - after.wct_s,
        )


@dataclass(frozen=True)
class LexicographicGain:
    """Signed component deltas for a lexicographic objective.

    A strict tardiness reduction remains beneficial even if the WCT tie-break
    increases.  Treating both reductions as independently nonnegative would
    silently turn the lexicographic contract into a Pareto constraint.
    """

    tardiness_reduction_s: float
    wct_reduction_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.tardiness_reduction_s) or not math.isfinite(
            self.wct_reduction_s
        ):
            raise ValueError("lexicographic gain components must be finite")

    def is_positive(self, *, atol: float = 1e-9) -> bool:
        if self.tardiness_reduction_s > atol:
            return True
        if self.tardiness_reduction_s < -atol:
            return False
        return self.wct_reduction_s > atol


@dataclass(frozen=True)
class WorkflowState:
    workflow_id: str
    healthy_finish_s: float
    unaffected_finish_s: float = 0.0
    deadline_s: float | None = None
    fairness_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.workflow_id:
            raise ValueError("workflow_id must be nonempty")
        _finite_nonnegative("healthy_finish_s", self.healthy_finish_s)
        _finite_nonnegative("unaffected_finish_s", self.unaffected_finish_s)
        if self.deadline_s is not None:
            _finite_nonnegative("deadline_s", self.deadline_s)
        if not math.isfinite(self.fairness_weight) or self.fairness_weight <= 0.0:
            raise ValueError("fairness_weight must be finite and positive")


@dataclass(frozen=True)
class TargetState:
    target_id: str
    head_block_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must be nonempty")
        _finite_nonnegative("head_block_s", self.head_block_s)


@dataclass(frozen=True)
class TargetOption:
    target_id: str
    resume_service_s: float
    unprotected_recovery_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must be nonempty")
        _finite_nonnegative("resume_service_s", self.resume_service_s)
        _finite_nonnegative(
            "unprotected_recovery_s", self.unprotected_recovery_s
        )


@dataclass(frozen=True)
class ShadowJob:
    """An active branch in the short failure-recovery horizon.

    ``source_domain`` equal to a scenario's failed domain marks an interrupted
    request.  Other jobs are healthy bystanders and must have exactly one target
    option.  ``queue_rank`` preserves FIFO order among bystanders at one target;
    recovery jobs may be interleaved around them by the exact shadow scheduler.
    """

    job_id: str
    workflow_id: str
    source_domain: str | None
    target_options: tuple[TargetOption, ...]
    tail_s: float = 0.0
    release_s: float = 0.0
    slack_lcb_s: float = 0.0
    criticality: float = 1.0
    queue_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.job_id or not self.workflow_id:
            raise ValueError("job_id and workflow_id must be nonempty")
        if not self.target_options:
            raise ValueError("target_options must be nonempty")
        target_ids = [option.target_id for option in self.target_options]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target_options must have unique targets")
        _finite_nonnegative("tail_s", self.tail_s)
        _finite_nonnegative("release_s", self.release_s)
        _finite_nonnegative("slack_lcb_s", self.slack_lcb_s)
        if not math.isfinite(self.criticality) or not 0.0 <= self.criticality <= 1.0:
            raise ValueError("criticality must be in [0, 1]")
        if self.queue_rank is not None and self.queue_rank < 0:
            raise ValueError("queue_rank must be nonnegative")

    def option(self, target_id: str) -> TargetOption:
        for option in self.target_options:
            if option.target_id == target_id:
                return option
        raise KeyError(f"job {self.job_id} has no target {target_id}")


@dataclass(frozen=True)
class RecoveryEffect:
    failure_domain: str
    job_id: str
    target_id: str
    protected_recovery_s: float

    def __post_init__(self) -> None:
        if not self.failure_domain or not self.job_id or not self.target_id:
            raise ValueError("effect identifiers must be nonempty")
        _finite_nonnegative("protected_recovery_s", self.protected_recovery_s)


@dataclass(frozen=True)
class ProtectionAction:
    action_id: str
    demand: float
    effects: tuple[RecoveryEffect, ...]
    resource_pool: str = "protection"

    def __post_init__(self) -> None:
        if not self.action_id or not self.resource_pool:
            raise ValueError("action_id and resource_pool must be nonempty")
        if not math.isfinite(self.demand) or self.demand <= 0.0:
            raise ValueError("action demand must be finite and positive")
        if not self.effects:
            raise ValueError("action must have at least one recovery effect")
        keys = [
            (effect.failure_domain, effect.job_id, effect.target_id)
            for effect in self.effects
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("action has duplicate recovery effects")


@dataclass(frozen=True)
class ScenarioSchedule:
    failure_domain: str
    selected_actions: tuple[str, ...]
    harm: Harm
    workflow_finish_s: tuple[tuple[str, float], ...]
    assignment: tuple[tuple[str, str], ...]
    target_order: tuple[tuple[str, tuple[str, ...]], ...]
    enumerated_schedules: int


@dataclass(frozen=True)
class ShadowScenario:
    scenario_id: str
    failure_domain: str
    workflows: tuple[WorkflowState, ...]
    targets: tuple[TargetState, ...]
    jobs: tuple[ShadowJob, ...]
    max_enumerated_schedules: int = 2_000_000

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.failure_domain:
            raise ValueError("scenario identifiers must be nonempty")
        if not self.workflows or not self.targets or not self.jobs:
            raise ValueError("scenario workflows, targets, and jobs must be nonempty")
        if self.max_enumerated_schedules <= 0:
            raise ValueError("max_enumerated_schedules must be positive")
        workflow_ids = [workflow.workflow_id for workflow in self.workflows]
        target_ids = [target.target_id for target in self.targets]
        job_ids = [job.job_id for job in self.jobs]
        if len(set(workflow_ids)) != len(workflow_ids):
            raise ValueError("workflow IDs must be unique")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target IDs must be unique")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("job IDs must be unique")
        workflow_set = set(workflow_ids)
        target_set = set(target_ids)
        queue_ranks: set[tuple[str, int]] = set()
        for job in self.jobs:
            if job.workflow_id not in workflow_set:
                raise ValueError(f"job {job.job_id} references an unknown workflow")
            if any(option.target_id not in target_set for option in job.target_options):
                raise ValueError(f"job {job.job_id} references an unknown target")
            if job.source_domain != self.failure_domain:
                if len(job.target_options) != 1:
                    raise ValueError("bystander jobs must be bound to one target")
                if job.queue_rank is None:
                    raise ValueError("bystander jobs require queue_rank")
                key = (job.target_options[0].target_id, job.queue_rank)
                if key in queue_ranks:
                    raise ValueError("bystander queue ranks must be unique per target")
                queue_ranks.add(key)

    @property
    def workflow_by_id(self) -> dict[str, WorkflowState]:
        return {workflow.workflow_id: workflow for workflow in self.workflows}

    @property
    def job_by_id(self) -> dict[str, ShadowJob]:
        return {job.job_id: job for job in self.jobs}

    @property
    def target_by_id(self) -> dict[str, TargetState]:
        return {target.target_id: target for target in self.targets}

    def validate_actions(self, actions: tuple[ProtectionAction, ...]) -> None:
        jobs = self.job_by_id
        for action in actions:
            for effect in action.effects:
                if effect.failure_domain != self.failure_domain:
                    continue
                try:
                    job = jobs[effect.job_id]
                except KeyError as exc:
                    raise ValueError(
                        f"action {action.action_id} references unknown job "
                        f"{effect.job_id} in {self.scenario_id}"
                    ) from exc
                if job.source_domain != self.failure_domain:
                    raise ValueError("protection effects may only cover interrupted jobs")
                option = job.option(effect.target_id)
                if effect.protected_recovery_s > option.unprotected_recovery_s:
                    raise ValueError("protection cannot increase recovery service")

    def _recovery_s(
        self,
        job: ShadowJob,
        target_id: str,
        selected: frozenset[str],
        actions: tuple[ProtectionAction, ...],
    ) -> float:
        recovery = job.option(target_id).unprotected_recovery_s
        for action in actions:
            if action.action_id not in selected:
                continue
            for effect in action.effects:
                if (
                    effect.failure_domain == self.failure_domain
                    and effect.job_id == job.job_id
                    and effect.target_id == target_id
                ):
                    recovery = min(recovery, effect.protected_recovery_s)
        return recovery

    @staticmethod
    def _preserves_bystander_fifo(
        order: tuple[ShadowJob, ...], failure_domain: str
    ) -> bool:
        ranks = [
            int(job.queue_rank)
            for job in order
            if job.source_domain != failure_domain and job.queue_rank is not None
        ]
        return ranks == sorted(ranks)

    def _harm(self, workflow_finish: dict[str, float]) -> Harm:
        tardiness = 0.0
        wct = 0.0
        for workflow in self.workflows:
            finish = workflow_finish[workflow.workflow_id]
            incremental_wct = max(finish - workflow.healthy_finish_s, 0.0)
            incremental_tardiness = 0.0
            if workflow.deadline_s is not None:
                fault_tardiness = max(finish - workflow.deadline_s, 0.0)
                healthy_tardiness = max(
                    workflow.healthy_finish_s - workflow.deadline_s, 0.0
                )
                incremental_tardiness = max(
                    fault_tardiness - healthy_tardiness, 0.0
                )
            tardiness += workflow.fairness_weight * incremental_tardiness
            wct += workflow.fairness_weight * incremental_wct
        return Harm(tardiness, wct)

    def schedule(
        self,
        selected_actions: frozenset[str],
        actions: tuple[ProtectionAction, ...],
    ) -> ScenarioSchedule:
        self.validate_actions(actions)
        known_actions = {action.action_id for action in actions}
        unknown = selected_actions - known_actions
        if unknown:
            raise ValueError(f"unknown selected actions: {sorted(unknown)}")

        affected = [
            job for job in self.jobs if job.source_domain == self.failure_domain
        ]
        bystanders = [
            job for job in self.jobs if job.source_domain != self.failure_domain
        ]
        assignment_choices = [
            tuple(option.target_id for option in job.target_options)
            for job in affected
        ]
        assignments = itertools.product(*assignment_choices)
        targets = self.target_by_id
        workflows = self.workflow_by_id
        best_key: tuple[object, ...] | None = None
        best_payload: tuple[
            Harm,
            dict[str, float],
            tuple[tuple[str, str], ...],
            tuple[tuple[str, tuple[str, ...]], ...],
        ] | None = None
        enumerated = 0

        for target_ids in assignments:
            assignment = {
                job.job_id: target_id
                for job, target_id in zip(affected, target_ids, strict=True)
            }
            for job in bystanders:
                assignment[job.job_id] = job.target_options[0].target_id
            jobs_by_target: dict[str, list[ShadowJob]] = {
                target_id: [] for target_id in targets
            }
            for job in self.jobs:
                jobs_by_target[assignment[job.job_id]].append(job)

            order_options: list[tuple[tuple[ShadowJob, ...], ...]] = []
            target_order_ids = sorted(targets)
            per_target_orders: list[tuple[tuple[ShadowJob, ...], ...]] = []
            for target_id in target_order_ids:
                target_jobs = jobs_by_target[target_id]
                valid_orders = tuple(
                    order
                    for order in itertools.permutations(target_jobs)
                    if self._preserves_bystander_fifo(order, self.failure_domain)
                )
                per_target_orders.append(valid_orders or ((),))
            order_options.append(tuple(per_target_orders))

            for orders in itertools.product(*order_options[0]):
                enumerated += 1
                if enumerated > self.max_enumerated_schedules:
                    raise RuntimeError(
                        f"scenario {self.scenario_id} exceeded exact schedule cap "
                        f"{self.max_enumerated_schedules}"
                    )
                completion: dict[str, float] = {}
                order_rows: list[tuple[str, tuple[str, ...]]] = []
                for target_id, order in zip(target_order_ids, orders, strict=True):
                    clock = targets[target_id].head_block_s
                    for job in order:
                        option = job.option(target_id)
                        clock = max(clock, job.release_s)
                        clock += option.resume_service_s + self._recovery_s(
                            job, target_id, selected_actions, actions
                        )
                        completion[job.job_id] = clock
                    order_rows.append((target_id, tuple(job.job_id for job in order)))

                finish = {
                    workflow_id: workflow.unaffected_finish_s
                    for workflow_id, workflow in workflows.items()
                }
                for job in self.jobs:
                    finish[job.workflow_id] = max(
                        finish[job.workflow_id], completion[job.job_id] + job.tail_s
                    )
                harm = self._harm(finish)
                assignment_row = tuple(sorted(assignment.items()))
                order_row = tuple(order_rows)
                key = (
                    harm.tardiness_s,
                    harm.wct_s,
                    assignment_row,
                    order_row,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_payload = (harm, finish, assignment_row, order_row)

        if best_payload is None:
            raise RuntimeError(f"scenario {self.scenario_id} has no feasible schedule")
        harm, finish, assignment_row, order_row = best_payload
        return ScenarioSchedule(
            failure_domain=self.failure_domain,
            selected_actions=tuple(sorted(selected_actions)),
            harm=harm,
            workflow_finish_s=tuple(sorted(finish.items())),
            assignment=assignment_row,
            target_order=order_row,
            enumerated_schedules=enumerated,
        )


@dataclass(frozen=True)
class WeightedScenario:
    probability: float
    scenario: ShadowScenario

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability) or self.probability <= 0.0:
            raise ValueError("scenario probability must be finite and positive")


@dataclass(frozen=True)
class PortfolioOutcome:
    selected_actions: tuple[str, ...]
    expected_harm: Harm
    scenarios: tuple[ScenarioSchedule, ...]


@dataclass(frozen=True)
class GreedyStep:
    step: int
    action_id: str
    demand: float
    primary_density: float
    secondary_density: float


@dataclass(frozen=True)
class PolicyResult:
    policy: PolicyName
    selected_actions: tuple[str, ...]
    used_budget: float
    true_outcome: PortfolioOutcome
    steps: tuple[GreedyStep, ...]


@dataclass(frozen=True)
class ExactCertificate:
    status: str
    objective: tuple[float, float]
    best_bound: tuple[float, float]
    gap: float
    solve_time_s: float
    enumerated_subsets: int
    enumerated_schedules: int
    instance_sha256: str


@dataclass(frozen=True)
class ExactResult:
    selected_actions: tuple[str, ...]
    used_budget: float
    outcome: PortfolioOutcome
    certificate: ExactCertificate


@dataclass(frozen=True)
class WorkflowHarmPortfolio:
    scenarios: tuple[WeightedScenario, ...]
    actions: tuple[ProtectionAction, ...]

    def __post_init__(self) -> None:
        if not self.scenarios or not self.actions:
            raise ValueError("portfolio scenarios and actions must be nonempty")
        scenario_ids = [entry.scenario.scenario_id for entry in self.scenarios]
        action_ids = [action.action_id for action in self.actions]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario IDs must be unique")
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action IDs must be unique")
        total_probability = sum(entry.probability for entry in self.scenarios)
        if not math.isclose(total_probability, 1.0, abs_tol=1e-9):
            raise ValueError("scenario probabilities must sum to one")
        domains = {entry.scenario.failure_domain for entry in self.scenarios}
        for action in self.actions:
            if any(effect.failure_domain not in domains for effect in action.effects):
                raise ValueError(f"action {action.action_id} references unknown domain")
        for entry in self.scenarios:
            entry.scenario.validate_actions(self.actions)

    @property
    def action_by_id(self) -> dict[str, ProtectionAction]:
        return {action.action_id: action for action in self.actions}

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def used_budget(self, selected: frozenset[str]) -> float:
        actions = self.action_by_id
        unknown = selected - actions.keys()
        if unknown:
            raise ValueError(f"unknown selected actions: {sorted(unknown)}")
        return sum(actions[action_id].demand for action_id in selected)

    def evaluate(self, selected: frozenset[str]) -> PortfolioOutcome:
        expected = Harm()
        schedules: list[ScenarioSchedule] = []
        for entry in self.scenarios:
            schedule = entry.scenario.schedule(selected, self.actions)
            schedules.append(schedule)
            expected = expected.plus(schedule.harm.scaled(entry.probability))
        return PortfolioOutcome(
            selected_actions=tuple(sorted(selected)),
            expected_harm=expected,
            scenarios=tuple(schedules),
        )

    def _surrogate_loss(
        self, policy: PolicyName, selected: frozenset[str]
    ) -> float:
        if policy == "aegis_joint":
            raise ValueError("aegis_joint uses the lexicographic true objective")
        loss = 0.0
        for entry in self.scenarios:
            scenario = entry.scenario
            for job in scenario.jobs:
                if job.source_domain != scenario.failure_domain:
                    continue
                recovery = min(
                    scenario._recovery_s(
                        job, option.target_id, selected, self.actions
                    )
                    for option in job.target_options
                )
                if policy == "load_size":
                    value = recovery
                elif policy == "criticality":
                    value = job.criticality * recovery
                elif policy == "request_slack":
                    value = max(recovery - job.slack_lcb_s, 0.0)
                else:
                    raise ValueError(f"unsupported policy {policy}")
                loss += entry.probability * value
        return loss

    def greedy(self, policy: PolicyName, *, budget: float) -> PolicyResult:
        _finite_nonnegative("budget", budget)
        selected: frozenset[str] = frozenset()
        used = 0.0
        steps: list[GreedyStep] = []
        current_outcome = self.evaluate(selected)
        while True:
            candidates: list[
                tuple[
                    tuple[float, float, str],
                    ProtectionAction,
                    LexicographicGain | float,
                ]
            ] = []
            for action in self.actions:
                if action.action_id in selected or used + action.demand > budget + 1e-12:
                    continue
                after_selected = selected | {action.action_id}
                if policy == "aegis_joint":
                    after_outcome = self.evaluate(after_selected)
                    gain = current_outcome.expected_harm.gain_to(
                        after_outcome.expected_harm
                    )
                    primary = gain.tardiness_reduction_s / action.demand
                    secondary = gain.wct_reduction_s / action.demand
                    payload: LexicographicGain | float = gain
                else:
                    before_loss = self._surrogate_loss(policy, selected)
                    after_loss = self._surrogate_loss(policy, after_selected)
                    scalar_gain = max(before_loss - after_loss, 0.0)
                    primary = scalar_gain / action.demand
                    secondary = 0.0
                    payload = scalar_gain
                # min() below: negate benefits, then stable action ID.
                candidates.append(
                    ((-primary, -secondary, action.action_id), action, payload)
                )
            if not candidates:
                break
            _key, chosen, payload = min(candidates, key=lambda row: row[0])
            if isinstance(payload, LexicographicGain):
                primary = payload.tardiness_reduction_s / chosen.demand
                secondary = payload.wct_reduction_s / chosen.demand
                positive = payload.is_positive()
            else:
                primary = float(payload) / chosen.demand
                secondary = 0.0
                positive = primary > 1e-9
            if not positive:
                break
            selected = selected | {chosen.action_id}
            used += chosen.demand
            current_outcome = self.evaluate(selected)
            steps.append(
                GreedyStep(
                    step=len(steps) + 1,
                    action_id=chosen.action_id,
                    demand=chosen.demand,
                    primary_density=primary,
                    secondary_density=secondary,
                )
            )
        return PolicyResult(
            policy=policy,
            selected_actions=tuple(sorted(selected)),
            used_budget=used,
            true_outcome=current_outcome,
            steps=tuple(steps),
        )

    def solve_exact(self, *, budget: float) -> ExactResult:
        """Exhaust all feasible action subsets and return an exact certificate."""

        _finite_nonnegative("budget", budget)
        started = time.perf_counter()
        best_key: tuple[object, ...] | None = None
        best: tuple[frozenset[str], float, PortfolioOutcome] | None = None
        enumerated_subsets = 0
        enumerated_schedules = 0
        for mask in range(1 << len(self.actions)):
            selected = frozenset(
                action.action_id
                for index, action in enumerate(self.actions)
                if mask & (1 << index)
            )
            used = self.used_budget(selected)
            if used > budget + 1e-12:
                continue
            enumerated_subsets += 1
            outcome = self.evaluate(selected)
            enumerated_schedules += sum(
                schedule.enumerated_schedules for schedule in outcome.scenarios
            )
            key = (
                outcome.expected_harm.tardiness_s,
                outcome.expected_harm.wct_s,
                used,
                tuple(sorted(selected)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (selected, used, outcome)
        if best is None:
            raise RuntimeError("no feasible action subset")
        selected, used, outcome = best
        elapsed = time.perf_counter() - started
        objective = outcome.expected_harm.key
        certificate = ExactCertificate(
            status="OPTIMAL",
            objective=objective,
            best_bound=objective,
            gap=0.0,
            solve_time_s=elapsed,
            enumerated_subsets=enumerated_subsets,
            enumerated_schedules=enumerated_schedules,
            instance_sha256=self.sha256,
        )
        return ExactResult(
            selected_actions=tuple(sorted(selected)),
            used_budget=used,
            outcome=outcome,
            certificate=certificate,
        )
