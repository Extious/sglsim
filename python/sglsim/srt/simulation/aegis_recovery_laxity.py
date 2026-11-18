"""Tiny semantic model for a recovery-deadline AegisKV policy.

This module is deliberately independent from the online SGLang-parity runtime.
It tests a simpler Algorithm-2 contract than the full workflow shadow model:

* Algorithm 1 turns each interrupted branch's conservative structural Slack
  into an absolute recovery due date.  This reference stores timestamps as
  offsets from one fixed failure/trace epoch; target availability and later
  release never recharge that deadline.
* Router supplies a fixed, load-aware recovery-target assignment and a causal
  target-availability offset; Conductor remains an advisory locality index.
* Each target exposes one serial recovery-envelope-bounded lane, distinct from
  the normal-serving floor.  Ready jobs follow ideal preemptive
  earliest-deadline-first (EDF); equal deadlines use shortest remaining
  service and then a stable ID.  Static EDD is the special case where all jobs
  are released together and no later preemption occurs.
* A complete KV action can only reduce one or more jobs' measured/modelled
  recovery service time.  It never changes correctness state until completion.

For one target with known release times and cost-free preemption, EDF is the
classic optimal rule for minimizing maximum lateness.  This ideal model does
not claim that a chunked transport has cost-free preemption, a bounded service
envelope, or an additive one-chunk optimality gap.  Those remain explicit proof
and implementation boundaries in the production design.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class RecoveryTarget:
    target_id: str
    availability_offset_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must be nonempty")
        _finite_nonnegative(
            "availability_offset_s", self.availability_offset_s
        )


@dataclass(frozen=True)
class RecoveryJob:
    job_id: str
    workflow_id: str
    target_id: str
    slack_lcb_s: float
    unprotected_service_s: float
    release_offset_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.job_id or not self.workflow_id or not self.target_id:
            raise ValueError("job identifiers must be nonempty")
        _finite_nonnegative("slack_lcb_s", self.slack_lcb_s)
        _finite_nonnegative(
            "unprotected_service_s", self.unprotected_service_s
        )
        _finite_nonnegative("release_offset_s", self.release_offset_s)

    @property
    def absolute_deadline_s(self) -> float:
        """Absolute due time in the reference epoch.

        ``slack_lcb_s`` is retained as the constructor field for compatibility
        with the first semantic model; it is not a duration that is added again
        at release or route reassignment.
        """

        return self.slack_lcb_s


@dataclass(frozen=True)
class RecoveryEffect:
    job_id: str
    protected_service_s: float

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("effect job_id must be nonempty")
        _finite_nonnegative("protected_service_s", self.protected_service_s)


@dataclass(frozen=True)
class ProtectionAction:
    action_id: str
    demand: float
    effects: tuple[RecoveryEffect, ...]

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be nonempty")
        if not math.isfinite(float(self.demand)) or float(self.demand) <= 0.0:
            raise ValueError("action demand must be finite and positive")
        if not self.effects:
            raise ValueError("action must have at least one effect")
        job_ids = [effect.job_id for effect in self.effects]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("action effects must have unique job IDs")


@dataclass(frozen=True)
class RecoveryRisk:
    """Lexicographic risk: worst branch tardiness, then workflow overrun sum."""

    maximum_tardiness_s: float
    total_workflow_overrun_s: float

    def __post_init__(self) -> None:
        _finite_nonnegative("maximum_tardiness_s", self.maximum_tardiness_s)
        _finite_nonnegative(
            "total_workflow_overrun_s", self.total_workflow_overrun_s
        )

    @property
    def key(self) -> tuple[float, float]:
        return (self.maximum_tardiness_s, self.total_workflow_overrun_s)

    def gain_to(self, after: "RecoveryRisk") -> "RecoveryRiskGain":
        return RecoveryRiskGain(
            maximum_tardiness_reduction_s=(
                self.maximum_tardiness_s - after.maximum_tardiness_s
            ),
            workflow_overrun_reduction_s=(
                self.total_workflow_overrun_s
                - after.total_workflow_overrun_s
            ),
        )


@dataclass(frozen=True)
class RecoveryRiskGain:
    maximum_tardiness_reduction_s: float
    workflow_overrun_reduction_s: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_tardiness_reduction_s)
            or not math.isfinite(self.workflow_overrun_reduction_s)
        ):
            raise ValueError("risk gain components must be finite")

    def is_positive(self, *, atol: float = 1e-9) -> bool:
        if self.maximum_tardiness_reduction_s > atol:
            return True
        if self.maximum_tardiness_reduction_s < -atol:
            return False
        return self.workflow_overrun_reduction_s > atol


@dataclass(frozen=True)
class RecoverySchedule:
    selected_actions: tuple[str, ...]
    risk: RecoveryRisk
    target_order: tuple[tuple[str, tuple[str, ...]], ...]
    target_timeline: tuple[
        tuple[str, tuple[tuple[str, float, float], ...]], ...
    ]
    completion_s: tuple[tuple[str, float], ...]
    lateness_s: tuple[tuple[str, float], ...]
    workflow_overrun_s: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RecoveryExecutionSlice:
    """An immutable piece of the history preceding a live input update."""

    job_id: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("executed-prefix job_id must be nonempty")
        _finite_nonnegative("start_s", self.start_s)
        _finite_nonnegative("end_s", self.end_s)
        if self.end_s <= self.start_s:
            raise ValueError("executed-prefix slice must have positive duration")


@dataclass(frozen=True)
class VersionedRecoverySuffixSchedule:
    """A schedule whose comparator is conditioned on one executed prefix."""

    route_version: int
    due_version: int
    service_version: int
    linearization_s: float
    initial_blocking_upper_s: float
    executed_prefix: tuple[RecoveryExecutionSlice, ...]
    suffix: RecoverySchedule


def schedule_versioned_suffix(
    *,
    failure_domain: str,
    target: RecoveryTarget,
    remaining_jobs: tuple[RecoveryJob, ...],
    executed_prefix: tuple[RecoveryExecutionSlice, ...],
    linearization_s: float,
    route_version: int,
    due_version: int,
    service_version: int,
    initial_blocking_upper_s: float = 0.0,
    old_contract_fenced_to_terminal: bool = True,
    max_quantum_s: float | None = None,
) -> VersionedRecoverySuffixSchedule:
    """Atomically re-key a lane and schedule only its remaining-work suffix.

    A route, live-deadline, or service-envelope change ends the previous EDF
    theorem interval.  This helper deliberately cannot rewrite the executed
    prefix or compare against a clairvoyant whole-history schedule.  At most one
    already-issued quantum may contribute ``initial_blocking_upper_s``; if the
    old contract no longer fences that quantum to a finite terminal bound, the
    new interval fails closed instead of claiming theorem mode.

    ``remaining_jobs`` already contains the rebound remaining service and new
    absolute deadlines.  It is intentionally a snapshot API rather than a
    future-state predictor.
    """

    if not failure_domain:
        raise ValueError("failure_domain must be nonempty")
    _finite_nonnegative("linearization_s", linearization_s)
    _finite_nonnegative(
        "initial_blocking_upper_s", initial_blocking_upper_s
    )
    for name, version in (
        ("route_version", route_version),
        ("due_version", due_version),
        ("service_version", service_version),
    ):
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if initial_blocking_upper_s > 0.0 and not old_contract_fenced_to_terminal:
        raise ValueError("in-flight quantum lacks an old-contract terminal fence")
    if not remaining_jobs:
        raise ValueError("remaining_jobs must be nonempty")
    if any(job.target_id != target.target_id for job in remaining_jobs):
        raise ValueError("remaining jobs must stay on the versioned target")

    previous_end = 0.0
    for index, item in enumerate(executed_prefix):
        if index and item.start_s < previous_end - 1e-12:
            raise ValueError("executed prefix must be ordered and nonoverlapping")
        if item.end_s > linearization_s + 1e-12:
            raise ValueError("executed prefix crosses the update linearization")
        previous_end = item.end_s

    rebound_target = RecoveryTarget(
        target_id=target.target_id,
        availability_offset_s=max(
            target.availability_offset_s,
            linearization_s + initial_blocking_upper_s,
        ),
    )
    instance = RecoveryLaxityInstance(
        failure_domain=failure_domain,
        targets=(rebound_target,),
        jobs=remaining_jobs,
        actions=(),
    )
    suffix = (
        instance.schedule(frozenset())
        if max_quantum_s is None
        else instance.schedule_chunked(
            frozenset(), max_quantum_s=max_quantum_s
        )
    )
    return VersionedRecoverySuffixSchedule(
        route_version=route_version,
        due_version=due_version,
        service_version=service_version,
        linearization_s=linearization_s,
        initial_blocking_upper_s=initial_blocking_upper_s,
        executed_prefix=executed_prefix,
        suffix=suffix,
    )


@dataclass(frozen=True)
class GreedyStep:
    step: int
    action_id: str
    demand: float
    maximum_tardiness_density: float
    workflow_overrun_density: float


@dataclass(frozen=True)
class GreedyResult:
    selected_actions: tuple[str, ...]
    used_budget: float
    schedule: RecoverySchedule
    steps: tuple[GreedyStep, ...]


@dataclass(frozen=True)
class ExactCertificate:
    status: str
    objective: tuple[float, float]
    best_bound: tuple[float, float]
    gap: float
    solve_time_s: float
    enumerated_subsets: int
    instance_sha256: str


@dataclass(frozen=True)
class ExactResult:
    selected_actions: tuple[str, ...]
    used_budget: float
    schedule: RecoverySchedule
    certificate: ExactCertificate


@dataclass(frozen=True)
class RecoveryLaxityInstance:
    failure_domain: str
    targets: tuple[RecoveryTarget, ...]
    jobs: tuple[RecoveryJob, ...]
    actions: tuple[ProtectionAction, ...]

    def __post_init__(self) -> None:
        if not self.failure_domain:
            raise ValueError("failure_domain must be nonempty")
        if not self.targets or not self.jobs:
            raise ValueError("targets and jobs must be nonempty")
        target_ids = [target.target_id for target in self.targets]
        job_ids = [job.job_id for job in self.jobs]
        action_ids = [action.action_id for action in self.actions]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target IDs must be unique")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("job IDs must be unique")
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action IDs must be unique")
        target_set = set(target_ids)
        job_by_id = {job.job_id: job for job in self.jobs}
        for job in self.jobs:
            if job.target_id not in target_set:
                raise ValueError(f"job {job.job_id} references an unknown target")
        for action in self.actions:
            for effect in action.effects:
                if effect.job_id not in job_by_id:
                    raise ValueError(
                        f"action {action.action_id} references unknown job "
                        f"{effect.job_id}"
                    )
                if (
                    effect.protected_service_s
                    > job_by_id[effect.job_id].unprotected_service_s + 1e-12
                ):
                    raise ValueError("protection cannot increase recovery service")

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
        unknown = selected - self.action_by_id.keys()
        if unknown:
            raise ValueError(f"unknown selected actions: {sorted(unknown)}")
        return sum(self.action_by_id[action_id].demand for action_id in selected)

    def _service_s(self, job: RecoveryJob, selected: frozenset[str]) -> float:
        service = job.unprotected_service_s
        for action in self.actions:
            if action.action_id not in selected:
                continue
            for effect in action.effects:
                if effect.job_id == job.job_id:
                    service = min(service, effect.protected_service_s)
        return service

    def schedule(self, selected: frozenset[str]) -> RecoverySchedule:
        """Build the ideal, cost-free-preemption EDF schedule."""

        return self._schedule(selected, max_quantum_s=None)

    def schedule_chunked(
        self,
        selected: frozenset[str],
        *,
        max_quantum_s: float,
    ) -> RecoverySchedule:
        """Build strict single-lane EDF with preemption only at boundaries.

        The caller supplies the maximum end-to-end quantum duration.  This
        semantic helper assumes one work-conserving lane, no cross-job stage
        overlap, no discarded work, and no setup outside the recorded quantum.
        A job's entire remaining work is released at its original release time;
        quantum boundaries only limit preemption and never create a new release.
        Under exactly those premises, extended Jackson/Schrage critical-block
        analysis gives an additive ``max_quantum_s`` maximum-lateness bound
        relative to ideal preemptive EDF.
        """

        quantum = float(max_quantum_s)
        if not math.isfinite(quantum) or quantum <= 0.0:
            raise ValueError("max_quantum_s must be finite and positive")
        return self._schedule(selected, max_quantum_s=quantum)

    def _schedule(
        self,
        selected: frozenset[str],
        *,
        max_quantum_s: float | None,
    ) -> RecoverySchedule:
        unknown = selected - self.action_by_id.keys()
        if unknown:
            raise ValueError(f"unknown selected actions: {sorted(unknown)}")
        completion: dict[str, float] = {}
        signed_lateness: dict[str, float] = {}
        target_order: list[tuple[str, tuple[str, ...]]] = []
        target_timeline: list[
            tuple[str, tuple[tuple[str, float, float], ...]]
        ] = []
        for target in sorted(self.targets, key=lambda item: item.target_id):
            jobs = [job for job in self.jobs if job.target_id == target.target_id]
            pending = sorted(
                jobs,
                key=lambda job: (
                    max(target.availability_offset_s, job.release_offset_s),
                    job.job_id,
                ),
            )
            remaining = {
                job.job_id: self._service_s(job, selected) for job in jobs
            }
            ready: list[tuple[float, float, str, RecoveryJob]] = []
            dispatch_order: list[str] = []
            timeline: list[tuple[str, float, float]] = []
            next_pending = 0
            clock = target.availability_offset_s

            while next_pending < len(pending) or ready:
                if not ready and next_pending < len(pending):
                    clock = max(
                        clock,
                        target.availability_offset_s,
                        pending[next_pending].release_offset_s,
                    )

                while next_pending < len(pending):
                    job = pending[next_pending]
                    release = max(
                        target.availability_offset_s, job.release_offset_s
                    )
                    if release > clock + 1e-12:
                        break
                    heapq.heappush(
                        ready,
                        (
                            job.absolute_deadline_s,
                            remaining[job.job_id],
                            job.job_id,
                            job,
                        ),
                    )
                    next_pending += 1

                if not ready:
                    continue

                _deadline, _remaining_key, _job_id, job = heapq.heappop(ready)
                service_left = remaining[job.job_id]
                if max_quantum_s is None:
                    next_release = math.inf
                    if next_pending < len(pending):
                        next_release = max(
                            target.availability_offset_s,
                            pending[next_pending].release_offset_s,
                        )
                    run_for = min(
                        service_left, max(next_release - clock, 0.0)
                    )
                    if math.isinf(next_release):
                        run_for = service_left
                else:
                    run_for = min(service_left, max_quantum_s)
                if run_for <= 1e-12 and service_left > 1e-12:
                    heapq.heappush(
                        ready,
                        (
                            job.absolute_deadline_s,
                            service_left,
                            job.job_id,
                            job,
                        ),
                    )
                    continue

                start = clock
                clock += run_for
                service_left = max(service_left - run_for, 0.0)
                remaining[job.job_id] = service_left
                dispatch_order.append(job.job_id)
                timeline.append((job.job_id, start, clock))

                if service_left <= 1e-12:
                    completion[job.job_id] = clock
                    signed_lateness[job.job_id] = (
                        clock - job.absolute_deadline_s
                    )
                else:
                    heapq.heappush(
                        ready,
                        (
                            job.absolute_deadline_s,
                            service_left,
                            job.job_id,
                            job,
                        ),
                    )
            target_order.append(
                (target.target_id, tuple(dispatch_order))
            )
            target_timeline.append((target.target_id, tuple(timeline)))

        workflow_overrun: dict[str, float] = {}
        for job in self.jobs:
            workflow_overrun[job.workflow_id] = max(
                workflow_overrun.get(job.workflow_id, 0.0),
                max(signed_lateness[job.job_id], 0.0),
            )
        risk = RecoveryRisk(
            maximum_tardiness_s=max(
                (max(value, 0.0) for value in signed_lateness.values()),
                default=0.0,
            ),
            total_workflow_overrun_s=sum(workflow_overrun.values()),
        )
        return RecoverySchedule(
            selected_actions=tuple(sorted(selected)),
            risk=risk,
            target_order=tuple(target_order),
            target_timeline=tuple(target_timeline),
            completion_s=tuple(sorted(completion.items())),
            lateness_s=tuple(sorted(signed_lateness.items())),
            workflow_overrun_s=tuple(sorted(workflow_overrun.items())),
        )

    def greedy(self, *, budget: float) -> GreedyResult:
        _finite_nonnegative("budget", budget)
        selected: frozenset[str] = frozenset()
        used = 0.0
        current = self.schedule(selected)
        steps: list[GreedyStep] = []
        while True:
            candidates: list[
                tuple[tuple[float, float, str], ProtectionAction, RecoveryRiskGain]
            ] = []
            for action in self.actions:
                if action.action_id in selected:
                    continue
                if used + action.demand > budget + 1e-12:
                    continue
                after = self.schedule(selected | {action.action_id})
                gain = current.risk.gain_to(after.risk)
                candidates.append(
                    (
                        (
                            -gain.maximum_tardiness_reduction_s / action.demand,
                            -gain.workflow_overrun_reduction_s / action.demand,
                            action.action_id,
                        ),
                        action,
                        gain,
                    )
                )
            if not candidates:
                break
            _key, action, gain = min(candidates, key=lambda row: row[0])
            if not gain.is_positive():
                break
            selected = selected | {action.action_id}
            used += action.demand
            current = self.schedule(selected)
            steps.append(
                GreedyStep(
                    step=len(steps) + 1,
                    action_id=action.action_id,
                    demand=action.demand,
                    maximum_tardiness_density=(
                        gain.maximum_tardiness_reduction_s / action.demand
                    ),
                    workflow_overrun_density=(
                        gain.workflow_overrun_reduction_s / action.demand
                    ),
                )
            )
        return GreedyResult(
            selected_actions=tuple(sorted(selected)),
            used_budget=used,
            schedule=current,
            steps=tuple(steps),
        )

    def solve_exact(self, *, budget: float) -> ExactResult:
        _finite_nonnegative("budget", budget)
        started = time.perf_counter()
        best_key: tuple[object, ...] | None = None
        best_selected: frozenset[str] | None = None
        best_schedule: RecoverySchedule | None = None
        enumerated = 0
        action_ids = tuple(sorted(self.action_by_id))
        for count in range(len(action_ids) + 1):
            for subset in itertools.combinations(action_ids, count):
                selected = frozenset(subset)
                if self.used_budget(selected) > budget + 1e-12:
                    continue
                enumerated += 1
                schedule = self.schedule(selected)
                key = (
                    schedule.risk.maximum_tardiness_s,
                    schedule.risk.total_workflow_overrun_s,
                    self.used_budget(selected),
                    tuple(sorted(selected)),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_selected = selected
                    best_schedule = schedule
        if best_selected is None or best_schedule is None:
            raise RuntimeError("instance has no feasible action subset")
        objective = best_schedule.risk.key
        return ExactResult(
            selected_actions=tuple(sorted(best_selected)),
            used_budget=self.used_budget(best_selected),
            schedule=best_schedule,
            certificate=ExactCertificate(
                status="OPTIMAL",
                objective=objective,
                best_bound=objective,
                gap=0.0,
                solve_time_s=max(time.perf_counter() - started, 0.0),
                enumerated_subsets=enumerated,
                instance_sha256=self.sha256,
            ),
        )
