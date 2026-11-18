"""Deterministic parent event coordination for offline DP x PP simulation.

This module deliberately does not contain scheduling, AegisKV policy, cache,
or failure semantics.  Those remain in the existing runtime components.  It
only orders external events and one in-flight batch completion per physical PP
stage.  Each ``(dp_rank, pp_rank)`` stage owns its own
:class:`LogicalClockDriver`; the parent timestamp is an event-merge cursor and
is never pushed into unrelated stages.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from enum import IntEnum
from typing import Any, Mapping

from sglsim.srt.simulation.global_des import EventPriority, GlobalDES, GlobalEvent
from sglsim.srt.simulation.time_driver import LogicalClockDriver


_NANOSECONDS_PER_SECOND = 1_000_000_000


def _seconds_to_nanoseconds(value: float, *, name: str) -> int:
    """Quantize nonnegative seconds to the first causal nanosecond.

    ``Decimal(str(value))`` avoids platform-dependent binary-float
    multiplication.  Rounding toward positive infinity guarantees that an
    event is never dispatched earlier than the seconds timestamp supplied by
    the caller.
    """

    try:
        seconds = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not seconds.is_finite() or seconds < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return int(
        (seconds * _NANOSECONDS_PER_SECOND).to_integral_value(
            rounding=ROUND_CEILING
        )
    )


def _nanoseconds_to_seconds(value: int) -> float:
    return int(value) / _NANOSECONDS_PER_SECOND


class OfflineEventPriority(IntEnum):
    """Offline names mapped onto the authoritative global tie order."""

    FAILURE_CONTROL = int(EventPriority.FAILURE)
    RANK_COMPLETION = int(EventPriority.COMPUTE_COMPLETION)
    REQUEST_ARRIVAL = int(EventPriority.REQUEST_ARRIVAL)
    COORDINATION = int(EventPriority.CONTROL)


class OfflineCausalBarrierError(RuntimeError):
    """Raised when the parent would outrun unpublished rank work."""


@dataclass(order=True, frozen=True)
class OfflineEvent:
    timestamp_s: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    event_id: str = field(compare=False)
    dp_rank: int | None = field(default=None, compare=False)
    pp_rank: int | None = field(default=None, compare=False)
    payload: Any = field(default=None, compare=False)


@dataclass(frozen=True)
class OfflineRankWorkBarrier:
    """Conservative lower bound for one dispatched Scheduler command.

    A worker command may mutate rank-local state and publish a new completion
    event.  Until its response is received, the parent cannot know that next
    event's timestamp.  The lower bound prevents the global cursor from
    advancing beyond the command's dispatch time in that interval.
    """

    barrier_id: str
    dp_rank: int
    pp_rank: int
    lower_bound_ns: int
    operation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def lower_bound_s(self) -> float:
        return _nanoseconds_to_seconds(self.lower_bound_ns)


class OfflineEventCoordinator:
    """Global-DES adapter with independent stage clocks and no max-sync.

    ``GlobalDES`` owns the only event heap and the only authoritative parent
    cursor.  ``stage_drivers`` remain independent clocks used by the mirrored
    SGLang schedulers; popping an event never pushes unrelated stages.
    """

    def __init__(
        self,
        *,
        dp_size: int,
        pp_size: int = 1,
        start_s: float = 0.0,
        global_des: GlobalDES | None = None,
    ) -> None:
        if int(dp_size) <= 0:
            raise ValueError("dp_size must be positive")
        if int(pp_size) <= 0:
            raise ValueError("pp_size must be positive")
        self._dp_size = int(dp_size)
        self._pp_size = int(pp_size)
        start_ns = _seconds_to_nanoseconds(start_s, name="start_s")
        if global_des is None:
            self.global_des = GlobalDES(start_ns=start_ns)
        else:
            if global_des.now_ns != start_ns:
                raise ValueError(
                    "start_s must match the supplied GlobalDES cursor "
                    f"({start_ns} != {global_des.now_ns} ns)"
                )
            self.global_des = global_des
        quantized_start_s = _nanoseconds_to_seconds(start_ns)
        self.stage_drivers = {
            (dp_rank, pp_rank): LogicalClockDriver(start_s=quantized_start_s)
            for dp_rank in range(self._dp_size)
            for pp_rank in range(self._pp_size)
        }
        # Compatibility view for PP=1 callers.  New physical PP code must use
        # ``stage_driver`` so later stages never inherit PP0's time.
        self.rank_drivers = {
            dp_rank: self.stage_drivers[(dp_rank, 0)]
            for dp_rank in range(self._dp_size)
        }
        # Metadata only: event ownership lives in GlobalDES._events.  This map
        # lets us reconstruct DP/PP identity for coordinator-originated events
        # after the global callback has been dispatched.
        self._offline_events_by_id: dict[str, OfflineEvent] = {}
        self._inflight_completion: dict[tuple[int, int], OfflineEvent] = {}
        self._rank_work_barriers: dict[
            tuple[int, int], OfflineRankWorkBarrier
        ] = {}
        self._rank_work_sequence = 0

    @property
    def now_s(self) -> float:
        """Seconds compatibility view of the nanosecond global cursor."""

        return _nanoseconds_to_seconds(self.global_des.now_ns)

    @property
    def now_ns(self) -> int:
        return self.global_des.now_ns

    @property
    def dp_size(self) -> int:
        return self._dp_size

    @property
    def pp_size(self) -> int:
        return self._pp_size

    def _validate_rank(self, dp_rank: int) -> int:
        rank = int(dp_rank)
        if not 0 <= rank < self._dp_size:
            raise ValueError(f"invalid DP rank {rank}")
        return rank

    def _validate_stage(self, dp_rank: int, pp_rank: int) -> tuple[int, int]:
        rank = self._validate_rank(dp_rank)
        stage = int(pp_rank)
        if not 0 <= stage < self._pp_size:
            raise ValueError(f"invalid PP rank {stage}")
        return rank, stage

    def stage_driver(self, dp_rank: int, pp_rank: int) -> LogicalClockDriver:
        return self.stage_drivers[self._validate_stage(dp_rank, pp_rank)]

    def begin_stage_work(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        lower_bound_s: float | None = None,
        operation: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> OfflineRankWorkBarrier:
        """Publish a conservative barrier before dispatching rank work."""

        key = self._validate_stage(dp_rank, pp_rank)
        if key in self._rank_work_barriers:
            pending = self._rank_work_barriers[key]
            raise RuntimeError(
                f"DP/PP stage {key} already has unpublished rank work "
                f"{pending.barrier_id}"
            )
        op = str(operation).strip()
        if not op:
            raise ValueError("rank work operation must be nonempty")
        lower_bound_ns = (
            self.global_des.now_ns
            if lower_bound_s is None
            else _seconds_to_nanoseconds(
                lower_bound_s, name="rank work lower_bound_s"
            )
        )
        if lower_bound_ns < self.global_des.now_ns:
            raise ValueError(
                "rank work lower bound cannot precede the global cursor"
            )
        barrier = OfflineRankWorkBarrier(
            barrier_id=f"rank-work-{self._rank_work_sequence:020d}",
            dp_rank=key[0],
            pp_rank=key[1],
            lower_bound_ns=lower_bound_ns,
            operation=op,
            metadata=copy.deepcopy(dict(metadata or {})),
        )
        self._rank_work_sequence += 1
        self._rank_work_barriers[key] = barrier
        return barrier

    def finish_stage_work(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        barrier_id: str,
    ) -> OfflineRankWorkBarrier:
        """Release exactly the barrier whose worker response was received."""

        key = self._validate_stage(dp_rank, pp_rank)
        pending = self._rank_work_barriers.get(key)
        if pending is None:
            raise RuntimeError(f"DP/PP stage {key} has no unpublished rank work")
        if pending.barrier_id != str(barrier_id):
            raise RuntimeError(
                f"stale rank work response for DP/PP stage {key}: "
                f"{barrier_id!r} != {pending.barrier_id!r}"
            )
        del self._rank_work_barriers[key]
        return pending

    def cancel_stage_work(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        barrier_id: str | None = None,
    ) -> OfflineRankWorkBarrier | None:
        """Cancel an unpublished command only during worker teardown/error."""

        key = self._validate_stage(dp_rank, pp_rank)
        pending = self._rank_work_barriers.get(key)
        if pending is None:
            return None
        if barrier_id is not None and pending.barrier_id != str(barrier_id):
            raise RuntimeError(
                f"stale rank work cancellation for DP/PP stage {key}"
            )
        del self._rank_work_barriers[key]
        return pending

    def stage_has_unpublished_work(self, dp_rank: int, pp_rank: int) -> bool:
        return self._validate_stage(dp_rank, pp_rank) in self._rank_work_barriers

    def rank_work_barriers(self) -> tuple[OfflineRankWorkBarrier, ...]:
        return tuple(
            sorted(
                self._rank_work_barriers.values(),
                key=lambda item: (
                    item.lower_bound_ns,
                    item.dp_rank,
                    item.pp_rank,
                    item.barrier_id,
                ),
            )
        )

    def schedule_at(
        self,
        timestamp_s: float,
        *,
        kind: str,
        priority: OfflineEventPriority | int,
        dp_rank: int | None = None,
        pp_rank: int | None = None,
        payload: Any = None,
    ) -> OfflineEvent:
        timestamp_ns = _seconds_to_nanoseconds(timestamp_s, name="timestamp_s")
        if timestamp_ns < self.global_des.now_ns:
            raise ValueError(
                f"cannot schedule {kind!r} at {timestamp_s}; "
                f"parent is at {self.now_s}"
            )
        if dp_rank is None:
            if pp_rank is not None:
                raise ValueError("pp_rank requires a dp_rank")
            rank = None
            stage = None
        else:
            rank = self._validate_rank(dp_rank)
            stage = (
                self._validate_stage(rank, pp_rank)[1]
                if pp_rank is not None
                else None
            )
        return self._schedule_at_ns(
            timestamp_ns,
            offline_timestamp_s=float(timestamp_s),
            kind=str(kind),
            dp_rank=rank,
            pp_rank=stage,
            priority=priority,
            payload=payload,
        )

    def _schedule_at_ns(
        self,
        timestamp_ns: int,
        *,
        offline_timestamp_s: float,
        kind: str,
        priority: OfflineEventPriority | int,
        dp_rank: int | None,
        pp_rank: int | None,
        payload: Any,
    ) -> OfflineEvent:
        global_event = self.global_des.schedule_at(
            timestamp_ns,
            kind=kind,
            priority=int(priority),
            payload=payload,
        )
        event = self._wrap_global_event(
            global_event,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            timestamp_s=offline_timestamp_s,
        )
        self._offline_events_by_id[global_event.event_id] = event
        return event

    @staticmethod
    def _wrap_global_event(
        event: GlobalEvent,
        *,
        dp_rank: int | None = None,
        pp_rank: int | None = None,
        timestamp_s: float | None = None,
    ) -> OfflineEvent:
        return OfflineEvent(
            timestamp_s=(
                _nanoseconds_to_seconds(event.timestamp_ns)
                if timestamp_s is None
                else float(timestamp_s)
            ),
            priority=event.priority,
            sequence=event.sequence,
            kind=event.kind,
            event_id=event.event_id,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            payload=event.payload,
        )

    def activate_idle_rank(self, dp_rank: int, *, at_s: float | None = None) -> float:
        """Compatibility wrapper that activates PP0 of one DP replica."""

        return self.activate_idle_stage(dp_rank, 0, at_s=at_s)

    def activate_idle_stage(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        at_s: float | None = None,
    ) -> float:
        """Move one idle stage to a causal event, never unrelated stages."""

        key = self._validate_stage(dp_rank, pp_rank)
        if key in self._inflight_completion:
            raise RuntimeError(
                f"DP/PP stage {key} already has an in-flight step"
            )
        timestamp_ns = (
            self.global_des.now_ns
            if at_s is None
            else _seconds_to_nanoseconds(at_s, name="at_s")
        )
        if timestamp_ns < self.global_des.now_ns:
            raise ValueError("stage activation cannot precede the parent cursor")
        driver = self.stage_drivers[key]
        driver_ns = _seconds_to_nanoseconds(driver.now(), name="stage clock")
        if timestamp_ns > driver_ns:
            driver.set_time(_nanoseconds_to_seconds(timestamp_ns))
        return driver.now()

    def schedule_rank_completion(
        self,
        dp_rank: int,
        *,
        duration_s: float,
        payload: Any = None,
    ) -> OfflineEvent:
        """Compatibility wrapper for a prepared PP0 scheduler step."""

        return self.schedule_stage_completion(
            dp_rank,
            0,
            duration_s=duration_s,
            payload=payload,
        )

    def schedule_stage_completion(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        duration_s: float,
        payload: Any = None,
    ) -> OfflineEvent:
        """Schedule one prepared scheduler step for one physical PP stage.

        The scheduler-side prepare phase has already applied admission and KV
        allocation effects.  Its completion phase must be called only when the
        returned event is popped.
        """

        key = self._validate_stage(dp_rank, pp_rank)
        if key in self._inflight_completion:
            raise RuntimeError(
                f"DP/PP stage {key} already has an in-flight step"
            )
        duration = float(duration_s)
        if duration < 0.0:
            raise ValueError("stage duration cannot be negative")
        driver = self.stage_drivers[key]
        driver_ns = _seconds_to_nanoseconds(driver.now(), name="stage clock")
        start_ns = max(driver_ns, self.global_des.now_ns)
        if start_ns > driver_ns:
            driver.set_time(_nanoseconds_to_seconds(start_ns))
        # Preserve the Scheduler's exact float completion timestamp for its
        # strict LogicalPipelinePlan validation.  Only the global heap key is
        # causally quantized upward to nanoseconds.
        completion_s = driver.now() + duration
        completion_ns = _seconds_to_nanoseconds(
            completion_s,
            name="stage completion",
        )
        event = self._schedule_at_ns(
            completion_ns,
            offline_timestamp_s=completion_s,
            kind="rank_completion",
            priority=OfflineEventPriority.RANK_COMPLETION,
            dp_rank=key[0],
            pp_rank=key[1],
            payload=payload,
        )
        self._inflight_completion[key] = event
        return event

    def cancel_rank_completion(
        self,
        dp_rank: int,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OfflineEvent | None:
        """Compatibility wrapper that cancels PP0 of one DP replica."""

        return self.cancel_stage_completion(
            dp_rank,
            0,
            reason=reason,
            metadata=metadata,
        )

    def cancel_stage_completion(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OfflineEvent | None:
        """Cancel the pending completion owned by one physical PP stage.

        The interrupted stage advances only to the current global cancellation
        instant, never to its planned completion.  No unrelated stage clock is
        synchronized.  Replaying a stage cancellation after cleanup is a
        no-op that returns ``None``.
        """

        key = self._validate_stage(dp_rank, pp_rank)
        event = self._inflight_completion.get(key)
        if event is None:
            return None
        cancellation_metadata = {
            **copy.deepcopy(dict(metadata or {})),
            "dp_rank": key[0],
            "pp_rank": key[1],
        }
        self.global_des.cancel_event(
            event.event_id,
            reason=reason,
            metadata=cancellation_metadata,
        )
        del self._inflight_completion[key]
        self._offline_events_by_id.pop(event.event_id, None)

        driver = self.stage_drivers[key]
        cancellation_s = self.now_s
        if cancellation_s > driver.now():
            driver.set_time(cancellation_s)
        return event

    def pop_next(self) -> OfflineEvent:
        if not self.global_des.has_events():
            raise IndexError("offline event queue is empty")
        candidate = self.global_des.peek_next()
        blocking = tuple(
            barrier
            for barrier in self.rank_work_barriers()
            if candidate.timestamp_ns >= barrier.lower_bound_ns
        )
        if blocking:
            first = blocking[0]
            raise OfflineCausalBarrierError(
                "global virtual time cannot dispatch "
                f"{candidate.timestamp_ns} ns ({candidate.kind}) while "
                f"{first.barrier_id} for DP/PP stage "
                f"({first.dp_rank}, {first.pp_rank}) has not published its "
                f"next event beyond lower bound {first.lower_bound_ns} ns"
            )
        # GlobalDES executes the callback (including transfer state changes or
        # failure handling) before returning the dispatched event.
        global_event = self.global_des.pop_next()
        event = self._offline_events_by_id.pop(global_event.event_id, None)
        coordinator_owned = event is not None
        if event is None:
            event = self._wrap_global_event(global_event)
        if coordinator_owned and event.kind == "rank_completion":
            if event.dp_rank is None or event.pp_rank is None:
                raise RuntimeError("stage completion has no DP/PP identity")
            key = self._validate_stage(event.dp_rank, event.pp_rank)
            pending = self._inflight_completion.get(key)
            if pending != event:
                raise RuntimeError(
                    f"stale stage completion for DP/PP stage {key}"
                )
            del self._inflight_completion[key]
            driver = self.stage_drivers[key]
            if event.timestamp_s > driver.now():
                driver.set_time(event.timestamp_s)
        return event

    def has_events(self) -> bool:
        return self.global_des.has_events()

    def rank_is_busy(self, dp_rank: int) -> bool:
        rank = self._validate_rank(dp_rank)
        return any(key[0] == rank for key in self._inflight_completion)

    def stage_is_busy(self, dp_rank: int, pp_rank: int) -> bool:
        return self._validate_stage(dp_rank, pp_rank) in self._inflight_completion

    def dp_makespan_s(self, dp_rank: int) -> float:
        rank = self._validate_rank(dp_rank)
        return max(
            driver.now()
            for (stage_dp_rank, _), driver in self.stage_drivers.items()
            if stage_dp_rank == rank
        )

    def makespan_s(self) -> float:
        return max(driver.now() for driver in self.stage_drivers.values())
