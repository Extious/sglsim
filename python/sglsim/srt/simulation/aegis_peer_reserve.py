"""Hard reserve calculation for AegisKV's peer-HBM safety envelope.

This module is a small theorem-facing calculator, not an allocator.  All
protection-side occupancy---committed shadows, protection destination
reservations/in-flight writes, and recovery pins---is charged to ``peer_bytes``.
During conservative D2D adoption the source shadow remains in ``peer_bytes``
while the freshly allocated, quarantined recovery destination is charged to
``primary_bytes``.  That duplicate must fit in physical headroom *outside* the
remaining primary reserve; recovery is not allowed to spend the safety margin
that protects normal primary growth.  ``d2d_pending_bytes`` is a locked subset
present on both sides; it prevents premature source reclaim but is not added a
third time.  The reserve covers primary growth that can occur before a bounded
peer reclaim/cancel reaches terminal state, an admitted allocation burst, and
an allocator-fragmentation margin.

The AIMD actuator's desired peer cap is deliberately separate from actual
occupancy.  Lowering the cap never frees bytes: it moves the ledger to
``DRAINING``, rejects new peer admission, and waits for generation-fenced
terminal reclaim/adoption transitions to reduce occupancy.  Recovery-pinned
or in-flight bytes therefore remain charged and may appear as an explicit cap
overdraft.  This reference model assumes its caller invokes
``terminal_reclaim_peer`` only for bytes that are in fact terminal and
reclaimable; pin eligibility remains a physical allocator obligation.

Without a finite, versioned envelope, theorem mode exposes zero peer budget.
That fail-closed rule avoids turning a measured mean reclaim time into an OOM
safety guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


def _finite_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class AegisPeerReserveEnvelope:
    primary_growth_rate_upper_bytes_per_s: float
    reclaim_terminal_upper_s: float
    allocation_burst_upper_bytes: int
    fragmentation_upper_bytes: int
    envelope_version: str
    source: str
    uses_future_information: bool = False

    def __post_init__(self) -> None:
        _finite_nonnegative(
            "primary_growth_rate_upper_bytes_per_s",
            self.primary_growth_rate_upper_bytes_per_s,
        )
        _finite_nonnegative(
            "reclaim_terminal_upper_s", self.reclaim_terminal_upper_s
        )
        for name in (
            "allocation_burst_upper_bytes",
            "fragmentation_upper_bytes",
        ):
            _nonnegative_int(name, getattr(self, name))
        if not self.envelope_version or not self.source:
            raise ValueError("reserve envelope version/source must be nonempty")
        if self.uses_future_information:
            raise ValueError("reserve envelope cannot use future information")

    @property
    def required_reserve_bytes(self) -> int:
        growth_during_terminal = (
            self.primary_growth_rate_upper_bytes_per_s
            * self.reclaim_terminal_upper_s
        )
        return math.ceil(
            growth_during_terminal
            + self.allocation_burst_upper_bytes
            + self.fragmentation_upper_bytes
        )


def theorem_peer_budget_bytes(
    *,
    kv_capacity_bytes: int,
    primary_occupied_bytes: int,
    alpha: float,
    envelope: AegisPeerReserveEnvelope | None,
) -> int:
    """Return a fail-closed peer occupancy cap for one target group/rank.

    The returned value is a cap on *total* peer occupancy, not free incremental
    bytes.  Admission must additionally check current peer occupancy and update
    the physical ledger atomically on every allocator transition.
    """

    for name, value in (
        ("kv_capacity_bytes", kv_capacity_bytes),
        ("primary_occupied_bytes", primary_occupied_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if primary_occupied_bytes > kv_capacity_bytes:
        raise ValueError("primary occupancy exceeds KV capacity")
    parsed_alpha = float(alpha)
    if not math.isfinite(parsed_alpha) or not 0.0 <= parsed_alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if envelope is None:
        return 0
    reserve = envelope.required_reserve_bytes
    physical_cap = max(
        0, kv_capacity_bytes - primary_occupied_bytes - reserve
    )
    policy_cap = math.floor(parsed_alpha * kv_capacity_bytes)
    return min(policy_cap, physical_cap)


class AegisPeerLedgerMode(str, Enum):
    NORMAL = "NORMAL"
    DRAINING = "DRAINING"


@dataclass
class AegisPeerHbmLedger:
    """Executable reference for the reserve invariant's state transitions."""

    capacity_bytes: int
    primary_bytes: int
    peer_bytes: int
    required_reserve_bytes: int
    remaining_reserve_bytes: int | None = None
    d2d_pending_bytes: int = 0
    desired_peer_cap_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "capacity_bytes",
            "primary_bytes",
            "peer_bytes",
            "required_reserve_bytes",
            "d2d_pending_bytes",
        ):
            _nonnegative_int(name, getattr(self, name))
        if self.remaining_reserve_bytes is None:
            self.remaining_reserve_bytes = self.required_reserve_bytes
        if self.desired_peer_cap_bytes is None:
            self.desired_peer_cap_bytes = self.capacity_bytes
        _nonnegative_int(
            "remaining_reserve_bytes", self.remaining_reserve_bytes
        )
        _nonnegative_int(
            "desired_peer_cap_bytes", self.desired_peer_cap_bytes
        )
        if self.remaining_reserve_bytes > self.required_reserve_bytes:
            raise ValueError("remaining reserve exceeds required reserve")
        if self.desired_peer_cap_bytes > self.capacity_bytes:
            raise ValueError("desired peer cap exceeds physical capacity")
        if self.d2d_pending_bytes > min(self.peer_bytes, self.primary_bytes):
            raise ValueError("pending D2D bytes exceed source or destination occupancy")
        self._assert_invariant()

    @property
    def mode(self) -> AegisPeerLedgerMode:
        if (
            self.remaining_reserve_bytes == self.required_reserve_bytes
            and self.peer_cap_overdraft_bytes == 0
            and self.d2d_pending_bytes == 0
        ):
            return AegisPeerLedgerMode.NORMAL
        return AegisPeerLedgerMode.DRAINING

    @property
    def peer_cap_overdraft_bytes(self) -> int:
        """Actual peer occupancy above the desired admission cap."""

        return max(0, self.peer_bytes - int(self.desired_peer_cap_bytes))

    @property
    def accounted_bytes(self) -> int:
        return (
            self.primary_bytes
            + self.peer_bytes
            + int(self.remaining_reserve_bytes)
        )

    def admit_peer(self, amount_bytes: int) -> None:
        amount = _nonnegative_int("amount_bytes", amount_bytes)
        if self.mode is not AegisPeerLedgerMode.NORMAL:
            raise RuntimeError("peer admission is disabled while DRAINING")
        if self.peer_bytes + amount > int(self.desired_peer_cap_bytes):
            raise RuntimeError("peer admission exceeds the desired peer cap")
        if self.accounted_bytes + amount > self.capacity_bytes:
            raise RuntimeError("peer admission exceeds the HBM ledger")
        self.peer_bytes += amount
        self._assert_invariant()

    def set_desired_peer_cap(self, new_cap_bytes: int) -> None:
        """Update admission intent without pretending current bytes vanished."""

        new_cap = _nonnegative_int("new_cap_bytes", new_cap_bytes)
        if new_cap > self.capacity_bytes:
            raise ValueError("desired peer cap exceeds physical capacity")
        self.desired_peer_cap_bytes = new_cap
        self._assert_invariant()

    def grow_primary(self, amount_bytes: int) -> None:
        amount = _nonnegative_int("amount_bytes", amount_bytes)
        if amount > int(self.remaining_reserve_bytes):
            raise RuntimeError("primary growth exceeds remaining reserve")
        self.primary_bytes += amount
        self.remaining_reserve_bytes -= amount
        self._assert_invariant()

    def shrink_primary(self, amount_bytes: int) -> None:
        amount = _nonnegative_int("amount_bytes", amount_bytes)
        if amount > self.primary_bytes:
            raise ValueError("primary shrink exceeds primary occupancy")
        self.primary_bytes -= amount
        self._replenish_reserve(amount)
        self._assert_invariant()

    def terminal_reclaim_peer(self, amount_bytes: int) -> None:
        amount = _nonnegative_int("amount_bytes", amount_bytes)
        reclaimable = self.peer_bytes - self.d2d_pending_bytes
        if amount > reclaimable:
            raise ValueError("peer reclaim exceeds terminal reclaimable bytes")
        self.peer_bytes -= amount
        self._replenish_reserve(amount)
        self._assert_invariant()

    def adopt_zero_copy(self, amount_bytes: int) -> None:
        amount = _nonnegative_int("amount_bytes", amount_bytes)
        reclaimable = self.peer_bytes - self.d2d_pending_bytes
        if amount > reclaimable:
            raise ValueError("zero-copy adoption exceeds available peer bytes")
        self.peer_bytes -= amount
        self.primary_bytes += amount
        self._assert_invariant()

    def begin_d2d_adoption(self, amount_bytes: int) -> None:
        amount = _nonnegative_int("amount_bytes", amount_bytes)
        if amount > self.peer_bytes - self.d2d_pending_bytes:
            raise ValueError("D2D adoption exceeds available peer bytes")
        if self.mode is not AegisPeerLedgerMode.NORMAL:
            raise RuntimeError("D2D adoption requires a NORMAL HBM ledger")
        if self.accounted_bytes + amount > self.capacity_bytes:
            raise RuntimeError(
                "D2D duplicate would consume protected primary reserve"
            )
        # Discretionary recovery cannot debit the normal-serving safety
        # reserve.  The temporary destination must fit in extra headroom.
        self.primary_bytes += amount
        self.d2d_pending_bytes += amount
        self._assert_invariant()

    def finish_d2d_adoption(self, amount_bytes: int) -> None:
        amount = _nonnegative_int("amount_bytes", amount_bytes)
        if amount > self.d2d_pending_bytes:
            raise ValueError("D2D finish exceeds pending adoption bytes")
        self.d2d_pending_bytes -= amount
        self.peer_bytes -= amount
        self._replenish_reserve(amount)
        self._assert_invariant()

    def abort_d2d_adoption(self, amount_bytes: int) -> None:
        amount = _nonnegative_int("amount_bytes", amount_bytes)
        if amount > self.d2d_pending_bytes:
            raise ValueError("D2D abort exceeds pending adoption bytes")
        self.d2d_pending_bytes -= amount
        self.primary_bytes -= amount
        self._replenish_reserve(amount)
        self._assert_invariant()

    def shrink_capacity(self, new_capacity_bytes: int) -> None:
        new_capacity = _nonnegative_int(
            "new_capacity_bytes", new_capacity_bytes
        )
        if self.accounted_bytes > new_capacity:
            raise RuntimeError("capacity shrink violates the HBM ledger")
        self.capacity_bytes = new_capacity
        # A policy cap cannot remain above the physical capacity.  Tightening it
        # is an admission-only update; any resulting overdraft remains charged
        # and keeps the ledger in DRAINING until terminal transitions occur.
        self.desired_peer_cap_bytes = min(
            int(self.desired_peer_cap_bytes), new_capacity
        )
        self._assert_invariant()

    def _replenish_reserve(self, freed_bytes: int) -> None:
        self.remaining_reserve_bytes = min(
            self.required_reserve_bytes,
            int(self.remaining_reserve_bytes) + freed_bytes,
        )

    def _assert_invariant(self) -> None:
        if int(self.desired_peer_cap_bytes) > self.capacity_bytes:
            raise ValueError("desired peer cap exceeds physical capacity")
        if self.accounted_bytes > self.capacity_bytes:
            raise ValueError("HBM ledger exceeds physical capacity")
