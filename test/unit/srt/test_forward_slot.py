import pytest

from sglsim.srt.simulation.forward_slot import ForwardSlot


def test_cpu_budget_zero_before_first_forward():
    slot = ForwardSlot()
    assert slot.cpu_budget_s(100.0) == 0.0


def test_cpu_budget_after_forward():
    slot = ForwardSlot()
    slot.record_forward(now=10.0, forward_s=0.05, sleep_s=0.05)
    assert slot.slot_end_wall == 10.05
    assert slot.cpu_budget_s(10.02) == pytest.approx(0.03)


def test_cpu_overflow_when_sched_exceeds_budget():
    slot = ForwardSlot()
    slot.record_forward(now=10.0, forward_s=0.05, sleep_s=0.05)
    overflow = slot.cpu_overflow_s(now=10.0, sched_elapsed_s=0.08)
    assert overflow == pytest.approx(0.03)


def test_slot_end_monotonic_with_late_now():
    slot = ForwardSlot()
    slot.record_forward(now=10.0, forward_s=0.05, sleep_s=0.05)
    slot.record_forward(now=10.04, forward_s=0.05, sleep_s=0.05)
    assert slot.slot_end_wall == pytest.approx(10.10)
