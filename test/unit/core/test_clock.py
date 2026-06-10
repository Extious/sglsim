import pytest

from sglsim.srt.simulation.clock import Clock


def test_clock_advances_monotonically():
    clock = Clock()
    clock.advance(0.5)
    clock.advance(1.0)

    assert clock.now == 1.5


def test_clock_set_rejects_moving_backwards():
    clock = Clock(now=1.0)

    with pytest.raises(ValueError):
        clock.set(0.5)

    assert clock.now == 1.0
