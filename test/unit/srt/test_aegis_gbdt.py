from __future__ import annotations

import numpy as np

from sglsim.srt.simulation.aegis_gbdt import (
    AegisGBDTConfig,
    AegisGBDTRegressor,
)


def test_histogram_gbdt_is_deterministic_and_learns_monotone_signal() -> None:
    x = np.linspace(0.0, 10.0, 240)
    features = np.column_stack([x, np.sin(x)])
    target = 4.0 + 3.5 * x + np.where(x > 5.0, 12.0, 0.0)
    config = AegisGBDTConfig(
        estimators=36,
        learning_rate=0.08,
        max_depth=3,
        min_leaf=8,
        max_bins=24,
    )
    first = AegisGBDTRegressor(config).fit(features, target)
    second = AegisGBDTRegressor(config).fit(features, target)
    first_prediction = first.predict(features)
    second_prediction = second.predict(features)

    assert np.array_equal(first_prediction, second_prediction)
    assert np.mean(np.abs(first_prediction - target)) < 3.0
    assert first_prediction[-1] > first_prediction[0]
    assert first.manifest()["sha256"] == second.manifest()["sha256"]


def test_histogram_gbdt_rejects_negative_remaining_time() -> None:
    features = np.asarray([[0.0], [1.0]])
    target = np.asarray([1.0, -1.0])
    try:
        AegisGBDTRegressor(
            AegisGBDTConfig(estimators=2, min_leaf=1)
        ).fit(features, target)
    except ValueError as exc:
        assert "nonnegative" in str(exc)
    else:
        raise AssertionError("negative remaining time was accepted")
