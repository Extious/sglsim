from __future__ import annotations

import pytest

from AegisKV.demo.fit_offline_driver_calibration import fit_driver_calibration


def test_fit_driver_calibration_recovers_linear_external_overhead():
    online = {
        "workflows": [
            {
                "request_routes": [
                    {"rid": "r0", "completion_index": 1, "completion_elapsed_s": 1.3},
                    {"rid": "r1", "completion_index": 2, "completion_elapsed_s": 2.4},
                    {"rid": "r2", "completion_index": 3, "completion_elapsed_s": 3.5},
                ]
            }
        ]
    }
    offline = {
        "workflows": [
            {
                "request_routes": [
                    {"rid": "r0", "completion_index": 1, "completion_elapsed_s": 1.0},
                    {"rid": "r1", "completion_index": 2, "completion_elapsed_s": 2.0},
                    {"rid": "r2", "completion_index": 3, "completion_elapsed_s": 3.0},
                ]
            }
        ]
    }

    fitted = fit_driver_calibration(online, offline)

    assert fitted["startup_overhead_s"] == pytest.approx(0.2)
    assert fitted["request_overhead_s"] == pytest.approx(0.1)
    assert fitted["rmse_s"] == pytest.approx(0.0)
