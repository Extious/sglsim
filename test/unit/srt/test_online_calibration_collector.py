from __future__ import annotations

import pytest

from AegisKV.demo.collect_online_calibration import merge_online_calibration


def test_merge_online_calibration_preserves_wct_and_attaches_common_schema():
    merged = merge_online_calibration(
        {
            "policy": "no_backup",
            "request_count": 1,
            "wct_summary_s": {"mean": 2.0},
            "failure_events": [{"event_id": "failure-01"}],
        },
        {
            "simulation_time_mode": "wall",
            "evidence_tier": "sglsim-online-instrumented",
            "identity": {"dp_size": 4},
            "physical_topology": {
                "process_count": 16,
                "stage_ids": ["dp0.pp0.tp0"],
            },
            "calibration_metrics": {"request_count": 1},
            "request_stats": {"r0": {"status": "completed"}},
            "scheduler_telemetry": [{"dp_rank": 0}],
            "aegis_status": {
                "event_history": {"mode": "unbounded_in_memory"}
            },
        },
    )

    assert merged["wct_summary_s"]["mean"] == 2.0
    assert merged["failure_events"][0]["event_id"] == "failure-01"
    assert merged["calibration_metrics"]["request_count"] == 1
    assert merged["scheduler_telemetry"] == [{"dp_rank": 0}]
    assert merged["physical_topology"]["process_count"] == 16
    assert merged["online_event_history"]["mode"] == "unbounded_in_memory"


def test_merge_online_calibration_rejects_partial_server_snapshot():
    with pytest.raises(ValueError, match="request count mismatch"):
        merge_online_calibration(
            {"request_count": 2},
            {
                "simulation_time_mode": "wall",
                "calibration_metrics": {"request_count": 1},
            },
        )


def test_merge_online_calibration_counts_retry_requests_in_server_snapshot():
    merged = merge_online_calibration(
        {
            "request_count": 1,
            "retry_summary": {"count": 1},
        },
        {
            "simulation_time_mode": "wall",
            "calibration_metrics": {"request_count": 2},
        },
    )

    assert merged["online_instrumentation"]["original_request_count"] == 1
    assert merged["online_instrumentation"]["retry_request_count"] == 1
    assert merged["online_instrumentation"]["observed_request_count"] == 2
