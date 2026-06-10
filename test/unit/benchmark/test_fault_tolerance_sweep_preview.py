from __future__ import annotations

import time
from pathlib import Path

from benchmark.synthesis_requests.diagnostics.run_fault_tolerance_sweeps import (
    DEFAULT_BASE,
    DEFAULT_BASELINE,
    _completed_keys,
    _entry_key,
    _run_timeout,
    _spec_key,
    _upsert_manifest,
    build_cases,
    build_preview,
    build_run_specs,
    build_status,
    filter_cases,
)


def test_fault_tolerance_full_sweep_preview_shape() -> None:
    cases = build_cases(
        base_config=Path(DEFAULT_BASE),
        baseline_config=Path(DEFAULT_BASELINE),
        smoke=False,
    )
    specs = build_run_specs(cases, repeats=5, seed_base=100, port_base=31000)
    preview = build_preview(specs)

    assert preview["total_runs"] == 85
    assert preview["axis_value_counts"] == {
        "backup_lag": 6,
        "critical_ratio": 5,
        "failure_timing_s": 4,
        "strategy": 2,
    }
    assert preview["seeds"] == [100, 101, 102, 103, 104]
    assert preview["ports"][0] == 31000
    assert preview["ports"][-1] == 31084
    first = preview["runs"][0]
    assert first["name"] == "baseline"
    assert first["client"]["agents"] == 4
    assert first["client"]["turns_per_agent"] == 3
    assert first["client"]["http_requests"] == 12
    assert first["server"]["model_path"] == "Qwen/Qwen3-8B"
    assert first["server"]["enable_hierarchical_cache"] is False
    assert first["server"]["inject_after_s"] == 40
    assert preview["runs"][-1]["name"] == "failure_t_44"
    assert preview["runs"][-1]["server"]["inject_after_s"] == 44


def test_fault_tolerance_smoke_sweep_preview_shape() -> None:
    cases = build_cases(
        base_config=Path(DEFAULT_BASE),
        baseline_config=Path(DEFAULT_BASELINE),
        smoke=True,
    )
    specs = build_run_specs(cases, repeats=2, seed_base=10, port_base=30000)
    preview = build_preview(specs)

    assert preview["total_runs"] == 16
    assert preview["axis_value_counts"] == {
        "backup_lag": 2,
        "critical_ratio": 2,
        "failure_timing_s": 2,
        "strategy": 2,
    }
    assert preview["runs"][0]["server"]["model_path"] == "mock-model"
    assert preview["runs"][0]["server"]["predictor"] == "constant"


def test_fault_tolerance_sweep_preview_filters_by_axis_and_name() -> None:
    cases = build_cases(
        base_config=Path(DEFAULT_BASE),
        baseline_config=Path(DEFAULT_BASELINE),
        smoke=False,
    )
    strategy_cases = filter_cases(cases, axes="strategy")
    strategy_specs = build_run_specs(
        strategy_cases,
        repeats=1,
        seed_base=7,
        port_base=32000,
    )
    strategy_preview = build_preview(strategy_specs)
    assert strategy_preview["total_runs"] == 2
    assert strategy_preview["axis_value_counts"] == {"strategy": 2}
    assert [run["name"] for run in strategy_preview["runs"]] == [
        "baseline",
        "gpu_critical",
    ]

    one_case = filter_cases(cases, names="failure_t_30")
    one_specs = build_run_specs(one_case, repeats=2, seed_base=9, port_base=32100)
    one_preview = build_preview(one_specs)
    assert one_preview["total_runs"] == 2
    assert one_preview["axis_value_counts"] == {"failure_timing_s": 1}
    assert one_preview["seeds"] == [9, 10]
    assert {run["server"]["inject_after_s"] for run in one_preview["runs"]} == {30}


def test_fault_tolerance_sweep_run_timeout_raises() -> None:
    try:
        with _run_timeout(0.01):
            time.sleep(0.05)
    except TimeoutError as exc:
        assert "run exceeded timeout_s" in str(exc)
    else:
        raise AssertionError("timeout did not fire")


def test_fault_tolerance_sweep_manifest_resume_helpers() -> None:
    cases = filter_cases(
        build_cases(
            base_config=Path(DEFAULT_BASE),
            baseline_config=Path(DEFAULT_BASELINE),
            smoke=True,
        ),
        names="baseline",
    )
    spec = build_run_specs(cases, repeats=1, seed_base=3, port_base=33000)[0]
    failed = {
        "name": "baseline",
        "axis": "strategy",
        "value": "baseline",
        "repeat_idx": 0,
        "seed": 3,
        "port": 33000,
        "status": "failed",
        "error": "old",
    }
    completed = {
        **failed,
        "status": "completed",
        "run_dir": "/tmp/run",
    }

    assert _entry_key(failed) == _spec_key(spec)
    assert _completed_keys([failed]) == set()
    assert _completed_keys([completed]) == {_spec_key(spec)}

    manifest = _upsert_manifest([failed], completed)
    assert manifest == [completed]


def test_fault_tolerance_sweep_status_summarizes_manifest() -> None:
    cases = filter_cases(
        build_cases(
            base_config=Path(DEFAULT_BASE),
            baseline_config=Path(DEFAULT_BASELINE),
            smoke=True,
        ),
        names="baseline,gpu_critical",
    )
    specs = build_run_specs(cases, repeats=1, seed_base=5, port_base=32100)
    manifest = [
        {
            "name": "baseline",
            "axis": "strategy",
            "value": "baseline",
            "repeat_idx": 0,
            "seed": 5,
            "port": 32100,
            "status": "completed",
            "run_dir": "/tmp/baseline",
        },
        {
            "name": "gpu_critical",
            "axis": "strategy",
            "value": "gpu_critical",
            "repeat_idx": 0,
            "seed": 5,
            "port": 32101,
            "status": "failed",
            "error": "timeout",
        },
    ]

    status = build_status(specs, manifest)

    assert status["total_runs"] == 2
    assert status["completed"] == 1
    assert status["failed"] == 1
    assert status["pending"] == 0
    assert status["axis_completed"] == {"strategy": 1}
    assert status["failed_entries"][0]["name"] == "gpu_critical"
