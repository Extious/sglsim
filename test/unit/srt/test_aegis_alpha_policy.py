from __future__ import annotations

import json

import pytest

from sglsim.srt.simulation import alpha_policy
from sglsim.srt.simulation.alpha_policy import (
    ENV_ALPHA_POLICY,
    ENV_FAILURE_RATE_PER_WINDOW,
    choose_alpha,
    pressure_from_request,
)


def test_pressure_marks_decode_dominant_critical_agent_as_highest():
    singleton = pressure_from_request(
        critical_agent=True,
        decode_dominant=False,
        stage_width=1,
    )
    fanout_tail = pressure_from_request(
        critical_agent=True,
        decode_dominant=True,
        stage_width=3,
    )
    noncritical = pressure_from_request(
        critical_agent=False,
        decode_dominant=False,
        stage_width=3,
    )

    assert noncritical < singleton < fanout_tail


def test_fallback_alpha_increases_with_pressure_and_failure(monkeypatch):
    monkeypatch.delenv(ENV_ALPHA_POLICY, raising=False)
    monkeypatch.delenv(ENV_FAILURE_RATE_PER_WINDOW, raising=False)

    low = choose_alpha(
        critical_agent=False,
        decode_dominant=False,
        stage_width=3,
        failure_rate_per_window=0.001,
    )
    high = choose_alpha(
        critical_agent=True,
        decode_dominant=True,
        stage_width=3,
        failure_rate_per_window=0.1,
    )

    assert low.source == "fallback_formula"
    assert low.alpha < high.alpha
    assert high.alpha == pytest.approx(0.845)


def test_policy_table_is_used_when_available(monkeypatch, tmp_path):
    path = tmp_path / "alpha_policy.json"
    path.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "critical_pressure": "low",
                        "failure_rate_per_window": 0.001,
                        "alpha": 0.2,
                    },
                    {
                        "critical_pressure": "high",
                        "failure_rate_per_window": 0.1,
                        "alpha": 0.8,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    alpha_policy._load_policy.cache_clear()
    monkeypatch.setenv(ENV_ALPHA_POLICY, str(path))

    decision = choose_alpha(
        critical_agent=True,
        decode_dominant=True,
        stage_width=3,
        failure_rate_per_window=0.08,
    )

    assert decision.alpha == 0.8
    assert decision.source == f"policy_table:{path}"


def test_request_supplied_pressure_overrides_class_proxy(monkeypatch, tmp_path):
    path = tmp_path / "alpha_policy.json"
    path.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "critical_pressure": "medium",
                        "failure_rate_per_window": 0.01,
                        "alpha": 0.45,
                    },
                    {
                        "critical_pressure": "high",
                        "failure_rate_per_window": 0.01,
                        "alpha": 0.85,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    alpha_policy._load_policy.cache_clear()
    monkeypatch.setenv(ENV_ALPHA_POLICY, str(path))

    decision = choose_alpha(
        critical_agent=True,
        decode_dominant=True,
        stage_width=3,
        critical_pressure=0.62,
        failure_rate_per_window=0.01,
    )

    assert decision.critical_pressure == pytest.approx(0.62)
    assert decision.alpha == 0.45
