from __future__ import annotations

import json

import pytest

from sglsim.srt.simulation import alpha_policy
from sglsim.srt.simulation.alpha_policy import (
    ENV_ALPHA_POLICY,
    ENV_FAILURE_RATE_PER_WINDOW,
    AlphaPolicyDecision,
    OnlineAlphaAdapter,
    choose_alpha,
    fine_tune_alpha,
    online_alpha_adapter_enabled,
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


def test_online_fine_tune_alpha_raises_on_slow_high_risk_trace():
    update = fine_tune_alpha(
        base_alpha=0.30,
        observed_wct_ratio=1.35,
        failure_rate_per_window=0.10,
        hbm_pressure=0.25,
    )

    assert update.source == "online_alpha_adapter:v1"
    assert update.alpha > 0.30
    assert update.alpha <= 0.60


def test_online_fine_tune_alpha_reduces_under_hbm_pressure():
    update = fine_tune_alpha(
        base_alpha=0.45,
        observed_wct_ratio=0.95,
        failure_rate_per_window=0.001,
        hbm_pressure=0.95,
    )

    assert update.alpha < 0.45
    assert update.alpha >= 0.05


def test_online_alpha_adapter_tracks_latency_and_tunes_mdp_action():
    adapter = OnlineAlphaAdapter()
    adapter.observe(
        critical_agent=True,
        decode_dominant=True,
        stage_width=3,
        latency_s=10.0,
    )
    adapter.observe(
        critical_agent=True,
        decode_dominant=True,
        stage_width=3,
        latency_s=13.5,
    )
    decision = AlphaPolicyDecision(
        alpha=0.3,
        critical_pressure=0.25,
        failure_rate_per_window=0.1,
        source="policy_table:test",
    )
    tuned = adapter.tune_decision(
        decision,
        critical_agent=True,
        decode_dominant=True,
        stage_width=3,
    )
    assert tuned.alpha > 0.3
    assert tuned.source == "online_alpha_adapter:v1"


def test_online_alpha_adapter_enabled_from_env(monkeypatch):
    monkeypatch.delenv("SGLSIM_AEGIS_ONLINE_ALPHA_ADAPTER", raising=False)
    assert not online_alpha_adapter_enabled()
    monkeypatch.setenv("SGLSIM_AEGIS_ONLINE_ALPHA_ADAPTER", "1")
    assert online_alpha_adapter_enabled()
