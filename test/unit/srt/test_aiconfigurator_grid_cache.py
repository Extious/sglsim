"""Tests for ISL grid quantization + interpolation in AIConfigurator predictor."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.time_predictor.aiconfigurator import AIConfiguratorTimePredictor
from sglsim.srt.time_predictor.base import ScheduleBatch, ScheduleRequest

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO / "benchmark/synthesis_requests/config/server.json"


@pytest.fixture(scope="module")
def predictor():
    if not DEFAULT_CFG.is_file():
        pytest.skip("benchmark server config missing")
    ConfigManager.load(str(DEFAULT_CFG))
    try:
        return ConfigManager.get_inference_time_predictor("Qwen/Qwen3-8B")
    except Exception as exc:
        pytest.skip(f"aiconfigurator unavailable: {exc}")


def _decode_batch(bs: int, kv: int) -> ScheduleBatch:
    return ScheduleBatch(
        reqs=[ScheduleRequest(extend_length=1, past_kv_length=kv) for _ in range(bs)]
    )


def test_decode_grid_reuses_static_cache(predictor) -> None:
    predictor._static_latency_cache.clear()
    batch_a = _decode_batch(10, 25_377)
    batch_b = _decode_batch(10, 25_378)
    t0 = time.perf_counter()
    lat_a = predictor.predict_infer_time(batch_a)
    mid = time.perf_counter()
    lat_b = predictor.predict_infer_time(batch_b)
    elapsed = (time.perf_counter() - mid) * 1e3
    assert lat_a > 0
    assert lat_b > 0
    assert abs(lat_a - lat_b) / lat_a < 0.02
    assert elapsed < 50.0
    assert len(predictor._static_latency_cache) <= 4


def test_decode_interpolation_between_grid_points(predictor) -> None:
    predictor._static_latency_cache.clear()
    lo = 25_344
    hi = 25_600
    mid = 25_472
    t_lo = predictor.predict_infer_time(_decode_batch(10, lo))
    predictor._static_latency_cache.clear()
    t_hi = predictor.predict_infer_time(_decode_batch(10, hi))
    predictor._static_latency_cache.clear()
    t_mid = predictor.predict_infer_time(_decode_batch(10, mid))
    expected = t_lo * 0.5 + t_hi * 0.5
    assert abs(t_mid - expected) / expected < 0.02


class _SummaryStub:
    def get_generation_latency_dict(self) -> dict[str, float]:
        return {"kernel": 20.0}

    def get_context_latency_dict(self) -> dict[str, float]:
        return {"kernel": 30.0}

    def check_oom(self) -> bool:
        return False


class _SessionStub:
    def __init__(self) -> None:
        self.calls = 0

    def run_static(self, runtime_config, mode: str):
        self.calls += 1
        return _SummaryStub()


def test_run_static_latency_ms_uses_cache_before_session_call() -> None:
    predictor = AIConfiguratorTimePredictor.__new__(AIConfiguratorTimePredictor)
    predictor._runtime_config_cls = lambda **kwargs: kwargs
    predictor._static_latency_cache = {}
    session = _SessionStub()
    predictor._session = session

    first = predictor._run_static_latency_ms(
        "static_gen",
        batch_size=8,
        isl=256,
        spread=0,
    )
    second = predictor._run_static_latency_ms(
        "static_gen",
        batch_size=8,
        isl=256,
        spread=0,
    )

    assert first == pytest.approx(0.02)
    assert second == pytest.approx(first)
    assert session.calls == 1


def test_run_static_latency_ms_ignores_unused_decode_spread_in_cache_key() -> None:
    predictor = AIConfiguratorTimePredictor.__new__(AIConfiguratorTimePredictor)
    predictor._runtime_config_cls = lambda **kwargs: kwargs
    predictor._static_latency_cache = {}
    session = _SessionStub()
    predictor._session = session

    first = predictor._run_static_latency_ms(
        "static_gen",
        batch_size=8,
        isl=256,
        spread=1,
    )
    second = predictor._run_static_latency_ms(
        "static_gen",
        batch_size=8,
        isl=256,
        spread=2,
    )

    assert second == pytest.approx(first)
    assert session.calls == 1


class _OomSummaryStub:
    def get_generation_latency_dict(self) -> dict[str, float]:
        return {"kernel": -12.0}

    def get_context_latency_dict(self) -> dict[str, float]:
        return {"kernel": -18.0}

    def check_oom(self) -> bool:
        return True


class _OomSessionStub:
    def run_static(self, runtime_config, mode: str):
        return _OomSummaryStub()


def test_run_static_latency_ms_keeps_oom_fallback_positive() -> None:
    predictor = AIConfiguratorTimePredictor.__new__(AIConfiguratorTimePredictor)
    predictor._runtime_config_cls = lambda **kwargs: kwargs
    predictor._static_latency_cache = {}
    predictor._session = _OomSessionStub()

    result = predictor._run_static_latency_ms(
        "static_gen",
        batch_size=4,
        isl=1024,
        spread=128,
    )

    assert result == pytest.approx(0.012)
    assert next(iter(predictor._static_latency_cache.values())) == pytest.approx(0.012)
