import pytest

from sglsim.srt.communication.config import (
    AlphaBetaPathConfig,
    CommunicationConfig,
    DEFAULT_D2H_ALPHA_S,
    DEFAULT_EFFICIENCY,
    DEFAULT_H2D_ALPHA_S,
)
from sglsim.srt.communication.kv_bytes import KVByteInfo
from sglsim.srt.communication.predictor import (
    CommunicationTimePredictor,
    effective_bandwidth_bytes_per_s,
    segment_duration_alpha_beta,
)
from sglsim.srt.communication.types import CommPath, CommunicationRequest
from sglsim.srt.simulation.manager.platform_config import PlatformConfig


def _sglang_est_duration(size_bytes: float, bandwidth_bytes_per_s: float, t0: float) -> float:
    bw = bandwidth_bytes_per_s * DEFAULT_EFFICIENCY
    eff_bw = effective_bandwidth_bytes_per_s(size_bytes, bw, t0)
    return size_bytes / eff_bw


def test_alpha_beta_matches_sglang_simulator_formula():
    bw = 64e9
    size = 16 * 128
    expected = _sglang_est_duration(size, bw, DEFAULT_H2D_ALPHA_S)
    got = segment_duration_alpha_beta(
        size,
        bw,
        AlphaBetaPathConfig(alpha_s=DEFAULT_H2D_ALPHA_S, efficiency=DEFAULT_EFFICIENCY),
    )
    assert got == pytest.approx(expected)


def test_legacy_mode_is_linear_bandwidth():
    predictor = CommunicationTimePredictor(
        comm_config=CommunicationConfig(mode="legacy"),
        platform=PlatformConfig(memory_read_bandwidth_gb=64.0),
        kv_info=KVByteInfo(
            bytes_per_token_all_layers=256,
            bytes_per_token_per_layer=8,
            num_layers=32,
        ),
    )
    pred = predictor.predict(
        CommunicationRequest(
            path=CommPath.H2D,
            host_page_ids=(0, 1),
            device_page_ids=(0, 1),
            page_size=16,
        )
    )
    num_bytes = 2 * 16 * 8 * 32
    assert pred.duration_s == pytest.approx(num_bytes / (64e9))


def test_d2h_uses_write_bandwidth_and_d2h_alpha():
    predictor = CommunicationTimePredictor(
        comm_config=CommunicationConfig(mode="alpha_beta"),
        platform=PlatformConfig(memory_write_bandwidth_gb=50.0),
        kv_info=KVByteInfo(
            bytes_per_token_all_layers=512,
            bytes_per_token_per_layer=16,
            num_layers=32,
        ),
    )
    pred = predictor.predict(
        CommunicationRequest(
            path=CommPath.D2H,
            host_page_ids=(0,),
            device_page_ids=(0,),
            page_size=16,
        )
    )
    size = 16 * 512
    expected = _sglang_est_duration(size, 50e9, DEFAULT_D2H_ALPHA_S)
    assert pred.duration_s == pytest.approx(expected)


def test_h2d_alpha_beta_matches_sglang_per_layer_load():
    predictor = CommunicationTimePredictor(
        comm_config=CommunicationConfig(mode="alpha_beta"),
        platform=PlatformConfig(memory_read_bandwidth_gb=64.0),
        kv_info=KVByteInfo(
            bytes_per_token_all_layers=512,
            bytes_per_token_per_layer=16,
            num_layers=32,
        ),
    )
    pred = predictor.predict(
        CommunicationRequest(
            path=CommPath.H2D,
            host_page_ids=(0,),
            device_page_ids=(0,),
            page_size=1,
        )
    )
    expected = 32 * _sglang_est_duration(16, 64e9, DEFAULT_H2D_ALPHA_S)
    all_layers_once = _sglang_est_duration(512, 64e9, DEFAULT_H2D_ALPHA_S)

    assert pred.duration_s == pytest.approx(expected)
    assert pred.duration_s > all_layers_once
