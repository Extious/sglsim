from __future__ import annotations

from sglsim.srt.communication.config import AlphaBetaPathConfig, CommunicationConfig
from sglsim.srt.communication.kv_bytes import (
    KVByteInfo,
    kv_page_bytes_all_layers,
    kv_page_bytes_per_layer,
)
from sglsim.srt.communication.segments import (
    split_contiguous_page_segments,
    split_contiguous_page_segments_from_ids,
)
from sglsim.srt.communication.types import (
    CommPath,
    CommunicationPrediction,
    CommunicationRequest,
)
from sglsim.srt.simulation.manager.platform_config import PlatformConfig
from sglsim.srt.simulation.manager.state import StateManager


def effective_bandwidth_bytes_per_s(
    size_bytes: float,
    bandwidth_bytes_per_s: float,
    alpha_s: float,
) -> float:
    if size_bytes <= 0:
        return bandwidth_bytes_per_s
    if bandwidth_bytes_per_s <= 0:
        return 0.0
    return size_bytes * bandwidth_bytes_per_s / (alpha_s * bandwidth_bytes_per_s + size_bytes)


def segment_duration_alpha_beta(
    size_bytes: float,
    bandwidth_bytes_per_s: float,
    alpha_cfg: AlphaBetaPathConfig,
) -> float:
    if size_bytes <= 0 or bandwidth_bytes_per_s <= 0:
        return 0.0
    bw = bandwidth_bytes_per_s * alpha_cfg.efficiency
    eff_bw = effective_bandwidth_bytes_per_s(size_bytes, bw, alpha_cfg.alpha_s)
    if eff_bw <= 0:
        return 0.0
    return size_bytes / eff_bw


def segment_duration_legacy(size_bytes: float, bandwidth_bytes_per_s: float) -> float:
    if size_bytes <= 0 or bandwidth_bytes_per_s <= 0:
        return 0.0
    return size_bytes / bandwidth_bytes_per_s


class CommunicationTimePredictor:
    def __init__(
        self,
        *,
        comm_config: CommunicationConfig,
        platform: PlatformConfig,
        kv_info: KVByteInfo,
    ) -> None:
        self._comm_config = comm_config
        self._platform = platform
        self._kv_info = kv_info

    @property
    def mode(self) -> str:
        return self._comm_config.mode

    def predict(self, req: CommunicationRequest) -> CommunicationPrediction:
        if req.path == CommPath.H2D:
            pred = self._predict_h2d(req)
        elif req.path == CommPath.D2H:
            pred = self._predict_d2h(req)
        elif req.path == CommPath.DISK_WRITE:
            pred = self._predict_disk_write(req)
        elif req.path == CommPath.DISK_READ:
            pred = self._predict_disk_read(req)
        else:
            raise ValueError(f"Unsupported comm path: {req.path}")

        self._record_metrics(pred)
        return pred

    def _predict_h2d(self, req: CommunicationRequest) -> CommunicationPrediction:
        segments = split_contiguous_page_segments(
            list(req.host_page_ids),
            list(req.device_page_ids),
        )
        bw = self._platform.memory_read_bandwidth_gb * 1e9
        alpha_cfg = self._comm_config.alpha_for(CommPath.H2D.value)
        per_layer_page_bytes = kv_page_bytes_per_layer(req.page_size, self._kv_info)
        duration = 0.0
        num_bytes = 0
        for seg_pages in segments:
            size_per_layer = seg_pages * per_layer_page_bytes
            num_bytes += size_per_layer * self._kv_info.num_layers
            duration += self._kv_info.num_layers * self._segment_duration(
                float(size_per_layer),
                bw,
                alpha_cfg,
            )
        source = self._comm_config.mode if duration > 0 else "none"
        return CommunicationPrediction(
            duration_s=duration,
            source=source,
            path=CommPath.H2D,
            num_bytes=num_bytes,
        )

    def _predict_d2h(self, req: CommunicationRequest) -> CommunicationPrediction:
        segments = split_contiguous_page_segments(
            list(req.host_page_ids),
            list(req.device_page_ids),
        )
        bw = self._platform.memory_write_bandwidth_gb * 1e9
        alpha_cfg = self._comm_config.alpha_for(CommPath.D2H.value)
        page_bytes = kv_page_bytes_all_layers(req.page_size, self._kv_info)
        duration = 0.0
        num_bytes = 0
        for seg_pages in segments:
            size_bytes = float(seg_pages * page_bytes)
            num_bytes += int(size_bytes)
            duration += self._segment_duration(size_bytes, bw, alpha_cfg)
        source = self._comm_config.mode if duration > 0 else "none"
        return CommunicationPrediction(
            duration_s=duration,
            source=source,
            path=CommPath.D2H,
            num_bytes=num_bytes,
        )

    def _predict_disk_write(self, req: CommunicationRequest) -> CommunicationPrediction:
        return self._predict_disk_io(req, CommPath.DISK_WRITE, self._platform.disk_write_bandwidth_gb)

    def _predict_disk_read(self, req: CommunicationRequest) -> CommunicationPrediction:
        return self._predict_disk_io(req, CommPath.DISK_READ, self._platform.disk_read_bandwidth_gb)

    def _predict_disk_io(
        self,
        req: CommunicationRequest,
        path: CommPath,
        bandwidth_gb: float,
    ) -> CommunicationPrediction:
        page_ids = list(req.page_ids)
        if not page_ids:
            return CommunicationPrediction(
                duration_s=0.0,
                source="none",
                path=path,
                num_bytes=0,
            )
        segments = split_contiguous_page_segments_from_ids(page_ids)
        bw = bandwidth_gb * 1e9
        alpha_cfg = self._comm_config.alpha_for(path.value)
        page_bytes = kv_page_bytes_all_layers(req.page_size, self._kv_info)
        duration = 0.0
        num_bytes = 0
        for seg_pages in segments:
            size_bytes = float(seg_pages * page_bytes)
            num_bytes += int(size_bytes)
            duration += self._segment_duration(size_bytes, bw, alpha_cfg)
        source = self._comm_config.mode if duration > 0 else "none"
        return CommunicationPrediction(
            duration_s=duration,
            source=source,
            path=path,
            num_bytes=num_bytes,
        )

    def _segment_duration(
        self,
        size_bytes: float,
        bandwidth_bytes_per_s: float,
        alpha_cfg: AlphaBetaPathConfig,
    ) -> float:
        if self._comm_config.mode == "legacy":
            return segment_duration_legacy(size_bytes, bandwidth_bytes_per_s)
        return segment_duration_alpha_beta(size_bytes, bandwidth_bytes_per_s, alpha_cfg)

    @staticmethod
    def _record_metrics(pred: CommunicationPrediction) -> None:
        if pred.duration_s <= 0:
            return
        if pred.path == CommPath.H2D:
            StateManager.inc_comm_h2d_s(pred.duration_s)
        elif pred.path == CommPath.D2H:
            StateManager.inc_comm_d2h_s(pred.duration_s)
        elif pred.path == CommPath.DISK_WRITE:
            StateManager.inc_comm_disk_write_s(pred.duration_s)
