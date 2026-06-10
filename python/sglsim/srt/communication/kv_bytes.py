from __future__ import annotations

from dataclasses import dataclass

from sglsim.srt.communication.config import DEFAULT_KV_BYTES_PER_TOKEN
from sglsim.srt.simulation.manager.cache_config import _kv_bytes_per_token, _model_dims
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.utils.prompt_builder import _MOCK_MODEL_PATHS


@dataclass(frozen=True)
class KVByteInfo:
    bytes_per_token_all_layers: int
    bytes_per_token_per_layer: int
    num_layers: int


def resolve_kv_byte_info(
    *,
    model_path: str,
    sched: SchedulerSimConfig,
    kv_bytes_per_token_override: int | None = None,
) -> KVByteInfo:
    if kv_bytes_per_token_override is not None:
        bpt = max(1, int(kv_bytes_per_token_override))
        num_layers = max(1, _resolve_num_layers(model_path, sched))
        return KVByteInfo(
            bytes_per_token_all_layers=bpt,
            bytes_per_token_per_layer=max(1, bpt // num_layers),
            num_layers=num_layers,
        )

    bpt = _kv_bytes_per_token(model_path, sched)
    dims = _model_dims(model_path)
    if bpt is not None and dims is not None:
        _, num_layers_raw, _ = dims
        num_layers = max(1, int(num_layers_raw) // max(1, sched.pp_size))
        return KVByteInfo(
            bytes_per_token_all_layers=bpt,
            bytes_per_token_per_layer=max(1, bpt // num_layers),
            num_layers=num_layers,
        )

    if model_path in _MOCK_MODEL_PATHS:
        bpt = DEFAULT_KV_BYTES_PER_TOKEN
        num_layers = 32
        return KVByteInfo(
            bytes_per_token_all_layers=bpt,
            bytes_per_token_per_layer=max(1, bpt // num_layers),
            num_layers=num_layers,
        )

    bpt = DEFAULT_KV_BYTES_PER_TOKEN
    num_layers = max(1, _resolve_num_layers(model_path, sched))
    return KVByteInfo(
        bytes_per_token_all_layers=bpt,
        bytes_per_token_per_layer=max(1, bpt // num_layers),
        num_layers=num_layers,
    )


def _resolve_num_layers(model_path: str, sched: SchedulerSimConfig) -> int:
    dims = _model_dims(model_path)
    if dims is not None:
        return max(1, int(dims[1]) // max(1, sched.pp_size))
    return 32


def kv_page_bytes_all_layers(page_size: int, info: KVByteInfo) -> int:
    return max(1, page_size * info.bytes_per_token_all_layers)


def kv_page_bytes_per_layer(page_size: int, info: KVByteInfo) -> int:
    return max(1, page_size * info.bytes_per_token_per_layer)
