from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from sglsim.srt.config import CacheConfig
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.utils.prompt_builder import _MOCK_MODEL_PATHS

logger = logging.getLogger(__name__)

_DEFAULT_PAGE_SIZE = 16
_DEFAULT_HICACHE_RATIO = 2.0
_MIN_L1_PAGES = 64


@dataclass(frozen=True)
class HiCacheSimConfig:
    enable_hierarchical_cache: bool = False
    hicache_ratio: float = _DEFAULT_HICACHE_RATIO
    hicache_size_gb: float = 0.0
    hicache_storage_backend: str | None = None
    hicache_storage_size_gb: float = 0.0
    hicache_write_policy: str = "write_through_selective"
    write_through_threshold: int = 2
    load_back_threshold: int = 10
    prefetch_threshold: int = 256
    prefetch_stop_policy: str = "best_effort"
    prefetch_capacity_limit_ratio: float = 0.8


def load_hicache_config(raw: dict | None) -> HiCacheSimConfig:
    raw = raw if isinstance(raw, dict) else {}
    backend = raw.get("hicache_storage_backend")
    return HiCacheSimConfig(
        enable_hierarchical_cache=bool(raw.get("enable_hierarchical_cache", False)),
        hicache_ratio=float(raw.get("hicache_ratio", _DEFAULT_HICACHE_RATIO)),
        hicache_size_gb=float(raw.get("hicache_size_gb", raw.get("hicache_size", 0)) or 0),
        hicache_storage_backend=str(backend) if backend else None,
        hicache_storage_size_gb=float(
            raw.get("hicache_storage_size_gb", raw.get("hicache_storage_size", 0)) or 0
        ),
        hicache_write_policy=str(
            raw.get("hicache_write_policy", "write_through_selective")
        ),
        write_through_threshold=int(raw.get("write_through_threshold", 2)),
        load_back_threshold=int(raw.get("load_back_threshold", 10)),
        prefetch_threshold=int(raw.get("prefetch_threshold", 256)),
        prefetch_stop_policy=str(raw.get("prefetch_stop_policy", "best_effort")),
        prefetch_capacity_limit_ratio=float(
            raw.get("prefetch_capacity_limit_ratio", 0.8)
        ),
    )


def _estimate_weight_bytes(
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    *,
    bytes_per_param: int = 2,
) -> int:
    params = 12 * num_layers * hidden_size * hidden_size + 2 * vocab_size * hidden_size
    return int(params * bytes_per_param)


def _kv_bytes_per_token(model_path: str, sched: SchedulerSimConfig) -> int | None:
    if model_path in _MOCK_MODEL_PATHS:
        return None
    try:
        from sglsim.srt.time_predictor.aiconfigurator import _get_perf_model

        perf_model = _get_perf_model(sched, ModelInfo.from_model_path(model_path))
        full_pipeline_bytes = int(
            perf_model.config.kvcache_quant_mode.value.memory
            * perf_model.get_kvcache_elements_per_token()
        )
        # AIConfigurator partitions KV heads for TP but reports all model
        # layers even when the deployment is pipeline-parallel.  A physical PP
        # rank stores KV only for its local (ceil-balanced) layer slice.
        parsed = _model_profile(model_path)
        if parsed is None:
            return full_pipeline_bytes
        num_layers = max(int(parsed["layers"]), 1)
        local_layers = math.ceil(num_layers / max(int(sched.pp_size), 1))
        return max(
            int(math.ceil(full_pipeline_bytes * local_layers / num_layers)),
            1,
        )
    except Exception as exc:
        logger.warning(
            "Failed to estimate KV bytes per token for model_path=%s: %s",
            model_path,
            exc,
        )
        return None


def _model_profile(model_path: str) -> dict[str, int] | None:
    if model_path in _MOCK_MODEL_PATHS:
        return None
    try:
        from aiconfigurator.sdk.utils import get_model_config_from_model_path

        parsed = get_model_config_from_model_path(model_path)
        return {
            "hidden_size": int(parsed["hidden_size"]),
            "layers": int(parsed["layers"]),
            "vocab": int(parsed["vocab"]),
            "inter_size": int(parsed.get("inter_size", 4 * parsed["hidden_size"])),
            "n": int(parsed.get("n", 1)),
            "n_kv": int(parsed.get("n_kv", parsed.get("n", 1))),
            "head_dim": int(
                parsed.get(
                    "d",
                    int(parsed["hidden_size"]) // max(int(parsed.get("n", 1)), 1),
                )
            ),
            "tie_word_embeddings": int(
                bool(dict(parsed.get("raw_config") or {}).get("tie_word_embeddings", False))
            ),
        }
    except Exception as exc:
        logger.warning(
            "Failed to load model dims for model_path=%s: %s",
            model_path,
            exc,
        )
        return None


def _model_dims(model_path: str) -> tuple[int, int, int] | None:
    profile = _model_profile(model_path)
    if profile is None:
        return None
    return (
        profile["hidden_size"],
        profile["layers"],
        profile["vocab"],
    )


def _estimate_per_rank_weight_bytes(
    profile: dict[str, int],
    sched: SchedulerSimConfig,
    *,
    bytes_per_param: int = 2,
) -> int:
    """Estimate the largest TP/PP rank's resident dense-model weights.

    This mirrors the physical ownership boundary that matters to SGLang's KV
    pool sizing: a PP rank owns only its local transformer layers, and a TP
    rank owns a shard of their matrices.  The first/last PP stages additionally
    own one vocabulary matrix (embedding or untied LM head), so the largest
    stage includes one such matrix rather than charging every rank for the
    complete model.
    """

    hidden = max(profile["hidden_size"], 1)
    layers = max(profile["layers"], 1)
    inter = max(profile["inter_size"], 1)
    n_heads = max(profile["n"], 1)
    n_kv = max(profile["n_kv"], 1)
    head_dim = max(profile["head_dim"], 1)
    tp_size = max(int(sched.tp_size), 1)
    pp_size = max(int(sched.pp_size), 1)
    local_layers = math.ceil(layers / pp_size)

    # Q/K/V/O plus gated MLP matrices.  Matrix sharding is approximated as
    # balanced across TP ranks; this is exact for the current TP=1 target.
    attention_params = (
        hidden * n_heads * head_dim
        + 2 * hidden * n_kv * head_dim
        + n_heads * head_dim * hidden
    )
    mlp_params = 3 * hidden * inter
    norm_params = 2 * hidden
    local_transformer_params = math.ceil(
        local_layers * (attention_params + mlp_params + norm_params) / tp_size
    )
    # With untied embeddings, stage 0 and the last stage each own one matrix;
    # with tied embeddings the same shared matrix is still resident on an edge
    # stage.  Use one vocabulary matrix for the largest-rank estimate.
    edge_vocab_params = math.ceil(profile["vocab"] * hidden / tp_size)
    return int((local_transformer_params + edge_vocab_params) * bytes_per_param)


def infer_l1_max_tokens(
    *,
    model_path: str,
    hbm_capacity_gb: float,
    mem_fraction_static: float,
    sched: SchedulerSimConfig,
) -> int:
    if model_path in _MOCK_MODEL_PATHS:
        return 65536

    bytes_per_token = _kv_bytes_per_token(model_path, sched)
    profile = _model_profile(model_path)
    if bytes_per_token is None or profile is None:
        return 65536

    hbm_bytes = int(hbm_capacity_gb * (1024**3))
    static_bytes = int(hbm_bytes * mem_fraction_static)
    weight_bytes = _estimate_per_rank_weight_bytes(profile, sched)
    kv_bytes = max(0, static_bytes - weight_bytes)
    if kv_bytes <= 0:
        return 4096
    return max(4096, kv_bytes // bytes_per_token)


def _tokens_to_pages(token_capacity: int, page_size: int) -> int:
    aligned = (max(token_capacity, 0) // page_size) * page_size
    return max(_MIN_L1_PAGES, aligned // page_size)


def apply_kv_capacity_scale(token_capacity: int, cache_raw: dict | None) -> int:
    raw = cache_raw if isinstance(cache_raw, dict) else {}
    scale = float(raw.get("kv_capacity_scale_factor", 1.0))
    if scale <= 0:
        raise ValueError("kv_capacity_scale_factor must be positive")
    return max(1, int(token_capacity * scale))


def resolve_cache_config(
    *,
    model_path: str,
    hbm_capacity_gb: float,
    sched: SchedulerSimConfig,
    cache_raw: dict | None,
    hicache: HiCacheSimConfig,
) -> CacheConfig:
    raw = cache_raw if isinstance(cache_raw, dict) else {}
    page_size = int(raw.get("page_size", _DEFAULT_PAGE_SIZE))

    explicit_l1 = raw.get("l1_capacity_pages")
    explicit_aegis_total_l1 = raw.get("aegis_total_l1_capacity_pages")
    explicit_l2 = raw.get("l2_capacity_pages")
    explicit_l3 = raw.get("l3_capacity_pages")

    if explicit_l1 is not None:
        l1_pages = int(explicit_l1)
        # kv_capacity_scale_factor compensates for page_size=1 in the
        # simulator tracking every token as an exact page, while real GPU
        # allocators carry internal fragmentation (alignment, page rounding)
        # that implicitly absorbs the last ~0.3-0.5 % of capacity pressure.
        l1_tokens_scaled = apply_kv_capacity_scale(l1_pages * page_size, raw)
        l1_pages = _tokens_to_pages(l1_tokens_scaled, page_size)
    else:
        l1_tokens = infer_l1_max_tokens(
            model_path=model_path,
            hbm_capacity_gb=hbm_capacity_gb,
            mem_fraction_static=sched.mem_fraction_static,
            sched=sched,
        )
        l1_tokens = apply_kv_capacity_scale(l1_tokens, raw)
        l1_pages = _tokens_to_pages(l1_tokens, page_size)
        logger.info(
            "Inferred l1_capacity_pages=%d (tokens=%d page_size=%d) from HBM %.0f GB "
            "mem_fraction_static=%.2f model=%s",
            l1_pages,
            l1_pages * page_size,
            page_size,
            hbm_capacity_gb,
            sched.mem_fraction_static,
            model_path,
        )

    bytes_per_token = _kv_bytes_per_token(model_path, sched)

    if explicit_l2 is not None:
        l2_pages = int(explicit_l2)
    elif hicache.enable_hierarchical_cache and bytes_per_token is not None:
        if hicache.hicache_size_gb > 0:
            l2_tokens = int(hicache.hicache_size_gb * 1e9 // bytes_per_token)
        else:
            l2_tokens = int(l1_pages * page_size * hicache.hicache_ratio)
        l2_pages = _tokens_to_pages(l2_tokens, page_size)
        logger.info(
            "Inferred l2_capacity_pages=%d (hicache_ratio=%.2f hicache_size_gb=%.2f)",
            l2_pages,
            hicache.hicache_ratio,
            hicache.hicache_size_gb,
        )
    else:
        l2_pages = 0

    if explicit_l3 is not None:
        l3_pages = int(explicit_l3)
    elif hicache.hicache_storage_backend and bytes_per_token is not None:
        if hicache.hicache_storage_size_gb > 0:
            l3_tokens = int(hicache.hicache_storage_size_gb * 1e9 // bytes_per_token)
            l3_pages = _tokens_to_pages(l3_tokens, page_size)
        else:
            l3_pages = l2_pages
        logger.info(
            "Inferred l3_capacity_pages=%d storage_backend=%s",
            l3_pages,
            hicache.hicache_storage_backend,
        )
    else:
        l3_pages = 0

    write_policy = hicache.hicache_write_policy
    if write_policy == "write_through":
        write_threshold = 1
    elif write_policy == "write_through_selective":
        write_threshold = hicache.write_through_threshold
    else:
        write_threshold = hicache.write_through_threshold

    return CacheConfig(
        page_size=page_size,
        l1_capacity_pages=l1_pages,
        aegis_total_l1_capacity_pages=(
            int(explicit_aegis_total_l1)
            if explicit_aegis_total_l1 is not None
            else None
        ),
        l2_capacity_pages=l2_pages,
        l3_capacity_pages=l3_pages,
        hicache_write_policy=write_policy,
        write_through_threshold=write_threshold,
        load_back_threshold=hicache.load_back_threshold,
        prefetch_threshold=hicache.prefetch_threshold,
        prefetch_stop_policy=hicache.prefetch_stop_policy,
        prefetch_capacity_limit_ratio=hicache.prefetch_capacity_limit_ratio,
        hicache_storage_backend=hicache.hicache_storage_backend,
    )
