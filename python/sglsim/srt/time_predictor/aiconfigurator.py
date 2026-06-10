"""AIConfigurator latency predictor."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from sglsim.srt.spec.accelerator.base import AcceleratorInfo
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.time_predictor.base import InferTimePredictor, ScheduleBatch

logger = logging.getLogger(__name__)

_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"
# Quantize decode/ctx ISL for static latency cache keys; linear interp between grid points.
_ISL_GRID = 256


def _gemm_probe_path(systems_path: str) -> Path:
    return Path(systems_path) / "data/a100_sxm/sglang/0.5.10/gemm_perf.txt"


def _systems_paths_have_lfs_pointers(paths: list[str]) -> bool:
    for base in paths:
        gemm = _gemm_probe_path(base)
        if not gemm.is_file():
            continue
        first = gemm.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if first.startswith(_LFS_POINTER_PREFIX):
            return True
    return False


def resolve_systems_paths(raw: dict | None = None) -> list[str]:
    raw = raw or {}
    database_path = raw.get("database_path") or os.environ.get("AICONFIGURATOR_SYSTEMS_PATH")
    if database_path:
        return [str(Path(database_path).expanduser())]

    from aiconfigurator.sdk.perf_database import get_systems_paths

    default_paths = list(get_systems_paths())
    if _systems_paths_have_lfs_pointers(default_paths):
        raise ValueError(
            "AIConfigurator perf data in the installed package are git-lfs pointers. "
            "Clone https://github.com/ai-dynamo/aiconfigurator, run `git lfs pull`, "
            "then either:\n"
            "  uv pip install /path/to/aiconfigurator\n"
            "  export AICONFIGURATOR_SYSTEMS_PATH=/path/to/aiconfigurator/src/aiconfigurator/systems"
        )
    return default_paths


def aiconfigurator_database_available(
    *,
    system: str = "a100_sxm",
    backend: str = "sglang",
    version: str = "0.5.10",
    raw: dict | None = None,
) -> bool:
    try:
        from aiconfigurator.sdk.perf_database import get_database

        return (
            get_database(
                system=system,
                backend=backend,
                version=version,
                systems_paths=resolve_systems_paths(raw),
            )
            is not None
        )
    except Exception:
        return False


def attention_dp_size_for_scheduler(sched_config: SchedulerSimConfig) -> int:
    """Match SGLang dense default: replica-level DP does not split attention."""
    return 1


def _get_perf_model(sched_config: SchedulerSimConfig, model: ModelInfo):
    from aiconfigurator.sdk import models
    from aiconfigurator.sdk.common import (
        CommQuantMode,
        FMHAQuantMode,
        GEMMQuantMode,
        KVCacheQuantMode,
        MoEQuantMode,
    )
    from aiconfigurator.sdk.config import ModelConfig

    model_config = ModelConfig(
        pp_size=sched_config.pp_size,
        tp_size=sched_config.tp_size,
        moe_tp_size=sched_config.tp_size // max(sched_config.ep_size, 1),
        moe_ep_size=sched_config.ep_size,
        attention_dp_size=attention_dp_size_for_scheduler(sched_config),
        gemm_quant_mode=GEMMQuantMode.bfloat16,
        moe_quant_mode=MoEQuantMode.bfloat16,
        kvcache_quant_mode=KVCacheQuantMode.bfloat16,
        fmha_quant_mode=FMHAQuantMode.bfloat16,
        comm_quant_mode=CommQuantMode.half,
        workload_distribution="power_law_1.2",
    )
    return models.get_model(
        model_path=model.model_path,
        model_config=model_config,
        backend_name=sched_config.backend_name,
    )


class AIConfiguratorTimePredictor(InferTimePredictor):
    name = "aiconfigurator"

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerSimConfig,
        raw: dict | None = None,
    ) -> None:
        super().__init__(model, hw, config, raw)
        raw = raw or {}
        self.prefill_scale_factor = float(raw.get("prefill_scale_factor", 1.0))
        self.decode_scale_factor = float(raw.get("decode_scale_factor", 1.0))
        # Log-based with-hicache / no-hicache decode throughput ratio (~1.54×).
        _default_hicache_ratio = float(raw.get("hicache_decode_speedup_ratio", 1.543))
        if "decode_scale_hicache" in raw:
            self.decode_scale_hicache = float(raw["decode_scale_hicache"])
        else:
            self.decode_scale_hicache = self.decode_scale_factor / _default_hicache_ratio
        self.forward_scale = float(raw.get("forward_scale", 1.0))
        from sglsim.srt.simulation.manager.config import ConfigManager

        self._hicache_enabled = ConfigManager.get_hicache_config().enable_hierarchical_cache

        if not config.backend_version:
            raise ValueError(
                "scheduler.backend_version is required for AIConfigurator predictor"
            )

        from aiconfigurator.sdk.backends.factory import get_backend
        from aiconfigurator.sdk.common import DatabaseMode
        from aiconfigurator.sdk.config import RuntimeConfig
        from aiconfigurator.sdk.inference_session import InferenceSession
        from aiconfigurator.sdk.perf_database import get_database

        database_mode = self._database_mode(raw.get("database_mode", "SILICON"))
        systems_paths = resolve_systems_paths(raw)
        database = get_database(
            system=hw.name,
            backend=config.backend_name,
            version=config.backend_version,
            systems_paths=systems_paths,
        )
        if database is None:
            raise ValueError(
                "Failed to initialize AIConfigurator database for "
                f"system={hw.name}, backend={config.backend_name}, "
                f"version={config.backend_version}, systems_paths={systems_paths}"
            )

        database.set_default_database_mode(database_mode)
        helper = database._nearest_1d_point_helper

        def wrapped_nearest_1d_point_helper(
            x: int, values: list[int], inner_only: bool = False
        ):
            return helper(x, values, False)

        database._nearest_1d_point_helper = wrapped_nearest_1d_point_helper

        self._session = InferenceSession(
            model=_get_perf_model(config, model),
            backend=get_backend(config.backend_name),
            database=database,
        )
        self._runtime_config_cls = RuntimeConfig
        self._static_latency_cache: dict[tuple, float] = {}

    @staticmethod
    def _database_mode(mode: str):
        from aiconfigurator.sdk.common import DatabaseMode

        return {
            "SILICON": DatabaseMode.SILICON,
            "HYBRID": DatabaseMode.HYBRID,
            "EMPIRICAL": DatabaseMode.EMPIRICAL,
            "SOL": DatabaseMode.SOL,
            "SOL_FULL": DatabaseMode.SOL_FULL,
        }.get(str(mode).upper(), DatabaseMode.SILICON)

    @staticmethod
    def _grid_bounds(value: int, step: int = _ISL_GRID) -> tuple[int, int]:
        lo = (max(value, 0) // step) * step
        return lo, lo + step

    @staticmethod
    def _lerp(lo: float, hi: float, alpha: float) -> float:
        if alpha <= 0.0:
            return lo
        if alpha >= 1.0:
            return hi
        return lo * (1.0 - alpha) + hi * alpha

    def _ctx_attn_flops_ratio_with_avg(self, batch: ScheduleBatch) -> float:
        if len(batch.reqs) == 1:
            return 1.0
        mean_past = np.mean([req.past_kv_length for req in batch.reqs])
        mean_input = np.mean([req.extend_length for req in batch.reqs])
        avg_flops = (mean_past + mean_past + mean_input) * mean_input / 2 * len(batch.reqs)
        actual_flops = 0.0
        for req in batch.reqs:
            actual_flops += (
                (req.past_kv_length + req.past_kv_length + req.extend_length)
                * req.extend_length
                / 2
            )
        return actual_flops / avg_flops

    def _decode_isl(self, batch: ScheduleBatch) -> int:
        """Mean KV context length for a decode batch (one token per req this step)."""
        past_lengths = [req.past_kv_length for req in batch.reqs]
        return int(round(float(np.mean(past_lengths))))

    def _decode_spread(self, batch: ScheduleBatch) -> int | None:
        past_lengths = [req.past_kv_length for req in batch.reqs]
        if len(past_lengths) <= 1:
            return None
        spread = max(past_lengths) - min(past_lengths)
        return spread if spread > 0 else None

    def _clamp_ctx_points(self, isl: int, prefix: int) -> tuple[int, int]:
        """Ensure isl > prefix so AIConfigurator effective_isl stays positive."""
        isl = max(isl, 1)
        prefix = min(prefix, isl - 1)
        return isl, prefix

    def _run_static_latency_ms(
        self,
        mode: str,
        *,
        batch_size: int,
        isl: int,
        prefix: int | None = None,
        spread: int | None = None,
        seq_scale: float | None = None,
    ) -> float:
        if mode == "static_gen":
            cache_key: tuple
            cache_key = ("static_gen", batch_size, isl)
            runtime_config = self._runtime_config_cls(
                batch_size=batch_size, isl=isl, osl=2
            )
        else:
            if seq_scale is not None and seq_scale >= 0.4:
                cache_key = (
                    "static_ctx",
                    batch_size,
                    isl,
                    prefix if prefix is not None else 0,
                    round(seq_scale, 4),
                )
                runtime_config = self._runtime_config_cls(
                    batch_size=batch_size,
                    isl=isl,
                    prefix=prefix if prefix is not None else 0,
                    osl=1,
                    seq_imbalance_correction_scale=seq_scale,
                )
            else:
                cache_key = (
                    "static_ctx",
                    batch_size,
                    isl,
                    prefix if prefix is not None else 0,
                )
                runtime_config = self._runtime_config_cls(
                    batch_size=batch_size,
                    isl=isl,
                    prefix=prefix if prefix is not None else 0,
                    osl=1,
                )

        cached = self._static_latency_cache.get(cache_key)
        if cached is not None:
            return cached

        summary = self._session.run_static(runtime_config, mode=mode)
        if mode == "static_gen":
            latency_dict = summary.get_generation_latency_dict()
        else:
            latency_dict = summary.get_context_latency_dict()

        infer_time = sum(latency_dict.values())
        if summary.check_oom():
            logger.warning(
                "AIConfigurator reported OOM during estimation "
                "mode=%s batch_size=%d isl=%d prefix=%s spread=%s seq_scale=%s sum_ms=%.6f",
                mode,
                batch_size,
                isl,
                prefix,
                spread,
                seq_scale,
                infer_time,
            )
            # OOM here means AIConfigurator could not validate the static point,
            # not that the simulator should skip the forward. SGLang would still
            # spend positive wall time on an admitted batch; keep the latency
            # estimate positive and let scheduler/KV admission handle memory.
            infer_time = abs(infer_time)
        result = infer_time / 1e3
        self._static_latency_cache[cache_key] = result
        return result

    def _interpolate_decode(
        self,
        *,
        batch_size: int,
        isl: int,
        spread: int | None,
        scale: float,
    ) -> float:
        lo, hi = self._grid_bounds(isl)
        t_lo = self._run_static_latency_ms(
            "static_gen",
            batch_size=batch_size,
            isl=lo,
            spread=spread,
        )
        if hi == lo:
            return t_lo * scale
        alpha = (isl - lo) / float(hi - lo)
        t_hi = self._run_static_latency_ms(
            "static_gen",
            batch_size=batch_size,
            isl=hi,
            spread=spread,
        )
        return self._lerp(t_lo, t_hi, alpha) * scale

    def _predict_ctx_latency(self, batch: ScheduleBatch, scale: float) -> float:
        mean_past = int(np.mean([req.past_kv_length for req in batch.reqs]))
        mean_input = int(np.mean([req.extend_length for req in batch.reqs]))
        if mean_input <= 0:
            return 0.0
        isl = mean_past + mean_input
        prefix = mean_past
        seq_scale = self._ctx_attn_flops_ratio_with_avg(batch)
        seq_arg = seq_scale if seq_scale >= 0.4 else None

        isl_lo, isl_hi = self._grid_bounds(isl)
        prefix_lo, prefix_hi = self._grid_bounds(prefix)
        if isl_lo == isl_hi and prefix_lo == prefix_hi:
            isl_pt, prefix_pt = self._clamp_ctx_points(isl_lo, prefix_lo)
            return self._run_static_latency_ms(
                "static_ctx",
                batch_size=batch.batch_size,
                isl=isl_pt,
                prefix=prefix_pt,
                seq_scale=seq_arg,
            ) * scale

        def at(isl_pt: int, prefix_pt: int) -> float:
            isl_pt, prefix_pt = self._clamp_ctx_points(isl_pt, prefix_pt)
            return self._run_static_latency_ms(
                "static_ctx",
                batch_size=batch.batch_size,
                isl=isl_pt,
                prefix=prefix_pt,
                seq_scale=seq_arg,
            )

        t00 = at(isl_lo, prefix_lo)
        t01 = at(isl_lo, prefix_hi)
        t10 = at(isl_hi, prefix_lo)
        t11 = at(isl_hi, prefix_hi)
        alpha_isl = 0.0 if isl_lo == isl_hi else (isl - isl_lo) / float(isl_hi - isl_lo)
        alpha_prefix = (
            0.0 if prefix_lo == prefix_hi else (prefix - prefix_lo) / float(prefix_hi - prefix_lo)
        )
        t0 = self._lerp(t00, t01, alpha_prefix)
        t1 = self._lerp(t10, t11, alpha_prefix)
        return self._lerp(t0, t1, alpha_isl) * scale

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0
        if batch.is_decode():
            isl = self._decode_isl(batch)
            spread = self._decode_spread(batch)
            scale = (
                self.decode_scale_hicache
                if self._hicache_enabled
                else self.decode_scale_factor
            )
            predicted = self._interpolate_decode(
                batch_size=batch.batch_size,
                isl=isl,
                spread=spread,
                scale=scale,
            )
        else:
            predicted = self._predict_ctx_latency(batch, self.prefill_scale_factor)
        return predicted * self.forward_scale
