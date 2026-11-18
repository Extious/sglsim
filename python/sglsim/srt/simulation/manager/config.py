from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from sglsim.srt.config import CacheConfig
from sglsim.srt.simulation.manager.env import config_path
from sglsim.srt.spec.accelerator.base import AcceleratorInfo
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.simulation.manager.cache_config import (
    HiCacheSimConfig,
    apply_kv_capacity_scale,
    load_hicache_config,
    resolve_cache_config,
)
from sglsim.srt.model_executor.model_runner_kv_cache_mixin import resolve_max_num_reqs
from sglsim.srt.simulation.manager.cache_config import infer_l1_max_tokens
from sglsim.srt.simulation.manager.scheduler_config import (
    SchedulerSimConfig,
    _parse_max_running,
    hbm_capacity_gb_from_platform,
    resolve_chunked_prefill_size,
)
from sglsim.srt.communication.config import CommunicationConfig
from sglsim.srt.communication.kv_bytes import resolve_kv_byte_info
from sglsim.srt.simulation.manager.platform_config import PlatformConfig
from sglsim.srt.time_predictor.base import InferTimePredictor
from sglsim.srt.utils.prompt_builder import _MOCK_MODEL_PATHS

logger = logging.getLogger(__name__)


class ConfigManager:
    _raw: dict[str, Any] | None = None
    _model: ModelInfo | None = None
    _scheduler: SchedulerSimConfig | None = None
    _platform: PlatformConfig | None = None
    _cache: CacheConfig | None = None
    _hicache: HiCacheSimConfig | None = None
    _predictor: InferTimePredictor | None = None
    _communication: CommunicationConfig | None = None
    _comm_predictor: Any | None = None

    @classmethod
    def apply_server_args_overrides(cls, server_args) -> None:
        if cls._raw is None:
            cls._raw = {}
        sched = cls._raw.setdefault("scheduler", {})
        if not isinstance(sched, dict):
            sched = {}
            cls._raw["scheduler"] = sched
        # Runtime process topology must have one authoritative view.  The
        # existing DP resolver already treats the larger of the simulation
        # config and CLI value as active; mirror that rule for every parallel
        # dimension before any Controller, Scheduler, cache, or predictor is
        # constructed.
        for key in ("dp_size", "tp_size", "pp_size"):
            configured = max(int(sched.get(key, 1)), 1)
            requested = max(int(getattr(server_args, key)), 1)
            sched[key] = max(configured, requested)
        if server_args.max_running_requests is not None:
            sched["max_running_requests"] = server_args.max_running_requests
        if server_args.load_balance_method is not None:
            sched["load_balance_method"] = server_args.load_balance_method
        if server_args.chunked_prefill_size is not None:
            sched["chunked_prefill_size"] = server_args.chunked_prefill_size
        if server_args.disable_overlap_schedule:
            sched["disable_overlap_schedule"] = True
        cls._scheduler = None
        cls._cache = None
        cls._communication = None
        cls._comm_predictor = None

    @classmethod
    def load(cls, path: str | None = None) -> None:
        cfg_path = path or config_path()
        if cfg_path and Path(cfg_path).exists():
            cls._raw = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        else:
            cls._raw = {}
        cls._model = None
        cls._scheduler = None
        cls._platform = None
        cls._cache = None
        cls._hicache = None
        cls._predictor = None
        cls._communication = None
        cls._comm_predictor = None

    @classmethod
    def get_scheduler_config(cls) -> SchedulerSimConfig:
        if cls._scheduler is None:
            raw = (cls._raw or {}).get("scheduler", {})
            if not isinstance(raw, dict):
                raw = {}
            platform_raw = (cls._raw or {}).get("platform", {})
            hbm_gb = hbm_capacity_gb_from_platform(platform_raw)
            infer_chunked = raw.get("chunked_prefill_size") is None
            chunked_prefill_size = resolve_chunked_prefill_size(
                raw.get("chunked_prefill_size"),
                hbm_capacity_gb=hbm_gb,
                infer_if_missing=infer_chunked,
            )
            if infer_chunked and chunked_prefill_size is not None:
                logger.info(
                    "Inferred chunked_prefill_size=%d from platform HBM %.0f GB",
                    chunked_prefill_size,
                    hbm_gb,
                )
            dp_size = int(raw.get("dp_size", 1))
            user_max_running = _parse_max_running(raw)
            model_path = str((cls._raw or {}).get("model_path", "mock-model"))
            context_len = cls._resolve_context_len(model_path) or 40960
            token_capacity = infer_l1_max_tokens(
                model_path=model_path,
                hbm_capacity_gb=hbm_gb,
                mem_fraction_static=float(raw.get("mem_fraction_static", 0.88)),
                sched=SchedulerSimConfig(
                    tp_size=int(raw.get("tp_size", 1)),
                    dp_size=dp_size,
                    pp_size=int(raw.get("pp_size", 1)),
                    mem_fraction_static=float(raw.get("mem_fraction_static", 0.88)),
                ),
            )
            token_capacity = apply_kv_capacity_scale(
                token_capacity,
                (cls._raw or {}).get("cache", {}),
            )
            if (
                chunked_prefill_size is not None
                and chunked_prefill_size > token_capacity
            ):
                raise ValueError(
                    "chunked_prefill_size exceeds per-rank L1 KV capacity: "
                    f"chunk={chunked_prefill_size}, capacity={token_capacity}, "
                    f"model={model_path}, pp_size={int(raw.get('pp_size', 1))}. "
                    "Use a physically feasible TP/PP deployment, raise the "
                    "static-memory budget, or explicitly lower the chunk size."
                )
            resolved_max_running = resolve_max_num_reqs(
                token_capacity=token_capacity,
                context_len=context_len,
                dp_size=dp_size,
                user_max_running_requests=user_max_running,
            )
            if user_max_running is None:
                logger.info(
                    "Resolved max_running_requests=%d (tokens=%d dp_size=%d)",
                    resolved_max_running,
                    token_capacity,
                    dp_size,
                )
            pp_size = int(raw.get("pp_size", 1))
            pp_max_micro_batch = raw.get("pp_max_micro_batch_size")
            if pp_max_micro_batch is None:
                pp_max_micro_batch = max(resolved_max_running // max(pp_size, 1), 1)
            cls._scheduler = SchedulerSimConfig(
                tp_size=int(raw.get("tp_size", 1)),
                dp_size=dp_size,
                ep_size=int(raw.get("ep_size", 1)),
                pp_size=pp_size,
                backend_name=str(raw.get("backend_name", "sglang")),
                backend_version=raw.get("backend_version", "0.5.10"),
                mem_fraction_static=float(raw.get("mem_fraction_static", 0.88)),
                max_running_reqs=resolved_max_running,
                max_running_requests=resolved_max_running,
                max_prefill_tokens=int(raw.get("max_prefill_tokens", 16384)),
                chunked_prefill_size=chunked_prefill_size,
                disable_overlap_schedule=bool(
                    raw.get("disable_overlap_schedule", False)
                )
                or pp_size > 1,
                load_balance_method=str(
                    raw.get("load_balance_method", "round_robin")
                ),
                schedule_policy=str(raw.get("schedule_policy", "fcfs")),
                decode_log_interval=int(raw.get("decode_log_interval", 40)),
                prefill_max_requests=(
                    int(raw["prefill_max_requests"])
                    if raw.get("prefill_max_requests") is not None
                    else None
                ),
                init_new_token_ratio=float(raw.get("init_new_token_ratio", 0.7)),
                pp_max_micro_batch_size=int(pp_max_micro_batch),
                enable_routing_key_affinity=bool(
                    raw.get("enable_routing_key_affinity", False)
                ),
            )
        return cls._scheduler

    @classmethod
    def materialize_scheduler_section(cls) -> dict[str, Any]:
        sched = cls.get_scheduler_config()
        return {
            "tp_size": sched.tp_size,
            "pp_size": sched.pp_size,
            "dp_size": sched.dp_size,
            "backend_name": sched.backend_name,
            "backend_version": sched.backend_version,
            "mem_fraction_static": sched.mem_fraction_static,
            "max_running_requests": sched.effective_max_running_requests,
            "load_balance_method": sched.load_balance_method,
            "schedule_policy": sched.schedule_policy,
            "decode_log_interval": sched.decode_log_interval,
            "chunked_prefill_size": sched.chunked_prefill_size,
            "max_prefill_tokens": sched.max_prefill_tokens,
            "disable_overlap_schedule": sched.disable_overlap_schedule,
        }

    @classmethod
    def get_platform_config(cls) -> PlatformConfig:
        if cls._platform is None:
            raw = (cls._raw or {}).get("platform", {})
            accel_raw = raw.get("accelerator", {})
            name = accel_raw.get("name", "a100_sxm") if isinstance(accel_raw, dict) else str(accel_raw)
            cls._platform = PlatformConfig(
                accelerator=AcceleratorInfo(
                    name=name,
                    hbm_capacity_gb=float(
                        accel_raw.get("hbm_capacity_gb", 80) if isinstance(accel_raw, dict) else 80
                    ),
                ),
                disk_read_bandwidth_gb=float(raw.get("disk_read_bandwidth_gb", 8)),
                disk_write_bandwidth_gb=float(raw.get("disk_write_bandwidth_gb", 8)),
                memory_read_bandwidth_gb=float(raw.get("memory_read_bandwidth_gb", 64)),
                memory_write_bandwidth_gb=float(raw.get("memory_write_bandwidth_gb", 64)),
            )
        return cls._platform

    @classmethod
    def get_hicache_config(cls) -> HiCacheSimConfig:
        if cls._hicache is None:
            raw = (cls._raw or {}).get("cache", {})
            cls._hicache = load_hicache_config(raw if isinstance(raw, dict) else {})
        return cls._hicache

    @classmethod
    def get_cache_config(cls, model_path: str | None = None) -> CacheConfig:
        if cls._cache is None:
            path = model_path or str((cls._raw or {}).get("model_path", "mock-model"))
            platform_raw = (cls._raw or {}).get("platform", {})
            hbm_gb = hbm_capacity_gb_from_platform(platform_raw)
            cache_raw = (cls._raw or {}).get("cache", {})
            cls._cache = resolve_cache_config(
                model_path=path,
                hbm_capacity_gb=hbm_gb,
                sched=cls.get_scheduler_config(),
                cache_raw=cache_raw if isinstance(cache_raw, dict) else {},
                hicache=cls.get_hicache_config(),
            )
        return cls._cache

    @classmethod
    def _resolve_context_len(cls, model_path: str) -> int | None:
        raw = cls._raw or {}
        if raw.get("context_length") is not None:
            return int(raw["context_length"])

        if model_path in _MOCK_MODEL_PATHS:
            return None

        try:
            from aiconfigurator.sdk.utils import get_model_config_from_model_path

            parsed = get_model_config_from_model_path(model_path)
            context = parsed.get("context")
            if context is not None:
                return int(context)
        except Exception as exc:
            logger.warning(
                "Failed to load context length for model_path=%s: %s",
                model_path,
                exc,
            )
        return None

    @classmethod
    def get_model_info(cls, model_path: str | None = None) -> ModelInfo:
        path = model_path or "mock-model"
        if cls._model is None or cls._model.model_path != path:
            context_len = cls._resolve_context_len(path)
            cls._model = ModelInfo.from_model_path(path, context_len=context_len)
            if context_len is not None:
                logger.info(
                    "Model context_len=%d for model_path=%s",
                    context_len,
                    path,
                )
        return cls._model

    @classmethod
    def get_inference_time_predictor(cls, model_path: str | None = None):
        if cls._predictor is None:
            predictor_raw = dict((cls._raw or {}).get("predictor", {}))
            name = str(predictor_raw.pop("name", "aiconfigurator"))
            model = cls.get_model_info(model_path)
            hw = cls.get_platform_config().accelerator
            sched = cls.get_scheduler_config()
            if model.model_path in _MOCK_MODEL_PATHS:
                from sglsim.srt.time_predictor.constant import ConstantTimePredictor

                cls._predictor = ConstantTimePredictor(
                    prefill_s=float(predictor_raw.get("prefill_s", 0.01)),
                    decode_s=float(predictor_raw.get("decode_s", 0.001)),
                    model=model,
                    hw=hw,
                    config=sched,
                    raw=predictor_raw,
                )
            elif name == "constant":
                from sglsim.srt.time_predictor.constant import ConstantTimePredictor

                cls._predictor = ConstantTimePredictor(
                    prefill_s=float(predictor_raw.get("prefill_s", 0.01)),
                    decode_s=float(predictor_raw.get("decode_s", 0.001)),
                    model=model,
                    hw=hw,
                    config=sched,
                    raw=predictor_raw,
                )
            elif name == "aiconfigurator":
                from sglsim.srt.time_predictor.aiconfigurator import (
                    AIConfiguratorTimePredictor,
                )

                cls._predictor = AIConfiguratorTimePredictor(
                    model, hw, sched, predictor_raw
                )
            else:
                raise ValueError(
                    f"Unsupported predictor.name={name!r}; "
                    "use aiconfigurator or constant"
                )
        return cls._predictor

    @classmethod
    def get_communication_config(cls) -> CommunicationConfig:
        if cls._communication is None:
            raw = (cls._raw or {}).get("communication", {})
            cls._communication = CommunicationConfig.from_dict(
                raw if isinstance(raw, dict) else {}
            )
        return cls._communication

    @classmethod
    def get_kv_cache_bytes_per_token(cls, model_path: str | None = None) -> int:
        from sglsim.srt.simulation.manager.cache_config import _kv_bytes_per_token

        path = model_path or str((cls._raw or {}).get("model_path", "mock-model"))
        comm = cls.get_communication_config()
        if comm.kv_bytes_per_token is not None:
            return int(comm.kv_bytes_per_token)
        bpt = _kv_bytes_per_token(path, cls.get_scheduler_config())
        if bpt is not None:
            return int(bpt)
        from sglsim.srt.communication.config import DEFAULT_KV_BYTES_PER_TOKEN

        return DEFAULT_KV_BYTES_PER_TOKEN

    @classmethod
    def get_num_hidden_layers(cls, model_path: str | None = None) -> int:
        from sglsim.srt.simulation.manager.cache_config import _model_dims

        path = model_path or str((cls._raw or {}).get("model_path", "mock-model"))
        dims = _model_dims(path)
        sched = cls.get_scheduler_config()
        if dims is not None:
            return max(1, int(dims[1]) // max(1, sched.pp_size))
        return 32

    @classmethod
    def materialize_communication_section(cls, model_path: str | None = None) -> dict[str, Any]:
        comm = cls.get_communication_config()
        path = model_path or str((cls._raw or {}).get("model_path", "mock-model"))
        kv_info = resolve_kv_byte_info(
            model_path=path,
            sched=cls.get_scheduler_config(),
            kv_bytes_per_token_override=comm.kv_bytes_per_token,
        )
        alpha_beta: dict[str, dict[str, float]] = {}
        for path_key in ("h2d", "d2h", "disk_write"):
            cfg = comm.alpha_for(path_key)
            alpha_beta[path_key] = {
                "alpha_us": cfg.alpha_s * 1e6,
                "efficiency": cfg.efficiency,
            }
        return {
            "mode": comm.mode,
            "kv_bytes_per_token": kv_info.bytes_per_token_all_layers,
            "num_hidden_layers": kv_info.num_layers,
            "alpha_beta": alpha_beta,
        }

    @classmethod
    def get_communication_predictor(
        cls,
        model_path: str | None = None,
    ) -> Any:
        from sglsim.srt.communication.predictor import CommunicationTimePredictor

        path = model_path or str((cls._raw or {}).get("model_path", "mock-model"))
        if cls._comm_predictor is None:
            comm = cls.get_communication_config()
            cls._comm_predictor = CommunicationTimePredictor(
                comm_config=comm,
                platform=cls.get_platform_config(),
                kv_info=resolve_kv_byte_info(
                    model_path=path,
                    sched=cls.get_scheduler_config(),
                    kv_bytes_per_token_override=comm.kv_bytes_per_token,
                ),
            )
        return cls._comm_predictor
