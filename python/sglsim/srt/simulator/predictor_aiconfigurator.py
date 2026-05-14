from __future__ import annotations

from typing import Optional

import numpy as np

from sglsim.srt.simulator.predictor import ScheduleBatch, ScheduleRequest
from sglsim.srt.simulator.types import AcceleratorInfo, DataType, ModelInfo, SchedulerConfig


def _get_aiconfigurator_sdk():
    from aiconfigurator.sdk import models
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.common import (
        CommQuantMode,
        DatabaseMode,
        FMHAQuantMode,
        GEMMQuantMode,
        KVCacheQuantMode,
        MoEQuantMode,
    )
    from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator.sdk.perf_database import get_database, get_systems_paths

    return {
        "models": models,
        "get_backend": get_backend,
        "CommQuantMode": CommQuantMode,
        "DatabaseMode": DatabaseMode,
        "FMHAQuantMode": FMHAQuantMode,
        "GEMMQuantMode": GEMMQuantMode,
        "KVCacheQuantMode": KVCacheQuantMode,
        "MoEQuantMode": MoEQuantMode,
        "ModelConfig": ModelConfig,
        "RuntimeConfig": RuntimeConfig,
        "InferenceSession": InferenceSession,
        "get_database": get_database,
        "get_systems_paths": get_systems_paths,
    }


def get_perf_model(sched_config: SchedulerConfig, model: ModelInfo):
    sdk = _get_aiconfigurator_sdk()
    gemm_quant_mode = {
        DataType.FP16: sdk["GEMMQuantMode"].float16,
        DataType.BF16: sdk["GEMMQuantMode"].float16,
        DataType.FP8: sdk["GEMMQuantMode"].fp8_block,
        DataType.INT8: sdk["GEMMQuantMode"].int8_wo,
        DataType.FP4: sdk["GEMMQuantMode"].nvfp4,
        DataType.INT4: sdk["GEMMQuantMode"].int4_wo,
        DataType.FP16_TENSOR: sdk["GEMMQuantMode"].float16,
        DataType.BF16_TENSOR: sdk["GEMMQuantMode"].float16,
        DataType.FP8_TENSOR: sdk["GEMMQuantMode"].fp8,
        DataType.INT8_TENSOR: sdk["GEMMQuantMode"].int8_wo,
        DataType.FP4_TENSOR: sdk["GEMMQuantMode"].nvfp4,
        DataType.INT4_TENSOR: sdk["GEMMQuantMode"].int4_wo,
    }
    kvcache_quant_mode = {
        DataType.FP16: sdk["KVCacheQuantMode"].float16,
        DataType.BF16: sdk["KVCacheQuantMode"].float16,
        DataType.FP8: sdk["KVCacheQuantMode"].fp8,
        DataType.INT8: sdk["KVCacheQuantMode"].int8,
    }
    fmha_quant_mode = {
        DataType.FP16: sdk["FMHAQuantMode"].float16,
        DataType.BF16: sdk["FMHAQuantMode"].float16,
        DataType.FP8: sdk["FMHAQuantMode"].fp8,
    }
    moe_quant_mode = {
        DataType.FP16: sdk["MoEQuantMode"].float16,
        DataType.BF16: sdk["MoEQuantMode"].float16,
        DataType.FP8: sdk["MoEQuantMode"].fp8,
        DataType.INT8: sdk["MoEQuantMode"].fp8,
        DataType.FP4: sdk["MoEQuantMode"].nvfp4,
        DataType.INT4: sdk["MoEQuantMode"].int4_wo,
    }
    comm_quant_mode = {
        DataType.FP16: sdk["CommQuantMode"].half,
        DataType.BF16: sdk["CommQuantMode"].half,
        DataType.FP8: sdk["CommQuantMode"].fp8,
        DataType.INT8: sdk["CommQuantMode"].int8,
    }

    model_config = sdk["ModelConfig"](
        pp_size=sched_config.pp_size,
        tp_size=sched_config.tp_size,
        moe_tp_size=sched_config.tp_size // sched_config.ep_size,
        moe_ep_size=sched_config.ep_size,
        attention_dp_size=sched_config.tp_size // sched_config.dp_size,
        gemm_quant_mode=gemm_quant_mode.get(sched_config.data_type, sdk["GEMMQuantMode"].float16),
        moe_quant_mode=moe_quant_mode.get(sched_config.data_type, sdk["MoEQuantMode"].float16),
        kvcache_quant_mode=kvcache_quant_mode.get(
            sched_config.kv_cache_data_type,
            sdk["KVCacheQuantMode"].float16,
        ),
        fmha_quant_mode=fmha_quant_mode.get(
            sched_config.kv_cache_data_type,
            sdk["FMHAQuantMode"].float16,
        ),
        comm_quant_mode=comm_quant_mode.get(sched_config.data_type, sdk["CommQuantMode"].half),
        workload_distribution="power_law_1.2",
    )

    return sdk["models"].get_model(
        model_path=model.model_path,
        model_config=model_config,
        backend_name=sched_config.backend_name,
    )


class AIConfiguratorTimePredictor:
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: Optional[str] = None,
        database_mode="SILICON",
        prefill_scale_factor: float = 1,
        decode_scale_factor: float = 1,
    ):
        self.model = model
        self.hw = hw
        self.config = config
        self.prefill_scale_factor = prefill_scale_factor
        self.decode_scale_factor = decode_scale_factor
        sdk = _get_aiconfigurator_sdk()
        if isinstance(database_mode, str):
            database_mode = self._get_database_mode(database_mode, sdk["DatabaseMode"])

        database = sdk["get_database"](
            system=hw.name,
            backend=config.backend_name,
            version=config.backend_version,
            systems_paths=[database_path] if database_path is not None else sdk["get_systems_paths"](),
        )
        if database is None:
            raise ValueError("Failed to initialize the AIConfigurator database.")
        database.set_default_database_mode(database_mode)

        db_nearest_1d_point_helper = database._nearest_1d_point_helper

        def wrapped_nearest_1d_point_helper(
            x: int,
            values: list[int],
            inner_only: bool = False,
        ):
            return db_nearest_1d_point_helper(x, values, inner_only)

        database._nearest_1d_point_helper = wrapped_nearest_1d_point_helper
        self._runtime_config = sdk["RuntimeConfig"]
        self._session = sdk["InferenceSession"](
            model=get_perf_model(config, model),
            backend=sdk["get_backend"](self.config.backend_name),
            database=database,
        )

    @staticmethod
    def _get_database_mode(mode: str, database_mode):
        return {
            "SILICON": database_mode.SILICON,
            "HYBRID": database_mode.HYBRID,
            "EMPIRICAL": database_mode.EMPIRICAL,
            "SOL": database_mode.SOL,
            "SOL_FULL": database_mode.SOL_FULL,
        }.get(mode.upper(), database_mode.SILICON)

    @staticmethod
    def ctx_attn_flops_ratio_with_avg(reqs: list[ScheduleRequest]) -> float:
        if len(reqs) == 1:
            return 1.0
        mean_past = np.mean([req.past_kv_length for req in reqs])
        mean_input = np.mean([req.extend_length for req in reqs])
        avg_flops = (mean_past + mean_past + mean_input) * mean_input / 2 * len(reqs)
        actual_flops = 0
        for req in reqs:
            actual_flops += (
                (req.past_kv_length + req.past_kv_length + req.extend_length)
                * req.extend_length
                / 2
            )
        return actual_flops / avg_flops

    def predict_requests(self, requests: list[dict]) -> float:
        batch = ScheduleBatch(
            [
                ScheduleRequest(
                    extend_length=item["extend_length"],
                    past_kv_length=item["past_kv_length"],
                )
                for item in requests
            ]
        )
        return self.predict_infer_time(batch)

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_decode():
            isl = int(np.mean([req.past_kv_length for req in batch.reqs]))
            runtime_config = self._runtime_config(batch_size=batch.batch_size, isl=isl, osl=2)
            summary = self._session.run_static(runtime_config, mode="static_gen")
            latency_dict = summary.get_generation_latency_dict()
        else:
            mean_past = np.mean([req.past_kv_length for req in batch.reqs])
            mean_input = np.mean([req.extend_length for req in batch.reqs])
            isl = int(mean_past + mean_input)
            prefix = int(mean_past)
            kwargs = {"batch_size": batch.batch_size, "isl": isl, "prefix": prefix, "osl": 1}
            seq_imbalance_correction_scale = self.ctx_attn_flops_ratio_with_avg(batch.reqs)
            if seq_imbalance_correction_scale >= 0.4:
                kwargs["seq_imbalance_correction_scale"] = seq_imbalance_correction_scale
            runtime_config = self._runtime_config(**kwargs)
            summary = self._session.run_static(runtime_config, mode="static_ctx")
            latency_dict = summary.get_context_latency_dict()

        infer_time = sum(latency_dict.values())
        if summary.check_oom():
            infer_time = -infer_time
        if batch.is_decode():
            infer_time *= self.decode_scale_factor
        else:
            infer_time *= self.prefill_scale_factor
        return infer_time / 1e3

