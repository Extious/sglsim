from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sglang.srt.simulator.config import SimulatorSettings
from sglang.srt.simulator.types import (
    AcceleratorInfo,
    DataType,
    ModelInfo,
    PlatformConfig,
    SchedulerConfig,
)


@dataclass
class ScheduleRequest:
    extend_length: int = 0
    past_kv_length: int = 0


@dataclass
class ScheduleBatch:
    reqs: list[ScheduleRequest] = field(default_factory=list)

    def request_info(self) -> list[list[int]]:
        return [[req.extend_length, req.past_kv_length] for req in self.reqs]

    @property
    def num_context_tokens(self) -> int:
        return sum(req.extend_length for req in self.reqs)

    @property
    def total_past_kv_length(self) -> int:
        return sum(req.past_kv_length for req in self.reqs)

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def is_prefill(self) -> bool:
        return not self.is_decode()

    def is_decode(self) -> bool:
        return all(req.extend_length <= 1 for req in self.reqs)

    @property
    def num_ctx_requests(self) -> int:
        return self.batch_size if self.is_prefill() else 0

    @property
    def num_gen_requests(self) -> int:
        return self.batch_size if self.is_decode() else 0


def _dtype_from_config(value: str | None) -> DataType | None:
    if value is None:
        return None
    try:
        return DataType(value.upper())
    except ValueError:
        return DataType.from_torch_dtype(value)


def _torch_dtype_name(value: Any) -> str:
    return str(value).strip("torch.")


def resolve_scheduler_config(server_args: Any, model_config: Any | None = None) -> SchedulerConfig:
    from sglang.version import __version__

    dtype = getattr(server_args, "dtype", None)
    if dtype == "auto":
        dtype = _torch_dtype_name(getattr(model_config, "dtype", None))
    data_type = DataType.from_torch_dtype(dtype) or DataType.FP16
    kv_cache_data_type = (
        DataType.from_torch_dtype(getattr(server_args, "kv_cache_dtype", None))
        or data_type
    )
    return SchedulerConfig(
        data_type=data_type,
        kv_cache_data_type=kv_cache_data_type,
        mem_fraction_static=getattr(server_args, "mem_fraction_static", None),
        max_total_tokens=getattr(server_args, "max_total_tokens", None),
        tp_size=getattr(server_args, "tp_size", 1),
        ep_size=getattr(server_args, "ep_size", 1),
        dp_size=getattr(server_args, "dp_size", 1),
        pp_size=getattr(server_args, "pp_size", 1),
        backend_name="sglang",
        backend_version=__version__,
    )


def resolve_model_info(model_config: Any) -> ModelInfo:
    attention_arch = getattr(model_config, "attention_arch", None)
    attention_arch_name = getattr(attention_arch, "name", None)
    if attention_arch_name is None:
        attention_arch_name = str(attention_arch)
    if attention_arch_name not in ("MHA", "MLA"):
        raise ValueError(f"Unsupported simulator attention architecture: {attention_arch}")

    kwargs = {
        "hf_config": getattr(model_config, "hf_text_config", None),
        "model_path": getattr(model_config, "model_path", None),
        "attention_arch": attention_arch_name,
        "context_len": getattr(model_config, "context_len", None),
        "hidden_size": getattr(model_config, "hidden_size", None),
        "head_dim": getattr(model_config, "head_dim", None),
        "num_attention_heads": getattr(model_config, "num_attention_heads", None),
        "num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", None),
        "v_head_dim": getattr(model_config, "v_head_dim", None),
        "vocab_size": getattr(model_config, "vocab_size", None),
        "torch_dtype": _torch_dtype_name(getattr(model_config, "dtype", None)),
    }
    if attention_arch_name == "MLA":
        kwargs.update(
            {
                "qk_rope_head_dim": getattr(model_config, "qk_rope_head_dim", None),
                "qk_nope_head_dim": getattr(model_config, "qk_nope_head_dim", None),
                "kv_lora_rank": getattr(model_config, "kv_lora_rank", None),
            }
        )
    return ModelInfo(**kwargs)


class SimulatorConfigManager:
    def __init__(self, settings: SimulatorSettings):
        self.settings = settings
        self.raw_config = settings.raw_config
        self._model_info: Optional[ModelInfo] = None
        self._platform_config: Optional[PlatformConfig] = None
        self._scheduler_config: Optional[SchedulerConfig] = None

    def set_model_info(self, model: ModelInfo) -> None:
        self._model_info = model

    def get_model_info(self) -> ModelInfo | None:
        return self._model_info

    def get_accelerator_info(self) -> AcceleratorInfo:
        platform_config = self.raw_config.get("platform", {})
        acc_info = platform_config.get("accelerator", {})
        hw = AcceleratorInfo.find_by_hw_name(acc_info.get("name"))
        if hw is not None:
            return hw
        return AcceleratorInfo(
            name=acc_info.get("name", "unknown"),
            vendor=acc_info.get("vendor"),
            hbm_bandwidth_gb=acc_info.get("hbm_bandwidth_gb"),
            hbm_capacity_gb=acc_info.get("hbm_capacity_gb"),
            inter_node_bandwidth_gb=acc_info.get("inter_node_bandwidth_gb"),
            intra_node_bandwidth_gb=acc_info.get("intra_node_bandwidth_gb"),
            tflops=acc_info.get("tflops", {}),
        )

    def get_platform_config(self) -> PlatformConfig:
        if self._platform_config is None:
            hw = self.get_accelerator_info()
            config = self.raw_config.get("platform", {})
            self._platform_config = PlatformConfig(
                device=hw,
                disk_capacity_gb=config.get("disk_capacity_gb"),
                disk_read_bandwidth_gb=config.get("disk_read_bandwidth_gb"),
                disk_write_bandwidth_gb=config.get("disk_write_bandwidth_gb"),
                memory_capacity_gb=config.get("memory_capacity_gb"),
                memory_read_bandwidth_gb=config.get("memory_read_bandwidth_gb"),
                memory_write_bandwidth_gb=config.get("memory_write_bandwidth_gb"),
                num_device_per_node=config.get("num_device_per_node", 8),
            )
        return self._platform_config

    def set_scheduler_config(self, config: SchedulerConfig) -> None:
        external_config = self.raw_config.get("scheduler", {})
        for field_name in (
            "tp_size",
            "dp_size",
            "ep_size",
            "pp_size",
            "backend_name",
            "backend_version",
        ):
            field_value = external_config.get(field_name)
            if field_value is not None:
                setattr(config, field_name, field_value)
        for field_name in ("data_type", "kv_cache_data_type"):
            field_value = _dtype_from_config(external_config.get(field_name))
            if field_value is not None:
                setattr(config, field_name, field_value)
        self._scheduler_config = config

    def get_scheduler_config(self) -> SchedulerConfig | None:
        return self._scheduler_config

    def get_inference_time_predictor(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        sched_config: SchedulerConfig,
    ) -> Any:
        predictor_config = self.raw_config.get("predictor", {})
        if predictor_config.get("name") != "aiconfigurator":
            raise ValueError(f"Unknown predictor name: {predictor_config.get('name')}")
        from sglang.srt.simulator.predictor_aiconfigurator import (
            AIConfiguratorTimePredictor,
        )

        return AIConfiguratorTimePredictor(
            model,
            hw=hw,
            config=sched_config,
            database_path=predictor_config.get("database_path"),
            database_mode=predictor_config.get("database_mode", "SILICON"),
            prefill_scale_factor=predictor_config.get("prefill_scale_factor", 1),
            decode_scale_factor=predictor_config.get("decode_scale_factor", 1),
        )


def build_simulator_predictor(server_args: Any, model_config: Any) -> Any:
    try:
        settings = SimulatorSettings.from_server_args(server_args)
        manager = SimulatorConfigManager(settings)
        manager.set_model_info(resolve_model_info(model_config))
        manager.set_scheduler_config(resolve_scheduler_config(server_args, model_config))
        model = manager.get_model_info()
        sched_config = manager.get_scheduler_config()
        if model is None or sched_config is None:
            return None
        return manager.get_inference_time_predictor(
            model,
            manager.get_accelerator_info(),
            sched_config,
        )
    except Exception:
        return None
