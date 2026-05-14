from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict, Optional, Union


_BYTES_MAP: dict["DataType", float] = {}
_TORCH_DTYPE_TO_DATA_TYPE: Dict[str, "DataType"] = {}
_ACCELERATORS: Dict[str, "AcceleratorInfo"] = {}
_ACCELERATOR_ALIASES: Dict[str, str] = {}


@unique
class DataType(Enum):
    INT4 = "INT4"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FP4 = "FP4"
    FP8 = "FP8"
    FP16 = "FP16"
    BF16 = "BF16"
    TF32 = "TF32"
    FP32 = "FP32"
    FP64 = "FP64"
    INT4_TENSOR = "INT4_TENSOR"
    INT8_TENSOR = "INT8_TENSOR"
    INT16_TENSOR = "INT16_TENSOR"
    INT32_TENSOR = "INT32_TENSOR"
    INT64_TENSOR = "INT64_TENSOR"
    FP4_TENSOR = "FP4_TENSOR"
    FP8_TENSOR = "FP8_TENSOR"
    FP16_TENSOR = "FP16_TENSOR"
    BF16_TENSOR = "BF16_TENSOR"
    TF32_TENSOR = "TF32_TENSOR"
    FP32_TENSOR = "FP32_TENSOR"
    FP64_TENSOR = "FP64_TENSOR"

    @property
    def bytes(self) -> float:
        return _BYTES_MAP.get(self, 1)

    @classmethod
    def tensor_suffix(cls) -> str:
        return "_TENSOR"

    @classmethod
    def from_torch_dtype(cls, dtype: str | None) -> Optional["DataType"]:
        if dtype is None:
            return None
        return _TORCH_DTYPE_TO_DATA_TYPE.get(str(dtype).lower().strip("torch."))


_BYTES_MAP.update(
    {
        DataType.INT4: 0.5,
        DataType.INT8: 1,
        DataType.INT16: 2,
        DataType.INT32: 4,
        DataType.INT64: 8,
        DataType.FP4: 0.5,
        DataType.FP8: 1,
        DataType.FP16: 2,
        DataType.BF16: 2,
        DataType.TF32: 4,
        DataType.FP32: 4,
        DataType.FP64: 8,
        DataType.INT4_TENSOR: 0.5,
        DataType.INT8_TENSOR: 1,
        DataType.INT16_TENSOR: 2,
        DataType.INT32_TENSOR: 4,
        DataType.INT64_TENSOR: 8,
        DataType.FP4_TENSOR: 0.5,
        DataType.FP8_TENSOR: 1,
        DataType.FP16_TENSOR: 2,
        DataType.BF16_TENSOR: 2,
        DataType.TF32_TENSOR: 4,
        DataType.FP32_TENSOR: 4,
        DataType.FP64_TENSOR: 8,
    }
)

_TORCH_DTYPE_TO_DATA_TYPE.update(
    {
        "fp8": DataType.FP8,
        "int8": DataType.INT8,
        "float8": DataType.FP8,
        "float16": DataType.FP16,
        "float32": DataType.FP32,
        "bfloat16": DataType.BF16,
        "auto": None,
    }
)


@dataclass
class AcceleratorInfo:
    name: str
    vendor: str | None = None
    hbm_capacity_gb: int | float | None = None
    hbm_bandwidth_gb: int | float | None = None
    intra_node_bandwidth_gb: Optional[int | float] = None
    inter_node_bandwidth_gb: int | float | None = 64
    device_alias: list[str] = field(default_factory=list)
    tflops: dict = field(default_factory=dict)
    ref: str = ""

    @classmethod
    def from_dict(cls, config: Dict, save_to_registry: bool = False):
        acc = cls(**config)
        if save_to_registry:
            _ACCELERATORS[acc.name.upper()] = acc
            for alias in acc.device_alias:
                _ACCELERATOR_ALIASES[alias] = acc.name.upper()
        return acc

    @staticmethod
    def find_by_hw_name(hw_name: str | None) -> Union[None, "AcceleratorInfo"]:
        if not hw_name:
            return None
        if hw_name in _ACCELERATOR_ALIASES:
            hw = _ACCELERATORS.get(_ACCELERATOR_ALIASES[hw_name])
            if hw is not None:
                hw = deepcopy(hw)
                hw.name = hw_name
            return hw
        return _ACCELERATORS.get(hw_name.upper())

    @staticmethod
    def list_all_hws() -> Dict[str, "AcceleratorInfo"]:
        return _ACCELERATORS

    @property
    def hbm_io_bw(self):
        return self.hbm_bandwidth_gb * 1e9 if self.hbm_bandwidth_gb else None

    @property
    def hbm_bytes(self):
        return self.hbm_capacity_gb * 1e9 if self.hbm_capacity_gb else None

    @property
    def intra_node_bw(self) -> Optional[float]:
        if self.intra_node_bandwidth_gb is None:
            return None
        return self.intra_node_bandwidth_gb * 1e9

    @property
    def inter_node_bw(self):
        if self.inter_node_bandwidth_gb is None:
            return None
        return self.inter_node_bandwidth_gb * 1e9

    def flops(self, datatype: Union[str, DataType] = DataType.FP16):
        if isinstance(datatype, DataType):
            datatype = datatype.value
        return self.tflops.get(datatype, 1) * 1e12

    def tensor_flops(self, datatype: Union[str, DataType] = DataType.FP16_TENSOR):
        if isinstance(datatype, DataType):
            datatype = datatype.value
        if not datatype.endswith(DataType.tensor_suffix()):
            datatype += DataType.tensor_suffix()
        tflops = self.tflops.get(datatype)
        return None if tflops is None else tflops * 1e12


@dataclass
class ModelInfo:
    hf_config: Optional[dict] = None
    model_path: Optional[str] = None
    attention_arch: Optional[str] = None
    context_len: Optional[int] = None
    hidden_size: Optional[int] = None
    head_dim: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    v_head_dim: Optional[int] = None
    vocab_size: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    torch_dtype: Optional[str] = None

    def is_mla(self) -> bool:
        return self.attention_arch == "MLA"


@dataclass
class SchedulerConfig:
    data_type: Optional[DataType] = None
    kv_cache_data_type: Optional[DataType] = None
    mem_fraction_static: Optional[float] = None
    max_total_tokens: Optional[int] = None
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    backend_name: str = "sglang"
    backend_version: Optional[str] = None


def _bandwidth_property(gb_attr: str):
    def getter(self):
        gb_value = getattr(self, gb_attr)
        return gb_value * 1e9 if gb_value else None

    return property(getter)


@dataclass
class PlatformConfig:
    device: Union[AcceleratorInfo, str]
    disk_capacity_gb: Optional[float] = None
    disk_read_bandwidth_gb: Optional[float] = None
    disk_write_bandwidth_gb: Optional[float] = None
    memory_capacity_gb: Optional[float] = None
    memory_read_bandwidth_gb: Optional[float] = None
    memory_write_bandwidth_gb: Optional[float] = None
    num_device_per_node: int = 8

    disk_read_bandwidth = _bandwidth_property("disk_read_bandwidth_gb")
    disk_write_bandwidth = _bandwidth_property("disk_write_bandwidth_gb")
    memory_read_bandwidth = _bandwidth_property("memory_read_bandwidth_gb")
    memory_write_bandwidth = _bandwidth_property("memory_write_bandwidth_gb")


AcceleratorInfo.from_dict(
    {
        "name": "NVIDIA H20",
        "device_alias": ["H20", "h20_sxm"],
        "tflops": {
            "FP8_TENSOR": 296,
            "INT8_TENSOR": 296,
            "FP16_TENSOR": 148,
            "BF16_TENSOR": 148,
            "FP32": 74,
        },
        "hbm_capacity_gb": 96,
        "hbm_bandwidth_gb": 4022,
        "inter_node_bandwidth_gb": 64,
        "intra_node_bandwidth_gb": 450,
        "vendor": "NVIDIA",
        "ref": "https://viperatech.com/product/nvidia-hgx-h20",
    },
    save_to_registry=True,
)
