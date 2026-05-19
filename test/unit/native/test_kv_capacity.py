from __future__ import annotations

from sglsim.srt.simulator.kv_capacity import (
    calc_kv_cache_cell_elems,
    calc_kv_cache_per_layer_elems,
)
from sglsim.srt.simulator.types import DataType, ModelInfo, SchedulerConfig


def test_calc_kv_cache_cell_elems_mha():
    model = ModelInfo(
        attention_arch="MHA",
        num_hidden_layers=36,
        num_key_value_heads=8,
        head_dim=128,
    )
    assert calc_kv_cache_cell_elems(model, tp_size=1, pp_size=1) == 8 * 128 * 36 * 2


def test_calc_kv_cache_per_layer_elems_mla():
    model = ModelInfo(
        attention_arch="MLA",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    assert calc_kv_cache_per_layer_elems(model, tp_size=1, pp_size=1) == 576


def test_get_kv_cache_bytes_uses_dtype():
    from sglsim.srt.simulator.kv_capacity import get_kv_cache_bytes

    model = ModelInfo(
        attention_arch="MHA",
        num_hidden_layers=2,
        num_key_value_heads=4,
        head_dim=8,
    )
    sched = SchedulerConfig(kv_cache_data_type=DataType.BF16)
    assert get_kv_cache_bytes(model, sched) == 4 * 8 * 2 * 2 * 2
