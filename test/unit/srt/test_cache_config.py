import json

from sglsim.srt.simulation.manager.cache_config import (
    infer_l1_max_tokens,
    load_hicache_config,
    resolve_cache_config,
)
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig


def test_load_hicache_config_defaults():
    cfg = load_hicache_config({})
    assert not cfg.enable_hierarchical_cache
    assert cfg.hicache_ratio == 2.0
    assert cfg.hicache_storage_backend is None


def test_infer_l1_max_tokens_mock_model():
    sched = SchedulerSimConfig(mem_fraction_static=0.88)
    tokens = infer_l1_max_tokens(
        model_path="mock-model",
        hbm_capacity_gb=80,
        mem_fraction_static=0.88,
        sched=sched,
    )
    assert tokens == 65536


def test_qwen32_pp4_capacity_accounts_for_pipeline_partitioning():
    sched = SchedulerSimConfig(
        mem_fraction_static=0.50,
        tp_size=1,
        pp_size=4,
        dp_size=4,
    )
    tokens = infer_l1_max_tokens(
        model_path="Qwen/Qwen3-32B",
        hbm_capacity_gb=80,
        mem_fraction_static=0.50,
        sched=sched,
    )
    assert tokens >= 32768


def test_resolve_cache_config_applies_kv_capacity_scale_factor():
    sched = SchedulerSimConfig(mem_fraction_static=0.88)
    cache = resolve_cache_config(
        model_path="mock-model",
        hbm_capacity_gb=80,
        sched=sched,
        cache_raw={"page_size": 16, "kv_capacity_scale_factor": 0.5},
        hicache=load_hicache_config({}),
    )
    assert cache.l1_capacity_pages == 2048


def test_resolve_cache_config_infers_l1_l2_without_l3_by_default(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(
        json.dumps(
            {
                "model_path": "Qwen/Qwen3-8B",
                "platform": {"accelerator": {"hbm_capacity_gb": 80}},
                "cache": {
                    "page_size": 16,
                    "enable_hierarchical_cache": True,
                    "hicache_ratio": 2.0,
                },
                "scheduler": {"mem_fraction_static": 0.88, "tp_size": 1, "pp_size": 1},
            }
        ),
        encoding="utf-8",
    )
    ConfigManager.load(str(cfg))
    cache = ConfigManager.get_cache_config("Qwen/Qwen3-8B")
    assert cache.page_size == 16
    assert cache.l1_capacity_pages >= 64
    assert cache.l2_capacity_pages > 0
    assert cache.l3_capacity_pages == 0


def test_resolve_cache_config_l3_disabled_without_storage_backend():
    sched = SchedulerSimConfig(mem_fraction_static=0.88)
    cache = resolve_cache_config(
        model_path="mock-model",
        hbm_capacity_gb=80,
        sched=sched,
        cache_raw={"page_size": 16, "l1_capacity_pages": 128, "l2_capacity_pages": 64},
        hicache=load_hicache_config({"enable_hierarchical_cache": True}),
    )
    assert cache.l3_capacity_pages == 0


def test_resolve_cache_config_honors_explicit_pages():
    sched = SchedulerSimConfig(mem_fraction_static=0.88)
    cache = resolve_cache_config(
        model_path="mock-model",
        hbm_capacity_gb=80,
        sched=sched,
        cache_raw={
            "l1_capacity_pages": 128,
            "l2_capacity_pages": 64,
            "l3_capacity_pages": 32,
        },
        hicache=load_hicache_config({"enable_hierarchical_cache": True}),
    )
    assert cache.l1_capacity_pages == 128
    assert cache.l2_capacity_pages == 64
    assert cache.l3_capacity_pages == 32
