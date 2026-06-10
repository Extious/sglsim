from sglsim.srt.config import CacheConfig, SchedulerConfig, SGLSimConfig, TopologyConfig


def test_config_has_research_friendly_defaults():
    cfg = SGLSimConfig(
        topology=TopologyConfig(dp_size=2),
        scheduler=SchedulerConfig(max_running_reqs=8),
        cache=CacheConfig(page_size=16, l1_capacity_pages=128),
    )

    assert cfg.topology.dp_size == 2
    assert cfg.topology.tp_size == 1
    assert cfg.topology.pp_size == 1
    assert cfg.cache.page_size == 16


def test_scheduler_config_defaults_match_baseline_plan():
    cfg = SchedulerConfig()

    assert cfg.max_running_reqs == 16
    assert cfg.max_prefill_tokens == 16384
    assert cfg.chunked_prefill_size is None
    assert not hasattr(cfg, "max_total_tokens")


def test_cache_config_defaults_match_baseline_plan():
    cfg = CacheConfig()

    assert cfg.page_size == 16
    assert cfg.l1_capacity_pages == 1024
    assert cfg.l2_capacity_pages == 0
    assert cfg.l3_capacity_pages == 0


def test_config_instances_do_not_share_nested_defaults():
    assert SGLSimConfig().cache is not SGLSimConfig().cache
