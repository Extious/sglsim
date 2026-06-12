from sglsim.fault_tolerance.config import ClientConfig, ExperimentConfig, FailureConfig, ServerExperimentConfig
from sglsim.fault_tolerance.load_model import estimate_load_overhead
from sglsim.fault_tolerance.steady_state import sweep_concurrency


def _base_config(**fixed_overrides) -> ExperimentConfig:
    fixed = {
        "input_len": 200,
        "output_len": 50,
        "num_requests": 24,
        "app_workers": 4,
        "shared_prefix_len": 128,
    }
    fixed.update(fixed_overrides)
    return ExperimentConfig(
        server=ServerExperimentConfig.from_dict(
            {"kv_backup": "none", "server": {"dp_size": 2}, "hicache_storage": {"backend": ""}}
        ),
        client=ClientConfig.from_dict({"client_mode": "fixed", "fixed": fixed}),
        failure=FailureConfig.from_dict({"inject_after_job": "99"}),
    )


def test_gpu_backup_overhead_grows_with_concurrency():
    low = estimate_load_overhead("gpu_backup", concurrency=2, dp_size=2)
    high = estimate_load_overhead("gpu_backup", concurrency=32, dp_size=2)
    assert high.latency_multiplier > low.latency_multiplier
    assert high.backup_coverage_scale < low.backup_coverage_scale


def test_concurrency_sweep_prefers_baseline_throughput_at_high_load():
    config = _base_config()
    summary = sweep_concurrency(
        config,
        strategies=["baseline", "host_backup", "gpu_backup"],
        concurrency_levels=[4, 32],
    )
    base_32 = summary["strategies"]["baseline"][1]["output_throughput"]
    gpu_32 = summary["strategies"]["gpu_backup"][1]["output_throughput"]
    base_4 = summary["strategies"]["baseline"][0]["output_throughput"]
    gpu_4 = summary["strategies"]["gpu_backup"][0]["output_throughput"]
    assert base_32 >= gpu_32
    assert gpu_32 <= base_32 * 0.55
    assert (gpu_4 / base_4) > (gpu_32 / base_32)


def test_steady_state_prefill_hit_same_across_strategies():
    summary = sweep_concurrency(
        _base_config(),
        strategies=["baseline", "host_backup", "gpu_backup"],
        concurrency_levels=[8],
    )
    hits = [
        summary["strategies"][name][0]["mean_prefill_cache_hit_rate"]
        for name in ("baseline", "host_backup", "gpu_backup")
    ]
    assert hits[0] == hits[1] == hits[2]
    assert hits[0] > 0.5
