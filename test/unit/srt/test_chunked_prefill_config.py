from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.scheduler_config import resolve_chunked_prefill_size


def test_null_chunked_prefill_size_is_inferred_from_hbm():
    size = resolve_chunked_prefill_size(
        None,
        hbm_capacity_gb=80.0,
        infer_if_missing=True,
    )
    assert size == 8192


def test_config_manager_infers_chunked_prefill_when_json_null():
    ConfigManager._scheduler = None
    ConfigManager._raw = {
        "model_path": "mock-model",
        "platform": {"accelerator": {"hbm_capacity_gb": 80}},
        "scheduler": {"chunked_prefill_size": None, "dp_size": 1},
    }
    sched = ConfigManager.get_scheduler_config()
    assert sched.chunked_prefill_size == 8192
