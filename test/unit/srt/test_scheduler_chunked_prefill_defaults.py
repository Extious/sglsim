from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.scheduler_config import (
    infer_chunked_prefill_size,
    resolve_chunked_prefill_size,
)


def test_infer_chunked_prefill_size_matches_sglang_tiers():
    assert infer_chunked_prefill_size(16) == 2048
    assert infer_chunked_prefill_size(24) == 2048
    assert infer_chunked_prefill_size(40) == 4096
    assert infer_chunked_prefill_size(80) == 8192
    assert infer_chunked_prefill_size(141) == 8192
    assert infer_chunked_prefill_size(192) == 16384


def test_resolve_chunked_prefill_size_infers_when_missing():
    assert (
        resolve_chunked_prefill_size(
            None,
            hbm_capacity_gb=80,
            infer_if_missing=True,
        )
        == 8192
    )


def test_resolve_chunked_prefill_size_honors_explicit_value():
    assert (
        resolve_chunked_prefill_size(
            4096,
            hbm_capacity_gb=80,
            infer_if_missing=True,
        )
        == 4096
    )
    assert (
        resolve_chunked_prefill_size(
            0,
            hbm_capacity_gb=80,
            infer_if_missing=True,
        )
        is None
    )


def test_resolve_chunked_prefill_size_null_disables_when_not_inferring():
    assert (
        resolve_chunked_prefill_size(
            None,
            hbm_capacity_gb=80,
            infer_if_missing=False,
        )
        is None
    )


def test_config_manager_infers_chunked_prefill_size_from_platform(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(
        """
        {
          "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
          },
          "scheduler": {"tp_size": 1, "dp_size": 1}
        }
        """,
        encoding="utf-8",
    )
    ConfigManager.load(str(cfg))
    sched = ConfigManager.get_scheduler_config()
    assert sched.chunked_prefill_size == 8192


def test_config_manager_explicit_chunked_prefill_size_overrides_inference(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(
        """
        {
          "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
          },
          "scheduler": {"chunked_prefill_size": 4096}
        }
        """,
        encoding="utf-8",
    )
    ConfigManager.load(str(cfg))
    sched = ConfigManager.get_scheduler_config()
    assert sched.chunked_prefill_size == 4096
