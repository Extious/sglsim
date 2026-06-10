import os
from unittest.mock import patch

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.pipeline_accounting import PipelineStages, compute_visible_gpu


def test_overlap_hides_h2d_behind_prior_inference():
    stages = PipelineStages(h2d_s=3.0, forward_s=2.0, backup_pending_s=3.0)
    overlapped = compute_visible_gpu(stages, last_infer_s=5.0, overlap=True).visible_gpu_s
    serial = compute_visible_gpu(stages, last_infer_s=5.0, overlap=False).visible_gpu_s
    assert overlapped == 3.0
    assert serial == 8.0
    assert overlapped < serial


def test_disable_overlap_on_consecutive_prefill_batches():
    from sglsim.srt.managers.schedule_batch import ForwardMode, ScheduleBatch
    from sglsim.srt.managers.scheduler import Scheduler
    from sglsim.srt.server_args import PortArgs, ServerArgs
    from sglsim.srt.simulation.manager.config import ConfigManager
    from sglsim.srt.utils.network import SimIpcRegistry

    SimIpcRegistry.install_global(None)
    ConfigManager._raw = {
        "scheduler": {"max_prefill_tokens": 8192, "chunked_prefill_size": 8192},
        "cache": {"page_size": 16, "l1_capacity_pages": 1000, "l2_capacity_pages": 0},
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    b0 = ScheduleBatch(0, 0, ForwardMode.PREFILL, [])
    b1 = ScheduleBatch(1, 0, ForwardMode.PREFILL, [])

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP", None)
        server_args = ServerArgs()
        scheduler = Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)
        scheduler.last_batch = b0
        assert scheduler.is_disable_overlap_for_batch(b1) is False

    with patch.dict(os.environ, {"SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP": "1"}, clear=False):
        server_args = ServerArgs()
        scheduler = Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)
        scheduler.last_batch = b0
        assert scheduler.is_disable_overlap_for_batch(b1) is True
        scheduler.last_batch = ScheduleBatch(2, 0, ForwardMode.DECODE, [])
        assert scheduler.is_disable_overlap_for_batch(b1) is False


def test_engine_respects_disable_overlap_flag(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text('{"scheduler": {"disable_overlap_schedule": true}}', encoding="utf-8")
    engine = Engine(
        ServerArgs(sim_config_path=str(cfg), disable_overlap_schedule=False)
    )
    engine.shutdown()
    assert ConfigManager.get_scheduler_config().disable_overlap_schedule is True

    engine2 = Engine(ServerArgs(disable_overlap_schedule=True))
    engine2.shutdown()
    assert ConfigManager.get_scheduler_config().disable_overlap_schedule is True


def test_pp_size_forces_disable_overlap_guard():
    server_args = ServerArgs(pp_size=2, disable_overlap_schedule=False)
    assert server_args.disable_overlap_schedule is True
