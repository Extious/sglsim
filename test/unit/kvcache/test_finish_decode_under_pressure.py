"""Regression: finish_request must not crash when L1 is full during decode tree finalize."""

from unittest.mock import patch

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool


def test_finalize_decode_in_tree_skips_when_l1_alloc_fails():
    pool = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=1000,
            l2_capacity_pages=1000,
            hicache_write_policy="write_through",
            write_through_threshold=1,
        )
    )
    prefill_len = 32
    req = Req("r0", list(range(prefill_len)), SamplingParams(max_new_tokens=16))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, prefill_len)
    pool.finalize_prefill_in_tree(req)
    for _ in range(4):
        pool.extend_decode(req, 1)
        req.append_output_token(0)

    assert len(req.decode_pages) == 4
    with patch.object(
        pool._cache,
        "extend_prefill",
        side_effect=RuntimeError("insufficient L1 KV cache pages"),
    ):
        pool.finish_request(req)
    assert req.decode_pages == []


def test_data_parallel_controller_raises_when_scheduler_died():
    import multiprocessing as mp
    from unittest.mock import MagicMock

    from sglsim.srt.managers.data_parallel_controller import (
        DataParallelController,
        SchedulerWorkerDiedError,
    )
    from sglsim.srt.server_args import PortArgs, ServerArgs
    from sglsim.srt.simulation.manager.config import ConfigManager

    ConfigManager._raw = {
        "scheduler": {"dp_size": 2, "load_balance_method": "round_robin"},
    }
    ConfigManager._scheduler = None
    server_args = ServerArgs(dp_size=2)
    port_args = PortArgs.init_new(server_args)
    dead_proc = MagicMock(spec=mp.Process)
    dead_proc.exitcode = 1
    alive_proc = MagicMock(spec=mp.Process)
    alive_proc.exitcode = None
    controller = DataParallelController(
        server_args,
        port_args,
        scheduler_procs=[dead_proc, alive_proc],
    )
    try:
        controller.ensure_workers_alive()
    except SchedulerWorkerDiedError as exc:
        assert "ranks=[0]" in str(exc)
    else:
        raise AssertionError("expected SchedulerWorkerDiedError")
