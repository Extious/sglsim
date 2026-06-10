from sglsim.srt.managers.schedule_batch import (
    ForwardMode,
    Req,
    SamplingParams,
    ScheduleBatch,
)
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.types import RequestStats
from sglsim.srt.utils.network import SimIpcRegistry


def _scheduler() -> Scheduler:
    SimIpcRegistry.install_global(None)
    ConfigManager._raw = {
        "scheduler": {
            "max_prefill_tokens": 16384,
            "chunked_prefill_size": None,
            "init_new_token_ratio": 0.0,
        },
        "cache": {
            "page_size": 1,
            "l1_capacity_pages": 100_000,
            "l2_capacity_pages": 0,
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    server_args = ServerArgs()
    return Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)


def _make_req(rid: str, prefill_len: int, max_new: int) -> Req:
    req = Req(rid, list(range(prefill_len)), SamplingParams(max_new_tokens=max_new))
    req.prefill_chunks_remaining = []
    req.kv_committed_length = prefill_len
    req.cached_length = prefill_len
    REQUEST_STATS[rid] = RequestStats(rid=rid)
    REQUEST_STATS[rid].input_length = prefill_len
    return req


def test_decode_allocates_kv_before_forward():
    """KV grows in update_running_batch (prepare path), before run_batch."""
    scheduler = _scheduler()
    req = _make_req("r0", prefill_len=128, max_new=4)
    scheduler.running_batch.reqs = [req]

    used_before, _, _, _ = scheduler.cache.kv_token_usage_details()
    assert req.kv_committed_length == 128

    batch = scheduler.update_running_batch(scheduler.running_batch)
    assert batch is not None
    assert batch.mode == ForwardMode.DECODE

    used_after_alloc, _, _, _ = scheduler.cache.kv_token_usage_details()
    assert used_after_alloc > used_before
    assert req.kv_committed_length == 129

    pred = batch.to_predict_batch()
    assert pred.reqs[0].past_kv_length == 129

    scheduler.run_batch(batch)
    scheduler.process_batch_result(batch)

    assert req.decode_length == 1
    assert req.kv_committed_length == 129


def test_prefill_predictor_uses_prefix_length_before_kv_allocation():
    scheduler = _scheduler()
    req = Req("prefill", list(range(192)), SamplingParams(max_new_tokens=4))
    req.kv_committed_length = 128
    req.cached_length = 128
    req.prefill_chunks_remaining = [64]
    REQUEST_STATS[req.rid] = RequestStats(rid=req.rid)

    captured = {}

    class Predictor:
        def predict_infer_time(self, batch):
            captured["extend"] = batch.reqs[0].extend_length
            captured["past"] = batch.reqs[0].past_kv_length
            return 0.01

    def extend_prefill(target_req, extend_tokens):
        assert target_req is req
        assert extend_tokens == 64
        target_req.kv_committed_length += extend_tokens

    scheduler.predictor = Predictor()
    scheduler.cache.extend_prefill = extend_prefill  # type: ignore[method-assign]
    batch = ScheduleBatch(0, 0, ForwardMode.PREFILL, [req])

    scheduler.run_batch(batch)

    assert req.kv_committed_length == 192
    assert captured == {"extend": 64, "past": 128}


def test_decode_token_usage_increases_over_steps():
    scheduler = _scheduler()
    req = _make_req("r0", prefill_len=64, max_new=8)
    scheduler.running_batch.reqs = [req]
    usage: list[int] = []

    for _ in range(4):
        batch = scheduler.update_running_batch(scheduler.running_batch)
        assert batch is not None
        assert batch.mode == ForwardMode.DECODE
        used, _, _, _ = scheduler.cache.kv_token_usage_details()
        usage.append(used)
        scheduler.run_batch(batch)
        scheduler.process_batch_result(batch)
        scheduler.running_batch = batch

    assert usage == sorted(usage)
    assert usage[-1] > usage[0]
