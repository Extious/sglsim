import logging

from sglsim.srt.managers.schedule_batch import ForwardMode, Req, SamplingParams, ScheduleBatch
from sglsim.srt.observability.scheduler_metrics_mixin import PrefillStats, SchedulerMetricsMixin


class _MetricsStub(SchedulerMetricsMixin):
    def __init__(self, interval: int) -> None:
        class _Cfg:
            decode_log_interval = interval

        self.sched_cfg = _Cfg()
        self.init_metrics()
        self.waiting_queue = []
        self.running_batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [])
        self.logger = logging.getLogger("test_decode_log_interval")
        self.cache = _CacheStub()


class _CacheStub:
    def kv_token_usage(self) -> tuple[int, int]:
        return (100, 1000)

    def kv_token_usage_details(
        self, *, logical_tokens: int | None = None
    ) -> tuple[int, int, int, int]:
        if logical_tokens is not None:
            return (logical_tokens, 1000, 800, 100)
        return (100, 1000, 800, 100)

    @staticmethod
    def logical_kv_tokens(reqs) -> int:
        return sum(req.total_length for req in reqs)


def test_decode_stats_logs_every_interval(caplog):
    stub = _MetricsStub(interval=40)
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [Req("r0", [1], SamplingParams(1))])
    with caplog.at_level(logging.INFO, logger="test_decode_log_interval"):
        for _ in range(39):
            stub.report_decode_stats(batch)
        assert "Decode batch" not in caplog.text
        stub.report_decode_stats(batch)
        assert "Decode batch [40]" in caplog.text


def test_sglang_profile_decode_log_matches_sglang_fields(caplog, monkeypatch):
    monkeypatch.setenv("SGLSIM_LOG_PROFILE", "sglang_projected")
    stub = _MetricsStub(interval=1)
    batch = ScheduleBatch(
        1,
        0,
        ForwardMode.DECODE,
        [Req("r0", list(range(100)), SamplingParams(max_new_tokens=0))],
    )
    batch.inference_dur_s = 0.5

    with caplog.at_level(logging.INFO, logger="test_decode_log_interval"):
        stub.report_decode_stats(batch)

    assert (
        "Decode batch, #running-req: 1, #token: 100, token usage: 0.10, "
        "cuda graph: True, gen throughput (token/s): 2.00"
    ) in caplog.text
    assert "#queue-req: 0" in caplog.text
    assert "Decode batch [" not in caplog.text
    assert "#logical-token" not in caplog.text
    assert "#avail-token" not in caplog.text


def test_sglang_profile_prefill_log_matches_sglang_fields(caplog, monkeypatch):
    monkeypatch.setenv("SGLSIM_LOG_PROFILE", "sglang_projected")
    stub = _MetricsStub(interval=1)
    stub.running_batch = ScheduleBatch(
        0,
        0,
        ForwardMode.DECODE,
        [
            Req("r0", list(range(50)), SamplingParams(max_new_tokens=0)),
            Req("r1", list(range(50)), SamplingParams(max_new_tokens=0)),
        ],
    )
    stats = PrefillStats(
        log_input_tokens=8192,
        log_hit_tokens=4096,
        new_token_ratio=1.0,
        num_running_reqs=2,
        num_new_seqs=1,
    )
    batch = ScheduleBatch(1, 0, ForwardMode.PREFILL, [])
    batch.inference_dur_s = 2.0

    with caplog.at_level(logging.INFO, logger="test_decode_log_interval"):
        stub.report_prefill_stats(batch, stats)

    assert "Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 4096, token usage: 0.10, #running-req: 2, #queue-req: 0, cuda graph: True, input throughput (token/s): 4096.00" in caplog.text
    assert "#logical-token" not in caplog.text
    assert "#avail-token" not in caplog.text


def test_decode_throughput_uses_accumulated_sim_duration(caplog):
    stub = _MetricsStub(interval=2)
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [Req("r0", [1], SamplingParams(1))])
    batch.inference_dur_s = 0.25

    with caplog.at_level(logging.INFO, logger="test_decode_log_interval"):
        stub.report_decode_stats(batch)
        stub.report_decode_stats(batch)

    assert "gen throughput (token/s): 4.00" in caplog.text
