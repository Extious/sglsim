from sglsim.srt.observability.bench_metrics import calc_bench_metrics
from sglsim.srt.simulation.types import RequestStats


def test_calc_bench_metrics_ttft_tpot():
    stats = RequestStats(
        rid="r0",
        status="completed",
        input_length=10,
        output_length=3,
        created_time=0.0,
        queue_end=0.5,
        last_event_time=1.0,
        gen_token_latencies=[0.1, 0.02, 0.03],
        cache_hit_tokens=4,
    )
    m = calc_bench_metrics([stats], 1.0)
    assert m["mean_ttft_ms"] == 100.0
    assert m["mean_tpot_ms"] == 25.0
    assert m["mean_itl_ms"] == 25.0
    assert m["mean_e2e_latency_ms"] == 1000.0
    assert m["prefix_cache_reused_ratio"] == 0.4


def test_calc_bench_metrics_excludes_aborted_requests_with_latencies():
    completed = RequestStats(
        rid="ok",
        status="completed",
        input_length=10,
        output_length=3,
        last_event_time=1.0,
        gen_token_latencies=[0.1, 0.02, 0.03],
    )
    aborted = RequestStats(
        rid="oom",
        status="aborted",
        input_length=100,
        output_length=8,
        last_event_time=2.0,
        gen_token_latencies=[0.5],
    )

    m = calc_bench_metrics([completed, aborted], 2.0)

    assert m["completed"] == 1
    assert m["total_input"] == 10
    assert m["total_output"] == 3
