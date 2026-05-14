from __future__ import annotations

from sglsim.srt.scheduling.request import Request, SamplingParams
from sglsim.srt.scheduling.scheduler import CPUScheduler
from sglsim.srt.simulator.scheduler import SchedulerSimulatorRuntime


class ConstantPredictor:
    def __init__(self, latency: float = 0.25):
        self.latency = latency
        self.requests = []

    def predict_requests(self, requests):
        self.requests.append(requests)
        return self.latency


def make_req(rid, created_time, total_request=2, input_len=4, output_len=3):
    return Request(
        rid=rid,
        input_ids=list(range(input_len)),
        sampling_params=SamplingParams(
            max_new_tokens=output_len,
            custom_params={
                "simulation": {
                    "created_time": created_time,
                    "total_request": total_request,
                }
            },
        ),
    )


def test_offline_buffers_until_all_requests_arrive():
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir="/tmp/sglsim/test-scheduler",
        num_warmup=0,
        overlap_schedule=False,
        predictor=ConstantPredictor(),
    )
    r1 = make_req("r1", 0.0, input_len=3)
    r2 = make_req("r2", 2.0)

    assert runtime.filter_recv_requests([r1]) == []
    first_arrival = runtime.filter_recv_requests([r2])

    assert [req.rid for req in first_arrival] == ["r1"]
    runtime.state.set_global_clock(2.0)
    second_arrival = runtime.filter_recv_requests([])
    assert [req.rid for req in second_arrival] == ["r2"]
    assert runtime.request_stats["r1"].input_length == 3


def test_blocking_returns_requests_immediately_and_records_stats():
    runtime = SchedulerSimulatorRuntime(
        mode="blocking",
        output_dir="/tmp/sglsim/test-scheduler",
        num_warmup=0,
        overlap_schedule=False,
        predictor=ConstantPredictor(0),
    )
    req = make_req("r1", 0.0, total_request=1, input_len=3)

    arrived = runtime.filter_recv_requests([req])

    assert arrived == [req]
    assert runtime.request_stats["r1"].input_length == 3
    assert runtime.request_stats["r1"].output_length == 3


def test_cpu_scheduler_runs_prefill_and_decode_steps():
    predictor = ConstantPredictor(0.1)
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir="/tmp/sglsim/test-scheduler",
        num_warmup=0,
        overlap_schedule=False,
        predictor=predictor,
    )
    scheduler = CPUScheduler(runtime)
    req = make_req("r1", 0.0, total_request=1, input_len=3, output_len=2)

    scheduler.add_requests([req])
    scheduler.run(max_steps=4)

    assert req.output_ids == [1, 1]
    assert runtime.request_stats["r1"].input_length == 3
    assert runtime.request_stats["r1"].gen_token_latencies == [0.1, 0.1]
    assert predictor.requests == [
        [{"extend_length": 3, "past_kv_length": 0}],
        [{"extend_length": 1, "past_kv_length": 4}],
    ]


def test_runtime_requires_predictor_for_run_requests():
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir="/tmp/sglsim/test-scheduler",
        num_warmup=0,
        overlap_schedule=False,
        predictor=None,
    )
    scheduler = CPUScheduler(runtime)
    req = make_req("r1", 0.0, total_request=1)
    scheduler.add_requests([req])

    try:
        scheduler.run(max_steps=1)
    except RuntimeError as exc:
        assert "predictor" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
