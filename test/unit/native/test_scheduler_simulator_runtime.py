from __future__ import annotations

from types import SimpleNamespace

import torch

from sglsim.srt.simulator.scheduler import (
    SchedulerSimulatorRuntime,
    build_prediction_requests,
)
from sglsim.srt.simulator.predictor_constant import ConstantTimePredictor


class Mode:
    def __init__(self, name: str):
        self.name = name

    def is_extend(self) -> bool:
        return self.name == "extend"

    def is_decode(self) -> bool:
        return self.name == "decode"


def _make_req(rid: str, created_time: float, total_request: int = 1):
    return SimpleNamespace(
        rid=rid,
        input_ids=[1, 2, 3],
        is_chunked=0,
        sampling_params=SimpleNamespace(
            max_new_tokens=2,
            custom_params={
                "simulation": {
                    "created_time": created_time,
                    "total_request": total_request,
                }
            },
        ),
    )


def test_build_prediction_requests_for_extend_and_decode():
    req = SimpleNamespace(extend_input_len=32, prefix_indices=[1, 2], output_ids=[3])
    batch = SimpleNamespace(forward_mode=Mode("extend"), reqs=[req])
    assert build_prediction_requests(batch) == [{"extend_length": 32, "past_kv_length": 3}]

    req2 = SimpleNamespace(extend_input_len=32, prefix_indices=[1, 2], output_ids=[3, 4])
    batch2 = SimpleNamespace(forward_mode=Mode("decode"), reqs=[req2])
    assert build_prediction_requests(batch2) == [{"extend_length": 1, "past_kv_length": 4}]


def test_offline_mode_replays_requests_by_created_time():
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir="/tmp/sglsim/runtime-test",
        num_warmup=0,
        overlap_schedule=False,
        predictor=ConstantTimePredictor(prefill_ms=1.0, decode_ms=1.0),
    )
    assert runtime.filter_recv_requests([_make_req("r1", created_time=1.0, total_request=2)]) == []
    assert runtime.offline_recv_all_request is False
    assert runtime.filter_recv_requests([_make_req("r2", created_time=2.0, total_request=2)]) == []
    assert runtime.offline_recv_all_request is True

    runtime.state.set_global_clock(1.5)
    assert [req.rid for req in runtime.filter_recv_requests([])] == ["r1"]
    runtime.state.set_global_clock(2.5)
    assert [req.rid for req in runtime.filter_recv_requests([])] == ["r2"]


def test_overlap_schedule_uses_backup_after_inference():
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir="/tmp/sglsim/runtime-overlap",
        num_warmup=0,
        overlap_schedule=True,
        predictor=ConstantTimePredictor(prefill_ms=1.0, decode_ms=1.0),
    )
    runtime.state.set_current_inference_dur(0.2)
    runtime.state.last_inference_dur = 0.1
    runtime.state.inc_hicache_l2_load_dur(0.05)
    runtime.state.inc_hicache_l2_backup_dur(0.03)

    req = _make_req("r1", created_time=0.0)
    runtime._record_arrived_requests([req])
    batch = SimpleNamespace(reqs=[req])
    runtime.on_process_batch_result(batch)

    assert runtime.request_stats["r1"].gen_token_latencies[0] == 0.23


def test_non_overlap_schedule_sums_load_inference_backup():
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir="/tmp/sglsim/runtime-no-overlap",
        num_warmup=0,
        overlap_schedule=False,
        predictor=ConstantTimePredictor(prefill_ms=1.0, decode_ms=1.0),
    )
    runtime.state.set_current_inference_dur(0.2)
    runtime.state.inc_hicache_l2_load_dur(0.05)
    runtime.state.inc_hicache_l2_backup_dur(0.03)

    req = _make_req("r1", created_time=0.0)
    runtime._record_arrived_requests([req])
    batch = SimpleNamespace(reqs=[req])
    runtime.on_process_batch_result(batch)

    assert runtime.request_stats["r1"].gen_token_latencies[0] == 0.28


def test_dump_outputs_and_reset_clears_state(tmp_path):
    runtime = SchedulerSimulatorRuntime(
        mode="blocking",
        output_dir=str(tmp_path),
        num_warmup=0,
        overlap_schedule=False,
        predictor=ConstantTimePredictor(),
    )
    req = SimpleNamespace(
        rid="r1",
        input_ids=[1, 2],
        is_chunked=0,
        sampling_params=SimpleNamespace(
            max_new_tokens=1,
            custom_params={"simulation": {"server_created_time": 1.0}},
        ),
    )
    runtime._record_arrived_requests([req])
    runtime.request_stats["r1"].gen_token_latencies.append(0.1)
    runtime.dump_outputs_and_reset()
    assert runtime.request_stats == {}
    assert (tmp_path / "metrics.json").is_file()
