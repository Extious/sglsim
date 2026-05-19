from __future__ import annotations

from pathlib import Path

from sglsim.srt.simulator.scheduler import SchedulerSimulatorRuntime
from sglsim.srt.simulator.predictor_constant import ConstantTimePredictor


def test_offline_simulation_metrics_use_simulated_clock(tmp_path: Path):
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir=str(tmp_path),
        num_warmup=0,
        overlap_schedule=False,
        predictor=ConstantTimePredictor(prefill_ms=1.0, decode_ms=1.0),
    )

    from types import SimpleNamespace

    req = SimpleNamespace(
        rid="offline-r1",
        input_ids=[1, 2, 3],
        is_chunked=0,
        prefix_indices=[],
        output_ids=[],
        cached_tokens=0,
        extend_input_len=3,
        sampling_params=SimpleNamespace(
            max_new_tokens=1,
            custom_params={
                "simulation": {
                    "created_time": 0.0,
                    "total_request": 1,
                }
            },
        ),
    )

    class ForwardMode:
        def is_extend(self):
            return True

        def is_decode(self):
            return False

    class GenerationBatchResult:
        pass

    runtime.filter_recv_requests([req])
    runtime.on_new_batch_prefill(
        SimpleNamespace(reqs=[req]), waiting_queue_len=0, running_batch_len=0
    )
    batch = SimpleNamespace(reqs=[req], forward_mode=ForwardMode())
    runtime.on_run_batch(batch, GenerationBatchResult())
    runtime.on_process_batch_result(batch)
    metrics = runtime.dump_outputs_and_reset()

    assert metrics["completed"] == 1
    assert metrics["mean_ttft_ms"] > 0
    assert (tmp_path / "metrics.json").is_file()
