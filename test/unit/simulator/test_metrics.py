import json
from pathlib import Path

from sglsim.srt.simulator.metrics import RequestStats, dump_simulator_outputs
from sglsim.srt.simulator.state import SimulatorState


def test_state_tracks_clock_and_hicache_durations():
    state = SimulatorState()
    state.step_global_clock(0.5)
    state.set_current_inference_dur(0.2)
    state.inc_hicache_l2_load_dur(0.1)
    state.inc_hicache_l2_backup_dur(0.3)

    assert state.global_clock == 0.5
    assert state.current_inference_dur == 0.2
    assert state.pop_hicache_l2_load_dur() == 0.1
    assert state.pop_hicache_l2_backup_dur() == 0.3


def test_metrics_dump_skips_warmup(tmp_path: Path):
    reqs = [
        RequestStats(rid="warmup", input_length=1, output_length=1, created_time=0, last_event_time=0.1, queue_start=0, queue_end=0, gen_token_latencies=[0.1]),
        RequestStats(rid="real", input_length=2, output_length=2, created_time=1, last_event_time=1.4, queue_start=1, queue_end=1.1, gen_token_latencies=[0.2, 0.2]),
    ]

    result = dump_simulator_outputs(str(tmp_path), reqs, [], num_warmup=1)

    assert result["completed"] == 1
    assert result["total_input"] == 2
    assert (tmp_path / "metrics.json").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["completed"] == 1
