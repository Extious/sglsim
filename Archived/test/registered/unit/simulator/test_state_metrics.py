import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.simulator.metrics import RequestStats, calc_metrics, dump_simulator_outputs
from sglang.srt.simulator.state import (
    SimulatorState,
    get_active_state,
    set_active_state,
)


class TestSimulatorState(unittest.TestCase):
    def test_clock_and_latency_accumulators(self):
        state = SimulatorState()

        state.step_global_clock(0.5)
        state.set_current_inference_dur(0.2)
        state.inc_hicache_l2_load_dur(0.03)
        state.inc_hicache_l2_backup_dur(0.04)

        self.assertEqual(state.global_clock, 0.5)
        self.assertEqual(state.current_inference_dur, 0.2)
        self.assertEqual(state.pop_hicache_l2_load_dur(), 0.03)
        self.assertEqual(state.pop_hicache_l2_backup_dur(), 0.04)
        self.assertEqual(state.pop_hicache_l2_load_dur(), 0)
        self.assertEqual(state.pop_hicache_l2_backup_dur(), 0)

        state.set_current_inference_dur(0.7)
        self.assertEqual(state.last_inference_dur, 0.2)

    def test_reset_clears_state(self):
        state = SimulatorState()
        state.step_global_clock(1)
        state.inc_iteration()
        state.set_current_inference_dur(2)
        state.reset()

        self.assertEqual(state.global_clock, 0)
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.current_inference_dur, 0)

    def test_active_state_accessor(self):
        state = SimulatorState()
        set_active_state(state)
        self.assertIs(get_active_state(), state)
        set_active_state(None)
        self.assertIsNone(get_active_state())


class TestSimulatorMetrics(unittest.TestCase):
    def test_calc_metrics_excludes_warmup_and_reports_ms(self):
        stats = [
            RequestStats(
                rid="warmup",
                created_time=0.0,
                queue_start=0.0,
                queue_end=0.1,
                input_length=10,
                output_length=2,
                gen_token_latencies=[0.2, 0.3],
            ),
            RequestStats(
                rid="req",
                created_time=1.0,
                queue_start=1.0,
                queue_end=1.1,
                input_length=20,
                output_length=3,
                gen_token_latencies=[0.4, 0.5, 0.6],
            ),
        ]

        metrics = calc_metrics(stats, num_warmup=1)

        self.assertEqual(metrics["completed"], 1)
        self.assertEqual(metrics["total_input"], 20)
        self.assertEqual(metrics["total_output"], 3)
        self.assertEqual(metrics["mean_ttft_ms"], 400)
        self.assertEqual(metrics["mean_itl_ms"], 550)
        self.assertEqual(metrics["mean_e2e_latency_ms"], 1500)

    def test_dump_outputs_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats = [
                RequestStats(
                    rid="req",
                    created_time=0.0,
                    queue_start=0.0,
                    queue_end=0.1,
                    input_length=5,
                    output_length=2,
                    gen_token_latencies=[0.2, 0.3],
                )
            ]
            iterations = [{"requests": [{"extend_length": 5}], "forward_latency": 0.2}]

            dump_simulator_outputs(tmp_dir, stats, iterations, num_warmup=0)

            self.assertTrue((Path(tmp_dir) / "metrics.json").exists())
            self.assertTrue((Path(tmp_dir) / "request.jsonl").exists())
            self.assertTrue((Path(tmp_dir) / "iteration.jsonl").exists())
            with open(Path(tmp_dir) / "metrics.json") as f:
                data = json.load(f)
            self.assertEqual(data["completed"], 1)


if __name__ == "__main__":
    unittest.main()
