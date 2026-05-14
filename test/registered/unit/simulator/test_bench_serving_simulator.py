import json
import tempfile
import unittest
from pathlib import Path

from sglang.bench_serving import (
    BenchmarkMetrics,
    inject_simulation_params,
    load_simulator_metrics,
)
from sglang.benchmark.datasets.common import DatasetRow


class TestBenchServingSimulator(unittest.TestCase):
    def test_inject_simulation_params(self):
        request = DatasetRow(prompt="hi", prompt_len=1, output_len=2)

        inject_simulation_params(request, created_time=1.5, total_request=3)

        sim = request.extra_request_body["sampling_params"]["custom_params"]["simulation"]
        self.assertEqual(sim["created_time"], 1.5)
        self.assertEqual(sim["total_request"], 3)

    def test_load_simulator_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = {field: 0 for field in BenchmarkMetrics.__dataclass_fields__}
            data["completed"] = 7
            with open(Path(tmp_dir) / "metrics.json", "w") as f:
                json.dump(data, f)

            metrics = load_simulator_metrics(tmp_dir)

        self.assertEqual(metrics.completed, 7)

    def test_load_simulator_metrics_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                load_simulator_metrics(tmp_dir)


if __name__ == "__main__":
    unittest.main()
