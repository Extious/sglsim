import json
from pathlib import Path

from sglsim.srt.http.server import SimulatorApp


class ConstantPredictor:
    def predict_requests(self, requests):
        del requests
        return 0.1


def test_simulator_app_generate_runs_cpu_scheduler_and_writes_metrics(tmp_path: Path):
    app = SimulatorApp(
        mode="offline",
        output_dir=str(tmp_path),
        predictor=ConstantPredictor(),
    )

    result = app.generate(
        input_ids=[1, 2, 3],
        max_new_tokens=2,
        rid="r1",
        simulation_params={"created_time": 0.0, "total_request": 1},
    )

    assert result["rid"] == "r1"
    assert result["output_ids"] == [1, 1]
    assert result["metrics"]["completed"] == 1
    assert (tmp_path / "metrics.json").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["completed"] == 1

