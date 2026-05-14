import json
from pathlib import Path

import pytest

from sglsim.bench_serving import (
    BenchmarkMetrics,
    DatasetRow,
    inject_simulation_params,
    load_simulator_metrics,
)


def test_inject_simulation_params():
    row = DatasetRow(prompt="hi", prompt_len=1, output_len=2)

    inject_simulation_params(row, created_time=1.5, total_request=3)

    simulation_params = row.extra_request_body["sampling_params"]["custom_params"]["simulation"]
    assert simulation_params["created_time"] == 1.5
    assert simulation_params["total_request"] == 3


def test_load_simulator_metrics(tmp_path: Path):
    data = {field: 0 for field in BenchmarkMetrics.__dataclass_fields__}
    data["completed"] = 1
    (tmp_path / "metrics.json").write_text(json.dumps(data))

    loaded_metrics = load_simulator_metrics(str(tmp_path))

    assert loaded_metrics.completed == 1


def test_load_simulator_metrics_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_simulator_metrics(str(tmp_path))

