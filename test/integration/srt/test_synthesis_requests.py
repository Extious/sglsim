import json
import socket
from pathlib import Path

import pytest

from benchmark.synthesis_requests.run import run_experiment
from sglsim.srt.time_predictor.aiconfigurator import aiconfigurator_database_available


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_config_dir(path: Path, *, port: int) -> None:
    path.mkdir()
    (path / "client.json").write_text(
        json.dumps(
            {
                "dataset_name": "generated-shared-prefix",
                "seed": 1,
                "request_rate": "inf",
                "max_concurrency": 2,
                "gsp_num_groups": 1,
                "gsp_prompts_per_group": 2,
                "gsp_system_prompt_len": 64,
                "gsp_question_len": 16,
                "gsp_output_len": 2,
                "gsp_range_ratio": 1.0,
                "gsp_fast_prepare": True,
                "gsp_send_routing_key": False,
                "gsp_num_turns": 1,
                "gsp_ordered": True,
            }
        ),
        encoding="utf-8",
    )
    (path / "server.json").write_text(
        json.dumps(
            {
                "experiment_name": "tiny",
                "model_path": "mock-model",
                "host": "127.0.0.1",
                "port": port,
                "platform": {
                    "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
                },
                "predictor": {"database_mode": "SILICON"},
                "scheduler": {
                    "dp_size": 1,
                    "tp_size": 1,
                    "pp_size": 1,
                    "backend_name": "sglang",
                    "backend_version": "0.5.10",
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.skipif(
    not aiconfigurator_database_available(),
    reason="AIConfigurator perf database unavailable (git lfs pull?)",
)
def test_synthesis_run_experiment(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_dir(config_dir, port=_free_port())
    results_root = tmp_path / "results"
    run_dir = run_experiment(
        config_dir,
        results_root,
        log_level="info",
    )
    assert (run_dir / "simulator_config.json").is_file()
    assert (run_dir / "request_details.csv").is_file()
    cfg = json.loads((run_dir / "simulator_config.json").read_text(encoding="utf-8"))
    assert cfg["workload"]["dataset_name"] == "generated-shared-prefix"
    assert cfg["workload"]["gsp_num_groups"] == 1
    csv_text = (run_dir / "request_details.csv").read_text(encoding="utf-8")
    assert csv_text.count("\n") >= 3
