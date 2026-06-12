import json
from pathlib import Path

from sglsim.fault_tolerance.config import (
    ClientConfig,
    FailureConfig,
    ServerExperimentConfig,
    load_experiment_config,
)


def test_failure_config_accepts_multi_agent_shape():
    cfg = FailureConfig.from_dict(
        {
            "inject_after_job": "4",
            "recover_after_job": "8",
            "delay": 10,
            "dp_rank": "0",
            "pp_rank": "0",
            "tp_rank": "0",
        }
    )

    assert cfg.inject_after_job == "4"
    assert cfg.recover_after_job == "8"
    assert cfg.delay == 10
    assert cfg.trigger_kind == "job"
    assert cfg.thresholds == [4]


def test_server_config_maps_remote_backup_strategy():
    cfg = ServerExperimentConfig.from_dict(
        {
            "name": "remote_backup-fixed",
            "kv_backup": "remote_backup",
            "server": {"dp_size": 2, "tp_size": 1, "pp_size": 1},
            "hicache_storage": {"backend": "remote_backup"},
        }
    )

    assert cfg.name == "remote_backup-fixed"
    assert cfg.strategy == "remote_backup"
    assert cfg.server.dp_size == 2
    assert cfg.hicache_storage.backend == "remote_backup"


def test_client_config_loads_fixed_workload():
    cfg = ClientConfig.from_dict(
        {
            "client_mode": "fixed",
            "fixed": {
                "input_len": 10000,
                "output_len": 1000,
                "num_requests": 12,
                "app_workers": 4,
            },
        }
    )

    assert cfg.client_mode == "fixed"
    assert cfg.fixed.input_len == 10000
    assert cfg.fixed.output_len == 1000
    assert cfg.fixed.num_requests == 12


def test_load_experiment_config_reads_three_json_files(tmp_path: Path):
    (tmp_path / "server.json").write_text(
        json.dumps({"name": "baseline", "kv_backup": "none", "server": {}})
    )
    (tmp_path / "client.json").write_text(
        json.dumps({"client_mode": "fixed", "fixed": {"num_requests": 2}})
    )
    (tmp_path / "failure.json").write_text(json.dumps({"inject_after_job": "1"}))

    cfg = load_experiment_config(tmp_path)

    assert cfg.server.strategy == "baseline"
    assert cfg.client.fixed.num_requests == 2
    assert cfg.failure.inject_after_job == "1"

