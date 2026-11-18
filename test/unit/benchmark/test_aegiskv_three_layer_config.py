from __future__ import annotations

import json
from pathlib import Path

import pytest

from AegisKV.experiments.online_config_v1 import config as config_mod
from AegisKV.experiments.online_config_v1 import run as run_mod
from benchmark.synthesis_requests.utils import load_config_dir


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = (
    ROOT
    / "AegisKV"
    / "experiments"
    / "online_config_v1"
    / "configs"
    / "online-aegiskv-smoke"
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_example_materializes_all_three_layers_for_existing_online_runner(tmp_path):
    config = config_mod.load_experiment_config(EXAMPLE)
    effective = config_mod.materialize_runtime_config(config, tmp_path / "effective")

    client_raw = json.loads((effective / "client.json").read_text(encoding="utf-8"))
    server_raw = json.loads((effective / "server.json").read_text(encoding="utf-8"))
    client, server = load_config_dir(effective)

    assert client_raw["dataset_name"] == "generated-shared-prefix"
    assert client_raw["gsp_failover_replay"] is True
    assert client_raw["gsp_failure_after_output_tokens"] == 4
    assert client_raw["gsp_failure_target_stage_id"] == 2
    assert client_raw["gsp_aegis_policy"] == "aegiskv"
    assert client_raw["gsp_backup_kv"] is True
    assert server_raw["aegiskv"]["design_semantics"] == config_mod.DESIGN_SEMANTICS
    assert client.gsp_failure_injection_count == 1
    assert server.dp_size == 2


def test_failure_and_policy_fields_are_rejected_from_request_layer(tmp_path):
    request = json.loads((EXAMPLE / "request.json").read_text(encoding="utf-8"))
    failure = json.loads((EXAMPLE / "failure.json").read_text(encoding="utf-8"))
    deployment = json.loads((EXAMPLE / "deployment.json").read_text(encoding="utf-8"))
    request["gsp_failure_after_output_tokens"] = 8
    request["gsp_aegis_policy"] = "aegiskv"

    _write_json(tmp_path / "request.json", request)
    _write_json(tmp_path / "failure.json", failure)
    _write_json(tmp_path / "deployment.json", deployment)

    with pytest.raises(ValueError, match="unknown request configuration keys"):
        config_mod.load_experiment_config(tmp_path)


def test_disabled_failure_layer_forces_all_active_switches_off(tmp_path):
    request = json.loads((EXAMPLE / "request.json").read_text(encoding="utf-8"))
    failure = json.loads((EXAMPLE / "failure.json").read_text(encoding="utf-8"))
    deployment = json.loads((EXAMPLE / "deployment.json").read_text(encoding="utf-8"))
    failure.update(
        {
            "enabled": False,
            "failover_replay": True,
            "dp_sweep_enabled": True,
            "after_output_tokens": 4,
        }
    )

    _write_json(tmp_path / "request.json", request)
    _write_json(tmp_path / "failure.json", failure)
    _write_json(tmp_path / "deployment.json", deployment)

    config = config_mod.load_experiment_config(tmp_path)
    client = config.effective_client()
    assert client["gsp_failover_replay"] is False
    assert client["gsp_dp_failure_sweep"] is False
    assert client["gsp_agent_slot_failure_sweep"] is False
    assert "gsp_failure_after_output_tokens" not in client


def test_execute_snapshots_source_effective_and_runtime_config(monkeypatch, tmp_path):
    def _fake_run_experiment(config_dir, results_root, **kwargs):
        client, server = load_config_dir(config_dir)
        assert client.gsp_failover_replay is True
        assert server.dp_size == 2
        run_dir = Path(results_root) / "fake-runtime"
        run_dir.mkdir(parents=True)
        return run_dir

    monkeypatch.setattr(run_mod, "run_experiment", _fake_run_experiment)
    campaign = run_mod.execute(EXAMPLE, tmp_path)

    manifest = json.loads(
        (campaign / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["execution_mode"] == "online_wall_clock_http"
    assert manifest["physical_result"] is False
    assert manifest["design_semantics"] == config_mod.DESIGN_SEMANTICS
    assert (campaign / "source_config" / "request.json").is_file()
    assert (campaign / "source_config" / "failure.json").is_file()
    assert (campaign / "source_config" / "deployment.json").is_file()
    assert (campaign / "effective_config" / "client.json").is_file()
    assert (campaign / "effective_config" / "server.json").is_file()


def test_enabled_failure_requires_a_trigger(tmp_path):
    request = json.loads((EXAMPLE / "request.json").read_text(encoding="utf-8"))
    failure = json.loads((EXAMPLE / "failure.json").read_text(encoding="utf-8"))
    deployment = json.loads((EXAMPLE / "deployment.json").read_text(encoding="utf-8"))
    failure.update(
        {
            "rate_per_window": None,
            "after_output_tokens": None,
            "injection_count": None,
            "dp_sweep_enabled": False,
            "agent_slot_sweep_enabled": False,
        }
    )

    _write_json(tmp_path / "request.json", request)
    _write_json(tmp_path / "failure.json", failure)
    _write_json(tmp_path / "deployment.json", deployment)

    with pytest.raises(ValueError, match="has no trigger"):
        config_mod.load_experiment_config(tmp_path)
