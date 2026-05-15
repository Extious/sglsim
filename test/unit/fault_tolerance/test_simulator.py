import json
from pathlib import Path

from sglsim.fault_tolerance.config import (
    ClientConfig,
    ExperimentConfig,
    FailureConfig,
    ServerExperimentConfig,
)
from sglsim.fault_tolerance.simulator import FaultToleranceSimulator, compare_strategies


def make_config(strategy: str, *, num_requests: int = 8) -> ExperimentConfig:
    kv_backup = {
        "baseline": "none",
        "remote_backup": "remote_backup",
        "host_backup": "host_backup",
    }[strategy]
    return ExperimentConfig(
        server=ServerExperimentConfig.from_dict(
            {
                "name": strategy,
                "kv_backup": kv_backup,
                "server": {"dp_size": 2},
                "hicache_storage": {"backend": kv_backup if kv_backup != "none" else ""},
            }
        ),
        client=ClientConfig.from_dict(
            {
                "client_mode": "fixed",
                "fixed": {
                    "input_len": 100,
                    "output_len": 4,
                    "num_requests": num_requests,
                    "app_workers": 2,
                },
            }
        ),
        failure=FailureConfig.from_dict(
            {"inject_after_job": "2", "recover_after_job": "5", "delay": 1}
        ),
    )


def test_strategy_impacts_failover_recompute_tokens():
    baseline = FaultToleranceSimulator(make_config("baseline")).run()
    remote = FaultToleranceSimulator(make_config("remote_backup")).run()
    host = FaultToleranceSimulator(make_config("host_backup")).run()

    assert baseline.metrics["total_recomputed_tokens"] > host.metrics["total_recomputed_tokens"]
    assert host.metrics["total_recomputed_tokens"] > remote.metrics["total_recomputed_tokens"]
    assert remote.metrics["total_remote_reuse_tokens"] > 0
    assert host.metrics["total_host_reuse_tokens"] > 0
    event_names = [event["event"] for event in baseline.events]
    assert "fault_scheduled" in event_names
    assert event_names.index("fault_scheduled") < event_names.index("fault_injected")
    assert any(event["event"] == "gpu_recovered" for event in remote.events)


def test_simulator_writes_multi_agent_compatible_outputs(tmp_path: Path):
    result = FaultToleranceSimulator(make_config("remote_backup")).run(output_dir=tmp_path)

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "internal_failover_metrics.json").exists()
    assert (tmp_path / "job_summary.csv").exists()
    assert (tmp_path / "task_summary.csv").exists()
    assert (tmp_path / "cache_hits.csv").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    failover = json.loads((tmp_path / "internal_failover_metrics.json").read_text())
    assert metrics["strategy"] == "remote_backup"
    assert metrics["completed"] == result.metrics["completed"]
    assert failover["impacted_job_ids"] == result.impacted_job_ids


def test_compare_strategies_runs_all_requested_strategies(tmp_path: Path):
    summary = compare_strategies(
        make_config("baseline", num_requests=6),
        strategies=["baseline", "remote_backup", "host_backup"],
        output_dir=tmp_path,
    )

    assert set(summary["strategies"]) == {"baseline", "remote_backup", "host_backup"}
    assert (tmp_path / "comparison.json").exists()
    assert summary["strategies"]["remote_backup"]["total_recomputed_tokens"] < summary["strategies"]["baseline"]["total_recomputed_tokens"]
