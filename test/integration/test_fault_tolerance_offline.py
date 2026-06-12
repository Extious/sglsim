from __future__ import annotations

from pathlib import Path

from sglsim.fault_tolerance.config import load_experiment_config
from sglsim.fault_tolerance.simulator import compare_strategies


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = REPO_ROOT / "benchmark" / "fault_tolerance" / "config" / "scenario"


def test_offline_fault_tolerance_sweep_orders_recovery_costs(tmp_path: Path):
    config = load_experiment_config(SCENARIO)
    summary = compare_strategies(
        config,
        strategies=["baseline", "host_backup", "gpu_backup"],
        output_dir=tmp_path,
    )
    baseline = summary["strategies"]["baseline"]
    host = summary["strategies"]["host_backup"]
    gpu = summary["strategies"]["gpu_backup"]

    assert baseline["total_recomputed_tokens"] > host["total_recomputed_tokens"]
    assert host["total_recomputed_tokens"] > gpu["total_recomputed_tokens"]
    assert (tmp_path / "compare.csv").is_file()
    assert baseline["mttr_s"] >= gpu["mttr_s"]
