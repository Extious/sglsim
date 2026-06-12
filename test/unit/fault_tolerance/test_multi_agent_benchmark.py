from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "benchmark/fault_tolerance/multi_agent/run.py"
PLOT = ROOT / "benchmark/fault_tolerance/multi_agent/plot.py"
TABLES = ROOT / "benchmark/fault_tolerance/multi_agent/paper_tables.py"
VERIFY = ROOT / "benchmark/fault_tolerance/multi_agent/verify_artifacts.py"
CONFIG = ROOT / "benchmark/fault_tolerance/multi_agent/config/aegiskv_coding"


def test_multi_agent_dry_run_lists_smoke_matrix():
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN),
            "--config-dir",
            str(CONFIG),
            "--output-dir",
            "/tmp/unused-aegiskv-dry",
            "--mode",
            "model",
            "--smoke",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    specs = json.loads(proc.stdout)

    assert len(specs) == 15
    assert any(spec["strategy"] == "aegiskv_segment" for spec in specs)
    assert any(spec["experiment"] == "budget" for spec in specs)


def test_multi_agent_smoke_artifact_contract(tmp_path):
    out = tmp_path / "aegiskv-smoke"
    subprocess.run(
        [
            sys.executable,
            str(RUN),
            "--config-dir",
            str(CONFIG),
            "--output-dir",
            str(out),
            "--mode",
            "model",
            "--smoke",
        ],
        check=True,
    )
    subprocess.run([sys.executable, str(PLOT), "--results-dir", str(out)], check=True)
    subprocess.run([sys.executable, str(TABLES), "--results-dir", str(out)], check=True)
    subprocess.run(
        [sys.executable, str(VERIFY), "--results-dir", str(out), "--profile", "smoke"],
        check=True,
    )

    for name in ("runs.csv", "segments.csv", "recovery_paths.csv", "summary.json"):
        assert (out / name).is_file()
    assert (out / "fig_strategy_recovery.png").stat().st_size > 0
    verification = json.loads((out / "artifact_verification.json").read_text())
    assert verification["passed"] is True


def test_multi_agent_online_dry_run_builds_online_pipeline_commands(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN),
            "--config-dir",
            str(CONFIG),
            "--output-dir",
            str(tmp_path / "online"),
            "--mode",
            "online",
            "--smoke",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    specs = json.loads(proc.stdout)

    assert len(specs) == 15
    assert all("benchmark/online/src/main.py" in " ".join(spec["command"]) for spec in specs)
    assert any("--client-mode" in spec["command"] for spec in specs)
    assert any(spec["strategy"] == "aegiskv_segment" for spec in specs)
