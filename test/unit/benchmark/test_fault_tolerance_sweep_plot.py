from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from benchmark.synthesis_requests.diagnostics.plot_fault_tolerance_sweeps import plot


def _write_run(
    root: Path,
    name: str,
    *,
    recompute_tokens: int,
    backed_up_tokens: int,
    overhead_s: float,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "completed": 1,
                "failed": 0,
                "duration": 2.0,
                "total_output": 40,
                "mean_e2e_latency_ms": 100.0,
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "request_details.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rid",
                "is_failover_retried",
                "critical_agent",
                "recompute_tokens",
                "pre_failover_backed_up_tokens",
                "failover_overhead_s",
                "e2e_latency_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rid": name,
                "is_failover_retried": "True",
                "critical_agent": "True",
                "recompute_tokens": recompute_tokens,
                "pre_failover_backed_up_tokens": backed_up_tokens,
                "failover_overhead_s": overhead_s,
                "e2e_latency_s": 0.1,
            }
        )
    return run_dir


def test_fault_tolerance_sweep_plot_aggregates_repeats(tmp_path: Path) -> None:
    run0 = _write_run(
        tmp_path,
        "run0",
        recompute_tokens=10,
        backed_up_tokens=30,
        overhead_s=0.1,
    )
    run1 = _write_run(
        tmp_path,
        "run1",
        recompute_tokens=20,
        backed_up_tokens=40,
        overhead_s=0.2,
    )
    manifest = [
        {
            "name": "gpu_critical",
            "axis": "strategy",
            "value": "gpu_critical",
            "repeat_idx": 0,
            "seed": 100,
            "run_dir": str(run0),
        },
        {
            "name": "gpu_critical",
            "axis": "strategy",
            "value": "gpu_critical",
            "repeat_idx": 1,
            "seed": 101,
            "run_dir": str(run1),
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    png = plot(manifest_path, tmp_path / "figures")

    assert png.is_file()
    assert (tmp_path / "figures" / "fault_tolerance_sweep_system.png").is_file()
    summary = json.loads(
        (tmp_path / "figures" / "fault_tolerance_sweep_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(summary) == 1
    row = summary[0]
    assert row["n"] == 2
    assert row["seeds"] == [100, 101]
    assert row["recompute_tokens"] == 15
    assert row["backed_up_tokens"] == 35
    assert math.isclose(row["reuse_rate"], (0.75 + (40 / 60)) / 2)
    assert row["mean_overhead_ms"] == 150
    assert row["output_throughput_tps"] == 20
    assert row["critical_mean_e2e_ms"] == 100
    assert row["recompute_tokens_stderr"] > 0
