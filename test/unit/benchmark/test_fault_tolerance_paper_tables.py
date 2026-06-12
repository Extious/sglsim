from __future__ import annotations

import json
import math
from pathlib import Path

from benchmark.synthesis_requests.diagnostics.fault_tolerance_paper_tables import (
    build_tables,
    write_tables,
)


def test_fault_tolerance_paper_tables_compute_derived_metrics(tmp_path: Path) -> None:
    summary = [
        {
            "name": "baseline",
            "axis": "strategy",
            "value": "baseline",
            "recompute_tokens": 100.0,
            "critical_recompute_tokens": 80.0,
            "backed_up_tokens": 0.0,
            "reuse_rate": 0.0,
            "mean_overhead_ms": 50.0,
            "output_throughput_tps": 200.0,
            "mean_e2e_ms": 1000.0,
        },
        {
            "name": "gpu_critical",
            "axis": "strategy",
            "value": "gpu_critical",
            "recompute_tokens": 25.0,
            "critical_recompute_tokens": 8.0,
            "backed_up_tokens": 75.0,
            "reuse_rate": 0.75,
            "mean_overhead_ms": 35.0,
            "output_throughput_tps": 220.0,
            "mean_e2e_ms": 900.0,
        },
        {
            "name": "lag_1",
            "axis": "backup_lag",
            "value": "1",
            "recompute_tokens": 10.0,
            "backed_up_tokens": 90.0,
            "reuse_rate": 0.9,
        },
        {
            "name": "critical_ratio_0.5",
            "axis": "critical_ratio",
            "value": "0.5",
            "critical_backed_up_tokens": 70.0,
            "critical_recompute_tokens": 5.0,
            "noncritical_recompute_tokens": 40.0,
            "critical_mean_e2e_ms": 900.0,
            "noncritical_mean_e2e_ms": 950.0,
        },
    ]
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    tables = build_tables(summary_path)

    assert math.isclose(tables["strategy"]["recompute_reduction_pct"], 75.0)
    assert math.isclose(
        tables["strategy"]["critical_recompute_reduction_pct"], 90.0
    )
    assert math.isclose(tables["strategy"]["gpu_backup_reuse_rate_pct"], 75.0)
    assert math.isclose(tables["strategy"]["output_throughput_change_pct"], 10.0)
    assert math.isclose(tables["strategy"]["mean_e2e_change_pct"], -10.0)
    assert len(tables["backup_lag"]) == 1
    assert len(tables["critical_ratio"]) == 1

    paths = write_tables(summary_path, tmp_path / "out")
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert paths["latex"].is_file()
    assert "Recompute reduction" in paths["markdown"].read_text(encoding="utf-8")
