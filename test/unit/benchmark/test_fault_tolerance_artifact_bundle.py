from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.synthesis_requests.diagnostics.build_fault_tolerance_artifacts import (
    build_artifacts,
)


def _write_run(
    root: Path,
    name: str,
    *,
    recompute_tokens: int,
    backed_up_tokens: int,
    critical_agent: bool,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "completed": 1,
                "failed": 0,
                "duration": 1.0,
                "total_output": 32,
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
                "is_failover_retried": "true",
                "critical_agent": str(critical_agent),
                "recompute_tokens": recompute_tokens,
                "pre_failover_backed_up_tokens": backed_up_tokens,
                "failover_overhead_s": 0.05,
                "e2e_latency_s": 0.1,
            }
        )
    return run_dir


def test_build_fault_tolerance_artifacts_bundle(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        recompute_tokens=100,
        backed_up_tokens=0,
        critical_agent=True,
    )
    gpu = _write_run(
        tmp_path,
        "gpu",
        recompute_tokens=20,
        backed_up_tokens=80,
        critical_agent=True,
    )
    manifest = [
        {
            "name": "baseline",
            "axis": "strategy",
            "value": "baseline",
            "repeat_idx": 0,
            "seed": 1,
            "run_dir": str(baseline),
        },
        {
            "name": "gpu_critical",
            "axis": "strategy",
            "value": "gpu_critical",
            "repeat_idx": 0,
            "seed": 1,
            "run_dir": str(gpu),
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bundle_path = build_artifacts(manifest_path, tmp_path / "figures")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["num_runs"] == 2
    assert bundle["axis_counts"] == {"strategy": 2}
    assert bundle["num_summary_rows"] == 2
    assert bundle["strategy_summary"]["recompute_reduction_pct"] == 80
    for path in bundle["artifacts"].values():
        assert Path(path).is_file()
