from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.synthesis_requests.diagnostics.build_fault_tolerance_artifacts import (
    build_artifacts,
)
from benchmark.synthesis_requests.diagnostics.verify_fault_tolerance_artifacts import (
    verify_bundle,
)


def _write_run(
    root: Path,
    name: str,
    *,
    recompute_tokens: int,
    backed_up_tokens: int,
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
                "critical_agent": "true",
                "recompute_tokens": recompute_tokens,
                "pre_failover_backed_up_tokens": backed_up_tokens,
                "failover_overhead_s": 0.05,
                "e2e_latency_s": 0.1,
            }
        )
    return run_dir


def test_fault_tolerance_verifier_smoke_passes_and_paper_fails(tmp_path: Path) -> None:
    manifest = []
    cases = [
        ("baseline", "strategy", "baseline", 100, 0),
        ("gpu_critical", "strategy", "gpu_critical", 20, 80),
        ("lag_1", "backup_lag", "1", 20, 80),
        ("critical_ratio_0.5", "critical_ratio", "0.5", 20, 80),
        ("failure_t_1", "failure_timing_s", "1", 20, 80),
    ]
    for idx, (name, axis, value, recompute, backed) in enumerate(cases):
        run_dir = _write_run(
            tmp_path,
            name,
            recompute_tokens=recompute,
            backed_up_tokens=backed,
        )
        manifest.append(
            {
                "name": name,
                "axis": axis,
                "value": value,
                "repeat_idx": 0,
                "seed": 1,
                "run_dir": str(run_dir),
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle_path = build_artifacts(manifest_path, tmp_path / "figures")

    smoke = verify_bundle(bundle_path, profile="smoke")
    paper = verify_bundle(bundle_path, profile="paper")

    assert smoke["passed"]
    assert not paper["passed"]
    assert any("backup_lag points" in failure for failure in paper["failures"])
    assert (tmp_path / "figures" / "fault_tolerance_verification.json").is_file()
