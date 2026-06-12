from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerifyThresholds:
    min_repeats_per_point: int
    min_strategy_points: int
    min_backup_lag_points: int
    min_critical_ratio_points: int
    min_failure_timing_points: int
    min_recompute_reduction_pct: float
    min_critical_recompute_reduction_pct: float
    min_gpu_backup_reuse_rate_pct: float
    max_mean_e2e_increase_pct: float
    max_throughput_drop_pct: float


SMOKE_THRESHOLDS = VerifyThresholds(
    min_repeats_per_point=1,
    min_strategy_points=2,
    min_backup_lag_points=1,
    min_critical_ratio_points=1,
    min_failure_timing_points=1,
    min_recompute_reduction_pct=1.0,
    min_critical_recompute_reduction_pct=1.0,
    min_gpu_backup_reuse_rate_pct=1.0,
    max_mean_e2e_increase_pct=50.0,
    max_throughput_drop_pct=50.0,
)


PAPER_THRESHOLDS = VerifyThresholds(
    min_repeats_per_point=3,
    min_strategy_points=2,
    min_backup_lag_points=4,
    min_critical_ratio_points=4,
    min_failure_timing_points=4,
    min_recompute_reduction_pct=10.0,
    min_critical_recompute_reduction_pct=25.0,
    min_gpu_backup_reuse_rate_pct=10.0,
    max_mean_e2e_increase_pct=20.0,
    max_throughput_drop_pct=20.0,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _thresholds(profile: str) -> VerifyThresholds:
    if profile == "smoke":
        return SMOKE_THRESHOLDS
    if profile == "paper":
        return PAPER_THRESHOLDS
    raise ValueError(f"Unknown profile={profile!r}")


def _point_counts(manifest: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in manifest:
        key = "|".join(
            (
                str(entry.get("axis", "")),
                str(entry.get("value", "")),
                str(entry.get("name", "")),
            )
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _axis_value_counts(manifest: list[dict[str, Any]]) -> dict[str, int]:
    values: dict[str, set[str]] = {}
    for entry in manifest:
        axis = str(entry.get("axis", ""))
        values.setdefault(axis, set()).add(str(entry.get("value", "")))
    return {axis: len(axis_values) for axis, axis_values in values.items()}


def _check(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def verify_bundle(bundle_path: Path, *, profile: str = "paper") -> dict[str, Any]:
    bundle = _read_json(bundle_path)
    thresholds = _thresholds(profile)
    output_dir = Path(str(bundle.get("output_dir", bundle_path.parent)))
    manifest_path = Path(str(bundle.get("manifest", "")))
    manifest = _read_json(manifest_path)
    paper_tables = _read_json(output_dir / "fault_tolerance_paper_tables.json")
    strategy = paper_tables.get("strategy", {})

    failures: list[str] = []
    artifact_paths = bundle.get("artifacts", {})
    _check(bool(artifact_paths), failures, "bundle has no artifacts map")
    for name, raw_path in artifact_paths.items():
        path = Path(str(raw_path))
        _check(path.is_file(), failures, f"missing artifact {name}: {path}")
        if path.is_file():
            _check(path.stat().st_size > 0, failures, f"empty artifact {name}: {path}")

    axis_counts = _axis_value_counts(manifest)
    _check(
        axis_counts.get("strategy", 0) >= thresholds.min_strategy_points,
        failures,
        f"strategy points {axis_counts.get('strategy', 0)} < {thresholds.min_strategy_points}",
    )
    _check(
        axis_counts.get("backup_lag", 0) >= thresholds.min_backup_lag_points,
        failures,
        f"backup_lag points {axis_counts.get('backup_lag', 0)} < {thresholds.min_backup_lag_points}",
    )
    _check(
        axis_counts.get("critical_ratio", 0) >= thresholds.min_critical_ratio_points,
        failures,
        f"critical_ratio points {axis_counts.get('critical_ratio', 0)} < {thresholds.min_critical_ratio_points}",
    )
    _check(
        axis_counts.get("failure_timing_s", 0) >= thresholds.min_failure_timing_points,
        failures,
        f"failure_timing_s points {axis_counts.get('failure_timing_s', 0)} < {thresholds.min_failure_timing_points}",
    )

    point_counts = _point_counts(manifest)
    short_points = {
        key: count
        for key, count in point_counts.items()
        if count < thresholds.min_repeats_per_point
    }
    _check(
        not short_points,
        failures,
        f"points below repeat threshold {thresholds.min_repeats_per_point}: {short_points}",
    )

    recompute_reduction = float(strategy.get("recompute_reduction_pct", 0.0) or 0.0)
    critical_reduction = float(
        strategy.get("critical_recompute_reduction_pct", 0.0) or 0.0
    )
    reuse_rate = float(strategy.get("gpu_backup_reuse_rate_pct", 0.0) or 0.0)
    e2e_change = float(strategy.get("mean_e2e_change_pct", 0.0) or 0.0)
    throughput_change = float(strategy.get("output_throughput_change_pct", 0.0) or 0.0)
    _check(
        recompute_reduction >= thresholds.min_recompute_reduction_pct,
        failures,
        f"recompute reduction {recompute_reduction:.2f}% < {thresholds.min_recompute_reduction_pct:.2f}%",
    )
    _check(
        critical_reduction >= thresholds.min_critical_recompute_reduction_pct,
        failures,
        f"critical recompute reduction {critical_reduction:.2f}% < {thresholds.min_critical_recompute_reduction_pct:.2f}%",
    )
    _check(
        reuse_rate >= thresholds.min_gpu_backup_reuse_rate_pct,
        failures,
        f"gpu backup reuse {reuse_rate:.2f}% < {thresholds.min_gpu_backup_reuse_rate_pct:.2f}%",
    )
    _check(
        e2e_change <= thresholds.max_mean_e2e_increase_pct,
        failures,
        f"mean e2e increase {e2e_change:.2f}% > {thresholds.max_mean_e2e_increase_pct:.2f}%",
    )
    _check(
        throughput_change >= -thresholds.max_throughput_drop_pct,
        failures,
        f"output throughput drop {throughput_change:.2f}% < -{thresholds.max_throughput_drop_pct:.2f}%",
    )

    report = {
        "profile": profile,
        "passed": not failures,
        "failures": failures,
        "axis_value_counts": axis_counts,
        "point_counts": point_counts,
        "strategy_summary": strategy,
    }
    report_path = output_dir / "fault_tolerance_verification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify FT paper artifact bundle")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    args = parser.parse_args(argv)
    report = verify_bundle(args.bundle, profile=args.profile)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
