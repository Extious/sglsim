#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[6]
EXP_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "experiments" / "sglsim_large"
FIGURE_CSV = (
    ROOT
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_normal_path_overhead.csv"
)
DEFAULT_RESULTS_ROOT = EXP_DIR / "results" / "normal_path_overhead"
DEFAULT_SCENARIOS = ("normal_overhead_baseline", "normal_overhead_aegis")
BASELINE_SCENARIO = "normal_overhead_baseline"
AEGIS_SCENARIO = "normal_overhead_aegis"
RETAINED_COMPARATORS = (
    "uniform Host",
    "uniform GPU",
    "SGLang baseline",
    "SGLang HiCache WT",
)
FIG11_FIELDS = (
    "policy",
    "p99_overhead_pct",
    "write_amplification",
    "evidence_tier",
    "source",
    "run_id",
    "git_sha",
    "remote_host",
    "slurm_job_id",
    "model_path",
    "workload_id",
    "dp_size",
    "tp_size",
    "failure_rank",
    "repeat",
    "note",
    "p99_overhead_pct_std",
    "p99_overhead_pct_ci95",
    "write_amplification_std",
    "write_amplification_ci95",
    "repeat_count",
    "repeat_run_ids",
)

for path in (ROOT, ROOT / "python", Path(__file__).resolve().parent):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from run_agentic_workflow_suite import run_agentic_suite  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(FIG11_FIELDS)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile over an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(var, 0.0))


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * _sample_std(values) / math.sqrt(len(values))


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _collect_run_dirs(summary: dict[str, Any]) -> dict[str, list[Path]]:
    by_scenario: dict[str, list[Path]] = {}
    for manifest in summary.get("manifests", []):
        if not isinstance(manifest, dict):
            continue
        scenario = str(manifest.get("scenario") or "").strip()
        if not scenario:
            continue
        for row in manifest.get("plan", []):
            if not isinstance(row, dict):
                continue
            run_dir = str(row.get("run_dir") or "").strip()
            if run_dir:
                by_scenario.setdefault(scenario, []).append(Path(run_dir))
    return by_scenario


def _request_details_path(run_dir: Path) -> Path:
    path = run_dir / "request_details.csv"
    if path.exists():
        return path
    matches = sorted(run_dir.glob("**/request_details.csv"))
    if not matches:
        raise FileNotFoundError(f"missing request_details.csv under {run_dir}")
    return matches[0]


def _is_completed(row: dict[str, str]) -> bool:
    status = str(row.get("status") or "").strip().lower()
    if not status:
        return True
    return status not in {"failed", "error", "cancelled", "timeout"}


def _run_metrics(run_dir: Path) -> dict[str, object]:
    rows = [row for row in _read_csv(_request_details_path(run_dir)) if _is_completed(row)]
    latencies = [
        _to_float(
            row.get("client_wall_e2e_latency_s")
            or row.get("e2e_latency_s")
            or row.get("total_latency_s")
        )
        for row in rows
    ]
    latencies = [value for value in latencies if value > 0.0]
    if not latencies:
        raise ValueError(f"no usable completed request latencies in {run_dir}")

    total_tokens = 0.0
    protected_tokens = 0.0
    critical_rows = 0
    for row in rows:
        input_tokens = _to_float(row.get("input_length"))
        generated_tokens = _to_float(
            row.get("generated_tokens") or row.get("output_length")
        )
        tokens = max(input_tokens + generated_tokens, 0.0)
        total_tokens += tokens
        host_tokens = max(_to_float(row.get("aegis_host_visible_tokens")), 0.0)
        gpu_tokens = max(_to_float(row.get("aegis_gpu_replica_tokens")), 0.0)
        protected_tokens += host_tokens + gpu_tokens
        if str(row.get("critical_agent") or "").strip().lower() in {"1", "true", "yes"}:
            critical_rows += 1

    write_amp = 1.0
    if total_tokens > 0.0:
        write_amp += protected_tokens / total_tokens

    return {
        "run_dir": str(run_dir),
        "request_count": len(rows),
        "critical_rows": critical_rows,
        "p99_ms": _percentile(latencies, 99.0) * 1000.0,
        "write_amplification": write_amp,
        "protected_tokens": protected_tokens,
        "total_tokens": total_tokens,
    }


def _scenario_metrics(run_dirs: list[Path]) -> list[dict[str, object]]:
    if not run_dirs:
        return []
    return [_run_metrics(run_dir) for run_dir in sorted(run_dirs)]


def _overhead_values(
    baseline: list[dict[str, object]],
    aegis: list[dict[str, object]],
) -> list[float]:
    paired = min(len(baseline), len(aegis))
    values: list[float] = []
    for index in range(paired):
        base = _to_float(baseline[index].get("p99_ms"))
        protected = _to_float(aegis[index].get("p99_ms"))
        if base <= 0.0:
            continue
        values.append((protected / base - 1.0) * 100.0)
    return values


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _online_rows(
    *,
    suite_id: str,
    baseline: list[dict[str, object]],
    aegis: list[dict[str, object]],
    evidence_tier: str,
) -> list[dict[str, object]]:
    if not baseline:
        raise ValueError("baseline scenario produced no runs")
    if not aegis:
        raise ValueError("AegisKV scenario produced no runs")

    overheads = _overhead_values(baseline, aegis)
    if not overheads:
        raise ValueError("no paired overhead values could be computed")

    baseline_amps = [_to_float(row.get("write_amplification")) for row in baseline]
    aegis_amps = [_to_float(row.get("write_amplification")) for row in aegis]
    run_ids = [
        str(row.get("run_dir") or "") for row in baseline + aegis if row.get("run_dir")
    ]
    common = {
        "source": "sglsim_agentic_online_normal_path_overhead_sweep",
        "run_id": suite_id,
        "git_sha": _git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": "aegiskv-sglsim-normal-path-overhead-agentic",
        "dp_size": "2",
        "tp_size": "1",
        "failure_rank": "",
        "repeat": f"mean-of-{min(len(baseline), len(aegis))}",
        "repeat_count": str(min(len(baseline), len(aegis))),
        "repeat_run_ids": ";".join(run_ids),
    }
    note = (
        "paired online no-failure SGLSim sweep; baseline has critical_agent_ratio=0 "
        "and AegisKV has critical_agent_ratio=0.25; preliminary until final "
        "policy/physical-KV movement sweeps are complete"
    )
    return [
        {
            "policy": "none",
            "p99_overhead_pct": "0.0",
            "write_amplification": _format_float(_mean(baseline_amps)),
            "evidence_tier": evidence_tier,
            "note": note,
            "p99_overhead_pct_std": "0.0",
            "p99_overhead_pct_ci95": "0.0",
            "write_amplification_std": _format_float(_sample_std(baseline_amps)),
            "write_amplification_ci95": _format_float(_ci95(baseline_amps)),
            **common,
        },
        {
            "policy": "AegisKV",
            "p99_overhead_pct": _format_float(_mean(overheads)),
            "write_amplification": _format_float(_mean(aegis_amps)),
            "evidence_tier": evidence_tier,
            "note": note,
            "p99_overhead_pct_std": _format_float(_sample_std(overheads)),
            "p99_overhead_pct_ci95": _format_float(_ci95(overheads)),
            "write_amplification_std": _format_float(_sample_std(aegis_amps)),
            "write_amplification_ci95": _format_float(_ci95(aegis_amps)),
            **common,
        },
    ]


def _retained_comparator_rows(existing_csv: Path) -> list[dict[str, str]]:
    if not existing_csv.exists():
        return []
    out: list[dict[str, str]] = []
    wanted = set(RETAINED_COMPARATORS)
    for row in _read_csv(existing_csv):
        if str(row.get("policy") or "") in wanted:
            out.append(row)
    return out


def derive_rows(
    *,
    summary: dict[str, Any],
    suite_id: str,
    include_existing_comparators: bool,
    final_evidence: bool,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    run_dirs = _collect_run_dirs(summary)
    baseline = _scenario_metrics(run_dirs.get(BASELINE_SCENARIO, []))
    aegis = _scenario_metrics(run_dirs.get(AEGIS_SCENARIO, []))
    evidence_tier = "final" if final_evidence else "sglsim-agentic-online-preliminary"
    online = _online_rows(
        suite_id=suite_id,
        baseline=baseline,
        aegis=aegis,
        evidence_tier=evidence_tier,
    )
    by_policy: dict[str, dict[str, object]] = {
        str(row["policy"]): row for row in online
    }
    if include_existing_comparators:
        for row in _retained_comparator_rows(FIGURE_CSV):
            by_policy[str(row["policy"])] = dict(row)

    ordered: list[dict[str, object]] = []
    for policy in (
        "none",
        "uniform Host",
        "uniform GPU",
        "AegisKV",
        "SGLang baseline",
        "SGLang HiCache WT",
    ):
        row = by_policy.get(policy)
        if row is not None:
            ordered.append(row)

    details = {
        "baseline_runs": baseline,
        "aegis_runs": aegis,
        "paired_overhead_pct": _overhead_values(baseline, aegis),
        "rows": len(ordered),
    }
    return ordered, details


def run_sweep(
    *,
    suite_id: str,
    runs: int,
    results_root: Path,
    output_csv: Path | None,
    publish: bool,
    final_evidence: bool,
    include_existing_comparators: bool,
    dry_run: bool,
    log_level: str,
) -> dict[str, Any]:
    summary = run_agentic_suite(
        scenarios=list(DEFAULT_SCENARIOS),
        results_root=results_root,
        suite_id=suite_id,
        runs=runs,
        dry_run=dry_run,
        log_level=log_level,
    )
    suite_root = Path(str(summary["suite_root"]))
    output_csv = output_csv or (suite_root / "fig_normal_path_overhead.csv")
    result: dict[str, Any] = {
        "suite_id": suite_id,
        "suite_root": str(suite_root),
        "dry_run": bool(dry_run),
        "scenarios": list(DEFAULT_SCENARIOS),
        "runs": int(runs),
        "output_csv": str(output_csv),
        "published_csv": str(FIGURE_CSV) if publish else "",
        "final_evidence": bool(final_evidence),
        "include_existing_comparators": bool(include_existing_comparators),
    }
    if dry_run:
        result["note"] = "dry run only; no normal-path overhead CSV was derived"
        (suite_root / "normal_path_overhead_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return result

    rows, details = derive_rows(
        summary=summary,
        suite_id=suite_id,
        include_existing_comparators=include_existing_comparators,
        final_evidence=final_evidence,
    )
    _write_csv(output_csv, rows)
    if publish:
        FIGURE_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_csv, FIGURE_CSV)

    result.update(details)
    (suite_root / "normal_path_overhead_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired no-failure SGLSim agentic workflows and derive "
            "Fig. 11 normal-path overhead rows."
        )
    )
    parser.add_argument("--suite-id", default="")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--final-evidence", action="store_true")
    parser.add_argument("--drop-existing-comparators", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    suite_id = args.suite_id or (
        "sglsim-normal-path-overhead-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    result = run_sweep(
        suite_id=suite_id,
        runs=args.runs,
        results_root=args.results_root,
        output_csv=args.output_csv,
        publish=bool(args.publish),
        final_evidence=bool(args.final_evidence),
        include_existing_comparators=not bool(args.drop_existing_comparators),
        dry_run=bool(args.dry_run),
        log_level=args.log_level,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
