"""Sweep workflow concurrency and injected failure intensity in logical time."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .run_offline_wct import DEFAULT_FAILURE_MANIFEST, DEFAULT_SERVER_CONFIG
from .workflow_client import DEFAULT_TRACE_ORACLE, SLACK_SOURCES


def _int_grid(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("grid must contain non-negative integers")
    return result


def _retry_totals(payload: dict[str, Any]) -> dict[str, int]:
    retries = [
        retry
        for workflow in payload.get("workflows", ())
        for retry in workflow.get("retry_events", ())
    ]
    return {
        "retry_count": len(retries),
        "restored_tokens": sum(int(row.get("restored_tokens", 0)) for row in retries),
        "recompute_tokens": sum(int(row.get("recompute_tokens", 0)) for row in retries),
    }


def _cell_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(payload.get("calibration_metrics") or {})
    requests = max(int(payload.get("request_count", 0)), 1)
    failures = int(payload.get("failure_count_injected", 0))
    makespan_s = max(float(metrics.get("makespan_s", 0.0)), 0.0)
    routes = dict(metrics.get("routes") or {})
    routes_sha256 = hashlib.sha256(
        json.dumps(routes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "policy": str(payload["policy"]),
        "slack_source": str(payload.get("slack_source", "algorithm1")),
        "workflow_concurrency": int(payload["workflow_concurrency"]),
        "failure_count_requested": int(payload["failure_count_requested"]),
        "failure_count_injected": failures,
        "request_count": int(payload["request_count"]),
        "failure_events_per_1000_requests": failures * 1000.0 / requests,
        "failure_events_per_logical_hour": (
            failures * 3600.0 / makespan_s if makespan_s > 0.0 else 0.0
        ),
        "logical_makespan_s": makespan_s,
        "mean_wct_s": float(payload["workflow_completion_time_s"]["mean"]),
        "p50_wct_s": float(payload["workflow_completion_time_s"]["p50"]),
        "p90_wct_s": float(payload["workflow_completion_time_s"]["p90"]),
        "p99_wct_s": float(payload["workflow_completion_time_s"]["p99"]),
        "host_runtime_s": float(payload.get("host_runtime_s", 0.0)),
        **_retry_totals(payload),
        "route_violations": int(
            payload.get("route_plan_audit", {}).get("violation_count", 0)
        ),
        "retry_route_violations": int(
            payload.get("route_plan_audit", {}).get("retry_violation_count", 0)
        ),
        "normal_routes_sha256": routes_sha256,
        "artifact": str(payload.get("identity", {}).get("artifact_path", "")),
    }


def _paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["workflow_concurrency"]), int(row["failure_count_requested"]))
        by_cell.setdefault(key, {})[str(row["policy"])] = row
    result: list[dict[str, Any]] = []
    for (concurrency, failures), policies in sorted(by_cell.items()):
        if set(policies) != {"aegiskv", "no_backup"}:
            continue
        aegis = policies["aegiskv"]
        baseline = policies["no_backup"]
        mean_base = float(baseline["mean_wct_s"])
        p99_base = float(baseline["p99_wct_s"])
        result.append(
            {
                "workflow_concurrency": concurrency,
                "failure_count_requested": failures,
                "failure_count_injected_aegiskv": aegis["failure_count_injected"],
                "failure_count_injected_no_backup": baseline["failure_count_injected"],
                "normal_routes_equal": (
                    aegis["normal_routes_sha256"]
                    == baseline["normal_routes_sha256"]
                ),
                "mean_wct_aegiskv_s": aegis["mean_wct_s"],
                "mean_wct_no_backup_s": mean_base,
                "mean_wct_improvement_pct": (
                    100.0 * (mean_base - float(aegis["mean_wct_s"])) / mean_base
                    if mean_base > 0.0
                    else 0.0
                ),
                "p99_wct_improvement_pct": (
                    100.0 * (p99_base - float(aegis["p99_wct_s"])) / p99_base
                    if p99_base > 0.0
                    else 0.0
                ),
                "aegiskv_retry_count": aegis["retry_count"],
                "no_backup_retry_count": baseline["retry_count"],
                "aegiskv_restored_tokens": aegis["restored_tokens"],
                "no_backup_restored_tokens": baseline["restored_tokens"],
                "additional_restored_tokens": (
                    int(aegis["restored_tokens"])
                    - int(baseline["restored_tokens"])
                ),
                "aegiskv_recompute_tokens": aegis["recompute_tokens"],
                "no_backup_recompute_tokens": baseline["recompute_tokens"],
                "recompute_reduction_tokens": (
                    int(baseline["recompute_tokens"])
                    - int(aegis["recompute_tokens"])
                ),
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Trace-oracle concurrency and failure sensitivity",
        "",
        "Algorithm 1 is bypassed. Slack is computed from future Weka API "
        "completion timestamps, so these are Algorithm-2/3 upper-bound results.",
        "The controller failure-hazard prior remains fixed; the manifest event "
        "prefix is the only failure-intensity input that changes.",
        "",
        "| Concurrency | Requested failures | Retries A/B | Mean WCT A/B (s) | "
        "Mean improvement | p99 improvement | Extra restored | Recompute reduction | "
        "Paired routes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["paired"]:
        lines.append(
            "| {workflow_concurrency} | {failure_count_requested} | "
            "{aegiskv_retry_count}/{no_backup_retry_count} | "
            "{mean_wct_aegiskv_s:.3f}/{mean_wct_no_backup_s:.3f} | "
            "{mean_wct_improvement_pct:.3f}% | {p99_wct_improvement_pct:.3f}% | "
            "{additional_restored_tokens} | {recompute_reduction_tokens} | "
            "{normal_routes_equal} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: failure count is not the same as critical "
            "victim exposure. A later retry may complete before the workflow tail "
            "and therefore change recovery counters without changing WCT. Failure-"
            "induced rerouting can also change absolute WCT across cells, so only "
            "the paired AegisKV-vs-no_backup delta within one cell is causal here.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.server_config.read_text(encoding="utf-8"))
    hazard = float(
        config.get("aegis_l1_l2", {}).get("failure_hazard_probability", 1.0)
    )
    specs: list[tuple[int, int, str, Path]] = []
    for concurrency in args.concurrency:
        for failure_count in args.failure_count:
            for policy in ("aegiskv", "no_backup"):
                output = args.output_dir / (
                    f"c{concurrency}-f{failure_count}-{policy}.json"
                )
                specs.append((concurrency, failure_count, policy, output))

    def run_cell(spec: tuple[int, int, str, Path]) -> dict[str, Any]:
        concurrency, failure_count, policy, output = spec
        if not output.is_file() or args.rerun:
            command = [
                sys.executable,
                "-m",
                "AegisKV.demo.run_offline_wct",
                "--server-config",
                str(args.server_config),
                "--trace",
                str(args.trace),
                "--route-plan",
                str(args.route_plan),
                "--failure-manifest",
                str(args.failure_manifest),
                "--output",
                str(output),
                "--policy",
                policy,
                "--workflow-concurrency",
                str(concurrency),
                "--failure-count",
                str(failure_count),
                "--max-workflows",
                str(args.max_workflows),
                "--max-requests-per-workflow",
                str(args.max_requests_per_workflow),
                "--slack-source",
                args.slack_source,
                "--artifact-detail",
                "compact",
            ]
            if policy == "aegiskv" and args.backup_route_plan is not None:
                command.extend(
                    ["--backup-route-plan", str(args.backup_route_plan)]
                )
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            print(completed.stdout.strip(), flush=True)
        return _cell_summary(json.loads(output.read_text(encoding="utf-8")))

    with ThreadPoolExecutor(max_workers=max(int(args.jobs), 1)) as executor:
        rows = list(executor.map(run_cell, specs))

    paired = _paired(rows)
    summary = {
        "schema_version": 1,
        "experiment": "AegisKV concurrency x injected-failure sensitivity",
        "slack_source": args.slack_source,
        "trace_oracle_upper_bound": args.slack_source == "trace_oracle",
        "controller_failure_hazard_probability_fixed": hazard,
        "failure_axis": (
            "prefix count from one frozen matched manifest; report observed "
            "events/request and events/logical-hour separately"
        ),
        "concurrency": args.concurrency,
        "failure_count": args.failure_count,
        "cells": rows,
        "paired": paired,
    }
    (args.output_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "sweep_cells.csv", rows)
    _write_csv(args.output_dir / "sweep_paired.csv", paired)
    _write_markdown(args.output_dir / "sweep_summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_ORACLE)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--backup-route-plan", type=Path, default=None)
    parser.add_argument("--failure-manifest", type=Path, default=DEFAULT_FAILURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=_int_grid, default=_int_grid("1,2,4,5"))
    parser.add_argument("--failure-count", type=_int_grid, default=_int_grid("0,1,2,3"))
    parser.add_argument("--max-workflows", type=int, default=5)
    parser.add_argument("--max-requests-per-workflow", type=int, default=128)
    parser.add_argument("--slack-source", choices=SLACK_SOURCES, default="trace_oracle")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Independent logical-time cells to execute concurrently.",
    )
    parser.add_argument("--rerun", action="store_true")
    return parser


def main() -> int:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary["paired"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
