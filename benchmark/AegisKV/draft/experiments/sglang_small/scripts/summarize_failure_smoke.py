#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[idx])


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def summarize(result_dir: Path) -> dict[str, object]:
    request_rows = read_jsonl(result_dir / "client_results.jsonl")
    event_rows = read_jsonl(result_dir / "failover_events.jsonl")
    runtime_summary_path = result_dir / "failure_summary.json"
    runtime_summary = (
        json.loads(runtime_summary_path.read_text())
        if runtime_summary_path.exists()
        else {}
    )

    ok_rows = [row for row in request_rows if bool(row.get("ok"))]
    failover_rows = [
        row
        for row in ok_rows
        if isinstance(row.get("failover"), dict) and bool(row["failover"])
    ]
    latencies = [float(row["latency_s"]) for row in ok_rows]

    event_counts: dict[str, int] = {}
    captured_request_count = 0
    dispatched_request_count = 0
    event_pre_failover_output_tokens = 0
    event_pre_failover_backed_up_tokens = 0
    for event in event_rows:
        event_name = str(event.get("event", ""))
        event_counts[event_name] = event_counts.get(event_name, 0) + 1
        if event_name == "gpu_failure_simulated":
            captured_request_count += int(event.get("num_requests") or 0)
        if event_name == "failover_dispatched":
            dispatched_request_count += int(event.get("num_requests") or 0)
            requests = event.get("requests")
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        event_pre_failover_output_tokens += int(
                            req.get("pre_failover_output_tokens") or 0
                        )
                        event_pre_failover_backed_up_tokens += int(
                            req.get("pre_failover_backed_up_tokens") or 0
                        )

    response_pre_failover_output_tokens = 0
    response_pre_failover_backed_up_tokens = 0
    for row in failover_rows:
        failover = row["failover"]
        response_pre_failover_output_tokens += int(
            failover.get("pre_failover_output_tokens") or 0
        )
        response_pre_failover_backed_up_tokens += int(
            failover.get("pre_failover_backed_up_tokens") or 0
        )

    return {
        "count": len(request_rows),
        "success_count": len(ok_rows),
        "failure_count": len(request_rows) - len(ok_rows),
        "failover_response_count": len(failover_rows),
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "event_count": len(event_rows),
        "event_counts": event_counts,
        "captured_request_count": captured_request_count,
        "dispatched_request_count": dispatched_request_count,
        "event_pre_failover_output_tokens": event_pre_failover_output_tokens,
        "event_pre_failover_backed_up_tokens": event_pre_failover_backed_up_tokens,
        "response_pre_failover_output_tokens": response_pre_failover_output_tokens,
        "response_pre_failover_backed_up_tokens": response_pre_failover_backed_up_tokens,
        "failure_injection_status": runtime_summary.get("failure_injection_status", 0),
        "recovery_status": runtime_summary.get("recovery_status", 0),
        "runtime_summary_event_count": runtime_summary.get("event_count", 0),
        "note": "posthoc summary from raw request JSONL and failover event JSONL",
    }


def write_csv(path: Path, summary: dict[str, object]) -> None:
    flat_summary = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in summary.items()
    }
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_summary.keys()))
        writer.writeheader()
        writer.writerow(flat_summary)


def write_candidate_csv(path: Path, run_id: str, summary: dict[str, object]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "policy",
                "request_count",
                "success_count",
                "failover_response_count",
                "captured_request_count",
                "dispatched_request_count",
                "gpu_recovered_events",
                "latency_p95_s",
                "latency_p99_s",
                "pre_failover_backed_up_tokens",
                "note",
            ],
        )
        writer.writeheader()
        event_counts = summary.get("event_counts", {})
        recovered = event_counts.get("gpu_recovered", 0) if isinstance(event_counts, dict) else 0
        writer.writerow(
            {
                "run_id": run_id,
                "policy": "aegiskv_sglang_dp2_failure_smoke",
                "request_count": summary["count"],
                "success_count": summary["success_count"],
                "failover_response_count": summary["failover_response_count"],
                "captured_request_count": summary["captured_request_count"],
                "dispatched_request_count": summary["dispatched_request_count"],
                "gpu_recovered_events": recovered,
                "latency_p95_s": summary["latency_p95_s"],
                "latency_p99_s": summary["latency_p99_s"],
                "pre_failover_backed_up_tokens": summary[
                    "response_pre_failover_backed_up_tokens"
                ],
                "note": "checked small-scale SGLang failure smoke; control-path evidence only",
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    summary = summarize(args.result_dir)
    run_id = args.run_id or args.result_dir.name
    (args.result_dir / "failure_summary_checked.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.result_dir / "failure_summary_checked.csv", summary)
    write_candidate_csv(
        args.result_dir / "candidate_fig_failure_smoke_sglang_checked.csv",
        run_id,
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
