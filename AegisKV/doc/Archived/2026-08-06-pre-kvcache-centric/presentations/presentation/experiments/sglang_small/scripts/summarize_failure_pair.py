#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(result_dir: Path) -> dict[str, object]:
    path = result_dir / "failure_summary_checked.json"
    if not path.exists():
        path = result_dir / "failure_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(payload: dict[str, object], key: str) -> float:
    return float(payload.get(key) or 0.0)


def as_int(payload: dict[str, object], key: str) -> int:
    return int(payload.get(key) or 0)


def ratio(value: float, baseline: float) -> float:
    return value / baseline if baseline > 0 else 0.0


def pct_delta(value: float, baseline: float) -> float:
    return (ratio(value, baseline) - 1.0) * 100.0 if baseline > 0 else 0.0


def policy_row(
    *,
    policy: str,
    role: str,
    summary: dict[str, object],
    retry_latency_norm: float,
    run_id: str,
    slurm_job_id: str,
    note: str,
    normal_path_overhead_measured: str = "false",
    prefetch_policy: str = "",
) -> dict[str, object]:
    request_count = as_int(summary, "count")
    backed_up = as_int(summary, "response_pre_failover_backed_up_tokens")
    output_tokens = as_int(summary, "response_pre_failover_output_tokens")
    return {
        "policy": policy,
        "role": role,
        "retry_latency_norm": retry_latency_norm,
        "latency_p99_s": as_float(summary, "latency_p99_s"),
        "latency_p95_s": as_float(summary, "latency_p95_s"),
        "latency_mean_s": as_float(summary, "latency_mean_s"),
        "request_count": request_count,
        "success_count": as_int(summary, "success_count"),
        "failover_response_count": as_int(summary, "failover_response_count"),
        "captured_request_count": as_int(summary, "captured_request_count"),
        "dispatched_request_count": as_int(summary, "dispatched_request_count"),
        "pre_failover_output_tokens": output_tokens,
        "pre_failover_backed_up_tokens": backed_up,
        "backed_up_tokens_per_request": ratio(backed_up, request_count),
        "backed_up_tokens_per_output_token": ratio(backed_up, output_tokens),
        "failure_injection_status": as_int(summary, "failure_injection_status"),
        "recovery_status": as_int(summary, "recovery_status"),
        "evidence_tier": "sglang-small-paired-smoke",
        "source": "sglang_dp2_failure_visibility_pair",
        "run_id": run_id,
        "remote_host": "hkbugpumanode",
        "slurm_job_id": slurm_job_id,
        "model_path": "Qwen/Qwen3-8B",
        "workload_id": "dp2_failure_smoke_4req_512tok_8s",
        "dp_size": 2,
        "tp_size": 1,
        "failure_rank": 0,
        "normal_path_overhead_measured": normal_path_overhead_measured,
        "hicache_storage_prefetch_policy": prefetch_policy,
        "note": note,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invisible-dir", type=Path, required=True)
    parser.add_argument("--visible-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--invisible-job-id", default="")
    parser.add_argument("--visible-job-id", default="")
    parser.add_argument("--prefetch-policy", default="")
    parser.add_argument(
        "--same-write-cost-control",
        action="store_true",
        help=(
            "Interpret the invisible run as remote_backup enabled but hidden "
            "from failover lookup, so both runs pay the same write path."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    invisible = load_summary(args.invisible_dir)
    visible = load_summary(args.visible_dir)

    invisible_p99 = as_float(invisible, "latency_p99_s")
    visible_p99 = as_float(visible, "latency_p99_s")
    invisible_backed_up = as_int(invisible, "response_pre_failover_backed_up_tokens")
    visible_backed_up = as_int(visible, "response_pre_failover_backed_up_tokens")
    invisible_output = as_int(invisible, "response_pre_failover_output_tokens")
    visible_output = as_int(visible, "response_pre_failover_output_tokens")

    if args.same_write_cost_control:
        invisible_policy = "SGLang remote_backup lookup disabled"
        invisible_role = "host_invisible_remote_backup_backend_lookup_disabled"
        invisible_note = (
            "remote_backup writes are enabled, but failover lookup is disabled; "
            "this controls for normal-path write cost while hiding the Host frontier"
        )
        visible_policy = "SGLang remote_backup lookup enabled"
        visible_role = "host_visible_remote_backup_backend_lookup_enabled"
        visible_note = (
            "remote_backup writes and failover lookup are both enabled; "
            "this exposes the same written Host frontier during recovery"
        )
        pair_note = (
            "paired same-write-cost summary for Fig. 11b/Fig. 15 evidence chain; "
            "both runs enable remote_backup writes, while only the visible run "
            "allows failure-time lookup into the failed rank frontier"
        )
        normal_path_overhead_measured = "matched-write-cost"
    else:
        invisible_policy = "SGLang invisible control path"
        invisible_role = "host_invisible_no_storage_backend"
        invisible_note = (
            "same failure-control path without a storage backend; "
            "validates retry continuity but exposes zero ACKed Host frontier"
        )
        visible_policy = "SGLang remote_backup frontier"
        visible_role = "host_visible_remote_backup_backend"
        visible_note = (
            "same failure-control shape with remote_backup storage; "
            "validates nonzero ACKed Host frontier but not final restore benefit"
        )
        pair_note = (
            "paired posthoc summary for Fig. 11b/Fig. 15 evidence chain; "
            "remote_backup proves a nonzero Host frontier, while final HPCA "
            "evidence still needs paired policy restore-latency benefit"
        )
        normal_path_overhead_measured = "false"

    rows = [
        policy_row(
            policy=invisible_policy,
            role=invisible_role,
            summary=invisible,
            retry_latency_norm=1.0,
            run_id=args.run_id,
            slurm_job_id=args.invisible_job_id or args.invisible_dir.name,
            note=invisible_note,
            normal_path_overhead_measured=normal_path_overhead_measured,
            prefetch_policy=args.prefetch_policy,
        ),
        policy_row(
            policy=visible_policy,
            role=visible_role,
            summary=visible,
            retry_latency_norm=ratio(visible_p99, invisible_p99),
            run_id=args.run_id,
            slurm_job_id=args.visible_job_id or args.visible_dir.name,
            note=visible_note,
            normal_path_overhead_measured=normal_path_overhead_measured,
            prefetch_policy=args.prefetch_policy,
        ),
    ]

    pair = {
        "run_id": args.run_id,
        "invisible_result_dir": str(args.invisible_dir),
        "visible_result_dir": str(args.visible_dir),
        "invisible_slurm_job_id": args.invisible_job_id or args.invisible_dir.name,
        "visible_slurm_job_id": args.visible_job_id or args.visible_dir.name,
        "invisible_latency_p99_s": invisible_p99,
        "visible_latency_p99_s": visible_p99,
        "visible_retry_latency_norm_vs_invisible": ratio(visible_p99, invisible_p99),
        "visible_latency_delta_pct_vs_invisible": pct_delta(visible_p99, invisible_p99),
        "invisible_pre_failover_output_tokens": invisible_output,
        "visible_pre_failover_output_tokens": visible_output,
        "invisible_pre_failover_backed_up_tokens": invisible_backed_up,
        "visible_pre_failover_backed_up_tokens": visible_backed_up,
        "frontier_token_delta": visible_backed_up - invisible_backed_up,
        "visible_backed_up_tokens_per_request": ratio(
            visible_backed_up, as_int(visible, "count")
        ),
        "visible_backed_up_tokens_per_output_token": ratio(
            visible_backed_up, visible_output
        ),
        "invisible_success_count": as_int(invisible, "success_count"),
        "visible_success_count": as_int(visible, "success_count"),
        "invisible_failover_response_count": as_int(
            invisible, "failover_response_count"
        ),
        "visible_failover_response_count": as_int(visible, "failover_response_count"),
        "evidence_tier": "sglang-small-paired-smoke",
        "source": "sglang_dp2_failure_visibility_pair",
        "hicache_storage_prefetch_policy": args.prefetch_policy,
        "note": pair_note,
    }

    (args.output_dir / "failure_visibility_pair_summary.json").write_text(
        json.dumps(pair, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "failure_visibility_pair_summary.csv", [pair])
    write_csv(args.output_dir / "failure_visibility_pair_rows.csv", rows)
    write_csv(
        args.output_dir / "candidate_fig_hostbackup_visibility_sglang_pair.csv",
        rows,
    )
    print(json.dumps(pair, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
