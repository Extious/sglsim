#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_by_policy(rows: list[dict[str, str]], policy: str) -> dict[str, str]:
    for row in rows:
        if row.get("policy") == policy:
            return row
    raise ValueError(f"missing policy {policy!r}")


def row_by_policy_if_present(rows: list[dict[str, str]], policy: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("policy") == policy:
            return row
    return None


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
    parser = argparse.ArgumentParser(
        description=(
            "Build Fig. 15 calibration anchors from normal-path overhead evidence "
            "and, when available, Host-visible retry evidence."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument(
        "--hostbackup-visibility-csv",
        type=Path,
        default=ROOT / "benchmark/AegisKV/draft/figures/data/measured/fig_hostbackup_visibility.csv",
        help="Optional measured Fig. 11b CSV containing SGLang/SGLSim retry-visibility anchors.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    sglang_base = row_by_policy(rows, "SGLang baseline")
    sglang_hicache = row_by_policy(rows, "SGLang HiCache WT")
    sglsim_base = row_by_policy(rows, "none")
    sglsim_aegis = row_by_policy(rows, "AegisKV")

    current_sha = git_sha()
    output_rows = [
        {
            "policy": "baseline",
            "metric": "normal_path_p99_overhead_pct",
            "sglang_norm": _float(sglang_base.get("p99_overhead_pct")),
            "sglsim_norm": _float(sglsim_base.get("p99_overhead_pct")),
            "evidence_tier": "mixed-preliminary",
            "source": "sglang_sglsim_normal_path_overhead_anchor",
            "run_id": args.run_id,
            "git_sha": current_sha,
            "remote_host": "hkbugpumanode",
            "slurm_job_id": sglang_base.get("slurm_job_id", ""),
            "model_path": sglang_base.get("model_path", "Qwen/Qwen3-8B"),
            "workload_id": "normal_path_overhead_anchor",
            "dp_size": 2,
            "tp_size": 1,
            "failure_rank": "",
            "repeat": sglang_base.get("repeat", "aggregate"),
            "note": (
                "calibration anchor from SGLang paired overhead aggregate and "
                "current SGLSim normal-path overhead evidence; not final policy ranking"
            ),
            "sglang_norm_std": 0.0,
            "sglang_norm_ci95": 0.0,
            "sglsim_norm_std": _float(sglsim_base.get("p99_overhead_pct_std")),
            "sglsim_norm_ci95": _float(sglsim_base.get("p99_overhead_pct_ci95")),
            "repeat_count": sglang_base.get("repeat_count", ""),
            "repeat_run_ids": sglang_base.get("repeat_run_ids", ""),
        },
        {
            "policy": "protected path",
            "metric": "normal_path_p99_overhead_pct",
            "sglang_norm": _float(sglang_hicache.get("p99_overhead_pct")),
            "sglsim_norm": _float(sglsim_aegis.get("p99_overhead_pct")),
            "evidence_tier": "mixed-preliminary",
            "source": "sglang_sglsim_normal_path_overhead_anchor",
            "run_id": args.run_id,
            "git_sha": current_sha,
            "remote_host": "hkbugpumanode",
            "slurm_job_id": sglang_hicache.get("slurm_job_id", ""),
            "model_path": sglang_hicache.get("model_path", "Qwen/Qwen3-8B"),
            "workload_id": "normal_path_overhead_anchor",
            "dp_size": 2,
            "tp_size": 1,
            "failure_rank": "",
            "repeat": sglang_hicache.get("repeat", "aggregate"),
            "note": (
                "SGLang point is HiCache write-through; SGLSim point is the "
                "current AegisKV preliminary online overhead evidence, so this "
                "row is an anchor rather than final policy-ranking evidence"
            ),
            "sglang_norm_std": 0.0,
            "sglang_norm_ci95": 0.0,
            "sglsim_norm_std": _float(sglsim_aegis.get("p99_overhead_pct_std")),
            "sglsim_norm_ci95": _float(sglsim_aegis.get("p99_overhead_pct_ci95")),
            "repeat_count": sglang_hicache.get("repeat_count", ""),
            "repeat_run_ids": sglang_hicache.get("repeat_run_ids", ""),
        },
    ]

    if args.hostbackup_visibility_csv.exists():
        visibility_rows = read_rows(args.hostbackup_visibility_csv)
        sglsim_retry_control = row_by_policy_if_present(
            visibility_rows, "baseline write-through invisible"
        )
        sglsim_host_visible = row_by_policy_if_present(
            visibility_rows, "Host visible only"
        )
        sglang_retry_control = row_by_policy_if_present(
            visibility_rows, "SGLang remote_backup lookup disabled"
        )
        sglang_host_visible = row_by_policy_if_present(
            visibility_rows, "SGLang remote_backup lookup enabled"
        )
        if (
            sglsim_retry_control is not None
            and sglsim_host_visible is not None
            and sglang_retry_control is not None
            and sglang_host_visible is not None
        ):
            visibility_run_id = (
                f"{args.run_id};"
                f"{sglsim_host_visible.get('run_id', '')};"
                f"{sglang_host_visible.get('run_id', '')}"
            )
            output_rows.extend(
                [
                    {
                        "policy": "retry control",
                        "metric": "retry_latency_norm",
                        "sglang_norm": _float(sglang_retry_control.get("retry_latency_norm"), 1.0),
                        "sglsim_norm": _float(sglsim_retry_control.get("retry_latency_norm"), 1.0),
                        "evidence_tier": "mixed-preliminary",
                        "source": "sglang_sglsim_failure_visibility_anchor",
                        "run_id": visibility_run_id,
                        "git_sha": current_sha,
                        "remote_host": sglang_retry_control.get("remote_host", ""),
                        "slurm_job_id": sglang_retry_control.get("slurm_job_id", ""),
                        "model_path": sglang_retry_control.get("model_path", "Qwen/Qwen3-8B"),
                        "workload_id": "hostbackup_visibility_anchor",
                        "dp_size": sglang_retry_control.get("dp_size", 2),
                        "tp_size": sglang_retry_control.get("tp_size", 1),
                        "failure_rank": sglang_retry_control.get("failure_rank", 0),
                        "repeat": sglang_retry_control.get("repeat", "aggregate"),
                        "note": (
                            "Retry-latency control anchor: SGLang lookup-disabled "
                            "remote_backup control and SGLSim write-through invisible "
                            "baseline are both normalized to their no-visible-restore path."
                        ),
                        "sglang_norm_std": _float(sglang_retry_control.get("retry_latency_norm_std")),
                        "sglang_norm_ci95": _float(sglang_retry_control.get("retry_latency_norm_ci95")),
                        "sglsim_norm_std": _float(sglsim_retry_control.get("retry_latency_norm_std")),
                        "sglsim_norm_ci95": _float(sglsim_retry_control.get("retry_latency_norm_ci95")),
                        "repeat_count": sglang_retry_control.get("repeat_count", ""),
                        "repeat_run_ids": sglang_retry_control.get("repeat_run_ids", ""),
                    },
                    {
                        "policy": "Host-visible retry",
                        "metric": "retry_latency_norm",
                        "sglang_norm": _float(sglang_host_visible.get("retry_latency_norm")),
                        "sglsim_norm": _float(sglsim_host_visible.get("retry_latency_norm")),
                        "evidence_tier": "mixed-preliminary",
                        "source": "sglang_sglsim_failure_visibility_anchor",
                        "run_id": visibility_run_id,
                        "git_sha": current_sha,
                        "remote_host": sglang_host_visible.get("remote_host", ""),
                        "slurm_job_id": sglang_host_visible.get("slurm_job_id", ""),
                        "model_path": sglang_host_visible.get("model_path", "Qwen/Qwen3-8B"),
                        "workload_id": "hostbackup_visibility_anchor",
                        "dp_size": sglang_host_visible.get("dp_size", 2),
                        "tp_size": sglang_host_visible.get("tp_size", 1),
                        "failure_rank": sglang_host_visible.get("failure_rank", 0),
                        "repeat": sglang_host_visible.get("repeat", "aggregate"),
                        "note": (
                            "Directional calibration anchor: SGLang same-write-cost "
                            "best-effort Host-visible retry and SGLSim Host-visible "
                            "replay both improve over their retry controls; this is "
                            "not final policy-ranking evidence."
                        ),
                        "sglang_norm_std": _float(sglang_host_visible.get("retry_latency_norm_std")),
                        "sglang_norm_ci95": _float(sglang_host_visible.get("retry_latency_norm_ci95")),
                        "sglsim_norm_std": _float(sglsim_host_visible.get("retry_latency_norm_std")),
                        "sglsim_norm_ci95": _float(sglsim_host_visible.get("retry_latency_norm_ci95")),
                        "repeat_count": sglang_host_visible.get("repeat_count", ""),
                        "repeat_run_ids": sglang_host_visible.get("repeat_run_ids", ""),
                    },
                ]
            )

    write_csv(args.output_csv, output_rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "input_csv": str(args.input_csv),
                "output_csv": str(args.output_csv),
                "rows": len(output_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
