#!/usr/bin/env python3
"""Chain: wait for 10w route calibration -> build plan -> paired A/B -> CDF."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/comp/zhanzhao/sglsim")
OUT = ROOT / "AegisKV/demo/results/fast_24w_oracle_ab_v1"
CALIB = OUT / "route_calibration_10w_no_backup_c4_f0.json"
ROUTE = OUT / "execution_route_plan_10w.json"
TRACE = ROOT / (
    "AegisKV/presentation/experiments/sglsim_large/traces/"
    "weka_subagent_cdf_10w_trace_oracle_workload.jsonl"
)
SERVER = ROOT / "AegisKV/demo/config/server_qwen3_32b_dp4_pp4_32k.json"
FAILURE = ROOT / (
    "AegisKV/demo/config/failure_manifest_qwen3_32b_dp4_pp4_32k_matched_v13.json"
)
PY = str(ROOT / ".venv/bin/python")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def wait_calibration(timeout_s: int = 7200) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if CALIB.is_file() and CALIB.stat().st_size > 1000:
            payload = json.loads(CALIB.read_text(encoding="utf-8"))
            if int(payload.get("request_count", 0)) > 0:
                print(f"calibration ready: requests={payload['request_count']}", flush=True)
                return
        time.sleep(15)
    raise TimeoutError(f"calibration not ready after {timeout_s}s: {CALIB}")


def build_route_plan() -> None:
    _run(
        [
            PY,
            "-m",
            "AegisKV.demo.build_execution_route_plan",
            "--source-result",
            str(CALIB),
            "--output",
            str(ROUTE),
        ]
    )


def run_pair(concurrency: int = 4, failure_count: int = 1) -> None:
    for policy in ("no_backup", "aegiskv"):
        output = OUT / f"c{concurrency}-f{failure_count}-{policy}.json"
        if output.is_file():
            print(f"skip existing {output}", flush=True)
            continue
        _run(
            [
                PY,
                "-m",
                "AegisKV.demo.run_offline_wct",
                "--server-config",
                str(SERVER),
                "--trace",
                str(TRACE),
                "--route-plan",
                str(ROUTE),
                "--failure-manifest",
                str(FAILURE),
                "--output",
                str(output),
                "--policy",
                policy,
                "--workflow-concurrency",
                str(concurrency),
                "--failure-count",
                str(failure_count),
                "--max-workflows",
                "10",
                "--slack-source",
                "trace_oracle",
                "--artifact-detail",
                "compact",
            ]
        )


def write_summary_and_cdf(concurrency: int = 4, failure_count: int = 1) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = {}
    for policy in ("aegiskv", "no_backup"):
        payload = json.loads(
            (OUT / f"c{concurrency}-f{failure_count}-{policy}.json").read_text()
        )
        wcts = sorted(float(w["wct_s"]) for w in payload["workflows"])
        cells[policy] = {
            "wcts": wcts,
            "mean": float(sum(wcts) / len(wcts)),
            "p50": wcts[len(wcts) // 2],
            "p90": wcts[max(int(0.9 * len(wcts)) - 1, 0)],
            "p99": wcts[max(int(0.99 * len(wcts)) - 1, 0)],
            "retries": sum(len(w.get("retry_events", [])) for w in payload["workflows"]),
            "restored": sum(
                int(r.get("restored_tokens", 0))
                for w in payload["workflows"]
                for r in w.get("retry_events", [])
            ),
            "recompute": sum(
                int(r.get("recompute_tokens", 0))
                for w in payload["workflows"]
                for r in w.get("retry_events", [])
            ),
            "host_runtime_s": float(payload.get("host_runtime_s", 0.0)),
            "request_count": int(payload.get("request_count", 0)),
        }

    mean_a = cells["aegiskv"]["mean"]
    mean_b = cells["no_backup"]["mean"]
    summary = {
        "schema_version": 1,
        "evidence_role": "fast smoke; not held-out primary gate",
        "workflow_count": 10,
        "workflow_concurrency": concurrency,
        "failure_count_requested": failure_count,
        "slack_source": "trace_oracle",
        "paired": {
            "mean_wct_aegiskv_s": mean_a,
            "mean_wct_no_backup_s": mean_b,
            "mean_wct_improvement_pct": 100.0 * (mean_b - mean_a) / mean_b if mean_b else 0.0,
            "p50_aegiskv_s": cells["aegiskv"]["p50"],
            "p50_no_backup_s": cells["no_backup"]["p50"],
            "p90_aegiskv_s": cells["aegiskv"]["p90"],
            "p90_no_backup_s": cells["no_backup"]["p90"],
            "p99_aegiskv_s": cells["aegiskv"]["p99"],
            "p99_no_backup_s": cells["no_backup"]["p99"],
            "retries_aegiskv": cells["aegiskv"]["retries"],
            "retries_no_backup": cells["no_backup"]["retries"],
            "additional_restored_tokens": (
                cells["aegiskv"]["restored"] - cells["no_backup"]["restored"]
            ),
            "recompute_reduction_tokens": (
                cells["no_backup"]["recompute"] - cells["aegiskv"]["recompute"]
            ),
        },
        "cells": cells,
    }
    (OUT / "fast_10w_ab_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for policy, style in (("aegiskv", "-"), ("no_backup", "--")):
        xs = np.array(cells[policy]["wcts"])
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", linestyle=style, label=policy)
    ax.set_xlabel("Workflow completion time (s)")
    ax.set_ylabel("CDF")
    ax.set_title(
        f"Fast 10-workflow WCT CDF (c={concurrency}, f={failure_count}, trace-oracle)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fast_10w_c4_f1_wct_cdf.png", dpi=160)
    fig.savefig(OUT / "fast_10w_c4_f1_wct_cdf.pdf")
    print(json.dumps(summary["paired"], ensure_ascii=False), flush=True)


def main() -> int:
    wait_calibration()
    build_route_plan()
    run_pair(concurrency=4, failure_count=1)
    write_summary_and_cdf(concurrency=4, failure_count=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
