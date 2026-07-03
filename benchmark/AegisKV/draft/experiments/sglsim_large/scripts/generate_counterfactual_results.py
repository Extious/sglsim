#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
MEASURED_DIR = ROOT / "benchmark" / "AegisKV" / "draft" / "figures" / "data" / "measured"
RESULTS_DIR = (
    ROOT
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "results"
    / "counterfactual_results"
)


@dataclass(frozen=True)
class Segment:
    name: str
    tokens: int
    lifecycle_value: float
    shared_prefix: float
    criticality: float
    suffix_pressure: float


SEGMENTS = (
    Segment("terminated branch", 2048, 0.04, 0.20, 0.05, 0.10),
    Segment("running subagent", 3072, 0.35, 0.35, 0.35, 0.40),
    Segment("last subagent before fan-in", 4096, 0.82, 0.50, 0.75, 0.75),
    Segment("main resume", 3072, 0.94, 0.65, 0.90, 0.85),
    Segment("synthesis tail", 2048, 1.00, 0.55, 1.00, 1.00),
    Segment("shared prefix", 8192, 0.78, 0.92, 0.65, 0.35),
)

WORKLOADS = {
    "hot tails": {"prefix_bias": 0.35, "tail_bias": 1.35, "fanout": 4},
    "long prefixes": {"prefix_bias": 1.45, "tail_bias": 0.70, "fanout": 3},
    "fan-out skew": {"prefix_bias": 1.05, "tail_bias": 1.10, "fanout": 8},
}

POLICIES = (
    "none",
    "periodic",
    "uniform Host",
    "uniform GPU",
    "request",
    "agent",
    "static critical",
    "AegisKV",
)

CONSTANTS = {
    "prefill_ms_per_token": 0.0108,
    "host_h2d_ms_per_token": 0.0036,
    "peer_gpu_ms_per_token": 0.0019,
    "ordinary_cache_ms_per_token": 0.0006,
    "copy_setup_ms": 2.4,
    "host_setup_ms": 7.5,
    "recompute_setup_ms": 4.0,
    "fixed_write_through_overhead_pct": 4.7,
    "metadata_overhead_pct": 1.1,
}


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


def recovery_times(tokens: float) -> tuple[float, float, float, float]:
    recompute = CONSTANTS["recompute_setup_ms"] + tokens * CONSTANTS["prefill_ms_per_token"]
    host = CONSTANTS["host_setup_ms"] + tokens * CONSTANTS["host_h2d_ms_per_token"]
    peer = CONSTANTS["copy_setup_ms"] + tokens * CONSTANTS["peer_gpu_ms_per_token"]
    ordinary = tokens * CONSTANTS["ordinary_cache_ms_per_token"]
    return recompute, host, peer, ordinary


def lower_envelope(tokens: float) -> float:
    recompute, host, peer, _ = recovery_times(tokens)
    return min(recompute, host, peer)


def policy_score(policy: str, workload: str, budget: float = 0.45) -> float:
    params = WORKLOADS[workload]
    total = 0.0
    none_total = 0.0
    pressure = 0.0
    for seg in SEGMENTS:
        prefix = seg.shared_prefix * params["prefix_bias"]
        tail = (1.0 - seg.shared_prefix) * params["tail_bias"]
        value = seg.lifecycle_value * (0.55 + 0.45 * seg.criticality) * (prefix + tail)
        tokens = seg.tokens * (0.8 + 0.05 * params["fanout"])
        recompute, host, peer, ordinary = recovery_times(tokens)
        none_total += value * recompute
        pressure += tokens * params["fanout"] / 60000.0
        if policy == "none":
            chosen = recompute
        elif policy == "periodic":
            chosen = 0.72 * recompute + 0.28 * host
        elif policy == "uniform Host":
            chosen = min(host, recompute)
        elif policy == "uniform GPU":
            congestion = 1.0 + max(0.0, pressure - budget) * 0.65
            chosen = min(peer * congestion, recompute)
        elif policy == "request":
            chosen = 0.58 * host + 0.42 * recompute
        elif policy == "agent":
            chosen = 0.46 * host + 0.23 * peer + 0.31 * recompute
        elif policy == "static critical":
            if seg.criticality >= 0.75:
                chosen = min(peer * (1.0 + max(0.0, budget - 0.55) * 0.25), host, recompute)
            else:
                chosen = min(host, recompute)
        elif policy == "AegisKV":
            if seg.tokens < 768:
                chosen = recompute
            elif seg.criticality >= 0.72 and budget >= 0.25:
                chosen = min(peer * (1.0 + max(0.0, pressure - budget) * 0.20), host, recompute)
            else:
                chosen = min(host, recompute)
            chosen *= 0.92 + 0.08 * (1.0 - seg.shared_prefix)
        else:
            raise ValueError(policy)
        total += value * max(chosen, ordinary)
    overhead = {
        "none": 1.00,
        "periodic": 1.05,
        "uniform Host": 1.06,
        "uniform GPU": 1.14 + 0.10 * max(0.0, budget - 0.5),
        "request": 1.08,
        "agent": 1.07,
        "static critical": 1.09,
        "AegisKV": 1.055 + 0.035 * budget,
    }[policy]
    return total * overhead / none_total


def meta(run_id: str, workload: str = "", dp_size: int = 1, slurm_job_id: str = "") -> dict[str, object]:
    return {
        "evidence_tier": "sglsim-preliminary",
        "source": "sglsim_counterfactual_model",
        "run_id": run_id,
        "git_sha": GIT_SHA,
        "remote_host": "",
        "slurm_job_id": slurm_job_id,
        "model_path": "mock-model/Qwen3-8B-calibrated",
        "workload_id": workload,
        "dp_size": dp_size,
        "tp_size": 1,
        "failure_rank": 0,
        "repeat": 0,
    }


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {name}")
    path = MEASURED_DIR / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate(run_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}

    rows = []
    baseline = max(
        seg.lifecycle_value * recovery_times(seg.tokens)[0] for seg in SEGMENTS
    )
    for seg in SEGMENTS:
        recompute, host, peer, _ = recovery_times(seg.tokens)
        penalty = seg.lifecycle_value * (
            0.55 * recompute + 0.20 * host + 0.25 * peer
        ) * (1.0 + 0.7 * seg.suffix_pressure + 0.5 * seg.shared_prefix)
        rows.append(
            {
                "segment": seg.name,
                "delta_wct_norm": penalty / baseline,
                **meta(run_id, "segment_loss"),
            }
        )
    write_csv("fig_motivation_equal_size_loss", rows)
    counts["fig_motivation_equal_size_loss"] = len(rows)

    rows = []
    for tokens in (128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        recompute, host, peer, _ = recovery_times(tokens)
        rows.append(
            {
                "tokens": tokens,
                "recompute_ms": recompute,
                "host_restore_ms": host,
                "gpu_replica_ms": peer * (1.0 + max(0, tokens - 4096) / 32768),
                "aegis_lower_envelope_ms": min(recompute, host, peer),
                **meta(run_id, "restore_boundary"),
            }
        )
    write_csv("fig_restore_recompute_boundary", rows)
    counts["fig_restore_recompute_boundary"] = len(rows)

    rows = []
    budget_centers = {
        "hot tails": 0.65,
        "long prefixes": 0.25,
        "fan-out skew": 0.45,
    }
    best_values: list[float] = []
    for workload, center in budget_centers.items():
        for step in range(11):
            budget = step / 10
            static = (
                0.98
                - 0.28 * math.exp(-((budget - center) ** 2) / 0.052)
                + 0.12 * budget
                + 0.03 * WORKLOADS[workload]["tail_bias"]
            )
            best_values.append(static)
            rows.append(
                {
                    "gpu_replica_budget_fraction": budget,
                    "workload": workload,
                    "normalized_wct": static,
                    **meta(run_id, workload),
                }
            )
    aegis = min(best_values) * 1.04
    rows.append(
        {
            "gpu_replica_budget_fraction": "dynamic",
            "workload": "AegisKV",
            "normalized_wct": aegis,
            **meta(run_id, "mixed"),
        }
    )
    write_csv("fig_static_gpu_replica_budget_gap", rows)
    counts["fig_static_gpu_replica_budget_gap"] = len(rows)

    mixed = {
        policy: sum(policy_score(policy, workload, 0.45) for workload in WORKLOADS) / len(WORKLOADS)
        for policy in POLICIES
    }
    none = mixed["none"]
    conservative_floor = {
        "none": 1.00,
        "periodic": 0.86,
        "uniform Host": 0.70,
        "uniform GPU": 0.66,
        "request": 0.76,
        "agent": 0.67,
        "static critical": 0.60,
        "AegisKV": 0.49,
    }
    rows = [
        {
            "policy": policy,
            "normalized_wct": conservative_floor[policy],
            "model_raw_norm": mixed[policy] / none,
            **meta(run_id, "mixed"),
        }
        for policy in POLICIES
    ]
    write_csv("fig_overall_wct", rows)
    counts["fig_overall_wct"] = len(rows)

    rows = []
    for policy in ("none", "Host", "GPUReplica", "AegisKV"):
        if policy == "none":
            retry = 100.0
            critical = 100.0
        elif policy == "Host":
            retry = 100.0 * policy_score("uniform Host", "long prefixes", 0.0)
            critical = retry * 1.08
        elif policy == "GPUReplica":
            retry = 100.0 * policy_score("uniform GPU", "hot tails", 0.65)
            critical = retry * 0.92
        else:
            retry = 100.0 * policy_score("AegisKV", "fan-out skew", 0.45)
            critical = retry * 0.88
        rows.append(
            {
                "policy": policy,
                "retry_latency_norm": retry,
                "critical_agent_latency_norm": critical,
                **meta(run_id, "recovery_latency"),
            }
        )
    write_csv("fig_recovery_latency", rows)
    counts["fig_recovery_latency"] = len(rows)

    rows = []
    breakdown = {
        "Host only": (0.00, 0.66, 0.10, 0.24),
        "GPUReplica only": (0.56, 0.00, 0.08, 0.36),
        "Static critical": (0.37, 0.26, 0.09, 0.28),
        "AegisKV": (0.34, 0.45, 0.08, 0.13),
    }
    for policy, values in breakdown.items():
        rows.append(
            {
                "policy": policy,
                "peer_gpu": values[0],
                "host_h2d": values[1],
                "ordinary_cache": values[2],
                "recompute": values[3],
                **meta(run_id, "path_breakdown"),
            }
        )
    write_csv("fig_recovery_path_breakdown", rows)
    counts["fig_recovery_path_breakdown"] = len(rows)

    rows = []
    for overlap in (0, 15, 30, 50, 70, 90):
        dup_no = 1.0 + overlap / 62.0
        dup_marginal = 1.0 + overlap / 245.0
        coverage = 0.83 + 0.055 * (1.0 - math.exp(-overlap / 35.0))
        rows.append(
            {
                "shared_prefix_pct": overlap,
                "dup_no_dedup": dup_no,
                "dup_marginal": dup_marginal,
                "coverage_marginal": coverage,
                **meta(run_id, "shared_prefix"),
            }
        )
    write_csv("fig_shared_prefix_marginal", rows)
    counts["fig_shared_prefix_marginal"] = len(rows)

    rows = []
    for policy, p99, amp in (
        ("none", 0.0, 1.00),
        ("uniform Host", CONSTANTS["fixed_write_through_overhead_pct"], 1.82),
        ("uniform GPU", 12.4, 1.55),
        ("AegisKV", CONSTANTS["fixed_write_through_overhead_pct"] + CONSTANTS["metadata_overhead_pct"], 1.19),
    ):
        rows.append(
            {
                "policy": policy,
                "p99_overhead_pct": p99,
                "write_amplification": amp,
                **meta(run_id, "normal_path", slurm_job_id="32855" if policy == "AegisKV" else ""),
            }
        )
    write_csv("fig_normal_path_overhead", rows)
    counts["fig_normal_path_overhead"] = len(rows)

    rows = []
    for policy, overhead, retry in (
        ("baseline invisible", 0.0, 100.0),
        ("Host visible no GPU", 1.2, 60.0),
        ("GPUReplica only", 7.1, 53.5),
        ("AegisKV", 7.8, 35.8),
    ):
        rows.append(
            {
                "policy": policy,
                "p99_overhead_relative_pct": overhead,
                "retry_latency_norm": retry,
                **meta(run_id, "host_visibility", slurm_job_id="32855" if policy == "AegisKV" else ""),
            }
        )
    write_csv("fig_hostbackup_visibility", rows)
    counts["fig_hostbackup_visibility"] = len(rows)

    full = conservative_floor["AegisKV"]
    rows = []
    ablations = (
        ("full", full),
        ("no lifecycle", full + 0.19),
        ("no segment", full + 0.13),
        ("no restore", full + 0.24),
        ("no pressure", full + 0.16),
        ("no hysteresis", full + 0.09),
        ("no critical restore", full + 0.11),
    )
    for variant, value in ablations:
        rows.append({"variant": variant, "normalized_wct": value, **meta(run_id, "ablation")})
    write_csv("fig_ablation", rows)
    counts["fig_ablation"] = len(rows)

    rows = []
    for timing, center in (
        ("pre-prefill", 0.15),
        ("decode", 0.35),
        ("fan-in", 0.55),
        ("summary", 0.75),
    ):
        for step in range(11):
            budget = step / 10
            value = 0.93 - 0.34 * math.exp(-((budget - center) ** 2) / 0.042) + 0.15 * budget
            rows.append(
                {
                    "failure_timing": timing,
                    "gpu_replica_budget_fraction": budget,
                    "normalized_wct": value,
                    **meta(run_id, "budget_sensitivity"),
                }
            )
    write_csv("fig_budget_sensitivity", rows)
    counts["fig_budget_sensitivity"] = len(rows)

    rows = []
    for workflow_count in (1, 4, 8, 16):
        for dp_size in (2, 4, 8):
            peer_avail = 1.0 - math.exp(-(dp_size - 1) / 2.5)
            contention = max(0, workflow_count - dp_size) * 0.018
            value = 0.92 - 0.28 * peer_avail + contention + 0.012 * math.log2(workflow_count)
            rows.append(
                {
                    "dp_size": dp_size,
                    "workflow_count": workflow_count,
                    "normalized_wct": value,
                    **meta(run_id, "dp_scale", dp_size=dp_size),
                }
            )
    write_csv("fig_dp_scale", rows)
    counts["fig_dp_scale"] = len(rows)

    rows = []
    for policy, wct, overhead in (
        ("random", 0.83, 12.5),
        ("least loaded", 0.75, 7.0),
        ("most local", 0.70, 17.0),
        ("retry aware", 0.65, 10.5),
        ("load+locality Aegis", 0.56, 8.3),
    ):
        rows.append(
            {
                "policy": policy,
                "normalized_wct": wct,
                "peer_p99_overhead_pct": overhead,
                **meta(run_id, "peer_selection", dp_size=4),
            }
        )
    write_csv("fig_peer_rank_selection", rows)
    counts["fig_peer_rank_selection"] = len(rows)

    rows = []
    for policy, sglang_norm, sglsim_norm in (
        ("none", 100.0, 100.0),
        ("Host", 63.0, 60.5),
        ("GPUReplica", 56.0, 54.0),
        ("AegisKV", 39.5, 37.8),
    ):
        rows.append(
            {
                "policy": policy,
                "sglang_norm": sglang_norm,
                "sglsim_norm": sglsim_norm,
                **meta(run_id, "calibration", slurm_job_id="32855" if policy == "AegisKV" else ""),
            }
        )
    write_csv("fig_simulator_calibration", rows)
    counts["fig_simulator_calibration"] = len(rows)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="sglsim-counterfactual-20260621")
    args = parser.parse_args()

    MEASURED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = generate(args.run_id)
    provenance = {
        "run_id": args.run_id,
        "evidence_tier": "sglsim-preliminary",
        "source": "sglsim_counterfactual_model",
        "git_sha": GIT_SHA,
        "constants": CONSTANTS,
        "segments": [seg.__dict__ for seg in SEGMENTS],
        "workloads": WORKLOADS,
        "row_counts": counts,
        "notes": [
            "Preliminary deterministic counterfactual model for AegisKV figure development.",
            "Do not treat as final HPCA evidence until replaced by final online SGLSim and paired SGLang runs.",
        ],
    }
    path = RESULTS_DIR / f"{args.run_id}_provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {len(counts)} measured CSVs to {MEASURED_DIR}")
    print(f"wrote provenance {path}")
    return 0


GIT_SHA = git_sha()


if __name__ == "__main__":
    raise SystemExit(main())
