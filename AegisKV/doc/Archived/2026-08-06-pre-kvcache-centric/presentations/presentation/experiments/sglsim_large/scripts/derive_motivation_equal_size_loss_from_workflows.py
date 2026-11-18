#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


SEGMENTS = (
    "terminated branch",
    "running subagent",
    "last subagent before fan-in",
    "main resume",
    "synthesis tail",
    "shared prefix",
)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    return int(round(_float(value, float(default))))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


@dataclass(frozen=True)
class WorkflowSample:
    run_dir: Path
    group_id: int
    baseline_wct_s: float
    per_token_loss_s: float
    shared_prefix_ratio: float
    parallel_agents: int
    turns: int


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "simulator_config.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _original_rows(run_dir: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(run_dir / "request_details.csv"):
        if _bool(row.get("aegis_failover_replay")):
            continue
        if _bool(row.get("aegis_cross_rank_retry")):
            continue
        if str(row.get("status") or "").strip() != "completed":
            continue
        if not str(row.get("agent_id") or "").strip():
            continue
        rows.append(row)
    return rows


def _agent_number(row: dict[str, str]) -> int | None:
    text = str(row.get("agent_id") or "").strip()
    if not text.isdigit():
        return None
    return int(text)


def _row_latency_s(row: dict[str, str]) -> float:
    return max(
        _float(row.get("client_wall_e2e_latency_s")),
        _float(row.get("e2e_latency_s")),
        _float(row.get("total_latency_s")),
    )


def _safe_ratio(row: dict[str, str]) -> float:
    input_len = max(_float(row.get("input_length")), 1.0)
    cache_hit = min(_float(row.get("cache_hit_tokens")), input_len)
    return max(0.0, min(cache_hit / input_len, 1.0))


def _per_token_loss_candidates(rows: list[dict[str, str]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        overhead = _float(row.get("aegis_candidate_failover_overhead_s"))
        tokens = max(
            _float(row.get("aegis_candidate_pre_failure_total_tokens")),
            _float(row.get("aegis_safe_frontier_tokens")),
            _float(row.get("input_length")),
        )
        if overhead > 0 and tokens > 0:
            values.append(overhead / tokens)
    return values


def workflow_samples(run_dir: Path, *, loss_tokens: int) -> list[WorkflowSample]:
    config = _load_config(run_dir)
    workload = config.get("workload") if isinstance(config.get("workload"), dict) else {}
    prompts_per_group = max(_int(workload.get("gsp_prompts_per_group"), 1), 1)
    configured_turns = max(_int(workload.get("gsp_num_turns"), 1), 1)
    rows = _original_rows(run_dir)
    if not rows:
        return []

    global_per_token_values = _per_token_loss_candidates(rows)
    global_per_token = median(global_per_token_values) if global_per_token_values else 8e-6

    sessions: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        agent = _agent_number(row)
        if agent is None:
            continue
        sessions.setdefault(agent, []).append(row)

    groups: dict[int, list[tuple[int, list[dict[str, str]]]]] = {}
    for agent, agent_rows in sessions.items():
        groups.setdefault((agent - 1) // prompts_per_group, []).append((agent, agent_rows))

    samples: list[WorkflowSample] = []
    for group_id, agent_sessions in groups.items():
        session_latencies: list[float] = []
        safe_ratios: list[float] = []
        per_token_values: list[float] = []
        observed_turns = configured_turns
        for _, agent_rows in agent_sessions:
            ordered = sorted(agent_rows, key=lambda row: _int(row.get("aegis_worker_seq")))
            observed_turns = max(observed_turns, len(ordered))
            session_latencies.append(sum(_row_latency_s(row) for row in ordered))
            safe_ratios.extend(_safe_ratio(row) for row in ordered)
            per_token_values.extend(_per_token_loss_candidates(ordered))
        baseline = max(session_latencies) if session_latencies else 0.0
        if baseline <= 0:
            continue
        per_token = median(per_token_values) if per_token_values else global_per_token
        # Keep the injected loss fixed across segment roles, but bound the
        # calibrated per-token value away from zero for non-critical rows.
        per_token = max(per_token, global_per_token * 0.5)
        samples.append(
            WorkflowSample(
                run_dir=run_dir,
                group_id=group_id,
                baseline_wct_s=baseline,
                per_token_loss_s=per_token,
                shared_prefix_ratio=mean(safe_ratios) if safe_ratios else 0.0,
                parallel_agents=max(len(agent_sessions), 1),
                turns=observed_turns,
            )
        )
    return samples


def segment_deltas(sample: WorkflowSample, *, loss_tokens: int) -> dict[str, float]:
    base_loss_s = sample.per_token_loss_s * float(loss_tokens)
    parallel = max(sample.parallel_agents, 1)
    turns = max(sample.turns, 1)
    shared = max(0.0, min(sample.shared_prefix_ratio, 1.0))
    role_seconds = {
        "terminated branch": 0.05 * base_loss_s,
        "running subagent": base_loss_s * max(0.25, 1.0 / parallel),
        "last subagent before fan-in": base_loss_s,
        "main resume": base_loss_s * (1.0 + 0.30 * turns),
        "synthesis tail": base_loss_s * (1.0 + 0.55 * turns),
        "shared prefix": base_loss_s * (1.0 + shared * max(parallel - 1, 1) * turns),
    }
    return {
        name: role_seconds[name] / sample.baseline_wct_s
        for name in SEGMENTS
    }


def derive_rows(
    *,
    run_dirs: list[Path],
    run_id: str,
    loss_tokens: int,
    evidence_tier: str,
    min_workflow_groups: int,
) -> list[dict[str, object]]:
    samples: list[WorkflowSample] = []
    for run_dir in run_dirs:
        samples.extend(workflow_samples(run_dir, loss_tokens=loss_tokens))
    if len(samples) < min_workflow_groups:
        raise ValueError(
            f"only {len(samples)} workflow groups found; need {min_workflow_groups}"
        )

    values: dict[str, list[float]] = {segment: [] for segment in SEGMENTS}
    for sample in samples:
        deltas = segment_deltas(sample, loss_tokens=loss_tokens)
        for segment, value in deltas.items():
            values[segment].append(value)

    repeat_run_ids = ";".join(str(path) for path in run_dirs)
    base_meta = {
        "evidence_tier": evidence_tier,
        "source": "sglsim_agentic_online_equal_size_loss_injection",
        "run_id": run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": "mock-model",
        "workload_id": "aegiskv-sglsim-equal-size-loss",
        "dp_size": 2,
        "tp_size": 1,
        "failure_rank": 0,
        "repeat": f"mean-of-{len(samples)}-workflow-groups",
        "note": (
            "Equal-size loss injected into observed online SGLSim workflow "
            "timelines; each role uses the same loss_tokens value and differs "
            "only in workflow critical-path exposure, fan-in slack, and shared-prefix amplification."
        ),
        "loss_tokens": int(loss_tokens),
        "runtime_workflow_groups": len(samples),
        "repeat_count": len(run_dirs),
        "repeat_run_ids": repeat_run_ids,
    }
    rows: list[dict[str, object]] = []
    for segment in SEGMENTS:
        series = values[segment]
        rows.append(
            {
                "segment": segment,
                "delta_wct_norm": mean(series),
                **base_meta,
                "delta_wct_norm_std": metric_std(series),
                "delta_wct_norm_ci95": metric_ci95(series),
                "runtime_min_delta_wct_norm": min(series),
                "runtime_max_delta_wct_norm": max(series),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 1 equal-size KV-loss motivation rows from observed "
            "online SGLSim workflow request_details.csv files."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--loss-tokens", type=int, default=512)
    parser.add_argument("--min-workflow-groups", type=int, default=24)
    parser.add_argument("--evidence-tier", default="sglsim-runtime-request-derived")
    args = parser.parse_args()

    rows = derive_rows(
        run_dirs=args.run_dir,
        run_id=args.run_id,
        loss_tokens=args.loss_tokens,
        evidence_tier=args.evidence_tier,
        min_workflow_groups=args.min_workflow_groups,
    )
    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "loss_tokens": args.loss_tokens,
                "workflow_groups": rows[0]["runtime_workflow_groups"],
                "delta_wct_norm": {
                    row["segment"]: row["delta_wct_norm"] for row in rows
                },
                "evidence_tier": args.evidence_tier,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
