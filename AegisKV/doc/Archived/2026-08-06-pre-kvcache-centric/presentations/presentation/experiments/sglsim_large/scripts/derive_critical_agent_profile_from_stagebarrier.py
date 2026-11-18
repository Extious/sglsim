#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT.parent.parent
    / "figures"
    / "data"
    / "measured"
    / "fig_critical_agent_profile.csv"
)


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    return int(round(_float(value, float(default))))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def metric_ci95(values: list[float]) -> float:
    return 1.96 * metric_std(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "simulator_config.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _original_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(run_dir / "request_details.csv"):
        if str(row.get("status") or "").strip() != "completed":
            continue
        if _bool(row.get("aegis_failover_replay")):
            continue
        if _bool(row.get("aegis_cross_rank_retry")):
            continue
        if not str(row.get("workflow_id") or "").strip():
            continue
        rows.append(row)
    return rows


def _latency_s(row: dict[str, str]) -> float:
    return max(
        _float(row.get("client_wall_e2e_latency_s")),
        _float(row.get("e2e_latency_s")),
        _float(row.get("total_latency_s")),
    )


def _output_tokens(row: dict[str, str]) -> int:
    return max(
        _int(row.get("target_output_tokens")),
        _int(row.get("generated_tokens")),
        _int(row.get("output_length")),
    )


def _slot_label(stage_id: int, stage_width: int, local_idx: int) -> str:
    if stage_width == 1:
        return f"S{stage_id} singleton"
    role = "long Critical" if local_idx == 0 else f"Non-Critical {local_idx}"
    return f"S{stage_id} fan-out {role}"


def _slot_sort_key(row: dict[str, object]) -> tuple[int, int]:
    return (int(row["stage_id"]), int(row["stage_local_agent_index"]))


def derive_rows(run_dirs: list[Path], *, run_id: str, evidence_tier: str) -> list[dict[str, object]]:
    slot_values: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    workflow_ids: set[str] = set()
    configs = [_load_config(run_dir) for run_dir in run_dirs]

    for run_dir in run_dirs:
        rows = _original_rows(run_dir)
        workflow_ids.update(str(row.get("workflow_id") or "") for row in rows)
        stage_max_latency: dict[tuple[str, int], float] = defaultdict(float)
        for row in rows:
            key = (str(row.get("workflow_id") or ""), _int(row.get("stage_id"), -1))
            stage_max_latency[key] = max(stage_max_latency[key], _latency_s(row))
        for row in rows:
            stage_id = _int(row.get("stage_id"), -1)
            stage_width = max(_int(row.get("stage_width"), 1), 1)
            local_idx = _int(row.get("stage_local_agent_index"), 0)
            latency = _latency_s(row)
            stage_key = (str(row.get("workflow_id") or ""), stage_id)
            slot_values[(stage_id, stage_width, local_idx)].append(
                {
                    "latency_ms": latency * 1000.0,
                    "output_tokens": float(_output_tokens(row)),
                    "stage_share": latency / max(stage_max_latency[stage_key], 1e-9),
                    "critical_agent": _bool(row.get("critical_agent")),
                    "decode_dominant": _bool(row.get("decode_dominant")),
                }
            )

    workflow_ids.discard("")
    first_config = configs[0] if configs else {}
    workload = first_config.get("workload") if isinstance(first_config.get("workload"), dict) else {}
    workflow_shape = workload.get("gsp_workflow_shape", "")
    if isinstance(workflow_shape, list):
        workflow_shape = "-".join(str(item) for item in workflow_shape)

    base_meta = {
        "evidence_tier": evidence_tier,
        "source": "sglsim_agentic_stagebarrier_critical_profile",
        "run_id": run_id,
        "git_sha": git_sha(),
        "remote_host": "",
        "slurm_job_id": "",
        "model_path": str(first_config.get("model_path") or "mock-model"),
        "workload_id": f"staged-{workflow_shape or 'unknown'}",
        "dp_size": _int(workload.get("dp_size"), 4),
        "tp_size": 1,
        "failure_rank": "",
        "repeat": f"mean-of-{len(workflow_ids)}-workflow-traces",
        "note": (
            "Derived from the online SGLSim 1-3-1-3-1 stage-barrier trace. "
            "The fan-out local agent 0 uses the long-decode workload knob and "
            "is marked Critical; other fan-out agents are Non-Critical. "
            "stage_critical_path_share is the request latency divided by the "
            "maximum latency within its workflow stage."
        ),
        "runtime_workflow_groups": len(workflow_ids),
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(str(path) for path in run_dirs),
    }

    output: list[dict[str, object]] = []
    for (stage_id, stage_width, local_idx), values in sorted(slot_values.items()):
        latencies = [float(item["latency_ms"]) for item in values]
        outputs = [float(item["output_tokens"]) for item in values]
        stage_shares = [float(item["stage_share"]) for item in values]
        critical = any(bool(item["critical_agent"]) for item in values)
        decode_dominant = any(bool(item["decode_dominant"]) for item in values)
        output.append(
            {
                "stage_slot": _slot_label(stage_id, stage_width, local_idx),
                "stage_id": stage_id,
                "stage_width": stage_width,
                "stage_local_agent_index": local_idx,
                "critical_agent": critical,
                "decode_dominant": decode_dominant,
                "mean_output_tokens": mean(outputs),
                "mean_latency_ms": mean(latencies),
                "p90_latency_ms": percentile(latencies, 90),
                "stage_critical_path_share": mean(stage_shares),
                "sample_count": len(values),
                **base_meta,
                "mean_latency_ms_std": metric_std(latencies),
                "mean_latency_ms_ci95": metric_ci95(latencies),
                "mean_output_tokens_std": metric_std(outputs),
                "mean_output_tokens_ci95": metric_ci95(outputs),
                "stage_critical_path_share_std": metric_std(stage_shares),
                "stage_critical_path_share_ci95": metric_ci95(stage_shares),
            }
        )
    output.sort(key=_slot_sort_key)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="sglsim-staged-13131-critical-profile")
    parser.add_argument(
        "--evidence-tier",
        default="sglsim-trace-derived-stagebarrier",
    )
    args = parser.parse_args()

    rows = derive_rows(
        [path.resolve() for path in args.run_dirs],
        run_id=args.run_id,
        evidence_tier=args.evidence_tier,
    )
    write_csv(args.output, rows)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(rows),
                "max_output_tokens": max(float(row["mean_output_tokens"]) for row in rows),
                "max_latency_ms": max(float(row["mean_latency_ms"]) for row in rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
