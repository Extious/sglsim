#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
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
    / "fig_motivation_equal_size_loss.csv"
)

for path in (REPO_ROOT, REPO_ROOT / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from sglsim.srt.simulation.recovery_model import aegis_restore_overhead_s  # noqa: E402


SEGMENT_ORDER = (
    "non-critical fan-out",
    "critical long-decode fan-out",
    "critical entry singleton",
    "critical fan-in singleton",
    "critical synthesis singleton",
)


@dataclass(frozen=True)
class AgentSession:
    workflow_id: str
    agent_id: str
    stage_id: int
    stage_width: int
    stage_local_agent_index: int
    critical_agent: bool
    decode_dominant: bool
    finish_time_s: float

    @property
    def segment(self) -> str:
        if self.stage_width > 1 and not self.critical_agent:
            return "non-critical fan-out"
        if self.decode_dominant:
            return "critical long-decode fan-out"
        if self.stage_width == 1 and self.stage_id == 0:
            return "critical entry singleton"
        if self.stage_width == 1 and self.stage_id == 2:
            return "critical fan-in singleton"
        if self.stage_width == 1 and self.stage_id == 4:
            return "critical synthesis singleton"
        return "other critical"


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
            ["git", "rev-parse", "HEAD"], text=True
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


def _agent_sessions(rows: list[dict[str, str]]) -> list[AgentSession]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("workflow_id") or ""), str(row.get("agent_id") or ""))].append(row)

    sessions: list[AgentSession] = []
    for (workflow_id, agent_id), session_rows in sorted(grouped.items()):
        first = session_rows[0]
        sessions.append(
            AgentSession(
                workflow_id=workflow_id,
                agent_id=agent_id,
                stage_id=_int(first.get("stage_id"), -1),
                stage_width=max(_int(first.get("stage_width"), 1), 1),
                stage_local_agent_index=_int(first.get("stage_local_agent_index"), 0),
                critical_agent=_bool(first.get("critical_agent")),
                decode_dominant=_bool(first.get("decode_dominant")),
                finish_time_s=max(_float(row.get("finish_time_s")) for row in session_rows),
            )
        )
    return sessions


def _exposure_by_segment(
    rows: list[dict[str, str]], *, loss_delay_s: float
) -> dict[str, list[float]]:
    sessions = _agent_sessions(rows)
    stage_finish: dict[tuple[str, int], float] = {}
    for session in sessions:
        key = (session.workflow_id, session.stage_id)
        stage_finish[key] = max(stage_finish.get(key, 0.0), session.finish_time_s)

    by_segment: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        if session.segment not in SEGMENT_ORDER:
            continue
        finish_without_failure = stage_finish[(session.workflow_id, session.stage_id)]
        exposed_delay = max(
            0.0,
            session.finish_time_s + loss_delay_s - finish_without_failure,
        )
        by_segment[session.segment].append(exposed_delay / max(loss_delay_s, 1e-9))
    return by_segment


def derive_rows(
    *,
    run_dirs: list[Path],
    run_id: str,
    evidence_tier: str,
    loss_tokens: int,
    min_workflows: int,
) -> list[dict[str, object]]:
    loss_delay_s = aegis_restore_overhead_s(
        action="recompute",
        token_count=loss_tokens,
        restored_tokens=0,
    )
    all_values: dict[str, list[float]] = defaultdict(list)
    workflow_ids: set[str] = set()
    config_payloads = [_load_config(run_dir) for run_dir in run_dirs]
    for run_dir in run_dirs:
        rows = _original_rows(run_dir)
        workflow_ids.update(str(row.get("workflow_id") or "") for row in rows)
        by_segment = _exposure_by_segment(rows, loss_delay_s=loss_delay_s)
        for segment, values in by_segment.items():
            all_values[segment].extend(values)

    workflow_ids.discard("")
    if len(workflow_ids) < min_workflows:
        raise ValueError(f"only {len(workflow_ids)} workflow traces found; need {min_workflows}")

    first_config = config_payloads[0] if config_payloads else {}
    workload = first_config.get("workload") if isinstance(first_config.get("workload"), dict) else {}
    workflow_shape = workload.get("gsp_workflow_shape", "")
    if isinstance(workflow_shape, list):
        workflow_shape = "-".join(str(item) for item in workflow_shape)

    base_meta = {
        "evidence_tier": evidence_tier,
        "source": "sglsim_agentic_stagebarrier_equal_size_loss_counterfactual",
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
            "Equal-size KV loss counterfactual derived from online SGLSim "
            "1-3-1-3-1 traces with per-workflow stage barriers. Each agent "
            "session receives the same recompute delay for the lost segment; "
            "the plotted value is the fraction of that delay exposed to WCT "
            "after fan-out slack is applied."
        ),
        "loss_tokens": int(loss_tokens),
        "loss_delay_ms": loss_delay_s * 1000.0,
        "runtime_workflow_groups": len(workflow_ids),
        "repeat_count": len(run_dirs),
        "repeat_run_ids": ";".join(str(path) for path in run_dirs),
    }

    rows: list[dict[str, object]] = []
    for segment in SEGMENT_ORDER:
        values = all_values.get(segment, [])
        if not values:
            continue
        rows.append(
            {
                "segment": segment,
                "delta_wct_norm": mean(values),
                **base_meta,
                "delta_wct_norm_std": metric_std(values),
                "delta_wct_norm_ci95": metric_ci95(values),
                "runtime_min_delta_wct_norm": min(values),
                "runtime_max_delta_wct_norm": max(values),
                "sample_count": len(values),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Fig. 1 equal-size KV loss motivation rows from online "
            "SGLSim stage-barrier workflow traces."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--loss-tokens", type=int, default=512)
    parser.add_argument("--min-workflows", type=int, default=4)
    parser.add_argument("--evidence-tier", default="sglsim-trace-derived-stagebarrier")
    args = parser.parse_args()

    rows = derive_rows(
        run_dirs=args.run_dir,
        run_id=args.run_id,
        evidence_tier=args.evidence_tier,
        loss_tokens=args.loss_tokens,
        min_workflows=args.min_workflows,
    )
    write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_csv": str(args.output_csv),
                "rows": len(rows),
                "loss_tokens": args.loss_tokens,
                "segments": {
                    str(row["segment"]): row["delta_wct_norm"] for row in rows
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
