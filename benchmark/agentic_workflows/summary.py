from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


EVIDENCE_TIER = "sglsim-agentic-online-preliminary"


@dataclass(frozen=True)
class WorkflowRunSummary:
    scenario: str
    run_dir: str
    experiment_name: str
    seed: int
    dp_size: int
    workflow_count: int
    turns: int
    groups: int
    prompts_per_group: int
    system_prompt_len: int
    question_len: int
    output_len: int
    range_ratio: float
    max_concurrency: int
    critical_agent_ratio: float
    failover_replay: bool
    failover_injection: str
    failover_retry_rank_offset: int
    failover_retry_policy: str
    aegis_policy: str
    aegis_ablation: str
    aegis_gpu_budget_fraction: float
    failure_after_output_tokens: int
    backup_kv: str
    completed: int
    failed: int
    request_rows: int
    replay_rows: int
    critical_rows: int
    candidate_retry_rows: int
    cross_rank_replay_rows: int
    hard_failure_rows: int
    host_lookup_hits: int
    peer_lookup_hits: int
    restored_tokens: int
    restored_prefill_tokens: int
    recompute_tokens: int
    host_lookup_tokens: int
    peer_lookup_tokens: int
    replay_action_host_restore: int
    replay_action_peer_gpu_restore: int
    replay_action_recompute: int
    replay_action_none: int
    replay_action_other: int
    replay_tokens_host_restore: int
    replay_tokens_peer_gpu_restore: int
    replay_tokens_recompute: int
    replay_tokens_none: int
    replay_tokens_other: int
    replay_data_plane_metadata_materialization: int
    replay_data_plane_physical_tensor_copy: int
    replay_data_plane_recompute: int
    replay_data_plane_none: int
    replay_data_plane_other: int
    mean_e2e_ms: float
    p90_e2e_ms: float
    mean_restore_sleep_ms: float
    evidence_tier: str = EVIDENCE_TIER

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: object) -> int:
    return int(round(_as_float(value)))


def _restore_action_bucket(row: dict[str, str]) -> str:
    action = str(row.get("aegis_replay_restore_action") or "").strip().lower()
    if action in {"host", "host_h2d", "host_restore"}:
        return "host_restore"
    if action in {"gpu", "gpu_replica", "gpu_restore", "peer_gpu", "peer_gpu_restore"}:
        return "peer_gpu_restore"
    if action in {"recompute", "recompute_only"}:
        return "recompute"
    if action in {"", "none", "no_restore"}:
        return "none"
    return "other"


def _restore_action_tokens(row: dict[str, str], bucket: str) -> int:
    if bucket in {"host_restore", "peer_gpu_restore"}:
        return _as_int(row.get("aegis_replay_restored_tokens"))
    return _as_int(row.get("aegis_replay_restored_tokens")) + _as_int(
        row.get("aegis_replay_recompute_tokens")
    )


def _restore_data_plane_bucket(row: dict[str, str]) -> str:
    value = str(row.get("aegis_replay_restore_data_plane") or "").strip().lower()
    if value in {"metadata_materialization", "metadata"}:
        return "metadata_materialization"
    if value in {"physical_tensor_copy", "cuda_tensor_copy", "physical_kv_copy"}:
        return "physical_tensor_copy"
    if value == "recompute":
        return "recompute"
    if value in {"", "none", "no_restore"}:
        return "none"
    return "other"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_request_rows(run_dir: Path) -> list[dict[str, str]]:
    path = Path(run_dir) / "request_details.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def summarize_run_dir(run_dir: Path, *, scenario: str = "") -> WorkflowRunSummary:
    run_dir = Path(run_dir)
    metrics = _load_json_if_exists(run_dir / "metrics.json")
    config = _load_json_if_exists(run_dir / "simulator_config.json")
    workload = config.get("workload") if isinstance(config.get("workload"), dict) else {}
    scheduler = config.get("scheduler") if isinstance(config.get("scheduler"), dict) else {}
    rows = read_request_rows(run_dir)

    replay_rows = [row for row in rows if _as_bool(row.get("aegis_failover_replay"))]
    critical_rows = [row for row in rows if _as_bool(row.get("critical_agent"))]
    candidate_rows = [row for row in rows if _as_bool(row.get("aegis_candidate_retry"))]
    cross_rank_rows = [row for row in replay_rows if _as_bool(row.get("aegis_cross_rank_retry"))]
    hard_rows = [row for row in replay_rows if _as_bool(row.get("aegis_hard_failure_injected"))]
    host_rows = [row for row in replay_rows if _as_bool(row.get("aegis_host_backup_lookup_hit"))]
    peer_rows = [row for row in replay_rows if _as_bool(row.get("aegis_peer_gpu_lookup_hit"))]
    action_counts = {
        "host_restore": 0,
        "peer_gpu_restore": 0,
        "recompute": 0,
        "none": 0,
        "other": 0,
    }
    action_tokens = dict.fromkeys(action_counts, 0)
    data_plane_counts = {
        "metadata_materialization": 0,
        "physical_tensor_copy": 0,
        "recompute": 0,
        "none": 0,
        "other": 0,
    }
    for row in replay_rows:
        bucket = _restore_action_bucket(row)
        action_counts[bucket] += 1
        data_plane_counts[_restore_data_plane_bucket(row)] += 1
        action_tokens["recompute"] += _as_int(
            row.get("aegis_replay_recompute_tokens")
        )
        if bucket in {"host_restore", "peer_gpu_restore"}:
            action_tokens[bucket] += _restore_action_tokens(row, bucket)
        elif bucket != "recompute":
            action_tokens[bucket] += _restore_action_tokens(row, bucket)

    latencies_ms = [
        _as_float(row.get("client_wall_e2e_latency_s") or row.get("e2e_latency_s")) * 1000.0
        for row in rows
        if _as_float(row.get("client_wall_e2e_latency_s") or row.get("e2e_latency_s")) > 0
    ]
    restore_sleep_ms = [
        _as_float(row.get("aegis_replay_restore_sleep_s")) * 1000.0
        for row in replay_rows
        if _as_float(row.get("aegis_replay_restore_sleep_s")) > 0
    ]

    experiment = config.get("experiment") if isinstance(config.get("experiment"), dict) else {}
    return WorkflowRunSummary(
        scenario=scenario,
        run_dir=str(run_dir),
        experiment_name=str(
            config.get("experiment_name") or experiment.get("name") or ""
        ),
        seed=_as_int(workload.get("seed", 0)),
        dp_size=_as_int(scheduler.get("dp_size", 0)),
        workflow_count=_as_int(workload.get("num_requests", 0)),
        turns=_as_int(workload.get("gsp_num_turns", 0)),
        groups=_as_int(workload.get("gsp_num_groups", 0)),
        prompts_per_group=_as_int(workload.get("gsp_prompts_per_group", 0)),
        system_prompt_len=_as_int(workload.get("gsp_system_prompt_len", 0)),
        question_len=_as_int(workload.get("gsp_question_len", 0)),
        output_len=_as_int(workload.get("gsp_output_len", 0)),
        range_ratio=_as_float(workload.get("gsp_range_ratio", 0.0)),
        max_concurrency=_as_int(workload.get("max_concurrency", 0)),
        critical_agent_ratio=_as_float(workload.get("gsp_critical_agent_ratio", 0.0)),
        failover_replay=_as_bool(workload.get("gsp_failover_replay", False)),
        failover_injection=str(workload.get("gsp_failover_injection") or ""),
        failover_retry_rank_offset=_as_int(
            workload.get("gsp_failover_retry_rank_offset", 0)
        ),
        failover_retry_policy=str(workload.get("gsp_failover_retry_policy") or ""),
        aegis_policy=str(workload.get("gsp_aegis_policy") or ""),
        aegis_ablation=str(workload.get("gsp_aegis_ablation") or ""),
        aegis_gpu_budget_fraction=_as_float(
            workload.get("gsp_aegis_gpu_budget_fraction", -1.0)
        ),
        failure_after_output_tokens=_as_int(
            workload.get("gsp_failure_after_output_tokens", 0)
        ),
        backup_kv=str(workload.get("gsp_backup_kv", "")),
        completed=_as_int(metrics.get("completed", 0)),
        failed=_as_int(metrics.get("failed", 0)),
        request_rows=len(rows),
        replay_rows=len(replay_rows),
        critical_rows=len(critical_rows),
        candidate_retry_rows=len(candidate_rows),
        cross_rank_replay_rows=len(cross_rank_rows),
        hard_failure_rows=len(hard_rows),
        host_lookup_hits=len(host_rows),
        peer_lookup_hits=len(peer_rows),
        restored_tokens=sum(_as_int(row.get("aegis_replay_restored_tokens")) for row in replay_rows),
        restored_prefill_tokens=sum(
            _as_int(row.get("aegis_replay_restored_prefill_tokens"))
            for row in replay_rows
        ),
        recompute_tokens=sum(_as_int(row.get("aegis_replay_recompute_tokens")) for row in replay_rows),
        host_lookup_tokens=sum(_as_int(row.get("aegis_host_backup_lookup_tokens")) for row in replay_rows),
        peer_lookup_tokens=sum(_as_int(row.get("aegis_peer_gpu_lookup_tokens")) for row in replay_rows),
        replay_action_host_restore=action_counts["host_restore"],
        replay_action_peer_gpu_restore=action_counts["peer_gpu_restore"],
        replay_action_recompute=action_counts["recompute"],
        replay_action_none=action_counts["none"],
        replay_action_other=action_counts["other"],
        replay_tokens_host_restore=action_tokens["host_restore"],
        replay_tokens_peer_gpu_restore=action_tokens["peer_gpu_restore"],
        replay_tokens_recompute=action_tokens["recompute"],
        replay_tokens_none=action_tokens["none"],
        replay_tokens_other=action_tokens["other"],
        replay_data_plane_metadata_materialization=data_plane_counts[
            "metadata_materialization"
        ],
        replay_data_plane_physical_tensor_copy=data_plane_counts[
            "physical_tensor_copy"
        ],
        replay_data_plane_recompute=data_plane_counts["recompute"],
        replay_data_plane_none=data_plane_counts["none"],
        replay_data_plane_other=data_plane_counts["other"],
        mean_e2e_ms=mean(latencies_ms) if latencies_ms else _as_float(metrics.get("mean_e2e_latency_ms")),
        p90_e2e_ms=_percentile(latencies_ms, 0.90) if latencies_ms else _as_float(metrics.get("p90_e2e_latency_ms")),
        mean_restore_sleep_ms=mean(restore_sleep_ms) if restore_sleep_ms else 0.0,
    )


def write_summary_csv(path: Path, summaries: list[WorkflowRunSummary]) -> None:
    rows = [summary.to_row() for summary in summaries]
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
