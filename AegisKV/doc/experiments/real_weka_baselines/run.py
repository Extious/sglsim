#!/usr/bin/env python3
"""Capacity-matched baseline replay on the complete real-Weka trace.

The current AegisKV result is imported from the independently audited full-
design replay.  Every additional protection baseline is replayed event by event
over the identical Weka layout.  Existing failure rows are used only as the
frozen paired fault schedule and Algorithm-1 Slack ticket values; they do not
provide baseline protection state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AegisKV.experiments.real_weka_full_design.run import (
    ReplayLayout,
    ReplayRequest,
    V100_HARDWARE_LABEL,
    V100_PROFILE_SEMANTICS,
    V100_RUNTIME_CONSUMPTION_CONTRACT,
    _scale_workflows,
    _verify_artifact,
    _recompute_decision,
    build_layout,
    preflight_v100_config,
)
from benchmark.agentic_workflows.weka_full_trace import (
    dataset_manifest,
    default_full_trace_path,
    load_weka_workflows,
    sha256_file,
)
from sglsim.srt.simulation.aegis_baselines import (
    BaselinePolicy,
    BaselineRequestProfile,
    LumenBaselineRuntime,
    LumenConfig,
    ProtectionBaselineConfig,
    ProtectionBaselineRuntime,
    aggregate_snapshots,
    bounded_state_restore,
    cacheflow_3d_restore,
    restore_or_recompute_split,
)
from sglsim.srt.simulation.aegis_independent_tiers import (
    DESIGN_SEMANTICS,
    RecoveryDecision,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
BASELINE_LABELS = {
    BaselinePolicy.DEJAVU_NEIGHBOR: "dejavu_neighbor",
    BaselinePolicy.RAIDSERVE_HOST: "raidserve_host",
    BaselinePolicy.KEVLARFLOW_RING: "kevlarflow_ring",
    BaselinePolicy.GHOSTSERVE_PARITY: "ghostserve_parity_tp8",
}
PRIMARY_POLICIES = (
    "recompute_fcfs",
    "dejavu_neighbor",
    "raidserve_host",
    "kevlarflow_ring",
    "lumen_load_aware",
    "aegiskv_latest",
)
ALL_E2E_POLICIES = (
    *PRIMARY_POLICIES[:-1],
    "recompute_load_aware",
    "ghostserve_parity_tp8",
    "aegiskv_latest",
)


@dataclass(frozen=True)
class AnchorFailure:
    event_id: str
    failure_time_ms: float
    request_id: str
    workflow_id: str
    stage_id: int
    agent_id: str
    source_worker: str
    activated_target: str
    produced_tokens: int
    predicted_remaining_slack_ms: float
    actual_join_slack_ms: float
    target_rotated_before_source: bool


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(path: str, *, relative_to: Path) -> Path:
    parsed = Path(path)
    return parsed.resolve() if parsed.is_absolute() else (relative_to / parsed).resolve()


def _verify_anchor_artifacts(
    anchor_dir: Path,
    artifact_hashes_record: dict[str, Any],
) -> dict[str, str]:
    hashes_path = _verify_artifact(
        artifact_hashes_record, label="V100 full anchor artifact hashes"
    )
    if hashes_path != anchor_dir / "artifact_hashes.json":
        raise ValueError("V100 full anchor artifact hash file is outside anchor dir")
    hashes = _load_json(hashes_path)
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("V100 full anchor artifact hash set is empty")
    required = {
        "algorithm1_metrics.json",
        "audit.json",
        "calibration_provenance.json",
        "config_resolved.json",
        "dataset_manifest.json",
        "failure_requests.csv",
        "failure_stages.csv",
        "fault_manifest.json",
        "runtime_aggregate.json",
        "workload_projection.json",
    }
    if not required.issubset(set(hashes)):
        raise ValueError("V100 full anchor artifact set is incomplete")
    normalized: dict[str, str] = {}
    for name, expected_sha in hashes.items():
        if Path(str(name)).name != str(name) or name == "artifact_hashes.json":
            raise ValueError("V100 full anchor artifact name is unsafe")
        path = anchor_dir / str(name)
        if not path.is_file():
            raise ValueError(f"V100 full anchor artifact is missing: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != str(expected_sha):
            raise ValueError(f"V100 full anchor artifact sha256 mismatch: {name}")
        normalized[str(name)] = actual_sha
    return normalized


def preflight_baseline_suite(
    suite_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    """Bind a baseline run to one immutable V100 config/profile/anchor chain."""

    suite_path = suite_path.resolve()
    suite = _load_json(suite_path)
    if suite.get("status") != "READY":
        raise ValueError(
            "baseline suite is not READY: "
            f"{suite.get('status', 'missing')}"
        )
    if not (
        suite.get("schema_version") == 2
        and suite.get("evidence_class") == "sglsim_v100_baseline_suite_binding"
        and suite.get("design_semantics") == DESIGN_SEMANTICS
        and suite.get("hardware_label") == V100_HARDWARE_LABEL
        and suite.get("profile_semantics") == V100_PROFILE_SEMANTICS
        and suite.get("paper_ready") is False
        and suite.get("formal_ready") is False
    ):
        raise ValueError("baseline V100 suite binding contract did not pass")
    config_record = suite.get("base_experiment_config")
    import_record = suite.get("base_import_manifest")
    anchor = suite.get("anchor_result")
    if not all(isinstance(value, dict) for value in (config_record, import_record, anchor)):
        raise ValueError("baseline V100 suite artifact bindings are incomplete")
    base_config_path = _verify_artifact(
        config_record, label="baseline V100 experiment config"
    )
    import_manifest_path = _verify_artifact(
        import_record, label="baseline V100 import manifest"
    )
    (
        base_config,
        runtime_config,
        calibration,
        projection,
        full_binding,
    ) = preflight_v100_config(base_config_path, import_manifest_path)
    anchor_dir = Path(str(anchor.get("path", ""))).resolve()
    expected_anchor_name = (
        f"full_393_workflows_v100_{full_binding['physical_profile_sha256'][:12]}"
    )
    if not anchor_dir.is_dir() or anchor_dir.name != expected_anchor_name:
        raise ValueError("baseline V100 full anchor path is invalid")
    artifact_hashes_record = anchor.get("artifact_hashes")
    if not isinstance(artifact_hashes_record, dict):
        raise ValueError("baseline V100 full anchor lacks artifact hashes")
    _verify_anchor_artifacts(anchor_dir, artifact_hashes_record)
    anchor_resolved = _load_json(anchor_dir / "config_resolved.json")
    anchor_audit = _load_json(anchor_dir / "audit.json")
    anchor_calibration = _load_json(anchor_dir / "calibration_provenance.json")
    anchor_dataset = _load_json(anchor_dir / "dataset_manifest.json")
    expected_dataset = base_config["dataset"]["expected_full_profile"]
    if not (
        anchor.get("config_sha256") == full_binding["config"]["sha256"]
        and anchor.get("import_manifest_sha256")
        == full_binding["import_manifest"]["sha256"]
        and anchor.get("physical_profile_sha256")
        == full_binding["physical_profile_sha256"]
        and anchor.get("source_manifest_sha256")
        == full_binding["source_manifest_sha256"]
        and anchor.get("config_manifest_sha256")
        == full_binding["config_manifest_sha256"]
        and anchor.get("config_verification_sha256")
        == full_binding["config_verification_sha256"]
        and anchor_resolved.get("input_binding") == full_binding
        and Path(str(anchor_resolved.get("output_dir_resolved", ""))).resolve()
        == anchor_dir
        and anchor_audit.get("status") == "PASS"
        and anchor_audit.get("paper_ready") is False
        and anchor_calibration.get("calibration_semantics")
        == V100_PROFILE_SEMANTICS
        and anchor_calibration.get("runtime_consumption_contract")
        == V100_RUNTIME_CONSUMPTION_CONTRACT
        and anchor_calibration.get("aimd_pressure", {}).get("calibrated")
        is True
        and anchor_calibration.get("aimd_pressure", {}).get(
            "sglsim_quantitative_aimd_pressure_authorized"
        )
        is True
        and anchor_calibration.get("physical_profile", {}).get("sha256")
        == full_binding["physical_profile_sha256"]
        and anchor_calibration.get("source_manifest", {}).get("sha256")
        == full_binding["source_manifest_sha256"]
        and anchor_calibration.get("config_manifest", {}).get("sha256")
        == full_binding["config_manifest_sha256"]
        and anchor_calibration.get("config_verification", {}).get("sha256")
        == full_binding["config_verification_sha256"]
        and all(
            int(anchor_dataset[key]) == int(expected_dataset[key])
            for key in (
                "workflow_count",
                "request_count",
                "input_tokens",
                "output_tokens",
            )
        )
    ):
        raise ValueError("baseline V100 anchor identity contract did not pass")
    binding = {
        **full_binding,
        "suite_config": {
            "path": str(suite_path),
            "sha256": sha256_file(suite_path),
        },
        "anchor_result": {
            "path": str(anchor_dir),
            "artifact_hashes_sha256": artifact_hashes_record["sha256"],
        },
    }
    return (
        suite,
        base_config,
        runtime_config,
        calibration,
        projection,
        anchor_dir,
        binding,
    )


def _derived_baseline_output_dir(
    requested: Path | None, *, profile_sha256: str, limit_workflows: int | None
) -> Path:
    prefix = (
        "full_393_workflows_baselines"
        if limit_workflows is None
        else f"smoke_{int(limit_workflows)}_workflows_baselines"
    )
    expected_name = f"{prefix}_v100_{profile_sha256[:12]}"
    output = (
        requested.resolve()
        if requested is not None
        else (EXPERIMENT_ROOT / "results" / expected_name).resolve()
    )
    if output.name != expected_name:
        raise ValueError(
            "baseline output directory must be derived from the physical "
            f"profile hash and named {expected_name!r}"
        )
    if output.exists():
        raise FileExistsError(f"baseline replay output is immutable: {output}")
    return output


def _read_anchor_failures(path: Path) -> tuple[dict[str, list[AnchorFailure]], list[dict[str, Any]]]:
    events: dict[str, list[AnchorFailure]] = defaultdict(list)
    aegis_rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            policy = row["policy"]
            if policy == "aegiskv_latest":
                aegis_rows.append(dict(row))
            if policy != "recompute_fcfs":
                continue
            anchor = AnchorFailure(
                event_id=row["event_id"],
                failure_time_ms=float(row["failure_time_ms"]),
                request_id=row["request_id"],
                workflow_id=row["workflow_id"],
                stage_id=int(row["stage_id"]),
                agent_id=row["agent_id"],
                source_worker=row["source_worker"],
                activated_target=row["activated_target"],
                produced_tokens=int(row["produced_tokens"]),
                predicted_remaining_slack_ms=float(row["predicted_remaining_slack_ms"]),
                actual_join_slack_ms=float(row["actual_join_slack_ms"]),
                target_rotated_before_source=row["target_rotated_before_source"] == "True",
            )
            events[anchor.event_id].append(anchor)
    return dict(events), aegis_rows


def _read_anchor_stages(path: Path) -> tuple[dict[tuple[str, str, int], float], list[dict[str, Any]]]:
    base: dict[tuple[str, str, int], float] = {}
    aegis_rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["policy"] == "aegiskv_latest":
                aegis_rows.append(dict(row))
            if row["policy"] == "recompute_fcfs":
                key = (row["event_id"], row["workflow_id"], int(row["stage_id"]))
                base[key] = float(row["base_stage_finish_ms"])
    return base, aegis_rows


def _baseline_configs(runtime_config: Any, suite: dict[str, Any]) -> dict[BaselinePolicy, ProtectionBaselineConfig]:
    ghost = suite["ghostserve"]
    parity_fraction = float(ghost["parity_shards"]) / float(ghost["data_shards"])
    encode_ms_per_token = runtime_config.kv_bytes_per_token / runtime_config.hbm_copy.bandwidth_bytes_per_ms * parity_fraction
    # Encoding is healthy-path protection traffic.  Decoding is failure-time
    # recovery and therefore consumes the Peer-HBM recovery rate only.
    decode_ms_per_token = runtime_config.hbm_load.ms_per_token / float(ghost["data_shards"])
    output: dict[BaselinePolicy, ProtectionBaselineConfig] = {}
    for policy in BaselinePolicy:
        output[policy] = ProtectionBaselineConfig(
            policy=policy,
            hbm_capacity_bytes=runtime_config.hbm_capacity_bytes,
            dram_capacity_bytes=runtime_config.dram_capacity_bytes,
            kv_bytes_per_token=runtime_config.kv_bytes_per_token,
            hbm_copy=runtime_config.hbm_copy,
            dram_copy=runtime_config.dram_copy,
            recompute=runtime_config.recompute,
            hbm_load=runtime_config.hbm_load,
            dram_load=runtime_config.dram_load,
            pp_size=runtime_config.pp_size,
            tp_size=runtime_config.tp_size,
            parity_data_shards=int(ghost["data_shards"]),
            parity_shards=int(ghost["parity_shards"]),
            parity_encode_ms_per_token=encode_ms_per_token,
            parity_decode_ms_per_token=decode_ms_per_token,
            event_log_limit=32,
        )
    return output


def _runtime_grid(
    configs: dict[BaselinePolicy, ProtectionBaselineConfig], dp_size: int
) -> dict[BaselinePolicy, dict[int, ProtectionBaselineRuntime]]:
    return {
        policy: {
            rank: ProtectionBaselineRuntime(config, target_id=f"target-{rank}")
            for rank in range(dp_size)
        }
        for policy, config in configs.items()
    }


def _profile(meta: ReplayRequest, *, total_tokens: int) -> BaselineRequestProfile:
    return BaselineRequestProfile(
        request_id=meta.row.rid,
        source_worker=meta.source_worker,
        source_failure_domain=f"source-domain-{meta.source_worker}",
        target_worker=meta.current_target,
        target_failure_domain=f"target-domain-{meta.current_target}",
        route_version=1,
        total_tokens=total_tokens,
        slack_ms=0.0,
    )


def _evaluate_policy_event(
    *,
    policy: str,
    anchors: Sequence[AnchorFailure],
    layout: ReplayLayout,
    decisions: Sequence[RecoveryDecision],
    base_stage_finishes: dict[tuple[str, str, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(anchors) != len(decisions):
        raise ValueError("anchor/decision cardinality mismatch")
    pairs = list(zip(anchors, decisions, strict=True))
    by_target: dict[str, list[tuple[AnchorFailure, RecoveryDecision]]] = defaultdict(list)
    for anchor, decision in pairs:
        by_target[decision.activated_target].append((anchor, decision))
    completion: dict[str, tuple[float, int, float]] = {}
    for target, target_pairs in sorted(by_target.items()):
        ordered = sorted(
            target_pairs,
            key=lambda pair: (
                layout.by_id[pair[0].request_id].global_start_ms,
                pair[0].request_id,
            ),
        )
        cumulative = 0.0
        for position, (anchor, decision) in enumerate(ordered):
            queue_start = cumulative
            cumulative += float(decision.recovery_ms)
            completion[anchor.request_id] = (cumulative, position, queue_start)
    failure_rows: list[dict[str, Any]] = []
    for anchor, decision in pairs:
        queue_completion, position, queue_start = completion[anchor.request_id]
        failure_rows.append(
            {
                "event_id": anchor.event_id,
                "failure_time_ms": anchor.failure_time_ms,
                "policy": policy,
                "ordering": "fcfs",
                "request_id": anchor.request_id,
                "workflow_id": anchor.workflow_id,
                "stage_id": anchor.stage_id,
                "agent_id": anchor.agent_id,
                "source_worker": anchor.source_worker,
                "activated_target": decision.activated_target,
                "queue_position": position,
                "queue_start_ms": queue_start,
                "queue_completion_delay_ms": queue_completion,
                "action": decision.action,
                "produced_tokens": anchor.produced_tokens,
                "restored_tokens": decision.restored_tokens,
                "recompute_tokens": decision.recompute_tokens,
                "recovery_service_ms": decision.recovery_ms,
                "predicted_remaining_slack_ms": anchor.predicted_remaining_slack_ms,
                "predicted_visible_harm_ms": max(queue_completion - anchor.predicted_remaining_slack_ms, 0.0),
                "actual_join_slack_ms": anchor.actual_join_slack_ms,
                "actual_slack_visible_harm_ms": max(queue_completion - anchor.actual_join_slack_ms, 0.0),
                "target_rotated_before_source": anchor.target_rotated_before_source,
                "counterfactual_non_destructive": True,
                "fidelity": (
                    "dedicated_tp8_component_control"
                    if policy == "ghostserve_parity_tp8"
                    else "capacity_matched_component_adaptation"
                ),
            }
        )
    by_stage: dict[tuple[str, int], list[AnchorFailure]] = defaultdict(list)
    for anchor in anchors:
        by_stage[(anchor.workflow_id, anchor.stage_id)].append(anchor)
    stage_rows: list[dict[str, Any]] = []
    for (workflow_id, stage_id), rows in sorted(by_stage.items()):
        base = base_stage_finishes[(rows[0].event_id, workflow_id, stage_id)]
        delay_by_agent: dict[str, float] = defaultdict(float)
        branch_finish: dict[str, float] = {}
        for anchor in rows:
            delay_by_agent[anchor.agent_id] = max(
                delay_by_agent[anchor.agent_id], completion[anchor.request_id][0]
            )
            branch_finish[anchor.agent_id] = min(
                branch_finish.get(anchor.agent_id, math.inf),
                base - anchor.actual_join_slack_ms,
            )
        modeled = max(
            [base]
            + [
                branch_finish[agent_id] + delay
                for agent_id, delay in delay_by_agent.items()
            ]
        )
        stage_rows.append(
            {
                "event_id": rows[0].event_id,
                "policy": policy,
                "workflow_id": workflow_id,
                "stage_id": stage_id,
                "affected_requests": len(rows),
                "affected_agents": len(delay_by_agent),
                "base_stage_finish_ms": base,
                "modeled_stage_finish_ms": modeled,
                "join_barrier_harm_ms": max(modeled - base, 0.0),
            }
        )
    return failure_rows, stage_rows


def replay_baselines(
    *,
    layout: ReplayLayout,
    anchor_events: dict[str, list[AnchorFailure]],
    anchor_fault_manifest: Sequence[dict[str, Any]],
    base_stage_finishes: dict[tuple[str, str, int], float],
    base_config: dict[str, Any],
    suite: dict[str, Any],
    runtime_config: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dp_size = int(base_config["cluster"]["dp_size"])
    prefix_q = int(base_config["chunking"]["prefix_tokens"])
    decode_q = int(base_config["chunking"]["decode_tokens"])
    configs = _baseline_configs(runtime_config, suite)
    runtimes = _runtime_grid(configs, dp_size)
    lumen_cfg = suite["lumen"]
    lumen = LumenBaselineRuntime(
        LumenConfig(
            holder_count=dp_size,
            bytes_per_holder=runtime_config.dram_capacity_bytes,
            kv_bytes_per_token=runtime_config.kv_bytes_per_token,
            copy_path=runtime_config.dram_copy,
            recompute=runtime_config.recompute,
            host_load=runtime_config.dram_load,
            placement_lambda=float(lumen_cfg["placement_lambda"]),
            dispatch_overload_factor=float(lumen_cfg["dispatch_overload_factor"]),
        )
    )
    routing_only = LumenBaselineRuntime(lumen.config)
    target_failures = {
        int(event["rank"]): float(event["time_ms"])
        for event in anchor_fault_manifest
        if event["kind"] == "target_failure"
    }
    source_failures = {
        int(event["rank"]): float(event["time_ms"])
        for event in anchor_fault_manifest
        if event["kind"] == "source_failure_snapshot"
        and event.get("schedule_kind") == "source_failure"
    }
    events: list[tuple[float, int, int, str, Any]] = []
    priority = {"target_failure": 0, "source_snapshot": 1, "start": 2, "seal": 3, "finish": 4}
    sequence = 0

    def append(at_ms: float, kind: str, payload: Any) -> None:
        nonlocal sequence
        sequence += 1
        events.append((float(at_ms), priority[kind], sequence, kind, payload))

    for meta in layout.requests:
        append(meta.global_start_ms, "start", meta.row.rid)
        append(meta.global_finish_ms, "finish", meta.row.rid)
    for rank, at_ms in target_failures.items():
        append(at_ms, "target_failure", rank)
    fault_event_by_id = {
        event["event_id"]: event
        for event in anchor_fault_manifest
        if event["kind"] == "source_failure_snapshot"
    }
    for event_id, anchors in anchor_events.items():
        if not anchors:
            continue
        append(fault_event_by_id[event_id]["time_ms"], "source_snapshot", event_id)
    heapq.heapify(events)
    failure_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    admitted: dict[str, Counter[str]] = {
        label: Counter() for label in BASELINE_LABELS.values()
    }
    sealed = 0
    started = time.monotonic()
    while events:
        at_ms, _priority, _seq, kind, payload = heapq.heappop(events)
        if kind == "start":
            meta = layout.by_id[payload]
            source_generation = 0 if at_ms < source_failures.get(meta.source_rank, math.inf) else 1
            target_generation = 0 if at_ms < target_failures.get(meta.target_rank, math.inf) else 1
            meta.source_worker = f"source-{meta.source_rank}-g{source_generation}"
            meta.current_target = f"target-{meta.target_rank}-g{target_generation}"
            meta.started = True
            profile = _profile(
                meta,
                total_tokens=meta.row.prompt_tokens + meta.row.output_tokens,
            )
            for policy in BaselinePolicy:
                runtimes[policy][meta.target_rank].register_request(profile, time_ms=at_ms)
            lumen.register_request(profile, source_rank=meta.source_rank, time_ms=at_ms)
            if meta.chunk_count:
                sequence += 1
                heapq.heappush(
                    events,
                    (
                        meta.seal_time_ms(0, prefix_q=prefix_q, decode_q=decode_q),
                        priority["seal"],
                        sequence,
                        "seal",
                        (meta.row.rid, 0),
                    ),
                )
        elif kind == "seal":
            request_id, ordinal = payload
            meta = layout.by_id[request_id]
            chunk = meta.chunk(ordinal, prefix_q=prefix_q, decode_q=decode_q)
            for policy, label in BASELINE_LABELS.items():
                decision = runtimes[policy][meta.target_rank].seal_chunk(
                    request_id, chunk, time_ms=at_ms
                )
                admitted[label][decision.reason] += 1
            lumen.seal_chunk(request_id, chunk, time_ms=at_ms)
            sealed += 1
            next_ordinal = ordinal + 1
            if next_ordinal < meta.chunk_count:
                sequence += 1
                heapq.heappush(
                    events,
                    (
                        meta.seal_time_ms(next_ordinal, prefix_q=prefix_q, decode_q=decode_q),
                        priority["seal"],
                        sequence,
                        "seal",
                        (request_id, next_ordinal),
                    ),
                )
            every = int(suite["audit"]["progress_every_chunks"])
            if sealed % every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"baseline progress sealed={sealed}/{layout.intended_chunk_count} "
                    f"rate={sealed / elapsed:.0f} chunks/s",
                    flush=True,
                )
        elif kind == "finish":
            meta = layout.by_id[payload]
            for policy in BaselinePolicy:
                runtime = runtimes[policy][meta.target_rank]
                if runtime.has_request(meta.row.rid):
                    runtime.finish_request(meta.row.rid, time_ms=at_ms)
            if lumen.has_request(meta.row.rid):
                lumen.finish_request(meta.row.rid, time_ms=at_ms)
            meta.finished = True
        elif kind == "target_failure":
            rank = int(payload)
            failed = f"target-{rank}-g0"
            replacement = f"target-{rank}-g1"
            for meta in layout.by_target_rank.get(rank, ()):
                if not meta.started or meta.finished or meta.current_target != failed:
                    continue
                for policy in BaselinePolicy:
                    runtime = runtimes[policy][rank]
                    if runtime.has_request(meta.row.rid):
                        runtime.rotate_target(
                            meta.row.rid,
                            replacement_target=replacement,
                            replacement_failure_domain=f"target-domain-{rank}-g1",
                            time_ms=at_ms,
                        )
                meta.current_target = replacement
                meta.rotated_before_source = True
        elif kind == "source_snapshot":
            anchors = anchor_events[str(payload)]
            decisions_by_policy: dict[str, list[RecoveryDecision]] = defaultdict(list)
            lumen_decisions: list[RecoveryDecision] = []
            for anchor in anchors:
                meta = layout.by_id[anchor.request_id]
                decisions_by_policy["recompute_fcfs"].append(
                    _recompute_decision(
                        meta,
                        anchor.produced_tokens,
                        anchor.predicted_remaining_slack_ms,
                        runtime_config.recompute,
                    )
                )
                for policy, label in BASELINE_LABELS.items():
                    decisions_by_policy[label].append(
                        runtimes[policy][meta.target_rank].estimate_source_failure(
                            anchor.request_id,
                            produced_tokens=anchor.produced_tokens,
                            remaining_slack_ms=anchor.predicted_remaining_slack_ms,
                            activated_target=meta.current_target,
                            time_ms=at_ms,
                        )
                    )
                lumen_decisions.append(
                    lumen.estimate_source_failure(
                        anchor.request_id,
                        produced_tokens=anchor.produced_tokens,
                        remaining_slack_ms=anchor.predicted_remaining_slack_ms,
                        time_ms=at_ms,
                    )
                )
            decisions_by_policy["lumen_load_aware"] = list(
                lumen.dispatch(lumen_decisions, target_count=dp_size)
            )
            decisions_by_policy["recompute_load_aware"] = list(
                routing_only.dispatch(
                    [
                        replace(decision, activated_target="unplaced")
                        for decision in decisions_by_policy["recompute_fcfs"]
                    ],
                    target_count=dp_size,
                )
            )
            for policy, decisions in decisions_by_policy.items():
                event_failures, event_stages = _evaluate_policy_event(
                    policy=policy,
                    anchors=anchors,
                    layout=layout,
                    decisions=decisions,
                    base_stage_finishes=base_stage_finishes,
                )
                failure_rows.extend(event_failures)
                stage_rows.extend(event_stages)
    for by_rank in runtimes.values():
        for runtime in by_rank.values():
            runtime.flush()
            runtime.validate_invariants()
    lumen.flush()
    runtime_summary = {
        BASELINE_LABELS[policy]: aggregate_snapshots(by_rank.values())
        for policy, by_rank in runtimes.items()
    }
    runtime_summary["lumen_load_aware"] = lumen.snapshot()
    runtime_summary["recompute_load_aware"] = routing_only.snapshot()
    return failure_rows, stage_rows, {
        "sealed_chunk_events": sealed,
        "intended_chunk_events": layout.intended_chunk_count,
        "event_loop_wall_seconds": time.monotonic() - started,
        "placement_counts": {
            name: dict(sorted(counts.items())) for name, counts in admitted.items()
        },
        "runtime": runtime_summary,
    }


def _normalize_anchor_rows(rows: Iterable[dict[str, Any]], *, policy: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    numeric = {
        "failure_time_ms",
        "queue_start_ms",
        "queue_completion_delay_ms",
        "recovery_service_ms",
        "predicted_remaining_slack_ms",
        "predicted_visible_harm_ms",
        "actual_join_slack_ms",
        "actual_slack_visible_harm_ms",
    }
    integers = {"stage_id", "queue_position", "produced_tokens", "restored_tokens", "recompute_tokens"}
    for source in rows:
        row = dict(source)
        row["policy"] = policy
        for field in numeric:
            if field in row:
                row[field] = float(row[field])
        for field in integers:
            if field in row:
                row[field] = int(row[field])
        row["fidelity"] = "native_latest_aegiskv_design"
        output.append(row)
    return output


def _normalize_anchor_stage_rows(rows: Iterable[dict[str, Any]], *, policy: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["policy"] = policy
        for field in ("stage_id", "affected_requests", "affected_agents"):
            row[field] = int(row[field])
        for field in ("base_stage_finish_ms", "modeled_stage_finish_ms", "join_barrier_harm_ms"):
            row[field] = float(row[field])
        output.append(row)
    return output


def summarize(
    failure_rows: Sequence[dict[str, Any]], stage_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    policies = sorted({str(row["policy"]) for row in failure_rows})
    for policy in policies:
        failures = [row for row in failure_rows if row["policy"] == policy]
        stages = [row for row in stage_rows if row["policy"] == policy]
        service = np.asarray([float(row["recovery_service_ms"]) for row in failures])
        queue = np.asarray([float(row["queue_completion_delay_ms"]) for row in failures])
        harm = np.asarray([float(row["join_barrier_harm_ms"]) for row in stages])
        output[policy] = {
            "failure_request_count": len(failures),
            "failure_stage_count": len(stages),
            "action_counts": dict(sorted(Counter(str(row["action"]) for row in failures).items())),
            "mean_recovery_service_ms": float(service.mean()) if len(service) else 0.0,
            "p95_recovery_service_ms": float(np.quantile(service, 0.95)) if len(service) else 0.0,
            "mean_queue_completion_delay_ms": float(queue.mean()) if len(queue) else 0.0,
            "p95_queue_completion_delay_ms": float(np.quantile(queue, 0.95)) if len(queue) else 0.0,
            "total_join_barrier_harm_ms": float(harm.sum()),
            "mean_join_barrier_harm_ms": float(harm.mean()) if len(harm) else 0.0,
            "p95_join_barrier_harm_ms": float(np.quantile(harm, 0.95)) if len(harm) else 0.0,
        }
    return output


def paired_bootstrap(
    stage_rows: Sequence[dict[str, Any]], *, resamples: int, seed: int
) -> dict[str, Any]:
    workflows = sorted({str(row["workflow_id"]) for row in stage_rows})
    index = {workflow_id: offset for offset, workflow_id in enumerate(workflows)}
    policies = sorted({str(row["policy"]) for row in stage_rows})
    values = {policy: np.zeros(len(workflows), dtype=float) for policy in policies}
    for row in stage_rows:
        values[str(row["policy"])][index[str(row["workflow_id"])]] += float(row["join_barrier_harm_ms"])
    treatment = values["aegiskv_latest"]
    rng = np.random.default_rng(int(seed))
    comparisons: dict[str, Any] = {}
    for policy, baseline in values.items():
        if policy == "aegiskv_latest":
            continue
        base_mean = float(baseline.mean())
        treatment_mean = float(treatment.mean())
        observed = 100.0 * (base_mean - treatment_mean) / base_mean if base_mean > 0 else 0.0
        samples = np.empty(int(resamples), dtype=float)
        for sample in range(int(resamples)):
            picked = rng.integers(0, len(workflows), size=len(workflows))
            base = float(baseline[picked].mean())
            aegis = float(treatment[picked].mean())
            samples[sample] = 100.0 * (base - aegis) / base if base > 0 else 0.0
        comparisons[policy] = {
            "metric": "workflow_aggregated_join_barrier_harm",
            "paired_workflow_count": len(workflows),
            "reduction_percent": observed,
            "bootstrap_95ci_percent": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        }
    return {
        "bootstrap_unit": "faulted_workflow",
        "bootstrap_resamples": int(resamples),
        "comparisons": comparisons,
    }


def restore_controls(
    failure_rows: Sequence[dict[str, Any]], runtime_config: Any, suite: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raid = [row for row in failure_rows if row["policy"] == "raidserve_host"]
    control = suite["restore_controls"]
    rows: list[dict[str, Any]] = []
    for row in raid:
        total = int(row["produced_tokens"])
        external = int(row["restored_tokens"])
        mandatory = total - external
        plans = [
            restore_or_recompute_split(
                total_tokens=total,
                mandatory_recompute_tokens=mandatory,
                io_ms_per_token=runtime_config.dram_load.ms_per_token,
                recompute_ms_per_token=runtime_config.recompute.ms_per_token,
                kv_bytes_per_token=runtime_config.kv_bytes_per_token,
            ),
            cacheflow_3d_restore(
                total_tokens=total,
                externally_available_tokens=external,
                io_ms_per_token=runtime_config.dram_load.ms_per_token,
                recompute_ms_per_token=runtime_config.recompute.ms_per_token,
                kv_bytes_per_token=runtime_config.kv_bytes_per_token,
                layers=int(control["cacheflow_layers"]),
                gpu_lanes=int(control["cacheflow_gpu_lanes"]),
                token_tiles=int(control["cacheflow_token_tiles"]),
            ),
            bounded_state_restore(
                total_tokens=total,
                externally_available_tokens=external,
                chunk_tokens=int(control["bsr_chunk_tokens"]),
                window_chunks=int(control["bsr_window_chunks"]),
                kv_bytes_per_token=runtime_config.kv_bytes_per_token,
                io_timing=runtime_config.dram_load,
                recompute_timing=runtime_config.recompute,
            ),
        ]
        for plan in plans:
            rows.append(
                {
                    "event_id": row["event_id"],
                    "request_id": row["request_id"],
                    "workflow_id": row["workflow_id"],
                    **asdict(plan),
                }
            )
    summary: dict[str, Any] = {}
    for planner in sorted({row["planner"] for row in rows}):
        selected = [row for row in rows if row["planner"] == planner]
        completion = np.asarray([float(row["completion_ms"]) for row in selected])
        staging = np.asarray([float(row["peak_staging_bytes"]) for row in selected])
        summary[planner] = {
            "request_count": len(selected),
            "mean_completion_ms": float(completion.mean()) if len(completion) else 0.0,
            "p95_completion_ms": float(np.quantile(completion, 0.95)) if len(completion) else 0.0,
            "mean_peak_staging_gib": float(staging.mean() / 2**30) if len(staging) else 0.0,
            "max_peak_staging_gib": float(staging.max() / 2**30) if len(staging) else 0.0,
        }
    return rows, summary


def audit(
    *,
    manifest: dict[str, Any],
    anchor_manifest: dict[str, Any],
    layout: ReplayLayout,
    replay_meta: dict[str, Any],
    failure_rows: Sequence[dict[str, Any]],
    stage_rows: Sequence[dict[str, Any]],
    calibration: dict[str, Any],
    input_binding: dict[str, Any],
) -> dict[str, Any]:
    policies = sorted({row["policy"] for row in failure_rows})
    request_keys = {
        policy: {
            (row["event_id"], row["request_id"])
            for row in failure_rows
            if row["policy"] == policy
        }
        for policy in policies
    }
    stage_keys = {
        policy: {
            (row["event_id"], row["workflow_id"], int(row["stage_id"]))
            for row in stage_rows
            if row["policy"] == policy
        }
        for policy in policies
    }
    expected_request_keys = request_keys["aegiskv_latest"]
    expected_stage_keys = stage_keys["aegiskv_latest"]
    gates = {
        "same_v100_profile_config_source_anchor_bound": bool(
            input_binding.get("hardware_label") == V100_HARDWARE_LABEL
            and input_binding.get("layout")
            == {"tp_size": 1, "pp_size": 1, "dp_size": 4}
            and len(str(input_binding.get("config", {}).get("sha256", "")))
            == 64
            and len(
                str(input_binding.get("import_manifest", {}).get("sha256", ""))
            )
            == 64
            and len(str(input_binding.get("physical_profile_sha256", "")))
            == 64
            and len(str(input_binding.get("source_manifest_sha256", "")))
            == 64
            and len(
                str(
                    input_binding.get("anchor_result", {}).get(
                        "artifact_hashes_sha256", ""
                    )
                )
            )
            == 64
            and calibration.get("calibration_semantics")
            == V100_PROFILE_SEMANTICS
            and calibration.get("runtime_consumption_contract")
            == V100_RUNTIME_CONSUMPTION_CONTRACT
            and calibration.get("physical_profile", {}).get("sha256")
            == input_binding.get("physical_profile_sha256")
            and calibration.get("source_manifest", {}).get("sha256")
            == input_binding.get("source_manifest_sha256")
        ),
        "full_dataset_exact": all(
            manifest[key] == anchor_manifest[key]
            for key in ("workflow_count", "request_count", "input_tokens", "output_tokens")
        ),
        "all_sealed_chunks_replayed": replay_meta["sealed_chunk_events"] == layout.intended_chunk_count,
        "paired_failure_request_keys": all(value == expected_request_keys for value in request_keys.values()),
        "paired_failure_stage_keys": all(value == expected_stage_keys for value in stage_keys.values()),
        "primary_and_dedicated_policies_present": set(ALL_E2E_POLICIES) == set(policies),
        "all_runtime_requests_reclaimed": all(
            snapshot.get("active_requests", 0) == 0
            for policy, aggregate in replay_meta["runtime"].items()
            for snapshot in (
                aggregate.get("snapshots", [])
                if "snapshots" in aggregate
                else [aggregate]
            )
        ),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    return {
        "status": "PASS" if not failed else "FAIL",
        "evidence_class": "full_real_trace_capacity_matched_metadata_baseline_simulation",
        "paper_ready": False,
        "physical_end_to_end_verified": False,
        "gates": gates,
        "failed_gates": failed,
        "boundaries": [
            "Weka request/token/timing scale is real; cross-workflow arrival is a deterministic modeled projection.",
            "All baseline labels are capacity-matched component adaptations unless explicitly marked native or a dedicated control.",
            "GhostServe parity is a declared TP=8 control and is excluded from the dense TP=1 primary-policy headline.",
            "CacheFlow and BSR are recovery-executor controls over already externalized state, not end-to-end protection policies.",
            "Four distinct V100S TP1/PP1/DP4 protection/recovery coefficients are consumed only by the matching DP4 simulation topology.",
            "SGLSim results are metadata-only and are not physical Mooncake end-to-end measurements.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "config.json")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-workflows", type=int)
    parser.add_argument("--skip-dataset-sha", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = args.config.resolve()
    (
        suite,
        base_config,
        runtime_config,
        calibration,
        projection,
        anchor_dir,
        input_binding,
    ) = preflight_baseline_suite(suite_path)
    dataset_path = (args.dataset or default_full_trace_path()).resolve()
    output_dir = _derived_baseline_output_dir(
        args.output_dir,
        profile_sha256=input_binding["physical_profile_sha256"],
        limit_workflows=args.limit_workflows,
    )
    started = time.monotonic()
    print(f"loading Weka trace: {dataset_path}", flush=True)
    source_workflows = load_weka_workflows(
        dataset_path, limit=args.limit_workflows
    )
    source_sha = "" if args.skip_dataset_sha else sha256_file(dataset_path)
    manifest = dataset_manifest(
        source_workflows,
        source_path=dataset_path,
        source_sha256=source_sha,
    )
    full_profile = args.limit_workflows is None
    if full_profile and source_sha != projection["source_trace_sha256"]:
        raise ValueError("baseline context projection source trace sha256 mismatch")
    if full_profile:
        expected = base_config["dataset"]["expected_full_profile"]
        mismatches = {
            key: (int(manifest[key]), int(expected[key]))
            for key in (
                "workflow_count",
                "request_count",
                "input_tokens",
                "output_tokens",
            )
            if int(manifest[key]) != int(expected[key])
        }
        if mismatches:
            raise ValueError(f"baseline full Weka dataset mismatch: {mismatches}")
    workflows = _scale_workflows(source_workflows, projection["factor"])
    output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"loaded workflows={manifest['workflow_count']} requests={manifest['request_count']}",
        flush=True,
    )
    layout = build_layout(workflows, base_config)
    anchor_events, aegis_failure_rows_raw = _read_anchor_failures(anchor_dir / "failure_requests.csv")
    base_stage_finishes, aegis_stage_rows_raw = _read_anchor_stages(anchor_dir / "failure_stages.csv")
    allowed_ids = set(layout.by_id)
    anchor_events = {
        event_id: [row for row in rows if row.request_id in allowed_ids]
        for event_id, rows in anchor_events.items()
    }
    anchor_events = {event_id: rows for event_id, rows in anchor_events.items() if rows}
    allowed_event_ids = set(anchor_events)
    allowed_stage_keys = {
        (row.event_id, row.workflow_id, row.stage_id)
        for rows in anchor_events.values()
        for row in rows
    }
    aegis_failure_rows_raw = [
        row for row in aegis_failure_rows_raw
        if row["event_id"] in allowed_event_ids and row["request_id"] in allowed_ids
    ]
    aegis_stage_rows_raw = [
        row
        for row in aegis_stage_rows_raw
        if (
            row["event_id"],
            row["workflow_id"],
            int(row["stage_id"]),
        )
        in allowed_stage_keys
    ]
    fault_manifest = _load_json(anchor_dir / "fault_manifest.json")
    baseline_failure_rows, baseline_stage_rows, replay_meta = replay_baselines(
        layout=layout,
        anchor_events=anchor_events,
        anchor_fault_manifest=fault_manifest,
        base_stage_finishes=base_stage_finishes,
        base_config=base_config,
        suite=suite,
        runtime_config=runtime_config,
    )
    aegis_failure_rows = _normalize_anchor_rows(aegis_failure_rows_raw, policy="aegiskv_latest")
    aegis_stage_rows = _normalize_anchor_stage_rows(aegis_stage_rows_raw, policy="aegiskv_latest")
    failure_rows = baseline_failure_rows + aegis_failure_rows
    stage_rows = baseline_stage_rows + aegis_stage_rows
    summaries = summarize(failure_rows, stage_rows)
    stats = suite["statistics"]
    bootstrap = paired_bootstrap(
        stage_rows,
        resamples=int(stats["bootstrap_resamples"]),
        seed=int(stats["bootstrap_seed"]),
    )
    control_rows, control_summary = restore_controls(failure_rows, runtime_config, suite)
    anchor_manifest = _load_json(anchor_dir / "dataset_manifest.json")
    audit_result = audit(
        manifest=manifest,
        anchor_manifest=anchor_manifest,
        layout=layout,
        replay_meta=replay_meta,
        failure_rows=failure_rows,
        stage_rows=stage_rows,
        calibration=calibration,
        input_binding=input_binding,
    )
    _dump_json(output_dir / "config_resolved.json", {
        "suite": suite,
        "base_config": base_config,
        "runtime_config": asdict(runtime_config),
        "dataset_resolved": str(dataset_path),
        "anchor_result_dir": str(anchor_dir),
        "input_binding": input_binding,
    })
    _dump_json(output_dir / "dataset_manifest.json", manifest)
    _dump_json(output_dir / "calibration_provenance.json", calibration)
    _dump_json(output_dir / "runtime_aggregate.json", replay_meta["runtime"])
    _dump_json(output_dir / "placement_counts.json", replay_meta["placement_counts"])
    _dump_json(output_dir / "summary.json", summaries)
    _dump_json(output_dir / "paired_bootstrap.json", bootstrap)
    _dump_json(output_dir / "restore_control_summary.json", control_summary)
    _write_csv(output_dir / "failure_requests.csv", failure_rows)
    _write_csv(output_dir / "failure_stages.csv", stage_rows)
    _write_csv(output_dir / "restore_controls.csv", control_rows)
    _dump_json(output_dir / "audit.json", audit_result)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _dump_json(output_dir / "artifact_hashes.json", hashes)
    result = {
        "status": audit_result["status"],
        "wall_seconds": time.monotonic() - started,
        "output_dir": str(output_dir),
        "dataset": {
            key: manifest[key]
            for key in ("workflow_count", "request_count", "input_tokens", "output_tokens")
        },
        "sealed_chunks": replay_meta["sealed_chunk_events"],
        "paired_failure_requests": len(aegis_failure_rows),
        "summary": summaries,
        "failed_gates": audit_result["failed_gates"],
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
