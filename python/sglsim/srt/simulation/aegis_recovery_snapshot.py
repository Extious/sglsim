"""Build causal AegisKV recovery snapshots from opt-in SGLSim telemetry.

The builder enforces a hard phase boundary: scheduler state at or before a
failure-control event is eligible input; request completion outcomes remain in
the post-hoc artifact and are never copied into the decision snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


SNAPSHOT_SCHEMA = "aegiskv-causal-recovery-snapshot-v3"
FORBIDDEN_CAUSAL_KEY_FRAGMENTS = (
    "actual_remaining",
    "actual_slack",
    "actual_output",
    "trace_oracle",
    "future_",
    "finish_time",
    "completion_time",
)
ALLOWED_CAUSAL_PROVENANCE_KEYS = frozenset(
    {
        "slack_uses_future_information",
        "remaining_prediction_uses_future_information",
    }
)
RECOVERY_STATE_EVENT_KINDS = frozenset(
    {
        "backup_ack",
        "backup_frontier_superseded",
        "host_pool_lease_evicted",
        "persistent_pool_lease_evicted",
        "gpu_recovery_l1_cleared",
    }
)
RECOVERY_EVENT_FIELDS = frozenset(
    {
        "kind",
        "sequence",
        "time",
        "rid",
        "backup_key",
        "execution_dp_rank",
        "backup_dp_rank",
        "source_dp_rank",
        "target_dp_rank",
        "dp_rank",
        "version",
        "workflow_id",
        "agent_id",
        "worker_seq",
        "l1_tokens",
        "l2_tokens",
        "l3_tokens",
        "safe_frontier_tokens",
        "continuous_recoverable_prefix_tokens",
        "materialization_scope",
        "source_committed_token_end",
        "group_commit_verified",
        "copy_completion_evidence",
        "execution_id",
        "block_generation",
        "recompute_tokens",
        "protected_recovery_service_s",
        "unprotected_recovery_service_s",
        "recovery_service_cost_source",
        "recovery_service_cost_calibration_id",
        "model_layout_id",
        "layout_compatible",
        "lost_l2_tokens",
        "lost_l3_tokens",
        "superseded_by",
        "updated_time",
    }
)


def _contains_forbidden_key(value: object, *, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            dotted = f"{path}.{key}" if path else key
            lowered = key.lower()
            if (
                lowered not in ALLOWED_CAUSAL_PROVENANCE_KEYS
                and any(
                    fragment in lowered
                    for fragment in FORBIDDEN_CAUSAL_KEY_FRAGMENTS
                )
            ):
                violations.append(dotted)
            violations.extend(_contains_forbidden_key(nested, path=dotted))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            violations.extend(
                _contains_forbidden_key(nested, path=f"{path}[{index}]")
            )
    return violations


def _nonfinite_number_paths(value: object, *, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            dotted = f"{path}.{key}" if path else key
            violations.extend(_nonfinite_number_paths(nested, path=dotted))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            violations.extend(
                _nonfinite_number_paths(nested, path=f"{path}[{index}]")
            )
    elif isinstance(value, float) and not math.isfinite(value):
        violations.append(path)
    return violations


def validate_causal_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("unexpected causal recovery snapshot schema")
    if snapshot.get("information_scope") != "causal_at_failure_event":
        raise ValueError("snapshot must declare causal_at_failure_event scope")
    cutoff = float(snapshot.get("cutoff_time_s", -1.0))
    if cutoff < 0.0:
        raise ValueError("snapshot cutoff_time_s must be nonnegative")
    failure = snapshot.get("failure_event")
    if not isinstance(failure, Mapping):
        raise ValueError("snapshot requires one failure_event")
    if str(failure.get("operation")) != "inject":
        raise ValueError("snapshot failure_event must be an injection")
    if not bool(failure.get("accepted", False)):
        raise ValueError("snapshot failure_event must be accepted")
    targets = snapshot.get("targets")
    requests = snapshot.get("active_requests")
    prefixes = snapshot.get("recoverable_prefixes_at_cutoff")
    copy_records = snapshot.get("copy_records_at_cutoff")
    if (
        not isinstance(targets, list)
        or not isinstance(requests, list)
        or not isinstance(prefixes, list)
        or not isinstance(copy_records, list)
    ):
        raise ValueError("snapshot targets and active_requests must be lists")
    violations = _contains_forbidden_key(snapshot)
    if violations:
        raise ValueError(
            "causal snapshot contains forbidden future fields: "
            + ", ".join(violations[:8])
        )
    for request in requests:
        if any(
            bool(request.get(field, False))
            for field in ALLOWED_CAUSAL_PROVENANCE_KEYS
        ):
            raise ValueError("causal snapshot declares use of future information")
    nonfinite = _nonfinite_number_paths(snapshot)
    if nonfinite:
        raise ValueError(
            "causal snapshot contains non-finite numbers: "
            + ", ".join(nonfinite[:8])
        )
    for target in targets:
        if float(target.get("observed_at_s", cutoff)) > cutoff + 1e-9:
            raise ValueError("target state occurs after failure cutoff")
    resource_state = snapshot.get("protection_resource_state")
    if isinstance(resource_state, Mapping) and float(
        resource_state.get("observed_at_s", cutoff)
    ) > cutoff + 1e-9:
        raise ValueError("resource state occurs after failure cutoff")
    for prefix in prefixes:
        if float(prefix.get("last_update_time_s", cutoff)) > cutoff + 1e-9:
            raise ValueError("recoverable prefix update occurs after failure cutoff")
        if not bool(prefix.get("reusable_for_recovery", False)):
            raise ValueError("recoverable prefix lacks reusable commit proof")
        source_end = prefix.get("source_committed_token_end")
        if source_end is None or int(
            prefix.get("continuous_recoverable_prefix_tokens", 0)
        ) > int(source_end):
            raise ValueError("recoverable prefix exceeds source commit")


def _select_failure_event(
    artifact: Mapping[str, Any], event_id: str
) -> dict[str, Any]:
    for raw in artifact.get("failure_events", ()):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("event_id")) == event_id:
            event = dict(raw)
            if event.get("operation") != "inject":
                raise ValueError(f"failure event {event_id} is not an injection")
            if not bool(event.get("accepted", False)):
                raise ValueError(f"failure event {event_id} was not accepted")
            return event
    raise KeyError(f"failure event {event_id!r} not found")


def _latest_rows_before(
    artifact: Mapping[str, Any], cutoff_s: float
) -> dict[tuple[int, int], dict[str, Any]]:
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in artifact.get("scheduler_telemetry", ()):
        if not isinstance(raw, Mapping):
            continue
        completed_at = float(raw.get("completed_at", -1.0))
        if completed_at < 0.0 or completed_at > cutoff_s + 1e-9:
            continue
        key = (int(raw.get("dp_rank", -1)), int(raw.get("pp_rank", 0)))
        if key[0] < 0:
            continue
        previous = latest.get(key)
        if previous is None or completed_at > float(previous["completed_at"]):
            latest[key] = dict(raw)
    return latest


def _event_time(raw: Mapping[str, Any]) -> float | None:
    for key in ("time", "observed_at_s", "completed_at", "updated_time"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _project_recovery_event(
    raw: Mapping[str, Any], *, timestamp: float
) -> dict[str, Any]:
    """Keep only fields consumed by causal recovery-state replay."""

    projected = {
        str(key): value
        for key, value in raw.items()
        if str(key) in RECOVERY_EVENT_FIELDS
    }
    projected["time"] = timestamp
    raw_updates = raw.get("hint_updates")
    if isinstance(raw_updates, list):
        projected["hint_updates"] = [
            {
                str(key): value
                for key, value in update.items()
                if str(key) in RECOVERY_EVENT_FIELDS
            }
            for update in raw_updates
            if isinstance(update, Mapping)
        ]
    return projected


def _nonnegative_int(value: object, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return max(int(default), 0)


def _signed_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _optional_nonnegative_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _normalize_hint_update(
    raw: Mapping[str, Any],
    *,
    event_time_s: float,
    event_sequence: int,
) -> dict[str, Any] | None:
    backup_key = str(raw.get("backup_key") or raw.get("rid") or "")
    target_dp_rank = _signed_int(
        raw.get("target_dp_rank", raw.get("backup_dp_rank", -1))
    )
    if not backup_key or target_dp_rank < 0:
        return None
    l1_tokens = _nonnegative_int(raw.get("l1_tokens", 0))
    l2_tokens = _nonnegative_int(raw.get("l2_tokens", 0))
    l3_tokens = _nonnegative_int(raw.get("l3_tokens", 0))
    tier_breakdown_complete = any(
        key in raw for key in ("l1_tokens", "l2_tokens", "l3_tokens")
    )
    tier_sum = l1_tokens + l2_tokens + l3_tokens
    prefix_tokens = _nonnegative_int(
        raw.get(
            "continuous_recoverable_prefix_tokens",
            raw.get("safe_frontier_tokens", tier_sum),
        )
    )
    if tier_breakdown_complete:
        prefix_tokens = tier_sum
    materialization_scope = str(
        raw.get("materialization_scope") or "legacy_unspecified"
    )
    raw_source_end = raw.get("source_committed_token_end")
    source_committed_token_end = (
        _nonnegative_int(raw_source_end) if raw_source_end is not None else None
    )
    group_commit_verified = bool(raw.get("group_commit_verified", False))
    copy_completion_evidence = str(
        raw.get("copy_completion_evidence") or ""
    )
    execution_id = str(raw.get("execution_id") or "")
    block_generation = _signed_int(raw.get("block_generation", -1))
    materialization_error: str | None = None
    if materialization_scope == "committed_prefix":
        if not group_commit_verified:
            materialization_error = "missing_group_commit_verification"
        elif source_committed_token_end is None:
            materialization_error = "missing_source_committed_token_end"
        elif prefix_tokens > source_committed_token_end:
            materialization_error = "prefix_exceeds_source_commit"
        elif not execution_id:
            materialization_error = "missing_execution_identity"
        elif block_generation < 0:
            materialization_error = "missing_block_generation"
    reusable_for_recovery = (
        materialization_scope == "committed_prefix"
        and materialization_error is None
    )
    return {
        "rid": str(raw.get("rid") or ""),
        "backup_key": backup_key,
        "version": max(_nonnegative_int(raw.get("version", 1)), 1),
        "source_dp_rank": _signed_int(
            raw.get("source_dp_rank", raw.get("execution_dp_rank", -1))
        ),
        "target_dp_rank": target_dp_rank,
        "workflow_id": str(raw.get("workflow_id") or ""),
        "agent_id": str(raw.get("agent_id") or ""),
        "worker_seq": _nonnegative_int(raw.get("worker_seq", 0)),
        "l1_tokens": l1_tokens,
        "l2_tokens": l2_tokens,
        "l3_tokens": l3_tokens,
        "continuous_recoverable_prefix_tokens": prefix_tokens,
        "materialization_scope": materialization_scope,
        "source_committed_token_end": source_committed_token_end,
        "group_commit_verified": group_commit_verified,
        "copy_completion_evidence": copy_completion_evidence,
        "execution_id": execution_id,
        "block_generation": block_generation,
        "materialization_error": materialization_error,
        "reusable_for_recovery": reusable_for_recovery,
        "recompute_tokens": _nonnegative_int(raw.get("recompute_tokens", 0)),
        "protected_recovery_service_s": _optional_nonnegative_float(
            raw.get("protected_recovery_service_s")
        ),
        "unprotected_recovery_service_s": _optional_nonnegative_float(
            raw.get("unprotected_recovery_service_s")
        ),
        "recovery_service_cost_source": str(
            raw.get("recovery_service_cost_source") or ""
        ),
        "recovery_service_cost_calibration_id": str(
            raw.get("recovery_service_cost_calibration_id") or ""
        ),
        "model_layout_id": str(raw.get("model_layout_id") or ""),
        "layout_compatible": (
            bool(raw.get("layout_compatible"))
            if raw.get("layout_compatible") is not None
            else None
        ),
        "tier_breakdown_complete": tier_breakdown_complete,
        "state_exact": True,
        "last_update_time_s": event_time_s,
        "last_event_sequence": event_sequence,
        "evidence_source": "controller_event_replay",
    }


def _replay_recoverable_prefixes(
    events: list[dict[str, Any]],
    *,
    failed_ranks: set[int],
    event_history: Mapping[str, Any] | None,
    provided_event_count: int,
    undated_event_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild live target prefixes from the causal controller journal.

    Plans do not count as copies.  A prefix becomes usable only after a backup
    ACK, and later supersede/eviction/L1-clear events are applied in journal
    order.  This prevents an old successful copy from being mistaken for a
    currently recoverable one.
    """

    ordered = sorted(
        enumerate(events),
        key=lambda item: (
            float(_event_time(item[1]) or 0.0),
            _signed_int(item[1].get("sequence", 0), 0),
            item[0],
        ),
    )
    live: dict[str, dict[str, Any]] = {}
    for _index, event in ordered:
        kind = str(event.get("kind") or "")
        timestamp = float(_event_time(event) or 0.0)
        sequence = _signed_int(event.get("sequence", 0), 0)
        if kind == "backup_ack":
            raw_updates = event.get("hint_updates")
            updates = (
                [row for row in raw_updates if isinstance(row, Mapping)]
                if isinstance(raw_updates, list)
                else [event]
            )
            for raw_update in updates:
                normalized = _normalize_hint_update(
                    raw_update,
                    event_time_s=timestamp,
                    event_sequence=sequence,
                )
                if normalized is None:
                    continue
                key = str(normalized["backup_key"])
                previous = live.get(key)
                if previous is not None and int(previous["version"]) > int(
                    normalized["version"]
                ):
                    continue
                live[key] = normalized
            continue

        backup_key = str(event.get("backup_key") or "")
        if kind == "backup_frontier_superseded":
            current = live.get(backup_key)
            if current is not None and int(current["version"]) <= max(
                _nonnegative_int(event.get("version", 1)), 1
            ):
                live.pop(backup_key, None)
            continue

        if kind in {
            "host_pool_lease_evicted",
            "persistent_pool_lease_evicted",
        }:
            current = live.get(backup_key)
            if current is None:
                continue
            event_version = max(
                _nonnegative_int(event.get("version", current["version"])), 1
            )
            if event_version != int(current["version"]):
                continue
            tier = "l2_tokens" if kind.startswith("host_") else "l3_tokens"
            lost_key = "lost_l2_tokens" if tier == "l2_tokens" else "lost_l3_tokens"
            lost_tokens = _nonnegative_int(event.get(lost_key, 0))
            current[tier] = max(int(current[tier]) - lost_tokens, 0)
            if not bool(current["tier_breakdown_complete"]):
                current["state_exact"] = False
            current["continuous_recoverable_prefix_tokens"] = max(
                int(current["continuous_recoverable_prefix_tokens"])
                - lost_tokens,
                0,
            )
            current["last_update_time_s"] = timestamp
            current["last_event_sequence"] = sequence
            continue

        if kind == "gpu_recovery_l1_cleared":
            cleared_rank = _signed_int(event.get("dp_rank", -1))
            for current in live.values():
                if int(current["target_dp_rank"]) != cleared_rank:
                    continue
                lost_tokens = int(current["l1_tokens"])
                current["l1_tokens"] = 0
                if not bool(current["tier_breakdown_complete"]):
                    current["state_exact"] = False
                    current["continuous_recoverable_prefix_tokens"] = (
                        int(current["l2_tokens"]) + int(current["l3_tokens"])
                    )
                else:
                    current["continuous_recoverable_prefix_tokens"] = max(
                        int(current["continuous_recoverable_prefix_tokens"])
                        - lost_tokens,
                        0,
                    )
                current["last_update_time_s"] = timestamp
                current["last_event_sequence"] = sequence

    history = dict(event_history or {})
    event_count = _signed_int(history.get("event_count", -1), -1)
    full_history = (
        str(history.get("mode") or "") == "unbounded_in_memory"
        and event_count == provided_event_count
        and undated_event_count == 0
    )
    prefixes: list[dict[str, Any]] = []
    for key in sorted(live):
        row = dict(live[key])
        target_rank = int(row["target_dp_rank"])
        row["target_id"] = f"dp:{target_rank}"
        row["survives_failure"] = target_rank not in failed_ranks
        row["valid_at_cutoff"] = True
        prefixes.append(row)
    return prefixes, {
        "mode": "ordered_controller_event_replay",
        "source_history_mode": str(history.get("mode") or "unknown"),
        "source_event_count": event_count if event_count >= 0 else None,
        "provided_event_count": provided_event_count,
        "selected_event_count": len(events),
        "undated_event_count": undated_event_count,
        "causal_replay_complete": full_history,
    }


def build_causal_recovery_snapshot(
    artifact: Mapping[str, Any], *, failure_event_id: str
) -> dict[str, Any]:
    """Select the latest per-stage state no later than one accepted failure."""

    if str(artifact.get("simulation_time_mode")) != "wall":
        raise ValueError("v1 recovery snapshot builder requires wall-clock telemetry")
    failure = _select_failure_event(artifact, failure_event_id)
    cutoff_s = float(failure.get("observed_at_s", -1.0))
    if cutoff_s < 0.0:
        raise ValueError("failure event lacks observed_at_s")
    latest = _latest_rows_before(artifact, cutoff_s)
    if not latest:
        raise ValueError("no scheduler telemetry exists before the failure event")

    failed_rank = int(failure.get("dp_rank", -1))
    stage_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (dp_rank, _pp_rank), row in sorted(latest.items()):
        stage_rows[dp_rank].append(row)

    targets: list[dict[str, Any]] = []
    active_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for dp_rank, rows in sorted(stage_rows.items()):
        pp0 = next(
            (row for row in rows if int(row.get("pp_rank", 0)) == 0),
            rows[0],
        )
        causal = pp0.get("causal_state")
        if not isinstance(causal, Mapping):
            raise ValueError(
                f"latest dp_rank={dp_rank} telemetry lacks causal_state"
            )
        target = {
            "target_id": f"dp:{dp_rank}",
            "dp_rank": dp_rank,
            "healthy_after_failure": dp_rank != failed_rank,
            "observed_at_s": float(pp0["completed_at"]),
            "waiting_depth": int(
                pp0.get("waiting_depth_after", pp0.get("waiting_depth", 0))
            ),
            "running_depth": int(
                pp0.get("running_depth_after", pp0.get("running_depth", 0))
            ),
            "last_batch_mode": str(pp0.get("batch_mode", "idle")),
            "last_batch_request_ids": list(pp0.get("request_ids", ())),
            "last_step_pipeline_s": float(pp0.get("pipeline_base_s", 0.0)),
            "waiting_request_ids": list(causal.get("waiting_request_ids", ())),
            "running_request_ids": list(causal.get("running_request_ids", ())),
            "max_running_requests": int(causal.get("max_running_requests", 0)),
            "l1_total_tokens": int(causal.get("l1_total_tokens", 0)),
            "l1_used_tokens": int(causal.get("l1_used_tokens", 0)),
            "l1_available_tokens": int(causal.get("l1_available_tokens", 0)),
            "l1_available_and_evictable_tokens": int(
                causal.get("l1_available_and_evictable_tokens", 0)
            ),
            "l2_total_tokens": int(causal.get("l2_total_tokens", 0)),
            "l2_used_tokens": int(causal.get("l2_used_tokens", 0)),
            "model_path": str(causal.get("model_path", "")),
            "model_reload_horizon_s": (
                failure.get("model_reload_horizon_s")
                if dp_rank == failed_rank
                else 0.0
            ),
            "model_reload_horizon_source": (
                str(
                    failure.get("model_reload_horizon_source")
                    or "unavailable"
                )
                if dp_rank == failed_rank
                else "resident_same_model_at_cutoff"
            ),
            "target_availability_offset_s": _optional_nonnegative_float(
                causal.get("target_availability_offset_s")
            ),
            "target_availability_source": str(
                causal.get("target_availability_source") or "unavailable"
            ),
            "target_availability_plan_version": str(
                causal.get("target_availability_plan_version") or ""
            ),
            "stages": [
                {
                    "pp_rank": int(row.get("pp_rank", 0)),
                    "observed_at_s": float(row["completed_at"]),
                    "last_step_pipeline_s": float(
                        row.get("pipeline_base_s", 0.0)
                    ),
                }
                for row in rows
            ],
        }
        targets.append(target)
        for raw_request in causal.get("active_requests", ()):
            if not isinstance(raw_request, Mapping):
                continue
            request = dict(raw_request)
            rid = str(request.get("rid", ""))
            if not rid:
                continue
            request["source_failure_domain"] = f"dp_rank:{dp_rank}"
            request["failure_role"] = (
                "interrupted" if dp_rank == failed_rank else "bystander"
            )
            active_by_key[(dp_rank, rid)] = request

    aegis_status = artifact.get("aegis_status", {})
    copy_events: list[dict[str, Any]] = []
    event_history: Mapping[str, Any] | None = None
    provided_event_count = 0
    undated_event_count = 0
    if isinstance(aegis_status, Mapping):
        raw_history = aegis_status.get("event_history")
        if isinstance(raw_history, Mapping):
            event_history = raw_history
        for raw in aegis_status.get("events", ()):
            if not isinstance(raw, Mapping):
                continue
            provided_event_count += 1
            timestamp = _event_time(raw)
            if timestamp is None:
                undated_event_count += 1
            elif (
                timestamp <= cutoff_s + 1e-9
                and str(raw.get("kind") or "")
                in RECOVERY_STATE_EVENT_KINDS
            ):
                copy_events.append(
                    _project_recovery_event(raw, timestamp=timestamp)
                )

    copy_records, copy_state_replay = _replay_recoverable_prefixes(
        copy_events,
        failed_ranks={failed_rank},
        event_history=event_history,
        provided_event_count=provided_event_count,
        undated_event_count=undated_event_count,
    )
    recoverable_prefixes = [
        row for row in copy_records if bool(row["reusable_for_recovery"])
    ]
    replay_complete = bool(copy_state_replay["causal_replay_complete"])
    for request in active_by_key.values():
        rid = str(request.get("rid") or "")
        object_id = str(request.get("kv_object_id") or "")
        source_rank = _signed_int(request.get("assigned_dp_rank", -1))
        candidates = [
            {
                "target_id": str(prefix["target_id"]),
                "target_dp_rank": int(prefix["target_dp_rank"]),
                "backup_key": str(prefix["backup_key"]),
                "version": int(prefix["version"]),
                "continuous_recoverable_prefix_tokens": int(
                    prefix["continuous_recoverable_prefix_tokens"]
                ),
                "l1_tokens": int(prefix["l1_tokens"]),
                "l2_tokens": int(prefix["l2_tokens"]),
                "l3_tokens": int(prefix["l3_tokens"]),
                "state_exact": bool(prefix["state_exact"]),
                "evidence_source": str(prefix["evidence_source"]),
                "materialization_scope": str(
                    prefix["materialization_scope"]
                ),
                "source_committed_token_end": int(
                    prefix["source_committed_token_end"]
                ),
                "group_commit_verified": bool(
                    prefix["group_commit_verified"]
                ),
                "copy_completion_evidence": str(
                    prefix["copy_completion_evidence"]
                ),
                "execution_id": str(prefix["execution_id"]),
                "block_generation": int(prefix["block_generation"]),
                "protected_recovery_service_s": prefix.get(
                    "protected_recovery_service_s"
                ),
                "unprotected_recovery_service_s": prefix.get(
                    "unprotected_recovery_service_s"
                ),
                "recovery_service_cost_source": str(
                    prefix.get("recovery_service_cost_source") or ""
                ),
                "recovery_service_cost_calibration_id": str(
                    prefix.get("recovery_service_cost_calibration_id") or ""
                ),
                "model_layout_id": str(
                    prefix.get("model_layout_id") or ""
                ),
                "layout_compatible": prefix.get("layout_compatible"),
            }
            for prefix in recoverable_prefixes
            if bool(prefix["survives_failure"])
            and (
                str(prefix.get("rid") or "") == rid
                or (object_id and str(prefix["backup_key"]) == object_id)
            )
            and (
                int(prefix["source_dp_rank"]) < 0
                or int(prefix["source_dp_rank"]) == source_rank
            )
        ]
        request["target_recovery_candidates"] = sorted(
            candidates,
            key=lambda row: (
                int(row["target_dp_rank"]), str(row["backup_key"])
            ),
        )
        reservations = [
            {
                "target_id": str(record["target_id"]),
                "target_dp_rank": int(record["target_dp_rank"]),
                "backup_key": str(record["backup_key"]),
                "version": int(record["version"]),
                "reserved_prefix_tokens": int(
                    record["continuous_recoverable_prefix_tokens"]
                ),
                "materialization_scope": str(
                    record["materialization_scope"]
                ),
            }
            for record in copy_records
            if not bool(record["reusable_for_recovery"])
            and bool(record["survives_failure"])
            and (
                str(record.get("rid") or "") == rid
                or (object_id and str(record["backup_key"]) == object_id)
            )
            and (
                int(record["source_dp_rank"]) < 0
                or int(record["source_dp_rank"]) == source_rank
            )
        ]
        request["target_metadata_reservations"] = sorted(
            reservations,
            key=lambda row: (
                int(row["target_dp_rank"]), str(row["backup_key"])
            ),
        )
        invalid_commits = [
            {
                "target_id": str(record["target_id"]),
                "target_dp_rank": int(record["target_dp_rank"]),
                "backup_key": str(record["backup_key"]),
                "version": int(record["version"]),
                "claimed_prefix_tokens": int(
                    record["continuous_recoverable_prefix_tokens"]
                ),
                "source_committed_token_end": record[
                    "source_committed_token_end"
                ],
                "materialization_error": str(
                    record["materialization_error"] or "unknown"
                ),
            }
            for record in copy_records
            if str(record["materialization_scope"]) == "committed_prefix"
            and not bool(record["reusable_for_recovery"])
            and bool(record["survives_failure"])
            and (
                str(record.get("rid") or "") == rid
                or (object_id and str(record["backup_key"]) == object_id)
            )
            and (
                int(record["source_dp_rank"]) < 0
                or int(record["source_dp_rank"]) == source_rank
            )
        ]
        request["target_invalid_commit_records"] = sorted(
            invalid_commits,
            key=lambda row: (
                int(row["target_dp_rank"]), str(row["backup_key"])
            ),
        )
        if not replay_complete:
            request["target_prefix_observation_status"] = (
                "INCOMPLETE_EVENT_HISTORY"
            )
        elif any(
            int(row["continuous_recoverable_prefix_tokens"]) > 0
            for row in candidates
        ):
            request["target_prefix_observation_status"] = "OBSERVED_POSITIVE"
        elif invalid_commits:
            request["target_prefix_observation_status"] = "INVALID_COMMIT_PROOF"
        elif reservations:
            request["target_prefix_observation_status"] = (
                "UNVERIFIED_MATERIALIZATION"
            )
        else:
            request["target_prefix_observation_status"] = "OBSERVED_ZERO"

    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA,
        "information_scope": "causal_at_failure_event",
        "source_evidence_tier": str(artifact.get("evidence_tier", "unknown")),
        "cutoff_time_s": cutoff_s,
        "failure_event": failure,
        "lost_dp_ranks": [failed_rank],
        "targets": targets,
        "active_requests": [
            active_by_key[key] for key in sorted(active_by_key)
        ],
        "copy_events_observed_before_cutoff": copy_events,
        "copy_records_at_cutoff": copy_records,
        "recoverable_prefixes_at_cutoff": recoverable_prefixes,
        "copy_state_replay": copy_state_replay,
        "posthoc_outcomes_included": False,
    }
    raw_resource_state = artifact.get("protection_resource_state")
    if isinstance(raw_resource_state, Mapping):
        observed_at = _optional_nonnegative_float(
            raw_resource_state.get("observed_at_s")
        )
        if observed_at is not None and observed_at <= cutoff_s + 1e-9:
            snapshot["protection_resource_state"] = {
                key: raw_resource_state.get(key)
                for key in (
                    "peer_hbm_budget_bytes",
                    "store_protection_budget_bytes",
                    "copy_rate_cap_bytes_per_s",
                    "foreground_reserve_source",
                    "resource_budget_version",
                )
            }
            snapshot["protection_resource_state"]["observed_at_s"] = observed_at
    validate_causal_snapshot(snapshot)
    canonical = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    snapshot["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
    validate_causal_snapshot(snapshot)
    return snapshot


def audit_snapshot_capabilities(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Report exactly which Algorithm-2 inputs one snapshot can support."""

    validate_causal_snapshot(snapshot)
    requests = [
        row for row in snapshot.get("active_requests", ()) if isinstance(row, Mapping)
    ]
    targets = [row for row in snapshot.get("targets", ()) if isinstance(row, Mapping)]
    interrupted = [row for row in requests if row.get("failure_role") == "interrupted"]
    bystanders = [row for row in requests if row.get("failure_role") == "bystander"]
    has_objects = bool(interrupted) and all(
        bool(row.get("kv_object_id")) for row in interrupted
    )
    has_reload = bool(targets) and all(
        row.get("model_reload_horizon_s") is not None for row in targets
    )
    reload_sources = {
        str(row.get("model_reload_horizon_source") or "") for row in targets
    }
    reload_is_modeled = any(
        source == "configured_causal_model" for source in reload_sources
    )
    prefix_statuses = {
        str(row.get("target_prefix_observation_status") or "")
        for row in interrupted
    }
    has_prefix_observability = bool(interrupted) and prefix_statuses <= {
        "OBSERVED_POSITIVE",
        "OBSERVED_ZERO",
    }
    positive_prefixes = sum(
        1
        for row in interrupted
        for candidate in row.get("target_recovery_candidates", ())
        if isinstance(candidate, Mapping)
        and int(candidate.get("continuous_recoverable_prefix_tokens", 0)) > 0
    )
    metadata_reservations = sum(
        1
        for row in interrupted
        for reservation in row.get("target_metadata_reservations", ())
        if isinstance(reservation, Mapping)
    )
    invalid_commit_records = sum(
        1
        for row in interrupted
        for record in row.get("target_invalid_commit_records", ())
        if isinstance(record, Mapping)
    )
    capabilities = {
        "source_failure_domain": "SUPPORTED" if interrupted else "NO_ACTIVE_VICTIM",
        "target_queue_batch_state": "SUPPORTED" if targets else "MISSING",
        "lost_compute_capacity": "SUPPORTED"
        if snapshot.get("lost_dp_ranks")
        else "MISSING",
        "causal_request_progress": "SUPPORTED" if requests else "MISSING",
        "bystander_identity": "SUPPORTED" if bystanders else "NO_ACTIVE_BYSTANDER",
        "kv_object_identity": "SUPPORTED" if has_objects else "MISSING",
        "target_reusable_prefix": (
            "SUPPORTED"
            if has_prefix_observability
            else "INVALID_COMMIT_PROOF"
            if "INVALID_COMMIT_PROOF" in prefix_statuses
            else "UNVERIFIED_MATERIALIZATION"
            if "UNVERIFIED_MATERIALIZATION" in prefix_statuses
            else "MISSING"
        ),
        "model_reload_horizon": (
            "MODELED"
            if has_reload and reload_is_modeled
            else "SUPPORTED"
            if has_reload
            else "MISSING"
        ),
        "bystander_counterfactual_completion": "REQUIRES_PAIRED_OUTCOME",
    }
    return {
        "schema_version": "aegiskv-snapshot-capability-audit-v3",
        "snapshot_sha256": str(snapshot.get("snapshot_sha256", "")),
        "interrupted_request_count": len(interrupted),
        "bystander_request_count": len(bystanders),
        "target_count": len(targets),
        "positive_target_prefix_count": positive_prefixes,
        "metadata_reservation_count": metadata_reservations,
        "invalid_commit_record_count": invalid_commit_records,
        "capabilities": capabilities,
        "g1_ready": all(
            capabilities[key] == "SUPPORTED"
            for key in (
                "source_failure_domain",
                "target_queue_batch_state",
                "lost_compute_capacity",
                "causal_request_progress",
                "bystander_identity",
                "kv_object_identity",
                "target_reusable_prefix",
                "model_reload_horizon",
            )
        )
        and capabilities["bystander_counterfactual_completion"] == "SUPPORTED",
    }


def audit_algorithm2_inputs(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a causal snapshot can instantiate Algorithm 2.

    A target prefix alone is insufficient.  The counterfactual objective also
    needs causal remaining-time predictions, target queue work in service-time
    units, executable alternative recovery costs, layout compatibility, and
    the current resource/link budget.  This audit prevents a developer from
    silently filling those fields with constants merely to obtain a ranking.
    """

    validate_causal_snapshot(snapshot)
    requests = [
        dict(row)
        for row in snapshot.get("active_requests", ())
        if isinstance(row, Mapping)
    ]
    targets = [
        dict(row)
        for row in snapshot.get("targets", ())
        if isinstance(row, Mapping)
    ]
    interrupted = [row for row in requests if row.get("failure_role") == "interrupted"]
    bystanders = [row for row in requests if row.get("failure_role") == "bystander"]

    def all_present(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
        return bool(rows) and all(
            all(row.get(field) not in (None, "") for field in fields)
            for row in rows
        )

    candidates = [
        dict(candidate)
        for row in interrupted
        for candidate in row.get("target_recovery_candidates", ())
        if isinstance(candidate, Mapping)
    ]
    reload_sources = {
        str(row.get("model_reload_horizon_source") or "") for row in targets
    }
    checks = {
        "interrupted_workflow_and_domain": bool(interrupted)
        and all(
            bool(row.get("workflow_id"))
            and bool(row.get("source_failure_domain"))
            for row in interrupted
        ),
        "committed_target_prefix": bool(candidates)
        and all(
            bool(candidate.get("group_commit_verified", False))
            and int(candidate.get("continuous_recoverable_prefix_tokens", 0))
            <= int(candidate.get("source_committed_token_end", -1))
            for candidate in candidates
        ),
        "interrupted_remaining_time_prediction": all_present(
            interrupted,
            (
                "predicted_remaining_lcb_ms",
                "predicted_remaining_q50_ms",
                "predicted_remaining_q90_ms",
                "remaining_prediction_source",
                "remaining_prediction_calibration_id",
            ),
        ),
        "bystander_remaining_time_prediction": all_present(
            bystanders,
            (
                "predicted_remaining_q50_ms",
                "remaining_prediction_source",
                "remaining_prediction_calibration_id",
            ),
        ),
        "target_queue_work_seconds": all_present(
            targets, ("target_queue_work_s",)
        ),
        "target_model_readiness_observed": bool(targets)
        and all_present(targets, ("model_reload_horizon_s",))
        and "configured_causal_model" not in reload_sources
        and "unavailable" not in reload_sources,
        "alternative_recovery_service_costs": bool(candidates)
        and all_present(
            candidates,
            (
                "protected_recovery_service_s",
                "unprotected_recovery_service_s",
            ),
        ),
        "layout_compatibility": bool(candidates)
        and all(
            candidate.get("layout_compatible") is True
            and bool(candidate.get("model_layout_id"))
            for candidate in candidates
        ),
        "resource_and_link_budget": isinstance(
            snapshot.get("protection_resource_state"), Mapping
        ),
        "identified_bystander_outcome_model": isinstance(
            snapshot.get("bystander_outcome_model"), Mapping
        )
        and str(
            snapshot.get("bystander_outcome_model", {}).get(
                "identification_status", ""
            )
        )
        == "VALIDATED",
    }
    blocking = [name for name, supported in checks.items() if not supported]
    runnable = not blocking
    return {
        "schema_version": "aegiskv-algorithm2-input-audit-v1",
        "snapshot_sha256": str(snapshot.get("snapshot_sha256") or ""),
        "information_scope": "causal_snapshot_only",
        "handcrafted_defaults_allowed": False,
        "interrupted_request_count": len(interrupted),
        "bystander_request_count": len(bystanders),
        "candidate_prefix_count": len(candidates),
        "checks": checks,
        "blocking_inputs": blocking,
        "algorithm2_runnable": runnable,
        "safe_result": "READY_TO_SCORE" if runnable else "REFUSED_TO_SCORE",
        "g2_status": "NOT_PROVEN" if not runnable else "REQUIRES_CALIBRATION",
    }


def audit_recovery_laxity_inputs(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the proposed EDD recovery-laxity online contract.

    Unlike :func:`audit_algorithm2_inputs`, this audit does not require a
    controller-side prediction of every bystander's counterfactual completion
    or every agent's full remaining-time interval.  It requires only the
    conservative Slack due date, a Router-owned fixed target assignment,
    causal target availability, executable recovery service costs, layout
    compatibility, and the resource caps that isolate foreground serving.

    Removing an unidentifiable input is not permission to invent a constant.
    Every retained field still needs explicit provenance and calibration.
    """

    validate_causal_snapshot(snapshot)
    requests = [
        dict(row)
        for row in snapshot.get("active_requests", ())
        if isinstance(row, Mapping)
    ]
    targets = [
        dict(row)
        for row in snapshot.get("targets", ())
        if isinstance(row, Mapping)
    ]
    interrupted = [
        row for row in requests if row.get("failure_role") == "interrupted"
    ]
    target_by_id = {
        str(row.get("target_id") or ""): row
        for row in targets
        if str(row.get("target_id") or "")
    }

    def nonempty(row: Mapping[str, Any], field: str) -> bool:
        return row.get(field) not in (None, "")

    def finite_nonnegative(row: Mapping[str, Any], field: str) -> bool:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value >= 0.0

    def sha256_value(row: Mapping[str, Any], field: str) -> bool:
        value = str(row.get(field) or "")
        return len(value) == 64 and all(
            character in "0123456789abcdef" for character in value.lower()
        )

    def valid_causal_slack(row: Mapping[str, Any]) -> bool:
        source = str(row.get("slack_source") or "")
        kind = str(row.get("slack_kind") or "").upper()
        need_state = str(row.get("need_state") or "").upper()
        if not (
            finite_nonnegative(row, "slack_lcb_ms")
            and nonempty(row, "slack_source")
            and str(row.get("slack_semantics") or "")
            == "revealed_dag_causal_slack_v1"
            and not bool(row.get("slack_uses_future_information", False))
            and sha256_value(row, "slack_input_provenance_sha256")
            and "oracle" not in source.lower()
            and "trace" not in source.lower()
            and need_state == "ACTIVE"
        ):
            return False
        if kind == "UNKNOWN":
            return (
                float(row["slack_lcb_ms"]) == 0.0
                and not bool(row.get("slack_usable", False))
            )
        if kind not in {"BOUNDED", "CALIBRATED"}:
            return False
        if not (
            bool(row.get("slack_usable", False))
            and nonempty(row, "slack_calibration_id")
            and finite_nonnegative(row, "slack_target_coverage")
            and 0.0 < float(row["slack_target_coverage"]) <= 1.0
        ):
            return False
        return True

    causal_slack = bool(interrupted) and all(
        valid_causal_slack(row) for row in interrupted
    )
    assigned_targets = bool(interrupted) and all(
        nonempty(row, "recovery_target_id")
        and str(row.get("recovery_target_id")) in target_by_id
        and bool(
            target_by_id[str(row.get("recovery_target_id"))].get(
                "healthy_after_failure", False
            )
        )
        and nonempty(row, "recovery_target_assignment_source")
        and str(row.get("recovery_target_assignment_source") or "")
        not in {"unavailable", "handcrafted_default"}
        and nonempty(row, "recovery_route_plan_version")
        for row in interrupted
    )
    assigned_target_rows = [
        target_by_id[str(row.get("recovery_target_id"))]
        for row in interrupted
        if str(row.get("recovery_target_id") or "") in target_by_id
    ]
    target_availability = assigned_targets and all(
        finite_nonnegative(row, "target_availability_offset_s")
        and nonempty(row, "target_availability_source")
        and str(row.get("target_availability_source") or "")
        not in {"unavailable", "handcrafted_default"}
        and nonempty(row, "target_availability_plan_version")
        for row in assigned_target_rows
    )

    candidate_groups = [
        [
            dict(candidate)
            for candidate in row.get("target_recovery_candidates", ())
            if isinstance(candidate, Mapping)
            and str(candidate.get("target_id") or "")
            == str(row.get("recovery_target_id") or "")
        ]
        for row in interrupted
    ]
    candidates = [
        candidate for group in candidate_groups for candidate in group
    ]
    committed_prefix = bool(candidate_groups) and all(
        group
        and all(
            bool(candidate.get("group_commit_verified", False))
            and int(candidate.get("continuous_recoverable_prefix_tokens", 0))
            <= int(candidate.get("source_committed_token_end", -1))
            for candidate in group
        )
        for group in candidate_groups
    )
    recovery_service_costs = bool(candidate_groups) and all(
        group
        and all(
            finite_nonnegative(candidate, "protected_recovery_service_s")
            and finite_nonnegative(candidate, "unprotected_recovery_service_s")
            and float(candidate["unprotected_recovery_service_s"]) > 0.0
            and float(candidate["protected_recovery_service_s"])
            <= float(candidate["unprotected_recovery_service_s"]) + 1e-12
            and nonempty(candidate, "recovery_service_cost_source")
            and str(candidate.get("recovery_service_cost_source") or "")
            not in {
                "default_coefficients",
                "handcrafted_default",
                "unavailable",
            }
            and nonempty(candidate, "recovery_service_cost_calibration_id")
            for candidate in group
        )
        for group in candidate_groups
    )
    layout_compatible = bool(candidate_groups) and all(
        group
        and all(
            candidate.get("layout_compatible") is True
            and nonempty(candidate, "model_layout_id")
            for candidate in group
        )
        for group in candidate_groups
    )
    resource_state = snapshot.get("protection_resource_state")
    resource_caps = (
        isinstance(resource_state, Mapping)
        and all(
            finite_nonnegative(resource_state, field)
            for field in (
                "peer_hbm_budget_bytes",
                "store_protection_budget_bytes",
                "copy_rate_cap_bytes_per_s",
            )
        )
        and nonempty(resource_state, "foreground_reserve_source")
        and str(resource_state.get("foreground_reserve_source") or "")
        not in {"default_coefficients", "handcrafted_default", "unavailable"}
        and nonempty(resource_state, "resource_budget_version")
        and finite_nonnegative(resource_state, "observed_at_s")
        and float(resource_state["observed_at_s"])
        <= float(snapshot.get("cutoff_time_s", -1.0)) + 1e-9
    )
    checks = {
        "interrupted_workflow_and_domain": bool(interrupted)
        and all(
            bool(row.get("workflow_id"))
            and bool(row.get("source_failure_domain"))
            for row in interrupted
        ),
        "causal_slack_due_date": causal_slack,
        "router_fixed_target_assignment": assigned_targets,
        "causal_target_availability": target_availability,
        "committed_target_prefix": committed_prefix,
        "calibrated_recovery_service_costs": recovery_service_costs,
        "layout_compatibility": layout_compatible,
        "foreground_bounded_resource_caps": resource_caps,
    }
    blocking = [name for name, supported in checks.items() if not supported]
    runnable = not blocking
    return {
        "schema_version": "aegiskv-recovery-laxity-input-audit-v1",
        "snapshot_sha256": str(snapshot.get("snapshot_sha256") or ""),
        "information_scope": "causal_snapshot_only",
        "online_model": "fixed_target_edd_recovery_lane",
        "handcrafted_defaults_allowed": False,
        "interrupted_request_count": len(interrupted),
        "assigned_target_count": len(assigned_target_rows),
        "candidate_prefix_count": len(candidates),
        "checks": checks,
        "blocking_inputs": blocking,
        "algorithm2_runnable": runnable,
        "safe_result": "READY_TO_SCORE" if runnable else "REFUSED_TO_SCORE",
        "g2_status": "NOT_PROVEN" if not runnable else "REQUIRES_CALIBRATION",
        "removed_online_requirements": [
            "bystander_counterfactual_completion_model",
            "full_remaining_time_interval_for_every_bystander",
            "controller_emulation_of_future_continuous_batching",
        ],
    }
