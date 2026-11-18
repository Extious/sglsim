from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .common import request_json, wait_for_server
from .workflow_client import (
    DEFAULT_SLACK_CALIBRATION,
    DEFAULT_TRACE,
    DemoWorkflow,
    POLICY_REQUEST_NAMES,
    RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION,
    SLACK_SOURCES,
    failure_manifest_retry_route_mode,
)


POLICY_LABELS = {
    "aegiskv": "AegisKV",
    "l1_only": "L1 only",
    "l2_only": "L2 only",
    "l3_only": "L3 only",
    "no_backup": "No backup",
}
DEFAULT_FAILURE_MANIFEST = (
    Path(__file__).resolve().parent / "config" / "failure_manifest.json"
)
WORKFLOW_LAUNCH_POLICY = "ordered_predecessor_chain_v1"
WORKFLOW_LAUNCH_STAGGER_S = 0.25


class OrderedWorkflowLaunchGate:
    """Make concurrent policy replays route-comparable across clean servers.

    ThreadPool submission order does not guarantee which workflow reaches the
    Router first.  That race changes the first cache-aware DP assignment and
    can dominate WCT after a failure.  The gate starts the initial active set
    in lexical order, then lets workflow ``i`` backfill only after workflow
    ``i-concurrency`` completes.  It does not select a DP rank or enter the
    request data path; SGLang-style cache/load-aware routing remains in charge.
    """

    def __init__(
        self,
        *,
        workflow_count: int,
        concurrency: int,
        start_monotonic: float,
        stagger_s: float = WORKFLOW_LAUNCH_STAGGER_S,
    ) -> None:
        self.workflow_count = max(int(workflow_count), 0)
        self.concurrency = max(int(concurrency), 1)
        self.start_monotonic = float(start_monotonic)
        self.stagger_s = max(float(stagger_s), 0.0)
        self._slots = threading.BoundedSemaphore(self.concurrency)
        self._completed = [
            threading.Event() for _ in range(self.workflow_count)
        ]

    def enter(self, index: int) -> None:
        index = int(index)
        predecessor = index - self.concurrency
        if predecessor >= 0:
            self._completed[predecessor].wait()
        self._slots.acquire()
        if index < self.concurrency and self.stagger_s > 0.0:
            due = self.start_monotonic + (index + 1) * self.stagger_s
            delay = due - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)

    def leave(self, index: int) -> None:
        index = int(index)
        self._completed[index].set()
        self._slots.release()


def progress_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.partial{output.suffix}")


def write_progress_checkpoint(
    *,
    output: Path,
    policy: str,
    summaries: list[dict[str, Any]],
    planned_workflows: int,
    total_requests: int,
    failure_events: list[dict[str, Any]],
) -> Path:
    """Atomically persist non-final progress without entering the data path."""

    path = progress_output_path(output)
    temporary = path.with_name(f"{path.name}.tmp")
    payload = {
        "schema_version": 1,
        "complete": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "planned_workflows": int(planned_workflows),
        "completed_workflows": len(summaries),
        "request_count": int(total_requests),
        "failure_events": list(failure_events),
        "workflows": sorted(
            summaries,
            key=lambda item: str(item["workflow_id"]),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_workflows(trace_path: Path) -> dict[str, list[dict[str, Any]]]:
    workflows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            workflows[str(row["workflow_id"])].append(row)
    if not workflows:
        raise ValueError(f"no workflows found in {trace_path}")
    return {
        workflow_id: sorted(
            rows,
            key=lambda row: (
                int(row.get("stage_id", 0)),
                float(row.get("arrival_t", 0.0)),
                int(row.get("group_id", 0)),
            ),
        )
        for workflow_id, rows in sorted(workflows.items())
    }


def has_fan_out_and_main_fan_in(rows: list[dict[str, Any]]) -> bool:
    """Return whether a workflow fans out and later joins at its main agent."""
    agents_by_stage: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        agents_by_stage[int(row.get("stage_id", 0))].add(
            str(row.get("agent_id", ""))
        )
    stages = sorted(agents_by_stage.items())
    for index, (_stage_id, agents) in enumerate(stages):
        if len(agents) <= 1:
            continue
        for _later_stage_id, later_agents in stages[index + 1 :]:
            if len(later_agents) == 1 and next(iter(later_agents)).endswith(
                "-main"
            ):
                return True
    return False


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def normalize_failure_manifest(
    manifest: dict[str, Any],
    *,
    dp_size: int,
    pp_size: int = 1,
    failure_count: int | None = None,
    source: str = "<in-memory failure manifest>",
) -> dict[str, Any]:
    schema_version = int(manifest.get("schema_version", 0))
    if schema_version not in (1, 2):
        raise ValueError(f"unsupported failure manifest schema: {source}")
    if manifest.get("time_origin") != "policy_workload_start":
        raise ValueError("failure manifest time_origin must be policy_workload_start")
    events = list(manifest.get("events", []))
    deployment = dict(manifest.get("deployment", {}))
    deployment_scope = str(deployment.get("failure_scope") or "node")
    retry_route_mode = failure_manifest_retry_route_mode(manifest)
    if failure_count is not None:
        events = events[: max(failure_count, 0)]
    previous_end = -1.0
    seen_ids: set[str] = set()
    seen_victim_keys: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(events):
        event_id = str(raw.get("event_id", f"failure-{index + 1:02d}"))
        offset_s = float(raw["offset_s"])
        dp_rank = int(raw["dp_rank"])
        pp_rank = int(raw.get("pp_rank", -1))
        failure_scope = str(raw.get("failure_scope") or deployment_scope)
        duration_s = float(raw["duration_s"])
        if event_id in seen_ids:
            raise ValueError(f"duplicate failure event_id: {event_id}")
        if offset_s < 0 or duration_s <= 0:
            raise ValueError(f"invalid timing for failure event {event_id}")
        if not 0 <= dp_rank < dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} in failure event {event_id}")
        if pp_rank >= 0 and not 0 <= pp_rank < pp_size:
            raise ValueError(f"invalid pp_rank={pp_rank} in failure event {event_id}")
        if offset_s < previous_end:
            raise ValueError(
                f"overlapping failure event {event_id}; previous event ends at "
                f"{previous_end:.3f}s"
            )
        seen_ids.add(event_id)
        previous_end = offset_s + duration_s
        victims: list[dict[str, Any]] = []
        for victim_raw in raw.get("victims", []):
            if not isinstance(victim_raw, dict):
                raise ValueError(f"invalid victim in failure event {event_id}")
            backup_key = str(victim_raw.get("backup_key") or "")
            execution_dp_rank = int(
                victim_raw.get("execution_dp_rank", dp_rank)
            )
            if not backup_key:
                raise ValueError(f"empty victim backup_key in {event_id}")
            if backup_key in seen_victim_keys:
                raise ValueError(f"duplicate failure victim: {backup_key}")
            if execution_dp_rank != dp_rank:
                raise ValueError(
                    f"victim {backup_key} executes on rank {execution_dp_rank}, "
                    f"not failed rank {dp_rank}"
                )
            backup_dp_rank: int | None = None
            if (
                retry_route_mode
                == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
            ):
                backup_dp_rank = int(victim_raw.get("backup_dp_rank", -1))
                if (
                    not 0 <= backup_dp_rank < dp_size
                    or backup_dp_rank == dp_rank
                ):
                    raise ValueError(
                        f"invalid backup_dp_rank={backup_dp_rank} for victim "
                        f"{backup_key}"
                    )
            elif "backup_dp_rank" in victim_raw:
                raise ValueError(
                    "formal router_committed failure victim must not prescribe "
                    f"backup_dp_rank: {backup_key}"
                )
            seen_victim_keys.add(backup_key)
            victim = {
                "backup_key": backup_key,
                "execution_dp_rank": execution_dp_rank,
                "pp_rank": pp_rank,
                "failure_scope": failure_scope,
                "retry_route_mode": retry_route_mode,
            }
            if backup_dp_rank is not None:
                victim["backup_dp_rank"] = backup_dp_rank
            victims.append(victim)
        normalized.append(
            {
                "event_id": event_id,
                "offset_s": offset_s,
                "dp_rank": dp_rank,
                "pp_rank": pp_rank,
                "failure_scope": failure_scope,
                "duration_s": duration_s,
                "victims": victims,
            }
        )
    return {
        "schema_version": schema_version,
        "time_origin": "policy_workload_start",
        "retry_route_mode": retry_route_mode,
        "deployment": deployment,
        "events": normalized,
    }


def load_failure_manifest(
    path: Path,
    *,
    dp_size: int,
    pp_size: int = 1,
    failure_count: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"failure manifest must be an object: {path}")
    return normalize_failure_manifest(
        manifest,
        dp_size=dp_size,
        pp_size=pp_size,
        failure_count=failure_count,
        source=str(path),
    )


def load_execution_route_plan(
    path: Path,
    *,
    dp_size: int,
) -> dict[str, Any]:
    """Load a measured request-to-DP plan for paired policy replay."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported execution route plan schema: {path}")
    if int(payload.get("dp_size", -1)) != int(dp_size):
        raise ValueError(
            f"route plan dp_size={payload.get('dp_size')}, expected {dp_size}"
        )
    raw_routes = payload.get("routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise ValueError(f"execution route plan has no routes: {path}")
    routes: dict[str, int] = {}
    for rid, raw in raw_routes.items():
        rank = int(
            raw.get("execution_dp_rank", -1)
            if isinstance(raw, dict)
            else raw
        )
        if not 0 <= rank < int(dp_size):
            raise ValueError(f"route plan has invalid rank {rank} for {rid}")
        routes[str(rid)] = rank
    return {
        **payload,
        "routes": {
            rid: {"execution_dp_rank": rank}
            for rid, rank in sorted(routes.items())
        },
    }


class FailureManifestCoordinator:
    """Replay fixed-time, fixed-rank soft failures independently of policy."""

    def __init__(
        self,
        *,
        base_url: str,
        manifest: dict[str, Any],
        start_monotonic: float,
    ) -> None:
        self.base_url = base_url
        self.manifest = manifest
        self.start_monotonic = start_monotonic
        self.events: list[dict[str, Any]] = []
        self.injected_failures = 0
        self._error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._injected_event_ids: set[str] = set()
        self._injected_offsets_s: dict[str, float] = {}
        self._manifest_order = {
            str(event["event_id"]): index
            for index, event in enumerate(self.manifest["events"])
        }
        self._thread = threading.Thread(
            target=self._run,
            name="failure-manifest-replay",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("failure manifest replay failed") from self._error

    def was_injected(self, event_id: str) -> bool:
        with self._state_lock:
            return str(event_id) in self._injected_event_ids

    def injected_offset_s(self, event_id: str) -> float | None:
        with self._state_lock:
            return self._injected_offsets_s.get(str(event_id))

    def snapshot_events(self) -> list[dict[str, Any]]:
        with self._state_lock:
            return sorted(
                (dict(event) for event in self.events),
                key=lambda event: self._manifest_order.get(
                    str(event.get("event_id") or ""), len(self._manifest_order)
                ),
            )

    def _wait_until(self, offset_s: float) -> None:
        deadline = self.start_monotonic + offset_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.05))

    def _run_event(self, manifest_event: dict[str, Any]) -> None:
        try:
            self._wait_until(float(manifest_event["offset_s"]))
            dp_rank = int(manifest_event["dp_rank"])
            result = request_json(
                self.base_url,
                "/simulate/fail_dp",
                payload={"dp_rank": dp_rank},
            )
            if not result.get("accepted", False):
                raise RuntimeError(
                    f"failure event {manifest_event['event_id']} rejected: {result}"
                )
            injected_offset_s = time.monotonic() - self.start_monotonic
            with self._state_lock:
                event_id = str(manifest_event["event_id"])
                self._injected_event_ids.add(event_id)
                self._injected_offsets_s[event_id] = injected_offset_s
                self.injected_failures += 1
            time.sleep(float(manifest_event["duration_s"]))
            recovered = request_json(
                self.base_url,
                "/simulate/recover_dp",
                payload={"dp_rank": dp_rank},
                timeout=1900.0,
            )
            if not recovered.get("released", False):
                raise RuntimeError(
                    f"recovery for {manifest_event['event_id']} rejected: {recovered}"
                )
            l1_clear = recovered.get("l1_clear", {})
            cache_result = l1_clear.get("cache_result", {})
            backup_result = l1_clear.get("backup_result", {})
            router_result = l1_clear.get("router_ledger_result", {})
            recovered_offset_s = time.monotonic() - self.start_monotonic
            with self._state_lock:
                self.events.append(
                    {
                        **manifest_event,
                        "injected_offset_s": round(injected_offset_s, 6),
                        "recovered_offset_s": round(recovered_offset_s, 6),
                        "actual_outage_s": round(
                            recovered_offset_s - injected_offset_s,
                            6,
                        ),
                        "recovery_control_latency_s": round(
                            recovered_offset_s
                            - injected_offset_s
                            - float(manifest_event["duration_s"]),
                            6,
                        ),
                        "l1_clear": {
                            "cleared_device_tokens": int(
                                cache_result.get("cleared_device_tokens", 0)
                            ),
                            "cleared_device_pages": int(
                                cache_result.get("cleared_device_pages", 0)
                            ),
                            "preserved_host_tokens": int(
                                cache_result.get("preserved_host_tokens", 0)
                            ),
                            "cleared_l1_backup_tokens": int(
                                backup_result.get(
                                    "cleared_l1_backup_tokens", 0
                                )
                            ),
                            "affected_backup_entries": int(
                                backup_result.get("affected_backup_entries", 0)
                            ),
                            "preserved_l2_backup_tokens": int(
                                backup_result.get(
                                    "preserved_l2_backup_tokens", 0
                                )
                            ),
                            "cleared_router_prefix_tokens": int(
                                router_result.get("cleared_prefix_tokens", 0)
                            ),
                        },
                    }
                )
        except BaseException as exc:
            with self._state_lock:
                if self._error is None:
                    self._error = exc

    def _run(self) -> None:
        event_threads = [
            threading.Thread(
                target=self._run_event,
                args=(dict(manifest_event),),
                name=f"failure-{manifest_event['event_id']}",
                daemon=True,
            )
            for manifest_event in self.manifest["events"]
        ]
        for thread in event_threads:
            thread.start()
        for thread in event_threads:
            thread.join()


def _workflow_args(
    args: argparse.Namespace,
    *,
    failure_manifest: dict[str, Any],
    policy_start_monotonic: float,
    failure_coordinator: FailureManifestCoordinator,
    execution_route_plan: dict[str, int],
) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=args.base_url,
        model=args.model,
        policy=args.policy,
        trace=args.trace,
        input_scale=args.input_scale,
        context_length=args.context_length,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        arrival_speedup=args.arrival_speedup,
        request_pause=0.0,
        stage_pause=0.0,
        linger_seconds=0.0,
        ack_timeout=args.ack_timeout,
        request_timeout=args.request_timeout,
        slack_calibration=args.slack_calibration,
        slack_source=args.slack_source,
        quiet=True,
        wait_for_ack=False,
        collect_final_status=False,
        dp_size=args.dp_size,
        pp_size=args.pp_size,
        failure_status_timeout=5.0,
        status_timeout=30.0,
        failure_manifest=failure_manifest,
        policy_start_monotonic=policy_start_monotonic,
        failure_coordinator=failure_coordinator,
        execution_route_plan=execution_route_plan,
    )


def run_policy(args: argparse.Namespace) -> dict[str, Any]:
    wait_for_server(args.base_url)
    trace_workflows = load_workflows(args.trace)
    workflows = {
        workflow_id: rows
        for workflow_id, rows in trace_workflows.items()
        if has_fan_out_and_main_fan_in(rows)
    }
    if not workflows:
        raise ValueError(f"no fan-out/fan-in workflows found in {args.trace}")
    fan_workflow_count = len(workflows)
    if args.max_requests_per_workflow > 0:
        workflows = {
            workflow_id: rows
            for workflow_id, rows in workflows.items()
            if len(rows) <= args.max_requests_per_workflow
        }
    if not workflows:
        raise ValueError(
            "no fan-out/fan-in workflows remain after the per-workflow "
            "request-count bound"
        )
    selected = list(workflows.items())[: args.max_workflows or None]
    total_requests = sum(len(rows) for _workflow_id, rows in selected)
    failure_manifest = load_failure_manifest(
        args.failure_manifest,
        dp_size=args.dp_size,
        pp_size=args.pp_size,
        failure_count=args.failure_count,
    )
    route_plan_payload: dict[str, Any] | None = None
    execution_route_plan: dict[str, int] = {}
    if args.route_plan is not None:
        route_plan_payload = load_execution_route_plan(
            args.route_plan,
            dp_size=args.dp_size,
        )
        execution_route_plan = {
            str(rid): int(spec["execution_dp_rank"])
            for rid, spec in route_plan_payload["routes"].items()
        }
        selected_rids = {
            str(row["rid"])
            for _workflow_id, rows in selected
            for row in rows
        }
        missing = sorted(selected_rids - set(execution_route_plan))
        if missing:
            raise ValueError(
                f"execution route plan misses {len(missing)} selected requests; "
                f"first={missing[0]}"
            )
    policy_start_monotonic = time.monotonic()
    coordinator = FailureManifestCoordinator(
        base_url=args.base_url,
        manifest=failure_manifest,
        start_monotonic=policy_start_monotonic,
    )
    coordinator.start()

    summaries: list[dict[str, Any]] = []
    worker_args = _workflow_args(
        args,
        failure_manifest=failure_manifest,
        policy_start_monotonic=policy_start_monotonic,
        failure_coordinator=coordinator,
        execution_route_plan=execution_route_plan,
    )

    launch_gate = OrderedWorkflowLaunchGate(
        workflow_count=len(selected),
        concurrency=args.workflow_concurrency,
        start_monotonic=policy_start_monotonic,
    )

    def run_one(index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
        launch_gate.enter(index)
        try:
            summary = DemoWorkflow(
                worker_args,
                rows,
            ).run()
        finally:
            launch_gate.leave(index)
        events = summary.pop("events", [])
        retry_events = [event for event in events if event.get("kind") == "retry_completed"]
        request_latencies = [
            float(event.get("latency_s", 0.0))
            for event in events
            if event.get("kind") == "request_completed"
        ]
        summary["retry_events"] = retry_events
        summary["algorithm1_slack_predictions"] = [
            dict(event["algorithm1_slack"])
            for event in events
            if event.get("kind") == "request_completed"
            and isinstance(event.get("algorithm1_slack"), dict)
        ]
        summary["request_latency_summary_s"] = {
            "mean": round(mean(request_latencies), 6) if request_latencies else 0.0,
            "p50": round(median(request_latencies), 6) if request_latencies else 0.0,
            "p90": round(percentile(request_latencies, 0.90), 6)
            if request_latencies
            else 0.0,
        }
        summary.pop("rank_states", None)
        return summary

    try:
        # Waiting successors need their own threads so the executor cannot
        # choose an arbitrary backfill workflow when a slot becomes available.
        with ThreadPoolExecutor(max_workers=max(len(selected), 1)) as pool:
            futures = {
                pool.submit(run_one, index, rows): workflow_id
                for index, (workflow_id, rows) in enumerate(selected)
            }
            for future in as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                print(
                    f"[{args.policy:<9}] {summary['workflow_id'][:8]} "
                    f"WCT={summary['wct_s']:.3f}s requests={summary['request_count']} "
                    f"retries={summary['retry_count']}",
                    flush=True,
                )
                write_progress_checkpoint(
                    output=args.output,
                    policy=args.policy,
                    summaries=summaries,
                    planned_workflows=len(selected),
                    total_requests=total_requests,
                    failure_events=coordinator.snapshot_events(),
                )
    finally:
        coordinator.join()

    summaries.sort(key=lambda item: str(item["workflow_id"]))
    wcts = [float(summary["wct_s"]) for summary in summaries]
    # Backup placement is asynchronous and intentionally outside request WCT.
    # Drain its ACKs after all workflows finish so diagnostic counts describe
    # the complete workload rather than an arbitrary queue snapshot.
    final_status: dict[str, Any] = {}
    ack_drain_deadline = time.monotonic() + 60.0
    while True:
        try:
            final_status = request_json(
                args.base_url,
                "/aegis/status",
                timeout=30.0,
            )
        except (TimeoutError, OSError):
            if time.monotonic() >= ack_drain_deadline:
                raise
            time.sleep(0.25)
            continue
        observed_acks = sum(
            1
            for hint in final_status.get("recovery_hints", [])
            if bool(hint.get("ack", False))
        )
        if args.policy == "no_backup" or observed_acks >= total_requests:
            break
        if time.monotonic() >= ack_drain_deadline:
            break
        time.sleep(0.1)
    status_events = list(final_status.get("events", []))
    backup_ack_events = [
        event for event in status_events if event.get("kind") == "backup_ack"
    ]
    authoritative_backup_hints = [
        hint
        for hint in final_status.get("recovery_hints", [])
        if isinstance(hint, dict) and bool(hint.get("ack", False))
    ]
    alpha_beta_updates = [
        {
            "sequence": int(event.get("sequence", 0)),
            "time": float(event.get("time", 0.0)),
            "dp_rank": int(event.get("backup_dp_rank", -1)),
            **dict(event["window_update"]),
        }
        for event in backup_ack_events
        if isinstance(event.get("window_update"), dict)
    ]
    node_beta_updates = [
        {
            "sequence": int(event.get("sequence", 0)),
            "time": float(event.get("time", 0.0)),
            **dict(event["node_host_window_update"]),
        }
        for event in backup_ack_events
        if isinstance(event.get("node_host_window_update"), dict)
    ]
    retry_events = [
        event
        for summary in summaries
        for event in summary.get("retry_events", [])
    ]
    restored_tokens = sum(int(event.get("restored_tokens", 0)) for event in retry_events)
    recompute_tokens = sum(int(event.get("recompute_tokens", 0)) for event in retry_events)
    recovery_tokens = restored_tokens + recompute_tokens
    capacity_limited_tokens = sum(
        int(event.get("capacity_limited_tokens", 0)) for event in retry_events
    )
    retry_latencies = [float(event.get("latency_s", 0.0)) for event in retry_events]
    recovery_actions: dict[str, int] = defaultdict(int)
    recovery_paths: dict[str, int] = defaultdict(int)
    l3_recovery_paths: dict[str, int] = defaultdict(int)
    gpu_direct_reuse_tokens = 0
    host_reuse_tokens = 0
    l3_reuse_tokens = 0
    for event in retry_events:
        recovery_actions[str(event.get("recovery_action", "unknown"))] += 1
        recovery_paths[str(event.get("host_restore_path", "none"))] += 1
        l3_recovery_paths[str(event.get("l3_restore_path", "none"))] += 1
        gpu_direct_reuse_tokens += int(event.get("gpu_reused_tokens", 0))
        host_reuse_tokens += int(event.get("host_reused_tokens", 0))
        l3_reuse_tokens += int(event.get("l3_reused_tokens", 0))
    expected_matched_victims = sorted(
        str(victim["backup_key"])
        for event in failure_manifest["events"]
        for victim in event.get("victims", [])
    )
    observed_matched_victims = sorted(
        str(event.get("backup_key") or "")
        for event in retry_events
        if str(event.get("backup_key") or "") in expected_matched_victims
    )
    matched_route_violations = [
        violation
        for summary in summaries
        for violation in summary.get("matched_route_violations", [])
    ]
    execution_route_plan_violations = [
        violation
        for summary in summaries
        for violation in summary.get("execution_route_plan_violations", [])
    ]
    matched_retry_route_violations = [
        violation
        for summary in summaries
        for violation in summary.get("matched_retry_route_violations", [])
    ]
    matched_timing_violations = [
        {
            "backup_key": str(event.get("backup_key") or ""),
            "matched_event_id": event.get("matched_event_id"),
            "matched_event_injected_offset_s": event.get(
                "matched_event_injected_offset_s"
            ),
            "retry_started_offset_s": event.get("retry_started_offset_s"),
            "retry_after_injection": event.get("retry_after_injection"),
        }
        for event in retry_events
        if str(event.get("backup_key") or "") in expected_matched_victims
        and event.get("retry_after_injection") is not True
    ]
    matched_victim_mode = bool(expected_matched_victims)
    backup_actions: dict[str, int] = defaultdict(int)
    for event in authoritative_backup_hints:
        backup_actions[
            str(event.get("recovery_action", "unknown"))
        ] += 1
    full_scale_replay = (
        args.input_scale == 1.0
        and (
            args.max_input_tokens is None
            or args.max_input_tokens >= args.context_length - 1
        )
        and (
            args.max_output_tokens is None
            or args.max_output_tokens >= args.context_length - 1
        )
    )
    payload = {
        "schema_version": 1,
        "evidence_tier": (
            "sglsim-online-measured"
            if full_scale_replay
            else "sglsim-online-scaled-smoke"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": args.policy,
        "policy_label": POLICY_LABELS[args.policy],
        "request_policy": POLICY_REQUEST_NAMES[args.policy],
        "model": args.model,
        "dp_size": args.dp_size,
        "pp_size": args.pp_size,
        "trace": str(args.trace.resolve()),
        "source_dataset": "semianalysisai/cc-traces-weka-062126",
        "slack_source": args.slack_source,
        "trace_oracle_upper_bound": args.slack_source == "trace_oracle",
        "workflow_filter": (
            "fan_out_then_main_fan_in"
            + (
                f"_and_requests_le_{args.max_requests_per_workflow}"
                if args.max_requests_per_workflow > 0
                else ""
            )
        ),
        "trace_workflow_count": len(trace_workflows),
        "fan_out_fan_in_workflow_count": fan_workflow_count,
        "eligible_workflow_count": len(workflows),
        "max_requests_per_workflow": args.max_requests_per_workflow,
        "selection_order": "lexicographic_workflow_id",
        "workflow_count": len(summaries),
        "workflow_concurrency": args.workflow_concurrency,
        "workflow_launch_policy": WORKFLOW_LAUNCH_POLICY,
        "workflow_launch_stagger_s": WORKFLOW_LAUNCH_STAGGER_S,
        "request_count": total_requests,
        "failure_count_requested": len(failure_manifest["events"]),
        "failure_count_injected": coordinator.injected_failures,
        "failure_manifest_path": str(args.failure_manifest.resolve()),
        "failure_manifest": failure_manifest,
        "execution_route_plan": route_plan_payload,
        "execution_route_plan_audit": {
            "enabled": route_plan_payload is not None,
            "expected_request_count": (
                total_requests if route_plan_payload is not None else 0
            ),
            "observed_request_count": sum(
                1
                for summary in summaries
                for route in summary.get("request_routes", [])
                if int(route.get("route_plan_dp_rank", -1)) >= 0
            ),
            "exact_match": (
                not execution_route_plan_violations
                if route_plan_payload is not None
                else None
            ),
            "violations": execution_route_plan_violations,
        },
        "failure_events": coordinator.snapshot_events(),
        "matched_victim_audit": {
            "enabled": matched_victim_mode,
            "expected_backup_keys": expected_matched_victims,
            "observed_backup_keys": observed_matched_victims,
            "exact_match": (
                observed_matched_victims == expected_matched_victims
                if matched_victim_mode
                else None
            ),
            "route_violations": matched_route_violations,
            "retry_route_violations": matched_retry_route_violations,
            "timing_violations": matched_timing_violations,
        },
        "retry_summary": {
            "count": len(retry_events),
            "restored_tokens": restored_tokens,
            "gpu_direct_reuse_tokens": gpu_direct_reuse_tokens,
            "host_reuse_tokens": host_reuse_tokens,
            "l3_reuse_tokens": l3_reuse_tokens,
            "recompute_tokens": recompute_tokens,
            "capacity_limited_tokens": capacity_limited_tokens,
            "restored_fraction": round(
                restored_tokens / recovery_tokens if recovery_tokens else 0.0,
                6,
            ),
            "mean_latency_s": round(mean(retry_latencies), 6)
            if retry_latencies
            else 0.0,
            "p90_latency_s": round(percentile(retry_latencies, 0.90), 6)
            if retry_latencies
            else 0.0,
            "recovery_actions": dict(sorted(recovery_actions.items())),
            "host_restore_paths": dict(sorted(recovery_paths.items())),
            "l3_restore_paths": dict(sorted(l3_recovery_paths.items())),
        },
        "backup_decision_summary": {
            "count": len(authoritative_backup_hints),
            "ack_event_count": len(backup_ack_events),
            "expected_count": 0 if args.policy == "no_backup" else total_requests,
            "ack_complete": (
                len(authoritative_backup_hints) >= total_requests
                if args.policy != "no_backup"
                else True
            ),
            "l1_tokens": sum(int(event.get("l1_tokens", 0)) for event in authoritative_backup_hints),
            "l2_tokens": sum(int(event.get("l2_tokens", 0)) for event in authoritative_backup_hints),
            "l3_tokens": sum(int(event.get("l3_tokens", 0)) for event in authoritative_backup_hints),
            "recompute_tokens": sum(
                int(event.get("recompute_tokens", 0)) for event in authoritative_backup_hints
            ),
            "actions": dict(sorted(backup_actions.items())),
        },
        "alpha_beta_updates": alpha_beta_updates,
        "rank_alpha_updates": alpha_beta_updates,
        "node_beta_updates": node_beta_updates,
        "final_rank_states": final_status.get("rank_states", {}),
        "final_node_host_states": final_status.get("node_host_states", {}),
        "final_persistent_pool_state": final_status.get(
            "persistent_pool_state", {}
        ),
        "aegis_profile_events": status_events,
        "replay": {
            "slack_source": args.slack_source,
            "slack_evidence_role": (
                "future-completion oracle upper bound; Algorithm 1 bypassed"
                if args.slack_source == "trace_oracle"
                else "causal calibrated Algorithm 1"
            ),
            "input_scale": args.input_scale,
            "context_length": args.context_length,
            "context_overflow_policy": "proportional_prompt_output_scale",
            "context_scaled_requests": sum(
                int(summary.get("context_scaled_requests", 0))
                for summary in summaries
            ),
            "min_context_scale": min(
                (
                    float(summary.get("min_context_scale", 1.0))
                    for summary in summaries
                ),
                default=1.0,
            ),
            "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "arrival_speedup": args.arrival_speedup,
        },
        "wct_summary_s": {
            "mean": round(mean(wcts), 6),
            "p50": round(median(wcts), 6),
            "p90": round(percentile(wcts, 0.90), 6),
            "p99": round(percentile(wcts, 0.99), 6),
            "max": round(max(wcts), 6),
        },
        "workflows": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress_output_path(args.output).unlink(missing_ok=True)
    print(f"wrote {args.output}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Real CC workflow completion times for one backup policy."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--policy", choices=tuple(POLICY_REQUEST_NAMES), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow-concurrency", type=int, default=20)
    parser.add_argument("--max-workflows", type=int, default=0)
    parser.add_argument("--max-requests-per-workflow", type=int, default=0)
    parser.add_argument("--failure-count", type=int, default=2)
    parser.add_argument(
        "--failure-manifest",
        type=Path,
        default=DEFAULT_FAILURE_MANIFEST,
    )
    parser.add_argument(
        "--route-plan",
        type=Path,
        default=None,
        help=(
            "Optional measured request-to-DP route plan. It constrains only "
            "execution source; Algorithm 2 still chooses backup targets."
        ),
    )
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--arrival-speedup", type=float, default=60.0)
    parser.add_argument("--ack-timeout", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument(
        "--slack-calibration",
        type=Path,
        default=DEFAULT_SLACK_CALIBRATION,
    )
    parser.add_argument(
        "--slack-source",
        choices=SLACK_SOURCES,
        default="algorithm1",
    )
    return parser


def main() -> int:
    run_policy(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
