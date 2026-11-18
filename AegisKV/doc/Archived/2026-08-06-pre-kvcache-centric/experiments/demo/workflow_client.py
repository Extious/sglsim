from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from sglsim.srt.simulation.aegis_slack import (
    AegisDagRequest,
    AegisDagSlackTicket,
    AegisOnlineProgressTracker,
    AegisSlackCalibration,
    build_fixed_dag_slack_tickets,
    fixed_dag_request_costs_ms,
)
from sglsim.srt.simulation.aegis_placement import normalize_failure_scope

from .common import paint, request_json, wait_for_hint, wait_for_server


DEFAULT_TRACE = (
    Path(__file__).resolve().parents[1]
    / "presentation"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "weka_subagent_cdf_24w_workload.jsonl"
)
DEFAULT_TRACE_ORACLE = DEFAULT_TRACE.with_name(
    "weka_subagent_cdf_5w_trace_oracle_workload.jsonl"
)
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_SLACK_CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "presentation"
    / "experiments"
    / "alg1_slack_calibration.json"
)
POLICY_REQUEST_NAMES = {
    "aegiskv": "aegiskv_l1_l2",
    "l1_only": "aegiskv_l1_only",
    "l2_only": "aegiskv_l2_only",
    "l3_only": "aegiskv_l3_only",
    "no_backup": "aegiskv_no_backup",
}
SLACK_SOURCES = ("algorithm1", "trace_oracle")
RETRY_ROUTE_MODE_ROUTER_COMMITTED = "router_committed"
RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION = "manifest_fixed_calibration"
RETRY_ROUTE_MODES = (
    RETRY_ROUTE_MODE_ROUTER_COMMITTED,
    RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION,
)


def failure_manifest_retry_route_mode(manifest: object) -> str:
    """Resolve whether a failure manifest is allowed to constrain Router b(r).

    Schema v1 is retained as an explicitly calibration-only compatibility
    contract.  Formal schema v2 manifests default to ``router_committed``:
    they select a failure event and victim request, but never prescribe the
    AegisKV backup or retry rank.
    """

    if not isinstance(manifest, dict):
        return RETRY_ROUTE_MODE_ROUTER_COMMITTED
    # Missing schema is an old in-process test/client contract and therefore
    # stays on the legacy calibration behavior.  Persisted formal manifests
    # must declare schema v2 explicitly.
    schema_version = int(manifest.get("schema_version", 1))
    default = (
        RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
        if schema_version <= 1
        else RETRY_ROUTE_MODE_ROUTER_COMMITTED
    )
    mode = str(manifest.get("retry_route_mode") or default)
    if mode not in RETRY_ROUTE_MODES:
        raise ValueError(
            f"unsupported failure-manifest retry_route_mode={mode!r}"
        )
    return mode


def _trace_token_ids(row: dict[str, Any], token_count: int) -> list[int]:
    """Build a deterministic carrier with stable per-agent prefix identity.

    Real CC publishes token counts and workflow metadata rather than the
    private source-code prompt.  Stable synthetic IDs let SGLSim's real radix
    and prefill paths observe the published length and cross-turn affinity.
    """

    identity = (
        f"{row.get('workflow_id', '')}:{row.get('agent_id', '')}"
    ).encode("utf-8")
    base = zlib.crc32(identity) % 131071
    return [((base + index) % 151936) for index in range(max(token_count, 0))]


def _load_workflow(trace_path: Path, workflow_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            workflow_id = str(row.get("workflow_id", ""))
            if workflow_id.startswith(workflow_prefix):
                rows.append(row)
    if not rows:
        raise ValueError(
            f"No workflow matching {workflow_prefix!r} in {trace_path}"
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("stage_id", 0)),
            float(row.get("arrival_t", 0.0)),
            int(row.get("group_id", 0)),
        ),
    )


def _scaled_tokens(
    row: dict[str, Any],
    *,
    input_scale: float,
    max_input_tokens: int | None,
    max_output_tokens: int | None,
    context_length: int,
) -> tuple[int, int, bool, bool, float]:
    prompt_len = max(int(row.get("prompt_len", 1)), 1)
    output_len = max(int(row.get("output_len", 1)), 1)
    scaled_input_tokens = max(
        int(math.ceil(prompt_len * max(input_scale, 0.0001))),
        128,
    )
    input_tokens = min(
        scaled_input_tokens,
        (
            max_input_tokens
            if max_input_tokens is not None
            else scaled_input_tokens
        ),
    )
    output_tokens = max(
        min(
            output_len,
            max_output_tokens if max_output_tokens is not None else output_len,
        ),
        1,
    )

    # SGLSim follows SGLang's strict boundary: input + requested output must be
    # smaller than context_length. Requests beyond the boundary are scaled as
    # a whole so the prompt/output ratio is retained instead of clipping only
    # the prompt. Requests already within the boundary are not changed.
    token_budget = max(context_length - 1, 2)
    context_scale = min(1.0, token_budget / (input_tokens + output_tokens))
    if context_scale < 1.0:
        scaled_output = round(output_tokens * context_scale)
        output_tokens = min(max(scaled_output, 1), token_budget - 1)
        input_tokens = token_budget - output_tokens

    return (
        input_tokens,
        output_tokens,
        input_tokens < prompt_len,
        output_tokens < output_len,
        context_scale,
    )


def _algorithm1_trace_position(row: dict[str, Any]) -> tuple[str, int]:
    """Map a normalized Real-CC request to its calibration context."""

    role = (
        "main_phase"
        if str(row.get("source_model_role") or "").strip().lower() == "main"
        else "subagent"
    )
    explicit_turn = row.get("turns_completed")
    if explicit_turn is not None:
        return role, max(int(explicit_turn), 1)
    match = re.search(r"-(?:m|t)(\d+)$", str(row.get("rid") or ""))
    return role, (int(match.group(1)) + 1 if match else 1)


def _trace_oracle_simulation_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Translate a precomputed future-completion trace oracle into runtime fields."""

    required = (
        "trace_oracle_slack_ms",
        "trace_oracle_critical_agent",
        "trace_oracle_branch_duration_ms",
        "trace_oracle_semantics",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(
            f"trace-oracle request {row.get('rid')} is missing {missing}; "
            "rebuild it with AegisKV.demo.build_trace_oracle_workload"
        )
    if not bool(row.get("trace_oracle_uses_future_information", False)):
        raise ValueError(
            f"trace-oracle request {row.get('rid')} lacks the future-information marker"
        )
    slack_ms = max(float(row["trace_oracle_slack_ms"]), 0.0)
    duration_ms = max(float(row["trace_oracle_branch_duration_ms"]), 0.0)
    critical = bool(row["trace_oracle_critical_agent"])
    semantics = str(row["trace_oracle_semantics"])
    return {
        "workflow_slack_ms": slack_ms,
        "aegis_slack_lcb_ms": slack_ms,
        "aegis_slack_q50_ms": slack_ms,
        "aegis_slack_q90_ms": slack_ms,
        "aegis_slack_raw_lcb_ms": slack_ms,
        "aegis_slack_target_coverage": 1.0,
        "aegis_slack_calibration_id": "trace-oracle-weka-api-time-v1",
        "aegis_slack_calibration_group": (
            f"stage={int(row.get('stage_id', 0))}|"
            f"agent={str(row.get('agent_id', ''))}"
        ),
        "aegis_slack_source": "trace_oracle_future_api_completion",
        "aegis_slack_stage_id": int(row.get("stage_id", 0)),
        "aegis_slack_stage_width": max(int(row.get("stage_width", 1)), 1),
        "aegis_slack_agent_id": str(row.get("agent_id", "")),
        "aegis_slack_predicted_critical": critical,
        "aegis_critical_probability": 1.0 if critical else 0.0,
        "aegis_duration_lcb_ms": duration_ms,
        "aegis_duration_q50_ms": duration_ms,
        "aegis_duration_q90_ms": duration_ms,
        "aegis_branch_request_count": int(
            row.get("trace_oracle_branch_request_count", 0)
        ),
        "aegis_branch_new_prefill_tokens": int(
            row.get("trace_oracle_branch_input_tokens", 0)
        ),
        "aegis_branch_output_tokens": int(
            row.get("trace_oracle_branch_output_tokens", 0)
        ),
        "aegis_slack_semantics": semantics,
        "aegis_slack_oracle": True,
        "aegis_slack_uses_future_information": True,
    }


class DemoWorkflow:
    def __init__(
        self,
        args: argparse.Namespace,
        rows: list[dict[str, Any]],
        *,
        on_request_completed: Callable[["DemoWorkflow", dict[str, Any]], None]
        | None = None,
    ) -> None:
        self.args = args
        self.rows = rows
        self.started = time.time()
        self.started_monotonic = time.monotonic()
        self.trace_origin = min(float(row.get("arrival_t", 0.0)) for row in rows)
        self.print_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.retry_lock = threading.Lock()
        self.completed: list[dict[str, Any]] = []
        self.agent_finished_monotonic: dict[tuple[int, str], float] = {}
        self.algorithm1_slack_evaluation: list[dict[str, Any]] = []
        self.handled_failure_keys: set[str] = set()
        self.events: list[dict[str, Any]] = []
        self.request_counter = 0
        self.retry_count = 0
        self.on_request_completed = on_request_completed
        self.slack_source = str(
            getattr(args, "slack_source", "algorithm1")
        ).strip()
        if self.slack_source not in SLACK_SOURCES:
            raise ValueError(
                f"unsupported slack_source={self.slack_source!r}; "
                f"expected one of {SLACK_SOURCES}"
            )
        calibration_path = Path(
            getattr(args, "slack_calibration", DEFAULT_SLACK_CALIBRATION)
        )
        self.slack_calibration = (
            AegisSlackCalibration.load(calibration_path)
            if self.slack_source == "algorithm1"
            else None
        )
        # Algorithm 2 needs a monotonic lineage sequence, not the phase-local
        # turn number used by Algorithm 1.  Main-agent request ids restart at
        # ``m0`` after a fan-in phase, so derive one sequence over the complete
        # workflow trace for each agent.
        lineage_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.lineage_worker_seq: dict[str, int] = {}
        for row in rows:
            lineage = (
                str(row.get("workflow_id") or ""),
                str(row.get("agent_id") or ""),
            )
            lineage_counts[lineage] += 1
            self.lineage_worker_seq[str(row["rid"])] = lineage_counts[lineage]
        dag_requests: list[AegisDagRequest] = []
        for row in rows:
            input_tokens, output_tokens, *_ = _scaled_tokens(
                row,
                input_scale=float(getattr(args, "input_scale", 1.0)),
                max_input_tokens=getattr(args, "max_input_tokens", None),
                max_output_tokens=getattr(args, "max_output_tokens", None),
                context_length=int(getattr(args, "context_length", 131072)),
            )
            dag_requests.append(
                AegisDagRequest(
                    request_id=str(row["rid"]),
                    stage_id=int(row.get("stage_id", 0)),
                    agent_id=str(
                        row.get("agent_id")
                        or row.get("workflow_id")
                        or row["rid"]
                    ),
                    order=int(row.get("group_id", 0)),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
        self.slack_tickets: dict[str, AegisDagSlackTicket] = {}
        self.algorithm1_request_cost_ms: dict[str, float] = {}
        self.algorithm1_progress_trackers: dict[
            int, AegisOnlineProgressTracker
        ] = {}
        if self.slack_source == "algorithm1":
            assert self.slack_calibration is not None
            self.slack_tickets = build_fixed_dag_slack_tickets(
                dag_requests,
                calibration=self.slack_calibration,
            )
            self.algorithm1_request_cost_ms = fixed_dag_request_costs_ms(
                dag_requests,
                calibration=self.slack_calibration,
            )
        else:
            for row in rows:
                _trace_oracle_simulation_fields(row)
        progress_requests: dict[
            int, dict[str, list[tuple[str, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        if self.slack_source == "algorithm1":
            for request in sorted(
                dag_requests,
                key=lambda value: (
                    value.stage_id,
                    value.agent_id,
                    value.order,
                    value.request_id,
                ),
            ):
                progress_requests[request.stage_id][request.agent_id].append(
                    (
                        request.request_id,
                        self.algorithm1_request_cost_ms[request.request_id],
                    )
                )
        self.algorithm1_stage_started_monotonic: dict[int, float] = {}
        for stage_id, branches in progress_requests.items():
            static_critical = {
                agent_id
                for agent_id, requests in branches.items()
                if requests
                and self.slack_tickets[requests[0][0]].predicted_critical
            }
            self.algorithm1_progress_trackers[stage_id] = (
                AegisOnlineProgressTracker(
                    request_costs_by_agent=dict(branches),
                    static_critical_agents=static_critical,
                )
            )
        self.policy_start_monotonic = float(
            getattr(args, "policy_start_monotonic", self.started_monotonic)
        )
        self.matched_failure_victims: dict[str, dict[str, Any]] = {}
        raw_route_plan = getattr(args, "execution_route_plan", {})
        self.execution_route_plan = {
            str(rid): int(rank)
            for rid, rank in (
                raw_route_plan.items()
                if isinstance(raw_route_plan, dict)
                else ()
            )
        }
        raw_backup_route_plan = getattr(args, "backup_route_plan", {})
        self.backup_route_plan = {
            str(rid): int(rank)
            for rid, rank in (
                raw_backup_route_plan.items()
                if isinstance(raw_backup_route_plan, dict)
                else ()
            )
        }
        failure_manifest = getattr(args, "failure_manifest", None)
        self.default_failure_scope = "node"
        self.failure_retry_route_mode = failure_manifest_retry_route_mode(
            failure_manifest
        )
        if isinstance(failure_manifest, dict):
            deployment = failure_manifest.get("deployment", {})
            if isinstance(deployment, dict):
                self.default_failure_scope = normalize_failure_scope(
                    deployment.get("failure_scope", "node")
                )
            for event in failure_manifest.get("events", []):
                if not isinstance(event, dict):
                    continue
                for victim in event.get("victims", []):
                    if not isinstance(victim, dict):
                        continue
                    backup_key = str(victim.get("backup_key") or "")
                    if not backup_key:
                        continue
                    self.matched_failure_victims[backup_key] = {
                        **dict(victim),
                        "event_id": str(event.get("event_id") or ""),
                        "offset_s": float(event.get("offset_s", 0.0)),
                        "dp_rank": int(event.get("dp_rank", -1)),
                        "pp_rank": int(event.get("pp_rank", -1)),
                        "failure_scope": normalize_failure_scope(
                            event.get(
                                "failure_scope", self.default_failure_scope
                            )
                        ),
                        "retry_route_mode": self.failure_retry_route_mode,
                    }

    def _log(self, message: str) -> None:
        if getattr(self.args, "quiet", False):
            return
        with self.print_lock:
            print(message, flush=True)

    def _record_event(self, kind: str, **payload: Any) -> None:
        with self.state_lock:
            self.events.append(
                {
                    "wall_time": time.time(),
                    "elapsed_s": round(time.time() - self.started, 3),
                    "kind": kind,
                    **payload,
                }
            )

    def _request_payload(
        self,
        row: dict[str, Any],
        *,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        retry: bool = False,
        failed_dp_rank: int = -1,
        backup_key: str | None = None,
    ) -> dict[str, Any]:
        critical = bool(row.get("critical_agent", False))
        stage_width = max(int(row.get("stage_width", 1)), 1)
        policy = str(getattr(self.args, "policy", "aegiskv"))
        slack_role, turns_completed = _algorithm1_trace_position(row)
        stage_id = int(row.get("stage_id", 0))
        agent_id = str(row.get("agent_id", ""))
        ticket: AegisDagSlackTicket | None = None
        progress_tracker: AegisOnlineProgressTracker | None = None
        online_progress_window_active = False
        online_progress_critical = False
        online_progress_slack_ms: float | None = None
        if self.slack_source == "trace_oracle":
            slack_fields = _trace_oracle_simulation_fields(row)
            effective_critical = bool(
                slack_fields["aegis_slack_predicted_critical"]
            )
        else:
            ticket = self.slack_tickets[str(row["rid"])]
            slack_fields = ticket.to_simulation_fields()
            with self.state_lock:
                progress_tracker = self.algorithm1_progress_trackers.get(stage_id)
                online_progress_window_active = bool(
                    progress_tracker is not None
                    and progress_tracker.ever_dynamically_critical_agents
                )
                online_progress_critical = bool(
                    progress_tracker is not None
                    and agent_id in progress_tracker.dynamically_critical_agents
                )
                online_progress_slack_ms = (
                    None
                    if progress_tracker is None
                    else progress_tracker.final_projected_slack_ms.get(agent_id)
                )
                effective_critical = bool(
                    ticket.predicted_critical
                    if progress_tracker is None
                    else progress_tracker.is_critical(agent_id)
                )
        effective_backup_key = backup_key or str(row["rid"])
        victim = self.matched_failure_victims.get(effective_backup_key, {})
        failure_scope = normalize_failure_scope(
            victim.get("failure_scope", self.default_failure_scope)
        )
        sim: dict[str, Any] = {
            "aegis_policy": POLICY_REQUEST_NAMES[policy],
            "backup_kv": not retry and policy != "no_backup",
            "aegis_backup_key": effective_backup_key,
            "workflow_id": str(row.get("workflow_id", "")),
            "workflow_shape": "real-cc-fan-in-fan-out",
            "agent_id": str(row.get("agent_id", "")),
            "worker_seq": int(self.lineage_worker_seq[str(row["rid"])]),
            "stage_id": int(row.get("stage_id", 0)),
            "stage_width": stage_width,
            "stage_local_agent_index": int(row.get("stage_local_agent_index", 0)),
            "critical_agent": effective_critical,
            "decode_dominant": bool(row.get("decode_dominant", False)),
            "aegis_slack_role": slack_role,
            "aegis_slack_turns_completed": turns_completed,
            "aegis_failure_scope": failure_scope,
            "aegis_failure_pp_rank": int(victim.get("pp_rank", -1)),
        }
        sim.update(slack_fields)
        if online_progress_window_active:
            progress_q50_ms = max(
                float(online_progress_slack_ms or 0.0),
                float(sim.get("aegis_slack_lcb_ms", 0.0)),
            )
            sim.update(
                {
                    "critical_agent": effective_critical,
                    "aegis_slack_predicted_critical": effective_critical,
                    "aegis_critical_probability": (
                        0.90 if effective_critical else 0.10
                    ),
                    "aegis_slack_online_progress_critical": (
                        online_progress_critical
                    ),
                    "aegis_slack_q50_ms": progress_q50_ms,
                    "aegis_slack_q90_ms": max(
                        float(sim.get("aegis_slack_q90_ms", 0.0)),
                        progress_q50_ms,
                    ),
                    "aegis_slack_source": (
                        f"{sim.get('aegis_slack_source', '')}"
                        "+online_progress_last_quarter_replacement"
                    ),
                    "aegis_slack_semantics": (
                        "fixed_dag_cpm_slack_v2+online_progress_v1"
                    ),
                }
            )
            if effective_critical:
                sim.update(
                    {
                        "workflow_slack_ms": 0.0,
                        "aegis_slack_lcb_ms": 0.0,
                        "aegis_slack_q50_ms": 0.0,
                    }
                )
        else:
            sim["aegis_slack_online_progress_critical"] = False
        sim["aegis_trace_critical_agent"] = critical
        sim["aegis_trace_oracle_critical_agent"] = bool(
            row.get("trace_oracle_critical_agent", False)
        )
        if not retry:
            route_plan_rank = self.execution_route_plan.get(str(row["rid"]))
            if route_plan_rank is not None:
                sim["aegis_route_plan_dp_rank"] = int(route_plan_rank)
            backup_route_plan_rank = self.backup_route_plan.get(str(row["rid"]))
            if backup_route_plan_rank is not None:
                sim["aegis_backup_route_plan_dp_rank"] = int(
                    backup_route_plan_rank
                )
            matched_route = self.matched_failure_victims.get(str(row["rid"]))
            if matched_route is not None:
                sim["aegis_matched_failure_victim"] = True
                sim["aegis_matched_failure_expected_execution_dp_rank"] = int(
                    matched_route.get("execution_dp_rank", -1)
                )
                sim["aegis_failure_retry_route_mode"] = (
                    self.failure_retry_route_mode
                )
                if (
                    self.failure_retry_route_mode
                    == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
                ):
                    sim["force_dp_rank"] = int(
                        matched_route.get("execution_dp_rank", -1)
                    )
                    backup_rank = int(
                        matched_route.get("backup_dp_rank", -1)
                    )
                    if backup_rank >= 0:
                        sim["aegis_backup_dp_rank"] = backup_rank
        if retry:
            sim.update(
                {
                    "failover_replay": True,
                    "soft_failure_injected": True,
                    "failed_dp_rank": failed_dp_rank,
                    "pre_failure_total_tokens": input_tokens + output_tokens,
                }
            )
            if (
                victim
                and self.failure_retry_route_mode
                == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
            ):
                sim["aegis_route_plan_retry_dp_rank"] = int(
                    victim.get("backup_dp_rank", -1)
                )
            elif victim:
                sim["aegis_retry_route_source"] = (
                    "router_committed_recovery_hint"
                )
        return {
            "model": self.args.model,
            "user": request_id,
            "input_ids": _trace_token_ids(row, input_tokens),
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "simulation": sim,
        }

    def _run_one(self, row: dict[str, Any]) -> None:
        if self.args.arrival_speedup > 0:
            trace_offset = max(
                float(row.get("arrival_t", 0.0)) - self.trace_origin,
                0.0,
            )
            due = self.started_monotonic + trace_offset / self.args.arrival_speedup
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        (
            input_tokens,
            output_tokens,
            input_adjusted,
            output_adjusted,
            context_scale,
        ) = _scaled_tokens(
            row,
            input_scale=self.args.input_scale,
            max_input_tokens=self.args.max_input_tokens,
            max_output_tokens=self.args.max_output_tokens,
            context_length=self.args.context_length,
        )
        rid = str(row["rid"])
        payload = self._request_payload(
            row,
            request_id=rid,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        tic = time.monotonic()
        response = request_json(
            self.args.base_url,
            "/v1/completions",
            payload=payload,
            timeout=self.args.request_timeout,
        )
        latency = time.monotonic() - tic
        stats = response.get("sim_stats", {})
        execution_rank = int(stats.get("aegis_assigned_dp_rank", -1))
        planned_backup_rank = int(stats.get("aegis_gpu_replica_dp_rank", -1))
        hint = None
        if self.args.policy != "no_backup" and getattr(
            self.args, "wait_for_ack", True
        ):
            hint = wait_for_hint(
                self.args.base_url,
                rid,
                timeout=self.args.ack_timeout,
            )
        with self.state_lock:
            self.request_counter += 1
            stage_id = int(row.get("stage_id", 0))
            stage_started = self.algorithm1_stage_started_monotonic.get(
                stage_id, self.started_monotonic
            )
            progress_decision = None
            if self.slack_source == "algorithm1":
                progress_decision = self.algorithm1_progress_trackers[
                    stage_id
                ].observe_completion(
                    request_id=rid,
                    elapsed_ms=max(
                        (time.monotonic() - stage_started) * 1000.0,
                        0.0,
                    ),
                )
            record = {
                "completion_index": self.request_counter,
                "rid": rid,
                "row": row,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "trace_prompt_tokens": int(row.get("prompt_len", 0)),
                "trace_output_tokens": int(row.get("output_len", 0)),
                "input_adjusted": input_adjusted,
                "output_adjusted": output_adjusted,
                "context_scale": context_scale,
                "execution_dp_rank": execution_rank,
                "planned_backup_dp_rank": planned_backup_rank,
                "route_plan_dp_rank": int(
                    payload["simulation"].get("aegis_route_plan_dp_rank", -1)
                ),
                "route_plan_applied": stats.get(
                    "aegis_route_plan_applied"
                ),
                "exact_admission_state_id": str(
                    stats.get("aegis_exact_admission_state_id", "")
                ),
                "exact_admission_information_mode": str(
                    stats.get(
                        "aegis_exact_admission_information_mode", ""
                    )
                ),
                "exact_observed_unavailable_dp_ranks": [
                    int(rank)
                    for rank in stats.get(
                        "aegis_exact_observed_unavailable_dp_ranks", []
                    )
                ],
                "latency_s": round(latency, 4),
                "completed_monotonic": time.monotonic(),
                "hint": hint,
                "algorithm1_online_progress": {
                    "bypassed": self.slack_source != "algorithm1",
                    "trigger_agent_id": (
                        progress_decision.trigger_agent_id
                        if progress_decision is not None
                        else None
                    ),
                    "trigger_progress_fraction": round(
                        progress_decision.trigger_progress_fraction, 6
                    ),
                    "active_critical_agents": sorted(
                        progress_decision.active_critical_agents
                    ),
                }
                if progress_decision is not None
                else {
                    "bypassed": True,
                    "trigger_agent_id": None,
                    "trigger_progress_fraction": 0.0,
                    "active_critical_agents": [],
                },
                "algorithm1_slack": {
                    key: value
                    for key, value in payload["simulation"].items()
                    if key.startswith("aegis_slack_")
                    or key == "workflow_slack_ms"
                },
            }
            self.completed.append(record)
        self._record_event("request_completed", **record)

        if self.on_request_completed is not None:
            self.on_request_completed(self, record)

        agent = str(row.get("agent_id", "")).split("-")[-1]
        if context_scale < 1.0:
            clip_text = f" context-scaled={context_scale:.4f}"
        else:
            clip_text = " prompt-adjusted" if input_adjusted else ""
            if output_adjusted:
                clip_text += " output-adjusted"
        ack_text = "ACK pending"
        if hint is not None:
            ack_text = (
                f"{hint.get('recovery_action')} "
                f"L1={hint.get('l1_tokens')} L2={hint.get('l2_tokens')} "
                f"alpha={float(hint.get('alpha', 0.0)):.2f} "
                f"beta={float(hint.get('beta', 0.0)):.2f}"
            )
        self._log(
            f"  [{rid:<22}] agent={agent:<5} "
            f"{input_tokens:4d}+{output_tokens:2d} tok  "
            f"route {execution_rank} -> backup {planned_backup_rank}  "
            f"{paint(ack_text, 'green' if hint else 'yellow')}  {latency:.2f}s"
            f"{paint(clip_text, 'yellow') if clip_text else ''}"
        )
        if hint and hint.get("window_update"):
            update = hint["window_update"]
            self._log(
                paint(
                    "    Algorithm 3 window "
                    f"{update.get('window_id')}: "
                    f"alpha {float(update.get('old_alpha', 0)):.2f}"
                    f"->{float(update.get('new_alpha', 0)):.2f}, "
                    f"beta {float(update.get('old_beta', 0)):.2f}"
                    f"->{float(update.get('new_beta', 0)):.2f}",
                    "cyan",
                )
            )
        self._maybe_retry_failed_rank()
        if self.args.request_pause > 0:
            time.sleep(self.args.request_pause)

    def inject_failure_and_retry(self, record: dict[str, Any]) -> None:
        """Inject one deterministic soft failure and include its retry in WCT."""

        failed_dp_rank = int(record["execution_dp_rank"])
        if failed_dp_rank < 0:
            return
        result = request_json(
            self.args.base_url,
            "/simulate/fail_dp",
            payload={"dp_rank": failed_dp_rank},
        )
        if not result.get("accepted", False):
            raise RuntimeError(
                f"soft failure for DP rank {failed_dp_rank} was rejected: {result}"
            )
        self._record_event(
            "failure_injected",
            failed_dp_rank=failed_dp_rank,
            backup_key=str(record["rid"]),
        )
        try:
            self._retry(record, failed_dp_rank)
        finally:
            request_json(
                self.args.base_url,
                "/simulate/recover_dp",
                payload={"dp_rank": failed_dp_rank},
            )
            self._record_event("failure_recovered", failed_dp_rank=failed_dp_rank)

    def _maybe_retry_failed_rank(self) -> None:
        if not self.retry_lock.acquire(blocking=False):
            return
        try:
            try:
                status = request_json(
                    self.args.base_url,
                    "/simulate/failure_status",
                    timeout=float(
                        getattr(self.args, "failure_status_timeout", 5.0)
                    ),
                )
            except (TimeoutError, OSError) as exc:
                # This GET is a best-effort detector, not part of inference.
                # A later request or stage poll retries detection.  Do not
                # discard an hours-long online run because the HTTP control
                # plane transiently missed one diagnostic deadline.
                self._record_event(
                    "failure_status_unavailable",
                    error_type=type(exc).__name__,
                )
                if not self.matched_failure_victims:
                    return
                status = {}
            failed = {
                int(rank)
                for rank in (
                    list(status.get("soft_failed_ranks", []))
                    + list(status.get("hard_failed_ranks", []))
                )
            }
            with self.state_lock:
                if self.matched_failure_victims:
                    policy_elapsed = (
                        time.monotonic() - self.policy_start_monotonic
                    )
                    failure_coordinator = getattr(
                        self.args, "failure_coordinator", None
                    )
                    candidates = [
                        record
                        for record in self.completed
                        if (
                            str(record["rid"])
                            not in self.handled_failure_keys
                            and str(record["rid"])
                            in self.matched_failure_victims
                            and (
                                failure_coordinator.was_injected(
                                    str(
                                        self.matched_failure_victims[
                                            str(record["rid"])
                                        ]["event_id"]
                                    )
                                )
                                if failure_coordinator is not None
                                else float(
                                    self.matched_failure_victims[
                                        str(record["rid"])
                                    ]["offset_s"]
                                )
                                <= policy_elapsed
                            )
                            and int(
                                self.matched_failure_victims[str(record["rid"])][
                                    "dp_rank"
                                ]
                            )
                            == int(record["execution_dp_rank"])
                        )
                    ]
                    selected = sorted(
                        candidates,
                        key=lambda record: (
                            float(
                                self.matched_failure_victims[str(record["rid"])][
                                    "offset_s"
                                ]
                            ),
                            int(record["completion_index"]),
                        ),
                    )
                else:
                    if not failed:
                        return
                    candidates = [
                        record
                        for record in self.completed
                        if int(record["execution_dp_rank"]) in failed
                        and str(record["rid"])
                        not in self.handled_failure_keys
                    ]
                    latest_by_agent: dict[str, dict[str, Any]] = {}
                    for record in candidates:
                        agent_id = str(record["row"].get("agent_id", ""))
                        current = latest_by_agent.get(agent_id)
                        if (
                            current is None
                            or record["completion_index"]
                            > current["completion_index"]
                        ):
                            latest_by_agent[agent_id] = record
                    selected = sorted(
                        latest_by_agent.values(),
                        key=lambda record: int(record["completion_index"]),
                    )
                self.handled_failure_keys.update(
                    str(record["rid"]) for record in selected
                )
            for record in selected:
                if self.matched_failure_victims:
                    failed_dp_rank = int(
                        self.matched_failure_victims[str(record["rid"])][
                            "dp_rank"
                        ]
                    )
                else:
                    failed_dp_rank = int(record["execution_dp_rank"])
                self._retry(record, failed_dp_rank)
        finally:
            self.retry_lock.release()

    def _retry(self, record: dict[str, Any], failed_dp_rank: int) -> None:
        retry_id = f"{record['rid']}-retry"
        matched_spec = self.matched_failure_victims.get(str(record["rid"]))
        matched_event_id = (
            str(matched_spec.get("event_id") or "")
            if matched_spec is not None
            else ""
        )
        failure_coordinator = getattr(self.args, "failure_coordinator", None)
        injected_offset_s = (
            failure_coordinator.injected_offset_s(matched_event_id)
            if failure_coordinator is not None and matched_event_id
            else None
        )
        retry_route_mode = (
            str(matched_spec.get("retry_route_mode"))
            if matched_spec is not None
            else self.failure_retry_route_mode
        )
        expected_retry_dp_rank: int | None
        expected_retry_rank_source: str | None
        if (
            matched_spec is not None
            and retry_route_mode
            == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
        ):
            expected_retry_dp_rank = int(
                matched_spec.get("backup_dp_rank", -1)
            )
            expected_retry_rank_source = "manifest_fixed_calibration"
        elif self.args.policy != "no_backup":
            planned_backup_rank = int(record.get("planned_backup_dp_rank", -1))
            expected_retry_dp_rank = (
                planned_backup_rank if planned_backup_rank >= 0 else None
            )
            expected_retry_rank_source = (
                "original_router_immutable_backup"
                if expected_retry_dp_rank is not None
                else None
            )
        else:
            expected_retry_dp_rank = None
            expected_retry_rank_source = None
        self._log(
            paint(
                f"  FAILURE detected on rank {failed_dp_rank}; retry {record['rid']} ...",
                "red",
            )
        )
        payload = self._request_payload(
            record["row"],
            request_id=retry_id,
            input_tokens=int(record["input_tokens"]),
            output_tokens=int(record["output_tokens"]),
            retry=True,
            failed_dp_rank=failed_dp_rank,
            backup_key=str(record["rid"]),
        )
        tic = time.monotonic()
        retry_started_offset_s = tic - self.policy_start_monotonic
        response = request_json(
            self.args.base_url,
            "/v1/completions",
            payload=payload,
            timeout=self.args.request_timeout,
        )
        latency = time.monotonic() - tic
        stats = response.get("sim_stats", {})
        retry_rank = int(stats.get("aegis_assigned_dp_rank", -1))
        action = str(stats.get("aegis_replay_restore_action", "recompute"))
        restored = int(stats.get("aegis_replay_restored_tokens", 0))
        recompute = int(stats.get("aegis_replay_recompute_tokens", 0))
        gpu_reused = int(stats.get("aegis_peer_gpu_lookup_tokens", 0))
        host_reused = int(stats.get("aegis_host_backup_lookup_tokens", 0))
        l3_reused = int(stats.get("aegis_l3_backup_lookup_tokens", 0))
        gpu_direct_reuse = bool(
            stats.get("aegis_replay_gpu_direct_reuse", False)
        )
        host_restore_path = str(
            stats.get("aegis_replay_host_restore_path", "none")
        )
        l3_restore_path = str(
            stats.get("aegis_replay_l3_restore_path", "none")
        )
        capacity_limited = int(
            stats.get("aegis_replay_capacity_limited_tokens", 0)
        )
        with self.state_lock:
            self.retry_count += 1
        self._record_event(
            "retry_completed",
            rid=retry_id,
            backup_key=record["rid"],
            failed_dp_rank=failed_dp_rank,
            retry_dp_rank=retry_rank,
            recovery_action=action,
            restored_tokens=restored,
            gpu_reused_tokens=gpu_reused,
            host_reused_tokens=host_reused,
            l3_reused_tokens=l3_reused,
            gpu_direct_reuse=gpu_direct_reuse,
            host_restore_path=host_restore_path,
            l3_restore_path=l3_restore_path,
            recompute_tokens=recompute,
            capacity_limited_tokens=capacity_limited,
            matched_event_id=matched_event_id or None,
            matched_expected_retry_dp_rank=(
                expected_retry_dp_rank
            ),
            matched_expected_retry_rank_source=expected_retry_rank_source,
            retry_route_mode=retry_route_mode,
            matched_manifest_offset_s=(
                float(matched_spec.get("offset_s", 0.0))
                if matched_spec is not None
                else None
            ),
            matched_event_injected_offset_s=(
                round(float(injected_offset_s), 6)
                if injected_offset_s is not None
                else None
            ),
            retry_started_offset_s=round(retry_started_offset_s, 6),
            retry_after_injection=(
                retry_started_offset_s >= float(injected_offset_s)
                if matched_spec is not None and injected_offset_s is not None
                else None
            ),
            latency_s=round(latency, 4),
        )
        self._log(
            paint(
                f"  RETRY complete: rank {failed_dp_rank} -> {retry_rank}; "
                f"action={action}, restored={restored}, recompute={recompute}, "
                f"{latency:.2f}s",
                "yellow",
            )
        )

    def run(self) -> dict[str, Any]:
        self.started = time.time()
        self.started_monotonic = time.monotonic()
        by_stage: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            by_stage[int(row.get("stage_id", 0))].append(row)

        workflow_id = str(self.rows[0].get("workflow_id", ""))
        if not getattr(self.args, "quiet", False):
            print(paint("AegisKV Real CC Agent Workflow", "bold"), flush=True)
            print(f"  workflow : {workflow_id}", flush=True)
            print(f"  source   : {self.rows[0].get('source_dataset')}", flush=True)
            print(f"  model    : {self.args.model}", flush=True)
            print(f"  policy   : {self.args.policy}", flush=True)
            print(f"  requests : {len(self.rows)} Real CC trace rows", flush=True)
        timing = (
            f"{self.args.arrival_speedup:g}x trace-time replay"
            if self.args.arrival_speedup > 0
            else "as-fast-as-possible replay"
        )
        if not getattr(self.args, "quiet", False):
            print(f"  timing   : {timing}", flush=True)
        shape = " -> ".join(
            f"stage {stage_id}[{len({str(r.get('agent_id')) for r in rows})} agent]"
            for stage_id, rows in sorted(by_stage.items())
        )
        if not getattr(self.args, "quiet", False):
            print(f"  DAG      : {shape}", flush=True)

        for stage_id, stage_rows in sorted(by_stage.items()):
            by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in stage_rows:
                by_agent[str(row.get("agent_id", ""))].append(row)
            title = (
                f"STAGE {stage_id}: "
                + ("FAN-OUT" if len(by_agent) > 1 else "MAIN / FAN-IN")
                + f" · {len(by_agent)} agent(s), {len(stage_rows)} request(s)"
            )
            self._log("\n" + paint(title, "blue"))
            self._record_event(
                "stage_start",
                stage_id=stage_id,
                agents=len(by_agent),
                requests=len(stage_rows),
            )
            stage_started_monotonic = time.monotonic()
            with self.state_lock:
                self.algorithm1_stage_started_monotonic[
                    stage_id
                ] = stage_started_monotonic
            with ThreadPoolExecutor(max_workers=max(len(by_agent), 1)) as executor:
                futures = [
                    executor.submit(
                        self._run_agent,
                        sorted(
                            agent_rows,
                            key=lambda row: (
                                float(row.get("arrival_t", 0.0)),
                                int(row.get("group_id", 0)),
                            ),
                        ),
                    )
                    for agent_rows in by_agent.values()
                ]
                for future in futures:
                    future.result()
            barrier_monotonic = time.monotonic()
            self._evaluate_algorithm1_stage(
                stage_id=stage_id,
                by_agent=by_agent,
                stage_started_monotonic=stage_started_monotonic,
                barrier_monotonic=barrier_monotonic,
            )
            self._record_event(
                "stage_barrier",
                stage_id=stage_id,
                algorithm1_agents=len(by_agent),
            )
            self._log(paint(f"  stage {stage_id} barrier reached", "cyan"))
            deadline = time.monotonic() + max(self.args.stage_pause, 0.0)
            while time.monotonic() < deadline:
                self._maybe_retry_failed_rank()
                time.sleep(min(0.25, max(deadline - time.monotonic(), 0.0)))

        deadline = time.monotonic() + max(self.args.linger_seconds, 0.0)
        while time.monotonic() < deadline:
            self._maybe_retry_failed_rank()
            time.sleep(min(0.25, max(deadline - time.monotonic(), 0.0)))

        wct_s = time.monotonic() - self.started_monotonic
        if getattr(self.args, "collect_final_status", True):
            status = request_json(
                self.args.base_url,
                "/aegis/status",
                timeout=float(getattr(self.args, "status_timeout", 30.0)),
            )
        else:
            # The policy suite takes one authoritative status snapshot
            # after every workflow and asynchronous backup ACK has drained.
            # Avoid downloading the growing event ledger once per workflow.
            status = {
                "dp_size": int(getattr(self.args, "dp_size", 8)),
                "rank_states": {},
                "recovery_hints": [],
            }
        action_counts: dict[str, int] = defaultdict(int)
        with self.state_lock:
            completed = list(self.completed)
        hints_by_key = {
            str(hint.get("backup_key")): hint
            for hint in status.get("recovery_hints", [])
        }
        for record in completed:
            hint = record.get("hint") or hints_by_key.get(str(record["rid"]))
            if hint is not None:
                action_counts[str(hint.get("recovery_action", "unknown"))] += 1
        request_routes = [
            {
                "rid": str(record["rid"]),
                "completion_index": int(record["completion_index"]),
                "completion_elapsed_s": round(
                    float(record.get("completed_monotonic", 0.0))
                    - self.policy_start_monotonic,
                    6,
                ),
                "execution_dp_rank": int(record["execution_dp_rank"]),
                "backup_dp_rank": int(record["planned_backup_dp_rank"]),
                "route_plan_dp_rank": int(
                    record.get("route_plan_dp_rank", -1)
                ),
                "route_plan_applied": record.get("route_plan_applied"),
                "exact_admission_state_id": str(
                    record.get("exact_admission_state_id", "")
                ),
                "exact_admission_information_mode": str(
                    record.get("exact_admission_information_mode", "")
                ),
                "exact_observed_unavailable_dp_ranks": list(
                    record.get(
                        "exact_observed_unavailable_dp_ranks", []
                    )
                ),
            }
            for record in completed
        ]
        expected_victims = sorted(
            str(row["rid"])
            for row in self.rows
            if str(row["rid"]) in self.matched_failure_victims
        )
        observed_victims = sorted(
            str(event.get("backup_key") or "")
            for event in self.events
            if event.get("kind") == "retry_completed"
            and str(event.get("backup_key") or "") in self.matched_failure_victims
        )
        route_violations: list[dict[str, Any]] = []
        for route in request_routes:
            spec = self.matched_failure_victims.get(str(route["rid"]))
            if spec is None:
                continue
            execution_matches = int(route["execution_dp_rank"]) == int(
                spec.get("execution_dp_rank", -1)
            )
            fixed_calibration_route = (
                str(spec.get("retry_route_mode"))
                == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
            )
            expected_backup_rank = (
                int(spec.get("backup_dp_rank", -1))
                if fixed_calibration_route
                else None
            )
            backup_matches = (
                not fixed_calibration_route
                or self.args.policy == "no_backup"
                or int(route["backup_dp_rank"]) == expected_backup_rank
            )
            if not execution_matches or not backup_matches:
                violation = {
                    "rid": route["rid"],
                    "retry_route_mode": str(
                        spec.get("retry_route_mode")
                    ),
                    "expected_execution_dp_rank": int(
                        spec.get("execution_dp_rank", -1)
                    ),
                    "actual_execution_dp_rank": int(
                        route["execution_dp_rank"]
                    ),
                    "actual_backup_dp_rank": int(route["backup_dp_rank"]),
                }
                if expected_backup_rank is not None:
                    violation["expected_backup_dp_rank"] = expected_backup_rank
                route_violations.append(violation)
        route_plan_violations = [
            {
                "rid": route["rid"],
                "expected_execution_dp_rank": int(
                    route["route_plan_dp_rank"]
                ),
                "actual_execution_dp_rank": int(route["execution_dp_rank"]),
            }
            for route in request_routes
            if int(route.get("route_plan_dp_rank", -1)) >= 0
            and int(route["route_plan_dp_rank"])
            != int(route["execution_dp_rank"])
            and route.get("route_plan_applied") is not False
        ]
        retry_route_violations = [
            {
                "backup_key": str(event.get("backup_key") or ""),
                "expected_retry_dp_rank": int(
                    event.get("matched_expected_retry_dp_rank", -1)
                ),
                "actual_retry_dp_rank": int(event.get("retry_dp_rank", -1)),
            }
            for event in self.events
            if event.get("kind") == "retry_completed"
            and event.get("matched_expected_retry_dp_rank") is not None
            and int(event.get("matched_expected_retry_dp_rank", -1))
            != int(event.get("retry_dp_rank", -1))
        ]
        return {
            "workflow_id": workflow_id,
            "source_dataset": self.rows[0].get("source_dataset"),
            "trace_path": str(self.args.trace.resolve()),
            "dp_size": int(status.get("dp_size", 0)),
            "request_count": self.request_counter,
            "retry_count": self.retry_count,
            "model": self.args.model,
            "policy": self.args.policy,
            "slack_source": self.slack_source,
            "trace_oracle_upper_bound": self.slack_source == "trace_oracle",
            "context_length": self.args.context_length,
            "context_overflow_policy": "proportional_prompt_output_scale",
            "arrival_speedup": self.args.arrival_speedup,
            "context_scaled_requests": sum(
                float(record.get("context_scale", 1.0)) < 1.0
                for record in completed
            ),
            "min_context_scale": min(
                (
                    float(record.get("context_scale", 1.0))
                    for record in completed
                ),
                default=1.0,
            ),
            "input_adjusted_requests": sum(
                bool(record.get("input_adjusted")) for record in completed
            ),
            "output_adjusted_requests": sum(
                bool(record.get("output_adjusted")) for record in completed
            ),
            "trace_prompt_tokens": sum(
                int(record.get("trace_prompt_tokens", 0)) for record in completed
            ),
            "replayed_prompt_tokens": sum(
                int(record.get("input_tokens", 0)) for record in completed
            ),
            "trace_output_tokens": sum(
                int(record.get("trace_output_tokens", 0)) for record in completed
            ),
            "replayed_output_tokens": sum(
                int(record.get("output_tokens", 0)) for record in completed
            ),
            "elapsed_s": round(time.time() - self.started, 3),
            "wct_s": round(wct_s, 6),
            "backup_action_counts": dict(sorted(action_counts.items())),
            "algorithm1_slack_evaluation": list(
                self.algorithm1_slack_evaluation
            ),
            "algorithm1_slack_accuracy": self._algorithm1_accuracy_summary(),
            "request_routes": sorted(request_routes, key=lambda row: row["rid"]),
            "matched_victim_expected": expected_victims,
            "matched_victim_observed": observed_victims,
            "matched_route_violations": route_violations,
            "execution_route_plan_violations": route_plan_violations,
            "matched_retry_route_violations": retry_route_violations,
            "rank_states": status.get("rank_states", {}),
            "events": self.events,
        }

    def _run_agent(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._run_one(row)
        if rows:
            key = (
                int(rows[0].get("stage_id", 0)),
                str(rows[0].get("agent_id", "")),
            )
            with self.state_lock:
                self.agent_finished_monotonic[key] = time.monotonic()

    def _evaluate_algorithm1_stage(
        self,
        *,
        stage_id: int,
        by_agent: dict[str, list[dict[str, Any]]],
        stage_started_monotonic: float,
        barrier_monotonic: float,
    ) -> None:
        if self.slack_source != "algorithm1":
            return
        tolerance_ms = 5.0
        rows: list[dict[str, Any]] = []
        with self.state_lock:
            for agent_id, agent_rows in sorted(by_agent.items()):
                finish = self.agent_finished_monotonic.get((stage_id, agent_id))
                if finish is None:
                    continue
                ticket: AegisDagSlackTicket = self.slack_tickets[
                    str(agent_rows[0]["rid"])
                ]
                actual_duration_ms = max(
                    (finish - stage_started_monotonic) * 1000.0, 0.0
                )
                actual_ms = max((barrier_monotonic - finish) * 1000.0, 0.0)
                predicted = ticket.envelope
                progress_tracker = self.algorithm1_progress_trackers[stage_id]
                online_progress_critical = (
                    agent_id
                    in progress_tracker.final_critical_agents
                    and bool(
                        progress_tracker.ever_dynamically_critical_agents
                    )
                )
                predicted_critical = (
                    agent_id in progress_tracker.final_critical_agents
                )
                actual_critical = actual_ms <= tolerance_ms
                rows.append(
                    {
                        "stage_id": stage_id,
                        "agent_id": agent_id,
                        "stage_width": len(by_agent),
                        "request_count": len(agent_rows),
                        "new_prefill_tokens": ticket.new_prefill_tokens,
                        "output_tokens": ticket.output_tokens,
                        "actual_duration_ms": round(actual_duration_ms, 3),
                        "predicted_duration_lcb_ms": round(
                            ticket.duration_lower_ms, 3
                        ),
                        "predicted_duration_q50_ms": round(
                            ticket.duration_median_ms, 3
                        ),
                        "predicted_duration_q90_ms": round(
                            ticket.duration_upper_ms, 3
                        ),
                        "actual_slack_ms": round(actual_ms, 3),
                        "predicted_lcb_ms": predicted.lower_ms,
                        "predicted_q50_ms": predicted.median_ms,
                        "predicted_q90_ms": predicted.upper_ms,
                        "lower_covered": actual_ms + tolerance_ms
                        >= predicted.lower_ms,
                        "interval_covered": (
                            actual_ms + tolerance_ms >= predicted.lower_ms
                            and actual_ms <= predicted.upper_ms + tolerance_ms
                        ),
                        "actual_critical": actual_critical,
                        "predicted_critical": predicted_critical,
                        "static_predicted_critical": ticket.predicted_critical,
                        "online_progress_critical": online_progress_critical,
                        "online_progress_trigger_elapsed_ms": round(
                            progress_tracker.trigger_elapsed_ms.get(
                                agent_id, 0.0
                            ),
                            3,
                        ),
                        "absolute_error_ms": round(
                            abs(actual_ms - predicted.median_ms), 3
                        ),
                    }
                )
            self.algorithm1_slack_evaluation.extend(rows)

    def _algorithm1_accuracy_summary(self) -> dict[str, Any]:
        if self.slack_source != "algorithm1":
            return {
                "examples": 0,
                "bypassed": True,
                "slack_source": self.slack_source,
                "trace_oracle_upper_bound": True,
                "gate_passed": False,
            }
        rows = list(self.algorithm1_slack_evaluation)
        if not rows:
            assert self.slack_calibration is not None
            return {
                "examples": 0,
                "target_lower_coverage": self.slack_calibration.target_lower_coverage,
                "gate_passed": False,
            }
        absolute_errors = sorted(float(row["absolute_error_ms"]) for row in rows)
        critical_rows = [row for row in rows if row["actual_critical"]]
        lower_coverage = sum(bool(row["lower_covered"]) for row in rows) / len(rows)
        interval_coverage = sum(bool(row["interval_covered"]) for row in rows) / len(rows)
        critical_recall = (
            sum(bool(row["predicted_critical"]) for row in critical_rows)
            / len(critical_rows)
            if critical_rows
            else 1.0
        )
        median_error = absolute_errors[len(absolute_errors) // 2]
        p90_error = absolute_errors[
            min(math.ceil(0.90 * len(absolute_errors)) - 1, len(absolute_errors) - 1)
        ]
        assert self.slack_calibration is not None
        target = self.slack_calibration.target_lower_coverage
        return {
            "examples": len(rows),
            "fanout_examples": sum(int(row["stage_width"]) > 1 for row in rows),
            "target_lower_coverage": target,
            "observed_lower_coverage": round(lower_coverage, 6),
            "observed_interval_coverage": round(interval_coverage, 6),
            "critical_recall": round(critical_recall, 6),
            "median_absolute_error_ms": round(median_error, 3),
            "p90_absolute_error_ms": round(p90_error, 3),
            "nonzero_lcb_fraction": round(
                sum(float(row["predicted_lcb_ms"]) > 0.0 for row in rows)
                / len(rows),
                6,
            ),
            "gate_passed": lower_coverage >= target and critical_recall >= 0.90,
        }


def _write_results(summary: dict[str, Any], results_root: Path) -> Path:
    run_dir = results_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    events = summary.pop("events")
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Real CC fan-in/fan-out workflow")
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--policy",
        choices=tuple(POLICY_REQUEST_NAMES),
        default="aegiskv",
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--workflow-prefix", default="00ca01c4")
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--arrival-speedup",
        type=float,
        default=60.0,
        help="Trace-time acceleration; 1 preserves original arrival timestamps, 0 disables waits.",
    )
    parser.add_argument("--request-pause", type=float, default=0.08)
    parser.add_argument("--stage-pause", type=float, default=1.5)
    parser.add_argument("--linger-seconds", type=float, default=3.0)
    parser.add_argument("--ack-timeout", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--slack-calibration",
        type=Path,
        default=DEFAULT_SLACK_CALIBRATION,
    )
    parser.add_argument(
        "--slack-source",
        choices=SLACK_SOURCES,
        default="algorithm1",
        help=(
            "algorithm1 uses the causal calibrated predictor; trace_oracle "
            "requires preannotated future API completions and is an upper bound"
        ),
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(wait_for_ack=True)
    args = parser.parse_args()

    rows = _load_workflow(args.trace, args.workflow_prefix)
    wait_for_server(args.base_url)
    summary = DemoWorkflow(args, rows).run()
    run_dir = _write_results(summary, args.results_root)
    print("\n" + paint("WORKFLOW COMPLETE", "green"), flush=True)
    print(
        f"  requests={summary['request_count']} retries={summary['retry_count']} "
        f"elapsed={summary['elapsed_s']:.2f}s",
        flush=True,
    )
    print(f"  backup actions={summary['backup_action_counts']}", flush=True)
    print(f"  artifacts={run_dir}", flush=True)


if __name__ == "__main__":
    main()
