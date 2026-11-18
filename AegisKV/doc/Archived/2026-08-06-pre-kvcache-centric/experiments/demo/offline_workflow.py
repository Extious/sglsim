"""Logical-time driver for the existing Real-CC AegisKV workflow payloads."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sglsim.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.simulation.calibration_metrics import metric_summary
from sglsim.srt.simulation.offline_cluster import (
    OfflineClusterResult,
    OfflineSchedulerCluster,
)
from sglsim.srt.simulation.offline_coordinator import OfflineEventPriority

from .workflow_client import (
    RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION,
    DemoWorkflow,
    _scaled_tokens,
)


@dataclass(frozen=True)
class OfflineWorkflowResult:
    workflow: dict[str, Any]
    cluster: OfflineClusterResult


class OfflineDemoWorkflow:
    """Run one DemoWorkflow DAG without HTTP, threads, or wall sleeps.

    Payload construction, Algorithm 1 tickets/progress, routing annotations,
    and AegisKV request fields come directly from :class:`DemoWorkflow`.
    Only the arrival/barrier driver is replaced by parent logical events.
    """

    def __init__(
        self,
        *,
        cluster: OfflineSchedulerCluster,
        args,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            raise ValueError("offline workflow requires at least one trace row")
        self.cluster = cluster
        self.args = args
        self.rows = list(rows)
        self.builder = DemoWorkflow(args, self.rows)
        self.workflow_id = str(self.rows[0].get("workflow_id", ""))
        self.started_at_s = cluster.coordinator.now_s
        # External online-driver costs (first tokenizer/control-plane setup and
        # per-response HTTP/client handling) are not scheduler service time.
        # They remain explicit calibration knobs at the workflow-driver layer.
        self.driver_startup_overhead_s = max(
            float(getattr(args, "driver_startup_overhead_s", 0.0)), 0.0
        )
        self.driver_request_overhead_s = max(
            float(getattr(args, "driver_request_overhead_s", 0.0)), 0.0
        )
        # The online workflow uses one concurrent worker per stage-local agent.
        # External response/HTTP handling serializes consecutive turns of one
        # agent, but not sibling agents in the same fan-out stage.
        self._driver_available_at_s: dict[str, float] = {}
        raw_completion_order = getattr(args, "completion_order_plan", {})
        selected_ids = {str(row["rid"]) for row in self.rows}
        self._planned_completion_order = {
            rid: int(index)
            for rid, index in (
                raw_completion_order.items()
                if isinstance(raw_completion_order, dict)
                else ()
            )
            if rid in selected_ids
        }
        if self._planned_completion_order:
            if set(self._planned_completion_order) != selected_ids:
                missing = sorted(selected_ids - set(self._planned_completion_order))
                raise ValueError(
                    "completion-order plan does not cover workflow "
                    f"{self.workflow_id}; first missing={missing[0]}"
                )
            expected = list(range(1, len(selected_ids) + 1))
            if sorted(self._planned_completion_order.values()) != expected:
                raise ValueError(
                    f"workflow {self.workflow_id} completion order is not contiguous"
                )
        self._next_planned_completion_index = 1
        self._pending_ordered_messages: dict[str, dict[str, Any]] = {}
        self._last_planned_delivery_at_s = self.started_at_s
        self.trace_origin = min(float(row.get("arrival_t", 0.0)) for row in rows)
        self._stages: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            self._stages[int(row.get("stage_id", 0))].append(row)
        self._stage_ids = sorted(self._stages)
        self._stage_position = 0
        self._agent_rows: dict[str, list[dict[str, Any]]] = {}
        self._agent_positions: dict[str, int] = {}
        self._stage_completed_agents: set[str] = set()
        self._active_stage_id: int | None = None
        self._stage_started_at_s = self.started_at_s
        self._message_cursor = len(cluster.detokenizer_messages)
        self._completed_records: list[dict[str, Any]] = []
        self._completed_ids: set[str] = set()
        self._retry_records: list[dict[str, Any]] = []
        self._retry_specs: dict[str, dict[str, Any]] = {}
        self._retry_completed_ids: set[str] = set()
        self._retry_barriers: dict[str, dict[str, Any]] = {}
        self._retry_to_barrier: dict[str, str] = {}
        self._injected_failure_ids: set[str] = set()
        self._recovered_failure_ids: set[str] = set()
        manifest = getattr(args, "failure_manifest", {})
        self._relevant_failure_ids = {
            str(event.get("event_id", ""))
            for event in (
                manifest.get("events", ()) if isinstance(manifest, dict) else ()
            )
            if isinstance(event, dict)
            and any(
                str(victim.get("backup_key", ""))
                in {str(row["rid"]) for row in self.rows}
                for victim in event.get("victims", ())
                if isinstance(victim, dict)
            )
        }
        self._finished = False
        self._started = False

    def _due_time(self, row: dict[str, Any], *, earliest_s: float) -> float:
        speedup = float(getattr(self.args, "arrival_speedup", 0.0))
        if speedup <= 0.0:
            return float(earliest_s)
        trace_offset = max(
            float(row.get("arrival_t", 0.0)) - self.trace_origin,
            0.0,
        )
        return max(
            float(earliest_s),
            self.started_at_s + trace_offset / speedup,
        )

    def _tokenized_request(
        self,
        row: dict[str, Any],
        *,
        request_id: str | None = None,
        retry: bool = False,
        failed_dp_rank: int = -1,
        backup_key: str | None = None,
    ) -> TokenizedGenerateReqInput:
        input_tokens, output_tokens, *_ = _scaled_tokens(
            row,
            input_scale=float(getattr(self.args, "input_scale", 1.0)),
            max_input_tokens=getattr(self.args, "max_input_tokens", None),
            max_output_tokens=getattr(self.args, "max_output_tokens", None),
            context_length=int(getattr(self.args, "context_length", 131072)),
        )
        payload = self.builder._request_payload(
            row,
            request_id=request_id or str(row["rid"]),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry=retry,
            failed_dp_rank=failed_dp_rank,
            backup_key=backup_key,
        )
        return TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid=str(payload["user"]),
                input_ids=list(payload["input_ids"]),
                max_new_tokens=int(payload["max_tokens"]),
                simulation=dict(payload["simulation"]),
            )
        )

    def _schedule_row(self, row: dict[str, Any], *, earliest_s: float) -> None:
        request = self._tokenized_request(row)
        self.cluster.schedule_request_arrival(
            request,
            at_s=self._due_time(row, earliest_s=earliest_s),
        )

    def _start_stage(self, stage_id: int) -> None:
        self._active_stage_id = int(stage_id)
        self._stage_started_at_s = self.cluster.coordinator.now_s
        self.builder.algorithm1_stage_started_monotonic[stage_id] = (
            self._stage_started_at_s
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._stages[stage_id]:
            grouped[str(row.get("agent_id", ""))].append(row)
        self._agent_rows = {
            agent: sorted(
                agent_rows,
                key=lambda row: (
                    float(row.get("arrival_t", 0.0)),
                    int(row.get("group_id", 0)),
                    str(row.get("rid", "")),
                ),
            )
            for agent, agent_rows in grouped.items()
        }
        self._agent_positions = {agent: 0 for agent in self._agent_rows}
        self._stage_completed_agents = set()
        initial_floor = self._stage_started_at_s
        if self._stage_position == 0:
            initial_floor += self.driver_startup_overhead_s
        for rows in self._agent_rows.values():
            self._schedule_row(rows[0], earliest_s=initial_floor)

    def _record_request_completion(self, message: dict[str, Any]) -> None:
        rid = str(message.get("rid", ""))
        if rid in self._completed_ids:
            return
        row = next(row for row in self.rows if str(row["rid"]) == rid)
        # The message is consumed only by a parent response-delivery event, so
        # this cursor includes the explicit external driver overhead.
        completed_at = self.cluster.coordinator.now_s
        stage_id = int(row.get("stage_id", 0))
        decision = None
        if self.builder.slack_source == "algorithm1":
            tracker = self.builder.algorithm1_progress_trackers[stage_id]
            decision = tracker.observe_completion(
                request_id=rid,
                elapsed_ms=max(
                    (completed_at - self._stage_started_at_s) * 1000.0,
                    0.0,
                ),
            )
        record = {
            "completion_index": len(self._completed_records) + 1,
            "rid": rid,
            "row": row,
            "completed_at_s": completed_at,
            "execution_dp_rank": int(message.get("aegis_assigned_dp_rank", -1)),
            "planned_backup_dp_rank": int(
                message.get("aegis_gpu_replica_dp_rank", -1)
            ),
            "route_plan_dp_rank": int(
                message.get("aegis_route_plan_dp_rank", -1)
            ),
            "route_plan_applied": message.get("aegis_route_plan_applied"),
            "exact_admission_state_id": str(
                message.get("aegis_exact_admission_state_id", "")
            ),
            "exact_admission_information_mode": str(
                message.get("aegis_exact_admission_information_mode", "")
            ),
            "exact_observed_unavailable_dp_ranks": [
                int(rank)
                for rank in message.get(
                    "aegis_exact_observed_unavailable_dp_ranks", []
                )
            ],
            "latency_s": max(
                completed_at - float(message.get("created_time", completed_at)),
                0.0,
            ),
            "algorithm1_online_progress": {
                "bypassed": False,
                "trigger_agent_id": decision.trigger_agent_id,
                "trigger_progress_fraction": round(
                    decision.trigger_progress_fraction, 6
                ),
                "active_critical_agents": sorted(decision.active_critical_agents),
            }
            if decision is not None
            else {
                "bypassed": True,
                "trigger_agent_id": None,
                "trigger_progress_fraction": 0.0,
                "active_critical_agents": [],
            },
        }
        self._completed_records.append(record)
        self._completed_ids.add(rid)
        retry_ids = self._maybe_schedule_matched_retries()
        if retry_ids:
            # Online ``DemoWorkflow._run_one`` calls the synchronous retry
            # poll before returning to its agent lane.  The detecting lane
            # therefore cannot launch a successor or satisfy a stage barrier
            # until every retry selected at this poll point has completed.
            barrier_id = f"request-completion:{rid}"
            self._retry_barriers[barrier_id] = {
                "pending_retry_ids": set(retry_ids),
                "row": row,
                "detected_at_s": completed_at,
            }
            for retry_id in retry_ids:
                self._retry_to_barrier[retry_id] = barrier_id
            return

        self._advance_agent_after_completion(row, completed_at=completed_at)

    def _advance_agent_after_completion(
        self,
        row: dict[str, Any],
        *,
        completed_at: float,
    ) -> None:
        """Resume the online-equivalent agent lane after its retry poll."""

        stage_id = int(row.get("stage_id", 0))
        agent = str(row.get("agent_id", ""))
        position = self._agent_positions[agent] + 1
        self._agent_positions[agent] = position
        agent_rows = self._agent_rows[agent]
        if position < len(agent_rows):
            self._schedule_row(agent_rows[position], earliest_s=completed_at)
            return

        self._stage_completed_agents.add(agent)
        self.builder.agent_finished_monotonic[(stage_id, agent)] = completed_at
        if self._stage_completed_agents != set(self._agent_rows):
            return
        self._complete_stage()

    def _schedule_retry(
        self,
        record: dict[str, Any],
        *,
        failed_dp_rank: int,
        event_id: str,
    ) -> str | None:
        backup_key = str(record["rid"])
        retry_id = f"{backup_key}-retry"
        if retry_id in self._retry_specs:
            return None
        victim = self.builder.matched_failure_victims.get(backup_key, {})
        retry_route_mode = str(
            victim.get(
                "retry_route_mode",
                self.builder.failure_retry_route_mode,
            )
        )
        if (
            retry_route_mode
            == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
        ):
            expected_retry_dp_rank: int | None = int(
                victim.get("backup_dp_rank", -1)
            )
            expected_retry_rank_source: str | None = (
                "manifest_fixed_calibration"
            )
        elif str(getattr(self.args, "policy", "aegiskv")) != "no_backup":
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
        request = self._tokenized_request(
            record["row"],
            request_id=retry_id,
            retry=True,
            failed_dp_rank=failed_dp_rank,
            backup_key=backup_key,
        )
        scheduled_at = self.cluster.coordinator.now_s
        self._retry_specs[retry_id] = {
            "rid": retry_id,
            "backup_key": backup_key,
            "failed_dp_rank": failed_dp_rank,
            "event_id": event_id,
            "retry_route_mode": retry_route_mode,
            "expected_retry_dp_rank": expected_retry_dp_rank,
            "expected_retry_rank_source": expected_retry_rank_source,
            "scheduled_at_s": scheduled_at,
        }
        self.cluster.schedule_request_arrival(request, at_s=scheduled_at)
        return retry_id

    def _maybe_schedule_matched_retries(self) -> list[str]:
        scheduled: list[str] = []
        for record in self._completed_records:
            backup_key = str(record["rid"])
            victim = self.builder.matched_failure_victims.get(backup_key)
            if victim is None:
                continue
            event_id = str(victim.get("event_id", ""))
            if event_id not in self._injected_failure_ids:
                continue
            retry_id = self._schedule_retry(
                record,
                failed_dp_rank=int(victim.get("dp_rank", -1)),
                event_id=event_id,
            )
            if retry_id is not None:
                scheduled.append(retry_id)
        return scheduled

    def _record_retry_completion(self, message: dict[str, Any]) -> None:
        retry_id = str(message.get("rid", ""))
        if retry_id in self._retry_completed_ids:
            return
        spec = self._retry_specs[retry_id]
        completed_at = self.cluster.coordinator.now_s
        self._retry_records.append(
            {
                **spec,
                "completed_at_s": completed_at,
                "retry_dp_rank": int(message.get("aegis_assigned_dp_rank", -1)),
                "recovery_action": str(
                    message.get("aegis_replay_restore_action", "recompute")
                ),
                "restored_tokens": int(
                    message.get("aegis_replay_restored_tokens", 0)
                ),
                "recompute_tokens": int(
                    message.get("aegis_replay_recompute_tokens", 0)
                ),
                "latency_s": max(completed_at - float(spec["scheduled_at_s"]), 0.0),
            }
        )
        self._retry_completed_ids.add(retry_id)
        barrier_id = self._retry_to_barrier.pop(retry_id, None)
        if barrier_id is None:
            return
        barrier = self._retry_barriers[barrier_id]
        barrier["pending_retry_ids"].discard(retry_id)
        if barrier["pending_retry_ids"]:
            return
        del self._retry_barriers[barrier_id]
        self._advance_agent_after_completion(
            barrier["row"],
            completed_at=completed_at,
        )

    def _complete_stage(self) -> None:
        assert self._active_stage_id is not None
        stage_id = self._active_stage_id
        barrier_at = max(
            self.builder.agent_finished_monotonic[(stage_id, agent)]
            for agent in self._agent_rows
        )
        self.builder._evaluate_algorithm1_stage(
            stage_id=stage_id,
            by_agent=self._agent_rows,
            stage_started_monotonic=self._stage_started_at_s,
            barrier_monotonic=barrier_at,
        )
        self._stage_position += 1
        if self._stage_position >= len(self._stage_ids):
            self._finished = True
            return
        pause = max(float(getattr(self.args, "stage_pause", 0.0)), 0.0)
        self.cluster.coordinator.schedule_at(
            barrier_at + pause,
            kind="workflow_stage_start",
            priority=OfflineEventPriority.COORDINATION,
            payload={
                "workflow_id": self.workflow_id,
                "stage_id": self._stage_ids[self._stage_position],
            },
        )

    def _schedule_response_delivery(self, message: dict[str, Any]) -> float:
        rid = str(message.get("rid", ""))
        server_completed_at = float(
            message.get("last_event_time", self.cluster.coordinator.now_s)
        )
        if rid in self._retry_specs:
            lane = f"retry:{self._retry_specs[rid]['backup_key']}"
        else:
            row = next(row for row in self.rows if str(row["rid"]) == rid)
            lane = (
                f"stage:{int(row.get('stage_id', 0))}:"
                f"agent:{str(row.get('agent_id', ''))}"
            )
        lane_available_at = self._driver_available_at_s.get(
            lane,
            self.started_at_s,
        )
        delivery_at = max(
            self.cluster.coordinator.now_s,
            server_completed_at,
            lane_available_at,
        ) + self.driver_request_overhead_s
        if rid in self._planned_completion_order:
            # Preserve only causal order.  Different online response workers
            # remain concurrent; the calibration plan must not serialize their
            # per-lane HTTP overhead.
            delivery_at = max(
                delivery_at,
                self._last_planned_delivery_at_s + 1e-9,
            )
        self._driver_available_at_s[lane] = delivery_at
        if rid in self._planned_completion_order:
            self._last_planned_delivery_at_s = delivery_at
        self.cluster.coordinator.schedule_at(
            delivery_at,
            kind="workflow_response_delivery",
            priority=OfflineEventPriority.COORDINATION,
            payload={
                "workflow_id": self.workflow_id,
                "message": dict(message),
            },
        )
        return delivery_at

    def _drain_planned_response_order(self) -> None:
        if not self._planned_completion_order:
            return
        by_index = {
            index: rid for rid, index in self._planned_completion_order.items()
        }
        while self._next_planned_completion_index in by_index:
            rid = by_index[self._next_planned_completion_index]
            message = self._pending_ordered_messages.get(rid)
            if message is None:
                break
            del self._pending_ordered_messages[rid]
            self._schedule_response_delivery(message)
            self._next_planned_completion_index += 1

    def _consume_new_messages(self) -> None:
        messages = self.cluster.detokenizer_messages[self._message_cursor :]
        self._message_cursor = len(self.cluster.detokenizer_messages)
        for message in messages:
            if not bool(message.get("finished", False)):
                continue
            rid = str(message.get("rid", ""))
            if rid not in self._retry_specs and not any(
                str(row["rid"]) == rid for row in self.rows
            ):
                continue
            if rid in self._planned_completion_order:
                self._pending_ordered_messages[rid] = dict(message)
                self._drain_planned_response_order()
            else:
                self._schedule_response_delivery(dict(message))

    def _workflow_complete(self) -> bool:
        return bool(
            self._finished
            and set(self._retry_specs) <= self._retry_completed_ids
        )

    @property
    def is_complete(self) -> bool:
        return self._workflow_complete()

    @property
    def expected_request_ids(self) -> set[str]:
        return {str(row["rid"]) for row in self.rows} | set(self._retry_specs)

    def start(self) -> None:
        if self._started:
            raise RuntimeError(f"offline workflow {self.workflow_id} already started")
        self._started = True
        self._start_stage(self._stage_ids[0])

    def handle_event(self, event) -> None:
        """Consume one already-processed parent event for this workflow."""

        if not self._started:
            return
        self._consume_new_messages()
        if event.kind == "failure_inject":
            payload = dict(event.payload or {})
            self.mark_failure_injected(str(payload.get("event_id", "")))
        elif event.kind == "workflow_stage_start":
            payload = dict(event.payload or {})
            if payload.get("workflow_id") == self.workflow_id:
                self._start_stage(int(payload["stage_id"]))
        elif event.kind == "workflow_response_delivery":
            payload = dict(event.payload or {})
            if payload.get("workflow_id") != self.workflow_id:
                return
            message = dict(payload.get("message") or {})
            rid = str(message.get("rid", ""))
            if rid in self._retry_specs:
                self._record_retry_completion(message)
            elif any(str(row["rid"]) == rid for row in self.rows):
                self._record_request_completion(message)

    def mark_failure_injected(self, event_id: str) -> None:
        self._injected_failure_ids.add(str(event_id))

    def mark_failure_recovered(self, event_id: str) -> None:
        """Record recovery without erasing matched injection history.

        Online ``FailureManifestCoordinator.was_injected`` remains true after
        recovery.  A completed victim is still selected at the next request
        completion poll, even when a short outage has already recovered.
        """

        self._recovered_failure_ids.add(str(event_id))

    def summary(self) -> dict[str, Any]:
        if not self._workflow_complete():
            raise RuntimeError(f"offline workflow {self.workflow_id} is incomplete")
        wct_s = max(
            (
                record["completed_at_s"]
                for record in self._completed_records + self._retry_records
            ),
            default=self.started_at_s,
        ) - self.started_at_s
        latencies = [record["latency_s"] for record in self._completed_records]
        return {
            "workflow_id": self.workflow_id,
            "simulation_time_mode": "logical",
            "request_count": len(self._completed_records),
            "retry_count": len(self._retry_records),
            "model": str(getattr(self.args, "model", "")),
            "policy": str(getattr(self.args, "policy", "aegiskv")),
            "slack_source": self.builder.slack_source,
            "trace_oracle_upper_bound": (
                self.builder.slack_source == "trace_oracle"
            ),
            "started_at_s": self.started_at_s,
            "completed_at_s": self.started_at_s + wct_s,
            "wct_s": wct_s,
            "driver_calibration": {
                "startup_overhead_s": self.driver_startup_overhead_s,
                "request_overhead_s": self.driver_request_overhead_s,
                "scope": "external workflow driver only; excluded from scheduler service metrics",
                "completion_order_replay": bool(
                    self._planned_completion_order
                ),
            },
            "request_latency_summary_s": metric_summary(latencies),
            "request_routes": [
                {
                    "rid": record["rid"],
                    "completion_index": record["completion_index"],
                    "completion_elapsed_s": record["completed_at_s"]
                    - self.started_at_s,
                    "execution_dp_rank": record["execution_dp_rank"],
                    "backup_dp_rank": record["planned_backup_dp_rank"],
                    "route_plan_dp_rank": record["route_plan_dp_rank"],
                    "route_plan_applied": record["route_plan_applied"],
                    "exact_admission_state_id": record[
                        "exact_admission_state_id"
                    ],
                    "exact_admission_information_mode": record[
                        "exact_admission_information_mode"
                    ],
                    "exact_observed_unavailable_dp_ranks": list(
                        record["exact_observed_unavailable_dp_ranks"]
                    ),
                }
                for record in self._completed_records
            ],
            "retry_events": list(self._retry_records),
            "algorithm1_slack_evaluation": list(
                self.builder.algorithm1_slack_evaluation
            ),
            "algorithm1_slack_accuracy": self.builder._algorithm1_accuracy_summary(),
        }

    def run(self, *, max_events: int = 1_000_000) -> OfflineWorkflowResult:
        self.start()
        for _ in range(max(int(max_events), 1)):
            if self._workflow_complete():
                break
            if not self.cluster.coordinator.has_events():
                self.cluster.pump_idle_ranks()
                if not self.cluster.coordinator.has_events():
                    missing = sorted(
                        str(row["rid"])
                        for row in self.rows
                        if str(row["rid"]) not in self._completed_ids
                    )
                    raise RuntimeError(
                        f"offline workflow stalled with requests {missing[:8]}"
                    )
            event = self.cluster.process_next_event()
            self.handle_event(event)
        else:
            raise RuntimeError("offline workflow exceeded max_events")

        cluster_result = self.cluster.run_until_complete(
            expected_request_ids={str(row["rid"]) for row in self.rows}
            | set(self._retry_specs),
            max_events=max_events,
        )
        self._consume_new_messages()
        return OfflineWorkflowResult(workflow=self.summary(), cluster=cluster_result)
