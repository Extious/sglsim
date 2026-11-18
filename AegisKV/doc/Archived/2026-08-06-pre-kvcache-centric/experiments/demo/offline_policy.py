"""Logical-time multi-workflow replay for AegisKV policy calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglsim.srt.simulation.offline_cluster import (
    OfflineClusterResult,
    OfflineSchedulerCluster,
)
from sglsim.srt.simulation.offline_coordinator import OfflineEventPriority

from .offline_workflow import OfflineDemoWorkflow


@dataclass(frozen=True)
class OfflinePolicyResult:
    workflows: list[dict[str, Any]]
    cluster: OfflineClusterResult
    launch_events: list[dict[str, Any]]


class OfflinePolicyReplay:
    """Replay the online ordered-predecessor concurrency contract on one DES."""

    def __init__(
        self,
        *,
        cluster: OfflineSchedulerCluster,
        args,
        selected_workflows: list[tuple[str, list[dict[str, Any]]]],
        failure_manifest: dict[str, Any],
        workflow_concurrency: int,
        launch_stagger_s: float,
    ) -> None:
        if not selected_workflows:
            raise ValueError("offline policy replay requires selected workflows")
        self.cluster = cluster
        self.args = args
        self.selected_workflows = list(selected_workflows)
        self.failure_manifest = dict(failure_manifest)
        self.workflow_concurrency = max(int(workflow_concurrency), 1)
        self.launch_stagger_s = max(float(launch_stagger_s), 0.0)
        self.policy_started_at_s = cluster.coordinator.now_s
        self._active: dict[int, OfflineDemoWorkflow] = {}
        self._completed: dict[int, dict[str, Any]] = {}
        self._all_expected_request_ids: set[str] = set()
        self._injected_failure_ids: set[str] = set()
        self._active_failure_ids: set[str] = set()
        self._launch_events: list[dict[str, Any]] = []

    def _schedule_launch(self, index: int, *, at_s: float) -> None:
        workflow_id, _rows = self.selected_workflows[index]
        self.cluster.coordinator.schedule_at(
            float(at_s),
            kind="workflow_launch",
            priority=OfflineEventPriority.COORDINATION,
            payload={"index": index, "workflow_id": workflow_id},
        )

    def _schedule_initial_events(self) -> None:
        initial = min(self.workflow_concurrency, len(self.selected_workflows))
        for index in range(initial):
            self._schedule_launch(
                index,
                at_s=(
                    self.policy_started_at_s
                    + (index + 1) * self.launch_stagger_s
                ),
            )
        for event in self.failure_manifest.get("events", ()):
            self.cluster.schedule_soft_failure(
                event_id=str(event["event_id"]),
                dp_rank=int(event["dp_rank"]),
                offset_s=self.policy_started_at_s + float(event["offset_s"]),
                duration_s=float(event["duration_s"]),
                payload={
                    key: value
                    for key, value in event.items()
                    if key not in {"event_id", "dp_rank", "offset_s", "duration_s"}
                },
            )

    def _launch(self, index: int) -> None:
        if index in self._active or index in self._completed:
            raise RuntimeError(f"workflow index {index} launched twice")
        workflow_id, rows = self.selected_workflows[index]
        workflow = OfflineDemoWorkflow(
            cluster=self.cluster,
            args=self.args,
            rows=rows,
        )
        for event_id in self._active_failure_ids:
            workflow.mark_failure_injected(event_id)
        workflow.start()
        self._active[index] = workflow
        self._launch_events.append(
            {
                "index": index,
                "workflow_id": workflow_id,
                "launched_at_s": self.cluster.coordinator.now_s,
            }
        )

    def _collect_completed_workflows(self) -> None:
        for index, workflow in list(self._active.items()):
            if not workflow.is_complete:
                continue
            self._all_expected_request_ids.update(workflow.expected_request_ids)
            self._completed[index] = workflow.summary()
            del self._active[index]
            successor = index + self.workflow_concurrency
            if successor < len(self.selected_workflows):
                self._schedule_launch(
                    successor,
                    at_s=self.cluster.coordinator.now_s,
                )

    def run(self, *, max_events: int = 10_000_000) -> OfflinePolicyResult:
        self._schedule_initial_events()
        for _ in range(max(int(max_events), 1)):
            if len(self._completed) == len(self.selected_workflows):
                break
            if not self.cluster.coordinator.has_events():
                self.cluster.pump_idle_ranks()
                if not self.cluster.coordinator.has_events():
                    active_ids = [
                        workflow.workflow_id for workflow in self._active.values()
                    ]
                    raise RuntimeError(
                        f"offline policy stalled with active workflows {active_ids}"
                    )
            event = self.cluster.process_next_event()
            if event.kind == "workflow_launch":
                self._launch(int(dict(event.payload or {})["index"]))
            elif event.kind == "failure_inject":
                event_id = str(dict(event.payload or {}).get("event_id", ""))
                self._injected_failure_ids.add(event_id)
                self._active_failure_ids.add(event_id)
            elif event.kind == "failure_recovery_requested":
                event_id = str(dict(event.payload or {}).get("event_id", ""))
                self._active_failure_ids.discard(event_id)
            for workflow in list(self._active.values()):
                workflow.handle_event(event)
                if event.kind == "failure_recovery_requested":
                    workflow.mark_failure_recovered(
                        str(dict(event.payload or {}).get("event_id", ""))
                    )
            self._collect_completed_workflows()
        else:
            raise RuntimeError("offline policy replay exceeded max_events")

        cluster_result = self.cluster.run_until_complete(
            expected_request_ids=self._all_expected_request_ids,
            max_events=max_events,
        )
        return OfflinePolicyResult(
            workflows=[self._completed[index] for index in sorted(self._completed)],
            cluster=cluster_result,
            launch_events=list(self._launch_events),
        )
