"""Causal parent-side exact rolling planner for offline AegisKV replay."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sglsim.srt.managers.io_struct import TokenizedGenerateReqInput
from sglsim.srt.simulation.aegis_exact_contract import CertifiedExactWindowPlan
from sglsim.srt.simulation.aegis_exact_optimizer import (
    ExactObjectiveWeights,
    ExactTopology,
)
from sglsim.srt.simulation.aegis_exact_rolling import (
    ExactRollingState,
    ExactWindowRequest,
    solve_causal_exact_rolling_window,
)
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster


class CausalExactRollingPlanner:
    """Solve one finite, state-matched request-admission window at a time.

    The class lives in the offline demo driver on purpose.  It imports HiGHS
    through ``aegis_exact_rolling``; Scheduler workers and online serving never
    import a solver.  Each solve uses only the request that has arrived plus the
    controller's causal stock/flow snapshot.  The realized future failure
    manifest is never passed to this class.
    """

    def __init__(
        self,
        *,
        topology: ExactTopology,
        weights: ExactObjectiveWeights,
        failure_hazard_probability: float,
        failure_scope: str,
        chunk_tokens: int,
        recompute_setup_s: float,
        recompute_s_per_token: float,
        protected_restore_setup_s: float,
        protected_restore_s_per_token: float,
        link_window_duration_s: float,
        kv_bytes_per_token_per_stage: int,
        time_limit_s: float,
        deterministic_seed: int,
        first_window_id: int,
        first_route_version: int,
        journal_path: Path,
    ) -> None:
        if first_window_id < 0 or first_route_version <= 0:
            raise ValueError("causal exact starting versions are invalid")
        self.topology = topology
        self.weights = weights
        self.failure_hazard_probability = float(failure_hazard_probability)
        self.failure_scope = str(failure_scope)
        self.chunk_tokens = int(chunk_tokens)
        self.recompute_setup_s = float(recompute_setup_s)
        self.recompute_s_per_token = float(recompute_s_per_token)
        self.protected_restore_setup_s = float(protected_restore_setup_s)
        self.protected_restore_s_per_token = float(protected_restore_s_per_token)
        self.link_window_duration_s = float(link_window_duration_s)
        self.kv_bytes_per_token_per_stage = int(
            kv_bytes_per_token_per_stage
        )
        self.time_limit_s = float(time_limit_s)
        self.deterministic_seed = int(deterministic_seed)
        self._next_window_id = int(first_window_id)
        self._next_route_version = int(first_route_version)
        self.journal_path = journal_path.resolve()
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        if self.journal_path.exists() and self.journal_path.stat().st_size > 0:
            raise FileExistsError(
                f"causal exact journal already exists: {self.journal_path}"
            )
        self.history: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    @staticmethod
    def _finite_fraction(value: object, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed):
            return default
        return min(max(parsed, 0.0), 1.0)

    def _append_journal(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def __call__(
        self,
        request: TokenizedGenerateReqInput,
        cluster: OfflineSchedulerCluster,
    ) -> CertifiedExactWindowPlan:
        with self._lock:
            sim = request.sampling_params.custom_params.setdefault(
                "simulation", {}
            )
            if bool(sim.get("failover_replay", False)):
                raise ValueError("retry requests reuse committed recovery metadata")
            source_dp_rank = cluster.dp_controller.preview_causal_execution_rank(
                request
            )
            snapshot = cluster.dp_controller.exact_rolling_state_snapshot(
                logical_time_s=cluster.coordinator.now_s,
                link_window_duration_s=self.link_window_duration_s,
                kv_bytes_per_token_per_stage=(
                    self.kv_bytes_per_token_per_stage
                ),
            )
            workflow_id = str(sim.get("workflow_id") or "")
            if not workflow_id:
                raise ValueError("causal exact request lacks workflow_id")
            critical_weight = self._finite_fraction(
                sim.get("aegis_critical_probability"),
                1.0 if bool(sim.get("critical_agent", False)) else 0.0,
            )
            slack_ms = max(
                float(
                    sim.get(
                        "aegis_slack_lcb_ms",
                        sim.get("workflow_slack_ms", 0.0),
                    )
                    or 0.0
                ),
                0.0,
            )
            rolling_state = ExactRollingState(
                logical_time_s=float(snapshot["logical_time_s"]),
                window_id=self._next_window_id,
                route_version=self._next_route_version,
                unavailable_dp_ranks=tuple(
                    snapshot["unavailable_dp_ranks"]
                ),
                resident_hbm_tokens=tuple(
                    tuple(row) for row in snapshot["resident_hbm_tokens"]
                ),
                resident_host_tokens_by_dp=tuple(
                    snapshot["resident_host_tokens_by_dp"]
                ),
                resident_host_tokens_by_node=tuple(
                    tuple(row)
                    for row in snapshot["resident_host_tokens_by_node"]
                ),
                used_link_bandwidth_bytes=tuple(
                    tuple(row)
                    for row in snapshot["used_link_bandwidth_bytes"]
                ),
                current_hbm_budget_tokens=tuple(
                    snapshot["current_hbm_budget_tokens"]
                ),
                current_host_budget_tokens=tuple(
                    snapshot["current_host_budget_tokens"]
                ),
            )
            solve = solve_causal_exact_rolling_window(
                instance_id=(
                    f"causal-{self._next_window_id}-{request.rid}"
                ),
                requests=(
                    ExactWindowRequest(
                        request_id=request.rid,
                        workflow_id=workflow_id,
                        source_dp_rank=source_dp_rank,
                        token_count=cluster.dp_controller._requested_tokens(
                            request
                        ),
                        slack_s=slack_ms / 1000.0,
                        critical_weight=critical_weight,
                    ),
                ),
                rolling_state=rolling_state,
                topology=self.topology,
                weights=self.weights,
                failure_hazard_probability=(
                    self.failure_hazard_probability
                ),
                failure_scope=self.failure_scope,
                chunk_tokens=self.chunk_tokens,
                recompute_setup_s=self.recompute_setup_s,
                recompute_s_per_token=self.recompute_s_per_token,
                protected_restore_setup_s=self.protected_restore_setup_s,
                protected_restore_s_per_token=(
                    self.protected_restore_s_per_token
                ),
                time_limit_s=self.time_limit_s,
                deterministic_seed=(
                    self.deterministic_seed + self._next_window_id
                ),
                deterministic_replay=True,
            )
            certificate = solve.result.certificate
            evidence = {
                "schema_version": 1,
                "kind": "causal_exact_rolling_solve",
                "request_id": request.rid,
                "workflow_id": workflow_id,
                "logical_time_s": cluster.coordinator.now_s,
                "window_id": self._next_window_id,
                "route_version": self._next_route_version,
                "future_failure_manifest_used": False,
                "admission_information_mode": "causal_observed",
                "controller_snapshot": snapshot,
                "finite_instance": asdict(solve.instance),
                "optimization_result": asdict(solve.result),
                "certified_exact_window_plan": solve.plan.to_dict(),
                "deterministic_replay": {
                    "decision_sha256": solve.replay_decision_sha256,
                    "objective": solve.replay_objective,
                },
            }
            self._append_journal(evidence)
            self.history.append(
                {
                    "request_id": request.rid,
                    "workflow_id": workflow_id,
                    "logical_time_s": cluster.coordinator.now_s,
                    "window_id": self._next_window_id,
                    "route_version": self._next_route_version,
                    "instance_sha256": certificate.instance_sha256,
                    "decision_sha256": certificate.decision_sha256,
                    "status": certificate.status,
                    "objective": certificate.objective,
                    "dual_bound": certificate.dual_bound,
                    "mip_gap": certificate.mip_gap,
                    "solve_time_s": certificate.solve_time_s,
                    "backup_dp_rank": solve.plan.routes[0].backup_dp_rank,
                    "execution_dp_rank": source_dp_rank,
                    "unavailable_dp_ranks": list(
                        rolling_state.unavailable_dp_ranks
                    ),
                    "budget_publication_mode": (
                        solve.plan.budget_publication_mode
                    ),
                }
            )
            self._next_window_id += 1
            self._next_route_version += 1
            return solve.plan
