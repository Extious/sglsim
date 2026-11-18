"""Data-parallel request dispatch aligned with sglang.srt.managers.data_parallel_controller."""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import threading
import time
import zlib
from dataclasses import asdict
from enum import Enum, auto
from typing import Any, Callable

from sglsim.srt.managers.aegis_gpu_replica import (
    DEFAULT_GPU_REPLICA_ALPHA,
    GpuReplicaLedger,
    normalized_fraction,
)
from sglsim.srt.managers.aegis_l1_l2 import (
    project_alpha_beta,
    resolve_aegis_per_node_dram_quota,
)
from sglsim.srt.managers.io_struct import TokenizedGenerateReqInput
from sglsim.srt.managers.pipeline_parallel import (
    DPBudgetSnapshot,
    PipelineProcessGroup,
    PipelineWorkerHandle,
    RecoveryHint,
    StageCommitAck,
    StageId,
    enumerate_pipeline_workers,
)
from sglsim.srt.server_args import PortArgs, ServerArgs

from sglsim.srt.simulation.alpha_policy import (
    ENV_ALPHA_POLICY,
    OnlineAlphaAdapter,
    choose_alpha,
    online_alpha_adapter_enabled,
    resolve_failure_rate_per_window,
)
from sglsim.srt.simulation.aegis_placement import (
    AegisClusterTopology,
    AegisNodeHostPool,
    AegisPathCostModel,
    AegisPersistentPool,
    AegisPlacementRequest,
    AegisTopologyAwarePlanner,
    normalize_failure_scope,
)
from sglsim.srt.simulation.aegis_budget import (
    DPBudgetPublisher,
    HostAdmissionRequest,
    NodeHostCapacityArbiter,
)
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.aegis_residual_dag import resolve_slack_for_design
from sglsim.srt.simulation.aegis_exact_contract import (
    CertifiedExactWindowPlan,
)
from sglsim.srt.simulation.event_journal import AppendOnlyJsonlJournal
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.network import SimZmqContext, SimZmqSocket, get_zmq_socket

logger = logging.getLogger(__name__)


def _nonnegative_float_or_zero(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed) or parsed < 0.0:
        return 0.0
    return parsed


def routing_key_to_dp_rank(routing_key: str, dp_size: int) -> int:
    if dp_size <= 1:
        return 0
    key_bytes = routing_key.encode("utf-8")
    return zlib.crc32(key_bytes) % dp_size


class LoadBalanceMethod(Enum):
    ROUND_ROBIN = auto()
    CACHE_AWARE = auto()

    @classmethod
    def from_str(cls, method: str) -> LoadBalanceMethod:
        name = method.upper().replace("-", "_")
        if name in ("AUTO", "ROUND_ROBIN"):
            return cls.ROUND_ROBIN
        if name in ("CACHE_AWARE", "CACHE_LOAD_AWARE"):
            return cls.CACHE_AWARE
        raise ValueError(f"Unsupported load_balance_method: {method}")


class SchedulerWorkerDiedError(RuntimeError):
    """Raised when a DP scheduler subprocess has exited unexpectedly."""


class DataParallelController:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        *,
        scheduler_procs: list[mp.Process] | None = None,
        scheduler_cmd_queues: list[mp.Queue] | None = None,
        pipeline_process_group: PipelineProcessGroup | None = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self._scheduler_procs = scheduler_procs
        self._scheduler_cmd_queues = scheduler_cmd_queues or []
        self._pipeline_process_group = pipeline_process_group
        self._now_fn = now_fn
        sched = ConfigManager.get_scheduler_config()
        self.load_balance_method = LoadBalanceMethod.from_str(
            sched.load_balance_method
        )
        self.enable_routing_key_affinity = sched.enable_routing_key_affinity
        self.round_robin_counter = 0
        self.routing_key_to_rank: dict[str, int] = {}
        ctx = SimZmqContext()
        self.recv_from_tokenizer = get_zmq_socket(
            ctx,
            SimZmqSocket.PULL,
            port_args.scheduler_input_ipc_name,
            bind=False,
        )
        self.workers = [
            get_zmq_socket(
                ctx,
                SimZmqSocket.PUSH,
                port_args.scheduler_input_ipc_name_for_dp(dp_rank),
                bind=True,
            )
            for dp_rank in range(sched.dp_size)
        ]
        self.status = [True] * len(self.workers)
        cache = ConfigManager.get_cache_config(server_args.model_path)
        aegis_total_l1_pages = (
            cache.aegis_total_l1_capacity_pages
            if cache.aegis_total_l1_capacity_pages is not None
            else cache.l1_capacity_pages
        )
        self._aegis_gpu_ledger = GpuReplicaLedger(
            dp_size=len(self.workers),
            total_cache_tokens=int(aegis_total_l1_pages) * int(cache.page_size),
        )
        self.soft_failed_ranks: set[int] = set()
        self._soft_failure_ref_counts: dict[int, int] = {}
        self._soft_failure_lock = threading.Lock()
        self.hard_failed_ranks: set[int] = set()
        self._hard_failure_lock = threading.Lock()
        self._online_alpha_adapter = OnlineAlphaAdapter()
        self._aegis_l1_l2_lock = threading.Lock()
        self._aegis_recovery_hints: dict[str, dict[str, Any]] = {}
        self._aegis_rank_states: dict[int, dict[str, Any]] = {}
        self._aegis_stage_ack_pending: dict[
            tuple[str, int, int, int], dict[str, Any]
        ] = {}
        self._aegis_stage_group_commit_latest: dict[
            tuple[str, int, int, int, str], dict[str, Any]
        ] = {}
        self._aegis_physical_tier_commit_lock = threading.Lock()
        self._aegis_physical_tier_commits: dict[str, dict[str, Any]] = {}
        self._aegis_stage_reservation_plans: dict[
            tuple[str, int, int], dict[str, Any]
        ] = {}
        self._aegis_stage_supersede_pending: dict[
            tuple[str, int, int], dict[str, Any]
        ] = {}
        self._aegis_stage_completion_pending: dict[
            tuple[str, int, int], dict[str, Any]
        ] = {}
        # Router-owned, versioned target-before-source rotations.  The source
        # Scheduler may keep publishing its admission-time target identity;
        # this table is the atomic fence that resolves those stale events to
        # the current standby without advancing route_version.
        self._aegis_target_rotations: dict[
            tuple[str, int, int], dict[str, Any]
        ] = {}
        # Algorithm 2 keeps one recoverable frontier per workflow-agent
        # lineage.  The controller owns this index because consecutive
        # frontiers may be placed on different target scheduler processes.
        self._aegis_lineage_latest: dict[tuple[str, str], dict[str, Any]] = {}
        self._aegis_superseded_backups: dict[str, dict[str, Any]] = {}
        # This is deliberately unbounded.  Formal evidence must never silently
        # lose early Router/placement/ACK/failure events because a diagnostic
        # deque reached its maxlen.  The optional JSONL journal below adds an
        # incremental durability boundary for long experiments.
        self._aegis_events: list[dict[str, Any]] = []
        self._aegis_event_sequence = 0
        self._gpu_recovery_lock = threading.Lock()
        self._gpu_recovery_waiters: dict[str, threading.Event] = {}
        self._gpu_recovery_results: dict[str, dict[str, Any]] = {}
        self._gpu_recovery_stage_results: dict[str, dict[str, Any]] = {}
        aegis_raw = (ConfigManager._raw or {}).get("aegis_l1_l2", {})
        if not isinstance(aegis_raw, dict):
            aegis_raw = {}
        self._aegis_recovery_cost_calibration_id = str(
            aegis_raw.get("recovery_cost_calibration_id") or ""
        )
        self._aegis_recovery_cost_source = str(
            aegis_raw.get("recovery_cost_source")
            or (
                "profiler_calibration"
                if self._aegis_recovery_cost_calibration_id
                else "default_coefficients"
            )
        )
        self._aegis_committed_copy_protocol = bool(
            aegis_raw.get("committed_copy_protocol", False)
        )
        self._aegis_design_semantics = str(
            aegis_raw.get("design_semantics") or "legacy_l1_l2_v1"
        )
        self._aegis_disable_slack_ordering = bool(
            aegis_raw.get("disable_slack_ordering", False)
        )
        journal_path = str(
            os.environ.get("SGLSIM_AEGIS_EVENT_JOURNAL_PATH")
            or aegis_raw.get("event_journal_path")
            or ""
        ).strip()
        self._aegis_event_journal = (
            AppendOnlyJsonlJournal(journal_path) if journal_path else None
        )
        self._aegis_rehome_after_recovery = bool(
            aegis_raw.get("rehome_after_recovery", False)
        )
        self._aegis_pending_rehome: dict[str, int] = {}
        self._aegis_exact_window_plan: CertifiedExactWindowPlan | None = None
        self._aegis_exact_budget_versions: dict[int, int] = {}
        self._aegis_exact_deployment_violations: list[dict[str, Any]] = []
        self._aegis_recompute_setup_ms = float(
            aegis_raw.get("recompute_setup_ms", 4.0)
        )
        self._aegis_recompute_ms_per_token = float(
            aegis_raw.get("recompute_ms_per_token", 0.0108)
        )
        topology_raw = aegis_raw.get("topology", {})
        if not isinstance(topology_raw, dict):
            topology_raw = {}
        self._aegis_topology = AegisClusterTopology.from_dict(
            topology_raw,
            dp_size=sched.dp_size,
            pp_size=sched.pp_size,
        )
        self._aegis_path_costs = AegisPathCostModel.from_dict(aegis_raw)
        self._aegis_placement_planner = AegisTopologyAwarePlanner(
            topology=self._aegis_topology,
            costs=self._aegis_path_costs,
        )
        legacy_host_capacity_tokens = max(
            int(cache.l2_capacity_pages) * int(cache.page_size), 1
        )
        self._aegis_per_node_dram_quota = resolve_aegis_per_node_dram_quota(
            aegis_raw,
            fallback_tokens=legacy_host_capacity_tokens,
        )
        host_capacity_tokens = self._aegis_per_node_dram_quota.quota_tokens
        initial_host_ratio = normalized_fraction(
            aegis_raw.get("initial_beta", 0.40)
        )
        if initial_host_ratio is None:
            initial_host_ratio = 0.40
        pinned_ratio = normalized_fraction(
            aegis_raw.get("host_pinned_quota_ratio", 0.25)
        )
        if pinned_ratio is None:
            pinned_ratio = 0.25
        host_alignment_tokens = max(
            int(
                aegis_raw.get(
                    "host_alignment_tokens",
                    aegis_raw.get("chunk_tokens", 512),
                )
            ),
            1,
        )
        host_max_regions_raw = aegis_raw.get("host_max_regions")
        host_max_regions = (
            max(int(host_max_regions_raw), 1)
            if host_max_regions_raw is not None
            else None
        )
        self._aegis_host_pools = {
            node_id: AegisNodeHostPool(
                node_id=node_id,
                capacity_tokens=host_capacity_tokens,
                # This layer represents physical allocation only.  Per-DP
                # beta is enforced by NodeHostCapacityArbiter below.
                protection_ratio=1.0,
                pinned_quota_tokens=int(host_capacity_tokens * pinned_ratio),
                alignment_tokens=host_alignment_tokens,
                max_regions=host_max_regions,
            )
            for node_id in self._aegis_topology.nodes
        }
        self._aegis_l3_pool = AegisPersistentPool(
            capacity_tokens=max(
                int(
                    aegis_raw.get(
                        "l3_capacity_tokens",
                        int(cache.l3_capacity_pages) * int(cache.page_size),
                    )
                ),
                0,
            ),
            manifest_quorum=max(int(aegis_raw.get("l3_manifest_quorum", 1)), 1),
        )
        initial_alpha = normalized_fraction(
            aegis_raw.get("initial_alpha", 0.20)
        )
        if initial_alpha is None:
            initial_alpha = 0.20
        max_alpha = normalized_fraction(aegis_raw.get("max_alpha", 1.0))
        if max_alpha is None:
            max_alpha = 1.0
        initial_alpha = min(initial_alpha, max_alpha)
        stage_hbm_limit = int(initial_alpha * self._aegis_gpu_ledger.total_cache_tokens)
        # beta is a DP-owned policy fraction of one node's Host capacity.
        # Multiple DP policies may sum above the physical capacity; the node
        # arbiter rejects oversubscription without rewriting any DP's beta.
        per_dp_host_total_tokens = host_capacity_tokens
        self._aegis_budget_publishers = {
            dp_rank: DPBudgetPublisher(
                dp_rank=dp_rank,
                pp_size=int(self.server_args.pp_size),
            )
            for dp_rank in range(len(self.workers))
        }
        self._aegis_node_capacity_arbiters = {
            node_id: NodeHostCapacityArbiter(
                node_id=node_id,
                physical_capacity_tokens=host_capacity_tokens,
                allowed_dp_ranks=tuple(
                    dp_rank
                    for dp_rank in range(len(self.workers))
                    if self._aegis_topology.node_for_rank(dp_rank) == node_id
                ),
            )
            for node_id in self._aegis_topology.nodes
        }
        for dp_rank, publisher in self._aegis_budget_publishers.items():
            host_policy_limit_tokens = (
                0
                if initial_host_ratio <= 0.0
                else min(
                    max(
                        int(initial_host_ratio * per_dp_host_total_tokens),
                        1,
                    ),
                    per_dp_host_total_tokens,
                )
            )
            bootstrap_alpha = (
                stage_hbm_limit
                / max(self._aegis_gpu_ledger.total_cache_tokens, 1)
            )
            bootstrap_beta = (
                host_policy_limit_tokens
                / max(per_dp_host_total_tokens, 1)
            )
            bootstrap_alpha, bootstrap_beta = project_alpha_beta(
                bootstrap_alpha,
                bootstrap_beta,
                normal_pressure=0.0,
            )
            stage_hbm_limit = int(
                bootstrap_alpha * self._aegis_gpu_ledger.total_cache_tokens
            )
            host_policy_limit_tokens = (
                0
                if bootstrap_beta <= 0.0
                else min(
                    max(int(bootstrap_beta * per_dp_host_total_tokens), 1),
                    per_dp_host_total_tokens,
                )
            )
            snapshot = DPBudgetSnapshot(
                dp_rank=dp_rank,
                budget_version=1,
                window_id=0,
                # alpha/beta are exact normalized views of the integer policy
                # budgets, never a second rounded source of truth.
                alpha=(
                    stage_hbm_limit
                    / max(self._aegis_gpu_ledger.total_cache_tokens, 1)
                ),
                beta=(
                    host_policy_limit_tokens
                    / max(per_dp_host_total_tokens, 1)
                ),
                hbm_limit_tokens_per_stage=tuple(
                    stage_hbm_limit for _ in range(int(self.server_args.pp_size))
                ),
                host_policy_limit_tokens=host_policy_limit_tokens,
                valid_from_logical_time=0.0,
                objective_id="bootstrap-config-not-optimized",
                solver_certificate_id="not-applicable-bootstrap",
            )
            publisher.publish(snapshot)
            node_id = self._aegis_topology.node_for_rank(dp_rank)
            self._aegis_node_capacity_arbiters[node_id].publish_dp_budget(snapshot)
            if (
                bool(aegis_raw.get("enabled", False))
                and (
                    self._pipeline_process_group is not None
                    or bool(self._scheduler_procs)
                )
            ):
                self._send_dp_runtime_command(
                    dp_rank,
                    ("aegis_publish_dp_budget", snapshot.to_dict()),
                )

    def _validate_rank(self, dp_rank: int) -> None:
        if dp_rank < 0 or dp_rank >= len(self.workers):
            raise ValueError(f"invalid DP rank {dp_rank}")

    def _has_runtime_command_target(self, dp_rank: int) -> bool:
        if self._pipeline_process_group is not None:
            return 0 <= dp_rank < self._pipeline_process_group.dp_size
        return 0 <= dp_rank < len(self._scheduler_cmd_queues)

    def _send_dp_runtime_command(self, dp_rank: int, message: object) -> None:
        self._validate_rank(dp_rank)
        if self._pipeline_process_group is not None:
            self._pipeline_process_group.broadcast_dp_command(dp_rank, message)
            return
        if dp_rank >= len(self._scheduler_cmd_queues):
            raise RuntimeError(f"DP rank {dp_rank} has no runtime command queue")
        self._scheduler_cmd_queues[dp_rank].put(message)

    def _send_stage_runtime_command(
        self, dp_rank: int, pp_rank: int, message: object
    ) -> None:
        self._validate_rank(dp_rank)
        if self._pipeline_process_group is None:
            # Unit-level/controller-only construction predates physical PP
            # handles.  Preserve observability by placing each stage-tagged
            # command on the DP queue; the real Engine always supplies a
            # PipelineProcessGroup for pp_size > 1.
            self._send_dp_runtime_command(dp_rank, message)
            return
        self._pipeline_process_group.send_stage_command(
            StageId(dp_rank=dp_rank, pp_rank=pp_rank, tp_rank=0),
            message,
        )

    @staticmethod
    def _stage_backup_key(backup_key: str, pp_rank: int) -> str:
        return f"{backup_key}::pp{int(pp_rank)}"

    @staticmethod
    def _pp_rank_from_stage_backup_key(backup_key: str) -> int | None:
        _root, marker, suffix = str(backup_key).rpartition("::pp")
        if not marker or not suffix.isdigit():
            return None
        return int(suffix)

    def soft_fail_rank(self, dp_rank: int) -> bool:
        self._validate_rank(dp_rank)
        with self._hard_failure_lock:
            if dp_rank in self.hard_failed_ranks:
                logger.info(
                    "Ignored soft-fail request for hard-failed DP rank %d",
                    dp_rank,
                )
                return False
            hard_failed = set(self.hard_failed_ranks)
        with self._soft_failure_lock:
            unavailable = {
                rank for rank, healthy in enumerate(self.status) if not healthy
            }
            unavailable.update(hard_failed)
            unavailable.update(self.soft_failed_ranks)
            unavailable.add(dp_rank)
            if (
                dp_rank not in self.soft_failed_ranks
                and len(unavailable) >= len(self.workers)
            ):
                logger.info(
                    "Ignored soft-fail request for DP rank %d; it would leave no "
                    "healthy DP worker (active soft-failed ranks=%s)",
                    dp_rank,
                    sorted(self.soft_failed_ranks),
                )
                return False
            self._soft_failure_ref_counts[dp_rank] = (
                self._soft_failure_ref_counts.get(dp_rank, 0) + 1
            )
            self.soft_failed_ranks.add(dp_rank)
        logger.warning("Soft-failed DP rank %d for simulated failover", dp_rank)
        return True

    def recover_soft_failed_rank(self, dp_rank: int) -> bool:
        self._validate_rank(dp_rank)
        with self._soft_failure_lock:
            count = self._soft_failure_ref_counts.get(dp_rank, 0)
            if count <= 0:
                logger.info(
                    "Ignored recover request for non-soft-failed DP rank %d",
                    dp_rank,
                )
                return False
            if count == 1:
                self._soft_failure_ref_counts.pop(dp_rank, None)
                self.soft_failed_ranks.discard(dp_rank)
            else:
                self._soft_failure_ref_counts[dp_rank] = count - 1
        logger.info("Recovered soft-failed DP rank %d", dp_rank)
        return True

    def clear_rank_l1_before_recovery(
        self,
        dp_rank: int,
        *,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Ask the failed scheduler to clear volatile GPU KV before release."""

        request_id, waiter = self.begin_rank_l1_clear(dp_rank)
        if not waiter.wait(timeout=max(float(timeout_s), 0.1)):
            with self._gpu_recovery_lock:
                self._gpu_recovery_waiters.pop(request_id, None)
                self._gpu_recovery_results.pop(request_id, None)
                self._gpu_recovery_stage_results.pop(request_id, None)
            raise TimeoutError(
                f"Timed out waiting for DP rank {dp_rank} to clear GPU L1"
            )
        result = self.finish_rank_l1_clear(request_id)
        if result is None:
            raise RuntimeError(f"Missing GPU L1 clear ACK for DP rank {dp_rank}")
        return result

    def begin_rank_l1_clear(
        self,
        dp_rank: int,
        *,
        request_id: str | None = None,
    ) -> tuple[str, threading.Event]:
        """Start the existing GPU-L1 clear handshake without wall blocking."""

        self._validate_rank(dp_rank)
        with self._soft_failure_lock:
            if dp_rank not in self.soft_failed_ranks:
                raise ValueError(f"DP rank {dp_rank} is not soft-failed")
        if not self._has_runtime_command_target(dp_rank):
            raise RuntimeError(f"DP rank {dp_rank} has no runtime command queue")
        resolved_id = request_id or f"gpu-l1-clear-{dp_rank}-{time.time_ns()}"
        waiter = threading.Event()
        with self._gpu_recovery_lock:
            if resolved_id in self._gpu_recovery_waiters:
                raise ValueError(f"duplicate GPU L1 clear request_id={resolved_id}")
            self._gpu_recovery_waiters[resolved_id] = waiter
            if int(self.server_args.pp_size) > 1:
                self._gpu_recovery_stage_results[resolved_id] = {
                    "dp_rank": int(dp_rank),
                    "stages": {},
                }
        self._send_dp_runtime_command(
            dp_rank,
            (
                "simulate_gpu_recovery_clear_l1",
                {"request_id": resolved_id, "dp_rank": int(dp_rank)},
            )
        )
        return resolved_id, waiter

    def finish_rank_l1_clear(self, request_id: str) -> dict[str, Any] | None:
        """Collect a completed nonblocking clear handshake, if available."""

        with self._gpu_recovery_lock:
            waiter = self._gpu_recovery_waiters.get(str(request_id))
            if waiter is None or not waiter.is_set():
                return None
            self._gpu_recovery_waiters.pop(str(request_id), None)
            self._gpu_recovery_stage_results.pop(str(request_id), None)
            return self._gpu_recovery_results.pop(str(request_id), None)

    def mark_rank_hard_failed(self, dp_rank: int) -> bool:
        self._validate_rank(dp_rank)
        with self._hard_failure_lock:
            already_failed = dp_rank in self.hard_failed_ranks
            self.hard_failed_ranks.add(dp_rank)
            self.status[dp_rank] = False
        with self._soft_failure_lock:
            self.soft_failed_ranks.discard(dp_rank)
            self._soft_failure_ref_counts.pop(dp_rank, None)
        if not already_failed:
            logger.warning("Hard-failed DP rank %d for simulated failover", dp_rank)
        return not already_failed

    def mark_rank_restarted(self, dp_rank: int) -> bool:
        self._validate_rank(dp_rank)
        with self._hard_failure_lock:
            was_failed = dp_rank in self.hard_failed_ranks
            self.hard_failed_ranks.discard(dp_rank)
            self.status[dp_rank] = True
        logger.info("Restarted hard-failed DP rank %d", dp_rank)
        return was_failed

    def soft_failed(self) -> list[int]:
        with self._soft_failure_lock:
            return sorted(self.soft_failed_ranks)

    def hard_failed(self) -> list[int]:
        with self._hard_failure_lock:
            return sorted(self.hard_failed_ranks)

    def _worker_available(self, dp_rank: int) -> bool:
        with self._soft_failure_lock:
            soft_failed = dp_rank in self.soft_failed_ranks
        with self._hard_failure_lock:
            hard_failed = dp_rank in self.hard_failed_ranks
        return self.status[dp_rank] and not soft_failed and not hard_failed

    def refresh_worker_status(self) -> list[int]:
        """Mark workers dead when their scheduler subprocess has exited."""
        if self._pipeline_process_group is not None:
            dead: list[int] = []
            for dp_rank in range(self._pipeline_process_group.dp_size):
                failed_stages = tuple(
                    worker
                    for worker in self._pipeline_process_group.for_dp(dp_rank)
                    if worker.process.exitcode is not None
                )
                if not failed_stages:
                    continue
                with self._hard_failure_lock:
                    expected_dead = dp_rank in self.hard_failed_ranks
                self.status[dp_rank] = False
                if expected_dead:
                    continue
                logger.error(
                    "DP pipeline worker(s) exited for rank %d: %s",
                    dp_rank,
                    [
                        {
                            "pp_rank": worker.spec.stage.pp_rank,
                            "tp_rank": worker.spec.stage.tp_rank,
                            "exitcode": worker.process.exitcode,
                        }
                        for worker in failed_stages
                    ],
                )
                dead.append(dp_rank)
            return dead
        if not self._scheduler_procs:
            return []
        dead: list[int] = []
        for dp_rank, proc in enumerate(self._scheduler_procs):
            if proc.exitcode is None:
                continue
            with self._hard_failure_lock:
                expected_dead = dp_rank in self.hard_failed_ranks
            if expected_dead:
                self.status[dp_rank] = False
                continue
            if self.status[dp_rank]:
                logger.error(
                    "DP scheduler worker %d exited with code %s",
                    dp_rank,
                    proc.exitcode,
                )
            self.status[dp_rank] = False
            dead.append(dp_rank)
        return dead

    def ensure_workers_alive(self) -> None:
        dead = self.refresh_worker_status()
        if dead:
            raise SchedulerWorkerDiedError(
                f"DP scheduler worker(s) died (ranks={dead})"
            )

    def replace_pipeline_process_group(
        self, group: PipelineProcessGroup
    ) -> None:
        """Atomically replace parent-owned PP handles after a DP restart."""

        if group.dp_size != len(self.workers):
            raise ValueError(
                "replacement pipeline group DP size does not match Router workers"
            )
        self._pipeline_process_group = group
        self._scheduler_procs = [
            group.leader_for_dp(dp_rank).process
            for dp_rank in range(group.dp_size)
        ]
        self._scheduler_cmd_queues = [
            group.leader_for_dp(dp_rank).command_queue
            for dp_rank in range(group.dp_size)
        ]

    def replace_scheduler_process(
        self,
        dp_rank: int,
        *,
        process: Any,
        command_queue: Any,
    ) -> None:
        """Replace one non-PP Scheduler handle after a hard restart.

        The online Engine already replaces an entire
        :class:`PipelineProcessGroup` for physical PP.  The offline DES also
        supports PP=1, where there is no pipeline group in the Controller;
        mutating the two private handle arrays from the simulator would make
        worker-liveness audits race the restart.  Keep that ownership here and
        reject accidental use while a pipeline group is active.
        """

        rank = int(dp_rank)
        self._validate_rank(rank)
        if self._pipeline_process_group is not None:
            raise RuntimeError(
                "replace_scheduler_process is only valid without a pipeline group"
            )
        if rank >= len(self._scheduler_procs) or rank >= len(
            self._scheduler_cmd_queues
        ):
            raise RuntimeError(f"DP rank {rank} has no Scheduler handle slot")
        if process is None or command_queue is None:
            raise ValueError("replacement process and command_queue are required")
        self._scheduler_procs[rank] = process
        self._scheduler_cmd_queues[rank] = command_queue

    def _choose_aegis_rotation_target(
        self,
        *,
        source_dp_rank: int,
        excluded_dp_ranks: set[int],
        failure_scope: str,
    ) -> int:
        """Choose a healthy, layout-compatible standby after target failure.

        The route version and source ownership are intentionally absent from
        the scoring objective: they are immutable inputs to target-before-
        source rotation.  Deterministic pressure/load ordering makes repeated
        failure delivery idempotent and excludes the source and every fenced
        target before considering a candidate.
        """

        self._validate_rank(source_dp_rank)
        scope = normalize_failure_scope(failure_scope)
        excluded = {int(rank) for rank in excluded_dp_ranks}
        excluded.add(int(source_dp_rank))
        candidates: list[tuple[float, int, int, int]] = []
        for rank in range(len(self.workers)):
            if rank in excluded or not self._worker_available(rank):
                continue
            if not self._aegis_topology.copy_survives(
                source_dp_rank=source_dp_rank,
                copy_dp_rank=rank,
                failure_scope=scope,
            ):
                continue
            budget = self._aegis_budget_publishers[rank].current
            if budget is None or sum(budget.hbm_limit_tokens_per_stage) <= 0:
                continue
            rank_state = self._aegis_rank_states.get(rank, {})
            stage_limit = max(sum(budget.hbm_limit_tokens_per_stage), 1)
            protected = max(int(rank_state.get("l1_used_tokens", 0)), 0)
            pressure = protected / stage_limit
            candidates.append(
                (
                    pressure,
                    self._aegis_gpu_ledger.inflight_requests(rank),
                    self._aegis_gpu_ledger.inflight_tokens(rank),
                    rank,
                )
            )
        return min(candidates)[-1] if candidates else -1

    def invalidate_aegis_physical_failure(
        self,
        *,
        failed_dp_ranks: tuple[int, ...] | list[int] | set[int] = (),
        failed_node_ids: tuple[int, ...] | list[int] | set[int] = (),
        reason: str = "physical_fail_stop",
    ) -> dict[str, Any]:
        """Invalidate Router-visible Aegis copies lost by a hard failure.

        Peer-HBM is owned by the target DP rank, while source-owned DRAM is
        owned by its physical node.  Losing one tier must not erase an
        independent surviving tier.  This method updates only Controller
        metadata and quota mirrors; the physical byte ledgers remain owned by
        ``GlobalDES``.
        """

        ranks = tuple(sorted({int(rank) for rank in failed_dp_ranks}))
        for rank in ranks:
            self._validate_rank(rank)
        nodes = tuple(sorted({int(node) for node in failed_node_ids}))
        unknown_nodes = set(nodes) - set(self._aegis_host_pools)
        if unknown_nodes:
            raise ValueError(
                f"unknown Aegis physical node IDs: {sorted(unknown_nodes)}"
            )
        reason_name = str(reason)
        if not reason_name:
            raise ValueError("physical failure reason must be nonempty")

        gpu_ledger_results = {
            str(rank): self._aegis_gpu_ledger.clear_rank_l1(rank)
            for rank in ranks
        }
        changed: list[dict[str, Any]] = []
        released_host_keys: list[tuple[int, str, int]] = []
        with self._aegis_l1_l2_lock:
            for backup_key, current in list(self._aegis_recovery_hints.items()):
                hint = dict(current)
                invalidated: list[str] = []
                target_rank = int(hint.get("target_dp_rank", -1))
                invalidated_hbm_reserved_tokens = 0
                if target_rank in ranks and any(
                    int(hint.get(field, 0)) > 0
                    for field in (
                        "l1_tokens",
                        "hbm_pending_tokens",
                        "reserved_l1_tokens",
                    )
                ):
                    invalidated_hbm_reserved_tokens = max(
                        int(hint.get("reserved_l1_tokens", 0)),
                        int(hint.get("l1_tokens", 0))
                        + int(hint.get("hbm_pending_tokens", 0)),
                    )
                    hint["l1_tokens"] = 0
                    hint["hbm_pending_tokens"] = 0
                    hint["reserved_l1_tokens"] = 0
                    frontiers = dict(
                        hint.get("tier_committed_source_frontiers") or {}
                    )
                    frontiers["hbm"] = 0
                    hint["tier_committed_source_frontiers"] = frontiers
                    stage_payloads = hint.get("stage_payloads")
                    if isinstance(stage_payloads, list):
                        cleared_stages = []
                        for raw_stage in stage_payloads:
                            if not isinstance(raw_stage, dict):
                                continue
                            stage = dict(raw_stage)
                            stage["l1_tokens"] = 0
                            stage["hbm_pending_tokens"] = 0
                            stage["reserved_l1_tokens"] = 0
                            tier_frontiers = dict(
                                stage.get("tier_committed_source_frontiers")
                                or {}
                            )
                            tier_frontiers["hbm"] = 0
                            stage["tier_committed_source_frontiers"] = (
                                tier_frontiers
                            )
                            cleared_stages.append(stage)
                        hint["stage_payloads"] = cleared_stages
                        hint["stage_hbm_frontiers"] = [
                            0 for _stage in cleared_stages
                        ]
                    invalidated.append("hbm")

                host_lease = hint.get("host_pool_lease")
                host_node = (
                    int(host_lease.get("node_id", -1))
                    if isinstance(host_lease, dict)
                    else -1
                )
                if host_node in nodes and any(
                    int(hint.get(field, 0)) > 0
                    for field in (
                        "l2_tokens",
                        "dram_pending_tokens",
                        "reserved_l2_tokens",
                    )
                ):
                    version = max(int(hint.get("version", 1)), 1)
                    hint["l2_tokens"] = 0
                    hint["dram_pending_tokens"] = 0
                    hint["reserved_l2_tokens"] = 0
                    hint["host_pool_lease"] = None
                    frontiers = dict(
                        hint.get("tier_committed_source_frontiers") or {}
                    )
                    frontiers["dram"] = 0
                    hint["tier_committed_source_frontiers"] = frontiers
                    invalidated.append("dram")
                    released_host_keys.append((host_node, backup_key, version))

                if not invalidated:
                    continue
                previous_invalidations = list(
                    hint.get("failure_invalidated_tiers") or ()
                )
                hint["failure_invalidated_tiers"] = sorted(
                    set(previous_invalidations).union(invalidated)
                )
                hint["failure_invalidation_reason"] = reason_name
                hint = self._refresh_aegis_recovery_hint(hint)
                self._aegis_recovery_hints[backup_key] = dict(hint)
                changed.append(
                    {
                        "backup_key": backup_key,
                        "version": max(int(hint.get("version", 1)), 1),
                        "target_dp_rank": target_rank,
                        "invalidated_tiers": tuple(invalidated),
                        "invalidated_hbm_reserved_tokens": (
                            invalidated_hbm_reserved_tokens
                        ),
                        "safe_frontier_tokens": int(
                            hint.get("safe_frontier_tokens", 0)
                        ),
                    }
                )

        target_rotations: list[dict[str, Any]] = []
        rotation_commands: list[tuple[int, int, dict[str, Any]]] = []
        for invalidated in changed:
            if "hbm" not in invalidated["invalidated_tiers"]:
                continue
            backup_key = str(invalidated["backup_key"])
            version = int(invalidated["version"])
            with self._aegis_l1_l2_lock:
                current = self._aegis_recovery_hints.get(backup_key)
                hint = dict(current) if isinstance(current, dict) else None
            if hint is None or int(hint.get("version", -1)) != version:
                continue
            source_rank = int(hint.get("source_dp_rank", -1))
            failed_target = int(invalidated["target_dp_rank"])
            route_version = max(int(hint.get("route_version", 1)), 1)
            # This path is target-before-source only.  Source failure advances
            # the route through Rebind and must not be mistaken for standby
            # rotation.
            if source_rank in ranks or not self._worker_available(source_rank):
                continue
            failure_scope = normalize_failure_scope(
                hint.get("failure_scope", "node")
            )
            fenced_targets = set(ranks)
            previous_rotation = hint.get("hbm_target_rotation")
            if isinstance(previous_rotation, dict):
                fenced_targets.update(
                    int(rank)
                    for rank in previous_rotation.get(
                        "failed_target_dp_ranks", ()
                    )
                )
            replacement = self._choose_aegis_rotation_target(
                source_dp_rank=source_rank,
                excluded_dp_ranks=fenced_targets,
                failure_scope=failure_scope,
            )
            if replacement < 0:
                degraded = {
                    "backup_key": backup_key,
                    "version": version,
                    "route_version": route_version,
                    "source_dp_rank": source_rank,
                    "failed_target_dp_rank": failed_target,
                    "replacement_target_dp_rank": None,
                    "failed_target_dp_ranks": sorted(fenced_targets),
                    "route_version_changed": False,
                    "degraded": True,
                    "reason": "no_healthy_compatible_standby",
                }
                with self._aegis_l1_l2_lock:
                    latest = self._aegis_recovery_hints.get(backup_key)
                    if (
                        isinstance(latest, dict)
                        and int(latest.get("version", -1)) == version
                        and int(latest.get("target_dp_rank", -1))
                        == failed_target
                    ):
                        latest = dict(latest)
                        latest["route_degraded_no_standby"] = True
                        latest["hbm_target_rotation"] = degraded
                        self._aegis_recovery_hints[backup_key] = latest
                target_rotations.append(degraded)
                self._append_aegis_event("route_degraded_no_standby", degraded)
                continue

            prior_generation = (
                int(previous_rotation.get("generation", 0))
                if isinstance(previous_rotation, dict)
                else 0
            )
            rotation = {
                "backup_key": backup_key,
                "version": version,
                "route_version": route_version,
                "source_dp_rank": source_rank,
                "previous_target_dp_rank": failed_target,
                "replacement_target_dp_rank": replacement,
                "failed_target_dp_ranks": sorted(
                    fenced_targets.union({failed_target})
                ),
                "generation": prior_generation + 1,
                "route_version_changed": False,
                "degraded": False,
                "dram_preserved": "dram"
                not in invalidated["invalidated_tiers"],
            }
            rotation_key = (backup_key, version, route_version)
            with self._aegis_l1_l2_lock:
                latest = self._aegis_recovery_hints.get(backup_key)
                if not (
                    isinstance(latest, dict)
                    and int(latest.get("version", -1)) == version
                    and int(latest.get("target_dp_rank", -1)) == failed_target
                ):
                    continue
                latest = dict(latest)
                latest["target_dp_rank"] = replacement
                latest["route_degraded_no_standby"] = False
                latest["hbm_target_rotation"] = dict(rotation)
                latest["hbm_pending_tokens"] = 0
                latest["reserved_l1_tokens"] = 0
                stage_payloads = latest.get("stage_payloads")
                if isinstance(stage_payloads, list):
                    latest["stage_payloads"] = [
                        {
                            **dict(stage),
                            "target_dp_rank": replacement,
                            "hbm_target_rotation": dict(rotation),
                        }
                        for stage in stage_payloads
                        if isinstance(stage, dict)
                    ]
                self._aegis_recovery_hints[backup_key] = latest
                self._aegis_target_rotations[rotation_key] = dict(rotation)
                for lineage, lineage_hint in self._aegis_lineage_latest.items():
                    if (
                        str(lineage_hint.get("backup_key") or "") == backup_key
                        and int(lineage_hint.get("version", -1)) == version
                    ):
                        lineage_hint["target_dp_rank"] = replacement

                replacement_budget = self._aegis_budget_publishers[
                    replacement
                ].current
                if replacement_budget is None:
                    raise RuntimeError(
                        f"replacement DP{replacement} has no published budget"
                    )
                for plan_key, original_plan in tuple(
                    self._aegis_stage_reservation_plans.items()
                ):
                    if plan_key[0] != backup_key or int(plan_key[1]) != version:
                        continue
                    pp_rank = int(plan_key[2])
                    gpu_candidate = dict(
                        original_plan.get("gpu_candidate") or {}
                    )
                    gpu_candidate.update(
                        {
                            "target_dp_rank": replacement,
                            "target_node_id": (
                                self._aegis_topology.node_for_rank(replacement)
                            ),
                            "failure_domain": (
                                f"node:{self._aegis_topology.node_for_rank(replacement)}"
                            ),
                            "reusable_dp_ranks": [replacement],
                            "pp_gpu_mapping": list(
                                self._aegis_topology.pp_gpu_mapping(
                                    source_dp_rank=source_rank,
                                    target_dp_rank=replacement,
                                )
                            ),
                        }
                    )
                    rotated_plan = {
                        **dict(original_plan),
                        "target_dp_rank": replacement,
                        "allowed_tiers": ["l1"],
                        "gpu_candidate": gpu_candidate,
                        "route_version": route_version,
                        "dp_budget_snapshot": replacement_budget.to_dict(),
                        "budget_version": replacement_budget.budget_version,
                        "hbm_failure_domain": (
                            f"dp-rank-{replacement}"
                        ),
                        "target_rotation": dict(rotation),
                        "target_rotation_hbm_only": True,
                    }
                    self._aegis_stage_reservation_plans[plan_key] = dict(
                        rotated_plan
                    )
                    rotation_commands.append(
                        (replacement, pp_rank, rotated_plan)
                    )
            target_rotations.append(rotation)
            self._append_aegis_event("standby_target_rotated", rotation)

        for target_rank, pp_rank, rotated_plan in rotation_commands:
            self._send_stage_runtime_command(
                target_rank,
                pp_rank,
                ("aegis_l1_l2_backup_plan", rotated_plan),
            )

        for node_id, backup_key, version in released_host_keys:
            self._aegis_host_pools[node_id].release(
                backup_key, max_version=version
            )
            self._aegis_node_capacity_arbiters[node_id].release(backup_key)
        result = {
            "reason": reason_name,
            "failed_dp_ranks": list(ranks),
            "failed_node_ids": list(nodes),
            "gpu_ledger_results": gpu_ledger_results,
            "invalidated_hints": changed,
            "released_host_keys": [
                {
                    "node_id": node_id,
                    "backup_key": backup_key,
                    "version": version,
                }
                for node_id, backup_key, version in released_host_keys
            ],
            "target_rotations": target_rotations,
        }
        self._append_aegis_event("physical_failure_invalidated", result)
        return result

    def replay_current_dp_budget(self, dp_rank: int) -> DPBudgetSnapshot:
        """Replay one immutable policy version to a restarted DP pipeline."""

        self._validate_rank(dp_rank)
        publisher = self._aegis_budget_publishers[dp_rank]
        snapshot = publisher.current
        if snapshot is None:
            raise RuntimeError(f"DP rank {dp_rank} has no published budget")
        publisher.reset_acknowledgements()
        self._send_dp_runtime_command(
            dp_rank,
            ("aegis_publish_dp_budget", snapshot.to_dict()),
        )
        return snapshot

    def replay_aegis_reservations_after_restart(
        self, dp_rank: int
    ) -> list[dict[str, Any]]:
        """Recreate only still-live tier metadata in a fresh Scheduler.

        A hard restart destroys the rank-local manager, but the Router and
        physical DES may still own a source-node DRAM copy.  Replaying the
        original plan verbatim would incorrectly reserve the failed HBM tier;
        therefore the current, failure-filtered recovery hint selects the
        exact subset of tiers that may be reconstructed.
        """

        self._validate_rank(dp_rank)
        publisher = self._aegis_budget_publishers[dp_rank]
        budget = publisher.current
        if budget is None:
            raise RuntimeError(f"DP rank {dp_rank} has no published budget")
        with self._aegis_l1_l2_lock:
            candidates = [
                (key, dict(plan))
                for key, plan in self._aegis_stage_reservation_plans.items()
                if int(plan.get("target_dp_rank", -1)) == dp_rank
            ]
            hints = {
                key: dict(value)
                for key, value in self._aegis_recovery_hints.items()
            }

        replayed: list[dict[str, Any]] = []
        for (_backup_key, _version, pp_rank), plan in sorted(candidates):
            root_key = str(
                plan.get("request_backup_key")
                or plan.get("backup_key")
                or ""
            )
            hint = hints.get(root_key)
            if not isinstance(hint, dict):
                continue
            original = {str(tier) for tier in plan.get("allowed_tiers", ())}
            surviving: list[str] = []
            if "l1" in original and any(
                int(hint.get(field, 0)) > 0
                for field in (
                    "l1_tokens",
                    "hbm_pending_tokens",
                    "reserved_l1_tokens",
                )
            ):
                surviving.append("l1")
            if "l2" in original and any(
                int(hint.get(field, 0)) > 0
                for field in (
                    "l2_tokens",
                    "dram_pending_tokens",
                    "reserved_l2_tokens",
                )
            ):
                surviving.append("l2")
            if not surviving:
                continue
            replay_plan = {
                **plan,
                "allowed_tiers": surviving,
                "dp_budget_snapshot": budget.to_dict(),
                "restart_rehydration": True,
            }
            self._send_stage_runtime_command(
                dp_rank,
                int(pp_rank),
                ("aegis_l1_l2_backup_plan", replay_plan),
            )
            replayed.append(
                {
                    "backup_key": root_key,
                    "version": max(int(plan.get("version", 1)), 1),
                    "dp_rank": dp_rank,
                    "pp_rank": int(pp_rank),
                    "allowed_tiers": list(surviving),
                }
            )
        self._append_aegis_event(
            "restart_reservations_replayed",
            {"dp_rank": dp_rank, "reservations": replayed},
        )
        return replayed

    def aegis_budget_acknowledgement_status(self) -> dict[str, Any]:
        """Return the all-physical-stage budget publication barrier."""

        by_dp: dict[str, dict[str, Any]] = {}
        all_acknowledged = True
        for dp_rank, publisher in sorted(self._aegis_budget_publishers.items()):
            current = publisher.current
            fully_acknowledged = bool(
                current is not None and publisher.fully_acknowledged
            )
            all_acknowledged = all_acknowledged and fully_acknowledged
            by_dp[str(dp_rank)] = {
                "budget_version": (
                    int(current.budget_version) if current is not None else None
                ),
                "window_id": (
                    int(current.window_id) if current is not None else None
                ),
                "acknowledged_pp_ranks": list(
                    publisher.acknowledged_pp_ranks
                ),
                "required_pp_ranks": list(
                    range(int(self.server_args.pp_size))
                ),
                "fully_acknowledged": fully_acknowledged,
            }
        return {
            "all_acknowledged": all_acknowledged,
            "dp_size": len(self.workers),
            "pp_size": int(self.server_args.pp_size),
            "by_dp": by_dp,
        }

    def exact_rolling_state_snapshot(
        self,
        *,
        logical_time_s: float,
        link_window_duration_s: float,
        kv_bytes_per_token_per_stage: int,
    ) -> dict[str, Any]:
        """Return causal stock/flow state for an offline exact rolling solve.

        Runtime code exposes only a solver-free snapshot.  The offline driver
        may feed this immutable state to HiGHS; online serving never imports the
        optimizer.  Resident stock is read from stage-local HBM ledgers and the
        node Host arbiter, while link flow is counted only inside the current
        fixed logical-time bandwidth interval.
        """

        now = float(logical_time_s)
        duration = float(link_window_duration_s)
        bytes_per_token = int(kv_bytes_per_token_per_stage)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("exact rolling logical time must be finite/nonnegative")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("exact rolling link window must be positive/finite")
        if bytes_per_token <= 0:
            raise ValueError("exact rolling KV bytes/token must be positive")
        interval_index = int(math.floor(now / duration))
        interval_start = interval_index * duration
        interval_end = interval_start + duration
        unavailable = tuple(
            rank
            for rank in range(len(self.workers))
            if not self._worker_available(rank)
        )
        with self._aegis_l1_l2_lock:
            rank_states = {
                rank: dict(self._aegis_rank_states.get(rank, {}))
                for rank in range(len(self.workers))
            }
            relevant_acks = [
                dict(event)
                for event in self._aegis_events
                if event.get("kind") == "backup_stage_ack"
                and interval_start <= float(event.get("time", -1.0)) < interval_end
            ]
            event_sequence = int(self._aegis_event_sequence)
        resident_hbm = []
        for dp_rank in range(len(self.workers)):
            stages = rank_states[dp_rank].get("stages", {})
            if not isinstance(stages, dict):
                stages = {}
            resident_hbm.append(
                tuple(
                    max(
                        int(
                            dict(stages.get(str(pp_rank), {}) or {}).get(
                                "l1_used_tokens", 0
                            )
                        ),
                        0,
                    )
                    for pp_rank in range(int(self.server_args.pp_size))
                )
            )
        host_by_dp = [0 for _ in range(len(self.workers))]
        host_by_node: list[tuple[int, int]] = []
        for node_id, arbiter in sorted(self._aegis_node_capacity_arbiters.items()):
            snapshot = arbiter.snapshot()
            host_by_node.append((node_id, int(snapshot["resident_tokens"])))
            for dp_rank, value in snapshot["resident_tokens_by_dp"].items():
                host_by_dp[int(dp_rank)] = int(value)
        link_used: dict[tuple[int, int], int] = {}
        for event in relevant_acks:
            source = int(event.get("source_dp_rank", -1))
            target = int(event.get("target_dp_rank", -1))
            if source < 0 or target < 0:
                continue
            copied_tokens = max(int(event.get("hbm_tokens", 0)), 0) + max(
                int(event.get("host_tokens", 0)), 0
            )
            key = (source, target)
            link_used[key] = link_used.get(key, 0) + (
                copied_tokens * bytes_per_token
            )
        budgets = []
        for dp_rank in range(len(self.workers)):
            current = self._aegis_budget_publishers[dp_rank].current
            if current is None:
                raise RuntimeError(f"DP rank {dp_rank} has no published budget")
            budgets.append(current)
        return {
            "logical_time_s": now,
            "unavailable_dp_ranks": unavailable,
            "resident_hbm_tokens": tuple(resident_hbm),
            "resident_host_tokens_by_dp": tuple(host_by_dp),
            "resident_host_tokens_by_node": tuple(host_by_node),
            "used_link_bandwidth_bytes": tuple(
                (source, target, used)
                for (source, target), used in sorted(link_used.items())
            ),
            "current_hbm_budget_tokens": tuple(
                int(snapshot.hbm_limit_tokens_per_stage[0])
                for snapshot in budgets
            ),
            "current_host_budget_tokens": tuple(
                int(snapshot.host_policy_limit_tokens) for snapshot in budgets
            ),
            "budget_versions": tuple(
                int(snapshot.budget_version) for snapshot in budgets
            ),
            "budget_window_ids": tuple(
                int(snapshot.window_id) for snapshot in budgets
            ),
            "link_window": {
                "index": interval_index,
                "start_s": interval_start,
                "end_s": interval_end,
                "duration_s": duration,
            },
            "event_sequence": event_sequence,
            "semantics": {
                "resident": "stock_at_route_admission",
                "link_used": "admitted_copy_flow_in_fixed_logical_interval",
                "unavailable": "causally_observed_controller_health",
            },
        }

    def install_certified_exact_window(
        self,
        plan: CertifiedExactWindowPlan | dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically publish a certified finite-window route/placement plan.

        The solver is offline-only.  This method consumes its lightweight wire
        contract and publishes one new DP-owned alpha/beta version.  No solver
        package is imported into the runtime controller.
        """

        exact = (
            CertifiedExactWindowPlan.from_dict(plan)
            if isinstance(plan, dict)
            else plan
        ).require_certified_optimal()
        if exact.dp_size != len(self.workers):
            raise ValueError(
                f"exact window DP size {exact.dp_size} != runtime {len(self.workers)}"
            )
        if exact.pp_size != int(self.server_args.pp_size):
            raise ValueError(
                "exact window PP size does not match runtime topology: "
                f"{exact.pp_size} != {self.server_args.pp_size}"
            )
        current_plan = self._aegis_exact_window_plan
        if current_plan is not None and exact.window_id <= current_plan.window_id:
            raise ValueError("exact rolling window_id must increase")

        budget_by_dp = {budget.dp_rank: budget for budget in exact.budgets}
        reuse_current_budget = (
            exact.budget_publication_mode == "reuse_current_fixed"
        )
        snapshots: dict[int, DPBudgetSnapshot] = {}
        publication_snapshots: dict[int, DPBudgetSnapshot] = {}
        hbm_reference = max(self._aegis_gpu_ledger.total_cache_tokens, 1)
        for dp_rank in range(len(self.workers)):
            publisher = self._aegis_budget_publishers[dp_rank]
            current = publisher.current
            if current is None:
                raise RuntimeError(f"DP rank {dp_rank} has no current budget")
            if not publisher.fully_acknowledged:
                raise RuntimeError(
                    "cannot install exact window before every stage ACKs the "
                    f"current DP{dp_rank} budget version {current.budget_version}"
                )
            if not reuse_current_budget and exact.window_id <= current.window_id:
                raise ValueError(
                    f"exact window {exact.window_id} is not newer than DP{dp_rank} "
                    f"window {current.window_id}"
                )
            decision = budget_by_dp[dp_rank]
            if decision.hbm_budget_tokens_per_stage > hbm_reference:
                raise ValueError("exact HBM budget exceeds runtime stage capacity")
            node_id = self._aegis_topology.node_for_rank(dp_rank)
            arbiter = self._aegis_node_capacity_arbiters[node_id]
            host_reference = max(arbiter.physical_capacity_tokens, 1)
            if decision.host_budget_tokens > host_reference:
                raise ValueError("exact Host budget exceeds node physical capacity")
            expected_alpha = (
                decision.hbm_budget_tokens_per_stage / hbm_reference
            )
            expected_beta = decision.host_budget_tokens / host_reference
            if not math.isclose(
                decision.alpha,
                expected_alpha,
                rel_tol=0.0,
                abs_tol=exact.numerical_tolerance,
            ):
                raise ValueError(
                    "exact alpha is not the normalized runtime HBM budget"
                )
            if not math.isclose(
                decision.beta,
                expected_beta,
                rel_tol=0.0,
                abs_tol=exact.numerical_tolerance,
            ):
                raise ValueError(
                    "exact beta is not the normalized runtime Host budget"
                )
            if arbiter.dp_resident_tokens(dp_rank) > decision.host_budget_tokens:
                raise RuntimeError(
                    "exact Host budget is below current DP residency; the "
                    "finite instance omitted required rolling state"
                )
            if reuse_current_budget:
                if current.hbm_limit_tokens_per_stage != tuple(
                    decision.hbm_budget_tokens_per_stage
                    for _ in range(exact.pp_size)
                ) or current.host_policy_limit_tokens != (
                    decision.host_budget_tokens
                ):
                    raise ValueError(
                        "route-only exact window fixed budgets differ from the "
                        f"current DP{dp_rank} publication"
                    )
                if not math.isclose(
                    current.alpha,
                    decision.alpha,
                    rel_tol=0.0,
                    abs_tol=exact.numerical_tolerance,
                ) or not math.isclose(
                    current.beta,
                    decision.beta,
                    rel_tol=0.0,
                    abs_tol=exact.numerical_tolerance,
                ):
                    raise ValueError(
                        "route-only exact window alpha/beta differ from current "
                        f"DP{dp_rank} publication"
                    )
                snapshots[dp_rank] = current
            else:
                snapshot = DPBudgetSnapshot(
                    dp_rank=dp_rank,
                    budget_version=current.budget_version + 1,
                    window_id=exact.window_id,
                    alpha=decision.alpha,
                    beta=decision.beta,
                    hbm_limit_tokens_per_stage=tuple(
                        decision.hbm_budget_tokens_per_stage
                        for _ in range(exact.pp_size)
                    ),
                    host_policy_limit_tokens=decision.host_budget_tokens,
                    valid_from_logical_time=exact.valid_from_logical_time,
                    objective_id=(
                        f"{exact.instance_id}:{exact.instance_sha256}:"
                        f"{exact.objective}"
                    ),
                    solver_certificate_id=exact.solver_certificate_id,
                )
                snapshots[dp_rank] = snapshot
                publication_snapshots[dp_rank] = snapshot

        # All checks precede mutation so a malformed plan cannot partially
        # advance one DP's version.
        for dp_rank, snapshot in publication_snapshots.items():
            self._aegis_budget_publishers[dp_rank].publish(snapshot)
            node_id = self._aegis_topology.node_for_rank(dp_rank)
            self._aegis_node_capacity_arbiters[node_id].publish_dp_budget(
                snapshot
            )
        self._aegis_exact_window_plan = exact
        self._aegis_exact_budget_versions = {
            dp_rank: snapshot.budget_version
            for dp_rank, snapshot in snapshots.items()
        }
        for dp_rank, snapshot in publication_snapshots.items():
            self._send_dp_runtime_command(
                dp_rank,
                ("aegis_publish_dp_budget", snapshot.to_dict()),
            )
        self._append_aegis_event(
            "certified_exact_window_installed",
            {
                "instance_id": exact.instance_id,
                "instance_sha256": exact.instance_sha256,
                "decision_sha256": exact.decision_sha256,
                "solver": exact.solver,
                "solver_version": exact.solver_version,
                "solver_status": exact.solver_status,
                "objective": exact.objective,
                "dual_bound": exact.dual_bound,
                "mip_gap": exact.mip_gap,
                "solve_time_s": exact.solve_time_s,
                "window_id": exact.window_id,
                "route_version": exact.route_version,
                "budget_versions": dict(self._aegis_exact_budget_versions),
                "request_count": len(exact.routes),
                "placement_count": len(exact.placements),
                "uses_future_manifest_oracle": (
                    exact.uses_future_manifest_oracle
                ),
                "budget_publication_mode": exact.budget_publication_mode,
                "budget_reused": reuse_current_budget,
            },
        )
        return {
            "instance_id": exact.instance_id,
            "decision_sha256": exact.decision_sha256,
            "window_id": exact.window_id,
            "route_version": exact.route_version,
            "budget_versions": dict(self._aegis_exact_budget_versions),
            "budget_publication_mode": exact.budget_publication_mode,
            "budget_reused": reuse_current_budget,
            "request_count": len(exact.routes),
        }

    def _record_assigned_rank(self, req: TokenizedGenerateReqInput, dp_rank: int) -> None:
        sim = req.sampling_params.custom_params.setdefault("simulation", {})
        sim["assigned_dp_rank"] = dp_rank

    @staticmethod
    def _policy_name(sim: dict) -> str:
        return (
            str(sim.get("aegis_policy") or ("aegiskv" if sim.get("backup_kv") else ""))
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @classmethod
    def _l1_l2_policy_enabled(cls, sim: dict) -> bool:
        return cls._policy_name(sim) in {
            "aegiskv_l1_l2",
            "aegiskv_v2",
            "aegiskv_demo",
            "aegiskv_l1_only",
            "aegiskv_l2_only",
            "aegiskv_l3_only",
            "aegiskv_no_backup",
        }

    @classmethod
    def _l1_l2_allowed_tiers(cls, sim: dict) -> tuple[str, ...]:
        policy = cls._policy_name(sim)
        if policy == "aegiskv_l1_only":
            return ("l1",)
        if policy == "aegiskv_l2_only":
            return ("l2",)
        if policy == "aegiskv_l3_only":
            return ()
        if policy == "aegiskv_no_backup":
            return ()
        return ("l1", "l2")

    @classmethod
    def _l3_allowed(cls, sim: dict) -> bool:
        policy = cls._policy_name(sim)
        if policy in {
            "aegiskv_no_backup",
            "aegiskv_l1_only",
            "aegiskv_l2_only",
            "no_backup",
            "none",
            "baseline",
        }:
            return False
        return policy in {
            "aegiskv_l1_l2",
            "aegiskv_v2",
            "aegiskv_demo",
            "aegiskv_l3_only",
        }

    @staticmethod
    def _ablation_name(sim: dict) -> str:
        return (
            str(sim.get("aegis_ablation") or "")
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _sim_bool(sim: dict, key: str, default: bool = False) -> bool:
        value = sim.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _sim_int(sim: dict, key: str, default: int = 0) -> int:
        try:
            return int(sim.get(key, default))
        except (TypeError, ValueError):
            return default

    def _effective_aegis_slack_ms(self, sim: dict[str, Any]) -> float:
        """Consume only causally proven Slack for the current design.

        The legacy path intentionally keeps its historical scalar behavior.
        For ``source_owned_independent_hbm_dram_v2`` a positive scalar without
        a visible-residual-DAG CPM ticket fails closed to zero.  Diagnostics
        are metadata-only and do not alter routing or KV state.
        """

        if self._aegis_disable_slack_ordering:
            value, reason = 0.0, "disabled_by_runtime_adapter"
        else:
            value, reason = resolve_slack_for_design(
                sim, design_semantics=self._aegis_design_semantics
            )
        sim["aegis_effective_slack_ms"] = value
        sim["aegis_slack_consumption_reason"] = reason
        sim["aegis_slack_ordering_disabled"] = (
            self._aegis_disable_slack_ordering
        )
        sim["aegis_slack_fail_closed"] = (
            self._aegis_design_semantics == DESIGN_SEMANTICS
            and not self._aegis_disable_slack_ordering
            and reason != "visible_residual_dag_cpm"
        )
        sim["aegis_slack_update_physical_movement"] = False
        return value

    @staticmethod
    def _cache_key(req: TokenizedGenerateReqInput, sim: dict) -> str:
        if sim.get("aegis_cache_key"):
            return str(sim["aegis_cache_key"])
        if req.routing_key:
            return f"route:{req.routing_key}"
        agent_id = str(sim.get("agent_id") or sim.get("worker_id") or "")
        if agent_id:
            workflow_id = str(sim.get("workflow_id") or "")
            return f"agent:{workflow_id}:{agent_id}" if workflow_id else f"agent:{agent_id}"
        return f"rid:{req.rid}"

    @staticmethod
    def _requested_tokens(req: TokenizedGenerateReqInput) -> int:
        input_tokens = (
            int(req.input_token_count)
            if req.input_token_count is not None
            else len(req.input_ids)
        )
        output_tokens = (
            int(req.original_max_new_tokens)
            if req.original_max_new_tokens is not None
            else int(req.sampling_params.max_new_tokens)
        )
        return max(input_tokens + output_tokens, 0)

    def _healthy_ranks(self) -> list[int]:
        return [rank for rank in range(len(self.workers)) if self._worker_available(rank)]

    def _gpu_replica_enabled(self, sim: dict) -> bool:
        policy = self._policy_name(sim)
        ablation = self._ablation_name(sim)
        no_backup_policy = policy in {"baseline", "none", "no_backup", "nobackup"}
        host_only_policy = policy in {
            "host",
            "host_only",
            "hostbackup",
            "host_backup",
            "uniform_host",
        }
        gpu_only_policy = policy in {
            "gpu",
            "gpu_only",
            "gpu_replica",
            "gpureplica",
            "uniform_gpu",
        }
        aegis_policy_enabled = policy in {"aegis", "aegiskv", "dynamic"}
        critical_agent = self._sim_bool(sim, "critical_agent", False)
        backup_kv = self._sim_bool(sim, "backup_kv", False)
        disable_lifecycle = ablation in {"no_lifecycle", "disable_lifecycle"}
        disable_critical_restore = (
            ablation in {"no_critical_restore", "disable_critical_restore"}
            and critical_agent
        )
        size_only_backup = ablation in {"size_only", "size_only_backup"}
        return (
            not no_backup_policy
            and not host_only_policy
            and not disable_lifecycle
            and not disable_critical_restore
            and (
                gpu_only_policy
                or (aegis_policy_enabled and (critical_agent or size_only_backup))
                or (backup_kv and (critical_agent or size_only_backup))
            )
        )

    def _aegis_routing_enabled(self, sim: dict) -> bool:
        policy = self._policy_name(sim)
        if policy in {"baseline", "none", "no_backup", "nobackup"}:
            return False
        return (
            policy
            in {
                "aegis",
                "aegiskv",
                "dynamic",
                "gpu",
                "gpu_only",
                "gpu_replica",
                "gpureplica",
                "uniform_gpu",
                "host",
                "host_only",
                "hostbackup",
                "host_backup",
                "uniform_host",
                "aegiskv_l1_l2",
                "aegiskv_v2",
                "aegiskv_demo",
                "aegiskv_l1_only",
                "aegiskv_l2_only",
                "aegiskv_l3_only",
                "aegiskv_no_backup",
            }
            or self._sim_bool(sim, "backup_kv", False)
        )

    def _resolve_gpu_alpha(self, sim: dict) -> float | None:
        alpha = normalized_fraction(sim.get("aegis_gpu_budget_fraction", -1.0))
        if alpha is not None:
            sim.setdefault("aegis_alpha_policy_source", "request_static")
            return alpha

        policy = self._policy_name(sim)
        if policy in {
            "gpu",
            "gpu_only",
            "gpu_replica",
            "gpureplica",
            "uniform_gpu",
        }:
            sim["aegis_gpu_budget_fraction"] = DEFAULT_GPU_REPLICA_ALPHA
            sim["aegis_alpha_policy_source"] = "gpu_replica_static"
            return DEFAULT_GPU_REPLICA_ALPHA

        if policy in {"aegis", "aegiskv", "dynamic"}:
            failure_rate_raw = sim.get(
                "aegis_failure_rate_per_window",
                sim.get("failure_rate_per_window"),
            )
            failure_rate = (
                None
                if failure_rate_raw is None
                else resolve_failure_rate_per_window(failure_rate_raw)
            )
            critical_pressure = normalized_fraction(
                sim.get("aegis_critical_pressure", -1.0)
            )
            policy_path = str(sim.get("alpha_policy_path") or os.getenv(ENV_ALPHA_POLICY, ""))
            decision = choose_alpha(
                critical_agent=self._sim_bool(sim, "critical_agent", False),
                decode_dominant=self._sim_bool(sim, "decode_dominant", False),
                stage_width=self._sim_int(sim, "stage_width", 1),
                critical_pressure=critical_pressure,
                failure_rate_per_window=failure_rate,
                policy_path=policy_path or None,
            )
            adapter_enabled = online_alpha_adapter_enabled(
                sim.get("aegis_online_alpha_adapter")
            )
            if adapter_enabled:
                sim["aegis_policy_alpha"] = decision.alpha
                decision = self._online_alpha_adapter.tune_decision(
                    decision,
                    critical_agent=self._sim_bool(sim, "critical_agent", False),
                    decode_dominant=self._sim_bool(sim, "decode_dominant", False),
                    stage_width=self._sim_int(sim, "stage_width", 1),
                )
                sim["aegis_online_alpha_delta"] = round(
                    decision.alpha - float(sim["aegis_policy_alpha"]), 3
                )
            sim["aegis_gpu_budget_fraction"] = decision.alpha
            sim["aegis_failure_rate_per_window"] = decision.failure_rate_per_window
            sim["aegis_alpha_policy_source"] = decision.source
            sim["aegis_critical_pressure"] = decision.critical_pressure
            return decision.alpha

        return None

    def _record_aegis_gpu_replica(
        self, req: TokenizedGenerateReqInput, dp_rank: int
    ) -> None:
        sim = req.sampling_params.custom_params.setdefault("simulation", {})
        cache_key = self._cache_key(req, sim)
        requested_tokens = self._requested_tokens(req)
        self._aegis_gpu_ledger.record_inference(
            rid=req.rid,
            dp_rank=dp_rank,
            cache_key=cache_key,
            token_count=requested_tokens,
        )
        sim["aegis_cache_key"] = cache_key
        if self._l1_l2_policy_enabled(sim):
            if not self._track_aegis_frontier_lineage(req, sim):
                return
            self._dispatch_l1_l2_backup_plan(
                req,
                source_dp_rank=dp_rank,
                cache_key=cache_key,
                requested_tokens=requested_tokens,
            )
            self._refresh_aegis_lineage_target(req, sim)
            return
        if not self._gpu_replica_enabled(sim):
            return
        alpha = self._resolve_gpu_alpha(sim)
        if alpha is None or alpha <= 0.0:
            return
        reservation = self._aegis_gpu_ledger.reserve(
            rid=req.rid,
            source_dp_rank=dp_rank,
            cache_key=cache_key,
            requested_tokens=requested_tokens,
            alpha=alpha,
            healthy_ranks=self._healthy_ranks(),
        )
        if reservation is None:
            return
        sim["gpu_replica_dp_rank"] = reservation.backup_dp_rank
        sim["gpu_replica_quota_tokens"] = reservation.quota_tokens
        sim["gpu_replica_overlap_tokens"] = reservation.overlap_tokens
        sim["gpu_replica_unique_tokens"] = reservation.unique_tokens
        sim["gpu_replica_reserved_tokens"] = reservation.reserved_tokens
        sim["gpu_replica_evictable"] = reservation.evictable

    def _append_aegis_event(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._aegis_l1_l2_lock:
            self._aegis_event_sequence += 1
            event = {
                "sequence": self._aegis_event_sequence,
                "time": self._now_fn(),
                "kind": str(kind),
                **dict(payload),
            }
            self._aegis_events.append(event)
            if self._aegis_event_journal is not None:
                self._aegis_event_journal.append(event)
            return dict(event)

    @staticmethod
    def _compact_aegis_recovery_hint(
        hint: dict[str, Any],
    ) -> dict[str, Any]:
        """Project one Router hint into the causal event journal.

        The journal entry is intentionally limited to fields needed to rebuild
        the continuously reusable prefix at a later failure cutoff.  It avoids
        serializing planner candidates or post-hoc request outcomes.
        """

        l1_tokens = max(int(hint.get("l1_tokens", 0)), 0)
        l2_tokens = max(int(hint.get("l2_tokens", 0)), 0)
        l3_tokens = max(int(hint.get("l3_tokens", 0)), 0)
        independent_tiers = (
            str(hint.get("design_semantics") or "") == DESIGN_SEMANTICS
        )
        raw_source_end = hint.get("source_committed_token_end")
        source_committed_token_end = (
            max(int(raw_source_end), 0)
            if raw_source_end is not None
            else None
        )
        return {
            "design_semantics": str(hint.get("design_semantics") or ""),
            "rid": str(hint.get("rid") or ""),
            "backup_key": str(
                hint.get("backup_key") or hint.get("rid") or ""
            ),
            "version": max(int(hint.get("version", 1)), 1),
            "source_dp_rank": int(hint.get("source_dp_rank", -1)),
            "target_dp_rank": int(hint.get("target_dp_rank", -1)),
            "workflow_id": str(hint.get("workflow_id") or ""),
            "agent_id": str(hint.get("agent_id") or ""),
            "worker_seq": max(int(hint.get("worker_seq", 0)), 0),
            "l1_tokens": l1_tokens,
            "l2_tokens": l2_tokens,
            "l3_tokens": l3_tokens,
            "continuous_recoverable_prefix_tokens": (
                max(
                    int(
                        hint.get(
                            "safe_frontier_tokens",
                            max(l1_tokens, l2_tokens, l3_tokens),
                        )
                    ),
                    0,
                )
                if independent_tiers
                else l1_tokens + l2_tokens + l3_tokens
            ),
            "materialization_scope": str(
                hint.get("materialization_scope")
                or "metadata_reservation_only"
            ),
            "source_committed_token_end": source_committed_token_end,
            "group_commit_verified": bool(
                hint.get("group_commit_verified", False)
            ),
            "copy_completion_evidence": str(
                hint.get("copy_completion_evidence") or ""
            ),
            "execution_id": str(hint.get("execution_id") or ""),
            "block_generation": int(hint.get("block_generation", -1)),
            "recompute_tokens": max(
                int(hint.get("recompute_tokens", 0)), 0
            ),
            "protected_recovery_service_s": max(
                _nonnegative_float_or_zero(
                    hint.get("protected_recovery_service_s", 0.0)
                ),
                0.0,
            ),
            "unprotected_recovery_service_s": max(
                _nonnegative_float_or_zero(
                    hint.get("unprotected_recovery_service_s", 0.0)
                ),
                0.0,
            ),
            "recovery_service_cost_source": str(
                hint.get("recovery_service_cost_source") or ""
            ),
            "recovery_service_cost_calibration_id": str(
                hint.get("recovery_service_cost_calibration_id") or ""
            ),
            "model_layout_id": str(hint.get("model_layout_id") or ""),
            "layout_compatible": hint.get("layout_compatible"),
            "all_stages_committed": bool(
                hint.get("all_stages_committed", True)
            ),
            "required_stage_count": max(
                int(hint.get("required_stage_count", 1)), 1
            ),
            "updated_time": float(hint.get("updated_time", 0.0)),
        }

    def close_event_journal(self) -> None:
        journal = self._aegis_event_journal
        if journal is not None:
            journal.close()

    def _record_exact_deployment_violation(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        violation = {
            "time": self._now_fn(),
            **dict(payload),
        }
        with self._aegis_l1_l2_lock:
            self._aegis_exact_deployment_violations.append(violation)
        self._append_aegis_event(
            "certified_exact_deployment_violation", violation
        )
        return violation

    @staticmethod
    def _backup_key(req: TokenizedGenerateReqInput, sim: dict) -> str:
        return str(sim.get("aegis_backup_key") or req.rid)

    def _track_aegis_frontier_lineage(
        self,
        req: TokenizedGenerateReqInput,
        sim: dict[str, Any],
    ) -> bool:
        """Retire the previous frontier when a newer agent turn is routed.

        Request-local backup keys remain the recovery lookup keys.  This
        controller-level lineage index only controls their lifetime; it does
        not alter Algorithm 1 Slack or Algorithm 2 tier scoring.
        """

        if not self._sim_bool(sim, "backup_kv", False):
            return True
        if self._sim_bool(sim, "failover_replay", False):
            return True
        workflow_id = str(sim.get("workflow_id") or "")
        agent_id = str(sim.get("agent_id") or "")
        worker_seq = self._sim_int(
            sim,
            "worker_seq",
            self._sim_int(sim, "aegis_worker_seq", 0),
        )
        if not workflow_id or not agent_id or worker_seq <= 0:
            return True

        backup_key = self._backup_key(req, sim)
        version = max(self._sim_int(sim, "aegis_backup_version", 1), 1)
        target_rank = self._sim_int(sim, "aegis_backup_dp_rank", -1)
        lineage = (workflow_id, agent_id)
        current = {
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "worker_seq": worker_seq,
            "backup_key": backup_key,
            "version": version,
            "target_dp_rank": target_rank,
            "rid": str(req.rid),
        }
        previous: dict[str, Any] | None = None
        stale = False
        with self._aegis_l1_l2_lock:
            latest = self._aegis_lineage_latest.get(lineage)
            if latest is not None:
                latest_seq = int(latest.get("worker_seq", 0))
                latest_key = str(latest.get("backup_key") or "")
                if worker_seq < latest_seq or (
                    worker_seq == latest_seq and backup_key != latest_key
                ):
                    stale = True
                elif worker_seq > latest_seq and backup_key != latest_key:
                    previous = dict(latest)
                    self._aegis_superseded_backups[latest_key] = {
                        **previous,
                        "superseded_by": backup_key,
                        "superseded_by_worker_seq": worker_seq,
                    }
                    self._aegis_recovery_hints.pop(latest_key, None)
            if not stale:
                self._aegis_lineage_latest[lineage] = current
                tombstone = self._aegis_superseded_backups.get(backup_key)
                if (
                    tombstone is not None
                    and version > int(tombstone.get("version", -1))
                ):
                    self._aegis_superseded_backups.pop(backup_key, None)

        if stale:
            sim["aegis_lineage_plan_stale"] = True
            self._append_aegis_event(
                "stale_lineage_plan_ignored",
                {
                    **current,
                    "latest_worker_seq": int(latest.get("worker_seq", 0)),
                    "latest_backup_key": str(latest.get("backup_key") or ""),
                },
            )
            return False
        if previous is None:
            return True

        previous_key = str(previous["backup_key"])
        previous_version = max(int(previous.get("version", 1)), 1)
        pp_size = int(self.server_args.pp_size)
        physical_keys = (
            tuple(
                self._stage_backup_key(previous_key, pp_rank)
                for pp_rank in range(pp_size)
            )
            if pp_size > 1
            else ()
        )
        released_host_tokens = 0
        released_l3_tokens = 0
        for release_key in (previous_key, *physical_keys):
            for node_id, pool in self._aegis_host_pools.items():
                lease = pool.release(
                    release_key, max_version=previous_version
                )
                if lease is not None:
                    released_host_tokens += max(
                        int(lease.allocated_tokens), 0
                    )
                self._aegis_node_capacity_arbiters[node_id].release(
                    release_key
                )
            persistent_lease = self._aegis_l3_pool.release(
                release_key,
                max_version=previous_version,
            )
            if persistent_lease is not None:
                released_l3_tokens += max(
                    int(persistent_lease.token_count), 0
                )
        previous_target = int(previous.get("target_dp_rank", -1))
        if self._has_runtime_command_target(previous_target):
            base_command = {
                "expected_version": previous_version,
                "superseded_by": backup_key,
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "worker_seq": int(previous.get("worker_seq", 0)),
                "superseded_by_worker_seq": worker_seq,
                "request_backup_key": previous_key,
            }
            if pp_size > 1:
                for pp_rank, stage_key in enumerate(physical_keys):
                    self._send_stage_runtime_command(
                        previous_target,
                        pp_rank,
                        (
                            "aegis_l1_l2_supersede_backup",
                            {
                                **base_command,
                                "backup_key": stage_key,
                                "pp_rank": pp_rank,
                                "required_pp_size": pp_size,
                            },
                        ),
                    )
            else:
                self._send_dp_runtime_command(
                    previous_target,
                    (
                        "aegis_l1_l2_supersede_backup",
                        {**base_command, "backup_key": previous_key},
                    ),
                )
        sim["aegis_superseded_backup_key"] = previous_key
        self._append_aegis_event(
            "backup_frontier_superseded",
            {
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "backup_key": previous_key,
                "version": previous_version,
                "worker_seq": int(previous.get("worker_seq", 0)),
                "target_dp_rank": previous_target,
                "superseded_by": backup_key,
                "superseded_by_worker_seq": worker_seq,
                "released_host_allocated_tokens": released_host_tokens,
                "released_l3_tokens": released_l3_tokens,
            },
        )
        return True

    def _refresh_aegis_lineage_target(
        self,
        req: TokenizedGenerateReqInput,
        sim: dict[str, Any],
    ) -> None:
        """Persist the immutable Router target selected after lineage admission."""

        workflow_id = str(sim.get("workflow_id") or "")
        agent_id = str(sim.get("agent_id") or "")
        backup_key = self._backup_key(req, sim)
        target_dp_rank = self._sim_int(sim, "aegis_backup_dp_rank", -1)
        if not workflow_id or not agent_id or target_dp_rank < 0:
            return
        with self._aegis_l1_l2_lock:
            current = self._aegis_lineage_latest.get((workflow_id, agent_id))
            if (
                isinstance(current, dict)
                and str(current.get("backup_key") or "") == backup_key
            ):
                current["target_dp_rank"] = target_dp_rank

    def _dispatch_l1_l2_backup_plan(
        self,
        req: TokenizedGenerateReqInput,
        *,
        source_dp_rank: int,
        cache_key: str,
        requested_tokens: int,
    ) -> None:
        sim = req.sampling_params.custom_params.setdefault("simulation", {})
        allowed_tiers = self._l1_l2_allowed_tiers(sim)
        l3_allowed = self._l3_allowed(sim)
        if not allowed_tiers and not l3_allowed:
            return
        if not self._sim_bool(sim, "backup_kv", bool(sim.get("critical_agent", False))):
            return
        backup_rank = self._sim_int(sim, "aegis_backup_dp_rank", -1)
        if (
            backup_rank < 0
            or backup_rank == int(source_dp_rank)
            or backup_rank >= len(self.workers)
            or not self._has_runtime_command_target(backup_rank)
        ):
            return
        exact_window = self._aegis_exact_window_plan
        exact_route = (
            exact_window.route_for(req.rid)
            if exact_window is not None
            else None
        )
        critical_probability = normalized_fraction(
            sim.get("aegis_critical_probability", -1.0)
        )
        if critical_probability is None:
            critical_probability = 0.90 if self._sim_bool(sim, "critical_agent") else 0.35
        failure_scope = normalize_failure_scope(
            sim.get("aegis_failure_scope", "node")
        )
        placement_request = AegisPlacementRequest(
            rid=req.rid,
            token_count=max(int(requested_tokens), 0),
            source_dp_rank=int(source_dp_rank),
            critical_probability=critical_probability,
            slack_lcb_ms=self._effective_aegis_slack_ms(sim),
            failure_scope=failure_scope,
        )
        rank_state = self._aegis_rank_states.get(backup_rank, {})
        gpu_pressure = float(rank_state.get("normal_l1_pressure", 0.0))
        host_pool_node = self._aegis_topology.node_for_rank(backup_rank)
        budget_snapshot = self._aegis_budget_publishers[backup_rank].current
        if budget_snapshot is None:
            raise RuntimeError(f"DP rank {backup_rank} has no published budget")
        host_capacity = self._aegis_node_capacity_arbiters[
            host_pool_node
        ].snapshot()
        node_host_pressure = float(host_capacity["resident_tokens"]) / max(
            int(host_capacity["physical_capacity_tokens"]), 1
        )
        dp_host_pressure = float(
            host_capacity["resident_tokens_by_dp"].get(backup_rank, 0)
        ) / max(
            int(budget_snapshot.host_policy_limit_tokens), 1
        )
        host_pressure = max(node_host_pressure, dp_host_pressure)
        gpu_candidate = self._aegis_placement_planner.gpu_candidate(
            placement_request,
            target_dp_rank=backup_rank,
            capacity_pressure=gpu_pressure,
        )
        host_candidate = self._aegis_placement_planner.host_candidate(
            placement_request,
            host_pool_node_id=host_pool_node,
            retry_dp_rank=backup_rank,
            capacity_pressure=host_pressure,
        )
        l3_snapshot = self._aegis_l3_pool.snapshot()
        l3_candidate = self._aegis_placement_planner.l3_candidate(
            placement_request,
            retry_dp_rank=backup_rank,
            storage_available=(
                l3_allowed
                and int(l3_snapshot["capacity_tokens"]) > 0
            ),
            manifest_quorum=int(l3_snapshot["manifest_quorum"]),
            capacity_pressure=(
                float(l3_snapshot["used_tokens"])
                / max(int(l3_snapshot["capacity_tokens"]), 1)
            ),
        )
        allowed_tiers = tuple(
            tier
            for tier in allowed_tiers
            if (tier == "l1" and gpu_candidate.valid)
            or (tier == "l2" and host_candidate.valid)
        )
        if (
            not allowed_tiers
            and not l3_candidate.valid
            and exact_route is None
        ):
            self._append_aegis_event(
                "backup_plan_rejected",
                {
                    "rid": req.rid,
                    "backup_key": self._backup_key(req, sim),
                    "failure_scope": failure_scope,
                    "gpu_reason": gpu_candidate.invalid_reason,
                    "host_reason": host_candidate.invalid_reason,
                    "l3_reason": l3_candidate.invalid_reason,
                },
            )
            return
        route_version = max(
            self._sim_int(
                sim,
                "aegis_route_version",
                self._sim_int(sim, "route_version", 1),
            ),
            1,
        )
        budget_version = budget_snapshot.budget_version
        plan = {
            "rid": req.rid,
            "backup_key": self._backup_key(req, sim),
            "source_dp_rank": int(source_dp_rank),
            "target_dp_rank": int(backup_rank),
            "token_count": max(int(requested_tokens), 0),
            "critical_probability": critical_probability,
            "slack_ms": self._effective_aegis_slack_ms(sim),
            "workflow_id": str(sim.get("workflow_id") or ""),
            "agent_id": str(sim.get("agent_id") or ""),
            "worker_seq": max(self._sim_int(sim, "worker_seq", 0), 0),
            "allowed_tiers": list(allowed_tiers),
            "failure_scope": failure_scope,
            "gpu_candidate": gpu_candidate.to_dict(),
            "host_candidate": host_candidate.to_dict(),
            "l3_candidate": l3_candidate.to_dict(),
            "dp_host_beta": budget_snapshot.beta,
            "host_budget_owner": "dp_controller",
            "node_host_capacity_owner": "node_physical_arbiter",
            "version": max(self._sim_int(sim, "aegis_backup_version", 1), 1),
            "route_version": route_version,
            "budget_version": budget_version,
            "created_time": self._now_fn(),
        }
        if exact_route is not None:
            if (
                exact_route.execution_dp_rank != int(source_dp_rank)
                or exact_route.backup_dp_rank != int(backup_rank)
            ):
                raise ValueError(
                    "runtime route differs from certified exact route for "
                    f"request {req.rid}"
                )
            expected_budget_version = self._aegis_exact_budget_versions.get(
                backup_rank
            )
            if expected_budget_version != budget_version:
                raise ValueError(
                    "certified exact plan budget version is not current on "
                    f"DP{backup_rank}"
                )
            if (
                exact_window.budget_publication_mode == "publish_new"
                and
                budget_snapshot.solver_certificate_id
                != exact_window.solver_certificate_id
            ):
                raise ValueError(
                    "DP budget certificate differs from exact placement plan"
                )
            plan.update(
                {
                    "placement_mode": "certified_exact",
                    "solver_instance_sha256": exact_window.instance_sha256,
                    "solver_decision_sha256": exact_window.decision_sha256,
                    "solver_certificate_id": (
                        exact_window.solver_certificate_id
                    ),
                    "exact_window_id": exact_window.window_id,
                    "exact_budget_publication_mode": (
                        exact_window.budget_publication_mode
                    ),
                    "budget_solver_certificate_id": (
                        budget_snapshot.solver_certificate_id
                    ),
                }
            )
        pp_size = int(self.server_args.pp_size)
        for pp_rank in range(pp_size):
            stage_plan = {
                **plan,
                "request_backup_key": plan["backup_key"],
                "backup_key": (
                    self._stage_backup_key(str(plan["backup_key"]), pp_rank)
                    if pp_size > 1
                    else str(plan["backup_key"])
                ),
                "source_pp_rank": pp_rank,
                "target_pp_rank": pp_rank,
                "required_pp_size": pp_size,
                "dp_budget_snapshot": budget_snapshot.to_dict(),
            }
            if exact_route is not None:
                assert exact_window is not None
                placements = exact_window.placements_for(req.rid, pp_rank)
                if sum(row.token_count for row in placements) != int(
                    requested_tokens
                ):
                    raise ValueError(
                        "certified exact stage placement token total does not "
                        f"match request {req.rid}"
                    )
                stage_plan["exact_chunk_placements"] = [
                    {
                        "chunk_id": row.chunk_id,
                        "token_count": row.token_count,
                        "tier": row.tier,
                    }
                    for row in placements
                ]
            self._send_stage_runtime_command(
                backup_rank,
                pp_rank,
                ("aegis_l1_l2_backup_plan", stage_plan),
            )
            with self._aegis_l1_l2_lock:
                self._aegis_stage_reservation_plans[
                    (
                        str(plan["backup_key"]),
                        int(plan["version"]),
                        int(pp_rank),
                    )
                ] = dict(stage_plan)
        sim["aegis_backup_plan_sent"] = True
        # Persist the exact Router reservation identity on the request.  The
        # source Scheduler must publish commit frontiers against this key, not
        # fall back to a routing/cache-affinity key that may name an agent.
        sim["aegis_backup_key"] = str(plan["backup_key"])
        sim["aegis_backup_dp_rank"] = backup_rank
        sim["gpu_replica_dp_rank"] = backup_rank
        sim["aegis_failure_scope"] = failure_scope
        sim["aegis_host_pool_node_id"] = host_pool_node
        sim["aegis_gpu_direct_reuse_dp_rank"] = backup_rank
        sim["aegis_route_version"] = route_version
        sim["aegis_budget_version"] = budget_version
        sim["aegis_reserved_token_count"] = max(int(requested_tokens), 0)
        sim["aegis_block_generation"] = max(
            self._sim_int(
                sim,
                "aegis_block_generation",
                int(plan["version"]) - 1,
            ),
            0,
        )
        sim["aegis_copy_execution_id"] = str(
            sim.get("aegis_copy_execution_id")
            or (
                f"{req.rid}:v{plan['version']}:route{route_version}:"
                f"source{source_dp_rank}:t{int(float(plan['created_time']) * 1e9)}"
            )
        )
        sim["aegis_routing_decision"] = {
            "request_id": req.rid,
            "execution_dp_rank": int(source_dp_rank),
            "backup_dp_rank": int(backup_rank),
            "route_version": route_version,
        }
        self._append_aegis_event(
            "backup_plan",
            {
                "rid": req.rid,
                "backup_key": plan["backup_key"],
                "execution_dp_rank": int(source_dp_rank),
                "backup_dp_rank": int(backup_rank),
                "tokens": int(requested_tokens),
                "failure_scope": failure_scope,
                "gpu_backup_path": list(gpu_candidate.backup_path),
                "gpu_reusable_dp_ranks": list(gpu_candidate.reusable_dp_ranks),
                "pp_gpu_mapping": list(gpu_candidate.pp_gpu_mapping),
                "host_pool_node_id": host_pool_node,
                "host_backup_path": list(host_candidate.backup_path),
                "host_recovery_path": list(host_candidate.recovery_path),
                "l3_backup_path": list(l3_candidate.backup_path),
                "l3_recovery_path": list(l3_candidate.recovery_path),
                "route_version": route_version,
                "budget_version": budget_version,
                "required_pp_stages": pp_size,
                "placement_mode": plan.get(
                    "placement_mode", "legacy_greedy"
                ),
                "solver_instance_sha256": plan.get(
                    "solver_instance_sha256", ""
                ),
                "solver_decision_sha256": plan.get(
                    "solver_decision_sha256", ""
                ),
            },
        )

    def _finalize_aegis_request_completion(
        self,
        *,
        backup_key: str,
        expected_version: int,
        target_dp_rank: int,
        reason: str,
    ) -> dict[str, Any]:
        """Release Router-owned metadata after all target stages are terminal."""

        key = str(backup_key)
        version = int(expected_version)
        released_hint: dict[str, Any] | None = None
        with self._aegis_l1_l2_lock:
            current = self._aegis_recovery_hints.get(key)
            if isinstance(current, dict) and int(current.get("version", -1)) == version:
                released_hint = dict(current)
                self._aegis_recovery_hints.pop(key, None)
            for lineage, latest in tuple(self._aegis_lineage_latest.items()):
                if (
                    str(latest.get("backup_key") or "") == key
                    and int(latest.get("version", -1)) == version
                ):
                    self._aegis_lineage_latest.pop(lineage, None)
            for pending_key in tuple(self._aegis_stage_reservation_plans):
                if pending_key[0] == key and int(pending_key[1]) == version:
                    self._aegis_stage_reservation_plans.pop(pending_key, None)

        pp_size = int(self.server_args.pp_size)
        physical_keys = (
            tuple(self._stage_backup_key(key, rank) for rank in range(pp_size))
            if pp_size > 1
            else ()
        )
        released_host_tokens = 0
        released_l3_tokens = 0
        for release_key in (key, *physical_keys):
            for node_id, pool in self._aegis_host_pools.items():
                lease = pool.release(release_key, max_version=version)
                if lease is not None:
                    released_host_tokens += max(
                        int(lease.allocated_tokens), 0
                    )
                self._aegis_node_capacity_arbiters[node_id].release(release_key)
            lease = self._aegis_l3_pool.release(
                release_key, max_version=version
            )
            if lease is not None:
                released_l3_tokens += max(int(lease.token_count), 0)
        rid = str((released_hint or {}).get("rid") or "")
        if rid:
            self._aegis_gpu_ledger.release_request(rid)
        result = {
            "backup_key": key,
            "expected_version": version,
            "target_dp_rank": int(target_dp_rank),
            "completed": True,
            "router_hint_released": released_hint is not None,
            "released_host_allocated_tokens": released_host_tokens,
            "released_l3_tokens": released_l3_tokens,
            "reason": str(reason),
        }
        self._append_aegis_event("request_completion_applied", result)
        return result

    def complete_aegis_request(
        self,
        *,
        backup_key: str,
        expected_version: int,
        target_dp_rank: int,
    ) -> dict[str, Any]:
        """Dispatch latest-design ``CompleteRequest`` to every target PP stage."""

        key = str(backup_key)
        if not key:
            raise ValueError("CompleteRequest requires a backup key")
        version = max(int(expected_version), 1)
        requested_target = int(target_dp_rank)
        with self._aegis_l1_l2_lock:
            current_hint = self._aegis_recovery_hints.get(key)
            if (
                isinstance(current_hint, dict)
                and int(current_hint.get("version", -1)) == version
            ):
                current_target = int(
                    current_hint.get("target_dp_rank", requested_target)
                )
                rotation = current_hint.get("hbm_target_rotation")
                if (
                    isinstance(rotation, dict)
                    and not bool(rotation.get("degraded", False))
                    and requested_target
                    in {
                        int(rank)
                        for rank in rotation.get(
                            "failed_target_dp_ranks", ()
                        )
                    }
                ):
                    target_dp_rank = current_target
        self._validate_rank(target_dp_rank)
        pp_size = int(self.server_args.pp_size)
        if not self._has_runtime_command_target(target_dp_rank):
            return {
                **self._finalize_aegis_request_completion(
                    backup_key=key,
                    expected_version=version,
                    target_dp_rank=target_dp_rank,
                    reason="target_unavailable_at_complete_request",
                ),
                "dispatched": False,
            }
        pending_key = (key, version, int(target_dp_rank))
        with self._aegis_l1_l2_lock:
            if pending_key in self._aegis_stage_completion_pending:
                return {
                    "backup_key": key,
                    "expected_version": version,
                    "target_dp_rank": int(target_dp_rank),
                    "completed": False,
                    "dispatched": False,
                    "reason": "completion_already_pending",
                }
            self._aegis_stage_completion_pending[pending_key] = {
                "results": {},
                "required_pp_size": pp_size,
            }
        for pp_rank in range(pp_size):
            stage_key = (
                self._stage_backup_key(key, pp_rank) if pp_size > 1 else key
            )
            self._send_stage_runtime_command(
                target_dp_rank,
                pp_rank,
                (
                    "aegis_l1_l2_complete_request",
                    {
                        "request_backup_key": key,
                        "stage_backup_key": stage_key,
                        "expected_version": version,
                        "required_pp_size": pp_size,
                    },
                ),
            )
        self._append_aegis_event(
            "request_completion_dispatched",
            {
                "backup_key": key,
                "expected_version": version,
                "target_dp_rank": int(target_dp_rank),
                "requested_target_dp_rank": requested_target,
                "required_pp_size": pp_size,
            },
        )
        return {
            "backup_key": key,
            "expected_version": version,
            "target_dp_rank": int(target_dp_rank),
            "requested_target_dp_rank": requested_target,
            "completed": False,
            "dispatched": True,
            "required_pp_size": pp_size,
        }

    @staticmethod
    def _canonical_aegis_physical_tier(tier: object) -> str:
        canonical = str(tier).strip().lower()
        if canonical not in {"hbm", "dram"}:
            raise ValueError(
                f"unsupported physical AegisKV tier {tier!r}; "
                "expected hbm or dram"
            )
        return canonical

    def commit_aegis_physical_tier(
        self,
        event: dict[str, Any],
        *,
        completed_tier: str,
        replay_after_target_restart: bool = False,
    ) -> bool:
        """Publish one atomically completed physical tier to its Scheduler.

        ``event`` is the original source-frontier event enriched with the
        :class:`TierNotification` fields from the offline physical transfer
        bridge.  A notification is accepted only when every expected TP shard
        ACKed the named GlobalDES group.  The group's ``token_end`` is the only
        frontier materialized; a later group for the other tier cannot make
        this tier recoverable.  Duplicate notifications are a no-op, while a
        reused group identity with different evidence fails closed.
        """

        if not isinstance(event, dict):
            raise TypeError("physical tier commit event must be a dict")
        tier = self._canonical_aegis_physical_tier(completed_tier)
        if not self._aegis_committed_copy_protocol:
            raise RuntimeError("physical tier commit requires committed-copy mode")
        if self._aegis_design_semantics != DESIGN_SEMANTICS:
            raise RuntimeError(
                "physical tier commit requires latest independent-tier semantics"
            )
        event = self._resolve_aegis_rotated_source_event(event)

        group_id = str(event.get("group_id") or "")
        if not group_id:
            raise ValueError("physical tier commit requires group_id")
        expected_shards = int(event.get("expected_shards", -1))
        acked_shards = int(event.get("acked_shards", -1))
        state = str(event.get("state") or "")
        notification_tier = str(event.get("tier") or "").strip().lower()
        expected_notification_tiers = {
            "hbm": {"hbm", "gpu_aegis"},
            "dram": {"dram", "node_dram"},
        }[tier]
        if not (
            event.get("success") is True
            and state == "COMMITTED"
            and expected_shards > 0
            and acked_shards == expected_shards
            and notification_tier in expected_notification_tiers
        ):
            self._append_aegis_event(
                "physical_tier_commit_rejected",
                {
                    "group_id": group_id,
                    "completed_tier": tier,
                    "state": state,
                    "success": event.get("success"),
                    "expected_shards": expected_shards,
                    "acked_shards": acked_shards,
                    "reason": "physical_group_not_fully_committed",
                },
            )
            return False

        token_start = int(event.get("token_start", -1))
        token_end = int(event.get("token_end", -1))
        if token_start < 0 or token_end <= token_start:
            raise ValueError("physical tier commit has an invalid token range")
        commit_event = {
            **dict(event),
            # A cumulative source event may have created several physical
            # groups.  Only this group's range has completed at this callback.
            "source_committed_token_end": token_end,
            "physical_group_id": group_id,
            "physical_group_commit_verified": True,
        }
        fingerprint = {
            key: commit_event.get(key)
            for key in (
                "request_backup_key",
                "source_dp_rank",
                "target_dp_rank",
                "source_pp_rank",
                "target_pp_rank",
                "required_pp_size",
                "version",
                "route_version",
                "budget_version",
                "source_committed_token_end",
                "reserved_token_count",
                "execution_id",
                "block_generation",
                "expected_shards",
                "acked_shards",
            )
        }
        fingerprint["completed_tier"] = tier

        physical_lock = getattr(
            self, "_aegis_physical_tier_commit_lock", None
        )
        if physical_lock is None:
            # Support focused unit construction through ``__new__`` without
            # weakening the production constructor's synchronization.
            physical_lock = threading.Lock()
            self._aegis_physical_tier_commit_lock = physical_lock
            self._aegis_physical_tier_commits = {}
        with physical_lock:
            prior = self._aegis_physical_tier_commits.get(group_id)
            if prior is not None:
                if prior["fingerprint"] != fingerprint:
                    raise ValueError(
                        "physical tier group identity was reused with "
                        "conflicting commit evidence"
                    )
                if not (
                    bool(replay_after_target_restart) and tier == "dram"
                ):
                    return False
            dispatched = self._apply_aegis_source_chunk_committed(
                commit_event,
                completed_tiers=(tier,),
                physical_group_id=group_id,
                require_source_live=False,
            )
            self._aegis_physical_tier_commits[group_id] = {
                "fingerprint": fingerprint,
                "dispatched": bool(dispatched),
                "dispatch_attempts": int(
                    (prior or {}).get("dispatch_attempts", 0)
                )
                + 1,
            }
            return bool(dispatched)

    def _apply_aegis_source_chunk_committed(
        self,
        event: dict[str, Any],
        *,
        completed_tiers: tuple[str, ...] | None = None,
        physical_group_id: str = "",
        require_source_live: bool = True,
    ) -> bool:
        """Forward one post-forward source frontier to the aligned target.

        This is deliberately distinct from the admission-time reservation.
        The command is issued only while both source and target are live; the
        later target ``GroupCommitAck`` is the recoverability boundary.
        """

        if not self._aegis_committed_copy_protocol:
            return False
        root_key = str(event.get("request_backup_key") or "")
        source_dp_rank = int(event.get("source_dp_rank", -1))
        target_dp_rank = int(event.get("target_dp_rank", -1))
        source_pp_rank = int(event.get("source_pp_rank", -1))
        target_pp_rank = int(event.get("target_pp_rank", -1))
        required_pp_size = int(event.get("required_pp_size", -1))
        version = max(int(event.get("version", 1)), 1)
        source_end = int(event.get("source_committed_token_end", -1))
        reserved_tokens = int(event.get("reserved_token_count", -1))
        execution_id = str(event.get("execution_id") or "")
        if not root_key or not execution_id:
            raise ValueError("source chunk commit lacks backup/execution identity")
        self._validate_rank(source_dp_rank)
        self._validate_rank(target_dp_rank)
        if source_dp_rank == target_dp_rank:
            raise ValueError("source chunk commit cannot target its source DP")
        if required_pp_size != int(self.server_args.pp_size):
            raise ValueError(
                "source chunk commit PP size differs from runtime topology"
            )
        if not (
            0 <= source_pp_rank < required_pp_size
            and source_pp_rank == target_pp_rank
        ):
            raise ValueError(
                "source chunk commit must use the same valid PP stage"
            )
        if source_end < 0 or reserved_tokens < 0 or source_end > reserved_tokens:
            raise ValueError(
                "source chunk commit exceeds admission-time reservation"
            )
        if require_source_live and not self._worker_available(source_dp_rank):
            self._append_aegis_event(
                "source_chunk_commit_rejected",
                {
                    "rid": str(event.get("rid") or ""),
                    "backup_key": root_key,
                    "source_dp_rank": source_dp_rank,
                    "target_dp_rank": target_dp_rank,
                    "pp_rank": source_pp_rank,
                    "version": version,
                    "reason": "source_unavailable_before_copy_dispatch",
                },
            )
            return False
        if not (
            self._worker_available(target_dp_rank)
            and self._has_runtime_command_target(target_dp_rank)
        ):
            self._append_aegis_event(
                "source_chunk_commit_rejected",
                {
                    "rid": str(event.get("rid") or ""),
                    "backup_key": root_key,
                    "source_dp_rank": source_dp_rank,
                    "target_dp_rank": target_dp_rank,
                    "pp_rank": source_pp_rank,
                    "version": version,
                    "reason": "target_unavailable_before_copy_dispatch",
                },
            )
            return False
        stage_backup_key = (
            self._stage_backup_key(root_key, target_pp_rank)
            if required_pp_size > 1
            else root_key
        )
        command = {
            **dict(event),
            "request_backup_key": root_key,
            "stage_backup_key": stage_backup_key,
        }
        if completed_tiers is not None:
            command.update(
                {
                    "completed_tiers": list(completed_tiers),
                    "physical_group_id": str(physical_group_id),
                    "physical_group_commit_verified": True,
                }
            )
        self._append_aegis_event(
            (
                "physical_tier_commit_dispatched"
                if completed_tiers is not None
                else "source_chunk_committed"
            ),
            {
                "rid": str(event.get("rid") or ""),
                "backup_key": root_key,
                "stage_backup_key": stage_backup_key,
                "source_dp_rank": source_dp_rank,
                "target_dp_rank": target_dp_rank,
                "pp_rank": source_pp_rank,
                "version": version,
                "route_version": max(int(event.get("route_version", 1)), 1),
                "budget_version": max(
                    int(event.get("budget_version", 1)), 1
                ),
                "source_committed_token_end": source_end,
                "reserved_token_count": reserved_tokens,
                "execution_id": execution_id,
                "block_generation": int(event.get("block_generation", -1)),
                "source_observed_time": float(
                    event.get("observed_time", self._now_fn())
                ),
                "completed_tiers": (
                    list(completed_tiers)
                    if completed_tiers is not None
                    else None
                ),
                "physical_group_id": str(physical_group_id),
            },
        )
        self._send_stage_runtime_command(
            target_dp_rank,
            target_pp_rank,
            ("aegis_materialize_committed_prefix", command),
        )
        return True

    def _aegis_rotation_for_identity(
        self,
        *,
        backup_key: str,
        version: int,
        route_version: int,
    ) -> dict[str, Any] | None:
        with self._aegis_l1_l2_lock:
            rotation = self._aegis_target_rotations.get(
                (str(backup_key), int(version), int(route_version))
            )
            return dict(rotation) if isinstance(rotation, dict) else None

    def _resolve_aegis_rotated_source_event(
        self, event: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve an admission-time source target through the Router fence."""

        root_key = str(event.get("request_backup_key") or "")
        version = max(int(event.get("version", 1)), 1)
        route_version = max(int(event.get("route_version", 1)), 1)
        rotation = self._aegis_rotation_for_identity(
            backup_key=root_key,
            version=version,
            route_version=route_version,
        )
        if rotation is None or rotation.get("degraded", False):
            return dict(event)
        resolved = dict(event)
        observed_target = int(resolved.get("target_dp_rank", -1))
        replacement = int(rotation["replacement_target_dp_rank"])
        failed_targets = {
            int(rank) for rank in rotation.get("failed_target_dp_ranks", ())
        }
        if observed_target == replacement:
            return resolved
        if observed_target not in failed_targets:
            raise ValueError(
                "source event target conflicts with the Router rotation fence"
            )
        resolved["target_dp_rank"] = replacement
        resolved["hbm_target_rotation"] = dict(rotation)
        resolved["source_event_target_dp_rank"] = observed_target
        return resolved

    def _merge_aegis_rotation_ack_payload(
        self,
        *,
        root_key: str,
        event: dict[str, Any],
        payload: dict[str, Any],
        is_group_commit: bool,
    ) -> tuple[dict[str, Any], bool, bool]:
        """Merge replacement HBM ACKs without rewriting independent DRAM.

        Returns ``(payload, is_rotation_ack, is_stale_old_target_ack)``.
        """

        version = max(int(payload.get("version", 1)), 1)
        route_version = max(
            int(event.get("route_version", payload.get("route_version", 1))),
            1,
        )
        rotation = self._aegis_rotation_for_identity(
            backup_key=root_key,
            version=version,
            route_version=route_version,
        )
        if rotation is None or rotation.get("degraded", False):
            return dict(payload), False, False
        target = int(payload.get("target_dp_rank", -1))
        replacement = int(rotation["replacement_target_dp_rank"])
        failed_targets = {
            int(rank) for rank in rotation.get("failed_target_dp_ranks", ())
        }
        if target in failed_targets and target != replacement:
            return dict(payload), False, True
        if target != replacement:
            raise ValueError(
                "backup ACK target conflicts with the Router rotation fence"
            )

        with self._aegis_l1_l2_lock:
            current = self._aegis_recovery_hints.get(root_key)
            base = dict(current) if isinstance(current, dict) else {}
        pp_rank = int(event.get("pp_rank", 0))
        raw_stages = base.get("stage_payloads")
        if isinstance(raw_stages, list):
            for raw_stage in raw_stages:
                if (
                    isinstance(raw_stage, dict)
                    and int(raw_stage.get("pp_rank", -1)) == pp_rank
                ):
                    base = dict(raw_stage)
                    break

        merged = dict(base)
        hbm_keys = (
            "rid",
            "backup_key",
            "source_dp_rank",
            "target_dp_rank",
            "version",
            "route_version",
            "budget_version",
            "token_count",
            "l1_tokens",
            "hbm_pending_tokens",
            "reserved_l1_tokens",
            "gpu_candidate",
            "alpha",
            "normal_l1_pressure",
            "hbm_copy_path",
            "decision_d_l1_tokens",
            "decision_u_l1_tokens",
            "decision_o_l1_tokens",
            "d_l1_tokens",
            "u_l1_tokens",
            "o_l1_tokens",
            "window_id",
            "updated_time",
            "design_semantics",
            "source_owned",
            "cross_tier_transfer",
        )
        for name in hbm_keys:
            if name in payload:
                merged[name] = payload[name]
        frontiers = dict(base.get("tier_committed_source_frontiers") or {})
        incoming_frontiers = payload.get("tier_committed_source_frontiers")
        if isinstance(incoming_frontiers, dict):
            frontiers["hbm"] = int(incoming_frontiers.get("hbm", 0))
        else:
            frontiers["hbm"] = int(payload.get("l1_tokens", 0))
        merged["tier_committed_source_frontiers"] = frontiers
        merged["target_dp_rank"] = replacement
        merged["hbm_target_rotation"] = dict(rotation)
        merged["route_degraded_no_standby"] = False
        if is_group_commit:
            for name in (
                "materialization_scope",
                "group_commit_verified",
                "copy_completion_evidence",
                "execution_id",
                "block_generation",
            ):
                if name in payload:
                    merged[name] = payload[name]
            merged["source_committed_token_end"] = max(
                int(base.get("source_committed_token_end") or 0),
                int(payload.get("source_committed_token_end") or 0),
            )
        merged = self._refresh_aegis_recovery_hint(merged)
        return merged, True, False

    def apply_aegis_rank_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type"))
        if event_type == "aegis_source_chunk_committed":
            self._apply_aegis_source_chunk_committed(
                self._resolve_aegis_rotated_source_event(event)
            )
            return
        if event_type == "aegis_dp_budget_ack":
            dp_rank = int(event.get("dp_rank", -1))
            pp_rank = int(event.get("pp_rank", -1))
            budget_version = int(event.get("budget_version", -1))
            self._validate_rank(dp_rank)
            publisher = self._aegis_budget_publishers[dp_rank]
            fully_acknowledged = publisher.acknowledge(
                pp_rank=pp_rank,
                budget_version=budget_version,
            )
            rank_state = event.get("rank_state")
            if isinstance(rank_state, dict):
                with self._aegis_l1_l2_lock:
                    target_state = dict(
                        self._aegis_rank_states.get(dp_rank, {})
                    )
                    stages = dict(target_state.get("stages", {}))
                    stages[str(pp_rank)] = dict(rank_state)
                    target_state.update(
                        {
                            "dp_rank": dp_rank,
                            "pp_size": int(self.server_args.pp_size),
                            "budget_version": budget_version,
                            "stages": stages,
                        }
                    )
                    self._aegis_rank_states[dp_rank] = target_state
            self._append_aegis_event(
                "dp_budget_stage_ack",
                {
                    "dp_rank": dp_rank,
                    "pp_rank": pp_rank,
                    "budget_version": budget_version,
                    "acknowledged_pp_ranks": list(
                        publisher.acknowledged_pp_ranks
                    ),
                    "fully_acknowledged": fully_acknowledged,
                },
            )
            return
        if event_type == "gpu_recovery_l1_clear_ack":
            self._apply_gpu_recovery_l1_clear(event)
            return
        if event_type == "aegis_l1_l2_complete_request_ack":
            dp_rank = int(event.get("dp_rank", -1))
            pp_rank = int(event.get("pp_rank", -1))
            required_pp_size = int(event.get("required_pp_size", -1))
            backup_key = str(event.get("request_backup_key") or "")
            expected_version = int(event.get("expected_version", -1))
            result = event.get("result")
            self._validate_rank(dp_rank)
            if required_pp_size != int(self.server_args.pp_size):
                raise ValueError("CompleteRequest ACK PP size mismatch")
            if not 0 <= pp_rank < required_pp_size:
                raise ValueError("CompleteRequest ACK PP rank is outside topology")
            pending_key = (backup_key, expected_version, dp_rank)
            with self._aegis_l1_l2_lock:
                pending = self._aegis_stage_completion_pending.get(pending_key)
                if pending is None:
                    return
                pending["results"][pp_rank] = (
                    dict(result) if isinstance(result, dict) else {}
                )
                applied_stage_count = len(pending["results"])
                rank_state = event.get("rank_state")
                if isinstance(rank_state, dict):
                    target_state = dict(self._aegis_rank_states.get(dp_rank, {}))
                    stages = dict(target_state.get("stages", {}))
                    stages[str(pp_rank)] = dict(rank_state)
                    target_state["stages"] = stages
                    self._aegis_rank_states[dp_rank] = target_state
                if applied_stage_count == required_pp_size:
                    completed = self._aegis_stage_completion_pending.pop(
                        pending_key
                    )
                else:
                    completed = None
            self._append_aegis_event(
                "request_completion_stage_applied",
                {
                    "backup_key": backup_key,
                    "expected_version": expected_version,
                    "target_dp_rank": dp_rank,
                    "pp_rank": pp_rank,
                    "manager_completed": bool(
                        isinstance(result, dict)
                        and result.get("completed", False)
                    ),
                    "applied_stage_count": applied_stage_count,
                    "required_stage_count": required_pp_size,
                },
            )
            if completed is not None:
                self._finalize_aegis_request_completion(
                    backup_key=backup_key,
                    expected_version=expected_version,
                    target_dp_rank=dp_rank,
                    reason="all_target_stages_complete_request_terminal",
                )
            return
        if event_type == "aegis_l1_l2_invalidate_host_lease_ack":
            dp_rank = int(event.get("dp_rank", -1))
            pp_rank = int(event.get("pp_rank", -1))
            rank_state = event.get("rank_state")
            if isinstance(rank_state, dict) and dp_rank >= 0 and pp_rank >= 0:
                with self._aegis_l1_l2_lock:
                    target_state = dict(
                        self._aegis_rank_states.get(dp_rank, {})
                    )
                    stages = dict(target_state.get("stages", {}))
                    stages[str(pp_rank)] = dict(rank_state)
                    target_state["stages"] = stages
                    self._aegis_rank_states[dp_rank] = target_state
            self._append_aegis_event(
                "host_lease_reconciliation_ack",
                {
                    "dp_rank": dp_rank,
                    "pp_rank": pp_rank,
                    "backup_key": str(event.get("backup_key") or ""),
                    "expected_version": int(
                        event.get("expected_version", -1)
                    ),
                    "applied": isinstance(event.get("result"), dict),
                },
            )
            return
        if event_type == "aegis_l1_l2_supersede_ack":
            rank_state = event.get("rank_state")
            dp_rank = int(event.get("dp_rank", -1))
            result = event.get("result")
            if dp_rank < 0 or not isinstance(result, dict):
                return
            self._validate_rank(dp_rank)
            pp_rank = int(event.get("pp_rank", 0))
            required_pp_size = int(event.get("required_pp_size", 1))
            if required_pp_size != int(self.server_args.pp_size):
                raise ValueError(
                    "frontier supersede ACK required_pp_size mismatch: "
                    f"event={required_pp_size}, controller={self.server_args.pp_size}"
                )
            if pp_rank < 0 or pp_rank >= required_pp_size:
                raise ValueError(
                    f"invalid supersede ACK PP rank {pp_rank} for "
                    f"PP size {required_pp_size}"
                )

            request_backup_key = str(
                event.get("request_backup_key")
                or result.get("backup_key")
                or ""
            )
            stage_backup_key = str(
                event.get("stage_backup_key")
                or result.get("backup_key")
                or ""
            )
            expected_version = int(result.get("expected_version", -1))
            pending_key = (
                request_backup_key,
                expected_version,
                dp_rank,
            )
            with self._aegis_l1_l2_lock:
                target_state = dict(
                    self._aegis_rank_states.get(dp_rank, {})
                )
                stages = dict(target_state.get("stages", {}))
                if isinstance(rank_state, dict):
                    stages[str(pp_rank)] = dict(rank_state)
                target_state.update(
                    {
                        "dp_rank": dp_rank,
                        "pp_size": required_pp_size,
                        "stages": stages,
                    }
                )
                self._aegis_rank_states[dp_rank] = target_state

                pending = self._aegis_stage_supersede_pending.setdefault(
                    pending_key,
                    {
                        "results": {},
                        "stage_backup_keys": {},
                        "required_pp_size": required_pp_size,
                    },
                )
                existing_key = pending["stage_backup_keys"].get(pp_rank)
                if existing_key is not None and existing_key != stage_backup_key:
                    raise ValueError(
                        "conflicting supersede ACK for PP stage "
                        f"{pp_rank}: {existing_key!r} vs {stage_backup_key!r}"
                    )
                pending["results"][pp_rank] = dict(result)
                pending["stage_backup_keys"][pp_rank] = stage_backup_key
                applied_stage_count = len(pending["results"])

            self._append_aegis_event(
                "backup_frontier_supersede_stage_applied",
                {
                    "target_dp_rank": dp_rank,
                    "pp_rank": pp_rank,
                    "backup_key": request_backup_key,
                    "stage_backup_key": stage_backup_key,
                    "expected_version": expected_version,
                    "released": bool(result.get("released", False)),
                    "released_l1_tokens": int(
                        result.get("released_l1_tokens", 0)
                    ),
                    "released_l2_tokens": int(
                        result.get("released_l2_tokens", 0)
                    ),
                    "applied_stage_count": applied_stage_count,
                    "required_stage_count": required_pp_size,
                },
            )
            if applied_stage_count < required_pp_size:
                return

            with self._aegis_l1_l2_lock:
                completed = self._aegis_stage_supersede_pending.pop(
                    pending_key
                )
            ordered_results = tuple(
                completed["results"][rank]
                for rank in range(required_pp_size)
            )
            self._append_aegis_event(
                "backup_frontier_supersede_applied",
                {
                    "target_dp_rank": dp_rank,
                    "backup_key": request_backup_key,
                    "expected_version": expected_version,
                    "superseded_by": str(
                        ordered_results[0].get("superseded_by") or ""
                    ),
                    "released": any(
                        bool(item.get("released", False))
                        for item in ordered_results
                    ),
                    "released_l1_tokens": sum(
                        int(item.get("released_l1_tokens", 0))
                        for item in ordered_results
                    ),
                    "released_l2_tokens": sum(
                        int(item.get("released_l2_tokens", 0))
                        for item in ordered_results
                    ),
                    "stage_backup_keys": [
                        completed["stage_backup_keys"][rank]
                        for rank in range(required_pp_size)
                    ],
                    "applied_stage_count": required_pp_size,
                    "required_stage_count": required_pp_size,
                    "all_stages_applied": True,
                },
            )
            return
        is_group_commit = event_type == "aegis_l1_l2_group_commit_ack"
        if event_type not in {
            "aegis_l1_l2_backup_ack",
            "aegis_l1_l2_group_commit_ack",
        }:
            return
        if (
            int(self.server_args.pp_size) > 1
            and "required_pp_size" in event
            and "pp_rank" in event
        ):
            self._apply_pipeline_backup_ack(event)
            return
        payload = event.get("payload")
        updated_hints = event.get("updated_hints")
        rank_state = event.get("rank_state")
        if not isinstance(payload, dict):
            return
        if is_group_commit:
            source_end = int(payload.get("source_committed_token_end", -1))
            committed_frontier = int(
                payload.get("safe_frontier_tokens", -1)
            )
            if not (
                str(payload.get("materialization_scope"))
                == "committed_prefix"
                and bool(payload.get("group_commit_verified", False))
                and source_end >= 0
                and 0 <= committed_frontier <= source_end
            ):
                raise ValueError(
                    "target GroupCommitAck lacks a valid committed-prefix proof"
                )
        backup_key = str(payload.get("backup_key") or payload.get("rid") or "")
        if not backup_key:
            return
        incoming_version = max(int(payload.get("version", 1)), 1)
        with self._aegis_l1_l2_lock:
            tombstone = self._aegis_superseded_backups.get(backup_key)
            current_hint = self._aegis_recovery_hints.get(backup_key)
            current_version = (
                max(int(current_hint.get("version", 1)), 1)
                if isinstance(current_hint, dict)
                else -1
            )
        payload, rotation_ack, stale_rotated_target_ack = (
            self._merge_aegis_rotation_ack_payload(
                root_key=backup_key,
                event=event,
                payload=dict(payload),
                is_group_commit=is_group_commit,
            )
        )
        if stale_rotated_target_ack:
            self._append_aegis_event(
                "stale_rotated_target_ack_ignored",
                {
                    "backup_key": backup_key,
                    "version": incoming_version,
                    "target_dp_rank": int(payload.get("target_dp_rank", -1)),
                    "group_commit": is_group_commit,
                },
            )
            return
        if (
            tombstone is not None
            and incoming_version <= int(tombstone.get("version", -1))
        ):
            self._append_aegis_event(
                "superseded_backup_ack_ignored",
                {
                    "backup_key": backup_key,
                    "incoming_version": incoming_version,
                    "superseded_by": str(tombstone.get("superseded_by") or ""),
                    "target_dp_rank": int(payload.get("target_dp_rank", -1)),
                },
            )
            return
        if current_version > incoming_version:
            self._append_aegis_event(
                "stale_backup_ack_ignored",
                {
                    "backup_key": backup_key,
                    "incoming_version": incoming_version,
                    "current_version": current_version,
                    "target_dp_rank": int(payload.get("target_dp_rank", -1)),
                },
            )
            return
        if (
            not is_group_commit
            and not rotation_ack
            and isinstance(current_hint, dict)
            and current_version == incoming_version
            and str(current_hint.get("materialization_scope"))
            == "committed_prefix"
            and bool(current_hint.get("group_commit_verified", False))
        ):
            self._append_aegis_event(
                "reservation_ack_ignored_after_group_commit",
                {
                    "backup_key": backup_key,
                    "version": incoming_version,
                    "target_dp_rank": int(payload.get("target_dp_rank", -1)),
                },
            )
            return
        target_dp_rank = int(payload.get("target_dp_rank", -1))
        pp_rank = int(event.get("pp_rank", 0))
        budget_version = int(
            event.get(
                "budget_version",
                payload.get("budget_version", 1),
            )
        )
        if (
            int(self.server_args.pp_size) == 1
            and 0 <= target_dp_rank < len(self.workers)
        ):
            current_budget = self._aegis_budget_publishers[
                target_dp_rank
            ].current
            if (
                current_budget is not None
                and budget_version == current_budget.budget_version
            ):
                self._aegis_budget_publishers[target_dp_rank].acknowledge(
                    pp_rank=pp_rank,
                    budget_version=budget_version,
                )
        if is_group_commit:
            merged = dict(current_hint) if isinstance(current_hint, dict) else {}
            merged.update(dict(payload))
            payload = merged
        elif rotation_ack:
            payload = self._refresh_aegis_recovery_hint(dict(payload))
        else:
            payload = self._sync_aegis_host_lease(dict(payload))
            payload = self._sync_aegis_l3_lease(payload)
        with self._aegis_l1_l2_lock:
            self._aegis_recovery_hints[backup_key] = dict(payload)
            if isinstance(updated_hints, list):
                for hint in updated_hints:
                    if not isinstance(hint, dict):
                        continue
                    replaced_key = str(
                        hint.get("backup_key") or hint.get("rid") or ""
                    )
                    if replaced_key:
                        tombstone = self._aegis_superseded_backups.get(replaced_key)
                        if (
                            tombstone is not None
                            and max(int(hint.get("version", 1)), 1)
                            <= int(tombstone.get("version", -1))
                        ):
                            continue
                        existing = self._aegis_recovery_hints.get(replaced_key)
                        existing_version = (
                            int(existing.get("version", 1))
                            if isinstance(existing, dict)
                            else -1
                        )
                        hint_version = max(int(hint.get("version", 1)), 1)
                        if existing_version > hint_version:
                            continue
                        self._aegis_recovery_hints[replaced_key] = dict(hint)
            if isinstance(rank_state, dict):
                self._aegis_rank_states[int(payload.get("target_dp_rank", -1))] = dict(
                    rank_state
                )
        with self._aegis_l1_l2_lock:
            self._aegis_recovery_hints[backup_key] = dict(payload)
        compact_updates = [self._compact_aegis_recovery_hint(payload)]
        if isinstance(updated_hints, list):
            compact_updates.extend(
                self._compact_aegis_recovery_hint(hint)
                for hint in updated_hints
                if isinstance(hint, dict)
            )
        self._append_aegis_event(
            "backup_ack",
            {
                "rid": str(payload.get("rid") or ""),
                "backup_key": backup_key,
                "execution_dp_rank": int(payload.get("source_dp_rank", -1)),
                "backup_dp_rank": int(payload.get("target_dp_rank", -1)),
                "action": str(payload.get("recovery_action") or "recompute"),
                "l1_tokens": int(payload.get("l1_tokens", 0)),
                "l2_tokens": int(payload.get("l2_tokens", 0)),
                "l3_tokens": int(payload.get("l3_tokens", 0)),
                "recompute_tokens": int(payload.get("recompute_tokens", 0)),
                "alpha": float(payload.get("alpha", 0.0)),
                "beta": float(payload.get("beta", 0.0)),
                "window_id": int(payload.get("window_id", 0)),
                "window_update": payload.get("window_update"),
                "failure_scope": str(payload.get("failure_scope") or "node"),
                "workflow_id": str(payload.get("workflow_id") or ""),
                "agent_id": str(payload.get("agent_id") or ""),
                "worker_seq": int(payload.get("worker_seq", 0)),
                "version": max(int(payload.get("version", 1)), 1),
                "safe_frontier_tokens": (
                    int(
                        payload.get(
                            "safe_frontier_tokens",
                            int(payload.get("l1_tokens", 0))
                            + int(payload.get("l2_tokens", 0))
                            + int(payload.get("l3_tokens", 0)),
                        )
                    )
                ),
                "updated_time": float(payload.get("updated_time", 0.0)),
                "hint_updates": compact_updates,
                "materialization_scope": str(
                    payload.get("materialization_scope")
                    or "metadata_reservation_only"
                ),
                "source_committed_token_end": payload.get(
                    "source_committed_token_end"
                ),
                "group_commit_verified": bool(
                    payload.get("group_commit_verified", False)
                ),
                "copy_completion_evidence": str(
                    payload.get("copy_completion_evidence") or ""
                ),
                "execution_id": str(payload.get("execution_id") or ""),
                "block_generation": int(
                    payload.get("block_generation", -1)
                ),
                "critical_host_floor_tokens": int(
                    payload.get("critical_host_floor_tokens", 0)
                ),
                "critical_host_floor_fraction": float(
                    payload.get("critical_host_floor_fraction", 0.0)
                ),
                "gpu_candidate": payload.get("gpu_candidate"),
                "host_candidate": payload.get("host_candidate"),
                "l3_candidate": payload.get("l3_candidate"),
                "host_pool_lease": payload.get("host_pool_lease"),
                "persistent_pool_lease": payload.get(
                    "persistent_pool_lease"
                ),
                "host_budget_owner": payload.get("host_budget_owner"),
                "host_budget_version": payload.get("host_budget_version"),
                "dp_host_policy_limit_tokens": payload.get(
                    "dp_host_policy_limit_tokens"
                ),
                "dp_host_resident_tokens": payload.get(
                    "dp_host_resident_tokens"
                ),
                "node_host_physical_capacity_tokens": payload.get(
                    "node_host_physical_capacity_tokens"
                ),
                "node_host_resident_tokens": payload.get(
                    "node_host_resident_tokens"
                ),
                "host_attempted_tokens": payload.get(
                    "host_attempted_tokens"
                ),
                "host_admitted_tokens": payload.get(
                    "host_admitted_tokens"
                ),
                "host_unmet_tokens": payload.get("host_unmet_tokens"),
            },
        )

    def _stage_commit_ack_from_payload(
        self,
        *,
        event: dict[str, Any],
        payload: dict[str, Any],
    ) -> StageCommitAck:
        pp_rank = int(event.get("pp_rank", -1))
        source_pp_rank = int(event.get("source_pp_rank", -1))
        target_pp_rank = int(event.get("target_pp_rank", -1))
        if not (pp_rank == source_pp_rank == target_pp_rank):
            raise ValueError(
                "stage ACK must preserve a same-stage Connector path: "
                f"local={pp_rank}, source={source_pp_rank}, target={target_pp_rank}"
            )
        source_dp_rank = int(payload.get("source_dp_rank", -1))
        target_dp_rank = int(payload.get("target_dp_rank", -1))
        independent_tiers = (
            str(payload.get("design_semantics") or "") == DESIGN_SEMANTICS
        )
        total_tokens = max(int(payload.get("token_count", 0)), 0)
        if total_tokens <= 0:
            protected_tokens = (
                int(payload.get("safe_frontier_tokens", 0))
                if independent_tiers
                else int(payload.get("l1_tokens", 0))
                + int(payload.get("l2_tokens", 0))
            )
            total_tokens = max(
                protected_tokens + int(payload.get("recompute_tokens", 0)),
                0,
            )
        committed_copy = (
            not independent_tiers
            or (
                str(payload.get("materialization_scope") or "")
                == "committed_prefix"
                and bool(payload.get("group_commit_verified", False))
            )
        )
        available_hbm_tokens = (
            min(max(int(payload.get("l1_tokens", 0)), 0), total_tokens)
            if committed_copy
            else 0
        )
        available_host_tokens = (
            min(max(int(payload.get("l2_tokens", 0)), 0), total_tokens)
            if committed_copy
            else 0
        )
        hbm_tokens = available_hbm_tokens
        host_tokens = min(
            available_host_tokens,
            max(total_tokens - hbm_tokens, 0),
        )
        # Persistent storage is not part of the finite HBM/Host/Recompute
        # stage contract yet.  Treat it conservatively as recomputation here;
        # legacy L3 metadata remains available in ``stage_payloads``.
        recompute_tokens = max(total_tokens - hbm_tokens - host_tokens, 0)
        committed_frontier = min(
            max(int(payload.get("safe_frontier_tokens", 0)), 0),
            hbm_tokens + host_tokens,
            total_tokens,
        )

        gpu_candidate = payload.get("gpu_candidate")
        host_candidate = payload.get("host_candidate")
        gpu_candidate = gpu_candidate if isinstance(gpu_candidate, dict) else {}
        host_candidate = host_candidate if isinstance(host_candidate, dict) else {}
        recovery_path: list[str] = []
        copy_latency_ms = 0.0
        restore_latency_ms = 0.0
        if independent_tiers:
            gpu_backup_path = tuple(gpu_candidate.get("backup_path", ())) or (
                "gpu_rdma",
            )
            gpu_recovery_path = tuple(
                gpu_candidate.get("recovery_path", ())
            ) or ("direct_hbm_reuse",)
            host_backup_path = tuple(
                host_candidate.get("backup_path", ())
            ) or ("d2h",)
            host_recovery_path = tuple(
                host_candidate.get("recovery_path", ())
            ) or ("h2d",)
            copy_candidates = []
            if available_hbm_tokens:
                copy_candidates.append(
                    self._aegis_path_costs.path_ms(
                        gpu_backup_path, available_hbm_tokens
                    )
                )
            if available_host_tokens:
                copy_candidates.append(
                    self._aegis_path_costs.path_ms(
                        host_backup_path, available_host_tokens
                    )
                )
            # Source D2D and source D2H are independent queues.  The modeled
            # copy critical path is their maximum, never an HBM->DRAM sum.
            copy_latency_ms = max(copy_candidates, default=0.0)
            candidates: list[
                tuple[float, int, str, float, tuple[str, ...]]
            ] = [
                (
                    (
                        self._aegis_path_costs.recompute_ms(total_tokens)
                        if total_tokens
                        else 0.0
                    ),
                    0,
                    "recompute",
                    0,
                    ("recompute",),
                )
            ]
            if available_hbm_tokens:
                hbm_restore = self._aegis_path_costs.path_ms(
                    gpu_recovery_path, available_hbm_tokens
                )
                candidates.append(
                    (
                        hbm_restore
                        + (
                            self._aegis_path_costs.recompute_ms(
                                total_tokens - available_hbm_tokens
                            )
                            if total_tokens > available_hbm_tokens
                            else 0.0
                        ),
                        available_hbm_tokens,
                        "hbm",
                        hbm_restore,
                        gpu_recovery_path,
                    )
                )
            if available_host_tokens:
                host_restore = self._aegis_path_costs.path_ms(
                    host_recovery_path, available_host_tokens
                )
                candidates.append(
                    (
                        host_restore
                        + (
                            self._aegis_path_costs.recompute_ms(
                                total_tokens - available_host_tokens
                            )
                            if total_tokens > available_host_tokens
                            else 0.0
                        ),
                        available_host_tokens,
                        "host",
                        host_restore,
                        host_recovery_path,
                    )
                )
            (
                _selected_total_ms,
                selected_prefix,
                selected_tier,
                restore_latency_ms,
                selected_path,
            ) = min(
                candidates,
                key=lambda item: (item[0], -item[1], item[2]),
            )
            hbm_tokens = selected_prefix if selected_tier == "hbm" else 0
            host_tokens = selected_prefix if selected_tier == "host" else 0
            recompute_tokens = max(total_tokens - selected_prefix, 0)
            committed_frontier = selected_prefix
            recovery_path = list(selected_path)
        else:
            recompute_tokens = max(total_tokens - hbm_tokens - host_tokens, 0)
            committed_frontier = min(
                max(int(payload.get("safe_frontier_tokens", 0)), 0),
                hbm_tokens + host_tokens,
                total_tokens,
            )
        if not independent_tiers and hbm_tokens:
            gpu_backup_path = tuple(gpu_candidate.get("backup_path", ()))
            gpu_recovery_path = tuple(gpu_candidate.get("recovery_path", ()))
            recovery_path.extend(str(item) for item in gpu_recovery_path)
            copy_latency_ms += self._aegis_path_costs.path_ms(
                gpu_backup_path or ("gpu_rdma",), hbm_tokens
            )
            restore_latency_ms += self._aegis_path_costs.path_ms(
                gpu_recovery_path or ("direct_hbm_reuse",), hbm_tokens
            )
        if not independent_tiers and host_tokens:
            host_backup_path = tuple(host_candidate.get("backup_path", ()))
            host_recovery_path = tuple(host_candidate.get("recovery_path", ()))
            recovery_path.extend(str(item) for item in host_recovery_path)
            copy_latency_ms += self._aegis_path_costs.path_ms(
                host_backup_path or ("d2h", "h2h"), host_tokens
            )
            restore_latency_ms += self._aegis_path_costs.path_ms(
                host_recovery_path or ("h2d",), host_tokens
            )
        if recompute_tokens and "recompute" not in recovery_path:
            recovery_path.append("recompute")
        if not recovery_path:
            recovery_path.append("recompute")
        target_node = self._aegis_topology.node_for_rank(target_dp_rank)
        bytes_per_token = max(
            int(payload.get("kv_bytes_per_token_per_stage", 0)), 0
        )
        return StageCommitAck(
            request_id=str(payload.get("rid") or ""),
            source_stage=StageId(source_dp_rank, pp_rank, 0),
            backup_stage=StageId(target_dp_rank, pp_rank, 0),
            route_version=max(int(event.get("route_version", 1)), 1),
            budget_version=max(int(event.get("budget_version", 1)), 1),
            stage_total_tokens=total_tokens,
            committed_frontier_tokens=committed_frontier,
            hbm_tokens=hbm_tokens,
            host_tokens=host_tokens,
            recompute_tokens=recompute_tokens,
            commit_logical_time=max(
                float(payload.get("updated_time", self._now_fn())), 0.0
            ),
            storage_failure_domain=f"node:{target_node}",
            recovery_path=tuple(recovery_path),
            copy_latency_s=max(copy_latency_ms, 0.0) / 1000.0,
            expected_restore_latency_s=max(restore_latency_ms, 0.0) / 1000.0,
            expected_recompute_latency_s=(
                (
                    self._aegis_path_costs.recompute_ms(recompute_tokens)
                    if recompute_tokens
                    else 0.0
                )
                / 1000.0
            ),
            copy_bytes=(
                available_hbm_tokens + available_host_tokens
                if independent_tiers
                else hbm_tokens + host_tokens
            )
            * bytes_per_token,
            link_id=(
                f"dp{source_dp_rank}.pp{pp_rank}->"
                f"dp{target_dp_rank}.pp{pp_rank}"
            ),
            ack_sequence=max(int(payload.get("sequence", 0)), 0),
        )

    def _apply_pipeline_backup_ack(self, event: dict[str, Any]) -> None:
        is_group_commit = (
            str(event.get("type")) == "aegis_l1_l2_group_commit_ack"
        )
        raw_payload = event.get("payload")
        if not isinstance(raw_payload, dict):
            return
        payload = dict(raw_payload)
        if is_group_commit:
            source_end = int(payload.get("source_committed_token_end", -1))
            frontier = int(payload.get("safe_frontier_tokens", -1))
            if not (
                str(payload.get("materialization_scope"))
                == "committed_prefix"
                and bool(payload.get("group_commit_verified", False))
                and source_end >= 0
                and 0 <= frontier <= source_end
                and str(payload.get("execution_id") or "")
            ):
                raise ValueError(
                    "pipeline GroupCommitAck lacks a valid committed-prefix proof"
                )
        root_key = str(
            event.get("request_backup_key")
            or payload.get("backup_key")
            or payload.get("rid")
            or ""
        )
        if not root_key:
            return
        incoming_version = max(int(payload.get("version", 1)), 1)
        with self._aegis_l1_l2_lock:
            tombstone = self._aegis_superseded_backups.get(root_key)
            current_hint = self._aegis_recovery_hints.get(root_key)
            current_version = (
                max(int(current_hint.get("version", 1)), 1)
                if isinstance(current_hint, dict)
                else -1
            )
        payload, rotation_ack, stale_rotated_target_ack = (
            self._merge_aegis_rotation_ack_payload(
                root_key=root_key,
                event=event,
                payload=payload,
                is_group_commit=is_group_commit,
            )
        )
        if stale_rotated_target_ack:
            self._append_aegis_event(
                "stale_rotated_pipeline_ack_ignored",
                {
                    "backup_key": root_key,
                    "version": incoming_version,
                    "target_dp_rank": int(payload.get("target_dp_rank", -1)),
                    "group_commit": is_group_commit,
                },
            )
            return
        if (
            tombstone is not None
            and incoming_version <= int(tombstone.get("version", -1))
        ):
            self._append_aegis_event(
                "superseded_pipeline_ack_ignored",
                {
                    "backup_key": root_key,
                    "incoming_version": incoming_version,
                    "superseded_by": str(
                        tombstone.get("superseded_by") or ""
                    ),
                },
            )
            return
        if current_version > incoming_version:
            self._append_aegis_event(
                "stale_pipeline_ack_ignored",
                {
                    "backup_key": root_key,
                    "incoming_version": incoming_version,
                    "current_version": current_version,
                },
            )
            return
        if (
            not is_group_commit
            and not rotation_ack
            and isinstance(current_hint, dict)
            and current_version == incoming_version
            and str(current_hint.get("materialization_scope"))
            == "committed_prefix"
            and bool(current_hint.get("group_commit_verified", False))
        ):
            self._append_aegis_event(
                "pipeline_reservation_ack_ignored_after_group_commit",
                {
                    "backup_key": root_key,
                    "version": incoming_version,
                },
            )
            return
        required_pp_size = int(event.get("required_pp_size", -1))
        if required_pp_size != int(self.server_args.pp_size):
            raise ValueError(
                "stage ACK required_pp_size does not match runtime topology: "
                f"ack={required_pp_size}, runtime={self.server_args.pp_size}"
            )
        pp_rank = int(event.get("pp_rank", -1))
        if not 0 <= pp_rank < required_pp_size:
            raise ValueError(f"stage ACK pp_rank={pp_rank} is outside topology")

        payload["request_backup_key"] = root_key
        payload["stage_backup_key"] = str(
            event.get("stage_backup_key") or payload.get("backup_key") or ""
        )
        payload["pp_rank"] = pp_rank
        payload["route_version"] = max(int(event.get("route_version", 1)), 1)
        payload["budget_version"] = max(int(event.get("budget_version", 1)), 1)
        if str(payload.get("design_semantics") or "") == DESIGN_SEMANTICS:
            source_end = payload.get("source_committed_token_end")
            payload["token_count"] = max(
                int(
                    source_end
                    if source_end is not None
                    else payload.get(
                        "token_count",
                        int(payload.get("safe_frontier_tokens", 0))
                        + int(payload.get("recompute_tokens", 0)),
                    )
                ),
                0,
            )
        else:
            payload["token_count"] = max(
                int(payload.get("l1_tokens", 0))
                + int(payload.get("l2_tokens", 0))
                + int(payload.get("recompute_tokens", 0)),
                0,
            )
        if is_group_commit:
            with self._aegis_l1_l2_lock:
                reservation = self._aegis_recovery_hints.get(root_key)
                reservation = (
                    dict(reservation)
                    if isinstance(reservation, dict)
                    else {}
                )
            reserved_stage_payload: dict[str, Any] = {}
            raw_stage_payloads = reservation.get("stage_payloads")
            if isinstance(raw_stage_payloads, list):
                for raw_stage in raw_stage_payloads:
                    if (
                        isinstance(raw_stage, dict)
                        and int(raw_stage.get("pp_rank", -1)) == pp_rank
                    ):
                        reserved_stage_payload = dict(raw_stage)
                        break
            reserved_stage_payload.update(payload)
            payload = reserved_stage_payload
        elif rotation_ack:
            payload = self._refresh_aegis_recovery_hint(payload)
        else:
            payload = self._sync_aegis_host_lease(payload)
            payload = self._sync_aegis_l3_lease(payload)
        stage_ack = self._stage_commit_ack_from_payload(
            event=event,
            payload=payload,
        )
        route_version = stage_ack.route_version
        budget_version = stage_ack.budget_version
        budget_publisher = self._aegis_budget_publishers[
            stage_ack.backup_stage.dp_rank
        ]
        budget_publisher.acknowledge(
            pp_rank=pp_rank,
            budget_version=budget_version,
        )
        version = incoming_version
        execution_id = str(payload.get("execution_id") or "")
        pending_key = (
            root_key,
            version,
            route_version,
            budget_version,
        )
        group_pending_key = (
            root_key,
            version,
            route_version,
            budget_version,
            execution_id,
        )
        rank_state = event.get("rank_state")
        target_dp_rank = stage_ack.backup_stage.dp_rank

        with self._aegis_l1_l2_lock:
            target_state = dict(self._aegis_rank_states.get(target_dp_rank, {}))
            stages = dict(target_state.get("stages", {}))
            if isinstance(rank_state, dict):
                stages[str(pp_rank)] = dict(rank_state)
            target_state.update(
                {
                    "dp_rank": target_dp_rank,
                    "pp_size": required_pp_size,
                    "stages": stages,
                    "normal_l1_pressure": max(
                        (
                            float(stage.get("normal_l1_pressure", 0.0))
                            for stage in stages.values()
                            if isinstance(stage, dict)
                        ),
                        default=0.0,
                    ),
                    "normal_l2_pressure": max(
                        (
                            float(stage.get("normal_l2_pressure", 0.0))
                            for stage in stages.values()
                            if isinstance(stage, dict)
                        ),
                        default=0.0,
                    ),
                }
            )
            self._aegis_rank_states[target_dp_rank] = target_state
            pending_store = (
                self._aegis_stage_group_commit_latest
                if is_group_commit
                else self._aegis_stage_ack_pending
            )
            selected_pending_key = (
                group_pending_key if is_group_commit else pending_key
            )
            pending = pending_store.setdefault(
                selected_pending_key,
                {
                    "acks": {},
                    "payloads": {},
                    "required_pp_size": required_pp_size,
                },
            )
            pending["acks"][pp_rank] = stage_ack
            pending["payloads"][pp_rank] = dict(payload)
            stage_count = len(pending["acks"])

        self._append_aegis_event(
            "backup_stage_ack",
            {
                "rid": stage_ack.request_id,
                "backup_key": root_key,
                "stage_backup_key": payload["stage_backup_key"],
                "source_dp_rank": stage_ack.source_stage.dp_rank,
                "target_dp_rank": target_dp_rank,
                "pp_rank": pp_rank,
                "route_version": route_version,
                "budget_version": budget_version,
                "committed_frontier_tokens": (
                    stage_ack.committed_frontier_tokens
                ),
                "hbm_tokens": stage_ack.hbm_tokens,
                "host_tokens": stage_ack.host_tokens,
                "recompute_tokens": stage_ack.recompute_tokens,
                "committed_stage_count": stage_count,
                "required_stage_count": required_pp_size,
            },
        )
        if stage_count < required_pp_size:
            return

        with self._aegis_l1_l2_lock:
            if is_group_commit:
                current = self._aegis_stage_group_commit_latest[
                    group_pending_key
                ]
                completed = {
                    "acks": dict(current["acks"]),
                    "payloads": dict(current["payloads"]),
                    "required_pp_size": current["required_pp_size"],
                }
            else:
                completed = self._aegis_stage_ack_pending.pop(pending_key)
        ordered_acks = tuple(
            completed["acks"][rank] for rank in range(required_pp_size)
        )
        ordered_payloads = tuple(
            completed["payloads"][rank] for rank in range(required_pp_size)
        )
        requested_tokens = min(
            ack.stage_total_tokens for ack in ordered_acks
        )
        recovery_hint = RecoveryHint.from_stage_acks(
            request_id=stage_ack.request_id,
            route_version=route_version,
            budget_version=budget_version,
            pp_size=required_pp_size,
            requested_tokens=requested_tokens,
            stage_acks=ordered_acks,
        )
        first = dict(ordered_payloads[0])
        independent_tiers = (
            str(first.get("design_semantics") or "") == DESIGN_SEMANTICS
        )
        if independent_tiers:
            stage_hbm_frontiers = [
                min(max(int(item.get("l1_tokens", 0)), 0), requested_tokens)
                for item in ordered_payloads
            ]
            stage_dram_frontiers = [
                min(max(int(item.get("l2_tokens", 0)), 0), requested_tokens)
                for item in ordered_payloads
            ]
            stage_safe_frontiers = [
                max(hbm, dram)
                for hbm, dram in zip(
                    stage_hbm_frontiers, stage_dram_frontiers
                )
            ]
            safe_frontier = min(stage_safe_frontiers, default=0)
            compatibility_hbm = min(stage_hbm_frontiers, default=0)
            compatibility_dram = min(stage_dram_frontiers, default=0)
        else:
            safe_frontier = recovery_hint.safe_frontier_tokens
            compatibility_hbm = min(ack.hbm_tokens for ack in ordered_acks)
            compatibility_dram = max(safe_frontier - compatibility_hbm, 0)
        aggregate = {
            **first,
            "backup_key": root_key,
            "stage_backup_key": None,
            "ack": True,
            "all_stages_committed": True,
            "committed_stage_count": required_pp_size,
            "required_stage_count": required_pp_size,
            "route_version": route_version,
            "budget_version": budget_version,
            "safe_frontier_tokens": safe_frontier,
            "l1_tokens": min(compatibility_hbm, safe_frontier),
            "l2_tokens": (
                min(compatibility_dram, safe_frontier)
                if independent_tiers
                else compatibility_dram
            ),
            "l3_tokens": 0,
            "recompute_tokens": max(requested_tokens - safe_frontier, 0),
            "recovery_action": (
                "stage_aligned_independent_tiers"
                if independent_tiers
                else "stage_aligned_backup"
            ),
            "expected_recovery_ms": 1000.0
            * (
                recovery_hint.total_restore_latency_s
                + recovery_hint.total_recompute_latency_s
            ),
            "protected_recovery_service_s": (
                recovery_hint.total_restore_latency_s
                + recovery_hint.total_recompute_latency_s
            ),
            "unprotected_recovery_service_s": (
                self._aegis_path_costs.recompute_ms(requested_tokens) / 1000.0
            ),
            "recovery_service_cost_source": (
                self._aegis_recovery_cost_source
            ),
            "recovery_service_cost_calibration_id": (
                self._aegis_recovery_cost_calibration_id
            ),
            "model_layout_id": str(first.get("model_layout_id") or ""),
            "layout_compatible": first.get("layout_compatible"),
            "recovery_hint": recovery_hint.to_dict(),
            "stage_acks": [ack.to_dict() for ack in ordered_acks],
            "stage_payloads": [dict(item) for item in ordered_payloads],
            "stage_safe_frontiers": (
                stage_safe_frontiers if independent_tiers else None
            ),
            "stage_hbm_frontiers": (
                stage_hbm_frontiers if independent_tiers else None
            ),
            "stage_dram_frontiers": (
                stage_dram_frontiers if independent_tiers else None
            ),
            "independent_tier_results": independent_tiers,
            "updated_time": max(
                float(item.get("updated_time", 0.0))
                for item in ordered_payloads
            ),
        }
        if is_group_commit:
            aggregate.update(
                {
                    "materialization_scope": "committed_prefix",
                    "source_committed_token_end": min(
                        int(item.get("source_committed_token_end", 0))
                        for item in ordered_payloads
                    ),
                    "group_commit_verified": all(
                        bool(item.get("group_commit_verified", False))
                        for item in ordered_payloads
                    ),
                    "copy_completion_evidence": (
                        "sglsim_modeled_group_commit"
                    ),
                    "execution_id": execution_id,
                    "block_generation": min(
                        int(item.get("block_generation", -1))
                        for item in ordered_payloads
                    ),
                }
            )
        elif rotation_ack:
            aggregate.update(
                {
                    "materialization_scope": str(
                        first.get("materialization_scope")
                        or "metadata_reservation_only"
                    ),
                    "source_committed_token_end": first.get(
                        "source_committed_token_end"
                    ),
                    "group_commit_verified": bool(
                        first.get("group_commit_verified", False)
                    ),
                    "copy_completion_evidence": str(
                        first.get("copy_completion_evidence") or ""
                    ),
                    "execution_id": str(first.get("execution_id") or ""),
                    "block_generation": int(
                        first.get("block_generation", -1)
                    ),
                    "hbm_target_rotation": dict(
                        first.get("hbm_target_rotation") or {}
                    ),
                }
            )
            aggregate = self._refresh_aegis_recovery_hint(aggregate)
        else:
            aggregate.update(
                {
                    "materialization_scope": "metadata_reservation_only",
                    "source_committed_token_end": None,
                    "group_commit_verified": False,
                }
            )
        with self._aegis_l1_l2_lock:
            self._aegis_recovery_hints[root_key] = aggregate
        self._append_aegis_event(
            "backup_ack",
            {
                "rid": stage_ack.request_id,
                "backup_key": root_key,
                "execution_dp_rank": stage_ack.source_stage.dp_rank,
                "backup_dp_rank": target_dp_rank,
                "action": str(aggregate["recovery_action"]),
                "safe_frontier_tokens": safe_frontier,
                "bottleneck_pp_rank": recovery_hint.bottleneck_pp_rank,
                "restored_tokens_by_stage": list(
                    recovery_hint.restored_tokens_by_stage
                ),
                "recomputed_tokens_by_stage": list(
                    recovery_hint.recomputed_tokens_by_stage
                ),
                "route_version": route_version,
                "budget_version": budget_version,
                "committed_stage_count": required_pp_size,
                "required_stage_count": required_pp_size,
                "version": max(int(aggregate.get("version", 1)), 1),
                "updated_time": float(aggregate.get("updated_time", 0.0)),
                "hint_updates": [
                    self._compact_aegis_recovery_hint(aggregate)
                ],
                "materialization_scope": str(
                    aggregate.get("materialization_scope")
                    or "metadata_reservation_only"
                ),
                "source_committed_token_end": aggregate.get(
                    "source_committed_token_end"
                ),
                "group_commit_verified": bool(
                    aggregate.get("group_commit_verified", False)
                ),
                "copy_completion_evidence": str(
                    aggregate.get("copy_completion_evidence") or ""
                ),
                "execution_id": str(
                    aggregate.get("execution_id") or ""
                ),
                "block_generation": int(
                    aggregate.get("block_generation", -1)
                ),
            },
        )

    def _observe_node_host_budget(
        self,
        payload: dict[str, Any],
        *,
        normal_l2_pressure: float,
    ) -> dict[str, Any]:
        """Attach DP-policy and node-capacity state without rewriting beta.

        Algorithm 3 publishes beta through the DPBudgetPublisher.  The node
        layer is a physical allocator only; it must never infer or mutate a
        second node-owned beta from admission traffic.
        """

        _ = normal_l2_pressure
        host_candidate = payload.get("host_candidate")
        if not isinstance(host_candidate, dict):
            return payload
        node_id = int(host_candidate.get("host_pool_node_id", -1))
        pool = self._aegis_host_pools.get(node_id)
        arbiter = self._aegis_node_capacity_arbiters.get(node_id)
        if pool is None or arbiter is None:
            return payload
        target_dp_rank = int(payload.get("target_dp_rank", -1))
        if target_dp_rank not in arbiter.allowed_dp_ranks:
            if len(arbiter.allowed_dp_ranks) != 1:
                return payload
            target_dp_rank = arbiter.allowed_dp_ranks[0]
        budget = self._aegis_budget_publishers[target_dp_rank].current
        if budget is None:
            raise RuntimeError(
                f"DP rank {target_dp_rank} has no published Host budget"
            )
        requested_logical = max(int(payload.get("host_attempted_tokens", 0)), 0)
        admitted_logical = max(int(payload.get("l2_tokens", 0)), 0)
        requested = pool.allocation_tokens(requested_logical)
        host_lease = payload.get("host_pool_lease")
        admitted = (
            max(int(host_lease.get("allocated_tokens", 0)), 0)
            if admitted_logical > 0 and isinstance(host_lease, dict)
            else 0
        )
        unmet_logical = max(requested_logical - admitted_logical, 0)
        unmet_allocated = max(requested - admitted, 0)
        arbiter_state = arbiter.snapshot()
        payload.update(
            {
                "host_budget_owner": "dp_controller",
                "host_budget_version": budget.budget_version,
                "beta": round(budget.beta, 6),
                "beta_scope": "dp_replica_policy",
                "dp_host_policy_limit_tokens": (
                    budget.host_policy_limit_tokens
                ),
                "dp_host_resident_tokens": int(
                    arbiter_state["resident_tokens_by_dp"].get(
                        target_dp_rank, 0
                    )
                ),
                "host_unique_offered_tokens": requested_logical,
                "host_attempted_tokens": requested_logical,
                "host_admitted_tokens": admitted_logical,
                "host_unmet_tokens": unmet_logical,
                "host_unmet_allocated_tokens": unmet_allocated,
                "node_host_capacity_owner": "node_physical_arbiter",
                "node_host_physical_capacity_tokens": int(
                    arbiter_state["physical_capacity_tokens"]
                ),
                "node_host_resident_tokens": int(
                    arbiter_state["resident_tokens"]
                ),
                "node_host_budget_state": arbiter_state,
            }
        )
        return payload

    def _sync_aegis_host_lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Commit one scheduler L2 decision through both ownership layers.

        The DP snapshot owns beta and provides the per-DP policy limit.  The
        node arbiter enforces that limit plus physical capacity.  The legacy
        Host pool remains only as the aligned/pinned physical lease ledger.
        """

        backup_key = str(payload.get("backup_key") or payload.get("rid") or "")
        host_candidate = payload.get("host_candidate")
        if not backup_key or not isinstance(host_candidate, dict):
            return payload
        node_id = int(host_candidate.get("host_pool_node_id", -1))
        pool = self._aegis_host_pools.get(node_id)
        arbiter = self._aegis_node_capacity_arbiters.get(node_id)
        if pool is None or arbiter is None:
            return payload
        target_rank = int(
            payload.get(
                "target_dp_rank",
                host_candidate.get("target_dp_rank", -1),
            )
        )
        if target_rank not in arbiter.allowed_dp_ranks:
            if len(arbiter.allowed_dp_ranks) != 1:
                raise ValueError(
                    "Host ACK does not identify a DP policy owner on node "
                    f"{node_id}: target_dp_rank={target_rank}"
                )
            target_rank = arbiter.allowed_dp_ranks[0]
            payload["target_dp_rank"] = target_rank
        budget = self._aegis_budget_publishers[target_rank].current
        if budget is None:
            raise RuntimeError(f"DP rank {target_rank} has no Host budget")

        stage_rank = self._pp_rank_from_stage_backup_key(backup_key)
        if stage_rank is None:
            stage_rank = int(payload.get("pp_rank", 0))
        if not 0 <= stage_rank < budget.pp_size:
            raise ValueError(
                f"Host ACK PP rank {stage_rank} is outside DP budget topology"
            )
        payload_version = max(int(payload.get("version", 1)), 1)
        pool.release(backup_key, max_version=payload_version)
        arbiter.release(backup_key)
        l2_tokens = max(int(payload.get("l2_tokens", 0)), 0)
        attempted_tokens = max(
            int(
                payload.get(
                    "host_attempted_tokens",
                    payload.get("decision_d_l2_tokens", l2_tokens),
                )
            ),
            0,
        )
        payload["host_unique_offered_tokens"] = attempted_tokens
        payload["host_attempted_tokens"] = attempted_tokens
        if l2_tokens <= 0:
            payload["host_pool_lease"] = None
            payload["host_capacity_lease"] = None
            payload["host_pool_state"] = pool.snapshot()
            return self._observe_node_host_budget(
                payload,
                normal_l2_pressure=float(
                    payload.get("normal_l2_pressure", 0.0)
                ),
            )
        utility = float(host_candidate.get("utility", 0.0) or 0.0)
        allocated_tokens = pool.allocation_tokens(l2_tokens)
        capacity_admission = arbiter.admit(
            HostAdmissionRequest(
                lease_key=backup_key,
                request_id=str(
                    payload.get("rid")
                    or payload.get("request_backup_key")
                    or backup_key
                ),
                workflow_id=str(
                    payload.get("workflow_id")
                    or payload.get("rid")
                    or backup_key
                ),
                chunk_id=max(int(payload.get("chunk_id", 0)), 0),
                dp_rank=target_rank,
                pp_rank=stage_rank,
                budget_version=max(
                    int(payload.get("budget_version", budget.budget_version)),
                    1,
                ),
                token_count=allocated_tokens,
                utility=max(utility, 0.0),
            )
        )
        payload["host_capacity_admission_reason"] = capacity_admission.reason
        payload["host_capacity_unmet_allocated_tokens"] = (
            capacity_admission.unmet_tokens
        )
        payload["host_unmet_utility"] = capacity_admission.unmet_utility
        admission = None
        if capacity_admission.admitted:
            admission = pool.admit(
                backup_key=backup_key,
                token_count=l2_tokens,
                utility_density=utility / max(allocated_tokens, 1),
                source_node_id=int(
                    host_candidate.get("source_node_id", -1)
                ),
                failure_scope=str(payload.get("failure_scope") or "node"),
                cache_resident_tokens=max(
                    int(payload.get("host_cache_resident_tokens", 0)), 0
                ),
                version=payload_version,
            )
        if (
            capacity_admission.admitted
            and capacity_admission.lease is not None
            and admission is not None
            and admission.admitted
            and admission.lease is not None
        ):
            for evicted in admission.evicted_leases:
                arbiter.release(evicted.backup_key)
            if (
                payload.get("placement_mode") == "certified_exact"
                and admission.evicted_leases
            ):
                payload["exact_deployment_violation"] = (
                    self._record_exact_deployment_violation(
                        {
                            "request_id": str(payload.get("rid") or ""),
                            "backup_key": backup_key,
                            "target_dp_rank": target_rank,
                            "pp_rank": stage_rank,
                            "budget_version": budget.budget_version,
                            "reason": "unexpected_physical_pool_eviction",
                            "evicted_backup_keys": list(
                                admission.evicted_backup_keys
                            ),
                            "solver_decision_sha256": str(
                                payload.get("solver_decision_sha256") or ""
                            ),
                        }
                    )
                )
            payload["host_pool_utility_density_per_allocated_token"] = (
                admission.lease.utility_density
            )
            payload["host_pool_lease"] = asdict(admission.lease)
            payload["host_capacity_lease"] = asdict(
                capacity_admission.lease
            )
            payload["host_pool_state"] = pool.snapshot()
            payload["host_pool_evicted_backup_keys"] = list(
                admission.evicted_backup_keys
            )
            payload["host_pool_evicted_versions"] = {
                lease.backup_key: lease.version
                for lease in admission.evicted_leases
            }
            self._downgrade_evicted_host_leases(
                node_id=node_id,
                evicted_leases=admission.evicted_leases,
                replacement_backup_key=backup_key,
            )
            return self._observe_node_host_budget(
                payload,
                normal_l2_pressure=float(
                    payload.get("normal_l2_pressure", 0.0)
                ),
            )

        if capacity_admission.lease is not None:
            arbiter.release(
                backup_key,
                expected_lease_version=(
                    capacity_admission.lease.lease_version
                ),
            )
        rejection_reason = (
            capacity_admission.reason
            if not capacity_admission.admitted
            else (
                admission.reason
                if admission is not None
                else "physical_host_pool_rejected"
            )
        )
        if payload.get("placement_mode") == "certified_exact":
            payload["exact_deployment_violation"] = (
                self._record_exact_deployment_violation(
                    {
                        "request_id": str(payload.get("rid") or ""),
                        "backup_key": backup_key,
                        "target_dp_rank": target_rank,
                        "pp_rank": stage_rank,
                        "budget_version": budget.budget_version,
                        "reason": rejection_reason,
                        "capacity_layer": (
                            "dp_policy_or_node_physical_arbiter"
                            if not capacity_admission.admitted
                            else "aligned_physical_host_pool"
                        ),
                        "solver_decision_sha256": str(
                            payload.get("solver_decision_sha256") or ""
                        ),
                    }
                )
            )

        # A rank-local admission that cannot obtain both DP policy and physical
        # capacity is downgraded before the Router stores the recovery hint.
        payload["l2_tokens"] = 0
        independent_tiers = (
            str(payload.get("design_semantics") or "") == DESIGN_SEMANTICS
        )
        if independent_tiers:
            source_end_raw = payload.get(
                "source_committed_token_end", payload.get("token_count")
            )
            source_end = max(
                int(
                    source_end_raw
                    if source_end_raw is not None
                    else int(payload.get("safe_frontier_tokens", 0))
                    + int(payload.get("recompute_tokens", 0))
                ),
                0,
            )
            surviving_frontier = min(
                max(
                    int(payload.get("l1_tokens", 0)),
                    int(payload.get("l3_tokens", 0)),
                ),
                source_end,
            )
            payload["safe_frontier_tokens"] = surviving_frontier
            payload["recompute_tokens"] = max(
                source_end - surviving_frontier, 0
            )
        else:
            payload["recompute_tokens"] = max(
                int(payload.get("recompute_tokens", 0)), 0
            ) + l2_tokens
            payload["safe_frontier_tokens"] = max(
                int(payload.get("safe_frontier_tokens", 0)) - l2_tokens, 0
            )
            payload["recovery_action"] = (
                "l1_backup"
                if int(payload.get("l1_tokens", 0)) > 0
                else "recompute"
            )
        payload["host_pool_lease"] = None
        payload["host_capacity_lease"] = None
        payload["host_pool_admission_error"] = rejection_reason
        payload["host_pool_state"] = pool.snapshot()
        version = max(int(payload.get("version", 1)), 1)
        if self._has_runtime_command_target(target_rank):
            command = (
                "aegis_l1_l2_invalidate_host_lease",
                {
                    "backup_key": backup_key,
                    "expected_version": version,
                    "host_pool_node_id": node_id,
                },
            )
            pp_rank = self._pp_rank_from_stage_backup_key(backup_key)
            if pp_rank is None:
                self._send_dp_runtime_command(target_rank, command)
            else:
                self._send_stage_runtime_command(target_rank, pp_rank, command)
            payload["host_pool_reconciliation_pending"] = True
        self._append_aegis_event(
            "host_pool_lease_rejected",
            {
                "backup_key": backup_key,
                "version": version,
                "host_pool_node_id": node_id,
                "target_dp_rank": target_rank,
                "lost_l2_tokens": l2_tokens,
                "reason": rejection_reason,
                "capacity_layer": (
                    "dp_policy_or_node_physical_arbiter"
                    if not capacity_admission.admitted
                    else "aligned_physical_host_pool"
                ),
            },
        )
        reconciled_payload = (
            self._refresh_aegis_recovery_hint(payload)
            if independent_tiers
            else payload
        )
        return self._observe_node_host_budget(
            reconciled_payload,
            normal_l2_pressure=float(
                payload.get("normal_l2_pressure", 0.0)
            ),
        )

    def _refresh_aegis_recovery_hint(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Recompute the authoritative frontier, action, and nominal cost."""

        l1_tokens = max(int(payload.get("l1_tokens", 0)), 0)
        l2_tokens = max(int(payload.get("l2_tokens", 0)), 0)
        l3_tokens = max(int(payload.get("l3_tokens", 0)), 0)
        recompute_tokens = max(int(payload.get("recompute_tokens", 0)), 0)
        if str(payload.get("design_semantics") or "") == DESIGN_SEMANTICS:
            source_end_raw = payload.get(
                "source_committed_token_end", payload.get("token_count")
            )
            source_end = max(
                int(
                    source_end_raw
                    if source_end_raw is not None
                    else int(payload.get("safe_frontier_tokens", 0))
                    + recompute_tokens
                ),
                0,
            )
            committed = (
                str(payload.get("materialization_scope") or "")
                == "committed_prefix"
                and bool(payload.get("group_commit_verified", False))
            )
            if not committed:
                safe_frontier = 0
                recompute_tokens = source_end
                expected_ms = (
                    self._aegis_path_costs.recompute_ms(source_end)
                    if source_end
                    else 0.0
                )
                action = "recompute"
            else:
                l1_tokens = min(l1_tokens, source_end)
                l2_tokens = min(l2_tokens, source_end)
                l3_tokens = min(l3_tokens, source_end)
                safe_frontier = max(l1_tokens, l2_tokens, l3_tokens)
                recompute_tokens = max(source_end - safe_frontier, 0)
                candidates = [
                    (
                        (
                            self._aegis_path_costs.recompute_ms(source_end)
                            if source_end
                            else 0.0
                        ),
                        "recompute",
                        0,
                    )
                ]
                if l1_tokens:
                    candidates.append(
                        (
                            self._aegis_path_costs.path_ms(
                                ("direct_hbm_reuse",), l1_tokens
                            )
                            + (
                                self._aegis_path_costs.recompute_ms(
                                    source_end - l1_tokens
                                )
                                if source_end > l1_tokens
                                else 0.0
                            ),
                            "l1_backup",
                            l1_tokens,
                        )
                    )
                if l2_tokens:
                    host_candidate = payload.get("host_candidate")
                    host_path = (
                        tuple(host_candidate.get("recovery_path", ()))
                        if isinstance(host_candidate, dict)
                        else ()
                    ) or ("h2d",)
                    candidates.append(
                        (
                            self._aegis_path_costs.path_ms(
                                host_path, l2_tokens
                            )
                            + (
                                self._aegis_path_costs.recompute_ms(
                                    source_end - l2_tokens
                                )
                                if source_end > l2_tokens
                                else 0.0
                            ),
                            "l2_backup",
                            l2_tokens,
                        )
                    )
                if l3_tokens:
                    candidates.append(
                        (
                            self._aegis_path_costs.path_ms(
                                ("l3_read", "h2d"), l3_tokens
                            )
                            + (
                                self._aegis_path_costs.recompute_ms(
                                    source_end - l3_tokens
                                )
                                if source_end > l3_tokens
                                else 0.0
                            ),
                            "l3_backup",
                            l3_tokens,
                        )
                    )
                expected_ms, action, _restored = min(
                    candidates,
                    key=lambda item: (item[0], -item[2], item[1]),
                )
            payload["l1_tokens"] = l1_tokens
            payload["l2_tokens"] = l2_tokens
            payload["l3_tokens"] = l3_tokens
            payload["safe_frontier_tokens"] = safe_frontier
            payload["continuous_recoverable_prefix_tokens"] = safe_frontier
            payload["recompute_tokens"] = recompute_tokens
            payload["recovery_action"] = action
            payload["expected_recovery_ms"] = round(expected_ms, 6)
            payload["residency"] = (
                "BYPASS"
                if not committed
                else "HBM_AND_DRAM"
                if l1_tokens and l2_tokens
                else "HBM_ONLY"
                if l1_tokens
                else "DRAM_ONLY"
                if l2_tokens
                else "BYPASS"
            )
            payload["protected_recovery_service_s"] = round(
                expected_ms / 1000.0, 9
            )
            payload["unprotected_recovery_service_s"] = round(
                (
                    self._aegis_path_costs.recompute_ms(source_end)
                    if source_end
                    else 0.0
                )
                / 1000.0,
                9,
            )
            payload["recovery_service_cost_source"] = (
                self._aegis_recovery_cost_source
            )
            payload["recovery_service_cost_calibration_id"] = (
                self._aegis_recovery_cost_calibration_id
            )
            payload["updated_time"] = self._now_fn()
            return payload
        protected_tiers = [
            tier
            for tier, tokens in (
                ("l1", l1_tokens),
                ("l2", l2_tokens),
                ("l3", l3_tokens),
            )
            if tokens > 0
        ]
        payload["safe_frontier_tokens"] = l1_tokens + l2_tokens + l3_tokens
        payload["recovery_action"] = (
            f"{'_'.join(protected_tiers)}_backup"
            if protected_tiers
            else "recompute"
        )
        expected_ms = 0.0
        if l1_tokens:
            expected_ms += self._aegis_path_costs.path_ms(
                ("direct_hbm_reuse",), l1_tokens
            )
        if l2_tokens:
            host_candidate = payload.get("host_candidate")
            host_path = (
                tuple(host_candidate.get("recovery_path", ()))
                if isinstance(host_candidate, dict)
                else ()
            ) or ("h2d",)
            expected_ms += self._aegis_path_costs.path_ms(host_path, l2_tokens)
        if l3_tokens:
            l3_candidate = payload.get("l3_candidate")
            l3_path = (
                tuple(l3_candidate.get("recovery_path", ()))
                if isinstance(l3_candidate, dict)
                else ()
            ) or ("l3_read", "h2d")
            expected_ms += self._aegis_path_costs.path_ms(l3_path, l3_tokens)
        if recompute_tokens:
            expected_ms += self._aegis_path_costs.recompute_ms(
                recompute_tokens
            )
        payload["expected_recovery_ms"] = round(expected_ms, 6)
        recovery_token_count = max(
            int(
                payload.get(
                    "source_committed_token_end",
                    payload.get(
                        "token_count",
                        l1_tokens + l2_tokens + l3_tokens + recompute_tokens,
                    ),
                )
                or 0
            ),
            0,
        )
        payload["protected_recovery_service_s"] = round(
            expected_ms / 1000.0, 9
        )
        payload["unprotected_recovery_service_s"] = round(
            self._aegis_path_costs.recompute_ms(recovery_token_count) / 1000.0,
            9,
        )
        payload["recovery_service_cost_source"] = (
            self._aegis_recovery_cost_source
        )
        payload["recovery_service_cost_calibration_id"] = (
            self._aegis_recovery_cost_calibration_id
        )
        payload["updated_time"] = self._now_fn()
        return payload

    def _sync_aegis_l3_lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Admit the residual frontier into external persistent storage."""

        backup_key = str(payload.get("backup_key") or payload.get("rid") or "")
        version = max(int(payload.get("version", 1)), 1)
        candidate = payload.get("l3_candidate")
        residual = max(int(payload.get("recompute_tokens", 0)), 0)
        payload["l3_tokens"] = 0
        if (
            not backup_key
            or not isinstance(candidate, dict)
            or not bool(candidate.get("valid", False))
            or float(candidate.get("utility", 0.0) or 0.0) <= 0.0
            or residual <= 0
        ):
            if backup_key:
                self._aegis_l3_pool.release(
                    backup_key, max_version=version
                )
            payload["persistent_pool_lease"] = None
            payload["persistent_pool_state"] = self._aegis_l3_pool.snapshot()
            return self._refresh_aegis_recovery_hint(payload)

        utility = float(candidate.get("utility", 0.0) or 0.0)
        candidate_tokens = max(
            int(candidate.get("token_count", residual) or residual), 1
        )
        residual_utility = utility * residual / candidate_tokens
        admission = self._aegis_l3_pool.admit(
            backup_key=backup_key,
            token_count=residual,
            utility_density=residual_utility / max(residual, 1),
            version=version,
        )
        if not admission.admitted or admission.lease is None:
            payload["persistent_pool_lease"] = None
            payload["persistent_pool_admission_error"] = admission.reason
            payload["persistent_pool_state"] = self._aegis_l3_pool.snapshot()
            return self._refresh_aegis_recovery_hint(payload)

        payload["l3_tokens"] = residual
        payload["recompute_tokens"] = 0
        payload["persistent_pool_lease"] = asdict(admission.lease)
        payload["persistent_pool_state"] = self._aegis_l3_pool.snapshot()
        payload["persistent_pool_evicted_backup_keys"] = [
            lease.backup_key for lease in admission.evicted_leases
        ]
        self._downgrade_evicted_persistent_leases(
            evicted_leases=admission.evicted_leases,
            replacement_backup_key=backup_key,
        )
        return self._refresh_aegis_recovery_hint(payload)

    def _downgrade_evicted_persistent_leases(
        self,
        *,
        evicted_leases: tuple[Any, ...],
        replacement_backup_key: str,
    ) -> None:
        events: list[dict[str, Any]] = []
        with self._aegis_l1_l2_lock:
            for lease in evicted_leases:
                backup_key = str(getattr(lease, "backup_key", ""))
                version = int(getattr(lease, "version", -1))
                hint = self._aegis_recovery_hints.get(backup_key)
                manifest = (
                    hint.get("persistent_pool_lease")
                    if isinstance(hint, dict)
                    else None
                )
                if (
                    not backup_key
                    or not isinstance(hint, dict)
                    or not isinstance(manifest, dict)
                    or int(hint.get("version", -1)) != version
                    or int(manifest.get("version", -1)) != version
                ):
                    continue
                downgraded = dict(hint)
                lost = max(int(downgraded.get("l3_tokens", 0)), 0)
                downgraded["l3_tokens"] = 0
                downgraded["recompute_tokens"] = max(
                    int(downgraded.get("recompute_tokens", 0)), 0
                ) + lost
                downgraded["persistent_pool_lease"] = None
                downgraded["persistent_pool_eviction_reason"] = (
                    "replaced_by_higher_utility_lease"
                )
                downgraded["persistent_pool_evicted_by"] = (
                    replacement_backup_key
                )
                self._aegis_recovery_hints[backup_key] = (
                    self._refresh_aegis_recovery_hint(downgraded)
                )
                events.append(
                    {
                        "backup_key": backup_key,
                        "version": version,
                        "lost_l3_tokens": lost,
                        "replacement_backup_key": replacement_backup_key,
                    }
                )
        for event in events:
            self._append_aegis_event(
                "persistent_pool_lease_evicted", event
            )

    def _downgrade_evicted_host_leases(
        self,
        *,
        node_id: int,
        evicted_leases: tuple[Any, ...],
        replacement_backup_key: str,
    ) -> None:
        """Atomically revoke stale Router hints, then reconcile schedulers.

        The controller-owned node pool is the source of truth.  Router hints
        are downgraded synchronously so retries cannot race the asynchronous
        rank-local cleanup.  The version carried by each lease makes a delayed
        cleanup harmless if the same key has already been replaced.
        """

        reconciliations: list[tuple[int, int | None, str, int, int]] = []
        arbiter = self._aegis_node_capacity_arbiters.get(int(node_id))
        with self._aegis_l1_l2_lock:
            for lease in evicted_leases:
                backup_key = str(getattr(lease, "backup_key", ""))
                expected_version = int(getattr(lease, "version", -1))
                if arbiter is not None and backup_key:
                    arbiter.release(backup_key)
                hint = self._aegis_recovery_hints.get(backup_key)
                if not backup_key or not isinstance(hint, dict):
                    continue
                host_lease = hint.get("host_pool_lease")
                if not isinstance(host_lease, dict):
                    continue
                if (
                    int(host_lease.get("node_id", -1)) != int(node_id)
                    or int(host_lease.get("version", -1)) != expected_version
                    or int(hint.get("version", -1)) != expected_version
                ):
                    continue
                downgraded = dict(hint)
                lost = max(int(downgraded.get("l2_tokens", 0)), 0)
                downgraded["l2_tokens"] = 0
                if (
                    str(downgraded.get("design_semantics") or "")
                    == DESIGN_SEMANTICS
                ):
                    source_end_raw = downgraded.get(
                        "source_committed_token_end",
                        downgraded.get("token_count"),
                    )
                    source_end = max(
                        int(
                            source_end_raw
                            if source_end_raw is not None
                            else int(
                                downgraded.get("safe_frontier_tokens", 0)
                            )
                            + int(downgraded.get("recompute_tokens", 0))
                        ),
                        0,
                    )
                    downgraded["recompute_tokens"] = max(
                        source_end
                        - max(
                            int(downgraded.get("l1_tokens", 0)),
                            int(downgraded.get("l3_tokens", 0)),
                        ),
                        0,
                    )
                else:
                    downgraded["recompute_tokens"] = max(
                        int(downgraded.get("recompute_tokens", 0)), 0
                    ) + lost
                downgraded["host_pool_lease"] = None
                downgraded["host_pool_eviction_reason"] = (
                    "replaced_by_higher_utility_lease"
                )
                downgraded["host_pool_evicted_by"] = replacement_backup_key
                self._aegis_recovery_hints[backup_key] = (
                    self._refresh_aegis_recovery_hint(downgraded)
                )
                target_rank = int(downgraded.get("target_dp_rank", -1))
                reconciliations.append(
                    (
                        target_rank,
                        self._pp_rank_from_stage_backup_key(backup_key),
                        backup_key,
                        expected_version,
                        lost,
                    )
                )

        for (
            target_rank,
            pp_rank,
            backup_key,
            expected_version,
            lost,
        ) in reconciliations:
            if self._has_runtime_command_target(target_rank):
                command = (
                    "aegis_l1_l2_invalidate_host_lease",
                    {
                        "backup_key": backup_key,
                        "expected_version": expected_version,
                        "host_pool_node_id": int(node_id),
                    },
                )
                if pp_rank is None:
                    self._send_dp_runtime_command(target_rank, command)
                else:
                    self._send_stage_runtime_command(
                        target_rank, pp_rank, command
                    )
            self._append_aegis_event(
                "host_pool_lease_evicted",
                {
                    "backup_key": backup_key,
                    "version": expected_version,
                    "host_pool_node_id": int(node_id),
                    "target_dp_rank": target_rank,
                    "lost_l2_tokens": lost,
                    "replacement_backup_key": replacement_backup_key,
                },
            )

    def _apply_gpu_recovery_l1_clear(self, event: dict[str, Any]) -> None:
        event = self._aggregate_pipeline_gpu_recovery_l1_clear(event)
        if event is None:
            return
        request_id = str(event.get("request_id") or "")
        dp_rank = int(event.get("dp_rank", -1))
        cache_result = event.get("cache_result")
        backup_result = event.get("backup_result")
        rank_state = event.get("rank_state")
        if not isinstance(cache_result, dict):
            cache_result = {}
        if not isinstance(backup_result, dict):
            backup_result = {}
        updated_hints = backup_result.get("updated_hints", [])
        with self._aegis_l1_l2_lock:
            if isinstance(updated_hints, list):
                for hint in updated_hints:
                    if not isinstance(hint, dict):
                        continue
                    backup_key = str(
                        hint.get("backup_key") or hint.get("rid") or ""
                    )
                    if backup_key:
                        self._aegis_recovery_hints[backup_key] = dict(hint)
            if isinstance(rank_state, dict) and dp_rank >= 0:
                self._aegis_rank_states[dp_rank] = dict(rank_state)
        ledger_result = self._aegis_gpu_ledger.clear_rank_l1(dp_rank)
        if self._aegis_rehome_after_recovery and dp_rank >= 0:
            for cache_key in ledger_result.get(
                "cleared_inference_cache_keys", ()
            ):
                if cache_key:
                    self._aegis_pending_rehome[str(cache_key)] = dp_rank
        result = {
            "request_id": request_id,
            "dp_rank": dp_rank,
            "cache_result": dict(cache_result),
            "backup_result": dict(backup_result),
            "rank_state": dict(rank_state) if isinstance(rank_state, dict) else {},
            "router_ledger_result": ledger_result,
            "cleared_time": float(event.get("cleared_time", self._now_fn())),
        }
        if isinstance(event.get("stage_results"), list):
            result.update(
                {
                    "stage_results": [
                        dict(item)
                        for item in event["stage_results"]
                        if isinstance(item, dict)
                    ],
                    "all_stages_cleared": bool(
                        event.get("all_stages_cleared", False)
                    ),
                    "cleared_stage_count": int(
                        event.get("cleared_stage_count", 0)
                    ),
                    "required_stage_count": int(
                        event.get("required_stage_count", 0)
                    ),
                }
            )
        self._append_aegis_event(
            "gpu_recovery_l1_cleared",
            {
                "dp_rank": dp_rank,
                "all_stages_cleared": bool(
                    event.get("all_stages_cleared", False)
                ),
                "cleared_stage_count": int(
                    event.get("cleared_stage_count", 1)
                ),
                "required_stage_count": int(
                    event.get("required_stage_count", 1)
                ),
                "cleared_device_tokens": int(
                    cache_result.get("cleared_device_tokens", 0)
                ),
                "cleared_l1_backup_tokens": int(
                    backup_result.get("cleared_l1_backup_tokens", 0)
                ),
                "preserved_l2_backup_tokens": int(
                    backup_result.get("preserved_l2_backup_tokens", 0)
                ),
                **ledger_result,
            },
        )
        with self._gpu_recovery_lock:
            self._gpu_recovery_results[request_id] = result
            waiter = self._gpu_recovery_waiters.get(request_id)
            if waiter is not None:
                waiter.set()

    def _aggregate_pipeline_gpu_recovery_l1_clear(
        self, event: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Wait for every PP stage before publishing one DP recovery ACK.

        A GPU-rank recovery is a DP-replica transaction: all independent PP
        workers must clear their stage-local volatile KV before the Router may
        make that replica eligible again.  PP=1 and legacy unit events without
        stage identity retain the original one-ACK behavior.
        """

        required_pp_size = int(self.server_args.pp_size)
        if required_pp_size <= 1 or "pp_rank" not in event:
            return event

        request_id = str(event.get("request_id") or "")
        if not request_id:
            raise ValueError("pipeline GPU L1 clear ACK is missing request_id")
        dp_rank = int(event.get("dp_rank", -1))
        self._validate_rank(dp_rank)
        pp_rank = int(event.get("pp_rank", -1))
        if not 0 <= pp_rank < required_pp_size:
            raise ValueError(
                f"GPU L1 clear ACK pp_rank={pp_rank} is outside "
                f"PP topology size {required_pp_size}"
            )

        normalized = dict(event)
        normalized["request_id"] = request_id
        normalized["dp_rank"] = dp_rank
        normalized["pp_rank"] = pp_rank
        with self._gpu_recovery_lock:
            pending = self._gpu_recovery_stage_results.setdefault(
                request_id,
                {"dp_rank": dp_rank, "stages": {}},
            )
            expected_dp_rank = int(pending.get("dp_rank", -1))
            if expected_dp_rank != dp_rank:
                raise ValueError(
                    "GPU L1 clear ACK DP mismatch: "
                    f"request={request_id}, expected={expected_dp_rank}, "
                    f"received={dp_rank}"
                )
            stages = pending.setdefault("stages", {})
            previous = stages.get(pp_rank)
            if previous is not None:
                if previous != normalized:
                    raise ValueError(
                        "conflicting duplicate GPU L1 clear ACK for "
                        f"request={request_id}, pp_rank={pp_rank}"
                    )
                return None
            stages[pp_rank] = normalized
            cleared_stage_count = len(stages)
            if cleared_stage_count < required_pp_size:
                ordered_stage_events = None
            else:
                missing = [
                    rank for rank in range(required_pp_size) if rank not in stages
                ]
                if missing:
                    raise ValueError(
                        f"GPU L1 clear ACK set is missing PP stages {missing}"
                    )
                ordered_stage_events = [
                    dict(stages[rank]) for rank in range(required_pp_size)
                ]
                self._gpu_recovery_stage_results.pop(request_id, None)

        self._append_aegis_event(
            "gpu_recovery_l1_stage_cleared",
            {
                "request_id": request_id,
                "dp_rank": dp_rank,
                "pp_rank": pp_rank,
                "cleared_stage_count": cleared_stage_count,
                "required_stage_count": required_pp_size,
            },
        )
        if ordered_stage_events is None:
            return None

        cache_result: dict[str, int | float] = {}
        backup_result: dict[str, Any] = {
            "dp_rank": dp_rank,
            "cleared_l1_backup_tokens": 0,
            "affected_backup_entries": 0,
            "preserved_l2_backup_tokens": 0,
            "updated_hints": [],
        }
        rank_states: dict[str, dict[str, Any]] = {}
        for stage_event in ordered_stage_events:
            stage_cache = stage_event.get("cache_result")
            if isinstance(stage_cache, dict):
                for key, value in stage_cache.items():
                    if isinstance(value, bool) or not isinstance(
                        value, (int, float)
                    ):
                        continue
                    cache_result[key] = cache_result.get(key, 0) + value
            stage_backup = stage_event.get("backup_result")
            if isinstance(stage_backup, dict):
                for key in (
                    "cleared_l1_backup_tokens",
                    "affected_backup_entries",
                    "preserved_l2_backup_tokens",
                ):
                    backup_result[key] += int(stage_backup.get(key, 0))
                hints = stage_backup.get("updated_hints")
                if isinstance(hints, list):
                    backup_result["updated_hints"].extend(
                        dict(hint) for hint in hints if isinstance(hint, dict)
                    )
            stage_state = stage_event.get("rank_state")
            if isinstance(stage_state, dict):
                rank_states[str(int(stage_event["pp_rank"]))] = dict(stage_state)

        return {
            "type": "gpu_recovery_l1_clear_ack",
            "request_id": request_id,
            "dp_rank": dp_rank,
            "pp_size": required_pp_size,
            "cache_result": cache_result,
            "backup_result": backup_result,
            "rank_state": {
                "dp_rank": dp_rank,
                "pp_size": required_pp_size,
                "stages": rank_states,
            },
            "stage_results": ordered_stage_events,
            "all_stages_cleared": True,
            "cleared_stage_count": required_pp_size,
            "required_stage_count": required_pp_size,
            "cleared_time": max(
                float(item.get("cleared_time", 0.0))
                for item in ordered_stage_events
            ),
        }

    def recovery_hint(self, backup_key: str) -> dict[str, Any] | None:
        with self._aegis_l1_l2_lock:
            hint = self._aegis_recovery_hints.get(str(backup_key))
            return dict(hint) if hint is not None else None

    def aegis_recompute_delay_s(self, token_count: int) -> float:
        """Return only the configured retry recomputation cost in seconds.

        Offline physical recovery replaces the aggregate tier-transfer
        estimate with a contended GlobalDES schedule, but it must retain this
        policy-owned recomputation component for the unprotected suffix.
        """

        tokens = max(int(token_count), 0)
        if tokens == 0:
            return 0.0
        return max(
            float(self._aegis_path_costs.recompute_ms(tokens)), 0.0
        ) / 1000.0

    def _resolve_lineage_recovery_hint(
        self, backup_key: str
    ) -> tuple[dict[str, Any] | None, str, tuple[str, ...]]:
        """Resolve an obsolete request key to its newest compatible frontier.

        Real-CC prompts are prefix-stable within one agent lineage.  If a
        failure manifest names an earlier completed turn after that agent has
        advanced, the latest protected frontier is a superset and can safely
        restore up to the failed request's requested token count.
        """

        key = str(backup_key)
        chain: list[str] = []
        visited: set[str] = set()
        with self._aegis_l1_l2_lock:
            while key and key not in visited:
                visited.add(key)
                chain.append(key)
                hint = self._aegis_recovery_hints.get(key)
                if isinstance(hint, dict):
                    return dict(hint), key, tuple(chain)
                tombstone = self._aegis_superseded_backups.get(key)
                if not isinstance(tombstone, dict):
                    break
                key = str(tombstone.get("superseded_by") or "")
        return None, key, tuple(chain)

    def aegis_status(self, *, after_sequence: int = 0) -> dict[str, Any]:
        with self._aegis_l1_l2_lock:
            hints = [dict(value) for value in self._aegis_recovery_hints.values()]
            states = {
                str(rank): dict(value)
                for rank, value in sorted(self._aegis_rank_states.items())
            }
            events = [
                dict(event)
                for event in self._aegis_events
                if int(event.get("sequence", 0)) > int(after_sequence)
            ]
            latest_sequence = self._aegis_event_sequence
            lineage_frontiers = [
                dict(value) for value in self._aegis_lineage_latest.values()
            ]
            superseded_backup_count = len(self._aegis_superseded_backups)
            exact = self._aegis_exact_window_plan
            exact_window = (
                None
                if exact is None
                else {
                    "instance_id": exact.instance_id,
                    "instance_sha256": exact.instance_sha256,
                    "decision_sha256": exact.decision_sha256,
                    "solver_certificate_id": exact.solver_certificate_id,
                    "solver": exact.solver,
                    "solver_version": exact.solver_version,
                    "solver_status": exact.solver_status,
                    "objective": exact.objective,
                    "dual_bound": exact.dual_bound,
                    "mip_gap": exact.mip_gap,
                    "solve_time_s": exact.solve_time_s,
                    "window_id": exact.window_id,
                    "route_version": exact.route_version,
                    "budget_publication_mode": (
                        exact.budget_publication_mode
                    ),
                    "budget_versions": dict(
                        self._aegis_exact_budget_versions
                    ),
                    "request_count": len(exact.routes),
                    "placement_count": len(exact.placements),
                    "certified_optimal": exact.certified_optimal,
                    "uses_future_manifest_oracle": (
                        exact.uses_future_manifest_oracle
                    ),
                    "admission_information_modes": sorted(
                        {
                            route.admission_information_mode
                            for route in exact.routes
                        }
                    ),
                }
            )
            exact_violations = [
                dict(value)
                for value in self._aegis_exact_deployment_violations
            ]
        router_load = [
            {
                "dp_rank": rank,
                "healthy": self._worker_available(rank),
                "inflight_requests": self._aegis_gpu_ledger.inflight_requests(rank),
                "inflight_tokens": self._aegis_gpu_ledger.inflight_tokens(rank),
            }
            for rank in range(len(self.workers))
        ]
        return {
            "policy": "per_request_aegiskv_tiered",
            "dp_size": len(self.workers),
            "per_node_aegis_dram_quota": (
                self._aegis_per_node_dram_quota.as_dict()
            ),
            "disable_slack_ordering": self._aegis_disable_slack_ordering,
            "latest_sequence": latest_sequence,
            "router_load": router_load,
            "rank_states": states,
            "node_host_states": {
                str(node_id): {
                    "pool": self._aegis_host_pools[node_id].snapshot(),
                    "physical_arbiter": self._aegis_node_capacity_arbiters[
                        node_id
                    ].snapshot(),
                    "policy_owner": "dp_controller",
                    "capacity_owner": "node_physical_arbiter",
                }
                for node_id in sorted(self._aegis_host_pools)
            },
            "persistent_pool_state": self._aegis_l3_pool.snapshot(),
            "recovery_hints": sorted(
                hints,
                key=lambda hint: float(hint.get("updated_time", 0.0)),
                reverse=True,
            ),
            "lineage_frontiers": sorted(
                lineage_frontiers,
                key=lambda row: (
                    str(row.get("workflow_id") or ""),
                    str(row.get("agent_id") or ""),
                ),
            ),
            "superseded_backup_count": superseded_backup_count,
            "certified_exact_window": exact_window,
            "exact_deployment_violation_count": len(exact_violations),
            "exact_deployment_violations": exact_violations,
            "event_history": {
                "mode": "unbounded_in_memory",
                "event_count": latest_sequence,
                "journal": (
                    self._aegis_event_journal.snapshot()
                    if self._aegis_event_journal is not None
                    else {"enabled": False}
                ),
            },
            "events": events,
        }

    def observe_completed_request(self, stats: object) -> None:
        if not online_alpha_adapter_enabled():
            return
        latency_s = float(getattr(stats, "service_time_s", 0.0) or 0.0)
        if latency_s <= 0.0:
            created = float(getattr(stats, "created_time", -1.0) or -1.0)
            last_event = float(getattr(stats, "last_event_time", -1.0) or -1.0)
            if created >= 0.0 and last_event >= created:
                latency_s = last_event - created
        if latency_s <= 0.0:
            return
        self._online_alpha_adapter.observe(
            critical_agent=bool(getattr(stats, "critical_agent", False)),
            decode_dominant=bool(getattr(stats, "decode_dominant", False)),
            stage_width=int(getattr(stats, "stage_width", 1) or 1),
            latency_s=latency_s,
        )

    def release_request(self, rid: str):
        return self._aegis_gpu_ledger.release_request(rid)

    def _send_to_worker(self, req: TokenizedGenerateReqInput, dp_rank: int) -> bool:
        if not self._worker_available(dp_rank):
            return False
        self._record_assigned_rank(req, dp_rank)
        self._record_aegis_gpu_replica(req, dp_rank)
        self.workers[dp_rank].send_pyobj(req)
        return True

    def preview_causal_execution_rank(
        self,
        req: TokenizedGenerateReqInput,
    ) -> int:
        """Preview the ordinary healthy execution route without mutating it.

        The offline exact driver calls this immediately before solving a route
        admission window.  It mirrors the non-exact prefix of
        ``cache_load_aware_scheduler`` so the finite problem receives the same
        causal execution DP that the fair matched router would choose.
        """

        sim = req.sampling_params.custom_params.setdefault("simulation", {})
        if not self._l1_l2_policy_enabled(sim):
            raise ValueError("causal exact route preview requires AegisKV L1/L2")
        cache_key = self._cache_key(req, sim)
        ranking = self._aegis_gpu_ledger.rank_execution_candidates(
            cache_key=cache_key,
            requested_tokens=self._requested_tokens(req),
            healthy_ranks=self._healthy_ranks(),
        )
        if not ranking:
            raise SchedulerWorkerDiedError(
                f"No healthy DP worker for request {req.rid}"
            )
        route_plan_rank = self._sim_int(sim, "aegis_route_plan_dp_rank", -1)
        route_plan_applied = route_plan_rank in ranking
        if route_plan_applied:
            ranking = [route_plan_rank] + [
                rank for rank in ranking if rank != route_plan_rank
            ]
        rehome_rank = self._aegis_pending_rehome.get(cache_key)
        if (
            rehome_rank is not None
            and self._worker_available(int(rehome_rank))
            and not route_plan_applied
        ):
            resolved_rehome = int(rehome_rank)
            ranking = [resolved_rehome] + [
                rank for rank in ranking if rank != resolved_rehome
            ]
        return int(ranking[0])

    def round_robin_scheduler(self, req: TokenizedGenerateReqInput) -> int | None:
        for _ in range(len(self.workers)):
            if self._worker_available(self.round_robin_counter):
                dp_rank = self.round_robin_counter
                self._send_to_worker(req, dp_rank)
                self.round_robin_counter = (self.round_robin_counter + 1) % len(
                    self.workers
                )
                return dp_rank
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
        rid = getattr(req, "rid", "?")
        logger.error("No healthy DP worker for request %s", rid)
        self.ensure_workers_alive()
        raise SchedulerWorkerDiedError(
            f"No healthy DP worker for request {rid}"
        )

    def routing_key_scheduler(self, req: TokenizedGenerateReqInput) -> int | None:
        if not req.routing_key:
            return self.round_robin_scheduler(req)

        mapped_rank = self.routing_key_to_rank.get(req.routing_key)
        if mapped_rank is not None and self._worker_available(mapped_rank):
            self._send_to_worker(req, mapped_rank)
            return mapped_rank

        for _ in range(len(self.workers)):
            rank = self.round_robin_counter
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
            if self._worker_available(rank):
                self.routing_key_to_rank[req.routing_key] = rank
                self._send_to_worker(req, rank)
                return rank

        rid = getattr(req, "rid", "?")
        logger.error("No healthy DP worker for routing-key request %s", rid)
        self.ensure_workers_alive()
        raise SchedulerWorkerDiedError(
            f"No healthy DP worker for request {rid}"
        )

    def cache_load_aware_scheduler(
        self, req: TokenizedGenerateReqInput, sim: dict
    ) -> int | None:
        if self._l1_l2_policy_enabled(sim):
            cache_key = self._cache_key(req, sim)
            requested_tokens = self._requested_tokens(req)
            ranking = self._aegis_gpu_ledger.rank_execution_candidates(
                cache_key=cache_key,
                requested_tokens=requested_tokens,
                healthy_ranks=self._healthy_ranks(),
            )
            if not ranking:
                return None
            exact_route = (
                self._aegis_exact_window_plan.route_for(req.rid)
                if self._aegis_exact_window_plan is not None
                else None
            )
            route_plan_rank = self._sim_int(
                sim, "aegis_route_plan_dp_rank", -1
            )
            route_plan_applied = route_plan_rank in ranking
            if exact_route is not None:
                observed_unavailable = tuple(
                    rank
                    for rank in range(len(self.workers))
                    if not self._worker_available(rank)
                )
                certified_unavailable = tuple(
                    sorted(exact_route.observed_unavailable_dp_ranks)
                )
                if observed_unavailable != certified_unavailable:
                    self._record_exact_deployment_violation(
                        {
                            "kind": "admission_state_mismatch",
                            "rid": req.rid,
                            "admission_state_id": (
                                exact_route.admission_state_id
                            ),
                            "admission_information_mode": (
                                exact_route.admission_information_mode
                            ),
                            "certified_unavailable_dp_ranks": list(
                                certified_unavailable
                            ),
                            "observed_unavailable_dp_ranks": list(
                                observed_unavailable
                            ),
                        }
                    )
                    raise RuntimeError(
                        "certified exact admission state does not match current "
                        f"availability for request {req.rid}: certified="
                        f"{certified_unavailable}, observed={observed_unavailable}"
                    )
                if (
                    route_plan_applied
                    and route_plan_rank != exact_route.execution_dp_rank
                ):
                    raise ValueError(
                        "matched execution route conflicts with certified "
                        f"exact route for request {req.rid}"
                    )
                workflow_id = str(sim.get("workflow_id") or "")
                if workflow_id and workflow_id != exact_route.workflow_id:
                    raise ValueError(
                        "request workflow_id conflicts with certified exact route"
                    )
                if exact_route.execution_dp_rank not in ranking:
                    raise RuntimeError(
                        "certified exact execution DP is unavailable for request "
                        f"{req.rid}"
                    )
                if (
                    exact_route.backup_dp_rank
                    == exact_route.execution_dp_rank
                    or not self._worker_available(exact_route.backup_dp_rank)
                ):
                    raise RuntimeError(
                        "certified exact backup DP is unavailable for request "
                        f"{req.rid}"
                    )
                ranking = [exact_route.execution_dp_rank] + [
                    rank
                    for rank in ranking
                    if rank != exact_route.execution_dp_rank
                ]
                route_plan_applied = (
                    route_plan_applied
                    and route_plan_rank == exact_route.execution_dp_rank
                )
            elif route_plan_applied:
                # Benchmark-only matched-route control.  This constrains the
                # execution source while preserving the normal Algorithm 2
                # backup-target scoring below.  When the planned worker is
                # unavailable, the unmodified healthy-rank ordering wins.
                ranking = [route_plan_rank] + [
                    rank for rank in ranking if rank != route_plan_rank
                ]
            rehome_rank = self._aegis_pending_rehome.get(cache_key)
            rehome = bool(
                exact_route is None
                and
                rehome_rank is not None
                and self._worker_available(int(rehome_rank))
                and not route_plan_applied
            )
            if rehome:
                rehome_rank = int(rehome_rank)
                ranking = [rehome_rank] + [
                    rank for rank in ranking if rank != rehome_rank
                ]
            execution_rank = int(ranking[0])
            backup_rank = (
                exact_route.backup_dp_rank
                if exact_route is not None
                else self._choose_aegis_backup_rank(
                    req=req,
                    sim=sim,
                    source_dp_rank=execution_rank,
                    candidate_ranks=ranking[1:],
                )
            )
            sim["aegis_cache_key"] = cache_key
            sim["aegis_cache_load_aware_routing"] = True
            sim["aegis_cache_load_aware_dp_rank"] = execution_rank
            sim["aegis_backup_dp_rank"] = backup_rank
            sim["aegis_router_top2"] = [execution_rank, backup_rank]
            if exact_route is not None:
                exact = self._aegis_exact_window_plan
                assert exact is not None
                sim.update(
                    {
                        "aegis_placement_mode": "certified_exact",
                        "aegis_exact_window_id": exact.window_id,
                        "aegis_route_version": exact.route_version,
                        "aegis_solver_instance_sha256": (
                            exact.instance_sha256
                        ),
                        "aegis_solver_decision_sha256": (
                            exact.decision_sha256
                        ),
                        "aegis_solver_certificate_id": (
                            exact.solver_certificate_id
                        ),
                        "aegis_exact_admission_time_s": (
                            exact_route.admission_time_s
                        ),
                        "aegis_exact_admission_state_id": (
                            exact_route.admission_state_id
                        ),
                        "aegis_exact_admission_information_mode": (
                            exact_route.admission_information_mode
                        ),
                        "aegis_exact_observed_unavailable_dp_ranks": list(
                            exact_route.observed_unavailable_dp_ranks
                        ),
                    }
                )
            if route_plan_rank >= 0:
                sim["aegis_route_plan_dp_rank"] = route_plan_rank
                sim["aegis_route_plan_applied"] = route_plan_applied
            if rehome:
                sim["aegis_rehome_after_recovery"] = True
                sim["aegis_rehome_dp_rank"] = execution_rank
            self._append_aegis_event(
                "router_top2",
                {
                    "rid": req.rid,
                    "execution_dp_rank": execution_rank,
                    "backup_dp_rank": backup_rank,
                    "ranking": ranking,
                    "execution_ranking": ranking,
                    "backup_target_scores": sim.get(
                        "aegis_backup_target_scores", []
                    ),
                    "route_plan_dp_rank": route_plan_rank,
                    "route_plan_applied": route_plan_applied,
                    "exact_admission_state_id": (
                        exact_route.admission_state_id
                        if exact_route is not None
                        else None
                    ),
                    "exact_admission_information_mode": (
                        exact_route.admission_information_mode
                        if exact_route is not None
                        else None
                    ),
                    "exact_observed_unavailable_dp_ranks": (
                        list(exact_route.observed_unavailable_dp_ranks)
                        if exact_route is not None
                        else []
                    ),
                },
            )
            if self._send_to_worker(req, execution_rank):
                if rehome:
                    self._aegis_pending_rehome.pop(cache_key, None)
                return execution_rank
            return None
        alpha = self._resolve_gpu_alpha(sim)
        if alpha is None:
            alpha = DEFAULT_GPU_REPLICA_ALPHA
        cache_key = self._cache_key(req, sim)
        requested_tokens = self._requested_tokens(req)
        dp_rank = self._aegis_gpu_ledger.choose_execution_rank(
            cache_key=cache_key,
            requested_tokens=requested_tokens,
            healthy_ranks=self._healthy_ranks(),
            alpha=alpha,
        )
        if dp_rank < 0:
            return None
        sim["aegis_cache_key"] = cache_key
        sim["aegis_cache_load_aware_routing"] = True
        sim["aegis_cache_load_aware_dp_rank"] = dp_rank
        if self._send_to_worker(req, dp_rank):
            return dp_rank
        return None

    def _choose_aegis_backup_rank(
        self,
        *,
        req: TokenizedGenerateReqInput,
        sim: dict[str, Any],
        source_dp_rank: int,
        candidate_ranks: list[int],
    ) -> int:
        """Choose a protection target after normal execution routing.

        Execution ranking remains SGLang-style cache/load aware.  The backup
        target is a separate Algorithm-2 decision over failure-domain
        feasibility, GPU/Host recovery utility, current protection pressure,
        and the queue delay a direct-GPU retry would encounter.
        """

        calibration_rank = self._sim_int(
            sim, "aegis_backup_route_plan_dp_rank", -1
        )
        if (
            calibration_rank != int(source_dp_rank)
            and calibration_rank in candidate_ranks
            and self._worker_available(calibration_rank)
        ):
            # Calibration-only paired replay hook.  Ordinary online/offline
            # requests do not carry this field and continue through the full
            # Algorithm-2 target scoring below.
            sim["aegis_backup_route_plan_applied"] = True
            sim["aegis_backup_target_scores"] = []
            sim["aegis_backup_target_objective"] = "calibration_route_plan"
            return calibration_rank
        if calibration_rank >= 0:
            sim["aegis_backup_route_plan_applied"] = False

        requested = self._requested_tokens(req)
        if requested <= 0:
            return -1
        critical_probability = normalized_fraction(
            sim.get("aegis_critical_probability", -1.0)
        )
        if critical_probability is None:
            critical_probability = (
                0.90 if self._sim_bool(sim, "critical_agent") else 0.35
            )
        placement_request = AegisPlacementRequest(
            rid=req.rid,
            token_count=requested,
            source_dp_rank=int(source_dp_rank),
            critical_probability=critical_probability,
            slack_lcb_ms=self._effective_aegis_slack_ms(sim),
            failure_scope=normalize_failure_scope(
                sim.get("aegis_failure_scope", "node")
            ),
        )
        allowed = self._l1_l2_allowed_tiers(sim)
        scores: list[dict[str, Any]] = []
        for rank_raw in candidate_ranks:
            rank = int(rank_raw)
            if (
                rank == int(source_dp_rank)
                or not 0 <= rank < len(self.workers)
                or not self._worker_available(rank)
            ):
                continue
            rank_state = self._aegis_rank_states.get(rank, {})
            gpu_budget = int(rank_state.get("l1_budget_tokens", -1))
            gpu_used = int(rank_state.get("l1_used_tokens", 0))
            protection_pressure = (
                gpu_used / max(gpu_budget, 1)
                if gpu_budget > 0
                else 0.0
            )
            gpu_pressure = max(
                float(rank_state.get("normal_l1_pressure", 0.0)),
                protection_pressure,
            )
            node_id = self._aegis_topology.node_for_rank(rank)
            host_snapshot = self._aegis_node_capacity_arbiters[
                node_id
            ].snapshot()
            budget_snapshot = self._aegis_budget_publishers[rank].current
            if budget_snapshot is None:
                raise RuntimeError(f"DP rank {rank} has no published budget")
            node_host_pressure = float(
                host_snapshot["resident_tokens"]
            ) / max(int(host_snapshot["physical_capacity_tokens"]), 1)
            dp_host_pressure = float(
                host_snapshot["resident_tokens_by_dp"].get(rank, 0)
            ) / max(int(budget_snapshot.host_policy_limit_tokens), 1)
            host_pressure = max(node_host_pressure, dp_host_pressure)
            gpu = self._aegis_placement_planner.gpu_candidate(
                placement_request,
                target_dp_rank=rank,
                capacity_pressure=gpu_pressure,
                path_pressure=min(
                    self._aegis_gpu_ledger.inflight_requests(rank) / 8.0,
                    1.0,
                ),
            )
            host = self._aegis_placement_planner.host_candidate(
                placement_request,
                host_pool_node_id=node_id,
                retry_dp_rank=rank,
                capacity_pressure=host_pressure,
            )
            l3_snapshot = self._aegis_l3_pool.snapshot()
            l3 = self._aegis_placement_planner.l3_candidate(
                placement_request,
                retry_dp_rank=rank,
                storage_available=(
                    self._l3_allowed(sim)
                    and int(l3_snapshot["capacity_tokens"]) > 0
                ),
                manifest_quorum=int(l3_snapshot["manifest_quorum"]),
                capacity_pressure=(
                    float(l3_snapshot["used_tokens"])
                    / max(int(l3_snapshot["capacity_tokens"]), 1)
                ),
            )
            feasible = [
                candidate
                for tier, candidate in (
                    ("l1", gpu),
                    ("l2", host),
                    ("l3", l3),
                )
                if (
                    (tier in allowed or (tier == "l3" and self._l3_allowed(sim)))
                    and candidate.valid
                )
            ]
            if not feasible:
                continue
            best = max(
                feasible,
                key=lambda candidate: (
                    candidate.utility,
                    -candidate.recovery_ms,
                ),
            )
            retry_queue_ms = (
                0.002 * self._aegis_gpu_ledger.inflight_tokens(rank)
            )
            scores.append(
                {
                    "rank": rank,
                    "node_id": node_id,
                    "best_tier": best.tier,
                    "best_utility": best.utility,
                    "best_recovery_ms": best.recovery_ms,
                    "retry_queue_ms": retry_queue_ms,
                    "gpu_valid": gpu.valid,
                    "gpu_utility": gpu.utility,
                    "host_valid": host.valid,
                    "host_utility": host.utility,
                    "l3_valid": l3.valid,
                    "l3_utility": l3.utility,
                    "objective": best.utility - critical_probability * retry_queue_ms,
                }
            )
        if not scores:
            sim["aegis_backup_target_scores"] = []
            return -1
        selected = max(
            scores,
            key=lambda row: (
                float(row["objective"]),
                -float(row["best_recovery_ms"]),
                -float(row["retry_queue_ms"]),
                -int(row["rank"]),
            ),
        )
        sim["aegis_backup_target_scores"] = scores
        sim["aegis_backup_target_objective"] = "recovery_utility_minus_retry_queue"
        return int(selected["rank"])

    def dispatch(self, req: TokenizedGenerateReqInput) -> int | None:
        self.ensure_workers_alive()
        sim = req.sampling_params.custom_params.get("simulation", {})
        if self._l1_l2_policy_enabled(sim) and self._sim_bool(
            sim, "failover_replay", False
        ):
            retry_rank = self._choose_l1_l2_retry_rank(req, sim)
            if retry_rank >= 0 and self._send_to_worker(req, retry_rank):
                return retry_rank
        force_dp_rank = sim.get("force_dp_rank")
        if force_dp_rank is not None:
            try:
                dp_rank = int(force_dp_rank)
            except (TypeError, ValueError):
                dp_rank = -1
            if 0 <= dp_rank < len(self.workers) and self._worker_available(dp_rank):
                self._send_to_worker(req, dp_rank)
                return dp_rank
            logger.warning(
                "Ignoring invalid forced DP rank %s for request %s",
                force_dp_rank,
                getattr(req, "rid", "?"),
            )

        if self.load_balance_method == LoadBalanceMethod.CACHE_AWARE:
            dp_rank = self.cache_load_aware_scheduler(req, sim)
            if dp_rank is not None:
                return dp_rank
            return self.round_robin_scheduler(req)

        if self.load_balance_method == LoadBalanceMethod.ROUND_ROBIN:
            if self._aegis_routing_enabled(sim):
                dp_rank = self.cache_load_aware_scheduler(req, sim)
                if dp_rank is not None:
                    return dp_rank
            # Mirror sglang round_robin_scheduler: when routing-key affinity is
            # enabled, sticky round-robin assigns the first-seen key to the next
            # healthy DP rank and keeps subsequent requests on that rank.
            if self.enable_routing_key_affinity:
                return self.routing_key_scheduler(req)
            return self.round_robin_scheduler(req)
        return None

    def _pipeline_independent_retry_option(
        self,
        *,
        hint: dict[str, Any],
        retry_dp_rank: int,
        requested_tokens: int,
        failed_dp_rank: int,
        live_primary_tokens: int,
    ) -> dict[str, Any]:
        """Select a common PP prefix from independent per-stage tier copies."""

        raw_payloads = hint.get("stage_payloads")
        if not isinstance(raw_payloads, list) or len(raw_payloads) != int(
            self.server_args.pp_size
        ):
            raise ValueError(
                "latest stage-aligned hint does not cover the PP topology"
            )
        requested = max(int(requested_tokens), 0)
        live_prefix = min(max(int(live_primary_tokens), 0), requested)
        failed_node = (
            self._aegis_topology.node_for_rank(failed_dp_rank)
            if 0 <= failed_dp_rank < len(self.workers)
            else -1
        )
        stage_capabilities: list[dict[str, Any]] = []
        for index, raw_payload in enumerate(raw_payloads):
            if not isinstance(raw_payload, dict):
                raise ValueError(
                    "latest stage-aligned hint contains a missing payload"
                )
            payload = dict(raw_payload)
            pp_rank = int(payload.get("pp_rank", index))
            target = int(payload.get("target_dp_rank", -1))
            gpu_candidate = payload.get("gpu_candidate")
            gpu_candidate = (
                gpu_candidate if isinstance(gpu_candidate, dict) else {}
            )
            reusable_raw = gpu_candidate.get(
                "reusable_dp_ranks", (target,)
            )
            reusable_ranks = {
                int(rank)
                for rank in reusable_raw
                if str(rank).lstrip("-").isdigit()
            }
            hbm_usable = int(retry_dp_rank) in reusable_ranks
            hbm_prefix = (
                min(max(int(payload.get("l1_tokens", 0)), 0), requested)
                if hbm_usable
                else 0
            )

            host_candidate = payload.get("host_candidate")
            host_candidate = (
                host_candidate if isinstance(host_candidate, dict) else {}
            )
            host_node = int(host_candidate.get("host_pool_node_id", -1))
            host_usable = bool(host_candidate.get("valid", True)) and host_node >= 0
            if "host_pool_lease" in payload and payload.get(
                "host_pool_lease"
            ) is None:
                host_usable = False
            failure_scope = normalize_failure_scope(
                payload.get("failure_scope", hint.get("failure_scope", "node"))
            )
            if host_usable and failed_node >= 0:
                if failure_scope == "node" and host_node == failed_node:
                    host_usable = False
                elif failure_scope == "rack" and (
                    self._aegis_topology.rack_for_node(host_node)
                    == self._aegis_topology.rack_for_node(failed_node)
                ):
                    host_usable = False
            dram_prefix = (
                min(max(int(payload.get("l2_tokens", 0)), 0), requested)
                if host_usable
                else 0
            )
            retry_node = self._aegis_topology.node_for_rank(retry_dp_rank)
            host_path = tuple(host_candidate.get("recovery_path", ())) or (
                ("h2d",)
                if host_usable and retry_node == host_node
                else ("h2h", "h2d")
            )
            paths: list[tuple[str, int, tuple[str, ...]]] = []
            if live_prefix:
                paths.append(("live", live_prefix, ("direct_hbm_reuse",)))
            if hbm_prefix:
                paths.append(("hbm", hbm_prefix, ("direct_hbm_reuse",)))
            if dram_prefix:
                paths.append(("dram", dram_prefix, host_path))
            stage_capabilities.append(
                {
                    "pp_rank": pp_rank,
                    "hbm_usable": hbm_usable,
                    "host_usable": host_usable,
                    "hbm_prefix": hbm_prefix,
                    "dram_prefix": dram_prefix,
                    "paths": paths,
                }
            )

        def recompute_ms(tokens: int) -> float:
            return (
                self._aegis_path_costs.recompute_ms(tokens)
                if int(tokens) > 0
                else 0.0
            )

        frontier_candidates = {0}
        for stage in stage_capabilities:
            frontier_candidates.update(
                int(prefix) for _tier, prefix, _path in stage["paths"]
            )
        feasible: list[tuple[float, int, list[dict[str, Any]]]] = []
        for frontier in sorted(frontier_candidates):
            stage_rows: list[dict[str, Any]] = []
            possible = True
            for stage in stage_capabilities:
                if frontier == 0:
                    tier = "recompute"
                    path: tuple[str, ...] = ()
                    load_ms = 0.0
                else:
                    path_candidates = [
                        (
                            self._aegis_path_costs.path_ms(path, frontier),
                            tier,
                            path,
                        )
                        for tier, capacity, path in stage["paths"]
                        if capacity >= frontier
                    ]
                    if not path_candidates:
                        possible = False
                        break
                    load_ms, tier, path = min(
                        path_candidates,
                        key=lambda item: (item[0], item[1]),
                    )
                total_ms = load_ms + recompute_ms(requested - frontier)
                stage_rows.append(
                    {
                        "pp_rank": int(stage["pp_rank"]),
                        "restored_tokens": frontier,
                        "recomputed_tokens": max(requested - frontier, 0),
                        "protected_frontier_tokens": max(
                            int(stage["hbm_prefix"]),
                            int(stage["dram_prefix"]),
                        ),
                        "hbm_usable": bool(stage["hbm_usable"]),
                        "host_usable": bool(stage["host_usable"]),
                        "selected_tier": tier,
                        "recovery_path": list(path) + (
                            ["recompute"] if requested > frontier else []
                        ),
                        "recovery_ms": total_ms,
                    }
                )
            if possible:
                feasible.append(
                    (
                        max(
                            (float(row["recovery_ms"]) for row in stage_rows),
                            default=recompute_ms(requested),
                        ),
                        frontier,
                        stage_rows,
                    )
                )
        recovery_ms, restored, stage_rows = min(
            feasible,
            key=lambda item: (item[0], -item[1]),
        )
        stage_rows.sort(key=lambda row: int(row["pp_rank"]))
        selected_tiers = {
            str(row["selected_tier"])
            for row in stage_rows
            if str(row["selected_tier"]) != "recompute"
        }
        action = (
            "recompute"
            if not selected_tiers
            else "stage_aligned_live_l1_reuse"
            if selected_tiers == {"live"}
            else "stage_aligned_l1_backup"
            if selected_tiers <= {"live", "hbm"}
            else "stage_aligned_l2_backup"
            if selected_tiers == {"dram"}
            else "stage_aligned_mixed_tier_backup"
        )
        any_hbm = bool(selected_tiers & {"live", "hbm"})
        any_dram = "dram" in selected_tiers
        stage_safe_frontiers = [
            int(row["protected_frontier_tokens"]) for row in stage_rows
        ]
        return {
            "l1_tokens": restored if any_hbm else 0,
            "l2_tokens": restored if any_dram else 0,
            "l3_tokens": 0,
            "live_primary_tokens": live_prefix,
            "restored_tokens": restored,
            "recompute_tokens": max(requested - restored, 0),
            "recovery_ms": recovery_ms,
            "action": action,
            "gpu_direct_reuse": any_hbm,
            "host_restore_path": (
                ("stage_aligned_host",) if any_dram else ()
            ),
            "l3_restore_path": (),
            "per_stage": stage_rows,
            "route_version": int(hint.get("route_version", 0)),
            "budget_version": int(hint.get("budget_version", 0)),
            "bottleneck_pp_rank": min(
                int(row["pp_rank"])
                for row in stage_rows
                if float(row["recovery_ms"]) == recovery_ms
            ),
            "safe_frontier_tokens": min(stage_safe_frontiers, default=0),
            "stage_safe_frontiers": stage_safe_frontiers,
            "independent_tier_results": True,
        }

    def _pipeline_retry_option(
        self,
        *,
        hint: dict[str, Any],
        retry_dp_rank: int,
        requested_tokens: int,
        failed_dp_rank: int,
        live_primary_tokens: int,
    ) -> dict[str, Any]:
        if str(hint.get("design_semantics") or "") == DESIGN_SEMANTICS:
            return self._pipeline_independent_retry_option(
                hint=hint,
                retry_dp_rank=retry_dp_rank,
                requested_tokens=requested_tokens,
                failed_dp_rank=failed_dp_rank,
                live_primary_tokens=live_primary_tokens,
            )
        raw_acks = hint.get("stage_acks")
        raw_payloads = hint.get("stage_payloads")
        if not isinstance(raw_acks, list) or len(raw_acks) != int(
            self.server_args.pp_size
        ):
            raise ValueError("stage-aligned hint does not cover the PP topology")
        payload_by_pp = {
            int(payload.get("pp_rank", index)): payload
            for index, payload in enumerate(raw_payloads or [])
            if isinstance(payload, dict)
        }
        requested = max(int(requested_tokens), 0)
        failed_node = (
            self._aegis_topology.node_for_rank(failed_dp_rank)
            if 0 <= failed_dp_rank < len(self.workers)
            else -1
        )
        stage_rows: list[dict[str, Any]] = []
        for raw_ack in raw_acks:
            if not isinstance(raw_ack, dict):
                raise ValueError("stage-aligned hint contains a missing ACK")
            ack = StageCommitAck.from_dict(raw_ack)
            pp_rank = ack.source_stage.pp_rank
            payload = payload_by_pp.get(pp_rank, {})
            gpu_candidate = payload.get("gpu_candidate")
            gpu_candidate = (
                gpu_candidate if isinstance(gpu_candidate, dict) else {}
            )
            reusable_raw = gpu_candidate.get(
                "reusable_dp_ranks", (ack.backup_stage.dp_rank,)
            )
            reusable_ranks = {
                int(rank)
                for rank in reusable_raw
                if str(rank).lstrip("-").isdigit()
            }
            hbm_usable = int(retry_dp_rank) in reusable_ranks

            host_candidate = payload.get("host_candidate")
            host_candidate = (
                host_candidate if isinstance(host_candidate, dict) else {}
            )
            host_lease = payload.get("host_pool_lease")
            host_node = int(host_candidate.get("host_pool_node_id", -1))
            host_usable = (
                ack.host_tokens > 0
                and isinstance(host_lease, dict)
                and host_node >= 0
            )
            failure_scope = normalize_failure_scope(
                payload.get("failure_scope", hint.get("failure_scope", "node"))
            )
            if host_usable and failed_node >= 0:
                if failure_scope == "node" and host_node == failed_node:
                    host_usable = False
                elif failure_scope == "rack" and (
                    self._aegis_topology.rack_for_node(host_node)
                    == self._aegis_topology.rack_for_node(failed_node)
                ):
                    host_usable = False

            if ack.hbm_tokens and ack.host_tokens:
                if hbm_usable and host_usable:
                    protected_frontier = ack.committed_frontier_tokens
                else:
                    # Without explicit chunk frontiers, tier interleaving is
                    # unknown.  Never infer a contiguous prefix from a partial
                    # subset of mixed-tier bytes.
                    protected_frontier = 0
            elif ack.hbm_tokens:
                protected_frontier = (
                    ack.committed_frontier_tokens if hbm_usable else 0
                )
            elif ack.host_tokens:
                protected_frontier = (
                    ack.committed_frontier_tokens if host_usable else 0
                )
            else:
                protected_frontier = 0
            protected_frontier = min(protected_frontier, requested)
            restored = max(
                min(int(live_primary_tokens), requested),
                protected_frontier,
            )
            recomputed = max(requested - restored, 0)
            restore_ms = 0.0
            paths: list[str] = []
            if restored <= live_primary_tokens and restored > 0:
                paths.append("live_l1_reuse")
                restore_ms += self._aegis_path_costs.path_ms(
                    ("direct_hbm_reuse",), restored
                )
            elif protected_frontier > 0:
                paths.extend(ack.recovery_path)
                restore_ms += ack.expected_restore_latency_s * 1000.0
            if recomputed:
                paths.append("recompute")
                restore_ms += self._aegis_path_costs.recompute_ms(recomputed)
            stage_rows.append(
                {
                    "pp_rank": pp_rank,
                    "restored_tokens": restored,
                    "recomputed_tokens": recomputed,
                    "protected_frontier_tokens": protected_frontier,
                    "hbm_usable": hbm_usable,
                    "host_usable": host_usable,
                    "recovery_path": paths or ["recompute"],
                    "recovery_ms": restore_ms,
                }
            )

        stage_rows.sort(key=lambda row: int(row["pp_rank"]))
        restored = min(
            int(row["restored_tokens"]) for row in stage_rows
        )
        recompute_tokens = max(requested - restored, 0)
        common_hbm = min(
            min(
                StageCommitAck.from_dict(raw_acks[index]).hbm_tokens,
                int(stage_rows[index]["protected_frontier_tokens"]),
            )
            if bool(stage_rows[index]["hbm_usable"])
            else 0
            for index in range(len(stage_rows))
        )
        host_tokens = max(restored - max(live_primary_tokens, common_hbm), 0)
        used_protected_hint = restored > live_primary_tokens
        return {
            "l1_tokens": max(min(common_hbm, restored), live_primary_tokens),
            "l2_tokens": host_tokens,
            "l3_tokens": 0,
            "live_primary_tokens": live_primary_tokens,
            "restored_tokens": restored,
            "recompute_tokens": recompute_tokens,
            "recovery_ms": max(
                float(row["recovery_ms"]) for row in stage_rows
            ),
            "action": (
                "stage_aligned_restore"
                if used_protected_hint
                else "live_l1_reuse"
                if restored > 0
                else "recompute"
            ),
            "gpu_direct_reuse": common_hbm > 0,
            "host_restore_path": (
                ("stage_aligned_host",) if host_tokens > 0 else ()
            ),
            "l3_restore_path": (),
            "per_stage": stage_rows,
            "route_version": int(hint.get("route_version", 0)),
            "budget_version": int(hint.get("budget_version", 0)),
            "bottleneck_pp_rank": min(
                int(row["pp_rank"])
                for row in stage_rows
                if int(row["restored_tokens"]) == restored
            ),
        }

    def _aegis_independent_retry_option(
        self,
        *,
        hint: dict[str, Any],
        retry_dp_rank: int,
        requested_tokens: int,
        failed_dp_rank: int,
        live_primary_tokens: int,
    ) -> dict[str, Any]:
        """Choose one complete latest-design recovery path.

        HBM and DRAM are overlapping, independently committed prefixes.  They
        are alternatives at failure time and must never be added as disjoint
        token ranges.  The selected path minimizes modeled recovery time; the
        unprotected suffix, if any, is recomputed.
        """

        requested = max(int(requested_tokens), 0)
        live_prefix = min(max(int(live_primary_tokens), 0), requested)
        target = int(hint.get("target_dp_rank", -1))

        gpu_candidate = hint.get("gpu_candidate")
        gpu_candidate = (
            gpu_candidate if isinstance(gpu_candidate, dict) else {}
        )
        reusable_raw = gpu_candidate.get("reusable_dp_ranks", (target,))
        reusable_gpu_ranks = {
            int(rank)
            for rank in reusable_raw
            if str(rank).lstrip("-").isdigit()
        }
        hbm_usable = int(retry_dp_rank) in reusable_gpu_ranks
        hbm_prefix = (
            min(max(int(hint.get("l1_tokens", 0)), 0), requested)
            if hbm_usable
            else 0
        )

        host_candidate = hint.get("host_candidate")
        host_restore_path: tuple[str, ...] = ()
        host_copy_valid = False
        if isinstance(host_candidate, dict) and bool(
            host_candidate.get("valid", True)
        ):
            host_node = int(host_candidate.get("host_pool_node_id", -1))
            host_copy_valid = host_node >= 0
            if "host_pool_lease" in hint and hint.get("host_pool_lease") is None:
                host_copy_valid = False
            failure_scope = normalize_failure_scope(
                hint.get("failure_scope", "node")
            )
            if host_copy_valid and failed_dp_rank >= 0:
                failed_node = self._aegis_topology.node_for_rank(
                    failed_dp_rank
                )
                if failure_scope == "node" and host_node == failed_node:
                    host_copy_valid = False
                elif failure_scope == "rack" and (
                    self._aegis_topology.rack_for_node(host_node)
                    == self._aegis_topology.rack_for_node(failed_node)
                ):
                    host_copy_valid = False
            if host_copy_valid:
                retry_node = self._aegis_topology.node_for_rank(retry_dp_rank)
                configured_path = tuple(
                    host_candidate.get("recovery_path", ())
                )
                host_restore_path = configured_path or (
                    ("h2d",)
                    if retry_node == host_node
                    else ("h2h", "h2d")
                )
        dram_prefix = (
            min(max(int(hint.get("l2_tokens", 0)), 0), requested)
            if host_copy_valid
            else 0
        )

        def recompute_ms(tokens: int) -> float:
            return (
                self._aegis_path_costs.recompute_ms(tokens)
                if int(tokens) > 0
                else 0.0
            )

        candidates: list[
            tuple[float, int, str, str, tuple[str, ...]]
        ] = [
            (
                recompute_ms(requested),
                0,
                "recompute",
                "recompute",
                (),
            )
        ]
        effective_hbm_prefix = max(live_prefix, hbm_prefix)
        if effective_hbm_prefix:
            hbm_action = (
                "live_l1_reuse"
                if live_prefix >= hbm_prefix
                else "l1_backup"
            )
            hbm_path = ("direct_hbm_reuse",)
            candidates.append(
                (
                    self._aegis_path_costs.path_ms(
                        hbm_path, effective_hbm_prefix
                    )
                    + recompute_ms(requested - effective_hbm_prefix),
                    effective_hbm_prefix,
                    hbm_action,
                    "hbm",
                    hbm_path,
                )
            )
        if dram_prefix:
            candidates.append(
                (
                    self._aegis_path_costs.path_ms(
                        host_restore_path, dram_prefix
                    )
                    + recompute_ms(requested - dram_prefix),
                    dram_prefix,
                    "l2_backup",
                    "dram",
                    host_restore_path,
                )
            )
        recovery_ms, restored, action, selected_tier, selected_path = min(
            candidates,
            key=lambda item: (item[0], -item[1], item[2]),
        )
        return {
            "l1_tokens": restored if selected_tier == "hbm" else 0,
            "l2_tokens": restored if selected_tier == "dram" else 0,
            "l3_tokens": 0,
            "available_l1_tokens": hbm_prefix,
            "available_l2_tokens": dram_prefix,
            "safe_frontier_tokens": max(hbm_prefix, dram_prefix),
            "live_primary_tokens": live_prefix,
            "restored_tokens": restored,
            "recompute_tokens": max(requested - restored, 0),
            "recovery_ms": recovery_ms,
            "action": action,
            "gpu_direct_reuse": selected_tier == "hbm",
            "host_restore_path": (
                selected_path if selected_tier == "dram" else ()
            ),
            "l3_restore_path": (),
            "selected_tier": selected_tier,
            "independent_tier_results": True,
        }

    def _aegis_retry_option(
        self,
        *,
        hint: dict[str, Any] | None,
        retry_dp_rank: int,
        requested_tokens: int,
        failed_dp_rank: int,
        cache_key: str = "",
    ) -> dict[str, Any]:
        """Materialize live-prefix and protected subsets for one retry rank."""

        requested = max(int(requested_tokens), 0)
        committed_source_end = (
            self._sim_int(hint, "source_committed_token_end", -1)
            if hint is not None
            else -1
        )
        committed_frontier = (
            self._sim_int(hint, "safe_frontier_tokens", -1)
            if hint is not None
            else -1
        )
        committed_generation = (
            self._sim_int(hint, "block_generation", -1)
            if hint is not None
            else -1
        )
        if (
            self._aegis_committed_copy_protocol
            and hint is not None
            and not (
                str(hint.get("materialization_scope"))
                == "committed_prefix"
                and bool(hint.get("group_commit_verified", False))
                and committed_source_end >= 0
                and 0 <= committed_frontier <= committed_source_end
                and bool(str(hint.get("execution_id") or ""))
                and committed_generation >= 0
            )
        ):
            # A capacity reservation is not a copy.  In committed-copy mode it
            # can affect future placement, but never a recovery decision.
            hint = None
        recompute_only_ms = self._aegis_path_costs.recompute_ms(requested)
        live_primary_tokens = (
            min(
                self._aegis_gpu_ledger.prefix_tokens(
                    int(retry_dp_rank), str(cache_key)
                ),
                requested,
            )
            if int(retry_dp_rank) != int(failed_dp_rank)
            else 0
        )
        if (
            hint is not None
            and bool(hint.get("all_stages_committed", False))
            and isinstance(hint.get("stage_acks"), list)
        ):
            return self._pipeline_retry_option(
                hint=hint,
                retry_dp_rank=retry_dp_rank,
                requested_tokens=requested,
                failed_dp_rank=failed_dp_rank,
                live_primary_tokens=live_primary_tokens,
            )
        if hint is None or not bool(hint.get("ack", False)):
            recompute_tokens = max(requested - live_primary_tokens, 0)
            recovery_ms = 0.0
            if live_primary_tokens:
                recovery_ms += self._aegis_path_costs.path_ms(
                    ("direct_hbm_reuse",), live_primary_tokens
                )
            if recompute_tokens:
                recovery_ms += self._aegis_path_costs.recompute_ms(
                    recompute_tokens
                )
            return {
                "l1_tokens": live_primary_tokens,
                "l2_tokens": 0,
                "l3_tokens": 0,
                "live_primary_tokens": live_primary_tokens,
                "restored_tokens": live_primary_tokens,
                "recompute_tokens": recompute_tokens,
                "recovery_ms": (
                    recovery_ms if live_primary_tokens else recompute_only_ms
                ),
                "action": (
                    "live_l1_reuse" if live_primary_tokens else "recompute"
                ),
                "gpu_direct_reuse": live_primary_tokens > 0,
                "host_restore_path": (),
                "l3_restore_path": (),
            }

        if str(hint.get("design_semantics") or "") == DESIGN_SEMANTICS:
            return self._aegis_independent_retry_option(
                hint=hint,
                retry_dp_rank=retry_dp_rank,
                requested_tokens=requested,
                failed_dp_rank=failed_dp_rank,
                live_primary_tokens=live_primary_tokens,
            )

        target = int(hint.get("target_dp_rank", -1))
        gpu_candidate = hint.get("gpu_candidate")
        if not isinstance(gpu_candidate, dict):
            gpu_candidate = {}
        reusable_raw = gpu_candidate.get("reusable_dp_ranks", (target,))
        reusable_gpu_ranks = {
            int(rank)
            for rank in reusable_raw
            if isinstance(rank, (int, float, str)) and str(rank).lstrip("-").isdigit()
        }
        gpu_direct_reuse = int(retry_dp_rank) in reusable_gpu_ranks
        backup_l1_tokens = (
            max(int(hint.get("l1_tokens", 0)), 0) if gpu_direct_reuse else 0
        )
        backup_l1_tokens = min(backup_l1_tokens, requested)
        l1_tokens = max(live_primary_tokens, backup_l1_tokens)

        host_candidate = hint.get("host_candidate")
        host_restore_path: tuple[str, ...] = ()
        host_copy_valid = False
        if isinstance(host_candidate, dict) and bool(
            host_candidate.get("valid", True)
        ):
            host_node = int(host_candidate.get("host_pool_node_id", -1))
            host_copy_valid = host_node >= 0
            if "host_pool_lease" in hint and hint.get("host_pool_lease") is None:
                host_copy_valid = False
            failure_scope = normalize_failure_scope(
                hint.get("failure_scope", "node")
            )
            if failed_dp_rank >= 0 and failure_scope in {"node", "rack"}:
                failed_node = self._aegis_topology.node_for_rank(failed_dp_rank)
                if failure_scope == "node" and host_node == failed_node:
                    host_copy_valid = False
                elif failure_scope == "rack" and (
                    self._aegis_topology.rack_for_node(host_node)
                    == self._aegis_topology.rack_for_node(failed_node)
                ):
                    host_copy_valid = False
            if host_copy_valid:
                retry_node = self._aegis_topology.node_for_rank(retry_dp_rank)
                host_restore_path = (
                    ("h2d",)
                    if retry_node == host_node
                    else ("h2h", "h2d")
                )
        else:
            # Backward-compatible hints predate explicit Host topology and are
            # only safe on their original target rank.
            host_copy_valid = int(retry_dp_rank) == target
            host_restore_path = ("h2d",) if host_copy_valid else ()

        l2_tokens = max(int(hint.get("l2_tokens", 0)), 0) if host_copy_valid else 0
        l2_tokens = min(l2_tokens, max(requested - l1_tokens, 0))
        persistent_manifest = hint.get("persistent_pool_lease")
        persistent_lease = self._aegis_l3_pool.get(
            str(hint.get("backup_key") or hint.get("rid") or "")
        )
        l3_candidate = hint.get("l3_candidate")
        l3_restore_path: tuple[str, ...] = ()
        l3_copy_valid = (
            isinstance(persistent_manifest, dict)
            and persistent_lease is not None
            and persistent_lease.committed
            and int(persistent_manifest.get("version", -1))
            == persistent_lease.version
            and int(hint.get("version", -1)) == persistent_lease.version
            and int(persistent_manifest.get("manifest_version", -1))
            == persistent_lease.manifest_version
            and int(persistent_manifest.get("manifest_quorum", 0)) >= 1
        )
        if l3_copy_valid:
            l3_restore_path = (
                tuple(l3_candidate.get("recovery_path", ()))
                if isinstance(l3_candidate, dict)
                else ()
            ) or ("l3_read", "h2d")
        l3_tokens = (
            max(int(hint.get("l3_tokens", 0)), 0)
            if l3_copy_valid
            else 0
        )
        l3_tokens = min(
            l3_tokens,
            max(requested - l1_tokens - l2_tokens, 0),
        )
        restored = l1_tokens + l2_tokens + l3_tokens
        recompute_tokens = max(requested - restored, 0)
        recovery_ms = 0.0
        if l1_tokens:
            recovery_ms += self._aegis_path_costs.path_ms(
                ("direct_hbm_reuse",), l1_tokens
            )
        if l2_tokens:
            recovery_ms += self._aegis_path_costs.path_ms(
                host_restore_path, l2_tokens
            )
        if l3_tokens:
            recovery_ms += self._aegis_path_costs.path_ms(
                l3_restore_path, l3_tokens
            )
        if recompute_tokens:
            recovery_ms += self._aegis_path_costs.recompute_ms(recompute_tokens)
        protected_tiers = [
            tier
            for tier, tokens in (
                (
                    "live_l1"
                    if live_primary_tokens >= backup_l1_tokens
                    and live_primary_tokens > 0
                    else "l1",
                    l1_tokens,
                ),
                ("l2", l2_tokens),
                ("l3", l3_tokens),
            )
            if tokens > 0
        ]
        if protected_tiers and protected_tiers[0] == "live_l1":
            action = (
                "live_l1_reuse"
                if len(protected_tiers) == 1
                else f"{'_'.join(protected_tiers)}_backup"
            )
        else:
            action = (
                f"{'_'.join(protected_tiers)}_backup"
                if protected_tiers
                else "recompute"
            )
        return {
            "l1_tokens": l1_tokens,
            "l2_tokens": l2_tokens,
            "l3_tokens": l3_tokens,
            "live_primary_tokens": live_primary_tokens,
            "restored_tokens": restored,
            "recompute_tokens": recompute_tokens,
            "recovery_ms": recovery_ms,
            "action": action,
            "gpu_direct_reuse": l1_tokens > 0,
            "host_restore_path": host_restore_path,
            "l3_restore_path": l3_restore_path,
        }

    def _choose_l1_l2_retry_rank(
        self, req: TokenizedGenerateReqInput, sim: dict
    ) -> int:
        healthy = self._healthy_ranks()
        if not healthy:
            return -1
        backup_key = self._backup_key(req, sim)
        hint, resolved_backup_key, lineage_chain = (
            self._resolve_lineage_recovery_hint(backup_key)
        )
        requested = max(
            self._sim_int(sim, "pre_failure_total_tokens", self._requested_tokens(req)),
            0,
        )
        target = int(hint.get("target_dp_rank", -1)) if hint else -1
        failed_dp_rank = self._sim_int(sim, "failed_dp_rank", -1)
        cache_key = self._cache_key(req, sim)
        options = {
            rank: self._aegis_retry_option(
                hint=hint,
                retry_dp_rank=rank,
                requested_tokens=requested,
                failed_dp_rank=failed_dp_rank,
                cache_key=cache_key,
            )
            for rank in healthy
        }

        def score(rank: int) -> tuple[float, int]:
            wait_ms = 0.002 * self._aegis_gpu_ledger.inflight_tokens(rank)
            return wait_ms + float(options[rank]["recovery_ms"]), rank

        route_plan_retry_rank = self._sim_int(
            sim, "aegis_route_plan_retry_dp_rank", -1
        )
        route_plan_retry_applied = route_plan_retry_rank in healthy
        retry_rank = (
            route_plan_retry_rank
            if route_plan_retry_applied
            else min(healthy, key=score)
        )
        option = options[retry_rank]
        l1_tokens = int(option["l1_tokens"])
        l2_tokens = int(option["l2_tokens"])
        l3_tokens = int(option["l3_tokens"])
        restored = int(option["restored_tokens"])
        action = str(option["action"])
        overhead_ms = float(option["recovery_ms"])
        live_primary_tokens = int(option.get("live_primary_tokens", 0))
        use_hint = restored > live_primary_tokens
        sim["aegis_retry_used_hint"] = use_hint
        sim["aegis_retry_live_prefix_tokens"] = live_primary_tokens
        sim["aegis_retry_resolved_backup_key"] = resolved_backup_key
        sim["aegis_retry_lineage_chain"] = list(lineage_chain)
        sim["aegis_retry_backup_dp_rank"] = target
        sim["aegis_retry_score_ms"] = round(score(retry_rank)[0], 3)
        if route_plan_retry_rank >= 0:
            sim["aegis_route_plan_retry_dp_rank"] = route_plan_retry_rank
            sim["aegis_route_plan_retry_applied"] = route_plan_retry_applied
        sim["restore_tokens"] = restored
        sim["restore_recompute_tokens"] = max(requested - restored, 0)
        sim["restore_action"] = action
        sim["restore_data_plane"] = (
            "metadata_materialization" if restored > 0 else "recompute"
        )
        sim["restore_timing_source"] = "aegiskv_tiered_hint"
        sim["restore_overhead_s"] = max(overhead_ms, 0.0) / 1000.0
        sim["host_backup_lookup_hit"] = l2_tokens > 0
        sim["host_backup_lookup_tokens"] = l2_tokens
        sim["l3_backup_lookup_hit"] = l3_tokens > 0
        sim["l3_backup_lookup_tokens"] = l3_tokens
        sim["peer_gpu_lookup_hit"] = l1_tokens > 0
        sim["peer_gpu_lookup_tokens"] = l1_tokens
        sim["aegis_retry_gpu_direct_reuse"] = bool(
            option["gpu_direct_reuse"]
        )
        sim["aegis_retry_host_restore_path"] = list(
            option["host_restore_path"]
        )
        sim["aegis_retry_l3_restore_path"] = list(
            option["l3_restore_path"]
        )
        per_stage = option.get("per_stage", [])
        sim["aegis_retry_per_stage"] = [
            dict(row) for row in per_stage if isinstance(row, dict)
        ]
        sim["aegis_retry_restored_tokens_by_stage"] = [
            int(row.get("restored_tokens", 0))
            for row in per_stage
            if isinstance(row, dict)
        ]
        sim["aegis_retry_recomputed_tokens_by_stage"] = [
            int(row.get("recomputed_tokens", 0))
            for row in per_stage
            if isinstance(row, dict)
        ]
        sim["aegis_retry_bottleneck_pp_rank"] = int(
            option.get("bottleneck_pp_rank", -1)
        )
        sim["assigned_dp_rank"] = retry_rank
        self._append_aegis_event(
            "recovery_reroute",
            {
                "rid": req.rid,
                "backup_key": backup_key,
                "resolved_backup_key": resolved_backup_key,
                "lineage_chain": list(lineage_chain),
                "failed_dp_rank": failed_dp_rank,
                "retry_dp_rank": retry_rank,
                "hint_dp_rank": target,
                "used_hint": use_hint,
                "action": action,
                "restored_tokens": restored,
                "live_primary_tokens": int(
                    option.get("live_primary_tokens", 0)
                ),
                "recompute_tokens": max(requested - restored, 0),
                "gpu_direct_reuse": bool(option["gpu_direct_reuse"]),
                "host_restore_path": list(option["host_restore_path"]),
                "l3_restore_path": list(option["l3_restore_path"]),
                "l3_reuse_tokens": l3_tokens,
                "recovery_ms": round(overhead_ms, 6),
                "route_plan_retry_dp_rank": route_plan_retry_rank,
                "route_plan_retry_applied": route_plan_retry_applied,
                "per_stage": [
                    dict(row) for row in per_stage if isinstance(row, dict)
                ],
                "bottleneck_pp_rank": int(
                    option.get("bottleneck_pp_rank", -1)
                ),
            },
        )
        return retry_rank


def launch_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    dp_rank: int,
    *,
    pp_rank: int = 0,
    tp_rank: int = 0,
    stats_queue: mp.Queue | None = None,
) -> tuple[mp.Process, mp.Queue, mp.Event]:
    from sglsim.srt.managers.scheduler import run_scheduler_process

    cmd_queue: mp.Queue = mp.Queue()
    ready_event = mp.Event()
    if stats_queue is None:
        stats_queue = mp.Queue()
    proc = mp.Process(
        target=run_scheduler_process,
        args=(
            server_args,
            port_args,
            dp_rank,
            ipc_registry,
            cmd_queue,
            stats_queue,
            ready_event,
            pp_rank,
            tp_rank,
        ),
        daemon=True,
    )
    proc.start()
    return proc, cmd_queue, ready_event


def launch_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    dp_size: int,
) -> tuple[list[mp.Process], list[mp.Queue], mp.Queue, list[mp.Event]]:
    processes: list[mp.Process] = []
    cmd_queues: list[mp.Queue] = []
    ready_events: list[mp.Event] = []
    stats_queue: mp.Queue = mp.Queue()

    for dp_rank in range(dp_size):
        proc, cmd_queue, ready_event = launch_scheduler_process(
            server_args,
            port_args,
            ipc_registry,
            dp_rank,
            stats_queue=stats_queue,
        )
        processes.append(proc)
        cmd_queues.append(cmd_queue)
        ready_events.append(ready_event)

    for event in ready_events:
        if not event.wait(timeout=60):
            raise RuntimeError("DP scheduler subprocess failed to become ready")

    return processes, cmd_queues, stats_queue, ready_events


def launch_pipeline_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    *,
    dp_size: int,
    pp_size: int,
    tp_size: int = 1,
    ready_timeout_s: float = 60.0,
) -> tuple[PipelineProcessGroup, mp.Queue]:
    """Launch one Scheduler process per local DP x PP x TP stage.

    This is the parent-process analogue of pinned SGLang
    ``DataParallelController.launch_tensor_parallel_group``.  The online Engine
    switches to it only together with the PP event loop, so process topology
    cannot get ahead of batch-lifecycle correctness.
    """

    if int(server_args.pp_size) != int(pp_size):
        raise ValueError(
            f"server_args.pp_size={server_args.pp_size} != pp_size={pp_size}"
        )
    if int(server_args.tp_size) != int(tp_size):
        raise ValueError(
            f"server_args.tp_size={server_args.tp_size} != tp_size={tp_size}"
        )
    stats_queue: mp.Queue = mp.Queue()
    handles: list[PipelineWorkerHandle] = []
    try:
        for spec in enumerate_pipeline_workers(
            dp_size=dp_size,
            pp_size=pp_size,
            tp_size=tp_size,
        ):
            proc, cmd_queue, ready_event = launch_scheduler_process(
                server_args,
                port_args,
                ipc_registry,
                spec.stage.dp_rank,
                pp_rank=spec.stage.pp_rank,
                tp_rank=spec.stage.tp_rank,
                stats_queue=stats_queue,
            )
            handles.append(
                PipelineWorkerHandle(
                    spec=spec,
                    process=proc,
                    command_queue=cmd_queue,
                    ready_event=ready_event,
                )
            )
        group = PipelineProcessGroup(
            dp_size=dp_size,
            pp_size=pp_size,
            tp_size=tp_size,
            workers=tuple(handles),
        )
        for worker in group.workers:
            if not worker.ready_event.wait(timeout=ready_timeout_s):
                raise RuntimeError(
                    "PP Scheduler subprocess failed to become ready: "
                    f"stage={worker.spec.stage}"
                )
        return group, stats_queue
    except Exception:
        for handle in handles:
            if handle.process.is_alive():
                handle.command_queue.put(("stop",))
        for handle in handles:
            handle.process.join(timeout=1.0)
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=1.0)
        raise


def launch_pipeline_dp_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    *,
    dp_size: int,
    dp_rank: int,
    pp_size: int,
    tp_size: int = 1,
    stats_queue: mp.Queue,
    ready_timeout_s: float = 60.0,
) -> tuple[PipelineWorkerHandle, ...]:
    """Restart every PP/TP worker belonging to one DP replica.

    Pinned SGLang treats the local PP group as one DP worker lifecycle unit.
    This helper preserves the original global worker specifications so the
    restarted handles can be atomically spliced into ``PipelineProcessGroup``.
    """

    if not 0 <= int(dp_rank) < int(dp_size):
        raise ValueError(f"dp_rank must be in [0, {dp_size}), got {dp_rank}")
    if int(server_args.pp_size) != int(pp_size):
        raise ValueError(
            f"server_args.pp_size={server_args.pp_size} != pp_size={pp_size}"
        )
    if int(server_args.tp_size) != int(tp_size):
        raise ValueError(
            f"server_args.tp_size={server_args.tp_size} != tp_size={tp_size}"
        )

    specs = tuple(
        spec
        for spec in enumerate_pipeline_workers(
            dp_size=dp_size,
            pp_size=pp_size,
            tp_size=tp_size,
        )
        if spec.stage.dp_rank == int(dp_rank)
    )
    handles: list[PipelineWorkerHandle] = []
    try:
        for spec in specs:
            proc, cmd_queue, ready_event = launch_scheduler_process(
                server_args,
                port_args,
                ipc_registry,
                spec.stage.dp_rank,
                pp_rank=spec.stage.pp_rank,
                tp_rank=spec.stage.tp_rank,
                stats_queue=stats_queue,
            )
            handles.append(
                PipelineWorkerHandle(
                    spec=spec,
                    process=proc,
                    command_queue=cmd_queue,
                    ready_event=ready_event,
                )
            )
        for worker in handles:
            if not worker.ready_event.wait(timeout=ready_timeout_s):
                raise RuntimeError(
                    "PP Scheduler subprocess failed to restart: "
                    f"stage={worker.spec.stage}"
                )
        return tuple(handles)
    except Exception:
        for handle in handles:
            if handle.process.is_alive():
                handle.command_queue.put(("stop",))
        for handle in handles:
            handle.process.join(timeout=1.0)
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=1.0)
        raise


def launch_dp_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    dp_size: int,
) -> tuple[list[mp.Process], list[mp.Queue], mp.Queue, list[mp.Event]]:
    return launch_scheduler_processes(
        server_args, port_args, ipc_registry, dp_size
    )
