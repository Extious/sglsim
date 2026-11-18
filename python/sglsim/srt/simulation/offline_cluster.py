"""Parent-side event-driven multi-DP offline scheduler cluster."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import queue
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

from sglsim.srt.managers.data_parallel_controller import DataParallelController
from sglsim.srt.managers.aegis_l1_l2 import (
    resolve_aegis_per_node_dram_quota,
)
from sglsim.srt.managers.detokenizer_manager import DetokenizerManager, _handle_message
from sglsim.srt.managers.io_struct import TokenizedGenerateReqInput
from sglsim.srt.managers.pipeline_parallel import (
    PPOutputEnvelope,
    PPProxyEnvelope,
    PPRequestEnvelope,
    PipelineProcessGroup,
    PipelineWorkerHandle,
    StageId,
    enumerate_pipeline_workers,
)
from sglsim.srt.managers.tokenizer_manager import resolve_dp_size
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.aegis_exact_contract import CertifiedExactWindowPlan
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.offline_aegis_transfer_bridge import (
    OfflineAegisTransferBridge,
    TierNotification,
)
from sglsim.srt.simulation.offline_coordinator import (
    OfflineEvent,
    OfflineEventCoordinator,
    OfflineEventPriority,
)
from sglsim.srt.simulation.offline_hicache_transfer_bridge import (
    OfflineHiCacheTransferBridge,
    OrdinaryHiCacheDirection,
    OrdinaryHiCacheTransferState,
)
from sglsim.srt.simulation.global_des import (
    CommitGroupState,
    EventPriority,
    GlobalDES,
    KvCopyState,
    StorageTier,
    TransferState,
)
from sglsim.srt.simulation.offline_resource_bridge import (
    OfflinePhysicalTopology,
    OfflineResourceBridge,
)
from sglsim.srt.simulation.offline_router_ledger import OfflineRouterLedger
from sglsim.srt.simulation.offline_worker import run_offline_scheduler_worker
from sglsim.srt.utils.network import (
    SimIpcRegistry,
    SimZmqContext,
    SimZmqSocket,
    build_dp_ipc_registry,
    build_dp_pp_ipc_registry,
    get_zmq_socket,
)


class _ListSender:
    """Minimal synchronous sink used by the parent detokenizer bridge."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def send_pyobj(self, message: Any) -> None:
        if isinstance(message, dict):
            self.messages.append(dict(message))


class _OfflineRequestSender:
    """Capture one controller-selected request without feeder-thread races."""

    def __init__(self, inbox: list[TokenizedGenerateReqInput]) -> None:
        self._inbox = inbox

    def send_pyobj(self, message: Any) -> None:
        if not isinstance(message, TokenizedGenerateReqInput):
            raise TypeError("offline scheduler input must be tokenized")
        self._inbox.append(message)


class _OfflineCommandSender:
    """Capture controller-to-rank commands for the next causal rank poll."""

    def __init__(self, inbox: list[Any]) -> None:
        self._inbox = inbox

    def put(self, message: Any) -> None:
        self._inbox.append(message)


@dataclass(frozen=True)
class OfflineClusterResult:
    telemetry: list[dict[str, Any]]
    detokenizer_messages: list[dict[str, Any]]
    rank_snapshots: dict[int, dict[str, Any]]
    failure_events: list[dict[str, Any]]
    exact_rolling_installations: list[dict[str, Any]]
    aegis_status: dict[str, Any]
    global_des_snapshot: dict[str, Any] = field(default_factory=dict)
    global_des_timeseries: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    physical_resource_bridge: dict[str, Any] = field(default_factory=dict)
    physical_aegis_transfers: dict[str, Any] = field(default_factory=dict)
    physical_hicache_transfers: dict[str, Any] = field(default_factory=dict)
    resource_audit_errors: list[str] = field(default_factory=list)
    router_output_ledger: dict[str, Any] = field(default_factory=dict)


class OfflineSchedulerCluster:
    """Drive real SGLSim schedulers with a parent DES instead of wall sleeps."""

    def __init__(
        self,
        server_args: ServerArgs,
        *,
        worker_timeout_s: float = 30.0,
        global_des: GlobalDES | None = None,
        physical_topology: OfflinePhysicalTopology | None = None,
    ) -> None:
        if server_args.sim_config_path is None:
            raise ValueError("offline cluster requires an explicit sim_config_path")
        if server_args.simulate_detokenizer_time:
            raise ValueError(
                "offline cluster does not yet support synchronous real-tokenizer ACKs"
            )
        ConfigManager.load(server_args.sim_config_path)
        ConfigManager.apply_server_args_overrides(server_args)
        resolved_scheduler = ConfigManager.get_scheduler_config()
        server_args.dp_size = int(resolved_scheduler.dp_size)
        server_args.tp_size = int(resolved_scheduler.tp_size)
        configured_pp_size = int(resolved_scheduler.pp_size)
        # Prefer the explicit ServerArgs flag; also honor sim-config smoke key.
        raw_cfg = ConfigManager._raw if hasattr(ConfigManager, "_raw") else {}
        aegis_raw = (raw_cfg or {}).get("aegis_l1_l2", {})
        if not isinstance(aegis_raw, dict):
            aegis_raw = {}
        cache_cfg = ConfigManager.get_cache_config(server_args.model_path)
        legacy_dram_tokens = max(
            int(cache_cfg.l2_capacity_pages) * int(cache_cfg.page_size), 1
        )
        self.aegis_per_node_dram_quota = resolve_aegis_per_node_dram_quota(
            aegis_raw,
            fallback_tokens=legacy_dram_tokens,
        )
        if (
            self.aegis_per_node_dram_quota.explicit
            and bool(aegis_raw.get("enabled", False))
            and str(aegis_raw.get("design_semantics") or "") == DESIGN_SEMANTICS
        ):
            if physical_topology is None:
                raise ValueError(
                    "explicit per-node Aegis DRAM quota requires physical topology"
                )
            quota_bytes = self.aegis_per_node_dram_quota.quota_bytes
            for node in physical_topology.nodes:
                if node.capacity_bytes - node.runtime_bytes < quota_bytes:
                    raise ValueError(
                        f"node {node.node_id} cannot bind configured Aegis DRAM quota"
                    )
        logical_pp = bool(getattr(server_args, "logical_pp", False)) or bool(
            (raw_cfg or {}).get("logical_pp", False)
        )
        server_args.logical_pp = logical_pp
        if logical_pp:
            # One Scheduler process per DP.  Keep configured_pp_size only as
            # metadata for artifacts; worker IPC uses process_pp_size=1.
            process_pp_size = 1
            server_args.pp_size = 1
        else:
            process_pp_size = configured_pp_size
            server_args.pp_size = configured_pp_size
            if server_args.pp_size > 1:
                server_args.disable_overlap_schedule = True
        if process_pp_size > 1 and server_args.tp_size != 1:
            raise NotImplementedError(
                "physical offline PP currently requires tp_size=1, matching "
                "the online PipelineProcessGroup contract"
            )
        self.server_args = server_args
        self.port_args = PortArgs.init_new(server_args)
        self.dp_size = resolve_dp_size(server_args)
        self.configured_pp_size = configured_pp_size
        self.logical_pp = logical_pp
        self.pp_size = int(process_pp_size)
        self.tp_size = int(server_args.tp_size)
        # SGLSim mirrors TP metadata/latency in one Scheduler process.  Physical
        # process expansion in this Goal is DP x PP; PP>1 currently requires
        # TP=1 in both online and offline runtimes.
        self.physical_tp_size = 1
        self.worker_timeout_s = max(float(worker_timeout_s), 1.0)
        self.coordinator = OfflineEventCoordinator(
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            global_des=global_des,
        )
        self.global_des = self.coordinator.global_des
        self.resource_bridge = (
            None
            if physical_topology is None
            else OfflineResourceBridge(self.global_des, physical_topology)
        )
        # Ordinary SGLang HiCache retains allocator ownership of L1/L2 bytes;
        # this adapter owns only the canonical physical DMA/ACK lifecycle.  A
        # topology is mandatory because the worker emits one symmetric TP view
        # that must be expanded onto every physical (pp,tp) shard.
        self.hicache_transfer_bridge = (
            None
            if physical_topology is None
            else OfflineHiCacheTransferBridge(
                self.global_des,
                physical_topology,
                on_group_terminal=self._on_hicache_group_terminal,
            )
        )
        self._aegis_raw_config = dict(aegis_raw)
        physical_aegis_enabled = bool(
            physical_topology is not None
            and bool(aegis_raw.get("enabled", False))
            and bool(aegis_raw.get("committed_copy_protocol", False))
            and str(aegis_raw.get("design_semantics") or "")
            == DESIGN_SEMANTICS
        )
        self.aegis_transfer_bridge = (
            OfflineAegisTransferBridge(self.global_des, physical_topology)
            if physical_aegis_enabled and physical_topology is not None
            else None
        )
        self.ipc_registry = (
            build_dp_pp_ipc_registry(
                self.port_args,
                self.dp_size,
                self.pp_size,
            )
            if self.pp_size > 1
            else build_dp_ipc_registry(self.port_args, self.dp_size)
        )
        SimIpcRegistry.install_global(self.ipc_registry)

        self._worker_specs = enumerate_pipeline_workers(
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            tp_size=self.physical_tp_size,
        )
        self._stage_keys = tuple(
            (
                spec.stage.dp_rank,
                spec.stage.pp_rank,
                spec.stage.tp_rank,
            )
            for spec in self._worker_specs
        )
        self._worker_incarnations = {key: 0 for key in self._stage_keys}
        self.control_queues = {key: mp.Queue() for key in self._stage_keys}
        self.result_queue = mp.Queue()
        self.stats_queue = mp.Queue()
        self.processes: list[mp.Process] = []
        ready_events: dict[tuple[int, int, int], Any] = {}
        process_by_stage: dict[tuple[int, int, int], mp.Process] = {}
        for spec in self._worker_specs:
            stage = spec.stage
            key = (stage.dp_rank, stage.pp_rank, stage.tp_rank)
            ready = mp.Event()
            process = mp.Process(
                target=run_offline_scheduler_worker,
                args=(
                    server_args,
                    self.port_args,
                    stage.dp_rank,
                    stage.pp_rank,
                    stage.tp_rank,
                    self.ipc_registry,
                    None,
                    self.control_queues[key],
                    self.result_queue,
                    self.stats_queue,
                    ready,
                    self.hicache_transfer_bridge is not None,
                    self._worker_incarnations[key],
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            process_by_stage[key] = process
            ready_events[key] = ready
        for key, ready in ready_events.items():
            if not ready.wait(timeout=self.worker_timeout_s):
                self.close()
                raise RuntimeError(
                    f"offline scheduler worker {key} failed to become ready"
                )
        self._process_by_stage = dict(process_by_stage)
        self._ready_event_by_stage = dict(ready_events)
        self._retired_control_queues: list[Any] = []

        self._runtime_command_inboxes = {key: [] for key in self._stage_keys}
        self.pipeline_process_group = PipelineProcessGroup(
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            tp_size=self.physical_tp_size,
            workers=tuple(
                PipelineWorkerHandle(
                    spec=spec,
                    process=process_by_stage[
                        (
                            spec.stage.dp_rank,
                            spec.stage.pp_rank,
                            spec.stage.tp_rank,
                        )
                    ],
                    command_queue=_OfflineCommandSender(
                        self._runtime_command_inboxes[
                            (
                                spec.stage.dp_rank,
                                spec.stage.pp_rank,
                                spec.stage.tp_rank,
                            )
                        ]
                    ),
                    ready_event=ready_events[
                        (
                            spec.stage.dp_rank,
                            spec.stage.pp_rank,
                            spec.stage.tp_rank,
                        )
                    ],
                )
                for spec in self._worker_specs
            ),
        )
        leader_processes = [
            self.pipeline_process_group.leader_for_dp(rank).process
            for rank in range(self.dp_size)
        ]
        leader_command_queues = [
            self.pipeline_process_group.leader_for_dp(rank).command_queue
            for rank in range(self.dp_size)
        ]
        self.dp_controller = DataParallelController(
            server_args,
            self.port_args,
            scheduler_procs=leader_processes,
            scheduler_cmd_queues=leader_command_queues,
            pipeline_process_group=(
                self.pipeline_process_group if self.pp_size > 1 else None
            ),
            now_fn=lambda: self.coordinator.now_s,
        )
        # Preserve the real controller's routing, placement, ledger, and ACK
        # side effects, but make its selected request handoff synchronous.  A
        # wall-clock scheduler polls its IPC queue repeatedly; a DES must not
        # advance to a future failure merely because an mp.Queue feeder has not
        # published a just-dispatched request yet.
        self._request_inboxes: list[list[TokenizedGenerateReqInput]] = [
            [] for _ in range(self.dp_size)
        ]
        self.dp_controller.workers = [
            _OfflineRequestSender(self._request_inboxes[rank])
            for rank in range(self.dp_size)
        ]
        self._pp_request_inboxes = {key: [] for key in self._stage_keys}
        self._pp_proxy_inboxes = {key: [] for key in self._stage_keys}
        self._pp_output_inboxes = {key: [] for key in self._stage_keys}
        self._stage_idle = {key: True for key in self._stage_keys}
        self._stage_blocked = {key: False for key in self._stage_keys}
        self._stage_buffered_work = {key: False for key in self._stage_keys}
        self._stage_next_sequence = {key: 0 for key in self._stage_keys}
        self._stage_owned_request_ids = {
            key: set() for key in self._stage_keys
        }
        self._terminal_request_stats: dict[str, dict[str, Any]] = {}
        self.detokenizer = DetokenizerManager(self.port_args)
        self._detok_reply_by_dp = {
            dp_rank: get_zmq_socket(
                SimZmqContext(),
                SimZmqSocket.PUSH,
                self.port_args.detokenizer_reply_ipc_name_for_dp(dp_rank),
                bind=False,
            )
            for dp_rank in range(self.dp_size)
        }
        self.recv_detokenizer_output = get_zmq_socket(
            SimZmqContext(),
            SimZmqSocket.PULL,
            self.port_args.tokenizer_ipc_name,
            bind=False,
        )
        self._request_sequence = 0
        self._pending_responses: dict[
            tuple[int, int, int, int], dict[str, Any]
        ] = {}
        self._closed = False
        self.telemetry: list[dict[str, Any]] = []
        self.detokenizer_messages: list[dict[str, Any]] = []
        self.failure_events: list[dict[str, Any]] = []
        self._scheduled_failure_ids: set[str] = set()
        self._completed_failure_ids: set[str] = set()
        self._active_failures: dict[str, dict[str, Any]] = {}
        self._pending_recoveries: dict[str, dict[str, Any]] = {}
        self._hard_failed_dp_ranks: set[int] = set()
        self._request_catalog: dict[str, TokenizedGenerateReqInput] = {}
        self.router_output_ledger = OfflineRouterLedger(
            model_revision=str(
                aegis_raw.get("model_revision") or server_args.model_path
            ),
            kv_abi=str(aegis_raw.get("kv_abi") or "sglsim-metadata-kv-v1"),
            parallel_layout=str(
                aegis_raw.get("parallel_layout")
                or f"dp{self.dp_size}-pp{self.configured_pp_size}-tp{self.tp_size}"
            ),
        )
        self._causal_exact_plan_provider: Callable[
            [TokenizedGenerateReqInput, "OfflineSchedulerCluster"],
            CertifiedExactWindowPlan | None,
        ] | None = None
        self.exact_rolling_installations: list[dict[str, Any]] = []
        self._latest_kv_resource_snapshots: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._physical_aegis_controller_dispatches: list[dict[str, Any]] = []
        self._deferred_physical_aegis_commits: dict[
            str, tuple[dict[str, Any], TierNotification]
        ] = {}
        self._physical_aegis_replay_records: dict[
            str, tuple[dict[str, Any], TierNotification]
        ] = {}
        self._physical_aegis_recovery_plans: list[dict[str, Any]] = []
        self._physical_recovery_admissions: dict[str, dict[str, Any]] = {}
        self._physical_recovery_admission_events: list[dict[str, Any]] = []
        self._pending_physical_aegis_completions: dict[
            tuple[str, int], dict[str, Any]
        ] = {}
        self._physical_aegis_completion_events: list[dict[str, Any]] = []
        self._physical_aegis_source_frontier_adjustments: list[
            dict[str, Any]
        ] = []
        self._hicache_ack_inboxes: dict[
            tuple[int, int, int], list[dict[str, Any]]
        ] = {key: [] for key in self._stage_keys}
        self._hicache_intent_stage: dict[str, tuple[int, int, int]] = {}
        self._hicache_blocked_prepared: dict[
            tuple[int, int, int], dict[str, Any]
        ] = {}
        self._hicache_failure_teardown_stages: set[
            tuple[int, int, int]
        ] = set()
        self._hicache_artifact_blockers: list[str] = []

    def _physical_recovery_gate_evidence(
        self, request_id: str
    ) -> dict[str, Any] | None:
        record = self._physical_recovery_admissions.get(str(request_id))
        if record is None:
            return None
        state = str(record.get("state") or "")
        evidence = self._physical_recovery_admission_row(record)
        group_id = str(record.get("group_id") or "")
        group = self.global_des.recovery_load_groups.get(group_id)
        evidence["canonical_group"] = (
            None if group is None else group.snapshot()
        )
        if state == "ADOPTED":
            if (
                group is None
                or group.state is not CommitGroupState.COMMITTED
                or group.adopted_shards != set(group.expected_shard_keys)
            ):
                raise RuntimeError(
                    f"detokenizer fence saw incomplete recovery group for {request_id}"
                )
        elif state == "FALLBACK_RECOMPUTE":
            if not str(record.get("fallback_reason") or "").strip():
                raise RuntimeError(
                    f"detokenizer fallback fence has no reason for {request_id}"
                )
            active_holds: list[str] = []
            if group is None and group_id:
                orphan_transfers = sorted(
                    transfer.transfer_id
                    for transfer in self.global_des.transfers.values()
                    if str(transfer.recovery_group_id or "") == group_id
                )
                if orphan_transfers:
                    raise RuntimeError(
                        "detokenizer fallback references a missing recovery "
                        "group with orphan transfers: "
                        + ", ".join(orphan_transfers)
                    )
            if group is not None:
                if not group.discarded:
                    raise RuntimeError(
                        f"detokenizer fallback retained recovery group {group_id}"
                    )
                if group.adopted_shards:
                    raise RuntimeError(
                        f"detokenizer fallback group {group_id} has adopted shards"
                    )
                canonical_group = group.snapshot()
                if not str(
                    canonical_group.get("terminal_reason") or ""
                ).strip():
                    raise RuntimeError(
                        f"detokenizer fallback group {group_id} has no terminal reason"
                    )
                if canonical_group.get("discard_scope") != "all":
                    raise RuntimeError(
                        f"detokenizer fallback group {group_id} has invalid discard scope"
                    )
                active_holds = sorted(
                    transfer_id
                    for transfer_id in group.transfer_ids
                    if self.global_des.transfers[
                        transfer_id
                    ].recovery_capacity_hold_active
                )
                if active_holds:
                    raise RuntimeError(
                        "detokenizer fallback retained RECOVERY_PENDING holds: "
                        + ", ".join(active_holds)
                    )
                evidence["fallback_discard_scope"] = canonical_group[
                    "discard_scope"
                ]
                evidence["fallback_abandoned_shards"] = canonical_group[
                    "abandoned_shards"
                ]
                evidence["fallback_terminal_reason"] = canonical_group[
                    "terminal_reason"
                ]
            evidence["active_recovery_capacity_hold_transfer_ids"] = active_holds
        else:
            raise RuntimeError(
                f"detokenizer output crossed physical recovery gate in state {state}"
            )
        return evidence

    def _ingest_worker_detokenizer_messages(self, response: dict[str, Any]) -> int:
        """Synchronously publish all scheduler messages caused by one event."""

        sender = _ListSender()
        for message in response.get("detokenizer_messages", ()):
            if not isinstance(message, dict):
                continue
            annotated = dict(message)
            rid = str(annotated.get("rid") or "")
            request = self._request_catalog.get(rid)
            if request is None:
                raise RuntimeError(
                    f"detokenizer output has no Router request catalog entry: {rid!r}"
                )
            # For a recovery execution, the first PP0 output is itself the
            # existing Scheduler pipeline's causal all-stage boundary.  TP is
            # one logical aggregate in SGLSim.  The ledger remains fail-closed
            # until this evidence activates the route.
            gate_evidence = self._physical_recovery_gate_evidence(rid)
            self.router_output_ledger.acknowledge_scheduler_boundary(
                rid,
                response_dp_rank=int(response.get("dp_rank", -1)),
                response_pp_rank=int(response.get("pp_rank", -1)),
                response_tp_rank=int(response.get("tp_rank", 0)),
                physical_recovery_gate=gate_evidence,
            )
            annotated.update(
                self.router_output_ledger.observe_scheduler_message(annotated)
            )
            _handle_message(
                annotated,
                model_path=self.server_args.model_path,
                send_to_tokenizer=sender,
                send_to_scheduler_by_dp=self._detok_reply_by_dp,
                simulate_detokenizer_time=False,
            )
        self.detokenizer_messages.extend(sender.messages)
        for message in sender.messages:
            if not bool(message.get("finished", False)):
                continue
            rid = str(message.get("rid") or "")
            for owned in self._stage_owned_request_ids.values():
                owned.discard(rid)
        if self.aegis_transfer_bridge is not None:
            for message in sender.messages:
                if not bool(message.get("finished", False)):
                    continue
                request = self._request_catalog.get(str(message.get("rid") or ""))
                if request is None:
                    continue
                sim = request.sampling_params.custom_params.get("simulation", {})
                if not isinstance(sim, dict):
                    continue
                failover_replay = bool(sim.get("failover_replay", False))
                backup_enabled = bool(sim.get("backup_kv", False))
                if not (failover_replay or backup_enabled):
                    continue
                backup_key = str(
                    sim.get("aegis_backup_key")
                    or sim.get("aegis_cache_key")
                    or (request.rid if backup_enabled else "")
                )
                hint = self.dp_controller.recovery_hint(backup_key)
                if not backup_key or not isinstance(hint, dict):
                    continue
                completion_key = (
                    backup_key,
                    max(int(hint.get("version", 1)), 1),
                )
                waiter = {
                    "request_backup_key": backup_key,
                    "version": completion_key[1],
                    "route_version": int(hint.get("route_version", 1)),
                    "source_dp_rank": int(hint.get("source_dp_rank", -1)),
                    "target_dp_rank": int(hint.get("target_dp_rank", -1)),
                    "request_finished": True,
                    "failover_replay_finished": failover_replay,
                    "registered_at_ns": self.global_des.now_ns,
                }
                self._pending_physical_aegis_completions.setdefault(
                    completion_key, waiter
                )
                self._maybe_complete_physical_aegis_request(*completion_key)
        return len(sender.messages)

    def _physical_aegis_identity(
        self,
        event: dict[str, Any],
        hint: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Resolve the copy identity from causal metadata and topology."""

        source_dp_rank = int(event.get("source_dp_rank", -1))
        source_pp_rank = int(event.get("source_pp_rank", 0))
        topology = self.resource_bridge.topology
        physical_tp_size = len(topology.group(source_dp_rank, source_pp_rank))

        def first_text(name: str, fallback: str) -> str:
            for source in (event, hint, self._aegis_raw_config):
                value = str(source.get(name) or "")
                if value:
                    return value
            return fallback

        return (
            first_text("model_revision", str(self.server_args.model_path)),
            first_text("kv_abi", "sglsim-metadata-kv-v1"),
            first_text(
                "parallel_layout",
                f"dp{self.dp_size}-pp{self.pp_size}-tp{physical_tp_size}",
            ),
        )

    def _physical_aegis_kv_bytes(
        self,
        event: dict[str, Any],
    ) -> dict[int, int]:
        source_dp_rank = int(event.get("source_dp_rank", -1))
        source_pp_rank = int(event.get("source_pp_rank", 0))
        snapshot = self._latest_kv_resource_snapshots.get(
            (source_dp_rank, source_pp_rank)
        )
        if not isinstance(snapshot, dict):
            raise RuntimeError(
                "physical Aegis source frontier has no Scheduler KV snapshot"
            )
        bytes_per_token = int(
            snapshot.get("kv_bytes_per_token_per_tp_rank", 0)
        )
        if bytes_per_token <= 0:
            raise ValueError(
                "physical Aegis source snapshot has invalid KV bytes/token"
            )
        group = self.resource_bridge.topology.group(
            source_dp_rank, source_pp_rank
        )
        logical_tp_size = int(snapshot.get("logical_tp_group_size", -1))
        if logical_tp_size != len(group):
            raise ValueError(
                "Scheduler KV snapshot TP size differs from physical topology"
            )
        return {spec.gpu.tp_rank: bytes_per_token for spec in group}

    def _promote_physical_source_frontier_owner(
        self,
        event: dict[str, Any],
    ) -> None:
        """Keep the just-produced request range in the source primary account.

        Under overlap scheduling, ``process_batch_result`` (and therefore the
        source-frontier event) can run inside the next ``prepare`` operation.
        By the after-prepare snapshot a terminal request is already absent from
        ``running_batch`` even though its owner pages are exactly the source of
        the newly emitted copy.  Re-project that one causal owner as primary at
        this event boundary; a later ordinary snapshot may classify the
        retained cache entry as prefix after all copy consumers finish.
        """

        source_dp_rank = int(event.get("source_dp_rank", -1))
        source_pp_rank = int(event.get("source_pp_rank", 0))
        key = (source_dp_rank, source_pp_rank)
        snapshot = self._latest_kv_resource_snapshots.get(key)
        rid = str(event.get("rid") or "")
        if not isinstance(snapshot, dict) or not rid:
            return
        owner_pages = (snapshot.get("l1_device") or {}).get("owner_pages", {})
        if rid not in owner_pages:
            return
        active = set(snapshot.get("running_request_ids", ()))
        if rid in active:
            return
        promoted = dict(snapshot)
        active.add(rid)
        promoted["running_request_ids"] = sorted(active)
        self.resource_bridge.consume_snapshot(
            promoted,
            phase="aegis_source_frontier",
            event_metadata={
                "request_backup_key": str(
                    event.get("request_backup_key") or ""
                ),
                "source_committed_token_end": int(
                    event.get("source_committed_token_end", 0)
                ),
            },
        )
        self._latest_kv_resource_snapshots[key] = promoted

    def _bound_physical_source_frontier_to_allocator(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        """Drop a source tail absent from the post-response allocator.

        ``Scheduler.process_batch_result`` emits the causal Aegis event before
        the worker returns its allocator snapshot.  Page-granular prefill and
        finished-request cache paths can retain fewer pages than the logical
        token frontier, including while the last output token is still being
        finalized.  A physical DMA may alias the request-owned pages visible
        in that same response (plus any earlier frontier already copied), but
        it must not borrow unrelated aggregate HBM bytes to fabricate a tail.

        The original Scheduler observation remains in
        ``source_observed_token_end``.  Recovery naturally recomputes the
        explicit difference because only the bounded frontier is committed to
        either independent protection tier.
        """

        normalized = dict(event)
        source_end = int(normalized.get("source_committed_token_end", 0))
        source_dp_rank = int(normalized.get("source_dp_rank", -1))
        source_pp_rank = int(normalized.get("source_pp_rank", 0))
        snapshot = self._latest_kv_resource_snapshots.get(
            (source_dp_rank, source_pp_rank)
        )
        if not isinstance(snapshot, dict):
            return normalized
        page_size = int(snapshot.get("page_size_tokens", 0))
        if page_size <= 0:
            raise ValueError(
                "physical Aegis source snapshot has invalid page size"
            )
        rid = str(normalized.get("rid") or "")
        owner_pages = (snapshot.get("l1_device") or {}).get(
            "owner_pages", {}
        )
        if not isinstance(owner_pages, dict):
            raise ValueError(
                "physical Aegis source snapshot has invalid owner_pages"
            )
        retained_page_frontier = min(
            source_end,
            int(owner_pages.get(rid, 0)) * page_size,
        )
        bridge = self.aegis_transfer_bridge
        assert bridge is not None
        previous_frontier = bridge.materialized_source_frontier_token_end(
            request_backup_key=str(
                normalized.get("request_backup_key")
                or normalized.get("backup_key")
                or ""
            ),
            source_dp_rank=source_dp_rank,
            pp_rank=source_pp_rank,
            version=int(normalized.get("version", 1)),
            route_version=int(normalized.get("route_version", 1)),
            execution_id=str(normalized.get("execution_id") or ""),
        )
        materialized_end = min(
            source_end,
            max(retained_page_frontier, previous_frontier),
        )
        if materialized_end >= source_end:
            return normalized

        normalized["source_committed_token_end"] = materialized_end
        normalized["physical_source_observed_token_end"] = source_end
        normalized["physical_source_retained_page_frontier"] = (
            retained_page_frontier
        )
        normalized["physical_source_previous_frontier"] = previous_frontier
        normalized["physical_source_deferred_recompute_tokens"] = (
            source_end - materialized_end
        )
        normalized["physical_source_frontier_adjustment_reason"] = (
            "logical_frontier_exceeds_allocator_owned_pages"
        )
        self._physical_aegis_source_frontier_adjustments.append(
            {
                "timestamp_ns": self.global_des.now_ns,
                "rid": rid,
                "request_backup_key": str(
                    normalized.get("request_backup_key")
                    or normalized.get("backup_key")
                    or ""
                ),
                "source_dp_rank": source_dp_rank,
                "source_pp_rank": source_pp_rank,
                "version": int(normalized.get("version", 1)),
                "route_version": int(normalized.get("route_version", 1)),
                "page_size_tokens": page_size,
                "source_observed_token_end": source_end,
                "retained_page_frontier": retained_page_frontier,
                "previous_materialized_frontier": previous_frontier,
                "physical_source_token_end": materialized_end,
                "deferred_recompute_tokens": source_end - materialized_end,
                "reason": normalized[
                    "physical_source_frontier_adjustment_reason"
                ],
            }
        )
        return normalized

    def _release_terminal_physical_aegis_sources(self) -> int:
        """Drop source aliases once no physical transfer can consume them."""

        active_states = {
            TransferState.QUEUED,
            TransferState.IN_FLIGHT,
            TransferState.ACKED,
        }
        released = 0
        for extent_id, extent in tuple(self.global_des.extents.items()):
            if not extent_id.startswith("aegis-extent-"):
                continue
            if any(
                transfer.extent_id == extent_id
                and transfer.state in active_states
                for transfer in self.global_des.transfers.values()
            ):
                continue
            for location, state in tuple(extent.locations.items()):
                if (
                    location.tier is StorageTier.GPU_PRIMARY
                    and state is KvCopyState.COMMITTED
                ):
                    self.global_des.release_extent_location(
                        extent_id,
                        location,
                        reason="offline_aegis_source_copy_terminal",
                    )
                    released += 1
        return released

    def _on_physical_aegis_tier_notification(
        self,
        source_event: dict[str, Any],
        notification: TierNotification,
    ) -> None:
        dispatched = False
        if notification.success:
            self._physical_aegis_replay_records[notification.group_id] = (
                dict(source_event),
                notification,
            )
            completed_tier = (
                "hbm"
                if notification.tier is StorageTier.GPU_AEGIS
                else "dram"
            )
            dispatched = self.dp_controller.commit_aegis_physical_tier(
                {**source_event, **notification.as_dict()},
                completed_tier=completed_tier,
            )
            if dispatched:
                self._deferred_physical_aegis_commits.pop(
                    notification.group_id, None
                )
            else:
                # The physical copy remains committed even when its target
                # Scheduler is temporarily unavailable.  Preserve the exact
                # notification so repair can replay only the control-plane
                # materialization command; never repeat the byte transfer.
                self._deferred_physical_aegis_commits[
                    notification.group_id
                ] = (dict(source_event), notification)
        released_sources = self._release_terminal_physical_aegis_sources()
        self._physical_aegis_controller_dispatches.append(
            {
                **notification.as_dict(),
                "controller_dispatched": bool(dispatched),
                "dispatch_attempt": "initial",
                "deferred_until_target_repair": bool(
                    notification.success and not dispatched
                ),
                "released_source_aliases": released_sources,
            }
        )
        self._maybe_complete_physical_aegis_request(
            notification.request_backup_key,
            notification.version,
        )

    def _maybe_complete_physical_aegis_request(
        self, request_backup_key: str, version: int
    ) -> bool:
        """Run CompleteRequest only after both physical tier paths are terminal."""

        bridge = self.aegis_transfer_bridge
        completion_key = (str(request_backup_key), int(version))
        waiter = self._pending_physical_aegis_completions.get(completion_key)
        if bridge is None or waiter is None:
            return False
        if not bridge.request_groups_terminal(*completion_key):
            return False
        matching_admissions = [
            record
            for record in self._physical_recovery_admissions.values()
            if str(record.get("request_backup_key") or "") == completion_key[0]
        ]
        if matching_admissions:
            latest_route = max(
                int(record.get("execution_route_version", 0))
                for record in matching_admissions
            )
            matching_admissions = [
                record
                for record in matching_admissions
                if int(record.get("execution_route_version", 0)) == latest_route
            ]
        if any(
            record.get("state")
            not in {"ADOPTED", "FALLBACK_RECOMPUTE"}
            for record in matching_admissions
        ):
            # A committed protection copy remains owned until the replay load
            # has either been adopted or explicitly abandoned by a fenced
            # execution.  Do not release it merely because the old request
            # emitted a terminal message first.
            return False
        for admission in matching_admissions:
            group_id = str(admission.get("group_id") or "")
            group = self.global_des.recovery_load_groups.get(group_id)
            if admission.get("state") == "ADOPTED" and (
                group is None
                or group.state is not CommitGroupState.COMMITTED
                or group.adopted_shards != set(group.expected_shard_keys)
            ):
                raise RuntimeError(
                    "physical Aegis completion crossed an incomplete adoption gate"
                )
            if admission.get("state") in {
                "FALLBACK_RECOMPUTE",
            } and group is not None and not group.discarded:
                raise RuntimeError(
                    "physical Aegis fallback retained an undiscarded recovery group"
                )
        result = self.dp_controller.complete_aegis_request(
            backup_key=completion_key[0],
            expected_version=completion_key[1],
            target_dp_rank=int(waiter["target_dp_rank"]),
        )
        row = {
            **dict(waiter),
            **dict(result),
            "timestamp_ns": self.global_des.now_ns,
        }
        if bool(result.get("completed", False)):
            released = bridge.release_owned_copies(
                request_backup_key=completion_key[0],
                version=completion_key[1],
                reason="request_completed_target_unavailable",
            )
            row["physical_release_count"] = len(released)
        self._physical_aegis_completion_events.append(row)
        self._pending_physical_aegis_completions.pop(completion_key, None)
        return True

    def _retry_waiting_physical_completions(self, request_backup_key: str) -> None:
        key = str(request_backup_key)
        for backup_key, version in tuple(
            self._pending_physical_aegis_completions
        ):
            if backup_key == key:
                self._maybe_complete_physical_aegis_request(backup_key, version)

    def _retry_deferred_physical_aegis_commits(
        self, repaired_dp_ranks: tuple[int, ...]
    ) -> list[dict[str, Any]]:
        """Replay committed-tier notifications after target process repair."""

        repaired = {int(rank) for rank in repaired_dp_ranks}
        attempts: list[dict[str, Any]] = []
        for group_id, (source_event, notification) in tuple(
            self._physical_aegis_replay_records.items()
        ):
            if int(source_event.get("target_dp_rank", -1)) not in repaired:
                continue
            # Target HBM is volatile and was lost with the restarted worker.
            # Source-owned DRAM is the only tier whose already-committed bytes
            # can survive and whose control-plane metadata may be rehydrated.
            if notification.tier is not StorageTier.NODE_DRAM:
                continue
            hint = self.dp_controller.recovery_hint(
                str(notification.request_backup_key)
            )
            if not isinstance(hint, dict) or not any(
                int(hint.get(field, 0)) > 0
                for field in (
                    "l2_tokens",
                    "dram_pending_tokens",
                    "reserved_l2_tokens",
                )
            ):
                continue
            completed_tier = (
                "hbm"
                if notification.tier is StorageTier.GPU_AEGIS
                else "dram"
            )
            dispatched = self.dp_controller.commit_aegis_physical_tier(
                {**source_event, **notification.as_dict()},
                completed_tier=completed_tier,
                replay_after_target_restart=True,
            )
            row = {
                **notification.as_dict(),
                "controller_dispatched": bool(dispatched),
                "dispatch_attempt": "post_repair_retry",
                "deferred_until_target_repair": not bool(dispatched),
                "released_source_aliases": 0,
            }
            attempts.append(row)
            self._physical_aegis_controller_dispatches.append(row)
            if dispatched:
                self._deferred_physical_aegis_commits.pop(group_id, None)
        return attempts

    def _apply_worker_aegis_event(self, event: dict[str, Any]) -> None:
        bridge = self.aegis_transfer_bridge
        if (
            bridge is None
            or str(event.get("type")) != "aegis_source_chunk_committed"
        ):
            self.dp_controller.apply_aegis_rank_event(event)
            if bridge is not None:
                bridge.apply_release_intents(
                    tuple(
                        item
                        for item in event.get("physical_releases", ())
                        if isinstance(item, dict)
                    )
                )
            return

        backup_key = str(
            event.get("request_backup_key")
            or event.get("backup_key")
            or ""
        )
        hint = self.dp_controller.recovery_hint(backup_key)
        if not isinstance(hint, dict):
            raise RuntimeError(
                "physical Aegis source frontier has no reservation hint for "
                f"{backup_key!r}"
            )
        source_event = self._bound_physical_source_frontier_to_allocator(event)
        self._promote_physical_source_frontier_owner(source_event)
        model_revision, kv_abi, parallel_layout = (
            self._physical_aegis_identity(source_event, hint)
        )
        source_event["timestamp_ns"] = self.global_des.now_ns
        bridge.consume_source_chunk_committed(
            source_event,
            hint,
            kv_bytes_per_token_by_tp_rank=(
                self._physical_aegis_kv_bytes(source_event)
            ),
            model_revision=model_revision,
            kv_abi=kv_abi,
            parallel_layout=parallel_layout,
            on_tier_notification=lambda notification: (
                self._on_physical_aegis_tier_notification(
                    source_event, notification
                )
            ),
        )
        if bool(source_event.get("request_finished", False)):
            completion_key = (
                backup_key,
                int(source_event.get("version", 1)),
            )
            waiter = {
                "request_backup_key": backup_key,
                "version": completion_key[1],
                "route_version": int(source_event.get("route_version", 1)),
                "source_dp_rank": int(source_event.get("source_dp_rank", -1)),
                "target_dp_rank": int(source_event.get("target_dp_rank", -1)),
                "request_finished": True,
                "registered_at_ns": self.global_des.now_ns,
            }
            existing = self._pending_physical_aegis_completions.get(completion_key)
            if existing is not None and existing != waiter:
                raise ValueError(
                    "conflicting physical AegisKV CompleteRequest waiter"
                )
            self._pending_physical_aegis_completions[completion_key] = waiter
            self._maybe_complete_physical_aegis_request(*completion_key)

    def _ingest_worker_aegis_events(self, response: dict[str, Any]) -> int:
        count = 0
        for event in response.get("aegis_events", ()):
            if not isinstance(event, dict):
                continue
            self._apply_worker_aegis_event(event)
            count += 1
        return count

    def _route_pipeline_messages(self, response: dict[str, Any]) -> int:
        """Move immutable PP envelopes to their parent-ordered target inbox."""

        count = 0
        routes = (
            (
                "pp_request_messages",
                PPRequestEnvelope,
                self._pp_request_inboxes,
            ),
            (
                "pp_proxy_messages",
                PPProxyEnvelope,
                self._pp_proxy_inboxes,
            ),
            (
                "pp_output_messages",
                PPOutputEnvelope,
                self._pp_output_inboxes,
            ),
        )
        for field, expected_type, inboxes in routes:
            for message in response.get(field, ()):
                if not isinstance(message, expected_type):
                    raise TypeError(
                        f"offline {field} contains {type(message).__name__}, "
                        f"expected {expected_type.__name__}"
                    )
                target = message.target_stage
                key = (target.dp_rank, target.pp_rank, target.tp_rank)
                if key not in inboxes:
                    raise ValueError(
                        f"offline PP message targets unknown stage {target}"
                    )
                inboxes[key].append(message)
                count += 1
        return count

    def _ingest_worker_resource_snapshots(
        self, response: dict[str, Any]
    ) -> int:
        """Project Scheduler allocator state at the current global event.

        ``prepare`` is intentionally split into before/after observations:
        SGLang admission and KV allocation happen in that boundary operation,
        so collapsing both views would hide the exact HBM mutation that the
        offline physical-resource experiment is meant to expose.
        """

        bridge = self.resource_bridge
        if bridge is None:
            return 0
        response_type = str(response.get("type") or "worker_response")
        snapshots: list[tuple[str, Any]] = []
        if response_type == "prepared":
            snapshots.extend(
                (
                    ("prepare_before", response.get("kv_resource_before_prepare")),
                    ("prepare_after", response.get("kv_resource_after_prepare")),
                )
            )
        elif response_type == "completed":
            snapshots.append(
                ("batch_complete", response.get("kv_resource_at_completion"))
            )
        else:
            snapshots.append(
                (response_type, response.get("kv_resource_snapshot"))
            )

        metadata = {
            "worker_response_type": response_type,
            "worker_request_id": int(response.get("request_id", -1)),
            "logical_now_s": float(
                response.get("logical_now_s", self.coordinator.now_s)
            ),
        }
        telemetry = response.get("telemetry")
        if isinstance(telemetry, dict):
            metadata["batch_mode"] = str(telemetry.get("batch_mode", ""))
        count = 0
        for phase, snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            # The allocator snapshot is taken around a prepared batch, while
            # Scheduler.running_batch may not yet contain that batch.  The
            # immutable step telemetry is the causal owner list for this
            # boundary and prevents current-batch pages from being mislabeled
            # as reusable prefix pages.
            snapshot = dict(snapshot)
            active_request_ids = set(snapshot.get("running_request_ids", ()))
            if isinstance(telemetry, dict):
                active_request_ids.update(telemetry.get("request_ids", ()))
            snapshot["running_request_ids"] = sorted(
                str(rid) for rid in active_request_ids
            )
            snapshot_key = (
                int(snapshot.get("dp_rank", -1)),
                int(snapshot.get("pp_rank", 0)),
            )
            self._latest_kv_resource_snapshots[snapshot_key] = dict(snapshot)
            bridge.consume_snapshot(
                snapshot,
                phase=phase,
                event_metadata=metadata,
            )
            count += 1
        return count

    def _adopt_physical_recovery_for_prepared_response(
        self, response: dict[str, Any]
    ) -> int:
        """Hand committed shard bytes to a successful Scheduler prepare.

        The worker has already mutated its metadata allocator, but the parent
        has not yet ingested either the before/after resource snapshot.  This
        is therefore the only boundary where the physical hold can be removed
        and the Scheduler aggregate can take ownership without an observable
        double count or an early release of later PP-stage capacity.
        """

        if str(response.get("type") or "") != "prepared":
            return 0
        telemetry = response.get("telemetry")
        if not isinstance(telemetry, dict):
            return 0
        request_ids = {
            str(rid) for rid in telemetry.get("request_ids", ()) if str(rid)
        }
        if not request_ids:
            return 0
        response_dp_rank = int(response.get("dp_rank", -1))
        response_pp_rank = int(response.get("pp_rank", 0))
        adopted_count = 0
        for request_id in sorted(request_ids):
            record = self._physical_recovery_admissions.get(request_id)
            if record is None or record["state"] not in {
                "READY_FOR_STAGE_ADOPTION",
                "STAGE_ADOPTION_IN_PROGRESS",
            }:
                continue
            if response_dp_rank != int(record["current_dp_rank"]):
                raise RuntimeError(
                    f"physical recovery {request_id} prepared on stale DP rank"
                )
            group_id = str(record.get("group_id") or "")
            group = self.global_des.recovery_load_groups.get(group_id)
            if group is None or group.state is not CommitGroupState.COMMITTED:
                raise RuntimeError(
                    f"physical recovery {request_id} reached prepare without a "
                    "committed group"
                )
            if self.logical_pp:
                shard_keys = tuple(group.expected_shard_keys)
            else:
                prefix = f"pp{response_pp_rank}/"
                shard_keys = tuple(
                    key
                    for key in group.expected_shard_keys
                    if key.startswith(prefix)
                )
            if not shard_keys:
                raise RuntimeError(
                    f"physical recovery group {group_id} has no PP "
                    f"{response_pp_rank} shards"
                )
            error = self.global_des.recovery_load_shards_adoption_error(
                group_id, shard_keys
            )
            if error is not None:
                raise RuntimeError(
                    f"physical recovery {request_id} cannot adopt prepared stage: "
                    f"{error}"
                )
            group = self.global_des.adopt_recovery_load_shards(
                group_id, shard_keys
            )
            record["adopted_shards"] = sorted(group.adopted_shards)
            if group.adopted_shards == set(group.expected_shard_keys):
                record["state"] = "ADOPTED"
                record["adopted_at_ns"] = self.global_des.now_ns
                event_name = "all_shards_adopted"
            else:
                record["state"] = "STAGE_ADOPTION_IN_PROGRESS"
                event_name = "stage_shards_adopted"
            self._physical_recovery_admission_events.append(
                self._physical_recovery_admission_row(
                    record,
                    event=event_name,
                    timestamp_ns=self.global_des.now_ns,
                )
            )
            adopted_count += len(shard_keys)
            if record["state"] == "ADOPTED":
                self._retry_waiting_physical_completions(
                    str(record["request_backup_key"])
                )
        return adopted_count

    def _ingest_worker_hicache_intents(
        self, response: dict[str, Any]
    ) -> int:
        """Submit worker-owned ordinary L1/L2 intents after capacity projection."""

        raw_intents = list(response.get("hicache_intents", ()))
        bridge = self.hicache_transfer_bridge
        if bridge is None:
            if raw_intents:
                raise RuntimeError(
                    "ordinary HiCache intents require a physical GlobalDES adapter"
                )
            return 0
        key = self._stage_key(
            int(response.get("dp_rank", -1)),
            int(response.get("pp_rank", 0)),
            int(response.get("tp_rank", 0)),
        )
        if key not in self._worker_incarnations:
            raise ValueError(f"ordinary HiCache response targets unknown stage {key}")
        incarnation = self._worker_incarnations[key]
        submitted = 0
        for raw in raw_intents:
            if not isinstance(raw, dict):
                raise TypeError("ordinary HiCache intent must be a dictionary")
            intent_id = str(raw.get("intent_id") or "")
            if not intent_id:
                raise ValueError("ordinary HiCache intent has no intent_id")
            if int(raw.get("worker_incarnation", -1)) != incarnation:
                raise RuntimeError(
                    f"ordinary HiCache intent {intent_id} has stale worker "
                    f"incarnation {raw.get('worker_incarnation')}; expected "
                    f"{incarnation}"
                )
            previous = self._hicache_intent_stage.get(intent_id)
            if previous is not None and previous != key:
                raise RuntimeError(
                    f"ordinary HiCache intent {intent_id} changed stage identity"
                )
            self._hicache_intent_stage[intent_id] = key
            bridge.submit_intent(raw, ready_ns=self.global_des.now_ns)
            submitted += 1
        return submitted

    @staticmethod
    def _blocking_hicache_intent_ids(
        response: dict[str, Any]
    ) -> tuple[str, ...]:
        return tuple(
            str(raw.get("intent_id"))
            for raw in response.get("hicache_intents", ())
            if isinstance(raw, dict)
            and bool(raw.get("blocks_compute", False))
            and str(raw.get("intent_id") or "")
        )

    def _register_prepared_hicache_gate(
        self,
        key: tuple[int, int, int],
        response: dict[str, Any],
    ) -> bool:
        """Hold one prepared Scheduler step until every blocking ACK commits."""

        intent_ids = self._blocking_hicache_intent_ids(response)
        if not intent_ids or bool(response.get("idle", False)):
            return False
        if key in self._hicache_blocked_prepared:
            raise RuntimeError(f"stage {key} already waits on ordinary HiCache")
        bridge = self.hicache_transfer_bridge
        if bridge is None:
            raise RuntimeError("ordinary HiCache compute gate has no bridge")
        missing = [intent_id for intent_id in intent_ids if intent_id not in bridge.groups]
        if missing:
            raise RuntimeError(
                f"ordinary HiCache prepared gate has unknown intents {missing}"
            )
        self._hicache_blocked_prepared[key] = {
            "response": response,
            "intent_ids": intent_ids,
            "duration_s": float(response.get("duration_s", 0.0)),
            "prepared_at_s": float(
                response.get(
                    "logical_now_s",
                    self.coordinator.stage_driver(key[0], key[1]).now(),
                )
            ),
            "completion_event_id": "",
        }
        return True

    def _on_hicache_group_terminal(self, row: dict[str, Any]) -> None:
        """Queue the canonical group ACK and release an all-intent compute gate."""

        intent_id = str(row.get("intent_id") or "")
        key = self._hicache_intent_stage.get(intent_id)
        if key is None:
            self._hicache_artifact_blockers.append(
                f"ordinary HiCache terminal callback has no stage: {intent_id}"
            )
            return
        expected_incarnation = self._worker_incarnations.get(key)
        callback_incarnation = int(row.get("worker_incarnation", -1))
        if callback_incarnation != expected_incarnation:
            # A hard-failed process cannot consume an ACK.  Its canonical
            # physical group is already terminal and the fresh worker starts
            # with empty allocator state under a new incarnation.
            return
        if key in self._hicache_failure_teardown_stages:
            return
        state = str(row.get("state") or "")
        if state not in {"COMMITTED", "ABORTED"}:
            self._hicache_artifact_blockers.append(
                f"ordinary HiCache terminal callback is nonterminal: "
                f"{intent_id}={state}"
            )
            return
        ack = {"intent_id": intent_id, "state": state}
        if ack not in self._hicache_ack_inboxes[key]:
            self._hicache_ack_inboxes[key].append(ack)

        blocked = self._hicache_blocked_prepared.get(key)
        if blocked is None or intent_id not in blocked["intent_ids"]:
            return
        if state == "ABORTED":
            self._hicache_artifact_blockers.append(
                "ordinary HiCache compute dependency aborted without worker "
                f"teardown: stage={key}, intent={intent_id}, "
                f"reason={row.get('failure_reason', '')}"
            )
            return
        bridge = self.hicache_transfer_bridge
        assert bridge is not None
        if not all(
            bridge.groups[blocker].state
            is OrdinaryHiCacheTransferState.COMMITTED
            for blocker in blocked["intent_ids"]
        ):
            return
        if blocked["completion_event_id"]:
            return
        prepared_at_s = float(blocked["prepared_at_s"])
        dependency_ready_s = self.global_des.now_ns / 1_000_000_000
        dependency_delay_s = max(dependency_ready_s - prepared_at_s, 0.0)
        event = self.coordinator.schedule_stage_completion(
            key[0],
            key[1],
            duration_s=float(blocked["duration_s"]),
            payload={
                "prepared": blocked["response"],
                "tp_rank": key[2],
                "hicache_dependency_delay_s": dependency_delay_s,
                "hicache_dependency_intent_ids": list(blocked["intent_ids"]),
            },
        )
        blocked["completion_event_id"] = event.event_id

    def _ingest_worker_response(self, response: dict[str, Any]) -> None:
        for rid, stats in dict(
            response.get("terminal_request_stats") or {}
        ).items():
            normalized_rid = str(rid)
            normalized_stats = copy.deepcopy(dict(stats))
            previous = self._terminal_request_stats.get(normalized_rid)
            if previous is not None and previous != normalized_stats:
                raise RuntimeError(
                    "conflicting terminal request stats for "
                    f"{normalized_rid!r}"
                )
            self._terminal_request_stats[normalized_rid] = normalized_stats
        # The allocator state produced by this Scheduler boundary must be
        # projected before a source-frontier event aliases those primary bytes
        # into physical transfer extents.
        self._route_pipeline_messages(response)
        self._adopt_physical_recovery_for_prepared_response(response)
        self._ingest_worker_resource_snapshots(response)
        # Provisional D2H host pages and H2D destination pages are visible in
        # the allocator snapshot above.  Only now may the physical adapter
        # validate/adopt that external capacity and reserve shared resources.
        self._ingest_worker_hicache_intents(response)
        self._ingest_worker_aegis_events(response)
        self._ingest_worker_detokenizer_messages(response)

        key = (
            int(response.get("dp_rank", -1)),
            int(response.get("pp_rank", 0)),
            int(response.get("tp_rank", 0)),
        )
        if key in self._stage_idle and "rank_idle" in response:
            self._stage_idle[key] = bool(response["rank_idle"])
        response_type = str(response.get("type", ""))
        if response_type == "prepared":
            self._stage_blocked[key] = bool(
                response.get("blocked_on_slot", False)
            )
            if not bool(response.get("idle", False)):
                self._stage_buffered_work[key] = False
        elif response_type == "completed":
            telemetry = response.get("telemetry") or {}
            sequence = telemetry.get("pp_sequence")
            if sequence is not None:
                self._stage_next_sequence[key] = int(sequence) + 1

    def _call(
        self,
        dp_rank: int,
        pp_rank: int,
        op: str,
        payload: dict | None = None,
        *,
        tp_rank: int = 0,
    ) -> dict:
        rank = int(dp_rank)
        stage = int(pp_rank)
        tensor = int(tp_rank)
        stage_key = (rank, stage, tensor)
        if stage_key not in self.control_queues:
            raise ValueError(f"unknown offline Scheduler stage {stage_key}")
        request_id = self._request_sequence
        self._request_sequence += 1
        command_payload = dict(payload or {})
        request_ids: set[str] = set()
        for request in command_payload.get("incoming_requests", ()):
            rid = str(getattr(request, "rid", "") or "")
            if rid:
                request_ids.add(rid)
        for field in ("pp_request_messages", "pp_proxy_messages"):
            for envelope in command_payload.get(field, ()):
                request_ids.update(
                    str(rid)
                    for rid in getattr(envelope, "request_ids", ())
                    if str(rid)
                )
        lower_bound_s = max(
            self.coordinator.now_s,
            float(
                command_payload.get(
                    "at_s",
                    command_payload.get(
                        "completed_at",
                        self.coordinator.stage_driver(rank, stage).now(),
                    ),
                )
            ),
        )
        barrier = self.coordinator.begin_stage_work(
            rank,
            stage,
            lower_bound_s=lower_bound_s,
            operation=str(op),
            metadata={
                "tp_rank": tensor,
                "request_id": request_id,
                "request_ids": sorted(request_ids),
            },
        )
        key = (rank, stage, tensor, request_id)
        dispatched = False
        try:
            self.control_queues[stage_key].put(
                (request_id, op, command_payload)
            )
            dispatched = True
            cached = self._pending_responses.pop(key, None)
            if cached is not None:
                response = cached
            else:
                while True:
                    try:
                        response = self.result_queue.get(
                            timeout=self.worker_timeout_s
                        )
                    except queue.Empty as exc:
                        raise TimeoutError(
                            f"offline worker {rank} timed out during {op}"
                        ) from exc
                    response_key = (
                        int(response.get("dp_rank", -1)),
                        int(response.get("pp_rank", 0)),
                        int(response.get("tp_rank", 0)),
                        int(response.get("request_id", -1)),
                    )
                    if response_key == key:
                        break
                    self._pending_responses[response_key] = response
        except BaseException:
            # Before queue publication there is no remote work to guard.  Once
            # published, however, a timeout/transport error means the parent
            # still has no safe next-event timestamp.  Keep that barrier live
            # so a caller cannot catch the exception and advance virtual time.
            if not dispatched:
                self.coordinator.cancel_stage_work(
                    rank,
                    stage,
                    barrier_id=barrier.barrier_id,
                )
            raise
        self.coordinator.finish_stage_work(
            rank,
            stage,
            barrier_id=barrier.barrier_id,
        )
        if response.get("type") == "error":
            raise RuntimeError(
                f"offline worker {stage_key} {op} failed: "
                f"{response.get('error_type', 'Error')}: {response.get('error', '')}"
            )
        return response

    def dispatch(self, request: TokenizedGenerateReqInput) -> int:
        sim = request.sampling_params.custom_params.setdefault("simulation", {})
        sim.setdefault("server_created_time", self.coordinator.now_s)
        request_id = str(request.rid)
        previous = self._request_catalog.get(request_id)
        if previous is not None and previous is not request:
            raise ValueError(
                f"offline request ID {request_id!r} was dispatched more than once"
            )
        self._request_catalog[request_id] = request
        self.router_output_ledger.register_dispatch(request)
        provider = self._causal_exact_plan_provider
        if provider is not None and not bool(sim.get("failover_replay", False)):
            plan = provider(request, self)
            if plan is None:
                raise RuntimeError(
                    f"causal exact provider returned no plan for {request.rid}"
                )
            installation = self.dp_controller.install_certified_exact_window(plan)
            self.exact_rolling_installations.append(
                {
                    "logical_time_s": self.coordinator.now_s,
                    "request_id": request.rid,
                    **installation,
                }
            )
        gate_candidate = bool(
            self.aegis_transfer_bridge is not None
            and sim.get("failover_replay", False)
        )
        captured: list[list[TokenizedGenerateReqInput]] | None = None
        if gate_candidate:
            captured = [[] for _ in range(self.dp_size)]
            actual_workers = self.dp_controller.workers
            self.dp_controller.workers = [
                _OfflineRequestSender(captured[rank])
                for rank in range(self.dp_size)
            ]
            try:
                assigned = self.dp_controller.dispatch(request)
            finally:
                self.dp_controller.workers = actual_workers
        else:
            assigned = self.dp_controller.dispatch(request)
        if assigned is None:
            raise RuntimeError(f"offline router did not assign request {request.rid}")
        if captured is not None:
            captured_requests = [
                item for inbox in captured for item in inbox
            ]
            if captured_requests != [request] or captured[int(assigned)] != [request]:
                raise RuntimeError(
                    "offline recovery dispatch did not produce one captured handoff"
                )
        self.router_output_ledger.confirm_dispatched_route(
            request_id,
            route_version=int(sim.get("aegis_route_version", 1)),
        )
        self.router_output_ledger.set_assignment(
            request_id,
            dp_rank=int(assigned),
            pp_size=self.pp_size,
            logical_tp_size=self.tp_size,
            configured_pp_size=self.configured_pp_size,
            logical_pp=self.logical_pp,
        )
        if captured is None:
            self._plan_physical_aegis_recovery(
                request,
                retry_dp_rank=int(assigned),
            )
            return int(assigned)

        try:
            plan = self._plan_physical_aegis_recovery(
                request,
                retry_dp_rank=int(assigned),
            )
        except (RuntimeError, ValueError) as exc:
            record = self._new_physical_recovery_admission(
                request,
                assigned_dp_rank=int(assigned),
                plan=None,
                planning_error=f"{type(exc).__name__}: {exc}",
            )
            assigned = self._activate_physical_recovery_fallback(
                record,
                reason=f"planning_failed:{type(exc).__name__}:{exc}",
            )
        else:
            if plan is None or not str(plan.get("group_id") or ""):
                self._request_inboxes[int(assigned)].append(request)
            else:
                record = self._new_physical_recovery_admission(
                    request,
                    assigned_dp_rank=int(assigned),
                    plan=plan,
                )
                gate = self.global_des.schedule_at(
                    int(plan["max_end_ns"]),
                    kind="physical_recovery_admission_gate",
                    priority=EventPriority.CONTROL,
                    payload={
                        "request_id": request_id,
                        "group_id": str(plan["group_id"]),
                    },
                )
                record["gate_event_id"] = gate.event_id
        return int(assigned)

    def _new_physical_recovery_admission(
        self,
        request: TokenizedGenerateReqInput,
        *,
        assigned_dp_rank: int,
        plan: dict[str, Any] | None,
        planning_error: str | None = None,
    ) -> dict[str, Any]:
        """Create the parent-owned gate record without exposing the request."""

        request_id = str(request.rid)
        if request_id in self._physical_recovery_admissions:
            raise RuntimeError(
                f"physical recovery admission already exists for {request_id!r}"
            )
        sim = request.sampling_params.custom_params.setdefault("simulation", {})
        group_id = str((plan or {}).get("group_id") or "") or None
        if group_id is None:
            matches = [
                group.group_id
                for group in self.global_des.recovery_load_groups.values()
                if group.retry_request_id == request_id
            ]
            if len(matches) == 1:
                group_id = matches[0]
        record: dict[str, Any] = {
            "request": request,
            "request_id": request_id,
            "request_backup_key": str(
                (plan or {}).get("request_backup_key")
                or sim.get("aegis_retry_resolved_backup_key")
                or sim.get("aegis_backup_key")
                or ""
            ),
            "assigned_dp_rank": int(assigned_dp_rank),
            "current_dp_rank": int(assigned_dp_rank),
            "route_version": int(
                (plan or {}).get("route_version")
                or sim.get("aegis_route_version", 1)
            ),
            "execution_route_version": int(
                sim.get("aegis_route_version", 1)
            ),
            "group_id": group_id,
            "state": "PENDING_TRANSFER" if plan is not None else "PLANNING_FAILED",
            "created_at_ns": self.global_des.now_ns,
            "gate_event_id": None,
            "released_to_scheduler_at_ns": None,
            "adopted_at_ns": None,
            "adopted_shards": [],
            "fallback_reason": None,
            "planning_error": planning_error,
            "plan": copy.deepcopy(plan),
        }
        self._physical_recovery_admissions[request_id] = record
        self._physical_recovery_admission_events.append(
            self._physical_recovery_admission_row(record, event="registered")
        )
        return record

    @staticmethod
    def _physical_recovery_admission_row(
        record: dict[str, Any],
        *,
        event: str | None = None,
        timestamp_ns: int | None = None,
    ) -> dict[str, Any]:
        row = {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key != "request"
        }
        if event is not None:
            row["event"] = str(event)
        if timestamp_ns is not None:
            row["timestamp_ns"] = int(timestamp_ns)
        return row

    def _set_router_assignment(self, request_id: str, dp_rank: int) -> None:
        self.router_output_ledger.set_assignment(
            request_id,
            dp_rank=int(dp_rank),
            pp_size=self.pp_size,
            logical_tp_size=self.tp_size,
            configured_pp_size=self.configured_pp_size,
            logical_pp=self.logical_pp,
        )

    def _activate_physical_recovery_fallback(
        self,
        record: dict[str, Any],
        *,
        reason: str,
    ) -> int:
        """Discard protected-prefix recovery and deterministically recompute."""

        request = record["request"]
        if not isinstance(request, TokenizedGenerateReqInput):
            raise TypeError("physical recovery admission lost its request")
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("physical recovery fallback reason must be nonempty")
        group_id = record.get("group_id")
        if group_id is not None and group_id in self.global_des.recovery_load_groups:
            group = self.global_des.recovery_load_groups[str(group_id)]
            if not group.adopted_shards:
                self.global_des.discard_recovery_load_group(
                    str(group_id), reason=normalized_reason
                )

        sim = request.sampling_params.custom_params.setdefault("simulation", {})
        total_tokens = max(
            int(sim.get("pre_failure_total_tokens", 0)),
            len(request.input_ids or [])
            + int(request.sampling_params.max_new_tokens),
        )
        recompute_delay_s = self.dp_controller.aegis_recompute_delay_s(total_tokens)
        sim.update(
            {
                "aegis_retry_used_hint": False,
                "peer_gpu_lookup_hit": False,
                "peer_gpu_lookup_tokens": 0,
                "host_backup_lookup_hit": False,
                "host_backup_lookup_tokens": 0,
                "l3_lookup_hit": False,
                "l3_lookup_tokens": 0,
                "restore_tokens": 0,
                "restore_recompute_tokens": total_tokens,
                "restore_action": "recompute",
                "restore_data_plane": "recompute",
                "restore_overhead_s": recompute_delay_s,
                "restore_recompute_delay_s": recompute_delay_s,
                "restore_timing_source": "global_des_recompute_fallback",
                "physical_recovery_fallback_reason": normalized_reason,
            }
        )

        current_rank = int(record["current_dp_rank"])
        if not self.dp_controller._worker_available(current_rank):
            raise RuntimeError(
                "unavailable physical recovery target requires a Router fence "
                "and a new execution; in-place route reassignment is forbidden"
            )
        else:
            self._request_inboxes[current_rank].append(request)

        record["state"] = "FALLBACK_RECOMPUTE"
        record["fallback_reason"] = normalized_reason
        record["released_to_scheduler_at_ns"] = self.global_des.now_ns
        self._physical_recovery_admission_events.append(
            self._physical_recovery_admission_row(
                record,
                event="fallback_released",
                timestamp_ns=self.global_des.now_ns,
            )
        )
        self._retry_waiting_physical_completions(
            str(record["request_backup_key"])
        )
        return current_rank

    def _resolve_physical_recovery_admission(self, event: OfflineEvent) -> int:
        payload = dict(event.payload or {})
        request_id = str(payload.get("request_id") or "")
        record = self._physical_recovery_admissions.get(request_id)
        if record is None:
            raise RuntimeError(
                f"physical recovery gate has no admission for {request_id!r}"
            )
        if record["state"] != "PENDING_TRANSFER":
            return int(record["current_dp_rank"])

        group_id = str(record.get("group_id") or "")
        group = self.global_des.recovery_load_groups.get(group_id)
        fallback_reason: str | None = None
        if group is None:
            fallback_reason = "missing_recovery_group"
        elif group.state is not CommitGroupState.COMMITTED:
            fallback_reason = f"recovery_group_{group.state.value.lower()}"
        elif int(group.route_version) != int(record["route_version"]):
            fallback_reason = "stale_group_route_version"
        else:
            hint = self.dp_controller.recovery_hint(
                str(record["request_backup_key"])
            )
            if not isinstance(hint, dict):
                fallback_reason = "stale_route:no_current_protection_hint"
            elif int(hint.get("route_version", -1)) != int(record["route_version"]):
                fallback_reason = "stale_route:route_version_changed"
            elif not self.dp_controller._worker_available(
                int(record["current_dp_rank"])
            ):
                fallback_reason = "target_rank_unavailable"
            else:
                fallback_reason = self.global_des.recovery_load_shards_adoption_error(
                    group_id
                )

        if fallback_reason is not None:
            return self._activate_physical_recovery_fallback(
                record, reason=fallback_reason
            )

        request = record["request"]
        assert isinstance(request, TokenizedGenerateReqInput)
        target_rank = int(record["current_dp_rank"])
        self._request_inboxes[target_rank].append(request)
        record["state"] = "READY_FOR_STAGE_ADOPTION"
        record["released_to_scheduler_at_ns"] = self.global_des.now_ns
        self._physical_recovery_admission_events.append(
            self._physical_recovery_admission_row(
                record,
                event="group_committed_scheduler_released",
                timestamp_ns=self.global_des.now_ns,
            )
        )
        return target_rank

    def _physical_recovery_admission_export(self) -> dict[str, Any]:
        """Export canonical gate state and every still-owned capacity hold."""

        rows = [
            self._physical_recovery_admission_row(record)
            for _, record in sorted(self._physical_recovery_admissions.items())
        ]
        group_ids = sorted(
            {
                str(record.get("group_id") or "")
                for record in self._physical_recovery_admissions.values()
                if str(record.get("group_id") or "")
            }
        )
        transfer_ids = sorted(
            {
                transfer_id
                for group_id in group_ids
                for transfer_id in self.global_des.recovery_load_groups[
                    group_id
                ].transfer_ids
            }
        )
        pending_holds = [
            {
                "request_id": self.global_des.recovery_load_groups[
                    str(transfer.recovery_group_id)
                ].retry_request_id,
                **transfer.snapshot(),
            }
            for transfer_id in transfer_ids
            for transfer in (self.global_des.transfers[transfer_id],)
            if transfer.recovery_capacity_hold_active
        ]
        return {
            "schema": "sglsim.physical_recovery_admission.v1",
            "admissions": rows,
            "events": copy.deepcopy(self._physical_recovery_admission_events),
            "canonical_groups": [
                self.global_des.recovery_load_groups[group_id].snapshot()
                for group_id in group_ids
            ],
            "canonical_transfers": [
                self.global_des.transfers[transfer_id].snapshot()
                for transfer_id in transfer_ids
            ],
            "pending_holds": pending_holds,
            "fallbacks": [
                {
                    "request_id": row["request_id"],
                    "state": row["state"],
                    "fallback_reason": row.get("fallback_reason"),
                }
                for row in rows
                if row.get("fallback_reason")
            ],
        }

    def _plan_physical_aegis_recovery(
        self,
        request: TokenizedGenerateReqInput,
        *,
        retry_dp_rank: int,
    ) -> dict[str, Any] | None:
        """Replace aggregate retry-copy timing with physical DES resources."""

        bridge = self.aegis_transfer_bridge
        if bridge is None:
            return None
        sim = request.sampling_params.custom_params.setdefault(
            "simulation", {}
        )
        if not bool(sim.get("failover_replay", False)):
            return None
        if not bool(sim.get("aegis_retry_used_hint", False)):
            return None

        hbm_tokens = max(int(sim.get("peer_gpu_lookup_tokens", 0)), 0)
        dram_tokens = max(int(sim.get("host_backup_lookup_tokens", 0)), 0)
        if hbm_tokens and dram_tokens:
            raise RuntimeError(
                "latest independent-tier retry selected both HBM and DRAM"
            )
        if not (hbm_tokens or dram_tokens):
            return None
        backup_key = str(
            sim.get("aegis_retry_resolved_backup_key")
            or sim.get("aegis_backup_key")
            or sim.get("request_backup_key")
            or ""
        )
        if not backup_key:
            raise RuntimeError("physical failover replay has no backup key")
        hint = self.dp_controller.recovery_hint(backup_key)
        if not isinstance(hint, dict):
            raise RuntimeError(
                f"physical failover replay has no committed hint for {backup_key!r}"
            )
        plan = bridge.plan_failover_recovery(
            retry_request_id=str(request.rid),
            request_backup_key=backup_key,
            retry_dp_rank=int(retry_dp_rank),
            tier_tokens={
                StorageTier.GPU_AEGIS: hbm_tokens,
                StorageTier.NODE_DRAM: dram_tokens,
            },
            ready_ns=self.global_des.now_ns,
            route_version=int(hint.get("route_version", 1)),
        )
        physical_wait_s = max(
            int(plan["max_end_ns"]) - self.global_des.now_ns,
            0,
        ) / 1_000_000_000.0
        recompute_delay_s = self.dp_controller.aegis_recompute_delay_s(
            int(sim.get("restore_recompute_tokens", 0))
        )
        modeled_overhead_s = max(float(sim.get("restore_overhead_s", 0.0)), 0.0)
        sim.update(
            {
                "restore_modeled_overhead_s_replaced": modeled_overhead_s,
                "restore_physical_transfer_wait_s": physical_wait_s,
                "restore_recompute_delay_s": recompute_delay_s,
                # The request itself remains outside the Scheduler inbox until
                # the physical group commits.  Charging that wait again at
                # Scheduler ingress would double count virtual time.
                "restore_overhead_s": recompute_delay_s,
                "restore_timing_source": "global_des_physical_recovery",
                "physical_recovery_plan": copy.deepcopy(plan),
            }
        )
        row = {
            **copy.deepcopy(plan),
            "modeled_overhead_s_replaced": modeled_overhead_s,
            "physical_transfer_wait_s": physical_wait_s,
            "recompute_delay_s": recompute_delay_s,
            "scheduler_ingress_delay_s": recompute_delay_s,
            "pending_admission_wait_s": physical_wait_s,
        }
        self._physical_aegis_recovery_plans.append(row)
        return copy.deepcopy(row)

    def set_causal_exact_plan_provider(
        self,
        provider: Callable[
            [TokenizedGenerateReqInput, "OfflineSchedulerCluster"],
            CertifiedExactWindowPlan | None,
        ]
        | None,
    ) -> None:
        """Install an offline-only causal solver callback.

        The callback runs in the parent driver before original-request routing.
        Scheduler workers and the online runtime still consume only the
        solver-free certified wire plan.
        """

        self._causal_exact_plan_provider = provider

    def schedule_request_arrival(
        self,
        request: TokenizedGenerateReqInput,
        *,
        at_s: float,
    ) -> OfflineEvent:
        """Inject a pre-tokenized request on the shared parent event heap."""

        return self.coordinator.schedule_at(
            float(at_s),
            kind="request_arrival",
            priority=OfflineEventPriority.REQUEST_ARRIVAL,
            payload=request,
        )

    def _stage_process(self, key: tuple[int, int, int]) -> mp.Process:
        try:
            return self._process_by_stage[key]
        except KeyError as exc:
            raise ValueError(f"unknown offline Scheduler stage {key}") from exc

    def _terminate_stage_worker(
        self, key: tuple[int, int, int]
    ) -> dict[str, Any]:
        """Terminate one physical Scheduler stage without a graceful flush."""

        process = self._stage_process(key)
        pid = process.pid
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        if process.is_alive():
            raise RuntimeError(f"failed to terminate offline Scheduler stage {key}")
        return {
            "dp_rank": key[0],
            "pp_rank": key[1],
            "tp_rank": key[2],
            "pid": pid,
            "exitcode": process.exitcode,
        }

    def _spawn_stage_worker(
        self, key: tuple[int, int, int]
    ) -> dict[str, Any]:
        """Restart one stage with a new empty Scheduler/cache process."""

        self._worker_incarnations[key] += 1
        old_queue = self.control_queues[key]
        self._retired_control_queues.append(old_queue)
        control_queue = mp.Queue()
        ready = mp.Event()
        process = mp.Process(
            target=run_offline_scheduler_worker,
            args=(
                self.server_args,
                self.port_args,
                key[0],
                key[1],
                key[2],
                self.ipc_registry,
                None,
                control_queue,
                self.result_queue,
                self.stats_queue,
                ready,
                self.hicache_transfer_bridge is not None,
                self._worker_incarnations[key],
            ),
            daemon=True,
        )
        process.start()
        if not ready.wait(timeout=self.worker_timeout_s):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            raise RuntimeError(f"offline Scheduler stage {key} failed to restart")
        self.control_queues[key] = control_queue
        self._process_by_stage[key] = process
        self._ready_event_by_stage[key] = ready
        return {
            "dp_rank": key[0],
            "pp_rank": key[1],
            "tp_rank": key[2],
            "pid": process.pid,
            "worker_incarnation": self._worker_incarnations[key],
        }

    def _replace_worker_handles_after_restart(self, dp_rank: int) -> None:
        """Atomically publish restarted process handles to the Controller."""

        workers = tuple(
            PipelineWorkerHandle(
                spec=spec,
                process=self._process_by_stage[
                    (
                        spec.stage.dp_rank,
                        spec.stage.pp_rank,
                        spec.stage.tp_rank,
                    )
                ],
                command_queue=_OfflineCommandSender(
                    self._runtime_command_inboxes[
                        (
                            spec.stage.dp_rank,
                            spec.stage.pp_rank,
                            spec.stage.tp_rank,
                        )
                    ]
                ),
                ready_event=self._ready_event_by_stage[
                    (
                        spec.stage.dp_rank,
                        spec.stage.pp_rank,
                        spec.stage.tp_rank,
                    )
                ],
            )
            for spec in self._worker_specs
        )
        group = PipelineProcessGroup(
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            tp_size=self.physical_tp_size,
            workers=workers,
        )
        self.pipeline_process_group = group
        self.processes = [
            self._process_by_stage[
                (
                    spec.stage.dp_rank,
                    spec.stage.pp_rank,
                    spec.stage.tp_rank,
                )
            ]
            for spec in self._worker_specs
        ]
        if self.pp_size > 1:
            self.dp_controller.replace_pipeline_process_group(group)
        else:
            leader = group.leader_for_dp(int(dp_rank))
            self.dp_controller.replace_scheduler_process(
                int(dp_rank),
                process=leader.process,
                command_queue=leader.command_queue,
            )

    @staticmethod
    def _event_request_ids(event: OfflineEvent | None) -> set[str]:
        if event is None or not isinstance(event.payload, dict):
            return set()
        prepared = event.payload.get("prepared")
        if not isinstance(prepared, dict):
            return set()
        telemetry = prepared.get("telemetry")
        if not isinstance(telemetry, dict):
            return set()
        return {
            str(rid)
            for rid in telemetry.get("request_ids", ())
            if str(rid)
        }

    def _physical_node_controller_id(self, node_id: str) -> int:
        if self.resource_bridge is None:
            raise RuntimeError("physical node failure requires a resource bridge")
        node_ids = [node.node_id for node in self.resource_bridge.topology.nodes]
        try:
            return node_ids.index(str(node_id))
        except ValueError as exc:
            raise ValueError(f"unknown physical node {node_id!r}") from exc

    def schedule_fail_stop(
        self,
        *,
        event_id: str,
        offset_s: float,
        duration_s: float,
        failure_scope: str = "worker",
        dp_rank: int | None = None,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Schedule a real offline fail-stop and empty-state restart.

        Unlike :meth:`schedule_soft_failure`, this path cancels prepared
        compute completions, terminates Scheduler subprocesses, invalidates
        physical HBM (and node DRAM for a node fault), and starts fresh worker
        processes only at the repair event.
        """

        resolved_id = str(event_id)
        if not resolved_id:
            raise ValueError("failure event_id must be nonempty")
        if resolved_id in self._scheduled_failure_ids:
            raise ValueError(f"duplicate offline failure event_id={resolved_id}")
        duration = float(duration_s)
        if duration <= 0.0:
            raise ValueError("failure duration must be positive")
        scope = str(failure_scope).lower()
        if scope not in {"worker", "gpu", "process", "node"}:
            raise ValueError(f"unsupported fail-stop scope {failure_scope!r}")
        if scope == "node":
            if not node_id:
                raise ValueError("node fail-stop requires node_id")
            scheduled_rank = None
        else:
            if dp_rank is None or not 0 <= int(dp_rank) < self.dp_size:
                raise ValueError("worker/gpu/process fail-stop requires valid dp_rank")
            if node_id is not None:
                raise ValueError("node_id is valid only for node fail-stop")
            scheduled_rank = int(dp_rank)
        self._scheduled_failure_ids.add(resolved_id)
        common = {
            "event_id": resolved_id,
            "duration_s": duration,
            "failure_mode": "hard_fail_stop",
            "failure_scope": scope,
            "dp_rank": scheduled_rank,
            "node_id": str(node_id) if node_id is not None else None,
            **dict(payload or {}),
        }
        self.coordinator.schedule_at(
            float(offset_s),
            kind="hard_failure_inject",
            priority=OfflineEventPriority.FAILURE_CONTROL,
            dp_rank=scheduled_rank,
            payload=common,
        )

    def _hard_failure_targets(
        self, payload: dict[str, Any]
    ) -> tuple[tuple[int, ...], str | None, int | None]:
        scope = str(payload.get("failure_scope") or "worker").lower()
        if scope != "node":
            rank = int(payload.get("dp_rank", -1))
            if not 0 <= rank < self.dp_size:
                raise ValueError("hard failure payload has no valid DP rank")
            return (rank,), None, None
        if self.resource_bridge is None:
            raise RuntimeError("node fail-stop requires physical topology")
        node_id = str(payload.get("node_id") or "")
        if not node_id:
            raise ValueError("node fail-stop payload has no node_id")
        ranks = tuple(
            sorted(
                {
                    spec.gpu.dp_rank
                    for spec in self.resource_bridge.topology.gpus
                    if spec.gpu.node_id == node_id
                }
            )
        )
        if not ranks:
            raise ValueError(f"node fail-stop targets no DP ranks on {node_id!r}")
        return ranks, node_id, self._physical_node_controller_id(node_id)

    def _abort_hicache_for_hard_failure(
        self,
        *,
        ranks: tuple[int, ...],
        node_id: str | None,
        event_id: str,
    ) -> dict[str, Any]:
        """Abort ordinary transfers and adopt lost provisional D2H holds."""

        bridge = self.hicache_transfer_bridge
        if bridge is None:
            return {}
        affected_ranks = {int(rank) for rank in ranks}
        active_groups = [
            group
            for group in bridge.groups.values()
            if group.intent.dp_rank in affected_ranks
            and group.state
            in {
                OrdinaryHiCacheTransferState.PENDING,
                OrdinaryHiCacheTransferState.IN_FLIGHT,
            }
        ]
        d2h_by_stage: dict[tuple[int, int], list[Any]] = {}
        for group in active_groups:
            if group.intent.direction is OrdinaryHiCacheDirection.D2H:
                d2h_by_stage.setdefault(
                    (group.intent.dp_rank, group.intent.pp_rank), []
                ).append(group.intent)

        teardown_keys = {
            key for key in self._stage_keys if key[0] in affected_ranks
        }
        self._hicache_failure_teardown_stages.update(teardown_keys)
        if node_id is not None:
            aborted_group_count = bridge.fail_node(
                node_id,
                reason=f"hard_fail_stop:{event_id}",
            )
        else:
            aborted_ids: set[str] = set()
            for spec in bridge.topology.gpus:
                if spec.gpu.dp_rank not in affected_ranks:
                    continue
                before = {
                    intent_id
                    for intent_id, group in bridge.groups.items()
                    if group.state
                    in {
                        OrdinaryHiCacheTransferState.PENDING,
                        OrdinaryHiCacheTransferState.IN_FLIGHT,
                    }
                }
                bridge.fail_gpu(
                    spec.gpu,
                    reason=f"hard_fail_stop:{event_id}",
                )
                aborted_ids.update(
                    intent_id
                    for intent_id in before
                    if bridge.groups[intent_id].state
                    is OrdinaryHiCacheTransferState.ABORTED
                )
            aborted_group_count = len(aborted_ids)

        projection_rollbacks: list[dict[str, Any]] = []
        # Node fail-stop destroys the entire node DRAM ledger below.  A worker
        # or GPU fail-stop preserves committed L2, so explicitly subtract only
        # the host pages whose all-TP D2H group did not commit.
        if node_id is None and d2h_by_stage:
            resource_bridge = self.resource_bridge
            if resource_bridge is None:
                raise RuntimeError(
                    "ordinary HiCache hard-failure rollback has no resource bridge"
                )
            for (dp_rank, pp_rank), intents in sorted(d2h_by_stage.items()):
                projection_rollbacks.append(
                    resource_bridge.release_provisional_hicache_d2h(
                        dp_rank,
                        pp_rank,
                        bytes_per_tp_rank=sum(
                            int(intent.size_bytes_per_tp_rank)
                            for intent in intents
                        ),
                        reason=(
                            f"ordinary_hicache_hard_failure_abort:{event_id}"
                        ),
                        intent_ids=tuple(
                            sorted(intent.intent_id for intent in intents)
                        ),
                    )
                )
        for key in teardown_keys:
            self._hicache_ack_inboxes[key].clear()
            self._hicache_blocked_prepared.pop(key, None)
        return {
            "aborted_group_count": aborted_group_count,
            "active_group_ids": sorted(
                group.intent.intent_id for group in active_groups
            ),
            "projection_rollbacks": projection_rollbacks,
        }

    def _inject_hard_failure(self, event: OfflineEvent) -> None:
        payload = dict(event.payload or {})
        event_id = str(payload.get("event_id") or f"failure-{event.sequence}")
        duration = float(payload.get("duration_s", 0.0))
        if duration <= 0.0:
            raise ValueError(f"hard failure {event_id} has invalid duration")
        ranks, node_id, controller_node_id = self._hard_failure_targets(payload)
        overlap = sorted(set(ranks).intersection(self._hard_failed_dp_ranks))
        if overlap:
            raise RuntimeError(
                f"hard failure {event_id} overlaps failed DP ranks {overlap}"
            )
        healthy_after = {
            rank for rank in range(self.dp_size) if rank not in ranks
        } - self._hard_failed_dp_ranks
        if not healthy_after:
            raise RuntimeError(
                f"hard failure {event_id} would leave no healthy DP rank"
            )

        ordinary_hicache_failure = self._abort_hicache_for_hard_failure(
            ranks=ranks,
            node_id=node_id,
            event_id=event_id,
        )

        affected_request_ids: set[str] = set()
        fenced_pending_admissions: list[str] = []
        for request_id, admission in self._physical_recovery_admissions.items():
            if (
                int(admission.get("current_dp_rank", -1)) not in ranks
                or admission.get("state")
                not in {
                    "PENDING_TRANSFER",
                    "READY_FOR_STAGE_ADOPTION",
                    "STAGE_ADOPTION_IN_PROGRESS",
                }
            ):
                continue
            affected_request_ids.add(request_id)
            gate_event_id = str(admission.get("gate_event_id") or "")
            if (
                admission.get("state") == "PENDING_TRANSFER"
                and gate_event_id
            ):
                admission["gate_event_cancelled"] = self.global_des.cancel_event(
                    gate_event_id,
                    reason="target_hard_failure",
                    metadata={"failure_event_id": event_id},
                )
            group_id = str(admission.get("group_id") or "")
            group = self.global_des.recovery_load_groups.get(group_id)
            adopted_at_failure = (
                [] if group is None else sorted(group.adopted_shards)
            )
            admission["partial_adoption_at_failure"] = bool(
                group is not None
                and adopted_at_failure
                and len(adopted_at_failure) < len(group.expected_shard_keys)
            )
            admission["lost_adopted_shards"] = adopted_at_failure
            if group is not None:
                if group.adopted_shards:
                    self.global_des.abandon_partially_adopted_recovery_group(
                        group_id,
                        reason=f"target_hard_failure:{event_id}",
                    )
                else:
                    self.global_des.discard_recovery_load_group(
                        group_id,
                        reason=f"target_hard_failure:{event_id}",
                    )
            admission["state"] = "FENCED_BY_HARD_FAILURE"
            admission["fallback_reason"] = f"target_hard_failure:{event_id}"
            admission["fenced_at_ns"] = self.global_des.now_ns
            fenced_pending_admissions.append(request_id)
            self._physical_recovery_admission_events.append(
                self._physical_recovery_admission_row(
                    admission,
                    event="target_failure_fenced",
                    timestamp_ns=self.global_des.now_ns,
                )
            )
        cancelled: list[dict[str, Any]] = []
        terminated: list[dict[str, Any]] = []
        for rank in ranks:
            for pp_rank in range(self.pp_size):
                key = self._stage_key(rank, pp_rank, 0)
                cancelled_event = self.coordinator.cancel_stage_completion(
                    rank,
                    pp_rank,
                    reason="hard_fail_stop",
                    metadata={"failure_event_id": event_id},
                )
                affected_request_ids.update(
                    self._event_request_ids(cancelled_event)
                )
                if cancelled_event is not None:
                    cancelled.append(
                        {
                            "event_id": cancelled_event.event_id,
                            "event_sequence": cancelled_event.sequence,
                            "planned_completion_s": cancelled_event.timestamp_s,
                            "dp_rank": rank,
                            "pp_rank": pp_rank,
                        }
                    )
                affected_request_ids.update(
                    str(request.rid) for request in self._request_inboxes[rank]
                )
                affected_request_ids.update(
                    self._stage_owned_request_ids[key]
                )
                for envelope in self._pp_request_inboxes[key]:
                    affected_request_ids.update(
                        str(rid) for rid in envelope.request_ids
                    )
                self._runtime_command_inboxes[key].clear()
                self._pp_request_inboxes[key].clear()
                self._pp_proxy_inboxes[key].clear()
                self._pp_output_inboxes[key].clear()
                self._stage_idle[key] = True
                self._stage_blocked[key] = False
                self._stage_buffered_work[key] = False
                self._stage_next_sequence[key] = 0
                self._stage_owned_request_ids[key].clear()
                terminated.append(self._terminate_stage_worker(key))
            self._request_inboxes[rank].clear()
            self.dp_controller.mark_rank_hard_failed(rank)
            self._hard_failed_dp_ranks.add(rank)
        self._hicache_failure_teardown_stages.difference_update(
            key for key in self._stage_keys if key[0] in set(ranks)
        )

        physical_lifecycle: dict[str, Any] | None = None
        if self.resource_bridge is not None:
            if node_id is not None:
                physical_lifecycle = self.resource_bridge.fail_node(
                    node_id,
                    reason="offline_node_fail_stop",
                    event_metadata={"failure_event_id": event_id},
                )
            else:
                per_rank = [
                    self.resource_bridge.fail_dp_rank(
                        rank,
                        reason="offline_dp_fail_stop",
                        event_metadata={"failure_event_id": event_id},
                    )
                    for rank in ranks
                ]
                physical_lifecycle = {"per_rank": per_rank}
        invalidation = self.dp_controller.invalidate_aegis_physical_failure(
            failed_dp_ranks=ranks,
            failed_node_ids=(
                () if controller_node_id is None else (controller_node_id,)
            ),
            reason=f"offline_fail_stop:{event_id}",
        )
        router_output_boundaries: list[dict[str, Any]] = []
        for rid in affected_request_ids:
            request = self._request_catalog.get(rid)
            if request is None:
                continue
            if not self.router_output_ledger.has_execution(rid):
                # Compatibility for low-level tests that explicitly seed the
                # parent catalog/stage-ownership boundary without calling
                # dispatch().  The ownership evidence makes this request
                # authoritative; recover its Router registration before the
                # failure fence rather than silently omitting the boundary.
                self.router_output_ledger.register_dispatch(request)
                request_sim = request.sampling_params.custom_params.setdefault(
                    "simulation", {}
                )
                request_sim["aegis_router_registration_recovered_at_failure"] = True
                self.router_output_ledger.set_assignment(
                    rid,
                    dp_rank=int(request_sim.get("force_dp_rank", ranks[0])),
                    pp_size=self.pp_size,
                    logical_tp_size=self.tp_size,
                    configured_pp_size=self.configured_pp_size,
                    logical_pp=self.logical_pp,
                )
            router_output_boundaries.append(
                self.router_output_ledger.fence_execution(
                    rid,
                    event_id=event_id,
                )
            )
            sim = request.sampling_params.custom_params.setdefault(
                "simulation", {}
            )
            sim["aegis_hard_failure_injected"] = True
            sim["failed_dp_rank"] = int(ranks[0])
            sim["failure_event_id"] = event_id

        record = {
            **payload,
            "event_id": event_id,
            "failure_mode": "hard_fail_stop",
            "affected_dp_ranks": list(ranks),
            "injected_offset_s": event.timestamp_s,
            "cancelled_completions": cancelled,
            "terminated_stages": terminated,
            "affected_request_ids": sorted(affected_request_ids),
            "fenced_pending_recovery_admissions": sorted(
                fenced_pending_admissions
            ),
            "physical_lifecycle": physical_lifecycle,
            "ordinary_hicache_failure": ordinary_hicache_failure,
            "controller_invalidation": invalidation,
            "router_output_boundaries": sorted(
                router_output_boundaries,
                key=lambda item: (
                    str(item["logical_request_id"]),
                    str(item["failed_execution_id"]),
                ),
            ),
        }
        self._active_failures[event_id] = record
        self.coordinator.schedule_at(
            float(event.timestamp_s) + duration,
            kind="hard_failure_repair",
            priority=OfflineEventPriority.FAILURE_CONTROL,
            payload={
                "event_id": event_id,
                "duration_s": duration,
                "failure_mode": "hard_fail_stop",
                "failure_scope": payload.get("failure_scope"),
                "dp_rank": payload.get("dp_rank"),
                "node_id": payload.get("node_id"),
            },
        )

    def _repair_hard_failure(self, event: OfflineEvent) -> None:
        payload = dict(event.payload or {})
        event_id = str(payload.get("event_id") or f"failure-{event.sequence}")
        record = self._active_failures.get(event_id)
        if not isinstance(record, dict):
            raise RuntimeError(f"hard repair precedes injection for {event_id}")
        ranks = tuple(int(rank) for rank in record["affected_dp_ranks"])
        node_id = str(record.get("node_id") or "") or None
        if node_id is not None:
            physical_repair = self.resource_bridge.repair_node(
                node_id,
                reason="offline_node_repair",
                event_metadata={"failure_event_id": event_id},
            )
        elif self.resource_bridge is not None:
            physical_repair = {
                "per_rank": [
                    self.resource_bridge.repair_dp_rank(
                        rank,
                        reason="offline_dp_repair",
                        event_metadata={"failure_event_id": event_id},
                    )
                    for rank in ranks
                ]
            }
        else:
            physical_repair = None

        restarted: list[dict[str, Any]] = []
        replayed_reservations: list[dict[str, Any]] = []
        for rank in ranks:
            for pp_rank in range(self.pp_size):
                restarted.append(
                    self._spawn_stage_worker(
                        self._stage_key(rank, pp_rank, 0)
                    )
                )
            self._replace_worker_handles_after_restart(rank)
            if not self.dp_controller.mark_rank_restarted(rank):
                raise RuntimeError(f"DP rank {rank} was not hard-failed at repair")
            self.dp_controller.replay_current_dp_budget(rank)
            if self.aegis_transfer_bridge is not None:
                replayed_reservations.extend(
                    self.dp_controller.replay_aegis_reservations_after_restart(
                        rank
                    )
                )
            self._hard_failed_dp_ranks.remove(rank)

        deferred_commit_retries = (
            self._retry_deferred_physical_aegis_commits(ranks)
            if self.aegis_transfer_bridge is not None
            else []
        )

        completed = self._active_failures.pop(event_id)
        completed.update(
            {
                "recovered_offset_s": event.timestamp_s,
                "actual_outage_s": event.timestamp_s
                - float(completed["injected_offset_s"]),
                "restarted_stages": restarted,
                "physical_repair": physical_repair,
                "replayed_aegis_reservations": replayed_reservations,
                "deferred_physical_commit_retries": deferred_commit_retries,
                "repair_restored_empty_volatile_state": True,
            }
        )
        self.failure_events.append(completed)
        self._completed_failure_ids.add(event_id)

    def schedule_soft_failure(
        self,
        *,
        event_id: str,
        dp_rank: int,
        offset_s: float,
        duration_s: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        resolved_id = str(event_id)
        if resolved_id in self._scheduled_failure_ids:
            raise ValueError(f"duplicate offline failure event_id={resolved_id}")
        if float(duration_s) <= 0.0:
            raise ValueError("failure duration must be positive")
        self._scheduled_failure_ids.add(resolved_id)
        # Recovery is scheduled only after a successful inject so dense
        # manifests can retry inject without orphan recovery events.
        common = {
            "event_id": resolved_id,
            "duration_s": float(duration_s),
            **dict(payload or {}),
        }
        self.coordinator.schedule_at(
            float(offset_s),
            kind="failure_inject",
            priority=OfflineEventPriority.FAILURE_CONTROL,
            dp_rank=int(dp_rank),
            payload=common,
        )

    def _inject_soft_failure(self, event: OfflineEvent) -> None:
        if event.dp_rank is None:
            raise ValueError("failure event has no DP rank")
        payload = dict(event.payload or {})
        event_id = str(payload.get("event_id") or f"failure-{event.sequence}")
        duration_s = float(payload.get("duration_s", 0.0))
        if duration_s <= 0.0:
            raise ValueError(f"failure {event_id} missing positive duration_s")
        accepted = self.dp_controller.soft_fail_rank(event.dp_rank)
        if not accepted:
            # Drain pending L1 clears before deciding to defer/skip. Dense
            # manifests otherwise spin forever on inject-only heap events.
            self.pump_ready_stages()
            self.drain_control_plane()
            accepted = self.dp_controller.soft_fail_rank(event.dp_rank)
        if not accepted:
            retry_count = int(payload.get("_inject_retry", 0))
            if retry_count >= 40:
                logger.warning(
                    "Skipping offline soft failure %s for DP rank %s after "
                    "%s deferred injects; keeping one healthy replica",
                    event_id,
                    event.dp_rank,
                    retry_count,
                )
                # Treat skipped injects as completed so run_until_complete does
                # not stall waiting for a recovery that will never be scheduled.
                self._completed_failure_ids.add(event_id)
                self.failure_events.append(
                    {
                        **payload,
                        "event_id": event_id,
                        "dp_rank": event.dp_rank,
                        "injected_offset_s": event.timestamp_s,
                        "skipped": True,
                        "skip_reason": "no_healthy_replica_available",
                    }
                )
                return
            self.coordinator.schedule_at(
                float(event.timestamp_s) + 10.0,
                kind="failure_inject",
                priority=OfflineEventPriority.FAILURE_CONTROL,
                dp_rank=int(event.dp_rank),
                payload={**payload, "_inject_retry": retry_count + 1},
            )
            return
        ordinary_hicache_aborted = 0
        if self.hicache_transfer_bridge is not None:
            aborted_ids: set[str] = set()
            for spec in self.hicache_transfer_bridge.topology.gpus:
                if spec.gpu.dp_rank != int(event.dp_rank):
                    continue
                before = {
                    intent_id
                    for intent_id, group in self.hicache_transfer_bridge.groups.items()
                    if group.state
                    in {
                        OrdinaryHiCacheTransferState.PENDING,
                        OrdinaryHiCacheTransferState.IN_FLIGHT,
                    }
                }
                self.hicache_transfer_bridge.fail_gpu(
                    spec.gpu,
                    reason=f"soft_gpu_failure:{event_id}",
                )
                aborted_ids.update(
                    intent_id
                    for intent_id in before
                    if self.hicache_transfer_bridge.groups[intent_id].state
                    is OrdinaryHiCacheTransferState.ABORTED
                )
            ordinary_hicache_aborted = len(aborted_ids)
        record = {
            **payload,
            "event_id": event_id,
            "dp_rank": event.dp_rank,
            "injected_offset_s": event.timestamp_s,
            "ordinary_hicache_aborted_group_count": ordinary_hicache_aborted,
        }
        self._active_failures[event_id] = record
        self.coordinator.schedule_at(
            float(event.timestamp_s) + duration_s,
            kind="failure_recovery_requested",
            priority=OfflineEventPriority.FAILURE_CONTROL,
            dp_rank=int(event.dp_rank),
            payload={
                key: value
                for key, value in record.items()
                if key != "injected_offset_s"
            },
        )

    def _request_soft_recovery(self, event: OfflineEvent) -> None:
        if event.dp_rank is None:
            raise ValueError("recovery event has no DP rank")
        payload = dict(event.payload or {})
        event_id = str(payload.get("event_id") or f"failure-{event.sequence}")
        if event_id not in self._active_failures:
            raise RuntimeError(f"recovery requested before injection for {event_id}")
        request_id = f"offline-l1-clear-{event_id}-{event.dp_rank}"
        resolved_id, waiter = self.dp_controller.begin_rank_l1_clear(
            event.dp_rank,
            request_id=request_id,
        )
        self._pending_recoveries[resolved_id] = {
            "event_id": event_id,
            "dp_rank": event.dp_rank,
            "requested_at_s": event.timestamp_s,
            "waiter": waiter,
        }

    def _finish_ready_recoveries(self) -> int:
        finished = 0
        for request_id, pending in list(self._pending_recoveries.items()):
            waiter = pending["waiter"]
            if not waiter.is_set():
                continue
            result = self.dp_controller.finish_rank_l1_clear(request_id)
            if result is None:
                continue
            event_id = str(pending["event_id"])
            dp_rank = int(pending["dp_rank"])
            if not self.dp_controller.recover_soft_failed_rank(dp_rank):
                raise RuntimeError(f"offline recovery release failed for {event_id}")
            record = self._active_failures.pop(event_id)
            recovered_at = max(
                self.coordinator.now_s,
                float(result.get("cleared_time", self.coordinator.now_s)),
            )
            record.update(
                {
                    "recovery_requested_offset_s": float(
                        pending["requested_at_s"]
                    ),
                    "recovered_offset_s": recovered_at,
                    "actual_outage_s": recovered_at
                    - float(record["injected_offset_s"]),
                    "recovery_control_latency_s": recovered_at
                    - float(pending["requested_at_s"]),
                    "l1_clear": result,
                }
            )
            self.failure_events.append(record)
            self._completed_failure_ids.add(event_id)
            del self._pending_recoveries[request_id]
            finished += 1
        return finished

    def _force_complete_pending_failures(self) -> int:
        """Release stranded soft failures after the request heap is empty."""

        hard_failures = sorted(
            event_id
            for event_id, record in self._active_failures.items()
            if str(record.get("failure_mode") or "") == "hard_fail_stop"
        )
        if hard_failures:
            raise RuntimeError(
                "hard fail-stop lost its scheduled repair event: "
                f"{hard_failures}"
            )
        finished = self._finish_ready_recoveries()
        for request_id, pending in list(self._pending_recoveries.items()):
            event_id = str(pending["event_id"])
            dp_rank = int(pending["dp_rank"])
            result = self.dp_controller.finish_rank_l1_clear(request_id)
            if result is None:
                result = {
                    "request_id": request_id,
                    "dp_rank": dp_rank,
                    "forced": True,
                    "cleared_time": self.coordinator.now_s,
                }
            if not self.dp_controller.recover_soft_failed_rank(dp_rank):
                # Refcount may already be zero if a prior forced release raced.
                logger.warning(
                    "Forced recovery for %s found no soft-fail ref on DP %s",
                    event_id,
                    dp_rank,
                )
            record = self._active_failures.pop(event_id, {
                "event_id": event_id,
                "dp_rank": dp_rank,
                "injected_offset_s": float(pending["requested_at_s"]),
            })
            recovered_at = float(self.coordinator.now_s)
            record.update(
                {
                    "recovery_requested_offset_s": float(
                        pending["requested_at_s"]
                    ),
                    "recovered_offset_s": recovered_at,
                    "actual_outage_s": recovered_at
                    - float(record.get("injected_offset_s", recovered_at)),
                    "recovery_control_latency_s": recovered_at
                    - float(pending["requested_at_s"]),
                    "l1_clear": result,
                    "forced_complete": True,
                }
            )
            self.failure_events.append(record)
            self._completed_failure_ids.add(event_id)
            del self._pending_recoveries[request_id]
            finished += 1
        for event_id, record in list(self._active_failures.items()):
            dp_rank = int(record.get("dp_rank", -1))
            if dp_rank >= 0:
                self.dp_controller.recover_soft_failed_rank(dp_rank)
            record = dict(record)
            record["forced_complete"] = True
            record["recovered_offset_s"] = float(self.coordinator.now_s)
            self.failure_events.append(record)
            self._completed_failure_ids.add(str(event_id))
            del self._active_failures[event_id]
            finished += 1
        return finished

    @staticmethod
    def _stage_key(
        dp_rank: int,
        pp_rank: int,
        tp_rank: int = 0,
    ) -> tuple[int, int, int]:
        return int(dp_rank), int(pp_rank), int(tp_rank)

    def _sync_stage_driver(self, key: tuple[int, int, int], response: dict) -> None:
        worker_now = float(response.get("logical_now_s", 0.0))
        driver = self.coordinator.stage_driver(key[0], key[1])
        if worker_now > driver.now():
            driver.set_time(worker_now)

    def _take_runtime_commands(self, key: tuple[int, int, int]) -> list[Any]:
        commands = list(self._runtime_command_inboxes[key])
        self._runtime_command_inboxes[key].clear()
        return commands

    def _take_hicache_acks(
        self, key: tuple[int, int, int]
    ) -> list[dict[str, Any]]:
        acknowledgements = list(self._hicache_ack_inboxes[key])
        self._hicache_ack_inboxes[key].clear()
        return acknowledgements

    def _take_output_messages(self, key: tuple[int, int, int]) -> list[Any]:
        messages = list(self._pp_output_inboxes[key])
        self._pp_output_inboxes[key].clear()
        return messages

    def _stage_dependencies_ready(self, key: tuple[int, int, int]) -> bool:
        dp_rank, pp_rank, _ = key
        if dp_rank in self._hard_failed_dp_ranks:
            return False
        if self._stage_buffered_work[key]:
            return True
        if pp_rank == 0:
            return bool(self._request_inboxes[dp_rank]) or not self._stage_idle[key]
        request_messages = self._pp_request_inboxes[key]
        proxy_messages = self._pp_proxy_inboxes[key]
        if not request_messages or not proxy_messages:
            return False
        expected = self._stage_next_sequence[key]
        request_sequence = int(request_messages[0].step_sequence)
        proxy_sequence = int(proxy_messages[0].step_sequence)
        if request_sequence != expected or proxy_sequence != expected:
            raise RuntimeError(
                "offline PP dependency sequence mismatch at "
                f"stage={key}: expected={expected}, "
                f"request={request_sequence}, proxy={proxy_sequence}"
            )
        return True

    def _poll_stage(self, key: tuple[int, int, int]) -> bool:
        """Apply boundary-visible commands/output without starting a step."""

        if key[0] in self._hard_failed_dp_ranks:
            return False
        if self.coordinator.stage_is_busy(key[0], key[1]):
            return False
        if key in self._hicache_blocked_prepared:
            # The worker owns a prepared step and accepts its ACKs only in the
            # eventual complete command.  Calling poll here would violate the
            # worker's one-pending-step invariant.
            return False
        runtime_commands = self._take_runtime_commands(key)
        output_messages = self._take_output_messages(key)
        hicache_acks = self._take_hicache_acks(key)
        if not runtime_commands and not output_messages and not hicache_acks:
            return False
        self.coordinator.activate_idle_stage(key[0], key[1])
        response = self._call(
            key[0],
            key[1],
            "poll",
            {
                "at_s": self.coordinator.stage_driver(key[0], key[1]).now(),
                "runtime_commands": runtime_commands,
                "pp_output_messages": output_messages,
                "hicache_acks": hicache_acks,
            },
            tp_rank=key[2],
        )
        self._ingest_worker_response(response)
        self._sync_stage_driver(key, response)
        if output_messages:
            # The worker pumped every available output before replying.  If its
            # next slot remains blocked, a later prepare response will say so.
            self._stage_blocked[key] = False
        return True

    def _drain_stage_boundaries(self) -> int:
        processed = 0
        limit = max(len(self._stage_keys) * (self.pp_size + 2), 16)
        for _ in range(limit):
            progress = False
            for key in self._stage_keys:
                if self._poll_stage(key):
                    processed += 1
                    progress = True
            if not progress:
                break
        else:
            raise RuntimeError("offline PP boundary routing failed to quiesce")
        self._finish_ready_recoveries()
        return processed

    def prepare_stage(
        self,
        dp_rank: int,
        pp_rank: int,
        *,
        tp_rank: int = 0,
    ) -> OfflineEvent | None:
        key = self._stage_key(dp_rank, pp_rank, tp_rank)
        if key[0] in self._hard_failed_dp_ranks:
            return None
        if self.coordinator.stage_is_busy(key[0], key[1]):
            return None
        if key in self._hicache_blocked_prepared:
            return None
        if self._stage_blocked[key] and not self._pp_output_inboxes[key]:
            return None
        if not self._stage_dependencies_ready(key):
            return None

        self.coordinator.activate_idle_stage(key[0], key[1])
        payload: dict[str, Any] = {
            "at_s": self.coordinator.stage_driver(key[0], key[1]).now(),
            "runtime_commands": self._take_runtime_commands(key),
            "pp_output_messages": self._take_output_messages(key),
            "hicache_acks": self._take_hicache_acks(key),
        }
        supplied_work = False
        if not self._stage_buffered_work[key]:
            if key[1] == 0:
                incoming = list(self._request_inboxes[key[0]])
                self._request_inboxes[key[0]].clear()
                payload["incoming_requests"] = incoming
                supplied_work = bool(incoming)
                self._stage_owned_request_ids[key].update(
                    str(request.rid) for request in incoming
                )
            else:
                request_message = self._pp_request_inboxes[key].pop(0)
                payload["pp_request_messages"] = [request_message]
                payload["pp_proxy_messages"] = [
                    self._pp_proxy_inboxes[key].pop(0)
                ]
                self._stage_owned_request_ids[key].update(
                    str(rid) for rid in request_message.request_ids
                )
                supplied_work = True

        response = self._call(
            key[0],
            key[1],
            "prepare",
            payload,
            tp_rank=key[2],
        )
        self._ingest_worker_response(response)
        self._sync_stage_driver(key, response)
        if bool(response.get("idle", False)):
            if bool(response.get("blocked_on_slot", False)) and supplied_work:
                self._stage_buffered_work[key] = True
            return None
        self._stage_buffered_work[key] = False
        if self._register_prepared_hicache_gate(key, response):
            return None
        return self.coordinator.schedule_stage_completion(
            key[0],
            key[1],
            duration_s=float(response.get("duration_s", 0.0)),
            payload={"prepared": response, "tp_rank": key[2]},
        )

    def prepare_rank(self, dp_rank: int) -> OfflineEvent | None:
        """Compatibility wrapper for PP0 of one DP replica."""

        return self.prepare_stage(dp_rank, 0)

    def complete_stage(self, event: OfflineEvent) -> dict:
        if (
            event.kind != "rank_completion"
            or event.dp_rank is None
            or event.pp_rank is None
        ):
            raise ValueError(
                "complete_stage requires a popped stage completion event"
            )
        tp_rank = int((event.payload or {}).get("tp_rank", 0))
        key = self._stage_key(event.dp_rank, event.pp_rank, tp_rank)
        dependency_delay_s = float(
            (event.payload or {}).get("hicache_dependency_delay_s", 0.0)
        )
        response = self._call(
            key[0],
            key[1],
            "complete",
            {
                "completed_at": event.timestamp_s,
                "runtime_commands": self._take_runtime_commands(key),
                "pp_output_messages": self._take_output_messages(key),
                "hicache_acks": self._take_hicache_acks(key),
                "hicache_dependency_delay_s": dependency_delay_s,
            },
            tp_rank=key[2],
        )
        self._ingest_worker_response(response)
        blocked = self._hicache_blocked_prepared.get(key)
        if blocked is not None:
            expected_event_id = str(blocked.get("completion_event_id") or "")
            if expected_event_id != event.event_id:
                raise RuntimeError(
                    f"ordinary HiCache stage {key} completed with stale gate event"
                )
            del self._hicache_blocked_prepared[key]
        self._sync_stage_driver(key, response)
        telemetry = dict(response.get("telemetry") or {})
        telemetry["event_sequence"] = event.sequence
        self.telemetry.append(telemetry)
        self.drain_control_plane()
        return response

    def complete_rank(self, event: OfflineEvent) -> dict:
        """Compatibility name retained for PP=1 workload drivers."""

        return self.complete_stage(event)

    def drain_control_plane(self) -> tuple[int, int]:
        aegis_count = 0
        while True:
            try:
                event = self.stats_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, dict):
                self._apply_worker_aegis_event(event)
            aegis_count += 1
        self._finish_ready_recoveries()

        while True:
            try:
                message = self.detokenizer.recv_from_scheduler.recv_pyobj(flags=1)
            except Exception:
                break
            _handle_message(
                message,
                model_path=self.server_args.model_path,
                send_to_tokenizer=self.detokenizer.send_to_tokenizer,
                send_to_scheduler_by_dp=self._detok_reply_by_dp,
                simulate_detokenizer_time=self.server_args.simulate_detokenizer_time,
            )
        detok_count = 0
        while True:
            try:
                message = self.recv_detokenizer_output.recv_pyobj(flags=1)
            except Exception:
                break
            if isinstance(message, dict):
                self.detokenizer_messages.append(message)
            detok_count += 1
        return aegis_count, detok_count

    def pump_ready_stages(self) -> int:
        """Poll boundaries, then prepare every causally ready PP stage."""

        self._drain_stage_boundaries()
        scheduled = 0
        for key in self._stage_keys:
            if self.coordinator.stage_is_busy(key[0], key[1]):
                continue
            if self.prepare_stage(
                key[0],
                key[1],
                tp_rank=key[2],
            ) is not None:
                scheduled += 1
        self.drain_control_plane()
        return scheduled

    def synchronize_dp_budget_publication(
        self,
        *,
        max_rounds: int = 128,
    ) -> dict[str, Any]:
        """Reach the same all-stage budget ACK barrier used by online startup."""

        for _ in range(max(int(max_rounds), 1)):
            self.pump_ready_stages()
            status = self.dp_controller.aegis_budget_acknowledgement_status()
            if status["all_acknowledged"]:
                return status
        raise RuntimeError(
            "offline DP x PP budget publication did not converge: "
            f"{self.dp_controller.aegis_budget_acknowledgement_status()}"
        )

    def pump_idle_ranks(self) -> int:
        """Compatibility alias for the stage-aware driver."""

        return self.pump_ready_stages()

    def process_next_event(self) -> OfflineEvent:
        """Process one parent event; workload drivers may react afterward."""

        event = self.coordinator.pop_next()
        if event.kind == "rank_completion":
            self.complete_stage(event)
            self.pump_ready_stages()
        elif event.kind == "hard_failure_inject":
            self._inject_hard_failure(event)
            self.drain_control_plane()
        elif event.kind == "hard_failure_repair":
            self._repair_hard_failure(event)
            self.pump_ready_stages()
        elif event.kind == "failure_inject":
            self._inject_soft_failure(event)
            self.drain_control_plane()
        elif event.kind == "failure_recovery_requested":
            self._request_soft_recovery(event)
            self.pump_ready_stages()
        elif event.kind == "physical_recovery_admission_gate":
            assigned = self._resolve_physical_recovery_admission(event)
            if not self.coordinator.stage_is_busy(assigned, 0):
                self.prepare_stage(assigned, 0)
            self.pump_ready_stages()
        elif event.kind == "request_arrival":
            request = event.payload
            if not isinstance(request, TokenizedGenerateReqInput):
                raise TypeError("offline request_arrival payload is not tokenized")
            assigned = self.dispatch(request)
            admission = self._physical_recovery_admissions.get(str(request.rid))
            pending_gate = bool(
                admission is not None
                and admission["state"] == "PENDING_TRANSFER"
            )
            if (
                not pending_gate
                and not self.coordinator.stage_is_busy(assigned, 0)
            ):
                self.prepare_stage(assigned, 0)
            self.pump_ready_stages()
        elif event.kind == "aegis_tier_commit_notification":
            # The GlobalDES callback only enqueues the controller's
            # tier-specific runtime command.  Apply it through the same
            # parent-ordered idle/batch boundary used by every other command.
            self.pump_ready_stages()
        elif event.kind in {
            "ordinary_hicache_transfer_start",
            "ordinary_hicache_transfer_completion",
        }:
            # Completion callbacks may have queued a nonblocking D2H ACK or
            # released an H2D admission gate.  Publish that boundary before a
            # later virtual-time event is allowed to overtake it.
            self.pump_ready_stages()
        else:
            # Coordination-only workload events are intentionally left to the
            # caller after their deterministic heap position is consumed.
            self.drain_control_plane()
        return event

    def run_until_complete(
        self,
        *,
        expected_request_ids: set[str],
        max_events: int = 1_000_000,
    ) -> OfflineClusterResult:
        expected = {str(rid) for rid in expected_request_ids}
        completed: set[str] = set()

        self.pump_ready_stages()

        for _ in range(max(int(max_events), 1)):
            for message in self.detokenizer_messages:
                if bool(message.get("finished", False)):
                    completed.add(str(message.get("rid", "")))
            if (
                expected <= completed
                and self._scheduled_failure_ids <= self._completed_failure_ids
            ):
                break
            if not self.coordinator.has_events():
                # Runtime backup commands and PP envelopes can target an
                # otherwise idle physical stage.
                self.pump_ready_stages()
                self.drain_control_plane()
                if (
                    expected <= completed
                    and self._scheduled_failure_ids <= self._completed_failure_ids
                ):
                    break
                if not self.coordinator.has_events():
                    # Dense failure manifests can leave a last L1-clear waiter
                    # stranded after the request heap drains.  Force-release so
                    # offline smoke runs terminate deterministically.
                    if self._pending_recoveries or self._active_failures:
                        self._force_complete_pending_failures()
                        continue
                    missing = sorted(expected - completed)
                    raise RuntimeError(
                        f"offline cluster stalled with incomplete requests: {missing[:8]}"
                    )
            self.process_next_event()
        else:
            raise RuntimeError("offline cluster exceeded max_events")

        # A finished PP0 detokenizer message can be emitted while later output
        # ring consumers or already-prepared bubbles remain.  Settle every
        # physical stage before taking snapshots.
        # A terminal frontier is split into the same prefix/decode chunks as
        # SlackLandlord.  Each TP-sharded chunk contributes transfer start,
        # completion, and notification events, so a fixed 128-step drain can
        # stop before CompleteRequest reaches its all-tier terminal barrier.
        queued_physical_events = len(self.global_des._events)
        settle_limit = max(
            len(self._stage_keys) * 128,
            queued_physical_events * 4 + len(self._stage_keys) * 32,
            128,
        )
        for _ in range(settle_limit):
            if self.coordinator.has_events():
                self.process_next_event()
                continue
            scheduled = self.pump_ready_stages()
            if scheduled == 0 and not self.coordinator.has_events():
                break
        else:
            raise RuntimeError("offline cluster failed to settle PP pipelines")

        stage_snapshots: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._drain_stage_boundaries()
        for key in self._stage_keys:
            # ``snapshot`` is intentionally read-only in the worker and does
            # not advance its rank-local clock.  First execute an empty poll
            # at the parent cursor so an idle rank's resource timestamp is
            # still comparable with the global physical ledger.
            aligned = self._call(
                key[0],
                key[1],
                "poll",
                {
                    "at_s": self.coordinator.now_s,
                    "runtime_commands": self._take_runtime_commands(key),
                    "pp_output_messages": self._take_output_messages(key),
                    "hicache_acks": self._take_hicache_acks(key),
                },
                tp_rank=key[2],
            )
            self._ingest_worker_response(aligned)
            self._sync_stage_driver(key, aligned)
            snapshot = self._call(
                key[0],
                key[1],
                "snapshot",
                {
                    "runtime_commands": self._take_runtime_commands(key),
                    "pp_output_messages": self._take_output_messages(key),
                    "hicache_acks": self._take_hicache_acks(key),
                },
                tp_rank=key[2],
            )
            self._ingest_worker_response(snapshot)
            self._sync_stage_driver(key, snapshot)
            stage_snapshots[key] = snapshot
        snapshots: dict[int, dict[str, Any]] = {}
        for rank in range(self.dp_size):
            stages = {
                str(pp_rank): stage_snapshots[(rank, pp_rank, 0)]
                for pp_rank in range(self.pp_size)
            }
            stable_stats = {
                rid: copy.deepcopy(stats)
                for rid, stats in self._terminal_request_stats.items()
                if int(stats.get("aegis_assigned_dp_rank", -1)) == rank
            }
            stable_stats.update(
                copy.deepcopy(dict(stages["0"].get("request_stats", {})))
            )
            stages["0"] = {
                **dict(stages["0"]),
                "request_stats": stable_stats,
            }
            leader = dict(stages["0"])
            leader["stages"] = stages
            leader["process_count"] = self.pp_size * self.physical_tp_size
            leader["dp_logical_makespan_s"] = self.coordinator.dp_makespan_s(
                rank
            )
            snapshots[rank] = leader
        self.drain_control_plane()
        aegis_status = self.dp_controller.aegis_status()
        resource_audit_errors: list[str] = []
        rank_work_barriers = self.coordinator.rank_work_barriers()
        if rank_work_barriers:
            resource_audit_errors.append(
                "offline rank work barriers remain unpublished: "
                + ", ".join(
                    barrier.barrier_id for barrier in rank_work_barriers
                )
            )
        if self.resource_bridge is None:
            resource_audit_errors.extend(self.global_des.audit_invariants())
        else:
            resource_audit_errors.extend(self.resource_bridge.audit())
        if self.aegis_transfer_bridge is not None:
            for error in self.aegis_transfer_bridge.audit_invariants():
                # GlobalDES's external-aggregate mode intentionally models
                # per-extent aliases only for primary sources.  Scheduler
                # prefix pages remain an aggregate account owned and checked
                # exactly by OfflineResourceBridge above, so they have no
                # internal KvExtent rows for the second audit to sum.
                if " prefix_kv: ledger=" in error:
                    continue
                resource_audit_errors.append(error)
        if self.hicache_transfer_bridge is not None:
            resource_audit_errors.extend(
                self.hicache_transfer_bridge.audit(require_terminal=True)
            )
        resource_audit_errors.extend(self._hicache_artifact_blockers)
        for key, acknowledgements in self._hicache_ack_inboxes.items():
            if acknowledgements:
                resource_audit_errors.append(
                    f"ordinary HiCache ACK inbox is not empty for {key}: "
                    f"{len(acknowledgements)}"
                )
        for key, blocked in self._hicache_blocked_prepared.items():
            resource_audit_errors.append(
                f"ordinary HiCache compute gate remains blocked for {key}: "
                f"{list(blocked.get('intent_ids', ())) }"
            )
        for key, snapshot in stage_snapshots.items():
            for error in snapshot.get("hicache_intent_audit", ()):
                resource_audit_errors.append(f"stage {key}: {error}")
        recovery_admission_export = self._physical_recovery_admission_export()
        for admission in recovery_admission_export["admissions"]:
            if admission.get("state") in {
                "PENDING_TRANSFER",
                "READY_FOR_STAGE_ADOPTION",
                "STAGE_ADOPTION_IN_PROGRESS",
            }:
                resource_audit_errors.append(
                    "physical recovery admission is nonterminal: "
                    f"{admission.get('request_id')}={admission.get('state')}"
                )
        for hold in recovery_admission_export["pending_holds"]:
            resource_audit_errors.append(
                "orphan RECOVERY_PENDING hold after experiment: "
                f"{hold.get('transfer_id')} for {hold.get('request_id')}"
            )
        resource_audit_errors = list(dict.fromkeys(resource_audit_errors))
        # Generic OfflineSchedulerCluster callers may intentionally exclude a
        # killed request and inspect the fenced state without scheduling a
        # replay.  The paper suite applies the stronger all-logical-complete
        # audit after its explicit replay orchestration.
        router_output_audit = self.router_output_ledger.audit(
            require_complete=False
        )
        if resource_audit_errors:
            raise RuntimeError(
                "offline physical-resource audit failed: "
                + "; ".join(resource_audit_errors)
            )
        return OfflineClusterResult(
            telemetry=list(self.telemetry),
            detokenizer_messages=list(self.detokenizer_messages),
            rank_snapshots=snapshots,
            failure_events=list(self.failure_events),
            exact_rolling_installations=list(self.exact_rolling_installations),
            aegis_status=aegis_status,
            global_des_snapshot=self.global_des.snapshot(),
            global_des_timeseries=self.global_des.export_timeseries(),
            physical_resource_bridge=(
                {}
                if self.resource_bridge is None
                else self.resource_bridge.export()
            ),
            physical_aegis_transfers=(
                {}
                if self.aegis_transfer_bridge is None
                else {
                    **self.aegis_transfer_bridge.export(),
                    "controller_dispatches": list(
                        self._physical_aegis_controller_dispatches
                    ),
                    "recovery_plans": copy.deepcopy(
                        self._physical_aegis_recovery_plans
                    ),
                    "completion_events": copy.deepcopy(
                        self._physical_aegis_completion_events
                    ),
                    "source_frontier_adjustments": copy.deepcopy(
                        self._physical_aegis_source_frontier_adjustments
                    ),
                    "recovery_admission_gate": recovery_admission_export,
                }
            ),
            physical_hicache_transfers=(
                {}
                if self.hicache_transfer_bridge is None
                else self.hicache_transfer_bridge.export(
                    require_terminal=True
                )
            ),
            resource_audit_errors=list(resource_audit_errors),
            router_output_ledger={
                **self.router_output_ledger.export(),
                "audit": router_output_audit,
            },
        )

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        controller = getattr(self, "dp_controller", None)
        if controller is not None:
            controller.close_event_journal()
        specs = list(getattr(self, "_worker_specs", ()))
        for spec, process in zip(specs, getattr(self, "processes", [])):
            if process.is_alive():
                try:
                    self._call(
                        spec.stage.dp_rank,
                        spec.stage.pp_rank,
                        "stop",
                        tp_rank=spec.stage.tp_rank,
                    )
                except Exception:
                    pass
        for process in getattr(self, "processes", []):
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        control_queues = getattr(self, "control_queues", {})
        queues = list(
            control_queues.values()
            if isinstance(control_queues, dict)
            else control_queues
        )
        queues.extend(getattr(self, "_retired_control_queues", ()))
        for name in ("result_queue", "stats_queue"):
            value = getattr(self, name, None)
            if value is not None:
                queues.append(value)
        registry = getattr(self, "ipc_registry", None)
        if registry is not None:
            queues.extend(registry._queues.values())
        seen: set[int] = set()
        for ipc_queue in queues:
            if id(ipc_queue) in seen:
                continue
            seen.add(id(ipc_queue))
            try:
                ipc_queue.close()
                ipc_queue.join_thread()
            except Exception:
                pass
        SimIpcRegistry.install_global(None)

    def __enter__(self) -> "OfflineSchedulerCluster":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
