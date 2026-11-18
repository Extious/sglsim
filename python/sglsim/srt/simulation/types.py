from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SimulationMode(str, Enum):
    STREAMING = "STREAMING"


@dataclass
class RequestStats:
    rid: str = ""
    status: str = "running"
    last_event_time: float = 0.0
    input_length: int = 0
    output_length: int = 0
    generated_tokens: int = 0
    cache_hit_tokens: int = 0
    load_back_tokens: int = 0
    prefetch_complete_tokens: int = 0
    queue_start: float = -1.0
    queue_end: float = -1.0
    created_time: float = -1.0
    service_time_s: float = 0.0
    gen_token_latencies: list[float] = field(default_factory=list)
    workflow_id: str = ""
    stage_id: int = -1
    stage_width: int = 1
    stage_local_agent_index: int = 0
    decode_dominant: bool = False
    workflow_shape: str = ""
    agent_id: str = ""
    critical_agent: bool = False
    aegis_policy: str = ""
    aegis_ablation: str = ""
    aegis_gpu_budget_fraction: float = -1.0
    aegis_failure_rate_per_window: float = -1.0
    aegis_alpha_policy_source: str = ""
    aegis_critical_pressure: float = -1.0
    aegis_slack_lcb_ms: float = 0.0
    aegis_slack_q50_ms: float = 0.0
    aegis_slack_q90_ms: float = 0.0
    aegis_slack_target_coverage: float = 0.0
    aegis_slack_calibration_id: str = ""
    aegis_slack_calibration_group: str = ""
    aegis_slack_source: str = ""
    aegis_slack_kind: str = ""
    aegis_need_state: str = ""
    aegis_slack_role: str = ""
    aegis_slack_turns_completed: int = 0
    aegis_backup_kv: bool = False
    aegis_worker_seq: int = 0
    aegis_assigned_dp_rank: int = -1
    aegis_route_plan_dp_rank: int = -1
    aegis_route_plan_applied: bool | None = None
    aegis_exact_admission_state_id: str = ""
    aegis_exact_admission_information_mode: str = ""
    aegis_exact_observed_unavailable_dp_ranks: list[int] = field(
        default_factory=list
    )
    aegis_host_visible_tokens: int = 0
    aegis_gpu_replica_tokens: int = 0
    aegis_gpu_replica_dp_rank: int = -1
    aegis_gpu_replica_quota_tokens: int = 0
    aegis_gpu_replica_overlap_tokens: int = 0
    aegis_gpu_replica_unique_tokens: int = 0
    aegis_gpu_replica_reserved_tokens: int = 0
    aegis_gpu_replica_evictable: bool = False
    aegis_safe_frontier_tokens: int = 0
    aegis_recompute_tokens: int = 0
    aegis_recovery_action: str = "none"
    aegis_expected_recovery_ms: float = 0.0
    aegis_candidate_retry: bool = False
    aegis_candidate_failed_dp_rank: int = -1
    aegis_candidate_pre_failure_output_tokens: int = 0
    aegis_candidate_pre_failure_total_tokens: int = 0
    aegis_candidate_backed_up_tokens: int = 0
    aegis_candidate_recompute_tokens: int = 0
    aegis_candidate_recovery_action: str = "none"
    aegis_candidate_failover_overhead_s: float = 0.0
    aegis_failover_replay: bool = False
    aegis_failed_source_dp_rank: int = -1
    aegis_replay_restored_tokens: int = 0
    aegis_replay_restored_prefill_tokens: int = 0
    aegis_replay_recompute_tokens: int = 0
    aegis_replay_capacity_limited_tokens: int = 0
    aegis_replay_restore_action: str = "none"
    aegis_replay_restore_overhead_s: float = 0.0
    aegis_replay_restore_sleep_s: float = 0.0
    aegis_replay_restore_data_plane: str = "none"
    aegis_replay_restore_timing_source: str = "default_coefficients"
    aegis_replay_gpu_direct_reuse: bool = False
    aegis_replay_host_restore_path: str = "none"
    aegis_replay_l3_restore_path: str = "none"
    aegis_cross_rank_retry: bool = False
    aegis_soft_failure_injected: bool = False
    aegis_hard_failure_injected: bool = False
    aegis_host_backup_lookup_hit: bool = False
    aegis_host_backup_lookup_tokens: int = 0
    aegis_l3_backup_lookup_hit: bool = False
    aegis_l3_backup_lookup_tokens: int = 0
    aegis_peer_gpu_lookup_hit: bool = False
    aegis_peer_gpu_lookup_tokens: int = 0

    def is_complete(self) -> bool:
        return self.status == "completed"

    def to_sim_stats_dict(self) -> dict:
        return {
            "rid": self.rid,
            "status": self.status,
            "created_time": self.created_time,
            "queue_start": self.queue_start,
            "queue_end": self.queue_end,
            "last_event_time": self.last_event_time,
            "service_time_s": self.service_time_s,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "generated_tokens": self.generated_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "load_back_tokens": self.load_back_tokens,
            "prefetch_complete_tokens": self.prefetch_complete_tokens,
            "workflow_id": self.workflow_id,
            "stage_id": self.stage_id,
            "stage_width": self.stage_width,
            "stage_local_agent_index": self.stage_local_agent_index,
            "decode_dominant": self.decode_dominant,
            "workflow_shape": self.workflow_shape,
            "gen_token_latencies": list(self.gen_token_latencies),
            "agent_id": self.agent_id,
            "critical_agent": self.critical_agent,
            "aegis_policy": self.aegis_policy,
            "aegis_ablation": self.aegis_ablation,
            "aegis_gpu_budget_fraction": self.aegis_gpu_budget_fraction,
            "aegis_failure_rate_per_window": self.aegis_failure_rate_per_window,
            "aegis_alpha_policy_source": self.aegis_alpha_policy_source,
            "aegis_critical_pressure": self.aegis_critical_pressure,
            "aegis_slack_lcb_ms": self.aegis_slack_lcb_ms,
            "aegis_slack_q50_ms": self.aegis_slack_q50_ms,
            "aegis_slack_q90_ms": self.aegis_slack_q90_ms,
            "aegis_slack_target_coverage": self.aegis_slack_target_coverage,
            "aegis_slack_calibration_id": self.aegis_slack_calibration_id,
            "aegis_slack_calibration_group": self.aegis_slack_calibration_group,
            "aegis_slack_source": self.aegis_slack_source,
            "aegis_slack_kind": self.aegis_slack_kind,
            "aegis_need_state": self.aegis_need_state,
            "aegis_slack_role": self.aegis_slack_role,
            "aegis_slack_turns_completed": self.aegis_slack_turns_completed,
            "aegis_backup_kv": self.aegis_backup_kv,
            "aegis_worker_seq": self.aegis_worker_seq,
            "aegis_assigned_dp_rank": self.aegis_assigned_dp_rank,
            "aegis_route_plan_dp_rank": self.aegis_route_plan_dp_rank,
            "aegis_route_plan_applied": self.aegis_route_plan_applied,
            "aegis_exact_admission_state_id": (
                self.aegis_exact_admission_state_id
            ),
            "aegis_exact_admission_information_mode": (
                self.aegis_exact_admission_information_mode
            ),
            "aegis_exact_observed_unavailable_dp_ranks": list(
                self.aegis_exact_observed_unavailable_dp_ranks
            ),
            "aegis_host_visible_tokens": self.aegis_host_visible_tokens,
            "aegis_gpu_replica_tokens": self.aegis_gpu_replica_tokens,
            "aegis_gpu_replica_dp_rank": self.aegis_gpu_replica_dp_rank,
            "aegis_gpu_replica_quota_tokens": self.aegis_gpu_replica_quota_tokens,
            "aegis_gpu_replica_overlap_tokens": self.aegis_gpu_replica_overlap_tokens,
            "aegis_gpu_replica_unique_tokens": self.aegis_gpu_replica_unique_tokens,
            "aegis_gpu_replica_reserved_tokens": self.aegis_gpu_replica_reserved_tokens,
            "aegis_gpu_replica_evictable": self.aegis_gpu_replica_evictable,
            "aegis_safe_frontier_tokens": self.aegis_safe_frontier_tokens,
            "aegis_recompute_tokens": self.aegis_recompute_tokens,
            "aegis_recovery_action": self.aegis_recovery_action,
            "aegis_expected_recovery_ms": self.aegis_expected_recovery_ms,
            "aegis_candidate_retry": self.aegis_candidate_retry,
            "aegis_candidate_failed_dp_rank": self.aegis_candidate_failed_dp_rank,
            "aegis_candidate_pre_failure_output_tokens": self.aegis_candidate_pre_failure_output_tokens,
            "aegis_candidate_pre_failure_total_tokens": self.aegis_candidate_pre_failure_total_tokens,
            "aegis_candidate_backed_up_tokens": self.aegis_candidate_backed_up_tokens,
            "aegis_candidate_recompute_tokens": self.aegis_candidate_recompute_tokens,
            "aegis_candidate_recovery_action": self.aegis_candidate_recovery_action,
            "aegis_candidate_failover_overhead_s": self.aegis_candidate_failover_overhead_s,
            "aegis_failover_replay": self.aegis_failover_replay,
            "aegis_failed_source_dp_rank": self.aegis_failed_source_dp_rank,
            "aegis_replay_restored_tokens": self.aegis_replay_restored_tokens,
            "aegis_replay_restored_prefill_tokens": self.aegis_replay_restored_prefill_tokens,
            "aegis_replay_recompute_tokens": self.aegis_replay_recompute_tokens,
            "aegis_replay_capacity_limited_tokens": self.aegis_replay_capacity_limited_tokens,
            "aegis_replay_restore_action": self.aegis_replay_restore_action,
            "aegis_replay_restore_overhead_s": self.aegis_replay_restore_overhead_s,
            "aegis_replay_restore_sleep_s": self.aegis_replay_restore_sleep_s,
            "aegis_replay_restore_data_plane": self.aegis_replay_restore_data_plane,
            "aegis_replay_restore_timing_source": self.aegis_replay_restore_timing_source,
            "aegis_replay_gpu_direct_reuse": self.aegis_replay_gpu_direct_reuse,
            "aegis_replay_host_restore_path": self.aegis_replay_host_restore_path,
            "aegis_replay_l3_restore_path": self.aegis_replay_l3_restore_path,
            "aegis_cross_rank_retry": self.aegis_cross_rank_retry,
            "aegis_soft_failure_injected": self.aegis_soft_failure_injected,
            "aegis_hard_failure_injected": self.aegis_hard_failure_injected,
            "aegis_host_backup_lookup_hit": self.aegis_host_backup_lookup_hit,
            "aegis_host_backup_lookup_tokens": self.aegis_host_backup_lookup_tokens,
            "aegis_l3_backup_lookup_hit": self.aegis_l3_backup_lookup_hit,
            "aegis_l3_backup_lookup_tokens": self.aegis_l3_backup_lookup_tokens,
            "aegis_peer_gpu_lookup_hit": self.aegis_peer_gpu_lookup_hit,
            "aegis_peer_gpu_lookup_tokens": self.aegis_peer_gpu_lookup_tokens,
        }
