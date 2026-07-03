from benchmark.synthesis_requests.utils import sim_stats_to_detail_row


def test_sim_stats_to_detail_row_preserves_aegis_fields():
    row = sim_stats_to_detail_row(
        "r0",
        {
            "status": "completed",
            "created_time": 1.0,
            "queue_start": 1.1,
            "queue_end": 1.2,
            "last_event_time": 2.0,
            "input_length": 100,
            "output_length": 8,
            "generated_tokens": 8,
            "cache_hit_tokens": 40,
            "load_back_tokens": 12,
            "agent_id": "agent-0",
            "critical_agent": True,
            "aegis_policy": "aegiskv",
            "aegis_gpu_budget_fraction": 0.25,
            "aegis_backup_kv": True,
            "aegis_worker_seq": 2,
            "aegis_assigned_dp_rank": 1,
            "aegis_host_visible_tokens": 108,
            "aegis_gpu_replica_tokens": 16,
            "aegis_safe_frontier_tokens": 108,
            "aegis_recompute_tokens": 0,
            "aegis_recovery_action": "gpu_replica",
            "aegis_expected_recovery_ms": 4.7,
            "aegis_candidate_retry": True,
            "aegis_candidate_failed_dp_rank": 1,
            "aegis_candidate_pre_failure_output_tokens": 4,
            "aegis_candidate_pre_failure_total_tokens": 104,
            "aegis_candidate_backed_up_tokens": 16,
            "aegis_candidate_recompute_tokens": 88,
            "aegis_candidate_recovery_action": "gpu_replica",
            "aegis_candidate_failover_overhead_s": 0.0036,
            "aegis_failover_replay": True,
            "aegis_failed_source_dp_rank": 1,
            "aegis_replay_restored_tokens": 16,
            "aegis_replay_restored_prefill_tokens": 12,
            "aegis_replay_recompute_tokens": 88,
            "aegis_replay_restore_action": "gpu_replica",
            "aegis_replay_restore_overhead_s": 0.0036,
            "aegis_replay_restore_sleep_s": 0.0037,
            "aegis_replay_restore_data_plane": "metadata_materialization",
            "aegis_replay_restore_timing_source": "physical_recovery_path_csv",
            "aegis_cross_rank_retry": True,
            "aegis_soft_failure_injected": True,
            "aegis_hard_failure_injected": True,
            "aegis_host_backup_lookup_hit": True,
            "aegis_host_backup_lookup_tokens": 16,
            "aegis_peer_gpu_lookup_hit": True,
            "aegis_peer_gpu_lookup_tokens": 8,
        },
        0.42,
    )

    assert row.agent_id == "agent-0"
    assert row.critical_agent is True
    assert row.aegis_policy == "aegiskv"
    assert row.aegis_gpu_budget_fraction == 0.25
    assert row.aegis_backup_kv is True
    assert row.aegis_worker_seq == 2
    assert row.aegis_assigned_dp_rank == 1
    assert row.aegis_host_visible_tokens == 108
    assert row.aegis_gpu_replica_tokens == 16
    assert row.aegis_safe_frontier_tokens == 108
    assert row.aegis_recompute_tokens == 0
    assert row.aegis_recovery_action == "gpu_replica"
    assert row.aegis_expected_recovery_ms == 4.7
    assert row.aegis_candidate_retry is True
    assert row.aegis_candidate_failed_dp_rank == 1
    assert row.aegis_candidate_pre_failure_output_tokens == 4
    assert row.aegis_candidate_pre_failure_total_tokens == 104
    assert row.aegis_candidate_backed_up_tokens == 16
    assert row.aegis_candidate_recompute_tokens == 88
    assert row.aegis_candidate_recovery_action == "gpu_replica"
    assert row.aegis_candidate_failover_overhead_s == 0.0036
    assert row.aegis_failover_replay is True
    assert row.aegis_failed_source_dp_rank == 1
    assert row.aegis_replay_restored_tokens == 16
    assert row.aegis_replay_restored_prefill_tokens == 12
    assert row.aegis_replay_recompute_tokens == 88
    assert row.aegis_replay_restore_action == "gpu_replica"
    assert row.aegis_replay_restore_overhead_s == 0.0036
    assert row.aegis_replay_restore_sleep_s == 0.0037
    assert row.aegis_replay_restore_data_plane == "metadata_materialization"
    assert row.aegis_replay_restore_timing_source == "physical_recovery_path_csv"
    assert row.aegis_cross_rank_retry is True
    assert row.aegis_soft_failure_injected is True
    assert row.aegis_hard_failure_injected is True
    assert row.aegis_host_backup_lookup_hit is True
    assert row.aegis_host_backup_lookup_tokens == 16
    assert row.aegis_peer_gpu_lookup_hit is True
    assert row.aegis_peer_gpu_lookup_tokens == 8
