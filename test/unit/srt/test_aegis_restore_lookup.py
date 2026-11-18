import threading

from sglsim.srt.config import CacheConfig
from sglsim.srt.entrypoints.engine import AegisHostBackupEntry, Engine
from sglsim.srt.managers.aegis_gpu_replica import GpuReplicaReservation
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.simulation.types import RequestStats


def test_aegis_restore_lookup_chooses_peer_gpu_when_lower_envelope_wins():
    engine = Engine.__new__(Engine)
    engine.dp_size = 2
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {
        ("agent-0", 0, 0): AegisHostBackupEntry(
            rid="0-t0",
            agent_id="agent-0",
            worker_seq=0,
            source_dp_rank=0,
            peer_dp_rank=1,
            host_visible_tokens=100,
            gpu_replica_tokens=12,
            input_length=96,
            generated_tokens=8,
            updated_time=1.0,
        )
    }
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 1,
            "pre_failure_total_tokens": 100,
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["host_backup_lookup_hit"] is True
    assert sim["host_backup_lookup_tokens"] == 100
    assert sim["peer_gpu_lookup_hit"] is True
    assert sim["peer_gpu_lookup_tokens"] == 12
    assert sim["restore_action"] == "gpu_replica"
    assert sim["restore_tokens"] == 12
    assert sim["restore_recompute_tokens"] == 88


def test_engine_marks_gpu_replica_backup_evictable_on_completion():
    class FakeDpController:
        def release_request(self, rid: str):
            assert rid == "0-t0"
            return GpuReplicaReservation(
                rid=rid,
                source_dp_rank=0,
                backup_dp_rank=2,
                cache_key="agent:a",
                requested_tokens=100,
                quota_tokens=30,
                overlap_tokens=12,
                unique_tokens=18,
                reserved_tokens=30,
                alpha=0.3,
                evictable=True,
            )

    engine = Engine.__new__(Engine)
    engine.dp_size = 3
    engine._dp_controller = FakeDpController()
    engine._completion_lock = threading.Lock()
    engine._completion_events = {}
    engine._completion_text = {}
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {}
    REQUEST_STATS.clear()

    engine._apply_detokenizer_message(
        {
            "rid": "0-t0",
            "finished": True,
            "status": "completed",
            "agent_id": "agent-a",
            "aegis_worker_seq": 0,
            "aegis_assigned_dp_rank": 0,
            "aegis_host_visible_tokens": 100,
            "aegis_gpu_replica_tokens": 30,
        }
    )

    stats = REQUEST_STATS["0-t0"]
    assert stats.aegis_gpu_replica_dp_rank == 2
    assert stats.aegis_gpu_replica_evictable is True
    entry = engine._aegis_host_backup[("agent-a", 0, 0)]
    assert entry.peer_dp_rank == 2
    assert entry.gpu_replica_evictable is True


def test_aegis_restore_lookup_caps_peer_frontier_by_gpu_budget():
    engine = Engine.__new__(Engine)
    engine.dp_size = 2
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {
        ("agent-0", 0, 0): AegisHostBackupEntry(
            rid="0-t0",
            agent_id="agent-0",
            worker_seq=0,
            source_dp_rank=0,
            peer_dp_rank=1,
            host_visible_tokens=100,
            gpu_replica_tokens=40,
            input_length=96,
            generated_tokens=8,
            updated_time=1.0,
        )
    }
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 1,
            "pre_failure_total_tokens": 100,
            "aegis_gpu_budget_fraction": 0.05,
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["peer_gpu_lookup_hit"] is True
    assert sim["peer_gpu_lookup_tokens"] == 5
    assert sim["restore_action"] == "gpu_replica"
    assert sim["restore_tokens"] == 5


def test_aegis_gpu_only_policy_records_peer_frontier_without_host_frontier():
    engine = Engine.__new__(Engine)
    engine.dp_size = 2
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {}
    stats = RequestStats(
        rid="0-t0",
        status="completed",
        agent_id="agent-0",
        aegis_worker_seq=0,
        aegis_assigned_dp_rank=0,
        aegis_host_visible_tokens=0,
        aegis_gpu_replica_tokens=12,
        input_length=96,
        generated_tokens=8,
    )

    engine._record_aegis_host_backup(stats)
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 1,
            "pre_failure_total_tokens": 100,
            "aegis_policy": "gpu_replica",
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["host_backup_lookup_hit"] is False
    assert sim["host_backup_lookup_tokens"] == 0
    assert sim["peer_gpu_lookup_hit"] is True
    assert sim["peer_gpu_lookup_tokens"] == 12
    assert sim["restore_action"] == "gpu_replica"
    assert sim["restore_tokens"] == 12


def test_aegis_restore_lookup_recomputes_when_host_backup_missing():
    engine = Engine.__new__(Engine)
    engine.dp_size = 2
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {}
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 1,
            "pre_failure_total_tokens": 100,
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["host_backup_lookup_hit"] is False
    assert sim["host_backup_lookup_tokens"] == 0
    assert sim["peer_gpu_lookup_hit"] is False
    assert sim["peer_gpu_lookup_tokens"] == 0
    assert sim["restore_action"] == "recompute"
    assert sim["restore_tokens"] == 0
    assert sim["restore_recompute_tokens"] == 100


def test_aegis_restore_lookup_demotes_peer_gpu_when_retry_rank_is_not_peer():
    engine = Engine.__new__(Engine)
    engine.dp_size = 3
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {
        ("agent-0", 0, 0): AegisHostBackupEntry(
            rid="0-t0",
            agent_id="agent-0",
            worker_seq=0,
            source_dp_rank=0,
            peer_dp_rank=1,
            host_visible_tokens=1000,
            gpu_replica_tokens=1000,
            input_length=996,
            generated_tokens=8,
            updated_time=1.0,
        )
    }
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 2,
            "pre_failure_total_tokens": 1000,
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["host_backup_lookup_hit"] is True
    assert sim["host_backup_lookup_tokens"] == 1000
    assert sim["peer_gpu_lookup_hit"] is False
    assert sim["peer_gpu_lookup_tokens"] == 0
    assert sim["restore_action"] == "host_restore"
    assert sim["restore_tokens"] == 1000


def test_aegis_no_lifecycle_ablation_forces_recompute_despite_frontier():
    engine = Engine.__new__(Engine)
    engine.dp_size = 2
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {
        ("agent-0", 0, 0): AegisHostBackupEntry(
            rid="0-t0",
            agent_id="agent-0",
            worker_seq=0,
            source_dp_rank=0,
            peer_dp_rank=1,
            host_visible_tokens=100,
            gpu_replica_tokens=12,
            input_length=96,
            generated_tokens=8,
            updated_time=1.0,
        )
    }
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 1,
            "pre_failure_total_tokens": 100,
            "aegis_ablation": "no_lifecycle",
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["host_backup_lookup_hit"] is False
    assert sim["peer_gpu_lookup_hit"] is False
    assert sim["restore_action"] == "recompute"
    assert sim["restore_tokens"] == 0


def test_aegis_no_restore_recompute_ablation_chooses_restore_first():
    engine = Engine.__new__(Engine)
    engine.dp_size = 2
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {
        ("agent-0", 0, 0): AegisHostBackupEntry(
            rid="0-t0",
            agent_id="agent-0",
            worker_seq=0,
            source_dp_rank=0,
            peer_dp_rank=1,
            host_visible_tokens=100,
            gpu_replica_tokens=12,
            input_length=96,
            generated_tokens=8,
            updated_time=1.0,
        )
    }
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 1,
            "pre_failure_total_tokens": 100,
            "aegis_ablation": "no_restore_recompute",
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["host_backup_lookup_hit"] is True
    assert sim["peer_gpu_lookup_hit"] is True
    assert sim["restore_action"] == "host_restore"
    assert sim["restore_tokens"] == 100


def test_aegis_restore_lookup_recomputes_when_host_evicted_and_peer_unavailable():
    engine = Engine.__new__(Engine)
    engine.dp_size = 3
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {
        ("agent-0", 0, 0): AegisHostBackupEntry(
            rid="0-t0",
            agent_id="agent-0",
            worker_seq=0,
            source_dp_rank=0,
            peer_dp_rank=1,
            host_visible_tokens=0,
            gpu_replica_tokens=64,
            input_length=96,
            generated_tokens=8,
            updated_time=1.0,
        )
    }
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 2,
            "pre_failure_total_tokens": 64,
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["host_backup_lookup_hit"] is False
    assert sim["host_backup_lookup_tokens"] == 0
    assert sim["peer_gpu_lookup_hit"] is False
    assert sim["peer_gpu_lookup_tokens"] == 0
    assert sim["restore_action"] == "recompute"
    assert sim["restore_tokens"] == 0
    assert sim["restore_recompute_tokens"] == 64


def test_aegis_restored_prefix_is_materialized_as_device_cache():
    cache = KVCachePool(
        CacheConfig(page_size=4, l1_capacity_pages=8, l2_capacity_pages=0)
    )
    req = Req(
        "agent-0-t0-retry",
        list(range(10)),
        SamplingParams(max_new_tokens=2),
    )

    restored = cache.materialize_restored_prefix(req, 9)

    assert restored == 8
    assert req.aegis_restored_prefill_tokens == 8
    assert req.cached_length == 8
    assert req.kv_committed_length == 8
    assert req.cache_protected_len == 8
    assert req.cache_hit_tokens == 8
    assert req.cache_hit_recorded is True
    assert cache.radix.match_prefix(req.input_ids[:9]).device_length == 8

    cache.prepare_prefill(req)

    assert req.cached_length == 8
    assert req.kv_committed_length == 8


def test_aegis_pending_restore_action_does_not_materialize_device_cache():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache = KVCachePool(
        CacheConfig(page_size=4, l1_capacity_pages=8, l2_capacity_pages=0)
    )
    req = Req(
        "agent-0-t0-retry",
        list(range(10)),
        SamplingParams(max_new_tokens=2),
    )
    stats = RequestStats(
        rid=req.rid,
        aegis_failover_replay=True,
        aegis_replay_restore_action="pending_host_copy",
        aegis_replay_restored_tokens=8,
    )

    scheduler._materialize_aegis_replay_restore(req, stats)

    assert stats.aegis_replay_restored_prefill_tokens == 0
    assert req.aegis_restored_prefill_tokens == 0
    assert req.cached_length == 0
    assert scheduler.cache.radix.match_prefix(req.input_ids[:9]).device_length == 0


def test_aegis_restored_locked_prefix_is_not_evicted_under_pressure():
    cache = KVCachePool(
        CacheConfig(page_size=4, l1_capacity_pages=4, l2_capacity_pages=0)
    )
    req = Req(
        "agent-0-t0-retry",
        list(range(10)),
        SamplingParams(max_new_tokens=2),
    )

    restored = cache.materialize_restored_prefix(req, 9)
    locked_node = cache.radix.get_node(req.last_node_id)
    cache.evict_from_tree_cache(16, owner="unit-aegis-pressure")

    assert restored == 8
    assert locked_node is not None
    assert locked_node.lock_ref > 0
    assert cache.radix.match_prefix(req.input_ids[:9]).device_length == 8


def test_aegis_restored_prefix_survives_divergent_radix_insert():
    cache = KVCachePool(
        CacheConfig(page_size=4, l1_capacity_pages=16, l2_capacity_pages=0)
    )
    retry = Req(
        "agent-0-t0-retry",
        list(range(12)),
        SamplingParams(max_new_tokens=2),
    )
    sibling = Req(
        "agent-1-t0",
        [0, 1, 2, 3, 100, 101, 102, 103, 104, 105],
        SamplingParams(max_new_tokens=0),
    )

    restored = cache.materialize_restored_prefix(retry, 9)
    cache.prepare_prefill(sibling)
    cache.extend_prefill(sibling, len(sibling.input_ids))
    cache.cache_unfinished_req(sibling)
    cache.finish_request(sibling)

    assert restored == 8
    assert retry.cached_length == 8
    assert cache.radix.match_prefix(retry.input_ids[:9]).device_length == 8


def test_aegis_baseline_request_without_metadata_does_not_publish_frontier():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_rank = 0
    req = Req(
        "baseline-r",
        list(range(16)),
        SamplingParams(max_new_tokens=4),
    )
    stats = RequestStats(
        rid=req.rid,
        input_length=16,
        generated_tokens=4,
        cache_hit_tokens=8,
        aegis_policy="baseline",
        aegis_backup_kv=False,
        critical_agent=False,
    )

    scheduler._refresh_aegis_stats(req, stats)

    assert stats.aegis_host_visible_tokens == 0
    assert stats.aegis_gpu_replica_tokens == 0
    assert stats.aegis_safe_frontier_tokens == 8
    assert stats.aegis_recompute_tokens == 12
    assert stats.aegis_candidate_retry is False
    assert stats.aegis_candidate_recovery_action == "none"


def test_aegis_policy_host_only_publishes_host_frontier_only():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_rank = 0
    req = Req(
        "host-r",
        list(range(16)),
        SamplingParams(max_new_tokens=4),
    )
    stats = RequestStats(
        rid=req.rid,
        input_length=16,
        generated_tokens=4,
        cache_hit_tokens=0,
        aegis_policy="host_only",
        aegis_backup_kv=True,
        critical_agent=False,
    )

    scheduler._refresh_aegis_stats(req, stats)

    assert stats.aegis_host_visible_tokens == 20
    assert stats.aegis_gpu_replica_tokens == 0
    assert stats.aegis_safe_frontier_tokens == 20
    assert stats.aegis_candidate_retry is True
    assert stats.aegis_candidate_recovery_action == "host_restore"


def test_aegis_policy_gpu_replica_publishes_peer_frontier_only():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_rank = 0
    req = Req(
        "gpu-r",
        list(range(32)),
        SamplingParams(max_new_tokens=8),
    )
    stats = RequestStats(
        rid=req.rid,
        input_length=32,
        generated_tokens=8,
        cache_hit_tokens=0,
        aegis_policy="gpu_replica",
        aegis_backup_kv=True,
        critical_agent=False,
    )

    scheduler._refresh_aegis_stats(req, stats)

    assert stats.aegis_host_visible_tokens == 0
    assert stats.aegis_gpu_replica_tokens == 8
    assert stats.aegis_safe_frontier_tokens == 8
    assert stats.aegis_candidate_retry is True
    assert stats.aegis_candidate_recovery_action == "gpu_replica"


def test_aegis_gpu_budget_caps_published_peer_frontier():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_rank = 0
    req = Req(
        "gpu-budget-r",
        list(range(100)),
        SamplingParams(max_new_tokens=20),
    )
    stats = RequestStats(
        rid=req.rid,
        input_length=100,
        generated_tokens=20,
        cache_hit_tokens=0,
        aegis_policy="gpu_replica",
        aegis_gpu_budget_fraction=0.1,
        aegis_backup_kv=True,
        critical_agent=False,
    )

    scheduler._refresh_aegis_stats(req, stats)

    assert stats.aegis_host_visible_tokens == 0
    assert stats.aegis_gpu_replica_tokens == 12
    assert stats.aegis_safe_frontier_tokens == 12
    assert stats.aegis_recompute_tokens == 108


def test_aegis_size_only_ablation_allows_noncritical_gpu_replica():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_rank = 0
    req = Req(
        "size-only-r",
        list(range(100)),
        SamplingParams(max_new_tokens=20),
    )
    stats = RequestStats(
        rid=req.rid,
        input_length=100,
        generated_tokens=20,
        cache_hit_tokens=0,
        aegis_policy="aegiskv",
        aegis_ablation="size_only_backup",
        aegis_backup_kv=True,
        critical_agent=False,
    )

    scheduler._refresh_aegis_stats(req, stats)

    assert stats.aegis_gpu_replica_tokens > 0
    assert stats.aegis_alpha_policy_source


def test_aegis_no_marginal_accounting_ablation_publishes_full_frontier():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_rank = 0
    req = Req(
        "no-marginal-r",
        list(range(100)),
        SamplingParams(max_new_tokens=20),
    )
    stats = RequestStats(
        rid=req.rid,
        input_length=100,
        generated_tokens=20,
        cache_hit_tokens=0,
        aegis_policy="aegiskv",
        aegis_ablation="no_marginal_accounting",
        aegis_gpu_budget_fraction=1.0,
        aegis_backup_kv=True,
        critical_agent=True,
    )

    scheduler._refresh_aegis_stats(req, stats)

    assert stats.aegis_gpu_replica_tokens == 120
    assert stats.aegis_safe_frontier_tokens == 120


def test_aegis_policy_no_backup_allows_recompute_retry_without_frontier():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_rank = 0
    req = Req(
        "none-r",
        list(range(16)),
        SamplingParams(max_new_tokens=4),
    )
    stats = RequestStats(
        rid=req.rid,
        input_length=16,
        generated_tokens=4,
        cache_hit_tokens=0,
        aegis_policy="no_backup",
        aegis_backup_kv=False,
        critical_agent=True,
    )

    scheduler._refresh_aegis_stats(req, stats)

    assert stats.aegis_host_visible_tokens == 0
    assert stats.aegis_gpu_replica_tokens == 0
    assert stats.aegis_safe_frontier_tokens == 0
    assert stats.aegis_candidate_retry is True
    assert stats.aegis_candidate_recovery_action == "recompute"


def test_aegis_no_peer_locality_ablation_ignores_peer_restore():
    engine = Engine.__new__(Engine)
    engine.dp_size = 2
    engine._aegis_host_backup_lock = threading.Lock()
    engine._aegis_host_backup = {
        ("agent-0", 0, 0): AegisHostBackupEntry(
            rid="0-t0",
            agent_id="agent-0",
            worker_seq=0,
            source_dp_rank=0,
            peer_dp_rank=1,
            host_visible_tokens=1000,
            gpu_replica_tokens=120,
            input_length=960,
            generated_tokens=80,
            updated_time=1.0,
        )
    }
    req = GenerateReqInput(
        rid="0-t0-retry",
        input_len=96,
        max_new_tokens=8,
        simulation={
            "failover_replay": True,
            "agent_id": "agent-0",
            "worker_seq": 0,
            "failed_dp_rank": 0,
            "retry_dp_rank": 1,
            "pre_failure_total_tokens": 1000,
            "aegis_ablation": "no_peer_locality",
        },
    )

    engine._apply_aegis_host_backup_lookup(req)

    sim = req.simulation
    assert sim["host_backup_lookup_hit"] is True
    assert sim["peer_gpu_lookup_hit"] is False
    assert sim["peer_gpu_lookup_tokens"] == 0
    assert sim["restore_action"] == "host_restore"


def test_aegis_no_critical_restore_ablation_suppresses_frontier():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_rank = 0
    req = Req(
        "critical-r",
        list(range(16)),
        SamplingParams(max_new_tokens=4),
    )
    stats = RequestStats(
        rid=req.rid,
        input_length=16,
        generated_tokens=4,
        cache_hit_tokens=0,
        aegis_policy="aegiskv",
        aegis_ablation="no_critical_restore",
        aegis_backup_kv=True,
        critical_agent=True,
    )

    scheduler._refresh_aegis_stats(req, stats)

    assert stats.aegis_host_visible_tokens == 0
    assert stats.aegis_gpu_replica_tokens == 0
    assert stats.aegis_candidate_retry is True
    assert stats.aegis_candidate_recovery_action == "recompute"
