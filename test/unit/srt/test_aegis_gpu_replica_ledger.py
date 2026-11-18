from sglsim.srt.managers.aegis_gpu_replica import GpuReplicaLedger


def test_gpu_replica_overlap_does_not_consume_alpha_quota():
    ledger = GpuReplicaLedger(dp_size=3, total_cache_tokens=100)
    ledger.record_inference(
        rid="warmup",
        dp_rank=1,
        cache_key="agent:a",
        token_count=80,
    )

    reservation = ledger.reserve(
        rid="r0",
        source_dp_rank=0,
        cache_key="agent:a",
        requested_tokens=100,
        alpha=0.3,
        healthy_ranks=[0, 1, 2],
    )

    assert reservation is not None
    assert reservation.backup_dp_rank == 1
    assert reservation.quota_tokens == 30
    assert reservation.overlap_tokens == 80
    assert reservation.unique_tokens == 20
    assert reservation.reserved_tokens == 100
    assert ledger.active_unique_tokens(1) == 20


def test_gpu_replica_release_marks_backup_evictable_and_frees_unique_tokens():
    ledger = GpuReplicaLedger(dp_size=2, total_cache_tokens=100)
    ledger.record_inference(
        rid="r0",
        dp_rank=0,
        cache_key="agent:a",
        token_count=100,
    )
    reservation = ledger.reserve(
        rid="r0",
        source_dp_rank=0,
        cache_key="agent:a",
        requested_tokens=30,
        alpha=0.3,
        healthy_ranks=[0, 1],
    )
    assert reservation is not None
    assert ledger.active_unique_tokens(1) == 30

    released = ledger.release_request("r0")

    assert released is not None
    assert released.evictable is True
    assert ledger.active_unique_tokens(1) == 0
    assert ledger.inflight_requests(0) == 0
    assert ledger.inflight_tokens(0) == 0
    assert ledger.prefix_tokens(1, "agent:a") == 30


def test_l1_clear_reports_execution_keys_but_not_backup_only_keys():
    ledger = GpuReplicaLedger(dp_size=3, total_cache_tokens=100)
    ledger.record_inference(
        rid="home",
        dp_rank=1,
        cache_key="agent:home",
        token_count=40,
    )
    reservation = ledger.reserve(
        rid="backup",
        source_dp_rank=0,
        cache_key="agent:backup-only",
        requested_tokens=20,
        alpha=0.5,
        healthy_ranks=[1],
    )

    assert reservation is not None
    result = ledger.clear_rank_l1(1)

    assert result["cleared_inference_cache_keys"] == ("agent:home",)
    assert result["cleared_prefix_keys"] == 2


def test_gpu_replica_falls_back_to_budget_load_when_cache_match_is_low():
    ledger = GpuReplicaLedger(dp_size=4, total_cache_tokens=100)
    busy = ledger.reserve(
        rid="busy",
        source_dp_rank=0,
        cache_key="other",
        requested_tokens=30,
        alpha=0.3,
        healthy_ranks=[0, 1],
    )
    assert busy is not None
    assert busy.backup_dp_rank == 1

    selected = ledger.choose_backup_rank(
        source_dp_rank=0,
        cache_key="agent:new",
        requested_tokens=30,
        alpha=0.3,
        healthy_ranks=[0, 1, 2, 3],
    )

    assert selected == 2


def test_execution_rank_prefers_cache_match_when_load_is_balanced():
    ledger = GpuReplicaLedger(dp_size=3, total_cache_tokens=100)
    ledger.record_inference(
        rid="warmup",
        dp_rank=1,
        cache_key="route:agent-a",
        token_count=80,
    )
    ledger.release_request("warmup")

    selected = ledger.choose_execution_rank(
        cache_key="route:agent-a",
        requested_tokens=100,
        healthy_ranks=[0, 1, 2],
    )

    assert selected == 1


def test_execution_rank_uses_shortest_load_when_workers_are_imbalanced():
    ledger = GpuReplicaLedger(dp_size=3, total_cache_tokens=100)
    # Pinned SGLang's cache-aware router switches to shortest queue only when
    # max-min exceeds the default absolute threshold of 32.
    for idx in range(33):
        ledger.record_inference(
            rid=f"busy-{idx}",
            dp_rank=1,
            cache_key="route:agent-a",
            token_count=80,
        )

    selected = ledger.choose_execution_rank(
        cache_key="route:agent-a",
        requested_tokens=100,
        healthy_ranks=[0, 1, 2],
    )

    assert selected == 0


def test_cache_load_aware_execution_prefers_existing_prefix_hit():
    ledger = GpuReplicaLedger(dp_size=3, total_cache_tokens=100)
    ledger.record_inference(
        rid="warmup-cache",
        dp_rank=1,
        cache_key="agent:shared",
        token_count=80,
    )
    ledger.record_inference(
        rid="busy-low-hit",
        dp_rank=2,
        cache_key="agent:other",
        token_count=20,
    )

    selected = ledger.choose_execution_rank(
        cache_key="agent:shared",
        requested_tokens=100,
        healthy_ranks=[0, 1, 2],
        alpha=0.3,
    )

    assert selected == 1


def test_cache_load_aware_execution_falls_back_to_lowest_token_load():
    ledger = GpuReplicaLedger(dp_size=3, total_cache_tokens=100)
    ledger.record_inference(
        rid="busy-r0",
        dp_rank=0,
        cache_key="agent:a",
        token_count=200,
    )
    ledger.record_inference(
        rid="busy-r1",
        dp_rank=1,
        cache_key="agent:b",
        token_count=50,
    )

    selected = ledger.choose_execution_rank(
        cache_key="agent:new",
        requested_tokens=100,
        healthy_ranks=[0, 1, 2],
        alpha=0.3,
    )

    assert selected == 2
