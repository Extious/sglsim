from __future__ import annotations

from sglsim.srt.mem_cache.storage.remote_backup.remote_backup_server import (
    REMOTE_BACKUP_ALL_DP_RANKS,
    RemoteBackupServer,
)
from sglsim.srt.mem_cache.storage.remote_backup.remote_backup_storage import (
    RemoteBackupStorage,
)


def test_remote_backup_match_and_get_can_use_pages_from_any_dp_rank():
    server = RemoteBackupServer(port=0, max_buffer_size_gb=1.0, page_size=1)

    server.insert_pages(
        dp_rank=1,
        token_ids=[10, 20, 30],
        page_start=0,
        pages=[b"rank-1-page-10", b"rank-1-page-20", b"rank-1-page-30"],
    )

    source_rank, data_count, navigable = server.match_prefix_from_with_source(
        REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20, 30],
        start_page=0,
    )

    assert source_rank == 1
    assert (data_count, navigable) == (3, 3)
    assert server.get_pages_by_tokens(
        REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20, 30],
        start_page=0,
        count=3,
    ) == [b"rank-1-page-10", b"rank-1-page-20", b"rank-1-page-30"]


def test_remote_backup_match_data_count_stops_at_first_missing_data_page():
    server = RemoteBackupServer(port=0, max_buffer_size_gb=1.0, page_size=1)

    server.insert_pages(
        dp_rank=1,
        token_ids=[10, 20, 30],
        page_start=1,
        pages=[b"rank-1-page-20", b"rank-1-page-30"],
    )

    source_rank, data_count, navigable = server.match_prefix_from_with_source(
        REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20, 30],
        start_page=0,
    )
    assert source_rank == 1
    assert (data_count, navigable) == (0, 3)

    source_rank, data_count, navigable = server.match_prefix_from_with_source(
        REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20, 30],
        start_page=1,
    )
    assert source_rank == 1
    assert (data_count, navigable) == (2, 2)


def test_remote_backup_all_rank_get_can_combine_pages_from_multiple_dp_ranks():
    server = RemoteBackupServer(port=0, max_buffer_size_gb=1.0, page_size=1)

    server.insert_pages(
        dp_rank=0,
        token_ids=[10, 20],
        page_start=0,
        pages=[b"rank-0-page-10"],
    )
    server.insert_pages(
        dp_rank=1,
        token_ids=[10, 20],
        page_start=1,
        pages=[b"rank-1-page-20"],
    )

    _source_rank, data_count, navigable = server.match_prefix_from_with_source(
        REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20],
        start_page=0,
    )

    assert (data_count, navigable) == (2, 2)
    assert server.get_pages_by_tokens(
        REMOTE_BACKUP_ALL_DP_RANKS,
        token_ids=[10, 20],
        start_page=0,
        count=2,
    ) == [b"rank-0-page-10", b"rank-1-page-20"]


def test_remote_backup_storage_read_lookup_uses_all_dp_ranks():
    storage = RemoteBackupStorage.__new__(RemoteBackupStorage)
    storage._dp_rank = 0

    assert storage._resolve_lookup_dp_rank(None) == REMOTE_BACKUP_ALL_DP_RANKS
    assert storage._resolve_lookup_dp_rank(1) == REMOTE_BACKUP_ALL_DP_RANKS
