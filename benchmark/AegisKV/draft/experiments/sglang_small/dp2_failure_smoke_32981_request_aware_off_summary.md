# DP=2 Failure Smoke 32981: Request-Aware Remote Backup Off

Purpose:

- test whether SGLang `remote_backup` failover prefetch misses in job `32980`
  were caused by request-aware lease supersede / GC;
- keep the probe non-invasive by launching the same `RemoteBackupServer` class
  through `scripts/launch_remote_backup_server_for_smoke.py` with
  `request_aware=False`.

Configuration:

- Slurm job: `32981`
- Node: `hkbugpusrv11`
- Model: `Qwen/Qwen3-8B`
- DP/TP: `2/1`
- Requests: `4`
- Max tokens: `512`
- Failure delay: `8 s`
- Remote backup: enabled
- Remote backup request-aware retention: disabled

Result:

| Metric | Value |
|---|---:|
| successful responses | 4/4 |
| failover responses | 4 |
| captured/re-dispatched requests | 4/4 |
| pre-failover output tokens | 660 |
| pre-failover backed-up tokens | 5,132 |
| p99 latency | 19.631 s |

Key log finding:

```text
DP1 _storage_hit_query_tokens: prefix_pages=0, data_pages=0, range=1297
DP1 Prefetch REVOKED ... only 0 storage hits (threshold=256)
REMOTE_BACKUP prefetch_match ... source_dp_rank=0 start_page=0 data_pages=0 navigable_pages=1297
REMOTE_BACKUP backup_put dp_rank=0 page_start=14 ...
```

Interpretation:

The negative result rules out request-aware lease GC as the primary cause of
the retry-side miss. The remote backup server retains a navigable token trie
and nonzero buffer data, but the written data begins at `page_start=14` because
the four prompts share an internal radix prefix. The retry rank queries from
`start_page=0`; since the first 14 pages have no `kv_data`, the contiguous
storage-hit prefix is zero and prefetch is revoked.

Next probe:

Run a single-request remote-backup failure smoke so the long prompt is likely
stored from `page_start=0`. If that yields `Prefetch ACCEPTED`, the remaining
implementation gap is cross-rank recovery over split radix prefixes rather
than remote-backup storage itself.
