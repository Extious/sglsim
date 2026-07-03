# DP=2 Failure Smoke 32982: Single-Request Remote Backup Storage-Hit Probe

Purpose:

- remove the four-request shared-prefix split that caused job `32981` to write
  remote-backup data from `page_start=14`;
- test whether a single in-flight request can produce a retry-side
  `remote_backup` storage match from `page_start=0`.

Configuration:

- Slurm job: `32982`
- Node: `hkbugpusrv11`
- Model: `Qwen/Qwen3-8B`
- DP/TP: `2/1`
- Requests: `1`
- Max tokens: `512`
- Failure delay: `8 s`
- Remote backup: enabled
- Remote backup request-aware retention: enabled

Result:

| Metric | Value |
|---|---:|
| successful responses | 1/1 |
| failover responses | 1 |
| captured/re-dispatched requests | 1/1 |
| pre-failover output tokens | 162 |
| pre-failover backed-up tokens | 1,297 |
| p99 latency | 17.718 s |

Key positive log evidence:

```text
REMOTE_BACKUP backup_put dp_rank=0 page_start=0 pages=128 ...
DP1 _storage_hit_query_tokens: prefix_pages=0, data_pages=1297, range=1297
DP1 Prefetch ACCEPTED ... 1297 pages (1297 tokens) from storage backend
REMOTE_BACKUP prefetch_get ... hit_pages=128/128
```

Remaining gap:

```text
DP1 Prefetch IO END ... completed=0 / expected=1297 tokens
DP1 Prefill batch ... #cached-token: 0
```

Interpretation:

The remote-backup storage path can locate and fetch backed-up prompt KV after a
DP-rank failure when the recoverable prefix begins at page 0. Therefore, the
previous four-request miss is mainly a split-radix prefix-contiguity problem:
the shared prefix is navigable but not backed with data, so suffix-only leaf
data from page 14 cannot satisfy the current contiguous-prefix prefetch rule.

The next HPCA-critical implementation gap is materialization: even when
`Prefetch ACCEPTED` and `prefetch_get` hits occur, the current run reports
`completed=0` and admits the retry with `#cached-token: 0`. Final SGLang
restore-benefit evidence needs the accepted storage pages to become target-rank
KV/cache-hit tokens before replay admission.
