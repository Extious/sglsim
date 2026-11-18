# DP=2 Failure Smoke 32991: Patched Shared-Prefix Wait-Complete Restore

Run:

- Slurm job: `32991`
- Node: `hkbugpusrv11`
- Workload: Qwen3-8B, DP=2, 4 routed in-flight requests, 512 max output tokens, 8 s failure delay
- Prompt mode: shared prefix (`UNIQUE_REQUEST_PREFIX=0`)
- Storage: `hicache_storage_backend=remote_backup`
- Prefetch policy: `hicache_storage_prefetch_policy=wait_complete`
- SGLang patch: split parents created during an in-flight write-through backup are storage-backed before the child suffix
- Result directory: `results/dp2_failure_smoke_32991/`

Summary:

- Request success: 4/4
- Failover response count: 4/4
- Captured request count: 4
- Dispatched request count: 4
- Recovery status: HTTP 200
- Pre-failover backed-up tokens: 5,188
- Pre-failover output tokens: 780
- p99 latency: 20.47 s

Restore-path evidence:

- Retry-side DP1 storage queries report `data_pages=1297` for all four requests.
- SGLang logs `Prefetch ACCEPTED` for all four requests.
- SGLang logs `Prefetch IO END ... completed=1297 / expected=1297 tokens` for all four requests.
- Retry prefill admits each request with `#cached-token: 1297`.

Interpretation:

This resolves the shared-prefix split-node restore gap exposed by job `32985`.
The failure was not concurrent prefetch itself: when a shared-prefix radix node
was split while its host write-through copy was still in flight, only the child
suffix entered remote storage after ACK, leaving the remote trie navigable but
not data-contiguous from page 0. The patched path records split parents under
the original host-write ACK and sends those prefix pages to storage before the
child suffix.

Limitations:

This is still a small DP=2 smoke with overlap disabled and a diagnostic
`remote_backup + wait_complete` path. It is stronger SGLang correctness and
calibration evidence for Fig. 11b/Fig. 15, but not final HPCA-scale recovery
benefit evidence.
