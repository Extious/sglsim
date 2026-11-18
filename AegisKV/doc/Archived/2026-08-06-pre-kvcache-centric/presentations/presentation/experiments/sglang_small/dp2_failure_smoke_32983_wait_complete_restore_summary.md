# DP=2 Failure Smoke 32983: Wait-Complete Remote Backup Restore

Run:

- Slurm job: `32983`
- Node: `hkbugpusrv11`
- Workload: Qwen3-8B, DP=2, 1 routed in-flight request, 512 max output tokens, 8 s failure delay
- Storage: `hicache_storage_backend=remote_backup`
- Prefetch policy: `hicache_storage_prefetch_policy=wait_complete`
- Result directory: `results/dp2_failure_smoke_32983/`

Summary:

- Request success: 1/1
- Failover response count: 1/1
- Captured request count: 1
- Dispatched request count: 1
- Recovery status: HTTP 200
- Pre-failover backed-up tokens: 1,297
- Pre-failover output tokens: 213
- p99 latency: 17.51 s

Restore-path evidence:

- Retry-side storage query on DP1 reports `data_pages=1297` from page 0.
- SGLang logs `Prefetch ACCEPTED ... 1297 pages (1297 tokens)`.
- Remote backup serves all pages in 128-page batches, ending with `hit_pages=17/17`.
- With `wait_complete`, SGLang logs `Prefetch IO END ... completed=1297 / expected=1297 tokens`.
- Retry prefill admits with `#cached-token: 1297` and only 213 new tokens.

Interpretation:

This resolves the materialization gap seen in job `32982`, where
`best_effort` allowed the scheduler to terminate prefetch before the remote
backup fetch could increment `completed_tokens`. The accepted remote pages were
fetchable in both jobs, but `wait_complete` is needed in this smoke to make
them visible to the target-rank HiRadixCache before prefill admission.

Limitations:

This is still a one-request restore-path smoke, not final HPCA evidence. It
does not replace the required paired Host-visible/Host-invisible policy
comparison under matched normal-path write-through cost, and it does not
exercise physical peer GPUReplica KV copy.
