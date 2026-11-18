# DP=2 Failure Smoke 32985: Four-Request Wait-Complete Probe

Run:

- Slurm job: `32985`
- Node: `hkbugpusrv11`
- Workload: Qwen3-8B, DP=2, 4 routed in-flight requests, 512 max output tokens, 8 s failure delay
- Storage: `hicache_storage_backend=remote_backup`
- Prefetch policy: `hicache_storage_prefetch_policy=wait_complete`
- Result directory: `results/dp2_failure_smoke_32985/`

Summary:

- Request success: 4/4
- Failover response count: 4/4
- Captured request count: 4
- Dispatched request count: 4
- Recovery status: HTTP 200
- Pre-failover backed-up tokens: 5,132
- Pre-failover output tokens: 688
- p99 latency: 18.73 s

Restore-path evidence:

- Before failure, the four-request shape again writes a nonzero Host-visible
  frontier: 5,132 backed-up tokens.
- Retry-side storage match on DP1 reports `range=1297` for each request, so
  the remote radix structure is navigable.
- Retry-side storage match still reports `data_pages=0` for all four requests,
  and each prefetch is revoked before IO.
- Retry prefill therefore admits with only ordinary local cache hits
  (`#cached-token: 0` or 28 in the batched retry rows), not restored
  remote-backup pages.

Interpretation:

`wait_complete` fixes the single-request materialization race in job `32983`,
but it does not fix the four-request split-prefix/storage-layout issue. The
concurrent prompt shape creates navigable remote radix entries whose leaf data
does not form a contiguous page-0 data prefix for the retry rank. Final
multi-request SGLang restore evidence therefore needs either a request shape
with page-0 backed data per retried request or a remote-backup matching policy
that can restore from the backed leaf range instead of requiring a contiguous
prefix from page 0.

Limitations:

This is still a diagnostic smoke. It validates the next implementation boundary
but should not be used as a final policy-latency datapoint.
