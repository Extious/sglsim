# DP=2 Failure Smoke 32987: Four-Request Unique-Prefix Wait-Complete Restore

Run:

- Slurm job: `32987`
- Node: `hkbugpusrv11`
- Workload: Qwen3-8B, DP=2, 4 routed in-flight requests, 512 max output tokens, 8 s failure delay
- Prompt mode: `UNIQUE_REQUEST_PREFIX=1`
- Storage: `hicache_storage_backend=remote_backup`
- Prefetch policy: `hicache_storage_prefetch_policy=wait_complete`
- Result directory: `results/dp2_failure_smoke_32987/`

Summary:

- Request success: 4/4
- Failover response count: 4/4
- Captured request count: 4
- Dispatched request count: 4
- Recovery status: HTTP 200
- Pre-failover backed-up tokens: 5,204
- Pre-failover output tokens: 669
- p99 latency: 21.88 s

Restore-path evidence:

- Unique first-token prompt prefixes make each long prompt write from
  `page_start=0` instead of the shared-prefix `page_start=14` shape seen in
  job `32985`.
- Retry-side storage queries on DP1 report `data_pages=1301` for all four
  requests.
- SGLang logs `Prefetch ACCEPTED` for all four requests.
- SGLang logs `Prefetch IO END ... completed=1301 / expected=1301 tokens` for
  all four requests.
- Retry prefill admits each request with `#cached-token: 1301`.

Interpretation:

This proves that four concurrent real SGLang failover retries can materialize
remote-backup pages on the target rank when the backed data forms a contiguous
page-0 prefix and `wait_complete` is used. Together with job `32985`, it
separates concurrent restore from shared-prefix split-node restore: concurrency
itself is not the blocker, but the shared-prefix storage layout is still an
open boundary for realistic multi-agent prompts.

Limitations:

This is a diagnostic prompt-shape probe. It deliberately removes the shared
prefix that AegisKV ultimately targets, so it is not final workload evidence.
It should be cited as a restore-path calibration point and a boundary test.
