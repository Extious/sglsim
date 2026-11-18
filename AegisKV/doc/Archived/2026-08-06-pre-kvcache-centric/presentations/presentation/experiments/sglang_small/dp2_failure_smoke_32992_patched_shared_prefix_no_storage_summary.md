# DP=2 Failure Smoke 32992: Patched Shared-Prefix No-Storage Control

Run:

- Slurm job: `32992`
- Node: `hkbugpusrv11`
- Workload: Qwen3-8B, DP=2, 4 routed in-flight requests, 512 max output tokens, 8 s failure delay
- Prompt mode: shared prefix (`UNIQUE_REQUEST_PREFIX=0`)
- Storage: none
- SGLang version: same patched working tree used by job `32991`
- Result directory: `results/dp2_failure_smoke_32992/`

Summary:

- Request success: 4/4
- Failover response count: 4/4
- Captured request count: 4
- Dispatched request count: 4
- Recovery status: HTTP 200
- Pre-failover backed-up tokens: 0
- Pre-failover output tokens: 744
- p99 latency: 16.93 s

Interpretation:

This is the same-code no-storage control for patched shared-prefix restore job
`32991`. It preserves the retry-control path while exposing zero Host-visible
frontier, so it is the cleaner control for the current Fig. 11b/Fig. 15 SGLang
support artifact.
