# DP=2 Shared-Prefix Failure Visibility Pair 32978/32991

Run pair:

- Host-invisible control: job `32978`, no storage backend, shared-prefix prompt
- Host-visible restore smoke: job `32991`, patched SGLang `remote_backup` with `hicache_storage_prefetch_policy=wait_complete`, shared-prefix prompt
- Result directory: `results/failure_visibility_pair_32978_32991/`

Summary:

- Control success: 4/4
- Visible success: 4/4
- Control failover response count: 4
- Visible failover response count: 4
- Control pre-failover backed-up tokens: 0
- Visible pre-failover backed-up tokens: 5,188
- Frontier delta: 5,188 tokens
- Visible backed-up tokens/request: 1,297
- Control p99 latency: 16.89 s
- Visible p99 latency: 20.47 s
- Visible p99 latency normalized to control: 1.212x

Restore-path evidence:

The visible run materializes all four shared-prefix retry prefixes after the
split-prefix storage fix: each request reaches `data_pages=1297`,
`Prefetch ACCEPTED`, `completed=1297 / expected=1297`, and retry
`#cached-token: 1297`.

Interpretation:

This supersedes the unique-prefix diagnostic pair `32989/32987` as the
strongest current small-scale SGLang support artifact for the Host-invisible vs
Host-visible restore chain. It validates capture, re-dispatch, remote-backup
frontier creation, remote fetch, target-rank materialization, and retry cache
reuse across four concurrent shared-prefix requests.

Limitations:

The visible path is 21.2% slower in this diagnostic run and still uses the
remote-backup smoke path rather than final AegisKV selective Host/GPUReplica
policy. Use this as correctness and SGLang/SGLSim calibration evidence, not as
the final recovery-benefit measurement.
