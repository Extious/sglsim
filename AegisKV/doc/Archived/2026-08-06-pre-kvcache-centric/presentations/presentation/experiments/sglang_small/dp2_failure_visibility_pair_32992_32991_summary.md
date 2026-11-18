# DP=2 Patched Shared-Prefix Failure Visibility Pair 32992/32991

Run pair:

- Host-invisible control: job `32992`, no storage backend, shared-prefix prompt, patched SGLang working tree
- Host-visible restore smoke: job `32991`, patched SGLang `remote_backup` with `hicache_storage_prefetch_policy=wait_complete`, shared-prefix prompt
- Result directory: `results/failure_visibility_pair_32992_32991/`

Summary:

- Control success: 4/4
- Visible success: 4/4
- Control failover response count: 4
- Visible failover response count: 4
- Control pre-failover backed-up tokens: 0
- Visible pre-failover backed-up tokens: 5,188
- Frontier delta: 5,188 tokens
- Visible backed-up tokens/request: 1,297
- Control p99 latency: 16.93 s
- Visible p99 latency: 20.47 s
- Visible p99 latency normalized to control: 1.210x

Restore-path evidence:

The visible run materializes all four shared-prefix retry prefixes after the
split-prefix storage fix: each request reaches `data_pages=1297`,
`Prefetch ACCEPTED`, `completed=1297 / expected=1297`, and retry
`#cached-token: 1297`.

Interpretation:

This supersedes both the earlier `32978/32991` posthoc pair and the
unique-prefix diagnostic pair `32989/32987` as the cleanest current small-scale
SGLang support artifact. Both sides use the same patched SGLang working tree,
the same shared-prefix workload shape, and the same failure-control path.

Limitations:

The visible path is 21.0% slower in this diagnostic run and still uses the
remote-backup smoke path rather than final AegisKV selective Host/GPUReplica
policy. Use this as correctness and SGLang/SGLSim calibration evidence, not as
the final recovery-benefit measurement.
