# DP=2 Single-Request Failure Visibility Pair 32984/32983

Run pair:

- Host-invisible control: job `32984`, no storage backend
- Host-visible restore smoke: job `32983`, `remote_backup` with
  `hicache_storage_prefetch_policy=wait_complete`
- Result directory: `results/failure_visibility_pair_32984_32983/`

Summary:

- Control success: 1/1
- Visible success: 1/1
- Control failover response count: 1
- Visible failover response count: 1
- Control pre-failover backed-up tokens: 0
- Visible pre-failover backed-up tokens: 1,297
- Frontier delta: 1,297 tokens
- Visible backed-up tokens/request: 1,297
- Control p99 latency: 15.58 s
- Visible p99 latency: 17.51 s
- Visible p99 latency normalized to control: 1.124x

Restore-path evidence:

The visible run is job `32983`, which materializes the accepted remote-backup
pages before retry admission: `Prefetch IO END ... completed=1297 / expected=1297`
and retry prefill `#cached-token: 1297`.

Interpretation:

This is the cleanest current single-request SGLang pair for the
Host-invisible vs Host-visible frontier chain. It proves that a real SGLang
retry can see and reuse a nonzero remote-backup frontier on the target rank
when the page-0 shape and `wait_complete` policy are used.

Limitations:

The visible run is slower in this one-request smoke, so the pair is not a
final recovery-latency benefit result. It is a correctness and calibration
artifact. Final HPCA evidence still needs repeated paired policy runs under
representative concurrency and matched normal-path cost.
