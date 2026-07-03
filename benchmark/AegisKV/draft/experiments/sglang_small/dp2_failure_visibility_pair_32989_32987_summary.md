# DP=2 Unique-Prefix Failure Visibility Pair 32989/32987

Run pair:

- Host-invisible control: job `32989`, no storage backend,
  `UNIQUE_REQUEST_PREFIX=1`
- Host-visible restore smoke: job `32987`, `remote_backup` with
  `hicache_storage_prefetch_policy=wait_complete` and
  `UNIQUE_REQUEST_PREFIX=1`
- Result directory: `results/failure_visibility_pair_32989_32987/`

Summary:

- Control success: 4/4
- Visible success: 4/4
- Control failover response count: 4
- Visible failover response count: 4
- Control pre-failover backed-up tokens: 0
- Visible pre-failover backed-up tokens: 5,204
- Frontier delta: 5,204 tokens
- Visible backed-up tokens/request: 1,301
- Control p99 latency: 16.73 s
- Visible p99 latency: 21.88 s
- Visible p99 latency normalized to control: 1.308x

Restore-path evidence:

The visible run materializes all four retry prefixes: each request reaches
`data_pages=1301`, `Prefetch ACCEPTED`, `completed=1301 / expected=1301`, and
retry `#cached-token: 1301`.

Interpretation:

This is the strongest current small-scale SGLang paired smoke for the
Host-invisible vs Host-visible restore chain. It validates capture,
re-dispatch, remote-backup frontier creation, remote fetch, target-rank
materialization, and retry cache reuse across four concurrent requests.

Limitations:

The prompt shape intentionally avoids a shared prefix, and the visible path is
30.8% slower in this diagnostic run. This pair is not a final recovery-benefit
measurement; it is a correctness and SGLang/SGLSim calibration artifact.
