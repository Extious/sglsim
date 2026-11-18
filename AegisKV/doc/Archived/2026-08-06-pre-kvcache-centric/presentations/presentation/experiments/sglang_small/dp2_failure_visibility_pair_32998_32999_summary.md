# DP=2 Same-Write-Cost Failure Visibility Pair 32998/32999

Run pair:

- Host-invisible control: job `32998`
- Host-visible remote_backup run: job `32999`
- Result directory: `results/failure_visibility_pair_same_write_32998_32999/`

Summary:

- Control success/failover responses: 4/4
- Visible success/failover responses: 4/4
- Control backed-up tokens: 20,548
- Visible backed-up tokens: 20,548
- Frontier delta: 0 tokens
- Visible backed-up tokens/request: 5,137.0
- Control p99 latency: 26.42 s
- Visible p99 latency: 39.31 s
- Visible p99 normalized to control: 1.487x
- Visible p99 delta: 48.7%

Interpretation:

Both jobs enable remote_backup writes with wait_complete and PROMPT_REPEAT=256. Job 32998 disables failover-time storage prefetch, so it pays the same write path but hides the Host frontier during retry. Job 32999 enables restore lookup and materializes 5,137 cached tokens per retried request. This is a clean visibility/isolation artifact; in this diagnostic wait_complete configuration the visible path remains slower, so it is not final recovery-benefit evidence.

Evidence status:

This is synchronized SGLang small-scale paired evidence. It remains a
calibration/correctness artifact unless the paired run demonstrates a
representative recovery-benefit result under matched normal-path cost.
