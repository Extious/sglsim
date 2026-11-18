# DP=2 Same-Write-Cost Best-Effort Failure Visibility Pair 33002/33003

Run pair:

- Host-invisible control: job `33002`
- Host-visible remote_backup run: job `33003`
- HiCache prefetch policy: `best_effort`
- Result directory: `results/failure_visibility_pair_same_write_best_effort_33002_33003/`

Summary:

- Control success/failover responses: 4/4
- Visible success/failover responses: 4/4
- Control backed-up tokens: 20,548
- Visible backed-up tokens: 20,548
- Frontier delta: 0 tokens
- Visible backed-up tokens/request: 5,137.0
- Control p99 latency: 26.43 s
- Visible p99 latency: 25.52 s
- Visible p99 normalized to control: 0.966x
- Visible p99 delta: -3.4%

Interpretation:

Both jobs enable remote_backup writes with PROMPT_REPEAT=256 and FAILURE_DELAY_S=12. Job 33002 disables failover-time storage prefetch, so it pays the same write path but hides the Host frontier during retry. Job 33003 enables best_effort lookup: retry queries find 5,137 data pages per request, prefetch is accepted, and one request partially materializes 2,816 pages before prefill. The visible best_effort path improves p99 versus the lookup-disabled control, but this is partial asynchronous restore evidence, not final full-restore policy evidence.

Evidence status:

This is synchronized SGLang small-scale paired evidence. It remains a
calibration/correctness artifact unless the paired run demonstrates a
representative recovery-benefit result under matched normal-path cost.
