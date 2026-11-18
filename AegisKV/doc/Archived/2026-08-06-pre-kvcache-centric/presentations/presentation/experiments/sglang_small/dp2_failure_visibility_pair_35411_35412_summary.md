# Strict-final recovery pair strict_final_same_write_wait_complete_r2

Run pair:

- Host-invisible control: job `35411`
- Host-visible remote_backup run: job `35412`
- HiCache prefetch policy: `wait_complete`
- Result directory: `results/strict_final_same_write_wait_complete_r2/`

Summary:

- Control success/failover responses: 4/4
- Visible success/failover responses: 4/4
- Control backed-up tokens: 18,494
- Visible backed-up tokens: 18,494
- Frontier delta: 0 tokens
- Visible backed-up tokens/request: 4,623.5
- Control p99 latency: 34.24 s
- Visible p99 latency: 44.37 s
- Visible p99 normalized to control: 1.296x
- Visible p99 delta: 29.6%

Interpretation:

Both runs pay remote_backup write cost; wait_complete makes the full Host restore path auditable even if it is slower.

Evidence status:

This is synchronized SGLang small-scale paired evidence. It remains a
calibration/correctness artifact unless the paired run demonstrates a
representative recovery-benefit result under matched normal-path cost.
