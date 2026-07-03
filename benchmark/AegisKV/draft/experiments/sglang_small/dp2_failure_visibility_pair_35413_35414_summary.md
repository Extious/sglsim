# Strict-final recovery pair strict_final_same_write_wait_complete_r3

Run pair:

- Host-invisible control: job `35413`
- Host-visible remote_backup run: job `35414`
- HiCache prefetch policy: `wait_complete`
- Result directory: `results/strict_final_same_write_wait_complete_r3/`

Summary:

- Control success/failover responses: 4/4
- Visible success/failover responses: 4/4
- Control backed-up tokens: 20,548
- Visible backed-up tokens: 20,548
- Frontier delta: 0 tokens
- Visible backed-up tokens/request: 5,137.0
- Control p99 latency: 26.92 s
- Visible p99 latency: 38.74 s
- Visible p99 normalized to control: 1.439x
- Visible p99 delta: 43.9%

Interpretation:

Both runs pay remote_backup write cost; wait_complete makes the full Host restore path auditable even if it is slower.

Evidence status:

This is synchronized SGLang small-scale paired evidence. It remains a
calibration/correctness artifact unless the paired run demonstrates a
representative recovery-benefit result under matched normal-path cost.
