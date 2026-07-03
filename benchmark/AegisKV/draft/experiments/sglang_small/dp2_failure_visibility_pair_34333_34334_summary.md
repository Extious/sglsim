# DP=2 Same-Write-Cost Best-Effort Failure Visibility Pair 34333/34334

Run pair:

- Host-invisible control: job `34333`
- Host-visible remote_backup run: job `34334`
- HiCache prefetch policy: `best_effort`
- Result directory: `results/failure_visibility_pair_same_write_best_effort_34333_34334/`

Summary:

- Control success/failover responses: 4/4
- Visible success/failover responses: 4/4
- Control backed-up tokens: 15,425
- Visible backed-up tokens: 15,425
- Frontier delta: 0 tokens
- Visible backed-up tokens/request: 3,856.2
- Control p99 latency: 35.46 s
- Visible p99 latency: 33.51 s
- Visible p99 normalized to control: 0.945x
- Visible p99 delta: -5.5%

Interpretation:

Repeated same-write-cost best_effort pair. Both runs enable remote_backup writes with PROMPT_REPEAT=256 and FAILURE_DELAY_S=12; job 34333 disables failover-time storage lookup, while job 34334 enables lookup. This repeat strengthens the Fig. 11b/Fig. 15 evidence chain but remains small-scale SGLang smoke evidence.

Evidence status:

This is synchronized SGLang small-scale paired evidence. It remains a
calibration/correctness artifact unless the paired run demonstrates a
representative recovery-benefit result under matched normal-path cost.
