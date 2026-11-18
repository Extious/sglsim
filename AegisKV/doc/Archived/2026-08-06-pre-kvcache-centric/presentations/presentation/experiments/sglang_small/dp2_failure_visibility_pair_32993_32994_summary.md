# DP=2 Long Shared-Prefix Failure Visibility Pair 32993/32994

Run pair:

- Host-invisible control: job `32993`
- Host-visible remote_backup run: job `32994`
- Result directory: `results/failure_visibility_pair_32993_32994/`

Summary:

- Control success/failover responses: 4/4
- Visible success/failover responses: 4/4
- Control backed-up tokens: 0
- Visible backed-up tokens: 20,548
- Frontier delta: 20,548 tokens
- Visible backed-up tokens/request: 5,137.0
- Control p99 latency: 22.06 s
- Visible p99 latency: 35.96 s
- Visible p99 normalized to control: 1.630x
- Visible p99 delta: 63.0%

Interpretation:

Longer shared-prefix SGLang probe with PROMPT_REPEAT=256 and FAILURE_DELAY_S=12. The remote_backup+wait_complete run proves a much larger ACKed Host frontier than the no-storage control, but the visible path remains slower in this diagnostic configuration, so this is calibration/correctness evidence rather than final recovery-benefit evidence.

Restore-path evidence:

The visible run materializes all four retry prefixes on the target rank. Each
retry query reaches `data_pages=5137`, each accepted prefetch completes
`5137 / 5137` tokens, and retry prefill admits with `#cached-token: 5137` and
only `184` new prompt tokens.

Evidence status:

This is synchronized SGLang small-scale paired evidence. It remains a
calibration/correctness artifact unless the paired run demonstrates a
representative recovery-benefit result under matched normal-path cost.
