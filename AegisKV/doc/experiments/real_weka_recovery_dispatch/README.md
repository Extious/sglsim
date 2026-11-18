# Recovery dispatch topology audit

This audit isolates one Mooncake/SGLsim semantic mismatch without consuming
any invalid transport or recomputation coefficient.

The frozen Mooncake Router replays one global Controller order serially. The
latest SGLsim evaluator already maintains one Slack-ordered queue per
`activated_target`. The staged Mooncake v13 patch adopts the same resource
model: serial within a target, concurrent across independent targets. Within
each target, the authoritative order is lower remaining Slack, then larger
current recovery harm, then stable request ID.

The audit reads only `event_id`, `request_id`, `policy`, and
`activated_target`. It deliberately ignores every duration, WCT, Slack value,
and action. Its unit-service parallelism fields are topological ceilings, not
predicted speedups. The pinned failure-row CSV predates the Slack/harm
tie-break correction, but the audit consumes no row order, duration, or harm;
the current evaluator hash and semantic marker are checked separately.

Run from the SGLsim repository root:

```bash
uv run python \
  AegisKV/experiments/real_weka_recovery_dispatch/audit_target_parallelism.py \
  --config AegisKV/experiments/real_weka_recovery_dispatch/config.json \
  --output /path/to/new-target-parallelism-audit.json
```

This diagnostic cannot be promoted to a physical, formal, or paper result.
The staged patch still requires the new source boundary and four-condition
physical pilot after the queued V100S calibrations finish.
