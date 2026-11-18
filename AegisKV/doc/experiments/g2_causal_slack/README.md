# G2 causal Slack baseline

This directory contains a diagnostic baseline for Algorithm 1. It asks whether
remaining workflow Slack can be recovered from causal, workflow-disjoint input
without using the trace's final `output_len` as a feature.

Run from the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  AegisKV/experiments/g2_causal_slack/run_workflow_disjoint_baseline.py
```

The script writes:

- `results/causal_slack_snapshots.csv`: fit, calibration, and evaluation rows;
- `results/causal_slack_baseline.json`: split, provenance, model, metrics, and
  limitations.

## Evidence boundary

This is a negative diagnostic, not a paper-ready result. The evaluation gate was
not pre-registered, there are only 19 workflows, and the archived SGLSim timing
labels came from a replay whose proportional context-overflow policy used final
trace output length. Final output length is not a model feature, but the replay
caveat still prevents promotion to clean online evidence.

The current evaluation partition has 5 workflows and 257 actionable rows:

- Slack MAE: `64,016.07 ms`;
- median per-snapshot Spearman: `-0.5`;
- observed lower-bound coverage at a 90% target: `100%`;
- nonzero lower-bound fraction: `0%`.

The safe interpretation is that the checked-in causal features do not identify
useful positive Slack. AegisKV therefore treats a learned Slack deadline as an
optional, gated enhancement. Its core path uses declared residual-plan bounds
when available and an event-driven `ACTIVE`/`DEFERRED` fallback otherwise.
