# Qwen3-32B DP4 x PP4 Real-Weka preflight

This directory preregisters the larger-model SGLsim configuration without
turning an 8B or transport-invalid profile into a 32B performance result.

The intended topology is Qwen3-32B BF16 with TP=1, PP=4, and system-DP=4.
SGLsim must therefore launch 16 Scheduler processes. Each pipeline stage owns
16 of the model's 64 layers. The model has a logical all-layer KV footprint of
262,144 bytes/token, or 65,536 bytes/token per pipeline stage.

The workload preserves all 393 workflows and 98,827 request identities from
the pinned Real-Weka trace. It reuses the preregistered token-projection grid,
not the old unscaled performance output. The primary scale remains 0.004;
0.008, 0.016, and 0.032 remain sensitivity points. Scaling changes token
volume, so none of these results may be called an unmodified Weka replay.

## Hard profile gate

The checked preflight is expected to report
`BLOCKED_MODEL_SPECIFIC_PROFILE`. Simulation remains unauthorized until
`profiles/qwen3-32b-bf16-dp4-pp4-v1.json` exists and binds all of the following:

- the exact Qwen3-32B revision and config hash;
- BF16 KV geometry and the TP1/PP4/DP4 layout;
- recomputation cost for this model/layout;
- independent source-owned Peer-HBM protection and recovery rates;
- independent source-owned direct-DRAM protection and recovery rates; and
- the 32,768-token experiment context and 4096/256 chunk layout.

The Qwen3-8B V100S profile, any RTX4090 TCP fallback, and archived DP4 x PP4
metadata outputs cannot satisfy this gate.

Run from the SGLsim repository root:

```bash
uv run python \
  AegisKV/experiments/real_weka_model_scale/audit_model_scale.py \
  --config AegisKV/experiments/real_weka_model_scale/config.json \
  --output /path/to/new-model-scale-audit.json
```

This command produces a readiness artifact only. It does not run the
simulator and never sets `formal_ready` or `paper_ready`.
