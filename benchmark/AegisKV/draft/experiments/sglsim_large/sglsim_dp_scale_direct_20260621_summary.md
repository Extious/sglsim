# SGLSim Direct DP-Scale Summary

Run id: `sglsim-direct-dp-scale-dp2r5-dp4r3-dp8r3-20260621`

This artifact is the active preliminary input for Fig. 14. It is a direct
SGLSim serving-path hard-failover replay sweep, not a policy-model row.

## Inputs

- DP=2: `sglsim-failover-replay-matrix-r5-20260621`, 5 runs
- DP=4: `sglsim-dp-scale-dp4-r3-20260621`, 3 runs
- DP=8: `sglsim-dp-scale-dp8-r3-20260621`, 3 runs
- Derivation script: `scripts/derive_dp_scale_from_summaries.py`
- Output CSV: `figures/data/measured/fig_dp_scale.csv`

## Results

- DP=2 normalized duration: `1.000`
- DP=4 normalized duration: `0.985`
- DP=8 normalized duration: `0.983`
- Completed rows: `1320/1320`
- Hard-failure replay rows: `264`

## Interpretation

The sweep proves that the SGLSim online serving topology and hard-failure replay
path run through DP=2, DP=4, and DP=8 without failed requests in this
low-concurrency workload. It does not yet prove high-concurrency scaling or
physical KV-copy behavior.
