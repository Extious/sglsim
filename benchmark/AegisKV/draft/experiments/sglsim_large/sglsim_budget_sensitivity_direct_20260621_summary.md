# SGLSim Budget Sensitivity Summary

Run id: `sglsim-fig13-budget-sensitivity-regime-r11-20260621`

This artifact is the active preliminary input for Fig. 13. It replaces the
older hard-coded heatmap with a SGLSim runtime replay-calibrated regime proxy.

## Inputs

- Source regime 1: `hot-tail`
  - Matrix: `sglsim-fig10-peer-replay-r3-20260621`
  - GPUReplica replay rows: `72`
- Source regime 2: `long-prefix`
  - Matrix: `sglsim-failover-replay-matrix-r5-20260621`
  - Host restore replay rows: `120`
- Source regime 3: `non-peer`
  - Matrix: `sglsim-fig10-recompute-replay-r3-20260621`
  - Recompute replay rows: `72`
- Derivation script: `scripts/derive_budget_sensitivity_from_replay.py`
- Output CSV: `figures/data/measured/fig_budget_sensitivity.csv`

## Model

- Recompute: `4.0 + 0.0108 * tokens` ms
- Host restore: `7.5 + 0.0036 * tokens` ms
- GPUReplica selected path:
  `2.4 + 0.0019 * peer_tokens + 0.0108 * suffix_tokens` ms
- Runtime replay rows used for calibration: `264`

## Results

- Hot-tail rows improve from `1.000` at zero peer budget to `0.979`, then
  saturate after the peer-visible tail is covered.
- Long-prefix rows are best at zero forced peer budget (`0.936`) and rise to
  `0.952` when the budget forces a peer split.
- Non-peer rows remain `1.000` for all GPUReplica budgets because no peer
  tokens are visible to the retry rank.

## Interpretation

Fig. 13 now supports a narrower claim: the useful GPUReplica budget depends on
failure-time visibility and recovery regime. This is still preliminary because
the CSV field named `failure_timing` is currently a recovery-regime proxy, not
a controlled timing injection sweep. Final evidence must replace this proxy
with repeated online timing sweeps and physical KV movement.
