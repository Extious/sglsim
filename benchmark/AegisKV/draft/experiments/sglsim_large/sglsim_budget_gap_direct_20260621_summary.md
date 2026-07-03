# SGLSim Static GPUReplica Budget-Gap Summary

Run id: `sglsim-fig3-static-budget-gap-r11-20260621`

This artifact is the active preliminary input for Fig. 3. It replaces the
older smooth policy-model budget curve with a SGLSim runtime replay-calibrated
counterfactual.

## Inputs

- Source regime 1: `long_prefix_host`
  - Matrix: `sglsim-failover-replay-matrix-r5-20260621`
  - Host restore rows: `120`
- Source regime 2: `short_peer_gpu`
  - Matrix: `sglsim-fig10-peer-replay-r3-20260621`
  - GPUReplica rows: `72`
- Source regime 3: `short_nonpeer_recompute`
  - Matrix: `sglsim-fig10-recompute-replay-r3-20260621`
  - Recompute rows: `72`
- Derivation script: `scripts/derive_budget_gap_from_replay.py`
- Output CSV: `figures/data/measured/fig_static_gpu_replica_budget_gap.csv`

## Model

- Recompute: `4.0 + 0.0108 * tokens` ms
- Host restore: `7.5 + 0.0036 * tokens` ms
- GPUReplica selected path:
  `2.4 + 0.0019 * peer_tokens + 0.0108 * suffix_tokens` ms
- Runtime replay rows used for calibration: `264`

## Results

- Hot-tail static budget: normalized WCT improves from `1.000` at zero peer
  budget to `0.979`, then saturates once peer-visible tail tokens are covered.
- Long-prefix static budget: Host-only is `0.936`; forcing a split through
  peer GPUReplica is about `0.952` in this replay mix.
- Fan-out/non-peer static budget: normalized WCT remains `1.000` because the
  retry rank has no peer-visible tokens.
- Dynamic AegisKV equal-regime replay counterfactual: `0.972`.

## Interpretation

Fig. 3 now supports a narrower and more defensible claim: static HBM budget
does not imply failure-time recoverability. Peer-visible hot tails benefit from
GPUReplica, Host-visible long prefixes should not be forced through peer
splits, and fan-out/non-peer retries need visibility-aware placement or
recompute. This is still preliminary because it replaces replay-row recovery
overheads with calibrated counterfactuals; final evidence needs online policy
sweeps with physical KV movement.
