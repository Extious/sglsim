# SGLSim Restore/Recompute Boundary Summary

Run id: `sglsim-fig2-restore-boundary-calibrated-r11-20260621`

This artifact is the active preliminary input for Fig. 2. It replaces the
older policy-derived restore/recompute boundary with curves calibrated from
direct SGLSim hard-failure replay rows.

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
- Derivation script: `scripts/derive_restore_boundary_from_replay.py`
- Output CSV: `figures/data/measured/fig_restore_recompute_boundary.csv`

## Fitted Model

- Recompute: `4.0 + 0.0108 * tokens` ms
- Host restore: `7.5 + 0.0036 * tokens` ms
- GPUReplica selected path:
  `2.4 + 0.0019 * peer_tokens + 0.0108 * suffix_tokens` ms
- Observed peer-token coverage in the GPUReplica regime: `34.2%`
- Runtime replay rows used for calibration: `264`

## Interpretation

Fig. 2 now uses the same serving-path evidence family as Figs. 9-10. The curve
shows why a backup hit is not enough: short non-peer suffixes can be cheaper to
recompute, short peer-visible hot tails can use GPUReplica plus suffix
recompute, and long prefixes favor Host H2D. AegisKV should therefore choose
the lower-envelope recovery action at failure time.

This is still not final physical KV-copy evidence. The calibration comes from
SGLSim runtime restore metadata and scheduler-side restore sleep; real SGLang
CUDA KV movement remains future validation.
