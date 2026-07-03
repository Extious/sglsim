# SGLSim Ablation Counterfactual Summary

Run id: `sglsim-fig12-ablation-counterfactual-r11-20260621`

This artifact is the active preliminary input for Fig. 12. It replaces the
older hand-tuned ablation multipliers with replay-calibrated counterfactuals
from the same hard-failure regimes used by Figs. 2, 3, 8, 9, and 10.

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
- Derivation script: `scripts/derive_ablation_from_replay.py`
- Output CSV: `figures/data/measured/fig_ablation.csv`

## Results

The figure normalizes complete AegisKV to `1.0`.

- `no lifecycle`: `1.029`
- `no segment granularity`: `1.007`
- `no restore/recompute`: `1.022`
- `no pressure`: `1.005`
- `no hysteresis`: `1.001`
- `no critical restore`: `1.029`

## Interpretation

The current low-concurrency replay mix is intentionally conservative: only
failure-recovery overhead changes, while each row's base service latency stays
fixed. Lifecycle and critical restore matter most because the current replay
rows are all critical-agent retries; restore/recompute choice matters when
Host/GPU restore is slower than recompute or only a peer-visible tail exists.
Pressure and hysteresis need larger online sweeps to become strong final
claims. Final Fig. 12 must replace these counterfactual rows with implemented
policy toggles and physical KV movement.
