# SGLSim HostBackup Visibility Summary

Run id: `sglsim-fig11b-hostbackup-visibility-r11-20260621`

This artifact is the active preliminary input for Fig. 11b. It replaces the
old model-only retry-latency rows with direct hard-failure replay ratios for
the recoverable regimes.

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
- Derivation script: `scripts/derive_hostbackup_visibility_from_replay.py`
- Output CSV: `figures/data/measured/fig_hostbackup_visibility.csv`

## Results

Retry latency is normalized to same-row no-backup recompute.

- Baseline write-through invisible: `1.000`
- Host visible only on long-prefix Host-visible rows: `0.622`
- GPUReplica only on peer-visible hot-tail rows: `0.650`
- AegisKV over both recoverable regimes: `0.632`

## Interpretation

The figure now supports the specific HostBackup visibility claim: paying for
write-through D2H is not enough unless the retry rank can discover and restore
the ACKed Host frontier. The retry-latency axis is direct SGLSim replay
evidence; the overhead axis is still a preliminary fixed write-through /
peer-copy anchor and must be replaced by paired policy measurements before
final submission.
