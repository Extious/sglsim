# Current Physical Recovery-Action-Calibrated Agentic Suite

Run id: `sglsim-current-physical-recovery-action-agentic-r3-20260702`

Environment:

- `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy`
- `SGLSIM_AEGIS_PHYSICAL_COPY_CSV=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/draft/experiments/sglang_small/results/physical_recovery_path_aggregate/physical_recovery_path_aggregate.csv`

This suite reruns the DP2 Host, DP4 peer, DP4 non-peer, and DP4 peer-hot-tail
hard-failover scenarios with the current request-detail schema. The restore
timing is calibrated by the SGLang/torch physical recovery-path aggregate, but
the online SGLSim recovery data plane remains metadata materialization.

## Coverage

- Scenarios: `hard_failover_host`, `hard_failover_peer`, `hard_failover_nonpeer`, `hard_failover_peer_hot_tail`
- Repeats: 3 per scenario
- Completed request rows: 1,440 / 1,440
- Failed request rows: 0
- Hard-failure replay rows: 288 / 288
- Restore rows with current data-plane field: 288 / 288
- Fig. 10 / Fig. 14 measured rows now use this run id.

## Aggregate Results

| Scenario | DP | Replay rows | Host actions | Peer actions | Recompute actions | Mean E2E ms | Mean restore sleep ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hard_failover_host` | 2 | 72 | 48 | 24 | 0 | 107.943 | 17.149 |
| `hard_failover_peer` | 4 | 72 | 48 | 24 | 0 | 107.054 | 16.914 |
| `hard_failover_nonpeer` | 4 | 72 | 72 | 0 | 0 | 108.306 | 17.342 |
| `hard_failover_peer_hot_tail` | 4 | 72 | 0 | 72 | 0 | 92.850 | 5.009 |

## Live Movement Audit

- Audit CSV: `results/agentic_workflows/sglsim-current-physical-recovery-action-agentic-r3-20260702/live_physical_movement_audit.csv`
- Audit report: `results/agentic_workflows/sglsim-current-physical-recovery-action-agentic-r3-20260702/live_physical_movement_audit.md`
- `aegis_replay_restore_data_plane=metadata_materialization`: 288
- `aegis_replay_restore_timing_source=physical_recovery_path_csv`: 288
- `physical_tensor_copy` rows: 0

## Boundary

This replaces the older recovery-action suite in the figure-evidence and P0
strict-final audits because every restore row now exposes the data-plane and
timing-source fields. It remains preliminary for strict-final claims until the
live retry data plane performs physical Host H2D or peer GPUReplica tensor
movement and reports `aegis_replay_restore_data_plane=physical_tensor_copy`.
