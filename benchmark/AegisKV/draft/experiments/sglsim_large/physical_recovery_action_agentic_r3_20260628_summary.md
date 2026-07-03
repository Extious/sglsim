# Physical Recovery-Action-Calibrated Agentic Suite

Run id: `sglsim-physical-recovery-action-agentic-r3-20260628`

Environment:

- `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy`
- `SGLSIM_AEGIS_PHYSICAL_COPY_CSV=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/draft/experiments/sglang_small/results/physical_recovery_path_aggregate/physical_recovery_path_aggregate.csv`

This suite reruns the DP2 Host, DP4 peer, DP4 non-peer, and DP4 peer-hot-tail
hard-failover scenarios with restore timing driven by the recovery-shaped
physical movement aggregate from SGLang small-scale jobs `34357/34358/34359`.
The online SGLSim recovery path is still metadata materialization plus calibrated
sleep; it does not perform CUDA tensor KV page movement.

## Coverage

- Scenarios: `hard_failover_host`, `hard_failover_peer`, `hard_failover_nonpeer`, `hard_failover_peer_hot_tail`
- Repeats: 3 per scenario
- Completed request rows: 1,440 / 1,440
- Failed request rows: 0
- Hard-failure replay rows: 288 / 288
- Fig. 10 / Fig. 14 rows now use this run id.

## Aggregate Results

| Scenario | DP | Replay rows | Host actions | Peer actions | Recompute actions | Mean E2E ms | Mean restore sleep ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hard_failover_host` | 2 | 72 | 48 | 24 | 0 | 106.830 | 17.148 |
| `hard_failover_peer` | 4 | 72 | 48 | 24 | 0 | 106.250 | 16.914 |
| `hard_failover_nonpeer` | 4 | 72 | 72 | 0 | 0 | 106.536 | 17.343 |
| `hard_failover_peer_hot_tail` | 4 | 72 | 0 | 72 | 0 | 92.319 | 5.005 |

The hot-tail scenario still selects `gpu_replica` on all 72 replay rows. The
long-prefix Host and peer scenarios split between Host restore and small
peer-visible hot-tail restore; the non-peer route falls back to Host restore.

## Figure Updates

- `fig_recovery_path_breakdown.csv`: replaces prior online agentic rows from
  `sglsim-physical-copy-agentic-r3-20260628`.
- `fig_dp_scale.csv`: replaces prior agentic DP-scale rows with this run id.

## Boundary

This is the strongest current online SGLSim timing evidence because it uses
action-level physical Host/GPUReplica restore measurements rather than raw copy
calibration. It remains preliminary for HPCA-final claims until the live
recovery path performs equivalent physical KV movement instead of metadata cache
materialization plus calibrated sleep.
