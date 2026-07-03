# Physical-Copy-Calibrated SGLSim Agentic Suite

This artifact runs the repeated SGLSim agentic hard-failover suite with
physical-copy-calibrated restore timing. It uses the same online HTTP runtime,
hard scheduler kill/restart replay path, HostBackup directory lookup, and peer
GPUReplica directory lookup as the earlier agentic suite, but enables:

- `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy`
- `SGLSIM_AEGIS_PHYSICAL_COPY_CSV=/home/comp/zhanzhao/sglsim/benchmark/AegisKV/draft/experiments/sglang_small/results/physical_kv_copy_aggregate/physical_kv_copy_aggregate.csv`

The run is still SGLSim metadata materialization. It does not execute physical
CUDA tensor KV movement, so it remains preliminary evidence.

## Run

- Suite id: `sglsim-physical-copy-agentic-r3-20260628`
- Suite root:
  `results/agentic_workflows/sglsim-physical-copy-agentic-r3-20260628`
- Scenarios:
  - `hard_failover_host`
  - `hard_failover_peer`
  - `hard_failover_nonpeer`
  - `hard_failover_peer_hot_tail`
- Repeats: 3 trials per scenario
- Total runtime rows: 1,440
- Total hard-failover replay rows: 288
- Failed rows: 0

## Aggregate

| Scenario | DP | Rows | Replay rows | Host actions | Peer actions | Recompute tokens | Mean E2E | Mean restore sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hard_failover_host` | 2 | 360 | 72 | 48 | 24 | 25,266 | 107.538 ms | 17.157 ms |
| `hard_failover_peer` | 4 | 360 | 72 | 48 | 24 | 24,520 | 106.054 ms | 16.925 ms |
| `hard_failover_nonpeer` | 4 | 360 | 72 | 72 | 0 | 0 | 106.660 ms | 17.352 ms |
| `hard_failover_peer_hot_tail` | 4 | 360 | 72 | 0 | 72 | 16,778 | 92.034 ms | 5.011 ms |

The long-prefix Host/peer scenarios still choose Host restore for most replay
tokens because peer GPUReplica covers only the hot tail and suffix recompute
dominates the lower envelope. The hot-tail peer scenario chooses `gpu_replica`
for all 72 replay rows and has the smallest restore sleep.

## Figure Usage

- Fig. 10 now uses this suite for the online action-distribution rows.
- Fig. 14 now uses this suite for the agentic DP-scale rows.

This strengthens the preliminary online evidence by aligning restore timing
with the repeated physical copy aggregate, but the final gate still requires
end-to-end physical CUDA KV movement or real SGLang policy measurements.
