# Physical Recovery Path Microbenchmark Aggregate

This artifact aggregates jobs `34357`, `34358`, and `34359` on `hkbugpusrv14`.
Each run used two NV12-connected A100 GPUs through Slurm on `hkbugpumanode`,
with 20 measured repeats and 5 warmup repeats per token count.

The benchmark moves KV-sized tensors through AegisKV-shaped backup and restore
actions. Unlike the earlier raw copy calibration, the restore actions write
into a retry-rank KV buffer. It is still an action-level microbenchmark, not
full end-to-end SGLang failure recovery under the live AegisKV policy.

## Key 8192-token Results

Assumption: Qwen3-8B bf16 KV footprint, 147,456 bytes/token.

| Action | Phase | Copy path | p50 ms | 95% CI | p50 GB/s |
|---|---|---|---:|---:|---:|
| `host_backup_write_d2h` | backup | D2H pinned | 46.521 | 0.010 | 25.97 |
| `host_restore_h2d` | restore | H2D pinned | 48.434 | 0.013 | 24.94 |
| `gpu_replica_write_p2p` | backup | GPU0 to GPU1 P2P | 4.324 | 0.003 | 279.37 |
| `gpu_replica_restore_p2p` | restore | GPU0 to GPU1 P2P | 4.330 | 0.005 | 278.98 |
| `local_retry_buffer_copy_d2d` | restore | local D2D | 1.382 | 0.000 | 874.35 |

## Outputs

- Raw run directories:
  - `results/physical_recovery_path_34357/`
  - `results/physical_recovery_path_34358/`
  - `results/physical_recovery_path_34359/`
- Aggregate CSV: `results/physical_recovery_path_aggregate/physical_recovery_path_aggregate.csv`
- Aggregate markdown: `results/physical_recovery_path_aggregate/physical_recovery_path_aggregate.md`

## Paper Boundary

This artifact upgrades the physical-copy evidence from generic bandwidth
calibration to recovery-shaped movement. It does not close the final HPCA gate
for Fig. 10 or Fig. 14 by itself, because the current online SGLSim recovery
still materializes metadata and sleeps according to the calibrated timing model.
The next gate is to make live Host/GPUReplica recovery invoke equivalent
physical KV page movement in the recovery path.
