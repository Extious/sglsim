# Physical KV Copy Microbenchmark Aggregate

This artifact aggregates three CUDA copy calibration runs for KV-sized tensors
on the `hkbugpumanode` Slurm cluster. It is physical movement calibration for
AegisKV's cost model, not end-to-end AegisKV recovery policy evidence.

## Runs

- Slurm jobs: `34338`, `34344`, `34345`
- Node: `hkbugpusrv14`
- GPU topology: two A100 GPUs connected by `NV12` according to
  `nvidia-smi topo -m`
- Torch: `2.9.1+cu128`
- SGLang branch/SHA: `multi-level-backup` /
  `360072905d9373a33fb96360474c9e6e1682f0ce`
- KV size assumption: `147456` bytes/token, matching a Qwen3-8B-style bf16
  KV estimate of `2 * 36 layers * 8 KV heads * 128 head_dim * 2 bytes`
- Per-run repeats: `20`, warmup: `5`

Raw and aggregate outputs:

- `results/physical_kv_copy_34338/`
- `results/physical_kv_copy_34344/`
- `results/physical_kv_copy_34345/`
- `results/physical_kv_copy_aggregate/physical_kv_copy_rows.csv`
- `results/physical_kv_copy_aggregate/physical_kv_copy_aggregate.csv`
- `results/physical_kv_copy_aggregate/physical_kv_copy_aggregate.md`

## Key Aggregate Numbers

For an 8192-token KV segment, about 1152 MiB under the above assumption:

| Path | p50 latency mean | 95% CI | p50 bandwidth mean |
|---|---:|---:|---:|
| H2D pinned host to GPU | 48.595 ms | 0.531 ms | 24.86 GB/s |
| D2H GPU to pinned host | 46.416 ms | 0.100 ms | 26.02 GB/s |
| GPU0 local D2D copy | 1.384 ms | 0.001 ms | 872.74 GB/s |
| GPU0 to GPU1 P2P copy | 4.368 ms | 0.004 ms | 276.57 GB/s |

For a 128-token KV segment, p50 means are 0.764 ms H2D, 0.802 ms D2H, and
0.089 ms P2P. The P2P path remains consistently faster than Host movement
across 128 to 8192 tokens.

## Interpretation

This aggregate gives the draft a repeated physical-copy calibration for peer
GPUReplica: on same-node NVLink A100s, P2P copy is about 11.1x faster than H2D
and about 10.6x faster than D2H at the 8192-token point. This supports the
qualitative shape of Fig. 2 and Fig. 10: HostBackup is the capacity path for
long prefixes, while GPUReplica is the low-latency path when the retry rank can
reach the recorded peer.

It does not close the final gate. The remaining missing evidence is end-to-end
integration where AegisKV triggers physical P2P/H2D page movement during
failure recovery and records the resulting WCT/recovery latency under the same
policy labels used by the paper figures.
