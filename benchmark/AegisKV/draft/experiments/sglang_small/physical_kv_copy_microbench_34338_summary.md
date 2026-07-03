# Physical KV Copy Microbenchmark 34338

This artifact calibrates real CUDA movement costs for KV-sized tensors on the
`hkbugpumanode` Slurm cluster. It is physical copy evidence for the AegisKV
cost model, not an end-to-end AegisKV policy result.

Superseding aggregate: `physical_kv_copy_microbench_aggregate_summary.md`
combines jobs `34338`, `34344`, and `34345` and should be used for paper-facing
numbers.

## Run

- Slurm job: `34338`
- Node: `hkbugpusrv14`
- Visible GPUs: `CUDA_VISIBLE_DEVICES=4,5`
- GPU topology: `NV12` between GPU4 and GPU5 according to `nvidia-smi topo -m`
- Torch: `2.9.1+cu128`
- SGLang repo: `/home/comp/25482351/sglang`
- SGLang branch/SHA: `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce`
- KV size assumption: `147456` bytes/token, matching a Qwen3-8B-style bf16
  KV estimate of `2 * 36 layers * 8 KV heads * 128 head_dim * 2 bytes`
- Repeats: `20`, warmup: `5`

Raw outputs:

- `results/physical_kv_copy_34338/physical_kv_copy_microbench.csv`
- `results/physical_kv_copy_34338/physical_kv_copy_microbench.json`
- `results/physical_kv_copy_34338/physical_kv_copy_microbench.md`
- `results/physical_kv_copy_34338/nvidia-smi-topo.txt`

## Key Numbers

For an 8192-token KV segment, about 1152 MiB under the above assumption:

| Path | p50 latency | p95 latency | p50 bandwidth |
|---|---:|---:|---:|
| H2D pinned host to GPU | 48.475 ms | 48.505 ms | 24.92 GB/s |
| D2H GPU to pinned host | 46.509 ms | 46.514 ms | 25.97 GB/s |
| GPU0 local D2D copy | 1.384 ms | 1.389 ms | 872.52 GB/s |
| GPU0 to GPU1 P2P copy | 4.368 ms | 4.419 ms | 276.52 GB/s |

The same linear trend holds from 128 to 8192 tokens. At 128 tokens, P2P is
0.089 ms p50 while H2D/D2H are 0.764/0.836 ms p50. At 4096 tokens, P2P is
2.194 ms p50 while H2D/D2H are 24.239/23.258 ms p50.

## Interpretation

This run gives the draft its first direct physical-copy calibration for peer
GPUReplica: on same-node NVLink A100s, P2P copy is roughly 10.6x faster than
Host movement for the 8192-token point. This supports the qualitative shape of
Fig. 2 and Fig. 10: HostBackup is the capacity path for long prefixes, while
GPUReplica is the low-latency path when the retry rank can reach the recorded
peer.

It does not by itself close the final gate. The remaining missing evidence is
end-to-end integration where AegisKV triggers physical P2P/H2D page movement
during failure recovery and records the resulting WCT/recovery latency under
the same policy labels used by the paper figures.
