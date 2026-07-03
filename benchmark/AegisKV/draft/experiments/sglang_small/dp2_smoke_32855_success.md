# DP=2 Smoke 32855 Success

The fourth DP=2 smoke attempt completed successfully as Slurm job `32855`.
This is the first end-to-end small-scale SGLang evidence for the AegisKV
experiment path.

## Job

| Field | Value |
|---|---|
| Slurm job id | `32855` |
| Timestamp | `2026-06-21T02:02:04+08:00` |
| Node | `hkbugpusrv15` |
| CUDA visible devices | `0,1` |
| SGLang branch/SHA | `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce` |
| Model | `Qwen/Qwen3-8B` |
| DP size | `2` |

## Server Path Verified

The run used the conservative smoke configuration:

```text
attention_backend='triton'
sampling_backend='pytorch'
disable_cuda_graph=True
enable_hierarchical_cache=True
hicache_write_policy='write_through'
```

Both DP ranks completed:

- torch distributed initialization;
- local Qwen3-8B snapshot load;
- KV cache allocation;
- hierarchical host KV allocation;
- SGLang HTTP server startup;
- server warmup and localhost health checks with `no_proxy=127.0.0.1,localhost`.

## Smoke Client Result

The smoke client sent 8 sequential `/v1/completions` requests and all 8
completed successfully.

| Metric | Value |
|---|---:|
| count | `8` |
| success_count | `8` |
| latency_mean_s | `2.8218578696250916` |
| latency_p50_s | `2.870140552520752` |
| latency_p95_s | `3.430915355682373` |
| latency_p99_s | `3.430915355682373` |

## Consequence

The `hkbugpumanode` Slurm path is now validated for small-scale real SGLang
experiments. This result should not be copied into `figures/data/measured/` as
paper evidence by itself because it is a smoke-only AegisKV/HiCache run without
a paired baseline, failure injection, Host restore, or GPUReplica comparison.

Raw pulled logs are under ignored local results:

```text
benchmark/AegisKV/draft/experiments/sglang_small/results/dp2_smoke_32855/
```
