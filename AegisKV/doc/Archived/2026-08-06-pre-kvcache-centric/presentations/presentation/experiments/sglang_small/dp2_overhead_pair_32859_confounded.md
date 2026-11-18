# DP=2 Overhead Pair 32859 Confounded Result

Slurm job `32859` ran the first paired no-failure overhead experiment on
`hkbugpumanode`.

## Job

| Field | Value |
|---|---|
| Slurm job id | `32859` |
| Node | `hkbugpusrv15` |
| CUDA visible devices | `0,1` |
| Model | `Qwen/Qwen3-8B` |
| SGLang branch/SHA | `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce` |

## Result

Both paired cases completed 8/8 requests:

| Case | Mean latency (s) | p95 (s) | p99 (s) |
|---|---:|---:|---:|
| `baseline_no_hicache` | `2.647512674331665` | `3.199028968811035` | `3.199028968811035` |
| `hicache_write_through` | `2.447645604610443` | `2.977292537689209` | `2.977292537689209` |

The computed p99 overhead was negative:

```text
p99_overhead_pct = -6.931366776720302
```

## Why This Is Not Final Evidence

The baseline server log shows that `baseline_no_hicache` captured and used
piecewise CUDA graphs, while the previous HiCache smoke path disabled piecewise
CUDA graph. That makes the comparison confounded by runtime execution mode and
startup behavior rather than isolating HiCache write-through overhead.

This run is useful because it validates the paired Slurm workflow and summary
pipeline, but it must not be copied into `figures/data/measured/` as final Fig.
11 evidence.

## Fix Applied

The paired Slurm script now passes:

```text
--disable-piecewise-cuda-graph
```

to both baseline and HiCache server launches. A new paired run is required.

Raw pulled logs are under ignored local results:

```text
benchmark/AegisKV/presentation/experiments/sglang_small/results/dp2_overhead_pair_32859/
```
