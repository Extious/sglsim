# SGLang DP=2 Overhead Pair Job 32899

This job adds a larger 128-request / 128-token matched no-failure overhead
repeat for Fig. 11.

## Provenance

- Host: `hkbugpusrv11`
- Slurm job: `32899`
- Model: `Qwen/Qwen3-8B`
- SGLang branch/SHA: `multi-level-backup` /
  `360072905d9373a33fb96360474c9e6e1682f0ce`
- Controls: `--attention-backend triton`, `--sampling-backend pytorch`,
  `--disable-cuda-graph`, `--disable-piecewise-cuda-graph`
- Baseline and HiCache write-through ran on the same Slurm allocation.

## Result

Both cases completed all requests:

- baseline: `128/128`
- HiCache write-through: `128/128`

Latency summary:

- baseline mean: `4.8232 s`; HiCache mean: `4.7300 s`; overhead: `-1.93%`
- baseline p95: `5.6089 s`; HiCache p95: `5.6892 s`; overhead: `1.43%`
- baseline p99: `5.9366 s`; HiCache p99: `5.8890 s`; overhead: `-0.80%`

## Interpretation

This strengthens the Fig. 11 real-system no-failure overhead anchor by adding
a larger repeat. The result remains smoke-scale evidence for the fixed
write-through path, not final AegisKV policy evidence.
