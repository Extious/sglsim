# DP=2 SGLang Overhead Pair 32874

This job adds a larger matched no-failure overhead repeat for Fig. 11.

- Slurm job: `32874`
- Node: `hkbugpusrv11`
- Model: `Qwen/Qwen3-8B`
- SGLang branch/SHA: `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce`
- Workload: 64 `/v1/completions` requests per case, 128 max output tokens
- Parallelism: DP=2, TP=1
- Controls: `--attention-backend triton`, `--sampling-backend pytorch`,
  `--disable-cuda-graph`, `--disable-piecewise-cuda-graph`

| Case | Completed | Mean latency | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| baseline no HiCache | 64/64 | 4.996 s | 5.034 s | 5.843 s | 5.928 s |
| HiCache write-through | 64/64 | 4.885 s | 4.896 s | 5.672 s | 5.976 s |

Derived overhead:

- Mean latency overhead: `-2.23%`
- P95 latency overhead: `-2.92%`
- P99 latency overhead: `0.80%`

After adding this job, the then-current matched aggregate included jobs `32861`, `32863`,
`32869`, `32870`, `32871`, and `32874`:

- Included count: `6`
- Mean overhead average: `0.58%`
- Mean overhead range: `-2.23%` to `3.05%`
- P95 overhead average: `-0.46%`
- P95 overhead range: `-4.09%` to `2.02%`
- P99 overhead average: `0.50%`
- P99 overhead range: `-3.93%` to `3.74%`

This aggregate has since been superseded by the seven-job aggregate that also
includes job `32899`.

The correct paper-level interpretation remains that HiCache write-through
normal-path overhead is within small-run noise under matched controls. This is
stronger real-system provenance for Fig. 11, but it is still not final AegisKV
policy evidence because it does not exercise physical AegisKV Host/GPUReplica
recovery paths.
