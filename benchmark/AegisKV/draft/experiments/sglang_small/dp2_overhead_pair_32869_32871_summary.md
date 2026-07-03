# DP=2 SGLang Overhead Pair Repeats 32869-32871

These jobs add three matched no-failure overhead repeats for Fig. 11. Each job
ran baseline no-HiCache and HiCache write-through on the same Slurm allocation
with matched CUDA graph controls:

- `--disable-cuda-graph`
- `--disable-piecewise-cuda-graph`
- `--attention-backend triton`
- `--sampling-backend pytorch`

All runs used `Qwen/Qwen3-8B`, DP=2, TP=1, 32 requests per case, and
128 max output tokens per request on `hkbugpumanode`.

| Job | Node | Baseline | HiCache WT | Mean overhead | P95 overhead | P99 overhead |
|---|---|---:|---:|---:|---:|---:|
| `32869` | `hkbugpusrv11` | 32/32 | 32/32 | -0.49% | -0.00% | 3.29% |
| `32870` | `hkbugpusrv11` | 32/32 | 32/32 | -0.35% | 1.86% | -2.93% |
| `32871` | `hkbugpusrv12` | 32/32 | 32/32 | 2.28% | 0.38% | 3.74% |

After adding these jobs, the matched aggregate at that point included jobs
`32861`, `32863`, `32869`, `32870`, and `32871`:

- Included count: `5`
- Mean overhead range: `-0.49%` to `3.05%`
- P99 overhead range: `-3.93%` to `3.74%`

This five-run aggregate was a historical intermediate snapshot and is no
longer used by the active figure package.

The correct paper-level interpretation is that HiCache write-through overhead
is within small-run noise under these matched controls. These runs are useful
real-system provenance for Fig. 11, but they are not final AegisKV policy
evidence.

Later job `32874` supersedes the aggregate above with a six-run aggregate. See
`dp2_overhead_pair_32874_summary.md` and
`results/aggregate_overhead_pairs/overhead_pair_aggregate.json` for the current
Fig. 11 SGLang small-scale aggregate.
