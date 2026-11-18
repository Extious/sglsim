# DP=2 Paired No-Failure Overhead Smoke: Job 32861

Status: usable small-scale SGLang paired evidence, not final paper data.

## Run

- Host: `hkbugpusrv15`
- Slurm job: `32861`
- SGLang repo: `/home/comp/25482351/sglang`
- Branch/SHA: `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce`
- Model: `Qwen/Qwen3-8B`
- Parallelism: `dp_size=2`, `tp_size=1`
- Requests: 8 completion requests per case
- Max generation tokens: 64
- Backend controls for both cases:
  - `--attention-backend triton`
  - `--sampling-backend pytorch`
  - `--disable-cuda-graph`
  - `--disable-piecewise-cuda-graph`

## Cases

| Case | HiCache | Successful requests | Mean latency (s) | p95 latency (s) | p99 latency (s) |
|---|---|---:|---:|---:|---:|
| baseline_no_hicache | disabled | 8/8 | 2.4866 | 2.9637 | 2.9637 |
| hicache_write_through | `--enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0 --hicache-write-policy write_through` | 8/8 | 2.5624 | 3.0235 | 3.0235 |

Derived paired overhead:

| Metric | Overhead |
|---|---:|
| mean latency | 3.05% |
| p95 latency | 2.02% |
| p99 latency | 2.02% |

## Interpretation

This run fixes the confound in job `32859`: both baseline and HiCache
write-through disable global and piecewise CUDA graph behavior. It is therefore
a usable early data point for Fig. 11's no-failure overhead story.

It is not final HPCA evidence because it is an 8-request smoke workload and
does not include the full AegisKV policy, failure injection, HostBackup
directory visibility, GPUReplica placement, repeated runs, or workload matrix.

## Local Artifacts

- Raw summary: `results/dp2_overhead_pair_32861/paired_overhead_summary.json`
- Candidate CSV: `results/dp2_overhead_pair_32861/candidate_fig_normal_path_overhead_sglang_pair.csv`
- Provenance: `results/dp2_overhead_pair_32861/provenance.txt`
- Slurm stdout/stderr:
  - `results/slurm-aegiskv-dp2-overhead-pair-32861.out`
  - `results/slurm-aegiskv-dp2-overhead-pair-32861.err`
