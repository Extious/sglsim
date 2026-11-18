# DP=2 Paired No-Failure Overhead Smoke: Job 32863

Status: usable larger small-scale SGLang paired evidence, not final paper data.

## Run

- Host: `hkbugpusrv11`
- Slurm job: `32863`
- SGLang repo: `/home/comp/25482351/sglang`
- Branch/SHA: `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce`
- Model: `Qwen/Qwen3-8B`
- Parallelism: `dp_size=2`, `tp_size=1`
- Requests: 32 completion requests per case
- Max generation tokens: 128
- Backend controls for both cases:
  - `--attention-backend triton`
  - `--sampling-backend pytorch`
  - `--disable-cuda-graph`
  - `--disable-piecewise-cuda-graph`

## Cases

| Case | HiCache | Successful requests | Mean latency (s) | p95 latency (s) | p99 latency (s) |
|---|---|---:|---:|---:|---:|
| baseline_no_hicache | disabled | 32/32 | 4.8096 | 5.7510 | 5.9420 |
| hicache_write_through | `--enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0 --hicache-write-policy write_through` | 32/32 | 4.8665 | 5.5160 | 5.7084 |

Derived paired overhead:

| Metric | Overhead |
|---|---:|
| mean latency | 1.18% |
| p95 latency | -4.09% |
| p99 latency | -3.93% |

## Interpretation

The run is matched and valid for the small-scale no-failure overhead evidence
set: both cases used the same A100 node allocation and the same CUDA graph
controls. The mean latency overhead is positive but small. The p95/p99 deltas
are negative, which should be interpreted as tail-latency noise at this sample
size rather than as a speedup claim.

Together with job `32861`, the current matched SGLang evidence supports only a
bounded-overhead claim: mean overhead is 1.18% to 3.05%, while p99 overhead is
within noise (-3.93% to 2.02%). It is not final HPCA evidence because it does
not include repeated workload classes, failure injection, HostBackup directory
visibility, GPUReplica placement, or full AegisKV policy execution.

## Local Artifacts

- Raw summary: `results/dp2_overhead_pair_32863/paired_overhead_summary.json`
- Candidate CSV: `results/dp2_overhead_pair_32863/candidate_fig_normal_path_overhead_sglang_pair.csv`
- Provenance: `results/dp2_overhead_pair_32863/provenance.txt`
- Slurm stdout/stderr:
  - `results/slurm-aegiskv-dp2-overhead-pair-32863.out`
  - `results/slurm-aegiskv-dp2-overhead-pair-32863.err`
