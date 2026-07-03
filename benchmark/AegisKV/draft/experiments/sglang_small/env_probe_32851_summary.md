# SGLang Env Probe 32851 Summary

This is the first real Slurm-backed environment probe for the AegisKV small
SGLang experiment path.

## Job

| Field | Value |
|---|---|
| Slurm job id | `32851` |
| Timestamp | `2026-06-21T01:40:09+08:00` |
| Submit directory | `/home/comp/25482351/aegiskv_sglang_small` |
| Execution host | `hkbugpusrv12` |
| Partition | `short` |
| CUDA visible devices | `4` |

## SGLang

| Field | Value |
|---|---|
| Repository | `/home/comp/25482351/sglang` |
| Branch | `multi-level-backup` |
| SHA | `360072905d9373a33fb96360474c9e6e1682f0ce` |

The worktree was not fully clean during the probe:

```text
 M .dockerignore
 M python/sglang/srt/mem_cache/cpp_radix_tree/.clang-format
 M sgl-model-gateway/LICENSE
```

These modified files appear unrelated to AegisKV runtime logic, but later
paper experiments should still record the dirty status in provenance.

## GPU / Slurm Evidence

The allocated node exposes A100-SXM4-80GB GPUs. `CUDA_VISIBLE_DEVICES=4` was
assigned for the probe. The cluster also reports `short` queues with A100,
H20, and RTX4080 resources, plus `long` V100 resources.

Raw pulled files are under ignored local results:

```text
benchmark/AegisKV/draft/experiments/sglang_small/results/env_32851/
```

## Consequence

The remote path is valid for the planned small-scale SGLang experiments. The
next real-system step is the DP=2 smoke job with HiCache `write_through`, then
paired baseline/AegisKV recovery measurements.
