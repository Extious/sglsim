# DP=2 Smoke 32853 Failure

The second DP=2 smoke attempt was submitted as Slurm job `32853` after
switching the launch path to the repository-managed `uv run python`
environment.

## What Worked

| Field | Value |
|---|---|
| Node | `hkbugpusrv11` |
| CUDA visible devices | `3,4` |
| SGLang branch/SHA | `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce` |
| Model | `Qwen/Qwen3-8B` |
| DP size | `2` |

The run reached the real SGLang DP scheduler startup path, loaded Qwen3-8B
weights on both DP ranks, and allocated the KV cache on both ranks.

Observed successful milestones from `server.log`:

```text
Load weight end. elapsed=58.66 s, type=Qwen3ForCausalLM
KV Cache is allocated. #tokens: 368047, K size: 25.27 GB, V size: 25.27 GB
```

## Failure

The server failed during CUDA graph capture when the flashinfer JIT cache tried
to load a generated shared library:

```text
/lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
```

This is an environment/toolchain incompatibility in the flashinfer CUDA graph
path, not a failure of the DP launch path or HiCache flags.

## Fix Applied

The smoke job now uses a conservative validation path that avoids this
flashinfer/CUDA-graph startup dependency:

```text
--attention-backend triton
--sampling-backend pytorch
--disable-cuda-graph
```

Raw pulled logs are under ignored local results:

```text
benchmark/AegisKV/draft/experiments/sglang_small/results/dp2_smoke_32853/
```
