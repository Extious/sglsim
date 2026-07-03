# DP=2 Smoke 32854 Failure

The third DP=2 smoke attempt was submitted as Slurm job `32854` after disabling
CUDA graph capture and avoiding the flashinfer attention path.

## What Worked

| Field | Value |
|---|---|
| Node | `hkbugpusrv11` |
| CUDA visible devices | `3,4` |
| SGLang branch/SHA | `multi-level-backup` / `360072905d9373a33fb96360474c9e6e1682f0ce` |
| Model | `Qwen/Qwen3-8B` |
| DP size | `2` |

The run reached full server startup with the conservative backend path:

```text
attention_backend='triton'
sampling_backend='pytorch'
disable_cuda_graph=True
```

Both DP ranks loaded weights, allocated KV cache, allocated hierarchical host
KV memory, and the HTTP server reported application startup complete.

## Failure

The job was cancelled after the server became reachable because local HTTP
requests were routed through the cluster HTTP proxy:

```text
HTTPConnectionPool(host='proxy.hkbu.edu.hk', port=8080): Max retries exceeded
with url: http://127.0.0.1:30854/model_info
```

The same proxy behavior also affects smoke client health checks unless the
client explicitly bypasses proxies for localhost.

## Fix Applied

`slurm/aegiskv_dp2_smoke.sbatch` now exports:

```text
NO_PROXY=127.0.0.1,localhost,...
no_proxy=127.0.0.1,localhost,...
```

`scripts/sglang_smoke_client.py` also uses a `ProxyHandler({})` opener so the
local smoke client never routes localhost requests through the cluster proxy.

Raw pulled logs are under ignored local results:

```text
benchmark/AegisKV/draft/experiments/sglang_small/results/dp2_smoke_32854/
```
