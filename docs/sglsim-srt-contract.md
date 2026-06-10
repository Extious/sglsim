# SGLSim SRT Contract

Metadata-only simulator under `sglsim.srt`, mirroring SGLang Runtime layout.

## SRT modules (simulated)

- `managers.tokenizer_manager`, `detokenizer_manager`, `scheduler`, `data_parallel_controller`
- `managers.cache_controller` (HiCacheController)
- `mem_cache.radix_cache`, `mem_cache.hiradix_cache`
- `model_executor.model_runner` (MockModel)
- `utils.network` (SimZmqSocket, not real ZMQ)
- `entrypoints.engine`, `entrypoints.http_server`
- `launch_server` CLI

## Included

- DP/TP/PP metadata, waiting queues, prefill/decode scheduling, chunked prefill
- radix prefix cache, HiCache L1/L2/L3 metadata, D2H/H2D timing, KV prefetch, L3 write
- AIConfigurator latency predictor (default)
- STREAMING arrival; wall-clock sleep simulation
- overlap schedule timing (default ON; `--disable-overlap-schedule` for serial HiCache clock)
- bench metrics: TTFT, TPOT, ITL, throughput table

## Excluded

- failure injection and recovery policies (separate branch)
- real GPU, CUDA/Triton, NCCL, upstream `sglang` imports

See also: [sglsim-srt-onboarding.md](sglsim-srt-onboarding.md), `reference/`.
