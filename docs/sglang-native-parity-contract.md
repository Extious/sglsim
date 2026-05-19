# SGLang Native Parity Contract

Reference tree: `/home/comp/zhanzhao/sglang`
Reference branch: `multi-level-backup`
Reference commit: `55044f182cd84544a03b50414644a2fa7a54640a`

Authoritative runtime rule:

`benchmark/online` must exercise a SGLang-shaped runtime. The online path must not use `RuntimeService.submit`.

Preserved upstream modules:

- `srt.entrypoints.http_server`
- `srt.entrypoints.engine`
- `srt.entrypoints.openai`
- `srt.managers.tokenizer_manager`
- `srt.managers.data_parallel_controller`
- `srt.managers.scheduler`
- `srt.managers.schedule_batch`
- `srt.managers.schedule_policy`
- `srt.managers.scheduler_output_processor_mixin`
- `srt.managers.scheduler_runtime_checker_mixin`
- `srt.managers.cache_controller`
- `srt.mem_cache`
- `srt.model_executor.model_runner` public contracts

Allowed simulation replacements:

- `ModelRunner.initialize`
- `ModelRunner.forward`
- `ModelRunner.sample`
- CUDA/Triton allocation kernels
- KV host/device/storage transfer payloads
- inter-rank process communication when local logical ranks are used
- latency prediction and wall-clock sleep

Forbidden shortcuts:

- fixed cache recovery ratios
- request-end synthetic failover metadata
- bypassing TokenizerManager/DataParallelController/Scheduler for failure handling
- online benchmark results produced by offline post-processing
