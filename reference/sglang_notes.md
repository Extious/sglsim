# SGLang Reference Notes

Pinned SGLang is the semantic reference for:

- scheduler queue transitions and `event_loop_normal`
- schedule batch lifecycle
- radix prefix cache and HiRadixCache / HiCacheController behavior
- TokenizerManager / DetokenizerManager ZMQ topology (simulated via SimZmq)
- bench_serving metrics fields

SGLSim implements these under `sglsim.srt.*` without importing `sglang.srt`.

Overlap schedule timing (load hidden by prior inference, backup after forward) follows
`tools/sglang-simulator` under OFFLINE mode; use `--disable-overlap-schedule` to fall back
to serial load+infer+backup. Not simulated: disaggregation, speculative decoding.
