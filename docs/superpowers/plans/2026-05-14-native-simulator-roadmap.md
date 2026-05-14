# Native Simulator Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build native simulator support in `/home/comp/zhanzhao/sglsim` through small, reviewable phases.

**Architecture:** Add a first-class `sglang.srt.simulator` runtime package, then wire it into server args, `ModelRunner`, `Scheduler`, HiCache, and `bench_serving`. Each phase produces a testable slice so the final behavior does not depend on import-time hooks.

**Tech Stack:** Python, SGLang SRT, pytest, native `sglang.srt.simulator` implementation.

---

## Execution Order

- [ ] **Phase 1: CLI and Config Foundation**

Plan: `docs/superpowers/plans/2026-05-14-native-simulator-phase-1-cli-config.md`

Outcome: `ServerArgs` accepts simulator options; `sglang.srt.simulator.config` validates and exposes simulator settings.

- [ ] **Phase 2: Metrics and Runtime State**

Plan: `docs/superpowers/plans/2026-05-14-native-simulator-phase-2-state-metrics.md`

Outcome: reusable simulator state, request stats, metric calculation, and output file writing.

- [ ] **Phase 3: ModelRunner Simulation Mode**

Plan: `docs/superpowers/plans/2026-05-14-native-simulator-phase-3-model-runner.md`

Outcome: `ModelRunner` can initialize without loading real weights and can return mock logits/tokens under `--simulator`.

- [ ] **Phase 4: Scheduler Simulation Mode**

Plan: `docs/superpowers/plans/2026-05-14-native-simulator-phase-4-scheduler.md`

Outcome: scheduler handles offline/blocking request timing, predicted latency, logical clock, and profile-triggered metrics dump.

- [ ] **Phase 5: HiCache and Allocator Simulation**

Plan: `docs/superpowers/plans/2026-05-14-native-simulator-phase-5-hicache-allocator.md`

Outcome: HiCache storage/transfer and CPU paged allocator behavior are native simulator branches instead of hooks.

- [ ] **Phase 6: bench_serving Simulator Mode**

Plan: `docs/superpowers/plans/2026-05-14-native-simulator-phase-6-bench-serving.md`

Outcome: `python -m sglang.bench_serving --simulator` injects logical request timestamps and reads backend simulator metrics.

## Final Verification

Run from `/home/comp/zhanzhao/sglsim`:

```bash
PYTHONPATH=python pytest test/registered/unit/simulator -q
PYTHONPATH=python python -m sglang.launch_server --help | grep simulator
PYTHONPATH=python python -m sglang.bench_serving --help | grep simulator
```

If a GPU-free smoke environment is available:

```bash
PYTHONPATH=python SGLANG_USE_CPU_ENGINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B-FP8 \
  --simulator \
  --sim-config-path test/registered/unit/simulator/assets/config.json \
  --skip-server-warmup
```
