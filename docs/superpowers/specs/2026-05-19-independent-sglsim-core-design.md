# Independent SGLSim Core Design

Date: 2026-05-19
Status: proposed design
Reference SGLang tree: `/home/comp/zhanzhao/sglang`
Reference SGLang branch: `multi-level-backup`
Reference SGLang commit: `55044f182cd84544a03b50414644a2fa7a54640a`

## Summary

SGLSim should become an independent simulator whose authoritative runtime no longer depends on vendored SGLang runtime modules. Scheduling, KV/cache management, failure handling, communication semantics, time advancement, and metrics should live in SGLSim-owned modules. SGLang remains the behavioral reference: its selected commit defines the semantics SGLSim is trying to reproduce, but SGLSim does not import SGLang runtime code on the authoritative path.

The target relationship is:

```text
benchmark / HTTP / CLI
  -> SGLSim adapters
  -> SGLSim core runtime
  -> SGLSim scheduler, KV/cache, communication, failover, metrics
```

SGLang is used only for reference fixtures, differential tests, and future semantic audits:

```text
SGLang reference commit
  -> fixture generation and behavior comparison
  -> tests that validate SGLSim core equivalence
```

## Motivation

The current native parity approach preserves correctness by running through a large SGLang-shaped runtime and patching selected GPU-specific operations. That gives strong short-term parity, but it creates long-term maintenance risk:

- many SGLang runtime files are copied into `python/sglsim/srt`;
- simulator behavior depends on upstream internals spread across scheduler, cache, model runner, server, and communication modules;
- dependency footprint remains closer to SGLang than to a CPU-only simulator;
- future SGLang changes can silently invalidate the local copy or the patch layer;
- it is difficult to tell which behavior belongs to SGLSim and which behavior is inherited from SGLang.

This design keeps the important part of SGLang parity: the scheduler, KV/cache, communication, and failover semantics. It changes the ownership boundary: SGLSim implements those semantics directly and proves equivalence through tests.

## Goals

- Make the authoritative SGLSim runtime independent from vendored SGLang runtime modules.
- Preserve SGLang-derived semantics for scheduling, KV/cache management, failover, retry, and logical rank communication.
- Keep benchmark outputs compatible with the current `benchmark/fault_tolerance` and `benchmark/perf` consumers.
- Support CPU-only, metadata-only simulation without real GPU inference, CUDA kernels, tensor KV transfer, or distributed GPU execution.
- Replace monkey patch based correctness with SGLSim-owned state machines and contract tests.
- Keep SGLang available as a reference oracle, not as the runtime foundation.

## Non-Goals

- Do not implement real model inference.
- Do not implement CUDA graphs, CUDA/Triton kernels, NCCL, or real tensor KV transfer.
- Do not reproduce the full SGLang HTTP server surface.
- Do not support every SGLang model, multimodal path, LoRA path, speculative decoding path, or disaggregation mode in the first independent runtime.
- Do not track latest SGLang automatically. The reference is a pinned commit until deliberately updated.

## Architectural Boundary

The new core must be import-clean:

```text
python/sglsim/core/** must not import:
  sglsim.srt.managers
  sglsim.srt.mem_cache
  sglsim.srt.entrypoints
  sglsim.srt.model_executor
  torch.cuda
  triton
```

Allowed dependencies inside core are standard library modules, lightweight structured data libraries already accepted by the project, and SGLSim-owned modules.

Adapters may import external or legacy modules, but adapters are not allowed to own scheduling, cache, failover, or time semantics.

## Proposed Module Layout

```text
python/sglsim/core/
  config.py
  request.py
  batch.py
  runtime.py
  clock.py
  scheduler.py
  policy.py
  predictor.py
  metrics.py
  events.py
  communicator.py
  failover.py
  cache/
    __init__.py
    radix.py
    allocator.py
    tiered_kv.py
    controller.py
    transfer.py

python/sglsim/adapters/
  benchmark/
    fault_tolerance.py
    perf.py
  http/
    openai_compatible.py
  sglang_reference/
    fixtures.py
    converters.py
    diff.py
```

Existing `python/sglsim/srt/simulator` modules can be migrated into this layout incrementally. The current `native` patch layer remains temporary during migration and is removed from the authoritative path after the new core passes contract tests.

## Core Data Model

Core data structures should be SGLSim-owned and small enough to reason about without reading SGLang internals:

```text
SimRequest
  request_id
  created_time
  queue_start
  input_ids
  max_new_tokens
  output_ids
  sampling_metadata
  simulation_metadata
  assigned_rank
  priority
  status
  failover_metadata

SimBatch
  batch_id
  forward_mode: prefill | decode
  rank_id
  requests
  prediction_items
  cache_operations

RankState
  rank_id
  status: healthy | failed | recovering
  waiting_queue
  running_batch
  retry_queue
  completed_requests

KVPage
  page_id
  owner_rank
  token_span
  residency: device | host | remote | missing
  ref_count
  lock_count
  backup_status

RuntimeEvent
  event_type
  timestamp
  rank_id
  request_id
  fields
```

Adapters convert benchmark, HTTP, or reference SGLang objects into these structures. Core logic never consumes SGLang `Req`, `ScheduleBatch`, `ServerArgs`, or cache classes directly.

## Scheduler Design

The scheduler is a SGLang-semantics state machine implemented by SGLSim.

Reference files:

```text
srt/managers/scheduler.py
srt/managers/schedule_batch.py
srt/managers/schedule_policy.py
```

Core modules:

```text
core/scheduler.py
core/batch.py
core/policy.py
```

Initial supported semantics:

- request lifecycle: queued, prefill, decode, completed, aborted, retrying;
- fixed workload request ingestion with explicit creation timestamps;
- blocking mode and offline mode;
- per-DP-rank queues;
- prefill batch creation;
- decode batch creation;
- running batch update after each simulated forward pass;
- retry queue handling for failed ranks;
- `retry-queue-same-priority-as-waiting` behavior;
- chunked prefill behavior needed by existing fault-tolerance tests;
- deterministic scheduling for golden fixture comparison.

The scheduler calls a predictor through a stable interface:

```text
predict_batch_latency(batch: SimBatch) -> seconds
```

The scheduler does not sleep by itself. It reports predicted work to `core/runtime.py`, which advances `core/clock.py`.

## KV And Cache Design

The KV/cache layer reproduces SGLang cache semantics with metadata-only storage.

Reference files:

```text
srt/mem_cache/radix_cache.py
srt/mem_cache/hiradix_cache.py
srt/mem_cache/memory_pool.py
srt/mem_cache/memory_pool_host.py
srt/managers/cache_controller.py
```

Core modules:

```text
core/cache/radix.py
core/cache/allocator.py
core/cache/tiered_kv.py
core/cache/controller.py
core/cache/transfer.py
```

Responsibilities:

- `PrefixRadixCache` implements prefix matching, insertion, eviction, lock references, and hit accounting.
- `KVPageAllocator` allocates page IDs, tracks owner ranks, maps request token ranges to pages, and frees pages.
- `TieredKVStore` tracks page residency across device, host, remote, and missing tiers.
- `CacheController` schedules backup, prefetch, load, and eviction operations.
- `TransferEstimator` converts page counts and configured bandwidth into simulated transfer latency.

The cache layer does not allocate real KV tensors. Page size, model shape, dtype, TP/PP/DP size, and memory fraction are represented as metadata and used for capacity and latency estimates.

## Communication Design

SGLSim should not copy SGLang's ZMQ, multiprocessing, or distributed GPU communication implementation. It should copy the message semantics.

Reference areas:

```text
srt/managers/data_parallel_controller.py
srt/managers/scheduler.py
srt/managers/io_struct.py
```

Core module:

```text
core/communicator.py
```

Interfaces:

```text
send(rank_id, message)
broadcast(message)
receive(rank_id)
drain()
```

Default backend:

```text
InProcessCommunicator
```

It models logical ranks in one process and preserves ordering rules needed by scheduler, cache, and failover simulation.

Test backend:

```text
TraceCommunicator
```

It replays recorded messages and validates expected message order.

Message types include generate request, batch result, cache operation, failure injection, recovery, health check, flush cache, and metrics flush. The exact message classes are SGLSim-owned DTOs.

## Failover Design

Failover becomes a first-class SGLSim controller instead of behavior patched into SGLang scheduler/cache internals.

Reference areas:

```text
srt/managers/scheduler.py failure handlers
srt/managers/cache_controller.py
srt/mem_cache/hiradix_cache.py remote backup behavior
benchmark/fault_tolerance/src/failure
```

Core module:

```text
core/failover.py
```

Responsibilities:

- track rank health state;
- resolve fault injection triggers;
- mark affected running or queued requests;
- consult cache metadata for checkpointed tokens and recoverable pages;
- create retry or reroute plans;
- update request failover metadata;
- emit failure, retry, recovery, and cache recovery events;
- restore rank state on recovery.

Failover should support baseline, host backup, and remote backup strategies. Strategy-specific behavior belongs in small policy objects rather than in the scheduler loop.

## Runtime Flow

The independent runtime loop is:

```text
1. Adapter receives benchmark, HTTP, or CLI request.
2. Adapter creates SimRequest.
3. Runtime enqueues request into SchedulerEngine.
4. SchedulerEngine selects the next SimBatch.
5. CacheController performs prefix match and schedules metadata-only cache operations.
6. Predictor estimates batch latency.
7. Runtime advances simulated clock.
8. CacheController records backup, prefetch, and load latencies.
9. FailureController checks fault and recovery conditions.
10. SchedulerEngine processes batch completion and retry decisions.
11. MetricsWriter emits request, iteration, cache, and failover outputs.
```

Blocking mode can optionally sleep according to simulated latency. Offline mode advances only the simulated clock.

## Output Compatibility

The new runtime must preserve the files consumed by existing benchmarks and plotters:

```text
metrics.json
request.jsonl
iteration.jsonl
events.jsonl
failover_events.jsonl
trace_log.json
job_summary.csv
task_summary.csv
cache_hits.csv
internal_failover_metrics.json
```

Field-level compatibility is part of the acceptance criteria. New fields may be added only when old consumers continue to work.

## SGLang Reference Usage

SGLang remains useful, but only outside the authoritative path.

Allowed uses:

- generate golden fixtures from the pinned reference commit;
- compare scheduler decisions for small deterministic workloads;
- compare prefix cache match and eviction behavior;
- compare failover events and retry outcomes;
- audit future SGLang updates before changing the SGLSim reference version.

Forbidden uses on the authoritative path:

- importing SGLang scheduler, cache, model runner, entrypoint, or distributed runtime modules;
- monkey patching SGLang internals to produce benchmark results;
- treating copied SGLang files as the implementation of SGLSim core.

## Migration Plan

### Phase 0: Freeze Reference And Contracts

- Confirm the pinned SGLang commit.
- Record the current authoritative small-workload outputs as fixtures.
- Define canonical DTOs, event schemas, and metrics schemas.
- Add import guard tests for `sglsim.core`.

Exit criteria:

- fixtures exist for fixed baseline, fixed remote backup, and a small perf workload;
- schema tests verify current output compatibility;
- import guard fails if core imports SGLang runtime modules.

### Phase 1: Build Core Runtime For Offline Simulation

- Implement `SimRequest`, `SimBatch`, `RankState`, `RuntimeEvent`, and configuration DTOs.
- Implement deterministic offline clock.
- Move `sglsim-fault-sim` onto `core/runtime.py`.
- Keep output files compatible with current offline estimator.

Exit criteria:

- existing `sglsim-fault-sim` examples pass using core runtime;
- output schema matches current estimator outputs;
- no SGLang runtime import is needed.

### Phase 2: Port Scheduler Semantics

- Implement queueing, prefill, decode, running batch, retry queue, and rank assignment.
- Add golden tests for batch sequence and request lifecycle.
- Add differential tests against the old native path for small deterministic cases.

Exit criteria:

- scheduler unit tests pass for fixed workloads, queue timing, retry priority, and chunked prefill cases;
- differential tests show equivalent batch-level behavior for covered cases.

### Phase 3: Port KV And Cache Semantics

- Implement metadata-only radix prefix cache.
- Implement page allocator and tiered residency.
- Implement backup, prefetch, load, and eviction metadata operations.
- Add golden tests for cache hit tokens, page residency, backed-up tokens, and prefetch-complete tokens.

Exit criteria:

- cache unit tests cover prefix matching, eviction, host/remote residency, and transfer latency;
- benchmark cache outputs match fixtures for covered workloads.

### Phase 4: Port Failover Semantics

- Implement rank failure and recovery state machine.
- Implement affected request detection.
- Implement retry/reroute planning.
- Implement baseline, host backup, and remote backup strategy policies.
- Emit compatible failover and cache hit events.

Exit criteria:

- fault injection and recovery tests pass without SGLang runtime imports;
- remote backup cache detail tests pass from core runtime outputs;
- internal failover metrics remain compatible.

### Phase 5: Switch Authoritative Benchmarks

- Route `benchmark/fault_tolerance` to the SGLSim core runtime.
- Keep old native parity path as an optional reference command during transition.
- Update docs to mark SGLSim core as authoritative.

Exit criteria:

- fault-tolerance benchmark runs without `python/sglsim/srt/managers`, `mem_cache`, or `entrypoints`;
- current integration tests pass against the core runtime;
- old native path is no longer required for authoritative results.

### Phase 6: Remove Runtime Vendoring

- Delete or quarantine copied SGLang runtime files that are no longer used.
- Remove `tools/sync_sglang_native_runtime.py` from the normal workflow.
- Keep only reference/differential tooling that points to an external SGLang checkout or fixture directory.

Exit criteria:

- package import does not require SGLang runtime dependencies;
- dependency list reflects CPU-only simulator requirements;
- parity/reference tests are optional and clearly separated from normal tests.

## Test Strategy

### Unit Tests

- scheduler lifecycle and batching;
- queue priority and retry behavior;
- radix prefix matching;
- page allocation and freeing;
- tiered cache residency;
- transfer latency estimation;
- failover trigger, retry, and recovery state machines;
- communicator ordering.

### Contract Tests

- fixed workload request metrics;
- cache hit CSV fields;
- failover event JSONL fields;
- benchmark summary fields;
- internal failover metrics inputs.

### Golden Tests

Golden fixtures are generated from the pinned reference behavior and stored as small, reviewable files. Golden tests compare SGLSim core output against:

- batch sequence;
- request lifecycle;
- prefix hit tokens;
- page residency;
- retry/reroute events;
- final metrics.

### Differential Tests

During migration, differential tests run the old native path and new core path on the same small deterministic workloads. They compare behavior at semantic boundaries rather than internal object identity.

## Compatibility And Deprecation

The current native patch path remains available during migration as `sglang-reference` or `legacy-native` mode. It is not the final authoritative path. Once Phase 5 exits, docs and tests should treat core runtime as authoritative and native patch mode as optional reference tooling.

Existing benchmark command shapes should be preserved where practical. If a command changes, the old command should fail with a clear message pointing to the replacement.

## Risks And Mitigations

Risk: SGLSim core diverges from SGLang semantics.

Mitigation: pin a reference commit, use golden fixtures, run differential tests during migration, and require explicit review for reference commit updates.

Risk: rewrite scope expands into full SGLang reimplementation.

Mitigation: define non-goals and implement only the semantics needed for CPU-only scheduling, KV metadata, communication semantics, failover, and benchmark outputs.

Risk: output compatibility breaks benchmark consumers.

Mitigation: make output schemas part of contract tests and keep old fields stable.

Risk: communication semantics are underspecified.

Mitigation: represent rank communication as typed messages and validate ordering with trace-based tests.

Risk: cache behavior is too approximate.

Mitigation: test prefix matching, page allocation, tier residency, backed-up token counts, and prefetch-complete token counts independently before integrating with scheduler and failover.

## Acceptance Criteria

The design is complete when:

- `benchmark/fault_tolerance` can run authoritative experiments without importing SGLang runtime modules;
- `sglsim-fault-sim` and perf simulation both use SGLSim core runtime;
- scheduling, KV/cache, communication, and failover semantics are covered by unit and golden tests;
- benchmark output files remain compatible with current consumers;
- SGLang reference usage is confined to fixtures, differential tests, or explicitly named reference tooling;
- normal package import and unit tests do not require the vendored SGLang runtime dependency set.

## Implementation Priority

The recommended first implementation target is a small fixed fault-tolerance workload:

```text
fixed requests
dp_size = 2
remote_backup strategy
one injected failure
one recovery
constant predictor
offline clock
```

This target exercises the important seams: request lifecycle, rank assignment, cache metadata, failure state, retry/reroute, and output compatibility. After it passes, expand to host backup, baseline, chunked prefill, AIConfigurator prediction, and HTTP/blocking mode.

## Self-Review

- No unresolved placeholders remain in this design.
- The design keeps SGLSim core independent while preserving SGLang as a semantic reference.
- The migration is staged so current native parity can remain available until core runtime is proven.
- Acceptance criteria are observable through tests and benchmark outputs.
