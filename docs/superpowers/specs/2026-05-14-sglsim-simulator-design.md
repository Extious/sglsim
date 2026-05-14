# SGLSim Simulator Design

## Goal

Build a CPU-only inference simulator in `sglsim` that matches the capabilities of `sglang/tools/sglang-simulator`, while integrating the simulator directly into the `minisgl` runtime instead of using dynamic import/class hooks.

The simulator must preserve Mini-SGLang's scheduling behavior as much as possible: request admission, prefix cache matching, chunked prefill, decode scheduling, streaming responses, and benchmark clients should still exercise the real scheduler paths. Model computation, CUDA kernels, real KV cache tensor storage, and GPU sampling are replaced by deterministic CPU simulation.

## Scope

In scope:

- Add a first-class simulation mode controlled by CLI/API configuration.
- Run the serving stack on CPU without CUDA initialization.
- Skip model weight loading and real attention/MLP execution.
- Preserve scheduler, prefix cache, page table, chunked prefill, and decode control flow.
- Generate deterministic dummy tokens for streaming and offline generation.
- Support `OFFLINE` virtual-clock simulation and `BLOCKING` sleep-based simulation.
- Predict per-iteration inference latency through a pluggable time predictor.
- Record request-level and iteration-level simulation stats.
- Dump `metrics.json`, `iteration.jsonl`, and `request.jsonl`.
- Support synthetic and trace-based benchmark replay, including request timestamps.
- Keep normal GPU inference behavior unchanged when simulation mode is disabled.

Out of scope for the first implementation:

- Numeric correctness of logits or generated text.
- CPU implementations of FlashInfer, FlashAttention, TensorRT-LLM, or MoE kernels.
- Real multi-GPU execution in simulator mode. The simulator config can model TP/DP/EP/PP sizes, but the process should run as a single CPU runtime initially.
- Full HiCache/L2 storage simulation. The state model will reserve fields for load/backup latency so it can be added later.
- Renaming all `minisgl` package paths to `sglsim`. The current codebase still primarily uses `python/minisgl`; simulator code should live under `python/sglsim/simulator` and integrate with `minisgl` until a package rename is done separately.

## Reference Behavior From SGLang

The current SGLang simulator uses hooks to replace or wrap these areas:

- `ModelRunner.initialize`: skip real model initialization, estimate KV capacity, allocate lightweight request/KV pools.
- `ModelRunner.forward`: return empty logits shaped like a real batch.
- `ModelRunner.sample`: return deterministic token ids.
- `Scheduler.__init__`: disable overlap scheduling and initialize the inference time predictor.
- `Scheduler.recv_requests`: ingest simulation request metadata, implement `OFFLINE` future queue, and initialize request stats.
- `Scheduler.get_new_batch_prefill`: record queue exit and prefix reuse.
- `Scheduler.run_batch`: build `ScheduleBatch`, predict latency, sleep or advance virtual time.
- `Scheduler.process_batch_result`: update virtual time and token latencies.
- `Scheduler.profile`: dump metrics and reset simulator state.
- Paged KV allocator kernels: replace Triton kernels with CPU Python implementations.
- Kernel load utilities: bypass GPU-specific extension loading.

In `sglsim`, these behaviors should become explicit branches in `Engine`, `Scheduler`, benchmark, and simulator modules instead of monkey patches.

## Recommended Architecture

### Package Layout

Create the simulator package under `python/sglsim/simulator`:

- `types.py`: shared dataclasses and enums.
- `state.py`: virtual clock and iteration state.
- `config.py`: JSON/env/CLI config loading and validation.
- `metrics.py`: metrics aggregation and JSONL serialization helpers.
- `predictor/base.py`: predictor protocol and `ScheduleBatch` conversion contract.
- `predictor/constant.py`: dependency-free predictor for tests and smoke runs.
- `predictor/aiconfigurator.py`: optional AIConfigurator predictor matching the SGLang simulator.
- `runtime.py`: runtime object that owns config, predictor, state, stats, and dump paths.
- `request.py`: helpers to extract simulation metadata from `SamplingParams`.
- `bench_serving.py`: trace and synthetic replay entry point.

Integrate this package into existing Mini-SGLang modules:

- `python/minisgl/core.py`
- `python/minisgl/engine/config.py`
- `python/minisgl/engine/engine.py`
- `python/minisgl/engine/sample.py`
- `python/minisgl/scheduler/config.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/table.py`
- `python/minisgl/server/args.py`
- `python/minisgl/server/api_server.py`
- `python/minisgl/llm/llm.py`
- `python/minisgl/benchmark/client.py`

### Simulation Runtime

Introduce a `SimulationRuntime` object initialized once per scheduler/engine process when simulation mode is enabled. It owns:

- `SimulationConfig`
- `SimulationState`
- `InferTimePredictor`
- `RequestStatsStore`
- `IterationStatsStore`
- `output_dir`

The runtime should expose a small API:

- `on_request_received(uid, input_len, output_len, metadata)`
- `on_prefill_scheduled(batch)`
- `predict_and_apply_forward_latency(batch)`
- `on_batch_processed(batch)`
- `dump_and_reset()`

This keeps simulator logic out of the main scheduler loop except for narrow call sites.

## Data Model

### SimulationConfig

Fields:

- `enabled: bool`
- `mode: SimulationMode`
- `output_dir: str`
- `config_path: str | None`
- `num_warmup: int`
- `disable_overlap_schedule: bool`
- `model: ModelInfo | None`
- `platform: PlatformConfig | None`
- `scheduler: SchedulerSimConfig`
- `predictor: PredictorConfig`

The JSON config should accept the same top-level ideas as `sglang-simulator`:

```json
{
  "platform": {
    "device": "H200",
    "num_device_per_node": 8
  },
  "scheduler": {
    "data_type": "bfloat16",
    "kv_cache_data_type": "bfloat16",
    "mem_fraction_static": 0.9,
    "max_total_tokens": null,
    "tp_size": 1,
    "ep_size": 1,
    "dp_size": 1,
    "pp_size": 1,
    "backend_name": "sglsim",
    "backend_version": "0.1.0"
  },
  "predictor": {
    "name": "constant",
    "prefill_latency_ms": 10.0,
    "decode_latency_ms": 1.0,
    "prefill_scale_factor": 1.0,
    "decode_scale_factor": 1.0
  }
}
```

`aiconfigurator` support should use the SGLang simulator's mapping logic after the constant predictor path is stable.

### Request Metadata

Extend `SamplingParams` with:

- `custom_params: dict | None = None`

Simulation clients pass:

```json
{
  "simulation": {
    "created_time": 0.0,
    "queue_start": 0.0,
    "server_created_time": 1710000000.0,
    "total_request": 1000
  }
}
```

Rules:

- In `OFFLINE` mode, `created_time` is required for benchmark replay requests.
- `queue_start` is optional and overrides the virtual enqueue timestamp.
- `total_request` is used by trace replay to know when all future requests have arrived.
- In `BLOCKING` mode, `server_created_time` is optional; wall-clock receive time is used if missing.
- Warmup or non-generation requests without simulation metadata are allowed but excluded from simulator metrics.

### ScheduleBatch

The predictor receives only the data it needs:

- `ScheduleRequest.extend_length`
- `ScheduleRequest.past_kv_length`

Mapping from Mini-SGLang:

- Prefill/chunked prefill:
  - `extend_length = req.extend_len`
  - `past_kv_length = req.cached_len`
- Decode:
  - `extend_length = 1`
  - `past_kv_length = req.cached_len`

The conversion should live in simulator code, not inside predictor implementations.

### RequestStats

Fields:

- `rid: str`
- `uid: int`
- `input_length: int`
- `output_length: int`
- `final_reused_tokens: int`
- `prefetch_complete_tokens: int`
- `queue_start: float`
- `queue_end: float`
- `created_time: float`
- `last_event_time: float`
- `gen_token_latencies: list[float]`

Completion is determined by generated token count reaching `output_length` or the scheduler marking the request finished. The simulator benchmark should default to `ignore_eos=True` so dummy tokens do not terminate early.

## Engine Design

### Normal Mode

When `simulation=False`, `Engine` behavior must remain unchanged.

### Simulation Mode

When `simulation=True`, `Engine.__init__` must not call:

- `torch.cuda.is_initialized`
- `torch.cuda.set_device`
- `torch.cuda.Stream`
- CUDA graph capture
- GPU memory queries
- real model creation
- weight loading
- real KV cache pool creation
- real attention backend creation
- real MoE backend creation

Instead:

- Use `torch.device("cpu")`.
- Initialize the global `Context`.
- Create a CPU page table sized by `max_running_req + 1` and simulated max sequence length.
- Create a `TableManager` token pool from CPU page table.
- Create no-op graph runner and no-op event abstractions.
- Store `model_config` from HF config so tokenizer/model metadata remains available.
- Store `vocab_size` for dummy token generation.

`Engine.forward_batch()` in simulation mode:

- Enters `ctx.forward_batch(batch)` so code using context remains consistent.
- Does not call `self.model.forward()`.
- Calls `req.complete_one()` for each request.
- Generates deterministic `next_tokens_cpu` and `next_tokens_gpu`.
- Returns `ForwardOutput` with a `NoOpEvent` whose `synchronize()` is a no-op.

Dummy token policy:

- Default token id should be `1` if `vocab_size > 1`, otherwise `0`.
- If token id equals EOS for a model, benchmark and simulator clients should set `ignore_eos=True`.

## Scheduler Design

### Loop Behavior

In simulation mode, disable overlap scheduling by default. This matches SGLang simulator's simplification and avoids CUDA stream assumptions. The simulator can still model overlap effects later in `SimulationState`.

The scheduler should keep using:

- `PrefillManager`
- `DecodeManager`
- `CacheManager`
- `TableManager`
- `_process_one_msg`
- `_schedule_next_batch`
- `_process_last_data`

### CPU Tensor Path

Several scheduler helpers currently allocate pinned host tensors and transfer to CUDA. Simulation mode needs CPU-only paths:

- `_make_positions`
- `_make_input_tuple`
- `_make_write_tuple`
- `CacheManager.allocate_paged`
- `_write_page_table`

Design rule:

- If target device is CPU, allocate ordinary CPU tensors and do not use `pin_memory=True` or `.to(cuda)`.
- If target device is CUDA, keep existing optimized behavior.

### Request Lifecycle Hooks

Simulation runtime calls should be added at stable scheduler boundaries:

1. On `UserMsg` accepted:
   - Extract simulation metadata.
   - Record `created_time`, `queue_start`, `input_length`, `output_length`.
   - For `OFFLINE`, support a future queue so trace requests become schedulable only after virtual time reaches their timestamp.

2. On prefill batch scheduled:
   - Record first `queue_end`.
   - Record `final_reused_tokens = req.cached_len`.
   - Leave chunked requests open across chunks.

3. On forward:
   - Convert Mini-SGLang batch to simulator `ScheduleBatch`.
   - Predict inference latency.
   - In `BLOCKING`, sleep for predicted latency and measure wall-clock forward duration.
   - In `OFFLINE`, store predicted latency as current inference duration.

4. On batch result processed:
   - Advance virtual clock for `OFFLINE`.
   - Record token latency for each non-chunk request.
   - Append iteration stats with request info and forward latency.

5. On dump/profile:
   - Write metrics and JSONL files.
   - Reset virtual state and stats.

### Offline Future Queue

For trace replay:

- Incoming generation requests with simulation metadata go into a heap ordered by `queue_start or created_time`.
- The scheduler only moves requests from future queue to normal prefill admission when `enqueue_time <= global_clock`.
- If no running/waiting work exists and the next future request is later than the current clock, jump the virtual clock to just after that request's enqueue time.
- Extra requests such as profile/flush/abort should bypass the future queue.

This mirrors the SGLang simulator's `OFFLINE` mode and prevents wall-clock sleep during trace replay.

## API And CLI

### CLI

Add server arguments:

- `--simulation`
- `--sim-config-path`
- `--sim-mode {OFFLINE,BLOCKING}`
- `--sim-output-dir`
- `--sim-num-warmup`

Simulation mode should force:

- `tp_size = 1` for runtime processes initially.
- `offline_mode` unchanged for `LLM`; online server still runs normally through FastAPI/ZMQ.
- `cuda_graph_max_bs = None`.
- overlap scheduling disabled.

The simulator config can still include modeled `tp_size`, `dp_size`, `ep_size`, and `pp_size` for predictor calculations.

### OpenAI API

Extend request parsing to preserve selected `extra_body` fields:

- `ignore_eos`
- `top_k`
- `top_p`
- `input_length_override`
- `simulation`

`simulation` must be stored in `SamplingParams.custom_params`.

### Python LLM API

Allow:

```python
LLM(
    model_path="...",
    simulation=True,
    simulation_config_path="...",
    simulation_mode="OFFLINE",
)
```

For direct `generate`, optional per-request simulation metadata can be added later. The initial direct API can use default created times in request order.

## Benchmark Design

Add `python/sglsim/simulator/bench_serving.py`.

Capabilities:

- Random synthetic workload:
  - number of requests
  - input length range
  - output length range
  - request rate or concurrency
- Trace replay:
  - timestamp
  - input length
  - output length
  - prompt text or generated prompt
- OpenAI-compatible streaming client, reusing `minisgl.benchmark.client` where possible.
- Each request sends `extra_body["simulation"]`.
- Output compatible with existing benchmark result processing.

For `OFFLINE` mode, the benchmark may send all requests quickly and rely on server-side virtual time. For `BLOCKING` mode, the client uses wall-clock request timing.

## Metrics

Dump files:

- `metrics.json`: aggregate metrics.
- `request.jsonl`: serialized `RequestStats`.
- `iteration.jsonl`: per-iteration request shapes and predicted latency.

Aggregate metrics:

- `num_requests`
- `completed`
- `total_input`
- `total_output`
- `duration`
- `request_throughput`
- `input_throughput`
- `output_throughput`
- `total_throughput`
- `prefix_cache_reused_ratio`
- `disk_prefetch_ratio`
- `mean_ttft_ms`, `median_ttft_ms`, `p90_ttft_ms`, `p95_ttft_ms`, `p99_ttft_ms`
- `mean_tpot_ms`, `median_tpot_ms`, `p90_tpot_ms`, `p95_tpot_ms`, `p99_tpot_ms`
- `mean_itl_ms`, `median_itl_ms`, `p90_itl_ms`, `p95_itl_ms`, `p99_itl_ms`, `max_itl_ms`
- `mean_e2e_latency_ms`, `median_e2e_latency_ms`, `p90_e2e_latency_ms`, `p95_e2e_latency_ms`, `p99_e2e_latency_ms`
- `mean_queue_ms`
- `time_cost`

Metric semantics should match SGLang simulator where possible:

- TTFT is the first generated token latency.
- TPOT excludes the first token.
- ITL is per-token latency after the first token.
- E2E is the sum of generated token latencies.
- Queue time is `queue_end - queue_start`.

## Error Handling

Configuration errors:

- Missing config path with `--simulation` is allowed if default constant predictor config is used.
- Unknown predictor names raise `ValueError` during startup.
- `aiconfigurator` predictor raises a clear import/config error if the dependency or database is unavailable.
- Invalid simulation mode raises during CLI parsing.

Request metadata errors:

- In `OFFLINE` benchmark mode, missing `simulation.created_time` excludes the request from metrics and logs a warning.
- In normal online use, missing simulation metadata uses current wall-clock or virtual-clock defaults.
- Non-generation requests are not counted in request metrics.

Runtime errors:

- Simulation mode must fail fast if code tries to initialize CUDA-only graph/attention/model paths.
- CPU tensor helpers should assert tensor device consistency in tests.
- Predictor OOM estimates may return negative latency; the runtime should log the condition and use `abs(latency)` for sleep/clock advancement while preserving the warning in iteration stats.

## Testing Plan

Unit tests:

- `SimulationState` virtual clock increment/reset.
- `ScheduleBatch` properties and decode/prefill classification.
- Constant predictor output for prefill and decode.
- Metrics aggregation for known request stats.
- Request metadata extraction from `SamplingParams.custom_params`.
- CPU tensor helper paths avoid pinned memory and CUDA transfers.

Scheduler/engine integration tests:

- `Engine(simulation=True)` does not initialize CUDA.
- `forward_batch` returns deterministic CPU tokens and a no-op event.
- `LLM(..., simulation=True).generate()` returns `max_tokens` tokens on a CPU-only environment.
- Prefix cache reuse is reflected in `final_reused_tokens`.
- Chunked prefill records queue end once and does not emit sampled tokens for chunk-only steps.
- Decode iterations produce one latency per generated token.

Serving tests:

- Start server with `--simulation` and hit `/v1/chat/completions`.
- Streaming returns the requested number of tokens with `ignore_eos=True`.
- `/flush_simulation` or profile endpoint writes metrics files.

Benchmark tests:

- Random workload benchmark completes with constant predictor.
- Trace replay in `OFFLINE` mode advances virtual time without sleeping for trace timestamps.
- `BLOCKING` mode performs real sleep approximately matching configured latency.

Regression tests:

- Normal non-simulation engine path still imports and preserves existing CLI defaults.
- Existing scheduler/cache unit tests continue to pass with CUDA assumptions guarded or skipped where appropriate.

## Implementation Phases

### Phase 1: Simulator Core

Create `python/sglsim/simulator` with types, state, config, metrics, constant predictor, and runtime scaffolding. Add tests that require only CPU and no Mini-SGLang server startup.

### Phase 2: CPU Engine Mode

Add simulation fields to config and implement CPU-only `Engine` initialization and `forward_batch`. Add no-op graph runner/event abstractions. Verify direct offline generation can run without CUDA.

### Phase 3: Scheduler Integration

Add CPU tensor paths, request metadata handling, schedule batch conversion, latency prediction, virtual-clock updates, and stats collection. Keep call sites narrow by using `SimulationRuntime`.

### Phase 4: API/CLI Integration

Add server arguments, OpenAI extra body propagation, `SamplingParams.custom_params`, and a dump/profile endpoint or command path. Verify online streaming behavior.

### Phase 5: Benchmark Replay

Add simulator benchmark entry point for random and trace workloads. Support `OFFLINE` future queue and `BLOCKING` wall-clock behavior.

### Phase 6: AIConfigurator Predictor

Port the SGLang simulator's AIConfigurator predictor and model/platform config mapping. Keep it optional so the constant predictor remains the default smoke-test path.

### Phase 7: Hardening And Documentation

Add docs, examples, error messages, and regression tests. Document known limitations around real multi-process TP and HiCache simulation.

## Acceptance Criteria

- `uv run python -m minisgl --simulation --model ...` starts without requiring a GPU.
- CPU-only `LLM(..., simulation=True).generate()` produces deterministic token ids.
- Online streaming returns exactly `max_tokens` tokens when `ignore_eos=True`.
- Random benchmark produces `metrics.json`, `request.jsonl`, and `iteration.jsonl`.
- Trace replay in `OFFLINE` mode uses virtual time and does not wait for real trace intervals.
- `BLOCKING` mode sleeps according to predictor output.
- Prefix cache reuse and chunked prefill affect scheduler behavior and stats.
- Normal Mini-SGLang GPU mode remains behaviorally unchanged when simulation is disabled.
