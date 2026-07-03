# Synthesis Requests Benchmark

Aligned with SGLang `bench_serving` + `generated-shared-prefix` dataset. Starts `launch_server` and drives `POST /v1/completions`.

## Layout

```
benchmark/synthesis_requests/
├── run.py
├── client.py           # burst client (request_rate=inf) + max_concurrency
├── server.py
├── utils.py            # GSP workload generation (same semantics as SGLang)
├── config/
│   ├── client.json     # dataset + gsp_* + traffic
│   ├── server.json
│   └── real_trace_20260529_133143/
├── logs/
└── results/YYYYMMDD_HHMMSS/
```

## Run

```bash
cd benchmark/synthesis_requests
uv run python run.py
```

No-overlap-first（建议先做语义/时延基线）：

```bash
uv run python benchmark/synthesis_requests/run.py \
  --config-dir benchmark/synthesis_requests/config/no_overlap \
  --disable-overlap-schedule
```

## client.json

### Traffic (bench_serving)

| Field | SGLang CLI | Default |
|-------|------------|---------|
| `request_rate` | `--request-rate` | `"inf"` (**required**; burst at start) |
| `max_concurrency` | `--max-concurrency` | `null` (unlimited) |

`request_rate` must be `"inf"` or `null`. Load shaping uses `max_concurrency` (cap in-flight HTTP POSTs). Finite request rates are not supported in this benchmark.

### GSP dataset (`generated-shared-prefix`)

| Field | SGLang CLI | Default |
|-------|------------|---------|
| `dataset_name` | `--dataset-name` | `generated-shared-prefix` |
| `gsp_num_groups` | `--gsp-num-groups` | 64 |
| `gsp_prompts_per_group` | `--gsp-prompts-per-group` | 16 |
| `gsp_system_prompt_len` | `--gsp-system-prompt-len` | 2048 |
| `gsp_question_len` | `--gsp-question-len` | 128 |
| `gsp_output_len` | `--gsp-output-len` | 256 |
| `gsp_range_ratio` | `--gsp-range-ratio` | 1.0 |
| `gsp_fast_prepare` | `--gsp-fast-prepare` | false |
| `gsp_send_routing_key` | `--gsp-send-routing-key` | false |
| `gsp_num_turns` | `--gsp-num-turns` | 1 |
| `gsp_ordered` | `--gsp-ordered` | false |
| `seed` | `--seed` | 1 |

Total requests: `gsp_num_groups * gsp_prompts_per_group`.

Tokenizer for workload generation uses `server.json` `model_path`. Use `mock-model` there for fast local runs.

## server.json

Simulator (`platform`, `predictor`, `scheduler`, `cache`) and `host` / `port` / `model_path`.

Default `config/server.json` enables **overlap scheduling** and **HiCache L1+L2 only** (`enable_hierarchical_cache: true`, `hicache_ratio: 2.0`, `hicache_write_policy: write_through`, `l3_capacity_pages: 0`). L3 is off. Disable HiCache with `enable_hierarchical_cache: false`.

`run.py` 现支持 `--disable-overlap-schedule`，会透传到 `sglsim.launch_server`。也可在 `server.json` 的 `scheduler.disable_overlap_schedule` 中固定关闭 overlap。

> 对齐 SGLang 语义：当 `pp_size > 1` 时，SGLSim 会强制关闭 overlap schedule。

### Calibration profile

`config/real_trace_20260529_133143/` is a real-trace-compatible profile for the provided SGLang run. It keeps the same generated-shared-prefix workload, disables HiCache, disables overlap scheduling, sets `page_size=1`, and uses `mem_fraction_static=0.83` so request-level cache hits and logical latencies can be compared against the real trace before predictor calibration.

`config/no_overlap/` 是 mini workload（16 组）no-overlap 档位，用于快速回归。

`config/parity_full_no_overlap/` / `config/parity_full_overlap/` 是与 2026-06-07 SGLang 参考对齐的 **full workload**（30 组 × 2 prompts × 3 turns = 180 请求，`max_concurrency=30`，`mem_fraction_static=0.836`）。对比参考：

- no-overlap：`results/with_hicache_without_overlap_sglang`
- overlap：`results/with_hicache_with_overlap_sglang`

```bash
# No-overlap first（必须先收敛再跑 overlap）
uv run python benchmark/synthesis_requests/run.py \
  --config-dir benchmark/synthesis_requests/config/parity_full_no_overlap \
  --disable-overlap-schedule

# Overlap（仅 no-overlap 达标后）
uv run python benchmark/synthesis_requests/run.py \
  --config-dir benchmark/synthesis_requests/config/parity_full_overlap
```

## Notes

- `gsp_fast_prepare: true` sets `prompt_len=1` when generating (same as SGLang); POST uses `input_len`.
- `gsp_send_routing_key: true` sends `X-SMG-Routing-Key` per group.
- `gsp_num_turns > 1`: same workload as SGLang (`turn_prompts` list per session). Client runs **sequential** `POST /v1/chat/completions` per turn with growing `messages` (user/assistant), matching SGLang `bench_serving` + `sglang-oai-chat`. Each turn uses rid `{session_rid}-t{turn}`; `request_details.csv` has one row per turn. `gsp_fast_prepare` is ignored for multi-turn (messages carry full text).
- `gsp_workflow_shape`: optional staged agentic workflow shape. For example
  `[1, 3, 1, 3, 1]` creates one workflow group with five stages and nine agent
  sessions. The client enforces a per-workflow stage barrier: stage `k + 1`
  starts only after all requests in stage `k` finish; agents inside the same
  fan-out stage run concurrently subject to `max_concurrency`. Different
  workflow groups can overlap. `gsp_stage_long_decode_agent_index` marks one
  fan-out agent as decode-dominant and multiplies its output length by
  `gsp_long_decode_output_multiplier`; singleton stages and decode-dominant
  fan-out agents are marked `critical_agent=true`.
- `gsp_failover_replay: true`: after a protected candidate turn completes, the
  client sends a real HTTP retry request with rid `{session_rid}-t{turn}-retry`
  and `simulation.failover_replay=true`. This is a benchmark-level replay
  smoke: the original request is not killed, but the retry row can be bracketed
  by runtime failure-control endpoints. `gsp_failover_injection=soft` calls
  `/simulate/fail_dp` before the retry and `/simulate/recover_dp` afterward.
  `gsp_failover_injection=hard-restart` calls `/simulate/kill_dp`, sends the
  retry while the source scheduler subprocess is dead, and then calls
  `/simulate/restart_dp`. By default, the retry row is forced to
  `(failed_dp_rank + 1) % dp_size`; `gsp_failover_retry_rank_offset` can
  override that offset for controlled recovery-rank experiments such as
  non-peer recompute regimes. The replay exercises the live
  tokenizer -> scheduler -> detokenizer path with failed-rank and restore-plan
  metadata. The Engine parent resolves `restore_tokens`,
  `restore_recompute_tokens`, `restore_action`, and `restore_overhead_s` from
  its runtime HostBackup and peer GPUReplica directories before dispatching the
  replay request. The scheduler sleeps for `restore_overhead_s` before admitting
  the replay request, so the restore plan is visible in wall-clock latency and
  queue time. For Host/GPU restore actions, the scheduler also materializes the
  restored prompt prefix as target-rank device-cache metadata before replay
  admission; this is reported separately from total protected KV tokens because
  a failure point can include decoded tokens that are not part of the replay
  prompt prefill.
- The default restore-overhead model intentionally preserves the current
  replay-calibrated formulas. To run physical-copy-calibrated SGLSim
  experiments, set `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy` and
  `SGLSIM_AEGIS_PHYSICAL_COPY_CSV` to the aggregate CSV produced by
  `benchmark/AegisKV/draft/experiments/sglang_small/scripts/aggregate_physical_kv_copy_microbench.py`.
  This uses the measured H2D and P2P p50 curves for Host/GPUReplica restore
  copy components while keeping recompute cost on the existing model.

## Server process model (aligned with SGLang SRT)

`sglsim.launch_server` follows the same launch order as SGLang HTTP mode:

1. Start scheduler subprocess(es) and wait until ready (`ConfigManager` + predictor loaded; no GPU weights).
2. Start detokenizer subprocess (loads tokenizer, decodes `output_ids`).
3. Initialize `TokenizerManager` in the HTTP parent (loads tokenizer, tokenizes prompts).
4. Serve FastAPI; `/health` returns 503 until step 1-3 complete.

Request path: `POST /v1/completions` or `POST /v1/chat/completions` (multi-turn GSP) -> `TokenizerManager` (encode + wall-clock tokenize sleep) -> scheduler subprocess (predict + wall-clock sleep) -> detokenizer subprocess (decode) -> parent drain -> response.

Simulation uses real wall-clock time; benchmark `duration` is `max(last_event_time)` from completed requests.

`dp_size=1` still uses a scheduler subprocess (not an in-process thread).

## SGLang alignment

Workload generation follows SGLang `generated-shared-prefix` semantics: pre-generate system prompts and questions, build `turn_prompts = [f"{system}\n\n{q0}"] + q[1:]`, and compute `prompt_len` via tokenizer encode (or `1` when `gsp_fast_prepare` is true). The HTTP client sends prompt text (not client-side `input_ids`); the server encodes again with the same tokenizer rules. With `mock-model`, mock encode is used on both sides; lengths are consistent but not HF-equivalent.

Optional `num_prompts` truncates the generated list (same as SGLang `--num-prompts`).

## Cache

Prepared requests and raw workload are cached under `benchmark/synthesis_requests/cache/`:

| File | Content |
|------|---------|
| `{key}_workload.pkl` | `list[WorkloadRequest]` |
| `{key}_prepared.pkl` | `list[PreparedRequest]` (HTTP bodies) |
| `{key}_manifest.json` | config fingerprint, counts, timestamp |

Cache key is a SHA-256 hash of `server.model_path` plus all GSP client fields (`seed`, `gsp_*`, `dataset_name`, `num_prompts`, etc.). A second run with the same config loads from cache (`Loading cached ...` in `logs/client.log`). Legacy cache at `~/.cache/sglsim/benchmark` is ignored.

Even on prepared-cache hit, the server still encodes/decodes each request at runtime (workload `prompt_len` is for stats; server encode is authoritative).

## Progress logs

GSP generation uses `tqdm` phases written to `logs/client.log` via the client logger:

- `Generating system prompts` (skipped when `gsp_fast_prepare`)
- `Generating questions`
- `Computing prompt lengths` (non-fast-prepare encode loop)

Server tokenizer load and encode wall time appear in `logs/server.log`.

## dp_size > 1

Each DP rank runs in its own scheduler subprocess with a **process-local** wall clock. HTTP tokenize wall time is measured in the parent and passed via `simulation.tokenize_dur` to the rank that handles the request; decode wall time is stepped in that rank's scheduler. Ranks do not share a unified logical clock across processes. Benchmark `duration` remains `max(last_event_time)` across completed requests.

## AegisKV metadata columns

The HTTP client forwards workflow metadata through the request `simulation`
payload:

- `agent_id`
- `critical_agent`
- `workflow_id`
- `stage_id`
- `stage_width`
- `stage_local_agent_index`
- `decode_dominant`
- `workflow_shape`
- `backup_kv`
- `worker_seq`
- `assigned_dp_rank`

The scheduler does not change admission or batch lifecycle based on these
fields. It records an online AegisKV bookkeeping view in `sim_stats` and
`request_details.csv`:

- `aegis_policy`
- `aegis_backup_kv`
- `aegis_worker_seq`
- `aegis_assigned_dp_rank`
- `aegis_host_visible_tokens`
- `aegis_gpu_replica_tokens`
- `aegis_safe_frontier_tokens`
- `aegis_recompute_tokens`
- `aegis_recovery_action`
- `aegis_expected_recovery_ms`
- `aegis_candidate_retry`
- `aegis_candidate_failed_dp_rank`
- `aegis_candidate_pre_failure_output_tokens`
- `aegis_candidate_pre_failure_total_tokens`
- `aegis_candidate_backed_up_tokens`
- `aegis_candidate_recompute_tokens`
- `aegis_candidate_recovery_action`
- `aegis_candidate_failover_overhead_s`
- `aegis_failover_replay`
- `aegis_failed_source_dp_rank`
- `aegis_replay_restored_tokens`
- `aegis_replay_restored_prefill_tokens`
- `aegis_replay_recompute_tokens`
- `aegis_replay_restore_action`
- `aegis_replay_restore_overhead_s`
- `aegis_replay_restore_sleep_s`
- `aegis_replay_restore_data_plane`
- `aegis_replay_restore_timing_source`
- `aegis_cross_rank_retry`
- `aegis_soft_failure_injected`
- `aegis_hard_failure_injected`
- `aegis_host_backup_lookup_hit`
- `aegis_host_backup_lookup_tokens`
- `aegis_peer_gpu_lookup_hit`
- `aegis_peer_gpu_lookup_tokens`

These columns are evidence plumbing for AegisKV experiments. They are not a
complete failure-recovery implementation by themselves. The `aegis_candidate_*`
columns are an online what-if recovery event computed from the observed request
frontier at completion, using the same request metadata, cache-hit counters, and
Aegis-visible Host/GPUReplica frontier. They do not abort or retry the live
request. The `aegis_failover_replay` rows are real HTTP retry requests, but
they are still benchmark-level replay rows rather than scheduler crash/recovery.
For replay rows, `aegis_soft_failure_injected=true` means the benchmark client
called the runtime soft-failure control plane before dispatching the retry.
`aegis_hard_failure_injected=true` means the client killed the recorded source
scheduler subprocess before dispatching the retry and restarted it after the
retry returned.
`aegis_host_backup_lookup_hit=true` means the Engine parent found a runtime
HostBackup directory entry for `(agent_id, worker_seq, failed_dp_rank)` and used
it to fill the replay restore metadata.
`aegis_peer_gpu_lookup_hit=true` means the replay target is the source rank's
recorded peer rank and a runtime GPUReplica directory entry exists for that
same `(agent_id, worker_seq, failed_dp_rank)` recovery point. The runtime still
chooses the lower-envelope action among recompute, Host restore, and peer GPU
restore.
`aegis_assigned_dp_rank != aegis_failed_source_dp_rank` means the retry-shaped
request was dispatched cross-rank by the simulation-only `force_dp_rank`
metadata.
`aegis_replay_restored_prefill_tokens` is the subset of restored KV that was
materialized as target-rank device-cache metadata and can therefore reduce the
replay prefill. Real CUDA tensor KV copy still needs validation before these
results can be marked final.
`aegis_replay_restore_data_plane` records whether replay restore used metadata
materialization, recompute, or future live physical tensor movement. Current
SGLSim rows use `metadata_materialization` for Host/GPUReplica restores.
`aegis_replay_restore_timing_source` records whether the sleep came from default
coefficients, a raw physical-copy CSV, or a recovery-shaped physical movement
CSV.
If `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy` is enabled, the reported
`aegis_replay_restore_overhead_s` uses the configured physical-copy aggregate
for H2D/P2P copy time. The runtime still materializes metadata, not real CUDA
KV tensors, so this is a calibrated timing run rather than end-to-end physical
movement evidence.
