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
