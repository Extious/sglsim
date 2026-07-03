# 05. 实现计划

## 1. 真实 SGLang 目标

Remote target inspected:

```text
hkbugpumanode:~/sglang
branch: multi-level-backup
```

Relevant source paths:

```text
python/sglang/srt/managers/cache_controller.py
python/sglang/srt/mem_cache/hiradix_cache.py
python/sglang/srt/mem_cache/radix_cache.py
python/sglang/srt/mem_cache/memory_pool.py
python/sglang/srt/mem_cache/memory_pool_host.py
python/sglang/srt/mem_cache/hicache_storage.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/managers/data_parallel_controller.py
python/sglang/srt/managers/tokenizer_manager.py
python/sglang/srt/managers/io_struct.py
```

Observed implementation facts that must shape AegisKV:

- `HiCacheController.write` handles GPU -> Host D2H and emits `ack_write_queue`。
- `HiCacheController.load` handles Host -> GPU H2D and emits `ack_load_queue`。
- `HiRadixCache.write_backup` sets `host_value` and tracks `ongoing_write_through`。
- write-through ACK is processed by `writing_check`; SGLang releases the temporary lock only after D2H completion。
- `HiRadixCache.evict` respects `lock_ref`; Host/GPU eviction must invalidate Aegis frontiers correctly。
- `match_prefix` distinguishes device-resident prefix from host-only prefix; host-only KV needs explicit load/restore。
- scheduler periodically calls `check_hicache_events` and `flush_write_through_acks`。

Simplifying assumption for this draft:

```text
Host Backup always uses fixed write_through D2H immediately after radix insertion.
Baseline also performs this D2H, but does not expose failed-rank Host KV to retry ranks.
```

## 2. SGLang 实现里程碑

### 里程碑 1：Metadata plumbing

Add optional `aegis` metadata to request input and propagate it through:

```text
tokenizer_manager -> Req -> scheduler -> radix cache
```

Fields:

```text
workflow_id
agent_id
agent_type
parent_agent_id
parallelism_degree
lifecycle_state
round_id
segment_type
critical_agent
serving_dp_rank
retry_dp_rank
retry_generation
```

Validation:

- request without `aegis` metadata behaves exactly as baseline SGLang；
- same prompt tokenization and output；
- no measurable latency overhead for metadata parsing。

### 里程碑 2：Radix node Aegis metadata

Attach metadata to touched radix nodes:

```text
TreeNode.aegis_meta:
  workflow_refs
  host_state
  peer_states
  safe_cover
  checkpoint_epoch
  last_decision
```

Hook points:

- match/insert path；
- node split；
- request lock release；
- `write_backup` submission；
- `writing_check` ACK；
- GPU eviction；
- Host eviction。

Validation:

- metadata follows node split and host_value span split；
- terminal lifecycle removes live refs；
- ordinary prefix cache behavior unchanged；
- Host frontier is invalidated when host pages are freed。

### 里程碑 3：固定 HostBackup directory

Implement HostBackup as a recovery directory over the existing fixed `write_through` path:

```text
radix insert / KV segment materialized
  -> HiRadixCache.write_backup
  -> HiCacheController.write
  -> wait for ack_write_queue
  -> writing_check observes ACK
  -> publish HostBackupEntry(source_dp_rank, host handle, token span)
```

HostBackupEntry fields:

```text
source_dp_rank
source_worker_id
workflow_id / agent_id
radix_path
token_span
host_page_ids or host handle
checkpoint_epoch
model / tokenizer / kv_layout / page_size
acked / invalidated
```

Validation:

- baseline write_through D2H still runs, but retry cannot use failed-rank Host KV；
- AegisKV Host-backed retry can restore prefix from failed/source rank Host memory before suffix replay；
- pending Host copy is never used as safe frontier；
- Host eviction invalidates the corresponding HostBackupEntry；
- no-failure p95/p99 overhead is compared against the same fixed-write-through baseline。

### 里程碑 4：GPUReplica 路径

Implement peer DP-rank copy:

```text
planner chooses GPUReplica(peer_dp_rank)
  -> source rank queues peer copy
  -> peer rank allocates replica GPU pages under budget
  -> peer records safe-cover metadata and optional radix prefix entry
  -> peer ACK updates source/frontier registry
```

Peer selection inputs:

- peer queue load；
- peer HBM free pages；
- shared-prefix locality；
- expected retry route；
- P2P bandwidth/queue；
- failure-domain correlation。

Validation:

- retry to peer rank gets prefix hit or peer restore；
- P2P copy duration and peer HBM pressure are logged；
- peer eviction/demotion invalidates `peer_states`；
- peer copy does not corrupt ordinary radix cache state。

### 里程碑 5：Shared frontier 与 restore-vs-recompute

Implement content-level safe frontier:

```text
frontier content: token prefix [0:k]
choices: GPUReplica(peer), Host, recompute suffix
```

Failure path:

```text
failure detected
  -> affected requests snapshotted
  -> retry requests constructed
  -> deepest safe frontier found
  -> compare peer restore, Host H2D, recompute
  -> request completes
```

Metrics:

- recovered tokens；
- recomputed tokens；
- retry latency；
- workflow completion time；
- peer copy time；
- Host D2H/H2D time；
- normal-path overhead。

## 3. SGLSim 目标

Remote target inspected:

```text
gpu10:~/sglsim
branch: aegiskv
```

Current visible source paths:

```text
python/sglsim/srt/mem_cache/hicache.py
python/sglsim/srt/mem_cache/hiradix_cache.py
python/sglsim/srt/mem_cache/radix_cache.py
python/sglsim/srt/mem_cache/transfer.py
python/sglsim/srt/mem_cache/memory_pool.py
python/sglsim/srt/mem_cache/memory_pool_host.py
python/sglsim/srt/managers/scheduler.py
python/sglsim/srt/managers/data_parallel_controller.py
python/sglsim/srt/managers/cache_controller.py
python/sglsim/srt/communication/predictor.py
python/sglsim/srt/simulation/pipeline_accounting.py
benchmark/synthesis_requests/
benchmark/synthesis_requests/diagnostics/
benchmark/agentic_workflows/
```

Important current limitation:

```text
python/sglsim/srt/mem_cache/aegis.py is not present as source.
```

`benchmark/agentic_workflows/` now has visible scenario/config, online runner,
suite, summary, and paper-candidate export source built on
`benchmark/synthesis_requests`. Therefore SGLSim is currently useful for
HiCache/radix/pipeline calibration and repeatable preliminary workflow
experiments, but final AegisKV policy and physical KV movement must still be
added before it can produce main-paper final evidence.

### 当前 worktree 中的实验性实现

The current large-scale AegisKV evidence path is implemented as online SGLSim
serving plus benchmark-level failover replay. It is stronger than offline CSV
post-processing, but it is still not a physical KV tensor copy protocol.

Request construction:

```text
benchmark.agentic_workflows.config
  -> materialize named scenario configs
  -> benchmark.synthesis_requests.run
  -> generated-shared-prefix workload
  -> real HTTP /v1/chat/completions or /v1/completions requests
```

For multi-turn agentic sessions, each session carries a shared-prefix prompt and
then sends sequential chat turns:

```text
rid = {session_rid}-t{turn_idx}
messages = previous user/assistant history + current user turn
simulation.worker_seq = turn_idx
simulation.agent_id = session/agent id
simulation.critical_agent = selected by gsp_critical_agent_ratio
simulation.backup_kv = critical_agent
```

The server path remains the normal online serving path:

```text
HTTP parent
  -> TokenizerManager
  -> DataParallelController
  -> per-DP Scheduler subprocess
  -> DetokenizerManager
  -> response.sim_stats
```

The scheduler records AegisKV bookkeeping fields in `sim_stats` and
`request_details.csv`, but does not change admission control, prefill/decode
batch lifecycle, or SGLang-style scheduling based on the Aegis metadata.

Current AegisKV bookkeeping:

```text
Host frontier:
  critical/Aegis-visible requests publish aegis_host_visible_tokens.

GPUReplica frontier:
  critical requests with backup_kv publish a metadata-level hot-tail
  aegis_gpu_replica_tokens frontier.

Safe frontier:
  max(Host-visible, GPUReplica, ordinary cache/load-back coverage).

Candidate retry:
  aegis_candidate_* fields describe a what-if failure point after the
  original request completes. They do not interrupt the original request.
```

Failover replay:

```text
if gsp_failover_replay and aegis_candidate_retry:
  client sends {rid}-retry as a real HTTP request
  simulation.failover_replay = true
  simulation.failed_dp_rank = aegis_candidate_failed_dp_rank
  simulation.retry_dp_rank = failed_dp_rank + configured offset
  simulation.force_dp_rank = retry_dp_rank
```

Failure injection modes:

```text
soft:
  /simulate/fail_dp before replay
  /simulate/recover_dp after replay
  source rank is marked unavailable, subprocess remains alive

hard-restart:
  /simulate/kill_dp before replay
  replay runs while source Scheduler subprocess is dead
  /simulate/restart_dp after replay
```

Runtime restore-plan resolution:

```text
Engine parent lookup key:
  (agent_id, worker_seq, failed_dp_rank)

If found:
  compute Host-visible tokens
  compute peer GPUReplica tokens if retry rank matches peer rank
  choose lower-envelope action among recompute, Host restore, GPUReplica
  stamp restore_tokens / restore_recompute_tokens / restore_action /
  restore_overhead_s into the replay request

Scheduler:
  sleep(restore_overhead_s)
  materialize restored prompt prefix as target-rank metadata cache for
  Host/GPU restore actions
```

This implementation can support preliminary evidence for cross-rank retry,
runtime Host/GPUReplica directory lookup, lower-envelope restore decisions, and
metadata-level restored-prefix effects. It cannot yet support final claims about
real peer GPU KV tensor P2P copy, CUDA page ownership, or complete SGLang
end-to-end recovery under high-concurrency failure injection.

## 4. SGLSim 实现里程碑

### 里程碑 1：保持 HiCache fidelity

Keep existing SGLSim behavior:

- `hicache.py` predicts D2H/H2D duration；
- `pending_backup_dur` models outstanding D2H；
- scheduler overlaps or serializes H2D, forward, and D2H backup；
- `hiradix_cache.py` models fixed `write_through` for this paper scope；
- radix eviction distinguishes GPU eviction, Host eviction, and Host-only load-back。

Validation:

- current SGLSim/SGLang HiCache calibration figures remain valid；
- no Aegis policy change is allowed to break baseline parity gates。

### 里程碑 2：加入 Aegis metadata model

Add simulator-side classes:

```text
AegisRequestMeta
AegisWorkflowRef
AegisNodeMeta
AegisDecision
AegisFrontier
```

Hook them into:

- request custom params / tokenizer manager；
- `ScheduleRequest` / `Req`；
- `HiRadixCache.prepare_prefill` and insert path；
- `write_backup` and `writing_check`；
- `evict_from_tree_cache` and `evict_host`；
- scheduler lifecycle events。

### 里程碑 3：加入 policy 与 copy accounting

Implement policy variants:

```text
none
uniform_host
uniform_peer_gpu
static_critical_path
aegiskv
```

Metrics to add:

```text
aegis_host_copy_tokens
aegis_peer_copy_tokens
aegis_shared_prefix_dedup_tokens
aegis_peer_locality_score
aegis_peer_load_score
aegis_frontier_acked_tokens
aegis_frontier_invalidated_tokens
recomputed_tokens
restored_tokens
```

### 里程碑 4：加入 peer GPU model

Add a simulator peer-copy path:

```text
source DP rank
  -> communication predictor path P2P
  -> target DP rank replica budget
  -> peer ACK
  -> peer eviction/demotion
```

The peer model must expose:

- per-rank replica HBM budget；
- per-rank serving load；
- per-rank prefix locality；
- P2P copy time；
- failure retry routing。

### 里程碑 5：Online workflow benchmark wrapper

Visible source now exists under `benchmark/agentic_workflows/`:

```text
config.py
run.py
suite.py
summary.py
combine.py
paper.py
aggregate.py
```

These scripts generate named workflow scenarios, materialize per-trial
`synthesis_requests` configs, run or dry-run the real serving benchmark, and
export preliminary paper-candidate rows. The final online suite still must run
real simulator components:

- `Engine` / `DataParallelController` / per-DP `Scheduler`；
- radix tree and HiCache D2H/H2D paths；
- Aegis metadata, Host ACK, PeerGPU ACK；
- failure injection and retry lifecycle。

Offline counterfactual code can exist, but main paper results should come from online simulator runs.

## 5. 论文实现口径

The paper should distinguish:

| Source | Main role |
|---|---|
| SGLang | real-system feasibility, overhead, concrete D2H/H2D/P2P/retry latency |
| SGLSim | scale-up sweeps, failure-timing matrix, peer-selection sensitivity, long-context stress |
| Offline analysis | appendix-only counterfactual or oracle upper bound |

Do not present offline model results as real online recovery evidence.

## 6. 首次投稿的最低实现要求

Required:

1. SGLang fixed `write_through` HostBackup directory with ACKed frontier。
2. SGLang failure injection and restore-vs-recompute decision。
3. SGLang microbenchmarks for D2H/H2D/P2P/recompute boundary。
4. SGLSim Aegis metadata and Host/GPUReplica policy hooks。
5. SGLSim online workflow suite for policy comparison and scale sweeps。
6. Correctness tests for radix lock, split, Host eviction, peer invalidation, and unsafe frontier rejection。

Nice to have:

- DRLPart-style budget controller；
- RemoteStore backend；
- multi-host failure-domain study。
