# SGLang TP/PP/DP/EP recovery-state audit

Audit date: 2026-07-29.  
Checkout: `/home/comp/zhanzhao/sglang`, branch `multi-level-backup`, commit
`dce5a12be0a4a084f341beec7da9d28a2c3492dd`.

## Recovery object, not just KV

End-to-end failover is:

`T_failover = T_detect + T_exec_ready + T_state_restore +
T_suffix_replay + T_requeue`.

KV protection can reduce state restore and suffix replay. It does not create:

- a replacement process/GPU;
- loaded model or expert weights;
- NCCL/collective groups;
- CUDA graphs/warmup;
- request routing, sampling, streaming, LoRA, or scheduler state.

The live request object keeps `output_ids`, `kv_committed_len`, and
`kv_allocated_len` separately in
`python/sglang/srt/managers/schedule_batch.py` lines 638--703. Here
`kv_committed_len` is local request-valid KV accounting, not a
failure-scope-durable protection commit, while `kv_allocated_len` may include
over-allocation/speculative state. A recovery design therefore needs three
distinct frontiers:

- `K`: request-valid materialized KV;
- `C`: scope-protected/group-committed KV;
- `E`: emitted output plus sampling/stream state.

Physical KV allocation has request-to-token and token-to-KV mappings in
`python/sglang/srt/mem_cache/memory_pool.py` lines 18--25 and 126--150.
Failover must reallocate physical slots on the target.

The current `DecodeKVCacheOffloadManager` is instantiated inside `Scheduler`
(`scheduler.py` lines 838--850) and constructs
`MHATokenToKVPoolHost`/`MLATokenToKVPoolHost` in that scheduler process
(`decode_kvcache_offload_manager.py` lines 34--102). Scheduler death therefore
releases the current pinned allocation. Process-fault coverage requires a
process-external Host-pool owner plus a scope-durable or rebuildable directory;
the current local Host path proves only GPU/device-state-loss coverage while
the scheduler remains alive.

The manifest is itself recovery state. It must survive the same declared fault
as the payload quorum through independent replication/co-storage, or be
rebuildable from self-describing surviving object headers and ACK records. A
sole Scheduler-, tokenizer-, or controller-local directory is not a durable
commit.

The failover path confirms this:

- `scheduler.py` lines 3594--3619 snapshots output tokens, input tokens,
  sampling parameters, streaming state, and backed-up tokens;
- `tokenizer_manager.py` lines 1534--1616 reconstructs a request with remaining
  generation budget, multimedia/LoRA/logprob state, output prefix, failed DP
  rank, and a new remote-backup generation.

## KV size and attention TP

For a standard MHA/GQA model:

`B_unique(n) = n * L * H_kv * (D_k + D_v) * element_size`.

With PP-local layers `L_p` and effective attention TP `T_a`:

`B_rank(n) = n * L_p * max(1, floor(H_kv/T_a)) *
(D_k + D_v) * element_size`.

For supported divisible configurations, aggregate physical amplification is:

`rho_TP = max(1, T_a/H_kv)`.

SGLang replicates KV heads when `H_kv < T_a`:

- `python/sglang/srt/configs/model_config.py` lines 678--685;
- `python/sglang/srt/models/qwen3_moe.py` lines 454--473.

Audited local bf16 configurations:

| Model | Unique KV/token | 32K tokens |
| --- | ---: | ---: |
| Qwen3-0.6B | 112 KiB | 3.5 GiB |
| Qwen3-8B | 144 KiB | 4.5 GiB |
| Qwen3-32B | 256 KiB | 8 GiB |

All have eight KV heads. Aggregate KV is unchanged for attention TP up to 8
when the heads divide evenly; it becomes 2x at TP=16 and 4x at TP=32. Coding
must deduplicate replicated logical heads. Physical copies of one logical KV
head are recovery alternatives; they are not all required components of the
request frontier.

## TP

- Each rank holds its PP-local layers and attention-TP KV heads; allocation
  sizing is in
  `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
  lines 120--169.
- One required rank loss stops the collective.
- A decodable protection extent covers the same token range and PP stage for
  every required unique logical head. For each logical head, one surviving
  full-copy alternative or one scope-valid coded decode quorum is sufficient.
- A replacement still needs weights, communicator, warmup, CUDA graph, and
  request-to-page reconstruction.
- The existing decode offload manager correctly takes `MIN` completion across
  TP ranks, but its later storage operation notification is not a durable
  group commit; see `live_sglang_storage_audit.md`. Its physical-rank `MIN` is
  current implementation behavior, not the desired logical-shard frontier when
  replicated heads provide alternatives.

## PP

- Each stage constructs its own layer range in
  `python/sglang/srt/distributed/utils.py` lines 99--135.
- The request-safe frontier is `min(F_stage)` across all required PP stages.
- A failed stage may restore its local KV or regenerate it from a protected
  boundary activation only when that anchor covers the exact same token extent
  and is stored outside the failed stage's declared fault domain.
- Qwen3-8B at PP=4: one raw bf16 hidden-size-4096 anchor is 8 KiB/token, while
  one nine-layer stage has about 36 KiB/token of aggregate KV before TP
  sharding. The 22.2% ratio is a sizing hypothesis, not an evaluated result.
- HCache physical p. 8 supports stage-local activation-to-KV restoration; it is
  a caching/restoration paper, not a fault-tolerance evaluation.

## DP

- A standard DP replica group collectively contains the corresponding complete
  TP/PP weight/runtime set and is a natural warm failover target when the group
  survives the declared fault domain and retains failover headroom.
- `data_parallel_controller.py` lines 559--708 excludes failed ranks and routes
  to a healthy rank.
- Storing protection in a healthy replica's HBM consumes the same headroom
  needed to absorb failed-worker load.
- A shared process-external local Host object or an independent remote Host
  object, plus a request manifest, is therefore the default; peer HBM is
  exceptional.
- This full-replica statement does not apply to `enable_dp_attention`, which
  partitions global TP into nested attention-DP/attention-TP and MoE groups
  rather than creating independent complete replicas
  (`data_parallel_controller.py` lines 508--526).

## Node/rack coding boundary

A full remote replica outside the failed node can cover source-node loss. A
single remote parity shard cannot do so when every data shard disappears with
the source node. A coded protection claim is valid only if removing every
declared failed node/rack still leaves at least `k` checksum-valid fragments
and a readable/rebuildable stripe manifest. A remote write ACK alone is not
that proof.

## EP/MoE

- EP degree does not directly enter the attention-KV byte formula; effective
  attention TP does.
- Expert routing/compute is separate from attention KV in
  `python/sglang/srt/models/qwen3_moe.py` lines 233--363.
- Expert failure requires weights, placement/routing epoch, and A2A/collective
  repair. KV backup cannot substitute for those.
- If a physical EP rank also owns an attention shard, recovery must restore
  both the logical KV component and expert executability.
- SGLang can stage expert weights in CPU buffers via
  `python/sglang/srt/elastic_ep/expert_backup_manager.py` lines 80--155.
- A committed token step need not separately persist historical expert choices;
  the in-flight step can roll back, while the manifest prevents replay under
  an incompatible expert placement, routing, or weight epoch. Equivalent
  logical experts may be remapped to a new physical placement.

## Peer-HBM decision

Persistent peer-HBM KV is justified only when all are true:

1. soft GPU/device or isolated worker-process failure scope, not
   node/power/rack;
2. sufficiently independent peer ownership and scope-durable metadata, so the
   backup is not reclaimed with the failed process;
3. measured NVLink/NVSwitch or other fast P2P;
4. spare HBM after serving and failover headroom;
5. Host round trip cannot satisfy the RTO;
6. short-TTL, chunked/coded, revocable state;
7. replacement rank, weights, communicator, and routing are already ready.

GhostServe physical p. 5 rejects persistent peer-HBM storage as its general
design and instead computes parity on GPU and stores it in Host memory. The
current gpu10 has two RTX 4090 GPUs connected by `SYS` across NUMA, with no
NVLink. The repository's A100/NVLink copy microbenchmark is useful path
calibration, not a default policy result.
