# Live SGLang KV-storage audit

## Scope and provenance

- Checkout: `/home/comp/zhanzhao/sglang`
- Branch / commit: `multi-level-backup` /
  `dce5a12be0a4a084f341beec7da9d28a2c3492dd`
- Describe: `v0.5.10-20-gdce5a12be0`
- Audit date: 2026-07-29
- This is a source-path audit of the live checkout. It is separate from the
  `profiler@fdebc938f7` execution evidence in `sglang_profiler/`.

## What the checkout actually implements

### The decode backup path is staged through Host memory

`python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`:

- lines 58--76 create a decode-side `HostKVCache`;
- lines 150--180 copy each new, stride-aligned KV extent asynchronously from
  device memory into that Host pool;
- lines 182--235 take the minimum completion count across TP ranks, wait for the
  D2H event, release the corresponding GPU KV pages, and only then enqueue the
  storage backup;
- lines 261--286 release the staging Host pages after consuming the current
  storage operation-completion notification.

The concrete state transition is therefore:

`GPU KV -> pinned Host KV pool -> storage backend attempt -> operation
completion notification -> release Host staging`.

This is not equivalent to a direct GPU-to-disk commit, and an asynchronous copy
is not yet a recoverable checkpoint before a scope-specific commit.

More importantly, the current `ack_backup_queue` must **not** be interpreted as
such a commit. `cache_controller.py` lines 1256--1261 stop a batched page write
when `page_set_func` fails, but lines 1298--1300 still enqueue the operation
unconditionally. `decode_kvcache_offload_manager.py` lines 261--270 consumes
that notification and releases all Host indices without checking
`completed_tokens`, the expected page count, a checksum, or a quorum/fault
domain. Thus the current queue records that the worker finished processing an
operation, not that every requested page is durably recoverable. A proposed
Protection Fabric needs an explicit success bitmap/count and
scope-specific/group-level commit before releasing its last protected payload
or advancing a frontier.

### The Host pool is currently a performance/cache pool, not a complete
protection abstraction

`python/sglang/srt/mem_cache/memory_pool_host.py` lines 151--209 allocate one
pinned Host KV pool. Its capacity is configured by an absolute size or a
Host-to-device ratio. The current protocol even requires the Host pool to be
larger than the device pool (lines 183--185). The pool does not distinguish:

- an evictable performance-cache page;
- a local GPU-failure snapshot;
- a cross-node recoverable replica/parity page;
- the committed version/fault domain of a protection copy.

`python/sglang/srt/managers/cache_controller.py` lines 279--389 attaches
separate write/load streams and background storage queues to this same Host
pool. Lines 1160--1200 enqueue Host-to-storage backup, and lines 1237--1300
advance page completion in a background thread.

This makes a unified pool feasible, but correctness needs an additional
protection lease/refcount and an independent-failure-domain commit record.
Merely finding a page in local Host memory, or receiving the current
operation-completion notification, is not proof that it survives a node
failure.

There is also an ownership boundary. `scheduler.py` lines 838--850 instantiate
`DecodeKVCacheOffloadManager` inside the Scheduler, and
`decode_kvcache_offload_manager.py` lines 34--102 construct and own the Host
pool from that manager. A genuine Scheduler-process exit destroys that
process-owned allocation. The current implementation therefore demonstrates
local protection against GPU/device-state loss, not process-crash durability.
An AegisKV process-fault claim requires a process-external Host-pool
owner/shared-memory daemon plus independently durable commit metadata, or a
remote protection domain.

### The remote backend is cross-worker storage, but one server is one failure
domain

`python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_storage.py`
lines 5--15 describe the data path explicitly:

`Host L2 -> serialize -> TCP -> remote-server DRAM`,

and restore in the reverse direction through local Host L2. Lines 61--80 expose
server/client/combined modes; the default combined mode co-locates the server
with a worker. Lines 281--348 serialize Host pages and send them to that
server.

`remote_backup_server.py` lines 5--24 define a centralized 1:N TCP service with
one shared DRAM radix buffer. Lines 26--37 add per-request leases and active /
inactive LRUs, which are useful building blocks for a shared cache/protection
pool. They do not add replication or erasure coding across multiple remote
servers. A single server ACK therefore protects against a source GPU/worker
failure only to the extent that the server is in an independent failure
domain; it does not by itself protect against failure of that server or its
node/rack.

### TP completeness is already a group property

`decode_kvcache_offload_manager.py` lines 186--200 all-reduce the available
write/backup completion counts with `MIN` across the TP group before advancing
progress. This is the correct direction: a request-level frontier cannot be
declared complete because only one TP shard finished.

`python/sglang/srt/configs/model_config.py` lines 678--685 also show an important
capacity effect: SGLang shards KV heads by attention TP, but replicates a KV
head when TP is larger than the number of KV heads so every GPU keeps at least
one head. Thus high TP does not monotonically reduce the aggregate KV footprint.

For a dense GQA model with `L` local layers, `H_kv` total KV heads, K/V head
dimensions `D_k,D_v`, scalar size `s`, and effective attention TP
`T_a=get_attention_tp_size()`, the aggregate physical bytes per token across
the attention-TP group are:

`B_phys = L * (D_k + D_v) * s * T_a *
max(1, floor(H_kv / T_a))`.

For bf16 and `D_k=D_v=D`, this reduces to the former
`2 * L * D * 2 bytes * ...` expression. When `T_a <= H_kv` and divides it,
this equals the unsharded KV size. When `T_a > H_kv`, the aggregate grows by
`T_a / H_kv` because KV heads are replicated.

## Concrete capacity examples from cached model configs

The locally cached Hugging Face configs give:

| Model | Layers | KV heads | Head dim | bf16 KV bytes/token | KV for 32K tokens |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 8 | 128 | 114,688 B | 3.50 GiB |
| Qwen3-8B | 36 | 8 | 128 | 147,456 B | 4.50 GiB |
| Qwen3-32B | 64 | 8 | 128 | 262,144 B | 8.00 GiB |

A full peer-GPU copy consumes the same aggregate capacity again when
`TP <= 8`. At `TP=16`, SGLang's one-head-per-GPU rule doubles the aggregate
Qwen3 KV footprint. TP reduces each rank's shard, but it does not make a
second full GPU-resident protection group cheap.

PP partitions those bytes by layer across stages. The aggregate remains the
same. For protection, replicated physical heads are alternatives: each required
logical attention shard must retain a surviving copy or decodable quorum, and
the request frontier is the minimum compatible frontier across those logical
shards and required PP stages.

## Existing physical-copy calibration and its boundary

The repository contains three-run A100/NVLink action-level microbenchmarks:

- `physical_recovery_path_aggregate.csv`, SHA256
  `ac33b26a602bbb420194d01b757620cbfa28b37383fc33c862de4eb03758a6f1`
- `physical_kv_copy_aggregate.csv`, SHA256
  `8116cf42d4e0a4921ae3b8153e88225dfe9017b28fe25314f41a0550004403b8`

For an 8,192-token Qwen3-8B-shaped KV extent (1,152 MiB), the recorded
recovery-shaped p50 values are approximately 46.5 ms D2H, 48.4 ms H2D, and
4.3 ms peer P2P on two NV12-connected A100s. This proves that same-node NVLink
can provide a fast copy path. It does **not** prove that peer HBM is the best
default protection tier:

- the copy shares a node-level failure domain;
- the backup consumes serving HBM;
- the artifact is an action microbenchmark, not an end-to-end policy result;
- the current `gpu10` host has two RTX 4090 GPUs connected through `SYS`
  across NUMA nodes, with no NVLink, so the A100 ratio cannot be generalized to
  this host.

## Design consequences

1. Reuse the existing Host offload page whenever possible; add protection
   metadata and a remote/coded, scope-specific commit instead of issuing a
   second D2H copy.
2. Treat local Host, remote Host replica/parity, and recompute as distinct
   failure-scope actions.
3. Manage Host capacity and copy-path bandwidth as separate resources.
4. Publish a vector frontier over TP/PP components; expose a request-level safe
   frontier only when the required expected-page set, checksum, topology epoch,
   independent-failure-domain components, and scope-durable manifest are
   committed. The lookup directory must be quorum-replicated outside the scope
   or rebuildable from self-describing commit markers co-stored with the
   protected objects.
5. Do not make peer-GPU KV replication the default. Reserve it for measured,
   very-low-slack cases with spare HBM and an independent enough failure scope.
