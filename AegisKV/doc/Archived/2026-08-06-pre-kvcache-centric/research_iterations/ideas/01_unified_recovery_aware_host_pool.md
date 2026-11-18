# Idea 1: Unified Recovery-Aware Host Memory Pool

## Claim

Use one physical pinned Host-memory pool for ordinary KV offload/cache and
fault-tolerance, but give pages typed leases. Protection should normally
**promote an already-offloaded page by metadata** rather than perform a second
GPU-to-Host copy.

The protection scope depends on who owns that pool. A worker-process-local
allocation survives a GPU/device-state reset only while the worker remains
alive. To cover a worker-process crash, AegisKV must place the payload and its
directory under a process-external node daemon/shared-memory owner, or
materialize the state in another process or fault domain.

This is the recommended primary AegisKV mechanism because it turns KV backup
into a classic capacity, bandwidth, tenancy, and lifecycle-management problem.

## Mechanism

Each Host page has one physical payload and one or more logical leases:

| Lease | Evictable? | What it proves |
| --- | --- | --- |
| `CACHE` | yes | performance reuse only |
| `LOCAL_PROTECT` | no until expiry | survives GPU/device-state loss; survives worker-process loss only with a process-external pool owner and durable directory |
| `REMOTE_SOURCE` | after remote ACK | source for an in-flight remote copy |
| `REMOTE_REPLICA` | no until expiry | survives source-node loss if fault domain is independent |
| `PARITY` | no until stripe expiry | contributes to a stripe whose surviving fragments still meet the decode quorum after the declared failure |
| `TRANSFER_BUFFER` | yes after I/O completion | no recovery value by itself |

The page directory stores:

`<content_id, request_id, version, token_range, logical_component,
topology_epoch, pool_owner_epoch, lease_types, coding_set, fault_domain,
checksum, refcount, expiry, ACK_set, commit_generation>`.

The directory is part of protected state, not an assumed immortal controller.
Every committed page/fragment carries a self-describing header, and the compact
commit record is replicated or coded at the same or stronger failure scope as
the payload. After a controller/process loss, a new directory can be rebuilt by
scanning surviving headers. A request frontier is not published until both its
data quorum and its scope-durable manifest record are recoverable.

Algorithm 2 follows a short mechanism chain:

1. look up whether the exact version/range already exists in the Host pool;
2. if yes, acquire a protection lease and checksum it without a second D2H;
3. verify that the payload owner and directory survive the declared failure;
4. if local Host satisfies that scope, commit locally;
5. otherwise use the same payload as the source of a full remote replica or a
   coded placement whose surviving fragments meet the decode quorum;
6. commit only after the data ACK/quorum and scope-durable manifest ACK;
7. downgrade to recompute if the capacity or path lease cannot meet the slack.

## Resource model

Let `M_host` be the manageable pinned Host pool. Algorithm 3 publishes

`C_protect(K) = alpha(K) * M_host`.

Here `alpha` is a protection-capacity reservation, not a second memory
allocation. A page with both `CACHE` and `LOCAL_PROTECT` leases is charged once
physically and once to protection accounting. A page may lose its evictable
cache lease while its protection lease remains.

Bandwidth is separate:

`B_p(K) = beta_p(K) * measured_capacity_p`

for D2H, Host memory bus, RNIC, and remote Host paths. The figure may show one
aggregate `beta`, while an implementation expands it to a per-path vector.
Capacity admission and path-credit acquisition must both succeed.

## Why local Host needs optional extra redundancy

Local Host memory is enough for a GPU/device reset if its allocation and
directory remain alive. It covers a worker-process crash only when another
process owns the pool and commit metadata. It is not enough for node power
loss, kernel failure, or rack failure. The requested failure scope therefore
selects:

- GPU/device: `LOCAL_PROTECT` in surviving Host memory;
- worker process: process-external `LOCAL_PROTECT`, or another process/domain;
- node: one full `REMOTE_REPLICA`, or a coded placement with at least `k`
  surviving fragments after loss of the entire source node;
- rack: a full replica outside the rack, or a decode quorum that remains after
  loss of the source rack;
- no strict recovery need: `CACHE` or recompute only.

The system never creates a second local Host payload merely to call it a
backup. Extra redundancy crosses a real fault-domain boundary.

## Evidence

- The live SGLang checkout already executes
  `GPU KV -> pinned Host KV pool -> storage write attempt -> operation
  completion notification`, and then releases Host staging pages; the current
  notification is not a full-page or durable commit; see
  [live storage audit](evidence/live_sglang_storage_audit.md).
- In the live checkout, `Scheduler` constructs
  `DecodeKVCacheOffloadManager`, which allocates its Host pool in that scheduler
  process. The current path therefore does not by itself prove worker-process
  crash survival; a process-external owner is an AegisKV implementation delta.
- The same audit shows that the current Host pool lacks typed protection
  leases, committed versions, failure-domain identity, process-external
  ownership, and a scope-durable/rebuildable directory. These are concrete
  implementation deltas.
- GhostServe, the only direct 2025--2026 KV fault-tolerance paper in the
  collected corpus, stores parity in Host memory rather than keeping a
  persistent peer-HBM replica (local PDF physical pp. 5--6).
- Mooncake, Strata, IMPRESS, NEO, OPKV, and FlexiCache establish that Host
  capacity and D2H/RNIC/storage traffic are already shared with performance
  caching/offload. Their state is not automatically fault-tolerant; see the
  [KV per-paper review](../../related_works/kvcache_backup/per-paper-review.md).
- Training systems show both the conventional GPU-to-Host-to-storage pipeline
  and newer in-memory/shadow alternatives. Host residence is a staging state,
  not synonymous with a committed checkpoint.

## Falsifiable experiment

Implement three modes in the same SGLang workload:

1. duplicate D2H backup into a separate Host buffer;
2. unified pool with zero-copy lease promotion;
3. unified pool plus remote parity/replica.

Measure:

- D2H bytes and Host write bandwidth per useful committed byte;
- pinned-Host peak capacity and cache hit rate;
- no-failure TTFT/TPOT/P99;
- committed-frontier coverage;
- GPU/device, worker-process, Host-owner, and node-failure recovery success;
- stale-page and premature-eviction violations.

The idea fails if typed leases cannot prevent cache eviction races, if page
aliasing/checksum overhead erases the avoided D2H benefit, or if unifying the
pool causes worse fragmentation than separate pools.

## Patent-friendly mechanism boundary

The protectable mechanism is a concrete state machine:

`cache lookup -> owner/scope check -> lease promotion -> optional remote
materialization -> data quorum + scope-durable manifest commit -> committed
frontier -> dependency-aware release`.

The claim does not depend on a complex global objective function.
