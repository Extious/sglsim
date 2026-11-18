# AegisKV final modeling report

Status: design authority for the 2026-07-29 system overview.  
Supersedes the peer-HBM-centric `alpha/beta` interpretation in earlier
iteration notes.

## 1. Final answer

The strongest model is:

> **AegisKV is a recovery-aware, multi-resource KV protection broker.**
> Algorithm 1 supplies a conservative failure-time recovery budget and failure
> scope. Algorithm 2 keeps each request inside a recoverable dirty-frontier
> envelope by acquiring typed Host-capacity, path-bandwidth, fault-domain, and
> recovery-target leases. Algorithm 3 changes those resource budgets across
> `K -> K+1` using a guarded, cause-coded state machine.

The physical default is a **unified Host Memory Pool**, not persistent
peer-GPU HBM:

- ordinary KV offload/cache pages are evictable;
- an existing page can be promoted to a non-evictable protection lease without
  a second D2H copy;
- local Host protects GPU/device-state loss; process-fault protection additionally
  requires a process-external Host-pool owner (or remote domain) and
  independently durable metadata;
- node/rack failure requires a full remote replica or a coded placement whose
  surviving fragments satisfy the decode quorum after that declared failure;
- peer HBM is an emergency option only when spare HBM, a fast path, a narrow
  soft-GPU failure scope, and an ultra-low RTO all hold.

The final meanings are:

- `alpha_n(K)`: protected-capacity share of node `n`'s manageable unified Host
  pool;
- `beta_{n,p}(K)`: foreground-safe backup bandwidth share for path `p`
  (`D2H`, Host DRAM, RNIC, remote DRAM, optional storage).

The overview shows one aggregate `beta` for readability; an implementation
must use a per-path vector with rate, burst, priority, and shared-resource
ownership.

This is an explicit redesign. It is not the current meaning of
`aegis_l1_l2.py`, where `alpha/beta` are L1/L2 capacity ratios.

## 2. Why this is a classic resource-management problem

The model borrows the useful abstraction, not a large objective function:

- Dominant Resource Fairness represents a job with a multi-resource demand
  vector rather than one “slot.”
- Tetris-style packing distinguishes capacity on several bottleneck resources
  and avoids declaring a job feasible from only one free resource.
- Gandiva-style introspection motivates feedback-driven allocation when
  workload resource use is predictable only after observation.
- Modern checkpoint systems separately manage device copy, Host buffers,
  network/storage drain, versions, and restart capacity.

For AegisKV, a protection job demands:

`<Host bytes, D2H credits, Host-bus credits, RNIC credits, remote bytes,
fault-domain lease, warm recovery target>`.

The mechanism remains a finite state machine:

`classify -> acquire -> schedule -> copy/promote -> verify -> commit -> release`.

Complex equations are offline calibration aids. They are not the patent or
fast-path mechanism.

Primary resource-management references:

- [DRF, NSDI 2011](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types);
- [Tetris, SIGCOMM 2014](https://wisr.cs.wisc.edu/papers/tetris.pdf);
- [Gandiva, OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/xiao).

## 3. What the 60-paper audit actually supports

The file-level audit is in
[CORPUS_AUDIT.md](../../related_works/CORPUS_AUDIT.md).

### 3.1 KV-cache corpus

The directory contains 30 validated 2025--2026 PDFs, 515 pages, and 30 unique
SHA-256 values:

- **1/30 direct KV fault-tolerance paper:** GhostServe;
- **29/30 adjacent systems papers:** tiering, restoration, offload,
  disaggregation, compression, scheduling, and execution.

This scarcity is itself an important result. HCache restores KV from an
intermediate activation, and Mooncake replicates hot blocks for load relief;
neither is direct failure backup. The full boundaries and evidence pages are
in:

- [KV manifest](../../related_works/kvcache_backup/manifest.csv);
- [KV per-paper review](../../related_works/kvcache_backup/per-paper-review.md).

GhostServe is the direct baseline:

- TP GPUs keep their original KV data shards;
- GPU resources transiently encode parity;
- parity is offloaded to Host memory rather than persistently occupying another
  GPU's HBM;
- recovery may combine parity restoration with recomputation;
- benefit depends on TP size and can disappear at TP=2;
- cross-node topology-aware placement and dynamic removal of failed GPUs remain
  outside its demonstrated scope.

Therefore the literature does **not** support peer-GPU KV backup as the default
2025--2026 design pattern.

### 3.2 Training checkpoint corpus

The directory contains 30 validated PDFs, 522 pages, and 30 unique SHA-256
values:

- main conference: 28;
- peer-reviewed workshop: 1;
- work in progress: 1;
- runtime snapshot/state replication: 11;
- format/loading/empirical I/O: 4;
- diagnosis/recovery/validation/planning without a new write path: 15.

Among the 11 runtime snapshot/state-replication systems:

- 9/11 explicitly use local, peer, or dedicated-server Host DRAM as the first
  state layer;
- of those 9, 5 explicitly continue to persistent storage;
- 1 mentions Host+SSD without a drain/commit protocol;
- 3 evaluated prototypes stop in Host memory.

The correct conclusion is:

> **Host DRAM is the dominant fast first layer in the direct subset, but
> Host-to-disk/remote persistence is not a universal endpoint.**

The path must be separated into:

1. snapshot creation;
2. D2H or network-side reconstruction;
3. Host residence;
4. remote/persistent drain;
5. scope-specific recoverable commit.

Representative differences:

- PCcheck pipelines GPU -> pinned DRAM -> SSD/PMEM and has a persistence/publication
  boundary.
- ByteCheckpoint pipelines D2H, serialization, shared-memory dump, and
  HDFS/NAS upload; a distributed integrity barrier defines completeness.
- FlowCheck reconstructs checkpoint state in CPU servers from mirrored gradient
  traffic instead of copying a GPU snapshot.
- Checkmate maintains a CPU shadow state from multicast gradients and does not
  implement a disk commit in its evaluated prototype.
- MoEvement keeps sparse distributed checkpoints and PP-boundary logs in peer
  Host memory.
- AdaCheck uses peer Host state plus a background persistent full checkpoint.

The full evidence is in:

- [training manifest](../../related_works/training_checkpoint/manifest.csv);
- [training per-paper review](../../related_works/training_checkpoint/per-paper-review.md).

## 4. Real SGLang profiler result and backup timing

The experiment actually ran the SGLang `profiler` branch at
`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` from an independent worktree.
The user's original checkout stayed on
`multi-level-backup@dce5a12be0a4a084f341beec7da9d28a2c3492dd`.

Configuration:

- gpu10 physical GPU 1, RTX 4090 24 GiB;
- Qwen3-0.6B, TP=1;
- Triton attention;
- eager, prefill/decode CUDA graphs disabled;
- PyTorch 2.9.1+cu128;
- non-exclusive shared host.

Stage-filtered trace:

| Metric | Prefill | Decode |
| --- | ---: | ---: |
| selected steps | 3 | 5 |
| median step duration | 24.013 ms | 22.819 ms |
| median kernel-union busy | 22.219 ms | 1.850 ms |
| median kernel-union occupancy | 92.51% | 8.15% |
| observed gap median | 0.672 us | 42.593 us |
| observed gap p95 | 10.848 us | 161.985 us |
| maximum observed gap | 514.852 us | 2,304.111 us |

Manipulated interference:

- pinned D2H buffer: 256 MiB;
- chunk: 16 MiB, synchronized per chunk;
- duration: 40.000 s;
- throughput: 25.416 GB/s;
- decode 1->64, n=6/group:
  baseline/interference/post medians =
  677.124/718.677/652.764 ms;
- interference versus the mean of the two control medians: **+8.08%**.

Prefill did not show a stable slowdown in the short test. That is not evidence
of free D2H bandwidth.

The usable timing conclusion is:

- prefill is dense in this configuration: pause backup by default or allow only
  tiny control chunks after an active guard;
- decode is only a candidate opportunity: use micro-chunks, revocable path
  credits, and foreground-SLO feedback;
- a kernel-free trace gap does not prove copy-engine, HBM, PCIe, NIC, or
  system-wide bandwidth availability;
- TP/PP/DP/EP, CUDA graphs, continuous batching, and exclusive GPUs require
  separate profiles.

Artifacts:

- [profiler report](../ideas/evidence/sglang_profiler/REPORT.md);
- [timeline metrics](../ideas/evidence/sglang_profiler/stage_timeline_metrics.json);
- [trace and artifact hashes](../ideas/evidence/sglang_profiler/SHA256SUMS).

## 5. Correct Slack semantics

Algorithm 1's Slack is a failure-time recovery budget:

`T_detect + T_reroute + T_restore + T_recompute(dirty suffix) <= Slack`.

It is **not** the deadline of a copy started now.

For request `i`, keep three different frontiers:

- `K_i`: local materialized-KV frontier;
- `C_i`: scope-specific group-committed KV frontier;
- `E_i`: emitted-output frontier already exposed to the client;
- `dirty_i = K_i - C_i`;
- `q_i`: conservative failure-time recompute throughput;
- `D_i`, `R_i`: detection/reroute and restore/setup budgets.

`E_i` is recovery metadata, not an alias for `K_i`: ordinary autoregressive
execution can offset them, and speculative decoding can separate them further.

The safely replayable dirty suffix is approximated by:

`U_i = floor(q_i * max(0, Slack_i - D_i - R_i))`.

If KV grows at `g_i` tokens/s, a protection job expiry can be derived from:

`T_due_i = (U_i - dirty_i) / g_i`.

Algorithm 2 must commit enough state before the derived expiry. If no
restore/recompute path can meet the original recovery budget, the output is
`RECOVERY_SLO_INFEASIBLE`, not “Recompute is safe.”

## 6. Algorithm 2: per-request dual-path decision

### 6.1 Step 1: build a protection job

Inputs:

- recovery budget and derived dirty-frontier expiry;
- request/version/token extent;
- criticality and DAG dependency lifetime;
- current physical locations;
- failure scope: GPU, process, node, rack;
- TP/PP/DP/EP topology epoch and group manifest.

### 6.2 Step 2: enumerate a bounded action set

1. no-copy defer/recompute;
2. zero-copy promotion of an existing Host cache/offload page;
3. new local Host copy;
4. remote Host replica;
5. remote Host parity;
6. PP boundary anchor;
7. emergency peer HBM.

### 6.3 Step 3: require a dual certificate

Every accepted action has two feasibility certificates:

**Normal-time commit path**

- enough `alpha` capacity;
- every `beta_p` path credit;
- fault-domain lease;
- expected page set, version, checksum, topology epoch;
- completion before the derived dirty-frontier expiry.

**Failure-time recovery path**

- surviving local/remote/parity sources;
- remote read/H2D/parity-decode capacity;
- a warm or reservable replacement GPU;
- model weights and communicator/routing reconstruction;
- suffix recompute estimate;
- total recovery within the original Slack.

A successful write without the second certificate is stored state, not an RTO
guarantee.

### 6.4 Step 4: acquire atomic leases

The allocator acquires:

- Host protected-capacity lease under `alpha`;
- token buckets on each shared path under `beta_p`;
- fault-domain lease;
- no-failure SLO headroom;
- replacement-target reservation or an explicit restoration-capacity plan.

A shared RNIC, remote pool, or rack link has one admission owner. Per-node
controllers cannot independently oversubscribe it.

### 6.5 Step 5: chunk, verify, commit

The proposed state machine is:

`planned -> copying/promoted -> verified -> scope-ACKed -> group-committed ->
superseded -> reclaimed`.

Commit contains:

- expected pages and successful pages;
- checksum;
- request generation and topology epoch;
- fault-domain ACK/quorum/parity set;
- TP/PP group-vector KV frontier;
- a separate emitted/output-token frontier plus sampling/stream state.

Only the longest contiguous request frontier whose required group components
are committed may be published. A physical KV slot is never a portable object
identifier; a target worker reallocates pages and reconstructs the
request-to-token-to-KV mapping.

The **logical group-commit boundary** includes both payload durability and
scope-durable manifest publication:

`payload ACK/quorum + scope-durable manifest ACK -> group-committed`.

An idempotent implementation first writes self-describing tentative fragments,
then publishes a commit manifest containing their content hashes, locations,
code parameters, topology epoch, and frontier. For process/node/rack scope, the
manifest must be acknowledged by a control-plane quorum outside the declared
failure domain, or a final commit marker must be co-stored with enough
self-describing payload fragments to survive that failure. The lookup
directory is only a cache and must be rebuildable by scanning those durable
markers. Payload without a durable manifest and manifest without a surviving
payload/decode quorum are both uncommitted and fall back to recomputation.

### 6.6 Window K example

Budget:

- `alpha(K)=0.10`;
- aggregate `beta(K)=0.30`, expanded internally to a path vector.

Requests:

- **R1, S=8 ms, GPU scope, Host cache hit:** add a `LOCAL_PROTECT`
  lease to the exact existing page, verify and commit; no second D2H.
- **R2, S=45 ms, node scope:** use existing Host source or chunked D2H,
  acquire RNIC/remote capacity, and reserve a restore target. Commit either one
  full replica outside the source node, or a coded placement whose surviving
  fragments still provide a decode quorum after the declared node failure.
- **R3, S=180 ms, replay cheap:** dirty suffix remains inside its recovery
  envelope; defer now and create a protection job before the derived expiry.

These are different results from the same Algorithm 2, not three
application-specific formulas.

## 7. Unified Host Memory Pool and redundancy

One physical page may carry typed leases:

| Lease | Eviction | Meaning |
| --- | --- | --- |
| `CACHE` | allowed | performance only |
| `LOCAL_PROTECT` | blocked to expiry | survives GPU/device-state loss; process fault only with external ownership |
| `REMOTE_SOURCE` | after remote completion | source for remote materialization |
| `REMOTE_REPLICA` | blocked to expiry | survives source-node loss if domain independent |
| `PARITY` | blocked to stripe expiry | reconstructs declared stripe |
| `TRANSFER_BUFFER` | after operation | no protection by itself |

The scope-durable manifest stores:

`<content, request, version, token range, topology epoch, lease types,
fault domain, checksum, refcount, expiry, commit set, code parameters,
payload locators>`.

The in-memory metadata directory indexes these manifests, but is not itself the
durability boundary. It is independently quorum-replicated outside the
declared scope or rebuildable from self-describing commit markers co-stored
with the protected objects.

This directly integrates CPU offload/cache and protection:

1. look up exact Host-resident page;
2. promote by metadata if local scope suffices;
3. use the same payload as the remote-copy/parity source if node/rack scope is
   required;
4. charge physical memory once;
5. prevent cache eviction while a protection lease is active;
6. reclaim by TTL and workflow dependency refcount.

Local Host needs no second same-node payload for a GPU/device-state failure.
The live SGLang Host pool is Scheduler-process-owned, so surviving a Scheduler
crash additionally requires AegisKV to move pool ownership to a
process-external shared-memory/Host daemon and make its manifest independently
durable; otherwise use a remote domain. Node/rack scope needs either a full
remote replica or a coded placement whose decode quorum survives the declared
failure, not merely one remote parity page.

## 8. Algorithm 3: `K -> K+1`

At the end of `K`, seal cause-coded telemetry:

- dirty-frontier overflow and useful committed bytes;
- capacity admission miss versus path admission miss;
- Host cache/protection occupancy, eviction, fragmentation;
- D2H/Host/RNIC/remote queue, achieved bandwidth, revocation and fail-slow;
- TTFT/TPOT/P99, throughput, OOM, cache/prefetch interference;
- restore/recompute result and replacement-target readiness;
- fault-domain/quorum failure.

The ordered governor is:

1. **Safety backoff.** Foreground SLO failure reduces the responsible path
   `beta_p` first; Host capacity pressure may reduce new `alpha` admission.
2. **Capacity adapt.** Urgent capacity overflow plus safe headroom increases
   `alpha`; stale/unused protection decreases it.
3. **Path rebalance.** Move credits away from a hot/fail-slow bottleneck.
4. **Hold.** Deadband, minimum dwell, and cooldown prevent oscillation.
5. **Canary publish.** Atomically publish one budget epoch; roll back on guard
   regression.

Example observations in `K`:

- protected Host capacity reaches 97%;
- urgent capacity admission overflows;
- sustained D2H trips the decode-interference guard;
- existing Host cache pages have high promotion value;
- remote RNIC has headroom.

Publish:

- `alpha(K+1)=0.16`;
- aggregate `beta(K+1)=0.18`;
- reduce new D2H credits;
- move available path share toward remote materialization from already-Host
  pages.

Window `K+1` runs the **same Algorithm 2** with budget epoch `vK+1`; its
candidate action set is unchanged, but its feasible set and admission outcomes
change because `alpha` and the per-path `beta_p` credits changed.

A lower budget applies to new admission. Existing committed leases are:

- grandfathered until TTL/dependency expiry; or
- migrated by `new commit -> old reclaim`.

An in-flight copy stops only at its declared extent boundary. This prevents a
window update from invalidating already-published recoverability.

## 9. TP / PP / DP / EP analysis

### TP

With `L` layers, `H_kv` KV heads, K/V head dimensions `D_k,D_v`, scalar size
`s`, and effective **attention** TP size `T_a`, the audited SGLang behavior
gives aggregate physical bytes per token:

`B_phys = L * (D_k + D_v) * s * T_a *
max(1, floor(H_kv/T_a))`.

For the audited dense GQA configurations, `D_k=D_v=D` and bf16 `s=2`, which
reduces to the earlier `2 * L * D * 2 * ...` expression.

- when `T_a` divides `H_kv`, each rank holds less but aggregate unique KV stays
  approximately constant;
- when `T_a > H_kv`, SGLang keeps at least one KV head/GPU, so aggregate state
  grows;
- one rank failure stalls the collective; KV alone does not rebuild the
  process, weights, communicator, or scheduler;
- protection is defined over unique logical attention-head shards under
  effective attention TP `T_a`; replicated physical copies are alternatives;
- for each required logical shard, first prove that a copy or coded quorum
  survives the declared fault, then take the minimum compatible frontier across
  all required logical shards and PP stages.

### PP

- each stage owns its layer subset; aggregate KV stays the same;
- the request frontier is the minimum compatible frontier across required
  stages;
- a compact stage-boundary activation can allow only the failed stage to
  regenerate its KV;
- that anchor must cover the same token extent and reside outside the failed
  stage's declared fault domain;
- without a boundary anchor, upstream replay may dominate recovery.

For Qwen3-8B at PP=4, a raw bf16 hidden-size-4096 boundary activation is about
8 KiB/token, while one 9-layer stage's KV is about 36 KiB/token before TP
sharding: a useful hypothesis, not yet an end-to-end result.

### DP

- only standard full-replica DP provides an independent warm target; it still
  does not share request state automatically;
- SGLang DP-attention repartitions the global attention-TP group and is not a
  full model/KV replica; its recovery is topology-aware attention/EP
  reconstruction, not ordinary DP reroute;
- peer-HBM backup consumes the same capacity needed to absorb failed-node load;
- prefer a shared Host/remote object and content-addressed prefix deduplication;
- the manifest records owner, eligible reroute group, and warm-target capacity;
- the audited SGLang branch already excludes failed DP ranks and reconstructs a
  request from prompt/output tokens, sampling state, stream state, LoRA/model
  context, and a backup generation. This is direct code evidence that KV alone
  is insufficient.

### EP

- attention KV normally follows attention TP/PP placement, not each expert;
- expert failure needs expert-placement/weight availability, expert
  shadow/reroute, and routing-epoch repair;
- if an EP rank also owns an attention shard, recovery must restore both KV and
  expert executability;
- extra KV copies do not repair an unavailable expert;
- the recovery manifest records the routing epoch to reject incompatible
  replay.

### Hybrid

The minimum request recovery object is:

`KV component manifests + KV frontier + emitted-output frontier +
request/sampling/stream/scheduler state + topology/routing epoch +
replacement-target plan`.

“KV backup succeeded” is insufficient if any of those required components is
missing.

## 10. When peer-GPU backup is valuable

Peer HBM can be valuable only when all are true:

- the failure scope is a soft GPU/device-state fault, not node/rack; a process
  fault additionally requires peer allocation and commit metadata to be owned
  outside the failed process;
- the peer is sufficiently independent for that declared scope;
- spare HBM is genuinely available after failover capacity reservation;
- the copy path is fast and does not contend with TP/EP collectives;
- the RTO is too small for Host restore;
- measured end-to-end recovery benefit exceeds the no-failure interference.

The repository's historical A100/NVLink action microbenchmark shows that
1,152 MiB can have a much faster peer path than D2H/H2D on that topology. It
does not establish a default policy. The current gpu10 GPUs are RTX 4090
devices connected through `SYS`, with no NVLink.

Thus the final figure grays peer HBM as an exception rather than placing it on
the main path.

## 11. Live SGLang implementation boundary

The live checkout currently stages decode KV as:

`GPU KV -> pinned Host pool -> storage write attempt -> operation-completion
queue -> release Host staging`.

It already provides useful ingredients:

- one pinned Host KV pool;
- D2H and storage background queues;
- TP-wide minimum progress;
- request-generation fields;
- remote Host storage.

That Host pool is currently instantiated inside the Scheduler process. It is
therefore evidence for a GPU/device-state-loss staging path, not for surviving
a real Scheduler-process crash.

But its current `ack_backup_queue` is **not a durable commit**:

- a batched page write breaks after `page_set_func` failure;
- the operation is still enqueued unconditionally;
- the decode manager releases all Host indices without checking completed
  pages, checksum, quorum, or fault domain.

The source audit and exact lines are in
[live_sglang_storage_audit.md](../ideas/evidence/live_sglang_storage_audit.md).

Required implementation delta:

1. process-external Host-pool ownership/shared memory for process-fault scope;
2. typed Host leases and physical-payload deduplication;
3. expected/success page sets, checksum, topology epoch, and error status;
4. full-replica or post-failure-decodable coded placement by declared scope;
5. scope-durable commit manifest/quorum, self-describing object headers, and a
   rebuildable directory;
6. logical-shard-aware TP/PP frontier with replicated heads as alternatives;
7. per-path token buckets and shared bottleneck owners;
8. foreground reservations and active interference guards;
9. replacement-target and restore-path certificate;
10. cause-coded telemetry and budget epochs;
11. grandfather/migrate/reclaim lifecycle on budget shrink;
12. persistent structured event/commit artifacts.

## 12. Patent-friendly mechanism chain

The strongest concrete chain is:

`recovery-budget/failure-scope ticket`
`-> dirty-frontier expiry`
`-> exact Host cache/offload lookup`
`-> zero-copy protection-lease promotion or remote materialization`
`-> atomic alpha capacity + beta path + fault-domain acquisition`
`-> payload-quorum + scope-durable-manifest group commit`
`-> dual restore/recompute certificate`
`-> cause-coded window feedback`
`-> K+1 canary budget publication and safe inventory transition`.

The potentially protectable part is the cooperating component protocol and
state transitions. Erasure coding, Host offload, DRF, EDF, token buckets,
checkpointing, and prefix caching individually remain prior art.

## 13. Evaluation contract

### Baselines

- recompute only;
- fixed local Host backup;
- fixed remote Host replica;
- GhostServe-style static parity;
- separate cache and backup pools;
- unified pool without zero-copy promotion;
- capacity-only controller;
- aggregate-bandwidth controller;
- full per-path `alpha + beta_p` broker.

### Required platforms

- PCIe-only node;
- NVLink/NVSwitch node;
- multi-node RDMA;
- controlled Host/RNIC/storage contention;
- TP/PP/DP/EP and hybrids;
- continuous batching with CUDA graphs;
- multi-agent DAGs with shared prefixes and tool waits.

### Metrics

- no-failure TTFT, TPOT, P99, throughput, OOM;
- Host physical bytes and fragmentation;
- D2H/RNIC/storage bytes per useful committed byte;
- protection-deadline expiry misses by cause;
- group-committed frontier coverage;
- GPU/device/process/node/rack recovery success, with process-external ownership
  reported separately;
- failure-to-first-token and workflow completion time;
- restored versus recomputed tokens/layers;
- replacement-target readiness;
- controller convergence, oscillation, rollback, and inventory migration.

### Critical fault injections

- page-write partial failure before operation completion;
- payload quorum without manifest publication and manifest without payload
  quorum;
- directory/control-plane loss followed by reconstruction from durable object
  markers;
- Scheduler-process crash with process-local versus external Host-pool owner;
- stale generation/topology epoch;
- one TP rank loss;
- one PP stage loss;
- DP owner loss with and without a warm target;
- DP-attention rank loss versus standard full-replica DP loss;
- EP expert loss;
- node loss against a lone parity fragment versus a post-failure-decodable
  coding placement;
- source node and remote protection node correlated loss;
- D2H/RNIC fail-slow;
- budget decrease with active committed/in-flight leases.

## 14. Evidence boundary

The diagram is a proposed system overview. The paper corpora, live source
audit, and profiler run justify the modeling choices; they do not prove the
full system's performance or correctness. Claims become implemented only after
the twelve implementation deltas above are completed and tested against the
evaluation contract.
