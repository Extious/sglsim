# 06. 实验计划

## 1. 实验问题

The evaluation should answer ten questions:

1. Does equal-size KV loss produce different workflow completion penalties depending on agent/lifecycle/segment position?
2. Does AegisKV reduce workflow completion time under GPU / DP-rank failure compared with no backup and static backup policies?
3. When is `GPUReplica` better than `Host`, and when is `Host` better than `GPUReplica`?
4. Does restore-vs-recompute-aware recovery outperform restore-all and recompute-all?
5. Does AegisKV control normal-path overhead under no failure?
6. Which signals matter most: lifecycle, segment granularity, restore/recompute cost, resource pressure, hysteresis?
7. Does SGLSim preserve the policy ranking observed in SGLang?
8. Under the same fixed `write_through` D2H path, how much does cross-rank HostBackup visibility improve recovery over baseline?
9. How much does peer-rank selection need load, KV locality, and retry-route awareness?
10. Does shared-prefix marginal accounting reduce duplicated backup bytes without reducing recovery coverage?

## 2. Workload

Minimum workflow families:

| Workload | Shape | Why needed |
|---|---|---|
| two-round fan-out/fan-in coding | `O0 -> S1/S2/S3 -> O0 -> S4/S5/S6 -> O0` | core multi-agent structure |
| tool pause/resume | main or subagent blocks and resumes | tests Host prefix value |
| final synthesis-heavy | long final answer / verification | tests hot critical tail |
| high parallelism swarm | many subagents, high parallelism | tests value dilution by parallelism |
| RAG agent | shared prefix + private retrieved context | tests prefix vs branch tail |

Each workload should record:

```text
workflow_id
agent_id
agent_type
lifecycle_state
parallelism_degree
segment_type
input_tokens
output_tokens
prefix_overlap
tool_wait_time
failure_time
sink completion time
```

## 3. Policy

### 主要 baseline

| Policy | Description |
|---|---|
| `none` | no failure-protected backup; recompute from ordinary cache/root |
| `baseline_write_through_invisible` | fixed write-through D2H exists, but retry rank cannot use failed-rank Host KV |
| `host_visible_all` | fixed write-through D2H exists and all ACKed Host frontiers are visible to retry |
| `uniform_peer_gpu` | every eligible segment backed to peer GPU until budget exhausted |
| `periodic_checkpoint` | checkpoint by token/time interval, no workflow value |
| `request_level_backup` | protect entire request prefix/tail as one unit |
| `agent_level_backup` | protect all segments of selected critical agents |
| `static_critical_path` | use role/topology priority but no runtime resource pressure |
| `restore_all` | always restore if backup exists |
| `recompute_all` | always recompute after failure |
| `AegisKV` | fixed HostBackup visibility + selective GPUReplica + restore/recompute + resource pressure |

### 基于源码路径的 policy 变体

| Policy | Description |
|---|---|
| `host_invisible_baseline` | same fixed write-through D2H, no cross-rank HostBackup directory |
| `host_visible_no_gpu_replica` | same fixed write-through D2H, retry can use failed-rank Host KV, no GPUReplica |
| `gpu_replica_only` | peer GPU backup available, failed-rank Host KV not used |
| `host_visible_plus_gpu_replica` | HostBackup directory plus selective GPUReplica |
| `peer_random` | GPUReplica peer chosen uniformly |
| `peer_least_loaded` | choose least-loaded peer rank |
| `peer_kv_locality` | choose peer with deepest existing prefix |
| `peer_load_plus_locality` | choose peer by load + locality + retry route + HBM score |
| `no_shared_prefix_dedup` | count Host and GPUReplica shared prefix value independently |
| `shared_prefix_marginal` | count marginal content frontier value once |

### 可选 upper bound

| Policy | Purpose |
|---|---|
| `oracle_segment_utility` | upper bound for segment selection |
| `oracle_restore_recompute` | upper bound for failure-time action |
| `offline_optimal_budget_ratio` | One Pool-style gap to dynamic budget controller |

## 4. 指标

### 用户可见指标

- workflow completion time；
- final answer latency；
- critical-agent recovery latency；
- retry latency；
- TTFT after resume / tool return；
- downstream stall time；
- p95 / p99 request latency。

### 系统指标

- normal-path throughput；
- no-failure p95/p99 overhead；
- P2P copy bandwidth；
- D2H backup bandwidth；
- H2D restore bandwidth；
- peer HBM usage；
- Host memory usage；
- backup write amplification；
- duplicated protected bytes；
- shared-prefix deduplicated bytes；
- restore queue pressure；
- fixed write-through D2H time；
- H2D load/prefetch queue time；
- write-through ACK delay；
- cross-rank Host restore time；
- peer-rank queue load；
- peer-prefix locality hit；
- recomputed tokens；
- recovered tokens。

### Policy 指标

- recovery coverage；
- peer_gpu / host_h2d / recompute path ratio；
- utility per GB；
- expired KV retention ratio；
- unsafe frontier rejection count；
- placement oscillation；
- gap to offline optimal budget ratio。

## 5. SGLang 实验

SGLang should provide real runtime evidence:

1. Metadata propagation overhead.
2. Host backup ACK and H2D restore latency.
3. GPUReplica copy and peer prefix-hit latency.
4. Failure injection retry path.
5. Normal-path overhead under no failure.
6. Small-scale policy comparison, likely DP=2 first.
7. Fixed write-through baseline vs AegisKV HostBackup visibility.
8. Lock/eviction correctness under pending Host copy, Host eviction, GPU eviction, and retry.

Expected SGLang figures:

- real Host vs GPUReplica microbenchmark；
- real failure timeline；
- real retry latency by recovery path；
- no-failure overhead with backup enabled；
- SGLang vs SGLSim calibration points。
- Host-visible vs Host-invisible retry comparison under identical write-through D2H；
- unsafe frontier rejection/correctness counter summary。

### 5.1 Host visibility 实验

Setup:

```text
same workload, same failure time, same backup budget
fixed write_through D2H for both baseline and AegisKV
vary only whether retry rank can discover failed-rank Host KV
```

Configurations:

| Config | Expected behavior |
|---|---|
| `baseline_write_through_invisible` | D2H exists, but failure retry recomputes because Host frontier is not visible |
| `host_visible_no_gpu_replica` | retry restores long ACKed prefixes from failed rank Host memory |
| `gpu_replica_only` | fast recovery when peer replica hits, lower coverage under peer HBM pressure |
| `AegisKV` | chooses GPUReplica for high-value peer-local segments and Host restore otherwise |

Metrics:

- ACKed frontier tokens before failure；
- fixed D2H pending time；
- no-failure p99 overhead relative to fixed-write-through baseline；
- retry latency；
- recovered vs recomputed tokens。

Expected result:

```text
baseline_write_through_invisible pays D2H but cannot recover from failed-rank Host KV
host_visible_no_gpu_replica reduces recomputation for long prefixes
GPUReplica wins for short high-value segments when peer locality/load is favorable
AegisKV combines both and tracks the lower recovery-time envelope
```

### 5.2 Radix 正确性实验

This is required before claiming system correctness.

Cases:

- node split after Host ACK；
- node split while Host copy pending；
- active request lock while backup decision changes；
- GPU eviction of Host-backed node；
- Host eviction of GPU-evicted node；
- peer replica demotion under peer pressure；
- failure retry while one frontier is pending and another is ACKed。

Expected result:

- no active locked KV is evicted；
- no pending frontier is used for recovery；
- split/merge preserves token span；
- invalidated Host/peer frontier is never selected；
- baseline SGLang behavior unchanged for requests without Aegis metadata。

Current unit-level artifact:

```text
test/unit/srt/test_aegis_restore_lookup.py
```

It covers missing/pending HostBackup rejection, pending restore action
rejection, restored-prefix lock protection, divergent radix insert after
restore, Host/peer frontier invalidation, peer-rank demotion, and baseline
requests without Aegis metadata. These tests validate SGLSim/SRT metadata
invariants; final claims still require real SGLang failure-retry and physical
KV-copy validation.

## 6. SGLSim 实验

SGLSim should provide scale and matrix evidence:

1. DP size sweep: 2, 4, 8.
2. Workflow count sweep: 1, 4, 8, 16.
3. Failure delay sweep: before prefill complete, during decode, near fan-in, after summary.
4. Context length sweep: 4K, 8K, 16K, 30K.
5. Peer HBM budget sweep.
6. Host memory / H2D bandwidth sweep.
7. Failure rank sensitivity.
8. Workload skew sensitivity.
9. Peer-rank selection sweep.
10. Shared-prefix overlap sweep.
11. Host-visible vs Host-invisible sweep under fixed write-through.

Expected SGLSim figures:

- overall WCT vs policy；
- recovery path breakdown；
- recompute avoided；
- transfer cost；
- rank sensitivity；
- budget sensitivity；
- scale-up throughput / WCT。
- peer selection load/locality breakdown；
- shared-prefix deduplication effectiveness；
- fixed D2H/H2D restore pressure over time。

SGLSim prerequisite:

```text
current visible source has HiCache/radix/pipeline simulation and a restored
agentic_workflows wrapper over synthesis_requests. Final figures still require
online AegisKV policy hooks plus physical Host/peer KV movement.
```

## 7. 动机实验

Setup:

```text
hold KV bytes lost constant
vary only agent/segment/lifecycle position
measure delta WCT
```

Required positions:

- terminated exploration branch；
- non-final running subagent；
- last subagent before fan-in；
- main agent waiting to resume；
- synthesis hot tail；
- fan-out shared prefix。

Expected result:

```text
delta WCT:
  terminated branch << non-final subagent < last subagent/main/synthesis
```

This figure should appear early in the paper.

## 8. Restore-vs-recompute 微基准

Measure:

```text
RecomputeCost(tokens)
HostRestoreCost(tokens)
PeerRestoreCost(tokens)
```

Vary:

- token length；
- KV block size；
- P2P bandwidth；
- H2D bandwidth；
- queue pressure；
- GPU compute speed；
- request rate / concurrency。

Expected result:

- GPUReplica wins for small high-value hot tails when peer HBM and P2P are available；
- Host wins for longer prefixes with high recompute cost；
- recompute wins for short segments or when copy paths are congested；
- AegisKV should approximate the lower envelope of the three choices。

## 9. Budget ratio 实验

Borrowing One Pool's ratio-sweep style:

```text
alpha_gpu = fraction of peer HBM used for GPUReplica backup
HostBackup is fixed write_through and not swept as a placement budget
```

Sweep:

```text
alpha_gpu in {0, 0.1, ..., 1.0}
```

Measure:

- WCT；
- normal-path overhead；
- recovery coverage；
- peer HBM pressure；
- Host restore pressure。

Expected result:

- optimal ratio shifts with workload and failure timing；
- Host-only and GPUReplica-only are not globally best；
- dynamic AegisKV is close to per-regime offline optimal。

## 10. Peer-rank selection 实验

Question:

```text
Does GPUReplica placement need both load and KV locality?
```

Compare:

- random peer；
- least-loaded peer；
- most-local peer；
- retry-route-aware peer；
- AegisKV load + locality + HBM + retry score。

Vary:

- DP size: 2, 4, 8；
- common-prefix overlap；
- peer HBM budget；
- source-peer P2P bandwidth；
- concurrent workflow skew；
- failure rank。

Metrics:

- peer replica hit rate；
- peer copy queue delay；
- retry queue delay；
- peer normal-path slowdown；
- recovered tokens from GPUReplica；
- WCT under failure。

Expected result:

```text
least-loaded ignores useful prefix locality
most-local overloads hot peer ranks
random is unstable
AegisKV load+locality gives best WCT and lower p99 overhead
```

## 11. Shared-prefix 去重实验

Question:

```text
Does marginal content accounting reduce duplicate backup without hurting recovery?
```

Setup:

- multiple agents share a long root prefix；
- branches have short private tails；
- vary overlap ratio from 0% to 90%；
- compare with/without shared-prefix marginal accounting。

Metrics:

- duplicated protected bytes；
- backup write amplification；
- ACKed safe frontier tokens；
- recovered tokens；
- WCT under failure；
- no-failure D2H/P2P overhead。

Expected result:

```text
no_shared_prefix_dedup overestimates value and copies same prefix twice
shared_prefix_marginal keeps recovery coverage but reduces bytes and overhead
benefit grows with prefix overlap
```

## 12. Ablation 实验

| Ablation | Expected effect |
|---|---|
| no lifecycle signal | over-protects completed / summarized segments |
| request-level backup | high write amplification |
| agent-level backup | protects low-value segments inside critical agents |
| no restore-vs-recompute | restores segments that should be recomputed |
| no resource pressure | overloads peer HBM or Host bandwidth |
| no hysteresis | placement oscillation and copy spikes |
| no critical-agent restore scheduling | restored requests still wait behind low-value work |
| no peer locality | peer copy lands on rank that cannot reuse prefix |
| no peer load signal | copy improves recovery but hurts peer p99 |
| no shared-prefix marginal accounting | duplicate Host/GPUReplica prefix copies |
| no HostBackup visibility | pays fixed D2H but still recomputes after source GPU failure |

## 13. 实验后可接受的结论

Strong claim only if supported:

```text
AegisKV reduces workflow completion time under failures while keeping no-failure p99 overhead bounded.
```

More cautious claim if results are mixed:

```text
AegisKV improves recovery for skewed multi-agent workflows and identifies when GPUReplica, Host, or recompute is preferable.
```
