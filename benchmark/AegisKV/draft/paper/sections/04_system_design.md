# 04. 系统模型与设计

## 1. 设计原则

AegisKV 不是另写一个 KV cache manager，而是 SGLang HiCache/radix cache 上的一层 failure-aware policy。SGLang 继续拥有：

- active KV ownership；
- radix prefix match/insert/split；
- request lock/ref 管理；
- ordinary eviction；
- Host D2H/H2D copy；
- prefetch、backup、ACK 队列；
- scheduler admission 和 retry request 生命周期。

AegisKV 只新增：

- workflow/agent metadata；
- radix node 的 failure value metadata；
- fixed HostBackup frontier directory；
- peer-rank selection；
- ACK 后的 safe frontier；
- failure-time restore-vs-recompute 选择。

这条边界很重要：AegisKV 的正确性必须继承 SGLang 已有的 lock、eviction、ACK 语义，而不是绕开这些语义。

## 2. 系统模型与优化目标

AegisKV 的建模对象是 SGLang radix tree 上的 KV segment / prefix
frontier，而不是整条 request。每个候选 segment 由 workflow id、agent
refs、radix path、token span、size 和 owner DP rank 标识；其价值来自
critical-path delay、redo cost、failure exposure、live reuse probability 和
restore-vs-recompute lower envelope。

当前 paper scope 固定为 GPU / DP-rank failure：source GPU KV 丢失，但
source Host、HostBackup directory 和其他 DP ranks 存活。可用恢复动作是：

- `GPUReplica`: 从健康 peer rank 的 GPUReplica 恢复或命中；
- `HostBackup`: 从 source-rank ACKed Host frontier 恢复；
- `Recompute`: 从最近 safe frontier 或 root 重算 suffix。

HostBackup 是 fixed SGLang `write_through` 产生的 ACKed state，不是
AegisKV 新增的 D2H placement decision。AegisKV 的主动决策集中在 peer
GPUReplica placement、ACKed safe frontier publication、failure-time
restore/recompute action，以及 shared-prefix marginal SafeCover accounting。

目标函数是最小化 failure-aware workflow completion time，同时约束
normal-path overhead、backup/restore queue pressure、peer HBM/P2P interference
和 policy actuation cost。在线实现采用 event-driven value-density greedy
policy；learning-based controller 只作为未来替代，不作为当前实现声明。

## 3. 基于源码的 SGLang 路径

当前 SGLang `multi-level-backup` 分支中，AegisKV 需要对齐以下路径：

| Path | Existing behavior | AegisKV implication |
|---|---|---|
| `cache_controller.HiCacheController.write` | GPU KV D2H 写到 Host pool，完成后进入 `ack_write_queue` | Host backup 必须等 ACK 后才是 safe frontier |
| `cache_controller.HiCacheController.load` | Host KV H2D 恢复到 GPU pool，完成后进入 `ack_load_queue` | failure restore 与 normal prefetch 共享 H2D 资源 |
| `HiRadixCache.write_backup` | node D2H 后设置 `host_value`，write-through 时临时 lock node/path | AegisKV 固定复用该 `write_through` Host Backup 路径 |
| `HiRadixCache.writing_check` | 检查 D2H ACK，释放 write-through lock，并可继续写 storage | safe frontier 更新应挂在这里 |
| `HiRadixCache.evict` | ordinary radix eviction respects `lock_ref` and Host state | Host page/GPUReplica invalidation must update frontier directory |
| `HiRadixCache.match_prefix` | 区分 device-resident prefix 与 host-only hit | recovery frontier 要明确是 GPU hit、Host restore 还是 recompute |
| `check_hicache_events` | scheduler 周期性 drain write/load/storage control queues | AegisKV 的 ACK 处理应在同一周期中完成 |

本文固定使用 `write_through`：radix node 插入或 KV segment materialized 后立即触发 D2H。baseline 也执行这条路径，但 baseline 的 retry rank 不知道 failed rank 的 Host KV frontier，因此 failure 后仍需要 recompute。AegisKV 的 HostBackup 贡献是增加 recovery directory 和 restore protocol，让 retry rank 可以使用 failed rank 上仍存活的 Host KV cache。

SGLSim 当前 `gpu10:~/sglsim` 主要提供 HiCache/radix/scheduler 的模拟实现：`hicache.py` 模拟 D2H/H2D duration 和 overlap，`hiradix_cache.py` 模拟 write-through Host metadata、eviction 和 `check_hicache_events`，`scheduler.py` 把 H2D、forward、D2H backup pending 放进 pipeline accounting。当前可见源码中没有完整 AegisKV policy 入口，因此 SGLSim 需要新增 AegisKV policy hook，而不是假设已有完整策略。

## 4. 架构

AegisKV 由六个模块组成：

```text
Workflow Metadata Propagation
  -> Radix Segment Metadata
  -> HostBackup Recovery Directory
  -> Backup Placement Planner
  -> Safe Frontier Manager
  -> Failure-Time Restore Scheduler
```

系统路径：

```text
multi-agent request
  -> tokenizer manager attaches aegis metadata
  -> scheduler carries metadata in Req
  -> radix cache updates touched node metadata
  -> fixed write_through D2H publishes HostBackup frontier after ACK
  -> planner optionally creates GPUReplica on a peer rank
  -> ACK marks HostBackup/GPUReplica safe frontier
  -> failure retry chooses GPUReplica, Host restore, or recompute
```

## 5. Metadata 传播

每个 request 携带 optional `aegis` metadata：

```json
{
  "workflow_id": "wf42",
  "agent_id": "S2",
  "agent_type": "subagent",
  "parent_agent_id": "O0",
  "parallelism_degree": 3,
  "lifecycle_state": "RunningDecode",
  "round_id": 1,
  "segment_type": "subagent_tail",
  "critical_agent": false,
  "serving_dp_rank": 0,
  "retry_dp_rank": 1,
  "retry_generation": 0
}
```

传播路径：

```text
request input / sampling custom params
  -> tokenizer manager
  -> Req
  -> scheduler admission
  -> radix match/insert/touch
  -> Aegis node metadata
```

没有 `aegis` metadata 的请求必须完全走 baseline SGLang 行为。

## 6. Radix segment metadata

每个 radix node 增加轻量 Aegis metadata：

```text
AegisNodeMeta:
  workflow_refs: live workflow/agent refs
  host_state: none/planned/pending/acked/evicted
  peer_states: peer_dp_rank -> planned/pending/acked/evicted/demoted
  safe_cover: token-span coverage already recoverable
  checkpoint_epoch
  last_decision
```

节点 split/merge 时，metadata 必须按 token span 一起 split/merge：

- shared prefix 的 `workflow_refs` 继承给 prefix node；
- child tail 只保留覆盖该 tail 的 agent refs；
- `host_state` 和 `peer_states` 必须按原 `host_value` / peer pages 的实际 span 切分；
- `safe_cover` 不能超过已 ACK 的连续 prefix frontier。

`workflow_refs` 不等于 SGLang active ref。request 完成、tool blocked、fan-in 等 lifecycle 变化只改变 workflow value，不直接改变 SGLang 的 lock/ref 规则。

## 7. 固定 HostBackup Recovery Directory

HostBackup 的 copy path 固定为 `write_through`：

```text
radix insert / KV segment materialized
  -> HiRadixCache.write_backup
  -> HiCacheController.write
  -> GPU-to-Host D2H
  -> ack_write_queue
  -> HiRadixCache.writing_check
  -> Aegis HostBackup directory publish
```

HostBackup directory 保存 failed rank Host KV 可被 retry rank 恢复所需的最小信息：

```text
HostBackupEntry:
  source_dp_rank
  source_worker_id
  workflow_id / agent_id
  radix_path
  token_span
  host_page_ids or host handle
  checkpoint_epoch
  model / tokenizer / kv_layout / page_size
  acked
  invalidated
```

baseline 与 AegisKV 的区别：

| System | D2H write_through | Cross-rank Host frontier visibility | Failure result |
|---|---|---|---|
| Baseline | yes | no | retry rank cannot use failed rank Host KV; recompute |
| AegisKV HostBackup | yes | yes | retry rank can restore ACKed Host frontier |

因此无 failure overhead 对比应把 `write_through` D2H 视为共同成本。AegisKV 额外成本主要来自 HostBackup directory metadata、failure-time lookup/restore，以及可选 GPUReplica peer copy。

## 8. Backup Placement Planner

planner 在以下事件触发：

| Event | Source | Aegis action |
|---|---|---|
| `prefill_insert` | radix cache | 新 prefix/segment 进入候选集 |
| `write_through_ack` | HiCache `writing_check` | 发布 HostBackup safe frontier |
| `request_lock_release` | scheduler/cache | active ref 释放，但 live workflow 可能仍需 protection |
| `lifecycle_change` | workflow runtime/request metadata | 更新 WCTImpact 和 LiveReuseProb |
| `parallelism_change` | workflow runtime | 调整 parallel branch value dilution |
| `fan_in_complete` | workflow runtime | 降级 branch tails，升级 merged summary/main frontier |
| `peer_ack` | peer copy manager | GPUReplica frontier becomes safe |
| `host_eviction` | HiCache Host eviction | Host safe frontier invalidation |
| `gpu_eviction` | radix eviction | GPU hit invalidation, Host frontier may remain |
| `resource_pressure_change` | scheduler/cache controller | 调整 threshold/budget |
| `failure_detected` | scheduler/DP controller | 切换到 restore scheduler |

第一版 decision rule：

```text
for each touched radix segment:
  compute workflow value and recompute cost
  compute current SafeCover
  observe HostBackup availability from write_through ACK directory
  score each peer GPU under load/locality/HBM/P2P pressure
  optionally submit a GPUReplica copy for high-value segments
```

## 9. GPUReplica 路径

`GPUReplica` 的语义是把 source DP rank 的 selected KV segment 复制到另一个 DP rank 的 GPU，使 failure retry 更可能在 peer 上获得 prefix hit 或低延迟 P2P restore。

流程：

```text
source rank computes segment
  -> planner selects peer_dp_rank using load + locality + HBM + retry score
  -> source queues peer copy
  -> peer allocates replica pages under replica budget
  -> peer records replica metadata and optional radix prefix entry
  -> peer ACK updates source/metadata epoch
  -> failure retry routes to peer when beneficial
```

peer 选择不能只用 round-robin。至少要考虑：

- peer waiting/running queue；
- peer HBM free pages；
- peer 是否已有同 workflow shared prefix；
- failure 后 retry 是否会路由到该 peer；
- source-peer P2P bandwidth 和 queue；
- peer 与 source 的 failure-domain correlation；
- 复制后是否会挤占 peer 的 normal serving KV。

## 10. Host 路径

`HostBackup` 的语义是 source DP rank / worker 的本地 Host memory backup。本文只考虑 GPU failure，因此 source GPU KV 丢失，但 source Host KV cache 仍然存在。retry request 可以被调度到另一个健康 DP rank，并从 failed/source rank 的 HostBackup frontier 恢复 KV。

流程：

```text
radix node inserted / KV segment materialized
  -> fixed write_through policy triggers HostBackup
  -> HiCacheController.write allocates source-rank host pages and submits D2H
  -> HiRadixCache keeps pending state and, when needed, write-through lock
  -> writing_check observes ACK
  -> Aegis publishes HostBackupEntry(source_dp_rank, host handle, token span)
  -> source GPU failure occurs
  -> retry request is routed to another DP rank
  -> retry rank looks up failed rank HostBackupEntry
  -> KV is restored from source Host to retry rank GPU
  -> missing suffix is recomputed
```

Host path 的关键限制：

- `host_value` 被设置不等于 copy 已经完成；
- pending D2H 不能作为 failure-safe frontier；
- Host page 被 evict 后必须撤销 safe frontier；
- retry rank 必须能通过 HostBackup directory 发现 failed/source rank 的 Host KV；
- baseline 即使有 `host_value`，也因为没有跨 rank frontier metadata 而不能恢复；
- H2D/cross-rank restore 与 normal prefetch/load 共享带宽和队列。

## 11. Shared Safe Frontier Manager

Safe frontier 以 radix prefix content 为单位，而不是以副本为单位：

```text
SafeFrontier(workflow, agent, retry_path):
  deepest contiguous prefix whose content is recoverable
  from at least one ACKed placement
```

它保存：

- token ids / token offset；
- radix path and node epoch；
- model、tokenizer、KV layout、page size；
- Host location and ACK epoch；
- peer rank and ACK epoch；
- retry generation；
- invalidation reason if Host/GPU replica is evicted。

当 Host 和 GPUReplica 覆盖相同 prefix 时，frontier manager 只登记一个 content frontier，并记录多个 restore choices：

```text
frontier F:
  content = tokens[0:k]
  choices = {GPUReplica(peer=2), Host(local), recompute_suffix}
```

failure 时再根据实时 queue 和 restore/recompute cost 选择 choice。

## 12. Failure-time restore scheduler

发生 source rank/GPU failure 后：

```text
1. DP controller/scheduler identifies affected live requests.
2. Tokenizer manager or scheduler constructs retry requests with retry_generation++.
3. Aegis scans retry radix path for deepest valid SafeFrontier.
4. For each frontier choice:
     estimate peer GPU hit/restore,
     Host H2D restore,
     recompute suffix.
5. Choose lowest WCT-aware recovery cost.
6. Scheduler prioritizes recovered critical agents and resumes normal serving.
```

Recovery cost:

```text
RecoveryCost =
  RestoreOrRecomputeTime
  + RetryQueueDelay
  + DownstreamStallPenalty
  + NormalPathInterference
```

即使有 backup，如果 Host H2D queue 或 peer queue 已经拥塞，AegisKV 仍可能选择从更浅 frontier recompute。

## 13. Lock 与 eviction 正确性

AegisKV 必须遵守以下 invariants：

| Invariant | Reason |
|---|---|
| 不驱逐 `lock_ref > 0` 的 active radix path | active request 仍在使用 KV |
| pending Host write 不算 safe | D2H 可能尚未完成 |
| ACK 后才能推进 checkpoint epoch | failure-time restore 只能使用一致 frontier |
| node split/merge 同步更新 Aegis metadata | 防止 frontier 指向错误 token span |
| GPU eviction 不一定删除 Host frontier | Host-only prefix 仍可 H2D restore |
| Host eviction 必须删除 Host frontier | 否则 failure lookup 会返回悬空副本 |
| peer replica eviction/demotion 必须更新 `peer_states` | 防止 retry 路由到无效 peer |
| ordinary SGLang prefix match 仍只返回 device-resident prefix | Host-only prefix 需要显式 load_back/restore |

这部分应在实现里用 unit tests 和 runtime assertions 覆盖。

## 14. 改动边界与不变语义

| Area | Change? | Detail |
|---|---|---|
| tokenizer output/token ids | no | Aegis metadata 不改变 prompt/tokenization |
| model forward kernels | no | 不改 attention/KV layout hot path |
| SGLang active KV ownership | no | active pages 仍由 existing pools 管理 |
| ordinary radix match/insert | minimal | 只附带 metadata，不改变 prefix semantics |
| existing HiCache D2H/H2D primitives | no | 复用 write/load/prefetch queues |
| request metadata | yes | 增加 optional workflow/agent fields |
| radix node metadata | yes | 增加 AegisNodeMeta 和 split/merge propagation |
| HostBackup recovery directory | yes | 把 fixed write_through ACKed Host KV 暴露给 failure retry |
| peer GPU replica manager | yes | 新增 peer page allocation、ACK、eviction/demotion |
| safe frontier registry | yes | 记录 ACKed recoverable content |
| failure retry policy | yes | 选择 GPUReplica、Host restore 或 recompute |
| SGLSim | yes | 新增 AegisKV policy hook、peer GPU model、failure matrix |

## 15. SGLSim 对齐

SGLSim 应先对齐本文固定的 SGLang 行为，再加入 AegisKV：

1. 保留当前 HiCache D2H/H2D duration、pending backup overlap、fixed write_through 和 radix eviction 模型；
2. 增加 `AegisRequestMeta` 和 `AegisNodeMeta`；
3. 在 `hiradix_cache.py` 的 match/insert/write_backup/evict/check_hicache_events 上补 Aegis hook；
4. 在 scheduler pipeline accounting 中分开记录 fixed write_through D2H、failure-time Host restore、peer-copy；
5. 在 DP controller/scheduler 中加入 rank failure、retry generation、peer route；
6. 在 benchmark 中补在线 workflow source 脚本，不依赖只存在于 `__pycache__` 的旧实验入口。

SGLSim 的主作用是做大规模矩阵和 sensitivity；SGLang 的主作用是真实系统可行性、开销和校准。

## 16. 为什么本稿不引入 RemoteStore

RemoteStore 会引入：

- cross-host network bandwidth；
- remote KV service consistency；
- durable recovery semantics；
- remote routing and lease GC；
- host failure recovery。

这些是后续扩展。当前论文先证明 `GPUReplica` 与 `Host` 在真实 SGLang/HiCache/radix 约束下可以减少 failure 后 workflow completion time，并控制 no-failure overhead。
