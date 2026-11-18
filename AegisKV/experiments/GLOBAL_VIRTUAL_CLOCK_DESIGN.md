# AegisKV 全局虚拟时钟模拟器设计

更新时间：2026-08-29

状态：**目标架构与 P0--P12 v3 三层配置已冻结。GlobalDES、真实 Offline Scheduler、逐物理 GPU/节点账本、普通 HiCache 与 AegisKV protection/recovery 物理 transfer、Router committed-position silent replay、canonical recovery group/准入门和 hard fail-stop 均已接通。HBM 与 DRAM 始终是 source-owned independent tiers，不存在 target-owned HBM→DRAM destage。当前权威 contract lineage 是 P0 v5，12/12 PASS，但 summary 明示 `contract_execution_only=true`、`full_scheduler_integration=false`、`performance_result=false`。三层配置可展开为 `12 + 2,134 = 2,146` cells；正式 P1--P12 已执行数仍为 0/2,134，完整 2,146-cell suite 未运行，`paper_ready=false`。PP2×TP4 只有 metadata contract 覆盖；真实 Scheduler 正式模型是 PP1（TP1/TP4）。Router 续跑使用合成 token continuation，不是真实 logits/RNG continuation；时延和带宽也仍是 modeled，不是真机数据。P11 workflow-simultaneous coverage 门仍未过 0.95。历史 613-cell aggregate metadata-DES 不满足本设计。**

设计语义权威来源：
[`Mooncake/aegiskv/design/current_design.md`](/home/comp/zhanzhao/Mooncake/aegiskv/design/current_design.md)

## 1. 目标与证据边界

目标不是再建立一套按请求公式累加的故障模拟，而是在同一条离散事件时间线上执行
SGLang-aligned continuous batching，并显式模拟：

- 每个 DP rank 的 request queue、prefill/decode batch 和完成事件；
- 每个物理 `(node, gpu, dp, pp, tp)` 的 HBM 组成和变化；
- 每个节点的共享 DRAM pool 组成和变化；
- 每个 KV extent 的 source、token interval、PP/TP shard、residency 与 commit state；
- Source Worker 到 Peer HBM、Source Worker 到 Store DRAM，以及故障恢复 load 的排队、开始和完成；
- GPU copy engine、PCIe/NVLink/NVSwitch、NIC/RDMA link 和 DRAM lane 的资源竞争；
- GPU/process/node failure、fencing、PENDING abort、recovery 和 repair；
- 每次状态变化前后的可审计快照。

这是纯元数据和预测时延模拟。即使所有 invariant 和实验都通过，输出仍是
`SGLSim offline virtual-time simulation`，不是实际 tensor copy 或真实 GPU 性能数据。

## 2. 当前实现、验证与尚未完成的证据

仓库中仍保留历史 aggregate 引擎，但目标路径已经不是“两条尚未合并的原型”。当前状态应按实现层和
证据层分别描述：

| 层次 | 当前可复核状态 | 仍不能据此声称什么 |
| --- | --- | --- |
| GlobalDES kernel | 单一 `(timestamp_ns, priority, sequence)` 堆、事件取消、资源日历、extent、transfer、tier-local shard-group commit、GPU/node failure/repair 已实现 | 不能单独证明真实 Scheduler 或论文 workload 已运行 |
| 真实 Scheduler 接入 | `OfflineSchedulerCluster` 的真实 SGLSim Scheduler worker 将 prepare/complete、请求到达和资源 snapshot 接到同一 GlobalDES；online wall-clock main 未改成逻辑时钟 | 单元/集成测试不是完整 P1--P12 lineage，也不是 online/physical validation |
| 物理资源接入 | `OfflineResourceBridge` 将 `(dp,pp,tp)` 映射到具体 GPU，逐卡投影 primary/prefix KV，并维护 node-local DRAM；DP/node fail-stop 后以空 volatile state repair | 元数据账本不是实际 HBM/DRAM 分配或 tensor copy |
| AegisKV protection | `OfflineAegisTransferBridge` 从 source frontier 建立每 TP shard 的 HBM/DRAM job；两个 tier 独立 reserve、完成、notify、commit/abort；禁止 HBM→DRAM destage | modeled bandwidth 下的完成时间不是硬件实测吞吐 |
| 普通 HiCache | `OfflineHiCacheTransferBridge` 将 Scheduler 的 D2H/H2D intent 展开到物理 PP/TP shards，与 AegisKV 共用 GlobalDES copy/PCIe/NIC/fabric/DRAM 日历；全 shard commit、故障 abort、provisional DRAM rollback、compute dependency gate 和 worker-incarnation stale ACK fencing 已接通 | 只是 radix/HiCache 元数据与 modeled transfer，不是真实 host/device memory copy |
| AegisKV recovery | hard source failure 可由已提交 Peer-HBM 或 node DRAM 建立 canonical `RecoveryLoadGroup`和每 shard `TransferRecord`；分片 adoption、`RECOVERY_PENDING`、PP×TP group 终态、明示 fallback 及 Router-output-to-physical-admission gate 已接通 | 当前仍只有 modeled recovery；未执行正式 P1--P12 recovery lineage，不能报告论文恢复时延 |
| Router silent replay | Router 以 committed output position `k` 为权威，按 `prompt + committed outputs`、remaining `N-k`、route version `v+1` 重建执行；commit-before-detokenizer、stale-route fencing、重复故障与 token conservation 已实现 | continuation token 是合成元数据，不是真实 model logits、sampling state 或 RNG 续跑 |
| hard fail-stop | 当前 cluster 对 exact worker 和完整 replica/node failure 可表示且执行 fail-stop；故障取消 in-flight compute、清除域内 volatile HBM，node failure 还清除本地 DRAM，repair 启动空 worker/ledger | 非退化 PP/TP 拓扑中的单 GPU/process target 无法精确对应当前 Scheduler process，因此 fail-close；只有 PP=TP=1 退化情况可等价执行。正式配置只使用 worker/node，也不覆盖 fail-slow/corruption/partition |
| 请求层 | TraceLab/P6 causal workload 已编译；P4 有 modeled fork/join overlay；P11 observed-Weka artifact 为 30 workflows/13,918 rows | P11 simultaneous coverage 0.928571 < 0.95、0 nonzero safe LCB；配置 readiness 不等于 cell 已执行 |
| runtime compiler | explicit frozen modeled bindings 下，P1--P12 deployment/failure/runtime projection 为 2,134/2,134 可编译 | 可编译只属于运行时层，不证明 workload、extent 生命周期、failure/replay 或性能正确 |
| extent release | completion/eviction physical release、tier-local eviction、stale-version isolation、terminal protection-zero 已实现 | 101529 是 pre-fix leak 负证据，104732 是 terminal aggregation 误标 complete 的负证据；都不能被之后的代码或测试追认为有效 lineage |
| suite runner | real-Scheduler runner、lineage、resume/check、transfer-class artifact 与 fail-close 框架已落盘；物理 recovery admission gate 会逐 Router execution ACK 联结 terminal recovery group | 正式 P1--P12 仍为 0/2,134；完整 2,146 cells 未运行，实现或 smoke 都不等于 paper-ready evidence |

代码级回归只描述测试状态，不应以固定 passed 数代替某次新鲜测试输出。当前权威
[`P0 v5`](results/20260829T121349Z_global_des_p0_contract_v5/P0/) 为 12/12 PASS；其
[`summary.json`](results/20260829T121349Z_global_des_p0_contract_v5/P0/summary.json) 明确写明
`contract_execution_only=true`、`full_scheduler_integration=false`、
`p1_p12_execution_ready=false`、`performance_result=false`、`real_hardware_measurement=false`。
这 12 个 contract scenarios 和 5 张诊断图只证明 GlobalDES contract gate；
`controller_bridge_target_rotation_closed_loop=true` 也只表示 current controller + bridge 的确定性
target-before-source contract 闭环，不能扩大为 full Scheduler 或性能证据。该不可变 lineage 不得
就地改写或用后续测试追认。历史 613-cell lineage 继续只作为
**legacy aggregate metadata-DES** 留档。

代表性
[`20260829T100844Z_global_des_scheduler_smoke`](results/20260829T100844Z_global_des_scheduler_smoke/)
只执行 P1 ordinal 0（recompute）和 P12 ordinal 0（analytical ticket），每个只有 1 request。P1
的 request 在 10 s failure 前已结束，所以没有受影响 request 或 recovery job；P12 有 2 个
protection jobs、共 301,989,888 bytes，2/2 modeled analytical-duration checks 与 DES 精确相等。
它们只验证 runner/artifact 路径，manifest 明确为 `paper_ready=false` 且
`full_2146_cell_suite_executed=false`。

[`20260829T101529Z_global_des_scheduler_smoke`](results/20260829T101529Z_global_des_scheduler_smoke/)
执行 P4 `full` 的 8 个 controlled fork/join requests，记录 46 个 protection jobs、18,496,880,640 bytes、
32 张 GPU 和 4 个 node-local DRAM pools。它是在 completion/eviction extent release chain 修复前生成的：
终止请求的 committed protection extent 可能残留，构成 extent-lifetime leak。其
[`plots_v3`](results/20260829T101529Z_global_des_scheduler_smoke_plots_v3/) 四图只能作为 pre-fix 诊断，
不能成为 correctness gate、容量结论或 P4 ablation。

release chain 代码修复后的第一次尝试
[`20260829T104732Z_global_des_scheduler_smoke`](results/20260829T104732Z_global_des_scheduler_smoke/)
记录了 terminal protection balances 归零，但 `requests.jsonl` 为 7 completed + 1 scheduled，suite 却标记
`complete=true`。因此它暴露的是 request-terminal aggregation bug，不是有效 post-fix lineage。

P11 输入已经固化在
[`data/observed/weka_agentx_causal_v1`](data/observed/weka_agentx_causal_v1/)：30 workflows、13,918 rows、
small/medium/long eligible boundaries 各 42 workflows，每 seed×stratum 1 个完整 workflow。43.45% requests
有 modeled ticket，0 个 safe-Slack LCB 非零；workflow-simultaneous coverage 为 0.928571，未达到 0.95。
它是 preliminary modeled input，不是 P11 execution evidence。

[`20260829T102845Z_p11_atomic_small_scheduler_smoke`](results/20260829T102845Z_p11_atomic_small_scheduler_smoke/)
完整执行 small/seed0/recompute 的一个 57-request workflow，57/57 completed，且明确禁止 smoke truncation。
它没有 protection/recovery，10 s failure 未命中 active request；只验证 atomic workflow replay 和真实
Offline Scheduler 输入路径，不是 P11 paired effect 或 predictor evidence。

## 3. “全局虚拟时钟”的正确定义

全局时钟不是每步把所有 DP rank 强制推进到同一个最大时间。那种 `max(shared, rank_local)` 会把
独立 DP 错误耦合，压缩并行差异，并可能把 makespan 从 `max(rank completion)` 变成近似串行和。

本设计采用标准 DES 语义：

1. 全局 kernel 持有唯一 `now_ns` 和最小堆；
2. 事件以 `(timestamp_ns, priority, sequence)` 稳定排序；
3. 每次只把 `now_ns` 推进到下一事件；
4. 每个 Scheduler actor、GPU、copy engine、NIC、link 和 DRAM lane 维护自己的 `available_at_ns`；
5. 一项工作只能在所有依赖与所需资源可用后开始；
6. 没有参与该事件的 rank/resource 不被强制推进；
7. 全局 makespan 为最后一个完成事件时间，等价于各 actor/resource 完成时间的最大值。

```text
request/failure events ─┐
scheduler completions ──┼─> GlobalEventKernel(min-heap, now_ns)
transfer completions ───┤              │
control-window events ──┘              ├─ Scheduler actors (DP/PP)
                                       ├─ GPU HBM ledgers (DP/PP/TP)
                                       ├─ Node DRAM ledgers
                                       ├─ copy engine / NIC / link calendars
                                       └─ append-only event + state journal
```

## 4. 复用 SGLSim，而不是整体替换

执行语义继续以 pinned SGLang 和 `python/sglsim/srt/` 为准：

- `Scheduler.get_next_batch_to_run`、PrefillAdder、continuous batching；
- `prepare_offline_step` / `complete_offline_step` 的无 sleep split；
- radix/HiCache 的 page allocation、lock、eviction 和 request release；
- AIConfigurator/InferTimePredictor 的 batch forward 时延；
- DataParallelController 的 route、failure fence、retry 和 AegisKV 控制面；
- current design 的 independent HBM/DRAM admission、PENDING/COMMITTED/ABORTED；
- 4096-token prefix chunk、256-token sealed decode chunk，以及完整 PP/TP shard ACK。

新的 DES 层只接管时间推进、物理资源日历和观察事件，不另写一套调度器。

当前实现锚点如下：

- [`global_des.py`](../../python/sglsim/srt/simulation/global_des.py)：唯一事件堆、账本、extent、transfer、commit group 与 failure/repair；
- [`offline_coordinator.py`](../../python/sglsim/srt/simulation/offline_coordinator.py)：Scheduler stage completion 与 GlobalDES 共用一条事件序；
- [`offline_resource_bridge.py`](../../python/sglsim/srt/simulation/offline_resource_bridge.py)：Scheduler snapshot 到逐 TP GPU、逐 node DRAM 的投影；
- [`offline_aegis_transfer_bridge.py`](../../python/sglsim/srt/simulation/offline_aegis_transfer_bridge.py)：物理 protection/recovery job 与独立 tier notification；
- [`offline_hicache_transfer_bridge.py`](../../python/sglsim/srt/simulation/offline_hicache_transfer_bridge.py)：普通 HiCache D2H/H2D intent 到物理 shard transfer、依赖和 commit/rollback 的映射；
- [`offline_router_ledger.py`](../../python/sglsim/srt/simulation/offline_router_ledger.py)：Router logical/execution route、committed-output position、fence 和 silent replay 守恒；
- [`offline_failure_domain.py`](../../python/sglsim/srt/simulation/offline_failure_domain.py)：故障 target 到 Scheduler/physical domain 的精确解析与不可表示 target fail-close；
- [`offline_cluster.py`](../../python/sglsim/srt/simulation/offline_cluster.py)：真实 Scheduler worker、hard fail-stop、repair、replay 与上述 bridge 的编排；
- [`global_des_artifacts.py`](../../python/sglsim/srt/simulation/global_des_artifacts.py)：原子化事件、资源、extent、transfer、request/failure 与审计输出。

## 5. 开源模拟器调研与选型

更完整的逐项目能力审计与链接见
[`OPEN_SOURCE_SIMULATOR_RESEARCH.md`](OPEN_SOURCE_SIMULATOR_RESEARCH.md)。

| 项目 | 可借鉴能力 | 与本目标的差距 | 结论 |
| --- | --- | --- | --- |
| [Inference Lab](https://github.com/doublewordai/inference-lab) | Rust 全局 event heap、MemoryGraph、GPU/node/cluster scope stores、flow/path contention；DP-attention 模式可把 rank 展开为 worker | 普通 TP 仍可聚合；vLLM facsimile；没有 GPU/node failure-repair 与 AegisKV commit 语义 | 最接近资源目标的 memory/transfer donor；不整体迁移 |
| [BLIS / inference-sim](https://github.com/inference-sim/inference-sim) | 成熟 Go DES、GPU→CPU→secondary 多级 KV、容量/LRU、异步 transfer station 和 contention | TP 内仍聚合；lifecycle/terminate 不是硬件 failure/repair；不是 SGLang | 借鉴 transfer station、资源 invariant 和多级存储建模 |
| [LLMServingSim 2.0](https://github.com/casys-kaist/LLMServingSim) | continuous batching、TP/PP/DP、异构硬件、ASTRA-Sim 网络/collective backend、离散时间 | Python frontend 以 vLLM 语义为主；没有 AegisKV source-owned independent HBM/DRAM、failure fence 和 SGLang radix/HiCache 生命周期 | 借鉴 ASTRA/内存计时并作为外部 baseline；不整体迁移 |
| [ASTRA-Sim](https://github.com/astra-sim/astra-sim) | 拓扑、collective、网络争用和分布式执行时延 | 不包含 serving scheduler、KV allocator、prefix cache 和 failure policy | 可作为第二阶段可插拔网络 backend |
| [Vidur](https://github.com/microsoft/vidur) | 快速 LLM serving DES、execution-time predictor、TP/PP、可视化/trace | 调度与内存语义不是 SGLang/AegisKV；缺少逐 extent failure/commit 语义 | 借鉴 predictor 校准和 trace 导出，不替换 SGLSim |
| [SGLang Simulator PR #33824](https://github.com/sgl-project/sglang/pull/33824) | SGLang allocator hook、future-arrival heap、batch 驱动逻辑时间 | 截至 2026-08-29 仍是 open PR；单进程/标量逻辑钟；没有跨 DP 全局资源协调；failure manager 不满足 AegisKV | 继续作为 SGLang 语义 donor，不能视为现成依赖 |
| [SimAI](https://github.com/aliyun/SimAI) | DP/TP/PP/EP/P-D collective 与网络模拟 | 每 DP replica 的 serving/KV 状态仍聚合；没有 DRAM KV tier/failure | 可作为 collective/network predictor adapter |
| [IBM Opal](https://github.com/IBM/opal-sim) | SimPy 全局时间、GPU/CPU/NVMe/DFS 平台层 | MOVE/COPY、tier eviction 与 storage→GPU copy 路径仍不完整；没有 failure | 只借平台层结构 |
| [Frontier](https://github.com/NetX-lab/Frontier) | 现代 vLLM DES、P/D 和 P→D transfer event | replica-level KV；没有独立节点 DRAM ledger/failure | 借鉴 event taxonomy |
| [llm-d inference simulator](https://github.com/llm-d/llm-d-inference-sim) | OpenAI-compatible 请求路径、KV block events、DP rank、remote prefill latency | 主要是实时服务模拟，不是全局虚拟时间 DES | 不适合作为核心 |

结论与 [`OPEN_SOURCE_SIMULATOR_RESEARCH.md`](OPEN_SOURCE_SIMULATOR_RESEARCH.md) 一致：**没有一个开源
模拟器同时满足本项目全部语义，采用混合 donor 架构而不引入第二套调度器。** 保留 SGLSim 的
SGLang-aligned scheduler；MemoryGraph/path contention 参考 Inference Lab，transfer station/invariant
参考 BLIS，collective/network backend 参考 ASTRA-Sim/SimAI，predictor/trace 参考 Vidur。外部 donor
只能替换底层 latency/network backend，不能改变 pinned SGLang 控制流或 AegisKV ownership/commit 语义。

## 6. 物理拓扑与状态模型

### 6.1 GPU identity

每张 GPU 使用不可歧义的物理标识：

```text
GpuId(node_id, local_gpu_id, dp_rank, pp_rank, tp_rank)
```

部署配置必须显式给出 placement，禁止只用 `dp_size × tp_size × pp_size` 后隐式假定摆放。
当 SGLSim 的 TP 仍在一个 Scheduler process 内镜像时，一个 allocator 的 token-page 生命周期可映射到
该 stage 的所有 TP shards；每个 TP GPU 的 KV bytes 按 KV-head/layer shard geometry 独立计算并输出，
不能把 replica 总字节简单复制到每张卡。

### 6.2 每 GPU HBM ledger

每张 GPU 的 HBM 必须满足：

```text
capacity = weights
         + runtime_reserve
         + primary_kv
         + radix_prefix_kv
         + aegis_pending
         + aegis_committed
         + recovery_pending
         + free
```

字段含义：

- `weights_bytes`：该 TP/PP shard 的静态模型权重；
- `runtime_reserve_bytes`：activation、workspace、CUDA graph 等保留量；
- `primary_kv_bytes`：当前 running/waiting request 已物化 KV；
- `radix_prefix_kv_bytes`：可复用且尚未归入 active primary 的 device-resident prefix；
- `aegis_pending_bytes`：已保留目标 slot、传输未完整 ACK；
- `aegis_committed_bytes`：可用于恢复的 Peer-HBM protection copy；
- `recovery_pending`：canonical recovery group 已为目标 shard 保留、但尚未 adoption 的 HBM；
- `free_bytes`：严格由前述分账推导。

`HBMQuotaAIMD` 只约束 `aegis_pending + aegis_committed <= alpha × protection_reserve`；
primary allocator 的 headroom 仍具有更高优先级。

### 6.3 每节点 DRAM ledger

每个节点独立维护：

```text
capacity = runtime_reserve
         + hicache_primary
         + aegis_pending
         + aegis_committed
         + free
```

禁止把 `host_dram_per_node × node_count` 合并成一个全局 pool。Store Master 可以在节点池之间选择，
但 admission、rent、victim frontier、PENDING reservation 和 node failure 都必须落到具体 pool。

### 6.4 KV extent

一个 sealed extent 至少包含：

```text
extent_id
request_id / workflow_id
route_version
model_revision / kv_abi / parallel_layout
ordinal
token_start / token_end
pp_rank / tp_rank / shard_count
bytes
source_gpu
tier = PRIMARY | PEER_HBM | NODE_DRAM
state = PENDING | COMMITTED | ABORTED | EVICTED | LOST
destination_gpu or destination_node
created/start/completed timestamps
```

HBM 与 DRAM 是两个独立 source-owned extent。不存在 HBM→DRAM writeback/destage；DRAM copy 必须从
仍存活的 source production buffer 直接发起。

## 7. Transfer 与资源竞争

每个 transfer job 记录：

```text
job_id, extent_id, copy_kind
source, destination, bytes
ready_ns, start_ns, end_ns
path_resource_ids
status, acked_shards, failure_reason
```

中央模型至少提供以下路径：

| Copy | 必需资源 |
| --- | --- |
| same-node Peer HBM D2D | source copy engine、NVLink/NVSwitch fabric、target copy engine |
| cross-node Peer HBM P2P/RDMA | source copy engine、source PCIe/NIC、network link、target NIC/PCIe、target copy engine |
| source → Store DRAM D2H | source copy engine、PCIe、destination node DRAM lane；跨节点时再占 NIC/link |
| DRAM → recovery GPU H2D | source node DRAM lane、PCIe/NIC/link、target copy engine |
| Peer HBM → recovery execution | source/target copy resource；如果恢复直接在该 target 上执行，则只记录 ownership activation，不制造虚构 copy |

首版使用 deterministic reservation calendar：

```text
start = max(ready, all(resource.available_at))
duration = setup + bytes / bottleneck_effective_bandwidth
end = start + duration
all(resource.available_at) = end
```

这比无竞争的聚合 lane 保真，同时保持可复现。后续可把路径 duration/争用委托给 ASTRA-Sim，但不改变
上层 event、extent 和 commit contract。

## 8. 事件模型与提交语义

### 8.1 事件类型

- workload：`request_arrival`、`workflow_release`；
- scheduler：`rank_ready`、`batch_prepare`、`batch_start`、`batch_complete`；
- primary KV：`kv_alloc`、`kv_append`、`kv_free`、`prefix_insert`、`prefix_evict`；
- protection：`chunk_seal`、`tier_admit`、`tier_bypass`、`reservation_create`；
- transfer：`transfer_queued`、`transfer_start`、`transfer_complete`、`shard_ack`；
- state：`extent_commit`、`extent_abort`、`extent_evict`；
- failure：`failure_inject`、`route_fence`、`volatile_hbm_loss`、`pending_abort`、`repair_complete`；
- recovery：`recovery_select`、`recovery_load_start`、`recovery_load_complete`、`silent_replay_complete`；
- control：`slack_update`、`aimd_window`、`quota_publish`；
- observation：`resource_snapshot`、`invariant_audit`。

### 8.2 PENDING 到 COMMITTED

1. SlackLandlord 在具体 pool 淘汰 victims、预留最终空间并创建 `PENDING`；
2. Source Worker 提交 HBM 和/或 DRAM transfer job；两者互不依赖；
3. job 只在全局 `transfer_complete` 事件发生时产生 shard ACK；
4. 所有 PP/TP shards ACK、route/layout 仍匹配后，该 tier 才变为 `COMMITTED`；
5. 失败只把对应 tier 变为 `ABORTED` 并释放对应 reservation；另一 tier 不受影响；
6. Recovery Executor 永不读取 `PENDING`。

当前 offline path 已按这一合同接通：控制面 intent 只建立物理 transfer group；只有 GlobalDES
完成全部 shard 并发出 tier notification 后，DataParallelController 才推进对应 tier frontier。HBM
失败不会终止 DRAM group，DRAM 失败也不会回滚已提交 HBM。

### 8.3 hard fail-stop 与物理恢复

hard failure 与 recovery 也必须在同一事件堆中完成，而不是修改一个 metadata flag：

1. `failure_inject` 的优先级高于同时间 compute/transfer completion；
2. failure-domain resolver 先把 target 精确映射到 Scheduler/physical domain；非退化拓扑中无法精确
   表示的单 GPU/process target 直接 fail-close，禁止将其扩大成整个 DP；
3. 被命中的 in-flight stage completion 先取消，再终止对应 Scheduler process；exact worker 清除该
   worker 的 volatile GPU state，node failure 清除节点内全部 GPU 和 node-local DRAM；其他节点已提交的
   source-owned 副本不受影响；
4. 失败使受影响的 protection PENDING group abort，并只使对应 committed tier 失效；HBM 和 DRAM
   独立判定，不创建 target-owned destage；
5. recovery planner 根据失败后的 committed coverage 选择 Peer-HBM、NODE_DRAM 或明示 recompute/fallback，
   并创建 canonical `RecoveryLoadGroup`、目标 HBM hold 和每 PP/TP shard `TransferRecord`；
6. Peer-HBM 已位于 retry GPU 时可记录 zero-copy ownership activation；DRAM recovery 则占用 modeled
   DRAM-read/PCIe/NIC/fabric/H2D 资源。每个 shard adoption 都受 route/incarnation 和 hold 门控；目标再失效时
   已 adoption shards 与未 adoption remainder 分开结算，不伪造整组恢复；
7. Router 以 committed output position `k` 建立 route `v+1`，重放 `prompt + committed outputs`并只保留
   remaining `N-k`。每个 replay execution ACK 必须联结到同 route 的 terminal physical recovery admission；
   ADOPTED 需要 canonical group/transfer/hold 完整闭环，FALLBACK 需要明示丢弃且无 adopted shard/活跃 hold；
8. 只有物理恢复准入门通过后 Scheduler 才能续跑；repair 只恢复空 worker/empty volatile ledger，
   不能复活丢失的 KV。

这些能力已进入当前代码和针对性验证，但还没有形成正式 P1--P12 性能 lineage。P0 v5
进一步把 current `DataParallelController` + `OfflineAegisTransferBridge` 的 target-before-source rotation、
随后 source/node failure 收进同一个 immutable contract cell；它仍然只是 **contract-only evidence**，
不是 full Scheduler integration、真实 logits/RNG recovery 或性能证据。

## 9. 输出数据合同

每个实验 cell 至少落盘：

| 文件 | 主键/粒度 | 内容 |
| --- | --- | --- |
| `events.jsonl` | `event_id` | 全局有序事件、before/after、因果 parent |
| `rank_steps.jsonl` | `(event_id, dp, pp)` | batch mode、request ids、queue depth、predicted durations |
| `gpu_hbm_timeseries.jsonl` | `(timestamp, gpu_id)` | 全部 HBM 分账和 alpha/quota |
| `node_dram_timeseries.jsonl` | `(timestamp, node_id)` | 全部 DRAM 分账 |
| `kv_extents.jsonl` | `(extent_id, tier, shard)` | token range、位置、状态历史 |
| `transfers.jsonl` / `transfer_groups.jsonl` | `job_id/group_id` | ordinary HiCache、Aegis protection 与 Aegis recovery 的 class/path、bytes、ready/start/end、排队、service、shard ACK 和 group 终态 |
| `requests.jsonl` | `request_id` | logical/execution route chain、committed output position、TTFT/ITL/E2E、restored/recomputed tokens |
| `failures.jsonl` | `failure_id` | victim、fence、loss、recovery、repair |
| `audit.json` | cell | invariant 结果、Router-output-to-physical-recovery-admission gate、pending holds、hash、版本、证据边界 |

当前原子 artifact writer 使用 JSONL；未来可无损派生 Parquet/CSV 供分析和绘图，但派生表不能替代上述
原始事件与 manifest hash。

## 10. 必须通过的 invariant

1. 任意时间每张 GPU 的各 HBM 分账非负且总和不超过 capacity；
2. 任意时间每个节点 DRAM 分账非负且总和不超过 capacity；
3. 同一 extent 的每个 tier 状态只允许合法单向迁移；
4. `COMMITTED` 必须有完整 shard ACK 和已完成的物理路径；
5. `PENDING` 不具有恢复价值；
6. source failure 只清除该 failure domain 的 volatile HBM，不能清除其他节点已提交 DRAM；
7. target-first failure 不改变 source ownership，不把旧 HBM record 转移给 replacement；
8. HBM 与 DRAM copy 互相独立，事件图中不存在 HBM→DRAM edge；
9. extent bytes 的生产、复制、释放和丢失满足 conservation audit；
10. 两个独立 DP 的 makespan 为并发最大值而不是 service time 之和；
11. 相同 config、seed 和 code hash 产生相同事件顺序和数据 hash；
12. 任何 failure/recovery 结果可追溯到同一 request 的连续 route versions。

## 11. 三层 JSON 配置合同

仍只保留三层配置；虚拟时间和物理资源属于部署层，不新增隐藏的第四层。

### Request 层

负责：trace/artifact hash、workflow sampling、到达过程、context/output、dependency DAG、seed、load axes。
当前 [`global_des_workload.py`](src/global_des_workload.py) 对 P1--P10 fail-close 编译：

- P1--P5、P7--P10 使用 SHA-256 固定的 TraceLab v0.0.2；以完整 provider session 为单位，按
  provider × round-count bin 抽样，并用 `timing_events.event_index_1.timestamp` 生成全局到达时间；
- TraceLab 中 21 个受 23 次逆序 transition 影响的 session 在 seeded sampling **之前**以完整显式列表排除，
  编译器反查 pinned DuckDB，禁止静默 reorder/clamp；
- P5 的 low/high/recovery 是同一完整 request universe 的三个独立 phase cells；60 秒是 nominal control
  interval，不是 request cutoff，边界允许 spill 而不截断；
- P6 使用独立的
  [`controlled_prefix_reuse-v1.json`](data/controlled_prefix/controlled_prefix_reuse-v1.json)，其内容 identity、
  refcount 和生成器可审计，并明确标记 `empirical_measurement=false`，不能冒充 TraceLab；
- 编译结果带 request-universe/compiled hashes，并可生成真实 `TokenizedGenerateReqInput`，使 prefix identity
  进入真实 radix/Scheduler 路径。

请求编译 readiness 不是执行证据。P11 的 observed Weka compiled artifact 已建立：126 个 eligible
parallel-stage workflows 按 requests/workflow 精确分为 42/42/42，10 seeds × 3 strata 每格无放回选
1 个完整 workflow，得到 30 workflows/13,918 rows。其 manifest/requests/request-universe/predictor
hashes 见 [`data/README.md`](data/README.md)。P11 60 个正式 cells 仍未执行。

另一个与 workload bytes 独立的必要条件是 **admission-visible causal ticket**：

- TraceLab 只暴露顺序 session，不伪造 observed parallel DAG；编译器为每个当前 frontier 生成结构完整、
  exact-zero 的 singleton ticket；
- P4 在完整 TraceLab background 上叠加 8 个 controlled four-branch fork/join workflows；正 Slack 只来自
  冻结的 admission predictor，明确 `uses_simulated_completion=false`；
- P11 使用 126 个 workflow 的 workflow-disjoint 5-fold cross-fit，只读取 request-return prefix 可见特征；
  6,047/13,918（43.45%）有 modeled ticket，7,871 显式 fail-close-zero；
- P11 workflow-simultaneous coverage 为 0.928571，低于 0.95，且 safe-Slack LCB 非零数为 0；
- P12 只使用明示 analytical ticket，其结果始终是 modeled/preliminary/non-paper-ready。

所有路径仍禁止用请求完成后才观测到的 duration/WCT 倒推 Slack。显式 fail-close-zero 不关闭 Aegis
protection，只阻止把未知 Slack 当成正排序信号。

### Failure 层

负责：failure type/domain、victim selector、触发逻辑时间、duration/repair、correlation、paired shadow plan、seed。

当前 v3 公共合同是 deterministic `fail_stop`：10,000,000,000 ns 触发，同时 tie 使用
GlobalDES priority，不生成 short-run Poisson，也不包含 MTTF/hazard axis。当前 cluster 依照
实际 `CompiledDeployment` 做 exact failure-domain resolution：exact worker 和 dense whole-replica node target
可表示，正式配置也只使用 worker/node。单 GPU/process 只在 PP=TP=1 的退化拓扑中与一个
Scheduler worker 精确等价；非退化 PP/TP 中不可精确表示时，compiler 和 runtime 都 fail-close，
不将部分 shard 错当整个 DP。故障清空可表示域内 volatile HBM，node 失效还清空节点内全部 GPU
和 node-local DRAM；repair 只恢复空 volatile state。正式编译必须显式选择
`dense_first_v1`/`seed_mod_dense_order_v1` target selector，以及 `serial_sum_v1`/`overlap_max_v1`
repair-delay composition；不允许猜测。P3 还冻结
`trigger = 10 s × reference_gpu_count / selected_gpu_count`（参考 4 nodes × 8 GPUs，half-up 到 ns）以保持
pre-injection per-GPU exposure。P5/P9/P12 显式禁用 failure。

### Deployment 层

负责：model/parallel layout、显式 GPU placement、每 GPU HBM 分账、每节点 DRAM、link/copy-engine
拓扑、带宽/setup/争用模型、AegisKV policy、chunk、AIMD、predictor、DES tie-break 和输出采样规则。

每个 lineage 必须保存 source config、resolved config、schema validation、canonical SHA-256、代码 hash 和
实际使用的 predictor/profile hash。CLI override 只能生成新的 resolved lineage，不能静默修改现有配置。

当前 deployment provenance 已冻结 Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218` 和 Qwen3-32B revision
`9216db5781bf21249d130ec9da846c4624c16137`；32B revision gate 已解除。runtime compiler 还显式实现
per-node DRAM quota、P3 exposure-preserving trigger scaling 和
`no_algorithm_1 -> aegis_l1_l2.disable_slack_ordering=true`。当 simulator revision、failure target selector
和 repair-delay policy 使用 explicit frozen modeled bindings 时，P1--P12 的 2,134/2,134 cells 可编译。
这个数字只属于 config-to-runtime projection，不代表真实 Scheduler 已执行这些 cells。
当前正式模型部署均为 PP1，分别覆盖 TP1/TP4；PP2×TP4 只在 GlobalDES metadata contract 中检查
PP×TP shard/group 语义，不是真实 `OfflineSchedulerCluster` 任意 PP×TP 运行能力的证据。

## 12. 新论文实验矩阵

新套件使用稳定 ID `P0--P12`。每个 policy/failure pair 共享同一 request universe、route shadow、
failure event 和 seed。下面是纯 SGLSim 虚拟时间实验；暂不包含真机。

| 组 | 核心问题 | 主要轴 | 主图 |
| --- | --- | --- | --- |
| P0 | 模拟器是否因果、守恒且可复现 | DP/TP/PP、tie、容量边界、failure timing | invariant table + event-order diff |
| P1 | 单 worker failure 下 AegisKV 是否改善 WCT | policy × model | WCT CDF + paired effect |
| P2 | 收益如何随 load/failure domain 变化 | load × worker/node failure | effect heatmap + queue/resource pressure |
| P3 | 模型/context/部署规模是否改变收益 | model × context × nodes/DP | scale curves + per-GPU exposure |
| P4 | Algorithm 1/2/3 各自贡献什么 | no-Slack/no-placement/no-AIMD/DRAM-only | paired ablation panels |
| P5 | HBMQuotaAIMD 是否稳定保护 primary headroom | phase load × alpha/budget | alpha/primary/protection stacked timeline |
| P6 | prefix reuse 与 replica placement 的作用 | reuse × same/cross-node target | gain curve + transfer bytes |
| P7 | 路径带宽和 repair horizon 的敏感性 | topology/path bandwidth × repair | phase diagram + queue/service breakdown |
| P8 | 多 worker/correlated node failure 是否可扩展 | failure count/domain | WCT/lateness + recovery Gantt |
| P9 | 无故障正常路径开销是多少 | policy × load | TTFT/TPOT/WCT + copy traffic/HBM tax |
| P10 | HBM/DRAM 容量边界在哪里 | HBM reserve × node DRAM | phase heatmap + residency mix |
| P11 | observed multi-agent DAG 上是否有效 | AgentX/Weka strata × policy | paired workflow effect + Slack coverage |
| P12 | transfer/resource model 是否自洽 | size/path/contention | analytical-vs-DES + utilization |

冻结配置位于 [`configs_v3/`](configs_v3/)，由
[`global_des_config.py`](src/global_des_config.py) fail-close 校验并机械展开为 **2,146 cells**：

```text
P0 contract = 12
P1--P12    = 160+240+640+40+60+200+240+160+160+120+60+54 = 2,134
full suite = 12+2,134 = 2,146
```

P0 v5 的 12 个 contract cells 已执行并为 12/12 PASS，但仍是 non-performance gate。2,134 个
P1--P12 cells 只完成 runtime compilation audit，正式执行数为 **0/2,134**；因此完整
2,146-cell suite 未运行，`paper_ready=false`。

## 13. 论文必须绘制的资源级图像

除 WCT/TTFT/TPOT 等结果图外，至少加入以下内部状态图，否则读者无法验证机制确实执行：

1. **每 GPU stacked HBM timeline**：weights、runtime、primary KV、radix prefix、Aegis pending/committed、recovery pending；
2. **每节点 DRAM stacked timeline**：HiCache、Aegis pending/committed、free；
3. **DP rank swimlane**：arrival、prefill、decode、failure、recovery；
4. **KV transfer Gantt**：分开 ordinary HiCache、Aegis protection 与 Aegis recovery，标出 source/destination、HBM/DRAM、queue wait、service、ACK；
5. **failure zoom-in**：fence 前后 HBM loss、DRAM persistence、恢复 load 和 silent replay；
6. **residency mix**：HBM_ONLY、DRAM_ONLY、HBM_AND_DRAM、BYPASS 随时间/负载变化；
7. **resource phase diagram**：load × HBM/DRAM capacity 下的 pressure、eviction、WCT；
8. **copy-path utilization**：NVLink/PCIe/NIC/link busy fraction 与排队时间。

资源时间线默认展示一个代表性 deterministic seed；统计结论必须来自所有 seeds，代表性 seed 的选择规则
预先写入配置，不能按曲线好看程度挑选。

## 14. 实施顺序与完成门

### Phase A：DES 核心

- 全局 event kernel；
- GPU/DRAM ledger；
- KV extent state machine；
- multi-resource transfer reservation；
- event/timeseries journal；
- invariant tests。

实现状态：kernel、逐 GPU/节点账本、extent、资源日历、事件取消、failure/repair、tier-local 全 shard
commit 和 journal 均已实现。P0 v5 的 12 个 deterministic contract scenarios 全部通过，但该 lineage
明确是 contract-only。Phase A 的 kernel 门已通过；不可将其扩大解释为后续 phase 或性能门通过。

### Phase B：SGLang scheduler 接入

- `OfflineSchedulerCluster` 作为唯一离线执行入口；
- scheduler prepare/complete 事件写入 global journal；
- Device/Host allocator mutation 写入 resource ledger；
- TP shard 映射到显式物理 GPU；
- 保持 online wall-clock 路径不变。

实现状态：`OfflineSchedulerCluster` 已让真实 Scheduler worker、stage completion、arrival 与物理 snapshot
进入同一 GlobalDES；普通 HiCache/radix 的 D2H/H2D intent 也已经由
`OfflineHiCacheTransferBridge` 展开为与 AegisKV 共享资源的显式 transfer lifecycle，包括全 shard commit、
故障 rollback 和 compute dependency gate。尚缺新的不可变 Scheduler-integration gate lineage，以及与
bounded online 的正式 accuracy artifact。现有 P0 v5 不覆盖此 phase。另外，真实 Scheduler 正式模型只是
PP1（TP1/TP4）；PP2×TP4 只是 metadata contract，不能当作任意 PP×TP scheduler integration。

完成门：相同 request/failure config 下，offline 与 bounded online 的路由、batch/KV 生命周期、完成顺序
通过既有 accuracy gate；资源时间线 invariant 全通过并写入新的 immutable manifest。

### Phase C：AegisKV transfer/commit 接入

- HBM/DRAM independent admission 预留具体资源；
- transfer-complete 后才 commit；
- failure abort/loss/recovery；
- AIMD 使用真实 ledger pressure。

实现状态：protection 与 canonical recovery physical jobs、HBM/DRAM 独立 completion/notification、全 PP/TP
shard commit/adoption、PENDING abort、空状态 repair、Router committed-position silent replay、物理 recovery admission gate
和 ordinary HiCache 共用资源争用已进入当前代码。recovery 以 `RecoveryLoadGroup` 与每 shard
`TransferRecord` 记录 load/adoption，不再是只有 metadata 的恢复标志。HBM/DRAM protection 继续保持
source-owned independent admission，禁止 HBM→DRAM destage。

限制也必须保留：非退化 PP/TP 下的单物理 GPU/process target 不能精确对应当前 Scheduler
worker，因此是 fail-close 而非已模拟；Router 续跑的 output token 是合成元数据，不是真实 logits/RNG；
带宽、setup 和 predictor 都是 modeled。此外，101529 P4 smoke 仍是不可改写的 pre-fix
extent-lifetime-leak 负证据；104732 仍是将 7 completed + 1 scheduled 错标为 complete 的负证据。
它们不能被当前代码或测试追认为有效 post-fix lineage。

完成门：除上述 code-level tests 外，target-first/source-first/PENDING-window/correlated-node 场景还必须由
同一新 gate lineage 输出 raw transfer/group/failure/resource tables 和 conservation audit；在此之前 Phase C
不得标记为 paper evidence complete。

### Phase D：实验与图像

- P0--P12 三层 JSON 已冻结为 `configs_v3`，且 suite 声明保持 `planned_unexecuted`；
- model revisions、P4 causal overlay、P11 observed artifact、per-node DRAM quota、P3 trigger scaling 和
  no-Algorithm-1 adapter 已绑定；P1--P12 2,134/2,134 在 explicit frozen modeled bindings 下可编译；
- P1--P12 runner/check/resume 已实现；P11 predictor coverage 0.928571 < 0.95 仍应 fail-close paper claim；
- 已产出 P1/P12 各1-request、P4 8-request 和 P11 一个完整 57-request workflow 的
  smoke/preliminary/non-paper-ready lineage；P4 101529 是 pre-fix artifact，104732 因终态聚合 bug 无效；
- 每组由同一 immutable lineage 生成原始表、统计、图像和报告；
- 旧 613-cell 结果留档，不与 v3 汇总。

当前状态：尚无正式 P1--P12 performance lineage，正式完成数仍为 0/2,134；完整 2,146-cell suite 未运行。代表性 smoke 的
claims 固定为 `paper_ready=false`、`full_2146_cell_suite_executed=false`，只能证明部分执行链工作。

完成门：所有 preregistered cells 完成、无 invariant failure、所有图可从 lineage 一键重建，并通过
校准/证据审计。此时才能把新结果写入 `EXPERIMENT_REPORT.md` 的“当前论文结果”部分。

## 15. 当前结论

混合 donor 路线已经落实为 SGLang-aligned offline Scheduler + GlobalDES + physical resource/transfer
bridges，而不是整体替换 SGLSim。代码级集成已经覆盖逐 GPU HBM、逐节点 DRAM、独立 HBM/DRAM
protection、ordinary HiCache contention、hard fail-stop、canonical physical recovery 和 Router silent replay；下一证据门不是
再增加一个 metadata shortcut，而是：

1. 生成新的 immutable Scheduler/Aegis active-failure integration lineage，逐项审计 request closure、ordinary/protection/recovery
   transfer groups、Router route chain、physical recovery admission、pending holds、terminal resource balances 和
   manifest/audit consistency；同时保留 101529 pre-fix 与 104732 invalid artifact 的原始负证据口径；
2. 将非退化拓扑的单 GPU/process failure 继续作为明示 fail-close 能力边界；只有引入真正
   physical-shard Scheduler actors 后，才能宣称该粒度故障已模拟。PP2×TP4 也不能从 metadata contract
   外推到 real Scheduler。
3. 让 P11 workflow-simultaneous interval 达到预注册 0.95；若 safe Slack 仍为零，明确报告 Algorithm 1 null；
4. 用真机数据校准 predictor、copy bandwidth/setup 和资源争用；在此之前所有时延与性能图只能标为 modeled，
   Router continuation 也必须继续标为 synthetic-token metadata replay，不是 logits/RNG 一致恢复；
5. 完成 simulator/timing/failure binding 的 run-owned immutable lineage，然后才运行正式 P1--P12 的
   2,134 cells 并生成完整 2,146-cell suite。

在新 lineage 通过前，最准确的结论是：**目标 simulator 的核心执行链已实现并通过代码级集成测试，
但正式 P1--P12 仍为 0/2,134，完整 2,146 cells 未跑；性能与 paper-ready 结论均未建立。**
