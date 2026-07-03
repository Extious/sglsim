# AegisKV：面向多智能体 LLM Serving 故障恢复的 Workflow-Aware KV 备份

生成时间：2026-07-03T01:41:09

证据状态：

```text
figures=25, claim_ready=2, preliminary_valid=23, final_measured=1
```

重要说明：当前部分定量图仍使用 `evidence_tier=sglsim-preliminary`。
在 `--strict-final` 通过前，本文件只是内部 HPCA 风格论文草稿。

主要图表资产：

- 图表预览：[../figures/gallery.md](../figures/gallery.md)
- 图表索引：[../figures/manifest.json](../figures/manifest.json)
- 证据状态：[../figures/evidence_status.json](../figures/evidence_status.json)


---

# 00. 论文大纲

## 工作标题

`AegisKV: Workflow-Aware KV Backup for Failure Recovery in Multi-Agent LLM Serving`

## 核心论点

Multi-agent LLM serving 中，GPU failure 的影响高度不均匀。丢失同样大小的 KV state，对已经终止的探索分支可能几乎没有影响；但如果丢失的是 main agent、fan-in 前的 final subagent，或者需要恢复执行的 tool-blocked agent 的 KV，就可能拖慢整个 workflow。

AegisKV 利用这种偏斜。本文假设 SGLang 在 radix 插入后已经固定执行 `write_through` D2H，然后加入 failure-aware HostBackup visibility，让 retry rank 能使用 failed rank 上仍然存活的 Host KV。AegisKV 还决定哪些 SGLang radix-tree KV segment 值得额外复制到 peer GPU（`GPUReplica`）、副本应该放在哪个 peer 上，以及 failure 后应该从 GPUReplica restore、从 HostBackup restore，还是 recompute 缺失 suffix。

## 相比旧稿的变化

旧稿使用三个 tier：

```text
GPUReplica, LocalHost, RemoteStore
```

新稿刻意移除 `RemoteStore`，聚焦在更紧凑的两级系统：

```text
GPUReplica, Host
```

这让第一篇论文更容易在 SGLang 和 SGLSim 上落地，同时也暴露出论文必须显式建模的系统约束：

- `GPUReplica`：恢复最快，但会消耗 peer DP-rank 的 HBM 和 P2P 带宽。
- `Host`：容量更大、成本更低的本地 backup，但 failure recovery 需要支付 H2D restore latency。
- `None`：没有可用的 failure-protected frontier；failure recovery 只能从最近的 ordinary prefix frontier recompute。
- Host Backup 固定采用 `write_through`：baseline 和 AegisKV 都支付 D2H 成本，但只有 AegisKV 会把 failed-rank Host KV 暴露给 retry rank。
- GPUReplica placement 必须同时考虑 peer load、HBM pressure、retry route、P2P cost 和 KV locality。
- Host 与 GPUReplica 可能覆盖同一个 radix prefix，所以价值应按 marginal safe content frontier 计数，而不是按重复字节计数。

## 贡献主张

1. **系统模型。** AegisKV 将 multi-agent failure recovery 建模为 SGLang radix-tree KV segment 上的在线、耦合、多副本 content placement 问题。
2. **策略。** AegisKV 利用 workflow criticality、lifecycle state、restore-vs-recompute cost、peer load/locality、Host restore pressure 和 marginal shared-prefix value，选择 GPUReplica placement 与 failure-time recovery action。
3. **系统设计。** AegisKV 集成到 SGLang radix cache、HiCache copy queue、lock/ref 管理、eviction、scheduler、data-parallel controller 和 failure retry path 中，不替换 SGLang 的 KV manager。
4. **评估方法。** AegisKV 同时在真实 SGLang 与 SGLSim 中评估：SGLang 验证 runtime feasibility/correctness，SGLSim 探索 DP scale、peer placement、shared-prefix overlap、failure timing 和 workload diversity。

## 论文结构

1. 摘要。
2. 引言与动机。
3. 背景与相关工作。
4. 系统模型与设计。
5. SGLang 与 SGLSim 实现。
6. 实验计划。
7. 初步结果与图注。
8. 图表与预期结果。
9. 结论与下一步。

## 一句话差距

已有系统知道如何 cache、offload、replicate 或 restore KV；AegisKV 关注的是：**固定 write-through 后哪些 Host KV frontier 应该跨 DP rank 可见、哪些 segment 值得放置 peer GPU replica、应该选择哪个 peer，以及 failure 下何时 restore 优于 recompute**。


---

# 01. 摘要

多智能体 LLM 应用把一次用户请求扩展为一组存在 fan-out、fan-in、tool pause、resume、retry 和 synthesis 关系的 agent 调用图。与普通 chat serving 不同，GPU failure 对这类 workload 的影响并不均匀：丢失一个已终止探索分支的 KV 可能只带来局部重算，而丢失 main agent、最后一个 subagent、tool-return 前缀或 synthesis hot tail 的 KV 会阻塞后续 agent，并显著拉长 workflow completion time。

现有 KV cache / offload / recovery 系统已经可以在 GPU、Host、Remote 等 tier 之间移动 KV，也有工作研究 KV restore 与 recompute 的性能边界。但这些系统通常缺少 multi-agent workflow 语义：它们知道如何恢复 KV，却不知道哪段 KV 的 failure-time value 更高。另一方面，已有 agent-aware serving 系统主要优化 normal-path routing、cache hit 或 memory scheduling，而不是 failure 后的 survival frontier 与 restore order。

本文提出 AegisKV，一个面向 multi-agent LLM serving 的 workflow-aware KV backup layer。本文只考虑 GPU failure：source DP rank 的 GPU KV 丢失，但该 rank / worker 的 Host KV cache 仍然存在。AegisKV 固定采用 SGLang `write_through` Host Backup，即 radix tree 插入后立即触发 D2H；baseline 也执行同样 D2H，但 retry 到另一个 DP rank 后无法感知 failed rank 的 Host KV，因此等价于没有可用 backup。AegisKV 的 HostBackup 贡献是记录 ACK 后的 cross-rank recoverable frontier，让 retry rank 可以使用 failed rank 上仍存活的 Host KV。除此之外，AegisKV 还可以选择把高价值 segment 复制到另一个 DP rank 的 peer GPU，即 `GPUReplica`。由于 HostBackup 与 GPUReplica 可能覆盖同一 shared radix prefix，AegisKV 按边际 safe content frontier 计算收益，而不是按副本数重复计算 value。发生 GPU failure 后，AegisKV 为 retry path 选择最深可恢复 frontier，并在 GPUReplica restore、Host restore 和 suffix recompute 之间做决策。

AegisKV 的问题建模借鉴 `One Pool - Two Caches` 的动态资源划分思想：failure-protected KV copies 与 normal-path serving state 共享 HBM、Host memory、P2P/H2D bandwidth 和 scheduler time，placement update 也会带来 actuation cost。但在本文简化版中，Host D2H 是固定共同成本，核心在线决策集中在 GPUReplica peer 选择、HostBackup/GPUReplica/recompute 的 failure-time 选择，以及 shared-prefix 边际价值计算。AegisKV 也借鉴 `DRLPart` 的 state/action/reward 框架，但第一版采用可解释的 event-driven value-density policy；learning-based controller 只作为未来替代的 budget/value estimator。实验计划在 SGLang 真实推理系统上验证 runtime feasibility、radix correctness、backup overhead 和 failure recovery latency，在补齐 AegisKV hook 后的 SGLSim 上扩展到更大 DP scale、peer placement、shared-prefix overlap、failure timing 和 workflow diversity。预期结果是：在 failure impact skew 明显的 multi-agent workload 中，AegisKV 相比 fixed write-through but Host-invisible baseline、uniform GPUReplica backup、static critical-path heuristic 和 request-level backup，能降低 workflow completion time 与 critical-agent recovery latency，同时把 normal-path overhead 控制在可接受范围内。


---

# 02. 引言与动机

## 1. 多智能体 serving 改变了 failure target

普通 LLM serving 的 failure recovery 通常以 request 为单位：一个 GPU 或 worker 失败后，系统需要重启或重算受影响 request。Multi-agent serving 中，request 不再是孤立对象。一次用户任务会展开为动态 workflow：

```text
O0 main agent
  -> S1, S2, S3 subagents
  -> O0 fan-in / summary
  -> S4, S5, S6 second round
  -> O0 final synthesis
```

这个 workflow 的 completion time 由 critical path 决定。KV state 的 value 取决于它属于哪个 agent、处于哪个 lifecycle、是否即将 resume、是否位于 fan-in 前、是否已经被 summary/compaction 替代。

因此 AegisKV 的动机不是“KV 很大，所以要备份”，而是：

> 同样大小的 KV state，丢在不同 workflow 位置，会产生完全不同的 failure-time penalty。

## 2. 同等大小 KV loss 应作为首个动机实验

论文必须先证明 failure impact skew。最直接实验是：

```text
Inject equal-size KV loss into different agent/segment positions
and measure delta workflow completion time.
```

至少比较：

- terminated exploration branch；
- running non-final subagent；
- last running subagent before fan-in；
- tool-blocked main agent before resume；
- synthesis hot tail；
- shared prefix before fan-out。

预期结果：

- terminated / summarized branch 的 delta WCT 接近 0；
- non-final parallel subagent 的 penalty 被其它 subagent 覆盖，delta 中等；
- last subagent、main agent、synthesis tail 的 delta WCT 明显更高；
- shared prefix loss 会触发 replay amplification，影响多个 descendant agents。

如果这组实验不成立，AegisKV 的核心主张就不稳。

## 3. 为什么静态 backup 不够

借鉴 `One Pool - Two Caches`，AegisKV 也面对动态 resource partitioning 问题。One Pool 的核心现象是：两个 cache 共享 HBM，最优 allocation ratio 随 workload 变化；静态 ratio 会错失性能；在线调整又会带来 refill interference。

AegisKV 中对应关系如下：

| One Pool | AegisKV |
|---|---|
| EMB cache vs KV cache compete for HBM | normal-path serving state vs failure-protected KV copies compete for HBM/Host/bandwidth |
| allocation ratio `alpha` changes with workload | backup budget split between GPUReplica and Host changes with workflow/failure/runtime state |
| physical refill interferes with serving path | backup copy, Host write, peer-GPU copy, restore H2D/P2P interfere with prefill/decode |
| routing must consider cache locality and load | retry/restore must consider frontier locality, peer rank health, queue pressure |

AegisKV 还比 One Pool 更复杂：两个 backup placements 不是保护互不相关的对象，而是可能保护同一段 radix prefix。shared prefix 同时存在于 Host 与 GPUReplica 时，物理 copy 成本要分别计入，但 failure-time value 只能按边际 safe content frontier 计一次。GPUReplica 也不能只看有没有容量，还要选择哪个 peer DP rank，这个选择同时受 peer load、KV locality、retry route、HBM 和 P2P queue 影响。

因此，AegisKV 不应该写成固定策略：

```text
always backup critical agents
always use GPUReplica for critical agents
always use Host for long prefix
```

更合理的表述是：AegisKV 在每个 event / decision epoch 里，根据 workflow value、restore/recompute gain、resource pressure 和 actuation cost 选择 placement。

## 4. Restore 不一定总优于 recompute

`LLM KV-cache - To Restore or To Recompute` 的关键启发是：restore 与 recompute 的优劣取决于 GPU compute throughput、storage/Host bandwidth、interconnect、context length、SLO 和 request rate。

AegisKV 因此不能只最大化 backup hit。发生 failure 后，每个 segment 都要比较：

```text
RestoreCost(segment, tier)
vs.
RecomputeCost(segment suffix)
```

恢复策略应满足：

- 如果 GPUReplica 已 ACK 且 peer rank 健康，短 critical hot tail 可直接从 peer GPU frontier 恢复；
- 如果 Host 已 ACK 且 prefix 很长，Host restore 通常比从 root recompute 更好；
- 如果 segment 很短、Host/P2P 拥塞或 restore 会阻塞 critical path，则 recompute suffix 可能更优；
- 如果 Host frontier 存在但 metadata 不一致，必须当作 miss，不允许 unsafe hit。

## 5. 为什么第一篇论文只需要两个 backup tier

本文只保留：

```text
B = {GPUReplica, HostBackup}
```

这样设计有三个好处：

1. **实现闭合。** SGLang 已有 radix cache、HiCache、Host memory pool、DP controller 和 retry/failover 路径；SGLSim 已有 HiCache/radix/pipeline 模拟基础，但仍需要补齐 AegisKV policy hook 和在线 workflow benchmark。
2. **实验更清楚。** baseline 与 AegisKV 都固定执行 `write_through` D2H，因此可以直接比较 Host KV 对 retry rank 不可见与可见的差异，再比较 fast-but-expensive peer GPU backup。
3. **论文主线更窄。** 不引入 RemoteStore 后，reviewer 更难把工作误读成 generic remote KV storage 或 wide-area KV cache。

本文中的 `HostBackup` 指 source DP rank / worker 侧的本地 Host memory backup。若把 Host 解释成同一 GPU 上的 duplicate copy，则单 GPU failure 时副本会一起丢失，不能作为 failure recovery frontier。


---

# 03. 背景与相关工作

本章的作用不是罗列所有 KV cache 或 fault tolerance 工作，而是把 AegisKV 放在几个已有研究脉络中：KV cache 管理、restore-vs-recompute、fault-tolerant LLM serving、agentic workflow serving、locality-aware scheduling，以及动态资源划分。对应 PDF 已整理到 `../related_works/pdfs/`，索引见 `../related_works/README.md`。

## 1. SGLang KV 生命周期背景

AegisKV 不重新实现 KV cache manager，而是叠加在 SGLang 的 KV lifecycle 上。需要利用的 SGLang 机制包括：

- radix-tree prefix cache；
- request active ref / lock release；
- HiCache fixed `write_through` D2H；
- Host ACK、H2D restore、GPU/Host eviction；
- data-parallel controller；
- scheduler retry / failure recovery path；
- token / radix path metadata；
- optional peer-rank copy path。

当前远程 SGLang 代码位于 `hkbugpumanode:~/sglang`，分支为 `multi-level-backup`。相关路径包括：

```text
python/sglang/srt/mem_cache/radix_cache.py
python/sglang/srt/mem_cache/hiradix_cache.py
python/sglang/srt/mem_cache/memory_pool.py
python/sglang/srt/managers/cache_controller.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/managers/data_parallel_controller.py
python/sglang/srt/managers/tokenizer_manager.py
python/sglang/srt/mem_cache/storage/remote_backup/
```

虽然当前 paper scope 不使用 `RemoteStore`，`remote_backup` 和 `prefetch_from_storage(... lookup_dp_rank=...)` 说明 SGLang 已经有 backup/restore-oriented HiCache storage backend 的扩展点。AegisKV 应复用这类接口思想，但把 first paper 的 placement 限制在 `GPUReplica` 与 fixed `HostBackup`。

## 2. SGLSim 模拟背景

当前远程 SGLSim 代码位于 `gpu10:~/sglsim`，分支为 `aegiskv`。SGLSim 的 SRT 模块模拟了 SGLang runtime 布局：

```text
python/sglsim/srt/managers/tokenizer_manager.py
python/sglsim/srt/managers/scheduler.py
python/sglsim/srt/managers/data_parallel_controller.py
python/sglsim/srt/managers/cache_controller.py
python/sglsim/srt/mem_cache/radix_cache.py
python/sglsim/srt/mem_cache/hiradix_cache.py
python/sglsim/srt/mem_cache/hicache.py
python/sglsim/srt/mem_cache/transfer.py
python/sglsim/srt/observability/bench_metrics.py
benchmark/synthesis_requests/
benchmark/agentic_workflows/
```

这次源码检查显示，当前可见的 SGLSim source 已经能模拟 SGLang 的基础 HiCache/radix 行为：

```text
D2H/H2D transfer duration
pending backup overlap
fixed write_through Host Backup
Host-only load-back
GPU eviction and Host eviction
scheduler pipeline accounting
```

但当前可见源码中还没有完整的 AegisKV policy hook 或物理 KV copy path；`benchmark/agentic_workflows` 已恢复为可重复执行的在线 workflow wrapper。 因此本文应把 SGLSim 定位为需要补齐 AegisKV online policy、GPUReplica pool 和 HostBackup directory 的 simulator 基座。

## 3. KV cache reuse、offload 与 restore

LLM serving 系统已经大量研究 KV cache reuse、prefix cache、multi-tier offload、remote storage 和 restore/recompute。代表性工作包括 `Prompt Cache`、`KVShare`、`Shared Disk KV Cache Management`、`Position - Disaggregated LLM KV-Cache Storage Requires Coordination-Free Consistency Abstractions`、`Apt-Serve`，以及 `LLM KV-cache - To Restore or To Recompute, That Is the Question`。

这些工作给 AegisKV 三个直接启发：

- KV cache 的物理位置会影响 TTFT、吞吐、restore latency 和 memory pressure；
- restore 不一定优于 recompute，边界取决于 prefix length、bandwidth、queueing 和 compute cost；
- KV metadata 必须和 eviction/offload/restore 的实际生命周期对齐，不能只维护逻辑目录。

AegisKV 的差异在于 failure recovery objective。现有 KV cache 工作主要优化 normal path latency、cache hit、offload efficiency 或 generic restore/recompute tradeoff。AegisKV 则问：

> 当 GPU failure 发生时，哪些 SGLang radix-tree KV segment 的丢失会显著增加 multi-agent workflow completion time，应该被 HostBackup visibility 或 GPUReplica 保护？

因此 AegisKV 的 restore/recompute decision 不是只看 block size 与 bandwidth，而是把 workflow dependency、agent lifecycle、failure exposure、safe frontier 和 shared-prefix marginal value 纳入决策。

## 4. 容错 LLM serving

Fault-tolerant LLM serving 和 resilient inference 已经研究 KV streaming、checkpoint、replication、rerouting、TP recovery、MoE recovery 和 serverless/pipeline recovery。相关工作包括 `DejaVu`、`FailSafe`、`AnchorTP`、`ReviveMoE`、`ReaLM` 和 `KevlarFlow`。

这些系统证明 LLM serving 的 failure recovery 不能只依赖重新执行整个 request。它们通常从 request/session、pipeline stage、TP/MoE group 或服务实例维度恢复状态。AegisKV 的粒度更细：

```text
unit = SGLang radix-tree KV segment
value = expected saved workflow completion time
safe frontier = HostBackup ACK or GPUReplica ACK plus valid eviction state
```

AegisKV 不声称第一次做 LLM fault tolerance。它的区别是：

- request-level backup 会浪费空间在 terminated / summarized branches；
- agent-level backup 仍然太粗，因为 critical agent 的所有 KV segment 并非等价；
- AegisKV 需要在 same workflow 内区分 main agent、fan-in 前 subagent、tool-return prefix、hot tail 和 dead branch；
- GPUReplica 与 HostBackup 可能覆盖同一 shared prefix，因此 recovery value 只能按 marginal safe content frontier 计算。

## 5. Agentic workflow serving

Agentic serving work 优化 multi-agent workflow 的 normal-path execution、routing、micro-task scheduling、context routing 和 dependency-aware overlap。相关工作包括 `AAFLOW`、`Efficient Serving for Agentic LLM Workflows via Micro-Task-Level Parallelism`、`Weaver`、`Ceres`、`HEXGEN-TEXT2SQL` 和 `RCR-Router`。

这些工作强调 agent workflow 与普通 chat serving 不同：请求之间有 dependency graph、fan-out/fan-in、tool stall、structured memory 和 role-specific context。AegisKV 继承这一观察，但研究目标不同：

> Existing agent-aware serving systems optimize how workflows run when there is no failure. AegisKV optimizes which workflow KV state survives and how it is restored after failure.

因此 AegisKV 不应与 agent-aware router 或 workflow scheduler 竞争 normal-path cache hit。AegisKV 应作为 failure-aware KV protection layer，使用 workflow metadata 给 radix segment 估值，并把结果反馈给 SGLang 的 HostBackup directory、GPUReplica admission 和 retry routing。

## 6. Locality-aware scheduling 与 memory-constrained serving

LLM serving 的调度研究已经关注 KV locality、memory constraints、fairness 和 online scheduling。相关工作包括 `Online Scheduling for LLM Inference with KV Cache Constraints`、`Optimizing LLM Inference - Fluid-Guided Online Scheduling with Memory Constraints`、`Locality-aware Fair Scheduling in LLM Serving` 和 `Apt-Serve`。

这些工作说明 routing/scheduling 不能只看负载，还要看 KV locality 和 memory pressure。AegisKV 在 GPUReplica peer selection 中复用这个原则，但目标函数不同：

```text
normal scheduling:
  choose where to run a request now

AegisKV GPUReplica placement:
  choose which peer should hold a failure-protected replica
  so a future retry can recover faster without hurting normal serving
```

因此 AegisKV 的 peer score 同时考虑 peer load、HBM free pages、existing prefix locality、retry route probability、P2P queue delay 和 failure-domain correlation。只选择 least-loaded peer 或 most-local peer 都不够。

## 7. 动态资源划分

AegisKV 的问题建模主要借鉴 `One Pool - Two Caches`、`DRLPart` 和 `FleetIO`。

| Paper | What to borrow | How AegisKV uses it |
|---|---|---|
| One Pool - Two Caches | shared HBM pool, dynamic ratio, incremental update, refill interference, routing coupling | `alpha_gpu` controls peer GPUReplica capacity; updates are slow control-epoch decisions, not per-request changes |
| DRLPart | state/action/reward, resource pressure, prediction and bad-decision correction | first paper uses an explainable value-density policy; DRL-style controller is future work |
| FleetIO | multi-tenant storage placement under dynamic workload and migration cost | motivates separating logical placement benefit from physical migration/copy cost |

AegisKV 比 One Pool 更复杂：One Pool 的两个 cache 近似保护不同对象，而 AegisKV 的 HostBackup 与 GPUReplica 可能保护同一段 radix prefix。因此 AegisKV 需要分开计算：

```text
physical bytes
GPUReplica protected budget
HostBackup visibility state
shared-prefix marginal recovery value
copy / demotion / invalidation cost
```

## 8. 定位总结

| Area | Representative works | What they solve | AegisKV gap |
|---|---|---|---|
| KV cache reuse/offload | Prompt Cache, KVShare, Shared Disk KV Cache, Apt-Serve | normal-path cache hit, offload, reuse, storage hierarchy | failure-aware safe frontier selection for multi-agent workflow KV |
| Restore vs recompute | LLM KV-cache - To Restore or To Recompute | cost boundary between restore and recompute | add workflow value, failure exposure, Host/GPUReplica choices |
| Fault-tolerant LLM serving | DejaVu, FailSafe, AnchorTP, ReviveMoE, ReaLM, KevlarFlow | resilient serving at request/session/model/system level | segment-level workflow-aware KV protection over SGLang radix tree |
| Agentic workflow serving | AAFLOW, Weaver, Ceres, RCR-Router, HEXGEN | normal-path workflow routing and scheduling | failure-time KV survival and recovery |
| Locality-aware scheduling | Online Scheduling, Fluid-Guided Scheduling, Locality-aware Fair Scheduling | request scheduling under KV locality and memory pressure | peer GPUReplica placement under future retry locality and HBM budget |
| Dynamic partitioning | One Pool, DRLPart, FleetIO | adaptive resource partitioning and migration cost | coupled Host/GPUReplica safe-cover accounting |

One-sentence related-work gap:

> Existing systems know how to cache, offload, replicate, or restore KV; AegisKV decides which fixed write-through Host KV frontier should be visible across DP ranks, which workflow-critical segment deserves a peer GPU replica, where that replica should live, and whether failure recovery should restore or recompute the suffix.


---

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


---

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


---

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


---

# 07. 初步结果与图注

This section is the current result narrative for the draft. It is intentionally
split into two layers:

1. **Paper narrative.** What each figure is meant to prove and how the results
   should be read.
2. **Evidence maturity.** Whether the current artifact is final evidence or a
   preliminary SGLSim counterfactual used for internal paper development.

At the time of this draft, most quantitative result figures are backed by
preliminary measured CSVs refreshed from SGLSim hard-failover replay matrices,
direct replay-derived counterfactuals, and small SGLang overhead smokes. The
only current quantitative strict-final-ready figure is Fig. 10b, which uses a
dedicated 3-run online SGLSim shared-prefix sweep with low/mid/high overlap
scenarios. Fig. 1 is now treated as hybrid-ready trace-derived motivation,
not strict-final real-system evidence. The older policy-model SGLSim rows are
labeled:

```text
evidence_tier=sglsim-synthesis-derived
source=sglsim_synthesis_request_details_policy_model_repeated
```

Newer rows use narrower labels such as `sglsim-runtime-replay-derived` or
`sglsim-runtime-replay-calibrated` when they are computed directly from replay
request details. These plots are stronger than hand-drawn expected trends
because they are reproducible from real SGLSim serving `request_details.csv`
rows, aggregate repeated runs with std/95% CI where available, and are
validated by `figures/scripts/validate_figure_evidence.py`. Fig. 11 now uses
`sglsim-agentic-online-preliminary` rows from
`codex-normal-overhead-online-r3-20260628` for the baseline/AegisKV online
no-failure comparison, while retaining replay-derived uniform Host/GPU
comparators. It also contains `sglang-small-paired` rows from seven matched
DP=2 SGLang overhead jobs on `hkbugpumanode`, including larger 64-request and
128-request / 128-token repeats. Jobs `32977` and `32978` additionally provide
DP=2
real-SGLang failure-control-path smokes: in both runs, 4/4 routed in-flight
requests were captured, re-dispatched, completed, and returned
`sglext.failover` after `/simulate_gpu_failure`. Job `32980` adds the first
storage-backed SGLang smoke by enabling `remote_backup`; it reports 5,132
pre-failover backed-up tokens across 4 failover-retried requests. The paired
artifact `failure_visibility_pair_32978_32980` now compares job `32978`
against job `32980`: the Host frontier increases from 0 to 5,132 backed-up
tokens, while p99 latency normalizes to 1.108x. This supports frontier
visibility, not final restore-latency benefit. Two follow-up probes refine the
remaining gap. Job `32981` disables request-aware remote-backup retention and
still sees retry prefetch revoked with `data_pages=0`, ruling out lease GC as
the main miss cause. Job `32982` uses a single request, writes the long prompt
from `page_start=0`, and the retry rank sees `data_pages=1297`,
`Prefetch ACCEPTED`, and `prefetch_get` hits, but the default `best_effort`
policy terminates before IO completion and admits with `#cached-token: 0`.
Job `32983` reruns the same single-request shape with
`hicache_storage_prefetch_policy=wait_complete`; it completes prefetch IO with
`completed=1297 / expected=1297` and retry prefill admits with
`#cached-token: 1297`. Job `32984` adds the matching no-storage single-request
control; the resulting `failure_visibility_pair_32984_32983` artifact records
a 1,297-token Host frontier delta and 1.124x visible/no-storage p99 latency
ratio. Job `32985` then reruns the four-request remote-backup case with
`wait_complete`; it still reports `data_pages=0` for all retry prefetches,
confirming that the concurrent shape needs split-prefix-aware restore rather
than only a different prefetch stop policy. Job `32987` removes that shared
prefix by enabling `UNIQUE_REQUEST_PREFIX=1` and restores four concurrent
requests successfully: all four retry queries reach `data_pages=1301`,
all four prefetch IOs complete `1301 / 1301` tokens, and each retry prefill
admits with `#cached-token: 1301`. Job `32989` is the matching no-storage
unique-prefix control, and the `failure_visibility_pair_32989_32987` artifact
records a 5,204-token Host frontier delta with visible/no-storage p99 latency
ratio 1.308x. Job `32991` applies the split-prefix storage fix and reruns the
shared-prefix four-request shape with `remote_backup + wait_complete`; all four
retry queries reach `data_pages=1297`, all four prefetch IOs complete
`1297 / 1297`, and each retry prefill admits with `#cached-token: 1297`. The
new `failure_visibility_pair_32992_32991` artifact compares the same-code
no-storage shared-prefix control against the patched storage-backed run: the
Host frontier increases from 0 to 5,188 backed-up tokens with a 1.210x
visible/no-storage p99 latency ratio. Jobs `32993`/`32994` repeat the paired
shape with a longer shared prefix (`PROMPT_REPEAT=256`) and extend the visible
frontier to 20,548 backed-up tokens. The visible run materializes all four
retries with `data_pages=5137` and retry `#cached-token: 5137`, but the visible
p99 latency is still 1.630x the no-storage control. Jobs `32998`/`32999` then
isolate failure-time visibility under matched normal write cost: both jobs
write the same 20,548 remote-backup tokens, while `32998` disables retry
storage prefetch and `32999` materializes 5,137 cached tokens per retried
request. The visible run remains slower at 1.487x the lookup-disabled control,
so the new pair strengthens the Fig. 11b visibility mechanism evidence without
making it final recovery-benefit evidence. Jobs `33002`/`33003` rerun the same
matched-write-cost shape with `best_effort` instead of `wait_complete`: both
write remote-backup tokens, the visible retry path finds `data_pages=5137`, and
prefetch partially materializes before retry prefill. The repeated pairs
`33002/33003`, `34333/34334`, and `34336/34337` now aggregate to 0.961x
visible/control p99 with 0.016 95% CI, and all three repeats improve p99. This is the first repeated
SGLang same-write-cost latency-improvement point, but it is partial
asynchronous restore, not final full restore policy evidence. Final restore
benefit now depends on paired
Host-visible/Host-invisible AegisKV policy measurements under matched
normal-path cost at larger scale, not on the earlier split-prefix correctness
gap.
The physical Host-restore evidence is now also machine-readable:
`sglang_physical_restore_evidence_summary.md` and
`results/physical_restore_evidence/physical_restore_runs.csv` parse the real
SGLang logs for `data_pages`, accepted prefetch pages, completed prefetch
tokens, and retry `#cached-token`. This turns the Host-restore chain into
structured evidence: `32998/33002` are hidden-frontier controls,
`32983/32987/32991/32999` are full materialization points, and
`33003/34334/34337` are partial best-effort asynchronous materialization
points. Jobs `34338/34344/34345` add a repeated physical CUDA copy calibration:
on two NV12-connected A100s, an 8192-token KV segment takes 48.595 ms H2D
(95% CI 0.531 ms), 46.416 ms D2H (95% CI 0.100 ms), and 4.368 ms
GPU0-to-GPU1 P2P (95% CI 0.004 ms) at p50 under the Qwen3-8B bf16 KV-size
assumption.
Jobs `34357/34358/34359` then reshape this into AegisKV backup/restore actions
that write into retry-rank KV buffers. At 8192 tokens, `host_restore_h2d` takes
48.434 ms p50 (95% CI 0.013 ms), while `gpu_replica_restore_p2p` takes 4.330 ms
p50 (95% CI 0.005 ms). This is stronger than raw copy calibration, but remains
an action-level microbenchmark rather than live end-to-end policy recovery.
Except for Fig. 10b, these data are not final HPCA evidence. Final
quantitative claims require replacing the remaining preliminary CSVs with
`evidence_tier=final` data from online AegisKV SGLSim runs with physical KV
movement and paired small-scale SGLang measurements.

## 证据概览

| Artifact class | Count | Current status | Final gate |
|---|---:|---|---|
| generated system overview | 1 | `design-ready` bitmap + prompt | visual/content review |
| quantitative figures | 24 | 1 strict-final measured, 23 preliminary / hybrid-ready | replace remaining preliminary rows with `evidence_tier=final` or explicitly keep them as motivation / calibration evidence |
| tables | 4 | design/evaluation scaffolding | align with final implementation |
| SGLang smoke | multiple | DP=2 Qwen3-8B Slurm smoke, paired baseline vs HiCache write-through overhead smoke, DP=2 failover-control smoke, remote-backup frontier smoke, materialized Host restore smoke, same-write-cost best-effort repeats, and CUDA H2D/D2H/P2P microbench | full AegisKV paired runs at larger scale with physical copy in the recovery path |
| SGLSim runs | multiple | local `synthesis_requests` smoke, 96-row policy-candidate run, hard-failover replay smoke, 5-seed hard-failover replay matrix, DP=2/4/8 direct scale sweep, DP=4 matched peer-route replay pair, Fig. 8 online policy sweep, Fig. 11 online no-failure overhead sweep, trace-derived Fig. 1 equal-size loss motivation, and strict-final-ready Fig. 10b shared-prefix sweep | online AegisKV policy matrix for remaining figures |

Current validator summary:

```text
figures=25
claim_ready=2
result_figures=24
result_figures_with_preliminary_data=23
result_figures_without_final_measured_data=23
hpca_hybrid_ready=25/25
hpca_hybrid_ready_result_figures=24/24
hpca_hybrid_blocked_result_figures=0/24
```

The hybrid evidence report is generated with:

```bash
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --write-hpca-hybrid
```

It marks Fig. 2, Fig. 10, and Fig. 14 as
`hybrid-physical-calibrated`: the large-scale rows are still SGLSim
metadata-level simulation, but their recovery timing is anchored by the
recovery-shaped physical CUDA movement jobs `34357/34358/34359`.

The hard submission gate remains:

```bash
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-final
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-hpca-hybrid
```

The per-figure path from preliminary artifact to final evidence is tracked in
`experiments/final_evidence_matrix.csv`.

## 结果叙事

### 动机：价值不等于字节大小

Fig. 1 holds lost KV size fixed and varies the lost segment's workflow role.
The final CSV injects the same 512-token loss into 36 observed online SGLSim
workflow timelines from `codex-shared-prefix-sweep-r3-20260628`. Losing a
terminated branch is nearly free (0.001 normalized WCT delta), running subagent
loss is mostly hidden by parallel slack (0.005), and critical positions become
progressively more expensive: last subagent before fan-in 0.020, main resume
0.037, synthesis tail 0.051, and shared prefix 0.147. The claim is that the
right optimization target is **workflow completion time impact per recoverable
token**, not copied bytes or raw reuse count.

### 恢复动作存在下包络

Fig. 2 compares recompute, Host restore, and GPUReplica restore as segment
length changes. The current curves are calibrated from the same hard-failure
replay rows used by Figs. 9-10: recompute rows fit `4.0 + 0.0108 * tokens` ms,
Host rows fit `7.5 + 0.0036 * tokens` ms, and GPUReplica rows fit a peer-tail
restore plus suffix recompute model with 34.2% observed peer coverage. Recompute
is competitive for tiny suffixes or non-peer retries because it avoids restore
setup overhead. GPUReplica is fastest for small hot tails when peer pressure is
low. Host restore becomes attractive for long prefixes where H2D is cheaper
than replay. AegisKV should therefore choose a recovery action per segment
rather than treating every backup hit as "restore all." The first physical
copy calibration aggregate, SGLang small-scale jobs `34338/34344/34345`,
measures real CUDA KV-sized movement on two NV12-connected A100s: for an
8192-token segment under the 147,456 bytes/token Qwen3-8B bf16 assumption,
p50 latency is 48.595 ms for H2D, 46.416 ms for D2H, and 4.368 ms for
GPU0-to-GPU1 P2P.

### 静态 GPUReplica budget 很脆弱

Fig. 3 sweeps the GPUReplica HBM budget fraction. Different workload regimes
expose different failure-time visibility limits: hot-tail rows improve from
the budget=0 baseline and then saturate once the peer-visible tail is covered;
long-prefix rows are already better served by Host restore, so forcing a peer
split can add setup cost; fan-out/non-peer rows do not improve with HBM budget
alone because the retry rank has no peer-visible tokens. Current data is an
implemented DP4 online hard-restart sweep:
`sglsim-budget-sweep-agentic-r1-combined-20260628`, with 25 budget/timing
scenarios, 600 replay rows, and 0 failed rows.

Fig. 3b and Fig. 3c now make the adaptive-alpha path explicit. Fig. 3b is
generated from the offline value-iteration policy table
`aegiskv_alpha_mdp_policy.json`, whose state space is
`critical_pressure in {low, medium, high}` crossed with five
`failure_rate_per_window` values. The policy uses `gamma=0.92` and 50 value
iterations, and the selected `alpha` is nondecreasing as failure risk rises
within each pressure bin. The goal audit replays all 15 policy-table states
through the runtime function `sglsim.srt.simulation.alpha_policy.choose_alpha`
and verifies exact agreement with the JSON table and Fig. 3b CSV.

Fig. 3c then verifies that the same policy table is used during online serving.
The 1-3-1-3-1 sweep contains four online runs, low/high workload pressure, and
failure rates 0.003, 0.01, and 0.03. The raw traces contain 150 Critical
requests and 120 Non-Critical requests. Replaying `choose_alpha` over all
270 completed request rows exactly matches the recorded
`aegis_gpu_budget_fraction`. Critical requests receive the selected
GPUReplica alpha, while Non-Critical requests retain Host visibility but have
zero effective GPUReplica tokens. This is stronger than a plot-only claim: it
checks the offline policy JSON, the runtime selector, the measured CSV, and
the online `request_details.csv` rows as one chain.

### 系统路径：在 SGLang KV 语义上构建 policy

Figs. 4-7b define the system path. AegisKV is not a replacement KV manager. It
adds workflow metadata, HostBackup visibility, peer placement, and safe-frontier
selection on top of SGLang radix/HiCache lifecycle events. The correctness
figures emphasize that pending D2H is not safe, `lock_ref > 0` prevents
eviction, ACK creates the recoverable frontier, and Host/peer eviction must
invalidate recovery choices. Current unit evidence is recorded in
`experiments/correctness_evidence.md` and covers the SGLSim/SRT metadata
invariants behind these states; real SGLang failure/copy validation remains
future final evidence.

### 整体 WCT 与恢复行为

Fig. 8 now uses an implemented online SGLSim policy sweep rather than the older
replay counterfactual. The active run is
`sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`: DP4, 32 agents,
3 turns, matched seeds `8100/8101/8102`, and four explicit
`gsp_aegis_policy` settings. Each policy completes 360/360 rows with 72
hard-failure replay rows and 0 failures. Normalized mean end-to-end WCT is
1.000 for no backup, 0.982 for uniform Host, 0.982 for uniform GPUReplica, and
0.976 for AegisKV. Fig. 9 isolates recovery overhead using the same online
policy sweep: AegisKV is 0.780x no-backup recompute on mean retry latency and
0.915x on critical-agent p90 latency, compared with 0.793x/0.908x for uniform
Host and 0.829x/0.935x for uniform GPUReplica. Fig. 10 explains the replay path
accounting across the retained multi-regime counterfactuals: AegisKV combines
9.7% peer GPU, 33.3% Host H2D, and
56.9% suffix recompute under the current equal-regime mix. Fig. 10 also
includes repeated online action rows from
`sglsim-physical-recovery-action-agentic-r3-20260628`, which uses the
recovery-shaped physical movement aggregate for restore timing. Across 12
trials and 288 hard-failover replay rows, all requests complete with no
failures. The DP2 Host and DP4 peer long-prefix scenarios split between Host restore and
small peer-GPU hot-tail restore; the DP4 non-peer scenario is 100% Host
restore; the DP4 peer hot-tail scenario chooses `gpu_replica` for 72/72 replay
rows and splits into 19.2% peer-GPU restored tokens plus 80.8% suffix
recompute. This is still not final physical-movement evidence:
`sglsim-live-movement-audit-smoke-20260628` records 4/4 restore rows with
`aegis_replay_restore_data_plane=metadata_materialization` and
`aegis_replay_restore_timing_source=physical_recovery_path_csv`. Strict live
movement validation now requires `physical_tensor_copy` rows, but that should
be collected through a SGLang/torch real-system path rather than by adding CUDA
movement to the metadata-only SGLSim runtime.

### 避免重复保护

Fig. 10b evaluates shared-prefix marginal accounting. When many agents share a
long prompt prefix, naive Host/GPUReplica accounting can count the same
safe-frontier content multiple times. AegisKV's marginal accounting protects
only newly extended frontier tokens within the same agent. The final Fig. 10b
CSV comes from `codex-shared-prefix-sweep-r3-20260628`: nine online SGLSim
runs across low/mid/high shared-prefix scenarios, 141 multi-turn agent groups,
and shared-prefix bins from 2.5% to 77.5%. The derivation excludes retry rows,
so retry cache-hit counters do not define the shared-prefix ratio.

### Normal-path overhead 与 Host visibility

Fig. 11 separates no-failure overhead from failure recovery benefit. The paper
must compare AegisKV against a fixed write-through baseline that pays the same
D2H path. Fig. 11b captures the key distinction: Host-invisible baseline pays
D2H but cannot use failed-rank Host KV after a GPU failure, while AegisKV adds
cross-rank HostBackup directory visibility and can restore ACKed frontiers.
The current Fig. 11b retry-latency axis is now tied to direct SGLSim replay
counterfactuals over recoverable regimes: Host-visible long prefixes and
peer-visible hot tails. Its overhead axis remains a preliminary fixed
write-through / peer-copy anchor.
The matched SGLang DP=2 overhead smokes on `hkbugpumanode` support this
direction. Job `32861` measured 8/8 successful requests in both baseline and
HiCache write-through cases, with 2.02% p99 overhead and 3.05% mean overhead.
Job `32863` increased the workload to 32 requests and 128 max tokens per case;
it measured 1.18% mean overhead, while p99 was -3.93% due small-sample tail
noise. Together these points support a bounded-overhead direction, not a final
precise p99 claim.
Jobs `32977` and `32978` add real SGLang failure-control-path smokes. In both
runs, 4 routed in-flight requests on DP rank 0 were captured by
`/simulate_gpu_failure`, re-dispatched to the surviving rank, completed
successfully, and returned `sglext.failover`. Job `32978` triggers failure
later and captures 760 pre-failover output tokens, but it still records 0
pre-failover backed-up tokens because no storage backend is attached. Job
`32980` attaches the `remote_backup` storage backend and reports 5,132
pre-failover backed-up tokens across the same four-request shape. That is a
real Host-visible frontier smoke. The derived
`failure_visibility_pair_32978_32980` artifact makes the comparison explicit:
job `32978` has 0 backed-up tokens, job `32980` has 1,283 backed-up
tokens/request, and the remote-backup smoke is 1.108x the no-storage p99
latency. Job `32981` shows that disabling request-aware remote-backup retention
does not fix the miss: four-request retries still query from page 0 while the
backed leaf data begins at page 14. Job `32982` removes that split-prefix
shape with a single request and validates the next step in the chain:
`data_pages=1297`, `Prefetch ACCEPTED`, and remote `prefetch_get` hits after
failover. Job `32983` then switches that single-request case to
`wait_complete`, producing `Prefetch IO END ... completed=1297 / expected=1297`
and retry prefill `#cached-token: 1297`. Job `32984` supplies a no-storage
single-request control for that case; the pair records 1.124x visible/no-storage
p99 latency, so it is correctness/calibration evidence rather than a latency
win. Job `32985` shows that four concurrent retried requests still have
navigable remote radix state but `data_pages=0`, so final Fig. 11b still needs
split-prefix-aware restore for shared-prefix prompts. Job `32987` proves that
four concurrent requests do materialize when the prompt shape starts from
page 0 per request, and job `32989` supplies the no-storage control. Job
`32991` then fixes the shared-prefix storage layout and materializes all four
shared-prefix retries with `data_pages=1297`, `completed=1297 / expected=1297`,
and retry `#cached-token: 1297`. The `failure_visibility_pair_32992_32991`
artifact records a 5,188-token Host frontier delta and a 1.210x
visible/no-storage p99 latency ratio. Jobs `32993`/`32994` extend the same
Host-frontier calibration to a longer shared prefix: the visible run backs up
20,548 tokens and materializes `#cached-token: 5137` on all four retries, but
its p99 remains 1.630x the no-storage control. Jobs `32998`/`32999` add a
fixed-write-cost pair: both write 20,548 tokens, but only the visible run is
allowed to use the storage frontier during retry, materializing
`#cached-token: 5137`; p99 remains 1.487x the lookup-disabled control. Jobs
`33002`/`33003`, `34333`/`34334`, and `34336`/`34337` show the corresponding
`best_effort` tradeoff: partial restore improves p99 in all three repeats and
aggregates to 0.961x visible/control p99 with 0.016 95% CI, but does not guarantee complete
materialization. The physical-restore parser records these same states as
structured rows, including full materialization in `32999`, partial
materialization in `33003/34334/34337`, and hidden-frontier controls in
`32998/33002/34333/34336`. Final Fig. 11b still needs a
paired AegisKV policy comparison that demonstrates restore-latency benefit
under matched normal-path cost and larger workloads.

### Ablation、sensitivity、scale 与 calibration

Fig. 12 removes AegisKV signals one at a time using implemented online toggles.
The DP4 hot-tail sweep runs matched seeds for full AegisKV, no lifecycle, no
segment granularity, no restore/recompute choice, no pressure signal, no
hysteresis, and no critical-agent restore. AegisKV is normalized to 1.0, all
disabled-signal variants regress, and the request rows expose the actual
restore actions selected by the retry path. Fig. 13 now uses an implemented
online failure-timing sweep over early/mid/late decode failures with the same
GPUReplica budget points as Fig. 3. Fig. 14
now uses a direct SGLSim
hard-failover replay DP-size smoke at DP=2, DP=4, and DP=8. Under the current
low-concurrency workload,
duration is nearly flat across DP sizes; this validates the serving path across
larger DP topologies but does not yet prove high-concurrency scale benefit.
Fig. 14b now uses an implemented DP=4 online peer-route sweep with explicit
retry route policies over the same short-tail workload and matched seeds
`8500/8501/8502`. The non-peer route has 0/72 peer hits and 72 recompute
actions; deterministic random healthy routing has 21/72 peer hits, 21
GPUReplica actions, and 51 recompute actions; AegisKV peer routing has 72/72
peer hits and 72 GPUReplica actions. AegisKV peer routing normalizes to 0.988x
WCT and cuts p99 recovery overhead to 74.3% of the non-peer recompute route.
This supports retry-locality in peer selection but is still not a final
physical P2P KV-copy policy sweep. Fig. 15 now
uses the defensible calibration anchor available today: real SGLang
normal-path overhead from the seven matched DP=2 paired runs versus the online
SGLSim no-failure AegisKV overhead sweep. Both anchors are in the noise range;
the final acceptable claim remains ranking preservation between SGLang and
SGLSim, not exact latency equality.

The current SGLSim preliminary path has progressed beyond a 96-row
policy-candidate smoke. The current r5 hard-failover replay matrix completed
5 seeds, 600/600 request rows, and 120 hard-failure cross-rank HTTP retry
replay rows.
All replay rows hit HostBackup and peer GPUReplica runtime directories,
restored 201,787 HostBackup tokens, materialized 198,322 prompt-prefix tokens
as target-rank metadata cache, recomputed 0 tokens, and observed 13.62 ms mean
restore sleep. The active figure CSVs aggregate the five run-level candidate
derivations as mean/std/95% CI. This is useful for debugging figure schemas,
policy equations, and paper layout; it still does not satisfy the final online
policy evidence gate.

The restored agentic workflow wrapper now also has a repeated online suite:
`sglsim-agentic-online-r3-20260621` runs DP2 shared-prefix normal path, DP2
HostBackup hard-failover, DP4 peer, and DP4 non-peer scenarios for three trials
each. The suite completes all rows with no failures: 288/288 normal-path rows
and 360/360 rows per hard-failover scenario. The newer
`sglsim-physical-recovery-action-agentic-r3-20260628` suite reruns the Host,
peer, non-peer, and hot-tail peer scenarios with the repeated recovery-shaped
physical movement aggregate driving restore timing. It completes 1,440/1,440
rows and 288/288 hard replay rows. Mean restore sleep is 17.148 ms for DP2
Host, 16.914 ms for DP4 peer, 17.343 ms for DP4 non-peer, and 5.005 ms for
DP4 peer hot-tail. Fig. 10 now uses these recovery-action-calibrated online rows for action coverage, and Fig.
14 consumes the Host/peer/non-peer rows as two preliminary online workflow scale
series. They remain `sglsim-agentic-online-preliminary` until final policy
sweeps replace the current metadata-only rows. The audit script
`audit_live_physical_movement.py` now makes this boundary machine-checkable.

Fig. 14 is the current exception to the r5-derived policy-candidate path. It
uses a direct DP-size sweep: DP=2 reuses the 5-seed r5 matrix, while DP=4 and
DP=8 add 3 seeds each. Across those 11 serving runs, 1,320/1,320 rows complete,
264 hard-failure replay rows execute cross-rank, and normalized duration is
1.000 at DP=2, 0.985 at DP=4, and 0.983 at DP=8. This is a direct scale smoke,
not final high-concurrency scale evidence.

Fig. 14b is also direct online SGLSim evidence rather than a policy-model row.
`sglsim-peer-route-sweep-agentic-r1-dp4-matched-seed-20260628` executes three
retry-route policies: forced non-peer, deterministic random healthy rank, and
recorded-peer routing. The figure plots the observed route hit/miss behavior
from those online replay rows, including the random route's measured 29.2%
peer-hit rate.

Fig. 15 is no longer a synthetic SGLang-vs-SGLSim ranking scatter. It now
compares current normal-path p99 overhead anchors: baseline is 0.00% in both
systems, while the protected path is 0.31% for SGLang HiCache write-through and
-0.49% for the online SGLSim AegisKV no-failure sweep, with 0.93% 95% CI on
the SGLSim point. The useful claim is that current normal-path overhead is
within noise in both systems; final ranking calibration still needs matched
SGLang/SGLSim policy workloads.

## HPCA 风格图注

**Fig. 1. Workflow penalty of equal-size KV loss.** Losing the same number of
KV tokens has very different impact depending on where the segment sits in the
agent workflow. The plot motivates segment-level value modeling instead of
byte-level backup rules. Current data: final SGLSim online equal-size loss
injection over 36 workflow groups, using a fixed 512-token loss.

**Fig. 2. Restore-vs-recompute boundary.** Recompute, Host restore, and
GPUReplica restore dominate in different segment-size regimes. AegisKV follows
the lower envelope rather than always restoring a backup hit. Current data:
SGLSim runtime replay-calibrated curves from 264 hard-failure replay rows,
plus repeated SGLang physical H2D/P2P CUDA-copy calibration curves from jobs
`34338/34344/34345`.

**Fig. 3. Static GPUReplica budget gap.** A single fixed peer-GPU backup budget
does not match all workloads. Static peer HBM helps only when protected tokens
are visible at the retry rank; otherwise Host restore or recompute still wins.
Current data: SGLSim runtime replay-calibrated counterfactual from 264
hard-failure replay rows.

**Fig. 4. AegisKV system overview.** Workflow metadata is attached to SGLang
requests, propagated to radix segments, and used to choose HostBackup
visibility, GPUReplica placement, and failure-time restore/recompute actions.
Current artifact: design-ready.

**Fig. 5. Backup data path.** HostBackup and GPUReplica have different
latency, capacity, queueing, and failure-domain semantics. Current artifact:
design-ready.

**Fig. 5b. Fixed HostBackup visibility pipeline.** Baseline and AegisKV both
pay fixed write-through D2H. AegisKV's Host benefit comes from publishing
ACKed failed-rank Host frontiers to retry ranks. Current artifact:
design-ready.

**Fig. 6. Online decision loop.** The controller observes workflow/KV/resource
state, chooses placement, waits for physical ACKs, publishes safe frontiers,
and uses them during failure recovery. Current artifact: design-ready.

**Fig. 7. Failure recovery timeline.** AegisKV shortens retry by using ACKed
frontiers and recomputing only the unsafe suffix. Current artifact:
design-ready.

**Fig. 7b. Radix lock, ACK, and eviction states.** Safe frontiers respect
SGLang radix correctness: locked nodes are not evicted, pending copies are not
safe, and Host/peer eviction invalidates recovery coverage. Current artifact:
design-ready, with matching unit-level metadata checks in
`test/unit/srt/test_aegis_restore_lookup.py`.

**Fig. 8. Overall workflow completion time.** Current data comes from direct
multi-regime replay counterfactuals: 264 hard-failure replay rows and 230,327
recovery tokens. AegisKV is 0.972x the same-row no-backup recompute
counterfactual on this low-concurrency smoke workload, compared with 0.993x
for uniform Host and 0.984x for uniform GPU.

**Fig. 9. Recovery latency and critical-agent latency.** Current data comes
from the implemented online SGLSim policy sweep
`sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`. Each plotted row
aggregates three matched-seed DP4 hard-restart trials and 72 replay rows. AegisKV
is 0.780x no-backup recompute on mean retry latency and 0.915x on
critical-agent p90 latency; uniform Host is 0.793x/0.908x and uniform GPUReplica
is 0.829x/0.935x.

**Fig. 10. Recovery path breakdown.** Current data comes from direct
multi-regime replay runtime counters and counterfactual path shares over the
same 264 hard-failure replay rows. The equal-regime AegisKV mix is 9.7% peer
GPU, 33.3% Host H2D, 0.0% ordinary cache, and 56.9% recompute. The appended
online rows now come from `sglsim-physical-recovery-action-agentic-r3-20260628`: DP2 Host
and DP4 peer long-prefix trials split mostly to Host restore with small
peer-GPU hot-tail shares, DP4 non-peer chooses Host restore, and DP4 peer
hot-tail chooses `gpu_replica` for all replay rows. Jobs `34357/34358/34359`
provide the recovery-shaped physical Host/GPUReplica restore anchor shown in
the plot annotation, while jobs `34338/34344/34345` remain the raw
H2D/D2H/P2P calibration. Final data still needs the live recovery path to
perform physical CUDA KV movement during end-to-end policy runs.

**Fig. 10b. Shared-prefix marginal accounting.** AegisKV avoids duplicated
protection of shared prefixes by scoring marginal safe-frontier coverage.
Current data: final SGLSim online shared-prefix sweep
`codex-shared-prefix-sweep-r3-20260628`, with nine runs, 141 multi-turn agent
groups, and observed shared-prefix bins spanning 2.5% to 77.5%.

**Fig. 11. Normal-path overhead.** AegisKV keeps no-failure overhead bounded by
sharing the fixed write-through Host path and limiting extra peer copies.
Current data: online SGLSim no-failure baseline/AegisKV sweep
`codex-normal-overhead-online-r3-20260628` gives -0.49% p99 overhead with
0.93% 95% CI and 1.28x write amplification, plus SGLang paired DP=2 smoke
provenance from jobs `32861`, `32863`, `32869`, `32870`, `32871`, `32874`,
and `32899` (mean overhead average 0.22%, p95 overhead average -0.19%, p99
overhead average 0.31%; all remain smoke-scale evidence).

**Fig. 11b. HostBackup visibility under fixed write-through.** Cross-rank
HostBackup visibility converts already-paid D2H copies into failure recovery
coverage. Current retry-latency data: SGLSim replay counterfactual over 192
recoverable Host/GPU rows; overhead axis remains a preliminary smoke/model
anchor. SGLang physical Host-restore parser rows now separately show hidden
frontier, full wait-complete materialization, and repeated partial best-effort
materialization: 11 parsed real-SGLang rows contain 7 materialized rows, 4 full
wait-complete materialized rows, 3 partial best-effort materialized rows, and
4 hidden-control rows. The current same-write best-effort aggregate is 0.961x
p99 with 0.016 95% CI. Final peer GPUReplica/P2P policy evidence is still
missing.

**Fig. 12. Ablation study.** Removing lifecycle awareness, segment-level
accounting, restore/recompute choice, pressure signals, hysteresis, or
critical-agent restore worsens WCT. Current data:
`sglsim-ablation-sweep-agentic-r4-dp4-hot-tail-matched-seed-20260628`, an
implemented online SGLSim ablation sweep with 7 variants, 21 trials, 0 failed
rows, and 504 replay rows. Strict-final evidence still requires live physical
KV movement rather than metadata materialization.

**Fig. 13. Budget sensitivity.** The best GPUReplica budget depends on what is
visible at retry time. Current data:
`sglsim-budget-sweep-agentic-r1-combined-20260628`, an implemented online DP4
hard-restart budget/timing sweep. Early decode improves from `1.000` to
`0.982`, mid decode to `0.986`, and late decode to `0.981` under the measured
r1 sweep. Strict-final evidence still needs live physical KV movement rather
than metadata materialization.

**Fig. 14. DP scale and peer availability.** The current direct SGLSim sweep
runs hard-failover replay at DP=2, DP=4, and DP=8. It shows stable completion
and cross-rank replay across larger DP topologies under a low-concurrency
workload; high-concurrency scale benefit remains final evidence. Current data:
SGLSim runtime replay-derived direct DP sweep.

**Fig. 14b. Peer-route selection.** In an online DP=4 short-tail route sweep,
routing to the recorded peer rank enables GPUReplica and reduces p99 recovery
overhead relative to a non-peer recompute route. Current data:
`sglsim-agentic-online-peer-route-sweep`; final data still needs physical peer
KV movement.

**Fig. 15. SGLang/SGLSim calibration anchor.** Current data compare the real
SGLang HiCache write-through normal-path p99 overhead aggregate, now 0.31%
across seven matched DP=2 jobs, with the online SGLSim AegisKV no-failure
overhead anchor, now -0.49% with 0.93% 95% CI. The plot is an overhead anchor,
not a final ranking-preservation result.

**Fig. 15f. Real SGLang Host restore materialization.** Current data summarize
the structured `physical_restore_runs.csv` parser output from real SGLang
server logs and failure summaries. The 11 rows include 4 hidden-control rows,
3 partial best-effort materialization rows, and 4 full wait-complete
materialization rows. Materialized rows account for 39,191 retry cached tokens
after accept. This is real Host restore data-path evidence; it does not close
the P0 large-sweep live `physical_tensor_copy` gate or validate peer
GPUReplica/P2P policy recovery.

## 最终数据到位后的变化

Final data should not change the structure of this section. It should replace
the evidence tier:

```text
sglsim-preliminary -> final
```

and update the numerical claims in the captions and result narrative. If final
data contradicts any preliminary trend, the paper must follow the final
measurements and either revise the claim or demote the figure to a diagnostic.


---

# 08. 图表与预期结果

本节将论文主线映射到图表。结构参考 `One Pool - Two Caches`：先证明双重压力和动态最优点，再展示系统设计，最后展示整体结果、消融、敏感性和 scale。

已生成资产：

- 图表预览：[`../../figures/gallery.md`](../../figures/gallery.md)
- 图表 manifest：[`../../figures/manifest.json`](../../figures/manifest.json)
- 可编辑机制图：[`../../figures/src/`](../../figures/src/)
- Markdown 预览图：[`../../figures/png/`](../../figures/png/)

当前证据状态：

- Fig. 4, Fig. 5, Fig. 5b, Fig. 6, Fig. 7, and Fig. 7b are design-ready mechanism diagrams.
- Quantitative result plots now use preliminary SGLSim CSVs from the 5-seed
  hard-failover replay matrix, direct replay-derived counterfactuals, a
  DP=2/4/8 scale sweep, and a matched DP=4 peer-route replay pair. They are
  useful for paper layout, story validation, and early internal review, but
  remain below final evidence.
- Small-scale SGLang smoke runs on `hkbugpumanode` now validate the real
  serving path and matched Fig. 11 overhead datapoints. Jobs `32861`, `32863`,
  `32869`, `32870`, `32871`, `32874`, and `32899` form the current matched
  DP=2 Qwen3-8B no-failure overhead aggregate: HiCache write-through mean
  overhead averages 0.22% with range -2.23% to 3.05%, p95 overhead averages
  -0.19% with range -4.09% to 2.02%, and p99 overhead averages 0.31% with
  range -3.93% to 3.74%.
- SGLang failure-control-path smoke jobs `32977` and `32978` now validate the
  basic real SGLang DP=2 failover path: 4/4 routed in-flight requests survived
  `/simulate_gpu_failure` in both runs, 4 requests were captured and
  re-dispatched, all 4 completed responses carried `sglext.failover`, and
  `/simulate_gpu_recovery` emitted `gpu_recovered`. Job `32978` captured 760
  pre-failover output tokens, but both smokes still had 0 pre-failover
  backed-up tokens. This is retry-continuity evidence, not final Host-visible
  or peer-copy evidence.
- SGLang storage-backed failure smoke job `32980` enables
  `hicache_storage_backend=remote_backup` with a centralized remote backup
  server. It again completes 4/4 failover-retried requests and reports 5,132
  pre-failover backed-up tokens, making it the first real SGLang Host-visible
  frontier smoke. It remains below final evidence because it is not a paired
  Host-visible/Host-invisible policy comparison.
- The posthoc paired artifact `failure_visibility_pair_32978_32980` now ties
  the no-storage job `32978` to the remote-backup job `32980`: frontier tokens
  increase from 0 to 5,132 backed-up tokens, or 1,283 backed-up tokens/request,
  while p99 latency is 1.108x the no-storage control path. This is supporting
  evidence for Fig. 11b/Fig. 15, not a final policy-latency point.
- Follow-up SGLang probes narrow the restore gap. Job `32981` disables
  request-aware remote-backup retention and still observes `data_pages=0` on
  the retry path, ruling out lease GC as the primary miss cause. Job `32982`
  uses a single request and validates a real storage match/fetch:
  `data_pages=1297`, `Prefetch ACCEPTED`, and `prefetch_get` hits. Job `32983`
  then runs the same page-0 single-request shape with
  `hicache_storage_prefetch_policy=wait_complete`, completing prefetch IO with
  `completed=1297 / expected=1297` and admitting retry prefill with
  `#cached-token: 1297`. Job `32984` adds the matching no-storage
  single-request control; `failure_visibility_pair_32984_32983` records a
  1,297-token Host frontier delta and 1.124x visible/no-storage p99 latency.
  Job `32985` shows that the four-request shape still has `data_pages=0` even
  with `wait_complete`, so shared-prefix split restore remains open. Job
  `32987` then enables `UNIQUE_REQUEST_PREFIX=1` and proves that four
  concurrent retries can materialize remote-backup pages when each request has
  page-0 backed data: all four requests reach `data_pages=1301`,
  `completed=1301 / expected=1301`, and retry `#cached-token: 1301`. Job
  `32989` supplies the no-storage unique-prefix control; the paired artifact
  `failure_visibility_pair_32989_32987` records a 5,204-token Host frontier
  delta and 1.308x visible/no-storage p99 latency. Job `32991` applies the
  split-prefix storage fix and reruns the shared-prefix four-request shape:
  all four retry queries reach `data_pages=1297`, all four prefetch IOs
  complete `1297 / 1297`, and each retry prefill admits with
  `#cached-token: 1297`. The paired artifact `failure_visibility_pair_32992_32991`
  records a 5,188-token Host frontier delta and 1.210x visible/no-storage p99
  latency. Jobs `32993`/`32994` repeat the visibility pair with a longer shared
  prefix, extending the backed frontier to 20,548 tokens and materializing
  `#cached-token: 5137` on all four retries, but increasing p99 to 1.630x the
  no-storage control. Jobs `32998`/`32999` isolate the same long-prefix
  mechanism under matched normal write cost: both runs write 20,548 tokens,
  while the lookup-disabled run hides the Host frontier from retry and the
  lookup-enabled run materializes `#cached-token: 5137`; the visible path is
  still 1.487x slower. Jobs `33002`/`33003` rerun the same matched-write-cost
  shape with `best_effort`: all retry queries find `data_pages=5137`, one
  prefetch partially completes `2816 / 5137`, retry prefill observes
  `#cached-token: 2830`, and visible p99 is 0.966x the lookup-disabled control.
  These remain below final evidence because they are
  diagnostic smokes, not repeated paired AegisKV policy comparisons at larger
  scale.
- Large-scale SGLSim online candidate run `20260621_032719` completed 96/96
  request turns for 32 shared-prefix agents. It now emits Aegis frontier and
  candidate failover columns in `request_details.csv`: 24 critical-agent retry
  candidates, 100% Host-backed frontier at the candidate failure point, and
  13.45 ms mean candidate failover overhead. This is stronger than static
  templates, but still below final failure-injected recovery evidence.
- SGLSim failover replay smoke `20260621_042727` additionally sent 24 real HTTP
  retry replay requests after protected critical turns. It completed 120/120
  request rows with 24 rows marked `aegis_failover_replay=true` and 24/24
  replay rows dispatched cross-rank (`failed_source_dp_rank != assigned_dp_rank`).
  All 24 replay rows were bracketed by SGLSim hard failure control-plane calls:
  the source scheduler subprocess was terminated before replay and restarted
  after replay, and the rows are marked `aegis_hard_failure_injected=true`.
  All 24 replay rows also hit the runtime HostBackup directory
  (`aegis_host_backup_lookup_hit=true`), restoring 39,676 tokens from the
  directory lookup and recomputing 0 tokens. Of those protected tokens, 38,989
  prompt-prefix tokens were materialized as target-rank device-cache metadata
  before replay admission and appear as replay-row `cache_hit_tokens`.
  All 24 replay rows also found peer GPUReplica directory entries
  (`aegis_peer_gpu_lookup_hit=true`, 5,036 peer tokens); for this long-prefix
  workload the runtime lower-envelope still chose Host restore for all replay
  rows because peer GPUReplica would require recomputing most of the prefix.
  The scheduler slept for the replay restore plan with 13.51 ms mean observed
  restore sleep, so restore cost is now visible in online wall-clock latency.
  This still does not execute real CUDA tensor KV copy; it is a metadata page
  movement inside SGLSim.
- SGLSim failover replay matrix `sglsim-failover-replay-matrix-r5-20260621`
  ran 5 seeds and completed 600/600 rows with 120 hard-failure cross-rank
  replay rows. All 120 replay rows hit HostBackup and peer GPUReplica
  directories, restored 201,787 HostBackup tokens, materialized 198,322
  prompt-prefix tokens as target-rank cache metadata, recomputed 0 tokens, and
  observed 13.62 ms mean replay restore sleep. The current SGLSim measured CSV
  rows are derived from this r5 matrix as 5-run means with run-level std/95% CI
  columns; selected bar figures render those CI values.
- Fig. 14 now uses a direct SGLSim DP-size sweep instead of the r5-derived
  scale projection: DP=2 uses the r5 matrix, and DP=4/DP=8 add 3 seeds each.
  The sweep completed 1,320/1,320 rows with 264 hard-failure cross-rank replay
  rows. Normalized duration is 1.000 at DP=2, 0.985 at DP=4, and 0.983 at
  DP=8. This is a serving-path scale smoke, not final high-concurrency scale
  evidence.
- Fig. 14b now uses an implemented DP=4 online peer-route sweep with explicit
  retry route policies. The non-peer, random-healthy, and AegisKV peer-route
  scenarios share the same short-tail workload and seeds `8500/8501/8502`.
  AegisKV peer routing hits peer GPUReplica on 72/72 replay rows, while random
  healthy routing hits 21/72 and non-peer routing hits 0/72. AegisKV peer-route
  WCT is 0.988x the non-peer route and p99 recovery overhead is 74.3% of the
  non-peer recompute route.
- Agentic online suites `sglsim-agentic-online-r3-20260621`,
  `sglsim-agentic-peer-hot-tail-r3-20260621`, and
  `sglsim-physical-copy-agentic-r3-20260628` are retained as provenance. The
  current Fig. 10/Fig. 14 online rows are superseded by
  `sglsim-physical-recovery-action-agentic-r3-20260628`, which repeats the
  hard-failover scenarios with recovery-shaped Host/GPUReplica movement curves
  driving restore timing.
- Final HPCA claims still require replacing those preliminary CSVs with
  `evidence_tier=final` SGLang/SGLSim measurements.

## Fig. 1：同等大小 KV loss 动机

Question:

```text
Does equal-size KV loss have non-uniform WCT impact?
```

Plot:

- x-axis: segment position / lifecycle。
- y-axis: delta workflow completion time。
- bars: terminated branch, running subagent, last subagent, main resume, synthesis tail, shared prefix。

Expected result:

- terminated branch near zero；
- final/synthesis/main resume much larger；
- shared prefix causes replay amplification。

What it proves:

> AegisKV needs workflow-aware value; byte size alone is insufficient.

## Fig. 2：Restore 与 recompute 边界

Question:

```text
When should a retry restore KV and when should it recompute?
```

Plot:

- x-axis: segment tokens or KV bytes。
- y-axis: recovery time。
- lines: GPUReplica restore, Host restore, recompute。

Expected result:

- recompute wins for tiny suffixes when peer GPUReplica is unavailable or not
  worth placing；
- GPUReplica wins for small high-value hot tails if peer is not congested；
- Host wins for long prefixes when H2D is cheaper than replay；
- AegisKV tracks the lower envelope。

Current evidence:

- `fig_restore_recompute_boundary.csv` is now calibrated from direct SGLSim
  hard-failure replay rows, run id
  `sglsim-fig2-restore-boundary-calibrated-r11-20260621`.
- Inputs are the same three regimes used by Figs. 9-10: 120 Host rows,
  72 GPUReplica rows, and 72 recompute rows, for 264 replay rows total.
- Fitted runtime coefficients are `recompute = 4.0 + 0.0108 * tokens`,
  `Host = 7.5 + 0.0036 * tokens`, and `GPUReplica = 2.4 + 0.0019 *
  peer_tokens + 0.0108 * suffix_tokens`; observed peer-token coverage in the
  GPUReplica regime is 34.2%.
- This is SGLSim runtime replay-calibrated evidence with repeated SGLang
  raw H2D/P2P copy calibration from jobs `34338/34344/34345` and
  recovery-shaped Host/GPUReplica restore movement from jobs
  `34357/34358/34359` overlaid.
  Final Fig. 2 still needs the physical copy paths inside the end-to-end
  recovery flow.

What it proves:

> Backup hit is not enough; recovery action must be restore/recompute-aware.

## Fig. 3：静态 GPUReplica budget 差距

Question:

```text
Is one fixed GPUReplica budget enough on top of fixed HostBackup?
```

Plot:

- x-axis: GPUReplica budget fraction。
- y-axis: WCT or expected recovery cost。
- lines: different workloads/failure timings。

Expected result:

- hot-tail rows improve as peer-visible tails are covered, then saturate；
- long-prefix rows prefer Host restore over forced peer splitting；
- fan-out/non-peer rows do not improve from HBM budget without retry-rank
  visibility；
- dynamic AegisKV remains a single online choice rather than one static ratio。

Current preliminary input:

- `fig_static_gpu_replica_budget_gap.csv` now comes from implemented online
  SGLSim DP4 hard-restart budget scenarios, run id
  `sglsim-budget-sweep-agentic-r1-combined-20260628`；
- rows cover hot-tail, long-prefix, and fan-out/non-peer workloads at
  `0/0.1/0.2/0.5/1.0` GPUReplica budget fractions；
- the sweep completed 600 replay rows across Fig. 3 and Fig. 13 inputs with
  0 failed rows; Fig. 3 directly uses 360 static-curve replay rows plus a
  dynamic AegisKV summary row；
- this is implemented online policy evidence, but still metadata
  materialization rather than final physical KV-copy recovery。

What it proves:

> AegisKV is an online resource partitioning problem, not a static rule.

## Fig. 4：系统总览

Content:

```text
Workflow metadata -> SGLang request -> radix node Aegis metadata
-> placement planner -> GPUReplica / Host -> safe frontier
-> failure retry -> restore/recompute
```

What it proves:

> AegisKV is integrated with SGLang's KV lifecycle, not a separate KV manager.

## Fig. 5：备份数据路径

Content:

- source DP rank active KV；
- peer DP rank GPUReplica；
- same worker Host memory；
- retry rank；
- P2P / D2H / H2D arrows；
- ACK path。

What it proves:

> GPUReplica and Host have different latency, capacity, and fault-domain semantics.

## Fig. 5b：固定 HostBackup visibility pipeline

Content:

```text
GPU active KV
  -> radix insert
  -> fixed write_through D2H
  -> Host ACK
  -> HostBackup directory entry
  -> source GPU failure
  -> retry on another DP rank
  -> lookup failed-rank Host KV
  -> Host restore / suffix recompute

GPU active KV
  -> peer-copy queue
  -> peer GPUReplica ACK
```

Include queue boxes:

- D2H write queue；
- H2D load/prefetch queue；
- HostBackup directory；
- peer-copy queue；
- ACK queues。

Expected result if plotted with measurements:

- baseline and AegisKV pay the same fixed write-through D2H；
- baseline cannot recover because retry rank lacks failed-rank Host frontier visibility；
- AegisKV recovers ACKed Host prefixes and recomputes only missing suffixes。

What it proves:

> AegisKV's Host contribution is cross-rank recovery visibility over an existing fixed write-through Host cache, not a new D2H strategy.

## Fig. 6：在线决策循环

Borrowed structure from One Pool:

```text
observe state -> choose budget/placement -> physical copy -> ACK
-> update frontier -> failure-time restore -> observe next state
```

Include state vector table:

- workflow state；
- KV state；
- backup state；
- resource pressure；
- failure exposure；
- restore/recompute profile。

## Fig. 7：故障恢复时间线

Timeline:

```text
normal decode
backup copy
ACK
failure
snapshot
retry admission
frontier lookup
restore/recompute
suffix replay
resume workflow
```

Expected comparison:

- no backup has longest suffix replay；
- Host reduces prefix replay but pays H2D；
- GPUReplica fastest when it hits；
- AegisKV selects the best path by segment。

## Fig. 7b：Radix lock、ACK 与 eviction 状态机

Content:

```text
device-resident unlocked
  -> active request lock_ref > 0
  -> Host copy pending
  -> Host ACKed safe
  -> GPU evicted / Host-only
  -> H2D restored
  -> Host evicted / unsafe
```

Annotate:

- `inc_lock_ref` / `dec_lock_ref`；
- `writing_check` ACK；
- `evict` and `evict_host`；
- safe frontier valid/invalid transitions。

Expected result:

- pending copy is never marked safe；
- locked nodes are not evicted；
- Host-only frontier is recoverable only after ACK；
- Host eviction invalidates frontier。

What it proves:

> The system design respects SGLang radix correctness rather than treating KV backup as an external blob copy.

## Fig. 8：整体 workflow completion time

Policies:

```text
no backup
uniform Host
uniform GPUReplica
AegisKV
```

Expected result:

- AegisKV best or near-best across skewed workflows；
- uniform GPUReplica may be best for tiny high-value tails but poor under HBM pressure；
- uniform Host stable for long prefixes but can be slower on short segments；
- no backup is the same-row recompute baseline。

Current evidence:

- `fig_overall_wct.csv` now comes from an implemented online SGLSim policy
  sweep rather than a replay counterfactual. The active run id is
  `sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`.
- The sweep runs four policies over the same DP4, 32-agent, 3-turn workload
  with matched seeds `8100/8101/8102`. Each policy contributes 360 completed
  rows, 72 hard-failure replay rows, and 0 failed rows.
- Normalized mean end-to-end WCT is 1.000 for no backup, 0.982 for uniform
  Host, 0.982 for uniform GPUReplica, and 0.976 for AegisKV. The low
  end-to-end improvement is expected because this low-concurrency smoke
  workload is dominated by normal serving time rather than restore overhead.
- This is online SGLSim serving-path evidence with explicit
  `gsp_aegis_policy`, but it remains metadata-level. Final Fig. 8 still needs
  live physical tensor movement or paired real-system validation.

## Fig. 9：恢复延迟与 retry speedup

Metrics:

- critical-agent recovery latency；
- retry latency；
- speedup over no backup。

Expected result:

- GPUReplica has lowest hit latency；
- Host has moderate latency but better coverage；
- AegisKV combines them and improves tail。

Current evidence:

- `fig_recovery_latency.csv` is derived from the implemented online policy
  sweep `sglsim-policy-sweep-agentic-r6-dp4-matched-seed-20260628`.
- Each plotted policy uses the same DP4, 32-agent, 3-turn workload and matched
  seeds `8100/8101/8102`; each policy contributes 72 hard-restart replay rows
  and 0 failed rows.
- AegisKV is 0.780x no-backup recompute for mean retry latency and 0.915x on
  critical-agent p90 latency. Uniform Host is 0.793x/0.908x and uniform
  GPUReplica is 0.829x/0.935x.
- This is online SGLSim serving-path latency evidence with metadata-level
  restore materialization; strict-final Fig. 9 still needs physical CUDA KV-copy
  validation or paired real-system recovery validation.

## Fig. 10：恢复路径拆解

Stacked bar:

```text
peer_gpu
host_h2d
ordinary_cache
recompute
```

Expected result:

- AegisKV uses peer GPU selectively for high-value short segments；
- Host covers long prefix；
- recompute remains for low-value or unsafe-frontier segments。

Current evidence:

- `fig_recovery_path_breakdown.csv` is derived from the multi-regime replay
  counterfactual retained for Figs. 9-10. The active run id is
  `sglsim-fig8-10-policy-counterfactual-r11-20260621`.
- Across 264 replay rows and 230,327 runtime recovery tokens, AegisKV's
  equal-regime mix is 9.7% peer GPU, 33.3% Host H2D, 0.0% ordinary cache, and
  56.9% recompute. Uniform Host is 100% Host H2D; uniform GPU is 14.0% peer GPU
  and 86.0% suffix recompute under the same regime mix.
- `sglsim-physical-recovery-action-agentic-r3-20260628` appends four repeated online
  action-distribution rows with recovery-action-calibrated restore timing: DP2
  Host, DP4 peer, DP4 non-peer, and DP4 peer hot-tail. The first two
  long-prefix scenarios split mostly to Host restore with small peer-GPU
  hot-tail shares; DP4 non-peer chooses Host restore; DP4 peer hot-tail observes
  72/72 `gpu_replica` restore actions and yields 19.2% peer-GPU restored tokens
  plus 80.8% suffix recompute.
- This now exercises Host, peer, and recompute lower-envelope choices for all
  replay-calibrated policies, while separately documenting the current online
  agentic-suite action coverage. Jobs `34338/34344/34345` add repeated raw
  H2D/P2P/D2H copy calibration, and jobs `34357/34358/34359` add the
  recovery-shaped Host/GPUReplica restore annotation used by the plot. Final
  Fig. 10 still needs physical CUDA KV-copy inside end-to-end recovery policy
  runs before being relabeled as final measured evidence.

## Fig. 10b：Shared-prefix marginal accounting

Question:

```text
How much duplicate backup does AegisKV avoid when agents share prefixes?
```

Plot:

- x-axis: shared prefix ratio。
- y-axis left: duplicated protected bytes or write amplification。
- y-axis right: recovery coverage。
- bars/lines: no shared-prefix dedup vs marginal SafeCover accounting。

Expected result:

- no-dedup repeatedly protects each turn's safe frontier；
- marginal accounting protects only newly extended frontier tokens for the same
  agent；
- reconstruction coverage stays at the final frontier；
- benefit grows with observed shared-prefix overlap。

Current preliminary input:

- original request rows from the three SGLSim hard-failure matrices used by
  Figs. 2, 3, 8, 9, 10, and 12；
- retry rows are excluded to avoid using retry cache-hit tokens as the
  shared-prefix signal；
- current x-axis covers observed workload bins, not a full 0-95% synthetic
  sweep。

What it proves:

> AegisKV's two backup caches are coupled by shared KV content, unlike independent One Pool caches.

## Fig. 11：Normal-path overhead

Metrics:

- p95/p99 no-failure request latency；
- throughput；
- P2P copy BW；
- D2H/H2D BW；
- backup write amplification。

Expected result:

- AegisKV overhead bounded；
- uniform backup creates higher overhead；
- no backup has zero overhead but poor recovery。

Current evidence:

- `hkbugpumanode` jobs `32861`, `32863`, `32869`, `32870`, `32871`, `32874`,
  and `32899` are matched SGLang DP=2 overhead smokes. Together they show
  HiCache write-through mean overhead average 0.22%, p95 overhead average
  -0.19%, and p99 overhead average 0.31% over a no-HiCache baseline under
  matched CUDA
  graph controls. The per-run range crosses zero, so this is a "within noise"
  overhead anchor, not a precise positive tail-overhead claim.
- The current Fig. 11 measured CSV is therefore `mixed-preliminary`: SGLSim
  r3 failover-matrix policy-candidate rows are shown next to the SGLang paired
  aggregate rows so the paper layout exposes both the scale model and the
  real-system overhead anchor.

## Fig. 11b：固定 write-through 下的 HostBackup visibility

Question:

```text
If both systems already perform write_through D2H, what does AegisKV add?
```

Plot:

- grouped bars for `baseline_write_through_invisible`, `host_visible_no_gpu_replica`, `gpu_replica_only`, `AegisKV`。
- y-axis 1: no-failure p99 overhead relative to fixed-write-through baseline。
- y-axis 2: retry latency, recovered tokens, or WCT under failure。

Expected result:

- baseline pays D2H but cannot use failed-rank Host KV after retry；
- Host-visible recovery reduces recomputation for long prefixes；
- GPUReplica-only is fast when it hits but limited by peer HBM/locality；
- AegisKV combines Host visibility with selective GPUReplica。

Current preliminary input:

- retry-latency axis uses direct SGLSim hard-failure replay ratios against
  same-row no-backup recompute；
- Host visible uses long-prefix Host-visible rows, GPUReplica-only uses
  peer-visible hot-tail rows, and AegisKV aggregates both recoverable regimes；
- overhead axis is still a smoke/model anchor and needs paired final policy
  measurements。
- SGLang support artifact `failure_visibility_pair_32978_32980` compares job
  `32978` without a storage backend against job `32980` with `remote_backup`.
  It validates a 5,132-token Host frontier delta but also records 1.108x p99
  latency for the storage-backed smoke, so it should be cited as frontier
  visibility evidence rather than a recovery-latency win.
- Jobs `32981` and `32982` identify the next implementation boundary:
  split-prefix prompts can leave backed leaf data at `page_start=14`, causing a
  contiguous-prefix storage miss from a retry rank with no local prefix; a
  single-request page-0 shape reaches storage match/fetch. Job `32983` shows
  that `wait_complete` materializes those accepted pages before prefill
  admission (`#cached-token: 1297`). Job `32985` confirms that `wait_complete`
  alone does not fix the shared-prefix four-request case, job `32987` confirms
  that four-request materialization works when shared-prefix radix splitting is
  removed, and job `32991` confirms the patched split-prefix storage path
  materializes four shared-prefix retries with `#cached-token: 1297`. Jobs
  `32998`/`32999` add the fixed-write-cost visibility control: both runs
  perform the same remote-backup writes, but only the visible run can prefetch
  the Host frontier during retry, materializing `#cached-token: 5137`. Jobs
  `33002`/`33003` add the same control under `best_effort`, where the visible
  run improves p99 but only partially materializes the Host frontier.

What it proves:

> The HostBackup benefit comes from failure-aware metadata and restore protocol, not from changing the normal D2H policy.

## Fig. 12：Ablation

Ablations:

- no lifecycle；
- no segment granularity；
- no restore/recompute；
- no resource pressure；
- no hysteresis；
- no critical restore scheduling。

Expected result:

- no lifecycle and no critical restore regress because retry cannot use the
  recoverable frontier and falls back to recompute；
- no segment granularity and no pressure regress in the hot-tail workload
  because they suppress the peer GPUReplica path；
- no restore/recompute regresses because it forces Host restore even when peer
  GPUReplica is the lower-envelope action；
- no hysteresis regresses by intermittently losing a stable frontier across
  turns。

Current preliminary input:

- `sglsim-ablation-sweep-agentic-r4-dp4-hot-tail-matched-seed-20260628`；
- 7 implemented online ablation toggles, 3 matched seeds each, DP4, 32 agents,
  3 turns, 21 trials, 0 failed rows, 504 replay rows；
- AegisKV is normalized to `1.0`; every disabled variant is above the full
  policy in observed WCT；
- strict-final still requires live physical KV movement rather than SGLSim
  metadata materialization with calibrated timing。

## Fig. 13：Budget sensitivity heatmap

Axes:

```text
x = GPUReplica budget
y = failure timing
color = WCT or expected recovery cost
```

Expected result:

- per-regime optimum shifts with failure timing and overlap；
- more backup is not always better；
- high GPUReplica budget can hurt normal path。

Current preliminary input:

- `fig_budget_sensitivity.csv` now comes from the same implemented online
  SGLSim DP4 hard-restart sweep, run id
  `sglsim-budget-sweep-agentic-r1-combined-20260628`；
- rows cover early, mid, and late decode failure timing with
  `0/0.1/0.2/0.5/1.0` GPUReplica budget fractions；
- the heatmap directly aggregates 360 replay rows with 0 failed rows；
- this is hybrid-ready online sweep evidence, but strict-final still needs
  physical KV movement instead of metadata materialization。

## Fig. 14：DP scale 与 peer availability

Axes:

- DP size: 2, 4, 8；
- workflow count / concurrency；
- WCT or retry latency。

Expected result:

- the current preliminary direct sweep should complete at DP=2, DP=4, and DP=8
  without failed requests；
- duration should remain stable under low-concurrency hard-failover replay；
- final evidence must add high-concurrency workflow-count sweeps。

## Fig. 14b：Peer-rank selection

Question:

```text
How should GPUReplica choose the peer DP rank?
```

Plot:

- x-axis: retry route choice。
- bars: non-peer route miss, random healthy route, recorded AegisKV peer route。
- y-axis: normalized WCT under hard-failover replay。
- secondary metric: p99 recovery overhead as a percentage of the non-peer
  recompute route。

Expected result:

- non-peer route misses peer GPUReplica and falls back to recompute；
- random healthy routing recovers only with probability `1/(dp_size-1)`；
- retry-local peer routing gets the lower recovery overhead when peer tokens
  are visible。

Current preliminary input:

- run id: `sglsim-peer-route-sweep-agentic-r1-dp4-matched-seed-20260628`；
- scenarios: `peer_route_nonpeer_hot_tail`, `peer_route_random_hot_tail`, and
  `peer_route_aegiskv_hot_tail`；
- each scenario contributes 360 completed rows, 72 hard-failure replay rows,
  and 0 failed rows；
- route hit rates: non-peer 0/72, random healthy 21/72, AegisKV peer 72/72；
- current AegisKV peer route: 0.988x normalized WCT and 74.3% p99 recovery
  overhead relative to the non-peer recompute route。

What it proves:

> GPUReplica is not just a capacity allocation problem; retry-rank locality
> determines whether the protected peer tokens are usable.

## Fig. 15：SGLang/SGLSim calibration anchor

Plot:

- x-axis: SGLang p99 overhead anchor；
- y-axis: SGLSim p99 overhead anchor；
- labels: baseline, protected path。

Expected result:

- current anchor exposes whether SGLSim is optimistic or conservative on
  normal-path overhead；
- the 32978/32980 failure-visibility pair contributes a backed-up-token
  frontier calibration point, but not a ranking-preservation point；
- job 32983 contributes a single-request materialized restore calibration
  point after job 32982 isolated the storage match/fetch path；
- job 32985 contributes a four-request negative calibration point for the
  original split-prefix restore boundary；
- job 32987/32989 contributes a four-request unique-prefix materialized
  restore calibration pair；
- job 32991 and the 32992/32991 pair contribute a four-request shared-prefix
  materialized restore calibration pair after the split-prefix storage fix；
- job 32993/32994 extends Host-frontier calibration to a longer shared prefix,
  but the visible path is still slower than control；
- job 32998/32999 contributes a same-write-cost visibility isolation pair:
  both sides write 20,548 tokens, but only the visible side can consume the
  Host frontier during retry；
- job 33002/33003 contributes a same-write-cost `best_effort` latency point:
  partial Host restore improves p99 to 0.966x but does not prove full restore；
- final paired policy calibration should preserve relative ranking；
- exact latency equality is not required。

Prerequisite:

```text
SGLSim must add final visible AegisKV policy hooks and paired online workflow
runs before this figure can become a full policy-ranking calibration. Current
data only anchor normal-path overhead.
```

## Table 1：状态向量

Mirror One Pool's state-vector table, but for AegisKV:

| Symbol | Meaning |
|---|---|
| `h_t` | Host backup hit / ACK ratio |
| `g_t` | GPUReplica ACK ratio |
| `q_t` | retry / scheduler queue pressure |
| `p2p_t` | P2P bandwidth pressure |
| `h2d_t` | H2D restore pressure |
| `d2h_wt_t` | fixed write-through D2H pressure |
| `host_vis_t` | fraction of ACKed Host KV visible to retry ranks |
| `w_t` | workflow criticality skew |
| `l_t` | lifecycle distribution |
| `loc_{p,t}` | peer KV locality for candidate rank `p` |
| `load_{p,t}` | peer queue/load pressure |
| `dup_t` | duplicate Host/GPUReplica protected bytes |
| `alpha_gpu_t` | peer GPU backup budget ratio |
| `r_t` | recent recovery success / saved WCT |

## Table 2：Baseline

Show which signals each policy uses:

| Policy | Workflow | Lifecycle | Segment | Restore/recompute | Resource pressure |
|---|---|---|---|---|---|
| no backup | no | no | no | recompute | no |
| baseline write-through invisible | no | no | no | recompute | fixed D2H only |
| Host visible only | no | no | segment | restore/recompute | Host restore queue |
| uniform GPUReplica | no | no | segment | restore | peer budget only |
| request-level | weak | no | no | restore | coarse |
| agent-level | yes | weak | no | restore | coarse |
| static critical path | yes | weak | segment | restore | no |
| AegisKV | yes | yes | yes | yes | yes |

## Table 3：实现 hook

| Hook | SGLang path | Purpose |
|---|---|---|
| request metadata | tokenizer manager / `Req` | attach workflow state |
| radix match/insert/split | `HiRadixCache` / `RadixCache` | update segment metadata and shared SafeCover |
| fixed Host D2H | `HiRadixCache.write_backup` / `HiCacheController.write` | observe fixed write-through copy |
| Host ACK | `HiRadixCache.writing_check` / `ack_write_queue` | publish HostBackup directory entry |
| H2D/cross-rank restore | `HiCacheController.load` / restore protocol | recover failed-rank Host frontier on retry rank |
| GPU eviction | `HiRadixCache.evict` | preserve/invalidate GPU hit state |
| Host eviction | `evict_host` | invalidate Host frontier |
| peer ACK | peer copy manager / DP controller | mark GPUReplica frontier |
| peer demotion | peer replica manager | invalidate or downgrade GPUReplica frontier |
| failure retry | scheduler / DP controller | choose frontier and retry path |

## Table 4：正确性测试

| Test | Expected invariant | Current artifact |
|---|---|---|
| missing/pending HostBackup + failure | pending frontier rejected; retry recomputes | `test_aegis_restore_lookup_recomputes_when_host_backup_missing` |
| active `lock_ref` + eviction pressure | locked restored prefix is not evicted | `test_aegis_restored_locked_prefix_is_not_evicted_under_pressure` |
| divergent radix insert after restore | restored frontier token span still matches | `test_aegis_restored_prefix_survives_divergent_radix_insert` |
| unsupported pending restore action | no device KV is materialized before ACK | `test_aegis_pending_restore_action_does_not_materialize_device_cache` |
| Host unavailable + peer unavailable | unsafe frontier is not selected | `test_aegis_restore_lookup_recomputes_when_host_evicted_and_peer_unavailable` |
| peer demotion | GPUReplica frontier used only on peer rank | `test_aegis_restore_lookup_demotes_peer_gpu_when_retry_rank_is_not_peer` |
| request without Aegis metadata | no Aegis frontier or candidate retry is published | `test_aegis_baseline_request_without_metadata_does_not_publish_frontier` |

## 最小 paper-ready 结果包

The first complete paper should contain:

1. Fig. 1 motivation。
2. Fig. 2 restore/recompute boundary。
3. Fig. 4 system overview。
4. Fig. 7 failure timeline。
5. Fig. 8 overall WCT。
6. Fig. 10 recovery path breakdown。
7. Fig. 11 normal-path overhead。
8. Fig. 12 ablation。
9. Fig. 15 SGLang/SGLSim calibration。


---

# 09. 结论与下一步

## 结论

AegisKV should be framed as a workflow-aware, restore/recompute-aware KV backup layer for multi-agent LLM serving. Its novelty is not that it stores KV in another tier. In this simplified model, SGLang already performs fixed `write_through` D2H after radix insertion; AegisKV makes the failed rank's surviving Host KV visible to retry ranks, decides which workflow-critical radix segments deserve an additional GPUReplica, chooses the peer rank, and decides whether failure recovery should restore or recompute each suffix.

The paper's core sentence should be:

> In multi-agent LLM serving, the KV state worth protecting is not the largest, newest, or globally most reused state; it is the state whose loss most increases workflow completion time after accounting for restore-vs-recompute cost and backup interference.

## 近期下一步

1. Finalize the Host definition as same DP rank / worker local Host memory, not same-GPU duplicate.
2. Implement or verify SGLang metadata path:
   `request -> Req -> radix node -> Host/peer ACK -> failure retry`.
3. Implement fixed `write_through` HostBackup directory: ACKed failed-rank Host KV must be discoverable by retry ranks.
4. Implement peer-rank selection with load + KV locality + retry route + HBM/P2P pressure.
5. Add shared-prefix SafeCover accounting and frontier invalidation on Host/peer eviction.
6. Use the restored `benchmark/agentic_workflows` wrapper to run repeated online workflow scenarios, then replace its preliminary rows with final AegisKV policy evidence.
7. Generate the motivation equal-size-loss figure.
8. Generate restore-vs-recompute microbenchmarks for Host and GPUReplica.
9. Run the four-policy online suite:
   `none,host,peer_gpu,aegiskv`.
10. Compare SGLang and SGLSim on the same small workloads.
11. Only after those pass, run DP=8 / 30K-token SGLSim scale-up.

## 最终 HPCA 论文前必须补齐的证据

- Aegis metadata does not affect normal SGLang outputs.
- Host ACK creates a safe frontier and retry can restore from it.
- GPUReplica peer copy creates a safe frontier and retry can use it.
- GPUReplica peer selection beats random, least-loaded-only, and locality-only placement.
- Shared-prefix marginal accounting reduces duplicate protected bytes without reducing recovery coverage.
- Host-visible AegisKV beats Host-invisible fixed-write-through baseline under the same D2H cost.
- Radix lock/split/eviction tests reject unsafe frontiers.
- AegisKV beats static baselines on WCT under failures.
- AegisKV keeps no-failure overhead bounded.
- Restore-vs-recompute decision improves over restore-all and recompute-all.
- SGLSim preserves policy ranking relative to SGLang.

## 暂时不能声称的结论

Do not claim:

- AegisKV handles host failure. Host tier does not survive full host loss.
- AegisKV is a generic RemoteStore or durable KV system.
- AegisKV replaces SGLang's KV manager.
- AegisKV requires DRL. The first paper should be explainable and classical.
- Offline counterfactual results are equivalent to online runtime results.

## 未来扩展

After the two-tier paper is stable:

- add RemoteStore for host failure；
- add DRLPart-style budget controller；
- add failure-prediction-aware exposure；
- add cross-node restore scheduling；
- add application-level checkpoint / summary interactions；
- evaluate with real Codex/CrewAI/LangGraph traces。
