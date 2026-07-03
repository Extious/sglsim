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
