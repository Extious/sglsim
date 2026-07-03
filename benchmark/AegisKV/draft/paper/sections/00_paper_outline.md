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
