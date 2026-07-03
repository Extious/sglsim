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
