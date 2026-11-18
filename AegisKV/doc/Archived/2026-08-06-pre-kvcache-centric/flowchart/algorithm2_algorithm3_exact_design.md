# AegisKV Algorithm 2 / Algorithm 3 设计

> 本文描述目标算法，而不是完成状态声明。目标模型只考虑 GPU failure，并在请求长度未知时逐 chunk 做在线备份决策。当前实现仍有一部分使用请求级收益按 token 比例分摊；具体差距见第 11 节。

对应系统图：

- [`system_overview.drawio`](./system_overview.drawio)
- [`system_overview.preview.png`](./system_overview.preview.png)

## 1. 一句话概括

AegisKV 的三个算法形成以下控制链：

1. Algorithm 1 计算请求的 Slack 和关键程度；
2. Algorithm 2 在每个新 KV chunk 到达时，决定是否备份到 HBM、Host，或者不备份；
3. Algorithm 3 根据窗口 $K$ 的实际需求、占用和未满足量，调整窗口 $K+1$ 每个 DP 统一的 $\alpha$ 和 $\beta$。

这里最重要的变化是：

> Algorithm 2 不再假设请求的最终长度已知，也不再把请求级总收益平均分给所有 chunk。它只比较当前新 chunk 带来的边际收益和边际成本。

## 2. 系统假设与核心符号

### 2.1 Failure 范围

本文只考虑：

- 单 GPU failure；
- 一块 PP GPU 失败会使所在 execution DP 暂时不可用；
- 请求转移到另一个健康的 backup DP；
- 不考虑 node、network、rack 或 correlated failure；
- KV 始终在相同 PP stage 之间传输。

若 execution DP 为 $e_r$、backup DP 为 $b_r$，则：

$$
b_r\ne e_r.
$$

在一个 route version 内，$b_r$ 保持不变。如果必须更换 backup DP，需要先在新 DP 完成完整前缀复制和 ACK，再原子切换 route version。

### 2.2 只保留这些符号

| 符号 | 含义 |
|---|---|
| $r$ | 一个请求 |
| $c$ | 一个 KV chunk |
| $e_r$ | 请求正在执行的 DP |
| $b_r$ | 请求的 backup DP |
| $a_r$ | 当前已经产生的最后一个 chunk |
| $k_r$ | 当前已经完整 ACK 的连续备份前缀 |
| $S_r$ | Algorithm 1 给请求计算的 Slack |
| $w_r$ | 请求关键程度 |
| $P_r^{\mathrm{fail}}$ | 当前决策区间内 execution GPU failure 的概率 |
| $\alpha_n$ | DP $n$ 可用于备份的 HBM 比例 |
| $\beta_n$ | DP $n$ 可用于备份的 Host 比例 |

其余二进制变量、线性化变量和对偶价格属于求解器实现细节，不放在主设计中。

## 3. 请求长度未知时如何在线决策

### 3.1 两个前缀

在任意时刻，Algorithm 2 只知道：

- produced frontier $a_r$：execution DP 已经计算到哪里；
- safe frontier $k_r$：backup DP 已经可靠备份到哪里。

始终有：

$$
0\le k_r\le a_r.
$$

算法不知道最终还会产生多少 chunk，也不能读取未来 failure manifest。

### 3.2 Safe frontier 的条件

一个 chunk 只有满足以下条件，才能推进 $k_r$：

1. 它已经复制到 backup DP；
2. execution PP stage $p$ 的 KV 只复制到 backup PP stage $p$；
3. 所有 PP stage 都完成版本匹配的 `chunk_commit_ack`；
4. 它之前的所有 chunk 也已经完成 ACK。

因此，如果 chunk 1、2、4 已经复制，但 chunk 3 缺失，则有效 safe frontier 仍然只能到 chunk 2。

### 3.3 跳过、补传和回收

- 如果之前跳过了 chunk $c$，只有在 source KV 仍存在并把它补传完成后，才能继续扩展后面的 safe frontier；
- 如果 source KV 已经被回收，这个缺口之后的 chunk 暂时都不能贡献恢复收益；
- budget 不足时，只能从备份前缀的尾部开始回收；
- 不能删除中间 chunk 后仍把后面的 chunk 计为有效备份。

## 4. Algorithm 2 如何计算 chunk 的恢复收益

### 4.1 当前前缀的恢复时间

假设当前请求已经产生到 $a_r$，其中前 $k$ 个 chunk 可从 backup DP 恢复，则：

$$
T_r(k)
=
T_{\mathrm{fixed}}
+T_{\mathrm{restore}}(1{:}k)
+T_{\mathrm{recompute}}(k{+}1{:}a_r).
$$

其中：

- $T_{\mathrm{fixed}}$ 包含 failure detection、reroute 和 resume；
- restore setup 只在确实恢复备份时支付一次；
- recompute setup 只在确实存在重算 suffix 时支付一次；
- HBM 和 Host 使用各自的恢复速度；
- 多个 PP stage 并行恢复时，由最慢 stage 和链路排队决定完成时间。

$k=0$ 表示当前前缀全部重算；$k=a_r$ 表示当前已经产生的 KV 全部恢复。

### 4.2 Slack 可见损失

请求的恢复时间不一定都会暴露给 workflow。Slack 能隐藏的部分不计入损失：

$$
L_r(k)
=
w_r\max\left(T_r(k)-S_r,0\right).
$$

因此：

- Slack 很大时，即使发生重算，也可能不影响 workflow；
- Slack 很小时，更长的恢复时间会直接增加 workflow completion time；
- $w_r$ 越高，保护该请求越重要。

### 4.3 新 chunk 的边际恢复收益

当 safe frontier 从 $c-1$ 扩展到 $c$ 时，新 chunk 的期望边际收益是：

$$
\Delta B_{r,c}
=
P_r^{\mathrm{fail}}
\left[L_r(c-1)-L_r(c)\right].
$$

这是 Algorithm 2 应使用的收益。

不能使用：

- 整个请求的累计恢复收益；
- 最终 token 数；
- 请求总收益乘以 `chunk_tokens / request_tokens`；
- 只按 chunk 大小计算的静态 density。

## 5. Chunk 的边际备份成本

新备份一个 chunk 的成本可以简写为：

$$
\Delta C_{r,c}
=
C_{\mathrm{copy}}
+C_{\mathrm{interference}}
+C_{\mathrm{residency}}
+C_{\mathrm{transition}}.
$$

分别表示：

- `copy`：实际复制 bytes、链路时间和 copy setup；
- `interference`：复制对正常 prefill/decode 的影响；
- `residency`：持续占用 HBM 或 Host 的 byte-time；
- `transition`：补传、tier migration 或 eviction 的开销。

需要区分两类资源：

- resident stock：已经 ACK、继续占用 HBM/Host 的 KV；
- new copy flow：当前控制周期真正需要传输的新 KV。

已经驻留的 chunk 不应该在每次重新求解时重复收取 copy bandwidth，但仍然占用容量并产生 residency cost。

Algorithm 2 必须同时满足：

- 每个 PP stage 的 HBM hard budget；
- 每个 DP 的 Host policy hard budget；
- node 物理 Host 容量；
- source DP 到 backup DP 的定向链路 budget。

当空间或带宽接近上限时，新 chunk 的机会成本会上升。这个机会成本由全局求解器从容量约束中体现，不需要在主设计中再引入一组 shadow-price 符号。

## 6. Fairness：为什么后面的 chunk 优先级通常更低

### 6.1 不能让长请求占满所有空间

请求越长，累计恢复收益和累计备份成本都会增加。如果始终按累计收益排序，长请求可能持续占用 HBM/Host，后来的短请求得不到任何保护。

因此，对每个请求的累计保护收益使用一个边际递减的公平函数。默认可以使用：

$$
F_r(k)
=
\log\left(\epsilon+B_r(k)\right),
$$

其中：

- $B_r(k)=P_r^{\mathrm{fail}}[L_r(0)-L_r(k)]$；
- $\epsilon$ 是防止对 0 取对数的小常数；
- 一个请求已经得到越多保护，继续保护它的公平边际收益越低。

新 chunk 的公平边际价值是：

$$
\Delta F_{r,c}=F_r(c)-F_r(c-1).
$$

如果需要防止通过拆分请求获取更多份额，可以在 workflow 或 tenant 粒度累计 $B$ 后再计算同一个公平函数。

### 6.2 优先级的直观形式

Algorithm 2 可以用下面的形式解释一个候选 chunk：

$$
\mathrm{Priority}_{r,c}
=
\Delta F_{r,c}-\Delta C_{r,c}.
$$

后续 chunk 的优先级通常下降，因为：

1. 同一个请求已经获得更多保护，公平边际收益下降；
2. 请求占用的总空间不断增加；
3. 系统剩余 budget 逐渐减少；
4. 后续 chunk 的剩余故障暴露时间通常更短。

但不要硬编码“chunk 编号越大，优先级一定越低”。以下情况可能使某个后续 chunk 暂时更重要：

- 它使恢复时间刚好越过 Slack 临界点；
- chunk 大小不同；
- 第一个 chunk 需要承担一次性 setup；
- Algorithm 1 更新了 Slack 或关键程度；
- source KV 即将被回收，这是最后补传机会。

因此，算法应通过边际收益、公平函数和 hard budget 自然得到优先级，而不是人为给 chunk 编号乘衰减系数。

## 7. Algorithm 2 的完整决策流程

### 7.1 每个新 chunk 到达时

1. Scheduler 报告 request、chunk、PP stage 和 bytes；
2. Controller 更新 produced frontier $a_r$；
3. 对每个活跃请求，构造当前可行的备份前缀方案；
4. 每个方案明确 backup DP、safe frontier、HBM/Host placement、copy cost 和资源占用；
5. 求解器在所有活跃请求之间联合选择方案；
6. Controller 只下发相对上一计划新增的 copy、migration 或 tail eviction；
7. 所有 PP stage 完成 `chunk_commit_ack` 后，原子推进 $k_r$。

### 7.2 求解目标

Algorithm 2 可以概括为：

$$
\max
\quad
\sum_r F_r(k_r)
-\mathrm{TotalBackupCost},
$$

同时满足：

- 每个请求只有一个 backup DP；
- backup DP 在 route version 内不变；
- 只保留连续 ACK 前缀；
- 同 PP stage 传输；
- HBM、Host、node Host 和链路 hard budget；
- 当前 GPU health、source KV availability 和版本约束。

因为当前 $a_r$ 是有限的，每个请求只有有限个可行前缀方案。可以先为每个方案计算公平价值和成本，再由 MILP 选择方案；因此不需要把对数函数直接交给 MILP。

Priority 只用于解释结果或快速筛选候选，不能把精确路径退化成逐 chunk greedy。

### 7.3 Budget 满时如何回收

对当前 safe frontier 为 $k_r$ 的请求，唯一安全的单步回收是：

$$
k_r\rightarrow k_r-1.
$$

Algorithm 2 比较所有请求尾部 chunk 的公平价值损失和释放资源量，再决定回收谁。不能简单使用 FIFO、LRU 或“优先驱逐最长请求”。

## 8. Algorithm 3：如何调整 alpha 和 beta

### 8.1 每个 DP 统一管理

对 DP $n$：

$$
\alpha_n
=
\frac{\text{HBM backup budget}}
{\text{maximum HBM backup budget}},
$$

$$
\beta_n
=
\frac{\text{Host backup budget}}
{\text{maximum Host backup budget}}.
$$

同一 DP 的所有 PP stage 使用相同的 $\alpha_n$、$\beta_n$ 和 budget version。

### 8.2 窗口 K 收集什么

Algorithm 3 保留 D/U/O 的原始含义，但不能只记录请求数或 token 数：

| 反馈 | 含义 | 至少记录 |
|---|---|---|
| D: Demand | Algorithm 2 尝试保护的需求 | chunk 数、copy bytes、公平价值 |
| U: Usage | 已 ACK 并实际驻留的使用量 | resident bytes、byte-time、peak、已实现价值 |
| O: Overflow | 想保护但没有兑现的需求 | 未满足 chunk、bytes、丢失的公平价值 |

同时单独记录正常路径保护指标：

- 推理队列压力；
- SLO miss；
- late ACK bytes；
- backup bandwidth overflow；
- copy 对 prefill/decode 的干扰。

这样可以区分：

- `D 高、U 高、O 高`：确实需要更多 budget；
- `D 低、U 低`：当前 budget 过大，可以回收；
- `O 高但正常路径压力也高`：不能简单扩容，应先降低 copy pressure 或调整 tier；
- `U 高但实现价值低`：空间被低价值长请求占用，需要加强 fairness 或 eviction。

### 8.3 从 K 到 K+1

Algorithm 3 的作用可以简写为：

$$
(\alpha_n,\beta_n)_{K+1}
=
\mathrm{Optimize}
\left(D_n^K,U_n^K,O_n^K,\mathrm{NormalPressure}_n^K\right).
$$

它同时考虑：

- 扩大 budget 能增加多少公平恢复价值；
- HBM/Host 被正常推理占用的机会成本；
- 窗口间改变 budget 的切换成本；
- 正常路径 SLO guard；
- 当前 GPU failure rate。

窗口 $K$ 是控制时间窗或事件窗，不是请求长度。Algorithm 2 可以在窗口内多次响应 chunk 到达，但 Algorithm 3 只在窗口边界更新一次 $\alpha/\beta$。

### 8.4 原子发布

1. Algorithm 3 生成新的 budget version；
2. Controller 检查 resident floor 和 node Host 物理容量；
3. 向该 DP 的全部 PP stage 广播相同的 $\alpha/\beta$；
4. 收齐所有 stage 的 `budget_ack`；
5. Algorithm 2 才能开始使用新 budget；
6. ACK 未完成时继续使用旧 version。

`budget_ack` 与 `chunk_commit_ack` 是不同协议，不能互相代替。

## 9. GPU failure 时如何恢复

GPU failure 发生后：

1. 标记 execution DP unavailable，停止向它发送新请求；
2. 根据已提交的 route version 找到 backup DP；
3. 读取故障前最后一个 safe frontier $k_r$；
4. 从 HBM/Host 恢复 chunk $1$ 到 $k_r$；
5. 重算 $k_r+1$ 到故障前 produced frontier $a_r$；
6. 恢复请求执行；
7. 记录 restored/recomputed chunk、bytes、延迟和 workflow-visible loss。

尚未完成所有 PP stage ACK 的 chunk 不能参与恢复。

## 10. “全局最优”的准确含义

### 10.1 可以声称什么

如果当前活跃请求、已经产生的 chunk、safe frontier、Slack、GPU failure rate、恢复时间、资源占用和可行方案全部被冻结，并且求解器返回 zero gap 且独立验证通过，可以声称：

> 该计划是在当前因果可见状态和冻结模型下，对所有活跃请求的全局最优公平前缀分配。

### 10.2 不能声称什么

该证书不能证明：

- 对未知最终请求长度的 clairvoyant offline 最优；
- 对未来请求到达和未来 GPU failure 时间的全局最优；
- failure rate、恢复时间和 interference 估计完全准确；
- 一个近似物理模型的 zero gap 等于真实系统最优。

如果要优化完整未来，需要给出 chunk continuation、EOS、请求到达和 failure 的概率模型，再求解随机动态规划。没有未来分布或 oracle 信息时，在线算法不可能保证与 clairvoyant offline optimum 完全相同。

## 11. 当前实现状态

### 已有基础

- backup DP 与 HBM/Host/Recompute placement；
- stage-aligned transfer；
- HBM、Host、node Host 和链路容量约束；
- 全 PP stage ACK 与连续恢复前缀；
- DP-wide $\alpha/\beta$ budget version；
- MILP zero-gap、独立验证和 deterministic replay；
- certified plan 的安装与 ACK contract。

### 仍需实现

1. 当前 causal exact instance 主要在 request admission 时建立，尚未由每个真实 chunk completion 驱动。
2. 当前实现仍将请求级 unprotected/protected loss 按 token 比例分给 chunk，尚未实现第 4 节的前缀边际收益。
3. 当前 exact instance 仍支持一般 failure scope，尚未收敛为本文 GPU-only contract。
4. 公平函数尚未接入 exact objective。
5. tail-only eviction、backfill 和 tier migration 尚未在同一个 exact problem 中联合求解。
6. HBM/Host restore latency、copy interference 和 byte-time residency 尚未完整标定。
7. Algorithm 3 尚未使用 chunk/byte/value 粒度的 D/U/O 自动发布窗口 $K+1$ budget。
8. 没有 certified plan 时仍可能走 `legacy_greedy`；该路径不能算作本文精确算法。

因此，本文是下一版 Algorithm 2/3 的设计规范，不是已经完成的实现声明。

## 12. 代码对应关系

| 功能 | 实现文件 |
|---|---|
| MILP 与最优性证书 | [`aegis_exact_optimizer.py`](../../python/sglsim/srt/simulation/aegis_exact_optimizer.py) |
| causal instance 与当前旧 loss 构造 | [`aegis_exact_rolling.py`](../../python/sglsim/srt/simulation/aegis_exact_rolling.py) |
| certified plan contract | [`aegis_exact_contract.py`](../../python/sglsim/srt/simulation/aegis_exact_contract.py) |
| budget publisher 与窗口指标 | [`aegis_budget.py`](../../python/sglsim/srt/simulation/aegis_budget.py) |
| exact plan 安装、stage placement 与 ACK | [`data_parallel_controller.py`](../../python/sglsim/srt/managers/data_parallel_controller.py) |
| legacy/observational window logic | [`aegis_l1_l2.py`](../../python/sglsim/srt/managers/aegis_l1_l2.py) |
