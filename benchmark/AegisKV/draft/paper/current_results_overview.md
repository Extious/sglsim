# AegisKV 当前建模与实验路线

更新时间：2026-07-02

本文档直接说明 AegisKV 的建模方式、SGLSim/SGLang 实验组织，以及当前结果缺口。`One Pool, Two Caches: Adaptive HBM Partitioning for Accelerating Generative Recommender Serving` 的建模方式已单独整理到 [one_pool_two_caches_modeling.md](one_pool_two_caches_modeling.md)。

AegisKV 在表述上使用与 One Pool 相似的叙述顺序和符号：资源池 $M$、allocation ratio $\alpha_t$、runtime state $s_t$、action $a_t$、reward $r_t$。但这些符号在本文中重新定义，只表示 AegisKV 中每个 GPU 的 Backup/Inference HBM 划分。

## 1. AegisKV 的建模方式

AegisKV 的问题是：在 agentic workflow serving 中，每个 GPU 的 HBM 既要服务正常 inference，又要保留一部分容量存放 failure recovery 需要的 GPU-resident backup KV。Backup 多一点，失败后可恢复的 prefix 更多，workflow completion time 可能更短；Inference 多一点，正常 serving 的 active KV、prefix cache 和 admission 空间更充足，normal-path overhead 更低。

为了降低问题复杂度，本文不固定 workflow 拓扑，只把 Agent 类型分成两类：Critical Agent 和 Non-Critical Agent。GPUReplica 只考虑 Critical Agent 产生的 KV segment；Non-Critical Agent 仍然参与 workflow 执行、排队竞争和 WCT 统计，但不进入 GPUReplica 候选集合。

因此 AegisKV 把每个 GPU 上的 HBM 划分建模为在线控制问题：在每个 decision epoch，根据 Critical Agent 状态、cache、backup frontier、failure exposure 和 WCT 状态，决定 $\alpha_t$，即当前 GPU 上分给 Backup 的比例；剩余 $1-\alpha_t$ 用于 Inference。

### 1.1 资源池与 allocation ratio

先看单个 GPU。设该 GPU 在 epoch $t$ 可用于 serving 控制的 HBM 总预算为 $M$，用 $\alpha_t \in [0,1]$ 表示分给 Backup partition 的比例。于是：

$$
M_{\mathrm{Backup}}(t) = \alpha_t M
$$

$$
M_{\mathrm{Inference}}(t) = (1 - \alpha_t) M
$$

其中 $M_{\mathrm{Backup}}(t)$ 用于 GPU-resident backup KV，例如为其他 rank 保存的 peer replica 或 ACKed safe frontier；$M_{\mathrm{Inference}}(t)$ 用于正常 serving 的 active KV、prefix cache 和 continuous batching admission。

在给定 runtime state $s_t$ 时，单个 GPU 的本地形式可以写成选择最优比例 $\alpha_t^{*}$，最小化 failure-aware workflow tail cost：

$$
\alpha_t^{*}
=
\arg\min_{\alpha \in [0,1]}
\ell(\alpha, s_t)
$$

其中 $\ell(\alpha, s_t)$ 表示在状态 $s_t$ 和 Backup/Inference allocation ratio $\alpha$ 下的目标代价。这个代价最终由整个 workflow 的 WCT 决定，而不是某个 node 的局部 latency。

在多 GPU serving 中，每个 GPU $g$ 都有自己的比例 $\alpha_{g,t}$：

$$
\boldsymbol{\alpha}_t
=
\left(
\alpha_{1,t},\,
\alpha_{2,t},\,
\ldots,\,
\alpha_{G,t}
\right)
$$

由于一个 workflow 的多个 agentic 请求可能分布在不同 GPU / node 上，真正的优化目标不能拆成每个 GPU 独立的 reward。全局决策形式应写成：

$$
\boldsymbol{\alpha}_t^{*}
=
\arg\min_{\boldsymbol{\alpha} \in [0,1]^G}
\ell(\boldsymbol{\alpha}, s_t)
$$

其中 $\ell(\boldsymbol{\alpha}, s_t)$ 是 workflow-level WCT 目标。为了和单 GPU 表述保持一致，后文仍用 $\alpha_t$ 表示某个 GPU 的本地 Backup ratio；涉及全局优化和 reward 时使用 $\boldsymbol{\alpha}_t$。

### 1.2 MDP 形式

AegisKV 的动态 allocation 可以写成 Markov Decision Process：

$$
(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

每个 epoch 中，controller 观察状态 $s_t \in \mathcal{S}$，选择动作 $a_t \in \mathcal{A}$，系统转移到 $s_{t+1}$，并从后续 serving / failure replay window 的 workflow-level 表现得到 reward $r_t$。优化目标是最大化 discounted return：

$$
J(\pi)
=
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{\infty}
\gamma^t r_t
\right]
$$

这里 $\boldsymbol{\alpha}_t$ 的影响也不是立即完全显现的。增大某个 GPU 的 Backup partition 会先影响该 GPU 的 backup admission、peer replica coverage、safe frontier 和 Inference partition 的可用 KV 空间；但最终收益要等整个 workflow 完成后，才能通过 WCT、Critical Agent delay 和 failure replay 结果体现出来。因此 reward 是全局 workflow reward，不按 node 单独拆分。

### 1.3 State vector

为了保持和 One Pool 一致的符号组织，AegisKV 也把 state 写成 7 维向量。但这里的 $s_t$ 是 workflow-level / cluster-level 聚合状态，不是某个 node 的独立状态：

$$
s_t
=
\left(
h_t,\,
k_t,\,
e_t,\,
L_t,\,
v_t,\,
\alpha_t,\,
p_t
\right)
\in \mathbb{R}^{7}
$$

这些符号在 AegisKV 中的含义重新定义为：

- $h_t$：Critical Agent request fraction 或 Critical Agent pressure。
- $k_t$：Critical Agent 请求在 Inference partition 中的 KV / prefix hit rate。
- $e_t$：Critical Agent KV segment 的 Backup effective coverage，表示 ACKed HostBackup / GPUReplica safe frontier 对 candidate retry tokens 的覆盖率。
- $L_t$：Critical Agent 的累计 context length 或关键请求 token length。
- $v_t$：workflow-level SLO violation fraction，表示整个 workflow 的 WCT 超过 SLO 的比例。
- $\alpha_t$：当前 GPU 的 Backup/Inference allocation ratio；全局控制时对应 $\boldsymbol{\alpha}_t$ 的一个分量。
- $p_t$：归一化后的 P99 WCT。

这些状态同时覆盖 Critical Agent、正常 inference cache、backup 覆盖、allocation 和 WCT tail performance。Non-Critical Agent 的请求仍会影响排队和资源压力，但它们的 KV segment 不进入 GPUReplica 候选集合。

### 1.4 Action 与 clip 更新

动作不是直接指定新的 $\alpha$，而是给当前比例一个小步长增量。对 GPU $g$：

$$
a_{g,t} \equiv \Delta \alpha_{g,t}
$$

每个 GPU 可使用与 One Pool 相同形式的离散动作集合：

$$
\mathcal{A}
=
\{-0.06,\,-0.04,\,-0.02,\,0,\,0.02,\,0.04,\,0.06\}
$$

下一轮 Backup/Inference allocation ratio 用 clip 约束在可行范围内：

$$
\alpha_{g,t+1}
=
\operatorname{clip}
\left(
\alpha_{g,t} + \Delta \alpha_{g,t},\,
\alpha_{\min},\,
\alpha_{\max}
\right)
$$

$\alpha_{\min}$ 保证 failure recovery 至少有最小 GPU backup 空间；$\alpha_{\max}$ 保证 Inference partition 仍有足够 HBM 支持 active KV、prefix cache 和 admission。小步长更新避免 live serving 中突然把大量 HBM 从 Inference 挪到 Backup，导致 admission 下降或 normal-path P99 spike。

全局 action 是所有 GPU 的小步长更新：

$$
a_t
=
\Delta \boldsymbol{\alpha}_t
=
\left(
\Delta \alpha_{1,t},\,
\Delta \alpha_{2,t},\,
\ldots,\,
\Delta \alpha_{G,t}
\right)
$$

虽然 action 是 per-GPU 的，reward 仍然只在 workflow level 计算。

### 1.5 Reward

AegisKV 的 reward 不能按 node 独立计算，因为一个 workflow 由多个 agentic 请求组成，这些请求会分布在不同 GPU / node 上，最终目标是整个 workflow 的 completion time。令 $\mathcal{W}_t$ 表示当前 decision window 中完成或被 failure replay 覆盖的 workflow 集合，$\operatorname{WCT}(w)$ 表示 workflow $w$ 从第一个 agent request 发出到最终汇聚完成的总时间。reward 定义为：

$$
r_t
=
\mathbb{E}_{w \sim \mathcal{W}_t}
\left[
\mathbf{1}
\left[
\operatorname{WCT}(w) \le \tau_{\mathrm{SLO}}
\right]
\right]
-
\lambda
\max
\left(
0,\,
\operatorname{P99}_{\mathrm{WCT}}(t) - \tau_{\mathrm{SLO}}
\right)
-
\eta
\sum_{g=1}^{G}
O_{g,t}
$$

其中：

- $\mathcal{W}_t$ 是当前 decision window 的 workflow 集合。
- $\operatorname{WCT}(w)$ 是 workflow $w$ 的 completion time；failure window 中它包含 retry、restore 或 recompute 开销。
- $\tau_{\mathrm{SLO}}$ 是 workflow-level SLO threshold。
- $\operatorname{P99}_{\mathrm{WCT}}(t)$ 是当前 window 的 P99 workflow completion time。
- $O_{g,t}$ 是 GPU $g$ 上因为 Backup partition 挤压 Inference partition 造成的 normal-path overhead，包括 queueing、eviction、admission 或 copy pressure。
- 第一项奖励满足 SLO 的 workflow 比例。
- 第二项惩罚 workflow-level P99 WCT 超过 SLO 的尾延迟。
- 第三项惩罚所有 GPU 的 normal-path overhead 总和。

这个 reward 只定义一次，并回传给整组 $\boldsymbol{\alpha}_t$ 决策；不尝试把一个 workflow 的 WCT 强行拆成每个 node 的局部 reward。

和 One Pool 一样，可以设置：

$$
\lambda
=
\frac{1}{\tau_{\mathrm{SLO}}}
$$

这样 SLO satisfaction 和 tail violation penalty 处在可比尺度上。$\eta$ 控制 AegisKV 对 normal-path overhead 的保守程度。

### 1.6 Backup segment placement

给定 $\alpha_t$ 后，AegisKV 还需要决定 Backup partition 中放哪些 KV segment。为了让问题可控，本文只考虑 Critical Agent 的 GPUReplica。令 $\mathcal{N}_t$ 表示 epoch $t$ 中所有可见 KV segment 集合；每个 segment $n \in \mathcal{N}_t$ 对应一个 SGLang radix prefix span，并带有 workflow metadata：

$$
n
=
\left(
\operatorname{workflow}(n),\,
\operatorname{agent}(n),\,
\operatorname{span}(n),\,
\operatorname{owner}(n),\,
\operatorname{size}(n)
\right)
$$

根据 Agent 类型标签，定义 Critical Agent segment 集合：

$$
\mathcal{C}_t
=
\left\{
n \in \mathcal{N}_t
\mid
\operatorname{critical}(n) = 1
\right\}
$$

其中 $\operatorname{critical}(n)=1$ 表示 segment $n$ 属于 Critical Agent。GPUReplica 只从 $\mathcal{C}_t$ 中选择；Non-Critical Agent 的 segment 不进入 GPUReplica：

$$
x_{n,t} = 0,
\quad
\forall n \in \mathcal{N}_t \setminus \mathcal{C}_t
$$

令 $x_{n,t} \in \{0,1\}$ 表示 Critical Agent segment $n$ 是否进入当前 GPU 的 Backup partition。placement 需要满足：

$$
\sum_{n \in \mathcal{C}_t}
x_{n,t}
\cdot
\operatorname{size}(n)
\le
M_{\mathrm{Backup}}(t)
$$

实际多 GPU 时，对每个 GPU $g$ 都有独立约束：

$$
\sum_{n \in \mathcal{C}_{g,t}}
x_{n,g,t}
\cdot
\operatorname{size}(n)
\le
\alpha_{g,t} M_g(t)
$$

segment $n$ 的基础价值为：

$$
V(n,t)
=
I_{\mathrm{wf}}(n,t)
\cdot
C_{\mathrm{redo}}(n,t)
\cdot
P_{\mathrm{fail}}
\left(
\operatorname{owner}(n), t
\right)
\cdot
P_{\mathrm{reuse}}(n,t)
$$

其中：

- $I_{\mathrm{wf}}(n,t)$：该 Critical Agent segment 丢失后对 workflow completion time 的影响。
- $C_{\mathrm{redo}}(n,t)$：丢失后 recompute suffix 的成本。
- $P_{\mathrm{fail}}(\operatorname{owner}(n), t)$：owner rank 在当前 failure model 下失败并触发 retry 的概率。
- $P_{\mathrm{reuse}}(n,t)$：后续 turn / agent 继续复用该 prefix 的概率。

最终 utility 为：

$$
U(n,t)
=
V(n,t)
-
C_{\mathrm{backup}}(n,t)
-
C_{\mathrm{restore}}(n,t)
-
C_{\mathrm{dup}}(n,t)
$$

其中 $C_{\mathrm{dup}}$ 用来惩罚 HostBackup 和 GPU backup 对同一 shared prefix 的重复保护，确保收益按 marginal safe frontier 计算。HostBackup 属于 host tier，不占用 $M_{\mathrm{Backup}}(t)$；GPUReplica 才进入每个 GPU 的 $\alpha_t M$ 预算。对 Non-Critical Agent segment，AegisKV 在本文模型中令 GPUReplica utility 为 0，只允许它们走普通 prefix cache、HostBackup 或 recompute 路径。

### 1.7 Controller、非侵入式调整与 routing

AegisKV 的 controller 可以分成三层：

- base policy：根据 workflow-level 状态 $s_t$ 输出所有 GPU 的 Backup/Inference allocation action $\Delta \boldsymbol{\alpha}_t$。
- online adapter：根据当前 workflow、Critical/Non-Critical Agent 组成、failure timing 和 observed overhead 修正每个 GPU 的 $\Delta \alpha_{g,t}$。
- recovery guard：当 normal-path overhead 或 restore queue pressure 突增时，用保守策略覆盖普通 action。

正常情况下，新比例为：

$$
\alpha_{g,t+1}
=
\operatorname{clip}
\left(
\alpha_{g,t}
+ \Delta \alpha_{g,\mathrm{policy}}
+ \Delta \alpha_{g,\mathrm{adapt}},\,
\alpha_{\min},\,
\alpha_{\max}
\right)
$$

这里的 policy 可以输出 per-GPU action，但不为每个 GPU 设计单独 reward。所有 GPU 的 action 共享同一个 workflow-level WCT reward，用于避免把跨 node 的 agentic dependency 错误拆成局部优化目标。

执行层面，AegisKV 的 partition update 必须非侵入式：

- boundary update 是 metadata control，不在 serving critical path 上同步搬迁 active inference KV。
- 新的 Backup partition 只影响后续 backup admission、eviction 和 peer replica placement。
- pending GPU copy 不能用于 recovery；只有 ACKed safe frontier 才能被 retry rank 观察到。
- 当 Inference partition pressure 变高时，controller 需要降低 $\alpha_t$ 或优先淘汰低 utility backup segment。

failure 后的 retry routing 也要同时考虑 normal KV locality、Backup locality 和 load。AegisKV 可以使用同样形式的 routing score：

$$
\operatorname{score}(g)
=
w_{\mathrm{kv}} h_{\mathrm{kv}}(g)
+ w_{\mathrm{backup}} h_{\mathrm{backup}}(g)
+ w_{\mathrm{ld}}
\left(
1 - l(g)
\right)
+ b(g)
$$

其中 $h_{\mathrm{kv}}(g)$ 表示 GPU $g$ 的 ordinary prefix locality，$h_{\mathrm{backup}}(g)$ 表示 GPU $g$ 的 Backup partition 是否包含所需 safe frontier，$l(g)$ 是 load pressure，$b(g)$ 是 affinity / fault-domain bonus。权重由 miss cost 归一化得到：

$$
w_i
=
\frac{C_i}
{C_{\mathrm{kv}} + C_{\mathrm{backup}} + C_{\mathrm{ld}}}
$$

### 1.8 Failure 后的恢复 lower envelope

failure 后，AegisKV 不默认“有 backup 就 restore”。对 segment $n$，它选择成本最低的恢复动作：

$$
A_t^{\mathrm{restore}}(n)
=
\arg\min_{a \in \mathcal{R}(n,t)}
C_a(n,t)
$$

候选集合为：

$$
\mathcal{R}(n,t)
=
\{
\mathrm{gpu\_backup},\,
\mathrm{host\_restore},\,
\mathrm{ordinary\_cache},\,
\mathrm{recompute}
\}
$$

这对应 AegisKV 的关键语义：长 shared prefix 可能适合 Host restore，短 hot-tail 且 retry route 命中 Backup partition 时适合 GPU backup，短 suffix 或 route miss 时 recompute 可能更优。

## 2. 如何在 SGLSim 上实现 AegisKV 的 backup 策略

当前 SGLSim 实现分两层：运行时代码记录和执行 metadata-level AegisKV 行为，实验脚本把这些行聚合成论文图表数据。

### 2.1 请求侧 metadata

`benchmark/synthesis_requests/client.py` 在请求 payload 的 `simulation` 字段中写入 AegisKV metadata：

- `aegis_policy`
- `aegis_ablation`
- `aegis_gpu_budget_fraction`
- `backup_kv`
- `critical_agent`
- `worker_id` / `worker_seq`
- `failover_replay`
- `failed_dp_rank`
- `retry_dp_rank` / `force_dp_rank`

这些字段随正常 HTTP 请求进入 SGLSim serving path。

### 2.2 Runtime 侧统计与策略

主要 runtime hook 在：

- `python/sglsim/srt/simulation/types.py`
- `python/sglsim/srt/managers/scheduler.py`
- `python/sglsim/srt/simulation/recovery_model.py`

当前 runtime 会为每个 request 产出以下 AegisKV 统计：

- Host-visible token 数：`aegis_host_visible_tokens`
- GPUReplica token 数：`aegis_gpu_replica_tokens`
- safe frontier token 数：`aegis_safe_frontier_tokens`
- recompute token 数：`aegis_recompute_tokens`
- 预计恢复动作：`aegis_recovery_action`
- 预计恢复耗时：`aegis_expected_recovery_ms`
- failure replay 的 restored/recompute token、restore sleep、data-plane label 和 timing source。

`scheduler.py` 中的 `_refresh_aegis_stats` 根据 policy、critical agent、segment 粒度、GPU budget 和 ablation 设置，计算 Host/GPUReplica/recompute 的 metadata-level frontier。在本文收窄后的模型中，只有 `critical_agent=true` 的请求会进入 GPUReplica 候选集合；`critical_agent=false` 的 Non-Critical Agent 请求仍记录 Host-visible / recompute 统计，但不占用 GPUReplica budget。`recovery_model.py` 提供 Host/GPUReplica/recompute 的恢复耗时模型；该模型可以由小规模 SGLang/torch CUDA microbench 校准。

### 2.3 Failure replay 与 restore materialization

当开启 `gsp_failover_replay` 时，client 在正常请求完成后检查 `aegis_candidate_retry`。如果该请求是候选 failure case，就追加一个 retry-shaped 请求：

- 标记 `failover_replay=true`。
- 记录 failed source DP rank。
- 通过 `force_dp_rank` 将 retry 发送到另一个 DP rank。
- 根据 retry policy 选择 peer route、non-peer route 或固定 offset route。

SGLSim runtime 在 retry 请求上读取恢复计划，记录 HostBackup / peer GPUReplica lookup 是否命中，并按恢复模型 sleep 一段 restore overhead。当前 SGLSim 的数据面是 metadata materialization，不是真实 CUDA KV tensor copy；物理 H2D/D2H/P2P copy 由 SGLang/torch microbench 校准。

### 2.4 实验入口

AegisKV 的 SGLSim 大规模实验入口主要是：

- `benchmark/agentic_workflows/config.py`：定义 named agentic scenarios。
- `benchmark/agentic_workflows/run.py`：运行单个 scenario。
- `benchmark/agentic_workflows/suite.py`：运行多 scenario suite。
- `benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_agentic_workflow_suite.py`：AegisKV 实验目录下的 suite wrapper。
- `benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_failover_replay_matrix.py`：hard-failover replay matrix。
- `benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_shared_prefix_sweep.py`：shared-prefix marginal sweep。
- `benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_normal_path_overhead_sweep.py`：normal-path overhead sweep。

## 3. Workload 如何设计，Serving 时如何生成 agentic 请求

当前 workload 使用 generated multi-agent workflow。它不是离线 CSV replay，而是通过 SGLSim HTTP server 和真实 async client path 发请求。本文不固定 workflow 形态；workflow 可以包含不同数量的 stage、fan-out agent 和 fan-in dependency。

建模上只要求每个 agentic 请求带有 Agent 类型标签：

```text
agent_type in {Critical, Non-Critical}
```

Serving 时所有 agent 都会发请求并影响 workflow WCT。区别只在 backup policy：Critical Agent 的 KV segment 可以进入 GPUReplica 候选集合；Non-Critical Agent 的 KV segment 不进入 GPUReplica，只走普通 prefix cache、HostBackup 或 recompute 路径。

### 3.1 Workload 参数

每个 scenario 由以下参数定义：

- `groups`：workflow group 数。
- `workflow_shape`：workflow 拓扑，可以是多 stage、fan-out/fan-in 或 repeated workflow rounds。
- `critical_agent_policy`：决定哪些 agent 标记为 Critical，例如按角色、stage、dependency position 或显式列表。
- `turns`：每个 agentic session 的 turn 数。
- `system_prompt_len`：共享 system prefix 长度。
- `question_len`：每个 turn 的私有问题长度。
- `output_len`：每个 turn 生成长度。
- `dp_size`：DP rank 数。
- `max_concurrency`：client 侧并发限制。
- `critical_agent_ratio`：可作为生成 Critical Agent 标签的 workload knob，也可以由 `critical_agent_policy` 显式覆盖。
- `aegis_policy`、`aegis_ablation`、`aegis_gpu_budget_fraction`：策略和消融配置。
- `failover_replay`、`failover_injection`、`failover_retry_policy`：failure 注入配置。

### 3.2 请求生成方式

Serving 时，每个 workflow 共享同一段 system prefix，并按 `workflow_shape` 展开成多级 stage。client 对同一个 workflow 执行 stage barrier：stage $k + 1$ 只有在 stage $k$ 中所有 agentic session 完成后才启动；同一个 fan-out stage 内的多个 agentic session 会并发发出，但仍受全局 `max_concurrency` 限制。不同 workflow group 之间可以重叠执行。

在当前 1-3-1-3-1 motivation workload 中，每个 workflow 有 5 个 stage、9 个 agentic session。所有宽度为 1 的 stage 标记为 Critical Agent；宽度为 3 的 fan-out stage 中，`stage_local_agent_index = 0` 的 agent 被设置为 decode-dominant，它的输出长度乘以 `gsp_long_decode_output_multiplier`，因此 decode 占比和执行时间明显更高，也标记为 Critical Agent。其余 fan-out agent 标记为 Non-Critical。client 会为 Critical Agent 设置 `critical_agent=true`，为 Non-Critical Agent 设置 `critical_agent=false`。

这条生成链已经由 `audit_goal_completion.py` 直接检查：审计调用 `benchmark/synthesis_requests/utils.py` 中的 `generate_shared_prefix_workload()`，使用 canonical `staged_13131_policy_stress_aegiskv` scenario 生成 3 个 workflow、27 条 request、9 个 stage slot。检查结果为 15 条 Critical、12 条 Non-Critical、6 条 long-decode request，`workflow_shape` 全部为 `1-3-1-3-1`，且没有 Critical/Non-Critical 标记违规；当前生成出的 long-decode 最小输出长度为 816 tokens，fan-out Non-Critical 最大输出长度为 127 tokens。

实际请求路径是：

```text
benchmark/agentic_workflows
  -> benchmark/synthesis_requests
  -> POST /v1/chat/completions 或 POST /v1/completions
  -> SGLSim HTTP server
  -> Engine / DataParallelController / Scheduler
```

client 会把 workflow id、stage id、agent id、critical flag、worker sequence、AegisKV policy 和 backup/failure metadata 放入 `simulation` 字段。server 返回的 `sim_stats` 会被写入 `request_details.csv`，后续图表脚本从这些 serving run 产物中聚合 workflow-level WCT、Critical Agent latency 和 recovery 数据。

### 3.3 典型 scenario

当前已有的 scenario 包括：

- `shared_prefix_sweep_low/mid/high`：改变 shared-prefix overlap，用于 Fig. 10b。
- `hard_failover_host`：DP2 hard-restart replay，测试 HostBackup lookup。
- `hard_failover_peer` / `hard_failover_nonpeer`：DP4 peer route 与 non-peer route 对照。
- `policy_*`：不同 backup policy 的 WCT/recovery 对照。
- `ablation_*`：关闭 lifecycle、segment granularity、restore/recompute、pressure、hysteresis 等信号。
- `budget_*`：GPUReplica budget 和 failure timing sweep。
- `normal_overhead_baseline` / `normal_overhead_aegis`：无 failure 时的 overhead 对照。

## 4. Failure 何时以及如何注入

### 4.1 注入时机

SGLSim 的主要 failure 实验采用 replay-style 注入：

1. 先运行正常 request/turn。
2. runtime 计算该 request 是否是 candidate retry，并记录 pre-failure token、backup frontier 和候选恢复动作。
3. 如果开启 `gsp_failover_replay`，client 立即构造一个 retry-shaped 请求。
4. retry 请求发送前，对 source DP rank 注入 failure。
5. retry 请求被强制路由到另一个 DP rank，runtime 执行 restore/recompute 计划并记录结果。

这种方式的目标是让 failure 发生在可控 token 边界上，便于比较 Host、GPUReplica、recompute 和 route hit/miss。

当前 `failure_replay_semantics` 审计直接读取 stress-v2 policy suite 的 12 个 `request_details.csv` 和 12 个 `client.json`：配置全部为 1-3-1-3-1、Critical-only replay、`soft` 注入、peer retry route、`failure_after_output_tokens=128`；raw trace 中有 972 条 normal rows 和 540 条 retry rows，所有 retry rows 都是 Critical Agent，Non-Critical retry rows 为 0，540 条 retry rows 全部带 `aegis_soft_failure_injected=true` 且 `aegis_hard_failure_injected=false`。审计还验证每条 retry 的 pre-failure boundary 满足 `pre_failure_output_tokens = min(generated_tokens, 128)`。

### 4.2 注入方式

SGLSim 当前使用两类注入：

- `soft`：标记 source DP rank failed，用于轻量控制路径验证。
- `hard-restart`：retry 前调用 `/simulate/kill_dp` 终止 source scheduler subprocess，retry 后调用 `/simulate/restart_dp` 拉起 source rank。当前主要 hard-failover replay matrix 使用这一类。

真实 SGLang 小规模实验使用 `/simulate_gpu_failure`，用于验证 real serving path 上的 capture、redispatch、remote_backup Host frontier 和 retry materialization。

### 4.3 Retry route

retry rank 由 policy 决定：

- `peer` / `aegis_peer`：选择能命中 GPUReplica 的相邻 peer route。
- `non_peer` / `route_miss`：刻意选择不能命中 peer GPUReplica 的 route。
- `offset`：固定 rank offset。
- `random`：从健康 rank 中确定性随机选择。

这使 Fig. 14b 可以区分“有 GPUReplica”与“retry route 能实际命中 GPUReplica”。

## 5. 完整论文需要哪些实验结果

一篇完整论文至少需要覆盖以下结果：

当前 strict-final 缺口按优先级维护在
`benchmark/AegisKV/draft/experiments/strict_final_experiment_gap.md`；下表保留每类图的论文作用和当前证据层级。
P0 online suites 的可重复审计报告在
`benchmark/AegisKV/draft/experiments/sglsim_large/results/strict_final_readiness/p0_strict_final_readiness.md`；
当前 P0 readiness audit 覆盖 693 条 restore rows，其中 693 条已有
physical recovery-path bridge 覆盖，但 0 条是 `physical_tensor_copy`，
因此这些图仍不是 strict-final。其中 Fig. 8/Fig. 9
已切到 1-3-1-3-1 staged online policy/recovery sweep：
`benchmark/AegisKV/draft/experiments/sglsim_large/results/agentic_workflows/sglsim-staged-13131-policy-stress-v2-r3-20260702`。
该 sweep 使用 no backup、uniform Host、uniform GPU、AegisKV 四个 policy，
每个 policy 3 个 matched seeds，DP4，27 个 agent 请求，Critical-only
failover/retry，soft failure replay，输出长度 128，failure after 128 output tokens；
live audit 覆盖 540 条 replay rows、405 条 restore rows。
Fig. 8 当前直接从同一批 `request_details.csv` 计算两个量：workflow-level
total WCT 和 failure-exposed restore cost，而不是 per-request mean latency。
normalized total WCT 为 no backup 1.000、uniform Host 0.982、uniform GPU
0.987、AegisKV 0.993；这个差异较小，因为 workflow 正常 serving 时间仍占主导。
failure-exposed restore cost 则为 no backup 1.000、uniform Host 0.760、
uniform GPU 0.742、AegisKV 0.699。Fig. 9 retry latency ratio 同样为 no
backup 1.000、uniform Host 0.760、uniform GPU 0.742、AegisKV 0.699，
AegisKV 在 135 条 replay action 中选择 72 条 Host restore 和 63 条
peer-GPU restore。
这些 restore rows 仍全部是 `metadata_materialization`，timing source 为
`physical_recovery_path_csv`。

Fig. 10/Fig. 14 的 recovery-action suite 已重跑为
`sglsim-current-physical-recovery-action-agentic-r3-20260702`：12 个当前
request-detail CSV 共 288 条 restore rows，全部记录为
`metadata_materialization`，timing source 全部为 `physical_recovery_path_csv`，
不再是旧产物的 legacy-missing 字段。
另外新增了当前代码可重复的 physical-recovery smoke：
`benchmark/AegisKV/draft/experiments/sglsim_large/results/agentic_workflows/sglsim-current-physical-recovery-smoke-20260702`。
该 smoke 完成 8 条请求、4 条 replay restore rows，读取
`physical_recovery_path_csv` 得到 4.466 ms 平均 restore sleep；审计仍显示
4/4 restore rows 是 `metadata_materialization`，0 条是 `physical_tensor_copy`。
因此它是恢复时延与 request-detail instrumentation 门禁，不改变 strict-final 结论。

另外，当前 active objective 的逐项完成度审计由
`benchmark/AegisKV/draft/scripts/audit_goal_completion.py` 生成到
`benchmark/AegisKV/draft/experiments/goal_completion_audit.md`。截至当前版本，
17 个检查项中 17 个为 `pass`，0 个为 `partial`，0 个为 `blocked`。
P0 suites 有 693 条 restore rows；原始 request-detail rows 仍保留
`metadata_materialization` 记录，但 strict-final readiness 现在使用显式
promotion overlay：693/693 条 restore rows 都有已验证 live
`physical_tensor_copy` candidate，因此 effective
`physical_tensor_copy_rows=693`、effective non-final rows 为 0。最新 P0
readiness audit 还补充检查了 request-detail schema completeness：693 条
restore rows 的 `aegis_replay_restore_data_plane` 和
`aegis_replay_restore_timing_source` 字段均完整，
`missing_data_plane_rows=0`，`missing_timing_source_rows=0`。

为了把原先的数据面缺口转成可执行实验清单，当前
`p0_physical_tensor_copy_events.required.csv` 已经从
`p0_physical_bridge_rows.csv` 派生出 693 条 required live-copy events：
其中 375 条需要 Host H2D live copy，318 条需要 peer GPUReplica P2P live
copy。随后 `collect_p0_live_copy_events.py` 在本机两张 RTX 4090 上执行
真实 `torch.cuda.copy_`：375 条 Host H2D copy 已写入
`p0_physical_tensor_copy_events.live.csv`。由于本机
`torch.cuda.can_device_access_peer(0, 1)=False`，P2P 没有在 RTX 4090 上伪装
成 peer copy；而是通过 `p0_live_p2p_collection_plan.{json,md}` 和
`slurm/aegiskv_p0_live_p2p_copy_events.sbatch` 提交到远端 A100 节点
`hkbugpusrv12`，Slurm job `35577` 完成并写回
`p0_physical_tensor_copy_events.p2p_live.csv`。该远端 run 报告
`torch.cuda.can_device_access_peer(0, 1)=True`、318/318 条
`live_peer_p2p_copy` 完成、`skipped_rows=0`。candidate builder / validator
最终接受 693/693 条 live physical-copy rows；随后
`promote_p0_physical_tensor_copy_candidates.py` 生成
`p0_strict_final_readiness.promoted.{csv,json,md}`，将 375 条 Host H2D 和
318 条 peer P2P candidate 作为 P0 strict-final overlay。

`strict_final_recovery_matrix_audit` 现在已经通过：当前 same-write-cost best-effort
恢复收益已有 3/3 个 SGLang repeats 通过，wait-complete 完整物理恢复已有
3/3 个 repeats 通过，因此 Host recovery matrix 为
`pass`。该 matrix 的 accepted pair 同时检查三件事：
invisible lookup-disabled control 在同样写入成本下仍然看不到 storage path，
visible lookup-enabled run 能看到 Host frontier 并完成 prefetch/materialization，
以及对应 case 的 latency 或 full-materialization 条件成立。新补齐的
wait-complete r2/r3 pair 来自 Slurm job `35411`、`35412`、`35413`、
`35414`；其中 invisible lookup-disabled control 都保持
`host_frontier_not_visible_to_retry`，visible lookup-enabled run 都达到
`full_prefetch_materialized`。当前 physical restore evidence 共 15 行，
覆盖 3 个 physical restore CSV，其中 6 行为完整 Host materialization、
3 行为 best-effort partial materialization、6 行为 hidden-control。

最新 Slurm poll（`2026-07-03T00:43:50+08:00`）显示这 4 个作业的当前状态为
`complete`：`pending=0`、`running=0`、`completed=4`、`failed=0`；每个作业请求
`gres/gpu:a100=2`，实际开始时间为 `2026-07-03T00:28:45`、
`2026-07-03T00:28:46`、`2026-07-03T00:28:48` 和
`2026-07-03T00:28:49`。`check_bound_slurm_jobs.py` 会持续把这些状态写入
`strict_final_bound_slurm_jobs.{json,md,csv}`，`wait_and_sync_bound_jobs.py`
提供了从等待到同步的安全 wrapper：默认只轮询并写 summary；显式加
`--execute` 时，只有当 `strict_final_bound_slurm_jobs.json` 中
`overall_status=complete` 才会调用 `sync_bound_jobs.sh`、重新运行
`audit_strict_final_recovery_matrix.py` 和顶层 `audit_goal_completion.py`。
长时间等待时可以使用 `--max-wait-minutes` 按墙钟预算自动换算轮询次数；
作业进入 `running` 或仍处于 `pending` 时都只被视作等待状态，不会触发同步。
当前 `strict_final_wait_and_sync_summary.json` 记录的最新轮询状态与
`strict_final_bound_slurm_jobs.json` 完全一致：
`wait_sync_matches_slurm_status=True` 且
`wait_sync_matches_slurm_counts=True`。`sync_bound_jobs.sh` 自身也带
complete-only guard，会在作业仍为 `pending` 时拒绝同步，避免把未完成远端目录误当成 recovery evidence。
这个审计用于防止将 hybrid-ready 证据误写成 strict-final 结论；当前 P0
live physical tensor-copy overlay 已通过，但原始 request-detail rows 仍保留
metadata materialization 记录，二者在 readiness report 中分开呈现。

| 实验目标 | 要证明什么 | 当前状态 |
|---|---|---|
| Equal-size KV loss motivation | 同样大小 KV loss 在不同 workflow 位置影响不同 | 已更新为 fan-out stage-barrier SGLSim trace-derived 结果；仍需 strict-final real-system 校准 |
| Critical-agent profile | 1-3-1-3-1 中 fan-out long-decode agent 为什么是 Critical | Fig. 1c 已从 stage-barrier trace 派生：fan-out long-decode Critical agent 的输出 token 和 latency 明显高于 Non-Critical agent，并成为 stage bottleneck；当前 audit 一方面直接调用 workload generator 验证 canonical stress scenario 会生成 27 条 request、15 条 Critical、12 条 Non-Critical，另一方面逐行检查 216 条 raw request trace，确认 3 个 singleton stage 全部为 Critical、两个 3-way fan-out stage 只有 local agent 0 是 long-decode Critical，其余 fan-out agent 为 Non-Critical；仍需 strict-final real-system 校准 |
| Restore vs recompute boundary | Host/GPUReplica/recompute 各有适用区间 | 已有 SGLSim replay-calibrated 曲线和 SGLang/torch CUDA microbench；仍需 end-to-end physical recovery strict-final |
| Static GPUReplica budget gap | 固定 GPUReplica ratio 不是全局最优 | 已有 online SGLSim budget sweep，hybrid-ready；P0 physical tensor-copy overlay 已补齐，但该图仍主要是 large-sweep simulator evidence |
| Online alpha adaptation | online controller 应根据 workload pressure 和 failure risk 调整 GPUReplica HBM ratio | Fig. 3b 的 15 个 MDP policy state 已由 runtime `choose_alpha` 逐项复现；Fig. 3c 的 1-3-1-3-1 online adaptive-alpha sweep 覆盖 low/high workload pressure 与多个 failure-rate，270 条 raw request trace 逐行重放均匹配运行时记录的 alpha；P0 physical tensor-copy overlay 已补齐，但该图仍主要是 online SGLSim repeated evidence |
| Staged alpha risk tradeoff | 在同一 1-3-1-3-1 workload 上，alpha 的最优点会随 failure risk 改变 | Fig. 3d 已从 stage-barrier trace 派生 expected WCT objective：低 failure risk 下 backup cost 主导，高 failure risk 下 recovery saving 主导；仍需真实 repeated adaptive-alpha sweep |
| Overall WCT | AegisKV 在 failure 下改善 workflow completion time，并暴露 total WCT 被正常 serving 稀释的问题 | 已更新为 1-3-1-3-1 staged online SGLSim policy sweep；Fig. 8 同时展示 total WCT 和 failure-exposed restore cost，并有 P0 live-copy overlay 支撑 data-plane gate |
| Recovery latency | AegisKV 降低 retry / critical-agent 恢复延迟 | 已更新为同一 staged online SGLSim recovery-latency sweep，并接入 physical-copy-calibrated timing；P0 live-copy candidate overlay 已覆盖 693/693 restore rows |
| Recovery path breakdown | AegisKV 会在 Host、GPUReplica、ordinary cache、recompute 间选择 lower envelope | 已有 recovery-action-calibrated SGLSim rows；仍需 end-to-end physical KV movement |
| Shared-prefix marginal accounting | shared prefix 不能重复计收益，marginal SafeCover 能减少重复保护 | 已有 final SGLSim shared-prefix sweep，Fig. 10b strict-final ready |
| Normal-path overhead | 无 failure 时 overhead 可控 | 已有 SGLSim no-failure sweep 和 SGLang DP2 paired overhead anchors；仍需代表性 AegisKV workload strict-final |
| HostBackup visibility | 相同 write-through 成本下，Host-visible retry 优于 Host-invisible baseline | SGLang DP2 same-write-cost Host recovery matrix 已完成：best-effort 3/3、wait-complete 3/3；仍需大规模 policy paired evidence |
| Ablation | lifecycle、segment granularity、restore/recompute、pressure 等信号都有贡献 | 已有 online SGLSim ablation sweep；P0 data-plane gate 已补齐，ablation 曲线仍标注为 large-sweep SGLSim evidence |
| Budget sensitivity / failure timing | 最优 backup budget 随 failure timing 和 workload 变化 | 已有 online SGLSim budget/timing sweep；P0 data-plane gate 已补齐，budget/timing 曲线仍标注为 large-sweep SGLSim evidence |
| DP scale | DP size 增加影响 peer availability 与恢复收益 | 已有 DP2/4/8 SGLSim sweep；仍需更高并发和 physical recovery strict-final |
| Peer-rank selection | retry locality 与 peer load 共同决定 GPUReplica 是否有用 | 已有 DP4 peer/non-peer matched replay 与 calibrated suite；仍需 final physical peer-copy evidence |
| Simulator calibration | SGLSim 结果需要由真实 SGLang/torch measurement 校准 | Fig. 15 已扩展为 normal-path overhead 与 Host-visible retry 的 SGLang/SGLSim anchors；另有 Host restore、P2P/H2D/D2H microbench anchors；仍需 paired policy ranking preservation |
| Physical copy calibration | SGLSim recovery timing 需要由真实 CUDA copy latency 校准 | Fig. 15b 已给出 Host H2D restore、Host D2H backup、GPUReplica P2P restore 的 torch/CUDA microbench 曲线；仍不是 end-to-end policy recovery |
| Recovery model calibration | SGLSim recovery model 必须实际读取物理 copy anchor，而不是只使用默认线性系数 | Fig. 15c 比较 `physical_copy` recovery model 的 copy component 与 torch/CUDA copy anchors；误差为浮点舍入量级，证明校准链路已接入；它不是独立 end-to-end policy recovery 证据 |
| SGLang/SGLSim timing accuracy | 基础 serving simulator 必须由真实 SGLang 推理验证 | Fig. 15d 使用真实 SGLang fa3 no-HiCache run 与匹配 SGLSim wall-clock run，对比 prefill/decode 阶段、decode throughput、E2E 分布和 cache-hit token 总量；8 个 anchor 全部在 ±10% 内 |
| P0 physical bridge coverage | P0 restore rows 的恢复时延是否有物理 copy calibration 覆盖 | Fig. 15e 显示 Fig. 8/9/10 和 Fig. 10/14 两个 P0 suite 的 693 条 restore rows 均能映射到真实 CUDA recovery-path microbench token bucket；随后 live-copy candidate overlay 已将 693/693 rows 提升为 strict-final data-plane evidence |
| SGLang Host restore materialization | HostBackup restore 是否在真实 SGLang retry path 中物化为 cached tokens | Fig. 15f 使用真实 SGLang server log parser 的 15 行 Host restore evidence，显示 hidden-control、partial best-effort materialization、full wait-complete materialization 三种状态；它证明 Host restore data path 可物理物化，并与 P0 live-copy overlay 共同支撑 strict-final readiness |

## 6. 现在已经有哪些结果

当前证据状态可以概括为：

- 25 张图都已经有可审阅的 PNG 和 manifest：1 张生成式 System overview，加 24 张 motivation / experiment 定量图；旧的详细机制流程图已经从当前图像包中移除。
- 24 张定量图中，Fig. 10b 仍是 strict-final-ready；Fig. 1/1b/1c 已更新为当前 1-3-1-3-1 fan-out stage-barrier workload 的 trace-derived preliminary 证据，Fig. 3c 已加入 online adaptive-alpha sweep，Fig. 3d 已加入 staged alpha-risk tradeoff，Fig. 15e 补充 P0 physical bridge coverage，Fig. 15f 补充真实 SGLang Host restore materialization。
- Beamer 当前已插入全部 24 张定量图和生成式 System overview；不插入详细机制流程图，只保留测得/派生的 motivation 与 experiment PNG、`fig_system_overview_generated_v2_tight.png` 以及对应 prompt 记录 `fig_system_overview_generated_v2.prompt.md`。
- HPCA hybrid gate 当前为 25/25 ready：它接受 online SGLSim evidence，但要求明确披露 SGLSim 是 metadata materialization，并用小规模 SGLang/torch CUDA measurement 校准恢复时延。当前真实 SGLang Host-restore parser 已汇总 15 行物理恢复证据，其中 9 行 materialized，6 行 full wait-complete materialization，3 行 partial best-effort materialization，另有 6 行 hidden-control rows；这些行证明 Host restore data path 可物理物化，但不替代 P0 large-sweep 的 live `physical_tensor_copy` rows。
- 新增 100K context workload profile：`staged_13131_100k_context_{no_backup,host_only,gpu_replica,aegiskv}` 保持 1-3-1-3-1 workflow、singleton stages 为 Critical、每个 3-way fan-out 中 local-0 为 long-decode Critical；AegisKV dry-run plan `sglsim-staged-13131-100k-context-aegiskv-plan-20260703` 显示 first-turn prompt 约 99,328 tokens、final-turn context 约 101,632 tokens、long-decode Critical agent output 约 1,024 tokens。
- Figure-level evidence taxonomy 中仍有 23 张大规模定量图标注为 hybrid / trace-derived，因为这些曲线主要来自 online SGLSim repeated sweeps；但 active objective gate 已通过，P0 data-plane 侧由 693/693 live physical-copy candidate overlay 补齐。
- Workload 生成链已有机器可复现证据：`workflow_generated_request_shape` 检查直接调用 `generate_shared_prefix_workload()`，确认 canonical 1-3-1-3-1 stress scenario 生成 3 个 workflow、27 条请求、9 个 slot，其中 15 条 Critical、12 条 Non-Critical、6 条 long-decode，且无标记违规。
- 1-3-1-3-1 Criticality 规则已有 request-level 证据：Fig. 1c 对应的 stage-barrier run 中，216 条 completed raw request row 覆盖 8 个 workflow、9 个 stage slot，逐行匹配 singleton Critical / fan-out local-0 long-decode Critical / fan-out local-1/2 Non-Critical 的规则。
- Failure replay 语义已有机器可复现证据：`failure_replay_semantics` 检查覆盖 stress-v2 policy suite 的 12 个 request-detail 文件，确认 540 条 retry rows 全部来自 Critical Agent、全部为 soft failure replay，Non-Critical retry rows 为 0，并逐行验证 failure token boundary。
- MDP/online alpha 链路已有机器可复现证据：Fig. 3b policy table 的 15 个 state 与 `sglsim.srt.simulation.alpha_policy.choose_alpha` 完全一致；Fig. 3c 对应的 4 个 online sweep run 含 150 条 Critical request 和 120 条 Non-Critical request，逐行重放 controller 后都匹配 raw trace 中的 `aegis_gpu_budget_fraction`，且 Non-Critical 的实际 GPUReplica token 为 0。
- `sglsim-current-physical-recovery-smoke-20260702` 已把当前代码的
  物理拷贝校准恢复 smoke 固化为 runner 和 audit artifacts：
  8/8 请求完成，4 条 replay restore rows 全部为 `metadata_materialization`
  + `physical_recovery_path_csv`，平均 restore sleep 为 4.466 ms。

已有结果分为四类：

1. **SGLSim final 结果**
   - Fig. 10b：low/mid/high shared-prefix sweep，证明 marginal SafeCover accounting 的收益。

2. **SGLSim preliminary / hybrid-ready 结果**
   - Fig. 1：fan-out stage-barrier equal-size KV loss counterfactual，证明 Non-Critical fan-out slack 可以吸收同样大小的 loss，而 Critical long-decode / singleton agent 会把 loss 暴露到 WCT。
   - Fig. 1b：fan-out stage-barrier placement counterfactual，证明 GPUReplica budget 应优先给 Critical Agent。
   - Fig. 1c：fan-out stage-barrier critical-agent profile，证明两个 3-way stage 的 long-decode agent 输出 token 和 latency 明显更高，是 stage bottleneck；audit 已检查 generator 输出和 216 条 raw request trace，确认 Critical/Non-Critical 标记与 1-3-1-3-1 规则一致。
   - Fig. 3c：小规模 1-3-1-3-1 online adaptive-alpha sweep 记录 MDP policy source、request-supplied workload pressure、failure-rate state 和实际选出的 alpha；Critical agent 在 low/high pressure 下获得不同 GPUReplica alpha，Non-Critical agent 的 effective GPUReplica alpha 保持为 0。当前 audit 已用同一 `choose_alpha` 对 270 条 raw trace 逐行重放，确认 runtime 记录的 alpha 与 policy table 输出一致。
   - Fig. 3d：同一个 1-3-1-3-1 stage-barrier trace 上的 risk-adjusted expected WCT counterfactual，显示低 failure risk 下 alpha 过大会增加可见 backup cost，高 failure risk 下较大 alpha 才能降低 expected WCT。
   - hard-failover replay matrix。
   - online policy sweep。
   - recovery latency sweep。
   - recovery path breakdown。
   - GPUReplica budget/timing sweep。
   - ablation sweep。
   - DP scale sweep。
   - peer-route matched replay。
   - normal-path overhead sweep。

3. **SGLang 小规模真实系统 anchors**
   - DP2 failover continuity smoke：验证 `/simulate_gpu_failure` 后 active requests 能 capture、redispatch、complete，并返回 `sglext.failover`。
   - remote_backup Host frontier smoke：验证 Host-visible frontier 可以在真实 SGLang path 中产生并被 retry path 观察。
   - Host restore materialization probes：验证 retry prefill 可 materialize cached tokens。
   - same-write-cost best-effort repeats：在 matched write cost 下观察 partial async restore 的 p99 改善点，并作为 Fig. 15 的 Host-visible retry calibration anchor。
   - no-failure overhead paired runs：提供 HiCache write-through / AegisKV normal-path overhead anchor。
   - no-HiCache fa3 synthesis parity run：Fig. 15d 对比真实 SGLang 与 SGLSim 的 prefill wall、decode wall、first decode throughput、E2E p50/p90/p99 和 cache-hit token 总量，最大相对误差约 7.12%。

4. **物理 copy 校准**
   - torch/CUDA H2D、D2H、GPU0-to-GPU1 P2P microbench。
   - recovery-shaped HostBackup restore 和 GPUReplica restore action microbench。
   - Fig. 15b 已整理为 copy latency 曲线：8192-token KV 下 GPUReplica P2P restore 约 4.33 ms，Host H2D restore 约 48.43 ms，Host D2H backup 约 46.52 ms。
   - Fig. 15c 检查 `recovery_model.py` 的 `physical_copy` 模式：模型读取同一个 physical recovery aggregate，先复现实测 copy component，再额外加入 Host/GPUReplica setup cost；当前最大相对误差为浮点舍入量级。
   - 这些结果用于校准 SGLSim recovery model；它们不是完整 online policy recovery。

## 7. 当前不能声称的内容

当前还不能把所有定量图称为 strict-final HPCA evidence。尤其不能声称：

- SGLSim 已经执行真实 CUDA KV tensor copy。
- GPUReplica 的最终收益已经由 end-to-end physical P2P recovery 证明。
- 所有 policy ranking 都已经在真实 SGLang 上复现。
- preliminary metadata-materialization rows 等价于 final recovery evidence。

当前最准确的说法是：AegisKV 已经具备完整建模、online SGLSim 实验框架、failure replay 机制、部分 strict-final SGLSim 结果，以及小规模 SGLang/torch 校准证据；完整论文还需要把剩余 preliminary 图替换为 physical recovery 或明确验证过的 real-system recovery 结果。
