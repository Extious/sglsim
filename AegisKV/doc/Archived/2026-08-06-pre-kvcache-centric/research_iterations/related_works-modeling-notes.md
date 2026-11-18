# AegisKV Related-Work Modeling Notes

这份笔记记录从论文问题建模中提炼出的、可进入 AegisKV formulation 的部分。它不把
任一 prior 的目标直接改名为 AegisKV，也不把 preprint、匿名稿或当前实现中的 proxy
写成已验证系统事实。

## 1. 先分清三种 “criticality”

| 量 | 代表论文 | 问题 | AegisKV 中的角色 |
|---|---|---|---|
| 功能/因果关键性 | CAIR、Sherlock | 改坏一个 agent/node 会让最终输出或 workflow 轨迹改变多少？ | 可选的语义权重 $I_u$；不能替代机器故障概率 |
| 时间关键路径 | Agentix、Pythia、Helium、Cortex | 一个 invocation 距完成还有多远、会阻塞多少下游、剩余 slack 多大？ | residual DAG 上的 $S_u$、remaining path 和 unblock pressure |
| 故障恢复价值 | DejaVu、LUMEN、Tarragon、GhostServe | KV/state 丢失后，restore/reconstruct/recompute 会增加多少 WCT？ | AegisKV 真正优化的 scenario-dependent loss reduction |

因此不能把 `critical_agent=true`、ATLAS priority、CAIR influence 或一个 failure rate
互相代换。AegisKV 的核心保护价值是：在给定 failure scenario 和恢复动作下，某个
**已提交且可用**的 segment replica 能减少多少 workflow loss。

## 2. 论文给出的关键建模组件

### Critical Agent / agentic workflow

- **CAIR (EMNLP 2025)**：对一次 agent activation 做反事实扰动，重跑下游并比较最终
  输出、局部输出和 workflow trace。可借鉴 `semantic influence`，但其 fault 是语义扰动，
  不是 worker fail-stop。
- **Sherlock (2025 preprint)**：在 workflow node 注入 fault，用下游失效范围、verifier
  成本和最后可信节点后的 rollback/re-execution 建模 vulnerability。可借鉴
  `downstream invalidation`，但其 correctness fault 不等于 KV-state loss。
- **Agentix (NSDI 2026)**：动态 agent program 不要求未来 DAG 全部已知；ATLAS 维护
  最长已观测依赖链。可作为 non-clairvoyant online state，不应直接等同于 failure impact。
- **Pythia (2026 preprint)**：学习 dominant workflow、remaining duration、unblock pressure
  和 KV lineage，并带预测置信度/fallback。其低频分支裁剪可能恰好遗漏罕见 failure path，
  所以 AegisKV scenario 集不能只由 dominant graph 构造。
- **KVFlow (2025 preprint)**：Agent Step Graph 把 future reuse distance 传播到 radix node；
  支持 segment/radix atom 粒度和 shared-prefix 去重。它的 CPU cache tier 不是 failure-domain
  replica。
- **Helium (2026 preprint)**：用 assignment、per-worker order、precedence 和 makespan 建模
  workflow serving。可借用调度外壳，但必须加入 failure state、freshness、capacity 和
  recourse，否则会退化为普通 cache-aware scheduling。
- **Cortex (SAA 2025)**：remaining SLO slack、stage selectivity 和 service time 是有用的
  priority signal；论文本身形式化较弱，只作 adjacent positioning。
- **SAGA (2026 arXiv-only)**：graph reuse、deadline urgency 和 cache state 的组合与本题邻近；
  截至 2026-07-14，其 venue/DOI 声明无法由官方 program/Crossref 核验，所以不依赖其
  theorem 或同行评审状态。

### Inference fault tolerance

- **DejaVu (ICML 2024)**：固定邻居的持续 KV streaming 和 ACK frontier。关键启发是：
  recovery 只能使用最后完整提交的 checkpoint，未提交 suffix 必须 replay。
- **FailSafe / RaidServe (2025 preprint / MLSys 2026 final title)**：HostBackup 全量底座、
  弹性 TP 和 failure-time load-aware routing。恢复成本不能只算 KV bytes，还要算 topology
  reconfiguration、survivor capacity 和 queueing。
- **Tarragon (2026 preprint)**：用 token index 与 layer frontier 表示失败点，并区分
  stateful attention worker 与 stateless expert worker；适合定义 failure-exposed work。
- **AnchorTP (DATE 2026)**：先求逻辑上的 minimum reload bytes，再安排物理 copy。
  AegisKV 同样要分离“保护带来的 loss reduction”和“带宽/overlap 实现”。
- **ReviveMoE (2026 preprint)**：metadata rollback 必须落到 generation-step consistency
  boundary；保留 token history 后才能安全 replay。
- **KevlarFlow (2026 preprint)**：固定 GPU replica ring 是直接 baseline；内存不足时丢副本并
  recompute。AegisKV 的差异是有限 HBM budget 下的 segment-level marginal value。
- **GhostServe (MLSys 2026)**：host parity 支持 reconstruct；一个 segment 的恢复代价应取
  restore、parity reconstruct、recompute 等互斥可行路径的下包络，不能把收益相加。
- **LUMEN (2026 preprint)**：明确出现三个决策点：故障前 checkpoint placement、故障时
  recovery dispatch、reload window 的 serving-capacity restoration。它是最接近的 request-level
  prior；AegisKV 需要明确 segment/shared-prefix 粒度、GPU HBM budget 和 workflow-WCT 差异。
- **Concordia (2026 preprint)**：dirty-page delta 只有在 payload 和 commit marker 完成后才
  有效。为 AegisKV 提供 committed frontier、checkpoint lag、D2H log 和 replay cost 语义。

## 3. Literature-grounded AegisKV model

### Sets and state

- workflow $w$ 的 residual graph：$D_w(t)=(V_w(t),E_w(t))$；
- KV/radix segment 集：$\mathcal N_t$，segment $n$ 属于 turn $u(n)$；
- worker/GPU 集：$\mathcal G$，failure-domain 映射 $h(g)$；
- protection location：HostBackup 和 peer GPUReplica；扩展时可加入 parity group；
- scenario 集：$\Omega_t$。

一个 scenario 不只是 “turn $u$ 失败”：

$$
\omega=(F_\omega,t_\omega,\xi_\omega),
$$

其中 $F_\omega$ 是同时失败的 worker/domain 集，$t_\omega$ 是失败时刻，$\xi_\omega$
包含当时 committed frontier、分支实现、surviving topology、带宽、queue/load 和 failure
duration。Replica 的有效概率必须是 owner 失败且目标 peer 幸存的联合事件。

### Three decision moments

1. **Budget epoch $\tau$**：慢时标选择 $\alpha_{g,\tau}$，得到
   $B_{g,\tau}=\alpha_{g,\tau}M_g$。
2. **Segment event $t$**：故障前选择 placement $X_t=\{x_{n,g,t}\}$。
3. **Failure scenario $\omega$**：选择 recourse $Y^\omega$，包括 recovery target、route，
   以及 GPU restore / host restore / parity reconstruct / suffix recompute。

Alg. 1 的 criticality analysis 是状态估计，不是第四种 placement action。

### Committed frontier and valid recovery

令 $q_{n,l}(t)$ 是 location $l$ 对 segment $n$ 最后已提交的位置/版本。恢复动作只能读：

$$
\operatorname{valid}_{n,l}^{\omega}=1
\iff
q_{n,l}(t_\omega)\text{ is committed}
\land l\notin F_\omega.
$$

如果 prefix 到 $q$ 可恢复、请求实际已推进到 $z>q$，则 $q+1{:}z$ 必须 replay。
这同时覆盖 DejaVu 的 ACK frontier、Tarragon 的 token/layer frontier、ReviveMoE 的 step
rollback 和 Concordia 的 commit marker。

### Scenario loss and tail risk

令无故障 baseline WCT 为 $T_w^0$，给定 placement/recourse 后 scenario WCT 为
$T_w^\omega(X,Y^\omega)$：

$$
\ell_w^\omega(X,Y^\omega)
=\left[T_w^\omega(X,Y^\omega)-T_w^0\right]_+.
$$

对 workflow loss 使用可退化的 mean--tail risk：

$$
\mathcal R_\beta(\ell)
=(1-\eta)\,\mathbb E_\omega[\ell]
+\eta\,\operatorname{CVaR}_\beta(\ell),
\qquad 0\le\eta\le1.
$$

$\eta=0$ 恢复当前 average failure-WCT 指标；$\eta>0$ 才表示明确保护尾部 failure
scenario，不能在没有 tail experiment 时声称已经验证。

### Two-stage placement plus recourse

HostBackup-only 是固定 baseline $X=\varnothing$。对一个 placement，failure-time recourse
先求最优可行动作：

$$
Q(X,\omega)=
\min_{Y^\omega\in\mathcal Y(X,\omega)}
\left[
\sum_w a_w\ell_w^\omega
+\lambda_r T_{\rm restore}
+\lambda_c T_{\rm recompute}
+\lambda_q T_{\rm queue/capacity}
+\lambda_t T_{\rm reconfig}
\right].
$$

placement 的 set-level 风险下降为：

$$
F_t(X)=
\mathcal R_\beta\!\left(Q(\varnothing,\omega)\right)
-\mathcal R_\beta\!\left(Q(X,\omega)\right).
$$

给定慢时标 budget，Problem B 为：

$$
X_t^*\in\arg\max_{X\in\mathcal F(\mathbf B_{\tau(t)})}
\left\{F_t(X)-C_{\rm transition}(X,X_{t-1})\right\}.
$$

这里按整个 selected set 计算 shared-prefix coverage 与 workflow WCT，避免同一份恢复收益
被多个 segment/downstream consumer 重复领取。一般情形不预设 $F_t$ 是 submodular。

### Recourse paths

对中断请求/segment $n$ 和路径
$p\in\{\text{gpu},\text{host},\text{parity},\text{recompute}\}$，令
$y_{n,p,g}^\omega\in\{0,1\}$。路径成本包含：

$$
C_{n,p,g}^{\omega}
=T_{\rm route/queue}
+T_{\rm restore/reconstruct}(q)
+T_{\rm replay}(q+1{:}z)
+T_{\rm resume}.
$$

必须满足：每个中断请求恰选一个 recovery target/path；选中的 copy 已提交且幸存；
survivor 有 HBM、prefill/decode 和带宽容量；共享 prefix 只 restore 一次；必要时完成 TP/PP/MoE
topology reconfiguration。

### Placement constraints

- 每个 peer 的 HBM budget；
- normal-path D2H/P2P bandwidth 与 copy-overlap budget；
- owner 与 replica 的 failure-domain anti-affinity；
- 每个 segment 至多一个额外 GPUReplica（当前 scope）；
- radix prefix closure、position/version validity 和 committed freshness；
- plan transition、minimum residency 和 churn/hysteresis；
- survivor capacity / queue feasibility，而不只计算传输 bytes。

## 4. Solver and claim boundary

- **Scenario oracle**：小规模 scenario set 用 MILP / branch-and-bound；对所声明的离散模型是
  exact，用来报告 online objective gap 和 WCT gap。
- **Online placement**：按“单位 unique HBM 带来的 risk-weighted set marginal reduction”做
  event-driven greedy；这是 approximation，不宣称 general optimum/submodular guarantee。
- **Failure-time dispatch**：在幸存 peer 上联合比较 route/queue 与有效 committed prefix，
  选择 GPU/host/parity/recompute 的最低可行 loss；负载过高时可放弃局部最快 restore 以降低
  workflow WCT。
- **Current implementation**：现有 SGLSim 路径仍是 cache/load heuristic 和 calibrated
  overlay，不是上面的 scenario-aware optimizer；general online residual DAG、physical live
  repartition 和 large-scale data path 仍是 integration gates。

## 5. Evaluation baselines implied by the literature

- HostBackup-only / full host checkpoint（FailSafe/RaidServe-style baseline）；
- fixed neighbor GPU streaming（DejaVu-style）；
- fixed ring GPU replica with recompute fallback（KevlarFlow-style）；
- request-level load-aware checkpoint holder/dispatch（LUMEN-style）；
- coded host parity（GhostServe-style）；
- restore-only、recompute-only、oracle min-path；
- random/LRU/size-only/static `critical_agent` placement；
- offline scenario oracle versus online greedy。

核心指标应至少报告 normal overhead、mean/p95/p99 failure-induced WCT、recovery bytes、
recompute GPU-time、committed checkpoint lag、survivor queue/capacity loss、plan churn，以及
online solver 相对 oracle 的 gap。
