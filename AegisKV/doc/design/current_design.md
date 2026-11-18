# AegisKV：System Design

> 本文档是 AegisKV System Design 的唯一设计来源。论文正文、三段伪代码和 System Overview 图必须与本文档同步。

## 1. System Overview

AegisKV 用 workflow Slack 连接正常期 KV 保护与故障期恢复。Store Master 根据 model、parallel layout 和 failure domain，只向控制面返回能够放置完整 KV chunk 的 DRAM 或 Peer-HBM pools；控制面随后在这些可行位置中作在线选择。

系统由三层组成：

| 层次 | 模块 | 主要状态与职责 |
|---|---|---|
| Application Layer | Workflow Runtime | GBDT 预测残余 DAG 中 agent 的剩余时间；CPM 计算当前 branch Slack |
| Serving Control Plane | Router / Conductor | route version、source Worker、兼容 failover target、故障 fencing 和已提交 output position |
| Serving Control Plane | Aegis Controller | Store Master / KV Directory、**SlackLandlord** 和 **HBMQuotaAIMD** |
| Worker Layer | LLM Worker | seal KV chunk、执行传输和恢复，并实施本地 HBM reserve 与 token bucket |

当前 System Overview 的唯一可编辑源图为 [`aegiskv-paper-system-design-v2.drawio`](../figures/aegiskv-paper-system-design-v2.drawio)。

### 1.1 Application: online Slack update

每当 LLM request 或 tool call 返回时，Workflow Runtime 根据当前已经观察到的 workflow 进度重新预测 agent 的剩余时间，并在从当前 active frontier 到下一 join 的残余 DAG 上运行 CPM。对 active branch $i$，

$$
s_i=[LS_i-ES_i]^+,
$$

其中 $ES_i$ 和 $LS_i$ 分别是 CPM 给出的 earliest start 与 latest start。对单层 fan-out/join，该式退化为 $[\max_j\hat T_j-\hat T_i]^+$。Algorithm 1 的 LaTeX 源为 [`slack-update.tex`](../paper/algorithms/slack-update.tex)。伪代码中，$G_t=(V_t,E_t)$ 是因果可见的残余 DAG，$H_t$ 是增量维护的执行摘要，$\mathcal N_t\subseteq V_t$ 是其中的 agent nodes，$\mathcal A_t\subseteq\mathcal N_t$ 是当前 active frontier。$\mathcal V_t^{\mathrm{start}}$ 与 $\mathcal V_t^{\mathrm{end}}$ 分别表示没有 predecessor 的 start nodes 和没有 successor 的 end nodes。对 active node，$\hat w_v$ 是预测剩余时间；对已知但尚未启动的 node，它是预测完整时间；fork、join 等控制节点的 $\hat w_v=0$。`UpdateSlack` 按以下顺序运行：

1. **Observe（Line 1）**：把刚返回的 request 或 tool event 写入当前已经观察到的 workflow state，并得到残余 DAG、agent-node set $\mathcal N_t$ 与 active frontier $\mathcal A_t$；不读取未来输出长度、完成时间或故障位置。
2. **Order（Line 2）**：生成拓扑序 $\pi$，并由 predecessor/successor 是否为空构造 $\mathcal V_t^{\mathrm{start}}$ 与 $\mathcal V_t^{\mathrm{end}}$。
3. **Predict loop（Lines 3--5）**：遍历 $V_t$；GBDT 只为 $\mathcal N_t$ 中当前可见的 agent nodes 预测 $\hat w_v$，结构节点权重为零。尚未因果可见的动态节点不进入本轮残余 DAG。
4. **Forward CPM loop（Lines 6--8）**：按拓扑序遍历每个 $v$。start node 令 $ES_v=0$；其余节点等待所有 predecessors，因此取 $\max_{u\in\operatorname{pred}(v)}(ES_u+\hat w_u)$。
5. **Backward CPM loops（Lines 9--14）**：先用所有 end nodes 的最晚预测完成时间得到 $C_t$，并逐个初始化其 $LS_v=C_t-\hat w_v$；再按逆拓扑序遍历其余 nodes，由 successors 的最小允许开始时间反推 $LS_v$。
6. **Slack and publish loop（Lines 15--17）**：遍历 active frontier，计算 $[LS_i-ES_i]^+$。若下一 join 的残余子图不完整，或 branch 缺少必要预测或 metadata，`Valid` 为 false 且 $s_i=0$；随后保存 $(s_i,t)$，并把 $s_i$ 附在该 branch 的下一次 LLM request 上。下一轮返回重新执行整个过程并覆盖旧值。

Controller 在 request 执行期间只按经过时间使用 $[s_i-(t-t_i)]^+$，不参与 GBDT 或 CPM 计算。

**Correctness boundary.** 在 $G_t$ 为 DAG 且 $\hat w_v\ge0$ 的前提下，Lines 6--14 是标准 CPM recurrence，因此 Line 16 得到的是预测残余 DAG 上的 exact total float：预测关键路径为零，其他 branch 非负。它不保证真实未来 Slack；未知结构、缺失预测和预测误差分别通过 zero-Slack fail-close 或后续 request return 重新计算处理，并且都不能绕过后两种算法的资源与提交检查。

**Complexity.** 令 $N=|\mathcal N_t|$，每个 agent 使用 $F$ 个特征，GBDT 含 $M$ 棵、最大深度为 $d$ 的树。一次更新的特征构造与预测为 $O(N(F+Md))$；拓扑排序、forward CPM 和 backward CPM 合计为 $O(|V_t|+|E_t|)$。因此总时间为

$$
O\!\left(N(F+Md)+|V_t|+|E_t|\right).
$$

固定特征 schema 与 GBDT 后，在线复杂度为 $O(|V_t|+|E_t|)$；除固定模型外，DAG、拓扑序、$ES/LS$ 和 Slack state 共占用 $O(|V_t|+|E_t|)$ 空间。该开销只在 request 或 tool return 时产生，而不是每个 token 触发。

Router 为每个 route version 绑定 model、KV ABI、parallel layout 和一个兼容 failover target。Peer-HBM copy 只建立在该 target。若 bound target 先于 source 失效，Router 保持 source ownership、output fence 和 route version 不变，并原子地把 standby identity 轮换到另一个健康且兼容的 failure domain；没有替代者时显式记录 degraded route。该轮换不把旧 target 的 Peer HBM 重新标记为新 target 所有，仍在执行的 source connector 对旧 target 的后续 HBM copy 失败后继续评估独立的直接 DRAM 准入。若 source 随后失效，Router 才推进 route version 并激活轮换后的 target；旧 Peer HBM 不再有效，但 source failure domain 之外已经提交的 DRAM copy 可被新 target 使用。

### 1.2 Normal path

1. Worker 把 KV 按 chunk 封装，并在 chunk 关闭后将其全部 TP/PP shards 报告给 KV Directory。Store Master 分别构造 Router-fixed Peer-HBM candidate 与直接 Store DRAM candidate；两个 candidate 使用各自 pool 的容量、rent、victim frontier 和 pending-copy 状态。
2. **SlackLandlord** 对 HBM 与 DRAM 分别返回本层的 `ADMIT_HBM`/`BYPASS` 与 `ADMIT_DRAM`/`BYPASS`。两个结果不互为前提；组合后必须支持 `HBM_ONLY`、`DRAM_ONLY`、`HBM_AND_DRAM` 和两层均未接纳的 `BYPASS`。`ADMIT_DRAM` 是正常一级路径，不是 HBM 失败后的异常兜底。
3. Router 先固定兼容的 Peer-HBM target。HBM 获准时，Source Worker 从仍存活的生产 buffers 发起 D2D/P2P copy 到该 target 的最终 HBM slots；DRAM 获准时，同一个 Source Worker 从生产 buffers 发起 D2H，直接提交到 Store DRAM 的最终空间。两条路径可以并行或分别执行，各自独立确认；不存在 target HBM 到 DRAM 的传输、任务登记或责任接管。
4. 每条路径先创建自己的 `PENDING` record。只有该路径的全部 shards、route version、model revision、KV ABI 和 layout 都匹配，且物理传输成功，才进入 `COMMITTED`。一条路径失败只把自己的 record 变为 `ABORTED` 并释放对应空间，不终止另一条已经提交或仍可从源 buffers 执行的路径。
5. 同一 ordinal 仅 HBM 提交时为 `HBM_ONLY`，仅 DRAM 提交时为 `DRAM_ONLY`，两者均提交时为 `HBM_AND_DRAM`。这些状态来自两条独立源侧传输的组合，而不是 HBM 到 DRAM 的状态迁移。Slack 更新只修改对应 pool 中该 request 当前 victim frontier 的 metadata key，不触发 HBM/DRAM copy 或迁移。
6. 若绑定的 HBM target 先于 source 失效，source 仍保留执行权和输出权；Router 在不改变 route version 的前提下选择兼容 replacement。旧 target 的 HBM record 不会被重新标成新 target 所有，已提交的独立 DRAM record 不受影响。后续 chunk 对新 target 重新进行本层 HBM 准入，同时照常独立评估 DRAM。
7. 请求输出完成后，Router 调用 `CompleteRequest`。Controller 在两条路径各自的 `PENDING` record 到达 `COMMITTED` 或 `ABORTED` 前拒绝最终回收，之后回收 Controller entries、Store keys 与 Peer-HBM slots。**HBMQuotaAIMD** 只根据每个 target 的 HBM pressure在窗口边界更新并发布 Peer-HBM 比例 $\alpha$；传输排队、速率和 token 仍由 Transfer Engine 管理。

### 1.3 Failure path

Router fence source、冻结输出并激活当前健康的 route-bound target；若原 target 已先失效，则使用正常期已经原子轮换的 replacement。Controller 在 Rebind 中比较上一版 designated HBM pool 的 failure domain 与新 source：匹配时该 pool 正是恢复 source，保留其 committed HBM 直到 recovery completion；不匹配时 abort 旧 pending HBM，并在计算 recovery harm 和排序前回收旧 target 上不可用的 committed HBM metadata，同时保持独立 committed DRAM record 不变。Aegis Controller 随后冻结剩余已提交 KV state，为每个 ready request 选择 Peer HBM、DRAM 或完整重算路径，并按 remaining-Slack due time 排列恢复。`HBM_ONLY` 与 `HBM_AND_DRAM` 只有在其原 route-bound target 仍存活时才可批量加载；`DRAM_ONLY` 从共享 Store 分批加载。Worker 只采用完整且版本兼容的连续 prefix，silent replay 到 Router 已提交的 output position；所有 ranks 完成后 Router 才恢复输出权限。若 source 在某条路径提交前失效，该路径的 `PENDING` record 安全终止；恢复只使用另一条已经 `COMMITTED` 的路径，否则重算。严格审计把 standby rotation、failure fence、两条独立 copy completion、目标 pool、实际 load bytes、恢复 activation 与 silent replay 绑定到同一请求的连续 route versions，不把旧 target HBM、未确认 copy 或无关共享 cache hit 当作恢复证据。

### 1.4 Controller state

KV Directory 持久化 chunk location、residency、source failure domain、route version、commit state 和资源占用。SlackLandlord 为 HBM 与 DRAM 分别维护 pool-local rent、每请求 suffix queue、每 pool 全局 lazy frontier heap 与 placement queue；HBMQuotaAIMD 的 $\alpha$ 与 stable-window counter 属于可重建 policy state。同一请求在一个 pool 的全局 heap 中最多暴露一个当前可安全释放的连续 suffix tail，其余 chunks 不参与该 pool 的全局排序。KV bytes 由 Source Worker 分别直接传到 Peer HBM 与 Store DRAM，不经过 Controller，也不在两个保护层之间转传。

## 2. SlackLandlord：Slack-Aware KV Chunk Management

核心 KV 管理算法命名为 **SlackLandlord**，LaTeX 源为 [`slack-landlord.tex`](../paper/algorithms/slack-landlord.tex)。它保留 Landlord 的 credit/rent 机制，但只对 KV chunks、storage pools 和 recovery harm 建模。

### 2.1 KV chunk and feasible state

请求 $i$ 的 KV 按 token 顺序切成 chunks。默认 prefix chunk 为 4096 tokens，decode chunk 为 256 tokens：

$$
q_{\mathrm{prefix}}=4096,\qquad q_{\mathrm{decode}}=256.
$$

这两个值是可配置的实现参数。算法的容量约束始终使用 chunk 的真实字节数 $b_c$；$b_c$ 由实际 token 数、model、KV dtype 和 parallel layout 决定。最后一个 prefix chunk 可以更短；尚未达到 256 tokens 的 open decode tail 不参与保护，达到边界后才 seal。一个 sealed chunk $c$ 包含恢复该 token interval 所需的全部 TP/PP shards，所有 shards ACK 后才能提交。

算法只区分 DRAM 和 Peer HBM。DRAM 的 segment 选择留在 Store Master 内部；Peer-HBM 的物理 Worker 已由 Router 为请求固定。因此 Algorithm 2 的输入 $x=(c,r)$ 已经包含目标 $r$，不再枚举 pool。Store Master 仍检查 model、KV format、layout 和 failure domain。Peer-HBM 的容量、rent 和 heap 属于 Router 选定的目标 Worker，不能与其他 Workers 的 HBM 容量合并。

请求 $i$ 独立维护 DRAM frontier $f_i^D$ 和 Peer-HBM frontier $f_i^H$。每层的新 admission 只能推进下一个 ordinal，淘汰只能从该层当前可安全释放的连续 suffix tail 开始，所以两层各自保持连续 prefix，但不要求 frontier 相等。一个 ordinal 的两个 committed records 导出三种 residency：只有 HBM 为 `HBM_ONLY`，两者都有为 `HBM_AND_DRAM`，只有 DRAM 为 `DRAM_ONLY`。HBM capacity pressure 可以直接删除低价值的 `HBM_ONLY` 或 `HBM_AND_DRAM` 安全尾部；删除的是备份副本，不要求 DRAM 已提交，后续故障缺少其他副本时相应区间重算。DRAM pool 只按自己的 frontier 和价值选择 DRAM victim，不与 HBM 做隐式级联。SSD 仍由 Store cache policy 管理。

HBM candidate 的 `Gain` 只表示该 HBM 路径的即时恢复收益，不扣除 DRAM 路径的未来成本；DRAM candidate 也只使用其直接 D2H 路径的收益与成本。两类 candidate 的 `base_hbm_chunk_id` 必须为空、兼容字段 `writeback_cost_ms` 必须为零。`base_dram_chunk_id` 与 `dependent_tier_recovery_ms` 已从当前 schema 删除；包含跨层前置依赖或 HBM 到 DRAM 传输的旧 manifest 和结果不属于当前设计证据。

### 2.2 Slack-aware recovery harm

对 pool $r$ 的候选评估，Controller 使用该层当前连续保护前缀得到立即故障时的恢复时间 $T_i^r(S_i^r)$；它包含该层 load、未保护 suffix 的重算以及 silent replay。信息不足时 Slack 取零。该层的 Slack-visible recovery harm 为

$$
J_i^r(S_i^r,t)=[T_i^r(S_i^r)-s_i(t)]^+.
$$

候选 $x=(c,r)$ 的 admission credit 是其减少的 recovery harm：

$$
A_x(t)=[J_i^r(S_i^r,t)-J_i^r(S_i^r\oplus x,t)]^+.
$$

`Gain(x,t)` 对 HBM 与 DRAM candidate 都返回各自在目标 pool 中带来的 $A_x$，不让另一层已经存在的副本把本层收益遮蔽，也不把另一层的未来传输成本并入本层准入。最后按 candidate 的真实字节数 $b_x$ 归一化。`Loss(Y,t)` 使用同一层的立即故障 harm 比较删除 $Y$ 前后差值并返回 joint loss $D_Y$。删除按各 tier 的 suffix 顺序执行，不做隐式 cross-tier cascade。

### 2.3 Landlord credit and rent

每个 pool $r$ 维护一个累计 rent $L_r$ 和一个 min-heap $Q_r$。每个 resident chunk 都保留其 Landlord key，但 $Q_r$ 对每个请求最多只暴露一个当前可安全淘汰的 suffix tail。candidate $c$ 在 rent $L_r'$ 下提交时获得

$$
H_c=L_r'+\frac{A_x}{b_c}.
$$

$A_x/b_c$ 表示该 chunk 每字节保护的 recovery harm；$L_r$ 是该 pool 已经收取的 rent，因而 resident 剩余 credit 为 $b_y[H_y-L_r]^+$。较新的 chunk 使旧 chunk 不再是 tail 时，旧 key 仍保存在 KV Directory；它再次成为 tail 时才重新进入 $Q_r$，不能按当前 rent 重新发放完整 credit。当 Slack 或 frontier 改变使当前 tail 的 deletion loss 从 $D^{old}$ 变为 $D^{new}$ 时，Controller 更新

$$
H_y\leftarrow\max\!\left(L_r,
H_y+\frac{D^{new}-D^{old}}{b_y}\right).
$$

一次 Slack update 只修改该请求当前 tail 的 indexed-heap entry；隐藏 chunk 保存旧 key 和 deletion-loss 版本，在重新成为 tail 时刷新。因此一次更新为 $O(\log R)$，不扫描该请求或 pool 的所有 chunks。pop 时仍会刷新 heap 顶部，直到它在当前状态下保持最小。

候选 $x$ 的 Landlord threshold 为

$$
\Theta_x=L_r+\frac{A_x}{b_x}.
$$

算法从 Store Master 读取目标 $r$ 当前对 placement 暴露的空闲空间 $f=\operatorname{Free}(r)$。只要 $f<b_x$，算法就从最小 $H_y$ 开始暂定淘汰 suffix chunks；每选择一个 $y$，令 $f\leftarrow f+b_y$。被检查的 tail 必须可安全删除，并排除 candidate 所属请求，避免删除新 chunk 的 prefix prerequisite。只有当所有被选 keys 都小于 $\Theta_x$，且 $A_x>D_Y$ 时，新 chunk 才能进入。通过后 $Y$ 立即淘汰，rent 推进到 $\max_{y\in Y}H_y$，释放的空间直接分配给处于 `PENDING` 状态的新 chunk。传输失败不恢复 $Y$。默认配置下，一个完整 4096-token prefix chunk 最多可能需要约 16 个 256-token decode chunks 的空间，因此淘汰结果是集合 $Y$，而不是单个被替换 chunk。

### 2.4 Runtime flow

Algorithm 2 只描述正常期 KV 管理。它对一个 sealed logical chunk 返回明确 tier 结果，并把每个 pool 的 Landlord 选择封装在 `TryAdmit` 中：

- **Lines 1--5：independent tier decision。** Algorithm 构造 Router-fixed HBM candidate 与直接 DRAM candidate，并分别在两个 pool 调用 `TryAdmit`。无论 HBM 是否获准，DRAM 都是独立的正常分支；组合结果为 `HBM_ONLY`、`DRAM_ONLY`、`HBM_AND_DRAM` 或 `BYPASS`。任一路径传输失败只终止该路径，不改变另一条路径的结果。
- **Lines 6--7：initialize one pool。** $x=(c,r)$ 表示把 $b_c$ bytes 的 candidate 放入已确定的目标 $r$。Line 6 计算本层 gain $A$、threshold $\Theta=L_r+A/b_c$，并建立该 pool 的临时 heap 视图 $Q$；Line 7 读取 $f=\operatorname{Free}(r)$。HBM 和 DRAM 各有独立的 $L_r$、per-request tail queue 与 $Q_r$。
- **Lines 8--13：build $Y$。** 只要 headroom $f<b_c$，就调用 `MinTail`。若没有合法 tail，或最小 key 已达到 $\Theta$，本层返回 false。选中 $y$ 后，Line 11 把它释放的真实 bytes 加到 $f$；Line 12 用 `PrevTail` 找到同一请求在删除 $Y$ 后新暴露的 suffix tail，刷新 key 后插入 $Q$。因此全局 heap 始终只包含每请求当前 tail，同一请求的其余 chunks 不参与全局排序。
- **Lines 19--23：`MinTail`。** Line 20 只检查当前 pool heap top。若 key 的 Slack/frontier 版本过期，Line 21 用当前 $Y$ 和 $t$ 重算 deletion-loss key；Line 22 弹出稳定最小 key，并跳过 candidate 自己的请求 $i$。$Q_r$ 按不变式只暴露 Store Master 已判定满足 layout、residency 和 suffix 安全条件的当前 tail。
- **Lines 14--18：admit and source copy。** Line 14 计算 $Y$ 的 joint loss，只接受 $A>D$。Lines 15--16 淘汰 $Y$、推进 rent、分配该层最终空间并创建 `PENDING` entry。Line 17 记录路径所有者为 Source Worker；Line 18 把 HBM D2D/P2P 或 DRAM D2H copy 交给 Transfer Engine。两条 copy queue 与 ACK 相互独立。
- **Lines 24--26：completion callback。** 所有 shards ACK 且 route/layout 仍有效时，Line 25 才提交 chunk 并发布该 pool 的 request-tail frontier。两类 candidate 都没有跨层 base 前置条件。失败时 Line 26 只把本层 candidate 改为 `ABORTED` 并释放本层空间；已淘汰的 $Y$ 不回滚，另一层 record 不受影响。

同一 pool 的 `SlackLandlord` 调用和容量更新由 Store Master 的 metadata queue 依次处理，因此一次调用只使用 Line 2 读取的 free-space snapshot；新容量从下一次调用开始生效。多个请求不会同时使用同一份 free-space 或 heap snapshot，不同 pools 可以并行。该顺序化只覆盖 Lines 1--11 的 metadata 计算与更新，不等待 KV 传输，也不提供 old/new 副本事务。

例如，prefix candidate 占 4096-token 的 bytes，而 pool 只剩 512-token 的空间；算法可能依次选择若干低-credit decode tails，直到总空间达到 4096 tokens 对应的 bytes。若 candidate 减少 1.8 s recovery harm，而 $Y$ 的 joint loss 为 0.6 s，则净收益为 1.2 s；反之直接 bypass。

### 2.5 Correctness and cost

SlackLandlord 只把 Store Master 判定可行、全部 shards 完整且版本有效的连续 chunks 标记为 `COMMITTED`。只有目标 HBM slots 与 HBM Controller commit 同时成立才可从 HBM 恢复；只有 Store DRAM bytes 与 DRAM Controller commit 同时成立才可从 DRAM 恢复，`PENDING` candidate 永远没有恢复价值。算法不保证 replacement 窗口中的冗余连续：$Y$ 淘汰后、$x$ 完成前发生故障时，恢复只使用当时真实 `COMMITTED` tier，并回退到其他 committed tier 或重算。credit/rent 不能绕过物理容量、suffix safety 或当前 $\alpha$ cap。算法不声称完整多层问题的 global optimum 或未经证明的 competitive ratio。

令 $R$ 为目标 pool 中的 active requests 数，$k=|Y|$ 为选中的 chunks 数，$u$ 为本次调用中的 stale-key refresh 次数，$m$ 为 chunk shard 数。`MinTail` 的每次 pop/update 和 `PrevTail` 的 insert 都为 $O(\log(R+1))$；$c$ 所属请求在临时视图中最多被跳过一次。`Gain` 由 frontier recovery estimator 常数时间计算；`Loss(Y,t)` 按受影响请求合并 $k$ 个 suffix changes，为 $O(k)$；安装和提交最多处理 $m$ 个 shard records。因此一次 Algorithm 2 的 Controller 时间为

$$
O\!\left((k+u+1)\log(R+1)+k+m\right).
$$

当 $R>1$ 时可简写为 $O((k+u)\log R+m)$。一个 key 在同一 Slack/frontier epoch 中最多刷新一次，因此保守的单次最坏情形是 $u=O(R+k)$；默认 4096/256 granularity 且 model/layout 相同时，纯 decode-tail 情形有 $k\le16$。$Q_r$ 持久占用 $O(R)$，临时 heap 视图只记录 $O(k+u)$ 个变更，所有 resident chunks 的 key 和版本在 KV Directory 中占用 $O(C)$ metadata。KV copy 由 Worker 异步执行，不计入上述 Controller 时间。

这些操作由 chunk seal、Slack update、commit 或 capacity change 触发，而不是每个 token 触发。Mooncake-style Store Master 只需要在 segment allocation、batch-put completion 和 eviction 前调用该 policy；KV bytes 仍由 Transfer Engine 直接移动。

## 3. HBMQuotaAIMD：Peer-HBM $\alpha$ Control

**HBMQuotaAIMD** 的 LaTeX 源目前沿用兼容文件名 [`guarded-aimd.tex`](../paper/algorithms/guarded-aimd.tex)。它只更新每个 Router-fixed Peer-HBM target $p$ 的保护比例 $\alpha_p$；它不选择 chunks，不做 Algorithm 2 的 admission，也不管理传输路径或 token。

令 $M_p$ 为 target HBM 容量。Algorithm 3 发布 $\alpha_p$ 后，该 target 的 shadow-KV quota 为 $\alpha_pM_p$。HBM pressure $q_p=1$ 表示 primary headroom 不足或出现 allocation/reclaim stall；否则 $q_p=0$。压力出现时，$\alpha_p$ 立即乘性下降；连续 $K$ 个稳定窗口后才加性增加：

$$
(\alpha_p',z_p')=
\begin{cases}
(\max\{\alpha_p^{min},\gamma\alpha_p\},0), & q_p=1,\\
(\min\{\alpha_p^{max},\alpha_p+\delta\},0), & q_p=0\ \land\ z_p+1\ge K,\\
(\alpha_p,z_p+1), & \text{otherwise},
\end{cases}
$$

其中 $0<\gamma<1$，$\delta>0$。Algorithm 3 的 Lines 1--7 显式遍历 targets，对每个 target 完成一次更新并发布新 $\alpha_p$。当 $\alpha_pM_p$ 低于当前保护用量时，Worker Local Resource Enforcer 立即阻止新的 Peer-HBM admission；Store Master 再按 HBM pool 自己的两级 victim index 直接删除低价值且可安全释放的连续 suffix HBM backups，直到满足 quota。`HBM_ONLY` 与 `HBM_AND_DRAM` 都可成为 victim，不要求 DRAM 副本存在；删除后若故障时没有其他副本，对应区间重算。AIMD 不选择具体 chunk，Transfer Engine 也不参与 victim 选择，只独立负责 copy queue、rate control 和 token 回收。

若 Peer-HBM targets 数为 $P$，一次 `ControlWindow` 时间为 $O(P)$，状态为 $O(P)$。它只在控制窗口边界运行，不进入 chunk admission 或 per-token decode 路径。

## 4. Implementation Boundary and Scope

Worker 的 KV Manager 按 4096-token prefix / 256-token decode 默认粒度 seal chunks；open decode tail 和未提交 chunks 在故障后重算。Transfer Engine 执行 Peer-HBM、DRAM/SSD write 与 recovery load，并独立管理 copy queue、rate 和 token。Local Resource Enforcer 实施 $\alpha_pM_p$ quota；现有 Worker allocator 仍独立保护 primary allocation 和 serving reserve。Recovery Executor 只采用完整且版本兼容的 chunks。

首版处理单个 GPU process、GPU 或 host 的 fail-stop，并假设 Application、Router、持久化 KV Directory、网络和兼容 Worker 可用。相同 model revision、KV ABI 和 parallel layout 的 DP replica 可以恢复；PP 需要完整 stages，TP 需要完整 ranks，首版不进行跨 layout KV 转换。

AegisKV 的核心贡献不是 GBDT、CPM、Landlord 或 AIMD 本身，而是：**用同一个 request-level Slack 和 fail-now recovery model 产生正常期的 KV chunk credit 与故障期的 recovery due time；SlackLandlord 在相互独立的 DRAM pool 与当前 Peer-HBM $\alpha$ envelope 内用 credit/rent 管理 KV chunks，HBMQuotaAIMD 只用慢时标调整并强制执行该 HBM envelope。**

## 5. Current Design Risks

### 5.1 Equal-density chunks may expose a zero-credit suffix

如果同一请求的一组 chunks 长期满足相同的保护价值密度 $A_j/b_j=\rho$，且它们进入 pool 时的 rent 单调增加，则其 keys 可能满足 $H_1\le H_2\le\cdots\le H_k$。当最新 tail 的 credit 耗尽、即 $L_r=H_k$ 时，较早 chunks 的保存 key 也可能已经不高于 $L_r$。它们不会被立即删除，但会在依次成为 tail 后以零 credit 进入 heap，从而在持续容量压力下形成连续 suffix eviction，并使淘汰集中到一个请求。Algorithm 2 会在释放足够 bytes 后停止，并在每个新 tail 暴露时使用最新 Slack 重新计算 deletion loss；$A_x>D_Y$ 也禁止 recovery harm 不降的 replacement。不过，等价值密度下的长期公平性和对抗性到达序列仍是尚未解决的设计边界，暂不在当前版本引入额外机制。

### 5.2 Asynchronous replacement creates a protection gap

Algorithm 2 在决策后立即淘汰 $Y$，而新 chunk 只有传输完成后才成为 `COMMITTED`。若故障恰好发生在该窗口，系统会失去 $Y$ 原有的恢复价值，并回退到其他 tier 或重算。当前设计有意接受这一短暂性能风险：$Y$ 本身是可淘汰的容错副本，不为其 replacement 再增加 staging 或 rollback。实现必须先把 $Y$ 标记为不可恢复，再复用其物理空间；`PENDING` chunk 永远不能被 Recovery Executor 使用。该选择消除了传输期间的 pool lock 和 old/new 原子事务，但窗口长度仍受实际传输时间影响。
