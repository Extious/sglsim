# One Pool, Two Caches 的建模方式

更新时间：2026-07-01

本文档单独整理 `One Pool, Two Caches: Adaptive HBM Partitioning for Accelerating Generative Recommender Serving` 的建模方式，供 AegisKV 建模章节参考其叙述逻辑与符号组织。

## 1. 问题定义

`One Pool, Two Caches` 的问题是：Generative Recommender serving 中，同一块 GPU HBM 同时服务两个 cache：

- EMB cache：缓存 hot embedding，减少 CPU/remote memory 到 GPU 的 H2D 传输。
- KV cache：缓存 transformer attention KV，减少长序列 recomputation。

二者共享 HBM，因此给 EMB 多一点容量会减少 embedding miss，但会压缩 KV cache；给 KV 多一点容量会减少 recompute，但会增加 embedding H2D。论文把这个问题建模为一个在线 HBM partitioning 控制问题。

## 2. 资源池与 allocation ratio

论文设 GPU HBM 总预算为 $M$，用 $\alpha_t \in [0,1]$ 表示第 $t$ 个 decision epoch 中分给 EMB cache 的比例。于是：

$$
M_{\mathrm{EMB}}(t) = \alpha_t M
$$

$$
M_{\mathrm{KV}}(t) = (1 - \alpha_t) M
$$

在给定 runtime state $s_t$ 时，目标是选择一个最优比例 $\alpha_t^{*}$，使端到端 P99 latency 最小：

$$
\alpha_t^{*}
=
\arg\min_{\alpha \in [0,1]}
\ell(\alpha, s_t)
$$

其中 $\ell(\alpha, s_t)$ 表示在状态 $s_t$ 和 allocation ratio $\alpha$ 下的端到端 P99 latency。

## 3. MDP 形式

论文把动态 allocation 建模为 Markov Decision Process：

$$
(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

每个 epoch 中，controller 观察状态 $s_t \in \mathcal{S}$，选择动作 $a_t \in \mathcal{A}$，系统转移到 $s_{t+1}$，并从后续 serving window 的表现得到 reward $r_t$。优化目标是最大化 discounted return：

$$
J(\pi)
=
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{\infty}
\gamma^t r_t
\right]
$$

这个建模的关键点是：调整 $\alpha_t$ 的效果不是立刻显现的。它会先影响 cache refill、eviction 和未来 reuse，然后才反映到后续 window 的 P99 latency 和 SLO satisfaction。

## 4. State vector

One Pool 使用 7 维 state vector：

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

各分量含义是：

- $h_t$：hot-user request fraction。
- $k_t$：最近 window 的 KV cache hit rate。
- $e_t$：最近 window 的 EMB cache hit rate。
- $L_t$：平均 user interaction sequence length。
- $v_t$：latency SLO violation fraction。
- $\alpha_t$：当前 EMB/KV allocation ratio。
- $p_t$：归一化后的 P99 end-to-end latency。

这些状态覆盖 workload、cache、当前 allocation 和 tail latency，因此比单一 hit-rate heuristic 更适合做在线控制。

## 5. Action 与 clip 更新

动作不是直接指定新的 $\alpha$，而是给当前比例一个小步长增量：

$$
a_t \equiv \Delta \alpha_t
$$

论文使用离散动作集合：

$$
\mathcal{A}
=
\{-0.06,\,-0.04,\,-0.02,\,0,\,0.02,\,0.04,\,0.06\}
$$

下一轮 allocation ratio 用 clip 约束在可行范围内：

$$
\alpha_{t+1}
=
\operatorname{clip}
\left(
\alpha_t + \Delta \alpha_t,\,
\alpha_{\min},\,
\alpha_{\max}
\right)
$$

这种小步长更新是为了避免 live serving 中突然搬迁大量 cache，引发 H2D refill 和 P99 spike。

## 6. Reward

One Pool 的 reward 同时考虑 SLO satisfaction 和 P99 violation penalty：

$$
r_t
=
\mathbb{E}_{\tau \sim \mathcal{T}_t}
\left[
\mathbf{1}
\left[
d_{\tau} \le \tau_{\mathrm{SLO}}
\right]
\right]
-
\lambda
\max
\left(
0,\,
\ell_t - \tau_{\mathrm{SLO}}
\right)
$$

其中：

- $\mathcal{T}_t$ 是当前 decision window 的 request 集合。
- $d_{\tau}$ 是 request $\tau$ 的 latency。
- $\tau_{\mathrm{SLO}}$ 是 SLO threshold。
- $\ell_t$ 是当前 window 的 P99 latency。
- 第一项奖励满足 SLO 的请求比例。
- 第二项惩罚 P99 超过 SLO 的尾延迟。

论文设置：

$$
\lambda
=
\frac{1}{\tau_{\mathrm{SLO}}}
$$

这样 SLO satisfaction 和 tail violation penalty 处在可比尺度上。

## 7. Controller、非侵入式调整与 routing

One Pool 的 SmartAllocator 不是单独一个 policy，而是三层控制：

- frozen PPO base policy：从 state 输出主要 allocation action。
- online residual adapter：修正 offline training 和 online workload 的偏差。
- burst recovery controller：检测 P99 burst 后用保守策略覆盖普通 action。

正常情况下，新比例为：

$$
\alpha_{t+1}
=
\operatorname{clip}
\left(
\alpha_t
+ \Delta \alpha_{\mathrm{ppo}}
+ \Delta \alpha_{\mathrm{adapt}},\,
\alpha_{\min},\,
\alpha_{\max}
\right)
$$

执行层面，论文强调 allocation update 必须非侵入式：

- logical boundary update 是 metadata-only，不搬迁 resident KV block。
- 新增 EMB 空间需要 refill，但 refill 走 background H2D，并受 bandwidth throttling 控制。
- 目标是把 allocation control 从 serving critical path 中解耦出来。

同时，由于不同 node 的 $\alpha$ 不同，请求路由也必须同时考虑 KV locality、EMB locality 和 load。论文使用 routing score：

$$
\operatorname{score}(n)
=
w_{\mathrm{kv}} h_{\mathrm{kv}}(n)
+ w_{\mathrm{emb}} h_{\mathrm{emb}}(n)
+ w_{\mathrm{ld}}
\left(
1 - l(n)
\right)
+ b(n)
$$

其中：

$$
b(n)
=
\epsilon \cdot \mathbf{1}
\left[
n = n_{\mathrm{aff}}
\right]
$$

这里 $h_{\mathrm{kv}}(n)$ 表示 node $n$ 是否有该 user 的 KV，$h_{\mathrm{emb}}(n)$ 表示该 node 对 hot embedding shard 的覆盖率，$l(n)$ 是 load pressure，$b(n)$ 是 affinity bonus。权重由 miss cost 归一化得到：

$$
w_i
=
\frac{C_i}
{C_{\mathrm{kv}} + C_{\mathrm{emb}} + C_{\mathrm{ld}}}
$$
