# AegisKV Latest-Design Presentation

本目录维护 AegisKV 最新设计演示稿。`paper_outline.tex` 与
`paper_outline.pdf` 是唯一入口；核心系统被明确收敛为三个算法：

1. 应用层的固定 DAG 工作流松弛时间估计；
2. 面向 KV 块的松弛感知 L1/L2/Recompute 备份决策；
3. 结合正常负载与上一窗口备份需求的 per-node L1/L2 Backup budget controller。

## 当前关键设计

- 任务分解型（Task-decomposition）与角色专门型（Role-specialized）工作流通过
  不同工作流适配器映射为同一个带属性执行 DAG；节点表示执行片段，而不是整个智能体。
- 算法 1 假设完整工作流 DAG 在执行前已知且运行中不变；所有边等价地取
  `p_e=1`，不学习边概率。运行时只更新节点的待执行、执行中、已完成状态，
  并预测智能体执行片段的剩余时长。
- 角色反馈必须在工作流启动前展开为固定轮次；运行中新增分支、节点、边或
  无限反馈循环均不在当前问题范围内。
- `critical_score` 定义为请求所属 DAG 节点对预测 WCT 的边际敏感度。
- Router 只根据正常推理负载和近似 Radix Tree 做 Cache-Aware Top-2：第一优
  Rank 执行请求，第二优 Rank 成为该请求固定的计划 Backup Rank；Router
  不读取 Backup 分区占用。
- 算法 2 在 DP rank 内使用松弛时间、故障概率、恢复路径性能、KV 块信息与当前
  `alpha_n/beta_n` 预算，为已经固定目的 Rank 的 KV chunk 选择
  L1/L2/Recompute；它不再选择 Backup Rank。
- 算法 2 同时发布容量截断前的逐节点、逐层影子需求 `D_n1/D_n2`、
  已 ACK 实际占用 `U_n1/U_n2` 和未满足需求 `O_n1/O_n2`。
- `alpha_n` 表示 node `n` 的可管理 L1 HBM 中分给 L1 Backup 的比例；
  `beta_n` 表示可管理 L2 Host DRAM 中分给 L2 Backup 的比例。
- 剩余 L1 HBM 服务 active KV/prefix cache；剩余 L2 Host DRAM 服务正常
  L2 cache、load-back 与 prefetch。
- 算法 3 在每个 DP rank 内运行，在窗口边界结合本地正常 L1/L2 负载、服务安全边界和算法 2
  的 `D/used/overflow` 反馈，发布下一窗口的 `alpha_n`、`beta_n`。
- Backup ACK 的权威状态保留在目标 DP rank；目标端按请求聚合后异步推送
  `RecoveryHint`。Router 只缓存只读摘要，在 failure 后结合当前负载和
  Radix Tree 零查询选择预计完成时间最短的 Retry Rank。
- 明确的时序为 `算法3(k) -> 算法2(k) -> 算法3(k+1)`；算法 2 的本窗口
  反馈不追溯修改当前 plan，只影响下一窗口容量预算。
- 当前负载回放图只支持 L1 `alpha_n` 的安全余量说明；L2 `beta_n` 仍需
  matched Host-DRAM actuation sweep；完整算法 3 的需求反馈响应也仍需单独验证。

## 构建与验证

```bash
python3 scripts/validate_paper_outline.py --compile
```

直接构建：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error paper_outline.tex
```

最终交付仍写到 `paper_outline.pdf`；渲染检查的临时 PNG 放在仓库根目录
`tmp/pdfs/`，不进入本目录。
