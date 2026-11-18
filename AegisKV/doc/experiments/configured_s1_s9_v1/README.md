# AegisKV S1–S9 三层配置实验套件

本目录把当前 S1–S9 实验统一成三个互不混杂的 JSON 层：

- `request.json`：请求工作负载、并发/上下文/reuse 等请求轴；
- `failure.json`：故障域、故障率、repair/带宽故障敏感性轴；
- `deployment.json`：模型布局、DP/节点拓扑、算法/基线，以及带审计链的输入数据。

`configs/suite.json` 冻结 current-active v3.13 九组的总量：843 个名义 cell，S8 与 S1 复用 60 个，所以为 783 个去重 cell。每组的矩阵维度会由 `config.py` 重新计算；同一个轴不能同时属于两个层。S3 的两个非对称子矩阵会分别记录公式和 cell 数。

## 运行

```bash
python AegisKV/experiments/configured_s1_s9_v1/run.py --validate-only
python AegisKV/experiments/configured_s1_s9_v1/run.py
```

第二条命令会创建不可覆盖的时间戳结果目录，保存三层配置快照、输入/配置/图像 SHA-256、S1–S9 的 PNG/PDF，以及一张 3×3 总览图。它当前的执行模式是 `consume_frozen_audited_result_and_render`：消费项目中已有且通过输入审计的 current-active v3.13 SGLSim 结果，重新绘图；这不是一次新的 783-cell DES 重跑。

## 证据边界

- S1–S8 是 SGLSim 元数据 DES 的 preliminary modeled 结果，不是 A100 物理结果，`paper_ready=false`。
- S3 已包含 Qwen3-8B 的 108/108 cells 与 Qwen3-32B 的 300/300 cells；32B service profile 和 failover delay 是 analytic modeled 值，不是 A100 物理测量。
- S6 已包含 `reuse_fraction=1.0`，共 60 个 modeled cells；仍需真实 prefix-identity workload 的外部验证。
- S9 是独立 RTX4090 物理 holdout 与冻结拟合曲线的对比；16 个 cell 中 15 个通过 10% MAPE 门限，整体门限为 `FAIL`，且它不是 WCT 结果。

所有图都会在标题、页脚或图内保留这些限制；运行器不会把 `MEASURED` CSV 行误称为物理 A100 测量，也不会把 calibration failure 删除后重画。

## Mooncake 会话参照

套件记录了 `01a005a8-95b8-7c41-af1b-50d06b711944` 作为实现责任和物理 pilot 证据边界的参照。该会话中的单节点 RTX4090 DP=2 与两节点 PP=2 pilot 不会混入 SGLSim S1–S8，也不会混入 calibration-only 的 S9。若会话中的历史设计与当前设计冲突，以 `source_owned_independent_hbm_dram_v2` 为准：HBM/DRAM 独立准入，不恢复 Peer-HBM 依赖 DRAM、target-owned writeback 或 background destage。
