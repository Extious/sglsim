# AegisKV experiments

这里是 AegisKV 当前唯一的可执行实验入口。目录布局参考
`/home/comp/zhanzhao/sglang/benchmark/multi_agent`，但把配置职责明确拆成：

完整的实验配置、10 组矩阵、结果表、图像解释和证据边界集中记录在
[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)。

- `configs/<group>/request.json`：trace、采样、到达、context 投影和 workload 轴；
- `configs/<group>/failure.json`：estimand、故障域、注入、维修、MTTF 和 failure 轴；
- `configs/<group>/deployment.json`：模型、TP/DP、A100/主存/带宽、策略和部署轴；
- `src/request/`、`src/failure/`、`src/deployment/`：三个配置层的执行代码；
- `src/simulation/`：统一调用 current-design complete metadata DES；
- `src/plot/`：只消费本次 lineage 的原始结果绘图；
- `research/`：配置取值、来源和证据边界；
- `results/`：不可覆盖的运行 lineage。

旧脚本和 preliminary 结果保留在 `../doc/experiments/`，不会被本入口读取。

## 快速使用

```bash
uv run --extra benchmark python AegisKV/experiments/run.py validate
uv run --extra benchmark python AegisKV/experiments/run.py fetch-data
uv run --extra benchmark python AegisKV/experiments/run.py all --profile smoke
uv run --extra benchmark python AegisKV/experiments/run.py all --profile paper
```

`paper` 和 `smoke` 都默认 fail-closed：TraceLab 固定发布文件缺失或哈希不匹配时不会
退化成 synthetic trace。`smoke` 仅缩小 session/round/轴数量用于快速路径验证；它仍读取
同一 TraceLab artifact，但不替代完整 `paper` 矩阵。S4/S5/S9 按配置使用显式标注的
controlled fork/join，只回答机制问题。E1 使用 pinned AgentX/Weka 全量源文件，保留所选
真实 DAG 的 request 相对时序、subagent 结构和 prefix hash；跨 workflow 到达与 Qwen3
context 是显式 controlled projection，不冒充生产固定到达率。

`paper` 的不确定性以独立 deterministic sampling seed 为统计单元，报告双侧
95% Student-t confidence interval；同一 seed 内的 workflow/session 只用于 paired
comparison，不被当成独立重复样本。当前测试边界：

```bash
uv run --extra benchmark pytest AegisKV/experiments/tests test/unit/srt/test_aegis_complete.py -q
```

## 当前已验证 lineage

- paper：`results/20260829T064719Z_paper_complete_v2/`（613/613 cells）；
- 完整结果解读：`results/20260829T064719Z_paper_complete_v2/analysis.md`；
- 审计：`results/20260829T064719Z_paper_complete_v2/audit.json`；
- S1--S9 + E1 PNG/PDF：`results/20260829T064719Z_paper_complete_v2/figures/`；
- 每图数据：`results/20260829T064719Z_paper_complete_v2/figure_sources/`。

该 lineage 的语义与复现审计为 PASS，整体状态是
`PASS_WITH_PHYSICAL_CALIBRATION_BLOCKER`，不是 `paper_ready_calibrated`。
该 lineage 共模拟 341,772 个 requests、18,376 个 workflow/session observations；
S1--S4、S6--S8 和 E1 使用 10 seeds，S5 使用 5 seeds，S9 使用 3 个公式验证 seeds。
E1 的 20 个真实 workflow 在 seed 间互不重叠；完整语料因果 interval 的 request coverage
为 98.18%、workflow-simultaneous coverage 为 92.86%，但安全 Slack LCB 全为 0。E1 的
paired AegisKV - recompute 效应为 `-70.0 ± 90.9 ms`，95% CI 跨零，不能写成确定优势。

该 lineage 按权威 `current_design.md` 使用默认 4,096-token prefix / 256-token decode
chunk；S9 的 1,024-token prefix 只服务于 analytical-versus-DES transfer 微验证。配置
validator、lineage audit 和随运行快照的设计文件 SHA-256 共同 fail-close。上一版
`20260829T055857Z_paper_complete_v2` 使用了不一致的 8,192-token 默认值，已被取代。
中间版 `20260829T062726Z_paper_complete_v2` 的结果数据与当前版相同，但其预执行研究
快照仍包含更早 E1 run 的动态结果数字，也已被当前 lineage 取代。

参数级证据链见 [`research/parameter_provenance.json`](research/parameter_provenance.json)，
配置选择与替代 trace/failure/deployment 方案的审稿人审计见
[`research/reviewer_configuration_audit.md`](research/reviewer_configuration_audit.md)。它们均随
lineage 复制到 `config_snapshot/` 并由 `manifest.json` 的 SHA-256 做 fail-closed 校验。

## 证据边界

本套件输出的是 SGLSim metadata-DES 结果，不是物理 A100 测量。A100 带宽、显存、主存和
Qwen3 结构参数有一手来源；模型 service-time 系数在获得同拓扑 A100 profile 前仍是 modeled
input。因此图可以作为模拟评估或机制验证，不能自动标记为 `physical` 或
`paper_ready_calibrated`。每个 lineage 的 `audit.json` 会保留这一 gate。

本机物理资源核查与不匹配原因见
[`research/physical_calibration_status.md`](research/physical_calibration_status.md)：现有可核验资源是
1-node 2xA100-80GB，而模拟部署是 4-node 8xA100-40GB/node，不能混作同拓扑校准。

## 物理校准与 matched replay 入口

[`physical/`](physical/README.md) 是从最终模拟 lineage 通往未来物理实验的 fail-closed
执行契约。它仍按 request/failure/deployment 三层 JSON 管理配置，并已固化一个覆盖
S1--S9 + E1 全部非 seed 轴的 201-cell、3-seed matched replay 静态计划。计划绑定最终
lineage 的 manifest/audit/summary/current-design 哈希、每个 source raw/shadow artifact、
物理配置和计划生成代码，但不包含模拟 outcome 数字，也不会自动提交作业。

当前历史 1-node 2xA100-80GB inventory 的 preflight 状态为
`BLOCKED_TOPOLOGY_OR_IDENTITY`。即使未来 4x8 A100-40GB inventory 通过，它也只授权
校准采集；在 Qwen3-8B/32B service time、四条 source-owned transfer path、failure/rebind
delay、5 次独立重复和 10% holdout-error gate 全部通过前，matched replay、性能结果和
paper-ready 标记仍为 false。
