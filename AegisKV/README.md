# AegisKV

AegisKV 是面向 agentic LLM workflow 的 KV-cache 容错设计。它用当前因果可见的 workflow Slack 连接正常期保护决策与故障期恢复顺序，在限制健康路径复制压力的同时，减少影响 workflow 完成时间的恢复停顿。

## 先读什么

- [doc/PROJECT_STATUS.md](doc/PROJECT_STATUS.md)：两个实现 thread 的汇总、当前实现/实验/论文状态和证据边界。
- [doc/design/current_design.md](doc/design/current_design.md)：唯一设计来源；其他文档、代码和图像与它冲突时，以这里为准。
- [experiments/README.md](experiments/README.md)：实验 lineage 索引，区分当前模拟、preliminary、blocked 和 invalidated 结果。
- [doc/paper/README.md](doc/paper/README.md)：SGLSim 侧论文草稿、构建方法和图表接入状态。

## 当前设计

- **Algorithm 1：Causal Slack**。Application 在 request/tool 返回时用 GBDT 预测当前可见 agent 的剩余时间，在残余 DAG 上运行 CPM；信息不足时 Slack 置零并 fail-close。
- **Algorithm 2：SlackLandlord**。HBM 与 DRAM 各自维护 pool-local rent、连续 suffix frontier 和 lazy heap，并分别完成准入、淘汰与提交。
- **独立保护路径**。Source Worker 分别执行 Source→Peer-HBM 的 D2D/P2P 和 Source→Store-DRAM 的 D2H。两层互不作为前置条件，结果允许 `HBM_ONLY`、`DRAM_ONLY`、`HBM_AND_DRAM` 和 `BYPASS`；不存在 target-owned HBM→DRAM writeback/destage。
- **独立提交与恢复**。每条路径各自经历 `PENDING → COMMITTED/ABORTED`；一条路径失败不终止另一条。恢复只使用完整、连续、版本兼容且已经提交的 prefix，否则重算。
- **Algorithm 3：HBMQuotaAIMD**。只根据 Peer-HBM pressure 调整每个 target 的保护比例 `alpha`。它不选择 chunk，也不管理传输速率；Transfer Engine 独立管理 copy queue、rate 和 token。

## 目录

- `experiments/`：当前可执行、三层 JSON 驱动的实验入口；脚本、配置、测试、结果和图像均在这里。
- `doc/`：设计、论文、related works、历史实验和归档材料；不再作为新实验脚本入口。

## 当前证据概览

- 当前三层 JSON S1--S9 + E1 campaign 位于 `experiments/`。正式 lineage
  `20260829T064719Z_paper_complete_v2` 完成 613/613 cells、341,772 requests，
  强审计 0 failed checks，并生成 10 张 PNG 与 10 张 PDF。它是隔离的 offline
  metadata-DES；SGLSim 主 serving 路径仍是 online real-wall-clock HTTP simulator。
  该 lineage 不是 A100 物理测量，状态为
  `PASS_WITH_PHYSICAL_CALIBRATION_BLOCKER`。
- 新 lineage 修复了上一版默认 8,192-token prefix chunk 与权威设计不一致的问题；
  S1--S8/E1 现在按 `current_design.md` 使用 4,096/256-token prefix/decode chunk，S9
  单独保留明确标注的 1,024-token transfer 微验证覆盖。设计文件本身随 lineage 快照并校验哈希。
  中间 lineage `20260829T062726Z_paper_complete_v2` 的模拟数值虽与当前版本逐字节一致，
  但其预执行研究快照残留了更早 E1 run 的动态结果数字，因此也已被当前 lineage 取代。
- E1 从 pinned 393-workflow AgentX/Weka 源文件构造公共 observed-DAG 外部有效性实验。
  20 个 paper workflow 在 10 个 seed 间无放回且零重叠；因果模型不使用未来信息，但
  安全 Slack LCB 全为 0。E1 的 paired WCT 区间跨零，因此只支持“保护路径被真实触发”，
  不支持 Algorithm 1 在真实 DAG 上已有显著收益。
- 每个请求、故障和部署参数的来源、证据等级和适用边界记录在
  `experiments/research/parameter_provenance.json`；面向审稿人的选择理由与替代方案审计见
  `experiments/research/reviewer_configuration_audit.md`。两者都会随 lineage 快照并校验哈希。
- 现行独立分层语义已经进入 SGLSim 元数据控制器、运行时适配器、输出契约和大量针对性测试。
- `sglsim_complete_v1` 已把三算法、真实 per-target HBM pool、共享 DRAM、target-first rotation、source fail-stop、output fence、worker activation、queue outage/rescheduling、committed-prefix recovery 和 silent replay 接入同一条连续时间 DES；complete-path demo 的 fail-closed audit 为 PASS。它是完整控制/故障语义模拟，不是物理 KV 传输。
- `sglsim_large_scale_v2` 已完成 3,110 个 paired policy-seed runs，并通过其 **simulation-only** 完整审计；它不建模完整 standby rotation，也不产生物理测量或执行式 WCT。
- `sglsim_wct_preliminary_v1` 提供实际 DES workflow completion time、S1–S9 图和机制分析，但明确为 `paper_ready=false`，其中若干结论仍受 modeled timing、样本量或校准 gate 限制。
- `real_weka_full_design` 的正式硬件校准 consumer 仍为 `BLOCKED_PAPER_REQUIREMENTS`：33 个 gap、`cells=[]`，不授权 replay、performance 或 paper-ready 声明。
- 旧 RTX4090/TCP `four_path_v2` 结果已经失效，只能作诊断；不得恢复为当前性能结论。
- `experiments/physical/` 已把最终模拟 lineage 固化成覆盖全部 S1--S9 + E1 非 seed 轴的
  201-cell matched-replay 静态计划，并对 4x8 A100-40GB inventory 做 fail-closed 检查。
  当前 2xA100-80GB 历史 inventory 被明确拒绝；该计划没有提交作业，也没有解除物理校准 gate。

## 证据规则

设计同步、代码实现、静态测试、metadata simulation、WCT DES、物理校准、真实 Mooncake end-to-end 实验和 paper-ready 结果是不同证据等级。一次测试 PASS、一次 LaTeX 构建或一张视觉合格图都不会自动提升其他等级。任何定量结论必须同时保留其配置、输入、seed、原始结果、审计和硬件/模拟标签。
