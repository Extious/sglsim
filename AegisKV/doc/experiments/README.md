# AegisKV experiment index

本目录同时保存现行模拟、preliminary WCT、硬件校准 consumer 和历史诊断。目录名中的 `latest`、`full` 或 `real` 不自动表示当前权威或 paper-ready；使用结果前先核对下表和该目录自己的 manifest/audit。

## 推荐入口

| 目录 | 用途 | 当前状态 | 使用规则 |
|---|---|---|---|
| `sglsim_complete_v1/` | 现行设计统一 metadata DES；三算法、独立分层、target-first rotation、source fail-stop、queue outage/rescheduling、fence/rebind/replay | complete-path demo audit PASS | 作为完整控制/故障语义模拟入口；仍是 synthetic-calibrated metadata-only，不得称为物理结果 |
| `sglsim_large_scale_v2/` | 393-workflow 大规模 metadata DES；baseline、三算法消融、并发/故障率/模型/规模/资源敏感性 | full audit PASS；simulation-only | 可使用其 audit-bound simulated estimands；不得称为 WCT 或物理结果；未建模完整 standby rotation |
| `sglsim_wct_preliminary_v1/` | 本地 DES workflow completion time；S1–S9、alpha、occupancy、reuse、topology、calibration | preliminary；`paper_ready=false` | 必须保留 modeled timing、样本量、共享样本、负结果和 calibration qualifier |
| `real_weka_full_design/` | current-design 物理 profile consumer、严格 replay gate 和 EuroSys readiness | `BLOCKED_PAPER_REQUIREMENTS`，33 gaps，`cells=[]` | 当前禁止 replay/performance claim；等待同一 lineage 的完整物理产物 |
| `real_weka_baselines/` | 与 full-design anchor 绑定的 baseline runner；包含旧诊断结果 | 旧 `four_path_v2` invalidated | 只能用新 V100/current profile 创建 immutable suite；旧数值不得进入论文 |
| `latest_design/` | 第一版 independent-tier 合成机制 study | historical prototype | 只用于机制演化；后续自审计已否定其“完整端到端实现”含义 |

## 其他目录

- `g0_g1/`、`g1_observability/`、`g2_causal_slack/`：分阶段机制与可观测性验证，按各自 README 使用。
- `real_weka_context_scale/`：真实 Weka context 合法性与映射审计。
- `real_weka_model_scale/`：模型规模与配置敏感性材料。
- `real_weka_recovery_dispatch/`：恢复 dispatch/target parallelism 研究；不应与主 full-design lineage 自动合并。
- `sglsim_complete_v1/results/`、`sglsim_large_scale_v2/results/` 与 `sglsim_wct_preliminary_v1/results/`：模拟原始/派生结果；不要为“整理目录”移动、覆盖或去重。

## 结果选择规则

1. **先看 audit，再看图。** 图像存在、视觉 PASS 或 PDF 能构建，不代表 underlying evidence PASS。
2. **同一结论只使用一个完整 lineage。** source、config、model、hardware、profile、raw result、analysis 和 figure SHA 必须连通。
3. **不跨硬件借用系数。** RTX4090、A100、V100S 或 synthetic A100-40GiB model 是不同 evidence class。
4. **不跨 estimand 改名。** fail-now harm、conditional recovery、normal-state burden 和执行式 WCT 不能互称。
5. **保留 null/negative results。** 优势不是有效性 gate，不得按结果符号筛 seed、cell 或图。
6. **不覆盖输出。** 修复 runner、输入或语义后创建新 lineage，并把旧目录标为 superseded/invalidated。

更完整的 thread 时间线、实现地图和 paper 状态见 [../PROJECT_STATUS.md](../PROJECT_STATUS.md)。
