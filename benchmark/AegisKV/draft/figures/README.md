# 图表资产

本目录保存 AegisKV 论文草稿的图表资产。

重要说明：结果图可能使用 preliminary SGLSim CSV 作为版式和内部审阅数据。重复 SGLSim replay 产生的 CSV 可以包含 run-level std/95% CI 列，部分柱状图会渲染这些区间。只有当对应 `evidence_tier` 变成 `final` 且 strict validator 通过后，它们才是最终 HPCA 定量证据。当前包只保留生成式 System overview 和动机/实验结果图，不再维护详细机制流程图。

重新生成所有图表：

```bash
python benchmark/AegisKV/draft/figures/scripts/generate_all_figures.py
```

## 已生成图表

| 图号 | 文件名 | 状态 | 最终证据来源 |
|---|---|---|---|
| Fig. 1 | `fig_motivation_equal_size_loss` | trace-derived-preliminary | SGLang/SGLSim failure injection |
| Fig. 1b | `fig_critical_backup_priority` | trace-derived-preliminary | SGLSim staged 1-3-1-3-1 backup sweep |
| Fig. 1c | `fig_critical_agent_profile` | trace-derived-preliminary | SGLSim staged 1-3-1-3-1 stage-barrier trace |
| Fig. 2 | `fig_restore_recompute_boundary` | sglsim-preliminary | SGLSim runtime replay calibrated |
| Fig. 3 | `fig_static_gpu_replica_budget_gap` | sglsim-preliminary | SGLSim runtime replay calibrated |
| Fig. 3b | `fig_alpha_workload_failure_policy` | model-based-preliminary | offline MDP/policy sweep over staged workflow states |
| Fig. 3c | `fig_online_alpha_adaptation` | sglsim-preliminary | SGLSim staged online adaptive-alpha sweep |
| Fig. 3d | `fig_staged_alpha_risk_tradeoff` | trace-derived-preliminary | SGLSim staged 1-3-1-3-1 alpha-risk counterfactual |
| Fig. 4 | `fig_system_overview` | design-ready | generated bitmap + prompt |
| Fig. 8 | `fig_overall_wct` | sglsim-preliminary | SGLSim runtime replay mixed |
| Fig. 9 | `fig_recovery_latency` | sglsim-preliminary | SGLang small + SGLSim large |
| Fig. 10 | `fig_recovery_path_breakdown` | sglsim-preliminary | SGLSim policy matrix |
| Fig. 10b | `fig_shared_prefix_marginal` | final-measured | SGLSim shared-prefix sweep |
| Fig. 11 | `fig_normal_path_overhead` | model-based-preliminary | SGLang no-failure overhead |
| Fig. 11b | `fig_hostbackup_visibility` | mixed-preliminary | SGLang Host visible/invisible pair |
| Fig. 12 | `fig_ablation` | sglsim-preliminary | SGLSim runtime replay calibrated |
| Fig. 13 | `fig_budget_sensitivity` | sglsim-preliminary | SGLSim replay regime budget sweep |
| Fig. 14 | `fig_dp_scale` | sglsim-preliminary | SGLSim direct DP sweep |
| Fig. 14b | `fig_peer_rank_selection` | sglsim-preliminary | SGLSim peer-route replay |
| Fig. 15 | `fig_simulator_calibration` | mixed-preliminary | paired SGLang/SGLSim runs |
| Fig. 15b | `fig_physical_recovery_copy_calibration` | preliminary-measured | torch/CUDA physical recovery-copy microbench |
| Fig. 15c | `fig_recovery_model_calibration_error` | model-based-preliminary | SGLSim recovery_model + torch/CUDA copy anchors |
| Fig. 15d | `fig_sglang_sglsim_accuracy` | mixed-preliminary | paired SGLang/SGLSim synthesis run |
| Fig. 15e | `fig_p0_physical_bridge_coverage` | sglsim-preliminary | P0 physical bridge coverage audit |
| Fig. 15f | `fig_sglang_host_restore_materialization` | preliminary-measured | real SGLang Host restore materialization parser |

## 目录

- `src/`: 生成式 system overview 的 prompt 记录。
- `png/`: Markdown 审阅使用的渲染图。
- `data/template/`: 结果图的 deterministic template CSV。
- `data/measured/`: 可替换 template 的 measured 或 preliminary CSV。
- `tables/`: 第 08 节中 Tables 1-4 的 Markdown 版本。
- `gallery.md`: 所有图的单页预览。
- `manifest.json`: 图表、数据来源和输出文件的机器可读索引。
