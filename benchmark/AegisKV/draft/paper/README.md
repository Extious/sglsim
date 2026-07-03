# AegisKV：面向多智能体 LLM Serving 的 Workflow-Aware KV 备份

这是新 AegisKV 论文的中文 Markdown 总入口。完整内容按章节拆分在 `sections/` 下，本文档用于固定论文主线、章节顺序和每章应承担的证明责任。

## 论文主张

Multi-agent LLM serving 中，GPU / DP-rank failure 后最值得保护的不是所有 KV，也不是最新 KV，而是那些一旦丢失会显著增加 workflow completion time 的 KV segment。本文固定采用 SGLang `write_through` Host Backup：radix tree 插入后立即 D2H，GPU failure 后 source rank 的 Host KV 仍然存在。baseline 也做 D2H，但 retry rank 感知不到 failed rank Host KV；AegisKV 通过 HostBackup directory 让 retry rank 能使用这部分 KV，并额外为高价值 segment 选择是否放置 GPUReplica。

## 章节顺序

1. [00. 论文大纲](sections/00_paper_outline.md)
2. [01. 摘要](sections/01_abstract.md)
3. [02. 引言与动机](sections/02_introduction_motivation.md)
4. [03. 背景与相关工作](sections/03_background_related_work.md)
5. [04. 系统模型与设计](sections/04_system_design.md)
6. [05. 实现计划](sections/05_implementation_plan.md)
7. [06. 实验计划](sections/06_evaluation_plan.md)
8. [07. 初步结果与图注](sections/07_preliminary_results.md)
9. [08. 图表与预期结果](sections/08_figures_and_expected_results.md)
10. [09. 结论与下一步](sections/09_conclusion_next_steps.md)

## 当前图像与完成度

- 当前 20 张图的预览：[`../figures/gallery.md`](../figures/gallery.md)
- 图表 manifest：[`../figures/manifest.json`](../figures/manifest.json)
- 图表证据状态：[`../figures/evidence_status.json`](../figures/evidence_status.json)
- 实验证据目录：[`../experiments/`](../experiments/)
- HPCA 2027 完成度审计：[`readiness_audit.md`](readiness_audit.md)

当前机制图已经可用于设计审阅；14 张结果图已经从 deterministic
SGLSim counterfactual CSV 重绘，并在图中标注为 `SGLSim preliminary`。
这些图可以支持内部 paper review 和叙事打磨，但不能作为最终定量
claim。下一步必须用 `hkbugpumanode` 上的 SGLang 小规模实验和 online
AegisKV SGLSim 大规模实验替换为 `evidence_tier=final` 数据。

当前 `hkbugpumanode` 已经验证两条小规模真实 SGLang 路径：job `32855`
完成 DP=2 HiCache write-through 端到端 smoke；job `32861` 完成 matched
baseline vs HiCache write-through no-failure overhead paired smoke。随后 job
`32863` 将 paired overhead smoke 扩大到 32 请求、128 max tokens。当前
matched aggregate 显示 mean overhead 为 1.18%-3.05%，p99 overhead 在
small-sample noise 内。这些结果支持实验管线可行性和 Fig. 11 方向，但还
不是最终 HPCA 定量结果。

投稿前硬门槛：

```bash
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-final
```

## 必须证明的结论

1. 同样大小 KV loss 在不同 agent/lifecycle/segment 上的 workflow penalty 明显不同。
2. GPUReplica、Host、recompute 各自有适用边界，不能固定使用某一种。
3. 固定 backup ratio 在不同 workflow / failure regime 下不是全局最优。
4. AegisKV 的 segment-level policy 优于 request-level、agent-level、uniform Host、uniform GPUReplica 和 static critical-path backup。
5. AegisKV 的 no-failure normal-path overhead 可控。
6. 在相同 fixed `write_through` D2H 成本下，AegisKV 的 HostBackup visibility 优于 Host-invisible baseline。
7. GPUReplica peer 选择必须同时考虑 load 与 KV locality，shared-prefix marginal accounting 能减少重复备份。
8. 补齐 AegisKV hook 后，SGLSim 能保持 SGLang 上观察到的 policy ranking，用于 scale-up 研究。

## 当前论文范围

```text
备份层级：
  GPUReplica：将选中的 KV segment 复制到另一个 DP rank 的 peer GPU。
  Host：将选中的 KV segment 复制到同一 DP rank / worker 的本地 host memory。
  None：没有 failure-protected copy，failure 后依赖 recompute。

本文不覆盖：
  RemoteStore
  完整 host failure recovery
  通用 KV offload 贡献
  强制 DRL controller
```

## 推荐写作顺序

先完成 `02_introduction_motivation.md`、`04_system_design.md` 和 `06_evaluation_plan.md`，因为它们决定论文是否成立。然后根据 SGLang / SGLSim 的真实实现状态更新 `05_implementation_plan.md`。最后用真实实验数据替换 `08_figures_and_expected_results.md` 中的预期趋势。
