# AegisKV 草稿目录

本目录是新的 AegisKV 中文论文草稿。它不是旧稿的简单复制，而是基于以下三类参考重新组织的问题建模、系统设计和实验计划：

- `One Pool - Two Caches`: 将 AegisKV 建模为动态资源划分问题，而不是静态备份规则。
- `DRLPart`: 借鉴 state/action/reward、预测模型和 bad-decision correction 的资源划分框架，但第一版仍采用可解释在线策略。
- `LLM KV-cache - To Restore or To Recompute`: 将 failure 后的恢复决策写成 restore-vs-recompute-aware policy，而不是默认 restore all。
- SGLang / SGLSim 源码检查：把 AegisKV 约束在现有 HiCache D2H/H2D、radix lock/eviction、ACK 和 scheduler pipeline 上建模。

## 文件结构

- `paper/README.md`: 中文论文总入口和章节索引。
- `paper/full_draft.md`: 由各章节拼接出的单文件审阅稿。
- `paper/current_results_overview.md`: 当前已有论文内容、20 张图、4 张表和 preliminary 实验证据总览。
- `paper/readiness_audit.md`: 面向 HPCA 2027 的当前完成度与实验证据缺口。
- `paper/sections/`: 论文正文分节。
- `figures/README.md`: 图表命名、绘制目标和数据来源规划。
- `figures/gallery.md`: 当前 20 张图的 PNG 预览。
- `figures/manifest.json`: 每张图的输出文件、数据来源和证据状态。
- `figures/evidence_status.json`: 图像和实测数据 readiness 校验结果。
- `figures/png/`: Markdown 审阅使用的渲染图。
- `figures/src/`: 机制图的可编辑 draw.io 源文件。
- `figures/data/`: template、measured 和 preliminary CSV。
- `figures/tables/`: 论文表格的 Markdown 版本。
- `experiments/`: SGLang 小规模实验、SGLSim 大规模实验和 measured CSV schema。
- `scripts/validate_draft_integrity.py`: 检查章节链接、figure 输出、measured/preliminary CSV 和 evidence gate。
- `scripts/build_full_draft.py`: 生成单文件 full draft。
- `related_works/README.md`: Related Work PDF 索引和每篇论文对应的使用理由。
- `related_works/pdfs/`: 当前 Related Work 引用到的 PDF 副本。

## 当前约束

本文只考虑两种 failure-protected backup placement：

```text
B = {GPUReplica, HostBackup}
```

其中 `GPUReplica` 是另一个 DP rank 的 peer GPU 上的 KV backup；`HostBackup` 是 source DP rank / worker 侧的本地 host memory backup。本文只考虑 GPU failure，因此 failed rank 的 Host KV 仍然存在；`None` 不是 backup tier，而是 failure 后没有可用 frontier 时的 recompute 结果。若将 `Host` 理解为同一 GPU 上的副本，则单 GPU failure 时副本会与 active KV 一起丢失，恢复语义不成立，因此本文不采用该解释。

## 当前源码约束

- Host Backup 固定采用 SGLang `write_through`：radix tree 插入后立即 D2H。baseline 与 AegisKV 都支付这部分 D2H 成本。
- baseline 的 retry rank 感知不到 failed rank Host KV；AegisKV 通过 HostBackup directory 暴露 ACKed Host frontier，使 retry rank 能使用 failed rank 上仍存活的 Host KV。
- SGLang radix/HiRadix 通过 `lock_ref`、write ACK、GPU eviction、Host eviction 管理 KV 生命周期；AegisKV 只能在 ACK 后登记 safe frontier。
- GPUReplica placement 需要考虑 peer load、HBM、P2P、retry route 和 KV locality。
- Host 与 GPUReplica 可能覆盖相同 shared prefix；收益按 marginal safe content frontier 计算，物理副本成本按 tier 分别计算。
- 当前可见 SGLSim source 提供 HiCache/radix/pipeline 基础模拟，以及 `benchmark/agentic_workflows/` 的命名场景包装器；最终 HPCA 证据仍需要补齐 physical KV movement 和 final online policy sweeps 后替换为 `evidence_tier=final`。
- `hkbugpumanode` 上已经完成两类小规模 SGLang smoke：DP=2 HiCache
  write-through 端到端请求验证，以及 matched baseline vs HiCache
  write-through no-failure overhead paired smoke。后者是 Fig. 11 的早期
  real-system provenance，但仍不是最终定量证据。

## 当前验证命令

```bash
python benchmark/AegisKV/draft/scripts/build_full_draft.py
python benchmark/AegisKV/draft/scripts/validate_draft_integrity.py
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --write-hpca-hybrid
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --strict-final
```

前三条应通过；`--strict-final` 在当前阶段应失败，因为部分结果图仍是
calibrated / preliminary evidence，不是 `final`。HPCA hybrid 报告只表示
“大规模 SGLSim + 小规模物理校准”的当前可用边界，不等价于 strict-final。
