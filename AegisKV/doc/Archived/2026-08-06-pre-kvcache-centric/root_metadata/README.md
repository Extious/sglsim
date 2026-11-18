# AegisKV

本目录保存 AegisKV 论文草稿、图表资产、实验证据和辅助脚本。原根目录 `AegisKV/` 已迁入这里，避免仓库根目录保留一套只服务论文的内容。

- `draft/`: 中文论文草稿、图表、证据 CSV、实验记录和校验脚本。
- `draft/paper/`: Markdown 论文正文和审阅稿。
- `draft/figures/`: PNG 图表预览、draw.io 机制图源文件、数据 CSV 和图表生成脚本。
- `draft/experiments/`: SGLang 小规模实验、SGLSim 大规模实验和 final evidence matrix。
- `draft/scripts/`: full draft 构建、完整性校验和 final gate 报告脚本。
- `beamer/`: AegisKV 问题建模、实验设计和已有结果的 Beamer 演示稿源码。

当前草稿聚焦两级 AegisKV 设计：

- `GPUReplica`: 将选中的 KV segment 备份到另一个 DP rank 的 peer GPU。
- `Host`: 将选中的 KV segment 备份到同一 DP rank / worker 侧的本地 host memory。

`RemoteStore` 暂不纳入当前论文范围。

常用校验命令：

```bash
python benchmark/AegisKV/draft/scripts/build_full_draft.py
python benchmark/AegisKV/draft/scripts/validate_draft_integrity.py
python benchmark/AegisKV/draft/figures/scripts/validate_figure_evidence.py --write-hpca-hybrid
```
