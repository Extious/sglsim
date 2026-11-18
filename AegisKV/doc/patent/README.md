# AegisKV 专利交底材料

## 当前交付物

- `AegisKV专利交底书-独立分层最终版.docx`：当前推荐交付稿。三项保护主线为工作流感知租值型 KV 优先级、动态租值驱动的独立分层 KV 管理和 AIMD 资源平衡，并仅保留四幅附图。
- `AegisKV专利交底书-严格模板版.docx` 及其 PDF：历史审阅基线，供版式对照。
- `template/AegisKV交底书模板.docx`：原始模板。
- `figures/aegiskv-patent-figures-independent-final.drawio`：附图 1--4 的多页、可编辑、纯黑白 draw.io 源文件。
- `figures/independent-final/figure-1.drawio.png` 至 `figure-4.drawio.png`：由上述 draw.io 原生导出的高分辨率附图。
- `scripts/build_patent_independent_final.py`：复制 latest 原稿、重写正文、原生导出四图并执行最终审计。
- `scripts/build_patent_disclosure.py`、`scripts/audit_patent_disclosure.py`：历史严格模板版的生成和审计脚本。

`AegisKV专利交底书.docx` 和 `AegisKV专利交底书-最新设计.docx` 均为历史版本，不再作为当前交付稿。

## 模板结构约束

当前文档仅使用模板已经存在的结构：

1. 缩略语和关键术语，包括缩略语表和关键术语解释；
2. 相关技术背景以及最接近的现有技术，包括背景、技术方案和缺点；
3. 本发明技术方案的详细阐述，包括技术问题、主要要点、系统架构、产品及取证、三个实施例和创新点表；
4. 发散思维；
5. 在先公开的现有技术文献信息、模板原有的 `专利交底书Checklist` 文本及文献表。

文档没有增加封面、文档结构页、第四实施例、附录、提交前待办或新的子章节。模板的三个表格位置和样式分别保持为缩略语表、创新点表及在先文献表。

三个实施例对应：

- 实施例一（4 步）：将请求 Slack 对应的故障恢复收益与命中率、复用收益、LRU 时序和字节数融合为租值型 KV 优先级；
- 实施例二（6 步）：通过每层租金、每请求连续尾部、本池最小堆和 Lazy Refresh 完成独立准入、源侧传输及分层提交；
- 实施例三（4 步）：AIMD 只更新显存保护额度，本地执行器与存储控制模块分工完成安全收缩。

正文先用“已计算中间记忆”解释 KVCache。3.2 将代表性现有技术分为 GPU 内存与 KV 管理、程序级调度、分离式服务与多层存储、迁移与容错四类，并用三条缺点分别引出三项保护主题。保护点一不限定 Slack 的预测或图分析算法，而是保护 Slack 如何进入容错恢复收益，并与 Hit Rate、复用收益和 LRU 信息共同形成单位字节租值键。保护点二采用当前独立分层语义：Peer-HBM 与 DRAM 分别维护租金、连续尾部和本池最小堆；信号更新只标记索引键陈旧，仅在检查堆顶或暴露新尾部时 Lazy Refresh。两层独立准入并由源 Worker 分别执行 D2D/P2P 和 D2H，最终组合为四种驻留结果。

AIMD 只负责更新 Peer-HBM 额度。额度降低后的新接纳阻止由本地执行器完成，具体尾部分块回收由存储控制模块按优先级完成；HBM_ONLY 与 HBM_AND_DRAM 均可直接进入安全回收候选，不要求先建立 DRAM 副本，但释放前必须原子撤销其恢复资格。若仅有 HBM 且源 Worker 仍存活，后续 DRAM 获准时仍由源 Worker 独立执行 D2H，不存在目标 HBM 到 DRAM 的中转。专利正文不使用“写回”称呼，源码中的兼容命名不受影响。

## 生成与检查

```bash
uv run --isolated --with python-docx==1.2.0 -- \
  python AegisKV/patent/scripts/build_patent_independent_final.py
```

最终版生成脚本同时检查：三项创新、三段中文伪代码、Slack/Hit Rate/LRU 复合租值、Lazy Refresh、独立分层关键语义、旧写回/DRAM 前置条件禁用语、无修订痕迹禁用语、三个模板表格、四幅嵌入图、图像分辨率、OOXML 压缩包完整性及字段刷新标志。正文公式中的 Unicode 兼容上下标会统一转换为普通字符配合 Word 原生上下标格式，避免因字体缺字产生方格。四幅附图由 draw.io 原生导出并单独目检。正式递交前仍建议在目标 Microsoft Word 版本中刷新字段，并重点复核跨页表格和分页。
