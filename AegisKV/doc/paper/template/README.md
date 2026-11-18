# EuroSys 2027 / ACM LaTeX 模板来源

下载日期：2026-08-06

EuroSys 2027 CFP 要求采用 SIGPLAN 样式的双栏投稿格式。本目录保存 ACM `acmart` 生产版模板：

- 上游包：`acmart` 2.19（2026-06-27）
- 下载来源：CTAN 官方镜像 `https://mirrors.ctan.org/macros/latex/contrib/acmart.zip`
- 原始 ZIP：`acmart.zip`
- SHA-256：`5011d4a698b993ffd4c9a6a24cf4cde7dc9deb4a7ec2dc357dbbe4ec15085618`
- 解压目录：`acmart/acmart/`
- `acmart.cls` 由该目录中的官方 `acmart.ins` 和 `acmart.dtx` 本地生成。

ACM 官网的两个模板直链在本机下载时均返回 HTTP 403，因此改用 README 明确称为生产版来源的 CTAN release，而不是 GitHub development snapshot。

当前匿名审稿配置：

```tex
\documentclass[sigplan,review,anonymous]{acmart}
\acmSubmissionID{<PAPER_ID>}
\renewcommand\footnotetextcopyrightpermission[1]{}
\settopmatter{printfolios=true,printacmref=false,printccs=false}
\renewcommand{\bibliofont}{\normalsize}
```

`sigplan` 已隐含双栏；不再额外传入会被 `acmart` 报为未使用的
`twocolumn` 选项。

`acmart` 默认从 `amsart` 继承 8 pt 参考文献字号，因此当前稿件显式将
参考文献恢复为 10 pt；否则不符合 EuroSys 对正文、参考文献、图和图注均
至少 10 pt、行距至少 12 pt 的要求。提交前必须再次核对 EuroSys 2027
CFP/HotCRP 是否发布了专用 header，并遵守双盲、12 页 technical content、
178 x 229 mm 版心、至少 8 mm 栏间距、灰度可读和 AI 工具使用披露要求。
