# AegisKV 技术文档

本目录用于编写 AegisKV 的连续技术文档，不使用 Beamer。

- LaTeX 入口：`aegiskv_technical_document.tex`
- 文档类型：`ctexart`，A4 技术报告
- 原 Beamer 演示稿：`../presentation/`

编译：

```bash
mkdir -p output/pdf
latexmk -xelatex \
  -interaction=nonstopmode \
  -halt-on-error \
  -outdir=output/pdf \
  aegiskv_technical_document.tex
```

生成的 PDF 位于：

```text
output/pdf/aegiskv_technical_document.pdf
```

清理编译中间文件：

```bash
latexmk -C -outdir=output/pdf aegiskv_technical_document.tex
```
