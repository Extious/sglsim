# AegisKV Beamer

本目录保存 AegisKV 研究汇报 Beamer 源文件，内容覆盖：

- AegisKV 问题建模
- Backup / Inference HBM 划分
- Critical / Non-Critical Agent 规则
- SGLSim 实验设计
- failure replay 与已有实验结果
- 当前证据边界和下一步

模板参考 `/home/comp/zhanzhao/beamer_example`，并复用其 `beamerthemeSimple.sty`。

## 文件结构

```text
benchmark/AegisKV/beamer/
  presentation.tex
  beamerthemeSimple.sty
  Figures/
  Makefile
  README.md
```

`Figures/` 中的图片来自 `benchmark/AegisKV/draft/figures/png/`，用于让演示稿目录可以独立编译。当前 deck 只插入生成式 System overview、motivation 图和实验结果图，不插入详细机制流程图。

## 编译

演示稿使用 `ctex` 支持中文，默认用 LuaLaTeX 编译：

```bash
cd benchmark/AegisKV/beamer
conda activate latex
make
```

等价手动命令：

```bash
lualatex -interaction=nonstopmode presentation.tex
lualatex -interaction=nonstopmode presentation.tex
```

清理中间文件：

```bash
make clean
```

## 当前环境说明

当前机器已配置 conda 环境 `latex`，TeX Live / TinyTeX 安装在：

```text
/home/comp/zhanzhao/miniforge3/envs/latex/tinytex
```

首次 shell 使用时如果 `conda` 尚未初始化，可先执行：

```bash
source /home/comp/zhanzhao/miniforge3/etc/profile.d/conda.sh
conda activate latex
```

激活后 `lualatex`、`xelatex`、`pdflatex`、`latexmk` 和 `pdftoppm` 都会从该环境中解析。
