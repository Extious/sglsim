# AegisKV Beamer

本目录保存 AegisKV 系统论文汇报 Beamer 源文件。`presentation.pdf` 应是完整论文式版本，而不是实验流水账或中间版本记录。内容覆盖：

- 论文核心论点与 motivation
- SGLang / SGLSim 背景和相关工作定位
- AegisKV 问题建模、MDP 控制器和 Backup / Inference HBM 划分
- HostBackup directory、GPUReplica、shared safe frontier 和 failure-time retry path
- SGLSim online serving 与 SGLang calibration 方法
- Motivation、policy matrix、alpha tradeoff、online alpha、real trace、ablation、SGLang/SGLSim accuracy、real-system calibration 等实验结果
- Scope / limitations 和结论

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

`Figures/` 中的图片来自 `benchmark/AegisKV/draft/figures/png/`，用于让演示稿目录可以独立编译。SGLang/SGLSim accuracy 图以 `benchmark/AegisKV/draft/figures/data/measured/fig_sglang_sglsim_accuracy.csv` 为来源生成。

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

## 本机环境说明

本机已配置 conda 环境 `latex`，TeX Live / TinyTeX 安装在：

```text
/home/comp/zhanzhao/miniforge3/envs/latex/tinytex
```

首次 shell 使用时如果 `conda` 尚未初始化，可先执行：

```bash
source /home/comp/zhanzhao/miniforge3/etc/profile.d/conda.sh
conda activate latex
```

激活后 `lualatex`、`xelatex`、`pdflatex`、`latexmk` 和 `pdftoppm` 都会从该环境中解析。
