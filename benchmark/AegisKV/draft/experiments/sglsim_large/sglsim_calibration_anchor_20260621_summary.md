# SGLang/SGLSim Calibration Anchor Summary

Run id: `codex-fig15-calibration-from-normal-overhead-r3-20260628`

This artifact is the active preliminary input for Fig. 15. It replaces the
older synthetic SGLang-vs-SGLSim ranking scatter with the calibration evidence
that is currently defensible: normal-path p99 overhead anchors from real
SGLang paired runs and the online SGLSim no-failure overhead sweep.

## Inputs

- SGLang source: seven matched DP=2 paired Slurm runs on `hkbugpumanode`
  (`32861`, `32863`, `32869`, `32870`, `32871`, `32874`, `32899`)
- SGLSim source: current `fig_normal_path_overhead.csv`, whose baseline and
  AegisKV rows come from `codex-normal-overhead-online-r3-20260628`
- Output CSV: `figures/data/measured/fig_simulator_calibration.csv`
- Metric: normal-path p99 overhead percentage

## Results

- Baseline: SGLang `0.00%`, SGLSim `0.00%`
- Protected path anchor: SGLang HiCache write-through `0.31%`, online SGLSim
  AegisKV no-failure sweep `-0.49%` with `0.93%` 95% CI

## Interpretation

This figure is now an honest calibration anchor rather than a claimed final
ranking-preservation result. It shows that normal-path p99 overhead is within
noise on both the small real SGLang HiCache write-through anchor and the online
SGLSim AegisKV no-failure sweep. Final Fig. 15 still requires paired
SGLang/SGLSim policy runs on matched workloads and should claim ranking
preservation only after those paired measurements exist.
