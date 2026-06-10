# Fault Tolerance HPCA Experiment Notes

This document records the current online SGLSim fault-tolerance experiment path.
Runtime semantics should remain aligned with pinned SGLang; these scripts only drive
the online HTTP benchmark and post-process its outputs.

## Scope

- Online logical GPU failure with `POST /simulate_gpu_failure` and recovery with
  `POST /simulate_gpu_recovery`.
- No-HiCache `gpu_backup` metadata strategy.
- Multi-agent generated-shared-prefix workload with critical-agent KV backup.
- Paper-style plots from repeated online runs.

Deferred from this experiment path: OS process kill, `host_backup`,
`remote_backup`, and HiCache-backed FT.

## Smoke

Use the smoke matrix to validate the benchmark and plotting pipeline quickly:

```bash
uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --smoke \
  --repeats 2 \
  --results-root /tmp/sglsim_ft_sweep_smoke \
  --log-level info

uv run --extra test python benchmark/synthesis_requests/diagnostics/plot_fault_tolerance_sweeps.py \
  /tmp/sglsim_ft_sweep_smoke/fault_tolerance_sweep_latest.json \
  --output-dir /tmp/sglsim_ft_sweep_figures

uv run --extra test python benchmark/synthesis_requests/diagnostics/fault_tolerance_paper_tables.py \
  /tmp/sglsim_ft_sweep_figures/fault_tolerance_sweep_summary.json \
  --output-dir /tmp/sglsim_ft_sweep_figures

uv run --extra test python benchmark/synthesis_requests/diagnostics/build_fault_tolerance_artifacts.py \
  /tmp/sglsim_ft_sweep_smoke/fault_tolerance_sweep_latest.json \
  --output-dir /tmp/sglsim_ft_sweep_figures

uv run --extra test python benchmark/synthesis_requests/diagnostics/verify_fault_tolerance_artifacts.py \
  /tmp/sglsim_ft_sweep_figures/fault_tolerance_artifacts.json \
  --profile smoke
```

Expected artifacts:

- `fault_tolerance_sweep_latest.json`: one entry per case and repeat.
- `fault_tolerance_sweep_raw_summary.json`: per-run metrics after CSV parsing.
- `fault_tolerance_sweep_summary.json`: grouped mean and standard error.
- `fault_tolerance_sweep_hpca.png` and `.pdf`: four-panel summary figure.
- `fault_tolerance_sweep_system.png` and `.pdf`: throughput, latency, and
  retried-request impact figure.
- `fault_tolerance_paper_tables.md`, `.tex`, and `.json`: paper-ready derived
  metrics from the aggregate summary.
- `fault_tolerance_artifacts.json`: bundle manifest with run counts, sweep-axis
  counts, strategy summary, and validated artifact paths.
- `fault_tolerance_verification.json`: smoke or paper gate report.

## Full Matrix

The full matrix uses the multi-agent FT configs under
`benchmark/synthesis_requests/config/`:

```bash
uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --dry-run \
  --case-names baseline,gpu_critical \
  --repeats 1 \
  --seed-base 100 \
  --port-base 31000 \
  --results-root benchmark/synthesis_requests/results/ft_hpca_sanity

uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --case-names baseline,gpu_critical \
  --repeats 1 \
  --seed-base 100 \
  --port-base 31000 \
  --results-root benchmark/synthesis_requests/results/ft_hpca_sanity \
  --run-timeout-s 900 \
  --log-level info

uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --dry-run \
  --repeats 5 \
  --seed-base 100 \
  --port-base 31000 \
  --results-root benchmark/synthesis_requests/results/ft_hpca_sweep

uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --repeats 5 \
  --seed-base 100 \
  --port-base 31000 \
  --results-root benchmark/synthesis_requests/results/ft_hpca_sweep \
  --run-timeout-s 900 \
  --max-runs 10 \
  --log-level info

uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --resume \
  --repeats 5 \
  --seed-base 100 \
  --port-base 31000 \
  --results-root benchmark/synthesis_requests/results/ft_hpca_sweep \
  --run-timeout-s 900 \
  --max-runs 10 \
  --log-level info

uv run --extra test python benchmark/synthesis_requests/diagnostics/run_fault_tolerance_sweeps.py \
  --status \
  --repeats 5 \
  --seed-base 100 \
  --port-base 31000 \
  --results-root benchmark/synthesis_requests/results/ft_hpca_sweep

# Repeat the --resume --max-runs 10 command until the manifest has 85 completed runs.

uv run --extra test python benchmark/synthesis_requests/diagnostics/plot_fault_tolerance_sweeps.py \
  benchmark/synthesis_requests/results/ft_hpca_sweep/fault_tolerance_sweep_latest.json \
  --output-dir benchmark/synthesis_requests/results/ft_hpca_figures

uv run --extra test python benchmark/synthesis_requests/diagnostics/fault_tolerance_paper_tables.py \
  benchmark/synthesis_requests/results/ft_hpca_figures/fault_tolerance_sweep_summary.json \
  --output-dir benchmark/synthesis_requests/results/ft_hpca_figures

uv run --extra test python benchmark/synthesis_requests/diagnostics/build_fault_tolerance_artifacts.py \
  benchmark/synthesis_requests/results/ft_hpca_sweep/fault_tolerance_sweep_latest.json \
  --output-dir benchmark/synthesis_requests/results/ft_hpca_figures

uv run --extra test python benchmark/synthesis_requests/diagnostics/verify_fault_tolerance_artifacts.py \
  benchmark/synthesis_requests/results/ft_hpca_figures/fault_tolerance_artifacts.json \
  --profile paper
```

The plot contains:

- Strategy comparison: baseline recompute vs critical-agent GPU backup reuse.
- Backup lag sensitivity: KV reuse as `peer_backup_lag_tokens` changes.
- Critical-agent coverage: critical reused tokens and non-critical recompute tokens.
- Failure timing: failover overhead across injection times. The current full
  matrix uses `30,36,40,44s`, chosen around the long-prompt multi-agent
  prefill/decode window observed in sanity runs.
- System impact: output throughput, end-to-end latency, agent-class latency, and
  retried-request count.

## Interpretation

For HPCA-style evidence, report both the aggregate JSON and the plotted figure.
The raw summary should be kept with the run directory so each plotted mean can be
traced back to concrete request-level CSV rows.

The `paper` verification profile is intentionally stricter than `smoke`: it
requires at least three repeats per sweep point and at least four points each for
backup lag, critical-agent ratio, and failure timing.
