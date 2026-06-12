# Multi-Agent AegisKV Fault Tolerance Benchmark

This benchmark implements the AegisKV system-design workload described in
`05_system_design.md`: a two-round fan-out/fan-in coding workflow where workflow
semantics decide which KV segments should be protected before failure.

The primary path is now **online**: `run.py --mode online` starts the SGLSim
server, the online fault injector, and a multi-agent HTTP client that issues real
`/v1/completions` requests for every workflow segment. The fast policy model is
still available as `--mode model` for debugging and artifact generation.

The AegisKV comparison is metadata-level and uses `GPUReplica` as the first
backup target. It does not implement Host sidecars, HiCache L2/L3 integration, or
real KV block copy. Online artifacts record the actual server/client/fault
injection logs; AegisKV segment-selection metrics are derived from those
workflow traces.

## Workflow

The synthetic workflow is:

```text
O0 -> E1/E2/E3 -> O0-J1 -> F1/F2/F3 -> O0-J2
```

Important AegisKV segments:

- `S1`: common system/tool prefix
- `S6`: diagnosis summary after round-1 fan-in
- `S15`: E2 tool-blocked test reproducer prefix
- `S21`: F2 tool-blocked regression verifier prefix
- `S8`: final synthesis prefix near the workflow sink

## Experiments

`run.py` supports five experiment groups:

- `strategy`: baseline vs whole-agent GPU backup vs AegisKV segment backup
- `failure_site`: sensitivity to failure location in the workflow
- `budget`: backup budget ratio sweep
- `lag`: peer backup lag sweep
- `scale`: number of concurrent workflows

## Commands

Online smoke run:

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/run.py \
  --config-dir benchmark/fault_tolerance/multi_agent/config/aegiskv_coding \
  --output-dir /tmp/sglsim_aegiskv_ma_smoke \
  --mode online \
  --smoke
```

For staged validation without launching the whole matrix:

```bash
SGLSIM_VENV=/path/to/native-env \
python benchmark/fault_tolerance/multi_agent/run.py \
  --config-dir benchmark/fault_tolerance/multi_agent/config/aegiskv_coding \
  --output-dir /tmp/sglsim_aegiskv_ma_online_smoke \
  --mode online \
  --smoke \
  --experiments scale \
  --max-runs 1
```

`SGLSIM_VENV` must point to an environment whose `bin/python` can import the
native SGLSim runtime dependencies, including `torch` and `safetensors`. If this
is not set, the benchmark uses the repository `.venv`.

Online dry run, useful before launching the full matrix:

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/run.py \
  --config-dir benchmark/fault_tolerance/multi_agent/config/aegiskv_coding \
  --output-dir /tmp/sglsim_aegiskv_ma_online_plan \
  --mode online \
  --smoke \
  --dry-run
```

Full online paper run:

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/run.py \
  --config-dir benchmark/fault_tolerance/multi_agent/config/aegiskv_coding \
  --output-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_full \
  --mode online \
  --repeats 5 \
  --experiments strategy,failure_site,budget,lag,scale
```

Fast model-only run:

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/run.py \
  --config-dir benchmark/fault_tolerance/multi_agent/config/aegiskv_coding \
  --output-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_model \
  --mode model \
  --repeats 5 \
  --experiments strategy,failure_site,budget,lag,scale
```

Plots:

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/plot.py \
  --results-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_full
```

Paper tables:

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/paper_tables.py \
  --results-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_full
```

Verification:

```bash
uv run --extra test python benchmark/fault_tolerance/multi_agent/verify_artifacts.py \
  --results-dir benchmark/fault_tolerance/multi_agent/results/aegiskv_full \
  --profile paper
```

## Outputs

Each run directory contains:

- `manifest.json`
- `summary.json`
- `runs.csv`
- `segments.csv`
- `recovery_paths.csv`
- `online_configs/` and `online_runs/` when `--mode online` is used
- `paper_tables.{json,md,tex}`
- `artifact_verification.json`
- `fig_strategy_recovery.{png,pdf}`
- `fig_wct_failure_site.{png,pdf}`
- `fig_budget_efficiency.{png,pdf}`
- `fig_backup_lag.{png,pdf}`
- `fig_recovery_path_distribution.{png,pdf}`
- `fig_scale.{png,pdf}`

## Paper Interpretation

Use `fig_strategy_recovery` as the primary result: AegisKV should reduce
recompute by protecting only segments that matter for failure-time workflow
recovery. Use `fig_budget_efficiency` to show that AegisKV spends the same
GPUReplica budget more effectively than whole-agent backup. Use
`fig_recovery_path_distribution` to explain where the benefit comes from:
full segment restore or prefix restore plus suffix replay, rather than opaque
retry behavior.
