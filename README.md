# SGLSim

This repository is being rebuilt as `sglsim`, a CPU-only simulator fork derived from SGLang.

The checkpointed SGLang-based project has been moved to `Archived/` and should be treated as migration reference material. New plans and new implementation files live outside `Archived/`.

## Supported

- Offline and blocking simulator modes
- AIConfigurator latency prediction
- CPU-only request scheduling and batching simulation
- Metadata-only KV/page/cache simulation
- Simulator metrics output
- Benchmark metrics loading
- Fault-tolerance strategy simulation for baseline, remote backup, and host backup

## Not Supported

- Real GPU inference
- CUDA graphs and CUDA kernels
- Distributed GPU execution
- Real model weight loading
- Real KV tensor transfer
- LoRA or checkpoint weight updates

## Fault-Tolerance Simulation

`sglsim` now has two fault-tolerance paths:

- Native runtime path: starts `sglsim.launch_server`, accepts real
  `benchmark/online` fixed-client HTTP requests, runs scheduler/cache/failover
  logic in process, and replaces only GPU forward/communication with CPU
  simulation.
- Offline estimator: `sglsim-fault-sim` replays benchmark-shaped events without
  starting a server. Use it for fast what-if estimates, not as the authoritative
  fault-tolerance experiment path.

Start a native simulator server:

```bash
uv run python -m sglsim.launch_server \
  --model-path simulator-model \
  --host 127.0.0.1 \
  --port 28000 \
  --dp-size 2 \
  --kv-backup remote_backup \
  --enable-cache-report \
  --enable-hicache \
  --hicache-storage-backend remote_backup \
  --hicache-storage-prefetch-policy wait_complete \
  --retry-queue-same-priority-as-waiting
```

Run the migrated fixed online benchmark against the native runtime through its
local deploy wrapper:

```bash
uv run python benchmark/online/src/main.py \
  --config remote_backup-fixed-4090 \
  --output-dir /tmp/sglsim/online/remote_backup
```

Run a single strategy from a multi-agent-style config directory:

```bash
uv run sglsim-fault-sim \
  --config-dir examples/fault_tolerance/remote_backup-fixed \
  --output-dir /tmp/sglsim/fault/remote_backup
```

Run the same workload across baseline, remote backup, and host backup:

```bash
uv run sglsim-fault-sim \
  --config-dir examples/fault_tolerance/baseline-fixed \
  --output-dir /tmp/sglsim/fault/compare \
  --compare
```

The offline estimator writes `metrics.json`, `events.jsonl`, `internal_failover_metrics.json`, `job_summary.csv`, `task_summary.csv`, and `cache_hits.csv` without starting a real server or touching GPU runtime.

Fault-tolerance simulation is event driven. Requests enter per-DP CPU scheduler queues, so queue wait time and rank contention affect completion order. The online benchmark currently supports fixed-client workloads only. Failure triggers are resolved from `job_completed`, `task_completed`, or timeline events; `delay` moves the actual `fault_injected` time, and recovery is modeled through a simulated `simulate_gpu_failure` / `simulate_gpu_recovery` state machine.

Baseline, host backup, and remote backup differ in checkpoint ratio, backup overhead, and failover penalty. A task is affected only when it overlaps the failed DP rank between the actual injection and recovery times. KV movement is metadata-only but page-level: each task receives page IDs, recovered page IDs, recovered token counts, and estimated host or remote transfer time.

To use AIConfigurator latency prediction for the same event model, pass a simulator config:

```bash
uv run sglsim-fault-sim \
  --config-dir examples/fault_tolerance/remote_backup-fixed \
  --output-dir /tmp/sglsim/fault/aiconfigurator \
  --sim-config-path /path/to/simulator-config.json
```

Without `--sim-config-path`, the fault simulator uses the built-in constant latency predictor.
