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

The simulator writes `metrics.json`, `events.jsonl`, `internal_failover_metrics.json`, `job_summary.csv`, `task_summary.csv`, and `cache_hits.csv` without starting a real server or touching GPU runtime.
