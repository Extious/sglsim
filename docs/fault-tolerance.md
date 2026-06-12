# Fault tolerance branch

The `fault-tolerance` branch extends `main` (`v0.1.0+`) with GPU failure injection, offline
strategy simulation, and multi-agent benchmarks.

## Layout

| Path | Purpose |
|------|---------|
| `python/sglsim/fault_tolerance/` | Offline FT strategy simulator + CLI (`sglsim-fault-sim`) |
| `benchmark/fault_tolerance/` | Offline compare runs + multi-agent AegisKV benchmark |
| `benchmark/online/` | Online multi-agent worker helpers |
| `examples/fault_tolerance/` | Fixed-strategy scenario configs |
| SRT hooks | `/simulate_gpu_failure`, failover batch routing, peer KV backup |

Synthesis-request FT configs and diagnostics live under
`benchmark/synthesis_requests/config/{ft_*,ma_ft_*}` and
`benchmark/synthesis_requests/diagnostics/run_fault_tolerance_*.py`.

## Quick start

```bash
uv sync --extra test --extra server
uv run sglsim-fault-sim --help
uv run python benchmark/fault_tolerance/run_offline.py --help
```

Online failure injection uses the HTTP control endpoints restored on this branch
(`POST /simulate_gpu_failure`, `POST /simulate_gpu_recovery`).
