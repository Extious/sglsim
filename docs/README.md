# SGLSim documentation

| Document | Purpose |
|----------|---------|
| [sglsim-srt-onboarding.md](sglsim-srt-onboarding.md) | CLI entrypoints, SGLang path mapping, simulation modes |
| [sglsim-srt-contract.md](sglsim-srt-contract.md) | Included / excluded scope |

**Branches:** `main` is the online parity simulator (tag `v0.1.0+`). Fault-tolerance work is on branch `fault-tolerance`.

### Roadmap (`docs/todo/`)

| Plan | Purpose |
|------|---------|
| [trace-calibration-design.md](todo/trace-calibration-design.md) | SGLang vs SGLSim trace compare (TTFT/TPOT/queue) |
| [offline-simulate-design.md](todo/offline-simulate-design.md) | Future logical-clock / fast offline simulation (not on main) |
| [cache-aware-scheduling-design.md](todo/cache-aware-scheduling-design.md) | LPM, DFS-weight, DP load balance |
| [communication-time-prediction-design.md](todo/communication-time-prediction-design.md) | Layered comm predictor, avoid NCCL double-count |

Recommended implementation order: communication P0, cache-aware P1, then trace calibration P0–P3 as regression gate.

Reference notes: `reference/`. Benchmark: `benchmark/synthesis_requests/`.
