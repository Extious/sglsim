# Agentic Workflow Benchmarks

AegisKV-facing SGLSim **online** benchmarks for multi-agent workflows. This
package wraps `benchmark/synthesis_requests`: every non-dry run starts the
SGLSim HTTP server, drives the real async client path, and records
`request_details.csv` with the Aegis metadata columns.

## Layout

```
benchmark/agentic_workflows/
├── config.py              # scenario registry + per-trial client.json/server.json materialization
├── run.py                 # single-scenario entrypoint (execute_scenario)
├── suite.py               # multi-scenario suite entrypoint (execute_suite)
├── summary.py             # summarize request_details.csv -> WorkflowRunSummary
├── aggregate.py           # aggregate paper-candidate rows by scenario (+CI95)
├── combine.py             # combine per-scenario summary CSVs
├── paper.py               # project combined summary into paper-facing candidate rows
├── config/                # shared static configs (e.g. hpca_smoke_suite.json)
└── results/               # default output root (gitignored)
```

## Named Generated-Shared-Prefix Scenarios

A registry of named GSP scenarios driven through `synthesis_requests`. Selected
scenarios (full list via `--list-scenarios` or `config.py`):

- `smoke`: cheap two-turn sanity run.
- `shared_prefix_dp2`: normal-path shared-prefix workload for cache/frontier
  measurements.
- `shared_prefix_sweep_{low,mid,high}`: Fig. 10b overlap sweep points.
- `hard_failover_host`: DP2 hard-restart replay with HostBackup lookup.
- `hard_failover_peer` / `hard_failover_nonpeer`: DP4 hard-restart replay with
  adjacent-peer / non-peer retry.
- `policy_*`, `peer_route_*`, `ablation_*`, `budget_*`: policy / routing /
  ablation / GPUReplica-budget sweeps.

### Usage

```bash
# List scenarios
python -m benchmark.agentic_workflows.run --list-scenarios

# Generate configs without launching SGLSim
python -m benchmark.agentic_workflows.run \
  --scenario hard_failover_host --dry-run --run-id dry-hard-host

# Run one scenario
python -m benchmark.agentic_workflows.run \
  --scenario hard_failover_host --run-id hpca-host-r3 --runs 3

# Run the default suite
python -m benchmark.agentic_workflows.suite \
  --suite-id hpca-agentic-r3 --runs 3

# Combine summaries -> paper candidates -> per-scenario aggregate
python -m benchmark.agentic_workflows.combine
python -m benchmark.agentic_workflows.paper
python -m benchmark.agentic_workflows.aggregate
```

Exported rows use `evidence_tier=sglsim-agentic-online-preliminary`. They must
not be copied into final measured figure CSVs until the remaining final-evidence
requirements in `benchmark/AegisKV/draft/experiments/sglsim_large/README.md` are satisfied.
