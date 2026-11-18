# Agentic Workflow Benchmarks

AegisKV-facing SGLSim **online** benchmarks for multi-agent workflows. This
package wraps `benchmark/synthesis_requests`: every non-dry run starts the
SGLSim HTTP server, drives the real async client path, and records
`request_details.csv` with the Aegis metadata columns.

## Layout

```
benchmark/agentic_workflows/
├── config.py              # scenario registry + per-trial client.json/server.json materialization
├── workload.py            # workflow-shape and main-agent/subagent summaries
├── failure.py             # explicit failure-injection plan validation
├── online.py              # online experiment plan materialization
├── run.py                 # single-scenario entrypoint (execute_scenario)
├── suite.py               # multi-scenario suite entrypoint (execute_suite)
├── rebuild.py             # AegisKV beamer rebuild entrypoint for 2a-2d
├── summary.py             # summarize request_details.csv -> WorkflowRunSummary
├── aggregate.py           # aggregate paper-candidate rows by scenario (+CI95)
├── combine.py             # combine per-scenario summary CSVs
├── paper.py               # project combined summary into paper-facing candidate rows
├── config/                # shared static configs (e.g. hpca_smoke_suite.json)
└── results/               # legacy empty local scratch root; not the default
```

Default online artifacts now live under:

```text
benchmark/AegisKV/draft/experiments/sglsim_large/results/agentic_workflows/
```

Each run writes:

- `configs/<scenario>/trial_*/client.json` and `server.json`
- `plans/trial_*/online_experiment_plan.json`
- `runs/trial_*/<timestamp>/request_details.csv`
- `agentic_workflow_manifest.json`
- `agentic_workflow_summary.csv` after non-dry runs

## Named Generated-Shared-Prefix Scenarios

A registry of named GSP scenarios driven through `synthesis_requests`. Selected
scenarios (full list via `--list-scenarios` or `config.py`):

- `smoke`: cheap two-turn sanity run.
- `motivation_main_subagent_dp8_failure`: DP8 100K-context motivation run that
  injects one serving-rank failure and compares main-agent singleton stages with
  subagent fan-out stages before introducing the Critical Agent terminology.
- `alpha_tradeoff_dp8_{baseload,highload}_{fail1,fail3}_a{005..060}`:
  DP8 100K-context alpha sweep points. They fix GPUReplica backup fraction
  through `gsp_aegis_gpu_budget_fraction`, vary load through 2/4 workflows and
  6/12 concurrency, and vary failure pressure through explicit injection count.
- `staged_13131_100k_context_{no_backup,host_only,gpu_replica,aegiskv}`:
  DP8 100K-context policy comparison points. GPUReplica-only fixes
  `gsp_aegis_gpu_budget_fraction=0.3`; all points use explicit soft failure
  replay and the same staged workflow shape.
- `claude-code-trace-jsonl`: real Claude Code trace workload converted from
  public trace rows into SGLSim multi-turn request payloads.
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

# Materialize a structured online plan without launching SGLSim
python -m benchmark.agentic_workflows.online \
  --scenario motivation_main_subagent_dp8_failure \
  --output-dir benchmark/AegisKV/draft/experiments/sglsim_large/results/agentic_workflows/rebuild_plan

# Rebuild the current AegisKV beamer evidence set
python -m benchmark.agentic_workflows.rebuild --clean --compile-beamer

# Rebuild only one stage
python -m benchmark.agentic_workflows.rebuild --stage motivation --clean
python -m benchmark.agentic_workflows.rebuild --stage policy_100k --clean
python -m benchmark.agentic_workflows.rebuild --stage alpha_dp8 --clean

# Run one scenario manually
python -m benchmark.agentic_workflows.run \
  --scenario motivation_main_subagent_dp8_failure \
  --run-id motivation-main-subagent-dp8-main-r1 \
  --runs 1

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

## Current Rebuild Protocol

The AegisKV beamer rebuild intentionally clears old figures and measured CSVs
before rerunning experiments. The `policy_100k` stage runs the DP8 no-backup /
HostBackup / GPUReplica / AegisKV comparison used by the beamer policy CDF.
The `alpha_dp8` stage runs the DP8 base/high-load x fail1/fail3 x 12-point
alpha sweep used by the beamer tradeoff figure. New Motivation, policy, and
alpha-tradeoff figures should only use data from fresh run
directories under the AegisKV experiment root above. Do not read stale files
from `benchmark/agentic_workflows/results/`.
