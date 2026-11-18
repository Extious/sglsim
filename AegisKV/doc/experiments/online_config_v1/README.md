# AegisKV online three-layer experiments

This entrypoint keeps every experiment readable as three independent source
files:

```text
configs/<experiment>/
├── request.json       # workload, arrivals, token shape, client concurrency
├── failure.json       # trigger, target, injection mode, retry/replay
└── deployment.json    # model, DP/TP/PP, cache, predictor, AegisKV policy
```

The source files are authoritative.  The runner validates layer ownership and
then generates the legacy `client.json` and `server.json` expected by
`benchmark.synthesis_requests`.  Failure and AegisKV policy fields are placed
in the generated client file only as a compatibility detail; experiment
authors must keep them in `failure.json` and `deployment.json` respectively.

## Validate and inspect

```bash
uv run python AegisKV/experiments/online_config_v1/run.py --dry-run
```

The command creates a campaign directory containing:

```text
source_config/       # immutable snapshot of the three authoring files
effective_config/    # generated client.json and server.json
experiment_manifest.json
```

The manifest records hashes for both source and effective configurations, so a
result can be tied back to its exact request, failure, and deployment inputs.

## Run online

```bash
uv run python AegisKV/experiments/online_config_v1/run.py \
  --config-dir AegisKV/experiments/online_config_v1/configs/online-aegiskv-smoke
```

This uses the existing online SGLSim path: real wall-clock time, HTTP server,
real asynchronous client, and the Engine/DataParallelController/Scheduler
process topology.  It does not use the offline DES runner.

## Evidence boundary

The engine remains metadata-only and replaces GPU forward with predicted
latency.  The manifest therefore always records `physical_result=false`.
Configuration completeness and a successful run do not constitute physical or
paper-ready performance evidence.
