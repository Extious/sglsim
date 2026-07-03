# 100K Context Staged 1-3-1-3-1 Workload Plan

This artifact records the current AegisKV 100K-context workload profile. It is
materialization evidence for the experiment configuration, not an online
measurement result.

## Scenario Family

- `staged_13131_100k_context_no_backup`
- `staged_13131_100k_context_host_only`
- `staged_13131_100k_context_gpu_replica`
- `staged_13131_100k_context_aegiskv`

All four scenarios share the same workload shape, seed, failure timing, and
retry policy. Only the AegisKV recovery policy changes.

## Workload Shape

- workflow shape: `1-3-1-3-1`
- workflows per run: `1`
- agents per workflow: `9`
- turns per agent: `3`
- DP size: `4`
- max concurrency: `3`
- failure injection: `soft`
- retry route: `peer`
- failure after output tokens: `128`

Critical-agent rule:

- every singleton stage is Critical;
- in each width-3 fan-out stage, `stage_local_agent_index=0` is the long-decode
  Critical Agent;
- the other width-3 fan-out agents are Non-Critical.

## Context Scale

For the AegisKV policy point:

- `gsp_system_prompt_len=98304`
- `gsp_question_len=1024`
- `gsp_output_len=128`
- `gsp_num_turns=3`
- estimated first-turn prompt tokens: `99328`
- estimated final-turn context tokens: `101632`
- estimated long-decode Critical output tokens: `1024`

Dry-run manifest:

- `results/agentic_workflows/sglsim-staged-13131-100k-context-aegiskv-plan-20260703/agentic_workflow_manifest.json`
- `results/agentic_workflows/sglsim-staged-13131-100k-context-aegiskv-plan-20260703/configs/staged_13131_100k_context_aegiskv/trial_000/client.json`

## Evidence Boundary

This plan proves that the benchmark layer can materialize the requested
100K-context, 1-3-1-3-1, Critical/Non-Critical workload. It does not yet prove
online WCT, recovery latency, or strict-final physical tensor-copy behavior.
