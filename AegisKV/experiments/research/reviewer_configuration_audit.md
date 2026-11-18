# Reviewer-facing experiment configuration audit

Audit date: 2026-08-29. This document distinguishes source-backed facts, derived values,
controlled experiment choices, and uncalibrated model inputs. The machine-readable counterpart is
[`parameter_provenance.json`](parameter_provenance.json).

## Verdict

| Layer | Current choice | Verdict | Remaining external-validity requirement |
| --- | --- | --- | --- |
| Request | Pinned TraceLab v0.0.2 sessions, controlled fork/join mechanism workload, and pinned AgentX/Weka observed-DAG external validity | Public token/reuse and observed-DAG inputs are reproducible; E1 is deliberately bounded to shorter DAGs and yields conservative zero Slack LCBs | Add uncapped fixed-schedule TraceLab and a long-DAG AgentX sensitivity |
| Failure | Paired conditional fail-stop, 1/2/4-worker and whole-node sensitivity, separately reported field-hazard weighting | Defensible when conditional effects and field incidence are not mixed | Obtain inference-serving field failure/repair data; current MTTF is a training-cluster proxy |
| Deployment | 4x DGX A100-40GB nodes, Qwen3-8B TP1 and Qwen3-32B TP4, 1 TiB host/node with 512 GiB assigned to AegisKV | Vendor geometry and memory arithmetic are reproducible | Same-topology service-time and transfer calibration followed by matched physical replay |

The current plots may be cited as metadata-DES mechanism evidence only. They are not promoted to
same-topology physical results by this audit.

## Request layer

### Why TraceLab remains the primary public trace

The pinned DuckDB SHA-256 contains 665,453 rounds, 8,058 provider sessions, and 743,819 tool calls.
It records per-round total input, cached prefix, newly appended input, output tokens, provider,
session, model, and timestamps. This directly exercises the two AegisKV properties that generic
chat traces do not: long-lived session state and imperfect prefix reuse. The
[TraceLab paper](https://arxiv.org/abs/2606.30560) characterizes those workload properties, and
[NVIDIA AIPerf's TraceLab replay specification](https://docs.nvidia.com/aiperf/dev/tutorials/datasets-inputs/replay-trace-lab-agentic-coding-traces)
documents exactly which fields can be replayed.

TraceLab also has two material limitations:

1. Qwen3-8B/32B default to a 40,960-token context, while the pinned trace has a 132,092-token median
   input and 90.05% of rounds exceed 40,960. The current workload is therefore a declared
   rank-preserving projection, not an unmodified replay.
2. TraceLab has no parent/child ground truth. NVIDIA's loader reconstructs some subagent joins by
   timing containment and explicitly calls the result a containment rate rather than accuracy.

For this reason, S1-S3 and S6-S8 use TraceLab for session WCT, request size, and reuse behavior,
while S4/S5/S9 use a controlled causal fork/join workload only for mechanism isolation.

### Alternatives considered

- **Pythia production coding-assistant trace.** The paper reports stable role/workflow structure,
  role-specific output statistics, and workflow-completion metrics. It is excellent evidence that a
  causal workflow metric matters, but the production trace is proprietary and cannot be the public
  experiment input. See local PDF pages 4, 5, 11, and 12.
- **Weka/AgentX trace.** E1 now reads the full pinned 1.847-GB artifact (393 workflows and 98,827
  model requests) and preserves recorded intra-workflow timing, prefix hashes, and observed
  subagent structure. The old 24-workflow oracle subset explicitly declares
  `uses_future_information=true` and remains excluded. E1 derives tickets only from
  request-return-prefix features with workflow-disjoint train/calibration/evaluation folds. Its
  event-complete paper sample is 20 distinct, untruncated workflows drawn without replacement from
  the 27 parallel workflows with at most 128 requests; this bounds resource use but excludes the
  long-DAG tail.

This configuration audit is frozen before execution and does not contain E1 outcome numbers.
Each immutable lineage's `analysis.md` reports the observed recovery actions, paired seed-level
effect, confidence interval, and null/negative interpretation. All safe AgentX Slack LCBs are zero,
so any observed E1 protection benefit must be attributed to Algorithm 2's committed extents rather
than Algorithm 1's recovery ordering.
- **ServeGen.** [ServeGen (NSDI 2026)](https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen)
  establishes that per-client, time-varying generation is more faithful than naive global traffic.
  It is a strong arrival-model sensitivity source, but it is generic production LLM traffic rather
  than a public agent DAG.
- **ShareGPT/Splitwise.** LUMEN uses these for conversational and high-concurrency regimes. They are
  useful generic-serving baselines but omit coding-agent tool gaps and the observed prefix split.

### Arrival interpretation

The primary run uses controlled Poisson session starts, then preserves observed within-session gaps
up to 30 seconds. This supports reproducible load sweeps but compresses longer human/tool pauses.
The paper must call this a time-compressed, load-controlled replay. An uncapped fixed-schedule replay
is an external-validity experiment, not something the current result silently claims.

E1 is different: it preserves every recorded AgentX request timestamp relative to its workflow and
uses a fixed two-second cross-workflow interleave. It is fixed schedule within each observed DAG,
but not a claim about production cross-workflow arrival intensity. Qwen service time is recomputed
from projected tokens; the source provider's API durations are used only to reconstruct causal
workflow observations and predictor labels.

## Failure layer

### Failure incidence

[Alibaba's NSDI 2025 Aegis paper](https://www.usenix.org/system/files/nsdi25-dong.pdf) reports that an
A100 fails after approximately 400 days on average. The central hard-failure MTTF is therefore
`400 x 24 = 9,600 GPU-hours`, with 0.5x/1x/2x sensitivity. [Meta's HPCA 2025 study](https://ai.meta.com/research/publications/revisiting-reliability-in-large-scale-machine-learning-research-clusters/)
reports 7.9 hours for a 1,024-GPU job, equivalent to 8,089.6 GPU-hours only under an independent
inverse-scaling interpretation. The Meta number includes more than GPU hard faults, so it is kept as
an order-of-magnitude cross-check and is not averaged with the Alibaba statistic.

Both studies describe training clusters. The configuration labels the resulting hazard as a proxy,
not an inference-serving field measurement. Short experiments do not inflate the Poisson rate to
force events; instead they report conditional-on-failure effects and hazard-weighted operational
risk separately.

### Failure severity and time

- The central single-worker fail-stop matches FailSafe's common 7-of-8-GPU scenario.
- S8 covers 1, 2, and 4 failed workers plus a whole-node outage. LUMEN evaluates one-to-five failed
  workers and a fixed 25% proportional-failure sweep, so this axis spans comparable severities.
- The central 30-second `repair_time_ms` is a capacity-restoration horizon anchored to LUMEN's
  29.9-second recovery-impact window. S7 covers 1/5/15/30 seconds. It is not described as a measured
  hardware replacement distribution.
- FailSafe reports 22 s recomputation, 530 ms host restore, 120 ms full recovery, and a 15 ms oracle
  on an 8xH100 TP deployment. The current 100 ms replacement and 50 ms routing components are
  controlled model inputs near that mechanism scale, not transferred H100 measurements.
- GhostServe's 5%-15% random-step failure-rate sweep is a stress frequency, not a field MTTF. The
  current suite does not use 15% as an operational probability.

Fail-slow, network partition, and silent-data-corruption experiments are intentionally excluded:
the current AegisKV contract protects KV state after a fail-stop and does not detect corrupted KV.

## Deployment layer

### GPU and host memory

[NVIDIA's A100 specification](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
gives 40 GB HBM2 and 1,555 GB/s HBM bandwidth. The
[DGX A100 guide](https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html) gives
eight A100 GPUs, 1 TiB system memory for the 320 GB system, and 200-Gbit/s cluster networking.
Assigning 512 GiB of the 1 TiB host to AegisKV is a controlled 50% budget. The current suite does
not claim a Host DRAM sensitivity experiment.

### Model arithmetic

The official [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) and
[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) configurations determine
the KV footprint:

- Qwen3-8B: `2 x 36 layers x 8 KV heads x 128 dimensions x 2 BF16 bytes = 147,456 bytes/token`.
- Qwen3-32B: `2 x 64 x 8 x 128 x 2 = 262,144 bytes/token`.

Qwen3-8B runs TP1/DP32. Qwen3-32B BF16 weights are approximately 65.6 GB, so TP1 cannot fit on an
A100-40GB; TP4/DP8 is used. The 6-GiB-per-GPU protection budget is an explicit experimental budget,
not a vendor default.

### Chunking contract

The authoritative [`current_design.md`](../../doc/design/current_design.md) specifies 4,096-token
prefix chunks and 256-token decode chunks. S1--S8 and E1 use those values exactly. S9's explicit
1,024-token prefix setting is limited to its analytical-versus-DES transfer micro-verification and
is not a deployment recommendation. Both the configuration validator and immutable-lineage audit
enforce this distinction.

### Bandwidth units and bounds

The JSON now distinguishes 200 `Gbit_per_s` line rate from 10/20/25 `GB_per_s` effective payload
rates. Twenty GB/s is below the line-rate conversion of 25 GB/s and near the 22.5-22.9 GB/s pinned
host-device example in NVIDIA's HGX software guide. The 263-275 GB/s same-node peer measurement is
diagnostic only and is not applied to cross-node, failure-domain-separated backup.

## Claims that remain blocked

1. Qwen3 service coefficients are modeled, not measured on the 4-node x 8-A100-40GB topology.
2. The 100 ms failover and 50 ms routing components are not calibrated on that topology.
3. TraceLab context is projected and session starts are load-controlled rather than fixed-schedule.
4. E1 provides public observed DAGs, but its context is projected, its cross-workflow interleave is
   controlled, its paper pool excludes workflows above 128 requests, and conservative Slack LCBs
   are zero; S4 remains the separate mechanism-only priority stress case.
5. The currently available 1-node x 2-A100-80GB host cannot clear a 32-GPU A100-40GB calibration
   gate.

These limitations are reviewer-visible by construction; no plot or successful semantic audit may
override them.
