# Experiment configuration rationale

## 1. Request layer

The primary serving trace is the pinned TraceLab `v0.0.2` DuckDB artifact. A direct audit of the
pinned SHA-256 finds 665,453 LLM rounds, 8,058 provider sessions and 743,819 tool calls. It is
better suited than the old single-provider Claude adapter because it preserves round-level input,
prefix, append, output, model, provider, session and timing fields for both Codex and Claude.
Sampling and confidence intervals use the provider-session, rather than individual rounds, as the
unit. NVIDIA AIPerf independently documents the same grouping and token fields.

The release artifact is not directly compatible with Qwen3: its audited median input is 132,092
tokens and 90.05% of rounds exceed 40,960 tokens. Both official Qwen3 configurations cap context
at 40,960. The paper experiment therefore uses a declared rank-preserving context projection and
reports projected results; it must not call them unmodified TraceLab replay. Prefix ratio and
newly appended tokens remain separate, so service time uses projected cache misses while active KV
footprint uses the projected full context.

For S1--S3 and S6--S8, session starts use a controlled Poisson process so load can be swept, while within-session gaps use
the observed timestamps capped at 30 seconds. This is a time-compressed, load-controlled replay,
not fixed-schedule replay. ServeGen shows why a per-client, time-varying arrival sensitivity is
ultimately preferable to a naive global arrival generator; an uncapped TraceLab fixed-schedule
external-validity replay therefore remains a paper-readiness item rather than an implicit claim.

TraceLab sessions are predominantly sequential coding-agent loops. NVIDIA AIPerf can infer some
subagent containment, but explicitly notes that TraceLab has no parent-link ground truth. Pythia
uses real production coding-assistant workflows, yet that trace is proprietary and cannot serve as
a reproducible public input. The primary workload therefore supports token/reuse analysis and
session WCT. A separate controlled fork/join workload supports causal-Slack and recovery-priority
mechanism tests.

E1 adds a public observed-DAG external-validity check from the pinned full AgentX/Weka artifact:
393 workflows, 98,827 model requests, 323 parallel stages, 64-token prefix hash blocks, and exact
recorded request timestamps. It preserves each selected workflow's complete intra-workflow
schedule and graph; workflow starts use a fixed 2-second interleave because the artifact does not
provide a production cross-workflow arrival process. Source API latency is never reused as Qwen
service time. The audited median input is 142,016 tokens, so the same declared Qwen3 projection is
required.

Complete per-chunk Aegis event retention is expensive. E1 therefore defines an eligibility pool of
the 27 observed parallel workflows with at most 128 requests, without truncating any selected DAG.
Ten seed partitions take two workflows each without replacement, yielding 20 distinct paper
workflows. This is a deliberately bounded external-validity sample and does not support a claim
about the excluded long-DAG tail.

E1's Algorithm-1 ticket is trained and scored with five workflow-disjoint folds. Features use only
the request-return prefix. Calibration takes the maximum absolute residual per calibration
workflow, rather than treating correlated request snapshots as independent. On all 111,072 causal
snapshots from 126 parallel workflows, the predeclared 95% interval attains 98.18% request-level
coverage and 92.86% simultaneous workflow coverage; Slack lower-bound coverage is 100%, but the
selected E1 tickets are all zero. This conservative null is retained rather than substituting the
oracle future-aware 24-workflow subset or a median prediction.

Configuration rationale is frozen before execution and therefore does not embed E1 outcome
numbers. Each immutable lineage reports the observed recovery actions, paired seed-level effect,
confidence interval, and null/negative interpretation in its own `analysis.md`; the audit does not
permit an outcome to rewrite the workload selection or causal-Slack contract.

Sources: `tracelab_paper`, `tracelab_repo`, `tracelab_release`,
`tracelab_artifact_audit`, `nvidia_aiperf_tracelab`, `agentx_methodology`, `aiperf_weka`,
`agentx_artifact_audit`, `weka_causal_source`, `servegen_nsdi26`, `pythia_local`,
`qwen3_8b_config`, and `qwen3_32b_config` in `source_registry.json`.

## 2. Failure layer

Field MTTF and forced fault experiments answer different questions and are never mixed:

1. **Conditional-on-failure:** inject a paired fail-stop into the active critical-tail round of a
   session selected from a policy-independent no-failure shadow schedule. This explicitly estimates
   failure-added session WCT and guarantees the complete path is exercised. Random active and
   correlated faults remain separate S2/S8 sensitivity axes, so the critical-tail estimand is not
   presented as a natural fault distribution.
2. **Hazard-weighted operational risk:** weight the paired impact by an A100 central MTTF of 9,600
   GPU-hours, with 4,800/9,600/19,200-hour sensitivity. A short simulation does not use an
   artificially high Poisson rate to create enough failures.

One failed GPU/replica is central because FailSafe evaluates the common 7-of-8-GPU case. Sensitivity
uses 1, 2 and 4 failed replicas plus a correlated whole-node outage. This follows the failure-count
and 25%-correlated-worker sensitivity style in LUMEN. The central 30-second capacity-restoration
horizon is anchored to LUMEN's 29.9-second failure-impact window; it is not labeled as a hardware
repair-time distribution. S7
uses 1/5/15/30-second sensitivity so that both the short-unavailability regime and sustained
replica loss remain identifiable inside the finite TraceLab replay. These durations are modeled
deployment inputs, not empirical universals.

S4 is a deliberately controlled ordering stress test: three active replicas fail in an
adversarial largest-Slack-first insertion order and their valid residual-DAG Slack metadata is
refreshed immediately before the fault. The refresh changes neither the request DAG nor its
causal evidence; it prevents queueing time from collapsing every remaining-Slack value to zero.
S4 therefore verifies the recovery-priority mechanism, not a natural field-fault distribution.

The 9,600-hour central MTTF is the exact conversion of Alibaba's approximately 400-day per-A100
failure interval. Meta's all-cause training-job result (1,024 GPUs at 7.9 hours, or 8,089.6
GPU-equivalent hours under independent scaling) is recorded only as an order-of-magnitude
cross-check. These different failure definitions are not pooled, and both remain training-cluster
proxies because a comparable inference-serving field MTTF trace is not public.

Sources: `meta_hpca_reliability`, `aegis_nsdi_reliability`, `failsafe_local`, `lumen_local`.

## 3. Deployment layer

The modeled cluster has four DGX A100 nodes. Each has eight A100-40GB GPUs and 1 TB standard host
memory. AegisKV receives 512 GiB host DRAM per node (2 TiB total), a controlled 50% budget that
does not assume all host memory is free. The current S1--S9 plus E1 matrix does not include a Host DRAM
capacity axis, so no Host DRAM sensitivity is claimed.

Qwen3-8B runs TP1 with 32 replicas. Qwen3-32B BF16 weights are about 65.6 GB, so TP1 cannot fit an
A100-40GB; it runs TP4 with eight replicas. Both use the official 40,960-token limit. The BF16 KV
footprints derived from the official configs are:

- Qwen3-8B: `2 * 36 * 8 * 128 * 2 = 147,456` bytes/token (144 KiB);
- Qwen3-32B: `2 * 64 * 8 * 128 * 2 = 262,144` bytes/token (256 KiB).

Peer protection reserves 6 GiB per 8B replica and 24 GiB per 32B TP4 replica (6 GiB per GPU).
The cross-node copy/load path uses 20 GB/s effective payload bandwidth, with 10/20/25-GB/s
sensitivity, bounded by the 200-Gbit/s (25-GB/s) line rate and about 22.5-GB/s pinned host-device
sample bandwidth. JSON keys explicitly distinguish `Gbit_per_s` from `GB_per_s`. The
263-275-GB/s HGX peer result is recorded for same-node diagnostics but is not incorrectly applied
to a cross-node failure-domain-separated standby.

Sources: `a100_ampere`, `a100_hgx_guide`, `dgx_a100_guide`, `qwen3_8b_config`,
`qwen3_32b_config`, `mooncake_local`.

### Current-design chunking

The authoritative [`current_design.md`](../../doc/design/current_design.md) fixes the default
chunking contract at 4,096 prefix tokens and 256 decode tokens. S1--S8 and E1 use that exact
default. S9 alone declares a 1,024-token prefix override because it is an
analytical-versus-DES transfer micro-verification; neither S9 nor its override is presented as the
production default. The validator and lineage audit fail closed on any other value, and the design
authority is copied and hashed into every new lineage.

## 4. Evidence limits

The SGLSim complete path models causal Slack, independent HBM/DRAM admission and commit,
fail-stop, route rotation/rebind, queue outage, recovery ordering, output fencing and silent replay.
It does not move physical KV tensors. Transfer and recovery bytes are derived from model configs;
service-time coefficients remain modeled inputs copied from the existing SGLSim profile until
same-topology A100 measurements exist. Consequently a run may pass semantic, provenance,
pairing and statistical audits while `physical_calibration=false` and
`paper_ready_calibrated=false`. Plotting never overrides these gates.
