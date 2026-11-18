# Full real-Weka result

> **INVALIDATED FOR PERFORMANCE CLAIMS (2026-08-23).** The Peer-HBM
> coefficient was derived from transfers whose Mooncake runtime explicitly
> selected `tcp`. These numbers are retained only as metadata/diagnostic
> output and are excluded from the current paper. They must be regenerated
> after the V100S physical DRAM, Peer-HBM, recompute, and independent-tier
> mechanism gates pass. Separately, the unscaled mapping is not executable by
> Qwen3-8B: 85,129 of 98,827 requests (86.14%) exceed its 40,960-token context
> limit. See `../transport-invalidation-rtx4090-20260823.json` and
> `../real_weka_context_scale/context-audit-20260823.json`.

The four-path deterministic replay completed in 435.7 seconds and passed every
experiment audit gate. `PASS` here means complete metadata-mechanism and
artifact validation; it does not mean paper-ready physical recovery evidence.
The superseded v1 replay remains diagnostic-only because it reused protection
bandwidths as recovery bandwidths.

## Scale

| Item | Final value |
|---|---:|
| Weka workflows | 393 |
| LLM requests | 98,827 |
| Input tokens | 21,635,381,376 |
| Output tokens | 106,474,498 |
| Subagent requests | 42,029 |
| Maximum active fan-out | 10 |
| Progressively sealed chunks | 5,697,618 |
| Paired fail-now request snapshots | 955 |
| Affected join stages | 898 |
| Faulted workflows used by bootstrap | 241 |

The materialized trace is 1,847,151,435 bytes and has SHA-256
`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`.

## Optimization result

The primary metric is workflow-aggregated join-barrier harm, paired by faulted
workflow and bootstrapped over 4,000 workflow resamples.

| Comparison to `aegiskv_latest` | Harm reduction | 95% paired bootstrap CI |
|---|---:|---:|
| recompute + FCFS | 6.2626% | [5.7244%, 6.8581%] |
| recompute + Slack due order | 6.2163% | [5.7284%, 6.7830%] |
| Aegis placement + FCFS | 0.0700% | [-0.0859%, 0.2641%] |
| zero-Slack placement + due order | -0.0658% | [-0.3150%, 0.1591%] |

The latest policy selected Peer HBM for 519 of 955 failed requests and full
recompute for 436. Its mean join-barrier harm was 22,594.1 ms. The observed
benefit over FCFS recompute is positive under this projection. The incremental
benefit of learned Slack over the zero-Slack capacity-matched placement remains
statistically indistinguishable from zero.

AegisKV scheduled 771,891 HBM chunks, committed 771,886, and copied 455.49 TB,
versus 853,701 scheduled chunks, 853,695 commits, and 500.44 TB for the
zero-Slack placement. Thus learned Slack reduced modeled normal-path HBM copy
bytes by 8.98% while the paired harm difference versus zero-Slack due ordering
remained statistically indistinguishable.

## Algorithm 1 accuracy boundary

The workflow-disjoint five-fold replay produced 111,072 causal active-agent
snapshots and 32,238 predicted request tickets; 66,589 unmodeled requests failed
closed to zero Slack. Only 17,593 of all 98,827 primary q50 tickets were nonzero.

Remaining-time median absolute error was 138.4 seconds and p90 error was
1,194.6 seconds. Slack median absolute error was 83.9 seconds and p90 error was
1,040.4 seconds. The nominal 90% remaining-time interval covered only 79.26%,
although the conservative Slack lower bound covered 99.36%. These errors rule
out a strong learned-Slack efficacy claim from this result.

## DRAM interpretation

With the checked four-path bundle, DRAM load projects to 1.2087 ms/token, while
recompute projects to 0.1074 ms/token. SlackLandlord therefore correctly gives
the first DRAM candidate nonpositive recovery gain and admits no DRAM jobs in
the optimization run. The DRAM state machine is still tested separately and is
excluded from the optimization summary: HBM commits while DRAM remains pending,
both later form `HBM_AND_DRAM`, and an injected HBM-copy failure leaves DRAM
committed and selected for recovery without any HBM-to-DRAM transfer.

## Verification

- Experiment strict audit: PASS, no failed gates.
- Four-path artifact-contract tests: 4 passed.
- Focused current-design simulation regression: 53 passed, including causal
  Slack, independent tiers, full design, baselines, and four-path provenance.

## Evidence boundary

The Weka request, token, topology, and within-workflow timing scale is real.
Cross-workflow arrivals are a deterministic Poisson projection because the
corpus has no global arrival clock. Mixed-model requests are mapped to the
Qwen3-8B BF16 PP2 KV footprint. Four distinct source-protection/recovery paths
from one hash-linked RTX4090 PP2 DP2 bundle are extrapolated to DP8, and 95,348
requests (96.48%) exceed the calibrated 4,096-token context. This result is
therefore a large calibrated SGLSim metadata study, not an end-to-end Mooncake
recovery measurement or paper-ready systems result.
