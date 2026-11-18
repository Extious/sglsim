# Complete real-Weka baseline study

> **INVALIDATED FOR PERFORMANCE CLAIMS (2026-08-23).** This archived result
> inherits the invalid RTX4090 TCP-based Peer-HBM coefficient from the full
> design replay. Preserve it only for diagnosis; do not cite its comparisons
> until the suite is regenerated with a valid physical profile. See
> `../transport-invalidation-rtx4090-20260823.json`.

## Result status

The complete replay is **PASS** for the declared metadata-simulation gates and
**not paper-ready physical evidence**. It uses the full Weka trace scale, a
shared failure set, byte-capacity-matched protection pools, and paired
workflow-level bootstrap intervals. It supports a calibrated advantage over
FCFS recomputation and several protection adaptations, but not a general win
over every load-aware failure-time scheduler.

Evidence class:
`full_real_trace_capacity_matched_metadata_baseline_simulation`.
`physical_end_to_end_verified=false`; `paper_ready=false`.

## Scale and fairness envelope

| Quantity | Audited value |
| --- | ---: |
| Workflows | 393 |
| Requests | 98,827 |
| Input tokens | 21,635,381,376 |
| Output tokens | 106,474,498 |
| Sealed chunks replayed | 5,697,618 |
| Paired failure requests | 955 |
| Faulted workflows used by bootstrap | 241 |
| Bootstrap resamples | 4,000 |

All policies receive the same requests, deterministic arrival projection,
source/standby mapping, failure snapshots, Qwen3-8B KV-byte mapping, DP=8/PP=2
topology, HBM/DRAM byte limits, and calibrated transfer/recompute coefficients.
The original Weka request scale and timing are real; the global cross-workflow
arrival clock is modeled because the dataset does not provide one.

## End-to-end comparison

The metric is total workflow join-barrier harm over the 955 paired failure
requests.  Negative change versus Recompute-FCFS is better.  In the final
column, positive values mean AegisKV is better than that baseline.

| Policy | Fidelity | Total harm (h) | Change vs. FCFS | AegisKV reduction vs. policy, paired 95% CI |
| --- | --- | ---: | ---: | ---: |
| Recompute-load-aware | isolation control | 5.4515 | -9.33% | -3.38% [-7.12%, -0.07%] |
| LUMEN-load-aware | capacity-matched adaptation | 5.5106 | -8.35% | -2.27% [-6.01%, +0.96%] |
| DejaVu-neighbor | capacity-matched adaptation | 5.6274 | -6.41% | -0.15% [-0.51%, +0.21%] |
| AegisKV-latest | native latest SGLSim design | 5.6360 | -6.26% | -- |
| KevlarFlow-ring | capacity-matched adaptation | 5.6918 | -5.33% | +0.98% [+0.51%, +1.49%] |
| Recompute-FCFS | native no-protection baseline | 6.0125 | 0.00% | +6.26% [+5.73%, +6.89%] |
| RaidServe-host | capacity-matched adaptation | 6.0125 | 0.00% | +6.26% [+5.74%, +6.88%] |
| GhostServe-parity-TP8 | dedicated topology control | 6.0125 | 0.00% | +6.26% [+5.73%, +6.87%] |

The important boundary is that load-aware recomputation still beats AegisKV by
3.38% with a CI narrowly below zero. LUMEN is 2.27% better at the point estimate
but its CI crosses zero, and DejaVu is statistically indistinguishable. AegisKV
does significantly beat FCFS recomputation, RaidServe-host, the GhostServe TP=8
control, and KevlarFlow-ring in this calibrated replay. Failure-time dispatch
remains a first-order contributor that must be combined with placement before a
general end-to-end claim is justified.

## What AegisKV does well, and why that is not yet a large WCT gain

- AegisKV commits 771,886 of 771,891 scheduled copies (99.999%), compared with
  96.40% for DejaVu, 87.21% for KevlarFlow, 0.104% for LUMEN, 0.0018% for
  GhostServe, and 0.0007% for RaidServe. This is a clear selectivity and copy
  completion result.
- It uses committed Peer-HBM state for 519 of 955 failed requests; 436 fall
  back to recomputation. Request hit rate is therefore 54.35%.
- Those hits restore 11.246% of failed produced token-work. The remaining
  88.754% is unprotected, so request hit rate materially overstates recovery
  coverage.
- At the corrected calibrated point, HBM restore cost is 42.94% of recompute
  cost per token. Combining 11.246% coverage with this cost ratio gives an
  analytic recovery-work reduction ceiling of approximately 6.42%, close to
  the measured 6.26% join-harm reduction versus FCFS recomputation.

The correct conclusion is therefore: the latest controller executes its
selected copies reliably and the corrected physical HBM recovery path creates
a meaningful opportunity, but coverage, predictor quality, and failure-time
dispatch still prevent a universal end-to-end advantage.

## Dedicated recovery-executor controls

CacheFlow and Bounded-State Restoration are evaluated only after a prefix has
already been externalized; neither chooses normal-path protection placement.

| Executor control | Mean completion | P95 completion | Max local staging |
| --- | ---: | ---: | ---: |
| Restore/recompute split | 22.007 s | 67.054 s | 0 GiB |
| BSR bounded window | 22.022 s | 67.070 s | 0 GiB |
| CacheFlow 3D executor | 2.751 s | 8.382 s | 0 GiB in this metadata model |

These numbers compare executor mechanics on the same 955 external-state
opportunities.  They are not end-to-end CacheFlow or BSR system results.

## Calibration and prediction boundary

- Physical coefficients come from one hash-linked RTX4090 Qwen3-8B PP=2 bundle:
  Peer-HBM protection 2,945,001.99 bytes/ms, direct DRAM protection 35,839.69
  bytes/ms, HBM recovery 3,196,430.17 bytes/ms, DRAM recovery 121,992.42
  bytes/ms, and recomputation 0.10744 ms/token plus 15.310 ms fixed cost.
- They are projected to DP=8.  96.48% of requests exceed the calibrated
  4,096-token context range.
- Algorithm 1 workflow-disjoint remaining-time median absolute error is
  138.43 s, P90 is 1,194.65 s, and MAE is 567.47 s.  Its nominal 90% interval
  covers 79.26% of observations.
- SGLSim moves metadata and charges calibrated time.  It does not execute the
  physical Mooncake D2D/D2H/load/replay path in this experiment.

## Figure index

| Artifact | Purpose |
| --- | --- |
| [`figures/fig1_aegiskv_latest_design.pdf`](figures/fig1_aegiskv_latest_design.pdf) | Latest independent-tier architecture and fail-stop recovery path |
| [`figures_four_path_v2/fig2_end_to_end_harm.pdf`](figures_four_path_v2/fig2_end_to_end_harm.pdf) | End-to-end harm and paired confidence intervals |
| [`figures_four_path_v2/fig3_cost_pareto.pdf`](figures_four_path_v2/fig3_cost_pareto.pdf) | Copy cost versus harm and copy-completion efficiency |
| [`figures_four_path_v2/fig4_protection_coverage.pdf`](figures_four_path_v2/fig4_protection_coverage.pdf) | Request hit rate versus protected token-work |
| [`figures_four_path_v2/fig5_restore_executors.pdf`](figures_four_path_v2/fig5_restore_executors.pdf) | CacheFlow/BSR/split executor controls |
| [`figures_four_path_v2/fig6_scale_and_predictor.pdf`](figures_four_path_v2/fig6_scale_and_predictor.pdf) | Dataset scale and Algorithm 1 accuracy |
| [`figures_four_path_v2/fig7_advantage_ceiling.pdf`](figures_four_path_v2/fig7_advantage_ceiling.pdf) | Analytic advantage ceiling and evidence gaps |
| [`figures/fig8_baseline_fidelity.pdf`](figures/fig8_baseline_fidelity.pdf) | Corpus coverage, baseline fidelity, and fairness contract |
| [`figures_four_path_v2/academic_figure_suite.pdf`](figures_four_path_v2/academic_figure_suite.pdf) | Six-page four-path quantitative figure bundle |
| [`figures/aegiskv-complete-academic-overview.drawio`](figures/aegiskv-complete-academic-overview.drawio) | Editable two-page system and evaluation overview |

Each quantitative figure is exported as PDF, PNG, and SVG.  The draw.io source
has embedded-XML PNG/SVG/PDF exports.  The complete machine-readable artifact
index is [`figures/manifest.json`](figures/manifest.json).

## Reproduction and audit

```bash
uv run python AegisKV/experiments/real_weka_baselines/run.py
uv run python AegisKV/experiments/real_weka_baselines/make_figures.py
uv run --extra test pytest test/unit/srt/test_aegis_*.py -q
```

The result gate is
[`results/full_393_workflows_four_path_v2/audit.json`](results/full_393_workflows_four_path_v2/audit.json).
It verifies the exact full dataset, all 5,697,618 sealed chunks, paired failure
request/stage keys, complete policy presence, and full runtime reclamation.  The
baseline fidelity contract and complete 69-PDF disposition are frozen in
[`../../related_works/BASELINE_SELECTION_2026-08-21.md`](../../related_works/BASELINE_SELECTION_2026-08-21.md).

The focused current-design simulation regression reports 53 passed tests,
including causal Slack, independent tiers, full design, baselines, and the new
four-path artifact contract. Broader legacy-suite status is tracked separately
and is not used to upgrade this metadata result to paper-ready evidence.
