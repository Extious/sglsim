# AegisKV baseline freeze and evidence boundary (2026-08-21)

This document freezes the comparison set for the real-Weka SGLSim study.  It
separates a paper's published system from a component that can actually be
represented by the current metadata-only dense DP/PP simulator.  A label ending
in **`-style adaptation` is not a reproduction of the complete paper system**.

## Corpus completion

The recursive audit covers every PDF below `AegisKV/related_works/`:

| Item | Audited value |
| --- | ---: |
| PDF files | 69 |
| Parseable physical pages | 1,169 / 1,169 |
| Unique SHA-256 payloads | 69 |
| Canonical works after version deduplication | 67 |
| First-page renders visually checked | 69 |

The two version pairs are GhostServe (top-level preprint and MLSys paper) and
AAFLOW (public version and local draft).  `2026/Maze___SC_2026.pdf` was present
but missing from the older `catalog.tsv`; the older README count of 38 therefore
does not describe the current recursive corpus.  Machine-readable evidence is
in `audit/corpus_file_audit.{json,csv}` and five contact sheets in `audit/`.
The 30-paper KV/storage subcorpus has the existing page-grounded review in
`kvcache_backup/per-paper-review.md`.

The audit establishes file coverage and mechanism selection.  It does not turn
paper-reported numbers into AegisKV measurements.

## Frozen fidelity vocabulary

| Level | Meaning in this study |
| --- | --- |
| `native` | The paper mechanism matches dense DP/PP, one failed source GPU, and the simulator exposes every resource needed by the mechanism. |
| `capacity-matched adaptation` | The core KV protection/recovery rule is implemented under the same HBM/DRAM byte limits and calibrated paths, but topology-specific parts are absent. |
| `component control` | One separable mechanism is simulated only in a dedicated ablation; it is not shown as the paper's end-to-end system. |
| `coverage only` | The paper is reviewed and discussed, but its required datapath is outside this simulator. |

## Primary end-to-end baselines

All primary policies use exactly the same Weka requests, deterministic
cross-workflow arrival projection, source/standby mapping, failure snapshots,
Qwen3-8B KV-byte mapping, total HBM/DRAM capacities, and physical transfer and
recompute coefficients.  A policy cannot receive unlimited memory merely
because the original paper assumed it.

| Result label | Paper mechanism retained | Fidelity and deliberate omissions |
| --- | --- | --- |
| `Recompute-FCFS` | Restart from the produced-token frontier with no protected KV. | Native lower bound; no protection traffic. |
| `DejaVu-neighbor` | Incremental chunk streaming to a fixed neighbor, ordered commit, and recovery from the last committed prefix. | Capacity-matched adaptation.  The original pipeline control/replay protocol is not claimed. |
| `RaidServe-host` | Proactive full KV checkpoint to host memory and restart from its committed prefix. | Capacity-matched adaptation of FailSafe/RaidServe's checkpoint path.  Elastic TP, hybrid attention, weight reload, and load router are omitted. |
| `KevlarFlow-ring` | Block-by-block fixed-ring GPU replication; replicas may be dropped under pressure and later fall back to recomputation. | Capacity-matched adaptation.  Independent-node PP communicator reinitialization is omitted. |
| `LUMEN-load-aware` | Checkpoint holder chosen by queue/copy load; failure-time locality-first dispatch followed by load balancing. | Capacity-matched adaptation.  Draft-assisted model-capacity restoration is omitted. |
| `AegisKV-latest` | Algorithm 1 GBDT remaining-time prediction then CPM Slack; Algorithm 2 independent source-owned HBM/DRAM SlackLandlord with PENDING/COMMITTED all-shard state; Algorithm 3 HBM-only guarded AIMD; Slack-due recovery. | Native latest SGLSim design. |

`Recompute-load-aware` is reported beside, but not counted as, a paper baseline.
It applies LUMEN's failure-time dispatch to requests with zero checkpointed KV,
thereby isolating routing/queueing benefit from checkpoint benefit.

## Dedicated topology/recovery controls

These are implemented and tested, but are not silently mixed into the primary
dense DP/PP bars.

| Control | Why it is separate |
| --- | --- |
| `GhostServe-parity-tp8` | GhostServe requires TP data shards plus host parity.  The main Weka configuration is TP=1, so its N=8, K=2 parity path is evaluated in a declared TP=8 scenario and never relabeled as a TP=1 result. |
| `CacheFlow-3D-restore` | CacheFlow is a restoration executor, not a failure-time placement policy.  It is compared on identical already-externalized prefixes with compute/I/O contention held fixed. |
| `BSR-windowed-restore` | Bounded-State Restoration constrains restore working set rather than selecting protection value.  It is evaluated as a bounded-staging executor with request-level commit/fallback. |
| `restore-or-recompute-split` | The anonymous draft supplies a closed-form split control.  It is an unpublished mechanism control, not a citable accepted system. |

## Complete top-level corpus disposition (39 files, 37 canonical works)

| Work | Relevance to AegisKV | Disposition |
| --- | --- | --- |
| DRLPart | Generic robust resource partitioning | Coverage only; different workload/resource model. |
| DejaVu | Incremental KV streaming and ACK frontier | Primary capacity-matched baseline. |
| Prompt Cache | Modular prefix reuse | Coverage only; no redundant failure copy. |
| Apt-Serve | Hybrid-cache admission/scheduling | Policy-adjacent ablation reference. |
| CAIR | Counterfactual functional agent influence | Criticality discussion only; not hardware-failure probability. |
| Continuum | Predicted return benefit versus TTL cost | Policy-adjacent reference. |
| Cortex | Workflow remaining-SLO Slack | Algorithm-1 related work; no KV recovery. |
| FailSafe / RaidServe | Host checkpoint plus elastic recovery | Primary host-checkpoint adaptation. |
| FleetIO | Multi-agent storage partitioning | Coverage only. |
| HEXGEN-TEXT2SQL | Agentic request scheduling | Workload/scheduling context only. |
| KVFlow | Agent-step-graph prefix reuse | Segment-granularity context; no failure copy. |
| KVShare | Multi-tenant reuse | Coverage only. |
| Locality-aware Fair Scheduling | Cache-aware fairness | Recovery-order sensitivity reference. |
| Online Scheduling with KV Constraints | Online admission/scheduling | Coverage only. |
| Fluid-Guided Online Scheduling | Memory-constrained scheduling | Coverage only. |
| RCR-Router | Role/context routing | Router context; no fail-stop checkpoint. |
| ReaLM | ABFT for accelerator bit errors | Coverage only; different fault model. |
| Shared Disk KV Cache | Shared persistence/reuse | Tier substrate; no direct fail-stop protocol. |
| Sherlock | Workflow semantic/execution vulnerability | Motivation only; different failure semantics. |
| AAFLOW (two versions) | Agentic workflow patterns | Workload context; version-deduplicated. |
| Agentix | Dynamic agent program critical path | Algorithm-1 related work; no KV protection. |
| AnchorTP | In-place state anchoring and elastic TP | Coverage only; requires single-node elastic TP and weight state. |
| Concordia | GPU dirty-page checkpoint log in CXL/DRAM | Component control only; lower-level persistent-kernel substrate. |
| GhostServe (two versions) | TP erasure-coded parity checkpoint | Dedicated TP=8 control; version-deduplicated. |
| Helium | Data-systems view of agentic serving | Context/taxonomy only. |
| LUMEN | Load-aware checkpoint placement and recovery dispatch | Primary capacity-matched baseline. |
| Maze | Distributed LLM-agent framework | Workflow context; no KV fail-stop mechanism. |
| One Pool, Two Caches | Adaptive HBM partitioning | Algorithm-3 adjacent reference; different recommender workload. |
| Pythia | Learned future workflow execution | Predictor/fallback discussion only. |
| ReviveMoE | Token replay, operation log, and expert recovery | Coverage only; requires MoE expert datapath. |
| SAGA | Workflow-atomic scheduling | Scheduling context; arXiv-only status retained. |
| Tarragon | Attention-worker KV checkpoint plus MoE healing | KV checkpoint component control; full MoE system is out of scope. |
| KevlarFlow | Background GPU ring replication | Primary capacity-matched baseline. |
| Ceres (confidential local draft) | Agent-difference-aware serving | Internal coverage only; do not distribute or cite as public. |
| Micro-task-level parallelism (local draft) | Agent workflow execution | Coverage only; unpublished status retained. |
| Restore or Recompute (anonymous draft) | Closed-form hybrid recovery split | Dedicated unpublished component control. |
| Coordination-free KV consistency (local position paper) | Consistency abstraction | Protocol discussion only; unpublished status retained. |
| Weaver (local draft) | Dependency-aware overlap | Coverage only; unpublished status retained. |

## Complete 30-paper KV/storage subcorpus disposition

Every item is reviewed page-by-page in `kvcache_backup/per-paper-review.md`.
The compact disposition below prevents an adjacent caching paper from being
misreported as fault-tolerance evidence.

| Papers | Disposition |
| --- | --- |
| GhostServe | Direct fault-tolerant paper; dedicated TP=8 parity control. |
| HCache, CacheBlend, DroidSpeak | Hybrid load/transfer/recompute substrate; recovery-executor controls only. |
| Mooncake, IMPRESS, Strata, ECHO, DirectKV, SYMPHONY, CacheSlide, Bidaw, SolidAttention | Tiering, movement, or restoration substrate; calibration/design references, not failure baselines. |
| KVCache in the Wild, DiffKV, Cache-Craft, Marconi, KV compression survey, Cache-Rethinking, SkipKV, Kitty, OPKV, FlexiCache | Retention, reuse, sparsity, or compression; no independent recoverable copy. |
| vAttention, POD-Attention, FlashInfer, Context Parallelism, NEO, LMetric, Prism | Allocator, kernel, execution, routing, or memory-resource adjacent; no fail-stop checkpoint contract. |

The detailed review has the exact title, venue/source, storage path, mechanism,
reported evidence, limitation, and physical PDF page pointers for all 30.

## Latest primary-source research checked after the local corpus

| Work (first public version) | Mechanism | Decision |
| --- | --- | --- |
| [CacheFlow](https://arxiv.org/abs/2604.25080) (2026-04-28) | 3D token/layer/GPU restoration and batch-aware marginal-gain scheduling | Implement as dedicated recovery-executor control. |
| [PBKV](https://arxiv.org/abs/2605.06472) (2026-05-07) | Predict future dynamic agent invocations for conservative eviction/prefetch | Adjacent normal-path cache control; not a failure baseline. |
| [TokenCake v3](https://arxiv.org/abs/2510.18586) (2026-05-20 revision) | Tool-call temporal offload plus graph/runtime-aware partitioning | Adjacent policy ablation reference; not a redundant copy protocol. |
| [Robust KV Management under Output-Length Uncertainty](https://arxiv.org/abs/2607.16892) (2026-07-18) | DRO reservation, routing, and prefix caching | Capacity-planning context; different objective/fault model. |
| [System-Aware KV Cache Optimization Survey](https://arxiv.org/abs/2607.08057) (2026-07-09) | Taxonomy over temporal execution, spatial placement/migration, and structural retention | Search cross-check and taxonomy only; not a system baseline. |
| [RestoreKV](https://arxiv.org/abs/2608.01247) (2026-08-02) | Learned compact complement for query-agnostic KV eviction | Accuracy-preserving compression context; it does not create an independent fail-stop copy. |
| [Pallas](https://arxiv.org/abs/2608.16477) (2026-08-17) | Predicted cellular handover; local prefix prefill plus streamed suffix | Planned-migration sensitivity only; not sudden GPU fail-stop. |
| [Bounded-State Restoration](https://arxiv.org/abs/2608.17826) (2026-08-18) | Complete external discovery with a bounded local installation window and request-level commit | Implement as dedicated bounded-staging recovery control. |
| [Aborted but Not Forgotten](https://arxiv.org/abs/2608.15939) (2026-08-16) | Rollback-consistency audit for stale retained KV | Correctness/transaction boundary, not performance baseline. |
| [Fail-Closed Lowering of Resident KV Claims](https://arxiv.org/abs/2606.01387) (2026-05-31) | Materialization predicates, ordered lifecycle events, and claim-scoped fail-closed outcomes | Correctness/evidence vocabulary only; not a protection or recovery-performance baseline. |

## Statistical and reporting contract

1. The default-scale headline is the complete 393-workflow Weka corpus, not a
   sampled request trace: 98,827 requests, 21,635,381,376 input tokens, and
   106,474,498 output tokens.
2. Workflow is the resampling unit for paired bootstrap confidence intervals.
3. All load projections are shown as a sweep because Weka lacks a global
   cross-workflow arrival clock; a favorable arrival rate is never selected as
   if it were measured ground truth.
4. End-to-end join-barrier harm, recovery service, copied bytes, copy backlog,
   capacity misses, and normal-path protection work are reported together.
5. Predictor error, mechanism probes, calibrated coefficients, metadata-only
   results, and physical end-to-end evidence remain separate gates.
6. Negative or statistically inconclusive regions remain in the figures.  No
   baseline parameter is tuned on its test failures without applying the same
   tuning budget to every policy.
