# AegisKV bottom-layer multi-agent inference fault-tolerance ideas

Status: research proposals, not implemented claims.  
Evidence cutoff: 2026-07-29.

## Recommended system spine

The eight ideas are deliberately not eight unrelated features. They form one
bottom-up systems stack:

1. [Unified Recovery-Aware Host Memory Pool](01_unified_recovery_aware_host_pool.md)
   is the physical resource substrate and the recommended main paper mechanism.
2. [Slack-Bounded Dirty Frontier](02_slack_bounded_dirty_frontier.md) converts
   the gap between request-valid materialized KV `K` and scope-protected KV `C`
   into an incremental protection deadline while emitted output `E` remains
   separate.
3. [Failure-Domain Adaptive Coding](03_failure_domain_adaptive_coding.md)
   decides whether process-externally owned local Host, a full remote replica,
   or a coded placement whose post-failure fragments satisfy the decode quorum
   provides the required failure coverage.
4. [Parallelism-Aware Recovery Manifest](04_parallelism_aware_recovery_manifest.md)
   makes logical-shard state coherent under TP/PP/DP/EP and makes the
   manifest/directory itself scope-durable or rebuildable instead of treating
   one KV buffer as a complete request.
5. [PP Boundary Anchors](05_pp_boundary_anchors.md) reduces cross-stage replay
   after a pipeline-stage failure.
6. [Profiler-Calibrated Backup Broker](06_profiler_calibrated_backup_broker.md)
   arbitrates Host capacity and D2H/RNIC bandwidth over `K -> K+1`.
7. [Lineage-Deduplicated Agent Protection](07_lineage_deduplicated_agent_protection.md)
   exploits shared prefixes and branch lifetimes in multi-agent DAGs.
8. [Mixed Restore/Recompute Recovery Broker](08_mixed_restore_recompute_broker.md)
   makes failure-time recovery a concurrent resource-scheduling problem.

## Evidence map

- [KV corpus manifest](../../related_works/kvcache_backup/manifest.csv) and
  [30-paper review](../../related_works/kvcache_backup/per-paper-review.md):
  1 direct KV fault-tolerance paper and 29 explicitly labeled adjacent systems
  papers.
- [Training/checkpoint corpus](../../related_works/training_checkpoint/):
  30 validated 2025--2026 papers, with per-paper staging and commit analysis.
- [Live SGLang storage audit](evidence/live_sglang_storage_audit.md): the
  existing `GPU -> Host -> storage attempt -> operation completion` path, its
  partial-write commit gap, Host-pool semantics, TP completion rule, and
  concrete KV sizes.
- [TP/PP/DP/EP recovery audit](evidence/parallelism_recovery_audit.md): minimum
  recovery state, logical-head deduplication, per-parallelism failure scope,
  and the peer-HBM value boundary.
- [Real SGLang profiler report](evidence/sglang_profiler/REPORT.md): actual
  `profiler@fdebc938f7` traces plus a manipulated pinned-D2H interference run.

## Common correctness contract

All eight proposals use the same contract:

- `planned`, `copying`, or merely Host-resident state is not recoverable state;
- a worker-process-local Host allocation protects GPU/device-state loss, not
  process loss; process scope requires an external payload owner and a durable
  or rebuildable directory;
- an extent becomes usable only after version, checksum, every required logical
  component, and either a full-copy ACK or coded quorum that survives the
  declared fault are complete;
- physical replicas of one logical attention shard are alternatives, not
  additional conjunctive components;
- the system separately tracks materialized-KV frontier `K`,
  scope-protected/group-committed KV frontier `C`, and emitted-output frontier
  `E`; the published recovery frontier is `C`;
- payload commitment also requires a manifest/directory that survives the same
  fault scope or is deterministically rebuildable from self-describing
  surviving objects and ACK records;
- stale, revoked, partially copied, or topology-epoch-mismatched state falls
  back to recomputation;
- foreground inference may revoke unstarted backup credits, but cannot revoke a
  committed protection lease before its dependency lifetime ends.

This keeps correctness independent of slack prediction and Algorithm 3's
control quality.
