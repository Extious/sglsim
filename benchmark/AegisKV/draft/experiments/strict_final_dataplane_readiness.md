# Strict-Final Data-Plane Readiness

This report separates final policy-sweep rows from real SGLang recovery-path evidence.
It must not relabel SGLSim metadata materialization as physical tensor KV movement.

- Status: `strict_final_ready`
- Strict-final ready: `true`
- Host recovery matrix ready: `true`

## P0 SGLSim Policy Rows

- Restore rows: 693
- Physical tensor copy rows: 0
- Non-final restore rows: 693
- Data planes: `{'metadata_materialization': 693}`
- Timing sources: `{'physical_recovery_path_csv': 693}`

## P0 Per-Suite Live Movement Audits

- Audit exists: `true`
- Consistent with P0 summary: `true`
- Restore rows: 693
- Physical tensor copy rows: 0
- Non-final restore rows: 693
- Missing data-plane rows: 0
- Missing timing-source rows: 0
- Schema complete: `true`

## P0 Physical Calibration Bridge

- Bridge ready: `true`
- Bridge eligible rows: 693/693
- Bridge missing rows: 0
- Bridge detail required rows: 693
- Bridge detail digest: `c6fba6283119b769aa800596a7185d055db97f42bd615924ea3ef3538c29b8f5`
- Restore actions: `{'gpu_replica': 318, 'host_restore': 375}`
- Physical actions: `{'gpu_replica_restore_p2p': 318, 'host_restore_h2d': 375}`
- Evidence tiers: `['sglang-small-physical-recovery-movement-smoke']`

## P0 Live Physical Tensor-Copy Candidate Audit

- Audit exists: `true`
- Status: `strict_final_candidate_ready`
- Strict-final candidate ready: `true`
- Consistent with current bridge rows: `true`
- Candidate bridge digest: `c6fba6283119b769aa800596a7185d055db97f42bd615924ea3ef3538c29b8f5`
- Required rows: 693
- Candidate rows: 693
- Accepted rows: 693
- Missing rows: 0
- Invalid rows: 0
- Extra rows: 0

## P0 Live Physical Tensor-Copy Candidate Generation

- Audit exists: `true`
- Status: `candidates_complete`
- Consistent with current bridge rows: `true`
- Generation bridge digest: `c6fba6283119b769aa800596a7185d055db97f42bd615924ea3ef3538c29b8f5`
- Event schema version: `p0-live-copy-event-v1`
- Event rows: 693
- Event sources: 2
- Missing event sources: 0
- Accepted rows: 693
- Missing rows: 0
- Rejected event rows: 0

## P0 Live Physical Tensor-Copy Event Template

- Event schema version: `p0-live-copy-event-v1`
- CSV exists: `true`
- Schema doc exists: `true`
- CSV: `/home/comp/zhanzhao/sglsim/benchmark/AegisKV/draft/experiments/sglsim_large/results/strict_final_readiness/p0_physical_tensor_copy_events.template.csv`
- Schema doc: `/home/comp/zhanzhao/sglsim/benchmark/AegisKV/draft/experiments/sglsim_large/results/strict_final_readiness/p0_physical_tensor_copy_event_schema.md`
- Template data rows: 0
- Template fields: `['suite_label', 'request_csv', 'rid', 'event_schema_version', 'event_kind', 'restore_action', 'restored_tokens', 'aegis_replay_restore_data_plane', 'aegis_replay_restore_timing_source', 'physical_copy_verified', 'copy_latency_ms', 'copy_bytes', 'source_artifact', 'physical_action', 'src_rank', 'dst_rank', 'src_device', 'dst_device', 'copy_backend', 'copy_started_at_ns', 'copy_ended_at_ns']`
- Required-event manifest exists: `true`
- Required-event manifest: `/home/comp/zhanzhao/sglsim/benchmark/AegisKV/draft/experiments/sglsim_large/results/strict_final_readiness/p0_physical_tensor_copy_events.required.csv`
- Required-event rows: 693
- Required event kinds: `{'live_host_h2d_copy': 375, 'live_peer_p2p_copy': 318}`
- Required restore actions: `{'gpu_replica': 318, 'host_restore': 375}`

## P0 Live Physical Tensor-Copy Collection

- Collection exists: `true`
- Output CSV: `/home/comp/zhanzhao/sglsim/benchmark/AegisKV/draft/experiments/sglsim_large/results/strict_final_readiness/p0_physical_tensor_copy_events.live.csv`
- Selected rows: 693/693
- Verified event rows: 375
- Event rows by kind: `{'live_host_h2d_copy': 375}`
- Skipped rows: 318
- Skipped rows by reason: `{'CUDA peer access is unavailable for 0->1': 318}`
- CUDA device count: 2
- CUDA peer access: `{'0->1': False, '1->0': False}`
- Measurement repeats: 1
- Bytes per token: 147456

## P0 Live P2P Collection Plan

- Plan exists: `true`
- Status: `p2p_live_copy_complete`
- P2P ready: `true`
- Required P2P rows: 318
- Required P2P tokens: 112684
- Existing P2P live rows: 318
- Existing P2P skipped rows: 0
- Remote target: `hkbugpumanode:~/aegiskv_sglsim_large`
- Sbatch: `benchmark/AegisKV/draft/experiments/sglsim_large/slurm/aegiskv_p0_live_p2p_copy_events.sbatch`

## P0 Promoted Strict-Final Overlay

- Promotion exists: `true`
- Status: `promoted_strict_final_ready`
- Promotion ready: `true`
- Consistent with current bridge rows: `true`
- Restore rows: 693
- Physical tensor copy rows: 693
- Non-final restore rows: 0
- Host H2D candidate rows: 375
- Peer P2P candidate rows: 318
- Problems: `[]`

## Real SGLang Host Recovery Matrix

- Matrix status: `pass`
- Accepted pairs: 6/6
- Physical restore rows: 15
- Physical restore CSVs: 3

## Real SGLang Host Restore Materialization

- CSVs: 3
- Rows: 15
- Materialized rows: 9
- Full prefetch materialized rows: 6
- Partial async materialized rows: 3
- Hidden-control rows: 6
- Retry cached tokens after accept: 80287
- Prefetch IO completed tokens: 80245
- States: `{'full_prefetch_materialized': 6, 'host_frontier_not_visible_to_retry': 6, 'partial_async_materialized': 3}`

## Bound Slurm Jobs

- Checked at: `2026-07-03T00:43:50+08:00`
- Status: `complete`
- Pending/running/completed/failed: 0/0/4/0

## Next Required Actions

- No remaining data-plane readiness action.
