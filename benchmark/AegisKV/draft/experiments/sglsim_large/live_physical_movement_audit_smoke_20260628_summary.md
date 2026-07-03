# Live Physical Movement Audit Smoke

Run id: `sglsim-live-movement-audit-smoke-20260628`

This smoke verifies that SGLSim request rows now record the recovery data plane
and timing source explicitly.

Environment:

- `SGLSIM_AEGIS_RECOVERY_MODEL=physical_copy`
- `SGLSIM_AEGIS_PHYSICAL_COPY_CSV=.../physical_recovery_path_aggregate.csv`

Result:

- request rows: 8
- hard-failover replay rows: 4
- restore rows: 4
- `aegis_replay_restore_data_plane=metadata_materialization`: 4
- `aegis_replay_restore_timing_source=physical_recovery_path_csv`: 4
- `physical_tensor_copy` rows: 0
- strict live-physical-movement audit: failed as expected

Artifacts:

- Suite root: `results/agentic_workflows/sglsim-live-movement-audit-smoke-20260628/`
- Audit CSV: `results/agentic_workflows/sglsim-live-movement-audit-smoke-20260628/live_physical_movement_audit.csv`
- Audit report: `results/agentic_workflows/sglsim-live-movement-audit-smoke-20260628/live_physical_movement_audit.md`

Boundary:

This smoke does not improve the current figure tier. It adds a machine-readable
gate: final physical recovery evidence requires restore rows with
`aegis_replay_restore_data_plane=physical_tensor_copy`, not
`metadata_materialization` plus calibrated sleep.
