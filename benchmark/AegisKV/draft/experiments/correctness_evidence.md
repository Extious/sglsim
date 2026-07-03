# AegisKV Correctness Evidence

This note maps the current correctness claims to unit-level artifacts. These
tests are not a substitute for final SGLang failure-injection experiments, but
they guard the SGLSim/SRT metadata invariants that the paper design depends on.

Command:

```bash
python -m pytest test/unit/srt/test_aegis_restore_lookup.py -q
```

Current result:

```text
9 passed
```

## Covered Invariants

| Claim | Test artifact | Evidence scope |
|---|---|---|
| Pending or missing HostBackup is not safe at failure time | `test_aegis_restore_lookup_recomputes_when_host_backup_missing` | Engine retry lookup falls back to recompute when no ACKed directory entry exists. |
| Unsupported pending restore action does not create device KV | `test_aegis_pending_restore_action_does_not_materialize_device_cache` | Scheduler restore materialization ignores non-ACKed/pending restore actions. |
| `lock_ref > 0` paths are not evicted | `test_aegis_restored_locked_prefix_is_not_evicted_under_pressure` | A restored radix prefix remains device-resident under eviction pressure while locked. |
| Radix insert after restore preserves the frontier span | `test_aegis_restored_prefix_survives_divergent_radix_insert` | A divergent sibling insert does not destroy the restored retry prefix. |
| Host eviction/unavailability invalidates Host frontier selection | `test_aegis_restore_lookup_recomputes_when_host_evicted_and_peer_unavailable` | Engine lookup chooses recompute when Host tokens are unavailable and the retry rank lacks peer GPUReplica. |
| Peer demotion invalidates GPUReplica frontier selection | `test_aegis_restore_lookup_demotes_peer_gpu_when_retry_rank_is_not_peer` | GPUReplica tokens are exposed only when the retry rank is the recorded peer rank. |
| Baseline requests preserve SGLang behavior | `test_aegis_baseline_request_without_metadata_does_not_publish_frontier` | Requests without Aegis metadata do not publish Host/GPUReplica frontiers or candidate retries. |

## Still Not Final Evidence

The current tests validate metadata and control-flow invariants inside SGLSim's
SRT mirror. Final paper evidence still needs:

- real SGLang failure retry experiments;
- physical Host H2D and peer GPUReplica copy measurements;
- final policy sweeps marked `evidence_tier=final`;
- regression coverage against a pinned SGLang source-level behavior audit.
