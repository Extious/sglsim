# Table 4: Correctness Tests

| Test | Expected invariant | Current artifact |
|---|---|---|
| missing/pending HostBackup + failure | pending frontier rejected; retry recomputes | `test_aegis_restore_lookup_recomputes_when_host_backup_missing` |
| active `lock_ref` + eviction pressure | locked restored prefix is not evicted | `test_aegis_restored_locked_prefix_is_not_evicted_under_pressure` |
| divergent radix insert after restore | restored frontier token span still matches | `test_aegis_restored_prefix_survives_divergent_radix_insert` |
| unsupported pending restore action | no device KV is materialized before ACK | `test_aegis_pending_restore_action_does_not_materialize_device_cache` |
| Host unavailable + peer unavailable | unsafe frontier is not selected | `test_aegis_restore_lookup_recomputes_when_host_evicted_and_peer_unavailable` |
| peer demotion | GPUReplica frontier used only on peer rank | `test_aegis_restore_lookup_demotes_peer_gpu_when_retry_rank_is_not_peer` |
| request without Aegis metadata | no Aegis frontier or candidate retry is published | `test_aegis_baseline_request_without_metadata_does_not_publish_frontier` |
