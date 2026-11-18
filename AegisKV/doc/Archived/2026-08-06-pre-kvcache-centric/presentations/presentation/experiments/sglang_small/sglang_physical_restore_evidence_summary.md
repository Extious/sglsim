# SGLang Physical Host-Restore Evidence Summary

This artifact makes the current physical Host-restore evidence auditable. It is parsed from real SGLang `server.log` files and failure summaries, not from SGLSim metadata rows.

Generated data:

- CSV: `benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/physical_restore_runs.csv`
- JSON: `benchmark/AegisKV/presentation/experiments/sglang_small/results/physical_restore_evidence/physical_restore_runs.json`
- Parser: `scripts/extract_physical_restore_evidence.py`
- Rows: 11
- States: `full_prefetch_materialized, host_frontier_not_visible_to_retry, partial_async_materialized`

## Current Rows

| Label | Job | Prefetch policy | Same write cost? | Host frontier tokens | Max data pages/request | Prefetch completed tokens | Retry cached tokens | State |
|---|---:|---|---|---:|---:|---:|---:|---|
| single_wait | 32983 | wait_complete | no | 1,297 | 1,297 | 1,297 | 1,297 | full_prefetch_materialized |
| unique_wait | 32987 | wait_complete | no | 5,204 | 1,301 | 5,204 | 5,204 | full_prefetch_materialized |
| shared_wait | 32991 | wait_complete | no | 5,188 | 1,297 | 5,188 | 5,188 | full_prefetch_materialized |
| same_write_wait_control | 32998 | wait_complete | yes, lookup disabled | 20,548 | 0 | 0 | 0 | host_frontier_not_visible_to_retry |
| same_write_wait_visible | 32999 | wait_complete | yes, lookup enabled | 20,548 | 5,137 | 20,548 | 20,548 | full_prefetch_materialized |
| same_write_best_effort_control | 33002 | best_effort | yes, lookup disabled | 20,548 | 0 | 0 | 0 | host_frontier_not_visible_to_retry |
| same_write_best_effort_visible | 33003 | best_effort | yes, lookup enabled | 20,548 | 5,137 | 2,816 | 2,830 | partial_async_materialized |
| same_write_best_effort_control_r2 | 34333 | best_effort | yes, lookup disabled | 15,425 | 0 | 0 | 0 | host_frontier_not_visible_to_retry |
| same_write_best_effort_visible_r2 | 34334 | best_effort | yes, lookup enabled | 15,425 | 5,137 | 2,304 | 2,318 | partial_async_materialized |
| same_write_best_effort_control_r3 | 34336 | best_effort | yes, lookup disabled | 15,425 | 0 | 0 | 0 | host_frontier_not_visible_to_retry |
| same_write_best_effort_visible_r3 | 34337 | best_effort | yes, lookup enabled | 15,425 | 5,137 | 1,792 | 1,806 | partial_async_materialized |

## Interpretation

The current SGLang evidence separates three cases that were previously mixed in prose:

1. Host frontier exists but is hidden from the retry rank: lookup-disabled controls write remote-backup tokens, but the retry path sees no `data_pages` and admits no storage-restored cached tokens.
2. Full physical Host restore materializes into retry cached tokens: wait-complete rows show positive `data_pages`, completed prefetch IO, and retry `#cached-token` equal to the restored page count.
3. Best-effort asynchronous restore can partially materialize before admission: best-effort rows may accept prefetch, complete a prefix subset, and admit only the pages ready at retry prefill.

This is stronger than metadata-only SGLSim evidence for Host restore materialization. It still is not final HPCA policy evidence because it is small-scale, Host-only, and does not validate peer GPUReplica P2P physical copy or a repeated full AegisKV policy matrix.
