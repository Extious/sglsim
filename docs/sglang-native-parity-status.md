# SGLang Native Parity Status

Active parity tests:

- `test/upstream_multi_agent/test_fault_injection_recovery.py`
- `test/upstream_multi_agent/test_remote_backup_cache_details.py`
- `test/upstream_multi_agent/test_remote_backup_prefetch.py`
- `test/upstream_multi_agent/test_chunked_prefill_backup_policy.py`
- `test/upstream_multi_agent/test_remote_backup_cross_rank.py`
- `test/upstream_multi_agent/test_failover_routing.py`
- `test/upstream_multi_agent/test_remote_backup_server_logging.py`
- `test/upstream_multi_agent/test_process_log_contract.py`

Runtime mode:

- authoritative online runtime: native SGLang parity
- legacy RuntimeService: removed from the authoritative online path

Current verification:

```bash
uv run --extra native pytest -q test/upstream_multi_agent
```

Result on 2026-05-18:

```text
24 passed
```

Performance simulation parity (native patch vs hook simulator):

| Capability | Status |
|------------|--------|
| blocking mode latency sleep | supported |
| offline trace replay (`created_time`, `future_queue`) | supported |
| overlap schedule timing model | supported |
| KV pool capacity estimate (`estimate_kv_cache_pool_capacity`) | supported |
| HiCache segmented host bandwidth model | supported |
| constant predictor fallback | supported |
| `bench_serving` / dataset tooling | not in scope (deferred) |

Verification:

```bash
uv run --extra native pytest -q test/unit/native/test_scheduler_simulator_runtime.py test/integration/test_native_offline_simulation.py
```
