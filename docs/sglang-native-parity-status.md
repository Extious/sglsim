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
