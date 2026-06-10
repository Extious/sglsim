# Communication Time Prediction Design

**Status:** Draft | **Phase:** P0 not started

## Related plans

- [trace-calibration-design.md](trace-calibration-design.md) — export `comm_*_s` in trace schema after P4
- [cache-aware-scheduling-design.md](cache-aware-scheduling-design.md) — no direct dependency

## Goal

Replace fixed-bandwidth communication timing in SGLSim with a **layered communication predictor** for H2D, D2H, D2D, Host-Host, disk/L3, and optional NCCL collectives, while staying **metadata-only** (no CUDA, NCCL runtime, or `import sglang`).

This design does **not** change AIConfigurator **inference** prediction. It adds a separate layer for cache transfer, KV migration, backup/restore, and future distributed communication events.

## Non-goals

- Real NCCL/CUDA execution
- Adding collective time to every inference batch by default (see [Double-counting contract](#double-counting-contract))
- Copy-engine resource contention modeling in the first release
- LogGP multi-message Host-Host (reserved; alpha-beta first)

## Current state

Implementation today in [hicache.py](../../python/sglsim/srt/mem_cache/hicache.py):

- H2D: `num_pages * 4096 / memory_read_bandwidth_gb`
- D2H: `num_pages * 4096 / memory_write_bandwidth_gb`
- L3 backup: per-page `4096 / disk_write_bandwidth_gb`

[PlatformConfig](../../python/sglsim/srt/simulation/manager/config.py) exposes scalar bandwidths only. [transfer.py](../../python/sglsim/srt/mem_cache/transfer.py) has a separate `TransferConfig.duration_s` — unify under the predictor in P0.

Problems:

- KV page size should be `page_size * kv_bytes_per_token`, not fixed 4096
- No fixed latency (alpha-beta)
- D2D / H2H not first-class
- NCCL not modeled outside inference predictor; risk of **double-counting** TP collectives

## Design principles

1. Metadata-only simulator; no upstream SGLang runtime import.
2. Communication timing is an explicit service, not embedded formulas in `HiCacheController`.
3. Table lookup when data exists; alpha-beta fallback; legacy bandwidth last.
4. Per-path and per-source metrics for experiments.
5. `add_inference_collectives` defaults to **false**.

## Architecture

```text
python/sglsim/srt/communication/
  __init__.py
  types.py              # CommPath, CommunicationRequest, CommunicationPrediction
  config.py             # CommunicationConfig from sim JSON
  predictor.py          # CommunicationTimePredictor (hybrid chain)
  table.py              # CSV interpolation (P2)
  aiconfigurator_comm.py # query_nccl adapter (P3)
  kv_bytes.py           # kv_page_bytes helper (P0)
```

### Hybrid prediction order

1. `NCCL` + AIConfigurator enabled -> `query_nccl` / `query_custom_allreduce`
2. Empirical table row match -> interpolate (P2)
3. Alpha-beta: `duration_s = alpha_us/1e6 + bytes/(bandwidth_gb_s*1e9)`
4. Legacy `PlatformConfig` bandwidth (compatibility)
5. Clear error if nothing applies

### Scheduler overlap (unchanged initially)

```text
visible_load   = max(h2d_duration - previous_inference_duration, 0)
visible_backup = max(d2h_or_disk_duration - current_inference_duration, 0)
```

HiCache integration: build `CommunicationRequest`, call `predict()`, use `pred.duration_s`.

## Double-counting contract

AIConfigurator inference latency may already include TP allreduce. Therefore:

- **Do not** add `CommPath.NCCL` to every batch.
- Use NCCL only for **explicit** events (future remote KV, etc.).
- Config default:

```json
{
  "communication": {
    "add_inference_collectives": false
  }
}
```

Document in [trace-calibration-design.md](trace-calibration-design.md): latency scale factors must not compensate for missing comm model conflation.

## Cross-plan integration

| Plan | Integration |
|------|-------------|
| Trace calibration | After P4, include `comm_h2d_s`, `comm_d2h_s`, `comm_disk_write_s`, `comm_source_*` in trace `run_meta` / aggregates |
| Cache-aware scheduling | Independent |
| Multi-DP per-rank clocks | Per-rank overlap locals unchanged; cross-rank KV uses NCCL path only when explicitly modeled |

Run trace HiCache-heavy workloads **after** P0 (correct `kv_page_bytes`) + HiCache wired to predictor.

## Phases

| Phase | Deliverables | Acceptance | Dependencies |
|-------|--------------|------------|--------------|
| **P0** | `types.py`, `config.py`, `predictor.py`, `kv_bytes.py`; alpha-beta + legacy; wire HiCache H2D/D2H/L3; unify `transfer.py` | Unit tests pass; changing `kv_bytes_per_token` changes duration; default configs similar magnitude to today | None |
| **P1** | Metrics `comm_*_s` on cache / `/metrics` | Benchmark JSON includes comm breakdown | P0 |
| **P2** | `table.py`, sample CSV under `reference/comm/`, hybrid table-first | Table overrides alpha-beta for H2D; missing row falls back in hybrid | P0 |
| **P3** | `aiconfigurator_comm.py`, fake DB tests | `CommPath.NCCL` works in tests without LFS data; no inference batch NCCL unless flag set | P0 |
| **P4** | Docs, materialized `communication` in `simulator_config.json`, examples A100/H100/mock | Users see source breakdown in benchmark output | P1–P3 |

### P0 detail (must-ship first)

- `kv_page_bytes(page_size_tokens, kv_bytes_per_token)` from model metadata; mock-model compatibility default documented in materialized config
- Replace all `num_pages * 4096` in HiCache with `num_pages * kv_page_bytes`
- Modes: `legacy`, `alpha_beta`, `hybrid` (table optional until P2)

### Tests to add

- `test_comm_alpha_beta_predictor.py`
- `test_comm_table_interpolation.py` (P2)
- `test_comm_aiconfigurator_adapter.py` (P3)
- `test_hicache_uses_kv_page_bytes.py`
- `test_hicache_comm_metrics.py` (P1)

```bash
uv run --extra test pytest test/unit/srt test/unit/kvcache -q
uv run --extra test pytest test/unit test/integration -q
```

## Status tracker

| Phase | Status | Notes |
|-------|--------|-------|
| P0 | not started | Predictor + kv bytes + HiCache wire |
| P1 | not started | Metrics |
| P2 | not started | Empirical tables |
| P3 | not started | AIConfigurator NCCL adapter |
| P4 | not started | Benchmark + trace export |

## Config schema

Simulator JSON `communication` section:

| Field | Description |
|-------|-------------|
| `mode` | `legacy` \| `alpha_beta` \| `table` \| `hybrid` |
| `default_memory_kind` | `pinned` \| `pageable` \| `unknown` |
| `tables` | Map path -> CSV path |
| `alpha_beta` | Per-path `{alpha_us, bandwidth_gb_s}` |
| `aiconfigurator_collectives.enabled` | NCCL via AIConfigurator |
| `fallback_to_legacy_bandwidth` | Last resort |
| `add_inference_collectives` | Default **false** |
| `kv_bytes_per_token` | Optional override for mock-model |

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| AIConfigurator LFS pointers | HYBRID + fake DB in unit tests |
| Hardware-specific tables misused | Platform metadata in CSV; materialize paths in benchmark output |
| Double-counting NCCL | `add_inference_collectives=false`; explicit events only |
| Copy-engine complexity | Defer; keep overlap formulas |
| Mock-model test breakage | Explicit compatibility default + test |

## Appendix: types (reference)

See original `CommunicationRequest` / `CommPath` definitions — implement in `communication/types.py` as specified in prior revision (H2D, D2H, D2D, H2H, DISK_READ, DISK_WRITE, NCCL).

## Appendix: example simulator config

```json
{
  "model_path": "Qwen/Qwen3-8B",
  "platform": {
    "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
  },
  "scheduler": {
    "backend_name": "sglang",
    "backend_version": "0.5.10",
    "tp_size": 1,
    "dp_size": 1
  },
  "cache": {
    "page_size": 16,
    "enable_hierarchical_cache": true,
    "hicache_ratio": 2.0,
    "hicache_write_policy": "write_through"
  },
  "communication": {
    "mode": "hybrid",
    "default_memory_kind": "pinned",
    "add_inference_collectives": false,
    "tables": {
      "h2d": "reference/comm/a100_sxm_h2d.csv",
      "d2h": "reference/comm/a100_sxm_d2h.csv"
    },
    "alpha_beta": {
      "h2d": {"alpha_us": 8.0, "bandwidth_gb_s": 48.0},
      "d2h": {"alpha_us": 8.0, "bandwidth_gb_s": 48.0},
      "disk_write": {"alpha_us": 30.0, "bandwidth_gb_s": 8.0}
    },
    "aiconfigurator_collectives": {
      "enabled": true,
      "database_mode": "HYBRID"
    },
    "fallback_to_legacy_bandwidth": true
  }
}
```

## Appendix: empirical CSV format

```csv
path,src,dst,memory_kind,size_bytes,latency_us,bandwidth_gb_s,source
h2d,host0,gpu0,pinned,4096,12.5,0.33,cuda_bandwidthTest
h2d,host0,gpu0,pinned,1048576,45.0,23.3,cuda_bandwidthTest
```

Lookup: filter by path; wildcard src/dst; log-size interpolation; conservative extrapolation.

## Appendix: data collection (offline)

- Host-device: CUDA bandwidthTest / nvbandwidth (pinned and pageable, 4 KiB to GiB)
- Device-device: nvbandwidth P2P matrix
- NCCL: nccl-tests for `tp_size` / future EP shapes
- Host-host: OSU benchmarks (optional until simulator consumes H2H)

## Links

- [sglsim-srt-contract.md](../sglsim-srt-contract.md)
- [reference/sglang_notes.md](../../reference/sglang_notes.md)
