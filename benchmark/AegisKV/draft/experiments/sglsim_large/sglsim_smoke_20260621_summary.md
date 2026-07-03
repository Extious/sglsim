# SGLSim Smoke 20260621 Summary

This local smoke run validates the maintainable SGLSim benchmark entry for the
AegisKV large-scale experiment path.

## Command

```bash
bash benchmark/AegisKV/draft/experiments/sglsim_large/scripts/run_synthesis_smoke.sh
```

## Run

| Field | Value |
|---|---|
| Run directory | `benchmark/AegisKV/draft/experiments/sglsim_large/results/synthesis_smoke/20260621_021131` |
| Benchmark entry | `benchmark/synthesis_requests/run.py` |
| Model | `mock-model` |
| DP size | `1` |
| Workload | generated-shared-prefix smoke |
| Sessions | `4` |
| Turns per session | `2` |

## Metrics

| Metric | Value |
|---|---:|
| completed | `8` |
| failed | `0` |
| duration | `0.16189193725585938` |
| mean_e2e_latency_ms | `43.79372368566692` |
| median_e2e_latency_ms | `41.96141741704196` |
| p90_e2e_latency_ms | `50.89596612378955` |
| max_e2e_latency_ms | `51.09975999221206` |
| total_input | `808` |
| total_output | `128` |

## Consequence

The local SGLSim synthesis benchmark path can be launched from
`benchmark/AegisKV/draft/experiments/sglsim_large/` and can export run-level CSV
summaries. This is a plumbing smoke test only. It does not supply measured
AegisKV figure data because the online AegisKV policy path and failure/recovery
accounting are still missing.
