# Fault Tolerance Paper Tables

## Strategy Summary

| Metric | Value |
|---|---:|
| Recompute reduction | 57.5% |
| Critical-agent recompute reduction | 100.0% |
| GPU backup KV reuse | 50.0% |
| Output throughput change | 21.6% |
| Mean e2e latency change | -16.9% |
| Mean failover overhead change | -1.3% |

## Backup Lag

| Lag tokens | KV reuse (%) | Recompute tokens | Backed-up tokens |
|---:|---:|---:|---:|
| 0 | 38.0 | 64769.0 | 36719.5 |
| 1 | 25.0 | 64615.0 | 21567.5 |
| 4 | 13.0 | 43151.5 | 15152.0 |
| 8 | 13.0 | 43155.0 | 15149.0 |
| 16 | 24.9 | 21652.5 | 21525.5 |
| 32 | 25.0 | 21619.5 | 21541.0 |

## Critical-Agent Coverage

| Critical ratio | Critical reused | Critical recompute | Non-critical recompute |
|---:|---:|---:|---:|
| 0.00 | 0.0 | 0.0 | 43321.5 |
| 0.25 | 0.0 | 0.0 | 43177.5 |
| 0.50 | 0.0 | 0.0 | 86569.0 |
| 0.75 | 43211.0 | 8.0 | 43110.0 |
| 1.00 | 86307.0 | 16.0 | 0.0 |
