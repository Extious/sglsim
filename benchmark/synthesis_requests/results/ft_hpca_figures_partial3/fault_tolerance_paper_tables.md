# Fault Tolerance Paper Tables

## Strategy Summary

| Metric | Value |
|---|---:|
| Recompute reduction | 57.5% |
| Critical-agent recompute reduction | 100.0% |
| GPU backup KV reuse | 33.4% |
| Output throughput change | 10.9% |
| Mean e2e latency change | -13.5% |
| Mean failover overhead change | -1.3% |

## Backup Lag

| Lag tokens | KV reuse (%) | Recompute tokens | Backed-up tokens |
|---:|---:|---:|---:|
| 0 | 25.3 | 43179.3 | 24479.7 |
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
| 0.50 | 0.0 | 0.0 | 43284.5 |
| 0.75 | 21605.5 | 4.0 | 21555.0 |
| 1.00 | 43153.5 | 8.0 | 0.0 |
