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
| 1 | 50.0 | 43077.0 | 28802.0 |
| 4 | 8.7 | 43130.3 | 10101.3 |
| 8 | 8.7 | 28770.0 | 10099.3 |
| 16 | 16.6 | 14435.0 | 14350.3 |
| 32 | 16.6 | 14413.0 | 14360.7 |

## Critical-Agent Coverage

| Critical ratio | Critical reused | Critical recompute | Non-critical recompute |
|---:|---:|---:|---:|
| 0.00 | 0.0 | 0.0 | 43248.0 |
| 0.25 | 0.0 | 0.0 | 28785.0 |
| 0.50 | 0.0 | 0.0 | 28856.3 |
| 0.75 | 14403.7 | 2.7 | 14370.0 |
| 1.00 | 28769.0 | 5.3 | 0.0 |
