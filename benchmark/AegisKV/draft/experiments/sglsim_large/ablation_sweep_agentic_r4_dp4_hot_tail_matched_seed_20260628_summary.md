# Fig. 12 Online Ablation Sweep Summary

Run id: `sglsim-ablation-sweep-agentic-r4-dp4-hot-tail-matched-seed-20260628`

Source CSVs:

- `results/agentic_workflows/sglsim-ablation-sweep-agentic-r4-dp4-hot-tail-matched-seed-20260628/agentic_workflow_paper_candidates.csv`
- `results/agentic_workflows/sglsim-ablation-sweep-agentic-r4-dp4-hot-tail-matched-seed-20260628/agentic_workflow_paper_aggregate.csv`
- `../../figures/data/measured/fig_ablation.csv`

Workload:

- DP size: 4
- Agent workflows: 32
- Turns: 3
- Shared-prefix shape: 64-token system prefix, 32-token question, 64 output tokens
- Critical-agent ratio: 0.25
- Failure injection: hard restart
- Retry route policy: peer
- Repeats: 3 matched seeds per variant (`8600`, `8601`, `8602`)

Result summary:

| Variant | Normalized WCT | Replay rows | Host restore | Peer GPU | Recompute |
|---|---:|---:|---:|---:|---:|
| AegisKV | 1.000 | 72 | 0 | 72 | 0 |
| no lifecycle | 1.007 | 72 | 0 | 0 | 72 |
| no segment granularity | 1.012 | 72 | 0 | 0 | 72 |
| no restore/recompute | 1.002 | 72 | 72 | 0 | 0 |
| no pressure | 1.010 | 72 | 0 | 0 | 72 |
| no hysteresis | 1.007 | 72 | 0 | 48 | 24 |
| no critical restore | 1.011 | 72 | 0 | 0 | 72 |

Evidence status:

- `evidence_tier=sglsim-agentic-online-ablation-sweep`
- `source=sglsim_agentic_online_ablation_sweep`
- Accepted by the HPCA hybrid gate as `hybrid-online-policy-sweep`.
- Still not strict-final evidence: SGLSim materializes metadata and uses calibrated physical-copy timing rather than live CUDA tensor movement in the online recovery path.
