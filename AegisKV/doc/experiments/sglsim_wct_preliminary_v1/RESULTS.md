# SGLSim WCT preliminary results

Status: **preliminary simulation only** (`paper_ready=false`). These results are
true workflow completion times from the local DES: workflow arrival through
serving-lane queue, service, visible protection-copy burden, replacement,
recovery, downstream queue propagation, and final completion. They are not the
older Slack-harm surrogate and are not physical A100 measurements.

## Passed result lineages

- Center S1/S4 and mechanism gate: `results/preliminary_v1/audit.json`
  (`PASS`, 16/16 checks). Raw per-workflow and per-probe arrays are in
  `results/preliminary_v1/raw/*.npz`; seed-level values and paired-bootstrap 95%
  CIs are in `summary.csv` and `aggregate_ci.csv`.
- Fixed-arrival S2/S5: `results/s2_fixed_arrival_v2/audit.json` (`PASS`, 9/9
  checks). Every cell has the same 90,000 ms arrival cutoff and controller
  opportunities at 30/60/90 seconds. Completion drains after the cutoff
  (92.08--107.08 seconds). Raw arrays, seed rows, and aggregate CIs are under
  that directory.
- The failed diagnostic `results/s2_fixed_duration_v1/audit.json` is retained
  unchanged and must not be plotted: proportional workflow counts still gave
  unequal trace horizons and 3--5 controller windows.
- Configs and seeds are frozen in `protocol.json` and
  `s2_fixed_arrival_protocol.json`. Each summary records the config, runner,
  parent-runner, and raw-file SHA-256 values.

## Preliminary numerical readout

### Targeted warm-failover benefit follow-up

The independently audited targeted follow-up is under
`results/targeted_benefit_v1/`. It was selected after the broad near-null
campaign and exploratory diagnostics, so it is reported as targeted
preliminary evidence rather than a preregistered general result. The complete
rectangular AegisKV/Native matrix covers remote DRAM at 5/10/16 GB/s, common
takeover cost at 0/500 ms, 1/2/4 active requests per GPU, five paired seeds,
256 workflows per cell, and 24 source-worker/GPU failure probes per cell. The
model is Qwen3-8B FP16 with a 40K context and system-DP=64 on an A100-40GB
hardware model; same-node Peer-HBM is 50 GB/s.

At remote DRAM 5 GB/s and 500 ms common takeover, AegisKV's affected-workflow
WCT reduction versus modeled Native Mooncake is:

| Active requests/GPU | Mean WCT reduction (95% CI) | p95 WCT reduction (95% CI) | Pairwise win fraction | Healthy mean-WCT overhead |
|---:|---:|---:|---:|---:|
| 1 | 24.57% [23.12%, 26.15%] | 24.56% [21.44%, 29.36%] | 97.89% | +0.00052% |
| 2 | 18.68% [17.80%, 19.62%] | 24.39% [21.69%, 27.10%] | 83.40% | -0.00333% |
| 4 | 15.64% [14.90%, 16.56%] | 6.78% [3.05%, 10.52%] | 73.22% | -0.12889% |

In the preregistered architecture comparison at 2 active requests/GPU,
AegisKV has the best modeled p95 reduction (24.39% versus GhostServe's 21.65%)
and the second-best mean reduction (18.68% versus GhostServe's 19.88%). This is
a strong intended-regime result, not evidence that AegisKV dominates every
policy under every load. The audit passes 12/12 integrity checks over 200 raw
cells without any performance threshold. It verifies paired workload/fault
tuples, raw hashes, source-owned independent HBM/DRAM paths, and zero
target-owned or HBM-to-DRAM copies.

At the center cell (Qwen3-8B, system-DP=64, target active requests/GPU=4,
failure rate 5.952 per 1K GPU-shard-hour), mean WCT normalized to modeled Native
Mooncake is:

| Variant | Normalized mean WCT | Paired 95% CI |
|---|---:|---:|
| Native Mooncake (modeled) | 1.000000 | [1.000000, 1.000000] |
| AegisKV full | 1.000001 | [0.999993, 1.000012] |
| no Algorithm 1 | 1.000004 | [1.000004, 1.000005] |
| no Algorithm 2 | 0.999999 | [0.999999, 1.000000] |
| no Algorithm 3 | 1.000001 | [0.999993, 1.000012] |

Thus the center-cell full result is null/slightly worse, not a simulated win.
The S1 full-range affected-workflow CDF is dominated by the fixed 60-second
replacement term. The target-aligned revision keeps the architecture-level CDF
and places p50/p95/p99 shifts versus modeled Native beside it, exposing the
near-overlap without manufacturing visual separation.

For S2 at rate 5.952, the AegisKV/modeled-Native ratios for target concurrency
1/2/4/8/16 are 1.000008, 1.000008, 1.000000, 0.964599, and 0.717211. The large
high-concurrency separation is a result of this simulator's registered
normal-path copy/fabric queue model and is artifact-sensitive; it is not a
physical-performance claim. In particular, the c=16 reduction is already 28.5%
at zero failures and changes only to 28.3%/27.7% at the two nonzero rates, so it
must not be described as a failure-recovery advantage. The frozen 90-second
trace has only 0.0095/0.0333 expected faults at those rates.

S5 uses only recorded fixed-arrival state. At realized loads
1.000/1.945/3.876/7.744/14.933 active requests per GPU, mean final alpha is
0.3000/0.2207/0.03825/0.03125/0.03125, corresponding exactly to
12.00/8.83/1.53/1.25/1.25 GiB under the configured 40 GiB HBM capacity. Alpha
never reaches zero. The residency panel reports event fractions only; it does
not infer occupancy GiB.

## Model fidelity and evidence boundary

- Native Mooncake is a **modeled approximation**: native-like round-robin
  routing plus asynchronous Store DRAM backup/restore through an existing
  host-Store component. It is not production Mooncake Router/Store failover.
- Native Mooncake and RAIDServe instantiate the same `RAIDSERVE_HOST` component
  in this TP1 model and are implementation-equivalent. RAIDServe remains in raw
  tables but its duplicate S1 curve is omitted; these are not two independent
  pieces of comparative evidence.
- Same-node/cross-node Peer-HBM and local/remote DRAM have independent bandwidth
  parameters and event counts. The center performance cells naturally observe
  `HBM_AND_DRAM`, `DRAM_ONLY`, and `BYPASS`; `HBM_ONLY` is not manufactured.
  A separate preregistered mechanism gate passes all four outcomes.
- Failure rates use a sparse, single-fault, first-order Poisson mixture over
  executed source-GPU/source-node fault replays. This is not a multi-fault
  continuous-time campaign.
- Qwen3-32B, DP16--256 scale, and separate topology/reuse/occupancy sensitivity
  matrices were not run in this first preliminary batch.

## Figures

PDF and PNG versions are under `results/preliminary_v1/figures/`:

- `s0_target_gap_diagnostic` (derived diagnostic; not a new experiment)
- `s1_wct_cdf_preliminary`
- `s2_concurrency_failure_preliminary`
- `s4_ablation_preliminary`
- `s5_alpha_load_preliminary`
- `results/targeted_benefit_v1/figures/targeted_benefit_preliminary`

`figures/manifest.json` binds S1/S4 to the center audit, S2/S5 to the passed
fixed-arrival-v2 audit, records the superseded S2 audit, and contains PDF hashes.

## S3/S6/S7 fixed-arrival extension

The extension manifest is `extension_manifest.json`. All three campaigns use
Qwen3-8B, three paired seeds, a common 90-second arrival cutoff, and controller
opportunities at 30/60/90 seconds.

- S3 (`results/s3_preliminary_v1`) passes 11/11 checks and covers system-DP
  16/32/64/128/256 plus context 4K/8K/16K/32K/40K. At rate 5.952, all
  AegisKV/modeled-Native WCT deltas are close to zero; the largest magnitude is
  -167 ppm at DP16. This is a near-null scale/context result, not a broad win.
- S6 final lineage is `results/s6_preliminary_v3` and passes 15/15 checks.
  `s6_preliminary_v1` and `v2` are explicitly invalidated. Occupancy is the
  integral of actual controller `used_bytes` samples over 90 seconds; every raw
  NPZ stores the step trace and audit reconstructs the HBM/DRAM byte-ms
  integrals. For reuse 0/25/50/75%, mean HBM occupancy is
  3.638/3.239/2.512/1.423 GiB/GPU and DRAM occupancy is
  6.009/4.593/3.052/1.462 GiB/GPU. Copy amplification is
  0.983/1.035/1.036/0.850; mean eviction counts are
  1513/1338/1066/135; recovery-hit fractions are
  0.381/0.387/0.394/0.372.
- S7 (`results/s7_preliminary_v1`) passes 14/14 checks after an independent
  re-audit corrected a gate that had mistakenly counted distinct values across
  seeds instead of pairing 30/60-second cells. The raw data were not changed.
  `cross+local` is the only topology with a material preliminary delta
  (AegisKV/modeled-Native = 1.001226); the other three are near parity. Raising
  repair delay from 30 to 60 seconds raises whole-workload fault-conditioned
  mean WCT from about 2371 to 2854 ms for both policies.

Extension figures:

- `results/s3_preliminary_v1/figures/s3_scale_context_preliminary.{pdf,png}`
- `results/s6_preliminary_v3/figures/s6_reuse_occupancy_preliminary.{pdf,png}`
- `results/s7_preliminary_v1/figures/s7_topology_repair_preliminary.{pdf,png}`

## Reproduction commands

```bash
PYTHONPATH=python python3 AegisKV/experiments/sglsim_wct_preliminary_v1/run_s2_fixed_duration.py \
  --output-dir AegisKV/experiments/sglsim_wct_preliminary_v1/results/s2_fixed_arrival_v2

python3 AegisKV/experiments/sglsim_wct_preliminary_v1/plot_results.py \
  --input-dir AegisKV/experiments/sglsim_wct_preliminary_v1/results/preliminary_v1 \
  --s2-dir AegisKV/experiments/sglsim_wct_preliminary_v1/results/s2_fixed_arrival_v2

PYTHONPATH=python python3 \
  AegisKV/experiments/sglsim_wct_preliminary_v1/run_targeted_benefit.py \
  --output-dir \
  AegisKV/experiments/sglsim_wct_preliminary_v1/results/targeted_benefit_v1
```
