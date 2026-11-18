# SGLSim paper-figure evidence audit

This is the S1-S9 subset of the 2026-08-27 AegisKV figure audit.  The virtual
target figures define research questions and visual structure, not values that
the measured or simulated results must be forced to match.

| Figure | Current evidence | Important limitation | Status |
|---|---|---|---|
| S1 | Six architectures, two failure domains, ten paired seeds, 60/60 cells and shared policy-free probes | Worker-fault curves mainly share the modeled 60 s replacement term; fine-grained policy separation is small | Visual PASS; preliminary, not paper-ready |
| S2 | Five concurrency levels, three failure rates, ten paired seeds, 50/50 jobs | C=8/16 loses benefit as pressure reaches 100 percent and Peer-HBM protection approaches zero; delays remain modeled | Visual PASS; preliminary, not physical |
| S3 | Qwen3-8B 108/108 cells and Qwen3-32B 300/300 cells over DP/context | Fixed probes dilute all-workflow gain as DP grows; the 32B profile is analytic | Visual PASS; preliminary, not paper-ready |
| S4 | Three algorithm ablations with paired mechanism and WCT views | Mechanisms separate, but total-WCT effects remain weak or negative | Mechanism PASS; performance contribution not established |
| S5 | Three-seed low/high/recovery trace with bounded alpha and capacity checks | Residual pressure makes recovery windows nonmonotonic | Visual and mechanism PASS; modeled only |
| S6 | Sixty shared-prefix cells with per-tier epochs and lifecycle audits | Needs validation with a real prefix-identity workload and broader failure exposure | Mechanism PASS; modeled only |
| S7 | 168 cells and 2,016 frozen failure-target probes | Off-path copies flatten bandwidth effects; replacement/activation timing dominates | Visual PASS; modeled timing, not physical A100 evidence |
| S8 | Six architectures and two failure domains derived from S1 | Same ten-seed sample as S1, so it is not an independent replication | Derived visual PASS; not paper-ready |
| S9 | Independent 80-sample RTX4090 holdout, overall MAPE 3.72 percent | 15/16 cells meet 10 percent; 64 MiB Peer-HBM recovery is 10.64 percent | Calibration-only; preregistered all-cell gate FAIL |

Global claim rules:

1. Keep raw WCT, mechanism events, failures, seeds, confidence intervals, and
   audit reports tied to the figure lineage.
2. Label the A100-40GiB setup as a configuration model, never as physical A100
   measurement.
3. Do not select seeds, cells, or result signs to make the observed curves
   resemble target sketches.
4. Keep visual readiness, implementation verification, physical calibration,
   and paper-ready evidence as separate states.
