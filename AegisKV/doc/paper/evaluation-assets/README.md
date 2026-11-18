# AegisKV SGLSim evaluation assets

This directory is the SGLSim-owned subset of the paper material reconstructed
from Codex thread `01a005a8-95b8-7c41-af1b-50d06b711944` on 2026-08-27.
The source snapshot is
`/home/comp/zhanzhao/Mooncake/aegiskv/experiment_plan/paper_figure_pack/`.
The Mooncake source files were copied, not moved or modified.

## Scope

- `figures/s1.{pdf,png}` through `figures/s9.{pdf,png}` are the final
  SGLSim/calibration panels from the source figure pack.
- `caption-catalog.md` contains paper captions with the evidence qualifiers
  preserved.
- `figure-gap-audit.md` records what each panel proves, what it does not prove,
  and the remaining evidence gap.
- `manifest.json`, `insertion-audit.csv`, and `SHA256SUMS` bind the migrated
  copies to the source snapshot.
- `figure-proof.tex` builds an acmart/EuroSys-width insertion proof.  The
  rendered proof is written to
  `output/pdf/aegiskv-sglsim-figure-insertion-proof.pdf` at the repository root.

Mooncake-owned physical panels R1-R6 and the full Mooncake paper PDFs are
intentionally excluded.  S9 is retained because it validates the simulator's
four-direction timing model, but its RTX4090 measurements remain external
calibration evidence.

## Evidence boundary

These panels are visually insertion-ready, not paper-ready empirical evidence.
They are metadata-DES results, derived views, or calibration-only evidence;
they are not physical A100 measurements.  In particular:

- S1 and S8 share the same ten-seed sample, so S8 is not an independent
  replication.
- S3's Qwen3-32B service profile and failover delays are analytic.
- S4 separates mechanism behavior, but its end-to-end WCT effects are weak or
  negative.
- S7 is dominated by modeled replacement/activation timing.
- S9 passes 15 of 16 per-cell 10-percent-MAPE checks and therefore does not
  pass the preregistered all-cell calibration gate.

Keep `paper_ready=false` until the underlying paper requirements are met.  A
successful LaTeX build or a visually clean figure is not evidence of physical
replication.

## Local lineage

The result arrays, audit reports, and figure-generation code remain under
`AegisKV/experiments/sglsim_wct_preliminary_v1/`.  The paper-local copies are
stable include targets; they do not replace those authoritative experiment
artifacts.

## Verification

From the repository root:

```bash
sha256sum -c AegisKV/paper/evaluation-assets/SHA256SUMS
sh AegisKV/paper/evaluation-assets/build-proof.sh
```
