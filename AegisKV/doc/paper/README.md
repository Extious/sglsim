# AegisKV EuroSys 2027 paper draft

> **Submission blocker (2026-08-27):** the editable System Overview source
> `../figures/aegiskv-paper-system-design-v2.drawio` and its current exports
> still contain the removed target-owned destage path. The TeX design section
> does not contain that path, but the build fingerprint does not inspect
> Draw.io XML. Rework and re-export the figure before treating a successful
> paper build as design-consistent. The newly migrated S1--S9 assets are also
> not yet included by `main.tex`.

> **Current simulation note (2026-08-29):** the newer three-layer complete
> campaign is `../../experiments/results/20260829T064719Z_paper_complete_v2`
> with S1--S9 plus E1 (10 PNG/PDF pairs). Those figures have not replaced the
> historical `evaluation-assets/` package and are not yet included by
> `main.tex`; this separation prevents a silent provenance swap. This lineage
> supersedes the earlier 8,192-token-prefix run and machine-checks the
> 4,096/256-token current-design default (with S9's labeled 1,024-token
> micro-verification override). It also replaces the numerically identical
> intermediate run whose pre-execution research snapshot still carried
> outcome numbers from an older E1 run.

This directory is an anonymous review draft organized as a complete paper:

- `Introduction` states the problem, insight, scope, and three design
  contributions;
- `Background and Motivation` contains the Real CC workflow analysis and the
  gap between local cache signals and workflow-visible failure cost;
- `System Design` is synchronized from `../design/current_design.md`;
- `Implementation` fixes the intended SGLang/Mooncake integration boundary;
- `Evaluation` fixes the research questions, paired methodology, fair
  baselines, and evidence gates without inventing unfinished results; and
- `Discussion and Limitations`, `Related Work`, and `Conclusion` close the
  claim boundary.

The Introduction, Implementation, Evaluation, Discussion, Related Work, and
Conclusion were added as paper-style sections on 2026-08-15.  The
Implementation section is currently a source-grounded mapping, not evidence
that the final physical prototype is complete.  The Evaluation section is a
frozen structure and methodology; quantitative claims remain gated on formal
artifacts.

## SGLSim evaluation asset package

The paper-local S1-S9 evaluation figures migrated from the Mooncake paper
workflow are under [`evaluation-assets/`](evaluation-assets/).  That package
contains stable PDF/PNG include targets, captions, a figure-level evidence
audit, source hashes, and a nine-page acmart insertion proof.  It intentionally
excludes Mooncake-owned physical panels R1-R6 and keeps every simulator and
calibration qualifier from the source session.

These assets are not included by `main.tex` automatically.  Visual insertion
readiness does not lift their `paper_ready=false` evidence status; each figure
must be introduced in Evaluation with its modeled, derived, or
calibration-only boundary intact.

## Design source and synchronization

`../design/current_design.md` is the only design source.  Do not first edit the
paper rendering or an architecture figure.  For each design revision:

1. update `current_design.md`;
2. synchronize its three-layer responsibilities, algorithms, protocols, and
   scope into `sections/03-system-design-v2.tex`;
3. keep the compiled pseudocode in `algorithms/slack-update.tex`,
   `algorithms/slack-landlord.tex`, and `algorithms/guarded-aimd.tex`, then
   replace all four design fingerprints with the output of
   `sha256sum ../design/current_design.md`; and
4. run `sh scripts/build.sh`.

The build checks the fingerprint before LaTeX starts.  This cannot judge the
quality of a translation, but it prevents a changed Markdown design from being
silently compiled with a previously synchronized TeX section.

## Build

The checked-in ACM template is under `template/acmart/acmart/`.  From this
directory, compile with:

```bash
sh scripts/build.sh
```

The local TeX trees vendor dependencies needed by the minimal TinyTeX
installation and contain the generated Libertine/NewTX font maps.  The build
therefore does not modify the system TeX installation.

The source uses EuroSys's ACM/SIGPLAN anonymous review mode:

```tex
\documentclass[sigplan,review,anonymous]{acmart}
```

Folio printing is enabled and the ACM reference/copyright block is suppressed.
The source also overrides acmart's inherited 8 pt bibliography with
`\normalsize`, because the EuroSys 2027 CFP requires at least 10 pt text on
12 pt leading, including references and captions.
The verified 2026-08-15 full-draft build has 11 physical pages: 9 pages of
technical content and 2 pages of references.  Recheck this count after every
result insertion; technical content and references are separated by an
explicit `\clearpage`.
Recheck the applicable EuroSys 2027 rule at submission time rather than
treating the current page count as a permanent format guarantee.
`sections/03-system-design-v2.tex` is its synchronized paper rendering.  The
older `sections/03-system-design.tex` is retained only as design history and is
not included by `main.tex`.

The paper uses an editable Draw.io architecture plus trace/model plots:

- `figures/real-cc-parallel-subagent-duration.pdf`, `figures/real-cc-workflow-8ab0a089.pdf`, and `figures/real-cc-failure-position-counterfactual.pdf`: Figure 1's workflow-level CDF, selected trace-reconstructed fan-out excerpt, and example-calibrated failure-target counterfactual;
- `figures/system-overview.pdf`: current three-layer architecture synchronized from `../figures/aegiskv-paper-system-design-v2.drawio`;
- `figures/system-design-v2.pdf`: audit copy of the same current architecture; the paper cites `system-overview.pdf`;
- `figures/motivation-kv-publication-comparison.pdf`: two-lane Motivation comparison of representative vLLM--MooncakeStoreConnector and SGLang HiCache publication timing, with no AegisKV content;
- `figures/motivation-recovery-scaling.pdf`: appendix/audit-only long-context overlay of three AIConfigurator full-prefix recomputation predictions plus uncalibrated peer-HBM, local-DRAM, and SSD defaults; it is deliberately excluded from the main submission because it is not physical failure-to-ready evidence;
- `figures/motivation-gpu-availability.png` and `preliminary-wct-kv-reuse.png`: workload input and preliminary result.

The source deliberately has no raster or placeholder fallback.  A missing
paper-local PDF therefore fails the build instead of silently producing an
incomplete submission.

## Submission checklist

- Replace the anonymous placeholder `ANON-000` with the actual submission ID.
- Recheck the EuroSys 2027 CFP and HotCRP instructions immediately before
  submission, including the required review-mode header and anonymity fields.
- Fully disclose the use of AI tools in the form and location required by the
  EuroSys 2027 CFP; do not assume this draft's source comments satisfy that
  disclosure requirement.
- Confirm that every figure label and caption remains at least 10 pt in the
  submitted PDF and that each trace/simulator/real-runtime evidence type is
  stated explicitly.
- Confirm US-letter or A4 output, a 178 x 229 mm text block, at least 8 mm
  between columns, page numbers, and grayscale readability against the final
  EuroSys checker rather than relying only on a successful LaTeX build.
- Re-run bibliography verification before submission.  Several directly
  relevant 2025--2026 systems are preprints whose titles or venue status may
  change.

## Scope and claim discipline

- Failure model: one declared fail-stop serving domain at a time.  Process/GPU
  and host loss are evaluated separately; colocated KV state is lost.  The
  application, Router, persistent KV Directory, output log, and one
  layout-compatible target survive.
- Parallelism is represented by an opaque compatibility digest over the full
  TP/PP/DP/EP/CP layout, but the first claim is same-layout DP-replica recovery
  with tested PP configurations.  TP is all-ranks-or-nothing; EP/CP are
  fail-closed validation cases, not supported recovery claims.  Online
  resharding and stage splicing are out of scope.
- Algorithm 1 runs in the application layer.  At each LLM-request or tool-call
  response it uses GBDT to predict active-agent durations and CPM to recompute
  branch Slack; the next request carries Slack as ordinary metadata.
- `SlackLandlord` runs in the Aegis Controller.  It separates a candidate's
  Slack-visible admission benefit from a resident's deletion harm; one
  bundle-price margin controls DRAM admission, peer-HBM promotion, and victim
  exchange.  It does not invoke MCF.  The same current Slack orders recovery.
- `GuardedAIMD` adjusts only residual peer-HBM and background-transfer budgets
  with independent loops.  Workers enforce the last budgets with primary and
  swap reserves plus token buckets; AIMD is not the capacity-safety proof.
- Recovery reads only sealed, group-committed, then atomically recovery-pinned
  peer prefixes or read-leased Store prefixes.  It truncates KV at the durable
  continuation boundary and teacher-forces the logged gap exactly once.  A
  peer shadow is not a normal cache hit before atomic adoption; output resumes
  under a route and next-sequence fence.
- Analytical results, protocol validation, and empirical benefit are separate
  evidence classes.  Slack and action-value claims use analysis; protocol
  conformance uses property/fault tests; opportunity, cost, and end-to-end
  benefit require measurements.
- Controlled simulator results are labelled as Motivation or preliminary
  evidence and are not presented as measurements of the complete protocol,
  physical data plane, or hard-kill implementation.
