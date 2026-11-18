# Figure provenance and evidence boundary

- `system-overview.pdf`/`.png`: current three-layer System Overview exported from `../../figures/aegiskv-paper-system-design-v2.drawio` and cited by System Design.
- `system-design-v2.pdf`/`.png`: audit copy of the same current export; `system-design.pdf`/`.png` is a legacy export no longer cited.
- `protection-coverage-comparison.pdf` and `.png`: legacy policy-comparison export retained for audit only; the Motivation section no longer cites it.
- `commit-recovery-protocol.pdf` and `.png`: legacy protocol export retained for audit only; the v2 paper uses compatibility-checked group commit, failure-time `FreezeAndPin`, durable-boundary adoption, and continuation/output fencing instead.
- `system-architecture.pdf`: page 2 of the more detailed `../../figures/aegiskv-master-vllm-mooncake.drawio`; it includes formulas and the analytical/correctness/empirical evidence split, so it is useful as a design appendix but too dense for the main paper.
- `real-cc-parallel-subagent-duration.pdf` and `.png`: workflow-level CDF of the maximum slowest-minus-fastest duration gap across strictly overlapping Real CC fan-outs.
- `real-cc-failure-position-counterfactual.pdf` and `.png`: restart-from-branch-start counterfactual for the three branches in the selected `57f2a77e...` example, with failure fixed at 80% progress and observed fan-in slack subtracted; not an end-to-end replay result.
- `real-cc-workflow-57f2a77e.pdf` and `.drawio.png`: native Draw.io export of a selected three-way fan-out excerpt reconstructed from the Real CC trace; it is not the complete 977-request workflow.
- `motivation-recovery-scaling.pdf` and `.png`: one-column overlay of full-prefix recomputation for three AIConfigurator model/parallel configurations.  It uses the public SGLSim predictor interface in SILICON database mode and marks an illustrative operator-declared budget; it is neither hardware measurement nor full failure-to-ready time.
- `motivation-recovery-frontier-gap.pdf` and `.drawio.png`: neutral Motivation schematic from `../../figures/motivation-recovery-frontier-gap.drawio`; it contains no AegisKV mechanism or system comparison.
- `motivation-failure-position.png` and `motivation-workflow-dag.png`: legacy synthetic Motivation assets not cited by the paper and not valid Real CC evidence.
- `motivation-gpu-availability.png`: the simulated GPU-availability input trace.
- `preliminary-wct-kv-reuse.png`: preliminary simulated WCT CDF and retry-token reuse result; it must not be used as evidence for the complete optimizer without reproduction.

The main-text Draw.io drawings intentionally omit internal figure titles; the
LaTeX caption is the single source of the figure title.  All files in this directory must be regenerated after their Draw.io sources
change; the paper intentionally fails if a required main-text PDF is absent.
Captions separate direct trace statistics, trace counterfactuals, modeled path
isolation, and complete-system claims. A final EuroSys result must use paired,
traceable measurements with sample counts, confidence intervals, versioned
configs, and an explicit real-runtime/simulator/analytical evidence label.
