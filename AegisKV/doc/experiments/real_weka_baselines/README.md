# Real-Weka capacity-matched baseline replay

> **Invalidated upstream calibration:** the archived `four_path_v2` baseline
> comparison inherits a Peer-HBM coefficient measured over Mooncake TCP, not
> CUDA P2P/D2D. It is diagnostic-only and excluded from the current paper.
> See `../transport-invalidation-rtx4090-20260823.json`.

This experiment replays the complete 393-workflow Weka trace through the
capacity-matched baseline controllers in
`python/sglsim/srt/simulation/aegis_baselines.py`. The audited current AegisKV
rows are paired by exact `(event_id, request_id)` with every additional policy.
The archived diagnostic anchor is
`../real_weka_full_design/results/full_393_workflows_four_path_v2`; no formal
baseline or figure entry reads it. A current anchor must instead be named
`full_393_workflows_v100_<physical-profile-sha12>` and carry the same config,
import-manifest, physical-profile, and source-manifest SHA chain as the suite.

Both the baseline runner and figure renderer now require a `READY` suite
whose embedded full-design config pins a `READY` physical calibration.
Consequently, neither the archived result bundle nor its figures can be
regenerated after the RTX4090/TCP invalidation. A replacement baseline suite
must reference a freshly imported V100S full-design config and a newly emitted
anchor result directory. Generate that immutable binding only after the full
replay passes:

```bash
uv run python AegisKV/experiments/real_weka_baselines/bind_v100_suite.py \
  --base-config /path/to/new-v100-sglsim-config.json \
  --base-import-manifest /path/to/new-v100-sglsim-import-manifest.json \
  --anchor-result /path/to/full_393_workflows_v100_<profile-sha12> \
  --output-config /path/to/baseline-suite-v100-<profile-sha12>.json
```

Run a small mechanism smoke first:

```bash
uv run python AegisKV/experiments/real_weka_baselines/run.py \
  --config /path/to/baseline-suite-v100-<profile-sha12>.json \
  --limit-workflows 2 --skip-dataset-sha \
  --output-dir /tmp/smoke_2_workflows_baselines_v100_<profile-sha12>
```

Run the complete dataset:

```bash
uv run python AegisKV/experiments/real_weka_baselines/run.py \
  --config /path/to/baseline-suite-v100-<profile-sha12>.json
```

The complete result is written to
`results/full_393_workflows_baselines_v100_<profile-sha12>` and refuses
overwrite. Render figures only from that bound result:

```bash
uv run python AegisKV/experiments/real_weka_baselines/make_figures.py \
  --suite-config /path/to/baseline-suite-v100-<profile-sha12>.json
```

The figure output is `figures_v100_<profile-sha12>` and also refuses
overwrite.

The fidelity and baseline-selection contract is frozen in
`AegisKV/related_works/BASELINE_SELECTION_2026-08-21.md`.  GhostServe is a
declared TP=8 control; CacheFlow and BSR are recovery-executor controls.  None
of those labels should be presented as a full paper-system reproduction.

The archived outcome, statistical interpretation, and figure index are in
[`RESULTS.md`](RESULTS.md). The invalidated four-path quantitative figure suite is
written to
`figures_four_path_v2/academic_figure_suite.pdf`; the superseded v1 figures
remain under `figures/`. The editable two-page system/baseline overview is
[`figures/aegiskv-complete-academic-overview.drawio`](figures/aegiskv-complete-academic-overview.drawio).

Every future V100 replay remains deliberately marked `paper_ready=false`: the
real Weka topology is combined with a preregistered token/interarrival
projection and metadata-only recovery execution. Matching TP1/PP1/system-DP4
physical rates do not make it a physical Mooncake end-to-end result.
