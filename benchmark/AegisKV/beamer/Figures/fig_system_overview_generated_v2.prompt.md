# fig_system_overview_generated_v2

Built-in `image_gen` prompt used for the accepted Beamer-local copy of
`fig_system_overview_generated_v2_tight.png`.

Canonical prompt record:

- `benchmark/AegisKV/draft/figures/src/fig_system_overview_generated_v2.prompt.md`

```text
Use case: infographic-diagram
Asset type: academic systems-paper system overview figure for a Beamer slide and paper draft
Primary request: Create a clean, publication-ready system overview diagram for AegisKV, a workflow-aware KV backup system for agentic LLM serving.
Scene/backdrop: pure white background, no texture, no gradient, no decorative objects.
Subject: A left-to-right systems pipeline. The flow begins with Workflow, then SGLang Serving, then Radix Metadata, then AegisKV Controller. From AegisKV Controller, split into two backup placement paths: GPUReplica above and HostBackup below. Both paths converge into Safe Frontier. Safe Frontier points to Failure Retry. Inside Failure Retry, show four ordered recovery choices: GPUReplica, HostBackup, Cache, Recompute.
Style/medium: crisp flat vector-like raster infographic, NSDI/OSDI/SOSP systems paper style, thin lines, consistent rounded rectangles, subtle muted colors, professional, no decorative icons.
Composition/framing: landscape 16:9, centered diagram, generous margins, aligned grid, all labels large and legible at slide size, readable at slide size.
Color palette: muted teal for workflow/frontier, muted blue for serving metadata, violet for AegisKV Controller, orange for HostBackup, red for GPUReplica, dark gray arrows, white background.
Text (verbatim): Use only these labels, spelled exactly: "Workflow", "SGLang Serving", "Radix Metadata", "AegisKV Controller", "GPUReplica", "HostBackup", "Safe Frontier", "Failure Retry", "Cache", "Recompute".
Constraints: no logos, no watermark, no 3D perspective, no tiny labels, no extra text, no misspellings, no overlapping arrows, no decorative illustration, no cloud vendor branding.
```
