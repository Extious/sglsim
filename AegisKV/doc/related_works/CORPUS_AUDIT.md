# 2025--2026 AegisKV related-work corpus audit

Audit date: 2026-07-29.

The 30-paper KV corpus remains under `kvcache_backup/`.  The training-
checkpoint corpus was reviewed in the same audit, but is not part of the
current AegisKV design baseline; it has been moved to
`../Archived/2026-08-06-pre-kvcache-centric/research_iterations/training_checkpoint-prior-art/`.
The counts below are therefore a historical combined-audit record, not a claim
that all 60 PDFs remain in this directory.

## File integrity

| Corpus | PDFs | Parsed pages | Bytes | Unique SHA-256 |
| --- | ---: | ---: | ---: | ---: |
| KV-cache protection and adjacent systems | 30 | 515 | 71,089,974 | 30 |
| Training checkpoint and adjacent fault tolerance | 30 | 522 | 74,185,082 | 30 |
| Selected total | 60 | 1,037 | 145,275,056 | 60 |

All 60 selected files:

- start with `%PDF-`;
- have a parseable, non-zero page count;
- match the byte count and SHA-256 in their corpus manifest;
- have unique content hashes across the two selected corpora.

The first page of all 30 files in each corpus was rendered into a contact sheet
and visually inspected:

- `tmp/pdfs/corpus_contact_sheets/kvcache_backup_first_pages.png`;
- `tmp/pdfs/corpus_contact_sheets/training_checkpoint_first_pages.png`.

GhostServe physical pp. 3--7 and 11 were rendered separately. Representative
training method pages from PCcheck, FlowCheck, ByteCheckpoint, Checkmate,
MoEvement, and AdaCheck were rendered into
`tmp/pdfs/training_checkpoint_method_pages/representative_method_pages.png`.
These render checks complement, but do not replace, the full text/page-level
review recorded in each `per-paper-review.md`.

## Scope honesty

### KV corpus

- Direct 2025--2026 fault-tolerant streaming-KV checkpoint paper: **1/30**
  (GhostServe).
- Explicitly labeled adjacent storage, tiering, restoration, caching,
  compression, scheduling, or execution papers: **29/30**.
- The directory therefore satisfies the requested 30-paper research corpus,
  but it does **not** claim that 30 direct KV-backup papers exist.

See:

- `kvcache_backup/manifest.csv`;
- `kvcache_backup/per-paper-review.md`.

### Training corpus

- Main-conference papers: **28/30**.
- Peer-reviewed workshop: **1/30**.
- Work-in-progress paper: **1/30**.
- Papers defining a runtime snapshot/state-replication path: **11/30**.
- Format/post-processing/loading/empirical-I/O papers: **4/30**.
- Recovery, diagnosis, validation, or capacity-planning papers without a new
  write path: **15/30**.

Among the 11 runtime snapshot/state-replication papers, 9 explicitly use local,
peer, or dedicated-server Host DRAM as the first state layer. Among those 9:

- 5 explicitly continue from Host to a persistent layer;
- 1 mentions Host+SSD without a drain/commit protocol;
- 3 stop at Host memory in the evaluated prototype.

The supported conclusion is therefore:

> Host DRAM is the dominant fast first layer in this direct subset, but
> Host-to-disk/remote persistence is not a universal endpoint. Asynchronous
> snapshot creation, Host residence, persistent drain, and recoverable commit
> are separate events.

See:

- `../Archived/2026-08-06-pre-kvcache-centric/research_iterations/training_checkpoint-prior-art/manifest.csv`;
- `../Archived/2026-08-06-pre-kvcache-centric/research_iterations/training_checkpoint-prior-art/per-paper-review.md`;
- `../Archived/2026-08-06-pre-kvcache-centric/research_iterations/training_checkpoint-prior-art/download_validation.csv`.

## Evidence boundary

Paper-reported speedups remain claims of their respective systems and
configurations. The corpus supports mechanism selection and experiment design;
it is not empirical evidence that AegisKV already implements or achieves those
results.
