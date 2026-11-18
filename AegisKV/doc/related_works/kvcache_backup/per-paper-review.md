# 2025–2026 KV-cache backup and systems corpus: per-paper evidence review

Audit date: 2026-07-29.

## Scope and non-overclaim boundary

This directory contains exactly 30 genuine 2025–2026 papers from FAST, USENIX ATC, EuroSys, SOSP, ASPLOS, SIGMOD, MLSys, OSDI, and NSDI. The corpus is intentionally broader than literal KV-cache backup because a systems design also needs storage tiers, bandwidth paths, cache admission/eviction, compression, disaggregation, and scheduling evidence.

**Only GhostServe (paper 16) is classified as a direct fault-tolerant streaming-KV checkpoint/backup design.** HCache restores KV from saved intermediate activations, Mooncake replicates hot blocks for load/congestion, and the remaining papers optimize capacity, reuse, movement, or execution. Those mechanisms are useful substrates, but this review does not reinterpret them as fault-tolerance evidence.

Evidence pages below are **1-based physical PDF page numbers**, not the page number printed inside the proceedings paper. Every local PDF passed magic-byte, page-count, and unique SHA-256 checks. The complete hashes and source URLs are in `manifest.csv`.

### Relevance tiers

| Tier | Meaning | Count |
|---|---|---:|
| T1 | direct fault-tolerant KV checkpoint / backup | 1 |
| T2 | externalized-state, tiering, or restoration substrate | 15 |
| T3 | KV management, reuse, sharing, or compression | 9 |
| T4 | execution, scheduling, or memory-resource adjacent | 5 |

### Corpus audit

- Year split: 2025 = 15; 2026 = 15.
- Venue counts: ASPLOS = 2, EuroSys = 2, FAST = 5, MLSys = 11, NSDI = 2, OSDI = 5, SIGMOD = 1, SOSP = 1, USENIX ATC = 1.
- Source provenance: 24 official conference PDFs; 5 author/institution-hosted accepted papers; 1 public author preprint corresponding to the SIGMOD 2025 Cache-Craft paper.
- Content hashes: 30 unique SHA-256 values; no duplicate PDF payload.

## Cross-paper systems takeaways for AegisKV

The direct and adjacent evidence supports a resource-management formulation with four separable layers: (1) request/application criticality supplies a value or deadline signal; (2) a protection action chooses no copy, recomputation/restoration, replication, or erasure coding; (3) a placement action assigns bytes to GPU HBM, host DRAM, remote memory, or SSD and chooses an independent failure domain; and (4) a transfer/execution scheduler enforces per-window capacity, bandwidth, CPU, GPU, and I/O budgets. This is a systems problem only when the lower three layers remain explicit rather than being collapsed into a request score.

GhostServe supplies the direct parity/checkpoint action, but its own limitations leave space for AegisKV: topology-aware cross-node placement, dynamic failure handling, multiple protection actions, and budget adaptation across time windows. HCache, DroidSpeak, and Cache-Craft show that recomputation and transfer can be hybrid actions. Mooncake, Strata, SYMPHONY, Bidaw, and Prism show why memory capacity, storage bandwidth, locality, and cluster placement should be first-class resources. Compression papers show that byte reduction must carry measured quality and runtime costs.

## Per-paper reviews

### 01. Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot

- **Metadata/source:** FAST 2025; [landing page](https://www.usenix.org/conference/fast25/presentation/qin); [source PDF](https://www.usenix.org/system/files/fast25-qin.pdf); local file `01_fast25_mooncake.pdf` (17 pages, 871530 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: disaggregated KV storage; replication is load-oriented.
- **Storage and data path:** GPU HBM, CPU DRAM, SSD, and NIC-connected disaggregated KVCache pool. prefill/decode GPU ↔ RDMA-capable KVCache pool under a global Conductor.
- **Redundancy/recovery meaning:** Hot KV blocks may be replicated to relieve hotspots; the paper does not define a durability or failure-recovery contract.
- **Mechanism:** Disaggregates KV storage from compute, trades storage capacity for saved prefill computation, and globally schedules KV movement and reuse.
- **Reported evidence:** The production-trace evaluation reports serving 115% and 107% more requests on the evaluated A800 and H800 clusters.
- **Reusable for AegisKV:** Useful evidence for treating capacity, transfer bandwidth, cache locality, and replication placement as coupled cluster resources.
- **Do-not-overclaim / limitation:** Replication is introduced for congestion/hot-block relief, not fault tolerance; no RPO, recovery completion rule, or correlated-failure analysis.
- **Evidence pages:** PDF pp. 2, 5, 9.

### 02. IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference

- **Metadata/source:** FAST 2025; [landing page](https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress); [source PDF](https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf); local file `02_fast25_impress.pdf` (16 pages, 1418658 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: multi-tier prefix persistence.
- **Storage and data path:** GPU HBM, host DRAM, and SSD. prefix KV creation → importance selection → SSD/CPU tier manager → GPU loading.
- **Redundancy/recovery meaning:** A single importance-filtered cached representation migrates through SSD → CPU → GPU; no redundant failure copy.
- **Mechanism:** Stores only important prefix KV entries and coordinates admission, eviction, and movement across three memory/storage tiers.
- **Reported evidence:** Reports up to 2.8× lower time-to-first-token relative to evaluated baselines.
- **Reusable for AegisKV:** Useful for tier-aware capacity accounting and for making transfer cost and retained-state value explicit in a placement controller.
- **Do-not-overclaim / limitation:** Importance filtering can trade accuracy for space; cached state is not redundant and the paper does not provide failure semantics.
- **Evidence pages:** PDF pp. 2, 6, 10.

### 03. KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider

- **Metadata/source:** USENIX ATC 2025; [landing page](https://www.usenix.org/conference/atc25/presentation/wang-jiahao); [source PDF](https://www.usenix.org/system/files/atc25-wang-jiahao.pdf); local file `03_atc25_kvcache_cache_in_wild.pdf` (19 pages, 11072579 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: production characterization and eviction.
- **Storage and data path:** Production KV-cache hierarchy including GPU memory and larger remote/secondary tiers. production request trace → cache-behavior analysis → workload-aware eviction.
- **Redundancy/recovery meaning:** No backup path; analyzes reuse, occupancy, eviction, and workload-aware cache policy.
- **Mechanism:** Characterizes real KV-cache reuse and derives workload-aware cache management from cloud traces.
- **Reported evidence:** Shows that production reuse and cache pressure are highly workload-dependent and that workload-aware policy improves the evaluated cache behavior.
- **Reusable for AegisKV:** Supports measuring request-class heterogeneity before assigning protection and capacity budgets.
- **Do-not-overclaim / limitation:** Trace evidence is from one large provider and does not specify redundant copies, recovery, or availability guarantees.
- **Evidence pages:** PDF pp. 2, 8, 13.

### 04. Fast State Restoration in LLM Serving with HCache

- **Metadata/source:** EuroSys 2025; [landing page](https://doi.org/10.1145/3689031.3696072); [source PDF](https://chenyoumin1993.github.io/papers/eurosys25-hcache.pdf); local file `04_eurosys25_hcache.pdf` (16 pages, 658222 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: exact state restoration, not fault-tolerant checkpointing.
- **Storage and data path:** Intermediate hidden activations in host DRAM or SSD; regenerated KV returns to GPU HBM. GPU hidden-state checkpoint → host DRAM/SSD → overlapped load plus recomputation → GPU KV.
- **Redundancy/recovery meaning:** Stores compact intermediate activations and reconstructs exact KV with combined I/O and GPU computation.
- **Mechanism:** Chooses a compact intermediate activation as a restore point and combines storage reads with recomputation using chunk and bubble-free scheduling.
- **Reported evidence:** Reports up to 1.93× speedup over KV offload, 1.92–2.40× less storage, and up to 5.73× lower TTFT than full recomputation.
- **Reusable for AegisKV:** Provides a concrete recompute-versus-store recovery action that can sit beside replication and erasure coding in a resource allocator.
- **Do-not-overclaim / limitation:** “Restoration” is reuse/resume from saved activations; the paper does not establish failure detection, replica independence, RPO, or service recovery guarantees.
- **Evidence pages:** PDF pp. 1, 6, 8.

### 05. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion

- **Metadata/source:** EuroSys 2025; [landing page](https://doi.org/10.1145/3689031.3696098); [source PDF](https://www.microsoft.com/en-us/research/uploads/prod/2024/09/eurosys25-final999.pdf); local file `05_eurosys25_cacheblend.pdf` (16 pages, 1404016 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: persistent RAG chunk KV reuse.
- **Storage and data path:** Persistent chunk KV on disk or a cache server, loaded into CPU/GPU memory. RAG chunk cache → load and fuse → selective recomputation → GPU attention.
- **Redundancy/recovery meaning:** A cached chunk has one reusable representation plus selective recomputation; no protection copy.
- **Mechanism:** Corrects cross-chunk context mismatch by selectively recomputing KV and fusing cached knowledge chunks.
- **Reported evidence:** Reports 2.2–3.3× TTFT improvement and 2.8–5× throughput improvement over full recomputation in evaluated workloads.
- **Reusable for AegisKV:** Useful for valuing a stored KV object by avoided recomputation and for separating immutable reusable chunks from per-request state.
- **Do-not-overclaim / limitation:** Targets RAG reuse and quality correction, not redundancy or recovery from component failures.
- **Evidence pages:** PDF pp. 1, 5, 9.

### 06. DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction

- **Metadata/source:** SOSP 2025; [landing page](https://doi.org/10.1145/3731569.3764810); [source PDF](https://ipads.se.sjtu.edu.cn/zh/publications/sosp25-zhang.pdf); local file `06_sosp25_diffkv.pdf` (15 pages, 957116 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: in-GPU KV compaction.
- **Storage and data path:** GPU HBM. live GPU KV → differentiated parallel compaction → compact GPU KV.
- **Redundancy/recovery meaning:** No off-device copy; differentiated K/V, token, and head compaction reduces the live cache.
- **Mechanism:** Assigns different compression treatment to keys, values, tokens, and heads and compacts them in parallel.
- **Reported evidence:** Reports 2.7–5.7× KV compression and 1.9–5.4× serving-throughput improvement.
- **Reusable for AegisKV:** Compression ratio can be modeled as a resource transform that changes protection bytes and transfer time.
- **Do-not-overclaim / limitation:** Primarily memory-efficiency work; compaction is not a backup and may introduce model/quality sensitivity.
- **Evidence pages:** PDF pp. 1, 5, 9.

### 07. vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention

- **Metadata/source:** ASPLOS 2025; [landing page](https://doi.org/10.1145/3669940.3707256); [source PDF](https://apanwariisc.github.io/publications/asplos-2025-vattention/vattention-asplos25.pdf); local file `07_asplos25_vattention.pdf` (18 pages, 2204206 bytes).
- **Classification:** T4 — execution, scheduling, or memory-resource adjacent; adjacent: virtual-memory allocation.
- **Storage and data path:** GPU virtual address space backed on demand by GPU HBM. contiguous GPU virtual range → on-demand physical-HBM mapping → attention kernel.
- **Redundancy/recovery meaning:** No external state or redundant copy; CUDA virtual-memory mapping provides contiguous virtual KV.
- **Mechanism:** Uses CUDA virtual memory to preserve contiguous KV addressing while growing physical allocation dynamically.
- **Reported evidence:** Reports up to 1.23× end-to-end serving improvement in the evaluated configurations.
- **Reusable for AegisKV:** Relevant to allocator fragmentation and admission capacity, but not to the protection mechanism itself.
- **Do-not-overclaim / limitation:** Remains in the GPU failure domain and provides neither persistent state nor recovery semantics.
- **Evidence pages:** PDF pp. 1, 5, 9.

### 08. POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference

- **Metadata/source:** ASPLOS 2025; [landing page](https://doi.org/10.1145/3676641.3715996); [source PDF](https://apanwariisc.github.io/publications/asplos-2025-pod-attention/pod-attention-asplos25.pdf); local file `08_asplos25_pod_attention.pdf` (16 pages, 2076122 bytes).
- **Classification:** T4 — execution, scheduling, or memory-resource adjacent; adjacent: GPU execution overlap.
- **Storage and data path:** GPU HBM and shared GPU compute resources. prefill attention + decode attention → fused/overlapped GPU execution.
- **Redundancy/recovery meaning:** No backup path; co-schedules prefill and decode attention.
- **Mechanism:** A hybrid kernel and scheduler exploit complementary resource behavior of prefill and decode.
- **Reported evidence:** Reports up to 59% and a mean 28% attention-kernel speedup in the evaluated cases.
- **Reusable for AegisKV:** Provides a warning that backup traffic and encoding work must be charged against an already overlapped GPU pipeline.
- **Do-not-overclaim / limitation:** No KV persistence, redundancy, failure model, or recovery path.
- **Evidence pages:** PDF pp. 1, 5, 8.

### 09. Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation

- **Metadata/source:** SIGMOD 2025; [landing page](https://arxiv.org/abs/2502.15734); [source PDF](https://arxiv.org/pdf/2502.15734); local file `09_sigmod25_cache_craft.pdf` (15 pages, 5044083 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: reusable chunk-cache persistence.
- **Storage and data path:** GPU/CPU cache with optional NVMe-backed chunk-cache storage. retrieved text chunks → chunk-cache lookup/load → selective recomputation → attention.
- **Redundancy/recovery meaning:** Persists reusable chunk KV and selectively recomputes affected tokens; no redundant protection copy.
- **Mechanism:** Makes cached RAG chunks reusable at different positions/contexts by identifying and recomputing context-sensitive portions.
- **Reported evidence:** Reports 51% less redundant computation than prefix caching, 75% less than full computation, up to 1.6× throughput, and about 2× lower latency.
- **Reusable for AegisKV:** Useful for object-granular benefit estimation and for accounting for partial recomputation as a recovery alternative.
- **Do-not-overclaim / limitation:** Targets cache reuse and quality; it does not define a failure domain, redundancy factor, or recovery guarantee.
- **Evidence pages:** PDF pp. 1, 7, 9.

### 10. Marconi: Prefix Caching for the Era of Hybrid LLMs

- **Metadata/source:** MLSys 2025; [landing page](https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/7c180af017258d239bac6248d1eb26ac-Paper-Conference.pdf); local file `10_mlsys25_marconi.pdf` (16 pages, 2162926 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: prefix-cache admission and eviction.
- **Storage and data path:** GPU and host-memory prefix caches for hybrid recurrent/attention models. hybrid-model prefix state → admission/eviction policy → exact-match prefix reuse.
- **Redundancy/recovery meaning:** No backup; reuse-aware and compute-saving-aware cache admission/eviction.
- **Mechanism:** Adapts prefix caching to hybrid models whose recurrent state cannot be rolled back like ordinary attention KV.
- **Reported evidence:** Reports up to 34.4× higher token hit rate and up to 71.1% / 617 ms lower TTFT in evaluated workloads.
- **Reusable for AegisKV:** Shows that saved-compute value and reuse probability should be separate resource-allocation signals.
- **Do-not-overclaim / limitation:** Exact-match and hybrid-model semantics constrain reuse; no durable protection or recovery.
- **Evidence pages:** PDF pp. 1, 5, 8.

### 11. Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

- **Metadata/source:** MLSys 2025; [landing page](https://proceedings.mlsys.org/paper_files/paper/2025/hash/26289c647c6828e862e271ca3c490486-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/26289c647c6828e862e271ca3c490486-Paper-Conference.pdf); local file `11_mlsys25_rethinking_kv_compression.pdf` (22 pages, 1348980 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: empirical compression study.
- **Storage and data path:** Primarily compressed KV in GPU HBM. baseline GPU KV → compression method → serving performance and output-length evaluation.
- **Redundancy/recovery meaning:** No backup path; evaluates eviction/quantization-style compression effects.
- **Mechanism:** A production-oriented benchmark and analysis of why nominal cache reduction may not yield serving gains.
- **Reported evidence:** Finds that compressed caches can increase generated sequence length and that memory savings do not automatically improve production throughput.
- **Reusable for AegisKV:** Supports treating compression as a measured action with accuracy and runtime penalties, not a free reduction factor.
- **Do-not-overclaim / limitation:** A study rather than a backup design; evaluated algorithms and workloads may not cover future long-context models.
- **Evidence pages:** PDF pp. 1, 4, 8.

### 12. Optimizing LLM Queries in Relational Data Analytics Workloads

- **Metadata/source:** MLSys 2025; [landing page](https://proceedings.mlsys.org/paper_files/paper/2025/hash/b5dc49f44db2fadc5c4d717c57f4a424-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/b5dc49f44db2fadc5c4d717c57f4a424-Paper-Conference.pdf); local file `12_mlsys25_optimizing_llm_queries.pdf` (15 pages, 1206640 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: request transformation for KV reuse.
- **Storage and data path:** Existing serving-engine prefix KV cache. relational batch → row/field reordering → ordinary prefix-cache hits.
- **Redundancy/recovery meaning:** No state movement or backup; reorders rows and fields to increase prefix overlap.
- **Mechanism:** Transforms query order so an unmodified LLM service obtains more KV-cache reuse.
- **Reported evidence:** Reports up to 3.4× faster job completion and 32% cost savings in evaluated analytics workloads.
- **Reusable for AegisKV:** Evidence that workload shaping can change cache demand before the resource manager chooses protection.
- **Do-not-overclaim / limitation:** Application/workload optimization; does not store, replicate, or recover KV state.
- **Evidence pages:** PDF pp. 1, 3, 8.

### 13. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

- **Metadata/source:** MLSys 2025; [landing page](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf); local file `13_mlsys25_flashinfer.pdf` (17 pages, 1693865 bytes).
- **Classification:** T4 — execution, scheduling, or memory-resource adjacent; adjacent: attention execution and KV layout.
- **Storage and data path:** GPU HBM in block-sparse/composable KV layouts. paged/composable GPU KV → JIT-specialized attention → serving framework.
- **Redundancy/recovery meaning:** No backup; configurable kernels and load-balanced scheduling consume live KV.
- **Mechanism:** Combines flexible KV formats, JIT templates, and load-balanced attention scheduling.
- **Reported evidence:** Reports 29–69% lower inter-token latency, 28–30% lower long-context latency, and 13–17% lower parallel-generation latency in highlighted evaluations.
- **Reusable for AegisKV:** Relevant to the runtime cost of layout conversion and to preserving compatibility with the serving attention path.
- **Do-not-overclaim / limitation:** Kernel/runtime engine, not external state protection; no independent failure domain.
- **Evidence pages:** PDF pp. 1, 4, 8.

### 14. Context Parallelism for Scalable Million-Token Inference

- **Metadata/source:** MLSys 2025; [landing page](https://proceedings.mlsys.org/paper_files/paper/2025/hash/78834433edc3291f4c6cbbd2759324db-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/78834433edc3291f4c6cbbd2759324db-Paper-Conference.pdf); local file `14_mlsys25_context_parallelism.pdf` (16 pages, 795469 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: distributed KV execution.
- **Storage and data path:** Distributed GPU HBM across nodes connected by NVLink/RDMA/TCP. sharded context/KV → pass-KV or pass-Q ring → exact prefill/decode.
- **Redundancy/recovery meaning:** KV or queries circulate for exact ring attention; distribution is for computation, not redundant protection.
- **Mechanism:** Two exact context-parallel ring variants cover full prefill, persistent-KV prefill, and decode.
- **Reported evidence:** Reports 1M-token Llama-3 405B prefill in 77 s on 128 H100 GPUs with 93% parallel efficiency.
- **Reusable for AegisKV:** Provides communication-topology and sharding evidence for charging protection traffic in multi-node serving.
- **Do-not-overclaim / limitation:** Sharding does not imply a recoverable duplicate; failure handling and parity placement are outside scope.
- **Evidence pages:** PDF pp. 1, 5, 9.

### 15. NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference

- **Metadata/source:** MLSys 2025; [landing page](https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2025/file/66a026c0d17040889b50f0dfa650e5e0-Paper-Conference.pdf); local file `15_mlsys25_neo.pdf` (16 pages, 722779 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: CPU offload.
- **Storage and data path:** GPU HBM and local host CPU DRAM. GPU model execution ↔ local CPU attention/KV pipeline.
- **Redundancy/recovery meaning:** Moves part of KV state and attention work to CPU; no independent redundant copy.
- **Mechanism:** Asymmetric GPU–CPU pipelining and load-aware scheduling enlarge effective batch capacity.
- **Reported evidence:** Reports up to 7.5× throughput on T4 and gains of 26% on A10G and 14% on H100 in evaluated setups.
- **Reusable for AegisKV:** Useful for modeling host memory and CPU cycles as jointly constrained protection resources.
- **Do-not-overclaim / limitation:** CPU state remains in the same node/failure domain and is optimized for throughput rather than recovery.
- **Evidence pages:** PDF pp. 1, 4, 8.

### 16. GhostServe: A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving

- **Metadata/source:** MLSys 2026; [landing page](https://proceedings.mlsys.org/paper_files/paper/2026/hash/d99e8e80a6c41e148db686918dd7eab3-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/d99e8e80a6c41e148db686918dd7eab3-Paper-Conference.pdf); local file `16_mlsys26_ghostserve.pdf` (15 pages, 2734372 bytes).
- **Classification:** T1 — direct fault-tolerant KV checkpoint / backup; direct: fault-tolerant streaming-KV checkpoint.
- **Storage and data path:** Parity shards in host memory; data shards remain across tensor-parallel GPU HBM. streaming GPU KV shards → shadow parity encoding → host DRAM parity → lost-shard reconstruction → resumed inference.
- **Redundancy/recovery meaning:** Chunk-granular erasure-coded parity protects N KV data shards and reconstructs a lost shard after a device fault.
- **Mechanism:** Protects only parity in host memory and pipelines encoding/copying with inference; can hybridize reconstruction with recomputation.
- **Reported evidence:** Reports up to 2.7× lower checkpoint latency and 2.1× faster single-batch recovery than replication, plus 1.2× median response-time improvement under evaluated faults.
- **Reusable for AegisKV:** The corpus’s direct baseline for an explicit protection action, redundancy cost, recovery path, and checkpoint/recovery objective.
- **Do-not-overclaim / limitation:** At TP=2 the reported benefit can vanish; hybrid recovery is needed in some regimes; cross-node/topology-aware placement and dynamic removal of failed GPUs are future work, and the evaluation centers on data-level GPU-memory soft failures rather than full-stack hard failures.
- **Evidence pages:** PDF pp. 1, 2, 4, 8, 11, 12.

### 17. Strata: Hierarchical Context Caching for Long Context Language Model Serving

- **Metadata/source:** OSDI 2026; [landing page](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang); [source PDF](https://www.usenix.org/system/files/osdi26-xie-zhiqiang.pdf); local file `17_osdi26_strata.pdf` (17 pages, 1624366 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: hierarchical context caching.
- **Storage and data path:** GPU HBM, CPU DRAM, and SSD. SSD/host context cache → GPU-assisted batched I/O → GPU HBM → cache-aware scheduling.
- **Redundancy/recovery meaning:** Hierarchical cache loading and eviction; no redundant protection copy.
- **Mechanism:** Decouples host and GPU layouts for large transfers and schedules around load latency and delay hits.
- **Reported evidence:** Reports up to 5× improvement over vLLM with LMCache and 3.75× over TensorRT-LLM in evaluated long-context workloads.
- **Reusable for AegisKV:** Strong systems evidence for exposing bandwidth, I/O granularity, hit delays, and tier occupancy to a controller.
- **Do-not-overclaim / limitation:** A performance cache, not a durable checkpoint; no parity, consistency, or failure-recovery contract.
- **Evidence pages:** PDF pp. 2, 5, 9.

### 18. ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs

- **Metadata/source:** OSDI 2026; [landing page](https://www.usenix.org/conference/osdi26/presentation/liu-guangda); [source PDF](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf); local file `18_osdi26_echo.pdf` (22 pages, 3563903 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: host-memory offload and prefetch.
- **Storage and data path:** GPU HBM and host DRAM connected by PCIe. sparse-attention index prediction → HBM eviction/host DRAM → lossless prefetch → GPU attention.
- **Redundancy/recovery meaning:** Dynamically evicts and recalls KV; “lossless” refers to prefetch accuracy/serving quality, not fault tolerance.
- **Mechanism:** Graph-friendly cache management and predictable-index prefetch hide offload latency for native sparse attention.
- **Reported evidence:** Reports up to 2.1× serving-throughput improvement in the evaluated workloads.
- **Reusable for AegisKV:** Useful for bandwidth-budget and prefetch-deadline constraints in a time-window resource controller.
- **Do-not-overclaim / limitation:** No redundant copy or recovery guarantee; applicability depends on native sparse-attention behavior.
- **Evidence pages:** PDF pp. 2, 6, 11.

### 19. No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs

- **Metadata/source:** OSDI 2026; [landing page](https://www.usenix.org/conference/osdi26/presentation/luo); [source PDF](https://www.usenix.org/system/files/osdi26-luo.pdf); local file `19_osdi26_directkv.pdf` (17 pages, 1576982 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: zero-copy CPU-memory extension.
- **Storage and data path:** CPU memory accessed from GPU over NVLink-C2C on GH200/GB200. CPU-resident KV → direct NVLink-C2C GPU access → attention kernel.
- **Redundancy/recovery meaning:** CPU memory extends active KV capacity with no staging buffer; no independent redundant copy.
- **Mechanism:** DirectKV removes GPU staging and one transfer leg by making coherent CPU memory directly consumable by GPU kernels.
- **Reported evidence:** Reports 50% less transfer traffic, 43% less GPU-memory use, and up to 1.2× performance improvement.
- **Reusable for AegisKV:** Shows that the same placement action has platform-dependent bandwidth and buffer costs.
- **Do-not-overclaim / limitation:** CPU and GPU are in the same system failure domain; benefits rely on NVLink-C2C coherent platforms.
- **Evidence pages:** PDF pp. 2, 5, 10.

### 20. Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling

- **Metadata/source:** OSDI 2026; [landing page](https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan); [source PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf); local file `20_osdi26_lmetric.pdf` (20 pages, 6788045 bytes).
- **Classification:** T4 — execution, scheduling, or memory-resource adjacent; adjacent: KV-aware request routing.
- **Storage and data path:** Per-instance GPU/CPU KV caches. request and cache affinity → LMetric score → serving-instance routing.
- **Redundancy/recovery meaning:** No backup; routing score multiplies new-prefill-token cost by current batch size.
- **Mechanism:** A tuning-light multiplicative metric jointly captures KV locality and load balance.
- **Reported evidence:** Reports TTFT reductions up to 92% and 39%, and TPOT reductions up to 24% and 51%, across highlighted comparisons.
- **Reusable for AegisKV:** A useful precedent for a simple, implementable decision rule rather than an opaque high-order objective.
- **Do-not-overclaim / limitation:** Chooses where a request runs; it neither moves nor protects state and assumes cache information is available.
- **Evidence pages:** PDF pp. 2, 6, 12.

### 21. Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning

- **Metadata/source:** OSDI 2026; [landing page](https://www.usenix.org/conference/osdi26/presentation/yu-shan); [source PDF](https://www.usenix.org/system/files/osdi26-yu-shan.pdf); local file `21_osdi26_prism.pdf` (20 pages, 2621869 bytes).
- **Classification:** T4 — execution, scheduling, or memory-resource adjacent; adjacent: GPU memory resource management.
- **Storage and data path:** Shared GPU HBM with reclaim/eviction to host memory across colocated models. multi-model demand → kvcached balloon driver → HBM reclaim/host eviction → elastic model serving.
- **Redundancy/recovery meaning:** Memory ballooning reclaims inactive allocations; host placement is capacity management, not a backup contract.
- **Mechanism:** Unifies spatial and temporal sharing with elastic memory allocation driven by production demand.
- **Reported evidence:** Reports production deployment experience at a scale of more than 10,000 GPUs.
- **Reusable for AegisKV:** Strong precedent for framing KV protection as cluster memory resource allocation with elastic budgets.
- **Do-not-overclaim / limitation:** Model-memory co-serving rather than per-request KV checkpointing; host eviction does not establish independent failure protection.
- **Evidence pages:** PDF pp. 2, 6, 11.

### 22. DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants

- **Metadata/source:** NSDI 2026; [landing page](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan); [source PDF](https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf); local file `22_nsdi26_droidspeak.pdf` (21 pages, 1263568 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: cross-model KV sharing.
- **Storage and data path:** GPU memory across distributed nodes running related model variants. source-model KV → cross-node transfer + selective-layer recomputation → target-model KV.
- **Redundancy/recovery meaning:** Transfers reusable KV and recomputes selected layers; the remote copy is for reuse, not durability.
- **Mechanism:** Pipelines cache transfer and selective recomputation to reuse prefixes across fine-tuned variants with the same architecture.
- **Reported evidence:** Reports up to 4× throughput and 3.1× prefill-latency improvement.
- **Reusable for AegisKV:** Useful for hybrid transfer-plus-recompute recovery and for topology-aware remote placement.
- **Do-not-overclaim / limitation:** Requires compatible model architectures and does not specify failure independence or recovery guarantees.
- **Evidence pages:** PDF pp. 2, 6, 9.

### 23. SYMPHONY: Enabling Compute-Memory Disaggregation in LLM Serving Systems

- **Metadata/source:** NSDI 2026; [landing page](https://www.usenix.org/conference/nsdi26/presentation/agarwal); [source PDF](https://www.usenix.org/system/files/nsdi26-agarwal.pdf); local file `23_nsdi26_symphony.pdf` (16 pages, 1788114 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: disaggregated state management.
- **Storage and data path:** GPU/host memory plus local/remote disk or blob-backed disaggregated storage. application advisory hint → disaggregated KV prefetch/eviction → request-level compute placement.
- **Redundancy/recovery meaning:** Moves conversational KV to disaggregated memory with predictive advisory requests; no redundant checkpoint guarantee.
- **Mechanism:** Uses advisory requests, priority movement, and fallback mechanisms to hide state transfer and load-balance stateful serving.
- **Reported evidence:** Reports up to 2.4× lower latency and 4× request capacity in evaluated workloads.
- **Reusable for AegisKV:** Supports separating application criticality hints from a lower-level memory/bandwidth controller.
- **Do-not-overclaim / limitation:** Advisory predictions may be wrong and mitigations are performance-oriented; redundancy and consistency under failure are not specified.
- **Evidence pages:** PDF pp. 2, 6, 9.

### 24. CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving

- **Metadata/source:** FAST 2026; [landing page](https://www.usenix.org/conference/fast26/presentation/liu-yang); [source PDF](https://www.usenix.org/system/files/fast26-liu-yang.pdf); local file `24_fast26_cacheslide.pdf` (18 pages, 5544984 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: position-aware cache reuse with spill.
- **Storage and data path:** GPU HBM with host DRAM/NVMe spill. agent prompt segments → RPDC lookup → host/NVMe spill/load → minimal-token correction → GPU.
- **Redundancy/recovery meaning:** Reuses one spilled representation and computes a correction; no protection copy.
- **Mechanism:** Exploits relative-position-dependent reuse and coalesces storage operations for agent workflows.
- **Reported evidence:** Reports 3.11–4.3× latency improvement and 3.5–5.8× throughput improvement in evaluated settings.
- **Reusable for AegisKV:** Useful for object-granular storage-bandwidth accounting and agent-workflow reuse prediction.
- **Do-not-overclaim / limitation:** Quality correction and learned/approximate reuse are not fault protection; failure semantics are absent.
- **Evidence pages:** PDF pp. 2, 6, 10.

### 25. Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation–Storage Awareness

- **Metadata/source:** FAST 2026; [landing page](https://www.usenix.org/conference/fast26/presentation/hu-shipeng); [source PDF](https://www.usenix.org/system/files/fast26-hu-shipeng.pdf); local file `25_fast26_bidaw.pdf` (17 pages, 3392041 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: compute/storage-aware tiering.
- **Storage and data path:** Host DRAM and SSD two-tier history cache loaded into GPU. interactive history KV → host/SSD placement → load-latency-aware scheduling → GPU.
- **Redundancy/recovery meaning:** One cached history copy is tiered by predicted future use; no redundant protection.
- **Mechanism:** Bidirectional hints coordinate the compute scheduler with storage placement and answer-predicted eviction.
- **Reported evidence:** Reports up to 3.58× latency improvement and 1.83× throughput improvement.
- **Reusable for AegisKV:** Directly motivates joint capacity/bandwidth scheduling across time windows rather than a request-only score.
- **Do-not-overclaim / limitation:** Predictive tiering, not redundancy; errors in future-use prediction and device failures have no stated recovery contract.
- **Evidence pages:** PDF pp. 2, 6, 10.

### 26. SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs

- **Metadata/source:** FAST 2026; [landing page](https://www.usenix.org/conference/fast26/presentation/zheng); [source PDF](https://www.usenix.org/system/files/fast26-zheng.pdf); local file `26_fast26_solidattention.pdf` (17 pages, 2306053 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: SSD-backed sparse KV.
- **Storage and data path:** GPU VRAM, host DRAM, and local SSD. SSD coarse KV blocks → speculative prefetch/DRAM → GPU sparse attention.
- **Redundancy/recovery meaning:** Sparse KV blocks are stored/prefetched from SSD; no redundant checkpoint.
- **Mechanism:** Co-designs dynamic attention sparsity, coarse-grained SSD layout, prefetch, and compute/I/O overlap.
- **Reported evidence:** Reports up to 3.1× performance improvement and 98% KV-memory-footprint reduction in the evaluated AIPC setting.
- **Reusable for AegisKV:** Provides concrete SSD bandwidth/granularity constraints and an overlap-aware transfer model.
- **Do-not-overclaim / limitation:** Designed for a local memory-constrained PC and sparse attention; SSD placement is not a failure-independent backup.
- **Evidence pages:** PDF pp. 2, 6, 9.

### 27. SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models

- **Metadata/source:** MLSys 2026; [landing page](https://proceedings.mlsys.org/paper_files/paper/2026/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf); local file `27_mlsys26_skipkv.pdf` (19 pages, 1227765 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: selective KV generation/retention.
- **Storage and data path:** GPU HBM. reasoning tokens → sentence-aware selection/steering → reduced GPU KV.
- **Redundancy/recovery meaning:** Skips generation/storage and evicts at sentence granularity; no external copy.
- **Mechanism:** Training-free semantic selection avoids unstable token-wise eviction and excessive revalidation in reasoning models.
- **Reported evidence:** Reports 26.7% higher accuracy than the evaluated eviction baseline, 1.6× shorter outputs, and 1.7× throughput.
- **Reusable for AegisKV:** Shows that retained bytes can change future compute and output length, so compression actions need measured side effects.
- **Do-not-overclaim / limitation:** Changes live generation behavior and provides no retained failure copy.
- **Evidence pages:** PDF pp. 1, 4, 8.

### 28. Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost

- **Metadata/source:** MLSys 2026; [landing page](https://proceedings.mlsys.org/paper_files/paper/2026/hash/e4d8d1b5120be349d3fff8878650cf45-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/e4d8d1b5120be349d3fff8878650cf45-Paper-Conference.pdf); local file `28_mlsys26_kitty.pdf` (15 pages, 1469688 bytes).
- **Classification:** T3 — KV management, reuse, sharing, or compression; adjacent: KV quantization.
- **Storage and data path:** Mixed 2-bit/4-bit KV pages in GPU HBM. full-precision KV → dynamic channel ranking → hardware-aware mixed-bit pages → attention.
- **Redundancy/recovery meaning:** No backup; channel-sensitive mixed precision reduces the live cache.
- **Mechanism:** Boosts precision only for sensitive key channels and uses a layout/kernel co-design for efficient 2-bit caching.
- **Reported evidence:** Reports nearly 8× memory reduction, up to 8× batch size, and 2.1–4.1× throughput in highlighted evaluations.
- **Reusable for AegisKV:** Could reduce backup bytes if independently validated, but needs an accuracy-aware protection policy.
- **Do-not-overclaim / limitation:** Quantization is not redundancy and accuracy depends on model/task/channel selection.
- **Evidence pages:** PDF pp. 1, 5, 8.

### 29. OPKV: A High-Throughput Plugin-Driven Framework for Recallable Sparsity in Paged KV Cache Systems

- **Metadata/source:** MLSys 2026; [landing page](https://proceedings.mlsys.org/paper_files/paper/2026/hash/71381211d0abef73ed1887b83c4547b1-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/71381211d0abef73ed1887b83c4547b1-Paper-Conference.pdf); local file `29_mlsys26_opkv.pdf` (15 pages, 816393 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: recallable CPU offload.
- **Storage and data path:** GPU HBM and CPU memory. paged GPU KV → sparsity plugin/object reaggregation → CPU offload/recall → hot GPU pages.
- **Redundancy/recovery meaning:** Offloaded non-critical KV may be recalled, but it is capacity spill rather than an independent checkpoint.
- **Mechanism:** A plugin interface plus object reaggregation and hot-page hits integrates recallable sparsity with paged-cache engines.
- **Reported evidence:** Reports 1.3–1.8× serving-throughput improvement.
- **Reusable for AegisKV:** A useful implementation pattern for keeping policy modular while the runtime owns physical placement and movement.
- **Do-not-overclaim / limitation:** Recallability is not failure recovery; CPU state may share the node failure domain and quality depends on sparsity policy.
- **Evidence pages:** PDF pp. 1, 4, 8.

### 30. FlexiCache: Leveraging Temporal Stability of Attention Heads for Efficient KV Cache Management

- **Metadata/source:** MLSys 2026; [landing page](https://proceedings.mlsys.org/paper_files/paper/2026/hash/94bcb01789fccf15afe2764d8fe0f40e-Abstract-Conference.html); [source PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/94bcb01789fccf15afe2764d8fe0f40e-Paper-Conference.pdf); local file `30_mlsys26_flexicache.pdf` (16 pages, 734640 bytes).
- **Classification:** T2 — externalized-state, tiering, or restoration substrate; adjacent: head-aware hierarchical KV placement.
- **Storage and data path:** GPU HBM and host memory. head-stability classifier → keep unstable heads in HBM → spill/rerank stable-head pages in host memory.
- **Redundancy/recovery meaning:** Stable-head pages spill to host and can return; no protection replica.
- **Mechanism:** Uses temporal stability differences among attention heads to decide hierarchical page residency.
- **Reported evidence:** Reports about 70% GPU-memory reduction, 1.38–1.55× offline throughput, and 1.6–2.1× online throughput.
- **Reusable for AegisKV:** Supports fine-grained objects and measured hotness/stability as inputs to lower-level placement.
- **Do-not-overclaim / limitation:** No redundancy or failure semantics; effectiveness and quality depend on head classification stability.
- **Evidence pages:** PDF pp. 1, 4, 7.

## Bottom line

The literature search did not reveal thirty direct KV-backup papers in 2025–2026; claiming otherwise would be misleading. It produced one direct fault-tolerant checkpoint paper and twenty-nine carefully labeled adjacent systems papers. That boundary is itself important prior-art evidence: AegisKV should be positioned as a lower-level, adaptive resource manager that jointly allocates protection mode, failure-domain placement, memory capacity, and movement bandwidth over time—not merely as another application-level criticality formula or another host-offload cache.
