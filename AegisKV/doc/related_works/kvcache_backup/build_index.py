#!/usr/bin/env python3
"""Build an evidence-audited index for the local 2025--2026 KV-cache corpus."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent

TIER_NAMES = {
    "T1": "direct fault-tolerant KV checkpoint / backup",
    "T2": "externalized-state, tiering, or restoration substrate",
    "T3": "KV management, reuse, sharing, or compression",
    "T4": "execution, scheduling, or memory-resource adjacent",
}


PAPERS = [
    {
        "id": 1,
        "year": 2025,
        "venue": "FAST",
        "title": "Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot",
        "landing_url": "https://www.usenix.org/conference/fast25/presentation/qin",
        "pdf_url": "https://www.usenix.org/system/files/fast25-qin.pdf",
        "pdf_filename": "01_fast25_mooncake.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: disaggregated KV storage; replication is load-oriented",
        "storage_target": "GPU HBM, CPU DRAM, SSD, and NIC-connected disaggregated KVCache pool",
        "redundancy_or_recovery_path": "Hot KV blocks may be replicated to relieve hotspots; the paper does not define a durability or failure-recovery contract",
        "system_path": "prefill/decode GPU ↔ RDMA-capable KVCache pool under a global Conductor",
        "key_mechanism": "Disaggregates KV storage from compute, trades storage capacity for saved prefill computation, and globally schedules KV movement and reuse",
        "reported_result": "The production-trace evaluation reports serving 115% and 107% more requests on the evaluated A800 and H800 clusters",
        "aegiskv_reuse": "Useful evidence for treating capacity, transfer bandwidth, cache locality, and replication placement as coupled cluster resources",
        "evidence_pages": "2;5;9",
        "limitations": "Replication is introduced for congestion/hot-block relief, not fault tolerance; no RPO, recovery completion rule, or correlated-failure analysis",
    },
    {
        "id": 2,
        "year": 2025,
        "venue": "FAST",
        "title": "IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference",
        "landing_url": "https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress",
        "pdf_url": "https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf",
        "pdf_filename": "02_fast25_impress.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: multi-tier prefix persistence",
        "storage_target": "GPU HBM, host DRAM, and SSD",
        "redundancy_or_recovery_path": "A single importance-filtered cached representation migrates through SSD → CPU → GPU; no redundant failure copy",
        "system_path": "prefix KV creation → importance selection → SSD/CPU tier manager → GPU loading",
        "key_mechanism": "Stores only important prefix KV entries and coordinates admission, eviction, and movement across three memory/storage tiers",
        "reported_result": "Reports up to 2.8× lower time-to-first-token relative to evaluated baselines",
        "aegiskv_reuse": "Useful for tier-aware capacity accounting and for making transfer cost and retained-state value explicit in a placement controller",
        "evidence_pages": "2;6;10",
        "limitations": "Importance filtering can trade accuracy for space; cached state is not redundant and the paper does not provide failure semantics",
    },
    {
        "id": 3,
        "year": 2025,
        "venue": "USENIX ATC",
        "title": "KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider",
        "landing_url": "https://www.usenix.org/conference/atc25/presentation/wang-jiahao",
        "pdf_url": "https://www.usenix.org/system/files/atc25-wang-jiahao.pdf",
        "pdf_filename": "03_atc25_kvcache_cache_in_wild.pdf",
        "source_kind": "official conference PDF",
        "tier": "T3",
        "directness": "adjacent: production characterization and eviction",
        "storage_target": "Production KV-cache hierarchy including GPU memory and larger remote/secondary tiers",
        "redundancy_or_recovery_path": "No backup path; analyzes reuse, occupancy, eviction, and workload-aware cache policy",
        "system_path": "production request trace → cache-behavior analysis → workload-aware eviction",
        "key_mechanism": "Characterizes real KV-cache reuse and derives workload-aware cache management from cloud traces",
        "reported_result": "Shows that production reuse and cache pressure are highly workload-dependent and that workload-aware policy improves the evaluated cache behavior",
        "aegiskv_reuse": "Supports measuring request-class heterogeneity before assigning protection and capacity budgets",
        "evidence_pages": "2;8;13",
        "limitations": "Trace evidence is from one large provider and does not specify redundant copies, recovery, or availability guarantees",
    },
    {
        "id": 4,
        "year": 2025,
        "venue": "EuroSys",
        "title": "Fast State Restoration in LLM Serving with HCache",
        "landing_url": "https://doi.org/10.1145/3689031.3696072",
        "pdf_url": "https://chenyoumin1993.github.io/papers/eurosys25-hcache.pdf",
        "pdf_filename": "04_eurosys25_hcache.pdf",
        "source_kind": "author-hosted accepted paper",
        "tier": "T2",
        "directness": "adjacent: exact state restoration, not fault-tolerant checkpointing",
        "storage_target": "Intermediate hidden activations in host DRAM or SSD; regenerated KV returns to GPU HBM",
        "redundancy_or_recovery_path": "Stores compact intermediate activations and reconstructs exact KV with combined I/O and GPU computation",
        "system_path": "GPU hidden-state checkpoint → host DRAM/SSD → overlapped load plus recomputation → GPU KV",
        "key_mechanism": "Chooses a compact intermediate activation as a restore point and combines storage reads with recomputation using chunk and bubble-free scheduling",
        "reported_result": "Reports up to 1.93× speedup over KV offload, 1.92–2.40× less storage, and up to 5.73× lower TTFT than full recomputation",
        "aegiskv_reuse": "Provides a concrete recompute-versus-store recovery action that can sit beside replication and erasure coding in a resource allocator",
        "evidence_pages": "1;6;8",
        "limitations": "“Restoration” is reuse/resume from saved activations; the paper does not establish failure detection, replica independence, RPO, or service recovery guarantees",
    },
    {
        "id": 5,
        "year": 2025,
        "venue": "EuroSys",
        "title": "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion",
        "landing_url": "https://doi.org/10.1145/3689031.3696098",
        "pdf_url": "https://www.microsoft.com/en-us/research/uploads/prod/2024/09/eurosys25-final999.pdf",
        "pdf_filename": "05_eurosys25_cacheblend.pdf",
        "source_kind": "author/institution-hosted accepted paper",
        "tier": "T2",
        "directness": "adjacent: persistent RAG chunk KV reuse",
        "storage_target": "Persistent chunk KV on disk or a cache server, loaded into CPU/GPU memory",
        "redundancy_or_recovery_path": "A cached chunk has one reusable representation plus selective recomputation; no protection copy",
        "system_path": "RAG chunk cache → load and fuse → selective recomputation → GPU attention",
        "key_mechanism": "Corrects cross-chunk context mismatch by selectively recomputing KV and fusing cached knowledge chunks",
        "reported_result": "Reports 2.2–3.3× TTFT improvement and 2.8–5× throughput improvement over full recomputation in evaluated workloads",
        "aegiskv_reuse": "Useful for valuing a stored KV object by avoided recomputation and for separating immutable reusable chunks from per-request state",
        "evidence_pages": "1;5;9",
        "limitations": "Targets RAG reuse and quality correction, not redundancy or recovery from component failures",
    },
    {
        "id": 6,
        "year": 2025,
        "venue": "SOSP",
        "title": "DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction",
        "landing_url": "https://doi.org/10.1145/3731569.3764810",
        "pdf_url": "https://ipads.se.sjtu.edu.cn/zh/publications/sosp25-zhang.pdf",
        "pdf_filename": "06_sosp25_diffkv.pdf",
        "source_kind": "author/institution-hosted accepted paper",
        "tier": "T3",
        "directness": "adjacent: in-GPU KV compaction",
        "storage_target": "GPU HBM",
        "redundancy_or_recovery_path": "No off-device copy; differentiated K/V, token, and head compaction reduces the live cache",
        "system_path": "live GPU KV → differentiated parallel compaction → compact GPU KV",
        "key_mechanism": "Assigns different compression treatment to keys, values, tokens, and heads and compacts them in parallel",
        "reported_result": "Reports 2.7–5.7× KV compression and 1.9–5.4× serving-throughput improvement",
        "aegiskv_reuse": "Compression ratio can be modeled as a resource transform that changes protection bytes and transfer time",
        "evidence_pages": "1;5;9",
        "limitations": "Primarily memory-efficiency work; compaction is not a backup and may introduce model/quality sensitivity",
    },
    {
        "id": 7,
        "year": 2025,
        "venue": "ASPLOS",
        "title": "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention",
        "landing_url": "https://doi.org/10.1145/3669940.3707256",
        "pdf_url": "https://apanwariisc.github.io/publications/asplos-2025-vattention/vattention-asplos25.pdf",
        "pdf_filename": "07_asplos25_vattention.pdf",
        "source_kind": "author-hosted accepted paper",
        "tier": "T4",
        "directness": "adjacent: virtual-memory allocation",
        "storage_target": "GPU virtual address space backed on demand by GPU HBM",
        "redundancy_or_recovery_path": "No external state or redundant copy; CUDA virtual-memory mapping provides contiguous virtual KV",
        "system_path": "contiguous GPU virtual range → on-demand physical-HBM mapping → attention kernel",
        "key_mechanism": "Uses CUDA virtual memory to preserve contiguous KV addressing while growing physical allocation dynamically",
        "reported_result": "Reports up to 1.23× end-to-end serving improvement in the evaluated configurations",
        "aegiskv_reuse": "Relevant to allocator fragmentation and admission capacity, but not to the protection mechanism itself",
        "evidence_pages": "1;5;9",
        "limitations": "Remains in the GPU failure domain and provides neither persistent state nor recovery semantics",
    },
    {
        "id": 8,
        "year": 2025,
        "venue": "ASPLOS",
        "title": "POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference",
        "landing_url": "https://doi.org/10.1145/3676641.3715996",
        "pdf_url": "https://apanwariisc.github.io/publications/asplos-2025-pod-attention/pod-attention-asplos25.pdf",
        "pdf_filename": "08_asplos25_pod_attention.pdf",
        "source_kind": "author-hosted accepted paper",
        "tier": "T4",
        "directness": "adjacent: GPU execution overlap",
        "storage_target": "GPU HBM and shared GPU compute resources",
        "redundancy_or_recovery_path": "No backup path; co-schedules prefill and decode attention",
        "system_path": "prefill attention + decode attention → fused/overlapped GPU execution",
        "key_mechanism": "A hybrid kernel and scheduler exploit complementary resource behavior of prefill and decode",
        "reported_result": "Reports up to 59% and a mean 28% attention-kernel speedup in the evaluated cases",
        "aegiskv_reuse": "Provides a warning that backup traffic and encoding work must be charged against an already overlapped GPU pipeline",
        "evidence_pages": "1;5;8",
        "limitations": "No KV persistence, redundancy, failure model, or recovery path",
    },
    {
        "id": 9,
        "year": 2025,
        "venue": "SIGMOD",
        "title": "Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation",
        "landing_url": "https://arxiv.org/abs/2502.15734",
        "pdf_url": "https://arxiv.org/pdf/2502.15734",
        "pdf_filename": "09_sigmod25_cache_craft.pdf",
        "source_kind": "public author preprint for SIGMOD 2025 paper",
        "tier": "T2",
        "directness": "adjacent: reusable chunk-cache persistence",
        "storage_target": "GPU/CPU cache with optional NVMe-backed chunk-cache storage",
        "redundancy_or_recovery_path": "Persists reusable chunk KV and selectively recomputes affected tokens; no redundant protection copy",
        "system_path": "retrieved text chunks → chunk-cache lookup/load → selective recomputation → attention",
        "key_mechanism": "Makes cached RAG chunks reusable at different positions/contexts by identifying and recomputing context-sensitive portions",
        "reported_result": "Reports 51% less redundant computation than prefix caching, 75% less than full computation, up to 1.6× throughput, and about 2× lower latency",
        "aegiskv_reuse": "Useful for object-granular benefit estimation and for accounting for partial recomputation as a recovery alternative",
        "evidence_pages": "1;7;9",
        "limitations": "Targets cache reuse and quality; it does not define a failure domain, redundancy factor, or recovery guarantee",
    },
    {
        "id": 10,
        "year": 2025,
        "venue": "MLSys",
        "title": "Marconi: Prefix Caching for the Era of Hybrid LLMs",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2025/file/7c180af017258d239bac6248d1eb26ac-Paper-Conference.pdf",
        "pdf_filename": "10_mlsys25_marconi.pdf",
        "source_kind": "official conference PDF",
        "tier": "T3",
        "directness": "adjacent: prefix-cache admission and eviction",
        "storage_target": "GPU and host-memory prefix caches for hybrid recurrent/attention models",
        "redundancy_or_recovery_path": "No backup; reuse-aware and compute-saving-aware cache admission/eviction",
        "system_path": "hybrid-model prefix state → admission/eviction policy → exact-match prefix reuse",
        "key_mechanism": "Adapts prefix caching to hybrid models whose recurrent state cannot be rolled back like ordinary attention KV",
        "reported_result": "Reports up to 34.4× higher token hit rate and up to 71.1% / 617 ms lower TTFT in evaluated workloads",
        "aegiskv_reuse": "Shows that saved-compute value and reuse probability should be separate resource-allocation signals",
        "evidence_pages": "1;5;8",
        "limitations": "Exact-match and hybrid-model semantics constrain reuse; no durable protection or recovery",
    },
    {
        "id": 11,
        "year": 2025,
        "venue": "MLSys",
        "title": "Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2025/hash/26289c647c6828e862e271ca3c490486-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2025/file/26289c647c6828e862e271ca3c490486-Paper-Conference.pdf",
        "pdf_filename": "11_mlsys25_rethinking_kv_compression.pdf",
        "source_kind": "official conference PDF",
        "tier": "T3",
        "directness": "adjacent: empirical compression study",
        "storage_target": "Primarily compressed KV in GPU HBM",
        "redundancy_or_recovery_path": "No backup path; evaluates eviction/quantization-style compression effects",
        "system_path": "baseline GPU KV → compression method → serving performance and output-length evaluation",
        "key_mechanism": "A production-oriented benchmark and analysis of why nominal cache reduction may not yield serving gains",
        "reported_result": "Finds that compressed caches can increase generated sequence length and that memory savings do not automatically improve production throughput",
        "aegiskv_reuse": "Supports treating compression as a measured action with accuracy and runtime penalties, not a free reduction factor",
        "evidence_pages": "1;4;8",
        "limitations": "A study rather than a backup design; evaluated algorithms and workloads may not cover future long-context models",
    },
    {
        "id": 12,
        "year": 2025,
        "venue": "MLSys",
        "title": "Optimizing LLM Queries in Relational Data Analytics Workloads",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2025/hash/b5dc49f44db2fadc5c4d717c57f4a424-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2025/file/b5dc49f44db2fadc5c4d717c57f4a424-Paper-Conference.pdf",
        "pdf_filename": "12_mlsys25_optimizing_llm_queries.pdf",
        "source_kind": "official conference PDF",
        "tier": "T3",
        "directness": "adjacent: request transformation for KV reuse",
        "storage_target": "Existing serving-engine prefix KV cache",
        "redundancy_or_recovery_path": "No state movement or backup; reorders rows and fields to increase prefix overlap",
        "system_path": "relational batch → row/field reordering → ordinary prefix-cache hits",
        "key_mechanism": "Transforms query order so an unmodified LLM service obtains more KV-cache reuse",
        "reported_result": "Reports up to 3.4× faster job completion and 32% cost savings in evaluated analytics workloads",
        "aegiskv_reuse": "Evidence that workload shaping can change cache demand before the resource manager chooses protection",
        "evidence_pages": "1;3;8",
        "limitations": "Application/workload optimization; does not store, replicate, or recover KV state",
    },
    {
        "id": 13,
        "year": 2025,
        "venue": "MLSys",
        "title": "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf",
        "pdf_filename": "13_mlsys25_flashinfer.pdf",
        "source_kind": "official conference PDF",
        "tier": "T4",
        "directness": "adjacent: attention execution and KV layout",
        "storage_target": "GPU HBM in block-sparse/composable KV layouts",
        "redundancy_or_recovery_path": "No backup; configurable kernels and load-balanced scheduling consume live KV",
        "system_path": "paged/composable GPU KV → JIT-specialized attention → serving framework",
        "key_mechanism": "Combines flexible KV formats, JIT templates, and load-balanced attention scheduling",
        "reported_result": "Reports 29–69% lower inter-token latency, 28–30% lower long-context latency, and 13–17% lower parallel-generation latency in highlighted evaluations",
        "aegiskv_reuse": "Relevant to the runtime cost of layout conversion and to preserving compatibility with the serving attention path",
        "evidence_pages": "1;4;8",
        "limitations": "Kernel/runtime engine, not external state protection; no independent failure domain",
    },
    {
        "id": 14,
        "year": 2025,
        "venue": "MLSys",
        "title": "Context Parallelism for Scalable Million-Token Inference",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2025/hash/78834433edc3291f4c6cbbd2759324db-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2025/file/78834433edc3291f4c6cbbd2759324db-Paper-Conference.pdf",
        "pdf_filename": "14_mlsys25_context_parallelism.pdf",
        "source_kind": "official conference PDF",
        "tier": "T3",
        "directness": "adjacent: distributed KV execution",
        "storage_target": "Distributed GPU HBM across nodes connected by NVLink/RDMA/TCP",
        "redundancy_or_recovery_path": "KV or queries circulate for exact ring attention; distribution is for computation, not redundant protection",
        "system_path": "sharded context/KV → pass-KV or pass-Q ring → exact prefill/decode",
        "key_mechanism": "Two exact context-parallel ring variants cover full prefill, persistent-KV prefill, and decode",
        "reported_result": "Reports 1M-token Llama-3 405B prefill in 77 s on 128 H100 GPUs with 93% parallel efficiency",
        "aegiskv_reuse": "Provides communication-topology and sharding evidence for charging protection traffic in multi-node serving",
        "evidence_pages": "1;5;9",
        "limitations": "Sharding does not imply a recoverable duplicate; failure handling and parity placement are outside scope",
    },
    {
        "id": 15,
        "year": 2025,
        "venue": "MLSys",
        "title": "NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2025/file/66a026c0d17040889b50f0dfa650e5e0-Paper-Conference.pdf",
        "pdf_filename": "15_mlsys25_neo.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: CPU offload",
        "storage_target": "GPU HBM and local host CPU DRAM",
        "redundancy_or_recovery_path": "Moves part of KV state and attention work to CPU; no independent redundant copy",
        "system_path": "GPU model execution ↔ local CPU attention/KV pipeline",
        "key_mechanism": "Asymmetric GPU–CPU pipelining and load-aware scheduling enlarge effective batch capacity",
        "reported_result": "Reports up to 7.5× throughput on T4 and gains of 26% on A10G and 14% on H100 in evaluated setups",
        "aegiskv_reuse": "Useful for modeling host memory and CPU cycles as jointly constrained protection resources",
        "evidence_pages": "1;4;8",
        "limitations": "CPU state remains in the same node/failure domain and is optimized for throughput rather than recovery",
    },
    {
        "id": 16,
        "year": 2026,
        "venue": "MLSys",
        "title": "GhostServe: A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2026/hash/d99e8e80a6c41e148db686918dd7eab3-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2026/file/d99e8e80a6c41e148db686918dd7eab3-Paper-Conference.pdf",
        "pdf_filename": "16_mlsys26_ghostserve.pdf",
        "source_kind": "official conference PDF",
        "tier": "T1",
        "directness": "direct: fault-tolerant streaming-KV checkpoint",
        "storage_target": "Parity shards in host memory; data shards remain across tensor-parallel GPU HBM",
        "redundancy_or_recovery_path": "Chunk-granular erasure-coded parity protects N KV data shards and reconstructs a lost shard after a device fault",
        "system_path": "streaming GPU KV shards → shadow parity encoding → host DRAM parity → lost-shard reconstruction → resumed inference",
        "key_mechanism": "Protects only parity in host memory and pipelines encoding/copying with inference; can hybridize reconstruction with recomputation",
        "reported_result": "Reports up to 2.7× lower checkpoint latency and 2.1× faster single-batch recovery than replication, plus 1.2× median response-time improvement under evaluated faults",
        "aegiskv_reuse": "The corpus’s direct baseline for an explicit protection action, redundancy cost, recovery path, and checkpoint/recovery objective",
        "evidence_pages": "1;2;4;8;11;12",
        "limitations": "At TP=2 the reported benefit can vanish; hybrid recovery is needed in some regimes; cross-node/topology-aware placement and dynamic removal of failed GPUs are future work, and the evaluation centers on data-level GPU-memory soft failures rather than full-stack hard failures",
    },
    {
        "id": 17,
        "year": 2026,
        "venue": "OSDI",
        "title": "Strata: Hierarchical Context Caching for Long Context Language Model Serving",
        "landing_url": "https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang",
        "pdf_url": "https://www.usenix.org/system/files/osdi26-xie-zhiqiang.pdf",
        "pdf_filename": "17_osdi26_strata.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: hierarchical context caching",
        "storage_target": "GPU HBM, CPU DRAM, and SSD",
        "redundancy_or_recovery_path": "Hierarchical cache loading and eviction; no redundant protection copy",
        "system_path": "SSD/host context cache → GPU-assisted batched I/O → GPU HBM → cache-aware scheduling",
        "key_mechanism": "Decouples host and GPU layouts for large transfers and schedules around load latency and delay hits",
        "reported_result": "Reports up to 5× improvement over vLLM with LMCache and 3.75× over TensorRT-LLM in evaluated long-context workloads",
        "aegiskv_reuse": "Strong systems evidence for exposing bandwidth, I/O granularity, hit delays, and tier occupancy to a controller",
        "evidence_pages": "2;5;9",
        "limitations": "A performance cache, not a durable checkpoint; no parity, consistency, or failure-recovery contract",
    },
    {
        "id": 18,
        "year": 2026,
        "venue": "OSDI",
        "title": "ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs",
        "landing_url": "https://www.usenix.org/conference/osdi26/presentation/liu-guangda",
        "pdf_url": "https://www.usenix.org/system/files/osdi26-liu-guangda.pdf",
        "pdf_filename": "18_osdi26_echo.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: host-memory offload and prefetch",
        "storage_target": "GPU HBM and host DRAM connected by PCIe",
        "redundancy_or_recovery_path": "Dynamically evicts and recalls KV; “lossless” refers to prefetch accuracy/serving quality, not fault tolerance",
        "system_path": "sparse-attention index prediction → HBM eviction/host DRAM → lossless prefetch → GPU attention",
        "key_mechanism": "Graph-friendly cache management and predictable-index prefetch hide offload latency for native sparse attention",
        "reported_result": "Reports up to 2.1× serving-throughput improvement in the evaluated workloads",
        "aegiskv_reuse": "Useful for bandwidth-budget and prefetch-deadline constraints in a time-window resource controller",
        "evidence_pages": "2;6;11",
        "limitations": "No redundant copy or recovery guarantee; applicability depends on native sparse-attention behavior",
    },
    {
        "id": 19,
        "year": 2026,
        "venue": "OSDI",
        "title": "No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs",
        "landing_url": "https://www.usenix.org/conference/osdi26/presentation/luo",
        "pdf_url": "https://www.usenix.org/system/files/osdi26-luo.pdf",
        "pdf_filename": "19_osdi26_directkv.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: zero-copy CPU-memory extension",
        "storage_target": "CPU memory accessed from GPU over NVLink-C2C on GH200/GB200",
        "redundancy_or_recovery_path": "CPU memory extends active KV capacity with no staging buffer; no independent redundant copy",
        "system_path": "CPU-resident KV → direct NVLink-C2C GPU access → attention kernel",
        "key_mechanism": "DirectKV removes GPU staging and one transfer leg by making coherent CPU memory directly consumable by GPU kernels",
        "reported_result": "Reports 50% less transfer traffic, 43% less GPU-memory use, and up to 1.2× performance improvement",
        "aegiskv_reuse": "Shows that the same placement action has platform-dependent bandwidth and buffer costs",
        "evidence_pages": "2;5;10",
        "limitations": "CPU and GPU are in the same system failure domain; benefits rely on NVLink-C2C coherent platforms",
    },
    {
        "id": 20,
        "year": 2026,
        "venue": "OSDI",
        "title": "Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling",
        "landing_url": "https://www.usenix.org/conference/osdi26/presentation/zhang-dingyan",
        "pdf_url": "https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf",
        "pdf_filename": "20_osdi26_lmetric.pdf",
        "source_kind": "official conference PDF",
        "tier": "T4",
        "directness": "adjacent: KV-aware request routing",
        "storage_target": "Per-instance GPU/CPU KV caches",
        "redundancy_or_recovery_path": "No backup; routing score multiplies new-prefill-token cost by current batch size",
        "system_path": "request and cache affinity → LMetric score → serving-instance routing",
        "key_mechanism": "A tuning-light multiplicative metric jointly captures KV locality and load balance",
        "reported_result": "Reports TTFT reductions up to 92% and 39%, and TPOT reductions up to 24% and 51%, across highlighted comparisons",
        "aegiskv_reuse": "A useful precedent for a simple, implementable decision rule rather than an opaque high-order objective",
        "evidence_pages": "2;6;12",
        "limitations": "Chooses where a request runs; it neither moves nor protects state and assumes cache information is available",
    },
    {
        "id": 21,
        "year": 2026,
        "venue": "OSDI",
        "title": "Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning",
        "landing_url": "https://www.usenix.org/conference/osdi26/presentation/yu-shan",
        "pdf_url": "https://www.usenix.org/system/files/osdi26-yu-shan.pdf",
        "pdf_filename": "21_osdi26_prism.pdf",
        "source_kind": "official conference PDF",
        "tier": "T4",
        "directness": "adjacent: GPU memory resource management",
        "storage_target": "Shared GPU HBM with reclaim/eviction to host memory across colocated models",
        "redundancy_or_recovery_path": "Memory ballooning reclaims inactive allocations; host placement is capacity management, not a backup contract",
        "system_path": "multi-model demand → kvcached balloon driver → HBM reclaim/host eviction → elastic model serving",
        "key_mechanism": "Unifies spatial and temporal sharing with elastic memory allocation driven by production demand",
        "reported_result": "Reports production deployment experience at a scale of more than 10,000 GPUs",
        "aegiskv_reuse": "Strong precedent for framing KV protection as cluster memory resource allocation with elastic budgets",
        "evidence_pages": "2;6;11",
        "limitations": "Model-memory co-serving rather than per-request KV checkpointing; host eviction does not establish independent failure protection",
    },
    {
        "id": 22,
        "year": 2026,
        "venue": "NSDI",
        "title": "DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants",
        "landing_url": "https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan",
        "pdf_url": "https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf",
        "pdf_filename": "22_nsdi26_droidspeak.pdf",
        "source_kind": "official conference PDF",
        "tier": "T3",
        "directness": "adjacent: cross-model KV sharing",
        "storage_target": "GPU memory across distributed nodes running related model variants",
        "redundancy_or_recovery_path": "Transfers reusable KV and recomputes selected layers; the remote copy is for reuse, not durability",
        "system_path": "source-model KV → cross-node transfer + selective-layer recomputation → target-model KV",
        "key_mechanism": "Pipelines cache transfer and selective recomputation to reuse prefixes across fine-tuned variants with the same architecture",
        "reported_result": "Reports up to 4× throughput and 3.1× prefill-latency improvement",
        "aegiskv_reuse": "Useful for hybrid transfer-plus-recompute recovery and for topology-aware remote placement",
        "evidence_pages": "2;6;9",
        "limitations": "Requires compatible model architectures and does not specify failure independence or recovery guarantees",
    },
    {
        "id": 23,
        "year": 2026,
        "venue": "NSDI",
        "title": "SYMPHONY: Enabling Compute-Memory Disaggregation in LLM Serving Systems",
        "landing_url": "https://www.usenix.org/conference/nsdi26/presentation/agarwal",
        "pdf_url": "https://www.usenix.org/system/files/nsdi26-agarwal.pdf",
        "pdf_filename": "23_nsdi26_symphony.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: disaggregated state management",
        "storage_target": "GPU/host memory plus local/remote disk or blob-backed disaggregated storage",
        "redundancy_or_recovery_path": "Moves conversational KV to disaggregated memory with predictive advisory requests; no redundant checkpoint guarantee",
        "system_path": "application advisory hint → disaggregated KV prefetch/eviction → request-level compute placement",
        "key_mechanism": "Uses advisory requests, priority movement, and fallback mechanisms to hide state transfer and load-balance stateful serving",
        "reported_result": "Reports up to 2.4× lower latency and 4× request capacity in evaluated workloads",
        "aegiskv_reuse": "Supports separating application criticality hints from a lower-level memory/bandwidth controller",
        "evidence_pages": "2;6;9",
        "limitations": "Advisory predictions may be wrong and mitigations are performance-oriented; redundancy and consistency under failure are not specified",
    },
    {
        "id": 24,
        "year": 2026,
        "venue": "FAST",
        "title": "CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving",
        "landing_url": "https://www.usenix.org/conference/fast26/presentation/liu-yang",
        "pdf_url": "https://www.usenix.org/system/files/fast26-liu-yang.pdf",
        "pdf_filename": "24_fast26_cacheslide.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: position-aware cache reuse with spill",
        "storage_target": "GPU HBM with host DRAM/NVMe spill",
        "redundancy_or_recovery_path": "Reuses one spilled representation and computes a correction; no protection copy",
        "system_path": "agent prompt segments → RPDC lookup → host/NVMe spill/load → minimal-token correction → GPU",
        "key_mechanism": "Exploits relative-position-dependent reuse and coalesces storage operations for agent workflows",
        "reported_result": "Reports 3.11–4.3× latency improvement and 3.5–5.8× throughput improvement in evaluated settings",
        "aegiskv_reuse": "Useful for object-granular storage-bandwidth accounting and agent-workflow reuse prediction",
        "evidence_pages": "2;6;10",
        "limitations": "Quality correction and learned/approximate reuse are not fault protection; failure semantics are absent",
    },
    {
        "id": 25,
        "year": 2026,
        "venue": "FAST",
        "title": "Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation–Storage Awareness",
        "landing_url": "https://www.usenix.org/conference/fast26/presentation/hu-shipeng",
        "pdf_url": "https://www.usenix.org/system/files/fast26-hu-shipeng.pdf",
        "pdf_filename": "25_fast26_bidaw.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: compute/storage-aware tiering",
        "storage_target": "Host DRAM and SSD two-tier history cache loaded into GPU",
        "redundancy_or_recovery_path": "One cached history copy is tiered by predicted future use; no redundant protection",
        "system_path": "interactive history KV → host/SSD placement → load-latency-aware scheduling → GPU",
        "key_mechanism": "Bidirectional hints coordinate the compute scheduler with storage placement and answer-predicted eviction",
        "reported_result": "Reports up to 3.58× latency improvement and 1.83× throughput improvement",
        "aegiskv_reuse": "Directly motivates joint capacity/bandwidth scheduling across time windows rather than a request-only score",
        "evidence_pages": "2;6;10",
        "limitations": "Predictive tiering, not redundancy; errors in future-use prediction and device failures have no stated recovery contract",
    },
    {
        "id": 26,
        "year": 2026,
        "venue": "FAST",
        "title": "SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs",
        "landing_url": "https://www.usenix.org/conference/fast26/presentation/zheng",
        "pdf_url": "https://www.usenix.org/system/files/fast26-zheng.pdf",
        "pdf_filename": "26_fast26_solidattention.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: SSD-backed sparse KV",
        "storage_target": "GPU VRAM, host DRAM, and local SSD",
        "redundancy_or_recovery_path": "Sparse KV blocks are stored/prefetched from SSD; no redundant checkpoint",
        "system_path": "SSD coarse KV blocks → speculative prefetch/DRAM → GPU sparse attention",
        "key_mechanism": "Co-designs dynamic attention sparsity, coarse-grained SSD layout, prefetch, and compute/I/O overlap",
        "reported_result": "Reports up to 3.1× performance improvement and 98% KV-memory-footprint reduction in the evaluated AIPC setting",
        "aegiskv_reuse": "Provides concrete SSD bandwidth/granularity constraints and an overlap-aware transfer model",
        "evidence_pages": "2;6;9",
        "limitations": "Designed for a local memory-constrained PC and sparse attention; SSD placement is not a failure-independent backup",
    },
    {
        "id": 27,
        "year": 2026,
        "venue": "MLSys",
        "title": "SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2026/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2026/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf",
        "pdf_filename": "27_mlsys26_skipkv.pdf",
        "source_kind": "official conference PDF",
        "tier": "T3",
        "directness": "adjacent: selective KV generation/retention",
        "storage_target": "GPU HBM",
        "redundancy_or_recovery_path": "Skips generation/storage and evicts at sentence granularity; no external copy",
        "system_path": "reasoning tokens → sentence-aware selection/steering → reduced GPU KV",
        "key_mechanism": "Training-free semantic selection avoids unstable token-wise eviction and excessive revalidation in reasoning models",
        "reported_result": "Reports 26.7% higher accuracy than the evaluated eviction baseline, 1.6× shorter outputs, and 1.7× throughput",
        "aegiskv_reuse": "Shows that retained bytes can change future compute and output length, so compression actions need measured side effects",
        "evidence_pages": "1;4;8",
        "limitations": "Changes live generation behavior and provides no retained failure copy",
    },
    {
        "id": 28,
        "year": 2026,
        "venue": "MLSys",
        "title": "Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2026/hash/e4d8d1b5120be349d3fff8878650cf45-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2026/file/e4d8d1b5120be349d3fff8878650cf45-Paper-Conference.pdf",
        "pdf_filename": "28_mlsys26_kitty.pdf",
        "source_kind": "official conference PDF",
        "tier": "T3",
        "directness": "adjacent: KV quantization",
        "storage_target": "Mixed 2-bit/4-bit KV pages in GPU HBM",
        "redundancy_or_recovery_path": "No backup; channel-sensitive mixed precision reduces the live cache",
        "system_path": "full-precision KV → dynamic channel ranking → hardware-aware mixed-bit pages → attention",
        "key_mechanism": "Boosts precision only for sensitive key channels and uses a layout/kernel co-design for efficient 2-bit caching",
        "reported_result": "Reports nearly 8× memory reduction, up to 8× batch size, and 2.1–4.1× throughput in highlighted evaluations",
        "aegiskv_reuse": "Could reduce backup bytes if independently validated, but needs an accuracy-aware protection policy",
        "evidence_pages": "1;5;8",
        "limitations": "Quantization is not redundancy and accuracy depends on model/task/channel selection",
    },
    {
        "id": 29,
        "year": 2026,
        "venue": "MLSys",
        "title": "OPKV: A High-Throughput Plugin-Driven Framework for Recallable Sparsity in Paged KV Cache Systems",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2026/hash/71381211d0abef73ed1887b83c4547b1-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2026/file/71381211d0abef73ed1887b83c4547b1-Paper-Conference.pdf",
        "pdf_filename": "29_mlsys26_opkv.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: recallable CPU offload",
        "storage_target": "GPU HBM and CPU memory",
        "redundancy_or_recovery_path": "Offloaded non-critical KV may be recalled, but it is capacity spill rather than an independent checkpoint",
        "system_path": "paged GPU KV → sparsity plugin/object reaggregation → CPU offload/recall → hot GPU pages",
        "key_mechanism": "A plugin interface plus object reaggregation and hot-page hits integrates recallable sparsity with paged-cache engines",
        "reported_result": "Reports 1.3–1.8× serving-throughput improvement",
        "aegiskv_reuse": "A useful implementation pattern for keeping policy modular while the runtime owns physical placement and movement",
        "evidence_pages": "1;4;8",
        "limitations": "Recallability is not failure recovery; CPU state may share the node failure domain and quality depends on sparsity policy",
    },
    {
        "id": 30,
        "year": 2026,
        "venue": "MLSys",
        "title": "FlexiCache: Leveraging Temporal Stability of Attention Heads for Efficient KV Cache Management",
        "landing_url": "https://proceedings.mlsys.org/paper_files/paper/2026/hash/94bcb01789fccf15afe2764d8fe0f40e-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2026/file/94bcb01789fccf15afe2764d8fe0f40e-Paper-Conference.pdf",
        "pdf_filename": "30_mlsys26_flexicache.pdf",
        "source_kind": "official conference PDF",
        "tier": "T2",
        "directness": "adjacent: head-aware hierarchical KV placement",
        "storage_target": "GPU HBM and host memory",
        "redundancy_or_recovery_path": "Stable-head pages spill to host and can return; no protection replica",
        "system_path": "head-stability classifier → keep unstable heads in HBM → spill/rerank stable-head pages in host memory",
        "key_mechanism": "Uses temporal stability differences among attention heads to decide hierarchical page residency",
        "reported_result": "Reports about 70% GPU-memory reduction, 1.38–1.55× offline throughput, and 1.6–2.1× online throughput",
        "aegiskv_reuse": "Supports fine-grained objects and measured hotness/stability as inputs to lower-level placement",
        "evidence_pages": "1;4;7",
        "limitations": "No redundancy or failure semantics; effectiveness and quality depend on head classification stability",
    },
]


def inspect_pdf(filename: str) -> dict[str, str | int]:
    path = ROOT / filename
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise ValueError(f"{filename}: missing PDF magic")
    return {
        "page_count": len(PdfReader(str(path)).pages),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def validate() -> list[dict]:
    if len(PAPERS) != 30:
        raise ValueError(f"expected 30 records, got {len(PAPERS)}")
    if [p["id"] for p in PAPERS] != list(range(1, 31)):
        raise ValueError("paper IDs must be contiguous from 1 through 30")
    if Counter(p["year"] for p in PAPERS) != Counter({2025: 15, 2026: 15}):
        raise ValueError("year split must be 15 papers for each of 2025 and 2026")
    if Counter(p["tier"] for p in PAPERS) != Counter({"T1": 1, "T2": 15, "T3": 9, "T4": 5}):
        raise ValueError("unexpected relevance-tier distribution")
    records = []
    hashes = set()
    expected_files = {p["pdf_filename"] for p in PAPERS}
    actual_files = {p.name for p in ROOT.glob("*.pdf")}
    if actual_files != expected_files:
        raise ValueError(
            f"PDF inventory mismatch: missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    for paper in PAPERS:
        record = dict(paper)
        record.update(inspect_pdf(paper["pdf_filename"]))
        if record["sha256"] in hashes:
            raise ValueError(f"duplicate PDF content: {paper['pdf_filename']}")
        hashes.add(record["sha256"])
        record["validation_status"] = "PDF magic OK; page count OK; SHA-256 unique"
        records.append(record)
    return records


def write_manifest(records: list[dict]) -> None:
    fields = [
        "id",
        "year",
        "venue",
        "title",
        "landing_url",
        "pdf_url",
        "pdf_filename",
        "source_kind",
        "tier",
        "directness",
        "storage_target",
        "redundancy_or_recovery_path",
        "system_path",
        "key_mechanism",
        "reported_result",
        "aegiskv_reuse",
        "evidence_pages",
        "limitations",
        "page_count",
        "bytes",
        "sha256",
        "validation_status",
    ]
    with (ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_review(records: list[dict]) -> None:
    year_counts = Counter(p["year"] for p in records)
    venue_counts = Counter(p["venue"] for p in records)
    tier_counts = Counter(p["tier"] for p in records)
    source_counts = Counter(p["source_kind"] for p in records)

    lines = [
        "# 2025–2026 KV-cache backup and systems corpus: per-paper evidence review",
        "",
        "Audit date: 2026-07-29.",
        "",
        "## Scope and non-overclaim boundary",
        "",
        "This directory contains exactly 30 genuine 2025–2026 papers from FAST, USENIX ATC, "
        "EuroSys, SOSP, ASPLOS, SIGMOD, MLSys, OSDI, and NSDI. The corpus is intentionally "
        "broader than literal KV-cache backup because a systems design also needs storage tiers, "
        "bandwidth paths, cache admission/eviction, compression, disaggregation, and scheduling evidence.",
        "",
        "**Only GhostServe (paper 16) is classified as a direct fault-tolerant streaming-KV "
        "checkpoint/backup design.** HCache restores KV from saved intermediate activations, Mooncake "
        "replicates hot blocks for load/congestion, and the remaining papers optimize capacity, reuse, "
        "movement, or execution. Those mechanisms are useful substrates, but this review does not "
        "reinterpret them as fault-tolerance evidence.",
        "",
        "Evidence pages below are **1-based physical PDF page numbers**, not the page number printed "
        "inside the proceedings paper. Every local PDF passed magic-byte, page-count, and unique "
        "SHA-256 checks. The complete hashes and source URLs are in `manifest.csv`.",
        "",
        "### Relevance tiers",
        "",
        "| Tier | Meaning | Count |",
        "|---|---|---:|",
    ]
    for tier in ("T1", "T2", "T3", "T4"):
        lines.append(f"| {tier} | {TIER_NAMES[tier]} | {tier_counts[tier]} |")
    lines.extend(
        [
            "",
            "### Corpus audit",
            "",
            f"- Year split: 2025 = {year_counts[2025]}; 2026 = {year_counts[2026]}.",
            f"- Venue counts: {', '.join(f'{k} = {v}' for k, v in sorted(venue_counts.items()))}.",
            "- Source provenance: 24 official conference PDFs; 5 author/institution-hosted accepted "
            "papers; 1 public author preprint corresponding to the SIGMOD 2025 Cache-Craft paper.",
            "- Content hashes: 30 unique SHA-256 values; no duplicate PDF payload.",
            "",
            "## Cross-paper systems takeaways for AegisKV",
            "",
            "The direct and adjacent evidence supports a resource-management formulation with four "
            "separable layers: (1) request/application criticality supplies a value or deadline signal; "
            "(2) a protection action chooses no copy, recomputation/restoration, replication, or erasure "
            "coding; (3) a placement action assigns bytes to GPU HBM, host DRAM, remote memory, or SSD "
            "and chooses an independent failure domain; and (4) a transfer/execution scheduler enforces "
            "per-window capacity, bandwidth, CPU, GPU, and I/O budgets. This is a systems problem only "
            "when the lower three layers remain explicit rather than being collapsed into a request score.",
            "",
            "GhostServe supplies the direct parity/checkpoint action, but its own limitations leave "
            "space for AegisKV: topology-aware cross-node placement, dynamic failure handling, multiple "
            "protection actions, and budget adaptation across time windows. HCache, DroidSpeak, and "
            "Cache-Craft show that recomputation and transfer can be hybrid actions. Mooncake, Strata, "
            "SYMPHONY, Bidaw, and Prism show why memory capacity, storage bandwidth, locality, and "
            "cluster placement should be first-class resources. Compression papers show that byte "
            "reduction must carry measured quality and runtime costs.",
            "",
            "## Per-paper reviews",
            "",
        ]
    )

    for paper in records:
        lines.extend(
            [
                f"### {paper['id']:02d}. {paper['title']}",
                "",
                f"- **Metadata/source:** {paper['venue']} {paper['year']}; "
                f"[landing page]({paper['landing_url']}); "
                f"[source PDF]({paper['pdf_url']}); local file `{paper['pdf_filename']}` "
                f"({paper['page_count']} pages, {paper['bytes']} bytes).",
                f"- **Classification:** {paper['tier']} — {TIER_NAMES[paper['tier']]}; "
                f"{paper['directness']}.",
                f"- **Storage and data path:** {paper['storage_target']}. {paper['system_path']}.",
                f"- **Redundancy/recovery meaning:** {paper['redundancy_or_recovery_path']}.",
                f"- **Mechanism:** {paper['key_mechanism']}.",
                f"- **Reported evidence:** {paper['reported_result']}.",
                f"- **Reusable for AegisKV:** {paper['aegiskv_reuse']}.",
                f"- **Do-not-overclaim / limitation:** {paper['limitations']}.",
                f"- **Evidence pages:** PDF pp. {paper['evidence_pages'].replace(';', ', ')}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Bottom line",
            "",
            "The literature search did not reveal thirty direct KV-backup papers in 2025–2026; "
            "claiming otherwise would be misleading. It produced one direct fault-tolerant checkpoint "
            "paper and twenty-nine carefully labeled adjacent systems papers. That boundary is itself "
            "important prior-art evidence: AegisKV should be positioned as a lower-level, adaptive "
            "resource manager that jointly allocates protection mode, failure-domain placement, memory "
            "capacity, and movement bandwidth over time—not merely as another application-level "
            "criticality formula or another host-offload cache.",
            "",
        ]
    )
    (ROOT / "per-paper-review.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = validate()
    write_manifest(records)
    write_review(records)
    print(
        "Built manifest.csv and per-paper-review.md for "
        f"{len(records)} unique PDFs; tiers={dict(Counter(p['tier'] for p in records))}"
    )


if __name__ == "__main__":
    main()
