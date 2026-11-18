# Related Work Source Index

This note preserves the curated source list and the reason each work matters to
`../sections/02_background_related.tex`. The former `pdfs/...` values below are
source filenames, not local links: the 25 PDF copies were intentionally removed
so that `paper_outline.pdf` remains the only PDF in this package. Formal
citations are maintained in `../ref.bib`.

## Problem Formulation And Resource Partitioning

| Paper | Local PDF | Why it is included |
|---|---|---|
| One Pool - Two Caches | `pdfs/One Pool - Two Caches - Adaptive HBM Partitioning for Accelerating Generative Recommender Serving.pdf` | Dynamic HBM partitioning, incremental budget changes, refill interference, routing-resource coupling |
| DRLPart | `pdfs/DRLPart - A Deep Reinforcement Learning Framework for Optimally Efficient and Robust Resource Partitioning on Commodity Servers.pdf` | State/action/reward framing and resource-pressure-aware partitioning |
| FleetIO | `pdfs/FleetIO - Managing Multi-Tenant Cloud Storage with Multi-Agent Reinforcement Learning.pdf` | Dynamic placement and migration cost under multi-tenant pressure |

## KV Cache, Restore, And Recompute

| Paper | Local PDF | Why it is included |
|---|---|---|
| LLM KV-cache - To Restore or To Recompute | `pdfs/LLM KV-cache - To Restore or To Recompute, That Is the Question.pdf` | Restore/recompute cost boundary for KV recovery |
| Position - Disaggregated LLM KV-Cache Storage Requires Coordination-Free Consistency Abstractions | `pdfs/Position - Disaggregated LLM KV-Cache Storage Requires Coordination-Free Consistency Abstractions.pdf` | KV storage consistency and metadata semantics |
| Prompt Cache | `pdfs/Prompt Cache - Modular Attention Reuse for Low-Latency Inference.pdf` | Prefix reuse and modular attention caching |
| KVShare | `pdfs/KVShare - An LLM Service System with Efficient and Effective Multi-Tenant KV Cache Reuse.pdf` | Multi-tenant KV reuse and sharing |
| Shared Disk KV Cache Management | `pdfs/Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs.pdf` | Cross-instance KV storage management |
| Apt-Serve | `pdfs/Apt-Serve - Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving.pdf` | Hybrid cache scheduling and memory pressure |

## Fault-Tolerant LLM Serving

| Paper | Local PDF | Why it is included |
|---|---|---|
| DejaVu | `pdfs/DejaVu - KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving.pdf` | KV-oriented fault-tolerant serving |
| FailSafe | `pdfs/FailSafe - High-performance Resilient Serving.pdf` | High-performance resilient LLM serving baseline |
| AnchorTP | `pdfs/AnchorTP - Resilient LLM Inference with State-Preserving Elastic Tensor Parallelism.pdf` | State-preserving recovery for parallel inference |
| ReviveMoE | `pdfs/ReviveMoE - Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments.pdf` | Hardware-failure recovery in large-scale serving |
| ReaLM | `pdfs/ReaLM - Reliable and Efficient Large Language Model Inference with Statistical Algorithm-Based Fault Tolerance.pdf` | Algorithm-assisted fault tolerance |
| KevlarFlow | `pdfs/Towards Resiliency in Large Language Model Serving with KevlarFlow.pdf` | Resilient LLM serving workflow/system context |

## Agentic Workflow Serving

| Paper | Local PDF | Why it is included |
|---|---|---|
| AAFLOW | `pdfs/AAFLOW - Scalable Patterns for Agentic AI Workflows.pdf` | Agentic workflow patterns and scaling |
| Efficient Serving for Agentic LLM Workflows | `pdfs/Efficient Serving for Agentic LLM Workflows via Micro-Task-Level Parallelism.pdf` | Workflow micro-task scheduling |
| Weaver | `pdfs/Weaver - Efficient Agentic Workflow Serving via Dependency-Aware Overlap and Prefill Co-location.pdf` | Dependency-aware workflow serving |
| Ceres | `pdfs/Ceres - Low-latency Multi-Agent Serving with Shared LLMs via Perceiving Differences across Agents.pdf` | Multi-agent serving with shared LLMs |
| HEXGEN-TEXT2SQL | `pdfs/HEXGEN-TEXT2SQL - Optimizing LLM Inference Request Scheduling for Agentic Text-to-SQL Workflow.pdf` | Agentic workflow scheduling |
| RCR-Router | `pdfs/RCR-Router - Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory.pdf` | Role-aware context routing |

## Locality-Aware And Memory-Constrained Scheduling

| Paper | Local PDF | Why it is included |
|---|---|---|
| Online Scheduling for LLM Inference with KV Cache Constraints | `pdfs/Online Scheduling for LLM Inference with KV Cache Constraints.pdf` | KV-constrained online scheduling |
| Optimizing LLM Inference - Fluid-Guided Online Scheduling with Memory Constraints | `pdfs/Optimizing LLM Inference - Fluid-Guided Online Scheduling with Memory Constraints.pdf` | Memory-constrained scheduling |
| Locality-aware Fair Scheduling in LLM Serving | `pdfs/Locality-aware Fair Scheduling in LLM Serving.pdf` | Joint locality and load/fairness consideration |

## Notes

- `Papers/` or the original publisher/arXiv source remains the paper library.
- When adding a citation, update `../ref.bib`,
  `../sections/02_background_related.tex`, and this rationale index together.
- Do not add PDF copies under `paper_outline/`; the validator enforces a single
  PDF deliverable.
