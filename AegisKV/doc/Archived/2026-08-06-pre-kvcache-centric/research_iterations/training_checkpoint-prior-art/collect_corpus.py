#!/usr/bin/env python3
"""Download and mechanically validate the training-fault-tolerance corpus.

The venue URL is the bibliographic authority.  ``pdf_url`` may be an
official proceedings copy or an author-hosted accepted/version-of-record
copy; the manifest records that distinction explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import html
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from pypdf import PdfReader


OUT = Path(__file__).resolve().parent

CANDIDATES = [
    {
        "id": "TC01",
        "year": 2025,
        "venue": "ASPLOS 2025",
        "venue_scope": "main",
        "title": "PCcheck: Persistent Concurrent Checkpointing for ML",
        "venue_url": "https://doi.org/10.1145/3669940.3707255",
        "pdf_url": "https://fotstrt.github.io/files/2025-pccheck.pdf",
        "pdf_source": "author accepted paper",
        "filename": "2025-asplos-pccheck.pdf",
        "directness": "direct_checkpoint",
    },
    {
        "id": "TC02",
        "year": 2025,
        "venue": "EuroSys 2025",
        "venue_scope": "main",
        "title": "FlowCheck: Decoupling Checkpointing and Training of Large-Scale Models",
        "venue_url": "https://doi.org/10.1145/3689031.3696088",
        "pdf_url": "https://jhc.sjtu.edu.cn/~bjiang/papers/Huang_EuroSys2025_FlowCheck.pdf",
        "pdf_source": "author accepted paper",
        "filename": "2025-eurosys-flowcheck.pdf",
        "directness": "direct_checkpoint",
    },
    {
        "id": "TC03",
        "year": 2025,
        "venue": "NSDI 2025",
        "venue_scope": "main",
        "title": "ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development",
        "venue_url": "https://www.usenix.org/conference/nsdi25/presentation/wan-borui",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2025-nsdi-bytecheckpoint.pdf",
        "directness": "direct_checkpoint",
    },
    {
        "id": "TC04",
        "year": 2025,
        "venue": "USENIX ATC 2025",
        "venue_scope": "main",
        "title": "Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism",
        "venue_url": "https://www.usenix.org/conference/atc25/presentation/lian",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2025-atc-universal-checkpointing.pdf",
        "directness": "direct_checkpoint",
    },
    {
        "id": "TC05",
        "year": 2025,
        "venue": "SOSP 2025",
        "venue_scope": "main",
        "title": "Robust LLM Training Infrastructure at ByteDance",
        "venue_url": "https://doi.org/10.1145/3731569.3764838",
        "pdf_url": "https://i.cs.hku.hk/~cwu/papers/brwan-sosp25.pdf",
        "pdf_source": "author version of record",
        "filename": "2025-sosp-byterobust.pdf",
        "directness": "direct_checkpoint_and_recovery",
    },
    {
        "id": "TC06",
        "year": 2025,
        "venue": "SOSP 2025",
        "venue_scope": "main",
        "title": "PhoenixOS: Concurrent OS-level GPU Checkpoint and Restore with Validated Speculation",
        "venue_url": "https://doi.org/10.1145/3731569.3764813",
        "pdf_url": "https://ipads.se.sjtu.edu.cn/zh/publications/sosp25-wei.pdf",
        "pdf_source": "author accepted paper",
        "filename": "2025-sosp-phoenixos.pdf",
        "directness": "direct_system_checkpoint_restore",
    },
    {
        "id": "TC07",
        "year": 2025,
        "venue": "PPoPP 2025",
        "venue_scope": "main",
        "title": "ATTNChecker: Highly-Optimized Fault Tolerant Attention for Large Language Model Training",
        "venue_url": "https://doi.org/10.1145/3710848.3710870",
        "pdf_url": "https://arxiv.org/pdf/2410.11720",
        "pdf_source": "author preprint corresponding to accepted paper",
        "filename": "2025-ppopp-attnchecker.pdf",
        "directness": "recovery_avoidance",
    },
    {
        "id": "TC08",
        "year": 2025,
        "venue": "HPDC 2025",
        "venue_scope": "main",
        "title": "FT2: First-Token-Inspired Online Fault Tolerance on Output Tokens for Large Language Model Training",
        "venue_url": "https://doi.org/10.1145/3731545.3731555",
        "pdf_url": "https://lishanyang.github.io/HPDC25_FT2.pdf",
        "pdf_source": "author accepted paper",
        "filename": "2025-hpdc-ft2.pdf",
        "directness": "recovery_avoidance",
    },
    {
        "id": "TC09",
        "year": 2025,
        "venue": "NSDI 2025",
        "venue_scope": "main",
        "title": "Minder: Faulty Machine Detection for Large-scale Distributed Model Training",
        "venue_url": "https://www.usenix.org/conference/nsdi25/presentation/deng",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2025-nsdi-minder.pdf",
        "directness": "recovery_support_detection",
    },
    {
        "id": "TC10",
        "year": 2025,
        "venue": "NSDI 2025",
        "venue_scope": "main",
        "title": "Holmes: Localizing Irregularities in LLM Training with Mega-scale GPU Clusters",
        "venue_url": "https://www.usenix.org/conference/nsdi25/presentation/yao",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2025-nsdi-holmes.pdf",
        "directness": "recovery_support_diagnosis",
    },
    {
        "id": "TC11",
        "year": 2025,
        "venue": "NSDI 2025",
        "venue_scope": "main",
        "title": "Evolution of Aegis: Fault Diagnosis for AI Model Training Service in Production",
        "venue_url": "https://www.usenix.org/conference/nsdi25/presentation/dong",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2025-nsdi-evolution-aegis.pdf",
        "directness": "recovery_support_diagnosis",
    },
    {
        "id": "TC12",
        "year": 2025,
        "venue": "OSDI 2025",
        "venue_scope": "main",
        "title": "Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks",
        "venue_url": "https://www.usenix.org/conference/osdi25/presentation/jiang",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2025-osdi-traincheck.pdf",
        "directness": "recovery_support_sdc_detection",
    },
    {
        "id": "TC13",
        "year": 2025,
        "venue": "SOSP 2025",
        "venue_scope": "main",
        "title": "Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training",
        "venue_url": "https://doi.org/10.1145/3731569.3764848",
        "pdf_url": "https://minlanyu.seas.harvard.edu/writeup/sosp25.pdf",
        "pdf_source": "author accepted paper",
        "filename": "2025-sosp-mycroft.pdf",
        "directness": "recovery_support_diagnosis",
    },
    {
        "id": "TC14",
        "year": 2025,
        "venue": "SOSP 2025",
        "venue_scope": "main",
        "title": "TrainVerify: Equivalence-Based Verification for Distributed LLM Training",
        "venue_url": "https://doi.org/10.1145/3731569.3764850",
        "pdf_url": "https://arxiv.org/pdf/2506.15961",
        "pdf_source": "author preprint corresponding to accepted paper",
        "filename": "2025-sosp-trainverify.pdf",
        "directness": "recovery_support_prevention",
    },
    {
        "id": "TC15",
        "year": 2025,
        "venue": "PDSW at SC 2025",
        "venue_scope": "peer-reviewed workshop",
        "title": "LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models",
        "venue_url": "https://doi.org/10.1145/3731599.3767515",
        "pdf_url": "https://arxiv.org/pdf/2602.22158",
        "pdf_source": "author preprint corresponding to workshop paper",
        "filename": "2025-pdsw-llmtailor.pdf",
        "directness": "direct_checkpoint",
    },
    {
        "id": "TC16",
        "year": 2025,
        "venue": "PDSW at SC 2025",
        "venue_scope": "work-in-progress",
        "title": "LLM training in practice: insights from 85,000 checkpoints",
        "venue_url": "https://sc25.supercomputing.org/proceedings/workshops/workshop_pages/ws_pdswwip108.html",
        "pdf_url": "https://pdsw.org/pdsw25/papers/wips/ws_pdswwip108s2-file1.pdf",
        "pdf_source": "official workshop WIP paper",
        "filename": "2025-pdsw-85000-checkpoints.pdf",
        "directness": "direct_empirical_wip",
    },
    {
        "id": "TC17",
        "year": 2026,
        "venue": "NSDI 2026",
        "venue_scope": "main",
        "title": "Checkmate: Zero Performance Overhead Model Checkpointing via Network Gradient Replication",
        "venue_url": "https://www.usenix.org/conference/nsdi26/presentation/bhardwaj",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-nsdi-checkmate.pdf",
        "directness": "direct_checkpoint",
    },
    {
        "id": "TC18",
        "year": 2026,
        "venue": "NSDI 2026",
        "venue_scope": "main",
        "title": "Sparse Checkpointing for Fast and Reliable MoE Training",
        "venue_url": "https://www.usenix.org/conference/nsdi26/presentation/gandhi",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-nsdi-moevement.pdf",
        "directness": "direct_checkpoint",
    },
    {
        "id": "TC19",
        "year": 2026,
        "venue": "FAST 2026",
        "venue_scope": "main",
        "title": "AdaCheck: An Adaptive Checkpointing System for Efficient LLM Training with Redundancy Utilization",
        "venue_url": "https://www.usenix.org/conference/fast26/presentation/liu-weijie",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-fast-adacheck.pdf",
        "directness": "direct_checkpoint",
    },
    {
        "id": "TC20",
        "year": 2026,
        "venue": "FAST 2026",
        "venue_scope": "main",
        "title": "GPU Checkpoint/Restore Made Fast and Lightweight",
        "venue_url": "https://www.usenix.org/conference/fast26/presentation/zeng",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-fast-gcr.pdf",
        "directness": "direct_system_checkpoint_restore",
    },
    {
        "id": "TC21",
        "year": 2026,
        "venue": "OSDI 2026",
        "venue_scope": "main",
        "title": "TrainMover: An Interruption-Resilient Runtime for ML Training",
        "venue_url": "https://www.usenix.org/conference/osdi26/presentation/lao",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-osdi-trainmover.pdf",
        "directness": "recovery_and_state_migration",
    },
    {
        "id": "TC22",
        "year": 2026,
        "venue": "OSDI 2026",
        "venue_scope": "main",
        "title": "RobustRL: Role-Based Fault Tolerance System for RL Post-Training",
        "venue_url": "https://www.usenix.org/conference/osdi26/presentation/chen-zhenqian",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-osdi-robustrl.pdf",
        "directness": "recovery_and_role_state",
    },
    {
        "id": "TC23",
        "year": 2026,
        "venue": "OSDI 2026",
        "venue_scope": "main",
        "title": "SDCs in the Wild: Characterizing and Diagnosing SDC-Defective GPUs in Production LLM Training",
        "venue_url": "https://www.usenix.org/conference/osdi26/presentation/zheng",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-osdi-sdcs-in-the-wild.pdf",
        "directness": "recovery_support_sdc_diagnosis",
    },
    {
        "id": "TC24",
        "year": 2026,
        "venue": "OSDI 2026",
        "venue_scope": "main",
        "title": "Safeguarding LLM Training at Scale: Online SDC Detection and Insights from 35 Million GPU Hours",
        "venue_url": "https://www.usenix.org/conference/osdi26/presentation/lei",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-osdi-aegis-sdc.pdf",
        "directness": "recovery_support_sdc_detection",
    },
    {
        "id": "TC25",
        "year": 2026,
        "venue": "OSDI 2026",
        "venue_scope": "main",
        "title": "OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training",
        "venue_url": "https://www.usenix.org/conference/osdi26/presentation/zhou-ziming",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-osdi-opguard.pdf",
        "directness": "recovery_support_debugging",
    },
    {
        "id": "TC26",
        "year": 2026,
        "venue": "MLSys 2026",
        "venue_scope": "main",
        "title": "Sparing Strategies to Minimize Reliability Impact On Large Training Jobs",
        "venue_url": "https://proceedings.mlsys.org/paper_files/paper/2026/hash/437bc4ccafd3fc6d4289bd10940be42b-Abstract-Conference.html",
        "pdf_url": "https://proceedings.mlsys.org/paper_files/paper/2026/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf",
        "pdf_source": "official proceedings",
        "filename": "2026-mlsys-sparing-strategies.pdf",
        "directness": "recovery_capacity_planning",
    },
    {
        "id": "TC27",
        "year": 2026,
        "venue": "NSDI 2026",
        "venue_scope": "main",
        "title": "EROICA: Online Performance Troubleshooting for Large-scale Model Training",
        "venue_url": "https://www.usenix.org/conference/nsdi26/presentation/guan-yu",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-nsdi-eroica.pdf",
        "directness": "recovery_support_diagnosis",
    },
    {
        "id": "TC28",
        "year": 2026,
        "venue": "OSDI 2026",
        "venue_scope": "main",
        "title": "Teaching the Old Dog New Tricks: Building Efficient Data Pipelines for Large-Scale LLM Pre-Training",
        "venue_url": "https://www.usenix.org/conference/osdi26/presentation/chen-luofan",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-osdi-checkpoint-loading-pipeline.pdf",
        "directness": "checkpoint_loading_and_replication",
    },
    {
        "id": "TC29",
        "year": 2026,
        "venue": "NSDI 2026",
        "venue_scope": "main",
        "title": "Di-PS: System-Algorithm Co-Design for Asynchronous and Heterogeneous Cross-cluster LLM Training at Scale",
        "venue_url": "https://www.usenix.org/conference/nsdi26/presentation/li-shengwei",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-nsdi-di-ps.pdf",
        "directness": "recovery_and_state_replication",
    },
    {
        "id": "TC30",
        "year": 2026,
        "venue": "NSDI 2026",
        "venue_scope": "main",
        "title": "FLARE: Anomaly Diagnostics for Divergent LLM Training in GPU Clusters of Thousand-Plus Scale",
        "venue_url": "https://www.usenix.org/conference/nsdi26/presentation/cui",
        "pdf_url": None,
        "pdf_source": "official proceedings",
        "filename": "2026-nsdi-flare.pdf",
        "directness": "recovery_support_diagnosis",
    },
]


def resolve_usenix_pdf(page_url: str) -> str:
    response = requests.get(page_url, timeout=60)
    response.raise_for_status()
    links = []
    for raw in re.findall(
        r"""href=["']([^"']+\.pdf(?:\?[^"']*)?)["']""",
        response.text,
        flags=re.I,
    ):
        link = urljoin(page_url, html.unescape(raw))
        low = link.lower()
        if "slides" in low or "prepub" in low or "proceedings" in low:
            continue
        links.append(link)
    if not links:
        raise RuntimeError(f"No paper PDF found on {page_url}")
    return links[0]


def main() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "AegisKV-research-corpus/1.0"
    rows = []
    for item in CANDIDATES:
        pdf_url = item["pdf_url"]
        if pdf_url is None:
            pdf_url = resolve_usenix_pdf(item["venue_url"])
        path = OUT / item["filename"]
        path = OUT / item["filename"]
        if path.exists() and path.read_bytes()[:5] == b"%PDF-":
            data = path.read_bytes()
            downloaded = False
        else:
            response = session.get(pdf_url, timeout=60)
            response.raise_for_status()
            data = response.content
            downloaded = True
        if not data.startswith(b"%PDF-"):
            raise RuntimeError(
                f"{item['id']} is not a PDF: {pdf_url} "
                f"({data[:20]!r})"
            )
        if downloaded:
            path.write_bytes(data)
        reader = PdfReader(path)
        digest = hashlib.sha256(data).hexdigest()
        rows.append(
            {
                **item,
                "pdf_url": pdf_url,
                "bytes": len(data),
                "pages": len(reader.pages),
                "sha256": digest,
                "pdf_magic": "ok",
            }
        )
        print(
            f"{item['id']} {path.name}: {len(reader.pages)} pages, "
            f"{len(data)} bytes, {digest[:12]}"
        )

    hashes = [row["sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Duplicate PDF hashes detected")

    fieldnames = list(rows[0])
    with (OUT / "download_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"validated {len(rows)} unique PDFs")


if __name__ == "__main__":
    main()
