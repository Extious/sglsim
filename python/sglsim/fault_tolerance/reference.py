from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sglsim.fault_tolerance.config import Strategy


@dataclass(frozen=True)
class SynthesisReference:
    """Calibration target: benchmark/synthesis_requests/results/with_hicache_sglang."""

    output_throughput_tok_s: float = 164.66849563195808
    input_throughput_tok_s: float = 16478.595326250936
    reference_concurrency: int = 16
    reference_dp_size: int = 2
    mean_e2e_latency_s: float = 22.40270512882853
    decode_ms_per_token: float = 31.0
    cached_prefill_tok_s: float = 8800.0
    shared_prefix_len: int = 4096
    output_len: int = 256

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SynthesisReference":
        if path is None:
            return cls()
        metrics_path = Path(path)
        if metrics_path.is_dir():
            metrics_path = metrics_path / "metrics.json"
        if not metrics_path.is_file():
            return cls()
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return cls(
            output_throughput_tok_s=float(data.get("output_throughput", cls.output_throughput_tok_s)),
            input_throughput_tok_s=float(data.get("input_throughput", cls.input_throughput_tok_s)),
            mean_e2e_latency_s=float(data.get("mean_e2e_latency_ms", cls.mean_e2e_latency_s * 1000)) / 1000.0,
        )


DEFAULT_REFERENCE = SynthesisReference()


def concurrency_utilization(concurrency: int, *, reference: SynthesisReference = DEFAULT_REFERENCE) -> float:
    ref = max(1, reference.reference_concurrency)
    return min(1.0, float(concurrency) / float(ref))


def calibrated_output_throughput(
    strategy: Strategy,
    *,
    concurrency: int,
    dp_size: int = 2,
    reference: SynthesisReference = DEFAULT_REFERENCE,
) -> float:
    """Cluster output tok/s calibrated to with_hicache_sglang at concurrency=16."""
    dp_scale = float(dp_size) / float(max(1, reference.reference_dp_size))
    base = reference.output_throughput_tok_s * concurrency_utilization(concurrency, reference=reference) * dp_scale

    if strategy == "baseline":
        return base

    if strategy == "host_backup":
        # Background D2H backup stream; small bandwidth tax, no VRAM halving.
        bw_tax = 0.012 * concurrency_utilization(concurrency, reference=reference)
        return base * (1.0 - 0.025 - bw_tax)

    if strategy == "gpu_backup":
        # Peer replication consumes ~half device KV budget under load (KevlarFlow Fig.9 style).
        load = concurrency_utilization(concurrency, reference=reference)
        vram_tax = 0.02 + 0.48 * load
        peer_copy_tax = 0.008 * load
        return base * (1.0 - vram_tax - peer_copy_tax)

    if strategy == "remote_backup":
        return base * (1.0 - 0.03 - 0.015 * concurrency_utilization(concurrency, reference=reference))

    return base


def estimate_request_service_s(
    *,
    input_len: int,
    output_len: int,
    shared_prefix_len: int,
    recompute_tokens: int = 0,
    reference: SynthesisReference = DEFAULT_REFERENCE,
) -> float:
    """Per-request wall time using synthesis-scale prefill/decode predictors."""
    cached_prefill = min(max(0, shared_prefix_len), input_len)
    fresh_prefill = max(0, input_len - cached_prefill) + max(0, recompute_tokens)
    prefill_s = fresh_prefill / reference.cached_prefill_tok_s
    decode_s = output_len * (reference.decode_ms_per_token / 1000.0)
    return prefill_s + decode_s
