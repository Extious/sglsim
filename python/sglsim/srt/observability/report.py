from __future__ import annotations

from sglsim.srt.observability.bench_metrics import BenchmarkMetrics


def print_benchmark_results(metrics: dict, backend: str = "sglsim") -> None:
    m = BenchmarkMetrics(**{k: metrics[k] for k in BenchmarkMetrics.__dataclass_fields__})
    print("============ Serving Benchmark Result ============")
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Successful requests:", m.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", m.duration))
    print("{:<40} {:<10}".format("Total input tokens:", m.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", m.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", m.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", m.output_throughput))
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", m.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median E2E Latency (ms):", m.median_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 E2E Latency (ms):", m.p99_e2e_latency_ms))
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", m.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", m.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", m.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", m.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", m.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", m.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-Token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", m.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", m.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P95 ITL (ms):", m.p95_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", m.p99_itl_ms))
    print("{:<40} {:<10.2f}".format("Max ITL (ms):", m.max_itl_ms))
    print("=" * 50)
