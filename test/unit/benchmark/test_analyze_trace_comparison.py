import csv

from benchmark.synthesis_requests.analyze_logs import summarize_trace_comparison


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_trace_comparison_uses_real_e2e_and_sim_logical_total(tmp_path):
    real = tmp_path / "real.csv"
    sim = tmp_path / "sim.csv"
    _write_csv(
        real,
        [
            {"rid": "0", "e2e_latency_s": "50", "cache_hit_tokens": "100"},
            {"rid": "1", "e2e_latency_s": "70", "cache_hit_tokens": "200"},
        ],
    )
    _write_csv(
        sim,
        [
            {
                "rid": "0",
                "e2e_latency_s": "1",
                "logical_total_latency_s": "55",
                "cache_hit_tokens": "100",
            },
            {
                "rid": "1",
                "e2e_latency_s": "1",
                "logical_total_latency_s": "65",
                "cache_hit_tokens": "180",
            },
        ],
    )

    summary = summarize_trace_comparison(real, sim)

    assert summary["request_count"] == 2
    assert summary["real_latency_field"] == "e2e_latency_s"
    assert summary["sim_latency_field"] == "logical_total_latency_s"
    assert summary["real_latency_p50_s"] == 60.0
    assert summary["sim_latency_p50_s"] == 60.0
    assert summary["latency_mae_s"] == 5.0
    assert summary["cache_hit_mae_tokens"] == 10.0
