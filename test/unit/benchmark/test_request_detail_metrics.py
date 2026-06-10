import csv

from benchmark.synthesis_requests.utils import sim_stats_to_detail_row, write_request_details_csv


def test_sim_stats_to_detail_row_latency_breakdown():
    row = sim_stats_to_detail_row(
        "1",
        {
            "status": "completed",
            "created_time": 10.0,
            "queue_start": 12.0,
            "queue_end": 15.0,
            "last_event_time": 20.0,
            "gen_token_latencies": [2.0, 1.0, 2.0],
            "input_length": 100,
            "output_length": 3,
            "generated_tokens": 2,
            "cache_hit_tokens": 80,
            "load_back_tokens": 32,
        },
        e2e_latency_s=1.5,
    )
    assert row.waiting_time_s == 3.0
    assert row.inference_time_s == 5.0
    assert row.total_latency_s == 10.0
    assert (
        row.waiting_time_s
        + row.inference_time_s
        + (12.0 - 10.0)
        == row.total_latency_s
    )
    assert row.client_wall_e2e_latency_s == 1.5
    assert row.logical_queue_time_s == 3.0
    assert row.logical_service_time_s == 5.0
    assert row.logical_total_latency_s == 10.0
    assert row.target_output_tokens == 3
    assert row.generated_tokens == 2
    assert row.cache_hit_tokens == 80
    assert row.load_back_tokens == 32
    assert row.ttft_ms == 2000.0
    assert row.mean_tbt_ms == 1500.0


def test_sim_stats_to_detail_row_zero_waiting_when_immediate_schedule():
    row = sim_stats_to_detail_row(
        "2",
        {
            "status": "completed",
            "created_time": 8.0,
            "queue_start": 8.0,
            "queue_end": 8.0,
            "last_event_time": 11.0,
            "gen_token_latencies": [3.0],
            "input_length": 50,
            "output_length": 1,
            "generated_tokens": 1,
            "cache_hit_tokens": 0,
        },
        e2e_latency_s=0.5,
    )
    assert row.waiting_time_s == 0.0
    assert row.inference_time_s == 3.0
    assert row.total_latency_s == 3.0
    assert row.waiting_time_s + row.inference_time_s == row.total_latency_s


def test_request_details_csv_exports_explicit_logical_and_wall_fields(tmp_path):
    row = sim_stats_to_detail_row(
        "3",
        {
            "status": "completed",
            "created_time": 0.0,
            "queue_start": 1.0,
            "queue_end": 2.0,
            "last_event_time": 5.0,
            "service_time_s": 2.5,
            "gen_token_latencies": [2.5],
            "input_length": 10,
            "output_length": 1,
        },
        e2e_latency_s=0.25,
    )

    path = write_request_details_csv(tmp_path, [row])

    with path.open(encoding="utf-8", newline="") as f:
        record = next(csv.DictReader(f))
    assert record["client_wall_e2e_latency_s"] == "0.25"
    assert record["logical_total_latency_s"] == "5.0"
    assert record["logical_service_time_s"] == "2.5"
    assert record["logical_queue_time_s"] == "1.0"
