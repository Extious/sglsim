from benchmark.synthesis_requests.analyze_logs import parse_server_log_lines, summarize_events


def test_parse_server_log_lines_supports_sglang_and_sim_timestamps():
    lines = [
        "[434.701s DP0] Decode batch [40], #running-req: 16, #token: 366528, token usage: 0.92, gen throughput (token/s): 1094.92, #queue-req: 0, #logical-token: 415525, #avail-token: 30912, #evictable-token: 0",
        "[2026-05-29 11:43:15.123 DP1] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, token usage: 0.02, #running-req: 0, #queue-req: 4, cuda graph: True, input throughput (token/s): 6481.02",
        "[2026-05-29 11:44:59 DP0] KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 940, #new_token_ratio: 0.0980 -> 0.9384",
        "[2026-05-29 11:45:02] INFO:     127.0.0.1:45824 - \"POST /v1/completions HTTP/1.1\" 200 OK",
        "Capturing num tokens (num_tokens=4 avail_mem=10.64 GB): 100%|##########| 58/58",
    ]

    events = parse_server_log_lines(lines)

    assert [event.event_type for event in events] == [
        "decode",
        "prefill",
        "kv_retract",
        "http_access",
    ]
    assert events[0].sim_time_s == 434.701
    assert events[0].dp_rank == 0
    assert events[0].fields["running_reqs"] == 16
    assert events[0].fields["logical_tokens"] == 415525
    assert events[1].wall_time == "2026-05-29 11:43:15.123"
    assert events[1].dp_rank == 1
    assert events[1].fields["cuda_graph"] is True
    assert events[2].fields["new_token_ratio_before"] == 0.0980
    assert events[2].fields["new_token_ratio_after"] == 0.9384
    assert events[3].fields["method"] == "POST"
    assert events[3].fields["path"] == "/v1/completions"
    assert events[3].fields["status_code"] == 200


def test_summarize_events_reports_core_alignment_metrics():
    events = parse_server_log_lines(
        [
            "[0.000s DP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #logical-token: 0, #avail-token: 397440, #evictable-token: 0, input throughput (token/s): 0.00",
            "[1.000s DP0] Decode batch [40], #running-req: 1, #token: 9000, token usage: 0.50, gen throughput (token/s): 300.00, #queue-req: 2, #logical-token: 9000, #avail-token: 1000, #evictable-token: 0",
            "[2.000s DP0] Decode batch [80], #running-req: 1, #token: 9100, token usage: 0.75, gen throughput (token/s): 320.00, #queue-req: 1, #logical-token: 9100, #avail-token: 900, #evictable-token: 0",
            "[2026-05-29 11:45:02] INFO:     127.0.0.1:45824 - \"POST /v1/completions HTTP/1.1\" 200 OK",
        ]
    )

    summary = summarize_events(events)

    assert summary["prefill_count"] == 1
    assert summary["decode_count"] == 2
    assert summary["http_access_count"] == 1
    assert summary["kv_retract_count"] == 0
    assert summary["max_token_usage"] == 0.75
    assert summary["decode_throughput_p50"] == 310.0
