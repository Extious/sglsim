import json

from benchmark.synthesis_requests.diagnostics.run_integrity import check_run_integrity
from benchmark.synthesis_requests.direct_engine import run_direct_experiment


def test_direct_engine_synthesis_writes_complete_run_artifacts(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "client.json").write_text(
        json.dumps(
            {
                "dataset_name": "generated-shared-prefix",
                "seed": 1,
                "request_rate": "inf",
                "max_concurrency": 2,
                "gsp_num_groups": 2,
                "gsp_prompts_per_group": 1,
                "gsp_system_prompt_len": 16,
                "gsp_question_len": 8,
                "gsp_output_len": 2,
                "gsp_range_ratio": 1,
                "gsp_fast_prepare": False,
                "gsp_send_routing_key": True,
                "gsp_num_turns": 2,
                "gsp_ordered": True,
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "server.json").write_text(
        json.dumps(
            {
                "experiment_name": "direct-tiny",
                "model_path": "mock-model",
                "host": "127.0.0.1",
                "port": 30000,
                "cache": {
                    "page_size": 1,
                    "l1_capacity_pages": 128,
                    "l2_capacity_pages": 256,
                    "enable_hierarchical_cache": True,
                    "hicache_write_policy": "write_through",
                },
                    "scheduler": {
                        "dp_size": 2,
                        "tp_size": 1,
                        "pp_size": 1,
                        "max_running_requests": 4,
                        "max_prefill_tokens": 32,
                        "chunked_prefill_size": 32,
                        "enable_routing_key_affinity": True,
                        "decode_log_interval": 1,
                        "disable_overlap_schedule": True,
                    },
                "predictor": {
                    "name": "heuristic",
                    "forward_scale": 0.001,
                },
                "platform": {
                    "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
                },
            }
        ),
        encoding="utf-8",
    )

    run_dir = run_direct_experiment(
        config_dir,
        tmp_path / "results",
        log_level="info",
        disable_overlap_schedule=True,
    )

    assert (run_dir / "request_details.csv").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "server.log").is_file()
    assert (run_dir / "server.stderr.log").is_file()
    assert (run_dir / "direct_engine_metrics.json").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["completed"] == 4
    assert metrics["failed"] == 0
    server_log = (run_dir / "server.log").read_text(encoding="utf-8")
    assert "Prefill batch" in server_log
    assert "Decode" in server_log
    assert check_run_integrity(run_dir).ok is True
