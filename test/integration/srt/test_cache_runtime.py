from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS
from sglsim.srt.server_args import ServerArgs

from test.unit.srt.scheduler_helpers import drive_scheduler


def test_scheduler_records_l3_write_pages():
    scheduler = drive_scheduler(
        ServerArgs(model_path="mock-model"),
        [
            GenerateReqInput(
                rid="r0",
                input_len=3,
                max_new_tokens=1,
                simulation={"created_time": 0.0},
            ),
        ],
    )
    assert scheduler.cache.metrics["l3_write_pages"] >= 0


def test_scaled_hicache_workload_completes(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(
        """
        {
          "cache": {
            "page_size": 16,
            "enable_hierarchical_cache": true,
            "hicache_ratio": 2.0,
            "hicache_write_policy": "write_through"
          },
          "scheduler": {
            "max_running_reqs": 16,
            "chunked_prefill_size": 2048,
            "mem_fraction_static": 0.1
          },
          "platform": {
            "accelerator": {"hbm_capacity_gb": 80}
          }
        }
        """,
        encoding="utf-8",
    )
    num_requests = 16
    scheduler = drive_scheduler(
        ServerArgs(
            model_path="mock-model",
            sim_config_path=str(cfg),
        ),
        [
            GenerateReqInput(
                rid=str(i + 1),
                input_len=2000,
                max_new_tokens=2,
                simulation={"created_time": float(i)},
            )
            for i in range(num_requests)
        ],
    )
    assert scheduler._stalled is False
    assert sum(1 for s in REQUEST_STATS.values() if s.status == "completed") == num_requests
