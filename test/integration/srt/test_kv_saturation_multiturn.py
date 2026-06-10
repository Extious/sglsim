"""Integration: KV saturation with multi-turn-style shared-prefix workload."""

from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS
from sglsim.srt.server_args import ServerArgs

from test.unit.srt.scheduler_helpers import drive_scheduler


def test_kv_saturation_multiturn_completes(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(
        """
        {
          "cache": {
            "page_size": 16,
            "enable_hierarchical_cache": true,
            "hicache_ratio": 2.0,
            "hicache_write_policy": "write_through_selective"
          },
          "scheduler": {
            "max_running_reqs": 16,
            "chunked_prefill_size": 512,
            "init_new_token_ratio": 1.0,
            "mem_fraction_static": 0.5
          },
          "platform": {
            "accelerator": {"hbm_capacity_gb": 80}
          }
        }
        """,
        encoding="utf-8",
    )
    reqs: list[GenerateReqInput] = []
    created = 0.0
    for group in range(4):
        for prompt in range(2):
            for turn in range(3):
                input_len = 768 + turn * 256
                rid = f"{group}-p{prompt}-t{turn}"
                reqs.append(
                    GenerateReqInput(
                        rid=rid,
                        input_len=input_len,
                        max_new_tokens=32,
                        simulation={"created_time": created},
                    )
                )
                created += 0.01

    scheduler = drive_scheduler(
        ServerArgs(
            model_path="mock-model",
            sim_config_path=str(cfg),
        ),
        reqs,
    )
    completed = sum(1 for s in REQUEST_STATS.values() if s.status == "completed")
    assert completed == len(reqs)
    assert scheduler._stalled is False
