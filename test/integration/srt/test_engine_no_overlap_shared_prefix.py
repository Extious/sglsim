from collections import Counter

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs


def _shared_prefix_requests() -> list[GenerateReqInput]:
    reqs: list[GenerateReqInput] = []
    created = 0.0
    for group in range(4):
        shared = [10_000 + group * 1000 + i for i in range(32)]
        for prompt in range(2):
            dialogue: list[int] = []
            for turn in range(3):
                question = [
                    20_000 + group * 1000 + prompt * 100 + turn * 16 + i
                    for i in range(16)
                ]
                rid = f"g{group}-p{prompt}-t{turn}"
                reqs.append(
                    GenerateReqInput(
                        rid=rid,
                        input_ids=shared + dialogue + question,
                        max_new_tokens=8,
                        routing_key=f"group-{group}",
                        simulation={"created_time": created},
                    )
                )
                dialogue.extend(question)
                dialogue.extend(
                    [
                        30_000 + group * 1000 + prompt * 100 + turn * 8 + i
                        for i in range(8)
                    ]
                )
                created += 0.001
    return reqs


def test_dp2_no_overlap_shared_prefix_engine_completes(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(
        """
        {
          "model_path": "mock-model",
          "cache": {
            "page_size": 1,
            "l1_capacity_pages": 192,
            "l2_capacity_pages": 384,
            "enable_hierarchical_cache": true,
            "hicache_ratio": 2.0,
            "hicache_write_policy": "write_through"
          },
          "scheduler": {
            "dp_size": 2,
            "tp_size": 1,
            "pp_size": 1,
            "max_running_requests": 16,
            "max_prefill_tokens": 64,
            "chunked_prefill_size": 64,
            "init_new_token_ratio": 0.65,
            "load_balance_method": "round_robin",
            "enable_routing_key_affinity": true,
            "decode_log_interval": 8,
            "disable_overlap_schedule": true
          },
          "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}
          },
          "predictor": {
            "name": "heuristic",
            "forward_scale": 0.001
          }
        }
        """,
        encoding="utf-8",
    )
    reqs = _shared_prefix_requests()
    engine = Engine(
        ServerArgs(
            model_path="mock-model",
            dp_size=2,
            sim_config_path=str(cfg),
            disable_overlap_schedule=True,
            log_level="warning",
        )
    )
    try:
        engine.start_background_scheduler()
        for req in reqs:
            engine.submit(req)
        for req in reqs:
            engine.wait_for_rid(req.rid, timeout_s=60)
        result = engine.run_until_idle()
    finally:
        engine.shutdown()

    stats = list(result.request_stats.values())
    assert len(stats) == len(reqs)
    assert Counter(s.status for s in stats) == {"completed": len(reqs)}
    assert result.metrics["completed"] == len(reqs)
    assert result.metrics["stalled"] is False
    assert sum(s.generated_tokens for s in stats) == sum(s.output_length for s in stats)
    assert sum(s.cache_hit_tokens for s in stats) > 0
