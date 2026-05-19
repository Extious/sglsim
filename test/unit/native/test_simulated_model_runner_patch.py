from __future__ import annotations

from types import SimpleNamespace

import torch

from sglsim.srt.simulator.native.model_runner_patch import install


def test_model_runner_patch_installs_once():
    install(SimpleNamespace())
    install(SimpleNamespace())


def test_model_runner_patch_cpu_initialize_forward_and_sample():
    install(SimpleNamespace())

    from sglsim.srt.model_executor.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.model_config = SimpleNamespace(
        dtype=torch.float16,
        context_len=128,
        num_hidden_layers=2,
        num_attention_layers=2,
        vocab_size=17,
    )
    runner.server_args = SimpleNamespace(
        max_total_tokens=256,
        max_running_requests=4,
    )
    runner.page_size = 1

    runner.initialize()
    assert runner.device == "cpu"
    assert runner.req_to_token_pool.size == 4
    assert runner.token_to_kv_pool.size == 256

    output = runner.forward(SimpleNamespace(batch_size=3))
    assert output.can_run_graph is False
    assert output.logits_output.next_token_logits.shape == (3, 17)

    next_token_ids = runner.sample(output.logits_output, SimpleNamespace())
    torch.testing.assert_close(next_token_ids, torch.ones(3, dtype=torch.int64))
