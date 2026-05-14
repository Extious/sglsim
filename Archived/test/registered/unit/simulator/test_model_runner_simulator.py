import types
import unittest

import torch

from sglang.srt.simulator.model_runner import (
    SimulatorMockModel,
    build_mock_logits_output,
    build_mock_next_token_ids,
)


class TestSimulatorModelRunnerHelpers(unittest.TestCase):
    def test_mock_model_has_forward(self):
        self.assertIsNone(SimulatorMockModel().forward())

    def test_build_mock_logits_output_uses_batch_size_and_vocab(self):
        runner = types.SimpleNamespace(
            device="cpu",
            model_config=types.SimpleNamespace(vocab_size=32000),
        )
        forward_batch = types.SimpleNamespace(batch_size=3)

        output = build_mock_logits_output(runner, forward_batch)

        self.assertEqual(tuple(output.next_token_logits.shape), (3, 32000))
        self.assertEqual(output.next_token_logits.device.type, "cpu")

    def test_build_mock_next_token_ids(self):
        runner = types.SimpleNamespace(device="cpu")
        forward_batch = types.SimpleNamespace(batch_size=4)

        ids = build_mock_next_token_ids(runner, forward_batch)

        self.assertTrue(torch.equal(ids, torch.ones(4, dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main()
