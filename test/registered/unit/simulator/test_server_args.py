import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.server_args import prepare_server_args


class TestSimulatorServerArgs(unittest.TestCase):
    def test_simulator_args_parse(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"platform": {}, "predictor": {"name": "dummy"}}, f)
            f.flush()

            args = prepare_server_args(
                [
                    "--model-path",
                    "dummy",
                    "--simulator",
                    "--sim-config-path",
                    f.name,
                    "--simulator-mode",
                    "blocking",
                    "--simulator-output-dir",
                    "/tmp/sglang_simulator_test",
                    "--simulator-num-warmup",
                    "2",
                ]
            )

        self.assertTrue(args.simulator)
        self.assertEqual(args.sim_config_path, f.name)
        self.assertEqual(args.simulator_mode, "blocking")
        self.assertEqual(args.simulator_output_dir, "/tmp/sglang_simulator_test")
        self.assertEqual(args.simulator_num_warmup, 2)

    def test_simulator_requires_config_path(self):
        with self.assertRaises(ValueError):
            prepare_server_args(["--model-path", "dummy", "--simulator"])

    def test_simulator_rejects_missing_config_file(self):
        missing = str(Path(tempfile.gettempdir()) / "missing-sglang-sim-config.json")
        with self.assertRaises(ValueError):
            prepare_server_args(
                ["--model-path", "dummy", "--simulator", "--sim-config-path", missing]
            )


if __name__ == "__main__":
    unittest.main()
