import json
import sys
import tempfile
import types
import unittest

from sglang.srt.simulator.config import SimulatorSettings
from sglang.srt.simulator.predictor import (
    build_simulator_predictor,
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang.srt.simulator.types import DataType


class DummyServerArgs:
    simulator = True
    sim_config_path = None
    simulator_mode = "offline"
    simulator_output_dir = "/tmp/sglang_simulator/output"
    simulator_num_warmup = 0


class TestSimulatorSettings(unittest.TestCase):
    def test_from_server_args_loads_json(self):
        data = {
            "platform": {"disk_read_bandwidth_gb": 8},
            "predictor": {"name": "dummy"},
            "scheduler": {"tp_size": 2},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(data, f)
            f.flush()

            args = DummyServerArgs()
            args.sim_config_path = f.name
            args.simulator_mode = "blocking"
            args.simulator_output_dir = "/tmp/sim-out"
            args.simulator_num_warmup = 3
            settings = SimulatorSettings.from_server_args(args)

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.mode, "blocking")
        self.assertEqual(settings.output_dir, "/tmp/sim-out")
        self.assertEqual(settings.num_warmup, 3)
        self.assertEqual(settings.raw_config["scheduler"]["tp_size"], 2)

    def test_disabled_settings_do_not_load_file(self):
        args = DummyServerArgs()
        args.simulator = False
        settings = SimulatorSettings.from_server_args(args)
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.raw_config, {})

    def test_resolve_scheduler_config_uses_native_types(self):
        args = DummyServerArgs()
        args.dtype = "float16"
        args.kv_cache_dtype = "auto"
        args.mem_fraction_static = 0.8
        args.max_total_tokens = 1024
        args.tp_size = 2
        args.ep_size = 1
        args.dp_size = 1
        args.pp_size = 1

        config = resolve_scheduler_config(args)

        self.assertEqual(config.data_type, DataType.FP16)
        self.assertEqual(config.kv_cache_data_type, DataType.FP16)
        self.assertEqual(config.tp_size, 2)
        self.assertEqual(config.backend_name, "sglang")

    def test_resolve_scheduler_config_auto_dtype_uses_model_config(self):
        args = DummyServerArgs()
        args.dtype = "auto"
        args.kv_cache_dtype = "auto"
        args.mem_fraction_static = 0.8
        args.max_total_tokens = 1024
        args.tp_size = 1
        args.ep_size = 1
        args.dp_size = 1
        args.pp_size = 1
        model_config = types.SimpleNamespace(dtype="torch.bfloat16")

        config = resolve_scheduler_config(args, model_config)

        self.assertEqual(config.data_type, DataType.BF16)
        self.assertEqual(config.kv_cache_data_type, DataType.BF16)

    def test_resolve_model_info_supports_mha_without_old_package(self):
        model_config = types.SimpleNamespace(
            dtype="torch.float16",
            attention_arch="MHA",
            hf_text_config={"model_type": "dummy"},
            model_path="/models/dummy",
            context_len=4096,
            hidden_size=2048,
            head_dim=128,
            num_attention_heads=16,
            num_hidden_layers=24,
            num_key_value_heads=16,
            v_head_dim=128,
            vocab_size=32000,
        )

        model = resolve_model_info(model_config)

        self.assertEqual(model.model_path, "/models/dummy")
        self.assertEqual(model.attention_arch, "MHA")
        self.assertFalse(model.is_mla())

    def test_build_simulator_predictor_uses_native_config(self):
        class DummyPredictor:
            def __init__(
                self,
                model,
                hw,
                config,
                database_path=None,
                database_mode="SILICON",
                prefill_scale_factor=1,
                decode_scale_factor=1,
            ):
                self.model = model
                self.hw = hw
                self.config = config
                self.database_path = database_path
                self.database_mode = database_mode
                self.prefill_scale_factor = prefill_scale_factor
                self.decode_scale_factor = decode_scale_factor

            def predict_requests(self, requests):
                return 0.123 + len(requests)

        fake_module = types.ModuleType("sglang.srt.simulator.predictor_aiconfigurator")
        fake_module.AIConfiguratorTimePredictor = DummyPredictor
        old_module = sys.modules.get("sglang.srt.simulator.predictor_aiconfigurator")
        sys.modules["sglang.srt.simulator.predictor_aiconfigurator"] = fake_module
        try:
            data = {
                "platform": {
                    "accelerator": {
                        "name": "unit-test-accel",
                        "vendor": "test",
                        "hbm_capacity_gb": 80,
                        "hbm_bandwidth_gb": 1000,
                    },
                },
                "predictor": {
                    "name": "aiconfigurator",
                    "database_mode": "EMPIRICAL",
                    "prefill_scale_factor": 2,
                    "decode_scale_factor": 3,
                },
                "scheduler": {
                    "tp_size": 4,
                    "data_type": "BF16",
                    "kv_cache_data_type": "FP8",
                    "backend_version": "test-version",
                },
            }
            with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
                json.dump(data, f)
                f.flush()

                args = DummyServerArgs()
                args.sim_config_path = f.name
                args.dtype = "float16"
                args.kv_cache_dtype = "auto"
                args.mem_fraction_static = 0.8
                args.max_total_tokens = 1024
                args.tp_size = 1
                args.ep_size = 1
                args.dp_size = 1
                args.pp_size = 1
                model_config = types.SimpleNamespace(
                    dtype="torch.float16",
                    attention_arch="MHA",
                    hf_text_config={},
                    model_path="/models/dummy",
                    context_len=4096,
                    hidden_size=2048,
                    head_dim=128,
                    num_attention_heads=16,
                    num_hidden_layers=24,
                    num_key_value_heads=16,
                    v_head_dim=128,
                    vocab_size=32000,
                )

                predictor = build_simulator_predictor(args, model_config)
        finally:
            if old_module is None:
                sys.modules.pop("sglang.srt.simulator.predictor_aiconfigurator", None)
            else:
                sys.modules["sglang.srt.simulator.predictor_aiconfigurator"] = old_module

        self.assertIsNotNone(predictor)
        self.assertEqual(predictor.hw.name, "unit-test-accel")
        self.assertEqual(predictor.config.tp_size, 4)
        self.assertEqual(predictor.config.data_type, DataType.BF16)
        self.assertEqual(predictor.config.kv_cache_data_type, DataType.FP8)
        self.assertEqual(predictor.predict_requests([{"extend_length": 1}]), 1.123)


if __name__ == "__main__":
    unittest.main()
