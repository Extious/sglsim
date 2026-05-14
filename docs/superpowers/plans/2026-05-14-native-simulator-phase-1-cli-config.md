# Native Simulator Phase 1 CLI Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class simulator CLI options and a small validated config facade without changing runtime execution.

**Architecture:** `ServerArgs` owns user-facing flags. `sglang.srt.simulator.config` converts those flags and the existing JSON config into a typed settings object used by later phases.

**Tech Stack:** Python dataclasses, argparse, pytest.

---

## Files

- Create: `python/sglang/srt/simulator/__init__.py`
- Create: `python/sglang/srt/simulator/config.py`
- Modify: `python/sglang/srt/server_args.py`
- Test: `test/registered/unit/simulator/test_server_args.py`
- Test: `test/registered/unit/simulator/test_simulator_config.py`

## Task 1: Add CLI Fields

- [ ] **Step 1: Write failing parser tests**

Create `test/registered/unit/simulator/test_server_args.py`:

```python
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
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_server_args.py -q
```

Expected: FAIL because `--simulator` is not recognized.

- [ ] **Step 3: Modify `server_args.py` fields**

In `python/sglang/srt/server_args.py`, add near the existing debug/forward hook area:

```python
    # For simulator mode
    simulator: bool = False
    sim_config_path: Optional[str] = None
    simulator_mode: Literal["offline", "blocking"] = "offline"
    simulator_output_dir: str = "/tmp/sglang_simulator/output"
    simulator_num_warmup: int = 0
```

Add `Literal` to imports if it is not already imported.

- [ ] **Step 4: Modify `server_args.py` CLI parser**

Near the `--forward-hooks` parser block, add:

```python
        # For simulator mode
        parser.add_argument(
            "--simulator",
            action="store_true",
            default=ServerArgs.simulator,
            help="Enable native SGLang simulator mode.",
        )
        parser.add_argument(
            "--sim-config-path",
            type=str,
            default=ServerArgs.sim_config_path,
            help="Path to simulator JSON config.",
        )
        parser.add_argument(
            "--simulator-mode",
            type=str,
            choices=["offline", "blocking"],
            default=ServerArgs.simulator_mode,
            help="Simulator execution mode. offline advances logical time; blocking sleeps for predicted latency.",
        )
        parser.add_argument(
            "--simulator-output-dir",
            type=str,
            default=ServerArgs.simulator_output_dir,
            help="Directory for simulator metrics output.",
        )
        parser.add_argument(
            "--simulator-num-warmup",
            type=int,
            default=ServerArgs.simulator_num_warmup,
            help="Number of initial simulator requests to exclude from metrics.",
        )
```

- [ ] **Step 5: Add post-init validation**

In `ServerArgs.__post_init__()`, after SSL validation and before the dummy-model short-circuit, add:

```python
        if self.simulator:
            if not self.sim_config_path:
                raise ValueError("--sim-config-path is required when --simulator is enabled.")
            if not os.path.exists(self.sim_config_path):
                raise ValueError(
                    f"--sim-config-path does not exist: {self.sim_config_path}"
                )
            self.simulator_mode = self.simulator_mode.lower()
```

Ensure `os` is already imported; if not, add it.

- [ ] **Step 6: Run parser tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_server_args.py -q
```

Expected: PASS.

## Task 2: Add Simulator Config Facade

- [ ] **Step 1: Write failing config tests**

Create `test/registered/unit/simulator/test_simulator_config.py`:

```python
import json
import tempfile
import unittest

from sglang.srt.simulator.config import SimulatorSettings


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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_simulator_config.py -q
```

Expected: FAIL because `sglang.srt.simulator.config` does not exist.

- [ ] **Step 3: Create simulator package**

Create `python/sglang/srt/simulator/__init__.py`:

```python
"""Native SGLang simulator runtime."""
```

- [ ] **Step 4: Implement `SimulatorSettings`**

Create `python/sglang/srt/simulator/config.py`:

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SimulatorSettings:
    enabled: bool
    config_path: str | None
    mode: str
    output_dir: str
    num_warmup: int
    raw_config: Dict[str, Any]

    @classmethod
    def from_server_args(cls, server_args) -> "SimulatorSettings":
        if not getattr(server_args, "simulator", False):
            return cls(
                enabled=False,
                config_path=None,
                mode="offline",
                output_dir="/tmp/sglang_simulator/output",
                num_warmup=0,
                raw_config={},
            )

        config_path = getattr(server_args, "sim_config_path", None)
        if not config_path:
            raise ValueError("--sim-config-path is required when simulator mode is enabled.")
        if not os.path.exists(config_path):
            raise ValueError(f"Simulator config path does not exist: {config_path}")

        with open(config_path) as f:
            raw_config = json.load(f)

        mode = getattr(server_args, "simulator_mode", "offline").lower()
        if mode not in ("offline", "blocking"):
            raise ValueError(f"Unsupported simulator mode: {mode}")

        return cls(
            enabled=True,
            config_path=config_path,
            mode=mode,
            output_dir=getattr(
                server_args,
                "simulator_output_dir",
                "/tmp/sglang_simulator/output",
            ),
            num_warmup=int(getattr(server_args, "simulator_num_warmup", 0)),
            raw_config=raw_config,
        )
```

- [ ] **Step 5: Run config tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_simulator_config.py -q
```

Expected: PASS.

## Task 3: Help Smoke Test

- [ ] **Step 1: Run launch server help**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python python -m sglang.launch_server --help | grep simulator
```

Expected: output includes `--simulator`, `--sim-config-path`, `--simulator-mode`, `--simulator-output-dir`, and `--simulator-num-warmup`.

- [ ] **Step 2: Commit phase 1**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
git add python/sglang/srt/server_args.py python/sglang/srt/simulator test/registered/unit/simulator
git commit -m "feat: add native simulator cli config"
```
