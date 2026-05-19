from __future__ import annotations

import json
import sys
import types
from importlib.util import find_spec
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SRC_ROOT = REPO_ROOT / "benchmark" / "online" / "src"

for path in (REPO_ROOT, PYTHON_ROOT, SRC_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


if "yaml" not in sys.modules and find_spec("yaml") is None:
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(stream):
        text = stream.read() if hasattr(stream, "read") else str(stream or "")
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            parsed: dict[str, str] = {}
            for raw_line in text.splitlines():
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                parsed[key.strip()] = value.strip().strip("'\"")
            return parsed

    yaml_stub.safe_load = _safe_load
    yaml_stub.dump = lambda data, stream=None, **_kwargs: json.dumps(data)
    sys.modules["yaml"] = yaml_stub

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.suppress_debug_info = True
    litellm_stub.drop_params = True
    sys.modules["litellm"] = litellm_stub
