from pathlib import Path


def test_engine_exposes_gpu_failure_controls():
    source = (Path(__file__).resolve().parents[3] / "python/sglsim/srt/entrypoints/engine.py").read_text()
    assert "def simulate_gpu_failure" in source
    assert "def simulate_gpu_recovery" in source
