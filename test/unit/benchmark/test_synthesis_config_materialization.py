from pathlib import Path

from benchmark.synthesis_requests.utils import load_config_dir, write_simulator_config


def test_real_trace_profile_materializes_aligned_server_settings(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    client, server = load_config_dir(
        repo_root / "benchmark" / "synthesis_requests" / "config"
    )

    path = write_simulator_config(
        tmp_path,
        client=client,
        server=server,
        timestamp="20260607_120000",
    )
    text = path.read_text(encoding="utf-8")

    assert '"experiment_name": "synthesis-a100-dp2-hicache-fa3"' in text
    assert '"page_size": 1' in text
    assert '"enable_hierarchical_cache": true' in text
    assert '"forward_scale": 1.0' in text
    assert '"dp_size": 2' in text
