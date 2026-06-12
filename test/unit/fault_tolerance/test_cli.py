from pathlib import Path

from sglsim.fault_tolerance.cli import main


def test_cli_compare_writes_outputs(tmp_path: Path):
    main(
        [
            "--config-dir",
            "examples/fault_tolerance/baseline-fixed",
            "--output-dir",
            str(tmp_path),
            "--compare",
        ]
    )

    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "baseline" / "metrics.json").exists()
    assert (tmp_path / "gpu_backup" / "metrics.json").exists()
    assert (tmp_path / "host_backup" / "metrics.json").exists()

