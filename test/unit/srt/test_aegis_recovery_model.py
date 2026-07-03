from __future__ import annotations

import csv

import pytest

from sglsim.srt.simulation import recovery_model
from sglsim.srt.simulation.recovery_model import (
    ENV_PHYSICAL_COPY_CSV,
    ENV_RECOVERY_MODEL,
    aegis_restore_data_plane,
    aegis_restore_overhead_ms,
    aegis_restore_overhead_s,
    aegis_restore_timing_source,
)


def test_default_recovery_model_matches_existing_coefficients(monkeypatch):
    monkeypatch.delenv(ENV_RECOVERY_MODEL, raising=False)
    monkeypatch.delenv(ENV_PHYSICAL_COPY_CSV, raising=False)

    assert aegis_restore_overhead_ms(
        action="recompute", token_count=100, restored_tokens=0
    ) == pytest.approx(5.08)
    assert aegis_restore_overhead_ms(
        action="host_restore", token_count=100, restored_tokens=80
    ) == pytest.approx(7.5 + 80 * 0.0036 + 20 * 0.0108)
    assert aegis_restore_overhead_s(
        action="gpu_replica", token_count=100, restored_tokens=12
    ) == pytest.approx((2.4 + 12 * 0.0019 + 88 * 0.0108) / 1000.0)


def test_physical_copy_model_uses_interpolated_h2d_and_p2p(monkeypatch, tmp_path):
    path = tmp_path / "physical_copy.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "tokens", "p50_ms_mean"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"path": "h2d_pinned", "tokens": 100, "p50_ms_mean": 1.0},
                {"path": "h2d_pinned", "tokens": 200, "p50_ms_mean": 3.0},
                {"path": "p2p_gpu0_to_gpu1", "tokens": 100, "p50_ms_mean": 0.5},
                {"path": "p2p_gpu0_to_gpu1", "tokens": 200, "p50_ms_mean": 1.5},
            ]
        )
    recovery_model._load_physical_copy_table.cache_clear()
    monkeypatch.setenv(ENV_RECOVERY_MODEL, "physical_copy")
    monkeypatch.setenv(ENV_PHYSICAL_COPY_CSV, str(path))

    assert aegis_restore_overhead_ms(
        action="host_restore", token_count=300, restored_tokens=150
    ) == pytest.approx(7.5 + 2.0 + 150 * 0.0108)
    assert aegis_restore_overhead_ms(
        action="gpu_replica", token_count=300, restored_tokens=150
    ) == pytest.approx(2.4 + 1.0 + 150 * 0.0108)


def test_physical_copy_model_accepts_recovery_action_csv(monkeypatch, tmp_path):
    path = tmp_path / "physical_recovery_path.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["action", "copy_path", "tokens", "p50_ms_mean"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "action": "host_backup_write_d2h",
                    "copy_path": "d2h_pinned",
                    "tokens": 100,
                    "p50_ms_mean": 99.0,
                },
                {
                    "action": "host_restore_h2d",
                    "copy_path": "h2d_pinned",
                    "tokens": 100,
                    "p50_ms_mean": 1.0,
                },
                {
                    "action": "gpu_replica_restore_p2p",
                    "copy_path": "p2p_gpu0_to_gpu1",
                    "tokens": 100,
                    "p50_ms_mean": 0.5,
                },
            ]
        )
    recovery_model._load_physical_copy_table.cache_clear()
    monkeypatch.setenv(ENV_RECOVERY_MODEL, "physical_copy")
    monkeypatch.setenv(ENV_PHYSICAL_COPY_CSV, str(path))

    assert aegis_restore_overhead_ms(
        action="host_restore", token_count=120, restored_tokens=100
    ) == pytest.approx(7.5 + 1.0 + 20 * 0.0108)
    assert aegis_restore_overhead_ms(
        action="gpu_replica", token_count=120, restored_tokens=100
    ) == pytest.approx(2.4 + 0.5 + 20 * 0.0108)
    recovery_model._physical_copy_table_kind.cache_clear()
    assert aegis_restore_timing_source() == "physical_recovery_path_csv"


def test_physical_copy_model_reports_raw_copy_timing_source(monkeypatch, tmp_path):
    path = tmp_path / "physical_copy.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "tokens", "p50_ms_mean"])
        writer.writeheader()
        writer.writerow(
            {"path": "h2d_pinned", "tokens": 100, "p50_ms_mean": 1.0}
        )
    recovery_model._physical_copy_table_kind.cache_clear()
    monkeypatch.setenv(ENV_RECOVERY_MODEL, "physical_copy")
    monkeypatch.setenv(ENV_PHYSICAL_COPY_CSV, str(path))

    assert aegis_restore_timing_source() == "physical_copy_csv"


def test_physical_copy_model_does_not_change_restore_data_plane(
    monkeypatch, tmp_path
):
    path = tmp_path / "physical_copy.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "tokens", "p50_ms_mean"])
        writer.writeheader()
        writer.writerow(
            {"path": "p2p_gpu0_to_gpu1", "tokens": 100, "p50_ms_mean": 1.0}
        )
    monkeypatch.setenv(ENV_RECOVERY_MODEL, "physical_copy")
    monkeypatch.setenv(ENV_PHYSICAL_COPY_CSV, str(path))

    assert aegis_restore_timing_source() == "physical_copy_csv"
    assert aegis_restore_data_plane("host_restore") == "metadata_materialization"
    assert aegis_restore_data_plane("gpu_replica") == "metadata_materialization"
    assert aegis_restore_data_plane("recompute") == "recompute"


def test_physical_copy_model_falls_back_when_csv_is_missing(monkeypatch, tmp_path):
    recovery_model._load_physical_copy_table.cache_clear()
    monkeypatch.setenv(ENV_RECOVERY_MODEL, "physical_copy")
    monkeypatch.setenv(ENV_PHYSICAL_COPY_CSV, str(tmp_path / "missing.csv"))

    assert aegis_restore_overhead_ms(
        action="host_restore", token_count=100, restored_tokens=80
    ) == pytest.approx(7.5 + 80 * 0.0036 + 20 * 0.0108)
