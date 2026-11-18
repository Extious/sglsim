from __future__ import annotations

from sglsim.srt.managers.aegis_l1_l2 import (
    AegisBackupPlan,
    AegisL1L2Config,
    AegisRankBackupManager,
    project_alpha_beta,
)


def test_project_alpha_beta_enforces_beta_ge_alpha() -> None:
    alpha, beta = project_alpha_beta(0.40, 0.20, normal_pressure=0.0)
    assert beta >= alpha
    assert (alpha, beta) == (0.40, 0.40)


def test_project_alpha_beta_shrinks_both_under_high_pressure() -> None:
    alpha, beta = project_alpha_beta(
        0.50,
        0.60,
        normal_pressure=0.80,
        previous_alpha=0.50,
        previous_beta=0.60,
    )
    assert beta >= alpha
    assert alpha < 0.50
    assert beta < 0.60


def test_project_alpha_beta_blocks_expansion_under_pressure() -> None:
    alpha, beta = project_alpha_beta(
        0.80,
        0.90,
        normal_pressure=0.40,
        previous_alpha=0.20,
        previous_beta=0.30,
    )
    assert alpha <= 0.20
    assert beta <= 0.30
    assert beta >= alpha


def test_algorithm3_keeps_beta_above_alpha_and_shrinks_under_pressure() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=4096,
            l2_total_tokens=4096,
            initial_alpha=0.40,
            initial_beta=0.20,
            chunk_tokens=256,
            window_requests=1,
            demand_ewma=1.0,
            publish_smoothing=1.0,
            hysteresis=0.0,
            burst_reserve_tokens=0,
            safety_margin=0.0,
        ),
    )
    ack = manager.place(
        AegisBackupPlan(
            rid="pressure",
            backup_key="pressure",
            source_dp_rank=0,
            target_dp_rank=1,
            token_count=256,
            critical_probability=0.9,
            slack_ms=0.0,
            normal_l1_pressure=0.50,
            normal_l2_pressure=0.50,
            allowed_tiers=("l1", "l2"),
        )
    )
    assert ack.window_update is not None
    assert ack.window_update["new_beta"] >= ack.window_update["new_alpha"]
    assert ack.window_update["new_alpha"] <= 0.40
    assert ack.window_update["new_beta"] <= 0.40
