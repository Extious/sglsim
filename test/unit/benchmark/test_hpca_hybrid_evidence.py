from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "figures"
    / "scripts"
    / "validate_figure_evidence.py"
)


def load_validator():
    spec = importlib.util.spec_from_file_location("aegiskv_figure_validator", VALIDATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_physical_recovery_aggregate_support_accepts_restore_actions(
    tmp_path: Path,
) -> None:
    module = load_validator()
    path = tmp_path / "physical_recovery_path_aggregate.csv"
    fields = [
        "action",
        "tokens",
        "repeat_count",
        "p50_ms_mean",
        "p50_ms_ci95",
        "slurm_job_ids",
        "hosts",
        "evidence_tier",
    ]
    _write_csv(
        path,
        fields,
        [
            {
                "action": "host_restore_h2d",
                "tokens": 8192,
                "repeat_count": 3,
                "p50_ms_mean": 48.4,
                "p50_ms_ci95": 0.02,
                "slurm_job_ids": "1;2;3",
                "hosts": "hkbugpusrv14",
                "evidence_tier": "sglang-small-physical-recovery-movement-smoke",
            },
            {
                "action": "gpu_replica_restore_p2p",
                "tokens": 8192,
                "repeat_count": 3,
                "p50_ms_mean": 4.3,
                "p50_ms_ci95": 0.01,
                "slurm_job_ids": "1;2;3",
                "hosts": "hkbugpusrv14",
                "evidence_tier": "sglang-small-physical-recovery-movement-smoke",
            },
        ],
    )
    module.PHYSICAL_RECOVERY_AGGREGATE = path

    ready, summary, problems = module._check_physical_recovery_aggregate()

    assert ready is True
    assert problems == []
    assert summary["host_restore_h2d"]["repeat_count"] == "3"
    assert summary["gpu_replica_restore_p2p"]["p50_ms_mean"] == "4.3"


def test_live_audit_support_accepts_metadata_boundary(tmp_path: Path) -> None:
    module = load_validator()
    path = tmp_path / "live_physical_movement_audit.csv"
    fields = [
        "request_rows",
        "replay_rows",
        "restore_rows",
        "physical_tensor_copy_rows",
        "nonfinal_restore_rows",
        "final_ready",
        "data_planes",
        "timing_sources",
    ]
    _write_csv(
        path,
        fields,
        [
            {
                "request_rows": 8,
                "replay_rows": 4,
                "restore_rows": 4,
                "physical_tensor_copy_rows": 0,
                "nonfinal_restore_rows": 4,
                "final_ready": "False",
                "data_planes": '{"metadata_materialization": 4}',
                "timing_sources": '{"physical_recovery_path_csv": 4}',
            }
        ],
    )
    module.LIVE_MOVEMENT_AUDIT = path

    ready, summary, problems = module._check_live_movement_audit()

    assert ready is True
    assert problems == []
    assert summary["final_ready"] is False
    assert summary["data_planes"] == {"metadata_materialization": 4}
    assert summary["timing_sources"] == {"physical_recovery_path_csv": 4}


def test_hybrid_gate_marks_calibrated_figures_without_relaxing_paired_gates(
    tmp_path: Path,
) -> None:
    module = load_validator()
    measured_dir = tmp_path / "measured"
    _write_csv(
        measured_dir / "fig_recovery_path_breakdown.csv",
        ["run_id", "repeat_count"],
        [
            {
                "run_id": module.RECOVERY_ACTION_RUN_ID,
                "repeat_count": 3,
            }
        ],
    )
    module.MEASURED_DIR = measured_dir
    support = module.HybridSupport(
        physical_recovery_ready=True,
        physical_recovery_summary={},
        live_audit_ready=True,
        live_audit_summary={},
        problems=[],
    )
    fig10 = module.FigureEvidence(
        figure="Fig. 10",
        stem="fig_recovery_path_breakdown",
        kind="result_template",
        manifest_status="sglsim-preliminary",
        measured_csv="data/measured/fig_recovery_path_breakdown.csv",
        measured_status="preliminary-valid",
        evidence_tier="sglsim-agentic-online-preliminary",
        row_count=1,
        problems=[],
    )
    fig9 = module.FigureEvidence(
        figure="Fig. 9",
        stem="fig_recovery_latency",
        kind="result_template",
        manifest_status="sglsim-preliminary",
        measured_csv="data/measured/fig_recovery_latency.csv",
        measured_status="preliminary-valid",
        evidence_tier="sglsim-runtime-replay-calibrated",
        row_count=1,
        problems=[],
    )

    hybrid, _ = module.build_hybrid_evidence([fig10, fig9], support)
    by_stem = {item.stem: item for item in hybrid}

    assert by_stem["fig_recovery_path_breakdown"].hybrid_ready is True
    assert (
        by_stem["fig_recovery_path_breakdown"].hybrid_status
        == "hybrid-physical-calibrated"
    )
    assert by_stem["fig_recovery_latency"].hybrid_ready is False
    assert by_stem["fig_recovery_latency"].hybrid_status == "needs-paired-measurements"


def test_hybrid_gate_accepts_explicit_paired_anchor_sources(tmp_path: Path) -> None:
    module = load_validator()
    measured_dir = tmp_path / "measured"
    _write_csv(
        measured_dir / "fig_recovery_latency.csv",
        ["source", "repeat_count"],
        [
            {
                "source": "sglsim_agentic_online_recovery_latency_sweep",
                "repeat_count": 3,
            }
        ],
    )
    _write_csv(
        measured_dir / "fig_hostbackup_visibility.csv",
        ["source", "repeat_count"],
        [
            {
                "source": "sglsim_replay_retry_counterfactual_plus_policy_overhead_anchor",
                "repeat_count": 11,
            },
            {
                "source": "sglang_failure_visibility_pair_aggregate",
                "repeat_count": 3,
            },
        ],
    )
    _write_csv(
        measured_dir / "fig_normal_path_overhead.csv",
        ["source", "repeat_count"],
        [
            {
                "source": "sglsim_agentic_online_normal_path_overhead_sweep",
                "repeat_count": 3,
            },
            {
                "source": "sglang_dp2_overhead_pair_aggregate",
                "repeat_count": 1,
            },
        ],
    )
    _write_csv(
        measured_dir / "fig_simulator_calibration.csv",
        ["source", "repeat_count"],
        [
            {
                "source": "sglang_sglsim_normal_path_overhead_anchor",
                "repeat_count": 1,
            }
        ],
    )
    module.MEASURED_DIR = measured_dir
    support = module.HybridSupport(
        physical_recovery_ready=True,
        physical_recovery_summary={},
        live_audit_ready=True,
        live_audit_summary={},
        problems=[],
    )
    figures = [
        module.FigureEvidence(
            figure=figure,
            stem=stem,
            kind="result_template",
            manifest_status="sglsim-preliminary",
            measured_csv=f"data/measured/{stem}.csv",
            measured_status="preliminary-valid",
            evidence_tier="mixed-preliminary",
            row_count=1,
            problems=[],
        )
        for figure, stem in [
            ("Fig. 9", "fig_recovery_latency"),
            ("Fig. 11", "fig_normal_path_overhead"),
            ("Fig. 11b", "fig_hostbackup_visibility"),
            ("Fig. 15", "fig_simulator_calibration"),
        ]
    ]

    hybrid, _ = module.build_hybrid_evidence(figures, support)

    assert {item.hybrid_status for item in hybrid} == {"hybrid-paired-anchor"}
    assert all(item.hybrid_ready for item in hybrid)


def test_hybrid_gate_accepts_sglang_host_restore_materialization(
    tmp_path: Path,
) -> None:
    module = load_validator()
    measured_dir = tmp_path / "measured"
    fields = [
        "state",
        "role",
        "rows",
        "retry_cached_tokens_after_accept",
        "prefetch_io_completed_tokens",
        "max_data_pages",
        "materialized",
        "evidence_tier",
        "source",
        "run_id",
        "note",
    ]
    _write_csv(
        measured_dir / "fig_sglang_host_restore_materialization.csv",
        fields,
        [
            {
                "state": "host_frontier_not_visible_to_retry",
                "role": "control",
                "rows": 4,
                "retry_cached_tokens_after_accept": 0,
                "prefetch_io_completed_tokens": 0,
                "max_data_pages": 0,
                "materialized": "False",
                "evidence_tier": "sglang-small-physical-restore-materialization",
                "source": "sglang_physical_restore_parser_summary",
                "run_id": "physical_restore_evidence_20260702",
                "note": "hidden control rows stay hidden",
            },
            {
                "state": "partial_async_materialized",
                "role": "materialized",
                "rows": 3,
                "retry_cached_tokens_after_accept": 18643,
                "prefetch_io_completed_tokens": 18601,
                "max_data_pages": 5137,
                "materialized": "True",
                "evidence_tier": "sglang-small-physical-restore-materialization",
                "source": "sglang_physical_restore_parser_summary",
                "run_id": "physical_restore_evidence_20260702",
                "note": "best-effort rows partially materialize",
            },
            {
                "state": "full_prefetch_materialized",
                "role": "materialized",
                "rows": 4,
                "retry_cached_tokens_after_accept": 20548,
                "prefetch_io_completed_tokens": 20548,
                "max_data_pages": 5137,
                "materialized": "True",
                "evidence_tier": "sglang-small-physical-restore-materialization",
                "source": "sglang_physical_restore_parser_summary",
                "run_id": "physical_restore_evidence_20260702",
                "note": "wait-complete rows fully materialize",
            },
        ],
    )
    module.MEASURED_DIR = measured_dir
    support = module.HybridSupport(
        physical_recovery_ready=False,
        physical_recovery_summary={},
        live_audit_ready=False,
        live_audit_summary={},
        problems=[],
    )
    figure = module.FigureEvidence(
        figure="Fig. 15f",
        stem="fig_sglang_host_restore_materialization",
        kind="result_template",
        manifest_status="preliminary-measured",
        measured_csv="data/measured/fig_sglang_host_restore_materialization.csv",
        measured_status="preliminary-valid",
        evidence_tier="sglang-small-physical-restore-materialization",
        row_count=3,
        problems=[],
    )

    hybrid, _ = module.build_hybrid_evidence([figure], support)

    assert len(hybrid) == 1
    item = hybrid[0]
    assert item.hybrid_ready is True
    assert item.hybrid_status == "hybrid-sglang-host-restore"
    assert "visible Host restore rows must materialize retry cached tokens" in item.requirements
    assert any("not P0 large-sweep live physical_tensor_copy" in caveat for caveat in item.caveats)
