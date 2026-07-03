from __future__ import annotations

from AegisKV.draft.scripts import audit_goal_completion as audit


def _by_id(payload):
    return {item["check_id"]: item for item in payload["checks"]}


def test_goal_completion_audit_keeps_strict_final_blocker_visible() -> None:
    payload = audit.run_audit()
    checks = _by_id(payload)

    assert payload["summary"]["complete"] is True
    assert payload["summary"]["blocked"] == 0
    assert payload["summary"]["partial"] == 0
    assert payload["summary"]["pass"] == 17
    assert checks["workflow_13131_config"]["status"] == audit.PASS
    assert checks["workflow_generated_request_shape"]["status"] == audit.PASS
    assert (
        "scenario=staged_13131_policy_stress_aegiskv"
        in checks["workflow_generated_request_shape"]["evidence"]
    )
    assert "generated_rows=27" in checks["workflow_generated_request_shape"]["evidence"]
    assert "workflows=3" in checks["workflow_generated_request_shape"]["evidence"]
    assert "slots=9" in checks["workflow_generated_request_shape"]["evidence"]
    assert "critical_rows=15" in checks["workflow_generated_request_shape"]["evidence"]
    assert (
        "noncritical_rows=12"
        in checks["workflow_generated_request_shape"]["evidence"]
    )
    assert (
        "long_decode_rows=6"
        in checks["workflow_generated_request_shape"]["evidence"]
    )
    assert (
        "shape_labels=['1-3-1-3-1']"
        in checks["workflow_generated_request_shape"]["evidence"]
    )
    assert "violations=[]" in checks["workflow_generated_request_shape"]["evidence"]
    assert checks["failure_replay_semantics"]["status"] == audit.PASS
    assert "config_files=12" in checks["failure_replay_semantics"]["evidence"]
    assert "request_files=12" in checks["failure_replay_semantics"]["evidence"]
    assert "normal_rows=972" in checks["failure_replay_semantics"]["evidence"]
    assert "retry_rows=540" in checks["failure_replay_semantics"]["evidence"]
    assert (
        "critical_retry_rows=540"
        in checks["failure_replay_semantics"]["evidence"]
    )
    assert (
        "noncritical_retry_rows=0"
        in checks["failure_replay_semantics"]["evidence"]
    )
    assert "soft_retry_rows=540" in checks["failure_replay_semantics"]["evidence"]
    assert "hard_retry_rows=0" in checks["failure_replay_semantics"]["evidence"]
    assert (
        "pre_failure_boundary_matches=True"
        in checks["failure_replay_semantics"]["evidence"]
    )
    assert "restore_rows=405" in checks["failure_replay_semantics"]["evidence"]
    assert "recompute_retry_rows=135" in checks["failure_replay_semantics"]["evidence"]
    assert "request_violations=[]" in checks["failure_replay_semantics"]["evidence"]
    assert checks["online_policy_sweep_results"]["status"] == audit.PASS
    assert checks["sglang_sglsim_accuracy"]["status"] == audit.PASS
    assert (
        "expected_metrics_present=True"
        in checks["sglang_sglsim_accuracy"]["evidence"]
    )
    assert "completed_pairs=[(60, 60)]" in checks["sglang_sglsim_accuracy"]["evidence"]
    assert "source_dirs_exist=True" in checks["sglang_sglsim_accuracy"]["evidence"]
    assert checks["motivation_critical_profile"]["status"] == audit.PASS
    assert "trace_shape=1-3-1-3-1" in checks["motivation_critical_profile"]["evidence"]
    assert "exact_criticality=True" in checks["motivation_critical_profile"]["evidence"]
    assert "trace_files_exist=True" in checks["motivation_critical_profile"]["evidence"]
    assert "raw_trace_rows=216" in checks["motivation_critical_profile"]["evidence"]
    assert "raw_trace_slots=9" in checks["motivation_critical_profile"]["evidence"]
    assert "raw_trace_workflows=8" in checks["motivation_critical_profile"]["evidence"]
    assert (
        "raw_trace_pattern_matches=True"
        in checks["motivation_critical_profile"]["evidence"]
    )
    assert "raw_trace_violations=[]" in checks["motivation_critical_profile"]["evidence"]
    assert (
        "runtime_workflow_groups=8"
        in checks["motivation_critical_profile"]["evidence"]
    )
    assert checks["motivation_backup_priority"]["status"] == audit.PASS
    assert "budget_points=11" in checks["motivation_backup_priority"]["evidence"]
    assert "complete_grid=True" in checks["motivation_backup_priority"]["evidence"]
    assert (
        "critical_best_mid_budgets=9/9"
        in checks["motivation_backup_priority"]["evidence"]
    )
    assert (
        "beats_noncritical_mid_budgets=9/9"
        in checks["motivation_backup_priority"]["evidence"]
    )
    assert "trace_files_exist=True" in checks["motivation_backup_priority"]["evidence"]

    strict_final = checks["strict_final_physical_dataplane"]
    assert strict_final["status"] == audit.PASS
    assert "physical_tensor_copy_rows=0" in strict_final["evidence"]
    assert "effective_physical_tensor_copy_rows=693" in strict_final["evidence"]
    assert "effective_nonfinal_restore_rows=0" in strict_final["evidence"]
    assert "physical_bridge_rows=693" in strict_final["evidence"]
    assert "physical_bridge_missing_rows=0" in strict_final["evidence"]
    assert "missing_data_plane_rows=0" in strict_final["evidence"]
    assert "missing_timing_source_rows=0" in strict_final["evidence"]
    assert (
        "dataplane_readiness_status=strict_final_ready"
        in strict_final["evidence"]
    )
    assert "host_matrix_ready=True" in strict_final["evidence"]
    assert "p0_bridge_ready=True" in strict_final["evidence"]
    assert "p0_live_audits_consistent=True" in strict_final["evidence"]
    assert "p0_live_restore_rows=693" in strict_final["evidence"]
    assert "p0_live_physical_tensor_copy_rows=0" in strict_final["evidence"]
    assert "p0_live_nonfinal_restore_rows=693" in strict_final["evidence"]
    assert "p0_live_schema_complete=True" in strict_final["evidence"]
    assert "p0_candidate_status=strict_final_candidate_ready" in strict_final["evidence"]
    assert "p0_candidate_ready=True" in strict_final["evidence"]
    assert "p0_candidate_required_rows=693" in strict_final["evidence"]
    assert "p0_candidate_accepted_rows=693" in strict_final["evidence"]
    assert "p0_candidate_missing_rows=0" in strict_final["evidence"]
    assert "p0_candidate_invalid_rows=0" in strict_final["evidence"]
    assert "p0_candidate_bridge_consistent=True" in strict_final["evidence"]
    assert "p0_candidate_generation_status=candidates_complete" in strict_final[
        "evidence"
    ]
    assert (
        "p0_candidate_generation_schema_version=p0-live-copy-event-v1"
        in strict_final["evidence"]
    )
    assert "p0_candidate_generation_event_rows=693" in strict_final["evidence"]
    assert "p0_candidate_generation_missing_sources=0" in strict_final["evidence"]
    assert "p0_candidate_generation_bridge_consistent=True" in strict_final["evidence"]
    assert "p0_required_live_copy_events=693" in strict_final["evidence"]
    assert "'live_host_h2d_copy': 375" in strict_final["evidence"]
    assert "'live_peer_p2p_copy': 318" in strict_final["evidence"]
    assert "p0_measured_live_copy_events=375" in strict_final["evidence"]
    assert "p0_live_copy_event_kinds={'live_host_h2d_copy': 375}" in strict_final[
        "evidence"
    ]
    assert "p0_live_copy_skipped_rows=318" in strict_final["evidence"]
    assert "CUDA peer access is unavailable for 0->1" in strict_final["evidence"]
    assert (
        "p0_live_p2p_plan_status=p2p_live_copy_complete"
        in strict_final["evidence"]
    )
    assert "p0_live_p2p_plan_required_rows=318" in strict_final["evidence"]
    assert "p0_live_p2p_plan_ready=True" in strict_final["evidence"]
    assert "p0_live_p2p_plan_remote=hkbugpumanode:~/aegiskv_sglsim_large" in strict_final[
        "evidence"
    ]
    assert "p0_promotion_status=promoted_strict_final_ready" in strict_final["evidence"]
    assert "p0_promotion_ready=True" in strict_final["evidence"]
    assert "p0_promotion_physical_tensor_copy_rows=693" in strict_final["evidence"]
    assert "p0_promotion_nonfinal_restore_rows=0" in strict_final["evidence"]
    assert "p0_promotion_bridge_consistent=True" in strict_final["evidence"]
    assert "sglang_physical_restore_rows=15" in strict_final["evidence"]
    assert "sglang_physical_materialized_rows=9" in strict_final["evidence"]
    assert "sglang_physical_full_rows=6" in strict_final["evidence"]
    assert "sglang_physical_partial_rows=3" in strict_final["evidence"]
    assert "sglang_physical_hidden_control_rows=6" in strict_final["evidence"]
    assert "sglang_matrix_pairs=6/6" in strict_final["evidence"]
    assert strict_final["next_step"] == "Strict-final is satisfied for P0 suites."


def test_goal_completion_audit_checks_alpha_policy_chain() -> None:
    payload = audit.run_audit()
    checks = _by_id(payload)

    mdp = checks["mdp_alpha_policy_table"]
    assert mdp["status"] == audit.PASS
    assert "iterations=50" in mdp["evidence"]
    assert "fig3b_policy_path_match=True" in mdp["evidence"]
    assert "fig3b_source_match=True" in mdp["evidence"]
    assert "runtime_selector_checks=15" in mdp["evidence"]
    assert "runtime_selector_matches_policy=True" in mdp["evidence"]
    assert "runtime_selector_source_match=True" in mdp["evidence"]

    online = checks["online_alpha_adaptation"]
    assert online["status"] == audit.PASS
    assert "online_policy_source_match=True" in online["evidence"]
    assert "critical_policy_alpha_matches_online=True" in online["evidence"]
    assert "noncritical_effective_alpha_zero=True" in online["evidence"]
    assert "raw_trace_files=4" in online["evidence"]
    assert "raw_policy_source_match=True" in online["evidence"]
    assert "raw_alpha_matches_fig3c=True" in online["evidence"]
    assert "runtime_trace_alpha_matches_raw=True" in online["evidence"]
    assert "raw_noncritical_gpu_tokens=0" in online["evidence"]


def test_goal_completion_audit_reports_completed_recovery_matrix() -> None:
    payload = audit.run_audit()
    checks = _by_id(payload)

    matrix = checks["strict_final_recovery_matrix_audit"]
    assert matrix["status"] == audit.PASS
    assert "overall_status=pass" in matrix["evidence"]
    assert "matrix_ready=True" in matrix["evidence"]
    assert "same_write_best_effort=pass:3/3" in matrix["evidence"]
    assert "same_write_wait_complete=pass:3/3" in matrix["evidence"]

    matrix_dir = (
        audit.EXPERIMENTS
        / "sglang_small"
        / "results"
        / "strict_final_recovery_matrix_audit"
    )
    slurm_payload = audit._read_json(matrix_dir / "strict_final_bound_slurm_jobs.json")
    assert slurm_payload
    assert slurm_payload["overall_status"] == "complete"

    wait_payload = audit._read_json(matrix_dir / "strict_final_wait_and_sync_summary.json")
    assert wait_payload
    assert wait_payload["status"] == "synced_and_audited"
    assert matrix["next_step"] == (
        "Use the completed Host recovery matrix as real-system recovery-path evidence."
    )


def test_goal_completion_audit_checks_beamer_without_design_flowcharts() -> None:
    payload = audit.run_audit()
    checks = _by_id(payload)

    figure_gate = checks["hpca_hybrid_figure_gate"]
    assert figure_gate["status"] == audit.PASS
    assert "hpca_hybrid_ready=25/25" in figure_gate["evidence"]
    assert "final_gate_summary_match=True" in figure_gate["evidence"]
    assert "final_gate_hybrid_summary_match=True" in figure_gate["evidence"]
    assert "fig1_trace_motivation=True" in figure_gate["evidence"]
    assert "fig3b_runtime_replay_boundary=True" in figure_gate["evidence"]
    assert "strict_ready_stems=['fig_shared_prefix_marginal']" in figure_gate["evidence"]
    assert "fig10b_only_strict_ready=True" in figure_gate["evidence"]
    beamer_compile = checks["beamer_compiles_and_references_figures"]
    assert beamer_compile["status"] == audit.PASS
    assert "evidence_status_sync=True" in beamer_compile["evidence"]
    assert "missing_evidence_status_terms=[]" in beamer_compile["evidence"]
    assert checks["beamer_no_design_flowcharts"]["status"] == audit.PASS
    assert checks["beamer_result_figure_coverage"]["status"] == audit.PASS
    assert checks["system_overview_generated_asset"]["status"] == audit.PASS
    assert "prompt_missing=[]" in checks["system_overview_generated_asset"]["evidence"]
    assert (
        "prompt_refs_tight_asset=True"
        in checks["system_overview_generated_asset"]["evidence"]
    )
    assert "image_width=1665" in checks["system_overview_generated_asset"]["evidence"]
    assert "image_height=689" in checks["system_overview_generated_asset"]["evidence"]
    assert "image_usable=True" in checks["system_overview_generated_asset"]["evidence"]
    assert (
        "forbidden_design_refs=[]"
        in checks["beamer_no_design_flowcharts"]["evidence"]
    )
    assert (
        "referenced_result_figures=24"
        in checks["beamer_result_figure_coverage"]["evidence"]
    )
    assert (
        "missing_result_figures=[]"
        in checks["beamer_result_figure_coverage"]["evidence"]
    )
    assert "design_refs=[]" in checks["beamer_result_figure_coverage"]["evidence"]
