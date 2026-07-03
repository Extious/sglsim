from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "scripts"
    / "report_strict_final_dataplane_readiness.py"
)
BRIDGE_DIGEST_FIELDS = (
    "suite_label",
    "request_csv",
    "rid",
    "restore_action",
    "restored_tokens",
    "physical_action",
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_strict_final_dataplane_readiness",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_p0_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "data_planes": json.dumps({"metadata_materialization": 405}),
            "final_ready": "False",
            "missing_data_plane_rows": "0",
            "missing_timing_source_rows": "0",
            "nonfinal_restore_rows": "405",
            "physical_tensor_copy_rows": "0",
            "restore_rows": "405",
            "schema_complete": "True",
            "timing_sources": json.dumps({"physical_recovery_path_csv": 405}),
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_bridge_csv(path: Path, *, eligible: int = 0, missing: int = 405) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "bridge_eligible_rows": str(eligible),
            "bridge_missing_rows": str(missing),
            "bridge_ready": str(eligible > 0 and missing == 0),
            "evidence_tiers": json.dumps(
                ["sglang-small-physical-recovery-movement-smoke"]
            ),
            "physical_actions": json.dumps({"host_restore_h2d": eligible}),
            "restore_actions": json.dumps({"host_restore": eligible}),
            "restore_rows": str(eligible + missing),
            "suite_label": "p0",
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bridge_required_digest(rows: list[dict[str, str]]) -> str:
    digest_rows = []
    for row in rows:
        if str(row.get("bridge_eligible")).strip().lower() not in {"1", "true", "yes"}:
            continue
        key = [
            str(row.get("suite_label") or "").strip(),
            str(row.get("request_csv") or "").strip(),
            str(row.get("rid") or "").strip(),
        ]
        digest_row = {
            field: str(row.get(field) or "").strip()
            for field in BRIDGE_DIGEST_FIELDS
        }
        digest_row["key"] = key
        digest_rows.append(digest_row)
    encoded = json.dumps(
        sorted(digest_rows, key=lambda item: item["key"]),
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_bridge_rows_csv(path: Path, *, rows: int = 405) -> str:
    fields = [
        "suite_label",
        "request_csv",
        "rid",
        "restore_action",
        "restored_tokens",
        "physical_action",
        "bridge_eligible",
    ]
    data = [
        {
            "suite_label": "p0",
            "request_csv": "/tmp/a/request_details.csv",
            "rid": f"r{index}",
            "restore_action": "host_restore",
            "restored_tokens": "128",
            "physical_action": "host_restore_h2d",
            "bridge_eligible": "true",
        }
        for index in range(rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
    return bridge_required_digest(data)


def write_physical_restore_csv(root: Path) -> None:
    path = root / "physical_restore_runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "slurm_job_id": "100",
            "physical_restore_state": "host_frontier_not_visible_to_retry",
            "max_data_pages": "0",
            "prefetch_io_completed_tokens_sum": "0",
            "retry_cached_tokens_after_accept_total": "0",
        },
        {
            "slurm_job_id": "101",
            "physical_restore_state": "full_prefetch_materialized",
            "max_data_pages": "5137",
            "prefetch_io_completed_tokens_sum": "5137",
            "retry_cached_tokens_after_accept_total": "5137",
        },
        {
            "slurm_job_id": "102",
            "physical_restore_state": "partial_async_materialized",
            "max_data_pages": "5137",
            "prefetch_io_completed_tokens_sum": "2816",
            "retry_cached_tokens_after_accept_total": "2830",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_p0_live_audits_json(path: Path, *, restore_rows: int = 405) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": "write_p0_live_movement_audits",
                "summary": {
                    "final_ready": False,
                    "missing_data_plane_rows": 0,
                    "missing_timing_source_rows": 0,
                    "nonfinal_restore_rows": restore_rows,
                    "physical_tensor_copy_rows": 0,
                    "restore_rows": restore_rows,
                    "schema_complete": True,
                    "suite_count": 1,
                },
                "suites": [
                    {
                        "figures": "Fig. 8/9/10",
                        "final_ready": False,
                        "label": "p0",
                        "nonfinal_restore_rows": restore_rows,
                        "physical_tensor_copy_rows": 0,
                        "restore_rows": restore_rows,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def write_candidate_audit_json(
    path: Path,
    *,
    accepted_rows: int = 0,
    missing_rows: int = 405,
    ready: bool = False,
    bridge_digest: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accepted_rows": accepted_rows,
                "bridge_required_digest": bridge_digest,
                "bridge_rows": "/tmp/bridge_rows.csv",
                "candidate_csv": "/tmp/candidates.csv",
                "candidate_rows": accepted_rows,
                "extra_rows": 0,
                "invalid_rows": [],
                "missing_rows": missing_rows,
                "required_rows": accepted_rows + missing_rows,
                "source": "validate_p0_physical_tensor_copy_candidates",
                "status": "strict_final_candidate_ready" if ready else "not_ready",
                "strict_final_candidate_ready": ready,
            }
        ),
        encoding="utf-8",
    )


def write_candidate_generation_audit_json(
    path: Path,
    *,
    status: str = "missing_event_sources",
    required_rows: int = 405,
    bridge_digest: str = "",
    event_schema_version: str = "p0-live-copy-event-v1",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accepted_rows": 0,
                "bridge_required_digest": bridge_digest,
                "bridge_rows": "/tmp/bridge_rows.csv",
                "event_schema_version": event_schema_version,
                "event_rows": 0,
                "event_sources": [],
                "missing_event_sources": ["/tmp/missing-events.csv"],
                "missing_rows": required_rows,
                "rejected_event_rows": 0,
                "required_rows": required_rows,
                "source": "build_p0_physical_tensor_copy_candidates",
                "status": status,
                "strict_final_candidate_ready": False,
            }
        ),
        encoding="utf-8",
    )


def write_live_copy_events_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "bytes_per_token": 147456,
                "copy_backend": "torch.cuda.copy_",
                "environment": {
                    "cuda_available": True,
                    "device_count": 2,
                    "peer_access": {"0->1": False, "1->0": False},
                    "torch_version": "2.3.1+cu121",
                },
                "event_kind_filter": "all",
                "event_rows": 375,
                "event_rows_by_kind": {"live_host_h2d_copy": 375},
                "latency_by_kind": {
                    "live_host_h2d_copy": {
                        "mean_ms": 11.67,
                        "p95_ms": 21.35,
                        "rows": 375,
                    }
                },
                "measurement_repeats": 1,
                "output_csv": "/tmp/events.live.csv",
                "required_rows": 693,
                "selected_rows": 693,
                "skipped_rows": 318,
                "skipped_rows_by_reason": {
                    "CUDA peer access is unavailable for 0->1": 318
                },
                "strict_final_candidate_ready": False,
                "warmup_repeats": 0,
            }
        ),
        encoding="utf-8",
    )


def write_p2p_collection_plan_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "commands": {"submit": "ssh gpubox sbatch p2p.sbatch"},
                "p2p_live_copy_event_rows": 0,
                "p2p_live_copy_skipped_rows": 0,
                "p2p_ready": False,
                "remote_host": "gpubox",
                "remote_workdir": "~/p0",
                "required_p2p_rows": 318,
                "required_p2p_tokens": 112684,
                "sbatch": "slurm/aegiskv_p0_live_p2p_copy_events.sbatch",
                "source": "plan_p0_live_p2p_collection",
                "status": "ready_to_submit_p2p_live_copy_job",
            }
        ),
        encoding="utf-8",
    )


def write_promotion_json(path: Path, *, ready: bool = True, rows: int = 405) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "host_h2d_candidate_rows": rows,
                "nonfinal_restore_rows": 0 if ready else rows,
                "peer_p2p_candidate_rows": 0,
                "physical_tensor_copy_rows": rows if ready else 0,
                "problems": [] if ready else ["not_ready"],
                "promoted_rows": 1,
                "promotion_ready": ready,
                "restore_rows": rows,
                "source": "promote_p0_physical_tensor_copy_candidates",
                "status": "promoted_strict_final_ready" if ready else "not_ready",
            }
        ),
        encoding="utf-8",
    )


def test_report_summarizes_live_copy_event_template(tmp_path: Path) -> None:
    module = load_module()
    template_csv = tmp_path / "events.template.csv"
    required_csv = tmp_path / "events.required.csv"
    schema_md = tmp_path / "event_schema.md"
    fields = [
        "suite_label",
        "request_csv",
        "rid",
        "event_schema_version",
        "event_kind",
        "aegis_replay_restore_data_plane",
        "physical_copy_verified",
        "copy_latency_ms",
        "copy_bytes",
    ]
    with template_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
    with required_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "suite_label",
                "request_csv",
                "rid",
                "event_kind",
                "restore_action",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "suite_label": "p0",
                "request_csv": "/tmp/a/request_details.csv",
                "rid": "r0",
                "event_kind": "live_host_h2d_copy",
                "restore_action": "host_restore",
            }
        )
        writer.writerow(
            {
                "suite_label": "p0",
                "request_csv": "/tmp/a/request_details.csv",
                "rid": "r1",
                "event_kind": "live_peer_p2p_copy",
                "restore_action": "gpu_replica",
            }
        )
    schema_md.write_text("schema\n", encoding="utf-8")

    summary = module.summarize_event_template(template_csv, schema_md, required_csv)

    assert summary["csv_exists"] is True
    assert summary["schema_md_exists"] is True
    assert summary["event_schema_version"] == "p0-live-copy-event-v1"
    assert summary["template_rows"] == 0
    assert summary["fields"] == fields
    assert summary["required_csv_exists"] is True
    assert summary["required_rows"] == 2
    assert summary["required_event_kinds"] == {
        "live_host_h2d_copy": 1,
        "live_peer_p2p_copy": 1,
    }
    assert summary["required_restore_actions"] == {
        "gpu_replica": 1,
        "host_restore": 1,
    }


def test_report_summarizes_live_copy_collection(tmp_path: Path) -> None:
    module = load_module()
    live_json = tmp_path / "events.live.json"
    write_live_copy_events_json(live_json)

    summary = module.summarize_live_copy_events(json.loads(live_json.read_text()))

    assert summary["exists"] is True
    assert summary["event_rows"] == 375
    assert summary["event_rows_by_kind"] == {"live_host_h2d_copy": 375}
    assert summary["skipped_rows"] == 318
    assert summary["skipped_rows_by_reason"] == {
        "CUDA peer access is unavailable for 0->1": 318
    }
    assert summary["peer_access"] == {"0->1": False, "1->0": False}
    assert summary["bytes_per_token"] == 147456


def test_report_summarizes_p2p_collection_plan(tmp_path: Path) -> None:
    module = load_module()
    plan_json = tmp_path / "p2p-plan.json"
    write_p2p_collection_plan_json(plan_json)

    summary = module.summarize_p2p_collection_plan(
        json.loads(plan_json.read_text())
    )

    assert summary["exists"] is True
    assert summary["status"] == "ready_to_submit_p2p_live_copy_job"
    assert summary["required_p2p_rows"] == 318
    assert summary["required_p2p_tokens"] == 112684
    assert summary["remote_host"] == "gpubox"
    assert summary["commands"]["submit"] == "ssh gpubox sbatch p2p.sbatch"


def test_report_summarizes_p0_promotion(tmp_path: Path) -> None:
    module = load_module()
    promotion_json = tmp_path / "promotion.json"
    write_promotion_json(promotion_json)

    summary = module.summarize_p0_promotion(
        json.loads(promotion_json.read_text())
    )

    assert summary["exists"] is True
    assert summary["status"] == "promoted_strict_final_ready"
    assert summary["promotion_ready"] is True
    assert summary["restore_rows"] == 405
    assert summary["physical_tensor_copy_rows"] == 405


def test_report_keeps_p0_metadata_rows_from_passing_strict_final(tmp_path: Path) -> None:
    module = load_module()
    p0_csv = tmp_path / "p0.csv"
    bridge_csv = tmp_path / "bridge.csv"
    bridge_rows_csv = tmp_path / "bridge_rows.csv"
    matrix_json = tmp_path / "matrix.json"
    slurm_json = tmp_path / "slurm.json"
    p0_live_json = tmp_path / "p0_live.json"
    candidate_json = tmp_path / "candidate.json"
    candidate_generation_json = tmp_path / "candidate_generation.json"
    physical_restore_root = tmp_path / "physical_restore"
    write_p0_csv(p0_csv)
    write_bridge_csv(bridge_csv)
    write_bridge_rows_csv(bridge_rows_csv, rows=0)
    write_physical_restore_csv(physical_restore_root)
    matrix_json.write_text(
        json.dumps(
            {
                "case_rows": [
                    {
                        "accepted_pairs": 3,
                        "case_id": "same_write_best_effort",
                        "observed_pairs": 3,
                        "required_repeats": 3,
                        "status": "pass",
                    },
                    {
                        "accepted_pairs": 1,
                        "case_id": "same_write_wait_complete",
                        "observed_pairs": 1,
                        "required_repeats": 3,
                        "status": "partial",
                    },
                ],
                "matrix_ready": False,
                "overall_status": "partial",
                "physical_restore_csv_count": 1,
                "physical_restore_rows": 11,
            }
        ),
        encoding="utf-8",
    )
    slurm_json.write_text(
        json.dumps(
            {
                "checked_at": "2026-07-02T19:55:23+08:00",
                "overall_status": "running",
                "status_counts": {
                    "completed": 0,
                    "failed": 0,
                    "pending": 3,
                    "running": 1,
                },
                "total_bound_jobs": 4,
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_payload(
        p0_csv=p0_csv,
        p0_live_audits_json=p0_live_json,
        p0_candidate_audit_json=candidate_json,
        p0_candidate_generation_audit_json=candidate_generation_json,
        p0_live_copy_events_json=tmp_path / "live_copy_events.json",
        p0_live_p2p_plan_json=tmp_path / "p2p_plan.json",
        p0_promotion_json=tmp_path / "promotion.json",
        bridge_csv=bridge_csv,
        bridge_rows_csv=bridge_rows_csv,
        matrix_json=matrix_json,
        slurm_json=slurm_json,
        physical_restore_root=physical_restore_root,
    )

    assert payload["strict_final_ready"] is False
    assert payload["host_matrix_ready"] is False
    assert payload["status"] == "waiting_on_matrix_and_p0_physical_bridge"
    assert payload["p0"]["restore_rows"] == 405
    assert payload["p0"]["physical_tensor_copy_rows"] == 0
    assert payload["p0"]["data_planes"] == {"metadata_materialization": 405}
    assert payload["p0_live_audits"]["exists"] is False
    assert payload["p0_live_audits_consistent"] is True
    assert payload["p0_physical_tensor_copy_candidate_audit"]["exists"] is False
    assert payload["p0_candidate_audit_bridge_consistent"] is True
    assert payload["p0_physical_tensor_copy_candidate_generation_audit"][
        "exists"
    ] is False
    assert payload["p0_candidate_generation_bridge_consistent"] is True
    assert payload["bridge"]["bridge_ready"] is False
    assert payload["matrix"]["accepted_pairs"] == 4
    assert payload["matrix"]["required_pairs"] == 6
    assert payload["slurm"]["running"] == 1
    assert payload["sglang_physical_restore"]["rows"] == 3
    assert payload["sglang_physical_restore"]["materialized_rows"] == 2
    assert payload["sglang_physical_restore"]["full_prefetch_materialized_rows"] == 1
    assert payload["sglang_physical_restore"]["partial_async_materialized_rows"] == 1
    assert payload["sglang_physical_restore"]["hidden_control_rows"] == 1
    assert payload["sglang_physical_restore"]["retry_cached_tokens_after_accept"] == 7967
    assert payload["sglang_physical_restore"]["ready_for_host_matrix"] is True
    assert "replace or bridge SGLSim P0" in " ".join(
        payload["next_required_actions"]
    )


def test_report_accepts_bridge_without_passing_strict_final(tmp_path: Path) -> None:
    module = load_module()
    p0_csv = tmp_path / "p0.csv"
    bridge_csv = tmp_path / "bridge.csv"
    bridge_rows_csv = tmp_path / "bridge_rows.csv"
    matrix_json = tmp_path / "matrix.json"
    slurm_json = tmp_path / "slurm.json"
    p0_live_json = tmp_path / "p0_live.json"
    candidate_json = tmp_path / "candidate.json"
    candidate_generation_json = tmp_path / "candidate_generation.json"
    physical_restore_root = tmp_path / "physical_restore"
    write_p0_csv(p0_csv)
    write_p0_live_audits_json(p0_live_json)
    write_bridge_csv(bridge_csv, eligible=405, missing=0)
    write_bridge_rows_csv(bridge_rows_csv, rows=405)
    write_physical_restore_csv(physical_restore_root)
    matrix_json.write_text(
        json.dumps(
            {
                "case_rows": [
                    {
                        "accepted_pairs": 4,
                        "case_id": "same_write_best_effort",
                        "observed_pairs": 4,
                        "required_repeats": 4,
                        "status": "pass",
                    }
                ],
                "matrix_ready": False,
                "overall_status": "partial",
                "physical_restore_csv_count": 1,
                "physical_restore_rows": 11,
            }
        ),
        encoding="utf-8",
    )
    slurm_json.write_text(
        json.dumps(
            {
                "checked_at": "2026-07-02T19:55:23+08:00",
                "overall_status": "pending",
                "status_counts": {
                    "completed": 0,
                    "failed": 0,
                    "pending": 4,
                    "running": 0,
                },
                "total_bound_jobs": 4,
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_payload(
        p0_csv=p0_csv,
        p0_live_audits_json=p0_live_json,
        p0_candidate_audit_json=candidate_json,
        p0_candidate_generation_audit_json=candidate_generation_json,
        p0_live_copy_events_json=tmp_path / "live_copy_events.json",
        p0_live_p2p_plan_json=tmp_path / "p2p_plan.json",
        p0_promotion_json=tmp_path / "promotion.json",
        bridge_csv=bridge_csv,
        bridge_rows_csv=bridge_rows_csv,
        matrix_json=matrix_json,
        slurm_json=slurm_json,
        physical_restore_root=physical_restore_root,
    )

    assert payload["strict_final_ready"] is False
    assert payload["p0"]["physical_tensor_copy_rows"] == 0
    assert payload["p0_live_audits"]["exists"] is True
    assert payload["p0_live_audits"]["restore_rows"] == 405
    assert payload["p0_live_audits_consistent"] is True
    assert payload["bridge"]["bridge_ready"] is True
    assert payload["bridge"]["bridge_eligible_rows"] == 405
    assert payload["p0_physical_tensor_copy_candidate_audit"]["status"] == (
        "missing_audit_json"
    )
    assert payload["p0_candidate_audit_bridge_consistent"] is True
    assert payload["p0_candidate_generation_bridge_consistent"] is True
    assert payload["sglang_physical_restore"]["rows"] == 3
    assert payload["sglang_physical_restore"]["materialized_rows"] == 2
    assert payload["status"] == "waiting_on_matrix_and_live_p0_physical_copy"
    assert "calibration bridge is complete" in " ".join(
        payload["next_required_actions"]
    )


def test_report_flags_p0_live_audit_mismatch(tmp_path: Path) -> None:
    module = load_module()
    p0_csv = tmp_path / "p0.csv"
    p0_live_json = tmp_path / "p0_live.json"
    bridge_csv = tmp_path / "bridge.csv"
    bridge_rows_csv = tmp_path / "bridge_rows.csv"
    matrix_json = tmp_path / "matrix.json"
    slurm_json = tmp_path / "slurm.json"
    candidate_json = tmp_path / "candidate.json"
    candidate_generation_json = tmp_path / "candidate_generation.json"
    physical_restore_root = tmp_path / "physical_restore"
    write_p0_csv(p0_csv)
    write_p0_live_audits_json(p0_live_json, restore_rows=404)
    write_bridge_csv(bridge_csv, eligible=405, missing=0)
    write_bridge_rows_csv(bridge_rows_csv, rows=405)
    write_physical_restore_csv(physical_restore_root)
    matrix_json.write_text(
        json.dumps({"case_rows": [], "matrix_ready": False}),
        encoding="utf-8",
    )
    slurm_json.write_text(
        json.dumps({"overall_status": "pending", "status_counts": {}}),
        encoding="utf-8",
    )

    payload = module.build_payload(
        p0_csv=p0_csv,
        p0_live_audits_json=p0_live_json,
        p0_candidate_audit_json=candidate_json,
        p0_candidate_generation_audit_json=candidate_generation_json,
        p0_live_copy_events_json=tmp_path / "live_copy_events.json",
        p0_live_p2p_plan_json=tmp_path / "p2p_plan.json",
        p0_promotion_json=tmp_path / "promotion.json",
        bridge_csv=bridge_csv,
        bridge_rows_csv=bridge_rows_csv,
        matrix_json=matrix_json,
        slurm_json=slurm_json,
        physical_restore_root=physical_restore_root,
    )

    assert payload["status"] == "p0_live_audit_mismatch"
    assert payload["strict_final_ready"] is False
    assert payload["p0_live_audits_consistent"] is False
    assert "regenerate P0 live movement audits" in " ".join(
        payload["next_required_actions"]
    )


def test_report_surfaces_ready_live_physical_copy_candidates(tmp_path: Path) -> None:
    module = load_module()
    p0_csv = tmp_path / "p0.csv"
    p0_live_json = tmp_path / "p0_live.json"
    bridge_csv = tmp_path / "bridge.csv"
    bridge_rows_csv = tmp_path / "bridge_rows.csv"
    candidate_json = tmp_path / "candidate.json"
    candidate_generation_json = tmp_path / "candidate_generation.json"
    matrix_json = tmp_path / "matrix.json"
    slurm_json = tmp_path / "slurm.json"
    physical_restore_root = tmp_path / "physical_restore"
    write_p0_csv(p0_csv)
    write_p0_live_audits_json(p0_live_json)
    write_bridge_csv(bridge_csv, eligible=405, missing=0)
    bridge_digest = write_bridge_rows_csv(bridge_rows_csv, rows=405)
    write_candidate_audit_json(
        candidate_json,
        accepted_rows=405,
        missing_rows=0,
        ready=True,
        bridge_digest=bridge_digest,
    )
    write_physical_restore_csv(physical_restore_root)
    matrix_json.write_text(
        json.dumps(
            {
                "case_rows": [
                    {
                        "accepted_pairs": 3,
                        "case_id": "same_write_wait_complete",
                        "observed_pairs": 3,
                        "required_repeats": 3,
                        "status": "pass",
                    }
                ],
                "matrix_ready": True,
                "overall_status": "pass",
                "physical_restore_csv_count": 1,
                "physical_restore_rows": 11,
            }
        ),
        encoding="utf-8",
    )
    slurm_json.write_text(
        json.dumps({"overall_status": "complete", "status_counts": {"completed": 4}}),
        encoding="utf-8",
    )

    payload = module.build_payload(
        p0_csv=p0_csv,
        p0_live_audits_json=p0_live_json,
        p0_candidate_audit_json=candidate_json,
        p0_candidate_generation_audit_json=candidate_generation_json,
        p0_live_copy_events_json=tmp_path / "live_copy_events.json",
        p0_live_p2p_plan_json=tmp_path / "p2p_plan.json",
        p0_promotion_json=tmp_path / "promotion.json",
        bridge_csv=bridge_csv,
        bridge_rows_csv=bridge_rows_csv,
        matrix_json=matrix_json,
        slurm_json=slurm_json,
        physical_restore_root=physical_restore_root,
    )

    assert payload["status"] == (
        "host_matrix_ready_physical_copy_candidates_ready_p0_rows_not_promoted"
    )
    assert payload["strict_final_ready"] is False
    assert payload["p0_physical_tensor_copy_candidate_audit"][
        "strict_final_candidate_ready"
    ] is True
    assert "promote the accepted live physical_tensor_copy candidate evidence" in " ".join(
        payload["next_required_actions"]
    )


def test_report_accepts_promoted_p0_strict_final_overlay(tmp_path: Path) -> None:
    module = load_module()
    p0_csv = tmp_path / "p0.csv"
    p0_live_json = tmp_path / "p0_live.json"
    bridge_csv = tmp_path / "bridge.csv"
    bridge_rows_csv = tmp_path / "bridge_rows.csv"
    candidate_json = tmp_path / "candidate.json"
    candidate_generation_json = tmp_path / "candidate_generation.json"
    promotion_json = tmp_path / "promotion.json"
    matrix_json = tmp_path / "matrix.json"
    slurm_json = tmp_path / "slurm.json"
    physical_restore_root = tmp_path / "physical_restore"
    write_p0_csv(p0_csv)
    write_p0_live_audits_json(p0_live_json)
    write_bridge_csv(bridge_csv, eligible=405, missing=0)
    bridge_digest = write_bridge_rows_csv(bridge_rows_csv, rows=405)
    write_candidate_audit_json(
        candidate_json,
        accepted_rows=405,
        missing_rows=0,
        ready=True,
        bridge_digest=bridge_digest,
    )
    write_promotion_json(promotion_json, ready=True, rows=405)
    write_physical_restore_csv(physical_restore_root)
    matrix_json.write_text(
        json.dumps(
            {
                "case_rows": [
                    {
                        "accepted_pairs": 3,
                        "case_id": "same_write_wait_complete",
                        "observed_pairs": 3,
                        "required_repeats": 3,
                        "status": "pass",
                    }
                ],
                "matrix_ready": True,
                "overall_status": "pass",
                "physical_restore_csv_count": 1,
                "physical_restore_rows": 11,
            }
        ),
        encoding="utf-8",
    )
    slurm_json.write_text(
        json.dumps({"overall_status": "complete", "status_counts": {"completed": 4}}),
        encoding="utf-8",
    )

    payload = module.build_payload(
        p0_csv=p0_csv,
        p0_live_audits_json=p0_live_json,
        p0_candidate_audit_json=candidate_json,
        p0_candidate_generation_audit_json=candidate_generation_json,
        p0_live_copy_events_json=tmp_path / "live_copy_events.json",
        p0_live_p2p_plan_json=tmp_path / "p2p_plan.json",
        p0_promotion_json=promotion_json,
        bridge_csv=bridge_csv,
        bridge_rows_csv=bridge_rows_csv,
        matrix_json=matrix_json,
        slurm_json=slurm_json,
        physical_restore_root=physical_restore_root,
    )

    assert payload["status"] == "strict_final_ready"
    assert payload["strict_final_ready"] is True
    assert payload["p0_promotion"]["promotion_ready"] is True
    assert payload["p0_promotion_bridge_consistent"] is True
    assert payload["next_required_actions"] == []


def test_report_surfaces_missing_candidate_event_sources(tmp_path: Path) -> None:
    module = load_module()
    p0_csv = tmp_path / "p0.csv"
    p0_live_json = tmp_path / "p0_live.json"
    bridge_csv = tmp_path / "bridge.csv"
    bridge_rows_csv = tmp_path / "bridge_rows.csv"
    candidate_json = tmp_path / "candidate.json"
    candidate_generation_json = tmp_path / "candidate_generation.json"
    matrix_json = tmp_path / "matrix.json"
    slurm_json = tmp_path / "slurm.json"
    physical_restore_root = tmp_path / "physical_restore"
    write_p0_csv(p0_csv)
    write_p0_live_audits_json(p0_live_json)
    write_bridge_csv(bridge_csv, eligible=405, missing=0)
    bridge_digest = write_bridge_rows_csv(bridge_rows_csv, rows=405)
    write_candidate_generation_audit_json(
        candidate_generation_json,
        bridge_digest=bridge_digest,
    )
    write_physical_restore_csv(physical_restore_root)
    matrix_json.write_text(
        json.dumps({"case_rows": [], "matrix_ready": False}),
        encoding="utf-8",
    )
    slurm_json.write_text(
        json.dumps({"overall_status": "pending", "status_counts": {}}),
        encoding="utf-8",
    )

    payload = module.build_payload(
        p0_csv=p0_csv,
        p0_live_audits_json=p0_live_json,
        p0_candidate_audit_json=candidate_json,
        p0_candidate_generation_audit_json=candidate_generation_json,
        p0_live_copy_events_json=tmp_path / "live_copy_events.json",
        p0_live_p2p_plan_json=tmp_path / "p2p_plan.json",
        p0_promotion_json=tmp_path / "promotion.json",
        bridge_csv=bridge_csv,
        bridge_rows_csv=bridge_rows_csv,
        matrix_json=matrix_json,
        slurm_json=slurm_json,
        physical_restore_root=physical_restore_root,
    )

    generation = payload["p0_physical_tensor_copy_candidate_generation_audit"]
    assert generation["status"] == "missing_event_sources"
    assert generation["event_schema_version"] == "p0-live-copy-event-v1"
    assert generation["missing_event_source_count"] == 1
    assert payload["p0_candidate_generation_bridge_consistent"] is True
    assert "fix missing live physical-copy event source paths" in " ".join(
        payload["next_required_actions"]
    )


def test_report_flags_stale_candidate_audit_row_counts(tmp_path: Path) -> None:
    module = load_module()
    p0_csv = tmp_path / "p0.csv"
    p0_live_json = tmp_path / "p0_live.json"
    bridge_csv = tmp_path / "bridge.csv"
    bridge_rows_csv = tmp_path / "bridge_rows.csv"
    candidate_json = tmp_path / "candidate.json"
    candidate_generation_json = tmp_path / "candidate_generation.json"
    matrix_json = tmp_path / "matrix.json"
    slurm_json = tmp_path / "slurm.json"
    physical_restore_root = tmp_path / "physical_restore"
    write_p0_csv(p0_csv)
    write_p0_live_audits_json(p0_live_json)
    write_bridge_csv(bridge_csv, eligible=405, missing=0)
    write_bridge_rows_csv(bridge_rows_csv, rows=405)
    write_candidate_audit_json(candidate_json, accepted_rows=1, missing_rows=0)
    write_candidate_generation_audit_json(candidate_generation_json, required_rows=404)
    write_physical_restore_csv(physical_restore_root)
    matrix_json.write_text(
        json.dumps({"case_rows": [], "matrix_ready": False}),
        encoding="utf-8",
    )
    slurm_json.write_text(
        json.dumps({"overall_status": "pending", "status_counts": {}}),
        encoding="utf-8",
    )

    payload = module.build_payload(
        p0_csv=p0_csv,
        p0_live_audits_json=p0_live_json,
        p0_candidate_audit_json=candidate_json,
        p0_candidate_generation_audit_json=candidate_generation_json,
        p0_live_copy_events_json=tmp_path / "live_copy_events.json",
        p0_live_p2p_plan_json=tmp_path / "p2p_plan.json",
        p0_promotion_json=tmp_path / "promotion.json",
        bridge_csv=bridge_csv,
        bridge_rows_csv=bridge_rows_csv,
        matrix_json=matrix_json,
        slurm_json=slurm_json,
        physical_restore_root=physical_restore_root,
    )

    assert payload["p0_candidate_audit_bridge_consistent"] is False
    assert payload["p0_candidate_generation_bridge_consistent"] is False
    actions = " ".join(payload["next_required_actions"])
    assert "rerun validate_p0_physical_tensor_copy_candidates.py" in actions
    assert "regenerate P0 physical_tensor_copy candidate generation audit" in actions
