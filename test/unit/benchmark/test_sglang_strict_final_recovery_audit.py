from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "audit_strict_final_recovery_matrix.py"
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_strict_final_recovery_audit", AUDIT_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pair(
    results_dir: Path,
    *,
    run_id: str,
    policy: str,
    invisible_job: str,
    visible_job: str,
    norm: float,
) -> None:
    pair_dir = results_dir / run_id
    pair_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "hicache_storage_prefetch_policy": policy,
        "invisible_slurm_job_id": invisible_job,
        "visible_slurm_job_id": visible_job,
        "invisible_success_count": 4,
        "visible_success_count": 4,
        "invisible_failover_response_count": 4,
        "visible_failover_response_count": 4,
        "invisible_pre_failover_backed_up_tokens": 4096,
        "visible_pre_failover_backed_up_tokens": 4096,
        "visible_retry_latency_norm_vs_invisible": norm,
        "visible_latency_delta_pct_vs_invisible": (norm - 1.0) * 100.0,
        "invisible_result_dir": str(results_dir / f"dp2_failure_smoke_{invisible_job}"),
        "visible_result_dir": str(results_dir / f"dp2_failure_smoke_{visible_job}"),
    }
    (pair_dir / "failure_visibility_pair_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def physical_row(
    *,
    job: str,
    policy: str,
    lookup_disabled: str,
    state: str,
    completion_ratio: float,
    cached: int,
) -> dict[str, object]:
    return {
        "slurm_job_id": job,
        "enable_remote_backup": "1",
        "disable_failover_remote_lookup": lookup_disabled,
        "hicache_storage_prefetch_policy": policy,
        "request_count": 4,
        "success_count": 4,
        "failover_response_count": 4,
        "pre_failover_backed_up_tokens": 4096,
        "max_data_pages": 512 if lookup_disabled == "0" else 0,
        "prefetch_accepted_count": 4 if lookup_disabled == "0" else 0,
        "prefetch_io_completion_ratio": completion_ratio,
        "retry_cached_tokens_after_accept_total": cached,
        "physical_restore_state": state,
    }


def write_plan_pairs(path: Path, *, best_effort: int = 3, wait_complete: int = 3) -> None:
    rows = []
    for i in range(best_effort):
        rows.append({"pair_id": f"best_{i}", "case_id": "same_write_best_effort"})
    for i in range(wait_complete):
        rows.append({"pair_id": f"wait_{i}", "case_id": "same_write_wait_complete"})
    write_csv(path, ["pair_id", "case_id"], rows)


def write_physical_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "slurm_job_id",
        "enable_remote_backup",
        "disable_failover_remote_lookup",
        "hicache_storage_prefetch_policy",
        "request_count",
        "success_count",
        "failover_response_count",
        "pre_failover_backed_up_tokens",
        "max_data_pages",
        "prefetch_accepted_count",
        "prefetch_io_completion_ratio",
        "retry_cached_tokens_after_accept_total",
        "physical_restore_state",
    ]
    write_csv(path, fields, rows)


def build_matrix(tmp_path: Path, *, wait_repeats: int) -> tuple[Path, Path, Path]:
    results_dir = tmp_path / "results"
    plan_pairs = tmp_path / "plan_pairs.csv"
    physical_csv = tmp_path / "physical.csv"
    write_plan_pairs(plan_pairs)

    physical_rows: list[dict[str, object]] = []
    for i in range(3):
        inv = f"10{i}"
        vis = f"20{i}"
        write_pair(
            results_dir,
            run_id=f"failure_visibility_pair_same_write_best_effort_{inv}_{vis}",
            policy="best_effort",
            invisible_job=inv,
            visible_job=vis,
            norm=0.94 + i * 0.01,
        )
        physical_rows.append(
            physical_row(
                job=inv,
                policy="best_effort",
                lookup_disabled="1",
                state="host_frontier_not_visible_to_retry",
                completion_ratio=0.0,
                cached=0,
            )
        )
        physical_rows.append(
            physical_row(
                job=vis,
                policy="best_effort",
                lookup_disabled="0",
                state="partial_async_materialized",
                completion_ratio=0.3,
                cached=1024,
            )
        )

    for i in range(wait_repeats):
        inv = f"30{i}"
        vis = f"40{i}"
        write_pair(
            results_dir,
            run_id=f"failure_visibility_pair_same_write_wait_complete_{inv}_{vis}",
            policy="wait_complete",
            invisible_job=inv,
            visible_job=vis,
            norm=1.2,
        )
        physical_rows.append(
            physical_row(
                job=inv,
                policy="wait_complete",
                lookup_disabled="1",
                state="host_frontier_not_visible_to_retry",
                completion_ratio=0.0,
                cached=0,
            )
        )
        physical_rows.append(
            physical_row(
                job=vis,
                policy="wait_complete",
                lookup_disabled="0",
                state="full_prefetch_materialized",
                completion_ratio=1.0,
                cached=4096,
            )
        )

    write_physical_rows(physical_csv, physical_rows)
    return results_dir, plan_pairs, physical_csv


def write_plan_json(path: Path) -> None:
    runs = []
    pairs = []
    for case_id, policy in (
        ("same_write_best_effort", "best_effort"),
        ("same_write_wait_complete", "wait_complete"),
    ):
        for repeat in range(1, 4):
            pair_id = f"strict_final_{case_id}_r{repeat}"
            invisible_label = f"{pair_id}_host_invisible_lookup_disabled"
            visible_label = f"{pair_id}_host_visible_lookup_enabled"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "case_id": case_id,
                    "repeat": repeat,
                    "prefetch_policy": policy,
                    "invisible_run_label": invisible_label,
                    "visible_run_label": visible_label,
                    "sync_pair_command": f"sync {pair_id}",
                    "physical_restore_command": f"physical {pair_id}",
                }
            )
            runs.append(
                {
                    "run_label": invisible_label,
                    "submit_command": f"submit {invisible_label}",
                }
            )
            runs.append(
                {
                    "run_label": visible_label,
                    "submit_command": f"submit {visible_label}",
                }
            )
    path.write_text(
        json.dumps({"runs": runs, "pairs": pairs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_complete_matrix_passes(tmp_path: Path) -> None:
    module = load_audit_module()
    results_dir, plan_pairs, physical_csv = build_matrix(tmp_path, wait_repeats=3)

    payload = module.run_audit(
        results_dir=results_dir,
        plan_pairs=plan_pairs,
        physical_restore_csv=physical_csv,
    )
    by_case = {row["case_id"]: row for row in payload["case_rows"]}

    assert payload["matrix_ready"] is True
    assert payload["overall_status"] == "pass"
    assert by_case["same_write_best_effort"]["accepted_pairs"] == 3
    assert by_case["same_write_wait_complete"]["accepted_pairs"] == 3


def test_strict_final_pair_dirs_are_discovered(tmp_path: Path) -> None:
    module = load_audit_module()
    results_dir = tmp_path / "results"
    write_pair(
        results_dir,
        run_id="strict_final_same_write_wait_complete_r2",
        policy="wait_complete",
        invisible_job="35411",
        visible_job="35412",
        norm=1.4,
    )

    rows = module.collect_pair_summaries(results_dir)

    assert [row["run_id"] for row in rows] == [
        "strict_final_same_write_wait_complete_r2"
    ]
    assert rows[0]["case_id"] == "same_write_wait_complete"
    assert rows[0]["pair_dir"] == str(
        results_dir / "strict_final_same_write_wait_complete_r2"
    )


def test_nested_physical_restore_csvs_are_discovered(tmp_path: Path) -> None:
    module = load_audit_module()
    results_dir, plan_pairs, physical_csv = build_matrix(tmp_path, wait_repeats=3)
    physical_root = tmp_path / "physical_restore_evidence"
    primary_csv = physical_root / "physical_restore_runs.csv"
    nested_csv = physical_root / "strict_final_wait_complete" / "physical_restore_runs.csv"
    primary_rows = []
    nested_rows = []

    for row in module.read_csv(physical_csv):
        job = str(row["slurm_job_id"])
        if job.startswith(("10", "20")):
            primary_rows.append(row)
        else:
            nested_rows.append(row)

    stale_wait_visible = dict(nested_rows[-1])
    stale_wait_visible["physical_restore_state"] = "partial_async_materialized"
    stale_wait_visible["prefetch_io_completion_ratio"] = "0.5"
    primary_rows.append(stale_wait_visible)

    write_physical_rows(primary_csv, primary_rows)
    write_physical_rows(nested_csv, nested_rows)

    payload = module.run_audit(
        results_dir=results_dir,
        plan_pairs=plan_pairs,
        physical_restore_csv=primary_csv,
    )
    by_case = {row["case_id"]: row for row in payload["case_rows"]}
    wait_rows = [
        row
        for row in payload["pair_rows"]
        if row["case_id"] == "same_write_wait_complete"
    ]

    assert payload["matrix_ready"] is True
    assert payload["physical_restore_csv_count"] == 2
    assert by_case["same_write_wait_complete"]["accepted_pairs"] == 3
    assert all(
        row["visible_physical_restore_source_csv"] == str(nested_csv)
        for row in wait_rows
    )


def test_incomplete_wait_complete_matrix_stays_partial(tmp_path: Path) -> None:
    module = load_audit_module()
    results_dir, plan_pairs, physical_csv = build_matrix(tmp_path, wait_repeats=1)
    plan_json = tmp_path / "plan.json"
    write_plan_json(plan_json)

    payload = module.run_audit(
        results_dir=results_dir,
        plan_pairs=plan_pairs,
        plan_json=plan_json,
        physical_restore_csv=physical_csv,
    )
    by_case = {row["case_id"]: row for row in payload["case_rows"]}
    markdown = module.render_markdown(payload)
    missing_ids = {row["pair_id"] for row in payload["missing_plan_pairs"]}

    assert payload["matrix_ready"] is False
    assert payload["overall_status"] == "partial"
    assert by_case["same_write_best_effort"]["status"] == "pass"
    assert by_case["same_write_wait_complete"]["status"] == "partial"
    assert missing_ids == {
        "strict_final_same_write_wait_complete_r2",
        "strict_final_same_write_wait_complete_r3",
    }
    assert all("best_effort" not in pair_id for pair_id in missing_ids)
    assert "submit strict_final_same_write_wait_complete_r2" in markdown
    assert "not a replacement" in markdown
    assert "SGLSim P0 `physical_tensor_copy` rows" in markdown
    assert "Invisible state" in markdown
    assert "host_frontier_not_visible_to_retry" in markdown


def test_rejects_pair_when_lookup_disabled_control_sees_storage(tmp_path: Path) -> None:
    module = load_audit_module()
    results_dir, plan_pairs, physical_csv = build_matrix(tmp_path, wait_repeats=3)
    rows = module.read_csv(physical_csv)
    for row in rows:
        if row["slurm_job_id"] == "300":
            row["max_data_pages"] = "512"
            row["prefetch_accepted_count"] = "1"
            row["retry_cached_tokens_after_accept_total"] = "512"
            row["physical_restore_state"] = "partial_async_materialized"
    write_physical_rows(physical_csv, rows)

    payload = module.run_audit(
        results_dir=results_dir,
        plan_pairs=plan_pairs,
        physical_restore_csv=physical_csv,
    )
    wait_rows = [
        row
        for row in payload["pair_rows"]
        if row["case_id"] == "same_write_wait_complete"
    ]
    rejected = [row for row in wait_rows if row["invisible_job_id"] == "300"]

    assert rejected
    assert rejected[0]["accepted"] == "false"
    assert rejected[0]["invisible_max_data_pages"] == 512
    assert '"invisible_hidden_control": false' in rejected[0]["checks"]
