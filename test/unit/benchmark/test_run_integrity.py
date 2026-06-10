import csv
import json

from benchmark.synthesis_requests.diagnostics.run_integrity import check_run_integrity


FIELDS = [
    "rid",
    "status",
    "output_length",
    "target_output_tokens",
    "generated_tokens",
    "e2e_latency_s",
]


def _write_run(path, rows, *, completed=None, failed=0):
    path.mkdir()
    with (path / "request_details.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if completed is None:
        completed = sum(1 for row in rows if row["status"] == "completed")
    (path / "metrics.json").write_text(
        json.dumps({"completed": completed, "failed": failed, "duration": 1.0}),
        encoding="utf-8",
    )


def test_run_integrity_accepts_complete_generated_rows(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            {
                "rid": "r0",
                "status": "completed",
                "output_length": "4",
                "target_output_tokens": "4",
                "generated_tokens": "4",
                "e2e_latency_s": "1.0",
            }
        ],
    )

    integrity = check_run_integrity(run_dir)

    assert integrity.ok is True
    assert integrity.row_count == 1
    assert integrity.completed_rows == 1


def test_run_integrity_rejects_aborted_short_output_and_bad_metrics(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            {
                "rid": "r0",
                "status": "aborted",
                "output_length": "8",
                "target_output_tokens": "8",
                "generated_tokens": "3",
                "e2e_latency_s": "1.0",
            }
        ],
        completed=1,
        failed=1,
    )

    integrity = check_run_integrity(run_dir)

    assert integrity.ok is False
    assert "non_completed_status=r0" in integrity.failures
    assert "short_generated_tokens=r0:3/8" in integrity.failures
    assert "metrics_completed_mismatch=1/0" in integrity.failures
    assert "metrics_failed=1" in integrity.failures


def test_run_integrity_checks_ref_row_count(tmp_path):
    sim_dir = tmp_path / "sim"
    ref_dir = tmp_path / "ref"
    _write_run(
        sim_dir,
        [
            {
                "rid": "r0",
                "status": "completed",
                "output_length": "4",
                "target_output_tokens": "4",
                "generated_tokens": "4",
                "e2e_latency_s": "1.0",
            }
        ],
    )
    _write_run(
        ref_dir,
        [
            {
                "rid": "r0",
                "status": "completed",
                "output_length": "4",
                "target_output_tokens": "",
                "generated_tokens": "",
                "e2e_latency_s": "1.0",
            },
            {
                "rid": "r1",
                "status": "completed",
                "output_length": "4",
                "target_output_tokens": "",
                "generated_tokens": "",
                "e2e_latency_s": "1.0",
            },
        ],
    )

    integrity = check_run_integrity(sim_dir, ref_dir=ref_dir)

    assert integrity.ok is False
    assert "row_count_mismatch=1/2" in integrity.failures
