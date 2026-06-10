from sglsim.srt.simulation.pipeline_accounting import (
    PipelineStages,
    compute_pipeline_sleep_s,
    compute_visible_gpu,
)


def test_overlap_hides_h2d_behind_prior_inference():
    stages = PipelineStages(h2d_s=3.0, forward_s=2.0, backup_pending_s=3.0)
    overlapped = compute_visible_gpu(stages, last_infer_s=5.0, overlap=True)
    serial = compute_visible_gpu(stages, last_infer_s=5.0, overlap=False)
    assert overlapped.visible_gpu_s == 3.0
    assert serial.visible_gpu_s == 8.0
    assert overlapped.visible_gpu_s < serial.visible_gpu_s
    assert overlapped.backup_remainder_s == 1.0
    assert overlapped.overlapped_backup_s == 2.0


def test_overlap_zero_h2d_when_last_infer_covers_load():
    stages = PipelineStages(h2d_s=3.0, forward_s=2.0, backup_pending_s=1.0)
    result = compute_visible_gpu(stages, last_infer_s=5.0, overlap=True)
    assert result.visible_gpu_s == 2.0
    assert result.backup_remainder_s == 0.0


def test_serial_sums_all_stages():
    stages = PipelineStages(h2d_s=3.0, forward_s=2.0, backup_pending_s=1.0)
    result = compute_visible_gpu(stages, last_infer_s=5.0, overlap=False)
    assert result.visible_gpu_s == 6.0


def test_zero_last_infer_shows_full_h2d():
    stages = PipelineStages(h2d_s=2.0, forward_s=1.0, backup_pending_s=0.5)
    result = compute_visible_gpu(stages, last_infer_s=0.0, overlap=True)
    assert result.visible_gpu_s == 3.0


def test_negative_inputs_clamped():
    stages = PipelineStages(h2d_s=-1.0, forward_s=2.0, backup_pending_s=-1.0)
    result = compute_visible_gpu(stages, last_infer_s=-1.0, overlap=True)
    assert result.visible_gpu_s == 2.0


def test_pipeline_sleep_overlap_forward_only():
    stages = PipelineStages(h2d_s=3.0, forward_s=2.0, backup_pending_s=3.0)
    result = compute_pipeline_sleep_s(stages, last_infer_s=5.0, overlap=True)
    assert result.sleep_s == 2.0
    assert result.forward_s == 2.0
    assert result.diagnostic_visible_gpu_s == 3.0
    assert result.comm_h2d_visible_s == 0.0
    assert result.comm_d2h_remainder_s == 1.0


def test_pipeline_sleep_no_overlap_sleeps_serial_transfer_and_forward():
    stages = PipelineStages(h2d_s=3.0, forward_s=2.0, backup_pending_s=1.0)
    result = compute_pipeline_sleep_s(stages, last_infer_s=5.0, overlap=False)
    assert result.sleep_s == 6.0
    assert result.diagnostic_visible_gpu_s == 6.0
    assert result.comm_h2d_visible_s == 3.0
    assert result.comm_d2h_remainder_s == 1.0
