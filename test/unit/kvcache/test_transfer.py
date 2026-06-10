import pytest

from sglsim.srt.mem_cache.transfer import TransferModel


def test_transfer_model_accounts_for_latency_and_bandwidth():
    model = TransferModel(fixed_latency_s=0.001, bandwidth_bytes_per_s=1000)

    assert model.duration_s(num_bytes=2000) == 2.001


def test_transfer_model_zero_bytes_take_no_time():
    model = TransferModel(fixed_latency_s=0.001, bandwidth_bytes_per_s=1000)

    assert model.duration_s(num_bytes=0) == 0.0


@pytest.mark.parametrize(
    ("fixed_latency_s", "bandwidth_bytes_per_s"),
    [
        (-0.001, 1000),
        (0.001, 0),
        (0.001, -1000),
    ],
)
def test_transfer_model_rejects_invalid_configuration(
    fixed_latency_s, bandwidth_bytes_per_s
):
    with pytest.raises(ValueError):
        TransferModel(
            fixed_latency_s=fixed_latency_s,
            bandwidth_bytes_per_s=bandwidth_bytes_per_s,
        )


def test_transfer_model_rejects_negative_bytes():
    model = TransferModel(fixed_latency_s=0.001, bandwidth_bytes_per_s=1000)

    with pytest.raises(ValueError):
        model.duration_s(num_bytes=-1)
