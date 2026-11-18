from __future__ import annotations

import pytest

from sglsim.srt.simulation.aegis_independent_tiers import ChunkSpec, ProtectionTier
from sglsim.srt.simulation.aegis_transfer_engine import (
    AegisTransferEngine,
    TransferSubmission,
)


def _spec() -> ChunkSpec:
    return ChunkSpec(
        request_id="r",
        ordinal=0,
        token_start=0,
        token_end=4,
        byte_size=4096,
        route_version=1,
        source_worker="source",
        hbm_target="target",
        model_revision="model",
        kv_abi="kv",
        parallel_layout="dp2-tp1-pp1",
        shard_count=2,
    )


def _submission(tier: ProtectionTier, **overrides: object) -> TransferSubmission:
    values: dict[str, object] = {
        "tier": tier,
        "spec": _spec(),
        "destination_id": "target" if tier is ProtectionTier.HBM else "store-dram",
        "copy_kind": "source_d2d" if tier is ProtectionTier.HBM else "source_d2h",
        "ready_at_ms": 1.0,
        "service_ms": 5.0,
        "transfer_succeeded": True,
        "acked_shards": (0, 1),
    }
    values.update(overrides)
    return TransferSubmission(**values)


def test_transfer_engine_has_independent_hbm_and_dram_rate_lanes() -> None:
    engine = AegisTransferEngine()
    hbm0 = engine.enqueue(_submission(ProtectionTier.HBM))
    dram0 = engine.enqueue(_submission(ProtectionTier.DRAM))
    hbm1 = engine.enqueue(_submission(ProtectionTier.HBM))

    assert hbm0.started_at_ms == 1.0
    assert dram0.started_at_ms == 1.0
    assert hbm1.started_at_ms == hbm0.completed_at_ms
    snapshot = engine.snapshot()
    assert snapshot["control_boundary"] == "queue_and_rate_limit_only"
    assert snapshot["transfer_engine_selects_victims"] is False
    assert not hasattr(engine, "select_victims")


@pytest.mark.parametrize(
    "tier,overrides,match",
    [
        (ProtectionTier.HBM, {"copy_kind": "source_d2h"}, "copy_kind"),
        (ProtectionTier.HBM, {"destination_id": "other"}, "route-bound"),
        (ProtectionTier.DRAM, {"destination_id": "target"}, "Store DRAM"),
        (ProtectionTier.DRAM, {"source_owned": False}, "source-owned"),
        (ProtectionTier.DRAM, {"target_owned": True}, "source-owned"),
        (ProtectionTier.DRAM, {"hbm_to_dram": True}, "source-owned"),
        (ProtectionTier.DRAM, {"acked_shards": (0,)}, "all shard ACKs"),
    ],
)
def test_transfer_engine_rejects_policy_or_legacy_path_semantics(
    tier: ProtectionTier, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _submission(tier, **overrides)


def test_transfer_submission_schema_has_no_victim_selection_field() -> None:
    with pytest.raises(TypeError, match="victim_ids"):
        TransferSubmission(
            **{
                **_submission(ProtectionTier.HBM).__dict__,
                "victim_ids": ("r:0",),
            }
        )
