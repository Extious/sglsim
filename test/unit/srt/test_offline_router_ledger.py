from __future__ import annotations

import pytest

from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.simulation.aegis_recovery_protocol import ProtocolViolation
from sglsim.srt.simulation.offline_coordinator import (
    OfflineEventCoordinator,
    OfflineEventPriority,
)
from sglsim.srt.simulation.offline_router_ledger import (
    CONTINUATION_EVIDENCE_CLASS,
    REPLAY_CONTRACT_SCHEMA,
    OfflineRouterLedger,
)


def _ledger() -> OfflineRouterLedger:
    return OfflineRouterLedger(
        model_revision="test-model-r1",
        kv_abi="test-kv-abi-v1",
        parallel_layout="dp2-pp2-tp1",
    )


def _original(
    ledger: OfflineRouterLedger,
    *,
    rid: str = "logical-0",
    prompt: tuple[int, ...] = (11, 12, 13),
    output_tokens: int = 5,
) -> TokenizedGenerateReqInput:
    request = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid=rid,
            input_ids=list(prompt),
            max_new_tokens=output_tokens,
            simulation={"synthetic_output_token_id": 97},
        )
    )
    ledger.register_dispatch(request)
    ledger.set_assignment(rid, dp_rank=0, pp_size=2, logical_tp_size=1)
    return request


def _message(rid: str, generated: int, *, final: bool = False) -> dict:
    return {
        "op": "finalize" if final else "update",
        "rid": rid,
        "generated_tokens": generated,
        "status": "completed" if final else "running",
    }


def _bind_replay(
    ledger: OfflineRouterLedger,
    spec: dict,
    *,
    rid: str,
    predecessor: str,
    event_id: str,
) -> TokenizedGenerateReqInput:
    request = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid=rid,
            input_ids=list(spec["replay_input_ids"]),
            max_new_tokens=int(spec["remaining_max_new_tokens"]),
            simulation={
                "failover_replay": True,
                "failure_event_id": event_id,
                "aegis_replay_of": predecessor,
                "aegis_logical_request_id": spec["logical_request_id"],
                "aegis_route_version": spec["next_route_version"],
                "aegis_router_replay_contract": REPLAY_CONTRACT_SCHEMA,
                "aegis_committed_output_position": spec[
                    "committed_output_position"
                ],
                "aegis_metadata_continuation_digest": spec[
                    "continuation_state_digest"
                ],
                "aegis_metadata_compatibility_hash": spec[
                    "continuation_compatibility_hash"
                ],
                "aegis_metadata_prefix_hash": spec[
                    "continuation_prefix_hash"
                ],
                "aegis_continuation_evidence_class": (
                    CONTINUATION_EVIDENCE_CLASS
                ),
                "pre_failure_total_tokens": spec["pre_failure_total_tokens"],
                "synthetic_output_token_id": spec["synthetic_output_token_id"],
            },
        )
    )
    ledger.register_dispatch(request)
    ledger.set_assignment(rid, dp_rank=1, pp_size=2, logical_tp_size=1)
    ledger.acknowledge_scheduler_boundary(
        rid,
        response_dp_rank=1,
        response_pp_rank=0,
        response_tp_rank=0,
    )
    return request


def test_mid_decode_replay_is_prompt_plus_committed_outputs_and_n_minus_k() -> None:
    ledger = _ledger()
    original = _original(ledger)
    first_publish = ledger.observe_scheduler_message(_message(original.rid, 2))
    assert first_publish["router_newly_published_sequence_start"] == 1
    assert first_publish["router_newly_published_sequence_end"] == 2

    boundary = ledger.fence_execution(original.rid, event_id="failure-1")
    spec = ledger.replay_spec(original.rid, event_id="failure-1")
    assert boundary["committed_output_position"] == 2
    assert boundary["published_output_position"] == 2
    assert boundary["pre_failure_total_tokens"] == 5
    assert spec["remaining_max_new_tokens"] == 3
    assert spec["replay_input_ids"] == [11, 12, 13, 97, 97]

    replay = _bind_replay(
        ledger,
        spec,
        rid="logical-0.replay.failure-1",
        predecessor=original.rid,
        event_id="failure-1",
    )
    replay_publish = ledger.observe_scheduler_message(
        _message(replay.rid, 3, final=True)
    )
    assert replay_publish["router_newly_published_sequence_start"] == 3
    assert replay_publish["router_newly_published_sequence_end"] == 5
    audit = ledger.audit(require_complete=True)
    assert audit["status"] == "PASS"
    exported = ledger.export()["logical_requests"][0]
    assert exported["committed_output_position"] == 5
    assert [entry["sequence"] for entry in exported["output_entries"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert exported["executions"][1]["base_output_sequence"] == 2
    assert exported["executions"][1]["requested_new_tokens"] == 3


def test_k_zero_failure_replays_the_full_original_budget() -> None:
    ledger = _ledger()
    original = _original(ledger, output_tokens=4)
    boundary = ledger.fence_execution(original.rid, event_id="failure-k0")
    spec = ledger.replay_spec(original.rid, event_id="failure-k0")
    assert boundary["committed_output_position"] == 0
    assert boundary["pre_failure_total_tokens"] == 3
    assert spec["remaining_max_new_tokens"] == 4
    assert spec["replay_input_ids"] == [11, 12, 13]


def test_k_equals_n_is_terminal_and_must_not_create_a_replay() -> None:
    ledger = _ledger()
    original = _original(ledger, output_tokens=3)
    ledger.observe_scheduler_message(_message(original.rid, 3, final=True))
    boundary = ledger.fence_execution(original.rid, event_id="after-terminal")
    spec = ledger.replay_spec(original.rid, event_id="after-terminal")
    assert boundary["committed_output_position"] == 3
    assert boundary["remaining_max_new_tokens"] == 0
    assert boundary["replay_required"] is False
    assert spec["replay_required"] is False
    assert ledger.audit(require_complete=True)["status"] == "PASS"


def test_same_time_failure_precedes_compute_completion() -> None:
    coordinator = OfflineEventCoordinator(dp_size=1)
    coordinator.schedule_at(
        1.25,
        kind="compute_marker",
        priority=OfflineEventPriority.RANK_COMPLETION,
    )
    coordinator.schedule_at(
        1.25,
        kind="failure_marker",
        priority=OfflineEventPriority.FAILURE_CONTROL,
    )
    assert coordinator.pop_next().kind == "failure_marker"
    assert coordinator.pop_next().kind == "compute_marker"


def test_repeat_failure_advances_v1_v2_v3_and_conserves_outputs() -> None:
    ledger = _ledger()
    original = _original(ledger, output_tokens=5)
    ledger.observe_scheduler_message(_message(original.rid, 1))
    ledger.fence_execution(original.rid, event_id="failure-v1")
    spec_v2 = ledger.replay_spec(original.rid, event_id="failure-v1")
    replay_v2 = _bind_replay(
        ledger,
        spec_v2,
        rid="logical-0.replay.v2",
        predecessor=original.rid,
        event_id="failure-v1",
    )
    ledger.observe_scheduler_message(_message(replay_v2.rid, 1))

    ledger.fence_execution(replay_v2.rid, event_id="failure-v2")
    spec_v3 = ledger.replay_spec(replay_v2.rid, event_id="failure-v2")
    assert spec_v3["committed_output_position"] == 2
    assert spec_v3["remaining_max_new_tokens"] == 3
    replay_v3 = _bind_replay(
        ledger,
        spec_v3,
        rid="logical-0.replay.v3",
        predecessor=replay_v2.rid,
        event_id="failure-v2",
    )
    ledger.observe_scheduler_message(_message(replay_v3.rid, 3, final=True))

    assert ledger.audit(require_complete=True)["status"] == "PASS"
    logical = ledger.export()["logical_requests"][0]
    assert [row["route_version"] for row in logical["executions"]] == [1, 2, 3]
    assert [row["committed_sequences"] for row in logical["executions"]] == [
        [1],
        [2],
        [3, 4, 5],
    ]


def test_stale_route_output_is_rejected_before_and_after_successor_bind() -> None:
    ledger = _ledger()
    original = _original(ledger, output_tokens=3)
    ledger.observe_scheduler_message(_message(original.rid, 1))
    ledger.fence_execution(original.rid, event_id="failure-stale")
    with pytest.raises(ProtocolViolation, match="stale worker output fence"):
        ledger.observe_scheduler_message(_message(original.rid, 2))

    spec = ledger.replay_spec(original.rid, event_id="failure-stale")
    replay = _bind_replay(
        ledger,
        spec,
        rid="logical-0.replay.stale",
        predecessor=original.rid,
        event_id="failure-stale",
    )
    with pytest.raises(ProtocolViolation, match="stale worker output fence"):
        ledger.observe_scheduler_message(_message(original.rid, 2))
    ledger.observe_scheduler_message(_message(replay.rid, 2, final=True))
    logical = ledger.export()["logical_requests"][0]
    assert len(logical["stale_output_rejections"]) == 2
    assert ledger.audit(require_complete=True)["status"] == "PASS"


def test_replay_output_is_fail_closed_until_all_rank_boundary_ack() -> None:
    ledger = _ledger()
    original = _original(ledger, output_tokens=2)
    ledger.fence_execution(original.rid, event_id="failure-ready")
    spec = ledger.replay_spec(original.rid, event_id="failure-ready")
    replay = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid="logical-0.replay.unready",
            input_ids=list(spec["replay_input_ids"]),
            max_new_tokens=2,
            simulation={
                "failover_replay": True,
                "failure_event_id": "failure-ready",
                "aegis_logical_request_id": spec["logical_request_id"],
                "aegis_route_version": 2,
                "aegis_router_replay_contract": REPLAY_CONTRACT_SCHEMA,
                "aegis_committed_output_position": 0,
                "aegis_metadata_continuation_digest": spec[
                    "continuation_state_digest"
                ],
                "aegis_metadata_compatibility_hash": spec[
                    "continuation_compatibility_hash"
                ],
                "aegis_metadata_prefix_hash": spec[
                    "continuation_prefix_hash"
                ],
                "pre_failure_total_tokens": 3,
                "synthetic_output_token_id": spec["synthetic_output_token_id"],
            },
        )
    )
    ledger.register_dispatch(replay)
    ledger.set_assignment(replay.rid, dp_rank=1, pp_size=2, logical_tp_size=1)
    with pytest.raises(ProtocolViolation, match="stale worker output fence"):
        ledger.observe_scheduler_message(_message(replay.rid, 1))
    with pytest.raises(ProtocolViolation, match="PP0/TP-aggregate"):
        ledger.acknowledge_scheduler_boundary(
            replay.rid,
            response_dp_rank=1,
            response_pp_rank=1,
            response_tp_rank=0,
        )
