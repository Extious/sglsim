"""Peer-GPU KV backup metadata tracker (KevlarFlow-style gpu_backup)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from sglsim.srt.managers.schedule_batch import Req, ScheduleBatch


@dataclass
class _PeerReqState:
    origin_input_len: int = 0
    decode_tokens: List[int] = field(default_factory=list)
    peer_backed_up: int = 0


class PeerKVReplicator:
    """Metadata-only peer backup: tracks decode tokens replicated to peer GPU."""

    def __init__(self, *, page_size: int, peer_backup_lag_tokens: int, dp_size: int):
        self.page_size = max(1, page_size)
        self.peer_backup_lag_tokens = max(0, peer_backup_lag_tokens)
        self.dp_size = max(1, dp_size)
        self._states: Dict[str, _PeerReqState] = {}

    def on_prefill_complete(self, req: Req) -> None:
        state = self._states.setdefault(req.rid, _PeerReqState())
        state.origin_input_len = len(req.origin_input_ids)

    def on_decode_step(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.output_ids:
                continue
            state = self._states.setdefault(req.rid, _PeerReqState())
            if not state.origin_input_len:
                state.origin_input_len = len(req.origin_input_ids)
            latest = req.output_ids[-1]
            state.decode_tokens.append(latest)
            lagged = max(0, len(state.decode_tokens) - self.peer_backup_lag_tokens)
            state.peer_backed_up = lagged

    def count_peer_backed_up_tokens(self, token_ids: list) -> int:
        if not token_ids:
            return 0
        for rid, state in self._states.items():
            total = state.origin_input_len + len(state.decode_tokens)
            if len(token_ids) >= total and token_ids[:total] == (
                token_ids[: state.origin_input_len] + state.decode_tokens
            ):
                return state.origin_input_len + state.peer_backed_up
        return 0

    def count_peer_backed_up_for_req(self, rid: str) -> int:
        state = self._states.get(rid)
        if state is None:
            return 0
        return state.origin_input_len + state.peer_backed_up

    def on_request_finished(self, rid: str) -> None:
        self._states.pop(rid, None)
