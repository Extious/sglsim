from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.network import SimIpcRegistry, SimZmqContext, SimZmqSocket, get_zmq_socket
from sglsim.srt.utils.tokenizer_runtime import (
    ensure_loaded,
    measure_encode,
    synthetic_assistant_unit_for_key,
)

if TYPE_CHECKING:
    from sglsim.srt.managers.data_parallel_controller import DataParallelController

logger = logging.getLogger(__name__)


def resolve_dp_size(server_args: ServerArgs) -> int:
    sched_cfg = ConfigManager.get_scheduler_config()
    return max(1, server_args.dp_size, sched_cfg.dp_size)


class TokenizerManager:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        dp_controller: DataParallelController | None = None,
        control_queue: Any | None = None,
    ) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self.dp_size = resolve_dp_size(server_args)
        self._enable_routing_key_affinity = (
            ConfigManager.get_scheduler_config().enable_routing_key_affinity
        )
        self._dp_controller = dp_controller
        self._control_queue = control_queue
        self._round_robin = 0
        self.rid_to_tokenized_state: dict[str, TokenizedGenerateReqInput] = {}
        ensure_loaded(server_args.model_path, logger=logger)

        ctx = SimZmqContext()
        use_ipc = SimIpcRegistry.get_global() is not None
        if self._dp_controller is None:
            self.send_to_scheduler = get_zmq_socket(
                ctx,
                SimZmqSocket.PUSH,
                port_args.scheduler_input_ipc_name_for_dp(0),
                bind=not use_ipc,
            )
        self.recv_from_detokenizer = get_zmq_socket(
            ctx,
            SimZmqSocket.PULL,
            port_args.tokenizer_ipc_name,
            bind=True,
        )

    def tokenize_request(self, req: GenerateReqInput) -> None:
        if req.input_ids is not None or req.input_len is not None:
            tokenize_dur = float(req.simulation.get("tokenize_dur", 0.0))
            if tokenize_dur > 0:
                time.sleep(tokenize_dur)
            return
        prompt = (req.prompt or "").strip()
        if not prompt:
            return
        model_path = self.server_args.model_path
        input_ids, tokenize_dur = measure_encode(model_path, prompt)
        req.input_ids = input_ids
        req.simulation["tokenize_dur"] = tokenize_dur
        if tokenize_dur > 0:
            time.sleep(tokenize_dur)

    def _pick_dp_rank(self, routing_key: str | None = None) -> int:
        if routing_key and self._enable_routing_key_affinity:
            from sglsim.srt.managers.data_parallel_controller import (
                routing_key_to_dp_rank,
            )

            return routing_key_to_dp_rank(routing_key, self.dp_size)
        if self.dp_size <= 1:
            return 0
        dp_rank = self._round_robin
        self._round_robin = (self._round_robin + 1) % self.dp_size
        return dp_rank

    def generate_request(self, req: GenerateReqInput) -> None:
        self.tokenize_request(req)
        req.simulation.setdefault("server_created_time", time.time())
        tok = TokenizedGenerateReqInput.from_generate(req)
        self.dispatch_tokenized_request(
            tok,
            preferred_dp_rank=self._pick_dp_rank(req.routing_key),
        )

    def dispatch_tokenized_request(
        self,
        tok: TokenizedGenerateReqInput,
        *,
        preferred_dp_rank: int | None = None,
    ) -> None:
        self.rid_to_tokenized_state[tok.rid] = tok
        dp_rank = preferred_dp_rank
        if dp_rank is None:
            dp_rank = self._pick_dp_rank(tok.routing_key)
        sim = tok.sampling_params.custom_params.setdefault("simulation", {})
        sim["assigned_dp_rank"] = dp_rank
        text_unit, token_id = synthetic_assistant_unit_for_key(
            self.server_args.model_path,
            tok.rid,
        )
        sim.setdefault("synthetic_output_text_unit", text_unit)
        sim.setdefault("synthetic_output_token_id", token_id)
        if self._control_queue is not None:
            self._control_queue.put(("inject", tok))
            return
        if self._dp_controller is not None:
            self._dp_controller.dispatch(tok)
            return
        self.send_to_scheduler.send_pyobj(tok)
