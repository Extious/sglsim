from __future__ import annotations

import logging
import os

from sglsim.srt.managers.tokenizer_manager import resolve_dp_size
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.utils.logging import configure_logger
from sglsim.srt.utils.network import SimIpcRegistry, SimZmqContext, SimZmqSocket, get_zmq_socket
from sglsim.srt.utils.tokenizer_runtime import measure_decode

logger = logging.getLogger(__name__)


class DetokenizerManager:
    """In-process bridge (legacy); production path uses run_detokenizer_process."""

    def __init__(self, port_args: PortArgs) -> None:
        ctx = SimZmqContext()
        self.recv_from_scheduler = get_zmq_socket(
            ctx,
            SimZmqSocket.PULL,
            port_args.detokenizer_ipc_name,
            bind=True,
        )
        self.send_to_tokenizer = get_zmq_socket(
            ctx,
            SimZmqSocket.PUSH,
            port_args.tokenizer_ipc_name,
            bind=False,
        )

    def event_loop_once(self) -> bool:
        try:
            msg = self.recv_from_scheduler.recv_pyobj(flags=1)
        except Exception:
            return False
        self.send_to_tokenizer.send_pyobj(msg)
        return True

    def event_loop_until_idle(self, max_steps: int = 100_000) -> None:
        for _ in range(max_steps):
            if not self.event_loop_once():
                break


def _handle_message(
    msg: dict,
    *,
    model_path: str,
    send_to_tokenizer,
    send_to_scheduler_by_dp: dict[int, object],
    simulate_detokenizer_time: bool = False,
) -> None:
    op = msg.get("op", "update")
    if op == "failover_batch":
        send_to_tokenizer.send_pyobj(msg)
        return
    rid = str(msg.get("rid", ""))
    if not rid:
        return

    if op == "finalize":
        if simulate_detokenizer_time:
            output_ids = list(msg.get("output_ids", []))
            text, decode_dur = measure_decode(model_path, output_ids)
            dp_rank = int(msg.get("dp_rank", 0))
            send_to_scheduler = send_to_scheduler_by_dp.get(dp_rank)
            if send_to_scheduler is None:
                logger.error("Unknown dp_rank=%d for rid=%s finalize", dp_rank, rid)
                return
            send_to_scheduler.send_pyobj(
                {
                    "op": "decode_ack",
                    "rid": rid,
                    "decode_dur": decode_dur,
                }
            )
        else:
            # Fast path: no real detokenization, no synchronous ack.
            text_unit = str(msg.get("synthetic_output_text_unit", ""))
            generated = int(msg.get("generated_tokens", 0))
            text = text_unit * generated if text_unit else ""
            decode_dur = 0.0
        out = dict(msg)
        out.pop("op", None)
        out.pop("output_ids", None)
        out["finished"] = True
        out["text"] = text
        out["decode_dur"] = decode_dur
        send_to_tokenizer.send_pyobj(out)
        return

    send_to_tokenizer.send_pyobj(msg)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry: SimIpcRegistry,
    ready_event,
) -> None:
    configure_logger(server_args)
    if server_args.sim_config_path:
        os.environ["SGLSIM_CONFIG_PATH"] = server_args.sim_config_path

    from sglsim.srt.utils.tokenizer_runtime import ensure_loaded

    SimIpcRegistry.install_global(ipc_registry)
    # Only pay the (large) tokenizer load when actually modeling detok latency.
    if server_args.simulate_detokenizer_time:
        ensure_loaded(server_args.model_path, logger=logger)

    ctx = SimZmqContext()
    recv_from_scheduler = get_zmq_socket(
        ctx,
        SimZmqSocket.PULL,
        port_args.detokenizer_ipc_name,
        bind=False,
    )
    send_to_tokenizer = get_zmq_socket(
        ctx,
        SimZmqSocket.PUSH,
        port_args.tokenizer_ipc_name,
        bind=False,
    )
    dp_size = resolve_dp_size(server_args)
    send_to_scheduler_by_dp = {
        dp_rank: get_zmq_socket(
            ctx,
            SimZmqSocket.PUSH,
            port_args.detokenizer_reply_ipc_name_for_dp(dp_rank),
            bind=False,
        )
        for dp_rank in range(dp_size)
    }
    ready_event.set()
    logger.info("Detokenizer subprocess ready")

    import gc

    gc.collect()
    gc.freeze()

    while True:
        try:
            msg = recv_from_scheduler.recv_pyobj()
        except Exception:
            break
        if msg.get("op") == "stop":
            break
        _handle_message(
            msg,
            model_path=server_args.model_path,
            send_to_tokenizer=send_to_tokenizer,
            send_to_scheduler_by_dp=send_to_scheduler_by_dp,
            simulate_detokenizer_time=server_args.simulate_detokenizer_time,
        )
