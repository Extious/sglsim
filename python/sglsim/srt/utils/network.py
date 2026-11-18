from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict, deque
from typing import Any

# In-process ZMQ simulation: endpoint -> queue of pyobj messages.

_QUEUES: dict[str, deque[Any]] = defaultdict(deque)
_NOBLOCK = 1
_ACTIVE_REGISTRY: SimIpcRegistry | None = None


class ZMQError(Exception):
    pass


class SimIpcRegistry:
    """Cross-process queues for endpoints shared between parent and scheduler children."""

    def __init__(self) -> None:
        self._queues: dict[str, mp.Queue] = {}

    def register(self, endpoint: str) -> mp.Queue:
        if endpoint not in self._queues:
            self._queues[endpoint] = mp.Queue()
        return self._queues[endpoint]

    def get_queue(self, endpoint: str) -> mp.Queue | None:
        return self._queues.get(endpoint)

    def reset(self, endpoint: str) -> mp.Queue:
        """Replace one registered channel with an empty process-safe queue.

        A hard-failed PP replica can leave half-consumed request/proxy/output
        envelopes behind.  Reusing that queue after process restart would mix
        two pipeline generations.  The parent resets only stage-local channels
        after every process in that DP replica has exited; unrelated DP and
        detokenizer channels keep their original queue identity.
        """

        if endpoint not in self._queues:
            raise KeyError(endpoint)
        queue: mp.Queue = mp.Queue()
        self._queues[endpoint] = queue
        return queue

    def endpoints(self) -> list[str]:
        return list(self._queues.keys())

    @classmethod
    def install_global(cls, registry: SimIpcRegistry | None) -> None:
        global _ACTIVE_REGISTRY
        _ACTIVE_REGISTRY = registry

    @classmethod
    def get_global(cls) -> SimIpcRegistry | None:
        return _ACTIVE_REGISTRY


def build_dp_ipc_registry(port_args: Any, dp_size: int) -> SimIpcRegistry:
    registry = SimIpcRegistry()
    for dp_rank in range(dp_size):
        registry.register(port_args.scheduler_input_ipc_name_for_dp(dp_rank))
        registry.register(port_args.detokenizer_reply_ipc_name_for_dp(dp_rank))
    registry.register(port_args.detokenizer_ipc_name)
    registry.register(port_args.tokenizer_ipc_name)
    return registry


def build_dp_pp_ipc_registry(
    port_args: Any, dp_size: int, pp_size: int
) -> SimIpcRegistry:
    """Build the DP ingress plus stage-local PP metadata endpoints.

    This is the process-topology registry required by pinned SGLang's PP event
    order.  The current online engine will switch to it together with the PP
    event-loop integration; exposing it separately keeps the existing PP=1
    serving path unchanged while its invariants are established.
    """

    if dp_size <= 0 or pp_size <= 0:
        raise ValueError("dp_size and pp_size must be positive")
    registry = SimIpcRegistry()
    for dp_rank in range(dp_size):
        for pp_rank in range(pp_size):
            registry.register(
                port_args.scheduler_input_ipc_name_for_stage(dp_rank, pp_rank)
            )
            registry.register(port_args.pp_proxy_ipc_name_for_stage(dp_rank, pp_rank))
            registry.register(port_args.pp_output_ipc_name_for_stage(dp_rank, pp_rank))
        registry.register(port_args.detokenizer_reply_ipc_name_for_dp(dp_rank))
    registry.register(port_args.detokenizer_ipc_name)
    registry.register(port_args.tokenizer_ipc_name)
    return registry


class SimZmqContext:
    def socket(self, socket_type: int) -> SimZmqSocket:
        return SimZmqSocket(socket_type)


class SimZmqSocket:
    PULL = 1
    PUSH = 2
    DEALER = 3

    def __init__(self, socket_type: int) -> None:
        self.socket_type = socket_type
        self.endpoint: str | None = None
        self._bound = False

    def _ipc_queue(self) -> mp.Queue | None:
        registry = SimIpcRegistry.get_global()
        if registry is None or self.endpoint is None:
            return None
        return registry.get_queue(self.endpoint)

    def bind(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._bound = True
        ipc = self._ipc_queue()
        if ipc is None:
            _QUEUES.setdefault(endpoint, deque())

    def connect(self, endpoint: str) -> None:
        self.endpoint = endpoint
        ipc = self._ipc_queue()
        if ipc is None:
            _QUEUES.setdefault(endpoint, deque())

    def send_pyobj(self, obj: Any) -> None:
        if self.endpoint is None:
            raise ZMQError("socket not bound or connected")
        ipc = self._ipc_queue()
        if ipc is not None:
            ipc.put(obj)
            return
        _QUEUES[self.endpoint].append(obj)

    def recv_pyobj(self, flags: int = 0) -> Any:
        if self.endpoint is None:
            raise ZMQError("socket not bound or connected")
        ipc = self._ipc_queue()
        if ipc is not None:
            if flags & _NOBLOCK:
                try:
                    return ipc.get_nowait()
                except Exception as exc:
                    raise ZMQError("NOBLOCK and empty queue") from exc
            return ipc.get()
        queue = _QUEUES[self.endpoint]
        if not queue:
            if flags & _NOBLOCK:
                raise ZMQError("NOBLOCK and empty queue")
            raise ZMQError("empty queue")
        return queue.popleft()


def get_zmq_socket(
    context: SimZmqContext,
    socket_type: int,
    endpoint: str,
    bind: bool,
) -> SimZmqSocket:
    sock = context.socket(socket_type)
    if bind:
        sock.bind(endpoint)
    else:
        sock.connect(endpoint)
    return sock
