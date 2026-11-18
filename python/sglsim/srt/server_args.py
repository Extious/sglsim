from __future__ import annotations

import argparse
import dataclasses
import uuid
from typing import Optional


@dataclasses.dataclass
class PortArgs:
    tokenizer_ipc_name: str
    scheduler_input_ipc_name: str
    detokenizer_ipc_name: str
    nccl_port: int = 0
    rpc_ipc_name: str = ""
    metrics_ipc_name: str = ""
    tokenizer_worker_ipc_name: Optional[str] = None

    @staticmethod
    def init_new(server_args: ServerArgs) -> PortArgs:
        uid = uuid.uuid4().hex[:8]
        return PortArgs(
            tokenizer_ipc_name=f"inproc://sglsim-tokenizer-{uid}",
            scheduler_input_ipc_name=f"inproc://sglsim-scheduler-{uid}",
            detokenizer_ipc_name=f"inproc://sglsim-detokenizer-{uid}",
            rpc_ipc_name=f"inproc://sglsim-rpc-{uid}",
            metrics_ipc_name=f"inproc://sglsim-metrics-{uid}",
        )

    def scheduler_input_ipc_name_for_dp(self, dp_rank: int) -> str:
        return f"{self.scheduler_input_ipc_name}-dp{dp_rank}"

    def scheduler_input_ipc_name_for_stage(
        self, dp_rank: int, pp_rank: int
    ) -> str:
        """Return the external/ring request endpoint for one PP stage.

        PP0 preserves the existing DP ingress endpoint so the Router continues
        to publish one request to the chosen replica.  Later PP stages receive
        only ordered metadata forwarded through the pipeline runtime.
        """

        if dp_rank < 0 or pp_rank < 0:
            raise ValueError("dp_rank and pp_rank must be nonnegative")
        if pp_rank == 0:
            return self.scheduler_input_ipc_name_for_dp(dp_rank)
        return f"{self.scheduler_input_ipc_name}-dp{dp_rank}-pp{pp_rank}"

    def pp_proxy_ipc_name_for_stage(self, dp_rank: int, pp_rank: int) -> str:
        if dp_rank < 0 or pp_rank < 0:
            raise ValueError("dp_rank and pp_rank must be nonnegative")
        return f"{self.scheduler_input_ipc_name}-dp{dp_rank}-pp{pp_rank}-proxy"

    def pp_output_ipc_name_for_stage(self, dp_rank: int, pp_rank: int) -> str:
        if dp_rank < 0 or pp_rank < 0:
            raise ValueError("dp_rank and pp_rank must be nonnegative")
        return f"{self.scheduler_input_ipc_name}-dp{dp_rank}-pp{pp_rank}-output"

    def detokenizer_reply_ipc_name_for_dp(self, dp_rank: int) -> str:
        return f"{self.detokenizer_ipc_name}-reply-dp{dp_rank}"

    @property
    def detokenizer_reply_ipc_name(self) -> str:
        return self.detokenizer_reply_ipc_name_for_dp(0)


@dataclasses.dataclass
class ServerArgs:
    model_path: str = "mock-model"
    host: str = "127.0.0.1"
    port: int = 30000
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    sim_config_path: Optional[str] = None
    streaming_arrival: bool = True
    disable_overlap_schedule: bool = False
    max_running_requests: int | None = None
    load_balance_method: str | None = None
    chunked_prefill_size: int | None = None
    log_level: str = "info"
    log_level_http: str | None = None
    simulate_detokenizer_time: bool = False
    scheduler_telemetry: bool = False
    # Smoke/fast path: keep configured PP latency metadata but spawn only one
    # Scheduler process per DP (no physical PP worker mesh).
    logical_pp: bool = False

    def __post_init__(self) -> None:
        # Align with SGLang: pipeline-parallel scheduling does not use overlap mode.
        if self.pp_size > 1 and not self.logical_pp:
            self.disable_overlap_schedule = True

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model-path", type=str, default="mock-model")
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=30000)
        parser.add_argument("--dp-size", type=int, default=1)
        parser.add_argument("--tp-size", type=int, default=1)
        parser.add_argument("--pp-size", type=int, default=1)
        parser.add_argument("--sim-config-path", type=str, default=None)
        parser.add_argument(
            "--streaming-arrival",
            action="store_true",
            default=True,
            help="Streaming request arrival: enqueue on submit (required for HTTP server).",
        )
        parser.add_argument(
            "--disable-overlap-schedule",
            action="store_true",
            help="Disable overlap timing (serial load+infer+backup on the global clock).",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=None,
            help="Cap concurrent running requests per DP worker (overrides sim config).",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=None,
            help="DP load balance method (round_robin only in sglsim).",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            type=int,
            default=None,
            help="Chunked prefill token budget (overrides sim config inference).",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default="info",
            help="Python logging level for SGLSim runtime.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=None,
            help="Uvicorn access log level (defaults to --log-level).",
        )
        parser.add_argument(
            "--simulate-detokenizer-time",
            action="store_true",
            help=(
                "Run the real tokenizer at request completion to model "
                "detokenization latency (slow; default off uses zero/analytic "
                "detok time and an async completion path)."
            ),
        )
        parser.add_argument(
            "--scheduler-telemetry",
            action="store_true",
            help=(
                "Collect per-DP scheduler/pipeline rows for online-vs-offline "
                "calibration. Disabled by default to preserve the normal "
                "wall-clock hot path."
            ),
        )
        parser.add_argument(
            "--logical-pp",
            action="store_true",
            help=(
                "Use one Scheduler process per DP even when pp_size>1. "
                "Intended for fast offline smoke; not a physical PP topology."
            ),
        )

    @classmethod
    def from_cli_args(cls, ns: argparse.Namespace) -> ServerArgs:
        return cls(
            model_path=ns.model_path,
            host=ns.host,
            port=ns.port,
            dp_size=ns.dp_size,
            tp_size=ns.tp_size,
            pp_size=ns.pp_size,
            sim_config_path=ns.sim_config_path,
            streaming_arrival=bool(ns.streaming_arrival),
            disable_overlap_schedule=bool(ns.disable_overlap_schedule),
            max_running_requests=ns.max_running_requests,
            load_balance_method=ns.load_balance_method,
            chunked_prefill_size=ns.chunked_prefill_size,
            log_level=str(ns.log_level),
            log_level_http=ns.log_level_http,
            simulate_detokenizer_time=bool(
                getattr(ns, "simulate_detokenizer_time", False)
            ),
            scheduler_telemetry=bool(
                getattr(ns, "scheduler_telemetry", False)
            ),
            logical_pp=bool(getattr(ns, "logical_pp", False)),
        )
