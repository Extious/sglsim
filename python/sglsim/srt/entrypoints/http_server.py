from __future__ import annotations

import argparse
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sglsim.srt.entrypoints.engine import Engine, launch_subprocesses
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.logging import configure_logger
from sglsim.srt.utils.prompt_builder import PromptBuilder
from sglsim.srt.utils.tokenizer_runtime import format_chat_messages

logger = logging.getLogger(__name__)


class ChatMessageBody(BaseModel):
    role: str
    content: str


class CompletionBody(BaseModel):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 1
    temperature: float = 0.0
    stream: bool = False
    ignore_eos: bool = True
    user: str | None = None
    input_ids: list[int] | None = None
    input_len: int | None = None
    simulation: dict[str, Any] = {}


class ChatCompletionBody(BaseModel):
    model: str = ""
    messages: list[ChatMessageBody] = []
    max_completion_tokens: int = 1
    temperature: float = 0.0
    stream: bool = False
    ignore_eos: bool = True
    user: str | None = None
    simulation: dict[str, Any] = {}


class SimulateDpFailureBody(BaseModel):
    dp_rank: int
    event_id: str | None = None


def _fixed_client_settings() -> dict[str, Any]:
    raw = ConfigManager._raw or {}
    fixed = raw.get("client", {}).get("fixed", {})
    return fixed if isinstance(fixed, dict) else {}


def _completion_from_generate(
    *,
    rid: str,
    body_model: str,
    max_tokens: int,
    engine: Engine,
    gen: GenerateReqInput,
) -> dict[str, Any]:
    try:
        stats, text = engine.generate_from_http(gen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    input_tokens = stats.input_length if stats else 0
    finish = stats.last_event_time if stats else 0.0
    sim_stats = stats.to_sim_stats_dict() if stats else {}
    return {
        "id": rid,
        "created": int(finish),
        "model": body_model or engine.server_args.model_path,
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": int(max_tokens),
            "total_tokens": input_tokens + int(max_tokens),
        },
        "sim_stats": sim_stats,
        "completion_text": text,
        "finish": finish,
        "stats": stats,
    }


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="SGLSim")
    app.state.engine = engine

    @app.get("/health")
    def health() -> JSONResponse:
        if not engine.is_ready():
            return JSONResponse(
                status_code=503,
                content={"status": "starting"},
            )
        return JSONResponse(content={"status": "ok"})

    @app.get("/simulate/failure_status")
    def simulate_failure_status() -> dict[str, Any]:
        return engine.soft_failure_status()

    @app.get("/aegis/status")
    def aegis_status(after_sequence: int = 0) -> dict[str, Any]:
        """Expose Router hints and rank-local Algorithm 2/3 metadata for demos."""

        return engine.aegis_status(after_sequence=max(int(after_sequence), 0))

    @app.get("/calibration/telemetry")
    def calibration_telemetry() -> dict[str, Any]:
        try:
            return engine.calibration_telemetry()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/simulate/fail_dp")
    def simulate_fail_dp(body: SimulateDpFailureBody) -> dict[str, Any]:
        try:
            return engine.soft_fail_dp_rank(int(body.dp_rank))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/simulate/recover_dp")
    def simulate_recover_dp(body: SimulateDpFailureBody) -> dict[str, Any]:
        try:
            return engine.recover_soft_failed_dp_rank(int(body.dp_rank))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/simulate/kill_dp")
    def simulate_kill_dp(body: SimulateDpFailureBody) -> dict[str, Any]:
        try:
            return engine.hard_fail_dp_rank(int(body.dp_rank))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/simulate/restart_dp")
    def simulate_restart_dp(body: SimulateDpFailureBody) -> dict[str, Any]:
        try:
            return engine.restart_hard_failed_dp_rank(int(body.dp_rank))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/completions")
    def completions(body: CompletionBody, request: Request) -> dict[str, Any]:
        rid = (body.user or "").strip() or "1"
        fixed = _fixed_client_settings()
        routing_key = request.headers.get("X-SMG-Routing-Key")
        gen = GenerateReqInput(
            rid=rid,
            max_new_tokens=int(body.max_tokens),
            routing_key=routing_key,
            simulation=dict(body.simulation or {}),
        )
        input_tokens = 0
        if body.input_ids is not None:
            gen.input_ids = list(body.input_ids)
            input_tokens = len(gen.input_ids)
        elif body.input_len is not None:
            gen.input_len = int(body.input_len)
            input_tokens = int(body.input_len)
        elif body.prompt.strip():
            gen.prompt = body.prompt
        elif fixed.get("input_len") is not None:
            gen.input_len = int(fixed["input_len"])
            input_tokens = int(fixed["input_len"])
        else:
            req_idx = max(int(rid) - 1, 0)
            input_len = int(fixed.get("input_len", body.max_tokens))
            seed = int(fixed.get("seed", 1))
            builder = PromptBuilder(engine.server_args.model_path, input_len, seed)
            gen.input_ids = builder.build_token_ids(req_idx)
            input_tokens = len(gen.input_ids)
        logger.info(
            "POST /v1/completions rid=%s max_tokens=%d",
            rid,
            int(body.max_tokens),
        )

        result = _completion_from_generate(
            rid=rid,
            body_model=body.model,
            max_tokens=int(body.max_tokens),
            engine=engine,
            gen=gen,
        )
        stats = result["stats"]
        text = result["completion_text"]
        finish = result["finish"]
        if input_tokens == 0:
            input_tokens = stats.input_length if stats else 0
        logger.info(
            "Completed /v1/completions rid=%s output_tokens=%d finish_time=%.3f",
            rid,
            stats.generated_tokens if stats else 0,
            finish,
        )
        logger.info(
            'INFO:     %s:%d - "POST /v1/completions HTTP/1.1" 200 OK',
            engine.server_args.host,
            engine.server_args.port,
        )
        return {
            "id": rid,
            "object": "text_completion",
            "created": int(finish),
            "model": result["model"],
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": int(body.max_tokens),
                "total_tokens": input_tokens + int(body.max_tokens),
            },
            "sim_stats": result["sim_stats"],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionBody, request: Request) -> dict[str, Any]:
        rid = (body.user or "").strip() or "1"
        routing_key = request.headers.get("X-SMG-Routing-Key")
        messages = [
            {"role": str(m.role), "content": str(m.content)} for m in body.messages
        ]
        if not messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        prompt = format_chat_messages(engine.server_args.model_path, messages)
        gen = GenerateReqInput(
            rid=rid,
            prompt=prompt,
            max_new_tokens=int(body.max_completion_tokens),
            routing_key=routing_key,
            simulation=dict(body.simulation or {}),
        )
        logger.info(
            "POST /v1/chat/completions rid=%s turns=%d max_completion_tokens=%d",
            rid,
            len(messages),
            int(body.max_completion_tokens),
        )
        result = _completion_from_generate(
            rid=rid,
            body_model=body.model,
            max_tokens=int(body.max_completion_tokens),
            engine=engine,
            gen=gen,
        )
        stats = result["stats"]
        text = result["completion_text"]
        finish = result["finish"]
        logger.info(
            "Completed /v1/chat/completions rid=%s output_tokens=%d finish_time=%.3f",
            rid,
            stats.generated_tokens if stats else 0,
            finish,
        )
        logger.info(
            'INFO:     %s:%d - "POST /v1/chat/completions HTTP/1.1" 200 OK',
            engine.server_args.host,
            engine.server_args.port,
        )
        return {
            "id": rid,
            "object": "chat.completion",
            "created": int(finish),
            "model": result["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": result["usage"],
            "sim_stats": result["sim_stats"],
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        from sglsim.srt.managers.scheduler import REQUEST_STATS
        from sglsim.srt.observability.bench_metrics import calc_bench_metrics

        engine._drain_detokenizer()
        duration = max((s.last_event_time for s in REQUEST_STATS.values()), default=0.0)
        out = calc_bench_metrics(list(REQUEST_STATS.values()), duration)
        out.update(engine._cache_metrics())
        return out

    return app


def launch_server(server_args: ServerArgs) -> None:
    import uvicorn

    if not server_args.streaming_arrival:
        raise ValueError("launch_server requires streaming_arrival=True")
    configure_logger(server_args)
    state = launch_subprocesses(server_args)
    engine = Engine.from_launch_state(state)
    engine.start_background_scheduler()
    app = create_app(engine)
    app.state.engine = engine
    http_level = server_args.log_level_http or server_args.log_level
    logger.info(
        "Launching SGLSim HTTP server on %s:%s model=%s clock=WALL+STREAMING",
        server_args.host,
        server_args.port,
        server_args.model_path,
    )
    try:
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=http_level,
        )
    finally:
        engine.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv)
    launch_server(ServerArgs.from_cli_args(ns))
