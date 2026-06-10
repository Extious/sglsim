from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from benchmark.synthesis_requests.utils import (
    DEFAULT_RESULTS_ROOT,
    ClientConfig,
    RequestDetailRow,
    ServerConfig,
    WorkloadRequest,
    generate_shared_prefix_workload,
    load_config_dir,
    new_run_dir,
    sim_stats_to_detail_row,
    write_metrics_json,
    write_request_details_csv,
    write_simulator_config,
)
from benchmark.synthesis_requests.cache_store import load_workload, save_workload
from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.utils.tokenizer_runtime import (
    format_chat_messages,
    measure_encode,
    synthetic_assistant_unit,
)


@contextmanager
def _direct_log_env(run_dir: Path):
    server_log = run_dir / "server.log"
    stderr_log = run_dir / "server.stderr.log"
    server_log.write_text("", encoding="utf-8")
    stderr_log.write_text("", encoding="utf-8")
    updates = {
        "SGLSIM_SERVER_LOG": str(server_log),
        "SGLSIM_LOG_PROFILE": "sglang_projected",
        "SGLSIM_LOG_BASE_TIME_EPOCH_S": str(time.time()),
        "SGLSIM_LOG_MS": "1",
    }
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_direct_workload(
    client: ClientConfig,
    server: ServerConfig,
) -> list[WorkloadRequest]:
    workload = load_workload(client, server)
    if workload is not None:
        return workload
    workload = generate_shared_prefix_workload(client, server)
    save_workload(client, server, workload)
    return workload


def build_direct_gsp_sessions(
    client: ClientConfig,
    server: ServerConfig,
) -> list[list[GenerateReqInput]]:
    """Build GSP sessions with tokenizer-equivalent ids for the direct Engine path."""
    if client.dataset_name != "generated-shared-prefix":
        raise ValueError(
            f"Unsupported dataset_name={client.dataset_name!r}; "
            "only generated-shared-prefix is supported"
        )

    workload = _load_direct_workload(client, server)
    assistant_unit, _ = synthetic_assistant_unit(server.model_path)
    sessions: list[list[GenerateReqInput]] = []
    created = 0.0
    for req in workload:
        session: list[GenerateReqInput] = []
        if isinstance(req.prompt, list):
            turn_prompts = list(req.prompt)
        else:
            turn_prompts = [req.prompt]
        messages: list[dict[str, str]] = []
        for turn, turn_prompt in enumerate(turn_prompts):
            rid = f"{req.rid}-t{turn}" if len(turn_prompts) > 1 else req.rid
            if len(turn_prompts) > 1:
                messages.append({"role": "user", "content": turn_prompt})
                prompt = format_chat_messages(server.model_path, messages)
                input_ids, tokenize_dur = measure_encode(server.model_path, prompt)
            else:
                input_ids, tokenize_dur = measure_encode(
                    server.model_path,
                    turn_prompt,
                )
            session.append(
                GenerateReqInput(
                    rid=rid,
                    input_ids=input_ids,
                    max_new_tokens=int(req.output_len),
                    routing_key=req.routing_key,
                    simulation={
                        "created_time": created,
                        "parallel_worker_client": len(turn_prompts) > 1,
                        "worker_id": req.agent_id or req.rid,
                        "worker_seq": turn,
                        "agent_id": req.agent_id or req.rid,
                        "critical_agent": bool(req.critical_agent),
                        "backup_kv": bool(req.critical_agent),
                        "tokenize_dur": tokenize_dur,
                    },
                )
            )
            if len(turn_prompts) > 1:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_unit * int(req.output_len),
                    }
                )
            created += 0.001
        sessions.append(session)
    return sessions


def _stats_to_row(req: GenerateReqInput, stats, e2e_latency_s: float) -> RequestDetailRow:
    return sim_stats_to_detail_row(
        req.rid,
        stats.to_sim_stats_dict(),
        e2e_latency_s=e2e_latency_s,
    )


def _run_sessions(
    engine: Engine,
    sessions: list[list[GenerateReqInput]],
    *,
    max_concurrency: int | None,
) -> list[RequestDetailRow]:
    limit = max_concurrency if max_concurrency is not None else len(sessions)
    limit = max(int(limit), 1)
    rows: list[RequestDetailRow] = []
    next_session = 0
    active: list[tuple[int, int, GenerateReqInput, float]] = []

    def _submit(session_idx: int, turn_idx: int) -> None:
        req = sessions[session_idx][turn_idx]
        start = time.perf_counter()
        engine.submit(req)
        active.append((session_idx, turn_idx, req, start))

    while next_session < len(sessions) and len(active) < limit:
        _submit(next_session, 0)
        next_session += 1

    def _pop_completed_active() -> tuple[int, int, GenerateReqInput, float]:
        deadline = time.monotonic() + 7200.0
        while True:
            engine._check_scheduler_workers_alive()
            engine._drain_detokenizer(max_messages=512)
            for idx, item in enumerate(active):
                _, _, req, _ = item
                if engine._completion_event(req.rid).is_set():
                    return active.pop(idx)
            if time.monotonic() >= deadline:
                active_rids = [item[2].rid for item in active]
                raise TimeoutError(f"Timed out waiting for active requests: {active_rids}")
            time.sleep(0.01)

    while active:
        session_idx, turn_idx, req, start = _pop_completed_active()
        stats = engine.wait_for_rid(req.rid, timeout_s=0.0)
        rows.append(_stats_to_row(req, stats, time.perf_counter() - start))
        next_turn = turn_idx + 1
        if next_turn < len(sessions[session_idx]):
            _submit(session_idx, next_turn)
        elif next_session < len(sessions):
            _submit(next_session, 0)
            next_session += 1

    rows.sort(key=lambda row: row.rid)
    return rows


def run_direct_experiment(
    config_dir: Path,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    *,
    log_level: str = "warning",
    disable_overlap_schedule: bool = False,
) -> Path:
    client, server = load_config_dir(config_dir)
    run_dir = new_run_dir(results_root)
    sim_config_path = write_simulator_config(
        run_dir,
        client=client,
        server=server,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        disable_overlap_schedule=disable_overlap_schedule or None,
    )
    sessions = build_direct_gsp_sessions(client, server)
    with _direct_log_env(run_dir):
        engine = Engine(
            ServerArgs(
                model_path=server.model_path,
                dp_size=server.dp_size,
                tp_size=server.tp_size,
                pp_size=server.pp_size,
                sim_config_path=str(sim_config_path),
                disable_overlap_schedule=disable_overlap_schedule,
                log_level=log_level,
            )
        )
        try:
            rows = _run_sessions(
                engine,
                sessions,
                max_concurrency=client.max_concurrency,
            )
            result = engine.run_until_idle()
        finally:
            engine.shutdown()

    write_request_details_csv(run_dir, rows)
    write_metrics_json(run_dir, rows)
    (run_dir / "direct_engine_metrics.json").write_text(
        json.dumps(result.metrics, indent=2),
        encoding="utf-8",
    )
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthesis benchmark via Engine, no HTTP socket.")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--log-level", type=str, default="warning")
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    args = parser.parse_args()
    run_dir = run_direct_experiment(
        args.config_dir,
        args.results_root,
        log_level=args.log_level,
        disable_overlap_schedule=args.disable_overlap_schedule,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
