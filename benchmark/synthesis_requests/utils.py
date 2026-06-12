from __future__ import annotations

import csv
import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sglsim.srt.utils.prompt_builder import _MOCK_MODEL_PATHS

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = BENCHMARK_DIR / "config"
DEFAULT_RESULTS_ROOT = BENCHMARK_DIR / "results"
LOGS_DIR = BENCHMARK_DIR / "logs"
CLIENT_LOG_PATH = LOGS_DIR / "client.log"
SERVER_LOG_PATH = LOGS_DIR / "server.log"
SERVER_STDERR_LOG_PATH = LOGS_DIR / "server.stderr.log"
ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


@dataclass(frozen=True)
class ClientConfig:
    dataset_name: str
    seed: int
    request_rate: float
    max_concurrency: int | None
    gsp_num_groups: int
    gsp_prompts_per_group: int
    gsp_system_prompt_len: int
    gsp_question_len: int
    gsp_output_len: int
    gsp_range_ratio: float
    gsp_fast_prepare: bool
    gsp_send_routing_key: bool
    gsp_num_turns: int
    gsp_ordered: bool
    num_prompts: int | None = None
    gsp_critical_agent_ids: tuple[str, ...] = ()
    gsp_critical_agent_ratio: float = 0.0

    @property
    def num_requests(self) -> int:
        total = self.gsp_num_groups * self.gsp_prompts_per_group
        if self.num_prompts is None:
            return total
        return min(self.num_prompts, total)


@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    host: str
    port: int
    dp_size: int
    tp_size: int
    pp_size: int
    raw: dict[str, Any]
    experiment_name: str = "synthesis"


@dataclass(frozen=True)
class WorkloadRequest:
    rid: str
    prompt: str | list[str]
    prompt_len: int
    output_len: int
    group_id: int
    routing_key: str | None = None
    input_ids: list[int] | None = None
    agent_id: str = ""
    critical_agent: bool = False

    @property
    def first_turn_prompt(self) -> str:
        if isinstance(self.prompt, list):
            return self.prompt[0]
        return self.prompt


@dataclass
class RequestDetailRow:
    rid: str
    status: str
    created_time_s: float
    queue_start_s: float
    queue_end_s: float
    finish_time_s: float
    waiting_time_s: float
    inference_time_s: float
    total_latency_s: float
    client_wall_e2e_latency_s: float
    logical_total_latency_s: float
    logical_service_time_s: float
    logical_queue_time_s: float
    input_length: int
    output_length: int
    target_output_tokens: int
    generated_tokens: int
    cache_hit_tokens: int
    load_back_tokens: int
    prefetch_complete_tokens: int
    agent_id: str
    critical_agent: bool
    e2e_latency_s: float
    ttft_ms: float
    mean_tbt_ms: float


def setup_file_logger(name: str, log_path: Path, log_level: str = "info") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def read_log_tail(path: Path | None, max_lines: int = 80) -> str:
    if path is None or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def parse_request_rate(value: Any) -> float:
    """Parse request_rate; synthesis benchmark only supports inf (burst at start)."""
    if value is None:
        return float("inf")
    if isinstance(value, str):
        if value.lower() in ("inf", "infinity"):
            return float("inf")
        return float(value)
    return float(value)


def require_request_rate_inf(rate: float) -> None:
    if rate != float("inf"):
        raise ValueError(
            "request_rate must be inf; use max_concurrency to control load"
        )


def load_config_dir(config_dir: Path) -> tuple[ClientConfig, ServerConfig]:
    client = load_client_config(config_dir / "client.json")
    server = load_server_config(config_dir / "server.json")
    return client, server


def load_client_config(path: Path) -> ClientConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    request_rate = parse_request_rate(raw.get("request_rate", float("inf")))
    require_request_rate_inf(request_rate)
    return ClientConfig(
        dataset_name=str(raw.get("dataset_name", "generated-shared-prefix")),
        seed=int(raw.get("seed", 1)),
        request_rate=request_rate,
        max_concurrency=(
            int(raw["max_concurrency"]) if raw.get("max_concurrency") is not None else None
        ),
        gsp_num_groups=int(raw["gsp_num_groups"]),
        gsp_prompts_per_group=int(raw["gsp_prompts_per_group"]),
        gsp_system_prompt_len=int(raw["gsp_system_prompt_len"]),
        gsp_question_len=int(raw["gsp_question_len"]),
        gsp_output_len=int(raw["gsp_output_len"]),
        gsp_range_ratio=float(raw.get("gsp_range_ratio", 1.0)),
        gsp_fast_prepare=bool(raw.get("gsp_fast_prepare", False)),
        gsp_send_routing_key=bool(raw.get("gsp_send_routing_key", False)),
        gsp_num_turns=int(raw.get("gsp_num_turns", 1)),
        gsp_ordered=bool(raw.get("gsp_ordered", False)),
        num_prompts=(
            int(raw["num_prompts"]) if raw.get("num_prompts") is not None else None
        ),
        gsp_critical_agent_ids=tuple(
            str(x) for x in raw.get("gsp_critical_agent_ids", [])
        ),
        gsp_critical_agent_ratio=float(raw.get("gsp_critical_agent_ratio", 0.0)),
    )


def load_server_config(path: Path) -> ServerConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sched = raw.get("scheduler", {})
    return ServerConfig(
        model_path=str(raw["model_path"]),
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 30000)),
        dp_size=int(sched.get("dp_size", 1)),
        tp_size=int(sched.get("tp_size", 1)),
        pp_size=int(sched.get("pp_size", 1)),
        raw=dict(raw),
        experiment_name=str(raw.get("experiment_name", raw.get("name", "synthesis"))),
    )


def new_run_dir(results_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_root) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def materialize_communication_section(server_raw: dict[str, Any]) -> dict[str, Any]:
    from sglsim.srt.communication.config import CommunicationConfig
    from sglsim.srt.communication.kv_bytes import resolve_kv_byte_info
    from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig

    comm = CommunicationConfig.from_dict(
        server_raw.get("communication") if isinstance(server_raw.get("communication"), dict) else {}
    )
    sched_raw = server_raw.get("scheduler", {})
    if not isinstance(sched_raw, dict):
        sched_raw = {}
    sched = SchedulerSimConfig(
        tp_size=int(sched_raw.get("tp_size", 1)),
        pp_size=int(sched_raw.get("pp_size", 1)),
        mem_fraction_static=float(sched_raw.get("mem_fraction_static", 0.88)),
    )
    model_path = str(server_raw.get("model_path", "mock-model"))
    kv_info = resolve_kv_byte_info(
        model_path=model_path,
        sched=sched,
        kv_bytes_per_token_override=comm.kv_bytes_per_token,
    )
    alpha_beta: dict[str, dict[str, float]] = {}
    for path_key in ("h2d", "d2h", "disk_write"):
        cfg = comm.alpha_for(path_key)
        alpha_beta[path_key] = {
            "alpha_us": cfg.alpha_s * 1e6,
            "efficiency": cfg.efficiency,
        }
    return {
        "mode": comm.mode,
        "kv_bytes_per_token": kv_info.bytes_per_token_all_layers,
        "num_hidden_layers": kv_info.num_layers,
        "alpha_beta": alpha_beta,
    }


def write_simulator_config(
    run_dir: Path,
    *,
    client: ClientConfig,
    server: ServerConfig,
    timestamp: str,
    disable_overlap_schedule: bool | None = None,
) -> Path:
    path = run_dir / "simulator_config.json"
    payload = dict(server.raw)
    if disable_overlap_schedule is not None:
        scheduler = dict(payload.get("scheduler") or {})
        scheduler["disable_overlap_schedule"] = bool(disable_overlap_schedule)
        payload["scheduler"] = scheduler
    payload["experiment"] = {
        "timestamp": timestamp,
        "name": server.experiment_name,
    }
    workload = asdict(client)
    workload["num_requests"] = client.num_requests
    workload["request_rate"] = (
        "inf" if client.request_rate == float("inf") else client.request_rate
    )
    payload["workload"] = workload
    payload["communication"] = materialize_communication_section(server.raw)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def compute_random_lens(full_len: int, range_ratio: float, num: int) -> list[int]:
    if full_len <= 0:
        return [0] * num
    return np.random.randint(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    ).tolist()


def _mock_gen_prompt(token_num: int, seed: int) -> str:
    if token_num <= 0:
        return ""
    parts: list[str] = []
    value = seed
    for i in range(token_num):
        value = (value * 131 + i) & 0xFFFFFFFF
        parts.append(f"t{value % 100000:05d}")
    return " ".join(parts)


def gen_prompt(tokenizer: Any | None, token_num: int, *, mock: bool) -> str:
    if token_num <= 0:
        return ""
    if mock:
        return _mock_gen_prompt(token_num, 0)
    all_available_tokens = get_available_tokens(tokenizer)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


@lru_cache(maxsize=4)
def get_available_tokens(tokenizer: Any) -> list[int]:
    return [tid for tid in tokenizer.get_vocab().values() if isinstance(tid, int)]


def _mock_encode(text: str) -> list[int]:
    if not text:
        return [0]
    ids: list[int] = []
    for i in range(0, len(text), 4):
        chunk = text[i : i + 4]
        value = 0
        for ch in chunk:
            value = (value * 131 + ord(ch)) & 0xFFFFFFFF
        ids.append(value % 32000)
    return ids


def _encode_prompt_len(
    tokenizer: Any | None, text: str, *, mock: bool
) -> int:
    if mock or tokenizer is None:
        return max(len(_mock_encode(text)), 1)
    return len(tokenizer.encode(text, add_special_tokens=False))


@lru_cache(maxsize=4)
def get_tokenizer(model_path: str) -> Any | None:
    if model_path in _MOCK_MODEL_PATHS:
        return None
    from sglsim.srt.utils.tokenizer_runtime import ensure_loaded

    return ensure_loaded(model_path)


def generate_shared_prefix_workload(
    client: ClientConfig,
    server: ServerConfig,
    *,
    logger: logging.Logger | None = None,
) -> list[WorkloadRequest]:
    from .progress import iter_with_progress, tqdm_logging

    if client.dataset_name != "generated-shared-prefix":
        raise ValueError(
            f"Unsupported dataset_name={client.dataset_name!r}; "
            "only generated-shared-prefix is supported"
        )

    model_path = server.model_path
    mock = model_path in _MOCK_MODEL_PATHS

    if logger:
        logger.info(
            "Generating GSP workload gsp_num_groups=%d gsp_prompts_per_group=%d "
            "gsp_system_prompt_len=%d gsp_question_len=%d gsp_output_len=%d "
            "gsp_range_ratio=%.3f gsp_num_turns=%d gsp_fast_prepare=%s",
            client.gsp_num_groups,
            client.gsp_prompts_per_group,
            client.gsp_system_prompt_len,
            client.gsp_question_len,
            client.gsp_output_len,
            client.gsp_range_ratio,
            client.gsp_num_turns,
            client.gsp_fast_prepare,
        )

    random.seed(client.seed)
    np.random.seed(client.seed)

    tokenizer = None if mock else get_tokenizer(model_path)

    system_prompt_lens = compute_random_lens(
        client.gsp_system_prompt_len, client.gsp_range_ratio, client.gsp_num_groups
    )
    question_lens = np.array(
        compute_random_lens(
            client.gsp_question_len,
            client.gsp_range_ratio,
            client.gsp_num_groups
            * client.gsp_prompts_per_group
            * client.gsp_num_turns,
        )
    ).reshape(client.gsp_num_groups, client.gsp_prompts_per_group, client.gsp_num_turns)
    output_lens = np.array(
        compute_random_lens(
            client.gsp_output_len,
            client.gsp_range_ratio,
            client.gsp_num_groups * client.gsp_prompts_per_group,
        )
    ).reshape(client.gsp_num_groups, client.gsp_prompts_per_group)

    system_prompts: list[str] = []
    if not client.gsp_fast_prepare:
        with tqdm_logging(logger):
            system_prompts = [
                gen_prompt(tokenizer, int(system_prompt_lens[i]), mock=mock)
                for i in iter_with_progress(
                    range(client.gsp_num_groups),
                    total=client.gsp_num_groups,
                    desc="Generating system prompts",
                    logger=logger,
                )
            ]

    questions: list[list[list[str]]] = [
        [[""] * client.gsp_num_turns for _ in range(client.gsp_prompts_per_group)]
        for _ in range(client.gsp_num_groups)
    ]
    with tqdm_logging(logger):
        for group_idx in iter_with_progress(
            range(client.gsp_num_groups),
            total=client.gsp_num_groups,
            desc="Generating questions",
            logger=logger,
        ):
            for prompt_idx in range(client.gsp_prompts_per_group):
                for turn_idx in range(client.gsp_num_turns):
                    questions[group_idx][prompt_idx][turn_idx] = gen_prompt(
                        tokenizer,
                        int(question_lens[group_idx, prompt_idx, turn_idx]),
                        mock=mock,
                    )

    run_random_str = uuid.uuid4().hex[:8]
    run_start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    draft_rows: list[tuple[int, int, str | list[str], int, int, str | None]] = []
    rid_counter = 0
    for group_idx in range(client.gsp_num_groups):
        routing_key = (
            f"{run_random_str}_{run_start_timestamp}_{group_idx}"
            if client.gsp_send_routing_key
            else None
        )
        system_prompt = system_prompts[group_idx] if system_prompts else ""
        for prompt_idx in range(client.gsp_prompts_per_group):
            rid_counter += 1
            turn_questions = questions[group_idx][prompt_idx]
            turn_prompts = [f"{system_prompt}\n\n{turn_questions[0]}"] + turn_questions[1:]
            full_prompt = (
                turn_prompts[0] if client.gsp_num_turns == 1 else turn_prompts
            )
            draft_rows.append(
                (
                    rid_counter,
                    group_idx,
                    full_prompt,
                    int(output_lens[group_idx, prompt_idx]),
                    1 if client.gsp_fast_prepare else -1,
                    routing_key,
                )
            )

    total_agents = len(draft_rows)
    explicit_critical = {str(x) for x in client.gsp_critical_agent_ids}
    critical_count = int(round(total_agents * client.gsp_critical_agent_ratio))
    critical_count = max(0, min(critical_count, total_agents))
    ratio_critical = {str(row[0]) for row in draft_rows[:critical_count]}
    critical_ids = explicit_critical | ratio_critical

    requests: list[WorkloadRequest] = []
    if client.gsp_fast_prepare:
        for rid, group_id, full_prompt, output_len, _, routing_key in draft_rows:
            agent_id = str(rid)
            requests.append(
                WorkloadRequest(
                    rid=agent_id,
                    prompt=full_prompt,
                    prompt_len=1,
                    output_len=output_len,
                    group_id=group_id,
                    routing_key=routing_key,
                    agent_id=agent_id,
                    critical_agent=agent_id in critical_ids,
                )
            )
    else:
        with tqdm_logging(logger):
            for rid, group_id, full_prompt, output_len, _, routing_key in iter_with_progress(
                draft_rows,
                total=len(draft_rows),
                desc="Computing prompt lengths",
                logger=logger,
            ):
                agent_id = str(rid)
                first_turn = (
                    full_prompt if isinstance(full_prompt, str) else full_prompt[0]
                )
                prompt_len = _encode_prompt_len(tokenizer, first_turn, mock=mock)
                requests.append(
                    WorkloadRequest(
                        rid=agent_id,
                        prompt=full_prompt,
                        prompt_len=prompt_len,
                        output_len=output_len,
                        group_id=group_id,
                        routing_key=routing_key,
                        agent_id=agent_id,
                        critical_agent=agent_id in critical_ids,
                    )
                )

    if client.num_prompts is not None:
        requests = requests[: client.num_prompts]

    if not client.gsp_ordered:
        random.shuffle(requests)

    if logger:
        logger.info("Generated %d GSP requests", len(requests))
    return requests


def sim_stats_to_detail_row(
    rid: str,
    sim_stats: dict[str, Any],
    e2e_latency_s: float,
) -> RequestDetailRow:
    created = float(sim_stats.get("created_time", -1.0))
    queue_start = float(sim_stats.get("queue_start", -1.0))
    queue_end = float(sim_stats.get("queue_end", -1.0))
    finish = float(sim_stats.get("last_event_time", 0.0))
    latencies = list(sim_stats.get("gen_token_latencies") or [])
    waiting = 0.0
    total_latency = 0.0
    if created >= 0 and finish >= 0:
        total_latency = max(finish - created, 0.0)
    if queue_start >= 0 and queue_end >= 0:
        waiting = max(queue_end - queue_start, 0.0)
    inference = 0.0
    if queue_end >= 0 and finish >= 0:
        inference = max(finish - queue_end, 0.0)
    service_time = float(sim_stats.get("service_time_s", 0.0))
    if service_time <= 0.0 and latencies:
        service_time = float(sum(latencies))
    ttft_ms = latencies[0] * 1000.0 if latencies else 0.0
    mean_tbt_ms = 0.0
    if len(latencies) > 1:
        mean_tbt_ms = float(np.mean(latencies[1:])) * 1000.0
    target_output_tokens = int(sim_stats.get("output_length", 0))
    return RequestDetailRow(
        rid=rid,
        status=str(sim_stats.get("status", "")),
        created_time_s=created,
        queue_start_s=queue_start,
        queue_end_s=queue_end,
        finish_time_s=finish,
        waiting_time_s=waiting,
        inference_time_s=inference,
        total_latency_s=total_latency,
        client_wall_e2e_latency_s=e2e_latency_s,
        logical_total_latency_s=total_latency,
        logical_service_time_s=service_time,
        logical_queue_time_s=waiting,
        input_length=int(sim_stats.get("input_length", 0)),
        output_length=target_output_tokens,
        target_output_tokens=target_output_tokens,
        generated_tokens=int(sim_stats.get("generated_tokens", target_output_tokens)),
        cache_hit_tokens=int(sim_stats.get("cache_hit_tokens", 0)),
        load_back_tokens=int(sim_stats.get("load_back_tokens", 0)),
        prefetch_complete_tokens=int(sim_stats.get("prefetch_complete_tokens", 0)),
        agent_id=str(sim_stats.get("agent_id", "")),
        critical_agent=bool(sim_stats.get("critical_agent", False)),
        e2e_latency_s=e2e_latency_s,
        ttft_ms=ttft_ms,
        mean_tbt_ms=mean_tbt_ms,
    )


def write_metrics_json(run_dir: Path, rows: list[RequestDetailRow]) -> Path:
    """Write metrics.json compatible with loop_gate / compare_results."""
    import statistics

    completed = [r for r in rows if r.status == "completed"]
    e2e = [r.e2e_latency_s for r in completed if r.e2e_latency_s > 0]
    created = [r.created_time_s for r in completed if r.created_time_s >= 0]
    finish = [r.finish_time_s for r in completed if r.finish_time_s > 0]
    duration = max(finish) - min(created) if created and finish else 0.0
    ordered = sorted(e2e)
    n = len(ordered)

    def pct(p: float) -> float:
        if n == 0:
            return 0.0
        return ordered[int(p * (n - 1))] * 1000.0

    metrics = {
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "duration": duration,
        "total_input": sum(r.input_length for r in completed),
        "total_output": sum(r.generated_tokens for r in completed),
        "mean_e2e_latency_ms": statistics.mean(e2e) * 1000.0 if e2e else 0.0,
        "median_e2e_latency_ms": statistics.median(e2e) * 1000.0 if e2e else 0.0,
        "p90_e2e_latency_ms": pct(0.9),
        "max_e2e_latency_ms": max(e2e) * 1000.0 if e2e else 0.0,
        "backend": "sglsim",
    }
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def write_request_details_csv(run_dir: Path, rows: list[RequestDetailRow]) -> Path:
    path = run_dir / "request_details.csv"
    fieldnames = [
        "rid",
        "status",
        "created_time_s",
        "queue_start_s",
        "queue_end_s",
        "finish_time_s",
        "waiting_time_s",
        "inference_time_s",
        "total_latency_s",
        "client_wall_e2e_latency_s",
        "logical_total_latency_s",
        "logical_service_time_s",
        "logical_queue_time_s",
        "input_length",
        "output_length",
        "target_output_tokens",
        "generated_tokens",
        "cache_hit_tokens",
        "load_back_tokens",
        "prefetch_complete_tokens",
        "agent_id",
        "critical_agent",
        "e2e_latency_s",
        "ttft_ms",
        "mean_tbt_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path
