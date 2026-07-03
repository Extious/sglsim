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
    gsp_workflow_shape: tuple[int, ...] = ()
    gsp_stage_long_decode_agent_index: int = 0
    gsp_long_decode_output_multiplier: float = 4.0
    num_prompts: int | None = None
    gsp_critical_agent_ids: tuple[str, ...] = ()
    gsp_critical_agent_ratio: float = 0.0
    gsp_failover_replay: bool = False
    gsp_failover_replay_critical_only: bool = True
    gsp_failover_injection: str = "soft"
    gsp_failover_retry_rank_offset: int = 1
    gsp_failover_retry_policy: str = ""
    gsp_aegis_policy: str = ""
    gsp_aegis_ablation: str = ""
    gsp_aegis_gpu_budget_fraction: float | None = None
    gsp_aegis_critical_pressure: float | None = None
    gsp_alpha_policy_path: str = ""
    gsp_failure_rate_per_window: float | None = None
    gsp_failure_after_output_tokens: int | None = None
    gsp_backup_kv: bool | None = None

    @property
    def num_requests(self) -> int:
        agents_per_group = (
            sum(self.gsp_workflow_shape)
            if self.gsp_workflow_shape
            else self.gsp_prompts_per_group
        )
        total = self.gsp_num_groups * agents_per_group
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
    workflow_id: str = ""
    stage_id: int = -1
    stage_width: int = 1
    stage_local_agent_index: int = 0
    decode_dominant: bool = False
    workflow_shape: str = ""

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
    workflow_id: str
    stage_id: int
    stage_width: int
    stage_local_agent_index: int
    decode_dominant: bool
    agent_id: str
    critical_agent: bool
    e2e_latency_s: float
    ttft_ms: float
    mean_tbt_ms: float
    aegis_policy: str = ""
    aegis_ablation: str = ""
    aegis_gpu_budget_fraction: float = -1.0
    aegis_failure_rate_per_window: float = -1.0
    aegis_alpha_policy_source: str = ""
    aegis_critical_pressure: float = -1.0
    aegis_backup_kv: bool = False
    aegis_worker_seq: int = 0
    aegis_assigned_dp_rank: int = -1
    aegis_host_visible_tokens: int = 0
    aegis_gpu_replica_tokens: int = 0
    aegis_safe_frontier_tokens: int = 0
    aegis_recompute_tokens: int = 0
    aegis_recovery_action: str = "none"
    aegis_expected_recovery_ms: float = 0.0
    aegis_candidate_retry: bool = False
    aegis_candidate_failed_dp_rank: int = -1
    aegis_candidate_pre_failure_output_tokens: int = 0
    aegis_candidate_pre_failure_total_tokens: int = 0
    aegis_candidate_backed_up_tokens: int = 0
    aegis_candidate_recompute_tokens: int = 0
    aegis_candidate_recovery_action: str = "none"
    aegis_candidate_failover_overhead_s: float = 0.0
    aegis_failover_replay: bool = False
    aegis_failed_source_dp_rank: int = -1
    aegis_replay_restored_tokens: int = 0
    aegis_replay_restored_prefill_tokens: int = 0
    aegis_replay_recompute_tokens: int = 0
    aegis_replay_restore_action: str = "none"
    aegis_replay_restore_overhead_s: float = 0.0
    aegis_replay_restore_sleep_s: float = 0.0
    aegis_replay_restore_data_plane: str = "none"
    aegis_replay_restore_timing_source: str = "default_coefficients"
    aegis_cross_rank_retry: bool = False
    aegis_soft_failure_injected: bool = False
    aegis_hard_failure_injected: bool = False
    aegis_host_backup_lookup_hit: bool = False
    aegis_host_backup_lookup_tokens: int = 0
    aegis_peer_gpu_lookup_hit: bool = False
    aegis_peer_gpu_lookup_tokens: int = 0


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
    workflow_shape_raw = raw.get("gsp_workflow_shape", ())
    if isinstance(workflow_shape_raw, str):
        workflow_shape = tuple(
            int(part)
            for part in workflow_shape_raw.replace("-", ",").split(",")
            if part.strip()
        )
    else:
        workflow_shape = tuple(int(x) for x in (workflow_shape_raw or ()))
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
        gsp_workflow_shape=workflow_shape,
        gsp_stage_long_decode_agent_index=int(
            raw.get("gsp_stage_long_decode_agent_index", 0)
        ),
        gsp_long_decode_output_multiplier=float(
            raw.get("gsp_long_decode_output_multiplier", 4.0)
        ),
        num_prompts=(
            int(raw["num_prompts"]) if raw.get("num_prompts") is not None else None
        ),
        gsp_critical_agent_ids=tuple(
            str(x) for x in raw.get("gsp_critical_agent_ids", [])
        ),
        gsp_critical_agent_ratio=float(raw.get("gsp_critical_agent_ratio", 0.0)),
        gsp_failover_replay=bool(raw.get("gsp_failover_replay", False)),
        gsp_failover_replay_critical_only=bool(
            raw.get("gsp_failover_replay_critical_only", True)
        ),
        gsp_failover_injection=str(raw.get("gsp_failover_injection", "soft")),
        gsp_failover_retry_rank_offset=int(
            raw.get("gsp_failover_retry_rank_offset", 1)
        ),
        gsp_failover_retry_policy=str(raw.get("gsp_failover_retry_policy", "")),
        gsp_aegis_policy=str(raw.get("gsp_aegis_policy", "")),
        gsp_aegis_ablation=str(raw.get("gsp_aegis_ablation", "")),
        gsp_aegis_gpu_budget_fraction=(
            float(raw["gsp_aegis_gpu_budget_fraction"])
            if "gsp_aegis_gpu_budget_fraction" in raw
            else None
        ),
        gsp_aegis_critical_pressure=(
            float(raw["gsp_aegis_critical_pressure"])
            if "gsp_aegis_critical_pressure" in raw
            else None
        ),
        gsp_alpha_policy_path=str(raw.get("gsp_alpha_policy_path", "")),
        gsp_failure_rate_per_window=(
            float(raw["gsp_failure_rate_per_window"])
            if "gsp_failure_rate_per_window" in raw
            else None
        ),
        gsp_failure_after_output_tokens=(
            int(raw["gsp_failure_after_output_tokens"])
            if "gsp_failure_after_output_tokens" in raw
            else None
        ),
        gsp_backup_kv=(
            bool(raw["gsp_backup_kv"]) if "gsp_backup_kv" in raw else None
        ),
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
    staged_shape = tuple(int(x) for x in client.gsp_workflow_shape if int(x) > 0)
    agents_per_group = sum(staged_shape) if staged_shape else client.gsp_prompts_per_group
    workflow_shape_label = "-".join(str(x) for x in staged_shape)

    system_prompt_lens = compute_random_lens(
        client.gsp_system_prompt_len, client.gsp_range_ratio, client.gsp_num_groups
    )
    question_lens = np.array(
        compute_random_lens(
            client.gsp_question_len,
            client.gsp_range_ratio,
            client.gsp_num_groups
            * agents_per_group
            * client.gsp_num_turns,
        )
    ).reshape(client.gsp_num_groups, agents_per_group, client.gsp_num_turns)
    output_lens = np.array(
        compute_random_lens(
            client.gsp_output_len,
            client.gsp_range_ratio,
            client.gsp_num_groups * agents_per_group,
        )
    ).reshape(client.gsp_num_groups, agents_per_group)

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
        [[""] * client.gsp_num_turns for _ in range(agents_per_group)]
        for _ in range(client.gsp_num_groups)
    ]
    with tqdm_logging(logger):
        for group_idx in iter_with_progress(
            range(client.gsp_num_groups),
            total=client.gsp_num_groups,
            desc="Generating questions",
            logger=logger,
        ):
            for prompt_idx in range(agents_per_group):
                for turn_idx in range(client.gsp_num_turns):
                    questions[group_idx][prompt_idx][turn_idx] = gen_prompt(
                        tokenizer,
                        int(question_lens[group_idx, prompt_idx, turn_idx]),
                        mock=mock,
                    )

    run_random_str = uuid.uuid4().hex[:8]
    run_start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    draft_rows: list[
        tuple[
            int,
            int,
            str | list[str],
            int,
            int,
            str | None,
            str,
            str,
            int,
            int,
            int,
            bool,
            bool,
            str,
        ]
    ] = []
    rid_counter = 0
    for group_idx in range(client.gsp_num_groups):
        routing_key = (
            f"{run_random_str}_{run_start_timestamp}_{group_idx}"
            if client.gsp_send_routing_key
            else None
        )
        system_prompt = system_prompts[group_idx] if system_prompts else ""
        workflow_id = f"wf-{group_idx}"
        staged_slots: list[tuple[int, int, int]] = []
        if staged_shape:
            for stage_idx, stage_width in enumerate(staged_shape):
                for stage_local_idx in range(stage_width):
                    staged_slots.append((stage_idx, stage_width, stage_local_idx))
        else:
            staged_slots = [
                (0, client.gsp_prompts_per_group, prompt_idx)
                for prompt_idx in range(client.gsp_prompts_per_group)
            ]
        for prompt_idx, (stage_idx, stage_width, stage_local_idx) in enumerate(staged_slots):
            rid_counter += 1
            turn_questions = questions[group_idx][prompt_idx]
            turn_prompts = [f"{system_prompt}\n\n{turn_questions[0]}"] + turn_questions[1:]
            full_prompt = (
                turn_prompts[0] if client.gsp_num_turns == 1 else turn_prompts
            )
            decode_dominant = bool(
                staged_shape
                and stage_width > 1
                and stage_local_idx == client.gsp_stage_long_decode_agent_index
            )
            output_len = int(output_lens[group_idx, prompt_idx])
            if decode_dominant:
                output_len = max(
                    output_len + 1,
                    int(round(output_len * client.gsp_long_decode_output_multiplier)),
                )
            stage_critical = bool(
                staged_shape and (stage_width == 1 or decode_dominant)
            )
            agent_id = (
                f"{workflow_id}-s{stage_idx}-a{stage_local_idx}"
                if staged_shape
                else str(rid_counter)
            )
            draft_rows.append(
                (
                    rid_counter,
                    group_idx,
                    full_prompt,
                    output_len,
                    1 if client.gsp_fast_prepare else -1,
                    routing_key,
                    workflow_id,
                    agent_id,
                    stage_idx,
                    stage_width,
                    stage_local_idx,
                    decode_dominant,
                    stage_critical,
                    workflow_shape_label,
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
        for (
            rid,
            group_id,
            full_prompt,
            output_len,
            _,
            routing_key,
            workflow_id,
            agent_id,
            stage_id,
            stage_width,
            stage_local_agent_index,
            decode_dominant,
            stage_critical,
            row_workflow_shape,
        ) in draft_rows:
            rid_id = str(rid)
            requests.append(
                WorkloadRequest(
                    rid=rid_id,
                    prompt=full_prompt,
                    prompt_len=1,
                    output_len=output_len,
                    group_id=group_id,
                    routing_key=routing_key,
                    agent_id=agent_id,
                    critical_agent=stage_critical or rid_id in critical_ids or agent_id in critical_ids,
                    workflow_id=workflow_id,
                    stage_id=stage_id,
                    stage_width=stage_width,
                    stage_local_agent_index=stage_local_agent_index,
                    decode_dominant=decode_dominant,
                    workflow_shape=row_workflow_shape,
                )
            )
    else:
        with tqdm_logging(logger):
            for (
                rid,
                group_id,
                full_prompt,
                output_len,
                _,
                routing_key,
                workflow_id,
                agent_id,
                stage_id,
                stage_width,
                stage_local_agent_index,
                decode_dominant,
                stage_critical,
                row_workflow_shape,
            ) in iter_with_progress(
                draft_rows,
                total=len(draft_rows),
                desc="Computing prompt lengths",
                logger=logger,
            ):
                rid_id = str(rid)
                first_turn = (
                    full_prompt if isinstance(full_prompt, str) else full_prompt[0]
                )
                prompt_len = _encode_prompt_len(tokenizer, first_turn, mock=mock)
                requests.append(
                    WorkloadRequest(
                        rid=rid_id,
                        prompt=full_prompt,
                        prompt_len=prompt_len,
                        output_len=output_len,
                        group_id=group_id,
                        routing_key=routing_key,
                        agent_id=agent_id,
                        critical_agent=stage_critical or rid_id in critical_ids or agent_id in critical_ids,
                        workflow_id=workflow_id,
                        stage_id=stage_id,
                        stage_width=stage_width,
                        stage_local_agent_index=stage_local_agent_index,
                        decode_dominant=decode_dominant,
                        workflow_shape=row_workflow_shape,
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
        workflow_id=str(sim_stats.get("workflow_id", "")),
        stage_id=int(sim_stats.get("stage_id", -1)),
        stage_width=int(sim_stats.get("stage_width", 1)),
        stage_local_agent_index=int(
            sim_stats.get("stage_local_agent_index", 0)
        ),
        decode_dominant=bool(sim_stats.get("decode_dominant", False)),
        agent_id=str(sim_stats.get("agent_id", "")),
        critical_agent=bool(sim_stats.get("critical_agent", False)),
        e2e_latency_s=e2e_latency_s,
        ttft_ms=ttft_ms,
        mean_tbt_ms=mean_tbt_ms,
        aegis_policy=str(sim_stats.get("aegis_policy", "")),
        aegis_ablation=str(sim_stats.get("aegis_ablation", "")),
        aegis_gpu_budget_fraction=float(
            sim_stats.get("aegis_gpu_budget_fraction", -1.0)
        ),
        aegis_failure_rate_per_window=float(
            sim_stats.get("aegis_failure_rate_per_window", -1.0)
        ),
        aegis_alpha_policy_source=str(sim_stats.get("aegis_alpha_policy_source", "")),
        aegis_critical_pressure=float(sim_stats.get("aegis_critical_pressure", -1.0)),
        aegis_backup_kv=bool(sim_stats.get("aegis_backup_kv", False)),
        aegis_worker_seq=int(sim_stats.get("aegis_worker_seq", 0)),
        aegis_assigned_dp_rank=int(sim_stats.get("aegis_assigned_dp_rank", -1)),
        aegis_host_visible_tokens=int(sim_stats.get("aegis_host_visible_tokens", 0)),
        aegis_gpu_replica_tokens=int(sim_stats.get("aegis_gpu_replica_tokens", 0)),
        aegis_safe_frontier_tokens=int(sim_stats.get("aegis_safe_frontier_tokens", 0)),
        aegis_recompute_tokens=int(sim_stats.get("aegis_recompute_tokens", 0)),
        aegis_recovery_action=str(sim_stats.get("aegis_recovery_action", "none")),
        aegis_expected_recovery_ms=float(
            sim_stats.get("aegis_expected_recovery_ms", 0.0)
        ),
        aegis_candidate_retry=bool(sim_stats.get("aegis_candidate_retry", False)),
        aegis_candidate_failed_dp_rank=int(
            sim_stats.get("aegis_candidate_failed_dp_rank", -1)
        ),
        aegis_candidate_pre_failure_output_tokens=int(
            sim_stats.get("aegis_candidate_pre_failure_output_tokens", 0)
        ),
        aegis_candidate_pre_failure_total_tokens=int(
            sim_stats.get("aegis_candidate_pre_failure_total_tokens", 0)
        ),
        aegis_candidate_backed_up_tokens=int(
            sim_stats.get("aegis_candidate_backed_up_tokens", 0)
        ),
        aegis_candidate_recompute_tokens=int(
            sim_stats.get("aegis_candidate_recompute_tokens", 0)
        ),
        aegis_candidate_recovery_action=str(
            sim_stats.get("aegis_candidate_recovery_action", "none")
        ),
        aegis_candidate_failover_overhead_s=float(
            sim_stats.get("aegis_candidate_failover_overhead_s", 0.0)
        ),
        aegis_failover_replay=bool(sim_stats.get("aegis_failover_replay", False)),
        aegis_failed_source_dp_rank=int(
            sim_stats.get("aegis_failed_source_dp_rank", -1)
        ),
        aegis_replay_restored_tokens=int(
            sim_stats.get("aegis_replay_restored_tokens", 0)
        ),
        aegis_replay_restored_prefill_tokens=int(
            sim_stats.get("aegis_replay_restored_prefill_tokens", 0)
        ),
        aegis_replay_recompute_tokens=int(
            sim_stats.get("aegis_replay_recompute_tokens", 0)
        ),
        aegis_replay_restore_action=str(
            sim_stats.get("aegis_replay_restore_action", "none")
        ),
        aegis_replay_restore_overhead_s=float(
            sim_stats.get("aegis_replay_restore_overhead_s", 0.0)
        ),
        aegis_replay_restore_sleep_s=float(
            sim_stats.get("aegis_replay_restore_sleep_s", 0.0)
        ),
        aegis_replay_restore_data_plane=str(
            sim_stats.get("aegis_replay_restore_data_plane", "none")
        ),
        aegis_replay_restore_timing_source=str(
            sim_stats.get(
                "aegis_replay_restore_timing_source",
                "default_coefficients",
            )
        ),
        aegis_cross_rank_retry=bool(sim_stats.get("aegis_cross_rank_retry", False)),
        aegis_soft_failure_injected=bool(
            sim_stats.get("aegis_soft_failure_injected", False)
        ),
        aegis_hard_failure_injected=bool(
            sim_stats.get("aegis_hard_failure_injected", False)
        ),
        aegis_host_backup_lookup_hit=bool(
            sim_stats.get("aegis_host_backup_lookup_hit", False)
        ),
        aegis_host_backup_lookup_tokens=int(
            sim_stats.get("aegis_host_backup_lookup_tokens", 0)
        ),
        aegis_peer_gpu_lookup_hit=bool(
            sim_stats.get("aegis_peer_gpu_lookup_hit", False)
        ),
        aegis_peer_gpu_lookup_tokens=int(
            sim_stats.get("aegis_peer_gpu_lookup_tokens", 0)
        ),
    )


def write_metrics_json(run_dir: Path, rows: list[RequestDetailRow]) -> Path:
    """Write metrics.json compatible with loop_gate / compare_results."""
    import statistics

    completed = [r for r in rows if r.status == "completed"]
    candidate_retries = [r for r in completed if r.aegis_candidate_retry]
    failover_replays = [r for r in completed if r.aegis_failover_replay]
    cross_rank_replays = [r for r in failover_replays if r.aegis_cross_rank_retry]
    soft_failure_injected = [
        r for r in failover_replays if r.aegis_soft_failure_injected
    ]
    hard_failure_injected = [
        r for r in failover_replays if r.aegis_hard_failure_injected
    ]
    host_backup_lookup_hits = [
        r for r in failover_replays if r.aegis_host_backup_lookup_hit
    ]
    peer_gpu_lookup_hits = [
        r for r in failover_replays if r.aegis_peer_gpu_lookup_hit
    ]
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
        "aegis_candidate_retries": len(candidate_retries),
        "aegis_candidate_total_backed_up_tokens": sum(
            r.aegis_candidate_backed_up_tokens for r in candidate_retries
        ),
        "aegis_candidate_total_recompute_tokens": sum(
            r.aegis_candidate_recompute_tokens for r in candidate_retries
        ),
        "aegis_candidate_mean_failover_overhead_ms": (
            statistics.mean(
                r.aegis_candidate_failover_overhead_s for r in candidate_retries
            )
            * 1000.0
            if candidate_retries
            else 0.0
        ),
        "aegis_failover_replays": len(failover_replays),
        "aegis_cross_rank_replays": len(cross_rank_replays),
        "aegis_soft_failure_injected_replays": len(soft_failure_injected),
        "aegis_hard_failure_injected_replays": len(hard_failure_injected),
        "aegis_host_backup_lookup_hit_replays": len(host_backup_lookup_hits),
        "aegis_host_backup_lookup_tokens": sum(
            r.aegis_host_backup_lookup_tokens for r in failover_replays
        ),
        "aegis_peer_gpu_lookup_hit_replays": len(peer_gpu_lookup_hits),
        "aegis_peer_gpu_lookup_tokens": sum(
            r.aegis_peer_gpu_lookup_tokens for r in failover_replays
        ),
        "aegis_replay_total_restored_tokens": sum(
            r.aegis_replay_restored_tokens for r in failover_replays
        ),
        "aegis_replay_total_restored_prefill_tokens": sum(
            r.aegis_replay_restored_prefill_tokens for r in failover_replays
        ),
        "aegis_replay_total_recompute_tokens": sum(
            r.aegis_replay_recompute_tokens for r in failover_replays
        ),
        "aegis_replay_mean_restore_sleep_ms": (
            statistics.mean(r.aegis_replay_restore_sleep_s for r in failover_replays)
            * 1000.0
            if failover_replays
            else 0.0
        ),
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
        "workflow_id",
        "stage_id",
        "stage_width",
        "stage_local_agent_index",
        "decode_dominant",
        "agent_id",
        "critical_agent",
        "aegis_policy",
        "aegis_ablation",
        "aegis_gpu_budget_fraction",
        "aegis_failure_rate_per_window",
        "aegis_alpha_policy_source",
        "aegis_critical_pressure",
        "aegis_backup_kv",
        "aegis_worker_seq",
        "aegis_assigned_dp_rank",
        "aegis_host_visible_tokens",
        "aegis_gpu_replica_tokens",
        "aegis_safe_frontier_tokens",
        "aegis_recompute_tokens",
        "aegis_recovery_action",
        "aegis_expected_recovery_ms",
        "aegis_candidate_retry",
        "aegis_candidate_failed_dp_rank",
        "aegis_candidate_pre_failure_output_tokens",
        "aegis_candidate_pre_failure_total_tokens",
        "aegis_candidate_backed_up_tokens",
        "aegis_candidate_recompute_tokens",
        "aegis_candidate_recovery_action",
        "aegis_candidate_failover_overhead_s",
        "aegis_failover_replay",
        "aegis_failed_source_dp_rank",
        "aegis_replay_restored_tokens",
        "aegis_replay_restored_prefill_tokens",
        "aegis_replay_recompute_tokens",
        "aegis_replay_restore_action",
        "aegis_replay_restore_overhead_s",
        "aegis_replay_restore_sleep_s",
        "aegis_replay_restore_data_plane",
        "aegis_replay_restore_timing_source",
        "aegis_cross_rank_retry",
        "aegis_soft_failure_injected",
        "aegis_hard_failure_injected",
        "aegis_host_backup_lookup_hit",
        "aegis_host_backup_lookup_tokens",
        "aegis_peer_gpu_lookup_hit",
        "aegis_peer_gpu_lookup_tokens",
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
