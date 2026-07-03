from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from .cache_store import (
    load_prepared,
    load_workload,
    save_prepared,
    save_workload,
    warn_legacy_cache,
)
from .utils import (
    ROUTING_KEY_HEADER,
    ClientConfig,
    RequestDetailRow,
    ServerConfig,
    WorkloadRequest,
    generate_shared_prefix_workload,
    read_log_tail,
    sim_stats_to_detail_row,
)

logger = logging.getLogger("sglsim.benchmark.synthesis_requests.client")
FAILOVER_REPLAY_CONTROL_LOCK = threading.Lock()


@dataclass(frozen=True)
class PreparedRequest:
    rid: str
    prompt: str
    prompt_len: int
    output_len: int
    body: bytes
    routing_key: str | None = None
    turn_prompts: tuple[str, ...] | None = None
    model_path: str = ""
    agent_id: str = ""
    critical_agent: bool = False
    workflow_id: str = ""
    stage_id: int = -1
    stage_width: int = 1
    stage_local_agent_index: int = 0
    decode_dominant: bool = False
    workflow_shape: str = ""
    aegis_policy: str = ""
    aegis_ablation: str = ""
    aegis_gpu_budget_fraction: float | None = None
    aegis_critical_pressure: float | None = None
    failure_rate_per_window: float | None = None
    failure_after_output_tokens: int | None = None
    backup_kv: bool | None = None
    dp_size: int = 1

    @property
    def is_multi_turn(self) -> bool:
        return self.turn_prompts is not None and len(self.turn_prompts) > 1


def _turn_prompts_from_workload(req: WorkloadRequest) -> tuple[str, ...] | None:
    if isinstance(req.prompt, list) and len(req.prompt) > 1:
        return tuple(req.prompt)
    return None


def _backup_kv_for_request(client: ClientConfig, req: WorkloadRequest) -> bool:
    if client.gsp_backup_kv is not None:
        return bool(client.gsp_backup_kv)
    return bool(req.critical_agent)


def _prepared_backup_kv(prepared: PreparedRequest) -> bool:
    if prepared.backup_kv is not None:
        return bool(prepared.backup_kv)
    return bool(prepared.critical_agent)


def _add_policy_to_simulation(
    simulation: dict[str, Any],
    *,
    aegis_policy: str,
    aegis_ablation: str = "",
    aegis_gpu_budget_fraction: float | None = None,
    aegis_critical_pressure: float | None = None,
    failure_rate_per_window: float | None = None,
    failure_after_output_tokens: int | None = None,
) -> dict[str, Any]:
    if aegis_policy:
        simulation["aegis_policy"] = str(aegis_policy)
    if aegis_ablation:
        simulation["aegis_ablation"] = str(aegis_ablation)
    if aegis_gpu_budget_fraction is not None:
        simulation["aegis_gpu_budget_fraction"] = float(aegis_gpu_budget_fraction)
    if aegis_critical_pressure is not None:
        simulation["aegis_critical_pressure"] = float(aegis_critical_pressure)
    if failure_rate_per_window is not None:
        simulation["aegis_failure_rate_per_window"] = float(failure_rate_per_window)
    if failure_after_output_tokens is not None:
        simulation["failure_after_output_tokens"] = int(failure_after_output_tokens)
    return simulation


def build_chat_payload(
    *,
    model_path: str,
    messages: list[dict[str, str]],
    output_len: int,
    rid: str,
    simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model_path,
        "messages": messages,
        "temperature": 0.0,
        "max_completion_tokens": int(output_len),
        "stream": False,
        "ignore_eos": True,
        "user": rid,
    }
    if simulation:
        payload["simulation"] = dict(simulation)
    return payload


def build_prepared_requests_from_workload(
    workload: list[WorkloadRequest],
    client: ClientConfig,
    server: ServerConfig,
) -> list[PreparedRequest]:
    if client.gsp_num_turns > 1 and client.gsp_fast_prepare:
        logger.warning(
            "gsp_num_turns=%d with gsp_fast_prepare; multi-turn uses chat messages "
            "instead of input_len",
            client.gsp_num_turns,
        )

    model_path = server.model_path
    prepared: list[PreparedRequest] = []
    for req in workload:
        turn_prompts = _turn_prompts_from_workload(req)
        if turn_prompts is not None:
            prepared.append(
                PreparedRequest(
                    rid=req.rid,
                    prompt=turn_prompts[0],
                    prompt_len=int(req.prompt_len),
                    output_len=req.output_len,
                    body=b"",
                    routing_key=req.routing_key,
                    turn_prompts=turn_prompts,
                    model_path=model_path,
                    agent_id=req.agent_id or req.rid,
                    critical_agent=bool(req.critical_agent),
                    workflow_id=req.workflow_id,
                    stage_id=int(req.stage_id),
                    stage_width=int(req.stage_width),
                    stage_local_agent_index=int(req.stage_local_agent_index),
                    decode_dominant=bool(req.decode_dominant),
                    workflow_shape=req.workflow_shape,
                    aegis_policy=str(client.gsp_aegis_policy or ""),
                    aegis_ablation=str(client.gsp_aegis_ablation or ""),
                    aegis_gpu_budget_fraction=client.gsp_aegis_gpu_budget_fraction,
                    aegis_critical_pressure=client.gsp_aegis_critical_pressure,
                    failure_rate_per_window=client.gsp_failure_rate_per_window,
                    failure_after_output_tokens=client.gsp_failure_after_output_tokens,
                    backup_kv=_backup_kv_for_request(client, req),
                    dp_size=server.dp_size,
                )
            )
            continue

        turn_prompt = req.first_turn_prompt
        simulation = _add_policy_to_simulation(
            {
                "parallel_worker_client": False,
                "worker_id": req.agent_id or req.rid,
                "worker_seq": 0,
                "agent_id": req.agent_id or req.rid,
                "critical_agent": bool(req.critical_agent),
                "workflow_id": req.workflow_id,
                "stage_id": int(req.stage_id),
                "stage_width": int(req.stage_width),
                "stage_local_agent_index": int(req.stage_local_agent_index),
                "decode_dominant": bool(req.decode_dominant),
                "workflow_shape": req.workflow_shape,
                "backup_kv": _backup_kv_for_request(client, req),
            },
            aegis_policy=str(client.gsp_aegis_policy or ""),
            aegis_ablation=str(client.gsp_aegis_ablation or ""),
            aegis_gpu_budget_fraction=client.gsp_aegis_gpu_budget_fraction,
            aegis_critical_pressure=client.gsp_aegis_critical_pressure,
            failure_rate_per_window=client.gsp_failure_rate_per_window,
            failure_after_output_tokens=client.gsp_failure_after_output_tokens,
        )
        payload: dict[str, Any] = {
            "model": model_path,
            "temperature": 0.0,
            "max_tokens": int(req.output_len),
            "stream": False,
            "ignore_eos": True,
            "user": req.rid,
            "simulation": simulation,
        }
        if client.gsp_fast_prepare:
            payload["input_len"] = int(req.prompt_len)
            actual_prompt_len = int(req.prompt_len)
        else:
            payload["prompt"] = turn_prompt
            actual_prompt_len = int(req.prompt_len)
        prepared.append(
            PreparedRequest(
                rid=req.rid,
                prompt=turn_prompt,
                prompt_len=actual_prompt_len,
                output_len=req.output_len,
                body=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
                routing_key=req.routing_key,
                model_path=model_path,
                agent_id=req.agent_id or req.rid,
                critical_agent=bool(req.critical_agent),
                workflow_id=req.workflow_id,
                stage_id=int(req.stage_id),
                stage_width=int(req.stage_width),
                stage_local_agent_index=int(req.stage_local_agent_index),
                decode_dominant=bool(req.decode_dominant),
                workflow_shape=req.workflow_shape,
                aegis_policy=str(client.gsp_aegis_policy or ""),
                aegis_ablation=str(client.gsp_aegis_ablation or ""),
                aegis_gpu_budget_fraction=client.gsp_aegis_gpu_budget_fraction,
                aegis_critical_pressure=client.gsp_aegis_critical_pressure,
                failure_rate_per_window=client.gsp_failure_rate_per_window,
                failure_after_output_tokens=client.gsp_failure_after_output_tokens,
                backup_kv=_backup_kv_for_request(client, req),
                dp_size=server.dp_size,
            )
        )
    return prepared


def _attach_server_context(
    prepared: list[PreparedRequest],
    server: ServerConfig,
) -> list[PreparedRequest]:
    out: list[PreparedRequest] = []
    for item in prepared:
        out.append(
            PreparedRequest(
                rid=item.rid,
                prompt=item.prompt,
                prompt_len=item.prompt_len,
                output_len=item.output_len,
                body=item.body,
                routing_key=item.routing_key,
                turn_prompts=item.turn_prompts,
                model_path=item.model_path or server.model_path,
                agent_id=item.agent_id,
                critical_agent=bool(item.critical_agent),
                workflow_id=getattr(item, "workflow_id", ""),
                stage_id=int(getattr(item, "stage_id", -1)),
                stage_width=int(getattr(item, "stage_width", 1)),
                stage_local_agent_index=int(
                    getattr(item, "stage_local_agent_index", 0)
                ),
                decode_dominant=bool(getattr(item, "decode_dominant", False)),
                workflow_shape=str(getattr(item, "workflow_shape", "") or ""),
                aegis_policy=str(getattr(item, "aegis_policy", "") or ""),
                aegis_ablation=str(getattr(item, "aegis_ablation", "") or ""),
                aegis_gpu_budget_fraction=getattr(
                    item, "aegis_gpu_budget_fraction", None
                ),
                aegis_critical_pressure=getattr(
                    item, "aegis_critical_pressure", None
                ),
                failure_rate_per_window=getattr(
                    item, "failure_rate_per_window", None
                ),
                failure_after_output_tokens=getattr(
                    item, "failure_after_output_tokens", None
                ),
                backup_kv=getattr(item, "backup_kv", None),
                dp_size=int(server.dp_size),
            )
        )
    return out


def build_prepared_requests(
    client: ClientConfig,
    server: ServerConfig,
) -> list[PreparedRequest]:
    warn_legacy_cache(logger)

    cached_prepared = load_prepared(client, server, logger=logger)
    if cached_prepared is not None:
        return _attach_server_context(cached_prepared, server)

    workload = load_workload(client, server, logger=logger)
    generated_workload = workload is None
    if workload is None:
        workload = generate_shared_prefix_workload(client, server, logger=logger)
        save_workload(client, server, workload, logger=logger)

    prepared = build_prepared_requests_from_workload(workload, client, server)
    save_prepared(client, server, prepared, logger=logger)
    if generated_workload:
        logger.info(
            "Built %d prepared requests from generated workload", len(prepared)
        )
    else:
        logger.info(
            "Built %d prepared requests from cached workload", len(prepared)
        )
    return prepared


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{int(port)}"


def _raise_if_process_exited(process: Any | None, log_path) -> None:
    if process is None:
        return
    exit_code = process.poll()
    if exit_code is None:
        return
    message = f"Server exited before readiness with exit code {exit_code}."
    log_tail = read_log_tail(log_path)
    if log_tail:
        message = f"{message}\nLast server log lines:\n{log_tail}"
    raise RuntimeError(message)


def wait_for_server_ready(
    host: str,
    port: int,
    timeout_s: float = 300.0,
    *,
    process: Any | None = None,
    log_path=None,
) -> None:
    url = f"{_base_url(host, port)}/health"
    logger.info("Waiting for server readiness at %s", url)
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        _raise_if_process_exited(process, log_path)
        try:
            req = urllib.request.Request(url, method="GET")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=5) as resp:
                if resp.status < 500:
                    logger.info("Server is ready at %s", url)
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
        _raise_if_process_exited(process, log_path)
        time.sleep(1)
    message = f"Timed out waiting for server readiness at {url} after {timeout_s}s"
    if last_error is not None:
        message = f"{message}; last error: {last_error}"
    log_tail = read_log_tail(log_path)
    if log_tail:
        message = f"{message}\nLast server log lines:\n{log_tail}"
    raise TimeoutError(message)


def _post_json_sync(
    url: str,
    body: bytes,
    *,
    routing_key: str | None,
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float]:
    headers = {"Content-Type": "application/json"}
    if routing_key:
        headers[ROUTING_KEY_HEADER] = routing_key
    start = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout_s) as resp:
        if resp.status >= 400:
            raise urllib.error.HTTPError(
                url, resp.status, resp.reason, resp.headers, resp.read()
            )
        raw = json.loads(resp.read().decode("utf-8"))
    e2e_latency_s = time.perf_counter() - start
    return (raw if isinstance(raw, dict) else {}), e2e_latency_s


def post_completions_sync(
    host: str,
    port: int,
    prepared: PreparedRequest,
    *,
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float]:
    url = f"{_base_url(host, port)}/v1/completions"
    return _post_json_sync(
        url,
        prepared.body,
        routing_key=prepared.routing_key,
        timeout_s=timeout_s,
    )


def post_chat_completions_sync(
    host: str,
    port: int,
    *,
    body: bytes,
    routing_key: str | None,
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float]:
    url = f"{_base_url(host, port)}/v1/chat/completions"
    return _post_json_sync(
        url,
        body,
        routing_key=routing_key,
        timeout_s=timeout_s,
    )


def set_soft_failed_dp_rank_sync(
    host: str,
    port: int,
    *,
    dp_rank: int,
    failed: bool,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    path = "/simulate/fail_dp" if failed else "/simulate/recover_dp"
    url = f"{_base_url(host, port)}{path}"
    body = json.dumps({"dp_rank": int(dp_rank)}, ensure_ascii=True).encode("utf-8")
    response, _ = _post_json_sync(
        url,
        body,
        routing_key=None,
        timeout_s=timeout_s,
    )
    return response


def set_hard_failed_dp_rank_sync(
    host: str,
    port: int,
    *,
    dp_rank: int,
    failed: bool,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    path = "/simulate/kill_dp" if failed else "/simulate/restart_dp"
    url = f"{_base_url(host, port)}{path}"
    body = json.dumps({"dp_rank": int(dp_rank)}, ensure_ascii=True).encode("utf-8")
    response, _ = _post_json_sync(
        url,
        body,
        routing_key=None,
        timeout_s=timeout_s,
    )
    return response


def _body_with_failure_injected(
    body: bytes,
    *,
    soft_injected: bool,
    hard_injected: bool,
) -> bytes:
    payload = json.loads(body.decode("utf-8"))
    sim = dict(payload.get("simulation") or {})
    sim["soft_failure_injected"] = bool(soft_injected)
    sim["hard_failure_injected"] = bool(hard_injected)
    payload["simulation"] = sim
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def post_chat_failover_replay_sync(
    host: str,
    port: int,
    *,
    body: bytes,
    routing_key: str | None,
    failed_rank: int,
    injection: str = "soft",
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float, bool]:
    if failed_rank < 0:
        response, latency_s = post_chat_completions_sync(
            host,
            port,
            body=body,
            routing_key=routing_key,
            timeout_s=timeout_s,
        )
        return response, latency_s, False
    mode = injection.strip().lower()
    with FAILOVER_REPLAY_CONTROL_LOCK:
        hard_mode = mode in {"hard", "hard-restart", "kill-restart"}
        if hard_mode:
            status = set_hard_failed_dp_rank_sync(
                host,
                port,
                dp_rank=failed_rank,
                failed=True,
            )
        else:
            status = set_soft_failed_dp_rank_sync(
                host,
                port,
                dp_rank=failed_rank,
                failed=True,
            )
        acquired = bool(status.get("accepted"))
        replay_body = _body_with_failure_injected(
            body,
            soft_injected=acquired and not hard_mode,
            hard_injected=acquired and hard_mode,
        )
        try:
            response, latency_s = post_chat_completions_sync(
                host,
                port,
                body=replay_body,
                routing_key=routing_key,
                timeout_s=timeout_s,
            )
        finally:
            if acquired:
                if hard_mode:
                    set_hard_failed_dp_rank_sync(
                        host,
                        port,
                        dp_rank=failed_rank,
                        failed=False,
                    )
                else:
                    set_soft_failed_dp_rank_sync(
                        host,
                        port,
                        dp_rank=failed_rank,
                        failed=False,
                    )
        return response, latency_s, acquired


def post_completion_failover_replay_sync(
    host: str,
    port: int,
    prepared: PreparedRequest,
    *,
    failed_rank: int,
    injection: str = "soft",
    timeout_s: float = 7200.0,
) -> tuple[dict[str, Any], float, bool]:
    if failed_rank < 0:
        response, latency_s = post_completions_sync(
            host,
            port,
            prepared,
            timeout_s=timeout_s,
        )
        return response, latency_s, False
    mode = injection.strip().lower()
    with FAILOVER_REPLAY_CONTROL_LOCK:
        hard_mode = mode in {"hard", "hard-restart", "kill-restart"}
        if hard_mode:
            status = set_hard_failed_dp_rank_sync(
                host,
                port,
                dp_rank=failed_rank,
                failed=True,
            )
        else:
            status = set_soft_failed_dp_rank_sync(
                host,
                port,
                dp_rank=failed_rank,
                failed=True,
            )
        acquired = bool(status.get("accepted"))
        replay = replace(
            prepared,
            body=_body_with_failure_injected(
                prepared.body,
                soft_injected=acquired and not hard_mode,
                hard_injected=acquired and hard_mode,
            ),
        )
        try:
            response, latency_s = post_completions_sync(
                host,
                port,
                replay,
                timeout_s=timeout_s,
            )
        finally:
            if acquired:
                if hard_mode:
                    set_hard_failed_dp_rank_sync(
                        host,
                        port,
                        dp_rank=failed_rank,
                        failed=False,
                    )
                else:
                    set_soft_failed_dp_rank_sync(
                        host,
                        port,
                        dp_rank=failed_rank,
                        failed=False,
                    )
        return response, latency_s, acquired


def _assistant_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _should_failover_replay(
    prepared: PreparedRequest,
    *,
    failover_replay: bool,
    failover_replay_critical_only: bool,
) -> bool:
    if not failover_replay:
        return False
    if failover_replay_critical_only and not prepared.critical_agent:
        return False
    return True


def _candidate_retry_enabled(sim_stats: dict[str, Any]) -> bool:
    value = sim_stats.get("aegis_candidate_retry", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _select_retry_rank(
    prepared: PreparedRequest,
    *,
    turn_idx: int,
    failed_rank: int,
    dp_size: int,
    retry_rank_offset: int,
    retry_policy: str,
) -> int:
    if failed_rank < 0 or dp_size <= 1:
        return failed_rank
    policy = str(retry_policy or "offset").strip().lower()
    if policy in {"", "offset", "fixed_offset"}:
        return (failed_rank + int(retry_rank_offset)) % dp_size
    if policy in {"peer", "peer_rank", "aegis_peer"}:
        return (failed_rank + 1) % dp_size
    if policy in {"non_peer", "nonpeer", "route_miss"}:
        return (failed_rank + 2) % dp_size if dp_size > 2 else (failed_rank + 1) % dp_size
    if policy in {"random_healthy", "random", "random_route"}:
        choices = [rank for rank in range(dp_size) if rank != failed_rank]
        key = f"{prepared.rid}:{turn_idx}:{failed_rank}:{dp_size}".encode("utf-8")
        index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(choices)
        return choices[index]
    raise ValueError(f"unknown failover retry policy: {retry_policy!r}")


def _failover_replay_simulation(
    prepared: PreparedRequest,
    *,
    turn_idx: int,
    sim_stats: dict[str, Any],
    retry_rank_offset: int = 1,
    retry_policy: str = "",
) -> dict[str, Any]:
    failed_rank = sim_stats.get(
        "aegis_candidate_failed_dp_rank",
        sim_stats.get("aegis_assigned_dp_rank", -1),
    )
    failure_after_output = sim_stats.get(
        "aegis_candidate_pre_failure_output_tokens",
        0,
    )
    try:
        failed_rank_int = int(failed_rank)
    except (TypeError, ValueError):
        failed_rank_int = -1
    dp_size = max(1, int(getattr(prepared, "dp_size", 1) or 1))
    retry_rank = _select_retry_rank(
        prepared,
        turn_idx=turn_idx,
        failed_rank=failed_rank_int,
        dp_size=dp_size,
        retry_rank_offset=retry_rank_offset,
        retry_policy=retry_policy,
    )
    policy = str(prepared.aegis_policy or "aegiskv")
    simulation = {
        "parallel_worker_client": True,
        "worker_id": str(prepared.agent_id or prepared.rid),
        "worker_seq": int(turn_idx),
        "agent_id": str(prepared.agent_id or prepared.rid),
        "critical_agent": bool(prepared.critical_agent),
        "workflow_id": str(prepared.workflow_id or ""),
        "stage_id": int(prepared.stage_id),
        "stage_width": int(prepared.stage_width),
        "stage_local_agent_index": int(prepared.stage_local_agent_index),
        "decode_dominant": bool(prepared.decode_dominant),
        "workflow_shape": str(prepared.workflow_shape or ""),
        "backup_kv": _prepared_backup_kv(prepared),
        "aegis_policy": policy,
        "failover_replay": True,
        "failed_dp_rank": failed_rank,
        "soft_failure_injected": False,
        "hard_failure_injected": False,
        "force_dp_rank": retry_rank,
        "retry_dp_rank": retry_rank,
        "retry_route_policy": str(retry_policy or "offset"),
        "failure_after_output_tokens": failure_after_output,
        "pre_failure_total_tokens": sim_stats.get(
            "aegis_candidate_pre_failure_total_tokens",
            0,
        ),
    }
    if prepared.aegis_ablation:
        simulation["aegis_ablation"] = str(prepared.aegis_ablation)
    if prepared.aegis_gpu_budget_fraction is not None:
        simulation["aegis_gpu_budget_fraction"] = float(
            prepared.aegis_gpu_budget_fraction
        )
    if prepared.aegis_critical_pressure is not None:
        simulation["aegis_critical_pressure"] = float(
            prepared.aegis_critical_pressure
        )
    if prepared.failure_rate_per_window is not None:
        simulation["aegis_failure_rate_per_window"] = float(
            prepared.failure_rate_per_window
        )
    return simulation


def run_multi_turn_session_sync(
    host: str,
    port: int,
    prepared: PreparedRequest,
    *,
    timeout_s: float = 7200.0,
    failover_replay: bool = False,
    failover_replay_critical_only: bool = True,
    failover_injection: str = "soft",
    failover_retry_rank_offset: int = 1,
    failover_retry_policy: str = "",
) -> list[tuple[str, dict[str, Any], float]]:
    if prepared.turn_prompts is None:
        raise ValueError("run_multi_turn_session_sync requires turn_prompts")
    messages: list[dict[str, str]] = []
    outputs: list[tuple[str, dict[str, Any], float]] = []
    for turn_idx, turn_prompt in enumerate(prepared.turn_prompts):
        turn_rid = f"{prepared.rid}-t{turn_idx}"
        messages.append({"role": "user", "content": turn_prompt})
        simulation = _add_policy_to_simulation(
            {
                "parallel_worker_client": True,
                "worker_id": str(prepared.agent_id or prepared.rid),
                "worker_seq": int(turn_idx),
                "agent_id": str(prepared.agent_id or prepared.rid),
                "critical_agent": bool(prepared.critical_agent),
                "workflow_id": str(prepared.workflow_id or ""),
                "stage_id": int(prepared.stage_id),
                "stage_width": int(prepared.stage_width),
                "stage_local_agent_index": int(prepared.stage_local_agent_index),
                "decode_dominant": bool(prepared.decode_dominant),
                "workflow_shape": str(prepared.workflow_shape or ""),
                "backup_kv": _prepared_backup_kv(prepared),
            },
            aegis_policy=str(prepared.aegis_policy or ""),
            aegis_ablation=str(prepared.aegis_ablation or ""),
            aegis_gpu_budget_fraction=prepared.aegis_gpu_budget_fraction,
            aegis_critical_pressure=prepared.aegis_critical_pressure,
            failure_rate_per_window=prepared.failure_rate_per_window,
            failure_after_output_tokens=prepared.failure_after_output_tokens,
        )
        payload = build_chat_payload(
            model_path=prepared.model_path,
            messages=messages,
            output_len=prepared.output_len,
            rid=turn_rid,
            simulation=simulation,
        )
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        response, e2e_latency_s = post_chat_completions_sync(
            host,
            port,
            body=body,
            routing_key=prepared.routing_key,
            timeout_s=timeout_s,
        )
        outputs.append((turn_rid, response, e2e_latency_s))
        response_for_history = response
        sim_stats = response.get("sim_stats") or {}
        if (
            _should_failover_replay(
                prepared,
                failover_replay=failover_replay,
                failover_replay_critical_only=failover_replay_critical_only,
            )
            and _candidate_retry_enabled(sim_stats)
        ):
            replay_rid = f"{turn_rid}-retry"
            replay_sim = _failover_replay_simulation(
                prepared,
                turn_idx=turn_idx,
                sim_stats=sim_stats,
                retry_rank_offset=failover_retry_rank_offset,
                retry_policy=failover_retry_policy,
            )
            failed_rank = int(replay_sim.get("failed_dp_rank", -1))
            replay_payload = build_chat_payload(
                model_path=prepared.model_path,
                messages=messages,
                output_len=prepared.output_len,
                rid=replay_rid,
                simulation=replay_sim,
            )
            replay_body = json.dumps(
                replay_payload, ensure_ascii=True
            ).encode("utf-8")
            replay_response, replay_latency_s, _ = post_chat_failover_replay_sync(
                host,
                port,
                body=replay_body,
                routing_key=prepared.routing_key,
                failed_rank=failed_rank,
                injection=failover_injection,
                timeout_s=timeout_s,
            )
            outputs.append((replay_rid, replay_response, replay_latency_s))
            response_for_history = replay_response
        messages.append(
            {"role": "assistant", "content": _assistant_content(response_for_history)}
        )
    return outputs


async def _bounded_post(
    host: str,
    port: int,
    prepared: PreparedRequest,
    semaphore: asyncio.Semaphore | None,
    failover_replay: bool,
    failover_replay_critical_only: bool,
    failover_injection: str,
    failover_retry_rank_offset: int,
    failover_retry_policy: str,
    progress: dict[str, int] | None = None,
) -> list[RequestDetailRow]:
    async def _run() -> list[RequestDetailRow]:
        if prepared.is_multi_turn:
            turn_outputs = await asyncio.to_thread(
                run_multi_turn_session_sync,
                host,
                port,
                prepared,
                failover_replay=failover_replay,
                failover_replay_critical_only=failover_replay_critical_only,
                failover_injection=failover_injection,
                failover_retry_rank_offset=failover_retry_rank_offset,
                failover_retry_policy=failover_retry_policy,
            )
            rows = []
            for turn_rid, response, e2e_latency_s in turn_outputs:
                sim_stats = response.get("sim_stats") or {}
                rows.append(
                    sim_stats_to_detail_row(turn_rid, sim_stats, e2e_latency_s)
                )
            return rows
        response, e2e_latency_s = await asyncio.to_thread(
            post_completions_sync, host, port, prepared
        )
        sim_stats = response.get("sim_stats") or {}
        rows = [sim_stats_to_detail_row(prepared.rid, sim_stats, e2e_latency_s)]
        if (
            _should_failover_replay(
                prepared,
                failover_replay=failover_replay,
                failover_replay_critical_only=failover_replay_critical_only,
            )
            and _candidate_retry_enabled(sim_stats)
        ):
            payload = json.loads(prepared.body.decode("utf-8"))
            replay_rid = f"{prepared.rid}-retry"
            payload["user"] = replay_rid
            replay_sim = _failover_replay_simulation(
                prepared,
                turn_idx=0,
                sim_stats=sim_stats,
                retry_rank_offset=failover_retry_rank_offset,
                retry_policy=failover_retry_policy,
            )
            failed_rank = int(replay_sim.get("failed_dp_rank", -1))
            payload["simulation"] = replay_sim
            replay = PreparedRequest(
                rid=replay_rid,
                prompt=prepared.prompt,
                prompt_len=prepared.prompt_len,
                output_len=prepared.output_len,
                body=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
                routing_key=prepared.routing_key,
                turn_prompts=prepared.turn_prompts,
                model_path=prepared.model_path,
                agent_id=prepared.agent_id,
                critical_agent=prepared.critical_agent,
                workflow_id=prepared.workflow_id,
                stage_id=prepared.stage_id,
                stage_width=prepared.stage_width,
                stage_local_agent_index=prepared.stage_local_agent_index,
                decode_dominant=prepared.decode_dominant,
                workflow_shape=prepared.workflow_shape,
                aegis_policy=str(prepared.aegis_policy or ""),
                aegis_ablation=str(prepared.aegis_ablation or ""),
                aegis_gpu_budget_fraction=prepared.aegis_gpu_budget_fraction,
                aegis_critical_pressure=prepared.aegis_critical_pressure,
                failure_rate_per_window=prepared.failure_rate_per_window,
                failure_after_output_tokens=prepared.failure_after_output_tokens,
                backup_kv=prepared.backup_kv,
                dp_size=int(getattr(prepared, "dp_size", 1) or 1),
            )
            replay_response, replay_e2e_latency_s, _ = await asyncio.to_thread(
                post_completion_failover_replay_sync,
                host,
                port,
                replay,
                failed_rank=failed_rank,
                injection=failover_injection,
            )
            rows.append(
                sim_stats_to_detail_row(
                    replay_rid,
                    replay_response.get("sim_stats") or {},
                    replay_e2e_latency_s,
                )
            )
        return rows

    if semaphore is not None:
        async with semaphore:
            rows = await _run()
    else:
        rows = await _run()

    if progress is not None:
        progress["done"] += 1
        logger.info(
            "Completed %d/%d sessions (rid=%s turns=%d)",
            progress["done"],
            progress["total"],
            prepared.rid,
            len(rows),
        )
    return rows


def _rid_sort_key(rid: str) -> tuple[Any, ...]:
    retry = 1 if rid.endswith("-retry") else 0
    core = rid[: -len("-retry")] if retry else rid

    def part(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    if "-t" in core:
        base, turn = core.rsplit("-t", 1)
        return (part(base), part(turn), retry)
    return (part(core), (0, 0), retry)


def _is_staged_workflow_request(prepared: PreparedRequest) -> bool:
    return bool(prepared.workflow_id) and int(prepared.stage_id) >= 0


def _stage_sort_key(prepared: PreparedRequest) -> tuple[int, int, tuple[Any, ...]]:
    return (
        int(prepared.stage_id) if int(prepared.stage_id) >= 0 else 0,
        int(prepared.stage_local_agent_index),
        _rid_sort_key(prepared.rid),
    )


def _group_prepared_workflows(
    prepared_requests: list[PreparedRequest],
) -> list[list[PreparedRequest]]:
    grouped: dict[str, list[PreparedRequest]] = defaultdict(list)
    order: list[str] = []
    for index, prepared in enumerate(prepared_requests):
        key = (
            str(prepared.workflow_id)
            if _is_staged_workflow_request(prepared)
            else f"__single_{index}_{prepared.rid}"
        )
        if key not in grouped:
            order.append(key)
        grouped[key].append(prepared)
    return [sorted(grouped[key], key=_stage_sort_key) for key in order]


async def _run_prepared_workflow_stages(
    workflow_items: list[PreparedRequest],
    host: str,
    port: int,
    semaphore: asyncio.Semaphore | None,
    failover_replay: bool,
    failover_replay_critical_only: bool,
    failover_injection: str,
    failover_retry_rank_offset: int,
    failover_retry_policy: str,
    progress: dict[str, int] | None,
) -> list[RequestDetailRow]:
    by_stage: dict[int, list[PreparedRequest]] = defaultdict(list)
    for prepared in workflow_items:
        stage_id = int(prepared.stage_id) if int(prepared.stage_id) >= 0 else 0
        by_stage[stage_id].append(prepared)

    rows: list[RequestDetailRow] = []
    for stage_id in sorted(by_stage):
        stage_tasks = [
            asyncio.create_task(
                _bounded_post(
                    host,
                    port,
                    prepared,
                    semaphore,
                    failover_replay,
                    failover_replay_critical_only,
                    failover_injection,
                    failover_retry_rank_offset,
                    failover_retry_policy,
                    progress,
                ),
                name=f"gsp-stage-{prepared.workflow_id or prepared.rid}-{stage_id}-{prepared.rid}",
            )
            for prepared in sorted(by_stage[stage_id], key=_stage_sort_key)
        ]
        nested = await asyncio.gather(*stage_tasks)
        rows.extend(row for session_rows in nested for row in session_rows)
    return rows


async def _run_client_async(
    host: str,
    port: int,
    prepared_requests: list[PreparedRequest],
    *,
    max_concurrency: int | None,
    failover_replay: bool,
    failover_replay_critical_only: bool,
    failover_injection: str,
    failover_retry_rank_offset: int,
    failover_retry_policy: str,
) -> list[RequestDetailRow]:
    semaphore = (
        asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
    )
    progress = {"done": 0, "total": len(prepared_requests)}
    if any(_is_staged_workflow_request(p) for p in prepared_requests):
        nested = await asyncio.gather(
            *[
                _run_prepared_workflow_stages(
                    workflow_items,
                    host,
                    port,
                    semaphore,
                    failover_replay,
                    failover_replay_critical_only,
                    failover_injection,
                    failover_retry_rank_offset,
                    failover_retry_policy,
                    progress,
                )
                for workflow_items in _group_prepared_workflows(prepared_requests)
            ]
        )
        rows = [row for workflow_rows in nested for row in workflow_rows]
        rows.sort(key=lambda r: _rid_sort_key(r.rid))
        return rows

    tasks: list[asyncio.Task[list[RequestDetailRow]]] = []
    for prepared in prepared_requests:
        tasks.append(
            asyncio.create_task(
                _bounded_post(
                    host,
                    port,
                    prepared,
                    semaphore,
                    failover_replay,
                    failover_replay_critical_only,
                    failover_injection,
                    failover_retry_rank_offset,
                    failover_retry_policy,
                    progress,
                ),
                name=f"gsp-rid-{prepared.rid}",
            )
        )
    if not tasks:
        return []
    nested = await asyncio.gather(*tasks)
    rows = [row for session_rows in nested for row in session_rows]
    rows.sort(key=lambda r: _rid_sort_key(r.rid))
    return rows


def run_client(
    host: str,
    port: int,
    prepared_requests: list[PreparedRequest],
    client: ClientConfig,
) -> list[RequestDetailRow]:
    multi_turn = sum(1 for p in prepared_requests if p.is_multi_turn)
    logger.info(
        "Running bench_serving-style client sessions=%d multi_turn_sessions=%d "
        "request_rate=inf max_concurrency=%s",
        len(prepared_requests),
        multi_turn,
        client.max_concurrency,
    )
    rows = asyncio.run(
        _run_client_async(
            host,
            port,
            prepared_requests,
            max_concurrency=client.max_concurrency,
            failover_replay=client.gsp_failover_replay,
            failover_replay_critical_only=client.gsp_failover_replay_critical_only,
            failover_injection=client.gsp_failover_injection,
            failover_retry_rank_offset=client.gsp_failover_retry_rank_offset,
            failover_retry_policy=client.gsp_failover_retry_policy,
        )
    )
    logger.info("Completed %d detail rows (%d sessions)", len(rows), len(prepared_requests))
    return rows
