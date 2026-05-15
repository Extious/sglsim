from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sglsim.fault_tolerance.config import ExperimentConfig, Strategy
from sglsim.fault_tolerance.metrics_utils import write_internal_failover_metrics


@dataclass(frozen=True)
class WorkloadRequest:
    job_id: str
    input_len: int
    output_len: int
    arrival_time: float
    dp_rank: int


@dataclass(frozen=True)
class StrategyModel:
    name: Strategy
    checkpoint_ratio: float
    backup_latency_s: float
    failover_latency_s: float


@dataclass(frozen=True)
class SimulationResult:
    strategy: Strategy
    metrics: dict[str, Any]
    events: list[dict[str, Any]]
    jobs: list[dict[str, Any]]
    impacted_job_ids: list[str]


STRATEGY_MODELS: dict[Strategy, StrategyModel] = {
    "baseline": StrategyModel(
        name="baseline",
        checkpoint_ratio=0.0,
        backup_latency_s=0.0,
        failover_latency_s=2.0,
    ),
    "host_backup": StrategyModel(
        name="host_backup",
        checkpoint_ratio=0.55,
        backup_latency_s=0.00002,
        failover_latency_s=0.8,
    ),
    "remote_backup": StrategyModel(
        name="remote_backup",
        checkpoint_ratio=0.8,
        backup_latency_s=0.00004,
        failover_latency_s=0.45,
    ),
}


class ConstantLatencyPredictor:
    def __init__(self, prefill_per_token_s: float = 0.0005, decode_per_token_s: float = 0.01):
        self.prefill_per_token_s = prefill_per_token_s
        self.decode_per_token_s = decode_per_token_s

    def predict_request(self, input_len: int, output_len: int, recomputed_tokens: int = 0) -> float:
        return (
            (input_len + recomputed_tokens) * self.prefill_per_token_s
            + output_len * self.decode_per_token_s
        )


class FaultToleranceSimulator:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        predictor: ConstantLatencyPredictor | None = None,
        base_time: datetime | None = None,
    ):
        self.config = config
        self.predictor = predictor or ConstantLatencyPredictor()
        self.base_time = base_time or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def run(self, output_dir: str | Path | None = None) -> SimulationResult:
        strategy = self.config.server.strategy
        model = STRATEGY_MODELS[strategy]
        workload = self._build_workload()
        fault_index = self._fault_index(workload)
        recover_index = self._recover_index(workload, fault_index)
        events: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []

        completion_cursor = 0.0
        planned_jobs: list[tuple[WorkloadRequest, float]] = []
        for req in workload:
            latency = self.predictor.predict_request(req.input_len, req.output_len, 0)
            completion_cursor = max(completion_cursor, req.arrival_time) + latency
            planned_jobs.append((req, completion_cursor))

        fault_time = planned_jobs[fault_index][1] if planned_jobs else 0.0
        recover_time = planned_jobs[recover_index][1] if planned_jobs else fault_time
        self._append_event(
            events,
            "fault_scheduled",
            at_s=fault_time,
            event_id=1,
            trigger_kind=self.config.failure.trigger_kind,
            trigger_count=fault_index + 1,
            inject_delay=self.config.failure.delay,
            dp_rank=self.config.failure.dp_rank,
            pp_rank=self.config.failure.pp_rank,
            tp_rank=self.config.failure.tp_rank,
        )
        self._append_event(
            events,
            "fault_injected",
            at_s=fault_time,
            event_id=1,
            kv_backup=self.config.server.kv_backup,
            dp_rank=self.config.failure.dp_rank,
            pp_rank=self.config.failure.pp_rank,
            tp_rank=self.config.failure.tp_rank,
        )
        if strategy != "baseline":
            self._append_event(
                events,
                "runtime_failover_reaction",
                at_s=fault_time + model.failover_latency_s,
                event_id=1,
                status="runtime_reaction",
                dp_rank=self.config.failure.dp_rank,
            )

        for index, (req, base_completion_time) in enumerate(planned_jobs):
            in_failure_window = fault_index <= index <= recover_index
            failed_rank = str(req.dp_rank) == self.config.failure.dp_rank
            affected = in_failure_window and failed_rank
            checkpointed = int(req.input_len * model.checkpoint_ratio) if affected else 0
            remote_reuse = checkpointed if affected and strategy == "remote_backup" else 0
            host_reuse = checkpointed if affected and strategy == "host_backup" else 0
            recomputed = req.input_len - checkpointed if affected else 0
            failover_penalty = model.failover_latency_s if affected else 0.0
            backup_penalty = req.input_len * model.backup_latency_s
            latency = self.predictor.predict_request(req.input_len, req.output_len, recomputed)
            base_latency = self.predictor.predict_request(req.input_len, req.output_len, 0)
            completion_time = base_completion_time + (latency - base_latency) + backup_penalty + failover_penalty
            job = {
                "job_id": req.job_id,
                "job": f"request_{req.job_id}",
                "worker_id": str(index % max(self.config.client.fixed.app_workers, 1)),
                "dp_rank": req.dp_rank,
                "input_len": req.input_len,
                "output_len": req.output_len,
                "created_time": req.arrival_time,
                "completion_time_s": completion_time,
                "latency_s": completion_time - req.arrival_time,
                "affected": affected,
                "checkpointed_tokens": checkpointed,
                "remote_reuse_tokens": remote_reuse,
                "host_reuse_tokens": host_reuse,
                "recomputed_tokens": recomputed,
            }
            jobs.append(job)
            self._append_event(
                events,
                "job_completed",
                at_s=completion_time,
                job_id=req.job_id,
                worker_id=job["worker_id"],
                dp_rank=req.dp_rank,
            )
            self._append_event(
                events,
                "task_completed",
                at_s=completion_time,
                job_id=req.job_id,
                task_label="fixed_request",
                agent_role="fixed_client",
                worker_id=job["worker_id"],
            )

        self._append_event(
            events,
            "gpu_recovered",
            at_s=max(recover_time, fault_time) + self.config.failure.recovery_delay_seconds(),
            event_id=1,
            status="recovered",
            dp_rank=self.config.failure.dp_rank,
        )
        events.sort(key=lambda item: item["timestamp"])
        impacted_job_ids = [job["job_id"] for job in jobs if job["affected"]]
        metrics = self._build_metrics(strategy, jobs)
        result = SimulationResult(
            strategy=strategy,
            metrics=metrics,
            events=events,
            jobs=jobs,
            impacted_job_ids=impacted_job_ids,
        )
        if output_dir is not None:
            self.write_outputs(Path(output_dir), result)
        return result

    def write_outputs(self, output_dir: Path, result: SimulationResult) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")
        with (output_dir / "events.jsonl").open("w") as f:
            for event in result.events:
                f.write(json.dumps(event) + "\n")
        self._write_job_summary(output_dir / "job_summary.csv", result.jobs)
        self._write_task_summary(output_dir / "task_summary.csv", result.jobs)
        self._write_cache_hits(output_dir / "cache_hits.csv", result.jobs)
        write_internal_failover_metrics(
            output_dir / "internal_failover_metrics.json",
            events=result.events,
            jobs=result.jobs,
        )

    def _build_workload(self) -> list[WorkloadRequest]:
        client = self.config.client
        dp_size = max(self.config.server.server.dp_size, 1)
        if client.client_mode == "crewai":
            num_requests = client.crewai.job_limit
            input_len = max(client.crewai.short_max_tokens * 4, 1)
            output_len = client.crewai.long_max_tokens
            interval = max(client.crewai.worker_start_stagger_s, 0.25)
        else:
            num_requests = client.fixed.num_requests
            input_len = client.fixed.input_len
            output_len = client.fixed.output_len
            interval = 0.25
        return [
            WorkloadRequest(
                job_id=str(index + 1),
                input_len=input_len,
                output_len=output_len,
                arrival_time=index * interval,
                dp_rank=index % dp_size,
            )
            for index in range(num_requests)
        ]

    def _fault_index(self, workload: list[WorkloadRequest]) -> int:
        if not workload:
            return 0
        if self.config.failure.trigger_kind == "timeline":
            threshold = self.config.failure.timeline_thresholds[0] if self.config.failure.timeline_thresholds else 0.0
            for index, req in enumerate(workload):
                if req.arrival_time >= threshold:
                    return index
            return len(workload) - 1
        threshold = self.config.failure.thresholds[0] if self.config.failure.thresholds else 1
        return min(max(threshold - 1, 0), len(workload) - 1)

    def _recover_index(self, workload: list[WorkloadRequest], fault_index: int) -> int:
        if not workload:
            return 0
        threshold = self.config.failure.recovery_job_threshold()
        if threshold is None:
            threshold = self.config.failure.recovery_task_threshold()
        if threshold is None:
            return min(fault_index + 1, len(workload) - 1)
        return min(max(threshold - 1, fault_index), len(workload) - 1)

    def _append_event(self, events: list[dict[str, Any]], event_type: str, *, at_s: float, **fields: Any) -> None:
        record = {
            "event": event_type,
            "timestamp": (self.base_time + timedelta(seconds=float(at_s))).isoformat(),
        }
        record.update(fields)
        events.append(record)

    def _build_metrics(self, strategy: Strategy, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        completed = len(jobs)
        total_input = sum(int(job["input_len"]) for job in jobs)
        total_output = sum(int(job["output_len"]) for job in jobs)
        duration = max((float(job["completion_time_s"]) for job in jobs), default=0.0)
        duration = max(duration, 1e-9)
        latencies = [float(job["latency_s"]) for job in jobs]
        return {
            "strategy": strategy,
            "completed": completed,
            "total_input": total_input,
            "total_output": total_output,
            "duration": duration,
            "request_throughput": completed / duration,
            "input_throughput": total_input / duration,
            "output_throughput": total_output / duration,
            "mean_e2e_latency_ms": _mean(latencies) * 1000,
            "total_remote_reuse_tokens": sum(int(job["remote_reuse_tokens"]) for job in jobs),
            "total_host_reuse_tokens": sum(int(job["host_reuse_tokens"]) for job in jobs),
            "total_recomputed_tokens": sum(int(job["recomputed_tokens"]) for job in jobs),
            "impacted_requests": sum(1 for job in jobs if job["affected"]),
        }

    def _write_job_summary(self, path: Path, jobs: list[dict[str, Any]]) -> None:
        fieldnames = [
            "job_id",
            "job",
            "year",
            "worker_id",
            "status",
            "job_completion_time_s",
            "llm_calls",
            "prefill_cached_tokens",
            "total_tokens",
            "error",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for job in jobs:
                writer.writerow(
                    {
                        "job_id": job["job_id"],
                        "job": job["job"],
                        "year": "",
                        "worker_id": job["worker_id"],
                        "status": "completed",
                        "job_completion_time_s": f"{job['completion_time_s']:.6f}",
                        "llm_calls": 1,
                        "prefill_cached_tokens": job["checkpointed_tokens"],
                        "total_tokens": int(job["input_len"]) + int(job["output_len"]),
                        "error": "",
                    }
                )

    def _write_task_summary(self, path: Path, jobs: list[dict[str, Any]]) -> None:
        fieldnames = [
            "job_id",
            "job",
            "year",
            "worker_id",
            "job_status",
            "agent",
            "task_label",
            "task_completion_time_s",
            "llm_calls",
            "prompt_tokens",
            "cached_prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prefill_cache_hit_rate",
            "rid",
            "error",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for job in jobs:
                input_len = int(job["input_len"])
                cached = int(job["checkpointed_tokens"])
                writer.writerow(
                    {
                        "job_id": job["job_id"],
                        "job": job["job"],
                        "year": "",
                        "worker_id": job["worker_id"],
                        "job_status": "completed",
                        "agent": "fixed_client",
                        "task_label": "fixed_request",
                        "task_completion_time_s": f"{job['completion_time_s']:.6f}",
                        "llm_calls": 1,
                        "prompt_tokens": input_len,
                        "cached_prompt_tokens": cached,
                        "completion_tokens": job["output_len"],
                        "total_tokens": input_len + int(job["output_len"]),
                        "prefill_cache_hit_rate": cached / input_len if input_len else 0,
                        "rid": f"rid-{job['job_id']}",
                        "error": "",
                    }
                )

    def _write_cache_hits(self, path: Path, jobs: list[dict[str, Any]]) -> None:
        fieldnames = [
            "job_id",
            "task_label",
            "rid",
            "l1_match",
            "l2_match",
            "remote_match",
            "remote_prefetch",
            "reused_device",
            "reused_host",
            "reused_storage",
            "is_failover_retried",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for job in jobs:
                writer.writerow(
                    {
                        "job_id": job["job_id"],
                        "task_label": "fixed_request",
                        "rid": f"rid-{job['job_id']}",
                        "l1_match": 0,
                        "l2_match": job["host_reuse_tokens"],
                        "remote_match": job["remote_reuse_tokens"],
                        "remote_prefetch": job["remote_reuse_tokens"],
                        "reused_device": 0,
                        "reused_host": job["host_reuse_tokens"],
                        "reused_storage": job["remote_reuse_tokens"],
                        "is_failover_retried": int(bool(job["affected"])),
                    }
                )


def compare_strategies(
    base_config: ExperimentConfig,
    *,
    strategies: list[Strategy],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir) if output_dir is not None else None
    summary: dict[str, Any] = {"strategies": {}}
    for strategy in strategies:
        config = base_config.with_strategy(strategy)
        strategy_dir = output_root / strategy if output_root is not None else None
        result = FaultToleranceSimulator(config).run(output_dir=strategy_dir)
        summary["strategies"][strategy] = result.metrics
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
