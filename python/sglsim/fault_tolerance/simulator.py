from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sglsim.fault_tolerance.config import ExperimentConfig, Strategy
from sglsim.fault_tolerance.metrics_utils import write_internal_failover_metrics
from sglsim.fault_tolerance.recovery import (
    compute_failover_recovery_cost,
    estimate_output_progress,
)
from sglsim.fault_tolerance.latency_breakdown import compute_latency_segments
from sglsim.fault_tolerance.load_model import (
    estimate_failover_recovery_hits,
    failover_recovery_hit_rate,
    prefill_cache_hit_rate,
    steady_state_hicache_hits,
)
from sglsim.fault_tolerance.reference import DEFAULT_REFERENCE, estimate_request_service_s
from sglsim.srt.communication.transfer import TransferEstimator


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
    backup_latency_s: float
    failover_latency_s: float
    steady_state_overhead_ratio: float


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
        backup_latency_s=0.0,
        failover_latency_s=2.0,
        steady_state_overhead_ratio=0.0,
    ),
    "host_backup": StrategyModel(
        name="host_backup",
        backup_latency_s=0.00002,
        failover_latency_s=0.8,
        steady_state_overhead_ratio=0.03,
    ),
    "remote_backup": StrategyModel(
        name="remote_backup",
        backup_latency_s=0.00004,
        failover_latency_s=0.45,
        steady_state_overhead_ratio=0.035,
    ),
    "gpu_backup": StrategyModel(
        name="gpu_backup",
        backup_latency_s=0.000005,
        failover_latency_s=0.05,
        steady_state_overhead_ratio=0.02,
    ),
}


class FaultToleranceSimulator:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        base_time: datetime | None = None,
        transfer: TransferEstimator | None = None,
        reference=DEFAULT_REFERENCE,
    ):
        self.config = config
        self.base_time = base_time or datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.transfer = transfer or TransferEstimator()
        self.reference = reference
        self.peer_backup_lag_tokens = max(
            0, int(self.config.server.sim.peer_backup_lag_tokens or 8)
        )

    def run(self, output_dir: str | Path | None = None) -> SimulationResult:
        strategy = self.config.server.strategy
        model = STRATEGY_MODELS[strategy]
        workload = self._build_workload()
        fault_index = self._fault_index(workload)
        recover_index = self._recover_index(workload, fault_index)
        events: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        shared_prefix = max(0, self.config.client.fixed.shared_prefix_len)
        dp_size = max(1, self.config.server.server.dp_size)

        completion_cursor = 0.0
        planned_jobs: list[tuple[WorkloadRequest, float]] = []
        for req in workload:
            latency = estimate_request_service_s(
                input_len=req.input_len,
                output_len=req.output_len,
                shared_prefix_len=shared_prefix,
                reference=self.reference,
            )
            latency *= 1.0 + model.steady_state_overhead_ratio
            completion_cursor = max(completion_cursor, req.arrival_time) + latency
            planned_jobs.append((req, completion_cursor))

        fault_time = planned_jobs[fault_index][1] if planned_jobs else 0.0
        recover_time = planned_jobs[recover_index][1] if planned_jobs else fault_time
        resume_time = fault_time + model.failover_latency_s
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
            "gpu_failure_simulated",
            at_s=fault_time,
            event_id=1,
            kv_backup=self.config.server.kv_backup,
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
                at_s=resume_time,
                event_id=1,
                status="runtime_reaction",
                dp_rank=self.config.failure.dp_rank,
            )
            self._append_event(
                events,
                "resume_accepted",
                at_s=resume_time,
                event_id=1,
                dp_rank=self.config.failure.dp_rank,
            )

        failed_dp_rank = int(self.config.failure.dp_rank)
        concurrency = max(1, self.config.client.fixed.app_workers)
        healthy_dp_rank = 1 - failed_dp_rank if dp_size > 1 else failed_dp_rank
        for index, (req, base_completion_time) in enumerate(planned_jobs):
            in_failure_window = fault_index <= index <= recover_index
            affected = in_failure_window and req.dp_rank == failed_dp_rank
            output_generated = (
                estimate_output_progress(
                    req.output_len,
                    fault_fraction=self.config.failure.fault_fraction,
                )
                if affected
                else 0
            )
            if affected:
                recovery = compute_failover_recovery_cost(
                    strategy,
                    input_len=req.input_len,
                    output_len=req.output_len,
                    output_generated=output_generated,
                    shared_prefix_len=shared_prefix,
                    transfer=self.transfer,
                    peer_backup_lag_tokens=self.peer_backup_lag_tokens,
                    failover_overhead_s=model.failover_latency_s,
                    concurrency=concurrency,
                )
                failover_hits = estimate_failover_recovery_hits(
                    strategy,
                    input_len=req.input_len,
                    output_generated=output_generated,
                    shared_prefix_len=shared_prefix,
                    peer_backup_lag_tokens=self.peer_backup_lag_tokens,
                    concurrency=concurrency,
                )
                steady_hits = steady_state_hicache_hits(
                    input_len=req.input_len,
                    shared_prefix_len=shared_prefix,
                    reference=self.reference,
                )
                l1_match = failover_hits.l1_match
                l2_match = failover_hits.l2_match
                host_reuse = failover_hits.host_backup_match if strategy == "host_backup" else 0
                peer_reuse = failover_hits.peer_replica_match if strategy == "gpu_backup" else 0
                remote_reuse = failover_hits.host_backup_match if strategy == "remote_backup" else 0
                prefill_hit_rate = prefill_cache_hit_rate(steady_hits, input_len=req.input_len)
                failover_hit_rate = failover_recovery_hit_rate(
                    failover_hits,
                    input_len=req.input_len,
                    output_generated=output_generated,
                )
            else:
                recovery = compute_failover_recovery_cost(
                    "baseline",
                    input_len=req.input_len,
                    output_len=req.output_len,
                    output_generated=0,
                    shared_prefix_len=shared_prefix,
                    transfer=self.transfer,
                )
                recovery = type(recovery)(
                    backed_up_tokens=0,
                    recompute_tokens=0,
                    restore_transfer_s=0.0,
                    failover_overhead_s=0.0,
                )
                steady_hits = steady_state_hicache_hits(
                    input_len=req.input_len,
                    shared_prefix_len=shared_prefix,
                    reference=self.reference,
                )
                l1_match = steady_hits.l1_match
                l2_match = steady_hits.l2_match
                host_reuse = 0
                peer_reuse = 0
                remote_reuse = 0
                prefill_hit_rate = prefill_cache_hit_rate(steady_hits, input_len=req.input_len)
                failover_hit_rate = 0.0

            failover_penalty = recovery.failover_overhead_s + recovery.restore_transfer_s if affected else 0.0
            backup_penalty = (req.input_len + req.output_len) * model.backup_latency_s
            latency = estimate_request_service_s(
                input_len=req.input_len,
                output_len=req.output_len,
                shared_prefix_len=shared_prefix,
                recompute_tokens=recovery.recompute_tokens,
                reference=self.reference,
            )
            base_latency = estimate_request_service_s(
                input_len=req.input_len,
                output_len=req.output_len,
                shared_prefix_len=shared_prefix,
                reference=self.reference,
            )
            completion_time = (
                base_completion_time + (latency - base_latency) + backup_penalty + failover_penalty
            )
            latency_s = completion_time - req.arrival_time
            segments = compute_latency_segments(
                strategy=strategy,
                input_len=req.input_len,
                output_len=req.output_len,
                shared_prefix_len=shared_prefix,
                affected=affected,
                latency_s=latency_s,
                recompute_tokens=recovery.recompute_tokens,
                restore_transfer_s=recovery.restore_transfer_s,
                failover_overhead_s=recovery.failover_overhead_s if affected else 0.0,
                output_generated=output_generated if affected else 0,
                fault_fraction=self.config.failure.fault_fraction,
                peer_backup_lag_tokens=self.peer_backup_lag_tokens,
                concurrency=concurrency,
                transfer=self.transfer,
                reference=self.reference,
            )
            job = {
                "job_id": req.job_id,
                "job": f"request_{req.job_id}",
                "worker_id": str(index % max(self.config.client.fixed.app_workers, 1)),
                "dp_rank": req.dp_rank,
                "input_len": req.input_len,
                "output_len": req.output_len,
                "created_time": req.arrival_time,
                "completion_time_s": completion_time,
                "latency_s": latency_s,
                "output_generated": output_generated if affected else 0,
                "latency_segments": segments.to_dict(),
                "affected": affected,
                "rerouted": affected,
                "reroute_from_dp_rank": failed_dp_rank if affected else req.dp_rank,
                "reroute_to_dp_rank": healthy_dp_rank if affected else req.dp_rank,
                "checkpointed_tokens": recovery.backed_up_tokens,
                "remote_reuse_tokens": remote_reuse,
                "host_reuse_tokens": host_reuse,
                "peer_reuse_tokens": peer_reuse,
                "recomputed_tokens": recovery.recompute_tokens,
                "restore_transfer_s": recovery.restore_transfer_s,
                "prefill_cache_hit_rate": prefill_hit_rate,
                "failover_cache_hit_rate": failover_hit_rate,
                "l1_match": l1_match,
                "l2_match": l2_match,
                "remote_match": remote_reuse,
                "peer_match": peer_reuse,
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

        recover_at = max(recover_time, fault_time) + self.config.failure.recovery_delay_seconds()
        self._append_event(
            events,
            "gpu_recovered",
            at_s=recover_at,
            event_id=1,
            status="recovered",
            dp_rank=self.config.failure.dp_rank,
        )
        events.sort(key=lambda item: item["timestamp"])
        impacted_job_ids = [job["job_id"] for job in jobs if job["affected"]]
        metrics = self._build_metrics(strategy, jobs, fault_time=fault_time, resume_time=resume_time)
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
        (output_dir / "jobs.json").write_text(
            json.dumps(result.jobs, indent=2) + "\n", encoding="utf-8"
        )
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
            threshold = (
                self.config.failure.timeline_thresholds[0]
                if self.config.failure.timeline_thresholds
                else 0.0
            )
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

    def _append_event(
        self, events: list[dict[str, Any]], event_type: str, *, at_s: float, **fields: Any
    ) -> None:
        record = {
            "event": event_type,
            "timestamp": (self.base_time + timedelta(seconds=float(at_s))).isoformat(),
        }
        record.update(fields)
        events.append(record)

    def _build_metrics(
        self,
        strategy: Strategy,
        jobs: list[dict[str, Any]],
        *,
        fault_time: float,
        resume_time: float,
    ) -> dict[str, Any]:
        completed = len(jobs)
        total_input = sum(int(job["input_len"]) for job in jobs)
        total_output = sum(int(job["output_len"]) for job in jobs)
        duration = max((float(job["completion_time_s"]) for job in jobs), default=0.0)
        duration = max(duration, 1e-9)
        latencies = [float(job["latency_s"]) for job in jobs]
        prefill_rates = [float(job.get("prefill_cache_hit_rate", 0.0)) for job in jobs]
        failover_rates = [
            float(job["failover_cache_hit_rate"])
            for job in jobs
            if job.get("affected")
        ]
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
            "mean_prefill_cache_hit_rate": _mean(prefill_rates),
            "mean_failover_cache_hit_rate": _mean(failover_rates),
            "total_remote_reuse_tokens": sum(int(job["remote_reuse_tokens"]) for job in jobs),
            "total_host_reuse_tokens": sum(int(job["host_reuse_tokens"]) for job in jobs),
            "total_peer_reuse_tokens": sum(int(job.get("peer_reuse_tokens", 0)) for job in jobs),
            "total_recomputed_tokens": sum(int(job["recomputed_tokens"]) for job in jobs),
            "total_restore_transfer_s": sum(float(job.get("restore_transfer_s", 0.0)) for job in jobs),
            "impacted_requests": sum(1 for job in jobs if job["affected"]),
            "mttr_s": max(0.0, resume_time - fault_time),
            "fault_time_s": fault_time,
            "resume_time_s": resume_time,
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
                        "prefill_cache_hit_rate": job.get("prefill_cache_hit_rate", cached / input_len if input_len else 0),
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
                        "l1_match": job.get("l1_match", 0),
                        "l2_match": job.get("l2_match", 0),
                        "remote_match": job.get("remote_match", job.get("remote_reuse_tokens", 0)),
                        "remote_prefetch": job.get("remote_match", job.get("remote_reuse_tokens", 0)),
                        "reused_device": job.get("peer_reuse_tokens", 0),
                        "reused_host": job.get("host_reuse_tokens", 0),
                        "reused_storage": job.get("remote_reuse_tokens", 0),
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
        _write_compare_csv(output_root / "compare.csv", summary["strategies"])
    return summary


def _write_compare_csv(path: Path, strategies: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "strategy",
        "mean_e2e_latency_ms",
        "mean_prefill_cache_hit_rate",
        "mean_failover_cache_hit_rate",
        "total_recomputed_tokens",
        "total_restore_transfer_s",
        "mttr_s",
        "impacted_requests",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in strategies.items():
            writer.writerow(
                {
                    "strategy": name,
                    "mean_e2e_latency_ms": metrics.get("mean_e2e_latency_ms", 0),
                    "mean_prefill_cache_hit_rate": metrics.get("mean_prefill_cache_hit_rate", 0),
                    "mean_failover_cache_hit_rate": metrics.get("mean_failover_cache_hit_rate", 0),
                    "total_recomputed_tokens": metrics.get("total_recomputed_tokens", 0),
                    "total_restore_transfer_s": metrics.get("total_restore_transfer_s", 0),
                    "mttr_s": metrics.get("mttr_s", 0),
                    "impacted_requests": metrics.get("impacted_requests", 0),
                }
            )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
