"""Clock-agnostic pipeline-parallel identities and wire contracts.

This module mirrors the metadata/control responsibilities of pinned SGLang's
``SchedulerPPMixin`` without importing SGLang or implementing a second
scheduler.  Online and offline drivers consume the same immutable messages.
The actual PP event-loop integration is intentionally kept outside these pure
contracts so every invariant can be unit tested first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


SCHEMA_VERSION = 1
PLACEMENT_TIERS = frozenset({"HBM", "HOST", "RECOMPUTE"})


class PipelineBatchIdentityError(RuntimeError):
    """Raised when PP stages disagree on a scheduled microbatch."""


def _require_nonnegative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


def _require_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, order=True)
class StageId:
    """Identity of one independently running Scheduler/worker stage."""

    dp_rank: int
    pp_rank: int
    tp_rank: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative("dp_rank", self.dp_rank)
        _require_nonnegative("pp_rank", self.pp_rank)
        _require_nonnegative("tp_rank", self.tp_rank)


@dataclass(frozen=True)
class PipelineWorkerSpec:
    """Launch identity for one DP x PP x TP Scheduler subprocess."""

    stage: StageId
    worker_index: int

    def __post_init__(self) -> None:
        _require_nonnegative("worker_index", self.worker_index)


@dataclass(frozen=True)
class PipelineWorkerHandle:
    """Parent-owned process/control handles for one Scheduler stage."""

    spec: PipelineWorkerSpec
    process: Any
    command_queue: Any
    ready_event: Any


@dataclass(frozen=True)
class PipelineProcessGroup:
    """Validated parent view of all Scheduler processes in a DP x PP group."""

    dp_size: int
    pp_size: int
    tp_size: int
    workers: tuple[PipelineWorkerHandle, ...]

    def __post_init__(self) -> None:
        expected = enumerate_pipeline_workers(
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            tp_size=self.tp_size,
        )
        observed = tuple(worker.spec for worker in self.workers)
        if observed != expected:
            raise ValueError(
                "pipeline worker handles do not match the declared DP x PP x TP "
                f"topology: expected={expected}, observed={observed}"
            )

    @property
    def process_count(self) -> int:
        return len(self.workers)

    @property
    def processes(self) -> tuple[Any, ...]:
        return tuple(worker.process for worker in self.workers)

    @property
    def command_queues(self) -> tuple[Any, ...]:
        return tuple(worker.command_queue for worker in self.workers)

    def for_stage(self, stage: StageId) -> PipelineWorkerHandle:
        for worker in self.workers:
            if worker.spec.stage == stage:
                return worker
        raise KeyError(stage)

    def for_dp(self, dp_rank: int) -> tuple[PipelineWorkerHandle, ...]:
        if not 0 <= dp_rank < self.dp_size:
            raise ValueError(
                f"dp_rank must be in [0, {self.dp_size}), got {dp_rank}"
            )
        return tuple(
            worker
            for worker in self.workers
            if worker.spec.stage.dp_rank == dp_rank
        )

    def leader_for_dp(self, dp_rank: int) -> PipelineWorkerHandle:
        return self.for_stage(StageId(dp_rank=dp_rank, pp_rank=0, tp_rank=0))

    def send_stage_command(self, stage: StageId, message: object) -> None:
        self.for_stage(stage).command_queue.put(message)

    def broadcast_dp_command(self, dp_rank: int, message: object) -> None:
        for worker in self.for_dp(dp_rank):
            worker.command_queue.put(message)

    def broadcast_command(self, message: object) -> None:
        for worker in self.workers:
            worker.command_queue.put(message)

    def alive_stages(self) -> tuple[StageId, ...]:
        return tuple(
            worker.spec.stage
            for worker in self.workers
            if worker.process.is_alive()
        )


def enumerate_pipeline_workers(
    *, dp_size: int, pp_size: int, tp_size: int = 1
) -> tuple[PipelineWorkerSpec, ...]:
    """Enumerate workers in DP, then PP, then TP order.

    Pinned SGLang's DP controller loops over the local PP range and then the TP
    range for each DP replica.  SGLSim currently supports a local topology, so
    this helper makes that process-count/rank contract explicit.
    """

    for name, value in (
        ("dp_size", dp_size),
        ("pp_size", pp_size),
        ("tp_size", tp_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    workers: list[PipelineWorkerSpec] = []
    for dp_rank in range(dp_size):
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                workers.append(
                    PipelineWorkerSpec(
                        stage=StageId(dp_rank, pp_rank, tp_rank),
                        worker_index=len(workers),
                    )
                )
    return tuple(workers)


@dataclass(frozen=True)
class RoutingDecision:
    """Router-owned immutable execution/backup decision for one version."""

    request_id: str
    workflow_id: str
    execution_dp_rank: int
    backup_dp_rank: int | None
    route_version: int
    decided_logical_time: float
    failure_domain_constraints: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be nonempty")
        if not self.workflow_id:
            raise ValueError("workflow_id must be nonempty")
        _require_nonnegative("execution_dp_rank", self.execution_dp_rank)
        if self.backup_dp_rank is not None:
            _require_nonnegative("backup_dp_rank", self.backup_dp_rank)
            if self.backup_dp_rank == self.execution_dp_rank:
                raise ValueError("backup_dp_rank must differ from execution_dp_rank")
        if self.route_version <= 0:
            raise ValueError("route_version must be positive")
        _require_nonnegative("decided_logical_time", self.decided_logical_time)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")


@dataclass(frozen=True)
class DPBudgetSnapshot:
    """One atomically published alpha/beta policy version for a DP replica."""

    dp_rank: int
    budget_version: int
    window_id: int
    alpha: float
    beta: float
    hbm_limit_tokens_per_stage: tuple[int, ...]
    host_policy_limit_tokens: int
    valid_from_logical_time: float
    objective_id: str
    solver_certificate_id: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonnegative("dp_rank", self.dp_rank)
        if self.budget_version <= 0:
            raise ValueError("budget_version must be positive")
        _require_nonnegative("window_id", self.window_id)
        _require_probability("alpha", self.alpha)
        _require_probability("beta", self.beta)
        if not self.hbm_limit_tokens_per_stage:
            raise ValueError("hbm_limit_tokens_per_stage must be nonempty")
        for limit in self.hbm_limit_tokens_per_stage:
            _require_nonnegative("hbm_limit_tokens_per_stage entry", limit)
        _require_nonnegative(
            "host_policy_limit_tokens", self.host_policy_limit_tokens
        )
        _require_nonnegative(
            "valid_from_logical_time", self.valid_from_logical_time
        )
        if not self.objective_id:
            raise ValueError("objective_id must be nonempty")
        if not self.solver_certificate_id:
            raise ValueError("solver_certificate_id must be nonempty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")

    @property
    def pp_size(self) -> int:
        return len(self.hbm_limit_tokens_per_stage)

    def hbm_limit_for_stage(self, pp_rank: int) -> int:
        if not 0 <= pp_rank < self.pp_size:
            raise ValueError(
                f"pp_rank must be in [0, {self.pp_size}), got {pp_rank}"
            )
        return self.hbm_limit_tokens_per_stage[pp_rank]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DPBudgetSnapshot":
        values = dict(raw)
        values["hbm_limit_tokens_per_stage"] = tuple(
            int(value)
            for value in values.get("hbm_limit_tokens_per_stage", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class PPStageBatchEnvelope:
    """Ordered PP request/proxy metadata carried between adjacent stages."""

    stage: StageId
    microbatch_id: int
    batch_id: int
    request_ids: tuple[str, ...]
    forward_mode: str
    token_frontiers: tuple[int, ...]
    route_versions: tuple[int, ...]
    budget_version: int
    causal_sequence: int
    proxy_metadata: tuple[tuple[str, object], ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonnegative("microbatch_id", self.microbatch_id)
        # SGLSim's persistent decode ``running_batch`` deliberately keeps the
        # ``-1`` sentinel assigned by ``Scheduler._new_running_batch``.  Pinned
        # SGLang does not put a synthetic batch id on this lifecycle at all, so
        # the wire identity must preserve (rather than renumber) that sentinel.
        # Values below -1 cannot be produced by either runtime path.
        if self.batch_id < -1:
            raise ValueError(f"batch_id must be at least -1, got {self.batch_id}")
        _require_nonnegative("causal_sequence", self.causal_sequence)
        if not self.forward_mode:
            raise ValueError("forward_mode must be nonempty")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request_ids must be unique within a batch")
        if not (
            len(self.request_ids)
            == len(self.token_frontiers)
            == len(self.route_versions)
        ):
            raise ValueError(
                "request_ids, token_frontiers, and route_versions must align"
            )
        for frontier in self.token_frontiers:
            _require_nonnegative("token frontier", frontier)
        if any(version <= 0 for version in self.route_versions):
            raise ValueError("route versions must be positive")
        if self.budget_version <= 0:
            raise ValueError("budget_version must be positive")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")

    def identity_tuple(self) -> tuple[object, ...]:
        """Fields that must match at every stage for this microbatch."""

        return (
            self.microbatch_id,
            self.batch_id,
            self.request_ids,
            self.forward_mode,
            self.token_frontiers,
            self.route_versions,
            self.budget_version,
            self.causal_sequence,
        )


def assert_batch_identity(
    expected: PPStageBatchEnvelope | None,
    observed: PPStageBatchEnvelope | None,
) -> None:
    """Require exact lifecycle identity while allowing stage rank to differ."""

    if expected is None or observed is None:
        if expected is observed:
            return
        raise PipelineBatchIdentityError(
            f"PP batch presence mismatch: expected={expected}, observed={observed}"
        )
    if expected.stage.dp_rank != observed.stage.dp_rank:
        raise PipelineBatchIdentityError("PP batch crossed DP replicas")
    if expected.stage.tp_rank != observed.stage.tp_rank:
        raise PipelineBatchIdentityError("PP batch crossed TP ranks")
    if expected.identity_tuple() != observed.identity_tuple():
        raise PipelineBatchIdentityError(
            "PP batch lifecycle mismatch: "
            f"expected_stage={expected.stage}, observed_stage={observed.stage}, "
            f"expected={expected.identity_tuple()}, "
            f"observed={observed.identity_tuple()}"
        )


@dataclass(frozen=True)
class PPRequestEnvelope:
    """Ordered new-request message forwarded before proxy/compute metadata."""

    source_stage: StageId
    target_stage: StageId
    step_sequence: int
    request_ids: tuple[str, ...]
    request_payloads: tuple[object, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_forward_edge(self.source_stage, self.target_stage)
        _require_nonnegative("step_sequence", self.step_sequence)
        if len(self.request_ids) != len(self.request_payloads):
            raise ValueError("request IDs and payloads must align")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request IDs must be unique in one PP request envelope")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")

    def forward_from(self, stage: StageId) -> PPRequestEnvelope:
        if stage != self.target_stage:
            raise ValueError("only the target stage may forward a request envelope")
        return PPRequestEnvelope(
            source_stage=stage,
            target_stage=StageId(
                dp_rank=stage.dp_rank,
                pp_rank=stage.pp_rank + 1,
                tp_rank=stage.tp_rank,
            ),
            step_sequence=self.step_sequence,
            request_ids=self.request_ids,
            request_payloads=self.request_payloads,
        )


@dataclass(frozen=True)
class PPProxyEnvelope:
    """Previous-stage compute/proxy readiness for one pipeline step."""

    source_stage: StageId
    target_stage: StageId
    step_sequence: int
    batch: PPStageBatchEnvelope | None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_forward_edge(self.source_stage, self.target_stage)
        _require_nonnegative("step_sequence", self.step_sequence)
        if self.batch is not None:
            if self.batch.stage != self.source_stage:
                raise ValueError("proxy batch must be owned by its source stage")
            if self.batch.causal_sequence != self.step_sequence:
                raise ValueError("proxy step and batch causal sequence must match")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")


@dataclass(frozen=True)
class PPOutputEnvelope:
    """Last-stage result metadata circulating last -> PP0 -> ... -> last."""

    origin_stage: StageId
    source_stage: StageId
    target_stage: StageId
    step_sequence: int
    batch_identity: tuple[object, ...] | None
    result_logical_time: float
    visited_pp_ranks: tuple[int, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.origin_stage.dp_rank != self.source_stage.dp_rank or (
            self.origin_stage.dp_rank != self.target_stage.dp_rank
        ):
            raise ValueError("output ring cannot cross DP replicas")
        if self.origin_stage.tp_rank != self.source_stage.tp_rank or (
            self.origin_stage.tp_rank != self.target_stage.tp_rank
        ):
            raise ValueError("output ring cannot cross TP ranks")
        _require_nonnegative("step_sequence", self.step_sequence)
        _require_nonnegative("result_logical_time", self.result_logical_time)
        if len(set(self.visited_pp_ranks)) != len(self.visited_pp_ranks):
            raise ValueError("output ring cannot visit one PP rank twice")
        if self.target_stage.pp_rank in self.visited_pp_ranks:
            raise ValueError("output target has already consumed this result")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")

    @classmethod
    def from_last_stage(
        cls,
        *,
        last_stage: StageId,
        pp_size: int,
        step_sequence: int,
        batch: PPStageBatchEnvelope | None,
        result_logical_time: float,
    ) -> PPOutputEnvelope:
        if pp_size <= 1 or last_stage.pp_rank != pp_size - 1:
            raise ValueError("output origin must be the declared last PP stage")
        if batch is not None and batch.stage != last_stage:
            raise ValueError("output batch must belong to the last stage")
        return cls(
            origin_stage=last_stage,
            source_stage=last_stage,
            target_stage=StageId(last_stage.dp_rank, 0, last_stage.tp_rank),
            step_sequence=step_sequence,
            batch_identity=batch.identity_tuple() if batch is not None else None,
            result_logical_time=result_logical_time,
        )

    def consume_and_forward(
        self, *, current_stage: StageId, pp_size: int
    ) -> PPOutputEnvelope | None:
        if current_stage != self.target_stage:
            raise ValueError("only the output target may consume this result")
        if pp_size <= 1 or self.origin_stage.pp_rank != pp_size - 1:
            raise ValueError("output ring PP size/origin mismatch")
        visited = self.visited_pp_ranks + (current_stage.pp_rank,)
        if current_stage.pp_rank == pp_size - 1:
            if current_stage != self.origin_stage:
                raise ValueError("output ring ended at a different last stage")
            return None
        return PPOutputEnvelope(
            origin_stage=self.origin_stage,
            source_stage=current_stage,
            target_stage=StageId(
                current_stage.dp_rank,
                current_stage.pp_rank + 1,
                current_stage.tp_rank,
            ),
            step_sequence=self.step_sequence,
            batch_identity=self.batch_identity,
            result_logical_time=self.result_logical_time,
            visited_pp_ranks=visited,
        )


def _validate_forward_edge(source: StageId, target: StageId) -> None:
    if source.dp_rank != target.dp_rank:
        raise ValueError("PP forward edge cannot cross DP replicas")
    if source.tp_rank != target.tp_rank:
        raise ValueError("PP forward edge cannot cross TP ranks")
    if target.pp_rank != source.pp_rank + 1:
        raise ValueError("PP forward edge must connect adjacent stages")


@dataclass(frozen=True)
class StagePlacementDecision:
    """One explicit stage-local HBM/Host/Recompute chunk outcome."""

    request_id: str
    source_stage: StageId
    backup_stage: StageId
    route_version: int
    budget_version: int
    chunk_id: int
    token_start: int
    token_end: int
    tier: str
    failure_domain: str
    utility: float
    copy_link: str
    planned_commit_deadline: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be nonempty")
        if self.source_stage.dp_rank == self.backup_stage.dp_rank:
            raise ValueError("source and backup stages must be in different DPs")
        if self.source_stage.pp_rank != self.backup_stage.pp_rank:
            raise ValueError("KV placement must map the same PP stage")
        if self.source_stage.tp_rank != self.backup_stage.tp_rank:
            raise ValueError("KV placement must map the same TP rank")
        if self.route_version <= 0 or self.budget_version <= 0:
            raise ValueError("route_version and budget_version must be positive")
        _require_nonnegative("chunk_id", self.chunk_id)
        _require_nonnegative("token_start", self.token_start)
        if self.token_end <= self.token_start:
            raise ValueError("token_end must be greater than token_start")
        if self.tier not in PLACEMENT_TIERS:
            raise ValueError(f"unsupported tier={self.tier!r}")
        if not self.failure_domain:
            raise ValueError("failure_domain must be nonempty")
        _require_nonnegative(
            "planned_commit_deadline", self.planned_commit_deadline
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    def validate_against(
        self, route: RoutingDecision, budget: DPBudgetSnapshot
    ) -> None:
        if self.request_id != route.request_id:
            raise ValueError("placement request does not match route")
        if self.route_version != route.route_version:
            raise ValueError("placement route version does not match route")
        if self.source_stage.dp_rank != route.execution_dp_rank:
            raise ValueError("placement source DP does not match route")
        if self.backup_stage.dp_rank != route.backup_dp_rank:
            raise ValueError("placement backup DP does not match immutable route")
        if budget.dp_rank != route.backup_dp_rank:
            raise ValueError("budget owner does not match backup DP")
        if self.budget_version != budget.budget_version:
            raise ValueError("placement budget version does not match snapshot")
        budget.hbm_limit_for_stage(self.backup_stage.pp_rank)


@dataclass(frozen=True)
class StageCommitAck:
    """Versioned committed frontier for one same-stage KV path."""

    request_id: str
    source_stage: StageId
    backup_stage: StageId
    route_version: int
    budget_version: int
    stage_total_tokens: int
    committed_frontier_tokens: int
    hbm_tokens: int
    host_tokens: int
    recompute_tokens: int
    commit_logical_time: float
    storage_failure_domain: str
    recovery_path: tuple[str, ...]
    copy_latency_s: float
    expected_restore_latency_s: float
    expected_recompute_latency_s: float
    copy_bytes: int
    link_id: str
    ack_sequence: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be nonempty")
        if self.source_stage.dp_rank == self.backup_stage.dp_rank:
            raise ValueError("source and backup stages must be in different DPs")
        if self.source_stage.pp_rank != self.backup_stage.pp_rank:
            raise ValueError("ACK must map the same PP stage")
        if self.source_stage.tp_rank != self.backup_stage.tp_rank:
            raise ValueError("ACK must map the same TP rank")
        if self.route_version <= 0 or self.budget_version <= 0:
            raise ValueError("route_version and budget_version must be positive")
        for name, value in (
            ("stage_total_tokens", self.stage_total_tokens),
            ("committed_frontier_tokens", self.committed_frontier_tokens),
            ("hbm_tokens", self.hbm_tokens),
            ("host_tokens", self.host_tokens),
            ("recompute_tokens", self.recompute_tokens),
            ("commit_logical_time", self.commit_logical_time),
            ("copy_latency_s", self.copy_latency_s),
            ("expected_restore_latency_s", self.expected_restore_latency_s),
            ("expected_recompute_latency_s", self.expected_recompute_latency_s),
            ("copy_bytes", self.copy_bytes),
            ("ack_sequence", self.ack_sequence),
        ):
            _require_nonnegative(name, value)
        if self.hbm_tokens + self.host_tokens + self.recompute_tokens != (
            self.stage_total_tokens
        ):
            raise ValueError(
                "HBM, Host, and Recompute tokens must partition stage_total_tokens"
            )
        if self.committed_frontier_tokens > self.hbm_tokens + self.host_tokens:
            raise ValueError("committed frontier cannot exceed protected tokens")
        if not self.storage_failure_domain:
            raise ValueError("storage_failure_domain must be nonempty")
        if not self.recovery_path:
            raise ValueError("recovery_path must be nonempty")
        if not self.link_id:
            raise ValueError("link_id must be nonempty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> StageCommitAck:
        def stage(name: str) -> StageId:
            value = raw.get(name)
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a stage mapping")
            return StageId(
                dp_rank=int(value["dp_rank"]),
                pp_rank=int(value["pp_rank"]),
                tp_rank=int(value.get("tp_rank", 0)),
            )

        return cls(
            request_id=str(raw.get("request_id") or ""),
            source_stage=stage("source_stage"),
            backup_stage=stage("backup_stage"),
            route_version=int(raw.get("route_version", 0)),
            budget_version=int(raw.get("budget_version", 0)),
            stage_total_tokens=int(raw.get("stage_total_tokens", 0)),
            committed_frontier_tokens=int(
                raw.get("committed_frontier_tokens", 0)
            ),
            hbm_tokens=int(raw.get("hbm_tokens", 0)),
            host_tokens=int(raw.get("host_tokens", 0)),
            recompute_tokens=int(raw.get("recompute_tokens", 0)),
            commit_logical_time=float(raw.get("commit_logical_time", 0.0)),
            storage_failure_domain=str(
                raw.get("storage_failure_domain") or ""
            ),
            recovery_path=tuple(
                str(item) for item in raw.get("recovery_path", ())
            ),
            copy_latency_s=float(raw.get("copy_latency_s", 0.0)),
            expected_restore_latency_s=float(
                raw.get("expected_restore_latency_s", 0.0)
            ),
            expected_recompute_latency_s=float(
                raw.get("expected_recompute_latency_s", 0.0)
            ),
            copy_bytes=int(raw.get("copy_bytes", 0)),
            link_id=str(raw.get("link_id") or ""),
            ack_sequence=int(raw.get("ack_sequence", 0)),
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class RecoveryHint:
    """All-stage recovery contract derived from version-matched ACKs."""

    request_id: str
    route_version: int
    budget_version: int
    failed_domains: tuple[str, ...]
    stage_acks: tuple[StageCommitAck | None, ...]
    safe_frontier_tokens: int
    bottleneck_pp_rank: int
    restored_tokens_by_stage: tuple[int, ...]
    recomputed_tokens_by_stage: tuple[int, ...]
    recovery_path_by_stage: tuple[tuple[str, ...], ...]
    total_restore_latency_s: float
    total_recompute_latency_s: float
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_stage_acks(
        cls,
        *,
        request_id: str,
        route_version: int,
        budget_version: int,
        pp_size: int,
        requested_tokens: int,
        stage_acks: Iterable[StageCommitAck],
        failed_domains: Iterable[str] = (),
    ) -> RecoveryHint:
        if pp_size <= 0:
            raise ValueError("pp_size must be positive")
        _require_nonnegative("requested_tokens", requested_tokens)
        failed = tuple(sorted(set(failed_domains)))
        by_pp: dict[int, StageCommitAck] = {}
        for ack in stage_acks:
            if ack.request_id != request_id:
                raise ValueError("ACK request does not match hint request")
            if ack.route_version != route_version:
                raise ValueError("ACK route version does not match hint")
            if ack.budget_version != budget_version:
                raise ValueError("ACK budget version does not match hint")
            pp_rank = ack.source_stage.pp_rank
            if not 0 <= pp_rank < pp_size:
                raise ValueError(f"ACK pp_rank={pp_rank} is outside PP size")
            if pp_rank in by_pp:
                raise ValueError(f"duplicate ACK for pp_rank={pp_rank}")
            by_pp[pp_rank] = ack

        ordered: list[StageCommitAck | None] = []
        frontiers: list[int] = []
        restored: list[int] = []
        recomputed: list[int] = []
        paths: list[tuple[str, ...]] = []
        restore_latencies: list[float] = []
        recompute_latencies: list[float] = []
        failed_set = set(failed)
        for pp_rank in range(pp_size):
            ack = by_pp.get(pp_rank)
            survives = (
                ack is not None
                and ack.storage_failure_domain not in failed_set
            )
            frontier = (
                min(ack.committed_frontier_tokens, requested_tokens)
                if survives and ack is not None
                else 0
            )
            ordered.append(ack if survives else None)
            frontiers.append(frontier)
            restored.append(frontier)
            recomputed.append(requested_tokens - frontier)
            paths.append(ack.recovery_path if survives and ack is not None else ("recompute",))
            restore_latencies.append(
                ack.expected_restore_latency_s
                if survives and ack is not None
                else 0.0
            )
            recompute_latencies.append(
                ack.expected_recompute_latency_s
                if survives and ack is not None
                else 0.0
            )

        safe_frontier = min(frontiers)
        bottleneck = min(
            pp_rank
            for pp_rank, frontier in enumerate(frontiers)
            if frontier == safe_frontier
        )
        return cls(
            request_id=request_id,
            route_version=route_version,
            budget_version=budget_version,
            failed_domains=failed,
            stage_acks=tuple(ordered),
            safe_frontier_tokens=safe_frontier,
            bottleneck_pp_rank=bottleneck,
            restored_tokens_by_stage=tuple(restored),
            recomputed_tokens_by_stage=tuple(recomputed),
            recovery_path_by_stage=tuple(paths),
            total_restore_latency_s=max(restore_latencies, default=0.0),
            total_recompute_latency_s=max(recompute_latencies, default=0.0),
        )

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be nonempty")
        if self.route_version <= 0 or self.budget_version <= 0:
            raise ValueError("route_version and budget_version must be positive")
        pp_size = len(self.stage_acks)
        if pp_size == 0:
            raise ValueError("stage_acks must be nonempty")
        if not (
            pp_size
            == len(self.restored_tokens_by_stage)
            == len(self.recomputed_tokens_by_stage)
            == len(self.recovery_path_by_stage)
        ):
            raise ValueError("per-stage RecoveryHint arrays must align")
        if not 0 <= self.bottleneck_pp_rank < pp_size:
            raise ValueError("bottleneck_pp_rank is outside stage range")
        _require_nonnegative("safe_frontier_tokens", self.safe_frontier_tokens)
        _require_nonnegative(
            "total_restore_latency_s", self.total_restore_latency_s
        )
        _require_nonnegative(
            "total_recompute_latency_s", self.total_recompute_latency_s
        )
        if self.safe_frontier_tokens != min(self.restored_tokens_by_stage):
            raise ValueError("safe frontier must be the all-stage bottleneck")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version={self.schema_version}")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "route_version": self.route_version,
            "budget_version": self.budget_version,
            "failed_domains": list(self.failed_domains),
            "stage_acks": [
                ack.to_dict() if ack is not None else None
                for ack in self.stage_acks
            ],
            "safe_frontier_tokens": self.safe_frontier_tokens,
            "bottleneck_pp_rank": self.bottleneck_pp_rank,
            "restored_tokens_by_stage": list(self.restored_tokens_by_stage),
            "recomputed_tokens_by_stage": list(
                self.recomputed_tokens_by_stage
            ),
            "recovery_path_by_stage": [
                list(path) for path in self.recovery_path_by_stage
            ],
            "total_restore_latency_s": self.total_restore_latency_s,
            "total_recompute_latency_s": self.total_recompute_latency_s,
            "schema_version": self.schema_version,
        }
