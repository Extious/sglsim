#!/usr/bin/env python3
"""Matched synthetic SGLSim campaign for the latest AegisKV design.

This experiment exercises the real metadata controller implemented under
``sglsim.srt``.  It is intentionally a modeled/synthetic mechanism study: KV
bytes are metadata and all transfer/recovery times come from the checked-in
configuration.  The script therefore does not label its output as physical or
paper-ready evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from sglsim.srt.simulation.aegis_independent_tiers import (
    DESIGN_SEMANTICS,
    AegisIndependentTierController,
    ChunkSpec,
    CopyState,
    IndependentTierConfig,
    ProtectionTier,
    RecoveryTiming,
    RequestRecoveryProfile,
)
from sglsim.srt.simulation.aegis_residual_dag import (
    RESIDUAL_DAG_SLACK_SEMANTICS,
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
)


EVIDENCE_CLASS = "modeled_synthetic_sglsim_metadata"
POLICIES = (
    "recompute_only",
    "static_independent_fcfs",
    "aegiskv_hbm_only",
    "aegiskv_dram_only",
    "aegiskv_latest",
)


@dataclass(frozen=True)
class ScenarioChunk:
    request_id: str
    ordinal: int
    token_start: int
    token_end: int
    seal_time_ms: float
    hbm_delay_ms: float
    dram_delay_ms: float
    hbm_succeeded: bool
    dram_succeeded: bool

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class ScenarioRequest:
    request_id: str
    total_tokens: int
    slack_budget_ms: float
    chunks: tuple[ScenarioChunk, ...]


@dataclass(frozen=True)
class Episode:
    seed: int
    failure_time_ms: float
    target_failed: bool
    requests: tuple[ScenarioRequest, ...]


def _request_slack_ticket(request: ScenarioRequest):
    """Build the synthetic fan-out as an explicit visible residual DAG."""

    graph = VisibleResidualDag(
        workflow_id=f"synthetic-{request.request_id}",
        graph_version=f"initial-{request.request_id}",
        observed_at_ms=0.0,
        nodes=(
            ResidualDagNode(
                request.request_id,
                "agent",
                0.0,
                active=True,
                agent_id=request.request_id,
            ),
            ResidualDagNode(
                f"critical-sibling-{request.request_id}",
                "agent",
                request.slack_budget_ms,
                active=True,
                agent_id=f"critical-sibling-{request.request_id}",
            ),
        ),
        edges=(),
        next_join_subgraph_complete=True,
    )
    return build_residual_dag_slack_tickets(graph)[request.request_id]


@dataclass
class _StaticRecord:
    chunk: ScenarioChunk
    state: CopyState
    admitted_sequence: int
    completion_validated: bool = False


class StaticIndependentFCFS:
    """Capacity-matched independent tiers without SlackLandlord."""

    def __init__(self, *, hbm_capacity: int, dram_capacity: int) -> None:
        self.limit = {
            ProtectionTier.HBM: int(hbm_capacity),
            ProtectionTier.DRAM: int(dram_capacity),
        }
        self.used = {ProtectionTier.HBM: 0, ProtectionTier.DRAM: 0}
        self.records: dict[
            ProtectionTier, dict[str, list[_StaticRecord]]
        ] = {
            ProtectionTier.HBM: defaultdict(list),
            ProtectionTier.DRAM: defaultdict(list),
        }
        self.sequence = 0

    def admit(self, tier: ProtectionTier, chunk: ScenarioChunk) -> bool:
        records = self.records[tier][chunk.request_id]
        if chunk.ordinal != len(records):
            return False
        if self.used[tier] + chunk.token_count > self.limit[tier]:
            return False
        self.sequence += 1
        records.append(
            _StaticRecord(
                chunk=chunk,
                state=CopyState.PENDING,
                admitted_sequence=self.sequence,
            )
        )
        self.used[tier] += chunk.token_count
        return True

    def _abort_from(
        self, tier: ProtectionTier, request_id: str, ordinal: int
    ) -> None:
        records = self.records[tier][request_id]
        while records and records[-1].chunk.ordinal >= ordinal:
            record = records.pop()
            self.used[tier] -= record.chunk.token_count

    def complete(
        self,
        tier: ProtectionTier,
        request_id: str,
        ordinal: int,
        *,
        succeeded: bool,
    ) -> None:
        records = self.records[tier][request_id]
        if ordinal >= len(records):
            return
        record = records[ordinal]
        if record.state is not CopyState.PENDING:
            return
        if not succeeded:
            self._abort_from(tier, request_id, ordinal)
            return
        record.completion_validated = True
        for candidate in records:
            if candidate.state is CopyState.COMMITTED:
                continue
            if not candidate.completion_validated:
                break
            candidate.state = CopyState.COMMITTED

    def contract_hbm(self, limit_tokens: int) -> None:
        tier = ProtectionTier.HBM
        self.limit[tier] = max(int(limit_tokens), 0)
        while self.used[tier] > self.limit[tier]:
            tails = [
                records[-1]
                for records in self.records[tier].values()
                if records
            ]
            if not tails:
                raise RuntimeError("static HBM contraction lost its victim set")
            # Static baseline has no Slack value.  Reclaim newest FCFS tails,
            # preserving each request's continuous prefix.
            victim = max(
                tails,
                key=lambda record: (
                    record.admitted_sequence,
                    record.chunk.request_id,
                ),
            )
            self._abort_from(
                tier, victim.chunk.request_id, victim.chunk.ordinal
            )

    def clear_hbm_target(self) -> None:
        self.records[ProtectionTier.HBM].clear()
        self.used[ProtectionTier.HBM] = 0

    def abort_pending_source_copies(self) -> None:
        for tier in (ProtectionTier.HBM, ProtectionTier.DRAM):
            for request_id in list(self.records[tier]):
                records = self.records[tier][request_id]
                pending = [
                    record.chunk.ordinal
                    for record in records
                    if record.state is CopyState.PENDING
                ]
                if pending:
                    self._abort_from(tier, request_id, min(pending))

    def committed_prefix(self, tier: ProtectionTier, request_id: str) -> int:
        frontier = 0
        for record in self.records[tier].get(request_id, ()):
            if record.state is not CopyState.COMMITTED:
                break
            if record.chunk.token_start != frontier:
                break
            frontier = record.chunk.token_end
        return frontier


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("design_semantics") != DESIGN_SEMANTICS:
        raise ValueError("experiment config does not select the latest design")
    return config


def _timing(config: dict[str, Any], section: str, tier: str) -> RecoveryTiming:
    raw = config[section]
    return RecoveryTiming(
        setup_ms=float(raw[f"{tier}_setup_ms"]),
        ms_per_token=float(raw[f"{tier}_ms_per_token"]),
    )


def _chunk_intervals(
    total_tokens: int, *, prefix_tokens: int, decode_tokens: int
) -> list[tuple[int, int]]:
    intervals = []
    start = 0
    if total_tokens:
        end = min(total_tokens, prefix_tokens)
        intervals.append((0, end))
        start = end
    while start < total_tokens:
        end = min(start + decode_tokens, total_tokens)
        intervals.append((start, end))
        start = end
    return intervals


def _build_episode(config: dict[str, Any], seed: int) -> Episode:
    rng = random.Random(seed)
    failure_time = float(config["failure_time_ms"])
    prefix_tokens = int(config["prefix_chunk_tokens"])
    decode_tokens = int(config["decode_chunk_tokens"])
    hbm_copy = _timing(config, "copy_timing", "hbm")
    dram_copy = _timing(config, "copy_timing", "dram")
    requests = []
    for index in range(int(config["requests_per_seed"])):
        request_id = f"seed-{seed:03d}-request-{index:03d}"
        total_tokens = rng.choices(
            config["token_choices"], weights=config["token_weights"], k=1
        )[0]
        value_class = rng.random()
        if value_class < 0.38:
            slack_budget = rng.uniform(0.0, 42.0)
        elif value_class < 0.76:
            slack_budget = rng.uniform(52.0, 105.0)
        else:
            slack_budget = rng.uniform(105.0, 185.0)
        arrival = rng.uniform(0.0, 18.0)
        seal_time = arrival + rng.uniform(0.2, 1.8)
        chunks = []
        for ordinal, (token_start, token_end) in enumerate(
            _chunk_intervals(
                total_tokens,
                prefix_tokens=prefix_tokens,
                decode_tokens=decode_tokens,
            )
        ):
            if ordinal:
                seal_time += rng.uniform(0.45, 1.35)
            token_count = token_end - token_start
            hbm_delay = (
                hbm_copy.setup_ms + token_count * hbm_copy.ms_per_token
            ) * rng.uniform(0.80, 1.25)
            dram_delay = (
                dram_copy.setup_ms + token_count * dram_copy.ms_per_token
            ) * rng.uniform(0.85, 1.30)
            chunks.append(
                ScenarioChunk(
                    request_id=request_id,
                    ordinal=ordinal,
                    token_start=token_start,
                    token_end=token_end,
                    seal_time_ms=round(seal_time, 9),
                    hbm_delay_ms=hbm_delay,
                    dram_delay_ms=dram_delay,
                    hbm_succeeded=(
                        rng.random()
                        < float(config["hbm_copy_success_probability"])
                    ),
                    dram_succeeded=(
                        rng.random()
                        < float(config["dram_copy_success_probability"])
                    ),
                )
            )
        requests.append(
            ScenarioRequest(
                request_id=request_id,
                total_tokens=int(total_tokens),
                slack_budget_ms=slack_budget,
                chunks=tuple(chunks),
            )
        )
    return Episode(
        seed=seed,
        failure_time_ms=failure_time,
        target_failed=(
            rng.random() < float(config["target_failure_probability"])
        ),
        requests=tuple(requests),
    )


def _controller_config(config: dict[str, Any]) -> IndependentTierConfig:
    recovery = config["recovery_timing"]
    copy = config["copy_timing"]
    return IndependentTierConfig(
        hbm_capacity_bytes=int(config["hbm_capacity_tokens"]),
        dram_capacity_bytes=int(config["dram_capacity_tokens"]),
        initial_hbm_alpha=float(config["initial_hbm_alpha"]),
        minimum_hbm_alpha=float(config["minimum_hbm_alpha"]),
        maximum_hbm_alpha=float(config["maximum_hbm_alpha"]),
        aimd_decrease_factor=float(config["aimd_decrease_factor"]),
        aimd_increase_step=float(config["aimd_increase_step"]),
        aimd_stable_windows=int(config["aimd_stable_windows"]),
        recompute=RecoveryTiming(
            float(recovery["recompute_setup_ms"]),
            float(recovery["recompute_ms_per_token"]),
        ),
        hbm_load=RecoveryTiming(
            float(recovery["hbm_setup_ms"]),
            float(recovery["hbm_ms_per_token"]),
        ),
        dram_load=RecoveryTiming(
            float(recovery["dram_setup_ms"]),
            float(recovery["dram_ms_per_token"]),
        ),
        hbm_copy=RecoveryTiming(
            float(copy["hbm_setup_ms"]),
            float(copy["hbm_ms_per_token"]),
        ),
        dram_copy=RecoveryTiming(
            float(copy["dram_setup_ms"]),
            float(copy["dram_ms_per_token"]),
        ),
        hbm_pool_id="experiment-peer-hbm-target-1",
        dram_pool_id="experiment-store-dram",
        dram_failure_domain="store-domain",
    )


def _push_event(
    events: list[tuple[float, int, str, Any]],
    sequence: list[int],
    time_ms: float,
    kind: str,
    payload: Any,
) -> None:
    sequence[0] += 1
    heapq.heappush(events, (float(time_ms), sequence[0], kind, payload))


def _latest_policy(
    episode: Episode,
    config: dict[str, Any],
    *,
    policy: str,
    allowed_tiers: Iterable[ProtectionTier],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controller = AegisIndependentTierController(_controller_config(config))
    request_by_id = {request.request_id: request for request in episode.requests}
    slack_tickets = {
        request.request_id: _request_slack_ticket(request)
        for request in episode.requests
    }
    for request in episode.requests:
        controller.register_request(
            RequestRecoveryProfile(
                request_id=request.request_id,
                source_worker="source-worker-0",
                source_failure_domain="source-node-0",
                hbm_target="target-worker-1",
                hbm_failure_domain="target-node-1",
                route_version=1,
                source_token_end=0,
                slack_ms=slack_tickets[request.request_id].slack_ms,
                model_revision="sglsim-modeled-model-v1",
                kv_abi="sglsim-kv-abi-v1",
                parallel_layout="dp3-tp1-pp1",
            )
        )
    events: list[tuple[float, int, str, Any]] = []
    sequence = [0]
    for request in episode.requests:
        for chunk in request.chunks:
            _push_event(
                events, sequence, chunk.seal_time_ms, "seal", chunk
            )
    _push_event(
        events,
        sequence,
        float(config["hbm_pressure_time_ms"]),
        "hbm_pressure",
        None,
    )
    if episode.target_failed:
        _push_event(
            events,
            sequence,
            float(config["target_failure_time_ms"]),
            "target_failure",
            None,
        )
    now_ms = 0.0
    admitted_copy_tokens = 0
    admitted_paths = 0
    modeled_copy_service_ms = 0.0
    copy_failures = 0
    allowed = tuple(allowed_tiers)
    while events and events[0][0] <= episode.failure_time_ms:
        event_time, _event_sequence, kind, payload = heapq.heappop(events)
        controller.advance_time(max(event_time - now_ms, 0.0))
        now_ms = event_time
        if kind == "seal":
            chunk: ScenarioChunk = payload
            request = request_by_id[chunk.request_id]
            controller.update_slack(
                chunk.request_id,
                max(request.slack_budget_ms - event_time, 0.0),
            )
            controller.update_source_frontier(
                chunk.request_id, chunk.token_end
            )
            profile = controller.request_profile(chunk.request_id)
            spec = ChunkSpec(
                request_id=chunk.request_id,
                ordinal=chunk.ordinal,
                token_start=chunk.token_start,
                token_end=chunk.token_end,
                byte_size=chunk.token_count,
                route_version=profile.route_version,
                source_worker=profile.source_worker,
                hbm_target=profile.hbm_target,
                model_revision=profile.model_revision,
                kv_abi=profile.kv_abi,
                parallel_layout=profile.parallel_layout,
                shard_count=1,
            )
            decision = controller.admit_chunk(
                spec, allowed_tiers=allowed
            )
            modeled_copy_service_ms += controller.copy_cost_ms(decision)
            for tier, admission, delay, succeeded in (
                (
                    ProtectionTier.HBM,
                    decision.hbm,
                    chunk.hbm_delay_ms,
                    chunk.hbm_succeeded,
                ),
                (
                    ProtectionTier.DRAM,
                    decision.dram,
                    chunk.dram_delay_ms,
                    chunk.dram_succeeded,
                ),
            ):
                if not admission.admitted:
                    continue
                admitted_paths += 1
                admitted_copy_tokens += chunk.token_count
                _push_event(
                    events,
                    sequence,
                    event_time + delay,
                    "copy_complete",
                    (tier, spec, succeeded),
                )
        elif kind == "copy_complete":
            tier, spec, succeeded = payload
            if controller.copy_state(
                spec.request_id, spec.ordinal, tier
            ) is not CopyState.PENDING:
                continue
            completion = controller.complete_copy(
                tier,
                spec.request_id,
                spec.ordinal,
                controller.completion_proof(
                    spec, tier, succeeded=succeeded
                ),
            )
            if completion.state is CopyState.ABORTED:
                copy_failures += 1
        elif kind == "hbm_pressure":
            controller.control_hbm_quota(pressure=True)
        elif kind == "target_failure":
            for request in episode.requests:
                profile = controller.request_profile(request.request_id)
                if profile.hbm_target != "target-worker-1":
                    continue
                controller.rotate_hbm_target(
                    request.request_id,
                    failed_target="target-worker-1",
                    replacement_target="replacement-worker-2",
                    replacement_failure_domain="replacement-node-2",
                )
        else:
            raise AssertionError(kind)

    controller.advance_time(max(episode.failure_time_ms - now_ms, 0.0))
    for request in episode.requests:
        controller.update_slack(
            request.request_id,
            max(request.slack_budget_ms - episode.failure_time_ms, 0.0),
        )
    activated_target = (
        "replacement-worker-2"
        if episode.target_failed
        else "target-worker-1"
    )
    failures = []
    for request in episode.requests:
        profile = controller.request_profile(request.request_id)
        if profile.source_token_end <= 0:
            continue
        decision = controller.recover_after_source_failure(
            request.request_id,
            activated_target=activated_target,
            next_route_version=2,
        )
        failures.append(
            {
                "seed": episode.seed,
                "request_id": request.request_id,
                "policy": policy,
                "source_tokens": profile.source_token_end,
                "slack_ms": profile.slack_ms,
                "target_failed_before_source": episode.target_failed,
                "action": decision.action,
                "l1_tokens": (
                    decision.restored_tokens
                    if decision.action == "PEER_HBM"
                    else 0
                ),
                "l2_tokens": (
                    decision.restored_tokens
                    if decision.action == "DRAM"
                    else 0
                ),
                "restored_tokens": decision.restored_tokens,
                "recompute_tokens": decision.recompute_tokens,
                "recovery_ms": decision.recovery_ms,
                "visible_harm_ms": decision.visible_harm_ms,
            }
        )
    controller.validate_invariants()
    snapshot = controller.snapshot()
    observed_tokens = sum(row["source_tokens"] for row in failures)
    return failures, {
        "seed": episode.seed,
        "policy": policy,
        "observed_source_tokens": observed_tokens,
        "admitted_copy_tokens": admitted_copy_tokens,
        "admitted_paths": admitted_paths,
        "modeled_copy_service_ms": modeled_copy_service_ms,
        "copy_failures": copy_failures,
        "hbm_used_tokens_at_failure": snapshot["pools"]["HBM"][
            "used_bytes"
        ],
        "dram_used_tokens_at_failure": snapshot["pools"]["DRAM"][
            "used_bytes"
        ],
        "hbm_alpha_at_failure": snapshot["hbm_alpha"],
        "target_failed_before_source": episode.target_failed,
        "slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
        "slack_updates_metadata_only": True,
    }


def _recovery_ms(
    config: dict[str, Any], tier: ProtectionTier | None, prefix: int, total: int
) -> float:
    raw = config["recovery_timing"]
    suffix = max(total - prefix, 0)
    result = 0.0
    if tier is ProtectionTier.HBM and prefix:
        result += float(raw["hbm_setup_ms"]) + prefix * float(
            raw["hbm_ms_per_token"]
        )
    elif tier is ProtectionTier.DRAM and prefix:
        result += float(raw["dram_setup_ms"]) + prefix * float(
            raw["dram_ms_per_token"]
        )
    if suffix:
        result += float(raw["recompute_setup_ms"]) + suffix * float(
            raw["recompute_ms_per_token"]
        )
    return result


def _static_policy(
    episode: Episode, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    static = StaticIndependentFCFS(
        hbm_capacity=int(config["hbm_capacity_tokens"]),
        dram_capacity=int(config["dram_capacity_tokens"]),
    )
    events: list[tuple[float, int, str, Any]] = []
    sequence = [0]
    for request in episode.requests:
        for chunk in request.chunks:
            _push_event(events, sequence, chunk.seal_time_ms, "seal", chunk)
    _push_event(
        events,
        sequence,
        float(config["hbm_pressure_time_ms"]),
        "hbm_pressure",
        None,
    )
    if episode.target_failed:
        _push_event(
            events,
            sequence,
            float(config["target_failure_time_ms"]),
            "target_failure",
            None,
        )
    admitted_copy_tokens = 0
    admitted_paths = 0
    modeled_copy_service_ms = 0.0
    copy_failures = 0
    observed_frontiers = defaultdict(int)
    copy_cfg = config["copy_timing"]
    while events and events[0][0] <= episode.failure_time_ms:
        event_time, _event_sequence, kind, payload = heapq.heappop(events)
        if kind == "seal":
            chunk: ScenarioChunk = payload
            observed_frontiers[chunk.request_id] = chunk.token_end
            admitted_delays = []
            for tier, delay, succeeded in (
                (ProtectionTier.HBM, chunk.hbm_delay_ms, chunk.hbm_succeeded),
                (ProtectionTier.DRAM, chunk.dram_delay_ms, chunk.dram_succeeded),
            ):
                if not static.admit(tier, chunk):
                    continue
                admitted_paths += 1
                admitted_copy_tokens += chunk.token_count
                timing_name = "hbm" if tier is ProtectionTier.HBM else "dram"
                admitted_delays.append(
                    float(copy_cfg[f"{timing_name}_setup_ms"])
                    + chunk.token_count
                    * float(copy_cfg[f"{timing_name}_ms_per_token"])
                )
                _push_event(
                    events,
                    sequence,
                    event_time + delay,
                    "copy_complete",
                    (tier, chunk, succeeded),
                )
            modeled_copy_service_ms += max(admitted_delays, default=0.0)
        elif kind == "copy_complete":
            tier, chunk, succeeded = payload
            before = static.committed_prefix(tier, chunk.request_id)
            static.complete(
                tier,
                chunk.request_id,
                chunk.ordinal,
                succeeded=succeeded,
            )
            after = static.committed_prefix(tier, chunk.request_id)
            if not succeeded and after <= before:
                copy_failures += 1
        elif kind == "hbm_pressure":
            static.contract_hbm(
                int(
                    int(config["hbm_capacity_tokens"])
                    * float(config["aimd_decrease_factor"])
                )
            )
        elif kind == "target_failure":
            static.clear_hbm_target()
        else:
            raise AssertionError(kind)
    static.abort_pending_source_copies()
    failures = []
    request_by_id = {request.request_id: request for request in episode.requests}
    for request_id, total in sorted(observed_frontiers.items()):
        request = request_by_id[request_id]
        slack = max(
            request.slack_budget_ms - episode.failure_time_ms, 0.0
        )
        hbm = (
            0
            if episode.target_failed
            else min(
                static.committed_prefix(ProtectionTier.HBM, request_id), total
            )
        )
        dram = min(
            static.committed_prefix(ProtectionTier.DRAM, request_id), total
        )
        candidates = [(_recovery_ms(config, None, 0, total), "RECOMPUTE", 0)]
        if hbm:
            candidates.append(
                (_recovery_ms(config, ProtectionTier.HBM, hbm, total), "PEER_HBM", hbm)
            )
        if dram:
            candidates.append(
                (_recovery_ms(config, ProtectionTier.DRAM, dram, total), "DRAM", dram)
            )
        recovery_ms, action, restored = min(
            candidates, key=lambda row: (row[0], -row[2], row[1])
        )
        failures.append(
            {
                "seed": episode.seed,
                "request_id": request_id,
                "policy": "static_independent_fcfs",
                "source_tokens": total,
                "slack_ms": slack,
                "target_failed_before_source": episode.target_failed,
                "action": action,
                "l1_tokens": restored if action == "PEER_HBM" else 0,
                "l2_tokens": restored if action == "DRAM" else 0,
                "restored_tokens": restored,
                "recompute_tokens": max(total - restored, 0),
                "recovery_ms": recovery_ms,
                "visible_harm_ms": max(recovery_ms - slack, 0.0),
            }
        )
    observed_tokens = sum(row["source_tokens"] for row in failures)
    return failures, {
        "seed": episode.seed,
        "policy": "static_independent_fcfs",
        "observed_source_tokens": observed_tokens,
        "admitted_copy_tokens": admitted_copy_tokens,
        "admitted_paths": admitted_paths,
        "modeled_copy_service_ms": modeled_copy_service_ms,
        "copy_failures": copy_failures,
        "hbm_used_tokens_at_failure": static.used[ProtectionTier.HBM],
        "dram_used_tokens_at_failure": static.used[ProtectionTier.DRAM],
        "hbm_alpha_at_failure": float(config["aimd_decrease_factor"]),
        "target_failed_before_source": episode.target_failed,
    }


def _recompute_policy(
    episode: Episode, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for request in episode.requests:
        sealed = [
            chunk
            for chunk in request.chunks
            if chunk.seal_time_ms <= episode.failure_time_ms
        ]
        if not sealed:
            continue
        total = sealed[-1].token_end
        slack = max(
            request.slack_budget_ms - episode.failure_time_ms, 0.0
        )
        recovery_ms = _recovery_ms(config, None, 0, total)
        rows.append(
            {
                "seed": episode.seed,
                "request_id": request.request_id,
                "policy": "recompute_only",
                "source_tokens": total,
                "slack_ms": slack,
                "target_failed_before_source": episode.target_failed,
                "action": "RECOMPUTE",
                "l1_tokens": 0,
                "l2_tokens": 0,
                "restored_tokens": 0,
                "recompute_tokens": total,
                "recovery_ms": recovery_ms,
                "visible_harm_ms": max(recovery_ms - slack, 0.0),
            }
        )
    return rows, {
        "seed": episode.seed,
        "policy": "recompute_only",
        "observed_source_tokens": sum(row["source_tokens"] for row in rows),
        "admitted_copy_tokens": 0,
        "admitted_paths": 0,
        "modeled_copy_service_ms": 0.0,
        "copy_failures": 0,
        "hbm_used_tokens_at_failure": 0,
        "dram_used_tokens_at_failure": 0,
        "hbm_alpha_at_failure": 0.0,
        "target_failed_before_source": episode.target_failed,
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def _bootstrap_ci(
    values: list[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _summarize(
    rows: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)
    stats_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_stats:
        stats_by_policy[row["policy"]].append(row)
    seed_harm: dict[str, dict[int, float]] = defaultdict(dict)
    for policy, policy_rows in by_policy.items():
        grouped = defaultdict(list)
        for row in policy_rows:
            grouped[int(row["seed"])].append(float(row["visible_harm_ms"]))
        seed_harm[policy] = {
            seed: sum(values) / len(values)
            for seed, values in grouped.items()
        }
    baseline = seed_harm["recompute_only"]
    static = seed_harm["static_independent_fcfs"]
    summaries = []
    for policy in POLICIES:
        policy_rows = by_policy[policy]
        recovery = [float(row["recovery_ms"]) for row in policy_rows]
        harm = [float(row["visible_harm_ms"]) for row in policy_rows]
        restored_fraction = [
            float(row["restored_tokens"]) / max(int(row["source_tokens"]), 1)
            for row in policy_rows
        ]
        paired_vs_recompute = [
            100.0 * (baseline[seed] - seed_harm[policy][seed]) / baseline[seed]
            for seed in sorted(baseline)
            if baseline[seed] > 0.0
        ]
        paired_vs_static = [
            100.0 * (static[seed] - seed_harm[policy][seed]) / static[seed]
            for seed in sorted(static)
            if static[seed] > 0.0
        ]
        ci_recompute = _bootstrap_ci(
            paired_vs_recompute,
            resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]),
        )
        ci_static = _bootstrap_ci(
            paired_vs_static,
            resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]) + 1,
        )
        normal = stats_by_policy[policy]
        observed = sum(int(row["observed_source_tokens"]) for row in normal)
        copy_tokens = sum(int(row["admitted_copy_tokens"]) for row in normal)
        actions = Counter(str(row["action"]) for row in policy_rows)
        summaries.append(
            {
                "policy": policy,
                "failures": len(policy_rows),
                "mean_recovery_ms": sum(recovery) / len(recovery),
                "p50_recovery_ms": _quantile(recovery, 0.50),
                "p95_recovery_ms": _quantile(recovery, 0.95),
                "mean_visible_harm_ms": sum(harm) / len(harm),
                "p95_visible_harm_ms": _quantile(harm, 0.95),
                "zero_visible_harm_fraction": sum(value == 0.0 for value in harm)
                / len(harm),
                "mean_restored_fraction": sum(restored_fraction)
                / len(restored_fraction),
                "visible_harm_reduction_vs_recompute_pct": (
                    sum(paired_vs_recompute) / len(paired_vs_recompute)
                ),
                "visible_harm_reduction_vs_recompute_ci95_pct": list(
                    ci_recompute
                ),
                "visible_harm_reduction_vs_static_pct": (
                    sum(paired_vs_static) / len(paired_vs_static)
                ),
                "visible_harm_reduction_vs_static_ci95_pct": list(ci_static),
                "copy_token_amplification": copy_tokens / max(observed, 1),
                "mean_modeled_copy_service_ms_per_seed": sum(
                    float(row["modeled_copy_service_ms"]) for row in normal
                )
                / len(normal),
                "mean_hbm_used_tokens_at_failure": sum(
                    int(row["hbm_used_tokens_at_failure"]) for row in normal
                )
                / len(normal),
                "mean_dram_used_tokens_at_failure": sum(
                    int(row["dram_used_tokens_at_failure"]) for row in normal
                )
                / len(normal),
                "action_counts": dict(sorted(actions.items())),
            }
        )
    return {
        "experiment_id": config["experiment_id"],
        "design_semantics": DESIGN_SEMANTICS,
        "evidence_class": EVIDENCE_CLASS,
        "policies": summaries,
    }


def _audit(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    keys_by_policy = defaultdict(set)
    violations = []
    for row in rows:
        keys_by_policy[row["policy"]].add(
            (int(row["seed"]), str(row["request_id"]))
        )
        if int(row["restored_tokens"]) > int(row["source_tokens"]):
            violations.append(
                f"restored_gt_source:{row['policy']}:{row['request_id']}"
            )
        if int(row["recompute_tokens"]) != int(row["source_tokens"]) - int(
            row["restored_tokens"]
        ):
            violations.append(
                f"partition_mismatch:{row['policy']}:{row['request_id']}"
            )
        if int(row["l1_tokens"]) and int(row["l2_tokens"]):
            violations.append(
                f"recovery_selected_tier_sum:{row['policy']}:{row['request_id']}"
            )
    reference = keys_by_policy["recompute_only"]
    matched = all(keys_by_policy[policy] == reference for policy in POLICIES)
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "passed": matched and not violations,
        "design_semantics": DESIGN_SEMANTICS,
        "evidence_class": EVIDENCE_CLASS,
        "matched_request_failure_keys": matched,
        "request_failure_keys_per_policy": len(reference),
        "policy_count": len(POLICIES),
        "row_count": len(rows),
        "restored_prefix_partition_violations": violations,
        "canonical_rows_sha256": hashlib.sha256(canonical).hexdigest(),
        "canonical_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "physical_runtime_claim": False,
        "paper_ready_claim": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    for seed in range(int(config["seeds"])):
        episode = _build_episode(config, seed)
        for runner in (
            lambda: _recompute_policy(episode, config),
            lambda: _static_policy(episode, config),
            lambda: _latest_policy(
                episode,
                config,
                policy="aegiskv_hbm_only",
                allowed_tiers=(ProtectionTier.HBM,),
            ),
            lambda: _latest_policy(
                episode,
                config,
                policy="aegiskv_dram_only",
                allowed_tiers=(ProtectionTier.DRAM,),
            ),
            lambda: _latest_policy(
                episode,
                config,
                policy="aegiskv_latest",
                allowed_tiers=(ProtectionTier.HBM, ProtectionTier.DRAM),
            ),
        ):
            policy_rows, policy_stats = runner()
            rows.extend(policy_rows)
            episode_stats.append(policy_stats)
    rows.sort(key=lambda row: (row["seed"], row["request_id"], row["policy"]))
    episode_stats.sort(key=lambda row: (row["seed"], row["policy"]))
    return rows, episode_stats


def main() -> None:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_dir / "config.json")
    parser.add_argument("--output-dir", type=Path, default=default_dir / "results")
    args = parser.parse_args()
    config = _load_config(args.config)
    rows, episode_stats = run(config)
    summary = _summarize(rows, episode_stats, config)
    audit = _audit(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_failures_path = args.output_dir / "raw_failures.csv"
    _write_csv(raw_failures_path, rows)
    _write_csv(args.output_dir / "episode_stats.csv", episode_stats)
    _write_csv(args.output_dir / "summary.csv", summary["policies"])
    audit.update(
        {
            "raw_failures_csv_sha256": _file_sha256(raw_failures_path),
            "config_file_sha256": _file_sha256(args.config),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))
    if not audit["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
