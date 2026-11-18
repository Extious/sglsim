from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import AgenticWorkflowScenario


@dataclass(frozen=True)
class FailureInjectionPlan:
    enabled: bool
    injection_mode: str
    injection_count: int
    inject_after_output_tokens: int
    target_stage_id: int | None
    target_stage_local_agent_index: int | None
    target_turn_idx: int | None
    retry_rank_offset: int
    retry_policy: str
    replay_critical_only: bool
    dp_rank_sweep: bool = False
    agent_slot_sweep: bool = False

    def to_row(self) -> dict[str, bool | int | str]:
        return asdict(self)

    @property
    def controlled_by_injection_count(self) -> bool:
        return (
            self.enabled
            and self.injection_count > 0
            and not self.dp_rank_sweep
            and not self.agent_slot_sweep
        )


def failure_plan_from_scenario(scenario: AgenticWorkflowScenario) -> FailureInjectionPlan:
    enabled = bool(scenario.failover_replay)
    dp_rank_sweep = bool(scenario.dp_failure_sweep)
    agent_slot_sweep = bool(scenario.agent_slot_failure_sweep)
    if dp_rank_sweep or agent_slot_sweep:
        injection_count = 0
    else:
        injection_count = int(
            scenario.failure_injection_count
            if scenario.failure_injection_count is not None
            else (1 if enabled else 0)
        )
    return FailureInjectionPlan(
        enabled=enabled,
        injection_mode=str(scenario.failover_injection if enabled else "none"),
        injection_count=injection_count,
        inject_after_output_tokens=int(scenario.failure_after_output_tokens or 0),
        target_stage_id=scenario.failure_target_stage_id,
        target_stage_local_agent_index=(
            scenario.failure_target_stage_local_agent_index
        ),
        target_turn_idx=scenario.failure_target_turn_idx,
        retry_rank_offset=int(scenario.failover_retry_rank_offset),
        retry_policy=str(scenario.failover_retry_policy or "peer"),
        replay_critical_only=bool(scenario.replay_critical_only),
        dp_rank_sweep=dp_rank_sweep,
        agent_slot_sweep=agent_slot_sweep,
    )


def validate_failure_plan(plan: FailureInjectionPlan) -> None:
    if not plan.enabled:
        return
    if plan.injection_mode not in {"soft", "hard", "hard-restart", "kill-restart"}:
        raise ValueError(f"unsupported failure injection mode: {plan.injection_mode!r}")
    if (
        not plan.dp_rank_sweep
        and not plan.agent_slot_sweep
        and plan.injection_count <= 0
    ):
        raise ValueError("enabled failure plan must inject at least one failure")
    if plan.inject_after_output_tokens <= 0:
        raise ValueError("failure injection must be pinned to output-token progress")
    if plan.retry_rank_offset <= 0:
        raise ValueError("retry_rank_offset must select a different DP rank")
