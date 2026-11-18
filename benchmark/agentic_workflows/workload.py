from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .config import AgenticWorkflowScenario


AgentRole = Literal["main_agent", "subagent"]


@dataclass(frozen=True)
class WorkflowShapeSummary:
    workflow_shape: str
    agents_per_workflow: int
    workflow_count: int
    total_agentic_requests: int
    main_agents_per_workflow: int
    subagents_per_workflow: int
    estimated_first_turn_prompt_tokens: int
    estimated_final_turn_context_tokens: int
    estimated_long_decode_output_tokens: int

    def to_row(self) -> dict[str, int | str]:
        return asdict(self)


def agent_role_for_stage(*, stage_width: int) -> AgentRole:
    return "main_agent" if int(stage_width) == 1 else "subagent"


def summarize_workflow_shape(scenario: AgenticWorkflowScenario) -> WorkflowShapeSummary:
    shape = tuple(int(x) for x in scenario.workflow_shape if int(x) > 0)
    agents_per_workflow = sum(shape) if shape else int(scenario.prompts_per_group)
    main_agents = sum(1 for width in shape if width == 1) if shape else 0
    subagents = agents_per_workflow - main_agents if shape else agents_per_workflow
    return WorkflowShapeSummary(
        workflow_shape="-".join(str(x) for x in shape) if shape else "flat",
        agents_per_workflow=agents_per_workflow,
        workflow_count=int(scenario.groups),
        total_agentic_requests=int(scenario.groups) * agents_per_workflow,
        main_agents_per_workflow=main_agents,
        subagents_per_workflow=subagents,
        estimated_first_turn_prompt_tokens=scenario.estimated_first_turn_prompt_tokens,
        estimated_final_turn_context_tokens=scenario.estimated_final_turn_context_tokens,
        estimated_long_decode_output_tokens=scenario.estimated_long_decode_output_tokens,
    )
