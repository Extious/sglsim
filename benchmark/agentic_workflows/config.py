from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent  # benchmark/agentic_workflows
REPO_ROOT = PACKAGE_DIR.parents[1]
AEGIS_LARGE_DIR = (
    REPO_ROOT / "benchmark" / "AegisKV" / "draft" / "experiments" / "sglsim_large"
)
DEFAULT_BASE_CONFIG_DIR = AEGIS_LARGE_DIR / "config_failover_replay"
DEFAULT_RESULTS_ROOT = AEGIS_LARGE_DIR / "results" / "agentic_workflows"
DEFAULT_ALPHA_POLICY_PATH = (
    AEGIS_LARGE_DIR / "policies" / "aegiskv_alpha_mdp_policy.json"
)


def _normalized_fraction(value: object) -> float | None:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction < 0.0:
        return None
    return max(0.0, min(fraction, 1.0))


@dataclass(frozen=True)
class AgenticWorkflowScenario:
    name: str
    description: str
    groups: int
    prompts_per_group: int
    turns: int
    max_concurrency: int
    system_prompt_len: int
    question_len: int
    output_len: int
    dp_size: int
    critical_agent_ratio: float
    failover_replay: bool
    failover_injection: str
    failover_retry_rank_offset: int
    seed: int
    start_port: int
    runs: int = 1
    workflow_shape: tuple[int, ...] = ()
    stage_context_lens: tuple[int, ...] = ()
    stage_long_decode_agent_index: int = 0
    long_decode_output_multiplier: float = 4.0
    range_ratio: float = 0.75
    ordered: bool = False
    fast_prepare: bool = False
    send_routing_key: bool = True
    replay_critical_only: bool = True
    disable_overlap_schedule: bool = False
    failover_retry_policy: str = ""
    aegis_policy: str = ""
    aegis_ablation: str = ""
    aegis_gpu_budget_fraction: float | None = None
    aegis_critical_pressure: float | None = None
    alpha_policy_path: str | None = None
    failure_rate_per_window: float | None = None
    aegis_online_alpha_adapter: bool = False
    failure_after_output_tokens: int | None = None
    failure_injection_count: int | None = None
    failure_target_stage_id: int | None = None
    failure_target_stage_local_agent_index: int | None = None
    failure_target_turn_idx: int | None = None
    backup_kv: bool | None = None
    l1_capacity_pages: int | None = None
    context_length: int | None = None
    max_workflow_concurrency: int | None = None
    dp_failure_sweep: bool = False
    dp_failure_sweep_hold_s: float = 25.0
    dp_failure_sweep_gap_s: float = 8.0
    dp_failure_sweep_initial_delay_s: float = 20.0
    dp_failure_sweep_injections_per_batch: int = 4
    dp_failure_sweep_batch_size: int = 32
    dp_failure_sweep_batch_gap_s: float = 0.0
    dp_failure_sweep_round2_enabled: bool = True
    dp_failure_sweep_round2_injections_per_batch: int = 2
    dp_failure_sweep_round2_batch_gap_s: float = 60.0
    dp_failure_sweep_round1_enabled: bool = True
    dp_failure_sweep_precovered_agent_slots: tuple[tuple[int, int], ...] = ()
    dp_failure_sweep_precovered_workflows: tuple[str, ...] = ()
    agent_slot_failure_sweep: bool = False
    agent_slot_sweep_hold_s: float = 18.0
    agent_slot_sweep_gap_s: float = 4.0
    agent_slot_sweep_initial_delay_s: float = 25.0
    agent_slot_sweep_min_injections: int = 2
    trace_jsonl_path: str | None = None

    @property
    def workflow_count(self) -> int:
        agents_per_workflow = (
            sum(self.workflow_shape)
            if self.workflow_shape
            else self.prompts_per_group
        )
        return self.groups * agents_per_workflow

    @property
    def estimated_first_turn_prompt_tokens(self) -> int:
        if self.stage_context_lens:
            return int(self.stage_context_lens[0])
        return int(self.system_prompt_len) + int(self.question_len)

    @property
    def estimated_final_turn_context_tokens(self) -> int:
        if self.stage_context_lens:
            return max(int(x) for x in self.stage_context_lens)
        return (
            int(self.system_prompt_len)
            + int(self.turns) * int(self.question_len)
            + max(int(self.turns) - 1, 0) * int(self.output_len)
        )

    @property
    def estimated_long_decode_output_tokens(self) -> int:
        return max(
            int(self.output_len) + 1,
            int(round(float(self.output_len) * float(self.long_decode_output_multiplier))),
        )

    def with_overrides(
        self,
        *,
        runs: int | None = None,
        seed: int | None = None,
        start_port: int | None = None,
    ) -> "AgenticWorkflowScenario":
        updates: dict[str, Any] = {}
        if runs is not None:
            updates["runs"] = int(runs)
        if seed is not None:
            updates["seed"] = int(seed)
        if start_port is not None:
            updates["start_port"] = int(start_port)
        return replace(self, **updates) if updates else self


SCENARIOS: dict[str, AgenticWorkflowScenario] = {
    "smoke": AgenticWorkflowScenario(
        name="smoke",
        description="Cheap two-turn generated-shared-prefix workflow without failover replay.",
        groups=1,
        prompts_per_group=2,
        turns=2,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=16,
        dp_size=4,
        critical_agent_ratio=0.5,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=101,
        start_port=30300,
    ),
    "physical_copy_restore_smoke": AgenticWorkflowScenario(
        name="physical_copy_restore_smoke",
        description="Cheap DP4 hard-restart replay for physical-copy-calibrated restore timing.",
        groups=1,
        prompts_per_group=2,
        turns=2,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=16,
        dp_size=4,
        critical_agent_ratio=1.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        failure_after_output_tokens=8,
        seed=8080,
        start_port=30820,
    ),
    "shared_prefix_dp2": AgenticWorkflowScenario(
        name="shared_prefix_dp2",
        description="DP2 shared-prefix agentic workload for normal-path cache and frontier measurements.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7000,
        start_port=30320,
        runs=3,
    ),
    "staged_13131_motivation": AgenticWorkflowScenario(
        name="staged_13131_motivation",
        description=(
            "1-3-1-3-1 staged workflow; singleton stages are Critical, and one "
            "agent in each 3-way fan-out has a long decode tail and is Critical."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8700,
        start_port=31420,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        failure_after_output_tokens=32,
        aegis_policy="aegiskv",
        alpha_policy_path=str(DEFAULT_ALPHA_POLICY_PATH),
        failure_rate_per_window=0.01,
        backup_kv=True,
    ),
    "staged_13131_alpha_online": AgenticWorkflowScenario(
        name="staged_13131_alpha_online",
        description=(
            "Small 1-3-1-3-1 staged workflow for online MDP alpha-decision "
            "evidence without failure replay."
        ),
        groups=3,
        prompts_per_group=9,
        turns=3,
        max_concurrency=9,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=8750,
        start_port=32420,
        runs=1,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        alpha_policy_path=str(DEFAULT_ALPHA_POLICY_PATH),
        failure_rate_per_window=0.01,
        backup_kv=True,
    ),
    "staged_13131_policy_no_backup": AgenticWorkflowScenario(
        name="staged_13131_policy_no_backup",
        description=(
            "1-3-1-3-1 staged workflow policy sweep point with Critical-agent "
            "backup disabled."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8900,
        start_port=33420,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="no_backup",
        backup_kv=False,
    ),
    "staged_13131_policy_host_only": AgenticWorkflowScenario(
        name="staged_13131_policy_host_only",
        description=(
            "1-3-1-3-1 staged workflow policy sweep point with HostBackup-only "
            "Critical-agent recovery frontier."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8900,
        start_port=33520,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="host_only",
        backup_kv=True,
    ),
    "staged_13131_policy_gpu_replica": AgenticWorkflowScenario(
        name="staged_13131_policy_gpu_replica",
        description=(
            "1-3-1-3-1 staged workflow policy sweep point with GPUReplica-only "
            "Critical-agent recovery frontier."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8900,
        start_port=33620,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="gpu_replica",
        backup_kv=True,
    ),
    "staged_13131_policy_aegiskv": AgenticWorkflowScenario(
        name="staged_13131_policy_aegiskv",
        description=(
            "1-3-1-3-1 staged workflow policy sweep point with AegisKV dynamic "
            "Host/GPUReplica recovery frontier."
        ),
        groups=8,
        prompts_per_group=9,
        turns=3,
        max_concurrency=24,
        system_prompt_len=512,
        question_len=192,
        output_len=48,
        dp_size=4,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8900,
        start_port=33720,
        runs=3,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=5.0,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        alpha_policy_path=str(DEFAULT_ALPHA_POLICY_PATH),
        failure_rate_per_window=0.01,
        backup_kv=True,
    ),
    "normal_overhead_baseline": AgenticWorkflowScenario(
        name="normal_overhead_baseline",
        description="DP2 no-failure baseline with critical/Aegis protection disabled.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=2,
        critical_agent_ratio=0.0,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7900,
        start_port=30720,
        runs=3,
    ),
    "normal_overhead_aegis": AgenticWorkflowScenario(
        name="normal_overhead_aegis",
        description="DP2 no-failure workload with AegisKV selective critical-agent protection enabled.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7900,
        start_port=30760,
        runs=3,
    ),
    "shared_prefix_sweep_low": AgenticWorkflowScenario(
        name="shared_prefix_sweep_low",
        description="Low-overlap shared-prefix sweep point with two turns and long per-turn questions.",
        groups=4,
        prompts_per_group=4,
        turns=2,
        max_concurrency=1,
        system_prompt_len=32,
        question_len=768,
        output_len=16,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7600,
        start_port=30600,
        runs=3,
    ),
    "shared_prefix_sweep_mid": AgenticWorkflowScenario(
        name="shared_prefix_sweep_mid",
        description="Mid-overlap shared-prefix sweep point matching the main three-turn workflow shape.",
        groups=4,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=32,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7700,
        start_port=30640,
        runs=3,
    ),
    "shared_prefix_sweep_high": AgenticWorkflowScenario(
        name="shared_prefix_sweep_high",
        description="High-overlap shared-prefix sweep point with five turns and a long shared system prefix.",
        groups=4,
        prompts_per_group=4,
        turns=5,
        max_concurrency=1,
        system_prompt_len=2048,
        question_len=64,
        output_len=16,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        seed=7800,
        start_port=30680,
        runs=3,
    ),
    "hard_failover_host": AgenticWorkflowScenario(
        name="hard_failover_host",
        description="DP2 hard-restart replay with HostBackup lookup on the retry path.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=2,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=7100,
        start_port=30360,
        runs=3,
    ),
    "policy_no_backup": AgenticWorkflowScenario(
        name="policy_no_backup",
        description="DP4 hard-restart policy sweep point with critical requests recomputing after failure.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8100,
        start_port=30860,
        runs=3,
        aegis_policy="no_backup",
        backup_kv=False,
    ),
    "policy_host_only": AgenticWorkflowScenario(
        name="policy_host_only",
        description="DP4 hard-restart policy sweep point with HostBackup-only recovery frontier.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8100,
        start_port=30900,
        runs=3,
        aegis_policy="host_only",
        backup_kv=True,
    ),
    "policy_gpu_replica": AgenticWorkflowScenario(
        name="policy_gpu_replica",
        description="DP4 hard-restart policy sweep point with GPUReplica-only recovery frontier.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8100,
        start_port=30940,
        runs=3,
        aegis_policy="gpu_replica",
        backup_kv=True,
    ),
    "policy_aegiskv": AgenticWorkflowScenario(
        name="policy_aegiskv",
        description="DP4 hard-restart policy sweep point with AegisKV dynamic Host/GPUReplica frontier.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8100,
        start_port=30980,
        runs=3,
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "hard_failover_peer": AgenticWorkflowScenario(
        name="hard_failover_peer",
        description="DP4 hard-restart replay where retry targets the adjacent peer rank.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=7200,
        start_port=30420,
        runs=3,
    ),
    "hard_failover_peer_hot_tail": AgenticWorkflowScenario(
        name="hard_failover_peer_hot_tail",
        description="DP4 hard-restart replay with short prefixes so peer GPUReplica hot-tail restore wins.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=7400,
        start_port=30540,
        runs=3,
    ),
    "hard_failover_nonpeer": AgenticWorkflowScenario(
        name="hard_failover_nonpeer",
        description="DP4 hard-restart replay where retry targets a non-peer rank.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=512,
        question_len=256,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=2,
        seed=7300,
        start_port=30480,
        runs=3,
    ),
    "peer_route_nonpeer_hot_tail": AgenticWorkflowScenario(
        name="peer_route_nonpeer_hot_tail",
        description="DP4 online peer-selection sweep point forcing a healthy non-peer retry route.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=2,
        seed=8500,
        start_port=31020,
        runs=3,
        failover_retry_policy="non_peer",
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "peer_route_random_hot_tail": AgenticWorkflowScenario(
        name="peer_route_random_hot_tail",
        description="DP4 online peer-selection sweep point routing retries to a deterministic random healthy rank.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8500,
        start_port=31060,
        runs=3,
        failover_retry_policy="random_healthy",
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "peer_route_aegiskv_hot_tail": AgenticWorkflowScenario(
        name="peer_route_aegiskv_hot_tail",
        description="DP4 online peer-selection sweep point routing retries to the recorded peer rank.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8500,
        start_port=31100,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        backup_kv=True,
    ),
    "ablation_full_aegiskv": AgenticWorkflowScenario(
        name="ablation_full_aegiskv",
        description="DP4 online Fig. 12 ablation baseline with all AegisKV recovery signals enabled.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31140,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_no_lifecycle": AgenticWorkflowScenario(
        name="ablation_no_lifecycle",
        description="DP4 online Fig. 12 ablation disabling lifecycle lookup across retries.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31180,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_lifecycle",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_no_segment_granularity": AgenticWorkflowScenario(
        name="ablation_no_segment_granularity",
        description="DP4 online Fig. 12 ablation requiring all-or-nothing restored frontiers.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31220,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_segment_granularity",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_no_restore_recompute": AgenticWorkflowScenario(
        name="ablation_no_restore_recompute",
        description="DP4 online Fig. 12 ablation using restore-first instead of restore/recompute lower envelope.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31260,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_restore_recompute",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_no_pressure": AgenticWorkflowScenario(
        name="ablation_no_pressure",
        description="DP4 online Fig. 12 ablation removing pressure-aware peer placement.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31300,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_pressure",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_no_hysteresis": AgenticWorkflowScenario(
        name="ablation_no_hysteresis",
        description="DP4 online Fig. 12 ablation removing stable peer-frontier hysteresis.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31340,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_hysteresis",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_no_critical_restore": AgenticWorkflowScenario(
        name="ablation_no_critical_restore",
        description="DP4 online Fig. 12 ablation disabling restore for critical agents.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31380,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_critical_restore",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_static_alpha": AgenticWorkflowScenario(
        name="ablation_static_alpha",
        description="DP4 online ablation using a fixed GPUReplica alpha instead of the online MDP selector.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31420,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="static_alpha",
        aegis_gpu_budget_fraction=0.2,
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_size_only": AgenticWorkflowScenario(
        name="ablation_size_only",
        description="DP4 online ablation that lets size-only backup place GPUReplica tokens for non-Critical requests.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31460,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="size_only_backup",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_no_peer_locality": AgenticWorkflowScenario(
        name="ablation_no_peer_locality",
        description="DP4 online ablation that keeps backup placement but disables peer-locality recovery hits.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31500,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_peer_locality",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
    "ablation_no_marginal_accounting": AgenticWorkflowScenario(
        name="ablation_no_marginal_accounting",
        description="DP4 online ablation that protects the full frontier instead of marginal hot-tail KV.",
        groups=8,
        prompts_per_group=4,
        turns=3,
        max_concurrency=1,
        system_prompt_len=64,
        question_len=32,
        output_len=64,
        dp_size=4,
        critical_agent_ratio=0.25,
        failover_replay=True,
        failover_injection="hard-restart",
        failover_retry_rank_offset=1,
        seed=8600,
        start_port=31540,
        runs=3,
        failover_retry_policy="peer",
        aegis_policy="aegiskv",
        aegis_ablation="no_marginal_accounting",
        failure_after_output_tokens=64,
        backup_kv=True,
    ),
}


def _register_staged_policy_lite_scenarios() -> None:
    common: dict[str, Any] = {
        "groups": 3,
        "prompts_per_group": 9,
        "turns": 3,
        "max_concurrency": 9,
        "system_prompt_len": 512,
        "question_len": 192,
        "output_len": 48,
        "dp_size": 4,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "hard-restart",
        "failover_retry_rank_offset": 1,
        "seed": 8950,
        "runs": 3,
        "workflow_shape": (1, 3, 1, 3, 1),
        "stage_long_decode_agent_index": 0,
        "long_decode_output_multiplier": 5.0,
        "failover_retry_policy": "peer",
        "failure_after_output_tokens": 64,
    }
    policy_specs = (
        ("no_backup", "backup disabled", "no_backup", False, 34420),
        ("host_only", "HostBackup-only recovery frontier", "host_only", True, 34520),
        (
            "gpu_replica",
            "GPUReplica-only recovery frontier",
            "gpu_replica",
            True,
            34620,
        ),
        ("aegiskv", "AegisKV dynamic Host/GPUReplica frontier", "aegiskv", True, 34720),
    )
    for suffix, label, policy, backup_kv, start_port in policy_specs:
        name = f"staged_13131_policy_lite_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "Lite 1-3-1-3-1 staged workflow policy sweep point with "
                f"Critical-agent {label}."
            ),
            start_port=start_port,
            aegis_policy=policy,
            backup_kv=backup_kv,
            alpha_policy_path=(
                str(DEFAULT_ALPHA_POLICY_PATH) if policy == "aegiskv" else None
            ),
            failure_rate_per_window=0.01 if policy == "aegiskv" else None,
            **common,
        )


def _register_staged_policy_stress_scenarios() -> None:
    common: dict[str, Any] = {
        "groups": 3,
        "prompts_per_group": 9,
        "turns": 3,
        "max_concurrency": 9,
        "system_prompt_len": 512,
        "question_len": 192,
        "output_len": 128,
        "dp_size": 4,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "soft",
        "failover_retry_rank_offset": 1,
        "seed": 9050,
        "runs": 3,
        "workflow_shape": (1, 3, 1, 3, 1),
        "stage_long_decode_agent_index": 0,
        "long_decode_output_multiplier": 8.0,
        "failover_retry_policy": "peer",
        "failure_after_output_tokens": 128,
    }
    policy_specs = (
        ("no_backup", "backup disabled", "no_backup", False, 35420),
        ("host_only", "HostBackup-only recovery frontier", "host_only", True, 35520),
        (
            "gpu_replica",
            "GPUReplica-only recovery frontier",
            "gpu_replica",
            True,
            35620,
        ),
        ("aegiskv", "AegisKV dynamic Host/GPUReplica frontier", "aegiskv", True, 35720),
    )
    for suffix, label, policy, backup_kv, start_port in policy_specs:
        name = f"staged_13131_policy_stress_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "Stress 1-3-1-3-1 staged workflow policy sweep point with "
                f"soft failure injection and Critical-agent {label}."
            ),
            start_port=start_port,
            aegis_policy=policy,
            backup_kv=backup_kv,
            alpha_policy_path=(
                str(DEFAULT_ALPHA_POLICY_PATH) if policy == "aegiskv" else None
            ),
            failure_rate_per_window=0.01 if policy == "aegiskv" else None,
            **common,
        )


def _register_staged_100k_context_scenarios() -> None:
    common: dict[str, Any] = {
        "groups": 2,
        "prompts_per_group": 9,
        "turns": 3,
        "max_concurrency": 6,
        "system_prompt_len": 98304,
        "question_len": 1024,
        "output_len": 128,
        "dp_size": 8,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "soft",
        "failover_retry_rank_offset": 1,
        "seed": 9150,
        "runs": 1,
        "workflow_shape": (1, 3, 1, 3, 1),
        "stage_context_lens": (40960, 40960, 81920, 40960, 122880),
        "stage_long_decode_agent_index": 0,
        "long_decode_output_multiplier": 8.0,
        "failover_retry_policy": "peer",
        "failure_after_output_tokens": 128,
        "l1_capacity_pages": 786432,
        "context_length": 196608,
    }
    policy_specs = (
        ("no_backup", "backup disabled", "no_backup", False, 36420),
        ("host_only", "HostBackup-only recovery frontier", "host_only", True, 36520),
        (
            "gpu_replica",
            "GPUReplica-only recovery frontier",
            "gpu_replica",
            True,
            36620,
        ),
        ("aegiskv", "AegisKV dynamic Host/GPUReplica frontier", "aegiskv", True, 36720),
    )
    for suffix, label, policy, backup_kv, start_port in policy_specs:
        name = f"staged_13131_100k_context_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "100K-context 1-3-1-3-1 staged workflow policy point with "
                f"Critical-agent {label}; singleton stages and one long-decode "
                "fan-out agent per width-3 stage are Critical."
            ),
            start_port=start_port,
            aegis_policy=policy,
            backup_kv=backup_kv,
            aegis_gpu_budget_fraction=0.3 if policy == "gpu_replica" else None,
            alpha_policy_path=(
                str(DEFAULT_ALPHA_POLICY_PATH) if policy == "aegiskv" else None
            ),
            failure_rate_per_window=0.01 if policy == "aegiskv" else None,
            **common,
        )


def _register_motivation_main_subagent_scenarios() -> None:
    workflow_groups = 24
    workflow_concurrency = 24
    common = dict(
        groups=workflow_groups,
        prompts_per_group=9,
        turns=3,
        max_concurrency=workflow_concurrency,
        max_workflow_concurrency=workflow_concurrency,
        system_prompt_len=98304,
        question_len=768,
        output_len=128,
        dp_size=8,
        critical_agent_ratio=0.0,
        replay_critical_only=False,
        failover_retry_rank_offset=1,
        seed=9251,
        runs=1,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_context_lens=(40960, 40960, 81920, 40960, 122880),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=8.0,
        failover_retry_policy="peer",
        failure_after_output_tokens=128,
        backup_kv=False,
        l1_capacity_pages=393216,
        context_length=196608,
        agent_slot_sweep_initial_delay_s=25.0,
        agent_slot_sweep_hold_s=18.0,
        agent_slot_sweep_min_injections=1,
    )
    specs = (
        (
            "motivation_main_subagent_dp8_no_failure",
            (
                "DP8 online 100K-context motivation baseline with 24 concurrent "
                "workflows and no failure."
            ),
            False,
            "none",
            0,
            None,
            None,
            None,
            False,
            False,
            37410,
        ),
        (
            "motivation_main_subagent_dp8_agent_sweep",
            (
                "DP8 online 100K-context motivation run with rotating "
                "stage/agent failure sweep under workflow concurrency."
            ),
            True,
            "soft",
            0,
            None,
            None,
            None,
            False,
            True,
            37415,
        ),
        (
            "motivation_main_subagent_dp8_failure",
            (
                "DP8 online 100K-context motivation run with one final main-agent "
                "failure per workflow under high concurrency."
            ),
            True,
            "soft",
            workflow_groups,
            4,
            0,
            0,
            False,
            False,
            37420,
        ),
        (
            "motivation_main_subagent_dp8_subagent_failure",
            (
                "DP8 online 100K-context motivation run with S1-A0 Critical "
                "subagent failure per workflow under high concurrency."
            ),
            True,
            "soft",
            workflow_groups,
            1,
            0,
            0,
            False,
            False,
            37430,
        ),
        (
            "motivation_main_subagent_dp8_noncritical_subagent_failure",
            (
                "DP8 online 100K-context motivation run with S1-A1 Non-Critical "
                "subagent failure per workflow under high concurrency."
            ),
            True,
            "soft",
            workflow_groups,
            1,
            1,
            0,
            False,
            False,
            37440,
        ),
        (
            "motivation_main_subagent_dp8_s1_a2_failure",
            (
                "DP8 online 100K-context motivation run with S1-A2 Non-Critical "
                "subagent failure per workflow under high concurrency."
            ),
            True,
            "soft",
            workflow_groups,
            1,
            2,
            0,
            False,
            False,
            37450,
        ),
        (
            "motivation_main_subagent_dp8_s3_a0_failure",
            (
                "DP8 online 100K-context motivation run with S3-A0 Critical "
                "subagent failure per workflow under high concurrency."
            ),
            True,
            "soft",
            workflow_groups,
            3,
            0,
            0,
            False,
            False,
            37460,
        ),
        (
            "motivation_main_subagent_dp8_s3_a1_failure",
            (
                "DP8 online 100K-context motivation run with S3-A1 Non-Critical "
                "subagent failure per workflow under high concurrency."
            ),
            True,
            "soft",
            workflow_groups,
            3,
            1,
            0,
            False,
            False,
            37470,
        ),
        (
            "motivation_main_subagent_dp8_s3_a2_failure",
            (
                "DP8 online 100K-context motivation run with S3-A2 Non-Critical "
                "subagent failure per workflow under high concurrency."
            ),
            True,
            "soft",
            workflow_groups,
            3,
            2,
            0,
            False,
            False,
            37480,
        ),
        (
            "motivation_main_subagent_dp8_profile",
            (
                "DP8 online 100K-context profile alias with 24 concurrent workflows, "
                "used to measure Critical/Non-Critical request classes without failure."
            ),
            False,
            "none",
            0,
            None,
            None,
            None,
            False,
            False,
            37490,
        ),
    )
    for (
        name,
        description,
        failover_replay,
        failover_injection,
        injection_count,
        target_stage_id,
        target_local_agent_index,
        target_turn_idx,
        dp_failure_sweep,
        agent_slot_failure_sweep,
        start_port,
    ) in specs:
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=description,
            failover_replay=failover_replay,
            failover_injection=failover_injection,
            failure_injection_count=injection_count,
            failure_target_stage_id=target_stage_id,
            failure_target_stage_local_agent_index=target_local_agent_index,
            failure_target_turn_idx=target_turn_idx,
            dp_failure_sweep=bool(dp_failure_sweep),
            agent_slot_failure_sweep=bool(agent_slot_failure_sweep),
            start_port=start_port,
            **common,
        )


def _register_motivation_dp_sweep_scenarios() -> None:
    workflow_groups = 64
    workflow_concurrency = 32
    common = dict(
        groups=workflow_groups,
        prompts_per_group=9,
        turns=3,
        max_concurrency=workflow_concurrency,
        max_workflow_concurrency=workflow_concurrency,
        system_prompt_len=98304,
        question_len=768,
        output_len=128,
        dp_size=8,
        critical_agent_ratio=0.0,
        replay_critical_only=False,
        failover_retry_rank_offset=1,
        seed=9251,
        runs=1,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_context_lens=(40960, 40960, 81920, 40960, 122880),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=8.0,
        failover_retry_policy="peer",
        failure_after_output_tokens=128,
        backup_kv=False,
        l1_capacity_pages=393216,
        context_length=196608,
        dp_failure_sweep_hold_s=25.0,
        dp_failure_sweep_gap_s=8.0,
        dp_failure_sweep_initial_delay_s=20.0,
        dp_failure_sweep_injections_per_batch=4,
        dp_failure_sweep_batch_size=32,
        dp_failure_sweep_batch_gap_s=0.0,
        dp_failure_sweep_round2_enabled=True,
        dp_failure_sweep_round2_injections_per_batch=2,
        dp_failure_sweep_round2_batch_gap_s=60.0,
    )
    specs = (
        (
            "motivation_main_subagent_dp8_dp_sweep_baseline",
            (
                "DP8 online 100K-context motivation baseline with 64 workflows, "
                "32 concurrent admission (two batches), and no failure."
            ),
            False,
            "none",
            0,
            None,
            None,
            None,
            False,
            37510,
        ),
        (
            "motivation_main_subagent_dp8_dp_sweep",
            (
                "DP8 online 100K-context motivation run with two-batch DP rank "
                "sweep: 8 soft failures (4 per batch) across DP0-7."
            ),
            True,
            "soft",
            0,
            None,
            None,
            None,
            True,
            37515,
        ),
    )
    for (
        name,
        description,
        failover_replay,
        failover_injection,
        injection_count,
        target_stage_id,
        target_local_agent_index,
        target_turn_idx,
        dp_failure_sweep,
        start_port,
    ) in specs:
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=description,
            failover_replay=failover_replay,
            failover_injection=failover_injection,
            failure_injection_count=injection_count,
            failure_target_stage_id=target_stage_id,
            failure_target_stage_local_agent_index=target_local_agent_index,
            failure_target_turn_idx=target_turn_idx,
            dp_failure_sweep=bool(dp_failure_sweep),
            start_port=start_port,
            **common,
        )

    SCENARIOS["motivation_main_subagent_dp8_dp_sweep_round2"] = AgenticWorkflowScenario(
        name="motivation_main_subagent_dp8_dp_sweep_round2",
        description=(
            "DP8 two-batch round2-only DP rank sweep for uncovered stage/agent "
            "slots after round1."
        ),
        failover_replay=True,
        failover_injection="soft",
        failure_injection_count=0,
        dp_failure_sweep=True,
        dp_failure_sweep_round1_enabled=False,
        dp_failure_sweep_precovered_agent_slots=(
            (0, 0),
            (1, 0),
            (1, 2),
            (2, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
        ),
        dp_failure_sweep_precovered_workflows=(
            "wf-4",
            "wf-7",
            "wf-10",
            "wf-12",
            "wf-43",
            "wf-46",
            "wf-55",
            "wf-59",
        ),
        start_port=37520,
        **{**common, "dp_failure_sweep_round2_injections_per_batch": 4},
    )


def _register_alpha_tradeoff_dp8_scenarios() -> None:
    alphas = (
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    )
    regimes = (
        ("baseload_fail1", "base load / one injected Critical-agent failure", 2, 6, 1, 37510),
        ("highload_fail1", "high load / one injected Critical-agent failure", 4, 12, 1, 37650),
        ("baseload_fail3", "base load / three injected Critical-agent failures", 2, 6, 3, 37790),
        ("highload_fail3", "high load / three injected Critical-agent failures", 4, 12, 3, 37930),
    )
    common = dict(
        prompts_per_group=9,
        turns=3,
        system_prompt_len=98304,
        question_len=768,
        output_len=128,
        dp_size=8,
        critical_agent_ratio=0.0,
        failover_replay=True,
        failover_injection="soft",
        failover_retry_rank_offset=1,
        replay_critical_only=True,
        seed=9350,
        runs=1,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_context_lens=(40960, 40960, 81920, 40960, 122880),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=8.0,
        failover_retry_policy="peer",
        failure_after_output_tokens=128,
        aegis_policy="gpu_replica",
        backup_kv=True,
        l1_capacity_pages=786432,
        context_length=196608,
    )
    for regime, label, groups, max_concurrency, injection_count, base_port in regimes:
        for idx, alpha in enumerate(alphas):
            suffix = f"a{int(round(alpha * 100)):03d}"
            name = f"alpha_tradeoff_dp8_{regime}_{suffix}"
            SCENARIOS[name] = AgenticWorkflowScenario(
                name=name,
                description=(
                    "DP8 online 100K-context alpha tradeoff point: "
                    f"{label}, alpha={alpha:.2f}."
                ),
                groups=groups,
                max_concurrency=max_concurrency,
                failure_injection_count=injection_count,
                aegis_gpu_budget_fraction=alpha,
                start_port=base_port + idx * 10,
                **common,
            )


def _register_online_alpha_dp8_scenarios() -> None:
    common = dict(
        prompts_per_group=9,
        turns=3,
        system_prompt_len=98304,
        question_len=768,
        output_len=128,
        dp_size=8,
        critical_agent_ratio=0.0,
        failover_replay=False,
        failover_injection="none",
        failover_retry_rank_offset=1,
        replay_critical_only=True,
        seed=9450,
        runs=1,
        workflow_shape=(1, 3, 1, 3, 1),
        stage_context_lens=(40960, 40960, 81920, 40960, 122880),
        stage_long_decode_agent_index=0,
        long_decode_output_multiplier=8.0,
        failover_retry_policy="peer",
        failure_after_output_tokens=128,
        aegis_policy="aegiskv",
        alpha_policy_path=str(DEFAULT_ALPHA_POLICY_PATH),
        backup_kv=True,
        l1_capacity_pages=786432,
        context_length=196608,
    )
    specs = (
        (
            "online_alpha_dp8_baseload_lowrisk",
            "base load online MDP alpha run with low failure rate",
            2,
            6,
            0.62,
            0.003,
            38110,
        ),
        (
            "online_alpha_dp8_baseload_highrisk",
            "base load online MDP alpha run with high failure rate",
            2,
            6,
            0.62,
            0.03,
            38130,
        ),
        (
            "online_alpha_dp8_highload_lowrisk",
            "high load online MDP alpha run with low failure rate",
            4,
            12,
            0.86,
            0.003,
            38150,
        ),
        (
            "online_alpha_dp8_highload_highrisk",
            "high load online MDP alpha run with high failure rate",
            4,
            12,
            0.86,
            0.03,
            38170,
        ),
    )
    for name, label, groups, max_concurrency, pressure, failure_rate, start_port in specs:
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "DP8 online 100K-context 1-3-1-3-1 workflow for "
                f"{label}; alpha is selected by the offline MDP policy table."
            ),
            groups=groups,
            max_concurrency=max_concurrency,
            aegis_critical_pressure=pressure,
            failure_rate_per_window=failure_rate,
            start_port=start_port,
            **common,
        )


def _budget_suffix(value: float) -> str:
    return f"b{int(round(value * 100)):03d}"


def _register_budget_sweep_scenarios() -> None:
    budgets = (0.0, 0.1, 0.2, 0.5, 1.0)
    common = {
        "groups": 8,
        "prompts_per_group": 4,
        "turns": 3,
        "max_concurrency": 1,
        "output_len": 64,
        "dp_size": 4,
        "critical_agent_ratio": 0.25,
        "failover_replay": True,
        "failover_injection": "hard-restart",
        "failover_retry_rank_offset": 1,
        "seed": 8700,
        "runs": 3,
        "failover_retry_policy": "peer",
        "aegis_policy": "aegiskv",
        "backup_kv": True,
    }
    workload_specs = (
        ("budget_hot_tail", "hot-tail", 64, 32, 31420, "peer", 1),
        ("budget_long_prefix", "long-prefix", 512, 256, 31620, "peer", 1),
        ("budget_nonpeer", "non-peer fan-out", 64, 32, 31820, "non_peer", 2),
    )
    for prefix, label, system_len, question_len, base_port, retry_policy, retry_offset in workload_specs:
        for index, budget in enumerate(budgets):
            name = f"{prefix}_{_budget_suffix(budget)}"
            SCENARIOS[name] = AgenticWorkflowScenario(
                name=name,
                description=(
                    "DP4 online Fig. 3 GPUReplica budget sweep for "
                    f"{label} workload at budget={budget:.2f}."
                ),
                system_prompt_len=system_len,
                question_len=question_len,
                start_port=base_port + index * 10,
                failover_retry_policy=retry_policy,
                failover_retry_rank_offset=retry_offset,
                aegis_gpu_budget_fraction=budget,
                **{
                    key: value
                    for key, value in common.items()
                    if key not in {"failover_retry_policy", "failover_retry_rank_offset"}
                },
            )

    timing_specs = (
        ("budget_timing_early", "early decode", 1, 32020),
        ("budget_timing_late", "late decode", 64, 32220),
    )
    for prefix, label, failure_after_output, base_port in timing_specs:
        for index, budget in enumerate(budgets):
            name = f"{prefix}_{_budget_suffix(budget)}"
            SCENARIOS[name] = AgenticWorkflowScenario(
                name=name,
                description=(
                    "DP4 online Fig. 13 failure-timing budget sweep for "
                    f"{label} at budget={budget:.2f}."
                ),
                system_prompt_len=64,
                question_len=32,
                start_port=base_port + index * 10,
                aegis_gpu_budget_fraction=budget,
                failure_after_output_tokens=failure_after_output,
                **common,
            )


def _register_policy_matrix_dp8_24w_fail3_scenarios() -> None:
    workflow_groups = 24
    workflow_concurrency = 24
    failure_target_stage_id = 1
    failure_target_stage_local_agent_index = 0
    common: dict[str, Any] = {
        "groups": workflow_groups,
        "prompts_per_group": 9,
        "turns": 3,
        "max_concurrency": workflow_concurrency,
        "max_workflow_concurrency": workflow_concurrency,
        "system_prompt_len": 98304,
        "question_len": 768,
        "output_len": 128,
        "dp_size": 8,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "soft",
        "failover_retry_rank_offset": 1,
        "replay_critical_only": True,
        "seed": 9300,
        "runs": 1,
        "ordered": True,
        "workflow_shape": (1, 3, 1, 3, 1),
        "stage_context_lens": (40960, 40960, 81920, 40960, 122880),
        "stage_long_decode_agent_index": 0,
        "long_decode_output_multiplier": 8.0,
        "failover_retry_policy": "peer",
        "failure_after_output_tokens": 128,
        "failure_injection_count": 3,
        "failure_target_stage_id": failure_target_stage_id,
        "failure_target_stage_local_agent_index": failure_target_stage_local_agent_index,
        "l1_capacity_pages": 786432,
        "context_length": 196608,
    }
    policy_specs = (
        ("no_backup", "backup disabled", "no_backup", False, None, 38010),
        ("host_only", "HostBackup-only recovery", "host_only", True, None, 38020),
        (
            "gpu_replica",
            "GPUReplica-only recovery with alpha=0.30",
            "gpu_replica",
            True,
            0.3,
            38030,
        ),
        (
            "aegiskv",
            "AegisKV with fixed alpha=0.30",
            "aegiskv",
            True,
            0.3,
            38040,
        ),
    )
    for suffix, label, policy, backup_kv, alpha, start_port in policy_specs:
        name = f"policy_matrix_dp8_24w_fail3_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "DP8 online 100K-context policy matrix point with 24 concurrent "
                "workflows, three pinned S1-A0 soft failures, and "
                f"{label}."
            ),
            start_port=start_port,
            aegis_policy=policy,
            backup_kv=backup_kv,
            aegis_gpu_budget_fraction=alpha,
            **common,
        )


def _register_aegiskv_alpha_policy_cdf_dp8_48w_fail8_scenarios() -> None:
    workflow_groups = 48
    workflow_concurrency = 24
    common: dict[str, Any] = {
        "groups": workflow_groups,
        "prompts_per_group": 9,
        "turns": 3,
        "max_concurrency": workflow_concurrency,
        "max_workflow_concurrency": workflow_concurrency,
        "system_prompt_len": 98304,
        "question_len": 768,
        "output_len": 128,
        "dp_size": 8,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "soft",
        "failover_retry_rank_offset": 1,
        "replay_critical_only": True,
        "seed": 9400,
        "runs": 1,
        "workflow_shape": (1, 3, 1, 3, 1),
        "stage_context_lens": (40960, 40960, 81920, 40960, 122880),
        "stage_long_decode_agent_index": 0,
        "long_decode_output_multiplier": 8.0,
        "failover_retry_policy": "peer",
        "failure_after_output_tokens": 128,
        "failure_injection_count": 8,
        "aegis_policy": "aegiskv",
        "backup_kv": True,
        "l1_capacity_pages": 786432,
        "context_length": 196608,
    }
    policy_specs = (
        (
            "fix_a010",
            "AegisKV fixed alpha=0.10",
            0.1,
            None,
            None,
            False,
            39210,
        ),
        (
            "fix_a030",
            "AegisKV fixed alpha=0.30",
            0.3,
            None,
            None,
            False,
            39220,
        ),
        (
            "fix_a050",
            "AegisKV fixed alpha=0.50",
            0.5,
            None,
            None,
            False,
            39230,
        ),
        (
            "mdp",
            "AegisKV with MDP and online alpha adapter",
            None,
            str(DEFAULT_ALPHA_POLICY_PATH),
            0.01,
            True,
            39240,
        ),
    )
    for (
        suffix,
        label,
        alpha,
        policy_path,
        failure_rate,
        online_alpha_adapter,
        start_port,
    ) in policy_specs:
        name = f"aegiskv_alpha_policy_cdf_dp8_48w_fail8_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "DP8 online 100K-context AegisKV alpha-policy CDF point with "
                f"48 workflows, concurrency={workflow_concurrency}, eight injected "
                f"soft failures, and {label}."
            ),
            start_port=start_port,
            aegis_gpu_budget_fraction=alpha,
            alpha_policy_path=policy_path,
            failure_rate_per_window=failure_rate,
            aegis_online_alpha_adapter=online_alpha_adapter,
            **common,
        )


def _register_weka_subagent_policy_cdf_dp8_24w_fail6_scenarios() -> None:
    trace_path = (
        AEGIS_LARGE_DIR
        / "traces"
        / "weka_subagent_cdf_24w_workload.jsonl"
    )
    workflow_concurrency = 24
    common: dict[str, Any] = {
        "groups": 1,
        "prompts_per_group": 1,
        "turns": 1,
        "max_concurrency": workflow_concurrency,
        "max_workflow_concurrency": workflow_concurrency,
        "system_prompt_len": 0,
        "question_len": 0,
        "output_len": 64,
        "dp_size": 8,
        "critical_agent_ratio": 0.0,
        "failover_replay": True,
        "failover_injection": "soft",
        "failover_retry_rank_offset": 1,
        "replay_critical_only": True,
        "seed": 9500,
        "runs": 1,
        "failover_retry_policy": "peer",
        "failure_after_output_tokens": 128,
        "failure_injection_count": 6,
        "l1_capacity_pages": 1572864,
        "context_length": 393216,
        "trace_jsonl_path": str(trace_path),
        "ordered": True,
        "fast_prepare": True,
        "send_routing_key": False,
    }
    policy_specs = (
        ("no_backup", "No backup", "no_backup", False, None, 39100),
        (
            "aegiskv",
            "AegisKV with fixed alpha=0.30",
            "aegiskv",
            True,
            0.3,
            39110,
        ),
    )
    for suffix, label, policy, backup_kv, alpha, start_port in policy_specs:
        name = f"weka_subagent_policy_cdf_dp8_24w_fail6_{suffix}"
        SCENARIOS[name] = AgenticWorkflowScenario(
            name=name,
            description=(
                "DP8 online Weka subagent-trace policy CDF point with 24 real "
                f"Claude Code workflows, concurrency={workflow_concurrency}, "
                f"six injected soft failures, and {label}."
            ),
            start_port=start_port,
            aegis_policy=policy,
            backup_kv=backup_kv,
            aegis_gpu_budget_fraction=alpha,
            **common,
        )


_register_staged_policy_lite_scenarios()
_register_staged_policy_stress_scenarios()
_register_staged_100k_context_scenarios()
_register_policy_matrix_dp8_24w_fail3_scenarios()
_register_aegiskv_alpha_policy_cdf_dp8_48w_fail8_scenarios()
_register_weka_subagent_policy_cdf_dp8_24w_fail6_scenarios()
_register_motivation_main_subagent_scenarios()
_register_motivation_dp_sweep_scenarios()
_register_alpha_tradeoff_dp8_scenarios()
_register_online_alpha_dp8_scenarios()
_register_budget_sweep_scenarios()


def scenario_names() -> list[str]:
    return sorted(SCENARIOS)


def get_scenario(name: str) -> AgenticWorkflowScenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        known = ", ".join(scenario_names())
        raise ValueError(f"Unknown scenario {name!r}; known scenarios: {known}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def materialize_synthesis_config(
    scenario: AgenticWorkflowScenario,
    *,
    base_config_dir: Path = DEFAULT_BASE_CONFIG_DIR,
    output_dir: Path,
    trial: int = 0,
) -> Path:
    client = _load_json(Path(base_config_dir) / "client.json")
    server = _load_json(Path(base_config_dir) / "server.json")
    normalized_alpha = _normalized_fraction(scenario.aegis_gpu_budget_fraction)

    dataset_name = (
        "claude-code-trace-jsonl"
        if scenario.trace_jsonl_path
        else "generated-shared-prefix"
    )
    client.update(
        {
            "dataset_name": dataset_name,
            "seed": int(scenario.seed) + int(trial),
            "request_rate": "inf",
            "max_concurrency": (
                None
                if scenario.max_workflow_concurrency is not None
                else int(scenario.max_concurrency)
            ),
            "gsp_max_workflow_concurrency": (
                int(scenario.max_workflow_concurrency)
                if scenario.max_workflow_concurrency is not None
                else None
            ),
            "gsp_num_groups": int(scenario.groups),
            "gsp_prompts_per_group": int(scenario.prompts_per_group),
            "gsp_system_prompt_len": int(scenario.system_prompt_len),
            "gsp_question_len": int(scenario.question_len),
            "gsp_output_len": int(scenario.output_len),
            "gsp_range_ratio": float(scenario.range_ratio),
            "gsp_fast_prepare": bool(scenario.fast_prepare),
            "gsp_send_routing_key": bool(scenario.send_routing_key),
            "gsp_num_turns": int(scenario.turns),
            "gsp_ordered": bool(scenario.ordered),
            "gsp_critical_agent_ratio": float(scenario.critical_agent_ratio),
            "gsp_failover_replay": bool(scenario.failover_replay),
            "gsp_failover_replay_critical_only": bool(scenario.replay_critical_only),
            "gsp_failover_injection": str(scenario.failover_injection),
            "gsp_failover_retry_rank_offset": int(
                scenario.failover_retry_rank_offset
            ),
            "gsp_failover_retry_policy": str(scenario.failover_retry_policy),
            "gsp_aegis_policy": str(scenario.aegis_policy),
            "gsp_aegis_ablation": str(scenario.aegis_ablation),
        }
    )
    if scenario.trace_jsonl_path:
        client["trace_jsonl_path"] = str(scenario.trace_jsonl_path)
        client["gsp_num_groups"] = 1
        client["gsp_prompts_per_group"] = 1
        client["gsp_ordered"] = True
        client["gsp_fast_prepare"] = True
        client["gsp_send_routing_key"] = False
    if scenario.workflow_shape:
        client["gsp_workflow_shape"] = list(scenario.workflow_shape)
        if scenario.stage_context_lens:
            client["gsp_stage_context_lens"] = list(scenario.stage_context_lens)
        client["gsp_stage_long_decode_agent_index"] = int(
            scenario.stage_long_decode_agent_index
        )
        client["gsp_long_decode_output_multiplier"] = float(
            scenario.long_decode_output_multiplier
        )
    if scenario.aegis_gpu_budget_fraction is not None:
        client["gsp_aegis_gpu_budget_fraction"] = float(
            scenario.aegis_gpu_budget_fraction
        )
    if scenario.aegis_critical_pressure is not None:
        client["gsp_aegis_critical_pressure"] = float(
            scenario.aegis_critical_pressure
        )
    if scenario.alpha_policy_path:
        client["gsp_alpha_policy_path"] = str(scenario.alpha_policy_path)
    if scenario.aegis_online_alpha_adapter:
        client["gsp_aegis_online_alpha_adapter"] = True
    if scenario.failure_rate_per_window is not None:
        client["gsp_failure_rate_per_window"] = float(
            scenario.failure_rate_per_window
        )
    if scenario.failure_after_output_tokens is not None:
        client["gsp_failure_after_output_tokens"] = int(
            scenario.failure_after_output_tokens
        )
    if scenario.failure_injection_count is not None:
        client["gsp_failure_injection_count"] = int(
            scenario.failure_injection_count
        )
    if scenario.failure_target_stage_id is not None:
        client["gsp_failure_target_stage_id"] = int(
            scenario.failure_target_stage_id
        )
    if scenario.failure_target_stage_local_agent_index is not None:
        client["gsp_failure_target_stage_local_agent_index"] = int(
            scenario.failure_target_stage_local_agent_index
        )
    if scenario.failure_target_turn_idx is not None:
        client["gsp_failure_target_turn_idx"] = int(
            scenario.failure_target_turn_idx
        )
    if scenario.backup_kv is not None:
        client["gsp_backup_kv"] = bool(scenario.backup_kv)
    if scenario.dp_failure_sweep:
        client["gsp_dp_failure_sweep"] = True
        client["gsp_dp_failure_sweep_hold_s"] = float(scenario.dp_failure_sweep_hold_s)
        client["gsp_dp_failure_sweep_gap_s"] = float(scenario.dp_failure_sweep_gap_s)
        client["gsp_dp_failure_sweep_initial_delay_s"] = float(
            scenario.dp_failure_sweep_initial_delay_s
        )
        client["gsp_dp_failure_sweep_injections_per_batch"] = int(
            scenario.dp_failure_sweep_injections_per_batch
        )
        client["gsp_dp_failure_sweep_batch_size"] = int(
            scenario.dp_failure_sweep_batch_size
        )
        client["gsp_dp_failure_sweep_batch_gap_s"] = float(
            scenario.dp_failure_sweep_batch_gap_s
        )
        client["gsp_dp_failure_sweep_round2_enabled"] = bool(
            scenario.dp_failure_sweep_round2_enabled
        )
        client["gsp_dp_failure_sweep_round2_injections_per_batch"] = int(
            scenario.dp_failure_sweep_round2_injections_per_batch
        )
        client["gsp_dp_failure_sweep_round2_batch_gap_s"] = float(
            scenario.dp_failure_sweep_round2_batch_gap_s
        )
        client["gsp_dp_failure_sweep_round1_enabled"] = bool(
            scenario.dp_failure_sweep_round1_enabled
        )
        if scenario.dp_failure_sweep_precovered_agent_slots:
            client["gsp_dp_failure_sweep_precovered_agent_slots"] = [
                [int(stage_id), int(local_idx)]
                for stage_id, local_idx in scenario.dp_failure_sweep_precovered_agent_slots
            ]
        if scenario.dp_failure_sweep_precovered_workflows:
            client["gsp_dp_failure_sweep_precovered_workflows"] = list(
                scenario.dp_failure_sweep_precovered_workflows
            )
        client["gsp_failure_injection_count"] = None
    if scenario.agent_slot_failure_sweep:
        client["gsp_agent_slot_failure_sweep"] = True
        client["gsp_agent_slot_sweep_hold_s"] = float(scenario.agent_slot_sweep_hold_s)
        client["gsp_agent_slot_sweep_gap_s"] = float(scenario.agent_slot_sweep_gap_s)
        client["gsp_agent_slot_sweep_min_injections"] = int(
            scenario.agent_slot_sweep_min_injections
        )
        client["gsp_failure_injection_count"] = None

    scheduler = dict(server.get("scheduler") or {})
    scheduler["dp_size"] = int(scenario.dp_size)
    scheduler["enable_routing_key_affinity"] = bool(scenario.send_routing_key)
    scheduler["disable_overlap_schedule"] = bool(scenario.disable_overlap_schedule)
    server["scheduler"] = scheduler
    if scenario.l1_capacity_pages is not None:
        cache = dict(server.get("cache") or {})
        l1_capacity_pages = int(scenario.l1_capacity_pages)
        alpha = (
            normalized_alpha
            if (
                normalized_alpha is not None
                and scenario.aegis_policy in {"gpu_replica", "gpureplica"}
            )
            else None
        )
        if alpha is None:
            cache["l1_capacity_pages"] = l1_capacity_pages
        else:
            cache["aegis_total_l1_capacity_pages"] = l1_capacity_pages
            cache["l1_capacity_pages"] = max(
                1,
                int(round(l1_capacity_pages * (1.0 - alpha))),
            )
        server["cache"] = cache
    if scenario.context_length is not None:
        server["context_length"] = int(scenario.context_length)
    server["port"] = int(scenario.start_port) + int(trial)
    server["experiment_name"] = f"agentic-{scenario.name}-t{int(trial):03d}"

    config_dir = Path(output_dir) / "configs" / scenario.name / f"trial_{int(trial):03d}"
    _write_json(config_dir / "client.json", client)
    _write_json(config_dir / "server.json", server)
    return config_dir


def scenario_plan_row(
    scenario: AgenticWorkflowScenario,
    *,
    config_dir: Path,
    trial: int,
) -> dict[str, Any]:
    return {
        "scenario": scenario.name,
        "trial": int(trial),
        "config_dir": str(config_dir),
        "seed": int(scenario.seed) + int(trial),
        "port": int(scenario.start_port) + int(trial),
        "dp_size": int(scenario.dp_size),
        "workflow_count": int(scenario.groups),
        "total_agentic_requests": int(scenario.workflow_count),
        "agents_per_workflow": int(scenario.prompts_per_group),
        "workflow_shape": list(scenario.workflow_shape),
        "stage_context_lens": list(scenario.stage_context_lens),
        "turns": int(scenario.turns),
        "system_prompt_len": int(scenario.system_prompt_len),
        "question_len": int(scenario.question_len),
        "output_len": int(scenario.output_len),
        "estimated_first_turn_prompt_tokens": int(
            scenario.estimated_first_turn_prompt_tokens
        ),
        "estimated_final_turn_context_tokens": int(
            scenario.estimated_final_turn_context_tokens
        ),
        "estimated_long_decode_output_tokens": int(
            scenario.estimated_long_decode_output_tokens
        ),
        "max_concurrency": int(scenario.max_concurrency),
        "max_workflow_concurrency": (
            int(scenario.max_workflow_concurrency)
            if scenario.max_workflow_concurrency is not None
            else None
        ),
        "dp_failure_sweep": bool(scenario.dp_failure_sweep),
        "dp_failure_sweep_hold_s": float(scenario.dp_failure_sweep_hold_s),
        "dp_failure_sweep_gap_s": float(scenario.dp_failure_sweep_gap_s),
        "dp_failure_sweep_initial_delay_s": float(
            scenario.dp_failure_sweep_initial_delay_s
        ),
        "dp_failure_sweep_injections_per_batch": int(
            scenario.dp_failure_sweep_injections_per_batch
        ),
        "dp_failure_sweep_batch_size": int(scenario.dp_failure_sweep_batch_size),
        "dp_failure_sweep_batch_gap_s": float(scenario.dp_failure_sweep_batch_gap_s),
        "dp_failure_sweep_round2_enabled": bool(scenario.dp_failure_sweep_round2_enabled),
        "dp_failure_sweep_round2_injections_per_batch": int(
            scenario.dp_failure_sweep_round2_injections_per_batch
        ),
        "dp_failure_sweep_round2_batch_gap_s": float(
            scenario.dp_failure_sweep_round2_batch_gap_s
        ),
        "agent_slot_failure_sweep": bool(scenario.agent_slot_failure_sweep),
        "agent_slot_sweep_hold_s": float(scenario.agent_slot_sweep_hold_s),
        "agent_slot_sweep_gap_s": float(scenario.agent_slot_sweep_gap_s),
        "agent_slot_sweep_initial_delay_s": float(
            scenario.agent_slot_sweep_initial_delay_s
        ),
        "failover_replay": bool(scenario.failover_replay),
        "failover_injection": scenario.failover_injection,
        "failover_retry_rank_offset": int(scenario.failover_retry_rank_offset),
        "failover_retry_policy": str(scenario.failover_retry_policy),
        "disable_overlap_schedule": bool(scenario.disable_overlap_schedule),
        "aegis_policy": str(scenario.aegis_policy),
        "aegis_ablation": str(scenario.aegis_ablation),
        "aegis_gpu_budget_fraction": scenario.aegis_gpu_budget_fraction,
        "aegis_critical_pressure": scenario.aegis_critical_pressure,
        "alpha_policy_path": scenario.alpha_policy_path,
        "aegis_online_alpha_adapter": bool(scenario.aegis_online_alpha_adapter),
        "failure_rate_per_window": scenario.failure_rate_per_window,
        "failure_after_output_tokens": scenario.failure_after_output_tokens,
        "failure_injection_count": scenario.failure_injection_count,
        "failure_target_stage_id": scenario.failure_target_stage_id,
        "failure_target_stage_local_agent_index": (
            scenario.failure_target_stage_local_agent_index
        ),
        "failure_target_turn_idx": scenario.failure_target_turn_idx,
        "backup_kv": scenario.backup_kv,
        "l1_capacity_pages": scenario.l1_capacity_pages,
        "context_length": scenario.context_length,
    }
